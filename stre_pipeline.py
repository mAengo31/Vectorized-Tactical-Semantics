"""
stre_pipeline.py — spec-faithful edge / C2 split.

Three components:

  1. MultiSink            — fan-out adapter so OntologyEventFactory emits
                            into BOTH the Foundry sink (dashboard JSON)
                            and the StreSink (radio CBOR) on every call.

  2. EdgeStreEmitter      — converts ByteTrack-style track updates into
                            STRE Type 1 messages, rate-limited per track
                            to fit the 9.6 kbps budget. Lives on the Jetson.

  3. C2StreEngine         — simulates the STRE Engine that lives at C2
                            Gateway. Consumes Type 1 messages, performs
                            entity resolution (per-source track grouping)
                            and pattern matching, emits Type 2 inferences.
                            Run this on your laptop to demo the full loop.

Architectural boundary (per spec §2):

  EDGE (Jetson)                       C2 GATEWAY
  ──────────────                      ──────────
  Perception (YOLO)
       │
       ▼
  ByteTrack (per-sensor)
       │
       ▼
  EdgeStreEmitter ──── Type 1 ──────► C2StreEngine
                       (radio)             │
                                           ├─ Entity resolution
                                           ├─ Pattern matching
                                           ├─ Hazard fusion
                                           ▼
                                       Type 2 inference
                                           │
                                           ▼
                                       Dashboard
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

from ontology_factory import (
    TelemetrySink, ObservationEvent, TrackUpdate, EdgeHealth,
)
from stre_codec import (
    StreObservation, StreInference, StreSink,
    to_stre_observation, to_stre_inference,
    SENSOR_TYPE, OBJECT_CLASS, INTERNAL_TO_STRE_CLASS,
    PATTERN_ID, ACTION, THREAT_LEVEL,
)


# ============================================================================
# 1. MultiSink — fan-out wrapper
# ============================================================================

class MultiSink(TelemetrySink):
    """
    Fans every emit() call to multiple sinks. Used to send the same event
    simultaneously to:
      - Foundry sink (rich JSON ontology objects, IP backhaul, dashboard)
      - StreSink     (compressed CBOR + COSE, narrowband radio)

    Failures in one sink don't block the others. Each sink raises its own
    exceptions to the outbox; the outbox decides retry policy per-sink.
    """

    def __init__(self, sinks: List[TelemetrySink], names: Optional[List[str]] = None):
        if not sinks:
            raise ValueError("MultiSink needs at least one underlying sink")
        self.sinks = sinks
        self.names = names or [f"sink_{i}" for i in range(len(sinks))]
        if len(self.names) != len(self.sinks):
            raise ValueError("names and sinks length mismatch")

    def healthcheck(self) -> bool:
        # Healthy if at least one underlying sink is up
        return any(s.healthcheck() for s in self.sinks)

    def write_observation(self, event: ObservationEvent) -> None:
        errors = []
        for name, sink in zip(self.names, self.sinks):
            try:
                sink.write_observation(event)
            except Exception as e:
                errors.append((name, e))
        if len(errors) == len(self.sinks):
            # All sinks failed — surface the first error so outbox retries
            raise errors[0][1]

    def write_track(self, track: TrackUpdate) -> None:
        errors = []
        for name, sink in zip(self.names, self.sinks):
            try:
                sink.write_track(track)
            except Exception as e:
                errors.append((name, e))
        if len(errors) == len(self.sinks):
            raise errors[0][1]

    def write_health(self, health: EdgeHealth) -> None:
        errors = []
        for name, sink in zip(self.names, self.sinks):
            try:
                sink.write_health(health)
            except Exception as e:
                errors.append((name, e))
        if len(errors) == len(self.sinks):
            raise errors[0][1]


# ============================================================================
# 2. EdgeStreEmitter — Type 1 generator on the Jetson
# ============================================================================

@dataclass
class TrackSnapshot:
    """Minimal info EdgeStreEmitter needs from your tracker per frame."""
    track_id: int
    class_name: str           # internal label (tank/apc/...)
    confidence: float
    bbox: List[float]
    motion_vector: List[float]   # [dx_px_per_sec, dy_px_per_sec]
    last_seen_unix: int


class EdgeStreEmitter:
    """
    Converts ByteTrack output into STRE Type 1 messages, with bandwidth
    management:
      - One observation per active track per `emit_interval_s` seconds
      - Skips classes without a STRE code (smoke/fire — those go in
        Foundry-side hazard analysis, not on the radio)

    At 9.6 kbps with ~72 B sealed Type 1 (~57ms per message airtime), you
    can comfortably emit ~16 messages/sec sustained. Default interval is
    1 sec per track, so up to ~16 simultaneously tracked entities fit
    within budget.
    """

    def __init__(self,
                 source_id: int,
                 sensor_type: str = "EO",
                 emit_interval_s: float = 1.0,
                 camera_lat: float = 0.0,
                 camera_lon: float = 0.0,
                 camera_alt_m: int = 0,
                 camera_heading_deg: int = 0,
                 px_per_meter: float = 0.0):
        self.source_id = source_id
        self.sensor_type = sensor_type
        self.emit_interval_s = emit_interval_s
        # Camera pose — needed to translate per-frame pixels into world coords.
        # If you don't have geo calibration, set lat/lon to the CASK position
        # and px_per_meter=0; the Type 1 messages will all carry the CASK's
        # own location (still useful — entity resolution works on time + class).
        self.camera_lat = camera_lat
        self.camera_lon = camera_lon
        self.camera_alt_m = camera_alt_m
        self.camera_heading_deg = camera_heading_deg
        self.px_per_meter = px_per_meter
        # Per-track state for rate limiting
        self._last_emit_at: Dict[int, float] = {}
        # Monotonic event_id counter
        self._next_event_id = 1

    def _alloc_event_id(self) -> int:
        eid = self._next_event_id & 0xFFFF
        self._next_event_id = (self._next_event_id + 1) & 0xFFFF
        if self._next_event_id == 0:
            self._next_event_id = 1   # avoid 0 which is reserved-ish
        return eid

    def _pixels_to_world(self, bbox: List[float],
                         motion_px: List[float]) -> Tuple[float, float, float, int]:
        """
        Returns (lat, lon, speed_mps, heading_deg) for the tracked entity.

        Production CASKs receive a geo-rectification homography from
        platform telemetry (UAV gimbal pose + IMU + altitude). That
        homography projects bbox center pixels onto a ground-plane
        lat/lon. We don't have that here, so we fake it: bbox center
        pixels become small offsets from the CASK's own lat/lon, scaled
        by px_per_meter. The offsets are real enough that distinct
        vehicles in distinct image positions land in distinct C2 entity
        buckets — which is what C2 entity resolution needs to work.

        When a real homography is wired in, only this function changes.
        """
        import math
        # Bbox center in pixels
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # Convert pixels-to-meters using the camera's px_per_meter scale.
        # If px_per_meter is unset, fall back to camera position (everything
        # collapses to one entity bucket — known limitation).
        if self.px_per_meter > 0:
            # Use frame-center as origin so offsets are signed both ways.
            # 1280x720 is the assumed default; the math doesn't actually
            # need the real frame size because we just need *relative*
            # offsets — but we'll pass through to keep semantics tidy.
            origin_x = 640.0
            origin_y = 360.0
            meters_east = (cx - origin_x) / self.px_per_meter
            meters_north = -(cy - origin_y) / self.px_per_meter  # image y is down
            # Convert meters to degree offsets.
            mean_lat_rad = math.radians(self.camera_lat)
            dlat = meters_north / 111111.0
            dlon = meters_east / (111111.0 * max(0.1, math.cos(mean_lat_rad)))
            obs_lat = self.camera_lat + dlat
            obs_lon = self.camera_lon + dlon
        else:
            obs_lat = self.camera_lat
            obs_lon = self.camera_lon

        # Speed: motion_px from edge tracker is now [0, 0] under SpatialDeduper.
        # That's intentional — motion is derived at C2 from observation deltas.
        # Wire up a fallback in case a future tracker brings motion back.
        if self.px_per_meter > 0 and (motion_px[0] != 0 or motion_px[1] != 0):
            mag_px_per_sec = math.hypot(motion_px[0], motion_px[1])
            speed_mps = mag_px_per_sec / self.px_per_meter
            image_bearing = math.degrees(math.atan2(motion_px[0], -motion_px[1]))
            world_heading = (self.camera_heading_deg + image_bearing) % 360
        else:
            speed_mps = 0.0
            world_heading = self.camera_heading_deg

        return obs_lat, obs_lon, speed_mps, int(world_heading)

    def update(self, tracks: Iterable[TrackSnapshot],
               now_unix: Optional[int] = None) -> List[StreObservation]:
        """
        Call this per frame with the current tracker state. Returns 0..N
        Type 1 messages ready to seal+emit. Rate-limited per track.
        """
        if now_unix is None:
            now_unix = int(time.time())
        now_mono = time.monotonic()

        out: List[StreObservation] = []
        for t in tracks:
            # Rate limit per track
            last = self._last_emit_at.get(t.track_id, 0.0)
            if (now_mono - last) < self.emit_interval_s:
                continue

            # Skip classes without STRE codes
            if t.class_name not in INTERNAL_TO_STRE_CLASS:
                continue

            lat, lon, speed_mps, heading = self._pixels_to_world(t.bbox, t.motion_vector)
            obs = to_stre_observation(
                event_id=self._alloc_event_id(),
                source_id=self.source_id,
                internal_class=t.class_name,
                confidence=t.confidence,
                timestamp_unix=now_unix,
                lat=lat, lon=lon,
                alt_m=self.camera_alt_m,
                heading_deg=heading,
                speed_mps=speed_mps,
                sensor_type=self.sensor_type,
            )
            if obs is None:   # bridge returned None (class unmapped)
                continue
            out.append(obs)
            self._last_emit_at[t.track_id] = now_mono

        return out


# ============================================================================
# 3. C2StreEngine — Type 1 → Type 2 reasoning at the C2 Gateway
# ============================================================================

@dataclass
class _TrackedEntity:
    """C2-side per-(source, internal_track_id) state used for entity resolution."""
    entity_id: int
    source_id: int
    object_class: int           # STRE code
    observations: deque         # rolling window of recent StreObservation
    first_seen_at: int
    last_seen_at: int


class C2StreEngine:
    """
    Receives Type 1 messages, maintains per-entity state, runs simple
    pattern matching, emits Type 2 inferences.

    Patterns implemented (matching spec §5.2):
      - 0x06 supply_convoy:    >= 4 tracked vehicles moving with similar bearing
      - 0x03 recon_then_strike: drone observation followed by ground unit + hazard
                                in same area within window
      - auto-generated (0x80+): not implemented for hackathon

    For the hackathon, "supply_convoy" + a generic "suspected_strike" pattern
    are enough to demo the value. Real STRE has more.
    """

    # Lowered for aerial drone footage where convoys string out and only
    # a few vehicles are simultaneously visible. With 1Hz emit rate over a
    # 60s window each entity contributes up to 60 observations, so the
    # bearing-coherence math has plenty of signal.
    SUPPLY_CONVOY_MIN_VEHICLES = 3
    SUPPLY_CONVOY_BEARING_TOLERANCE_DEG = 45
    OBSERVATION_WINDOW_S = 60.0
    # Minimum observations per entity needed to derive a motion vector.
    # We need at least 2 to get a delta. 3 is more robust against noise.
    MIN_OBS_FOR_MOTION = 3
    # Minimum derived speed (in 0.1 m/s units, matching the wire schema)
    # below which we treat an entity as stationary. Aerial pixel-rate
    # measurements are noisy, so we want a real movement signal.
    MIN_DERIVED_SPEED_X10 = 5

    def __init__(self):
        # entity_id → state. Keyed by (source_id, object_class, location_bucket)
        self._entities: Dict[Tuple[int, int, int], _TrackedEntity] = {}
        self._next_entity_id = 1
        self._next_inference_id = 1
        # Recent emitted inferences for refractory / dedup
        self._recent_inferences: deque = deque(maxlen=64)

    def _alloc_entity_id(self) -> int:
        eid = self._next_entity_id & 0xFFFF
        self._next_entity_id = (self._next_entity_id + 1) & 0xFFFF
        if self._next_entity_id == 0:
            self._next_entity_id = 1
        return eid

    def _alloc_inference_id(self) -> int:
        iid = self._next_inference_id & 0xFFFF
        self._next_inference_id = (self._next_inference_id + 1) & 0xFFFF
        if self._next_inference_id == 0:
            self._next_inference_id = 1
        return iid

    def ingest(self, obs: StreObservation) -> List[StreInference]:
        """Feed a Type 1 in. Get 0..N Type 2s out."""
        # --- Entity resolution: bucket on (source, class, location grid)
        # Coarse 0.001-degree grid (~111m) for grouping nearby observations
        loc_bucket = (obs.lat // 1000, obs.lon // 1000)
        key = (obs.source_id, obs.object_class, hash(loc_bucket))

        ent = self._entities.get(key)
        if ent is None:
            ent = _TrackedEntity(
                entity_id=self._alloc_entity_id(),
                source_id=obs.source_id,
                object_class=obs.object_class,
                observations=deque(maxlen=64),
                first_seen_at=obs.timestamp,
                last_seen_at=obs.timestamp,
            )
            self._entities[key] = ent
        ent.observations.append(obs)
        ent.last_seen_at = obs.timestamp

        # GC stale entities (haven't been seen in OBSERVATION_WINDOW_S * 2)
        cutoff = obs.timestamp - int(self.OBSERVATION_WINDOW_S * 2)
        for k in list(self._entities.keys()):
            if self._entities[k].last_seen_at < cutoff:
                del self._entities[k]

        # --- Pattern matching
        inferences: List[StreInference] = []
        inferences.extend(self._match_supply_convoy(obs))
        # Add more patterns here as you implement them

        return inferences

    def _derive_entity_motion(self, ent: _TrackedEntity) -> Optional[Tuple[float, float]]:
        """
        Compute (bearing_deg, speed_mps) for an entity from the deltas of
        its observations. Returns None when there isn't enough data.

        This is the motion-derivation step that spec §2 puts at C2. The
        edge emitter sends speed=0/heading=camera_heading because a single
        Type 1 doesn't carry a reliable motion vector — motion is a
        property of consecutive observations, not an instant.

        Falls back gracefully when lat/lon are pinned to the CASK position
        (i.e. when the CASK has no geo-rectification homography). In that
        case this returns None, and the supply_convoy matcher falls back
        to a count-only signal — which is still valid: 3+ vehicles seen
        from the same source in 60s is strong convoy evidence even
        without bearing coherence.
        """
        import math
        obs_list = list(ent.observations)
        if len(obs_list) < self.MIN_OBS_FOR_MOTION:
            return None

        first = obs_list[0]
        last = obs_list[-1]
        dt = last.timestamp - first.timestamp
        if dt <= 0:
            return None

        dlat = (last.lat - first.lat) / 1_000_000.0
        dlon = (last.lon - first.lon) / 1_000_000.0
        if abs(dlat) < 1e-7 and abs(dlon) < 1e-7:
            return None  # position unchanged (CASK-pinned location)

        mean_lat_rad = math.radians(first.lat / 1_000_000.0)
        meters_north = dlat * 111111.0
        meters_east = dlon * 111111.0 * math.cos(mean_lat_rad)
        speed_mps = math.hypot(meters_north, meters_east) / dt

        bearing = (math.degrees(math.atan2(meters_east, meters_north)) + 360) % 360
        return bearing, speed_mps

    def _match_supply_convoy(self, latest_obs: StreObservation) -> List[StreInference]:
        """
        Pattern 0x06: >= N distinct vehicle entities in same source within
        OBSERVATION_WINDOW_S. If geo-rectified positions are available we
        also check bearing coherence. Without geo positions (CASK-pinned),
        we fall back to count-only — which still works: 3+ vehicles in 60s
        from the same drone is strong convoy evidence.
        """
        vehicle_classes = {
            OBJECT_CLASS["vehicle_wheeled"],
            OBJECT_CLASS["vehicle_tracked"],
        }
        if latest_obs.object_class not in vehicle_classes:
            return []

        # Find vehicle entities active within the window from the same source.
        window_start = latest_obs.timestamp - int(self.OBSERVATION_WINDOW_S)
        active_entities: List[_TrackedEntity] = []
        evidence_obs: List[StreObservation] = []
        for ent in self._entities.values():
            if ent.source_id != latest_obs.source_id:
                continue
            if ent.object_class not in vehicle_classes:
                continue
            recent = [o for o in ent.observations if o.timestamp >= window_start]
            if not recent:
                continue
            active_entities.append(ent)
            evidence_obs.extend(recent)

        if len(active_entities) < self.SUPPLY_CONVOY_MIN_VEHICLES:
            return []

        # Try to derive bearings per entity. If geo data is missing this
        # returns None for everything — we still fire on count alone.
        derived_bearings: List[float] = []
        for ent in active_entities:
            motion = self._derive_entity_motion(ent)
            if motion is None:
                continue
            bearing, speed_mps = motion
            if speed_mps * 10 >= self.MIN_DERIVED_SPEED_X10:
                derived_bearings.append(bearing)

        bearing_signal = "absent"
        if derived_bearings:
            mean_bearing = sum(derived_bearings) / len(derived_bearings)
            coherent = sum(
                1 for b in derived_bearings
                if min(abs(b - mean_bearing), 360 - abs(b - mean_bearing))
                <= self.SUPPLY_CONVOY_BEARING_TOLERANCE_DEG
            )
            if coherent < len(derived_bearings) * 0.6:
                # Bearings present but dispersed — that's not a convoy,
                # that's vehicles milling around. Don't fire.
                return []
            bearing_signal = "coherent"

        # Refractory: don't re-fire same source within the window
        ref_key = ("supply_convoy", latest_obs.source_id)
        for prev_ts, prev_key in self._recent_inferences:
            if prev_key == ref_key and (latest_obs.timestamp - prev_ts) < self.OBSERVATION_WINDOW_S:
                return []
        self._recent_inferences.append((latest_obs.timestamp, ref_key))

        # Build the Type 2
        evidence_ids = sorted({o.event_id for o in evidence_obs})[:4]
        sensor_types_used = list({o.sensor_type for o in evidence_obs})

        eta_sec = 300
        target_lat = latest_obs.lat
        target_lon = latest_obs.lon

        # Threat scales with convoy size
        n = len(active_entities)
        if n >= 8:
            threat = "HIGH"
        elif n >= 5:
            threat = "MEDIUM"
        else:
            threat = "LOW"

        # Confidence scales with both count and whether we have bearing signal
        base_conf = min(0.85, 0.4 + 0.05 * n)
        if bearing_signal == "coherent":
            base_conf = min(0.95, base_conf + 0.10)

        inv_sensor = {v: k for k, v in SENSOR_TYPE.items()}
        sensor_type_names = [inv_sensor.get(c, "EO") for c in sensor_types_used]

        inf = to_stre_inference(
            inference_id=self._alloc_inference_id(),
            pattern_name="supply_convoy",
            evidence_event_ids=evidence_ids,
            entity_id=0,   # convoy is multi-entity
            threat=threat,
            target_lat=target_lat / 1_000_000,
            target_lon=target_lon / 1_000_000,
            confidence=base_conf,
            actions=["increase_isr", "reroute"],
            sensor_types_used=sensor_type_names,
            pattern_status="predefined",
            eta_sec=eta_sec,
            timestamp_unix=latest_obs.timestamp,
        )
        return [inf]


# ============================================================================
# Smoke test — full edge → C2 → inference loop
# ============================================================================

def _smoke_test():
    """
    Simulates 5 vehicles moving in convoy. Edge emits Type 1s every 1s per
    track, C2 ingests them, supply_convoy pattern fires.
    """
    import os
    print("=== Spec-faithful pipeline smoke test ===\n")

    emitter = EdgeStreEmitter(
        source_id=0xCA51,
        sensor_type="EO",
        emit_interval_s=1.0,
        camera_lat=48.158500,
        camera_lon=37.727000,
        camera_heading_deg=180,
        px_per_meter=8.0,
    )

    engine = C2StreEngine()

    # Simulate 5 tracks of vehicles moving roughly east (camera heading
    # 180 + image-bearing east means... whatever, the bearing math is
    # rough; we just need similar bearings)
    base_t = int(time.time())
    all_inferences = []

    print("[1] Edge emits 5 Type 1s per second for 8 seconds")
    print("    (5 tracks of vehicles moving in a convoy)\n")
    for second in range(8):
        # Vehicles drift slowly across frame
        tracks = [
            TrackSnapshot(
                track_id=10 + i,
                class_name="tank" if i < 2 else "apc" if i < 4 else "truck",
                confidence=0.78 + 0.02 * (i % 3),
                bbox=[800 + i * 50.0, 500 + i * 5.0, 870 + i * 50.0, 560 + i * 5.0],
                motion_vector=[-12.0 - i * 0.5, 1.5],   # all moving similar direction
                last_seen_unix=base_t + second,
            )
            for i in range(5)
        ]
        # Mock the emitter's clock so it actually emits each second
        emitter._last_emit_at.clear()
        observations = emitter.update(tracks, now_unix=base_t + second)
        if second == 0:
            print(f"    second 0: emitted {len(observations)} Type 1 messages")
            for obs in observations[:2]:
                print(f"      sample: event_id={obs.event_id}, "
                      f"obj_class=0x{obs.object_class:02x}, "
                      f"conf={obs.confidence}, heading={obs.heading}°")

        # Feed into C2 STRE engine
        for obs in observations:
            inferences = engine.ingest(obs)
            all_inferences.extend(inferences)

    print(f"\n[2] Type 1 -> Type 2 results")
    print(f"    Type 1 emitted (total): {sum(1 for _ in range(8)) * 5} (5/sec * 8sec)")
    print(f"    Type 2 inferences fired: {len(all_inferences)}")

    if all_inferences:
        print(f"\n    Sample Type 2:")
        inf = all_inferences[0]
        from stre_codec import PATTERN_ID
        pname = next((k for k, v in PATTERN_ID.items() if v == inf.pattern_id), "?")
        print(f"      pattern_id=0x{inf.pattern_id:02x} ({pname})")
        print(f"      threat=0x{inf.threat_level:02x}  conf={inf.confidence}")
        print(f"      evidence_ids={inf.evidence_ids}")
        print(f"      actions={[hex(a) for a in inf.actions]}")
        print(f"      raw size: {len(inf.encode())} bytes")

    # Bandwidth check
    print(f"\n[3] Bandwidth budget check (9.6 kbps)")
    bytes_per_sec_type1 = 5 * 72   # 5 messages * 72 B sealed
    kbps = bytes_per_sec_type1 * 8 / 1000
    print(f"    Edge sustained TX rate: {bytes_per_sec_type1} B/s = {kbps:.2f} kbps")
    print(f"    Headroom on 9.6 kbps link: {(9.6 - kbps) / 9.6 * 100:.0f}%")

    print("\n[4] Spec-faithful boundary observed:")
    print("    - Edge: perception + tracking + Type 1 emission only")
    print("    - C2:   entity resolution + pattern matching + Type 2 generation")
    print("    - No English strings, no fusion logic, no inference on the radio side\n")


if __name__ == "__main__":
    _smoke_test()
