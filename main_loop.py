"""
main_loop.py — full CASK demo orchestrator.

Wires every component together end-to-end:

    video file ──► EnsembleDetector (perception)
                          │
                          ▼
                   SimpleIouTracker (stubs ByteTrack — swap tomorrow)
                          │
                          ▼
                   EdgeStreEmitter (Type 1 generation, rate-limited)
                          │
                          ├──► StreSink (sealed CBOR to wire_log.bin)
                          │
                          └──► C2StreEngine (entity resolution + patterns)
                                      │
                                      ▼
                                Type 2 inferences
                                      │
                                      └──► OntologyEventFactory.emit_observation
                                              │
                                              └──► JsonlSink (Foundry/dashboard)

Output:
    {output_dir}/annotated/frame_NNNN.jpg     — viz frames you can scroll
    {output_dir}/wire_log.bin                  — sealed CBOR over-the-wire bytes
    {output_dir}/foundry_events.jsonl          — what the dashboard sees
    {output_dir}/run_summary.md                — final stats

Usage:
    python main_loop.py path/to/footage.mp4
    python main_loop.py footage.mp4 --output ./demo_run --skip 3 --max-seconds 30

Run on Mac for development (CPU/MPS), Jetson for the actual demo.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

from ensemble_detector import EnsembleDetector, Detection
from ontology_factory import OntologyEventFactory, JsonlSink, now_iso
from stre_codec import (
    AntiReplayState, CoseSealer, StreSink, StreObservation, StreInference,
    OBJECT_CLASS, PATTERN_ID, ACTION, THREAT_LEVEL, SENSOR_TYPE,
)
from stre_pipeline import (
    EdgeStreEmitter, C2StreEngine, TrackSnapshot,
)
from vlm_channel import (
    VlmChannelSink, RichDetection, FrameDigest, SceneCaption,
    normalize_label, compose_scene_description,
)
from network_transport import (
    StreUdpTransmitter, NetworkAwareStreSink,
    get_or_create_psk, get_persistent_counter_path,
)


# ============================================================================
# Spatial deduper — minimum edge-side bookkeeping
# ============================================================================
#
# Per spec §2, entity resolution is the C2 Gateway's job, not the edge's.
# The edge just needs to:
#   (1) prevent emitting one Type 1 per detection per frame (would blow
#       the 9.6 kbps budget by 10x);
#   (2) attach a stable-ish id to each detection so EdgeStreEmitter's
#       per-track rate limiter can do its job.
#
# We do this by hashing detections into a coarse spatial grid by class.
# Any detection of class X falling into grid cell (gx, gy) gets the same
# track_id. When it moves to a neighboring cell, it gets a new track_id —
# and that's fine. The C2StreEngine re-resolves identity by bucketing on
# (source_id, object_class, location_grid), which is the same idea but
# performed across CASKs at the gateway. Per the spec, that's where it
# belongs.
#
# Motion vectors are NOT computed at edge. They get derived at C2 from
# consecutive observation deltas in the same entity bucket.

@dataclass
class _Track:
    """Edge-side detection record. Not a real track — see SpatialDeduper docs."""
    track_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    motion_vector: List[float] = field(default_factory=lambda: [0.0, 0.0])
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    last_frame_idx: int = 0
    miss_count: int = 0


class SpatialDeduper:
    """
    Pure (class, grid_cell) -> stable_id mapping. Zero tracking logic.

    Replaces the SimpleIouTracker stub. The motivation is intentional:
    spec §2 puts entity resolution at the C2 Gateway, not the edge. Doing
    real frame-to-frame tracking on the Jetson would (a) waste compute,
    (b) duplicate work that C2 will do anyway, (c) suppress raw
    observations that C2 actually wants for pattern matching.

    Args:
        grid_px: spatial bucket size. Smaller = more IDs (more emission),
                 larger = coarser dedup. 80px is a reasonable default for
                 1080p aerial drone footage.
    """

    def __init__(self, grid_px: int = 80, stale_after_s: float = 30.0):
        self.grid_px = grid_px
        self.stale_after_s = stale_after_s
        self._id_for_cell: Dict[Tuple[str, int, int], int] = {}
        self._last_seen: Dict[int, float] = {}
        self._next_id = 1

    def update(self, detections: List[Detection], frame_idx: int,
               frame_shape: Tuple[int, int]) -> List[_Track]:
        now = time.monotonic()
        out: List[_Track] = []

        for d in detections:
            cx = int((d.bbox[0] + d.bbox[2]) / 2) // self.grid_px
            cy = int((d.bbox[1] + d.bbox[3]) / 2) // self.grid_px
            key = (d.label, cx, cy)

            if key not in self._id_for_cell:
                self._id_for_cell[key] = self._next_id
                self._next_id = (self._next_id + 1) & 0xFFFF
                if self._next_id == 0:
                    self._next_id = 1
            tid = self._id_for_cell[key]
            self._last_seen[tid] = now

            out.append(_Track(
                track_id=tid,
                class_name=d.label,
                confidence=float(d.conf),
                bbox=list(d.bbox),
                motion_vector=[0.0, 0.0],   # derived at C2 from deltas
                first_seen_at=now,
                last_seen_at=now,
                last_frame_idx=frame_idx,
                miss_count=0,
            ))

        # GC: drop IDs unseen for stale_after_s. Lets us reuse cells that
        # vehicles have left.
        cutoff = now - self.stale_after_s
        for tid, last_t in list(self._last_seen.items()):
            if last_t < cutoff:
                del self._last_seen[tid]
        # Drop matching cell entries
        for k, tid in list(self._id_for_cell.items()):
            if tid not in self._last_seen:
                del self._id_for_cell[k]

        return out

    def consume_terminations(self) -> List[Dict]:
        # No track lifecycle to terminate. Hazard detection lives at C2
        # and uses absence of observations rather than termination events.
        return []


# ============================================================================
# Annotation
# ============================================================================

_TRACK_COLORS = [
    (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 255, 56),
    (56, 255, 255), (255, 56, 255), (180, 120, 60), (120, 60, 180),
]


def annotate(frame: np.ndarray, tracks: List[_Track],
             recent_type2: List[StreInference],
             frame_idx: int, fps: float, type1_count: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Scale fonts to image size — readable on a monitor across a room
    scale = max(0.7, min(w, h) / 1000.0)
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = scale * 0.7
    label_thick = max(1, int(scale * 1.5))

    # Draw tracks
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        if x1 == x2 == y1 == y2 == 0:
            continue   # skip degenerate boxes
        color = _TRACK_COLORS[t.track_id % len(_TRACK_COLORS)]
        # Thicker box, more visible
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(scale * 2.5)))
        label = f"#{t.track_id} {t.class_name} {int(t.confidence*100)}%"
        (tw, th), _ = cv2.getTextSize(label, label_font, label_scale, label_thick)
        # Filled label background
        pad = 4
        cv2.rectangle(out, (x1, y1 - th - pad*2),
                      (x1 + tw + pad*2, y1), color, -1)
        cv2.putText(out, label, (x1 + pad, y1 - pad),
                    label_font, label_scale, (0, 0, 0), label_thick)

    # HUD top bar — chunky for demo readability
    hud_h = max(40, int(scale * 50))
    cv2.rectangle(out, (0, 0), (w, hud_h), (15, 15, 15), -1)
    pname_lookup = {v: k for k, v in PATTERN_ID.items()}
    last_inf = recent_type2[-1] if recent_type2 else None
    inf_str = ""
    if last_inf:
        pname = pname_lookup.get(last_inf.pattern_id, f"0x{last_inf.pattern_id:02x}")
        threat_lookup = {v: k for k, v in THREAT_LEVEL.items()}
        threat = threat_lookup.get(last_inf.threat_level, "?")
        inf_str = f"  |  C2: {pname}  threat={threat}  conf={last_inf.confidence}%"
    hud = (f"CASK 0xCA51   frame={frame_idx}   fps={fps:.1f}   "
           f"tracks={len(tracks)}   wire_emitted={type1_count}{inf_str}")
    cv2.putText(out, hud, (10, int(hud_h * 0.7)),
                label_font, scale * 0.6, (180, 220, 80), label_thick)

    # Big banner when an inference fires recently (last 30 frames)
    if last_inf and (frame_idx - getattr(last_inf, "_fired_at_frame", -100) < 30):
        pname_str = pname_lookup.get(last_inf.pattern_id, '?').upper()
        banner = f">>> {pname_str} DETECTED <<<"
        banner_h = max(50, int(scale * 60))
        cv2.rectangle(out, (0, hud_h), (w, hud_h + banner_h), (0, 0, 120), -1)
        (bw, bh), _ = cv2.getTextSize(banner, label_font, scale, label_thick + 1)
        cv2.putText(out, banner, ((w - bw) // 2, hud_h + int(banner_h * 0.7)),
                    label_font, scale, (200, 200, 255), label_thick + 1)

    return out


# ============================================================================
# Main loop
# ============================================================================

@dataclass
class RunConfig:
    video_path: str                       # path to video file OR webcam index as string ("0", "1")
    output_dir: str = "cask_run"
    muaythai_weights: str = ""           # path to MilitaryConvoy-YOLO11L.pt
    device: str = "cpu"                   # "cuda" | "mps" | "cpu"
    frame_skip: int = 3                   # process 1 of every N frames
    gdino_every_n: int = 10               # within processed frames
    max_seconds: float = 0.0              # 0 = whole video / unlimited webcam
    save_annotated: bool = True
    frame_save_every: int = 1             # save 1 of every N processed frames (1 = save all)
    cask_source_id: int = 0xCA51
    camera_lat: float = 48.158500
    camera_lon: float = 37.727000
    camera_heading_deg: int = 180
    px_per_meter: float = 8.0
    emit_interval_s: float = 1.0          # min seconds between Type 1s per track
    dedup_grid_px: int = 80               # SpatialDeduper grid cell size
    webcam: bool = False                  # True = treat video_path as webcam index
    webcam_width: int = 1280              # requested webcam capture width
    webcam_height: int = 720              # requested webcam capture height
    show_window: bool = False             # cv2.imshow live preview (set False over SSH)
    fullscreen: bool = False              # open imshow window fullscreen
    c2_host: str = ""                     # if set, fan STRE UDP to this host
    c2_port: int = 9601
    use_persistent_psk: bool = True       # use ~/.cask/cask.psk (vs per-run)
    gdino_box_threshold: float = 0.20     # lower = more grounding_dino contributions
    gdino_text_threshold: float = 0.20
    gdino_prompt: str = ""                # custom open-vocab prompt for gdino


@dataclass
class RunStats:
    frames_seen: int = 0
    frames_processed: int = 0
    detections_total: int = 0
    type1_total: int = 0
    type2_total: int = 0
    inferences: List[Dict] = field(default_factory=list)
    type2_by_pattern: Dict[str, int] = field(default_factory=dict)


def _generate_psk(out_dir: Path) -> bytes:
    """
    For demos: write the PSK to disk so we can decrypt wire_log.bin afterward.
    In production this comes from secure provisioning (e.g. RFC 5869 HKDF
    from a hardware-backed root key) and never touches disk in plaintext.
    The demo PSK lives in {output_dir}/demo_psk.bin and is regenerated each
    run — no key reuse across runs.
    """
    psk_path = out_dir / "demo_psk.bin"
    psk = os.urandom(32)
    psk_path.write_bytes(psk)
    psk_path.chmod(0o600)
    return psk


def run(cfg: RunConfig) -> RunStats:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated"
    if cfg.save_annotated:
        annotated_dir.mkdir(exist_ok=True)

    # --- Components ----------------------------------------------------------
    print(f"[init] EnsembleDetector loading on {cfg.device}...")
    detector = EnsembleDetector(
        muaythai_weights=cfg.muaythai_weights,
        device=cfg.device,
        gdino_every_n=cfg.gdino_every_n,
        gdino_box_threshold=cfg.gdino_box_threshold,
        gdino_text_threshold=cfg.gdino_text_threshold,
        gdino_prompt=cfg.gdino_prompt or None,
    )
    tracker = SpatialDeduper(grid_px=cfg.dedup_grid_px)

    # Foundry-side sink (rich JSON for the dashboard)
    foundry_sink = JsonlSink(str(out_dir / "foundry_events.jsonl"))
    factory = OntologyEventFactory(
        sensor_id=f"cask-{cfg.cask_source_id:04x}",
        outbox_path=str(out_dir / "outbox.db"),
        sink=foundry_sink,
    )
    factory.start()

    # Radio-side sink (sealed CBOR for the wire) — uses persistent PSK
    # and counter from ~/.cask/. C2 must have the same PSK file. The
    # counter persists across runs/reboots so an attacker cannot replay
    # a captured wire_log from a previous run.
    if cfg.use_persistent_psk:
        psk = get_or_create_psk()
        counter_path = get_persistent_counter_path()
    else:
        # Legacy per-run PSK — only useful for hermetic demos and tests
        psk = _generate_psk(out_dir)
        counter_path = out_dir / "stre_counter.txt"

    counter_state = AntiReplayState(
        source_id=cfg.cask_source_id,
        state_path=str(counter_path),
    )
    sealer = CoseSealer(psk, source_id=cfg.cask_source_id, counter_state=counter_state)
    local_stre_sink = StreSink(sealer, wire_log_path=str(out_dir / "wire_log.bin"))

    # If a C2 host is configured, fan every sealed message over UDP to it.
    # Local file write is unconditional — wire_log.bin always reflects what
    # edge perceived, regardless of network state. UDP send is best-effort
    # and silently blackholes during a kill switch / jamming.
    udp_tx = None
    if cfg.c2_host:
        udp_tx = StreUdpTransmitter(cfg.c2_host, cfg.c2_port)
        print(f"[net] STRE UDP transmitter -> {cfg.c2_host}:{cfg.c2_port}")
    else:
        print(f"[net] no C2 host configured (running edge-only, no live stream)")
    stre_sink = NetworkAwareStreSink(local_stre_sink, udp_tx)

    # Rich open-vocabulary channel — runs alongside STRE on a higher-bandwidth
    # path (in production: satcom burst or mesh proximity; in demo: separate
    # JSONL files). Carries everything STRE Type 1 schema can't: free-text
    # labels, pixel-space bboxes, frame-level counts, scene-level captions
    # on Type 2 fire.
    vlm_sink = VlmChannelSink(str(out_dir))

    # Edge: Type 1 emitter
    emitter = EdgeStreEmitter(
        source_id=cfg.cask_source_id,
        sensor_type="EO",
        emit_interval_s=cfg.emit_interval_s,
        camera_lat=cfg.camera_lat,
        camera_lon=cfg.camera_lon,
        camera_heading_deg=cfg.camera_heading_deg,
        px_per_meter=cfg.px_per_meter,
    )

    # C2: STRE engine (lives at gateway in real deployment; here we run it
    # in-process to demo the full loop on a single machine)
    engine = C2StreEngine()

    # --- Open video / webcam -------------------------------------------------
    if cfg.webcam:
        # Webcam index given as the video_path string ("0", "1", ...)
        try:
            cam_index = int(cfg.video_path)
        except ValueError:
            raise SystemExit(f"--webcam requires integer index, got '{cfg.video_path}'")
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.webcam_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab freshest frame
        if not cap.isOpened():
            raise SystemExit(f"could not open webcam {cam_index}")
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = -1
        print(f"[webcam] device {cam_index}: {actual_w}x{actual_h} @ {fps_in:.1f} fps")
    else:
        cap = cv2.VideoCapture(cfg.video_path)
        if not cap.isOpened():
            raise SystemExit(f"could not open {cfg.video_path}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[video] {cfg.video_path}: {total_frames} frames @ {fps_in:.1f} fps")

    # For webcam, max_seconds=0 means run forever (until Ctrl-C)
    if cfg.webcam:
        max_frames = int(cfg.max_seconds * fps_in) if cfg.max_seconds > 0 else 10**9
    else:
        max_frames = int(cfg.max_seconds * fps_in) if cfg.max_seconds > 0 else total_frames
    stats = RunStats()
    recent_type2: List[StreInference] = []
    last_progress_t = time.monotonic()
    loop_start = time.monotonic()

    print(f"[run] processing every {cfg.frame_skip} frames; "
          f"gdino every {cfg.gdino_every_n} processed frames")
    if cfg.webcam:
        print(f"[run] webcam mode — Ctrl-C to stop")
    if cfg.show_window:
        # Use NORMAL so the window is resizable. Pre-create with a sensible
        # size so it doesn't open at native webcam res (often 640x480).
        cv2.namedWindow("CASK", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CASK", 1280, 720)
        if cfg.fullscreen:
            cv2.setWindowProperty("CASK", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        print(f"[run] live window open — press 'q' to stop, 'f' to toggle fullscreen")
    print()

    frame_idx = 0
    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                if cfg.webcam:
                    # Brief stall — retry a few times before giving up
                    time.sleep(0.05)
                    continue
                break
            stats.frames_seen += 1

            # Frame skipping
            if frame_idx % cfg.frame_skip != 0:
                frame_idx += 1
                continue

            t0 = time.monotonic()

            # 1. Perception
            det_result = detector.detect(frame)
            stats.detections_total += len(det_result.detections)

            # 1b. VLM channel: capture full open-vocab perception output BEFORE
            # the STRE bridge throws away anything not in the closed enum.
            # We don't yet know which of these will get Type 1 emitted (that
            # happens after dedup + rate-limit), so we mark is_in_stre=False
            # initially and patch the count later via the FrameDigest.
            frame_h, frame_w = frame.shape[:2]
            now_unix = int(time.time())
            vlm_detections_this_frame: List[RichDetection] = []
            for d in det_result.detections:
                rd = RichDetection(
                    event_id=0,   # patched below if this becomes a Type 1
                    frame_idx=frame_idx,
                    source_id=cfg.cask_source_id,
                    timestamp_unix=now_unix,
                    label=d.label,
                    label_normalized=normalize_label(d.label),
                    source_model=d.source,
                    bbox_xyxy=list(d.bbox),
                    frame_width=frame_w,
                    frame_height=frame_h,
                    confidence=float(d.conf),
                    is_in_stre=False,
                )
                vlm_detections_this_frame.append(rd)

            # 2. Tracking (spatial dedup only — see SpatialDeduper docstring)
            tracks = tracker.update(det_result.detections, frame_idx, frame.shape)

            # 3. Edge: Type 1 emission (rate-limited per track)
            snapshots = [
                TrackSnapshot(
                    track_id=t.track_id,
                    class_name=t.class_name,
                    confidence=t.confidence,
                    bbox=t.bbox,
                    motion_vector=t.motion_vector,
                    last_seen_unix=int(time.time()),
                )
                for t in tracks
            ]
            type1_messages = emitter.update(snapshots)

            # Foundry mirror: emit a track event for every active track this
            # frame, carrying the actual pixel-space bbox. This is the
            # dashboard-facing stream — NOT bandwidth-constrained, so we can
            # afford to ship the visual context that STRE Type 1 deliberately
            # strips (it would never fit in 67 sealed bytes anyway).
            for t in tracks:
                factory.emit_track(
                    track_id=t.track_id,
                    class_name=t.class_name,
                    confidence=t.confidence,
                    first_seen_at=now_iso(),
                    last_seen_at=now_iso(),
                    motion_vector=t.motion_vector,
                    latest_bbox=t.bbox,            # ← actual pixel bbox now
                )

            # Tactical wire: write Type 1s to the sealed CBOR log
            for obs in type1_messages:
                stre_sink.write_type1(obs)
                stats.type1_total += 1

            # 4. C2 side: ingest Type 1s, fire Type 2s
            for obs in type1_messages:
                inferences = engine.ingest(obs)
                for inf in inferences:
                    inf._fired_at_frame = frame_idx   # for HUD banner
                    stre_sink.write_type2(inf)
                    recent_type2.append(inf)
                    if len(recent_type2) > 10:
                        recent_type2.pop(0)
                    stats.type2_total += 1
                    pname = next((k for k, v in PATTERN_ID.items()
                                  if v == inf.pattern_id), f"0x{inf.pattern_id:02x}")
                    stats.type2_by_pattern[pname] = stats.type2_by_pattern.get(pname, 0) + 1
                    stats.inferences.append({
                        "frame": frame_idx,
                        "pattern": pname,
                        "threat_level": inf.threat_level,
                        "confidence": inf.confidence,
                        "evidence_ids": inf.evidence_ids,
                        "actions": inf.actions,
                    })
                    # Foundry dashboard mirror
                    threat_lookup = {v: k for k, v in THREAT_LEVEL.items()}
                    action_lookup = {v: k for k, v in ACTION.items()}
                    factory.emit_observation(
                        event_type="stre_inference",
                        summary=f"STRE pattern '{pname}' fired (threat={threat_lookup.get(inf.threat_level, '?')}, "
                                f"conf={inf.confidence}%, evidence={inf.evidence_ids})",
                        confidence=inf.confidence / 100.0,
                        raw_metadata={
                            "pattern_id": inf.pattern_id,
                            "pattern_name": pname,
                            "threat_level": threat_lookup.get(inf.threat_level, "?"),
                            "evidence_event_ids": inf.evidence_ids,
                            "recommended_actions": [action_lookup.get(a, hex(a))
                                                    for a in inf.actions],
                            "target_lat": inf.target_lat / 1e6,
                            "target_lon": inf.target_lon / 1e6,
                            "eta_sec": inf.eta_sec,
                        },
                    )
                    print(
                        f"\n  >>> INFERENCE FIRED: pattern={pname} "
                        f"threat={threat_lookup.get(inf.threat_level, '?')} "
                        f"conf={inf.confidence}%\n"
                        f"      evidence event_ids={inf.evidence_ids}  "
                        f"actions={[action_lookup.get(a, hex(a)) for a in inf.actions]}\n"
                    )

            stats.frames_processed += 1

            # 1c. VLM channel: now that STRE has filtered/dedup'd, mark which
            # detections actually made it onto the tactical wire. We match
            # by label (post-bridge classes only) — imperfect but adequate
            # for the diagnostic value of stre_drop_rate.
            stre_emitted_count_this_frame = len(type1_messages)
            stre_classes_emitted: Dict[str, int] = {}
            for obs in type1_messages:
                inv_class = {v: k for k, v in OBJECT_CLASS.items()}
                cn = inv_class.get(obs.object_class, "unknown")
                stre_classes_emitted[cn] = stre_classes_emitted.get(cn, 0) + 1

            for rd in vlm_detections_this_frame:
                # Mark in_stre if any STRE message of compatible class was
                # emitted. This is approximate but useful diagnostic signal.
                # The semantics: "this detection's class was represented on
                # the wire", not "this exact detection became this exact msg".
                from stre_codec import INTERNAL_TO_STRE_CLASS
                stre_code = INTERNAL_TO_STRE_CLASS.get(rd.label_normalized.replace(
                    "tracked_vehicle", "tank"
                ).replace("wheeled_vehicle", "truck"))
                if stre_code is not None:
                    inv = {v: k for k, v in OBJECT_CLASS.items()}
                    cn = inv.get(stre_code, "")
                    if stre_classes_emitted.get(cn, 0) > 0:
                        rd.is_in_stre = True
                vlm_sink.write_detection(rd)

            # Frame digest: cheap to compute, high info density for analysts.
            if vlm_detections_this_frame:
                by_label: Dict[str, int] = {}
                by_source: Dict[str, int] = {}
                for d in vlm_detections_this_frame:
                    by_label[d.label_normalized] = by_label.get(d.label_normalized, 0) + 1
                    by_source[d.source_model] = by_source.get(d.source_model, 0) + 1
                stre_emitted = sum(1 for d in vlm_detections_this_frame if d.is_in_stre)
                digest = FrameDigest(
                    frame_idx=frame_idx,
                    source_id=cfg.cask_source_id,
                    timestamp_unix=now_unix,
                    detection_counts_by_label=by_label,
                    detection_counts_by_source=by_source,
                    total_detections=len(vlm_detections_this_frame),
                    mean_confidence=sum(d.confidence for d in vlm_detections_this_frame)
                                    / len(vlm_detections_this_frame),
                    stre_emitted_count=stre_emitted,
                    stre_drop_rate=1 - stre_emitted / len(vlm_detections_this_frame),
                )
                vlm_sink.write_frame_digest(digest)

            # Scene caption: only on Type 2 fire. High value, low frequency.
            for inf_obj in [i for i in recent_type2[-3:]
                            if getattr(i, "_fired_at_frame", -1) == frame_idx]:
                pname = next((k for k, v in PATTERN_ID.items()
                              if v == inf_obj.pattern_id), f"0x{inf_obj.pattern_id:02x}")
                labels_seen = [d.label for d in vlm_detections_this_frame]
                composition = {}
                for d in vlm_detections_this_frame:
                    composition[d.label_normalized] = composition.get(d.label_normalized, 0) + 1
                scene = SceneCaption(
                    inference_id=inf_obj.inference_id,
                    frame_idx=frame_idx,
                    source_id=cfg.cask_source_id,
                    timestamp_unix=now_unix,
                    open_vocab_labels=labels_seen,
                    composition=composition,
                    pattern_name=pname,
                    description=compose_scene_description(labels_seen, composition, pname),
                )
                vlm_sink.write_scene_caption(scene)

            # 5. Annotation
            annotated = None
            if cfg.save_annotated or cfg.show_window:
                elapsed = time.monotonic() - loop_start
                avg_fps = stats.frames_processed / elapsed if elapsed > 0 else 0.0
                annotated = annotate(frame, tracks, recent_type2,
                                     frame_idx, avg_fps, stats.type1_total)
            if (cfg.save_annotated and annotated is not None
                    and stats.frames_processed % cfg.frame_save_every == 0):
                cv2.imwrite(
                    str(annotated_dir / f"frame_{frame_idx:06d}.jpg"),
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
            if cfg.show_window and annotated is not None:
                cv2.imshow("CASK", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[run] 'q' pressed — stopping")
                    break
                elif key == ord('f'):
                    cur = cv2.getWindowProperty("CASK", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        "CASK", cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if cur == cv2.WINDOW_FULLSCREEN
                                          else cv2.WINDOW_FULLSCREEN,
                    )

            # 6. Progress reporting — every 2s, show what was actually seen
            now = time.monotonic()
            if now - last_progress_t >= 2.0:
                elapsed = now - loop_start
                fps_eff = stats.frames_processed / elapsed if elapsed > 0 else 0.0
                max_str = "∞" if cfg.webcam and cfg.max_seconds == 0 else str(max_frames)

                # Build a "what did we see" summary from this frame's detections.
                # This is the diagnostic the user actually wants: not 'tracks=N',
                # but 'I saw 3 tanks (avg 65%) and 2 trucks (avg 48%) this frame'.
                if vlm_detections_this_frame:
                    by_label_w_conf: Dict[str, List[float]] = {}
                    by_source_count: Dict[str, int] = {}
                    for d in vlm_detections_this_frame:
                        by_label_w_conf.setdefault(d.label, []).append(d.confidence)
                        by_source_count[d.source_model] = by_source_count.get(d.source_model, 0) + 1
                    # Sort labels by mean confidence, take top 4 for terminal width
                    label_summary = sorted(
                        by_label_w_conf.items(),
                        key=lambda kv: -sum(kv[1]) / len(kv[1])
                    )[:4]
                    seen_str = ", ".join(
                        f"{lbl}×{len(confs)}@{int(100*sum(confs)/len(confs))}%"
                        for lbl, confs in label_summary
                    )
                    src_str = "+".join(f"{s}:{c}" for s, c in by_source_count.items())
                else:
                    seen_str = "(nothing)"
                    src_str = "-"

                print(
                    f"  [t={elapsed:5.1f}s f={frame_idx:>5}/{max_str} fps={fps_eff:4.1f}] "
                    f"saw: {seen_str:<55s} "
                    f"src={src_str:<25s} "
                    f"wire={stats.type1_total} infer={stats.type2_total}"
                )
                last_progress_t = now

            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[run] Ctrl-C — shutting down cleanly...")
    finally:
        cap.release()
        if cfg.show_window:
            cv2.destroyAllWindows()
        factory.stop()

    # --- Summary -------------------------------------------------------------
    elapsed = time.monotonic() - loop_start
    summary_path = out_dir / "run_summary.md"
    with open(summary_path, "w") as f:
        f.write("# CASK Run Summary\n\n")
        f.write(f"- Video: `{cfg.video_path}`\n")
        f.write(f"- Output dir: `{out_dir}`\n")
        f.write(f"- Wall time: {elapsed:.1f}s\n")
        f.write(f"- Frames seen: {stats.frames_seen}\n")
        f.write(f"- Frames processed: {stats.frames_processed} "
                f"(skip={cfg.frame_skip})\n")
        f.write(f"- Effective FPS: "
                f"{stats.frames_processed / elapsed if elapsed > 0 else 0:.2f}\n\n")
        f.write("## Detection volume\n\n")
        f.write(f"- Total detections: {stats.detections_total}\n")
        f.write(f"- Type 1 emitted (sealed CBOR over wire): {stats.type1_total}\n")
        f.write(f"- Type 2 inferences fired: {stats.type2_total}\n\n")
        f.write("## Bandwidth\n\n")
        wire_bytes = stre_sink.stats()["total_bytes_sealed"]
        kbps = (wire_bytes * 8) / max(1, elapsed) / 1000
        f.write(f"- Sealed CBOR bytes on wire: {wire_bytes}\n")
        f.write(f"- Average wire rate: {kbps:.2f} kbps "
                f"(budget: 9.6 kbps; headroom: "
                f"{(9.6 - kbps) / 9.6 * 100:.0f}%)\n\n")
        f.write("## Patterns fired\n\n")
        if not stats.type2_by_pattern:
            f.write("_(no patterns matched in this run)_\n\n")
        else:
            f.write("| Pattern | Count |\n|---|---|\n")
            for k, v in sorted(stats.type2_by_pattern.items(),
                                key=lambda kv: -kv[1]):
                f.write(f"| `{k}` | {v} |\n")
            f.write("\n")
        f.write("## Inferences\n\n")
        for inf in stats.inferences[:20]:
            f.write(f"- frame {inf['frame']}: `{inf['pattern']}` "
                    f"threat={inf['threat_level']} "
                    f"conf={inf['confidence']} "
                    f"evidence={inf['evidence_ids']}\n")
        if len(stats.inferences) > 20:
            f.write(f"- ... ({len(stats.inferences) - 20} more)\n")

    # JSON dump for programmatic inspection
    with open(out_dir / "run_stats.json", "w") as f:
        json.dump(asdict(stats), f, indent=2, default=str)

    print(f"\n=== Run complete ===")
    print(f"  wall time: {elapsed:.1f}s")
    print(f"  frames processed: {stats.frames_processed}")
    print(f"  Type 1 emitted: {stats.type1_total}")
    print(f"  Type 2 fired:   {stats.type2_total}  by pattern: {stats.type2_by_pattern}")
    print(f"  wire bytes: {wire_bytes} ({kbps:.2f} kbps avg)")
    vlm_stats = vlm_sink.stats()
    print(f"  VLM channel: {vlm_stats['detections_written']} detections, "
          f"{vlm_stats['frames_written']} frame digests, "
          f"{vlm_stats['scenes_written']} scene captions "
          f"({vlm_stats['bytes_written']} bytes)")
    print(f"  outputs in: {out_dir}/")
    return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="CASK end-to-end demo orchestrator")
    p.add_argument("video", help="Path to input video file")
    p.add_argument("--output", default="cask_run")
    p.add_argument("--weights", default="",
                   help="Path to MilitaryConvoy-YOLO11L .pt weights "
                        "(downloaded from HF on first run if blank — but YMMV)")
    p.add_argument("--device", default="auto",
                   help="cuda | mps | cpu | auto")
    p.add_argument("--skip", type=int, default=3,
                   help="Process every Nth frame")
    p.add_argument("--gdino-every", type=int, default=10,
                   help="Run Grounding DINO every Nth processed frame")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Stop after N seconds of input video (0 = whole video)")
    p.add_argument("--no-annotated", action="store_true",
                   help="Skip writing annotated frames (faster)")
    p.add_argument("--frame-save-every", type=int, default=1,
                   help="Save 1 of every N processed frames to disk "
                        "(1 = save all; 30 = save ~once per second at skip=2)")
    p.add_argument("--source-id", type=lambda x: int(x, 0), default=0xCA51,
                   help="CASK source_id (uint16, hex or decimal)")
    p.add_argument("--webcam", action="store_true",
                   help="Treat the 'video' positional arg as a webcam index "
                        "(e.g. '0' or '1'). Runs forever until Ctrl-C.")
    p.add_argument("--show-window", action="store_true",
                   help="Pop an OpenCV preview window (skip if running over SSH "
                        "without X forwarding). Press 'q' in the window to stop, "
                        "'f' to toggle fullscreen.")
    p.add_argument("--fullscreen", action="store_true",
                   help="Open the preview window fullscreen "
                        "(requires --show-window)")
    p.add_argument("--dedup-grid-px", type=int, default=80,
                   help="SpatialDeduper grid cell size in pixels (default 80)")
    p.add_argument("--emit-interval-s", type=float, default=1.0,
                   help="Min seconds between Type 1 emissions per stable id")
    p.add_argument("--c2-host", default="",
                   help="If set, send STRE UDP to this hostname/IP. Without "
                        "this flag, edge runs locally with no live stream.")
    p.add_argument("--c2-port", type=int, default=9601,
                   help="UDP port for STRE (default: 9601)")
    p.add_argument("--ephemeral-psk", action="store_true",
                   help="Generate a fresh per-run PSK in {output}/demo_psk.bin "
                        "instead of using the persistent ~/.cask/cask.psk. "
                        "Useful for hermetic testing; not for live demo with C2.")
    p.add_argument("--gdino-box-threshold", type=float, default=0.20,
                   help="Grounding DINO box confidence threshold. Lower = "
                        "more contributions, more noise. Default 0.20.")
    p.add_argument("--gdino-text-threshold", type=float, default=0.20,
                   help="Grounding DINO text-grounding threshold. Default 0.20.")
    p.add_argument("--gdino-prompt", default="",
                   help="Custom GDino prompt. Default tuned for military "
                        "convoy footage. Use a richer prompt to surface "
                        "more open-vocab labels.")
    args = p.parse_args()

    if args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    cfg = RunConfig(
        video_path=args.video,
        output_dir=args.output,
        muaythai_weights=args.weights,
        device=device,
        frame_skip=args.skip,
        gdino_every_n=args.gdino_every,
        max_seconds=args.max_seconds,
        save_annotated=not args.no_annotated,
        frame_save_every=args.frame_save_every,
        cask_source_id=args.source_id,
        webcam=args.webcam,
        show_window=args.show_window,
        fullscreen=args.fullscreen,
        dedup_grid_px=args.dedup_grid_px,
        emit_interval_s=args.emit_interval_s,
        c2_host=args.c2_host,
        c2_port=args.c2_port,
        use_persistent_psk=not args.ephemeral_psk,
        gdino_box_threshold=args.gdino_box_threshold,
        gdino_text_threshold=args.gdino_text_threshold,
        gdino_prompt=args.gdino_prompt,
    )
    run(cfg)


if __name__ == "__main__":
    main()
