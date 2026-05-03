"""
ontology_factory.py — CASK ontology object factory with kill-switch
resilient outbox.

Architecture:

    detection event ──► OntologyEventFactory.emit()
                              │
                              ▼
                      ┌───────────────────┐
                      │  Local SQLite     │   (always written first;
                      │  outbox           │    survives restart, kill-switch)
                      └───────────────────┘
                              │
                              ▼
                      ┌───────────────────┐
                      │  TelemetrySink    │   (interface: cask | jsonl)
                      └───────────────────┘
                         │              │
                         ▼              ▼
                 CaskOntologySink  JsonlSink (fallback)
                 (Foundry sync)    (always works)

Key resilience properties:
  - Every event written to local SQLite *before* sync attempt
  - Sync runs on a background worker; perception loop never blocks on network
  - Failed syncs increment retry_count; exponential backoff per event
  - When network restored, drain queue oldest-first; preserve original timestamps
  - Idempotency via event_id (UUID) — re-syncing a duplicate is a no-op
  - SinkSelector lets you swap CASK ↔ JSONL with one config flag for demo

For the hackathon: implement the JsonlSink today, stub the CaskOntologySink
to write its OSDK calls, plug in real OSDK once you have credentials from
the Palantir mentor. The kill-switch demo works regardless of which sink
you're using because the buffering happens in the SQLite layer below.

Schema mirrors what we discussed for CASK ontology:
  - Sensor (registered once at startup)
  - ObservationEvent (one per emit)
  - Track (created/updated as ByteTrack assigns IDs)
  - EdgeHealth (heartbeat once per second)

Usage:
    from ontology_factory import OntologyEventFactory, JsonlSink

    factory = OntologyEventFactory(
        sensor_id="cask-jetson-01",
        outbox_path="cask_outbox.db",
        sink=JsonlSink("events.jsonl"),
    )
    factory.start()  # spawns background sync worker

    factory.emit_observation(
        event_type="track_summary",
        summary="4 armored vehicles approaching from camera-left",
        objects=[{"class": "tank", "count": 3, "confidence": 0.82}],
        track_ids=[12, 13, 14, 15],
    )

    factory.emit_health(inference_fps=4.2, queue_depth=3)

    factory.stop()  # graceful shutdown
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from typing import List, Dict, Optional, Any


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class ObservationEvent:
    event_id: str
    sensor_id: str
    event_type: str           # 'track_summary' | 'visual_hazard' | 'footage_lost' | etc
    timestamp: str            # ISO8601 UTC, time of capture (NOT sync)
    summary: str              # human-readable string for dashboard
    confidence: float
    objects: List[Dict[str, Any]] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackUpdate:
    track_id: int
    sensor_id: str
    class_name: str
    confidence: float
    first_seen_at: str
    last_seen_at: str
    motion_vector: List[float]      # [dx_px_per_sec, dy_px_per_sec]
    latest_bbox: List[float]        # [x1, y1, x2, y2]


@dataclass
class EdgeHealth:
    sensor_id: str
    timestamp: str
    camera_ok: bool
    inference_fps: float
    network_ok: bool
    queue_depth: int


# --------------------------------------------------------------------------
# Sinks (where events ultimately land)
# --------------------------------------------------------------------------

class TelemetrySink(ABC):
    """Common interface for any destination — CASK/Foundry or JSONL fallback."""

    @abstractmethod
    def write_observation(self, event: ObservationEvent) -> None:
        """Should raise on failure so the outbox knows to retry."""
        ...

    @abstractmethod
    def write_track(self, track: TrackUpdate) -> None:
        ...

    @abstractmethod
    def write_health(self, health: EdgeHealth) -> None:
        ...

    @abstractmethod
    def healthcheck(self) -> bool:
        """Returns True if the sink is reachable. Used by the worker
        to decide whether to attempt sync."""
        ...


class JsonlSink(TelemetrySink):
    """Append events to a JSON-lines file. Always works — your safety net."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, kind: str, payload: dict) -> None:
        line = json.dumps({"_kind": kind, **payload})
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")

    def write_observation(self, event: ObservationEvent) -> None:
        self._append("observation", asdict(event))

    def write_track(self, track: TrackUpdate) -> None:
        self._append("track", asdict(track))

    def write_health(self, health: EdgeHealth) -> None:
        self._append("health", asdict(health))

    def healthcheck(self) -> bool:
        return True


class CaskOntologySink(TelemetrySink):
    """
    OSDK-backed sink. STUB — replace the inner calls with the actual OSDK
    Python client once you have it from the Palantir mentor.

    The pattern below shows where each OSDK call goes. Until then, this
    behaves like JsonlSink (writes to a separate file labeled cask_).

    Talk to the mentor about:
      1. The OSDK Python package name (likely a custom-generated SDK
         per CASK enrollment)
      2. The ontology object type names for your enrollment
         (might not be exactly 'ObservationEvent' / 'Track' / 'EdgeHealth')
      3. Auth: probably a token bundle, possibly mTLS cert
      4. Whether write goes through a Foundry write-back action or
         directly to the ontology
    """

    def __init__(self, jsonl_fallback_path: str = "cask_attempted.jsonl"):
        self._fallback = JsonlSink(jsonl_fallback_path)
        self._network_up = True  # flip via .set_network() to simulate kill-switch
        # TODO real init:
        # self.client = osdk.Client(token=..., url=...)
        # self.ontology = self.client.ontology("YourOntologyName")

    def set_network(self, up: bool) -> None:
        """Manual kill-switch hook for demo / testing."""
        self._network_up = up

    def healthcheck(self) -> bool:
        return self._network_up
        # Real implementation:
        # try:
        #     self.client.ping(timeout=2.0)
        #     return True
        # except Exception:
        #     return False

    def write_observation(self, event: ObservationEvent) -> None:
        if not self._network_up:
            raise ConnectionError("CASK network down (kill-switch)")
        # Real OSDK pattern:
        #   self.ontology.objects.ObservationEvent.create(
        #       eventId=event.event_id,
        #       sensorId=event.sensor_id,
        #       eventType=event.event_type,
        #       timestamp=event.timestamp,
        #       summary=event.summary,
        #       confidence=event.confidence,
        #       objectsJson=json.dumps(event.objects),
        #       trackIds=event.track_ids,
        #   ).execute()
        self._fallback.write_observation(event)

    def write_track(self, track: TrackUpdate) -> None:
        if not self._network_up:
            raise ConnectionError("CASK network down (kill-switch)")
        # Real OSDK pattern:
        #   self.ontology.objects.Track.upsert(
        #       trackId=track.track_id,
        #       sensorId=track.sensor_id,
        #       className=track.class_name,
        #       ...
        #   ).execute()
        self._fallback.write_track(track)

    def write_health(self, health: EdgeHealth) -> None:
        if not self._network_up:
            raise ConnectionError("CASK network down (kill-switch)")
        self._fallback.write_health(health)


# --------------------------------------------------------------------------
# Outbox (durable buffer between emit and sink)
# --------------------------------------------------------------------------

class Outbox:
    """SQLite-backed append-only event log with sync state tracking."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS outbox (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id        TEXT UNIQUE NOT NULL,
        kind            TEXT NOT NULL,        -- 'observation' | 'track' | 'health'
        payload_json    TEXT NOT NULL,
        created_at      TEXT NOT NULL,        -- when emit() was called
        synced_at       TEXT,                  -- NULL until sync succeeds
        retry_count     INTEGER NOT NULL DEFAULT 0,
        next_retry_at   TEXT                   -- exponential backoff target
    );
    CREATE INDEX IF NOT EXISTS idx_outbox_unsynced
        ON outbox (synced_at, next_retry_at)
        WHERE synced_at IS NULL;
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")  # better concurrent reads
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def append(self, event_id: str, kind: str, payload: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO outbox
                   (event_id, kind, payload_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (event_id, kind, json.dumps(payload), now_iso()),
            )

    def fetch_unsynced(self, limit: int = 50) -> List[Dict]:
        """Pull next batch of pending events, oldest first."""
        now = now_iso()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT id, event_id, kind, payload_json, created_at, retry_count
                   FROM outbox
                   WHERE synced_at IS NULL
                     AND (next_retry_at IS NULL OR next_retry_at <= ?)
                   ORDER BY id ASC
                   LIMIT ?""",
                (now, limit),
            ).fetchall()
        return [
            {
                "row_id": r[0], "event_id": r[1], "kind": r[2],
                "payload": json.loads(r[3]), "created_at": r[4],
                "retry_count": r[5],
            }
            for r in rows
        ]

    def mark_synced(self, row_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE outbox SET synced_at = ? WHERE id = ?",
                (now_iso(), row_id),
            )

    def mark_retry(self, row_id: int, retry_count: int) -> None:
        # Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 5 minutes
        backoff_s = min(300, 2 ** retry_count)
        next_retry = datetime.now(timezone.utc).timestamp() + backoff_s
        next_retry_iso = datetime.fromtimestamp(next_retry, timezone.utc) \
            .isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE outbox
                   SET retry_count = ?, next_retry_at = ?
                   WHERE id = ?""",
                (retry_count + 1, next_retry_iso, row_id),
            )

    def stats(self) -> Dict[str, int]:
        with self._lock, self._connect() as conn:
            total, synced, pending = conn.execute(
                """SELECT
                       COUNT(*),
                       COUNT(synced_at),
                       SUM(CASE WHEN synced_at IS NULL THEN 1 ELSE 0 END)
                   FROM outbox"""
            ).fetchone()
        return {
            "total": total or 0,
            "synced": synced or 0,
            "pending": pending or 0,
        }


# --------------------------------------------------------------------------
# Sync worker (the loop that drains outbox -> sink)
# --------------------------------------------------------------------------

class SyncWorker(threading.Thread):
    """Background thread. Polls outbox, attempts sync, handles failures."""

    def __init__(self, outbox: Outbox, sink: TelemetrySink,
                 poll_interval_s: float = 1.0, batch_size: int = 50):
        super().__init__(daemon=True, name="cask-sync-worker")
        self.outbox = outbox
        self.sink = sink
        self.poll_interval_s = poll_interval_s
        self.batch_size = batch_size
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                if not self.sink.healthcheck():
                    # Network down — back off but don't drain, keep buffering.
                    time.sleep(self.poll_interval_s * 2)
                    continue

                batch = self.outbox.fetch_unsynced(self.batch_size)
                if not batch:
                    time.sleep(self.poll_interval_s)
                    continue

                for item in batch:
                    if self._stop_event.is_set():
                        break
                    try:
                        self._dispatch(item)
                        self.outbox.mark_synced(item["row_id"])
                    except Exception as e:
                        # Per-event failure: bump retry, keep going
                        self.outbox.mark_retry(item["row_id"], item["retry_count"])
                        # On first failure of a batch, healthcheck again — if
                        # the network just went down, exit early to back off.
                        if not self.sink.healthcheck():
                            break
            except Exception as e:
                # Worker should never die. Log and continue.
                print(f"[sync_worker] unexpected error: {e}")
                time.sleep(self.poll_interval_s * 5)

    def _dispatch(self, item: Dict):
        kind = item["kind"]
        payload = item["payload"]
        if kind == "observation":
            self.sink.write_observation(ObservationEvent(**payload))
        elif kind == "track":
            self.sink.write_track(TrackUpdate(**payload))
        elif kind == "health":
            self.sink.write_health(EdgeHealth(**payload))
        else:
            raise ValueError(f"unknown kind: {kind}")


# --------------------------------------------------------------------------
# Public factory API
# --------------------------------------------------------------------------

class OntologyEventFactory:
    """
    Top-level entry point. Use this from the perception loop and the
    event aggregator.

    Methods are non-blocking with respect to network state — every
    emit() returns after writing to local SQLite. Sync happens async.
    """

    def __init__(self, sensor_id: str, outbox_path: str, sink: TelemetrySink):
        self.sensor_id = sensor_id
        self.outbox = Outbox(outbox_path)
        self.sink = sink
        self._worker: Optional[SyncWorker] = None

    def start(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = SyncWorker(self.outbox, self.sink)
            self._worker.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._worker is not None:
            # Final drain: try to flush anything still in the outbox before exit
            try:
                while True:
                    batch = self.outbox.fetch_unsynced(50)
                    if not batch or not self.sink.healthcheck():
                        break
                    for item in batch:
                        try:
                            self._worker._dispatch(item)
                            self.outbox.mark_synced(item["row_id"])
                        except Exception:
                            self.outbox.mark_retry(item["row_id"], item["retry_count"])
                            break
            except Exception:
                pass
            self._worker.stop()
            self._worker.join(timeout=timeout_s)
            self._worker = None

    # --- emit methods ---

    def emit_observation(self, event_type: str, summary: str,
                         objects: List[Dict] = None, track_ids: List[int] = None,
                         confidence: float = 1.0,
                         timestamp: Optional[str] = None,
                         raw_metadata: Optional[Dict] = None) -> str:
        event = ObservationEvent(
            event_id=str(uuid.uuid4()),
            sensor_id=self.sensor_id,
            event_type=event_type,
            timestamp=timestamp or now_iso(),
            summary=summary,
            confidence=confidence,
            objects=objects or [],
            track_ids=track_ids or [],
            raw_metadata=raw_metadata or {},
        )
        self.outbox.append(event.event_id, "observation", asdict(event))
        return event.event_id

    def emit_track(self, track_id: int, class_name: str, confidence: float,
                   first_seen_at: str, last_seen_at: str,
                   motion_vector: List[float], latest_bbox: List[float]) -> None:
        track = TrackUpdate(
            track_id=track_id,
            sensor_id=self.sensor_id,
            class_name=class_name,
            confidence=confidence,
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
            motion_vector=motion_vector,
            latest_bbox=latest_bbox,
        )
        # Use deterministic event_id per track update so re-emits dedupe
        ev_id = f"track-{track_id}-{last_seen_at}"
        self.outbox.append(ev_id, "track", asdict(track))

    def emit_health(self, camera_ok: bool, inference_fps: float,
                    network_ok: bool, queue_depth: int) -> None:
        health = EdgeHealth(
            sensor_id=self.sensor_id,
            timestamp=now_iso(),
            camera_ok=camera_ok,
            inference_fps=inference_fps,
            network_ok=network_ok,
            queue_depth=queue_depth,
        )
        ev_id = f"health-{self.sensor_id}-{health.timestamp}"
        self.outbox.append(ev_id, "health", asdict(health))

    def stats(self) -> Dict[str, int]:
        return self.outbox.stats()


# --------------------------------------------------------------------------
# Demo / smoke test (simulates kill-switch sequence)
# --------------------------------------------------------------------------

def _demo():
    """
    Simulates the demo arc:
      1. Network up   -> events flow through to sink
      2. Network down -> events buffer locally
      3. Network up   -> backlog drains in order
    """
    print("=== DEMO: kill-switch resilience ===\n")
    sink = CaskOntologySink(jsonl_fallback_path="/tmp/cask_demo.jsonl")
    factory = OntologyEventFactory(
        sensor_id="cask-demo-01",
        outbox_path="/tmp/cask_demo.db",
        sink=sink,
    )
    factory.start()

    print("[1] Network UP — emit 3 observations, expect them to sync immediately")
    for i in range(3):
        factory.emit_observation(
            event_type="track_summary",
            summary=f"3 tanks moving north (event {i})",
            objects=[{"class": "tank", "count": 3, "confidence": 0.85}],
            track_ids=[10 + i, 11 + i, 12 + i],
        )
    time.sleep(2)
    print(f"   stats: {factory.stats()}\n")

    print("[2] Kill-switch ENGAGED — emit 5 observations, expect buffer to grow")
    sink.set_network(False)
    for i in range(5):
        factory.emit_observation(
            event_type="track_summary",
            summary=f"buffered event {i} (network was down)",
            objects=[{"class": "apc", "count": 1}],
        )
    time.sleep(2)
    print(f"   stats: {factory.stats()}")
    print("   (pending should be ~5)\n")

    print("[3] Kill-switch RELEASED — backlog should drain")
    sink.set_network(True)
    time.sleep(3)
    print(f"   stats: {factory.stats()}")
    print("   (pending should be ~0)\n")

    factory.stop()
    print("done. inspect /tmp/cask_demo.db and /tmp/cask_attempted.jsonl")


if __name__ == "__main__":
    _demo()
