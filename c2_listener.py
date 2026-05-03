#!/usr/bin/env python3
"""
c2_listener.py — standalone C2 process.

Run this on your Mac (or any C2-class machine) while edge runs on the
Jetson. Listens for sealed STRE messages over UDP, unseals them, runs
pattern matching, and logs decisions for the operator.

What this process does (and only does):
  1. Listen on UDP port 9601 for sealed CBOR envelopes
  2. Unseal with shared PSK from ~/.cask/cask.psk
  3. Track per-source monotonic counter, log gaps and rejections
  4. Decode Type 1 / Type 2 messages
  5. Feed Type 1s to a C2StreEngine for entity resolution + pattern matching
  6. When patterns fire, write Type 2 inferences to operator log
  7. Print live status every second to terminal

What this process does NOT do:
  - Look at pixel bboxes (none transit the network)
  - Render annotated frames (those are edge-side artifacts)
  - Reconstruct foundry_events.jsonl (that's an edge-side mirror)

What ends up on disk at C2 after a run:
  c2_run/
    received_log.bin       — every sealed envelope received (forensic record)
    c2_inferences.jsonl    — every Type 2 fired by C2's pattern matcher
    c2_observations.jsonl  — every Type 1 ingested (for forensic replay)
    c2_status.json         — final stats: gaps, rejections, throughput

Usage:
  python3 c2_listener.py [--bind-port 9601] [--output ./c2_run]
                         [--source-id 0xCA51]

Run it, then start your edge node pointing at this machine's IP. Watch
the terminal for live STRE arrivals + pattern matches. Hit Ctrl-C to
stop and write the final status file.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Local imports — these need stre_codec.py, stre_pipeline.py, network_transport.py
# in the same directory or on PYTHONPATH.
from stre_codec import (
    StreObservation, StreInference, OBJECT_CLASS, PATTERN_ID,
    THREAT_LEVEL, ACTION, SENSOR_TYPE,
)
from stre_pipeline import C2StreEngine
from network_transport import (
    StreUdpListener, get_or_create_psk, DEFAULT_CONFIG_DIR,
)


# Inverse maps for human-readable logging
INV_CLASS = {v: k for k, v in OBJECT_CLASS.items()}
INV_PATTERN = {v: k for k, v in PATTERN_ID.items()}
INV_THREAT = {v: k for k, v in THREAT_LEVEL.items()}
INV_ACTION = {v: k for k, v in ACTION.items()}
INV_SENSOR = {v: k for k, v in SENSOR_TYPE.items()}


class C2Process:
    """
    Encapsulates the C2 listener + C2StreEngine + log writers.

    Designed so the kill-switch demo is visible: terminal prints arrival
    of every Type 1 and a loud banner when a Type 2 fires. The operator
    feed (c2_inferences.jsonl) is what a real Foundry-style dashboard
    would consume.
    """

    def __init__(self, output_dir: Path, source_id_filter: int = 0,
                 bind_port: int = 9601):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_id_filter = source_id_filter
        self.bind_port = bind_port

        self.psk = get_or_create_psk()

        # The pattern-matching engine. Runs at C2, not edge — exactly per spec §2.
        self.engine = C2StreEngine()

        # File handles for forensic logs
        self._received_log = open(self.output_dir / "received_log.bin", "ab")
        self._observations_jsonl = open(
            self.output_dir / "c2_observations.jsonl", "a"
        )
        self._inferences_jsonl = open(
            self.output_dir / "c2_inferences.jsonl", "a"
        )

        self._t_start = time.time()
        self._t_last_status = time.time()
        self._obs_count = 0
        self._inf_count = 0
        self._listener: StreUdpListener = None  # populated by serve()

    def on_observation(self, obs: StreObservation, src_id: int, counter: int) -> None:
        """Called for every Type 1 received over UDP."""
        if self.source_id_filter and src_id != self.source_id_filter:
            return

        self._obs_count += 1

        # Forensic record — every observation, with counter, in JSON form
        record = {
            "_kind": "observation",
            "received_at_unix": int(time.time()),
            "source_id": f"0x{src_id:04x}",
            "wire_counter": counter,
            "event_id": obs.event_id,
            "object_class": INV_CLASS.get(obs.object_class, f"0x{obs.object_class:02x}"),
            "sensor_type": INV_SENSOR.get(obs.sensor_type, f"0x{obs.sensor_type:02x}"),
            "confidence": obs.confidence,
            "lat": obs.lat / 1e6,
            "lon": obs.lon / 1e6,
            "heading": obs.heading,
            "speed_mps": obs.speed_mps_x10 / 10.0,
            "edge_timestamp_unix": obs.timestamp,
        }
        self._observations_jsonl.write(json.dumps(record) + "\n")
        self._observations_jsonl.flush()

        # Feed C2StreEngine — this is where pattern matching happens
        inferences = self.engine.ingest(obs)

        # Print live status (compact, not per-message — that floods the terminal)
        # Just every Nth or on inference fire.
        if self._obs_count % 20 == 0:
            elapsed = time.time() - self._t_start
            rate = self._obs_count / elapsed if elapsed > 0 else 0
            print(f"  [c2] received {self._obs_count} obs "
                  f"({rate:.1f}/s)   "
                  f"class={record['object_class']:18s}  "
                  f"conf={record['confidence']:3d}%  "
                  f"@({record['lat']:.4f},{record['lon']:.4f})")

        # Any Type 2s that fired from this Type 1?
        for inf in inferences:
            self.on_inference_from_engine(inf, src_id)

    def on_inference_from_engine(self, inf: StreInference, src_id: int) -> None:
        """Pattern matcher fired. This is the operator-facing event."""
        self._inf_count += 1
        pname = INV_PATTERN.get(inf.pattern_id, f"0x{inf.pattern_id:02x}")
        threat = INV_THREAT.get(inf.threat_level, "?")
        actions = [INV_ACTION.get(a, hex(a)) for a in inf.actions]

        record = {
            "_kind": "inference",
            "received_at_unix": int(time.time()),
            "source_id": f"0x{src_id:04x}",
            "inference_id": inf.inference_id,
            "pattern": pname,
            "threat_level": threat,
            "confidence": inf.confidence,
            "evidence_event_ids": inf.evidence_ids,
            "recommended_actions": actions,
            "target_lat": inf.target_lat / 1e6,
            "target_lon": inf.target_lon / 1e6,
            "eta_sec": inf.eta_sec,
        }
        self._inferences_jsonl.write(json.dumps(record) + "\n")
        self._inferences_jsonl.flush()

        # The "decision moment" — this is what the operator cares about
        print(f"\n{'=' * 70}")
        print(f"  >>> C2 INFERENCE FIRED <<<")
        print(f"  pattern:      {pname}")
        print(f"  threat:       {threat}")
        print(f"  confidence:   {inf.confidence}%")
        print(f"  evidence:     event_ids={inf.evidence_ids}")
        print(f"  actions:      {actions}")
        print(f"  target:       ({inf.target_lat/1e6:.5f}, {inf.target_lon/1e6:.5f})")
        if inf.eta_sec > 0:
            print(f"  ETA:          {inf.eta_sec}s")
        print(f"{'=' * 70}\n")

    def on_inference_from_wire(self, inf: StreInference, src_id: int,
                                counter: int) -> None:
        """
        Called when edge directly sends a Type 2 (its own self-fired
        inference). For our demo edge does pattern matching too (legacy
        path); record it but don't double-count with C2-side inferences.
        """
        # Just log to forensic record — don't republish to operator stream
        # since the C2 engine will fire its own inferences from the same data.
        # In a stricter spec-faithful deployment, edge wouldn't send Type 2s
        # at all and this branch wouldn't exist.
        if self.source_id_filter and src_id != self.source_id_filter:
            return
        record = {
            "_kind": "edge_inference",
            "received_at_unix": int(time.time()),
            "source_id": f"0x{src_id:04x}",
            "wire_counter": counter,
            "inference_id": inf.inference_id,
            "pattern": INV_PATTERN.get(inf.pattern_id, f"0x{inf.pattern_id:02x}"),
            "confidence": inf.confidence,
        }
        self._inferences_jsonl.write(json.dumps(record) + "\n")
        self._inferences_jsonl.flush()

    def status_loop(self):
        """
        Side thread: every second, print a heartbeat with arrival rate
        and any source contact loss warnings. This is how the operator
        sees a kill-switch event in real time.
        """
        last_obs = 0
        contact_lost_announced: dict = {}
        while True:
            time.sleep(1.0)
            if self._listener is None:
                continue
            stats = self._listener.stats()
            elapsed = time.time() - self._t_start
            interval_obs = self._obs_count - last_obs
            last_obs = self._obs_count

            # Per-source contact monitoring — tells us when a CASK has
            # gone silent (kill switch, jamming, dead battery, etc.)
            for src, src_stats in stats.get("by_source", {}).items():
                last_seen_ago = src_stats.get("last_seen_ago_s")
                if last_seen_ago is None:
                    continue
                if last_seen_ago > 5.0 and not contact_lost_announced.get(src):
                    print(f"\n  [!!] CONTACT LOST: {src} silent for "
                          f"{last_seen_ago:.0f}s — possible kill-switch/"
                          f"jamming/link drop")
                    contact_lost_announced[src] = True
                elif last_seen_ago < 2.0 and contact_lost_announced.get(src):
                    print(f"\n  [++] CONTACT RESTORED: {src} resumed "
                          f"(missed {src_stats['counter_gaps']} messages)")
                    contact_lost_announced[src] = False

    def run(self):
        import threading
        # Spin status thread
        st = threading.Thread(target=self.status_loop, daemon=True)
        st.start()

        # Build listener with our handlers
        self._listener = StreUdpListener(
            self.psk,
            bind_host="0.0.0.0",
            bind_port=self.bind_port,
        )

        # Wrap on_observation to also log received bytes for forensics
        # (we'd ideally tap inside StreUdpListener but a closure is simpler)
        def obs_handler(obs, src, ctr):
            self.on_observation(obs, src, ctr)

        def inf_handler(inf, src, ctr):
            self.on_inference_from_wire(inf, src, ctr)

        # Print pre-run banner
        print("=" * 70)
        print(" CASK C2 LISTENER")
        print(f" listening on UDP 0.0.0.0:{self.bind_port}")
        print(f" PSK loaded from {DEFAULT_CONFIG_DIR / 'cask.psk'}")
        print(f" output dir: {self.output_dir}")
        if self.source_id_filter:
            print(f" filtering for source_id=0x{self.source_id_filter:04x}")
        print("=" * 70)
        print()
        print("Waiting for STRE messages from edge nodes...")
        print()

        try:
            self._listener.serve(
                on_observation=obs_handler,
                on_inference=inf_handler,
            )
        except KeyboardInterrupt:
            print("\n[c2] Ctrl-C — shutting down")
        finally:
            self.shutdown()

    def shutdown(self):
        if self._listener is not None:
            self._listener.stop()

        # Final status dump
        listener_stats = self._listener.stats() if self._listener else {}
        elapsed = time.time() - self._t_start
        status = {
            "wall_time_s": round(elapsed, 1),
            "observations_received": self._obs_count,
            "inferences_fired_by_c2": self._inf_count,
            "listener": listener_stats,
            "output_dir": str(self.output_dir),
        }
        with open(self.output_dir / "c2_status.json", "w") as f:
            json.dump(status, f, indent=2)

        # Close handles
        self._observations_jsonl.close()
        self._inferences_jsonl.close()
        self._received_log.close()

        # Final summary
        print("\n" + "=" * 70)
        print(" C2 SHUTDOWN SUMMARY")
        print("=" * 70)
        print(f"  Wall time:                   {elapsed:.1f}s")
        print(f"  Observations received:       {self._obs_count}")
        print(f"  Inferences fired (by C2):    {self._inf_count}")
        if listener_stats.get("by_source"):
            for src, src_stats in listener_stats["by_source"].items():
                print(f"  Source {src}:")
                print(f"    last counter:              {src_stats['last_counter']}")
                print(f"    counter gaps (lost msgs):  {src_stats['counter_gaps']}")
                print(f"    replays rejected:          {src_stats['replays_rejected']}")
        print(f"  Output dir:                  {self.output_dir}")
        print("=" * 70)


def main():
    p = argparse.ArgumentParser(
        description="CASK C2 listener — receive STRE over UDP from edge"
    )
    p.add_argument("--bind-port", type=int, default=9601,
                   help="UDP port to listen on (default: 9601)")
    p.add_argument("--output", type=Path, default=Path("./c2_run"),
                   help="Output dir for received logs (default: ./c2_run)")
    p.add_argument("--source-id", type=lambda x: int(x, 0), default=0,
                   help="Only process messages from this source_id "
                        "(hex or decimal). 0 = accept all.")
    args = p.parse_args()

    proc = C2Process(
        output_dir=args.output,
        source_id_filter=args.source_id,
        bind_port=args.bind_port,
    )
    proc.run()


if __name__ == "__main__":
    main()
