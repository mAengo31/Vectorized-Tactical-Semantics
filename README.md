# Vectorized Tactical Semantics (CASK)

Edge-to-C2 perception pipeline for contested, low-bandwidth tactical networks.

A camera + Jetson on the edge runs an ensemble detector (YOLO + Grounding DINO),
turns tracks into a fixed-schema binary protocol (**STRE** — Sensor-to-Tactical
Radio Encoding), seals each message with COSE / ChaCha20-Poly1305, and ships it
over a 9.6 kbps UDP link to a C2 listener that performs entity resolution,
pattern matching, and emits Type 2 inferences to an operator dashboard.

A second, higher-bandwidth **VLM channel** carries the rich, open-vocabulary
semantics (captions, per-detection bboxes, model attribution) on the side —
joinable to STRE events by `source_id` + timestamps when the link allows.

## Architecture

```
EDGE (Jetson)                                       C2 GATEWAY
──────────────                                      ──────────
video frames
     │
     ▼
EnsembleDetector  (YOLO closed-set + Grounding DINO open-vocab)
     │
     ▼
SimpleIouTracker  (ByteTrack stub)
     │
     ├──► EdgeStreEmitter ──── Type 1 (sealed CBOR, ~67B) ──UDP──► StreUdpListener
     │                                                                  │
     │                                                                  ▼
     │                                                            C2StreEngine
     │                                                              │     │
     │                                                              │     ├─ entity resolution
     │                                                              │     ├─ pattern matching
     │                                                              ▼     ▼
     │                                                          Type 2 inferences
     │                                                                  │
     ├──► VlmChannelSink ───── rich JSON (high-BW side channel) ───────┤
     │                                                                  ▼
     └──► OntologyEventFactory ──► JsonlSink / CASK Foundry sink ──► dashboard
                                  (SQLite outbox, kill-switch resilient)
```

## Wire discipline

- **Type 1 (Observation)** — fixed-schema CBOR array, ~28 B raw, ~67 B sealed.
- **Type 2 (Inference)** — fixed-schema CBOR array, ~42 B raw.
- All payloads are **integer codes**, not strings. Class taxonomy, pattern IDs,
  threat levels, sensor types: all enums (see `stre_codec.py`).
- Lat/lon: int32 with 1e6 multiplier (~11 cm precision globally).
- Timestamp: uint32 unix seconds (good until 2106).
- Anti-replay: per-source monotonic counter, persisted to disk so it survives
  restarts. Counter gaps at C2 are logged, not retransmitted (a 5-second-old
  Type 1 is worse than no Type 1).

## Files

| File | Role |
|---|---|
| `main_loop.py` | Edge orchestrator: video → detection → tracks → STRE + Foundry events |
| `c2_listener.py` | Standalone C2 process: UDP receive → unseal → pattern matching |
| `ensemble_detector.py` | YOLO + Grounding DINO fused perception with class-aware NMS |
| `stre_codec.py` | STRE v0.1 encoder/decoder + COSE sealer (ChaCha20-Poly1305) |
| `stre_pipeline.py` | `EdgeStreEmitter` (rate-limited Type 1 generation) + `C2StreEngine` |
| `network_transport.py` | UDP transmitter/listener, PSK + counter persistence |
| `vlm_channel.py` | High-bandwidth side channel (open-vocab labels, bboxes, captions) |
| `ontology_factory.py` | SQLite outbox + sink fan-out (JSONL today, CASK/Foundry tomorrow) |
| `inspect_wire.py` | Forensic decoder for `wire_log.bin` |
| `test_integration.py` | End-to-end smoke tests |

## Quick start

### 1. Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install opencv-python numpy cbor2 cryptography ultralytics transformers torch pillow
```

Drop a YOLO weights file at `models/military_convoy.pt` (not committed; bring
your own).

### 2. Run the edge demo (local, no network)

```bash
python main_loop.py path/to/footage.mp4 --output ./demo_run --skip 3 --max-seconds 30
```

Outputs:

```
demo_run/
  annotated/frame_NNNN.jpg     # viz frames
  wire_log.bin                  # sealed CBOR over-the-wire bytes
  foundry_events.jsonl          # what the dashboard sees
  run_summary.md                # final stats
```

### 3. Run the full edge ↔ C2 split

On the C2 machine (laptop):

```bash
python c2_listener.py --bind-port 9601 --output ./c2_run
```

On the edge (same network):

```bash
python main_loop.py footage.mp4 --c2-host <C2-IP> --c2-port 9601
```

The shared PSK at `~/.cask/cask.psk` is auto-generated on first run; copy it to
the C2 machine before starting.

### 4. Forensic inspection

```bash
python inspect_wire.py demo_run/wire_log.bin
```

## Design notes

- **One channel, two sinks.** STRE is the only thing that crosses the tactical
  radio. The VLM rich channel runs on a separate (assumed higher-bandwidth)
  link. C2 makes decisions from STRE alone; rich-channel data only enriches
  forensic playback and dashboard captions.
- **Edge owns perception, C2 owns inference.** The edge does not pattern-match
  or fuse — it ships observations. Entity resolution and Type 2 inference happen
  at C2. This keeps the edge stateless across restarts and lets a single C2
  fuse multiple sensors.
- **Kill-switch resilient.** Edge keeps sealing and writing `wire_log.bin`
  locally regardless of network state. Counter advances during outages. C2 logs
  the gap on recovery and resumes; nothing is replayed.
- **No strings on the wire.** Class taxonomy, pattern IDs, action codes — all
  integers. The dashboard reconstructs human-readable text from the codes.

## Status

Hackathon prototype. The CASK / Foundry sink is stubbed (`JsonlSink` is the
working path); ByteTrack is a `SimpleIouTracker` shim; the real model weights
are not in-repo.
