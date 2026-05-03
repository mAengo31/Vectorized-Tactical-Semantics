# CASK — Connected Adaptive Sensor Kit

A two-tier edge-AI perception system for tactical intelligence. Edge nodes (Jetson Orin Nano + camera) run computer-vision detection on real-time video, encode each detection into a bandwidth-disciplined wire protocol, and stream sealed messages over WiFi/UDP to a Command-and-Control gateway that performs entity resolution and pattern matching.

Built for the **xTech / Cerebral Valley NatSec Hackathon** (May 2-3, 2026, San Francisco), co-hosted with Palantir.

> **Status:** Working end-to-end on Jetson Orin Nano CUDA → Mac C2 listener over WiFi. Live demo with two-tier semantic protocol, kill-switch resilience, persistent shared key, and pattern-matching at C2.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [System architecture](#system-architecture)
- [The wire protocol — STRE Message Format v0.1](#the-wire-protocol--stre-message-format-v01)
- [What runs where](#what-runs-where)
- [Quickstart](#quickstart)
- [Live network demo (Jetson → Mac)](#live-network-demo-jetson--mac)
- [Kill-switch demonstration](#kill-switch-demonstration)
- [File-by-file](#file-by-file)
- [Models used](#models-used)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Known limitations](#known-limitations)
- [Output files reference](#output-files-reference)

---

## Why this exists

Modern tactical operations generate firehoses of sensor data — cameras, radar, signals, telemetry — but the radio links to command-and-control are narrowband. A 9.6 kbps tactical radio (TICN, P-999K class) cannot carry raw video. It cannot even carry verbose JSON. Yet decisions need to be made in real time across distributed sensors.

CASK addresses this with a deliberate two-tier architecture:

1. **Edge nodes do the perception.** Computer vision runs on the Jetson, generating rich detection data (bounding boxes, free-text labels, confidence scores).
2. **Edge nodes ship impoverished but cryptographically-sealed observations** over the bandwidth-disciplined link, using a fixed-class protocol (STRE).
3. **C2 ingests observations from many sensors and performs pattern matching.** Entity resolution, behavioral patterns, and threat inference all happen *at C2*, not at edge.

This separation is the entire architectural insight: edge AI extracts decision-relevant signal from pixels, the protocol carries only what tactical decision-making needs, and the C2 gateway is where multi-sensor fusion happens.

---

## System architecture

```
EDGE (Jetson Orin Nano)                                    C2 (Mac / decision node)
─────────────────────────                                  ──────────────────────────

  USB webcam / video file
         │
         ▼
  EnsembleDetector
    ├── muaythai_sahi (YOLO11-L military detector + SAHI tiling)
    └── grounding_dino (open-vocab VLM, threshold-tunable)
         │
         ▼
  SpatialDeduper (per-frame deduplication on grid + class)
         │
         ▼
  EdgeStreEmitter (rate-limited Type 1 emission per stable id)
         │
         ▼
  CoseSealer (ChaCha20-Poly1305, persistent PSK,
              persistent monotonic counter)
         │
         ├──► wire_log.bin               (LOCAL — always written)
         │
         └──► UDP datagram, port 9601 ────────────────►  StreUdpListener
                                          (over WiFi)         │
                                                              ▼
                                                         CoseSealer.unseal
                                                              │
                                                              ▼
                                                         C2StreEngine
                                                         entity resolution
                                                         pattern matching
                                                              │
                                                              ▼
                                                         Type 2 inference
                                                         (e.g. supply_convoy)
                                                              │
                                                              ▼
                                                         c2_observations.jsonl
                                                         c2_inferences.jsonl

LOCAL EDGE ARTIFACTS (never transit network):
  foundry_events.jsonl   — dashboard mirror with raw labels + pixel bboxes
  vlm_scenes.jsonl       — natural-language captions on Type 2 fire
  outbox.db              — SQLite forensic mirror of foundry events
  annotated/*.jpg        — visual evidence frames
```

**One channel crosses the network: STRE Type 1/2 over UDP.** Pixel bboxes, raw labels, and scene captions stay on the edge as forensic artifacts.

---

## The wire protocol — STRE Message Format v0.1

CASK implements the **Semantic Tactical Reporting Encoding (STRE) v0.1** specification, designed for two-tier edge↔C2 messaging over narrowband tactical radio.

### Type 1 — Observation (~28-34 bytes raw, ~67-72 sealed)
Emitted by edge for each detected entity. Closed-vocabulary integer codes only.

```
event_id (uint16)         — per-source unique
source_id (uint16)        — which CASK
sensor_type (uint8)       — EO/IR/RADAR/SIGINT/...
object_class (uint8)      — fixed enum: drone, vehicle_wheeled,
                            vehicle_tracked, personnel, artillery,
                            aircraft_fixed, aircraft_rotary,
                            missile_launcher, watercraft, structure, ...
lat, lon (int32 × 2)      — micro-degrees
alt_m (int16)             — meters MSL
heading (uint16)          — 0-359 degrees
speed_mps_x10 (uint16)    — speed in 0.1 m/s units
confidence (uint8)        — 0-100
timestamp (uint32)        — unix epoch
```

### Type 2 — Inference (~42 bytes raw, ~80 sealed)
Generated by **C2's pattern matcher**, not edge. Carries the protocol's reasoning conclusions.

```
inference_id (uint16)
pattern_id (uint8)        — fixed enum: atgm_ambush, drone_swarm,
                            recon_then_strike, arty_registration,
                            flanking_maneuver, supply_convoy,
                            retreat_pattern, fpv_attack_run,
                            ew_jamming_sweep, counter_battery
evidence_ids (uint16[])   — up to 4 supporting Type 1 event_ids
threat_level (uint8)      — LOW / MEDIUM / HIGH / CRITICAL
target_lat, target_lon    — int32 × 2
eta_sec (uint16)
confidence (uint8)
actions (uint8[])         — recommended responses (hold/reroute/
                            increase_isr/engage/withdraw/...)
```

### Sealing layer
Every Type 1 and Type 2 is wrapped in a COSE envelope:
- **ChaCha20-Poly1305** AEAD encryption with a 256-bit pre-shared key
- **Monotonic counter** persisted across reboots (anti-replay)
- **Length-prefixed framing** for stream serialization

### Verified properties (from real runs)
- Wire entropy: **7.99 bits/byte** (statistically indistinguishable from random)
- Sealed message size: **66-78 bytes**, average ~68 bytes
- Compression vs verbose JSON: **6.9x smaller**
- Counter monotonicity: verified across 600+ message runs
- Sustained throughput: **3-9 kbps** in real-footage runs (well under 9.6 kbps spec budget)

---

## What runs where

| Role | Hardware | Entry point | Outputs |
|---|---|---|---|
| **Edge** | Jetson Orin Nano (CUDA 12.6, JetPack 6.2) | `main_loop.py` | `wire_log.bin`, `foundry_events.jsonl`, `vlm_scenes.jsonl`, `outbox.db` |
| **C2** | Any Linux/macOS with Python 3.10+ | `c2_listener.py` | `c2_observations.jsonl`, `c2_inferences.jsonl`, `c2_status.json` |

Both processes share `~/.cask/cask.psk` (32-byte symmetric key) distributed out-of-band during setup.

---

## Quickstart

### Prerequisites

**Edge (Jetson)**:
- JetPack 6.2 / CUDA 12.6 / Python 3.10
- PyTorch 2.5.0 (NVIDIA Jetson build) and torchvision 0.20.0 (built from source — see Jetson torchvision compatibility matrix)
- cuSPARSELt 0.6.3.2

**Both sides**:
- `pip install ultralytics sahi transformers cbor2 cryptography opencv-python pillow huggingface_hub`

**Models**:
- Closed-vocab military detector: `MuayThaiLegz/MilitaryConvoy-YOLO11L` — the 49 MB `.pt` file ships in this repo at `models/military_convoy.pt` via **Git LFS**. Make sure you have [`git-lfs`](https://git-lfs.com) installed (`brew install git-lfs && git lfs install`) **before** cloning, or run `git lfs pull` after.
- Open-vocab VLM: `IDEA-Research/grounding-dino-tiny` (auto-downloaded by Transformers on first run)

### Hermetic test (single machine, no network)

```bash
# Synthetic end-to-end smoke test — generates fake detections,
# runs the full pipeline, verifies outputs.
python3 test_integration.py
```

### Edge run on a video file (no live C2)

```bash
python3 main_loop.py videos/sample.mov \
    --weights models/military_convoy.pt \
    --device cuda \
    --output ./run \
    --skip 5 \
    --gdino-every 30 \
    --max-seconds 30 \
    --emit-interval-s 2.0
```

### Inspect the wire log

```bash
python3 inspect_wire.py run/wire_log.bin --psk run/demo_psk.bin --limit 10
```

Decodes sealed messages, prints per-message details, computes wire entropy, verifies counter monotonicity.

---

## Live network demo (Jetson → Mac)

This is the production-mode demo: real edge inference on the Jetson, real network transit to C2 on the Mac.

### One-time setup

**1. Generate the persistent shared key on the Jetson:**

```bash
ssh eg4@<jetson-ip>
cd ~/cask
python3 -c "from network_transport import get_or_create_psk; get_or_create_psk()"
# Prints: [psk] Generated new PSK at /home/eg4/.cask/cask.psk
```

**2. Distribute the key to the Mac (out-of-band):**

```bash
# On Mac:
mkdir -p ~/.cask
scp eg4@<jetson-ip>:~/.cask/cask.psk ~/.cask/cask.psk
chmod 600 ~/.cask/cask.psk

# Verify both sides have identical bytes
md5sum ~/.cask/cask.psk
ssh eg4@<jetson-ip> 'md5sum ~/.cask/cask.psk'
# Both md5 hashes must match
```

### Run the demo

**Terminal 1 — Mac (C2):**

```bash
cd ~/local_documents/natsecHack
python3 c2_listener.py --output ./c2_run
```

You'll see:
```
======================================================================
 CASK C2 LISTENER
 listening on UDP 0.0.0.0:9601
 PSK loaded from /Users/edward/.cask/cask.psk
======================================================================
Waiting for STRE messages from edge nodes...
```

**Terminal 2 — Jetson (edge), in tmux:**

```bash
ssh eg4@<jetson-ip>
tmux new -s edge
source ~/cask-venv/bin/activate
cd ~/cask

python3 main_loop.py 0 --webcam \
    --weights models/military_convoy.pt \
    --device cuda \
    --output ./live_run \
    --skip 2 \
    --gdino-every 10 \
    --gdino-box-threshold 0.18 \
    --gdino-text-threshold 0.18 \
    --c2-host <YOUR-MAC-IP> \
    --c2-port 9601 \
    --no-annotated
```

For a live monitor view on a Jetson-attached HDMI display, add `--show-window --fullscreen` and run with `export DISPLAY=:0` from the SSH session, or run from a local Jetson terminal.

### What you'll see

**On the Mac (C2):**

```
[c2] received 20 obs (0.1/s)   class=vehicle_tracked     conf= 73%  @(48.1584,37.7261)
[c2] received 40 obs (0.3/s)   class=vehicle_wheeled     conf= 45%  @(48.1585,37.7268)
[c2] received 60 obs (0.5/s)   class=vehicle_tracked     conf= 60%  @(48.1587,37.7263)
...

======================================================================
  >>> C2 INFERENCE FIRED <<<
  pattern:      supply_convoy
  threat:       HIGH
  confidence:   95%
  evidence:     event_ids=[237, 238, 239, 240]
  actions:      ['increase_isr', 'reroute']
  target:       (48.15862, 37.72654)
======================================================================
```

This is **C2 making a tactical decision from semantic events received over the wire**. The pattern matching ran on the Mac, fed by Type 1 observations sealed and shipped from the Jetson.

---

## Kill-switch demonstration

To demonstrate network-loss resilience to judges:

**1.** Start the demo as above. Confirm `[c2] received N obs` is flowing.

**2.** From a third terminal on Mac, block UDP traffic:

```bash
sudo pfctl -e 2>/dev/null
echo "block out proto udp from any to any port 9601" | sudo pfctl -ef -
```

**3.** Within ~5 seconds, the C2 terminal prints:

```
[!!] CONTACT LOST: 0xca51 silent for 5s — possible kill-switch/jamming/link drop
```

The Jetson keeps detecting and writing to `wire_log.bin` locally. UDP packets blackhole. Counter advances normally on edge.

**4.** Release the block:

```bash
sudo pfctl -F all -f /etc/pf.conf
```

**5.** C2 logs a counter gap and resumes ingestion:

```
[c2-listener] counter gap from src=0xca51: missed 47 messages (last=198, now=246)
[++] CONTACT RESTORED: 0xca51 resumed (missed 47 messages)
```

**Why no replay?** Type 1 observations describe real-time positions. A 30-second-old observation is *worse* than a missed one — it implies the operator is looking at stale truth. Per the spec design, edge does not retransmit dropped Type 1s.

---

## File-by-file

| File | Lines | Purpose |
|---|---|---|
| `main_loop.py` | 822 | Edge orchestrator. CLI, video/webcam capture, perception, tracking, STRE emission, network fan-out, annotation. |
| `c2_listener.py` | 350 | Standalone C2 process. Listens for STRE UDP, unseals, runs C2StreEngine, prints operator events, writes log files. |
| `network_transport.py` | 544 | UDP transmitter/listener, persistent PSK loader, `NetworkAwareStreSink` wrapper. Counter-gap detection on the receive side. |
| `stre_codec.py` | 608 | STRE Type 1 / Type 2 dataclasses with CBOR encode/decode. COSE sealer (ChaCha20-Poly1305 + monotonic counter). Internal-label-to-STRE-class bridge. |
| `stre_pipeline.py` | 619 | `EdgeStreEmitter` (rate-limited per-track Type 1 emission, synthetic geo-rectification) and `C2StreEngine` (entity resolution + pattern matchers for `supply_convoy` and others). |
| `ensemble_detector.py` | 435 | Two-detector ensemble: muaythai_sahi (YOLO11-L + SAHI tiling) + grounding_dino-tiny (open-vocab VLM). Class-aware NMS + match fusion. |
| `ontology_factory.py` | 574 | Foundry-side dashboard event factory. Emits track and observation events. SQLite outbox for kill-switch resilient sync. |
| `vlm_channel.py` | 155 | Natural-language scene captions on Type 2 fire. `vlm_scenes.jsonl` writer. Local-only artifact, never transits network. |
| `inspect_wire.py` | 170 | CLI tool to decode `wire_log.bin` offline. Per-message detail, wire entropy, counter check. |
| `test_integration.py` | 139 | End-to-end synthetic test: builds fake detections, runs the full pipeline, asserts outputs. |
| `hazard_detector.py` | 585 | Temporal hazard detection (explosions/fires/strikes). Self-contained, not currently wired into main_loop. Future work. |

---

## Models used

### Closed-vocabulary military detector
- **HuggingFace repo**: [`MuayThaiLegz/MilitaryConvoy-YOLO11L`](https://huggingface.co/MuayThaiLegz/MilitaryConvoy-YOLO11L)
- **Shipped in repo**: `models/military_convoy.pt` (49 MB, tracked via [Git LFS](https://git-lfs.com)). Clone with `git-lfs` installed or run `git lfs pull` to fetch the actual blob; otherwise you'll get a 134-byte pointer file.
- **Architecture**: YOLO11-Large (Ultralytics, October 2024 release)
- **Loaded via**: SAHI (`AutoDetectionModel.from_pretrained`, `model_type='ultralytics'`) for tile-based inference on high-resolution aerial frames
- **Output classes**: `tank`, `apc`, `artillery`, `military_vehicle`, `civilian_vehicle`, `military_object`
- **Default confidence threshold**: 0.15 (tunable via `--muaythai-conf`)

> Note: this model is YOLO11-L, not YOLOv8. They share inference code from Ultralytics but have different layer counts and head designs. YOLO11 was released October 2024 as the successor to YOLOv8.

### Open-vocabulary VLM detector
- **HuggingFace repo**: `IDEA-Research/grounding-dino-tiny` (the *tiny* variant, not base)
- **Loaded via**: HuggingFace Transformers (`AutoProcessor` + `AutoModelForZeroShotObjectDetection`)
- **Default text prompt**:
  ```
  tank. apc. armored vehicle. military truck. soldier. smoke. fire. drone. helicopter.
  ```
- **Default thresholds**: `box_threshold=0.20`, `text_threshold=0.20`
- **Tunable via**: `--gdino-box-threshold`, `--gdino-text-threshold`, `--gdino-prompt`

### Ensemble logic
- Frame-skip cadence: muaythai_sahi every processed frame, grounding_dino every Nth frame (`--gdino-every`, default 10)
- Match fusion: when muaythai and grounding_dino detect overlapping bboxes (IoU > 0.5), muaythai's clean closed-vocab label takes precedence; the detection's source becomes `'fused'`
- Unmatched grounding_dino detections survive as `source='gdino'` with their open-vocab label
- Class-aware NMS suppresses within-class duplicates

---

## Design decisions worth knowing

### 1. C2 makes decisions, edge does perception
Pattern matching, entity resolution, and threat inference run *at C2*, not edge. Edge ships only Type 1 observations. This is not a performance choice — it's an architectural one. Multi-sensor fusion is impossible if each edge node draws its own conclusions; only a centralized C2 sees the union of observations from many CASKs.

### 2. STRE is deliberately impoverished
The 18-class object enum collapses model labels: `tank`, `apc`, `BMP-3` all become `vehicle_tracked` on the wire. This is the spec's idea of *decision-relevant universal vocabulary*. Edge knows it's a T-72; C2 only needs to know it's a tracked vehicle. Bandwidth-disciplined by design. Raw labels are preserved in `foundry_events.jsonl` as forensic artifacts.

### 3. UDP, not TCP
Tactical radio is lossy. TCP retransmits old observations, which is *worse* than dropping them — a 5-second-old position implies the operator is looking at stale truth. Edge fires Type 1s and forgets them. C2 detects gaps via the counter and logs them. No retry logic.

### 4. One channel only
We considered a second TCP channel for "rich" data (pixel bboxes, scene captions). We rejected it. C2 makes decisions from semantic events, not pixels. If the rich data were on the decision path, the system would have already failed — the edge was supposed to do that work. Rich data is a forensic artifact, retrieved out-of-band post-mission.

### 5. Persistent PSK + counter
The pre-shared key lives in `~/.cask/cask.psk` and is the same across runs. The anti-replay counter persists in `~/.cask/cask.counter` across reboots. Without persistence, an adversary who recorded `wire_log.bin` from a prior run could replay it. With persistence, replays are rejected.

### 6. Local artifacts are unconditional
Edge writes `wire_log.bin` and `foundry_events.jsonl` *before* attempting network transmission. If the network is down, perception data is preserved locally and can be retrieved post-mission. Network state never affects local state.

### 7. SAHI for high-resolution tiling
Aerial drone footage is often 4K. Stock YOLO at 640×640 misses small objects after downsampling. SAHI splits the image into overlapping 640×640 tiles, runs detection on each, and merges results — at the cost of ~6x inference time. For aerial work, this trade-off is necessary.

---

## Known limitations

- **`hazard_detector.py` is not wired in.** Temporal event detection (explosions, fires, debris fields) is implemented but not currently invoked by `main_loop.py`. Future work.
- **Pattern matching is rule-based, not learned.** `supply_convoy` fires on heuristics: ≥3 vehicles in coherent motion within a 60-second window. Real systems would use temporal HMM or transformer-based pattern recognition.
- **Geo-rectification is synthetic.** Pixel→lat/lon mapping is a fake homography in `EdgeStreEmitter._pixels_to_world`. A real CASK has a calibrated camera + IMU + GPS for true geographic projection.
- **C2StreEngine is single-source.** Pattern matching currently buckets observations from one CASK. Multi-CASK fusion (the architectural endgame) requires entity resolution across distinct sensor frames — implemented in scaffold but not stress-tested.
- **No cert-based auth.** PSK distribution is manual via `scp`. Production would use mutual-TLS certs with a CA, key rotation, and HSM-backed key storage.
- **Pattern thresholds are hand-tuned.** `MIN_VEHICLES=3`, `BEARING_TOLERANCE=45°`, `WINDOW=60s` for `supply_convoy`. Production would learn these from labeled deployment data.

---

## Output files reference

After a run, the edge node's `output/` directory contains:

| File | Contents |
|---|---|
| `wire_log.bin` | Length-prefixed sealed CBOR envelopes. Every Type 1 and Type 2 emitted. Decode with `inspect_wire.py`. |
| `foundry_events.jsonl` | Dashboard mirror: tracks (with pixel bboxes, raw labels, motion vectors) and inferences (with full evidence chains). One JSON per line. |
| `vlm_scenes.jsonl` | Natural-language captions on Type 2 fire. Sparse — only fires when patterns match. |
| `outbox.db` | SQLite WAL journal of dashboard events. Each row has `synced_at` for kill-switch buffering semantics. |
| `stre_counter.txt` | Last-emitted counter (legacy per-run path). Persistent counter lives at `~/.cask/cask.counter`. |
| `demo_psk.bin` | Per-run PSK (legacy, ephemeral mode). Persistent PSK lives at `~/.cask/cask.psk`. |
| `run_summary.md` | Human-readable run statistics. |
| `run_stats.json` | Programmatic run statistics. |
| `annotated/frame_*.jpg` | Per-frame visual evidence with bbox overlays + HUD. Disable with `--no-annotated`. |

After a C2 run, `c2_run/` contains:

| File | Contents |
|---|---|
| `c2_observations.jsonl` | Every Type 1 received over the wire, decoded, with original counter and source. |
| `c2_inferences.jsonl` | Every Type 2 fired by C2's pattern matcher, with evidence chains. |
| `c2_status.json` | Final stats: counter gaps, replays rejected, last-seen-per-source. |

---

## Acknowledgments

- **STRE Message Format v0.1** — the wire-protocol spec published for the hackathon.
- **Palantir** — co-host, ontology integration concepts.
- **Brave1 Dataroom** — Ukraine MoD frontline AI data initiative; informed the kill-switch resilience design.
- **MuayThaiLegz** for the `MilitaryConvoy-YOLO11L` weights on HuggingFace.
- **IDEA-Research** for `grounding-dino-tiny`.
- **CMU "DensePose From WiFi"** team — separate work, but shaped the through-wall sensing thinking that informed the network-loss handling here.

---

## License

Demo / hackathon code. No license specified. Contact the author before reuse.
