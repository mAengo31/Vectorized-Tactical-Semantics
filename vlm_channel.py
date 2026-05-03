"""
vlm_channel.py — high-bandwidth open-vocabulary channel.

STRE Type 1 ships over 9.6 kbps tactical radio: closed-vocabulary integer
codes, ~67B sealed per message. Designed for always-on operation but
deliberately impoverished — drops everything our perception models actually
see beyond a hardcoded enum of 18 object classes.

This module is the parallel rich channel. In production it would run over a
higher-bandwidth path (satcom burst, mesh proximity, store-and-forward IP
backhaul). It's not subject to the 9.6 kbps constraint, so we can ship:

  - Open-vocabulary labels from grounding_dino ('truck towing artillery
    piece', 'soldiers crouching beside vehicle')
  - Per-detection bounding boxes in pixel space, source model attribution,
    confidence breakdowns
  - Frame-level summaries: counts by class, scene composition stats
  - On-fire VLM scene captions (when an STRE Type 2 inference fires, capture
    a richer description of the supporting frame as evidence)

Two key design decisions:

  1. **Cross-link reconciliation by ID.** Every rich-channel message carries
     the same source_id and timestamps as the matching STRE messages, plus
     event_ids for the per-detection records. C2 can join across channels
     to enrich a Type 2 inference with the rich captions of its evidence.

  2. **Schema is JSON, not CBOR.** We don't need wire compactness here —
     this channel runs on links where bandwidth is plentiful. JSON is
     human-readable, extensible, and matches what existing C2 dashboards
     expect (it's what your Foundry side already consumes).

Sinks below mirror the StreSink pattern: they implement TelemetrySink-style
methods so OntologyEventFactory can fan out to them via MultiSink.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


# ============================================================================
# Schema
# ============================================================================

@dataclass
class RichDetection:
    """One detection from the perception layer, with everything STRE drops.

    Fields not in STRE Type 1:
      - free-text label (open-vocabulary VLM output)
      - bbox in pixel space (STRE only carries lat/lon)
      - source model attribution (which detector fired)
      - frame_idx (for cross-referencing with annotated frames)
    """
    event_id: int                 # matches the STRE Type 1 event_id when bridged
    frame_idx: int
    source_id: int                # CASK device id
    timestamp_unix: int

    # Open-vocab semantic content — the stuff STRE throws away
    label: str                    # 'tank', 'truck towing artillery', 'soldiers crouching'
    label_normalized: str         # internal canonical form (e.g. 'tank' → 'tank')
    source_model: str             # 'muaythai' | 'grounding_dino' | other

    # Pixel-space data — STRE has no fields for this
    bbox_xyxy: List[float]        # [x1, y1, x2, y2] in source frame coords
    frame_width: int
    frame_height: int

    # Confidence and any model-specific signal
    confidence: float             # 0.0 - 1.0
    is_in_stre: bool              # True if this detection also got a Type 1 emitted


@dataclass
class FrameDigest:
    """Per-frame summary. Cheap to ship, high information density.

    Use case: a C2 analyst scrubbing through a long observation period wants
    to know "how many things, of what types, in this minute" without
    reconstructing per-detection data. This digest gives them that at one
    message per processed frame.
    """
    frame_idx: int
    source_id: int
    timestamp_unix: int

    detection_counts_by_label: Dict[str, int]   # {'tank': 3, 'truck': 2, ...}
    detection_counts_by_source: Dict[str, int]  # {'muaythai': 4, 'grounding_dino': 7}
    total_detections: int
    mean_confidence: float

    # How much of the perception stream is making it into STRE? Useful for
    # spotting silently-dropped detections.
    stre_emitted_count: int       # Type 1s emitted from this frame
    stre_drop_rate: float         # 1 - (stre_emitted / total)


@dataclass
class SceneCaption:
    """High-value burst: when a Type 2 fires, capture richer context."""
    inference_id: int             # links back to the STRE Type 2 inference_id
    frame_idx: int
    source_id: int
    timestamp_unix: int

    # The free-text labels grounding_dino assigned in the trigger frame
    open_vocab_labels: List[str]
    # Aggregate stats: "5 vehicles, 2 of which are tracked"
    composition: Dict[str, int]
    # Pattern that triggered this caption
    pattern_name: str
    # The natural-language scene description (in production: VLM-generated;
    # for demo: composed from labels + counts)
    description: str


# ============================================================================
# Sink
# ============================================================================

class VlmChannelSink:
    """Writes rich-channel records to JSONL files alongside the STRE wire log.

    Three streams:
      detections.jsonl   — one line per RichDetection
      frames.jsonl       — one line per FrameDigest
      scenes.jsonl       — one line per SceneCaption (sparse, on Type 2 fire)

    All lines are pure JSON. The C2 dashboard or any analyst tool can stream
    them without requiring a CBOR decoder. This is the demo-readable channel.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detections_path = self.output_dir / "vlm_detections.jsonl"
        self.frames_path = self.output_dir / "vlm_frames.jsonl"
        self.scenes_path = self.output_dir / "vlm_scenes.jsonl"
        self._lock = threading.Lock()
        self._stats = {
            "detections_written": 0,
            "frames_written": 0,
            "scenes_written": 0,
            "bytes_written": 0,
        }

    def _append(self, path: Path, obj: Any) -> None:
        line = json.dumps(obj, default=str, separators=(",", ":")) + "\n"
        with self._lock, open(path, "a") as f:
            f.write(line)
        self._stats["bytes_written"] += len(line)

    def write_detection(self, det: RichDetection) -> None:
        self._append(self.detections_path, asdict(det))
        self._stats["detections_written"] += 1

    def write_frame_digest(self, dig: FrameDigest) -> None:
        self._append(self.frames_path, asdict(dig))
        self._stats["frames_written"] += 1

    def write_scene_caption(self, scene: SceneCaption) -> None:
        self._append(self.scenes_path, asdict(scene))
        self._stats["scenes_written"] += 1

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ============================================================================
# Helpers — build the rich records from raw detector output
# ============================================================================

def normalize_label(raw: str) -> str:
    """
    Map any free-text or closed-vocab label to a coarse canonical form.
    Used only for FrameDigest counting; the original `label` text is
    preserved on the detection itself.
    """
    text = raw.lower().strip()
    if any(w in text for w in ["tank", "tracked", "ifv", "bmp"]):
        return "tracked_vehicle"
    if any(w in text for w in ["apc", "armored personnel"]):
        return "tracked_vehicle"
    if any(w in text for w in ["truck", "wheeled", "humvee", "lav", "transporter"]):
        return "wheeled_vehicle"
    if any(w in text for w in ["artillery", "howitzer", "mortar", "cannon"]):
        return "artillery"
    if any(w in text for w in ["soldier", "infantry", "person", "personnel", "troop"]):
        return "personnel"
    if any(w in text for w in ["drone", "uav", "quadcopter", "unmanned aerial"]):
        return "drone"
    if any(w in text for w in ["helicopter", "rotorcraft", "rotary"]):
        return "rotary_aircraft"
    if any(w in text for w in ["military_vehicle", "military vehicle", "vehicle"]):
        return "vehicle_other"
    return "unknown"


def compose_scene_description(labels: List[str], composition: Dict[str, int],
                              pattern_name: str) -> str:
    """
    Demo-grade caption. In production this would be a VLM call (Florence-2,
    PaliGemma, Qwen2-VL) on the trigger frame. For the hackathon we compose
    deterministically from the detection inventory.
    """
    parts = []
    if "tracked_vehicle" in composition:
        n = composition["tracked_vehicle"]
        parts.append(f"{n} tracked armored vehicle{'s' if n != 1 else ''}")
    if "wheeled_vehicle" in composition:
        n = composition["wheeled_vehicle"]
        parts.append(f"{n} wheeled vehicle{'s' if n != 1 else ''}")
    if "personnel" in composition:
        n = composition["personnel"]
        parts.append(f"{n} dismounted personnel")
    if "artillery" in composition:
        parts.append(f"{composition['artillery']} artillery position(s)")
    if "vehicle_other" in composition:
        parts.append(f"{composition['vehicle_other']} unspecified vehicle(s)")

    inventory = ", ".join(parts) if parts else "no classified entities"
    distinctive = [l for l in labels if len(l.split()) > 1][:3]
    distinctive_clause = ""
    if distinctive:
        distinctive_clause = f" Distinctive observations: {'; '.join(distinctive)}."

    return (f"Scene contains {inventory}. Pattern '{pattern_name}' triggered "
            f"on this frame.{distinctive_clause}")


# ============================================================================
# Smoke test
# ============================================================================

def _smoke_test():
    import os
    print("=== VLM channel smoke test ===\n")

    out_dir = "/tmp/vlm_test"
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    sink = VlmChannelSink(out_dir)

    base_t = int(time.time())

    # Simulate 5 detections: 3 from muaythai (clean labels), 2 from grounding_dino
    # (rich free-text). All 5 from the same frame.
    detections = [
        RichDetection(
            event_id=101, frame_idx=18, source_id=0xCA51, timestamp_unix=base_t,
            label="tank", label_normalized=normalize_label("tank"),
            source_model="muaythai",
            bbox_xyxy=[820, 540, 920, 620], frame_width=1280, frame_height=720,
            confidence=0.78, is_in_stre=True,
        ),
        RichDetection(
            event_id=102, frame_idx=18, source_id=0xCA51, timestamp_unix=base_t,
            label="apc", label_normalized=normalize_label("apc"),
            source_model="muaythai",
            bbox_xyxy=[680, 555, 770, 615], frame_width=1280, frame_height=720,
            confidence=0.72, is_in_stre=True,
        ),
        RichDetection(
            event_id=103, frame_idx=18, source_id=0xCA51, timestamp_unix=base_t,
            label="truck", label_normalized=normalize_label("truck"),
            source_model="muaythai",
            bbox_xyxy=[450, 550, 540, 610], frame_width=1280, frame_height=720,
            confidence=0.65, is_in_stre=True,
        ),
        RichDetection(
            event_id=0, frame_idx=18, source_id=0xCA51, timestamp_unix=base_t,
            label="armored personnel carrier with mounted heavy weapon system",
            label_normalized=normalize_label("armored personnel carrier with mounted heavy weapon system"),
            source_model="grounding_dino",
            bbox_xyxy=[680, 555, 770, 615], frame_width=1280, frame_height=720,
            confidence=0.61,
            # Note: is_in_stre=False — the bridge dropped this label.
            is_in_stre=False,
        ),
        RichDetection(
            event_id=0, frame_idx=18, source_id=0xCA51, timestamp_unix=base_t,
            label="dismounted soldiers crouching behind vehicle",
            label_normalized=normalize_label("dismounted soldiers crouching behind vehicle"),
            source_model="grounding_dino",
            bbox_xyxy=[640, 580, 700, 660], frame_width=1280, frame_height=720,
            confidence=0.54,
            is_in_stre=False,
        ),
    ]
    for d in detections:
        sink.write_detection(d)

    # Frame digest
    by_label: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    stre_count = 0
    for d in detections:
        by_label[d.label_normalized] = by_label.get(d.label_normalized, 0) + 1
        by_source[d.source_model] = by_source.get(d.source_model, 0) + 1
        if d.is_in_stre:
            stre_count += 1

    digest = FrameDigest(
        frame_idx=18, source_id=0xCA51, timestamp_unix=base_t,
        detection_counts_by_label=by_label,
        detection_counts_by_source=by_source,
        total_detections=len(detections),
        mean_confidence=sum(d.confidence for d in detections) / len(detections),
        stre_emitted_count=stre_count,
        stre_drop_rate=1 - stre_count / len(detections),
    )
    sink.write_frame_digest(digest)

    # Scene caption (fired because supply_convoy hit on this frame)
    labels_seen = [d.label for d in detections]
    composition = dict(by_label)
    description = compose_scene_description(labels_seen, composition, "supply_convoy")
    scene = SceneCaption(
        inference_id=4242, frame_idx=18, source_id=0xCA51, timestamp_unix=base_t,
        open_vocab_labels=labels_seen,
        composition=composition,
        pattern_name="supply_convoy",
        description=description,
    )
    sink.write_scene_caption(scene)

    print(f"[1] Wrote {sink.stats()}\n")

    # Read back and pretty-print one of each kind
    print("[2] Sample VLM detection (rich open-vocab record):")
    line = open(sink.detections_path).readlines()[3]   # the grounding_dino one
    print(f"    {json.dumps(json.loads(line), indent=2)[:500]}...\n")

    print("[3] Frame digest:")
    line = open(sink.frames_path).read().strip()
    print(f"    {json.dumps(json.loads(line), indent=2)}\n")

    print("[4] Scene caption:")
    line = open(sink.scenes_path).read().strip()
    parsed = json.loads(line)
    print(f"    inference_id: {parsed['inference_id']}")
    print(f"    pattern: {parsed['pattern_name']}")
    print(f"    description: {parsed['description']}")
    print(f"    composition: {parsed['composition']}")
    print(f"    rich labels seen: {parsed['open_vocab_labels']}\n")

    print("[5] What this gives you that STRE doesn't:")
    print("    - Free-text labels from grounding_dino preserved in full")
    print(f"      (e.g. {detections[3].label!r})")
    print("    - Pixel-space bboxes for replay/visual verification")
    print("    - Per-frame counts and STRE drop-rate diagnostics")
    print(f"      (this frame had {len(detections)} detections, {stre_count} made it to wire)")
    print("    - Source-model attribution (which detector caught what)")
    print("    - Scene-level natural-language caption on Type 2 fire")


if __name__ == "__main__":
    _smoke_test()
