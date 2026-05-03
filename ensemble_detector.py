"""
ensemble_detector.py — fused perception layer.

Runs muaythai_sahi (closed-set, clean labels) + grounding_dino (open-vocab,
high recall) on a frame, merges via class-aware NMS, and returns a unified
detection list ready for the tracker / event engine.

Design rules:
  - muaythai_sahi labels WIN when both detectors agree on a box (IoU > 0.5).
    Reasoning: closed-set labels are usable for ontology types; gdino's
    multi-prompt strings are not.
  - gdino-only boxes are kept but labeled "military_object" (generic).
    Better to have a tracked unknown than a missed known.
  - Confidence is the max of the two when fused.
  - Class taxonomy is normalized to a fixed schema so downstream code
    sees stable labels regardless of model upstream.

Usage:
    from ensemble_detector import EnsembleDetector
    det = EnsembleDetector(
        muaythai_weights="path/to/MilitaryConvoy-YOLO11L.pt",
        gdino_id="IDEA-Research/grounding-dino-tiny",
        device="cuda",  # or "mps" / "cpu"
    )
    result = det.detect(frame_bgr, run_gdino=True)
    # result.detections: list of {bbox, label, conf, source}
    # result.gdino_ran: bool (did we run the slow path this call)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
from PIL import Image


# --------------------------------------------------------------------------
# Class taxonomy normalization
# --------------------------------------------------------------------------

# Map raw model labels to our stable ontology classes.
# Anything not in here gets passed through as-is, then bucketed at write time.
LABEL_NORM = {
    # muaythai labels
    "tank":              "tank",
    "artillery":         "artillery",
    "apc":               "apc",
    "afv":               "apc",
    "military-vehicle":  "military_vehicle",
    "military_vehicle":  "military_vehicle",
    "car":               "civilian_vehicle",
    "truck":             "truck",
    # grounding_dino can produce concatenated multi-token strings;
    # we handle those with substring matching below.
}

# Substring rules for gdino's messy multi-phrase outputs.
# Order matters — first match wins.
GDINO_SUBSTRING_RULES = [
    ("tank",                          "tank"),
    ("armored personnel carrier",     "apc"),
    ("apc",                           "apc"),
    ("artillery",                     "artillery"),
    ("military truck",                "truck"),
    ("military vehicle",              "military_vehicle"),
    ("armored vehicle",               "military_vehicle"),
    ("helicopter",                    "helicopter"),
    ("drone",                         "drone"),
    ("soldier",                       "soldier"),
    ("smoke",                         "smoke"),
    ("fire",                          "fire"),
    ("explosion",                     "fire"),
    ("car",                           "civilian_vehicle"),
]


def normalize_label(raw: str, source: str) -> str:
    """Map raw label to stable class. source = 'muaythai' or 'gdino'."""
    raw_lower = raw.lower().strip()
    if raw_lower in LABEL_NORM:
        return LABEL_NORM[raw_lower]
    if source == "gdino":
        for needle, normalized in GDINO_SUBSTRING_RULES:
            if needle in raw_lower:
                return normalized
        # Fallback — gdino caught something but no rule matched
        return "military_object"
    return raw_lower or "unknown"


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: List[float]         # [x1, y1, x2, y2] in pixel coords
    label: str                # normalized class
    conf: float               # 0..1
    source: str               # "muaythai" | "gdino" | "fused"
    raw_labels: List[str] = field(default_factory=list)  # original strings for debugging

    def area(self) -> float:
        return max(0.0, (self.bbox[2] - self.bbox[0])) * max(0.0, (self.bbox[3] - self.bbox[1]))


@dataclass
class DetectionResult:
    detections: List[Detection]
    gdino_ran: bool
    timing_ms: Dict[str, float]


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def class_aware_nms(detections: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
    """Standard NMS but only suppresses within the same class."""
    if not detections:
        return []
    # Group by class
    by_class: Dict[str, List[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.label, []).append(d)
    kept: List[Detection] = []
    for label, group in by_class.items():
        group.sort(key=lambda d: d.conf, reverse=True)
        while group:
            best = group.pop(0)
            kept.append(best)
            group = [d for d in group if iou(best.bbox, d.bbox) < iou_thresh]
    return kept


def fuse_overlapping(muaythai_dets: List[Detection],
                     gdino_dets: List[Detection],
                     iou_thresh: float = 0.5) -> List[Detection]:
    """
    For each muaythai detection, find overlapping gdino detections.
    On match: keep muaythai's label (closed-set is more useful), bump
    confidence to max of both, mark source='fused'.
    Unmatched gdino detections survive as 'gdino' source with normalized
    label; if normalization couldn't infer a class, they become 'military_object'.
    """
    fused: List[Detection] = []
    gdino_matched = [False] * len(gdino_dets)

    for m in muaythai_dets:
        best_iou = 0.0
        best_idx = -1
        for i, g in enumerate(gdino_dets):
            if gdino_matched[i]:
                continue
            score = iou(m.bbox, g.bbox)
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_idx >= 0 and best_iou >= iou_thresh:
            g = gdino_dets[best_idx]
            gdino_matched[best_idx] = True
            fused.append(Detection(
                bbox=m.bbox,
                label=m.label,
                conf=max(m.conf, g.conf),
                source="fused",
                raw_labels=m.raw_labels + g.raw_labels,
            ))
        else:
            fused.append(m)

    # gdino-only survivors
    for i, g in enumerate(gdino_dets):
        if not gdino_matched[i]:
            fused.append(g)

    return fused


# --------------------------------------------------------------------------
# Detector wrappers
# --------------------------------------------------------------------------

class MuaythaiSahiDetector:
    def __init__(self, weights_path: str, device: str = "cuda",
                 conf: float = 0.15, slice_size: int = 640, overlap: float = 0.20):
        from sahi import AutoDetectionModel
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=weights_path,
            confidence_threshold=conf,
            device=device,
        )
        self.slice_size = slice_size
        self.overlap = overlap

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        from sahi.predict import get_sliced_prediction
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        result = get_sliced_prediction(
            pil, self.model,
            slice_height=self.slice_size, slice_width=self.slice_size,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
            verbose=0,
        )
        out = []
        for obj in result.object_prediction_list:
            b = obj.bbox
            raw = obj.category.name
            out.append(Detection(
                bbox=[b.minx, b.miny, b.maxx, b.maxy],
                label=normalize_label(raw, "muaythai"),
                conf=float(obj.score.value),
                source="muaythai",
                raw_labels=[raw],
            ))
        return out


class GroundingDinoDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: str = "cuda",
                 prompt: str = "tank. apc. armored vehicle. military truck. soldier. smoke. fire. drone. helicopter."):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.prompt = prompt

    def detect(self, frame_bgr: np.ndarray,
               box_threshold: float = 0.20,
               text_threshold: float = 0.20) -> List[Detection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, text=self.prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = [pil.size[::-1]]

        # Defensive multi-version handling (transformers API drift)
        results = None
        attempts = [
            lambda: self.processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs.input_ids,
                threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=target_sizes),
            lambda: self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=target_sizes),
        ]
        for fn in attempts:
            try:
                results = fn()
                break
            except (TypeError, AttributeError):
                continue
        if results is None:
            return []

        r = results[0]
        labels_field = r.get("labels", r.get("text_labels", []))
        out = []
        for box, score, label in zip(r["boxes"], r["scores"], labels_field):
            raw = str(label).strip()
            out.append(Detection(
                bbox=box.tolist(),
                label=normalize_label(raw, "gdino"),
                conf=float(score),
                source="gdino",
                raw_labels=[raw],
            ))
        return out


# --------------------------------------------------------------------------
# Ensemble orchestrator
# --------------------------------------------------------------------------

class EnsembleDetector:
    """
    Two-tier perception:
      - muaythai_sahi runs every call (fast-ish closed-set with clean labels)
      - grounding_dino runs every gdino_every_n calls (slow open-vocab recall)

    On gdino-skip frames, returns muaythai-only results.
    On gdino-run frames, returns fused results.
    """

    def __init__(self, muaythai_weights: str, device: str = "cuda",
                 gdino_id: str = "IDEA-Research/grounding-dino-tiny",
                 gdino_prompt: Optional[str] = None,
                 muaythai_conf: float = 0.15,
                 gdino_box_threshold: float = 0.20,
                 gdino_text_threshold: float = 0.20,
                 gdino_every_n: int = 1,
                 nms_iou: float = 0.5,
                 fuse_iou: float = 0.5):
        print(f"[ensemble] loading muaythai_sahi on {device}...")
        self.muaythai = MuaythaiSahiDetector(
            muaythai_weights, device=device, conf=muaythai_conf
        )
        print(f"[ensemble] loading grounding_dino on {device} "
              f"(box_thresh={gdino_box_threshold}, "
              f"text_thresh={gdino_text_threshold})...")
        kwargs = {"model_id": gdino_id, "device": device}
        if gdino_prompt:
            kwargs["prompt"] = gdino_prompt
        self.gdino = GroundingDinoDetector(**kwargs)
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold
        self.gdino_every_n = gdino_every_n
        self.nms_iou = nms_iou
        self.fuse_iou = fuse_iou
        self._call_count = 0

    def detect(self, frame_bgr: np.ndarray,
               run_gdino: Optional[bool] = None) -> DetectionResult:
        self._call_count += 1
        timing = {}

        t0 = time.perf_counter()
        muaythai_dets = self.muaythai.detect(frame_bgr)
        timing["muaythai_ms"] = (time.perf_counter() - t0) * 1000

        # Decide whether to run gdino this call
        if run_gdino is None:
            run_gdino = (self._call_count % self.gdino_every_n) == 0

        if run_gdino:
            t0 = time.perf_counter()
            gdino_dets = self.gdino.detect(
                frame_bgr,
                box_threshold=self.gdino_box_threshold,
                text_threshold=self.gdino_text_threshold,
            )
            timing["gdino_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            fused = fuse_overlapping(muaythai_dets, gdino_dets, self.fuse_iou)
            fused = class_aware_nms(fused, self.nms_iou)
            timing["fuse_ms"] = (time.perf_counter() - t0) * 1000
        else:
            timing["gdino_ms"] = 0.0
            t0 = time.perf_counter()
            fused = class_aware_nms(muaythai_dets, self.nms_iou)
            timing["fuse_ms"] = (time.perf_counter() - t0) * 1000

        timing["total_ms"] = sum(timing.values())
        return DetectionResult(detections=fused, gdino_ran=run_gdino, timing_ms=timing)


# --------------------------------------------------------------------------
# CLI / smoke test
# --------------------------------------------------------------------------

def _draw(image: np.ndarray, dets: List[Detection]) -> np.ndarray:
    color_by_source = {
        "muaythai": (180, 220, 80),    # green-yellow
        "gdino":    (220, 80, 220),    # magenta
        "fused":    (80, 220, 220),    # cyan (the good ones)
    }
    out = image.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.bbox)
        color = color_by_source.get(d.source, (200, 200, 200))
        thickness = 3 if d.source == "fused" else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{d.label} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def _smoke_test():
    """Run on a single frame to verify wiring. Edit paths for your env."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--weights", required=True,
                   help="Path to MilitaryConvoy-YOLO11L .pt weights")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="ensemble_demo.jpg")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    det = EnsembleDetector(
        muaythai_weights=args.weights,
        device=args.device,
        gdino_every_n=1,
    )
    result = det.detect(img, run_gdino=True)

    print(f"\n=== ensemble result ===")
    print(f"total detections: {len(result.detections)}")
    by_source = {}
    by_class = {}
    for d in result.detections:
        by_source[d.source] = by_source.get(d.source, 0) + 1
        by_class[d.label] = by_class.get(d.label, 0) + 1
    print(f"by source: {by_source}")
    print(f"by class : {by_class}")
    print(f"timing   : {result.timing_ms}")

    annotated = _draw(img, result.detections)
    cv2.imwrite(args.out, annotated)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    _smoke_test()
