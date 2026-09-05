#!/usr/bin/env python3
"""Audit FARM associations using prompt-free DINOv3 crop consistency.

The stage embeds several saved views of every active object with the same
DINOv3 ViT-S+/16 family used for FARM association.  It derives an outlier gate
from the scene's own similarity distribution and writes a filtered copy of the
presentation state.  Categories and captions are recorded only for reporting;
they are never inputs to the gate.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def _numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _resolve_observation_path(row: dict, mask_dir: Path) -> Path | None:
    raw = Path(str(row.get("path") or ""))
    candidates = [raw]
    if raw.name:
        candidates.append(mask_dir / raw.parent.name / raw.name)
    return next((path for path in candidates if path.is_file()), None)


def _quality(row: dict) -> float:
    detection = max(float(row.get("score") or 0.0), 1.0e-3)
    raw_pixels = max(int(row.get("raw_pixels") or 0), 1)
    inlier_pixels = max(int(row.get("inlier_pixels") or 0), 0)
    inlier_ratio = float(np.clip(inlier_pixels / raw_pixels, 0.0, 1.0))
    return detection * math.sqrt(max(inlier_ratio, 1.0e-4)) * math.log1p(inlier_pixels)


def _load_crops(rows: object, mask_dir: Path, max_crops: int) -> list[dict]:
    if not isinstance(rows, (list, tuple)):
        return []
    selected: list[dict] = []
    seen_images: set[int] = set()
    candidates = sorted((row for row in rows if isinstance(row, dict)), key=_quality, reverse=True)
    for row in candidates:
        image_id = int(row.get("image_id", -1))
        if image_id in seen_images:
            continue
        path = _resolve_observation_path(row, mask_dir)
        if path is None:
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                jpeg = np.asarray(archive["crop_jpeg_bytes"], dtype=np.uint8).tobytes()
            image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        except Exception:
            continue
        selected.append({"image_id": image_id, "path": str(path), "image": image, "quality": _quality(row)})
        seen_images.add(image_id)
        if max_crops > 0 and len(selected) >= max_crops:
            break
    return selected


def _embed(images: list[Image.Image], model_path: Path, batch_size: int, device: str) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
    model = model.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), max(1, batch_size)):
            inputs = processor(images=images[start : start + batch_size], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            result = model(**inputs)
            features = result.last_hidden_state[:, 0].float()
            features = torch.nn.functional.normalize(features, dim=-1)
            outputs.append(features.cpu().numpy())
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _robust_lower_gate(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"threshold": -1.0, "median": None, "mad": None, "q1": None, "q3": None}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    q1, q3 = [float(value) for value in np.percentile(values, [25.0, 75.0])]
    mad_gate = median - 3.0 * 1.4826 * mad
    tukey_gate = q1 - 1.5 * (q3 - q1)
    # Use the more conservative lower fence: only strong scene-relative
    # outliers are rejected, with no class-dependent threshold.
    threshold = float(np.clip(min(mad_gate, tukey_gate), -1.0, 1.0))
    return {"threshold": threshold, "median": median, "mad": mad, "q1": q1, "q3": q3}


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-crops", type=int, default=6)
    parser.add_argument("--min-crops", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = args.scene_state.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    active = _numpy(state["active"]).astype(bool, copy=True)
    object_ids = _numpy(state["object_id"]).astype(np.int64, copy=False)
    count = int(object_ids.shape[0])
    observations = state.get("object_mask_observations") or []
    categories = state.get("object_category") or []

    object_rows: list[dict] = []
    all_images: list[Image.Image] = []
    slices: dict[int, tuple[int, int]] = {}
    for index in np.flatnonzero(active).tolist():
        crops = _load_crops(
            observations[index] if index < len(observations) else [],
            args.mask_dir.expanduser().resolve(),
            max(0, int(args.max_crops)),
        )
        start = len(all_images)
        all_images.extend(item["image"] for item in crops)
        slices[index] = (start, len(all_images))
        object_rows.append({
            "index": index,
            "object_id": int(object_ids[index]),
            "category": str(categories[index] if index < len(categories) else ""),
            "crop_count": len(crops),
            "image_ids": [int(item["image_id"]) for item in crops],
            "crop_paths": [str(item["path"]) for item in crops],
        })
    if not all_images:
        raise RuntimeError("No saved object crops were resolved")
    embeddings = _embed(all_images, args.model.expanduser().resolve(), int(args.batch_size), str(args.device))

    medians: list[float] = []
    for row in object_rows:
        start, end = slices[int(row["index"])]
        vectors = embeddings[start:end]
        if vectors.shape[0] >= 2:
            similarities = vectors @ vectors.T
            pairs = similarities[np.triu_indices(vectors.shape[0], k=1)]
            row["median_pairwise_similarity"] = float(np.median(pairs))
            row["min_pairwise_similarity"] = float(np.min(pairs))
            row["p25_pairwise_similarity"] = float(np.percentile(pairs, 25.0))
            medians.append(row["median_pairwise_similarity"])
        else:
            row["median_pairwise_similarity"] = None
            row["min_pairwise_similarity"] = None
            row["p25_pairwise_similarity"] = None

    gate = _robust_lower_gate(np.asarray(medians, dtype=np.float64))
    scores = np.full((count,), np.nan, dtype=np.float32)
    crop_counts = np.zeros((count,), dtype=np.int32)
    statuses = ["not_evaluated"] * count
    rejected: list[int] = []
    for row in object_rows:
        index = int(row["index"])
        score = row.get("median_pairwise_similarity")
        crop_count = int(row["crop_count"])
        enough = crop_count >= int(args.min_crops) and score is not None
        passed = bool(enough and float(score) >= float(gate["threshold"]))
        status = "visual_pass" if passed else ("visual_inconsistent" if enough else "insufficient_crops")
        row["status"] = status
        row["threshold"] = gate["threshold"]
        statuses[index] = status
        crop_counts[index] = crop_count
        if score is not None:
            scores[index] = float(score)
        if not passed:
            active[index] = False
            rejected.append(int(row["object_id"]))

    state["active"] = torch.as_tensor(active, dtype=torch.bool)
    state["object_visual_consistency"] = torch.as_tensor(scores, dtype=torch.float32)
    state["object_visual_crop_count"] = torch.as_tensor(crop_counts, dtype=torch.int32)
    state["object_visual_status"] = statuses
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["visual_consistency"] = {
            "source": str(source),
            "automatic": True,
            "model": str(args.model.expanduser().resolve()),
            "retained_objects": int(active.sum()),
            "threshold": gate["threshold"],
        }
    else:
        payload = state

    duration_seconds = time.perf_counter() - started
    report = {
        "schema": "farm.visual-consistency-audit.v1",
        "created_unix_s": time.time(),
        "source_scene_state": str(source),
        "model": str(args.model.expanduser().resolve()),
        "policy": "prompt-free DINOv3 multi-crop consistency; scene-relative robust lower fence",
        "input_objects": len(object_rows),
        "retained_objects": int(active.sum()),
        "rejected_objects": rejected,
        "timing": {
            "duration_seconds": duration_seconds,
            "seconds_per_input_object": duration_seconds / max(len(object_rows), 1),
            "embedded_crops": len(all_images),
            "stage": "prompt_free_dino_multiview_consistency",
        },
        "gate": gate,
        "objects": object_rows,
    }
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_state)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("input_objects", "retained_objects", "rejected_objects", "gate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
