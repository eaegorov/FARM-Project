#!/usr/bin/env python3
"""Audit repeated part/whole mask relationships in a saved FARM scene.

The analysis is category agnostic.  It links tracks only through repeated
same-view mask containment, metric 3D proximity, and visual feature agreement.
No object names or scene-specific prompts are used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported scene state: {path}")
    return state


def _as_numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _unique_image_count(observations: object) -> int:
    if not isinstance(observations, list):
        return 0
    return len(
        {
            int(row.get("image_id", -1))
            for row in observations
            if isinstance(row, dict) and int(row.get("image_id", -1)) >= 0
        }
    )


def _mask_path(mask_root: Path, object_id: int, observation: dict) -> Path:
    recorded = Path(str(observation.get("path") or ""))
    return mask_root / f"object_{object_id:06d}" / recorded.name


def _load_raw_crop(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        shape = np.asarray(data["raw_shape"], dtype=np.int32).reshape(2)
        bbox = np.asarray(data["raw_bbox_xyxy"], dtype=np.int32).reshape(4)
        flat = np.unpackbits(
            np.asarray(data["raw_bits"], dtype=np.uint8), bitorder="little"
        )
    height, width = int(shape[0]), int(shape[1])
    mask = flat[: height * width].reshape(height, width).astype(bool, copy=False)
    return mask, bbox


def _intersection(mask_a: np.ndarray, box_a: np.ndarray, mask_b: np.ndarray, box_b: np.ndarray) -> int:
    x0 = max(int(box_a[0]), int(box_b[0]))
    y0 = max(int(box_a[1]), int(box_b[1]))
    x1 = min(int(box_a[2]), int(box_b[2]))
    y1 = min(int(box_a[3]), int(box_b[3]))
    if x1 <= x0 or y1 <= y0:
        return 0
    a = mask_a[y0 - int(box_a[1]) : y1 - int(box_a[1]), x0 - int(box_a[0]) : x1 - int(box_a[0])]
    b = mask_b[y0 - int(box_b[1]) : y1 - int(box_b[1]), x0 - int(box_b[0]) : x1 - int(box_b[0])]
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    if height <= 0 or width <= 0:
        return 0
    return int(np.count_nonzero(a[:height, :width] & b[:height, :width]))


def _feature_similarity(features: np.ndarray, a: int, b: int) -> float:
    if features.ndim != 2 or max(a, b) >= features.shape[0]:
        return float("nan")
    va = features[a].astype(np.float64, copy=False)
    vb = features[b].astype(np.float64, copy=False)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 1e-12 else float("nan")


def classify_part_whole_relation(
    *,
    contained_frames: int,
    common_frames: int,
    observations_a: int,
    observations_b: int,
    median_iou: float,
    median_area_ratio: float,
    centre_distance_m: float,
    feature_similarity: float,
    min_shared_frames: int = 2,
    min_contained_fraction: float = 0.5,
    min_track_coverage: float = 0.5,
    max_center_distance_m: float = 1.5,
    min_feature_similarity: float = 0.55,
    duplicate_iou: float = 0.75,
    duplicate_area_ratio: float = 0.70,
) -> tuple[str, dict]:
    """Classify containment without semantic labels or fail-open evidence."""
    contained = max(0, int(contained_frames))
    common = max(0, int(common_frames))
    shorter_track = max(1, min(int(observations_a), int(observations_b)))
    contained_fraction = contained / float(max(common, 1))
    track_coverage = contained / float(shorter_track)
    finite_similarity = bool(np.isfinite(feature_similarity))
    feature_ok = (
        float(min_feature_similarity) < 0.0
        or (finite_similarity and float(feature_similarity) >= float(min_feature_similarity))
    )
    base_pass = (
        contained >= int(min_shared_frames)
        and contained_fraction >= float(min_contained_fraction)
        and track_coverage >= float(min_track_coverage)
        and float(centre_distance_m) <= float(max_center_distance_m)
        and feature_ok
    )
    duplicate = (
        contained >= int(min_shared_frames)
        and contained_fraction >= float(min_contained_fraction)
        and float(centre_distance_m) <= float(max_center_distance_m)
        and feature_ok
        and float(median_iou) >= float(duplicate_iou)
        and float(median_area_ratio) >= float(duplicate_area_ratio)
    )
    relation_type = "duplicate_overlap" if duplicate else "part_whole" if base_pass else "rejected"
    return relation_type, {
        "contained_fraction": contained_fraction,
        "track_coverage": track_coverage,
        "feature_finite": finite_similarity,
        "feature_valid": finite_similarity,
        "feature_ok": feature_ok,
        "base_pass": base_pass,
        "duplicate_overlap": duplicate,
    }


def analyze(
    scene_state: Path,
    mask_root: Path,
    *,
    min_observations: int,
    min_frame_containment: float,
    min_shared_frames: int,
    min_contained_fraction: float,
    min_track_coverage: float,
    max_center_distance_m: float,
    min_feature_similarity: float,
    duplicate_iou: float,
    duplicate_area_ratio: float,
) -> dict:
    started = time.perf_counter()
    state = _load_state(scene_state)
    object_ids = _as_numpy(state.get("object_id"), np.int64).reshape(-1)
    means = _as_numpy(state.get("means"), np.float64)
    active = _as_numpy(state.get("active"), bool).reshape(-1)
    features = _as_numpy(state.get("features"), np.float32)
    observations = state.get("object_mask_observations") or []
    counts = _as_numpy(state.get("count"), np.int64).reshape(-1)

    selected: dict[int, dict] = {}
    frames: dict[int, list[dict]] = defaultdict(list)
    missing_masks = 0
    for index, object_id_raw in enumerate(object_ids.tolist()):
        object_id = int(object_id_raw)
        rows = observations[index] if index < len(observations) else []
        unique_views = _unique_image_count(rows)
        if index >= active.size or not bool(active[index]) or unique_views < min_observations:
            continue
        selected[object_id] = {
            "index": index,
            "observations": unique_views,
            "mapping_count": int(counts[index]) if index < counts.size else unique_views,
        }
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            image_id = int(row.get("image_id", -1))
            path = _mask_path(mask_root, object_id, row)
            if image_id < 0 or not path.is_file():
                missing_masks += 1
                continue
            mask, bbox = _load_raw_crop(path)
            frames[image_id].append(
                {
                    "object_id": object_id,
                    "index": index,
                    "mask": mask,
                    "bbox": bbox,
                    "area": int(np.count_nonzero(mask)),
                }
            )

    pair_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for image_id, rows in frames.items():
        for left, right in itertools.combinations(rows, 2):
            if left["object_id"] == right["object_id"]:
                continue
            pair = tuple(sorted((int(left["object_id"]), int(right["object_id"]))))
            area_a, area_b = int(left["area"]), int(right["area"])
            if min(area_a, area_b) <= 0:
                continue
            intersection = _intersection(left["mask"], left["bbox"], right["mask"], right["bbox"])
            union = area_a + area_b - intersection
            containment = intersection / float(min(area_a, area_b))
            cover_a = intersection / float(area_a)
            cover_b = intersection / float(area_b)
            pair_rows[pair].append(
                {
                    "image_id": int(image_id),
                    "containment": containment,
                    "iou": intersection / float(union) if union > 0 else 0.0,
                    "area_ratio": min(area_a, area_b) / float(max(area_a, area_b)),
                    "smaller_object_id": int(left["object_id"] if area_a <= area_b else right["object_id"]),
                    "cover_a": cover_a,
                    "cover_b": cover_b,
                }
            )

    relations: list[dict] = []
    for (object_a, object_b), rows in pair_rows.items():
        contained = [row for row in rows if row["containment"] >= min_frame_containment]
        if not contained:
            continue
        index_a = selected[object_a]["index"]
        index_b = selected[object_b]["index"]
        distance = float(np.linalg.norm(means[index_a] - means[index_b]))
        similarity = _feature_similarity(features, index_a, index_b)
        smaller_votes = [int(row["smaller_object_id"]) for row in contained]
        smaller_id = max(set(smaller_votes), key=lambda item: (smaller_votes.count(item), -item))
        median_iou = statistics.median(row["iou"] for row in contained)
        median_area_ratio = statistics.median(row["area_ratio"] for row in contained)
        relation_type, diagnostics = classify_part_whole_relation(
            contained_frames=len(contained),
            common_frames=len(rows),
            observations_a=selected[object_a]["observations"],
            observations_b=selected[object_b]["observations"],
            median_iou=median_iou,
            median_area_ratio=median_area_ratio,
            centre_distance_m=distance,
            feature_similarity=similarity,
            min_shared_frames=min_shared_frames,
            min_contained_fraction=min_contained_fraction,
            min_track_coverage=min_track_coverage,
            max_center_distance_m=max_center_distance_m,
            min_feature_similarity=min_feature_similarity,
            duplicate_iou=duplicate_iou,
            duplicate_area_ratio=duplicate_area_ratio,
        )
        accepted = relation_type == "part_whole"
        relations.append(
            {
                "object_a": object_a,
                "object_b": object_b,
                "observations_a": selected[object_a]["observations"],
                "observations_b": selected[object_b]["observations"],
                "common_frames": len(rows),
                "contained_frames": len(contained),
                "contained_fraction": diagnostics["contained_fraction"],
                "track_coverage": diagnostics["track_coverage"],
                "median_containment": statistics.median(row["containment"] for row in contained),
                "median_iou": median_iou,
                "median_area_ratio": median_area_ratio,
                "smaller_object_id": smaller_id,
                "centre_distance_m": distance,
                "feature_cosine_similarity": similarity,
                "feature_similarity_finite": diagnostics["feature_finite"],
                "relation_type": relation_type,
                "accepted": accepted,
                "frames": [int(row["image_id"]) for row in contained],
            }
        )

    relations.sort(
        key=lambda row: (
            bool(row["accepted"]),
            int(row["contained_frames"]),
            float(row["median_containment"]),
        ),
        reverse=True,
    )
    return {
        "schema": "farm.part-whole-audit.v1",
        "created_unix_s": time.time(),
        "source_scene_state": str(scene_state.resolve()),
        "mask_root": str(mask_root.resolve()),
        "policy": {
            "category_agnostic": True,
            "min_observations": min_observations,
            "min_frame_containment": min_frame_containment,
            "min_shared_frames": min_shared_frames,
            "min_contained_fraction": min_contained_fraction,
            "min_track_coverage": min_track_coverage,
            "max_center_distance_m": max_center_distance_m,
            "min_feature_similarity": min_feature_similarity,
            "duplicate_iou": duplicate_iou,
            "duplicate_area_ratio": duplicate_area_ratio,
        },
        "objects_evaluated": len(selected),
        "frames_evaluated": len(frames),
        "missing_masks": missing_masks,
        "candidate_relations": len(relations),
        "accepted_relations": sum(bool(row["accepted"]) for row in relations),
        "relations": relations,
        "timing": {"duration_seconds": time.perf_counter() - started},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-frame-containment", type=float, default=0.95)
    parser.add_argument("--min-shared-frames", type=int, default=2)
    parser.add_argument("--min-contained-fraction", type=float, default=0.5)
    parser.add_argument("--min-track-coverage", type=float, default=0.5)
    parser.add_argument("--max-center-distance-m", type=float, default=1.5)
    parser.add_argument("--min-feature-similarity", type=float, default=0.55)
    parser.add_argument("--duplicate-iou", type=float, default=0.75)
    parser.add_argument("--duplicate-area-ratio", type=float, default=0.70)
    args = parser.parse_args()
    report = analyze(
        args.scene_state,
        args.mask_root,
        min_observations=args.min_observations,
        min_frame_containment=args.min_frame_containment,
        min_shared_frames=args.min_shared_frames,
        min_contained_fraction=args.min_contained_fraction,
        min_track_coverage=args.min_track_coverage,
        max_center_distance_m=args.max_center_distance_m,
        min_feature_similarity=args.min_feature_similarity,
        duplicate_iou=args.duplicate_iou,
        duplicate_area_ratio=args.duplicate_area_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("objects_evaluated", "candidate_relations", "accepted_relations", "timing")}, indent=2))


if __name__ == "__main__":
    main()
