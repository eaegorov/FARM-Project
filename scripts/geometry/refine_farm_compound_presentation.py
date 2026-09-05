#!/usr/bin/env python3
"""Refit probable compound OBBs on independently supported consensus cells.

This is a cheap, presentation-only refinement.  It does not discover semantic
classes, activate objects, or promote geometry to a metric pass.  Existing
mask-depth consensus cells define the object, a separately exported 3DGS cloud
rejects unsupported cells, and strict paired gates decide whether a candidate
may replace the compound boxes in a separate output state.

The source state is never overwritten.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.geometry.refine_farm_compound_geometry import _fit_floor_obb, _points_inside_obb
from scripts.geometry.refine_farm_object_geometry import _load_frames, _observation_points
try:
    from scripts.geometry.farm_geometry_axes import normalize_up_vector
except ModuleNotFoundError:  # package import
    from scripts.geometry.farm_geometry_axes import normalize_up_vector


def _finite_points(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    points = np.asarray(value, dtype=np.float32).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _rotation_from_wxyz(value: object) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(quat).all() or norm <= 1.0e-9:
        raise ValueError("Invalid OBB quaternion")
    quat /= norm
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _obb_from_record(record: Mapping[str, object]) -> dict:
    center = record.get("center_m", record.get("center"))
    dimensions = record.get("dimensions_lwh_m", record.get("dimensions_m", record.get("dimensions")))
    quaternion = record.get("wxyz")
    if center is None or dimensions is None or quaternion is None:
        raise ValueError("Compound OBB record is incomplete")
    result = {
        "center": np.asarray(center, dtype=np.float64).reshape(3),
        "dimensions": np.asarray(dimensions, dtype=np.float64).reshape(3),
        "rotation_matrix": _rotation_from_wxyz(quaternion),
        "wxyz": np.asarray(quaternion, dtype=np.float64).reshape(4),
    }
    if not np.isfinite(result["center"]).all() or not np.isfinite(result["dimensions"]).all():
        raise ValueError("Compound OBB record contains non-finite values")
    if np.any(result["dimensions"] <= 0.0):
        raise ValueError("Compound OBB dimensions must be positive")
    return result


def _obb_record(obb: Mapping[str, object], *, cells: int, supported_rate: float) -> dict:
    matrix = np.asarray(obb["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    xyzw = Rotation.from_matrix(matrix).as_quat()
    wxyz = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if wxyz[0] < 0.0:
        wxyz *= -1.0
    dimensions = np.asarray(obb["dimensions"], dtype=np.float64).reshape(3)
    return {
        "center_m": np.asarray(obb["center"], dtype=np.float64).round(6).tolist(),
        "dimensions_lwh_m": dimensions.round(6).tolist(),
        "wxyz": wxyz.round(8).tolist(),
        "volume_m3": float(np.prod(dimensions)),
        "orientation_confidence": float(obb.get("orientation_confidence", 0.0)),
        "gravity_tilt_degrees": float(obb.get("gravity_tilt_degrees", 0.0)),
        "consensus_cells": int(cells),
        "gaussian_supported_rate": float(supported_rate),
    }


def _obb_corners(obb: Mapping[str, object]) -> np.ndarray:
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * (0.5 * np.asarray(obb["dimensions"], dtype=np.float64))
    return local @ np.asarray(obb["rotation_matrix"], dtype=np.float64).T + np.asarray(obb["center"])


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    low = np.maximum(first[:2], second[:2])
    high = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(high[0] - low[0])) * max(0.0, float(high[1] - low[1]))
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(area_first + area_second - intersection, 1.0e-12)


def _project(points: np.ndarray, observation: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    inverse_pose = np.linalg.inv(np.asarray(observation["pose"], dtype=np.float64))
    homogeneous = np.column_stack((points, np.ones((points.shape[0],), dtype=np.float64)))
    camera = (inverse_pose @ homogeneous.T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1.0e-5)
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    projected = (np.asarray(observation["K"], dtype=np.float64) @ camera[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def _projection_metrics(boxes: Sequence[Mapping[str, object]], observations: Sequence[Mapping[str, object]]) -> dict:
    if not boxes or not observations:
        return {
            "valid_box_views": 0,
            "median_union_box_iou": 0.0,
            "p25_union_box_iou": 0.0,
            "center_inside_rate": 0.0,
        }
    all_corners = np.concatenate([_obb_corners(box) for box in boxes], axis=0)
    centers = np.stack([np.asarray(box["center"], dtype=np.float64) for box in boxes])
    ious: list[float] = []
    center_inside: list[float] = []
    for observation in observations:
        bbox = np.asarray(observation["raw_bbox"], dtype=np.float64).reshape(4)
        center_pixels, center_valid = _project(centers, observation)
        hit = (
            center_valid
            & (center_pixels[:, 0] >= bbox[0])
            & (center_pixels[:, 0] <= bbox[2])
            & (center_pixels[:, 1] >= bbox[1])
            & (center_pixels[:, 1] <= bbox[3])
        )
        center_inside.append(float(np.any(hit)))
        corner_pixels, corner_valid = _project(all_corners, observation)
        if bool(np.all(corner_valid)):
            projected_bbox = np.r_[corner_pixels.min(axis=0), corner_pixels.max(axis=0)]
            ious.append(_bbox_iou(projected_bbox, bbox))
    return {
        "valid_box_views": len(ious),
        "median_union_box_iou": float(np.median(ious)) if ious else 0.0,
        "p25_union_box_iou": float(np.percentile(ious, 25.0)) if ious else 0.0,
        "min_union_box_iou": float(np.min(ious)) if ious else 0.0,
        "center_inside_rate": float(np.mean(center_inside)) if center_inside else 0.0,
        "per_view_union_box_iou": ious,
    }


def _rotation_delta_degrees(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    relative = np.asarray(first["rotation_matrix"]).T @ np.asarray(second["rotation_matrix"])
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def fit_supported_compound(
    points: np.ndarray,
    supported: np.ndarray,
    current_boxes: Sequence[Mapping[str, object]],
    *,
    parent_box: Mapping[str, object],
    split_axis: int,
    split_threshold_m: float,
    quantile: float,
    up_vector: object | None = None,
) -> dict:
    """Fit child boxes from supported cells while preserving saved topology."""
    points_np = _finite_points(points)
    support_np = np.asarray(supported, dtype=bool).reshape(-1)
    if support_np.shape[0] != points_np.shape[0]:
        raise ValueError("supported must match points")
    if len(current_boxes) != 2:
        raise ValueError("Only persisted two-component compounds are supported")
    axis = int(split_axis)
    if axis not in (0, 1, 2):
        raise ValueError("split_axis must be 0, 1, or 2")
    parent = _obb_from_record(parent_box)
    local = (points_np.astype(np.float64) - parent["center"]) @ parent["rotation_matrix"]
    component_masks = (local[:, axis] <= float(split_threshold_m), local[:, axis] > float(split_threshold_m))
    if min(int((mask & support_np).sum()) for mask in component_masks) < 24:
        raise ValueError("Too few independently supported cells in a component")
    candidates = [
        _fit_floor_obb(
            points_np[mask & support_np], float(quantile), up_vector=up_vector
        )
        for mask in component_masks
    ]
    current = [_obb_from_record(box) for box in current_boxes]
    current_inside = np.logical_or.reduce([_points_inside_obb(points_np, box) for box in current])
    candidate_inside_by_child = [_points_inside_obb(points_np, box) for box in candidates]
    candidate_inside = np.logical_or.reduce(candidate_inside_by_child)
    current_volume = float(sum(np.prod(box["dimensions"]) for box in current))
    candidate_volume = float(sum(np.prod(box["dimensions"]) for box in candidates))
    return {
        "current_boxes": current,
        "candidate_boxes": candidates,
        "candidate_records": [
            _obb_record(
                box,
                cells=int(inside.sum()),
                supported_rate=float(support_np[inside].mean()),
            )
            for box, inside in zip(candidates, candidate_inside_by_child)
        ],
        "current_inside": current_inside,
        "candidate_inside": candidate_inside,
        "current_coverage": float(current_inside.mean()),
        "candidate_coverage": float(candidate_inside.mean()),
        "current_supported_coverage": float((current_inside & support_np).sum() / max(support_np.sum(), 1)),
        "candidate_supported_coverage": float((candidate_inside & support_np).sum() / max(support_np.sum(), 1)),
        "current_inside_support_rate": float((current_inside & support_np).sum() / max(current_inside.sum(), 1)),
        "candidate_inside_support_rate": float((candidate_inside & support_np).sum() / max(candidate_inside.sum(), 1)),
        "current_volume_m3": current_volume,
        "candidate_volume_m3": candidate_volume,
        "volume_inflation": float(candidate_volume / max(current_volume, 1.0e-12) - 1.0),
        "rotation_delta_degrees": [
            _rotation_delta_degrees(before, after) for before, after in zip(current, candidates)
        ],
    }


def evaluate_guards(
    result: Mapping[str, object],
    current_projection: Mapping[str, float],
    candidate_projection: Mapping[str, float],
    *,
    min_supported_coverage: float,
    min_inside_support_rate: float,
    max_volume_inflation: float,
    max_rotation_delta_degrees: float,
    min_projection_ratio: float,
) -> dict[str, bool]:
    current_iou = float(current_projection.get("median_union_box_iou", 0.0))
    current_p25 = float(current_projection.get("p25_union_box_iou", 0.0))
    return {
        "all_consensus_coverage_not_worse": float(result["candidate_coverage"]) >= float(result["current_coverage"]),
        "supported_coverage_not_worse": float(result["candidate_supported_coverage"]) >= float(result["current_supported_coverage"]),
        "supported_coverage_floor": float(result["candidate_supported_coverage"]) >= float(min_supported_coverage),
        "inside_support_rate_floor": float(result["candidate_inside_support_rate"]) >= float(min_inside_support_rate),
        "volume_inflation_limit": float(result["volume_inflation"]) <= float(max_volume_inflation),
        "rotation_delta_limit": max(result["rotation_delta_degrees"], default=float("inf")) <= float(max_rotation_delta_degrees),
        "median_projection_not_worse": float(candidate_projection.get("median_union_box_iou", 0.0)) >= float(min_projection_ratio) * current_iou,
        "p25_projection_not_worse": float(candidate_projection.get("p25_union_box_iou", 0.0)) >= float(min_projection_ratio) * current_p25,
        "center_support_not_worse": float(candidate_projection.get("center_inside_rate", 0.0)) >= float(current_projection.get("center_inside_rate", 0.0)),
    }


def _load_observations(
    object_id: int,
    mask_root: Path,
    *,
    state_images: list,
    frames_root: Path,
    frames_by_stem: dict[str, dict],
) -> list[dict]:
    observations: list[dict] = []
    for index, path in enumerate(sorted((mask_root / f"object_{object_id:06d}").glob("*.npz"))):
        observation = _observation_points(
            path,
            state_images=state_images,
            frames_root=frames_root,
            frames_by_stem=frames_by_stem,
            max_points=32,
            seed=20260805 + index,
        )
        if observation is not None:
            observations.append(observation)
    return observations


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--cloud-npz", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-scene-state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--object-id", type=int, action="append", default=[])
    parser.add_argument("--quantile", type=float, default=0.01)
    parser.add_argument("--min-supported-coverage", type=float, default=0.98)
    parser.add_argument("--min-inside-support-rate", type=float, default=0.98)
    parser.add_argument("--max-volume-inflation", type=float, default=0.05)
    parser.add_argument("--max-rotation-delta-degrees", type=float, default=2.0)
    parser.add_argument("--min-projection-ratio", type=float, default=0.95)
    args = parser.parse_args()

    source = args.scene_state.resolve()
    output = args.output_scene_state.resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source scene state")
    wrapper = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("state"), dict):
        raise ValueError("Expected a wrapped FARM state dictionary")
    output_wrapper = copy.deepcopy(wrapper)
    state = output_wrapper["state"]
    resolved_up = normalize_up_vector(
        state.get("object_geometry_up_vector", [0.0, 1.0, 0.0])
    )
    means = state.get("means")
    count = (
        int(means.shape[0])
        if isinstance(means, torch.Tensor) and means.ndim > 0
        else len(state.get("object_caption") or [])
    )
    object_ids = state.get("object_id")
    if isinstance(object_ids, torch.Tensor):
        object_ids_np = object_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        object_ids_np = np.arange(count, dtype=np.int64)
    index_by_id = {int(value): index for index, value in enumerate(object_ids_np.tolist())}
    requested = sorted(set(args.object_id)) if args.object_id else sorted(index_by_id)

    cloud_archive = np.load(args.cloud_npz, allow_pickle=False)
    cloud = _finite_points(cloud_archive["xyz"])
    cloud_tree = cKDTree(cloud)
    frames_root, frames_by_stem = _load_frames(args.frames_json)
    boxes_all = state.get("object_compound_boxes") or []
    diagnostics_all = state.get("object_compound_geometry_diagnostics") or []
    consensus_all = state.get("object_geometry_consensus_points") or []
    statuses = state.get("object_geometry_status") or []
    active = state.get("active")
    rows: list[dict] = []
    accepted = 0

    for object_id in requested:
        index = index_by_id.get(int(object_id))
        if index is None:
            rows.append({"object_id": int(object_id), "accepted": False, "reason": "object_id_not_found"})
            continue
        status = str(statuses[index] if index < len(statuses) else "").strip().lower()
        if "compound_geometry_probable" not in status or "rejected" in status:
            rows.append({"object_id": int(object_id), "accepted": False, "reason": "not_probable_compound"})
            continue
        current_records = boxes_all[index] if index < len(boxes_all) else []
        diagnostics = diagnostics_all[index] if index < len(diagnostics_all) else {}
        compound = diagnostics.get("compound", {}) if isinstance(diagnostics, dict) else {}
        points = _finite_points(consensus_all[index] if index < len(consensus_all) else np.zeros((0, 3)))
        if len(current_records) != 2 or points.shape[0] < 48 or not isinstance(compound, dict):
            rows.append({"object_id": int(object_id), "accepted": False, "reason": "compound_evidence_incomplete"})
            continue
        parent_box = compound.get("parent_box")
        split_axis = compound.get("axis")
        split_threshold = compound.get("threshold_local_m")
        support_radius = float((diagnostics.get("gaussian_support") or {}).get("support_radius_m", 0.12))
        if parent_box is None or split_axis is None or split_threshold is None or not math.isfinite(support_radius):
            rows.append({"object_id": int(object_id), "accepted": False, "reason": "split_or_support_metadata_missing"})
            continue
        distances = cloud_tree.query(points, k=1, workers=-1)[0]
        supported = distances <= support_radius
        try:
            fit = fit_supported_compound(
                points,
                supported,
                current_records,
                parent_box=parent_box,
                split_axis=int(split_axis),
                split_threshold_m=float(split_threshold),
                quantile=float(args.quantile),
                up_vector=resolved_up,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append({"object_id": int(object_id), "accepted": False, "reason": "fit_failed", "error": str(error)})
            continue
        observations = _load_observations(
            int(object_id),
            args.mask_root,
            state_images=state.get("images") or [],
            frames_root=frames_root,
            frames_by_stem=frames_by_stem,
        )
        current_projection = _projection_metrics(fit["current_boxes"], observations)
        candidate_projection = _projection_metrics(fit["candidate_boxes"], observations)
        guards = evaluate_guards(
            fit,
            current_projection,
            candidate_projection,
            min_supported_coverage=float(args.min_supported_coverage),
            min_inside_support_rate=float(args.min_inside_support_rate),
            max_volume_inflation=float(args.max_volume_inflation),
            max_rotation_delta_degrees=float(args.max_rotation_delta_degrees),
            min_projection_ratio=float(args.min_projection_ratio),
        )
        is_active = bool(active[index]) if isinstance(active, torch.Tensor) else bool(active[index])
        guards["probable_status_preserved"] = "compound_geometry_probable" in status
        guards["inactive_status_preserved"] = not is_active
        accepted_row = bool(all(guards.values()))
        before = fit["current_inside"]
        after = fit["candidate_inside"]
        added = after & ~before
        removed = before & ~after
        row = {
            "object_id": int(object_id),
            "accepted": accepted_row,
            "reason": "all_preview_guards_passed" if accepted_row else "preview_guard_failed",
            "geometry_status": status,
            "active": is_active,
            "consensus_cells": int(points.shape[0]),
            "3dgs_support": {
                "radius_m": support_radius,
                "supported_cells": int(supported.sum()),
                "supported_rate": float(supported.mean()),
                "median_nn_m": float(np.median(distances)),
                "p90_nn_m": float(np.percentile(distances, 90.0)),
            },
            "current": {
                "coverage": fit["current_coverage"],
                "supported_coverage": fit["current_supported_coverage"],
                "inside_support_rate": fit["current_inside_support_rate"],
                "summed_volume_m3": fit["current_volume_m3"],
                "projection": current_projection,
                "boxes": current_records,
            },
            "candidate": {
                "coverage": fit["candidate_coverage"],
                "supported_coverage": fit["candidate_supported_coverage"],
                "inside_support_rate": fit["candidate_inside_support_rate"],
                "summed_volume_m3": fit["candidate_volume_m3"],
                "volume_inflation": fit["volume_inflation"],
                "rotation_delta_degrees": fit["rotation_delta_degrees"],
                "projection": candidate_projection,
                "boxes": fit["candidate_records"],
            },
            "delta": {
                "added_cells": int(added.sum()),
                "added_supported_cells": int((added & supported).sum()),
                "added_unsupported_cells": int((added & ~supported).sum()),
                "removed_cells": int(removed.sum()),
                "removed_supported_cells": int((removed & supported).sum()),
                "removed_unsupported_cells": int((removed & ~supported).sum()),
            },
            "guards": guards,
        }
        if accepted_row:
            boxes_all[index] = fit["candidate_records"]
            accepted += 1
        rows.append(row)

    state["object_compound_boxes"] = boxes_all
    output_wrapper.setdefault("meta", {})["compound_presentation_refinement"] = {
        "source_scene_state": str(source),
        "cloud_npz": str(args.cloud_npz.resolve()),
        "accepted_objects": accepted,
        "requested_object_ids": requested,
        "semantic_prompts_used": False,
        "metric_status_changed": False,
        "resolved_up": resolved_up.astype(float).tolist(),
    }
    _atomic_torch_save(output_wrapper, output)
    report = {
        "schema": "farm.compound-presentation-refinement.v1",
        "created_unix_s": time.time(),
        "source_scene_state": str(source),
        "output_scene_state": str(output),
        "cloud_npz": str(args.cloud_npz.resolve()),
        "policy": {
            "semantic_prompts_used": False,
            "metric_status_changed": False,
            "resolved_up": resolved_up.astype(float).tolist(),
            "quantile": float(args.quantile),
            "min_supported_coverage": float(args.min_supported_coverage),
            "min_inside_support_rate": float(args.min_inside_support_rate),
            "max_volume_inflation": float(args.max_volume_inflation),
            "max_rotation_delta_degrees": float(args.max_rotation_delta_degrees),
            "min_projection_ratio": float(args.min_projection_ratio),
        },
        "requested_objects": len(requested),
        "accepted_objects": accepted,
        "duration_seconds": time.perf_counter() - started,
        "objects": rows,
    }
    _atomic_json_save(report, args.report.resolve())
    print(json.dumps({"output_scene_state": str(output), "report": str(args.report.resolve()), "accepted": accepted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
