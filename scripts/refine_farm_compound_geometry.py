#!/usr/bin/env python3
"""Recover probable compound geometry for unresolved FARM assemblies.

The regular assembly builder merges the persistent FARM voxel buffers.  Those
buffers are intentionally coarse and can contain stale points from an earlier
association.  This post-processing stage instead reconstructs unquantized
per-view RGB-D evidence from the saved masks, keeps only geometry repeated in
multiple views, and approximates a non-convex assembly with at most two
gravity-aligned OBBs.

The stage is deliberately conservative:

* only assembly rows whose geometry is rejected or still pending are touched;
* a 3DGS presentation cloud is used as an independent geometric support gate;
* a split must reduce the parent OBB volume substantially while retaining at
  least 90 percent of the consensus cells;
* accepted results are marked ``assembly_compound_geometry_probable`` and are
  kept inactive pending a later presentation/retrieval policy decision.

The input state and the original FARM voxel buffers are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    from farm_geometry_axes import add_up_arguments, policy_from_args, write_state_up_policy
except ModuleNotFoundError:  # package import: scripts.refine_farm_compound_geometry
    from scripts.farm_geometry_axes import add_up_arguments, policy_from_args, write_state_up_policy
from refine_farm_object_geometry import (
    _fit_robust_obb,
    _load_frames,
    _numpy,
    _observation_points,
)
from scene_graph.utils.geometry import VOXEL_BASE_V, decode_voxel_keys_numpy


DEFAULT_ELIGIBLE_STATUSES = frozenset(
    {
        "",
        "not_evaluated",
        "geometry_rejected",
        "assembly_geometry_pending",
        "assembly_geometry_rejected",
        "assembly_needs_geometry",
        "assembly_compound_geometry_pending",
        "assembly_compound_geometry_rejected",
    }
)

LEGACY_UNVALIDATED_STATUS = "assembly_voxel_pass"


def _finite_points(points: object) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return array[np.isfinite(array).all(axis=1)]


def _fit_floor_obb(
    points: np.ndarray,
    quantile: float,
    *,
    up_axis_index: int = 1,
    up_vector: object | None = None,
) -> dict:
    return _fit_robust_obb(
        _finite_points(points),
        float(quantile),
        orientation_mode="gravity_yaw",
        up_axis_index=int(up_axis_index),
        up_vector=up_vector,
    )


def _obb_volume(obb: Mapping[str, object]) -> float:
    dimensions = np.asarray(obb["dimensions"], dtype=np.float64).reshape(3)
    return float(np.prod(np.maximum(dimensions, 0.0)))


def _points_inside_obb(points: np.ndarray, obb: Mapping[str, object]) -> np.ndarray:
    points_np = _finite_points(points).astype(np.float64, copy=False)
    center = np.asarray(obb["center"], dtype=np.float64).reshape(3)
    rotation = np.asarray(obb["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    half = 0.5 * np.asarray(obb["dimensions"], dtype=np.float64).reshape(3)
    local = (points_np - center) @ rotation
    return np.all(np.abs(local) <= half + 1.0e-6, axis=1)


def _box_record(
    obb: Mapping[str, object],
    *,
    point_count: int,
    point_fraction: float,
    gaussian_supported_rate: float,
) -> dict:
    dimensions = np.asarray(obb["dimensions"], dtype=np.float64).reshape(3)
    return {
        "center_m": np.asarray(obb["center"], dtype=np.float64).round(6).tolist(),
        "dimensions_lwh_m": dimensions.round(6).tolist(),
        "wxyz": np.asarray(obb["wxyz"], dtype=np.float64).round(8).tolist(),
        "volume_m3": float(np.prod(dimensions)),
        "orientation_confidence": float(obb.get("orientation_confidence", 0.0)),
        "gravity_tilt_degrees": float(obb.get("gravity_tilt_degrees", 0.0)),
        "up_vector": np.asarray(obb.get("up_vector", [0.0, 1.0, 0.0]), dtype=np.float64).tolist(),
        "consensus_cells": int(point_count),
        "consensus_fraction": float(point_fraction),
        "gaussian_supported_rate": float(gaussian_supported_rate),
    }


def build_multiview_consensus(
    points_by_view: Mapping[int, np.ndarray],
    *,
    grid_m: float,
    min_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell centres and distinct-view counts for repeated geometry.

    Multiple masks belonging to the same physical image must be concatenated
    before calling this helper.  A cell therefore receives at most one vote
    from a view, preventing nested part masks from inflating its support.
    """

    grid = float(grid_m)
    if not math.isfinite(grid) or grid <= 0.0:
        raise ValueError("grid_m must be finite and positive")
    required = max(1, int(min_views))
    cell_views: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for image_id in sorted(points_by_view):
        points = _finite_points(points_by_view[image_id])
        if points.shape[0] == 0:
            continue
        keys = np.unique(np.floor(points / grid).astype(np.int64), axis=0)
        for key in keys.tolist():
            cell_views[(int(key[0]), int(key[1]), int(key[2]))].add(int(image_id))

    retained = sorted(key for key, views in cell_views.items() if len(views) >= required)
    if not retained:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    keys_np = np.asarray(retained, dtype=np.int64)
    centres = (keys_np.astype(np.float64) + 0.5) * grid
    view_counts = np.asarray([len(cell_views[key]) for key in retained], dtype=np.int32)
    return centres.astype(np.float32), view_counts


def gaussian_support_metrics(
    distances_m: Sequence[float] | np.ndarray,
    *,
    grid_m: float,
    min_supported_rate: float = 0.90,
) -> dict:
    """Summarise and gate point-to-3DGS-centre distances.

    The thresholds include the consensus-cell quantisation error and are
    intentionally looser than the typical 1--5 cm raw RGB-D residuals.
    """

    distances = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    distances = distances[np.isfinite(distances) & (distances >= 0.0)]
    support_radius = max(0.08, 1.5 * float(grid_m))
    p95_limit = max(0.15, 3.0 * float(grid_m))
    if distances.size == 0:
        return {
            "points": 0,
            "median_nn_m": None,
            "p90_nn_m": None,
            "p95_nn_m": None,
            "support_radius_m": support_radius,
            "supported_rate": 0.0,
            "p95_limit_m": p95_limit,
            "passed": False,
        }
    median = float(np.median(distances))
    p90 = float(np.percentile(distances, 90.0))
    p95 = float(np.percentile(distances, 95.0))
    supported_rate = float(np.mean(distances <= support_radius))
    passed = (
        median <= support_radius
        and p95 <= p95_limit
        and supported_rate >= float(min_supported_rate)
    )
    return {
        "points": int(distances.size),
        "median_nn_m": median,
        "p90_nn_m": p90,
        "p95_nn_m": p95,
        "support_radius_m": support_radius,
        "supported_rate": supported_rate,
        "p95_limit_m": p95_limit,
        "passed": bool(passed),
    }


def coarse_voxel_support_metrics(
    distances_m: Sequence[float] | np.ndarray,
    *,
    voxel_spacing_m: float,
) -> dict:
    """Audit whether persistent FARM voxels are safe for OBB expansion."""

    distances = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    distances = distances[np.isfinite(distances) & (distances >= 0.0)]
    median_limit = max(0.10, float(voxel_spacing_m))
    p90_limit = max(0.25, 3.0 * float(voxel_spacing_m))
    if distances.size == 0:
        return {
            "points": 0,
            "voxel_spacing_m": float(voxel_spacing_m),
            "median_nn_m": None,
            "p90_nn_m": None,
            "p95_nn_m": None,
            "median_limit_m": median_limit,
            "p90_limit_m": p90_limit,
            "passed": False,
        }
    median = float(np.median(distances))
    p90 = float(np.percentile(distances, 90.0))
    return {
        "points": int(distances.size),
        "voxel_spacing_m": float(voxel_spacing_m),
        "median_nn_m": median,
        "p90_nn_m": p90,
        "p95_nn_m": float(np.percentile(distances, 95.0)),
        "median_limit_m": median_limit,
        "p90_limit_m": p90_limit,
        "passed": bool(median <= median_limit and p90 <= p90_limit),
    }


def find_binary_compound(
    points: np.ndarray,
    *,
    gaussian_supported: np.ndarray | None = None,
    quantile: float = 0.025,
    max_volume_ratio: float = 0.65,
    min_coverage: float = 0.90,
    min_child_fraction: float = 0.15,
    min_child_cells: int = 24,
    min_child_gaussian_rate: float = 0.90,
    split_quantiles: Iterable[float] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    up_axis_index: int = 1,
    up_vector: object | None = None,
) -> dict:
    """Find a generic two-OBB approximation of a non-convex point set."""

    points_np = _finite_points(points)
    if points_np.shape[0] < 2 * max(1, int(min_child_cells)):
        return {"accepted": False, "reason": "insufficient_consensus_cells", "candidate_count": 0}
    if gaussian_supported is None:
        supported = np.ones((points_np.shape[0],), dtype=bool)
    else:
        supported = np.asarray(gaussian_supported, dtype=bool).reshape(-1)
        if supported.shape[0] != points_np.shape[0]:
            raise ValueError("gaussian_supported must match the number of points")

    parent = _fit_floor_obb(
        points_np,
        float(quantile),
        up_axis_index=up_axis_index,
        up_vector=up_vector,
    )
    parent_volume = max(_obb_volume(parent), 1.0e-9)
    parent_local = (
        points_np.astype(np.float64)
        - np.asarray(parent["center"], dtype=np.float64).reshape(1, 3)
    ) @ np.asarray(parent["rotation_matrix"], dtype=np.float64)
    minimum = max(int(min_child_cells), int(math.ceil(float(min_child_fraction) * points_np.shape[0])))
    candidates: list[dict] = []

    for axis in range(3):
        values = parent_local[:, axis]
        for split_quantile in split_quantiles:
            q = float(split_quantile)
            if not 0.0 < q < 1.0:
                continue
            threshold = float(np.quantile(values, q))
            first_mask = values <= threshold
            second_mask = ~first_mask
            counts = (int(first_mask.sum()), int(second_mask.sum()))
            if min(counts) < minimum:
                continue
            masks = (first_mask, second_mask)
            child_boxes = [
                _fit_floor_obb(
                    points_np[mask],
                    float(quantile),
                    up_axis_index=up_axis_index,
                    up_vector=up_vector,
                )
                for mask in masks
            ]
            covered = np.zeros((points_np.shape[0],), dtype=bool)
            for child in child_boxes:
                covered |= _points_inside_obb(points_np, child)
            coverage = float(covered.mean())
            child_support_rates = [float(supported[mask].mean()) for mask in masks]
            summed_volume = float(sum(_obb_volume(child) for child in child_boxes))
            volume_ratio = summed_volume / parent_volume
            accepted = (
                volume_ratio <= float(max_volume_ratio)
                and coverage >= float(min_coverage)
                and min(child_support_rates) >= float(min_child_gaussian_rate)
            )
            candidates.append(
                {
                    "accepted": bool(accepted),
                    "axis": int(axis),
                    "split_quantile": q,
                    "threshold_local_m": threshold,
                    "child_masks": masks,
                    "child_boxes_raw": child_boxes,
                    "child_counts": counts,
                    "child_supported_rates": child_support_rates,
                    "coverage": coverage,
                    "summed_child_volume_m3": summed_volume,
                    "volume_ratio": volume_ratio,
                }
            )

    eligible = [candidate for candidate in candidates if candidate["accepted"]]
    parent_record = _box_record(
        parent,
        point_count=points_np.shape[0],
        point_fraction=1.0,
        gaussian_supported_rate=float(supported.mean()),
    )
    if not eligible:
        best = min(candidates, key=lambda row: (row["volume_ratio"], -row["coverage"])) if candidates else None
        return {
            "accepted": False,
            "reason": "no_split_passed_gates",
            "candidate_count": len(candidates),
            "parent_box": parent_record,
            "best_rejected": (
                {
                    key: best[key]
                    for key in (
                        "axis",
                        "split_quantile",
                        "child_counts",
                        "child_supported_rates",
                        "coverage",
                        "summed_child_volume_m3",
                        "volume_ratio",
                    )
                }
                if best is not None
                else None
            ),
        }

    selected = min(eligible, key=lambda row: (row["volume_ratio"], -row["coverage"]))
    children = []
    for child, count, support_rate in zip(
        selected["child_boxes_raw"],
        selected["child_counts"],
        selected["child_supported_rates"],
    ):
        children.append(
            _box_record(
                child,
                point_count=int(count),
                point_fraction=float(count / points_np.shape[0]),
                gaussian_supported_rate=float(support_rate),
            )
        )
    return {
        "accepted": True,
        "reason": "compound_split_passed",
        "candidate_count": len(candidates),
        "parent_box": parent_record,
        "axis": int(selected["axis"]),
        "split_quantile": float(selected["split_quantile"]),
        "threshold_local_m": float(selected["threshold_local_m"]),
        "coverage": float(selected["coverage"]),
        "summed_child_volume_m3": float(selected["summed_child_volume_m3"]),
        "volume_ratio": float(selected["volume_ratio"]),
        "children": children,
    }


def _ensure_list_field(state: dict, key: str, count: int, default_factory) -> list:
    value = state.get(key)
    if isinstance(value, list):
        result = copy.deepcopy(value)
    elif isinstance(value, tuple):
        result = copy.deepcopy(list(value))
    else:
        result = []
    while len(result) < count:
        result.append(default_factory())
    if len(result) > count:
        result = result[:count]
    state[key] = result
    return result


def _query_tree(tree: cKDTree, points: np.ndarray) -> np.ndarray:
    points_np = _finite_points(points)
    if points_np.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    try:
        distances = tree.query(points_np, k=1, workers=-1)[0]
    except TypeError:  # pragma: no cover - compatibility with old SciPy.
        distances = tree.query(points_np, k=1)[0]
    return np.asarray(distances, dtype=np.float64)


def _load_cloud(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        if "xyz" not in data.files:
            raise ValueError(f"Presentation cloud has no xyz array: {path}")
        points = _finite_points(data["xyz"])
    if points.shape[0] < 1000:
        raise ValueError(f"Presentation cloud is unexpectedly small: {points.shape[0]} points")
    provenance: dict = {"cloud_npz": str(path), "points": int(points.shape[0])}
    manifest_path = path.parent.parent / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cloud_manifest = manifest.get("cloud") if isinstance(manifest, dict) else None
            if isinstance(cloud_manifest, dict):
                provenance["manifest"] = str(manifest_path)
                provenance["source_ply"] = str(cloud_manifest.get("source_ply") or "")
                provenance["source_vertices"] = int(cloud_manifest.get("source_vertices") or 0)
        except (OSError, ValueError, TypeError):
            pass
    return points, provenance


def _is_assembly(member_ids: object) -> bool:
    if not isinstance(member_ids, (list, tuple)) or len(member_ids) < 2:
        return False
    return True


def _eligible_assembly(member_ids: object, geometry_status: object) -> bool:
    if not _is_assembly(member_ids):
        return False
    return str(geometry_status or "").strip() in DEFAULT_ELIGIBLE_STATUSES


def _coarse_points_for_index(state: dict, index: int) -> tuple[np.ndarray, float]:
    flat_value = state.get("object_voxel_keys_flat")
    offsets_value = state.get("object_voxel_keys_offsets")
    levels_value = state.get("object_voxel_levels")
    if flat_value is None or offsets_value is None or levels_value is None:
        return np.zeros((0, 3), dtype=np.float32), float(VOXEL_BASE_V)
    flat = _numpy(flat_value, np.int64).reshape(-1)
    offsets = _numpy(offsets_value, np.int64).reshape(-1)
    levels = _numpy(levels_value, np.int64).reshape(-1)
    if index < 0 or index + 1 >= offsets.size or index >= levels.size:
        return np.zeros((0, 3), dtype=np.float32), float(VOXEL_BASE_V)
    start, end, level = int(offsets[index]), int(offsets[index + 1]), int(levels[index])
    spacing = float(VOXEL_BASE_V) * float(1 << max(level, 0))
    if start < 0 or end <= start or end > flat.size:
        return np.zeros((0, 3), dtype=np.float32), spacing
    points = decode_voxel_keys_numpy(flat[start:end], level)
    return _finite_points(points), spacing


def _load_points_by_view(
    object_id: int,
    mask_root: Path,
    *,
    state_images: list,
    frames_root: Path,
    frames_by_stem: dict[str, dict],
    max_points_per_mask: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    valid_masks = 0
    for path in sorted((mask_root / f"object_{object_id:06d}").glob("*.npz")):
        observation = _observation_points(
            path,
            state_images=state_images,
            frames_root=frames_root,
            frames_by_stem=frames_by_stem,
            max_points=max(0, int(max_points_per_mask)),
            seed=int(seed) + int(object_id) * 1009,
        )
        if observation is None:
            continue
        grouped[int(observation["image_id"])].append(_finite_points(observation["points"]))
        valid_masks += 1
    points_by_view = {
        image_id: np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
        for image_id, point_sets in grouped.items()
        if point_sets
    }
    return points_by_view, {
        "mask_files": len(list((mask_root / f"object_{object_id:06d}").glob("*.npz"))),
        "valid_masks": valid_masks,
        "unique_views": len(points_by_view),
        "raw_points": int(sum(points.shape[0] for points in points_by_view.values())),
    }


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--assembly-mask-dir", type=Path, required=True)
    parser.add_argument("--cloud-npz", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--grid-min-m", type=float, default=0.04)
    parser.add_argument("--grid-max-m", type=float, default=0.08)
    parser.add_argument("--grid-voxel-fraction", type=float, default=0.50)
    parser.add_argument("--min-view-fraction", type=float, default=0.20)
    parser.add_argument("--min-consensus-views", type=int, default=2)
    parser.add_argument("--max-points-per-mask", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.025)
    parser.add_argument("--max-volume-ratio", type=float, default=0.65)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--min-child-fraction", type=float, default=0.15)
    parser.add_argument("--min-child-cells", type=int, default=24)
    parser.add_argument("--min-gaussian-supported-rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260805)
    add_up_arguments(parser, default_axis="y")
    args = parser.parse_args()
    up_policy = policy_from_args(args)

    source = args.scene_state.expanduser().resolve()
    output_state = args.output_state.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    if source == output_state:
        raise ValueError("--output-state must differ from --scene-state")
    for required in (source, args.frames_json, args.assembly_mask_dir, args.cloud_npz):
        path = Path(required).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise TypeError(f"Unsupported scene state: {source}")
    payload = copy.deepcopy(payload)
    state = payload["state"]
    ids = _numpy(state.get("object_id"), np.int64).reshape(-1)
    count = int(ids.size)
    if count == 0:
        raise ValueError("Scene state contains no objects")
    active_value = state.get("active")
    active = _numpy(active_value, bool).reshape(-1).copy()
    if active.size != count:
        raise ValueError("active and object_id lengths differ")
    state_images = list(state.get("images") or [])
    if not state_images:
        raise ValueError("Scene state has no image records")

    members = _ensure_list_field(state, "object_assembly_member_ids", count, list)
    statuses = _ensure_list_field(state, "object_geometry_status", count, lambda: "")
    compound_boxes = _ensure_list_field(state, "object_compound_boxes", count, list)
    consensus_points = _ensure_list_field(
        state,
        "object_geometry_consensus_points",
        count,
        lambda: torch.empty((0, 3), dtype=torch.float32),
    )
    diagnostics = _ensure_list_field(state, "object_compound_geometry_diagnostics", count, dict)

    frames_root, frames_by_stem = _load_frames(args.frames_json.expanduser().resolve())
    mask_root = args.assembly_mask_dir.expanduser().resolve()
    cloud_points, cloud_provenance = _load_cloud(args.cloud_npz.expanduser().resolve())
    tree_started = time.perf_counter()
    cloud_tree = cKDTree(cloud_points)
    tree_seconds = time.perf_counter() - tree_started

    rows: list[dict] = []
    accepted_count = 0
    eligible_count = 0
    legacy_audited_count = 0
    legacy_supported_count = 0
    for index, object_id_value in enumerate(ids.tolist()):
        old_status = str(statuses[index] or "")
        if not _is_assembly(members[index]):
            continue
        coarse_points, voxel_spacing = _coarse_points_for_index(state, index)
        coarse_distances = _query_tree(cloud_tree, coarse_points)
        coarse_gaussian = coarse_voxel_support_metrics(
            coarse_distances,
            voxel_spacing_m=voxel_spacing,
        )
        if _eligible_assembly(members[index], old_status):
            eligibility_reason = "explicit_rejected_or_pending_status"
        elif old_status == LEGACY_UNVALIDATED_STATUS:
            # The legacy builder labelled a coarse union-of-voxels OBB as a
            # pass without raw RGB-D or 3DGS validation.  Re-open only the
            # objectively unsupported rows; leave supported legacy geometry
            # bit-for-bit untouched.
            legacy_audited_count += 1
            if bool(coarse_gaussian["passed"]):
                legacy_supported_count += 1
                continue
            eligibility_reason = "legacy_voxel_support_rejected"
        else:
            continue
        eligible_count += 1
        object_started = time.perf_counter()
        object_id = int(object_id_value)
        active[index] = False
        compound_boxes[index] = []
        consensus_points[index] = torch.empty((0, 3), dtype=torch.float32)

        grid = float(
            np.clip(
                float(args.grid_voxel_fraction) * voxel_spacing,
                float(args.grid_min_m),
                float(args.grid_max_m),
            )
        )
        points_by_view, observation_audit = _load_points_by_view(
            object_id,
            mask_root,
            state_images=state_images,
            frames_root=frames_root,
            frames_by_stem=frames_by_stem,
            max_points_per_mask=int(args.max_points_per_mask),
            seed=int(args.seed),
        )
        unique_views = len(points_by_view)
        min_views = max(
            int(args.min_consensus_views),
            int(math.ceil(float(args.min_view_fraction) * unique_views)),
        )
        points, view_counts = build_multiview_consensus(
            points_by_view,
            grid_m=grid,
            min_views=min_views,
        )
        consensus_points[index] = torch.as_tensor(points, dtype=torch.float32)

        consensus_distances = _query_tree(cloud_tree, points)
        gaussian = gaussian_support_metrics(
            consensus_distances,
            grid_m=grid,
            min_supported_rate=float(args.min_gaussian_supported_rate),
        )
        support_mask = consensus_distances <= float(gaussian["support_radius_m"])

        if unique_views < int(args.min_consensus_views):
            result = {"accepted": False, "reason": "insufficient_unique_views", "candidate_count": 0}
            status = "assembly_compound_geometry_pending"
        elif points.shape[0] < 2 * int(args.min_child_cells):
            result = {"accepted": False, "reason": "insufficient_consensus_cells", "candidate_count": 0}
            status = "assembly_compound_geometry_pending"
        elif not bool(gaussian["passed"]):
            result = {"accepted": False, "reason": "gaussian_support_gate_failed", "candidate_count": 0}
            status = "assembly_compound_geometry_rejected"
        else:
            result = find_binary_compound(
                points,
                gaussian_supported=support_mask,
                quantile=float(args.quantile),
                max_volume_ratio=float(args.max_volume_ratio),
                min_coverage=float(args.min_coverage),
                min_child_fraction=float(args.min_child_fraction),
                min_child_cells=int(args.min_child_cells),
                min_child_gaussian_rate=float(args.min_gaussian_supported_rate),
                up_axis_index=up_policy.axis_index,
                up_vector=up_policy.vector,
            )
            status = (
                "assembly_compound_geometry_probable"
                if bool(result.get("accepted"))
                else "assembly_compound_geometry_rejected"
            )
        if bool(result.get("accepted")):
            compound_boxes[index] = copy.deepcopy(result.get("children") or [])
            accepted_count += 1
        statuses[index] = status

        row = {
            "object_id": object_id,
            "member_ids": [int(value) for value in members[index]],
            "input_geometry_status": old_status,
            "output_geometry_status": status,
            "eligibility_reason": eligibility_reason,
            "active": False,
            "grid_m": grid,
            "minimum_distinct_views": min_views,
            "observation_evidence": observation_audit,
            "consensus_cells": int(points.shape[0]),
            "consensus_view_count_median": (
                float(np.median(view_counts)) if view_counts.size else 0.0
            ),
            "gaussian_support": gaussian,
            "coarse_voxel_gaussian_support": coarse_gaussian,
            "compound": result,
            "duration_seconds": time.perf_counter() - object_started,
        }
        diagnostics[index] = copy.deepcopy(row)
        rows.append(row)

    state["active"] = torch.as_tensor(active, dtype=torch.bool)
    state["object_geometry_status"] = statuses
    state["object_compound_boxes"] = compound_boxes
    state["object_geometry_consensus_points"] = consensus_points
    state["object_compound_geometry_diagnostics"] = diagnostics
    write_state_up_policy(state, up_policy)
    payload["saved_unix_s"] = time.time()
    payload.setdefault("meta", {})["compound_geometry_refinement"] = {
        "source": str(source),
        "automatic": True,
        "semantic_prompts_used": False,
        "eligible_assemblies": eligible_count,
        "probable_compound_assemblies": accepted_count,
        "legacy_voxel_pass_audited": legacy_audited_count,
        "legacy_voxel_pass_supported_untouched": legacy_supported_count,
        "cloud": cloud_provenance,
        "up": up_policy.to_dict(),
    }

    duration = time.perf_counter() - started
    report = {
        "schema": "farm.compound-geometry-audit.v1",
        "created_unix_s": time.time(),
        "source_scene_state": str(source),
        "output_scene_state": str(output_state),
        "cloud": cloud_provenance,
        "up": up_policy.to_dict(),
        "policy": {
            "eligible_statuses": sorted(DEFAULT_ELIGIBLE_STATUSES),
            "legacy_unvalidated_status": LEGACY_UNVALIDATED_STATUS,
            "legacy_policy": "process_only_when_coarse_3dgs_support_gate_fails",
            "semantic_prompts_used": False,
            "grid_min_m": float(args.grid_min_m),
            "grid_max_m": float(args.grid_max_m),
            "grid_voxel_fraction": float(args.grid_voxel_fraction),
            "min_view_fraction": float(args.min_view_fraction),
            "min_consensus_views": int(args.min_consensus_views),
            "quantile": float(args.quantile),
            "max_components": 2,
            "max_volume_ratio": float(args.max_volume_ratio),
            "min_coverage": float(args.min_coverage),
            "min_child_fraction": float(args.min_child_fraction),
            "min_child_cells": int(args.min_child_cells),
            "min_gaussian_supported_rate": float(args.min_gaussian_supported_rate),
            "accepted_status": "assembly_compound_geometry_probable",
            "accepted_active": False,
            "up_axis": up_policy.axis,
            "up_direction": up_policy.direction,
            "up_vector": up_policy.vector.astype(float).tolist(),
            "up_source": up_policy.source,
        },
        "eligible_assemblies": eligible_count,
        "probable_compound_assemblies": accepted_count,
        "rejected_or_pending_assemblies": eligible_count - accepted_count,
        "legacy_voxel_pass_audited": legacy_audited_count,
        "legacy_voxel_pass_supported_untouched": legacy_supported_count,
        "objects": rows,
        "timing": {
            "duration_seconds": duration,
            "cloud_tree_seconds": tree_seconds,
            "stage": "raw_multiview_compound_obb_refinement",
        },
    }
    output_state.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_state)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eligible_assemblies": eligible_count,
                "probable_compound_assemblies": accepted_count,
                "output_state": str(output_state),
                "output_report": str(output_report),
                "duration_seconds": duration,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
