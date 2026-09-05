#!/usr/bin/env python3
"""Validate FARM metric geometry against the verified 3DGS surface cloud.

This stage is deliberately cheap and semantic-free.  Raw RGB-D mask points
and accumulated object voxels are checked independently so coarse voxel
memory cannot silently validate an OBB fitted from the same voxels.
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
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    from scripts.geometry.refine_farm_object_geometry import _load_frames, _numpy, _observation_points
except ModuleNotFoundError:  # Direct script execution adds scripts/ to sys.path.
    from scripts.geometry.refine_farm_object_geometry import _load_frames, _numpy, _observation_points
from scene_graph.utils.geometry import VOXEL_BASE_V, decode_voxel_keys_numpy
from scene_graph.map_update.mask_observations import resolve_object_mask_observations


def gaussian_support_gate(
    *,
    raw_median_m: float,
    voxel_median_m: float,
    voxel_p90_m: float,
    voxel_spacing_m: float,
    max_raw_median_m: float = 0.08,
) -> tuple[bool, dict]:
    """Return a scale-aware, fail-closed surface-support decision."""
    voxel_median_limit = max(0.10, float(voxel_spacing_m))
    voxel_p90_limit = max(0.25, 3.0 * float(voxel_spacing_m))
    raw_ok = math.isfinite(raw_median_m) and raw_median_m <= float(max_raw_median_m)
    voxel_ok = (
        math.isfinite(voxel_median_m)
        and math.isfinite(voxel_p90_m)
        and voxel_median_m <= voxel_median_limit
        and voxel_p90_m <= voxel_p90_limit
    )
    support_signal_count = int(raw_ok) + int(voxel_ok)
    surface_tier = (
        "surface_pass"
        if support_signal_count == 2
        else "surface_borderline"
        if support_signal_count == 1
        else "surface_rejected"
    )
    return raw_ok and voxel_ok, {
        "raw_supported": raw_ok,
        "voxel_supported": voxel_ok,
        "raw_median_limit_m": float(max_raw_median_m),
        "voxel_median_limit_m": voxel_median_limit,
        "voxel_p90_limit_m": voxel_p90_limit,
        "support_signal_count": support_signal_count,
        "surface_tier": surface_tier,
    }


def _distances(tree: cKDTree, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    distances, _ = tree.query(points, k=1, workers=-1)
    return np.asarray(distances, dtype=np.float32)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("inf")


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--direct-mask-root", type=Path, required=True)
    parser.add_argument("--assembly-mask-root", type=Path, required=True)
    parser.add_argument("--surface-cloud", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-points-per-observation", type=int, default=800)
    parser.add_argument("--max-raw-median-m", type=float, default=0.08)
    parser.add_argument("--include-inactive-assemblies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    payload = torch.load(args.scene_state.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise TypeError(f"Unsupported scene state: {args.scene_state}")
    payload = copy.deepcopy(payload)
    state = payload["state"]
    object_ids = _numpy(state["object_id"], np.int64).reshape(-1)
    active = _numpy(state["active"], bool).copy()
    active_input = active.copy()
    count = int(object_ids.size)
    members = state.get("object_assembly_member_ids") or [[] for _ in range(count)]
    images = list(state.get("images") or [])
    frames_root, frames_by_stem = _load_frames(args.frames_json.expanduser().resolve())

    cloud_started = time.perf_counter()
    with np.load(args.surface_cloud.expanduser().resolve(), allow_pickle=False) as cloud:
        xyz = np.asarray(cloud["xyz"], dtype=np.float32)
    tree = cKDTree(xyz)
    tree_seconds = time.perf_counter() - cloud_started

    flat = _numpy(state.get("object_voxel_keys_flat", []), np.int64)
    offsets = _numpy(state.get("object_voxel_keys_offsets", []), np.int64)
    levels = _numpy(state.get("object_voxel_levels", []), np.int64)
    raw_median = np.full((count,), np.inf, dtype=np.float32)
    raw_p90 = np.full((count,), np.inf, dtype=np.float32)
    raw_le_05 = np.zeros((count,), dtype=np.float32)
    voxel_median = np.full((count,), np.inf, dtype=np.float32)
    voxel_p90 = np.full((count,), np.inf, dtype=np.float32)
    support_status = ["not_evaluated"] * count
    rows: list[dict] = []

    evaluation_indices = [
        index
        for index in range(count)
        if bool(active[index])
        or (
            bool(args.include_inactive_assemblies)
            and index < len(members)
            and isinstance(members[index], (list, tuple))
            and len(members[index]) > 0
        )
    ]
    direct_root = args.direct_mask_root.expanduser().resolve()
    assembly_root = args.assembly_mask_root.expanduser().resolve()
    mask_index = resolve_object_mask_observations(state, [direct_root, assembly_root])
    for index in evaluation_indices:
        object_id = int(object_ids[index])
        is_assembly = index < len(members) and isinstance(members[index], (list, tuple)) and len(members[index]) > 0
        point_sets: list[np.ndarray] = []
        for mask_path in mask_index.for_index(index):
            observation = _observation_points(
                mask_path,
                state_images=images,
                frames_root=frames_root,
                frames_by_stem=frames_by_stem,
                max_points=max(0, int(args.max_points_per_observation)),
                seed=int(args.seed) + object_id * 1009,
            )
            if observation is not None:
                point_sets.append(observation["points"])
        raw_points = np.concatenate(point_sets, axis=0) if point_sets else np.zeros((0, 3), dtype=np.float32)
        raw_distances = _distances(tree, raw_points)

        voxel_points = np.zeros((0, 3), dtype=np.float32)
        level = int(levels[index]) if index < levels.size else 0
        if index + 1 < offsets.size:
            start, end = int(offsets[index]), int(offsets[index + 1])
            if 0 <= start < end <= flat.size:
                voxel_points = decode_voxel_keys_numpy(flat[start:end], level)
        voxel_distances = _distances(tree, voxel_points)
        spacing = float(VOXEL_BASE_V) * float(1 << level)
        raw_median[index] = _quantile(raw_distances, 0.5)
        raw_p90[index] = _quantile(raw_distances, 0.9)
        raw_le_05[index] = float(np.mean(raw_distances <= 0.05)) if raw_distances.size else 0.0
        voxel_median[index] = _quantile(voxel_distances, 0.5)
        voxel_p90[index] = _quantile(voxel_distances, 0.9)
        passed, diagnostics = gaussian_support_gate(
            raw_median_m=float(raw_median[index]),
            voxel_median_m=float(voxel_median[index]),
            voxel_p90_m=float(voxel_p90[index]),
            voxel_spacing_m=spacing,
            max_raw_median_m=float(args.max_raw_median_m),
        )
        support_status[index] = str(diagnostics["surface_tier"])
        was_active = bool(active[index])
        if bool(args.enforce) and was_active and not passed:
            active[index] = False
        rows.append(
            {
                "object_id": object_id,
                "is_assembly": is_assembly,
                "assembly_member_ids": list(members[index]) if is_assembly else [],
                "was_active": was_active,
                "active_after_gate": bool(active[index]),
                "status": support_status[index],
                "raw_points": int(raw_points.shape[0]),
                "raw_surface_median_m": float(raw_median[index]),
                "raw_surface_p90_m": float(raw_p90[index]),
                "raw_surface_fraction_le_05m": float(raw_le_05[index]),
                "voxel_points": int(voxel_points.shape[0]),
                "voxel_level": level,
                "voxel_spacing_m": spacing,
                "voxel_surface_median_m": float(voxel_median[index]),
                "voxel_surface_p90_m": float(voxel_p90[index]),
                **diagnostics,
            }
        )

    state["active"] = torch.as_tensor(active, dtype=torch.bool)
    state["object_geometry_surface_status"] = support_status
    state["object_geometry_raw_surface_median_m"] = torch.as_tensor(raw_median, dtype=torch.float32)
    state["object_geometry_raw_surface_p90_m"] = torch.as_tensor(raw_p90, dtype=torch.float32)
    state["object_geometry_raw_surface_fraction_le_05m"] = torch.as_tensor(raw_le_05, dtype=torch.float32)
    state["object_geometry_voxel_surface_median_m"] = torch.as_tensor(voxel_median, dtype=torch.float32)
    state["object_geometry_voxel_surface_p90_m"] = torch.as_tensor(voxel_p90, dtype=torch.float32)
    duration = time.perf_counter() - started
    report = {
        "schema": "farm.gaussian-surface-support.v2",
        "source_scene_state": str(args.scene_state.expanduser().resolve()),
        "surface_cloud": str(args.surface_cloud.expanduser().resolve()),
        "surface_cloud_points": int(xyz.shape[0]),
        "evaluated_objects": len(rows),
        "passed_objects": sum(row["status"] == "surface_pass" for row in rows),
        "borderline_objects": sum(row["status"] == "surface_borderline" for row in rows),
        "rejected_objects": sum(row["status"] == "surface_rejected" for row in rows),
        "active_before": int(sum(row["was_active"] for row in rows)),
        "active_after": int(active.sum()),
        "demoted_active_objects": [row["object_id"] for row in rows if row["was_active"] and not row["active_after_gate"]],
        "diagnostic_nonpass_active_objects": [
            row["object_id"]
            for row in rows
            if row["was_active"] and row["status"] != "surface_pass"
        ],
        "active_state_mutated": not np.array_equal(active, active_input),
        "mask_observation_contract": mask_index.diagnostics,
        "policy": {
            "max_raw_median_m": float(args.max_raw_median_m),
            "voxel_median_limit_m": "max(0.10, voxel_spacing)",
            "voxel_p90_limit_m": "max(0.25, 3*voxel_spacing)",
            "category_agnostic": True,
            "enforce": bool(args.enforce),
        },
        "timing": {"duration_seconds": duration, "tree_build_seconds": tree_seconds},
        "objects": rows,
    }
    payload["state"] = state
    payload["saved_unix_s"] = time.time()
    payload.setdefault("meta", {})["gaussian_surface_support"] = {
        "surface_cloud": str(args.surface_cloud.expanduser().resolve()),
        "evaluated_objects": len(rows),
        "active_after": int(active.sum()),
        "enforce": bool(args.enforce),
        "borderline_objects": report["borderline_objects"],
        "rejected_objects": report["rejected_objects"],
        "demoted_active_objects": report["demoted_active_objects"],
    }
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_state)
    args.output_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("evaluated_objects", "passed_objects", "rejected_objects", "active_after", "demoted_active_objects", "timing")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
