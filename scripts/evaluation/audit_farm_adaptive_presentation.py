#!/usr/bin/env python3
"""Audit final presentation OBB support and route only canonical objects.

The audit is deliberately detector/VLM-free. It recomputes containment against
final presentation OBBs from accumulated voxels and, when available, the
provisional dense Gaussian assignments. Existing held-out reverse-render QC is
an independent signal. The output is an exact allowlist for cheap refit or
bounded full-COLMAP/SAM3 rescue.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from farm_runtime.full_colmap_rescue import quaternion_wxyz_to_matrix
from scene_graph.utils.geometry import decode_voxel_keys_numpy
from tools.farm_shaper_bridge.common import open_graphdeco_ply


DEFORMABLE_HEADS = {"cloth", "fabric", "mat", "rug", "tarp"}


def _numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _state_scalar(state: dict, key: str, index: int) -> float | None:
    values = state.get(key)
    if values is None:
        return None
    array = _numpy(values).reshape(-1)
    return _finite_float(array[index]) if index < array.size else None


def inside_rate(points: np.ndarray, center: np.ndarray, dimensions: np.ndarray, wxyz: np.ndarray) -> float | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not points.size:
        return None
    rotation = quaternion_wxyz_to_matrix(wxyz)
    local = (points - center.reshape(1, 3)) @ rotation
    inside = np.all(np.abs(local) <= 0.5 * dimensions.reshape(1, 3) + 1.0e-6, axis=1)
    return float(np.mean(inside))


def route_object(
    *,
    voxel_inside: float | None,
    gaussian_inside: float | None,
    heldout_status: str,
    heldout_iou: float | None,
    heldout_precision: float | None,
    projected_iou: float | None,
    box_support: float | None,
    deformable: bool,
) -> tuple[str, list[str], float]:
    """Return route, auditable reasons, and deterministic severity."""

    hard_voxel = 0.60 if deformable else 0.70
    leave_voxel = 0.75 if deformable else 0.85
    hard_gaussian = 0.70 if deformable else 0.80
    leave_gaussian = 0.75 if deformable else 0.85
    hard: list[str] = []
    medium: list[str] = []
    severity = 0.0

    if voxel_inside is None:
        hard.append("voxel_support_missing")
        severity += 1.0
    elif voxel_inside < hard_voxel:
        hard.append("voxel_inside_below_hard_threshold")
        severity += hard_voxel - voxel_inside
    elif voxel_inside < leave_voxel:
        medium.append("voxel_inside_below_leave_threshold")
        severity += 0.5 * (leave_voxel - voxel_inside)

    if gaussian_inside is not None:
        if gaussian_inside < hard_gaussian:
            hard.append("gaussian_inside_below_hard_threshold")
            severity += hard_gaussian - gaussian_inside
        elif gaussian_inside < leave_gaussian:
            medium.append("gaussian_inside_below_leave_threshold")
            severity += 0.5 * (leave_gaussian - gaussian_inside)

    if heldout_status == "rejected":
        weak_iou = heldout_iou is not None and heldout_iou < 0.45
        weak_precision = heldout_precision is not None and heldout_precision < 0.65
        if weak_iou or weak_precision:
            hard.append("heldout_reverse_render_rejected")
            severity += max(0.0, 0.45 - (heldout_iou or 0.0))
            severity += max(0.0, 0.65 - (heldout_precision or 0.0))
        else:
            medium.append("heldout_reverse_render_rejected_borderline")
    elif heldout_status == "provisional":
        medium.append("heldout_reverse_render_provisional")

    if projected_iou is not None and projected_iou < 0.35:
        hard.append("projected_box_iou_below_hard_threshold")
        severity += 0.35 - projected_iou
    elif projected_iou is not None and projected_iou < 0.50:
        medium.append("projected_box_iou_below_leave_threshold")
    if box_support is not None and box_support < 0.70:
        hard.append("box_support_below_hard_threshold")
        severity += 0.70 - box_support
    elif box_support is not None and box_support < 0.80:
        medium.append("box_support_below_leave_threshold")

    if hard:
        return "full_colmap_sam3", hard + medium, severity
    if medium:
        return "cheap_refit", medium, severity
    return "leave", ["all_available_quality_gates_pass"], severity


def _heldout_rows(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    return {
        int(row["object_id"]): row
        for row in payload.get("objects", [])
        if isinstance(row, dict) and row.get("object_id") is not None
    }


def _write_ids(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(f"{int(row['object_id'])}\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--presentation-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path)
    parser.add_argument("--gaussian-object-ids", type=Path)
    parser.add_argument("--heldout-qc", type=Path)
    parser.add_argument("--meters-per-scene-unit", type=float, default=1.0)
    parser.add_argument("--maximum-hard-objects", type=int, default=16)
    args = parser.parse_args()
    if (args.gaussian_ply is None) != (args.gaussian_object_ids is None):
        parser.error("--gaussian-ply and --gaussian-object-ids must be supplied together")
    if args.maximum_hard_objects < 1:
        parser.error("--maximum-hard-objects must be positive")
    started = time.perf_counter()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    wrapper = torch.load(args.scene_state.expanduser().resolve(strict=True), map_location="cpu", weights_only=False)
    state = wrapper["state"] if isinstance(wrapper, dict) and isinstance(wrapper.get("state"), dict) else wrapper
    if not isinstance(state, dict):
        raise TypeError("unsupported scene state")
    catalog = json.loads(args.presentation_catalog.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(catalog, list):
        raise TypeError("presentation catalog must be a list")
    ids = _numpy(state["object_id"], np.int64).reshape(-1)
    index_by_id = {int(value): index for index, value in enumerate(ids.tolist())}
    centers = _numpy(state["object_box_centers_m"], np.float64)
    dimensions = _numpy(state["object_box_dimensions_m"], np.float64)
    quaternions = _numpy(state["object_box_wxyz"], np.float64)
    flat = _numpy(state.get("object_voxel_keys_flat", []), np.int64)
    offsets = _numpy(state.get("object_voxel_keys_offsets", []), np.int64)
    levels = _numpy(state.get("object_voxel_levels", []), np.int64)
    heldout = _heldout_rows(args.heldout_qc)

    gaussian_ids = gaussian_table = None
    if args.gaussian_ply is not None:
        gaussian_ids = np.load(args.gaussian_object_ids.expanduser().resolve(strict=True), mmap_mode="r")
        gaussian_table = open_graphdeco_ply(args.gaussian_ply.expanduser().resolve(strict=True))
        if gaussian_ids.shape != (gaussian_table.count,):
            raise ValueError("Gaussian ID array length differs from source PLY vertex count")

    rows: list[dict] = []
    for item in catalog:
        object_id = int(item["id"])
        if object_id not in index_by_id:
            raise ValueError(f"presentation object {object_id} is absent from state")
        index = index_by_id[object_id]
        voxel_points = np.zeros((0, 3), dtype=np.float32)
        if index + 1 < offsets.size:
            start, end = int(offsets[index]), int(offsets[index + 1])
            if 0 <= start < end <= flat.size:
                voxel_points = decode_voxel_keys_numpy(flat[start:end], int(levels[index]))
        voxel_inside = inside_rate(voxel_points, centers[index], dimensions[index], quaternions[index])

        gaussian_count = 0
        gaussian_inside = None
        if gaussian_ids is not None and gaussian_table is not None:
            selected = np.flatnonzero(gaussian_ids == object_id)
            gaussian_count = int(selected.size)
            if selected.size:
                points = np.column_stack(
                    (gaussian_table.data["x"][selected], gaussian_table.data["y"][selected], gaussian_table.data["z"][selected])
                ).astype(np.float32, copy=False)
                points *= float(args.meters_per_scene_unit)
                gaussian_inside = inside_rate(points, centers[index], dimensions[index], quaternions[index])

        held = heldout.get(object_id, {})
        summaries = held.get("summaries") if isinstance(held.get("summaries"), dict) else {}
        heldout_iou = _finite_float((summaries.get("iou") or {}).get("median"))
        heldout_precision = _finite_float((summaries.get("precision") or {}).get("median"))
        category = str(item.get("category") or "unknown").strip().lower()
        deformable = category in DEFORMABLE_HEADS
        projected_iou = _state_scalar(state, "object_geometry_projected_box_iou", index)
        box_support = _state_scalar(state, "object_geometry_box_support_rate", index)
        route, reasons, severity = route_object(
            voxel_inside=voxel_inside,
            gaussian_inside=gaussian_inside,
            heldout_status=str(held.get("status") or "unavailable").lower(),
            heldout_iou=heldout_iou,
            heldout_precision=heldout_precision,
            projected_iou=projected_iou,
            box_support=box_support,
            deformable=deformable,
        )
        rows.append({
            "object_id": object_id,
            "category": category,
            "deformable": deformable,
            "route": route,
            "route_reasons": reasons,
            "severity": severity,
            "voxel_points": int(voxel_points.shape[0]),
            "voxel_inside_rate_final_obb": voxel_inside,
            "provisional_gaussians": gaussian_count,
            "gaussian_inside_rate_final_obb": gaussian_inside,
            "heldout_status": str(held.get("status") or "unavailable").lower(),
            "heldout_median_iou": heldout_iou,
            "heldout_median_precision": heldout_precision,
            "projected_box_iou": projected_iou,
            "box_support_rate": box_support,
            "valid_observations": _state_scalar(state, "object_geometry_valid_observations", index),
            "semantic_status": str((state.get("object_semantic_status") or [""] * len(ids))[index]),
            "semantic_tier": str((state.get("object_semantic_tier") or [""] * len(ids))[index]),
        })

    hard = sorted((row for row in rows if row["route"] == "full_colmap_sam3"), key=lambda row: (-row["severity"], row["object_id"]))
    if len(hard) > args.maximum_hard_objects:
        admitted = {row["object_id"] for row in hard[: args.maximum_hard_objects]}
        for row in rows:
            if row["route"] == "full_colmap_sam3" and row["object_id"] not in admitted:
                row["route"] = "cheap_refit"
                row["route_reasons"].append("full_colmap_budget_demoted_to_cheap_refit")
        hard = hard[: args.maximum_hard_objects]
    medium = sorted((row for row in rows if row["route"] == "cheap_refit"), key=lambda row: (-row["severity"], row["object_id"]))
    leave = sorted((row for row in rows if row["route"] == "leave"), key=lambda row: row["object_id"])
    rows.sort(key=lambda row: row["object_id"])
    duration = time.perf_counter() - started
    report = {
        "schema": "farm.adaptive-presentation-audit.v1",
        "source_scene_state": str(args.scene_state.expanduser().resolve()),
        "presentation_catalog": str(args.presentation_catalog.expanduser().resolve()),
        "policy": {
            "canonical_presentation_first": True,
            "rigid_leave_voxel_inside": 0.85,
            "rigid_hard_voxel_inside": 0.70,
            "rigid_hard_gaussian_inside": 0.80,
            "deformable_leave_voxel_inside": 0.75,
            "deformable_hard_voxel_inside": 0.60,
            "deformable_hard_gaussian_inside": 0.70,
            "maximum_hard_objects": int(args.maximum_hard_objects),
            "detector_or_vlm_calls": 0,
        },
        "summary": {
            "presentation_objects": len(rows),
            "leave": len(leave),
            "cheap_refit": len(medium),
            "full_colmap_sam3": len(hard),
            "gaussian_supported_objects": sum(row["provisional_gaussians"] > 0 for row in rows),
        },
        "timing": {"duration_seconds": duration},
        "objects": rows,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_ids(output / "hard_ids.txt", hard)
    _write_ids(output / "medium_ids.txt", medium)
    _write_ids(output / "leave_ids.txt", leave)
    markdown = [
        "# Canonical-first geometry audit", "",
        f"Objects: {len(rows)}; leave: {len(leave)}; cheap refit: {len(medium)}; full-COLMAP/SAM3: {len(hard)}.", "",
        "| ID | label | route | voxel in OBB | Gaussian in OBB | heldout | reasons |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        voxel = "n/a" if row["voxel_inside_rate_final_obb"] is None else f"{100*row['voxel_inside_rate_final_obb']:.1f}%"
        gaussian = "n/a" if row["gaussian_inside_rate_final_obb"] is None else f"{100*row['gaussian_inside_rate_final_obb']:.1f}%"
        markdown.append(f"| {row['object_id']} | {row['category']} | {row['route']} | {voxel} | {gaussian} | {row['heldout_status']} | {', '.join(row['route_reasons'])} |")
    (output / "AUDIT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "timing": report["timing"], "hard_ids": [row["object_id"] for row in hard], "medium_ids": [row["object_id"] for row in medium]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
