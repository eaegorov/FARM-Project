#!/usr/bin/env python3
"""Refine FARM presentation boxes from their saved multi-view RGB-D evidence.

This is an automatic post-processing stage.  It reconstructs each object's
inlier-mask pixels with the same metric depth and camera poses used by FARM,
rejects inconsistent observations, fits robust world-space boxes, and audits
the result by projecting every refined centre back into its source detections.
The mapping state is never overwritten.
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
import re
import time
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.geometry.farm_geometry_axes import (
        add_up_arguments,
        horizontal_plane_basis,
        normalize_up_vector,
        policy_from_args,
        write_state_up_policy,
    )
except ModuleNotFoundError:  # package import: scripts.geometry.refine_farm_object_geometry
    from scripts.geometry.farm_geometry_axes import (
        add_up_arguments,
        horizontal_plane_basis,
        normalize_up_vector,
        policy_from_args,
        write_state_up_policy,
    )
from scene_graph.utils.geometry import VOXEL_BASE_V, decode_voxel_keys_numpy
from scene_graph.map_update.mask_observations import resolve_object_mask_observations
from farm_runtime.obb_release_evidence import (
    OBBReleaseEvidencePolicy,
    evaluate_obb_release_evidence,
)


IMAGE_ID_RE = re.compile(r"img_(\d+)")


def _record_value(record: object, key: str, default: object = None) -> object:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _load_frames(path: Path) -> tuple[Path, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"No frames list in {path}")
    by_stem: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("rgb_path", "source_image"):
            value = str(row.get(key) or "")
            if value:
                by_stem[Path(value).stem] = row
    return path.parent, by_stem


def _select_observation_paths(
    state: dict,
    index: int,
    resolved_paths: list[Path],
    *,
    preferred_source: str,
    minimum_preferred: int,
) -> tuple[list[Path], dict]:
    """Prefer a complete, provenance-tagged mask subset without weakening fallback."""

    source = str(preferred_source or "").strip()
    all_paths = list(resolved_paths)
    if not source:
        return all_paths, {
            "mode": "all_resolved",
            "preferred_source": None,
            "preferred_records": 0,
            "selected_paths": len(all_paths),
            "fallback_used": False,
        }
    rows = state.get("object_mask_observations")
    row = rows[index] if isinstance(rows, list) and 0 <= index < len(rows) else []
    preferred_tails: set[tuple[str, str]] = set()
    preferred_records = 0
    for record in row if isinstance(row, (list, tuple)) else []:
        if not isinstance(record, dict) or str(record.get("source") or "") != source:
            continue
        text = str(record.get("path") or record.get("mask_path") or "").strip()
        if not text:
            continue
        path = Path(text)
        if len(path.parts) < 2:
            continue
        preferred_tails.add((path.parts[-2], path.parts[-1]))
        preferred_records += 1
    preferred = [
        path for path in all_paths
        if len(path.parts) >= 2 and (path.parts[-2], path.parts[-1]) in preferred_tails
    ]
    threshold = max(1, int(minimum_preferred))
    use_preferred = len(preferred) >= threshold
    return (preferred if use_preferred else all_paths), {
        "mode": "preferred_source" if use_preferred else "all_resolved_fallback",
        "preferred_source": source,
        "preferred_records": preferred_records,
        "preferred_resolved_paths": len(preferred),
        "minimum_preferred_observations": threshold,
        "selected_paths": len(preferred) if use_preferred else len(all_paths),
        "fallback_used": not use_preferred,
    }

def _unpack_mask(data: np.lib.npyio.NpzFile, kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    if f"{kind}_bits" not in data.files:
        return None
    shape = np.asarray(data[f"{kind}_shape"], dtype=np.int32).reshape(2)
    bbox = np.asarray(data[f"{kind}_bbox_xyxy"], dtype=np.int32).reshape(4)
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return None
    flat = np.unpackbits(np.asarray(data[f"{kind}_bits"], dtype=np.uint8), bitorder="little")
    mask = flat[: height * width].reshape(height, width).astype(bool, copy=False)
    return mask, bbox


def _observation_points(
    mask_path: Path,
    *,
    state_images: list,
    frames_root: Path,
    frames_by_stem: dict[str, dict],
    max_points: int,
    seed: int,
) -> dict | None:
    match = IMAGE_ID_RE.search(mask_path.name)
    if match is None:
        return None
    image_id = int(match.group(1))
    if image_id < 0 or image_id >= len(state_images):
        return None
    record = state_images[image_id]
    source_ref = str(_record_value(record, "source_ref", "") or _record_value(record, "storage_path", ""))
    frame = frames_by_stem.get(Path(source_ref).stem)
    if frame is None:
        return None
    depth_rel = str(frame.get("depth_path") or "")
    if not depth_rel:
        return None
    depth_path = frames_root / depth_rel
    if not depth_path.is_file():
        return None
    pose = _numpy(_record_value(record, "pose"), np.float64)
    intrinsics = np.asarray(frame.get("K"), dtype=np.float64)
    if pose.shape != (4, 4) or intrinsics.shape != (3, 3):
        return None

    with np.load(mask_path, allow_pickle=False) as data:
        packed = _unpack_mask(data, "inlier") or _unpack_mask(data, "raw")
        if packed is None:
            return None
        mask, bbox = packed
        raw_bbox = np.asarray(data["raw_bbox_xyxy"], dtype=np.float64).reshape(4)
    depth = np.load(depth_path, mmap_mode="r")
    if depth.ndim != 2:
        return None
    x0, y0, x1, y1 = [int(v) for v in bbox]
    height = min(mask.shape[0], max(0, depth.shape[0] - y0), max(0, y1 - y0))
    width = min(mask.shape[1], max(0, depth.shape[1] - x0), max(0, x1 - x0))
    if height <= 0 or width <= 0:
        return None
    ys_local, xs_local = np.nonzero(mask[:height, :width])
    ys, xs = ys_local + y0, xs_local + x0
    z = np.asarray(depth[ys, xs], dtype=np.float32)
    valid = np.isfinite(z) & (z > 0.05) & (z < 80.0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if z.size < 24:
        return None
    if max_points > 0 and z.size > max_points:
        rng = np.random.default_rng(seed + image_id * 104729)
        keep = rng.choice(z.size, size=max_points, replace=False)
        ys, xs, z = ys[keep], xs[keep], z[keep]

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points_cam = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
    points_world = points_cam @ pose[:3, :3].T + pose[:3, 3]
    finite = np.isfinite(points_world).all(axis=1)
    points_world = points_world[finite].astype(np.float32, copy=False)
    if points_world.shape[0] < 24:
        return None
    return {
        "image_id": image_id,
        "physical_timestamp": str(frame.get("physical_timestamp") or "").strip(),
        "points": points_world,
        "centroid": np.median(points_world, axis=0),
        "raw_bbox": raw_bbox,
        "pose": pose,
        "K": intrinsics,
    }


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a deterministic unit quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray([0.25 * scale, (m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            scale = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1.0e-12)) * 2.0
            quat = np.asarray([(m[2, 1] - m[1, 2]) / scale, 0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
        elif axis == 1:
            scale = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1.0e-12)) * 2.0
            quat = np.asarray([(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale])
        else:
            scale = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1.0e-12)) * 2.0
            quat = np.asarray([(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale])
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat.astype(np.float32)


def _fit_robust_obb(
    points: np.ndarray,
    quantile: float,
    *,
    orientation_mode: str,
    up_axis_index: int = 1,
    up_vector: object | None = None,
) -> dict:
    """Fit a deterministic robust OBB with either free 3D or gravity-yaw orientation."""
    origin = np.median(points, axis=0).astype(np.float64)
    centered = points.astype(np.float64) - origin
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    if up_vector is None:
        up = np.zeros((3,), dtype=np.float64)
        up[int(up_axis_index)] = 1.0
    else:
        up = normalize_up_vector(up_vector)
    if orientation_mode == "gravity_yaw":
        plane_basis = horizontal_plane_basis(up)
        plane_points = centered @ plane_basis
        plane_covariance = plane_points.T @ plane_points / max(plane_points.shape[0] - 1, 1)
        plane_eigenvalues, plane_eigenvectors = np.linalg.eigh(plane_covariance)
        plane_order = np.argsort(plane_eigenvalues)[::-1]
        plane_eigenvalues = plane_eigenvalues[plane_order]
        yaw_vector = plane_eigenvectors[:, plane_order[0]]
        horizontal_major = plane_basis @ yaw_vector
        major = int(np.argmax(np.abs(horizontal_major)))
        if horizontal_major[major] < 0.0:
            horizontal_major *= -1.0
        horizontal_major /= max(float(np.linalg.norm(horizontal_major)), 1.0e-12)
        horizontal_minor = np.cross(up, horizontal_major)
        horizontal_minor /= max(float(np.linalg.norm(horizontal_minor)), 1.0e-12)
        # Local dimensions are length × width × height.  The third OBB axis is
        # exactly the scene gravity axis, so every face is parallel or normal
        # to the floor while yaw remains data-driven.
        basis = np.column_stack((horizontal_major, horizontal_minor, up))
        if np.linalg.det(basis) < 0.0:
            basis[:, 1] *= -1.0
        eigenvalues = np.diag(basis.T @ covariance @ basis)
        orientation_confidence = float(
            np.clip(
                (plane_eigenvalues[0] - plane_eigenvalues[1])
                / max(float(plane_eigenvalues[0]), 1.0e-12),
                0.0,
                1.0,
            )
        )
    else:
        eigenvalues, basis = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, basis = eigenvalues[order], basis[:, order]
        for column in range(3):
            major = int(np.argmax(np.abs(basis[:, column])))
            if basis[major, column] < 0.0:
                basis[:, column] *= -1.0
        if np.linalg.det(basis) < 0.0:
            basis[:, 2] *= -1.0
        long_gap = float((eigenvalues[0] - eigenvalues[1]) / max(eigenvalues[0], 1.0e-12))
        plane_gap = float((eigenvalues[1] - eigenvalues[2]) / max(eigenvalues[1], 1.0e-12))
        orientation_confidence = float(np.clip(min(long_gap, plane_gap), 0.0, 1.0))
    local = centered @ basis
    q = float(np.clip(quantile, 0.0, 0.20))
    lo, hi = np.quantile(local, [q, 1.0 - q], axis=0)
    dimensions = np.maximum(hi - lo, 0.08)
    padding = np.maximum(0.025, dimensions * 0.025)
    dimensions = dimensions + 2.0 * padding
    local_center = 0.5 * (lo + hi)
    center = origin + basis @ local_center
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    corners = center + (signs * (0.5 * dimensions)) @ basis.T
    vertical_alignment = float(np.max(np.abs(basis.T @ up)))
    gravity_tilt_degrees = float(np.degrees(np.arccos(np.clip(vertical_alignment, 0.0, 1.0))))
    return {
        "center": center.astype(np.float32),
        "dimensions": dimensions.astype(np.float32),
        "rotation_matrix": basis.astype(np.float32),
        "wxyz": _matrix_to_wxyz(basis),
        "corners": corners.astype(np.float32),
        "orientation_confidence": orientation_confidence,
        "orientation_mode": orientation_mode,
        "up_axis_index": int(up_axis_index),
        "up_vector": up.astype(np.float32),
        "gravity_alignment": vertical_alignment,
        "gravity_tilt_degrees": gravity_tilt_degrees,
        "eigenvalues": eigenvalues.astype(np.float32),
    }


def _obb_from_local_extents(base: dict, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Keep an RGB-D orientation while changing its metric centre/extents."""
    rotation = np.asarray(base["rotation_matrix"], dtype=np.float64)
    local_center = 0.5 * (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64))
    dimensions = np.maximum(np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64), 0.08)
    center = np.asarray(base["center"], dtype=np.float64) + rotation @ local_center
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    candidate = dict(base)
    candidate.update({
        "center": center.astype(np.float32),
        "dimensions": dimensions.astype(np.float32),
        "corners": (center + (signs * (0.5 * dimensions)) @ rotation.T).astype(np.float32),
    })
    return candidate


def _yaw_align_planar_normal(
    base: dict,
    consensus_normal: np.ndarray,
    *,
    normal_axis: int,
    maximum_gravity_elevation_degrees: float,
) -> tuple[dict, dict]:
    """Rotate a gravity-constrained OBB normal toward train RGB-D consensus.

    The correction is deliberately narrow: only yaw about the already declared
    gravity vector may change. Centre and dimensions remain the caller's
    responsibility. Free-3D fits, horizontal surfaces and normals inconsistent
    with gravity are left untouched and reported as such.
    """

    rotation = np.asarray(base["rotation_matrix"], dtype=np.float64)
    normal = np.asarray(consensus_normal, dtype=np.float64).reshape(3)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    up = normalize_up_vector(base.get("up_vector", [0.0, 1.0, 0.0]))
    vertical_axis = int(np.argmax(np.abs(rotation.T @ up)))
    common = {
        "mode": "train_rgbd_consensus_normal_yaw_v1",
        "applied": False,
        "normal_axis": int(normal_axis),
        "vertical_axis": vertical_axis,
        "yaw_correction_degrees": 0.0,
        "consensus_gravity_elevation_degrees": float(
            np.degrees(np.arcsin(np.clip(abs(float(np.dot(normal, up))), 0.0, 1.0)))
        ),
    }
    if str(base.get("orientation_mode") or "") != "gravity_yaw":
        return base, {
            **common,
            "status": "not_applicable",
            "reason": "orientation_is_not_gravity_yaw",
        }
    if int(normal_axis) not in (0, 1, 2):
        return base, {**common, "status": "rejected", "reason": "normal_axis_invalid"}
    if vertical_axis == int(normal_axis):
        return base, {
            **common,
            "status": "not_applicable",
            "reason": "planar_normal_is_gravity_axis",
        }
    elevation = float(common["consensus_gravity_elevation_degrees"])
    if elevation > float(maximum_gravity_elevation_degrees):
        return base, {
            **common,
            "status": "rejected",
            "reason": "consensus_normal_not_gravity_compatible",
        }

    current = rotation[:, int(normal_axis)]
    current_horizontal = current - up * float(np.dot(current, up))
    target_horizontal = normal - up * float(np.dot(normal, up))
    current_norm = float(np.linalg.norm(current_horizontal))
    target_norm = float(np.linalg.norm(target_horizontal))
    if current_norm <= 1.0e-9 or target_norm <= 1.0e-9:
        return base, {
            **common,
            "status": "rejected",
            "reason": "horizontal_normal_projection_degenerate",
        }
    current_horizontal /= current_norm
    target_horizontal /= target_norm
    if float(np.dot(current_horizontal, target_horizontal)) < 0.0:
        target_horizontal *= -1.0
    cosine = float(np.clip(np.dot(current_horizontal, target_horizontal), -1.0, 1.0))
    sine = float(np.dot(up, np.cross(current_horizontal, target_horizontal)))
    angle = float(np.arctan2(sine, cosine))
    skew = np.asarray(
        [
            [0.0, -up[2], up[1]],
            [up[2], 0.0, -up[0]],
            [-up[1], up[0], 0.0],
        ],
        dtype=np.float64,
    )
    yaw = (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    aligned_rotation = yaw @ rotation
    if not np.allclose(
        aligned_rotation.T @ aligned_rotation, np.eye(3), atol=1.0e-6
    ) or not np.isclose(np.linalg.det(aligned_rotation), 1.0, atol=1.0e-6):
        return base, {
            **common,
            "status": "rejected",
            "reason": "aligned_rotation_invalid",
        }

    candidate = dict(base)
    candidate["rotation_matrix"] = aligned_rotation.astype(np.float32)
    candidate["wxyz"] = _matrix_to_wxyz(aligned_rotation)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    center = np.asarray(candidate["center"], dtype=np.float64)
    dimensions = np.asarray(candidate["dimensions"], dtype=np.float64)
    candidate["corners"] = (
        center + (signs * (0.5 * dimensions)) @ aligned_rotation.T
    ).astype(np.float32)
    candidate["gravity_alignment"] = float(
        np.max(np.abs(aligned_rotation.T @ up))
    )
    candidate["gravity_tilt_degrees"] = float(
        np.degrees(
            np.arccos(np.clip(candidate["gravity_alignment"], 0.0, 1.0))
        )
    )
    return candidate, {
        **common,
        "status": "candidate",
        "applied": True,
        "yaw_correction_degrees": float(np.degrees(angle)),
        "post_alignment_normal_axis_degrees": float(
            np.degrees(
                np.arccos(
                    np.clip(
                        abs(
                            float(
                                np.dot(
                                    aligned_rotation[:, int(normal_axis)], normal
                                )
                            )
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
        ),
    }


def _voxel_inside_rate(obb: dict, voxel_points: np.ndarray) -> float:
    if voxel_points.shape[0] == 0:
        return 0.0
    rotation = np.asarray(obb["rotation_matrix"], dtype=np.float64)
    local = (voxel_points.astype(np.float64) - np.asarray(obb["center"], dtype=np.float64)) @ rotation
    half = 0.5 * np.asarray(obb["dimensions"], dtype=np.float64)
    return float(np.mean(np.all(np.abs(local) <= half + 1.0e-6, axis=1)))


def _voxel_in_plane_inside_rate(
    obb: dict,
    voxel_points: np.ndarray,
    *,
    normal_axis: int,
) -> float:
    """Audit legacy voxel support without trusting depth along a proven plane normal."""

    points = np.asarray(voxel_points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0 or int(normal_axis) not in (0, 1, 2):
        return 0.0
    rotation = np.asarray(obb["rotation_matrix"], dtype=np.float64)
    local = (points - np.asarray(obb["center"], dtype=np.float64)) @ rotation
    half = 0.5 * np.asarray(obb["dimensions"], dtype=np.float64)
    axes = [axis for axis in range(3) if axis != int(normal_axis)]
    return float(np.mean(np.all(np.abs(local[:, axes]) <= half[axes] + 1.0e-6, axis=1)))


def _planar_surface_evidence(
    base: dict,
    observations: list[dict],
    *,
    support_iou: float,
    min_independent_views: int,
    min_planar_view_rate: float,
    max_plane_eigenvalue_ratio: float,
    min_in_plane_eigenvalue_ratio: float,
    max_normal_angle_q90_degrees: float,
    max_median_view_thickness_m: float,
    max_q90_view_thickness_m: float,
    max_median_view_thickness_ratio: float,
    max_plane_offset_spread_m: float,
    max_normal_axis_angle_degrees: float,
    min_projected_box_iou: float,
    min_box_support_rate: float,
    max_constrained_thickness_m: float,
    min_alignment_reprojection_retention: float = 0.85,
) -> tuple[dict, dict]:
    """Prove a thin surface from independent RGB-D views, then bound its thickness.

    This is deliberately category-free. A view contributes only when its own
    depth samples form a 2D surface rather than a line or a volume. Multiple
    cameras at one physical timestamp count once. The returned OBB is changed
    only after the constrained box itself passes the projection gates.
    """

    view_rows: list[dict] = []
    missing_timestamps = 0
    for observation in observations:
        timestamp = str(observation.get("physical_timestamp") or "").strip()
        if not timestamp:
            missing_timestamps += 1
            continue
        points = np.asarray(observation.get("points"), dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 24:
            continue
        origin = np.median(points, axis=0)
        centered = points - origin
        covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        smallest, middle, largest = [float(value) for value in eigenvalues]
        if middle <= 1.0e-12 or largest <= 1.0e-12:
            continue
        normal = eigenvectors[:, 0]
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        normal_values = centered @ normal
        in_plane_values = centered @ eigenvectors[:, 1:]
        thickness = float(
            np.quantile(normal_values, 0.90) - np.quantile(normal_values, 0.10)
        )
        in_plane_extents = np.quantile(in_plane_values, 0.90, axis=0) - np.quantile(
            in_plane_values, 0.10, axis=0
        )
        minor_extent = max(float(np.min(in_plane_extents)), 1.0e-9)
        row = {
            "physical_timestamp": timestamp,
            "normal": normal,
            "plane_center_m": origin,
            "planarity_eigenvalue_ratio": smallest / middle,
            "in_plane_eigenvalue_ratio": middle / largest,
            "depth_thickness_m": thickness,
            "depth_thickness_ratio": thickness / minor_extent,
            "in_plane_minor_extent_m": minor_extent,
        }
        row["locally_planar"] = bool(
            row["planarity_eigenvalue_ratio"] <= float(max_plane_eigenvalue_ratio)
            and row["in_plane_eigenvalue_ratio"] >= float(min_in_plane_eigenvalue_ratio)
        )
        view_rows.append(row)

    by_timestamp: dict[str, list[dict]] = {}
    for row in view_rows:
        by_timestamp.setdefault(str(row["physical_timestamp"]), []).append(row)
    timestamp_rows = [
        min(
            rows,
            key=lambda row: (
                not bool(row["locally_planar"]),
                float(row["planarity_eigenvalue_ratio"]),
                float(row["depth_thickness_ratio"]),
            ),
        )
        for _, rows in sorted(by_timestamp.items())
    ]
    planar_rows = [row for row in timestamp_rows if bool(row["locally_planar"])]
    independent_views = len(timestamp_rows)
    planar_view_rate = float(len(planar_rows) / max(independent_views, 1))

    reasons: list[str] = []
    if independent_views < int(min_independent_views):
        reasons.append("insufficient_independent_physical_timestamps")
    if planar_view_rate < float(min_planar_view_rate):
        reasons.append("insufficient_locally_planar_view_rate")

    normal_axis = -1
    consensus_normal = np.zeros((3,), dtype=np.float64)
    normal_angle_q90 = float("inf")
    median_thickness = float("inf")
    q90_thickness = float("inf")
    median_thickness_ratio = float("inf")
    plane_offset_spread = float("inf")
    normal_axis_angle = float("inf")
    constrained = base
    constrained_projection = {"median_box_iou": 0.0, "box_support_rate": 0.0}
    unaligned_projection = {
        "median_box_iou": 0.0,
        "q25_box_iou": 0.0,
        "box_support_rate": 0.0,
    }
    alignment_diagnostics = {
        "mode": "train_rgbd_consensus_normal_yaw_v1",
        "status": "not_evaluated",
        "applied": False,
        "reason": "no_planar_rows",
    }
    alignment_materiality_degrees = float("inf")
    alignment_projection_retention = 0.0
    target_thickness = float("nan")

    if planar_rows:
        scatter = sum(
            np.outer(np.asarray(row["normal"]), np.asarray(row["normal"]))
            for row in planar_rows
        )
        _, consensus_vectors = np.linalg.eigh(scatter)
        consensus_normal = consensus_vectors[:, -1]
        consensus_normal /= max(float(np.linalg.norm(consensus_normal)), 1.0e-12)
        angles = np.degrees(
            np.arccos(
                np.clip(
                    [
                        abs(float(np.dot(row["normal"], consensus_normal)))
                        for row in planar_rows
                    ],
                    0.0,
                    1.0,
                )
            )
        )
        normal_angle_q90 = float(np.quantile(angles, 0.90))
        thicknesses = np.asarray(
            [float(row["depth_thickness_m"]) for row in planar_rows], dtype=np.float64
        )
        thickness_ratios = np.asarray(
            [float(row["depth_thickness_ratio"]) for row in planar_rows],
            dtype=np.float64,
        )
        median_thickness = float(np.median(thicknesses))
        q90_thickness = float(np.quantile(thicknesses, 0.90))
        median_thickness_ratio = float(np.median(thickness_ratios))
        rotation = np.asarray(base["rotation_matrix"], dtype=np.float64)
        alignments = np.abs(rotation.T @ consensus_normal)
        normal_axis = int(np.argmax(alignments))
        normal_axis_angle = float(
            np.degrees(np.arccos(np.clip(alignments[normal_axis], 0.0, 1.0)))
        )
        axis = rotation[:, normal_axis]
        plane_offsets = np.asarray(
            [
                float(
                    np.dot(
                        np.asarray(row["plane_center_m"], dtype=np.float64)
                        - np.asarray(base["center"], dtype=np.float64),
                        axis,
                    )
                )
                for row in planar_rows
            ],
            dtype=np.float64,
        )
        plane_offset_spread = float(
            np.quantile(plane_offsets, 0.90) - np.quantile(plane_offsets, 0.10)
        )
        median_minor_extent = float(
            np.median([row["in_plane_minor_extent_m"] for row in planar_rows])
        )
        ratio_cap = max(
            0.08, float(max_median_view_thickness_ratio) * median_minor_extent
        )
        target_thickness = min(
            max(0.08, median_thickness + 0.05),
            float(max_constrained_thickness_m),
            ratio_cap,
        )
        unaligned_base = base
        base_half = 0.5 * np.asarray(base["dimensions"], dtype=np.float64)
        lo, hi = -base_half.copy(), base_half.copy()
        normal_center = float(np.median(plane_offsets))
        lo[normal_axis] = normal_center - 0.5 * target_thickness
        hi[normal_axis] = normal_center + 0.5 * target_thickness
        unaligned_candidate = _obb_from_local_extents(unaligned_base, lo, hi)
        unaligned_projection, _ = _projection_metrics(
            unaligned_candidate["center"],
            unaligned_candidate["corners"],
            observations,
            support_iou=support_iou,
        )

        aligned_base, alignment_diagnostics = _yaw_align_planar_normal(
            base,
            consensus_normal,
            normal_axis=normal_axis,
            maximum_gravity_elevation_degrees=max_normal_angle_q90_degrees,
        )
        in_plane_axes = [value for value in range(3) if value != normal_axis]
        alignment_materiality_degrees = float(
            np.degrees(
                np.arctan2(
                    max(target_thickness, 1.0e-9),
                    max(
                        float(
                            np.max(np.asarray(base["dimensions"])[in_plane_axes])
                        ),
                        1.0e-9,
                    ),
                )
            )
        )
        yaw_correction = abs(
            float(alignment_diagnostics.get("yaw_correction_degrees", 0.0))
        )
        use_alignment = bool(
            alignment_diagnostics.get("status") == "candidate"
            and yaw_correction > alignment_materiality_degrees
        )
        if alignment_diagnostics.get("status") == "candidate" and not use_alignment:
            alignment_diagnostics = {
                **alignment_diagnostics,
                "status": "not_material",
                "applied": False,
                "reason": "yaw_error_below_thickness_materiality_angle",
            }
        working_base = aligned_base if use_alignment else unaligned_base
        working_rotation = np.asarray(
            working_base["rotation_matrix"], dtype=np.float64
        )
        working_axis = working_rotation[:, normal_axis]
        working_offsets = np.asarray(
            [
                float(
                    np.dot(
                        np.asarray(row["plane_center_m"], dtype=np.float64)
                        - np.asarray(working_base["center"], dtype=np.float64),
                        working_axis,
                    )
                )
                for row in planar_rows
            ],
            dtype=np.float64,
        )
        working_half = 0.5 * np.asarray(
            working_base["dimensions"], dtype=np.float64
        )
        lo, hi = -working_half.copy(), working_half.copy()
        working_normal_center = float(np.median(working_offsets))
        lo[normal_axis] = working_normal_center - 0.5 * target_thickness
        hi[normal_axis] = working_normal_center + 0.5 * target_thickness
        candidate = _obb_from_local_extents(working_base, lo, hi)
        constrained_projection, _ = _projection_metrics(
            candidate["center"],
            candidate["corners"],
            observations,
            support_iou=support_iou,
        )
        baseline_iou = max(
            float(unaligned_projection["median_box_iou"]), 1.0e-12
        )
        alignment_projection_retention = float(
            float(constrained_projection["median_box_iou"]) / baseline_iou
        )
        if use_alignment and alignment_projection_retention < float(
            min_alignment_reprojection_retention
        ):
            reasons.append("consensus_aligned_obb_reprojection_regression")
        constrained = candidate

    if normal_angle_q90 > float(max_normal_angle_q90_degrees):
        reasons.append("unstable_plane_normals")
    if median_thickness > float(max_median_view_thickness_m):
        reasons.append("median_per_view_depth_thickness_exceeds_limit")
    if q90_thickness > float(max_q90_view_thickness_m):
        reasons.append("q90_per_view_depth_thickness_exceeds_limit")
    if median_thickness_ratio > float(max_median_view_thickness_ratio):
        reasons.append("relative_per_view_depth_thickness_exceeds_limit")
    if plane_offset_spread > float(max_plane_offset_spread_m):
        reasons.append("cross_view_plane_offset_spread_exceeds_limit")
    if normal_axis_angle > float(max_normal_axis_angle_degrees):
        reasons.append("plane_normal_not_aligned_to_obb_axis")
    if float(constrained_projection["median_box_iou"]) < float(min_projected_box_iou):
        reasons.append("constrained_obb_reprojection_iou_below_limit")
    if float(constrained_projection["box_support_rate"]) < float(min_box_support_rate):
        reasons.append("constrained_obb_view_support_below_limit")

    accepted = not reasons
    diagnostics = {
        "schema": "farm.planar-rgbd-thickness-evidence.v1",
        "accepted": accepted,
        "policy": (
            "planar_rgbd_normal_axis_exception_v1"
            if accepted
            else "full_3d_voxel_retention"
        ),
        "reasons": reasons,
        "observation_views": len(view_rows),
        "missing_physical_timestamp_views": missing_timestamps,
        "independent_physical_timestamps": independent_views,
        "locally_planar_timestamps": len(planar_rows),
        "planar_view_rate": planar_view_rate,
        "normal_axis": normal_axis,
        "consensus_normal": consensus_normal.astype(float).tolist(),
        "normal_angle_q90_degrees": normal_angle_q90,
        "median_per_view_depth_thickness_m": median_thickness,
        "q90_per_view_depth_thickness_m": q90_thickness,
        "median_per_view_depth_thickness_ratio": median_thickness_ratio,
        "cross_view_plane_offset_spread_m": plane_offset_spread,
        "normal_axis_angle_degrees": normal_axis_angle,
        "normal_alignment": {
            **alignment_diagnostics,
            "materiality_angle_degrees": alignment_materiality_degrees,
            "minimum_reprojection_retention": float(
                min_alignment_reprojection_retention
            ),
            "median_reprojection_iou_retention": alignment_projection_retention,
            "unaligned_median_projected_box_iou": float(
                unaligned_projection["median_box_iou"]
            ),
            "aligned_median_projected_box_iou": float(
                constrained_projection["median_box_iou"]
            ),
        },
        "constrained_thickness_m": target_thickness,
        "constrained_median_projected_box_iou": float(
            constrained_projection["median_box_iou"]
        ),
        "constrained_box_support_rate": float(
            constrained_projection["box_support_rate"]
        ),
        "thresholds": {
            "min_independent_views": int(min_independent_views),
            "min_planar_view_rate": float(min_planar_view_rate),
            "max_plane_eigenvalue_ratio": float(max_plane_eigenvalue_ratio),
            "min_in_plane_eigenvalue_ratio": float(min_in_plane_eigenvalue_ratio),
            "max_normal_angle_q90_degrees": float(max_normal_angle_q90_degrees),
            "max_median_view_thickness_m": float(max_median_view_thickness_m),
            "max_q90_view_thickness_m": float(max_q90_view_thickness_m),
            "max_median_view_thickness_ratio": float(max_median_view_thickness_ratio),
            "max_plane_offset_spread_m": float(max_plane_offset_spread_m),
            "max_normal_axis_angle_degrees": float(max_normal_axis_angle_degrees),
            "min_projected_box_iou": float(min_projected_box_iou),
            "min_box_support_rate": float(min_box_support_rate),
            "max_constrained_thickness_m": float(max_constrained_thickness_m),
            "min_alignment_reprojection_retention": float(
                min_alignment_reprojection_retention
            ),
        },
        "views": [
            {
                key: value
                for key, value in row.items()
                if key not in {"normal", "plane_center_m"}
            }
            for row in timestamp_rows
        ],
    }
    return (constrained if accepted else base), diagnostics


def _anchored_voxel_component(
    voxel_points: np.ndarray,
    base: dict,
    *,
    voxel_size_m: float,
    neighborhood_steps: int = 2,
    anchor_shell_m: float = 0.08,
) -> tuple[np.ndarray, dict]:
    """Keep the coherent voxel component anchored to the RGB-D object surface.

    A distant floater or background fragment must not enlarge the OBB merely
    because it belongs to the same historical FARM object ID. Components are
    formed on the native voxel lattice, then ranked by overlap with a small
    metric shell around the independently fitted RGB-D box.
    """

    points = np.asarray(voxel_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if (
        points.shape[0] == 0
        or not math.isfinite(float(voxel_size_m))
        or float(voxel_size_m) <= 0.0
    ):
        return np.zeros((0, 3), dtype=np.float64), {
            "voxel_points": int(points.shape[0]),
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "selected_component_fraction": 0.0,
            "selected_component_anchor_points": 0,
            "anchor_shell_m": float(anchor_shell_m),
            "neighborhood_steps": int(neighborhood_steps),
        }
    steps = max(1, int(neighborhood_steps))
    voxel_size = float(voxel_size_m)
    grid = np.rint((points - 0.5 * voxel_size) / voxel_size).astype(np.int64)
    coordinate_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for index, coordinate in enumerate(grid.tolist()):
        coordinate_to_indices.setdefault(tuple(map(int, coordinate)), []).append(index)

    parents = np.arange(points.shape[0], dtype=np.int64)
    sizes = np.ones(points.shape[0], dtype=np.int64)

    def find(index: int) -> int:
        while int(parents[index]) != index:
            parents[index] = parents[int(parents[index])]
            index = int(parents[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if int(sizes[left_root]) < int(sizes[right_root]):
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    offsets = [
        (dx, dy, dz)
        for dx in range(-steps, steps + 1)
        for dy in range(-steps, steps + 1)
        for dz in range(-steps, steps + 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    for coordinate, indices in coordinate_to_indices.items():
        first = indices[0]
        for duplicate in indices[1:]:
            union(first, duplicate)
        for dx, dy, dz in offsets:
            neighbor = (
                coordinate[0] + dx,
                coordinate[1] + dy,
                coordinate[2] + dz,
            )
            for other in coordinate_to_indices.get(neighbor, ()):
                union(first, other)

    components: dict[int, list[int]] = {}
    for index in range(points.shape[0]):
        components.setdefault(find(index), []).append(index)

    rotation = np.asarray(base["rotation_matrix"], dtype=np.float64)
    local = (points - np.asarray(base["center"], dtype=np.float64)) @ rotation
    half = 0.5 * np.asarray(base["dimensions"], dtype=np.float64)
    outside = np.maximum(np.abs(local) - half[None, :], 0.0)
    distance_to_base = np.linalg.norm(outside, axis=1)
    shell = max(float(anchor_shell_m), 2.0 * voxel_size)
    ranked: list[tuple[tuple, list[int], int, float]] = []
    for indices in components.values():
        values = np.asarray(indices, dtype=np.int64)
        anchor_points = int(np.count_nonzero(distance_to_base[values] <= shell))
        median_distance = float(np.median(distance_to_base[values]))
        key = (
            anchor_points > 0,
            anchor_points,
            len(indices),
            -median_distance,
        )
        ranked.append((key, indices, anchor_points, median_distance))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected_indices = ranked[0][1] if ranked else []
    selected_anchor_points = ranked[0][2] if ranked else 0
    largest_size = max((len(indices) for indices in components.values()), default=0)
    selected = points[np.asarray(selected_indices, dtype=np.int64)]
    total = max(points.shape[0], 1)
    return selected, {
        "voxel_points": int(points.shape[0]),
        "component_count": len(components),
        "largest_component_fraction": float(largest_size / total),
        "selected_component_fraction": float(len(selected_indices) / total),
        "selected_component_anchor_points": int(selected_anchor_points),
        "selected_component_points": len(selected_indices),
        "anchor_shell_m": shell,
        "neighborhood_steps": steps,
    }


def _select_voxel_supported_obb(
    base: dict,
    voxel_points: np.ndarray,
    observations: list[dict],
    *,
    voxel_size_m: float,
    support_iou: float,
    min_support_rate: float,
    min_projected_iou: float,
    expansion_penalty: float,
    score_tie_tolerance: float,
    planar_evidence: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Choose the most reprojection-accurate RGB-D/voxel extent candidate.

    Object names never enter this decision.  RGB-D fixes orientation and a
    minimum extent; FARM voxel-memory proposes progressively wider robust
    extents.  Full multi-view reprojection remains the primary selector.
    """
    points = np.asarray(voxel_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    planar = planar_evidence if isinstance(planar_evidence, dict) else {}
    planar_exception = bool(planar.get("accepted", False))
    normal_axis = int(planar.get("normal_axis", -1))
    retention_policy = str(planar.get("policy", "full_3d_voxel_retention"))
    expansion_points, component_diagnostics = _anchored_voxel_component(
        points, base, voxel_size_m=float(voxel_size_m)
    )
    base_volume = max(float(np.prod(base["dimensions"])), 1.0e-9)
    candidates: list[dict] = []

    def add_candidate(label: str, obb: dict) -> None:
        projection, _ = _projection_metrics(
            obb["center"], obb["corners"], observations, support_iou=support_iou
        )
        voxel_inside_rate = _voxel_inside_rate(obb, points)
        voxel_in_plane_inside_rate = _voxel_in_plane_inside_rate(
            obb, points, normal_axis=normal_axis
        )
        candidates.append(
            {
                "label": label,
                "obb": obb,
                "projection": projection,
                "voxel_inside_rate": voxel_inside_rate,
                "voxel_in_plane_inside_rate": voxel_in_plane_inside_rate,
                "voxel_retention_gate_rate": (
                    voxel_in_plane_inside_rate
                    if planar_exception
                    else voxel_inside_rate
                ),
                "volume_expansion": float(np.prod(obb["dimensions"]) / base_volume),
            }
        )

    add_candidate("rgbd_planar_constrained" if planar_exception else "rgbd", base)
    if expansion_points.shape[0] >= 24:
        rotation = np.asarray(base["rotation_matrix"], dtype=np.float64)
        local = (
            expansion_points - np.asarray(base["center"], dtype=np.float64)
        ) @ rotation
        base_half = 0.5 * np.asarray(base["dimensions"], dtype=np.float64)
        for quantile in (0.10, 0.075, 0.05, 0.025):
            voxel_lo, voxel_hi = np.quantile(local, [quantile, 1.0 - quantile], axis=0)
            lo = np.minimum(-base_half, voxel_lo)
            hi = np.maximum(base_half, voxel_hi)
            if planar_exception:
                lo[normal_axis] = -base_half[normal_axis]
                hi[normal_axis] = base_half[normal_axis]
            add_candidate(f"voxel_q{quantile:g}", _obb_from_local_extents(base, lo, hi))

    eligible = [
        candidate
        for candidate in candidates
        if (
            float(candidate["projection"]["box_support_rate"])
            >= float(min_support_rate)
            and float(candidate["projection"]["median_box_iou"])
            >= float(min_projected_iou)
        )
    ] or candidates

    def joint_score(candidate: dict) -> float:
        iou = max(float(candidate["projection"]["median_box_iou"]), 0.0)
        view_support = max(float(candidate["projection"]["box_support_rate"]), 0.0)
        voxel_support = max(float(candidate["voxel_retention_gate_rate"]), 0.0)
        expansion = max(float(candidate["volume_expansion"]), 1.0)
        return (
            math.sqrt(iou * voxel_support)
            * (view_support**0.25)
            / (expansion ** float(expansion_penalty))
        )

    best_score = max(joint_score(candidate) for candidate in eligible)
    near_best = [
        candidate
        for candidate in eligible
        if joint_score(candidate)
        >= best_score * (1.0 - max(0.0, float(score_tie_tolerance)))
    ]
    selected = min(
        near_best,
        key=lambda candidate: (
            float(candidate["volume_expansion"]),
            -float(candidate["projection"]["median_box_iou"]),
        ),
    )
    diagnostics = {
        "candidate": str(selected["label"]),
        "voxel_inside_rate": float(selected["voxel_inside_rate"]),
        "base_voxel_inside_rate": float(candidates[0]["voxel_inside_rate"]),
        "voxel_in_plane_inside_rate": float(selected["voxel_in_plane_inside_rate"]),
        "voxel_retention_gate_rate": float(selected["voxel_retention_gate_rate"]),
        "voxel_retention_policy": retention_policy,
        "planar_normal_axis_exception_applied": planar_exception,
        "planar_normal_axis": normal_axis,
        "planar_evidence": planar,
        "volume_expansion": float(selected["volume_expansion"]),
        "candidate_count": len(candidates),
        "joint_score": joint_score(selected),
        "best_joint_score": best_score,
        "score_tie_tolerance": float(score_tie_tolerance),
        "voxel_components": component_diagnostics,
        "candidates": [
            {
                "label": str(candidate["label"]),
                "median_box_iou": float(candidate["projection"]["median_box_iou"]),
                "box_support_rate": float(candidate["projection"]["box_support_rate"]),
                "voxel_inside_rate": float(candidate["voxel_inside_rate"]),
                "voxel_in_plane_inside_rate": float(
                    candidate["voxel_in_plane_inside_rate"]
                ),
                "voxel_retention_gate_rate": float(
                    candidate["voxel_retention_gate_rate"]
                ),
                "volume_expansion": float(candidate["volume_expansion"]),
                "dimensions_m": np.asarray(candidate["obb"]["dimensions"])
                .round(5)
                .tolist(),
                "joint_score": joint_score(candidate),
            }
            for candidate in candidates
        ],
    }
    return selected["obb"], selected["projection"], diagnostics


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x0, y0 = np.maximum(first[:2], second[:2])
    x1, y1 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(area_first + area_second - intersection, 1.0e-12)


def _single_projection_metrics(center: np.ndarray, corners: np.ndarray, observation: dict) -> dict:
    world = np.r_[center.astype(np.float64), 1.0]
    inverse_pose = np.linalg.inv(observation["pose"])
    camera = inverse_pose @ world
    if not np.isfinite(camera).all() or camera[2] <= 1.0e-5:
        return {"inside": 0.0, "normalized_error": float("inf"), "pixel_error": float("inf"), "box_iou": 0.0}
    projected = observation["K"] @ camera[:3]
    uv = projected[:2] / projected[2]
    x0, y0, x1, y1 = observation["raw_bbox"]
    target = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
    error = float(np.linalg.norm(uv - target))
    diagonal = max(float(math.hypot(x1 - x0, y1 - y0)), 1.0)
    corners_camera = (inverse_pose @ np.column_stack((corners, np.ones((8,)))).T).T[:, :3]
    box_iou = 0.0
    if np.all(corners_camera[:, 2] > 1.0e-5):
        projected_corners = (observation["K"] @ corners_camera.T).T
        projected_corners = projected_corners[:, :2] / projected_corners[:, 2:3]
        projected_bbox = np.r_[projected_corners.min(axis=0), projected_corners.max(axis=0)]
        box_iou = _bbox_iou(projected_bbox, observation["raw_bbox"])
    return {
        "inside": float(x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1),
        "normalized_error": error / diagonal,
        "pixel_error": error,
        "box_iou": box_iou,
    }


def _projection_metrics(
    center: np.ndarray,
    corners: np.ndarray,
    observations: list[dict],
    *,
    support_iou: float,
) -> tuple[dict, list[dict]]:
    rows = [_single_projection_metrics(center, corners, observation) for observation in observations]
    finite_rows = [row for row in rows if math.isfinite(float(row["normalized_error"]))]
    if not finite_rows:
        return {
            "center_inside_rate": 0.0,
            "median_normalized_error": float("inf"),
            "median_pixel_error": float("inf"),
            "median_box_iou": 0.0,
            "q25_box_iou": 0.0,
            "box_support_rate": 0.0,
        }, rows
    box_ious = np.asarray([float(row["box_iou"]) for row in rows], dtype=np.float64)
    supported = [
        float(row["inside"] > 0.5 and float(row["box_iou"]) >= float(support_iou))
        for row in rows
    ]
    return {
        "center_inside_rate": float(np.mean([row["inside"] for row in rows])),
        "median_normalized_error": float(np.median([row["normalized_error"] for row in finite_rows])),
        "median_pixel_error": float(np.median([row["pixel_error"] for row in finite_rows])),
        "median_box_iou": float(np.median(box_ious)),
        "q25_box_iou": float(np.percentile(box_ious, 25.0)),
        "box_support_rate": float(np.mean(supported)),
    }, rows


def _independent_timestamp_projection_evidence(
    center: np.ndarray,
    corners: np.ndarray,
    observations: list[dict],
    *,
    support_iou: float,
) -> dict:
    """De-duplicate cameras at a physical moment before release statistics.

    A simultaneous camera family contributes one independent observation. We
    retain its best geometrically supported view, which prevents a partial or
    occluded sibling camera from forcing a larger box while providing no extra
    vote for release.
    """

    _, view_metrics = _projection_metrics(
        center, corners, observations, support_iou=support_iou
    )
    by_timestamp: dict[str, list[dict]] = {}
    missing_or_invalid = 0
    for observation, metric in zip(observations, view_metrics):
        timestamp = str(observation.get("physical_timestamp") or "").strip()
        if not timestamp.isdecimal() or int(timestamp) <= 0:
            missing_or_invalid += 1
            continue
        row = {
            "physical_timestamp": timestamp,
            "image_id": int(observation.get("image_id", -1)),
            "center_inside": bool(float(metric["inside"]) > 0.5),
            "projected_box_iou": float(metric["box_iou"]),
            "normalized_center_error": (
                float(metric["normalized_error"])
                if math.isfinite(float(metric["normalized_error"]))
                else None
            ),
        }
        by_timestamp.setdefault(timestamp, []).append(row)

    selected: list[dict] = []
    for timestamp in sorted(by_timestamp, key=int):
        # This selection is diagnostic only and never authorizes OBB mutation.
        best = max(
            by_timestamp[timestamp],
            key=lambda row: (
                bool(row["center_inside"]),
                float(row["projected_box_iou"]),
                -(
                    float(row["normalized_center_error"])
                    if row["normalized_center_error"] is not None
                    else float("inf")
                ),
            ),
        )
        selected.append(best)
    ious = np.asarray(
        [float(row["projected_box_iou"]) for row in selected], dtype=np.float64
    )
    return {
        "schema": "farm.obb-independent-timestamp-projection.v1",
        "observation_views": len(observations),
        "missing_or_invalid_physical_timestamp_views": missing_or_invalid,
        "independent_physical_timestamps": len(selected),
        "timestamp_median_projected_box_iou": (
            float(np.median(ious)) if ious.size else None
        ),
        "timestamp_q25_projected_box_iou": (
            float(np.quantile(ious, 0.25)) if ious.size else None
        ),
        "supported_timestamp_rate": (
            float(
                np.mean(
                    [
                        bool(row["center_inside"])
                        and float(row["projected_box_iou"]) >= float(support_iou)
                        for row in selected
                    ]
                )
            )
            if selected
            else 0.0
        ),
        "timestamp_rows": selected,
        "contract": {
            "multiple_cameras_at_one_physical_timestamp_count_once": True,
            "best_supported_camera_does_not_expand_obb": True,
        },
    }


def _refine_object(
    object_id: int,
    mask_paths: list[Path],
    *,
    state_images: list,
    voxel_points: np.ndarray,
    voxel_size_m: float,
    frames_root: Path,
    frames_by_stem: dict[str, dict],
    max_points_per_observation: int,
    quantile: float,
    orientation_mode: str,
    up_axis_index: int,
    up_vector: np.ndarray,
    min_refit_box_iou: float,
    max_refit_center_error: float,
    final_support_iou: float,
    min_box_support_rate: float,
    min_median_projected_box_iou: float,
    voxel_expansion_penalty: float,
    score_tie_tolerance: float,
    planar_thresholds: dict,
    seed: int,
) -> dict:
    observations: list[dict] = []
    for path in mask_paths:
        obs = _observation_points(
            path,
            state_images=state_images,
            frames_root=frames_root,
            frames_by_stem=frames_by_stem,
            max_points=max_points_per_observation,
            seed=seed + object_id * 1009,
        )
        if obs is not None:
            observations.append(obs)
    if len(observations) < 2:
        return {"status": "insufficient_depth", "valid_observations": len(observations)}

    centroids = np.stack([obs["centroid"] for obs in observations]).astype(np.float64)
    consensus = np.median(centroids, axis=0)
    distances = np.linalg.norm(centroids - consensus, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = max(0.30, median_distance + 3.0 * 1.4826 * mad)
    consistent = distances <= threshold
    if int(consistent.sum()) < 2:
        consistent[:] = True
    retained_by_centroid = [
        obs for obs, keep in zip(observations, consistent.tolist()) if keep
    ]
    points = np.concatenate([obs["points"] for obs in retained_by_centroid], axis=0)
    initial_obb = _fit_robust_obb(
        points,
        quantile,
        orientation_mode=orientation_mode,
        up_axis_index=up_axis_index,
        up_vector=up_vector,
    )
    _, initial_view_metrics = _projection_metrics(
        initial_obb["center"],
        initial_obb["corners"],
        retained_by_centroid,
        support_iou=final_support_iou,
    )
    retained_by_reprojection = [
        observation
        for observation, metric in zip(retained_by_centroid, initial_view_metrics)
        if (
            float(metric["inside"]) > 0.5
            and float(metric["normalized_error"]) <= float(max_refit_center_error)
            and float(metric["box_iou"]) >= float(min_refit_box_iou)
        )
    ]
    min_refit_count = max(3, int(math.ceil(0.5 * len(retained_by_centroid))))
    refit_applied = len(retained_by_reprojection) >= min_refit_count
    fit_observations = (
        retained_by_reprojection if refit_applied else retained_by_centroid
    )
    points = np.concatenate([obs["points"] for obs in fit_observations], axis=0)
    base_obb = _fit_robust_obb(
        points,
        quantile,
        orientation_mode=orientation_mode,
        up_axis_index=up_axis_index,
        up_vector=up_vector,
    )
    base_obb, planar_evidence = _planar_surface_evidence(
        base_obb,
        fit_observations,
        support_iou=final_support_iou,
        **planar_thresholds,
    )
    obb, projection, voxel_diagnostics = _select_voxel_supported_obb(
        base_obb,
        voxel_points,
        observations,
        voxel_size_m=float(voxel_size_m),
        support_iou=final_support_iou,
        min_support_rate=min_box_support_rate,
        min_projected_iou=min_median_projected_box_iou,
        expansion_penalty=voxel_expansion_penalty,
        score_tie_tolerance=score_tie_tolerance,
        planar_evidence=planar_evidence,
    )
    center = obb["center"]
    dimensions = obb["dimensions"]
    independent_projection = _independent_timestamp_projection_evidence(
        center,
        obb["corners"],
        observations,
        support_iou=final_support_iou,
    )
    return {
        "status": "refined",
        "valid_observations": len(observations),
        "centroid_consistent_observations": len(retained_by_centroid),
        "consistent_observations": len(fit_observations),
        "observation_consistency": float(len(retained_by_centroid) / len(observations)),
        "fit_observation_rate": float(len(fit_observations) / len(observations)),
        "reprojection_refit_applied": refit_applied,
        "centroid_consensus_threshold_m": threshold,
        "center_m": center.astype(np.float32),
        "dimensions_m": dimensions.astype(np.float32),
        "wxyz": obb["wxyz"],
        "orientation_confidence": obb["orientation_confidence"],
        "orientation_mode": obb["orientation_mode"],
        "gravity_alignment": obb["gravity_alignment"],
        "gravity_tilt_degrees": obb["gravity_tilt_degrees"],
        "obb_eigenvalues": obb["eigenvalues"],
        "center_inside_detection_rate": projection["center_inside_rate"],
        "median_reprojection_error_normalized": projection["median_normalized_error"],
        "median_reprojection_error_px": projection["median_pixel_error"],
        "median_projected_box_iou": projection["median_box_iou"],
        "q25_projected_box_iou": projection["q25_box_iou"],
        "box_support_rate": projection["box_support_rate"],
        "independent_timestamp_projection": independent_projection,
        "voxel_inside_obb_rate": voxel_diagnostics["voxel_inside_rate"],
        "base_voxel_inside_obb_rate": voxel_diagnostics["base_voxel_inside_rate"],
        "voxel_in_plane_inside_obb_rate": voxel_diagnostics[
            "voxel_in_plane_inside_rate"
        ],
        "voxel_retention_gate_rate": voxel_diagnostics["voxel_retention_gate_rate"],
        "voxel_retention_policy": voxel_diagnostics["voxel_retention_policy"],
        "planar_normal_axis_exception_applied": voxel_diagnostics[
            "planar_normal_axis_exception_applied"
        ],
        "planar_normal_axis": voxel_diagnostics["planar_normal_axis"],
        "planar_evidence": planar_evidence,
        "voxel_volume_expansion": voxel_diagnostics["volume_expansion"],
        "voxel_extent_candidate": voxel_diagnostics["candidate"],
        "voxel_extent_candidates": voxel_diagnostics["candidates"],
        "voxel_component_count": int(
            voxel_diagnostics["voxel_components"]["component_count"]
        ),
        "voxel_largest_component_fraction": float(
            voxel_diagnostics["voxel_components"]["largest_component_fraction"]
        ),
        "voxel_selected_component_fraction": float(
            voxel_diagnostics["voxel_components"]["selected_component_fraction"]
        ),
        "voxel_selected_component_anchor_points": int(
            voxel_diagnostics["voxel_components"]["selected_component_anchor_points"]
        ),
    }


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--min-depth-observations", type=int, default=3)
    parser.add_argument("--min-consistency", type=float, default=0.60)
    parser.add_argument("--min-center-inside-rate", type=float, default=0.70)
    parser.add_argument("--max-normalized-reprojection-error", type=float, default=0.35)
    parser.add_argument("--min-median-projected-box-iou", type=float, default=0.20)
    parser.add_argument("--min-refit-box-iou", type=float, default=0.08)
    parser.add_argument("--max-refit-center-error", type=float, default=0.45)
    parser.add_argument("--final-support-iou", type=float, default=0.20)
    parser.add_argument("--min-box-support-rate", type=float, default=0.60)
    parser.add_argument("--min-voxel-inside-rate", type=float, default=0.45)
    parser.add_argument(
        "--min-voxel-component-fraction",
        type=float,
        default=0.35,
        help=(
            "Reject spatially fragmented voxel ownership; the selected component "
            "must contain this fraction of all object voxels."
        ),
    )
    parser.add_argument(
        "--max-voxel-volume-expansion",
        type=float,
        default=1.75,
        help="Reject OBB growth beyond the independently fitted RGB-D base volume.",
    )
    parser.add_argument("--voxel-expansion-penalty", type=float, default=0.15)
    parser.add_argument(
        "--candidate-score-tie-tolerance",
        type=float,
        default=0.005,
        help="Prefer the tighter OBB when its joint score is within this fraction of the best candidate.",
    )
    planar = parser.add_argument_group("category-free planar RGB-D gate")
    planar.add_argument("--planar-min-independent-views", type=int, default=4)
    planar.add_argument("--planar-min-view-rate", type=float, default=0.80)
    planar.add_argument("--planar-max-plane-eigenvalue-ratio", type=float, default=0.04)
    planar.add_argument(
        "--planar-min-in-plane-eigenvalue-ratio", type=float, default=0.08
    )
    planar.add_argument(
        "--planar-max-normal-angle-q90-degrees", type=float, default=10.0
    )
    planar.add_argument(
        "--planar-max-median-view-thickness-m", type=float, default=0.12
    )
    planar.add_argument("--planar-max-q90-view-thickness-m", type=float, default=0.18)
    planar.add_argument(
        "--planar-max-median-view-thickness-ratio", type=float, default=0.18
    )
    planar.add_argument("--planar-max-plane-offset-spread-m", type=float, default=0.20)
    planar.add_argument(
        "--planar-max-normal-axis-angle-degrees", type=float, default=15.0
    )
    planar.add_argument("--planar-min-projected-box-iou", type=float, default=0.50)
    planar.add_argument("--planar-min-box-support-rate", type=float, default=0.80)
    planar.add_argument(
        "--planar-max-constrained-thickness-m",
        type=float,
        default=0.18,
        help=(
            "Hard ceiling for only the independently proven plane-normal axis; "
            "the two in-plane axes remain RGB-D/voxel supported."
        ),
    )
    planar.add_argument(
        "--planar-min-alignment-reprojection-retention",
        type=float,
        default=0.85,
        help=(
            "Fail closed when consensus-normal yaw alignment retains less than "
            "this fraction of the unaligned train median reprojection IoU."
        ),
    )
    parser.add_argument(
        "--min-material-orientation-confidence", type=float, default=0.15
    )
    parser.add_argument("--orientation-materiality-threshold", type=float, default=0.25)
    release = parser.add_argument_group("fail-closed OBB publication evidence")
    release.add_argument(
        "--release-min-independent-physical-timestamps", type=int, default=4
    )
    release.add_argument(
        "--release-min-timestamp-median-projected-box-iou",
        type=float,
        default=0.50,
    )
    release.add_argument(
        "--release-min-timestamp-q25-projected-box-iou", type=float, default=0.40
    )
    release.add_argument(
        "--release-min-material-orientation-confidence",
        type=float,
        default=0.35,
    )
    parser.add_argument("--max-points-per-observation", type=int, default=1400)
    parser.add_argument("--quantile", type=float, default=0.05)
    parser.add_argument(
        "--orientation-mode",
        choices=("pca_3d", "gravity_yaw"),
        default="pca_3d",
        help="Use free 3D PCA or constrain OBB roll/pitch to the scene gravity axis.",
    )
    add_up_arguments(parser, default_axis="y")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--assembly-only",
        action="store_true",
        help="Evaluate provisional assembly rows only and preserve every direct object bit-for-bit.",
    )
    parser.add_argument(
        "--object-id-file",
        type=Path,
        help=(
            "Evaluate only the exact newline-delimited object IDs in this file. "
            "This is mutually exclusive with --assembly-only and preserves every "
            "unlisted row bit-for-bit."
        ),
    )
    parser.add_argument(
        "--preferred-observation-source",
        default="",
        help=(
            "Use only canonical mask observations with this exact provenance source "
            "when enough resolve; otherwise retain the complete observation set."
        ),
    )
    parser.add_argument(
        "--minimum-preferred-observations",
        type=int,
        default=3,
        help="Minimum resolved preferred-source views required before source isolation.",
    )
    args = parser.parse_args()
    if args.assembly_only and args.object_id_file is not None:
        parser.error("--assembly-only and --object-id-file are mutually exclusive")
    if not 0.0 <= float(args.min_voxel_component_fraction) <= 1.0:
        parser.error("--min-voxel-component-fraction must be in [0, 1]")
    if (
        not math.isfinite(float(args.max_voxel_volume_expansion))
        or float(args.max_voxel_volume_expansion) < 1.0
    ):
        parser.error("--max-voxel-volume-expansion must be finite and at least one")
    if int(args.release_min_independent_physical_timestamps) < 2:
        parser.error(
            "--release-min-independent-physical-timestamps must be at least two"
        )
    if int(args.planar_min_independent_views) < 4:
        parser.error("--planar-min-independent-views must be at least four")
    unit_interval_args = (
        "planar_min_view_rate",
        "planar_max_plane_eigenvalue_ratio",
        "planar_min_in_plane_eigenvalue_ratio",
        "planar_max_median_view_thickness_ratio",
        "planar_min_projected_box_iou",
        "planar_min_box_support_rate",
        "planar_min_alignment_reprojection_retention",
        "min_material_orientation_confidence",
        "orientation_materiality_threshold",
        "release_min_timestamp_median_projected_box_iou",
        "release_min_timestamp_q25_projected_box_iou",
        "release_min_material_orientation_confidence",
    )
    for name in unit_interval_args:
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    required_release = OBBReleaseEvidencePolicy()
    release_policy_is_weaker = (
        int(args.release_min_independent_physical_timestamps)
        < required_release.minimum_independent_physical_timestamps
        or float(args.release_min_timestamp_median_projected_box_iou)
        < required_release.minimum_timestamp_median_projected_box_iou
        or float(args.release_min_timestamp_q25_projected_box_iou)
        < required_release.minimum_timestamp_q25_projected_box_iou
        or float(args.orientation_materiality_threshold)
        > required_release.orientation_materiality_threshold
        or float(args.release_min_material_orientation_confidence)
        < required_release.minimum_material_orientation_confidence
    )
    if release_policy_is_weaker:
        parser.error(
            "fail-closed OBB release thresholds may be strengthened but not weakened"
        )
    up_policy = policy_from_args(args)
    up_axis_index = up_policy.axis_index
    release_policy = OBBReleaseEvidencePolicy(
        minimum_independent_physical_timestamps=int(
            args.release_min_independent_physical_timestamps
        ),
        minimum_timestamp_median_projected_box_iou=float(
            args.release_min_timestamp_median_projected_box_iou
        ),
        minimum_timestamp_q25_projected_box_iou=float(
            args.release_min_timestamp_q25_projected_box_iou
        ),
        orientation_materiality_threshold=float(args.orientation_materiality_threshold),
        minimum_material_orientation_confidence=float(
            args.release_min_material_orientation_confidence
        ),
    )


    source = args.scene_state.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    active = _numpy(state["active"]).astype(bool, copy=True)
    object_ids = _numpy(state["object_id"]).astype(np.int64, copy=False)
    means = _numpy(state["means"], np.float32)
    cov6 = _numpy(state["cov6"], np.float32)
    count = int(object_ids.shape[0])
    state_images = list(state.get("images") or [])
    voxel_flat = _numpy(state.get("object_voxel_keys_flat", []), np.int64)
    voxel_offsets = _numpy(state.get("object_voxel_keys_offsets", []), np.int64)
    voxel_levels = _numpy(state.get("object_voxel_levels", []), np.int64)
    frames_root, frames_by_stem = _load_frames(args.frames_json.expanduser().resolve())
    def existing_array(key: str, fallback: np.ndarray, dtype: np.dtype) -> np.ndarray:
        value = state.get(key)
        if value is None:
            return fallback.copy()
        array = _numpy(value, dtype)
        return array.copy() if array.shape == fallback.shape else fallback.copy()

    box_centers = existing_array("object_box_centers_m", means, np.float32)
    box_dimensions = existing_array(
        "object_box_dimensions_m", np.full((count, 3), np.nan, dtype=np.float32), np.float32
    )
    identity_wxyz = np.zeros((count, 4), dtype=np.float32)
    identity_wxyz[:, 0] = 1.0
    box_wxyz = existing_array("object_box_wxyz", identity_wxyz, np.float32)
    prior_statuses = state.get("object_geometry_status")
    statuses = list(prior_statuses) if isinstance(prior_statuses, (list, tuple)) else ["not_evaluated"] * count
    if len(statuses) < count:
        statuses.extend(["not_evaluated"] * (count - len(statuses)))
    inside_rates = existing_array("object_geometry_inside_rate", np.zeros((count,), dtype=np.float32), np.float32)
    normalized_errors = existing_array(
        "object_geometry_reprojection_error", np.full((count,), np.inf, dtype=np.float32), np.float32
    )
    valid_observations = existing_array(
        "object_geometry_valid_observations", np.zeros((count,), dtype=np.int32), np.int32
    )
    consistent_observations = existing_array(
        "object_geometry_consistent_observations", np.zeros((count,), dtype=np.int32), np.int32
    )
    projected_box_ious = existing_array(
        "object_geometry_projected_box_iou", np.zeros((count,), dtype=np.float32), np.float32
    )
    box_support_rates = existing_array(
        "object_geometry_box_support_rate", np.zeros((count,), dtype=np.float32), np.float32
    )
    voxel_inside_rates = existing_array(
        "object_geometry_voxel_inside_rate", np.zeros((count,), dtype=np.float32), np.float32
    )
    voxel_volume_expansions = existing_array(
        "object_geometry_voxel_volume_expansion", np.ones((count,), dtype=np.float32), np.float32
    )
    voxel_component_fractions = existing_array(
        "object_geometry_voxel_component_fraction",
        np.zeros((count,), dtype=np.float32),
        np.float32,
    )
    voxel_retention_gate_rates = existing_array(
        "object_geometry_voxel_retention_gate_rate",
        np.zeros((count,), dtype=np.float32),
        np.float32,
    )
    planar_exception_applied = existing_array(
        "object_geometry_planar_exception_applied",
        np.zeros((count,), dtype=bool),
        bool,
    )
    planar_normal_axes = existing_array(
        "object_geometry_planar_normal_axis",
        np.full((count,), -1, dtype=np.int8),
        np.int8,
    )
    planar_independent_views = existing_array(
        "object_geometry_planar_independent_views",
        np.zeros((count,), dtype=np.int16),
        np.int16,
    )
    planar_normal_q90_degrees = existing_array(
        "object_geometry_planar_normal_q90_degrees",
        np.full((count,), np.inf, dtype=np.float32),
        np.float32,
    )
    planar_median_view_thickness_m = existing_array(
        "object_geometry_planar_median_view_thickness_m",
        np.full((count,), np.inf, dtype=np.float32),
        np.float32,
    )
    prior_retention_policies = state.get("object_geometry_voxel_retention_policy")
    voxel_retention_policies = (
        list(prior_retention_policies)
        if isinstance(prior_retention_policies, (list, tuple))
        else ["full_3d_voxel_retention"] * count
    )
    prior_planar_reasons = state.get("object_geometry_planar_evidence_reasons")
    planar_evidence_reasons = (
        [list(value) for value in prior_planar_reasons]
        if isinstance(prior_planar_reasons, (list, tuple))
        else [[] for _ in range(count)]
    )
    if len(voxel_retention_policies) < count:
        voxel_retention_policies.extend(
            ["full_3d_voxel_retention"] * (count - len(voxel_retention_policies))
        )
    if len(planar_evidence_reasons) < count:
        planar_evidence_reasons.extend(
            [[] for _ in range(count - len(planar_evidence_reasons))]
        )
    prior_candidates = state.get("object_geometry_voxel_extent_candidate")
    voxel_extent_candidates = list(prior_candidates) if isinstance(prior_candidates, (list, tuple)) else ["not_evaluated"] * count
    if len(voxel_extent_candidates) < count:
        voxel_extent_candidates.extend(["not_evaluated"] * (count - len(voxel_extent_candidates)))
    orientation_confidences = existing_array(
        "object_geometry_orientation_confidence", np.zeros((count,), dtype=np.float32), np.float32
    )
    gravity_tilt_degrees = existing_array(
        "object_geometry_gravity_tilt_degrees", np.full((count,), np.nan, dtype=np.float32), np.float32
    )
    geometry_fit_passes = existing_array(
        "object_geometry_fit_pass", np.zeros((count,), dtype=bool), bool
    )
    geometry_release_ready = existing_array(
        "object_geometry_release_ready", np.zeros((count,), dtype=bool), bool
    )
    independent_physical_timestamps = existing_array(
        "object_geometry_independent_physical_timestamps",
        np.zeros((count,), dtype=np.int16),
        np.int16,
    )
    timestamp_median_projected_box_ious = existing_array(
        "object_geometry_timestamp_median_projected_box_iou",
        np.zeros((count,), dtype=np.float32),
        np.float32,
    )
    timestamp_q25_projected_box_ious = existing_array(
        "object_geometry_timestamp_q25_projected_box_iou",
        np.zeros((count,), dtype=np.float32),
        np.float32,
    )
    prior_release_routes = state.get("object_geometry_release_route")
    geometry_release_routes = (
        list(prior_release_routes)
        if isinstance(prior_release_routes, (list, tuple))
        else ["not_evaluated"] * count
    )
    prior_release_reasons = state.get("object_geometry_release_reasons")
    geometry_release_reasons = (
        [list(value) for value in prior_release_reasons]
        if isinstance(prior_release_reasons, (list, tuple))
        else [[] for _ in range(count)]
    )
    if len(geometry_release_routes) < count:
        geometry_release_routes.extend(
            ["not_evaluated"] * (count - len(geometry_release_routes))
        )
    if len(geometry_release_reasons) < count:
        geometry_release_reasons.extend(
            [[] for _ in range(count - len(geometry_release_reasons))]
        )

    rows: list[dict] = []

    assembly_members = state.get("object_assembly_member_ids") or []
    assembly_indices = {
        index
        for index in range(min(count, len(assembly_members)))
        if isinstance(assembly_members[index], (list, tuple)) and len(assembly_members[index]) > 0
    }
    if args.object_id_file is not None:
        requested_path = args.object_id_file.expanduser().resolve(strict=True)
        requested_ids: list[int] = []
        for line_number, raw in enumerate(
            requested_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            try:
                object_id = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid object ID on line {line_number} of {requested_path}: {value!r}"
                ) from exc
            if object_id < 0:
                raise ValueError(f"object IDs must be non-negative: {object_id}")
            requested_ids.append(object_id)
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("--object-id-file contains duplicate object IDs")
        index_by_id = {int(value): index for index, value in enumerate(object_ids.tolist())}
        missing = sorted(set(requested_ids) - set(index_by_id))
        if missing:
            raise ValueError(f"--object-id-file contains unknown object IDs: {missing}")
        evaluation_indices = [index_by_id[object_id] for object_id in requested_ids]
    elif args.assembly_only:
        evaluation_indices = sorted(assembly_indices)
    else:
        evaluation_indices = np.flatnonzero(active).tolist()
    mask_index = resolve_object_mask_observations(
        state,
        args.mask_dir.expanduser().resolve(),
        include_object_ids=[int(object_ids[index]) for index in evaluation_indices],
    )

    for index in evaluation_indices:
        object_id = int(object_ids[index])
        voxel_points = np.zeros((0, 3), dtype=np.float32)
        voxel_level = int(voxel_levels[index]) if index < voxel_levels.size else 0
        voxel_size_m = float(VOXEL_BASE_V) * float(1 << voxel_level)
        if index + 1 < voxel_offsets.size and index < voxel_levels.size:
            start, end = int(voxel_offsets[index]), int(voxel_offsets[index + 1])
            if 0 <= start < end <= voxel_flat.size:
                voxel_points = np.asarray(
                    decode_voxel_keys_numpy(voxel_flat[start:end], voxel_level),
                    dtype=np.float32,
                ).reshape(-1, 3)
        observation_paths, observation_selection = _select_observation_paths(
            state,
            index,
            mask_index.for_index(index),
            preferred_source=str(args.preferred_observation_source),
            minimum_preferred=int(args.minimum_preferred_observations),
        )
        result = _refine_object(
            object_id,
            observation_paths,
            state_images=state_images,
            voxel_points=voxel_points,
            voxel_size_m=voxel_size_m,
            frames_root=frames_root,
            frames_by_stem=frames_by_stem,
            max_points_per_observation=max(0, int(args.max_points_per_observation)),
            quantile=float(args.quantile),
            orientation_mode=str(args.orientation_mode),
            up_axis_index=up_axis_index,
            up_vector=up_policy.vector,
            min_refit_box_iou=float(args.min_refit_box_iou),
            max_refit_center_error=float(args.max_refit_center_error),
            final_support_iou=float(args.final_support_iou),
            min_box_support_rate=float(args.min_box_support_rate),
            min_median_projected_box_iou=float(args.min_median_projected_box_iou),
            voxel_expansion_penalty=float(args.voxel_expansion_penalty),
            score_tie_tolerance=float(args.candidate_score_tie_tolerance),
            planar_thresholds={
                "min_independent_views": int(args.planar_min_independent_views),
                "min_planar_view_rate": float(args.planar_min_view_rate),
                "max_plane_eigenvalue_ratio": float(
                    args.planar_max_plane_eigenvalue_ratio
                ),
                "min_in_plane_eigenvalue_ratio": float(
                    args.planar_min_in_plane_eigenvalue_ratio
                ),
                "max_normal_angle_q90_degrees": float(
                    args.planar_max_normal_angle_q90_degrees
                ),
                "max_median_view_thickness_m": float(
                    args.planar_max_median_view_thickness_m
                ),
                "max_q90_view_thickness_m": float(args.planar_max_q90_view_thickness_m),
                "max_median_view_thickness_ratio": float(
                    args.planar_max_median_view_thickness_ratio
                ),
                "max_plane_offset_spread_m": float(
                    args.planar_max_plane_offset_spread_m
                ),
                "max_normal_axis_angle_degrees": float(
                    args.planar_max_normal_axis_angle_degrees
                ),
                "min_projected_box_iou": float(args.planar_min_projected_box_iou),
                "min_box_support_rate": float(args.planar_min_box_support_rate),
                "max_constrained_thickness_m": float(
                    args.planar_max_constrained_thickness_m
                ),
                "min_alignment_reprojection_retention": float(
                    args.planar_min_alignment_reprojection_retention
                ),
            },
            seed=int(args.seed),
        )
        valid = int(result.get("valid_observations", 0))
        consistent = int(result.get("consistent_observations", 0))
        consistency = float(result.get("observation_consistency", 0.0))
        inside = float(result.get("center_inside_detection_rate", 0.0))
        norm_error = float(
            result.get("median_reprojection_error_normalized", float("inf"))
        )
        projected_box_iou = float(result.get("median_projected_box_iou", 0.0))
        box_support_rate = float(result.get("box_support_rate", 0.0))
        voxel_inside_rate = float(result.get("voxel_inside_obb_rate", 0.0))
        voxel_volume_expansion = float(result.get("voxel_volume_expansion", 1.0))
        voxel_component_fraction = float(
            result.get("voxel_selected_component_fraction", 0.0)
        )
        voxel_extent_candidate = str(result.get("voxel_extent_candidate", "rgbd"))
        voxel_retention_gate_rate = float(
            result.get("voxel_retention_gate_rate", voxel_inside_rate)
        )
        voxel_retention_policy = str(
            result.get("voxel_retention_policy", "full_3d_voxel_retention")
        )
        planar_evidence = (
            result.get("planar_evidence")
            if isinstance(result.get("planar_evidence"), dict)
            else {}
        )
        planar_exception = bool(
            result.get("planar_normal_axis_exception_applied", False)
        )
        orientation_confidence = float(result.get("orientation_confidence", 0.0))
        candidate_dimensions = _numpy(
            result.get("dimensions_m", [1.0, 1.0, 1.0]), np.float32
        )
        material_dimensions = (
            candidate_dimensions[:2]
            if str(args.orientation_mode) == "gravity_yaw"
            else candidate_dimensions
        )
        orientation_materiality = float(
            1.0
            - np.min(material_dimensions)
            / max(float(np.max(material_dimensions)), 1.0e-6)
        )
        orientation_required = orientation_materiality >= float(
            args.orientation_materiality_threshold
        )
        orientation_reliable = (
            not orientation_required
            or orientation_confidence >= float(args.min_material_orientation_confidence)
        )
        fit_checks = (
            ("fit_status_not_refined", result.get("status") == "refined"),
            ("insufficient_depth_observations", valid >= int(args.min_depth_observations)),
            ("observation_consistency_below_limit", consistency >= float(args.min_consistency)),
            ("center_inside_rate_below_limit", inside >= float(args.min_center_inside_rate)),
            (
                "normalized_reprojection_error_above_limit",
                norm_error <= float(args.max_normalized_reprojection_error),
            ),
            (
                "median_projected_box_iou_below_fit_limit",
                projected_box_iou >= float(args.min_median_projected_box_iou),
            ),
            ("box_support_rate_below_limit", box_support_rate >= float(args.min_box_support_rate)),
            (
                "voxel_retention_rate_below_limit",
                voxel_retention_gate_rate >= float(args.min_voxel_inside_rate),
            ),
            (
                "voxel_component_fraction_below_limit",
                voxel_component_fraction >= float(args.min_voxel_component_fraction),
            ),
            (
                "voxel_volume_expansion_above_limit",
                voxel_volume_expansion <= float(args.max_voxel_volume_expansion),
            ),
            ("fit_orientation_confidence_below_limit", orientation_reliable),
        )
        fit_reasons = [reason for reason, accepted in fit_checks if not accepted]
        geometry_fit_passed = not fit_reasons
        independent_projection = (
            result.get("independent_timestamp_projection")
            if isinstance(result.get("independent_timestamp_projection"), dict)
            else {}
        )
        release_evidence = evaluate_obb_release_evidence(
            geometry_fit_passed=geometry_fit_passed,
            independent_physical_timestamps=independent_projection.get(
                "independent_physical_timestamps", 0
            ),
            timestamp_median_projected_box_iou=independent_projection.get(
                "timestamp_median_projected_box_iou"
            ),
            timestamp_q25_projected_box_iou=independent_projection.get(
                "timestamp_q25_projected_box_iou"
            ),
            orientation_materiality=orientation_materiality,
            orientation_confidence=orientation_confidence,
            policy=release_policy,
        )
        passed = bool(release_evidence["passed"])
        if result.get("status") == "refined":
            box_centers[index] = result["center_m"]
            box_dimensions[index] = result["dimensions_m"]
            box_wxyz[index] = result["wxyz"]
        is_assembly = index in assembly_indices
        status = (
            ("assembly_geometry_pass" if passed else "assembly_geometry_rejected")
            if is_assembly
            else ("geometry_pass" if passed else "geometry_rejected")
        )
        statuses[index] = status
        active[index] = bool(passed)
        geometry_fit_passes[index] = bool(geometry_fit_passed)
        geometry_release_ready[index] = bool(passed)
        independent_physical_timestamps[index] = int(
            independent_projection.get("independent_physical_timestamps", 0)
        )
        timestamp_median_projected_box_ious[index] = float(
            independent_projection.get("timestamp_median_projected_box_iou") or 0.0
        )
        timestamp_q25_projected_box_ious[index] = float(
            independent_projection.get("timestamp_q25_projected_box_iou") or 0.0
        )
        geometry_release_routes[index] = str(release_evidence["route"])
        geometry_release_reasons[index] = list(release_evidence["reasons"])

        inside_rates[index] = inside
        normalized_errors[index] = norm_error
        valid_observations[index] = valid
        consistent_observations[index] = consistent
        projected_box_ious[index] = projected_box_iou
        box_support_rates[index] = box_support_rate
        voxel_inside_rates[index] = voxel_inside_rate
        voxel_volume_expansions[index] = voxel_volume_expansion
        voxel_component_fractions[index] = voxel_component_fraction
        voxel_extent_candidates[index] = voxel_extent_candidate
        voxel_retention_gate_rates[index] = voxel_retention_gate_rate
        voxel_retention_policies[index] = voxel_retention_policy
        planar_exception_applied[index] = planar_exception
        planar_normal_axes[index] = int(result.get("planar_normal_axis", -1))
        planar_independent_views[index] = int(
            planar_evidence.get("independent_physical_timestamps", 0)
        )
        planar_normal_q90_degrees[index] = float(
            planar_evidence.get("normal_angle_q90_degrees", float("inf"))
        )
        planar_median_view_thickness_m[index] = float(
            planar_evidence.get("median_per_view_depth_thickness_m", float("inf"))
        )
        planar_evidence_reasons[index] = list(planar_evidence.get("reasons") or [])
        orientation_confidences[index] = orientation_confidence
        gravity_tilt_degrees[index] = float(
            result.get("gravity_tilt_degrees", float("nan"))
        )
        row = {
            "object_id": object_id,
            "category": str(
                (state.get("object_category") or [""] * count)[index] or ""
            ),
            "status": status,
            "geometry_fit_passed": bool(geometry_fit_passed),
            "geometry_fit_reasons": fit_reasons,
            "geometry_release_ready": bool(passed),
            "geometry_release_route": str(release_evidence["route"]),
            "geometry_release_reasons": list(release_evidence["reasons"]),
            "train_geometry_evidence": release_evidence,
            "independent_timestamp_projection": independent_projection,
            "is_assembly": is_assembly,
            "assembly_member_ids": list(assembly_members[index]) if is_assembly else [],
            "observation_selection": observation_selection,
            "valid_observations": valid,
            "consistent_observations": consistent,
            "centroid_consistent_observations": int(
                result.get("centroid_consistent_observations", consistent)
            ),
            "reprojection_refit_applied": bool(
                result.get("reprojection_refit_applied", False)
            ),
            "observation_consistency": consistency,
            "fit_observation_rate": float(
                result.get("fit_observation_rate", consistency)
            ),
            "center_inside_detection_rate": inside,
            "median_reprojection_error_normalized": (
                norm_error if math.isfinite(norm_error) else None
            ),
            "median_reprojection_error_px": result.get("median_reprojection_error_px"),
            "center_shift_m": float(np.linalg.norm(box_centers[index] - means[index])),
            "box_center_m": box_centers[index].round(5).tolist(),
            "box_dimensions_m": box_dimensions[index].round(5).tolist(),
            "box_wxyz": box_wxyz[index].round(7).tolist(),
            "orientation_confidence": float(orientation_confidences[index]),
            "orientation_mode": str(
                result.get("orientation_mode", args.orientation_mode)
            ),
            "up_axis": up_policy.legacy_state_axis,
            "up_direction": up_policy.direction,
            "up_vector": up_policy.vector.astype(float).tolist(),
            "gravity_alignment": float(result.get("gravity_alignment", 0.0)),
            "gravity_tilt_degrees": float(gravity_tilt_degrees[index]),
            "orientation_materiality": orientation_materiality,
            "orientation_required": orientation_required,
            "orientation_reliable": orientation_reliable,
            "obb_eigenvalues": _numpy(result.get("obb_eigenvalues", [0.0, 0.0, 0.0]))
            .round(7)
            .tolist(),
            "median_projected_box_iou": projected_box_iou,
            "q25_projected_box_iou": float(result.get("q25_projected_box_iou", 0.0)),
            "box_support_rate": box_support_rate,
            "voxel_inside_obb_rate": voxel_inside_rate,
            "base_voxel_inside_obb_rate": float(
                result.get("base_voxel_inside_obb_rate", 0.0)
            ),
            "voxel_in_plane_inside_obb_rate": float(
                result.get("voxel_in_plane_inside_obb_rate", 0.0)
            ),
            "voxel_retention_gate_rate": voxel_retention_gate_rate,
            "voxel_retention_gate": (
                voxel_retention_gate_rate >= float(args.min_voxel_inside_rate)
            ),
            "voxel_retention_policy": voxel_retention_policy,
            "planar_normal_axis_exception_applied": planar_exception,
            "planar_normal_axis": int(result.get("planar_normal_axis", -1)),
            "planar_evidence": planar_evidence,
            "voxel_volume_expansion": voxel_volume_expansion,
            "voxel_component_count": int(result.get("voxel_component_count", 0)),
            "voxel_largest_component_fraction": float(
                result.get("voxel_largest_component_fraction", 0.0)
            ),
            "voxel_selected_component_fraction": voxel_component_fraction,
            "voxel_selected_component_anchor_points": int(
                result.get("voxel_selected_component_anchor_points", 0)
            ),
            "voxel_component_gate": (
                voxel_component_fraction >= float(args.min_voxel_component_fraction)
            ),
            "voxel_volume_expansion_gate": (
                voxel_volume_expansion <= float(args.max_voxel_volume_expansion)
            ),
            "voxel_extent_candidate": voxel_extent_candidate,
            "voxel_extent_candidates": result.get("voxel_extent_candidates", []),
        }
        original_dimensions = 5.0 * np.sqrt(
            np.clip(cov6[index, [0, 3, 5]], 1.0e-4, None)
        )
        row["original_box_dimensions_5sigma_m"] = original_dimensions.round(5).tolist()
        row["original_box_volume_m3"] = float(np.prod(original_dimensions))
        row["refined_box_volume_m3"] = (
            float(np.prod(box_dimensions[index]))
            if np.isfinite(box_dimensions[index]).all()
            else None
        )
        rows.append(row)

    state["active"] = torch.as_tensor(active, dtype=torch.bool)
    state["object_box_centers_m"] = torch.as_tensor(box_centers, dtype=torch.float32)
    state["object_box_dimensions_m"] = torch.as_tensor(box_dimensions, dtype=torch.float32)
    state["object_box_wxyz"] = torch.as_tensor(box_wxyz, dtype=torch.float32)
    state["object_geometry_status"] = statuses
    state["object_geometry_fit_pass"] = torch.as_tensor(
        geometry_fit_passes, dtype=torch.bool
    )
    state["object_geometry_release_ready"] = torch.as_tensor(
        geometry_release_ready, dtype=torch.bool
    )
    state["object_geometry_independent_physical_timestamps"] = torch.as_tensor(
        independent_physical_timestamps, dtype=torch.int16
    )
    state["object_geometry_timestamp_median_projected_box_iou"] = torch.as_tensor(
        timestamp_median_projected_box_ious, dtype=torch.float32
    )
    state["object_geometry_timestamp_q25_projected_box_iou"] = torch.as_tensor(
        timestamp_q25_projected_box_ious, dtype=torch.float32
    )
    state["object_geometry_release_route"] = geometry_release_routes
    state["object_geometry_release_reasons"] = geometry_release_reasons

    state["object_geometry_inside_rate"] = torch.as_tensor(inside_rates, dtype=torch.float32)
    state["object_geometry_reprojection_error"] = torch.as_tensor(normalized_errors, dtype=torch.float32)
    state["object_geometry_valid_observations"] = torch.as_tensor(valid_observations, dtype=torch.int32)
    state["object_geometry_consistent_observations"] = torch.as_tensor(consistent_observations, dtype=torch.int32)
    state["object_geometry_projected_box_iou"] = torch.as_tensor(projected_box_ious, dtype=torch.float32)
    state["object_geometry_box_support_rate"] = torch.as_tensor(box_support_rates, dtype=torch.float32)
    state["object_geometry_voxel_inside_rate"] = torch.as_tensor(voxel_inside_rates, dtype=torch.float32)
    state["object_geometry_voxel_volume_expansion"] = torch.as_tensor(voxel_volume_expansions, dtype=torch.float32)
    state["object_geometry_voxel_component_fraction"] = torch.as_tensor(
        voxel_component_fractions, dtype=torch.float32
    )
    state["object_geometry_voxel_extent_candidate"] = voxel_extent_candidates
    state["object_geometry_voxel_retention_gate_rate"] = torch.as_tensor(
        voxel_retention_gate_rates, dtype=torch.float32
    )
    state["object_geometry_voxel_retention_policy"] = voxel_retention_policies
    state["object_geometry_planar_exception_applied"] = torch.as_tensor(
        planar_exception_applied, dtype=torch.bool
    )
    state["object_geometry_planar_normal_axis"] = torch.as_tensor(
        planar_normal_axes, dtype=torch.int8
    )
    state["object_geometry_planar_independent_views"] = torch.as_tensor(
        planar_independent_views, dtype=torch.int16
    )
    state["object_geometry_planar_normal_q90_degrees"] = torch.as_tensor(
        planar_normal_q90_degrees, dtype=torch.float32
    )
    state["object_geometry_planar_median_view_thickness_m"] = torch.as_tensor(
        planar_median_view_thickness_m, dtype=torch.float32
    )
    state["object_geometry_planar_evidence_reasons"] = planar_evidence_reasons
    state["object_geometry_orientation_confidence"] = torch.as_tensor(
        orientation_confidences, dtype=torch.float32
    )
    state["object_geometry_gravity_tilt_degrees"] = torch.as_tensor(
        gravity_tilt_degrees, dtype=torch.float32
    )
    prior_modes = state.get("object_geometry_orientation_mode")
    orientation_modes = (
        list(prior_modes)
        if isinstance(prior_modes, (list, tuple))
        else [str(args.orientation_mode)] * count
    )
    if len(orientation_modes) < count:
        orientation_modes.extend([str(args.orientation_mode)] * (count - len(orientation_modes)))
    for index in evaluation_indices:
        orientation_modes[index] = str(args.orientation_mode)
    state["object_geometry_orientation_mode"] = orientation_modes
    write_state_up_policy(state, up_policy)
    passed_evaluated = sum(str(row.get("status", "")).endswith("_pass") for row in rows)
    fit_passed_evaluated = sum(
        bool(row.get("geometry_fit_passed")) for row in rows
    )
    view_rescue_evaluated = sum(
        row.get("geometry_release_route") == "view_rescue" for row in rows
    )
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["geometry_refinement"] = {
            "source": str(source),
            "automatic": True,
            "orientation_mode": str(args.orientation_mode),
            "up": up_policy.to_dict(),
            "fit_passed_evaluated_objects": int(fit_passed_evaluated),
            "passed_evaluated_objects": int(passed_evaluated),
            "view_rescue_evaluated_objects": int(view_rescue_evaluated),
            "obb_release_policy": release_policy.to_dict(),
            "active_objects": int(active.sum()),
            "evaluated_objects": len(rows),
            "assembly_only": bool(args.assembly_only),
            "object_id_filter": (
                str(args.object_id_file.expanduser().resolve())
                if args.object_id_file is not None else None
            ),
        }
    else:
        payload = state

    duration_seconds = time.perf_counter() - started
    report = {
        "schema": "farm.object-geometry-audit.v3",
        "created_unix_s": time.time(),
        "source_scene_state": str(source),
        "evaluated_objects": len(rows),
        "fit_passed_objects": int(fit_passed_evaluated),
        "passed_objects": int(passed_evaluated),
        "rejected_objects": int(len(rows) - passed_evaluated),
        "view_rescue_objects": int(view_rescue_evaluated),
        "active_objects": int(active.sum()),
        "obb_release_evidence_contract": {
            "schema": "farm.obb-train-release-evidence.v1",
            "policy": release_policy.to_dict(),
            "geometry_pass_requires_release_evidence": True,
            "provisional_fit_is_not_release_ready": True,
            "multiple_cameras_at_one_physical_timestamp_count_once": True,
            "robust_statistics_use_one_supported_view_per_timestamp": True,
            "partial_or_occluded_view_does_not_authorize_obb_expansion": True,
            "failed_release_objects_route_to_more_evidence": True,
        },
        "assembly_only": bool(args.assembly_only),
        "object_id_filter": (
            str(args.object_id_file.expanduser().resolve())
            if args.object_id_file is not None
            else None
        ),
        "up": up_policy.to_dict(),
        "mask_observation_contract": mask_index.diagnostics,
        "timing": {
            "duration_seconds": duration_seconds,
            "seconds_per_evaluated_object": duration_seconds / max(len(rows), 1),
            "stage": "multi_view_rgbd_obb_refinement",
        },
        "thresholds": {
            "preferred_observation_source": str(args.preferred_observation_source or "")
            or None,
            "minimum_preferred_observations": int(args.minimum_preferred_observations),
            "min_depth_observations": int(args.min_depth_observations),
            "min_consistency": float(args.min_consistency),
            "min_center_inside_rate": float(args.min_center_inside_rate),
            "max_normalized_reprojection_error": float(
                args.max_normalized_reprojection_error
            ),
            "min_median_projected_box_iou": float(args.min_median_projected_box_iou),
            "min_refit_box_iou": float(args.min_refit_box_iou),
            "max_refit_center_error": float(args.max_refit_center_error),
            "final_support_iou": float(args.final_support_iou),
            "min_box_support_rate": float(args.min_box_support_rate),
            "min_voxel_inside_rate": float(args.min_voxel_inside_rate),
            "min_voxel_component_fraction": float(args.min_voxel_component_fraction),
            "max_voxel_volume_expansion": float(args.max_voxel_volume_expansion),
            "voxel_expansion_penalty": float(args.voxel_expansion_penalty),
            "candidate_score_tie_tolerance": float(args.candidate_score_tie_tolerance),
            "planar_min_independent_views": int(args.planar_min_independent_views),
            "planar_min_view_rate": float(args.planar_min_view_rate),
            "planar_max_plane_eigenvalue_ratio": float(
                args.planar_max_plane_eigenvalue_ratio
            ),
            "planar_min_in_plane_eigenvalue_ratio": float(
                args.planar_min_in_plane_eigenvalue_ratio
            ),
            "planar_max_normal_angle_q90_degrees": float(
                args.planar_max_normal_angle_q90_degrees
            ),
            "planar_max_median_view_thickness_m": float(
                args.planar_max_median_view_thickness_m
            ),
            "planar_max_q90_view_thickness_m": float(
                args.planar_max_q90_view_thickness_m
            ),
            "planar_max_median_view_thickness_ratio": float(
                args.planar_max_median_view_thickness_ratio
            ),
            "planar_max_plane_offset_spread_m": float(
                args.planar_max_plane_offset_spread_m
            ),
            "planar_max_normal_axis_angle_degrees": float(
                args.planar_max_normal_axis_angle_degrees
            ),
            "planar_min_projected_box_iou": float(args.planar_min_projected_box_iou),
            "planar_min_box_support_rate": float(args.planar_min_box_support_rate),
            "planar_max_constrained_thickness_m": float(
                args.planar_max_constrained_thickness_m
            ),
            "planar_min_alignment_reprojection_retention": float(
                args.planar_min_alignment_reprojection_retention
            ),
            "min_material_orientation_confidence": float(
                args.min_material_orientation_confidence
            ),
            "release_min_independent_physical_timestamps": int(
                args.release_min_independent_physical_timestamps
            ),
            "release_min_timestamp_median_projected_box_iou": float(
                args.release_min_timestamp_median_projected_box_iou
            ),
            "release_min_timestamp_q25_projected_box_iou": float(
                args.release_min_timestamp_q25_projected_box_iou
            ),
            "release_min_material_orientation_confidence": float(
                args.release_min_material_orientation_confidence
            ),
            "orientation_materiality_threshold": float(
                args.orientation_materiality_threshold
            ),
            "orientation_mode": str(args.orientation_mode),
            "up_axis": up_policy.axis,
            "up_direction": up_policy.direction,
            "up_vector": up_policy.vector.astype(float).tolist(),
            "up_source": up_policy.source,
        },
        "objects": rows,
    }
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_state)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("evaluated_objects", "passed_objects", "rejected_objects")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
