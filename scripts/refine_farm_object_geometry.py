#!/usr/bin/env python3
"""Refine FARM presentation boxes from their saved multi-view RGB-D evidence.

This is an automatic post-processing stage.  It reconstructs each object's
inlier-mask pixels with the same metric depth and camera poses used by FARM,
rejects inconsistent observations, fits robust world-space boxes, and audits
the result by projecting every refined centre back into its source detections.
The mapping state is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

try:
    from farm_geometry_axes import (
        add_up_arguments,
        horizontal_plane_basis,
        normalize_up_vector,
        policy_from_args,
        write_state_up_policy,
    )
except ModuleNotFoundError:  # package import: scripts.refine_farm_object_geometry
    from scripts.farm_geometry_axes import (
        add_up_arguments,
        horizontal_plane_basis,
        normalize_up_vector,
        policy_from_args,
        write_state_up_policy,
    )
from scene_graph.utils.geometry import decode_voxel_keys_numpy
from scene_graph.map_update.mask_observations import resolve_object_mask_observations


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


def _voxel_inside_rate(obb: dict, voxel_points: np.ndarray) -> float:
    if voxel_points.shape[0] == 0:
        return 0.0
    rotation = np.asarray(obb["rotation_matrix"], dtype=np.float64)
    local = (voxel_points.astype(np.float64) - np.asarray(obb["center"], dtype=np.float64)) @ rotation
    half = 0.5 * np.asarray(obb["dimensions"], dtype=np.float64)
    return float(np.mean(np.all(np.abs(local) <= half + 1.0e-6, axis=1)))


def _select_voxel_supported_obb(
    base: dict,
    voxel_points: np.ndarray,
    observations: list[dict],
    *,
    support_iou: float,
    min_support_rate: float,
    min_projected_iou: float,
    expansion_penalty: float,
    score_tie_tolerance: float,
) -> tuple[dict, dict, dict]:
    """Choose the most reprojection-accurate RGB-D/voxel extent candidate.

    Object names never enter this decision.  RGB-D fixes orientation and a
    minimum extent; FARM voxel-memory proposes progressively wider robust
    extents.  Full multi-view reprojection remains the primary selector.
    """
    points = np.asarray(voxel_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    base_volume = max(float(np.prod(base["dimensions"])), 1.0e-9)
    candidates: list[dict] = []

    def add_candidate(label: str, obb: dict) -> None:
        projection, _ = _projection_metrics(
            obb["center"], obb["corners"], observations, support_iou=support_iou
        )
        candidates.append({
            "label": label,
            "obb": obb,
            "projection": projection,
            "voxel_inside_rate": _voxel_inside_rate(obb, points),
            "volume_expansion": float(np.prod(obb["dimensions"]) / base_volume),
        })

    add_candidate("rgbd", base)
    if points.shape[0] >= 24:
        rotation = np.asarray(base["rotation_matrix"], dtype=np.float64)
        local = (points - np.asarray(base["center"], dtype=np.float64)) @ rotation
        base_half = 0.5 * np.asarray(base["dimensions"], dtype=np.float64)
        for quantile in (0.10, 0.075, 0.05, 0.025):
            voxel_lo, voxel_hi = np.quantile(local, [quantile, 1.0 - quantile], axis=0)
            lo = np.minimum(-base_half, voxel_lo)
            hi = np.maximum(base_half, voxel_hi)
            add_candidate(f"voxel_q{quantile:g}", _obb_from_local_extents(base, lo, hi))

    eligible = [
        candidate
        for candidate in candidates
        if (
            float(candidate["projection"]["box_support_rate"]) >= float(min_support_rate)
            and float(candidate["projection"]["median_box_iou"]) >= float(min_projected_iou)
        )
    ] or candidates
    def joint_score(candidate: dict) -> float:
        iou = max(float(candidate["projection"]["median_box_iou"]), 0.0)
        view_support = max(float(candidate["projection"]["box_support_rate"]), 0.0)
        voxel_support = max(float(candidate["voxel_inside_rate"]), 0.0)
        expansion = max(float(candidate["volume_expansion"]), 1.0)
        return math.sqrt(iou * voxel_support) * (view_support ** 0.25) / (expansion ** float(expansion_penalty))

    best_score = max(joint_score(candidate) for candidate in eligible)
    near_best = [
        candidate
        for candidate in eligible
        if joint_score(candidate) >= best_score * (1.0 - max(0.0, float(score_tie_tolerance)))
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
        "volume_expansion": float(selected["volume_expansion"]),
        "candidate_count": len(candidates),
        "joint_score": joint_score(selected),
        "best_joint_score": best_score,
        "score_tie_tolerance": float(score_tie_tolerance),
        "candidates": [
            {
                "label": str(candidate["label"]),
                "median_box_iou": float(candidate["projection"]["median_box_iou"]),
                "box_support_rate": float(candidate["projection"]["box_support_rate"]),
                "voxel_inside_rate": float(candidate["voxel_inside_rate"]),
                "volume_expansion": float(candidate["volume_expansion"]),
                "dimensions_m": np.asarray(candidate["obb"]["dimensions"]).round(5).tolist(),
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


def _refine_object(
    object_id: int,
    mask_paths: list[Path],
    *,
    state_images: list,
    voxel_points: np.ndarray,
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
    retained_by_centroid = [obs for obs, keep in zip(observations, consistent.tolist()) if keep]
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
    fit_observations = retained_by_reprojection if refit_applied else retained_by_centroid
    points = np.concatenate([obs["points"] for obs in fit_observations], axis=0)
    base_obb = _fit_robust_obb(
        points,
        quantile,
        orientation_mode=orientation_mode,
        up_axis_index=up_axis_index,
        up_vector=up_vector,
    )
    obb, projection, voxel_diagnostics = _select_voxel_supported_obb(
        base_obb,
        voxel_points,
        observations,
        support_iou=final_support_iou,
        min_support_rate=min_box_support_rate,
        min_projected_iou=min_median_projected_box_iou,
        expansion_penalty=voxel_expansion_penalty,
        score_tie_tolerance=score_tie_tolerance,
    )
    center = obb["center"]
    dimensions = obb["dimensions"]
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
        "voxel_inside_obb_rate": voxel_diagnostics["voxel_inside_rate"],
        "base_voxel_inside_obb_rate": voxel_diagnostics["base_voxel_inside_rate"],
        "voxel_volume_expansion": voxel_diagnostics["volume_expansion"],
        "voxel_extent_candidate": voxel_diagnostics["candidate"],
        "voxel_extent_candidates": voxel_diagnostics["candidates"],
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
    parser.add_argument("--voxel-expansion-penalty", type=float, default=0.15)
    parser.add_argument(
        "--candidate-score-tie-tolerance",
        type=float,
        default=0.005,
        help="Prefer the tighter OBB when its joint score is within this fraction of the best candidate.",
    )
    parser.add_argument("--min-material-orientation-confidence", type=float, default=0.15)
    parser.add_argument("--orientation-materiality-threshold", type=float, default=0.25)
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
    up_policy = policy_from_args(args)
    up_axis_index = up_policy.axis_index

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
        if index + 1 < voxel_offsets.size and index < voxel_levels.size:
            start, end = int(voxel_offsets[index]), int(voxel_offsets[index + 1])
            if 0 <= start < end <= voxel_flat.size:
                voxel_points = np.asarray(
                    decode_voxel_keys_numpy(voxel_flat[start:end], int(voxel_levels[index])),
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
            seed=int(args.seed),
        )
        valid = int(result.get("valid_observations", 0))
        consistent = int(result.get("consistent_observations", 0))
        consistency = float(result.get("observation_consistency", 0.0))
        inside = float(result.get("center_inside_detection_rate", 0.0))
        norm_error = float(result.get("median_reprojection_error_normalized", float("inf")))
        projected_box_iou = float(result.get("median_projected_box_iou", 0.0))
        box_support_rate = float(result.get("box_support_rate", 0.0))
        voxel_inside_rate = float(result.get("voxel_inside_obb_rate", 0.0))
        voxel_volume_expansion = float(result.get("voxel_volume_expansion", 1.0))
        voxel_extent_candidate = str(result.get("voxel_extent_candidate", "rgbd"))
        orientation_confidence = float(result.get("orientation_confidence", 0.0))
        candidate_dimensions = _numpy(result.get("dimensions_m", [1.0, 1.0, 1.0]), np.float32)
        material_dimensions = (
            candidate_dimensions[:2]
            if str(args.orientation_mode) == "gravity_yaw"
            else candidate_dimensions
        )
        orientation_materiality = float(
            1.0 - np.min(material_dimensions) / max(float(np.max(material_dimensions)), 1.0e-6)
        )
        orientation_required = orientation_materiality >= float(args.orientation_materiality_threshold)
        orientation_reliable = (
            not orientation_required
            or orientation_confidence >= float(args.min_material_orientation_confidence)
        )
        passed = (
            result.get("status") == "refined"
            and valid >= int(args.min_depth_observations)
            and consistency >= float(args.min_consistency)
            and inside >= float(args.min_center_inside_rate)
            and norm_error <= float(args.max_normalized_reprojection_error)
            and projected_box_iou >= float(args.min_median_projected_box_iou)
            and box_support_rate >= float(args.min_box_support_rate)
            and voxel_inside_rate >= float(args.min_voxel_inside_rate)
            and orientation_reliable
        )
        if result.get("status") == "refined":
            box_centers[index] = result["center_m"]
            box_dimensions[index] = result["dimensions_m"]
            box_wxyz[index] = result["wxyz"]
        is_assembly = index in assembly_indices
        status = (
            "assembly_geometry_pass" if passed else "assembly_geometry_rejected"
        ) if is_assembly else ("geometry_pass" if passed else "geometry_rejected")
        statuses[index] = status
        active[index] = bool(passed)
        inside_rates[index] = inside
        normalized_errors[index] = norm_error
        valid_observations[index] = valid
        consistent_observations[index] = consistent
        projected_box_ious[index] = projected_box_iou
        box_support_rates[index] = box_support_rate
        voxel_inside_rates[index] = voxel_inside_rate
        voxel_volume_expansions[index] = voxel_volume_expansion
        voxel_extent_candidates[index] = voxel_extent_candidate
        orientation_confidences[index] = orientation_confidence
        gravity_tilt_degrees[index] = float(result.get("gravity_tilt_degrees", float("nan")))
        row = {
            "object_id": object_id,
            "category": str((state.get("object_category") or [""] * count)[index] or ""),
            "status": status,
            "is_assembly": is_assembly,
            "assembly_member_ids": list(assembly_members[index]) if is_assembly else [],
            "observation_selection": observation_selection,
            "valid_observations": valid,
            "consistent_observations": consistent,
            "centroid_consistent_observations": int(result.get("centroid_consistent_observations", consistent)),
            "reprojection_refit_applied": bool(result.get("reprojection_refit_applied", False)),
            "observation_consistency": consistency,
            "fit_observation_rate": float(result.get("fit_observation_rate", consistency)),
            "center_inside_detection_rate": inside,
            "median_reprojection_error_normalized": norm_error if math.isfinite(norm_error) else None,
            "median_reprojection_error_px": result.get("median_reprojection_error_px"),
            "center_shift_m": float(np.linalg.norm(box_centers[index] - means[index])),
            "box_center_m": box_centers[index].round(5).tolist(),
            "box_dimensions_m": box_dimensions[index].round(5).tolist(),
            "box_wxyz": box_wxyz[index].round(7).tolist(),
            "orientation_confidence": float(orientation_confidences[index]),
            "orientation_mode": str(result.get("orientation_mode", args.orientation_mode)),
            "up_axis": up_policy.legacy_state_axis,
            "up_direction": up_policy.direction,
            "up_vector": up_policy.vector.astype(float).tolist(),
            "gravity_alignment": float(result.get("gravity_alignment", 0.0)),
            "gravity_tilt_degrees": float(gravity_tilt_degrees[index]),
            "orientation_materiality": orientation_materiality,
            "orientation_required": orientation_required,
            "orientation_reliable": orientation_reliable,
            "obb_eigenvalues": _numpy(result.get("obb_eigenvalues", [0.0, 0.0, 0.0])).round(7).tolist(),
            "median_projected_box_iou": projected_box_iou,
            "q25_projected_box_iou": float(result.get("q25_projected_box_iou", 0.0)),
            "box_support_rate": box_support_rate,
            "voxel_inside_obb_rate": voxel_inside_rate,
            "base_voxel_inside_obb_rate": float(result.get("base_voxel_inside_obb_rate", 0.0)),
            "voxel_volume_expansion": voxel_volume_expansion,
            "voxel_extent_candidate": voxel_extent_candidate,
            "voxel_extent_candidates": result.get("voxel_extent_candidates", []),
        }
        original_dimensions = 5.0 * np.sqrt(np.clip(cov6[index, [0, 3, 5]], 1.0e-4, None))
        row["original_box_dimensions_5sigma_m"] = original_dimensions.round(5).tolist()
        row["original_box_volume_m3"] = float(np.prod(original_dimensions))
        row["refined_box_volume_m3"] = (
            float(np.prod(box_dimensions[index])) if np.isfinite(box_dimensions[index]).all() else None
        )
        rows.append(row)

    state["active"] = torch.as_tensor(active, dtype=torch.bool)
    state["object_box_centers_m"] = torch.as_tensor(box_centers, dtype=torch.float32)
    state["object_box_dimensions_m"] = torch.as_tensor(box_dimensions, dtype=torch.float32)
    state["object_box_wxyz"] = torch.as_tensor(box_wxyz, dtype=torch.float32)
    state["object_geometry_status"] = statuses
    state["object_geometry_inside_rate"] = torch.as_tensor(inside_rates, dtype=torch.float32)
    state["object_geometry_reprojection_error"] = torch.as_tensor(normalized_errors, dtype=torch.float32)
    state["object_geometry_valid_observations"] = torch.as_tensor(valid_observations, dtype=torch.int32)
    state["object_geometry_consistent_observations"] = torch.as_tensor(consistent_observations, dtype=torch.int32)
    state["object_geometry_projected_box_iou"] = torch.as_tensor(projected_box_ious, dtype=torch.float32)
    state["object_geometry_box_support_rate"] = torch.as_tensor(box_support_rates, dtype=torch.float32)
    state["object_geometry_voxel_inside_rate"] = torch.as_tensor(voxel_inside_rates, dtype=torch.float32)
    state["object_geometry_voxel_volume_expansion"] = torch.as_tensor(voxel_volume_expansions, dtype=torch.float32)
    state["object_geometry_voxel_extent_candidate"] = voxel_extent_candidates
    state["object_geometry_orientation_confidence"] = torch.as_tensor(orientation_confidences, dtype=torch.float32)
    state["object_geometry_gravity_tilt_degrees"] = torch.as_tensor(gravity_tilt_degrees, dtype=torch.float32)
    prior_modes = state.get("object_geometry_orientation_mode")
    orientation_modes = list(prior_modes) if isinstance(prior_modes, (list, tuple)) else [str(args.orientation_mode)] * count
    if len(orientation_modes) < count:
        orientation_modes.extend([str(args.orientation_mode)] * (count - len(orientation_modes)))
    for index in evaluation_indices:
        orientation_modes[index] = str(args.orientation_mode)
    state["object_geometry_orientation_mode"] = orientation_modes
    write_state_up_policy(state, up_policy)
    passed_evaluated = sum(str(row.get("status", "")).endswith("_pass") for row in rows)
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["geometry_refinement"] = {
            "source": str(source),
            "automatic": True,
            "orientation_mode": str(args.orientation_mode),
            "up": up_policy.to_dict(),
            "passed_evaluated_objects": int(passed_evaluated),
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
        "passed_objects": int(passed_evaluated),
        "rejected_objects": int(len(rows) - passed_evaluated),
        "active_objects": int(active.sum()),
        "assembly_only": bool(args.assembly_only),
        "object_id_filter": (
            str(args.object_id_file.expanduser().resolve())
            if args.object_id_file is not None else None
        ),
        "up": up_policy.to_dict(),
        "mask_observation_contract": mask_index.diagnostics,
        "timing": {
            "duration_seconds": duration_seconds,
            "seconds_per_evaluated_object": duration_seconds / max(len(rows), 1),
            "stage": "multi_view_rgbd_obb_refinement",
        },
        "thresholds": {
            "preferred_observation_source": str(args.preferred_observation_source or "") or None,
            "minimum_preferred_observations": int(args.minimum_preferred_observations),
            "min_depth_observations": int(args.min_depth_observations),
            "min_consistency": float(args.min_consistency),
            "min_center_inside_rate": float(args.min_center_inside_rate),
            "max_normalized_reprojection_error": float(args.max_normalized_reprojection_error),
            "min_median_projected_box_iou": float(args.min_median_projected_box_iou),
            "min_refit_box_iou": float(args.min_refit_box_iou),
            "max_refit_center_error": float(args.max_refit_center_error),
            "final_support_iou": float(args.final_support_iou),
            "min_box_support_rate": float(args.min_box_support_rate),
            "min_voxel_inside_rate": float(args.min_voxel_inside_rate),
            "voxel_expansion_penalty": float(args.voxel_expansion_penalty),
            "candidate_score_tie_tolerance": float(args.candidate_score_tie_tolerance),
            "min_material_orientation_confidence": float(args.min_material_orientation_confidence),
            "orientation_materiality_threshold": float(args.orientation_materiality_threshold),
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
