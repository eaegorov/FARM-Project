#!/usr/bin/env python3
"""Conservatively recover geometry for rejected FARM objects.

This stage is deliberately independent from the online mapper.  It copies an
already refined scene state, freezes every active ``geometry_pass`` row, and
only inspects direct ``geometry_rejected`` objects using their saved raw masks
and the original metric depth maps.  No detector, VLM, semantic vocabulary, or
text prompt is used.

The default mode is audit-only.  ``--activate`` may promote a candidate only
when it passes every geometry gate, retains at least the requested fraction of
saved views, and Pareto-dominates the rejected baseline.  Otherwise the source
row is preserved exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


IMAGE_ID_RE = re.compile(r"img_(\d+)")
@dataclass(frozen=True)
class GeometryGates:
    min_depth_observations: int = 3
    min_consistency: float = 0.65
    min_center_inside_rate: float = 0.80
    max_normalized_reprojection_error: float = 0.30
    min_median_projected_box_iou: float = 0.35
    min_box_support_rate: float = 0.70
    min_voxel_inside_rate: float = 0.55
    min_material_orientation_confidence: float = 0.15
    orientation_materiality_threshold: float = 0.25


def _numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _record_value(record: object, key: str, default: object = None) -> object:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _load_refiner_helpers() -> Any:
    """Import the existing tested OBB/refinement helpers in both entry modes."""
    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root, project_root / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return importlib.import_module("scripts.refine_farm_object_geometry")


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


def _unpack_raw_mask(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray] | None:
    required = {"raw_bits", "raw_shape", "raw_bbox_xyxy"}
    if not required.issubset(data.files):
        return None
    height, width = np.asarray(data["raw_shape"], dtype=np.int32).reshape(2).tolist()
    bbox = np.asarray(data["raw_bbox_xyxy"], dtype=np.int32).reshape(4)
    if height <= 0 or width <= 0:
        return None
    flat = np.unpackbits(np.asarray(data["raw_bits"], dtype=np.uint8), bitorder="little")
    mask = flat[: int(height) * int(width)].reshape(int(height), int(width)).astype(bool, copy=False)
    return mask, bbox


def adaptive_erode_mask(
    mask: np.ndarray,
    *,
    min_retention: float = 0.55,
    min_pixels: int = 24,
) -> tuple[np.ndarray, int, float]:
    """Use one-pixel erosion only when it does not destroy a thin mask.

    This is intentionally bounded to radii 0/1.  The retained-area test is a
    content-independent proxy for mask thickness and fails closed to radius 0.
    """
    source = np.asarray(mask, dtype=bool)
    source_count = int(source.sum())
    if source_count < max(1, int(min_pixels)):
        return source.copy(), 0, 1.0
    eroded = binary_erosion(source, structure=np.ones((3, 3), dtype=bool), iterations=1, border_value=0)
    retained = float(eroded.sum() / max(source_count, 1))
    if int(eroded.sum()) < int(min_pixels) or retained < float(min_retention):
        return source.copy(), 0, retained
    return np.asarray(eroded, dtype=bool), 1, retained


def depth_mad_keep(
    depth_values: np.ndarray,
    *,
    k_mad: float = 1.0,
    min_mad_m: float = 0.03,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the strict depth-mode mask used by the recovery candidate."""
    depth = np.asarray(depth_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(depth) & (depth > 0.05) & (depth < 80.0)
    keep = np.zeros(depth.shape, dtype=bool)
    if not finite.any():
        return keep, {"median_depth_m": math.nan, "raw_mad_m": math.nan, "used_mad_m": math.nan}
    valid_depth = depth[finite]
    median = float(np.median(valid_depth))
    raw_mad = float(np.median(np.abs(valid_depth - median)))
    used_mad = max(raw_mad, float(min_mad_m))
    keep = finite & (depth >= median - float(k_mad) * used_mad) & (depth <= median + float(k_mad) * used_mad)
    return keep, {
        "median_depth_m": median,
        "raw_mad_m": raw_mad,
        "used_mad_m": used_mad,
    }


def largest_voxel_component_indices(
    points: np.ndarray,
    *,
    voxel_size_m: float,
    max_voxels: int = 12_000,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Find the largest 26-connected metric component with bounded cost.

    If a mask creates too many occupied cells, the voxel size is increased
    deterministically until the graph is bounded by ``max_voxels``.  Components
    are ranked by represented point count rather than voxel count.
    """
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_indices = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    xyz = xyz[finite_indices]
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), {
            "component_points": 0,
            "component_fraction": 0.0,
            "occupied_voxels": 0,
            "voxel_size_m": float(voxel_size_m),
        }

    size = max(float(voxel_size_m), 1.0e-4)
    unique = inverse = counts = None
    for _ in range(6):
        quantized = np.floor(xyz / size).astype(np.int64)
        unique, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
        if unique.shape[0] <= max(1, int(max_voxels)):
            break
        scale = max(1.10, (unique.shape[0] / max(float(max_voxels), 1.0)) ** (1.0 / 3.0) * 1.05)
        size *= scale
    assert unique is not None and inverse is not None and counts is not None

    if unique.shape[0] == 1:
        labels = np.zeros((1,), dtype=np.int32)
    else:
        pairs = cKDTree(unique.astype(np.float64, copy=False)).query_pairs(
            r=1.01,
            p=np.inf,
            output_type="ndarray",
        )
        if pairs.size == 0:
            labels = np.arange(unique.shape[0], dtype=np.int32)
        else:
            rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
            columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
            graph = coo_matrix((np.ones(rows.shape[0], dtype=np.uint8), (rows, columns)), shape=(unique.shape[0], unique.shape[0]))
            _, labels = connected_components(graph.tocsr(), directed=False, return_labels=True)
    component_weights = np.bincount(labels, weights=counts.astype(np.float64))
    best_label = int(np.flatnonzero(component_weights == component_weights.max())[0])
    selected_voxels = labels == best_label
    local_indices = np.flatnonzero(selected_voxels[inverse])
    selected = finite_indices[local_indices].astype(np.int64, copy=False)
    return selected, {
        "component_points": int(selected.size),
        "component_fraction": float(selected.size / max(xyz.shape[0], 1)),
        "occupied_voxels": int(unique.shape[0]),
        "voxel_size_m": float(size),
    }


def robust_view_consensus(
    centroids: np.ndarray,
    robust_diagonals: np.ndarray,
    *,
    scale_fraction: float = 0.25,
    min_threshold_m: float = 0.08,
    max_threshold_m: float = 0.30,
) -> tuple[np.ndarray, float]:
    """Select the largest compact medoid neighbourhood across views.

    Unlike a transitive graph component, a medoid neighbourhood cannot absorb
    a chain of progressively drifting background centroids.
    """
    centers = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    diagonals = np.asarray(robust_diagonals, dtype=np.float64).reshape(-1)
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), float(min_threshold_m)
    finite_diagonals = diagonals[np.isfinite(diagonals) & (diagonals > 0.0)]
    scale = float(np.median(finite_diagonals)) if finite_diagonals.size else float(min_threshold_m / max(scale_fraction, 1.0e-6))
    threshold = float(np.clip(float(scale_fraction) * scale, float(min_threshold_m), float(max_threshold_m)))
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    best = np.asarray([0], dtype=np.int64)
    best_dispersion = math.inf
    for seed in range(centers.shape[0]):
        members = np.flatnonzero(pairwise[seed] <= threshold)
        for _ in range(2):
            consensus = np.median(centers[members], axis=0)
            refined = np.flatnonzero(np.linalg.norm(centers - consensus, axis=1) <= threshold)
            if np.array_equal(refined, members) or refined.size == 0:
                break
            members = refined
        consensus = np.median(centers[members], axis=0)
        dispersion = float(np.mean(np.linalg.norm(centers[members] - consensus, axis=1)))
        if members.size > best.size or (
            members.size == best.size
            and (dispersion < best_dispersion - 1.0e-12 or (abs(dispersion - best_dispersion) <= 1.0e-12 and tuple(members) < tuple(best)))
        ):
            best = members.astype(np.int64, copy=False)
            best_dispersion = dispersion
    return best, threshold


def rejected_geometry_indices(statuses: list[str] | tuple[str, ...], count: int) -> list[int]:
    """Return direct rejected rows only; assemblies and pending rows are excluded."""
    return [index for index in range(min(int(count), len(statuses))) if str(statuses[index]) == "geometry_rejected"]


def _finite_metric(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def assess_candidate(
    baseline: dict[str, object],
    candidate: dict[str, object],
    gates: GeometryGates,
    *,
    max_volume_ratio: float = 1.25,
    pareto_tolerance: float = 1.0e-4,
    min_strict_improvement: float = 5.0e-3,
) -> dict[str, object]:
    """Fail-closed hard-gate and Pareto assessment for one recovered OBB."""
    hard_failures: list[str] = []
    if str(candidate.get("status")) != "refined":
        hard_failures.append("candidate_not_refined")
    if int(candidate.get("valid_observations", 0)) < int(gates.min_depth_observations):
        hard_failures.append("depth_observations")
    if float(candidate.get("view_retention_rate", 0.0)) < float(gates.min_consistency):
        hard_failures.append("view_retention")
    if float(candidate.get("observation_consistency", 0.0)) < float(gates.min_consistency):
        hard_failures.append("observation_consistency")
    if float(candidate.get("center_inside_detection_rate", 0.0)) < float(gates.min_center_inside_rate):
        hard_failures.append("center_inside")
    if float(candidate.get("median_reprojection_error_normalized", math.inf)) > float(gates.max_normalized_reprojection_error):
        hard_failures.append("reprojection_error")
    if float(candidate.get("median_projected_box_iou", 0.0)) < float(gates.min_median_projected_box_iou):
        hard_failures.append("projected_box_iou")
    if float(candidate.get("box_support_rate", 0.0)) < float(gates.min_box_support_rate):
        hard_failures.append("box_support")
    if float(candidate.get("voxel_inside_obb_rate", 0.0)) < float(gates.min_voxel_inside_rate):
        hard_failures.append("voxel_inside")
    if bool(candidate.get("orientation_required", False)) and float(candidate.get("orientation_confidence", 0.0)) < float(gates.min_material_orientation_confidence):
        hard_failures.append("orientation_confidence")

    candidate_volume = _finite_metric(candidate.get("box_volume_m3"))
    baseline_volume = _finite_metric(baseline.get("box_volume_m3"))
    volume_ratio = None
    if candidate_volume is None or candidate_volume <= 0.0:
        hard_failures.append("invalid_volume")
    elif baseline_volume is not None and baseline_volume > 0.0:
        volume_ratio = candidate_volume / baseline_volume
        if volume_ratio > float(max_volume_ratio) + float(pareto_tolerance):
            hard_failures.append("volume_expansion")

    regressions: list[str] = []
    improvements: list[str] = []
    maximize = (
        "observation_consistency",
        "center_inside_detection_rate",
        "median_projected_box_iou",
        "box_support_rate",
        "voxel_inside_obb_rate",
    )
    minimize = ("median_reprojection_error_normalized",)
    for key in maximize:
        old, new = _finite_metric(baseline.get(key)), _finite_metric(candidate.get(key))
        if old is None or new is None:
            continue
        if new + float(pareto_tolerance) < old:
            regressions.append(key)
        if new > old + float(min_strict_improvement):
            improvements.append(key)
    for key in minimize:
        old, new = _finite_metric(baseline.get(key)), _finite_metric(candidate.get(key))
        if new is None:
            regressions.append(key)
            continue
        if old is None:
            improvements.append(key)
            continue
        if new > old + float(pareto_tolerance):
            regressions.append(key)
        if new < old - float(min_strict_improvement):
            improvements.append(key)

    if bool(candidate.get("orientation_required", False)):
        old = _finite_metric(baseline.get("orientation_confidence"))
        new = _finite_metric(candidate.get("orientation_confidence"))
        if old is not None and new is not None and new + float(pareto_tolerance) < old:
            regressions.append("orientation_confidence")
        elif old is not None and new is not None and new > old + float(min_strict_improvement):
            improvements.append("orientation_confidence")

    eligible = not hard_failures and not regressions and bool(improvements)
    return {
        "eligible": bool(eligible),
        "hard_gate_failures": hard_failures,
        "pareto_regressions": sorted(set(regressions)),
        "strict_improvements": sorted(set(improvements)),
        "volume_ratio_vs_baseline": volume_ratio,
    }


def _hash_bytes(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def _fingerprint_update(digest: Any, value: object) -> None:
    """Hash nested state values without serialization/storage identifiers."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"torch")
        _hash_bytes(digest, str(array.dtype).encode())
        _hash_bytes(digest, repr(array.shape).encode())
        _hash_bytes(digest, array.tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy")
        _hash_bytes(digest, str(array.dtype).encode())
        _hash_bytes(digest, repr(array.shape).encode())
        _hash_bytes(digest, array.tobytes(order="C"))
        return
    if isinstance(value, np.generic):
        _fingerprint_update(digest, np.asarray(value))
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"bool1" if value else b"bool0")
        return
    if isinstance(value, int):
        digest.update(b"int")
        _hash_bytes(digest, str(value).encode())
        return
    if isinstance(value, float):
        digest.update(b"float")
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        digest.update(b"str")
        _hash_bytes(digest, value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        digest.update(b"bytes")
        _hash_bytes(digest, value)
        return
    if isinstance(value, Path):
        digest.update(b"path")
        _hash_bytes(digest, str(value).encode("utf-8"))
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list" if isinstance(value, list) else b"tuple")
        digest.update(len(value).to_bytes(8, byteorder="little", signed=False))
        for item in value:
            _fingerprint_update(digest, item)
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        keyed = sorted(((_fingerprint(key), key) for key in value), key=lambda row: row[0])
        digest.update(len(keyed).to_bytes(8, byteorder="little", signed=False))
        for _, key in keyed:
            _fingerprint_update(digest, key)
            _fingerprint_update(digest, value[key])
        return
    digest.update(b"fallback")
    _hash_bytes(digest, f"{type(value).__module__}.{type(value).__qualname__}".encode())
    _hash_bytes(digest, repr(value).encode("utf-8"))


def _fingerprint(value: object) -> str:
    digest = hashlib.blake2b(digest_size=20)
    _fingerprint_update(digest, value)
    return digest.hexdigest()


def snapshot_indexed_rows(state: dict, indices: list[int], count: int) -> dict[str, str]:
    """Fingerprint every state field indexed by object row for an invariant."""
    snapshot: dict[str, str] = {}
    for key, value in state.items():
        selected: object | None = None
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == count:
            selected = value[indices].clone()
        elif isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == count:
            selected = value[indices].copy()
        elif isinstance(value, (list, tuple)) and len(value) == count:
            selected = [value[index] for index in indices]
        if selected is not None:
            snapshot[key] = _fingerprint(selected)
    return snapshot


def assert_indexed_rows_unchanged(state: dict, indices: list[int], count: int, expected: dict[str, str]) -> None:
    actual = snapshot_indexed_rows(state, indices, count)
    if actual != expected:
        changed = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        raise AssertionError(f"Protected object rows changed in fields: {changed}")


def _robust_diagonal(points: np.ndarray) -> float:
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] < 2:
        return 0.0
    lo, hi = np.quantile(xyz, [0.10, 0.90], axis=0)
    return float(np.linalg.norm(hi - lo))


def _raw_observation_points(
    mask_path: Path,
    *,
    state_images: list,
    frames_root: Path,
    frames_by_stem: dict[str, dict],
    max_points: int,
    min_component_fraction: float,
    adaptive_erosion_min_retention: float,
    k_mad: float,
    min_mad_m: float,
    base_component_voxel_m: float,
    max_component_voxels: int,
    max_component_input_points: int,
    seed: int,
) -> tuple[dict | None, dict]:
    diagnostics: dict[str, object] = {"mask_file": mask_path.name, "status": "invalid"}
    match = IMAGE_ID_RE.search(mask_path.name)
    if match is None:
        diagnostics["reason"] = "missing_image_id"
        return None, diagnostics
    image_id = int(match.group(1))
    if image_id < 0 or image_id >= len(state_images):
        diagnostics["reason"] = "image_id_out_of_range"
        return None, diagnostics
    record = state_images[image_id]
    source_ref = str(_record_value(record, "source_ref", "") or _record_value(record, "storage_path", ""))
    frame = frames_by_stem.get(Path(source_ref).stem)
    if frame is None:
        diagnostics["reason"] = "missing_frame"
        return None, diagnostics
    depth_rel = str(frame.get("depth_path") or "")
    depth_path = frames_root / depth_rel
    if not depth_rel or not depth_path.is_file():
        diagnostics["reason"] = "missing_depth"
        return None, diagnostics
    pose = _numpy(_record_value(record, "pose"), np.float64)
    intrinsics = np.asarray(frame.get("K"), dtype=np.float64)
    if pose.shape != (4, 4) or intrinsics.shape != (3, 3):
        diagnostics["reason"] = "invalid_camera"
        return None, diagnostics

    with np.load(mask_path, allow_pickle=False) as data:
        packed = _unpack_raw_mask(data)
    if packed is None:
        diagnostics["reason"] = "missing_raw_mask"
        return None, diagnostics
    raw_mask, bbox = packed
    mask, erosion_radius, erosion_retention = adaptive_erode_mask(
        raw_mask,
        min_retention=adaptive_erosion_min_retention,
        min_pixels=24,
    )
    depth = np.load(depth_path, mmap_mode="r")
    if depth.ndim != 2:
        diagnostics["reason"] = "invalid_depth"
        return None, diagnostics
    x0, y0, x1, y1 = [int(value) for value in bbox]
    height = min(mask.shape[0], max(0, depth.shape[0] - y0), max(0, y1 - y0))
    width = min(mask.shape[1], max(0, depth.shape[1] - x0), max(0, x1 - x0))
    if height <= 0 or width <= 0:
        diagnostics["reason"] = "mask_outside_depth"
        return None, diagnostics
    ys_local, xs_local = np.nonzero(mask[:height, :width])
    ys, xs = ys_local + y0, xs_local + x0
    z_all = np.asarray(depth[ys, xs], dtype=np.float32)
    depth_keep, depth_stats = depth_mad_keep(z_all, k_mad=k_mad, min_mad_m=min_mad_m)
    ys, xs, z = ys[depth_keep], xs[depth_keep], z_all[depth_keep]
    diagnostics.update({
        "raw_pixels": int(raw_mask.sum()),
        "candidate_mask_pixels": int(mask.sum()),
        "erosion_radius_px": int(erosion_radius),
        "erosion_one_pixel_retention": float(erosion_retention),
        "depth_mode_pixels": int(z.size),
        **depth_stats,
    })
    if z.size < 24:
        diagnostics["reason"] = "insufficient_depth_mode_points"
        return None, diagnostics

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if min(abs(fx), abs(fy)) < 1.0e-9:
        diagnostics["reason"] = "invalid_intrinsics"
        return None, diagnostics

    # Oversized failed masks can cover almost the full 896-pixel image.  A
    # regular image-grid sample preserves surface connectivity while bounding
    # the 3D component graph; the final OBB uses at most 1,400 points anyway.
    source_depth_points = int(z.size)
    component_limit = max(24, int(max_component_input_points))
    sampling_stride = max(1, int(math.ceil(math.sqrt(source_depth_points / component_limit))))
    if sampling_stride > 1:
        sampled = np.flatnonzero(
            ((xs - x0) % sampling_stride == 0)
            & ((ys - y0) % sampling_stride == 0)
        )
        if sampled.size < 24:
            sampled = np.linspace(0, source_depth_points - 1, min(component_limit, source_depth_points), dtype=np.int64)
        elif sampled.size > component_limit:
            rng = np.random.default_rng(int(seed) + image_id * 104729)
            sampled = np.sort(rng.choice(sampled, size=component_limit, replace=False))
        ys, xs, z = ys[sampled], xs[sampled], z[sampled]
    diagnostics["component_input_points"] = int(z.size)
    diagnostics["component_input_fraction"] = float(z.size / max(source_depth_points, 1))
    diagnostics["component_sampling_stride_px"] = int(sampling_stride)

    points_cam = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
    points_world = points_cam @ pose[:3, :3].T + pose[:3, 3]
    finite = np.isfinite(points_world).all(axis=1)
    points_world = points_world[finite].astype(np.float32, copy=False)
    if points_world.shape[0] < 24:
        diagnostics["reason"] = "insufficient_world_points"
        return None, diagnostics

    metric_pixel = float(np.median(z)) * 0.5 * (1.0 / abs(fx) + 1.0 / abs(fy))
    initial_voxel_size = float(
        np.clip(max(float(base_component_voxel_m), 4.0 * metric_pixel * sampling_stride), 0.02, 0.12)
    )
    component_indices, component_stats = largest_voxel_component_indices(
        points_world,
        voxel_size_m=initial_voxel_size,
        max_voxels=max_component_voxels,
    )
    points_world = points_world[component_indices]
    diagnostics.update(component_stats)
    if points_world.shape[0] < 24 or float(component_stats["component_fraction"]) < float(min_component_fraction):
        diagnostics["reason"] = "weak_largest_3d_component"
        return None, diagnostics

    if max_points > 0 and points_world.shape[0] > max_points:
        rng = np.random.default_rng(int(seed) + image_id * 104729)
        selected = rng.choice(points_world.shape[0], size=int(max_points), replace=False)
        points_world = points_world[selected]
    diagnostics["sampled_component_points"] = int(points_world.shape[0])
    diagnostics["status"] = "valid"
    diagnostics.pop("reason", None)
    raw_bbox = np.asarray([x0, y0, x1, y1], dtype=np.float64)
    return {
        "image_id": image_id,
        "points": points_world,
        "centroid": np.median(points_world, axis=0),
        "robust_diagonal_m": _robust_diagonal(points_world),
        "raw_bbox": raw_bbox,
        "pose": pose,
        "K": intrinsics,
    }, diagnostics


def _candidate_from_observations(
    observations: list[dict],
    *,
    total_saved_views: int,
    voxel_points: np.ndarray,
    refiner: Any,
    gates: GeometryGates,
    up_axis_index: int,
    quantile: float,
    final_support_iou: float,
    min_refit_box_iou: float,
    max_refit_center_error: float,
    voxel_expansion_penalty: float,
    score_tie_tolerance: float,
) -> dict[str, object]:
    if len(observations) < int(gates.min_depth_observations):
        return {
            "status": "insufficient_depth",
            "valid_observations": len(observations),
            "total_saved_views": int(total_saved_views),
            "view_retention_rate": 0.0,
        }
    centroids = np.stack([row["centroid"] for row in observations])
    diagonals = np.asarray([row["robust_diagonal_m"] for row in observations], dtype=np.float64)
    cluster_indices, consensus_threshold = robust_view_consensus(centroids, diagonals)
    retained = [observations[int(index)] for index in cluster_indices]
    minimum_views = max(
        int(gates.min_depth_observations),
        int(math.ceil(float(gates.min_consistency) * max(int(total_saved_views), 1))),
    )
    if len(retained) < minimum_views:
        return {
            "status": "insufficient_view_consensus",
            "valid_observations": len(observations),
            "consistent_observations": len(retained),
            "total_saved_views": int(total_saved_views),
            "view_retention_rate": float(len(retained) / max(int(total_saved_views), 1)),
            "observation_consistency": float(len(retained) / max(int(total_saved_views), 1)),
            "centroid_consensus_threshold_m": consensus_threshold,
        }

    points = np.concatenate([row["points"] for row in retained], axis=0)
    initial = refiner._fit_robust_obb(
        points,
        float(quantile),
        orientation_mode="gravity_yaw",
        up_axis_index=int(up_axis_index),
    )
    _, view_metrics = refiner._projection_metrics(
        initial["center"], initial["corners"], retained, support_iou=float(final_support_iou)
    )
    reprojection_retained = [
        observation
        for observation, metric in zip(retained, view_metrics)
        if (
            float(metric["inside"]) > 0.5
            and float(metric["normalized_error"]) <= float(max_refit_center_error)
            and float(metric["box_iou"]) >= float(min_refit_box_iou)
        )
    ]
    fit_observations = reprojection_retained if len(reprojection_retained) >= minimum_views else retained
    points = np.concatenate([row["points"] for row in fit_observations], axis=0)
    base_obb = refiner._fit_robust_obb(
        points,
        float(quantile),
        orientation_mode="gravity_yaw",
        up_axis_index=int(up_axis_index),
    )
    obb, projection, voxel_diagnostics = refiner._select_voxel_supported_obb(
        base_obb,
        voxel_points,
        observations,
        support_iou=float(final_support_iou),
        min_support_rate=float(gates.min_box_support_rate),
        min_projected_iou=float(gates.min_median_projected_box_iou),
        expansion_penalty=float(voxel_expansion_penalty),
        score_tie_tolerance=float(score_tie_tolerance),
    )
    dimensions = np.asarray(obb["dimensions"], dtype=np.float32)
    horizontal = dimensions[:2]
    materiality = float(1.0 - np.min(horizontal) / max(float(np.max(horizontal)), 1.0e-6))
    orientation_required = materiality >= float(gates.orientation_materiality_threshold)
    final_count = len(fit_observations)
    view_rate = float(final_count / max(int(total_saved_views), 1))
    return {
        "status": "refined",
        "valid_observations": len(observations),
        "consistent_observations": final_count,
        "total_saved_views": int(total_saved_views),
        "view_retention_rate": view_rate,
        "observation_consistency": view_rate,
        "centroid_consensus_observations": len(retained),
        "centroid_consensus_threshold_m": consensus_threshold,
        "reprojection_refit_applied": fit_observations is reprojection_retained,
        "center_m": np.asarray(obb["center"], dtype=np.float32),
        "dimensions_m": dimensions,
        "wxyz": np.asarray(obb["wxyz"], dtype=np.float32),
        "box_volume_m3": float(np.prod(dimensions)),
        "orientation_confidence": float(obb["orientation_confidence"]),
        "orientation_required": orientation_required,
        "orientation_materiality": materiality,
        "gravity_alignment": float(obb["gravity_alignment"]),
        "gravity_tilt_degrees": float(obb["gravity_tilt_degrees"]),
        "center_inside_detection_rate": float(projection["center_inside_rate"]),
        "median_reprojection_error_normalized": float(projection["median_normalized_error"]),
        "median_reprojection_error_px": float(projection["median_pixel_error"]),
        "median_projected_box_iou": float(projection["median_box_iou"]),
        "q25_projected_box_iou": float(projection["q25_box_iou"]),
        "box_support_rate": float(projection["box_support_rate"]),
        "voxel_inside_obb_rate": float(voxel_diagnostics["voxel_inside_rate"]),
        "base_voxel_inside_obb_rate": float(voxel_diagnostics["base_voxel_inside_rate"]),
        "voxel_volume_expansion": float(voxel_diagnostics["volume_expansion"]),
        "voxel_extent_candidate": str(voxel_diagnostics["candidate"]),
        "voxel_extent_candidates": voxel_diagnostics["candidates"],
    }


def _baseline_metrics(state: dict, index: int, count: int) -> dict[str, object]:
    def scalar(key: str, default: float) -> float:
        value = state.get(key)
        if value is None:
            return float(default)
        array = _numpy(value)
        return float(array[index]) if array.shape == (count,) else float(default)

    def row(key: str, width: int, default: float = math.nan) -> np.ndarray:
        value = state.get(key)
        if value is None:
            return np.full((width,), default, dtype=np.float32)
        array = _numpy(value, np.float32)
        return array[index].copy() if array.shape == (count, width) else np.full((width,), default, dtype=np.float32)

    valid = int(round(scalar("object_geometry_valid_observations", 0.0)))
    consistent = int(round(scalar("object_geometry_consistent_observations", 0.0)))
    dimensions = row("object_box_dimensions_m", 3)
    volume = float(np.prod(dimensions)) if np.isfinite(dimensions).all() and np.all(dimensions > 0.0) else math.nan
    return {
        "valid_observations": valid,
        "consistent_observations": consistent,
        "observation_consistency": float(consistent / max(valid, 1)),
        "center_inside_detection_rate": scalar("object_geometry_inside_rate", 0.0),
        "median_reprojection_error_normalized": scalar("object_geometry_reprojection_error", math.inf),
        "median_projected_box_iou": scalar("object_geometry_projected_box_iou", 0.0),
        "box_support_rate": scalar("object_geometry_box_support_rate", 0.0),
        "voxel_inside_obb_rate": scalar("object_geometry_voxel_inside_rate", 0.0),
        "orientation_confidence": scalar("object_geometry_orientation_confidence", 0.0),
        "box_center_m": row("object_box_centers_m", 3).round(6).tolist(),
        "box_dimensions_m": dimensions.round(6).tolist(),
        "box_wxyz": row("object_box_wxyz", 4).round(7).tolist(),
        "box_volume_m3": volume,
    }


def _json_candidate(candidate: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in candidate.items():
        if isinstance(value, np.ndarray):
            result[key] = value.round(7).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        elif isinstance(value, float) and not math.isfinite(value):
            result[key] = None
        else:
            result[key] = value
    return result


def _json_safe(value: object) -> object:
    """Recursively convert NumPy values and non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _state_array(state: dict, key: str, shape: tuple[int, ...], dtype: np.dtype, fill: float) -> np.ndarray:
    value = state.get(key)
    if value is not None:
        array = _numpy(value, dtype)
        if array.shape == shape:
            return array.copy()
    return np.full(shape, fill, dtype=dtype)


def main() -> int:
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--activate", action="store_true", help="Promote eligible candidates; default is audit-only.")
    parser.add_argument("--min-depth-observations", type=int, default=3)
    parser.add_argument("--min-consistency", type=float, default=0.65)
    parser.add_argument("--min-center-inside-rate", type=float, default=0.80)
    parser.add_argument("--max-normalized-reprojection-error", type=float, default=0.30)
    parser.add_argument("--min-median-projected-box-iou", type=float, default=0.35)
    parser.add_argument("--final-support-iou", type=float, default=0.20)
    parser.add_argument("--min-box-support-rate", type=float, default=0.70)
    parser.add_argument("--min-voxel-inside-rate", type=float, default=0.55)
    parser.add_argument("--min-material-orientation-confidence", type=float, default=0.15)
    parser.add_argument("--orientation-materiality-threshold", type=float, default=0.25)
    parser.add_argument("--up-axis", choices=("auto", "x", "y", "z"), default="auto")
    parser.add_argument("--max-points-per-observation", type=int, default=1400)
    parser.add_argument("--quantile", type=float, default=0.05)
    parser.add_argument("--depth-mode-k-mad", type=float, default=1.0)
    parser.add_argument("--depth-mode-min-mad-m", type=float, default=0.03)
    parser.add_argument("--adaptive-erosion-min-retention", type=float, default=0.55)
    parser.add_argument("--min-component-fraction", type=float, default=0.25)
    parser.add_argument("--component-voxel-m", type=float, default=0.025)
    parser.add_argument("--max-component-voxels", type=int, default=12000)
    parser.add_argument("--max-component-input-points", type=int, default=20000)
    parser.add_argument("--min-refit-box-iou", type=float, default=0.08)
    parser.add_argument("--max-refit-center-error", type=float, default=0.45)
    parser.add_argument("--voxel-expansion-penalty", type=float, default=0.15)
    parser.add_argument("--candidate-score-tie-tolerance", type=float, default=0.005)
    parser.add_argument("--max-volume-ratio", type=float, default=1.25)
    parser.add_argument("--pareto-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--min-strict-improvement", type=float, default=5.0e-3)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    source = args.scene_state.expanduser().resolve()
    output_state = args.output_state.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    if output_state == source:
        raise ValueError("--output-state must not overwrite --scene-state")
    if output_report == source:
        raise ValueError("--output-report must not overwrite --scene-state")

    load_started = time.perf_counter()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    if not isinstance(state, dict):
        raise TypeError("Scene state payload is not a dictionary")
    object_ids = _numpy(state["object_id"], np.int64)
    count = int(object_ids.shape[0])
    active_source = _numpy(state["active"]).astype(bool, copy=False)
    statuses_source = list(state.get("object_geometry_status") or [])
    if len(statuses_source) < count:
        statuses_source.extend(["not_evaluated"] * (count - len(statuses_source)))
    frozen_indices = [
        index
        for index in range(count)
        if bool(active_source[index]) and str(statuses_source[index]) == "geometry_pass"
    ]
    frozen_snapshot = snapshot_indexed_rows(state, frozen_indices, count)
    evaluation_indices = rejected_geometry_indices(statuses_source, count)
    mutable_indices = set(evaluation_indices) if bool(args.activate) else set()
    protected_indices = [index for index in range(count) if index not in mutable_indices]
    protected_snapshot = snapshot_indexed_rows(state, protected_indices, count)
    frames_root, frames_by_stem = _load_frames(args.frames_json.expanduser().resolve())
    mask_dir = args.mask_dir.expanduser().resolve()
    refiner = _load_refiner_helpers()
    load_seconds = time.perf_counter() - load_started

    gates = GeometryGates(
        min_depth_observations=int(args.min_depth_observations),
        min_consistency=float(args.min_consistency),
        min_center_inside_rate=float(args.min_center_inside_rate),
        max_normalized_reprojection_error=float(args.max_normalized_reprojection_error),
        min_median_projected_box_iou=float(args.min_median_projected_box_iou),
        min_box_support_rate=float(args.min_box_support_rate),
        min_voxel_inside_rate=float(args.min_voxel_inside_rate),
        min_material_orientation_confidence=float(args.min_material_orientation_confidence),
        orientation_materiality_threshold=float(args.orientation_materiality_threshold),
    )
    state_up_axis = str(state.get("object_geometry_up_axis") or "y")
    up_axis = state_up_axis if str(args.up_axis) == "auto" else str(args.up_axis)
    if up_axis not in {"x", "y", "z"}:
        raise ValueError(f"Invalid state up axis: {up_axis!r}")
    up_axis_index = {"x": 0, "y": 1, "z": 2}[up_axis]
    state_images = list(state.get("images") or [])
    voxel_flat = _numpy(state.get("object_voxel_keys_flat", []), np.int64)
    voxel_offsets = _numpy(state.get("object_voxel_keys_offsets", []), np.int64)
    voxel_levels = _numpy(state.get("object_voxel_levels", []), np.int64)

    active = active_source.copy()
    statuses = list(statuses_source)
    box_centers = _state_array(state, "object_box_centers_m", (count, 3), np.float32, math.nan)
    box_dimensions = _state_array(state, "object_box_dimensions_m", (count, 3), np.float32, math.nan)
    box_wxyz = _state_array(state, "object_box_wxyz", (count, 4), np.float32, math.nan)
    inside_rates = _state_array(state, "object_geometry_inside_rate", (count,), np.float32, 0.0)
    normalized_errors = _state_array(state, "object_geometry_reprojection_error", (count,), np.float32, math.inf)
    valid_observations = _state_array(state, "object_geometry_valid_observations", (count,), np.int32, 0)
    consistent_observations = _state_array(state, "object_geometry_consistent_observations", (count,), np.int32, 0)
    projected_box_ious = _state_array(state, "object_geometry_projected_box_iou", (count,), np.float32, 0.0)
    box_support_rates = _state_array(state, "object_geometry_box_support_rate", (count,), np.float32, 0.0)
    voxel_inside_rates = _state_array(state, "object_geometry_voxel_inside_rate", (count,), np.float32, 0.0)
    voxel_volume_expansions = _state_array(state, "object_geometry_voxel_volume_expansion", (count,), np.float32, 1.0)
    orientation_confidences = _state_array(state, "object_geometry_orientation_confidence", (count,), np.float32, 0.0)
    gravity_tilt_degrees = _state_array(state, "object_geometry_gravity_tilt_degrees", (count,), np.float32, math.nan)
    prior_extent = state.get("object_geometry_voxel_extent_candidate")
    extent_candidates = list(prior_extent) if isinstance(prior_extent, (list, tuple)) else ["not_evaluated"] * count
    if len(extent_candidates) < count:
        extent_candidates.extend(["not_evaluated"] * (count - len(extent_candidates)))
    prior_modes = state.get("object_geometry_orientation_mode")
    orientation_modes = list(prior_modes) if isinstance(prior_modes, (list, tuple)) else ["gravity_yaw"] * count
    if len(orientation_modes) < count:
        orientation_modes.extend(["gravity_yaw"] * (count - len(orientation_modes)))

    rows: list[dict[str, object]] = []
    eligible_count = 0
    activated_count = 0
    evaluation_started = time.perf_counter()
    for index in evaluation_indices:
        object_started = time.perf_counter()
        object_id = int(object_ids[index])
        object_mask_dir = mask_dir / f"object_{object_id:06d}"
        mask_paths = sorted(object_mask_dir.glob("*.npz"))
        observations: list[dict] = []
        observation_diagnostics: list[dict] = []
        for mask_path in mask_paths:
            observation, diagnostics = _raw_observation_points(
                mask_path,
                state_images=state_images,
                frames_root=frames_root,
                frames_by_stem=frames_by_stem,
                max_points=max(0, int(args.max_points_per_observation)),
                min_component_fraction=float(args.min_component_fraction),
                adaptive_erosion_min_retention=float(args.adaptive_erosion_min_retention),
                k_mad=float(args.depth_mode_k_mad),
                min_mad_m=float(args.depth_mode_min_mad_m),
                base_component_voxel_m=float(args.component_voxel_m),
                max_component_voxels=max(1, int(args.max_component_voxels)),
                max_component_input_points=max(24, int(args.max_component_input_points)),
                seed=int(args.seed) + object_id * 1009,
            )
            observation_diagnostics.append(diagnostics)
            if observation is not None:
                observations.append(observation)

        voxel_points = np.zeros((0, 3), dtype=np.float32)
        if index + 1 < voxel_offsets.size and index < voxel_levels.size:
            start, end = int(voxel_offsets[index]), int(voxel_offsets[index + 1])
            if 0 <= start < end <= voxel_flat.size:
                voxel_points = np.asarray(
                    refiner.decode_voxel_keys_numpy(voxel_flat[start:end], int(voxel_levels[index])),
                    dtype=np.float32,
                ).reshape(-1, 3)
        candidate = _candidate_from_observations(
            observations,
            total_saved_views=len(mask_paths),
            voxel_points=voxel_points,
            refiner=refiner,
            gates=gates,
            up_axis_index=up_axis_index,
            quantile=float(args.quantile),
            final_support_iou=float(args.final_support_iou),
            min_refit_box_iou=float(args.min_refit_box_iou),
            max_refit_center_error=float(args.max_refit_center_error),
            voxel_expansion_penalty=float(args.voxel_expansion_penalty),
            score_tie_tolerance=float(args.candidate_score_tie_tolerance),
        )
        baseline = _baseline_metrics(state, index, count)
        assessment = assess_candidate(
            baseline,
            candidate,
            gates,
            max_volume_ratio=float(args.max_volume_ratio),
            pareto_tolerance=float(args.pareto_tolerance),
            min_strict_improvement=float(args.min_strict_improvement),
        )
        eligible = bool(assessment["eligible"])
        eligible_count += int(eligible)
        activated = bool(args.activate and eligible)
        if activated:
            active[index] = True
            statuses[index] = "geometry_pass"
            box_centers[index] = candidate["center_m"]
            box_dimensions[index] = candidate["dimensions_m"]
            box_wxyz[index] = candidate["wxyz"]
            inside_rates[index] = float(candidate["center_inside_detection_rate"])
            normalized_errors[index] = float(candidate["median_reprojection_error_normalized"])
            valid_observations[index] = int(candidate["valid_observations"])
            consistent_observations[index] = int(candidate["consistent_observations"])
            projected_box_ious[index] = float(candidate["median_projected_box_iou"])
            box_support_rates[index] = float(candidate["box_support_rate"])
            voxel_inside_rates[index] = float(candidate["voxel_inside_obb_rate"])
            voxel_volume_expansions[index] = float(candidate["voxel_volume_expansion"])
            extent_candidates[index] = f"recovery:{candidate['voxel_extent_candidate']}"
            orientation_confidences[index] = float(candidate["orientation_confidence"])
            gravity_tilt_degrees[index] = float(candidate["gravity_tilt_degrees"])
            orientation_modes[index] = "gravity_yaw"
            activated_count += 1

        category_values = state.get("object_category") or []
        category = str(category_values[index] or "") if index < len(category_values) else ""
        rows.append({
            "object_id": object_id,
            "index": int(index),
            "category": category,
            "source_status": "geometry_rejected",
            "result": "activated" if activated else ("eligible_audit_only" if eligible else "preserved_rejected"),
            "candidate_mode": "raw_adaptive_e01_depth_mad1_largest3d_gravity_yaw",
            "baseline": baseline,
            "candidate": _json_candidate(candidate),
            "assessment": assessment,
            "saved_mask_views": len(mask_paths),
            "valid_candidate_views": len(observations),
            "observation_diagnostics": observation_diagnostics,
            "timing_seconds": time.perf_counter() - object_started,
        })
    evaluation_seconds = time.perf_counter() - evaluation_started

    state["active"] = torch.as_tensor(active, dtype=torch.bool)
    state["object_geometry_status"] = statuses
    state["object_box_centers_m"] = torch.as_tensor(box_centers, dtype=torch.float32)
    state["object_box_dimensions_m"] = torch.as_tensor(box_dimensions, dtype=torch.float32)
    state["object_box_wxyz"] = torch.as_tensor(box_wxyz, dtype=torch.float32)
    state["object_geometry_inside_rate"] = torch.as_tensor(inside_rates, dtype=torch.float32)
    state["object_geometry_reprojection_error"] = torch.as_tensor(normalized_errors, dtype=torch.float32)
    state["object_geometry_valid_observations"] = torch.as_tensor(valid_observations, dtype=torch.int32)
    state["object_geometry_consistent_observations"] = torch.as_tensor(consistent_observations, dtype=torch.int32)
    state["object_geometry_projected_box_iou"] = torch.as_tensor(projected_box_ious, dtype=torch.float32)
    state["object_geometry_box_support_rate"] = torch.as_tensor(box_support_rates, dtype=torch.float32)
    state["object_geometry_voxel_inside_rate"] = torch.as_tensor(voxel_inside_rates, dtype=torch.float32)
    state["object_geometry_voxel_volume_expansion"] = torch.as_tensor(voxel_volume_expansions, dtype=torch.float32)
    state["object_geometry_voxel_extent_candidate"] = extent_candidates
    state["object_geometry_orientation_confidence"] = torch.as_tensor(orientation_confidences, dtype=torch.float32)
    state["object_geometry_gravity_tilt_degrees"] = torch.as_tensor(gravity_tilt_degrees, dtype=torch.float32)
    state["object_geometry_orientation_mode"] = orientation_modes
    assert_indexed_rows_unchanged(state, frozen_indices, count, frozen_snapshot)
    assert_indexed_rows_unchanged(state, protected_indices, count, protected_snapshot)

    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["rejected_geometry_recovery"] = {
            "source": str(source),
            "mode": "activate" if args.activate else "audit_only",
            "candidate_mode": "raw_adaptive_e01_depth_mad1_largest3d_gravity_yaw",
            "evaluated_rejected_objects": len(evaluation_indices),
            "eligible_candidates": int(eligible_count),
            "activated_candidates": int(activated_count),
            "frozen_geometry_pass_objects": len(frozen_indices),
            "protected_non_candidate_objects": len(protected_indices),
        }
    else:
        payload = state

    report = {
        "schema": "farm.rejected-geometry-recovery.v1",
        "created_unix_s": time.time(),
        "source_scene_state": str(source),
        "output_scene_state": str(output_state),
        "mode": "activate" if args.activate else "audit_only",
        "candidate_mode": "raw_adaptive_e01_depth_mad1_largest3d_gravity_yaw",
        "evaluated_rejected_objects": len(evaluation_indices),
        "eligible_candidates": int(eligible_count),
        "activated_candidates": int(activated_count),
        "preserved_rejected_objects": int(len(evaluation_indices) - activated_count),
        "frozen_geometry_pass_objects": len(frozen_indices),
        "frozen_rows_verified": True,
        "protected_non_candidate_objects": len(protected_indices),
        "protected_rows_verified": True,
        "active_objects_before": int(active_source.sum()),
        "active_objects_after": int(active.sum()),
        "gates": asdict(gates),
        "candidate_parameters": {
            "up_axis": up_axis,
            "quantile": float(args.quantile),
            "depth_mode_k_mad": float(args.depth_mode_k_mad),
            "depth_mode_min_mad_m": float(args.depth_mode_min_mad_m),
            "adaptive_erosion_radii_px": [0, 1],
            "adaptive_erosion_min_retention": float(args.adaptive_erosion_min_retention),
            "min_component_fraction": float(args.min_component_fraction),
            "component_voxel_m": float(args.component_voxel_m),
            "max_component_voxels": int(args.max_component_voxels),
            "max_component_input_points": int(args.max_component_input_points),
            "max_volume_ratio": float(args.max_volume_ratio),
            "pareto_tolerance": float(args.pareto_tolerance),
            "min_strict_improvement": float(args.min_strict_improvement),
        },
        "timing": {
            "load_seconds": load_seconds,
            "candidate_evaluation_seconds": evaluation_seconds,
            "seconds_per_rejected_object": evaluation_seconds / max(len(evaluation_indices), 1),
        },
        "objects": rows,
    }
    output_state.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    torch.save(payload, output_state)
    report["timing"]["state_write_seconds"] = time.perf_counter() - write_started
    report["timing"]["total_seconds"] = time.perf_counter() - total_started
    output_report.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "mode": report["mode"],
        "evaluated_rejected_objects": report["evaluated_rejected_objects"],
        "eligible_candidates": report["eligible_candidates"],
        "activated_candidates": report["activated_candidates"],
        "frozen_rows_verified": report["frozen_rows_verified"],
        "protected_rows_verified": report["protected_rows_verified"],
        "total_seconds": report["timing"]["total_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
