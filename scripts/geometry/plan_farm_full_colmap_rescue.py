#!/usr/bin/env python3
"""Plan a bounded full-COLMAP rescue pass for uncertain scene objects.

The normal pipeline maps a trajectory-connected subset of physical timestamps.
This planner searches the complete registered pinhole reconstruction only for
objects whose multi-view geometry is weak.  It does not run segmentation and
does not mutate the scene state.  The resulting exact image list is consumed by
the later RGB-D/mask rescue step.

Ranking is category-agnostic.  The default retrieval path matches metric object
support to COLMAP sparse points and follows their image tracks; a fitted OBB is
therefore not a prerequisite and cannot constrain the camera shortlist.  An OBB
projection path remains available only as an explicit compatibility fallback.
Train and heldout cameras are split by physical timestamp before selection.
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
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.full_colmap_view_planner import (  # noqa: E402
    allocate_global_view_budget as allocate_global_view_budget_core,
    allocate_split_global_view_budget as allocate_split_global_view_budget_core,
    angular_separation_degrees as angular_separation_degrees_core,
    bounded_support_points,
    build_sparse_track_index,
    choose_diverse_views as choose_diverse_views_core,
    choose_next_best_views as choose_next_best_views_core,
    choose_train_heldout_views as choose_train_heldout_views_core,
    match_support_to_sparse_tracks,
    model_file_manifest,
    physical_timestamp_fold,
    plan_temporal_episode_topup,
    project_metric_support,
    summarize_track_counts,
    temporal_episode_frame_count,
    track_evidence_tier,
    validate_disjoint_timestamp_folds,
)
from farm_runtime.full_colmap_rescue import sha256_file  # noqa: E402
from scene_graph.utils.geometry import decode_voxel_keys_numpy  # noqa: E402


def _numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def quaternion_wxyz_to_matrix(value: object) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("invalid OBB quaternion")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def obb_corners(center_m: object, dimensions_m: object, wxyz: object) -> np.ndarray:
    center = np.asarray(center_m, dtype=np.float64).reshape(3)
    dimensions = np.asarray(dimensions_m, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all() or not np.isfinite(dimensions).all():
        raise ValueError("non-finite OBB")
    if np.any(dimensions <= 0.0):
        raise ValueError("non-positive OBB dimensions")
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = 0.5 * signs * dimensions
    return center[None, :] + local @ quaternion_wxyz_to_matrix(wxyz).T


def camera_intrinsics(camera: Any) -> np.ndarray:
    model_name = getattr(camera, "model_name", None)
    model_value = getattr(camera, "model", None)
    model = str(
        model_name or getattr(model_value, "name", model_value) or ""
    ).split(".")[-1].upper()
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if model == "PINHOLE" and params.size >= 4:
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE" and params.size >= 3:
        fx, cx, cy = params[:3]
        fy = fx
    else:
        raise ValueError(f"full-COLMAP rescue requires PINHOLE cameras, got {model!r}")
    if min(float(fx), float(fy)) <= 0.0:
        raise ValueError("camera focal length must be positive")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def image_world_to_camera(image: Any) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world.matrix(), dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"unexpected COLMAP pose shape {matrix.shape}")
    return matrix


def project_obb(
    corners_m: np.ndarray,
    image: Any,
    camera: Any,
    *,
    meters_per_scene_unit: float,
    min_margin_ratio: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict[str, object] | None:
    if meters_per_scene_unit <= 0.0:
        raise ValueError("meters_per_scene_unit must be positive")
    corners_scene = np.asarray(corners_m, dtype=np.float64) / float(meters_per_scene_unit)
    world_to_camera = image_world_to_camera(image)
    camera_points = corners_scene @ world_to_camera[:, :3].T + world_to_camera[:, 3]
    depths = camera_points[:, 2]
    if not np.isfinite(camera_points).all() or np.any(depths <= 1.0e-5):
        return None
    K = camera_intrinsics(camera)
    pixels_h = camera_points @ K.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    x0, y0 = pixels.min(axis=0)
    x1, y1 = pixels.max(axis=0)
    width, height = int(camera.width), int(camera.height)
    bbox_width = float(x1 - x0)
    bbox_height = float(y1 - y0)
    if bbox_width <= 1.0 or bbox_height <= 1.0:
        return None
    area_ratio = bbox_width * bbox_height / float(width * height)
    margin_px = min(float(x0), float(y0), float(width - x1), float(height - y1))
    margin_ratio = margin_px / float(max(1, min(width, height)))
    if (
        area_ratio < float(min_area_ratio)
        or area_ratio > float(max_area_ratio)
        or margin_ratio < float(min_margin_ratio)
    ):
        return None
    camera_to_world = np.linalg.inv(
        np.vstack((world_to_camera, np.asarray([0.0, 0.0, 0.0, 1.0])))
    )
    camera_center_m = camera_to_world[:3, 3] * float(meters_per_scene_unit)
    object_center_m = np.asarray(corners_m, dtype=np.float64).mean(axis=0)
    view = object_center_m - camera_center_m
    distance_m = float(np.linalg.norm(view))
    if not math.isfinite(distance_m) or distance_m <= 1.0e-5:
        return None
    view_direction = view / distance_m
    center_offset = float(
        np.linalg.norm(
            np.asarray([(x0 + x1) * 0.5 / width - 0.5, (y0 + y1) * 0.5 / height - 0.5])
        )
    )
    geometric_score = (
        math.sqrt(area_ratio)
        * (1.0 + min(margin_ratio, 0.25))
        / (1.0 + center_offset)
    )
    return {
        "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "camera_center_m": camera_center_m.tolist(),
        "area_ratio": float(area_ratio),
        "margin_ratio": float(margin_ratio),
        "distance_m": distance_m,
        "view_direction": view_direction.tolist(),
        "geometric_score": float(geometric_score),
    }


def crop_sharpness(image_path: Path, bbox_xyxy: Iterable[float]) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    x0, y0, x1, y1 = [int(round(value)) for value in bbox_xyxy]
    height, width = image.shape[:2]
    pad_x = max(8, int(round(0.08 * max(1, x1 - x0))))
    pad_y = max(8, int(round(0.08 * max(1, y1 - y0))))
    x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
    crop = image[y0:y1, x0:x1]
    if crop.size < 256:
        return 0.0
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def safe_image_path(root: Path, name: str) -> Path:
    root = root.expanduser().resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"COLMAP image path escapes image root: {name!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def identity_key(
    name: str,
    pattern: re.Pattern[str] | None,
    *,
    require_match: bool = False,
) -> tuple[str, str, str]:
    if pattern is None:
        if require_match:
            raise ValueError(
                "physical-timestamp split requires identity_contract.regex"
            )
        return Path(name).stem, "", ""
    match = pattern.search(name)
    if match is None:
        if require_match:
            raise ValueError(
                f"COLMAP image does not satisfy identity contract: {name!r}"
            )
        return Path(name).stem, "", ""
    groups = match.groupdict()
    timestamp = str(groups.get("timestamp") or Path(name).stem)
    camera = str(groups.get("camera") or "")
    family = str(groups.get("family") or groups.get("view") or "")
    return timestamp, camera, family


def build_camera_stream_timeline(
    reconstruction: object,
    pattern: re.Pattern[str] | None,
    *,
    require_match: bool,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Index every full-COLMAP image inside its camera-by-family stream."""

    streams: dict[str, list[tuple[tuple[int, int | str], str]]] = {}
    for image in reconstruction.images.values():
        name = str(image.name)
        timestamp, camera, family = identity_key(
            name, pattern, require_match=require_match
        )
        order: tuple[int, int | str] = (
            (0, int(timestamp)) if timestamp.isdigit() else (1, timestamp)
        )
        streams.setdefault(f"{camera}_{family}", []).append((order, name))
    timeline: dict[str, int] = {}
    names_by_stream: dict[str, list[str]] = {}
    for stream, rows in streams.items():
        names_by_stream[stream] = [name for _, name in sorted(rows)]
        for index, name in enumerate(names_by_stream[stream]):
            timeline[name] = index
    return timeline, names_by_stream


def build_projected_temporal_context(
    reconstruction: object,
    support_points_m: np.ndarray,
    seeds: list[dict[str, object]],
    *,
    stream_names: dict[str, list[str]],
    stream_timeline: dict[str, int],
    identity_pattern: re.Pattern[str] | None,
    require_identity_match: bool,
    existing_names: set[str],
    existing_timestamps: set[str],
    heldout_fraction: float,
    split_seed: str,
    meters_per_scene_unit: float,
    maximum_neighbor_frame_gap: int,
    minimum_projected_support_points: int,
    minimum_in_frame_support_ratio: float,
    min_margin_ratio: float,
    min_area_ratio: float,
    max_area_ratio: float,
    support_bbox_quantile: float,
) -> list[dict[str, object]]:
    """Project only bounded stream neighbours; they can never become seeds."""

    images_by_name = {
        str(image.name): image for image in reconstruction.images.values()
    }
    context: dict[str, dict[str, object]] = {}
    for seed in seeds:
        name = str(seed.get("name") or "")
        stream = str(seed.get("camera_stream_id") or "")
        seed_index = stream_timeline.get(name)
        names = stream_names.get(stream, [])
        if seed_index is None:
            continue
        for delta in range(
            -int(maximum_neighbor_frame_gap),
            int(maximum_neighbor_frame_gap) + 1,
        ):
            if delta == 0:
                continue
            index = int(seed_index) + delta
            if index < 0 or index >= len(names):
                continue
            neighbour_name = names[index]
            timestamp, camera_name, family = identity_key(
                neighbour_name,
                identity_pattern,
                require_match=require_identity_match,
            )
            if (
                neighbour_name in existing_names
                or timestamp in existing_timestamps
                or physical_timestamp_fold(
                    timestamp,
                    heldout_fraction=float(heldout_fraction),
                    split_seed=str(split_seed),
                )
                != "train"
            ):
                continue
            image = images_by_name[neighbour_name]
            camera = reconstruction.cameras[image.camera_id]
            projected = project_metric_support(
                support_points_m,
                image_world_to_camera(image),
                camera_intrinsics(camera),
                width=int(camera.width),
                height=int(camera.height),
                meters_per_scene_unit=float(meters_per_scene_unit),
                min_margin_ratio=float(min_margin_ratio),
                min_area_ratio=float(min_area_ratio),
                max_area_ratio=float(max_area_ratio),
                min_in_frame_ratio=float(minimum_in_frame_support_ratio),
                bbox_quantile=float(support_bbox_quantile),
            )
            if (
                projected is None
                or int(projected["projected_support_points"])
                < int(minimum_projected_support_points)
            ):
                continue
            projected["geometric_score"] = (
                float(projected["geometric_score"]) * 0.60
            )
            context[neighbour_name] = {
                "name": neighbour_name,
                "colmap_image_id": int(image.image_id),
                "colmap_camera_id": int(image.camera_id),
                "physical_timestamp": timestamp,
                "camera": camera_name,
                "family": family,
                "camera_stream_id": stream,
                "camera_stream_frame_index": index,
                "retrieval_source": "bounded_temporal_projection_context",
                "bbox_source": "projected_object_support",
                "track_evidence_tier": "temporal_projection_only",
                "track_support_count": 0,
                "track_support_fraction": 0.0,
                "temporal_context_only": True,
                "sharpness": None,
                "score": float(projected["geometric_score"]),
                "split": "train",
                **projected,
            }
    return list(context.values())


def angular_separation_degrees(a: object, b: object) -> float:
    return angular_separation_degrees_core(a, b)


def choose_diverse_views(
    candidates: list[dict[str, object]],
    *,
    limit: int,
    min_view_angle_degrees: float,
) -> list[dict[str, object]]:
    return choose_diverse_views_core(
        candidates,
        limit=int(limit),
        min_view_angle_degrees=float(min_view_angle_degrees),
    )


def object_support_points(
    state: dict,
    index: int,
    *,
    trim_quantile: float,
    max_points: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Decode the canonical object voxel cloud without consulting its OBB."""

    flat = _numpy(state.get("object_voxel_keys_flat", []), np.int64).reshape(-1)
    offsets = _numpy(state.get("object_voxel_keys_offsets", []), np.int64).reshape(-1)
    levels = _numpy(state.get("object_voxel_levels", []), np.int64).reshape(-1)
    if index < 0 or index >= levels.size or index + 1 >= offsets.size:
        return np.zeros((0, 3), dtype=np.float64), {
            "source": "object_voxels",
            "status": "missing",
            "bounded_points": 0,
        }
    start, end = int(offsets[index]), int(offsets[index + 1])
    if start < 0 or end <= start or end > flat.size:
        return np.zeros((0, 3), dtype=np.float64), {
            "source": "object_voxels",
            "status": "invalid_offsets",
            "bounded_points": 0,
        }
    decoded = decode_voxel_keys_numpy(flat[start:end], int(levels[index]))
    support, diagnostics = bounded_support_points(
        decoded,
        trim_quantile=float(trim_quantile),
        max_points=int(max_points),
    )
    return support, {
        "source": "object_voxels",
        "status": "available" if support.size else "empty",
        "voxel_level": int(levels[index]),
        **diagnostics,
    }


def choose_train_heldout_views(
    candidates: list[dict[str, object]],
    *,
    train_limit: int,
    heldout_limit: int,
    min_view_angle_degrees: float,
    heldout_fraction: float,
    split_seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split physical timestamps globally before view ranking/selection."""

    return choose_train_heldout_views_core(
        candidates,
        train_limit=int(train_limit),
        heldout_limit=int(heldout_limit),
        min_view_angle_degrees=float(min_view_angle_degrees),
        heldout_fraction=float(heldout_fraction),
        split_seed=str(split_seed),
    )


def choose_heldout_reference_views(
    candidates: list[dict[str, object]],
    *,
    heldout_limit: int,
    min_view_angle_degrees: float,
    heldout_fraction: float,
    split_seed: str,
    minimum_views: int = 1,
    min_camera_baseline_m: float = 0.15,
    min_baseline_distance_ratio: float = 0.04,
    relaxed_view_angle_degrees: float = 6.0,
    relaxed_camera_baseline_m: float = 0.08,
    relaxed_baseline_distance_ratio: float = 0.02,
    priority: float = 0.0,
    return_diagnostics: bool = False,
) -> list[dict[str, object]] | tuple[list[dict[str, object]], dict[str, object]]:
    """Choose only the deterministic heldout partition using bounded NBV."""

    heldout_pool: list[dict[str, object]] = []
    for source in candidates:
        row = dict(source)
        timestamp = str(row.get("physical_timestamp") or "")
        if physical_timestamp_fold(
            timestamp,
            heldout_fraction=float(heldout_fraction),
            split_seed=str(split_seed),
        ) != "heldout":
            continue
        row["split"] = "heldout"
        heldout_pool.append(row)
    if int(heldout_limit) < 1:
        raise ValueError("heldout reference limit must be positive")
    selected, diagnostics = choose_next_best_views_core(
        heldout_pool,
        minimum_views=int(minimum_views),
        maximum_views=int(heldout_limit),
        min_view_angle_degrees=float(min_view_angle_degrees),
        min_camera_baseline_m=float(min_camera_baseline_m),
        min_baseline_distance_ratio=float(min_baseline_distance_ratio),
        relaxed_view_angle_degrees=min(
            float(relaxed_view_angle_degrees), float(min_view_angle_degrees)
        ),
        relaxed_camera_baseline_m=float(relaxed_camera_baseline_m),
        relaxed_baseline_distance_ratio=float(relaxed_baseline_distance_ratio),
        priority=float(priority),
    )
    diagnostics["fold"] = "heldout"
    diagnostics["fold_candidate_rows"] = len(heldout_pool)
    return (selected, diagnostics) if return_diagnostics else selected


def choose_train_heldout_next_best_views(
    candidates: list[dict[str, object]],
    *,
    train_limit: int,
    heldout_limit: int,
    minimum_train_views: int,
    minimum_heldout_views: int,
    min_view_angle_degrees: float,
    min_camera_baseline_m: float,
    min_baseline_distance_ratio: float,
    relaxed_view_angle_degrees: float,
    relaxed_camera_baseline_m: float,
    relaxed_baseline_distance_ratio: float,
    heldout_fraction: float,
    split_seed: str,
    priority: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Partition physical timestamps before independent train/heldout NBV."""

    pools: dict[str, list[dict[str, object]]] = {"train": [], "heldout": []}
    for source in candidates:
        row = dict(source)
        fold = physical_timestamp_fold(
            str(row.get("physical_timestamp") or ""),
            heldout_fraction=float(heldout_fraction),
            split_seed=str(split_seed),
        )
        row["split"] = fold
        pools[fold].append(row)
    common = {
        "min_view_angle_degrees": float(min_view_angle_degrees),
        "min_camera_baseline_m": float(min_camera_baseline_m),
        "min_baseline_distance_ratio": float(min_baseline_distance_ratio),
        "relaxed_view_angle_degrees": float(relaxed_view_angle_degrees),
        "relaxed_camera_baseline_m": float(relaxed_camera_baseline_m),
        "relaxed_baseline_distance_ratio": float(relaxed_baseline_distance_ratio),
        "priority": float(priority),
    }
    train, train_audit = choose_next_best_views_core(
        pools["train"],
        minimum_views=int(minimum_train_views),
        maximum_views=int(train_limit),
        **common,
    )
    heldout, heldout_audit = choose_next_best_views_core(
        pools["heldout"],
        minimum_views=int(minimum_heldout_views),
        maximum_views=int(heldout_limit),
        **common,
    )
    train_audit["fold"] = "train"
    heldout_audit["fold"] = "heldout"
    validate_disjoint_timestamp_folds(train, heldout)
    return train, heldout, train_audit, heldout_audit


def finalize_planned_object_views(
    object_row: dict[str, object],
    train_views: list[dict[str, object]],
    heldout_views: list[dict[str, object]],
    *,
    heldout_reference_only: bool,
    minimum_train_views: int,
    minimum_heldout_views: int,
    minimum_temporal_episode_frames: int = 1,
) -> dict[str, object]:
    """Apply mode-specific minimums without ever promoting heldout to train."""

    validate_disjoint_timestamp_folds(train_views, heldout_views)
    rejection_reasons: list[str] = []
    if heldout_reference_only:
        if train_views:
            raise ValueError("heldout-reference-only planning produced train views")
    elif len(train_views) < int(minimum_train_views):
        rejection_reasons.append("insufficient_train_physical_timestamps")
    if (
        not heldout_reference_only
        and int(minimum_temporal_episode_frames) > 1
        and temporal_episode_frame_count(train_views)
        < int(minimum_temporal_episode_frames)
    ):
        rejection_reasons.append("insufficient_pose_continuous_temporal_episode")
    if len(heldout_views) < int(minimum_heldout_views):
        rejection_reasons.append("insufficient_heldout_physical_timestamps")
    if rejection_reasons:
        train_views = []
        heldout_views = []
        planning_status = "insufficient_independent_views"
    else:
        planning_status = (
            "planned_heldout_reference_only"
            if heldout_reference_only
            else "planned"
        )
    return {
        **{
            key: value
            for key, value in object_row.items()
            if key not in {"train_candidates", "heldout_candidates"}
        },
        "planning_status": planning_status,
        "planning_rejection_reasons": rejection_reasons,
        "selected_count": len(train_views),
        "selected_views": train_views,
        "train_views": train_views,
        "heldout_count": len(heldout_views),
        "heldout_views": heldout_views,
        "render_view_count": len(train_views) + len(heldout_views),
    }


def allocate_global_view_budget(
    object_rows: list[dict[str, object]],
    *,
    max_total_views: int,
) -> tuple[dict[str, dict[str, object]], list[list[dict[str, object]]]]:
    """Allocate scarce images round-robin so later objects are not starved."""

    return allocate_global_view_budget_core(
        [list(row.get("diverse_candidates") or []) for row in object_rows],
        max_total_views=int(max_total_views),
    )


def allocate_split_global_view_budget(
    object_rows: list[dict[str, object]],
    *,
    max_total_views: int,
    max_heldout_views: int,
) -> tuple[
    dict[str, dict[str, object]],
    list[list[dict[str, object]]],
    list[list[dict[str, object]]],
]:
    """Allocate a reserved heldout budget, then spend the remainder on train."""

    return allocate_split_global_view_budget_core(
        [list(row.get("train_candidates") or []) for row in object_rows],
        [list(row.get("heldout_candidates") or []) for row in object_rows],
        max_total_views=int(max_total_views),
        max_heldout_views=int(max_heldout_views),
    )


def build_support_track_candidates(
    reconstruction: object,
    sparse_index: object,
    support_points_m: np.ndarray,
    *,
    identity_pattern: re.Pattern[str] | None,
    require_identity_match: bool,
    existing_names: set[str],
    existing_timestamps: set[str],
    meters_per_scene_unit: float,
    maximum_track_distance_m: float,
    minimum_object_track_match_fraction: float,
    minimum_track_support_points: int,
    minimum_track_support_fraction: float,
    minimum_rescue_track_support_points: int,
    minimum_projected_support_points: int,
    minimum_rescue_projected_support_points: int,
    minimum_rescue_in_frame_support_ratio: float,
    min_margin_ratio: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_in_frame_support_ratio: float,
    support_bbox_quantile: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Retrieve candidate cameras from object-support COLMAP tracks."""

    image_counts, match_diagnostics = match_support_to_sparse_tracks(
        support_points_m,
        sparse_index,
        maximum_distance_m=float(maximum_track_distance_m),
    )
    matched_sparse_points = int(match_diagnostics["matched_sparse_points"] or 0)
    if float(match_diagnostics["matched_support_fraction"] or 0.0) < float(
        minimum_object_track_match_fraction
    ):
        return [], {
            **match_diagnostics,
            "retrieval_source": "object_support_colmap_tracks",
            "object_track_match_gate": False,
            "minimum_object_track_match_fraction": float(
                minimum_object_track_match_fraction
            ),
            "track_gate_rejections": 0,
            "projection_gate_rejections": 0,
            "eligible_images": 0,
        }
    evidence_by_image = summarize_track_counts(
        image_counts, matched_sparse_points=matched_sparse_points
    )
    candidates: list[dict[str, object]] = []
    rejected_by_track_gate = 0
    rejected_by_projection_gate = 0
    sparse_track_rescue_candidates = 0
    for image_id, evidence in sorted(evidence_by_image.items()):
        track_count = int(evidence["track_support_count"])
        track_fraction = float(evidence["track_support_fraction"])
        evidence_tier = track_evidence_tier(
            track_count=track_count,
            track_fraction=track_fraction,
            minimum_track_support_points=int(minimum_track_support_points),
            minimum_track_support_fraction=float(minimum_track_support_fraction),
            minimum_rescue_track_support_points=int(
                minimum_rescue_track_support_points
            ),
        )
        if evidence_tier is None:
            rejected_by_track_gate += 1
            continue
        try:
            image = reconstruction.images[int(image_id)]
            camera = reconstruction.cameras[int(image.camera_id)]
        except (KeyError, IndexError):
            rejected_by_projection_gate += 1
            continue
        name = str(image.name)
        timestamp, camera_name, family = identity_key(
            name,
            identity_pattern,
            require_match=bool(require_identity_match),
        )
        if name in existing_names or timestamp in existing_timestamps:
            continue
        projected = project_metric_support(
            support_points_m,
            image_world_to_camera(image),
            camera_intrinsics(camera),
            width=int(camera.width),
            height=int(camera.height),
            meters_per_scene_unit=float(meters_per_scene_unit),
            min_margin_ratio=float(min_margin_ratio),
            min_area_ratio=float(min_area_ratio),
            max_area_ratio=float(max_area_ratio),
            min_in_frame_ratio=float(min_in_frame_support_ratio),
            bbox_quantile=float(support_bbox_quantile),
        )
        if (
            projected is None
            or int(projected["projected_support_points"])
            < int(minimum_projected_support_points)
        ):
            rejected_by_projection_gate += 1
            continue
        if evidence_tier == "sparse_track_rescue" and (
            int(projected["projected_support_points"])
            < int(minimum_rescue_projected_support_points)
            or float(projected["in_frame_support_ratio"])
            < float(minimum_rescue_in_frame_support_ratio)
        ):
            rejected_by_projection_gate += 1
            continue
        sparse_track_rescue_candidates += int(evidence_tier == "sparse_track_rescue")
        track_strength = math.sqrt(
            min(1.0, track_count / max(float(minimum_track_support_points) * 2.0, 1.0))
            * min(
                1.0,
                track_fraction
                / max(float(minimum_track_support_fraction) * 2.0, 1.0e-6),
            )
        )
        projected["geometric_score"] = float(projected["geometric_score"]) * (
            0.65 + 0.35 * track_strength
        ) * (
            0.72 if evidence_tier == "sparse_track_rescue" else 1.0
        ) * (
            1.08 if family == "center" else 1.0
        )
        candidates.append(
            {
                "name": name,
                "colmap_image_id": int(image.image_id),
                "colmap_camera_id": int(image.camera_id),
                "physical_timestamp": timestamp,
                "camera": camera_name,
                "family": family,
                "retrieval_source": "object_support_colmap_tracks",
                "bbox_source": "projected_object_support",
                "track_evidence_tier": evidence_tier,
                **evidence,
                **projected,
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["geometric_score"]),
            int(row["track_support_count"]),
            str(row["name"]),
        ),
        reverse=True,
    )
    return candidates, {
        **match_diagnostics,
        "retrieval_source": "object_support_colmap_tracks",
        "object_track_match_gate": True,
        "minimum_object_track_match_fraction": float(
            minimum_object_track_match_fraction
        ),
        "track_gate_rejections": rejected_by_track_gate,
        "projection_gate_rejections": rejected_by_projection_gate,
        "sparse_track_rescue_candidates": sparse_track_rescue_candidates,
        "sparse_track_rescue_requires_post_render_depth_gate": True,
        "minimum_rescue_projected_support_points": int(
            minimum_rescue_projected_support_points
        ),
        "eligible_images": len(candidates),
    }


def object_rescue_priority(state: dict, index: int, observations: int) -> tuple[float, list[str]]:
    """Score only geometric evidence; semantic uncertainty has its own queue."""

    reasons: list[str] = []
    score = 0.0
    statuses = state.get("object_geometry_status") or []
    status = str(statuses[index] if index < len(statuses) else "")
    if "rejected" in status:
        score += 2.0
        reasons.append("geometry_rejected")
    elif not status.endswith("pass"):
        score += 2.0
        reasons.append("geometry_not_passed")
    if observations < 5:
        score += float(5 - observations)
        reasons.append(f"only_{observations}_observations")
    projected = _numpy(
        state.get("object_geometry_projected_box_iou", np.zeros((index + 1,))),
        np.float64,
    ).reshape(-1)
    if index < projected.size and float(projected[index]) < 0.45:
        score += 1.5
        reasons.append("weak_projected_box_iou")
    support = _numpy(
        state.get("object_geometry_box_support_rate", np.zeros((index + 1,))),
        np.float64,
    ).reshape(-1)
    if index < support.size and float(support[index]) < 0.75:
        score += 1.0
        reasons.append("weak_box_support")
    return score, reasons


def stratified_object_queue(
    rows: list[dict[str, object]], max_objects: int
) -> list[dict[str, object]]:
    """Reserve most refinement capacity for active objects, not rejected noise."""

    if max_objects < 1:
        raise ValueError("max_objects must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (float(row["priority"]), -int(row["object_id"])),
        reverse=True,
    )
    active_rows = [row for row in ordered if bool(row.get("active"))]
    inactive_rows = [row for row in ordered if not bool(row.get("active"))]
    inactive_limit = max(1, int(max_objects) // 4)
    selected = active_rows[: max(0, int(max_objects) - inactive_limit)]
    selected.extend(inactive_rows[:inactive_limit])
    if len(selected) < int(max_objects):
        chosen = {int(row["object_id"]) for row in selected}
        selected.extend(
            row for row in ordered if int(row["object_id"]) not in chosen
        )
    return selected[: int(max_objects)]


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported scene state: {path}")
    return state


def load_requested_object_ids(path: Path | None) -> set[int] | None:
    """Load an exact canonical object allowlist, rejecting ambiguous input."""

    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    requested: set[int] = set()
    for line_number, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            object_id = int(value)
        except ValueError as exc:
            raise ValueError(
                f"invalid object ID at {resolved}:{line_number}: {value!r}"
            ) from exc
        if object_id < 0:
            raise ValueError(f"object IDs must be non-negative: {object_id}")
        requested.add(object_id)
    if not requested:
        raise ValueError(f"object ID allowlist is empty: {resolved}")
    return requested


def load_exclusion_frame_artifacts(
    frames_json: Path,
    additional_frames_json: Iterable[Path] = (),
    *,
    allow_stem_timestamp_fallback: bool = False,
) -> dict[str, object]:
    """Load one exclusion universe under one exact scale/identity contract.

    Repeated images and physical timestamps are intentionally allowed here:
    these artifacts are a union of already-consumed views, not a new selection.
    Every malformed or incompatible input fails before COLMAP planning begins.
    """

    requested_paths = [frames_json, *additional_frames_json]
    artifacts: list[dict[str, object]] = []
    union_names: set[str] = set()
    union_timestamps: set[str] = set()
    total_frame_rows = 0
    reference_scale: float | None = None
    reference_regex: str | None = None
    identity_pattern: re.Pattern[str] | None = None
    require_identity_match = not bool(allow_stem_timestamp_fallback)

    for input_index, requested_path in enumerate(requested_paths):
        path = requested_path.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"exclusion frames artifact must be an object: {path}")
        frame_rows = payload.get("frames")
        if not isinstance(frame_rows, list) or not frame_rows:
            raise ValueError(f"no frames list in exclusion artifact {path}")
        try:
            scale = float(payload.get("meters_per_scene_unit") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"exclusion artifact has invalid meters_per_scene_unit: {path}"
            ) from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"exclusion artifact lacks positive meters_per_scene_unit: {path}"
            )
        identity = payload.get("identity_contract")
        regex_text = str(
            identity.get("regex") if isinstance(identity, dict) else ""
        )
        if input_index == 0:
            reference_scale = scale
            reference_regex = regex_text
            identity_pattern = re.compile(regex_text) if regex_text else None
            if require_identity_match and identity_pattern is None:
                raise ValueError(
                    "frames JSON lacks identity_contract.regex; physical train/heldout "
                    "independence cannot be guaranteed"
                )
            if identity_pattern is not None and "timestamp" not in identity_pattern.groupindex:
                raise ValueError(
                    "identity_contract.regex requires a named timestamp group"
                )
        else:
            assert reference_scale is not None and reference_regex is not None
            if not math.isclose(
                scale,
                reference_scale,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "incompatible meters_per_scene_unit across exclusion artifacts: "
                    f"{path} has {scale!r}, expected {reference_scale!r}"
                )
            if regex_text != reference_regex:
                raise ValueError(
                    "incompatible identity_contract.regex across exclusion artifacts: "
                    f"{path}"
                )

        input_names: set[str] = set()
        input_timestamps: set[str] = set()
        for row_index, row in enumerate(frame_rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"invalid frame row {row_index} in exclusion artifact {path}"
                )
            name = str(row.get("source_image") or "").strip()
            if not name:
                raise ValueError(
                    f"frame row {row_index} has no source_image in exclusion artifact {path}"
                )
            timestamp = identity_key(
                name,
                identity_pattern,
                require_match=require_identity_match,
            )[0]
            input_names.add(name)
            input_timestamps.add(timestamp)
        total_frame_rows += len(frame_rows)
        union_names.update(input_names)
        union_timestamps.update(input_timestamps)
        artifacts.append(
            {
                "role": "primary" if input_index == 0 else "additional",
                "input_index": input_index,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
                "meters_per_scene_unit": scale,
                "identity_contract_regex": regex_text,
                "frame_rows": len(frame_rows),
                "unique_names": len(input_names),
                "unique_physical_timestamps": len(input_timestamps),
            }
        )

    assert reference_scale is not None and reference_regex is not None
    return {
        "primary_path": Path(str(artifacts[0]["path"])),
        "meters_per_scene_unit": reference_scale,
        "identity_contract_regex": reference_regex,
        "identity_pattern": identity_pattern,
        "require_identity_match": require_identity_match,
        "existing_names": union_names,
        "existing_timestamps": union_timestamps,
        "provenance": {
            "artifacts": artifacts,
            "union": {
                "artifact_count": len(artifacts),
                "frame_rows": total_frame_rows,
                "unique_names": len(union_names),
                "unique_physical_timestamps": len(union_timestamps),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument(
        "--additional-exclusion-frames-json",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional frames.json artifact whose image names and physical "
            "timestamps are excluded from selection; repeatable."
        ),
    )
    parser.add_argument(
        "--object-id-file",
        type=Path,
        help=(
            "Exact newline-delimited canonical object IDs to plan. When supplied, "
            "the adaptive priority gate is bypassed but all geometric/image gates remain."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-names-output", type=Path)
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--views-per-object", type=int)
    parser.add_argument("--heldout-views-per-object", type=int, default=2)
    parser.add_argument("--max-total-views", type=int, default=96)
    parser.add_argument(
        "--max-heldout-total-views",
        type=int,
        help="Reserved global heldout image budget; defaults to one third of total.",
    )
    parser.add_argument("--minimum-train-views-per-object", type=int)
    parser.add_argument("--minimum-heldout-views-per-object", type=int, default=2)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", default="farm-full-colmap-v1")
    parser.add_argument(
        "--heldout-reference-only",
        action="store_true",
        help=(
            "Select only the deterministic heldout partition for frozen reference "
            "evaluation. Requires --object-id-file and forbids train fitting/merge."
        ),
    )
    parser.add_argument(
        "--post-adaptation-unseen-heldout",
        action="store_true",
        help=(
            "Strict frozen QC mode: requires heldout-reference-only, OBB retrieval, "
            "three unseen timestamps, hashed images, and explicit prior-frame exclusions."
        ),
    )
    parser.add_argument(
        "--frozen-concept-prompt-manifest",
        type=Path,
        help=(
            "Bind a post-adaptation unseen plan to the exact SHA-256 of an "
            "already-frozen concept prompt manifest. Prompt contents never "
            "participate in camera/view ranking."
        ),
    )
    parser.add_argument("--sharpness-candidates", type=int, default=8)
    parser.add_argument("--min-view-angle-degrees", type=float, default=12.0)
    parser.add_argument("--min-camera-baseline-m", type=float, default=0.15)
    parser.add_argument("--min-baseline-distance-ratio", type=float, default=0.04)
    parser.add_argument(
        "--relaxed-view-angle-degrees",
        type=float,
        default=6.0,
        help="Minimum-only fallback; never used to add redundancy views.",
    )
    parser.add_argument(
        "--relaxed-camera-baseline-m",
        type=float,
        default=0.08,
        help="Minimum-only fallback; rendered depth remains a later hard gate.",
    )
    parser.add_argument(
        "--relaxed-baseline-distance-ratio",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--minimum-temporal-episode-frames",
        type=int,
        default=1,
        help=(
            "When greater than one, reserve this many pose-continuous train frames "
            "in one camera-by-family stream; heldout is never expanded."
        ),
    )
    parser.add_argument(
        "--maximum-temporal-neighbor-frame-gap", type=int, default=3
    )
    parser.add_argument(
        "--maximum-temporal-translation-m", type=float, default=0.5
    )
    parser.add_argument(
        "--maximum-temporal-rotation-degrees", type=float, default=21.0
    )
    parser.add_argument("--min-margin-ratio", type=float, default=0.035)
    parser.add_argument("--min-area-ratio", type=float, default=0.0025)
    parser.add_argument("--max-area-ratio", type=float, default=0.55)
    parser.add_argument("--min-priority", type=float, default=1.0)
    parser.add_argument(
        "--retrieval-mode",
        choices=("support_tracks", "support_tracks_then_obb", "obb"),
        default="support_tracks",
        help=(
            "Camera retrieval authority. The default never consults the current OBB; "
            "OBB fallback must be requested explicitly."
        ),
    )
    parser.add_argument("--minimum-object-support-points", type=int, default=24)
    parser.add_argument("--maximum-object-support-points", type=int, default=4096)
    parser.add_argument("--support-trim-quantile", type=float, default=0.995)
    parser.add_argument("--maximum-track-distance-m", type=float, default=0.08)
    parser.add_argument(
        "--minimum-object-track-match-fraction",
        type=float,
        default=0.30,
        help=(
            "Fail closed before camera ranking when too little of the metric "
            "object support can be linked to full-COLMAP tracks."
        ),
    )
    parser.add_argument("--minimum-track-support-points", type=int, default=2)
    parser.add_argument("--minimum-track-support-fraction", type=float, default=0.01)
    parser.add_argument(
        "--minimum-rescue-track-support-points", type=int, default=1
    )
    parser.add_argument("--minimum-projected-support-points", type=int, default=8)
    parser.add_argument(
        "--minimum-rescue-projected-support-points", type=int, default=24
    )
    parser.add_argument(
        "--minimum-rescue-in-frame-support-ratio",
        type=float,
        default=0.95,
    )
    parser.add_argument("--minimum-in-frame-support-ratio", type=float, default=0.85)
    parser.add_argument("--support-bbox-quantile", type=float, default=0.01)
    parser.add_argument(
        "--allow-stem-timestamp-fallback",
        action="store_true",
        help=("Allow image stems as timestamp IDs when no identity regex matches; "
              "unsafe for datasets with virtual views of one physical capture."),
    )
    parser.add_argument(
        "--hash-selected-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record SHA-256 for every selected source image in plan provenance.",
    )
    args = parser.parse_args()
    frozen_prompt_selection_contract: dict[str, object] | None = None
    if args.frozen_concept_prompt_manifest is not None:
        if args.heldout_reference_only and not args.post_adaptation_unseen_heldout:
            parser.error(
                "frozen prompt heldout planning requires "
                "--post-adaptation-unseen-heldout"
            )
        frozen_prompt_path = (
            args.frozen_concept_prompt_manifest.expanduser().resolve(strict=True)
        )
        frozen_prompt_payload = json.loads(
            frozen_prompt_path.read_text(encoding="utf-8")
        )
        if (
            not isinstance(frozen_prompt_payload, dict)
            or frozen_prompt_payload.get("schema")
            != "farm.sam3-concept-prompts.v1"
        ):
            parser.error(
                "frozen concept prompt manifest must use schema "
                "farm.sam3-concept-prompts.v1"
            )
        frozen_prompt_selection_contract = {
            "schema": "farm.sam3-concept-prompt-selection-binding.v1",
            "selection_role": (
                "post_adaptation_unseen_heldout"
                if args.post_adaptation_unseen_heldout
                else "train_fit_candidate"
            ),
            "fit_candidate_authorized": not bool(
                args.post_adaptation_unseen_heldout
            ),
            "release_authorized": False,
            "prompt_manifest_path": str(frozen_prompt_path),
            "prompt_manifest_sha256": sha256_file(frozen_prompt_path),
            "selection_frozen_after_prompt_manifest": True,
            "category_agnostic_view_ranking": True,
            "target_conditioned_view_selection": False,
            "prompt_used_for_view_ranking": False,
        }
    if args.post_adaptation_unseen_heldout:
        if not args.heldout_reference_only:
            parser.error(
                "--post-adaptation-unseen-heldout requires --heldout-reference-only"
            )
        if args.retrieval_mode != "obb":
            parser.error(
                "--post-adaptation-unseen-heldout requires --retrieval-mode=obb"
            )
        if args.minimum_heldout_views_per_object < 3:
            parser.error(
                "post-adaptation unseen heldout requires at least 3 timestamps"
            )
        if not args.hash_selected_images:
            parser.error(
                "post-adaptation unseen heldout requires hashed selected images"
            )
        if not args.additional_exclusion_frames_json:
            parser.error(
                "post-adaptation unseen heldout requires prior-frame exclusions"
            )
        if args.minimum_temporal_episode_frames != 1:
            parser.error(
                "post-adaptation unseen heldout forbids temporal train expansion"
            )
    if args.heldout_reference_only:
        if args.object_id_file is None:
            parser.error("--heldout-reference-only requires --object-id-file")
        if args.views_per_object not in (None, 0):
            parser.error(
                "--heldout-reference-only forbids a positive --views-per-object"
            )
        if args.minimum_train_views_per_object not in (None, 0):
            parser.error(
                "--heldout-reference-only forbids a positive "
                "--minimum-train-views-per-object"
            )
        args.views_per_object = 0
        args.minimum_train_views_per_object = 0
    else:
        args.views_per_object = (
            5 if args.views_per_object is None else args.views_per_object
        )
        args.minimum_train_views_per_object = (
            3
            if args.minimum_train_views_per_object is None
            else args.minimum_train_views_per_object
        )
    if min(
        args.max_objects,
        args.heldout_views_per_object,
        args.max_total_views,
        args.minimum_heldout_views_per_object,
        args.minimum_object_support_points,
        args.maximum_object_support_points,
        args.minimum_track_support_points,
        args.minimum_rescue_track_support_points,
        args.minimum_projected_support_points,
        args.minimum_rescue_projected_support_points,
        args.minimum_temporal_episode_frames,
        args.maximum_temporal_neighbor_frame_gap,
    ) < 1:
        parser.error("object/view budgets must be positive")
    if not args.heldout_reference_only and min(
        args.views_per_object, args.minimum_train_views_per_object
    ) < 1:
        parser.error("train-view budgets must be positive")
    if args.minimum_train_views_per_object > args.views_per_object:
        parser.error("minimum train views cannot exceed --views-per-object")
    if (
        not args.heldout_reference_only
        and args.minimum_temporal_episode_frames > 1
        and args.views_per_object
        < args.minimum_train_views_per_object
        + args.minimum_temporal_episode_frames
        - 1
    ):
        parser.error(
            "temporal episode requires views-per-object >= "
            "minimum-train-views + minimum-temporal-episode-frames - 1"
        )
    if args.minimum_heldout_views_per_object > args.heldout_views_per_object:
        parser.error("minimum heldout views cannot exceed heldout budget")
    if not 0.0 < args.heldout_fraction < 1.0:
        parser.error("--heldout-fraction must be in (0, 1)")
    if not 0.5 <= args.support_trim_quantile <= 1.0:
        parser.error("--support-trim-quantile must be in [0.5, 1]")
    if not 0.0 <= args.minimum_object_track_match_fraction <= 1.0:
        parser.error("--minimum-object-track-match-fraction must be in [0, 1]")
    if not 0.0 <= args.minimum_track_support_fraction <= 1.0:
        parser.error("--minimum-track-support-fraction must be in [0, 1]")
    if not 0.0 <= args.minimum_in_frame_support_ratio <= 1.0:
        parser.error("--minimum-in-frame-support-ratio must be in [0, 1]")
    if not 0.0 <= args.minimum_rescue_in_frame_support_ratio <= 1.0:
        parser.error("--minimum-rescue-in-frame-support-ratio must be in [0, 1]")
    if args.minimum_rescue_in_frame_support_ratio < args.minimum_in_frame_support_ratio:
        parser.error("rescue in-frame ratio cannot be weaker than the normal gate")
    if (
        args.minimum_rescue_projected_support_points
        < args.minimum_projected_support_points
    ):
        parser.error("rescue projected support cannot be weaker than normal")
    if (
        args.minimum_rescue_track_support_points
        > args.minimum_track_support_points
    ):
        parser.error("rescue track support cannot exceed the normal track gate")
    if not 0.0 <= args.support_bbox_quantile < 0.25:
        parser.error("--support-bbox-quantile must be in [0, 0.25)")
    if not 0.0 <= args.relaxed_view_angle_degrees <= args.min_view_angle_degrees:
        parser.error("relaxed view angle must be in [0, min-view-angle]")
    if not 0.0 <= args.relaxed_camera_baseline_m <= args.min_camera_baseline_m:
        parser.error("relaxed camera baseline must be in [0, min-camera-baseline]")
    if not (
        0.0
        <= args.relaxed_baseline_distance_ratio
        <= args.min_baseline_distance_ratio
    ):
        parser.error("relaxed baseline ratio must be in [0, strict baseline ratio]")
    if min(args.min_camera_baseline_m, args.min_baseline_distance_ratio) <= 0.0:
        parser.error("strict baseline thresholds must be positive")
    if min(
        args.maximum_temporal_translation_m,
        args.maximum_temporal_rotation_degrees,
    ) <= 0.0:
        parser.error("temporal pose thresholds must be positive")
    max_heldout_total_views = (
        int(args.max_heldout_total_views)
        if args.max_heldout_total_views is not None
        else (
            int(args.max_total_views)
            if args.heldout_reference_only
            else max(
                1,
                min(
                    int(args.max_total_views) - 1,
                    int(args.max_total_views) // 3,
                ),
            )
        )
    )
    if args.heldout_reference_only and not (
        1 <= max_heldout_total_views <= int(args.max_total_views)
    ):
        parser.error("heldout total budget must be in [1, max-total-views]")
    if not args.heldout_reference_only and not (
        1 <= max_heldout_total_views < int(args.max_total_views)
    ):
        parser.error("heldout total budget must be in [1, max-total-views)")

    started = time.perf_counter()
    exclusion_contract = load_exclusion_frame_artifacts(
        args.frames_json,
        args.additional_exclusion_frames_json,
        allow_stem_timestamp_fallback=bool(args.allow_stem_timestamp_fallback),
    )
    frames_path = exclusion_contract["primary_path"]
    scale = float(exclusion_contract["meters_per_scene_unit"])
    identity_pattern = exclusion_contract["identity_pattern"]
    require_identity_match = bool(exclusion_contract["require_identity_match"])
    existing_names = set(exclusion_contract["existing_names"])
    existing_timestamps = set(exclusion_contract["existing_timestamps"])

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("pycolmap is required for full-COLMAP rescue planning") from exc
    colmap_model_path = args.colmap_model.expanduser().resolve(strict=True)
    reconstruction = pycolmap.Reconstruction(str(colmap_model_path))
    camera_stream_timeline, camera_stream_names = build_camera_stream_timeline(
        reconstruction,
        identity_pattern,
        require_match=require_identity_match,
    )
    image_root = args.image_root.expanduser().resolve()
    scene_state_path = args.scene_state.expanduser().resolve(strict=True)
    state = load_state(scene_state_path)
    object_ids = _numpy(state.get("object_id"), np.int64).reshape(-1)
    object_count = int(object_ids.size)
    active = _numpy(state.get("active", np.ones((object_count,), dtype=bool)), bool).reshape(-1)

    def optional_matrix(value: object, columns: int) -> np.ndarray:
        if value is None:
            return np.full((object_count, columns), np.nan, dtype=np.float64)
        array = _numpy(value, np.float64)
        if array.shape != (object_count, columns):
            return np.full((object_count, columns), np.nan, dtype=np.float64)
        return array

    centers = optional_matrix(
        state.get("object_box_centers_m", state.get("means")), 3
    )
    dimensions = optional_matrix(state.get("object_box_dimensions_m"), 3)
    quaternions = optional_matrix(state.get("object_box_wxyz"), 4)
    observations = state.get("object_mask_observations") or []
    sparse_index = (
        build_sparse_track_index(reconstruction, meters_per_scene_unit=scale)
        if args.retrieval_mode != "obb"
        else None
    )
    requested_ids = load_requested_object_ids(args.object_id_file)
    known_ids = {int(value) for value in object_ids.tolist()}
    if requested_ids is not None:
        missing_ids = sorted(requested_ids - known_ids)
        if missing_ids:
            raise ValueError(f"requested object IDs are absent from scene state: {missing_ids}")

    object_queue: list[dict[str, object]] = []
    preplan_rejections: list[dict[str, object]] = []
    for index, object_id_raw in enumerate(object_ids.tolist()):
        object_id = int(object_id_raw)
        if requested_ids is not None and object_id not in requested_ids:
            continue
        observation_rows = observations[index] if index < len(observations) else []
        unique_observations = len(
            {
                int(row.get("image_id", -1))
                for row in observation_rows
                if isinstance(row, dict) and int(row.get("image_id", -1)) >= 0
            }
        )
        priority, reasons = object_rescue_priority(state, index, unique_observations)
        if requested_ids is None and priority < float(args.min_priority):
            continue
        if (
            requested_ids is None
            and index < active.size
            and not bool(active[index])
            and "geometry_rejected" not in reasons
        ):
            continue
        support_points = np.zeros((0, 3), dtype=np.float64)
        support_diagnostics: dict[str, object] = {
            "source": "object_voxels",
            "status": "not_requested",
            "bounded_points": 0,
        }
        if args.retrieval_mode != "obb":
            support_points, support_diagnostics = object_support_points(
                state,
                index,
                trim_quantile=float(args.support_trim_quantile),
                max_points=int(args.maximum_object_support_points),
            )
        support_available = (
            support_points.shape[0] >= int(args.minimum_object_support_points)
        )
        corners: np.ndarray | None = None
        if args.retrieval_mode != "support_tracks":
            try:
                corners = obb_corners(
                    centers[index], dimensions[index], quaternions[index]
                )
            except (ValueError, IndexError):
                corners = None
        obb_available = corners is not None
        if not support_available and not obb_available:
            preplan_rejections.append(
                {
                    "object_id": object_id,
                    "state_index": index,
                    "status": "insufficient_retrieval_geometry",
                    "retrieval_mode": str(args.retrieval_mode),
                    "support": support_diagnostics,
                    "obb_fallback_available": False,
                }
            )
            continue
        object_queue.append(
            {
                "object_id": object_id,
                "state_index": index,
                "active": bool(index < active.size and active[index]),
                "priority": priority,
                "priority_reasons": reasons,
                "observations": unique_observations,
                "support": support_diagnostics,
                "retrieval_mode": str(args.retrieval_mode),
                "_support_points_m": support_points if support_available else None,
                "_obb_corners_m": corners,
            }
        )
    if requested_ids is None:
        object_queue = stratified_object_queue(object_queue, int(args.max_objects))
    else:
        object_queue = sorted(object_queue, key=lambda row: int(row["object_id"]))
        if len(object_queue) > int(args.max_objects):
            raise ValueError(
                f"exact object allowlist has {len(object_queue)} plannable objects, "
                f"exceeding --max-objects={args.max_objects}"
            )

    planned_objects: list[dict[str, object]] = []
    for object_row in object_queue:
        support_raw = object_row.pop("_support_points_m", None)
        support_points = (
            np.asarray(support_raw, dtype=np.float64).reshape(-1, 3)
            if support_raw is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        corners_raw = object_row.pop("_obb_corners_m", None)
        corners = (
            np.asarray(corners_raw, dtype=np.float64).reshape(8, 3)
            if corners_raw is not None
            else None
        )
        candidates: list[dict[str, object]] = []
        retrieval_diagnostics: dict[str, object] = {}
        if support_points.size and sparse_index is not None:
            candidates, retrieval_diagnostics = build_support_track_candidates(
                reconstruction,
                sparse_index,
                support_points,
                identity_pattern=identity_pattern,
                require_identity_match=require_identity_match,
                existing_names=existing_names,
                existing_timestamps=existing_timestamps,
                meters_per_scene_unit=scale,
                maximum_track_distance_m=float(args.maximum_track_distance_m),
                minimum_object_track_match_fraction=float(
                    args.minimum_object_track_match_fraction
                ),
                minimum_track_support_points=int(args.minimum_track_support_points),
                minimum_track_support_fraction=float(args.minimum_track_support_fraction),
                minimum_rescue_track_support_points=int(
                    args.minimum_rescue_track_support_points
                ),
                minimum_projected_support_points=int(args.minimum_projected_support_points),
                minimum_rescue_projected_support_points=int(
                    args.minimum_rescue_projected_support_points
                ),
                minimum_rescue_in_frame_support_ratio=float(
                    args.minimum_rescue_in_frame_support_ratio
                ),
                min_margin_ratio=float(args.min_margin_ratio),
                min_area_ratio=float(args.min_area_ratio),
                max_area_ratio=float(args.max_area_ratio),
                min_in_frame_support_ratio=float(args.minimum_in_frame_support_ratio),
                support_bbox_quantile=float(args.support_bbox_quantile),
            )
        unique_support_timestamps = {
            str(row["physical_timestamp"]) for row in candidates
        }
        fallback_needed = (
            args.retrieval_mode == "obb"
            or (
                args.retrieval_mode == "support_tracks_then_obb"
                and len(unique_support_timestamps)
                < int(args.minimum_train_views_per_object)
                + int(args.minimum_heldout_views_per_object)
            )
        )
        if fallback_needed and corners is not None:
            by_name = {str(row["name"]): row for row in candidates}
            for image in reconstruction.images.values():
                name = str(image.name)
                timestamp, camera_name, family = identity_key(
                    name,
                    identity_pattern,
                    require_match=require_identity_match,
                )
                if name in existing_names or timestamp in existing_timestamps:
                    continue
                camera = reconstruction.cameras[image.camera_id]
                projected = project_obb(
                    corners,
                    image,
                    camera,
                    meters_per_scene_unit=scale,
                    min_margin_ratio=float(args.min_margin_ratio),
                    min_area_ratio=float(args.min_area_ratio),
                    max_area_ratio=float(args.max_area_ratio),
                )
                if projected is None:
                    continue
                projected["geometric_score"] = float(projected["geometric_score"]) * (
                    1.08 if family == "center" else 1.0
                )
                by_name.setdefault(
                    name,
                    {
                        "name": name,
                        "colmap_image_id": int(image.image_id),
                        "colmap_camera_id": int(image.camera_id),
                        "physical_timestamp": timestamp,
                        "camera": camera_name,
                        "family": family,
                        "retrieval_source": "explicit_obb_projection_fallback",
                        "bbox_source": "projected_current_obb",
                        **projected,
                    },
                )
            candidates = list(by_name.values())
            retrieval_diagnostics = {
                **retrieval_diagnostics,
                "obb_fallback_used": True,
                "obb_fallback_reason": (
                    "requested_obb_mode"
                    if args.retrieval_mode == "obb"
                    else "insufficient_support_track_timestamps"
                ),
            }
        else:
            retrieval_diagnostics = {
                **retrieval_diagnostics,
                "obb_fallback_used": False,
            }
        for candidate in candidates:
            candidate["camera_stream_id"] = (
                f"{candidate.get('camera', '')}_{candidate.get('family', '')}"
            )
            candidate["camera_stream_frame_index"] = camera_stream_timeline.get(
                str(candidate["name"])
            )
        candidates.sort(
            key=lambda row: float(row["geometric_score"]), reverse=True
        )
        for candidate in candidates[: int(args.sharpness_candidates)]:
            path = safe_image_path(image_root, str(candidate["name"]))
            sharpness = crop_sharpness(path, candidate["bbox_xyxy"])
            candidate["sharpness"] = sharpness
            candidate["score"] = float(candidate["geometric_score"]) * (
                1.0 + 0.08 * math.log1p(max(0.0, sharpness))
            )
        for candidate in candidates[int(args.sharpness_candidates) :]:
            candidate["sharpness"] = None
            candidate["score"] = float(candidate["geometric_score"])
        if args.heldout_reference_only:
            train_candidates = []
            heldout_result = choose_heldout_reference_views(
                candidates,
                heldout_limit=int(args.heldout_views_per_object),
                min_view_angle_degrees=float(args.min_view_angle_degrees),
                heldout_fraction=float(args.heldout_fraction),
                split_seed=str(args.split_seed),
                minimum_views=int(args.minimum_heldout_views_per_object),
                min_camera_baseline_m=float(args.min_camera_baseline_m),
                min_baseline_distance_ratio=float(args.min_baseline_distance_ratio),
                relaxed_view_angle_degrees=float(args.relaxed_view_angle_degrees),
                relaxed_camera_baseline_m=float(args.relaxed_camera_baseline_m),
                relaxed_baseline_distance_ratio=float(
                    args.relaxed_baseline_distance_ratio
                ),
                priority=float(object_row["priority"]),
                return_diagnostics=True,
            )
            heldout_candidates, heldout_selection = heldout_result
            train_selection = {
                "fold": "train",
                "selected_views": 0,
                "state_fit_authorized": False,
                "temporal_episode": {
                    "enabled": False,
                    "reason": "heldout_reference_only",
                    "episode_frames": 0,
                },
            }
        else:
            (
                train_candidates,
                heldout_candidates,
                train_selection,
                heldout_selection,
            ) = choose_train_heldout_next_best_views(
                candidates,
                train_limit=int(args.views_per_object),
                heldout_limit=int(args.heldout_views_per_object),
                minimum_train_views=int(args.minimum_train_views_per_object),
                minimum_heldout_views=int(args.minimum_heldout_views_per_object),
                min_view_angle_degrees=float(args.min_view_angle_degrees),
                min_camera_baseline_m=float(args.min_camera_baseline_m),
                min_baseline_distance_ratio=float(args.min_baseline_distance_ratio),
                relaxed_view_angle_degrees=float(args.relaxed_view_angle_degrees),
                relaxed_camera_baseline_m=float(args.relaxed_camera_baseline_m),
                relaxed_baseline_distance_ratio=float(
                    args.relaxed_baseline_distance_ratio
                ),
                heldout_fraction=float(args.heldout_fraction),
                split_seed=str(args.split_seed),
                priority=float(object_row["priority"]),
            )
            train_pool: list[dict[str, object]] = []
            for source in candidates:
                if physical_timestamp_fold(
                    str(source.get("physical_timestamp") or ""),
                    heldout_fraction=float(args.heldout_fraction),
                    split_seed=str(args.split_seed),
                ) != "train":
                    continue
                row = dict(source)
                row["split"] = "train"
                train_pool.append(row)
            projected_temporal_context = (
                build_projected_temporal_context(
                    reconstruction,
                    support_points,
                    train_candidates[
                        : int(args.minimum_train_views_per_object)
                    ],
                    stream_names=camera_stream_names,
                    stream_timeline=camera_stream_timeline,
                    identity_pattern=identity_pattern,
                    require_identity_match=require_identity_match,
                    existing_names=existing_names,
                    existing_timestamps=existing_timestamps,
                    heldout_fraction=float(args.heldout_fraction),
                    split_seed=str(args.split_seed),
                    meters_per_scene_unit=scale,
                    maximum_neighbor_frame_gap=int(
                        args.maximum_temporal_neighbor_frame_gap
                    ),
                    minimum_projected_support_points=int(
                        args.minimum_rescue_projected_support_points
                    ),
                    minimum_in_frame_support_ratio=float(
                        args.minimum_rescue_in_frame_support_ratio
                    ),
                    min_margin_ratio=float(args.min_margin_ratio),
                    min_area_ratio=float(args.min_area_ratio),
                    max_area_ratio=float(args.max_area_ratio),
                    support_bbox_quantile=float(args.support_bbox_quantile),
                )
                if int(args.minimum_temporal_episode_frames) > 1
                and support_points.size
                else []
            )
            train_pool_by_name = {
                str(row["name"]): row for row in train_pool
            }
            for row in projected_temporal_context:
                train_pool_by_name.setdefault(str(row["name"]), row)
            train_candidates, temporal_selection = plan_temporal_episode_topup(
                list(train_pool_by_name.values()),
                train_candidates,
                minimum_diverse_views=int(args.minimum_train_views_per_object),
                maximum_views=int(args.views_per_object),
                minimum_episode_frames=int(args.minimum_temporal_episode_frames),
                maximum_neighbor_frame_gap=int(
                    args.maximum_temporal_neighbor_frame_gap
                ),
                maximum_translation_m=float(args.maximum_temporal_translation_m),
                maximum_rotation_degrees=float(
                    args.maximum_temporal_rotation_degrees
                ),
            )
            temporal_selection["projected_context_candidates"] = len(
                projected_temporal_context
            )
            temporal_selection["projection_only_context_is_seed_eligible"] = False
            train_selection["temporal_episode"] = temporal_selection
        planned_objects.append(
            {
                **object_row,
                "candidate_count": len(candidates),
                "train_candidate_count": sum(
                    physical_timestamp_fold(
                        str(row["physical_timestamp"]),
                        heldout_fraction=float(args.heldout_fraction),
                        split_seed=str(args.split_seed),
                    ) == "train"
                    for row in candidates
                ),
                "heldout_candidate_count": sum(
                    physical_timestamp_fold(
                        str(row["physical_timestamp"]),
                        heldout_fraction=float(args.heldout_fraction),
                        split_seed=str(args.split_seed),
                    ) == "heldout"
                    for row in candidates
                ),
                "retrieval_diagnostics": retrieval_diagnostics,
                "next_best_view_selection": {
                    "train": train_selection,
                    "heldout": heldout_selection,
                },
                "train_candidates": train_candidates,
                "heldout_candidates": heldout_candidates,
            }
        )

    if args.heldout_reference_only:
        _, heldout_admitted = allocate_global_view_budget_core(
            [list(row.get("heldout_candidates") or []) for row in planned_objects],
            max_total_views=max_heldout_total_views,
        )
        train_admitted = [[] for _ in planned_objects]
    else:
        _, train_admitted, heldout_admitted = allocate_split_global_view_budget(
            planned_objects,
            max_total_views=int(args.max_total_views),
            max_heldout_views=max_heldout_total_views,
        )
    output_objects: list[dict[str, object]] = []
    for object_row, train_views, heldout_views in zip(
        planned_objects, train_admitted, heldout_admitted, strict=True
    ):
        output_objects.append(
            finalize_planned_object_views(
                object_row,
                train_views,
                heldout_views,
                heldout_reference_only=bool(args.heldout_reference_only),
                minimum_train_views=int(args.minimum_train_views_per_object),
                minimum_heldout_views=int(args.minimum_heldout_views_per_object),
                minimum_temporal_episode_frames=int(
                    args.minimum_temporal_episode_frames
                ),
            )
        )

    selected_global = {
        str(view["name"]): view
        for row in output_objects
        for field in ("train_views", "heldout_views")
        for view in row[field]
    }
    selected_names = sorted(
        selected_global,
        key=lambda name: (
            str(selected_global[name].get("physical_timestamp") or ""),
            str(selected_global[name].get("camera") or ""),
            str(selected_global[name].get("family") or ""),
            name,
        ),
    )
    global_train_views = [
        view for row in output_objects for view in row["train_views"]
    ]
    global_heldout_views = [
        view for row in output_objects for view in row["heldout_views"]
    ]
    validate_disjoint_timestamp_folds(global_train_views, global_heldout_views)
    train_names = sorted({str(row["name"]) for row in global_train_views})
    heldout_names = sorted({str(row["name"]) for row in global_heldout_views})
    selected_image_provenance: list[dict[str, object]] = []
    for name in selected_names:
        source_path = safe_image_path(image_root, name)
        row: dict[str, object] = {
            "name": name,
            "bytes": int(source_path.stat().st_size),
            "split": str(selected_global[name].get("split") or ""),
        }
        if args.hash_selected_images:
            row["sha256"] = sha256_file(source_path)
        selected_image_provenance.append(row)
    report = {
        "schema": "farm.full-colmap-rescue-plan.v1",
        "planner_revision": "support-tracks-next-best-view-temporal.v4",
        "created_unix_s": time.time(),
        "source_scene_state": str(scene_state_path),
        "colmap_model": str(colmap_model_path),
        "image_root": str(image_root),
        "frames_json": str(frames_path),
        "additional_exclusion_frames_json": [
            str(path.expanduser().resolve())
            for path in args.additional_exclusion_frames_json
        ],
        "concept_prompt_selection_contract": frozen_prompt_selection_contract,
        "meters_per_scene_unit": scale,
        "policy": {
            "adaptive_only": requested_ids is None,
            "heldout_reference_only": bool(args.heldout_reference_only),
            "post_adaptation_unseen_heldout": bool(
                args.post_adaptation_unseen_heldout
            ),
            "selection_frozen_before_sam3_targets": bool(
                args.post_adaptation_unseen_heldout
            ),
            "target_conditioned_view_selection": False,
            "unseen_selection_basis": (
                "frozen_obb_colmap_camera_geometry"
                if args.post_adaptation_unseen_heldout
                else None
            ),
            "state_fit_authorized": not bool(args.heldout_reference_only),
            "geometry_merge_authorized": not bool(args.heldout_reference_only),
            "semantic_mutation_authorized": not bool(args.heldout_reference_only),
            "category_agnostic_view_ranking": True,
            "exclude_existing_physical_timestamps": True,
            "exclusion_artifact_duplicates_are_exclusions_only": True,
            "whole_object_frame_required": True,
            "default_retrieval_uses_current_obb": False,
            "retrieval_mode": str(args.retrieval_mode),
            "max_objects": int(args.max_objects),
            "views_per_object": int(args.views_per_object),
            "heldout_views_per_object": int(args.heldout_views_per_object),
            "minimum_train_views_per_object": int(args.minimum_train_views_per_object),
            "minimum_heldout_views_per_object": int(args.minimum_heldout_views_per_object),
            "max_total_views": int(args.max_total_views),
            "max_heldout_total_views": max_heldout_total_views,
            "heldout_fraction": float(args.heldout_fraction),
            "split_seed": str(args.split_seed),
            "split_unit": "physical_timestamp",
            "identity_stem_fallback_enabled": bool(args.allow_stem_timestamp_fallback),
            "sharpness_candidates": int(args.sharpness_candidates),
            "min_view_angle_degrees": float(args.min_view_angle_degrees),
            "min_camera_baseline_m": float(args.min_camera_baseline_m),
            "min_baseline_distance_ratio": float(args.min_baseline_distance_ratio),
            "relaxed_view_angle_degrees": float(
                args.relaxed_view_angle_degrees
            ),
            "relaxed_camera_baseline_m": float(
                args.relaxed_camera_baseline_m
            ),
            "relaxed_baseline_distance_ratio": float(
                args.relaxed_baseline_distance_ratio
            ),
            "adaptive_view_budget": True,
            "maximum_redundancy_views_per_fold_object": 2,
            "temporal_episode_train_only": True,
            "minimum_temporal_episode_frames": int(
                args.minimum_temporal_episode_frames
            ),
            "maximum_temporal_neighbor_frame_gap": int(
                args.maximum_temporal_neighbor_frame_gap
            ),
            "maximum_temporal_translation_m": float(
                args.maximum_temporal_translation_m
            ),
            "maximum_temporal_rotation_degrees": float(
                args.maximum_temporal_rotation_degrees
            ),
            "temporal_episode_preserves_diverse_minimum": True,
            "heldout_temporal_expansion_authorized": False,
            "sparse_track_support_is_occlusion_proxy": True,
            "rendered_depth_is_post_selection_hard_gate": True,
            "min_margin_ratio": float(args.min_margin_ratio),
            "min_area_ratio": float(args.min_area_ratio),
            "max_area_ratio": float(args.max_area_ratio),
            "minimum_object_support_points": int(args.minimum_object_support_points),
            "maximum_object_support_points": int(args.maximum_object_support_points),
            "support_trim_quantile": float(args.support_trim_quantile),
            "maximum_track_distance_m": float(args.maximum_track_distance_m),
            "minimum_object_track_match_fraction": float(
                args.minimum_object_track_match_fraction
            ),
            "minimum_track_support_points": int(args.minimum_track_support_points),
            "minimum_track_support_fraction": float(args.minimum_track_support_fraction),
            "minimum_rescue_track_support_points": int(
                args.minimum_rescue_track_support_points
            ),
            "minimum_projected_support_points": int(args.minimum_projected_support_points),
            "minimum_rescue_projected_support_points": int(
                args.minimum_rescue_projected_support_points
            ),
            "minimum_rescue_in_frame_support_ratio": float(
                args.minimum_rescue_in_frame_support_ratio
            ),
            "sparse_track_rescue_score_multiplier": 0.72,
            "minimum_in_frame_support_ratio": float(args.minimum_in_frame_support_ratio),
            "support_bbox_quantile": float(args.support_bbox_quantile),
            "min_priority": float(args.min_priority),
            "selection_mode": (
                "exact_object_allowlist"
                if requested_ids is not None
                else "adaptive_priority"
            ),
            "requested_object_ids": sorted(requested_ids) if requested_ids is not None else None,
        },
        "provenance": {
            "source_scene_state": {
                "path": str(scene_state_path),
                "sha256": sha256_file(scene_state_path),
            },
            "frames_json": {
                "path": str(frames_path),
                "sha256": sha256_file(frames_path),
                "bytes": int(frames_path.stat().st_size),
            },
            "exclusion_frames": exclusion_contract["provenance"],
            "colmap_model_files": model_file_manifest(colmap_model_path),
            "planner_sources": [
                {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                {
                    "path": str(ROOT / "src/farm_runtime/full_colmap_view_planner.py"),
                    "sha256": sha256_file(
                        ROOT / "src/farm_runtime/full_colmap_view_planner.py"
                    ),
                },
            ],
            "selected_source_images": selected_image_provenance,
            "selected_image_hashes_enabled": bool(args.hash_selected_images),
        },
        "registered_images": len(reconstruction.images),
        "sparse_track_index": {
            "enabled": sparse_index is not None,
            "sparse_points": (
                int(sparse_index.point_ids.size) if sparse_index is not None else 0
            ),
            "track_observation_edges": (
                sum(len(rows) for rows in sparse_index.image_ids_by_point)
                if sparse_index is not None
                else 0
            ),
        },
        "existing_selected_images": len(existing_names),
        "existing_physical_timestamps": len(existing_timestamps),
        "objects_routed": len(object_queue) + len(preplan_rejections),
        "objects_considered": len(object_queue),
        "objects_with_rescue_views": sum(
            int(row["render_view_count"]) > 0 for row in output_objects
        ),
        "objects_with_heldout_reference_views": sum(
            bool(args.heldout_reference_only)
            and int(row["heldout_count"]) > 0
            for row in output_objects
        ),
        "preplan_rejections": preplan_rejections,
        "objects_rejected_after_view_selection": sum(
            row["planning_status"]
            not in {"planned", "planned_heldout_reference_only"}
            for row in output_objects
        ),
        "unique_rescue_views": len(selected_names),
        "unique_train_views": len(train_names),
        "unique_heldout_views": len(heldout_names),
        "train_names": train_names,
        "heldout_names": heldout_names,
        "selected_names": selected_names,
        "selected_names_role": (
            "heldout_reference_only_non_merge"
            if args.heldout_reference_only
            else "rgbd_render_union_of_train_and_heldout"
        ),
        "consumption_contract": {
            "train": (
                "train_unconsumed"
                if args.heldout_reference_only
                else "mapping_covisibility_sam3_merge_obb_fit"
            ),
            "heldout": (
                "heldout_reference_only"
                if args.heldout_reference_only
                else "frozen_evaluation_only"
            ),
            "state_fit_authorized": not bool(args.heldout_reference_only),
            "geometry_merge_authorized": not bool(args.heldout_reference_only),
            "semantic_mutation_authorized": not bool(args.heldout_reference_only),
            "selection_frozen_before_sam3_targets": bool(
                args.post_adaptation_unseen_heldout
            ),
            "post_adaptation_state_mutation_authorized": not bool(
                args.post_adaptation_unseen_heldout
            ),
        },
        "objects": output_objects,
        "timing": {
            "duration_seconds": time.perf_counter() - started,
            "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "gpu_used": False,
        },
    }
    output = args.output.expanduser().resolve()
    selected_output = (
        args.selected_names_output.expanduser().resolve()
        if args.selected_names_output
        else output.with_name("selected_names.txt")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selected_output.write_text(
        "".join(f"{name}\n" for name in selected_names), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "registered_images",
                    "objects_considered",
                    "objects_with_rescue_views",
                    "unique_rescue_views",
                    "timing",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
