#!/usr/bin/env python3
"""Plan a bounded full-COLMAP rescue pass for uncertain scene objects.

The normal pipeline maps a trajectory-connected subset of physical timestamps.
This planner searches the complete registered pinhole reconstruction only for
objects whose multi-view geometry is weak.  It does not run segmentation and
does not mutate the scene state.  The resulting exact image list is consumed by
the later RGB-D/mask rescue step.

Ranking is category-agnostic: projected OBB coverage, full-frame containment,
image sharpness, physical-timestamp uniqueness, and viewpoint diversity.
Semantic fields affect only which objects need rescue; names never affect the
camera score or object association.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


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


def identity_key(name: str, pattern: re.Pattern[str] | None) -> tuple[str, str, str]:
    if pattern is None:
        return Path(name).stem, "", ""
    match = pattern.search(name)
    if match is None:
        return Path(name).stem, "", ""
    groups = match.groupdict()
    timestamp = str(groups.get("timestamp") or Path(name).stem)
    camera = str(groups.get("camera") or "")
    family = str(groups.get("family") or groups.get("view") or "")
    return timestamp, camera, family


def angular_separation_degrees(a: object, b: object) -> float:
    va = np.asarray(a, dtype=np.float64).reshape(3)
    vb = np.asarray(b, dtype=np.float64).reshape(3)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def choose_diverse_views(
    candidates: list[dict[str, object]],
    *,
    limit: int,
    min_view_angle_degrees: float,
) -> list[dict[str, object]]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            float(row.get("score", 0.0)),
            float(row.get("area_ratio", 0.0)),
            float(row.get("margin_ratio", 0.0)),
            str(row.get("name", "")),
        ),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    timestamps: set[str] = set()
    for row in ordered:
        timestamp = str(row.get("physical_timestamp") or "")
        if timestamp in timestamps:
            continue
        if selected and any(
            angular_separation_degrees(row["view_direction"], prior["view_direction"])
            < float(min_view_angle_degrees)
            for prior in selected
        ):
            continue
        selected.append(row)
        timestamps.add(timestamp)
        if len(selected) >= int(limit):
            return selected
    for row in ordered:
        timestamp = str(row.get("physical_timestamp") or "")
        if timestamp in timestamps:
            continue
        selected.append(row)
        timestamps.add(timestamp)
        if len(selected) >= int(limit):
            break
    return selected



def allocate_global_view_budget(
    object_rows: list[dict[str, object]],
    *,
    max_total_views: int,
) -> tuple[dict[str, dict[str, object]], list[list[dict[str, object]]]]:
    """Allocate scarce images round-robin so later objects are not starved."""

    if max_total_views < 1:
        raise ValueError("max_total_views must be positive")
    selected_global: dict[str, dict[str, object]] = {}
    admitted: list[list[dict[str, object]]] = [[] for _ in object_rows]
    maximum_rank = max(
        (len(row.get("diverse_candidates") or []) for row in object_rows),
        default=0,
    )
    for rank in range(maximum_rank):
        for object_index, object_row in enumerate(object_rows):
            candidates = object_row.get("diverse_candidates") or []
            if rank >= len(candidates):
                continue
            candidate = candidates[rank]
            name = str(candidate["name"])
            if name not in selected_global:
                if len(selected_global) >= max_total_views:
                    continue
                selected_global[name] = candidate
            admitted[object_index].append(candidate)
    return selected_global, admitted

def object_rescue_priority(state: dict, index: int, observations: int) -> tuple[float, list[str]]:
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
    semantic_statuses = state.get("object_semantic_status") or []
    semantic_status = str(
        semantic_statuses[index] if index < len(semantic_statuses) else ""
    ).lower()
    if "unresolved" in semantic_status or "component" in semantic_status:
        score += 5.0
        reasons.append("semantic_whole_object_uncertain")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-names-output", type=Path)
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--views-per-object", type=int, default=5)
    parser.add_argument("--max-total-views", type=int, default=96)
    parser.add_argument("--sharpness-candidates", type=int, default=8)
    parser.add_argument("--min-view-angle-degrees", type=float, default=12.0)
    parser.add_argument("--min-margin-ratio", type=float, default=0.035)
    parser.add_argument("--min-area-ratio", type=float, default=0.0025)
    parser.add_argument("--max-area-ratio", type=float, default=0.55)
    parser.add_argument("--min-priority", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.max_objects, args.views_per_object, args.max_total_views) < 1:
        parser.error("object/view budgets must be positive")

    started = time.perf_counter()
    frames_path = args.frames_json.expanduser().resolve()
    frames_payload = json.loads(frames_path.read_text(encoding="utf-8"))
    frame_rows = frames_payload.get("frames") if isinstance(frames_payload, dict) else None
    if not isinstance(frame_rows, list):
        raise ValueError(f"no frames list in {frames_path}")
    scale = float(frames_payload.get("meters_per_scene_unit") or 0.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("frames JSON lacks positive meters_per_scene_unit")
    identity_pattern_text = str(
        (frames_payload.get("identity_contract") or {}).get("regex") or ""
    ).strip()
    identity_pattern = re.compile(identity_pattern_text) if identity_pattern_text else None
    existing_names = {
        str(row.get("source_image") or "")
        for row in frame_rows
        if isinstance(row, dict)
    }
    existing_timestamps = {
        identity_key(name, identity_pattern)[0] for name in existing_names if name
    }

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("pycolmap is required for full-COLMAP rescue planning") from exc
    reconstruction = pycolmap.Reconstruction(str(args.colmap_model.expanduser().resolve()))
    image_root = args.image_root.expanduser().resolve()
    state = load_state(args.scene_state.expanduser().resolve())
    object_ids = _numpy(state.get("object_id"), np.int64).reshape(-1)
    active = _numpy(state.get("active"), bool).reshape(-1)
    centers = _numpy(state.get("object_box_centers_m", state.get("means")), np.float64)
    dimensions = _numpy(state.get("object_box_dimensions_m"), np.float64)
    quaternions = _numpy(state.get("object_box_wxyz"), np.float64)
    observations = state.get("object_mask_observations") or []

    object_queue: list[dict[str, object]] = []
    for index, object_id_raw in enumerate(object_ids.tolist()):
        if index >= centers.shape[0] or index >= dimensions.shape[0] or index >= quaternions.shape[0]:
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
        if priority < float(args.min_priority):
            continue
        if index < active.size and not bool(active[index]) and "geometry_rejected" not in reasons:
            continue
        try:
            corners = obb_corners(centers[index], dimensions[index], quaternions[index])
        except ValueError:
            continue
        object_queue.append(
            {
                "object_id": int(object_id_raw),
                "state_index": index,
                "active": bool(index < active.size and active[index]),
                "priority": priority,
                "priority_reasons": reasons,
                "observations": unique_observations,
                "corners_m": corners,
            }
        )
    object_queue = stratified_object_queue(object_queue, int(args.max_objects))

    planned_objects: list[dict[str, object]] = []
    for object_row in object_queue:
        candidates: list[dict[str, object]] = []
        corners = np.asarray(object_row.pop("corners_m"), dtype=np.float64)
        for image in reconstruction.images.values():
            name = str(image.name)
            timestamp, camera_name, family = identity_key(name, identity_pattern)
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
                1.12 if family == "center" else 0.97
            )
            candidates.append(
                {
                    "name": name,
                    "colmap_image_id": int(image.image_id),
                    "colmap_camera_id": int(image.camera_id),
                    "physical_timestamp": timestamp,
                    "camera": camera_name,
                    "family": family,
                    **projected,
                }
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
        diverse = choose_diverse_views(
            candidates,
            limit=int(args.views_per_object),
            min_view_angle_degrees=float(args.min_view_angle_degrees),
        )
        planned_objects.append(
            {
                **object_row,
                "candidate_count": len(candidates),
                "diverse_candidates": diverse,
            }
        )

    selected_global, admitted_by_object = allocate_global_view_budget(
        planned_objects, max_total_views=int(args.max_total_views)
    )
    output_objects: list[dict[str, object]] = []
    for object_row, admitted in zip(planned_objects, admitted_by_object):
        output_objects.append(
            {
                **{
                    key: value
                    for key, value in object_row.items()
                    if key != "diverse_candidates"
                },
                "selected_count": len(admitted),
                "selected_views": admitted,
            }
        )

    selected_names = sorted(
        selected_global,
        key=lambda name: (
            str(selected_global[name].get("physical_timestamp") or ""),
            str(selected_global[name].get("camera") or ""),
            str(selected_global[name].get("family") or ""),
            name,
        ),
    )
    report = {
        "schema": "farm.full-colmap-rescue-plan.v1",
        "created_unix_s": time.time(),
        "source_scene_state": str(args.scene_state.expanduser().resolve()),
        "colmap_model": str(args.colmap_model.expanduser().resolve()),
        "image_root": str(image_root),
        "frames_json": str(frames_path),
        "meters_per_scene_unit": scale,
        "policy": {
            "adaptive_only": True,
            "category_agnostic_view_ranking": True,
            "exclude_existing_physical_timestamps": True,
            "whole_object_frame_required": True,
            "max_objects": int(args.max_objects),
            "views_per_object": int(args.views_per_object),
            "max_total_views": int(args.max_total_views),
            "sharpness_candidates": int(args.sharpness_candidates),
            "min_view_angle_degrees": float(args.min_view_angle_degrees),
            "min_margin_ratio": float(args.min_margin_ratio),
            "min_area_ratio": float(args.min_area_ratio),
            "max_area_ratio": float(args.max_area_ratio),
            "min_priority": float(args.min_priority),
        },
        "registered_images": len(reconstruction.images),
        "existing_selected_images": len(existing_names),
        "objects_considered": len(object_queue),
        "objects_with_rescue_views": sum(
            int(row["selected_count"]) > 0 for row in output_objects
        ),
        "unique_rescue_views": len(selected_names),
        "selected_names": selected_names,
        "objects": output_objects,
        "timing": {"duration_seconds": time.perf_counter() - started},
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

