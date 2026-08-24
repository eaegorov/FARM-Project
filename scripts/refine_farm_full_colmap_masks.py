#!/usr/bin/env python3
"""Associate bounded rescue tracks to metric objects and refine their masks.

This pass is intentionally category agnostic.  It first proves that a rescue
track belongs to one planned metric OBB in at least two independent images.
Only then does SAM3 receive automatic positive/negative points and a geometric
box.  The output mask must retain the seed, reject competing masks, and remain
consistent with rendered depth before it can enter the canonical state.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_graph.utils.geometry import decode_voxel_keys_numpy  # noqa: E402

from farm_runtime.full_colmap_rescue import (  # noqa: E402
    backproject_mask_points,
    clip_bbox,
    connected_component_overlapping_seed,
    distributed_points,
    feature_cosine,
    frame_by_state_image,
    mask_bbox,
    mask_box_metrics,
    mask_shape_metrics,
    numpy_array,
    pack_mask,
    points_inside_obb,
    project_world_points_seed,
    project_metric_obb,
    resolve_observation_path,
    sha256_file,
    multiview_voxel_consensus,
    voxel_downsample_points,
    unpack_mask_archive,
)


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported scene state: {path}")
    return state


def load_frames(path: Path) -> tuple[dict, list[dict], dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"frames JSON lacks frames: {path}")
    frames = [dict(row) for row in rows if isinstance(row, dict)]
    by_stem = {
        Path(str(row.get("rgb_path") or "")).stem: row
        for row in frames
        if str(row.get("rgb_path") or "")
    }
    if len(by_stem) != len(frames):
        raise ValueError("rescue frames have missing or duplicate RGB stems")
    return payload, frames, by_stem


def validate_metric_contract(plan: Mapping[str, Any], frames: Mapping[str, Any]) -> None:
    """Fail closed before SAM when renderer and OBB metric scales differ."""

    planned = float(plan.get("meters_per_scene_unit") or 0.0)
    rendered = float(frames.get("meters_per_scene_unit") or 0.0)
    if not math.isfinite(planned) or planned <= 0.0:
        raise ValueError("rescue plan lacks a positive meters_per_scene_unit")
    if not math.isfinite(rendered) or rendered <= 0.0:
        raise ValueError("rescue frames lack a positive meters_per_scene_unit")
    tolerance = max(1e-9, 1e-9 * abs(planned))
    if abs(planned - rendered) > tolerance:
        raise ValueError(
            "rescue renderer metric scale does not match the full-COLMAP plan: "
            f"{rendered:.12g} != {planned:.12g}"
        )
    if str(frames.get("depth_units") or "") != "metres":
        raise ValueError("rescue depth must be expressed in metres")
    if str(frames.get("pose_translation_units") or "") != "metres":
        raise ValueError("rescue camera translations must be expressed in metres")


def _feature_at(state: dict, index: int) -> np.ndarray:
    values = state.get("features")
    if values is None:
        return np.zeros((0,), dtype=np.float32)
    array = numpy_array(values, np.float32)
    return array[index] if array.ndim == 2 and 0 <= index < array.shape[0] else np.zeros((0,), dtype=np.float32)



def _object_voxel_points(state: dict, index: int) -> np.ndarray:
    flat = numpy_array(state.get("object_voxel_keys_flat", []), np.int64).reshape(-1)
    offsets = numpy_array(state.get("object_voxel_keys_offsets", []), np.int64).reshape(-1)
    levels = numpy_array(state.get("object_voxel_levels", []), np.int64).reshape(-1)
    if index < 0 or index >= levels.size or index + 1 >= offsets.size:
        return np.zeros((0, 3), dtype=np.float32)
    start, end = int(offsets[index]), int(offsets[index + 1])
    if start < 0 or end <= start or end > flat.size:
        return np.zeros((0, 3), dtype=np.float32)
    points = decode_voxel_keys_numpy(flat[start:end], int(levels[index]))
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def add_projected_voxel_fallbacks(
    plan: dict,
    source_state: dict,
    rescue_state: dict,
    rescue_frames_by_stem: Mapping[str, Mapping[str, Any]],
    associations: list[dict],
    *,
    minimum_views: int,
) -> list[dict]:
    """Add geometry-only candidates when transient detector tracks miss an object."""

    source_ids = numpy_array(source_state["object_id"], np.int64).reshape(-1)
    source_index = {int(value): index for index, value in enumerate(source_ids.tolist())}
    centers = numpy_array(source_state["object_box_centers_m"], np.float64)
    dimensions = numpy_array(source_state["object_box_dimensions_m"], np.float64)
    quaternions = numpy_array(source_state["object_box_wxyz"], np.float64)
    image_by_source: dict[str, int] = {}
    rescue_images = list(rescue_state.get("images") or [])
    for image_id in range(len(rescue_images)):
        frame = frame_by_state_image(rescue_images, rescue_frames_by_stem, image_id)
        if frame is None:
            continue
        source_image = str(frame.get("source_image") or "")
        if source_image and source_image not in image_by_source:
            image_by_source[source_image] = image_id
    existing = {int(row["object_id"]) for row in associations}
    output = list(associations)
    for object_row in plan.get("objects") or []:
        if not isinstance(object_row, dict):
            continue
        object_id = int(object_row.get("object_id", -1))
        index = source_index.get(object_id)
        if object_id in existing or index is None:
            continue
        points = _object_voxel_points(source_state, index)
        if points.shape[0] < 24:
            continue
        matched_views: list[dict] = []
        for selected in object_row.get("selected_views") or []:
            if not isinstance(selected, dict):
                continue
            source_image = str(selected.get("name") or "")
            image_id = image_by_source.get(source_image)
            if image_id is None:
                continue
            matched_views.append({
                "rescue_image_id": int(image_id),
                "source_image": source_image,
                "seed_origin": "projected_metric_voxels",
            })
        if len(matched_views) < max(1, int(minimum_views)):
            continue
        output.append({
            "object_id": object_id,
            "source_state_index": int(index),
            "priority": float(object_row.get("priority") or 0.0),
            "priority_reasons": list(object_row.get("priority_reasons") or []),
            "association_margin": None,
            "candidate_count": 0,
            "target_center_m": centers[index].tolist(),
            "target_dimensions_m": dimensions[index].tolist(),
            "target_wxyz": quaternions[index].tolist(),
            "rescue_track_index": None,
            "rescue_track_id": None,
            "score": None,
            "normalized_center_distance": None,
            "feature_cosine_similarity": None,
            "median_mask_inside_box": None,
            "median_box_pixel_coverage": None,
            "median_bbox_iou": None,
            "association_mode": "projected_metric_voxel_fallback",
            "matched_views": matched_views,
        })
    return sorted(output, key=lambda row: int(row["object_id"]))

def associate_tracks(
    plan: dict,
    source_state: dict,
    rescue_state: dict,
    rescue_frames_by_stem: Mapping[str, Mapping[str, Any]],
    rescue_mask_root: Path,
    *,
    minimum_views: int,
    maximum_normalized_center_distance: float,
    minimum_score: float,
    minimum_margin: float,
) -> list[dict]:
    source_ids = numpy_array(source_state["object_id"], np.int64).reshape(-1)
    source_index = {int(value): index for index, value in enumerate(source_ids.tolist())}
    centers = numpy_array(source_state["object_box_centers_m"], np.float64)
    dimensions = numpy_array(source_state["object_box_dimensions_m"], np.float64)
    quaternions = numpy_array(source_state["object_box_wxyz"], np.float64)
    rescue_means = numpy_array(rescue_state["means"], np.float64)
    rescue_images = list(rescue_state.get("images") or [])
    rescue_observations = list(rescue_state.get("object_mask_observations") or [])
    target_rows: list[dict] = []

    for object_row in plan.get("objects") or []:
        if not isinstance(object_row, dict) or int(object_row.get("selected_count") or 0) < minimum_views:
            continue
        object_id = int(object_row["object_id"])
        index = source_index.get(object_id)
        if index is None or index >= centers.shape[0] or index >= dimensions.shape[0]:
            continue
        selected_names = {
            str(row.get("name") or "")
            for row in object_row.get("selected_views") or []
            if isinstance(row, dict)
        }
        candidates: list[dict] = []
        scale = max(float(np.linalg.norm(dimensions[index])), 0.25)
        for rescue_index in range(rescue_means.shape[0]):
            normalized_distance = float(np.linalg.norm(rescue_means[rescue_index] - centers[index]) / scale)
            if not math.isfinite(normalized_distance) or normalized_distance > maximum_normalized_center_distance:
                continue
            feature_similarity = feature_cosine(
                _feature_at(source_state, index), _feature_at(rescue_state, rescue_index)
            )
            matched_views: list[dict] = []
            observations = rescue_observations[rescue_index] if rescue_index < len(rescue_observations) else []
            for observation in observations if isinstance(observations, list) else []:
                if not isinstance(observation, Mapping):
                    continue
                rescue_image_id = int(observation.get("image_id", -1))
                frame = frame_by_state_image(rescue_images, rescue_frames_by_stem, rescue_image_id)
                if frame is None or str(frame.get("source_image") or "") not in selected_names:
                    continue
                sidecar = resolve_observation_path(observation, rescue_mask_root)
                if sidecar is None:
                    continue
                raw = unpack_mask_archive(sidecar, "raw")
                projected = project_metric_obb(
                    centers[index], dimensions[index], quaternions[index], frame
                )
                if projected is None:
                    continue
                metrics = mask_box_metrics(raw, projected)
                accepted = (
                    metrics["mask_inside_box"] >= 0.20
                    and (
                        metrics["box_pixel_coverage"] >= 0.025
                        or metrics["bbox_iou"] >= 0.04
                    )
                )
                if accepted:
                    matched_views.append(
                        {
                            "rescue_image_id": rescue_image_id,
                            "source_image": str(frame.get("source_image") or ""),
                            "mask": str(sidecar),
                            **metrics,
                        }
                    )
            if len(matched_views) < minimum_views:
                continue
            median_inside = float(np.median([row["mask_inside_box"] for row in matched_views]))
            median_coverage = float(np.median([row["box_pixel_coverage"] for row in matched_views]))
            median_iou = float(np.median([row["bbox_iou"] for row in matched_views]))
            score = (
                0.34 * min(len(matched_views) / max(minimum_views + 1, 1), 1.0)
                + 0.26 * math.exp(-normalized_distance)
                + 0.16 * max(0.0, feature_similarity)
                + 0.14 * median_inside
                + 0.10 * min(1.0, median_coverage / 0.20)
            )
            candidates.append(
                {
                    "rescue_track_index": rescue_index,
                    "rescue_track_id": int(numpy_array(rescue_state["object_id"], np.int64)[rescue_index]),
                    "score": score,
                    "normalized_center_distance": normalized_distance,
                    "feature_cosine_similarity": feature_similarity,
                    "median_mask_inside_box": median_inside,
                    "median_box_pixel_coverage": median_coverage,
                    "median_bbox_iou": median_iou,
                    "matched_views": matched_views,
                }
            )
        candidates.sort(key=lambda row: (float(row["score"]), len(row["matched_views"])), reverse=True)
        if not candidates:
            continue
        best = candidates[0]
        runner_up = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        margin = float(best["score"]) - runner_up
        if float(best["score"]) < minimum_score or margin < minimum_margin:
            continue
        target_rows.append(
            {
                "object_id": object_id,
                "source_state_index": index,
                "priority": float(object_row.get("priority") or 0.0),
                "priority_reasons": list(object_row.get("priority_reasons") or []),
                "association_margin": margin,
                "candidate_count": len(candidates),
                "target_center_m": centers[index].tolist(),
                "target_dimensions_m": dimensions[index].tolist(),
                "target_wxyz": quaternions[index].tolist(),
                **best,
            }
        )

    # A transient rescue track cannot prove two different canonical objects.
    target_rows.sort(key=lambda row: (float(row["score"]), float(row["association_margin"])), reverse=True)
    used_tracks: set[int] = set()
    exclusive: list[dict] = []
    for row in target_rows:
        track = int(row["rescue_track_index"])
        if track in used_tracks:
            continue
        used_tracks.add(track)
        exclusive.append(row)
    exclusive.sort(key=lambda row: int(row["object_id"]))
    return exclusive


def _all_masks_for_image(rescue_state: dict, root: Path, image_id: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for observations in rescue_state.get("object_mask_observations") or []:
        for observation in observations if isinstance(observations, list) else []:
            if not isinstance(observation, Mapping) or int(observation.get("image_id", -1)) != image_id:
                continue
            path = resolve_observation_path(observation, root)
            if path is not None:
                masks.append(unpack_mask_archive(path, "raw"))
    return masks




def _depth_for_frame(frames_path: Path, frame: Mapping[str, Any]) -> np.ndarray:
    depth_path = (frames_path.parent / str(frame["depth_path"])).resolve(strict=True)
    return np.asarray(np.load(depth_path, mmap_mode="r"), dtype=np.float32)


def _multiview_world_support(
    association: Mapping[str, Any],
    rescue_state: Mapping[str, Any],
    rescue_frames_by_stem: Mapping[str, Mapping[str, Any]],
    rescue_root: Path,
    frames_path: Path,
    object_voxels: np.ndarray,
    *,
    minimum_views: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse independent full-COLMAP mask seeds into bounded metric support.

    This is the useful REST3D pattern without trusting a semantic label: masks
    from independent camera views are backprojected, clipped to the metric OBB,
    voxel-deduplicated, and only then reused as SAM3 prompts in every view.
    """

    if str(association.get("association_mode") or "") == "projected_metric_voxel_fallback":
        points = np.asarray(object_voxels, dtype=np.float32).reshape(-1, 3)
        if points.size:
            points = points[points_inside_obb(
                points,
                association["target_center_m"],
                association["target_dimensions_m"],
                association["target_wxyz"],
                expansion=1.15,
            )]
        support = voxel_downsample_points(points)
        return support, {
            "origin": "source_metric_voxels",
            "contributing_views": 0,
            "raw_points": int(points.shape[0]),
            "support_points": int(support.shape[0]),
        }

    images = list(rescue_state.get("images") or [])
    per_view: list[np.ndarray] = []
    contributors: list[str] = []
    raw_points = 0
    inside_points = 0
    for view in association.get("matched_views") or []:
        if not isinstance(view, Mapping) or not str(view.get("mask") or ""):
            continue
        image_id = int(view.get("rescue_image_id", -1))
        frame = frame_by_state_image(images, rescue_frames_by_stem, image_id)
        if frame is None:
            continue
        raw = unpack_mask_archive(Path(str(view["mask"])).resolve(strict=True), "raw")
        points = backproject_mask_points(raw, _depth_for_frame(frames_path, frame), frame)
        raw_points += int(points.shape[0])
        if points.size:
            points = points[points_inside_obb(
                points,
                association["target_center_m"],
                association["target_dimensions_m"],
                association["target_wxyz"],
                expansion=1.35,
            )]
        inside_points += int(points.shape[0])
        if points.shape[0] < 24:
            continue
        per_view.append(points)
        contributors.append(str(frame.get("source_image") or image_id))
    if len(per_view) < int(minimum_views):
        return np.zeros((0, 3), dtype=np.float32), {
            "origin": "independent_full_colmap_masks",
            "contributing_views": len(per_view),
            "contributors": contributors,
            "raw_points": raw_points,
            "inside_obb_points": inside_points,
            "support_points": 0,
        }
    support, consensus_audit = multiview_voxel_consensus(
        per_view, minimum_views=int(minimum_views)
    )
    return support, {
        "origin": "independent_full_colmap_masks",
        "contributing_views": len(per_view),
        "contributors": contributors,
        "raw_points": raw_points,
        "voxel_consensus": consensus_audit,
        "inside_obb_points": inside_points,
        "support_points": int(support.shape[0]),
    }


def _prompt_box(raw: np.ndarray, projected: np.ndarray, expansion: float) -> np.ndarray:
    raw_bbox = mask_bbox(raw)
    if raw_bbox is None:
        raise ValueError("empty seed mask")
    union = np.r_[np.minimum(raw_bbox[:2], projected[:2]), np.maximum(raw_bbox[2:], projected[2:])]
    span = np.maximum(union[2:] - union[:2], 8.0)
    union[:2] -= expansion * span + 8.0
    union[2:] += expansion * span + 8.0
    clipped = clip_bbox(union, raw.shape[1], raw.shape[0])
    if clipped is None:
        raise ValueError("invalid prompt box")
    return clipped


def _infer_sam3(torch_module, processor, model, rgb: np.ndarray, points, labels, box):
    session = processor.init_video_session(
        video=[[Image.fromarray(np.ascontiguousarray(rgb))]],
        inference_device="cuda",
        inference_state_device="cuda",
        processing_device="cpu",
        video_storage_device="cuda",
        max_vision_features_cache_size=1,
        dtype=torch_module.bfloat16,
    )
    processor.add_inputs_to_inference_session(
        session,
        frame_idx=0,
        obj_ids=1,
        input_points=[[points]],
        input_labels=[[labels]],
        input_boxes=[[[float(value) for value in box]]],
    )
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.inference_mode():
        output = model(session, frame_idx=0)
    torch_module.cuda.synchronize()
    masks = processor.post_process_masks(
        [output.pred_masks.detach().float()],
        original_sizes=[[rgb.shape[0], rgb.shape[1]]],
        mask_threshold=0,
        binarize=True,
    )[0]
    while masks.ndim > 2:
        masks = masks[0]
    logits = getattr(output, "object_score_logits", None)
    confidence = 1.0
    if logits is not None:
        value = float(logits.detach().float().mean().cpu().item())
        confidence = 1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0))))
    return masks.detach().cpu().numpy() > 0, confidence, time.perf_counter() - started


def _depth_refine(
    sam_mask: np.ndarray,
    seed: np.ndarray,
    depth: np.ndarray,
    frame: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    preserve_seed: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    valid_depth = np.isfinite(depth) & (depth > 0.05) & (depth < 80.0)
    seed_depth = np.asarray(depth[np.asarray(seed, dtype=bool) & valid_depth], dtype=np.float32)
    if seed_depth.size < 24:
        return np.asarray(seed, dtype=bool), {
            "depth_valid_fraction": 0.0,
            "depth_or_obb_inlier_fraction": 0.0,
            "obb_inside_fraction": 0.0,
        }
    low, high = np.percentile(seed_depth, [2.0, 98.0])
    median = float(np.median(seed_depth))
    mad = float(np.median(np.abs(seed_depth - median)))
    padding = max(0.20, 3.0 * 1.4826 * mad)
    depth_gate = valid_depth & (depth >= low - padding) & (depth <= high + padding)
    sam = np.asarray(sam_mask, dtype=bool)
    ys, xs = np.nonzero(sam & valid_depth)
    inside_canvas = np.zeros_like(sam)
    if ys.size:
        points = backproject_mask_points(sam & valid_depth, depth, frame)
        inside = points_inside_obb(
            points,
            target["target_center_m"],
            target["target_dimensions_m"],
            target["target_wxyz"],
            expansion=1.45,
        )
        inside_canvas[ys, xs] = inside
    gated = sam & valid_depth & (depth_gate | inside_canvas)
    component = connected_component_overlapping_seed(
        gated | seed if preserve_seed else gated, seed
    )
    final = component | seed if preserve_seed else component
    sam_pixels = max(int(sam.sum()), 1)
    valid_pixels = int((sam & valid_depth).sum())
    return final, {
        "depth_valid_fraction": valid_pixels / sam_pixels,
        "depth_or_obb_inlier_fraction": int((sam & (depth_gate | inside_canvas)).sum()) / sam_pixels,
        "obb_inside_fraction": int((sam & inside_canvas).sum()) / sam_pixels,
    }


def _encode_crop(rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("cannot encode empty crop")
    x0, y0, x1, y1 = [int(value) for value in bbox]
    pad_x = max(32, int(round(0.45 * (x1 - x0))))
    pad_y = max(32, int(round(0.45 * (y1 - y0))))
    left, top = max(0, x0 - pad_x), max(0, y0 - pad_y)
    right, bottom = min(rgb.shape[1], x1 + pad_x), min(rgb.shape[0], y1 + pad_y)
    crop = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise RuntimeError("failed to JPEG-encode refined crop")
    return {
        "crop_jpeg_bytes": np.asarray(encoded, dtype=np.uint8),
        "crop_bbox_xyxy": np.asarray([x0 - left, y0 - top, x1 - left, y1 - top], dtype=np.int32),
        "crop_shape": np.asarray([bottom - top, right - left], dtype=np.int32),
    }


def _write_sidecar(path: Path, rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    raw_bits, raw_shape, raw_bbox = pack_mask(mask)
    crop = _encode_crop(rgb, mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_shape=np.asarray(mask.shape, dtype=np.int32),
        raw_bits=raw_bits,
        raw_shape=raw_shape,
        raw_bbox_xyxy=raw_bbox,
        inlier_bits=raw_bits,
        inlier_shape=raw_shape,
        inlier_bbox_xyxy=raw_bbox,
        **crop,
    )
    return {
        "mask_relative": path.as_posix(),
        "mask_sha256": sha256_file(path),
        "stored_mask_pixels": int(mask.sum()),
        "crop_jpeg_bytes_len": int(crop["crop_jpeg_bytes"].size),
        "crop_bbox_xyxy": crop["crop_bbox_xyxy"].tolist(),
        "crop_shape": crop["crop_shape"].tolist(),
    }


def _write_visual(path: Path, rgb: np.ndarray, seed: np.ndarray, sam: np.ndarray, final: np.ndarray) -> None:
    def overlay(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        output = rgb.copy()
        output[mask] = (0.32 * output[mask] + 0.68 * np.asarray(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 255, 255), 2)
        return output

    panels = [rgb, overlay(seed, (245, 70, 90)), overlay(sam, (255, 190, 45)), overlay(final, (40, 225, 155))]
    sheet = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])




def _postprocess_tracker_mask(
    torch_module, processor, output, height: int, width: int
) -> tuple[np.ndarray, float]:
    masks = processor.post_process_masks(
        [output.pred_masks.detach().float()],
        original_sizes=[[int(height), int(width)]],
        mask_threshold=0,
        binarize=True,
    )[0]
    while masks.ndim > 2:
        masks = masks[0]
    logits = getattr(output, "object_score_logits", None)
    confidence = 1.0
    if logits is not None:
        value = float(logits.detach().float().mean().cpu().item())
        confidence = 1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0))))
    return masks.detach().cpu().numpy() > 0, confidence


def _propagate_best_mask_bidirectionally(
    torch_module,
    processor,
    model,
    association: Mapping[str, Any],
    view_results: list[dict],
    rescue_state: Mapping[str, Any],
    frames_by_stem: Mapping[str, Mapping[str, Any]],
    frames_path: Path,
    object_voxels: np.ndarray,
    world_support: np.ndarray,
    output: Path,
    source_image_count: int,
) -> tuple[list[dict], float, dict[str, Any]]:
    """Track the strongest accepted mask and revalidate every propagated view."""

    accepted_rows = [
        row for row in view_results
        if bool(row.get("accepted")) and str(row.get("mask_relative") or "")
    ]
    audit: dict[str, Any] = {
        "enabled": True,
        "seed_views": len(accepted_rows),
        "attempted_views": 0,
        "accepted_views": 0,
        "reason": None,
    }
    if not accepted_rows:
        audit["reason"] = "no_independently_accepted_seed"
        return view_results, 0.0, audit
    existing_by_source = {str(row.get("source_image") or ""): row for row in view_results}
    entries: list[dict[str, Any]] = []
    rescue_images = list(rescue_state.get("images") or [])
    seen_sources: set[str] = set()
    for matched in association.get("matched_views") or []:
        rescue_image_id = int(matched.get("rescue_image_id", -1))
        frame = frame_by_state_image(rescue_images, frames_by_stem, rescue_image_id)
        if frame is None:
            continue
        source_image = str(frame.get("source_image") or "")
        if not source_image or source_image in seen_sources:
            continue
        seen_sources.add(source_image)
        rgb_path = (frames_path.parent / str(frame["rgb_path"])).resolve(strict=True)
        depth_path = (frames_path.parent / str(frame["depth_path"])).resolve(strict=True)
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(rgb_path)
        entries.append({
            "source_image": source_image,
            "rescue_image_id": rescue_image_id,
            "frame": frame,
            "rgb": cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            "depth": np.asarray(np.load(depth_path, mmap_mode="r"), dtype=np.float32),
        })
    entries.sort(key=lambda row: (int(row["frame"].get("timestamp_ns") or 0), row["source_image"]))
    if len(entries) < 2:
        audit["reason"] = "fewer_than_two_video_frames"
        return view_results, 0.0, audit
    seed = max(
        accepted_rows,
        key=lambda row: (float(row.get("confidence") or 0.0), int(row.get("final_pixels") or 0)),
    )
    seed_source = str(seed["source_image"] or "")
    seed_indices = [index for index, row in enumerate(entries) if row["source_image"] == seed_source]
    if not seed_indices:
        audit["reason"] = "accepted_seed_not_in_video_sequence"
        return view_results, 0.0, audit
    seed_index = int(seed_indices[0])
    seed_mask = unpack_mask_archive(output / str(seed["mask_relative"]), "raw")
    video = [[Image.fromarray(np.ascontiguousarray(row["rgb"])) for row in entries]]

    def propagate_direction(reverse: bool) -> dict[int, Any]:
        # REST3D found that forward propagation mutates cached frame outputs.
        # A fresh, identically seeded session per direction prevents reverse
        # tracking from consuming forward-state residue.
        session = processor.init_video_session(
            video=video,
            inference_device="cuda",
            inference_state_device="cuda",
            processing_device="cpu",
            video_storage_device="cuda",
            max_vision_features_cache_size=max(1, len(entries)),
            dtype=torch_module.bfloat16,
        )
        processor.add_inputs_to_inference_session(
            session, frame_idx=seed_index, obj_ids=1, input_masks=[seed_mask]
        )
        return {
            int(predicted.frame_idx): predicted
            for predicted in model.propagate_in_video_iterator(
                session, start_frame_idx=seed_index,
                max_frame_num_to_track=len(entries), reverse=reverse
            )
        }

    torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.inference_mode():
        outputs = propagate_direction(False)
        if seed_index > 0:
            outputs.update(propagate_direction(True))
    torch_module.cuda.synchronize()
    seconds = time.perf_counter() - started
    audit["seed_source_image"] = seed_source
    audit["sequence_frames"] = len(entries)
    for frame_index, entry in enumerate(entries):
        source_image = str(entry["source_image"])
        prior = existing_by_source.get(source_image)
        if prior is not None and bool(prior.get("accepted")):
            continue
        predicted = outputs.get(frame_index)
        if predicted is None:
            continue
        audit["attempted_views"] += 1
        rgb = np.asarray(entry["rgb"], dtype=np.uint8)
        depth = np.asarray(entry["depth"], dtype=np.float32)
        frame = entry["frame"]
        sam, confidence = _postprocess_tracker_mask(
            torch_module, processor, predicted, rgb.shape[0], rgb.shape[1]
        )
        projected = project_metric_obb(
            association["target_center_m"], association["target_dimensions_m"],
            association["target_wxyz"], frame,
        )
        if projected is None:
            continue
        projected_seed, projected_seed_metrics = project_world_points_seed(
            object_voxels, frame, depth, dilation_pixels=5, minimum_projected_points=12
        )
        multiview_seed, multiview_projection_metrics = project_world_points_seed(
            world_support, frame, depth, dilation_pixels=3, minimum_projected_points=12
        )
        guidance_seed = projected_seed | multiview_seed
        if int(guidance_seed.sum()) < 24:
            continue
        final, depth_metrics = _depth_refine(
            sam, guidance_seed, depth, frame, association, preserve_seed=False
        )
        final_pixels = int(final.sum())
        seed_pixels = max(int(guidance_seed.sum()), 1)
        seed_recall = float(np.logical_and(final, guidance_seed).sum()) / seed_pixels
        border = np.zeros_like(final)
        border[:3] = border[-3:] = True
        border[:, :3] = border[:, -3:] = True
        border_fraction = float(np.logical_and(final, border).sum()) / max(final_pixels, 1)
        shape_metrics = mask_shape_metrics(final)
        projected_box_metrics = mask_box_metrics(final, projected)
        quality_gates = {
            "tracker_confidence": confidence >= 0.50,
            "propagated_metric_seed_coverage": seed_recall >= 0.70,
            "image_border": border_fraction <= 0.20,
            "depth_valid": depth_metrics["depth_valid_fraction"] >= 0.50,
            "depth_or_obb": depth_metrics["depth_or_obb_inlier_fraction"] >= 0.55,
            "dominant_component": shape_metrics["dominant_component_fraction"] >= 0.92,
            "significant_components": shape_metrics["significant_component_count"] <= 2,
            "projected_obb_containment": projected_box_metrics["mask_inside_box"] >= 0.70,
            "projected_obb_support": (
                projected_box_metrics["box_pixel_coverage"] >= 0.03
                or projected_box_metrics["bbox_iou"] >= 0.08
            ),
            "image_area": final_pixels <= int(0.45 * final.size),
        }
        accepted = _view_acceptance(final_pixels, quality_gates)
        global_image_id = source_image_count + int(entry["rescue_image_id"])
        visual_relative = Path("visuals") / (
            f"object_{int(association['object_id']):06d}__img_{global_image_id:06d}__propagated.jpg"
        )
        _write_visual(output / visual_relative, rgb, guidance_seed, sam, final)
        row = {
            "source_image": source_image,
            "rescue_image_id": int(entry["rescue_image_id"]),
            "global_image_id": global_image_id,
            "accepted": accepted,
            "seed_origin": "rest3d_bidirectional_propagation",
            "refinement_origin": "rest3d_bidirectional_propagation",
            "propagation_seed_source_image": seed_source,
            "confidence": confidence,
            "sam_pixels": int(sam.sum()),
            "final_pixels": final_pixels,
            "propagated_metric_seed_pixels": seed_pixels,
            "propagated_metric_seed_recall": seed_recall,
            "border_fraction": border_fraction,
            "rejection_reasons": [key for key, passed in quality_gates.items() if not passed],
            "quality_gates": quality_gates,
            "advisory_gates": {},
            "inference_seconds": seconds / max(len(outputs), 1),
            "visual_relative": visual_relative.as_posix(),
            "projection_seed_metrics": projected_seed_metrics,
            "multiview_support_projection": multiview_projection_metrics,
            **shape_metrics,
            **{f"projected_{key}": value for key, value in projected_box_metrics.items()},
            **depth_metrics,
        }
        if prior is not None:
            row["independent_attempt"] = prior
        if accepted:
            relative = Path("masks") / f"object_{int(association['object_id']):06d}" / f"img_{global_image_id:06d}_det_0000.npz"
            row.update(_write_sidecar(output / relative, rgb, final))
            row["mask_relative"] = relative.as_posix()
            audit["accepted_views"] += 1
        if prior is None:
            view_results.append(row)
        else:
            view_results[view_results.index(prior)] = row
    audit["reason"] = "completed"
    return view_results, seconds, audit


def _view_acceptance(final_pixels: int, hard_gates: Mapping[str, bool]) -> bool:
    """Accept a view only on safety gates; keep coverage evidence advisory."""

    return int(final_pixels) > 0 and all(bool(value) for value in hard_gates.values())


def _object_acceptance(
    view_rows: list[Mapping[str, Any]],
    support_audit: Mapping[str, Any],
    *,
    minimum_views: int,
    minimum_support_points: int = 24,
) -> tuple[bool, str, dict[str, bool | int]]:
    accepted_ids = {
        int(row["global_image_id"])
        for row in view_rows
        if bool(row.get("accepted")) and row.get("global_image_id") is not None
    }
    independent_ids = {
        int(row["global_image_id"])
        for row in view_rows
        if bool(row.get("accepted"))
        and row.get("global_image_id") is not None
        and str(row.get("refinement_origin") or "independent_sam3")
        != "rest3d_bidirectional_propagation"
    }
    propagated_ids = accepted_ids - independent_ids
    view_gate = len(accepted_ids) >= int(minimum_views)
    seed_gate = len(independent_ids) >= 1
    support_points = int(support_audit.get("support_points") or 0)
    support_gate = support_points >= int(minimum_support_points)
    accepted = view_gate and seed_gate and support_gate
    if accepted:
        status = "accepted"
    elif not support_gate:
        status = "rejected_insufficient_metric_support"
    elif not seed_gate:
        status = "rejected_without_independent_seed"
    else:
        status = "rejected_insufficient_refined_views"
    return accepted, status, {
        "accepted_refined_views": len(accepted_ids),
        "independent_refined_views": len(independent_ids),
        "propagated_refined_views": len(propagated_ids),
        "independent_seed_gate": seed_gate,
        "minimum_refined_views": int(minimum_views),
        "metric_support_points": support_points,
        "minimum_metric_support_points": int(minimum_support_points),
        "refined_view_gate": view_gate,
        "metric_support_gate": support_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--rescue-state", type=Path, required=True)
    parser.add_argument("--rescue-frames-json", type=Path, required=True)
    parser.add_argument("--rescue-mask-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-association-views", type=int, default=2)
    parser.add_argument("--maximum-normalized-center-distance", type=float, default=2.5)
    parser.add_argument("--minimum-association-score", type=float, default=0.45)
    parser.add_argument("--minimum-association-margin", type=float, default=0.07)
    parser.add_argument("--minimum-accepted-views", type=int, default=2)
    parser.add_argument("--box-expansion", type=float, default=0.12)
    parser.add_argument("--maximum-area-expansion", type=float, default=4.5)
    parser.add_argument(
        "--bidirectional-propagation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="REST3D-style forward/backward propagation from the best independently accepted seed.",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty rescue output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir()
    (output / "visuals").mkdir()
    started = time.perf_counter()
    plan_path = args.plan.expanduser().resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "farm.full-colmap-rescue-plan.v1":
        raise ValueError("unsupported full-COLMAP rescue plan")
    source_state = load_state(args.source_state.expanduser().resolve(strict=True))
    rescue_state = load_state(args.rescue_state.expanduser().resolve(strict=True))
    frames_path = args.rescue_frames_json.expanduser().resolve(strict=True)
    frames_payload, rescue_frames, frame_by_stem = load_frames(frames_path)
    validate_metric_contract(plan, frames_payload)
    rescue_root = args.rescue_mask_root.expanduser().resolve(strict=True)
    associations = associate_tracks(
        plan,
        source_state,
        rescue_state,
        frame_by_stem,
        rescue_root,
        minimum_views=int(args.minimum_association_views),
        maximum_normalized_center_distance=float(args.maximum_normalized_center_distance),
        minimum_score=float(args.minimum_association_score),
        minimum_margin=float(args.minimum_association_margin),
    )
    associations = add_projected_voxel_fallbacks(
        plan,
        source_state,
        rescue_state,
        frame_by_stem,
        associations,
        minimum_views=int(args.minimum_association_views),
    )
    model_load_seconds = 0.0
    inference_seconds = 0.0
    if associations:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 rescue requires CUDA; CPU fallback is forbidden")
        torch.cuda.reset_peak_memory_stats()
        model_path = args.model.expanduser().resolve(strict=True)
        load_started = time.perf_counter()
        processor = Sam3TrackerVideoProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = Sam3TrackerVideoModel.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.bfloat16
        ).eval().to("cuda")
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
    else:
        processor = model = None

    source_image_count = len(source_state.get("images") or [])
    object_results: list[dict] = []
    for association in associations:
        object_id = int(association["object_id"])
        object_voxels = _object_voxel_points(
            source_state, int(association["source_state_index"])
        )
        world_support, multiview_support_audit = _multiview_world_support(
            association,
            rescue_state,
            frame_by_stem,
            rescue_root,
            frames_path,
            object_voxels,
            minimum_views=int(args.minimum_association_views),
        )
        view_results: list[dict] = []
        for view in association["matched_views"]:
            rescue_image_id = int(view["rescue_image_id"])
            frame = frame_by_state_image(
                list(rescue_state.get("images") or []), frame_by_stem, rescue_image_id
            )
            if frame is None:
                continue
            rgb_path = (frames_path.parent / str(frame["rgb_path"])).resolve(strict=True)
            depth_path = (frames_path.parent / str(frame["depth_path"])).resolve(strict=True)
            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(rgb_path)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth = np.asarray(np.load(depth_path, mmap_mode="r"), dtype=np.float32)
            seed_origin = str(view.get("seed_origin") or "rescue_mapping_track")
            projected_seed = seed_origin == "projected_metric_voxels"
            projection_seed_metrics: dict[str, float | int] = {}
            if seed_origin == "projected_metric_voxels":
                raw, projection_seed_metrics = project_world_points_seed(
                    object_voxels, frame, depth, dilation_pixels=5,
                    minimum_projected_points=12,
                )
                seed = raw
                if int(seed.sum()) < 24:
                    view_results.append({
                        "source_image": str(frame.get("source_image") or ""),
                        "rescue_image_id": rescue_image_id,
                        "global_image_id": source_image_count + rescue_image_id,
                        "accepted": False,
                        "seed_origin": seed_origin,
                        "rejection_reasons": ["projected_voxel_seed_insufficient"],
                        "projection_seed_metrics": projection_seed_metrics,
                    })
                    continue
            else:
                raw_path = Path(str(view["mask"])).resolve(strict=True)
                raw = unpack_mask_archive(raw_path, "raw")
                try:
                    seed = unpack_mask_archive(raw_path, "inlier")
                except ValueError:
                    seed = raw
                if int(seed.sum()) < 24:
                    seed = raw
            projected = project_metric_obb(
                association["target_center_m"],
                association["target_dimensions_m"],
                association["target_wxyz"],
                frame,
            )
            if projected is None:
                continue
            multiview_seed, multiview_projection_metrics = project_world_points_seed(
                world_support,
                frame,
                depth,
                dilation_pixels=3,
                minimum_projected_points=12,
            )
            guidance_seed = seed | multiview_seed
            box = _prompt_box(raw | multiview_seed, projected, float(args.box_expansion))
            positives = distributed_points(guidance_seed, 12)
            other = np.zeros_like(raw)
            for candidate in _all_masks_for_image(rescue_state, rescue_root, rescue_image_id):
                if candidate.shape == raw.shape and not np.array_equal(candidate, raw):
                    other |= candidate
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            box_canvas = np.zeros_like(raw)
            box_canvas[y0:y1, x0:x1] = True
            seed_dilated = cv2.dilate(guidance_seed.astype(np.uint8), np.ones((41, 41), np.uint8), iterations=1).astype(bool)
            if projected_seed:
                projected_clipped = clip_bbox(
                    projected, raw.shape[1], raw.shape[0]
                )
                if projected_clipped is None:
                    continue
                px0, py0, px1, py1 = [
                    int(round(value)) for value in projected_clipped
                ]
                projected_canvas = np.zeros_like(raw)
                projected_canvas[py0:py1, px0:px1] = True
                projected_canvas = cv2.dilate(
                    projected_canvas.astype(np.uint8),
                    np.ones((17, 17), np.uint8), iterations=1,
                ).astype(bool)
                negative_region = box_canvas & ~projected_canvas
                hard_negative = other & negative_region
                ring = negative_region
            else:
                hard_negative = other & box_canvas & ~seed_dilated
                ring = box_canvas & ~seed_dilated
            negatives = (distributed_points(hard_negative, 6) + distributed_points(ring, 4))[:10]
            points = positives + negatives
            labels = [1] * len(positives) + [0] * len(negatives)
            if len(positives) < 3:
                continue
            sam, confidence, seconds = _infer_sam3(torch, processor, model, rgb, points, labels, box)
            inference_seconds += seconds
            final, depth_metrics = _depth_refine(
                sam, raw, depth, frame, association, preserve_seed=not projected_seed
            )
            raw_pixels = max(int(raw.sum()), 1)
            final_pixels = int(final.sum())
            raw_recall = float(np.logical_and(final, raw).sum()) / raw_pixels
            multiview_seed_pixels = int(multiview_seed.sum())
            multiview_seed_recall = (
                float(np.logical_and(final, multiview_seed).sum()) / multiview_seed_pixels
                if multiview_seed_pixels
                else 1.0
            )
            positive_inclusion = float(np.mean([final[int(y), int(x)] for x, y in positives]))
            negative_inclusion = float(np.mean([final[int(y), int(x)] for x, y in negatives])) if negatives else 0.0
            other_fraction = float(np.logical_and(final, other).sum()) / max(final_pixels, 1)
            area_expansion = final_pixels / raw_pixels
            border = np.zeros_like(final)
            border[:3] = border[-3:] = True
            border[:, :3] = border[:, -3:] = True
            border_fraction = float(np.logical_and(final, border).sum()) / max(final_pixels, 1)
            shape_metrics = mask_shape_metrics(final)
            projected_box_metrics = mask_box_metrics(final, projected)
            hard_gates = {
                "positive_inclusion": positive_inclusion >= 0.80,
                "negative_exclusion": negative_inclusion <= 0.35,
                "image_border": border_fraction <= 0.20,
                "depth_valid": depth_metrics["depth_valid_fraction"] >= 0.50,
                "depth_or_obb": depth_metrics["depth_or_obb_inlier_fraction"] >= 0.55,
                "dominant_component": shape_metrics["dominant_component_fraction"] >= 0.92,
                "significant_components": shape_metrics["significant_component_count"] <= 2,
            }
            advisory_gates = {
                "multiview_support_recall": multiview_seed_recall >= 0.80,
            }
            if projected_seed:
                origin_gates = {
                    "projected_seed_coverage": raw_recall >= 0.70,
                    "projected_obb_containment": projected_box_metrics["mask_inside_box"] >= 0.70,
                    "projected_obb_support": (
                        projected_box_metrics["box_pixel_coverage"] >= 0.03
                        or projected_box_metrics["bbox_iou"] >= 0.08
                    ),
                    "image_area": final_pixels <= int(0.45 * final.size),
                }
            else:
                origin_gates = {
                    "mapping_seed_recall": raw_recall >= 0.90,
                    "mapping_seed_area_expansion": (
                        1.0 <= area_expansion <= float(args.maximum_area_expansion)
                    ),
                }
            if not projected_seed:
                advisory_gates["competing_mask_exclusion"] = other_fraction <= 0.25
            quality_gates = {**hard_gates, **origin_gates}
            accepted = _view_acceptance(final_pixels, quality_gates)
            rejection_reasons = [
                key for key, passed in quality_gates.items() if not passed
            ]
            global_image_id = source_image_count + rescue_image_id
            visual_relative = Path("visuals") / f"object_{object_id:06d}__img_{global_image_id:06d}.jpg"
            _write_visual(output / visual_relative, rgb, raw, sam, final)
            row = {
                "source_image": str(frame.get("source_image") or ""),
                "rescue_image_id": rescue_image_id,
                "global_image_id": global_image_id,
                "accepted": accepted,
                "seed_origin": seed_origin,
                "projection_seed_metrics": projection_seed_metrics,
                "multiview_support_projection": multiview_projection_metrics,
                "multiview_seed_pixels": multiview_seed_pixels,
                "multiview_seed_recall": multiview_seed_recall,
                "confidence": confidence,
                "raw_pixels": raw_pixels,
                "sam_pixels": int(sam.sum()),
                "final_pixels": final_pixels,
                "raw_recall": raw_recall,
                "positive_inclusion": positive_inclusion,
                "negative_inclusion": negative_inclusion,
                "other_mask_fraction": other_fraction,
                "area_expansion": area_expansion,
                "border_fraction": border_fraction,
                "rejection_reasons": rejection_reasons,
                "quality_gates": quality_gates,
                "advisory_gates": advisory_gates,
                "inference_seconds": seconds,
                "prompt_box_xyxy": [float(value) for value in box],
                "positive_points": positives,
                "negative_points": negatives,
                "visual_relative": visual_relative.as_posix(),
                **shape_metrics,
                **{f"projected_{key}": value for key, value in projected_box_metrics.items()},
                **depth_metrics,
            }
            if accepted:
                relative = Path("masks") / f"object_{object_id:06d}" / f"img_{global_image_id:06d}_det_0000.npz"
                sidecar = _write_sidecar(output / relative, rgb, final)
                sidecar["mask_relative"] = relative.as_posix()
                row.update(sidecar)
            view_results.append(row)
        independent_accepted = [row for row in view_results if bool(row.get("accepted"))]
        propagation_audit: dict[str, Any] = {
            "enabled": bool(args.bidirectional_propagation),
            "reason": "not_required",
            "seed_views": len(independent_accepted),
            "attempted_views": 0,
            "accepted_views": 0,
        }
        if (
            bool(args.bidirectional_propagation)
            and 0 < len(independent_accepted) < len(association.get("matched_views") or [])
        ):
            view_results, propagation_seconds, propagation_audit = (
                _propagate_best_mask_bidirectionally(
                    torch, processor, model, association, view_results, rescue_state,
                    frame_by_stem, frames_path, object_voxels, world_support, output,
                    source_image_count,
                )
            )
            inference_seconds += propagation_seconds
        accepted_views = [row for row in view_results if bool(row.get("accepted"))]
        accepted, object_status, object_quality_gates = _object_acceptance(
            view_results,
            multiview_support_audit,
            minimum_views=int(args.minimum_accepted_views),
        )
        object_results.append(
            {
                **{key: value for key, value in association.items() if key != "matched_views"},
                "status": object_status,
                "accepted": accepted,
                "accepted_views": len(accepted_views),
                "multiview_support": multiview_support_audit,
                "bidirectional_propagation": propagation_audit,
                "object_quality_gates": object_quality_gates,
                "views": view_results,
            }
        )

    report = {
        "schema": "farm.full-colmap-mask-refinement.v1",
        "status": "PASS",
        "created_unix_s": time.time(),
        "plan": str(plan_path),
        "source_state": str(args.source_state.expanduser().resolve()),
        "rescue_state": str(args.rescue_state.expanduser().resolve()),
        "rescue_frames_json": str(frames_path),
        "model": {
            "path": str(args.model),
            "local_files_only": True,
            "text_prompts": False,
            "manual_prompts": False,
        },
        "policy": {
            "category_agnostic_association": True,
            "exclusive_track_assignment": True,
            "projected_metric_voxel_fallback": True,
            "rest3d_inspired_sparse_metric_seed": True,
            "rest3d_inspired_multiview_metric_fusion": True,
            "rest3d_bidirectional_video_propagation": bool(args.bidirectional_propagation),
            "independent_forward_reverse_sessions": True,
            "propagated_views_revalidated_with_metric_depth_and_obb": True,
            "multiview_support_gate": True,
            "automatic_positive_negative_points": True,
            "metric_obb_box_prompt": True,
            "rendered_depth_gate": True,
            "minimum_association_views": int(args.minimum_association_views),
            "minimum_accepted_views": int(args.minimum_accepted_views),
            "minimum_metric_support_points": 24,
        },
        "associations": len(associations),
        "accepted_objects": sum(bool(row["accepted"]) for row in object_results),
        "accepted_object_ids": [int(row["object_id"]) for row in object_results if row["accepted"]],
        "objects": object_results,
        "timing": {
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "resources": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "hashes": {
            "plan_sha256": sha256_file(plan_path),
            "source_state_sha256": sha256_file(args.source_state.expanduser().resolve()),
            "rescue_state_sha256": sha256_file(args.rescue_state.expanduser().resolve()),
        },
    }
    report_path = output / "result.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    marker = {
        "schema": "farm.full-colmap-mask-refinement.success.v1",
        "status": "success",
        "result": "result.json",
        "result_sha256": sha256_file(report_path),
    }
    (output / "_SUCCESS.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "associations", "accepted_objects", "accepted_object_ids")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
