#!/usr/bin/env python3
"""Associate bounded rescue tracks to metric objects and refine their masks.

This pass is intentionally category agnostic.  It first proves that a rescue
track belongs to one planned metric OBB in at least two independent images.
Only then does SAM3 receive automatic positive/negative points and a geometric
box.  The output mask must retain the seed, reject competing masks, and remain
consistent with rendered depth before it can enter the canonical state.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import gc
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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_graph.utils.geometry import decode_voxel_keys_numpy  # noqa: E402

from farm_runtime.full_colmap_refinement_contract import (  # noqa: E402
    HELDOUT_REFINEMENT_ROLE,
    HELDOUT_VIEW_ROLE,
    TRAIN_REFINEMENT_ROLE,
    prepare_plan_for_view_role,
)
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
from farm_runtime.second_chance_refinement import (  # noqa: E402
    evaluate_second_chance_candidate,
    load_concept_prompt_manifest,
    validate_acceptance_prompt_plan_binding,
    select_unique_identity_compatible_mapping_mask,
    select_cross_view_consistent_candidates,
)


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported scene state: {path}")
    return state


def load_object_id_file(path: Path | None) -> set[int]:
    """Load an optional exact object-ID allowlist used by refinement policy."""

    if path is None:
        return set()
    resolved = path.expanduser().resolve(strict=True)
    values: set[int] = set()
    for line_number, raw in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
        values.add(object_id)
    if not values:
        raise ValueError(f"object ID allowlist is empty: {resolved}")
    return values


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


def validate_metric_contract(
    plan: Mapping[str, Any], frames: Mapping[str, Any]
) -> None:
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
    return (
        array[index]
        if array.ndim == 2 and 0 <= index < array.shape[0]
        else np.zeros((0,), dtype=np.float32)
    )


def _object_voxel_points(state: dict, index: int) -> np.ndarray:
    flat = numpy_array(state.get("object_voxel_keys_flat", []), np.int64).reshape(-1)
    offsets = numpy_array(state.get("object_voxel_keys_offsets", []), np.int64).reshape(
        -1
    )
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
    view_split: str = "train",
) -> list[dict]:
    """Complete detector associations with role-scoped metric-voxel views.

    A detector track is useful evidence, but it must not hide the other views
    selected by the full-COLMAP planner. Missing selected views are therefore
    attempted with category-agnostic projected metric voxels. They remain
    explicitly tagged and must pass the same SAM3/depth/multiview gates later.
    """

    source_ids = numpy_array(source_state["object_id"], np.int64).reshape(-1)
    source_index = {
        int(value): index for index, value in enumerate(source_ids.tolist())
    }
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
    output = [
        {
            **row,
            "matched_views": [dict(view) for view in row.get("matched_views") or []],
        }
        for row in associations
    ]
    existing = {int(row["object_id"]): row for row in output}
    for object_row in plan.get("objects") or []:
        if not isinstance(object_row, dict):
            continue
        object_id = int(object_row.get("object_id", -1))
        index = source_index.get(object_id)
        if index is None:
            continue
        points = _object_voxel_points(source_state, index)
        if points.shape[0] < 24:
            continue
        association = existing.get(object_id)
        matched_views = association["matched_views"] if association is not None else []
        seen_sources = {
            str(view.get("source_image") or "")
            for view in matched_views
            if isinstance(view, Mapping)
        }
        selected_by_source = {
            str(view.get("name") or ""): view
            for view in object_row.get("selected_views") or []
            if isinstance(view, Mapping)
        }
        seen_timestamps: set[str] = set()
        for view in matched_views:
            if not isinstance(view, Mapping):
                continue
            physical_timestamp = str(view.get("physical_timestamp") or "")
            if physical_timestamp:
                seen_timestamps.add(physical_timestamp)
            source = str(view.get("source_image") or "")
            selected = selected_by_source.get(source)
            selected_timestamp = (
                str(selected.get("physical_timestamp") or "")
                if selected is not None
                else ""
            )
            if selected_timestamp:
                seen_timestamps.add(selected_timestamp)
        projected_completion_views = 0
        for selected in object_row.get("selected_views") or []:
            if not isinstance(selected, dict):
                continue
            split = str(selected.get("split") or "train")
            if split != view_split:
                continue
            source_image = str(selected.get("name") or "")
            physical_timestamp = str(selected.get("physical_timestamp") or "")
            if source_image in seen_sources or (
                physical_timestamp and physical_timestamp in seen_timestamps
            ):
                continue
            image_id = image_by_source.get(source_image)
            if image_id is None:
                continue
            matched_views.append(
                {
                    "rescue_image_id": int(image_id),
                    "source_image": source_image,
                    "physical_timestamp": physical_timestamp,
                    "seed_origin": "projected_metric_voxels",
                }
            )
            seen_sources.add(source_image)
            if physical_timestamp:
                seen_timestamps.add(physical_timestamp)
            projected_completion_views += 1
        if association is not None:
            if projected_completion_views:
                association["association_mode"] = (
                    "rescue_mapping_track_plus_projected_metric_voxel_completion"
                )
                association["projected_completion_views"] = projected_completion_views
            else:
                association.setdefault("association_mode", "rescue_mapping_track")
                association.setdefault("projected_completion_views", 0)
            continue
        if len(matched_views) < max(1, int(minimum_views)):
            continue
        fallback = {
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
            "projected_completion_views": len(matched_views),
            "matched_views": matched_views,
        }
        output.append(fallback)
        existing[object_id] = fallback
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
    source_index = {
        int(value): index for index, value in enumerate(source_ids.tolist())
    }
    centers = numpy_array(source_state["object_box_centers_m"], np.float64)
    dimensions = numpy_array(source_state["object_box_dimensions_m"], np.float64)
    quaternions = numpy_array(source_state["object_box_wxyz"], np.float64)
    rescue_means = numpy_array(rescue_state["means"], np.float64)
    rescue_images = list(rescue_state.get("images") or [])
    rescue_observations = list(rescue_state.get("object_mask_observations") or [])
    target_rows: list[dict] = []

    for object_row in plan.get("objects") or []:
        if (
            not isinstance(object_row, dict)
            or int(object_row.get("selected_count") or 0) < minimum_views
        ):
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
            normalized_distance = float(
                np.linalg.norm(rescue_means[rescue_index] - centers[index]) / scale
            )
            if (
                not math.isfinite(normalized_distance)
                or normalized_distance > maximum_normalized_center_distance
            ):
                continue
            feature_similarity = feature_cosine(
                _feature_at(source_state, index),
                _feature_at(rescue_state, rescue_index),
            )
            matched_views: list[dict] = []
            observations = (
                rescue_observations[rescue_index]
                if rescue_index < len(rescue_observations)
                else []
            )
            for observation in observations if isinstance(observations, list) else []:
                if not isinstance(observation, Mapping):
                    continue
                rescue_image_id = int(observation.get("image_id", -1))
                frame = frame_by_state_image(
                    rescue_images, rescue_frames_by_stem, rescue_image_id
                )
                if (
                    frame is None
                    or str(frame.get("source_image") or "") not in selected_names
                ):
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
                accepted = metrics["mask_inside_box"] >= 0.20 and (
                    metrics["box_pixel_coverage"] >= 0.025
                    or metrics["bbox_iou"] >= 0.04
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
            median_inside = float(
                np.median([row["mask_inside_box"] for row in matched_views])
            )
            median_coverage = float(
                np.median([row["box_pixel_coverage"] for row in matched_views])
            )
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
                    "rescue_track_id": int(
                        numpy_array(rescue_state["object_id"], np.int64)[rescue_index]
                    ),
                    "score": score,
                    "normalized_center_distance": normalized_distance,
                    "feature_cosine_similarity": feature_similarity,
                    "median_mask_inside_box": median_inside,
                    "median_box_pixel_coverage": median_coverage,
                    "median_bbox_iou": median_iou,
                    "matched_views": matched_views,
                }
            )
        candidates.sort(
            key=lambda row: (float(row["score"]), len(row["matched_views"])),
            reverse=True,
        )
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
    target_rows.sort(
        key=lambda row: (float(row["score"]), float(row["association_margin"])),
        reverse=True,
    )
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


def _all_masks_for_image(
    rescue_state: dict, root: Path, image_id: int
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for observations in rescue_state.get("object_mask_observations") or []:
        for observation in observations if isinstance(observations, list) else []:
            if (
                not isinstance(observation, Mapping)
                or int(observation.get("image_id", -1)) != image_id
            ):
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

    if (
        str(association.get("association_mode") or "")
        == "projected_metric_voxel_fallback"
    ):
        points = np.asarray(object_voxels, dtype=np.float32).reshape(-1, 3)
        if points.size:
            points = points[
                points_inside_obb(
                    points,
                    association["target_center_m"],
                    association["target_dimensions_m"],
                    association["target_wxyz"],
                    expansion=1.15,
                )
            ]
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
        points = backproject_mask_points(
            raw, _depth_for_frame(frames_path, frame), frame
        )
        raw_points += int(points.shape[0])
        if points.size:
            points = points[
                points_inside_obb(
                    points,
                    association["target_center_m"],
                    association["target_dimensions_m"],
                    association["target_wxyz"],
                    expansion=1.35,
                )
            ]
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
    union = np.r_[
        np.minimum(raw_bbox[:2], projected[:2]), np.maximum(raw_bbox[2:], projected[2:])
    ]
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


def _infer_sam3_concept(
    torch_module,
    processor,
    model,
    rgb: np.ndarray,
    text_prompt: str,
) -> tuple[list[dict[str, Any]], float]:
    """Return all text-prompt proposals; instance selection happens downstream."""

    session = processor.init_video_session(
        video=[[Image.fromarray(np.ascontiguousarray(rgb))]],
        inference_device="cuda",
        inference_state_device="cuda",
        processing_device="cpu",
        video_storage_device="cuda",
        max_vision_features_cache_size=1,
        dtype=torch_module.bfloat16,
    )
    processor.add_text_prompt(session, text_prompt)
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.inference_mode():
        raw = model(session, frame_idx=0)
    torch_module.cuda.synchronize()
    output = processor.postprocess_outputs(session, raw)
    masks = output["masks"]
    scores = output["scores"]
    boxes = output["boxes"]
    object_ids = output["object_ids"]
    proposals: list[dict[str, Any]] = []
    for index in range(int(masks.shape[0])):
        proposals.append(
            {
                "concept_object_id": int(object_ids[index].detach().cpu().item()),
                "confidence": float(scores[index].detach().float().cpu().item()),
                "box_xyxy": boxes[index].detach().float().cpu().tolist(),
                "mask": masks[index].detach().cpu().numpy().astype(bool),
            }
        )
    return proposals, time.perf_counter() - started


def _depth_refine(
    sam_mask: np.ndarray,
    seed: np.ndarray,
    depth: np.ndarray,
    frame: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    preserve_seed: bool,
    minimum_padding_m: float = 0.04,
    maximum_padding_m: float = 0.18,
    obb_expansion: float = 1.15,
) -> tuple[np.ndarray, dict[str, float]]:
    """Keep the SAM component on the seed's observed depth surface.

    The OBB is deliberately audit-only here. Letting an expanded, possibly
    stale OBB override the depth test creates a circular acceptance path:
    inaccurate boxes admit background pixels which then make the next box
    larger. Attached parts at a different depth must instead earn support in
    another physical view.
    """

    if minimum_padding_m < 0.0 or maximum_padding_m < minimum_padding_m:
        raise ValueError("depth padding must satisfy 0 <= minimum <= maximum")
    if obb_expansion < 1.0:
        raise ValueError("OBB audit expansion must be at least one")
    valid_depth = np.isfinite(depth) & (depth > 0.05) & (depth < 80.0)
    seed_depth = np.asarray(
        depth[np.asarray(seed, dtype=bool) & valid_depth], dtype=np.float32
    )
    if seed_depth.size < 24:
        return np.asarray(seed, dtype=bool), {
            "depth_valid_fraction": 0.0,
            "depth_inlier_fraction": 0.0,
            "depth_or_obb_inlier_fraction": 0.0,
            "obb_inside_fraction": 0.0,
            "depth_padding_m": 0.0,
            "obb_audit_expansion": float(obb_expansion),
        }
    low, high = np.percentile(seed_depth, [2.0, 98.0])
    median = float(np.median(seed_depth))
    mad = float(np.median(np.abs(seed_depth - median)))
    padding = float(np.clip(3.0 * 1.4826 * mad, minimum_padding_m, maximum_padding_m))
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
            expansion=obb_expansion,
        )
        inside_canvas[ys, xs] = inside
    gated = sam & valid_depth & depth_gate
    component = connected_component_overlapping_seed(
        gated | seed if preserve_seed else gated, seed
    )
    final = component | seed if preserve_seed else component
    sam_pixels = max(int(sam.sum()), 1)
    valid_pixels = int((sam & valid_depth).sum())
    return final, {
        "depth_valid_fraction": valid_pixels / sam_pixels,
        "depth_inlier_fraction": int((sam & depth_gate).sum()) / sam_pixels,
        "depth_or_obb_inlier_fraction": int((sam & (depth_gate | inside_canvas)).sum())
        / sam_pixels,
        "obb_inside_fraction": int((sam & inside_canvas).sum()) / sam_pixels,
        "depth_padding_m": padding,
        "obb_audit_expansion": float(obb_expansion),
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
        "crop_bbox_xyxy": np.asarray(
            [x0 - left, y0 - top, x1 - left, y1 - top], dtype=np.int32
        ),
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


def _write_visual(
    path: Path, rgb: np.ndarray, seed: np.ndarray, sam: np.ndarray, final: np.ndarray
) -> None:
    def overlay(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        output = rgb.copy()
        output[mask] = (0.32 * output[mask] + 0.68 * np.asarray(color)).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, (255, 255, 255), 2)
        return output

    panels = [
        rgb,
        overlay(seed, (245, 70, 90)),
        overlay(sam, (255, 190, 45)),
        overlay(final, (40, 225, 155)),
    ]
    sheet = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )


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


def tracking_stream_id(frame: Mapping[str, Any]) -> str:
    """Return a camera/view-family identity that must never be crossed by tracking."""

    camera = str(frame.get("camera") or "").strip()
    if camera:
        return camera
    source = str(frame.get("source_image") or "").strip()
    parts = Path(source).stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("cam") and parts[1].isdigit():
        return f"{parts[0]}_{'_'.join(parts[2:])}"
    # Unknown camera provenance is isolated instead of being silently mixed.
    return f"isolated:{source or frame.get('image_id') or 'unknown'}"


def tracking_physical_timestamp(frame: Mapping[str, Any]) -> int | None:
    """Return capture time without trusting the prepared RGB-D render order."""

    explicit = frame.get("physical_timestamp_ns")
    if explicit is not None:
        try:
            value = int(explicit)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    source = str(frame.get("source_image") or "").strip()
    parts = Path(source).stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("cam") and parts[1].isdigit():
        value = int(parts[1])
        return value if value > 0 else None
    fallback = frame.get("timestamp_ns")
    try:
        value = int(fallback)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _camera_motion(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[float | None, float | None]:
    """Return camera-centre translation and relative rotation in degrees."""

    first = np.asarray(left.get("T_world_cam"), dtype=np.float64)
    second = np.asarray(right.get("T_world_cam"), dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        return None, None
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None, None
    translation_m = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation_degrees = float(np.degrees(np.arccos(cosine)))
    return translation_m, rotation_degrees


def partition_tracking_episodes(
    entries: list[dict[str, Any]],
    *,
    maximum_gap_multiplier: float,
    maximum_translation_m: float,
    maximum_rotation_degrees: float,
    maximum_episode_frames: int,
) -> list[list[dict[str, Any]]]:
    """Split frames into camera-pure, continuous and motion-bounded episodes."""

    if not math.isfinite(float(maximum_gap_multiplier)) or maximum_gap_multiplier < 1:
        raise ValueError("maximum gap multiplier must be finite and at least one")
    if not math.isfinite(float(maximum_translation_m)) or maximum_translation_m <= 0:
        raise ValueError("maximum translation must be finite and positive")
    if (
        not math.isfinite(float(maximum_rotation_degrees))
        or not 0 < maximum_rotation_degrees <= 180
    ):
        raise ValueError("maximum rotation must be in (0, 180]")
    if int(maximum_episode_frames) < 2:
        raise ValueError("maximum episode frames must be at least two")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        row = dict(entry)
        row["tracking_stream_id"] = tracking_stream_id(row["frame"])
        grouped[row["tracking_stream_id"]].append(row)
    episodes: list[list[dict[str, Any]]] = []
    for stream_id in sorted(grouped):
        rows = sorted(
            grouped[stream_id],
            key=lambda row: (
                tracking_physical_timestamp(row["frame"]) or 0,
                str(row["source_image"]),
            ),
        )
        timestamps = [tracking_physical_timestamp(row["frame"]) for row in rows]
        positive_gaps = [
            right - left
            for left, right in zip(timestamps, timestamps[1:])
            if left is not None and right is not None and right > left
        ]
        nominal_gap = float(np.quantile(positive_gaps, 0.25)) if positive_gaps else None
        maximum_gap = (
            nominal_gap * float(maximum_gap_multiplier)
            if nominal_gap is not None
            else None
        )
        current: list[dict[str, Any]] = []
        previous_timestamp: int | None = None
        previous_frame: Mapping[str, Any] | None = None
        for row in rows:
            timestamp = tracking_physical_timestamp(row["frame"])
            if timestamp is None:
                if current:
                    episodes.append(current)
                    current = []
                episodes.append([row])
                previous_timestamp = None
                previous_frame = None
                continue
            translation_m, rotation_degrees = (
                _camera_motion(previous_frame, row["frame"])
                if previous_frame is not None
                else (None, None)
            )
            discontinuity = bool(current) and (
                len(current) >= int(maximum_episode_frames)
                or previous_timestamp is None
                or timestamp <= previous_timestamp
                or (
                    maximum_gap is not None
                    and timestamp - previous_timestamp > maximum_gap
                )
                or (
                    translation_m is not None
                    and translation_m > float(maximum_translation_m)
                )
                or (
                    rotation_degrees is not None
                    and rotation_degrees > float(maximum_rotation_degrees)
                )
            )
            if current and discontinuity:
                episodes.append(current)
                current = []
            current.append(row)
            previous_timestamp = timestamp
            previous_frame = row["frame"]
        if current:
            episodes.append(current)
    return episodes


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
    *,
    minimum_model_confidence: float,
    maximum_area_expansion: float,
    minimum_depth_padding_m: float,
    maximum_depth_padding_m: float,
    obb_audit_expansion: float,
    maximum_tracking_gap_multiplier: float,
    maximum_tracking_translation_m: float,
    maximum_tracking_rotation_degrees: float,
    maximum_tracking_episode_frames: int,
) -> tuple[list[dict], float, dict[str, Any]]:
    """Track only inside camera-pure, temporally continuous episodes.

    Cross-camera and cross-view-family association belongs to the 3D evidence
    merger. Feeding such jumps to a video tracker creates identity drift while
    looking like extra multiview support.
    """

    accepted_rows = [
        row
        for row in view_results
        if bool(row.get("accepted")) and str(row.get("mask_relative") or "")
    ]
    audit: dict[str, Any] = {
        "enabled": True,
        "seed_views": len(accepted_rows),
        "attempted_views": 0,
        "accepted_views": 0,
        "reason": None,
        "camera_stream_isolation": True,
        "maximum_area_expansion": float(maximum_area_expansion),
        "area_expansion_seed_reference": "projected_metric_guidance_seed",
        "physical_timestamp_source": "explicit field or source-image stem",
        "maximum_tracking_gap_multiplier": float(maximum_tracking_gap_multiplier),
        "maximum_tracking_translation_m": float(maximum_tracking_translation_m),
        "maximum_tracking_rotation_degrees": float(maximum_tracking_rotation_degrees),
        "maximum_tracking_episode_frames": int(maximum_tracking_episode_frames),
        "episodes": [],
    }
    if not accepted_rows:
        audit["reason"] = "no_independently_accepted_seed"
        return view_results, 0.0, audit

    existing_by_source = {
        str(row.get("source_image") or ""): row for row in view_results
    }
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
        depth_path = (frames_path.parent / str(frame["depth_path"])).resolve(
            strict=True
        )
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(rgb_path)
        entries.append(
            {
                "source_image": source_image,
                "rescue_image_id": rescue_image_id,
                "frame": frame,
                "rgb": cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
                "depth": np.asarray(
                    np.load(depth_path, mmap_mode="r"), dtype=np.float32
                ),
            }
        )
    episodes = partition_tracking_episodes(
        entries,
        maximum_gap_multiplier=float(maximum_tracking_gap_multiplier),
        maximum_translation_m=float(maximum_tracking_translation_m),
        maximum_rotation_degrees=float(maximum_tracking_rotation_degrees),
        maximum_episode_frames=int(maximum_tracking_episode_frames),
    )
    audit["stream_count"] = len(
        {row["tracking_stream_id"] for episode in episodes for row in episode}
    )
    audit["episode_count"] = len(episodes)
    total_seconds = 0.0

    for episode_index, episode in enumerate(episodes):
        stream_id = str(episode[0]["tracking_stream_id"])
        episode_sources = {str(row["source_image"]) for row in episode}
        seed_candidates = [
            row
            for row in accepted_rows
            if str(row.get("source_image") or "") in episode_sources
        ]
        episode_audit: dict[str, Any] = {
            "episode_index": int(episode_index),
            "stream_id": stream_id,
            "frames": len(episode),
            "first_physical_timestamp": tracking_physical_timestamp(
                episode[0]["frame"]
            ),
            "last_physical_timestamp": tracking_physical_timestamp(
                episode[-1]["frame"]
            ),
            "seed_views": len(seed_candidates),
            "attempted_views": 0,
            "accepted_views": 0,
            "reason": None,
        }
        audit["episodes"].append(episode_audit)
        if len(episode) < 2:
            episode_audit["reason"] = "single_frame_episode"
            continue
        if not seed_candidates:
            episode_audit["reason"] = "no_accepted_seed_in_episode"
            continue
        seed = max(
            seed_candidates,
            key=lambda row: (
                float(row.get("confidence") or 0.0),
                int(row.get("final_pixels") or 0),
            ),
        )
        seed_source = str(seed["source_image"] or "")
        seed_indices = [
            index
            for index, row in enumerate(episode)
            if row["source_image"] == seed_source
        ]
        if not seed_indices:
            episode_audit["reason"] = "accepted_seed_not_in_episode"
            continue
        seed_index = int(seed_indices[0])
        seed_mask = unpack_mask_archive(output / str(seed["mask_relative"]), "raw")
        video = [[Image.fromarray(np.ascontiguousarray(row["rgb"])) for row in episode]]

        def propagate_direction(reverse: bool) -> dict[int, Any]:
            session = processor.init_video_session(
                video=video,
                inference_device="cuda",
                inference_state_device="cuda",
                processing_device="cpu",
                video_storage_device="cuda",
                max_vision_features_cache_size=max(1, len(episode)),
                dtype=torch_module.bfloat16,
            )
            processor.add_inputs_to_inference_session(
                session, frame_idx=seed_index, obj_ids=1, input_masks=[seed_mask]
            )
            return {
                int(predicted.frame_idx): predicted
                for predicted in model.propagate_in_video_iterator(
                    session,
                    start_frame_idx=seed_index,
                    max_frame_num_to_track=len(episode),
                    reverse=reverse,
                )
            }

        torch_module.cuda.synchronize()
        started = time.perf_counter()
        with torch_module.inference_mode():
            outputs = propagate_direction(False)
            if seed_index > 0:
                outputs.update(propagate_direction(True))
        torch_module.cuda.synchronize()
        episode_seconds = time.perf_counter() - started
        total_seconds += episode_seconds
        episode_audit["seed_source_image"] = seed_source

        for frame_index, entry in enumerate(episode):
            source_image = str(entry["source_image"])
            prior = existing_by_source.get(source_image)
            if prior is not None and bool(prior.get("accepted")):
                continue
            predicted = outputs.get(frame_index)
            if predicted is None:
                continue
            audit["attempted_views"] += 1
            episode_audit["attempted_views"] += 1
            rgb = np.asarray(entry["rgb"], dtype=np.uint8)
            depth = np.asarray(entry["depth"], dtype=np.float32)
            frame = entry["frame"]
            sam, confidence = _postprocess_tracker_mask(
                torch_module, processor, predicted, rgb.shape[0], rgb.shape[1]
            )
            projected = project_metric_obb(
                association["target_center_m"],
                association["target_dimensions_m"],
                association["target_wxyz"],
                frame,
            )
            if projected is None:
                continue
            projected_seed, projected_seed_metrics = project_world_points_seed(
                object_voxels,
                frame,
                depth,
                dilation_pixels=1,
                minimum_projected_points=12,
            )
            multiview_seed, multiview_projection_metrics = project_world_points_seed(
                world_support,
                frame,
                depth,
                dilation_pixels=3,
                minimum_projected_points=12,
            )
            guidance_seed = projected_seed | multiview_seed
            if int(guidance_seed.sum()) < 24:
                continue
            final, depth_metrics = _depth_refine(
                sam,
                guidance_seed,
                depth,
                frame,
                association,
                preserve_seed=False,
                minimum_padding_m=minimum_depth_padding_m,
                maximum_padding_m=maximum_depth_padding_m,
                obb_expansion=obb_audit_expansion,
            )
            final_pixels = int(final.sum())
            seed_pixels = int(guidance_seed.sum())
            area_expansion_gate = bounded_area_expansion_gate(
                seed_pixels=seed_pixels,
                final_pixels=final_pixels,
                maximum_area_expansion=maximum_area_expansion,
            )
            seed_recall = (
                float(np.logical_and(final, guidance_seed).sum()) / seed_pixels
            )
            border = np.zeros_like(final)
            border[:3] = border[-3:] = True
            border[:, :3] = border[:, -3:] = True
            border_fraction = float(np.logical_and(final, border).sum()) / max(
                final_pixels, 1
            )
            shape_metrics = mask_shape_metrics(final)
            projected_box_metrics = mask_box_metrics(final, projected)
            quality_gates = {
                "tracker_confidence": confidence >= float(minimum_model_confidence),
                "propagated_metric_seed_coverage": seed_recall >= 0.70,
                "image_border": border_fraction <= 0.20,
                "depth_valid": depth_metrics["depth_valid_fraction"] >= 0.50,
                "depth_surface": depth_metrics["depth_inlier_fraction"] >= 0.55,
                "dominant_component": shape_metrics["dominant_component_fraction"]
                >= 0.92,
                "significant_components": shape_metrics["significant_component_count"]
                <= 2,
                "projected_obb_containment": projected_box_metrics["mask_inside_box"]
                >= 0.70,
                "projected_obb_support": (
                    projected_box_metrics["box_pixel_coverage"] >= 0.03
                    or projected_box_metrics["bbox_iou"] >= 0.08
                ),
                "image_area": final_pixels <= int(0.45 * final.size),
                "bounded_seed_area_expansion": bool(area_expansion_gate["passed"]),
            }
            accepted = _view_acceptance(final_pixels, quality_gates)
            global_image_id = source_image_count + int(entry["rescue_image_id"])
            visual_relative = Path("visuals") / (
                f"object_{int(association['object_id']):06d}__img_"
                f"{global_image_id:06d}__propagated.jpg"
            )
            _write_visual(output / visual_relative, rgb, guidance_seed, sam, final)
            row = {
                "source_image": source_image,
                "rescue_image_id": int(entry["rescue_image_id"]),
                "global_image_id": global_image_id,
                "physical_timestamp_ns": int(tracking_physical_timestamp(frame) or 0),
                "tracking_stream_id": stream_id,
                "tracking_episode_index": int(episode_index),
                "accepted": accepted,
                "seed_origin": "rest3d_bidirectional_propagation",
                "refinement_origin": "rest3d_bidirectional_propagation",
                "propagation_seed_source_image": seed_source,
                "confidence": confidence,
                "sam_pixels": int(sam.sum()),
                "final_pixels": final_pixels,
                "propagated_metric_seed_pixels": seed_pixels,
                "propagated_metric_seed_recall": seed_recall,
                "area_expansion": area_expansion_gate["area_expansion"],
                "area_expansion_gate": {
                    **area_expansion_gate,
                    "seed_reference": "projected_metric_guidance_seed",
                },
                "border_fraction": border_fraction,
                "rejection_reasons": [
                    key for key, passed in quality_gates.items() if not passed
                ],
                "quality_gates": quality_gates,
                "advisory_gates": {},
                "inference_seconds": episode_seconds / max(len(outputs), 1),
                "visual_relative": visual_relative.as_posix(),
                "projection_seed_metrics": projected_seed_metrics,
                "multiview_support_projection": multiview_projection_metrics,
                **shape_metrics,
                **{
                    f"projected_{key}": value
                    for key, value in projected_box_metrics.items()
                },
                **depth_metrics,
            }
            if prior is not None:
                row["independent_attempt"] = prior
            if accepted:
                relative = (
                    Path("masks")
                    / (f"object_{int(association['object_id']):06d}")
                    / f"img_{global_image_id:06d}_det_0000.npz"
                )
                row.update(_write_sidecar(output / relative, rgb, final))
                row["mask_relative"] = relative.as_posix()
                audit["accepted_views"] += 1
                episode_audit["accepted_views"] += 1
            if prior is None:
                view_results.append(row)
            else:
                view_results[view_results.index(prior)] = row
                existing_by_source[source_image] = row
        episode_audit["reason"] = "completed"

    audit["reason"] = "completed"
    return view_results, total_seconds, audit


def negative_prompt_regions(
    box_canvas: np.ndarray,
    guidance_seed: np.ndarray,
    competing_masks: np.ndarray,
    *,
    projected_canvas: np.ndarray | None = None,
    whole_object_expansion: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build negative prompt regions without suppressing wanted missing parts.

    A target explicitly selected by the VLM whole-object audit may grow into
    *unknown* pixels inside its prompt box. Pixels already claimed by another
    instance remain hard negatives in every mode; otherwise expansion can
    consume a neighbouring wall, cabinet, or object. The exterior ring is an
    additional background prior, not a replacement for instance exclusion.
    """

    box_mask = np.asarray(box_canvas, dtype=bool)
    seed_mask = np.asarray(guidance_seed, dtype=bool)
    other_mask = np.asarray(competing_masks, dtype=bool)
    if not (box_mask.shape == seed_mask.shape == other_mask.shape):
        raise ValueError("negative prompt masks must have identical shapes")
    if whole_object_expansion:
        exterior = (
            cv2.dilate(
                box_mask.astype(np.uint8),
                np.ones((31, 31), dtype=np.uint8),
                iterations=1,
            ).astype(bool)
            & ~box_mask
        )
        hard_negative = other_mask & (box_mask | exterior)
        return hard_negative, exterior & ~hard_negative
    if projected_canvas is not None:
        projected_mask = np.asarray(projected_canvas, dtype=bool)
        if projected_mask.shape != box_mask.shape:
            raise ValueError("projected prompt mask must match box canvas")
        negative_region = box_mask & ~projected_mask
        hard_negative = other_mask & box_mask
        return hard_negative, negative_region & ~hard_negative
    seed_dilated = cv2.dilate(
        seed_mask.astype(np.uint8),
        np.ones((41, 41), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    negative_region = box_mask & ~seed_dilated
    hard_negative = other_mask & box_mask
    return hard_negative, negative_region & ~hard_negative


def select_mask_candidate(
    raw: np.ndarray,
    refined: np.ndarray,
    metric_core: np.ndarray,
    projected_box: np.ndarray,
    competing_masks: np.ndarray,
    *,
    minimum_margin: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose one mask once; detector boxes remain prompts, never final geometry."""

    def evaluate(mask: np.ndarray) -> dict[str, float]:
        candidate = np.asarray(mask, dtype=bool)
        pixels = max(int(candidate.sum()), 1)
        core_pixels = int(np.asarray(metric_core, dtype=bool).sum())
        core_recall = (
            float(np.logical_and(candidate, metric_core).sum()) / core_pixels
            if core_pixels
            else 1.0
        )
        competing_fraction = (
            float(np.logical_and(candidate, competing_masks).sum()) / pixels
        )
        box = mask_box_metrics(candidate, projected_box)
        shape = mask_shape_metrics(candidate)
        # The current OBB is a weak prior only. Independent metric support,
        # instance exclusion, and topology dominate the A/B choice so a stale
        # box cannot validate a similarly stale mask.
        score = (
            0.40 * core_recall
            + 0.25 * (1.0 - min(1.0, competing_fraction))
            + 0.20 * float(shape["dominant_component_fraction"])
            + 0.10 * float(box["mask_inside_box"])
            + 0.05 * min(1.0, float(box["bbox_iou"]) / 0.50)
        )
        return {
            "score": float(score),
            "metric_core_recall": core_recall,
            "projected_obb_containment": float(box["mask_inside_box"]),
            "projected_bbox_iou": float(box["bbox_iou"]),
            "dominant_component_fraction": float(shape["dominant_component_fraction"]),
            "competing_mask_fraction": competing_fraction,
            "pixels": float(int(candidate.sum())),
        }

    raw_metrics = evaluate(raw)
    refined_metrics = evaluate(refined)
    use_refined = refined_metrics["score"] >= raw_metrics["score"] + float(
        minimum_margin
    )
    selected = np.asarray(refined if use_refined else raw, dtype=bool)
    return selected, {
        "selected": "sam3_refined" if use_refined else "yoloe_seed_preserved",
        "minimum_score_margin": float(minimum_margin),
        "score_margin": float(refined_metrics["score"] - raw_metrics["score"]),
        "raw": raw_metrics,
        "sam3_refined": refined_metrics,
    }


def _view_acceptance(final_pixels: int, hard_gates: Mapping[str, bool]) -> bool:
    """Accept a view only on safety gates; keep coverage evidence advisory."""

    return int(final_pixels) > 0 and all(bool(value) for value in hard_gates.values())


def bounded_area_expansion_gate(
    *,
    seed_pixels: int | float,
    final_pixels: int | float,
    maximum_area_expansion: float,
) -> dict[str, float | bool]:
    """Apply one inclusive, category-agnostic area bound to a metric seed."""

    seed = float(seed_pixels)
    final = float(final_pixels)
    maximum = float(maximum_area_expansion)
    if not math.isfinite(seed) or seed <= 0.0:
        raise ValueError("area-expansion seed pixels must be finite and positive")
    if not math.isfinite(final) or final < 0.0:
        raise ValueError("area-expansion final pixels must be finite and non-negative")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum area expansion must be finite and positive")
    expansion = final / seed
    return {
        "seed_pixels": seed,
        "final_pixels": final,
        "area_expansion": expansion,
        "maximum_area_expansion": maximum,
        "passed": expansion <= maximum,
    }


def build_concept_second_chance_targets(
    plan: Mapping[str, Any],
    source_state: Mapping[str, Any],
    rescue_state: Mapping[str, Any],
    rescue_frames_by_stem: Mapping[str, Mapping[str, Any]],
    prompts: Mapping[int, Mapping[str, Any]],
    *,
    minimum_views: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join explicit prompts to role-scoped planned frames without YOLOE tracks."""

    if minimum_views < 2:
        raise ValueError("concept second chance requires at least two views")
    source_ids = numpy_array(source_state["object_id"], np.int64).reshape(-1)
    source_index = {
        int(value): index for index, value in enumerate(source_ids.tolist())
    }
    centers = numpy_array(source_state["object_box_centers_m"], np.float64)
    dimensions = numpy_array(source_state["object_box_dimensions_m"], np.float64)
    quaternions = numpy_array(source_state["object_box_wxyz"], np.float64)
    image_by_source: dict[str, int] = {}
    rescue_images = list(rescue_state.get("images") or [])
    for image_id in range(len(rescue_images)):
        frame = frame_by_state_image(
            rescue_images, rescue_frames_by_stem, image_id
        )
        if frame is None:
            continue
        source_image = str(frame.get("source_image") or "")
        if source_image and source_image not in image_by_source:
            image_by_source[source_image] = image_id

    plan_rows = {
        int(row["object_id"]): row
        for row in plan.get("objects") or []
        if isinstance(row, Mapping) and row.get("object_id") is not None
    }
    targets: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for object_id, prompt_row in sorted(prompts.items()):
        object_row = plan_rows.get(int(object_id))
        index = source_index.get(int(object_id))
        reasons: list[str] = []
        if object_row is None:
            reasons.append("object_not_in_active_role_plan")
        if index is None:
            reasons.append("object_missing_from_source_state")
        matched: list[dict[str, Any]] = []
        timestamps: set[str] = set()
        if object_row is not None:
            for selected in object_row.get("selected_views") or []:
                if not isinstance(selected, Mapping):
                    continue
                source_image = str(selected.get("name") or "")
                image_id = image_by_source.get(source_image)
                physical_timestamp = str(
                    selected.get("physical_timestamp") or ""
                )
                if (
                    image_id is None
                    or not physical_timestamp
                    or physical_timestamp in timestamps
                ):
                    continue
                matched.append(
                    {
                        "rescue_image_id": int(image_id),
                        "source_image": source_image,
                        "physical_timestamp": physical_timestamp,
                        "seed_origin": "sam3_text_concept_second_chance",
                    }
                )
                timestamps.add(physical_timestamp)
        if len(timestamps) < int(minimum_views):
            reasons.append("insufficient_independent_planned_frames")
        if reasons:
            rejections.append(
                {
                    "object_id": int(object_id),
                    "status": "rejected_before_concept_inference",
                    "rejection_reasons": reasons,
                    "planned_frames": len(matched),
                    "independent_physical_timestamps": len(timestamps),
                }
            )
            continue
        assert index is not None
        targets.append(
            {
                "object_id": int(object_id),
                "source_state_index": int(index),
                "target_center_m": centers[index].tolist(),
                "target_dimensions_m": dimensions[index].tolist(),
                "target_wxyz": quaternions[index].tolist(),
                "association_mode": "sam3_text_concept_second_chance",
                "identity_anchor": {
                    "type": "explicit_concept_plus_projected_metric_hit",
                    "prompt": str(prompt_row["prompt"]),
                    "provenance": dict(prompt_row["provenance"]),
                    "authorization": str(prompt_row["authorization"]),
                    "identity_anchor_authorized": bool(
                        prompt_row["identity_anchor_authorized"]
                    ),
                    "bound_semantic_evidence": prompt_row.get(
                        "bound_semantic_evidence"
                    ),
                },
                "matched_views": matched,
            }
        )
    return targets, rejections


def _concept_competing_masks(
    rescue_state: Mapping[str, Any],
    rescue_root: Path,
    image_id: int,
    candidate: np.ndarray,
    identity_anchor: np.ndarray,
    depth: np.ndarray,
    *,
    minimum_anchor_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exclude at most one unambiguous metric-compatible mapping mask."""

    masks = [
        mask
        for mask in _all_masks_for_image(dict(rescue_state), rescue_root, image_id)
        if mask.shape == candidate.shape
    ]
    excluded_index, identity_audit = (
        select_unique_identity_compatible_mapping_mask(
            candidate,
            identity_anchor,
            masks,
            depth,
            minimum_anchor_pixels=int(minimum_anchor_pixels),
        )
    )
    competing = np.zeros_like(candidate, dtype=bool)
    for index, mask in enumerate(masks):
        if index != excluded_index:
            competing |= mask
    return competing, {
        "mapping_masks": len(masks),
        "excluded_identity_anchor_mask": excluded_index is not None,
        "excluded_mapping_mask_index": excluded_index,
        "excluded_mask_is_geometry_reference": excluded_index is not None,
        "identity_compatibility": identity_audit,
        "policy": "exclude_at_most_one_unique_metric_compatible_mapping_mask",
    }


def _object_acceptance(
    view_rows: list[Mapping[str, Any]],
    support_audit: Mapping[str, Any],
    *,
    minimum_views: int,
    minimum_independent_views: int = 1,
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
    accepted_timestamps = {
        str(row.get("physical_timestamp_ns") or f"image:{row.get('global_image_id')}")
        for row in view_rows
        if bool(row.get("accepted")) and row.get("global_image_id") is not None
    }
    independent_timestamps = {
        str(row.get("physical_timestamp_ns") or f"image:{row.get('global_image_id')}")
        for row in view_rows
        if bool(row.get("accepted"))
        and row.get("global_image_id") is not None
        and str(row.get("refinement_origin") or "independent_sam3")
        != "rest3d_bidirectional_propagation"
    }
    view_gate = len(accepted_timestamps) >= int(minimum_views)
    seed_gate = len(independent_timestamps) >= int(minimum_independent_views)
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
    return (
        accepted,
        status,
        {
            "accepted_refined_views": len(accepted_ids),
            "independent_refined_views": len(independent_ids),
            "propagated_refined_views": len(propagated_ids),
            "accepted_physical_timestamps": len(accepted_timestamps),
            "independent_physical_timestamps": len(independent_timestamps),
            "independent_seed_gate": seed_gate,
            "minimum_refined_views": int(minimum_views),
            "minimum_independent_views": int(minimum_independent_views),
            "metric_support_points": support_points,
            "minimum_metric_support_points": int(minimum_support_points),
            "refined_view_gate": view_gate,
            "metric_support_gate": support_gate,
        },
    )


def _run_concept_second_chance(
    torch_module,
    processor,
    model,
    targets: list[dict[str, Any]],
    object_results: list[dict[str, Any]],
    source_state: Mapping[str, Any],
    rescue_state: Mapping[str, Any],
    rescue_frames_by_stem: Mapping[str, Mapping[str, Any]],
    rescue_root: Path,
    frames_path: Path,
    output: Path,
    *,
    source_image_count: int,
    minimum_accepted_views: int,
    minimum_independent_views: int,
    second_chance_minimum_physical_timestamps: int,
    minimum_metric_support_points: int,
    minimum_model_confidence: float,
    minimum_depth_padding_m: float,
    maximum_depth_padding_m: float,
    obb_audit_expansion: float,
    maximum_competing_mask_fraction: float,
    maximum_area_expansion: float,
    minimum_anchor_pixels: int,
    minimum_identity_surface_fraction: float,
    minimum_cycle_recall: float,
    maximum_centroid_distance_m: float,
    consensus_voxel_size_m: float,
    maximum_candidates_per_view: int,
    maximum_exact_combinations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Run text proposals only for explicit, still-rejected prompt targets."""

    by_object = {
        int(row["object_id"]): row
        for row in object_results
        if row.get("object_id") is not None
    }
    rescue_images = list(rescue_state.get("images") or [])
    audits: list[dict[str, Any]] = []
    inference_seconds = 0.0
    for target in targets:
        object_id = int(target["object_id"])
        existing = by_object.get(object_id)
        if existing is not None and bool(existing.get("accepted")):
            audits.append(
                {
                    "object_id": object_id,
                    "status": "skipped_primary_object_already_accepted",
                    "accepted": False,
                    "model_invoked": False,
                }
            )
            continue
        object_voxels = _object_voxel_points(
            dict(source_state), int(target["source_state_index"])
        )
        candidates: list[dict[str, Any]] = []
        frame_rejections: list[dict[str, Any]] = []
        for view in target["matched_views"]:
            image_id = int(view["rescue_image_id"])
            frame = frame_by_state_image(
                rescue_images, rescue_frames_by_stem, image_id
            )
            if frame is None:
                frame_rejections.append(
                    {
                        "source_image": str(view.get("source_image") or ""),
                        "reason": "prepared_frame_not_resolved",
                    }
                )
                continue
            rgb_path = (frames_path.parent / str(frame["rgb_path"])).resolve(
                strict=True
            )
            depth_path = (frames_path.parent / str(frame["depth_path"])).resolve(
                strict=True
            )
            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(rgb_path)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth = np.asarray(np.load(depth_path, mmap_mode="r"), dtype=np.float32)
            identity_anchor, anchor_projection = project_world_points_seed(
                object_voxels,
                frame,
                depth,
                dilation_pixels=5,
                minimum_projected_points=8,
            )
            if int(identity_anchor.sum()) < int(minimum_anchor_pixels):
                frame_rejections.append(
                    {
                        "source_image": str(frame.get("source_image") or ""),
                        "physical_timestamp": str(
                            view.get("physical_timestamp") or ""
                        ),
                        "reason": "projected_metric_identity_anchor_insufficient",
                        "identity_anchor_pixels": int(identity_anchor.sum()),
                        "projection": anchor_projection,
                    }
                )
                continue
            proposals, seconds = _infer_sam3_concept(
                torch_module,
                processor,
                model,
                rgb,
                str(target["identity_anchor"]["prompt"]),
            )
            inference_seconds += seconds
            if not proposals:
                frame_rejections.append(
                    {
                        "source_image": str(frame.get("source_image") or ""),
                        "physical_timestamp": str(
                            view.get("physical_timestamp") or ""
                        ),
                        "reason": "sam3_concept_returned_no_proposals",
                    }
                )
                continue
            per_view: list[dict[str, Any]] = []
            for proposal_index, proposal in enumerate(proposals):
                raw = np.asarray(proposal["mask"], dtype=bool)
                if raw.shape != depth.shape:
                    frame_rejections.append(
                        {
                            "source_image": str(frame.get("source_image") or ""),
                            "reason": "sam3_concept_mask_shape_mismatch",
                            "proposal_index": int(proposal_index),
                        }
                    )
                    continue
                final, depth_metrics = _depth_refine(
                    raw,
                    raw,
                    depth,
                    frame,
                    target,
                    preserve_seed=False,
                    minimum_padding_m=float(minimum_depth_padding_m),
                    maximum_padding_m=float(maximum_depth_padding_m),
                    obb_expansion=float(obb_audit_expansion),
                )
                competing, competition_audit = _concept_competing_masks(
                    rescue_state,
                    rescue_root,
                    image_id,
                    final,
                    identity_anchor,
                    depth,
                    minimum_anchor_pixels=int(minimum_anchor_pixels),
                )
                local, world_points = evaluate_second_chance_candidate(
                    final,
                    identity_anchor,
                    competing,
                    depth,
                    frame,
                    identity_world_points=object_voxels,
                    confidence=float(proposal["confidence"]),
                    minimum_confidence=float(minimum_model_confidence),
                    maximum_competing_fraction=float(
                        maximum_competing_mask_fraction
                    ),
                    minimum_anchor_pixels=int(minimum_anchor_pixels),
                    minimum_metric_points=24,
                    minimum_identity_surface_fraction=float(
                        minimum_identity_surface_fraction
                    ),
                )
                concept_pixels = max(int(raw.sum()), 1)
                final_pixels = int(final.sum())
                area_gate = bounded_area_expansion_gate(
                    seed_pixels=concept_pixels,
                    final_pixels=final_pixels,
                    maximum_area_expansion=float(maximum_area_expansion),
                )
                area_retention = float(final_pixels / concept_pixels)
                extra_gates = {
                    "prompt_provenance_authorized": bool(
                        target["identity_anchor"]["identity_anchor_authorized"]
                    ),
                    "depth_surface": float(depth_metrics["depth_inlier_fraction"])
                    >= 0.55,
                    "bounded_concept_area_expansion": bool(area_gate["passed"]),
                    "concept_depth_area_retention": area_retention >= 0.35,
                }
                local["quality_gates"].update(extra_gates)
                local["passed_local_gates"] = all(
                    bool(value) for value in local["quality_gates"].values()
                )
                local["rejection_reasons"] = [
                    key
                    for key, passed in local["quality_gates"].items()
                    if not bool(passed)
                ]
                candidate_key = (
                    f"object:{object_id}:image:{image_id}:proposal:{proposal_index}"
                )
                per_view.append(
                    {
                        "candidate_key": candidate_key,
                        "object_id": object_id,
                        "source_image": str(frame.get("source_image") or ""),
                        "rescue_image_id": image_id,
                        "global_image_id": int(source_image_count + image_id),
                        "physical_timestamp": str(
                            view.get("physical_timestamp") or ""
                        ),
                        "concept_object_id": int(proposal["concept_object_id"]),
                        "concept_box_xyxy": list(proposal["box_xyxy"]),
                        "concept_mask": raw,
                        "mask": final,
                        "identity_anchor_mask": identity_anchor,
                        "frame": frame,
                        "depth": depth,
                        "rgb": rgb,
                        "world_points": world_points,
                        "passed_local_gates": bool(local["passed_local_gates"]),
                        "local_score": float(local["local_score"]),
                        "local_audit": local,
                        "depth_metrics": depth_metrics,
                        "area_expansion_gate": {
                            **area_gate,
                            "seed_reference": "sam3_text_concept_mask",
                        },
                        "concept_depth_area_retention": area_retention,
                        "identity_anchor_projection": anchor_projection,
                        "competition_audit": competition_audit,
                    }
                )
            per_view.sort(
                key=lambda row: (
                    bool(row["passed_local_gates"]),
                    float(row["local_score"]),
                    str(row["candidate_key"]),
                ),
                reverse=True,
            )
            candidates.extend(per_view[: int(maximum_candidates_per_view)])

        selected, consistency = select_cross_view_consistent_candidates(
            candidates,
            minimum_physical_timestamps=int(
                second_chance_minimum_physical_timestamps
            ),
            minimum_consensus_points=int(minimum_metric_support_points),
            minimum_cycle_recall=float(minimum_cycle_recall),
            maximum_centroid_distance_m=float(maximum_centroid_distance_m),
            consensus_voxel_size_m=float(consensus_voxel_size_m),
            maximum_exact_combinations=int(maximum_exact_combinations),
        )
        consistency.pop("support", None)
        primary_views = list(existing.get("views") or []) if existing is not None else []
        primary_accepted_sources = {
            str(row.get("source_image") or "")
            for row in primary_views
            if bool(row.get("accepted"))
        }
        second_chance_views: list[dict[str, Any]] = []
        for candidate in candidates:
            key = str(candidate["candidate_key"])
            cluster_selected = key in selected
            source_image = str(candidate["source_image"])
            already_primary = source_image in primary_accepted_sources
            accepted = cluster_selected and not already_primary
            local_audit = dict(candidate["local_audit"])
            rejection_reasons = list(local_audit["rejection_reasons"])
            if bool(candidate["passed_local_gates"]) and not cluster_selected:
                rejection_reasons.append("cross_view_3d_cycle_consistency")
            if cluster_selected and already_primary:
                rejection_reasons.append("primary_view_already_accepted")
            visual_relative = (
                Path("visuals")
                / (
                    f"object_{object_id:06d}__img_"
                    f"{int(candidate['global_image_id']):06d}__concept_"
                    f"{int(candidate['concept_object_id']):04d}.jpg"
                )
            )
            _write_visual(
                output / visual_relative,
                candidate["rgb"],
                candidate["identity_anchor_mask"],
                candidate["concept_mask"],
                candidate["mask"],
            )
            row = {
                "source_image": source_image,
                "rescue_image_id": int(candidate["rescue_image_id"]),
                "global_image_id": int(candidate["global_image_id"]),
                "physical_timestamp_ns": str(candidate["physical_timestamp"]),
                "accepted": accepted,
                "refinement_origin": "independent_sam3_text_concept_second_chance",
                "seed_origin": "sam3_text_concept_second_chance",
                "candidate_key": key,
                "cluster_selected": cluster_selected,
                "primary_view_already_accepted": already_primary,
                "concept_object_id": int(candidate["concept_object_id"]),
                "concept_box_xyxy": list(candidate["concept_box_xyxy"]),
                "confidence": float(local_audit["confidence"]),
                "final_pixels": int(local_audit["candidate_pixels"]),
                "identity_anchor_pixels": int(
                    local_audit["identity_anchor_pixels"]
                ),
                "identity_anchor_fraction_of_candidate": float(
                    local_audit["identity_anchor_fraction_of_candidate"]
                ),
                "legacy_anchor_coverage_advisory": float(
                    local_audit["legacy_anchor_coverage_advisory"]
                ),
                "legacy_anchor_coverage_is_hard_gate": False,
                "other_mask_fraction": float(
                    local_audit["competing_mask_fraction"]
                ),
                "quality_gates": dict(local_audit["quality_gates"]),
                "rejection_reasons": rejection_reasons,
                "area_expansion_gate": candidate["area_expansion_gate"],
                "concept_depth_area_retention": float(
                    candidate["concept_depth_area_retention"]
                ),
                "identity_anchor_projection": candidate[
                    "identity_anchor_projection"
                ],
                "competition_audit": candidate["competition_audit"],
                "visual_relative": visual_relative.as_posix(),
                **{
                    key: value
                    for key, value in local_audit.items()
                    if key
                    in {
                        "depth_valid_fraction",
                        "image_area_fraction",
                        "border_fraction",
                        "metric_points",
                        "metric_identity_surface_points",
                        "metric_identity_surface_fraction",
                        "component_count",
                        "significant_component_count",
                        "dominant_component_fraction",
                        "bbox_fill_fraction",
                    }
                },
                **candidate["depth_metrics"],
            }
            if accepted:
                relative = (
                    Path("masks")
                    / f"object_{object_id:06d}"
                    / f"img_{int(candidate['global_image_id']):06d}_det_0001.npz"
                )
                sidecar = _write_sidecar(
                    output / relative, candidate["rgb"], candidate["mask"]
                )
                sidecar["mask_relative"] = relative.as_posix()
                row.update(sidecar)
            second_chance_views.append(row)

        support_audit = {
            "origin": "sam3_text_concept_cross_view_cycle_consensus",
            "contributing_views": int(
                consistency.get("independent_physical_timestamps") or 0
            ),
            "support_points": int(consistency.get("support_points") or 0),
            "voxel_consensus": consistency.get("voxel_consensus"),
        }
        combined_views = primary_views + second_chance_views
        accepted, status, object_gates = _object_acceptance(
            combined_views,
            support_audit,
            minimum_views=int(minimum_accepted_views),
            minimum_independent_views=int(minimum_independent_views),
            minimum_support_points=int(minimum_metric_support_points),
        )
        second_chance_audit = {
            "object_id": object_id,
            "status": status,
            "accepted": accepted,
            "model_invoked": True,
            "identity_anchor": target["identity_anchor"],
            "legacy_mask_geometry_coverage_required": False,
            "local_candidate_count": len(candidates),
            "frame_rejections": frame_rejections,
            "cross_view_consistency": consistency,
            "newly_accepted_views": sum(
                bool(row.get("accepted")) for row in second_chance_views
            ),
        }
        audits.append(second_chance_audit)
        if existing is None:
            existing = {
                **{
                    key: value
                    for key, value in target.items()
                    if key != "matched_views"
                },
                "primary_status": "not_associated_by_primary_route",
                "views": [],
                "bidirectional_propagation": {
                    "enabled": False,
                    "reason": "concept_second_chance_only",
                    "seed_views": 0,
                    "attempted_views": 0,
                    "accepted_views": 0,
                },
            }
            object_results.append(existing)
            by_object[object_id] = existing
        else:
            existing.setdefault("primary_status", str(existing.get("status") or ""))
            existing.setdefault(
                "primary_multiview_support", existing.get("multiview_support")
            )
        existing.update(
            {
                "status": status,
                "accepted": accepted,
                "accepted_views": int(
                    len(
                        {
                            int(row["global_image_id"])
                            for row in combined_views
                            if bool(row.get("accepted"))
                            and row.get("global_image_id") is not None
                        }
                    )
                ),
                "association_mode": "sam3_text_concept_second_chance",
                "multiview_support": support_audit,
                "object_quality_gates": object_gates,
                "second_chance": second_chance_audit,
                "views": combined_views,
            }
        )
    object_results.sort(key=lambda row: int(row["object_id"]))
    return object_results, audits, inference_seconds


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--view-role",
        choices=("train", "heldout-reference"),
        default="train",
        help=(
            "Use train views for fitting or frozen heldout views for evaluation-only reference masks."
        ),
    )
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
    parser.add_argument("--minimum-accepted-views", type=int, default=3)
    parser.add_argument("--minimum-independent-views", type=int, default=2)
    parser.add_argument("--minimum-metric-support-points", type=int, default=48)
    parser.add_argument("--minimum-model-confidence", type=float, default=0.70)
    parser.add_argument("--minimum-depth-padding-m", type=float, default=0.04)
    parser.add_argument("--maximum-depth-padding-m", type=float, default=0.18)
    parser.add_argument("--obb-audit-expansion", type=float, default=1.15)
    parser.add_argument("--maximum-competing-mask-fraction", type=float, default=0.15)
    parser.add_argument(
        "--maximum-tracking-gap-multiplier",
        type=float,
        default=2.5,
        help=(
            "Split a camera/view-family episode when its physical timestamp gap "
            "exceeds this multiple of that stream's lower-quartile positive source gap."
        ),
    )
    parser.add_argument(
        "--maximum-tracking-translation-m",
        type=float,
        default=0.50,
        help="Split an episode when consecutive camera centres move farther.",
    )
    parser.add_argument(
        "--maximum-tracking-rotation-degrees",
        type=float,
        default=21.0,
        help="Split an episode when consecutive camera orientations rotate farther.",
    )
    parser.add_argument(
        "--maximum-tracking-episode-frames",
        type=int,
        default=32,
        help="Bound tracker state, latency and drift in every camera-pure episode.",
    )
    parser.add_argument("--box-expansion", type=float, default=0.12)
    parser.add_argument("--maximum-area-expansion", type=float, default=3.0)
    parser.add_argument(
        "--whole-object-expansion-id-file",
        type=Path,
        help=(
            "Exact newline-delimited IDs whose VLM evidence requests whole-object "
            "completion; unknown interior pixels may grow but competing instances "
            "remain hard negative prompts."
        ),
    )
    parser.add_argument(
        "--second-chance-concept-prompts-json",
        type=Path,
        help=(
            "Default-off explicit Qwen/VLM concept prompts. Enabling this loads "
            "the full SAM3 text model only after the unchanged primary pass."
        ),
    )
    parser.add_argument(
        "--second-chance-minimum-physical-timestamps", type=int, default=2
    )
    parser.add_argument(
        "--second-chance-minimum-anchor-pixels", type=int, default=3
    )
    parser.add_argument(
        "--second-chance-minimum-identity-surface-fraction",
        type=float,
        default=0.20,
        help="Minimum fraction of candidate metric points near source identity voxels.",
    )
    parser.add_argument(
        "--second-chance-minimum-cycle-recall", type=float, default=0.45
    )
    parser.add_argument(
        "--second-chance-maximum-centroid-distance-m", type=float, default=0.45
    )
    parser.add_argument(
        "--second-chance-consensus-voxel-size-m", type=float, default=0.04
    )
    parser.add_argument(
        "--second-chance-maximum-exact-combinations", type=int, default=4096
    )
    parser.add_argument(
        "--second-chance-maximum-candidates-per-view", type=int, default=3
    )
    parser.add_argument(
        "--bidirectional-propagation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="REST3D-style forward/backward propagation from the best independently accepted seed.",
    )
    args = parser.parse_args()
    if int(args.minimum_accepted_views) < int(args.minimum_independent_views):
        raise ValueError("minimum accepted views must be >= minimum independent views")
    if int(args.minimum_independent_views) < 1:
        raise ValueError("minimum independent views must be positive")
    if int(args.minimum_metric_support_points) < 24:
        raise ValueError("minimum metric support points must be at least 24")
    if not 0.0 <= float(args.minimum_model_confidence) <= 1.0:
        raise ValueError("minimum model confidence must be in [0, 1]")
    if not 0.0 <= float(args.maximum_competing_mask_fraction) <= 1.0:
        raise ValueError("maximum competing mask fraction must be in [0, 1]")
    if (
        not math.isfinite(float(args.maximum_area_expansion))
        or float(args.maximum_area_expansion) <= 0.0
    ):
        raise ValueError("maximum area expansion must be finite and positive")
    if (
        not math.isfinite(float(args.maximum_tracking_gap_multiplier))
        or float(args.maximum_tracking_gap_multiplier) < 1.0
    ):
        raise ValueError("maximum tracking gap multiplier must be at least one")
    if (
        not math.isfinite(float(args.maximum_tracking_translation_m))
        or float(args.maximum_tracking_translation_m) <= 0.0
    ):
        raise ValueError("maximum tracking translation must be finite and positive")
    if (
        not math.isfinite(float(args.maximum_tracking_rotation_degrees))
        or not 0.0 < float(args.maximum_tracking_rotation_degrees) <= 180.0
    ):
        raise ValueError("maximum tracking rotation must be in (0, 180]")
    if int(args.maximum_tracking_episode_frames) < 2:
        raise ValueError("maximum tracking episode frames must be at least two")
    if int(args.second_chance_minimum_physical_timestamps) < 2:
        raise ValueError(
            "second chance requires at least two physical timestamps"
        )
    if int(args.second_chance_minimum_anchor_pixels) < 1:
        raise ValueError("second chance minimum anchor pixels must be positive")
    if not 0.0 <= float(
        args.second_chance_minimum_identity_surface_fraction
    ) <= 1.0:
        raise ValueError(
            "second chance identity surface fraction must be in [0, 1]"
        )
    if not 0.0 <= float(args.second_chance_minimum_cycle_recall) <= 1.0:
        raise ValueError("second chance cycle recall must be in [0, 1]")
    if (
        not math.isfinite(float(args.second_chance_maximum_centroid_distance_m))
        or float(args.second_chance_maximum_centroid_distance_m) <= 0.0
    ):
        raise ValueError("second chance centroid distance must be positive")
    if (
        not math.isfinite(float(args.second_chance_consensus_voxel_size_m))
        or float(args.second_chance_consensus_voxel_size_m) <= 0.0
    ):
        raise ValueError("second chance consensus voxel size must be positive")
    if not 1 <= int(args.second_chance_maximum_candidates_per_view) <= 3:
        raise ValueError("second chance candidates per view must be in [1, 3]")
    if not 1 <= int(args.second_chance_maximum_exact_combinations) <= 65536:
        raise ValueError(
            "second chance exact combination cap must be in [1, 65536]"
        )
    if (
        args.view_role == HELDOUT_VIEW_ROLE
        and args.whole_object_expansion_id_file is not None
    ):
        raise ValueError(
            "heldout-reference forbids fit-derived whole-object expansion allowlists"
        )
    plan_path = args.plan.expanduser().resolve(strict=True)
    raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frames_path = args.rescue_frames_json.expanduser().resolve(strict=True)
    frames_payload, rescue_frames, frame_by_stem = load_frames(frames_path)
    plan, input_view_contract = prepare_plan_for_view_role(
        raw_plan, frames_payload, str(args.view_role)
    )
    validate_metric_contract(plan, frames_payload)
    concept_prompts_path = (
        args.second_chance_concept_prompts_json.expanduser().resolve(strict=True)
        if args.second_chance_concept_prompts_json is not None
        else None
    )
    concept_prompts = (
        load_concept_prompt_manifest(concept_prompts_path)
        if concept_prompts_path is not None
        else {}
    )
    concept_plan_binding = (
        validate_acceptance_prompt_plan_binding(
            raw_plan, concept_prompts, concept_prompts_path,
            view_role=str(args.view_role),
        )
        if concept_prompts_path is not None
        else None
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty rescue output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir()
    (output / "visuals").mkdir()
    started = time.perf_counter()
    source_state = load_state(args.source_state.expanduser().resolve(strict=True))
    rescue_state = load_state(args.rescue_state.expanduser().resolve(strict=True))
    rescue_root = args.rescue_mask_root.expanduser().resolve(strict=True)
    whole_object_expansion_ids = load_object_id_file(
        args.whole_object_expansion_id_file
    )
    concept_targets, concept_target_rejections = build_concept_second_chance_targets(
        plan,
        source_state,
        rescue_state,
        frame_by_stem,
        concept_prompts,
        minimum_views=int(args.second_chance_minimum_physical_timestamps),
    )
    associations = associate_tracks(
        plan,
        source_state,
        rescue_state,
        frame_by_stem,
        rescue_root,
        minimum_views=int(args.minimum_association_views),
        maximum_normalized_center_distance=float(
            args.maximum_normalized_center_distance
        ),
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
        view_split=str(input_view_contract["active_split"]),
    )
    model_path = args.model.expanduser().resolve(strict=True)
    model_load_seconds = 0.0
    inference_seconds = 0.0
    if associations:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 rescue requires CUDA; CPU fallback is forbidden")
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        processor = Sam3TrackerVideoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = (
            Sam3TrackerVideoModel.from_pretrained(
                str(model_path), local_files_only=True, dtype=torch.bfloat16
            )
            .eval()
            .to("cuda")
        )
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
    else:
        processor = model = None

    source_image_count = len(source_state.get("images") or [])
    object_results: list[dict] = []
    for association in associations:
        object_id = int(association["object_id"])
        whole_object_expansion = object_id in whole_object_expansion_ids
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
            rgb_path = (frames_path.parent / str(frame["rgb_path"])).resolve(
                strict=True
            )
            depth_path = (frames_path.parent / str(frame["depth_path"])).resolve(
                strict=True
            )
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
                    object_voxels,
                    frame,
                    depth,
                    dilation_pixels=5,
                    minimum_projected_points=12,
                )
                seed = raw
                if int(seed.sum()) < 24:
                    view_results.append(
                        {
                            "source_image": str(frame.get("source_image") or ""),
                            "rescue_image_id": rescue_image_id,
                            "global_image_id": source_image_count + rescue_image_id,
                            "accepted": False,
                            "seed_origin": seed_origin,
                            "rejection_reasons": ["projected_voxel_seed_insufficient"],
                            "projection_seed_metrics": projection_seed_metrics,
                        }
                    )
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
            box = _prompt_box(
                raw | multiview_seed, projected, float(args.box_expansion)
            )
            positives = distributed_points(guidance_seed, 12)
            other = np.zeros_like(raw)
            for candidate in _all_masks_for_image(
                rescue_state, rescue_root, rescue_image_id
            ):
                if candidate.shape == raw.shape and not np.array_equal(candidate, raw):
                    other |= candidate
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            box_canvas = np.zeros_like(raw)
            box_canvas[y0:y1, x0:x1] = True
            if projected_seed:
                projected_clipped = clip_bbox(projected, raw.shape[1], raw.shape[0])
                if projected_clipped is None:
                    continue
                px0, py0, px1, py1 = [int(round(value)) for value in projected_clipped]
                projected_canvas = np.zeros_like(raw)
                projected_canvas[py0:py1, px0:px1] = True
                projected_canvas = cv2.dilate(
                    projected_canvas.astype(np.uint8),
                    np.ones((17, 17), np.uint8),
                    iterations=1,
                ).astype(bool)
            else:
                projected_canvas = None
            hard_negative, ring = negative_prompt_regions(
                box_canvas,
                guidance_seed,
                other,
                projected_canvas=projected_canvas,
                whole_object_expansion=whole_object_expansion,
            )
            negatives = (
                distributed_points(hard_negative, 6) + distributed_points(ring, 4)
            )[:10]
            points = positives + negatives
            labels = [1] * len(positives) + [0] * len(negatives)
            if len(positives) < 3:
                continue
            sam, confidence, seconds = _infer_sam3(
                torch, processor, model, rgb, points, labels, box
            )
            inference_seconds += seconds
            core_guided_shrink = (
                not projected_seed
                and int(multiview_support_audit.get("contributing_views") or 0) >= 2
                and int(multiview_seed.sum()) >= 24
            )
            depth_seed = guidance_seed if core_guided_shrink else raw
            candidate_final, depth_metrics = _depth_refine(
                sam,
                depth_seed,
                depth,
                frame,
                association,
                preserve_seed=not projected_seed and not core_guided_shrink,
                minimum_padding_m=float(args.minimum_depth_padding_m),
                maximum_padding_m=float(args.maximum_depth_padding_m),
                obb_expansion=float(args.obb_audit_expansion),
            )
            if core_guided_shrink:
                final, candidate_selection = select_mask_candidate(
                    raw, candidate_final, multiview_seed, projected, other
                )
            else:
                final = candidate_final
                candidate_selection = {
                    "selected": "sam3_refined",
                    "reason": "legacy_fail_closed_without_independent_metric_core",
                }
            raw_pixels = max(int(raw.sum()), 1)
            final_pixels = int(final.sum())
            raw_recall = float(np.logical_and(final, raw).sum()) / raw_pixels
            multiview_seed_pixels = int(multiview_seed.sum())
            multiview_seed_recall = (
                float(np.logical_and(final, multiview_seed).sum())
                / multiview_seed_pixels
                if multiview_seed_pixels
                else 1.0
            )
            positive_inclusion = float(
                np.mean([final[int(y), int(x)] for x, y in positives])
            )
            negative_inclusion = (
                float(np.mean([final[int(y), int(x)] for x, y in negatives]))
                if negatives
                else 0.0
            )
            other_fraction = float(np.logical_and(final, other).sum()) / max(
                final_pixels, 1
            )
            area_expansion_gate = bounded_area_expansion_gate(
                seed_pixels=raw_pixels,
                final_pixels=final_pixels,
                maximum_area_expansion=float(args.maximum_area_expansion),
            )
            area_expansion = float(area_expansion_gate["area_expansion"])
            border = np.zeros_like(final)
            border[:3] = border[-3:] = True
            border[:, :3] = border[:, -3:] = True
            border_fraction = float(np.logical_and(final, border).sum()) / max(
                final_pixels, 1
            )
            shape_metrics = mask_shape_metrics(final)
            projected_box_metrics = mask_box_metrics(final, projected)
            hard_gates = {
                "sam3_confidence": confidence >= float(args.minimum_model_confidence),
                "positive_inclusion": positive_inclusion >= 0.80,
                "negative_exclusion": negative_inclusion <= 0.35,
                "image_border": border_fraction <= 0.20,
                "depth_valid": depth_metrics["depth_valid_fraction"] >= 0.50,
                "depth_surface": depth_metrics["depth_inlier_fraction"] >= 0.55,
                "dominant_component": shape_metrics["dominant_component_fraction"]
                >= 0.92,
                "significant_components": shape_metrics["significant_component_count"]
                <= 2,
                "competing_mask_exclusion": other_fraction
                <= float(args.maximum_competing_mask_fraction),
                "bounded_seed_area_expansion": bool(area_expansion_gate["passed"]),
            }
            advisory_gates = {
                "multiview_support_recall": multiview_seed_recall >= 0.80,
            }
            if projected_seed:
                origin_gates = {
                    "projected_seed_coverage": raw_recall >= 0.70,
                    "projected_obb_containment": projected_box_metrics[
                        "mask_inside_box"
                    ]
                    >= 0.70,
                    "projected_obb_support": (
                        projected_box_metrics["box_pixel_coverage"] >= 0.03
                        or projected_box_metrics["bbox_iou"] >= 0.08
                    ),
                    "image_area": final_pixels <= int(0.45 * final.size),
                }
            else:
                if core_guided_shrink:
                    selected_metrics = candidate_selection[
                        (
                            "sam3_refined"
                            if candidate_selection["selected"] == "sam3_refined"
                            else "raw"
                        )
                    ]
                    origin_gates = {
                        "metric_core_recall": float(
                            selected_metrics["metric_core_recall"]
                        )
                        >= 0.80,
                        "core_guided_area_change": (0.35 <= area_expansion),
                        "candidate_score_not_worse": (
                            candidate_selection["selected"] == "yoloe_seed_preserved"
                            or float(candidate_selection["score_margin"])
                            >= float(candidate_selection["minimum_score_margin"])
                        ),
                    }
                else:
                    origin_gates = {
                        "mapping_seed_recall": raw_recall >= 0.90,
                        "mapping_seed_area_expansion": (1.0 <= area_expansion),
                    }
            quality_gates = {**hard_gates, **origin_gates}
            accepted = _view_acceptance(final_pixels, quality_gates)
            rejection_reasons = [
                key for key, passed in quality_gates.items() if not passed
            ]
            global_image_id = source_image_count + rescue_image_id
            visual_relative = (
                Path("visuals")
                / f"object_{object_id:06d}__img_{global_image_id:06d}.jpg"
            )
            _write_visual(output / visual_relative, rgb, raw, sam, final)
            row = {
                "source_image": str(frame.get("source_image") or ""),
                "rescue_image_id": rescue_image_id,
                "global_image_id": global_image_id,
                "physical_timestamp_ns": int(tracking_physical_timestamp(frame) or 0),
                "accepted": accepted,
                "seed_origin": seed_origin,
                "core_guided_shrink": core_guided_shrink,
                "candidate_selection": candidate_selection,
                "whole_object_expansion": whole_object_expansion,
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
                "area_expansion_gate": {
                    **area_expansion_gate,
                    "seed_reference": (
                        "projected_metric_voxel_seed"
                        if projected_seed
                        else "mapping_raw_seed"
                    ),
                },
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
                **{
                    f"projected_{key}": value
                    for key, value in projected_box_metrics.items()
                },
                **depth_metrics,
            }
            if accepted:
                relative = (
                    Path("masks")
                    / f"object_{object_id:06d}"
                    / f"img_{global_image_id:06d}_det_0000.npz"
                )
                sidecar = _write_sidecar(output / relative, rgb, final)
                sidecar["mask_relative"] = relative.as_posix()
                row.update(sidecar)
            view_results.append(row)
        independent_accepted = [
            row for row in view_results if bool(row.get("accepted"))
        ]
        propagation_audit: dict[str, Any] = {
            "enabled": bool(args.bidirectional_propagation),
            "reason": "not_required",
            "seed_views": len(independent_accepted),
            "attempted_views": 0,
            "accepted_views": 0,
        }
        if bool(args.bidirectional_propagation) and 0 < len(independent_accepted) < len(
            association.get("matched_views") or []
        ):
            view_results, propagation_seconds, propagation_audit = (
                _propagate_best_mask_bidirectionally(
                    torch,
                    processor,
                    model,
                    association,
                    view_results,
                    rescue_state,
                    frame_by_stem,
                    frames_path,
                    object_voxels,
                    world_support,
                    output,
                    source_image_count,
                    minimum_model_confidence=float(args.minimum_model_confidence),
                    maximum_area_expansion=float(args.maximum_area_expansion),
                    minimum_depth_padding_m=float(args.minimum_depth_padding_m),
                    maximum_depth_padding_m=float(args.maximum_depth_padding_m),
                    obb_audit_expansion=float(args.obb_audit_expansion),
                    maximum_tracking_gap_multiplier=float(
                        args.maximum_tracking_gap_multiplier
                    ),
                    maximum_tracking_translation_m=float(
                        args.maximum_tracking_translation_m
                    ),
                    maximum_tracking_rotation_degrees=float(
                        args.maximum_tracking_rotation_degrees
                    ),
                    maximum_tracking_episode_frames=int(
                        args.maximum_tracking_episode_frames
                    ),
                )
            )
            inference_seconds += propagation_seconds
        accepted_views = [row for row in view_results if bool(row.get("accepted"))]
        accepted, object_status, object_quality_gates = _object_acceptance(
            view_results,
            multiview_support_audit,
            minimum_views=int(args.minimum_accepted_views),
            minimum_independent_views=int(args.minimum_independent_views),
            minimum_support_points=int(args.minimum_metric_support_points),
        )
        object_results.append(
            {
                **{
                    key: value
                    for key, value in association.items()
                    if key != "matched_views"
                },
                "status": object_status,
                "accepted": accepted,
                "accepted_views": len(accepted_views),
                "multiview_support": multiview_support_audit,
                "bidirectional_propagation": propagation_audit,
                "object_quality_gates": object_quality_gates,
                "views": view_results,
            }
        )

    concept_model_load_seconds = 0.0
    concept_inference_seconds = 0.0
    second_chance_audits = list(concept_target_rejections)
    primary_results_by_id = {
        int(row["object_id"]): row
        for row in object_results
        if row.get("object_id") is not None
    }
    pending_concept_targets: list[dict[str, Any]] = []
    for target in concept_targets:
        existing = primary_results_by_id.get(int(target["object_id"]))
        if existing is not None and bool(existing.get("accepted")):
            second_chance_audits.append(
                {
                    "object_id": int(target["object_id"]),
                    "status": "skipped_primary_object_already_accepted",
                    "accepted": False,
                    "model_invoked": False,
                }
            )
        else:
            pending_concept_targets.append(target)

    if pending_concept_targets:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3 concept second chance requires CUDA; CPU fallback is forbidden"
            )
        if not associations:
            torch.cuda.reset_peak_memory_stats()
        if model is not None:
            del model
        if processor is not None:
            del processor
        model = processor = None
        gc.collect()
        torch.cuda.empty_cache()
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        concept_load_started = time.perf_counter()
        processor = Sam3VideoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = (
            Sam3VideoModel.from_pretrained(
                str(model_path), local_files_only=True, dtype=torch.bfloat16
            )
            .eval()
            .to("cuda")
        )
        torch.cuda.synchronize()
        concept_model_load_seconds = time.perf_counter() - concept_load_started
        object_results, runtime_audits, concept_inference_seconds = (
            _run_concept_second_chance(
                torch,
                processor,
                model,
                pending_concept_targets,
                object_results,
                source_state,
                rescue_state,
                frame_by_stem,
                rescue_root,
                frames_path,
                output,
                source_image_count=source_image_count,
                minimum_accepted_views=int(args.minimum_accepted_views),
                minimum_independent_views=int(args.minimum_independent_views),
                second_chance_minimum_physical_timestamps=int(
                    args.second_chance_minimum_physical_timestamps
                ),
                minimum_metric_support_points=int(
                    args.minimum_metric_support_points
                ),
                minimum_model_confidence=float(args.minimum_model_confidence),
                minimum_depth_padding_m=float(args.minimum_depth_padding_m),
                maximum_depth_padding_m=float(args.maximum_depth_padding_m),
                obb_audit_expansion=float(args.obb_audit_expansion),
                maximum_competing_mask_fraction=float(
                    args.maximum_competing_mask_fraction
                ),
                maximum_area_expansion=float(args.maximum_area_expansion),
                minimum_anchor_pixels=int(
                    args.second_chance_minimum_anchor_pixels
                ),
                minimum_identity_surface_fraction=float(
                    args.second_chance_minimum_identity_surface_fraction
                ),
                minimum_cycle_recall=float(
                    args.second_chance_minimum_cycle_recall
                ),
                maximum_centroid_distance_m=float(
                    args.second_chance_maximum_centroid_distance_m
                ),
                consensus_voxel_size_m=float(
                    args.second_chance_consensus_voxel_size_m
                ),
                maximum_candidates_per_view=int(
                    args.second_chance_maximum_candidates_per_view
                ),
                maximum_exact_combinations=int(
                    args.second_chance_maximum_exact_combinations
                ),
            )
        )
        second_chance_audits.extend(runtime_audits)
        inference_seconds += concept_inference_seconds
        del model, processor
        model = processor = None
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "schema": "farm.full-colmap-mask-refinement.v1",
        "status": "PASS",
        "refinement_role": (
            HELDOUT_REFINEMENT_ROLE
            if args.view_role == HELDOUT_VIEW_ROLE
            else TRAIN_REFINEMENT_ROLE
        ),
        "input_view_contract": input_view_contract,
        "created_unix_s": time.time(),
        "plan": str(plan_path),
        "source_state": str(args.source_state.expanduser().resolve()),
        "rescue_state": str(args.rescue_state.expanduser().resolve()),
        "rescue_frames_json": str(frames_path),
        "model": {
            "path": str(model_path),
            "local_files_only": True,
            "text_prompts": concept_prompts_path is not None,
            "manual_prompts": False,
            "primary_architecture": "Sam3TrackerVideoModel",
            "second_chance_architecture": "Sam3VideoModel"
            if concept_prompts_path is not None
            else None,
        },
        "policy": {
            "view_role": str(args.view_role),
            "state_fit_authorized": args.view_role != HELDOUT_VIEW_ROLE,
            "heldout_reference_is_evaluation_only": args.view_role == HELDOUT_VIEW_ROLE,
            "merge_apply_forbidden": args.view_role == HELDOUT_VIEW_ROLE,
            "reference_masks_are_pseudo_labels_not_ground_truth": args.view_role
            == HELDOUT_VIEW_ROLE,
            "category_agnostic_association": True,
            "exclusive_track_assignment": True,
            "projected_metric_voxel_fallback": True,
            "rest3d_inspired_sparse_metric_seed": True,
            "rest3d_inspired_multiview_metric_fusion": True,
            "rest3d_bidirectional_video_propagation": bool(
                args.bidirectional_propagation
            ),
            "independent_forward_reverse_sessions": True,
            "propagated_views_revalidated_with_metric_depth_and_obb": True,
            "multiview_support_gate": True,
            "core_guided_mask_shrink_requires_two_independent_views": True,
            "single_ab_selection_between_yoloe_and_sam3": True,
            "detector_box_is_prompt_only": True,
            "automatic_positive_negative_points": True,
            "whole_object_expansion_ids": sorted(whole_object_expansion_ids),
            "whole_object_expansion_keeps_competing_instances_negative": True,
            "whole_object_expansion_requires_vlm_audit": bool(
                whole_object_expansion_ids
            ),
            "whole_object_expansion_still_uses_single_yoloe_sam3_ab": True,
            "metric_obb_box_prompt": True,
            "rendered_depth_gate": True,
            "minimum_association_views": int(args.minimum_association_views),
            "minimum_accepted_views": int(args.minimum_accepted_views),
            "minimum_independent_views": int(args.minimum_independent_views),
            "minimum_metric_support_points": int(args.minimum_metric_support_points),
            "minimum_model_confidence": float(args.minimum_model_confidence),
            "minimum_depth_padding_m": float(args.minimum_depth_padding_m),
            "maximum_depth_padding_m": float(args.maximum_depth_padding_m),
            "obb_audit_expansion": float(args.obb_audit_expansion),
            "maximum_competing_mask_fraction": float(
                args.maximum_competing_mask_fraction
            ),
            "maximum_area_expansion": float(args.maximum_area_expansion),
            "maximum_area_expansion_is_hard_gate_for_all_seed_origins": True,
            "area_expansion_hard_gate_name": "bounded_seed_area_expansion",
            "area_expansion_reference_by_seed_origin": {
                "projected_metric_voxels": "projected_metric_voxel_seed",
                "rest3d_bidirectional_propagation": "projected_metric_guidance_seed",
                "rescue_mapping_track": "mapping_raw_seed",
            },
            "tracking_stream_isolation": "exact camera/view-family",
            "tracking_uses_physical_source_timestamps": True,
            "maximum_tracking_gap_multiplier": float(
                args.maximum_tracking_gap_multiplier
            ),
            "maximum_tracking_translation_m": float(
                args.maximum_tracking_translation_m
            ),
            "maximum_tracking_rotation_degrees": float(
                args.maximum_tracking_rotation_degrees
            ),
            "maximum_tracking_episode_frames": int(
                args.maximum_tracking_episode_frames
            ),
            "cross_stream_tracking_forbidden": True,
            "cross_stream_association_requires_3d_evidence": True,
            "object_acceptance_counts_physical_timestamps": True,
            "obb_cannot_override_depth_surface_gate": True,
            "second_chance_text_concept_enabled": concept_prompts_path is not None,
            "second_chance_is_default_off": True,
            "second_chance_primary_thresholds_unchanged": True,
            "second_chance_prompt_requires_immutable_semantic_audit_binding": True,
            "second_chance_legacy_geometry_coverage_is_advisory_only": True,
            "second_chance_minimum_physical_timestamps": int(
                args.second_chance_minimum_physical_timestamps
            ),
            "second_chance_minimum_identity_surface_fraction": float(
                args.second_chance_minimum_identity_surface_fraction
            ),
            "second_chance_minimum_cycle_recall": float(
                args.second_chance_minimum_cycle_recall
            ),
            "second_chance_maximum_centroid_distance_m": float(
                args.second_chance_maximum_centroid_distance_m
            ),
            "second_chance_consensus_voxel_size_m": float(
                args.second_chance_consensus_voxel_size_m
            ),
            "second_chance_maximum_candidates_per_view": int(
                args.second_chance_maximum_candidates_per_view
            ),
            "second_chance_maximum_exact_combinations": int(
                args.second_chance_maximum_exact_combinations
            ),
        },
        "associations": len(associations),
        "accepted_objects": sum(bool(row["accepted"]) for row in object_results),
        "accepted_object_ids": [
            int(row["object_id"]) for row in object_results if row["accepted"]
        ],
        "second_chance": {
            "enabled": concept_prompts_path is not None,
            "prompt_manifest": str(concept_prompts_path)
            if concept_prompts_path is not None
            else None,
            "prompt_plan_binding": concept_plan_binding,
            "requested_object_ids": sorted(concept_prompts),
            "role_scoped_target_object_ids": [
                int(row["object_id"]) for row in concept_targets
            ],
            "pending_model_target_object_ids": [
                int(row["object_id"]) for row in pending_concept_targets
            ],
            "target_rejections": concept_target_rejections,
            "audits": second_chance_audits,
        },
        "objects": object_results,
        "timing": {
            "model_load_seconds": model_load_seconds + concept_model_load_seconds,
            "primary_model_load_seconds": model_load_seconds,
            "concept_model_load_seconds": concept_model_load_seconds,
            "inference_seconds": inference_seconds,
            "concept_inference_seconds": concept_inference_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "resources": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "hashes": {
            "plan_sha256": sha256_file(plan_path),
            "rescue_frames_sha256": sha256_file(frames_path),
            "source_state_sha256": sha256_file(
                args.source_state.expanduser().resolve()
            ),
            "rescue_state_sha256": sha256_file(
                args.rescue_state.expanduser().resolve()
            ),
            **(
                {"concept_prompts_sha256": sha256_file(concept_prompts_path)}
                if concept_prompts_path is not None
                else {}
            ),
        },
    }
    report_path = output / "result.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    marker = {
        "schema": "farm.full-colmap-mask-refinement.success.v1",
        "status": "success",
        "refinement_role": report["refinement_role"],
        "result": "result.json",
        "result_sha256": sha256_file(report_path),
    }
    (output / "_SUCCESS.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "associations",
                    "accepted_objects",
                    "accepted_object_ids",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
