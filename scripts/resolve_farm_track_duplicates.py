#!/usr/bin/env python3
"""Resolve fragmented FARM tracks for presentation without deleting evidence.

The online mapper can leave two persistent tracks for one physical instance
when their image sets are disjoint.  This post-process uses only saved visual
features and metric 3D evidence.  It never changes ``active`` or any OBB: a
conservative display status points redundant rows at a canonical track while
the strict/diagnostic layers retain the complete metric state.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree

from scene_graph.map_update.mask_observations import get_pairwise_mask_overlap_records
from scene_graph.utils.geometry import decode_voxel_keys_numpy


BASE_CANDIDATE_DISTANCE_M = 0.80
MAX_SCALE_CANDIDATE_DISTANCE_M = 1.50
MAX_SCALE_NEAR_DISTANCE_M = 0.40
MASK_OVERRIDE_MIN_IOU = 0.80
MASK_OVERRIDE_MIN_CONTAINMENT = 0.95
MASK_OVERRIDE_MIN_AREA_RATIO = 0.75
MASK_OVERRIDE_MIN_FEATURE_COSINE = 0.92
MASK_RELAXED_MIN_IOU = 0.75
MASK_RELAXED_MIN_CONTAINMENT = 0.95
MASK_RELAXED_MIN_AREA_RATIO = 0.75
MASK_RELAXED_MIN_FEATURE_COSINE = 0.95
MASK_RELAXED_MAX_CENTER_DISTANCE_M = 0.35
MASK_OVERWHELMING_MIN_IOU = 0.875
MASK_OVERWHELMING_MIN_CONTAINMENT = 0.985
MASK_OVERWHELMING_MIN_AREA_RATIO = 0.85
MASK_OVERWHELMING_MIN_FEATURE_COSINE = 0.975
MASK_OVERWHELMING_MAX_CENTER_DISTANCE_M = 0.35
GEOMETRY_OVERWHELMING_MIN_FEATURE_COSINE = 0.85
GEOMETRY_OVERWHELMING_MAX_CENTER_DISTANCE_M = 0.20
GEOMETRY_OVERWHELMING_MAX_NORMALIZED_CENTER_DISTANCE = 0.12
GEOMETRY_OVERWHELMING_MIN_NEAR = 0.98
GEOMETRY_OVERWHELMING_MIN_INSIDE = 0.55
GEOMETRY_OVERWHELMING_MAX_INSIDE = 0.95
GEOMETRY_OVERWHELMING_MIN_VOLUME_RATIO = 0.50
GEOMETRY_OVERWHELMING_MIN_AXIS_RATIO = 0.55
EXPLICIT_ASSEMBLY_MIN_CONTAINMENT = 0.95
EXPLICIT_ASSEMBLY_MIN_SHARED_FRAMES = 2
LINEAGE_MIN_ANCHOR_FEATURE_COSINE = 0.975
LINEAGE_MAX_ASSEMBLY_CENTER_DISTANCE_M = 1.50
LINEAGE_MIN_CHILD_NEAR = 0.90
LINEAGE_MIN_CHILD_INSIDE = 0.85
LINEAGE_MAX_CHILD_PARENT_VOLUME_RATIO = 0.50
LARGE_MASK_MIN_FEATURE_COSINE = 0.95
LARGE_MASK_MAX_NORMALIZED_CENTER_DISTANCE = 0.12
LARGE_MASK_MIN_VOLUME_RATIO = 0.60
LARGE_MASK_MIN_SORTED_AXIS_RATIO = 0.80
LARGE_MASK_MIN_OBB_DIAGONAL_M = 2.00
PERSISTED_MASK_CANDIDATE_MIN_IOU = 0.65
PERSISTED_MASK_CANDIDATE_MIN_CONTAINMENT = 0.95
PERSISTED_MASK_CANDIDATE_MIN_AREA_RATIO = 0.65
TRACK_FRAGMENT_MIN_IOU = 0.68
TRACK_FRAGMENT_MIN_CONTAINMENT = 0.96
TRACK_FRAGMENT_MIN_AREA_RATIO = 0.68
TRACK_FRAGMENT_MIN_FEATURE_COSINE = 0.970
TRACK_FRAGMENT_MAX_NORMALIZED_CENTER_DISTANCE = 0.22
TRACK_FRAGMENT_MIN_MAX_SCALE_NEAR = 0.99
TRACK_FRAGMENT_MIN_MAX_INSIDE = 0.85
TRACK_FRAGMENT_MIN_VOLUME_RATIO = 0.34
TRACK_FRAGMENT_MIN_SORTED_AXIS_RATIO = 0.35
TRACK_FRAGMENT_MAX_COMMON_OVER_MIN = 0.30
TRACK_ALIAS_MIN_IOU = 0.79
TRACK_ALIAS_MIN_CONTAINMENT = 0.985
TRACK_ALIAS_MIN_AREA_RATIO = 0.80
TRACK_ALIAS_MIN_FEATURE_COSINE = 0.96
TRACK_ALIAS_MAX_NORMALIZED_CENTER_DISTANCE = 0.25
TRACK_ALIAS_MIN_VOLUME_RATIO = 0.85
TRACK_ALIAS_MIN_SORTED_AXIS_RATIO = 0.85
TRACK_ALIAS_MIN_SCALE_NEAR = 0.88
TRACK_ALIAS_MIN_INSIDE = 0.50
TRACK_ALIAS_MIN_MAX_INSIDE = 0.70
TRACK_ALIAS_MAX_COMMON_OVER_MIN = 0.20
TRACK_SPLIT_EXACT_MIN_IOU = 0.65
TRACK_SPLIT_EXACT_MIN_CONTAINMENT = 0.99
TRACK_SPLIT_EXACT_MIN_AREA_RATIO = 0.65
TRACK_SPLIT_EXACT_MIN_FEATURE_COSINE = 0.94
TRACK_SPLIT_EXACT_MAX_NORMALIZED_CENTER_DISTANCE = 0.30
TRACK_SPLIT_EXACT_MIN_TRACK_VIEWS = 3
TRACK_SPLIT_EXACT_MIN_VIEW_COUNT_RATIO = 2.0
TRACK_SPLIT_EXACT_MIN_COMMON_VIEWS = 1
TRACK_SPLIT_EXACT_MAX_COMMON_OVER_MIN = 0.35


@dataclass(frozen=True)
class PairEvidence:
    first_id: int
    second_id: int
    feature_cosine: float
    center_distance_m: float
    first_inside_second: float
    second_inside_first: float
    first_near_second: float
    second_near_first: float
    first_thinness: float
    second_thinness: float
    semantic_compatible: bool
    cannot_link: bool
    cannot_link_overridden: bool
    three_d_corroborated: bool
    candidate_distance_m: float
    near_distance_m: float
    common_frames: int
    first_track_views: int
    second_track_views: int
    common_over_min_track: float
    track_view_count_ratio: float
    normalized_center_distance: float
    exact_resolved_category: bool
    compared_mask_frames: int
    mask_image_id: int | None
    mask_iou: float
    mask_containment: float
    mask_area_ratio: float
    mask_evidence_source: str
    mask_evidence_fingerprint: str
    candidate_source: str
    raw_relation: str
    override_tier: str
    relation: str


@dataclass(frozen=True)
class SameFrameMaskEvidence:
    common_frames: int = 0
    compared_frames: int = 0
    compared_pairs: int = 0
    image_id: int | None = None
    iou: float = 0.0
    containment: float = 0.0
    area_ratio: float = 0.0
    first_area_pixels: int = 0
    second_area_pixels: int = 0
    first_path: str = ""
    second_path: str = ""
    source: str = "none"
    evidence_fingerprint: str = ""


@dataclass(frozen=True)
class TrackSplitEvidence:
    first_views: int
    second_views: int
    common_views: int
    common_over_min: float


def _scalar_at(state: Mapping[str, object], key: str, index: int, default: float = 0.0) -> float:
    values = state.get(key)
    try:
        if isinstance(values, torch.Tensor):
            return float(values[index].item())
        if isinstance(values, (list, tuple)) and index < len(values):
            return float(values[index])
    except (IndexError, TypeError, ValueError):
        pass
    return float(default)


def _text_at(state: Mapping[str, object], key: str, index: int) -> str:
    values = state.get(key)
    if isinstance(values, (list, tuple)) and index < len(values):
        return str(values[index] or "").strip()
    return ""


def _resolved_semantic_keep(state: Mapping[str, object], index: int) -> bool:
    category = _text_at(state, "object_category", index).lower()
    decision = _text_at(state, "object_caption_decision", index).lower()
    tier = _text_at(state, "object_semantic_tier", index).lower()
    return bool(
        category
        and "unresolved" not in category
        and "unknown" not in category
        and decision in {"keep", "relabel"}
        and tier in {"confirmed", "probable"}
    )


def _presentation_eligible_direct(
    state: Mapping[str, object],
    index: int,
    *,
    active: np.ndarray,
    members: object,
) -> bool:
    member_rows = members if isinstance(members, (list, tuple)) else []
    is_assembly = bool(
        index < len(member_rows)
        and isinstance(member_rows[index], (list, tuple))
        and member_rows[index]
    )
    return bool(
        0 <= index < active.size
        and bool(active[index])
        and not is_assembly
        and _text_at(state, "object_geometry_status", index).lower() == "geometry_pass"
        and _resolved_semantic_keep(state, index)
    )


def _persisted_pair_mask_candidates(
    state: Mapping[str, object], eligible_object_ids: set[int]
) -> dict[frozenset[int], Mapping[str, object]]:
    """Return saved-mask pairs that must enter offline evaluation.

    The persisted online store is bounded and filesystem-independent. This
    deliberately broad index only expands candidate generation; suppression
    remains governed by the stricter multi-cue classifiers below.
    """
    store = state.get("object_pair_mask_overlap_evidence")
    if not isinstance(store, Mapping) or store.get("schema") != "farm.object-pair-mask-overlap.v1":
        return {}
    pairs = store.get("pairs")
    if not isinstance(pairs, Mapping):
        return {}
    selected: dict[frozenset[int], Mapping[str, object]] = {}
    for entry in pairs.values():
        if not isinstance(entry, Mapping):
            continue
        try:
            first_id = int(entry.get("first_object_id"))
            second_id = int(entry.get("second_object_id"))
        except (TypeError, ValueError):
            continue
        if (
            first_id == second_id
            or first_id not in eligible_object_ids
            or second_id not in eligible_object_ids
        ):
            continue
        best: Mapping[str, object] | None = None
        best_rank: tuple[float, float, float, int] | None = None
        raw_images = entry.get("images")
        for record in raw_images if isinstance(raw_images, list) else []:
            if not isinstance(record, Mapping):
                continue
            try:
                image_id = int(record.get("image_id", -1))
                iou = float(record.get("raw_iou", -1.0))
                containment = float(record.get("raw_containment", -1.0))
                area_ratio = float(record.get("raw_area_ratio", -1.0))
            except (TypeError, ValueError):
                continue
            if (
                image_id < 0
                or iou < PERSISTED_MASK_CANDIDATE_MIN_IOU
                or containment < PERSISTED_MASK_CANDIDATE_MIN_CONTAINMENT
                or area_ratio < PERSISTED_MASK_CANDIDATE_MIN_AREA_RATIO
            ):
                continue
            rank = (iou, containment, area_ratio, -image_id)
            if best_rank is None or rank > best_rank:
                best = record
                best_rank = rank
        if best is not None:
            selected[frozenset((first_id, second_id))] = best
    return selected


def _spatial_candidate_pairs(
    *,
    eligible: list[int],
    object_ids: np.ndarray,
    centers: np.ndarray,
    dimensions: np.ndarray,
    cannot_links: object,
    frame_cache: Mapping[int, set[int]],
    near_distance_m: float,
) -> set[tuple[int, int]]:
    """Return the legacy spatial candidate set without an O(N^2) scan."""
    finite = [index for index in eligible if np.isfinite(centers[index]).all()]
    if len(finite) < 2:
        return set()
    tree = cKDTree(np.asarray(centers[finite], dtype=np.float64))
    selected: set[tuple[int, int]] = set()
    for first_pos, second_pos in tree.query_pairs(
        r=MAX_SCALE_CANDIDATE_DISTANCE_M
    ):
        first, second = sorted((finite[first_pos], finite[second_pos]))
        first_id = int(object_ids[first])
        second_id = int(object_ids[second])
        cannot_link = _cannot_linked(cannot_links, first_id, second_id)
        co_visible = bool(frame_cache[first].intersection(frame_cache[second]))
        scale_candidate_distance, _ = _scale_aware_distances(
            dimensions[first], dimensions[second], near_distance_m
        )
        candidate_distance = (
            scale_candidate_distance
            if not cannot_link and not co_visible
            else BASE_CANDIDATE_DISTANCE_M
        )
        center_distance = float(np.linalg.norm(centers[first] - centers[second]))
        if np.isfinite(center_distance) and center_distance <= candidate_distance:
            selected.add((first, second))
    return selected


def _numpy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _cannot_linked(cannot_links: object, first_id: int, second_id: int) -> bool:
    if not isinstance(cannot_links, Mapping):
        return False

    def linked(source: int, target: int) -> bool:
        values = cannot_links.get(source)
        if values is None:
            values = cannot_links.get(str(source), ())
        for value in values or ():
            try:
                if int(value) == target:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    return linked(first_id, second_id) or linked(second_id, first_id)


def _object_frame_ids(state: dict, index: int) -> set[int]:
    frame_ids: set[int] = set()
    for key in ("object_image_ids", "viewpoint_image_ids"):
        rows = state.get(key) or []
        values = rows[index] if index < len(rows) else []
        for value in values if isinstance(values, (list, tuple, set)) else []:
            try:
                frame_ids.add(int(value))
            except (TypeError, ValueError):
                continue
    rows = state.get("object_mask_observations") or []
    observations = rows[index] if index < len(rows) else []
    for observation in observations if isinstance(observations, list) else []:
        if not isinstance(observation, Mapping):
            continue
        try:
            image_id = int(observation.get("image_id", -1))
        except (TypeError, ValueError):
            continue
        if image_id >= 0:
            frame_ids.add(image_id)
    return frame_ids


def _mask_rows_by_image(rows: object) -> dict[int, list[Mapping[str, object]]]:
    by_image: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for observation in rows if isinstance(rows, list) else []:
        if not isinstance(observation, Mapping):
            continue
        try:
            image_id = int(observation.get("image_id", -1))
        except (TypeError, ValueError):
            continue
        if image_id >= 0:
            by_image[image_id].append(observation)
    return dict(by_image)


def _mask_observation_path(
    mask_root: Path,
    object_id: int,
    observation: Mapping[str, object],
) -> Path | None:
    recorded_raw = str(observation.get("path") or observation.get("mask_path") or "")
    if not recorded_raw:
        return None
    recorded = Path(recorded_raw)
    candidates = [mask_root / f"object_{object_id:06d}" / recorded.name]
    if not recorded.is_absolute():
        candidates.append(mask_root / recorded)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_raw_mask_crop(
    path: Path,
    cache: dict[Path, tuple[np.ndarray, np.ndarray, int] | None],
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if path in cache:
        return cache[path]
    result: tuple[np.ndarray, np.ndarray, int] | None = None
    try:
        with np.load(path, allow_pickle=False) as data:
            shape = np.asarray(data["raw_shape"], dtype=np.int32).reshape(2)
            bbox = np.asarray(data["raw_bbox_xyxy"], dtype=np.int32).reshape(4)
            height, width = int(shape[0]), int(shape[1])
            if height <= 0 or width <= 0:
                raise ValueError("empty packed mask")
            flat = np.unpackbits(
                np.asarray(data["raw_bits"], dtype=np.uint8), bitorder="little"
            )[: height * width]
            if flat.size != height * width:
                raise ValueError("truncated packed mask")
            crop = flat.reshape(height, width).astype(bool, copy=False)
            area = int(np.count_nonzero(crop))
            if area > 0:
                result = (crop, bbox, area)
    except (KeyError, OSError, ValueError):
        result = None
    cache[path] = result
    return result


def _mask_intersection_pixels(
    first_mask: np.ndarray,
    first_bbox: np.ndarray,
    second_mask: np.ndarray,
    second_bbox: np.ndarray,
) -> int:
    x0 = max(int(first_bbox[0]), int(second_bbox[0]))
    y0 = max(int(first_bbox[1]), int(second_bbox[1]))
    x1 = min(int(first_bbox[2]), int(second_bbox[2]))
    y1 = min(int(first_bbox[3]), int(second_bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return 0
    first = first_mask[
        y0 - int(first_bbox[1]) : y1 - int(first_bbox[1]),
        x0 - int(first_bbox[0]) : x1 - int(first_bbox[0]),
    ]
    second = second_mask[
        y0 - int(second_bbox[1]) : y1 - int(second_bbox[1]),
        x0 - int(second_bbox[0]) : x1 - int(second_bbox[0]),
    ]
    height = min(first.shape[0], second.shape[0])
    width = min(first.shape[1], second.shape[1])
    if height <= 0 or width <= 0:
        return 0
    return int(np.count_nonzero(first[:height, :width] & second[:height, :width]))


def _saved_pair_mask_evidence(
    records: object,
    *,
    first_id: int,
    second_id: int,
    common_frames: int,
) -> SameFrameMaskEvidence:
    compared_frames: set[int] = set()
    compared_pairs = 0
    best: SameFrameMaskEvidence | None = None
    best_key: tuple[float, ...] | None = None
    swap_areas = int(first_id) > int(second_id)
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, Mapping):
            continue
        try:
            image_id = int(record.get("image_id", -1))
            iou = float(record.get("raw_iou", -1.0))
            containment = float(record.get("raw_containment", -1.0))
            area_ratio = float(record.get("raw_area_ratio", -1.0))
            first_area = int(record.get("first_area_pixels", 0))
            second_area = int(record.get("second_area_pixels", 0))
            pair_count = max(1, int(record.get("comparison_count", 1)))
        except (TypeError, ValueError):
            continue
        if image_id < 0 or not all(
            np.isfinite(value) and 0.0 <= value <= 1.0
            for value in (iou, containment, area_ratio)
        ):
            continue
        if swap_areas:
            first_area, second_area = second_area, first_area
        compared_frames.add(image_id)
        compared_pairs += pair_count
        passes = (
            iou >= MASK_OVERRIDE_MIN_IOU
            and containment >= MASK_OVERRIDE_MIN_CONTAINMENT
            and area_ratio >= MASK_OVERRIDE_MIN_AREA_RATIO
        )
        key = (float(passes), iou, containment, area_ratio, -float(image_id))
        if best_key is None or key > best_key:
            best_key = key
            best = SameFrameMaskEvidence(
                common_frames=max(int(common_frames), len(compared_frames)),
                image_id=image_id,
                iou=iou,
                containment=containment,
                area_ratio=area_ratio,
                first_area_pixels=first_area,
                second_area_pixels=second_area,
                source="online_pair_summary",
                evidence_fingerprint=str(record.get("evidence_fingerprint") or ""),
            )
    if best is None:
        return SameFrameMaskEvidence(common_frames=int(common_frames))
    return SameFrameMaskEvidence(
        **{
            **best.__dict__,
            "common_frames": max(int(common_frames), len(compared_frames)),
            "compared_frames": len(compared_frames),
            "compared_pairs": compared_pairs,
        }
    )


def same_frame_mask_evidence(
    *,
    first_id: int,
    second_id: int,
    first_rows: object,
    second_rows: object,
    mask_root: Path | None,
    mask_cache: dict[Path, tuple[np.ndarray, np.ndarray, int] | None] | None = None,
    saved_pair_evidence: object = None,
) -> SameFrameMaskEvidence:
    first_by_image = _mask_rows_by_image(first_rows)
    second_by_image = _mask_rows_by_image(second_rows)
    common = sorted(set(first_by_image).intersection(second_by_image))
    cache = mask_cache if mask_cache is not None else {}
    compared_frames: set[int] = set()
    compared_pairs = 0
    best: SameFrameMaskEvidence | None = None
    best_key: tuple[float, ...] | None = None
    for image_id in common if mask_root is not None else []:
        for first_row in first_by_image[image_id]:
            first_path = _mask_observation_path(mask_root, first_id, first_row)
            if first_path is None:
                continue
            first = _load_raw_mask_crop(first_path, cache)
            if first is None:
                continue
            for second_row in second_by_image[image_id]:
                second_path = _mask_observation_path(mask_root, second_id, second_row)
                if second_path is None:
                    continue
                second = _load_raw_mask_crop(second_path, cache)
                if second is None:
                    continue
                compared_frames.add(image_id)
                compared_pairs += 1
                first_mask, first_bbox, first_area = first
                second_mask, second_bbox, second_area = second
                intersection = _mask_intersection_pixels(
                    first_mask, first_bbox, second_mask, second_bbox
                )
                union = first_area + second_area - intersection
                iou = intersection / float(union) if union > 0 else 0.0
                containment = intersection / float(min(first_area, second_area))
                area_ratio = min(first_area, second_area) / float(max(first_area, second_area))
                passes = (
                    iou >= MASK_OVERRIDE_MIN_IOU
                    and containment >= MASK_OVERRIDE_MIN_CONTAINMENT
                    and area_ratio >= MASK_OVERRIDE_MIN_AREA_RATIO
                )
                key = (float(passes), iou, containment, area_ratio)
                if best_key is None or key > best_key:
                    best_key = key
                    best = SameFrameMaskEvidence(
                        common_frames=len(common),
                        image_id=image_id,
                        iou=iou,
                        containment=containment,
                        area_ratio=area_ratio,
                        first_area_pixels=first_area,
                        second_area_pixels=second_area,
                        first_path=str(first_path),
                        second_path=str(second_path),
                        source="canonical_sidecars",
                    )
    if best is not None:
        return SameFrameMaskEvidence(
            **{
                **best.__dict__,
                "compared_frames": len(compared_frames),
                "compared_pairs": compared_pairs,
            }
        )
    # A decoded canonical sidecar always wins. The online summary is consulted
    # only when the capped/relocated sidecars provide no comparable mask pair.
    return _saved_pair_mask_evidence(
        saved_pair_evidence,
        first_id=first_id,
        second_id=second_id,
        common_frames=len(common),
    )


def mask_duplicate_override_allowed(
    evidence: SameFrameMaskEvidence,
    *,
    feature_cosine: float,
    three_d_corroborated: bool,
) -> bool:
    return bool(
        evidence.image_id is not None
        and evidence.iou >= MASK_OVERRIDE_MIN_IOU
        and evidence.containment >= MASK_OVERRIDE_MIN_CONTAINMENT
        and evidence.area_ratio >= MASK_OVERRIDE_MIN_AREA_RATIO
        and feature_cosine >= MASK_OVERRIDE_MIN_FEATURE_COSINE
        and three_d_corroborated
    )


def classify_same_frame_duplicate_override(
    evidence: SameFrameMaskEvidence,
    *,
    feature_cosine: float,
    three_d_corroborated: bool,
    semantic_compatible: bool,
    center_distance_m: float,
    raw_relation: str,
    first_near_second: float,
    second_near_first: float,
    first_inside_second: float,
    second_inside_first: float,
    first_volume: float,
    second_volume: float,
) -> str | None:
    """Return a high-precision same-frame cannot-link override tier.

    ``cannot_link`` is strong negative evidence, so a mask alone is never
    sufficient. Every accepted tier starts from a duplicate-like geometric
    relation; raw ``distinct`` and part/whole relations remain fail-closed.
    """
    if evidence.image_id is None or not raw_relation.startswith("duplicate_"):
        return None
    if raw_relation in {"part_surface", "part_contained"}:
        return None
    if mask_duplicate_override_allowed(
        evidence,
        feature_cosine=feature_cosine,
        three_d_corroborated=three_d_corroborated,
    ):
        return "strict"
    if (
        raw_relation in {"duplicate_strict", "duplicate_geometric"}
        and three_d_corroborated
        and semantic_compatible
        and feature_cosine >= MASK_RELAXED_MIN_FEATURE_COSINE
        and center_distance_m <= MASK_RELAXED_MAX_CENTER_DISTANCE_M
        and evidence.iou >= MASK_RELAXED_MIN_IOU
        and evidence.containment >= MASK_RELAXED_MIN_CONTAINMENT
        and evidence.area_ratio >= MASK_RELAXED_MIN_AREA_RATIO
    ):
        return "relaxed_3d"
    volume_ratio = min(first_volume, second_volume) / max(
        first_volume, second_volume, 1.0e-12
    )
    if (
        feature_cosine >= MASK_OVERWHELMING_MIN_FEATURE_COSINE
        and center_distance_m <= MASK_OVERWHELMING_MAX_CENTER_DISTANCE_M
        and evidence.iou >= MASK_OVERWHELMING_MIN_IOU
        and evidence.containment >= MASK_OVERWHELMING_MIN_CONTAINMENT
        and evidence.area_ratio >= MASK_OVERWHELMING_MIN_AREA_RATIO
        and min(first_near_second, second_near_first) >= 0.70
        and min(first_inside_second, second_inside_first) >= 0.55
        and max(first_inside_second, second_inside_first) >= 0.85
        and volume_ratio >= 0.50
    ):
        return "overwhelming_2d"
    return None


def _is_overwhelming_mask_geometry_conflict(
    evidence: SameFrameMaskEvidence,
    *,
    feature_cosine: float,
    raw_relation: str,
) -> bool:
    """Flag, but never merge, a strong mask/visual versus 3D disagreement."""
    return bool(
        raw_relation == "distinct"
        and evidence.image_id is not None
        and feature_cosine >= 0.95
        and evidence.iou >= MASK_OVERWHELMING_MIN_IOU
        and evidence.containment >= MASK_OVERWHELMING_MIN_CONTAINMENT
        and evidence.area_ratio >= MASK_OVERWHELMING_MIN_AREA_RATIO
    )


def _classify_large_object_same_mask_override(
    evidence: SameFrameMaskEvidence,
    *,
    feature_cosine: float,
    center_distance_m: float,
    first_dimensions: np.ndarray,
    second_dimensions: np.ndarray,
    semantic_compatible: bool,
    raw_relation: str,
) -> str | None:
    """Recover one large fragmented instance without changing metric state.

    Large objects can have incomplete, weakly intersecting voxel supports even
    when two same-frame detector masks describe the same physical instance.
    Raw ``distinct`` is allowed only under an overwhelming, fully auditable
    2D/semantic/metric-shape contract; nested or small objects remain review.
    """
    if (
        raw_relation != "distinct"
        or evidence.image_id is None
        or evidence.iou < MASK_OVERWHELMING_MIN_IOU
        or evidence.containment < MASK_OVERWHELMING_MIN_CONTAINMENT
        or evidence.area_ratio < MASK_OVERWHELMING_MIN_AREA_RATIO
        or feature_cosine < LARGE_MASK_MIN_FEATURE_COSINE
        or not semantic_compatible
    ):
        return None
    first_dims = np.asarray(first_dimensions, dtype=np.float64).reshape(-1)
    second_dims = np.asarray(second_dimensions, dtype=np.float64).reshape(-1)
    if (
        first_dims.size != 3
        or second_dims.size != 3
        or not np.isfinite(first_dims).all()
        or not np.isfinite(second_dims).all()
        or np.any(first_dims <= 0.0)
        or np.any(second_dims <= 0.0)
    ):
        return None
    first_diagonal = float(np.linalg.norm(first_dims))
    second_diagonal = float(np.linalg.norm(second_dims))
    min_diagonal = min(first_diagonal, second_diagonal)
    normalized_center = center_distance_m / max(min_diagonal, 1.0e-12)
    first_volume = float(np.prod(first_dims))
    second_volume = float(np.prod(second_dims))
    volume_ratio = min(first_volume, second_volume) / max(
        first_volume, second_volume, 1.0e-12
    )
    first_sorted = np.sort(first_dims)
    second_sorted = np.sort(second_dims)
    axis_ratio = float(np.min(
        np.minimum(first_sorted, second_sorted)
        / np.maximum(first_sorted, second_sorted)
    ))
    if (
        min_diagonal >= LARGE_MASK_MIN_OBB_DIAGONAL_M
        and normalized_center <= LARGE_MASK_MAX_NORMALIZED_CENTER_DISTANCE
        and volume_ratio >= LARGE_MASK_MIN_VOLUME_RATIO
        and axis_ratio >= LARGE_MASK_MIN_SORTED_AXIS_RATIO
    ):
        return "duplicate_large_object_same_mask_override"
    return None


def _classify_geometry_overwhelming_duplicate(
    *,
    feature_cosine: float,
    center_distance_m: float,
    first_near_second: float,
    second_near_first: float,
    first_inside_second: float,
    second_inside_first: float,
    first_volume: float,
    second_volume: float,
    first_dimensions: np.ndarray,
    second_dimensions: np.ndarray,
    cannot_link: bool,
    co_visible: bool,
) -> str | None:
    """Recover disjoint-history fragments only under overwhelming metric fit."""
    if cannot_link or co_visible:
        return None
    first_dims = np.asarray(first_dimensions, dtype=np.float64).reshape(-1)
    second_dims = np.asarray(second_dimensions, dtype=np.float64).reshape(-1)
    if (
        first_dims.size != 3
        or second_dims.size != 3
        or not np.isfinite(first_dims).all()
        or not np.isfinite(second_dims).all()
        or np.any(first_dims <= 0.0)
        or np.any(second_dims <= 0.0)
    ):
        return None
    min_diagonal = min(float(np.linalg.norm(first_dims)), float(np.linalg.norm(second_dims)))
    normalized_center_distance = center_distance_m / max(min_diagonal, 1.0e-12)
    volume_ratio = min(first_volume, second_volume) / max(
        first_volume, second_volume, 1.0e-12
    )
    axis_ratio = float(
        np.min(np.minimum(first_dims, second_dims) / np.maximum(first_dims, second_dims))
    )
    if (
        feature_cosine >= GEOMETRY_OVERWHELMING_MIN_FEATURE_COSINE
        and center_distance_m <= GEOMETRY_OVERWHELMING_MAX_CENTER_DISTANCE_M
        and normalized_center_distance <= GEOMETRY_OVERWHELMING_MAX_NORMALIZED_CENTER_DISTANCE
        and min(first_near_second, second_near_first) >= GEOMETRY_OVERWHELMING_MIN_NEAR
        and min(first_inside_second, second_inside_first) >= GEOMETRY_OVERWHELMING_MIN_INSIDE
        and max(first_inside_second, second_inside_first) >= GEOMETRY_OVERWHELMING_MAX_INSIDE
        and volume_ratio >= GEOMETRY_OVERWHELMING_MIN_VOLUME_RATIO
        and axis_ratio >= GEOMETRY_OVERWHELMING_MIN_AXIS_RATIO
    ):
        return "duplicate_geometry_overwhelming"
    return None


def _scale_aware_distances(
    first_dimensions: np.ndarray,
    second_dimensions: np.ndarray,
    base_near_distance_m: float,
) -> tuple[float, float]:
    first_scale = float(np.linalg.norm(np.maximum(first_dimensions, 0.0)))
    second_scale = float(np.linalg.norm(np.maximum(second_dimensions, 0.0)))
    scale = min(first_scale, second_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 0.0
    candidate = min(
        MAX_SCALE_CANDIDATE_DISTANCE_M,
        max(BASE_CANDIDATE_DISTANCE_M, 0.60 * scale),
    )
    near = max(
        float(base_near_distance_m),
        min(MAX_SCALE_NEAR_DISTANCE_M, 0.18 * scale),
    )
    return candidate, near


def _strong_bidirectional_3d(
    *,
    first_near_second: float,
    second_near_first: float,
    first_inside_second: float,
    second_inside_first: float,
    first_volume: float,
    second_volume: float,
) -> bool:
    volume_ratio = min(first_volume, second_volume) / max(
        first_volume, second_volume, 1.0e-12
    )
    return bool(
        min(first_near_second, second_near_first) >= 0.75
        and min(first_inside_second, second_inside_first) >= 0.45
        and max(first_inside_second, second_inside_first) >= 0.65
        and volume_ratio >= 0.50
    )


def _classify_scale_aware_duplicate(
    *,
    feature_cosine: float,
    center_distance_m: float,
    candidate_distance_m: float,
    first_near_second: float,
    second_near_first: float,
    first_inside_second: float,
    second_inside_first: float,
    first_volume: float,
    second_volume: float,
    semantics_match: bool,
) -> str:
    feature_floor = 0.94 if semantics_match else 0.96
    volume_ratio = min(first_volume, second_volume) / max(
        first_volume, second_volume, 1.0e-12
    )
    if (
        feature_cosine >= feature_floor
        and center_distance_m <= candidate_distance_m
        and min(first_near_second, second_near_first) >= 0.70
        and min(first_inside_second, second_inside_first) >= 0.45
        and max(first_inside_second, second_inside_first) >= 0.60
        and volume_ratio >= 0.55
    ):
        return "duplicate_scale_aware"
    return "distinct"


def _wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


_TEXT_STOP = {
    "a", "an", "and", "with", "the", "on", "of", "in", "to", "or",
    "object", "rectangular", "industrial", "visible", "large", "small",
}


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 1 and token not in _TEXT_STOP
    }


def semantic_compatible(
    first_category: object,
    second_category: object,
    first_caption: object,
    second_caption: object,
) -> bool:
    first_cat = _tokens(first_category)
    second_cat = _tokens(second_category)
    unresolved = {"unresolved", "unknown"}
    first_resolved = first_cat and not (first_cat & unresolved)
    second_resolved = second_cat and not (second_cat & unresolved)
    if first_resolved and second_resolved and (
        first_cat == second_cat or first_cat <= second_cat or second_cat <= first_cat
    ):
        return True
    first_text = _tokens(first_caption)
    second_text = _tokens(second_caption)
    union = first_text | second_text
    jaccard = len(first_text & second_text) / max(len(union), 1)
    return bool(jaccard >= 0.50)


def _resolved_category_agreement(first_category: object, second_category: object) -> bool:
    first = _tokens(first_category)
    second = _tokens(second_category)
    unresolved = {"unresolved", "unknown"}
    return bool(
        first
        and second
        and not (first & unresolved)
        and not (second & unresolved)
        and (first == second or first <= second or second <= first)
    )


def _resolved_category_exact(first_category: object, second_category: object) -> bool:
    first = _tokens(first_category)
    second = _tokens(second_category)
    unresolved = {"unresolved", "unknown"}
    return bool(
        first
        and second
        and not (first & unresolved)
        and not (second & unresolved)
        and first == second
    )


def _track_split_evidence(
    first_frames: set[int], second_frames: set[int]
) -> TrackSplitEvidence:
    common = len(first_frames.intersection(second_frames))
    denominator = min(len(first_frames), len(second_frames))
    return TrackSplitEvidence(
        first_views=len(first_frames),
        second_views=len(second_frames),
        common_views=common,
        common_over_min=(common / denominator if denominator else 0.0),
    )


def _classify_same_frame_track_fragment_override(
    evidence: SameFrameMaskEvidence,
    track: TrackSplitEvidence,
    *,
    feature_cosine: float,
    center_distance_m: float,
    first_dimensions: np.ndarray,
    second_dimensions: np.ndarray,
    first_near_second_scaled: float,
    second_near_first_scaled: float,
    first_inside_second: float,
    second_inside_first: float,
    categories_resolved: bool,
    category_agreement: bool,
    category_exact: bool,
    raw_relation: str,
) -> str | None:
    """Recover split detector tracks under auditable mask/3D/track evidence.

    This is deliberately limited to co-visible track branches.  It does not
    use scene ids or category names and cannot promote unresolved semantics.
    """
    if (
        evidence.image_id is None
        or track.common_views < 1
        or not categories_resolved
    ):
        return None
    first_dims = np.asarray(first_dimensions, dtype=np.float64).reshape(-1)
    second_dims = np.asarray(second_dimensions, dtype=np.float64).reshape(-1)
    if (
        first_dims.size != 3
        or second_dims.size != 3
        or not np.isfinite(first_dims).all()
        or not np.isfinite(second_dims).all()
        or np.any(first_dims <= 0.0)
        or np.any(second_dims <= 0.0)
    ):
        return None
    min_diagonal = min(float(np.linalg.norm(first_dims)), float(np.linalg.norm(second_dims)))
    normalized_center = center_distance_m / max(min_diagonal, 1.0e-12)
    first_volume = float(np.prod(first_dims))
    second_volume = float(np.prod(second_dims))
    volume_ratio = min(first_volume, second_volume) / max(
        first_volume, second_volume, 1.0e-12
    )
    first_sorted = np.sort(first_dims)
    second_sorted = np.sort(second_dims)
    axis_ratio = float(np.min(
        np.minimum(first_sorted, second_sorted)
        / np.maximum(first_sorted, second_sorted)
    ))
    min_track_views = min(track.first_views, track.second_views)
    max_track_views = max(track.first_views, track.second_views)
    track_view_ratio = max_track_views / max(min_track_views, 1)

    if (
        category_agreement
        and raw_relation != "part_surface"
        and (raw_relation == "part_contained" or raw_relation == "distinct" or raw_relation.startswith("duplicate_"))
        and feature_cosine >= TRACK_FRAGMENT_MIN_FEATURE_COSINE
        and evidence.iou >= TRACK_FRAGMENT_MIN_IOU
        and evidence.containment >= TRACK_FRAGMENT_MIN_CONTAINMENT
        and evidence.area_ratio >= TRACK_FRAGMENT_MIN_AREA_RATIO
        and normalized_center <= TRACK_FRAGMENT_MAX_NORMALIZED_CENTER_DISTANCE
        and max(first_near_second_scaled, second_near_first_scaled)
        >= TRACK_FRAGMENT_MIN_MAX_SCALE_NEAR
        and max(first_inside_second, second_inside_first)
        >= TRACK_FRAGMENT_MIN_MAX_INSIDE
        and volume_ratio >= TRACK_FRAGMENT_MIN_VOLUME_RATIO
        and axis_ratio >= TRACK_FRAGMENT_MIN_SORTED_AXIS_RATIO
        and track.common_over_min <= TRACK_FRAGMENT_MAX_COMMON_OVER_MIN
    ):
        return "resolved_same_category"

    if (
        not category_agreement
        and raw_relation == "duplicate_overlap_fragment"
        and feature_cosine >= TRACK_ALIAS_MIN_FEATURE_COSINE
        and evidence.iou >= TRACK_ALIAS_MIN_IOU
        and evidence.containment >= TRACK_ALIAS_MIN_CONTAINMENT
        and evidence.area_ratio >= TRACK_ALIAS_MIN_AREA_RATIO
        and normalized_center <= TRACK_ALIAS_MAX_NORMALIZED_CENTER_DISTANCE
        and volume_ratio >= TRACK_ALIAS_MIN_VOLUME_RATIO
        and axis_ratio >= TRACK_ALIAS_MIN_SORTED_AXIS_RATIO
        and min(first_near_second_scaled, second_near_first_scaled)
        >= TRACK_ALIAS_MIN_SCALE_NEAR
        and min(first_inside_second, second_inside_first) >= TRACK_ALIAS_MIN_INSIDE
        and max(first_inside_second, second_inside_first)
        >= TRACK_ALIAS_MIN_MAX_INSIDE
        and track.common_over_min <= TRACK_ALIAS_MAX_COMMON_OVER_MIN
    ):
        return "resolved_alias"

    if (
        category_exact
        and raw_relation == "distinct"
        and feature_cosine >= TRACK_SPLIT_EXACT_MIN_FEATURE_COSINE
        and evidence.iou >= TRACK_SPLIT_EXACT_MIN_IOU
        and evidence.containment >= TRACK_SPLIT_EXACT_MIN_CONTAINMENT
        and evidence.area_ratio >= TRACK_SPLIT_EXACT_MIN_AREA_RATIO
        and 0.0 <= normalized_center
        <= TRACK_SPLIT_EXACT_MAX_NORMALIZED_CENTER_DISTANCE
        and min_track_views >= TRACK_SPLIT_EXACT_MIN_TRACK_VIEWS
        and track_view_ratio >= TRACK_SPLIT_EXACT_MIN_VIEW_COUNT_RATIO
        and track.common_views >= TRACK_SPLIT_EXACT_MIN_COMMON_VIEWS
        and track.common_views <= min_track_views
        and track.common_over_min <= TRACK_SPLIT_EXACT_MAX_COMMON_OVER_MIN
    ):
        return "same_category_large_track_split"
    return None


def classify_pair(
    *,
    feature_cosine: float,
    center_distance_m: float,
    first_inside_second: float,
    second_inside_first: float,
    first_near_second: float,
    second_near_first: float,
    first_thinness: float,
    second_thinness: float,
    first_volume: float,
    second_volume: float,
    semantics_match: bool,
) -> str:
    """Return a fail-closed display relation for one metric pair."""
    near_min = min(first_near_second, second_near_first)
    inside_min = min(first_inside_second, second_inside_first)
    inside_max = max(first_inside_second, second_inside_first)
    # A compact reconstruction fully supported inside a much larger track is
    # a reversible part/whole relation even when their global crop features
    # differ. This catches payload/face fragments without merging nearby
    # independent objects: support, containment, scale and centre must all
    # agree metrically.
    contained_records = (
        (first_volume, first_near_second, first_inside_second, second_volume),
        (second_volume, second_near_first, second_inside_first, first_volume),
    )
    if center_distance_m <= 0.65 and any(
        child_volume * 2.0 <= parent_volume
        and near_parent >= 0.90
        and inside_parent >= 0.85
        for child_volume, near_parent, inside_parent, parent_volume in contained_records
    ):
        return "part_contained"
    if center_distance_m > 0.80 or feature_cosine < 0.86:
        return "distinct"
    if feature_cosine >= 0.92 and near_min >= 0.75 and inside_min >= 0.45:
        return "duplicate_strict"
    if feature_cosine >= 0.86 and near_min >= 0.90 and inside_min >= 0.70:
        return "duplicate_geometric"
    # A smaller track can be an incomplete reconstruction of the same
    # semantically stable instance. Require near-identical visual evidence,
    # strong one-way OBB containment and close metric centres. The resolver's
    # cannot-link gate remains authoritative, so this affects only the
    # reversible presentation layer.
    if (
        semantics_match
        and feature_cosine >= 0.92
        and center_distance_m <= 0.65
        and inside_max >= 0.90
        and max(first_near_second, second_near_first) >= 0.60
    ):
        return "duplicate_contained_semantic"
    # Open-vocabulary labels can differ for two fragments of one physical
    # instance. The high visual threshold plus bidirectional voxel proximity
    # avoids using text as a manual scene-specific hint.
    if (
        feature_cosine >= 0.90
        and center_distance_m <= 0.65
        and near_min >= 0.55
        and inside_min >= 0.35
        and inside_max >= 0.50
    ):
        return "duplicate_overlap_fragment"
    if (
        semantics_match
        and feature_cosine >= 0.875
        and near_min >= 0.55
        and inside_min >= 0.35
        and inside_max >= 0.40
        and center_distance_m <= 0.65
    ):
        return "duplicate_fragment"
    # A very thin track may be the top/face of a larger instance.  Require its
    # own points to be both close to and substantially inside the parent.
    records = (
        (first_thinness, first_volume, first_near_second, first_inside_second, second_volume),
        (second_thinness, second_volume, second_near_first, second_inside_first, first_volume),
    )
    if any(
        thinness <= 0.20
        and volume * 1.25 <= parent_volume
        and near_parent >= 0.65
        and inside_parent >= 0.40
        for thinness, volume, near_parent, inside_parent, parent_volume in records
    ):
        return "part_surface"
    return "distinct"


def _canonical_score(state: dict, index: int) -> float:
    def scalar(key: str, default: float = 0.0) -> float:
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0 and index < int(value.shape[0]):
            return float(value[index].item())
        return default

    def text(key: str) -> str:
        value = state.get(key) or []
        return str(value[index] if index < len(value) else "").strip().lower()

    tier_score = {"confirmed": 3.0, "probable": 2.0, "geometry_only": 0.5}.get(
        text("object_semantic_tier"), 0.0
    )
    category = text("object_category")
    resolved = bool(category and "unresolved" not in category and "unknown" not in category)
    return (
        10.0 * tier_score
        + (3.0 if resolved else 0.0)
        + 2.0 * math.log1p(max(scalar("object_evidence_count"), 0.0))
        + scalar("object_geometry_projected_box_iou")
        + scalar("object_geometry_inside_rate")
        + scalar("object_geometry_voxel_inside_rate")
    )


def resolve_duplicates(
    state: dict,
    *,
    near_distance_m: float = 0.20,
    mask_root: Path | None = None,
) -> tuple[dict, list[PairEvidence]]:
    active = _numpy(state["active"], bool).reshape(-1)
    object_ids = _numpy(state["object_id"], np.int64).reshape(-1)
    centers = _numpy(state["object_box_centers_m"], np.float64)
    dimensions = _numpy(state["object_box_dimensions_m"], np.float64)
    orientations = _numpy(state["object_box_wxyz"], np.float64)
    features = _numpy(state.get("features", np.zeros((active.size, 0))), np.float64)
    flat = _numpy(state["object_voxel_keys_flat"], np.int64)
    offsets = _numpy(state["object_voxel_keys_offsets"], np.int64)
    levels = _numpy(state["object_voxel_levels"], np.int64)
    statuses = state.get("object_geometry_status") or []
    categories = state.get("object_category") or []
    captions = state.get("object_caption") or []
    members = state.get("object_assembly_member_ids") or []
    cannot_links = state.get("cannot_link_object_ids") or {}
    mask_observations = state.get("object_mask_observations") or []
    mask_root = mask_root.expanduser().resolve() if mask_root is not None else None
    if features.ndim != 2 or features.shape[0] != active.size or features.shape[1] == 0:
        raise ValueError("Saved per-object visual features are required for fail-closed deduplication")
    feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(feature_norms, 1.0e-12)

    eligible: list[int] = []
    point_cache: dict[int, np.ndarray] = {}
    tree_cache: dict[int, cKDTree] = {}
    for index in np.flatnonzero(active).tolist():
        status = str(statuses[index] if index < len(statuses) else "").lower()
        assembly = index < len(members) and isinstance(members[index], (list, tuple)) and bool(members[index])
        if status != "geometry_pass" or assembly:
            continue
        start, end = int(offsets[index]), int(offsets[index + 1])
        points = np.asarray(decode_voxel_keys_numpy(flat[start:end], int(levels[index])), dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 16:
            continue
        eligible.append(index)
        point_cache[index] = points
        tree_cache[index] = cKDTree(points)

    parents = {index: index for index in eligible}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    evidence: list[PairEvidence] = []
    residual_candidates: list[dict] = []
    review_only_conflicts: list[dict] = []
    rejected_part_suppressions: list[dict] = []
    mask_evidence_source_counts: dict[str, int] = defaultdict(int)
    mask_cache: dict[Path, tuple[np.ndarray, np.ndarray, int] | None] = {}
    frame_cache = {index: _object_frame_ids(state, index) for index in eligible}
    eligible_object_ids = {int(object_ids[index]) for index in eligible}
    persisted_mask_candidates = _persisted_pair_mask_candidates(
        state, eligible_object_ids
    )
    spatial_pairs = _spatial_candidate_pairs(
        eligible=eligible,
        object_ids=object_ids,
        centers=centers,
        dimensions=dimensions,
        cannot_links=cannot_links,
        frame_cache=frame_cache,
        near_distance_m=near_distance_m,
    )
    eligible_by_id = {int(object_ids[index]): index for index in eligible}
    persisted_pairs: set[tuple[int, int]] = set()
    for object_id_pair in persisted_mask_candidates:
        pair_ids = sorted(object_id_pair)
        if len(pair_ids) != 2:
            continue
        first = eligible_by_id.get(int(pair_ids[0]))
        second = eligible_by_id.get(int(pair_ids[1]))
        if first is not None and second is not None:
            persisted_pairs.add(tuple(sorted((first, second))))
    candidate_neighbors: dict[int, list[int]] = defaultdict(list)
    for first, second in sorted(spatial_pairs.union(persisted_pairs)):
        candidate_neighbors[first].append(second)
    candidate_generation = {
        "spatial_pairs_evaluated": 0,
        "persisted_mask_pairs_available": len(persisted_mask_candidates),
        "persisted_mask_pairs_evaluated": 0,
        "persisted_mask_only_pairs_evaluated": 0,
        "union_pairs_evaluated": 0,
    }
    part_edges: list[dict] = []
    for first in eligible:
        for second in candidate_neighbors.get(first, []):
            first_id = int(object_ids[first])
            second_id = int(object_ids[second])
            pair_key = frozenset((first_id, second_id))
            persisted_mask_candidate = pair_key in persisted_mask_candidates
            cannot_link = _cannot_linked(cannot_links, first_id, second_id)
            shared_frames = frame_cache[first].intersection(frame_cache[second])
            track_evidence = _track_split_evidence(
                frame_cache[first], frame_cache[second]
            )
            co_visible = bool(shared_frames)
            scale_eligible = not cannot_link and not co_visible
            scale_candidate_distance, scale_near_distance = _scale_aware_distances(
                dimensions[first], dimensions[second], near_distance_m
            )
            candidate_distance = (
                scale_candidate_distance if scale_eligible else BASE_CANDIDATE_DISTANCE_M
            )
            center_distance = float(np.linalg.norm(centers[first] - centers[second]))
            feature_cosine = float(features[first] @ features[second])
            # Low global crop similarity is meaningful for duplicate merging,
            # but not for a strict contained-part test: a payload/face can look
            # very different from the whole that geometrically contains it.
            spatial_candidate = bool(
                np.isfinite(center_distance) and center_distance <= candidate_distance
            )
            if not np.isfinite(center_distance) or not (
                spatial_candidate or persisted_mask_candidate
            ):
                continue
            candidate_source = (
                "both"
                if spatial_candidate and persisted_mask_candidate
                else "spatial"
                if spatial_candidate
                else "persisted_pair_mask"
            )
            candidate_generation["union_pairs_evaluated"] += 1
            candidate_generation["spatial_pairs_evaluated"] += int(spatial_candidate)
            candidate_generation["persisted_mask_pairs_evaluated"] += int(
                persisted_mask_candidate
            )
            candidate_generation["persisted_mask_only_pairs_evaluated"] += int(
                persisted_mask_candidate and not spatial_candidate
            )
            first_points, second_points = point_cache[first], point_cache[second]
            first_distances = tree_cache[second].query(first_points, k=1)[0]
            second_distances = tree_cache[first].query(second_points, k=1)[0]
            first_near = float(np.mean(first_distances <= near_distance_m))
            second_near = float(np.mean(second_distances <= near_distance_m))
            first_near_scaled = float(np.mean(first_distances <= scale_near_distance))
            second_near_scaled = float(np.mean(second_distances <= scale_near_distance))
            first_local = (first_points - centers[second]) @ _wxyz_to_matrix(orientations[second])
            second_local = (second_points - centers[first]) @ _wxyz_to_matrix(orientations[first])
            first_inside = float(np.mean(np.all(np.abs(first_local) <= dimensions[second] / 2.0 + 0.025, axis=1)))
            second_inside = float(np.mean(np.all(np.abs(second_local) <= dimensions[first] / 2.0 + 0.025, axis=1)))
            first_volume = float(np.prod(dimensions[first]))
            second_volume = float(np.prod(dimensions[second]))
            first_thinness = float(np.min(dimensions[first]) / max(np.max(dimensions[first]), 1.0e-9))
            second_thinness = float(np.min(dimensions[second]) / max(np.max(dimensions[second]), 1.0e-9))
            first_category = categories[first] if first < len(categories) else ""
            second_category = categories[second] if second < len(categories) else ""
            first_category_tokens = _tokens(first_category)
            second_category_tokens = _tokens(second_category)
            categories_resolved = bool(
                first_category_tokens
                and second_category_tokens
                and not (first_category_tokens & {"unresolved", "unknown"})
                and not (second_category_tokens & {"unresolved", "unknown"})
            )
            category_agreement = _resolved_category_agreement(
                first_category, second_category
            )
            category_exact = _resolved_category_exact(
                first_category, second_category
            )
            semantics_match = categories_resolved and semantic_compatible(
                first_category,
                second_category,
                captions[first] if first < len(captions) else "",
                captions[second] if second < len(captions) else "",
            )
            relation = classify_pair(
                feature_cosine=feature_cosine,
                center_distance_m=center_distance,
                first_inside_second=first_inside,
                second_inside_first=second_inside,
                first_near_second=first_near,
                second_near_first=second_near,
                first_thinness=first_thinness,
                second_thinness=second_thinness,
                first_volume=first_volume,
                second_volume=second_volume,
                semantics_match=semantics_match,
            )
            relation_first_near = first_near
            relation_second_near = second_near
            if relation == "distinct" and scale_eligible:
                relation = _classify_scale_aware_duplicate(
                    feature_cosine=feature_cosine,
                    center_distance_m=center_distance,
                    candidate_distance_m=scale_candidate_distance,
                    first_near_second=first_near_scaled,
                    second_near_first=second_near_scaled,
                    first_inside_second=first_inside,
                    second_inside_first=second_inside,
                    first_volume=first_volume,
                    second_volume=second_volume,
                    semantics_match=semantics_match,
                )
                if relation == "duplicate_scale_aware":
                    relation_first_near = first_near_scaled
                    relation_second_near = second_near_scaled

            raw_relation = relation
            override_tier = ""
            if raw_relation == "distinct":
                geometry_relation = _classify_geometry_overwhelming_duplicate(
                    feature_cosine=feature_cosine,
                    center_distance_m=center_distance,
                    first_near_second=first_near,
                    second_near_first=second_near,
                    first_inside_second=first_inside,
                    second_inside_first=second_inside,
                    first_volume=first_volume,
                    second_volume=second_volume,
                    first_dimensions=dimensions[first],
                    second_dimensions=dimensions[second],
                    cannot_link=cannot_link,
                    co_visible=co_visible,
                )
                if geometry_relation is not None:
                    relation = geometry_relation
                    override_tier = "geometry_overwhelming"

            first_rows = mask_observations[first] if first < len(mask_observations) else []
            second_rows = mask_observations[second] if second < len(mask_observations) else []
            mask_evidence = same_frame_mask_evidence(
                first_id=first_id,
                second_id=second_id,
                first_rows=first_rows,
                second_rows=second_rows,
                mask_root=mask_root,
                mask_cache=mask_cache,
                saved_pair_evidence=get_pairwise_mask_overlap_records(
                    state, first_id, second_id
                ),
            )
            mask_evidence_source_counts[mask_evidence.source] += 1
            three_d_corroborated = _strong_bidirectional_3d(
                first_near_second=first_near,
                second_near_first=second_near,
                first_inside_second=first_inside,
                second_inside_first=second_inside,
                first_volume=first_volume,
                second_volume=second_volume,
            )
            mask_override_tier = classify_same_frame_duplicate_override(
                mask_evidence,
                feature_cosine=feature_cosine,
                three_d_corroborated=three_d_corroborated,
                semantic_compatible=semantics_match,
                center_distance_m=center_distance,
                raw_relation=raw_relation,
                first_near_second=first_near,
                second_near_first=second_near,
                first_inside_second=first_inside,
                second_inside_first=second_inside,
                first_volume=first_volume,
                second_volume=second_volume,
            )
            large_object_mask_relation = _classify_large_object_same_mask_override(
                mask_evidence,
                feature_cosine=feature_cosine,
                center_distance_m=center_distance,
                first_dimensions=dimensions[first],
                second_dimensions=dimensions[second],
                semantic_compatible=semantics_match,
                raw_relation=raw_relation,
            )
            track_fragment_tier = None
            if (
                large_object_mask_relation is None
                and mask_override_tier is None
                and (cannot_link or co_visible)
            ):
                track_fragment_tier = _classify_same_frame_track_fragment_override(
                    mask_evidence,
                    track_evidence,
                    feature_cosine=feature_cosine,
                    center_distance_m=center_distance,
                    first_dimensions=dimensions[first],
                    second_dimensions=dimensions[second],
                    first_near_second_scaled=first_near_scaled,
                    second_near_first_scaled=second_near_scaled,
                    first_inside_second=first_inside,
                    second_inside_first=second_inside,
                    categories_resolved=categories_resolved,
                    category_agreement=category_agreement,
                    category_exact=category_exact,
                    raw_relation=raw_relation,
                )
            cannot_link_overridden = False
            rejection_reason = ""
            if large_object_mask_relation is not None:
                relation = large_object_mask_relation
                override_tier = "large_object_same_mask"
                cannot_link_overridden = cannot_link
            elif track_fragment_tier is not None:
                relation = "duplicate_same_frame_track_fragment_override"
                override_tier = f"track_fragment_{track_fragment_tier}"
                cannot_link_overridden = cannot_link
            elif cannot_link and relation not in {"part_surface", "part_contained"}:
                if mask_override_tier is not None:
                    relation = {
                        "strict": "duplicate_same_frame_mask_override",
                        "relaxed_3d": "duplicate_same_frame_mask_relaxed_3d_override",
                        "overwhelming_2d": "duplicate_same_frame_mask_overwhelming_override",
                    }[mask_override_tier]
                    override_tier = mask_override_tier
                    cannot_link_overridden = True
                else:
                    relation = "distinct"
                    if mask_evidence.compared_frames == 0:
                        rejection_reason = "cannot_link_without_comparable_same_frame_mask"
                    elif raw_relation == "distinct":
                        rejection_reason = "cannot_link_raw_relation_distinct"
                    else:
                        rejection_reason = "cannot_link_mask_override_tier_not_met"
            elif co_visible and relation.startswith("duplicate_"):
                if mask_override_tier is not None:
                    relation = {
                        "strict": "duplicate_same_frame_mask_override",
                        "relaxed_3d": "duplicate_same_frame_mask_relaxed_3d_override",
                        "overwhelming_2d": "duplicate_same_frame_mask_overwhelming_override",
                    }[mask_override_tier]
                    override_tier = mask_override_tier
                else:
                    relation = "distinct"
                    rejection_reason = (
                        "co_visible_without_comparable_same_frame_mask"
                        if mask_evidence.compared_frames == 0
                        else "co_visible_mask_threshold_not_met"
                    )
            if relation == "distinct" and _is_overwhelming_mask_geometry_conflict(
                mask_evidence,
                feature_cosine=feature_cosine,
                raw_relation=raw_relation,
            ):
                review_only_conflicts.append({
                    "first_id": first_id,
                    "second_id": second_id,
                    "raw_relation": raw_relation,
                    "feature_cosine": feature_cosine,
                    "center_distance_m": center_distance,
                    "mask_image_id": mask_evidence.image_id,
                    "mask_iou": mask_evidence.iou,
                    "mask_containment": mask_evidence.containment,
                    "mask_area_ratio": mask_evidence.area_ratio,
                    "mask_evidence_source": mask_evidence.source,
                    "candidate_source": candidate_source,
                    "disposition": "review_only_no_auto_merge",
                    "reason": "overwhelming_mask_geometry_conflict",
                })
            elif (
                relation == "distinct"
                and raw_relation == "duplicate_overlap_fragment"
                and (cannot_link or co_visible)
            ):
                review_only_conflicts.append({
                    "first_id": first_id,
                    "second_id": second_id,
                    "raw_relation": raw_relation,
                    "feature_cosine": feature_cosine,
                    "center_distance_m": center_distance,
                    "mask_image_id": mask_evidence.image_id,
                    "mask_iou": mask_evidence.iou,
                    "mask_containment": mask_evidence.containment,
                    "mask_area_ratio": mask_evidence.area_ratio,
                    "mask_evidence_source": mask_evidence.source,
                    "candidate_source": candidate_source,
                    "semantic_compatible": semantics_match,
                    "cannot_link": cannot_link,
                    "co_visible": co_visible,
                    "disposition": "review_only_no_auto_merge",
                    "reason": "duplicate_overlap_fragment_track_or_semantic_conflict",
                })
            if relation == "distinct":
                residual_near = max(first_near_scaled, second_near_scaled)
                residual_inside = max(first_inside, second_inside)
                if feature_cosine >= 0.90 and (
                    (residual_near >= 0.60 and residual_inside >= 0.60)
                    or mask_evidence.iou >= 0.50
                ):
                    residual_candidates.append(
                        {
                            "first_id": first_id,
                            "second_id": second_id,
                            "feature_cosine": feature_cosine,
                            "center_distance_m": center_distance,
                            "candidate_distance_m": candidate_distance,
                            "base_near_distance_m": float(near_distance_m),
                            "scale_near_distance_m": scale_near_distance,
                            "first_near_second": first_near_scaled,
                            "second_near_first": second_near_scaled,
                            "first_inside_second": first_inside,
                            "second_inside_first": second_inside,
                            "semantic_compatible": semantics_match,
                            "cannot_link": cannot_link,
                            "co_visible": co_visible,
                            "common_frames": len(shared_frames),
                            "first_track_views": track_evidence.first_views,
                            "second_track_views": track_evidence.second_views,
                            "common_over_min_track": track_evidence.common_over_min,
                            "track_view_count_ratio": (
                                max(track_evidence.first_views, track_evidence.second_views)
                                / max(min(track_evidence.first_views, track_evidence.second_views), 1)
                            ),
                            "normalized_center_distance": (
                                center_distance
                                / max(
                                    min(
                                        float(np.linalg.norm(dimensions[first])),
                                        float(np.linalg.norm(dimensions[second])),
                                    ),
                                    1.0e-12,
                                )
                            ),
                            "exact_resolved_category": category_exact,
                            "mask_image_id": mask_evidence.image_id,
                            "mask_iou": mask_evidence.iou,
                            "mask_containment": mask_evidence.containment,
                            "mask_area_ratio": mask_evidence.area_ratio,
                            "mask_evidence_source": mask_evidence.source,
                            "mask_evidence_fingerprint": mask_evidence.evidence_fingerprint,
                            "candidate_source": candidate_source,
                            "three_d_corroborated": three_d_corroborated,
                            "raw_relation": raw_relation,
                            "rejection_reason": rejection_reason
                            or "strong_visual_geometry_below_merge_threshold",
                        }
                    )
                continue
            evidence.append(
                PairEvidence(
                    first_id=first_id, second_id=second_id,
                    feature_cosine=feature_cosine, center_distance_m=center_distance,
                    first_inside_second=first_inside, second_inside_first=second_inside,
                    first_near_second=relation_first_near,
                    second_near_first=relation_second_near,
                    first_thinness=first_thinness, second_thinness=second_thinness,
                    semantic_compatible=semantics_match, cannot_link=cannot_link,
                    cannot_link_overridden=cannot_link_overridden,
                    three_d_corroborated=three_d_corroborated,
                    candidate_distance_m=candidate_distance,
                    near_distance_m=(
                        scale_near_distance
                        if relation == "duplicate_scale_aware"
                        else float(near_distance_m)
                    ),
                    common_frames=len(shared_frames),
                    first_track_views=track_evidence.first_views,
                    second_track_views=track_evidence.second_views,
                    common_over_min_track=track_evidence.common_over_min,
                    track_view_count_ratio=(
                        max(track_evidence.first_views, track_evidence.second_views)
                        / max(min(track_evidence.first_views, track_evidence.second_views), 1)
                    ),
                    normalized_center_distance=(
                        center_distance
                        / max(
                            min(
                                float(np.linalg.norm(dimensions[first])),
                                float(np.linalg.norm(dimensions[second])),
                            ),
                            1.0e-12,
                        )
                    ),
                    exact_resolved_category=category_exact,
                    compared_mask_frames=mask_evidence.compared_frames,
                    mask_image_id=mask_evidence.image_id,
                    mask_iou=mask_evidence.iou,
                    mask_containment=mask_evidence.containment,
                    mask_area_ratio=mask_evidence.area_ratio,
                    mask_evidence_source=mask_evidence.source,
                    mask_evidence_fingerprint=mask_evidence.evidence_fingerprint,
                    candidate_source=candidate_source,
                    raw_relation=raw_relation,
                    override_tier=override_tier,
                    relation=relation,
                )
            )
            if relation in {"part_surface", "part_contained"}:
                if relation == "part_contained":
                    first_is_child = bool(
                        first_volume * 2.0 <= second_volume
                        and first_near >= 0.90
                        and first_inside >= 0.85
                    )
                else:
                    first_is_child = bool(
                        first_thinness <= 0.20
                        and first_volume * 1.25 <= second_volume
                        and first_near >= 0.65
                        and first_inside >= 0.40
                    )
                child, parent = (first, second) if first_is_child else (second, first)
                if not semantics_match or cannot_link:
                    rejected_part_suppressions.append({
                        "child_id": int(object_ids[child]),
                        "parent_id": int(object_ids[parent]),
                        "relation": relation,
                        "semantic_compatible": semantics_match,
                        "cannot_link": cannot_link,
                        "explicit_part_whole_evidence": False,
                        "reason": (
                            "cannot_link"
                            if cannot_link
                            else "semantic_incompatible"
                        ),
                    })
                else:
                    part_edges.append({
                        "child": child,
                        "parent": parent,
                        "relation": relation,
                        "semantic_compatible": semantics_match,
                        "cannot_link": cannot_link,
                    })
            else:
                union(first, second)

    grouped: dict[int, list[int]] = {}
    for index in eligible:
        grouped.setdefault(find(index), []).append(index)
    components = [indices for indices in grouped.values() if len(indices) > 1]
    display_status = ["inactive" if not bool(active[i]) else "canonical" for i in range(active.size)]
    canonical_ids = np.full((active.size,), -1, dtype=np.int64)
    group_ids: list[list[int]] = [[] for _ in range(active.size)]
    reasons = ["" for _ in range(active.size)]
    evidence_by_pair = {
        frozenset((row.first_id, row.second_id)): row.relation for row in evidence
    }
    groups_report: list[dict] = []
    for indices in components:
        canonical = max(indices, key=lambda index: (_canonical_score(state, index), -int(object_ids[index])))
        ids = sorted(int(object_ids[index]) for index in indices)
        for index in indices:
            group_ids[index] = ids
            if index == canonical:
                continue
            display_status[index] = "duplicate_suppressed"
            canonical_ids[index] = int(object_ids[canonical])
            pair_reason = evidence_by_pair.get(
                frozenset((int(object_ids[index]), int(object_ids[canonical]))),
                "duplicate_transitive",
            )
            reasons[index] = pair_reason
        groups_report.append(
            {
                "canonical_id": int(object_ids[canonical]),
                "member_ids": ids,
                "suppressed_ids": [value for value in ids if value != int(object_ids[canonical])],
                "canonical_score": _canonical_score(state, canonical),
            }
        )

    index_by_id = {int(object_id): index for index, object_id in enumerate(object_ids.tolist())}
    part_report: list[dict] = []
    for edge in part_edges:
        child = int(edge["child"])
        parent = int(edge["parent"])
        relation = str(edge["relation"])
        if display_status[child] == "duplicate_suppressed":
            continue
        parent_canonical_id = (
            int(canonical_ids[parent])
            if int(canonical_ids[parent]) >= 0
            else int(object_ids[parent])
        )
        display_parent = index_by_id.get(parent_canonical_id)
        if display_parent is None or not _presentation_eligible_direct(
            state, display_parent, active=active, members=members
        ):
            rejected_part_suppressions.append({
                "child_id": int(object_ids[child]),
                "parent_id": int(object_ids[parent]),
                "display_parent_id": parent_canonical_id,
                "relation": relation,
                "semantic_compatible": bool(edge["semantic_compatible"]),
                "cannot_link": bool(edge["cannot_link"]),
                "explicit_part_whole_evidence": False,
                "reason": "parent_not_presentation_eligible",
            })
            continue
        child_id = int(object_ids[child])
        display_parent_category = categories[display_parent] if display_parent < len(categories) else ""
        child_category = categories[child] if child < len(categories) else ""
        display_tokens = _tokens(display_parent_category)
        child_tokens = _tokens(child_category)
        display_semantics_resolved = bool(
            display_tokens
            and child_tokens
            and not (display_tokens & {"unresolved", "unknown"})
            and not (child_tokens & {"unresolved", "unknown"})
        )
        display_semantics_match = display_semantics_resolved and semantic_compatible(
            child_category,
            display_parent_category,
            captions[child] if child < len(captions) else "",
            captions[display_parent] if display_parent < len(captions) else "",
        )
        display_cannot_link = _cannot_linked(
            cannot_links, child_id, parent_canonical_id
        )
        if not display_semantics_match or display_cannot_link:
            rejected_part_suppressions.append({
                "child_id": child_id,
                "parent_id": int(object_ids[parent]),
                "display_parent_id": parent_canonical_id,
                "relation": relation,
                "semantic_compatible": display_semantics_match,
                "cannot_link": display_cannot_link,
                "explicit_part_whole_evidence": False,
                "reason": (
                    "redirected_parent_cannot_link"
                    if display_cannot_link
                    else "redirected_parent_semantic_incompatible"
                ),
            })
            continue
        display_status[child] = "part_suppressed"
        canonical_ids[child] = parent_canonical_id
        reasons[child] = relation
        group_ids[child] = sorted({int(object_ids[child]), int(object_ids[parent]), parent_canonical_id})
        part_report.append(
            {
                "child_id": int(object_ids[child]),
                "parent_id": int(object_ids[parent]),
                "display_parent_id": parent_canonical_id,
                "relation": relation,
            }
        )

    # A validated whole and its direct parts are both valuable audit rows, but
    # rendering them simultaneously produces the exact double-box failure the
    # presentation layer is meant to avoid.  Suppress members only when the
    # parent has both semantic provenance and either strict metric assembly
    # geometry or a well-formed probable compound.  This is display metadata;
    # the active metric state and every diagnostic layer remain untouched.
    compound_boxes = state.get("object_compound_boxes") or []
    unified_compound_ids = {
        int(value) for value in state.get("object_compound_presentation_unified_ids", [])
    }
    decisions = state.get("object_caption_decision") or []
    assembly_candidates: dict[int, list[int]] = {}
    for parent_index in range(active.size):
        member_ids = members[parent_index] if parent_index < len(members) else []
        if not isinstance(member_ids, (list, tuple)) or not member_ids:
            continue
        status = str(statuses[parent_index] if parent_index < len(statuses) else "").lower()
        category = str(categories[parent_index] if parent_index < len(categories) else "").strip().lower()
        decision = str(decisions[parent_index] if parent_index < len(decisions) else "").strip().lower()
        semantic_keep = bool(
            category and "unresolved" not in category and "unknown" not in category
            and decision in {"keep", "relabel"}
        )
        metric_parent = bool(active[parent_index]) and status == "assembly_geometry_pass"
        records = compound_boxes[parent_index] if parent_index < len(compound_boxes) else []
        probable_parent = (
            "compound_geometry_probable" in status
            and "rejected" not in status
            and isinstance(records, (list, tuple))
            and (
                len(records) == 2
                or (len(records) == 1 and int(object_ids[parent_index]) in unified_compound_ids)
            )
        )
        if not semantic_keep or not (metric_parent or probable_parent):
            continue
        for member_id in member_ids:
            try:
                assembly_candidates.setdefault(int(member_id), []).append(parent_index)
            except (TypeError, ValueError):
                continue

    assembly_report: list[dict] = []
    parent_to_members: dict[int, list[int]] = {}
    for member_id, parent_indices in sorted(assembly_candidates.items()):
        member_index = index_by_id.get(member_id)
        if member_index is None:
            continue
        parent_index = max(
            parent_indices,
            key=lambda index: (
                bool(active[index]), _canonical_score(state, index), -int(object_ids[index])
            ),
        )
        parent_id = int(object_ids[parent_index])
        display_status[member_index] = "assembly_member_suppressed"
        canonical_ids[member_index] = parent_id
        reasons[member_index] = "validated_assembly_member"
        group_ids[member_index] = sorted({member_id, parent_id})
        parent_to_members.setdefault(parent_id, []).append(member_id)
    for parent_id, member_ids in sorted(parent_to_members.items()):
        assembly_report.append({
            "assembly_id": parent_id,
            "suppressed_member_ids": sorted(set(member_ids)),
            "relation": "validated_assembly_member",
        })

    # A rejected synthetic assembly still preserves useful provenance: its
    # member set came from accepted multi-view part/whole relations.  Transfer
    # that lineage to a surviving direct whole only when an assembly anchor
    # and the direct whole have overwhelming saved same-frame mask agreement.
    # This recovers the hierarchy without reviving rejected geometry or using
    # any scene/category-specific rules.
    assembly_containment = state.get("object_assembly_containment")
    assembly_shared_frames = state.get("object_assembly_shared_frames")
    assembly_relations = state.get("object_assembly_relation_type") or []
    rejected_assembly_lineage: list[dict] = []
    for assembly_index in range(active.size):
        member_ids_raw = members[assembly_index] if assembly_index < len(members) else []
        if not isinstance(member_ids_raw, (list, tuple)) or len(member_ids_raw) < 2:
            continue
        assembly_status = _text_at(state, "object_geometry_status", assembly_index).lower()
        assembly_relation = str(
            assembly_relations[assembly_index]
            if assembly_index < len(assembly_relations)
            else ""
        ).lower()
        if (
            bool(active[assembly_index])
            or "assembly" not in assembly_status
            or "rejected" not in assembly_status
            or assembly_relation != "part_whole_candidate"
            or not _resolved_semantic_keep(state, assembly_index)
        ):
            continue
        containment = _scalar_at(
            {"values": assembly_containment}, "values", assembly_index
        ) if assembly_containment is not None else 0.0
        shared_count = int(_scalar_at(
            {"values": assembly_shared_frames}, "values", assembly_index
        )) if assembly_shared_frames is not None else 0
        if (
            containment < EXPLICIT_ASSEMBLY_MIN_CONTAINMENT
            or shared_count < EXPLICIT_ASSEMBLY_MIN_SHARED_FRAMES
        ):
            continue
        member_indices = [
            index_by_id[value]
            for value in (int(raw) for raw in member_ids_raw)
            if value in index_by_id
        ]
        if len(member_indices) < 2:
            continue
        anchor = max(
            member_indices,
            key=lambda index: (
                _scalar_at(state, "object_evidence_count", index),
                float(np.prod(np.maximum(dimensions[index], 0.0))),
                -int(object_ids[index]),
            ),
        )
        anchor_id = int(object_ids[anchor])
        assembly_category = categories[assembly_index] if assembly_index < len(categories) else ""
        assembly_caption = captions[assembly_index] if assembly_index < len(captions) else ""
        member_id_set = {int(object_ids[index]) for index in member_indices}
        matches: list[tuple[tuple[float, ...], int, SameFrameMaskEvidence, float, float]] = []
        for whole in eligible:
            whole_id = int(object_ids[whole])
            if whole_id in member_id_set or not _presentation_eligible_direct(
                state, whole, active=active, members=members
            ):
                continue
            if not semantic_compatible(
                assembly_category,
                categories[whole] if whole < len(categories) else "",
                assembly_caption,
                captions[whole] if whole < len(captions) else "",
            ):
                continue
            assembly_distance = float(np.linalg.norm(centers[assembly_index] - centers[whole]))
            if (
                not np.isfinite(assembly_distance)
                or assembly_distance > LINEAGE_MAX_ASSEMBLY_CENTER_DISTANCE_M
            ):
                continue
            anchor_feature = float(features[anchor] @ features[whole])
            if anchor_feature < LINEAGE_MIN_ANCHOR_FEATURE_COSINE:
                continue
            anchor_volume = float(np.prod(np.maximum(dimensions[anchor], 0.0)))
            whole_volume = float(np.prod(np.maximum(dimensions[whole], 0.0)))
            volume_ratio = min(anchor_volume, whole_volume) / max(
                anchor_volume, whole_volume, 1.0e-12
            )
            if volume_ratio < 0.50:
                continue
            anchor_mask = _saved_pair_mask_evidence(
                get_pairwise_mask_overlap_records(state, anchor_id, whole_id),
                first_id=anchor_id,
                second_id=whole_id,
                common_frames=0,
            )
            if (
                anchor_mask.image_id is None
                or anchor_mask.iou < MASK_OVERWHELMING_MIN_IOU
                or anchor_mask.containment < MASK_OVERWHELMING_MIN_CONTAINMENT
                or anchor_mask.area_ratio < MASK_OVERWHELMING_MIN_AREA_RATIO
            ):
                continue
            rank = (
                anchor_mask.iou,
                anchor_mask.containment,
                anchor_mask.area_ratio,
                anchor_feature,
                volume_ratio,
                -assembly_distance,
                _canonical_score(state, whole),
                -whole_id,
            )
            matches.append((rank, whole, anchor_mask, anchor_feature, volume_ratio))

        lineage_row = {
            "assembly_id": int(object_ids[assembly_index]),
            "assembly_status": assembly_status,
            "assembly_relation": assembly_relation,
            "assembly_containment": containment,
            "assembly_shared_frames": shared_count,
            "anchor_id": anchor_id,
            "member_ids": sorted(member_id_set),
            "disposition": "no_verified_surviving_whole",
            "suppressed_member_ids": [],
            "rejected_member_ids": [],
        }
        if not matches:
            rejected_assembly_lineage.append(lineage_row)
            continue
        _, whole, anchor_mask, anchor_feature, anchor_volume_ratio = max(
            matches, key=lambda row: row[0]
        )
        whole_id = int(object_ids[whole])
        suppressed_members: list[int] = []
        rejected_members: list[dict] = []
        _, child_near_distance = _scale_aware_distances(
            dimensions[whole], dimensions[whole], near_distance_m
        )
        whole_rotation = _wxyz_to_matrix(orientations[whole])
        whole_volume = float(np.prod(np.maximum(dimensions[whole], 0.0)))
        for child in member_indices:
            child_id = int(object_ids[child])
            if child == anchor or child_id == whole_id or child not in point_cache:
                continue
            child_points = point_cache[child]
            child_distances = tree_cache[whole].query(child_points, k=1)[0]
            child_near = float(np.mean(child_distances <= child_near_distance))
            child_local = (child_points - centers[whole]) @ whole_rotation
            child_inside = float(np.mean(np.all(
                np.abs(child_local) <= dimensions[whole] / 2.0 + 0.025, axis=1
            )))
            child_volume = float(np.prod(np.maximum(dimensions[child], 0.0)))
            volume_ratio = child_volume / max(whole_volume, 1.0e-12)
            accepted = bool(
                volume_ratio <= LINEAGE_MAX_CHILD_PARENT_VOLUME_RATIO
                and child_near >= LINEAGE_MIN_CHILD_NEAR
                and child_inside >= LINEAGE_MIN_CHILD_INSIDE
            )
            if not accepted:
                rejected_members.append({
                    "member_id": child_id,
                    "child_near": child_near,
                    "child_inside": child_inside,
                    "child_parent_volume_ratio": volume_ratio,
                    "reason": "strict_contained_part_threshold_not_met",
                })
                continue
            if display_status[child] == "duplicate_suppressed":
                rejected_members.append({
                    "member_id": child_id,
                    "reason": "already_duplicate_suppressed",
                })
                continue
            display_status[child] = "assembly_member_suppressed"
            canonical_ids[child] = whole_id
            reasons[child] = "rejected_assembly_lineage_member"
            group_ids[child] = sorted({child_id, anchor_id, whole_id})
            suppressed_members.append(child_id)
        lineage_row.update({
            "disposition": "linked_to_verified_surviving_whole",
            "surviving_whole_id": whole_id,
            "anchor_feature_cosine": anchor_feature,
            "anchor_volume_ratio": anchor_volume_ratio,
            "anchor_mask_image_id": anchor_mask.image_id,
            "anchor_mask_iou": anchor_mask.iou,
            "anchor_mask_containment": anchor_mask.containment,
            "anchor_mask_area_ratio": anchor_mask.area_ratio,
            "anchor_mask_evidence_source": anchor_mask.source,
            "anchor_mask_evidence_fingerprint": anchor_mask.evidence_fingerprint,
            "suppressed_member_ids": sorted(suppressed_members),
            "rejected_member_ids": rejected_members,
        })
        rejected_assembly_lineage.append(lineage_row)

    state["object_display_status"] = display_status
    state["object_duplicate_canonical_id"] = torch.as_tensor(canonical_ids, dtype=torch.int64)
    state["object_duplicate_group_ids"] = group_ids
    state["object_duplicate_reason"] = reasons
    residual_candidates.sort(
        key=lambda row: (
            -float(row["feature_cosine"]),
            -max(float(row["first_near_second"]), float(row["second_near_first"])),
            float(row["center_distance_m"]),
            int(row["first_id"]),
            int(row["second_id"]),
        )
    )
    track_fragment_override_counts = {
        tier: sum(row.override_tier == tier for row in evidence)
        for tier in (
            "track_fragment_resolved_same_category",
            "track_fragment_resolved_alias",
            "track_fragment_same_category_large_track_split",
        )
    }
    report = {
        "schema": "farm.track-dedup.v2",
        "active_metric_objects": int(active.sum()),
        "eligible_direct_objects": len(eligible),
        "duplicate_groups": sorted(groups_report, key=lambda row: row["canonical_id"]),
        "part_hierarchy": sorted(part_report, key=lambda row: row["child_id"]),
        "rejected_part_suppressions": sorted(
            rejected_part_suppressions,
            key=lambda row: (int(row["child_id"]), int(row["parent_id"])),
        ),
        "assembly_hierarchy": assembly_report,
        "rejected_assembly_lineage": rejected_assembly_lineage,
        "suppressed_duplicates": int(sum(status == "duplicate_suppressed" for status in display_status)),
        "suppressed_parts": int(sum(status == "part_suppressed" for status in display_status)),
        "suppressed_assembly_members": int(sum(status == "assembly_member_suppressed" for status in display_status)),
        "suppressed_objects": int(sum(status.endswith("_suppressed") for status in display_status)),
        "residual_strong_candidate_count": len(residual_candidates),
        "residual_strong_candidates": residual_candidates[:100],
        "review_only_conflicts": sorted(
            review_only_conflicts,
            key=lambda row: (int(row["first_id"]), int(row["second_id"])),
        ),
        "track_fragment_override_counts": track_fragment_override_counts,
        "candidate_generation": candidate_generation,
        "mask_evidence_source_counts": dict(sorted(mask_evidence_source_counts.items())),
        "decision_thresholds": {
            "base_candidate_distance_m": BASE_CANDIDATE_DISTANCE_M,
            "max_scale_candidate_distance_m": MAX_SCALE_CANDIDATE_DISTANCE_M,
            "max_scale_near_distance_m": MAX_SCALE_NEAR_DISTANCE_M,
            "cannot_link_override": {
                "min_iou": MASK_OVERRIDE_MIN_IOU,
                "min_containment": MASK_OVERRIDE_MIN_CONTAINMENT,
                "min_area_ratio": MASK_OVERRIDE_MIN_AREA_RATIO,
                "min_feature_cosine": MASK_OVERRIDE_MIN_FEATURE_COSINE,
                "requires_3d_corroboration": True,
            },
            "cannot_link_override_relaxed_3d": {
                "raw_relations": ["duplicate_strict", "duplicate_geometric"],
                "min_iou": MASK_RELAXED_MIN_IOU,
                "min_containment": MASK_RELAXED_MIN_CONTAINMENT,
                "min_area_ratio": MASK_RELAXED_MIN_AREA_RATIO,
                "min_feature_cosine": MASK_RELAXED_MIN_FEATURE_COSINE,
                "max_center_distance_m": MASK_RELAXED_MAX_CENTER_DISTANCE_M,
                "requires_3d_corroboration": True,
                "requires_semantic_compatibility": True,
            },
            "cannot_link_override_overwhelming_2d": {
                "min_iou": MASK_OVERWHELMING_MIN_IOU,
                "min_containment": MASK_OVERWHELMING_MIN_CONTAINMENT,
                "min_area_ratio": MASK_OVERWHELMING_MIN_AREA_RATIO,
                "min_feature_cosine": MASK_OVERWHELMING_MIN_FEATURE_COSINE,
                "max_center_distance_m": MASK_OVERWHELMING_MAX_CENTER_DISTANCE_M,
                "min_near": 0.70,
                "min_inside": 0.55,
                "max_inside": 0.85,
                "min_volume_ratio": 0.50,
            },
            "geometry_overwhelming_disjoint_history": {
                "min_feature_cosine": GEOMETRY_OVERWHELMING_MIN_FEATURE_COSINE,
                "max_center_distance_m": GEOMETRY_OVERWHELMING_MAX_CENTER_DISTANCE_M,
                "max_normalized_center_distance": GEOMETRY_OVERWHELMING_MAX_NORMALIZED_CENTER_DISTANCE,
                "min_near": GEOMETRY_OVERWHELMING_MIN_NEAR,
                "min_inside": GEOMETRY_OVERWHELMING_MIN_INSIDE,
                "max_inside": GEOMETRY_OVERWHELMING_MAX_INSIDE,
                "min_volume_ratio": GEOMETRY_OVERWHELMING_MIN_VOLUME_RATIO,
                "min_axis_ratio": GEOMETRY_OVERWHELMING_MIN_AXIS_RATIO,
                "requires_disjoint_history": True,
            },
            "part_suppression": {
                "requires_semantic_compatibility": True,
                "rejects_cannot_link": True,
                "requires_presentation_eligible_direct_parent": True,
                "semantic_incompatible_exception": "strict_explicit_assembly_lineage_only",
            },
            "rejected_assembly_lineage": {
                "min_assembly_containment": EXPLICIT_ASSEMBLY_MIN_CONTAINMENT,
                "min_assembly_shared_frames": EXPLICIT_ASSEMBLY_MIN_SHARED_FRAMES,
                "min_anchor_feature_cosine": LINEAGE_MIN_ANCHOR_FEATURE_COSINE,
                "max_assembly_center_distance_m": LINEAGE_MAX_ASSEMBLY_CENTER_DISTANCE_M,
                "min_anchor_mask_iou": MASK_OVERWHELMING_MIN_IOU,
                "min_anchor_mask_containment": MASK_OVERWHELMING_MIN_CONTAINMENT,
                "min_anchor_mask_area_ratio": MASK_OVERWHELMING_MIN_AREA_RATIO,
                "min_child_near": LINEAGE_MIN_CHILD_NEAR,
                "min_child_inside": LINEAGE_MIN_CHILD_INSIDE,
                "max_child_parent_volume_ratio": LINEAGE_MAX_CHILD_PARENT_VOLUME_RATIO,
            },
            "persisted_mask_candidate_union": {
                "min_iou": PERSISTED_MASK_CANDIDATE_MIN_IOU,
                "min_containment": PERSISTED_MASK_CANDIDATE_MIN_CONTAINMENT,
                "min_area_ratio": PERSISTED_MASK_CANDIDATE_MIN_AREA_RATIO,
                "raw_distinct_is_review_only": True,
            },
            "same_frame_track_fragment": {
                "requires_resolved_category_agreement": True,
                "min_iou": TRACK_FRAGMENT_MIN_IOU,
                "min_containment": TRACK_FRAGMENT_MIN_CONTAINMENT,
                "min_area_ratio": TRACK_FRAGMENT_MIN_AREA_RATIO,
                "min_feature_cosine": TRACK_FRAGMENT_MIN_FEATURE_COSINE,
                "max_normalized_center_distance": TRACK_FRAGMENT_MAX_NORMALIZED_CENTER_DISTANCE,
                "min_max_scale_near": TRACK_FRAGMENT_MIN_MAX_SCALE_NEAR,
                "min_max_inside": TRACK_FRAGMENT_MIN_MAX_INSIDE,
                "min_volume_ratio": TRACK_FRAGMENT_MIN_VOLUME_RATIO,
                "min_sorted_axis_ratio": TRACK_FRAGMENT_MIN_SORTED_AXIS_RATIO,
                "max_common_over_min_track": TRACK_FRAGMENT_MAX_COMMON_OVER_MIN,
                "raw_part_surface_allowed": False,
                "presentation_only": True,
            },
            "same_frame_resolved_alias_fragment": {
                "raw_relation": "duplicate_overlap_fragment",
                "requires_two_resolved_categories": True,
                "requires_category_disagreement": True,
                "min_iou": TRACK_ALIAS_MIN_IOU,
                "min_containment": TRACK_ALIAS_MIN_CONTAINMENT,
                "min_area_ratio": TRACK_ALIAS_MIN_AREA_RATIO,
                "min_feature_cosine": TRACK_ALIAS_MIN_FEATURE_COSINE,
                "max_normalized_center_distance": TRACK_ALIAS_MAX_NORMALIZED_CENTER_DISTANCE,
                "min_volume_ratio": TRACK_ALIAS_MIN_VOLUME_RATIO,
                "min_sorted_axis_ratio": TRACK_ALIAS_MIN_SORTED_AXIS_RATIO,
                "min_scale_near": TRACK_ALIAS_MIN_SCALE_NEAR,
                "min_inside": TRACK_ALIAS_MIN_INSIDE,
                "min_max_inside": TRACK_ALIAS_MIN_MAX_INSIDE,
                "max_common_over_min_track": TRACK_ALIAS_MAX_COMMON_OVER_MIN,
                "presentation_only": True,
            },
            "same_category_large_track_split": {
                "raw_relation": "distinct",
                "requires_exact_resolved_category": True,
                "min_iou": TRACK_SPLIT_EXACT_MIN_IOU,
                "min_containment": TRACK_SPLIT_EXACT_MIN_CONTAINMENT,
                "min_area_ratio": TRACK_SPLIT_EXACT_MIN_AREA_RATIO,
                "min_feature_cosine": TRACK_SPLIT_EXACT_MIN_FEATURE_COSINE,
                "max_normalized_center_distance": TRACK_SPLIT_EXACT_MAX_NORMALIZED_CENTER_DISTANCE,
                "min_track_views": TRACK_SPLIT_EXACT_MIN_TRACK_VIEWS,
                "min_view_count_ratio": TRACK_SPLIT_EXACT_MIN_VIEW_COUNT_RATIO,
                "min_common_views": TRACK_SPLIT_EXACT_MIN_COMMON_VIEWS,
                "max_common_over_min_track": TRACK_SPLIT_EXACT_MAX_COMMON_OVER_MIN,
                "presentation_only": True,
            },
            "large_object_same_mask_override": {
                "raw_relation": "distinct",
                "min_iou": MASK_OVERWHELMING_MIN_IOU,
                "min_containment": MASK_OVERWHELMING_MIN_CONTAINMENT,
                "min_area_ratio": MASK_OVERWHELMING_MIN_AREA_RATIO,
                "min_feature_cosine": LARGE_MASK_MIN_FEATURE_COSINE,
                "requires_semantic_compatibility": True,
                "max_normalized_center_distance": LARGE_MASK_MAX_NORMALIZED_CENTER_DISTANCE,
                "min_volume_ratio": LARGE_MASK_MIN_VOLUME_RATIO,
                "min_sorted_axis_ratio": LARGE_MASK_MIN_SORTED_AXIS_RATIO,
                "min_obb_diagonal_m": LARGE_MASK_MIN_OBB_DIAGONAL_M,
                "presentation_only": True,
            },
        },
    }
    return report, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--near-distance-m", type=float, default=0.20)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = torch.load(args.scene_state, map_location="cpu", weights_only=False)
    state = payload["state"] if isinstance(payload, dict) and isinstance(payload.get("state"), dict) else payload
    active_before = _numpy(state["active"], bool).copy()
    report, evidence = resolve_duplicates(
        state,
        near_distance_m=float(args.near_distance_m),
        mask_root=args.mask_root,
    )
    if not np.array_equal(active_before, _numpy(state["active"], bool)):
        raise AssertionError("Deduplication must not mutate FARM active state")
    report["source_scene_state"] = str(args.scene_state.resolve())
    report["output_scene_state"] = str(args.output_state.resolve())
    report["mask_root"] = str(args.mask_root.expanduser().resolve())
    report["near_distance_m"] = float(args.near_distance_m)
    report["pair_evidence"] = [row.__dict__ for row in evidence]
    report["duration_seconds"] = time.perf_counter() - started
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload.setdefault("meta", {})["track_dedup"] = {
            "schema": report["schema"],
            "source": report["source_scene_state"],
            "suppressed_objects": report["suppressed_objects"],
            "duplicate_groups": report["duplicate_groups"],
        }
    else:
        payload = state
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_state)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("active_metric_objects", "eligible_direct_objects", "suppressed_objects", "duplicate_groups", "duration_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
