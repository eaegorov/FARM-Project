"""Geometry-only helpers for scene-agnostic full-COLMAP view retrieval.

The planner intentionally starts from metric object support rather than from a
possibly stale object box.  Object support points are matched to nearby COLMAP
points, and the image tracks of those points provide a bounded camera shortlist.
No category name, detector box, or fitted OBB participates in this retrieval.

All helpers are deterministic and CPU-only so the selection policy can be unit
tested independently from pycolmap and the GPU refinement stages.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SparseTrackIndex:
    """Metric sparse points and the registered images observing each point."""

    point_ids: np.ndarray
    xyz_m: np.ndarray
    image_ids_by_point: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        point_ids = np.asarray(self.point_ids)
        xyz_m = np.asarray(self.xyz_m)
        if point_ids.ndim != 1:
            raise ValueError("point_ids must have shape [N]")
        if xyz_m.shape != (point_ids.size, 3):
            raise ValueError("xyz_m must have shape [N,3]")
        if len(self.image_ids_by_point) != point_ids.size:
            raise ValueError("image_ids_by_point must align with sparse points")


def bounded_support_points(
    points_m: object,
    *,
    trim_quantile: float,
    max_points: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return a finite, robust and deterministically bounded object seed cloud."""

    if not 0.5 <= float(trim_quantile) <= 1.0:
        raise ValueError("trim_quantile must be in [0.5, 1.0]")
    if int(max_points) < 1:
        raise ValueError("max_points must be positive")
    values = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    input_points = int(values.shape[0])
    values = values[np.isfinite(values).all(axis=1)]
    finite_points = int(values.shape[0])
    trimmed_points = finite_points
    if values.shape[0] >= 8 and float(trim_quantile) < 1.0:
        centre = np.median(values, axis=0)
        radial = np.linalg.norm(values - centre[None, :], axis=1)
        cutoff = float(np.quantile(radial, float(trim_quantile)))
        keep = radial <= cutoff + 1.0e-12
        # A malformed cloud must not collapse to a tiny seed merely because of
        # robust trimming.  Retrieval is safer when it fails closed downstream.
        if int(keep.sum()) >= max(4, int(math.ceil(0.5 * values.shape[0]))):
            values = values[keep]
        trimmed_points = int(values.shape[0])
    if values.shape[0] > int(max_points):
        indices = np.linspace(
            0, values.shape[0] - 1, int(max_points), dtype=np.int64
        )
        values = values[indices]
    result = np.ascontiguousarray(values, dtype=np.float64)
    return result, {
        "input_points": input_points,
        "finite_points": finite_points,
        "trimmed_points": trimmed_points,
        "bounded_points": int(result.shape[0]),
        "trim_quantile": float(trim_quantile),
    }


def build_sparse_track_index(
    reconstruction: object,
    *,
    meters_per_scene_unit: float,
) -> SparseTrackIndex:
    """Extract an immutable sparse-track index from a pycolmap reconstruction."""

    if not math.isfinite(float(meters_per_scene_unit)) or meters_per_scene_unit <= 0:
        raise ValueError("meters_per_scene_unit must be positive")
    points = getattr(reconstruction, "points3D", None)
    if points is None or not hasattr(points, "items"):
        raise TypeError("reconstruction.points3D must be an indexed mapping")
    point_ids: list[int] = []
    xyz_m: list[np.ndarray] = []
    image_ids_by_point: list[tuple[int, ...]] = []
    for point_id, point in sorted(points.items(), key=lambda item: int(item[0])):
        xyz = np.asarray(getattr(point, "xyz", []), dtype=np.float64).reshape(-1)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        track = getattr(point, "track", None)
        elements = getattr(track, "elements", ()) if track is not None else ()
        image_ids = sorted(
            {
                int(getattr(element, "image_id"))
                for element in elements
                if getattr(element, "image_id", None) is not None
            }
        )
        if not image_ids:
            continue
        point_ids.append(int(point_id))
        xyz_m.append(xyz * float(meters_per_scene_unit))
        image_ids_by_point.append(tuple(image_ids))
    xyz_array = (
        np.stack(xyz_m).astype(np.float64, copy=False)
        if xyz_m
        else np.zeros((0, 3), dtype=np.float64)
    )
    return SparseTrackIndex(
        point_ids=np.asarray(point_ids, dtype=np.int64),
        xyz_m=xyz_array,
        image_ids_by_point=tuple(image_ids_by_point),
    )


def match_support_to_sparse_tracks(
    support_points_m: object,
    sparse: SparseTrackIndex,
    *,
    maximum_distance_m: float,
) -> tuple[dict[int, int], dict[str, float | int | None]]:
    """Match object support to COLMAP points and count track evidence per image.

    Each support sample contributes at most one nearest sparse point.  Sparse
    points are then deduplicated, preventing a dense object voxel cloud from
    artificially inflating the evidence of a single COLMAP track.
    """

    if not math.isfinite(float(maximum_distance_m)) or maximum_distance_m <= 0:
        raise ValueError("maximum_distance_m must be positive")
    support = np.asarray(support_points_m, dtype=np.float64).reshape(-1, 3)
    support = support[np.isfinite(support).all(axis=1)]
    if support.size == 0 or sparse.xyz_m.size == 0:
        return {}, {
            "support_points": int(support.shape[0]),
            "matched_sparse_points": 0,
            "matched_support_fraction": 0.0,
            "median_match_distance_m": None,
            "maximum_match_distance_m": float(maximum_distance_m),
            "candidate_images": 0,
        }
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError("scipy is required for sparse-track retrieval") from exc
    tree = cKDTree(sparse.xyz_m)
    distances, indices = tree.query(
        support,
        k=1,
        distance_upper_bound=float(maximum_distance_m),
        workers=1,
    )
    valid = np.isfinite(distances) & (indices >= 0) & (indices < sparse.xyz_m.shape[0])
    matched_indices = np.unique(indices[valid].astype(np.int64, copy=False))
    image_counts: dict[int, int] = {}
    for sparse_index in matched_indices.tolist():
        for image_id in sparse.image_ids_by_point[int(sparse_index)]:
            image_counts[int(image_id)] = image_counts.get(int(image_id), 0) + 1
    valid_distances = distances[valid]
    return image_counts, {
        "support_points": int(support.shape[0]),
        "matched_sparse_points": int(matched_indices.size),
        "matched_support_fraction": float(valid.sum() / max(support.shape[0], 1)),
        "median_match_distance_m": (
            float(np.median(valid_distances)) if valid_distances.size else None
        ),
        "maximum_match_distance_m": float(maximum_distance_m),
        "candidate_images": len(image_counts),
    }


def project_metric_support(
    support_points_m: object,
    world_to_camera: object,
    intrinsics: object,
    *,
    width: int,
    height: int,
    meters_per_scene_unit: float,
    min_margin_ratio: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_in_frame_ratio: float,
    bbox_quantile: float,
) -> dict[str, object] | None:
    """Project object support without using a detector box or fitted OBB."""

    if min(int(width), int(height)) < 2:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= float(bbox_quantile) < 0.25:
        raise ValueError("bbox_quantile must be in [0, 0.25)")
    if not 0.0 <= float(min_in_frame_ratio) <= 1.0:
        raise ValueError("min_in_frame_ratio must be in [0, 1]")
    if not math.isfinite(float(meters_per_scene_unit)) or meters_per_scene_unit <= 0:
        raise ValueError("meters_per_scene_unit must be positive")
    points_m = np.asarray(support_points_m, dtype=np.float64).reshape(-1, 3)
    points_m = points_m[np.isfinite(points_m).all(axis=1)]
    pose = np.asarray(world_to_camera, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if points_m.shape[0] < 2 or pose.shape != (3, 4) or K.shape != (3, 3):
        return None
    points_scene = points_m / float(meters_per_scene_unit)
    camera_points = points_scene @ pose[:, :3].T + pose[:, 3]
    front = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1.0e-5)
    if int(front.sum()) < 2:
        return None
    camera_front = camera_points[front]
    pixels_h = camera_front @ K.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    inside = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height))
    )
    in_frame_ratio = float(inside.sum() / max(points_m.shape[0], 1))
    if in_frame_ratio < float(min_in_frame_ratio) or int(inside.sum()) < 2:
        return None
    visible_pixels = pixels[inside]
    q = float(bbox_quantile)
    lo = np.quantile(visible_pixels, q, axis=0) if q else visible_pixels.min(axis=0)
    hi = (
        np.quantile(visible_pixels, 1.0 - q, axis=0)
        if q
        else visible_pixels.max(axis=0)
    )
    x0, y0 = lo.tolist()
    x1, y1 = hi.tolist()
    bbox_width, bbox_height = float(x1 - x0), float(y1 - y0)
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
    homogeneous = np.vstack((pose, np.asarray([0.0, 0.0, 0.0, 1.0])))
    try:
        camera_to_world = np.linalg.inv(homogeneous)
    except np.linalg.LinAlgError:
        return None
    camera_center_m = camera_to_world[:3, 3] * float(meters_per_scene_unit)
    support_center_m = np.median(points_m, axis=0)
    view = support_center_m - camera_center_m
    distance_m = float(np.linalg.norm(view))
    if not math.isfinite(distance_m) or distance_m <= 1.0e-5:
        return None
    view_direction = view / distance_m
    center_offset = float(
        np.linalg.norm(
            np.asarray(
                [
                    (x0 + x1) * 0.5 / float(width) - 0.5,
                    (y0 + y1) * 0.5 / float(height) - 0.5,
                ]
            )
        )
    )
    geometric_score = (
        math.sqrt(area_ratio)
        * math.sqrt(max(in_frame_ratio, 0.0))
        * (1.0 + min(margin_ratio, 0.25))
        / (1.0 + center_offset)
    )
    return {
        "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "camera_center_m": camera_center_m.tolist(),
        "area_ratio": float(area_ratio),
        "margin_ratio": float(margin_ratio),
        "in_frame_support_ratio": in_frame_ratio,
        "projected_support_points": int(inside.sum()),
        "distance_m": distance_m,
        "view_direction": view_direction.tolist(),
        "geometric_score": float(geometric_score),
    }


def physical_timestamp_fold(
    timestamp: str,
    *,
    heldout_fraction: float,
    split_seed: str,
) -> str:
    """Assign a physical timestamp to a stable scene-global train/heldout fold."""

    if not timestamp:
        raise ValueError("physical timestamp must be non-empty")
    if not 0.0 < float(heldout_fraction) < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    digest = hashlib.sha256(
        f"farm-full-colmap-fold-v1\0{split_seed}\0{timestamp}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)
    return "heldout" if value < float(heldout_fraction) else "train"


def angular_separation_degrees(first: object, second: object) -> float:
    """Return a stable angular distance between two view directions."""

    a = np.asarray(first, dtype=np.float64).reshape(3)
    b = np.asarray(second, dtype=np.float64).reshape(3)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def camera_baseline_m(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    """Return a finite metric camera-centre baseline, or zero when unavailable."""

    try:
        left = np.asarray(first.get("camera_center_m"), dtype=np.float64).reshape(3)
        right = np.asarray(second.get("camera_center_m"), dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return 0.0
    baseline = float(np.linalg.norm(left - right))
    return baseline if math.isfinite(baseline) else 0.0


def candidate_reliability(row: Mapping[str, object]) -> float:
    """Bounded pre-render visibility/detail proxy; rendered depth stays a hard gate."""

    in_frame = np.clip(float(row.get("in_frame_support_ratio", 0.0)), 0.0, 1.0)
    margin = np.clip(float(row.get("margin_ratio", 0.0)) / 0.15, 0.0, 1.0)
    area = np.clip(
        math.sqrt(max(0.0, float(row.get("area_ratio", 0.0))) / 0.05),
        0.0,
        1.0,
    )
    track = np.clip(
        float(row.get("track_support_fraction", 0.0)) / 0.05, 0.0, 1.0
    )
    return float(0.35 * in_frame + 0.20 * margin + 0.25 * area + 0.20 * track)


def adaptive_view_limit(
    candidates: Sequence[Mapping[str, object]],
    *,
    minimum_views: int,
    maximum_views: int,
    priority: float = 0.0,
) -> tuple[int, dict[str, object]]:
    """Spend at most two redundancy views beyond the evidence minimum."""

    if int(minimum_views) < 1 or int(maximum_views) < int(minimum_views):
        raise ValueError("adaptive view limits are inconsistent")
    timestamps = {str(row.get("physical_timestamp") or "") for row in candidates}
    timestamps.discard("")
    available = len(timestamps)
    reliability = sorted(
        (candidate_reliability(row) for row in candidates), reverse=True
    )[: int(minimum_views)]
    mean_reliability = float(np.mean(reliability)) if reliability else 0.0
    extra = int(available > int(minimum_views))
    extra += int(
        available > int(minimum_views) + extra
        and (mean_reliability < 0.62 or float(priority) >= 2.0)
    )
    target = min(int(maximum_views), available, int(minimum_views) + extra)
    return target, {
        "minimum_views": int(minimum_views),
        "maximum_views": int(maximum_views),
        "available_physical_timestamps": available,
        "mean_top_minimum_reliability": mean_reliability,
        "priority": float(priority),
        "redundancy_views": max(0, target - int(minimum_views)),
        "target_views": target,
        "bounded_extra_views": 2,
    }


def choose_next_best_views(
    candidates: Sequence[Mapping[str, object]],
    *,
    minimum_views: int,
    maximum_views: int,
    min_view_angle_degrees: float,
    min_camera_baseline_m: float,
    min_baseline_distance_ratio: float,
    relaxed_view_angle_degrees: float,
    relaxed_camera_baseline_m: float,
    relaxed_baseline_distance_ratio: float,
    priority: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select timestamp-unique NBVs with bounded minimum-only relaxation."""

    if int(minimum_views) < 1 or int(maximum_views) < int(minimum_views):
        raise ValueError("next-best-view limits are inconsistent")
    if not 0 <= relaxed_view_angle_degrees <= min_view_angle_degrees:
        raise ValueError("relaxed angle must be in [0, strict angle]")
    if not 0 <= relaxed_camera_baseline_m <= min_camera_baseline_m:
        raise ValueError("relaxed baseline must be in [0, strict baseline]")
    if not 0 <= relaxed_baseline_distance_ratio <= min_baseline_distance_ratio:
        raise ValueError("relaxed baseline ratio must be in [0, strict ratio]")

    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (
            float(row.get("score", 0.0)),
            candidate_reliability(row),
            str(row.get("name", "")),
        ),
        reverse=True,
    )
    representatives: list[dict[str, object]] = []
    seen_timestamps: set[str] = set()
    for row in ordered:
        timestamp = str(row.get("physical_timestamp") or "")
        if not timestamp or timestamp in seen_timestamps:
            continue
        seen_timestamps.add(timestamp)
        row["view_reliability"] = candidate_reliability(row)
        representatives.append(row)

    target, budget = adaptive_view_limit(
        representatives,
        minimum_views=minimum_views,
        maximum_views=maximum_views,
        priority=priority,
    )
    selected: list[dict[str, object]] = []
    remaining = representatives.copy()
    maximum_score = max(
        (max(0.0, float(row.get("score", 0.0))) for row in remaining),
        default=1.0,
    ) or 1.0
    while remaining and len(selected) < target:
        ranked: list[tuple[tuple[float, str], dict[str, object]]] = []
        for source in remaining:
            row = dict(source)
            if not selected:
                angle, baseline, ratio, tier, admissible = 180.0, math.inf, math.inf, "seed", True
            else:
                angle = min(
                    angular_separation_degrees(row["view_direction"], prior["view_direction"])
                    for prior in selected
                )
                baseline = min(camera_baseline_m(row, prior) for prior in selected)
                distances = [
                    float(value)
                    for value in [
                        row.get("distance_m", 0.0),
                        *(prior.get("distance_m", 0.0) for prior in selected),
                    ]
                    if math.isfinite(float(value)) and float(value) > 0
                ]
                ratio = baseline / min(distances) if distances else 0.0
                strict = angle >= min_view_angle_degrees or (
                    baseline >= min_camera_baseline_m
                    and ratio >= min_baseline_distance_ratio
                )
                relaxed = angle >= relaxed_view_angle_degrees or (
                    baseline >= relaxed_camera_baseline_m
                    and ratio >= relaxed_baseline_distance_ratio
                )
                if strict:
                    tier, admissible = "strict", True
                elif len(selected) < minimum_views and relaxed:
                    tier, admissible = "minimum_only_relaxed", True
                else:
                    tier, admissible = "rejected_diversity", False
            if not admissible:
                continue
            quality = max(0.0, float(row.get("score", 0.0))) / maximum_score
            angle_gain = min(1.0, angle / max(min_view_angle_degrees, 1e-9))
            baseline_gain = min(
                1.0,
                max(
                    baseline / max(min_camera_baseline_m, 1e-9),
                    ratio / max(min_baseline_distance_ratio, 1e-9),
                ),
            )
            utility = float(
                0.45 * quality
                + 0.25 * row["view_reliability"]
                + 0.20 * angle_gain
                + 0.10 * baseline_gain
            )
            row.update(
                {
                    "selection_diversity_tier": tier,
                    "minimum_angle_to_selected_degrees": None if not selected else float(angle),
                    "minimum_camera_baseline_to_selected_m": None if not selected else float(baseline),
                    "minimum_baseline_distance_ratio": None if not selected else float(ratio),
                    "next_best_view_utility": utility,
                }
            )
            ranked.append(((utility, str(row.get("name", ""))), row))
        if not ranked:
            break
        chosen = max(ranked, key=lambda item: item[0])[1]
        selected.append(chosen)
        chosen_name = str(chosen.get("name") or "")
        remaining = [row for row in remaining if str(row.get("name") or "") != chosen_name]

    tiers = [str(row.get("selection_diversity_tier")) for row in selected]
    return selected, {
        "candidate_rows": len(candidates),
        "unique_physical_timestamps": len(representatives),
        "physical_timestamp_deduplicated_rows": len(candidates) - len(representatives),
        "selected_views": len(selected),
        "minimum_evidence_met": len(selected) >= minimum_views,
        "strict_selected_views": sum(tier in {"seed", "strict"} for tier in tiers),
        "relaxed_minimum_only_views": tiers.count("minimum_only_relaxed"),
        "diversity_relaxation_used": "minimum_only_relaxed" in tiers,
        "depth_support_status": "deferred_to_post_render_hard_gate",
        "occlusion_proxy": "matched_sparse_track_visibility",
        "thresholds": {
            "strict_angle_degrees": float(min_view_angle_degrees),
            "strict_camera_baseline_m": float(min_camera_baseline_m),
            "strict_baseline_distance_ratio": float(min_baseline_distance_ratio),
            "relaxed_angle_degrees": float(relaxed_view_angle_degrees),
            "relaxed_camera_baseline_m": float(relaxed_camera_baseline_m),
            "relaxed_baseline_distance_ratio": float(relaxed_baseline_distance_ratio),
        },
        "budget": budget,
    }


def plan_temporal_episode_topup(
    candidates: Sequence[Mapping[str, object]],
    next_best_views: Sequence[Mapping[str, object]],
    *,
    minimum_diverse_views: int,
    maximum_views: int,
    minimum_episode_frames: int,
    maximum_neighbor_frame_gap: int,
    maximum_translation_m: float,
    maximum_rotation_degrees: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Reserve a bounded same-stream episode around a strong NBV seed."""

    views = [dict(row) for row in next_best_views]
    if int(minimum_episode_frames) <= 1:
        return views[: int(maximum_views)], {
            "enabled": False,
            "reason": "minimum_episode_frames_le_one",
            "episode_frames": 0,
        }
    if min(
        int(minimum_diverse_views),
        int(maximum_views),
        int(maximum_neighbor_frame_gap),
    ) < 1:
        raise ValueError("temporal episode budgets must be positive")
    if min(float(maximum_translation_m), float(maximum_rotation_degrees)) <= 0:
        raise ValueError("temporal pose limits must be positive")
    required_budget = int(minimum_diverse_views) + int(minimum_episode_frames) - 1
    if int(maximum_views) < required_budget:
        return views[: int(maximum_views)], {
            "enabled": True,
            "planned": False,
            "reason": "insufficient_budget_for_diverse_plus_episode",
            "required_views": required_budget,
            "maximum_views": int(maximum_views),
            "episode_frames": 0,
        }

    representatives: dict[tuple[str, str], dict[str, object]] = {}
    for source in candidates:
        row = dict(source)
        timestamp = str(row.get("physical_timestamp") or "")
        stream = str(row.get("camera_stream_id") or "")
        try:
            row["_temporal_index"] = int(row["camera_stream_frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not timestamp or not stream:
            continue
        key = (stream, timestamp)
        incumbent = representatives.get(key)
        rank = (
            float(row.get("score", 0.0)),
            candidate_reliability(row),
            str(row.get("name", "")),
        )
        incumbent_rank = (
            (
                float(incumbent.get("score", 0.0)),
                candidate_reliability(incumbent),
                str(incumbent.get("name", "")),
            )
            if incumbent is not None
            else None
        )
        if incumbent is None or rank > incumbent_rank:
            representatives[key] = row

    by_stream: dict[str, list[dict[str, object]]] = {}
    for row in representatives.values():
        by_stream.setdefault(str(row["camera_stream_id"]), []).append(row)
    for rows in by_stream.values():
        rows.sort(
            key=lambda row: (
                int(row["_temporal_index"]),
                str(row.get("name") or ""),
            )
        )

    def transition_allowed(
        left: Mapping[str, object], right: Mapping[str, object]
    ) -> bool:
        frame_gap = abs(int(left["_temporal_index"]) - int(right["_temporal_index"]))
        return (
            1 <= frame_gap <= int(maximum_neighbor_frame_gap)
            and camera_baseline_m(left, right) <= float(maximum_translation_m)
            and angular_separation_degrees(
                left["view_direction"], right["view_direction"]
            )
            <= float(maximum_rotation_degrees)
        )

    options: list[tuple[tuple[float, str], list[dict[str, object]]]] = []
    for seed_source in views[: int(minimum_diverse_views)]:
        seed_name = str(seed_source.get("name") or "")
        rows = by_stream.get(str(seed_source.get("camera_stream_id") or ""), [])
        positions = [
            index
            for index, row in enumerate(rows)
            if str(row.get("name") or "") == seed_name
        ]
        if not positions:
            continue
        seed_position = positions[0]
        episode = [dict(rows[seed_position])]
        left_position = right_position = seed_position
        while len(episode) < int(minimum_episode_frames):
            neighbours: list[
                tuple[tuple[float, int, str], str, dict[str, object]]
            ] = []
            left_endpoint = rows[left_position]
            for index in range(left_position - 1, -1, -1):
                row = rows[index]
                gap = int(left_endpoint["_temporal_index"]) - int(
                    row["_temporal_index"]
                )
                if gap > int(maximum_neighbor_frame_gap):
                    break
                if transition_allowed(row, left_endpoint):
                    neighbours.append(
                        (
                            (
                                candidate_reliability(row),
                                -gap,
                                str(row.get("name") or ""),
                            ),
                            "left",
                            row,
                        )
                    )
            right_endpoint = rows[right_position]
            for index in range(right_position + 1, len(rows)):
                row = rows[index]
                gap = int(row["_temporal_index"]) - int(
                    right_endpoint["_temporal_index"]
                )
                if gap > int(maximum_neighbor_frame_gap):
                    break
                if transition_allowed(right_endpoint, row):
                    neighbours.append(
                        (
                            (
                                candidate_reliability(row),
                                -gap,
                                str(row.get("name") or ""),
                            ),
                            "right",
                            row,
                        )
                    )
            if not neighbours:
                break
            _, side, chosen = max(neighbours, key=lambda item: item[0])
            episode.append(dict(chosen))
            if side == "left":
                left_position = rows.index(chosen)
            else:
                right_position = rows.index(chosen)
        if len(episode) >= int(minimum_episode_frames):
            episode.sort(key=lambda row: int(row["_temporal_index"]))
            quality = float(
                np.mean([candidate_reliability(row) for row in episode])
            )
            options.append(((quality, seed_name), episode))

    if not options:
        return views[: int(maximum_views)], {
            "enabled": True,
            "planned": False,
            "reason": "no_pose_continuous_same_stream_window",
            "required_episode_frames": int(minimum_episode_frames),
            "episode_frames": 0,
        }

    episode = max(options, key=lambda item: item[0])[1]
    core_names = {
        str(row.get("name") or "")
        for row in views[: int(minimum_diverse_views)]
    }
    episode_seed = next(
        row for row in episode if str(row.get("name") or "") in core_names
    )
    episode_id = (
        f"{episode_seed['camera_stream_id']}@"
        f"{episode_seed['physical_timestamp']}"
    )
    episode_names = {str(row.get("name") or "") for row in episode}
    output: list[dict[str, object]] = []
    for row in episode:
        clean = {key: value for key, value in row.items() if key != "_temporal_index"}
        clean["temporal_episode_id"] = episode_id
        clean["temporal_episode_role"] = (
            "seed"
            if str(clean.get("name") or "")
            == str(episode_seed.get("name") or "")
            else "context"
        )
        output.append(clean)
    for source in views:
        if len(output) >= int(maximum_views):
            break
        if str(source.get("name") or "") not in episode_names:
            output.append(dict(source))

    return output, {
        "enabled": True,
        "planned": True,
        "reason": "same_stream_pose_continuous_window_found",
        "episode_id": episode_id,
        "camera_stream_id": str(episode_seed["camera_stream_id"]),
        "seed_image": str(episode_seed["name"]),
        "episode_frames": len(episode),
        "physical_timestamps": [
            str(row.get("physical_timestamp") or "") for row in episode
        ],
        "stream_frame_indices": [int(row["_temporal_index"]) for row in episode],
        "minimum_diverse_views": int(minimum_diverse_views),
        "retained_diverse_views": sum(
            str(row.get("name") or "") in core_names for row in output
        ),
        "maximum_neighbor_frame_gap": int(maximum_neighbor_frame_gap),
        "maximum_translation_m": float(maximum_translation_m),
        "maximum_rotation_degrees": float(maximum_rotation_degrees),
        "maximum_views": int(maximum_views),
    }


def temporal_episode_frame_count(views: Sequence[Mapping[str, object]]) -> int:
    """Return the largest episode group surviving global allocation."""

    counts: dict[str, int] = {}
    for row in views:
        episode_id = str(row.get("temporal_episode_id") or "")
        if episode_id:
            counts[episode_id] = counts.get(episode_id, 0) + 1
    return max(counts.values(), default=0)


def choose_diverse_views(
    candidates: Sequence[Mapping[str, object]],
    *,
    limit: int,
    min_view_angle_degrees: float,
) -> list[dict[str, object]]:
    """Select score-ranked, timestamp-unique and direction-diverse cameras."""

    if int(limit) < 1:
        raise ValueError("limit must be positive")
    ordered = sorted(
        (dict(row) for row in candidates),
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
        if not timestamp or timestamp in timestamps:
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
    # Diversity is a preference, while physical timestamp independence is a
    # hard requirement. Fill a small directional shortfall deterministically.
    for row in ordered:
        timestamp = str(row.get("physical_timestamp") or "")
        if not timestamp or timestamp in timestamps:
            continue
        selected.append(row)
        timestamps.add(timestamp)
        if len(selected) >= int(limit):
            break
    return selected


def choose_train_heldout_views(
    candidates: Sequence[Mapping[str, object]],
    *,
    train_limit: int,
    heldout_limit: int,
    min_view_angle_degrees: float,
    heldout_fraction: float,
    split_seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Assign timestamp folds before independently choosing diverse views."""

    train_pool: list[dict[str, object]] = []
    heldout_pool: list[dict[str, object]] = []
    for source in candidates:
        row = dict(source)
        timestamp = str(row.get("physical_timestamp") or "")
        fold = physical_timestamp_fold(
            timestamp,
            heldout_fraction=float(heldout_fraction),
            split_seed=str(split_seed),
        )
        row["split"] = fold
        (heldout_pool if fold == "heldout" else train_pool).append(row)
    train = choose_diverse_views(
        train_pool,
        limit=int(train_limit),
        min_view_angle_degrees=float(min_view_angle_degrees),
    )
    heldout = choose_diverse_views(
        heldout_pool,
        limit=int(heldout_limit),
        min_view_angle_degrees=float(min_view_angle_degrees),
    )
    validate_disjoint_timestamp_folds(train, heldout)
    return train, heldout


def allocate_global_view_budget(
    object_candidates: Sequence[Sequence[Mapping[str, object]]],
    *,
    max_total_views: int,
) -> tuple[dict[str, dict[str, object]], list[list[dict[str, object]]]]:
    """Round-robin unique image allocation, avoiding object-order starvation."""

    if int(max_total_views) < 1:
        raise ValueError("max_total_views must be positive")
    candidates = [[dict(row) for row in rows] for rows in object_candidates]
    selected_global: dict[str, dict[str, object]] = {}
    admitted: list[list[dict[str, object]]] = [[] for _ in candidates]
    maximum_rank = max((len(rows) for rows in candidates), default=0)
    for rank in range(maximum_rank):
        for object_index, rows in enumerate(candidates):
            if rank >= len(rows):
                continue
            row = rows[rank]
            name = str(row.get("name") or "")
            if not name:
                continue
            if name not in selected_global:
                if len(selected_global) >= int(max_total_views):
                    continue
                selected_global[name] = row
            admitted[object_index].append(row)
    return selected_global, admitted


def allocate_split_global_view_budget(
    train_candidates: Sequence[Sequence[Mapping[str, object]]],
    heldout_candidates: Sequence[Sequence[Mapping[str, object]]],
    *,
    max_total_views: int,
    max_heldout_views: int,
) -> tuple[
    dict[str, dict[str, object]],
    list[list[dict[str, object]]],
    list[list[dict[str, object]]],
]:
    """Reserve heldout capacity, then allocate train from the remaining budget."""

    if len(train_candidates) != len(heldout_candidates):
        raise ValueError("train/heldout object rows must align")
    if int(max_total_views) < 2:
        raise ValueError("max_total_views must be at least two")
    if not 1 <= int(max_heldout_views) < int(max_total_views):
        raise ValueError("max_heldout_views must be in [1, max_total_views)")
    heldout_global, heldout = allocate_global_view_budget(
        heldout_candidates, max_total_views=int(max_heldout_views)
    )
    train_global, train = allocate_global_view_budget(
        train_candidates,
        max_total_views=max(1, int(max_total_views) - len(heldout_global)),
    )
    overlap = set(train_global) & set(heldout_global)
    if overlap:
        raise ValueError(f"image leakage between folds: {sorted(overlap)[:5]}")
    return {**train_global, **heldout_global}, train, heldout


def model_file_manifest(model_root: object) -> list[dict[str, object]]:
    """List exact COLMAP model inputs for content-hash provenance."""

    from pathlib import Path

    root = Path(model_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"COLMAP model is not a directory: {root}")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name
        in {
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "cameras.txt",
            "images.txt",
            "points3D.txt",
        }
    ]
    result: list[dict[str, object]] = []
    for path in sorted(candidates, key=lambda item: item.name):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        result.append(
            {
                "name": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    if not result:
        raise ValueError(f"no COLMAP model files found in {root}")
    return result


def summarize_track_counts(
    image_counts: Mapping[int, int], *, matched_sparse_points: int
) -> dict[int, dict[str, float | int]]:
    """Normalize image evidence while retaining auditable integer counts."""

    denominator = max(1, int(matched_sparse_points))
    return {
        int(image_id): {
            "track_support_count": int(count),
            "track_support_fraction": float(int(count) / denominator),
        }
        for image_id, count in image_counts.items()
    }


def track_evidence_tier(
    *,
    track_count: int,
    track_fraction: float,
    minimum_track_support_points: int,
    minimum_track_support_fraction: float,
    minimum_rescue_track_support_points: int,
) -> str | None:
    """Classify sparse retrieval evidence without confusing it with depth."""

    if (
        int(track_count) >= int(minimum_track_support_points)
        and float(track_fraction) >= float(minimum_track_support_fraction)
    ):
        return "strict"
    if int(track_count) >= int(minimum_rescue_track_support_points):
        return "sparse_track_rescue"
    return None


def validate_disjoint_timestamp_folds(
    train_views: Sequence[Mapping[str, object]],
    heldout_views: Sequence[Mapping[str, object]],
) -> None:
    """Fail if a physical timestamp leaked between fitting and heldout views."""

    train = {str(row.get("physical_timestamp") or "") for row in train_views}
    heldout = {str(row.get("physical_timestamp") or "") for row in heldout_views}
    train.discard("")
    heldout.discard("")
    overlap = sorted(train & heldout)
    if overlap:
        raise ValueError(f"physical timestamp leakage between folds: {overlap[:5]}")
