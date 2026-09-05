"""Object-visible paired tracker episode planning primitives.

The v1 tracker subset balanced camera streams, but did not guarantee that any
one physical object was visible in both its train and heldout halves.  This
module plans *pairs* of short episodes around one canonical object.  A frame is
eligible only when metric object support is linked to COLMAP tracks and that
support projects successfully into the frame; temporal neighbours are never
silently admitted.

Selection is deterministic and CPU-only.  It does not inspect an OBB, decode an
RGB image, or assign semantic labels.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

import numpy as np

from .tracker_subset_planner import TrackerFrame, camera_motion

SCHEMA = "farm.tracker-subset-plan.v2"
PLANNER_REVISION = "object-anchor-paired-support-tracks.v2"


@dataclass(frozen=True)
class SupportEvidence:
    """Per-frame geometric evidence for one canonical object."""

    object_id: int
    frame: TrackerFrame
    track_support_count: int
    track_support_fraction: float
    projected_support_points: int
    in_frame_support_ratio: float
    area_ratio: float
    margin_ratio: float
    geometric_score: float
    view_direction: tuple[float, float, float]
    rescue_anchor: bool = False

    def __post_init__(self) -> None:
        if int(self.object_id) < 0:
            raise ValueError("object_id must be non-negative")
        if int(self.track_support_count) < 1:
            raise ValueError("track_support_count must be positive")
        if int(self.projected_support_points) < 2:
            raise ValueError("projected_support_points must be at least two")
        bounded = {
            "track_support_fraction": self.track_support_fraction,
            "in_frame_support_ratio": self.in_frame_support_ratio,
            "area_ratio": self.area_ratio,
            "margin_ratio": self.margin_ratio,
            "geometric_score": self.geometric_score,
        }
        for name, value in bounded.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(self.track_support_fraction) <= 1.0:
            raise ValueError("track_support_fraction must be in (0, 1]")
        if not 0.0 < float(self.in_frame_support_ratio) <= 1.0:
            raise ValueError("in_frame_support_ratio must be in (0, 1]")
        if not 0.0 < float(self.area_ratio) <= 1.0:
            raise ValueError("area_ratio must be in (0, 1]")
        if float(self.margin_ratio) < 0.0 or float(self.geometric_score) <= 0.0:
            raise ValueError("margin_ratio/geometric_score are invalid")
        direction = np.asarray(self.view_direction, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("view_direction must be a finite 3-vector")
        if float(np.linalg.norm(direction)) <= 1.0e-8:
            raise ValueError("view_direction must be non-zero")


@dataclass(frozen=True)
class ObjectVisibleWindow:
    """A contiguous exact-stream window whose every frame sees one object."""

    object_id: int
    camera: str
    view_family: str
    frames: tuple[SupportEvidence, ...]
    anchor_index: int
    registered_stream_start: int
    registered_stream_end: int
    coverage_score: float
    diversity_score: float
    total_score: float
    metrics: Mapping[str, object]

    @property
    def names(self) -> frozenset[str]:
        return frozenset(row.frame.name for row in self.frames)

    @property
    def timestamps(self) -> frozenset[Decimal]:
        return frozenset(row.frame.identity.timestamp_value for row in self.frames)

    @property
    def stream_key(self) -> tuple[str, str]:
        return self.camera, self.view_family


@dataclass(frozen=True)
class PairedObjectWindows:
    object_id: int
    first: ObjectVisibleWindow
    second: ObjectVisibleWindow
    cross_view_angle_degrees: float
    stream_diversity: float
    pair_score: float


@dataclass(frozen=True)
class SelectedPair:
    object_id: int
    train: ObjectVisibleWindow
    heldout: ObjectVisibleWindow
    pair: PairedObjectWindows


def _direction_angle_degrees(first: object, second: object) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(3)
    b = np.asarray(second, dtype=np.float64).reshape(3)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _mean_direction(rows: Sequence[SupportEvidence]) -> np.ndarray:
    values = np.asarray([row.view_direction for row in rows], dtype=np.float64)
    direction = values.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        return values[len(values) // 2]
    return direction / norm


def _window_scores(
    rows: Sequence[SupportEvidence],
    *,
    anchor_index: int,
    preferred_frames: int,
    maximum_translation_m: float,
    maximum_rotation_degrees: float,
) -> tuple[float, float, float, dict[str, object]]:
    track_fraction = np.asarray(
        [row.track_support_fraction for row in rows], dtype=np.float64
    )
    track_count = np.asarray(
        [row.track_support_count for row in rows], dtype=np.float64
    )
    projected = np.asarray(
        [row.projected_support_points for row in rows], dtype=np.float64
    )
    in_frame = np.asarray(
        [row.in_frame_support_ratio for row in rows], dtype=np.float64
    )
    area = np.asarray([row.area_ratio for row in rows], dtype=np.float64)
    geometric = np.asarray([row.geometric_score for row in rows], dtype=np.float64)
    # Each component is deliberately bounded: no one unusually dense sparse
    # track can overpower the requirement that *all* frames retain support.
    track_component = 0.5 * min(1.0, float(np.median(track_count)) / 10.0) + 0.5 * min(
        1.0, float(np.median(track_fraction)) / 0.05
    )
    projection_component = 0.5 * float(np.min(in_frame)) + 0.5 * min(
        1.0, math.sqrt(max(float(np.median(area)), 0.0) / 0.10)
    )
    support_component = min(1.0, float(np.median(projected)) / 64.0)
    geometric_component = min(1.0, float(np.median(geometric)) / 0.35)
    coverage = float(
        0.30 * track_component
        + 0.35 * projection_component
        + 0.20 * support_component
        + 0.15 * geometric_component
    )
    first_pose = rows[0].frame.world_from_camera_m
    last_pose = rows[-1].frame.world_from_camera_m
    translation_m, rotation_degrees = camera_motion(first_pose, last_pose)
    view_angle = _direction_angle_degrees(
        rows[0].view_direction, rows[-1].view_direction
    )
    diversity = float(
        0.40 * min(1.0, translation_m / max(float(maximum_translation_m), 1.0e-6))
        + 0.35
        * min(1.0, rotation_degrees / max(float(maximum_rotation_degrees), 1.0e-6))
        + 0.25 * min(1.0, view_angle / 30.0)
    )
    length_component = max(
        0.0, 1.0 - abs(len(rows) - int(preferred_frames)) / max(preferred_frames, 1)
    )
    centre = 0.5 * (len(rows) - 1)
    anchor_centrality = max(0.0, 1.0 - abs(anchor_index - centre) / max(centre, 1.0))
    rescue_bonus = 1.0 if rows[anchor_index].rescue_anchor else 0.0
    total = float(
        0.67 * coverage
        + 0.22 * diversity
        + 0.06 * length_component
        + 0.03 * anchor_centrality
        + 0.02 * rescue_bonus
    )
    return coverage, diversity, total, {
        "minimum_track_support_count": int(np.min(track_count)),
        "median_track_support_count": float(np.median(track_count)),
        "minimum_track_support_fraction": float(np.min(track_fraction)),
        "median_track_support_fraction": float(np.median(track_fraction)),
        "minimum_projected_support_points": int(np.min(projected)),
        "median_projected_support_points": float(np.median(projected)),
        "minimum_in_frame_support_ratio": float(np.min(in_frame)),
        "median_area_ratio": float(np.median(area)),
        "median_geometric_score": float(np.median(geometric)),
        "endpoint_translation_m": float(translation_m),
        "endpoint_rotation_degrees": float(rotation_degrees),
        "endpoint_view_angle_degrees": float(view_angle),
        "anchor_centrality": float(anchor_centrality),
        "rescue_anchor": bool(rescue_bonus),
    }


def enumerate_object_visible_windows(
    registered_frames: Sequence[TrackerFrame],
    evidence: Sequence[SupportEvidence],
    *,
    minimum_frames: int = 8,
    maximum_frames: int = 12,
    preferred_frames: int = 10,
    gap_multiplier: float = 2.5,
    maximum_translation_m: float = 0.50,
    maximum_rotation_degrees: float = 21.0,
    maximum_windows: int = 96,
) -> tuple[list[ObjectVisibleWindow], dict[str, object]]:
    """Enumerate evidence-complete windows without bridging an ineligible frame."""

    minimum = int(minimum_frames)
    maximum = int(maximum_frames)
    preferred = int(preferred_frames)
    if not 2 <= minimum <= preferred <= maximum:
        raise ValueError("episode bounds must satisfy 2 <= minimum <= preferred <= maximum")
    if int(maximum_windows) < 1:
        raise ValueError("maximum_windows must be positive")
    if not math.isfinite(float(gap_multiplier)) or float(gap_multiplier) < 1.0:
        raise ValueError("gap_multiplier must be finite and at least one")
    if float(maximum_translation_m) <= 0.0:
        raise ValueError("maximum_translation_m must be positive")
    if not 0.0 < float(maximum_rotation_degrees) <= 180.0:
        raise ValueError("maximum_rotation_degrees must be in (0, 180]")
    object_ids = {int(row.object_id) for row in evidence}
    if len(object_ids) != 1:
        raise ValueError("evidence must describe exactly one canonical object")
    object_id = next(iter(object_ids))
    evidence_by_name: dict[str, SupportEvidence] = {}
    for row in evidence:
        if row.frame.name in evidence_by_name:
            raise ValueError(f"duplicate object evidence frame: {row.frame.name}")
        evidence_by_name[row.frame.name] = row
    registered_by_name = {row.name: row for row in registered_frames}
    missing = sorted(set(evidence_by_name).difference(registered_by_name))
    if missing:
        raise ValueError(f"evidence frames are absent from registered COLMAP: {missing[:5]}")
    grouped: dict[tuple[str, str], list[TrackerFrame]] = defaultdict(list)
    for frame in registered_frames:
        grouped[frame.stream_key].append(frame)
    windows: list[ObjectVisibleWindow] = []
    diagnostics: Counter[str] = Counter()
    eligible_run_lengths: list[int] = []
    for stream_key in sorted(grouped):
        ordered = sorted(
            grouped[stream_key],
            key=lambda row: (row.identity.timestamp_value, row.name),
        )
        timestamps = [row.identity.timestamp_value for row in ordered]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(f"duplicate physical timestamp in exact stream {stream_key}")
        positive_gaps = [
            float(right - left)
            for left, right in zip(timestamps, timestamps[1:])
            if right > left
        ]
        lower_quartile = (
            float(np.quantile(np.asarray(positive_gaps, dtype=np.float64), 0.25))
            if positive_gaps
            else None
        )
        maximum_gap = (
            float(lower_quartile) * float(gap_multiplier)
            if lower_quartile is not None
            else None
        )
        runs: list[tuple[int, list[SupportEvidence]]] = []
        run_start = 0
        current: list[SupportEvidence] = []
        previous_frame: TrackerFrame | None = None
        for stream_index, frame in enumerate(ordered):
            row = evidence_by_name.get(frame.name)
            reasons: list[str] = []
            if row is None:
                reasons.append("frame_lacks_object_support_track_projection")
            if current and previous_frame is not None and row is not None:
                gap = float(
                    frame.identity.timestamp_value
                    - previous_frame.identity.timestamp_value
                )
                translation_m, rotation_degrees = camera_motion(
                    previous_frame.world_from_camera_m, frame.world_from_camera_m
                )
                if maximum_gap is not None and gap > maximum_gap:
                    reasons.append("timestamp_gap")
                if translation_m > float(maximum_translation_m):
                    reasons.append("camera_translation")
                if rotation_degrees > float(maximum_rotation_degrees):
                    reasons.append("camera_rotation")
            if reasons:
                if current:
                    runs.append((run_start, current))
                    eligible_run_lengths.append(len(current))
                current = []
                for reason in reasons:
                    diagnostics[reason] += 1
            if row is not None:
                if not current:
                    run_start = stream_index
                current.append(row)
                previous_frame = frame
            else:
                previous_frame = None
        if current:
            runs.append((run_start, current))
            eligible_run_lengths.append(len(current))
        for registered_start, run in runs:
            if len(run) < minimum:
                diagnostics["short_eligible_run"] += 1
                continue
            for length in range(minimum, min(maximum, len(run)) + 1):
                for offset in range(0, len(run) - length + 1):
                    rows = tuple(run[offset : offset + length])
                    interior = range(1, length - 1)
                    anchor_index = max(
                        interior,
                        key=lambda index: (
                            bool(rows[index].rescue_anchor),
                            float(rows[index].geometric_score),
                            -abs(index - 0.5 * (length - 1)),
                            rows[index].frame.name,
                        ),
                    )
                    coverage, diversity, total, metrics = _window_scores(
                        rows,
                        anchor_index=anchor_index,
                        preferred_frames=preferred,
                        maximum_translation_m=maximum_translation_m,
                        maximum_rotation_degrees=maximum_rotation_degrees,
                    )
                    windows.append(
                        ObjectVisibleWindow(
                            object_id=object_id,
                            camera=stream_key[0],
                            view_family=stream_key[1],
                            frames=rows,
                            anchor_index=anchor_index,
                            registered_stream_start=registered_start + offset,
                            registered_stream_end=registered_start + offset + length - 1,
                            coverage_score=coverage,
                            diversity_score=diversity,
                            total_score=total,
                            metrics=metrics,
                        )
                    )
    windows.sort(
        key=lambda row: (
            row.total_score,
            row.coverage_score,
            -abs(len(row.frames) - preferred),
            row.camera,
            row.view_family,
            row.frames[0].frame.name,
        ),
        reverse=True,
    )
    deduplicated: list[ObjectVisibleWindow] = []
    seen: set[frozenset[str]] = set()
    for window in windows:
        if window.names in seen:
            continue
        seen.add(window.names)
        deduplicated.append(window)
        if len(deduplicated) >= int(maximum_windows):
            break
    return deduplicated, {
        "object_id": object_id,
        "registered_frames": len(registered_frames),
        "evidence_frames": len(evidence),
        "eligible_runs": len(eligible_run_lengths),
        "eligible_run_length_max": max(eligible_run_lengths, default=0),
        "candidate_windows_before_limit": len(windows),
        "candidate_windows": len(deduplicated),
        "rejection_break_counts": dict(sorted(diagnostics.items())),
    }


def pair_object_windows(
    windows: Sequence[ObjectVisibleWindow],
    *,
    maximum_pairs: int = 256,
) -> list[PairedObjectWindows]:
    """Form object-consistent, RGB-disjoint and timestamp-disjoint pairs."""

    if int(maximum_pairs) < 1:
        raise ValueError("maximum_pairs must be positive")
    object_ids = {row.object_id for row in windows}
    if len(object_ids) > 1:
        raise ValueError("windows from different canonical objects cannot be paired")
    result: list[PairedObjectWindows] = []
    for left_index, left in enumerate(windows):
        for right in windows[left_index + 1 :]:
            if left.names & right.names or left.timestamps & right.timestamps:
                continue
            cross_angle = _direction_angle_degrees(
                _mean_direction(left.frames), _mean_direction(right.frames)
            )
            stream_diversity = 1.0 if left.stream_key != right.stream_key else 0.0
            pair_score = float(
                0.76 * 0.5 * (left.total_score + right.total_score)
                + 0.17 * min(1.0, cross_angle / 45.0)
                + 0.07 * stream_diversity
            )
            result.append(
                PairedObjectWindows(
                    object_id=left.object_id,
                    first=left,
                    second=right,
                    cross_view_angle_degrees=cross_angle,
                    stream_diversity=stream_diversity,
                    pair_score=pair_score,
                )
            )
    result.sort(
        key=lambda row: (
            row.pair_score,
            row.cross_view_angle_degrees,
            row.first.frames[0].frame.name,
            row.second.frames[0].frame.name,
        ),
        reverse=True,
    )
    return result[: int(maximum_pairs)]


def _orientation_key(
    pair: PairedObjectWindows, train: ObjectVisibleWindow, heldout: ObjectVisibleWindow
) -> tuple[float, int]:
    digest = hashlib.sha256(
        (
            f"{pair.object_id}\0{train.frames[0].frame.name}\0"
            f"{heldout.frames[0].frame.name}"
        ).encode("utf-8")
    ).digest()
    return train.coverage_score - heldout.coverage_score, int.from_bytes(
        digest[:8], "big"
    )


def select_paired_object_episodes(
    pairs_by_object: Mapping[int, Sequence[PairedObjectWindows]],
    *,
    target_frames: int = 160,
    minimum_total_frames: int = 120,
    maximum_total_frames: int = 200,
) -> tuple[list[SelectedPair], dict[str, object]]:
    """Greedily choose whole pairs under global RGB/timestamp isolation gates."""

    target = int(target_frames)
    minimum = int(minimum_total_frames)
    maximum = int(maximum_total_frames)
    if not 1 <= minimum <= target <= maximum:
        raise ValueError("total frame bounds must satisfy 1 <= minimum <= target <= maximum")
    remaining = {
        int(object_id): list(rows)
        for object_id, rows in pairs_by_object.items()
        if rows
    }
    selected: list[SelectedPair] = []
    used_names: set[str] = set()
    train_timestamps: set[Decimal] = set()
    heldout_timestamps: set[Decimal] = set()
    used_streams: Counter[tuple[str, str]] = Counter()
    total = 0
    rejection_counts: Counter[str] = Counter()
    while remaining and total < target:
        candidates: list[tuple[float, int, SelectedPair]] = []
        for object_id, object_pairs in sorted(remaining.items()):
            for pair in object_pairs:
                orientations = [
                    (pair.first, pair.second),
                    (pair.second, pair.first),
                ]
                orientations.sort(
                    key=lambda row: _orientation_key(pair, row[0], row[1]),
                    reverse=True,
                )
                admitted: SelectedPair | None = None
                for train, heldout in orientations:
                    pair_names = train.names | heldout.names
                    if pair_names & used_names:
                        rejection_counts["duplicate_rgb"] += 1
                        continue
                    if train.timestamps & heldout_timestamps:
                        rejection_counts["train_timestamp_already_heldout"] += 1
                        continue
                    if heldout.timestamps & train_timestamps:
                        rejection_counts["heldout_timestamp_already_train"] += 1
                        continue
                    if total + len(train.frames) + len(heldout.frames) > maximum:
                        rejection_counts["maximum_total_frames"] += 1
                        continue
                    admitted = SelectedPair(
                        object_id=object_id,
                        train=train,
                        heldout=heldout,
                        pair=pair,
                    )
                    break
                if admitted is None:
                    continue
                new_streams = sum(
                    used_streams[row.stream_key] == 0
                    for row in (admitted.train, admitted.heldout)
                )
                marginal = float(pair.pair_score + 0.025 * new_streams)
                candidates.append((marginal, -object_id, admitted))
                # The list is score-ranked, so a single feasible proposal per
                # object is sufficient for this greedy round.
                break
        if not candidates:
            break
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        chosen = candidates[0][2]
        selected.append(chosen)
        used_names.update(chosen.train.names | chosen.heldout.names)
        train_timestamps.update(chosen.train.timestamps)
        heldout_timestamps.update(chosen.heldout.timestamps)
        used_streams[chosen.train.stream_key] += len(chosen.train.frames)
        used_streams[chosen.heldout.stream_key] += len(chosen.heldout.frames)
        total += len(chosen.train.frames) + len(chosen.heldout.frames)
        remaining.pop(chosen.object_id)
    if total < minimum:
        raise ValueError(
            "object-paired tracker coverage is insufficient: "
            f"selected {total}, required {minimum}, objects_with_pairs "
            f"{sum(bool(rows) for rows in pairs_by_object.values())}"
        )
    if train_timestamps & heldout_timestamps:
        raise AssertionError("global physical timestamp leakage after selection")
    if len(used_names) != total:
        raise AssertionError("duplicate RGB after selection")
    selected.sort(key=lambda row: row.object_id)
    return selected, {
        "target_frames": target,
        "minimum_total_frames": minimum,
        "maximum_total_frames": maximum,
        "selected_frames": total,
        "selected_pairs": len(selected),
        "selected_objects": [row.object_id for row in selected],
        "train_frames": sum(len(row.train.frames) for row in selected),
        "heldout_frames": sum(len(row.heldout.frames) for row in selected),
        "train_physical_timestamps": len(train_timestamps),
        "heldout_physical_timestamps": len(heldout_timestamps),
        "unique_rgb": len(used_names),
        "selected_exact_streams": len(used_streams),
        "frames_by_exact_stream": {
            f"{camera}::{family}": count
            for (camera, family), count in sorted(used_streams.items())
        },
        "selection_rejection_counts": dict(sorted(rejection_counts.items())),
        "selection_policy": (
            "whole object pair, best feasible quality plus new-stream diversity; "
            "QA object IDs never affect eligibility or ranking"
        ),
    }


def selected_pairs_to_episode_rows(
    selected: Sequence[SelectedPair],
) -> list[dict[str, object]]:
    """Serialize paired windows using the v1 frame layout plus v2 evidence."""

    rows: list[dict[str, object]] = []
    for pair in selected:
        pair_id = f"object-{pair.object_id:06d}"
        for split, window in (("train", pair.train), ("heldout", pair.heldout)):
            episode_id = (
                f"{pair_id}__{window.camera}__{window.view_family}__{split}"
            )
            edge_translations: list[float] = []
            edge_rotations: list[float] = []
            gaps: list[float] = []
            for left, right in zip(window.frames, window.frames[1:]):
                translation, rotation = camera_motion(
                    left.frame.world_from_camera_m,
                    right.frame.world_from_camera_m,
                )
                edge_translations.append(translation)
                edge_rotations.append(rotation)
                gaps.append(
                    float(
                        right.frame.identity.timestamp_value
                        - left.frame.identity.timestamp_value
                    )
                )
            rows.append(
                {
                    "episode_id": episode_id,
                    "pair_id": pair_id,
                    "anchor_object_id": pair.object_id,
                    "camera": window.camera,
                    "view_family": window.view_family,
                    "split": split,
                    "frame_count": len(window.frames),
                    "anchor_frame_count": 1,
                    "object_evidence_frame_count": len(window.frames),
                    "all_frames_object_visible": True,
                    "first_physical_timestamp": window.frames[
                        0
                    ].frame.identity.physical_timestamp,
                    "last_physical_timestamp": window.frames[
                        -1
                    ].frame.identity.physical_timestamp,
                    "registered_stream_span": {
                        "start_index": window.registered_stream_start,
                        "end_index": window.registered_stream_end,
                        "contiguous": True,
                    },
                    "scores": {
                        "coverage": window.coverage_score,
                        "within_window_diversity": window.diversity_score,
                        "window_total": window.total_score,
                        "pair_total": pair.pair.pair_score,
                        "cross_window_view_angle_degrees": (
                            pair.pair.cross_view_angle_degrees
                        ),
                        "cross_window_stream_diversity": pair.pair.stream_diversity,
                    },
                    "evidence_summary": dict(window.metrics),
                    "continuity": {
                        "maximum_observed_timestamp_gap": max(gaps, default=None),
                        "maximum_observed_translation_m": max(
                            edge_translations, default=None
                        ),
                        "maximum_observed_rotation_degrees": max(
                            edge_rotations, default=None
                        ),
                    },
                    "frames": [
                        {
                            "episode_index": index,
                            "name": evidence.frame.name,
                            "colmap_image_id": evidence.frame.colmap_image_id,
                            "colmap_camera_id": evidence.frame.colmap_camera_id,
                            "physical_timestamp": (
                                evidence.frame.identity.physical_timestamp
                            ),
                            "is_anchor_preference": index == window.anchor_index,
                            "is_rescue_plan_anchor": evidence.rescue_anchor,
                            "T_world_camera_m": np.asarray(
                                evidence.frame.world_from_camera_m,
                                dtype=np.float64,
                            ).tolist(),
                            "object_evidence": {
                                "canonical_object_id": evidence.object_id,
                                "retrieval_source": "object_support_colmap_tracks",
                                "bbox_source": "projected_object_support",
                                "track_support_count": evidence.track_support_count,
                                "track_support_fraction": (
                                    evidence.track_support_fraction
                                ),
                                "projected_support_points": (
                                    evidence.projected_support_points
                                ),
                                "in_frame_support_ratio": (
                                    evidence.in_frame_support_ratio
                                ),
                                "area_ratio": evidence.area_ratio,
                                "margin_ratio": evidence.margin_ratio,
                                "geometric_score": evidence.geometric_score,
                                "view_direction": list(evidence.view_direction),
                            },
                        }
                        for index, evidence in enumerate(window.frames)
                    ],
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["anchor_object_id"]),
            0 if row["split"] == "train" else 1,
            str(row["episode_id"]),
        )
    )
    return rows


def validate_paired_episode_rows(
    episodes: Sequence[Mapping[str, object]],
    *,
    minimum_frames: int,
    maximum_frames: int,
) -> dict[str, object]:
    """Fail closed on all v2 pairing, evidence and leakage invariants."""

    if not episodes:
        raise ValueError("paired episode list is empty")
    by_object: dict[int, Counter[str]] = defaultdict(Counter)
    names: set[str] = set()
    train_timestamps: set[str] = set()
    heldout_timestamps: set[str] = set()
    for episode in episodes:
        object_id = int(episode.get("anchor_object_id", -1))
        split = str(episode.get("split") or "")
        if object_id < 0 or split not in {"train", "heldout"}:
            raise ValueError("episode lacks canonical object/split")
        frames = episode.get("frames")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise ValueError("episode frames must be a sequence")
        if not int(minimum_frames) <= len(frames) <= int(maximum_frames):
            raise ValueError("episode frame count violates bounds")
        if int(episode.get("object_evidence_frame_count", -1)) != len(frames):
            raise ValueError("not every episode frame carries object evidence")
        if not bool(episode.get("all_frames_object_visible")):
            raise ValueError("episode is not certified object-visible")
        camera = str(episode.get("camera") or "")
        family = str(episode.get("view_family") or "")
        stream_positions: list[int] = []
        timestamps: set[str] = set()
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise ValueError("frame row must be an object")
            name = str(frame.get("name") or "")
            if not name or name in names:
                raise ValueError(f"duplicate or empty RGB name: {name!r}")
            names.add(name)
            timestamp = str(frame.get("physical_timestamp") or "")
            if not timestamp:
                raise ValueError("frame lacks physical timestamp")
            timestamps.add(timestamp)
            evidence = frame.get("object_evidence")
            if not isinstance(evidence, Mapping):
                raise ValueError("frame lacks object support-track evidence")
            if int(evidence.get("canonical_object_id", -1)) != object_id:
                raise ValueError("frame evidence object differs from episode anchor")
            if evidence.get("retrieval_source") != "object_support_colmap_tracks":
                raise ValueError("non support-track retrieval source is forbidden")
            if evidence.get("bbox_source") != "projected_object_support":
                raise ValueError("OBB/detector projection is forbidden")
            if int(frame.get("episode_index", -1)) != index:
                raise ValueError("episode frame order is not exact")
            stream_positions.append(index)
        span = episode.get("registered_stream_span")
        if not isinstance(span, Mapping) or not bool(span.get("contiguous")):
            raise ValueError("episode lacks registered-stream continuity proof")
        start = int(span.get("start_index", -1))
        end = int(span.get("end_index", -1))
        if start < 0 or end - start + 1 != len(stream_positions):
            raise ValueError("registered-stream span does not match episode")
        by_object[object_id][split] += 1
        (train_timestamps if split == "train" else heldout_timestamps).update(
            timestamps
        )
        if not camera or not family:
            raise ValueError("episode lacks exact camera/view_family")
    unpaired = {
        object_id: dict(counts)
        for object_id, counts in by_object.items()
        if counts["train"] < 1 or counts["heldout"] < 1
    }
    if unpaired:
        raise ValueError(f"canonical object lacks train/heldout pair: {unpaired}")
    overlap = sorted(train_timestamps & heldout_timestamps)
    if overlap:
        raise ValueError(f"global physical timestamp leakage: {overlap[:5]}")
    return {
        "objects": len(by_object),
        "episodes": len(episodes),
        "frames": len(names),
        "train_physical_timestamps": len(train_timestamps),
        "heldout_physical_timestamps": len(heldout_timestamps),
        "global_timestamp_overlap": 0,
        "duplicate_rgb": 0,
    }
