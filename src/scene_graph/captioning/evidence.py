"""View-provenance contract for multi-pass semantic evidence.

Repeated VLM requests are not independent evidence when they inspect the same
source images.  This module provides a small, model-agnostic contract for
partitioning crop observations by source ``image_id``, recording relocatable
evidence fingerprints, and proving whether label-agreeing events have
sufficiently independent views.

The helpers intentionally know nothing about scene names, object classes, or
semantic labels.  Callers first filter to positive, label-agreeing events, then
pass those events to :func:`select_independent_evidence`.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar


EVIDENCE_CONTRACT_SCHEMA = "farm.semantic-view-evidence.v1"
PARTITION_CONTRACT_SCHEMA = "farm.semantic-view-partitions.v1"
CAMERA_POSE_CONTRACT_SCHEMA = "farm.semantic-camera-pose-evidence.v1"
DEFAULT_MIN_VIEWS_PER_EVENT = 3
DEFAULT_MIN_TOTAL_UNIQUE_VIEWS = 6
DEFAULT_MAX_VIEW_OVERLAP = 0.25
DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES = 10.0
DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE = 0.10
DEFAULT_MAX_PARTITIONS = 3
DEFAULT_CROP_DETAIL_FLOOR_RATIO = 0.80
DEFAULT_MAX_CROP_COMBINATIONS = 4096
POSE_AWARE_CROP_SELECTION_SCHEMA = "farm.pose-aware-crop-selection.v1"

# Sources that share a channel deliberately reuse a deterministic view fold.
# This keeps every request useful when tracks are short while making the
# correlation explicit.  In particular, B/C are disjoint once six views are
# available and B/C/paired are pairwise disjoint once nine views are available.
SOURCE_VIEW_CHANNELS: dict[str, int] = {
    "initial_blind_review": 0,
    "blind_pass_a": 0,
    "blind_pass_b": 0,
    "independent_verification": 1,
    "verification_pass": 1,
    "blind_pass_c": 1,
    "paired_crop_blind_adjudicator": 2,
    "paired_physical_form_guard": 2,
    "physical_form_guard": 2,
    # Physical recovery is first attempted on the guard fold. A second,
    # form-only verification uses channel 1 so it is disjoint whenever the
    # track has at least six source views. Both events remain confirmation-
    # ineligible; the split only gates a conservative probable noun.
    "physical_form_recovery": 2,
    "physical_form_recovery_verification": 1,
    "multi_source_fusion": 2,
    "neutral_form_fallback": 2,
}

# These names are two serializations of the same inference request in legacy
# review/ensemble artifacts and must never count as separate events.
EVENT_SOURCE_ALIASES: dict[str, str] = {
    "initial_blind_review": "initial_blind_review",
    "blind_pass_a": "initial_blind_review",
    "independent_verification": "independent_verification",
    "verification_pass": "independent_verification",
}

_IMAGE_ID_RE = re.compile(r"(?:^|_)img_(\d+)(?:_|\.|$)")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CandidateT = TypeVar("_CandidateT")


def _finite_vector(value: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float] | None:
    magnitude = _norm(vector)
    if not math.isfinite(magnitude) or magnitude <= 1e-9:
        return None
    return tuple(float(value) / magnitude for value in vector)  # type: ignore[return-value]


def _camera_pose_from_transform(value: object) -> dict[str, list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    rows = [_finite_vector(row, 4) for row in value]
    if any(row is None for row in rows):
        return None
    matrix = [tuple(row or ()) for row in rows]
    if any(
        abs(matrix[3][index] - expected) > 1e-5
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        return None
    rotation_columns = [
        (matrix[0][column], matrix[1][column], matrix[2][column])
        for column in range(3)
    ]
    if any(abs(_norm(column) - 1.0) > 2e-3 for column in rotation_columns):
        return None
    if any(
        abs(sum(left[index] * right[index] for index in range(3))) > 2e-3
        for left, right in itertools.combinations(rotation_columns, 2)
    ):
        return None
    forward = _unit(rotation_columns[2])
    if forward is None:
        return None
    return {
        "camera_center_world_m": [matrix[index][3] for index in range(3)],
        "camera_forward_world": list(forward),
    }


def load_frame_pose_index(frames_json: Path) -> dict[int, dict[str, list[float]]]:
    """Load validated metric ``T_world_cam`` poses keyed by FARM frame index.

    Crop sidecars encode the zero-based index into ``frames`` as ``img_XXXXXX``.
    Invalid individual transforms are omitted so downstream evidence fails
    closed instead of silently treating an image id as a camera viewpoint.
    """

    payload = json.loads(Path(frames_json).read_text(encoding="utf-8"))
    units = str(payload.get("pose_translation_units") or "").strip().lower()
    if units not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError(f"frames.json camera translations are not metric: {units!r}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("frames.json is missing a frames list")
    result: dict[int, dict[str, list[float]]] = {}
    for index, row in enumerate(frames):
        pose = _camera_pose_from_transform(
            row.get("T_world_cam") if isinstance(row, Mapping) else None
        )
        if pose is not None:
            result[int(index)] = pose
    return result


def _pose_canonical(
    target: Sequence[float],
    poses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CAMERA_POSE_CONTRACT_SCHEMA,
        "target_position_world_m": [float(value) for value in target],
        "camera_poses": [
            {
                "image_id": int(row["image_id"]),
                "camera_center_world_m": [
                    float(value) for value in row["camera_center_world_m"]
                ],
                "camera_forward_world": [
                    float(value) for value in row["camera_forward_world"]
                ],
            }
            for row in sorted(poses, key=lambda item: int(item["image_id"]))
        ],
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_path(candidate: Any) -> Path:
    if isinstance(candidate, (str, Path)):
        return Path(candidate)
    if isinstance(candidate, (tuple, list)) and candidate:
        value = candidate[0]
        if isinstance(value, (str, Path)):
            return Path(value)
    raise TypeError("Crop candidates must be paths or non-empty (path, payload) pairs")


def crop_image_id(candidate: Any) -> int | None:
    """Extract the mapping image id encoded in a FARM crop sidecar name."""

    match = _IMAGE_ID_RE.search(_candidate_path(candidate).name)
    return int(match.group(1)) if match else None


def _source_channel(source: str) -> int:
    normalized = str(source or "").strip().lower()
    if normalized in SOURCE_VIEW_CHANNELS:
        return SOURCE_VIEW_CHANNELS[normalized]
    # Unknown future passes remain deterministic without extending a manual
    # scene/object policy.  Only the source name affects their channel.
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % DEFAULT_MAX_PARTITIONS


def _partition_count(
    unique_image_count: int,
    *,
    min_views_per_partition: int,
    max_partitions: int,
) -> int:
    minimum = max(1, int(min_views_per_partition))
    maximum = max(1, int(max_partitions))
    return max(1, min(maximum, int(unique_image_count) // minimum))


def partition_crop_candidates(
    candidates: Sequence[_CandidateT],
    source: str,
    *,
    min_views_per_partition: int = DEFAULT_MIN_VIEWS_PER_EVENT,
    max_partitions: int = DEFAULT_MAX_PARTITIONS,
) -> tuple[list[_CandidateT], dict[str, Any]]:
    """Return one deterministic temporal view fold for ``source``.

    All sidecars from one ``image_id`` are assigned to the same fold.  Parsed
    image ids are sorted and distributed round-robin, which gives each fold
    broad temporal coverage.  Unparseable legacy names are retained in a
    deterministic fold but later manifests mark their provenance incomplete,
    so they can be reviewed without being used to claim confirmation.
    """

    parsed_ids = sorted(
        {
            image_id
            for image_id in (crop_image_id(candidate) for candidate in candidates)
            if image_id is not None
        }
    )
    partition_count = _partition_count(
        len(parsed_ids),
        min_views_per_partition=min_views_per_partition,
        max_partitions=max_partitions,
    )
    channel = _source_channel(source)
    partition_index = channel % partition_count
    image_partitions = {
        image_id: index % partition_count
        for index, image_id in enumerate(parsed_ids)
    }

    selected: list[_CandidateT] = []
    unparsed_count = 0
    for candidate in candidates:
        image_id = crop_image_id(candidate)
        if image_id is None:
            unparsed_count += 1
            name = _candidate_path(candidate).name
            digest = hashlib.sha256(name.encode("utf-8")).digest()
            candidate_partition = int.from_bytes(digest[:4], "big") % partition_count
        else:
            candidate_partition = image_partitions[image_id]
        if candidate_partition == partition_index:
            selected.append(candidate)

    selected.sort(
        key=lambda candidate: (
            crop_image_id(candidate) is None,
            crop_image_id(candidate) if crop_image_id(candidate) is not None else math.inf,
            _candidate_path(candidate).name,
        )
    )
    partition_image_ids = sorted(
        {
            image_id
            for image_id in (crop_image_id(candidate) for candidate in selected)
            if image_id is not None
        }
    )
    metadata = {
        "schema": PARTITION_CONTRACT_SCHEMA,
        "source": str(source),
        "source_channel": int(channel),
        "partition_id": f"view-fold-{partition_index}-of-{partition_count}",
        "partition_index": int(partition_index),
        "partition_count": int(partition_count),
        "unique_image_count": len(parsed_ids),
        "partition_image_ids": partition_image_ids,
        "unparsed_candidate_count": int(unparsed_count),
    }
    return selected, metadata


def _candidate_detail(candidate: Any) -> int:
    """Return the crop-detail proxy already used by the review selector."""

    if isinstance(candidate, (tuple, list)) and len(candidate) > 1:
        try:
            return max(0, int(len(candidate[1])))
        except (TypeError, ValueError):
            return 0
    return 0


def _crop_selection_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float]:
    """Return minimum angle, median angle and minimum normalized baseline."""

    if len(rows) < 2:
        return 0.0, 0.0, 0.0
    angles: list[float] = []
    normalized_baselines: list[float] = []
    for left, right in itertools.combinations(rows, 2):
        angles.append(_angle_degrees(left["view_ray"], right["view_ray"]))
        baseline = _distance(
            left["camera_center_world_m"], right["camera_center_world_m"]
        )
        normalized_baselines.append(
            baseline
            / max(
                1e-9,
                (float(left["range_m"]) + float(right["range_m"])) / 2.0,
            )
        )
    return (
        float(min(angles)),
        float(statistics.median(angles)),
        float(min(normalized_baselines)),
    )


def _selection_diagnostics(
    *,
    method: str,
    reason: str,
    requested: int,
    candidate_count: int,
    posed_unique_count: int,
    fallback_detail: int,
    selected_detail: int,
    metrics: tuple[float, float, float] | None,
    detail_floor_ratio: float,
    min_viewpoint_separation_degrees: float,
    min_normalized_camera_baseline: float,
    combination_count: int,
    evaluated_combination_count: int,
    target_position_complete: bool,
) -> dict[str, Any]:
    minimum_angle, median_angle, minimum_baseline = metrics or (
        None,
        None,
        None,
    )
    detail_ratio = (
        float(selected_detail) / float(fallback_detail)
        if fallback_detail > 0
        else (1.0 if selected_detail == 0 else None)
    )
    gate_met = bool(
        metrics is not None
        and minimum_angle is not None
        and minimum_baseline is not None
        and minimum_angle >= min_viewpoint_separation_degrees
        and minimum_baseline >= min_normalized_camera_baseline
    )
    return {
        "schema": POSE_AWARE_CROP_SELECTION_SCHEMA,
        "method": str(method),
        "reason": str(reason),
        "requested_crop_count": int(requested),
        "partition_candidate_count": int(candidate_count),
        "posed_unique_image_count": int(posed_unique_count),
        "target_position_complete": bool(target_position_complete),
        "selected_pose_provenance_complete": bool(
            metrics is not None and posed_unique_count >= requested
        ),
        "quality_proxy": "grounded_jpeg_bytes",
        "detail_floor_ratio": float(detail_floor_ratio),
        "fallback_detail_bytes": int(fallback_detail),
        "selected_detail_bytes": int(selected_detail),
        "detail_ratio_to_temporal_fallback": detail_ratio,
        "min_pairwise_viewpoint_angle_degrees": minimum_angle,
        "median_pairwise_viewpoint_angle_degrees": median_angle,
        "min_pairwise_normalized_camera_baseline": minimum_baseline,
        "min_viewpoint_separation_degrees": float(
            min_viewpoint_separation_degrees
        ),
        "min_normalized_camera_baseline": float(
            min_normalized_camera_baseline
        ),
        "diversity_gate_met": gate_met,
        "candidate_combination_count": int(combination_count),
        "evaluated_combination_count": int(evaluated_combination_count),
    }


def select_pose_diverse_crop_candidates(
    candidates: Sequence[_CandidateT],
    fallback: Sequence[_CandidateT],
    requested: int,
    *,
    frame_pose_index: Mapping[int, Mapping[str, Any]] | None,
    object_position_world_m: object,
    detail_floor_ratio: float = DEFAULT_CROP_DETAIL_FLOOR_RATIO,
    min_viewpoint_separation_degrees: float = (
        DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES
    ),
    min_normalized_camera_baseline: float = (
        DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE
    ),
    max_combinations: int = DEFAULT_MAX_CROP_COMBINATIONS,
) -> tuple[list[_CandidateT], dict[str, Any]]:
    """Select diverse crops without changing image count or model calls.

    The supplied fallback is the existing temporal/detail selection and is
    returned unchanged whenever metric pose provenance is insufficient. A
    replacement must retain a fixed fraction of its aggregate grounded-JPEG
    detail. Search is exact for ordinary FARM crop banks and deterministically
    beam-bounded for unusually large banks.
    """

    fallback_rows = list(fallback)
    requested_count = max(1, int(requested))
    detail_floor = min(1.0, max(0.0, float(detail_floor_ratio)))
    angle_threshold = max(0.0, float(min_viewpoint_separation_degrees))
    baseline_threshold = max(0.0, float(min_normalized_camera_baseline))
    search_cap = max(1, int(max_combinations))
    fallback_detail = sum(
        _candidate_detail(candidate) for candidate in fallback_rows
    )
    target = _finite_vector(object_position_world_m, 3)

    def fallback_result(
        reason: str,
        *,
        posed_unique_count: int = 0,
        combination_count: int = 0,
    ) -> tuple[list[_CandidateT], dict[str, Any]]:
        return fallback_rows, _selection_diagnostics(
            method="temporal_fallback",
            reason=reason,
            requested=requested_count,
            candidate_count=len(candidates),
            posed_unique_count=posed_unique_count,
            fallback_detail=fallback_detail,
            selected_detail=fallback_detail,
            metrics=None,
            detail_floor_ratio=detail_floor,
            min_viewpoint_separation_degrees=angle_threshold,
            min_normalized_camera_baseline=baseline_threshold,
            combination_count=combination_count,
            evaluated_combination_count=0,
            target_position_complete=target is not None,
        )

    if requested_count < 2:
        return fallback_result(
            "requested_crop_count_below_pose_diversity_minimum"
        )
    if len(fallback_rows) != requested_count:
        return fallback_result("fallback_crop_count_below_requested")
    if target is None:
        return fallback_result("invalid_or_missing_object_position")
    if not isinstance(frame_pose_index, Mapping):
        return fallback_result("missing_frame_pose_index")

    # One observation image is one viewpoint. Prefer its highest-detail crop
    # and use the canonical path as a deterministic tie-breaker.
    unique_candidates: dict[int, _CandidateT] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            crop_image_id(item) is None,
            (
                crop_image_id(item)
                if crop_image_id(item) is not None
                else math.inf
            ),
            -_candidate_detail(item),
            str(_candidate_path(item)),
        ),
    ):
        image_id = crop_image_id(candidate)
        if image_id is not None:
            unique_candidates.setdefault(int(image_id), candidate)

    posed: list[dict[str, Any]] = []
    for image_id in sorted(unique_candidates):
        raw_pose = frame_pose_index.get(image_id)
        if not isinstance(raw_pose, Mapping):
            continue
        center = _finite_vector(raw_pose.get("camera_center_world_m"), 3)
        forward = _unit(
            _finite_vector(raw_pose.get("camera_forward_world"), 3) or ()
        )
        if center is None or forward is None:
            continue
        view_ray = _view_ray(center, target)
        if view_ray is None:
            continue
        candidate = unique_candidates[image_id]
        posed.append(
            {
                "candidate": candidate,
                "image_id": int(image_id),
                "camera_center_world_m": center,
                "view_ray": view_ray,
                "range_m": _distance(center, target),
                "detail": _candidate_detail(candidate),
            }
        )

    if len(posed) < requested_count:
        return fallback_result(
            "insufficient_unique_posed_views",
            posed_unique_count=len(posed),
        )

    combination_count = math.comb(len(posed), requested_count)

    def partial_rank(indices: tuple[int, ...]) -> tuple[Any, ...]:
        rows = [posed[index] for index in indices]
        if len(rows) >= 2:
            minimum_angle, _, minimum_baseline = _crop_selection_metrics(rows)
        else:
            minimum_angle, minimum_baseline = 0.0, 0.0
        details = [int(row["detail"]) for row in rows]
        return (
            minimum_angle,
            minimum_baseline,
            min(details) if details else 0,
            sum(details),
            tuple(-int(row["image_id"]) for row in rows),
        )

    search_mode = "pose_aware_exact"
    if combination_count <= search_cap:
        index_combinations: Iterable[tuple[int, ...]] = (
            itertools.combinations(range(len(posed)), requested_count)
        )
    else:
        search_mode = "pose_aware_beam"
        beam: list[tuple[int, ...]] = [tuple()]
        for depth in range(requested_count):
            remaining = requested_count - depth - 1
            expanded = (
                prefix + (index,)
                for prefix in beam
                for index in range(
                    (prefix[-1] + 1) if prefix else 0,
                    len(posed) - remaining,
                )
            )
            beam = heapq.nlargest(
                search_cap,
                expanded,
                key=partial_rank,
            )
            if not beam:
                break
        posed_index_by_id = {
            int(row["image_id"]): index for index, row in enumerate(posed)
        }
        fallback_image_ids = [
            crop_image_id(candidate) for candidate in fallback_rows
        ]
        if (
            all(image_id is not None for image_id in fallback_image_ids)
            and len(set(fallback_image_ids)) == requested_count
            and all(
                int(image_id) in posed_index_by_id
                for image_id in fallback_image_ids
                if image_id is not None
            )
        ):
            fallback_indices = tuple(
                sorted(
                    posed_index_by_id[int(image_id)]
                    for image_id in fallback_image_ids
                    if image_id is not None
                )
            )
            if fallback_indices not in beam:
                if len(beam) >= search_cap:
                    beam[-1] = fallback_indices
                else:
                    beam.append(fallback_indices)
        index_combinations = beam

    best_pass: tuple[
        tuple[Any, ...],
        list[_CandidateT],
        tuple[float, float, float],
        int,
    ] | None = None
    best_available: tuple[
        tuple[Any, ...],
        list[_CandidateT],
        tuple[float, float, float],
        int,
    ] | None = None
    evaluated = 0
    minimum_detail = detail_floor * float(fallback_detail)
    for indices in index_combinations:
        if len(indices) != requested_count:
            continue
        rows = [posed[index] for index in indices]
        selected = [row["candidate"] for row in rows]
        selected_detail = sum(int(row["detail"]) for row in rows)
        if selected_detail + 1e-9 < minimum_detail:
            continue
        evaluated += 1
        metrics = _crop_selection_metrics(rows)
        minimum_angle, median_angle, minimum_baseline = metrics
        stable_ids = tuple(-int(row["image_id"]) for row in rows)
        details = [int(row["detail"]) for row in rows]
        passes = bool(
            minimum_angle >= angle_threshold
            and minimum_baseline >= baseline_threshold
        )
        if passes:
            rank = (
                min(details),
                selected_detail,
                minimum_angle,
                minimum_baseline,
                median_angle,
                stable_ids,
            )
            if best_pass is None or rank > best_pass[0]:
                best_pass = (rank, selected, metrics, selected_detail)
        rank = (
            minimum_angle,
            minimum_baseline,
            min(details),
            selected_detail,
            median_angle,
            stable_ids,
        )
        if best_available is None or rank > best_available[0]:
            best_available = (rank, selected, metrics, selected_detail)

    choice = best_pass or best_available
    if choice is None:
        return fallback_result(
            "no_pose_combination_meets_detail_floor",
            posed_unique_count=len(posed),
            combination_count=combination_count,
        )
    _, selected, metrics, selected_detail = choice
    reason = (
        "diversity_gate_met"
        if best_pass is not None
        else "best_available_below_diversity_gate"
    )
    diagnostics = _selection_diagnostics(
        method=search_mode,
        reason=reason,
        requested=requested_count,
        candidate_count=len(candidates),
        posed_unique_count=len(posed),
        fallback_detail=fallback_detail,
        selected_detail=selected_detail,
        metrics=metrics,
        detail_floor_ratio=detail_floor,
        min_viewpoint_separation_degrees=angle_threshold,
        min_normalized_camera_baseline=baseline_threshold,
        combination_count=combination_count,
        evaluated_combination_count=evaluated,
        target_position_complete=True,
    )
    return selected, diagnostics


def crop_evidence_manifest(
    object_id: int,
    crops: Sequence[Any],
    *,
    partition: Mapping[str, Any] | None = None,
    frame_pose_index: Mapping[int, Mapping[str, Any]] | None = None,
    object_position_world_m: object = None,
) -> dict[str, Any]:
    """Build relocatable image and camera-pose provenance for one request."""

    paths = [_candidate_path(candidate) for candidate in crops]
    parsed = [crop_image_id(path) for path in paths]
    image_ids = sorted({int(value) for value in parsed if value is not None})
    observation_names = sorted(path.name for path in paths)
    canonical = {
        "schema": EVIDENCE_CONTRACT_SCHEMA,
        "object_id": int(object_id),
        "crop_image_ids": image_ids,
        "crop_observation_names": observation_names,
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    result: dict[str, Any] = {
        "evidence_contract_schema": EVIDENCE_CONTRACT_SCHEMA,
        "evidence_object_id": int(object_id),
        "crop_image_ids": image_ids,
        "crop_image_ids_complete": all(value is not None for value in parsed),
        "crop_observation_names": observation_names,
        "crop_observation_count": len(paths),
        "evidence_fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    target = _finite_vector(object_position_world_m, 3)
    pose_rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        raw_pose = (frame_pose_index or {}).get(image_id)
        if not isinstance(raw_pose, Mapping):
            continue
        center = _finite_vector(raw_pose.get("camera_center_world_m"), 3)
        forward = _unit(_finite_vector(raw_pose.get("camera_forward_world"), 3) or ())
        if center is None or forward is None:
            continue
        pose_rows.append({
            "image_id": int(image_id),
            "camera_center_world_m": list(center),
            "camera_forward_world": list(forward),
        })
    pose_complete = bool(
        result["crop_image_ids_complete"]
        and image_ids
        and target is not None
        and len(pose_rows) == len(image_ids)
    )
    pose_payload = (
        _pose_canonical(target, pose_rows)
        if pose_complete and target is not None
        else None
    )
    result.update({
        "camera_pose_contract_schema": CAMERA_POSE_CONTRACT_SCHEMA,
        "camera_pose_source": "rgbd/frames.json:T_world_cam",
        "camera_pose_provenance_complete": pose_complete,
        "target_position_world_m": list(target) if target is not None else [],
        "crop_camera_poses": pose_rows,
        "camera_pose_fingerprint_sha256": (
            _canonical_sha256(pose_payload) if pose_payload is not None else ""
        ),
    })
    if partition is not None:
        result.update({
            "crop_partition_id": str(partition.get("partition_id") or ""),
            "crop_partition_index": int(partition.get("partition_index") or 0),
            "crop_partition_count": int(partition.get("partition_count") or 1),
        })
        selection = partition.get("selection_diagnostics")
        if isinstance(selection, Mapping):
            result["crop_selection_diagnostics"] = dict(selection)
    return result


def evidence_event_id(event: Mapping[str, Any]) -> str:
    """Return a stable event identity while collapsing known legacy aliases."""

    source = str(event.get("source") or "").strip().lower()
    if source in EVENT_SOURCE_ALIASES:
        return EVENT_SOURCE_ALIASES[source]
    return str(
        event.get("event_id")
        or event.get("evidence_event_id")
        or event.get("request_id")
        or source
    ).strip()


def evidence_view_ids(event: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Return verified unique view ids, or ``None`` for incomplete provenance."""

    if event.get("crop_image_ids_complete") is not True:
        return None
    fingerprint = str(event.get("evidence_fingerprint_sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(fingerprint):
        return None
    raw = event.get("crop_image_ids")
    if not isinstance(raw, (list, tuple)):
        return None
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            return None
        try:
            integer = int(value)
        except (TypeError, ValueError):
            return None
        if integer < 0 or float(value) != float(integer):
            return None
        values.append(integer)
    return tuple(sorted(set(values)))


def evidence_camera_poses(
    event: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[dict[str, Any], ...]] | None:
    """Return hash-verified target/pose provenance, or ``None``."""

    if event.get("camera_pose_provenance_complete") is not True:
        return None
    fingerprint = str(
        event.get("camera_pose_fingerprint_sha256") or ""
    ).strip().lower()
    if not _SHA256_RE.fullmatch(fingerprint):
        return None
    target = _finite_vector(event.get("target_position_world_m"), 3)
    raw_rows = event.get("crop_camera_poses")
    view_ids = evidence_view_ids(event)
    if target is None or not isinstance(raw_rows, (list, tuple)) or view_ids is None:
        return None
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or isinstance(raw.get("image_id"), bool):
            return None
        try:
            image_id = int(raw.get("image_id"))
        except (TypeError, ValueError):
            return None
        center = _finite_vector(raw.get("camera_center_world_m"), 3)
        forward = _unit(_finite_vector(raw.get("camera_forward_world"), 3) or ())
        if image_id < 0 or image_id in seen or center is None or forward is None:
            return None
        seen.add(image_id)
        rows.append({
            "image_id": image_id,
            "camera_center_world_m": list(center),
            "camera_forward_world": list(forward),
        })
    if tuple(sorted(seen)) != view_ids:
        return None
    canonical = _pose_canonical(target, rows)
    if _canonical_sha256(canonical) != fingerprint:
        return None
    return target, tuple(sorted(rows, key=lambda item: item["image_id"]))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return _norm(tuple(float(a) - float(b) for a, b in zip(left, right)))


def _view_ray(
    camera_center: Sequence[float], target: Sequence[float]
) -> tuple[float, float, float] | None:
    return _unit(tuple(float(target[index]) - float(camera_center[index]) for index in range(3)))


def _angle_degrees(left: Sequence[float], right: Sequence[float]) -> float:
    cosine = sum(float(a) * float(b) for a, b in zip(left, right))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def evidence_pose_diversity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    min_viewpoint_separation_degrees: float = DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES,
    min_normalized_camera_baseline: float = DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE,
) -> dict[str, Any]:
    """Measure robust cross-event viewpoint and camera-baseline diversity.

    Different image ids are insufficient: temporally adjacent frames may show
    the same side of an object.  We use the smaller of the two directional
    median-nearest statistics, so every event must contribute genuinely new
    object-centric rays and camera centres.
    """

    left_id, right_id = evidence_event_id(left), evidence_event_id(right)
    result: dict[str, Any] = {
        "left_event_id": left_id,
        "right_event_id": right_id,
        "pose_independent": False,
        "reason": "incomplete_camera_pose_provenance",
        "left_to_right_median_nearest_viewpoint_angle_degrees": None,
        "right_to_left_median_nearest_viewpoint_angle_degrees": None,
        "symmetric_median_nearest_viewpoint_angle_degrees": None,
        "left_to_right_median_nearest_baseline_m": None,
        "right_to_left_median_nearest_baseline_m": None,
        "symmetric_median_nearest_baseline_m": None,
        "left_to_right_median_nearest_normalized_baseline": None,
        "right_to_left_median_nearest_normalized_baseline": None,
        "symmetric_median_nearest_normalized_baseline": None,
        "min_viewpoint_separation_degrees": float(
            max(0.0, min_viewpoint_separation_degrees)
        ),
        "min_normalized_camera_baseline": float(
            max(0.0, min_normalized_camera_baseline)
        ),
    }
    left_pose, right_pose = evidence_camera_poses(left), evidence_camera_poses(right)
    if left_pose is None or right_pose is None:
        return result
    left_target, left_rows = left_pose
    right_target, right_rows = right_pose
    if _distance(left_target, right_target) > 1e-4:
        result["reason"] = "target_position_mismatch"
        return result

    left_centres = [row["camera_center_world_m"] for row in left_rows]
    right_centres = [row["camera_center_world_m"] for row in right_rows]
    left_rays = [_view_ray(center, left_target) for center in left_centres]
    right_rays = [_view_ray(center, left_target) for center in right_centres]
    if any(ray is None for ray in left_rays + right_rays):
        result["reason"] = "camera_at_target_position"
        return result
    angle_matrix = [
        [_angle_degrees(left_ray or (), right_ray or ()) for right_ray in right_rays]
        for left_ray in left_rays
    ]
    baseline_matrix = [
        [_distance(left_center, right_center) for right_center in right_centres]
        for left_center in left_centres
    ]
    left_ranges = [_distance(center, left_target) for center in left_centres]
    right_ranges = [_distance(center, left_target) for center in right_centres]
    normalized_matrix = [
        [
            baseline_matrix[i][j] / max(1e-9, (left_ranges[i] + right_ranges[j]) / 2.0)
            for j in range(len(right_centres))
        ]
        for i in range(len(left_centres))
    ]

    def directional(matrix: Sequence[Sequence[float]]) -> tuple[float, float]:
        forward = float(statistics.median(min(row) for row in matrix))
        reverse = float(statistics.median(
            min(matrix[i][j] for i in range(len(matrix)))
            for j in range(len(matrix[0]))
        ))
        return forward, reverse

    angle_lr, angle_rl = directional(angle_matrix)
    baseline_lr, baseline_rl = directional(baseline_matrix)
    normalized_lr, normalized_rl = directional(normalized_matrix)
    angle_symmetric = min(angle_lr, angle_rl)
    baseline_symmetric = min(baseline_lr, baseline_rl)
    normalized_symmetric = min(normalized_lr, normalized_rl)
    result.update({
        "left_to_right_median_nearest_viewpoint_angle_degrees": angle_lr,
        "right_to_left_median_nearest_viewpoint_angle_degrees": angle_rl,
        "symmetric_median_nearest_viewpoint_angle_degrees": angle_symmetric,
        "left_to_right_median_nearest_baseline_m": baseline_lr,
        "right_to_left_median_nearest_baseline_m": baseline_rl,
        "symmetric_median_nearest_baseline_m": baseline_symmetric,
        "left_to_right_median_nearest_normalized_baseline": normalized_lr,
        "right_to_left_median_nearest_normalized_baseline": normalized_rl,
        "symmetric_median_nearest_normalized_baseline": normalized_symmetric,
    })
    if angle_symmetric < result["min_viewpoint_separation_degrees"]:
        result["reason"] = "viewpoint_separation_below_threshold"
        return result
    if normalized_symmetric < result["min_normalized_camera_baseline"]:
        result["reason"] = "camera_baseline_below_threshold"
        return result
    result["pose_independent"] = True
    result["reason"] = "diverse_camera_pose_support"
    return result


def evidence_view_overlap(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    min_views_per_event: int = DEFAULT_MIN_VIEWS_PER_EVENT,
    max_overlap_coefficient: float = DEFAULT_MAX_VIEW_OVERLAP,
    min_viewpoint_separation_degrees: float = DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES,
    min_normalized_camera_baseline: float = DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE,
) -> dict[str, Any]:
    """Quantify correlation between two inference events.

    The overlap coefficient, rather than Jaccard alone, makes a subset of a
    larger crop set fully correlated (coefficient 1.0).
    """

    left_id, right_id = evidence_event_id(left), evidence_event_id(right)
    left_views, right_views = evidence_view_ids(left), evidence_view_ids(right)
    result: dict[str, Any] = {
        "left_event_id": left_id,
        "right_event_id": right_id,
        "left_view_count": len(left_views) if left_views is not None else None,
        "right_view_count": len(right_views) if right_views is not None else None,
        "intersection_count": None,
        "union_count": None,
        "overlap_coefficient": None,
        "jaccard": None,
        "independent": False,
        "reason": "incomplete_view_provenance",
        "pose_diversity": {},
    }
    if not left_id or not right_id:
        result["reason"] = "missing_event_id"
        return result
    if left_id == right_id:
        result["reason"] = "same_inference_event"
        return result
    if left.get("confirmation_eligible") is False or right.get("confirmation_eligible") is False:
        result["reason"] = "confirmation_ineligible_event"
        return result
    if left_views is None or right_views is None:
        return result
    minimum = max(1, int(min_views_per_event))
    if len(left_views) < minimum or len(right_views) < minimum:
        result["reason"] = "insufficient_unique_views"
        return result

    left_set, right_set = set(left_views), set(right_views)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    overlap = intersection / min(len(left_set), len(right_set))
    jaccard = intersection / union if union else 1.0
    result.update({
        "intersection_count": int(intersection),
        "union_count": int(union),
        "overlap_coefficient": float(overlap),
        "jaccard": float(jaccard),
    })
    left_fingerprint = str(left.get("evidence_fingerprint_sha256") or "")
    right_fingerprint = str(right.get("evidence_fingerprint_sha256") or "")
    if left_fingerprint == right_fingerprint:
        result["reason"] = "same_evidence_fingerprint"
        return result
    threshold = min(1.0, max(0.0, float(max_overlap_coefficient)))
    if overlap > threshold:
        result["reason"] = "view_overlap_above_threshold"
        return result
    pose = evidence_pose_diversity(
        left,
        right,
        min_viewpoint_separation_degrees=min_viewpoint_separation_degrees,
        min_normalized_camera_baseline=min_normalized_camera_baseline,
    )
    result["pose_diversity"] = pose
    if not pose["pose_independent"]:
        result["reason"] = str(pose["reason"])
        return result
    result["independent"] = True
    result["reason"] = "independent_image_ids_and_camera_poses"
    return result


def _confidence(event: Mapping[str, Any]) -> float:
    try:
        value = float(event.get("confidence") or event.get("review_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _deduplicate_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for raw in events:
        event = dict(raw)
        event_id = evidence_event_id(event)
        if not event_id:
            continue
        current = by_id.get(event_id)
        candidate_rank = (
            evidence_view_ids(event) is not None,
            len(evidence_view_ids(event) or ()),
            _confidence(event),
        )
        current_rank = (
            evidence_view_ids(current) is not None,
            len(evidence_view_ids(current) or ()),
            _confidence(current),
        ) if current is not None else (False, -1, -1.0)
        if current is None or candidate_rank > current_rank:
            event["event_id"] = event_id
            by_id[event_id] = event
    return [by_id[key] for key in sorted(by_id)]


def select_independent_evidence(
    events: Iterable[Mapping[str, Any]],
    *,
    min_views_per_event: int = DEFAULT_MIN_VIEWS_PER_EVENT,
    min_total_unique_views: int = DEFAULT_MIN_TOTAL_UNIQUE_VIEWS,
    max_overlap_coefficient: float = DEFAULT_MAX_VIEW_OVERLAP,
    min_viewpoint_separation_degrees: float = DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES,
    min_normalized_camera_baseline: float = DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the largest pairwise-independent event group.

    ``events`` must already be positive and label-agreeing.  The returned list
    is empty unless at least two events prove independent view support.  The
    diagnostic payload is JSON-serializable and explains every fail-closed
    downgrade for legacy or correlated evidence.
    """

    deduplicated = _deduplicate_events(events)
    eligible: list[dict[str, Any]] = []
    incomplete: list[str] = []
    incomplete_pose: list[str] = []
    under_observed: list[str] = []
    explicitly_ineligible: list[str] = []
    minimum = max(1, int(min_views_per_event))
    minimum_total = max(1, int(min_total_unique_views))
    for event in deduplicated:
        event_id = evidence_event_id(event)
        views = evidence_view_ids(event)
        if event.get("confirmation_eligible") is False:
            explicitly_ineligible.append(event_id)
        elif views is None:
            incomplete.append(event_id)
        elif len(views) < minimum:
            under_observed.append(event_id)
        elif evidence_camera_poses(event) is None:
            incomplete_pose.append(event_id)
        else:
            eligible.append(event)

    overlaps = [
        evidence_view_overlap(
            left,
            right,
            min_views_per_event=minimum,
            max_overlap_coefficient=max_overlap_coefficient,
            min_viewpoint_separation_degrees=min_viewpoint_separation_degrees,
            min_normalized_camera_baseline=min_normalized_camera_baseline,
        )
        for left, right in itertools.combinations(eligible, 2)
    ]
    overlap_by_pair = {
        frozenset((row["left_event_id"], row["right_event_id"])): row
        for row in overlaps
    }

    valid_groups: list[tuple[dict[str, Any], ...]] = []
    for size in range(len(eligible), 1, -1):
        for group in itertools.combinations(eligible, size):
            view_union = set().union(
                *(set(evidence_view_ids(event) or ()) for event in group)
            )
            if len(view_union) >= minimum_total and all(
                overlap_by_pair[
                    frozenset((evidence_event_id(left), evidence_event_id(right)))
                ]["independent"]
                for left, right in itertools.combinations(group, 2)
            ):
                valid_groups.append(group)
        if valid_groups:
            break

    def group_rank(group: tuple[dict[str, Any], ...]) -> tuple[Any, ...]:
        view_union = set().union(*(set(evidence_view_ids(event) or ()) for event in group))
        group_overlaps = [
            float(
                overlap_by_pair[
                    frozenset((evidence_event_id(left), evidence_event_id(right)))
                ]["overlap_coefficient"]
            )
            for left, right in itertools.combinations(group, 2)
        ]
        return (
            -len(group),
            -len(view_union),
            max(group_overlaps, default=0.0),
            -sum(_confidence(event) for event in group),
            tuple(evidence_event_id(event) for event in group),
        )

    selected = list(min(valid_groups, key=group_rank)) if valid_groups else []
    reason = (
        "independent_view_consensus"
        if selected
        else "insufficient_provenance_complete_events"
        if len(eligible) < 2
        else "correlated_view_evidence"
    )
    diagnostics = {
        "schema": EVIDENCE_CONTRACT_SCHEMA,
        "candidate_event_count": len(deduplicated),
        "eligible_event_count": len(eligible),
        "selected_event_count": len(selected),
        "selected_event_ids": [evidence_event_id(event) for event in selected],
        "incomplete_provenance_event_ids": sorted(incomplete),
        "incomplete_pose_provenance_event_ids": sorted(incomplete_pose),
        "under_observed_event_ids": sorted(under_observed),
        "confirmation_ineligible_event_ids": sorted(explicitly_ineligible),
        "min_views_per_event": minimum,
        "min_total_unique_views": minimum_total,
        "max_overlap_coefficient": float(
            min(1.0, max(0.0, float(max_overlap_coefficient)))
        ),
        "min_viewpoint_separation_degrees": float(
            max(0.0, min_viewpoint_separation_degrees)
        ),
        "min_normalized_camera_baseline": float(
            max(0.0, min_normalized_camera_baseline)
        ),
        "pairwise_overlaps": overlaps,
        "confirmation_ready": len(selected) >= 2,
        "reason": reason,
    }
    return selected, diagnostics


__all__ = [
    "CAMERA_POSE_CONTRACT_SCHEMA",
    "DEFAULT_CROP_DETAIL_FLOOR_RATIO",
    "DEFAULT_MAX_CROP_COMBINATIONS",
    "DEFAULT_MAX_PARTITIONS",
    "DEFAULT_MAX_VIEW_OVERLAP",
    "DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE",
    "DEFAULT_MIN_VIEWS_PER_EVENT",
    "DEFAULT_MIN_TOTAL_UNIQUE_VIEWS",
    "DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES",
    "EVIDENCE_CONTRACT_SCHEMA",
    "EVENT_SOURCE_ALIASES",
    "PARTITION_CONTRACT_SCHEMA",
    "POSE_AWARE_CROP_SELECTION_SCHEMA",
    "SOURCE_VIEW_CHANNELS",
    "crop_evidence_manifest",
    "crop_image_id",
    "evidence_camera_poses",
    "evidence_event_id",
    "evidence_pose_diversity",
    "evidence_view_ids",
    "evidence_view_overlap",
    "load_frame_pose_index",
    "partition_crop_candidates",
    "select_pose_diverse_crop_candidates",
    "select_independent_evidence",
]
