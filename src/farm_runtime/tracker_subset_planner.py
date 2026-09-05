"""CPU-only planning primitives for bounded multi-view tracker episodes.

The tracker benchmark must not be assembled by sparsely mixing convenient
frames.  This module gives it an explicit, scene-agnostic contract:

* an operator-provided named regex defines physical capture identity;
* exact ``camera x view_family`` streams are never mixed;
* physical timestamps receive one global train/heldout fold;
* temporal and camera-pose discontinuities terminate an episode; and
* selection balances streams while existing rescue views are only a soft
  preference inside otherwise valid continuous windows.

The module deliberately has no torch, image-decoder, detector, or tracker
dependency.  It can therefore be tested and run before any GPU stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Pattern, Sequence

import numpy as np

SCHEMA = "farm.tracker-subset-plan.v1"
SPLIT_POLICY = "global-contiguous-physical-timestamp-blocks-v1"


@dataclass(frozen=True)
class ImageIdentity:
    """Identity parsed from a registered COLMAP image name."""

    camera: str
    view_family: str
    physical_timestamp: str
    timestamp_value: Decimal

    @property
    def stream_key(self) -> tuple[str, str]:
        return self.camera, self.view_family


@dataclass(frozen=True)
class TrackerFrame:
    """One registered image and its metric camera-to-world pose."""

    name: str
    colmap_image_id: int
    colmap_camera_id: int
    identity: ImageIdentity
    world_from_camera_m: np.ndarray

    def __post_init__(self) -> None:
        pose = np.asarray(self.world_from_camera_m, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("world_from_camera_m must be a finite 4x4 matrix")

    @property
    def stream_key(self) -> tuple[str, str]:
        return self.identity.stream_key


@dataclass(frozen=True)
class ContinuityRegion:
    """A camera-pure, fold-pure sequence with no rejected adjacent edge."""

    camera: str
    view_family: str
    split: str
    frames: tuple[TrackerFrame, ...]
    lower_quartile_gap: float | None
    maximum_gap: float | None

    @property
    def group_key(self) -> tuple[str, str, str]:
        return self.camera, self.view_family, self.split


def compile_identity_pattern(
    expression: str,
    *,
    camera_group: str = "camera",
    timestamp_group: str = "timestamp",
    view_family_group: str = "family",
) -> Pattern[str]:
    """Compile and validate the named image-identity expression."""

    if not str(expression).strip():
        raise ValueError("identity regex must not be empty")
    try:
        pattern = re.compile(str(expression))
    except re.error as exc:
        raise ValueError(f"invalid identity regex: {exc}") from exc
    required = {str(camera_group), str(timestamp_group), str(view_family_group)}
    missing = sorted(required.difference(pattern.groupindex))
    if missing:
        raise ValueError(
            "identity regex is missing named group(s): " + ", ".join(missing)
        )
    return pattern


def parse_image_identity(
    name: str,
    pattern: Pattern[str],
    *,
    camera_group: str = "camera",
    timestamp_group: str = "timestamp",
    view_family_group: str = "family",
) -> ImageIdentity:
    """Parse an exact physical identity, rejecting partial/empty matches."""

    match = pattern.fullmatch(str(name))
    if match is None:
        raise ValueError(f"image name does not satisfy identity regex: {name!r}")
    values = match.groupdict()
    camera = str(values.get(camera_group) or "").strip()
    view_family = str(values.get(view_family_group) or "").strip()
    timestamp = str(values.get(timestamp_group) or "").strip()
    if not camera or not view_family or not timestamp:
        raise ValueError(f"image identity contains an empty named group: {name!r}")
    try:
        timestamp_value = Decimal(timestamp)
    except InvalidOperation as exc:
        raise ValueError(
            f"physical timestamp must be numeric, got {timestamp!r} in {name!r}"
        ) from exc
    if not timestamp_value.is_finite():
        raise ValueError(
            f"physical timestamp must be finite, got {timestamp!r} in {name!r}"
        )
    return ImageIdentity(
        camera=camera,
        view_family=view_family,
        physical_timestamp=timestamp,
        timestamp_value=timestamp_value,
    )


def metric_world_from_camera(
    camera_from_world: object,
    *,
    meters_per_scene_unit: float,
) -> np.ndarray:
    """Invert a COLMAP 3x4 pose and express its translation in metres."""

    scale = float(meters_per_scene_unit)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("meters_per_scene_unit must be finite and positive")
    pose = np.asarray(camera_from_world, dtype=np.float64)
    if pose.shape != (3, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_from_world must be a finite 3x4 matrix")
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :] = pose
    try:
        world_from_camera = np.linalg.inv(homogeneous)
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera_from_world is singular") from exc
    rotation = world_from_camera[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-4):
        raise ValueError("COLMAP camera rotation is not orthonormal")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("COLMAP camera rotation has an invalid handedness")
    world_from_camera[:3, 3] *= scale
    return world_from_camera


def camera_motion(
    first_world_from_camera_m: object,
    second_world_from_camera_m: object,
) -> tuple[float, float]:
    """Return camera-centre translation in metres and rotation in degrees."""

    first = np.asarray(first_world_from_camera_m, dtype=np.float64)
    second = np.asarray(second_world_from_camera_m, dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("camera poses must have shape 4x4")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("camera poses must be finite")
    translation_m = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation_degrees = float(np.degrees(np.arccos(cosine)))
    return translation_m, rotation_degrees


def _hash_score(value: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def assign_global_timestamp_folds(
    timestamps: Iterable[Decimal],
    *,
    heldout_fraction: float,
    block_size: int,
    split_seed: str,
) -> tuple[dict[Decimal, str], list[dict[str, object]]]:
    """Assign global, contiguous timestamp blocks to train or heldout.

    Per-frame hashing would fragment every video into one- or two-frame runs.
    Instead, the globally ordered physical timeline is chunked into contiguous
    blocks.  Whole blocks are deterministically ranked for heldout assignment,
    so a timestamp can never be train for one camera and heldout for another.
    """

    fraction = float(heldout_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    if int(block_size) < 1:
        raise ValueError("fold block size must be positive")
    ordered = sorted(set(timestamps))
    if not ordered:
        raise ValueError("cannot split an empty physical timestamp set")
    blocks = [
        ordered[index : index + int(block_size)]
        for index in range(0, len(ordered), int(block_size))
    ]
    if len(blocks) == 1:
        heldout_indices: set[int] = set()
    else:
        heldout_count = int(round(len(blocks) * fraction))
        heldout_count = min(len(blocks) - 1, max(1, heldout_count))
        ranked = sorted(
            range(len(blocks)),
            key=lambda index: (
                _hash_score(
                    f"{index}:{blocks[index][0]}:{blocks[index][-1]}",
                    str(split_seed),
                ),
                index,
            ),
        )
        heldout_indices = set(ranked[:heldout_count])
    folds: dict[Decimal, str] = {}
    provenance: list[dict[str, object]] = []
    for index, block in enumerate(blocks):
        split = "heldout" if index in heldout_indices else "train"
        for timestamp in block:
            folds[timestamp] = split
        provenance.append(
            {
                "block_index": int(index),
                "split": split,
                "timestamp_count": len(block),
                "first_timestamp": str(block[0]),
                "last_timestamp": str(block[-1]),
                "assignment_score": str(
                    _hash_score(f"{index}:{block[0]}:{block[-1]}", str(split_seed))
                ),
            }
        )
    return folds, provenance


def _timestamp_gap(first: TrackerFrame, second: TrackerFrame) -> float:
    return float(second.identity.timestamp_value - first.identity.timestamp_value)


def build_continuity_regions(
    frames: Sequence[TrackerFrame],
    timestamp_folds: Mapping[Decimal, str],
    *,
    gap_multiplier: float = 2.5,
    maximum_translation_m: float = 0.50,
    maximum_rotation_degrees: float = 21.0,
) -> tuple[list[ContinuityRegion], dict[str, object]]:
    """Partition exact streams at time, fold, translation, or rotation breaks."""

    if not math.isfinite(float(gap_multiplier)) or float(gap_multiplier) < 1.0:
        raise ValueError("gap_multiplier must be finite and at least one")
    if (
        not math.isfinite(float(maximum_translation_m))
        or float(maximum_translation_m) <= 0.0
    ):
        raise ValueError("maximum_translation_m must be finite and positive")
    if (
        not math.isfinite(float(maximum_rotation_degrees))
        or not 0.0 < float(maximum_rotation_degrees) <= 180.0
    ):
        raise ValueError("maximum_rotation_degrees must be in (0, 180]")
    grouped: dict[tuple[str, str], list[TrackerFrame]] = defaultdict(list)
    for frame in frames:
        if frame.identity.timestamp_value not in timestamp_folds:
            raise ValueError(
                f"timestamp has no global fold: {frame.identity.physical_timestamp!r}"
            )
        grouped[frame.stream_key].append(frame)
    regions: list[ContinuityRegion] = []
    stream_rows: list[dict[str, object]] = []
    break_counts: Counter[str] = Counter()
    for stream_key in sorted(grouped):
        ordered = sorted(
            grouped[stream_key],
            key=lambda row: (row.identity.timestamp_value, row.name),
        )
        seen: set[Decimal] = set()
        for row in ordered:
            timestamp = row.identity.timestamp_value
            if timestamp in seen:
                raise ValueError(
                    "duplicate physical timestamp inside exact stream "
                    f"{stream_key}: {row.identity.physical_timestamp}"
                )
            seen.add(timestamp)
        positive_gaps = [
            _timestamp_gap(left, right)
            for left, right in zip(ordered, ordered[1:])
            if right.identity.timestamp_value > left.identity.timestamp_value
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
        current: list[TrackerFrame] = []
        for row in ordered:
            split = timestamp_folds[row.identity.timestamp_value]
            reasons: list[str] = []
            if current:
                previous = current[-1]
                previous_split = timestamp_folds[previous.identity.timestamp_value]
                gap = _timestamp_gap(previous, row)
                translation_m, rotation_degrees = camera_motion(
                    previous.world_from_camera_m, row.world_from_camera_m
                )
                if split != previous_split:
                    reasons.append("global_fold_boundary")
                if maximum_gap is not None and gap > maximum_gap:
                    reasons.append("timestamp_gap")
                if translation_m > float(maximum_translation_m):
                    reasons.append("camera_translation")
                if rotation_degrees > float(maximum_rotation_degrees):
                    reasons.append("camera_rotation")
            if current and reasons:
                for reason in reasons:
                    break_counts[reason] += 1
                previous_split = timestamp_folds[current[-1].identity.timestamp_value]
                regions.append(
                    ContinuityRegion(
                        camera=stream_key[0],
                        view_family=stream_key[1],
                        split=previous_split,
                        frames=tuple(current),
                        lower_quartile_gap=lower_quartile,
                        maximum_gap=maximum_gap,
                    )
                )
                current = []
            current.append(row)
        if current:
            regions.append(
                ContinuityRegion(
                    camera=stream_key[0],
                    view_family=stream_key[1],
                    split=timestamp_folds[current[-1].identity.timestamp_value],
                    frames=tuple(current),
                    lower_quartile_gap=lower_quartile,
                    maximum_gap=maximum_gap,
                )
            )
        stream_rows.append(
            {
                "camera": stream_key[0],
                "view_family": stream_key[1],
                "registered_frames": len(ordered),
                "lower_quartile_timestamp_gap": lower_quartile,
                "maximum_allowed_timestamp_gap": maximum_gap,
            }
        )
    return regions, {
        "exact_stream_count": len(grouped),
        "raw_region_count": len(regions),
        "break_counts": dict(sorted(break_counts.items())),
        "streams": stream_rows,
    }


def _region_anchor_count(region: ContinuityRegion, anchor_names: set[str]) -> int:
    return sum(frame.name in anchor_names for frame in region.frames)


def _best_contiguous_slice(
    region: ContinuityRegion,
    length: int,
    anchor_names: set[str],
) -> tuple[ContinuityRegion, list[ContinuityRegion]]:
    """Consume one anchor-preferred slice and retain only valid leftovers."""

    requested = int(length)
    if requested < 1 or requested > len(region.frames):
        raise ValueError("invalid contiguous slice length")
    best_start = 0
    best_score: tuple[int, int] | None = None
    for start in range(0, len(region.frames) - requested + 1):
        window = region.frames[start : start + requested]
        score = (sum(frame.name in anchor_names for frame in window), -start)
        if best_score is None or score > best_score:
            best_start, best_score = start, score
    selected = replace(
        region,
        frames=region.frames[best_start : best_start + requested],
    )
    leftovers: list[ContinuityRegion] = []
    before = region.frames[:best_start]
    after = region.frames[best_start + requested :]
    if before:
        leftovers.append(replace(region, frames=before))
    if after:
        leftovers.append(replace(region, frames=after))
    return selected, leftovers


def _limited_group_keys(
    pools: Mapping[tuple[str, str, str], Sequence[ContinuityRegion]],
    *,
    maximum_groups: int,
    anchor_names: set[str],
) -> list[tuple[str, str, str]]:
    """Keep stream coverage before a second fold from the same stream."""

    keys = sorted(pools)
    if len(keys) <= int(maximum_groups):
        return keys
    by_stream: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in keys:
        by_stream[key[:2]].append(key)
    chosen: list[tuple[str, str, str]] = []
    split_counts: Counter[str] = Counter()
    for round_index in range(2):
        for stream in sorted(by_stream):
            options = [key for key in by_stream[stream] if key not in chosen]
            if not options or len(chosen) >= int(maximum_groups):
                continue
            options.sort(
                key=lambda key: (
                    split_counts[key[2]],
                    -sum(
                        _region_anchor_count(region, anchor_names)
                        for region in pools[key]
                    ),
                    key[2],
                )
            )
            chosen.append(options[0])
            split_counts[options[0][2]] += 1
        if len(chosen) >= int(maximum_groups):
            break
    return sorted(chosen)


def select_balanced_episodes(
    regions: Sequence[ContinuityRegion],
    *,
    anchor_names: Iterable[str] = (),
    target_frames: int = 200,
    minimum_total_frames: int = 160,
    maximum_total_frames: int = 240,
    minimum_episode_frames: int = 8,
    maximum_episode_frames: int = 32,
) -> tuple[list[ContinuityRegion], dict[str, object]]:
    """Select bounded contiguous windows with least-covered-group scheduling."""

    target = int(target_frames)
    minimum_total = int(minimum_total_frames)
    maximum_total = int(maximum_total_frames)
    minimum_episode = int(minimum_episode_frames)
    maximum_episode = int(maximum_episode_frames)
    if minimum_episode < 2 or maximum_episode < minimum_episode:
        raise ValueError("episode bounds must satisfy 2 <= minimum <= maximum")
    if minimum_total < 1 or maximum_total < minimum_total:
        raise ValueError("total bounds must satisfy 1 <= minimum <= maximum")
    if not minimum_total <= target <= maximum_total:
        raise ValueError("target_frames must be inside total frame bounds")
    anchors = {str(name) for name in anchor_names if str(name)}
    pools: dict[tuple[str, str, str], list[ContinuityRegion]] = defaultdict(list)
    discarded_short = 0
    for region in regions:
        if len(region.frames) < minimum_episode:
            discarded_short += len(region.frames)
            continue
        pools[region.group_key].append(region)
    if not pools:
        raise ValueError("no continuous region satisfies minimum episode length")
    for key in pools:
        pools[key].sort(
            key=lambda region: (
                -_region_anchor_count(region, anchors),
                -len(region.frames),
                region.frames[0].identity.timestamp_value,
            )
        )
    maximum_groups = max(1, maximum_total // minimum_episode)
    active_keys = _limited_group_keys(
        pools, maximum_groups=maximum_groups, anchor_names=anchors
    )
    pools = {key: list(pools[key]) for key in active_keys}
    selected: list[ContinuityRegion] = []
    selected_by_group: Counter[tuple[str, str, str]] = Counter()
    total = 0
    while total < target:
        available = [
            key
            for key, candidates in pools.items()
            if any(len(region.frames) >= minimum_episode for region in candidates)
        ]
        if not available:
            break
        available.sort(
            key=lambda key: (
                selected_by_group[key],
                sum(1 for row in selected if row.split == key[2]),
                key,
            )
        )
        progressed = False
        for group_index, key in enumerate(available):
            if total >= target:
                break
            remaining_groups = max(1, len(available) - group_index)
            desired = int(round((target - total) / remaining_groups))
            desired = max(minimum_episode, min(maximum_episode, desired))
            room = maximum_total - total
            if room < minimum_episode:
                break
            candidates = [
                (index, region)
                for index, region in enumerate(pools[key])
                if len(region.frames) >= minimum_episode
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    -int(len(item[1].frames) >= desired),
                    -_region_anchor_count(item[1], anchors),
                    -len(item[1].frames),
                    item[1].frames[0].identity.timestamp_value,
                )
            )
            region_index, region = candidates[0]
            remaining_to_target = target - total
            length = min(desired, maximum_episode, len(region.frames), room)
            if 0 < remaining_to_target < minimum_episode:
                length = min(minimum_episode, maximum_episode, len(region.frames), room)
            else:
                length = min(length, remaining_to_target)
            if length < minimum_episode:
                continue
            window, leftovers = _best_contiguous_slice(region, length, anchors)
            pools[key].pop(region_index)
            pools[key].extend(
                leftover
                for leftover in leftovers
                if len(leftover.frames) >= minimum_episode
            )
            selected.append(window)
            selected_by_group[key] += len(window.frames)
            total += len(window.frames)
            progressed = True
        if not progressed:
            break
    if total < minimum_total:
        available_frames = sum(
            len(region.frames)
            for region in regions
            if len(region.frames) >= minimum_episode
        )
        raise ValueError(
            "continuous tracker coverage is insufficient: "
            f"selected {total}, required {minimum_total}, eligible {available_frames}"
        )
    if total > maximum_total:
        raise AssertionError("balanced selector exceeded maximum_total_frames")
    selected.sort(
        key=lambda region: (
            region.camera,
            region.view_family,
            region.split,
            region.frames[0].identity.timestamp_value,
        )
    )
    by_stream: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    for region in selected:
        stream = f"{region.camera}::{region.view_family}"
        group = f"{stream}::{region.split}"
        by_stream[stream] += len(region.frames)
        by_group[group] += len(region.frames)
        by_split[region.split] += len(region.frames)
    return selected, {
        "target_frames": target,
        "minimum_total_frames": minimum_total,
        "maximum_total_frames": maximum_total,
        "selected_frames": total,
        "selected_episodes": len(selected),
        "eligible_exact_stream_split_groups": len(pools),
        "selected_exact_streams": len(by_stream),
        "discarded_short_region_frames": discarded_short,
        "frames_by_split": dict(sorted(by_split.items())),
        "frames_by_exact_stream": dict(sorted(by_stream.items())),
        "frames_by_exact_stream_split": dict(sorted(by_group.items())),
        "balance_policy": "least-selected exact camera x view_family x split first",
        "anchor_policy": "soft contiguous-window preference only",
    }


def load_anchor_names(path: Path | None) -> tuple[set[str], dict[str, object]]:
    """Read V11/V10-style selected names without interpreting object geometry."""

    if path is None:
        return set(), {"provided": False, "names": 0}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"anchor plan is not valid JSON: {source}") from exc
    names: set[str] = set()

    def visit(value: object, key: str = "", in_view_collection: bool = False) -> None:
        collection = (
            in_view_collection
            or key.endswith("_views")
            or key
            in {
                "views",
                "anchors",
            }
        )
        if isinstance(value, Mapping):
            if collection:
                for candidate_key in ("name", "source_image", "image_name"):
                    candidate = value.get(candidate_key)
                    if isinstance(candidate, str) and candidate:
                        names.add(candidate)
            for child_key, child in value.items():
                if (
                    child_key
                    in {
                        "selected_names",
                        "anchor_names",
                        "train_names",
                        "heldout_names",
                    }
                    and isinstance(child, Sequence)
                    and not isinstance(child, (str, bytes))
                ):
                    names.update(
                        str(item) for item in child if isinstance(item, str) and item
                    )
                visit(child, str(child_key), collection)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                if collection and isinstance(child, str) and child:
                    names.add(child)
                visit(child, key, collection)

    visit(payload)
    return names, {
        "provided": True,
        "path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "names": len(names),
        "schema": payload.get("schema") if isinstance(payload, Mapping) else None,
    }


def episode_edge_metrics(frames: Sequence[TrackerFrame]) -> dict[str, float | None]:
    """Summarise adjacent-edge continuity for provenance and validation."""

    gaps: list[float] = []
    translations: list[float] = []
    rotations: list[float] = []
    for left, right in zip(frames, frames[1:]):
        gaps.append(_timestamp_gap(left, right))
        translation, rotation = camera_motion(
            left.world_from_camera_m, right.world_from_camera_m
        )
        translations.append(translation)
        rotations.append(rotation)
    return {
        "maximum_observed_timestamp_gap": max(gaps) if gaps else None,
        "maximum_observed_translation_m": max(translations) if translations else None,
        "maximum_observed_rotation_degrees": max(rotations) if rotations else None,
    }


def episodes_to_manifest_rows(
    episodes: Sequence[ContinuityRegion], anchor_names: Iterable[str]
) -> list[dict[str, object]]:
    """Serialize selected episodes with ordered poses and exact source names."""

    anchors = set(anchor_names)
    rows: list[dict[str, object]] = []
    counters: Counter[tuple[str, str, str]] = Counter()
    for episode in episodes:
        key = episode.group_key
        ordinal = counters[key]
        counters[key] += 1
        safe_camera = re.sub(r"[^A-Za-z0-9_.-]+", "-", episode.camera).strip("-")
        safe_family = re.sub(r"[^A-Za-z0-9_.-]+", "-", episode.view_family).strip("-")
        episode_id = f"{safe_camera}__{safe_family}__{episode.split}__{ordinal:03d}"
        edge_metrics = episode_edge_metrics(episode.frames)
        rows.append(
            {
                "episode_id": episode_id,
                "camera": episode.camera,
                "view_family": episode.view_family,
                "split": episode.split,
                "frame_count": len(episode.frames),
                "anchor_frame_count": sum(
                    frame.name in anchors for frame in episode.frames
                ),
                "first_physical_timestamp": episode.frames[
                    0
                ].identity.physical_timestamp,
                "last_physical_timestamp": episode.frames[
                    -1
                ].identity.physical_timestamp,
                "continuity": {
                    "lower_quartile_timestamp_gap": episode.lower_quartile_gap,
                    "maximum_allowed_timestamp_gap": episode.maximum_gap,
                    **edge_metrics,
                },
                "frames": [
                    {
                        "episode_index": index,
                        "name": frame.name,
                        "colmap_image_id": frame.colmap_image_id,
                        "colmap_camera_id": frame.colmap_camera_id,
                        "physical_timestamp": frame.identity.physical_timestamp,
                        "is_anchor_preference": frame.name in anchors,
                        "T_world_camera_m": np.asarray(
                            frame.world_from_camera_m, dtype=np.float64
                        ).tolist(),
                    }
                    for index, frame in enumerate(episode.frames)
                ],
            }
        )
    return rows


def validate_episode_rows(
    episodes: Sequence[Mapping[str, object]],
    *,
    minimum_episode_frames: int,
    maximum_episode_frames: int,
    maximum_translation_m: float,
    maximum_rotation_degrees: float,
) -> None:
    """Fail closed on leakage, duplicates, mixed streams, or broken windows."""

    names: set[str] = set()
    train_timestamps: set[str] = set()
    heldout_timestamps: set[str] = set()
    for episode in episodes:
        frames = episode.get("frames")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise ValueError("episode frames must be a sequence")
        if (
            not int(minimum_episode_frames)
            <= len(frames)
            <= int(maximum_episode_frames)
        ):
            raise ValueError("episode violates frame-count bounds")
        split = str(episode.get("split") or "")
        if split not in {"train", "heldout"}:
            raise ValueError("episode has an invalid split")
        previous_timestamp: Decimal | None = None
        previous_pose: np.ndarray | None = None
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ValueError("episode frame must be an object")
            name = str(frame.get("name") or "")
            if not name or name in names:
                raise ValueError(f"duplicate or empty selected image name: {name!r}")
            names.add(name)
            timestamp_text = str(frame.get("physical_timestamp") or "")
            timestamp = Decimal(timestamp_text)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("episode timestamps are not strictly increasing")
            pose = np.asarray(frame.get("T_world_camera_m"), dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("episode contains an invalid camera pose")
            if previous_pose is not None:
                translation, rotation = camera_motion(previous_pose, pose)
                if translation > float(maximum_translation_m) + 1.0e-9:
                    raise ValueError("episode crosses the camera translation gate")
                if rotation > float(maximum_rotation_degrees) + 1.0e-9:
                    raise ValueError("episode crosses the camera rotation gate")
            (train_timestamps if split == "train" else heldout_timestamps).add(
                str(timestamp.normalize())
            )
            previous_timestamp, previous_pose = timestamp, pose
    overlap = train_timestamps.intersection(heldout_timestamps)
    if overlap:
        raise ValueError(
            "physical timestamp leakage between train and heldout: "
            + ", ".join(sorted(overlap)[:5])
        )
