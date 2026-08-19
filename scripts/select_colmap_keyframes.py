#!/usr/bin/env python3
"""Select a compact, connected set of physical observations from COLMAP.

Unlike the historical factory selector, this implementation does not assume a
specific camera count, filename prefix, virtual-view family, or rig baseline.
Images are grouped into physical observations with a configurable regular
expression containing configurable named sensor, timestamp and virtual-view
groups.  A physical observation, not an individual virtual view, is the unit
of trajectory sampling.  COLMAP coordinates are converted to metres at the
input boundary so all translation thresholds have unambiguous metric units.

The output is deliberately simple and stable:

* ``selected_names.txt``: exact COLMAP image names to pass downstream;
* ``selection_manifest.json``: grouping, geometry, connectivity and all
  resolved parameters;
* ``visuals/01_selection_dashboard_4k.jpg`` and an optional contact sheet.

The selector is category-agnostic and never reads segmentation results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_NAME_REGEX = (
    r"(?:^|/)(?P<sensor>.+?)[_-](?P<timestamp>\d+)"
    r"(?:[_-](?P<family>[^/.]+))?\.[^.]+$"
)


@dataclass(frozen=True)
class RegisteredView:
    image_id: int
    name: str
    camera_id: int
    sensor: str
    timestamp: str
    family: str
    center: np.ndarray
    rotation: np.ndarray
    tracks: frozenset[int]


@dataclass(frozen=True)
class Observation:
    timestamp: str
    views: tuple[RegisteredView, ...]
    center: np.ndarray
    rotation: np.ndarray
    tracks: frozenset[int]

    @property
    def min_support(self) -> int:
        return min((len(view.tracks) for view in self.views), default=0)


def _csv(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometry-aware, rig-aware keyframe selection for a COLMAP model."
    )
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", default="scene")
    parser.add_argument("--name-regex", default=DEFAULT_NAME_REGEX)
    parser.add_argument("--sensor-group", default="sensor")
    parser.add_argument("--timestamp-group", default="timestamp")
    parser.add_argument("--family-group", default="family")
    parser.add_argument(
        "--meters-per-scene-unit",
        type=float,
        default=1.0,
        help="Metric scale from preflight: metres represented by one COLMAP/3DGS unit.",
    )
    parser.add_argument(
        "--sensor",
        action="append",
        default=[],
        help="Required physical sensor name; repeat or use comma-separated values. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--anchor-family",
        action="append",
        default=[],
        help="Virtual family used for trajectory/connectivity; defaults to center when present.",
    )
    parser.add_argument(
        "--output-family",
        action="append",
        default=[],
        help="Families emitted for selected timestamps. '*' emits every available family.",
    )
    parser.add_argument("--fallback-family", default="default")
    parser.add_argument("--min-sensor-coverage", type=float, default=0.95)
    parser.add_argument("--target-observations", type=int, default=160)
    parser.add_argument("--max-observations", type=int, default=180)
    parser.add_argument("--max-output-views", type=int, default=500)
    parser.add_argument("--rotation-weight-m-per-rad", type=float, default=0.50)
    parser.add_argument("--max-motion-gap", type=float, default=1.35)
    parser.add_argument("--max-translation-gap-m", type=float, default=1.30)
    parser.add_argument("--max-rotation-gap-deg", type=float, default=50.0)
    parser.add_argument("--min-shared-tracks", type=int, default=20)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.015)
    parser.add_argument(
        "--expected-baseline-m",
        type=float,
        default=0.0,
        help="Optional two-sensor metric baseline; <=0 disables the check.",
    )
    parser.add_argument("--baseline-tolerance-m", type=float, default=0.005)
    parser.add_argument(
        "--baseline-sensors",
        nargs=2,
        metavar=("SENSOR_A", "SENSOR_B"),
        help="Exact distinct sensor pair used by --expected-baseline-m; required for rigs with more than two selected sensors.",
    )
    parser.add_argument("--contact-sheet-count", type=int, default=48)
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args(argv)


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def validate_meters_per_scene_unit(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("--meters-per-scene-unit must be finite and strictly positive")
    return scale


def resolve_baseline_sensors(
    selected_sensors: Sequence[str],
    requested: Sequence[str] | None,
    *,
    required: bool,
) -> tuple[str, str] | None:
    """Resolve an exact baseline pair without silently choosing from an N-rig."""

    sensors = list(selected_sensors)
    if requested is not None:
        pair = tuple(str(value).strip() for value in requested)
        if len(pair) != 2 or not pair[0] or not pair[1]:
            raise ValueError("--baseline-sensors requires exactly two non-empty sensor names")
        if pair[0] == pair[1]:
            raise ValueError("--baseline-sensors values must be distinct")
        missing = [value for value in pair if value not in sensors]
        if missing:
            raise ValueError(f"--baseline-sensors are not selected/present: {missing}")
        return pair
    if not required:
        return None
    if len(sensors) != 2:
        raise ValueError(
            "--expected-baseline-m requires --baseline-sensors when selected sensor count is not two"
        )
    return sensors[0], sensors[1]


def baseline_pair_key(pair: tuple[str, str], sensor_order: Sequence[str]) -> str:
    first, second = sorted(pair, key=list(sensor_order).index)
    return f"{first}::{second}"


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = second @ first.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def point_ids(image: pycolmap.Image) -> frozenset[int]:
    return frozenset(
        int(point.point3D_id) for point in image.points2D if point.has_point3D()
    )


def parse_image_identity(
    name: str,
    pattern: re.Pattern[str],
    camera_id: int,
    fallback_family: str,
    sensor_group: str = "sensor",
    timestamp_group: str = "timestamp",
    family_group: str = "family",
) -> tuple[str, str, str]:
    match = pattern.search(name.replace("\\", "/"))
    if match is None:
        # A valid fallback for ordinary monocular COLMAP datasets: every image
        # is one observation and the camera id is its stable sensor identity.
        return f"camera_{camera_id}", Path(name).stem, fallback_family
    groups = match.groupdict()
    if timestamp_group not in groups:
        raise ValueError(
            f"Configured timestamp group {timestamp_group!r} is absent from --name-regex"
        )
    sensor = (groups.get(sensor_group) or f"camera_{camera_id}").strip()
    timestamp = (groups.get(timestamp_group) or "").strip()
    family = (groups.get(family_group) or fallback_family).strip()
    if not sensor or not timestamp or not family:
        raise ValueError(f"Empty grouping field parsed from COLMAP image name {name!r}")
    return sensor, timestamp, family


def load_registered_views(
    reconstruction: pycolmap.Reconstruction,
    name_regex: str,
    fallback_family: str,
    meters_per_scene_unit: float = 1.0,
    sensor_group: str = "sensor",
    timestamp_group: str = "timestamp",
    family_group: str = "family",
) -> list[RegisteredView]:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    try:
        pattern = re.compile(name_regex)
    except re.error as error:
        raise ValueError(f"Invalid --name-regex: {error}") from error
    views: list[RegisteredView] = []
    for image_id, image in reconstruction.images.items():
        sensor, timestamp, family = parse_image_identity(
            image.name,
            pattern,
            int(image.camera_id),
            fallback_family,
            sensor_group,
            timestamp_group,
            family_group,
        )
        matrix = np.asarray(image.cam_from_world.matrix(), dtype=np.float64)
        views.append(
            RegisteredView(
                image_id=int(image_id),
                name=image.name,
                camera_id=int(image.camera_id),
                sensor=sensor,
                timestamp=timestamp,
                family=family,
                center=np.asarray(image.projection_center(), dtype=np.float64) * scale,
                rotation=matrix[:, :3],
                tracks=point_ids(image),
            )
        )
    if len(views) < 2:
        raise ValueError("COLMAP contains fewer than two registered images")
    return sorted(
        views,
        key=lambda view: (natural_key(view.timestamp), view.sensor, view.family, view.name),
    )


def resolve_grouping(
    views: Sequence[RegisteredView],
    requested_sensors: Sequence[str],
    requested_anchor_families: Sequence[str],
    min_sensor_coverage: float,
) -> tuple[list[str], list[str], dict[str, object]]:
    timestamps = sorted({view.timestamp for view in views}, key=natural_key)
    families = sorted({view.family for view in views})
    sensors = sorted({view.sensor for view in views})
    if requested_anchor_families:
        anchor_families = list(dict.fromkeys(requested_anchor_families))
    elif "center" in families:
        anchor_families = ["center"]
    else:
        counts = {family: sum(view.family == family for view in views) for family in families}
        anchor_families = [max(counts, key=counts.get)]

    if requested_sensors:
        selected_sensors = list(dict.fromkeys(requested_sensors))
    else:
        denominator = max(len(timestamps), 1)
        coverage = {
            sensor: len(
                {
                    view.timestamp
                    for view in views
                    if view.sensor == sensor and view.family in anchor_families
                }
            )
            / denominator
            for sensor in sensors
        }
        selected_sensors = [
            sensor for sensor in sensors if coverage[sensor] >= min_sensor_coverage
        ]
        if not selected_sensors:
            selected_sensors = [max(coverage, key=coverage.get)]

    missing_sensors = sorted(set(selected_sensors) - set(sensors))
    missing_families = sorted(set(anchor_families) - set(families))
    if missing_sensors or missing_families:
        raise ValueError(
            f"Unknown grouping values: sensors={missing_sensors}, families={missing_families}"
        )
    return selected_sensors, anchor_families, {
        "detected_sensors": sensors,
        "detected_families": families,
        "timestamp_count": len(timestamps),
    }


def build_observations(
    views: Sequence[RegisteredView],
    sensors: Sequence[str],
    anchor_families: Sequence[str],
) -> tuple[list[Observation], dict[str, list[RegisteredView]]]:
    by_timestamp: dict[str, list[RegisteredView]] = {}
    for view in views:
        by_timestamp.setdefault(view.timestamp, []).append(view)
    observations: list[Observation] = []
    selected_sensor_set = set(sensors)
    anchor_set = set(anchor_families)
    for timestamp in sorted(by_timestamp, key=natural_key):
        candidates = [
            view
            for view in by_timestamp[timestamp]
            if view.sensor in selected_sensor_set and view.family in anchor_set
        ]
        present_sensors = {view.sensor for view in candidates}
        if not set(sensors).issubset(present_sensors):
            continue
        candidates.sort(key=lambda view: (sensors.index(view.sensor), view.family, view.name))
        # Use one stable physical sensor for orientation at every timestamp.
        # Per-frame support ranking can alternate between opposed lenses and
        # invent ~180-degree motion spikes that the rig never made.
        representative = candidates[0]
        observations.append(
            Observation(
                timestamp=timestamp,
                views=tuple(candidates),
                center=np.mean(np.stack([view.center for view in candidates]), axis=0),
                rotation=representative.rotation,
                tracks=frozenset().union(*(view.tracks for view in candidates)),
            )
        )
    if len(observations) < 2:
        raise ValueError(
            "Fewer than two complete physical observations remain after grouping; "
            f"required sensors={list(sensors)}, anchor families={list(anchor_families)}"
        )
    return observations, by_timestamp


def motion_arrays(
    observations: Sequence[Observation], rotation_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    translations = np.zeros(len(observations), dtype=np.float64)
    rotations = np.zeros(len(observations), dtype=np.float64)
    for index in range(1, len(observations)):
        translations[index] = np.linalg.norm(
            observations[index].center - observations[index - 1].center
        )
        rotations[index] = rotation_angle_deg(
            observations[index - 1].rotation, observations[index].rotation
        )
    step_motion = translations + rotation_weight * np.deg2rad(rotations)
    return translations, rotations, step_motion, np.cumsum(step_motion)


def robust_quality(observations: Sequence[Observation]) -> np.ndarray:
    raw = np.log1p(np.asarray([observation.min_support for observation in observations], dtype=float))
    low, high = np.quantile(raw, [0.05, 0.95])
    return np.clip((raw - low) / max(float(high - low), 1e-8), 0.0, 1.0)


def initial_selection(
    cumulative_motion: np.ndarray, quality: np.ndarray, target_count: int
) -> set[int]:
    count = len(cumulative_motion)
    target_count = min(max(target_count, 2), count)
    targets = np.linspace(cumulative_motion[0], cumulative_motion[-1], target_count)
    boundaries = (targets[:-1] + targets[1:]) * 0.5
    selected: set[int] = set()
    for target_index, target in enumerate(targets):
        low = -np.inf if target_index == 0 else boundaries[target_index - 1]
        high = np.inf if target_index == len(targets) - 1 else boundaries[target_index]
        candidates = np.flatnonzero(
            (cumulative_motion >= low) & (cumulative_motion <= high)
        )
        if not len(candidates):
            candidates = np.asarray([int(np.argmin(np.abs(cumulative_motion - target)))])
        finite_span = high - low if np.isfinite(low) and np.isfinite(high) else 0.0
        half_width = max(0.5 * finite_span, cumulative_motion[-1] / max(target_count - 1, 1), 1e-6)
        score = np.abs(cumulative_motion[candidates] - target) / half_width
        score += 0.28 * (1.0 - quality[candidates])
        selected.add(int(candidates[int(np.argmin(score))]))
    selected.update((0, count - 1))
    return selected


def edge_metrics(
    left: int,
    right: int,
    observations: Sequence[Observation],
    cumulative_motion: np.ndarray,
) -> dict[str, float | int | str]:
    first, second = observations[left], observations[right]
    shared = len(first.tracks & second.tracks)
    denominator = max(min(len(first.tracks), len(second.tracks)), 1)
    return {
        "left_index": left,
        "right_index": right,
        "left_timestamp": first.timestamp,
        "right_timestamp": second.timestamp,
        "observation_gap": right - left,
        "translation_m": float(np.linalg.norm(second.center - first.center)),
        "rotation_deg": rotation_angle_deg(first.rotation, second.rotation),
        "motion_gap": float(cumulative_motion[right] - cumulative_motion[left]),
        "shared_tracks": shared,
        "overlap_ratio": float(shared / denominator),
    }


def violation_reasons(edge: dict[str, float | int | str], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if float(edge["motion_gap"]) > args.max_motion_gap:
        reasons.append("motion")
    if float(edge["translation_m"]) > args.max_translation_gap_m:
        reasons.append("translation")
    if float(edge["rotation_deg"]) > args.max_rotation_gap_deg:
        reasons.append("rotation")
    if int(edge["shared_tracks"]) < args.min_shared_tracks:
        reasons.append("shared_tracks")
    if float(edge["overlap_ratio"]) < args.min_overlap_ratio:
        reasons.append("overlap")
    return reasons


def choose_bridge(
    left: int,
    right: int,
    observations: Sequence[Observation],
    cumulative_motion: np.ndarray,
    quality: np.ndarray,
) -> int:
    candidates = np.arange(left + 1, right, dtype=int)
    if not len(candidates):
        return -1
    midpoint = 0.5 * (cumulative_motion[left] + cumulative_motion[right])
    span = max(float(cumulative_motion[right] - cumulative_motion[left]), 1e-6)
    midpoint_penalty = np.abs(cumulative_motion[candidates] - midpoint) / span
    connectivity = np.zeros(len(candidates), dtype=np.float64)
    for output_index, candidate in enumerate(candidates):
        tracks = observations[int(candidate)].tracks
        left_denominator = max(min(len(tracks), len(observations[left].tracks)), 1)
        right_denominator = max(min(len(tracks), len(observations[right].tracks)), 1)
        left_overlap = len(tracks & observations[left].tracks) / left_denominator
        right_overlap = len(tracks & observations[right].tracks) / right_denominator
        connectivity[output_index] = min(left_overlap, right_overlap)
    score = midpoint_penalty + 0.20 * (1.0 - quality[candidates]) - 0.35 * connectivity
    return int(candidates[int(np.argmin(score))])


def add_bridges(
    selected: set[int],
    observations: Sequence[Observation],
    cumulative_motion: np.ndarray,
    quality: np.ndarray,
    args: argparse.Namespace,
) -> tuple[set[int], dict[int, list[str]]]:
    reasons_by_index: dict[int, list[str]] = {}
    while len(selected) < args.max_observations:
        additions: list[tuple[float, int, list[str]]] = []
        ordered = sorted(selected)
        for left, right in zip(ordered, ordered[1:]):
            edge = edge_metrics(left, right, observations, cumulative_motion)
            reasons = violation_reasons(edge, args)
            if not reasons or right - left <= 1:
                continue
            bridge = choose_bridge(left, right, observations, cumulative_motion, quality)
            if bridge < 0 or bridge in selected:
                continue
            severity = max(
                float(edge["motion_gap"]) / max(args.max_motion_gap, 1e-9),
                float(edge["translation_m"]) / max(args.max_translation_gap_m, 1e-9),
                float(edge["rotation_deg"]) / max(args.max_rotation_gap_deg, 1e-9),
                args.min_shared_tracks / max(float(edge["shared_tracks"]), 1.0),
                args.min_overlap_ratio / max(float(edge["overlap_ratio"]), 1e-9),
            )
            additions.append((severity, bridge, reasons))
        if not additions:
            break
        additions.sort(reverse=True)
        for _, bridge, reasons in additions[: args.max_observations - len(selected)]:
            selected.add(bridge)
            reasons_by_index.setdefault(bridge, []).extend(reasons)
    return selected, reasons_by_index


def resolve_output_views(
    selected_observations: Sequence[Observation],
    by_timestamp: dict[str, list[RegisteredView]],
    sensors: Sequence[str],
    output_families: Sequence[str],
    anchor_families: Sequence[str],
) -> list[RegisteredView]:
    resolved_families = list(output_families) or list(anchor_families)
    all_families = "*" in resolved_families
    sensor_set = set(sensors)
    family_set = set(resolved_families)
    selected: list[RegisteredView] = []
    for observation in selected_observations:
        candidates = [
            view
            for view in by_timestamp[observation.timestamp]
            if view.sensor in sensor_set and (all_families or view.family in family_set)
        ]
        candidates.sort(key=lambda view: (sensors.index(view.sensor), view.family, view.name))
        selected.extend(candidates)
    return selected


def pairwise_baselines(
    observations: Sequence[Observation],
    sensors: Sequence[str],
    meters_per_scene_unit: float = 1.0,
) -> dict[str, dict[str, object]]:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    values: dict[tuple[str, str], list[float]] = {}
    for observation in observations:
        centers: dict[str, list[np.ndarray]] = {}
        for view in observation.views:
            centers.setdefault(view.sensor, []).append(view.center)
        means = {sensor: np.mean(np.stack(items), axis=0) for sensor, items in centers.items()}
        for first_index, first in enumerate(sensors):
            for second in sensors[first_index + 1 :]:
                if first in means and second in means:
                    values.setdefault((first, second), []).append(
                        float(np.linalg.norm(means[first] - means[second]))
                    )
    result: dict[str, dict[str, object]] = {}
    for pair, pair_values in values.items():
        metric = np.asarray(pair_values, dtype=np.float64)
        raw = metric / scale
        raw_stats = {
            "min": float(raw.min()),
            "p05": float(np.percentile(raw, 5.0)),
            "median": float(np.median(raw)),
            "p95": float(np.percentile(raw, 95.0)),
            "max": float(raw.max()),
        }
        metric_stats = {
            "min": float(metric.min()),
            "p05": float(np.percentile(metric, 5.0)),
            "median": float(np.median(metric)),
            "p95": float(np.percentile(metric, 95.0)),
            "max": float(metric.max()),
        }
        result[f"{pair[0]}::{pair[1]}"] = {
            "count": int(len(metric)),
            # Legacy flat values remain metric. Explicit nested values remove
            # ambiguity when a non-unit scene scale is used.
            "min": metric_stats["min"],
            "median": metric_stats["median"],
            "max": metric_stats["max"],
            "scene_units": raw_stats,
            "metres": metric_stats,
        }
    return result


def resolve_image(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_file():
        return direct
    basename = root / Path(name).name
    if basename.is_file():
        return basename
    raise FileNotFoundError(f"Image not found for COLMAP name {name!r}")


def save_contact_sheet(
    selected_observations: Sequence[Observation],
    by_timestamp: dict[str, list[RegisteredView]],
    image_root: Path,
    output_path: Path,
    count: int,
) -> None:
    positions = np.unique(
        np.linspace(0, len(selected_observations) - 1, min(count, len(selected_observations)))
        .round()
        .astype(int)
    )
    columns = 8
    rows = int(math.ceil(len(positions) / columns))
    cell_width, cell_height = 400, 290
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 80), "#0b1118")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (20, 20),
        f"Selected physical observations: {len(selected_observations)}",
        fill="white",
        font=font,
    )
    for grid_index, position in enumerate(positions):
        observation = selected_observations[int(position)]
        view = max(by_timestamp[observation.timestamp], key=lambda item: len(item.tracks))
        image = Image.open(resolve_image(image_root, view.name)).convert("RGB")
        image = ImageOps.fit(
            image, (cell_width - 16, cell_height - 48), method=Image.Resampling.LANCZOS
        )
        x = (grid_index % columns) * cell_width + 8
        y = (grid_index // columns) * cell_height + 64
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + cell_height - 43), observation.timestamp, fill="white", font=font)
    sheet.save(output_path, quality=92, subsampling=0)


def save_dashboard(
    observations: Sequence[Observation],
    selected_indices: Sequence[int],
    cumulative_motion: np.ndarray,
    edges: Sequence[dict[str, object]],
    output_path: Path,
    scene_id: str,
    args: argparse.Namespace,
    selected_view_count: int,
    baseline_check: dict[str, object],
) -> None:
    centers = np.stack([observation.center for observation in observations])
    selected_centers = centers[selected_indices]
    plane_axes = np.argsort(np.std(centers, axis=0))[-2:]
    edge_motion = np.asarray([float(edge["motion_gap"]) for edge in edges])
    edge_translation = np.asarray([float(edge["translation_m"]) for edge in edges])
    edge_rotation = np.asarray([float(edge["rotation_deg"]) for edge in edges])
    edge_overlap = np.asarray([float(edge["overlap_ratio"]) for edge in edges])
    edge_shared = np.asarray([float(edge["shared_tracks"]) for edge in edges])
    support = np.asarray([observation.min_support for observation in observations])
    source_indices = np.arange(len(observations))
    selected_indices_array = np.asarray(selected_indices, dtype=int)
    selected_edge_indices = np.arange(len(edges))
    axis_names = np.asarray(["X", "Y", "Z"])
    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 3, figsize=(24, 13.5), dpi=160)
    figure.patch.set_facecolor("#0b1118")
    figure.suptitle(
        f"{scene_id} | geometry-aware selection | "
        f"{len(observations)} → {len(selected_indices)} timestamps "
        f"({selected_view_count} views)",
        fontsize=22,
        fontweight="bold",
    )
    axis = axes[0, 0]
    axis.plot(
        centers[:, plane_axes[0]], centers[:, plane_axes[1]],
        color="#52616b", linewidth=1.0, label="all poses",
    )
    points = axis.scatter(
        selected_centers[:, plane_axes[0]],
        selected_centers[:, plane_axes[1]],
        c=np.arange(len(selected_centers)),
        cmap="turbo",
        s=24,
        label="selected",
    )
    axis.scatter(
        selected_centers[0, plane_axes[0]], selected_centers[0, plane_axes[1]],
        marker="*", s=180, color="#3ef3a5", label="start",
    )
    axis.scatter(
        selected_centers[-1, plane_axes[0]], selected_centers[-1, plane_axes[1]],
        marker="X", s=110, color="#ff6b6b", label="end",
    )
    axis.set_title(
        f"{axis_names[plane_axes[0]]}{axis_names[plane_axes[1]]} trajectory coverage"
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(loc="best", fontsize=8)
    figure.colorbar(points, ax=axis, fraction=0.045, label="selected order")
    axis.grid(alpha=0.2)
    axis = axes[0, 1]
    axis.plot(source_indices, cumulative_motion, color="#42c8f5", linewidth=1.7)
    axis.scatter(
        selected_indices_array, cumulative_motion[selected_indices_array],
        color="#ffd166", s=18,
    )
    axis.set_title("Hybrid trajectory arc (translation + rotation)")
    axis.set_xlabel("physical observation index")
    axis.set_ylabel("cumulative motion units")
    axis.grid(alpha=0.2)
    axis = axes[0, 2]
    axis.plot(source_indices, support, color="#8b9dc3", linewidth=1.0)
    axis.scatter(
        selected_indices_array, support[selected_indices_array],
        color="#35d0ba", s=18, label="selected",
    )
    axis.set_yscale("symlog", linthresh=20)
    axis.set_title("COLMAP support of selected observations")
    axis.set_xlabel("physical observation index")
    axis.set_ylabel("registered 3D tracks")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    axis = axes[1, 0]
    axis.plot(selected_edge_indices, edge_motion, color="#ffd166", label="motion gap")
    axis.plot(
        selected_edge_indices, edge_translation,
        color="#42c8f5", alpha=0.75, label="translation, m",
    )
    axis.axhline(args.max_motion_gap, color="#ff6b6b", linestyle="--", label="motion limit")
    second = axis.twinx()
    second.plot(
        selected_edge_indices, edge_rotation,
        color="#c77dff", alpha=0.72, label="rotation",
    )
    second.axhline(args.max_rotation_gap_deg, color="#c77dff", linestyle=":", alpha=0.8)
    axis.set_title("Final consecutive-pair motion")
    axis.set_xlabel("selected edge")
    axis.set_ylabel("motion / translation")
    second.set_ylabel("rotation, degrees")
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(alpha=0.2)
    axis = axes[1, 1]
    axis.plot(selected_edge_indices, edge_overlap * 100.0, color="#35d0ba", label="overlap ratio")
    axis.axhline(
        args.min_overlap_ratio * 100.0,
        color="#ff6b6b", linestyle="--", label="overlap floor",
    )
    second = axis.twinx()
    second.plot(selected_edge_indices, edge_shared, color="#f78c6c", alpha=0.60)
    second.axhline(args.min_shared_tracks, color="#f78c6c", linestyle=":", alpha=0.85)
    axis.set_title("Final COLMAP track connectivity")
    axis.set_xlabel("selected edge")
    axis.set_ylabel("overlap, %")
    second.set_ylabel("shared tracks")
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(alpha=0.2)
    axis = axes[1, 2]
    intervals = np.diff(selected_indices_array)
    bins = np.arange(0.5, max(float(intervals.max()) + 1.5, 3.5))
    axis.hist(intervals, bins=bins, color="#42c8f5", edgecolor="#0b1118")
    bad_edges = sum(bool(edge.get("violations")) for edge in edges)
    baseline = baseline_check.get("observed_median_m")
    baseline_line = f"Rig baseline: {float(baseline):.5f} m" if baseline is not None else "Rig baseline: n/a"
    summary = (
        f"Input timestamps: {len(observations)}\n"
        f"Selected timestamps: {len(selected_indices)}\n"
        f"Selected views: {selected_view_count}\n"
        f"Reduction: {len(observations) / len(selected_indices):.2f}×\n"
        f"Median interval: {np.median(intervals):.1f} frames\n"
        f"Max interval: {int(intervals.max())} frames\n"
        f"Median edge overlap: {np.median(edge_overlap) * 100.0:.1f}%\n"
        f"Min shared tracks: {int(edge_shared.min())}\n"
        f"Residual flagged edges: {bad_edges}\n"
        f"{baseline_line}"
    )
    axis.text(
        0.98, 0.97, summary, transform=axis.transAxes,
        va="top", ha="right", fontsize=10,
        bbox={"facecolor": "#15212d", "edgecolor": "#35d0ba", "alpha": 0.95, "pad": 9},
    )
    axis.set_title("Sampling interval distribution")
    axis.set_xlabel("original-observation interval")
    axis.set_ylabel("count")
    axis.grid(alpha=0.2)
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)


def stable_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Validate scale and grouping before creating even an empty output folder.
    # This makes misconfigured automated runs side-effect free.
    args.meters_per_scene_unit = validate_meters_per_scene_unit(
        args.meters_per_scene_unit
    )
    for option in ("sensor_group", "timestamp_group", "family_group"):
        if not str(getattr(args, option)).strip():
            raise ValueError(f"--{option.replace('_', '-')} cannot be empty")
    if args.expected_baseline_m > 0 and (
        not math.isfinite(args.expected_baseline_m)
        or not math.isfinite(args.baseline_tolerance_m)
        or args.baseline_tolerance_m < 0
    ):
        raise ValueError("Expected baseline and tolerance must be finite; tolerance must be >= 0")
    # Reject malformed explicit pairs before touching the output directory.
    if args.baseline_sensors is not None:
        raw_pair = [str(value).strip() for value in args.baseline_sensors]
        if not all(raw_pair) or raw_pair[0] == raw_pair[1]:
            raise ValueError("--baseline-sensors must contain two distinct non-empty names")
    started = time.perf_counter()
    output = args.output.resolve()
    visuals = output / "visuals"
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_visuals:
        visuals.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(str(args.colmap.resolve()))
    views = load_registered_views(
        reconstruction,
        args.name_regex,
        args.fallback_family,
        args.meters_per_scene_unit,
        args.sensor_group,
        args.timestamp_group,
        args.family_group,
    )
    requested_sensors = _csv(args.sensor)
    requested_anchors = _csv(args.anchor_family)
    requested_outputs = _csv(args.output_family)
    sensors, anchor_families, detected = resolve_grouping(
        views, requested_sensors, requested_anchors, args.min_sensor_coverage
    )
    baseline_sensors = resolve_baseline_sensors(
        sensors,
        args.baseline_sensors,
        required=args.expected_baseline_m > 0,
    )
    observations, by_timestamp = build_observations(views, sensors, anchor_families)

    output_families = requested_outputs or anchor_families
    per_observation_views = max(
        (
            len(
                [
                    view
                    for view in by_timestamp[observation.timestamp]
                    if view.sensor in set(sensors)
                    and ("*" in output_families or view.family in set(output_families))
                ]
            )
            for observation in observations
        ),
        default=1,
    )
    max_by_view_budget = max(2, args.max_output_views // max(per_observation_views, 1))
    args.max_observations = min(args.max_observations, max_by_view_budget, len(observations))
    args.target_observations = min(args.target_observations, args.max_observations)

    translations, rotations, _, cumulative_motion = motion_arrays(
        observations, args.rotation_weight_m_per_rad
    )
    quality = robust_quality(observations)
    selected = initial_selection(cumulative_motion, quality, args.target_observations)
    initial_count = len(selected)
    selected, bridge_reasons = add_bridges(
        selected, observations, cumulative_motion, quality, args
    )
    selected_indices = sorted(selected)
    selected_observations = [observations[index] for index in selected_indices]
    output_views = resolve_output_views(
        selected_observations, by_timestamp, sensors, output_families, anchor_families
    )
    if len(output_views) > args.max_output_views:
        raise RuntimeError(
            f"Resolved {len(output_views)} views, above max-output-views={args.max_output_views}"
        )

    edges: list[dict[str, object]] = []
    for left, right in zip(selected_indices, selected_indices[1:]):
        edge: dict[str, object] = dict(edge_metrics(left, right, observations, cumulative_motion))
        edge["violations"] = violation_reasons(edge, args)
        edges.append(edge)
    baselines = pairwise_baselines(
        observations, sensors, args.meters_per_scene_unit
    )
    baseline_check = {
        "enabled": args.expected_baseline_m > 0,
        "sensors": list(baseline_sensors) if baseline_sensors is not None else None,
        "passed": True,
    }
    if args.expected_baseline_m > 0:
        assert baseline_sensors is not None
        key = baseline_pair_key(baseline_sensors, sensors)
        pair_stats = baselines.get(key)
        metric_stats = pair_stats.get("metres", {}) if pair_stats else {}
        median = metric_stats.get("median")
        minimum = metric_stats.get("min")
        maximum = metric_stats.get("max")
        max_abs_error = (
            max(
                abs(float(minimum) - args.expected_baseline_m),
                abs(float(maximum) - args.expected_baseline_m),
            )
            if minimum is not None and maximum is not None
            else None
        )
        passed = (
            pair_stats is not None
            and int(pair_stats.get("count", 0)) > 0
            and max_abs_error is not None
            and max_abs_error <= args.baseline_tolerance_m
        )
        baseline_check.update(
            {
                "pair": key,
                "expected_m": args.expected_baseline_m,
                "tolerance_m": args.baseline_tolerance_m,
                "observed_median_m": median,
                "max_abs_error_m": max_abs_error,
                "criterion": "every paired timestamp within tolerance",
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"Metric baseline check failed: {baseline_check}")

    names = [view.name for view in output_views]
    (output / "selected_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    manifest = {
        "schema": "farm.colmap-keyframes.v2",
        "status": "ok",
        "scene_id": args.scene_id,
        "metric_scale": {
            "meters_per_scene_unit": args.meters_per_scene_unit,
            "input_translation_units": "COLMAP scene units",
            "selection_translation_units": "metres",
        },
        "inputs": {
            "colmap": str(args.colmap.resolve()),
            "images": str(args.images.resolve()) if args.images else None,
            "registered_images": len(views),
            "registered_points3D": len(reconstruction.points3D),
        },
        "grouping": {
            "name_regex": args.name_regex,
            "sensor_group": args.sensor_group,
            "timestamp_group": args.timestamp_group,
            "family_group": args.family_group,
            "fallback_family": args.fallback_family,
            "sensors": sensors,
            "anchor_families": anchor_families,
            "output_families": output_families,
            "baseline_sensors": list(baseline_sensors) if baseline_sensors is not None else None,
            **detected,
        },
        "selection": {
            "candidate_observations": len(observations),
            "target_observations": args.target_observations,
            "initial_observations": initial_count,
            "selected_observations": len(selected_observations),
            "selected_views": len(output_views),
            "max_observations": args.max_observations,
            "max_output_views": args.max_output_views,
            "trajectory_length_m": float(translations.sum()),
            "hybrid_motion_length": float(cumulative_motion[-1]),
            "residual_flagged_edges": sum(bool(edge["violations"]) for edge in edges),
        },
        "thresholds": {
            "rotation_weight_m_per_rad": args.rotation_weight_m_per_rad,
            "max_motion_gap": args.max_motion_gap,
            "max_translation_gap_m": args.max_translation_gap_m,
            "max_rotation_gap_deg": args.max_rotation_gap_deg,
            "min_shared_tracks": args.min_shared_tracks,
            "min_overlap_ratio": args.min_overlap_ratio,
        },
        "baseline_pairs": baselines,
        "baseline_check": baseline_check,
        "selected": [
            {
                "index": index,
                "timestamp": observations[index].timestamp,
                "center_xyz": observations[index].center.tolist(),
                "center_xyz_m": observations[index].center.tolist(),
                "anchor_names": [view.name for view in observations[index].views],
                "bridge_reasons": sorted(set(bridge_reasons.get(index, []))),
            }
            for index in selected_indices
        ],
        "edges": edges,
        "selected_names_sha256": stable_sha256(names),
        "duration_seconds": round(time.perf_counter() - started, 4),
    }
    (output / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.no_visuals:
        save_dashboard(
            observations,
            selected_indices,
            cumulative_motion,
            edges,
            visuals / "01_selection_dashboard_4k.jpg",
            args.scene_id,
            args,
            len(output_views),
            baseline_check,
        )
        if args.images is not None and args.contact_sheet_count > 0:
            save_contact_sheet(
                selected_observations,
                by_timestamp,
                args.images.resolve(),
                visuals / "02_selected_contact_sheet.jpg",
                args.contact_sheet_count,
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "observations": len(selected_observations),
                "views": len(output_views),
                "flagged_edges": manifest["selection"]["residual_flagged_edges"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
