#!/usr/bin/env python3
"""Derive evidence-tier catalogs and visual QA from a FARM scene state.

The raw mapper intentionally retains fresh single-view objects.  This report
does not mutate the scene state; it exports practical downstream catalogs that
separate tentative detections from objects supported by multiple observations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from farm_geometry_axes import horizontal_plane_basis, normalize_up_vector, resolve_up_policy
except ModuleNotFoundError:  # package import: scripts.analyze_farm_scene_quality
    from scripts.farm_geometry_axes import (
        horizontal_plane_basis,
        normalize_up_vector,
        resolve_up_policy,
    )


TIER_ORDER = ("single_view", "tentative", "multi_view", "robust", "stable")
TIER_COLORS = {
    "single_view": "#667085",
    "tentative": "#f4b942",
    "multi_view": "#22d3ee",
    "robust": "#4ade80",
    "stable": "#f472b6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--prep-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--presentation-catalog",
        type=Path,
        default=None,
        help="Authoritative exported presentation catalog for visible-object counts.",
    )
    parser.add_argument(
        "--acceptance-report",
        type=Path,
        default=None,
        help="Optional final-acceptance result used as the authoritative release gate.",
    )
    parser.add_argument(
        "--timing-summary",
        type=Path,
        default=None,
        help="Optional JSON with per-stage/total wall-clock timings.",
    )
    parser.add_argument(
        "--resource-summary",
        type=Path,
        default=None,
        help="Optional JSON with per-stage/run peak GPU-memory measurements.",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Scene identifier for report titles; otherwise read from the inputs.",
    )
    parser.add_argument(
        "--require-metric-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether missing/invalid metric-scale evidence fails QA.",
    )
    parser.add_argument(
        "--require-semantics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require captions/Qwen embeddings for QA; disable for geometry-first mapping.",
    )
    return parser.parse_args()


def as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def embedding_dim(rows: list, index: int) -> int:
    if index >= len(rows):
        return 0
    value = rows[index]
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    try:
        return len(value)
    except TypeError:
        return 0


def evidence_tier(observations: int) -> str:
    if observations >= 10:
        return "stable"
    if observations >= 5:
        return "robust"
    if observations >= 3:
        return "multi_view"
    if observations == 2:
        return "tentative"
    return "single_view"


def detection_confidence(state: dict, index: int, category: str) -> float | None:
    rows = state.get("object_detection_category_conf") or []
    if index >= len(rows) or not isinstance(rows[index], dict) or not rows[index]:
        return None
    candidates = rows[index]
    if category in candidates:
        return float(candidates[category])
    return float(max(candidates.values()))


def region_membership(state: dict) -> dict[int, list[str]]:
    output: dict[int, list[str]] = {}
    labels = list(state.get("region_labels") or [])
    object_lists = list(state.get("region_object_lists") or [])
    for region_index, object_ids in enumerate(object_lists):
        label = str(labels[region_index]) if region_index < len(labels) else f"region_{region_index}"
        for object_id in object_ids:
            output.setdefault(int(object_id), []).append(label)
    return output


def camera_positions(state: dict) -> np.ndarray:
    rows = []
    for value in state.get("image_positions") or []:
        point = as_numpy(value).reshape(-1)
        if len(point) >= 3 and np.isfinite(point[:3]).all():
            rows.append(point[:3])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def _mapping(value: object) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_numeric_values(value: object) -> list[float]:
    """Collect numeric leaves while rejecting booleans and non-finite values."""

    output: list[float] = []
    if isinstance(value, Mapping):
        for item in value.values():
            output.extend(_finite_numeric_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            output.extend(_finite_numeric_values(item))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            output.append(number)
    return output


def _nonfinite_numeric_count(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(_nonfinite_numeric_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nonfinite_numeric_count(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return int(not np.isfinite(float(value)))
    return 0


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _source(path: Path, json_path: str) -> str:
    return f"{path.expanduser().resolve()}#{json_path}"


def _metric(
    value: object,
    *,
    unit: str | None,
    source: str,
    definition: str,
    threshold: str | None = None,
    status: str | None = None,
    sample_size: int | None = None,
) -> dict:
    return {
        "value": _json_safe(value),
        "unit": unit,
        "source": source,
        "definition": definition,
        "threshold": threshold,
        "status": status,
        "sample_size": sample_size,
    }


def _count_value(value: object) -> int | None:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(value)
    return None


def resolve_scene_id(explicit: str | None, prep: dict, selection: dict) -> str:
    candidates = (
        explicit,
        prep.get("scene_id"),
        _mapping(prep.get("scene")).get("scene_id"),
        selection.get("scene_id"),
        _mapping(selection.get("scene")).get("scene_id"),
    )
    for value in candidates:
        if value is not None and str(value).strip():
            return " ".join(str(value).strip().split())
    return "scene"


def selection_summary(selection: dict) -> dict:
    selected = _mapping(selection.get("selection")) or selection
    observation_value = None
    source_key = "selected_observations"
    source_path = "$.selection.selected_observations"
    candidates = (
        (selected.get("selected_observations"), "selected_observations", "$.selection.selected_observations"),
        (selection.get("selected_observations"), "selected_observations", "$.selected_observations"),
        (selected.get("selected_timestamps"), "selected_timestamps", "$.selection.selected_timestamps"),
        (selection.get("selected_timestamps"), "selected_timestamps", "$.selected_timestamps"),
        (selection.get("selected"), "selected", "$.selected"),
    )
    for candidate, candidate_key, candidate_path in candidates:
        if candidate is not None:
            observation_value = candidate
            source_key = candidate_key
            source_path = candidate_path
            break
    observations = _count_value(observation_value)
    views = _count_value(selected.get("selected_views"))
    views_path = "$.selection.selected_views"
    if views is None:
        views = _count_value(selection.get("selected_views"))
        views_path = "$.selected_views"
    residual = selected.get("residual_flagged_edges")
    residual_path = "$.selection.residual_flagged_edges"
    if residual is None:
        residual = selection.get("residual_flagged_edges")
        residual_path = "$.residual_flagged_edges"
    graph_connected = (
        isinstance(residual, (int, float, np.integer, np.floating))
        and not isinstance(residual, bool)
        and np.isfinite(residual)
        and float(residual) == 0.0
    )
    inputs = _mapping(selection.get("inputs")) or _mapping(selection.get("source"))
    registered = inputs.get("registered_images", selection.get("registered_images"))
    return {
        "observations": observations,
        "observation_source": source_key,
        "observation_path": source_path,
        "views": views,
        "views_path": views_path,
        "residual_flagged_edges": residual,
        "residual_path": residual_path,
        "graph_connected": bool(graph_connected),
        "registered_images": registered,
    }


def alignment_summary(prep: dict) -> dict:
    alignment = _mapping(prep.get("alignment_qa"))
    if isinstance(alignment.get("result"), Mapping):
        alignment = _mapping(alignment["result"])
    reasons: list[str] = []
    if not alignment:
        reasons.append("alignment_qa is missing")
    if alignment.get("passed") is not True:
        reasons.append("alignment_qa.passed is not true")
    thresholds = _mapping(alignment.get("thresholds")) or _mapping(
        alignment.get("configured_thresholds")
    )
    aggregate = (
        _mapping(alignment.get("aggregate"))
        or _mapping(alignment.get("metrics"))
        or _mapping(alignment.get("summary"))
    )
    threshold_values = _finite_numeric_values(thresholds)
    aggregate_values = _finite_numeric_values(aggregate)
    if not thresholds or not threshold_values:
        reasons.append("alignment_qa configured thresholds are missing or non-finite")
    if not aggregate or not aggregate_values:
        reasons.append("alignment_qa aggregate metrics are missing or non-finite")
    if _nonfinite_numeric_count(thresholds):
        reasons.append("alignment_qa contains a non-finite configured threshold")
    if _nonfinite_numeric_count(aggregate):
        reasons.append("alignment_qa contains a non-finite aggregate metric")
    return {
        "passed": not reasons,
        "reported_passed": alignment.get("passed"),
        "thresholds": thresholds,
        "aggregate": aggregate,
        "source_path": "$.alignment_qa",
        "reasons": reasons,
    }


def metric_summary(
    prep: dict,
    selection: dict,
    required_override: bool | None,
) -> dict:
    metric = (
        _mapping(prep.get("metric_scale_check"))
        or _mapping(prep.get("metric_scale"))
        or _mapping(prep.get("scale_check"))
    )
    source_path = "$.metric_scale_check"
    source_owner = "prep"
    if metric and not _mapping(prep.get("metric_scale_check")):
        source_path = "$.metric_scale" if _mapping(prep.get("metric_scale")) else "$.scale_check"
    if not metric:
        metric = _mapping(selection.get("metric_scale_check"))
        source_path = "$.metric_scale_check"
        source_owner = "selection"
    if not metric:
        metric = _mapping(selection.get("metric_scale")) or _mapping(selection.get("baseline_check"))
        source_path = "$.metric_scale" if _mapping(selection.get("metric_scale")) else "$.baseline_check"
        source_owner = "selection"
    if required_override is not None:
        required = bool(required_override)
        required_source = "cli"
    elif "required" in metric:
        required = bool(metric.get("required"))
        required_source = "metric.required"
    elif "enabled" in metric:
        required = bool(metric.get("enabled"))
        required_source = "metric.enabled"
    elif "require_metric_scale" in prep:
        required = bool(prep.get("require_metric_scale"))
        required_source = "prep.require_metric_scale"
    else:
        required = True
        required_source = "legacy_default"
    evidence_keys = (
        "median_scene_units",
        "median_rig_baseline_scene_units",
        "observed_median",
        "meters_per_colmap_unit",
        "inferred_meters_per_colmap_unit",
        "median_baseline_m",
        "baseline_m",
        "observed_baseline_m",
        "scale_m_per_unit",
        "meters_per_unit",
    )
    evidence = {
        key: metric[key]
        for key in evidence_keys
        if key in metric
        and isinstance(metric[key], (int, float, np.integer, np.floating))
        and not isinstance(metric[key], bool)
        and np.isfinite(metric[key])
    }
    reasons: list[str] = []
    if required and not metric:
        reasons.append("metric-scale evidence is missing")
    if required and metric.get("passed") is not True:
        reasons.append("metric-scale check did not explicitly pass")
    if required and not evidence:
        reasons.append("metric-scale measurement is missing or non-finite")
    invalid_evidence = [
        key
        for key in evidence_keys
        if key in metric
        and isinstance(metric[key], (int, float, np.integer, np.floating))
        and not isinstance(metric[key], bool)
        and not np.isfinite(metric[key])
    ]
    if required and invalid_evidence:
        reasons.append(f"non-finite metric-scale values: {', '.join(invalid_evidence)}")
    return {
        "required": required,
        "required_source": required_source,
        "passed": (not required) or not reasons,
        "reported_passed": metric.get("passed"),
        "evidence": evidence,
        "details": metric,
        "source_path": source_path,
        "source_owner": source_owner,
        "reasons": reasons,
    }


def mapping_state_health(state: dict) -> dict:
    reasons: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    expected_widths = {"means": 3, "cov6": 6}
    for key, width in expected_widths.items():
        if key not in state:
            reasons.append(f"{key} is missing")
        try:
            array = np.asarray(
                as_numpy(state.get(key, np.empty((0, width)))), dtype=np.float64
            )
        except Exception as exc:  # corrupt tensor-like input must fail closed
            reasons.append(f"{key} is unreadable: {exc}")
            array = np.empty((0, width))
        if array.ndim != 2 or array.shape[1] != width:
            reasons.append(f"{key} must have shape [N,{width}]")
            array = np.empty((0, width))
        arrays[key] = array
    n_objects = int(arrays["means"].shape[0])
    if arrays["cov6"].shape[0] != n_objects:
        reasons.append("means/cov6 object counts differ")
    vectors: dict[str, np.ndarray] = {}
    defaults = {
        "active": np.zeros(n_objects, dtype=bool),
        "count": np.zeros(n_objects, dtype=np.int64),
        "object_id": np.arange(n_objects, dtype=np.int64),
    }
    for key, default in defaults.items():
        if key not in state:
            reasons.append(f"{key} is missing")
        try:
            array = as_numpy(state.get(key, default)).reshape(-1)
        except Exception as exc:
            reasons.append(f"{key} is unreadable: {exc}")
            array = np.empty(0)
        if len(array) != n_objects:
            reasons.append(f"{key} length differs from means")
        vectors[key] = array
    nonfinite_means = int((~np.isfinite(arrays["means"])).sum())
    nonfinite_cov6 = int((~np.isfinite(arrays["cov6"])).sum())
    try:
        nonfinite_counts = (
            int((~np.isfinite(vectors["count"].astype(np.float64))).sum())
            if len(vectors["count"]) else 0
        )
    except (TypeError, ValueError):
        reasons.append("count must be numeric")
        nonfinite_counts = int(len(vectors["count"]))
    active_count = (
        int(vectors["active"].astype(bool).sum())
        if len(vectors["active"]) == n_objects else 0
    )
    structurally_valid = not reasons
    return {
        "structurally_valid": structurally_valid,
        "finite_geometry": nonfinite_means == 0 and nonfinite_cov6 == 0 and nonfinite_counts == 0,
        "nonempty_active": active_count > 0,
        "object_count": n_objects,
        "active_count": active_count,
        "inactive_count": max(0, n_objects - active_count),
        "nonfinite_means": nonfinite_means,
        "nonfinite_cov6": nonfinite_cov6,
        "nonfinite_counts": nonfinite_counts,
        "reasons": reasons,
    }


def state_up_policy(state: dict):
    vector = state.get("object_geometry_up_vector")
    axis = str(state.get("object_geometry_up_axis", "y")).lower()
    direction = str(state.get("object_geometry_up_direction", "positive")).lower()
    if vector is not None:
        return resolve_up_policy(up_vector=normalize_up_vector(vector))
    if axis == "vector":
        raise ValueError("state declares vector up but object_geometry_up_vector is missing")
    return resolve_up_policy(up_axis=axis, up_direction=direction)


def ground_projection(points: np.ndarray, up_policy) -> tuple[np.ndarray, tuple[str, str]]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if up_policy.axis is not None:
        indices = [index for index in range(3) if index != {"x": 0, "y": 1, "z": 2}[up_policy.axis]]
        labels = tuple(f"world {'XYZ'[index]}, m" for index in indices)
        return values[:, indices], labels
    basis = horizontal_plane_basis(up_policy.vector)
    return values @ basis, ("ground basis U, m", "ground basis V, m")


def load_optional_json(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Optional summary must contain a JSON object: {path}")
    return dict(payload)


def acceptance_release_check(report: Mapping, presentation_count: int) -> tuple[bool, dict]:
    """Validate a persisted acceptance result against the exported catalog."""

    counts = _mapping(report.get("counts"))
    status = str(report.get("status") or "").upper()
    reasons: list[str] = []
    if status not in {"PASS", "WARN"}:
        reasons.append("status_not_releaseable")
    for key in (
        "remaining_auto_clusters",
        "remaining_blocking_clusters",
        "label_hard_errors",
    ):
        try:
            value = int(counts[key])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{key}_missing_or_invalid")
        else:
            if value != 0:
                reasons.append(f"{key}_nonzero")
    try:
        accepted_count = int(counts["presentation_after"])
    except (KeyError, TypeError, ValueError):
        accepted_count = None
        reasons.append("presentation_after_missing_or_invalid")
    if accepted_count is not None and accepted_count != presentation_count:
        reasons.append("presentation_after_catalog_mismatch")
    return not reasons, {
        "status": status or "MISSING",
        "counts": counts,
        "presentation_catalog_count": presentation_count,
        "reasons": reasons,
    }


def _finite_number(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number) and number >= 0.0:
            return number
    return None


def _extract_value(record: Mapping, keys: tuple[tuple[str, float], ...]) -> tuple[float | None, str | None]:
    for key, factor in keys:
        number = _finite_number(record.get(key))
        if number is not None:
            return number * factor, key
    return None, None


DURATION_KEYS = (
    ("duration_seconds", 1.0),
    ("wall_seconds", 1.0),
    ("elapsed_seconds", 1.0),
    ("seconds", 1.0),
)
DEVICE_PEAK_USED_KEYS = (
    ("device_peak_used_gib", 1.0),
    ("peak_used_gib", 1.0),
    ("peak_used_mb", 1.0 / 1024.0),
    # Legacy summaries did not encode their scope. They remain readable, but
    # the profile marks that scope as unspecified rather than implying PID
    # attribution.
    ("peak_vram_gib", 1.0),
    ("peak_gpu_memory_gib", 1.0),
    ("max_vram_gib", 1.0),
    ("peak_vram_mib", 1.0 / 1024.0),
    ("peak_gpu_memory_mib", 1.0 / 1024.0),
    ("peak_vram_mb", 1.0 / 1024.0),
    ("peak_vram_bytes", 1.0 / float(2**30)),
    ("peak_gpu_memory_bytes", 1.0 / float(2**30)),
)
INCREMENTAL_PEAK_DELTA_KEYS = (
    ("incremental_peak_delta_gib", 1.0),
    ("peak_delta_gib", 1.0),
    ("peak_delta_mb", 1.0 / 1024.0),
    ("gpu_device_peak_delta_mb", 1.0 / 1024.0),
)


def _stage_records(payload: dict, source_file: Path | None) -> list[dict]:
    if not payload or source_file is None:
        return []
    container: object = None
    container_path = "$"
    for key in ("stages", "stage_timings", "timings", "stage_resources"):
        if isinstance(payload.get(key), (Mapping, list)):
            container = payload[key]
            container_path = f"$.{key}"
            break
    rows: list[tuple[str, Mapping, str]] = []
    if isinstance(container, Mapping):
        for name, value in container.items():
            if isinstance(value, Mapping):
                rows.append((str(name), value, f"{container_path}.{name}"))
            elif _finite_number(value) is not None:
                rows.append((str(name), {"duration_seconds": value}, f"{container_path}.{name}"))
    elif isinstance(container, list):
        for index, value in enumerate(container):
            if not isinstance(value, Mapping):
                continue
            name = value.get("stage") or value.get("name") or f"stage_{index}"
            rows.append((str(name), value, f"{container_path}[{index}]"))
    output = []
    for name, record, record_path in rows:
        duration, duration_key = _extract_value(record, DURATION_KEYS)
        device_peak, device_peak_key = _extract_value(record, DEVICE_PEAK_USED_KEYS)
        incremental_peak, incremental_peak_key = _extract_value(
            record, INCREMENTAL_PEAK_DELTA_KEYS
        )
        measurement_scope = str(record.get("measurement_scope") or "").strip() or None
        output.append(
            {
                "stage": name,
                "duration_seconds": duration,
                "duration_source": (
                    _source(source_file, f"{record_path}.{duration_key}") if duration_key else None
                ),
                "device_peak_used_gib": device_peak,
                "device_peak_used_source": (
                    _source(source_file, f"{record_path}.{device_peak_key}")
                    if device_peak_key else None
                ),
                "incremental_peak_delta_gib": incremental_peak,
                "incremental_peak_delta_source": (
                    _source(source_file, f"{record_path}.{incremental_peak_key}")
                    if incremental_peak_key else None
                ),
                "measurement_scope": measurement_scope,
                "measurement_scope_source": (
                    _source(source_file, f"{record_path}.measurement_scope")
                    if measurement_scope else None
                ),
            }
        )
    return output


def _global_value(
    payload: dict,
    source_file: Path | None,
    keys: tuple[tuple[str, float], ...],
) -> tuple[float | None, str | None]:
    if not payload or source_file is None:
        return None, None
    containers = [(payload, "$")]
    for key in ("summary", "total", "resources"):
        if isinstance(payload.get(key), Mapping):
            containers.append((payload[key], f"$.{key}"))
    for container, container_path in containers:
        value, key = _extract_value(container, keys)
        if key:
            return value, _source(source_file, f"{container_path}.{key}")
    return None, None


def run_profile(
    timing_payload: dict,
    timing_path: Path | None,
    resource_payload: dict,
    resource_path: Path | None,
) -> dict:
    merged: dict[str, dict] = {}
    for row in _stage_records(timing_payload, timing_path) + _stage_records(
        resource_payload, resource_path
    ):
        target = merged.setdefault(
            row["stage"],
            {
                "stage": row["stage"],
                "duration_seconds": None,
                "duration_source": None,
                "device_peak_used_gib": None,
                "device_peak_used_source": None,
                "incremental_peak_delta_gib": None,
                "incremental_peak_delta_source": None,
                "measurement_scope": None,
                "measurement_scope_source": None,
            },
        )
        for key in (
            "duration_seconds",
            "duration_source",
            "device_peak_used_gib",
            "device_peak_used_source",
            "incremental_peak_delta_gib",
            "incremental_peak_delta_source",
            "measurement_scope",
            "measurement_scope_source",
        ):
            if row.get(key) is not None:
                target[key] = row[key]
    stages = list(merged.values())
    total, total_source = _global_value(
        timing_payload,
        timing_path,
        (("total_seconds", 1.0), ("duration_seconds", 1.0), ("wall_seconds", 1.0)),
    )
    if total is None:
        durations = [row["duration_seconds"] for row in stages if row["duration_seconds"] is not None]
        if durations:
            total = float(sum(durations))
            sources = [row["duration_source"] for row in stages if row["duration_source"]]
            total_source = "derived:sum(" + ",".join(sources) + ")"
    device_peak, device_peak_source = _global_value(
        resource_payload, resource_path, DEVICE_PEAK_USED_KEYS
    )
    if device_peak is None:
        peaks = [
            row["device_peak_used_gib"]
            for row in stages
            if row["device_peak_used_gib"] is not None
        ]
        if peaks:
            device_peak = float(max(peaks))
            sources = [
                row["device_peak_used_source"]
                for row in stages
                if row["device_peak_used_source"]
            ]
            device_peak_source = "derived:max(" + ",".join(sources) + ")"
    incremental_peak, incremental_peak_source = _global_value(
        resource_payload, resource_path, INCREMENTAL_PEAK_DELTA_KEYS
    )
    if incremental_peak is None:
        deltas = [
            row["incremental_peak_delta_gib"]
            for row in stages
            if row["incremental_peak_delta_gib"] is not None
        ]
        if deltas:
            incremental_peak = float(max(deltas))
            sources = [
                row["incremental_peak_delta_source"]
                for row in stages
                if row["incremental_peak_delta_source"]
            ]
            incremental_peak_source = "derived:max(" + ",".join(sources) + ")"
    measurement_scope = str(resource_payload.get("measurement_scope") or "").strip()
    measurement_scope_source: str | None = None
    if measurement_scope and resource_path is not None:
        measurement_scope_source = _source(resource_path, "$.measurement_scope")
    if not measurement_scope:
        stage_scopes = sorted(
            {
                str(row["measurement_scope"])
                for row in stages
                if row.get("measurement_scope")
            }
        )
        if len(stage_scopes) == 1:
            measurement_scope = stage_scopes[0]
            sources = [
                row["measurement_scope_source"]
                for row in stages
                if row.get("measurement_scope_source")
            ]
            measurement_scope_source = "derived:unique(" + ",".join(sources) + ")"
        elif stage_scopes:
            measurement_scope = "mixed:" + ",".join(stage_scopes)
            sources = [
                row["measurement_scope_source"]
                for row in stages
                if row.get("measurement_scope_source")
            ]
            measurement_scope_source = "derived:mixed(" + ",".join(sources) + ")"
        elif device_peak is not None or incremental_peak is not None:
            measurement_scope = "legacy_unspecified_not_assumed_process_attributed"
            measurement_scope_source = "derived:legacy GPU-memory key without scope"
        else:
            measurement_scope = "unavailable"
            measurement_scope_source = "unavailable:no resource summary supplied"
    return {
        "total_seconds": total,
        "total_source": total_source or "unavailable:no timing summary supplied",
        "device_peak_used_gib": device_peak,
        "device_peak_used_source": (
            device_peak_source or "unavailable:no resource summary supplied"
        ),
        "incremental_peak_delta_gib": incremental_peak,
        "incremental_peak_delta_source": (
            incremental_peak_source or "unavailable:no resource summary supplied"
        ),
        "memory_measurement_scope": measurement_scope,
        "memory_measurement_scope_source": measurement_scope_source,
        "stages": stages,
    }


def duplicate_summary(state: dict, presentation_count: int | None = None) -> dict:
    display_statuses = state.get("object_display_status")
    canonical_value = state.get("object_duplicate_canonical_id")
    if isinstance(display_statuses, (list, tuple)):
        canonical_ids = (
            as_numpy(canonical_value).reshape(-1)
            if canonical_value is not None else np.full(len(display_statuses), -1)
        )
        if len(canonical_ids) != len(display_statuses):
            raise ValueError("object_display_status and object_duplicate_canonical_id differ in length")
        suppressed = [
            index for index, value in enumerate(display_statuses)
            if str(value).endswith("_suppressed")
        ]
        groups = {
            int(canonical_ids[index]) for index in suppressed
            if int(canonical_ids[index]) >= 0
        }
        active = as_numpy(state.get("active", np.zeros(len(display_statuses)))).astype(bool).reshape(-1)
        geometry_statuses = list(state.get("object_geometry_status") or [""] * len(display_statuses))
        semantic_tiers = list(state.get("object_semantic_tier") or [""] * len(display_statuses))
        compound_boxes = list(state.get("object_compound_boxes") or [[] for _ in display_statuses])
        probable = sum(
            "compound_geometry_probable" in str(status).lower()
            and "rejected" not in str(status).lower()
            and index < len(compound_boxes)
            and len(compound_boxes[index] or []) == 2
            for index, status in enumerate(geometry_statuses)
        )
        derived_presentation_visible = sum(
            not str(display_statuses[index]).endswith("_suppressed")
            and (
                (
                    index < len(active) and bool(active[index])
                    and index < len(semantic_tiers)
                    and str(semantic_tiers[index]).strip().lower() in {"confirmed", "probable"}
                )
                or (
                    index < len(geometry_statuses)
                    and "compound_geometry_probable" in str(geometry_statuses[index]).lower()
                    and "rejected" not in str(geometry_statuses[index]).lower()
                    and index < len(compound_boxes)
                    and len(compound_boxes[index] or []) == 2
                )
            )
            for index in range(len(display_statuses))
        )
        return {
            "available": True,
            "resolved_groups": len(groups),
            "resolved_losers": len(suppressed),
            "suppressed_duplicates": sum(str(value) == "duplicate_suppressed" for value in display_statuses),
            "suppressed_parts": sum(str(value) == "part_suppressed" for value in display_statuses),
            "suppressed_assembly_members": sum(str(value) == "assembly_member_suppressed" for value in display_statuses),
            "metric_active": int(active.sum()),
            "presentation_visible": int(
                presentation_count
                if presentation_count is not None else derived_presentation_visible
            ),
            "presentation_visible_derived_from_state": int(derived_presentation_visible),
            "presentation_count_authoritative": presentation_count is not None,
            "probable_compounds": int(probable),
            "source_path": "$.state.object_display_status,object_duplicate_canonical_id",
        }
    rows = state.get("loser_object_ids")
    if not isinstance(rows, (list, tuple)):
        return {
            "available": False,
            "resolved_groups": None,
            "resolved_losers": None,
            "presentation_visible": presentation_count,
            "presentation_count_authoritative": presentation_count is not None,
            "probable_compounds": None,
            "source_path": "$.state.loser_object_ids",
        }
    groups = 0
    losers: set[int] = set()
    for row in rows:
        if not isinstance(row, (list, tuple, set)) or not row:
            continue
        groups += 1
        for value in row:
            try:
                losers.add(int(value))
            except (TypeError, ValueError):
                continue
    return {
        "available": True,
        "resolved_groups": groups,
        "resolved_losers": len(losers),
        "presentation_visible": presentation_count,
        "presentation_count_authoritative": presentation_count is not None,
        "probable_compounds": None,
        "source_path": "$.state.loser_object_ids",
    }


def object_records(state: dict) -> tuple[list[dict], dict[str, int]]:
    means = as_numpy(state.get("means", np.zeros((0, 3)))).reshape(-1, 3)
    refined = state.get("object_box_centers_m")
    centers = means if refined is None else as_numpy(refined).reshape(-1, 3)
    if centers.shape != means.shape:
        raise ValueError("object_box_centers_m must align with means")
    cov6 = as_numpy(state.get("cov6", np.zeros((len(means), 6)))).reshape(-1, 6)
    active = as_numpy(state.get("active", np.ones(len(means), dtype=bool))).astype(bool)
    counts = as_numpy(state.get("count", np.zeros(len(means), dtype=int))).reshape(-1)
    object_ids = as_numpy(state.get("object_id", np.arange(len(means)))).reshape(-1)
    categories = list(state.get("object_category") or [])
    supercategories = list(state.get("object_supercategory") or [])
    captions = list(state.get("object_caption") or [])
    qwen = list(state.get("object_qwen3_vl_embedding") or [])
    siglip = list(state.get("object_siglip2_embedding") or [])
    region_by_object = region_membership(state)

    records = []
    for index in np.flatnonzero(active):
        category = str(categories[index] or "unlabeled") if index < len(categories) else "unlabeled"
        observations = int(counts[index])
        record = {
            "id": int(object_ids[index]),
            "evidence_tier": evidence_tier(observations),
            "observations": observations,
            "category": category,
            "supercategory": (
                str(supercategories[index] or "unknown")
                if index < len(supercategories) else "unknown"
            ),
            "caption": str(captions[index] or "") if index < len(captions) else "",
            "detection_confidence": detection_confidence(state, int(index), category),
            "position_world_m": centers[index].astype(float).tolist(),
            "cov6": cov6[index].astype(float).tolist(),
            "qwen3_vl_embedding_dim": embedding_dim(qwen, int(index)),
            "siglip2_embedding_dim": embedding_dim(siglip, int(index)),
            "regions_experimental": region_by_object.get(int(index), []),
        }
        records.append(record)
    records.sort(key=lambda item: (-item["observations"], item["category"], item["id"]))
    invariants = {
        "nan_means": int(np.isnan(means).sum()),
        "nan_cov6": int(np.isnan(cov6).sum()),
        "nonfinite_means": int((~np.isfinite(means)).sum()),
        "nonfinite_object_box_centers": int((~np.isfinite(centers)).sum()),
        "nonfinite_cov6": int((~np.isfinite(cov6)).sum()),
        "active_count": int(active.sum()),
        "inactive_count": int((~active).sum()),
    }
    return records, invariants


def _format_duration(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value >= 3600.0:
        return f"{value / 3600.0:.2f} h"
    if value >= 60.0:
        return f"{value / 60.0:.1f} min"
    return f"{value:.1f} s"


def load_presentation_catalog(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("presentation catalog must be a JSON array")
    ids: list[int] = []
    for row in payload:
        if not isinstance(row, dict) or row.get("id") is None:
            raise ValueError("every presentation catalog row must contain an object id")
        ids.append(int(row["id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("presentation catalog contains duplicate object ids")
    return payload


def select_chart_stage_rows(
    rows: list[dict], key: str, limit: int,
) -> tuple[list[dict], int]:
    eligible = [row for row in rows if row.get(key) is not None]
    ranked = sorted(
        eligible,
        key=lambda row: (-float(row[key]), str(row.get("stage") or "")),
    )[:limit]
    return sorted(
        ranked, key=lambda row: (float(row[key]), str(row.get("stage") or ""))
    ), len(eligible)


def stage_display_name(value: object) -> str:
    return str(value or "unknown").replace("_", " ")


def save_dashboard(
    records: list[dict],
    positions: np.ndarray,
    output_path: Path,
    *,
    scene_id: str,
    up_policy: object,
    checks: dict[str, bool],
    selection_info: dict,
    alignment_info: dict,
    run_info: dict,
    duplicate_info: dict,
    caption_coverage: int,
    qwen_coverage: int,
    siglip_coverage: int,
    acceptance_info: dict | None = None,
) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 13.5), dpi=160, facecolor="#0a0d14")
    grid = fig.add_gridspec(
        3, 6, height_ratios=(0.27, 1.0, 1.0), hspace=0.34, wspace=0.34
    )
    reliable = [row for row in records if row["observations"] >= 3]
    overall_pass = all(checks.values())
    hero = (
        ("STRUCTURAL QA", "PASS" if overall_pass else "NEEDS REVIEW"),
        ("SELECTED VIEWS", str(selection_info.get("views") or "unavailable")),
        ("METRIC OBJECTS", str(len(records))),
        ("PRESENTATION", str(duplicate_info.get("presentation_visible", "unavailable"))),
        ("TOTAL TIME", _format_duration(run_info.get("total_seconds"))),
        (
            "DEVICE PEAK USED",
            (
                f"{run_info['device_peak_used_gib']:.1f} GiB"
                if run_info.get("device_peak_used_gib") is not None else "unavailable"
            ),
        ),
    )
    for index, (label, value) in enumerate(hero):
        axis = fig.add_subplot(grid[0, index])
        axis.set_facecolor("#111827")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#263244")
        axis.text(0.06, 0.76, label, transform=axis.transAxes, color="#9aa6b5", fontsize=9)
        color = "#4ade80" if label == "STRUCTURAL QA" and overall_pass else "#e8edf4"
        if label == "STRUCTURAL QA" and not overall_pass:
            color = "#f59e0b"
        axis.text(
            0.06, 0.29, value, transform=axis.transAxes, color=color,
            fontsize=17, fontweight="bold", va="center",
        )

    ax_map = fig.add_subplot(grid[1:, :3])
    ax_time = fig.add_subplot(grid[1, 3:5])
    ax_vram = fig.add_subplot(grid[1, 5])
    ax_tiers = fig.add_subplot(grid[2, 3:5])
    ax_checks = fig.add_subplot(grid[2, 5])
    for axis in (ax_map, ax_time, ax_vram, ax_tiers, ax_checks):
        axis.set_facecolor("#0f1520")

    projected_positions, axis_labels = ground_projection(positions, up_policy)
    if len(projected_positions):
        ax_map.plot(
            projected_positions[:, 0], projected_positions[:, 1], color="#596579",
            linewidth=1.0, alpha=0.75, label=f"camera trajectory (n={len(positions)})", zorder=0,
        )
    for tier in TIER_ORDER:
        tier_rows = [row for row in records if row["evidence_tier"] == tier]
        if not tier_rows:
            continue
        points_world = np.asarray([row["position_world_m"] for row in tier_rows])
        points, _ = ground_projection(points_world, up_policy)
        sizes = np.asarray([row["observations"] for row in tier_rows], dtype=float)
        ax_map.scatter(
            points[:, 0], points[:, 1], s=16 + np.minimum(sizes, 30) * 1.8,
            color=TIER_COLORS[tier], edgecolors="#e8edf4", linewidths=0.3,
            alpha=0.86, label=f"{tier}: {len(tier_rows)}", zorder=2,
        )
    for row in reliable[:10]:
        point, _ = ground_projection(np.asarray([row["position_world_m"]]), up_policy)
        ax_map.annotate(
            f"{row['id']}: {row['category']}", point[0], xytext=(4, 5),
            textcoords="offset points", color="#e8edf4", fontsize=7,
        )
    ax_map.set_title(
        f"Object evidence map · ground-plane projection (active n={len(records)})", fontsize=14
    )
    ax_map.set_xlabel(axis_labels[0])
    ax_map.set_ylabel(axis_labels[1])
    ax_map.grid(color="#344054", alpha=0.27)
    ax_map.axis("equal")
    ax_map.legend(loc="best", fontsize=8, facecolor="#101828")

    timing_rows, timing_total = select_chart_stage_rows(
        run_info["stages"], "duration_seconds", 8
    )
    if timing_rows:
        bars = ax_time.barh(
            [stage_display_name(row["stage"]) for row in timing_rows],
            [row["duration_seconds"] for row in timing_rows],
            color="#38bdf8", alpha=0.82,
        )
        ax_time.bar_label(
            bars,
            labels=[_format_duration(row["duration_seconds"]) for row in timing_rows],
            padding=3, fontsize=8,
        )
        ax_time.set_xlabel("wall time, s")
        ax_time.set_xlim(0.0, max(row["duration_seconds"] for row in timing_rows) * 1.18)
        ax_time.grid(axis="x", color="#344054", alpha=0.27)
        ax_time.set_title(
            f"Longest stage times · {len(timing_rows)} shown / {timing_total} measured",
            fontsize=12,
        )
    else:
        ax_time.axis("off")
        ax_time.text(0.5, 0.55, "Stage timing unavailable", ha="center", fontsize=13)
        ax_time.text(0.5, 0.43, "No timing summary supplied", ha="center", color="#9aa6b5", fontsize=9)

    vram_rows, vram_total = select_chart_stage_rows(
        run_info["stages"], "device_peak_used_gib", 6
    )
    if vram_rows:
        labels = [stage_display_name(row["stage"]) for row in vram_rows]
        device_values = [row["device_peak_used_gib"] for row in vram_rows]
        delta_values = [row.get("incremental_peak_delta_gib") for row in vram_rows]
        if any(value is not None for value in delta_values):
            positions = np.arange(len(vram_rows), dtype=float)
            device_bars = ax_vram.barh(
                positions - 0.18, device_values, height=0.34,
                color="#a78bfa", alpha=0.82, label="device used",
            )
            delta_indices = [
                index for index, value in enumerate(delta_values) if value is not None
            ]
            delta_bars = ax_vram.barh(
                positions[delta_indices] + 0.18,
                [delta_values[index] for index in delta_indices],
                height=0.34, color="#f59e0b", alpha=0.82, label="incremental delta",
            )
            ax_vram.set_yticks(positions, labels, fontsize=7)
            ax_vram.bar_label(device_bars, fmt="%.1f", padding=2, fontsize=6)
            ax_vram.bar_label(delta_bars, fmt="%.1f", padding=2, fontsize=6)
            ax_vram.legend(fontsize=6, facecolor="#101828")
        else:
            bars = ax_vram.barh(
                labels, device_values, color="#a78bfa", alpha=0.82,
            )
            ax_vram.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
        ax_vram.set_xlabel("whole-device memory, GiB")
        ax_vram.set_xlim(0.0, max(device_values) * 1.20)
        ax_vram.set_title(
            f"GPU memory · {len(vram_rows)} shown / {vram_total} measured", fontsize=10
        )
        ax_vram.grid(axis="x", color="#344054", alpha=0.27)
    elif run_info.get("device_peak_used_gib") is not None:
        bars = ax_vram.barh(
            ["device peak used"], [run_info["device_peak_used_gib"]], color="#a78bfa"
        )
        ax_vram.bar_label(bars, fmt="%.1f GiB", padding=2, fontsize=8)
        ax_vram.set_xlim(left=0.0)
        ax_vram.set_title("Measured whole-device peak (n=1)", fontsize=11)
    else:
        ax_vram.axis("off")
        ax_vram.text(0.5, 0.55, "VRAM unavailable", ha="center", fontsize=12)
        ax_vram.text(0.5, 0.43, "No resource summary", ha="center", color="#9aa6b5", fontsize=8)

    tier_counts = Counter(row["evidence_tier"] for row in records)
    values = [tier_counts[tier] for tier in TIER_ORDER]
    bars = ax_tiers.bar(TIER_ORDER, values, color=[TIER_COLORS[tier] for tier in TIER_ORDER])
    ax_tiers.bar_label(bars, padding=3, fontsize=9)
    ax_tiers.set_title(f"Object evidence tiers (active n={len(records)})", fontsize=12)
    ax_tiers.set_ylabel("object count")
    ax_tiers.set_ylim(bottom=0.0)
    ax_tiers.grid(axis="y", color="#344054", alpha=0.27)
    ax_tiers.tick_params(axis="x", rotation=15)
    ax_tiers.text(
        0.01, 0.94, "1 | 2 | 3–4 | 5–9 | 10+ observations",
        transform=ax_tiers.transAxes, color="#9aa6b5", fontsize=8, va="top",
    )

    ax_checks.axis("off")
    denominator = max(len(records), 1)
    acceptance_info = acceptance_info or {}
    acceptance_counts = acceptance_info.get("counts") or {}
    rows = [
        ("SELECTION", "PASS" if checks.get("colmap_selected_graph_connected") else "FAIL"),
        ("views / residual edges", f"{selection_info.get('views')} / {selection_info.get('residual_flagged_edges')}"),
        ("RGB-D ALIGNMENT", "PASS" if checks.get("rgbd_alignment_passed") else "FAIL"),
        ("finite thresholds / metrics", f"{len(_finite_numeric_values(alignment_info.get('thresholds')))} / {len(_finite_numeric_values(alignment_info.get('aggregate')))}"),
        ("GEOMETRY", "PASS" if checks.get("finite_scene_geometry") else "FAIL"),
        ("reliable / active", f"{len(reliable)} / {len(records)}"),
        (
            "presentation / probable",
            f"{duplicate_info.get('presentation_visible', 'n/a')} / "
            f"{duplicate_info.get('probable_compounds', 'n/a')}",
        ),
        (
            "FINAL ACCEPTANCE",
            str(acceptance_info.get("status") or "unavailable"),
        ),
        (
            "shown / held",
            f"{acceptance_counts.get('presentation_after', 'n/a')} / "
            f"{acceptance_counts.get('held_objects', 'n/a')}",
        ),
        (
            "open overlaps / label errors",
            f"{acceptance_counts.get('remaining_auto_clusters', 'n/a')}+"
            f"{acceptance_counts.get('remaining_blocking_clusters', 'n/a')} / "
            f"{acceptance_counts.get('label_hard_errors', 'n/a')}",
        ),
        ("SEMANTICS", f"captions {caption_coverage / denominator:.0%}"),
        ("embedding coverage", f"Qwen {qwen_coverage / denominator:.0%} · SigLIP {siglip_coverage / denominator:.0%}"),
    ]
    y = 0.98
    for label, value in rows:
        is_header = label in {"SELECTION", "RGB-D ALIGNMENT", "GEOMETRY", "FINAL ACCEPTANCE", "SEMANTICS"}
        color = "#d7dee8" if is_header else "#9aa6b5"
        ax_checks.text(
            0.02, y, label, transform=ax_checks.transAxes, color=color,
            fontsize=8.5 if is_header else 7.8, fontweight="bold" if is_header else "normal", va="top",
        )
        ax_checks.text(
            0.98, y, value, transform=ax_checks.transAxes,
            color="#e8edf4", fontsize=8, ha="right", va="top",
        )
        y -= 0.080 if is_header else 0.065

    fig.suptitle(
        f"{scene_id} · structural and evidence QA",
        fontsize=21, fontweight="bold", x=0.02, ha="left", y=0.992,
    )
    fig.text(
        0.02, 0.014,
        "Structural PASS is not a recall/label guarantee. Full stage tables, sources and thresholds: quality_index.json and REPORT.md.",
        color="#8793a3", fontsize=8.5,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.065)
    fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.pt.expanduser(), map_location="cpu", weights_only=False)
    state = payload.get("state", payload) if isinstance(payload, dict) else {}
    frames_payload = json.loads(args.frames_json.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    prep = json.loads(args.prep_summary.read_text(encoding="utf-8"))
    timing_payload = load_optional_json(args.timing_summary)
    resource_payload = load_optional_json(args.resource_summary)
    presentation_catalog = load_presentation_catalog(args.presentation_catalog)
    acceptance_payload = load_optional_json(args.acceptance_report)
    profile = run_profile(
        timing_payload, args.timing_summary, resource_payload, args.resource_summary
    )
    scene_id = resolve_scene_id(args.scene_id, prep, selection)
    health = mapping_state_health(state)
    try:
        up_policy = state_up_policy(state)
        up_policy_valid = True
        up_policy_reason = None
    except ValueError as exc:
        up_policy = resolve_up_policy(up_axis="y")
        up_policy_valid = False
        up_policy_reason = str(exc)
    if health["structurally_valid"] and health["finite_geometry"]:
        records, invariants = object_records(state)
    else:
        records = []
        invariants = {
            "nan_means": health["nonfinite_means"],
            "nan_cov6": health["nonfinite_cov6"],
            "nonfinite_means": health["nonfinite_means"],
            "nonfinite_cov6": health["nonfinite_cov6"],
            "active_count": health["active_count"],
            "inactive_count": health["inactive_count"],
        }
    reliable = [row for row in records if row["observations"] >= 3]
    stable = [row for row in records if row["observations"] >= 5]
    tier_counts = Counter(row["evidence_tier"] for row in records)
    categories_raw = Counter(row["category"] for row in records)
    categories_reliable = Counter(row["category"] for row in reliable)

    qwen_coverage = sum(row["qwen3_vl_embedding_dim"] > 0 for row in records)
    siglip_coverage = sum(row["siglip2_embedding_dim"] > 0 for row in records)
    caption_coverage = sum(bool(row["caption"].strip()) for row in records)
    selection_info = selection_summary(selection)
    alignment_info = alignment_summary(prep)
    metric_info = metric_summary(prep, selection, args.require_metric_scale)
    duplicate_info = duplicate_summary(
        state,
        presentation_count=(
            len(presentation_catalog) if presentation_catalog is not None else None
        ),
    )
    acceptance_ok = True
    acceptance_info: dict = {}
    if args.acceptance_report is not None:
        acceptance_ok, acceptance_info = acceptance_release_check(
            acceptance_payload,
            len(presentation_catalog or []),
        )
    if isinstance(frames_payload, Mapping):
        frames_count = _count_value(frames_payload.get("frames")) or 0
        frames_json_path = "$.frames"
    else:
        frames_count = _count_value(frames_payload) or 0
        frames_json_path = "$"
    checks = {
        "selection_nonempty": (
            selection_info["observations"] is not None
            and int(selection_info["observations"]) > 0
            and selection_info["views"] is not None
            and int(selection_info["views"]) > 0
        ),
        "colmap_selected_graph_connected": bool(selection_info["graph_connected"]),
        "rgbd_frames_present": frames_count > 0,
        "rgbd_alignment_passed": bool(alignment_info["passed"]),
        "metric_scale_passed": bool(metric_info["passed"]),
        "mapping_state_valid": bool(health["structurally_valid"]),
        "mapping_has_active_objects": bool(health["nonempty_active"]),
        "finite_scene_geometry": bool(health["finite_geometry"]),
        "up_policy_valid": up_policy_valid,
    }
    if args.acceptance_report is not None:
        checks["final_acceptance_release_safe"] = acceptance_ok
    if args.require_semantics:
        checks["caption_coverage_complete"] = len(records) > 0 and caption_coverage == len(records)
        checks["qwen_vl_embedding_coverage_complete"] = len(records) > 0 and qwen_coverage == len(records)
    status = "pass" if all(checks.values()) else "needs_review"
    pt_path = args.pt.expanduser().resolve()
    metric_source_file = (
        args.prep_summary if metric_info["source_owner"] == "prep" else args.selection_manifest
    )
    check_details = {
        "selection_nonempty": _metric(
            {"observations": selection_info["observations"], "views": selection_info["views"]},
            unit="count",
            source=_source(args.selection_manifest, selection_info["observation_path"]),
            definition="Selected observations and rendered views admitted to mapping.",
            threshold="observations > 0 and views > 0",
            status="pass" if checks["selection_nonempty"] else "fail",
        ),
        "colmap_selected_graph_connected": _metric(
            selection_info["residual_flagged_edges"], unit="edges",
            source=_source(args.selection_manifest, selection_info["residual_path"]),
            definition="Selected COLMAP graph edges that violate the configured connectivity gate.",
            threshold="residual_flagged_edges == 0",
            status="pass" if checks["colmap_selected_graph_connected"] else "fail",
        ),
        "rgbd_frames_present": _metric(
            frames_count, unit="frames", source=_source(args.frames_json, frames_json_path),
            definition="Prepared RGB-D frame records available to object mapping.",
            threshold="frames > 0", status="pass" if checks["rgbd_frames_present"] else "fail",
            sample_size=frames_count,
        ),
        "rgbd_alignment_passed": _metric(
            alignment_info["reported_passed"], unit=None,
            source=_source(args.prep_summary, alignment_info["source_path"]),
            definition="RGB-D/pose alignment QA, including finite configured thresholds and aggregate measurements.",
            threshold="alignment_qa.passed == true; thresholds and aggregate metrics present and finite",
            status="pass" if checks["rgbd_alignment_passed"] else "fail",
        ),
        "metric_scale_passed": _metric(
            metric_info["evidence"], unit="configured metric units",
            source=_source(metric_source_file, metric_info["source_path"]),
            definition="Observed evidence for conversion from reconstruction coordinates to metric scene coordinates.",
            threshold=("explicit pass and at least one finite measurement" if metric_info["required"] else "not required"),
            status="pass" if checks["metric_scale_passed"] else "fail",
        ),
        "mapping_state_valid": _metric(
            {"objects": health["object_count"], "reasons": health["reasons"]}, unit="objects",
            source=f"{pt_path}#$.state",
            definition="Required FARM mapping arrays exist and have mutually consistent shapes.",
            threshold="means[N,3], cov6[N,6], active/count/object_id[N]",
            status="pass" if checks["mapping_state_valid"] else "fail",
        ),
        "mapping_has_active_objects": _metric(
            health["active_count"], unit="objects", source=f"{pt_path}#$.state.active",
            definition="Objects retained as active in the mapping state.", threshold="active_count > 0",
            status="pass" if checks["mapping_has_active_objects"] else "fail",
        ),
        "finite_scene_geometry": _metric(
            {
                "nonfinite_means": health["nonfinite_means"],
                "nonfinite_cov6": health["nonfinite_cov6"],
                "nonfinite_counts": health["nonfinite_counts"],
            },
            unit="values", source=f"{pt_path}#$.state.means,cov6,count",
            definition="Non-finite values in object centres, covariance encoding and observation counts.",
            threshold="all counts == 0", status="pass" if checks["finite_scene_geometry"] else "fail",
        ),
        "up_policy_valid": _metric(
            up_policy.to_dict() if up_policy_valid else {"error": up_policy_reason}, unit=None,
            source=f"{pt_path}#$.state.object_geometry_up_*",
            definition="Exact normalized scene up direction used for ground-plane geometry and QA projection.",
            threshold="finite non-zero vector or explicit signed x/y/z axis",
            status="pass" if checks["up_policy_valid"] else "fail",
        ),
    }
    if args.require_semantics:
        check_details["caption_coverage_complete"] = _metric(
            caption_coverage, unit="objects", source=f"{pt_path}#$.state.object_caption",
            definition="Active objects with a non-empty caption.", threshold="captioned == active",
            status="pass" if checks["caption_coverage_complete"] else "fail",
            sample_size=len(records),
        )
        check_details["qwen_vl_embedding_coverage_complete"] = _metric(
            qwen_coverage, unit="objects", source=f"{pt_path}#$.state.object_qwen3_vl_embedding",
            definition="Active objects with a non-empty Qwen-VL embedding.", threshold="embedded == active",
            status="pass" if checks["qwen_vl_embedding_coverage_complete"] else "fail",
            sample_size=len(records),
        )
    if args.acceptance_report is not None:
        check_details["final_acceptance_release_safe"] = _metric(
            acceptance_info,
            unit=None,
            source=f"{args.acceptance_report.expanduser().resolve()}#$",
            definition=(
                "Model-free final release gate: no remaining automatic/blocking overlap "
                "clusters, no label hard errors, and exact exported presentation count."
            ),
            threshold="status in {PASS,WARN}; remaining counts == 0; presentation_after == catalog",
            status="pass" if acceptance_ok else "fail",
        )
    metric_index = {
        "qa_status": _metric(
            status, unit=None, source="derived:all(required quality checks)",
            definition="Overall scene QA decision.", threshold="every required check passes", status=status,
        ),
        "selected_views": _metric(
            selection_info["views"], unit="views",
            source=_source(args.selection_manifest, selection_info["views_path"]),
            definition="Pinhole views selected for mapping.", threshold="> 0",
        ),
        "active_objects": _metric(
            len(records), unit="objects", source=f"{pt_path}#$.state.active",
            definition="Finite active objects available to the report.", threshold="> 0",
        ),
        "presentation_visible_objects": _metric(
            duplicate_info.get("presentation_visible"), unit="objects",
            source=(
                f"{args.presentation_catalog.expanduser().resolve()}#$"
                if args.presentation_catalog is not None else
                f"{pt_path}#$.state.object_display_status,active,object_semantic_tier,object_geometry_status"
            ),
            definition="Non-suppressed metric objects plus validated probable compounds visible in the default viewer layer.",
        ),
        "probable_compound_objects": _metric(
            duplicate_info.get("probable_compounds"), unit="objects",
            source=f"{pt_path}#$.state.object_compound_boxes,object_geometry_status",
            definition="Inactive compound geometries retained for transparent presentation and audit.",
        ),
        "reliable_objects": _metric(
            len(reliable), unit="objects", source="derived:active objects with count >= 3",
            definition="Active objects supported by at least three observations.", threshold="observations >= 3",
            sample_size=len(records),
        ),
        "total_time_seconds": _metric(
            profile["total_seconds"], unit="seconds", source=profile["total_source"],
            definition="Measured end-to-end wall time; derived as stage sum only when no explicit total exists.",
        ),
        "device_peak_used_gib": _metric(
            profile["device_peak_used_gib"], unit="GiB",
            source=profile["device_peak_used_source"],
            definition=(
                "Maximum whole-device GPU memory used observed on the selected device; "
                "this is not attributed to the FARM process."
            ),
        ),
        "incremental_peak_delta_gib": _metric(
            profile["incremental_peak_delta_gib"], unit="GiB",
            source=profile["incremental_peak_delta_source"],
            definition=(
                "Maximum observed increase from the per-stage whole-device baseline; "
                "this is not attributed to the FARM process."
            ),
        ),
        "gpu_memory_measurement_scope": _metric(
            profile["memory_measurement_scope"], unit=None,
            source=profile["memory_measurement_scope_source"],
            definition="Scope of the supplied GPU-memory telemetry.",
        ),
        "duplicate_losers_resolved": _metric(
            duplicate_info["resolved_losers"], unit="objects",
            source=f"{pt_path}#{duplicate_info['source_path']}",
            definition="Unique loser object IDs attached to retained duplicate winners; unavailable if not recorded.",
        ),
        "acceptance_held_objects": _metric(
            (acceptance_info.get("counts") or {}).get("held_objects"),
            unit="objects",
            source=(
                f"{args.acceptance_report.expanduser().resolve()}#$.counts.held_objects"
                if args.acceptance_report is not None else None
            ),
            definition="Presentation-only duplicate objects hidden by the final acceptance pass.",
        ),
    }
    report = {
        "schema": "farm.scene-quality.v2",
        "scene_id": scene_id,
        "qa_status": status,
        "checks": checks,
        "check_details": check_details,
        "metrics": metric_index,
        "semantic_enrichment_required": bool(args.require_semantics),
        "input": {
            "frames": frames_count,
            "selected_observations": selection_info["observations"],
            "observation_source": selection_info["observation_source"],
            "timestamps": (
                selection_info["observations"]
                if selection_info["observation_source"] == "selected_timestamps" else None
            ),
            "selected_views": selection_info["views"],
            "registered_images_available": selection_info["registered_images"],
            "metric_scale": metric_info,
        },
        "scene": {
            "objects_total": health["object_count"],
            "objects_active_raw": len(records),
            "objects_inactive": health["inactive_count"],
            "objects_reliable_min_3_observations": len(reliable),
            "objects_robust_min_5_observations": len(stable),
            "captioned_active": caption_coverage,
            "qwen_vl_embedded_active": qwen_coverage,
            "siglip2_embedded_active": siglip_coverage,
            "regions_experimental": len(state.get("region_labels") or []),
            "up": up_policy.to_dict() if up_policy_valid else {"error": up_policy_reason},
            "mapping_health": health,
            "duplicate_resolution": duplicate_info,
        },
        "run_profile": profile,
        "final_acceptance": acceptance_info,
        "evidence_tiers": {tier: tier_counts[tier] for tier in TIER_ORDER},
        "categories_raw": dict(categories_raw.most_common()),
        "categories_reliable": dict(categories_reliable.most_common()),
        "caveats": [
            "Raw active objects include single-view detections by design; use reliable_objects_min3.json for downstream work.",
            "Region labels are experimental unless the scene configuration supplies a validated vocabulary.",
            (
                "GPU-memory metrics are whole-device observations on the selected GPU, "
                "not FARM-process allocations; unavailable values are never imputed as zero."
            ),
            (
                "SigLIP2 is auxiliary; Qwen3-VL embeddings and captions are required in this report."
                if args.require_semantics
                else "Semantic enrichment is intentionally deferred until after geometry and appearance filtering."
            ),
        ],
    }
    report = _json_safe(report)
    (output_dir / "quality_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "reliable_objects_min3.json").write_text(
        json.dumps(reliable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "robust_objects_min5.json").write_text(
        json.dumps(stable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    semantic_note = (
        "Semantic coverage is required by this report."
        if args.require_semantics
        else "Semantic enrichment is intentionally deferred until after geometry/appearance filtering."
    )
    table_rows = []
    for name, detail in check_details.items():
        observed = json.dumps(detail["value"], ensure_ascii=False)
        source = str(detail["source"]).replace("|", "\\|")
        threshold = str(detail.get("threshold") or "n/a").replace("|", "\\|")
        table_rows.append(
            f"| {name} | {detail['status'].upper()} | {observed} | {threshold} | {source} |"
        )
    notes = f"""# FARM scene result quality — {scene_id}

Overall status: **{status.upper()}**

- Input: {selection_info['observations']} selected observations / {selection_info['views']} views; {frames_count} prepared RGB-D records.
- Mapping state: {len(records)} finite active objects. {semantic_note}
- Presentation catalog: {duplicate_info.get('presentation_visible', 'unavailable')} objects (authoritative exported count).
- Final acceptance: {acceptance_info.get('status', 'unavailable')}; held {((acceptance_info.get('counts') or {}).get('held_objects', 'n/a'))}; open overlap/label hard errors {((acceptance_info.get('counts') or {}).get('remaining_auto_clusters', 'n/a'))}+{((acceptance_info.get('counts') or {}).get('remaining_blocking_clusters', 'n/a'))}/{((acceptance_info.get('counts') or {}).get('label_hard_errors', 'n/a'))}.
- Reliable catalog: {len(reliable)} objects with at least 3 observations.
- Conservative catalog: {len(stable)} objects with at least 5 observations.
- Total time: {_format_duration(profile['total_seconds'])}.
- GPU device peak used: {f"{profile['device_peak_used_gib']:.1f} GiB" if profile['device_peak_used_gib'] is not None else "unavailable"}.
- Incremental peak delta from stage baseline: {f"{profile['incremental_peak_delta_gib']:.1f} GiB" if profile['incremental_peak_delta_gib'] is not None else "unavailable"}.
- GPU-memory scope: `{profile['memory_measurement_scope']}` (not process-attributed).

| Check | Status | Observed | Threshold | Source |
|---|---:|---|---|---|
{chr(10).join(table_rows)}

The dashboard uses measured values only. Definitions and machine-readable sources are in `quality_index.json`.
"""
    (output_dir / "README.md").write_text(notes, encoding="utf-8")
    quality_index = {
        "schema": "farm.scene-quality-index.v1",
        "scene_id": scene_id,
        "qa_status": status,
        "metrics": metric_index,
        "checks": check_details,
        "stages": profile["stages"],
        "artifacts": {
            "summary": "quality_summary.json",
            "dashboard": "01_evidence_tier_dashboard_4k.jpg",
            "reliable_catalog": "reliable_objects_min3.json",
            "robust_catalog": "robust_objects_min5.json",
            "readme": "README.md",
        },
    }
    (output_dir / "quality_index.json").write_text(
        json.dumps(_json_safe(quality_index), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_dashboard(
        records,
        camera_positions(state),
        output_dir / "01_evidence_tier_dashboard_4k.jpg",
        scene_id=scene_id,
        up_policy=up_policy,
        checks=checks,
        selection_info=selection_info,
        alignment_info=alignment_info,
        run_info=profile,
        duplicate_info=duplicate_info,
        caption_coverage=caption_coverage,
        qwen_coverage=qwen_coverage,
        siglip_coverage=siglip_coverage,
        acceptance_info=acceptance_info,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
