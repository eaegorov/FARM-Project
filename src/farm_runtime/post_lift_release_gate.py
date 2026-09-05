"""Post-lift publication gates for FARM object masks and OBBs.

The Gaussian-lift held-out decision answers whether an instance assignment is
usable.  It does not, by itself, prove that the presentation OBB is tight, nor
does it prove that a proposed label is independently verified.  This module
keeps those three decisions separate and fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PostLiftPolicy:
    """Explicit, scene-agnostic thresholds used after dense Gaussian lifting."""

    minimum_build_timestamps: int = 3
    minimum_heldout_timestamps: int = 2
    minimum_good_heldout_timestamps: int = 2
    minimum_gaussians: int = 500
    minimum_median_iou: float = 0.60
    minimum_q25_iou: float = 0.45
    minimum_median_precision: float = 0.70
    minimum_median_recall: float = 0.65
    minimum_worst_precision: float = 0.35
    minimum_worst_recall: float = 0.40
    minimum_component_fraction: float = 0.85
    minimum_gaussian_inside_obb: float = 0.80
    minimum_obb_to_support_volume_ratio: float = 0.60
    maximum_obb_to_support_volume_ratio: float = 3.00
    maximum_normalized_support_center_offset: float = 0.45
    support_quantile: float = 0.01
    scene_support_quantile: float = 0.01
    scene_support_margin_fraction: float = 0.05
    minimum_gaussian_inside_scene_support: float = 0.80
    maximum_obb_center_distance_scene_diagonal: float = 0.02
    maximum_object_support_to_scene_diagonal: float = 0.60
    minimum_pairwise_obb_iou_for_review: float = 0.50
    minimum_pairwise_obb_containment_for_review: float = 0.65
    minimum_nested_obb_volume_ratio_for_review: float = 0.05
    relation_grid_resolution: int = 17

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_meters_per_scene_unit(value: object) -> float:
    """Validate the explicit conversion from source PLY XYZ to metres."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("meters_per_scene_unit must be a finite positive number")
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "meters_per_scene_unit must be a finite positive number"
        ) from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("meters_per_scene_unit must be a finite positive number")
    return scale


def scene_points_to_meters(points_scene: np.ndarray, scale: object) -> np.ndarray:
    """Convert PLY coordinates to metres without silently assuming unit scale."""

    metres_per_scene_unit = validate_meters_per_scene_unit(scale)
    points = np.asarray(points_scene, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("scene points must have shape (N, 3)")
    return points * metres_per_scene_unit


def verify_metric_scale_provenance(
    expected_scale: object, payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Verify every supported scale declaration in a provenance document."""

    expected = validate_meters_per_scene_unit(expected_scale)
    paths = (
        ("meters_per_scene_unit",),
        ("inputs", "meters_per_scene_unit"),
        ("policy", "meters_per_scene_unit"),
        ("metric_scale", "meters_per_scene_unit"),
        ("metric_scale_check", "meters_per_scene_unit"),
        ("resolved", "meters_per_scene_unit"),
    )
    declarations: list[dict[str, Any]] = []
    for keys in paths:
        current: object = payload
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            value = validate_meters_per_scene_unit(current)
            declarations.append({"json_path": "$." + ".".join(keys), "value": value})
    if not declarations:
        raise ValueError("metric scale provenance has no supported scale declaration")
    mismatches = [
        row
        for row in declarations
        if not np.isclose(float(row["value"]), expected, rtol=1.0e-9, atol=1.0e-12)
    ]
    if mismatches:
        rendered = ", ".join(
            f"{row['json_path']}={row['value']!r}" for row in mismatches
        )
        raise ValueError(
            f"metric scale provenance disagrees with explicit CLI scale {expected!r}: "
            f"{rendered}"
        )
    return declarations


def _summary_value(
    heldout: Mapping[str, Any], metric: str, statistic: str
) -> float | None:
    summaries = heldout.get("summaries")
    values = summaries.get(metric) if isinstance(summaries, Mapping) else None
    value = values.get(statistic) if isinstance(values, Mapping) else None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def quaternion_wxyz_to_matrix(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("OBB quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("OBB quaternion must be non-zero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _validated_rotation_matrix(value: object) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("OBB rotation matrix must contain nine finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("OBB rotation matrix must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("OBB rotation matrix must be right-handed")
    return rotation


def resolve_obb_geometry(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Resolve an OBB without guessing an orientation from object covariance.

    FARM has emitted both legacy and canonical field names. Quaternion order is
    accepted only when it is explicit in the field name; matrices must be
    finite, orthonormal and right-handed. Duplicate declarations must agree.
    ``cov6`` is deliberately not treated as OBB rotation: it describes support
    covariance and cannot safely recover the axis permutation of dimensions.
    """

    if not isinstance(record, Mapping):
        raise TypeError("OBB record must be a mapping")
    nested = record.get("obb")
    sources: list[tuple[str, Mapping[str, Any]]] = [("$", record)]
    if isinstance(nested, Mapping):
        sources.append(("$.obb", nested))

    def resolve_vector(
        aliases: Sequence[str], *, width: int, label: str
    ) -> tuple[np.ndarray, str]:
        values: list[tuple[str, np.ndarray]] = []
        for prefix, source in sources:
            for alias in aliases:
                if alias not in source or source[alias] is None:
                    continue
                value = np.asarray(source[alias], dtype=np.float64).reshape(-1)
                path = f"{prefix}.{alias}"
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise ValueError(
                        f"OBB {label} at {path} must contain {width} finite values"
                    )
                values.append((path, value))
        if not values:
            raise ValueError(f"OBB {label} is missing")
        path, value = values[0]
        for other_path, other in values[1:]:
            if not np.allclose(other, value, rtol=1.0e-6, atol=1.0e-8):
                raise ValueError(
                    f"conflicting OBB {label} declarations: {path} and {other_path}"
                )
        return value, path

    center, center_source = resolve_vector(
        ("center_m", "position_world_m", "box_center_m"),
        width=3,
        label="center",
    )
    dimensions, dimensions_source = resolve_vector(
        (
            "dimensions_m",
            "metric_dimensions_m",
            "dimensions_lwh_m",
            "box_dimensions_m",
        ),
        width=3,
        label="dimensions",
    )
    if np.any(dimensions <= 0.0):
        raise ValueError("OBB dimensions must be positive")

    rotations: list[tuple[str, np.ndarray]] = []
    for prefix, source in sources:
        for alias in ("wxyz", "orientation_wxyz", "quaternion_wxyz", "box_wxyz"):
            if alias in source and source[alias] is not None:
                path = f"{prefix}.{alias}"
                rotations.append((path, quaternion_wxyz_to_matrix(source[alias])))
        for alias in ("xyzw", "orientation_xyzw", "quaternion_xyzw", "box_xyzw"):
            if alias in source and source[alias] is not None:
                path = f"{prefix}.{alias}"
                xyzw = np.asarray(source[alias], dtype=np.float64).reshape(-1)
                if xyzw.shape != (4,) or not np.isfinite(xyzw).all():
                    raise ValueError(
                        f"OBB quaternion at {path} must contain four finite values"
                    )
                rotations.append(
                    (path, quaternion_wxyz_to_matrix(xyzw[[3, 0, 1, 2]]))
                )
        for alias in (
            "rotation_matrix",
            "rotation_world_from_obb",
            "obb_to_world_rotation_matrix",
        ):
            if alias in source and source[alias] is not None:
                path = f"{prefix}.{alias}"
                rotations.append((path, _validated_rotation_matrix(source[alias])))
    if not rotations:
        raise ValueError(
            "OBB rotation is missing; provide an explicit wxyz/xyzw or rotation matrix"
        )
    rotation_source, rotation = rotations[0]
    for other_source, other in rotations[1:]:
        if not np.allclose(other, rotation, rtol=1.0e-5, atol=1.0e-5):
            raise ValueError(
                "conflicting OBB rotation declarations: "
                f"{rotation_source} and {other_source}"
            )
    return center, dimensions, rotation, {
        "center": center_source,
        "dimensions": dimensions_source,
        "rotation": rotation_source,
    }


def gaussian_obb_support_metrics(
    points_world_m: np.ndarray,
    *,
    center_m: object,
    dimensions_m: object,
    wxyz: object | None = None,
    rotation_matrix: object | None = None,
    quantile: float,
) -> dict[str, Any]:
    """Measure tightness and centring in the published OBB coordinate frame."""

    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    center = np.asarray(center_m, dtype=np.float64).reshape(-1)
    dimensions = np.asarray(dimensions_m, dtype=np.float64).reshape(-1)
    if center.shape != (3,) or dimensions.shape != (3,):
        raise ValueError("OBB center and dimensions must be 3-vectors")
    if not np.isfinite(center).all() or not np.isfinite(dimensions).all():
        raise ValueError("OBB center and dimensions must be finite")
    if np.any(dimensions <= 0.0):
        raise ValueError("OBB dimensions must be positive")
    if not 0.0 <= float(quantile) < 0.5:
        raise ValueError("support quantile must lie in [0, 0.5)")
    if points.shape[0] < 8:
        return {
            "finite_gaussians": int(points.shape[0]),
            "gaussian_inside_obb_fraction": None,
            "robust_support_dimensions_m": None,
            "obb_to_support_volume_ratio": None,
            "normalized_support_center_offset": None,
        }

    if (wxyz is None) == (rotation_matrix is None):
        raise ValueError("provide exactly one OBB quaternion or rotation matrix")
    rotation = (
        quaternion_wxyz_to_matrix(wxyz)
        if wxyz is not None
        else _validated_rotation_matrix(rotation_matrix)
    )
    local = (points - center.reshape(1, 3)) @ rotation
    inside = np.all(np.abs(local) <= 0.5 * dimensions.reshape(1, 3) + 1.0e-6, axis=1)
    low = np.quantile(local, float(quantile), axis=0)
    high = np.quantile(local, 1.0 - float(quantile), axis=0)
    support_dimensions = np.maximum(high - low, 1.0e-6)
    support_center = 0.5 * (low + high)
    box_volume = float(np.prod(dimensions))
    support_volume = float(np.prod(support_dimensions))
    normalized_offset = float(
        np.linalg.norm(support_center / np.maximum(dimensions, 1.0e-6))
    )
    return {
        "finite_gaussians": int(points.shape[0]),
        "gaussian_inside_obb_fraction": float(np.mean(inside)),
        "robust_support_dimensions_m": support_dimensions.astype(float).tolist(),
        "obb_to_support_volume_ratio": box_volume / max(support_volume, 1.0e-12),
        "normalized_support_center_offset": normalized_offset,
    }


def scene_support_envelope(
    points_world_m: np.ndarray, *, quantile: float
) -> dict[str, Any]:
    """Build a robust, scene-adaptive envelope from the complete source PLY.

    The quantile envelope intentionally ignores a small tail of 3DGS floaters.
    It is not an object-size prior: it only provides a common coordinate frame
    for detecting masks/OBBs supported primarily by spatial outliers.
    """

    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not 0.0 <= float(quantile) < 0.5:
        raise ValueError("scene support quantile must lie in [0, 0.5)")
    if points.shape[0] < 8:
        raise ValueError("scene support requires at least eight finite Gaussians")
    low = np.quantile(points, float(quantile), axis=0)
    high = np.quantile(points, 1.0 - float(quantile), axis=0)
    dimensions = high - low
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 1.0e-9):
        raise ValueError("scene support envelope is degenerate")
    return {
        "finite_scene_gaussians": int(points.shape[0]),
        "quantile": float(quantile),
        "low_m": low.astype(float).tolist(),
        "high_m": high.astype(float).tolist(),
        "dimensions_m": dimensions.astype(float).tolist(),
        "diagonal_m": float(np.linalg.norm(dimensions)),
    }


def gaussian_scene_support_metrics(
    points_world_m: np.ndarray,
    *,
    center_m: object,
    robust_support_dimensions_m: object,
    scene_envelope: Mapping[str, Any],
    margin_fraction: float,
) -> dict[str, Any]:
    """Measure whether object support is spatially plausible in the full PLY."""

    if not 0.0 <= float(margin_fraction) <= 1.0:
        raise ValueError("scene support margin fraction must lie in [0, 1]")
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    center = np.asarray(center_m, dtype=np.float64).reshape(-1)
    low = np.asarray(scene_envelope.get("low_m"), dtype=np.float64).reshape(-1)
    high = np.asarray(scene_envelope.get("high_m"), dtype=np.float64).reshape(-1)
    dimensions = np.asarray(
        scene_envelope.get("dimensions_m"), dtype=np.float64
    ).reshape(-1)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("OBB center must contain three finite values")
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or dimensions.shape != (3,)
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or not np.isfinite(dimensions).all()
        or np.any(dimensions <= 0.0)
    ):
        raise ValueError("scene support envelope is invalid")
    diagonal = float(scene_envelope.get("diagonal_m") or 0.0)
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("scene support diagonal must be finite and positive")

    margin = dimensions * float(margin_fraction)
    expanded_low = low - margin
    expanded_high = high + margin
    inside = (
        np.all((points >= expanded_low) & (points <= expanded_high), axis=1)
        if points.size
        else np.zeros(0, dtype=bool)
    )
    delta = np.maximum(np.maximum(expanded_low - center, center - expanded_high), 0.0)
    support_dimensions = np.asarray(
        robust_support_dimensions_m, dtype=np.float64
    ).reshape(-1)
    support_diagonal = (
        float(np.linalg.norm(support_dimensions))
        if support_dimensions.shape == (3,)
        and np.isfinite(support_dimensions).all()
        and np.all(support_dimensions >= 0.0)
        else None
    )
    return {
        "finite_object_gaussians": int(points.shape[0]),
        "gaussian_inside_scene_support_fraction": (
            float(np.mean(inside)) if inside.size else None
        ),
        "obb_center_inside_scene_support": bool(np.all(delta == 0.0)),
        "obb_center_distance_scene_diagonal": float(np.linalg.norm(delta) / diagonal),
        "object_support_diagonal_m": support_diagonal,
        "object_support_to_scene_diagonal": (
            support_diagonal / diagonal if support_diagonal is not None else None
        ),
    }


def evaluate_scene_support_gate(
    support: Mapping[str, Any], *, policy: PostLiftPolicy
) -> tuple[bool, list[str]]:
    """Reject spatial floaters and implausibly scene-sized object masks."""

    inside = support.get("gaussian_inside_scene_support_fraction")
    center_inside = support.get("obb_center_inside_scene_support")
    center_distance = support.get("obb_center_distance_scene_diagonal")
    size_ratio = support.get("object_support_to_scene_diagonal")
    reasons: list[str] = []
    if (
        inside is None
        or not np.isfinite(float(inside))
        or float(inside) < policy.minimum_gaussian_inside_scene_support
    ):
        reasons.append("gaussian_support_is_scene_spatial_outlier")
    if center_inside is not True and (
        center_distance is None
        or not np.isfinite(float(center_distance))
        or float(center_distance) > policy.maximum_obb_center_distance_scene_diagonal
    ):
        reasons.append("obb_center_is_scene_spatial_outlier")
    if (
        size_ratio is None
        or not np.isfinite(float(size_ratio))
        or float(size_ratio) > policy.maximum_object_support_to_scene_diagonal
    ):
        reasons.append("object_support_spans_implausible_scene_fraction")
    return not reasons, reasons


def evaluate_lift_release_contract(
    result: Mapping[str, Any],
    build: Mapping[str, Any],
    heldout: Mapping[str, Any],
    gaussian_ids: np.ndarray,
    *,
    source_gaussian_count: int,
    artifacts_verified: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate global lift provenance and exclusive dense ownership.

    Per-object scores cannot override a non-release lift, an incomplete heldout
    pass, or a dense ID bank that disagrees with heldout verification.
    """

    labels = np.asarray(gaussian_ids)
    if labels.ndim != 1:
        raise ValueError("Gaussian ID array must be one-dimensional")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Gaussian ID array must have integer dtype")
    dense_ids, dense_counts = np.unique(labels[labels >= 0], return_counts=True)
    dense_count_by_id = {
        int(object_id): int(count)
        for object_id, count in zip(dense_ids, dense_counts, strict=True)
    }

    raw_heldout_objects = heldout.get("objects")
    raw_build_objects = build.get("objects")
    heldout_objects = (
        raw_heldout_objects if isinstance(raw_heldout_objects, list) else []
    )
    build_objects = raw_build_objects if isinstance(raw_build_objects, list) else []
    heldout_by_id: dict[int, Mapping[str, Any]] = {}
    duplicate_heldout_ids: set[int] = set()
    for row in heldout_objects:
        if not isinstance(row, Mapping) or "object_id" not in row:
            continue
        object_id = int(row["object_id"])
        if object_id in heldout_by_id:
            duplicate_heldout_ids.add(object_id)
        heldout_by_id[object_id] = row
    build_ids = {
        int(row["object_id"])
        for row in build_objects
        if isinstance(row, Mapping) and "object_id" in row
    }
    verified_ids = sorted(
        object_id
        for object_id, row in heldout_by_id.items()
        if str(row.get("status") or "").lower() == "verified"
    )
    dense_id_values = sorted(dense_count_by_id)
    count_mismatches = [
        {
            "object_id": object_id,
            "dense_gaussians": dense_count_by_id.get(object_id, 0),
            "reported_provisional_gaussians": int(
                (heldout_by_id.get(object_id) or {}).get("provisional_gaussians") or 0
            ),
        }
        for object_id in sorted(set(verified_ids) | set(dense_id_values))
        if dense_count_by_id.get(object_id, 0)
        != int((heldout_by_id.get(object_id) or {}).get("provisional_gaussians") or 0)
    ]
    summary = heldout.get("summary")
    summary_ids = (
        sorted(int(value) for value in summary.get("verified_object_ids", []))
        if isinstance(summary, Mapping)
        and isinstance(summary.get("verified_object_ids"), list)
        else []
    )

    def integer_equals(value: object, expected: int) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return False
        try:
            return int(value) == int(expected)
        except (TypeError, ValueError, OverflowError):
            return False

    result_inputs = result.get("inputs")
    result_contracts = result.get("contracts")
    quality_gate = result.get("quality_gate")
    result_counts = result.get("counts")
    result_qc = result.get("qc_summary")
    fit_contract = result.get("frozen_full_colmap_fit_contract")
    quality_integrity_checks = (
        ("gaussian_lift_status_not_pass", result.get("status") == "PASS"),
        (
            "gaussian_lift_quality_status_not_pass",
            result.get("quality_status") == "PASS",
        ),
        (
            "gaussian_lift_quality_gate_not_passed",
            isinstance(quality_gate, Mapping) and quality_gate.get("passed") is True,
        ),
        (
            "build_manifest_not_frozen",
            build.get("status") == "frozen_pending_heldout",
        ),
        ("heldout_manifest_not_complete", heldout.get("status") == "complete"),
        (
            "build_opened_heldout_masks",
            build.get("global_heldout_masks_opened") is False,
        ),
        (
            "build_not_source_order_preserving",
            build.get("source_order_preserved") is True,
        ),
        ("build_used_forced_fill", build.get("forced_fill") is False),
        (
            "heldout_timestamps_not_globally_disjoint",
            heldout.get("global_timestamp_disjoint") is True,
        ),
        (
            "heldout_rejections_not_returned_to_unknown",
            heldout.get("failed_objects_return_to_unknown") is True,
        ),
        (
            "heldout_runner_up_promotion_enabled",
            heldout.get("runner_up_promotion") is False,
        ),
        (
            "dense_labels_not_declared_source_ordered",
            isinstance(result_contracts, Mapping)
            and result_contracts.get("dense_arrays_in_source_ply_order") is True,
        ),
        (
            "dense_labels_not_verified_only",
            isinstance(result_contracts, Mapping)
            and result_contracts.get("verified_only_canonical_labels") is True,
        ),
        (
            "unknown_instance_id_contract_mismatch",
            isinstance(result_contracts, Mapping)
            and integer_equals(result_contracts.get("unknown_instance_id"), -1),
        ),
        (
            "source_ply_declared_mutated",
            isinstance(result_contracts, Mapping)
            and result_contracts.get("source_ply_mutated") is False,
        ),
        (
            "gaussians_declared_deleted",
            isinstance(result_contracts, Mapping)
            and integer_equals(result_contracts.get("gaussians_deleted"), 0),
        ),
        (
            "global_physical_timestamp_holdout_contract_missing",
            isinstance(result_contracts, Mapping)
            and result_contracts.get("global_physical_timestamp_holdout") is True,
        ),
        (
            "full_colmap_train_fit_contract_invalid",
            isinstance(fit_contract, Mapping)
            and fit_contract.get("schema") == "farm.full-colmap-train-fit.v1"
            and fit_contract.get("fit_splits") == ["train"]
            and fit_contract.get("heldout_consumed") is False
            and fit_contract.get("heldout_updates_candidate") is False
            and fit_contract.get("frozen_before_heldout_evaluation") is True
            and fit_contract.get("heldout_frames_opened_by_lift") is False,
        ),
        ("artifact_integrity_not_verified", artifacts_verified),
        (
            "dense_label_length_mismatch",
            labels.shape == (int(source_gaussian_count),),
        ),
        ("dense_labels_below_unknown_id", not bool(np.any(labels < -1))),
        ("duplicate_heldout_object_ids", not duplicate_heldout_ids),
        (
            "dense_ids_disagree_with_heldout_verified_ids",
            dense_id_values == verified_ids,
        ),
        (
            "heldout_summary_disagrees_with_verified_rows",
            summary_ids == verified_ids,
        ),
        (
            "verified_dense_id_missing_from_build",
            set(dense_id_values).issubset(build_ids),
        ),
        ("verified_dense_count_mismatch", not count_mismatches),
        (
            "source_gaussian_count_contract_mismatch",
            isinstance(result_inputs, Mapping)
            and integer_equals(
                result_inputs.get("source_gaussian_count"), source_gaussian_count
            )
            and integer_equals(
                build.get("source_gaussian_count"), source_gaussian_count
            ),
        ),
        (
            "result_dense_counts_mismatch",
            isinstance(result_counts, Mapping)
            and integer_equals(
                result_counts.get("source_gaussians"), source_gaussian_count
            )
            and integer_equals(
                result_counts.get("verified_gaussians"),
                int(np.count_nonzero(labels >= 0)),
            )
            and integer_equals(
                result_counts.get("unknown_gaussians"),
                int(np.count_nonzero(labels == -1)),
            )
            and integer_equals(
                result_counts.get("verified_objects"), len(verified_ids)
            ),
        ),
        (
            "result_qc_summary_mismatch",
            isinstance(result_qc, Mapping)
            and sorted(int(value) for value in result_qc.get("verified_object_ids", []))
            == verified_ids,
        ),
    )
    publication_checks = (
        (
            "gaussian_lift_not_release_eligible",
            result.get("release_eligible") is True,
        ),
    )
    quality_integrity_reasons = [
        reason for reason, passed in quality_integrity_checks if not passed
    ]
    publication_reasons = [
        reason for reason, passed in publication_checks if not passed
    ]
    reasons = [*quality_integrity_reasons, *publication_reasons]
    metrics = {
        "source_gaussians": int(source_gaussian_count),
        "dense_verified_gaussians": int(np.count_nonzero(labels >= 0)),
        "unknown_gaussians": int(np.count_nonzero(labels == -1)),
        "dense_single_owner_encoding": True,
        "dense_verified_object_ids": dense_id_values,
        "heldout_verified_object_ids": verified_ids,
        "heldout_summary_verified_object_ids": summary_ids,
        "heldout_blocked_object_ids": sorted(set(heldout_by_id) - set(verified_ids)),
        "unexpected_dense_object_ids": sorted(set(dense_id_values) - set(verified_ids)),
        "duplicate_heldout_object_ids": sorted(duplicate_heldout_ids),
        "verified_dense_count_mismatches": count_mismatches,
        "artifact_integrity_verified": bool(artifacts_verified),
        "quality_integrity_passed": not quality_integrity_reasons,
        "quality_integrity_reasons": quality_integrity_reasons,
        "publication_provenance_passed": not publication_reasons,
        "publication_provenance_reasons": publication_reasons,
    }
    return not reasons, reasons, metrics


def route_post_lift_object(
    *,
    diagnostic_upstream_passed: bool,
    publication_upstream_passed: bool,
    mask_passed: bool,
    obb_passed: bool,
    scene_support_passed: bool,
    relation_passed: bool,
    semantic_passed: bool,
) -> str:
    """Choose one queue while keeping experimental evidence non-publishable."""

    if publication_upstream_passed and not diagnostic_upstream_passed:
        raise ValueError("publication upstream cannot pass when diagnostics are blocked")
    if not diagnostic_upstream_passed:
        return "upstream_lift_blocked"
    if not mask_passed:
        return "mask_geometry_refinement"
    if not obb_passed:
        return "obb_refit"
    if not scene_support_passed:
        return "spatial_support_review"
    if not relation_passed:
        return "relation_verification"
    if not semantic_passed:
        return "semantic_verification"
    if publication_upstream_passed:
        return "production_ready"
    return "experimental_nonrelease_quality_verified"


def summarize_scene_release(
    rows: Sequence[Mapping[str, Any]], *, upstream_passed: bool
) -> dict[str, Any]:
    """Classify a complete scene audit while allowing a safe object subset."""

    object_ids = [int(row["object_id"]) for row in rows]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("scene release rows contain duplicate object IDs")
    published = sorted(
        int(row["object_id"]) for row in rows if row.get("route") == "production_ready"
    )
    blocked = sorted(set(object_ids) - set(published))
    subset_eligible = bool(upstream_passed) and bool(published)
    complete_eligible = subset_eligible and not blocked
    return {
        "status": (
            "PASS" if complete_eligible else "PARTIAL" if subset_eligible else "BLOCKED"
        ),
        "release_eligible": subset_eligible,
        "complete_scene_release_eligible": complete_eligible,
        "published_object_ids": published,
        "blocked_object_ids": blocked,
    }


def _obb_lattice_points(
    center: np.ndarray,
    dimensions: np.ndarray,
    rotation: np.ndarray,
    *,
    resolution: int,
) -> np.ndarray:
    if int(resolution) < 5:
        raise ValueError("relation grid resolution must be at least 5")
    axis = (np.arange(int(resolution), dtype=np.float64) + 0.5) / float(
        resolution
    ) - 0.5
    unit = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    local = unit * dimensions.reshape(1, 3)
    return local @ rotation.T + center.reshape(1, 3)


def _fraction_inside_obb(
    points_world_m: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
    rotation: np.ndarray,
) -> float:
    local = (points_world_m - center.reshape(1, 3)) @ rotation
    inside = np.all(
        np.abs(local) <= 0.5 * dimensions.reshape(1, 3) + 1.0e-9,
        axis=1,
    )
    return float(np.mean(inside))


def pairwise_obb_relation_metrics(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    grid_resolution: int,
) -> dict[str, float | bool]:
    """Estimate OBB intersection symmetrically for duplicate/nesting review.

    A deterministic lattice is sampled in both boxes. Keeping both directional
    estimates avoids missing a compact OBB nested in a much larger one. These
    metrics only route a pair to review; they never merge or suppress identity.
    """

    center_a, dimensions_a, rotation_a, _ = resolve_obb_geometry(first)
    center_b, dimensions_b, rotation_b, _ = resolve_obb_geometry(second)
    samples_a = _obb_lattice_points(
        center_a, dimensions_a, rotation_a, resolution=int(grid_resolution)
    )
    samples_b = _obb_lattice_points(
        center_b, dimensions_b, rotation_b, resolution=int(grid_resolution)
    )
    fraction_a_in_b = _fraction_inside_obb(
        samples_a, center_b, dimensions_b, rotation_b
    )
    fraction_b_in_a = _fraction_inside_obb(
        samples_b, center_a, dimensions_a, rotation_a
    )
    volume_a = float(np.prod(dimensions_a))
    volume_b = float(np.prod(dimensions_b))
    estimate_from_a = volume_a * fraction_a_in_b
    estimate_from_b = volume_b * fraction_b_in_a
    # A compact nested box can fall between all points of the large-box grid.
    # The larger directional estimate avoids that false negative. It remains a
    # diagnostic estimate and is never used for automatic identity mutation.
    intersection = min(max(estimate_from_a, estimate_from_b), min(volume_a, volume_b))
    union = max(volume_a + volume_b - intersection, 1.0e-12)
    volume_ratio = min(volume_a, volume_b) / max(volume_a, volume_b)
    return {
        "first_fraction_inside_second": fraction_a_in_b,
        "second_fraction_inside_first": fraction_b_in_a,
        "maximum_directional_containment": max(fraction_a_in_b, fraction_b_in_a),
        "approximate_intersection_volume_m3": intersection,
        "approximate_iou": float(np.clip(intersection / union, 0.0, 1.0)),
        "smaller_to_larger_volume_ratio": volume_ratio,
        "first_center_inside_second": bool(
            _fraction_inside_obb(
                center_a.reshape(1, 3), center_b, dimensions_b, rotation_b
            )
            == 1.0
        ),
        "second_center_inside_first": bool(
            _fraction_inside_obb(
                center_b.reshape(1, 3), center_a, dimensions_a, rotation_a
            )
            == 1.0
        ),
    }


def find_obb_relation_review_pairs(
    objects: Mapping[int, Mapping[str, Any]],
    eligible_object_ids: Sequence[int],
    *,
    policy: PostLiftPolicy,
) -> list[dict[str, Any]]:
    """Return strongly overlapping/nested OBB pairs without changing identity."""

    if not 0.0 <= policy.minimum_pairwise_obb_iou_for_review <= 1.0:
        raise ValueError("pairwise OBB IoU threshold must lie in [0, 1]")
    if not 0.0 <= policy.minimum_pairwise_obb_containment_for_review <= 1.0:
        raise ValueError("pairwise OBB containment threshold must lie in [0, 1]")
    if not 0.0 <= policy.minimum_nested_obb_volume_ratio_for_review <= 1.0:
        raise ValueError("nested OBB volume-ratio threshold must lie in [0, 1]")
    if int(policy.relation_grid_resolution) < 5:
        raise ValueError("relation grid resolution must be at least 5")

    ids = sorted({int(value) for value in eligible_object_ids})
    missing = [value for value in ids if value not in objects]
    if missing:
        raise ValueError(f"relation review references missing object IDs: {missing}")
    pairs: list[dict[str, Any]] = []
    for index, first_id in enumerate(ids):
        for second_id in ids[index + 1 :]:
            metrics = pairwise_obb_relation_metrics(
                objects[first_id],
                objects[second_id],
                grid_resolution=policy.relation_grid_resolution,
            )
            reasons: list[str] = []
            if float(metrics["approximate_iou"]) >= float(
                policy.minimum_pairwise_obb_iou_for_review
            ):
                reasons.append("strong_obb_overlap_possible_duplicate")
            if float(metrics["maximum_directional_containment"]) >= float(
                policy.minimum_pairwise_obb_containment_for_review
            ) and float(metrics["smaller_to_larger_volume_ratio"]) >= float(
                policy.minimum_nested_obb_volume_ratio_for_review
            ):
                reasons.append("nested_obb_possible_duplicate_or_part")
            if reasons:
                pairs.append(
                    {
                        "first_object_id": first_id,
                        "second_object_id": second_id,
                        "first_category": str(objects[first_id].get("category") or ""),
                        "second_category": str(
                            objects[second_id].get("category") or ""
                        ),
                        "reasons": reasons,
                        "metrics": metrics,
                    }
                )
    return pairs


def evaluate_mask_gate(
    heldout: Mapping[str, Any] | None,
    build: Mapping[str, Any] | None,
    *,
    heldout_schema: str,
    candidate_gaussian_count: int,
    final_gaussian_count: int,
    policy: PostLiftPolicy,
) -> tuple[bool, list[str], dict[str, Any]]:
    row = dict(heldout or {})
    build_row = dict(build or {})
    frozen_schema = heldout_schema == "farm.frozen-heldout-qc.v1"
    metrics = {
        "heldout_schema": heldout_schema,
        "heldout_status": str(row.get("status") or "missing").lower(),
        "build_geometry_status": str(
            build_row.get("geometry_gate") or "missing"
        ).upper(),
        "build_timestamps": int(build_row.get("build_timestamp_count") or 0),
        "heldout_timestamps": int(row.get("independent_physical_timestamps") or 0),
        "good_heldout_timestamps": int(row.get("good_timestamps") or 0),
        "build_provisional_gaussians": int(build_row.get("provisional_gaussians") or 0),
        "candidate_gaussians": int(candidate_gaussian_count),
        "final_gaussians": int(final_gaussian_count),
        "median_iou": _summary_value(row, "iou", "median"),
        "q25_iou": _summary_value(row, "iou", "q25"),
        "median_precision": _summary_value(row, "precision", "median"),
        "minimum_precision": _summary_value(row, "precision", "minimum"),
        "median_recall": _summary_value(row, "recall", "median"),
        "minimum_recall": _summary_value(row, "recall", "minimum"),
        "minimum_component_fraction": _summary_value(
            row, "largest_component_fraction", "minimum"
        ),
    }
    checks: list[tuple[str, bool]] = [
        ("authoritative_frozen_heldout_schema_required", frozen_schema),
        ("heldout_not_verified", metrics["heldout_status"] == "verified"),
        (
            "build_geometry_gate_not_passed",
            metrics["build_geometry_status"] == "PASS",
        ),
        (
            "insufficient_build_timestamps",
            metrics["build_timestamps"] >= policy.minimum_build_timestamps,
        ),
        (
            "insufficient_heldout_timestamps",
            metrics["heldout_timestamps"] >= policy.minimum_heldout_timestamps,
        ),
        (
            "insufficient_good_heldout_timestamps",
            metrics["good_heldout_timestamps"]
            >= policy.minimum_good_heldout_timestamps,
        ),
        (
            "insufficient_final_gaussians",
            metrics["final_gaussians"] >= policy.minimum_gaussians,
        ),
        (
            "candidate_gaussian_count_mismatch",
            metrics["candidate_gaussians"] == metrics["final_gaussians"],
        ),
        (
            "build_gaussian_count_mismatch",
            metrics["build_provisional_gaussians"] == metrics["final_gaussians"],
        ),
    ]
    threshold_checks = (
        ("median_iou_below_threshold", "median_iou", policy.minimum_median_iou),
        ("q25_iou_below_threshold", "q25_iou", policy.minimum_q25_iou),
        (
            "median_precision_below_threshold",
            "median_precision",
            policy.minimum_median_precision,
        ),
        (
            "median_recall_below_threshold",
            "median_recall",
            policy.minimum_median_recall,
        ),
        (
            "worst_precision_below_threshold",
            "minimum_precision",
            policy.minimum_worst_precision,
        ),
        (
            "worst_recall_below_threshold",
            "minimum_recall",
            policy.minimum_worst_recall,
        ),
        (
            "fragmented_heldout_render",
            "minimum_component_fraction",
            policy.minimum_component_fraction,
        ),
    )
    for reason, key, threshold in threshold_checks:
        value = metrics[key]
        checks.append((reason, value is not None and float(value) >= float(threshold)))
    reasons = [reason for reason, passed in checks if not passed]
    return not reasons, reasons, metrics


def evaluate_obb_gate(
    support: Mapping[str, Any], *, policy: PostLiftPolicy
) -> tuple[bool, list[str]]:
    inside = support.get("gaussian_inside_obb_fraction")
    volume_ratio = support.get("obb_to_support_volume_ratio")
    center_offset = support.get("normalized_support_center_offset")
    reasons: list[str] = []
    if inside is None or float(inside) < policy.minimum_gaussian_inside_obb:
        reasons.append("gaussian_support_outside_obb")
    if volume_ratio is None:
        reasons.append("obb_support_volume_unavailable")
    elif float(volume_ratio) < policy.minimum_obb_to_support_volume_ratio:
        reasons.append("obb_too_small_for_gaussian_support")
    elif float(volume_ratio) > policy.maximum_obb_to_support_volume_ratio:
        reasons.append("obb_too_large_for_gaussian_support")
    if (
        center_offset is None
        or float(center_offset) > policy.maximum_normalized_support_center_offset
    ):
        reasons.append("obb_off_center_from_gaussian_support")
    return not reasons, reasons
