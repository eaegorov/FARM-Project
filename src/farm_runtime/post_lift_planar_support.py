"""Train-only planar support pruning for dense Gaussian instance masks.

Thin physical objects are a special case for Gaussian lifting.  Their source
Gaussians can occupy several depth layers even when RGB-D observations prove
that the object is a single surface.  Expanding the presentation OBB to cover
those layers turns reconstruction uncertainty into false physical volume.

This module instead proposes a conservative *label-only* ablation.  It keeps
the OBB unchanged and removes an object's Gaussian ownership only outside the
fixed consensus plane/slab proven by independent train RGB-D observations.
The proposal is rejected before rendering if it loses in-plane extent or
coverage.  A surviving proposal is still not accepted here: it must be frozen
and independently reverse-rendered on held-out views.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


PLANAR_EVIDENCE_SCHEMA = "farm.planar-rgbd-thickness-evidence.v1"
PLANAR_POLICY = "planar_rgbd_normal_axis_exception_v1"


@dataclass(frozen=True)
class PlanarSupportPolicy:
    """Scene-agnostic safeguards for a consensus-normal slab proposal."""

    minimum_independent_timestamps: int = 4
    minimum_planar_view_rate: float = 0.80
    maximum_normal_angle_q90_degrees: float = 10.0
    maximum_consensus_to_obb_axis_degrees: float = 15.0
    maximum_planar_dimension_ratio: float = 0.35
    support_quantile: float = 0.01
    minimum_excess_normal_support_ratio: float = 1.35
    minimum_gaussians: int = 500
    minimum_retained_gaussians: int = 500
    minimum_retained_fraction: float = 0.50
    maximum_pruned_fraction: float = 0.50
    minimum_material_pruned_fraction: float = 0.02
    minimum_in_plane_extent_retention: float = 0.85
    in_plane_grid_resolution: int = 12
    minimum_in_plane_cell_coverage: float = 0.90
    maximum_candidate_normal_support_ratio: float = 1.05
    minimum_original_obb_inside_fraction_for_no_followup: float = 0.80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_vector(value: object, *, width: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must contain {width} finite values")
    return result


def _rotation(value: object) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_matrix must contain nine finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("rotation_matrix must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("rotation_matrix must be right-handed")
    return rotation


def _robust_extents(points: np.ndarray, quantile: float) -> np.ndarray:
    low, high = np.quantile(points, [quantile, 1.0 - quantile], axis=0)
    return np.maximum(high - low, 1e-9)


def _in_plane_cell_coverage(
    local: np.ndarray,
    retained: np.ndarray,
    *,
    axes: tuple[int, int],
    quantile: float,
    resolution: int,
) -> tuple[float, int, int]:
    values = local[:, axes]
    low, high = np.quantile(values, [quantile, 1.0 - quantile], axis=0)
    extent = high - low
    if np.any(extent <= 1e-9):
        return 0.0, 0, 0
    cell = np.floor((values - low) / extent * int(resolution)).astype(np.int64)
    valid = np.all((cell >= 0) & (cell < int(resolution)), axis=1)
    occupied = {tuple(row) for row in cell[valid].tolist()}
    kept = {tuple(row) for row in cell[valid & retained].tolist()}
    return (
        float(len(kept) / len(occupied)) if occupied else 0.0,
        len(occupied),
        len(kept),
    )


def _safe_number(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def evaluate_planar_support_candidate(
    points_world_m: np.ndarray,
    *,
    center_m: object,
    dimensions_m: object,
    rotation_matrix: object,
    planar_evidence: Mapping[str, Any],
    policy: PlanarSupportPolicy = PlanarSupportPolicy(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a keep mask and an auditable train-only pruning decision.

    The slab normal comes only from accepted RGB-D planar evidence.  Gaussian
    points may decide whether the predeclared proposal is safe to *attempt*,
    but they never rotate or enlarge the physical OBB.  No held-out artifact is
    accepted by this API.
    """

    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        raise ValueError("points_world_m contains non-finite rows")
    center = _finite_vector(center_m, width=3, field="center_m")
    dimensions = _finite_vector(dimensions_m, width=3, field="dimensions_m")
    if np.any(dimensions <= 0.0):
        raise ValueError("dimensions_m must be positive")
    rotation = _rotation(rotation_matrix)
    quantile = float(policy.support_quantile)
    if not 0.0 <= quantile < 0.5:
        raise ValueError("support_quantile must lie in [0, 0.5)")
    if policy.in_plane_grid_resolution < 2:
        raise ValueError("in_plane_grid_resolution must be at least two")

    keep_all = np.ones(points.shape[0], dtype=bool)
    common = {
        "policy": policy.to_dict(),
        "input_gaussians": int(points.shape[0]),
        "obb_mutated": False,
        "ownership_promoted": False,
        "heldout_consumed": False,
        "heldout_validation_required": True,
    }
    applicability: list[str] = []
    if not isinstance(planar_evidence, Mapping):
        applicability.append("planar_evidence_missing")
        planar_evidence = {}
    if planar_evidence.get("schema") != PLANAR_EVIDENCE_SCHEMA:
        applicability.append("planar_evidence_schema_invalid")
    if planar_evidence.get("accepted") is not True:
        applicability.append("planar_evidence_not_accepted")
    if planar_evidence.get("policy") != PLANAR_POLICY:
        applicability.append("planar_policy_not_accepted")
    if list(planar_evidence.get("reasons") or []):
        applicability.append("planar_evidence_has_reasons")
    independent = int(planar_evidence.get("independent_physical_timestamps") or 0)
    if independent < policy.minimum_independent_timestamps:
        applicability.append("insufficient_independent_train_timestamps")
    planar_rate = _safe_number(planar_evidence.get("planar_view_rate"), 0.0)
    if planar_rate < policy.minimum_planar_view_rate:
        applicability.append("insufficient_planar_train_view_rate")
    normal_q90 = _safe_number(planar_evidence.get("normal_angle_q90_degrees"))
    if normal_q90 > policy.maximum_normal_angle_q90_degrees:
        applicability.append("unstable_train_plane_normals")
    try:
        normal_axis = int(planar_evidence.get("normal_axis"))
    except (TypeError, ValueError):
        normal_axis = -1
    if normal_axis not in (0, 1, 2):
        applicability.append("planar_normal_axis_invalid")
    if points.shape[0] < policy.minimum_gaussians:
        applicability.append("insufficient_gaussians")
    if applicability:
        return keep_all, {
            **common,
            "status": "not_applicable",
            "reasons": applicability,
            "recommended_route": "leave",
            "normal_axis": normal_axis,
        }

    consensus = _finite_vector(
        planar_evidence.get("consensus_normal"),
        width=3,
        field="planar_evidence.consensus_normal",
    )
    norm = float(np.linalg.norm(consensus))
    if norm <= 1e-9:
        raise ValueError("planar consensus normal must be non-zero")
    consensus /= norm
    alignment = np.abs(rotation.T @ consensus)
    resolved_axis = int(np.argmax(alignment))
    axis_angle = float(
        np.degrees(np.arccos(np.clip(alignment[normal_axis], 0.0, 1.0)))
    )
    if resolved_axis != normal_axis:
        return keep_all, {
            **common,
            "status": "rejected",
            "reasons": ["consensus_normal_disagrees_with_declared_axis"],
            "recommended_route": "obb_refit",
            "normal_axis": normal_axis,
            "resolved_consensus_axis": resolved_axis,
            "consensus_to_obb_axis_degrees": axis_angle,
        }

    in_plane_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    planar_ratio = float(dimensions[normal_axis] / np.min(dimensions[list(in_plane_axes)]))
    if planar_ratio > policy.maximum_planar_dimension_ratio:
        return keep_all, {
            **common,
            "status": "not_applicable",
            "reasons": ["physical_obb_is_not_planar"],
            "recommended_route": "leave",
            "normal_axis": normal_axis,
            "physical_planar_dimension_ratio": planar_ratio,
        }
    if axis_angle > policy.maximum_consensus_to_obb_axis_degrees:
        return keep_all, {
            **common,
            "status": "rejected",
            "reasons": ["consensus_normal_too_far_from_obb_axis"],
            "recommended_route": "obb_refit",
            "normal_axis": normal_axis,
            "consensus_to_obb_axis_degrees": axis_angle,
        }

    constrained = _safe_number(planar_evidence.get("constrained_thickness_m"))
    normal_dimension = float(dimensions[normal_axis])
    thickness_tolerance = max(1e-4, 1e-3 * normal_dimension)
    if not np.isfinite(constrained) or abs(constrained - normal_dimension) > thickness_tolerance:
        return keep_all, {
            **common,
            "status": "rejected",
            "reasons": ["planar_thickness_disagrees_with_physical_obb"],
            "recommended_route": "obb_refit",
            "normal_axis": normal_axis,
            "constrained_thickness_m": constrained if np.isfinite(constrained) else None,
            "physical_normal_dimension_m": normal_dimension,
        }

    local = (points - center.reshape(1, 3)) @ rotation
    original_extents = _robust_extents(local, quantile)
    normal_support_ratio = float(original_extents[normal_axis] / normal_dimension)
    if normal_support_ratio < policy.minimum_excess_normal_support_ratio:
        return keep_all, {
            **common,
            "status": "not_applicable",
            "reasons": ["normal_support_is_not_excessive"],
            "recommended_route": "leave",
            "normal_axis": normal_axis,
            "robust_support_dimensions_m": original_extents.tolist(),
            "normal_support_to_obb_ratio": normal_support_ratio,
        }

    # The center and thickness are fixed by train RGB-D geometry.  Gaussian
    # support cannot shift the slab toward a denser or more convenient layer.
    normal_distance = (points - center.reshape(1, 3)) @ consensus
    half = 0.5 * normal_dimension
    epsilon = max(1e-6, 1e-6 * normal_dimension)
    retained = np.abs(normal_distance) <= half + epsilon
    retained_count = int(np.count_nonzero(retained))
    retained_fraction = float(retained_count / max(points.shape[0], 1))
    pruned_fraction = 1.0 - retained_fraction
    candidate_extents = (
        _robust_extents(local[retained], quantile)
        if retained_count >= 2
        else np.zeros(3, dtype=np.float64)
    )
    in_plane_retention = (
        candidate_extents[list(in_plane_axes)]
        / np.maximum(original_extents[list(in_plane_axes)], 1e-9)
    )
    cell_coverage, occupied_cells, retained_cells = _in_plane_cell_coverage(
        local,
        retained,
        axes=in_plane_axes,
        quantile=quantile,
        resolution=policy.in_plane_grid_resolution,
    )
    candidate_normal_extent = (
        float(
            np.quantile(normal_distance[retained], 1.0 - quantile)
            - np.quantile(normal_distance[retained], quantile)
        )
        if retained_count >= 2
        else 0.0
    )
    inside_original_obb = np.all(
        np.abs(local) <= 0.5 * dimensions.reshape(1, 3) + 1e-6,
        axis=1,
    )
    candidate_obb_inside_fraction = (
        float(np.mean(inside_original_obb[retained])) if retained_count else 0.0
    )

    reasons: list[str] = []
    if retained_count < policy.minimum_retained_gaussians:
        reasons.append("insufficient_retained_gaussians")
    if retained_fraction < policy.minimum_retained_fraction:
        reasons.append("retained_fraction_below_limit")
    if pruned_fraction > policy.maximum_pruned_fraction:
        reasons.append("pruned_fraction_above_limit")
    if pruned_fraction < policy.minimum_material_pruned_fraction:
        reasons.append("pruning_is_not_material")
    if np.any(in_plane_retention < policy.minimum_in_plane_extent_retention):
        reasons.append("in_plane_extent_not_preserved")
    if cell_coverage < policy.minimum_in_plane_cell_coverage:
        reasons.append("in_plane_cell_coverage_not_preserved")
    if candidate_normal_extent > (
        policy.maximum_candidate_normal_support_ratio * normal_dimension
    ):
        reasons.append("candidate_normal_slab_exceeds_physical_thickness")

    accepted = not reasons
    followup = (
        "obb_refit_after_mask_qc"
        if accepted
        and candidate_obb_inside_fraction
        < policy.minimum_original_obb_inside_fraction_for_no_followup
        else "none"
    )
    return (retained if accepted else keep_all), {
        **common,
        "status": "candidate" if accepted else "rejected",
        "reasons": reasons,
        "recommended_route": "mask_geometry_refinement" if accepted else "obb_refit",
        "followup_route": followup,
        "normal_axis": normal_axis,
        "consensus_normal_world": consensus.tolist(),
        "consensus_to_obb_axis_degrees": axis_angle,
        "physical_planar_dimension_ratio": planar_ratio,
        "physical_normal_dimension_m": normal_dimension,
        "slab_center_m": center.tolist(),
        "slab_half_thickness_m": half,
        "robust_support_dimensions_m": original_extents.tolist(),
        "normal_support_to_obb_ratio": normal_support_ratio,
        "retained_gaussians": retained_count,
        "pruned_gaussians": int(points.shape[0] - retained_count),
        "retained_fraction": retained_fraction,
        "pruned_fraction": pruned_fraction,
        "candidate_robust_support_dimensions_in_obb_frame_m": candidate_extents.tolist(),
        "candidate_normal_support_in_consensus_frame_m": candidate_normal_extent,
        "in_plane_extent_retention": in_plane_retention.tolist(),
        "in_plane_occupied_cells": occupied_cells,
        "in_plane_retained_cells": retained_cells,
        "in_plane_cell_coverage": cell_coverage,
        "candidate_gaussian_inside_original_obb_fraction": candidate_obb_inside_fraction,
    }


def apply_pruning_to_labels(
    labels: np.ndarray,
    *,
    object_id: int,
    object_keep_mask: np.ndarray,
) -> np.ndarray:
    """Demote rejected object rows to unknown without ownership promotion."""

    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    selected = np.flatnonzero(values == int(object_id))
    keep = np.asarray(object_keep_mask, dtype=bool).reshape(-1)
    if keep.shape != (selected.size,):
        raise ValueError("object keep mask does not match assigned Gaussian count")
    result = values.copy()
    result[selected[~keep]] = -1
    if np.any((result != values) & (result != -1)):
        raise AssertionError("planar pruning may only demote ownership to unknown")
    return result
