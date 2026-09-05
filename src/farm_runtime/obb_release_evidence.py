"""Scene-agnostic train-evidence gate for publishing FARM metric OBBs.

The RGB-D fitter has deliberately permissive thresholds so that a provisional
box remains available to view-rescue and diagnostics.  Publication needs a
separate, stricter decision: enough independent physical moments, robust
reprojection support, and a stable orientation when yaw materially changes
the box.  A failed gate never authorizes expanding an OBB to match a partial
or occluded 2D proposal; it only routes the object to more evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EVIDENCE_SCHEMA = "farm.obb-train-release-evidence.v1"


@dataclass(frozen=True)
class OBBReleaseEvidencePolicy:
    """Conservative defaults for scene-independent OBB publication."""

    minimum_independent_physical_timestamps: int = 4
    minimum_timestamp_median_projected_box_iou: float = 0.50
    minimum_timestamp_q25_projected_box_iou: float = 0.40
    orientation_materiality_threshold: float = 0.25
    minimum_material_orientation_confidence: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def evaluate_obb_release_evidence(
    *,
    geometry_fit_passed: bool,
    independent_physical_timestamps: object,
    timestamp_median_projected_box_iou: object,
    timestamp_q25_projected_box_iou: object,
    orientation_materiality: object,
    orientation_confidence: object,
    policy: OBBReleaseEvidencePolicy | None = None,
) -> dict[str, Any]:
    """Return an auditable fail-closed release decision.

    Median and q25 are computed upstream after de-duplicating simultaneous
    cameras by physical timestamp.  The lower quartile tolerates an isolated
    partial/occluded observation when at least four moments are present, while
    rejecting a consistently marginal fit.  Low evidence never mutates the
    candidate OBB.
    """

    policy = policy or OBBReleaseEvidencePolicy()
    if int(policy.minimum_independent_physical_timestamps) < 2:
        raise ValueError("minimum independent physical timestamps must be at least two")
    unit_values = (
        policy.minimum_timestamp_median_projected_box_iou,
        policy.minimum_timestamp_q25_projected_box_iou,
        policy.orientation_materiality_threshold,
        policy.minimum_material_orientation_confidence,
    )
    if any(not 0.0 <= float(value) <= 1.0 for value in unit_values):
        raise ValueError("OBB release evidence thresholds must be in [0, 1]")

    try:
        independent = int(independent_physical_timestamps)
    except (TypeError, ValueError):
        independent = 0
    independent = max(independent, 0)
    median_iou = _finite_float(timestamp_median_projected_box_iou)
    q25_iou = _finite_float(timestamp_q25_projected_box_iou)
    materiality = _finite_float(orientation_materiality)
    confidence = _finite_float(orientation_confidence)
    orientation_required = bool(
        materiality is not None
        and materiality >= float(policy.orientation_materiality_threshold)
    )

    reasons: list[str] = []
    if geometry_fit_passed is not True:
        reasons.append("geometry_fit_gate_not_passed")
    if independent < int(policy.minimum_independent_physical_timestamps):
        reasons.append("insufficient_independent_physical_timestamps")
    if median_iou is None or median_iou < float(
        policy.minimum_timestamp_median_projected_box_iou
    ):
        reasons.append("timestamp_median_projected_box_iou_below_release_limit")
    if q25_iou is None or q25_iou < float(
        policy.minimum_timestamp_q25_projected_box_iou
    ):
        reasons.append("timestamp_q25_projected_box_iou_below_release_limit")
    if orientation_required and (
        confidence is None
        or confidence < float(policy.minimum_material_orientation_confidence)
    ):
        reasons.append("material_orientation_confidence_below_release_limit")

    evidence_reasons = {
        "insufficient_independent_physical_timestamps",
        "timestamp_median_projected_box_iou_below_release_limit",
        "timestamp_q25_projected_box_iou_below_release_limit",
        "material_orientation_confidence_below_release_limit",
    }
    route = (
        "release"
        if not reasons
        else (
            "view_rescue"
            if any(reason in evidence_reasons for reason in reasons)
            else "obb_refit"
        )
    )
    return {
        "schema": EVIDENCE_SCHEMA,
        "passed": not reasons,
        "route": route,
        "reasons": reasons,
        "geometry_fit_passed": geometry_fit_passed is True,
        "independent_physical_timestamps": independent,
        "timestamp_median_projected_box_iou": median_iou,
        "timestamp_q25_projected_box_iou": q25_iou,
        "orientation_materiality": materiality,
        "orientation_confidence": confidence,
        "orientation_required": orientation_required,
        "policy": policy.to_dict(),
        "contract": {
            "multiple_cameras_at_one_physical_timestamp_count_once": True,
            "robust_statistics_use_one_supported_view_per_timestamp": True,
            "partial_or_occluded_view_does_not_authorize_obb_expansion": True,
            "failed_release_evidence_preserves_provisional_fit_for_diagnostics": True,
        },
    }


def validate_serialized_obb_release_evidence(value: object) -> list[str]:
    """Validate a serialized pass claim and return deterministic failures."""

    if not isinstance(value, dict) or value.get("schema") != EVIDENCE_SCHEMA:
        return ["geometry_release_evidence_missing_or_invalid"]
    reasons = value.get("reasons")
    if (
        value.get("passed") is not True
        or value.get("route") != "release"
        or not isinstance(reasons, list)
        or reasons
        or value.get("geometry_fit_passed") is not True
    ):
        return ["geometry_release_evidence_not_passed"]
    policy = value.get("policy")
    contract = value.get("contract")
    if not isinstance(policy, dict) or not isinstance(contract, dict):
        return ["geometry_release_evidence_missing_or_invalid"]
    if any(
        contract.get(key) is not True
        for key in (
            "multiple_cameras_at_one_physical_timestamp_count_once",
            "robust_statistics_use_one_supported_view_per_timestamp",
            "partial_or_occluded_view_does_not_authorize_obb_expansion",
            "failed_release_evidence_preserves_provisional_fit_for_diagnostics",
        )
    ):
        return ["geometry_release_evidence_missing_or_invalid"]
    try:
        minimum = int(policy["minimum_independent_physical_timestamps"])
        independent = int(value["independent_physical_timestamps"])
        median_iou = float(value["timestamp_median_projected_box_iou"])
        q25_iou = float(value["timestamp_q25_projected_box_iou"])
        min_median = float(policy["minimum_timestamp_median_projected_box_iou"])
        min_q25 = float(policy["minimum_timestamp_q25_projected_box_iou"])
        materiality_threshold = float(policy["orientation_materiality_threshold"])
        min_confidence = float(policy["minimum_material_orientation_confidence"])
    except (KeyError, TypeError, ValueError):
        return ["geometry_release_evidence_missing_or_invalid"]
    materiality = _finite_float(value.get("orientation_materiality"))
    confidence = _finite_float(value.get("orientation_confidence"))
    orientation_required = bool(
        materiality is not None and materiality >= materiality_threshold
    )
    required = OBBReleaseEvidencePolicy()
    if (
        minimum < required.minimum_independent_physical_timestamps
        or min_median < required.minimum_timestamp_median_projected_box_iou
        or min_q25 < required.minimum_timestamp_q25_projected_box_iou
        or materiality_threshold > required.orientation_materiality_threshold
        or min_confidence < required.minimum_material_orientation_confidence
    ):
        return ["geometry_release_evidence_policy_weaker_than_required"]
    numbers = (
        median_iou,
        q25_iou,
        min_median,
        min_q25,
        materiality_threshold,
        min_confidence,
    )
    optional_orientation_numbers = (materiality, confidence)
    if (
        minimum < 2
        or independent < minimum
        or not all(np.isfinite(number) for number in numbers)
        or not all(0.0 <= number <= 1.0 for number in numbers)
        or any(
            number is not None and not 0.0 <= number <= 1.0
            for number in optional_orientation_numbers
        )
        or median_iou < min_median
        or q25_iou < min_q25
        or value.get("orientation_required") is not orientation_required
        or (
            orientation_required and (confidence is None or confidence < min_confidence)
        )
    ):
        return ["geometry_release_evidence_claim_contradicted"]
    return []
