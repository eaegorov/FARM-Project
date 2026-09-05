"""Fail-closed audit of pose-aware initial/verification crop folds."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


SCHEMA = "farm.semantic-crop-selection-audit.v1"


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def audit_selection_row(
    row: Mapping[str, Any],
    *,
    minimum_angle_degrees: float = 10.0,
    minimum_normalized_baseline: float = 0.1,
) -> dict[str, Any]:
    object_id = int(row["id"])
    initial = row.get("semantic_initial_evidence_selection")
    verification = row.get("semantic_verification_evidence_selection")
    reasons: list[str] = []
    if not isinstance(initial, Mapping) or not isinstance(verification, Mapping):
        return {
            "object_id": object_id,
            "status": "rejected",
            "qwen_ready": False,
            "reason_codes": ["selection_contract_missing"],
            "initial_image_ids": [],
            "verification_image_ids": [],
        }
    if initial.get("schema") != "farm.semantic-view-partitions.v1":
        reasons.append("initial_selection_schema_invalid")
    if verification.get("schema") != "farm.semantic-view-partitions.v1":
        reasons.append("verification_selection_schema_invalid")
    initial_diag = dict(initial.get("selection_diagnostics") or {})
    verification_diag = dict(verification.get("selection_diagnostics") or {})
    initial_joint = dict(initial.get("cross_fold_pose_selection") or {})
    verification_joint = dict(verification.get("cross_fold_pose_selection") or {})
    if initial_joint != verification_joint:
        reasons.append("cross_fold_contract_disagrees_between_partitions")
    if initial_joint.get("schema") != "farm.cross-fold-pose-selection.v1":
        reasons.append("cross_fold_schema_invalid")
    if initial_joint.get("status") != "selected":
        reasons.append("strict_joint_pose_selection_not_found")

    initial_ids = [
        int(value)
        for value in (
            initial_joint.get("selected_initial_image_ids")
            or initial.get("selected_image_ids")
            or []
        )
    ]
    verification_ids = [
        int(value)
        for value in (
            initial_joint.get("selected_verification_image_ids")
            or verification.get("selected_image_ids")
            or []
        )
    ]
    if not initial_ids or len(set(initial_ids)) != len(initial_ids):
        reasons.append("initial_image_ids_missing_or_duplicate")
    if not verification_ids or len(set(verification_ids)) != len(verification_ids):
        reasons.append("verification_image_ids_missing_or_duplicate")
    if initial.get("selected_image_ids_complete") is not True:
        reasons.append("initial_selected_image_ids_incomplete")
    if verification.get("selected_image_ids_complete") is not True:
        reasons.append("verification_selected_image_ids_incomplete")
    overlap = sorted(set(initial_ids) & set(verification_ids))
    if overlap:
        reasons.append("initial_verification_image_overlap")
    separation = dict(verification.get("blind_verification_separation") or {})
    if separation.get("status") != "disjoint" or separation.get("all_image_ids_parseable") is not True:
        reasons.append("disjoint_source_image_contract_not_proven")

    metric_rows = {
        "initial_within_fold": dict(initial_joint.get("initial_within_fold") or {}),
        "verification_within_fold": dict(initial_joint.get("verification_within_fold") or {}),
        "cross_fold": dict(initial_joint.get("cross_fold") or {}),
    }
    metrics: dict[str, dict[str, float | None]] = {}
    for name, values in metric_rows.items():
        angle_key = (
            "symmetric_median_nearest_viewpoint_angle_degrees"
            if name == "cross_fold"
            else "min_pairwise_viewpoint_angle_degrees"
        )
        baseline_key = (
            "symmetric_median_nearest_normalized_baseline"
            if name == "cross_fold"
            else "min_pairwise_normalized_camera_baseline"
        )
        angle = _finite(values.get(angle_key))
        baseline = _finite(values.get(baseline_key))
        metrics[name] = {
            "angle_degrees": angle,
            "normalized_camera_baseline": baseline,
        }
        if angle is None or angle < float(minimum_angle_degrees):
            reasons.append(f"{name}_viewpoint_angle_below_threshold")
        if baseline is None or baseline < float(minimum_normalized_baseline):
            reasons.append(f"{name}_camera_baseline_below_threshold")
    if metric_rows["cross_fold"].get("pose_independent") is not True:
        reasons.append("cross_fold_pose_independence_not_proven")
    if initial_diag.get("diversity_gate_met") is not True:
        reasons.append("initial_diversity_gate_not_met")
    if verification_diag.get("diversity_gate_met") is not True:
        reasons.append("verification_diversity_gate_not_met")
    if initial_diag.get("selected_pose_provenance_complete") is not True:
        reasons.append("initial_pose_provenance_incomplete")
    if verification_diag.get("selected_pose_provenance_complete") is not True:
        reasons.append("verification_pose_provenance_incomplete")

    reasons = sorted(set(reasons))
    return {
        "object_id": object_id,
        "status": "ready" if not reasons else "rejected",
        "qwen_ready": not reasons,
        "reason_codes": reasons or ["strict_disjoint_pose_aware_folds_ready"],
        "initial_image_ids": initial_ids,
        "verification_image_ids": verification_ids,
        "overlap_image_ids": overlap,
        "metrics": metrics,
        "thresholds": {
            "minimum_viewpoint_angle_degrees": float(minimum_angle_degrees),
            "minimum_normalized_camera_baseline": float(minimum_normalized_baseline),
        },
    }


def audit_selection_payload(
    payload: Mapping[str, Any],
    requested_object_ids: Sequence[int],
    *,
    minimum_angle_degrees: float = 10.0,
    minimum_normalized_baseline: float = 0.1,
) -> dict[str, Any]:
    if payload.get("schema") != "farm.open-vocabulary-crop-review.v2":
        raise ValueError("selection_catalog_schema_invalid")
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise ValueError("selection_catalog_objects_missing")
    by_id = {int(row["id"]): row for row in rows if isinstance(row, Mapping) and "id" in row}
    if len(by_id) != len(rows):
        raise ValueError("selection_catalog_object_ids_invalid_or_duplicate")
    results: list[dict[str, Any]] = []
    for object_id in sorted(set(map(int, requested_object_ids))):
        if object_id not in by_id:
            results.append({
                "object_id": object_id,
                "status": "missing_input",
                "qwen_ready": False,
                "reason_codes": ["object_absent_from_review_queue_or_dry_run"],
                "initial_image_ids": [],
                "verification_image_ids": [],
            })
            continue
        results.append(audit_selection_row(
            by_id[object_id],
            minimum_angle_degrees=minimum_angle_degrees,
            minimum_normalized_baseline=minimum_normalized_baseline,
        ))
    ready = [row["object_id"] for row in results if row["qwen_ready"]]
    return {
        "schema": SCHEMA,
        "requested_object_ids": sorted(set(map(int, requested_object_ids))),
        "object_count": len(results),
        "ready_count": len(ready),
        "ready_object_ids": ready,
        "blocked_object_ids": [
            row["object_id"] for row in results if not row["qwen_ready"]
        ],
        "policy": {
            "no_vlm_requests_performed": True,
            "exact_disjoint_image_ids_required": True,
            "within_fold_and_cross_fold_pose_gates_required": True,
            "candidate_conditioned_prompts_forbidden": True,
        },
        "objects": results,
    }


__all__ = ["SCHEMA", "audit_selection_payload", "audit_selection_row"]
