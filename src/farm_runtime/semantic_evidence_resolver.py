"""Fail-closed resolution of targeted semantic evidence into one scene cohort."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from scene_graph.captioning.evidence import evidence_view_ids
from scene_graph.captioning.label_contract import (
    independent_blind_whole_identity_consensus,
    normalize_open_noun,
)


REVIEW_SCHEMA = "farm.open-vocabulary-crop-review.v2"
PLAN_SCHEMA = "farm.semantic-evidence-resolution-plan.v1"
RESULT_SCHEMA = "farm.semantic-evidence-resolved-cohort.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_event_pair(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact current raw blind pair; inherited or extra votes fail."""

    raw_events = row.get("semantic_evidence")
    if not isinstance(raw_events, list) or len(raw_events) != 2:
        raise ValueError("exactly_two_current_semantic_events_required")
    events = [dict(event) for event in raw_events if isinstance(event, Mapping)]
    if len(events) != 2:
        raise ValueError("semantic_event_not_an_object")
    initial = [
        event
        for event in events
        if str(event.get("source") or "")
        in {"initial_blind_review", "contextual_dual_panel_review"}
    ]
    verification = [
        event
        for event in events
        if str(event.get("source") or "") == "independent_verification"
    ]
    if len(initial) != 1 or len(verification) != 1:
        raise ValueError("exact_initial_and_verification_event_pair_required")
    return initial[0], verification[0]


def pair_fingerprints(row: Mapping[str, Any]) -> list[str]:
    initial, verification = semantic_event_pair(row)
    values = [
        str(initial.get("evidence_fingerprint_sha256") or ""),
        str(verification.get("evidence_fingerprint_sha256") or ""),
    ]
    if any(_SHA256_RE.fullmatch(value) is None for value in values):
        raise ValueError("semantic_event_fingerprint_missing")
    return values


def all_event_fingerprints(row: Mapping[str, Any]) -> list[str]:
    """Bind even an incomplete targeted audit before rejecting it fail-closed."""

    raw_events = row.get("semantic_evidence")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("at_least_one_semantic_event_required")
    values: list[str] = []
    for event in raw_events:
        if not isinstance(event, Mapping):
            raise ValueError("semantic_event_not_an_object")
        value = str(event.get("evidence_fingerprint_sha256") or "")
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("semantic_event_fingerprint_missing")
        values.append(value)
    return values


def assess_pair(row: Mapping[str, Any], *, minimum_confidence: float) -> dict[str, Any]:
    initial, verification = semantic_event_pair(row)
    return dict(independent_blind_whole_identity_consensus(
        initial,
        verification,
        minimum_confidence=float(minimum_confidence),
    ))


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _pair_strength(row: Mapping[str, Any], consensus: Mapping[str, Any]) -> dict[str, Any]:
    initial, verification = semantic_event_pair(row)
    independence = dict(consensus.get("evidence_independence") or {})
    pose = dict(independence.get("pose_diversity") or independence)
    contracts = [
        dict(initial.get("label_contract") or {}),
        dict(verification.get("label_contract") or {}),
    ]
    return {
        "accepted": consensus.get("accepted") is True,
        "category": normalize_open_noun(consensus.get("canonical_category")),
        "mask_scope_complete": consensus.get("mask_scope_incomplete") is not True,
        "minimum_confidence": min(
            (_finite(contract.get("confidence")) for contract in contracts),
            default=0.0,
        ),
        "minimum_event_view_count": min(
            (len(evidence_view_ids(event) or ()) for event in (initial, verification)),
            default=0,
        ),
        "total_unique_views": len(
            set(evidence_view_ids(initial) or ())
            | set(evidence_view_ids(verification) or ())
        ),
        "pose_angle_degrees": _finite(
            pose.get("symmetric_median_nearest_viewpoint_angle_degrees")
        ),
        "normalized_camera_baseline": _finite(
            pose.get("symmetric_median_nearest_normalized_baseline")
        ),
    }


def targeted_pair_is_stronger(
    base_row: Mapping[str, Any],
    targeted_row: Mapping[str, Any],
    *,
    minimum_confidence: float,
) -> tuple[bool, dict[str, Any]]:
    """Permit only an accepted targeted pair that cannot weaken accepted truth."""

    base_consensus = assess_pair(base_row, minimum_confidence=minimum_confidence)
    targeted_consensus = assess_pair(targeted_row, minimum_confidence=minimum_confidence)
    base_strength = _pair_strength(base_row, base_consensus)
    targeted_strength = _pair_strength(targeted_row, targeted_consensus)
    reasons: list[str] = []
    if targeted_consensus.get("accepted") is not True:
        reasons.append("targeted_raw_blind_consensus_not_accepted")
    if base_consensus.get("accepted") is True:
        # Targeted overlays are a rescue mechanism for unresolved cohort rows,
        # not an update channel for already accepted semantics. Replacing an
        # accepted base would need a new scene-wide base cohort with its own
        # provenance review; this also prevents an older replay from winning.
        reasons.append("accepted_base_consensus_locked")
        if base_strength["category"] != targeted_strength["category"]:
            reasons.append("accepted_base_category_cannot_be_overwritten")
        comparable = (
            "mask_scope_complete",
            "minimum_confidence",
            "minimum_event_view_count",
            "total_unique_views",
            "pose_angle_degrees",
            "normalized_camera_baseline",
        )
        weaker = [
            key for key in comparable
            if targeted_strength[key] < base_strength[key]
        ]
        if weaker:
            reasons.append("targeted_evidence_weaker:" + ",".join(weaker))
        if not weaker and not any(
            targeted_strength[key] > base_strength[key] for key in comparable
        ):
            reasons.append("targeted_evidence_not_strictly_stronger")
    return not reasons, {
        "accepted": not reasons,
        "reason_codes": reasons or ["targeted_evidence_strictly_stronger"],
        "base": base_strength,
        "targeted": targeted_strength,
        "targeted_consensus": targeted_consensus,
    }


def _load_catalog(path: Path, expected_sha256: str, expected_schema: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256_file(resolved) != str(expected_sha256).lower():
        raise ValueError(f"catalog_sha256_mismatch:{resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"catalog_schema_mismatch:{resolved}")
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise ValueError(f"catalog_objects_missing:{resolved}")
    ids = [int(row["id"]) for row in rows if isinstance(row, Mapping) and "id" in row]
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError(f"catalog_object_ids_invalid_or_duplicate:{resolved}")
    return payload


def resolve_plan(plan: Mapping[str, Any], *, plan_path: Path | None = None) -> dict[str, Any]:
    """Resolve a base cohort plus cryptographically pinned targeted overlays."""

    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("resolution_plan_schema_mismatch")
    base_spec = plan.get("base_catalog")
    if not isinstance(base_spec, Mapping):
        raise ValueError("base_catalog_spec_missing")
    base_path = Path(str(base_spec.get("path") or ""))
    base_schema = str(base_spec.get("schema") or "")
    if base_schema != REVIEW_SCHEMA:
        raise ValueError("base_catalog_schema_not_allowed")
    base = _load_catalog(base_path, str(base_spec.get("sha256") or ""), base_schema)
    resolved_rows = {int(row["id"]): copy.deepcopy(row) for row in base["objects"]}
    initial_base_object_count = len(resolved_rows)
    decisions: list[dict[str, Any]] = []
    seen_overlay_ids: set[int] = set()
    minimum_confidence = min(
        1.0,
        max(0.85, _finite(plan.get("minimum_identity_confidence"), 0.85)),
    )

    for raw_spec in plan.get("targeted_overlays") or []:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("targeted_overlay_spec_invalid")
        object_id = int(raw_spec.get("object_id"))
        if object_id in seen_overlay_ids:
            raise ValueError(f"duplicate_targeted_overlay_object_id:{object_id}")
        seen_overlay_ids.add(object_id)
        operation = str(raw_spec.get("operation") or "overlay")
        if operation not in {"overlay", "add"}:
            raise ValueError(f"targeted_operation_invalid:{object_id}")
        base_exists = object_id in resolved_rows
        if operation == "overlay" and not base_exists:
            raise ValueError(f"targeted_object_absent_from_base:{object_id}")
        if operation == "add" and (
            base_exists or raw_spec.get("base_object_absent") is not True
        ):
            raise ValueError(f"targeted_add_base_absence_not_proven:{object_id}")
        overlay_schema = str(raw_spec.get("schema") or "")
        if overlay_schema != REVIEW_SCHEMA:
            raise ValueError(f"targeted_catalog_schema_not_allowed:{object_id}")
        overlay_path = Path(str(raw_spec.get("path") or ""))
        overlay = _load_catalog(
            overlay_path, str(raw_spec.get("sha256") or ""), overlay_schema
        )
        overlay_rows = overlay["objects"]
        if len(overlay_rows) != 1 or int(overlay_rows[0]["id"]) != object_id:
            raise ValueError(f"targeted_catalog_scope_not_exact:{object_id}")
        base_row = resolved_rows.get(object_id)
        targeted_row = overlay_rows[0]
        base_fingerprints = pair_fingerprints(base_row) if base_row is not None else []
        targeted_fingerprints = all_event_fingerprints(targeted_row)
        if list(raw_spec.get("base_event_fingerprints_sha256") or []) != base_fingerprints:
            raise ValueError(f"base_event_fingerprint_binding_mismatch:{object_id}")
        if list(raw_spec.get("targeted_event_fingerprints_sha256") or []) != targeted_fingerprints:
            raise ValueError(f"targeted_event_fingerprint_binding_mismatch:{object_id}")

        if base_row is None:
            try:
                targeted_consensus = assess_pair(
                    targeted_row, minimum_confidence=minimum_confidence
                )
                targeted_strength = _pair_strength(targeted_row, targeted_consensus)
                accepted = targeted_consensus.get("accepted") is True
                comparison = {
                    "accepted": accepted,
                    "reason_codes": (
                        ["targeted_absent_object_exact_consensus_accepted"]
                        if accepted
                        else ["targeted_raw_blind_consensus_not_accepted"]
                    ),
                    "base": {"object_absent": True},
                    "targeted": targeted_strength,
                    "targeted_consensus": targeted_consensus,
                }
            except ValueError as exc:
                accepted = False
                comparison = {
                    "accepted": False,
                    "reason_codes": [f"targeted_contract_invalid:{exc}"],
                    "base": {"object_absent": True},
                    "targeted": {},
                    "targeted_consensus": {},
                }
        else:
            try:
                accepted, comparison = targeted_pair_is_stronger(
                    base_row, targeted_row, minimum_confidence=minimum_confidence
                )
            except ValueError as exc:
                base_consensus = assess_pair(
                    base_row, minimum_confidence=minimum_confidence
                )
                accepted = False
                comparison = {
                    "accepted": False,
                    "reason_codes": [f"targeted_contract_invalid:{exc}"],
                    "base": _pair_strength(base_row, base_consensus),
                    "targeted": {},
                    "targeted_consensus": {},
                }
        if not accepted:
            if base_row is not None and any(
                "contained_part" in reason
                for reason in (
                    list(comparison.get("reason_codes") or [])
                    + list(
                    (comparison.get("targeted_consensus") or {}).get(
                        "reason_codes"
                    ) or []
                    )
                )
            ):
                retained = copy.deepcopy(base_row)
                retained["semantic_resolution_review_routes"] = sorted(set(
                    list(retained.get("semantic_resolution_review_routes") or [])
                    + ["relationship_verification"]
                ))
                retained.setdefault(
                    "semantic_resolution_rejected_overlay_evidence", []
                ).append({
                    "targeted_catalog": str(overlay_path.resolve()),
                    "targeted_catalog_sha256": str(raw_spec["sha256"]),
                    "targeted_event_fingerprints_sha256": targeted_fingerprints,
                    "reason_codes": list(comparison.get("reason_codes") or []),
                    "targeted_consensus_reason_codes": list(
                        (comparison.get("targeted_consensus") or {}).get(
                            "reason_codes"
                        ) or []
                    ),
                })
                resolved_rows[object_id] = retained
            decisions.append({
                "object_id": object_id,
                "status": (
                    "base_retained" if base_row is not None
                    else "targeted_add_rejected"
                ),
                "source_catalog": (
                    str(base_path.resolve()) if base_row is not None
                    else str(overlay_path.resolve())
                ),
                "source_catalog_sha256": (
                    str(base_spec["sha256"]) if base_row is not None
                    else str(raw_spec["sha256"])
                ),
                "comparison": comparison,
            })
            continue

        consensus = dict(comparison["targeted_consensus"])
        selected = copy.deepcopy(base_row if base_row is not None else targeted_row)
        selected["semantic_evidence"] = copy.deepcopy(targeted_row["semantic_evidence"])
        selected.update({
            "review_category": str(consensus["canonical_category"]),
            "review_decision": "keep",
            "review_confidence": float(comparison["targeted"]["minimum_confidence"]),
            "review_gate_reason": "resolved_targeted_exact_blind_consensus",
            "independent_blind_consensus": consensus,
            "semantic_publication_action": "publish_independent_blind_consensus",
            "semantic_quarantined": False,
            "semantic_resolution_provenance": {
                "schema": RESULT_SCHEMA,
                "base_catalog_sha256": str(base_spec["sha256"]),
                "targeted_catalog": str(overlay_path.resolve()),
                "targeted_catalog_sha256": str(raw_spec["sha256"]),
                "targeted_object_id": object_id,
                "base_event_fingerprints_sha256": base_fingerprints,
                "selected_event_fingerprints_sha256": targeted_fingerprints,
                "comparison": comparison,
            },
        })
        resolved_rows[object_id] = selected
        decisions.append({
            "object_id": object_id,
            "status": (
                "targeted_overlay_selected" if base_row is not None
                else "targeted_object_added"
            ),
            "source_catalog": str(overlay_path.resolve()),
            "source_catalog_sha256": str(raw_spec["sha256"]),
            "comparison": comparison,
        })

    resolved_catalog = copy.deepcopy(base)
    resolved_catalog["objects"] = [resolved_rows[key] for key in sorted(resolved_rows)]
    resolved_catalog["semantic_resolution"] = {
        "schema": RESULT_SCHEMA,
        "plan": str(plan_path.resolve()) if plan_path else None,
        "base_catalog": str(base_path.resolve()),
        "base_catalog_sha256": str(base_spec["sha256"]),
        "base_object_count": initial_base_object_count,
        "resolved_object_count": len(resolved_rows),
        "targeted_overlay_count": len(decisions),
        "selected_overlay_count": sum(
            row["status"] in {
                "targeted_overlay_selected", "targeted_object_added",
            }
            for row in decisions
        ),
        "minimum_identity_confidence": minimum_confidence,
        "policy": {
            "exact_catalog_sha256_required": True,
            "exact_review_schema_required": True,
            "exact_single_object_targeted_scope_required": True,
            "exact_event_fingerprint_binding_required": True,
            "two_current_unconditioned_blind_events_required": True,
            "independent_disjoint_image_and_pose_gates_required": True,
            "accepted_base_cannot_be_weakened_or_renamed": True,
            "accepted_base_is_immutable_to_targeted_overlays": True,
            "absent_object_add_requires_explicit_absence_and_exact_consensus": True,
            "invalid_or_incomplete_targeted_pair_retains_base": True,
        },
        "decisions": decisions,
    }
    return {
        "schema": RESULT_SCHEMA,
        "base_catalog_sha256": str(base_spec["sha256"]),
        "object_count": len(resolved_rows),
        "decisions": decisions,
        "resolved_catalog": resolved_catalog,
    }


__all__ = [
    "PLAN_SCHEMA", "RESULT_SCHEMA", "REVIEW_SCHEMA", "all_event_fingerprints",
    "assess_pair", "pair_fingerprints", "resolve_plan", "semantic_event_pair",
    "sha256_file", "targeted_pair_is_stronger",
]
