"""Conservative physical-form vetoes and fail-closed visible-form recovery.

The confirmation guard never invents or replaces a label. After that guard
hard-rejects a noun, a separate recovery contract may retain at most a
probable conservative form when two blind, low-overlap view events agree.
Rules are category-family constraints and contain no scene or object IDs.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

from scene_graph.captioning.evidence import (
    evidence_event_id,
    evidence_pose_diversity,
    evidence_view_ids,
)
from scene_graph.captioning.label_contract import (
    assess_open_vocabulary_label,
    normalize_open_noun,
    shared_explicit_form_hypernym,
)


PHYSICAL_FORM_SCHEMA = "farm.semantic-physical-form.v1"
PHYSICAL_RECOVERY_SCHEMA = "farm.semantic-physical-form-recovery.v1"
PHYSICAL_GUARD_RECOVERY_SCHEMA = "farm.semantic-guard-visible-form-recovery.v1"
PHYSICAL_RECOVERY_MIN_CONFIDENCE = 0.85
PHYSICAL_GUARD_RECOVERY_MIN_CONFIDENCE = 0.90
PHYSICAL_RECOVERY_MIN_VIEWS_PER_EVENT = 3
PHYSICAL_RECOVERY_MIN_TOTAL_UNIQUE_VIEWS = 6
PHYSICAL_RECOVERY_MAX_VIEW_OVERLAP = 0.25

_PLANAR_FORMS = {
    "access cover",
    "cover plate",
    "manhole cover",
    "panel",
    "plate",
    "poster",
    "sheet",
    "sign",
    "signboard",
}
_HORIZONTAL_SUPPORT_FORMS = {
    "bench",
    "desk",
    "table",
    "workbench",
    "work table",
}
_BULKY_WHEELED_WHOLES = {
    "car",
    "construction vehicle",
    "forklift",
    "grain harvester",
    "harvester",
    "scissor lift",
    "truck",
    "utility vehicle",
    "vehicle",
}
_HARD_MODEL_VERDICTS = {"contradiction", "part_only"}
_UNINFORMATIVE_RECOVERY_NOUNS = {
    "",
    "component",
    "fixture",
    "item",
    "object",
    "part",
    "thing",
    "unknown",
    "unresolved object",
}
_NEUTRAL_VISIBLE_FORM_HEADS = {
    "assembly",
    "bar",
    "beam",
    "block",
    "box",
    "cabinet",
    "cage",
    "column",
    "container",
    "cover",
    "cylinder",
    "disc",
    "disk",
    "enclosure",
    "frame",
    "grate",
    "grating",
    "grid",
    "housing",
    "lattice",
    "panel",
    "pipe",
    "platform",
    "plate",
    "rail",
    "rod",
    "screen",
    "sheet",
    "shell",
    "slab",
    "structure",
    "surface",
    "tube",
}
_NEUTRAL_VISIBLE_FORM_MODIFIERS = {
    "bounded",
    "circular",
    "cylindrical",
    "flat",
    "framed",
    "horizontal",
    "lattice",
    "low",
    "open",
    "rectangular",
    "ribbed",
    "slatted",
    "solid",
    "tubular",
    "vertical",
    "wheeled",
} | _NEUTRAL_VISIBLE_FORM_HEADS


def normalize_physical_category(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _category_tokens(value: object) -> tuple[str, ...]:
    normalized = normalize_physical_category(value)
    return tuple(
        word[:-1] if word.endswith("s") and len(word) > 4 else word
        for word in normalized.split()
    )


def _neutral_visible_form(value: object) -> str:
    """Project an open noun onto construction/geometry, never material/function."""

    tokens = _category_tokens(value)
    if not tokens or tokens[-1] not in _NEUTRAL_VISIBLE_FORM_HEADS:
        return ""
    retained = [
        token for token in tokens[:-1]
        if token in _NEUTRAL_VISIBLE_FORM_MODIFIERS
    ] + [tokens[-1]]
    return " ".join(dict.fromkeys(retained))


def _lexical_category_agreement(left: object, right: object) -> bool:
    left_tokens, right_tokens = _category_tokens(left), _category_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return bool(
        left_tokens == right_tokens
        or set(left_tokens) <= set(right_tokens)
        or set(right_tokens) <= set(left_tokens)
    )


def _conservative_category_agreement(left: object, right: object) -> str:
    """Return a conservative shared visible-form noun, or an empty string."""

    left_tokens = _category_tokens(_neutral_visible_form(left))
    right_tokens = _category_tokens(_neutral_visible_form(right))
    if not left_tokens or not right_tokens:
        return ""
    if left_tokens == right_tokens:
        return " ".join(left_tokens)
    left_set, right_set = set(left_tokens), set(right_tokens)
    if left_set <= right_set:
        return " ".join(left_tokens)
    if right_set <= left_set:
        return " ".join(right_tokens)
    if left_tokens[-1] == right_tokens[-1]:
        return left_tokens[-1]
    return ""


def _recovery_candidate_assessment(
    event: Mapping[str, Any] | None,
    *,
    rejected_category: object,
    minimum_confidence: float,
    prefix: str,
) -> dict[str, Any]:
    payload = dict(event or {})
    if isinstance(payload.get("label_contract"), Mapping):
        strict = assess_open_vocabulary_label(
            payload.get("label_contract"), minimum_confidence=minimum_confidence
        )
        category = normalize_open_noun(strict.get("category"))
        original = normalize_open_noun(rejected_category)
        reasons = [] if strict.get("usable") else [
            f"{prefix}_{reason}"
            for reason in strict.get("reason_codes") or ["contract_invalid"]
        ]
        if original and category == original:
            reasons.append(f"{prefix}_repeats_rejected_category")
        topology = str(strict.get("topology") or "unknown")
        standalone = topology in {"standalone_whole", "carrier_payload"}
        component_only = topology in {
            "standalone_component", "attached_component", "partial_unbounded"
        }
        return {
            "valid": not reasons,
            "category": category,
            "confidence": float(strict.get("confidence") or 0.0),
            "decision": "keep" if strict.get("usable") else "unknown",
            "standalone_bounded": standalone,
            "component_only": component_only,
            "event_id": evidence_event_id(payload),
            "view_ids": list(evidence_view_ids(payload) or []),
            "reason_codes": sorted(set(reasons)),
            "label_contract": dict(strict.get("contract") or {}),
        }
    category = normalize_physical_category(
        payload.get("category") or payload.get("review_category")
    )
    original = normalize_physical_category(rejected_category)
    decision = str(
        payload.get("decision") or payload.get("review_decision") or "unknown"
    ).strip().lower()
    try:
        confidence = float(
            payload.get("confidence") or payload.get("review_confidence") or 0.0
        )
    except (TypeError, ValueError):
        confidence = 0.0
    attributes = {
        normalize_physical_category(value).replace(" ", "_")
        for value in (
            payload.get("diagnostic_attributes")
            or payload.get("attributes")
            or payload.get("review_attributes")
            or []
        )
    }
    standalone = bool(
        payload.get("standalone_bounded") is True
        or "standalone_bounded" in attributes
    )
    component_only = bool(
        payload.get("component_only") is True or "component_only" in attributes
    )
    reasons: list[str] = []
    if decision != "keep":
        reasons.append(f"{prefix}_decision_not_keep")
    if confidence < float(minimum_confidence):
        reasons.append(f"{prefix}_confidence_below_threshold")
    if category in _UNINFORMATIVE_RECOVERY_NOUNS:
        reasons.append(f"{prefix}_noun_not_concrete")
    if original and _lexical_category_agreement(category, original):
        reasons.append(f"{prefix}_repeats_rejected_category")
    if not standalone:
        reasons.append(f"{prefix}_not_standalone_bounded")
    if component_only:
        reasons.append(f"{prefix}_component_only")
    return {
        "valid": not reasons,
        "category": category,
        "confidence": confidence,
        "decision": decision,
        "standalone_bounded": standalone,
        "component_only": component_only,
        "event_id": evidence_event_id(payload),
        "view_ids": list(evidence_view_ids(payload) or []),
        "reason_codes": reasons,
    }


def _metric_dimensions(value: object) -> np.ndarray | None:
    try:
        dimensions = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
        return None
    return dimensions


def assess_physical_form(
    category: object,
    metric_dimensions_m: object,
    *,
    model_guard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed, JSON-serializable confirmation assessment."""

    normalized = normalize_physical_category(category)
    dimensions = _metric_dimensions(metric_dimensions_m)
    reasons: list[str] = []
    hard_veto = False
    verdict = "compatible"

    metrics: dict[str, float | list[float]] = {}
    if dimensions is None:
        verdict = "insufficient"
        reasons.append("missing_or_invalid_metric_dimensions")
    else:
        ordered = np.sort(dimensions)
        min_to_median = float(ordered[0] / ordered[1])
        min_to_max = float(ordered[0] / ordered[2])
        height_to_horizontal = float(dimensions[2] / max(dimensions[0], dimensions[1]))
        metrics = {
            "dimensions_m": [float(value) for value in dimensions],
            "min_to_median": min_to_median,
            "min_to_max": min_to_max,
            "height_to_max_horizontal": height_to_horizontal,
        }

        if normalized in _PLANAR_FORMS and min_to_median > 0.55:
            hard_veto = True
            verdict = "contradiction"
            reasons.append("planar_noun_has_strongly_volumetric_metric_shape")
        if (
            normalized in _HORIZONTAL_SUPPORT_FORMS
            and height_to_horizontal > 1.45
            and float(dimensions[2]) > 0.80
        ):
            hard_veto = True
            verdict = "contradiction"
            reasons.append("horizontal_support_noun_has_dominant_vertical_extent")
        if (
            normalized in _BULKY_WHEELED_WHOLES
            and min_to_max < 0.12
            and float(ordered[2]) > 0.80
        ):
            hard_veto = True
            verdict = "part_only"
            reasons.append("bulky_wheeled_whole_has_thin_component_shape")

    guard_payload = dict(model_guard or {})
    guard_present = bool(guard_payload)
    proposed_category = normalize_physical_category(
        guard_payload.get("proposed_category")
    )
    guard_verdict = normalize_physical_category(
        guard_payload.get("verdict") or guard_payload.get("category")
    ).replace(" ", "_")
    try:
        guard_confidence = float(guard_payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        guard_confidence = 0.0
    try:
        supported_views = int(guard_payload.get("supported_view_count") or 0)
    except (TypeError, ValueError):
        supported_views = 0
    if guard_present and proposed_category and proposed_category != normalized:
        guard_verdict = "insufficient"
        verdict = "insufficient"
        reasons.append("paired_visual_guard_category_mismatch")
    elif (
        guard_verdict in _HARD_MODEL_VERDICTS
        and guard_confidence >= 0.90
        and supported_views >= 2
    ):
        hard_veto = True
        verdict = guard_verdict
        reasons.append(f"paired_visual_guard_{guard_verdict}")
    elif (
        guard_verdict == "compatible"
        and guard_confidence >= 0.80
        and supported_views >= 2
        and not hard_veto
    ):
        verdict = "compatible"
    elif not hard_veto:
        verdict = "insufficient"
        reasons.append(
            "paired_visual_guard_insufficient"
            if guard_present
            else "paired_visual_guard_missing"
        )

    return {
        "schema": PHYSICAL_FORM_SCHEMA,
        "category": normalized,
        "verdict": verdict,
        "hard_veto": bool(hard_veto),
        "confirmation_allowed": bool(
            not hard_veto
            and dimensions is not None
            and guard_present
            and guard_verdict == "compatible"
            and guard_confidence >= 0.80
            and supported_views >= 2
            and (not proposed_category or proposed_category == normalized)
        ),
        "reason_codes": sorted(set(reasons)),
        "metrics": metrics,
        "model_guard": guard_payload,
    }


def assess_guard_visible_form_recovery(
    rejected_assessment: Mapping[str, Any] | None,
    metric_dimensions_m: object,
    *,
    minimum_confidence: float = PHYSICAL_GUARD_RECOVERY_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Return a diagnostic-only result for the candidate-conditioned guard.

    A veto request has seen the rejected noun.  It can reject or withhold a
    tier, but it can never originate a replacement noun.  Blind recovery, when
    explicitly enabled, is handled by :func:`assess_physical_recovery`.
    """

    rejected = dict(rejected_assessment or {})
    guard = dict(rejected.get("model_guard") or {})
    trigger = bool(
        rejected.get("hard_veto")
        and str(rejected.get("verdict") or "") in _HARD_MODEL_VERDICTS
    )
    guard_verdict = normalize_physical_category(guard.get("verdict")).replace(
        " ", "_"
    )
    topology = normalize_physical_category(guard.get("topology")).replace(" ", "_")
    complete_bounded = guard.get("complete_bounded") is True
    reasons: list[str] = ["candidate_conditioned_guard_veto_only"]
    if not trigger:
        reasons.append("guard_recovery_not_triggered_without_hard_veto")
    if guard_verdict not in _HARD_MODEL_VERDICTS:
        reasons.append("guard_recovery_guard_verdict_not_hard_veto")
    reasons = sorted(set(reasons))
    return {
        "schema": PHYSICAL_GUARD_RECOVERY_SCHEMA,
        "triggered": trigger,
        "accepted": False,
        "action": "suppress_geometry_only",
        "category": "unresolved object",
        "description": "",
        "confidence": 0.0,
        "topology": topology or "unknown",
        "complete_bounded": complete_bounded,
        "maximum_semantic_tier": "geometry_only",
        "confirmation_eligible": False,
        "metric_physical_assessment": assess_physical_form(
            "", metric_dimensions_m
        ),
        "reason_codes": reasons,
    }


def assess_physical_recovery(
    recovery: Mapping[str, Any] | None,
    rejected_assessment: Mapping[str, Any] | None,
    *,
    verification: Mapping[str, Any] | None = None,
    rejected_category: object = "",
    metric_dimensions_m: object = None,
    minimum_confidence: float = PHYSICAL_RECOVERY_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Accept only an unconditioned, bounded neutral-form recovery.

    A recovery is never confirmation evidence.  This helper only decides
    whether a hard-rejected noun may be replaced by a conservative probable
    visible form or must be hidden as geometry-only.
    """

    rejected = dict(rejected_assessment or {})
    trigger = bool(
        rejected.get("hard_veto")
        and str(rejected.get("verdict") or "") in _HARD_MODEL_VERDICTS
    )
    primary = _recovery_candidate_assessment(
        recovery,
        rejected_category=rejected_category,
        minimum_confidence=minimum_confidence,
        prefix="recovery_primary",
    )
    secondary = _recovery_candidate_assessment(
        verification,
        rejected_category=rejected_category,
        minimum_confidence=minimum_confidence,
        prefix="recovery_verification",
    )
    primary_contract = primary.get("label_contract") or {}
    secondary_contract = secondary.get("label_contract") or {}
    if primary_contract and secondary_contract:
        category = (
            primary["category"]
            if normalize_open_noun(primary["category"])
            == normalize_open_noun(secondary["category"])
            else shared_explicit_form_hypernym(
                primary_contract, secondary_contract
            )
        )
    else:
        category = _conservative_category_agreement(
            primary["category"], secondary["category"]
        )
    reasons: list[str] = []
    if not trigger:
        reasons.append("recovery_not_triggered_without_hard_physical_veto")
    reasons.extend(primary["reason_codes"])
    reasons.extend(secondary["reason_codes"])
    if primary["valid"] and secondary["valid"] and not category:
        reasons.append("recovery_visible_form_disagreement")
    if category in _UNINFORMATIVE_RECOVERY_NOUNS:
        reasons.append("recovery_agreed_noun_not_concrete")
    if primary["valid"] and secondary["valid"] and not category:
        reasons.append("recovery_agreed_noun_not_conservative_visible_form")

    primary_views = set(primary["view_ids"])
    secondary_views = set(secondary["view_ids"])
    unique_views = sorted(primary_views | secondary_views)
    overlap: float | None = None
    if not primary_views:
        reasons.append("recovery_primary_view_provenance_incomplete")
    elif len(primary_views) < PHYSICAL_RECOVERY_MIN_VIEWS_PER_EVENT:
        reasons.append("recovery_primary_insufficient_unique_views")
    if not secondary_views:
        reasons.append("recovery_verification_view_provenance_incomplete")
    elif len(secondary_views) < PHYSICAL_RECOVERY_MIN_VIEWS_PER_EVENT:
        reasons.append("recovery_verification_insufficient_unique_views")
    if primary_views and secondary_views:
        overlap = len(primary_views & secondary_views) / min(
            len(primary_views), len(secondary_views)
        )
        if overlap > PHYSICAL_RECOVERY_MAX_VIEW_OVERLAP:
            reasons.append("recovery_view_overlap_above_threshold")
    if len(unique_views) < PHYSICAL_RECOVERY_MIN_TOTAL_UNIQUE_VIEWS:
        reasons.append("recovery_insufficient_total_unique_views")
    if not primary["event_id"] or not secondary["event_id"]:
        reasons.append("recovery_missing_event_id")
    elif primary["event_id"] == secondary["event_id"]:
        reasons.append("recovery_same_inference_event")
    primary_fingerprint = str(
        (recovery or {}).get("evidence_fingerprint_sha256") or ""
    )
    secondary_fingerprint = str(
        (verification or {}).get("evidence_fingerprint_sha256") or ""
    )
    if (
        primary_fingerprint
        and secondary_fingerprint
        and primary_fingerprint == secondary_fingerprint
    ):
        reasons.append("recovery_same_evidence_fingerprint")

    pose_diversity = evidence_pose_diversity(
        dict(recovery or {}), dict(verification or {})
    )
    if not pose_diversity.get("pose_independent"):
        reasons.append(f"recovery_{pose_diversity.get('reason')}")

    guard = dict(rejected.get("model_guard") or {})
    guard_verdict = normalize_physical_category(guard.get("verdict")).replace(
        " ", "_"
    )
    guard_topology = normalize_physical_category(guard.get("topology")).replace(
        " ", "_"
    )
    if not guard:
        reasons.append("recovery_guard_provenance_missing")
    elif guard_verdict not in _HARD_MODEL_VERDICTS:
        reasons.append("recovery_guard_verdict_not_hard_veto")
    elif normalize_physical_category(guard.get("decision")) != "keep":
        reasons.append("recovery_guard_decision_not_keep")
    elif guard_topology in {
        "standalone_component", "attached_component", "partial_unbounded",
        "background_surface", "mixed_targets", "unknown",
    } or guard.get("complete_bounded") is not True:
        reasons.append("recovery_guard_does_not_support_complete_bounded_target")

    metric_assessment = assess_physical_form(category, metric_dimensions_m)
    if not metric_assessment.get("metrics"):
        reasons.append("recovery_missing_or_invalid_metric_dimensions")
    if metric_assessment.get("hard_veto"):
        reasons.append("recovery_agreed_noun_metric_contradiction")

    reasons = sorted(set(reasons))
    accepted = not reasons
    return {
        "schema": PHYSICAL_RECOVERY_SCHEMA,
        "triggered": trigger,
        "accepted": accepted,
        "action": "recovered_probable" if accepted else "suppress_geometry_only",
        "category": category if accepted else "unresolved object",
        "confidence": min(primary["confidence"], secondary["confidence"]),
        "standalone_bounded": bool(
            primary["standalone_bounded"] and secondary["standalone_bounded"]
        ),
        "component_only": bool(
            primary["component_only"] or secondary["component_only"]
        ),
        "primary": primary,
        "verification": secondary,
        "categories_agree": bool(category),
        "view_overlap_coefficient": overlap,
        "pose_diversity": pose_diversity,
        "unique_view_count": len(unique_views),
        "unique_view_ids": unique_views,
        "metric_physical_assessment": metric_assessment,
        "maximum_semantic_tier": "probable" if accepted else "geometry_only",
        "reason_codes": reasons or ["conservative_visible_form_recovered"],
    }


__all__: Sequence[str] = (
    "PHYSICAL_GUARD_RECOVERY_MIN_CONFIDENCE",
    "PHYSICAL_GUARD_RECOVERY_SCHEMA",
    "PHYSICAL_FORM_SCHEMA",
    "PHYSICAL_RECOVERY_MIN_CONFIDENCE",
    "PHYSICAL_RECOVERY_MAX_VIEW_OVERLAP",
    "PHYSICAL_RECOVERY_MIN_TOTAL_UNIQUE_VIEWS",
    "PHYSICAL_RECOVERY_MIN_VIEWS_PER_EVENT",
    "PHYSICAL_RECOVERY_SCHEMA",
    "assess_physical_form",
    "assess_guard_visible_form_recovery",
    "assess_physical_recovery",
    "normalize_physical_category",
)
