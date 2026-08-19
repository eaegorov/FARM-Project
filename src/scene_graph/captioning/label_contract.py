"""Strict open-vocabulary label contract for mask-grounded 3D objects.

The contract deliberately separates an object's physical head noun from a
geometric primitive and from part/whole composition.  It contains no scene
names, object ids, class vocabulary or synonym table.  Confidence is only a
request-level gate; semantic tiers remain provenance decisions.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from scene_graph.captioning.evidence import (
    DEFAULT_MAX_VIEW_OVERLAP,
    DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE,
    DEFAULT_MIN_TOTAL_UNIQUE_VIEWS,
    DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES,
    DEFAULT_MIN_VIEWS_PER_EVENT,
    evidence_pose_diversity,
    evidence_view_ids,
)


OPEN_VOCABULARY_LABEL_SCHEMA = "farm.open-vocabulary-object-label.v1"
SHARED_FORM_HYPERNYM_SOURCE = "explicit_shared_form_hypernym_consensus"

CATEGORY_ROLES = {
    "whole_object",
    "carrier",
    "standalone_component",
    "whole_form",
    "primitive_form",
    "unknown",
}
SPECIFICITY_LEVELS = {
    "specific_identity",
    "object_kind",
    "generic_form",
    "unknown",
}
TOPOLOGIES = {
    "standalone_whole",
    "carrier_payload",
    "standalone_component",
    "attached_component",
    "partial_unbounded",
    "background_surface",
    "mixed_targets",
    "unknown",
}
SHAPE_PROFILES = {
    "planar",
    "elongated_rigid",
    "elongated_flexible",
    "compact_volumetric",
    "open_frame",
    "lattice",
    "articulated",
    "irregular",
    "unknown",
}
IDENTITY_BASES = {
    "diagnostic_geometry",
    "readable_text",
    "stable_appearance",
    "context_only",
    "insufficient",
}
UNKNOWN_NOUNS = {"", "unknown", "unresolved object", "object", "item", "thing"}

OPEN_VOCABULARY_JSON_INSTRUCTIONS = """Return one strict JSON object with all
keys below. Do not copy a noun from these instructions and do not use a closed
class list.
- category: a concise singular physical head noun for the complete masked
  entity, or \"unknown\".
- category_role: whole_object, carrier, standalone_component, whole_form,
  primitive_form, or unknown.
- form_hypernym: a broader open-vocabulary whole-object/form noun, or unknown.
- primitive_form: a geometry or cross-section phrase, or unknown. A primitive
  describes shape and must not be used as category.
- head_noun_is_primitive: true only when category itself is merely geometric.
- specificity: specific_identity, object_kind, generic_form, or unknown.
- topology: standalone_whole, carrier_payload, standalone_component,
  attached_component, partial_unbounded, background_surface, mixed_targets,
  or unknown.
- complete_bounded: true, false, or null.
- carrier_category: the complete carrier noun only for carrier_payload,
  otherwise unknown.
- payload_categories: visible payload nouns only for carrier_payload.
- shape_profile: planar, elongated_rigid, elongated_flexible,
  compact_volumetric, open_frame, lattice, articulated, irregular, or unknown.
- identity_basis: diagnostic_geometry, readable_text, stable_appearance,
  context_only, or insufficient.
- diagnostic_parts: short names of target-owned visible parts.
- diagnostic_view_count: number of supplied views supporting the noun.
- description: visible target evidence only.
- attributes: visible non-category attributes only.
- confidence: 0..1.
- decision: keep or unknown.

Name the whole bounded target, not its material, purpose, neighbour, support,
payload, hidden parent, or one primitive in its shape. If the mask contains a
complete carrier and payload, category names the carrier and payload_categories
remain separate. If a specific identity lacks diagnostic target-owned evidence
in at least two views, use a supported whole-form hypernym. Return unknown for
an attached fragment, incomplete/unbounded surface, mixed targets, or unstable
identity."""


def normalize_open_noun(value: object) -> str:
    """Normalize spelling without stemming or an object-vocabulary lookup."""

    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def has_open_vocabulary_contract_evidence(value: object) -> bool:
    """Detect a current strict contract even when no proposal was accepted.

    Geometry-only ensemble and reconciliation rows intentionally have no
    winning top-level ``label_contract``. Their rejected/conflicting strict
    events remain nested in provenance lists, and must still prevent legacy
    caption or detector fallbacks from resurrecting an old noun.
    """

    pending: list[object] = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if not isinstance(current, Mapping):
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if current.get("strict_open_vocabulary_seen") is True:
            return True
        if isinstance(current.get("label_contract"), Mapping) or isinstance(
            current.get("review_label_contract"), Mapping
        ):
            return True
        for key in (
            "semantic_evidence",
            "candidates",
            "reconciliation_evidence",
        ):
            children = current.get(key)
            if isinstance(children, Sequence) and not isinstance(
                children, (str, bytes, bytearray)
            ):
                pending.extend(children)
    return False


def _string_list(value: object, *, maximum: int = 8) -> tuple[list[str], bool]:
    if not isinstance(value, list):
        return [], False
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return [], False
        text = item.strip()
        if text:
            output.append(text)
    return output[:maximum], True


def _enum(value: object, allowed: set[str], default: str = "unknown") -> tuple[str, bool]:
    normalized = normalize_open_noun(value).replace(" ", "_")
    return (normalized, True) if normalized in allowed else (default, False)


def parse_open_vocabulary_label(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize one model response without semantic coercion."""

    source = dict(payload or {})
    required = {
        "category", "category_role", "form_hypernym", "primitive_form",
        "head_noun_is_primitive", "specificity", "topology",
        "complete_bounded", "carrier_category", "payload_categories",
        "shape_profile", "identity_basis", "diagnostic_parts",
        "diagnostic_view_count", "description", "attributes", "confidence",
        "decision",
    }
    reasons: list[str] = []
    missing = sorted(required - set(source))
    if missing:
        reasons.extend(f"missing_{name}" for name in missing)

    category = normalize_open_noun(source.get("category")) or "unknown"
    form_hypernym = normalize_open_noun(source.get("form_hypernym")) or "unknown"
    primitive_form = normalize_open_noun(source.get("primitive_form")) or "unknown"
    carrier_category = normalize_open_noun(source.get("carrier_category")) or "unknown"
    category_role, category_role_valid = _enum(source.get("category_role"), CATEGORY_ROLES)
    specificity, specificity_valid = _enum(source.get("specificity"), SPECIFICITY_LEVELS)
    topology, topology_valid = _enum(source.get("topology"), TOPOLOGIES)
    shape_profile, shape_valid = _enum(source.get("shape_profile"), SHAPE_PROFILES)
    identity_basis, basis_valid = _enum(source.get("identity_basis"), IDENTITY_BASES, "insufficient")
    for valid, reason in (
        (category_role_valid, "invalid_category_role"),
        (specificity_valid, "invalid_specificity"),
        (topology_valid, "invalid_topology"),
        (shape_valid, "invalid_shape_profile"),
        (basis_valid, "invalid_identity_basis"),
    ):
        if not valid:
            reasons.append(reason)

    primitive_flag = source.get("head_noun_is_primitive")
    if not isinstance(primitive_flag, bool):
        reasons.append("invalid_head_noun_is_primitive")
        primitive_flag = True
    complete_bounded = source.get("complete_bounded")
    if complete_bounded is not None and not isinstance(complete_bounded, bool):
        reasons.append("invalid_complete_bounded")
        complete_bounded = None

    payload_categories, payload_valid = _string_list(source.get("payload_categories"))
    diagnostic_parts, parts_valid = _string_list(source.get("diagnostic_parts"))
    attributes, attributes_valid = _string_list(source.get("attributes"))
    if not payload_valid:
        reasons.append("invalid_payload_categories")
    if not parts_valid:
        reasons.append("invalid_diagnostic_parts")
    if not attributes_valid:
        reasons.append("invalid_attributes")

    view_count = source.get("diagnostic_view_count")
    if isinstance(view_count, bool) or not isinstance(view_count, int) or view_count < 0:
        reasons.append("invalid_diagnostic_view_count")
        view_count = 0
    try:
        confidence = float(source.get("confidence"))
    except (TypeError, ValueError):
        confidence = math.nan
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        reasons.append("invalid_confidence")
        confidence = 0.0

    decision = normalize_open_noun(source.get("decision"))
    if decision not in {"keep", "unknown"}:
        reasons.append("invalid_decision")
        decision = "unknown"
    if not isinstance(source.get("description"), str):
        reasons.append("invalid_description")
    description = str(source.get("description") or "").strip()

    if category_role == "primitive_form" and primitive_flag is not True:
        reasons.append("primitive_role_flag_mismatch")
    if category_role != "primitive_form" and primitive_flag is True and decision == "keep":
        reasons.append("primitive_flag_role_mismatch")
    if topology == "carrier_payload":
        if category_role != "carrier":
            reasons.append("carrier_payload_role_mismatch")
        if carrier_category in UNKNOWN_NOUNS:
            reasons.append("carrier_payload_missing_carrier")
        if normalize_open_noun(category) != normalize_open_noun(carrier_category):
            reasons.append("carrier_payload_category_mismatch")
        if not payload_categories:
            reasons.append("carrier_payload_missing_payload")
    elif carrier_category not in UNKNOWN_NOUNS or payload_categories:
        reasons.append("noncarrier_has_carrier_payload_fields")

    return {
        "schema": OPEN_VOCABULARY_LABEL_SCHEMA,
        "contract_valid": not reasons,
        "reason_codes": sorted(set(reasons)),
        "category": category,
        "category_role": category_role,
        "form_hypernym": form_hypernym,
        "primitive_form": primitive_form,
        "head_noun_is_primitive": bool(primitive_flag),
        "specificity": specificity,
        "topology": topology,
        "complete_bounded": complete_bounded,
        "carrier_category": carrier_category,
        "payload_categories": [normalize_open_noun(value) for value in payload_categories],
        "shape_profile": shape_profile,
        "identity_basis": identity_basis,
        "diagnostic_parts": diagnostic_parts,
        "diagnostic_view_count": int(view_count),
        "description": description,
        "attributes": attributes,
        "confidence": float(confidence),
        "decision": decision,
    }


def assess_open_vocabulary_label(
    payload: Mapping[str, Any] | None,
    *,
    minimum_confidence: float = 0.85,
) -> dict[str, Any]:
    """Choose a presentation-safe head noun or fail closed."""

    contract = (
        dict(payload or {})
        if (payload or {}).get("schema") == OPEN_VOCABULARY_LABEL_SCHEMA
        else parse_open_vocabulary_label(payload)
    )
    reasons = list(contract.get("reason_codes") or [])
    if contract.get("contract_valid") is not True:
        reasons.append("label_contract_invalid")
    if contract.get("decision") != "keep":
        reasons.append("label_decision_not_keep")
    if float(contract.get("confidence") or 0.0) < float(minimum_confidence):
        reasons.append("label_confidence_below_threshold")
    if int(contract.get("diagnostic_view_count") or 0) < 2:
        reasons.append("label_not_supported_in_two_views")
    category = normalize_open_noun(contract.get("category"))
    role = str(contract.get("category_role") or "unknown")
    topology = str(contract.get("topology") or "unknown")
    if category in UNKNOWN_NOUNS:
        reasons.append("label_head_noun_unknown")
    if role == "primitive_form" or contract.get("head_noun_is_primitive") is True:
        reasons.append("label_head_noun_is_geometric_primitive")
    primitive = normalize_open_noun(contract.get("primitive_form"))
    if primitive not in UNKNOWN_NOUNS and primitive == category:
        reasons.append("label_repeats_primitive_form")
    if topology in {
        "attached_component", "partial_unbounded", "background_surface",
        "mixed_targets", "unknown",
    }:
        reasons.append(f"label_topology_{topology}")
    if topology in {"standalone_whole", "carrier_payload", "standalone_component"}:
        if contract.get("complete_bounded") is not True:
            reasons.append("label_not_complete_bounded")

    specificity = str(contract.get("specificity") or "unknown")
    basis = str(contract.get("identity_basis") or "insufficient")
    downgraded = False
    identity_support_missing = bool(
        specificity in {"specific_identity", "object_kind"}
        and (
            not contract.get("diagnostic_parts")
            or basis in {"context_only", "insufficient"}
            or (
                specificity == "specific_identity"
                and basis not in {"diagnostic_geometry", "readable_text"}
            )
        )
    )
    if identity_support_missing:
        fallback = normalize_open_noun(contract.get("form_hypernym"))
        if fallback not in UNKNOWN_NOUNS and fallback != primitive:
            category = fallback
            role = "whole_form"
            downgraded = True
        else:
            reasons.append("identity_lacks_diagnostic_support")
    if specificity == "generic_form" and basis in {"context_only", "insufficient"}:
        reasons.append("generic_form_lacks_target_owned_support")

    if topology == "carrier_payload":
        carrier = normalize_open_noun(contract.get("carrier_category"))
        if carrier not in UNKNOWN_NOUNS:
            category = carrier
        maximum_tier = "probable"
    elif topology == "standalone_component" or role == "standalone_component":
        maximum_tier = "probable"
    else:
        maximum_tier = "probable" if downgraded else "confirmed"

    reasons = sorted(set(reasons))
    usable = not reasons
    payloads = list(contract.get("payload_categories") or [])
    description = str(contract.get("description") or "").strip()
    attributes = list(contract.get("attributes") or [])
    if usable and downgraded:
        # The original free text was written for the rejected specific noun;
        # carrying it forward can preserve the same functional/material
        # hallucination under a safer head noun.
        description = f"bounded visible {category}"
        attributes = []
    elif usable and topology == "carrier_payload" and payloads:
        description = f"{category} carrying {', '.join(payloads)}"
    return {
        "schema": OPEN_VOCABULARY_LABEL_SCHEMA,
        "usable": usable,
        "category": category if usable else "unresolved object",
        "category_role": role,
        "form_hypernym": normalize_open_noun(contract.get("form_hypernym")),
        "maximum_semantic_tier": maximum_tier if usable else "geometry_only",
        "description": description if usable else "",
        "attributes": attributes if usable else [],
        "confidence": float(contract.get("confidence") or 0.0) if usable else 0.0,
        "topology": topology,
        "complete_bounded": contract.get("complete_bounded"),
        "reason_codes": reasons or ["open_vocabulary_label_accepted"],
        "contract": contract,
    }


def label_contract_from_event(event: Mapping[str, Any] | None) -> dict[str, Any] | None:
    value = (event or {}).get("label_contract")
    return dict(value) if isinstance(value, Mapping) else None


def event_label_assessment(
    event: Mapping[str, Any] | None,
    *,
    minimum_confidence: float = 0.85,
) -> dict[str, Any]:
    contract = label_contract_from_event(event)
    if contract is None:
        return {
            "usable": False,
            "category": "unresolved object",
            "maximum_semantic_tier": "geometry_only",
            "reason_codes": ["missing_open_vocabulary_label_contract"],
            "contract": {},
        }
    return assess_open_vocabulary_label(contract, minimum_confidence=minimum_confidence)


def shared_explicit_form_hypernym(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> str:
    left_value = normalize_open_noun(left.get("form_hypernym"))
    right_value = normalize_open_noun(right.get("form_hypernym"))
    if left_value in UNKNOWN_NOUNS or right_value in UNKNOWN_NOUNS:
        return ""
    left_primitive = normalize_open_noun(left.get("primitive_form"))
    right_primitive = normalize_open_noun(right.get("primitive_form"))
    return left_value if left_value == right_value and left_value not in {left_primitive, right_primitive} else ""


def adaptive_alternate_plan(
    initial_event: Mapping[str, Any] | None,
    alternate_manifest: Mapping[str, Any] | None,
    *,
    probable_confidence: float = 0.90,
) -> dict[str, Any]:
    """Plan one alternate blind request before spending a model call."""

    initial = event_label_assessment(initial_event, minimum_confidence=0.85)
    unresolved_or_low = bool(
        not initial.get("usable")
        or float(initial.get("confidence") or 0.0) < float(probable_confidence)
    )
    left_views = set(evidence_view_ids(dict(initial_event or {})) or ())
    right_views = set(evidence_view_ids(dict(alternate_manifest or {})) or ())
    overlap = (
        len(left_views & right_views) / min(len(left_views), len(right_views))
        if left_views and right_views else None
    )
    pose = evidence_pose_diversity(
        dict(initial_event or {}), dict(alternate_manifest or {})
    )
    confirmation_preflight = bool(
        len(left_views) >= DEFAULT_MIN_VIEWS_PER_EVENT
        and len(right_views) >= DEFAULT_MIN_VIEWS_PER_EVENT
        and len(left_views | right_views) >= DEFAULT_MIN_TOTAL_UNIQUE_VIEWS
        and overlap is not None
        and overlap <= DEFAULT_MAX_VIEW_OVERLAP
        and pose.get("pose_independent") is True
    )
    request = bool(unresolved_or_low or confirmation_preflight)
    return {
        "request_alternate": request,
        "reason": (
            "unresolved_or_low_confidence"
            if unresolved_or_low
            else "pose_confirmable"
            if confirmation_preflight
            else "single_high_confidence_probable_is_sufficient"
        ),
        "unresolved_or_low_confidence": unresolved_or_low,
        "confirmation_preflight": confirmation_preflight,
        "view_overlap_coefficient": overlap,
        "unique_view_count": len(left_views | right_views),
        "pose_diversity": pose,
        "thresholds": {
            "min_views_per_event": DEFAULT_MIN_VIEWS_PER_EVENT,
            "min_total_unique_views": DEFAULT_MIN_TOTAL_UNIQUE_VIEWS,
            "max_view_overlap": DEFAULT_MAX_VIEW_OVERLAP,
            "min_viewpoint_separation_degrees": DEFAULT_MIN_VIEWPOINT_SEPARATION_DEGREES,
            "min_normalized_camera_baseline": DEFAULT_MIN_NORMALIZED_CAMERA_BASELINE,
        },
    }


__all__: Sequence[str] = (
    "OPEN_VOCABULARY_JSON_INSTRUCTIONS",
    "OPEN_VOCABULARY_LABEL_SCHEMA",
    "SHARED_FORM_HYPERNYM_SOURCE",
    "adaptive_alternate_plan",
    "assess_open_vocabulary_label",
    "event_label_assessment",
    "has_open_vocabulary_contract_evidence",
    "label_contract_from_event",
    "normalize_open_noun",
    "parse_open_vocabulary_label",
    "shared_explicit_form_hypernym",
)
