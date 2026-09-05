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
    evidence_view_overlap,
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
PUBLICATION_FORBIDDEN_GENERIC_NOUNS = UNKNOWN_NOUNS | {
    "machine",
    "unit",
    "device",
    "equipment",
    "container",
}
DEFAULT_MIN_VISIBLE_TARGET_COVERAGE = 0.85

OPEN_VOCABULARY_JSON_INSTRUCTIONS = """Return one strict JSON object with all
keys below. Do not copy a noun from these instructions and do not use a closed
class list.
- category: a concise singular physical head noun for the physical target to
  which the consistent masked pixels belong, even when coverage is incomplete,
  or \"unknown\".
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
- context_sufficient: true only when the supplied target pixels plus stable
  target-linked context are sufficient to distinguish the physical noun;
  false when a tighter/wider/different view is needed; null only when unknown.
- visible_target_coverage: estimated 0..1 fraction of all currently visible
  pixels of the complete physical target that lie inside the outlined mask.
  Ignore fully occluded surfaces; do not mistake a well-filled component for
  complete coverage of its parent object. Use null when it cannot be judged.
- missing_visible_parts: target-owned parts that are visibly outside the mask;
  list a part only when its pixels are actually visible outside the outline.
  Do not list an expected but absent, optional, or fully occluded part. Use an
  empty list when no target-owned pixels are visibly omitted by the mask.
- included_non_target: visible foreign objects or background regions included
  by the mask; use an empty list when none are visible.
- description: visible target evidence only.
- attributes: visible non-category attributes only.
- confidence: 0..1.
- decision: keep or unknown.

Name the whole bounded target, not its material, purpose, neighbour, support,
payload, hidden parent, or one primitive in its shape. If the mask contains a
complete carrier and payload, category names the carrier and payload_categories
remain separate. If a specific identity lacks diagnostic target-owned evidence
in at least two views, use a supported whole-form hypernym. When target-owned
evidence supports a noun but the mask is partial, return that noun with decision
keep while accurately marking topology, coverage, and missing parts; this is an
identity cue, not permission to publish. Return unknown for mixed targets,
background-only masks, context-only guesses, or unstable identity."""
_OPEN_VOCABULARY_RESPONSE_PROPERTIES: dict[str, Any] = {
    "category": {"type": "string"},
    "category_role": {"type": "string", "enum": sorted(CATEGORY_ROLES)},
    "form_hypernym": {"type": "string"},
    "primitive_form": {"type": "string"},
    "head_noun_is_primitive": {"type": "boolean"},
    "specificity": {"type": "string", "enum": sorted(SPECIFICITY_LEVELS)},
    "topology": {"type": "string", "enum": sorted(TOPOLOGIES)},
    "complete_bounded": {"type": ["boolean", "null"]},
    "carrier_category": {"type": "string"},
    "payload_categories": {
        "type": "array", "items": {"type": "string"}, "maxItems": 8,
    },
    "shape_profile": {"type": "string", "enum": sorted(SHAPE_PROFILES)},
    "identity_basis": {"type": "string", "enum": sorted(IDENTITY_BASES)},
    "diagnostic_parts": {
        "type": "array", "items": {"type": "string"}, "maxItems": 8,
    },
    "diagnostic_view_count": {"type": "integer", "minimum": 0},
    "context_sufficient": {"type": ["boolean", "null"]},
    "visible_target_coverage": {
        "type": ["number", "null"], "minimum": 0.0, "maximum": 1.0,
    },
    "missing_visible_parts": {
        "type": "array", "items": {"type": "string"}, "maxItems": 8,
    },
    "included_non_target": {
        "type": "array", "items": {"type": "string"}, "maxItems": 8,
    },
    "description": {"type": "string"},
    "attributes": {
        "type": "array", "items": {"type": "string"}, "maxItems": 8,
    },
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "decision": {"type": "string", "enum": ["keep", "unknown"]},
}
OPEN_VOCABULARY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "farm_open_vocabulary_object_label",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": _OPEN_VOCABULARY_RESPONSE_PROPERTIES,
            "required": list(_OPEN_VOCABULARY_RESPONSE_PROPERTIES),
            "additionalProperties": False,
        },
    },
}




def normalize_open_noun(value: object) -> str:
    """Normalize spelling without stemming or an object-vocabulary lookup."""

    noun = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return {"palet": "pallet"}.get(noun, noun)


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
    # These fields were added to the v1 contract additively so archived FARM
    # events remain replayable. New requests explicitly ask for them; their
    # presence is recorded separately and is mandatory only for the independent
    # whole-object readiness gate.
    whole_object_fields = {
        "context_sufficient", "visible_target_coverage",
        "missing_visible_parts", "included_non_target",
    }
    whole_object_evidence_complete = whole_object_fields.issubset(source)
    context_sufficient = source.get("context_sufficient")
    if context_sufficient is not None and not isinstance(context_sufficient, bool):
        reasons.append("invalid_context_sufficient")
        context_sufficient = None
    visible_target_coverage = source.get("visible_target_coverage")
    if visible_target_coverage is not None:
        try:
            visible_target_coverage = float(visible_target_coverage)
        except (TypeError, ValueError):
            visible_target_coverage = math.nan
        if (
            not math.isfinite(visible_target_coverage)
            or not 0.0 <= visible_target_coverage <= 1.0
        ):
            reasons.append("invalid_visible_target_coverage")
            visible_target_coverage = None
    missing_visible_parts, missing_parts_valid = _string_list(
        source.get("missing_visible_parts")
        if "missing_visible_parts" in source else []
    )
    included_non_target, non_target_valid = _string_list(
        source.get("included_non_target")
        if "included_non_target" in source else []
    )
    if "missing_visible_parts" in source and not missing_parts_valid:
        reasons.append("invalid_missing_visible_parts")
    if "included_non_target" in source and not non_target_valid:
        reasons.append("invalid_included_non_target")


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
        "context_sufficient": context_sufficient,
        "visible_target_coverage": visible_target_coverage,
        "missing_visible_parts": missing_visible_parts,
        "included_non_target": included_non_target,
        "whole_object_evidence_complete": whole_object_evidence_complete,
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

def assess_open_vocabulary_identity(
    payload: Mapping[str, Any] | None,
    *,
    minimum_confidence: float = 0.85,
) -> dict[str, Any]:
    """Retain a target-supported noun for selective mask refinement.

    Unlike assess_open_vocabulary_label, this does not require a whole instance.
    A partial/component mask can therefore provide an identity cue, but the
    result is explicitly not presentation-safe.
    """

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
    if contract.get("context_sufficient") is False:
        reasons.append("label_context_insufficient")
    category = normalize_open_noun(contract.get("category"))
    primitive = normalize_open_noun(contract.get("primitive_form"))
    if category in UNKNOWN_NOUNS:
        reasons.append("label_head_noun_unknown")
    if (
        contract.get("category_role") == "primitive_form"
        or contract.get("head_noun_is_primitive") is True
        or (primitive not in UNKNOWN_NOUNS and category == primitive)
    ):
        reasons.append("label_head_noun_is_geometric_primitive")
    topology = str(contract.get("topology") or "unknown")
    if topology in {"background_surface", "mixed_targets", "unknown"}:
        reasons.append(f"label_topology_{topology}")
    basis = str(contract.get("identity_basis") or "insufficient")
    if basis in {"context_only", "insufficient"} or not contract.get("diagnostic_parts"):
        reasons.append("identity_lacks_target_owned_support")
    reasons = sorted(set(reasons))
    return {
        "schema": OPEN_VOCABULARY_LABEL_SCHEMA,
        "usable": not reasons,
        "category": category if not reasons else "unresolved object",
        "category_role": str(contract.get("category_role") or "unknown"),
        "form_hypernym": normalize_open_noun(contract.get("form_hypernym")),
        "maximum_semantic_tier": "probable" if not reasons else "geometry_only",
        "description": str(contract.get("description") or "").strip() if not reasons else "",
        "attributes": list(contract.get("attributes") or []) if not reasons else [],
        "confidence": float(contract.get("confidence") or 0.0) if not reasons else 0.0,
        "topology": topology,
        "complete_bounded": contract.get("complete_bounded"),
        "presentation_safe": False,
        "reason_codes": reasons or ["open_vocabulary_identity_accepted_for_refinement"],
        "contract": contract,
    }


def assess_open_vocabulary_whole_identity(
    payload: Mapping[str, Any] | None,
    *,
    minimum_confidence: float = 0.85,
    minimum_visible_target_coverage: float = DEFAULT_MIN_VISIBLE_TARGET_COVERAGE,
) -> dict[str, Any]:
    """Validate a whole-object noun separately from its mask completeness.

    A strict contract may identify a whole physical object even when the mask
    omits some of its visible pixels.  That noun is useful for targeted mask
    refinement, but an attached/standalone component may never be promoted to
    its parent object and incomplete scope may never imply inpainting safety.
    """

    contract = parse_open_vocabulary_label(payload)
    identity = assess_open_vocabulary_identity(
        contract, minimum_confidence=minimum_confidence
    )
    reasons = list(identity.get("reason_codes") or []) if not identity.get(
        "usable"
    ) else []
    category = normalize_open_noun(contract.get("category"))
    role = str(contract.get("category_role") or "unknown")
    topology = str(contract.get("topology") or "unknown")
    if contract.get("whole_object_evidence_complete") is not True:
        reasons.append("missing_whole_object_evidence")
    if contract.get("context_sufficient") is not True:
        reasons.append("whole_identity_context_insufficient")
    if category in PUBLICATION_FORBIDDEN_GENERIC_NOUNS:
        reasons.append("whole_identity_head_noun_generic_or_unknown")
    if role not in {"whole_object", "whole_form", "carrier"}:
        reasons.append(f"whole_identity_category_role_{role}")
    if topology in {
        "standalone_component",
        "attached_component",
        "background_surface",
        "mixed_targets",
        "unknown",
    }:
        reasons.append(f"whole_identity_topology_{topology}")
    if topology == "carrier_payload" and role != "carrier":
        reasons.append("whole_identity_carrier_role_mismatch")
    if topology != "carrier_payload" and role == "carrier":
        reasons.append("whole_identity_carrier_topology_mismatch")

    # A model can return the same specific noun twice while its own natural
    # language evidence says that noun is merely contained inside the masked
    # carrier. For example, "a box ... with a hose reel inside" cannot prove
    # that a mask covering the box is a whole hose reel. Detect this from the
    # raw contract instead of maintaining scene- or object-specific vetoes.
    description = re.sub(
        r"\s+", " ", str(contract.get("description") or "").lower()
    ).strip()
    category_words = category.split()
    category_phrases = [category]
    if len(category_words) >= 3:
        category_phrases.append(" ".join(category_words[-2:]))
    carrier_nouns = {
        "box", "cabinet", "case", "container", "enclosure", "housing",
    }
    carrier_pattern = "(?:" + "|".join(sorted(carrier_nouns)) + ")"
    for phrase in category_phrases:
        matches = list(re.finditer(
            rf"\b{re.escape(phrase)}\b\s+(?:is\s+)?(?:visible\s+)?(?:inside|within)\b",
            description,
        ))
        category_is_carrier = bool(
            carrier_nouns & set(re.findall(r"[a-z]+", phrase))
        )
        carrier_contains_category = bool(re.search(
            rf"\b{carrier_pattern}\b[^.;]{{0,80}}\b"
            rf"(?:contains|containing|houses|housing)\b[^.;]{{0,40}}\b"
            rf"(?:a|an|the)?\s*{re.escape(phrase)}\b",
            description,
        ))
        if (not category_is_carrier) and (
            carrier_contains_category or any(
            carrier_nouns & set(re.findall(r"[a-z]+", description[:match.start()]))
            for match in matches
            )
        ):
            reasons.append("whole_identity_category_described_as_contained_part")
            break

    scope_reasons: list[str] = []
    if contract.get("complete_bounded") is not True:
        scope_reasons.append("mask_scope_not_complete_bounded")
    coverage = contract.get("visible_target_coverage")
    if (
        coverage is None
        or float(coverage) < float(minimum_visible_target_coverage)
    ):
        scope_reasons.append("mask_scope_visible_coverage_below_threshold")
    if contract.get("missing_visible_parts"):
        scope_reasons.append("mask_scope_missing_visible_parts")
    if contract.get("included_non_target"):
        scope_reasons.append("mask_scope_includes_non_target")

    reasons = sorted(set(reasons))
    scope_reasons = sorted(set(scope_reasons))
    usable = not reasons
    return {
        **identity,
        "usable": usable,
        "category": category if usable else "unresolved object",
        "confidence": (
            float(contract.get("confidence") or 0.0) if usable else 0.0
        ),
        "maximum_semantic_tier": "probable" if usable else "geometry_only",
        "reason_codes": reasons or [
            "whole_identity_accepted_independently_of_mask_scope"
        ],
        "mask_scope_incomplete": bool(scope_reasons),
        "mask_scope_reason_codes": scope_reasons or ["mask_scope_complete"],
        "geometry_refinement_required": bool(scope_reasons),
        "inpainting_eligible": bool(usable and not scope_reasons),
        "contract": contract,
    }


def independent_blind_whole_identity_consensus(
    initial: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    minimum_confidence: float = 0.85,
) -> dict[str, Any]:
    """Require two unconditioned, pose-independent raw contracts to agree."""

    reasons: list[str] = []

    def assess(event: Mapping[str, Any], prefix: str) -> tuple[str, dict[str, Any]]:
        raw = event.get("label_contract")
        if not isinstance(raw, Mapping):
            raw = event.get("review_label_contract")
        assessment = assess_open_vocabulary_whole_identity(
            raw if isinstance(raw, Mapping) else None,
            minimum_confidence=minimum_confidence,
        )
        contract = assessment.get("contract") or {}
        if contract.get("contract_valid") is not True:
            reasons.append(f"{prefix}_label_contract_invalid")
        noun = normalize_open_noun(assessment.get("category"))
        if noun in PUBLICATION_FORBIDDEN_GENERIC_NOUNS:
            reasons.append(f"{prefix}_category_generic_or_unresolved")
        if (
            not assessment.get("usable")
            and contract.get("contract_valid") is True
            and normalize_open_noun(contract.get("category"))
            not in PUBLICATION_FORBIDDEN_GENERIC_NOUNS
        ):
            reasons.append(f"{prefix}_whole_identity_not_supported")
        for detail in assessment.get("reason_codes") or []:
            if detail == "whole_identity_category_described_as_contained_part":
                reasons.append(f"{prefix}_{detail}")
        return noun, assessment

    event_pair = any(
        key in initial or key in verification
        for key in ("source", "event_id", "crop_image_ids")
    )
    initial_event_id = str(initial.get("event_id") or "")
    verification_event_id = str(verification.get("event_id") or "")
    initial_ids = set(evidence_view_ids(initial) or ())
    verification_ids = set(evidence_view_ids(verification) or ())
    overlap: dict[str, Any] = {}
    if event_pair:
        initial_source = str(initial.get("source") or "")
        verification_source = str(verification.get("source") or "")
        if initial_source not in {
            "initial_blind_review",
            "contextual_dual_panel_review",
        }:
            reasons.append("initial_event_source_invalid")
        if verification_source != "independent_verification":
            reasons.append("verification_event_source_invalid")
        try:
            initial_object_id = int(initial.get("evidence_object_id"))
            verification_object_id = int(verification.get("evidence_object_id"))
        except (TypeError, ValueError):
            initial_object_id = verification_object_id = -1
        if (
            initial_object_id < 0
            or initial_object_id != verification_object_id
            or initial_event_id
            != f"object:{initial_object_id}:{initial_source}"
            or verification_event_id
            != f"object:{initial_object_id}:independent_verification"
        ):
            reasons.append("semantic_event_identity_or_object_mismatch")
        if not initial_event_id or initial_event_id == verification_event_id:
            reasons.append("semantic_event_ids_missing_or_not_unique")
        if any(
            event.get("confirmation_eligible") is not True
            for event in (initial, verification)
        ):
            reasons.append("event_pair_not_confirmation_eligible")
        if any(
            str(event.get("semantic_vote_independence") or "")
            != "independent"
            for event in (initial, verification)
        ):
            reasons.append("event_pair_not_explicitly_independent")
        if any(
            str(event.get("request_conditioning") or "") != "none_blind"
            for event in (initial, verification)
        ):
            reasons.append("event_pair_not_unconditioned")
        overlap = evidence_view_overlap(initial, verification)
        if overlap.get("independent") is not True:
            reasons.append(
                "event_pair_"
                + str(overlap.get("reason") or "provenance_not_independent")
            )

    initial_noun, initial_assessment = assess(initial, "initial")
    verification_noun, verification_assessment = assess(
        verification, "verification"
    )
    if initial_noun and verification_noun and initial_noun != verification_noun:
        reasons.append("independent_blind_category_disagreement")
    accepted = bool(not reasons and initial_noun == verification_noun)
    scope_reasons = sorted(set(
        list(initial_assessment.get("mask_scope_reason_codes") or [])
        + list(verification_assessment.get("mask_scope_reason_codes") or [])
    ) - {"mask_scope_complete"})
    result: dict[str, Any] = {
        "schema": "farm.independent-blind-label-consensus.v1",
        "status": "accepted" if accepted else "unresolved",
        "accepted": accepted,
        "initial_category": initial_noun,
        "verification_category": verification_noun,
        "canonical_category": initial_noun if accepted else "",
        "initial_maximum_semantic_tier": initial_assessment.get(
            "maximum_semantic_tier"
        ),
        "verification_maximum_semantic_tier": verification_assessment.get(
            "maximum_semantic_tier"
        ),
        "reason_codes": sorted(set(reasons)) or [
            "exact_safe_canonical_noun_agreement"
        ],
        "identity_contract_source": "raw_label_contract",
        "mask_scope_incomplete": bool(scope_reasons),
        "mask_scope_reason_codes": scope_reasons or ["mask_scope_complete"],
        "geometry_refinement_required": bool(scope_reasons),
        "inpainting_eligible": bool(accepted and not scope_reasons),
    }
    if event_pair:
        result.update({
            "evidence_pair_source": "exact_current_semantic_evidence_events",
            "initial_event_id": initial_event_id,
            "verification_event_id": verification_event_id,
            "initial_crop_image_ids": sorted(initial_ids),
            "verification_crop_image_ids": sorted(verification_ids),
            "initial_evidence_fingerprint_sha256": str(
                initial.get("evidence_fingerprint_sha256") or ""
            ),
            "verification_evidence_fingerprint_sha256": str(
                verification.get("evidence_fingerprint_sha256") or ""
            ),
            "evidence_independence": overlap,
        })
    return result


def assess_whole_object_readiness(
    payload: Mapping[str, Any] | None,
    *,
    minimum_confidence: float = 0.85,
    minimum_visible_target_coverage: float = DEFAULT_MIN_VISIBLE_TARGET_COVERAGE,
) -> dict[str, Any]:
    """Require explicit segmentation completeness in addition to a safe noun."""

    label = assess_open_vocabulary_label(
        payload, minimum_confidence=minimum_confidence
    )
    contract = dict(label.get("contract") or {})
    reasons = list(label.get("reason_codes") or []) if not label.get("usable") else []
    if contract.get("whole_object_evidence_complete") is not True:
        reasons.append("missing_whole_object_evidence")
    if contract.get("context_sufficient") is not True:
        reasons.append("whole_object_context_insufficient")
    coverage = contract.get("visible_target_coverage")
    if coverage is None or float(coverage) < float(minimum_visible_target_coverage):
        reasons.append("visible_target_coverage_below_threshold")
    if contract.get("missing_visible_parts"):
        reasons.append("mask_missing_visible_target_parts")
    if contract.get("included_non_target"):
        reasons.append("mask_contains_non_target_regions")
    reasons = sorted(set(reasons))
    return {
        **label,
        "ready": not reasons,
        "visible_target_coverage": coverage,
        "minimum_visible_target_coverage": float(minimum_visible_target_coverage),
        "reason_codes": reasons or ["whole_object_evidence_accepted"],
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
    "DEFAULT_MIN_VISIBLE_TARGET_COVERAGE",
    "OPEN_VOCABULARY_JSON_INSTRUCTIONS",
    "OPEN_VOCABULARY_LABEL_SCHEMA",
    "OPEN_VOCABULARY_RESPONSE_FORMAT",
    "PUBLICATION_FORBIDDEN_GENERIC_NOUNS",
    "SHARED_FORM_HYPERNYM_SOURCE",
    "adaptive_alternate_plan",
    "assess_open_vocabulary_identity",
    "assess_open_vocabulary_whole_identity",
    "assess_whole_object_readiness",
    "assess_open_vocabulary_label",
    "event_label_assessment",
    "has_open_vocabulary_contract_evidence",
    "independent_blind_whole_identity_consensus",
    "label_contract_from_event",
    "normalize_open_noun",
    "parse_open_vocabulary_label",
    "shared_explicit_form_hypernym",
)
