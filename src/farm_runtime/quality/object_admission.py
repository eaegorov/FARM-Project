"""Conservative, model-backed admission of physical object candidates.

This module does not run a model, infer names, merge instances, or discard masks.
The caller supplies grounded views and trustworthy physical timestamps. Agreement
is a catalog routing proposal, not ground truth or proof of 3D mask completeness.
"""

from __future__ import annotations

from collections.abc import Mapping
import json


SCHEMA = "farm.object-admission.v1"
SCOPES = frozenset({
    "single_whole_object", "object_part", "structural_surface",
    "multiple_independent_objects", "unclear",
})
MASK_FITS = frozenset({
    "fits_target", "partial_target", "crosses_objects", "unclear",
})
CONFIDENCES = frozenset({"high", "medium", "low"})
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scope", "mask_fit", "confidence", "reason"],
    "properties": {
        "scope": {"type": "string", "enum": sorted(SCOPES)},
        "mask_fit": {"type": "string", "enum": sorted(MASK_FITS)},
        "confidence": {"type": "string", "enum": sorted(CONFIDENCES)},
        "reason": {"type": "string", "minLength": 1},
    },
}

PROMPT = """Assess the highlighted selection in ONE actual source photograph. The full view and aligned close-up show the SAME selection. Outlines, shading and markers are artificial annotations, not object appearance. Unshaded surrounding pixels give context; they are not selected. Judge what the highlighted pixels select, not an imagined object suggested by its crop or a detector name.
Choose the physical scope:
- single_whole_object: one independently meaningful physical object or one integrated equipment assembly. A cabinet with its fitted doors and controls can be one object. An integrated conveyor can be one object if the visible structure supports this. Exact name, function and model number are unnecessary.
- object_part: a selected component of a larger object, such as an enclosure door or front panel. A complete-looking panel close-up does not make it a whole cabinet.
- structural_surface: a region of surrounding building or support structure rather than an independently meaningful item. Do not classify a mounted device or sign as structure merely because it touches a wall.
- multiple_independent_objects: the selection spans a collection of separate objects. Touching, similar colour or aligned placement does not establish one assembly.
- unclear: the selection or its physical identity cannot be established from the photograph.
Judge mask_fit separately: fits_target if the selection follows the visible extent of that physical target; partial_target if it misses a substantial visible part of the target; crosses_objects if it includes unrelated objects/surfaces; unclear if this cannot be judged. For multiple_independent_objects, crosses_objects is appropriate when separate objects are selected together. fits_target never asserts that hidden geometry is complete. Ordinary occlusion or image cropping does not itself imply partial_target: all visible target pixels can be selected even when its rest lies behind an occluder or outside the photograph. The visible context must still support its identity as one whole physical object. Detached selected islands may be visible pieces of an occluded object; do not infer random fragments from disconnectedness alone.
Do not turn a partial mask of a whole object into an independent component unless the selected component itself is visually identifiable. A poster/sign, extinguisher, loose container or independently mounted device may be a separate meaningful object even when small or contained in another mask. External pipes, cables and neighbouring equipment are not automatically integral parts. Never join a row of cabinets or machines merely to obtain a larger object. Never reject solely because an object is small or at the image border.
Use high confidence only when the highlighted extent and its physical scope are directly supported by the photograph. If important boundaries or attachment are ambiguous, use medium/low or unclear. Do not invent a precise equipment name. Return ONLY JSON with exactly:
{"scope":"single_whole_object|object_part|structural_surface|multiple_independent_objects|unclear","mask_fit":"fits_target|partial_target|crosses_objects|unclear","confidence":"high|medium|low","reason":"brief visual reason about the selected extent, under 40 words"}."""


def _unique_json_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate response field")
        value[key] = item
    return value


def validate_admission_response(raw):
    """Return a fresh validated response; malformed model output raises ValueError.

    Model output cannot supply an object ID, timestamp, or an acceptance flag.
    A mapping is supported for providers that already decode structured output.
    """
    if isinstance(raw, str):
        try:
            result = json.loads(raw, object_pairs_hook=_unique_json_fields)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid admission JSON") from exc
    elif isinstance(raw, Mapping):
        result = dict(raw)
    else:
        raise ValueError("admission response must be JSON text or a mapping")
    if not isinstance(result, dict) or set(result) != {
        "scope", "mask_fit", "confidence", "reason"
    }:
        raise ValueError("exact admission response fields required")
    for field, allowed in (
        ("scope", SCOPES), ("mask_fit", MASK_FITS), ("confidence", CONFIDENCES)
    ):
        if not isinstance(result[field], str) or result[field] not in allowed:
            raise ValueError(f"invalid admission {field}")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("nonempty visual reason required")
    return dict(result)


def aggregate_admission(observations, minimum_independent_timestamps=2):
    """Route one candidate using unanimous independent physical-time evidence.

    Each observation is a mapping with ``view_id`` (int or nonempty str),
    ``physical_timestamp`` (nonempty str from trusted frame metadata), and
    ``response`` (JSON or mapping). Optional ``dependency_valid=True``,
    ``fallback_used=False`` and ``validation_error=None`` describe runner checks.
    The caller must set these when source/mask dependencies fail; this pure
    function cannot verify images, hashes or claimed metadata by itself.

    Repeated cameras, crops or model calls from one physical timestamp count
    once. Reusing a view ID with different timestamps is invalid. Every supplied
    observation participates: no invalid or contrary answer is silently dropped.
    Only high-confidence consistent whole-object / fits-target evidence admits a
    primary object. Other consistent scopes route to context, part or collection;
    they remain available for later processing. Partial masks defer for recovery.
    This function neither edits membership nor resolves duplicate identities.
    """
    if type(minimum_independent_timestamps) is not int or minimum_independent_timestamps < 2:
        raise ValueError("at least two independent physical timestamps required")
    if isinstance(observations, (str, bytes, Mapping)):
        raise ValueError("observations must be an iterable of observation mappings")

    views = []
    by_time = {}
    view_times = {}
    reasons = set()
    scopes, fits = set(), set()
    for index, observation in enumerate(observations):
        row = dict(observation_index=index, view_id=None, physical_timestamp=None,
                   response=None, reason_codes=[])
        errors = set()
        if not isinstance(observation, Mapping):
            errors.add("invalid_observation")
        else:
            view_id = observation.get("view_id")
            if type(view_id) is int or isinstance(view_id, str) and view_id.strip():
                row["view_id"] = str(view_id).strip()
            else:
                errors.add("invalid_view_id")
            timestamp = observation.get("physical_timestamp")
            if isinstance(timestamp, str) and timestamp.strip():
                row["physical_timestamp"] = timestamp.strip()
            else:
                errors.add("invalid_physical_timestamp")
            if type(observation.get("dependency_valid", True)) is not bool:
                errors.add("invalid_dependency_status")
            elif not observation.get("dependency_valid", True):
                errors.add("dependency_invalid")
            if type(observation.get("fallback_used", False)) is not bool:
                errors.add("invalid_fallback_status")
            elif observation.get("fallback_used", False):
                errors.add("fallback_used")
            if observation.get("validation_error") is not None:
                errors.add("runner_validation_error")
            try:
                response = validate_admission_response(observation.get("response"))
            except ValueError as exc:
                errors.add("invalid_response")
                row["validation_error"] = str(exc)
            else:
                row["response"] = response
                scopes.add(response["scope"])
                fits.add(response["mask_fit"])
                if response["confidence"] != "high":
                    errors.add("confidence_below_high")
                if response["scope"] == "unclear" or response["mask_fit"] == "unclear":
                    errors.add("unclear_response")
                if response["mask_fit"] == "partial_target":
                    errors.add("partial_target_requires_recovery")
                if (response["mask_fit"] == "crosses_objects"
                        and response["scope"] != "multiple_independent_objects"):
                    errors.add("mask_crosses_objects")
        timestamp, view_id = row["physical_timestamp"], row["view_id"]
        if timestamp is not None:
            by_time.setdefault(timestamp, []).append(row)
        if view_id is not None and timestamp is not None:
            view_times.setdefault(view_id, set()).add(timestamp)
        row["reason_codes"] = sorted(errors)
        reasons.update(errors)
        views.append(row)

    if not views:
        reasons.add("no_observations")
    if any(len(times) > 1 for times in view_times.values()):
        reasons.add("view_timestamp_conflict")
    if len(scopes) > 1:
        reasons.add("scope_conflict")
    if len(fits) > 1:
        reasons.add("mask_fit_conflict")

    timestamp_audit = []
    supporting = []
    for timestamp, rows in sorted(by_time.items()):
        local_reasons = {reason for row in rows for reason in row["reason_codes"]}
        local_scopes = {row["response"]["scope"] for row in rows if row["response"]}
        local_fits = {row["response"]["mask_fit"] for row in rows if row["response"]}
        if len(local_scopes) > 1:
            local_reasons.add("scope_conflict")
        if len(local_fits) > 1:
            local_reasons.add("mask_fit_conflict")
        if any(len(view_times.get(row["view_id"], set())) > 1 for row in rows):
            local_reasons.add("view_timestamp_conflict")
        if not local_reasons:
            supporting.append(timestamp)
        timestamp_audit.append(dict(
            physical_timestamp=timestamp,
            view_ids=sorted({row["view_id"] for row in rows if row["view_id"] is not None}),
            observation_count=len(rows),
            scope=next(iter(local_scopes)) if len(local_scopes) == 1 else "unclear",
            mask_fit=next(iter(local_fits)) if len(local_fits) == 1 else "unclear",
            reason_codes=sorted(local_reasons),
        ))
    if len(supporting) < minimum_independent_timestamps:
        reasons.add("insufficient_independent_timestamps")

    scope = next(iter(scopes)) if len(scopes) == 1 else "unclear"
    mask_fit = next(iter(fits)) if len(fits) == 1 else "unclear"
    decision = "defer"
    if not reasons:
        decision = {
            "single_whole_object": "admit_whole_object",
            "object_part": "route_part",
            "structural_surface": "route_context",
            "multiple_independent_objects": "route_collection",
        }[scope]
    return dict(
        schema=SCHEMA, decision=decision, admitted=decision == "admit_whole_object",
        evidence_status="model_review_not_ground_truth", scope=scope, mask_fit=mask_fit,
        minimum_independent_timestamps=minimum_independent_timestamps,
        independent_timestamp_count=len(by_time), supporting_timestamps=supporting,
        supporting_timestamp_count=len(supporting), reason_codes=sorted(reasons),
        timestamp_audit=timestamp_audit, view_audit=views,
    )


SCHEMA_V2 = "farm.object-admission-witness.v2"
TARGET_KINDS = frozenset({
    "independent_item", "intrinsic_component", "building_structure", "collection", "unclear",
})
PARENT_KINDS = frozenset({"bounded_manufactured_object", "building_support", "none", "unclear"})
PARENT_RELATIONS = frozenset({"integral_component", "mounted_separate", "none", "unclear"})
RESPONSE_JSON_SCHEMA_V2 = {
    "type": "object", "additionalProperties": False,
    "required": ["target_kind", "mask_fit", "confidence", "reason", "parent_witness"],
    "properties": {
        "target_kind": {"type": "string", "enum": sorted(TARGET_KINDS)},
        "mask_fit": {"type": "string", "enum": sorted(MASK_FITS)},
        "confidence": {"type": "string", "enum": sorted(CONFIDENCES)},
        "reason": {"type": "string", "minLength": 1},
        "parent_witness": {
            "type": "object", "additionalProperties": False,
            "required": ["kind", "relation", "visual_evidence"],
            "properties": {
                "kind": {"type": "string", "enum": sorted(PARENT_KINDS)},
                "relation": {"type": "string", "enum": sorted(PARENT_RELATIONS)},
                "visual_evidence": {"type": "string"},
            },
        },
    },
}

PROMPT_V2 = """Judge the highlighted pixels in each actual photograph, using the full view and aligned close-up together. Coloured outlines and shading are annotations, not object appearance. Unhighlighted context is not selected.
Choose target_kind:
independent_item = one independently meaningful physical item or integrated equipment assembly, including a whole cabinet or integrated conveyor. An item can remain independent when mounted on a wall or another item. Mounted signs, safety devices and separate enclosures do not become intrinsic components merely through attachment.
intrinsic_component = a constituent part of a visibly larger, bounded manufactured object, such as a fitted enclosure door. This requires a parent witness: describe the larger object's visible body OUTSIDE the selection and the integral connection. Merely saying attached or mounted does not establish an intrinsic component.
building_structure = the building surface or supporting construction itself.
collection = several independent items selected together; touching or similarity does not establish one assembly.
unclear = the highlighted target cannot be identified at this physical level. No exact equipment name is required.
Supply parent_witness with kind bounded_manufactured_object|building_support|none|unclear, relation integral_component|mounted_separate|none|unclear, and visual_evidence describing what is actually visible. A wall, floor or room used as support is not a manufactured equipment parent. Mounting an otherwise independent item on a cabinet is mounted_separate, not integral_component. If no parent is relevant, use none/none and an empty visual_evidence. If the necessary witness is not visible, do not invent one.
Separately choose mask_fit: fits_target follows the target's full VISIBLE extent; partial_target misses substantial visible target regions; crosses_objects includes unrelated objects/surfaces; unclear cannot be assessed. Occlusion or image cropping alone does not imply partial_target, and fits_target never certifies hidden geometry. A partial mask of an item is not automatically an intrinsic component. Disconnected visible patches alone do not establish multiple items.
Give confidence high|medium|low and a short visual reason. Use high only for directly supported physical scope and visible mask extent. Each response must contain exactly target_kind, mask_fit, confidence, reason, parent_witness; the caller supplies the view identifiers and response envelope."""


def validate_parent_witness_response(raw):
    """Validate a v2 model assertion, without asserting its visual truth.

    The witness is required even when its kind/relation are explicitly ``none``.
    Cross-field physical consistency is handled by aggregate_witness_admission
    so a malformed physical assertion is retained as a deferred audit record.
    """
    if isinstance(raw, str):
        try:
            result = json.loads(raw, object_pairs_hook=_unique_json_fields)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid witness admission JSON") from exc
    elif isinstance(raw, Mapping):
        result = dict(raw)
    else:
        raise ValueError("witness response must be JSON text or a mapping")
    if not isinstance(result, dict) or set(result) != {
        "target_kind", "mask_fit", "confidence", "reason", "parent_witness"
    }:
        raise ValueError("exact witness admission fields required")
    for field, allowed in (
        ("target_kind", TARGET_KINDS), ("mask_fit", MASK_FITS), ("confidence", CONFIDENCES)
    ):
        if not isinstance(result[field], str) or result[field] not in allowed:
            raise ValueError(f"invalid witness admission {field}")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("nonempty visual reason required")
    witness = result["parent_witness"]
    if not isinstance(witness, Mapping) or set(witness) != {"kind", "relation", "visual_evidence"}:
        raise ValueError("exact parent witness fields required")
    for field, allowed in (("kind", PARENT_KINDS), ("relation", PARENT_RELATIONS)):
        if not isinstance(witness[field], str) or witness[field] not in allowed:
            raise ValueError(f"invalid parent witness {field}")
    if not isinstance(witness["visual_evidence"], str):
        raise ValueError("parent visual evidence must be a string")
    return dict(result, parent_witness=dict(witness))


def _parent_witness_errors(response):
    target = response["target_kind"]
    witness = response["parent_witness"]
    kind, relation = witness["kind"], witness["relation"]
    errors = set()
    if kind == "unclear" or relation == "unclear":
        errors.add("unclear_parent_witness")
    if (kind == "none") != (relation == "none"):
        errors.add("inconsistent_parent_witness")
    if kind not in {"none", "unclear"} and not witness["visual_evidence"].strip():
        errors.add("missing_parent_visual_evidence")
    if target == "intrinsic_component":
        if kind != "bounded_manufactured_object" or relation != "integral_component":
            errors.add("intrinsic_component_requires_bounded_integral_parent")
    elif target == "independent_item" and relation == "integral_component":
        errors.add("independent_item_conflicts_with_integral_relation")
    elif target == "collection" and relation == "integral_component":
        errors.add("collection_conflicts_with_integral_relation")
    elif target == "building_structure" and kind == "bounded_manufactured_object":
        errors.add("building_structure_conflicts_with_equipment_parent")
    return errors


def aggregate_witness_admission(observations, minimum_independent_timestamps=2):
    """Apply v2 parent-witness checks and the unchanged physical-time consensus.

    Observation metadata has the same contract as aggregate_admission. A missing
    or inconsistent component witness always defers; it never promotes a part to
    an independent item. Positive independent-item evidence is still mandatory
    for admission. Witness prose remains an unverified model assertion: this
    function cannot verify pixels or establish a particular parent instance.
    It does not authorize physical merging, suppression or membership transfer.
    """
    if isinstance(observations, (str, bytes, Mapping)):
        raise ValueError("observations must be an iterable of observation mappings")
    scope_map = {
        "independent_item": "single_whole_object", "intrinsic_component": "object_part",
        "building_structure": "structural_surface", "collection": "multiple_independent_objects",
        "unclear": "unclear",
    }
    translated, audit, kinds = [], [], set()
    for observation in observations:
        if not isinstance(observation, Mapping):
            translated.append(observation)
            audit.append(dict(response=None, reason_codes=[], validation_error=None))
            continue
        row = dict(observation)
        response, errors, validation_error = None, set(), None
        try:
            response = validate_parent_witness_response(observation.get("response"))
        except ValueError as exc:
            errors.add("invalid_witness_response")
            validation_error = str(exc)
            row["response"] = None
        else:
            kinds.add(response["target_kind"])
            errors.update(_parent_witness_errors(response))
            row["response"] = dict(
                scope=scope_map[response["target_kind"]], mask_fit=response["mask_fit"],
                confidence=response["confidence"], reason=response["reason"],
            )
        if errors and row.get("validation_error") is None:
            row["validation_error"] = "parent_witness_policy_failed"
        translated.append(row)
        audit.append(dict(response=response, reason_codes=sorted(errors), validation_error=validation_error))
    result = aggregate_admission(translated, minimum_independent_timestamps)
    result["schema"] = SCHEMA_V2
    result["target_kind"] = next(iter(kinds)) if len(kinds) == 1 else "unclear"
    result["parent_witness_is_ground_truth"] = False
    for view_row, witness in zip(result["view_audit"], audit):
        view_row["normalized_scope_response"] = view_row["response"]
        view_row["response"] = witness["response"]
        view_row["reason_codes"] = sorted(set(view_row["reason_codes"]) | set(witness["reason_codes"]))
        if witness["validation_error"] is not None:
            view_row["validation_error"] = witness["validation_error"]
    result["reason_codes"] = sorted(set(result["reason_codes"]) | {
        reason for row in audit for reason in row["reason_codes"]
    })
    for timestamp in result["timestamp_audit"]:
        matching = [row for row in result["view_audit"]
                    if row["physical_timestamp"] == timestamp["physical_timestamp"]]
        timestamp["reason_codes"] = sorted(set(timestamp["reason_codes"]) | {
            reason for row in matching for reason in row["reason_codes"]
        })
    return result
