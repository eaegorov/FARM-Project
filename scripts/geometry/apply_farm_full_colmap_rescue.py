#!/usr/bin/env python3
"""Apply full-COLMAP rescue with separate existence and label gates.

SAM3, multi-view RGB-D geometry and a blind whole-object VLM decision establish
that an object exists. Only two provably independent blind semantic votes may
publish a label; semantic uncertainty cannot erase validated geometry.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.obb_release_evidence import (  # noqa: E402
    validate_serialized_obb_release_evidence,
)
from farm_runtime.full_colmap_refinement_contract import (  # noqa: E402
    require_train_fit_refinement,
)
from scene_graph.captioning.label_contract import (  # noqa: E402
    assess_open_vocabulary_label,
    assess_open_vocabulary_whole_identity,
    independent_blind_whole_identity_consensus,
    normalize_open_noun,
)

BLIND_EXISTENCE_SOURCES = {
    "initial_blind_review",
    "contextual_dual_panel_review",
}


CATEGORY_FAMILY_ALIASES = {
    "cleaner": "cleaning machine",
    "floor cleaner": "cleaning machine",
    "floor scrubber": "cleaning machine",
    "vacuum cleaner": "cleaning machine",
    "palet": "pallet",
}
UNRESOLVED_CATEGORIES = {"", "object", "unknown", "unresolved object"}


PUBLISH_FORBIDDEN_GENERIC_CATEGORIES = UNRESOLVED_CATEGORIES | {
    "machine",
    "unit",
    "device",
    "equipment",
    "container",
    "item",
    "thing",
}

SEMANTIC_CONFLICT_REASONS = {
    "vlm_candidate_category_conflicts_independent_blind",
    "vlm_review_verification_mode_not_independent_blind",
    "vlm_independent_blind_consensus_missing_or_invalid",
    "vlm_independent_blind_consensus_not_accepted",
    "vlm_independent_blind_consensus_category_disagreement",
    "vlm_independent_blind_consensus_event_provenance_mismatch",
}

INDEPENDENT_BLIND_CONSENSUS_SCHEMA = (
    "farm.independent-blind-label-consensus.v1"
)
GRAVITY_UPRIGHT_NORMALIZATION_SCHEMA = (
    "farm.gravity-upright-normalization.v1"
)


def _canonical_category(value: object) -> str:
    category = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return CATEGORY_FAMILY_ALIASES.get(category, category)


def _blind_category(row: Mapping[str, Any]) -> str:
    """Return the independent canonical/blind category, never the final candidate."""

    contextual = row.get("contextual_semantic_review")
    whole = (
        contextual.get("whole_object_assessment")
        if isinstance(contextual, Mapping)
        else None
    )
    canonical_values: list[object] = [row.get("category")]
    contract = row.get("label_contract")
    if isinstance(contract, Mapping):
        canonical_values.append(contract.get("category"))
    if isinstance(contextual, Mapping):
        canonical_values.append(contextual.get("contextual_category"))
    if isinstance(whole, Mapping):
        canonical_values.append(whole.get("category"))
        whole_contract = whole.get("contract")
        if isinstance(whole_contract, Mapping):
            canonical_values.append(whole_contract.get("category"))
    canonical = {
        _canonical_category(value)
        for value in canonical_values
        if _canonical_category(value) not in UNRESOLVED_CATEGORIES
    }
    if len(canonical) == 1:
        return next(iter(canonical))
    if len(canonical) > 1:
        return ""

    events = [
        event
        for event in row.get("semantic_evidence") or []
        if isinstance(event, Mapping)
        and str(event.get("source") or "") in BLIND_EXISTENCE_SOURCES
    ]
    if len(events) != 1:
        return ""
    contract = events[0].get("label_contract")
    value = (
        contract.get("category") or events[0].get("category")
        if isinstance(contract, Mapping)
        else events[0].get("category")
    )
    return _canonical_category(value)


def _event_category(event: Mapping[str, Any]) -> str:
    contract = event.get("label_contract")
    value = event.get("category")
    if not str(value or "").strip() and isinstance(contract, Mapping):
        value = contract.get("category")
    return _canonical_category(value)


def _exact_event_category(event: Mapping[str, Any]) -> str:
    contract = event.get("label_contract")
    value = event.get("category")
    if not str(value or "").strip() and isinstance(contract, Mapping):
        value = contract.get("category")
    return _exact_category(value)


def _publication_category_reasons(
    initial: Mapping[str, Any], verification: Mapping[str, Any]
) -> list[str]:
    initial_category = _event_category(initial)
    verification_category = _event_category(verification)
    reasons: list[str] = []
    if not initial_category or not verification_category:
        reasons.append("vlm_two_pass_category_missing")
    if (
        initial_category in PUBLISH_FORBIDDEN_GENERIC_CATEGORIES
        or verification_category in PUBLISH_FORBIDDEN_GENERIC_CATEGORIES
    ):
        reasons.append("vlm_category_too_generic_for_publication")
    if (
        initial_category
        and verification_category
        and initial_category != verification_category
    ):
        reasons.append("vlm_two_pass_category_disagreement")
    return reasons


def _exact_category(value: object) -> str:
    """Normalize spelling only; never collapse distinct object families."""

    return normalize_open_noun(value)


def _event_source_images(event: Mapping[str, Any]) -> set[str] | None:
    """Return complete gravity-upright source identities, or fail closed."""

    manifest = event.get("gravity_upright_normalization")
    if not isinstance(manifest, Mapping):
        return None
    if manifest.get("schema") != GRAVITY_UPRIGHT_NORMALIZATION_SCHEMA:
        return None
    if manifest.get("provenance_complete") is not True:
        return None
    rows = manifest.get("crops")
    if not isinstance(rows, list) or not rows:
        return None
    image_ids: set[int] = set()
    source_images: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") not in {
            "applied",
            "already_upright",
        }:
            return None
        try:
            image_id = int(row.get("image_id"))
        except (TypeError, ValueError):
            return None
        source_image = str(row.get("source_image") or "").strip()
        if image_id < 0 or image_id in image_ids or not source_image:
            return None
        image_ids.add(image_id)
        source_images.add(source_image)
    event_ids = {
        int(value)
        for value in event.get("crop_image_ids") or []
        if not isinstance(value, bool) and str(value).isdigit()
    }
    if image_ids != event_ids or len(source_images) != len(rows):
        return None
    return source_images


def _independent_event_reasons(
    event: Mapping[str, Any], prefix: str, minimum_confidence: float
) -> list[str]:
    reasons: list[str] = []
    if event.get("confirmation_eligible") is not True:
        reasons.append(f"vlm_{prefix}_not_confirmation_eligible")
    if str(event.get("semantic_vote_independence") or "") != "independent":
        reasons.append(f"vlm_{prefix}_not_explicitly_independent")
    if str(event.get("request_conditioning") or "") != "none_blind":
        reasons.append(f"vlm_{prefix}_request_not_unconditioned")
    if str(event.get("decision") or "").lower() != "keep":
        reasons.append(f"vlm_{prefix}_decision_not_keep")
    try:
        confidence = float(event.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if not np.isfinite(confidence) or confidence < minimum_confidence:
        reasons.append(f"vlm_{prefix}_confidence_below_threshold")
    contract = event.get("label_contract")
    assessment = assess_open_vocabulary_label(
        contract if isinstance(contract, Mapping) else None,
        minimum_confidence=minimum_confidence,
    )
    if not assessment.get("usable"):
        reasons.append(f"vlm_{prefix}_label_contract_invalid")
    elif _exact_category(assessment.get("category")) != _exact_category(
        event.get("category")
    ):
        reasons.append(f"vlm_{prefix}_event_contract_category_mismatch")
    image_ids = {
        int(value)
        for value in event.get("crop_image_ids") or []
        if not isinstance(value, bool) and str(value).isdigit()
    }
    if event.get("crop_image_ids_complete") is not True or len(image_ids) < 2:
        reasons.append(f"vlm_{prefix}_multiview_evidence_missing")
    fingerprint = str(event.get("evidence_fingerprint_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        reasons.append(f"vlm_{prefix}_evidence_fingerprint_invalid")
    if _event_source_images(event) is None:
        reasons.append(f"vlm_{prefix}_source_view_provenance_incomplete")
    return reasons


def _category_compatible(prior: object, blind: object) -> bool:
    left = _canonical_category(prior)
    right = _canonical_category(blind)
    if left in UNRESOLVED_CATEGORIES or right in UNRESOLVED_CATEGORIES:
        return True
    # Token containment is not semantic identity. Only exact normalized
    # matches or the explicit conservative aliases above are compatible.
    return left == right


FORWARD_COMPATIBLE_GEOMETRY_DEFAULTS = {
    "object_geometry_voxel_component_fraction": 0.0,
    "object_geometry_voxel_retention_gate_rate": 0.0,
    "object_geometry_voxel_retention_policy": "full_3d_voxel_retention",
    "object_geometry_planar_exception_applied": False,
    "object_geometry_planar_normal_axis": -1,
    "object_geometry_planar_independent_views": 0,
    "object_geometry_planar_normal_q90_degrees": float("inf"),
    "object_geometry_planar_median_view_thickness_m": float("inf"),
    "object_geometry_planar_evidence_reasons": [],
}


GEOMETRY_KEYS = (
    "active",
    "object_box_centers_m",
    "object_box_dimensions_m",
    "object_box_wxyz",
    "object_geometry_status",
    "object_geometry_inside_rate",
    "object_geometry_reprojection_error",
    "object_geometry_valid_observations",
    "object_geometry_consistent_observations",
    "object_geometry_projected_box_iou",
    "object_geometry_box_support_rate",
    "object_geometry_voxel_inside_rate",
    "object_geometry_voxel_component_fraction",
    "object_geometry_voxel_volume_expansion",
    "object_geometry_voxel_extent_candidate",
    "object_geometry_voxel_retention_gate_rate",
    "object_geometry_voxel_retention_policy",
    "object_geometry_planar_exception_applied",
    "object_geometry_planar_normal_axis",
    "object_geometry_planar_independent_views",
    "object_geometry_planar_normal_q90_degrees",
    "object_geometry_planar_median_view_thickness_m",
    "object_geometry_planar_evidence_reasons",
    "object_geometry_orientation_confidence",
    "object_geometry_gravity_tilt_degrees",
    "object_geometry_orientation_mode",
    "object_mask_observations",
    "object_image_ids",
    "viewpoint_image_ids",
)


def _load(path: Path) -> tuple[Any, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported scene state: {path}")
    return payload, state


def _array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_object_ids(path: Path) -> set[int]:
    values: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        object_id = int(value)
        if object_id < 0:
            raise ValueError("object IDs must be non-negative")
        values.add(object_id)
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_row(
    destination: dict, source: dict, key: str, index: int, count: int
) -> None:
    value = source.get(key)
    target = destination.get(key)
    if target is None and key in FORWARD_COMPATIBLE_GEOMETRY_DEFAULTS:
        default = FORWARD_COMPATIBLE_GEOMETRY_DEFAULTS[key]
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and value.shape[0] == count
        ):
            target = torch.full_like(value, default)
        elif (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == count
        ):
            target = np.full_like(value, default)
        elif isinstance(value, (list, tuple)) and len(value) == count:
            target = [copy.deepcopy(default) for _ in range(count)]
        else:
            raise ValueError(f"cannot initialize row-aligned rescue field {key!r}")
        destination[key] = target
    if isinstance(value, torch.Tensor) and isinstance(target, torch.Tensor):
        if (
            value.ndim >= 1
            and target.ndim >= 1
            and value.shape[0] == target.shape[0] == count
        ):
            target[index] = value[index]
            return
    if isinstance(value, np.ndarray) and isinstance(target, np.ndarray):
        if (
            value.ndim >= 1
            and target.ndim >= 1
            and value.shape[0] == target.shape[0] == count
        ):
            target[index] = value[index]
            return
    if isinstance(value, (list, tuple)) and isinstance(target, (list, tuple)):
        if len(value) == len(target) == count:
            mutable = list(target)
            mutable[index] = copy.deepcopy(value[index])
            destination[key] = mutable
            return
    raise ValueError(f"cannot copy row-aligned rescue field {key!r}")


def _assessed_label(
    row: Mapping[str, Any], minimum_confidence: float
) -> dict[str, Any]:
    contract = row.get("review_label_contract")
    if not isinstance(contract, Mapping):
        return {
            "usable": False,
            "category": "unresolved object",
            "description": "",
            "attributes": [],
            "reason_codes": ["label_contract_missing"],
        }
    return assess_open_vocabulary_label(
        contract, minimum_confidence=float(minimum_confidence)
    )


def _initial_vlm_geometry_gate(
    row: Mapping[str, Any], minimum_confidence: float
) -> tuple[bool, list[str]]:
    """Validate blind whole-object evidence independently of label verification."""

    reasons: list[str] = []
    events = [
        event
        for event in row.get("semantic_evidence") or []
        if isinstance(event, Mapping)
        and str(event.get("source") or "") in BLIND_EXISTENCE_SOURCES
    ]
    if len(events) != 1:
        return False, ["vlm_initial_blind_evidence_missing_or_duplicated"]
    event = events[0]
    if str(event.get("decision") or "").lower() != "keep":
        reasons.append("vlm_initial_decision_not_keep")
    confidence = float(event.get("confidence") or 0.0)
    if not np.isfinite(confidence) or confidence < minimum_confidence:
        reasons.append("vlm_initial_confidence_below_threshold")
    if event.get("confirmation_eligible") is not True:
        reasons.append("vlm_initial_evidence_not_confirmation_eligible")
    contract = event.get("label_contract")
    if not isinstance(contract, Mapping) or not bool(contract.get("contract_valid")):
        reasons.append("vlm_initial_label_contract_invalid")
    else:
        if contract.get("complete_bounded") is not True:
            reasons.append("vlm_initial_not_complete_bounded_object")
        if str(contract.get("topology") or "") not in {
            "standalone_whole",
            "carrier_payload",
        }:
            reasons.append("vlm_initial_topology_not_whole")
        if int(contract.get("diagnostic_view_count") or 0) < 2:
            reasons.append("vlm_initial_insufficient_diagnostic_views")
    image_ids = {
        int(value)
        for value in event.get("crop_image_ids") or []
        if isinstance(value, (int, np.integer)) or str(value).isdigit()
    }
    if len(image_ids) < 2:
        reasons.append("vlm_initial_multiview_evidence_missing")
    return not reasons, sorted(set(reasons))


def _vlm_gate(
    row: Mapping[str, Any],
    minimum_confidence: float,
    *,
    verification_mode: str = "",
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if verification_mode != "independent_blind":
        reasons.append("vlm_review_verification_mode_not_independent_blind")
    if str(row.get("review_decision") or "").lower() != "keep":
        reasons.append("vlm_decision_not_keep")
    confidence = float(row.get("review_confidence") or 0.0)
    if not np.isfinite(confidence) or confidence < minimum_confidence:
        reasons.append("vlm_confidence_below_threshold")
    contract = row.get("review_label_contract")
    assessment = _assessed_label(row, minimum_confidence)
    if not bool(assessment.get("usable")):
        reasons.extend(
            f"vlm_label_{value}"
            for value in assessment.get("reason_codes") or ["assessment_failed"]
        )
    if not isinstance(contract, Mapping) or not bool(contract.get("contract_valid")):
        reasons.append("vlm_label_contract_invalid")
    else:
        if contract.get("complete_bounded") is not True:
            reasons.append("vlm_not_complete_bounded_object")
        if str(contract.get("topology") or "") not in {
            "standalone_whole",
            "carrier_payload",
        }:
            reasons.append("vlm_topology_not_whole")
        if int(contract.get("diagnostic_view_count") or 0) < 2:
            reasons.append("vlm_insufficient_diagnostic_views")
    consensus = row.get("independent_blind_consensus")
    if (
        not isinstance(consensus, Mapping)
        or consensus.get("schema") != INDEPENDENT_BLIND_CONSENSUS_SCHEMA
    ):
        reasons.append("vlm_independent_blind_consensus_missing_or_invalid")
        consensus = {}
    else:
        if (
            consensus.get("accepted") is not True
            or str(consensus.get("status") or "") != "accepted"
        ):
            reasons.append("vlm_independent_blind_consensus_not_accepted")
    consensus_initial = _exact_category(consensus.get("initial_category"))
    consensus_verification = _exact_category(consensus.get("verification_category"))
    consensus_category = _exact_category(consensus.get("canonical_category"))
    if (
        not consensus_initial
        or not consensus_verification
        or not consensus_category
        or len({consensus_initial, consensus_verification, consensus_category}) != 1
    ):
        reasons.append("vlm_independent_blind_consensus_category_disagreement")
    if consensus_category in PUBLISH_FORBIDDEN_GENERIC_CATEGORIES:
        reasons.append("vlm_independent_blind_consensus_category_generic")

    evidence = [
        event
        for event in row.get("semantic_evidence") or []
        if isinstance(event, Mapping)
    ]
    initial_events = [
        event
        for event in evidence
        if str(event.get("source") or "") in BLIND_EXISTENCE_SOURCES
    ]
    verification_events = [
        event
        for event in evidence
        if str(event.get("source") or "") == "independent_verification"
    ]
    if len(evidence) != 2 or len(initial_events) != 1 or len(verification_events) != 1:
        reasons.append("vlm_two_pass_verification_missing")
    initial = initial_events[0] if len(initial_events) == 1 else {}
    verification = verification_events[0] if len(verification_events) == 1 else {}
    if initial:
        reasons.extend(
            _independent_event_reasons(initial, "initial", minimum_confidence)
        )
    if verification:
        reasons.extend(
            _independent_event_reasons(
                verification, "verification", minimum_confidence
            )
        )
    initial_category = _exact_event_category(initial)
    verification_category = _exact_event_category(verification)
    if (
        not initial_category
        or not verification_category
        or initial_category != verification_category
        or initial_category != consensus_category
    ):
        reasons.append("vlm_two_pass_category_disagreement")
    if (
        initial_category in PUBLISH_FORBIDDEN_GENERIC_CATEGORIES
        or verification_category in PUBLISH_FORBIDDEN_GENERIC_CATEGORIES
    ):
        reasons.append("vlm_category_too_generic_for_publication")

    candidate_category = _exact_category(assessment.get("category"))
    top_level_category = _exact_category(row.get("review_category"))
    if (
        candidate_category != consensus_category
        or top_level_category != consensus_category
    ):
        reasons.append("vlm_top_level_category_conflicts_blind_consensus")
    blind_category = consensus_initial or _exact_category(_blind_category(row))
    if blind_category in PUBLISH_FORBIDDEN_GENERIC_CATEGORIES:
        reasons.append("vlm_independent_blind_category_missing_or_generic")
    elif candidate_category in PUBLISH_FORBIDDEN_GENERIC_CATEGORIES:
        reasons.append("vlm_candidate_category_missing_or_generic")
    elif candidate_category != blind_category:
        reasons.append("vlm_candidate_category_conflicts_independent_blind")
    initial_ids = {
        int(value)
        for value in initial.get("crop_image_ids") or []
        if isinstance(value, (int, np.integer)) or str(value).isdigit()
    }
    verification_ids = {
        int(value)
        for value in verification.get("crop_image_ids") or []
        if isinstance(value, (int, np.integer)) or str(value).isdigit()
    }
    image_ids = initial_ids | verification_ids
    if len(image_ids) < 3:
        reasons.append("vlm_multiview_evidence_missing")
    if not initial_ids or not verification_ids or initial_ids & verification_ids:
        reasons.append("vlm_two_pass_views_not_independent")
    initial_event_id = str(initial.get("event_id") or "")
    verification_event_id = str(verification.get("event_id") or "")
    expected_initial_event_id = (
        f"object:{int(row.get('id', -1))}:{str(initial.get('source') or '')}"
    )
    expected_verification_event_id = (
        f"object:{int(row.get('id', -1))}:independent_verification"
    )
    consensus_initial_ids = {
        int(value)
        for value in consensus.get("initial_crop_image_ids") or []
        if not isinstance(value, bool) and str(value).isdigit()
    }
    consensus_verification_ids = {
        int(value)
        for value in consensus.get("verification_crop_image_ids") or []
        if not isinstance(value, bool) and str(value).isdigit()
    }
    if (
        consensus.get("evidence_pair_source")
        != "exact_current_semantic_evidence_events"
        or not initial_event_id
        or not verification_event_id
        or initial_event_id == verification_event_id
        or initial_event_id != expected_initial_event_id
        or verification_event_id != expected_verification_event_id
        or str(consensus.get("initial_event_id") or "") != initial_event_id
        or str(consensus.get("verification_event_id") or "")
        != verification_event_id
        or consensus_initial_ids != initial_ids
        or consensus_verification_ids != verification_ids
        or str(consensus.get("initial_evidence_fingerprint_sha256") or "")
        != str(initial.get("evidence_fingerprint_sha256") or "")
        or str(consensus.get("verification_evidence_fingerprint_sha256") or "")
        != str(verification.get("evidence_fingerprint_sha256") or "")
    ):
        reasons.append(
            "vlm_independent_blind_consensus_event_provenance_mismatch"
        )
    initial_sources = _event_source_images(initial)
    verification_sources = _event_source_images(verification)
    if (
        initial_sources is None
        or verification_sources is None
        or initial_sources.intersection(verification_sources)
    ):
        reasons.append("vlm_two_pass_physical_source_views_not_independent")
    fingerprints = {
        str(event.get("evidence_fingerprint_sha256") or "").strip()
        for event in (initial, verification)
        if str(event.get("evidence_fingerprint_sha256") or "").strip()
    }
    if len(fingerprints) == 1:
        reasons.append("vlm_two_pass_evidence_not_independent")
    physical = row.get("physical_form_assessment") or {}
    if isinstance(physical, Mapping) and physical.get("hard_veto") is True:
        reasons.append("vlm_prior_physical_form_hard_veto")
    recovery = row.get("physical_form_recovery_assessment") or {}
    if (
        isinstance(recovery, Mapping)
        and recovery.get("triggered") is True
        and recovery.get("accepted") is not True
    ):
        reasons.append("vlm_prior_physical_recovery_rejected")
    return not reasons, sorted(set(reasons))


def _independent_identity_re_adjudication(
    row: Mapping[str, Any],
    minimum_confidence: float,
    *,
    verification_mode: str,
) -> dict[str, Any]:
    """Recover raw blind identity without upgrading incomplete mask scope."""

    evidence = [
        event for event in row.get("semantic_evidence") or []
        if isinstance(event, Mapping)
    ]
    initial = [
        event for event in evidence
        if str(event.get("source") or "") in BLIND_EXISTENCE_SOURCES
    ]
    verification = [
        event for event in evidence
        if str(event.get("source") or "") == "independent_verification"
    ]
    if (
        verification_mode != "independent_blind"
        or len(evidence) != 2
        or len(initial) != 1
        or len(verification) != 1
    ):
        return {
            "schema": INDEPENDENT_BLIND_CONSENSUS_SCHEMA,
            "status": "unresolved",
            "accepted": False,
            "canonical_category": "",
            "reason_codes": ["exact_current_blind_event_pair_missing"],
            "identity_contract_source": "raw_label_contract",
            "mask_scope_incomplete": False,
            "mask_scope_reason_codes": ["mask_scope_unknown"],
            "geometry_refinement_required": False,
            "inpainting_eligible": False,
        }
    result = independent_blind_whole_identity_consensus(
        initial[0], verification[0],
        minimum_confidence=float(minimum_confidence),
    )
    reasons = [
        str(reason) for reason in result.get("reason_codes") or []
        if str(reason) != "exact_safe_canonical_noun_agreement"
    ]
    initial_sources = _event_source_images(initial[0])
    verification_sources = _event_source_images(verification[0])
    if (
        initial_sources is None
        or verification_sources is None
        or initial_sources.intersection(verification_sources)
    ):
        reasons.append("physical_source_views_not_independent")
    accepted = bool(result.get("accepted") is True and not reasons)
    result.update({
        "status": "accepted" if accepted else "unresolved",
        "accepted": accepted,
        "canonical_category": (
            str(result.get("canonical_category") or "") if accepted else ""
        ),
        "reason_codes": sorted(set(reasons)) or [
            "exact_safe_canonical_noun_agreement"
        ],
        "inpainting_eligible": bool(
            accepted and not result.get("mask_scope_incomplete")
        ),
    })
    return result


def _identity_assessment_from_events(
    row: Mapping[str, Any], minimum_confidence: float
) -> dict[str, Any]:
    for event in row.get("semantic_evidence") or []:
        if not isinstance(event, Mapping):
            continue
        contract = event.get("label_contract")
        if isinstance(contract, Mapping):
            assessment = assess_open_vocabulary_whole_identity(
                contract, minimum_confidence=float(minimum_confidence)
            )
            if assessment.get("usable"):
                return assessment
    return {
        "category": "unresolved object",
        "description": "",
        "attributes": [],
        "confidence": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--refined-state", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--prior-catalog", type=Path, required=True)
    parser.add_argument(
        "--geometry-refinement-id-file",
        type=Path,
        help="Exact router-approved IDs allowed to replace geometry.",
    )
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-vlm-confidence", type=float, default=0.70)
    parser.add_argument(
        "--preserve-prior-label-on-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the canonical prior label when geometry/existence pass but the "
            "candidate-conditioned label verification does not."
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()

    refinement_path = args.refinement.expanduser().resolve(strict=True)
    refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
    require_train_fit_refinement(
        refinement,
        refinement_path=refinement_path,
    )

    source_path = args.source_state.expanduser().resolve(strict=True)
    refined_path = args.refined_state.expanduser().resolve(strict=True)
    source_payload, source_state = _load(source_path)
    _, refined_state = _load(refined_path)
    payload = copy.deepcopy(source_payload)
    state = (
        payload["state"]
        if isinstance(payload, dict) and isinstance(payload.get("state"), dict)
        else payload
    )
    if not isinstance(state, dict):
        raise TypeError("copied source state is invalid")
    source_ids = _array(state["object_id"]).astype(np.int64).reshape(-1)
    refined_ids = _array(refined_state["object_id"]).astype(np.int64).reshape(-1)
    if not np.array_equal(source_ids, refined_ids):
        raise ValueError("source and refined state object IDs differ")
    count = int(source_ids.size)
    index_by_id = {int(value): index for index, value in enumerate(source_ids.tolist())}

    refined_candidates = {
        int(row["object_id"]): row
        for row in refinement.get("objects") or []
        if isinstance(row, dict) and bool(row.get("accepted"))
    }
    all_refined_candidate_ids = sorted(refined_candidates)
    geometry_allowlist_path: Path | None = None
    excluded_by_router_ids: list[int] = []
    if args.geometry_refinement_id_file is not None:
        geometry_allowlist_path = args.geometry_refinement_id_file.expanduser().resolve(
            strict=True
        )
        geometry_allowlist = _load_object_ids(geometry_allowlist_path)
        excluded_by_router_ids = sorted(set(refined_candidates) - geometry_allowlist)
        refined_candidates = {
            key: value
            for key, value in refined_candidates.items()
            if key in geometry_allowlist
        }
    geometry_path = args.geometry_report.expanduser().resolve(strict=True)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry.get("schema") != "farm.object-geometry-audit.v3":
        raise ValueError("invalid geometry report")
    geometry_by_id = {
        int(row["object_id"]): row
        for row in geometry.get("objects") or []
        if isinstance(row, dict)
    }
    review_path = args.review_report.expanduser().resolve(strict=True)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema") != "farm.open-vocabulary-crop-review.v2":
        raise ValueError("invalid VLM review report")
    if not bool(review.get("verification_enabled")):
        raise ValueError("full-COLMAP rescue requires the second VLM verification pass")
    if "request_error_count" not in review:
        raise ValueError("VLM rescue review has no transport-success contract")
    try:
        request_error_count = int(review["request_error_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid VLM rescue transport-success contract") from exc
    transport_error_fields = (
        "initial_request_error_ids",
        "verification_request_error_ids",
    )
    if any(
        not isinstance(review.get(field), list) for field in transport_error_fields
    ):
        raise ValueError("VLM rescue review has incomplete transport-success evidence")
    if request_error_count != 0 or any(
        review.get(field) for field in transport_error_fields
    ):
        raise ValueError("VLM rescue review contains transport failures")
    verification_mode = str(review.get("verification_mode") or "")
    review_by_id = {
        int(row["id"]): row
        for row in review.get("objects") or []
        if isinstance(row, dict)
    }
    prior_catalog_path = args.prior_catalog.expanduser().resolve(strict=True)
    prior_catalog = json.loads(prior_catalog_path.read_text(encoding="utf-8"))
    if not isinstance(prior_catalog, list):
        raise TypeError("prior catalog must be a JSON array")
    catalog_by_id = {
        int(row["id"]): copy.deepcopy(row)
        for row in prior_catalog
        if isinstance(row, dict)
    }

    if len(refined_state.get("images") or []) < len(source_state.get("images") or []):
        raise ValueError("refined state lost source images")
    state["images"] = copy.deepcopy(refined_state.get("images") or [])
    state["image_positions"] = copy.deepcopy(refined_state.get("image_positions") or [])

    statuses = ["not_planned"] * count
    evidence_rows: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    accepted_ids: list[int] = []
    label_verified_by_id: dict[int, bool] = {}
    semantic_identity_verified_by_id: dict[int, bool] = {}
    identity_re_adjudication_by_id: dict[int, dict[str, Any]] = {}
    semantic_conflict_by_id: dict[int, bool] = {}
    explicit_semantic_quarantine_by_id: dict[int, bool] = {}
    decisions: list[dict[str, Any]] = []
    for object_id in sorted(refined_candidates):
        index = index_by_id.get(object_id)
        reasons: list[str] = []
        label_reasons: list[str] = []
        prior_category = str(
            (catalog_by_id.get(object_id) or {}).get("category")
            or (state.get("object_category") or [""] * count)[index]
            if index is not None
            else ""
        )
        blind_category = ""
        identity_compatible: bool | None = None
        semantic_conflict = False
        explicit_semantic_quarantine = False
        archived_explicit_semantic_quarantine = False
        semantic_identity_verified = False
        identity_re_adjudication: dict[str, Any] = {
            "accepted": False,
            "mask_scope_incomplete": False,
            "geometry_refinement_required": False,
            "inpainting_eligible": False,
        }
        geometry_row = geometry_by_id.get(object_id)
        geometry_evidence_failures = (
            validate_serialized_obb_release_evidence(
                geometry_row.get("train_geometry_evidence")
            )
            if isinstance(geometry_row, Mapping)
            else ["geometry_release_evidence_missing_or_invalid"]
        )
        if (
            geometry_row is None
            or str(geometry_row.get("status") or "") != "geometry_pass"
            or geometry_evidence_failures
        ):
            reasons.append("geometry_not_pass")
            reasons.extend(geometry_evidence_failures)
        review_row = review_by_id.get(object_id)
        label_verified = False
        if review_row is None:
            reasons.append("vlm_review_missing")
        else:
            archived_explicit_semantic_quarantine = bool(
                review_row.get("semantic_quarantined") is True
                or str(review_row.get("semantic_publication_action") or "")
                == "quarantine_unresolved_semantics"
            )
            identity_re_adjudication = _independent_identity_re_adjudication(
                review_row,
                float(args.minimum_vlm_confidence),
                verification_mode=verification_mode,
            )
            semantic_identity_verified = bool(
                identity_re_adjudication.get("accepted") is True
                and identity_re_adjudication.get("mask_scope_incomplete") is True
            )
            quarantine_re_adjudicated = bool(
                semantic_identity_verified
                and str(review_row.get("review_gate_reason") or "")
                == "independent_blind_consensus_failed"
            )
            explicit_semantic_quarantine = bool(
                archived_explicit_semantic_quarantine
                and not quarantine_re_adjudicated
            )
            existence_pass, existence_reasons = _initial_vlm_geometry_gate(
                review_row, float(args.minimum_vlm_confidence)
            )
            if not existence_pass:
                reasons.extend(existence_reasons)
            label_verified, label_reasons = _vlm_gate(
                review_row,
                float(args.minimum_vlm_confidence),
                verification_mode=verification_mode,
            )
            blind_category = str(
                identity_re_adjudication.get("canonical_category")
                or _blind_category(review_row)
            )
            identity_compatible = _category_compatible(prior_category, blind_category)
            candidate_conflict = bool(
                SEMANTIC_CONFLICT_REASONS.intersection(label_reasons)
            )
            if explicit_semantic_quarantine:
                label_reasons.append("vlm_semantics_explicitly_quarantined")
                semantic_conflict = True
            if (
                existence_pass
                and not label_verified
                and not semantic_identity_verified
                and (candidate_conflict or not identity_compatible)
            ):
                semantic_conflict = True
                if not identity_compatible:
                    label_reasons.append(
                        "vlm_initial_category_conflicts_prior_identity_without_verified_relabel"
                    )
        accepted = index is not None and not reasons
        if accepted:
            for key in GEOMETRY_KEYS:
                _copy_row(state, refined_state, key, index, count)
            if label_verified:
                assessment = _assessed_label(
                    review_row, float(args.minimum_vlm_confidence)
                )
                category = str(assessment["category"])
                description = str(assessment.get("description") or "")
                attributes = [
                    str(value).strip()
                    for value in assessment.get("attributes") or []
                    if str(value).strip()
                ][:8]
                semantic_tier = "probable"
                semantic_status = "full_colmap_sam3_vlm_rescued"
                statuses[index] = "accepted_label_verified"
            elif semantic_identity_verified:
                assessment = _identity_assessment_from_events(
                    review_row, float(args.minimum_vlm_confidence)
                )
                category = str(identity_re_adjudication["canonical_category"])
                description = str(assessment.get("description") or "")
                attributes = [
                    str(value).strip()
                    for value in assessment.get("attributes") or []
                    if str(value).strip()
                ][:8]
                semantic_tier = "probable"
                semantic_status = (
                    "full_colmap_identity_verified_mask_scope_refinement_required"
                )
                statuses[index] = (
                    "accepted_identity_verified_scope_refinement_required"
                )
            elif explicit_semantic_quarantine:
                category = "unresolved object"
                description = (
                    "Validated whole physical object with multi-view SAM3/RGB-D "
                    "geometry; independent blind semantics are quarantined."
                )
                attributes = []
                semantic_tier = "geometry_only"
                semantic_status = (
                    "full_colmap_sam3_geometry_rescued_semantic_quarantined"
                )
                statuses[index] = "accepted_geometry_only"
            elif semantic_conflict or args.preserve_prior_label_on_uncertainty:
                prior_row = catalog_by_id.get(object_id, {})
                category = normalize_open_noun(
                    prior_row.get("category")
                    or state.get("object_category", [""] * count)[index]
                )
                description = str(
                    prior_row.get("description")
                    or state.get("object_caption", [""] * count)[index]
                )
                attributes = list(
                    prior_row.get("attributes")
                    or state.get("object_key_attributes", [[]] * count)[index]
                    or []
                )[:8]
                semantic_tier = str(
                    prior_row.get("semantic_tier")
                    or state.get("object_semantic_tier", ["probable"] * count)[index]
                )
                semantic_status = str(
                    prior_row.get("semantic_status")
                    or state.get("object_semantic_status", ["prior_semantics"] * count)[
                        index
                    ]
                )
                statuses[index] = "accepted_geometry_prior_semantics_preserved"
            else:
                category = "unresolved object"
                description = (
                    "Validated whole physical object with multi-view SAM3/RGB-D geometry; "
                    "the candidate-conditioned label audit remained uncertain."
                )
                attributes = []
                semantic_tier = "geometry_only"
                semantic_status = (
                    "full_colmap_sam3_geometry_rescued_semantic_conflict"
                    if semantic_conflict
                    else "full_colmap_sam3_geometry_rescued"
                )
                statuses[index] = "accepted_geometry_only"
            for key, value in (
                ("object_category", category),
                ("object_caption", description),
                ("object_key_attributes", attributes),
                ("object_semantic_tier", semantic_tier),
                ("object_semantic_status", semantic_status),
            ):
                values = list(state.get(key) or ["" for _ in range(count)])
                if len(values) != count:
                    raise ValueError(f"semantic field {key!r} is not object-aligned")
                values[index] = value
                state[key] = values
            evidence_rows[index] = copy.deepcopy(
                review_row.get("semantic_evidence") or []
            )
            accepted_ids.append(object_id)
            label_verified_by_id[object_id] = bool(label_verified)
            semantic_identity_verified_by_id[object_id] = bool(
                semantic_identity_verified
            )
            identity_re_adjudication_by_id[object_id] = copy.deepcopy(
                identity_re_adjudication
            )
            semantic_conflict_by_id[object_id] = bool(semantic_conflict)
            explicit_semantic_quarantine_by_id[object_id] = bool(
                explicit_semantic_quarantine
            )
        elif index is not None:
            statuses[index] = "rejected_preserved_original"
        decisions.append(
            {
                "object_id": object_id,
                "accepted": bool(accepted),
                "label_verified": bool(label_verified),
                "semantic_identity_verified": bool(semantic_identity_verified),
                "semantic_identity_re_adjudication": identity_re_adjudication,
                "mask_scope_incomplete": bool(
                    semantic_identity_verified
                    and identity_re_adjudication.get("mask_scope_incomplete")
                ),
                "geometry_refinement_required": bool(
                    semantic_identity_verified
                    and identity_re_adjudication.get(
                        "geometry_refinement_required"
                    )
                ),
                "inpainting_eligible": False,
                "status": statuses[index] if index is not None else "unknown_object_id",
                "reasons": sorted(set(reasons)),
                "label_reasons": sorted(set(label_reasons)),
                "refined_views": int(
                    refined_candidates[object_id].get("accepted_views") or 0
                ),
                "geometry_status": str((geometry_row or {}).get("status") or "missing"),
                "prior_category": prior_category,
                "blind_category": blind_category,
                "blind_prior_identity_compatible": identity_compatible,
                "semantic_conflict_quarantined": bool(semantic_conflict),
                "explicit_semantic_quarantine": bool(
                    explicit_semantic_quarantine
                ),
                "archived_explicit_semantic_quarantine": bool(
                    archived_explicit_semantic_quarantine
                ),
                "vlm_category": str(
                    (review_row or {}).get("review_category") or "unknown"
                ),
                "vlm_confidence": float(
                    (review_row or {}).get("review_confidence") or 0.0
                ),
            }
        )
    state["object_full_colmap_rescue_status"] = statuses
    state["object_full_colmap_rescue_semantic_evidence"] = evidence_rows

    for object_id in accepted_ids:
        index = index_by_id[object_id]
        review_row = review_by_id[object_id]
        label_verified = bool(label_verified_by_id.get(object_id))
        semantic_identity_verified = bool(
            semantic_identity_verified_by_id.get(object_id)
        )
        identity_re_adjudication = identity_re_adjudication_by_id.get(
            object_id, {}
        )
        semantic_conflict = bool(semantic_conflict_by_id.get(object_id))
        explicit_semantic_quarantine = bool(
            explicit_semantic_quarantine_by_id.get(object_id)
        )
        if label_verified:
            assessment = _assessed_label(review_row, float(args.minimum_vlm_confidence))
            category = str(assessment["category"])
            description = str(assessment.get("description") or "")
            attributes = list(assessment.get("attributes") or [])[:8]
            semantic_tier = "probable"
            semantic_status = "full_colmap_sam3_vlm_rescued"
            semantic_gate_reason = "geometry_sam3_and_two_pass_label_agreement"
            publication_decision = "keep"
            publication_confidence = float(review_row.get("review_confidence") or 0.0)
        elif semantic_identity_verified:
            assessment = _identity_assessment_from_events(
                review_row, float(args.minimum_vlm_confidence)
            )
            category = str(identity_re_adjudication["canonical_category"])
            description = str(assessment.get("description") or "")
            attributes = list(assessment.get("attributes") or [])[:8]
            semantic_tier = "probable"
            semantic_status = (
                "full_colmap_identity_verified_mask_scope_refinement_required"
            )
            semantic_gate_reason = (
                "independent_blind_identity_verified_mask_scope_incomplete"
            )
            publication_decision = "keep"
            publication_confidence = min(
                (
                    float((event.get("label_contract") or {}).get("confidence") or 0.0)
                    for event in review_row.get("semantic_evidence") or []
                    if isinstance(event, Mapping)
                ),
                default=0.0,
            )
        elif explicit_semantic_quarantine:
            category = "unresolved object"
            description = (
                "Validated whole physical object with multi-view SAM3/RGB-D "
                "geometry; independent blind semantics are quarantined."
            )
            attributes = []
            semantic_tier = "geometry_only"
            semantic_status = (
                "full_colmap_sam3_geometry_rescued_semantic_quarantined"
            )
            semantic_gate_reason = (
                "geometry_rescued_explicit_semantic_quarantine"
            )
            publication_decision = "unknown"
            publication_confidence = 0.0
        elif semantic_conflict or args.preserve_prior_label_on_uncertainty:
            prior_row = catalog_by_id.get(object_id, {})
            category = normalize_open_noun(
                prior_row.get("category")
                or state.get("object_category", [""] * count)[index]
            )
            description = str(
                prior_row.get("description")
                or state.get("object_caption", [""] * count)[index]
            )
            attributes = list(
                prior_row.get("attributes")
                or state.get("object_key_attributes", [[]] * count)[index]
                or []
            )[:8]
            semantic_tier = str(
                prior_row.get("semantic_tier")
                or state.get("object_semantic_tier", ["probable"] * count)[index]
            )
            semantic_status = str(
                prior_row.get("semantic_status")
                or state.get("object_semantic_status", ["prior_semantics"] * count)[
                    index
                ]
            )
            semantic_gate_reason = (
                "geometry_rescued_semantic_conflict_prior_preserved"
                if semantic_conflict
                else "geometry_rescued_prior_semantics_preserved"
            )
            prior_decision = str(prior_row.get("review_decision") or "").lower()
            publication_decision = (
                prior_decision if prior_decision in {"keep", "relabel"} else "unknown"
            )
            publication_confidence = float(prior_row.get("review_confidence") or 0.0)
        else:
            category = "unresolved object"
            description = (
                "Validated whole physical object with multi-view SAM3/RGB-D geometry; "
                "semantic identity requires review."
            )
            attributes = []
            semantic_tier = "geometry_only"
            semantic_status = (
                "full_colmap_sam3_geometry_rescued_semantic_conflict"
                if semantic_conflict
                else "full_colmap_sam3_geometry_rescued"
            )
            semantic_gate_reason = (
                "geometry_rescued_semantic_conflict_quarantined"
                if semantic_conflict
                else "geometry_sam3_and_blind_whole_object_vlm"
            )
            publication_decision = "unknown"
            publication_confidence = float(review_row.get("review_confidence") or 0.0)
        center_m = _array(state["object_box_centers_m"])[index].astype(float)
        dimensions_m = _array(state["object_box_dimensions_m"])[index].astype(float)
        wxyz = _array(state["object_box_wxyz"])[index].astype(float)
        if (
            center_m.shape != (3,)
            or dimensions_m.shape != (3,)
            or wxyz.shape != (4,)
            or not np.isfinite(center_m).all()
            or not np.isfinite(dimensions_m).all()
            or not np.isfinite(wxyz).all()
            or np.any(dimensions_m <= 0.0)
            or float(np.linalg.norm(wxyz)) <= 1.0e-12
        ):
            raise ValueError(f"accepted object {object_id} has invalid canonical OBB")
        wxyz = wxyz / float(np.linalg.norm(wxyz))
        row = catalog_by_id.get(object_id, {"id": object_id})
        row.update(
            {
                "observations": len(
                    set(map(int, state["object_image_ids"][index] or []))
                ),
                "position_world_m": center_m.tolist(),
                "center_m": center_m.tolist(),
                "cov6": _array(state["cov6"])[index].astype(float).tolist(),
                "metric_dimensions_m": dimensions_m.tolist(),
                "dimensions_m": dimensions_m.tolist(),
                "wxyz": wxyz.tolist(),
                "category": category,
                "description": description,
                "attributes": attributes,
                "semantic_tier": semantic_tier,
                "semantic_status": semantic_status,
                "review_decision": publication_decision,
                "review_confidence": publication_confidence,
                "label_publication_eligible": bool(label_verified),
                "semantic_identity_verified": semantic_identity_verified,
                "semantic_identity_re_adjudication": copy.deepcopy(
                    identity_re_adjudication
                ),
                "mask_scope_incomplete": bool(
                    semantic_identity_verified
                    and identity_re_adjudication.get("mask_scope_incomplete")
                ),
                "mask_scope_reason_codes": list(
                    identity_re_adjudication.get("mask_scope_reason_codes") or []
                ),
                "geometry_refinement_required": bool(
                    semantic_identity_verified
                    and identity_re_adjudication.get(
                        "geometry_refinement_required"
                    )
                ),
                "geometry_refinement_route": (
                    "mask_geometry_refinement"
                    if semantic_identity_verified else "none"
                ),
                "inpainting_eligible": False,
                "semantic_conflict_quarantined": bool(
                    semantic_conflict or explicit_semantic_quarantine
                ),
                "explicit_semantic_quarantine": explicit_semantic_quarantine,
                "semantic_gate_reason": semantic_gate_reason,
                "negative_evidence_sources": [],
                "candidates": list(row.get("candidates") or [])
                + list(review_row.get("semantic_evidence") or []),
            }
        )
        catalog_by_id[object_id] = row
    catalog = [catalog_by_id[key] for key in sorted(catalog_by_id)]

    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        payload["state"] = state
        payload["saved_unix_s"] = time.time()
        payload.setdefault("meta", {})["full_colmap_rescue_acceptance"] = {
            "accepted_object_ids": accepted_ids,
            "policy": "sam3_geometry_and_blind_vlm_existence_with_separate_label_verification",
            "review_sha256": _sha256(review_path),
        }
    else:
        payload = state
    output_state = args.output_state.expanduser().resolve()
    output_state.parent.mkdir(parents=True, exist_ok=True)
    temporary_state = output_state.with_suffix(".pt.tmp")
    torch.save(payload, temporary_state)
    temporary_state.replace(output_state)
    output_catalog = args.output_catalog.expanduser().resolve()
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_catalog, catalog)

    report = {
        "schema": "farm.full-colmap-rescue-acceptance.v1",
        "status": "PASS",
        "policy": {
            "rest3d_inspired_vlm_mask_refinement": True,
            "sam3_mask_refinement": True,
            "multi_view_rgbd_geometry_required": True,
            "blind_vlm_whole_object_review_required": True,
            "independent_blind_review_report_required_for_label_publication": True,
            "independent_blind_consensus_contract_required": True,
            "independent_blind_consensus_exact_event_provenance_required": True,
            "vlm_transport_success_required": True,
            "two_pass_exact_safe_category_consensus_required": True,
            "category_token_containment_is_not_identity": True,
            "generic_category_publication_forbidden": True,
            "candidate_conditioned_verification_can_publish_label": False,
            "candidate_conditioned_verification_can_erase_geometry": False,
            "unverified_relabel_can_replace_geometry": False,
            "uncertain_label_becomes_geometry_only": not bool(
                args.preserve_prior_label_on_uncertainty
            ),
            "uncertain_label_preserves_prior_semantics": bool(
                args.preserve_prior_label_on_uncertainty
            ),
            "uncertain_existence_preserves_original": True,
            "semantic_conflict_quarantines_label_without_erasing_geometry": True,
            "explicit_review_quarantine_overrides_prior_label_preservation": True,
            "raw_identity_consensus_is_separate_from_mask_scope": True,
            "incomplete_mask_scope_blocks_label_release_and_inpainting": True,
            "semantic_conflict_preserves_prior_semantics": True,
            "minimum_vlm_confidence": float(args.minimum_vlm_confidence),
            "review_verification_mode": verification_mode,
        },
        "candidate_objects": len(refined_candidates),
        "refinement_accepted_candidate_ids": all_refined_candidate_ids,
        "router_geometry_allowlist_path": (
            str(geometry_allowlist_path) if geometry_allowlist_path else None
        ),
        "router_excluded_candidate_ids": excluded_by_router_ids,
        "router_filtered_candidate_objects": len(refined_candidates),
        "accepted_objects": len(accepted_ids),
        "accepted_object_ids": accepted_ids,
        "label_verified_objects": sum(label_verified_by_id.values()),
        "label_verified_object_ids": [
            object_id
            for object_id in accepted_ids
            if label_verified_by_id.get(object_id)
        ],
        "semantic_identity_verified_scope_refinement_objects": sum(
            semantic_identity_verified_by_id.values()
        ),
        "semantic_identity_verified_scope_refinement_object_ids": [
            object_id for object_id in accepted_ids
            if semantic_identity_verified_by_id.get(object_id)
        ],
        "geometry_only_objects": sum(
            statuses[index_by_id[object_id]] == "accepted_geometry_only"
            for object_id in accepted_ids
        ),
        "geometry_only_object_ids": [
            object_id
            for object_id in accepted_ids
            if statuses[index_by_id[object_id]] == "accepted_geometry_only"
        ],
        "semantic_conflict_geometry_only_objects": sum(
            statuses[index_by_id[object_id]] == "accepted_geometry_only"
            and semantic_conflict_by_id.get(object_id, False)
            for object_id in accepted_ids
        ),
        "semantic_conflict_geometry_only_object_ids": [
            object_id
            for object_id in accepted_ids
            if statuses[index_by_id[object_id]] == "accepted_geometry_only"
            and semantic_conflict_by_id.get(object_id, False)
        ],
        "prior_semantics_preserved_objects": sum(
            statuses[index_by_id[object_id]]
            == "accepted_geometry_prior_semantics_preserved"
            for object_id in accepted_ids
        ),
        "prior_semantics_preserved_object_ids": [
            object_id
            for object_id in accepted_ids
            if statuses[index_by_id[object_id]]
            == "accepted_geometry_prior_semantics_preserved"
        ],
        "rejected_objects": len(refined_candidates) - len(accepted_ids),
        "decisions": decisions,
        "timing_seconds": time.perf_counter() - started,
        "hashes": {
            "source_state_sha256": _sha256(source_path),
            "refined_state_sha256": _sha256(refined_path),
            "refinement_sha256": _sha256(refinement_path),
            "geometry_report_sha256": _sha256(geometry_path),
            "review_report_sha256": _sha256(review_path),
            "geometry_refinement_id_file_sha256": (
                _sha256(geometry_allowlist_path) if geometry_allowlist_path else None
            ),
            "output_state_sha256": _sha256(output_state),
            "output_catalog_sha256": _sha256(output_catalog),
        },
    }
    output_report = args.output_report.expanduser().resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
