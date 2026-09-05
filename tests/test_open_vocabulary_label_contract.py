from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

from scene_graph.captioning.evidence import crop_evidence_manifest
from scene_graph.captioning.label_contract import (
    OPEN_VOCABULARY_LABEL_SCHEMA,
    adaptive_alternate_plan,
    assess_open_vocabulary_identity,
    assess_whole_object_readiness,
    assess_open_vocabulary_label,
    parse_open_vocabulary_label,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, relative: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract(
    category: str = "storage rack",
    *,
    category_role: str = "whole_object",
    form_hypernym: str = "support frame",
    primitive_form: str = "rectangular prism",
    primitive: bool = False,
    specificity: str = "object_kind",
    topology: str = "standalone_whole",
    complete_bounded: bool | None = True,
    carrier_category: str = "unknown",
    payload_categories: list[str] | None = None,
    shape_profile: str = "open_frame",
    identity_basis: str = "diagnostic_geometry",
    diagnostic_parts: list[str] | None = None,
    views: int = 3,
    confidence: float = 0.95,
    decision: str = "keep",
) -> dict:
    return parse_open_vocabulary_label({
        "category": category,
        "category_role": category_role,
        "form_hypernym": form_hypernym,
        "primitive_form": primitive_form,
        "head_noun_is_primitive": primitive,
        "specificity": specificity,
        "topology": topology,
        "complete_bounded": complete_bounded,
        "carrier_category": carrier_category,
        "payload_categories": payload_categories or [],
        "shape_profile": shape_profile,
        "identity_basis": identity_basis,
        "diagnostic_parts": (
            ["repeated uprights", "crossbars"]
            if diagnostic_parts is None
            else diagnostic_parts
        ),
        "diagnostic_view_count": views,
        "description": "bounded target-owned structure visible across views",
        "attributes": [],
        "confidence": confidence,
        "decision": decision,
    })


def _event(
    source: str,
    contract: dict,
    image_ids: tuple[int, ...],
    angles: tuple[float, ...],
) -> dict:
    assert len(image_ids) == len(angles)
    pose_index = {}
    crops = []
    for image_id, degrees in zip(image_ids, angles):
        angle = math.radians(degrees)
        centre = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        pose_index[image_id] = {
            "camera_center_world_m": centre,
            "camera_forward_world": [-centre[0] / 10.0, -centre[1] / 10.0, 0.0],
        }
        crops.append((Path(f"img_{image_id:06d}_det_0001.npz"), b"jpeg"))
    manifest = crop_evidence_manifest(
        7,
        crops,
        frame_pose_index=pose_index,
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    return {
        "source": source,
        "event_id": f"event:{source}",
        "category": contract["category"],
        "description": contract["description"],
        "attributes": contract["attributes"],
        "confidence": contract["confidence"],
        "decision": contract["decision"],
        "label_contract": contract,
        "confirmation_eligible": True,
        **manifest,
    }


def test_complete_diagnostic_whole_object_is_usable() -> None:
    result = assess_open_vocabulary_label(_contract())
    assert result["usable"] is True
    assert result["category"] == "storage rack"
    assert result["maximum_semantic_tier"] == "confirmed"

def test_obvious_open_noun_spelling_error_is_normalized() -> None:
    contract = _contract("palet", form_hypernym="container")
    assert contract["category"] == "pallet"
    assert assess_open_vocabulary_label(contract)["category"] == "pallet"


def test_whole_object_readiness_requires_explicit_complete_mask_evidence() -> None:
    legacy = assess_whole_object_readiness(_contract())
    assert legacy["ready"] is False
    assert "missing_whole_object_evidence" in legacy["reason_codes"]

    contract = _contract()
    contract.update({
        "context_sufficient": True,
        "visible_target_coverage": 0.94,
        "missing_visible_parts": [],
        "included_non_target": [],
    })
    parsed = parse_open_vocabulary_label(contract)
    result = assess_whole_object_readiness(parsed)
    assert parsed["whole_object_evidence_complete"] is True
    assert result["ready"] is True


def test_partial_mask_keeps_identity_only_for_targeted_refinement() -> None:
    contract = _contract(topology="partial_unbounded", complete_bounded=False)
    contract.update({
        "context_sufficient": True,
        "visible_target_coverage": 0.52,
        "missing_visible_parts": ["lower frame", "rear uprights"],
        "included_non_target": [],
    })
    parsed = parse_open_vocabulary_label(contract)
    identity = assess_open_vocabulary_identity(parsed)
    whole = assess_whole_object_readiness(parsed)
    assert identity["usable"] is True
    assert identity["category"] == "storage rack"
    assert identity["presentation_safe"] is False
    assert identity["maximum_semantic_tier"] == "probable"
    assert identity["description"]
    assert whole["ready"] is False
    assert "visible_target_coverage_below_threshold" in whole["reason_codes"]
    assert "mask_missing_visible_target_parts" in whole["reason_codes"]


def test_partial_identity_survives_adjudication_at_probable_only() -> None:
    semantic = _load_script(
        "farm_open_vocab_adjudicate_partial_identity",
        "scripts/semantics/adjudicate_farm_semantics.py",
    )
    contract = _contract(
        topology="partial_unbounded", complete_bounded=False,
    )
    contract.update({
        "context_sufficient": True,
        "visible_target_coverage": 0.55,
        "missing_visible_parts": ["rear uprights"],
        "included_non_target": [],
    })
    events = [
        _event("blind_pass_a", contract, (1, 2, 3), (0, 20, 40)),
        _event("blind_pass_c", contract, (4, 5, 6), (120, 140, 160)),
    ]
    tier, chosen, reason, negatives = semantic.adaptive_open_vocabulary_decision(
        events, 0.90
    )
    assert tier == "probable"
    assert chosen is not None and chosen["category"] == "storage rack"
    assert chosen["confirmation_eligible"] is False
    assert reason == "correlated_exact_open_vocabulary_agreement"
    assert negatives == []




def test_unsupported_specific_identity_downgrades_to_explicit_form() -> None:
    result = assess_open_vocabulary_label(_contract(
        "specialized storage machine",
        specificity="specific_identity",
        identity_basis="context_only",
        diagnostic_parts=[],
    ))
    assert result["usable"] is True
    assert result["category"] == "support frame"
    assert result["category_role"] == "whole_form"
    assert result["maximum_semantic_tier"] == "probable"
    assert result["description"] == "bounded visible support frame"
    assert result["attributes"] == []


def test_lexical_agreement_cannot_rescue_context_only_identity() -> None:
    contract = _contract(
        "specialized storage machine",
        specificity="specific_identity",
        form_hypernym="unknown",
        identity_basis="context_only",
        diagnostic_parts=[],
    )
    semantic = _load_script(
        "farm_open_vocab_adjudicate_bad_identity",
        "scripts/semantics/adjudicate_farm_semantics.py",
    )
    events = [
        _event("blind_pass_a", contract, (1, 2, 3), (0, 20, 40)),
        _event("blind_pass_c", contract, (4, 5, 6), (120, 140, 160)),
    ]
    tier, chosen, reason, _ = semantic.adaptive_open_vocabulary_decision(
        events, 0.90
    )
    assert tier == "geometry_only"
    assert chosen is None
    assert reason == "open_vocabulary_identity_unresolved"


def test_primitive_head_and_attached_component_fail_closed() -> None:
    primitive = assess_open_vocabulary_label(_contract(
        "cylindrical shape",
        category_role="primitive_form",
        primitive_form="cylindrical shape",
        primitive=True,
        specificity="generic_form",
        shape_profile="compact_volumetric",
    ))
    attached = assess_open_vocabulary_label(_contract(
        "control module",
        category_role="standalone_component",
        topology="attached_component",
        complete_bounded=False,
    ))
    assert primitive["usable"] is False
    assert "label_head_noun_is_geometric_primitive" in primitive["reason_codes"]
    assert attached["usable"] is False
    assert "label_topology_attached_component" in attached["reason_codes"]


def test_carrier_payload_names_carrier_and_is_never_confirmed() -> None:
    result = assess_open_vocabulary_label(_contract(
        "mobile carrier",
        category_role="carrier",
        form_hypernym="carrier",
        specificity="object_kind",
        topology="carrier_payload",
        carrier_category="mobile carrier",
        payload_categories=["panel", "coil"],
        shape_profile="articulated",
        diagnostic_parts=["wheels", "load deck"],
    ))
    assert result["usable"] is True
    assert result["category"] == "mobile carrier"
    assert result["maximum_semantic_tier"] == "probable"
    assert "carrying panel, coil" in result["description"]


def test_missing_contract_fields_fail_closed() -> None:
    parsed = parse_open_vocabulary_label({"category": "storage rack"})
    result = assess_open_vocabulary_label(parsed)
    assert parsed["schema"] == OPEN_VOCABULARY_LABEL_SCHEMA
    assert parsed["contract_valid"] is False
    assert result["usable"] is False
    assert "label_contract_invalid" in result["reason_codes"]


def test_diverse_exact_blind_events_confirm() -> None:
    semantic = _load_script(
        "farm_open_vocab_adjudicate_diverse",
        "scripts/semantics/adjudicate_farm_semantics.py",
    )
    contract = _contract()
    events = [
        _event("blind_pass_a", contract, (1, 2, 3), (0, 20, 40)),
        _event("blind_pass_c", contract, (4, 5, 6), (120, 140, 160)),
    ]
    tier, chosen, reason, negatives = semantic.adaptive_open_vocabulary_decision(
        events, 0.90
    )
    assert tier == "confirmed"
    assert chosen is not None and chosen["category"] == "storage rack"
    assert reason == "independent_disjoint_view_consensus"
    assert negatives == []


def test_disjoint_ids_with_near_duplicate_poses_do_not_confirm() -> None:
    semantic = _load_script(
        "farm_open_vocab_adjudicate_correlated",
        "scripts/semantics/adjudicate_farm_semantics.py",
    )
    contract = _contract()
    events = [
        _event("blind_pass_a", contract, (1, 2, 3), (0.0, 0.2, 0.4)),
        _event("blind_pass_c", contract, (4, 5, 6), (1.0, 1.2, 1.4)),
    ]
    tier, _, reason, _ = semantic.adaptive_open_vocabulary_decision(events, 0.90)
    assert tier == "probable"
    assert reason == "correlated_exact_open_vocabulary_agreement"


def test_disagreement_uses_only_explicit_shared_form_hypernym() -> None:
    semantic = _load_script(
        "farm_open_vocab_adjudicate_hypernym",
        "scripts/semantics/adjudicate_farm_semantics.py",
    )
    left = _contract("storage rack", form_hypernym="support frame")
    right = _contract("equipment stand", form_hypernym="support frame")
    events = [
        _event("blind_pass_a", left, (1, 2, 3), (0, 20, 40)),
        _event("blind_pass_c", right, (4, 5, 6), (120, 140, 160)),
    ]
    tier, chosen, reason, _ = semantic.adaptive_open_vocabulary_decision(
        events, 0.90
    )
    assert tier == "probable"
    assert chosen is not None and chosen["category"] == "support frame"
    assert reason == "explicit_shared_form_hypernym"


def test_shared_form_hypernym_survives_pipeline_at_probable_only() -> None:
    adjudicator = _load_script(
        "farm_open_vocab_adjudicate_hypernym_e2e",
        "scripts/semantics/adjudicate_farm_semantics.py",
    )
    left = _contract("storage rack", form_hypernym="support frame")
    right = _contract("equipment stand", form_hypernym="support frame")
    events = [
        _event("blind_pass_a", left, (1, 2, 3), (0, 20, 40)),
        _event("blind_pass_c", right, (4, 5, 6), (120, 140, 160)),
    ]
    tier, chosen, reason, negatives = adjudicator.adaptive_open_vocabulary_decision(
        events, 0.90
    )
    ensemble = adjudicator.output_row(
        {"id": 7}, chosen, tier, events, reason, negatives, events
    )
    derived = [
        event for event in ensemble["semantic_evidence"]
        if event.get("source") == "explicit_shared_form_hypernym_consensus"
    ]
    assert len(derived) == 1
    assert derived[0]["confirmation_eligible"] is False
    assert len(derived[0]["derived_from_event_ids"]) == 2

    reconciler = _load_script(
        "farm_open_vocab_reconcile_hypernym_e2e",
        "scripts/semantics/reconcile_farm_semantics.py",
    )
    evidence = reconciler._candidate_evidence(ensemble, {})
    reconciled = reconciler._strict_current_resolution(ensemble, evidence, {})
    assert reconciled is not None
    assert reconciled["category"] == "support frame"
    assert reconciled["semantic_tier"] == "probable"

    finalizer = _load_script(
        "farm_open_vocab_finalizer_hypernym_e2e",
        "scripts/semantics/finalize_farm_semantic_consensus.py",
    )
    category, _ = finalizer.choose_category({}, {}, ensemble, reconciled, {})
    final_evidence = finalizer.collect_semantic_evidence(
        {}, {}, ensemble, reconciled
    )
    final_tier, supporting, assessment = finalizer.semantic_tier_assessment(
        category, final_evidence
    )
    assert category == "support frame"
    assert final_tier == "probable"
    assert len(supporting) == 1
    assert assessment["confirmation_eligible"] is False

    from farm_pipeline.final_acceptance import _eligible_candidate_votes

    _, structured = _eligible_candidate_votes(ensemble)
    assert "support frame" in structured


def test_adaptive_planner_requests_only_low_or_pose_confirmable() -> None:
    contract = _contract()
    initial = _event("blind_pass_a", contract, (1, 2, 3), (0, 20, 40))
    diverse = _event("blind_pass_c", contract, (4, 5, 6), (120, 140, 160))
    correlated = _event("blind_pass_c", contract, (4, 5, 6), (1, 21, 41))
    low = _event(
        "blind_pass_a", _contract(confidence=0.86), (1, 2, 3), (0, 20, 40)
    )
    assert adaptive_alternate_plan(initial, diverse)["reason"] == "pose_confirmable"
    assert adaptive_alternate_plan(initial, correlated)["request_alternate"] is False
    assert adaptive_alternate_plan(low, correlated)["reason"] == "unresolved_or_low_confidence"


def test_finalizer_requires_exact_strict_head_and_honors_tier_cap() -> None:
    finalizer = _load_script(
        "farm_open_vocab_finalizer",
        "scripts/semantics/finalize_farm_semantic_consensus.py",
    )
    contract = _contract(
        "mobile carrier",
        category_role="carrier",
        form_hypernym="carrier",
        topology="carrier_payload",
        carrier_category="mobile carrier",
        payload_categories=["panel"],
        shape_profile="articulated",
        diagnostic_parts=["wheels", "load deck"],
    )
    events = [
        _event("blind_pass_a", contract, (1, 2, 3), (0, 20, 40)),
        _event("blind_pass_c", contract, (4, 5, 6), (120, 140, 160)),
    ]
    assert finalizer.category_supported_by_event("carrier", events[0]) is False
    tier, _, assessment = finalizer.semantic_tier_assessment(
        "mobile carrier",
        events,
        physical_assessment={"confirmation_allowed": True},
    )
    assert tier == "probable"
    assert assessment["confirmation_ready"] is True


def test_finalizer_never_resurrects_prior_after_strict_failure() -> None:
    finalizer = _load_script(
        "farm_open_vocab_finalizer_fail_closed",
        "scripts/semantics/finalize_farm_semantic_consensus.py",
    )
    rejected = _contract(
        topology="attached_component",
        complete_bounded=False,
    )
    ensemble = {
        "category": "unresolved object",
        "semantic_tier": "geometry_only",
        "review_decision": "unknown",
        "label_contract": rejected,
    }
    category, reason = finalizer.choose_category(
        {"category": "legacy machine"},
        {"category": "legacy machine"},
        ensemble,
        ensemble,
        {"legacy machine": 0.99},
    )
    assert category == "unresolved object"
    assert reason == "strict_open_vocabulary_failed_closed"


def test_strict_failure_nested_only_stays_fail_closed_across_boundaries() -> None:
    rejected = _contract(
        topology="attached_component",
        complete_bounded=False,
    )
    event = {
        "source": "blind_pass_a",
        "category": rejected["category"],
        "confidence": rejected["confidence"],
        "decision": "keep",
        "label_contract": rejected,
    }

    adjudicator = _load_script(
        "farm_open_vocab_adjudicate_nested_failure",
        "scripts/semantics/adjudicate_farm_semantics.py",
    )
    ensemble = adjudicator.output_row(
        {"id": 11},
        None,
        "geometry_only",
        [],
        "open_vocabulary_identity_unresolved",
        semantic_evidence=[event],
    )
    assert ensemble["strict_open_vocabulary_seen"] is True
    assert "label_contract" not in ensemble

    reconciler = _load_script(
        "farm_open_vocab_reconcile_nested_failure",
        "scripts/semantics/reconcile_farm_semantics.py",
    )
    evidence = reconciler._candidate_evidence(
        ensemble,
        {"category": "legacy machine", "description": "legacy"},
    )
    assert all(row.get("source") != "prior_farm_caption" for row in evidence)
    reconciled = reconciler._strict_current_resolution(ensemble, evidence, {})
    assert reconciled is not None
    assert reconciled["category"] == "unresolved object"
    assert reconciled["semantic_tier"] == "geometry_only"
    assert reconciled["semantic_gate_reason"] == (
        "strict_open_vocabulary_contract_unresolved"
    )

    finalizer = _load_script(
        "farm_open_vocab_finalizer_nested_failure",
        "scripts/semantics/finalize_farm_semantic_consensus.py",
    )
    category, reason = finalizer.choose_category(
        {"category": "legacy machine"},
        {"category": "legacy machine"},
        ensemble,
        reconciled,
        {"legacy machine": 0.99},
    )
    assert category == "unresolved object"
    assert reason == "strict_open_vocabulary_failed_closed"
