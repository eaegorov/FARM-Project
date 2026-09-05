from __future__ import annotations

from scripts.semantics.merge_farm_contextual_semantics import merge_object
from scene_graph.captioning.label_contract import parse_open_vocabulary_label


def _contract(category: str) -> dict:
    return parse_open_vocabulary_label({
        "category": category,
        "category_role": "whole_object",
        "form_hypernym": "wheeled lifting tool",
        "primitive_form": "rectangular prism",
        "head_noun_is_primitive": False,
        "specificity": "specific_identity",
        "topology": "standalone_whole",
        "complete_bounded": True,
        "carrier_category": "unknown",
        "payload_categories": [],
        "shape_profile": "compact_volumetric",
        "identity_basis": "diagnostic_geometry",
        "diagnostic_parts": ["lifting arm", "wheels", "handle socket"],
        "diagnostic_view_count": 4,
        "context_sufficient": True,
        "visible_target_coverage": 1.0,
        "missing_visible_parts": [],
        "included_non_target": [],
        "description": "A floor jack supported by wheels on the workshop floor.",
        "attributes": [],
        "confidence": 0.98,
        "decision": "keep",
    })


def _review(object_id: int, category: str) -> dict:
    contract = _contract(category)
    return {
        "id": object_id,
        "semantic_evidence": [{
            "source": "contextual_dual_panel_review",
            "event_id": f"object:{object_id}:contextual_dual_panel_review",
            "label_contract": contract,
            "decision": "keep",
            "confirmation_eligible": True,
            "evidence_fingerprint_sha256": "f" * 64,
        }],
    }


def _unknown_review(object_id: int) -> dict:
    review = _review(object_id, "floor jack")
    review["semantic_evidence"][0]["decision"] = "unknown"
    return review



def test_independent_unknown_vetoes_candidate_conditioned_keep() -> None:
    object_id = 253
    review = _review(object_id, "mallet")
    review["semantic_evidence"].extend([
        {
            "source": "candidate_context_adjudication",
            "event_id": f"object:{object_id}:candidate_context_adjudication",
            "label_contract": _contract("mallet"),
            "decision": "keep",
            "confirmation_eligible": False,
            "evidence_fingerprint_sha256": "a" * 64,
        },
        {
            "source": "independent_verification",
            "event_id": f"object:{object_id}:independent_verification",
            "label_contract": {
                **_contract("unknown"),
                "category": "unknown",
                "decision": "unknown",
                "confidence": 0.1,
                "context_sufficient": False,
            },
            "decision": "unknown",
            "confirmation_eligible": False,
            "evidence_fingerprint_sha256": "b" * 64,
        },
    ])
    merged, outcome = merge_object(
        {"id": object_id, "category": "mallet", "semantic_tier": "probable"},
        review,
        minimum_confidence=0.90,
    )
    assert outcome == "identity_quarantined"
    assert merged["category"] == "unresolved object"


def test_contextual_review_repairs_probable_or_unresolved_noun() -> None:
    base = {
        "id": 253,
        "category": "mallet",
        "semantic_tier": "probable",
        "semantic_gate_reason": "single_high_confidence_open_vocabulary_label",
        "semantic_evidence": [],
    }
    merged, outcome = merge_object(
        base, _review(253, "floor jack"), minimum_confidence=0.90
    )
    assert outcome == "contextual_preferred"
    assert merged["category"] == "floor jack"
    assert merged["semantic_tier"] == "probable"
    assert merged["contextual_semantic_review"]["whole_object_ready"] is True
    assert len(merged["semantic_evidence"]) == 1


def test_contextual_review_cannot_override_confirmed_conflict() -> None:
    base = {
        "id": 1,
        "category": "cabinet",
        "semantic_tier": "confirmed",
        "semantic_gate_reason": "independent_consensus",
        "semantic_evidence": [],
    }
    merged, outcome = merge_object(
        base, _review(1, "floor jack"), minimum_confidence=0.90
    )
    assert outcome == "confirmed_conflict_preserved"
    assert merged["category"] == "cabinet"
    assert merged["semantic_tier"] == "confirmed"
    assert len(merged["semantic_evidence"]) == 1


def test_contextual_unknown_quarantines_unconfirmed_prior_noun() -> None:
    base = {
        "id": 19,
        "category": "ruler",
        "semantic_tier": "probable",
        "semantic_gate_reason": "single_high_confidence_open_vocabulary_label",
        "semantic_evidence": [],
    }
    merged, outcome = merge_object(
        base, _unknown_review(19), minimum_confidence=0.90
    )
    assert outcome == "identity_quarantined"
    assert merged["category"] == "unresolved object"
    assert merged["semantic_tier"] == "geometry_only"
    assert merged["semantic_gate_reason"] == "contextual_identity_quarantined"


def test_contextual_unknown_does_not_erase_confirmed_prior_noun() -> None:
    base = {
        "id": 19,
        "category": "overhead light",
        "semantic_tier": "confirmed",
        "semantic_gate_reason": "independent_consensus",
        "semantic_evidence": [],
    }
    merged, outcome = merge_object(
        base, _unknown_review(19), minimum_confidence=0.90
    )
    assert outcome == "confirmed_identity_preserved"
    assert merged["category"] == "overhead light"
    assert merged["semantic_tier"] == "confirmed"
