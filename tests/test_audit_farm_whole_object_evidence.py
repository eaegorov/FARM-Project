from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import sys
from pathlib import Path

from scene_graph.captioning.evidence import crop_evidence_manifest
from scene_graph.captioning.label_contract import parse_open_vocabulary_label


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "farm_whole_object_audit",
        ROOT / "scripts/evaluation/audit_farm_whole_object_evidence.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract(*, whole: bool, category: str = "utility cart") -> dict:
    return parse_open_vocabulary_label({
        "category": category,
        "category_role": "whole_object",
        "form_hypernym": "cart",
        "primitive_form": "open frame",
        "head_noun_is_primitive": False,
        "specificity": "object_kind",
        "topology": "standalone_whole" if whole else "partial_unbounded",
        "complete_bounded": whole,
        "carrier_category": "unknown",
        "payload_categories": [],
        "shape_profile": "open_frame",
        "identity_basis": "diagnostic_geometry",
        "diagnostic_parts": ["wheels", "frame"],
        "diagnostic_view_count": 3,
        "context_sufficient": True,
        "visible_target_coverage": 0.94 if whole else 0.52,
        "missing_visible_parts": [] if whole else ["lower frame"],
        "included_non_target": [],
        "description": "wheeled open frame",
        "attributes": [],
        "confidence": 0.96,
        "decision": "keep",
    })


def _event(source: str, contract: dict, ids: tuple[int, ...], angle: float) -> dict:
    crops = []
    poses = {}
    for offset, image_id in enumerate(ids):
        theta = math.radians(angle + 4.0 * offset)
        center = [4.0 * math.cos(theta), 4.0 * math.sin(theta), 0.5]
        poses[image_id] = {
            "camera_center_world_m": center,
            "camera_forward_world": [-center[0], -center[1], -center[2]],
        }
        crops.append((Path(f"img_{image_id:06d}_det_0000.npz"), b"jpeg"))
    manifest = crop_evidence_manifest(
        7,
        crops,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    return {
        "source": source,
        "event_id": f"object:7:{source}",
        "confirmation_eligible": True,
        "semantic_vote_independence": "independent",
        "request_conditioning": "none_blind",
        "category": contract["category"],
        "confidence": contract["confidence"],
        "label_contract": contract,
        **manifest,
    }


def _row(left: dict, right: dict) -> dict:
    return {
        "id": 7,
        "position_world_m": [0.0, 0.0, 0.0],
        "metric_dimensions_m": [0.8, 0.5, 1.0],
        "wxyz": [1.0, 0.0, 0.0, 0.0],
        "semantic_evidence": [left, right],
    }


def test_two_independent_whole_events_are_ready() -> None:
    audit = _load_script()
    contract = _contract(whole=True)
    row = _row(
        _event("blind_pass_a", contract, (1, 2, 3), 20.0),
        _event("blind_pass_c", contract, (4, 5, 6), 120.0),
    )
    result = audit.audit_object(row)
    assert result["status"] == "whole_ready"
    assert result["category"] == "utility cart"
    assert result["inpainting"]["tier"] == "needs_gaussian_lift"


def test_partial_identity_is_routed_to_refinement_not_publication() -> None:
    audit = _load_script()
    contract = _contract(whole=False)
    row = _row(
        _event("blind_pass_a", contract, (1, 2, 3), 20.0),
        _event("blind_pass_c", contract, (4, 5, 6), 120.0),
    )
    result = audit.audit_object(row)
    assert result["status"] == "refine"
    assert result["category"] == "utility cart"
    assert "missing_visible_parts" in result["refinement_signals"]
    assert result["inpainting"]["tier"] == "not_ready"


def test_conflicting_independent_identities_fail_closed() -> None:
    audit = _load_script()
    left = _contract(whole=True, category="utility cart")
    right = _contract(whole=True, category="storage rack")
    row = _row(
        _event("blind_pass_a", left, (1, 2, 3), 20.0),
        _event("blind_pass_c", right, (4, 5, 6), 120.0),
    )
    result = audit.audit_object(row)
    assert result["status"] == "reject"
    assert result["category"] == ""
    assert result["decision_reasons"] == ["conflicting_identity_labels"]

def test_contextual_quarantine_cannot_become_detector_refinement_prompt() -> None:
    audit = _load_script()
    contract = _contract(whole=False, category="ruler")
    row = _row(
        _event("blind_pass_a", contract, (1, 2, 3), 20.0),
        _event("blind_pass_c", contract, (4, 5, 6), 120.0),
    )
    row["semantic_gate_reason"] = "contextual_identity_quarantined"

    result = audit.audit_object(row)

    assert result["status"] == "reject"
    assert result["category"] == ""
    assert result["decision_reasons"] == ["contextual_identity_quarantined"]


def test_independent_blind_quarantine_overrides_agreeing_event_payloads() -> None:
    audit = _load_script()
    contract = _contract(whole=True)
    row = _row(
        _event("blind_pass_a", contract, (1, 2, 3), 20.0),
        _event("blind_pass_c", contract, (4, 5, 6), 120.0),
    )
    row.update({
        "semantic_quarantined": True,
        "semantic_publication_action": "quarantine_unresolved_semantics",
    })

    result = audit.audit_object(row)

    assert result["status"] == "reject"
    assert result["category"] == ""
    assert result["decision_reasons"] == [
        "independent_blind_semantics_quarantined"
    ]
    assert result["inpainting"]["tier"] == "not_ready"


def test_scope_incomplete_raw_consensus_re_adjudicates_only_semantic_identity() -> None:
    audit = _load_script()
    complete = _contract(whole=True, category="sign")
    incomplete_source = dict(complete)
    incomplete_source.update({
        "complete_bounded": False,
        "visible_target_coverage": 0.95,
        "missing_visible_parts": [],
        "included_non_target": [],
    })
    incomplete = parse_open_vocabulary_label(incomplete_source)
    row = _row(
        _event(
            "contextual_dual_panel_review", complete, (21, 72, 195, 253, 796), 0.0
        ),
        _event(
            "independent_verification", incomplete, (34, 36, 172, 205, 745), 80.0
        ),
    )
    row.update({
        "semantic_quarantined": True,
        "semantic_publication_action": "quarantine_unresolved_semantics",
        "review_gate_reason": "independent_blind_consensus_failed",
    })

    result = audit.audit_object(row)

    assert result["status"] == "refine"
    assert result["category"] == "sign"
    assert result["semantic_quarantine_re_adjudicated"] is True
    consensus = result["semantic_identity_re_adjudication"]
    assert consensus["accepted"] is True
    assert consensus["canonical_category"] == "sign"
    assert consensus["mask_scope_incomplete"] is True
    assert result["mask_scope"]["status"] == "incomplete"
    assert result["geometry_refinement"] == {
        "required": True,
        "route": "mask_geometry_refinement",
    }
    assert result["inpainting"]["tier"] == "not_ready"
    matrix = audit.semantic_status_matrix_row(row, result)
    assert matrix["current_label"] == "unresolved object"
    assert matrix["exact_noun_consensus"]["category"] == "sign"
    assert matrix["source_quarantine"] is True
    assert matrix["source_quarantine_re_adjudicated"] is True
    assert matrix["effective_quarantine"] is False
    assert matrix["routes"] == ["mask_geometry_refinement"]
    assert matrix["label_publishable"] is False
    assert matrix["inpainting_ready"] is False


def test_status_matrix_publishes_only_complete_independent_whole_consensus() -> None:
    audit = _load_script()
    contract = _contract(whole=True, category="sign")
    row = _row(
        _event(
            "contextual_dual_panel_review", contract, (1, 2, 3), 10.0
        ),
        _event(
            "independent_verification", contract, (4, 5, 6), 100.0
        ),
    )
    row.update({
        "review_category": "sign",
        "review_decision": "keep",
        "semantic_publication_action": "publish_independent_blind_consensus",
    })

    result = audit.audit_object(row)
    matrix = audit.semantic_status_matrix_row(row, result)

    assert result["status"] == "whole_ready"
    assert matrix["current_label"] == "sign"
    assert matrix["exact_noun_consensus"]["accepted"] is True
    assert matrix["generic_or_unknown_raw_noun"] is False
    assert matrix["category_conflict"] is False
    assert matrix["effective_quarantine"] is False
    assert matrix["mask_scope"]["status"] == "complete"
    assert matrix["routes"] == ["none"]
    assert matrix["label_publishable"] is True
    assert matrix["inpainting_ready"] is False


def test_thin_appendage_needs_mask_specific_part_coverage_before_inpainting() -> None:
    audit = _load_script()
    contract = _contract(whole=True, category="fire extinguisher")
    contract["diagnostic_parts"] = ["body", "handle", "hose", "nozzle"]
    row = _row(
        _event("contextual_dual_panel_review", contract, (1, 2, 3), 10.0),
        _event("independent_verification", contract, (4, 5, 6), 100.0),
    )
    row.update({
        "review_category": "fire extinguisher",
        "review_decision": "keep",
        "semantic_publication_action": "publish_independent_blind_consensus",
    })
    heldout = {
        "status": "verified",
        "provisional_gaussians": 5000,
        "summaries": {
            "iou": {"median": 0.95},
            "precision": {"minimum": 0.95},
            "recall": {"minimum": 0.95},
        },
    }

    result = audit.audit_object(row, heldout=heldout)
    matrix = audit.semantic_status_matrix_row(row, result)

    assert result["status"] == "whole_ready"
    assert result["semantic_identity_re_adjudication"]["accepted"] is True
    assert result["mask_scope"] == {
        "status": "incomplete",
        "reason_codes": ["whole_instance_thin_appendage_coverage_unverified"],
    }
    assert result["geometry_refinement"]["required"] is True
    assert result["inpainting"]["tier"] == "not_ready"
    assert matrix["label_publishable"] is False
    assert matrix["inpainting_ready"] is False


def test_legacy_failed_consensus_cannot_publish_preserved_prior() -> None:
    audit = _load_script()
    contract = _contract(whole=True)
    row = _row(
        _event("blind_pass_a", contract, (1, 2, 3), 20.0),
        _event("blind_pass_c", contract, (4, 5, 6), 120.0),
    )
    row.update({
        "semantic_publication_action": "preserve_prior_label",
        "independent_blind_consensus": {
            "status": "unresolved",
            "accepted": False,
        },
    })

    result = audit.audit_object(row)

    assert result["status"] == "reject"
    assert result["category"] == ""
    assert result["decision_reasons"] == [
        "legacy_unverified_prior_semantics_quarantined"
    ]


def test_verified_heldout_without_linked_count_is_not_inpainting_candidate() -> None:
    audit = _load_script()
    contract = _contract(whole=True)
    row = _row(
        _event("blind_pass_a", contract, (1, 2, 3), 20.0),
        _event("blind_pass_c", contract, (4, 5, 6), 120.0),
    )
    heldout = {
        "status": "verified",
        "summaries": {
            "iou": {"median": 0.9},
            "precision": {"minimum": 0.9},
            "recall": {"minimum": 0.9},
        },
    }

    result = audit.audit_object(row, heldout=heldout)

    assert result["status"] == "whole_ready"
    assert result["inpainting"] == {
        "tier": "needs_gaussian_count_evidence",
        "reason": "verified_heldout_qc_missing_linked_gaussian_count",
    }


def test_linked_gaussian_counts_require_candidate_and_qc_hash_chain(tmp_path) -> None:
    audit = _load_script()
    build_path = tmp_path / "build_manifest.json"
    candidate_path = tmp_path / "candidate.json"
    build = {
        "schema_version": "farm.gaussian-lift.build.v1",
        "status": "frozen_pending_heldout",
        "objects": [{
            "object_id": 7,
            "geometry_gate": "PASS",
            "provisional_gaussians": 1234,
        }],
    }
    build_path.write_text(json.dumps(build), encoding="utf-8")
    build_sha = hashlib.sha256(build_path.read_bytes()).hexdigest()
    candidate = {
        "schema": "farm.frozen-heldout-candidate.v1",
        "status": "frozen",
        "objects": [{"object_id": 7}],
        "provenance": {
            "consumed_input_hashes": {str(build_path): build_sha}
        },
    }
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    qc = {
        "contract": {
            "all_consumed_inputs_sha256_verified_and_rechecked": True
        },
        "provenance": {
            "candidate_manifest": {"sha256": candidate_sha}
        },
    }

    counts, provenance = audit._linked_gaussian_counts(
        build_path, candidate_path, qc
    )

    assert counts == {7: 1234}
    assert provenance["status"] == "verified"
    assert provenance["build_manifest_sha256"] == build_sha
    assert provenance["candidate_manifest_sha256"] == candidate_sha
