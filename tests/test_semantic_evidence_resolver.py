from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from farm_runtime.semantic_crop_selection_audit import audit_selection_payload
from farm_runtime.semantic_evidence_resolver import (
    PLAN_SCHEMA,
    REVIEW_SCHEMA,
    all_event_fingerprints,
    pair_fingerprints,
    resolve_plan,
    sha256_file,
    targeted_pair_is_stronger,
)
from scene_graph.captioning.evidence import crop_evidence_manifest
from scene_graph.captioning.label_contract import parse_open_vocabulary_label


def _contract(category: str = "sign", *, complete: bool = True, confidence: float = 0.98) -> dict:
    return parse_open_vocabulary_label({
        "category": category,
        "category_role": "whole_object",
        "form_hypernym": "sign",
        "primitive_form": "rectangle",
        "head_noun_is_primitive": False,
        "specificity": "specific_identity",
        "topology": "standalone_whole",
        "complete_bounded": complete,
        "carrier_category": "unknown",
        "payload_categories": [],
        "shape_profile": "planar",
        "identity_basis": "readable_text",
        "diagnostic_parts": ["text", "border"],
        "diagnostic_view_count": 3,
        "context_sufficient": True,
        "visible_target_coverage": 1.0 if complete else 0.95,
        "missing_visible_parts": [],
        "included_non_target": [],
        "whole_object_evidence_complete": True,
        "description": "bounded safety sign",
        "attributes": [],
        "confidence": confidence,
        "decision": "keep",
    })


def _event(object_id: int, source: str, contract: dict, ids: tuple[int, ...], angle: float) -> dict:
    crops = []
    poses = {}
    for offset, image_id in enumerate(ids):
        theta = math.radians(angle + offset * 3.0)
        center = [5.0 * math.cos(theta), 5.0 * math.sin(theta), 0.5]
        norm = math.sqrt(sum(value * value for value in center))
        poses[image_id] = {
            "camera_center_world_m": center,
            "camera_forward_world": [-value / norm for value in center],
        }
        crops.append((Path(f"img_{image_id:06d}_det_0000.npz"), b"jpeg"))
    manifest = crop_evidence_manifest(
        object_id,
        crops,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    return {
        "source": source,
        "event_id": f"object:{object_id}:{source}",
        "confirmation_eligible": True,
        "semantic_vote_independence": "independent",
        "request_conditioning": "none_blind",
        "label_contract": contract,
        **manifest,
    }


def _row(
    object_id: int,
    category: str = "sign",
    *,
    verification_angle: float,
    complete: bool = True,
    confidence: float = 0.98,
) -> dict:
    contract = _contract(category, complete=complete, confidence=confidence)
    return {
        "id": object_id,
        "category": "poster",
        "semantic_evidence": [
            _event(object_id, "contextual_dual_panel_review", contract, (1, 2, 3), 0.0),
            _event(object_id, "independent_verification", contract, (4, 5, 6), verification_angle),
        ],
    }


def _catalog(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"schema": REVIEW_SCHEMA, "objects": rows}), encoding="utf-8")
    return path


def _plan(base: Path, overlay: Path, base_row: dict, overlay_row: dict) -> dict:
    return {
        "schema": PLAN_SCHEMA,
        "minimum_identity_confidence": 0.85,
        "base_catalog": {
            "path": str(base), "sha256": sha256_file(base), "schema": REVIEW_SCHEMA,
        },
        "targeted_overlays": [{
            "object_id": int(overlay_row["id"]),
            "path": str(overlay),
            "sha256": sha256_file(overlay),
            "schema": REVIEW_SCHEMA,
            "base_event_fingerprints_sha256": pair_fingerprints(base_row),
            "targeted_event_fingerprints_sha256": pair_fingerprints(overlay_row),
        }],
    }


def test_resolver_selects_exact_stronger_pose_independent_overlay(tmp_path: Path) -> None:
    base_row = _row(37, verification_angle=5.0)
    overlay_row = _row(37, verification_angle=80.0, complete=False)
    base = _catalog(tmp_path / "base.json", [base_row])
    overlay = _catalog(tmp_path / "overlay.json", [overlay_row])
    result = resolve_plan(_plan(base, overlay, base_row, overlay_row))

    assert result["decisions"][0]["status"] == "targeted_overlay_selected"
    resolved = result["resolved_catalog"]["objects"][0]
    assert resolved["review_category"] == "sign"
    assert resolved["semantic_publication_action"] == "publish_independent_blind_consensus"
    assert resolved["semantic_resolution_provenance"]["targeted_object_id"] == 37


def test_resolver_retains_accepted_base_against_weaker_or_renaming_overlay() -> None:
    base = _row(37, verification_angle=90.0)
    weaker = _row(37, verification_angle=80.0, complete=False, confidence=0.90)
    accepted, report = targeted_pair_is_stronger(base, weaker, minimum_confidence=0.85)
    assert accepted is False
    assert "accepted_base_consensus_locked" in report["reason_codes"]
    assert any(reason.startswith("targeted_evidence_weaker:") for reason in report["reason_codes"])

    renamed = _row(37, "poster", verification_angle=100.0)
    accepted, report = targeted_pair_is_stronger(base, renamed, minimum_confidence=0.85)
    assert accepted is False
    assert "accepted_base_consensus_locked" in report["reason_codes"]
    assert "accepted_base_category_cannot_be_overwritten" in report["reason_codes"]


def test_resolver_rejects_unpinned_or_candidate_conditioned_overlay(tmp_path: Path) -> None:
    base_row = _row(37, verification_angle=5.0)
    overlay_row = _row(37, verification_angle=80.0)
    base = _catalog(tmp_path / "base.json", [base_row])
    overlay = _catalog(tmp_path / "overlay.json", [overlay_row])
    plan = _plan(base, overlay, base_row, overlay_row)
    plan["targeted_overlays"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="catalog_sha256_mismatch"):
        resolve_plan(plan)

    conditioned = copy.deepcopy(overlay_row)
    conditioned["semantic_evidence"][1]["confirmation_eligible"] = False
    conditioned["semantic_evidence"][1]["request_conditioning"] = "candidate_label"
    accepted, report = targeted_pair_is_stronger(
        base_row, conditioned, minimum_confidence=0.85
    )
    assert accepted is False
    assert "targeted_raw_blind_consensus_not_accepted" in report["reason_codes"]


def test_resolver_cannot_lower_identity_confidence_floor(tmp_path: Path) -> None:
    base_row = _row(37, verification_angle=5.0)
    overlay_row = _row(37, verification_angle=80.0, confidence=0.80)
    base = _catalog(tmp_path / "base.json", [base_row])
    overlay = _catalog(tmp_path / "overlay.json", [overlay_row])
    plan = _plan(base, overlay, base_row, overlay_row)
    plan["minimum_identity_confidence"] = 0.1

    result = resolve_plan(plan)

    assert result["decisions"][0]["status"] == "base_retained"
    assert result["resolved_catalog"]["semantic_resolution"][
        "minimum_identity_confidence"
    ] == 0.85


def test_contained_part_noun_cannot_replace_masked_carrier() -> None:
    contract = _contract("fire hose reel")
    contract["description"] = (
        "A red rectangular box mounted on a wall with a visible hose reel "
        "inside and a front door."
    )
    row = {
        "id": 108,
        "category": "fire hose cabinet",
        "semantic_evidence": [
            _event(108, "contextual_dual_panel_review", contract, (1, 2, 3), 0.0),
            _event(108, "independent_verification", contract, (4, 5, 6), 80.0),
        ],
    }

    consensus = targeted_pair_is_stronger(
        _row(108, "fire hose cabinet", verification_angle=5.0),
        row,
        minimum_confidence=0.85,
    )[1]["targeted_consensus"]

    assert consensus["accepted"] is False
    assert "initial_whole_identity_not_supported" in consensus["reason_codes"]
    assert "verification_whole_identity_not_supported" in consensus["reason_codes"]
    assert any("contained_part" in reason for reason in consensus["reason_codes"])


@pytest.mark.parametrize("description", [
    "a red wall box containing a fire hose reel",
    "a red cabinet which contains the fire hose reel",
    "a red housing that houses a fire hose reel",
    "a red box with a fire hose reel within",
])
def test_containment_wording_rejects_part_as_whole(description: str) -> None:
    contract = _contract("fire hose reel")
    contract["description"] = description
    row = {
        "id": 108,
        "semantic_evidence": [
            _event(108, "contextual_dual_panel_review", contract, (1, 2, 3), 0.0),
            _event(108, "independent_verification", contract, (4, 5, 6), 80.0),
        ],
    }
    consensus = targeted_pair_is_stronger(
        _row(108, "fire hose cabinet", verification_angle=5.0),
        row,
        minimum_confidence=0.85,
    )[1]["targeted_consensus"]
    assert consensus["accepted"] is False
    assert any("contained_part" in reason for reason in consensus["reason_codes"])


@pytest.mark.parametrize("category,description", [
    ("fire extinguisher", "standalone fire extinguisher with hose attached"),
    ("hose reel cabinet", "a bounded wall-mounted hose reel cabinet"),
])
def test_containment_parser_does_not_reject_standalone_or_carrier(
    category: str, description: str,
) -> None:
    contract = _contract(category)
    contract["description"] = description
    row = {
        "id": 2,
        "semantic_evidence": [
            _event(2, "contextual_dual_panel_review", contract, (1, 2, 3), 0.0),
            _event(2, "independent_verification", contract, (4, 5, 6), 80.0),
        ],
    }
    assert targeted_pair_is_stronger(
        _row(2, category, verification_angle=5.0), row,
        minimum_confidence=0.85,
    )[1]["targeted_consensus"]["accepted"] is True


def test_resolver_adds_absent_object_only_with_exact_consensus(tmp_path: Path) -> None:
    base = _catalog(tmp_path / "base.json", [_row(37, verification_angle=80.0)])
    added_row = _row(2, "fire extinguisher", verification_angle=80.0)
    added = _catalog(tmp_path / "added.json", [added_row])
    plan = {
        "schema": PLAN_SCHEMA,
        "base_catalog": {
            "path": str(base), "sha256": sha256_file(base), "schema": REVIEW_SCHEMA,
        },
        "targeted_overlays": [{
            "operation": "add",
            "base_object_absent": True,
            "object_id": 2,
            "path": str(added),
            "sha256": sha256_file(added),
            "schema": REVIEW_SCHEMA,
            "base_event_fingerprints_sha256": [],
            "targeted_event_fingerprints_sha256": all_event_fingerprints(added_row),
        }],
    }

    result = resolve_plan(plan)

    assert result["object_count"] == 2
    assert result["decisions"][0]["status"] == "targeted_object_added"


def test_incomplete_targeted_pair_is_bound_then_base_retained(tmp_path: Path) -> None:
    base_row = _row(0, "tray", verification_angle=80.0)
    targeted_row = copy.deepcopy(base_row)
    targeted_row["semantic_evidence"] = targeted_row["semantic_evidence"][:1]
    base = _catalog(tmp_path / "base.json", [base_row])
    targeted = _catalog(tmp_path / "targeted.json", [targeted_row])
    plan = {
        "schema": PLAN_SCHEMA,
        "base_catalog": {
            "path": str(base), "sha256": sha256_file(base), "schema": REVIEW_SCHEMA,
        },
        "targeted_overlays": [{
            "object_id": 0,
            "path": str(targeted),
            "sha256": sha256_file(targeted),
            "schema": REVIEW_SCHEMA,
            "base_event_fingerprints_sha256": pair_fingerprints(base_row),
            "targeted_event_fingerprints_sha256": all_event_fingerprints(targeted_row),
        }],
    }

    result = resolve_plan(plan)

    assert result["decisions"][0]["status"] == "base_retained"
    assert result["decisions"][0]["comparison"]["reason_codes"] == [
        "targeted_contract_invalid:exactly_two_current_semantic_events_required"
    ]


def test_contained_part_rejection_adds_relationship_route_without_rename(
    tmp_path: Path,
) -> None:
    base_row = _row(108, "fire hose cabinet", verification_angle=5.0)
    contract = _contract("fire hose reel")
    contract["description"] = "red wall box with a visible hose reel inside"
    targeted_row = {
        "id": 108,
        "semantic_evidence": [
            _event(108, "contextual_dual_panel_review", contract, (1, 2, 3), 0.0),
            _event(108, "independent_verification", contract, (4, 5, 6), 80.0),
        ],
    }
    base = _catalog(tmp_path / "base.json", [base_row])
    targeted = _catalog(tmp_path / "targeted.json", [targeted_row])
    result = resolve_plan(_plan(base, targeted, base_row, targeted_row))
    resolved = result["resolved_catalog"]["objects"][0]

    assert result["decisions"][0]["status"] == "base_retained"
    assert resolved["category"] == "poster"
    assert resolved["semantic_resolution_review_routes"] == [
        "relationship_verification"
    ]


def _selection_row(object_id: int) -> dict:
    joint = {
        "schema": "farm.cross-fold-pose-selection.v1",
        "status": "selected",
        "selected_initial_image_ids": [1, 2, 3],
        "selected_verification_image_ids": [4, 5, 6],
        "initial_within_fold": {
            "min_pairwise_viewpoint_angle_degrees": 12.0,
            "min_pairwise_normalized_camera_baseline": 0.2,
        },
        "verification_within_fold": {
            "min_pairwise_viewpoint_angle_degrees": 11.0,
            "min_pairwise_normalized_camera_baseline": 0.15,
        },
        "cross_fold": {
            "pose_independent": True,
            "symmetric_median_nearest_viewpoint_angle_degrees": 13.0,
            "symmetric_median_nearest_normalized_baseline": 0.25,
        },
    }
    diag = {"diversity_gate_met": True, "selected_pose_provenance_complete": True}
    return {
        "id": object_id,
        "semantic_initial_evidence_selection": {
            "schema": "farm.semantic-view-partitions.v1",
            "selection_diagnostics": diag,
            "cross_fold_pose_selection": joint,
            "selected_image_ids": [1, 2, 3],
            "selected_image_ids_complete": True,
        },
        "semantic_verification_evidence_selection": {
            "schema": "farm.semantic-view-partitions.v1",
            "selection_diagnostics": diag,
            "cross_fold_pose_selection": joint,
            "selected_image_ids": [4, 5, 6],
            "selected_image_ids_complete": True,
            "blind_verification_separation": {
                "status": "disjoint", "all_image_ids_parseable": True,
            },
        },
    }


def test_selection_audit_emits_only_strict_ready_ids_and_marks_missing() -> None:
    row = _selection_row(2)
    result = audit_selection_payload(
        {"schema": REVIEW_SCHEMA, "objects": [row]}, [2, 27]
    )
    assert result["ready_object_ids"] == [2]
    assert result["blocked_object_ids"] == [27]
    assert result["objects"][1]["status"] == "missing_input"

    broken = copy.deepcopy(row)
    broken["semantic_verification_evidence_selection"]["cross_fold_pose_selection"][
        "selected_verification_image_ids"
    ] = [3, 4, 5]
    result = audit_selection_payload(
        {"schema": REVIEW_SCHEMA, "objects": [broken]}, [2]
    )
    assert result["ready_object_ids"] == []
    assert "initial_verification_image_overlap" in result["objects"][0]["reason_codes"]
