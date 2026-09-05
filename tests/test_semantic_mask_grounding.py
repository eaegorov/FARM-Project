from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scene_graph.captioning.evidence import (
    crop_evidence_manifest,
    gravity_upright_evidence_manifest,
    gravity_upright_orientation,
    load_frame_pose_index,
)

ROOT = Path(__file__).resolve().parents[1]


def _pose_index(image_ids) -> dict:
    result = {}
    for image_id in image_ids:
        angle = math.radians(float(image_id) * 20.0)
        center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        result[int(image_id)] = {
            "camera_center_world_m": center,
            "camera_forward_world": [-center[0] / 10.0, -center[1] / 10.0, 0.0],
        }
    return result


def _load(name: str, relative: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(source: str, category: str) -> dict:
    return {
        "source": source,
        "category": category,
        "description": category,
        "attributes": [],
        "confidence": 0.95,
        "decision": "keep",
    }


def _crop_row(
    image_id: int,
    detail: int,
    *,
    member: int | None = None,
) -> tuple[Path, bytes]:
    member_token = f"_member_{member:06d}" if member is not None else ""
    return (
        Path(
            f"/run/masks/object_000007/img_{image_id:06d}"
            f"{member_token}_det_0000.npz"
        ),
        b"x" * detail,
    )


def _crop_ids(rows) -> set[int]:
    return {
        int(row[0].name.split("img_", 1)[1].split("_", 1)[0])
        for row in rows
    }


def test_crop_candidates_isolate_saved_mask_on_neutral_background(tmp_path: Path) -> None:
    review = _load("farm_mask_grounding", "scripts/semantics/review_farm_object_crops.py")
    object_dir = tmp_path / "object_000051"
    object_dir.mkdir()
    image = np.full((20, 20, 3), 245, dtype=np.uint8)
    image[8:12, 8:12] = np.asarray([10, 20, 230], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    raw = np.ones((4, 4), dtype=np.uint8)
    np.savez_compressed(
        object_dir / "img_000001_det_0001.npz",
        crop_jpeg_bytes=np.asarray(encoded, dtype=np.uint8),
        raw_bits=np.packbits(raw.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(raw.shape, dtype=np.int32),
        crop_bbox_xyxy=np.asarray([8, 8, 12, 12], dtype=np.int32),
    )

    rows = review.crop_candidates([object_dir / "img_000001_det_0001.npz"])
    assert len(rows) == 1
    grounded = cv2.imdecode(np.frombuffer(rows[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert grounded is not None
    assert float(grounded[:5, :5].mean()) < 60.0
    assert int(grounded[9:11, 9:11, 2].mean()) > 150


def test_crop_candidates_add_dim_source_context_without_recolouring_target(
    tmp_path: Path,
) -> None:
    review = _load(
        "farm_contextual_semantic_crop", "scripts/semantics/review_farm_object_crops.py"
    )
    object_dir = tmp_path / "mapping" / "masks" / "object_000007"
    object_dir.mkdir(parents=True)
    image = np.full((20, 24, 3), 190, dtype=np.uint8)
    image[5:15, 6:18] = np.asarray([20, 30, 230], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    mask = np.ones((10, 12), dtype=np.uint8)
    np.savez_compressed(
        object_dir / "img_000000_det_0000.npz",
        crop_jpeg_bytes=np.asarray(encoded, dtype=np.uint8),
        raw_bits=np.packbits(mask.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(mask.shape, dtype=np.int32),
        crop_bbox_xyxy=np.asarray([6, 5, 18, 15], dtype=np.int32),
        raw_bbox_xyxy=np.asarray([30, 30, 42, 40], dtype=np.int32),
    )

    rgbd = tmp_path / "rgbd"
    rgbd.mkdir()
    source = np.full((80, 100, 3), 200, dtype=np.uint8)
    source[30:40, 30:42] = np.asarray([20, 30, 230], dtype=np.uint8)
    assert cv2.imwrite(str(rgbd / "frame.jpg"), source)
    (rgbd / "frames.json").write_text(
        '{"frames":[{"rgb_path":"frame.jpg"}]}', encoding="utf-8"
    )

    rows = review.crop_candidates(
        [object_dir / "img_000000_det_0000.npz"],
        frames_json=rgbd / "frames.json",
    )
    grounded = cv2.imdecode(np.frombuffer(rows[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert grounded is not None
    assert grounded.shape[0] > image.shape[0]
    assert grounded.shape[1] > image.shape[1]
    # Context remains visible but subordinate; target pixels retain natural red.
    assert 60.0 < float(grounded[:10, :10].mean()) < 130.0
    target = grounded[30:40, 30:42]
    assert int(target[:, :, 2].mean()) > 170

    dual_rows = review.crop_candidates(
        [object_dir / "img_000000_det_0000.npz"],
        frames_json=rgbd / "frames.json",
        natural_context_panel=True,
    )
    dual = cv2.imdecode(
        np.frombuffer(dual_rows[0][1], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert dual is not None
    assert dual.shape[0] == grounded.shape[0]
    assert dual.shape[1] == 2 * grounded.shape[1]
    assert float(dual[:, grounded.shape[1]:].mean()) > float(dual[:, :grounded.shape[1]].mean())


def test_metric_dimensions_are_added_as_scale_evidence_without_class_hint() -> None:
    review = _load("farm_metric_semantic_evidence", "scripts/semantics/review_farm_object_crops.py")
    prompt = review.prompt_with_metric_evidence(
        "Identify the object.", {"metric_dimensions_m": [0.76163, 0.48061, 1.04]}
    )
    assert "L × W × H" in prompt
    assert "0.762 × 0.481 × 1.040 metres" in prompt
    assert "third value is the gravity-aligned height" in prompt
    assert "physical-scale sanity check" in prompt
    assert "cabinet" not in prompt and "book" not in prompt


def test_candidate_label_sets_keep_conflicts_as_untrusted_hypotheses(
    tmp_path: Path,
) -> None:
    review = _load(
        "farm_candidate_context_adjudication",
        "scripts/semantics/review_farm_object_crops.py",
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps([{"id": 253, "category": "mallet"}]), encoding="utf-8"
    )
    second.write_text(
        json.dumps({
            "objects": [{
                "id": 253,
                "review_label_contract": {"category": "floor jack"},
            }]
        }),
        encoding="utf-8",
    )
    labels = review.candidate_label_sets([first, second])
    assert labels[253] == {"mallet", "floor jack"}
    prompt = review.candidate_adjudication_prompt("Inspect.", labels[253])
    assert "untrusted alternatives" in prompt
    assert "floor jack" in prompt and "mallet" in prompt


def test_candidate_conditioned_verification_may_refine_open_vocabulary_noun() -> None:
    review = _load(
        "farm_verification_refinement", "scripts/semantics/review_farm_object_crops.py"
    )
    reason = review.verification_gate_reason(
        {"category": "sign"}, {"review_category": "poster"}
    )
    assert reason == "candidate_conditioned_verification_revised_category"


def _strict_review_vote(review, category: str) -> dict:
    contract = review.parse_open_vocabulary_label(
        {
            "category": category,
            "category_role": "whole_object",
            "form_hypernym": category,
            "primitive_form": "unknown",
            "head_noun_is_primitive": False,
            "specificity": "object_kind",
            "topology": "standalone_whole",
            "complete_bounded": True,
            "carrier_category": "unknown",
            "payload_categories": [],
            "shape_profile": "compact_volumetric",
            "identity_basis": "diagnostic_geometry",
            "diagnostic_parts": ["bounded body", "distinctive supports"],
            "diagnostic_view_count": 3,
            "context_sufficient": True,
            "visible_target_coverage": 0.95,
            "missing_visible_parts": [],
            "included_non_target": [],
            "description": f"bounded visible {category}",
            "attributes": [],
            "confidence": 0.94,
            "decision": "keep",
        }
    )
    assessment = review.assess_open_vocabulary_label(
        contract, minimum_confidence=0.0
    )
    return {
        "review_category": assessment["category"],
        "review_description": assessment["description"],
        "review_attributes": assessment["attributes"],
        "review_confidence": 0.94,
        "review_decision": "keep",
        "review_label_contract": contract,
    }


def _initial_vote(vote: dict) -> dict:
    return {
        "category": vote["review_category"],
        "description": vote["review_description"],
        "attributes": vote["review_attributes"],
        "confidence": vote["review_confidence"],
        "decision": vote["review_decision"],
        "label_contract": vote["review_label_contract"],
    }


def _semantic_vote_event(
    review,
    category: str,
    source: str,
    image_ids: list[int],
    fingerprint: str,
) -> dict:
    vote = _strict_review_vote(review, category)
    crops = [_crop_row(image_id, 100) for image_id in image_ids]
    manifest = crop_evidence_manifest(
        8,
        crops,
        frame_pose_index=_pose_index(image_ids),
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    return {
        "source": source,
        "event_id": f"object:8:{source}",
        "category": vote["review_category"],
        "description": vote["review_description"],
        "attributes": vote["review_attributes"],
        "confidence": vote["review_confidence"],
        "decision": vote["review_decision"],
        "label_contract": vote["review_label_contract"],
        "confirmation_eligible": True,
        "semantic_vote_independence": "independent",
        "request_conditioning": "none_blind",
        **manifest,
    }


def test_raw_blind_consensus_keeps_identity_when_only_mask_scope_is_incomplete() -> None:
    review = _load(
        "farm_independent_identity_incomplete_scope",
        "scripts/semantics/review_farm_object_crops.py",
    )
    initial = _semantic_vote_event(
        review, "sign", "contextual_dual_panel_review", [0, 6, 12, 18, 24], "a"
    )
    verification = _semantic_vote_event(
        review, "sign", "independent_verification", [2, 8, 14, 20, 26], "b"
    )
    partial = dict(verification["label_contract"])
    partial.update({
        "complete_bounded": False,
        "visible_target_coverage": 0.95,
        "missing_visible_parts": [],
        "included_non_target": [],
    })
    verification["label_contract"] = review.parse_open_vocabulary_label(partial)
    # Reproduce the stale derived presentation fields in the immutable V7 row.
    verification.update({"category": "unresolved object", "decision": "unknown"})

    consensus = review.independent_blind_consensus(
        initial, verification, minimum_confidence=0.85
    )
    row = {
        "id": 8,
        "category": "monitor",
        "semantic_quarantined": True,
        "semantic_publication_action": "quarantine_unresolved_semantics",
    }
    review.apply_independent_blind_result(
        row, initial, verification, minimum_confidence=0.85
    )

    assert consensus["accepted"] is True
    assert consensus["canonical_category"] == "sign"
    assert consensus["identity_contract_source"] == "raw_label_contract"
    assert consensus["mask_scope_incomplete"] is True
    assert consensus["geometry_refinement_required"] is True
    assert consensus["inpainting_eligible"] is False
    assert row["semantic_identity_verified"] is True
    assert row["semantic_publication_action"] == (
        "retain_verified_identity_for_mask_refinement"
    )
    assert row["label_publication_eligible"] is False
    assert row["inpainting_eligible"] is False


def test_partial_component_cannot_be_promoted_by_matching_blind_nouns() -> None:
    review = _load(
        "farm_independent_partial_component",
        "scripts/semantics/review_farm_object_crops.py",
    )
    initial = _semantic_vote_event(
        review, "monitor", "initial_blind_review", [0, 6, 12], "a"
    )
    verification = _semantic_vote_event(
        review, "monitor", "independent_verification", [2, 8, 14], "b"
    )
    for event in (initial, verification):
        component = dict(event["label_contract"])
        component.update({
            "category_role": "standalone_component",
            "topology": "standalone_component",
            "complete_bounded": False,
            "visible_target_coverage": 0.95,
        })
        event["label_contract"] = review.parse_open_vocabulary_label(component)

    consensus = review.independent_blind_consensus(
        initial, verification, minimum_confidence=0.85
    )

    assert consensus["accepted"] is False
    assert "initial_whole_identity_not_supported" in consensus["reason_codes"]
    assert "verification_whole_identity_not_supported" in consensus["reason_codes"]


def test_incomplete_scope_still_rejects_invalid_or_mismatching_raw_contracts() -> None:
    review = _load(
        "farm_independent_incomplete_scope_counterexamples",
        "scripts/semantics/review_farm_object_crops.py",
    )
    initial = _semantic_vote_event(
        review, "sign", "initial_blind_review", [0, 6, 12], "a"
    )
    verification = _semantic_vote_event(
        review, "poster", "independent_verification", [2, 8, 14], "b"
    )
    partial = dict(verification["label_contract"])
    partial.update({"complete_bounded": False, "visible_target_coverage": 0.95})
    verification["label_contract"] = review.parse_open_vocabulary_label(partial)

    mismatch = review.independent_blind_consensus(
        initial, verification, minimum_confidence=0.85
    )
    assert mismatch["accepted"] is False
    assert "independent_blind_category_disagreement" in mismatch["reason_codes"]

    verification["label_contract"] = {}
    invalid = review.independent_blind_consensus(
        initial, verification, minimum_confidence=0.85
    )
    assert invalid["accepted"] is False
    assert "verification_label_contract_invalid" in invalid["reason_codes"]


def test_independent_verification_uses_unconditioned_prompt() -> None:
    review = _load(
        "farm_independent_blind_prompt",
        "scripts/semantics/review_farm_object_crops.py",
    )
    initial = {"category": "laptop", "description": "candidate text"}
    independent = review.verification_request_contract(
        "independent_blind", blind_prompt="Inspect without hints.", initial=initial
    )
    assert independent["prompt"] == "Inspect without hints."
    assert "laptop" not in independent["prompt"]
    assert independent["prompt_id"] == "independent_blind_verification.v1"
    assert independent["confirmation_eligible"] is True
    assert independent["semantic_vote_independence"] == "independent"
    assert independent["request_conditioning"] == "none_blind"

    compatibility = review.verification_request_contract(
        "candidate_conditioned", blind_prompt="Inspect.", initial=initial
    )
    assert "laptop" in compatibility["prompt"]
    assert compatibility["confirmation_eligible"] is False


def test_independent_blind_folds_are_disjoint_from_same_candidate_universe() -> None:
    review = _load(
        "farm_independent_blind_folds",
        "scripts/semantics/review_farm_object_crops.py",
    )
    candidates = [_crop_row(image_id, 100) for image_id in range(6)]
    initial, _ = review.choose_evidence_crops(
        candidates, "initial_blind_review", 3
    )
    verification, partition = review.choose_evidence_crops(
        candidates, "independent_verification", 3
    )
    verification, audit = review.enforce_disjoint_verification_fold(
        initial, verification, partition
    )
    assert _crop_ids(initial) == {0, 2, 4}
    assert _crop_ids(verification) == {1, 3, 5}
    assert _crop_ids(initial).isdisjoint(_crop_ids(verification))
    assert _crop_ids(initial) | _crop_ids(verification) == set(range(6))
    assert audit["blind_verification_separation"]["status"] == "disjoint"


def test_independent_blind_consensus_uses_exact_current_event_payloads() -> None:
    review = _load(
        "farm_independent_blind_exact_events",
        "scripts/semantics/review_farm_object_crops.py",
    )
    initial = _semantic_vote_event(
        review, "tray", "initial_blind_review", [0, 6], "a"
    )
    verification = _semantic_vote_event(
        review, "bin", "independent_verification", [2, 111], "b"
    )
    # Reproduce V4's stale inherited fields. They must not override the event.
    verification["review_category"] = "tray"
    verification["review_label_contract"] = initial["label_contract"]

    consensus = review.independent_blind_consensus(
        initial, verification, minimum_confidence=0.65
    )

    assert consensus["accepted"] is False
    assert consensus["initial_category"] == "tray"
    assert consensus["verification_category"] == "bin"
    assert "independent_blind_category_disagreement" in consensus["reason_codes"]


def test_independent_blind_event_pair_rejects_stale_inherited_events() -> None:
    review = _load(
        "farm_independent_blind_stale_events",
        "scripts/semantics/review_farm_object_crops.py",
    )
    initial = _semantic_vote_event(
        review, "sign", "initial_blind_review", [0, 6], "a"
    )
    stale = _semantic_vote_event(
        review, "poster", "independent_verification", [2, 111], "b"
    )
    current = _semantic_vote_event(
        review, "sign", "independent_verification", [4, 113], "c"
    )
    row = {"id": 8, "semantic_evidence": [initial, stale]}

    with pytest.raises(ValueError, match="exactly one current initial event"):
        review.current_independent_blind_event_pair(row, current)


def test_independent_blind_consensus_requires_unique_event_ids() -> None:
    review = _load(
        "farm_independent_blind_unique_event_ids",
        "scripts/semantics/review_farm_object_crops.py",
    )
    initial = _semantic_vote_event(
        review, "pallet", "initial_blind_review", [0, 6], "a"
    )
    verification = _semantic_vote_event(
        review, "pallet", "independent_verification", [2, 111], "b"
    )
    verification["event_id"] = initial["event_id"]

    consensus = review.independent_blind_consensus(
        initial, verification, minimum_confidence=0.65
    )

    assert consensus["accepted"] is False
    assert "semantic_event_ids_missing_or_not_unique" in consensus["reason_codes"]


def test_independent_blind_safe_agreement_is_publishable() -> None:
    review = _load(
        "farm_independent_blind_agreement",
        "scripts/semantics/review_farm_object_crops.py",
    )
    first = _strict_review_vote(review, "pallet")
    second = _strict_review_vote(review, "palet")
    row = {"id": 8, "category": "cabinet"}
    consensus = review.apply_independent_blind_result(
        row, _initial_vote(first), second, minimum_confidence=0.65
    )
    assert consensus["accepted"] is True
    assert consensus["canonical_category"] == "pallet"
    assert row["review_category"] == "pallet"
    assert row["review_decision"] == "keep"
    assert row["category"] == "cabinet"


def test_independent_blind_disagreement_quarantines_unproven_prior_label() -> None:
    review = _load(
        "farm_independent_blind_disagreement",
        "scripts/semantics/review_farm_object_crops.py",
    )
    first = _strict_review_vote(review, "floor cleaner")
    second = _strict_review_vote(review, "laptop")
    row = {"id": 16, "category": "cleaner"}
    consensus = review.apply_independent_blind_result(
        row, _initial_vote(first), second, minimum_confidence=0.65
    )
    assert consensus["accepted"] is False
    assert "independent_blind_category_disagreement" in consensus["reason_codes"]
    assert row["category"] == "cleaner"
    assert row["semantic_prior_label"]["category"] == "cleaner"
    assert row["review_category"] == "unknown"
    assert row["review_decision"] == "unknown"
    assert row["semantic_publication_action"] == "quarantine_unresolved_semantics"
    assert row["semantic_status"] == "independent_blind_unresolved"
    assert row["semantic_quarantined"] is True
    assert row["semantic_review_required"] is True
    assert row["label_publication_eligible"] is False
    assert row["unresolved_semantic_hypotheses"] == ["floor cleaner", "laptop"]
    assert row["review_verification"]["review_category"] == "laptop"


def test_independent_blind_generic_or_invalid_vote_is_unresolved() -> None:
    review = _load(
        "farm_independent_blind_generic",
        "scripts/semantics/review_farm_object_crops.py",
    )
    generic = _strict_review_vote(review, "machine")
    generic_consensus = review.independent_blind_consensus(
        _initial_vote(generic), generic, minimum_confidence=0.65
    )
    assert generic_consensus["accepted"] is False
    assert "initial_category_generic_or_unresolved" in generic_consensus[
        "reason_codes"
    ]

    concrete = _strict_review_vote(review, "pallet")
    invalid = dict(concrete)
    invalid["review_label_contract"] = {}
    invalid_consensus = review.independent_blind_consensus(
        _initial_vote(concrete), invalid, minimum_confidence=0.65
    )
    assert invalid_consensus["accepted"] is False
    assert "verification_label_contract_invalid" in invalid_consensus[
        "reason_codes"
    ]


def test_agreeing_generic_votes_are_explicitly_quarantined() -> None:
    review = _load(
        "farm_independent_blind_generic_quarantine",
        "scripts/semantics/review_farm_object_crops.py",
    )
    first = _strict_review_vote(review, "machine")
    second = _strict_review_vote(review, "machine")
    row = {"id": 8, "category": "cleaner"}

    consensus = review.apply_independent_blind_result(
        row, _initial_vote(first), second, minimum_confidence=0.65
    )

    assert consensus["accepted"] is False
    assert row["review_category"] == "unknown"
    assert row["semantic_quarantined"] is True
    assert row["semantic_quarantine_reason_codes"] == [
        "initial_category_generic_or_unresolved",
        "verification_category_generic_or_unresolved",
    ]
    assert row["unresolved_semantic_hypotheses"] == []


def test_invalid_metric_dimensions_are_not_added_to_prompt() -> None:
    review = _load("farm_invalid_metric_semantic_evidence", "scripts/semantics/review_farm_object_crops.py")
    assert review.prompt_with_metric_evidence(
        "Identify the object.", {"metric_dimensions_m": [1.0, 0.0, 2.0]}
    ) == "Identify the object."


def test_active_only_catalog_filter_uses_scene_state() -> None:
    review = _load("farm_active_semantic_filter", "scripts/semantics/review_farm_object_crops.py")
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    state = {
        "object_id": np.asarray([3, 1, 2], dtype=np.int64),
        "active": np.asarray([True, False, True], dtype=np.bool_),
    }
    assert review.filter_active_rows(rows, state) == [{"id": 2}, {"id": 3}]
def test_initial_crop_selection_uses_pose_diversity_additively() -> None:

    review = _load(
        "farm_pose_aware_initial_selection",
        "scripts/semantics/review_farm_object_crops.py",
    )
    candidates = [
        _crop_row(0, 100),
        _crop_row(1, 100),
        _crop_row(2, 100),
        _crop_row(3, 80),
        _crop_row(4, 80),
    ]
    poses = _pose_index((0, 1, 2, 3, 4))

    selected, partition = review.choose_evidence_crops(
        candidates,
        "initial_blind_review",
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
        pose_aware=True,
    )
    unchanged, unchanged_partition = review.choose_evidence_crops(
        candidates,
        "independent_verification",
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
    )

    assert _crop_ids(selected) == {0, 1, 2}
    assert partition["selection_diagnostics"]["method"] == "pose_aware_exact"
    assert partition["selection_diagnostics"]["diversity_gate_met"] is True
    assert unchanged == review.choose_crops(
        sorted(candidates, key=lambda item: len(item[1]), reverse=True),
        3,
    )
    assert "selection_diagnostics" not in unchanged_partition


def test_joint_blind_fold_selection_uses_strict_unseen_pose_channel() -> None:
    review = _load(
        "farm_joint_blind_pose_selection",
        "scripts/semantics/review_farm_object_crops.py",
    )
    candidates = [_crop_row(image_id, 100) for image_id in range(18)]
    poses = {}
    for image_id in range(18):
        temporal_index, fold = divmod(image_id, 3)
        angle_degrees = 60.0 * temporal_index + (1.0 if fold == 1 else 25.0 if fold == 2 else 0.0)
        angle = math.radians(angle_degrees)
        center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        poses[image_id] = {
            "camera_center_world_m": center,
            "camera_forward_world": [
                -center[0] / 10.0,
                -center[1] / 10.0,
                0.0,
            ],
        }

    initial, _ = review.choose_evidence_crops(
        candidates,
        "initial_blind_review",
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
        pose_aware=True,
    )
    fixed_verification, _ = review.choose_evidence_crops(
        candidates,
        "independent_verification",
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
        pose_aware=True,
    )
    selected_initial, selected_verification, diagnostic = (
        review.optimize_cross_fold_pose_diversity(
            candidates,
            initial,
            fixed_verification,
            3,
            frame_pose_index=poses,
            object_position_world_m=[0.0, 0.0, 0.0],
        )
    )

    assert diagnostic["status"] == "selected"
    assert diagnostic["verification_pool_policy"] == (
        "all_posed_views_outside_initial_partition"
    )
    assert diagnostic["cross_fold"]["pose_independent"] is True
    assert diagnostic["cross_fold"][
        "symmetric_median_nearest_viewpoint_angle_degrees"
    ] >= 10.0
    assert diagnostic["initial_within_fold"][
        "min_pairwise_viewpoint_angle_degrees"
    ] >= 10.0
    assert diagnostic["verification_within_fold"][
        "min_pairwise_viewpoint_angle_degrees"
    ] >= 10.0
    assert _crop_ids(selected_initial).isdisjoint(
        _crop_ids(selected_verification)
    )
    assert any(image_id % 3 == 2 for image_id in _crop_ids(selected_verification))


def test_pose_selection_preserves_assembly_member_balancing() -> None:
    review = _load(
        "farm_pose_aware_assembly_selection",
        "scripts/semantics/review_farm_object_crops.py",
    )
    candidates = [
        _crop_row(0, 100, member=1),
        _crop_row(1, 95, member=1),
        _crop_row(2, 90, member=1),
        _crop_row(3, 85, member=2),
        _crop_row(4, 80, member=2),
    ]
    plain, _ = review.choose_evidence_crops(
        candidates,
        "initial_blind_review",
        3,
    )
    selected, partition = review.choose_evidence_crops(
        candidates,
        "initial_blind_review",
        3,
        frame_pose_index=_pose_index((0, 1, 2, 3, 4)),
        object_position_world_m=[0.0, 0.0, 0.0],
        pose_aware=True,
    )

    assert selected == plain
    assert partition["selection_diagnostics"]["method"] == "temporal_fallback"
    assert (
        partition["selection_diagnostics"]["reason"]
        == "assembly_member_balancing_preserved"
    )


def test_negative_review_cannot_be_overridden_by_one_later_guess() -> None:
    semantic = _load("farm_semantic_negative", "scripts/semantics/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [_candidate("blind_pass_b", "tool")], ["strict_two_pass"], 0.9
    )
    assert tier == "geometry_only"
    assert chosen is None
    assert reason == "negative_independent_evidence"


def test_same_model_repeated_votes_do_not_override_unknown_evidence() -> None:
    semantic = _load("farm_semantic_consensus", "scripts/semantics/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [
            _candidate("blind_pass_a", "cabinet"),
            _candidate("blind_pass_c", "cabinet"),
        ],
        ["verification_pass"],
        0.9,
    )
    assert tier == "geometry_only"
    assert chosen is None
    assert reason == "negative_independent_evidence"


def test_two_matching_votes_without_negative_evidence_are_confirmed() -> None:
    semantic = _load("farm_semantic_positive_consensus", "scripts/semantics/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [_candidate("blind_pass_a", "cabinet"), _candidate("blind_pass_b", "cabinet")],
        [],
        0.9,
    )
    assert tier == "probable"
    assert chosen is not None and chosen["category"] == "cabinet"
    assert reason == "correlated_or_single_high_confidence_evidence"


def test_conflicting_labels_are_not_confirmed_without_independent_majority() -> None:
    semantic = _load("farm_semantic_conflict", "scripts/semantics/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [
            _candidate("blind_pass_a", "cabinet"),
            _candidate("blind_pass_b", "bed"),
        ],
        [],
        0.9,
    )
    assert tier == "geometry_only"
    assert chosen is None
    assert reason == "conflicting_independent_evidence"


def test_neutral_fallback_cannot_override_any_informative_vote() -> None:
    semantic = _load("farm_neutral_last_resort", "scripts/semantics/adjudicate_farm_semantics.py")
    assert semantic.neutral_fallback_allowed([]) is True
    assert semantic.neutral_fallback_allowed([_candidate("blind_pass_a", "server")]) is False


def test_physical_recovery_request_is_blind_provenance_complete_and_ineligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    semantic = _load(
        "farm_physical_recovery", "scripts/semantics/reconcile_farm_semantics.py"
    )
    captured: dict = {}

    def fake_request(url, model, row, crops, prompt, **kwargs):
        captured.update({"row": row, "prompt": prompt, "crops": crops})
        return {
            "review_category": "pipe frame",
            "review_description": "bounded frame holding visible pipes",
            "review_attributes": ["standalone_bounded", "metal frame"],
            "review_confidence": 0.93,
            "review_decision": "keep",
            "review_raw_response": "{}",
        }

    monkeypatch.setattr(semantic, "request_review", fake_request)
    crops = [
        (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
        for value in (4, 7, 10)
    ]
    event = semantic.physical_recovery_event(
        "http://unused",
        "model",
        {"id": 1, "category": "workbench", "metric_dimensions_m": [1, 1, 2], "position_world_m": [0, 0, 0]},
        crops,
        {
            "partition_id": "view-fold-2-of-3",
            "partition_index": 2,
            "partition_count": 3,
        },
        _pose_index((4, 7, 10)),
    )

    assert event["source"] == "physical_form_recovery"
    assert event["confirmation_eligible"] is False
    assert event["crop_image_ids"] == [4, 7, 10]
    assert event["crop_partition_id"] == "view-fold-2-of-3"
    assert event["standalone_bounded"] is True
    assert event["camera_pose_provenance_complete"] is True
    assert set(captured["row"]) == {"metric_dimensions_m"}
    assert "workbench" not in captured["prompt"].lower()

    verification = semantic.physical_recovery_event(
        "http://unused",
        "model",
        {"id": 1, "category": "workbench", "metric_dimensions_m": [1, 1, 2], "position_world_m": [0, 0, 0]},
        [
            (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
            for value in (5, 8, 11)
        ],
        {
            "partition_id": "view-fold-1-of-3",
            "partition_index": 1,
            "partition_count": 3,
        },
        _pose_index((5, 8, 11)),
        source="physical_form_recovery_verification",
    )
    assert verification["source"] == "physical_form_recovery_verification"
    assert verification["event_id"].endswith(
        ":physical_form_recovery_verification"
    )
    assert verification["crop_image_ids"] == [5, 8, 11]
    assert verification["confirmation_eligible"] is False
    assert set(captured["row"]) == {"metric_dimensions_m"}
    assert "workbench" not in captured["prompt"].lower()


def test_recovery_preflight_rejects_reused_views_before_second_request(
    tmp_path: Path,
) -> None:
    semantic = _load(
        "farm_physical_recovery_preflight", "scripts/semantics/reconcile_farm_semantics.py"
    )
    primary = [
        (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
        for value in (1, 2, 3)
    ]
    independent = [
        (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
        for value in (4, 5, 6)
    ]
    reused = list(primary)
    first_partition = {
        "partition_id": "view-fold-0-of-2",
        "partition_index": 0,
        "partition_count": 2,
    }
    second_partition = {
        "partition_id": "view-fold-1-of-2",
        "partition_index": 1,
        "partition_count": 2,
    }

    poses = _pose_index((1, 2, 3, 4, 5, 6))
    accepted = semantic.independent_recovery_crop_contract(
        7, primary, first_partition, independent, second_partition,
        poses, [0.0, 0.0, 0.0],
    )
    rejected = semantic.independent_recovery_crop_contract(
        7, primary, first_partition, reused, first_partition,
        poses, [0.0, 0.0, 0.0],
    )

    assert accepted["eligible"] is True
    assert accepted["unique_view_count"] == 6
    assert accepted["view_overlap_coefficient"] == 0.0
    assert rejected["eligible"] is False
    assert "recovery_view_overlap_above_threshold" in rejected["reason_codes"]
    assert "recovery_same_evidence_fingerprint" in rejected["reason_codes"]


def test_review_prefers_complete_refined_observation_subset(tmp_path: Path) -> None:
    review = _load(
        "farm_preferred_semantic_evidence",
        "scripts/semantics/review_farm_object_crops.py",
    )
    original = tmp_path / "object_000007/img_000001_det_0000.npz"
    refined_a = tmp_path / "object_000007/img_000010_det_0000.npz"
    refined_b = tmp_path / "object_000007/img_000011_det_0000.npz"
    paths = [original, refined_a, refined_b]
    state = {
        "object_id": [7],
        "object_mask_observations": [[
            {"path": str(original), "source": "mapping"},
            {"path": str(refined_a), "source": "full_colmap_sam3_refinement"},
            {"path": str(refined_b), "source": "full_colmap_sam3_refinement"},
        ]],
    }
    selected, audit = review.select_preferred_observation_paths(
        state, 7, paths,
        preferred_source="full_colmap_sam3_refinement",
        minimum_preferred=2,
    )
    assert selected == [refined_a, refined_b]
    assert audit["mode"] == "preferred_source"
    selected, audit = review.select_preferred_observation_paths(
        state, 7, paths[:-1],
        preferred_source="full_colmap_sam3_refinement",
        minimum_preferred=2,
    )
    assert selected == paths[:-1]
    assert audit["mode"] == "all_resolved_fallback"


def test_review_hydrates_pose_selection_context_from_canonical_state() -> None:
    review = _load(
        "farm_metric_context_hydration",
        "scripts/semantics/review_farm_object_crops.py",
    )
    rows = [{"id": 9, "category": "machine"}, {"id": 77, "category": "box"}]
    state = {
        "object_id": np.asarray([77, 9, 123], dtype=np.int64),
        "object_box_centers_m": np.asarray(
            [[7.0, 7.1, 7.2], [0.1, 0.2, 0.3], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
        "object_box_dimensions_m": np.asarray(
            [[0.7, 0.8, 0.9], [1.1, 1.2, 1.3], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        ),
    }

    audit = review.hydrate_rows_from_state(rows, state)

    assert audit == {"matched_rows": 2, "position_rows": 2, "dimension_rows": 2}
    assert np.allclose(rows[0]["position_world_m"], [0.1, 0.2, 0.3])
    assert np.allclose(rows[0]["metric_dimensions_m"], [1.1, 1.2, 1.3])
    assert np.allclose(rows[1]["position_world_m"], [7.0, 7.1, 7.2])


def _rolled_pose_index(turns_clockwise: int) -> dict:
    # Start with image right +X, image down -Y for world up +Y. Rotating the
    # source camera clockwise changes its two image-plane axes accordingly.
    angle = math.radians(90.0 * turns_clockwise)
    right = [math.cos(angle), math.sin(angle), 0.0]
    down = [math.sin(angle), -math.cos(angle), 0.0]
    return {
        3: {
            "camera_center_world_m": [0.0, 0.0, 0.0],
            "camera_forward_world": [0.0, 0.0, 1.0],
            "camera_right_world": right,
            "camera_down_world": down,
        }
    }


def test_gravity_projection_selects_inverse_exact_quarter_turn() -> None:
    for source_turns in range(4):
        audit = gravity_upright_orientation(
            3,
            frame_pose_index=_rolled_pose_index(source_turns),
            world_up_vector=[0.0, 1.0, 0.0],
            world_up_source="test",
        )
        assert audit["applied_quarter_turns_ccw"] == source_turns
        assert audit["applied_rotation_degrees_ccw"] in {0, 90, 180, 270}
        assert abs(float(audit["residual_roll_degrees"])) < 1.0e-8


def test_frame_pose_loader_retains_image_axes_for_upright_provenance(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames.json"
    frames.write_text(
        json.dumps({
            "pose_translation_units": "m",
            "frames": [{
                "source_image": "cam00_000123_center.png",
                "T_world_cam": [
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, -1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }],
        }),
        encoding="utf-8",
    )
    pose = load_frame_pose_index(frames)[0]
    assert pose["camera_right_world"] == [0.0, 1.0, 0.0]
    assert pose["camera_down_world"] == [1.0, 0.0, 0.0]
    assert pose["T_world_cam"][1][3] == 2.0
    assert pose["source_image"] == "cam00_000123_center.png"


def test_mask_and_both_context_panels_share_identical_quarter_turn(
    tmp_path: Path,
) -> None:
    review = _load(
        "farm_upright_mask_alignment", "scripts/semantics/review_farm_object_crops.py"
    )
    source = np.full((180, 260, 3), [70, 120, 70], dtype=np.uint8)
    source[55:95, 150:170] = [10, 20, 240]
    crop = source[55:95, 150:170].copy()
    ok, encoded = cv2.imencode(".jpg", crop)
    assert ok
    raw = np.ones((40, 20), dtype=np.uint8)
    sidecar = tmp_path / "mask.npz"
    np.savez_compressed(
        sidecar,
        crop_jpeg_bytes=np.asarray(encoded, dtype=np.uint8),
        raw_bits=np.packbits(raw.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(raw.shape, dtype=np.int32),
        crop_bbox_xyxy=np.asarray([0, 0, 20, 40], dtype=np.int32),
        raw_bbox_xyxy=np.asarray([150, 55, 170, 95], dtype=np.int32),
    )
    with np.load(sidecar, allow_pickle=False) as payload:
        rotated = review._mask_grounded_crop(
            payload,
            bytes(np.asarray(encoded, dtype=np.uint8)),
            source_image=source,
            natural_context_panel=True,
            quarter_turns_ccw=1,
        )
    image = cv2.imdecode(np.frombuffer(rotated, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[1] % 2 == 0
    panel_width = image.shape[1] // 2
    mask_panel = image[:, :panel_width]
    context_panel = image[:, panel_width:]
    mask_red = (mask_panel[:, :, 2] > 170) & (
        mask_panel[:, :, 2].astype(np.int16)
        > mask_panel[:, :, 1].astype(np.int16) + 50
    )
    context_red = (context_panel[:, :, 2] > 170) & (
        context_panel[:, :, 2].astype(np.int16)
        > context_panel[:, :, 1].astype(np.int16) + 50
    )
    intersection = int(np.logical_and(mask_red, context_red).sum())
    union = int(np.logical_or(mask_red, context_red).sum())
    assert union > 100
    # CONTEXT intentionally recolours the boundary yellow, so compare the
    # retained red interiors and their centroid rather than demanding boundary
    # identity after JPEG encoding.
    assert intersection / union > 0.55
    expanded_mask = cv2.dilate(
        mask_red.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)
    ).astype(bool)
    assert float(np.logical_and(context_red, expanded_mask).sum()) / float(
        context_red.sum()
    ) > 0.95


def test_missing_pose_is_recorded_and_preserves_source_orientation(
    tmp_path: Path,
) -> None:
    review = _load(
        "farm_upright_no_pose", "scripts/semantics/review_farm_object_crops.py"
    )
    audit = gravity_upright_orientation(
        9,
        frame_pose_index={},
        world_up_vector=[0.0, -1.0, 0.0],
        world_up_source="test",
    )
    assert audit["status"] == "unavailable"
    assert audit["reason"] == "missing_T_world_cam"
    assert audit["applied_quarter_turns_ccw"] == 0

    image = np.zeros((30, 50, 3), dtype=np.uint8)
    image[:10, :10] = [20, 20, 240]
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path = tmp_path / "img_000009_det_0000.npz"
    np.savez_compressed(
        path, crop_jpeg_bytes=np.asarray(encoded, dtype=np.uint8)
    )
    result = review.crop_candidates(
        [path], frame_pose_index={}, world_up_vector=[0.0, -1.0, 0.0]
    )
    decoded = cv2.imdecode(
        np.frombuffer(result[0][1], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape[:2] == (30, 50)
    assert int(decoded[:10, :10, 2].mean()) > 150


def test_semantic_event_records_upright_source_orientation_and_hash() -> None:
    review = _load(
        "farm_upright_event_provenance", "scripts/semantics/review_farm_object_crops.py"
    )
    crops = [(Path("img_000003_det_0000.npz"), b"jpeg")]
    manifest = gravity_upright_evidence_manifest(
        crops,
        frame_pose_index=_rolled_pose_index(1),
        world_up_vector=[0.0, 1.0, 0.0],
        world_up_source="geometry_report:$.up.vector",
    )
    assert manifest["provenance_complete"] is True
    assert len(manifest["fingerprint_sha256"]) == 64
    crop = manifest["crops"][0]
    assert crop["source_orientation"] == "clockwise_90"
    assert crop["applied_quarter_turns_ccw"] == 1
    assert crop["operation"] == "numpy.rot90_no_interpolation"
    event = review.semantic_evidence_event(
        {
            "id": 7,
            "review_category": "pallet",
            "review_description": "bounded slotted platform",
            "review_confidence": 0.9,
            "review_decision": "keep",
        },
        "initial_blind_review",
        "object:7:initial_blind_review",
        crops=crops,
        frame_pose_index=_rolled_pose_index(1),
        world_up_vector=[0.0, 1.0, 0.0],
        world_up_source="geometry_report:$.up.vector",
        confirmation_eligible=False,
        ineligibility_reason="candidate_conditioned_on_initial_review",
    )
    assert event["gravity_upright_normalization"] == manifest
    assert event["semantic_vote_independence"] == "not_independent"


def test_candidate_verification_reusing_blind_fold_is_not_requested() -> None:
    review = _load(
        "farm_disjoint_verification_gate", "scripts/semantics/review_farm_object_crops.py"
    )
    initial = [_crop_row(1, 10), _crop_row(2, 10)]
    reused = [_crop_row(2, 10), _crop_row(3, 10)]
    accepted, audit = review.enforce_disjoint_verification_fold(
        initial, reused, {"partition_id": "view-fold-0-of-1"}
    )
    assert accepted == []
    separation = audit["blind_verification_separation"]
    assert separation["status"] == "unavailable"
    assert separation["overlap_image_ids"] == [2]

    disjoint, audit = review.enforce_disjoint_verification_fold(
        initial, [_crop_row(3, 10), _crop_row(4, 10)],
        {"partition_id": "view-fold-1-of-2"},
    )
    assert _crop_ids(disjoint) == {3, 4}
    assert audit["blind_verification_separation"]["status"] == "disjoint"


def test_world_up_contract_prefers_cli_then_geometry_then_default() -> None:
    review = _load(
        "farm_world_up_contract", "scripts/semantics/review_farm_object_crops.py"
    )
    geometry = {"up": {"vector": [0.0, 0.0, 2.0]}}
    assert review.resolve_world_up_contract(None, geometry)["vector"] == [0.0, 0.0, 1.0]
    cli = review.resolve_world_up_contract("1,0,0", geometry)
    assert cli["vector"] == [1.0, 0.0, 0.0]
    assert cli["source"] == "cli:--world-up-vector"
    assert review.resolve_world_up_contract(None, None)["vector"] == [0.0, -1.0, 0.0]
