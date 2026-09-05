from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import farm_runtime.second_chance_refinement as second_chance
from farm_runtime.full_colmap_rescue import (
    backproject_mask_points,
    voxel_downsample_points,
)
from farm_runtime.second_chance_refinement import (
    PLAN_PROMPT_BINDING_SCHEMA,
    PROMPT_SCHEMA,
    evaluate_second_chance_candidate,
    load_concept_prompt_manifest,
    select_cross_view_consistent_candidates,
    select_unique_identity_compatible_mapping_mask,
    validate_acceptance_prompt_plan_binding,
)


def _frame() -> dict:
    return {
        "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        "T_world_cam": np.eye(4, dtype=np.float64).tolist(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_audit(object_id: int = 37, category: str = "sign") -> dict:
    return {
        "schema": "farm.whole-object-evidence-audit.v1",
        "objects": [
            {
                "object_id": object_id,
                "identity_consensus": {
                    "confirmed_category": category,
                    "confirmed_categories": [category],
                    "conflicting_categories": [],
                    "group_diagnostics": {
                        category: {
                            "confirmation_ready": True,
                            "reason": "independent_view_consensus",
                        }
                    },
                },
                "semantic_identity_re_adjudication": {
                    "schema": "farm.independent-blind-label-consensus.v1",
                    "status": "accepted",
                    "accepted": True,
                    "initial_category": category,
                    "verification_category": category,
                    "canonical_category": category,
                    "reason_codes": ["exact_safe_canonical_noun_agreement"],
                    "identity_contract_source": "raw_label_contract",
                    "initial_event_id": "blind-a",
                    "verification_event_id": "blind-b",
                    "evidence_independence": {
                        "independent": True,
                        "pose_diversity": {"pose_independent": True},
                    },
                },
            }
        ],
    }


def _write_manifest(
    path: Path,
    *,
    authorization: str,
    prompt: str = "sign",
    object_id: int = 37,
    evidence_path: Path | None = None,
    evidence_sha256: str | None = None,
) -> None:
    provenance = {
        "source": "frozen-independent-qwen-audit",
        "prompt_frozen_before_target_selection": True,
    }
    if evidence_path is not None:
        provenance["evidence_path"] = str(evidence_path)
    if evidence_sha256 is not None:
        provenance["evidence_sha256"] = evidence_sha256
    path.write_text(
        json.dumps(
            {
                "schema": PROMPT_SCHEMA,
                "objects": [
                    {
                        "object_id": object_id,
                        "prompt": prompt,
                        "authorization": authorization,
                        "provenance": provenance,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_acceptance_prompt_is_bound_to_exact_immutable_semantic_audit(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "whole_object_audit.json"
    audit_path.write_text(json.dumps(_semantic_audit()), encoding="utf-8")
    manifest_path = tmp_path / "prompts.json"
    _write_manifest(
        manifest_path,
        authorization="acceptance",
        evidence_path=audit_path,
        evidence_sha256=_sha256(audit_path),
    )

    prompts = load_concept_prompt_manifest(manifest_path)

    row = prompts[37]
    assert row["identity_anchor_authorized"] is True
    assert row["bound_semantic_evidence"]["canonical_category"] == "sign"
    assert row["bound_semantic_evidence"]["evidence_sha256"] == _sha256(audit_path)
    assert all(row["bound_semantic_evidence"]["semantic_evidence_gates"].values())


def _bound_plan(manifest_path: Path) -> dict:
    return {
        "policy": {
            "heldout_reference_only": True,
            "post_adaptation_unseen_heldout": True,
            "state_fit_authorized": False,
            "category_agnostic_view_ranking": True,
            "target_conditioned_view_selection": False,
        },
        "objects": [{"object_id": 37, "heldout_views": []}],
        "concept_prompt_selection_contract": {
            "schema": PLAN_PROMPT_BINDING_SCHEMA,
            "selection_role": "post_adaptation_unseen_heldout",
            "fit_candidate_authorized": False,
            "release_authorized": False,
            "prompt_manifest_sha256": _sha256(manifest_path),
            "selection_frozen_after_prompt_manifest": True,
            "category_agnostic_view_ranking": True,
            "target_conditioned_view_selection": False,
            "prompt_used_for_view_ranking": False,
        },
    }


def test_acceptance_prompt_requires_exact_frozen_plan_binding(tmp_path: Path) -> None:
    audit_path = tmp_path / "whole_object_audit.json"
    audit_path.write_text(json.dumps(_semantic_audit()), encoding="utf-8")
    manifest_path = tmp_path / "prompts.json"
    _write_manifest(
        manifest_path,
        authorization="acceptance",
        evidence_path=audit_path,
        evidence_sha256=_sha256(audit_path),
    )
    prompts = load_concept_prompt_manifest(manifest_path)

    audit = validate_acceptance_prompt_plan_binding(
        _bound_plan(manifest_path), prompts, manifest_path,
        view_role="heldout-reference",
    )

    assert audit["verified"] is True
    assert audit["acceptance_binding_required"] is True
    assert audit["acceptance_object_ids"] == [37]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "concept_prompt_selection_contract"),
        ("hash", "exact_prompt_manifest_sha256"),
        ("target_conditioned", "target_conditioned_view_selection_disabled"),
        ("object", "acceptance_objects_present_in_plan"),
    ],
)
def test_acceptance_prompt_rejects_missing_or_mismatched_plan_binding(
    tmp_path: Path, mutation: str, message: str
) -> None:
    audit_path = tmp_path / "whole_object_audit.json"
    audit_path.write_text(json.dumps(_semantic_audit()), encoding="utf-8")
    manifest_path = tmp_path / "prompts.json"
    _write_manifest(
        manifest_path,
        authorization="acceptance",
        evidence_path=audit_path,
        evidence_sha256=_sha256(audit_path),
    )
    prompts = load_concept_prompt_manifest(manifest_path)
    plan = _bound_plan(manifest_path)
    if mutation == "missing":
        plan.pop("concept_prompt_selection_contract")
    elif mutation == "hash":
        plan["concept_prompt_selection_contract"]["prompt_manifest_sha256"] = (
            "0" * 64
        )
    elif mutation == "target_conditioned":
        plan["concept_prompt_selection_contract"][
            "target_conditioned_view_selection"
        ] = True
    else:
        plan["objects"] = [{"object_id": 38, "heldout_views": []}]

    with pytest.raises(ValueError, match=message):
        validate_acceptance_prompt_plan_binding(
            plan, prompts, manifest_path, view_role="heldout-reference"
        )


def test_acceptance_prompt_train_binding_is_fit_candidate_not_release(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "whole_object_audit.json"
    audit_path.write_text(json.dumps(_semantic_audit()), encoding="utf-8")
    manifest_path = tmp_path / "prompts.json"
    _write_manifest(
        manifest_path,
        authorization="acceptance",
        evidence_path=audit_path,
        evidence_sha256=_sha256(audit_path),
    )
    prompts = load_concept_prompt_manifest(manifest_path)
    plan = _bound_plan(manifest_path)
    plan["policy"].update(
        heldout_reference_only=False,
        post_adaptation_unseen_heldout=False,
        state_fit_authorized=True,
    )
    plan["concept_prompt_selection_contract"].update(
        selection_role="train_fit_candidate",
        fit_candidate_authorized=True,
    )

    audit = validate_acceptance_prompt_plan_binding(
        plan, prompts, manifest_path, view_role="train"
    )

    assert audit["fit_candidate_authorized"] is True
    assert audit["evaluation_only"] is False
    assert audit["release_authorized"] is False


def test_diagnostic_prompt_does_not_require_plan_binding(tmp_path: Path) -> None:
    manifest_path = tmp_path / "diagnostic.json"
    _write_manifest(
        manifest_path,
        authorization="diagnostic_only",
        object_id=2,
        prompt="fire extinguisher",
    )
    prompts = load_concept_prompt_manifest(manifest_path)

    audit = validate_acceptance_prompt_plan_binding({}, prompts, manifest_path)

    assert audit["verified"] is True
    assert audit["acceptance_binding_required"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "SHA-256"),
        ("prompt", "prompt_exactly_matches_canonical_category"),
        ("object", "exactly one matching object row"),
    ],
)
def test_acceptance_prompt_rejects_unbound_or_mismatched_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    audit_path = tmp_path / "whole_object_audit.json"
    audit_path.write_text(json.dumps(_semantic_audit()), encoding="utf-8")
    manifest_path = tmp_path / "prompts.json"
    _write_manifest(
        manifest_path,
        authorization="acceptance",
        prompt="warning sign" if mutation == "prompt" else "sign",
        object_id=38 if mutation == "object" else 37,
        evidence_path=audit_path,
        evidence_sha256="0" * 64 if mutation == "hash" else _sha256(audit_path),
    )

    with pytest.raises(ValueError, match=message):
        load_concept_prompt_manifest(manifest_path)


def test_diagnostic_prompt_can_run_but_cannot_be_identity_anchor(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "diagnostic.json"
    _write_manifest(
        manifest_path,
        authorization="diagnostic_only",
        object_id=2,
        prompt="fire extinguisher",
    )

    row = load_concept_prompt_manifest(manifest_path)[2]

    assert row["identity_anchor_authorized"] is False
    assert row["bound_semantic_evidence"] is None


def test_unique_geometry_match_is_excluded_from_competition() -> None:
    candidate = np.zeros((64, 64), dtype=bool)
    candidate[20:42, 20:42] = True
    anchor = np.zeros_like(candidate)
    anchor[26:32, 26:32] = True
    target_mapping = candidate.copy()
    distractor = np.zeros_like(candidate)
    distractor[4:15, 4:15] = True
    depth = np.full(candidate.shape, 2.0, dtype=np.float32)
    depth[distractor] = 3.0

    selected, audit = select_unique_identity_compatible_mapping_mask(
        candidate, anchor, [target_mapping, distractor], depth
    )

    assert selected == 0
    assert audit["top_anchor_hits"] == 36
    assert audit["runner_up_anchor_hits"] == 0
    assert all(audit["uniqueness_gates"].values())


def test_ambiguous_geometry_matches_remain_competition() -> None:
    candidate = np.zeros((64, 64), dtype=bool)
    candidate[18:44, 18:44] = True
    anchor = np.zeros_like(candidate)
    anchor[26:34, 26:34] = True
    left = candidate.copy()
    right = candidate.copy()
    right[:, 18:20] = False
    depth = np.full(candidate.shape, 2.0, dtype=np.float32)

    selected, audit = select_unique_identity_compatible_mapping_mask(
        candidate, anchor, [left, right], depth
    )

    assert selected is None
    assert audit["top_anchor_hits"] == audit["runner_up_anchor_hits"]
    assert not all(audit["uniqueness_gates"].values())


def _candidate_and_world_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frame = _frame()
    mask = np.zeros((64, 64), dtype=bool)
    mask[12:52, 12:52] = True
    depth = np.ones(mask.shape, dtype=np.float32)
    world = voxel_downsample_points(
        backproject_mask_points(mask, depth, frame),
        voxel_size_m=0.02,
        max_points=5_000,
    )
    return mask, depth, world, frame


def test_large_neighbor_plane_with_sparse_identity_hits_is_rejected() -> None:
    mask, depth, world, frame = _candidate_and_world_points()
    anchor = np.zeros_like(mask)
    anchor[20:24, 20:24] = True

    audit, _ = evaluate_second_chance_candidate(
        mask,
        anchor,
        np.zeros_like(mask),
        depth,
        frame,
        identity_world_points=world[:12],
        confidence=0.99,
        minimum_confidence=0.70,
        maximum_competing_fraction=0.15,
        maximum_identity_surface_distance_m=0.001,
        minimum_identity_surface_points=12,
        minimum_identity_surface_fraction=0.20,
    )

    assert audit["metric_identity_surface_points"] >= 12
    assert audit["metric_identity_surface_fraction"] < 0.20
    assert audit["quality_gates"]["frontmost_metric_identity_surface"] is False
    assert audit["passed_local_gates"] is False


def test_foreground_identity_switch_rejects_valid_depth_neighbor() -> None:
    mask, foreground_depth, foreground_world, frame = _candidate_and_world_points()
    anchor = np.zeros_like(mask)
    anchor[22:30, 22:30] = True
    true_object_depth = np.full(mask.shape, 2.0, dtype=np.float32)
    true_object_world = voxel_downsample_points(
        backproject_mask_points(mask, true_object_depth, frame),
        voxel_size_m=0.02,
        max_points=5_000,
    )

    audit, _ = evaluate_second_chance_candidate(
        mask,
        anchor,
        np.zeros_like(mask),
        foreground_depth,
        frame,
        identity_world_points=true_object_world,
        confidence=0.99,
        minimum_confidence=0.70,
        maximum_competing_fraction=0.15,
    )

    assert foreground_world.shape[0] >= 24
    assert audit["depth_valid_fraction"] == 1.0
    assert audit["quality_gates"]["frontmost_metric_identity_surface"] is False
    assert audit["passed_local_gates"] is False


def test_two_physical_timestamps_need_bidirectional_cycle_and_voxel_consensus() -> None:
    mask, depth, world, frame = _candidate_and_world_points()
    candidates = []
    for key, timestamp in (("a", "100"), ("b", "200")):
        candidates.append(
            {
                "candidate_key": key,
                "physical_timestamp": timestamp,
                "passed_local_gates": True,
                "local_score": 0.9,
                "mask": mask,
                "depth": depth,
                "frame": frame,
                "world_points": world,
            }
        )

    selected, audit = select_cross_view_consistent_candidates(
        candidates,
        minimum_consensus_points=24,
        minimum_cycle_recall=0.45,
    )

    assert selected == {"a", "b"}
    assert audit["accepted"] is True
    assert audit["independent_physical_timestamps"] == 2
    assert audit["pairwise_cycle"][0]["passed"] is True


def test_exact_cross_view_search_recovers_clique_missed_by_greedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask, depth, world, frame = _candidate_and_world_points()
    timestamps = {"a": "1", "b": "2", "c": "3", "d": "4", "f": "4", "e": "5"}
    scores = {"d": 0.99, "f": 0.98, "e": 0.97, "a": 0.50, "b": 0.49, "c": 0.48}
    compatible = {
        frozenset(("a", "b")),
        frozenset(("a", "c")),
        frozenset(("b", "c")),
        frozenset(("d", "a")),
        frozenset(("e", "b")),
        frozenset(("f", "c")),
    }

    def fake_cycle(source: dict, target: dict) -> tuple[float, dict]:
        pair = frozenset((source["candidate_key"], target["candidate_key"]))
        recall = 1.0 if pair in compatible else 0.0
        return recall, {"cycle_recall": recall}

    monkeypatch.setattr(second_chance, "_cycle_recall", fake_cycle)
    candidates = [
        {
            "candidate_key": key,
            "physical_timestamp": timestamps[key],
            "passed_local_gates": True,
            "local_score": scores[key],
            "mask": mask,
            "depth": depth,
            "frame": frame,
            "world_points": world,
        }
        for key in ("d", "f", "e", "a", "b", "c")
    ]

    selected, audit = select_cross_view_consistent_candidates(
        candidates,
        minimum_physical_timestamps=3,
        minimum_consensus_points=24,
    )

    assert selected == {"a", "b", "c"}
    assert audit["accepted"] is True
    assert audit["exact_search_space"] == 47
    assert audit["evaluated_exact_combinations"] >= 1


def test_exact_cross_view_search_fails_closed_above_bound() -> None:
    mask, depth, world, frame = _candidate_and_world_points()
    candidates = [
        {
            "candidate_key": str(index),
            "physical_timestamp": str(index),
            "passed_local_gates": True,
            "local_score": 0.9,
            "mask": mask,
            "depth": depth,
            "frame": frame,
            "world_points": world,
        }
        for index in range(5)
    ]

    selected, audit = select_cross_view_consistent_candidates(
        candidates, maximum_exact_combinations=16
    )

    assert selected == set()
    assert audit["status"] == "rejected_exact_search_bound"


def test_same_physical_timestamp_cannot_fake_cross_view_independence() -> None:
    mask, depth, world, frame = _candidate_and_world_points()
    candidates = [
        {
            "candidate_key": key,
            "physical_timestamp": "100",
            "passed_local_gates": True,
            "local_score": 0.9,
            "mask": mask,
            "depth": depth,
            "frame": frame,
            "world_points": world,
        }
        for key in ("virtual-left", "virtual-right")
    ]

    selected, audit = select_cross_view_consistent_candidates(candidates)

    assert selected == set()
    assert audit["accepted"] is False
    assert "independent_physical_timestamps" in audit["rejection_reasons"]
