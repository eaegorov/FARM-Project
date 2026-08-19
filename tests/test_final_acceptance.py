from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_pipeline.final_acceptance import (  # noqa: E402
    AcceptancePolicy,
    apply_presentation_holdouts,
    build_overlap_clusters,
    candidate_budget,
    classify_pair_records,
    deterministic_label_sample,
    merge_pair_evidence,
    refresh_runtime_status,
    run_acceptance_policy,
    runtime_budget_seconds,
)


def _state(ids: list[int], categories: list[str] | None = None) -> dict:
    size = len(ids)
    return {
        "object_id": torch.tensor(ids, dtype=torch.int64),
        "active": torch.ones(size, dtype=torch.bool),
        "object_category": categories or ["fixture"] * size,
        "object_semantic_tier": ["probable"] * size,
        "object_display_status": ["canonical"] * size,
        "object_duplicate_canonical_id": torch.full((size,), -1, dtype=torch.int64),
        "object_duplicate_group_ids": [[] for _ in ids],
        "object_duplicate_reason": [""] * size,
        "object_geometry_status": ["geometry_pass"] * size,
        "object_compound_boxes": [[] for _ in ids],
        "object_box_dimensions_m": torch.ones((size, 3), dtype=torch.float32),
        "object_box_centers_m": torch.arange(size * 3, dtype=torch.float32).reshape(size, 3),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * size),
        "object_evidence_count": torch.arange(1, size + 1, dtype=torch.float32),
        "object_geometry_projected_box_iou": torch.full((size,), 0.8),
        "object_geometry_inside_rate": torch.full((size,), 0.8),
        "object_geometry_voxel_inside_rate": torch.full((size,), 0.8),
        "object_mask_observations": [[{"path": f"mask_{value}.npz"}] for value in ids],
        "features": torch.arange(size * 4, dtype=torch.float32).reshape(size, 4),
    }


def _pair(first: int, second: int, **overrides) -> dict:
    row = {
        "first_id": first,
        "second_id": second,
        "feature_cosine": 0.98,
        "center_distance_m": 0.20,
        "first_near_second": 0.97,
        "second_near_first": 0.99,
        "mask_iou": 0.75,
        "mask_containment": 0.99,
        "mask_area_ratio": 0.80,
        "raw_relation": "distinct",
    }
    row.update(overrides)
    return row


def _semantic(object_id: int, category: str = "fixture", tier: str = "probable") -> dict:
    confirmed = tier == "confirmed"
    views = 6 if confirmed else 3
    groups = 2 if confirmed else 1
    return {
        "id": object_id,
        "category": category,
        "semantic_tier": tier,
        "semantic_independent_group_count": groups,
        "semantic_unique_view_count": views,
        "semantic_confirmation_eligible": confirmed,
        "semantic_max_support_overlap": 0.20 if confirmed else None,
        "physical_form_assessment": {"hard_veto": False},
        "candidates": [{
            "source": "blind_fixture",
            "event_id": f"object:{object_id}:blind_fixture",
            "evidence_fingerprint_sha256": f"fingerprint-{object_id}",
            "category": category,
            "decision": "keep",
            "confidence": 0.95,
            "confirmation_eligible": True,
        }],
    }


def test_pair_union_and_three_auto_rules_are_order_invariant() -> None:
    state = _state([1, 2, 3, 4, 5, 6], ["unit", "unit", "a", "b", "c", "d"])
    audit = {
        "pair_evidence": [
            _pair(
                2, 1, raw_relation="part_contained",
                mask_iou=0.68, mask_containment=0.94, mask_area_ratio=0.68,
            ),
            _pair(3, 4, raw_relation="duplicate_fragment", first_near_second=0.91,
                  second_near_first=0.91, mask_containment=0.985),
        ],
        "residual_strong_candidates": [
            _pair(1, 2, raw_relation="distinct", feature_cosine=0.96),
            _pair(5, 6, feature_cosine=0.91, mask_iou=0.84,
                  mask_containment=0.97, mask_area_ratio=0.86),
        ],
        "review_only_conflicts": [],
    }
    records, errors = merge_pair_evidence(audit)
    assert errors == []
    assert [(row["first_id"], row["second_id"]) for row in records] == [(1, 2), (3, 4), (5, 6)]

    classified, errors = classify_pair_records(records, state, {1, 2, 3, 4, 5, 6})
    assert errors == []
    assert [row["classification"] for row in classified] == ["auto_holdout"] * 3
    assert {row["matched_rule"] for row in classified} == {
        "same_category_mask_geometry_consensus",
        "resolver_duplicate_with_mask_geometry",
        "overwhelming_label_independent_overlap",
    }


def test_pair_boundary_separates_ambiguous_distinct_and_hidden() -> None:
    state = _state([1, 2, 3, 4], ["first", "second", "third", "fourth"])
    records, _ = merge_pair_evidence({
        "residual_strong_candidates": [
            _pair(1, 2, feature_cosine=0.93, mask_iou=0.60,
                  mask_containment=0.92, mask_area_ratio=0.65,
                  first_near_second=0.86, second_near_first=0.86),
            _pair(2, 3, feature_cosine=0.70, mask_iou=0.20,
                  mask_containment=0.30, mask_area_ratio=0.20),
            _pair(3, 4),
        ]
    })
    classified, _ = classify_pair_records(records, state, {1, 2, 3})
    assert [row["classification"] for row in classified] == [
        "release_blocking_ambiguous",
        "similar_distinct",
        "hidden_context",
    ]


def test_cluster_holdout_is_deterministic_idempotent_and_presentation_only() -> None:
    state = _state([20, 10, 30], ["unit", "unit", "unit"])
    state["object_evidence_count"] = torch.tensor([2.0, 5.0, 1.0])
    records, _ = merge_pair_evidence({
        "pair_evidence": [_pair(20, 10), _pair(20, 30)]
    })
    classified, _ = classify_pair_records(records, state, {10, 20, 30})
    clusters = build_overlap_clusters(classified, state)
    assert len(clusters) == 1
    assert clusters[0]["canonical_id"] == 10

    protected = {
        key: value
        for key, value in state.items()
        if key not in {
            "object_display_status", "object_duplicate_canonical_id",
            "object_duplicate_group_ids", "object_duplicate_reason",
        }
    }
    output, holdouts = apply_presentation_holdouts(state, clusters)
    assert [row["id"] for row in holdouts] == [20, 30]
    assert output["active"] is protected["active"]
    assert output["object_box_dimensions_m"] is protected["object_box_dimensions_m"]
    assert output["object_mask_observations"] is protected["object_mask_observations"]
    assert torch.equal(output["active"], state["active"])
    assert torch.equal(output["object_box_centers_m"], state["object_box_centers_m"])

    second, second_holdouts = apply_presentation_holdouts(output, clusters)
    assert second_holdouts == holdouts
    assert second["object_display_status"] == output["object_display_status"]
    assert torch.equal(
        second["object_duplicate_canonical_id"], output["object_duplicate_canonical_id"]
    )


def test_blocking_edge_prevents_component_auto_holdout() -> None:
    state = _state([1, 2, 3], ["unit", "unit", "other"])
    records, _ = merge_pair_evidence({
        "pair_evidence": [
            _pair(1, 2),
            _pair(2, 3, feature_cosine=0.93, mask_iou=0.60,
                  mask_containment=0.92, mask_area_ratio=0.65,
                  first_near_second=0.86, second_near_first=0.86),
        ]
    })
    classified, _ = classify_pair_records(records, state, {1, 2, 3})
    cluster = build_overlap_clusters(classified, state)[0]
    assert cluster["classification"] == "release_blocking_ambiguous"
    output, holdouts = apply_presentation_holdouts(state, [cluster])
    assert holdouts == []
    assert output["object_display_status"] == state["object_display_status"]


def test_release_status_truth_table_and_one_pass_cleanup() -> None:
    state = _state([1, 2], ["unit", "unit"])
    semantic = [_semantic(1, "unit"), _semantic(2, "unit")]
    audit = {"residual_strong_candidates": [_pair(1, 2)]}
    output, report, _, _ = run_acceptance_policy(
        state, audit, semantic, scene_id="fixture",
        decodable_crop_counts={1: 3, 2: 3}, apply_holdouts=True,
    )
    assert report["status"] == "WARN"
    assert report["counts"]["held_objects"] == 1
    assert report["counts"]["remaining_auto_clusters"] == 0
    assert report["runtime"]["automatic_holdout_passes"] == 1
    assert sum(status.endswith("_suppressed") for status in output["object_display_status"]) == 1

    _, verify, _, _ = run_acceptance_policy(
        output, audit, semantic, scene_id="fixture",
        decodable_crop_counts={1: 3, 2: 3}, apply_holdouts=False,
    )
    assert verify["counts"]["held_objects"] == 0
    assert verify["counts"]["remaining_auto_clusters"] == 0

    ambiguous = {"residual_strong_candidates": [
        _pair(1, 2, feature_cosine=0.93, mask_iou=0.60,
              mask_containment=0.92, mask_area_ratio=0.65,
              first_near_second=0.86, second_near_first=0.86)
    ]}
    _, failed, _, _ = run_acceptance_policy(
        state, ambiguous, semantic, scene_id="fixture",
        decodable_crop_counts={1: 3, 2: 3}, apply_holdouts=True,
    )
    assert failed["status"] == "FAIL"
    assert {row["code"] for row in failed["errors"]} == {
        "release_blocking_ambiguous_overlap_cluster"
    }


def test_semantic_contracts_and_structured_recovery_are_fail_closed() -> None:
    state = _state([1, 2, 3], ["robust", "form", "bad"])
    state["object_semantic_tier"] = ["confirmed", "probable", "probable"]
    good = _semantic(1, "robust", "confirmed")
    recovery = _semantic(2, "form")
    recovery["candidates"] = [{
        "source": "blind_fixture", "category": "wrong", "decision": "keep",
        "confirmation_eligible": True, "confidence": 1.0,
        "event_id": "object:2:blind", "evidence_fingerprint_sha256": "two",
    }]
    recovery["physical_form_recovery"] = {
        "visible_form_category": "form", "topology": "standalone",
        "complete_bounded": True, "carrier_category": "", "payload_categories": [],
    }
    recovery["physical_form_recovery_assessment"] = {
        "triggered": True, "accepted": True, "action": "recovered_probable",
        "category": "form", "topology": "standalone", "complete_bounded": True,
        "maximum_semantic_tier": "probable",
        "metric_physical_assessment": {"hard_veto": False},
    }
    bad = _semantic(3, "bad")
    bad["physical_form_recovery"] = {
        "visible_form_category": "bad", "topology": "component_only",
        "complete_bounded": False, "carrier_category": "carrier",
        "payload_categories": ["payload"],
    }
    bad["physical_form_recovery_assessment"] = copy.deepcopy(
        recovery["physical_form_recovery_assessment"]
    )
    bad["physical_form_recovery_assessment"]["category"] = "bad"

    _, report, _, labels = run_acceptance_policy(
        state, {}, [good, recovery, bad], scene_id="fixture",
        decodable_crop_counts={1: 3, 2: 3, 3: 3}, apply_holdouts=True,
    )
    assert report["status"] == "FAIL"
    by_id = {row["id"]: row for row in labels}
    assert by_id[1]["hard_error_codes"] == []
    assert by_id[2]["hard_error_codes"] == []
    assert "recovery_guard_not_standalone" in by_id[3]["hard_error_codes"]
    assert "recovery_carrier_payload_conflict" in by_id[3]["hard_error_codes"]


def test_missing_crop_or_false_confirmed_support_is_hard_failure() -> None:
    state = _state([1], ["unit"])
    state["object_semantic_tier"] = ["confirmed"]
    row = _semantic(1, "unit", "confirmed")
    row["semantic_independent_group_count"] = 1
    _, report, _, labels = run_acceptance_policy(
        state, {}, [row], scene_id="fixture",
        decodable_crop_counts={1: 0}, apply_holdouts=True,
    )
    assert report["status"] == "FAIL"
    assert set(labels[0]["hard_error_codes"]) >= {
        "confirmed_independence_contract_violation",
        "presentation_label_has_no_decodable_crop",
    }


def test_deterministic_sample_is_bounded_stratified_and_order_independent() -> None:
    rows = []
    for object_id in range(100):
        rows.append({
            "id": object_id,
            "category": f"category-{object_id % 9}",
            "semantic_tier": "confirmed" if object_id % 3 == 0 else "probable",
            "hard_error_codes": ["fixture"] if object_id in {7, 91} else [],
            "uncertainty_score": (object_id % 11) / 10.0,
            "metric_volume_m3": float(object_id + 1),
        })
    first = deterministic_label_sample(rows, "scene")
    shuffled = list(rows)
    random.Random(17).shuffle(shuffled)
    second = deterministic_label_sample(shuffled, "scene")
    assert len(first) == 20
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {7, 91}.issubset({row["id"] for row in first})
    assert len({(row["semantic_tier"], row["category"]) for row in first}) > 3


def test_runtime_and_candidate_budgets_are_exact_and_fail_closed() -> None:
    assert runtime_budget_seconds(0) == 30.0
    assert runtime_budget_seconds(2500) == 50.0
    assert runtime_budget_seconds(10_000) == 90.0
    assert candidate_budget(2) == 128
    assert candidate_budget(100_000) == 50_000

    state = _state([1], ["unit"])
    policy = AcceptancePolicy(candidate_multiplier=1, candidate_absolute_cap=1)
    audit = {"pair_evidence": [_pair(1, 2), _pair(1, 3)]}
    _, report, _, _ = run_acceptance_policy(
        state, audit, [_semantic(1)], scene_id="fixture",
        decodable_crop_counts={1: 3}, policy=policy,
    )
    assert report["status"] == "FAIL"
    assert "acceptance_candidate_budget_exhausted" in {
        row["code"] for row in report["errors"]
    }
    assert report["counts"]["held_objects"] == 0
    assert report["runtime"]["automatic_holdout_passes"] == 0

    refreshed = refresh_runtime_status({
        "status": "PASS", "errors": [], "warnings": [],
        "runtime": {"budget_seconds": 30.0},
    }, 30.001)
    assert refreshed["status"] == "WARN"
    assert refreshed["warnings"][0]["code"] == "acceptance_runtime_budget_exceeded"
