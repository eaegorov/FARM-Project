from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_farm_retention_funnel", ROOT / "scripts/evaluation/build_farm_retention_funnel.py"
)
assert SPEC and SPEC.loader
funnel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(funnel)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_quality_funnel_separates_structural_pass_from_recall_warn(tmp_path: Path) -> None:
    run = tmp_path / "run"
    state = {
        "object_id": torch.arange(10),
        "count": torch.tensor([1, 1, 2, 3, 3, 4, 5, 5, 8, 10]),
        "object_mask_observations": [[] for _ in range(10)],
    }
    (run / "mapping").mkdir(parents=True)
    torch.save({"state": state}, run / "mapping/scene_state_raw.pt")
    (run / "mapping/debug_trace.jsonl").write_text(
        json.dumps({
            "event": "frame", "batch_size": 2,
            "segmentation": {"n_raw": 20},
            "segmentation_post_filter": {"n_raw": 12},
        }) + "\n",
        encoding="utf-8",
    )
    _write(run / "manifest.json", {"scene_id": "synthetic"})
    _write(run / "qa/geometry/audit.json", {"evaluated_objects": 4, "passed_objects": 3})
    _write(run / "qa/visual_consistency/audit.json", {"retained_objects": 3})
    _write(run / "qa/surface_support/audit.json", {
        "passed_objects": 1, "rejected_objects": 2,
        "active_before": 3, "active_after": 3, "policy": {"enforce": False},
    })
    _write(run / "qa/semantics/consensus/semantic_consensus_report.json", {"resolved": 2})
    _write(run / "qa/semantics/consensus/semantic_consensus_catalog.json", [
        {
            "id": 1,
            "semantic_tier": "confirmed",
            "semantic_evidence_event_ids": ["blind_a", "blind_b"],
            "semantic_independent_group_count": 2,
            "semantic_unique_view_count": 6,
            "semantic_confirmation_eligible": True,
            "semantic_max_support_overlap": 0.25,
        },
        {"id": 2, "semantic_tier": "probable", "semantic_evidence_event_ids": ["blind_a"]},
        {"id": 3, "semantic_tier": "geometry_only", "semantic_evidence_event_ids": []},
    ])
    _write(run / "qa/dedup/audit.json", {"duplicate_groups": [], "suppressed_objects": 0})
    _write(run / "qa/acceptance/result.json", {
        "schema": "farm.final-acceptance.v1",
        "status": "WARN",
        "counts": {
            "presentation_after": 2,
            "held_objects": 1,
            "auto_holdout_clusters_initial": 1,
            "blocking_clusters_initial": 0,
            "remaining_auto_clusters": 0,
            "remaining_blocking_clusters": 0,
            "label_hard_errors": 0,
        },
        "classification_after": {
            "auto_holdout": 0,
            "release_blocking_ambiguous": 0,
            "similar_distinct": 4,
            "hidden_context": 6,
        },
        "warnings": [{"code": "high_label_uncertainty", "count": 1}],
    })
    _write(run / "final/catalog.json", [{"id": 1}, {"id": 2}, {"id": 3}])
    _write(run / "final/presentation_catalog.json", [{"id": 1}, {"id": 2}])

    report = funnel.build_funnel(run)
    assert report["structural_status"] == "PASS"
    assert report["quality_status"] == "WARN"
    assert report["funnel"]["detections_raw"] == 20
    assert report["funnel"]["tracks_ge_3"] == 7
    assert report["details"]["surface_enforced"] is False
    assert report["details"]["acceptance_held_objects"] == 1
    assert report["details"]["residual_duplicate_like_visible_count"] == 0
    assert report["details"]["residual_similar_distinct_visible_count"] == 4
    assert report["details"]["residual_hidden_endpoint_count"] == 6
    assert "high_label_uncertainty" in {row["code"] for row in report["warnings"]}
    assert "surface_diagnostic_high_rejection" in {
        row["code"] for row in report["warnings"]
    }
    visual = tmp_path / "retention_4k.jpg"
    funnel.render_dashboard(report, visual)
    image = cv2.imread(str(visual))
    assert image is not None and image.shape[:2] == (2160, 3840)


def test_residual_candidates_separate_duplicate_like_from_distinct_and_hidden() -> None:
    rows = [
        {"first_id": 1, "second_id": 2, "raw_relation": "duplicate_strict"},
        {"first_id": 2, "second_id": 3, "raw_relation": "distinct"},
        {"first_id": 3, "second_id": 4, "raw_relation": "duplicate_overlap_fragment"},
    ]

    result = funnel.classify_residual_candidates(rows, {1, 2, 3})

    assert result["visible_duplicate_like"] == [rows[0]]
    assert result["visible_similar_distinct"] == [rows[1]]
    assert result["hidden_endpoint"] == [rows[2]]


def test_dashboard_warning_layout_reserves_footer_and_bounds_rows() -> None:
    warnings = [
        {"code": f"warning_{index}", "detail": "long diagnostic detail " * 8}
        for index in range(7)
    ]

    layout = funnel.dashboard_warning_layout(warnings)

    assert len(layout) == 4
    assert min(position for _, position in layout) >= 0.13
    assert "+4 additional warnings" in layout[-1][0]


def test_presentation_parent_invariant_resolves_chains_and_fails_closed() -> None:
    catalog = [
        {
            "id": 34, "semantic_tier": "probable", "metric_active": True,
            "display_status": "part_suppressed", "canonical_object_id": 21,
            "presentation_visible": False,
        },
        {
            "id": 21, "semantic_tier": "geometry_only", "metric_active": True,
            "display_status": "canonical", "canonical_object_id": -1,
            "presentation_visible": False,
        },
        {
            "id": 182, "semantic_tier": "probable", "metric_active": True,
            "display_status": "duplicate_suppressed", "canonical_object_id": 262,
            "presentation_visible": False,
        },
        {
            "id": 262, "semantic_tier": "probable", "metric_active": True,
            "display_status": "part_suppressed", "canonical_object_id": 256,
            "presentation_visible": False,
        },
        {
            "id": 256, "semantic_tier": "confirmed", "metric_active": True,
            "display_status": "canonical", "canonical_object_id": -1,
            "presentation_visible": True,
        },
        {
            "id": 99, "semantic_tier": "confirmed", "metric_active": True,
            "display_status": "duplicate_suppressed", "canonical_object_id": 99,
            "presentation_visible": False,
        },
    ]

    assert funnel.presentation_parent_violations(catalog, {256}) == [34, 99]


def test_false_confirmed_semantics_fail_structural_invariant(tmp_path: Path) -> None:
    run = tmp_path / "run"
    state = {
        "object_id": torch.tensor([1]),
        "count": torch.tensor([5]),
        "object_mask_observations": [[]],
    }
    (run / "mapping").mkdir(parents=True)
    torch.save({"state": state}, run / "mapping/scene_state_raw.pt")
    _write(run / "manifest.json", {"scene_id": "synthetic"})
    _write(run / "qa/geometry/audit.json", {"evaluated_objects": 1, "passed_objects": 1})
    _write(run / "qa/visual_consistency/audit.json", {"retained_objects": 1})
    _write(run / "qa/surface_support/audit.json", {
        "passed_objects": 1, "rejected_objects": 0,
        "active_before": 1, "active_after": 1, "policy": {"enforce": False},
    })
    _write(run / "qa/semantics/consensus/semantic_consensus_report.json", {"resolved": 1})
    _write(run / "qa/semantics/consensus/semantic_consensus_catalog.json", [{
        "id": 1,
        "semantic_tier": "confirmed",
        "semantic_evidence_event_ids": ["blind_a", "blind_b"],
        "semantic_independent_group_count": 1,
        "semantic_unique_view_count": 12,
        "semantic_confirmation_eligible": True,
        "semantic_max_support_overlap": 0.0,
    }])
    _write(run / "qa/dedup/audit.json", {"duplicate_groups": [], "suppressed_objects": 0})
    _write(run / "final/catalog.json", [{"id": 1}])
    _write(run / "final/presentation_catalog.json", [{"id": 1}])

    report = funnel.build_funnel(run)
    assert report["structural_status"] == "FAIL"
    assert "confirmed_semantics_violate_independence_contract" in report["structural_errors"]


def test_confirmed_semantic_contract_is_fail_closed() -> None:
    valid = {
        "semantic_tier": "confirmed",
        "semantic_independent_group_count": 2,
        "semantic_unique_view_count": 6,
        "semantic_confirmation_eligible": True,
        "semantic_max_support_overlap": 0.25,
    }
    assert funnel.confirmed_semantic_contract_violation(valid) is False
    invalid = [
        {"semantic_tier": "confirmed", "semantic_evidence_event_ids": ["a", "b"]},
        {**valid, "semantic_independent_group_count": 1},
        {**valid, "semantic_unique_view_count": 5},
        {**valid, "semantic_confirmation_eligible": False},
        {**valid, "semantic_max_support_overlap": 0.251},
    ]
    assert all(funnel.confirmed_semantic_contract_violation(row) for row in invalid)


def test_missing_canonical_mask_reference_is_structural_failure(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    state = {
        "object_id": torch.tensor([7]),
        "object_mask_observations": [[{"path": "/old/object_000007/img_000001_det_0000.npz"}]],
    }
    result = funnel._canonical_mask_integrity(state, root)
    assert result["canonical_contract_available"] is True
    assert result["missing_references"] == 1
    assert result["orphan_files"] == 0
