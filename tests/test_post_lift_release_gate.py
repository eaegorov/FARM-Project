from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.post_lift_release_gate import (
    PostLiftPolicy,
    evaluate_lift_release_contract,
    evaluate_mask_gate,
    evaluate_obb_gate,
    evaluate_scene_support_gate,
    find_obb_relation_review_pairs,
    gaussian_obb_support_metrics,
    gaussian_scene_support_metrics,
    resolve_obb_geometry,
    route_post_lift_object,
    scene_support_envelope,
    scene_points_to_meters,
    summarize_scene_release,
    validate_meters_per_scene_unit,
    verify_metric_scale_provenance,
)
from farm_runtime.obb_release_evidence import evaluate_obb_release_evidence
from scripts.evaluation.build_farm_post_lift_release_gate import (
    _frozen_heldout_contract,
    _semantic_gate,
    _train_obb_release_binding,
)


def _heldout() -> dict:
    return {
        "status": "verified",
        "independent_physical_timestamps": 3,
        "good_timestamps": 3,
        "summaries": {
            "iou": {"median": 0.78, "q25": 0.68},
            "precision": {"median": 0.86, "minimum": 0.71},
            "recall": {"median": 0.83, "minimum": 0.69},
            "largest_component_fraction": {"minimum": 0.97},
        },
    }


def _released_geometry_row() -> dict:
    return {
        "object_id": 7,
        "geometry_release_ready": True,
        "box_center_m": [1.0, 2.0, 3.0],
        "box_dimensions_m": [0.8, 0.4, 0.2],
        "box_wxyz": [1.0, 0.0, 0.0, 0.0],
        "train_geometry_evidence": evaluate_obb_release_evidence(
            geometry_fit_passed=True,
            independent_physical_timestamps=5,
            timestamp_median_projected_box_iou=0.8,
            timestamp_q25_projected_box_iou=0.7,
            orientation_materiality=0.5,
            orientation_confidence=0.8,
        ),
    }


def test_train_obb_binding_requires_exact_catalog_geometry() -> None:
    catalog = {
        "id": 7,
        "center_m": [1.0, 2.0, 3.0],
        "dimensions_m": [0.8, 0.4, 0.2],
        "wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    passed, reasons, details = _train_obb_release_binding(
        catalog,
        _released_geometry_row(),
        audit_schema="farm.object-geometry-audit.v3",
    )
    assert passed
    assert reasons == []
    assert details["geometry_matches_catalog"] is True

    catalog["center_m"] = [1.02, 2.0, 3.0]
    passed, reasons, _ = _train_obb_release_binding(
        catalog,
        _released_geometry_row(),
        audit_schema="farm.object-geometry-audit.v3",
    )
    assert not passed
    assert "catalog_obb_differs_from_train_released_obb" in reasons


def test_train_obb_binding_fails_closed_without_valid_strict_evidence() -> None:
    catalog = {
        "id": 7,
        "center_m": [1.0, 2.0, 3.0],
        "dimensions_m": [0.8, 0.4, 0.2],
        "wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    passed, reasons, _ = _train_obb_release_binding(
        catalog,
        None,
        audit_schema=None,
    )
    assert not passed
    assert reasons == [
        "train_geometry_audit_schema_invalid",
        "train_geometry_audit_object_missing",
    ]

    row = _released_geometry_row()
    row["train_geometry_evidence"]["policy"][
        "minimum_timestamp_median_projected_box_iou"
    ] = 0.1
    passed, reasons, _ = _train_obb_release_binding(
        catalog,
        row,
        audit_schema="farm.object-geometry-audit.v3",
    )
    assert not passed
    assert "geometry_release_evidence_policy_weaker_than_required" in reasons


def test_mask_gate_requires_exact_final_count_and_independent_views() -> None:
    policy = PostLiftPolicy()
    passed, reasons, _ = evaluate_mask_gate(
        _heldout(),
        {
            "build_timestamp_count": 5,
            "geometry_gate": "PASS",
            "provisional_gaussians": 900,
        },
        heldout_schema="farm.frozen-heldout-qc.v1",
        candidate_gaussian_count=900,
        final_gaussian_count=900,
        policy=policy,
    )
    assert passed
    assert reasons == []

    rejected, reasons, _ = evaluate_mask_gate(
        _heldout(),
        {
            "build_timestamp_count": 2,
            "geometry_gate": "PASS",
            "provisional_gaussians": 900,
        },
        heldout_schema="farm.frozen-heldout-qc.v1",
        candidate_gaussian_count=900,
        final_gaussian_count=899,
        policy=policy,
    )
    assert not rejected
    assert "insufficient_build_timestamps" in reasons
    assert "candidate_gaussian_count_mismatch" in reasons
    assert "build_gaussian_count_mismatch" in reasons


def test_mask_gate_rejects_fragmented_or_incomplete_heldout_result() -> None:
    row = _heldout()
    row["summaries"]["recall"]["median"] = 0.31
    row["summaries"]["largest_component_fraction"]["minimum"] = 0.40
    passed, reasons, _ = evaluate_mask_gate(
        row,
        {
            "build_timestamp_count": 5,
            "geometry_gate": "PASS",
            "provisional_gaussians": 900,
        },
        heldout_schema="farm.frozen-heldout-qc.v1",
        candidate_gaussian_count=900,
        final_gaussian_count=900,
        policy=PostLiftPolicy(),
    )
    assert not passed
    assert "median_recall_below_threshold" in reasons
    assert "fragmented_heldout_render" in reasons


def test_mask_gate_requires_upstream_per_object_geometry_pass() -> None:
    passed, reasons, metrics = evaluate_mask_gate(
        _heldout(),
        {
            "build_timestamp_count": 5,
            "geometry_gate": "REJECT",
            "provisional_gaussians": 900,
        },
        heldout_schema="farm.frozen-heldout-qc.v1",
        candidate_gaussian_count=900,
        final_gaussian_count=900,
        policy=PostLiftPolicy(),
    )
    assert not passed
    assert metrics["build_geometry_status"] == "REJECT"
    assert "build_geometry_gate_not_passed" in reasons


def test_mask_gate_rejects_implicit_internal_heldout_schema() -> None:
    passed, reasons, _ = evaluate_mask_gate(
        _heldout(),
        {
            "build_timestamp_count": 5,
            "geometry_gate": "PASS",
            "provisional_gaussians": 900,
        },
        heldout_schema="farm.gaussian-lift.heldout-qc.v1",
        candidate_gaussian_count=900,
        final_gaussian_count=900,
        policy=PostLiftPolicy(),
    )
    assert not passed
    assert "authoritative_frozen_heldout_schema_required" in reasons


def test_obb_support_gate_detects_large_empty_box_and_offset() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(-0.2, 0.2, size=(3000, 3))
    support = gaussian_obb_support_metrics(
        points,
        center_m=[1.9, 0.0, 0.0],
        dimensions_m=[4.0, 4.0, 4.0],
        wxyz=[1.0, 0.0, 0.0, 0.0],
        quantile=0.01,
    )
    passed, reasons = evaluate_obb_gate(support, policy=PostLiftPolicy())
    assert not passed
    assert "obb_too_large_for_gaussian_support" in reasons
    assert "obb_off_center_from_gaussian_support" in reasons


def test_obb_support_gate_accepts_tight_rotated_support() -> None:
    rng = np.random.default_rng(11)
    points = rng.uniform([-0.45, -0.2, -0.1], [0.45, 0.2, 0.1], size=(4000, 3))
    theta = np.deg2rad(35.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points = points @ rotation.T + np.asarray([2.0, -1.0, 0.5])
    support = gaussian_obb_support_metrics(
        points,
        center_m=[2.0, -1.0, 0.5],
        dimensions_m=[0.9, 0.4, 0.2],
        wxyz=[np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)],
        quantile=0.01,
    )
    passed, reasons = evaluate_obb_gate(support, policy=PostLiftPolicy())
    assert passed
    assert reasons == []


def test_obb_resolver_accepts_explicit_aliases_and_rotation_matrix() -> None:
    theta = np.deg2rad(30.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center, dimensions, resolved, sources = resolve_obb_geometry(
        {
            "position_world_m": [1.0, 2.0, 3.0],
            "metric_dimensions_m": [0.8, 0.4, 0.2],
            "obb": {"rotation_world_from_obb": rotation.tolist()},
        }
    )
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dimensions, [0.8, 0.4, 0.2])
    np.testing.assert_allclose(resolved, rotation)
    assert sources == {
        "center": "$.position_world_m",
        "dimensions": "$.metric_dimensions_m",
        "rotation": "$.obb.rotation_world_from_obb",
    }


def test_obb_resolver_handles_explicit_xyzw_and_rejects_ambiguous_cov6() -> None:
    _, _, rotation, sources = resolve_obb_geometry(
        {
            "center_m": [0.0, 0.0, 0.0],
            "dimensions_m": [1.0, 2.0, 3.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    )
    np.testing.assert_allclose(rotation, np.eye(3))
    assert sources["rotation"] == "$.orientation_xyzw"

    with pytest.raises(ValueError, match="rotation is missing"):
        resolve_obb_geometry(
            {
                "position_world_m": [0.0, 0.0, 0.0],
                "metric_dimensions_m": [1.0, 2.0, 3.0],
                "cov6": [1.0, 0.0, 0.0, 2.0, 0.0, 3.0],
            }
        )


def test_obb_resolver_rejects_conflicting_or_reflected_rotations() -> None:
    with pytest.raises(ValueError, match="conflicting OBB rotation"):
        resolve_obb_geometry(
            {
                "center_m": [0.0, 0.0, 0.0],
                "dimensions_m": [1.0, 1.0, 1.0],
                "wxyz": [1.0, 0.0, 0.0, 0.0],
                "orientation_wxyz": [0.0, 1.0, 0.0, 0.0],
            }
        )
    with pytest.raises(ValueError, match="right-handed"):
        resolve_obb_geometry(
            {
                "center_m": [0.0, 0.0, 0.0],
                "dimensions_m": [1.0, 1.0, 1.0],
                "rotation_matrix": np.diag([1.0, 1.0, -1.0]).tolist(),
            }
        )


def test_scene_support_gate_rejects_remote_floaters_and_scene_sized_mask() -> None:
    rng = np.random.default_rng(19)
    scene = rng.uniform(-5.0, 5.0, size=(8000, 3))
    envelope = scene_support_envelope(scene, quantile=0.01)
    remote = rng.uniform([19.8, -0.2, -0.2], [20.2, 0.2, 0.2], size=(900, 3))
    metrics = gaussian_scene_support_metrics(
        remote,
        center_m=[20.0, 0.0, 0.0],
        robust_support_dimensions_m=[0.4, 0.4, 0.4],
        scene_envelope=envelope,
        margin_fraction=0.05,
    )
    passed, reasons = evaluate_scene_support_gate(metrics, policy=PostLiftPolicy())
    assert not passed
    assert "gaussian_support_is_scene_spatial_outlier" in reasons
    assert "obb_center_is_scene_spatial_outlier" in reasons

    scene_sized = dict(metrics)
    scene_sized.update(
        {
            "gaussian_inside_scene_support_fraction": 1.0,
            "obb_center_inside_scene_support": True,
            "obb_center_distance_scene_diagonal": 0.0,
            "object_support_to_scene_diagonal": 0.9,
        }
    )
    passed, reasons = evaluate_scene_support_gate(scene_sized, policy=PostLiftPolicy())
    assert not passed
    assert "object_support_spans_implausible_scene_fraction" in reasons


def test_scene_support_gate_accepts_compact_support_in_scene_core() -> None:
    rng = np.random.default_rng(23)
    scene = rng.uniform(-5.0, 5.0, size=(8000, 3))
    envelope = scene_support_envelope(scene, quantile=0.01)
    points = rng.uniform([-0.4, -0.2, -0.1], [0.4, 0.2, 0.1], size=(900, 3))
    metrics = gaussian_scene_support_metrics(
        points,
        center_m=[0.0, 0.0, 0.0],
        robust_support_dimensions_m=[0.8, 0.4, 0.2],
        scene_envelope=envelope,
        margin_fraction=0.05,
    )
    passed, reasons = evaluate_scene_support_gate(metrics, policy=PostLiftPolicy())
    assert passed
    assert reasons == []


def _global_release_inputs() -> tuple[dict, dict, dict, np.ndarray]:
    labels = np.asarray([-1, 0, 0, 7, 7, 7], dtype=np.int32)
    result = {
        "status": "PASS",
        "quality_status": "PASS",
        "quality_gate": {"passed": True},
        "release_eligible": True,
        "inputs": {"source_gaussian_count": len(labels)},
        "counts": {
            "source_gaussians": len(labels),
            "verified_gaussians": 5,
            "unknown_gaussians": 1,
            "verified_objects": 2,
        },
        "qc_summary": {"verified_object_ids": [0, 7]},
        "frozen_full_colmap_fit_contract": {
            "schema": "farm.full-colmap-train-fit.v1",
            "fit_splits": ["train"],
            "heldout_consumed": False,
            "heldout_updates_candidate": False,
            "frozen_before_heldout_evaluation": True,
            "heldout_frames_opened_by_lift": False,
        },
        "contracts": {
            "dense_arrays_in_source_ply_order": True,
            "verified_only_canonical_labels": True,
            "unknown_instance_id": -1,
            "source_ply_mutated": False,
            "gaussians_deleted": 0,
            "global_physical_timestamp_holdout": True,
        },
    }
    build = {
        "status": "frozen_pending_heldout",
        "source_gaussian_count": len(labels),
        "global_heldout_masks_opened": False,
        "source_order_preserved": True,
        "forced_fill": False,
        "objects": [{"object_id": 0}, {"object_id": 7}, {"object_id": 9}],
    }
    heldout = {
        "status": "complete",
        "global_timestamp_disjoint": True,
        "failed_objects_return_to_unknown": True,
        "runner_up_promotion": False,
        "summary": {"verified_object_ids": [0, 7]},
        "objects": [
            {"object_id": 0, "status": "verified", "provisional_gaussians": 2},
            {"object_id": 7, "status": "verified", "provisional_gaussians": 3},
            {"object_id": 9, "status": "rejected", "provisional_gaussians": 10},
        ],
    }
    return result, build, heldout, labels


def test_global_lift_release_contract_accepts_exclusive_heldout_bound_ids() -> None:
    result, build, heldout, labels = _global_release_inputs()
    passed, reasons, metrics = evaluate_lift_release_contract(
        result,
        build,
        heldout,
        labels,
        source_gaussian_count=len(labels),
        artifacts_verified=True,
    )
    assert passed
    assert reasons == []
    assert metrics["dense_single_owner_encoding"] is True
    assert metrics["dense_verified_object_ids"] == [0, 7]
    assert metrics["heldout_blocked_object_ids"] == [9]
    assert metrics["quality_integrity_passed"] is True
    assert metrics["publication_provenance_passed"] is True


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda result, build, heldout, labels: result.update(
                release_eligible=False
            ),
            "gaussian_lift_not_release_eligible",
        ),
        (
            lambda result, build, heldout, labels: heldout.update(status="running"),
            "heldout_manifest_not_complete",
        ),
        (
            lambda result, build, heldout, labels: labels.__setitem__(1, 9),
            "dense_ids_disagree_with_heldout_verified_ids",
        ),
        (
            lambda result, build, heldout, labels: labels.__setitem__(0, -2),
            "dense_labels_below_unknown_id",
        ),
    ],
)
def test_global_lift_release_contract_fails_closed(
    mutation: object, reason: str
) -> None:
    result, build, heldout, labels = _global_release_inputs()
    mutation(result, build, heldout, labels)  # type: ignore[operator]
    passed, reasons, metrics = evaluate_lift_release_contract(
        result,
        build,
        heldout,
        labels,
        source_gaussian_count=len(labels),
        artifacts_verified=True,
    )
    assert not passed
    assert reason in reasons
    if reason == "gaussian_lift_not_release_eligible":
        assert metrics["quality_integrity_passed"] is True
        assert metrics["quality_integrity_reasons"] == []
        assert metrics["publication_provenance_passed"] is False
        assert metrics["publication_provenance_reasons"] == [
            "gaussian_lift_not_release_eligible"
        ]


def test_scene_release_allows_only_verified_subset_and_counts_blocked_rows() -> None:
    summary = summarize_scene_release(
        [
            {"object_id": 2, "route": "production_ready"},
            {"object_id": 7, "route": "obb_refit"},
            {"object_id": 9, "route": "semantic_verification"},
        ],
        upstream_passed=True,
    )
    assert summary == {
        "status": "PARTIAL",
        "release_eligible": True,
        "complete_scene_release_eligible": False,
        "published_object_ids": [2],
        "blocked_object_ids": [7, 9],
    }

    blocked = summarize_scene_release(
        [{"object_id": 2, "route": "production_ready"}],
        upstream_passed=False,
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["release_eligible"] is False


def test_experimental_routing_never_becomes_production_ready() -> None:
    common = {
        "diagnostic_upstream_passed": True,
        "publication_upstream_passed": False,
        "mask_passed": True,
        "obb_passed": True,
        "scene_support_passed": True,
        "relation_passed": True,
        "semantic_passed": True,
    }
    route = route_post_lift_object(**common)
    assert route == "experimental_nonrelease_quality_verified"
    summary = summarize_scene_release(
        [{"object_id": 7, "route": route}], upstream_passed=False
    )
    assert summary["status"] == "BLOCKED"
    assert summary["release_eligible"] is False
    assert summary["complete_scene_release_eligible"] is False
    assert summary["published_object_ids"] == []


def test_experimental_routing_preserves_actionable_per_object_queues() -> None:
    route = route_post_lift_object(
        diagnostic_upstream_passed=True,
        publication_upstream_passed=False,
        mask_passed=True,
        obb_passed=False,
        scene_support_passed=False,
        relation_passed=False,
        semantic_passed=False,
    )
    assert route == "obb_refit"
    semantic_route = route_post_lift_object(
        diagnostic_upstream_passed=True,
        publication_upstream_passed=False,
        mask_passed=True,
        obb_passed=True,
        scene_support_passed=True,
        relation_passed=True,
        semantic_passed=False,
    )
    assert semantic_route == "semantic_verification"
    with pytest.raises(ValueError, match="publication upstream"):
        route_post_lift_object(
            diagnostic_upstream_passed=False,
            publication_upstream_passed=True,
            mask_passed=True,
            obb_passed=True,
            scene_support_passed=True,
            relation_passed=True,
            semantic_passed=True,
        )


def test_semantic_gate_stays_fail_closed_without_whole_ready_consensus() -> None:
    candidate = {"category": "sign"}
    passed, reasons, published = _semantic_gate(
        candidate, {"status": "needs_independent_evidence", "category": "sign"}
    )
    assert passed is False
    assert reasons == ["whole_object_semantic_consensus_missing"]
    assert published == "unresolved object"

    passed, reasons, published = _semantic_gate(
        candidate, {"status": "whole_ready", "category": "unit"}
    )
    assert passed is False
    assert "presentation_and_audited_label_disagree" in reasons
    assert published == "unresolved object"

    passed, reasons, published = _semantic_gate(
        candidate, {"status": "whole_ready", "category": "sign"}
    )
    assert passed is True
    assert reasons == []
    assert published == "sign"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_frozen_heldout_contract_allows_verified_subset_and_is_sha_bound(
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.npy"
    ply_path = tmp_path / "candidate.ply"
    result_path = tmp_path / "result.json"
    candidate_path = tmp_path / "candidate.json"
    qc_path = tmp_path / "qc.json"
    labels = np.asarray([-1, 7, 0], dtype=np.int32)
    np.save(labels_path, labels)
    ply_path.write_bytes(b"immutable-ply")
    result_path.write_text("{}", encoding="utf-8")

    def spec(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }

    candidate = {
        "schema": "farm.frozen-heldout-candidate.v1",
        "status": "frozen",
        "meters_per_scene_unit": 1.0,
        "fit_contract": {
            "fit_splits": ["train"],
            "heldout_consumed": False,
            "heldout_updates_candidate": False,
            "frozen_before_heldout_evaluation": True,
        },
        "gaussian_contract": {
            "source_candidate_gaussian_order_aligned": True,
            "source_gaussian_count": 3,
            "candidate_gaussian_count": 3,
            "candidate_frozen": True,
            "labels_frozen": True,
            "heldout_used_for_fit": False,
        },
        "gaussian_artifacts": {
            "candidate_ply": spec(ply_path),
            "candidate_labels": spec(labels_path),
        },
        "upstream_lift_result": spec(result_path),
        "objects": [{"object_id": 7}, {"object_id": 0}],
    }
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    qc = {
        "schema": "farm.frozen-heldout-qc.v1",
        "status": "BLOCKED",
        "meters_per_scene_unit": 1.0,
        "contract": {
            "read_only": True,
            "heldout_used_for_state_fit": False,
            "heldout_updates_masks_obb_labels": False,
            "acceptance_aggregated_by_independent_physical_timestamp": True,
            "duplicate_views_at_one_timestamp_cannot_inflate_independence": True,
            "all_consumed_inputs_sha256_verified_and_rechecked": True,
        },
        "counts": {
            "candidate_objects": 2,
            "verified": 1,
            "collect_more_evidence": 0,
            "rejected": 1,
        },
        "routes": {
            "heldout_verified": [7],
            "collect_independent_heldout_evidence": [],
            "reject_candidate_train_only_refinement": [0],
        },
        "objects": [
            {"object_id": 7, "status": "verified", "route": "heldout_verified"},
            {
                "object_id": 0,
                "status": "rejected",
                "route": "reject_candidate_train_only_refinement",
            },
        ],
        "provenance": {
            "candidate_manifest": {
                "sha256": _sha(candidate_path),
                "schema": "farm.frozen-heldout-candidate.v1",
            }
        },
    }
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    passed, reasons, counts, _ = _frozen_heldout_contract(
        qc_path,
        candidate_path,
        qc,
        candidate,
        ids_path=labels_path,
        ply_path=ply_path,
        result_path=result_path,
        gaussian_ids=labels,
        table_count=3,
        meters_per_scene_unit=1.0,
    )
    assert passed, reasons
    assert counts == {0: 1, 7: 1}

    qc["provenance"]["candidate_manifest"]["sha256"] = "0" * 64
    passed, reasons, _, _ = _frozen_heldout_contract(
        qc_path,
        candidate_path,
        qc,
        candidate,
        ids_path=labels_path,
        ply_path=ply_path,
        result_path=result_path,
        gaussian_ids=labels,
        table_count=3,
        meters_per_scene_unit=1.0,
    )
    assert not passed
    assert "frozen_qc_candidate_sha256_mismatch" in reasons


def test_metric_scale_contract_is_explicit_and_checks_provenance() -> None:
    points_m = scene_points_to_meters(
        np.asarray([[2.0, -4.0, 8.0]], dtype=np.float64), 0.25
    )
    np.testing.assert_allclose(points_m, [[0.5, -1.0, 2.0]])
    declarations = verify_metric_scale_provenance(
        0.25,
        {
            "meters_per_scene_unit": 0.25,
            "metric_scale_check": {"meters_per_scene_unit": 0.25},
        },
    )
    assert [row["json_path"] for row in declarations] == [
        "$.meters_per_scene_unit",
        "$.metric_scale_check.meters_per_scene_unit",
    ]
    with pytest.raises(ValueError, match="disagrees"):
        verify_metric_scale_provenance(0.25, {"meters_per_scene_unit": 1.0})


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), "bad", True])
def test_metric_scale_contract_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        validate_meters_per_scene_unit(value)


def test_relation_router_flags_nested_mask_pass_obbs_without_mutation() -> None:
    objects = {
        1: {
            "category": "machine",
            "center_m": [0.0, 0.0, 0.0],
            "dimensions_m": [2.0, 2.0, 2.0],
            "wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        2: {
            "category": "machine panel",
            "center_m": [0.0, 0.0, 0.0],
            "dimensions_m": [1.0, 1.0, 1.0],
            "wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        3: {
            "category": "remote machine",
            "center_m": [10.0, 0.0, 0.0],
            "dimensions_m": [1.0, 1.0, 1.0],
            "wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }
    pairs = find_obb_relation_review_pairs(objects, [1, 2, 3], policy=PostLiftPolicy())
    assert [(row["first_object_id"], row["second_object_id"]) for row in pairs] == [
        (1, 2)
    ]
    assert "nested_obb_possible_duplicate_or_part" in pairs[0]["reasons"]
    assert pairs[0]["metrics"]["maximum_directional_containment"] == pytest.approx(1.0)
    assert objects[1]["category"] == "machine"
    assert objects[2]["category"] == "machine panel"
