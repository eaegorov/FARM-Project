import numpy as np
import pytest
import torch

from scripts.recover_farm_rejected_geometry import (
    GeometryGates,
    adaptive_erode_mask,
    assert_indexed_rows_unchanged,
    assess_candidate,
    depth_mad_keep,
    largest_voxel_component_indices,
    rejected_geometry_indices,
    robust_view_consensus,
    snapshot_indexed_rows,
)


def _passing_candidate(**overrides):
    candidate = {
        "status": "refined",
        "valid_observations": 5,
        "view_retention_rate": 0.80,
        "observation_consistency": 0.80,
        "center_inside_detection_rate": 0.90,
        "median_reprojection_error_normalized": 0.20,
        "median_projected_box_iou": 0.50,
        "box_support_rate": 0.80,
        "voxel_inside_obb_rate": 0.70,
        "orientation_required": True,
        "orientation_confidence": 0.60,
        "box_volume_m3": 1.10,
    }
    candidate.update(overrides)
    return candidate


def _rejected_baseline(**overrides):
    baseline = {
        "observation_consistency": 0.75,
        "center_inside_detection_rate": 0.85,
        "median_reprojection_error_normalized": 0.25,
        "median_projected_box_iou": 0.34,
        "box_support_rate": 0.75,
        "voxel_inside_obb_rate": 0.65,
        "orientation_confidence": 0.55,
        "box_volume_m3": 1.0,
    }
    baseline.update(overrides)
    return baseline


def test_adaptive_erosion_preserves_thin_mask_and_erodes_solid_mask() -> None:
    thin = np.zeros((9, 9), dtype=bool)
    thin[:, 4] = True
    thin_result, thin_radius, _ = adaptive_erode_mask(thin, min_pixels=3)

    solid = np.ones((9, 9), dtype=bool)
    solid_result, solid_radius, _ = adaptive_erode_mask(solid, min_pixels=3)

    assert thin_radius == 0
    assert np.array_equal(thin_result, thin)
    assert solid_radius == 1
    assert solid_result.sum() == 49


def test_depth_mad_filter_rejects_a_separate_background_mode() -> None:
    depths = np.asarray([2.00, 2.01, 2.02, 2.03, 4.0, np.nan], dtype=np.float32)
    keep, diagnostics = depth_mad_keep(depths, k_mad=1.0, min_mad_m=0.03)

    assert keep.tolist() == [True, True, True, True, False, False]
    assert diagnostics["median_depth_m"] == pytest.approx(2.02, abs=1.0e-5)
    assert diagnostics["used_mad_m"] == pytest.approx(0.03)


def test_largest_metric_component_is_selected_by_point_count() -> None:
    large = np.asarray([[0.01 * index, 0.0, 0.0] for index in range(20)], dtype=np.float32)
    small = np.asarray([[3.0 + 0.01 * index, 0.0, 0.0] for index in range(7)], dtype=np.float32)
    points = np.concatenate((small, large), axis=0)

    indices, diagnostics = largest_voxel_component_indices(points, voxel_size_m=0.04)

    assert indices.size == large.shape[0]
    assert np.all(points[indices, 0] < 1.0)
    assert diagnostics["component_fraction"] == pytest.approx(20.0 / 27.0)


def test_view_consensus_rejects_distant_centroid_outliers_without_chaining() -> None:
    centroids = np.asarray(
        [[0.00, 0.0, 0.0], [0.03, 0.0, 0.0], [-0.02, 0.01, 0.0], [1.2, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=np.float32,
    )
    diagonals = np.ones((5,), dtype=np.float32) * 0.4

    members, threshold = robust_view_consensus(centroids, diagonals)

    assert members.tolist() == [0, 1, 2]
    assert threshold == pytest.approx(0.10)


def test_candidate_requires_hard_gates_pareto_and_a_strict_improvement() -> None:
    result = assess_candidate(_rejected_baseline(), _passing_candidate(), GeometryGates())

    assert result["eligible"] is True
    assert result["hard_gate_failures"] == []
    assert result["pareto_regressions"] == []
    assert "median_projected_box_iou" in result["strict_improvements"]


@pytest.mark.parametrize(
    ("candidate", "failure_kind", "failure_value"),
    [
        (_passing_candidate(view_retention_rate=0.60), "hard_gate_failures", "view_retention"),
        (_passing_candidate(box_volume_m3=1.40), "hard_gate_failures", "volume_expansion"),
        (_passing_candidate(box_support_rate=0.71), "pareto_regressions", "box_support_rate"),
    ],
)
def test_candidate_fails_closed_on_gate_volume_or_pareto_regression(candidate, failure_kind, failure_value) -> None:
    result = assess_candidate(_rejected_baseline(), candidate, GeometryGates())

    assert result["eligible"] is False
    assert failure_value in result[failure_kind]


def test_only_direct_geometry_rejected_rows_are_selected() -> None:
    statuses = [
        "geometry_pass",
        "geometry_rejected",
        "assembly_geometry_rejected",
        "not_evaluated",
        "geometry_rejected",
    ]

    assert rejected_geometry_indices(statuses, len(statuses)) == [1, 4]


def test_frozen_row_invariant_covers_tensor_array_and_list_fields() -> None:
    state = {
        "tensor": torch.tensor([[1.0, 2.0], [3.0, float("nan")]]),
        "array": np.asarray([[10, 11], [20, 21]], dtype=np.int32),
        "labels": ["pass", "reject"],
        "not_object_indexed": [1, 2, 3],
    }
    snapshot = snapshot_indexed_rows(state, [0], count=2)

    state["tensor"][1, 0] = 99.0
    assert_indexed_rows_unchanged(state, [0], count=2, expected=snapshot)

    state["array"][0, 0] = -1
    with pytest.raises(AssertionError, match="array"):
        assert_indexed_rows_unchanged(state, [0], count=2, expected=snapshot)


def test_consensus_points_nested_tensors_use_stable_value_fingerprints() -> None:
    state = {
        "object_geometry_consensus_points": [
            torch.empty((0, 3), dtype=torch.float32),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        ],
        "object_geometry_status": ["geometry_pass", "assembly_geometry_pass"],
    }
    snapshot = snapshot_indexed_rows(state, [0, 1], count=2)

    # Repeated hashing, and even equivalent storage replacement, must not
    # create the old pickle/storage-ID false positive.
    assert_indexed_rows_unchanged(state, [0, 1], count=2, expected=snapshot)
    state["object_geometry_consensus_points"][1] = state["object_geometry_consensus_points"][1].clone()
    assert_indexed_rows_unchanged(state, [0, 1], count=2, expected=snapshot)

    state["object_geometry_consensus_points"][1][0, 0] += 0.25
