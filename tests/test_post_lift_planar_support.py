from __future__ import annotations

import numpy as np
import pytest

from farm_runtime.post_lift_planar_support import (
    PlanarSupportPolicy,
    apply_pruning_to_labels,
    evaluate_planar_support_candidate,
)


def _evidence(normal: list[float], *, axis: int = 1, thickness: float = 0.12) -> dict:
    return {
        "schema": "farm.planar-rgbd-thickness-evidence.v1",
        "accepted": True,
        "policy": "planar_rgbd_normal_axis_exception_v1",
        "reasons": [],
        "independent_physical_timestamps": 5,
        "planar_view_rate": 1.0,
        "normal_angle_q90_degrees": 1.0,
        "normal_axis": axis,
        "consensus_normal": normal,
        "constrained_thickness_m": thickness,
    }


def _grid_points(*, slope: float = 0.0, outliers: bool = True) -> np.ndarray:
    x, z = np.meshgrid(np.linspace(-0.95, 0.95, 45), np.linspace(-0.45, 0.45, 25))
    y = slope * x + 0.008 * np.sin(5.0 * z)
    core = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    if not outliers:
        return core
    # Full in-plane coverage prevents an outlier layer from masquerading as a
    # missing side of the physical object.
    extra = core[::5].copy()
    extra[:, 1] += 0.28
    return np.concatenate((core, extra), axis=0)


def test_consensus_normal_slab_prunes_depth_without_losing_in_plane_support() -> None:
    points = _grid_points()
    keep, result = evaluate_planar_support_candidate(
        points,
        center_m=[0.0, 0.0, 0.0],
        dimensions_m=[2.0, 0.12, 1.0],
        rotation_matrix=np.eye(3),
        planar_evidence=_evidence([0.0, 1.0, 0.0]),
    )
    assert result["status"] == "candidate"
    assert result["recommended_route"] == "mask_geometry_refinement"
    assert result["heldout_consumed"] is False
    assert result["obb_mutated"] is False
    assert 0 < result["pruned_gaussians"] < len(points)
    assert result["in_plane_cell_coverage"] == pytest.approx(1.0)
    assert min(result["in_plane_extent_retention"]) > 0.98
    assert np.all(np.abs(points[keep, 1]) <= 0.060001)


def test_rgbd_consensus_normal_handles_bounded_obb_axis_mismatch() -> None:
    slope = 0.13
    points = _grid_points(slope=slope)
    normal = np.asarray([-slope, 1.0, 0.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    keep, result = evaluate_planar_support_candidate(
        points,
        center_m=[0.0, 0.0, 0.0],
        dimensions_m=[2.0, 0.12, 1.0],
        rotation_matrix=np.eye(3),
        planar_evidence=_evidence(normal.tolist()),
    )
    assert result["status"] == "candidate"
    assert 7.0 < result["consensus_to_obb_axis_degrees"] < 8.0
    assert min(result["in_plane_extent_retention"]) > 0.95
    assert result["followup_route"] == "obb_refit_after_mask_qc"
    assert keep.mean() > 0.75


def test_wrong_consensus_slab_fails_closed_when_it_crops_in_plane_extent() -> None:
    points = _grid_points(slope=0.30, outliers=False)
    policy = PlanarSupportPolicy(
        maximum_consensus_to_obb_axis_degrees=20.0,
        minimum_in_plane_extent_retention=0.90,
    )
    keep, result = evaluate_planar_support_candidate(
        points,
        center_m=[0.0, 0.0, 0.0],
        dimensions_m=[2.0, 0.12, 1.0],
        rotation_matrix=np.eye(3),
        # An incorrect train consensus may not remove one spatial half merely
        # to make the normal-axis number look better.
        planar_evidence=_evidence([0.0, 1.0, 0.0]),
        policy=policy,
    )
    assert result["status"] == "rejected"
    assert "in_plane_extent_not_preserved" in result["reasons"]
    assert result["recommended_route"] == "obb_refit"
    assert keep.all()


def test_non_planar_evidence_is_not_applicable() -> None:
    evidence = _evidence([0.0, 1.0, 0.0])
    evidence["accepted"] = False
    keep, result = evaluate_planar_support_candidate(
        _grid_points(),
        center_m=[0.0, 0.0, 0.0],
        dimensions_m=[2.0, 0.12, 1.0],
        rotation_matrix=np.eye(3),
        planar_evidence=evidence,
    )
    assert result["status"] == "not_applicable"
    assert "planar_evidence_not_accepted" in result["reasons"]
    assert keep.all()


def test_label_application_can_only_demote_to_unknown() -> None:
    labels = np.asarray([7, 7, -1, 3, 7], dtype=np.int32)
    result = apply_pruning_to_labels(
        labels, object_id=7, object_keep_mask=np.asarray([True, False, True])
    )
    np.testing.assert_array_equal(result, [7, -1, -1, 3, 7])
    np.testing.assert_array_equal(labels, [7, 7, -1, 3, 7])
    with pytest.raises(ValueError, match="does not match"):
        apply_pruning_to_labels(labels, object_id=7, object_keep_mask=[True])
