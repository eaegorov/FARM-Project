from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from scripts.geometry.apply_farm_full_colmap_rescue import _copy_row  # noqa: E402
from scripts.geometry.refine_farm_object_geometry import (  # noqa: E402
    _fit_robust_obb,
    _planar_surface_evidence,
    _select_voxel_supported_obb,
    _yaw_align_planar_normal,
)

sys.path.remove(str(SCRIPTS))


def _observation(points: np.ndarray, timestamp: str) -> dict:
    k = np.asarray([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])
    uv = (k @ points.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return {
        "points": points.astype(np.float32),
        "centroid": np.median(points, axis=0),
        "physical_timestamp": timestamp,
        "pose": np.eye(4),
        "K": k,
        "raw_bbox": np.r_[uv.min(0) - 2.0, uv.max(0) + 2.0],
    }


def _plane(seed: int, angle: float = 0.0, z: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, 25), np.linspace(-0.55, 0.55, 17), indexing="ij"
    )
    local = np.c_[x.ravel(), y.ravel(), rng.normal(0.0, 0.006, x.size)]
    radians = np.radians(angle)
    rotation = np.asarray(
        [
            [np.cos(radians), 0.0, np.sin(radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(radians), 0.0, np.cos(radians)],
        ]
    )
    return local @ rotation.T + [0.0, 0.0, z]


def _thresholds() -> dict:
    return {
        "min_independent_views": 4,
        "min_planar_view_rate": 0.80,
        "max_plane_eigenvalue_ratio": 0.04,
        "min_in_plane_eigenvalue_ratio": 0.08,
        "max_normal_angle_q90_degrees": 10.0,
        "max_median_view_thickness_m": 0.12,
        "max_q90_view_thickness_m": 0.18,
        "max_median_view_thickness_ratio": 0.18,
        "max_plane_offset_spread_m": 0.20,
        "max_normal_axis_angle_degrees": 15.0,
        "min_projected_box_iou": 0.50,
        "min_box_support_rate": 0.80,
        "max_constrained_thickness_m": 0.18,
    }


def _evaluate(observations: list[dict]) -> tuple[dict, dict, dict]:
    points = np.concatenate([row["points"] for row in observations])
    base = _fit_robust_obb(points, 0.05, orientation_mode="pca_3d")
    constrained, evidence = _planar_surface_evidence(
        base, observations, support_iou=0.20, **_thresholds()
    )
    return base, constrained, evidence


def test_independent_planar_views_bound_the_normal_axis() -> None:
    observations = [
        _observation(_plane(i, z=4.0 + offset), f"t-{i}")
        for i, offset in enumerate((0.0, 0.015, -0.01, 0.02, -0.005))
    ]
    base, constrained, evidence = _evaluate(observations)
    axis = int(evidence["normal_axis"])
    assert evidence["accepted"] is True
    assert evidence["policy"] == "planar_rgbd_normal_axis_exception_v1"
    assert evidence["independent_physical_timestamps"] == 5
    assert constrained["dimensions"][axis] <= 0.18
    assert constrained["dimensions"][axis] <= base["dimensions"][axis]
    assert evidence["constrained_box_support_rate"] >= 0.80


def test_volumetric_views_do_not_activate_planar_policy() -> None:
    rng = np.random.default_rng(11)
    observations = [
        _observation(
            rng.uniform([-1.0, -0.55, 3.65], [1.0, 0.55, 4.35], (900, 3)),
            f"t-{i}",
        )
        for i in range(5)
    ]
    base, constrained, evidence = _evaluate(observations)
    assert evidence["accepted"] is False
    assert "insufficient_locally_planar_view_rate" in evidence["reasons"]
    assert evidence["policy"] == "full_3d_voxel_retention"
    np.testing.assert_array_equal(constrained["dimensions"], base["dimensions"])


def test_unstable_plane_normals_fail_closed() -> None:
    observations = [
        _observation(_plane(i, angle), f"t-{i}")
        for i, angle in enumerate((0.0, 24.0, -27.0, 42.0, -38.0))
    ]
    _, _, evidence = _evaluate(observations)
    assert evidence["accepted"] is False
    assert "unstable_plane_normals" in evidence["reasons"]


def _yaw_rotation(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    return np.asarray(
        [
            [np.cos(radians), 0.0, np.sin(radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(radians), 0.0, np.cos(radians)],
        ],
        dtype=np.float64,
    )


def test_planar_normal_alignment_changes_only_gravity_yaw() -> None:
    base = _manual_obb([2.2, 1.2, 0.20])
    base.update(
        {
            "rotation_matrix": _yaw_rotation(8.0),
            "orientation_mode": "gravity_yaw",
            "up_vector": np.asarray([0.0, 1.0, 0.0]),
        }
    )
    candidate, audit = _yaw_align_planar_normal(
        base,
        np.asarray([0.0, 0.0, 1.0]),
        normal_axis=2,
        maximum_gravity_elevation_degrees=10.0,
    )
    rotation = np.asarray(candidate["rotation_matrix"])
    assert audit["status"] == "candidate"
    assert audit["applied"] is True
    assert abs(audit["yaw_correction_degrees"]) == pytest.approx(8.0)
    np.testing.assert_allclose(rotation[:, 1], [0.0, 1.0, 0.0], atol=1e-6)
    assert abs(float(np.dot(rotation[:, 2], [0.0, 0.0, 1.0]))) > 0.999999
    np.testing.assert_array_equal(candidate["center"], base["center"])
    np.testing.assert_array_equal(candidate["dimensions"], base["dimensions"])


def test_planar_normal_alignment_rejects_gravity_incompatible_normal() -> None:
    base = _manual_obb([2.2, 1.2, 0.20])
    base.update(
        {
            "rotation_matrix": _yaw_rotation(8.0),
            "orientation_mode": "gravity_yaw",
            "up_vector": np.asarray([0.0, 1.0, 0.0]),
        }
    )
    candidate, audit = _yaw_align_planar_normal(
        base,
        np.asarray([0.0, 0.5, np.sqrt(0.75)]),
        normal_axis=2,
        maximum_gravity_elevation_degrees=10.0,
    )
    assert candidate is base
    assert audit["status"] == "rejected"
    assert audit["reason"] == "consensus_normal_not_gravity_compatible"


def test_material_planar_yaw_error_uses_consensus_before_thickness_fit() -> None:
    observations = [_observation(_plane(i), f"t-{i}") for i in range(5)]
    base = _manual_obb([2.2, 1.2, 0.30])
    rotation = _yaw_rotation(8.0)
    base.update(
        {
            "rotation_matrix": rotation,
            "orientation_mode": "gravity_yaw",
            "up_vector": np.asarray([0.0, 1.0, 0.0]),
            "corners": (
                (np.asarray(base["corners"]) - np.asarray(base["center"]))
                @ rotation.T
                + np.asarray(base["center"])
            ),
        }
    )
    constrained, evidence = _planar_surface_evidence(
        base,
        observations,
        support_iou=0.20,
        **_thresholds(),
    )
    normal_axis = int(evidence["normal_axis"])
    alignment = evidence["normal_alignment"]
    assert evidence["accepted"] is True
    assert alignment["status"] == "candidate"
    assert alignment["applied"] is True
    assert alignment["median_reprojection_iou_retention"] >= 0.85
    corrected = np.asarray(constrained["rotation_matrix"])
    assert abs(float(np.dot(corrected[:, normal_axis], [0.0, 0.0, 1.0]))) > 0.999999
    assert constrained["dimensions"][normal_axis] <= 0.18


def test_duplicate_cameras_at_too_few_timestamps_fail_closed() -> None:
    observations = [
        _observation(_plane(i), "same" if i < 4 else "second") for i in range(8)
    ]
    _, _, evidence = _evaluate(observations)
    assert evidence["independent_physical_timestamps"] == 2
    assert "insufficient_independent_physical_timestamps" in evidence["reasons"]


def _manual_obb(dimensions: list[float]) -> dict:
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float32)
    dims = np.asarray(dimensions, dtype=np.float32)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return {
        "center": center,
        "dimensions": dims,
        "rotation_matrix": np.eye(3, dtype=np.float32),
        "corners": center + signs * (0.5 * dims),
    }


def test_normal_leakage_cannot_inflate_thickness_but_in_plane_can_expand() -> None:
    base = _manual_obb([1.0, 1.0, 0.10])
    observations = [_observation(_plane(i), f"t-{i}") for i in range(5)]
    x, y, z = np.meshgrid(
        np.arange(-0.8, 0.801, 0.05),
        np.arange(-0.4, 0.401, 0.05),
        np.arange(3.6, 4.401, 0.05),
        indexing="ij",
    )
    selected, _, diagnostics = _select_voxel_supported_obb(
        base,
        np.c_[x.ravel(), y.ravel(), z.ravel()],
        observations,
        voxel_size_m=0.05,
        support_iou=0.20,
        min_support_rate=0.0,
        min_projected_iou=0.0,
        expansion_penalty=0.0,
        score_tie_tolerance=0.0,
        planar_evidence={
            "accepted": True,
            "policy": "planar_rgbd_normal_axis_exception_v1",
            "normal_axis": 2,
        },
    )
    assert diagnostics["planar_normal_axis_exception_applied"] is True
    assert selected["dimensions"][2] == pytest.approx(0.10)
    assert selected["dimensions"][0] > 1.20
    assert diagnostics["voxel_in_plane_inside_rate"] > diagnostics["voxel_inside_rate"]


def test_rejected_planar_evidence_preserves_volumetric_selection() -> None:
    base = _manual_obb([1.0, 1.0, 0.10])
    observations = [_observation(_plane(i), f"t-{i}") for i in range(5)]
    voxels = np.random.default_rng(3).uniform(
        [-0.7, -0.4, 3.7], [0.7, 0.4, 4.3], (800, 3)
    )
    kwargs = dict(
        voxel_size_m=0.05,
        support_iou=0.20,
        min_support_rate=0.0,
        min_projected_iou=0.0,
        expansion_penalty=0.15,
        score_tie_tolerance=0.005,
    )
    legacy = _select_voxel_supported_obb(base, voxels, observations, **kwargs)
    rejected = _select_voxel_supported_obb(
        base,
        voxels,
        observations,
        planar_evidence={"accepted": False, "policy": "full_3d_voxel_retention"},
        **kwargs,
    )
    np.testing.assert_array_equal(legacy[0]["dimensions"], rejected[0]["dimensions"])
    assert legacy[2]["candidate"] == rejected[2]["candidate"]


@pytest.mark.parametrize(
    ("key", "value", "default"),
    (
        ("object_geometry_voxel_retention_gate_rate", torch.tensor([0.91, 0.82]), 0.0),
        (
            "object_geometry_voxel_retention_policy",
            ["planar", "full"],
            "full_3d_voxel_retention",
        ),
        (
            "object_geometry_planar_exception_applied",
            torch.tensor([True, False]),
            False,
        ),
        ("object_geometry_planar_normal_axis", torch.tensor([2, -1]), -1),
        ("object_geometry_planar_independent_views", torch.tensor([5, 0]), 0),
        ("object_geometry_planar_evidence_reasons", [[], ["noisy"]], []),
    ),
)
def test_old_state_initializes_only_new_diagnostics(
    key: str, value: object, default: object
) -> None:
    destination: dict = {}
    _copy_row(destination, {key: value}, key, 0, 2)
    assert destination[key][0] == value[0]
    assert destination[key][1] == default


def test_missing_core_geometry_remains_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot copy row-aligned rescue field"):
        _copy_row(
            {},
            {"object_box_dimensions_m": torch.ones((2, 3))},
            "object_box_dimensions_m",
            0,
            2,
        )
