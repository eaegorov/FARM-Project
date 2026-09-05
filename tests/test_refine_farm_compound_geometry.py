import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.geometry.refine_farm_compound_geometry import (  # noqa: E402
    build_multiview_consensus,
    coarse_voxel_support_metrics,
    find_binary_compound,
    gaussian_support_metrics,
)


def _solid_box(center, dimensions, spacing=0.05):
    center = np.asarray(center, dtype=np.float32)
    dimensions = np.asarray(dimensions, dtype=np.float32)
    axes = [
        np.arange(-0.5 * size, 0.5 * size + 0.25 * spacing, spacing, dtype=np.float32)
        for size in dimensions
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid + center


def _l_shape():
    horizontal = _solid_box([0.0, 0.85, 0.0], [2.0, 0.30, 0.25])
    vertical = _solid_box([-0.82, 0.0, 0.0], [0.35, 2.0, 0.25])
    return np.unique(np.round(np.concatenate((horizontal, vertical), axis=0), 5), axis=0)


def test_multiview_consensus_counts_distinct_views_not_duplicate_points():
    common = np.asarray([[0.01, 0.01, 0.01], [0.11, 0.01, 0.01]], dtype=np.float32)
    points_by_view = {
        1: np.concatenate((common, common, [[1.01, 1.01, 1.01]]), axis=0),
        2: common + np.asarray([0.005, 0.0, 0.0], dtype=np.float32),
        3: common + np.asarray([0.0, 0.005, 0.0], dtype=np.float32),
    }

    centres, counts = build_multiview_consensus(points_by_view, grid_m=0.1, min_views=2)

    assert centres.shape == (2, 3)
    assert counts.tolist() == [3, 3]


def test_gaussian_support_gate_distinguishes_aligned_and_shifted_geometry():
    aligned = gaussian_support_metrics(
        np.asarray([0.01, 0.02, 0.03, 0.04] * 10),
        grid_m=0.05,
        min_supported_rate=0.90,
    )
    shifted = gaussian_support_metrics(
        np.asarray([0.02] * 4 + [0.35] * 36),
        grid_m=0.05,
        min_supported_rate=0.90,
    )

    assert aligned["passed"] is True
    assert shifted["passed"] is False
    assert aligned["supported_rate"] > shifted["supported_rate"]


def test_coarse_voxel_support_gate_accounts_for_voxel_spacing():
    valid = coarse_voxel_support_metrics(np.full((100,), 0.09), voxel_spacing_m=0.16)
    invalid = coarse_voxel_support_metrics(
        np.concatenate((np.full((50,), 0.22), np.full((50,), 0.70))),
        voxel_spacing_m=0.16,
    )

    assert valid["passed"] is True
    assert invalid["passed"] is False


def test_l_shape_is_represented_by_two_compact_floor_aligned_boxes():
    points = _l_shape()
    result = find_binary_compound(
        points,
        gaussian_supported=np.ones((points.shape[0],), dtype=bool),
        max_volume_ratio=0.65,
        min_coverage=0.90,
        min_child_fraction=0.15,
        min_child_cells=24,
    )

    assert result["accepted"] is True
    assert len(result["children"]) == 2
    assert result["volume_ratio"] <= 0.65
    assert result["coverage"] >= 0.90
    for child in result["children"]:
        assert child["consensus_cells"] >= 24
        assert child["consensus_fraction"] >= 0.15
        assert child["gravity_tilt_degrees"] < 1.0e-4


def test_compact_cuboid_is_not_needlessly_split():
    points = _solid_box([0.0, 0.0, 0.0], [1.4, 0.8, 0.6])
    result = find_binary_compound(
        points,
        gaussian_supported=np.ones((points.shape[0],), dtype=bool),
        max_volume_ratio=0.65,
        min_coverage=0.90,
        min_child_fraction=0.15,
        min_child_cells=24,
    )

    assert result["accepted"] is False
    assert result["reason"] == "no_split_passed_gates"
