from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from scripts.geometry.refine_farm_compound_geometry import _fit_floor_obb  # noqa: E402
from scripts.geometry.refine_farm_compound_presentation import (  # noqa: E402
    _obb_record,
    _rotation_from_wxyz,
    _projection_metrics,
    evaluate_guards,
    fit_supported_compound,
)


def _record(obb: dict) -> dict:
    return _obb_record(obb, cells=100, supported_rate=1.0)


def test_supported_fit_improves_supported_coverage_without_semantics() -> None:
    rng = np.random.default_rng(20260805)
    first = np.column_stack(
        (
            rng.uniform(-2.8, -1.0, 240),
            rng.uniform(-0.15, 0.15, 240),
            rng.uniform(-0.18, 0.18, 240),
        )
    ).astype(np.float32)
    second = np.column_stack(
        (
            rng.uniform(1.0, 1.35, 240),
            rng.uniform(-0.15, 1.55, 240),
            rng.uniform(-0.20, 0.20, 240),
        )
    ).astype(np.float32)
    unsupported = np.asarray(
        [
            [-3.7, 0.0, 0.9],
            [-3.6, 0.0, -0.9],
            [2.1, 2.4, 0.8],
            [2.0, -0.8, -0.8],
        ]
        * 4,
        dtype=np.float32,
    )
    points = np.concatenate((first, second, unsupported), axis=0)
    supported = np.ones((points.shape[0],), dtype=bool)
    supported[-unsupported.shape[0] :] = False
    split = points[:, 0] <= 0.0
    current_boxes = [
        _record(_fit_floor_obb(points[split], 0.025)),
        _record(_fit_floor_obb(points[~split], 0.025)),
    ]
    parent_box = {
        "center_m": [0.0, 0.0, 0.0],
        "dimensions_lwh_m": [8.0, 4.0, 4.0],
        "wxyz": [1.0, 0.0, 0.0, 0.0],
    }

    result = fit_supported_compound(
        points,
        supported,
        current_boxes,
        parent_box=parent_box,
        split_axis=0,
        split_threshold_m=0.0,
        quantile=0.01,
    )

    assert len(result["candidate_records"]) == 2
    assert result["candidate_supported_coverage"] >= result["current_supported_coverage"]
    assert result["candidate_inside_support_rate"] >= result["current_inside_support_rate"]
    assert all(np.isfinite(row["center_m"]).all() for row in result["candidate_records"])
    assert all(float(row["gravity_tilt_degrees"]) == 0.0 for row in result["candidate_records"])


def test_supported_fit_preserves_exact_oblique_world_up() -> None:
    rng = np.random.default_rng(14)
    up = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    up /= np.linalg.norm(up)
    first = rng.normal(size=(160, 3)).astype(np.float32) * [0.8, 0.2, 0.15]
    second = rng.normal(size=(160, 3)).astype(np.float32) * [0.3, 0.7, 0.15]
    second[:, 0] += 2.5
    points = np.concatenate((first, second), axis=0)
    supported = np.ones(points.shape[0], dtype=bool)
    current = [
        _record(_fit_floor_obb(first, 0.025, up_vector=up)),
        _record(_fit_floor_obb(second, 0.025, up_vector=up)),
    ]
    result = fit_supported_compound(
        points, supported, current,
        parent_box={
            "center_m": [1.25, 0.0, 0.0],
            "dimensions_lwh_m": [5.0, 5.0, 5.0],
            "wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        split_axis=0, split_threshold_m=0.0, quantile=0.025,
        up_vector=up,
    )
    for record in result["candidate_records"]:
        rotation = _rotation_from_wxyz(record["wxyz"])
        assert float(np.dot(rotation[:, 2], up)) >= 0.999


def test_preview_guards_accept_improvement_and_reject_volume_regression() -> None:
    result = {
        "current_coverage": 0.94,
        "candidate_coverage": 0.95,
        "current_supported_coverage": 0.95,
        "candidate_supported_coverage": 0.985,
        "candidate_inside_support_rate": 0.99,
        "volume_inflation": 0.035,
        "rotation_delta_degrees": [0.14, 0.0],
    }
    current_projection = {
        "median_union_box_iou": 0.28,
        "p25_union_box_iou": 0.20,
        "center_inside_rate": 0.93,
    }
    candidate_projection = {
        "median_union_box_iou": 0.29,
        "p25_union_box_iou": 0.205,
        "center_inside_rate": 0.93,
    }
    guards = evaluate_guards(
        result,
        current_projection,
        candidate_projection,
        min_supported_coverage=0.98,
        min_inside_support_rate=0.98,
        max_volume_inflation=0.05,
        max_rotation_delta_degrees=2.0,
        min_projection_ratio=0.95,
    )
    assert all(guards.values())

    regressed = dict(result, volume_inflation=0.20)
    guards = evaluate_guards(
        regressed,
        current_projection,
        candidate_projection,
        min_supported_coverage=0.98,
        min_inside_support_rate=0.98,
        max_volume_inflation=0.05,
        max_rotation_delta_degrees=2.0,
        min_projection_ratio=0.95,
    )
    assert guards["volume_inflation_limit"] is False


def test_projection_metrics_use_compound_union_and_centres() -> None:
    identity = np.eye(3, dtype=np.float64)
    boxes = [
        {
            "center": np.asarray([-0.5, 0.0, 5.0]),
            "dimensions": np.asarray([0.8, 0.8, 0.8]),
            "rotation_matrix": identity,
        },
        {
            "center": np.asarray([0.5, 0.0, 5.0]),
            "dimensions": np.asarray([0.8, 0.8, 0.8]),
            "rotation_matrix": identity,
        },
    ]
    observation = {
        "pose": np.eye(4, dtype=np.float64),
        "K": np.asarray([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]),
        "raw_bbox": np.asarray([80.0, 88.0, 120.0, 112.0]),
    }

    metrics = _projection_metrics(boxes, [observation])

    assert metrics["valid_box_views"] == 1
    assert metrics["median_union_box_iou"] > 0.0
    assert metrics["center_inside_rate"] == 1.0
