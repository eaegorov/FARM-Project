from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.geometry.plan_farm_full_colmap_rescue import build_support_track_candidates

from farm_runtime.full_colmap_view_planner import (
    SparseTrackIndex,
    allocate_split_global_view_budget,
    bounded_support_points,
    build_sparse_track_index,
    choose_train_heldout_views,
    match_support_to_sparse_tracks,
    model_file_manifest,
    physical_timestamp_fold,
    project_metric_support,
    validate_disjoint_timestamp_folds,
)


class _Pose:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def matrix(self) -> np.ndarray:
        return self._matrix


def _image(image_id: int, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        image_id=image_id,
        camera_id=3,
        name=name,
        cam_from_world=_Pose(
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        ),
    )


def test_support_is_finite_robust_and_bounded() -> None:
    core = np.column_stack(
        (
            np.linspace(-0.2, 0.2, 20),
            np.zeros((20,)),
            np.full((20,), 4.0),
        )
    )
    points = np.vstack((core, [[100.0, 100.0, 100.0], [np.nan, 0.0, 0.0]]))
    support, diagnostics = bounded_support_points(
        points, trim_quantile=0.95, max_points=8
    )
    assert support.shape == (8, 3)
    assert np.isfinite(support).all()
    assert float(np.max(np.linalg.norm(support, axis=1))) < 10.0
    assert diagnostics["input_points"] == 22
    assert diagnostics["finite_points"] == 21
    assert diagnostics["bounded_points"] == 8


def test_sparse_track_index_uses_metric_scale_and_deduplicates_images() -> None:
    points = {
        9: SimpleNamespace(
            xyz=np.asarray([1.0, 2.0, 3.0]),
            track=SimpleNamespace(
                elements=[
                    SimpleNamespace(image_id=7),
                    SimpleNamespace(image_id=7),
                    SimpleNamespace(image_id=8),
                ]
            ),
        ),
        4: SimpleNamespace(
            xyz=np.asarray([0.0, 0.0, 1.0]),
            track=SimpleNamespace(elements=[SimpleNamespace(image_id=3)]),
        ),
    }
    index = build_sparse_track_index(
        SimpleNamespace(points3D=points), meters_per_scene_unit=0.5
    )
    assert index.point_ids.tolist() == [4, 9]
    assert index.xyz_m.tolist() == [[0.0, 0.0, 0.5], [0.5, 1.0, 1.5]]
    assert index.image_ids_by_point == ((3,), (7, 8))


def test_nearest_sparse_tracks_count_unique_colmap_points() -> None:
    sparse = SparseTrackIndex(
        point_ids=np.asarray([11, 12, 13], dtype=np.int64),
        xyz_m=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [9.0, 0.0, 4.0]]),
        image_ids_by_point=((1, 2), (1,), (9,)),
    )
    # The first sparse point is hit twice but contributes once per image.
    support = np.asarray(
        [[0.01, 0.0, 4.0], [0.02, 0.0, 4.0], [0.99, 0.0, 4.0], [20.0, 0.0, 4.0]]
    )
    counts, diagnostics = match_support_to_sparse_tracks(
        support, sparse, maximum_distance_m=0.05
    )
    assert counts == {1: 2, 2: 1}
    assert diagnostics["matched_sparse_points"] == 2
    assert diagnostics["matched_support_fraction"] == pytest.approx(0.75)


def test_support_projection_needs_no_obb_and_checks_frame_containment() -> None:
    points = np.asarray(
        [
            [-0.25, -0.20, 4.0],
            [-0.25, 0.20, 4.0],
            [0.25, -0.20, 4.0],
            [0.25, 0.20, 4.0],
        ]
    )
    projected = project_metric_support(
        points,
        _image(1, "unused").cam_from_world.matrix(),
        np.asarray([[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]]),
        width=800,
        height=600,
        meters_per_scene_unit=1.0,
        min_margin_ratio=0.02,
        min_area_ratio=0.001,
        max_area_ratio=0.5,
        min_in_frame_ratio=1.0,
        bbox_quantile=0.0,
    )
    assert projected is not None
    assert projected["bbox_xyxy"] == pytest.approx([350.0, 260.0, 450.0, 340.0])
    assert projected["projected_support_points"] == 4


def test_physical_timestamp_split_is_global_deterministic_and_disjoint() -> None:
    candidates = [
        {
            "name": f"cam00_{timestamp:06d}_center.jpg",
            "physical_timestamp": f"{timestamp:06d}",
            "view_direction": [float(timestamp % 3), 1.0, 1.0],
            "score": float(100 - timestamp),
            "area_ratio": 0.1,
            "margin_ratio": 0.1,
        }
        for timestamp in range(1, 80)
    ]
    train, heldout = choose_train_heldout_views(
        candidates,
        train_limit=8,
        heldout_limit=4,
        min_view_angle_degrees=0.0,
        heldout_fraction=0.25,
        split_seed="test-scene",
    )
    assert len(train) == 8
    assert len(heldout) == 4
    validate_disjoint_timestamp_folds(train, heldout)
    for row in train + heldout:
        expected = physical_timestamp_fold(
            row["physical_timestamp"],
            heldout_fraction=0.25,
            split_seed="test-scene",
        )
        assert row["split"] == expected


def test_split_global_budget_reserves_heldout_without_starving_objects() -> None:
    train_candidates = []
    heldout_candidates = []
    for object_id in range(4):
        train_candidates.append(
            [{"name": f"train-{object_id}-{rank}"} for rank in range(4)]
        )
        heldout_candidates.append(
            [{"name": f"heldout-{object_id}-{rank}"} for rank in range(2)]
        )
    selected, train, heldout = allocate_split_global_view_budget(
        train_candidates,
        heldout_candidates,
        max_total_views=12,
        max_heldout_views=4,
    )
    assert len(selected) == 12
    assert [len(values) for values in train] == [2, 2, 2, 2]
    assert [len(values) for values in heldout] == [1, 1, 1, 1]


def test_timestamp_overlap_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="timestamp leakage"):
        validate_disjoint_timestamp_folds(
            [{"physical_timestamp": "42"}], [{"physical_timestamp": "42"}]
        )


def test_colmap_model_manifest_hashes_only_contract_files(tmp_path) -> None:
    (tmp_path / "cameras.bin").write_bytes(b"cameras")
    (tmp_path / "images.bin").write_bytes(b"images")
    (tmp_path / "points3D.bin").write_bytes(b"points")
    (tmp_path / "unrelated.tmp").write_bytes(b"ignore")
    rows = model_file_manifest(tmp_path)
    assert [row["name"] for row in rows] == [
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    ]
    assert all(len(str(row["sha256"])) == 64 for row in rows)


def test_object_track_match_coverage_fails_before_camera_ranking() -> None:
    sparse = SparseTrackIndex(
        point_ids=np.asarray([11], dtype=np.int64),
        xyz_m=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        image_ids_by_point=((1,),),
    )
    support = np.asarray(
        [
            [0.01, 0.0, 4.0],
            [1.0, 0.0, 4.0],
            [2.0, 0.0, 4.0],
            [3.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    candidates, diagnostics = build_support_track_candidates(
        SimpleNamespace(),
        sparse,
        support,
        identity_pattern=re.compile(
            r"^(?P<camera>cam\\d+)_(?P<timestamp>\\d+)_(?P<family>.+)\\.png$"
        ),
        require_identity_match=True,
        existing_names=set(),
        existing_timestamps=set(),
        meters_per_scene_unit=1.0,
        maximum_track_distance_m=0.05,
        minimum_object_track_match_fraction=0.30,
        minimum_track_support_points=1,
        minimum_rescue_track_support_points=1,
        minimum_track_support_fraction=0.01,
        minimum_projected_support_points=1,
        minimum_rescue_projected_support_points=1,
        minimum_rescue_in_frame_support_ratio=0.0,
        min_margin_ratio=0.0,
        min_area_ratio=0.0,
        max_area_ratio=1.0,
        min_in_frame_support_ratio=0.0,
        support_bbox_quantile=0.0,
    )
    assert candidates == []
    assert diagnostics["matched_support_fraction"] == pytest.approx(0.25)
    assert diagnostics["object_track_match_gate"] is False
