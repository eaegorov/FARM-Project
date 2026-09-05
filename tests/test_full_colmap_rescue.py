from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scene_graph.captioning.evidence import crop_evidence_manifest

from scripts.geometry.plan_farm_full_colmap_rescue import (
    allocate_global_view_budget,
    angular_separation_degrees,
    choose_diverse_views,
    identity_key,
    load_requested_object_ids,
    obb_corners,
    object_rescue_priority,
    project_obb,
    stratified_object_queue,
)
from farm_runtime.obb_release_evidence import (
    evaluate_obb_release_evidence,
)
from farm_runtime.full_colmap_view_planner import (
    adaptive_view_limit,
    choose_next_best_views,
    plan_temporal_episode_topup,
    temporal_episode_frame_count,
    track_evidence_tier,
)
from farm_runtime.full_colmap_rescue import (
    distributed_points,
    pack_mask,
    mask_shape_metrics,
    project_world_points_seed,
    multiview_voxel_consensus,
    voxel_downsample_points,
    unpack_mask_archive,
)
from scripts.geometry.apply_farm_full_colmap_rescue import (
    _category_compatible,
    _copy_row,
    _independent_identity_re_adjudication,
    _initial_vlm_geometry_gate,
    _vlm_gate,
    main as apply_full_colmap_rescue,
)
from scripts.geometry.refine_farm_object_geometry import _select_observation_paths
from scripts.geometry.refine_farm_full_colmap_masks import (
    _depth_refine,
    _object_acceptance,
    _view_acceptance,
    _multiview_world_support,
    add_projected_voxel_fallbacks,
    bounded_area_expansion_gate,
    negative_prompt_regions,
    partition_tracking_episodes,
    select_mask_candidate,
    tracking_physical_timestamp,
    tracking_stream_id,
    validate_metric_contract,
)


class _Pose:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def matrix(self) -> np.ndarray:
        return self._matrix


def _image(world_to_camera: np.ndarray | None = None) -> SimpleNamespace:
    matrix = (
        np.asarray(world_to_camera, dtype=np.float64)
        if world_to_camera is not None
        else np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
    )
    return SimpleNamespace(cam_from_world=_Pose(matrix), image_id=7, camera_id=3)


def _camera() -> SimpleNamespace:
    model = SimpleNamespace(name="PINHOLE")
    return SimpleNamespace(
        model=model,
        model_name=None,
        params=np.asarray([800.0, 800.0, 400.0, 300.0]),
        width=800,
        height=600,
    )


def test_exact_object_allowlist_is_strict_and_deduplicated(tmp_path) -> None:
    allowlist = tmp_path / "ids.txt"
    allowlist.write_text("# canonical presentation\n217\n55\n217\n", encoding="utf-8")
    assert load_requested_object_ids(allowlist) == {55, 217}

    allowlist.write_text("not-an-id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid object ID"):
        load_requested_object_ids(allowlist)

    allowlist.write_text("# no IDs\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist is empty"):
        load_requested_object_ids(allowlist)


def test_semantic_uncertainty_does_not_trigger_geometry_rescue() -> None:
    state = {
        "object_geometry_status": ["geometry_pass"],
        "object_geometry_projected_box_iou": np.asarray([0.9]),
        "object_geometry_box_support_rate": np.asarray([0.9]),
        "object_semantic_status": ["unresolved_component"],
    }

    priority, reasons = object_rescue_priority(state, 0, observations=5)

    assert priority == 0.0
    assert reasons == []


def test_core_guided_ab_selection_can_shrink_contaminated_yoloe_mask() -> None:
    raw = np.zeros((20, 20), dtype=bool)
    raw[4:16, 3:17] = True
    refined = np.zeros_like(raw)
    refined[6:14, 6:14] = True
    core = np.zeros_like(raw)
    core[8:12, 8:12] = True
    competing = np.zeros_like(raw)
    competing[4:16, 3:7] = True
    selected, audit = select_mask_candidate(
        raw, refined, core, np.asarray([5.0, 5.0, 15.0, 15.0]), competing
    )
    assert audit["selected"] == "sam3_refined"
    assert np.array_equal(selected, refined)
    assert (
        audit["sam3_refined"]["competing_mask_fraction"]
        < audit["raw"]["competing_mask_fraction"]
    )


def test_core_guided_ab_selection_preserves_yoloe_when_sam_loses_core() -> None:
    raw = np.zeros((20, 20), dtype=bool)
    raw[5:15, 5:15] = True
    refined = np.zeros_like(raw)
    refined[1:4, 1:4] = True
    core = np.zeros_like(raw)
    core[8:12, 8:12] = True
    selected, audit = select_mask_candidate(
        raw, refined, core, np.asarray([4.0, 4.0, 16.0, 16.0]), np.zeros_like(raw)
    )
    assert audit["selected"] == "yoloe_seed_preserved"
    assert np.array_equal(selected, raw)


def test_whole_object_expansion_keeps_competing_instances_negative() -> None:
    shape = (120, 160)
    box = np.zeros(shape, dtype=bool)
    box[10:110, 20:140] = True
    seed = np.zeros(shape, dtype=bool)
    seed[54:66, 74:86] = True
    competing = np.zeros(shape, dtype=bool)
    competing[12:108, 22:138] = True
    hard_negative, ring = negative_prompt_regions(
        box,
        seed,
        competing,
        whole_object_expansion=True,
    )
    assert np.logical_and(hard_negative, box).any()
    assert not np.logical_and(ring, box).any()
    assert ring.any()
    assert not np.logical_and(hard_negative, ring).any()

    shrink_hard, shrink_ring = negative_prompt_regions(
        box, seed, competing, whole_object_expansion=False
    )
    assert np.logical_and(shrink_ring, box).any()
    assert np.logical_and(shrink_hard, box).any()


def test_depth_refinement_does_not_use_large_obb_to_admit_background() -> None:
    seed = np.zeros((24, 24), dtype=bool)
    seed[8:14, 5:10] = True
    sam = seed.copy()
    sam[8:14, 10:17] = True
    depth = np.full(seed.shape, 3.0, dtype=np.float32)
    depth[seed] = 1.0
    frame = {
        "T_world_cam": np.eye(4, dtype=np.float64).tolist(),
        "K": [[100.0, 0.0, 12.0], [0.0, 100.0, 12.0], [0.0, 0.0, 1.0]],
    }
    target = {
        "target_center_m": [0.0, 0.0, 1.0],
        "target_dimensions_m": [100.0, 100.0, 100.0],
        "target_wxyz": [1.0, 0.0, 0.0, 0.0],
    }

    final, audit = _depth_refine(sam, seed, depth, frame, target, preserve_seed=True)

    assert np.array_equal(final, seed)
    assert audit["obb_inside_fraction"] == pytest.approx(1.0)
    assert audit["depth_inlier_fraction"] < audit["depth_or_obb_inlier_fraction"]
    assert audit["depth_padding_m"] == pytest.approx(0.04)


def test_object_acceptance_counts_independent_physical_timestamps() -> None:
    rows = [
        {"accepted": True, "global_image_id": 1, "physical_timestamp_ns": 10},
        {"accepted": True, "global_image_id": 2, "physical_timestamp_ns": 10},
        {
            "accepted": True,
            "global_image_id": 3,
            "physical_timestamp_ns": 20,
            "refinement_origin": "rest3d_bidirectional_propagation",
        },
    ]
    accepted, _, audit = _object_acceptance(
        rows,
        {"support_points": 100},
        minimum_views=2,
        minimum_independent_views=2,
    )
    assert not accepted
    assert audit["accepted_physical_timestamps"] == 2
    assert audit["independent_physical_timestamps"] == 1

    rows[1]["physical_timestamp_ns"] = 30
    accepted, _, audit = _object_acceptance(
        rows,
        {"support_points": 100},
        minimum_views=2,
        minimum_independent_views=2,
    )
    assert accepted
    assert audit["independent_physical_timestamps"] == 2


def _tracking_entry(
    source: str,
    camera: str,
    render_timestamp: int,
    *,
    x_m: float = 0.0,
    yaw_degrees: float = 0.0,
) -> dict:
    angle = np.radians(yaw_degrees)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose[0, 3] = x_m
    return {
        "source_image": source,
        "frame": {
            "source_image": source,
            "camera": camera,
            "timestamp_ns": render_timestamp,
            "T_world_cam": pose.tolist(),
        },
    }


def _partition(
    rows: list[dict], *, maximum_episode_frames: int = 32
) -> list[list[dict]]:
    return partition_tracking_episodes(
        rows,
        maximum_gap_multiplier=2.5,
        maximum_translation_m=0.50,
        maximum_rotation_degrees=21.0,
        maximum_episode_frames=maximum_episode_frames,
    )


def test_tracking_episodes_never_mix_camera_or_view_family() -> None:
    rows = [
        _tracking_entry("cam00_000001_center.png", "cam00_center", 400),
        _tracking_entry("cam01_000001_center.png", "cam01_center", 100),
        _tracking_entry("cam00_000002_center.png", "cam00_center", 300),
        _tracking_entry("cam00_000002_yaw_left.png", "cam00_yaw_left", 200),
        _tracking_entry("cam00_000100_center.png", "cam00_center", 1),
    ]
    episodes = _partition(rows)

    assert len(episodes) == 4
    for episode in episodes:
        assert len({row["tracking_stream_id"] for row in episode}) == 1
        timestamps = [tracking_physical_timestamp(row["frame"]) for row in episode]
        assert all(
            left is not None and right is not None and right > left
            for left, right in zip(timestamps, timestamps[1:])
        )
    assert sorted(len(episode) for episode in episodes) == [1, 1, 1, 2]


def test_tracking_uses_physical_source_time_not_rgbd_render_order() -> None:
    first = _tracking_entry("cam00_000010_center.png", "cam00_center", 900_000_000)
    second = _tracking_entry("cam00_000020_center.png", "cam00_center", 1)
    assert tracking_physical_timestamp(first["frame"]) == 10
    assert tracking_physical_timestamp(second["frame"]) == 20
    episodes = _partition([second, first])
    assert [row["source_image"] for row in episodes[0]] == [
        "cam00_000010_center.png",
        "cam00_000020_center.png",
    ]


def test_tracking_episodes_split_on_motion_and_bound_session_length() -> None:
    translated = [
        _tracking_entry("cam00_000010_center.png", "cam00_center", 1, x_m=0.0),
        _tracking_entry("cam00_000020_center.png", "cam00_center", 2, x_m=0.1),
        _tracking_entry("cam00_000030_center.png", "cam00_center", 3, x_m=0.7),
    ]
    assert [len(episode) for episode in _partition(translated)] == [2, 1]

    rotated = [
        _tracking_entry("cam00_000010_center.png", "cam00_center", 1),
        _tracking_entry("cam00_000020_center.png", "cam00_center", 2, yaw_degrees=22.0),
    ]
    assert [len(episode) for episode in _partition(rotated)] == [1, 1]

    bounded = [
        _tracking_entry(f"cam00_{timestamp:06d}_center.png", "cam00_center", timestamp)
        for timestamp in range(10, 70, 10)
    ]
    assert [
        len(episode) for episode in _partition(bounded, maximum_episode_frames=3)
    ] == [3, 3]


def test_tracking_stream_fallback_preserves_view_family() -> None:
    assert (
        tracking_stream_id({"source_image": "cam01_001234_yaw_left.png"})
        == "cam01_yaw_left"
    )
    assert tracking_stream_id({"source_image": "unknown.png"}).startswith("isolated:")


def test_project_obb_requires_full_frame_margin() -> None:
    corners = obb_corners([0.0, 0.0, 4.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    accepted = project_obb(
        corners,
        _image(),
        _camera(),
        meters_per_scene_unit=1.0,
        min_margin_ratio=0.03,
        min_area_ratio=0.001,
        max_area_ratio=0.5,
    )
    assert accepted is not None
    assert accepted["area_ratio"] == pytest.approx(0.1088435374)
    shifted = obb_corners([-2.0, 0.0, 4.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    assert (
        project_obb(
            shifted,
            _image(),
            _camera(),
            meters_per_scene_unit=1.0,
            min_margin_ratio=0.08,
            min_area_ratio=0.001,
            max_area_ratio=0.5,
        )
        is None
    )


def test_diverse_selection_deduplicates_timestamp_and_view_direction() -> None:
    candidates = [
        {
            "name": "best",
            "score": 4.0,
            "area_ratio": 0.1,
            "margin_ratio": 0.1,
            "physical_timestamp": "1",
            "view_direction": [0.0, 0.0, 1.0],
        },
        {
            "name": "same-timestamp",
            "score": 3.9,
            "area_ratio": 0.1,
            "margin_ratio": 0.1,
            "physical_timestamp": "1",
            "view_direction": [1.0, 0.0, 0.0],
        },
        {
            "name": "same-view",
            "score": 3.8,
            "area_ratio": 0.1,
            "margin_ratio": 0.1,
            "physical_timestamp": "2",
            "view_direction": [0.01, 0.0, 1.0],
        },
        {
            "name": "diverse",
            "score": 3.0,
            "area_ratio": 0.1,
            "margin_ratio": 0.1,
            "physical_timestamp": "3",
            "view_direction": [1.0, 0.0, 0.0],
        },
    ]
    selected = choose_diverse_views(candidates, limit=2, min_view_angle_degrees=12.0)
    assert [row["name"] for row in selected] == ["best", "diverse"]


def test_global_view_budget_is_allocated_round_robin_without_starvation() -> None:
    rows = [
        {
            "diverse_candidates": [
                {"name": f"object-{object_index}-view-{view_index}"}
                for view_index in range(5)
            ]
        }
        for object_index in range(4)
    ]
    selected, admitted = allocate_global_view_budget(rows, max_total_views=8)
    assert len(selected) == 8
    assert [len(values) for values in admitted] == [2, 2, 2, 2]


def test_identity_key_groups_virtual_views_by_physical_timestamp() -> None:
    import re

    pattern = re.compile(
        r"(?P<camera>cam[0-9]+)_(?P<timestamp>[0-9]+)_(?P<view>center|yaw_left)"
    )
    assert identity_key("cam01_001234_yaw_left.png", pattern) == (
        "001234",
        "cam01",
        "yaw_left",
    )


def test_angular_separation_is_metric_and_stable() -> None:
    assert angular_separation_degrees([0, 0, 1], [1, 0, 0]) == pytest.approx(90.0)
    assert angular_separation_degrees([0, 0, 0], [1, 0, 0]) == 0.0


def _nbv_candidate(
    name: str,
    timestamp: str,
    direction: list[float],
    center: list[float],
    *,
    score: float,
) -> dict[str, object]:
    return {
        "name": name,
        "physical_timestamp": timestamp,
        "view_direction": direction,
        "camera_center_m": center,
        "distance_m": 4.0,
        "score": score,
        "area_ratio": 0.04,
        "margin_ratio": 0.15,
        "in_frame_support_ratio": 0.98,
        "track_support_fraction": 0.06,
    }


def _choose_nbv(candidates: list[dict[str, object]], maximum: int = 3):
    return choose_next_best_views(
        candidates,
        minimum_views=2,
        maximum_views=maximum,
        min_view_angle_degrees=12.0,
        min_camera_baseline_m=0.15,
        min_baseline_distance_ratio=0.04,
        relaxed_view_angle_degrees=6.0,
        relaxed_camera_baseline_m=0.08,
        relaxed_baseline_distance_ratio=0.02,
    )


def test_next_best_view_uses_metric_baseline_when_angle_is_small() -> None:
    candidates = [
        _nbv_candidate("a", "100", [0, 0, 1], [0, 0, 0], score=1.0),
        _nbv_candidate("b", "200", [0.05, 0, 1], [0.30, 0, 0], score=0.9),
    ]

    selected, audit = _choose_nbv(candidates, maximum=2)

    assert [row["name"] for row in selected] == ["a", "b"]
    assert selected[1]["minimum_angle_to_selected_degrees"] < 12.0
    assert selected[1]["minimum_camera_baseline_to_selected_m"] == pytest.approx(0.30)
    assert selected[1]["selection_diversity_tier"] == "strict"
    assert audit["diversity_relaxation_used"] is False


def test_next_best_view_relaxation_only_fills_minimum_not_redundancy() -> None:
    candidates = [
        _nbv_candidate("a", "100", [0, 0, 1], [0, 0, 0], score=1.0),
        _nbv_candidate("b", "200", [0.14, 0, 1], [0.09, 0, 0], score=0.9),
        _nbv_candidate("c", "300", [-0.14, 0, 1], [-0.09, 0, 0], score=0.8),
    ]

    selected, audit = _choose_nbv(candidates)

    assert len(selected) == 2
    assert selected[1]["selection_diversity_tier"] == "minimum_only_relaxed"
    assert audit["minimum_evidence_met"] is True
    assert audit["relaxed_minimum_only_views"] == 1
    assert audit["budget"]["target_views"] == 3


def test_next_best_view_rejects_correlated_minimum_below_relaxed_floor() -> None:
    candidates = [
        _nbv_candidate("a", "100", [0, 0, 1], [0, 0, 0], score=1.0),
        _nbv_candidate("b", "200", [0.02, 0, 1], [0.03, 0, 0], score=0.9),
    ]

    selected, audit = _choose_nbv(candidates, maximum=2)

    assert [row["name"] for row in selected] == ["a"]
    assert audit["minimum_evidence_met"] is False


def test_next_best_view_deduplicates_virtual_views_by_physical_timestamp() -> None:
    candidates = [
        _nbv_candidate("a-center", "100", [0, 0, 1], [0, 0, 0], score=1.0),
        _nbv_candidate("a-left", "100", [1, 0, 0], [1, 0, 0], score=0.5),
        _nbv_candidate("b", "200", [1, 0, 0], [1, 0, 0], score=0.9),
    ]

    selected, audit = _choose_nbv(candidates, maximum=2)

    assert [row["name"] for row in selected] == ["a-center", "b"]
    assert audit["unique_physical_timestamps"] == 2
    assert audit["physical_timestamp_deduplicated_rows"] == 1


def test_adaptive_view_budget_is_bounded_to_two_redundancy_views() -> None:
    candidates = [
        {
            **_nbv_candidate(
                str(index),
                str(index),
                [float(index), 0, 1],
                [float(index), 0, 0],
                score=1.0,
            ),
            "in_frame_support_ratio": 0.1,
            "track_support_fraction": 0.0,
            "area_ratio": 0.001,
            "margin_ratio": 0.0,
        }
        for index in range(20)
    ]

    target, audit = adaptive_view_limit(
        candidates, minimum_views=2, maximum_views=12, priority=10.0
    )

    assert target == 4
    assert audit["redundancy_views"] == 2
    assert audit["bounded_extra_views"] == 2


def test_sparse_track_rescue_tier_is_bounded_and_explicit() -> None:
    common = {
        "minimum_track_support_points": 2,
        "minimum_track_support_fraction": 0.01,
        "minimum_rescue_track_support_points": 1,
    }

    assert (
        track_evidence_tier(track_count=3, track_fraction=0.03, **common)
        == "strict"
    )
    assert (
        track_evidence_tier(track_count=1, track_fraction=0.001, **common)
        == "sparse_track_rescue"
    )
    assert (
        track_evidence_tier(track_count=0, track_fraction=0.0, **common)
        is None
    )


def test_mask_archive_roundtrip_preserves_exact_pixels(tmp_path) -> None:
    mask = np.zeros((31, 47), dtype=bool)
    mask[4:15, 8:21] = True
    mask[18:27, 29:42] = True
    bits, shape, bbox = pack_mask(mask)
    path = tmp_path / "mask.npz"
    np.savez_compressed(
        path,
        image_shape=np.asarray(mask.shape, dtype=np.int32),
        raw_bits=bits,
        raw_shape=shape,
        raw_bbox_xyxy=bbox,
    )
    assert np.array_equal(unpack_mask_archive(path), mask)


def test_distributed_points_are_inside_and_spatially_distinct() -> None:
    mask = np.zeros((80, 120), dtype=bool)
    mask[10:70, 15:105] = True
    points = distributed_points(mask, 8)
    assert len(points) == 8
    assert len({(int(x), int(y)) for x, y in points}) == 8
    assert all(mask[int(y), int(x)] for x, y in points)


def test_mask_shape_metrics_rejects_fragmented_projected_seed_islands() -> None:
    mask = np.zeros((120, 160), dtype=bool)
    mask[25:80, 30:90] = True
    mask[15:25, 120:130] = True
    metrics = mask_shape_metrics(mask)
    assert metrics["component_count"] == 2
    assert metrics["significant_component_count"] == 2
    assert metrics["dominant_component_fraction"] < 0.98
    clean = np.zeros_like(mask)
    clean[25:80, 30:90] = True
    metrics = mask_shape_metrics(clean)
    assert metrics["component_count"] == 1
    assert metrics["dominant_component_fraction"] == 1.0


def test_projected_metric_voxel_seed_requires_rendered_depth_consistency() -> None:
    points = np.asarray(
        [
            [-0.10, -0.10, 2.0],
            [0.0, -0.10, 2.0],
            [0.10, -0.10, 2.0],
            [-0.10, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.10, 0.0, 2.0],
            [-0.10, 0.10, 2.0],
            [0.0, 0.10, 2.0],
            [0.10, 0.10, 2.0],
        ],
        dtype=np.float32,
    )
    frame = {
        "T_world_cam": np.eye(4, dtype=np.float64).tolist(),
        "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
    }
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    seed, audit = project_world_points_seed(
        points, frame, depth, dilation_pixels=2, minimum_projected_points=8
    )
    assert audit["projected_points"] == 9
    assert audit["depth_consistent_points"] == 9
    assert audit["seed_pixels"] == int(seed.sum()) > 9
    wrong_depth = np.full((64, 64), 5.0, dtype=np.float32)
    rejected, audit = project_world_points_seed(
        points, frame, wrong_depth, dilation_pixels=2, minimum_projected_points=8
    )
    assert not rejected.any()
    assert audit["depth_consistent_points"] == 0


def test_projected_voxel_fallback_covers_objects_without_transient_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "object_id": np.asarray([7, 9], dtype=np.int64),
        "object_box_centers_m": np.asarray([[0, 0, 2], [1, 0, 2]], dtype=np.float32),
        "object_box_dimensions_m": np.ones((2, 3), dtype=np.float32),
        "object_box_wxyz": np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
    }
    rescue = {
        "images": [
            {"source_ref": "prepared/a.png"},
            {"source_ref": "prepared/b.png"},
        ]
    }
    frames = {
        "a": {"rgb_path": "prepared/a.png", "source_image": "a.jpg"},
        "b": {"rgb_path": "prepared/b.png", "source_image": "b.jpg"},
    }
    plan = {
        "objects": [
            {"object_id": 7, "selected_views": [{"name": "a.jpg"}, {"name": "b.jpg"}]},
            {"object_id": 9, "selected_views": [{"name": "a.jpg"}, {"name": "b.jpg"}]},
        ]
    }
    monkeypatch.setattr(
        "scripts.geometry.refine_farm_full_colmap_masks._object_voxel_points",
        lambda state, index: np.zeros((30, 3), dtype=np.float32),
    )
    existing = [{"object_id": 7}]
    output = add_projected_voxel_fallbacks(
        plan, source, rescue, frames, existing, minimum_views=2
    )
    assert [int(row["object_id"]) for row in output] == [7, 9]
    fallback = output[1]
    assert fallback["association_mode"] == "projected_metric_voxel_fallback"
    assert [row["source_image"] for row in fallback["matched_views"]] == [
        "a.jpg",
        "b.jpg",
    ]


def _projected_voxel_completion_inputs(
    monkeypatch: pytest.MonkeyPatch,
    selected_views: list[dict],
) -> tuple[dict, dict, dict, dict, list[dict]]:
    source = {
        "object_id": np.asarray([7], dtype=np.int64),
        "object_box_centers_m": np.asarray([[0, 0, 2]], dtype=np.float32),
        "object_box_dimensions_m": np.ones((1, 3), dtype=np.float32),
        "object_box_wxyz": np.asarray([[1, 0, 0, 0]], dtype=np.float32),
    }
    rescue = {
        "images": [
            {"source_ref": "prepared/a.png"},
            {"source_ref": "prepared/b.png"},
            {"source_ref": "prepared/c.png"},
            {"source_ref": "prepared/heldout.png"},
        ]
    }
    frames = {
        stem: {
            "rgb_path": f"prepared/{stem}.png",
            "source_image": f"{stem}.jpg",
        }
        for stem in ("a", "b", "c", "heldout")
    }
    plan = {"objects": [{"object_id": 7, "selected_views": selected_views}]}
    monkeypatch.setattr(
        "scripts.geometry.refine_farm_full_colmap_masks._object_voxel_points",
        lambda state, index: np.zeros((30, 3), dtype=np.float32),
    )
    detector_view = {
        "rescue_image_id": 0,
        "source_image": "a.jpg",
        "physical_timestamp": "100",
        "seed_origin": "rescue_mapping_track",
        "mask": "detector-mask.npz",
    }
    associations = [
        {
            "object_id": 7,
            "association_mode": "rescue_mapping_track",
            "matched_views": [detector_view],
        }
    ]
    return plan, source, rescue, frames, associations


def test_projected_voxel_completion_augments_existing_association(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        {"name": "a.jpg", "physical_timestamp": "100", "split": "train"},
        {"name": "b.jpg", "physical_timestamp": "200", "split": "train"},
    ]
    plan, source, rescue, frames, associations = _projected_voxel_completion_inputs(
        monkeypatch, selected
    )

    output = add_projected_voxel_fallbacks(
        plan, source, rescue, frames, associations, minimum_views=2
    )

    assert [view["source_image"] for view in output[0]["matched_views"]] == [
        "a.jpg",
        "b.jpg",
    ]
    assert output[0]["association_mode"] == (
        "rescue_mapping_track_plus_projected_metric_voxel_completion"
    )
    assert output[0]["projected_completion_views"] == 1


def test_projected_voxel_completion_preserves_original_detector_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        {"name": "a.jpg", "physical_timestamp": "100", "split": "train"},
        {"name": "b.jpg", "physical_timestamp": "200", "split": "train"},
    ]
    plan, source, rescue, frames, associations = _projected_voxel_completion_inputs(
        monkeypatch, selected
    )
    original_view = copy.deepcopy(associations[0]["matched_views"][0])

    output = add_projected_voxel_fallbacks(
        plan, source, rescue, frames, associations, minimum_views=2
    )

    assert output[0]["matched_views"][0] == original_view
    assert output[0]["matched_views"][0]["seed_origin"] == "rescue_mapping_track"
    assert output[0]["matched_views"][1]["seed_origin"] == "projected_metric_voxels"


def test_projected_voxel_completion_never_adds_heldout_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        {"name": "a.jpg", "physical_timestamp": "100", "split": "train"},
        {"name": "heldout.jpg", "physical_timestamp": "200", "split": "heldout"},
        {"name": "c.jpg", "physical_timestamp": "300", "split": "train"},
    ]
    plan, source, rescue, frames, associations = _projected_voxel_completion_inputs(
        monkeypatch, selected
    )

    output = add_projected_voxel_fallbacks(
        plan, source, rescue, frames, associations, minimum_views=2
    )

    assert [view["source_image"] for view in output[0]["matched_views"]] == [
        "a.jpg",
        "c.jpg",
    ]


def test_projected_voxel_completion_never_adds_duplicate_physical_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        {"name": "b.jpg", "physical_timestamp": "100", "split": "train"},
        {"name": "c.jpg", "physical_timestamp": "200", "split": "train"},
    ]
    plan, source, rescue, frames, associations = _projected_voxel_completion_inputs(
        monkeypatch, selected
    )

    output = add_projected_voxel_fallbacks(
        plan, source, rescue, frames, associations, minimum_views=2
    )

    assert [view["source_image"] for view in output[0]["matched_views"]] == [
        "a.jpg",
        "c.jpg",
    ]


def test_projected_voxel_completion_does_not_mutate_input_associations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        {"name": "a.jpg", "physical_timestamp": "100", "split": "train"},
        {"name": "b.jpg", "physical_timestamp": "200", "split": "train"},
    ]
    plan, source, rescue, frames, associations = _projected_voxel_completion_inputs(
        monkeypatch, selected
    )
    original = copy.deepcopy(associations)

    output = add_projected_voxel_fallbacks(
        plan, source, rescue, frames, associations, minimum_views=2
    )

    assert associations == original
    assert output is not associations
    assert output[0] is not associations[0]
    assert output[0]["matched_views"] is not associations[0]["matched_views"]
    assert output[0]["matched_views"][0] is not associations[0]["matched_views"][0]


def test_voxel_downsample_is_finite_deterministic_and_bounded() -> None:
    points = np.asarray(
        [
            [0.001, 0.001, 0.001],
            [0.009, 0.009, 0.009],
            [0.021, 0.001, 0.001],
            [np.nan, 0.0, 0.0],
            [0.041, 0.001, 0.001],
        ],
        dtype=np.float32,
    )
    first = voxel_downsample_points(points, voxel_size_m=0.02, max_points=2)
    second = voxel_downsample_points(points, voxel_size_m=0.02, max_points=2)
    assert first.shape == (2, 3)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert first[0].tolist() == pytest.approx([0.001, 0.001, 0.001])


def test_multiview_world_support_requires_two_independent_metric_views(
    tmp_path,
) -> None:
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    np.save(tmp_path / "d0.npy", depth)
    np.save(tmp_path / "d1.npy", depth)
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True
    bits, shape, bbox = pack_mask(mask)

    mask_paths = []
    for index in range(2):
        path = tmp_path / f"m{index}.npz"
        np.savez_compressed(
            path,
            image_shape=np.asarray(mask.shape, dtype=np.int32),
            raw_bits=bits,
            raw_shape=shape,
            raw_bbox_xyxy=bbox,
        )
        mask_paths.append(path)
    frame_base = {
        "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        "T_world_cam": np.eye(4, dtype=np.float64).tolist(),
    }
    frames = {
        "rgb0": {
            **frame_base,
            "rgb_path": "rgb0.jpg",
            "depth_path": "d0.npy",
            "source_image": "a.jpg",
        },
        "rgb1": {
            **frame_base,
            "rgb_path": "rgb1.jpg",
            "depth_path": "d1.npy",
            "source_image": "b.jpg",
        },
    }
    rescue_state = {"images": [{"source_ref": "rgb0.jpg"}, {"source_ref": "rgb1.jpg"}]}
    association = {
        "target_center_m": [0.0, 0.0, 2.0],
        "target_dimensions_m": [1.0, 1.0, 1.0],
        "target_wxyz": [1.0, 0.0, 0.0, 0.0],
        "matched_views": [
            {"rescue_image_id": index, "mask": str(mask_paths[index])}
            for index in range(2)
        ],
    }
    support, audit = _multiview_world_support(
        association,
        rescue_state,
        frames,
        tmp_path,
        tmp_path / "frames.json",
        np.zeros((0, 3), dtype=np.float32),
    )
    assert support.shape[0] > 0
    assert audit["contributing_views"] == 2
    association["matched_views"] = association["matched_views"][:1]
    rejected, audit = _multiview_world_support(
        association,
        rescue_state,
        frames,
        tmp_path,
        tmp_path / "frames.json",
        np.zeros((0, 3), dtype=np.float32),
    )
    assert rejected.shape == (0, 3)
    assert audit["contributing_views"] == 1


def test_multiview_voxel_consensus_requires_distinct_views() -> None:
    first = np.asarray(
        [[0.001, 0.001, 1.001], [0.002, 0.002, 1.002], [0.20, 0.0, 1.0]],
        dtype=np.float32,
    )
    second = np.asarray([[0.006, 0.004, 1.004], [0.40, 0.0, 1.0]], dtype=np.float32)
    support, audit = multiview_voxel_consensus(
        [first, second], voxel_size_m=0.02, minimum_views=2
    )
    assert support.shape == (1, 3)
    assert support[0].tolist() == pytest.approx([0.0035, 0.0025, 1.0025])
    assert audit["input_views"] == 2
    assert audit["consensus_voxels_before_limit"] == 1
    rejected, audit = multiview_voxel_consensus(
        [first], voxel_size_m=0.02, minimum_views=2
    )
    assert rejected.shape == (0, 3)

    assert audit["consensus_voxels_before_limit"] == 0


def test_object_acceptance_requires_views_and_metric_support() -> None:
    views = [
        {"accepted": True, "global_image_id": 10},
        {"accepted": True, "global_image_id": 11},
    ]
    accepted, status, gates = _object_acceptance(
        views, {"support_points": 30}, minimum_views=2
    )
    assert accepted and status == "accepted"
    assert gates["refined_view_gate"] and gates["metric_support_gate"]
    accepted, status, gates = _object_acceptance(
        views, {"support_points": 0}, minimum_views=2
    )
    assert not accepted
    assert status == "rejected_insufficient_metric_support"
    assert not gates["metric_support_gate"]

    mixed = [
        {"accepted": True, "global_image_id": 10},
        {
            "accepted": True,
            "global_image_id": 11,
            "refinement_origin": "rest3d_bidirectional_propagation",
        },
    ]
    accepted, status, gates = _object_acceptance(
        mixed, {"support_points": 30}, minimum_views=2
    )
    assert accepted and status == "accepted"
    assert gates["independent_refined_views"] == 1
    assert gates["propagated_refined_views"] == 1

    propagated_only = [
        {**row, "refinement_origin": "rest3d_bidirectional_propagation"}
        for row in views
    ]
    accepted, status, gates = _object_acceptance(
        propagated_only, {"support_points": 30}, minimum_views=2
    )
    assert not accepted
    assert status == "rejected_without_independent_seed"
    assert not gates["independent_seed_gate"]


def _accepted_vlm_row() -> dict:
    category = "industrial cabinet"
    contract = {
        "schema": "farm.open-vocabulary-object-label.v1",
        "contract_valid": True,
        "reason_codes": [],
        "category": category,
        "category_role": "whole_object",
        "form_hypernym": "cabinet",
        "primitive_form": "rectangular",
        "head_noun_is_primitive": False,
        "specificity": "object_kind",
        "topology": "standalone_whole",
        "complete_bounded": True,
        "carrier_category": "unknown",
        "payload_categories": [],
        "shape_profile": "compact_volumetric",
        "identity_basis": "diagnostic_geometry",
        "diagnostic_parts": ["door", "frame"],
        "diagnostic_view_count": 4,
        "description": "A complete bounded industrial cabinet.",
        "attributes": ["metal"],
        "confidence": 0.91,
        "decision": "keep",
    }

    def event(source: str, image_ids: list[int], fingerprint: str) -> dict:
        return {
            "source": source,
            "event_id": f"object:7:{source}",
            "category": category,
            "decision": "keep",
            "confidence": 0.91,
            "confirmation_eligible": True,
            "semantic_vote_independence": "independent",
            "request_conditioning": "none_blind",
            "crop_image_ids": image_ids,
            "crop_image_ids_complete": True,
            "evidence_fingerprint_sha256": fingerprint * 64,
            "label_contract": copy.deepcopy(contract),
            "gravity_upright_normalization": {
                "schema": "farm.gravity-upright-normalization.v1",
                "provenance_complete": True,
                "crops": [
                    {
                        "status": "already_upright",
                        "image_id": image_id,
                        "source_image": f"camera_{image_id:06d}.png",
                    }
                    for image_id in image_ids
                ],
            },
        }

    return {
        "id": 7,
        "review_decision": "keep",
        "review_confidence": 0.91,
        "review_category": category,
        "review_label_contract": contract,
        "independent_blind_consensus": {
            "schema": "farm.independent-blind-label-consensus.v1",
            "status": "accepted",
            "accepted": True,
            "initial_category": category,
            "verification_category": category,
            "canonical_category": category,
            "evidence_pair_source": "exact_current_semantic_evidence_events",
            "initial_event_id": "object:7:initial_blind_review",
            "verification_event_id": "object:7:independent_verification",
            "initial_crop_image_ids": [101, 102],
            "verification_crop_image_ids": [103, 104],
            "initial_evidence_fingerprint_sha256": "a" * 64,
            "verification_evidence_fingerprint_sha256": "b" * 64,
            "reason_codes": ["exact_safe_canonical_noun_agreement"],
        },
        "semantic_evidence": [
            event("initial_blind_review", [101, 102], "a"),
            event("independent_verification", [103, 104], "b"),
        ],
    }


def _scope_incomplete_identity_vlm_row() -> dict:
    row = _accepted_vlm_row()
    folds = ([101, 107, 113], [104, 110, 116])
    all_ids = folds[0] + folds[1]
    poses = {}
    for image_id in all_ids:
        angle = math.radians(float(image_id - 100) * 10.0)
        center = [5.0 * math.cos(angle), 5.0 * math.sin(angle), 0.5]
        norm = math.sqrt(sum(value * value for value in center))
        poses[image_id] = {
            "camera_center_world_m": center,
            "camera_forward_world": [
                -center[0] / norm, -center[1] / norm, -center[2] / norm,
            ],
        }
    for event, image_ids in zip(row["semantic_evidence"], folds):
        event.update(crop_evidence_manifest(
            7,
            [
                (Path(f"img_{image_id:06d}_det_0000.npz"), b"jpeg")
                for image_id in image_ids
            ],
            frame_pose_index=poses,
            object_position_world_m=[0.0, 0.0, 0.0],
        ))
        event["category"] = "sign"
        event["gravity_upright_normalization"]["crops"] = [
            {
                "status": "already_upright",
                "image_id": image_id,
                "source_image": f"camera_{image_id:06d}.png",
            }
            for image_id in image_ids
        ]
        event["label_contract"].update({
            "category": "sign",
            "context_sufficient": True,
            "visible_target_coverage": 1.0,
            "missing_visible_parts": [],
            "included_non_target": [],
            "whole_object_evidence_complete": True,
        })
    verification = row["semantic_evidence"][1]
    verification["label_contract"].update({
        "complete_bounded": False,
        "visible_target_coverage": 0.95,
    })
    verification.update({"category": "unresolved object", "decision": "unknown"})
    row.update({
        "review_decision": "unknown",
        "review_category": "unknown",
        "review_confidence": 0.0,
        "review_label_contract": None,
        "semantic_publication_action": "quarantine_unresolved_semantics",
        "semantic_quarantined": True,
        "semantic_review_required": True,
        "review_gate_reason": "independent_blind_consensus_failed",
        "independent_blind_consensus": {
            "schema": "farm.independent-blind-label-consensus.v1",
            "status": "unresolved",
            "accepted": False,
            "initial_category": "sign",
            "verification_category": "unresolved object",
            "canonical_category": "",
        },
    })
    return row


def _candidate_conflict_vlm_row() -> dict:
    row = _accepted_vlm_row()
    canonical = "fire hose cabinet"
    candidate = "fire extinguisher box"
    row["category"] = canonical
    row["label_contract"] = {"category": canonical}
    row["contextual_semantic_review"] = {
        "status": "applied",
        "contextual_category": canonical,
        "whole_object_assessment": {
            "category": canonical,
            "contract": {"category": canonical},
        },
    }
    row["review_category"] = candidate
    row["review_label_contract"]["category"] = candidate
    row["semantic_evidence"][0]["source"] = "contextual_dual_panel_review"
    for event in row["semantic_evidence"]:
        event["category"] = candidate
    row["semantic_evidence"][0]["label_contract"]["category"] = candidate
    return row


def _run_apply_fixture(
    tmp_path,
    monkeypatch,
    review_row: dict,
    *,
    preserve_prior_label: bool,
    prior_category: str = "monitor",
    legacy_source_without_component_fraction: bool = False,
    legacy_geometry_without_release_evidence: bool = False,
    verification_mode: str = "independent_blind",
    review_report_overrides: dict | None = None,
) -> tuple[dict, list[dict], dict, dict, dict]:
    source_state = {
        "object_id": [7],
        "active": [True],
        "object_box_centers_m": [[1.0, 2.0, 3.0]],
        "object_box_dimensions_m": [[0.5, 0.6, 0.7]],
        "object_box_wxyz": [[1.0, 0.0, 0.0, 0.0]],
        "object_geometry_status": ["original_geometry"],
        "object_geometry_inside_rate": [0.51],
        "object_geometry_reprojection_error": [0.49],
        "object_geometry_valid_observations": [2],
        "object_geometry_consistent_observations": [2],
        "object_geometry_projected_box_iou": [0.52],
        "object_geometry_box_support_rate": [0.53],
        "object_geometry_voxel_inside_rate": [0.54],
        "object_geometry_voxel_component_fraction": [0.55],
        "object_geometry_voxel_volume_expansion": [1.2],
        "object_geometry_voxel_extent_candidate": [[0.5, 0.6, 0.7]],
        "object_geometry_voxel_retention_gate_rate": [0.54],
        "object_geometry_voxel_retention_policy": ["full_3d_voxel_retention"],
        "object_geometry_planar_exception_applied": [False],
        "object_geometry_planar_normal_axis": [-1],
        "object_geometry_planar_independent_views": [0],
        "object_geometry_planar_normal_q90_degrees": [float("inf")],
        "object_geometry_planar_median_view_thickness_m": [float("inf")],
        "object_geometry_planar_evidence_reasons": [["not_evaluated"]],
        "object_geometry_orientation_confidence": [0.56],
        "object_geometry_gravity_tilt_degrees": [4.0],
        "object_geometry_orientation_mode": ["original"],
        "object_mask_observations": [[{"path": "original-mask.npz"}]],
        "object_image_ids": [[10, 11]],
        "viewpoint_image_ids": [[10, 11]],
        "object_category": [prior_category],
        "object_caption": [f"Prior {prior_category} label."],
        "object_key_attributes": [["black"]],
        "object_semantic_tier": ["probable"],
        "object_semantic_status": ["prior_semantics"],
        "cov6": [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
        "images": [{"id": 10}, {"id": 11}],
        "image_positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    refined_state = copy.deepcopy(source_state)
    refined_state.update(
        {
            "object_box_centers_m": [[9.0, 8.0, 7.0]],
            "object_box_dimensions_m": [[1.5, 1.6, 1.7]],
            "object_geometry_status": ["geometry_pass"],
            "object_geometry_inside_rate": [0.91],
            "object_geometry_voxel_component_fraction": [0.95],
            "object_geometry_voxel_retention_gate_rate": [0.94],
            "object_geometry_voxel_retention_policy": [
                "planar_rgbd_normal_axis_exception_v1"
            ],
            "object_geometry_planar_exception_applied": [True],
            "object_geometry_planar_normal_axis": [2],
            "object_geometry_planar_independent_views": [5],
            "object_geometry_planar_normal_q90_degrees": [2.0],
            "object_geometry_planar_median_view_thickness_m": [0.04],
            "object_geometry_planar_evidence_reasons": [[]],
            "object_geometry_orientation_mode": ["gravity_constrained"],
            "object_mask_observations": [[{"path": "refined-mask.npz"}]],
            "object_image_ids": [[101, 102, 103]],
            "viewpoint_image_ids": [[101, 102, 103]],
        }
    )
    if legacy_source_without_component_fraction:
        source_state.pop("object_geometry_voxel_component_fraction")
        refined_state["object_geometry_voxel_component_fraction"] = torch.tensor(
            [0.95], dtype=torch.float32
        )
    source_path = tmp_path / "source.pt"
    refined_path = tmp_path / "refined.pt"
    refinement_path = tmp_path / "refinement.json"
    geometry_path = tmp_path / "geometry.json"
    review_path = tmp_path / "review.json"
    prior_catalog_path = tmp_path / "prior.json"
    output_state_path = tmp_path / "output.pt"
    output_catalog_path = tmp_path / "output-catalog.json"
    output_report_path = tmp_path / "output-report.json"
    torch.save({"state": source_state}, source_path)
    torch.save({"state": refined_state}, refined_path)
    refinement_path.write_text(
        json.dumps(
            {
                "schema": "farm.full-colmap-mask-refinement.v1",
                "status": "PASS",
                "refinement_role": "train_fit",
                "input_view_contract": {
                    "active_split": "train",
                    "state_fit_authorized": True,
                },
                "policy": {"state_fit_authorized": True},
                "objects": [{"object_id": 7, "accepted": True, "accepted_views": 3}],
            }
        ),
        encoding="utf-8",
    )
    geometry_path.write_text(
        json.dumps(
            {
                "schema": "farm.object-geometry-audit.v3",
                "objects": [
                    {
                        "object_id": 7,
                        "status": "geometry_pass",
                        "train_geometry_evidence": (
                            None
                            if legacy_geometry_without_release_evidence
                            else evaluate_obb_release_evidence(
                                geometry_fit_passed=True,
                                independent_physical_timestamps=5,
                                timestamp_median_projected_box_iou=0.75,
                                timestamp_q25_projected_box_iou=0.65,
                                orientation_materiality=0.50,
                                orientation_confidence=0.80,
                            )
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review_payload = {
        "schema": "farm.open-vocabulary-crop-review.v2",
        "verification_enabled": True,
        "verification_mode": verification_mode,
        "request_error_count": 0,
        "initial_request_error_ids": [],
        "verification_request_error_ids": [],
        "objects": [{**review_row, "id": 7}],
    }
    review_payload.update(review_report_overrides or {})
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    prior_catalog = [
        {
            "id": 7,
            "category": prior_category,
            "description": f"Prior {prior_category} label.",
            "attributes": ["black"],
            "semantic_tier": "probable",
            "semantic_status": "prior_semantics",
            "review_decision": "keep",
            "review_confidence": 0.88,
        }
    ]
    prior_catalog_path.write_text(json.dumps(prior_catalog), encoding="utf-8")
    argv = [
        "apply_farm_full_colmap_rescue.py",
        "--source-state",
        str(source_path),
        "--refined-state",
        str(refined_path),
        "--refinement",
        str(refinement_path),
        "--geometry-report",
        str(geometry_path),
        "--review-report",
        str(review_path),
        "--prior-catalog",
        str(prior_catalog_path),
        "--output-state",
        str(output_state_path),
        "--output-catalog",
        str(output_catalog_path),
        "--output-report",
        str(output_report_path),
    ]
    argv.append(
        "--preserve-prior-label-on-uncertainty"
        if preserve_prior_label
        else "--no-preserve-prior-label-on-uncertainty"
    )
    monkeypatch.setattr(sys, "argv", argv)
    assert apply_full_colmap_rescue() == 0
    output_payload = torch.load(
        output_state_path, map_location="cpu", weights_only=False
    )
    return (
        output_payload["state"],
        json.loads(output_catalog_path.read_text(encoding="utf-8")),
        json.loads(output_report_path.read_text(encoding="utf-8")),
        source_state,
        refined_state,
    )


def test_conflicting_blind_label_quarantines_semantics_but_copies_geometry(
    tmp_path, monkeypatch
) -> None:
    conflict = _accepted_vlm_row()
    conflict["review_decision"] = "unknown"
    conflict["review_category"] = "server"
    conflict["review_label_contract"]["category"] = "server"
    conflict["semantic_evidence"][0]["category"] = "server"
    conflict["semantic_evidence"][0]["label_contract"]["category"] = "server"
    conflict["semantic_evidence"][1]["category"] = "server"
    conflict["semantic_evidence"][1]["decision"] = "unknown"

    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        conflict,
        preserve_prior_label=False,
    )

    assert state["object_box_centers_m"] == refined["object_box_centers_m"]
    assert state["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_geometry_voxel_component_fraction"] == [0.95]
    assert state["object_category"] == ["monitor"]
    assert state["object_semantic_tier"] == ["probable"]
    assert state["object_semantic_status"] == ["prior_semantics"]
    assert state["object_full_colmap_rescue_status"] == [
        "accepted_geometry_prior_semantics_preserved"
    ]
    assert catalog[0]["center_m"] == refined["object_box_centers_m"][0]
    assert catalog[0]["position_world_m"] == refined["object_box_centers_m"][0]
    assert catalog[0]["dimensions_m"] == refined["object_box_dimensions_m"][0]
    assert (
        catalog[0]["metric_dimensions_m"]
        == refined["object_box_dimensions_m"][0]
    )
    assert catalog[0]["wxyz"] == refined["object_box_wxyz"][0]
    assert catalog[0]["category"] == "monitor"
    assert catalog[0]["review_decision"] == "keep"
    assert catalog[0]["label_publication_eligible"] is False
    assert catalog[0]["semantic_conflict_quarantined"] is True
    assert (
        catalog[0]["semantic_gate_reason"]
        == "geometry_rescued_semantic_conflict_prior_preserved"
    )
    decision = report["decisions"][0]
    assert decision["accepted"] is True
    assert decision["reasons"] == []
    assert (
        "vlm_initial_category_conflicts_prior_identity_without_verified_relabel"
        in decision["label_reasons"]
    )
    assert report["semantic_conflict_geometry_only_objects"] == 0
    assert report["semantic_conflict_geometry_only_object_ids"] == []
    assert report["prior_semantics_preserved_objects"] == 1
    assert report["prior_semantics_preserved_object_ids"] == [7]
    assert (
        report["policy"]["semantic_conflict_quarantines_label_without_erasing_geometry"]
        is True
    )


def test_explicit_review_quarantine_never_preserves_unproven_prior(
    tmp_path, monkeypatch
) -> None:
    row = _accepted_vlm_row()
    row.update({
        "review_decision": "unknown",
        "review_category": "unknown",
        "review_confidence": 0.0,
        "review_label_contract": None,
        "semantic_publication_action": "quarantine_unresolved_semantics",
        "semantic_quarantined": True,
        "semantic_review_required": True,
    })
    row["semantic_evidence"][1]["decision"] = "unknown"

    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        row,
        preserve_prior_label=True,
        prior_category="monitor",
    )

    assert state["object_box_centers_m"] == refined["object_box_centers_m"]
    assert state["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_category"] == ["unresolved object"]
    assert state["object_semantic_tier"] == ["geometry_only"]
    assert state["object_semantic_status"] == [
        "full_colmap_sam3_geometry_rescued_semantic_quarantined"
    ]
    assert state["object_full_colmap_rescue_status"] == ["accepted_geometry_only"]
    assert catalog[0]["category"] == "unresolved object"
    assert catalog[0]["review_decision"] == "unknown"
    assert catalog[0]["review_confidence"] == 0.0
    assert catalog[0]["explicit_semantic_quarantine"] is True
    assert catalog[0]["label_publication_eligible"] is False
    assert report["prior_semantics_preserved_object_ids"] == []
    assert report["semantic_conflict_geometry_only_object_ids"] == [7]
    assert (
        report["policy"][
            "explicit_review_quarantine_overrides_prior_label_preservation"
        ]
        is True
    )


def test_apply_re_adjudicates_raw_identity_but_blocks_inpainting(
    tmp_path, monkeypatch
) -> None:
    review_row = _scope_incomplete_identity_vlm_row()
    consensus = _independent_identity_re_adjudication(
        review_row, 0.70, verification_mode="independent_blind"
    )
    assert consensus["accepted"] is True
    assert consensus["canonical_category"] == "sign"
    assert consensus["mask_scope_incomplete"] is True
    assert consensus["inpainting_eligible"] is False

    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        review_row,
        preserve_prior_label=True,
        prior_category="monitor",
    )

    assert state["object_box_centers_m"] == refined["object_box_centers_m"]
    assert state["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_category"] == ["sign"]
    assert state["object_full_colmap_rescue_status"] == [
        "accepted_identity_verified_scope_refinement_required"
    ]
    assert catalog[0]["category"] == "sign"
    assert catalog[0]["semantic_identity_verified"] is True
    assert catalog[0]["mask_scope_incomplete"] is True
    assert catalog[0]["geometry_refinement_required"] is True
    assert catalog[0]["geometry_refinement_route"] == "mask_geometry_refinement"
    assert catalog[0]["label_publication_eligible"] is False
    assert catalog[0]["inpainting_eligible"] is False
    assert catalog[0]["explicit_semantic_quarantine"] is False
    assert report["label_verified_object_ids"] == []
    assert report[
        "semantic_identity_verified_scope_refinement_object_ids"
    ] == [7]
    assert report["decisions"][0]["inpainting_eligible"] is False


def test_candidate_conditioned_label_conflict_preserves_prior_semantics(
    tmp_path, monkeypatch
) -> None:
    review_row = _candidate_conflict_vlm_row()
    label_accepted, label_reasons = _vlm_gate(
        review_row, 0.70, verification_mode="independent_blind"
    )
    assert not label_accepted
    assert "vlm_candidate_category_conflicts_independent_blind" in label_reasons

    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        review_row,
        preserve_prior_label=False,
        prior_category="fire hose cabinet",
    )

    assert state["object_box_centers_m"] == refined["object_box_centers_m"]
    assert state["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_category"] == ["fire hose cabinet"]
    assert state["object_semantic_status"] == ["prior_semantics"]
    assert state["object_full_colmap_rescue_status"] == [
        "accepted_geometry_prior_semantics_preserved"
    ]
    assert catalog[0]["category"] == "fire hose cabinet"
    assert catalog[0]["label_publication_eligible"] is False
    assert catalog[0]["semantic_conflict_quarantined"] is True
    assert (
        catalog[0]["semantic_gate_reason"]
        == "geometry_rescued_semantic_conflict_prior_preserved"
    )
    decision = report["decisions"][0]
    assert decision["blind_category"] == "fire hose cabinet"
    assert decision["blind_prior_identity_compatible"] is True
    assert decision["semantic_conflict_quarantined"] is True
    assert (
        "vlm_candidate_category_conflicts_independent_blind"
        in decision["label_reasons"]
    )
    assert report["prior_semantics_preserved_object_ids"] == [7]
    assert report["semantic_conflict_geometry_only_object_ids"] == []
    assert (
        report["policy"]["candidate_conditioned_verification_can_publish_label"]
        is False
    )
    assert report["policy"]["semantic_conflict_preserves_prior_semantics"] is True


def test_valid_independent_blind_consensus_publishes_label(
    tmp_path, monkeypatch
) -> None:
    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        _accepted_vlm_row(),
        preserve_prior_label=True,
    )

    assert state["object_box_centers_m"] == refined["object_box_centers_m"]
    assert state["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_category"] == ["industrial cabinet"]
    assert state["object_full_colmap_rescue_status"] == ["accepted_label_verified"]
    assert catalog[0]["category"] == "industrial cabinet"
    assert catalog[0]["label_publication_eligible"] is True
    assert report["label_verified_object_ids"] == [7]


def test_legacy_geometry_pass_without_release_evidence_is_not_applied(
    tmp_path, monkeypatch
) -> None:
    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        _accepted_vlm_row(),
        preserve_prior_label=True,
        legacy_geometry_without_release_evidence=True,
    )

    assert refined["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_box_centers_m"] == source["object_box_centers_m"]
    assert state["object_full_colmap_rescue_status"] == [
        "rejected_preserved_original"
    ]
    assert catalog[0]["category"] == "monitor"
    decision = report["decisions"][0]
    assert decision["accepted"] is False
    assert "geometry_release_evidence_missing_or_invalid" in decision["reasons"]


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("spoofed_top_level", "vlm_top_level_category_conflicts_blind_consensus"),
        (
            "candidate_conditioned",
            "vlm_review_verification_mode_not_independent_blind",
        ),
        (
            "missing_consensus",
            "vlm_independent_blind_consensus_missing_or_invalid",
        ),
        ("missing_evidence", "vlm_two_pass_verification_missing"),
        (
            "disagreement",
            "vlm_independent_blind_consensus_category_disagreement",
        ),
        ("generic", "vlm_independent_blind_consensus_category_generic"),
    ),
)
def test_untrusted_semantics_never_publish_and_preserve_prior_label(
    tmp_path, monkeypatch, case: str, expected_reason: str
) -> None:
    row = _accepted_vlm_row()
    verification_mode = "independent_blind"
    if case == "spoofed_top_level":
        row["review_category"] = "server"
        row["review_label_contract"]["category"] = "server"
    elif case == "candidate_conditioned":
        verification_mode = "candidate_conditioned"
    elif case == "missing_consensus":
        row.pop("independent_blind_consensus")
    elif case == "missing_evidence":
        row["semantic_evidence"] = row["semantic_evidence"][:1]
    elif case == "disagreement":
        consensus = row["independent_blind_consensus"]
        consensus.update(
            {
                "status": "unresolved",
                "accepted": False,
                "verification_category": "server",
                "canonical_category": "",
            }
        )
        row["semantic_evidence"][1]["category"] = "server"
        row["semantic_evidence"][1]["label_contract"]["category"] = "server"
    elif case == "generic":
        row["independent_blind_consensus"].update(
            {
                "initial_category": "unit",
                "verification_category": "unit",
                "canonical_category": "unit",
            }
        )
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(case)

    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        row,
        preserve_prior_label=True,
        prior_category="monitor",
        verification_mode=verification_mode,
    )

    assert state["object_box_centers_m"] == refined["object_box_centers_m"]
    assert state["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_category"] == ["monitor"]
    assert state["object_full_colmap_rescue_status"] == [
        "accepted_geometry_prior_semantics_preserved"
    ]
    assert catalog[0]["category"] == "monitor"
    assert catalog[0]["label_publication_eligible"] is False
    assert report["label_verified_object_ids"] == []
    assert expected_reason in report["decisions"][0]["label_reasons"]


def test_independent_blind_gate_requires_disjoint_physical_source_views() -> None:
    row = _accepted_vlm_row()
    verification_manifest = row["semantic_evidence"][1][
        "gravity_upright_normalization"
    ]
    verification_manifest["crops"][0]["source_image"] = "camera_000101.png"

    accepted, reasons = _vlm_gate(
        row, 0.70, verification_mode="independent_blind"
    )

    assert not accepted
    assert "vlm_two_pass_physical_source_views_not_independent" in reasons


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("semantic_vote_independence", "vlm_verification_not_explicitly_independent"),
        ("request_conditioning", "vlm_verification_request_not_unconditioned"),
        ("confirmation_eligible", "vlm_verification_not_confirmation_eligible"),
    ),
)
def test_independent_blind_gate_requires_explicit_vote_provenance(
    field: str, reason: str
) -> None:
    row = _accepted_vlm_row()
    row["semantic_evidence"][1].pop(field)

    accepted, reasons = _vlm_gate(
        row, 0.70, verification_mode="independent_blind"
    )

    assert not accepted
    assert reason in reasons


def test_apply_rejects_review_transport_failures(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="contains transport failures"):
        _run_apply_fixture(
            tmp_path,
            monkeypatch,
            _accepted_vlm_row(),
            preserve_prior_label=True,
            review_report_overrides={
                "request_error_count": 1,
                "verification_request_error_ids": [7],
            },
        )


def test_incomplete_blind_review_preserves_original_geometry(
    tmp_path, monkeypatch
) -> None:
    incomplete = _accepted_vlm_row()
    incomplete["semantic_evidence"][0]["label_contract"]["complete_bounded"] = False

    state, catalog, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        incomplete,
        preserve_prior_label=False,
    )

    assert refined["object_box_centers_m"] != source["object_box_centers_m"]
    assert state["object_box_centers_m"] == source["object_box_centers_m"]
    assert state["object_geometry_voxel_component_fraction"] == [0.55]
    assert state["object_full_colmap_rescue_status"] == ["rejected_preserved_original"]
    assert catalog[0]["category"] == "monitor"
    assert report["accepted_object_ids"] == []
    assert report["semantic_conflict_geometry_only_objects"] == 0
    assert (
        "vlm_initial_not_complete_bounded_object" in report["decisions"][0]["reasons"]
    )


def test_apply_initializes_missing_forward_compatible_geometry_diagnostic(
    tmp_path, monkeypatch
) -> None:
    state, _, report, source, refined = _run_apply_fixture(
        tmp_path,
        monkeypatch,
        _accepted_vlm_row(),
        preserve_prior_label=False,
        legacy_source_without_component_fraction=True,
    )

    key = "object_geometry_voxel_component_fraction"
    assert key not in source
    assert isinstance(state[key], torch.Tensor)
    assert torch.equal(state[key], refined[key])
    assert state[key].tolist() == pytest.approx([0.95])
    assert report["accepted_object_ids"] == [7]


@pytest.mark.parametrize("container_type", ("list", "numpy", "torch"))
def test_copy_row_initializes_only_accepted_forward_compatible_diagnostic_row(
    container_type: str,
) -> None:
    key = "object_geometry_voxel_component_fraction"
    values = [0.1, 0.95, 0.3]
    if container_type == "numpy":
        values = np.asarray(values, dtype=np.float32)
    elif container_type == "torch":
        values = torch.tensor(values, dtype=torch.float32)
    destination = {}

    _copy_row(destination, {key: values}, key, index=1, count=3)

    actual = destination[key]
    if not isinstance(actual, list):
        actual = actual.tolist()
    assert actual == pytest.approx([0.0, 0.95, 0.0])


@pytest.mark.parametrize(
    "key", ("object_box_centers_m", "object_box_dimensions_m", "object_box_wxyz")
)
def test_copy_row_never_initializes_missing_core_geometry_field(key: str) -> None:
    destination = {}
    source = {key: torch.ones((3, 3), dtype=torch.float32)}

    with pytest.raises(ValueError, match="cannot copy row-aligned rescue field"):
        _copy_row(destination, source, key, index=1, count=3)

    assert destination == {}


def test_copy_row_rejects_misaligned_forward_compatible_geometry_diagnostic() -> None:
    key = "object_geometry_voxel_component_fraction"
    destination = {}
    source = {key: torch.ones(2, dtype=torch.float32)}

    with pytest.raises(ValueError, match="cannot initialize row-aligned rescue field"):
        _copy_row(destination, source, key, index=1, count=3)

    assert destination == {}


def test_blind_whole_object_gate_preserves_geometry_when_label_audit_is_unknown() -> (
    None
):
    row = _accepted_vlm_row()
    accepted, reasons = _initial_vlm_geometry_gate(row, 0.70)
    assert accepted and reasons == []

    label_unknown = copy.deepcopy(row)
    label_unknown["review_decision"] = "unknown"
    label_unknown["semantic_evidence"][1]["decision"] = "unknown"
    accepted, reasons = _initial_vlm_geometry_gate(label_unknown, 0.70)
    assert accepted and reasons == []
    label_accepted, label_reasons = _vlm_gate(
        label_unknown, 0.70, verification_mode="independent_blind"
    )
    assert not label_accepted
    assert "vlm_decision_not_keep" in label_reasons

    partial = copy.deepcopy(row)
    partial["semantic_evidence"][0]["label_contract"]["complete_bounded"] = False
    accepted, reasons = _initial_vlm_geometry_gate(partial, 0.70)
    assert not accepted
    assert "vlm_initial_not_complete_bounded_object" in reasons


def test_dual_context_whole_object_review_is_valid_blind_existence_evidence() -> None:
    row = _accepted_vlm_row()
    row["semantic_evidence"][0]["source"] = "contextual_dual_panel_review"
    row["semantic_evidence"][0]["event_id"] = (
        "object:7:contextual_dual_panel_review"
    )
    row["independent_blind_consensus"]["initial_event_id"] = (
        "object:7:contextual_dual_panel_review"
    )

    geometry_accepted, geometry_reasons = _initial_vlm_geometry_gate(row, 0.70)
    label_accepted, label_reasons = _vlm_gate(
        row, 0.70, verification_mode="independent_blind"
    )

    assert geometry_accepted and geometry_reasons == []
    assert label_accepted and label_reasons == []


def test_unverified_geometry_relabel_requires_prior_identity_compatibility() -> None:
    assert _category_compatible("cleaner", "vacuum cleaner")
    assert _category_compatible("floor cleaner", "floor scrubber")
    assert _category_compatible("palet", "pallet")
    assert not _category_compatible("cabinet", "fire hose cabinet")
    assert not _category_compatible("monitor", "server")
    assert not _category_compatible("cabinet", "backpack")


def test_two_pass_publication_requires_concrete_category_consensus() -> None:
    mismatch = _accepted_vlm_row()
    mismatch["semantic_evidence"][1]["category"] = "server"
    accepted, reasons = _vlm_gate(
        mismatch, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_two_pass_category_disagreement" in reasons

    generic = _accepted_vlm_row()
    for event in generic["semantic_evidence"]:
        event["category"] = "unit"
    accepted, reasons = _vlm_gate(
        generic, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_category_too_generic_for_publication" in reasons

    spelling = _accepted_vlm_row()
    spelling["category"] = "palet"
    spelling["label_contract"] = {"category": "palet"}
    spelling["review_category"] = "pallet"
    spelling["review_label_contract"]["category"] = "pallet"
    spelling["semantic_evidence"][0]["category"] = "palet"
    spelling["semantic_evidence"][0]["label_contract"]["category"] = "palet"
    spelling["semantic_evidence"][1]["category"] = "pallet"
    spelling["semantic_evidence"][1]["label_contract"]["category"] = "pallet"
    spelling["independent_blind_consensus"].update(
        {
            "initial_category": "palet",
            "verification_category": "pallet",
            "canonical_category": "pallet",
        }
    )
    accepted, reasons = _vlm_gate(
        spelling, 0.70, verification_mode="independent_blind"
    )
    assert accepted and reasons == []


def test_rest3d_inspired_vlm_gate_requires_whole_multiview_two_pass_object() -> None:
    accepted, reasons = _vlm_gate(
        _accepted_vlm_row(), 0.70, verification_mode="independent_blind"
    )
    assert accepted and reasons == []

    partial = _accepted_vlm_row()
    partial["review_label_contract"]["complete_bounded"] = False
    accepted, reasons = _vlm_gate(
        partial, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_not_complete_bounded_object" in reasons

    component = _accepted_vlm_row()
    component["review_label_contract"]["topology"] = "standalone_component"
    component["review_label_contract"]["category_role"] = "standalone_component"
    accepted, reasons = _vlm_gate(
        component, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_topology_not_whole" in reasons

    one_view = _accepted_vlm_row()
    one_view["semantic_evidence"][0]["crop_image_ids"] = [101]
    one_view["semantic_evidence"][1]["crop_image_ids"] = [101]
    accepted, reasons = _vlm_gate(
        one_view, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_multiview_evidence_missing" in reasons
    assert "vlm_two_pass_views_not_independent" in reasons

    reused = _accepted_vlm_row()
    reused["semantic_evidence"][1]["crop_image_ids"] = [101, 102]
    accepted, reasons = _vlm_gate(
        reused, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_two_pass_views_not_independent" in reasons

    same_evidence = _accepted_vlm_row()
    for event in same_evidence["semantic_evidence"]:
        event["evidence_fingerprint_sha256"] = "a" * 64
    accepted, reasons = _vlm_gate(
        same_evidence, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_two_pass_evidence_not_independent" in reasons

    vetoed = _accepted_vlm_row()
    vetoed["physical_form_assessment"] = {"hard_veto": True}
    accepted, reasons = _vlm_gate(
        vetoed, 0.70, verification_mode="independent_blind"
    )
    assert not accepted
    assert "vlm_prior_physical_form_hard_veto" in reasons


def test_geometry_prefers_complete_sam3_subset_and_falls_back_when_sparse(
    tmp_path,
) -> None:
    original = tmp_path / "masks/object_000007/img_000001_det_0000.npz"
    preferred_a = tmp_path / "masks/object_000007/img_000010_det_0000.npz"
    preferred_b = tmp_path / "masks/object_000007/img_000011_det_0000.npz"
    preferred_c = tmp_path / "masks/object_000007/img_000012_det_0000.npz"
    paths = [original, preferred_a, preferred_b, preferred_c]
    state = {
        "object_mask_observations": [
            [
                {"path": str(original), "source": "mapping"},
                *(
                    {"path": str(path), "source": "full_colmap_sam3_refinement"}
                    for path in (preferred_a, preferred_b, preferred_c)
                ),
            ]
        ]
    }
    selected, audit = _select_observation_paths(
        state,
        0,
        paths,
        preferred_source="full_colmap_sam3_refinement",
        minimum_preferred=3,
    )
    assert selected == [preferred_a, preferred_b, preferred_c]
    assert audit["mode"] == "preferred_source"
    selected, audit = _select_observation_paths(
        state,
        0,
        paths[:-1],
        preferred_source="full_colmap_sam3_refinement",
        minimum_preferred=3,
    )
    assert selected == paths[:-1]
    assert audit["mode"] == "all_resolved_fallback"


def test_view_acceptance_uses_only_hard_safety_gates() -> None:
    hard = {"depth_or_obb": True, "dominant_component": True}
    advisory = {"multiview_support_recall": False, "competing_mask_exclusion": False}
    assert _view_acceptance(512, hard)
    assert not _view_acceptance(0, hard)
    assert not _view_acceptance(512, {**hard, "negative_exclusion": False})
    assert not all(advisory.values())


def test_bounded_area_expansion_rejects_observed_projected_seed_leakage() -> None:
    gate = bounded_area_expansion_gate(
        seed_pixels=20_000,
        final_pixels=62_299,
        maximum_area_expansion=3.0,
    )
    assert gate["area_expansion"] == pytest.approx(3.11495)
    assert gate["passed"] is False
    assert not _view_acceptance(
        62_299, {"bounded_seed_area_expansion": bool(gate["passed"])}
    )


@pytest.mark.parametrize("expansion", [0.35, 1.0, 3.0])
def test_bounded_area_expansion_accepts_values_at_or_below_limit(
    expansion: float,
) -> None:
    gate = bounded_area_expansion_gate(
        seed_pixels=100,
        final_pixels=100 * expansion,
        maximum_area_expansion=3.0,
    )
    assert gate["area_expansion"] == pytest.approx(expansion)
    assert gate["passed"] is True
    assert _view_acceptance(
        int(100 * expansion), {"bounded_seed_area_expansion": bool(gate["passed"])}
    )


@pytest.mark.parametrize(
    ("seed", "final", "maximum"),
    [(0, 1, 3), (float("nan"), 1, 3), (1, -1, 3), (1, 1, float("inf"))],
)
def test_bounded_area_expansion_rejects_invalid_inputs(
    seed: float, final: float, maximum: float
) -> None:
    with pytest.raises(ValueError):
        bounded_area_expansion_gate(
            seed_pixels=seed,
            final_pixels=final,
            maximum_area_expansion=maximum,
        )


def test_stratified_queue_reserves_three_quarters_for_active_objects() -> None:
    rows = [
        {"object_id": value, "priority": 20.0 - value, "active": value >= 8}
        for value in range(16)
    ]
    selected = stratified_object_queue(rows, 8)
    assert sum(bool(row["active"]) for row in selected) == 6
    assert sum(not bool(row["active"]) for row in selected) == 2


def test_refinement_fails_closed_on_renderer_metric_scale_mismatch() -> None:
    plan = {"meters_per_scene_unit": 1.0016384971180945}
    frames = {
        "meters_per_scene_unit": 1.0016384971180945,
        "depth_units": "metres",
        "pose_translation_units": "metres",
    }
    validate_metric_contract(plan, frames)
    with pytest.raises(ValueError, match="metric scale does not match"):
        validate_metric_contract(plan, {**frames, "meters_per_scene_unit": 0.0295})
    with pytest.raises(ValueError, match="depth must be expressed in metres"):
        validate_metric_contract(plan, {**frames, "depth_units": "scene_units"})


def _temporal_candidate(
    name: str,
    *,
    timestamp: str,
    stream: str,
    frame_index: int,
    centre_x: float,
    score: float = 1.0,
) -> dict[str, object]:
    return {
        "name": name,
        "physical_timestamp": timestamp,
        "camera_stream_id": stream,
        "camera_stream_frame_index": frame_index,
        "camera_center_m": [centre_x, 0.0, 0.0],
        "view_direction": [0.0, 0.0, 1.0],
        "distance_m": 2.0,
        "score": score,
        "area_ratio": 0.05,
        "margin_ratio": 0.15,
        "in_frame_support_ratio": 1.0,
        "track_support_fraction": 0.05,
        "split": "train",
    }


def test_temporal_topup_keeps_diverse_minimum_and_builds_same_stream_window() -> None:
    episode = [
        _temporal_candidate(
            f"cam00_{timestamp}_center.png",
            timestamp=timestamp,
            stream="cam00_center",
            frame_index=index,
            centre_x=0.05 * index,
            score=1.2 if index == 11 else 1.0,
        )
        for index, timestamp in ((10, "000100"), (11, "000110"), (12, "000120"))
    ]
    diverse = [
        episode[1],
        _temporal_candidate(
            "cam01_000500_yaw_left.png",
            timestamp="000500",
            stream="cam01_yaw_left",
            frame_index=50,
            centre_x=1.0,
        ),
        _temporal_candidate(
            "cam00_000900_pitch_up.png",
            timestamp="000900",
            stream="cam00_pitch_up",
            frame_index=90,
            centre_x=2.0,
        ),
    ]

    selected, audit = plan_temporal_episode_topup(
        episode + diverse[1:],
        diverse,
        minimum_diverse_views=3,
        maximum_views=5,
        minimum_episode_frames=3,
        maximum_neighbor_frame_gap=2,
        maximum_translation_m=0.5,
        maximum_rotation_degrees=21.0,
    )

    assert audit["planned"] is True
    assert audit["camera_stream_id"] == "cam00_center"
    assert audit["physical_timestamps"] == ["000100", "000110", "000120"]
    assert audit["retained_diverse_views"] == 3
    assert temporal_episode_frame_count(selected) == 3
    assert len({str(row["physical_timestamp"]) for row in selected}) == 5


def test_temporal_topup_never_crosses_stream_or_pose_jump() -> None:
    seed = _temporal_candidate(
        "seed.png",
        timestamp="000100",
        stream="cam00_center",
        frame_index=10,
        centre_x=0.0,
    )
    wrong_stream = _temporal_candidate(
        "wrong-stream.png",
        timestamp="000110",
        stream="cam01_center",
        frame_index=11,
        centre_x=0.05,
    )
    pose_jump = _temporal_candidate(
        "pose-jump.png",
        timestamp="000120",
        stream="cam00_center",
        frame_index=11,
        centre_x=1.0,
    )
    other_a = _temporal_candidate(
        "other-a.png",
        timestamp="000500",
        stream="cam00_yaw_left",
        frame_index=50,
        centre_x=2.0,
    )
    other_b = _temporal_candidate(
        "other-b.png",
        timestamp="000900",
        stream="cam00_pitch_up",
        frame_index=90,
        centre_x=3.0,
    )

    selected, audit = plan_temporal_episode_topup(
        [seed, wrong_stream, pose_jump, other_a, other_b],
        [seed, other_a, other_b],
        minimum_diverse_views=3,
        maximum_views=5,
        minimum_episode_frames=3,
        maximum_neighbor_frame_gap=2,
        maximum_translation_m=0.5,
        maximum_rotation_degrees=21.0,
    )

    assert audit["planned"] is False
    assert audit["reason"] == "no_pose_continuous_same_stream_window"
    assert temporal_episode_frame_count(selected) == 0
