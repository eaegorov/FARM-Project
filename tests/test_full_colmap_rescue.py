from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.plan_farm_full_colmap_rescue import (
    allocate_global_view_budget,
    angular_separation_degrees,
    choose_diverse_views,
    identity_key,
    obb_corners,
    project_obb,
    stratified_object_queue,
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
from scripts.apply_farm_full_colmap_rescue import _initial_vlm_geometry_gate, _vlm_gate
from scripts.refine_farm_object_geometry import _select_observation_paths
from scripts.refine_farm_full_colmap_masks import (
    _object_acceptance,
    _view_acceptance,
    _multiview_world_support,
    add_projected_voxel_fallbacks,
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
    assert project_obb(
        shifted,
        _image(),
        _camera(),
        meters_per_scene_unit=1.0,
        min_margin_ratio=0.08,
        min_area_ratio=0.001,
        max_area_ratio=0.5,
    ) is None


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
    selected = choose_diverse_views(
        candidates, limit=2, min_view_angle_degrees=12.0
    )
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
        [[-0.10, -0.10, 2.0], [0.0, -0.10, 2.0], [0.10, -0.10, 2.0],
         [-0.10, 0.0, 2.0], [0.0, 0.0, 2.0], [0.10, 0.0, 2.0],
         [-0.10, 0.10, 2.0], [0.0, 0.10, 2.0], [0.10, 0.10, 2.0]],
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
    plan = {"objects": [
        {"object_id": 7, "selected_views": [{"name": "a.jpg"}, {"name": "b.jpg"}]},
        {"object_id": 9, "selected_views": [{"name": "a.jpg"}, {"name": "b.jpg"}]},
    ]}
    monkeypatch.setattr(
        "scripts.refine_farm_full_colmap_masks._object_voxel_points",
        lambda state, index: np.zeros((30, 3), dtype=np.float32),
    )
    existing = [{"object_id": 7}]
    output = add_projected_voxel_fallbacks(
        plan, source, rescue, frames, existing, minimum_views=2
    )
    assert [int(row["object_id"]) for row in output] == [7, 9]
    fallback = output[1]
    assert fallback["association_mode"] == "projected_metric_voxel_fallback"
    assert [row["source_image"] for row in fallback["matched_views"]] == ["a.jpg", "b.jpg"]


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


def test_multiview_world_support_requires_two_independent_metric_views(tmp_path) -> None:
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
        "rgb0": {**frame_base, "rgb_path": "rgb0.jpg", "depth_path": "d0.npy", "source_image": "a.jpg"},
        "rgb1": {**frame_base, "rgb_path": "rgb1.jpg", "depth_path": "d1.npy", "source_image": "b.jpg"},
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
    second = np.asarray(
        [[0.006, 0.004, 1.004], [0.40, 0.0, 1.0]], dtype=np.float32
    )
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
        {"accepted": True, "global_image_id": 11,
         "refinement_origin": "rest3d_bidirectional_propagation"},
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
    return {
        "review_decision": "keep",
        "review_confidence": 0.91,
        "review_category": "industrial cabinet",
        "review_label_contract": {
            "schema": "farm.open-vocabulary-object-label.v1",
            "contract_valid": True,
            "reason_codes": [],
            "category": "industrial cabinet",
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
        },
        "semantic_evidence": [
            {
                "source": "initial_blind_review",
                "decision": "keep",
                "confidence": 0.91,
                "confirmation_eligible": True,
                "crop_image_ids": [101, 102],
                "label_contract": {
                    "contract_valid": True,
                    "complete_bounded": True,
                    "topology": "standalone_whole",
                    "diagnostic_view_count": 2,
                },
            },
            {
                "source": "independent_verification",
                "decision": "keep",
                "crop_image_ids": [103, 104],
            },
        ],
    }


def test_blind_whole_object_gate_preserves_geometry_when_label_audit_is_unknown() -> None:
    row = _accepted_vlm_row()
    accepted, reasons = _initial_vlm_geometry_gate(row, 0.70)
    assert accepted and reasons == []

    label_unknown = copy.deepcopy(row)
    label_unknown["review_decision"] = "unknown"
    label_unknown["semantic_evidence"][1]["decision"] = "unknown"
    accepted, reasons = _initial_vlm_geometry_gate(label_unknown, 0.70)
    assert accepted and reasons == []
    label_accepted, label_reasons = _vlm_gate(label_unknown, 0.70)
    assert not label_accepted
    assert "vlm_decision_not_keep" in label_reasons

    partial = copy.deepcopy(row)
    partial["semantic_evidence"][0]["label_contract"]["complete_bounded"] = False
    accepted, reasons = _initial_vlm_geometry_gate(partial, 0.70)
    assert not accepted
    assert "vlm_initial_not_complete_bounded_object" in reasons


def test_rest3d_inspired_vlm_gate_requires_whole_multiview_two_pass_object() -> None:
    accepted, reasons = _vlm_gate(_accepted_vlm_row(), 0.70)
    assert accepted and reasons == []

    partial = _accepted_vlm_row()
    partial["review_label_contract"]["complete_bounded"] = False
    accepted, reasons = _vlm_gate(partial, 0.70)
    assert not accepted
    assert "vlm_not_complete_bounded_object" in reasons

    component = _accepted_vlm_row()
    component["review_label_contract"]["topology"] = "standalone_component"
    component["review_label_contract"]["category_role"] = "standalone_component"
    accepted, reasons = _vlm_gate(component, 0.70)
    assert not accepted
    assert "vlm_topology_not_whole" in reasons

    one_view = _accepted_vlm_row()
    one_view["semantic_evidence"][0]["crop_image_ids"] = [101]
    one_view["semantic_evidence"][1]["crop_image_ids"] = [101]
    accepted, reasons = _vlm_gate(one_view, 0.70)
    assert not accepted
    assert "vlm_multiview_evidence_missing" in reasons


def test_geometry_prefers_complete_sam3_subset_and_falls_back_when_sparse(tmp_path) -> None:
    original = tmp_path / "masks/object_000007/img_000001_det_0000.npz"
    preferred_a = tmp_path / "masks/object_000007/img_000010_det_0000.npz"
    preferred_b = tmp_path / "masks/object_000007/img_000011_det_0000.npz"
    preferred_c = tmp_path / "masks/object_000007/img_000012_det_0000.npz"
    paths = [original, preferred_a, preferred_b, preferred_c]
    state = {
        "object_mask_observations": [[
            {"path": str(original), "source": "mapping"},
            *(
                {"path": str(path), "source": "full_colmap_sam3_refinement"}
                for path in (preferred_a, preferred_b, preferred_c)
            ),
        ]]
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
