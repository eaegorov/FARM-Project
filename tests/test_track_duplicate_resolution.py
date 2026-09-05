from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from scripts.geometry import resolve_farm_track_duplicates as dedup_module
from scripts.geometry.resolve_farm_track_duplicates import (
    SameFrameMaskEvidence,
    TrackSplitEvidence,
    _classify_geometry_overwhelming_duplicate,
    _classify_large_object_same_mask_override,
    _classify_same_frame_track_fragment_override,
    _is_overwhelming_mask_geometry_conflict,
    _spatial_candidate_pairs,
    classify_pair,
    classify_same_frame_duplicate_override,
    mask_duplicate_override_allowed,
    resolve_duplicates,
    semantic_compatible,
)
from scene_graph.utils.geometry import voxelize_points


def _relation(**overrides) -> str:
    values = {
        "feature_cosine": 0.95,
        "center_distance_m": 0.20,
        "first_inside_second": 0.70,
        "second_inside_first": 0.65,
        "first_near_second": 0.95,
        "second_near_first": 0.94,
        "first_thinness": 0.40,
        "second_thinness": 0.45,
        "first_volume": 1.0,
        "second_volume": 1.1,
        "semantics_match": True,
    }
    values.update(overrides)
    return classify_pair(**values)


def test_strict_duplicate_requires_geometry_and_features():
    assert _relation() == "duplicate_strict"
    assert _relation(feature_cosine=0.70) == "distinct"
    assert _relation(center_distance_m=1.2) == "distinct"


def test_thin_surface_requires_support_inside_larger_parent():
    assert _relation(
        feature_cosine=0.89,
        first_inside_second=0.48,
        second_inside_first=0.10,
        first_near_second=0.72,
        second_near_first=0.20,
        first_thinness=0.12,
        first_volume=0.20,
        second_volume=1.20,
        semantics_match=False,
    ) == "part_surface"
    assert _relation(
        feature_cosine=0.89,
        first_inside_second=0.20,
        second_inside_first=0.10,
        first_near_second=0.30,
        second_near_first=0.20,
        first_thinness=0.12,
        first_volume=0.20,
        second_volume=1.20,
        semantics_match=False,
    ) == "distinct"


def test_contained_part_uses_metric_support_without_visual_aliasing():
    assert _relation(
        feature_cosine=0.78,
        center_distance_m=0.25,
        first_inside_second=0.92,
        second_inside_first=0.18,
        first_near_second=0.95,
        second_near_first=0.47,
        first_volume=0.10,
        second_volume=0.47,
        semantics_match=False,
    ) == "part_contained"
    assert _relation(
        feature_cosine=0.78,
        center_distance_m=0.25,
        first_inside_second=0.70,
        second_inside_first=0.18,
        first_near_second=0.95,
        second_near_first=0.47,
        first_volume=0.10,
        second_volume=0.47,
        semantics_match=False,
    ) == "distinct"


def test_contained_semantic_duplicate_requires_strong_one_way_support():
    assert _relation(
        first_inside_second=0.08,
        second_inside_first=0.97,
        first_near_second=0.42,
        second_near_first=0.61,
        semantics_match=True,
    ) == "duplicate_contained_semantic"
    assert _relation(
        first_inside_second=0.08,
        second_inside_first=0.97,
        first_near_second=0.42,
        second_near_first=0.61,
        semantics_match=False,
    ) == "distinct"


def test_visual_fragment_rule_does_not_hide_weakly_similar_intersection():
    assert _relation(
        feature_cosine=0.905,
        first_inside_second=0.54,
        second_inside_first=0.39,
        first_near_second=0.72,
        second_near_first=0.59,
        semantics_match=False,
    ) == "duplicate_overlap_fragment"
    assert _relation(
        feature_cosine=0.79,
        first_inside_second=0.85,
        second_inside_first=0.67,
        first_near_second=0.90,
        second_near_first=0.76,
        semantics_match=False,
    ) == "distinct"


def test_semantic_compatibility_is_category_or_caption_evidence():
    assert semantic_compatible("device", "electronic device", "", "")
    assert semantic_compatible("floor cleaner", "floor scrubber", "gray machine with handle and wheels", "gray machine with handle and wheels")
    assert not semantic_compatible("cabinet", "poster", "metal doors", "safety instructions")


def test_resolver_preserves_active_and_marks_only_redundant_row():
    grid = np.stack(np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij"), axis=-1).reshape(-1, 3)
    points = torch.as_tensor(grid * 0.04, dtype=torch.float32)
    first = voxelize_points(points, level=0)
    second = voxelize_points(points + 0.01, level=0)
    flat = torch.cat((first, second))
    state = {
        "active": torch.tensor([True, True]),
        "object_id": torch.tensor([10, 20]),
        "features": torch.tensor([[1.0, 0.0], [0.99, 0.01]], dtype=torch.float32),
        "object_box_centers_m": torch.tensor([[0.06, 0.06, 0.06], [0.07, 0.07, 0.07]]),
        "object_box_dimensions_m": torch.tensor([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "object_voxel_keys_flat": flat,
        "object_voxel_keys_offsets": torch.tensor([0, first.numel(), flat.numel()]),
        "object_voxel_levels": torch.tensor([0, 0]),
        "object_geometry_status": ["geometry_pass", "geometry_pass"],
        "object_category": ["machine", "machine"],
        "object_caption": ["gray machine", "gray machine"],
        "object_assembly_member_ids": [[], []],
        "object_semantic_tier": ["confirmed", "confirmed"],
        "object_evidence_count": torch.tensor([8, 4]),
        "object_geometry_projected_box_iou": torch.tensor([0.8, 0.6]),
        "object_geometry_inside_rate": torch.tensor([1.0, 1.0]),
        "object_geometry_voxel_inside_rate": torch.tensor([0.9, 0.8]),
    }
    active_before = state["active"].clone()
    report, evidence = resolve_duplicates(state)
    assert torch.equal(state["active"], active_before)
    assert report["suppressed_objects"] == 1
    assert report["duplicate_groups"][0]["canonical_id"] == 10
    assert state["object_display_status"] == ["canonical", "duplicate_suppressed"]
    assert int(state["object_duplicate_canonical_id"][1]) == 10
    assert evidence[0].relation.startswith("duplicate_")


def test_validated_assembly_suppresses_members_for_presentation_only():
    points_a = torch.as_tensor(np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"), axis=-1).reshape(-1, 3) * 0.04, dtype=torch.float32)
    points_b = points_a + torch.tensor([2.0, 0.0, 0.0])
    first = voxelize_points(points_a, level=0)
    second = voxelize_points(points_b, level=0)
    assembly = torch.cat((first, second))
    flat = torch.cat((first, second, assembly))
    state = {
        "active": torch.tensor([True, True, True]),
        "object_id": torch.tensor([10, 20, 30]),
        "features": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=torch.float32),
        "object_box_centers_m": torch.tensor([[0.04, 0.04, 0.04], [2.04, 0.04, 0.04], [1.04, 0.04, 0.04]]),
        "object_box_dimensions_m": torch.tensor([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [2.2, 0.2, 0.2]]),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        "object_voxel_keys_flat": flat,
        "object_voxel_keys_offsets": torch.tensor([0, first.numel(), first.numel() + second.numel(), flat.numel()]),
        "object_voxel_levels": torch.tensor([0, 0, 0]),
        "object_geometry_status": ["geometry_pass", "geometry_pass", "assembly_geometry_pass"],
        "object_category": ["part a", "part b", "machine"],
        "object_caption": ["part a", "part b", "assembled machine"],
        "object_caption_decision": ["keep", "keep", "keep"],
        "object_assembly_member_ids": [[], [], [10, 20]],
        "object_semantic_tier": ["confirmed", "confirmed", "confirmed"],
        "object_evidence_count": torch.tensor([5, 5, 8]),
        "object_geometry_projected_box_iou": torch.ones(3),
        "object_geometry_inside_rate": torch.ones(3),
        "object_geometry_voxel_inside_rate": torch.ones(3),
    }
    active_before = state["active"].clone()
    report, _ = resolve_duplicates(state)
    assert torch.equal(state["active"], active_before)
    assert state["object_display_status"] == [
        "assembly_member_suppressed", "assembly_member_suppressed", "canonical"
    ]
    assert state["object_duplicate_canonical_id"].tolist() == [30, 30, -1]
    assert report["suppressed_assembly_members"] == 2


def _pair_state(
    *,
    cannot_link: bool,
    co_visible: bool,
    point_shift: float = 0.01,
    dimensions: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> dict:
    grid = np.stack(
        np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    first_points = torch.as_tensor(grid * 0.04, dtype=torch.float32)
    second_points = first_points + torch.tensor([point_shift, 0.0, 0.0])
    first = voxelize_points(first_points, level=0)
    second = voxelize_points(second_points, level=0)
    flat = torch.cat((first, second))
    frame_rows = [[61], [61]] if co_visible else [[59], [61]]
    return {
        "active": torch.tensor([True, True]),
        "object_id": torch.tensor([95, 98]),
        "features": torch.tensor([[1.0, 0.0], [0.99, 0.01]], dtype=torch.float32),
        "object_box_centers_m": torch.tensor(
            [[0.06, 0.06, 0.06], [0.06 + point_shift, 0.06, 0.06]]
        ),
        "object_box_dimensions_m": torch.tensor([dimensions, dimensions]),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        "object_voxel_keys_flat": flat,
        "object_voxel_keys_offsets": torch.tensor([0, first.numel(), flat.numel()]),
        "object_voxel_levels": torch.tensor([0, 0]),
        "object_geometry_status": ["geometry_pass", "geometry_pass"],
        "object_category": ["machine", "machine"],
        "object_caption": ["gray industrial machine", "gray industrial machine"],
        "object_assembly_member_ids": [[], []],
        "object_semantic_tier": ["confirmed", "confirmed"],
        "object_evidence_count": torch.tensor([8, 4]),
        "object_geometry_projected_box_iou": torch.tensor([0.8, 0.6]),
        "object_geometry_inside_rate": torch.tensor([1.0, 1.0]),
        "object_geometry_voxel_inside_rate": torch.tensor([0.9, 0.8]),
        "object_image_ids": frame_rows,
        "object_mask_observations": [[], []],
        "cannot_link_object_ids": {95: {98}, 98: {95}} if cannot_link else {},
    }


def _write_mask(mask_root: Path, object_id: int, frame_id: int, mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    object_dir = mask_root / f"object_{object_id:06d}"
    object_dir.mkdir(parents=True, exist_ok=True)
    filename = f"img_{frame_id:06d}_det_0000.npz"
    path = object_dir / filename
    np.savez_compressed(
        path,
        image_shape=np.asarray(mask.shape, dtype=np.int32),
        raw_bits=np.packbits(crop.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(crop.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([x0, y0, x1, y1], dtype=np.int32),
    )
    return {
        "image_id": frame_id,
        "path": f"/farm-run/mapping/masks/object_{object_id:06d}/{filename}",
        "raw_pixels": int(np.count_nonzero(mask)),
    }


def _saved_pair_summary(
    *,
    iou: float,
    containment: float,
    area_ratio: float,
    first_id: int = 95,
    second_id: int = 98,
) -> dict:
    first_id, second_id = sorted((int(first_id), int(second_id)))
    return {
        "schema": "farm.object-pair-mask-overlap.v1",
        "max_images_per_pair": 16,
        "pairs": {
            f"{first_id}:{second_id}": {
                "first_object_id": first_id,
                "second_object_id": second_id,
                "images": [{
                    "image_id": 61,
                    "raw_iou": iou,
                    "raw_containment": containment,
                    "raw_area_ratio": area_ratio,
                    "raw_intersection_pixels": 1520 if iou > 0 else 0,
                    "first_area_pixels": 1600,
                    "second_area_pixels": 1520 if iou > 0 else 400,
                    "comparison_count": 1,
                    "update_count": 1,
                    "first_detection_idx": 0,
                    "second_detection_idx": 1,
                    "image_shape": [64, 64],
                    "source": "online_raw_detection_masks",
                    "evidence_fingerprint": "a" * 64,
                }],
            }
        },
    }


def test_resolved_same_category_track_fragment_boundaries_and_controls():
    evidence = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=859,
        iou=0.799,
        containment=0.997,
        area_ratio=0.804,
    )
    track = TrackSplitEvidence(23, 19, 1, 1 / 19)
    values = {
        "feature_cosine": 0.984,
        "center_distance_m": 0.388,
        "first_dimensions": np.asarray([1.233, 1.214, 1.210]),
        "second_dimensions": np.asarray([1.082, 0.424, 1.361]),
        "first_near_second_scaled": 0.846,
        "second_near_first_scaled": 1.0,
        "first_inside_second": 0.463,
        "second_inside_first": 0.875,
        "categories_resolved": True,
        "category_agreement": True,
        "category_exact": True,
        "raw_relation": "part_contained",
    }
    assert _classify_same_frame_track_fragment_override(
        evidence, track, **values
    ) == "resolved_same_category"
    assert _classify_same_frame_track_fragment_override(
        evidence, track, **{**values, "raw_relation": "part_surface"}
    ) is None

    insufficient_multicue = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=71,
        iou=0.60,
        containment=0.90,
        area_ratio=0.62,
    )
    assert _classify_same_frame_track_fragment_override(
        insufficient_multicue,
        TrackSplitEvidence(12, 8, 1, 1 / 8),
        **{
            **values,
            "feature_cosine": 0.90,
            "center_distance_m": 1.40,
            "first_dimensions": np.asarray([1.0, 0.8, 0.6]),
            "second_dimensions": np.asarray([0.9, 0.7, 0.5]),
            "first_near_second_scaled": 0.40,
            "second_near_first_scaled": 0.50,
            "first_inside_second": 0.30,
            "second_inside_first": 0.40,
            "raw_relation": "distinct",
        },
    ) is None


def test_resolved_alias_track_fragment_is_shape_strict_and_raw_relation_bounded():
    evidence = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=126,
        iou=0.797,
        containment=0.991,
        area_ratio=0.810,
    )
    track = TrackSplitEvidence(6, 9, 1, 1 / 6)
    values = {
        "feature_cosine": 0.962,
        "center_distance_m": 0.507,
        "first_dimensions": np.asarray([1.667, 0.904, 1.135]),
        "second_dimensions": np.asarray([1.603, 1.011, 0.962]),
        "first_near_second_scaled": 0.888,
        "second_near_first_scaled": 0.942,
        "first_inside_second": 0.530,
        "second_inside_first": 0.731,
        "categories_resolved": True,
        "category_agreement": False,
        "category_exact": False,
        "raw_relation": "duplicate_overlap_fragment",
    }
    assert _classify_same_frame_track_fragment_override(
        evidence, track, **values
    ) == "resolved_alias"
    assert _classify_same_frame_track_fragment_override(
        evidence, track, **{**values, "raw_relation": "distinct"}
    ) is None
    assert _classify_same_frame_track_fragment_override(
        evidence, track, **{**values, "categories_resolved": False}
    ) is None
    assert _classify_same_frame_track_fragment_override(
        evidence,
        track,
        **{**values, "first_dimensions": np.asarray([0.4, 0.4, 0.4])},
    ) is None


def test_same_category_large_track_split_contract_is_fail_closed():
    evidence = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=71,
        iou=0.65,
        containment=0.99,
        area_ratio=0.65,
    )
    dimensions = np.asarray([4.0, 3.0, 2.0])
    track = TrackSplitEvidence(3, 6, 1, 1 / 3)
    values = {
        "feature_cosine": 0.94,
        "center_distance_m": 0.30 * float(np.linalg.norm(dimensions)),
        "first_dimensions": dimensions,
        "second_dimensions": dimensions,
        "first_near_second_scaled": 0.0,
        "second_near_first_scaled": 0.0,
        "first_inside_second": 0.0,
        "second_inside_first": 0.0,
        "categories_resolved": True,
        "category_agreement": True,
        "category_exact": True,
        "raw_relation": "distinct",
    }
    assert _classify_same_frame_track_fragment_override(
        evidence, track, **values
    ) == "same_category_large_track_split"

    rejected = [
        (SameFrameMaskEvidence(**{**evidence.__dict__, "iou": 0.649}), track, values),
        (SameFrameMaskEvidence(**{**evidence.__dict__, "containment": 0.989}), track, values),
        (SameFrameMaskEvidence(**{**evidence.__dict__, "area_ratio": 0.649}), track, values),
        (evidence, track, {**values, "feature_cosine": 0.939}),
        (evidence, track, {**values, "center_distance_m": 0.301 * float(np.linalg.norm(dimensions))}),
        (evidence, TrackSplitEvidence(2, 6, 1, 0.5), values),
        (evidence, TrackSplitEvidence(4, 7, 1, 0.25), values),
        (evidence, TrackSplitEvidence(3, 6, 0, 0.0), values),
        (evidence, TrackSplitEvidence(3, 6, 2, 2 / 3), values),
        (evidence, track, {**values, "category_exact": False}),
        (evidence, track, {**values, "categories_resolved": False}),
        (evidence, track, {**values, "raw_relation": "part_contained"}),
        (evidence, track, {**values, "raw_relation": "part_surface"}),
    ]
    for rejected_evidence, rejected_track, rejected_values in rejected:
        assert _classify_same_frame_track_fragment_override(
            rejected_evidence, rejected_track, **rejected_values
        ) is None


def test_spatial_candidate_tree_preserves_distance_contract_and_nonfinite_filter():
    centers = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.2, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    dimensions = np.asarray([[0.25, 0.25, 0.25]] * 4)
    values = {
        "eligible": [0, 1, 2, 3],
        "object_ids": np.asarray([10, 20, 30, 40]),
        "centers": centers,
        "dimensions": dimensions,
        "cannot_links": {},
        "frame_cache": {0: {1}, 1: {2}, 2: {3}, 3: {4}},
        "near_distance_m": 0.20,
    }
    assert _spatial_candidate_pairs(**values) == {(0, 1), (1, 2)}
    assert _spatial_candidate_pairs(**{**values, "eligible": [3, 2, 1, 0]}) == {
        (0, 1),
        (1, 2),
    }


def test_track_fragment_override_is_presentation_only_and_audited(tmp_path: Path):
    state = _pair_state(cannot_link=True, co_visible=True)
    state["object_image_ids"] = [[1, 3, 5, 61], [2, 4, 6, 61]]
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.70, containment=0.97, area_ratio=0.70
    )
    active_before = state["active"].clone()
    centers_before = state["object_box_centers_m"].clone()
    dimensions_before = state["object_box_dimensions_m"].clone()
    orientations_before = state["object_box_wxyz"].clone()

    report, evidence = resolve_duplicates(state, mask_root=tmp_path / "relocated")

    assert torch.equal(state["active"], active_before)
    assert torch.equal(state["object_box_centers_m"], centers_before)
    assert torch.equal(state["object_box_dimensions_m"], dimensions_before)
    assert torch.equal(state["object_box_wxyz"], orientations_before)
    assert report["duplicate_groups"][0]["member_ids"] == [95, 98]
    assert evidence[0].relation == "duplicate_same_frame_track_fragment_override"
    assert evidence[0].override_tier == "track_fragment_resolved_same_category"
    assert evidence[0].common_over_min_track == 0.25
    assert report["track_fragment_override_counts"][
        "track_fragment_resolved_same_category"
    ] == 1


def test_resolved_alias_override_flows_through_cannot_link(monkeypatch, tmp_path: Path):
    state = _pair_state(cannot_link=True, co_visible=True)
    state["object_category"] = ["unit alpha", "unit beta"]
    state["object_caption"] = ["bounded alpha", "bounded beta"]
    state["object_image_ids"] = [[1, 3, 5, 7, 9, 61], [2, 4, 6, 8, 10, 12, 14, 16, 61]]
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.80, containment=0.99, area_ratio=0.81
    )
    monkeypatch.setattr(
        dedup_module, "classify_pair", lambda **_: "duplicate_overlap_fragment"
    )
    monkeypatch.setattr(dedup_module, "_strong_bidirectional_3d", lambda **_: False)

    report, evidence = resolve_duplicates(state, mask_root=tmp_path / "relocated")

    assert report["duplicate_groups"][0]["member_ids"] == [95, 98]
    assert evidence[0].relation == "duplicate_same_frame_track_fragment_override"
    assert evidence[0].override_tier == "track_fragment_resolved_alias"


def test_same_category_large_track_split_is_presentation_only_and_audited(
    monkeypatch, tmp_path: Path
):
    state = _pair_state(
        cannot_link=True,
        co_visible=True,
        point_shift=0.05,
        dimensions=(0.25, 0.25, 0.25),
    )
    state["object_image_ids"] = [
        [1, 3, 5, 61],
        [2, 4, 6, 8, 10, 12, 14, 61],
    ]
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.66, containment=0.995, area_ratio=0.66
    )
    monkeypatch.setattr(dedup_module, "classify_pair", lambda **_: "distinct")
    monkeypatch.setattr(dedup_module, "_strong_bidirectional_3d", lambda **_: False)
    active_before = state["active"].clone()

    report, evidence = resolve_duplicates(state, mask_root=tmp_path / "relocated")

    assert torch.equal(state["active"], active_before)
    assert state["object_display_status"] == ["canonical", "duplicate_suppressed"]
    assert report["duplicate_groups"][0]["member_ids"] == [95, 98]
    assert evidence[0].relation == "duplicate_same_frame_track_fragment_override"
    assert evidence[0].override_tier == (
        "track_fragment_same_category_large_track_split"
    )
    assert evidence[0].exact_resolved_category is True
    assert evidence[0].track_view_count_ratio == 2.0
    assert report["track_fragment_override_counts"][
        "track_fragment_same_category_large_track_split"
    ] == 1
    thresholds = report["decision_thresholds"]["same_category_large_track_split"]
    assert thresholds["raw_relation"] == "distinct"
    assert thresholds["requires_exact_resolved_category"] is True
    assert thresholds["presentation_only"] is True


def test_broad_persisted_candidate_is_audited_but_raw_distinct_stays_visible(
    tmp_path: Path,
):
    state = _pair_state(
        cannot_link=True,
        co_visible=True,
        point_shift=1.0,
        dimensions=(0.25, 0.25, 0.25),
    )
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.70, containment=0.97, area_ratio=0.70
    )

    report, evidence = resolve_duplicates(state, mask_root=tmp_path / "relocated")

    assert evidence == []
    assert report["duplicate_groups"] == []
    assert state["object_display_status"] == ["canonical", "canonical"]
    assert report["candidate_generation"] == {
        "spatial_pairs_evaluated": 0,
        "persisted_mask_pairs_available": 1,
        "persisted_mask_pairs_evaluated": 1,
        "persisted_mask_only_pairs_evaluated": 1,
        "union_pairs_evaluated": 1,
    }
    residual = report["residual_strong_candidates"][0]
    assert residual["raw_relation"] == "distinct"
    assert residual["candidate_source"] == "persisted_pair_mask"


def test_same_frame_duplicate_masks_can_narrowly_override_cannot_link(tmp_path: Path):
    state = _pair_state(cannot_link=True, co_visible=True)
    first_mask = np.zeros((64, 64), dtype=bool)
    first_mask[10:50, 10:50] = True
    second_mask = np.zeros((64, 64), dtype=bool)
    second_mask[10:50, 12:50] = True
    state["object_mask_observations"] = [
        [_write_mask(tmp_path, 95, 61, first_mask)],
        [_write_mask(tmp_path, 98, 61, second_mask)],
    ]

    report, evidence = resolve_duplicates(state, mask_root=tmp_path)

    assert report["duplicate_groups"][0]["member_ids"] == [95, 98]
    assert evidence[0].relation == "duplicate_same_frame_mask_override"
    assert evidence[0].cannot_link_overridden is True
    assert evidence[0].mask_iou >= 0.80
    assert evidence[0].mask_containment >= 0.95
    assert evidence[0].mask_area_ratio >= 0.75


def test_relaxed_mask_with_strong_3d_can_override_cannot_link(tmp_path: Path):
    state = _pair_state(cannot_link=True, co_visible=True)
    first_mask = np.zeros((64, 64), dtype=bool)
    first_mask[10:50, 10:50] = True
    second_mask = np.zeros((64, 64), dtype=bool)
    second_mask[10:50, 14:45] = True
    state["object_mask_observations"] = [
        [_write_mask(tmp_path, 95, 61, first_mask)],
        [_write_mask(tmp_path, 98, 61, second_mask)],
    ]

    report, evidence = resolve_duplicates(state, mask_root=tmp_path)

    assert report["duplicate_groups"][0]["member_ids"] == [95, 98]
    assert evidence[0].relation == "duplicate_same_frame_mask_relaxed_3d_override"
    assert evidence[0].override_tier == "relaxed_3d"
    assert 0.75 <= evidence[0].mask_iou < 0.80


def test_adjacent_co_visible_objects_are_not_merged_without_duplicate_masks(tmp_path: Path):
    state = _pair_state(cannot_link=False, co_visible=True)
    first_mask = np.zeros((64, 64), dtype=bool)
    first_mask[10:30, 5:25] = True
    second_mask = np.zeros((64, 64), dtype=bool)
    second_mask[10:30, 35:55] = True
    state["object_mask_observations"] = [
        [_write_mask(tmp_path, 95, 61, first_mask)],
        [_write_mask(tmp_path, 98, 61, second_mask)],
    ]

    report, evidence = resolve_duplicates(state, mask_root=tmp_path)

    assert report["duplicate_groups"] == []
    assert state["object_display_status"] == ["canonical", "canonical"]
    assert evidence == []
    assert report["residual_strong_candidates"][0]["rejection_reason"] == (
        "co_visible_mask_threshold_not_met"
    )


def test_online_pair_summary_survives_missing_sidecars_and_is_relocation_stable(
    tmp_path: Path,
):
    results = []
    for name in ("moved-a", "moved-b"):
        state = _pair_state(cannot_link=True, co_visible=True)
        state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
            iou=0.90, containment=1.0, area_ratio=0.95
        )
        report, evidence = resolve_duplicates(state, mask_root=tmp_path / name)
        results.append((report["duplicate_groups"], evidence[0]))

    assert results[0][0] == results[1][0]
    assert results[0][0][0]["member_ids"] == [95, 98]
    for _, evidence in results:
        assert evidence.relation == "duplicate_same_frame_mask_override"
        assert evidence.mask_evidence_source == "online_pair_summary"
        assert evidence.mask_evidence_fingerprint == "a" * 64


def test_saved_disjoint_masks_do_not_merge_adjacent_co_visible_objects(tmp_path: Path):
    state = _pair_state(cannot_link=False, co_visible=True)
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.0, containment=0.0, area_ratio=1.0
    )

    report, evidence = resolve_duplicates(state, mask_root=tmp_path / "relocated")

    assert evidence == []
    assert report["duplicate_groups"] == []
    assert state["object_display_status"] == ["canonical", "canonical"]
    residual = report["residual_strong_candidates"][0]
    assert residual["rejection_reason"] == "co_visible_mask_threshold_not_met"
    assert residual["mask_evidence_source"] == "online_pair_summary"


def test_comparable_canonical_sidecar_precedes_saved_overlap_summary(tmp_path: Path):
    state = _pair_state(cannot_link=False, co_visible=True)
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.90, containment=1.0, area_ratio=0.95
    )
    first_mask = np.zeros((64, 64), dtype=bool)
    first_mask[10:30, 5:25] = True
    second_mask = np.zeros((64, 64), dtype=bool)
    second_mask[10:30, 35:55] = True
    state["object_mask_observations"] = [
        [_write_mask(tmp_path, 95, 61, first_mask)],
        [_write_mask(tmp_path, 98, 61, second_mask)],
    ]

    report, evidence = resolve_duplicates(state, mask_root=tmp_path)

    assert evidence == []
    assert report["duplicate_groups"] == []
    residual = report["residual_strong_candidates"][0]
    assert residual["mask_evidence_source"] == "canonical_sidecars"
    assert residual["rejection_reason"] == "co_visible_mask_threshold_not_met"


def test_nested_part_mask_cannot_override_cannot_link():
    evidence = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=61,
        iou=0.25,
        containment=1.0,
        area_ratio=0.25,
    )
    assert not mask_duplicate_override_allowed(
        evidence, feature_cosine=0.99, three_d_corroborated=True
    )
    assert classify_same_frame_duplicate_override(
        evidence,
        feature_cosine=0.99,
        three_d_corroborated=True,
        semantic_compatible=True,
        center_distance_m=0.05,
        raw_relation="duplicate_strict",
        first_near_second=1.0,
        second_near_first=1.0,
        first_inside_second=1.0,
        second_inside_first=0.25,
        first_volume=0.25,
        second_volume=1.0,
    ) is None


def test_overwhelming_2d_tier_repairs_label_disagreement_but_is_fail_closed():
    evidence = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=61,
        iou=0.884,
        containment=0.997,
        area_ratio=0.888,
    )
    values = {
        "feature_cosine": 0.979,
        "three_d_corroborated": False,
        "semantic_compatible": False,
        "center_distance_m": 0.25,
        "raw_relation": "duplicate_overlap_fragment",
        "first_near_second": 0.733,
        "second_near_first": 1.0,
        "first_inside_second": 0.598,
        "second_inside_first": 0.922,
        "first_volume": 1.0,
        "second_volume": 0.63,
    }
    assert classify_same_frame_duplicate_override(evidence, **values) == "overwhelming_2d"
    assert classify_same_frame_duplicate_override(
        evidence, **{**values, "feature_cosine": 0.974}
    ) is None
    assert classify_same_frame_duplicate_override(
        evidence, **{**values, "first_near_second": 0.69}
    ) is None


def test_raw_distinct_overwhelming_mask_is_review_only(tmp_path: Path):
    state = _pair_state(
        cannot_link=True,
        co_visible=True,
        point_shift=0.70,
        dimensions=(0.25, 0.25, 0.25),
    )
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:50, 10:50] = True
    state["object_mask_observations"] = [
        [_write_mask(tmp_path, 95, 61, mask)],
        [_write_mask(tmp_path, 98, 61, mask)],
    ]

    report, evidence = resolve_duplicates(state, mask_root=tmp_path)

    assert evidence == []
    assert report["duplicate_groups"] == []
    assert report["review_only_conflicts"] == [{
        "first_id": 95,
        "second_id": 98,
        "raw_relation": "distinct",
        "feature_cosine": report["review_only_conflicts"][0]["feature_cosine"],
        "center_distance_m": report["review_only_conflicts"][0]["center_distance_m"],
        "mask_image_id": 61,
        "mask_iou": 1.0,
        "mask_containment": 1.0,
        "mask_area_ratio": 1.0,
        "mask_evidence_source": "canonical_sidecars",
        "candidate_source": "spatial",
        "disposition": "review_only_no_auto_merge",
        "reason": "overwhelming_mask_geometry_conflict",
    }]
    assert report["residual_strong_candidates"][0]["raw_relation"] == "distinct"


def test_geometry_overwhelming_rule_requires_disjoint_bidirectional_fit():
    values = {
        "feature_cosine": 0.868,
        "center_distance_m": 0.071,
        "first_near_second": 1.0,
        "second_near_first": 1.0,
        "first_inside_second": 0.575,
        "second_inside_first": 0.978,
        "first_volume": 0.069,
        "second_volume": 0.036,
        "first_dimensions": np.asarray([0.377, 0.252, 0.728]),
        "second_dimensions": np.asarray([0.351, 0.155, 0.661]),
        "cannot_link": False,
        "co_visible": False,
    }
    assert _classify_geometry_overwhelming_duplicate(**values) == (
        "duplicate_geometry_overwhelming"
    )
    assert _classify_geometry_overwhelming_duplicate(
        **{**values, "co_visible": True}
    ) is None
    assert _classify_geometry_overwhelming_duplicate(
        **{**values, "cannot_link": True}
    ) is None
    assert _classify_geometry_overwhelming_duplicate(
        **{**values, "second_near_first": 0.70}
    ) is None


def test_geometry_overwhelming_rule_rejects_nested_child_scale():
    assert _classify_geometry_overwhelming_duplicate(
        feature_cosine=0.99,
        center_distance_m=0.03,
        first_near_second=1.0,
        second_near_first=1.0,
        first_inside_second=1.0,
        second_inside_first=0.60,
        first_volume=0.20,
        second_volume=1.0,
        first_dimensions=np.asarray([0.30, 0.30, 0.30]),
        second_dimensions=np.asarray([1.0, 1.0, 1.0]),
        cannot_link=False,
        co_visible=False,
    ) is None


def test_raw_distinct_mask_conflict_helper_never_becomes_override():
    evidence = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=61,
        iou=0.90,
        containment=0.996,
        area_ratio=0.90,
    )
    assert _is_overwhelming_mask_geometry_conflict(
        evidence, feature_cosine=0.97, raw_relation="distinct"
    )
    assert classify_same_frame_duplicate_override(
        evidence,
        feature_cosine=0.99,
        three_d_corroborated=True,
        semantic_compatible=True,
        center_distance_m=0.05,
        raw_relation="distinct",
        first_near_second=1.0,
        second_near_first=1.0,
        first_inside_second=1.0,
        second_inside_first=1.0,
        first_volume=1.0,
        second_volume=1.0,
    ) is None


def test_large_object_same_mask_override_is_strict_and_shape_aware():
    evidence = SameFrameMaskEvidence(
        common_frames=1,
        compared_frames=1,
        compared_pairs=1,
        image_id=917,
        iou=0.90,
        containment=0.996,
        area_ratio=0.90,
    )
    values = {
        "feature_cosine": 0.964,
        "center_distance_m": 1.0,
        "first_dimensions": np.asarray([10.0, 6.0, 4.0]),
        "second_dimensions": np.asarray([9.0, 5.2, 3.8]),
        "semantic_compatible": True,
        "raw_relation": "distinct",
    }
    assert _classify_large_object_same_mask_override(evidence, **values) == (
        "duplicate_large_object_same_mask_override"
    )
    assert _classify_large_object_same_mask_override(
        evidence, **{**values, "raw_relation": "duplicate_overlap_fragment"}
    ) is None
    assert _classify_large_object_same_mask_override(
        evidence, **{**values, "semantic_compatible": False}
    ) is None
    assert _classify_large_object_same_mask_override(
        evidence,
        **{
            **values,
            "first_dimensions": np.asarray([0.4, 0.4, 0.4]),
            "second_dimensions": np.asarray([0.4, 0.4, 0.4]),
            "center_distance_m": 0.04,
        },
    ) is None
    assert _classify_large_object_same_mask_override(
        evidence,
        **{**values, "second_dimensions": np.asarray([8.0, 5.0, 2.0])},
    ) is None
    assert _classify_large_object_same_mask_override(
        evidence,
        **{**values, "second_dimensions": np.asarray([15.0, 4.0, 4.0])},
    ) is None


def test_large_out_of_radius_same_mask_pair_merges_presentation_only(tmp_path: Path):
    state = _pair_state(
        cannot_link=True,
        co_visible=True,
        point_shift=1.0,
        dimensions=(10.0, 6.0, 4.0),
    )
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.90, containment=0.996, area_ratio=0.90
    )
    active_before = state["active"].clone()

    report, evidence_rows = resolve_duplicates(
        state, mask_root=tmp_path / "relocated"
    )

    assert torch.equal(state["active"], active_before)
    assert report["duplicate_groups"][0]["member_ids"] == [95, 98]
    assert report["review_only_conflicts"] == []
    assert evidence_rows[0].raw_relation == "distinct"
    assert evidence_rows[0].relation == "duplicate_large_object_same_mask_override"
    assert evidence_rows[0].override_tier == "large_object_same_mask"
    assert evidence_rows[0].candidate_source == "persisted_pair_mask"


def test_scale_aware_near_distance_only_merges_disjoint_strong_tracks():
    state = _pair_state(
        cannot_link=False,
        co_visible=False,
        point_shift=0.35,
        dimensions=(2.5, 1.2, 2.8),
    )

    report, evidence = resolve_duplicates(state)

    assert report["duplicate_groups"][0]["member_ids"] == [95, 98]
    assert evidence[0].relation == "duplicate_scale_aware"
    assert evidence[0].near_distance_m > 0.20


def _part_state(*, semantic_match: bool, cannot_link: bool, parent_visible: bool = True) -> dict:
    child_points = torch.as_tensor(
        np.stack(
            np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3) * 0.04,
        dtype=torch.float32,
    )
    parent_points = torch.as_tensor(
        np.stack(
            np.meshgrid(np.arange(6), np.arange(6), np.arange(6), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3) * 0.04,
        dtype=torch.float32,
    )
    child = voxelize_points(child_points, level=0)
    parent = voxelize_points(parent_points, level=0)
    flat = torch.cat((child, parent))
    child_category = "machine" if semantic_match else "fire extinguisher"
    parent_category = "machine" if semantic_match else "barrier"
    return {
        "active": torch.tensor([True, True]),
        "object_id": torch.tensor([34, 21]),
        "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "object_box_centers_m": torch.tensor([[0.04, 0.04, 0.04], [0.10, 0.10, 0.10]]),
        "object_box_dimensions_m": torch.tensor([[0.16, 0.16, 0.16], [0.40, 0.40, 0.40]]),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        "object_voxel_keys_flat": flat,
        "object_voxel_keys_offsets": torch.tensor([0, child.numel(), flat.numel()]),
        "object_voxel_levels": torch.tensor([0, 0]),
        "object_geometry_status": ["geometry_pass", "geometry_pass"],
        "object_category": [child_category, parent_category],
        "object_caption": [child_category, parent_category],
        "object_caption_decision": ["keep", "keep" if parent_visible else "unknown"],
        "object_assembly_member_ids": [[], []],
        "object_semantic_tier": ["probable", "confirmed" if parent_visible else "geometry_only"],
        "object_evidence_count": torch.tensor([3, 12]),
        "object_geometry_projected_box_iou": torch.ones(2),
        "object_geometry_inside_rate": torch.ones(2),
        "object_geometry_voxel_inside_rate": torch.ones(2),
        "object_image_ids": [[1], [2]],
        "object_mask_observations": [[], []],
        "cannot_link_object_ids": {34: {21}, 21: {34}} if cannot_link else {},
    }


def test_semantic_mismatch_or_cannot_link_never_suppresses_geometry_part():
    cases = (
        (_part_state(semantic_match=False, cannot_link=False), "semantic_incompatible"),
        (_part_state(semantic_match=True, cannot_link=True), "cannot_link"),
    )
    for state, reason in cases:
        report, _ = resolve_duplicates(state)
        assert state["object_display_status"] == ["canonical", "canonical"]
        assert report["part_hierarchy"] == []
        assert report["rejected_part_suppressions"][0]["reason"] == reason
        assert report["rejected_part_suppressions"][0]["child_id"] == 34


def test_geometry_only_parent_cannot_hide_presentation_eligible_child():
    state = _part_state(
        semantic_match=True, cannot_link=False, parent_visible=False
    )
    report, _ = resolve_duplicates(state)
    assert state["object_display_status"] == ["canonical", "canonical"]
    assert report["part_hierarchy"] == []
    assert report["rejected_part_suppressions"][0]["reason"] == (
        "parent_not_presentation_eligible"
    )


def test_part_edge_is_revalidated_after_parent_duplicate_redirect():
    state = _part_state(semantic_match=True, cannot_link=False)
    original_flat = state["object_voxel_keys_flat"]
    original_offsets = state["object_voxel_keys_offsets"]
    parent_keys = original_flat[int(original_offsets[1]) : int(original_offsets[2])]
    state["object_voxel_keys_flat"] = torch.cat((original_flat, parent_keys))
    state["object_voxel_keys_offsets"] = torch.tensor(
        [0, int(original_offsets[1]), int(original_offsets[2]), original_flat.numel() + parent_keys.numel()]
    )
    state["active"] = torch.tensor([True, True, True])
    state["object_id"] = torch.tensor([34, 395, 417])
    state["features"] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    state["object_box_centers_m"] = torch.cat(
        (state["object_box_centers_m"], state["object_box_centers_m"][1:2]), dim=0
    )
    state["object_box_dimensions_m"] = torch.cat(
        (state["object_box_dimensions_m"], state["object_box_dimensions_m"][1:2]), dim=0
    )
    state["object_box_wxyz"] = torch.cat(
        (state["object_box_wxyz"], state["object_box_wxyz"][1:2]), dim=0
    )
    state["object_voxel_levels"] = torch.tensor([0, 0, 0])
    for key in (
        "object_geometry_status", "object_category", "object_caption",
        "object_caption_decision", "object_semantic_tier",
    ):
        state[key] = [*state[key], state[key][1]]
    state["object_assembly_member_ids"] = [[], [], []]
    state["object_evidence_count"] = torch.tensor([3, 5, 12])
    for key in (
        "object_geometry_projected_box_iou", "object_geometry_inside_rate",
        "object_geometry_voxel_inside_rate",
    ):
        state[key] = torch.ones(3)
    state["object_image_ids"] = [[1], [2], [3]]
    state["object_mask_observations"] = [[], [], []]
    state["cannot_link_object_ids"] = {34: {417}, 417: {34}}

    report, _ = resolve_duplicates(state)

    assert state["object_display_status"] == [
        "canonical", "duplicate_suppressed", "canonical"
    ]
    assert report["part_hierarchy"] == []
    assert any(
        row.get("display_parent_id") == 417
        and row["reason"] == "redirected_parent_cannot_link"
        for row in report["rejected_part_suppressions"]
    )


def _rejected_assembly_lineage_state() -> dict:
    whole_points = torch.as_tensor(
        np.stack(
            np.meshgrid(np.arange(6), np.arange(6), np.arange(6), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3) * 0.04,
        dtype=torch.float32,
    )
    child_points = torch.as_tensor(
        np.stack(
            np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3) * 0.04 + 0.04,
        dtype=torch.float32,
    )
    rows = [
        voxelize_points(whole_points, level=0),
        voxelize_points(child_points, level=0),
        voxelize_points(whole_points, level=0),
        voxelize_points(whole_points, level=0),
    ]
    flat = torch.cat(rows)
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + row.numel())
    state = {
        "active": torch.tensor([False, True, True, False]),
        "object_id": torch.tensor([100, 119, 1198, 1243]),
        "features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.999, 0.001], [1.0, 0.0]],
            dtype=torch.float32,
        ),
        "object_box_centers_m": torch.tensor(
            [[0.10, 0.10, 0.10], [0.08, 0.08, 0.08], [0.10, 0.10, 0.10], [0.10, 0.10, 0.10]]
        ),
        "object_box_dimensions_m": torch.tensor(
            [[0.35, 0.35, 0.35], [0.15, 0.15, 0.15], [0.40, 0.40, 0.40], [0.40, 0.40, 0.40]]
        ),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
        "object_voxel_keys_flat": flat,
        "object_voxel_keys_offsets": torch.tensor(offsets),
        "object_voxel_levels": torch.tensor([0, 0, 0, 0]),
        "object_geometry_status": [
            "geometry_rejected", "geometry_pass", "geometry_pass",
            "assembly_compound_geometry_rejected",
        ],
        "object_category": ["", "platform", "scissor lift", "scissor lift"],
        "object_caption": ["", "metal platform", "red scissor lift", "red scissor lift"],
        "object_caption_decision": ["", "keep", "keep", "keep"],
        "object_assembly_member_ids": [[], [], [], [100, 119]],
        "object_assembly_relation_type": ["", "", "", "part_whole_candidate"],
        "object_assembly_containment": torch.tensor([0.0, 0.0, 0.0, 0.99]),
        "object_assembly_shared_frames": torch.tensor([0, 0, 0, 4]),
        "object_semantic_tier": ["", "probable", "probable", "probable"],
        "object_evidence_count": torch.tensor([12, 4, 5, 16]),
        "object_geometry_projected_box_iou": torch.ones(4),
        "object_geometry_inside_rate": torch.ones(4),
        "object_geometry_voxel_inside_rate": torch.ones(4),
        "object_image_ids": [[], [61], [61], []],
        "object_mask_observations": [[], [], [], []],
        "cannot_link_object_ids": {119: {1198}, 1198: {119}},
        "object_pair_mask_overlap_evidence": _saved_pair_summary(
            iou=0.90,
            containment=0.999,
            area_ratio=0.90,
            first_id=100,
            second_id=1198,
        ),
    }
    return state


def test_rejected_assembly_lineage_can_hide_only_strict_contained_member():
    state = _rejected_assembly_lineage_state()
    active_before = state["active"].clone()
    report, _ = resolve_duplicates(state)
    assert torch.equal(state["active"], active_before)
    assert state["object_display_status"][1] == "assembly_member_suppressed"
    assert int(state["object_duplicate_canonical_id"][1]) == 1198
    lineage = report["rejected_assembly_lineage"][0]
    assert lineage["assembly_id"] == 1243
    assert lineage["anchor_id"] == 100
    assert lineage["surviving_whole_id"] == 1198
    assert lineage["suppressed_member_ids"] == [119]
    assert lineage["anchor_mask_evidence_source"] == "online_pair_summary"


def test_out_of_radius_persisted_mask_pair_is_review_only_not_merged(tmp_path: Path):
    state = _pair_state(
        cannot_link=True,
        co_visible=True,
        point_shift=1.0,
        dimensions=(0.25, 0.25, 0.25),
    )
    state["object_pair_mask_overlap_evidence"] = _saved_pair_summary(
        iou=0.90, containment=0.996, area_ratio=0.90
    )
    report, evidence = resolve_duplicates(state, mask_root=tmp_path / "relocated")
    assert evidence == []
    assert report["duplicate_groups"] == []
    conflict = report["review_only_conflicts"][0]
    assert (conflict["first_id"], conflict["second_id"]) == (95, 98)
    assert conflict["candidate_source"] == "persisted_pair_mask"
    assert conflict["raw_relation"] == "distinct"
    assert report["candidate_generation"]["persisted_mask_only_pairs_evaluated"] == 1


def test_cannot_link_overlap_fragment_is_explicit_review_conflict(monkeypatch):
    state = _pair_state(cannot_link=True, co_visible=False)
    monkeypatch.setattr(
        dedup_module, "classify_pair", lambda **_: "duplicate_overlap_fragment"
    )
    report, evidence = dedup_module.resolve_duplicates(state)
    assert evidence == []
    assert report["duplicate_groups"] == []
    conflict = report["review_only_conflicts"][0]
    assert conflict["raw_relation"] == "duplicate_overlap_fragment"
    assert conflict["cannot_link"] is True
    assert conflict["reason"] == (
        "duplicate_overlap_fragment_track_or_semantic_conflict"
    )
