import math
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_farm_part_whole import (
    classify_part_whole_relation,
    semantic_assembly_hint,
)
from scripts.build_farm_object_assemblies import (
    _copy_member_review_masks,
    _merge_mask_files,
    _unpack_mask_canvas,
    canonical_member_tuple,
    reviewed_assembly_eligible,
)
from scripts.build_farm_tiered_state import assembly_tiered_eligibility


def _classify_relation(**overrides):
    values = {
        "contained_frames": 3,
        "common_frames": 3,
        "observations_a": 5,
        "observations_b": 5,
        "median_iou": 0.40,
        "median_area_ratio": 0.40,
        "centre_distance_m": 0.25,
        "feature_similarity": 0.90,
        "min_shared_frames": 2,
        "min_contained_fraction": 0.50,
        "min_track_coverage": 0.50,
        "max_center_distance_m": 1.50,
        "min_feature_similarity": 0.55,
        "duplicate_iou": 0.75,
        "duplicate_area_ratio": 0.70,
    }
    values.update(overrides)
    return classify_part_whole_relation(**values)


def _tiered_eligibility(**overrides):
    values = {
        "geometry_active": True,
        "geometry_status": "assembly_geometry_pass",
        "state_member_ids": [4, 7],
        "review_member_ids": [7, 4],
        "review_decision": "keep",
        "review_category": "crane",
    }
    values.update(overrides)
    return assembly_tiered_eligibility(**values)


def test_canonical_member_tuple_is_order_independent_and_unique() -> None:
    assert canonical_member_tuple([7, 4, 7]) == (4, 7)
    assert canonical_member_tuple((299, 141)) == (141, 299)
    assert canonical_member_tuple(None) == ()


def test_validated_assembly_with_matching_review_is_tiered_eligible() -> None:
    eligible, reason = _tiered_eligibility()

    assert eligible is True
    assert reason


@pytest.mark.parametrize(
    ("overrides", "expected_reason_fragment"),
    [
        ({"geometry_active": False}, "inactive"),
        ({"geometry_status": "assembly_geometry_pending"}, "geometry"),
        ({"geometry_status": "assembly_geometry_rejected"}, "geometry"),
        # The legacy self-fitted voxel status is not independent geometry QA.
        ({"geometry_status": "assembly_voxel_pass"}, "geometry"),
        ({"review_decision": "reject"}, "review"),
        ({"review_category": "unknown"}, "category"),
        ({"review_category": ""}, "category"),
        ({"state_member_ids": []}, "member"),
        ({"review_member_ids": [4, 8]}, "member"),
    ],
)
def test_tiered_assembly_requires_geometry_semantics_and_exact_provenance(
    overrides: dict,
    expected_reason_fragment: str,
) -> None:
    eligible, reason = _tiered_eligibility(**overrides)

    assert eligible is False
    assert expected_reason_fragment in reason.lower()


def test_part_whole_relation_accepts_repeated_nested_evidence() -> None:
    relation_class, diagnostics = _classify_relation(
        contained_frames=2,
        common_frames=2,
        observations_a=4,
        observations_b=4,
        median_iou=0.36,
        median_area_ratio=0.36,
        centre_distance_m=0.19,
        feature_similarity=0.97,
    )

    assert relation_class == "part_whole"
    assert diagnostics["contained_fraction"] == pytest.approx(1.0)
    assert diagnostics["track_coverage"] == pytest.approx(0.5)


def test_near_duplicate_tracks_are_not_promoted_to_part_whole() -> None:
    relation_class, diagnostics = _classify_relation(
        median_iou=0.82,
        median_area_ratio=0.83,
    )

    assert relation_class == "duplicate_overlap"
    assert diagnostics["duplicate_overlap"] is True


def test_containment_only_in_shared_frames_does_not_hide_low_track_coverage() -> None:
    relation_class, diagnostics = _classify_relation(
        contained_frames=2,
        common_frames=2,
        observations_a=8,
        observations_b=9,
        median_iou=0.45,
        median_area_ratio=0.45,
    )

    assert relation_class == "rejected"
    assert diagnostics["contained_fraction"] == pytest.approx(1.0)
    assert diagnostics["track_coverage"] == pytest.approx(0.25)


@pytest.mark.parametrize("feature_similarity", [math.nan, math.inf, -math.inf])
def test_nonfinite_feature_similarity_fails_closed(feature_similarity: float) -> None:
    relation_class, diagnostics = _classify_relation(
        feature_similarity=feature_similarity,
    )

    assert relation_class == "rejected"
    assert diagnostics["feature_valid"] is False


def test_structurally_hinted_adjacent_fragments_can_form_review_candidate() -> None:
    relation_class, diagnostics = _classify_relation(
        contained_frames=0,
        common_frames=2,
        observations_a=3,
        observations_b=2,
        median_iou=0.04,
        median_area_ratio=0.35,
        centre_distance_m=1.20,
        feature_similarity=0.70,
        adjacent_frames=2,
        median_bbox_gap_ratio=0.05,
        semantic_assembly_hint=True,
    )

    assert relation_class == "complementary_parts"
    assert diagnostics["complementary_parts"] is True


def test_adjacent_fragments_without_structural_hint_fail_closed() -> None:
    relation_class, _ = _classify_relation(
        contained_frames=0,
        common_frames=2,
        observations_a=3,
        observations_b=2,
        median_iou=0.04,
        median_area_ratio=0.35,
        centre_distance_m=1.20,
        feature_similarity=0.70,
        adjacent_frames=2,
        median_bbox_gap_ratio=0.05,
        semantic_assembly_hint=False,
    )

    assert relation_class == "rejected"


def test_semantic_uncertainty_alone_never_authorizes_assembly() -> None:
    hinted, reasons = semantic_assembly_hint(
        {
            "semantic_status": "physical_component_or_contradiction_suppressed",
            "topology": "standalone_whole",
            "category_role": "whole_object",
            "category": "unresolved object",
        }
    )

    assert hinted is False
    assert reasons == []


def test_explicit_carrier_payload_topology_authorizes_review_candidate() -> None:
    hinted, reasons = semantic_assembly_hint(
        {"topology": "carrier_payload", "category": "box"}
    )

    assert hinted is True
    assert reasons == ["topology:carrier_payload"]


@pytest.mark.parametrize(
    ("review", "eligible"),
    [
        ({
            "review_decision": "keep", "review_category": "crane",
            "review_description": "complete crane",
            "review_label_contract": {
                "contract_valid": True, "complete_bounded": True,
                "topology": "standalone_whole",
            },
        }, True),
        ({
            "review_decision": "keep", "review_category": "crane",
            "review_description": "fragment",
            "review_label_contract": {
                "contract_valid": True, "complete_bounded": False,
                "topology": "attached_component",
            },
        }, False),
        ({
            "review_decision": "keep", "review_category": "crane",
            "review_description": "mixed masks",
            "review_label_contract": {
                "contract_valid": True, "complete_bounded": True,
                "topology": "mixed_targets",
            },
        }, False),
        ({"review_decision": "reject", "review_category": "crane", "review_description": "fragment"}, False),
        ({"review_decision": "keep", "review_category": "unknown", "review_description": "fragment"}, False),
        ({"review_decision": "keep", "review_category": "crane", "review_description": ""}, False),
        ({}, False),
    ],
)
def test_reviewed_assembly_materialisation_is_fail_closed(review: dict, eligible: bool) -> None:
    actual, _ = reviewed_assembly_eligible(review)
    assert actual is eligible


def _write_mask(
    path: Path,
    *,
    shape: tuple[int, int],
    raw_bbox: tuple[int, int, int, int],
    raw_crop: np.ndarray,
    inlier_bbox: tuple[int, int, int, int],
    inlier_crop: np.ndarray,
    crop_bytes: list[int],
) -> None:
    np.savez_compressed(
        path,
        image_shape=np.asarray(shape, dtype=np.int32),
        raw_bits=np.packbits(raw_crop.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(raw_crop.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray(raw_bbox, dtype=np.int32),
        inlier_bits=np.packbits(inlier_crop.reshape(-1), bitorder="little"),
        inlier_shape=np.asarray(inlier_crop.shape, dtype=np.int32),
        inlier_bbox_xyxy=np.asarray(inlier_bbox, dtype=np.int32),
        crop_jpeg_bytes=np.asarray(crop_bytes, dtype=np.uint8),
        crop_bbox_xyxy=np.asarray([0, 0, 1, 1], dtype=np.int32),
        crop_shape=np.asarray([1, 1], dtype=np.int32),
    )


def test_shared_frame_member_masks_are_logically_unioned(tmp_path: Path) -> None:
    shape = (8, 10)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    destination = tmp_path / "merged.npz"
    _write_mask(
        first,
        shape=shape,
        raw_bbox=(1, 1, 4, 3),
        raw_crop=np.ones((2, 3), dtype=np.uint8),
        inlier_bbox=(2, 1, 4, 2),
        inlier_crop=np.ones((1, 2), dtype=np.uint8),
        crop_bytes=[1, 2],
    )
    _write_mask(
        second,
        shape=shape,
        raw_bbox=(6, 5, 9, 7),
        raw_crop=np.ones((2, 3), dtype=np.uint8),
        inlier_bbox=(7, 6, 9, 7),
        inlier_crop=np.ones((1, 2), dtype=np.uint8),
        crop_bytes=[3, 4, 5, 6],
    )

    assert _merge_mask_files([first, second], destination) == 4
    expected_raw = np.zeros(shape, dtype=bool)
    expected_raw[1:3, 1:4] = True
    expected_raw[5:7, 6:9] = True
    expected_inlier = np.zeros(shape, dtype=bool)
    expected_inlier[1:2, 2:4] = True
    expected_inlier[6:7, 7:9] = True
    with np.load(destination, allow_pickle=False) as merged:
        assert np.array_equal(_unpack_mask_canvas(merged, "raw", shape), expected_raw)
        assert np.array_equal(_unpack_mask_canvas(merged, "inlier", shape), expected_inlier)
        assert merged["crop_jpeg_bytes"].tolist() == [3, 4, 5, 6]


def test_review_bank_keeps_member_crops_separate(tmp_path: Path) -> None:
    source_a = tmp_path / "a.npz"
    source_b = tmp_path / "b.npz"
    np.savez_compressed(source_a, crop_jpeg_bytes=np.asarray([1], dtype=np.uint8))
    np.savez_compressed(source_b, crop_jpeg_bytes=np.asarray([2], dtype=np.uint8))
    output = tmp_path / "review"
    output.mkdir()
    copied = _copy_member_review_masks(
        [({}, 4, source_a), ({}, 7, source_b)], output, 19
    )
    assert copied == 2
    assert sorted(path.name for path in output.glob("*.npz")) == [
        "img_000019_member_000004.npz",
        "img_000019_member_000007.npz",
    ]
