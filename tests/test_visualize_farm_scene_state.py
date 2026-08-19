import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from visualize_farm_scene_state import (  # noqa: E402
    classify_acceptance_candidates,
    classify_residual_candidates,
    make_acceptance_dashboard,
    make_label_uncertainty_dashboard,
    make_segmentation_sheet,
    spatial_label_indices,
    strongest_incident_edges,
)


def _frame(root: Path, index: int) -> dict:
    relative = Path("rgb") / f"frame_{index:03d}.jpg"
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 50, 3), 210, dtype=np.uint8)
    assert cv2.imwrite(str(root / relative), image)
    return {"frame_id": f"{index:06d}", "camera": "center", "rgb_path": str(relative)}


def _mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = np.ones((10, 12), dtype=np.uint8)
    np.savez_compressed(
        path,
        image_shape=np.asarray([40, 50], dtype=np.int32),
        raw_bits=np.packbits(crop.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(crop.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([5, 6, 17, 16], dtype=np.int32),
    )


def test_contact_sheet_reconstructs_saved_npz_masks_without_frame_jpegs(tmp_path):
    frames_root = tmp_path / "rgbd"
    frames = [_frame(frames_root, 0)]
    masks = tmp_path / "masks"
    _mask(masks / "object_000007" / "img_000000_det_0000.npz")
    output = tmp_path / "sheet.jpg"

    sheet, overlay_count, source = make_segmentation_sheet(
        masks, frames_root, frames, 10, output
    )

    assert output.is_file() and output.stat().st_size > 0
    assert sheet.shape == (570, 3400, 3)
    assert overlay_count == 1
    assert source == "saved_object_masks"
    # The rendered cell must contain colour that is absent from the neutral source RGB.
    assert np.any(sheet[50:570, :680, 0] != sheet[50:570, :680, 1])


def test_contact_sheet_falls_back_to_explicit_source_rgb(tmp_path):
    frames_root = tmp_path / "rgbd"
    frames = [_frame(frames_root, index) for index in range(2)]
    output = tmp_path / "sheet.jpg"

    sheet, overlay_count, source = make_segmentation_sheet(
        tmp_path / "empty_masks", frames_root, frames, 10, output
    )

    assert output.is_file() and output.stat().st_size > 0
    assert sheet.shape == (570, 3400, 3)
    assert overlay_count == 0
    assert source == "source_rgb_no_saved_masks"


def test_strongest_incident_edges_is_deterministic_and_covers_incident_nodes():
    edges = [
        (0, 1, 1.0), (0, 2, 4.0), (1, 2, 3.0), (1, 3, 2.0), (2, 3, 1.0),
    ]
    selected = strongest_incident_edges(edges)
    assert selected == strongest_incident_edges(list(reversed(edges)))
    assert {(i, j) for i, j, _ in selected} == {(0, 2), (1, 2), (1, 3)}
    assert {value for i, j, _ in selected for value in (i, j)} == {0, 1, 2, 3}


def test_spatial_labels_are_bounded_and_deterministic():
    indices = np.arange(30)
    points = np.column_stack((indices % 6, indices // 6)).astype(float)
    counts = np.arange(30, 0, -1)
    object_ids = np.arange(100, 130)
    first = spatial_label_indices(indices, points, counts, object_ids, limit=8)
    assert first == spatial_label_indices(indices, points, counts, object_ids, limit=8)
    assert len(first) == len(set(first)) == 8


def test_residual_classification_never_calls_distinct_relation_duplicate():
    rows = [
        {"first_id": 1, "second_id": 2, "raw_relation": "duplicate_strict"},
        {"first_id": 2, "second_id": 3, "raw_relation": "distinct"},
        {"first_id": 3, "second_id": 4, "raw_relation": "duplicate_overlap_fragment"},
    ]
    result = classify_residual_candidates(
        {"residual_strong_candidates": rows}, {1, 2, 3}
    )
    assert result["visible_duplicate_like"] == [rows[0]]
    assert result["visible_similar_distinct"] == [rows[1]]
    assert result["hidden_endpoint"] == [rows[2]]


def test_acceptance_classifier_uses_policy_evidence_not_raw_label() -> None:
    state = {
        "object_id": [1, 2],
        "object_category": ["machine", "machine"],
        "object_box_dimensions_m": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    }
    audit = {"pair_evidence": [{
        "first_id": 1, "second_id": 2, "raw_relation": "distinct",
        "feature_cosine": 0.99, "center_distance_m": 0.1,
        "mask_iou": 0.9, "mask_containment": 0.99, "mask_area_ratio": 0.9,
        "first_near_second": 0.99,
    }]}

    groups = classify_acceptance_candidates(state, audit, {1, 2}, {})

    assert len(groups["auto_holdout"]) == 1
    assert groups["auto_holdout"][0]["matched_rule"] == "same_category_mask_geometry_consensus"


def test_acceptance_and_label_dashboards_are_exact_4k(tmp_path: Path) -> None:
    acceptance = {
        "status": "WARN",
        "counts": {
            "presentation_before": 2, "presentation_after": 1, "held_objects": 1,
            "remaining_auto_clusters": 0, "remaining_blocking_clusters": 0,
            "label_hard_errors": 0, "merged_candidate_pairs": 1,
            "raw_candidate_records": 1, "uncertain_labels": 1,
            "high_uncertainty_labels": 0,
        },
        "classification_before": {"auto_holdout": 1},
        "classification_after": {"hidden_context": 1},
        "runtime": {"elapsed_seconds": 0.2, "budget_seconds": 30, "model_calls": 0},
        "holdouts": [{"id": 2, "canonical_id": 1, "reason": "acceptance:test"}],
        "warnings": [{"code": "sparse_presentation_crop_evidence", "count": 1}],
    }
    labels = {
        "sample_ids": [1],
        "objects": [{
            "id": 1, "category": "machine", "semantic_tier": "probable",
            "uncertainty_score": 0.6, "unique_view_count": 4,
            "independent_group_count": 1, "decodable_crop_count": 1,
            "uncertainty_reasons": ["probable_semantic_tier"],
        }],
    }
    acceptance_path = tmp_path / "acceptance_4k.jpg"
    labels_path = tmp_path / "labels_4k.jpg"

    make_acceptance_dashboard(
        acceptance, {"clusters": [{}]}, labels, "synthetic", acceptance_path
    )
    make_label_uncertainty_dashboard(labels, tmp_path / "masks", "synthetic", labels_path)

    for path in (acceptance_path, labels_path):
        image = cv2.imread(str(path))
        assert image is not None and image.shape[:2] == (2160, 3840)
