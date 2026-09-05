from copy import deepcopy

import numpy as np
import pytest

from farm_runtime.quality_baseline import json_digest
from farm_runtime.quality_benchmark import (
    LAYERS, annotation_template, assign_splits, check_annotation, compare_reviews,
    exposure_ledger, image_identity, mask_metrics, rasterize_regions, timestamp_macro_metrics, verify_packet,
)


def packet():
    return {"schema": "farm.manual-gold-packet.v1", "objects": [{"object_id": 1}],
            "split_by_timestamp": {"000001": "dev", "000002": "test"},
            "historically_observed_timestamps": ["000001"],
            "observations": [{"observation_id": "a", "object_id": 1,
                              "physical_timestamp": "000001", "image_name": "cam00_000001_center.png",
                              "camera": "cam00", "direction": "center", "split": "dev", "shape_hw": [8, 8]}]}


def annotated(p):
    a = annotation_template(p, "dev")
    a.update(reviewer_id="alice", independent=True)
    a["observations"][0].update(reviewed=True, state="visible", object_identity="1", regions=[
        {"layer": "whole_object", "points": [[1, 1], [3, 1], [3, 3], [1, 3]]},
        {"layer": "core", "points": [[1, 1], [2, 1], [2, 2], [1, 2]]},
        {"layer": "protected", "points": [[5, 5], [7, 5], [7, 7], [5, 7]]},
    ])
    a["objects"][0].update(scope_reviewed=True, attachment_policy="exclude",
                           carrier_content_policy="not_applicable")
    return a


def test_physical_timestamp_all_lenses_and_directions_are_one_fold():
    split = assign_splits([f"{i:06d}" for i in range(100)], {"000001"}, seed="frozen")
    assert split == assign_splits(list(reversed(list(split))), {"000001"}, seed="frozen")
    assert split["000001"] != "test"
    assert image_identity("00012_cam01_000001_yaw_left_123.jpg")["timestamp"] == "000001"
    p = packet()
    p["observations"].append({**p["observations"][0], "observation_id": "b", "camera": "cam01",
                              "direction": "yaw_left", "image_name": "cam01_000001_yaw_left.png", "split": "test"})
    assert "timestamp_split_leak:b" in verify_packet(p)


def test_historically_seen_test_is_rejected():
    p = packet()
    p["historically_observed_timestamps"].append("000002")
    assert "test_contains_historically_observed_timestamp" in verify_packet(p)


def test_blank_packet_cannot_count_as_manual_gold():
    p = packet()
    errors = check_annotation(p, annotation_template(p, "dev"))
    assert "missing_human_reviewer" in errors
    assert "unreviewed_observation:a" in errors


def test_two_independent_reviews_and_hash_are_required():
    p = packet()
    a = annotated(p)
    assert not check_annotation(p, a)
    b = deepcopy(a)
    assert compare_reviews(p, a, b)["status"] == "BLOCKED"
    b["reviewer_id"] = "bob"
    assert compare_reviews(p, a, b)["status"] == "AGREED"
    b["annotation_method"] = "SAM3"
    assert "pseudo_gt_not_allowed" in check_annotation(p, b)
    p["objects"].append({"object_id": 2})
    assert "packet_hash_mismatch" in check_annotation(p, a)


def test_manual_disagreement_is_preserved_not_voted_away():
    p = packet()
    a = annotated(p)
    b = deepcopy(a)
    b["reviewer_id"] = "bob"
    b["observations"][0]["regions"][0]["points"][1] = [4, 1]
    result = compare_reviews(p, a, b)
    assert result["status"] == "REQUIRES_ADJUDICATION"
    assert result["disagreements"][0]["different_pixels"]["whole_object"] > 0


def test_unknown_pixels_are_not_false_positives_and_boundary_is_separate():
    masks = {k: np.zeros((3, 4), dtype=bool) for k in LAYERS}
    masks["whole_object"][0, :2] = True
    masks["core"][0, 0] = True
    masks["boundary"][0, 1] = True
    masks["protected"][2, 3] = True
    masks["unknown"] = ~np.logical_or.reduce(list(masks.values()))
    prediction = np.ones((3, 4), dtype=bool)
    m = mask_metrics(prediction, masks)
    assert m["tp"] == 1 and m["fp"] == 1
    assert m["predicted_unknown_pixels"] == 9
    assert m["evaluated_pixels"] == 2
    assert m["boundary_coverage"] == 1


def test_polygon_holes_are_unknown_until_independently_labelled():
    masks = rasterize_regions([{"layer": "whole_object", "points": [[0, 0], [6, 0], [6, 6], [0, 6]],
                                "holes": [[[2, 2], [4, 2], [4, 4], [2, 4]]]}], (7, 7))
    assert masks["unknown"][3, 3]
    assert masks["whole_object"][0, 0]
    assert not masks["background"].any()


@pytest.mark.parametrize("region", [
    {"layer": "core", "points": [[0, 0], [2, 0], [2, 2]]},
    {"layer": "whole_object", "points": [[-1, 0], [2, 0], [2, 2]]},
    {"layer": "whole_object", "points": [[0, 0], [float("nan"), 0], [2, 2]]},
])
def test_invalid_or_inconsistent_polygons_fail(region):
    with pytest.raises(ValueError):
        rasterize_regions([region], (4, 4))


def test_five_virtual_directions_do_not_outvote_one_physical_timestamp():
    rows = [{"object_id": 1, "physical_timestamp": "000001", "image_name": f"cam00_000001_{d}.png", "iou": 1.0}
            for d in ("center", "yaw_left", "yaw_right", "pitch_up", "pitch_down")]
    rows.append({"object_id": 1, "physical_timestamp": "000002", "image_name": "cam00_000002_center.png", "iou": 0.0})
    result = timestamp_macro_metrics(rows)
    assert result["per_object"][0]["iou"] == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        timestamp_macro_metrics(rows + [rows[0]])


def test_metric_dimensions_need_measurement_evidence():
    p = packet()
    a = annotated(p)
    a["objects"][0]["physical_dimensions_m"] = [1, 2, 3]
    assert "unsubstantiated_metric_dimensions:1" in check_annotation(p, a)


def test_metric_evidence_is_structured_and_ambiguous_values_are_rejected():
    p = packet()
    a = annotated(p)
    obj = a["objects"][0]
    obj.update(physical_dimensions_m=[1, 2, 3], dimension_evidence="yes")
    assert "unsubstantiated_metric_dimensions:1" in check_annotation(p, a)
    obj["dimension_evidence"] = {"measurement_method": "measured with ruler",
        "scale_source": "calibrated ruler", "object_scope": "visible outer body",
        "axis_definition": "width, depth, height in documented upright object frame",
        "uncertainty_m": [0.01, 0.01, 0.02]}
    assert not check_annotation(p, a)
    obj["physical_dimensions_m"] = ["unknown", 2, 3]
    assert "unsubstantiated_metric_dimensions:1" in check_annotation(p, a)


def test_caption_disagreement_is_preserved_and_unknown_object_rejected():
    p = packet()
    a = annotated(p)
    b = deepcopy(a)
    b["reviewer_id"] = "bob"
    b["objects"][0]["caption"] = "Different interpretation of the visible hose."
    report = compare_reviews(p, a, b)
    assert report["status"] == "REQUIRES_ADJUDICATION"
    assert "caption" in report["disagreements"][0]["scope_fields"]
    b["objects"].append({**b["objects"][0], "object_id": 999})
    assert "unknown_scope_object" in check_annotation(p, b)


def test_renamed_navigation_visuals_mark_source_timestamp_as_seen(tmp_path):
    import json
    visuals = tmp_path / "factory" / "visuals"
    visuals.mkdir(parents=True)
    (visuals / "manifest.json").write_text(json.dumps({
        "schema": "farm.gold-navigation-visuals.v1",
        "source_hashes": {"cam01_000997_yaw_left.png": "source-hash"}}))
    ledger = exposure_ledger([tmp_path])
    assert ledger["observed_timestamps"] == ["000997"]
    assert ledger["frame_manifests"][0]["physical_timestamps"] == 1
    splits = assign_splits(["000997", "000998"], set(ledger["observed_timestamps"]), seed="fixed", test_fraction=1)
    assert splits["000997"] != "test" and splits["000998"] == "test"
