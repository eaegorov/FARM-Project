import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality.scope_family_review import (
    combine_packets,
    consensus_response,
    mark_ambiguous_parents,
    plan_packets,
    validate_response,
)


def answer(children=(2, 3), scope="single_whole_object"):
    return dict(parent_scope=scope, confidence="high", reason="One enclosure with a door and attached sign.",
                children=[dict(child_id=oid, relation="integral_part" if oid == 2 else "separate_object",
                               confidence="high", reason="A fitted door." if oid == 2 else "A mounted sign.") for oid in children],
                label="cabinet", caption="A white enclosure.")


def responses(value=None, children=(2, 3)):
    value = answer(children) if value is None else value
    return [dict(child_order=list(children)[::direction], frame_order=[10, 20][::direction],
                 raw=json.dumps(value), parsed=copy.deepcopy(value), validation_error=None)
            for direction in (1, -1)]


@pytest.mark.parametrize("change", [
    lambda r: r.update(extra=True),
    lambda r: r.update(parent_scope="cabinet"),
    lambda r: r["children"][0].update(child_id=True),
    lambda r: r["children"][0].update(child_id=999),
    lambda r: r["children"].append(copy.deepcopy(r["children"][0])),
    lambda r: r["children"].pop(),
    lambda r: r["children"][0].update(relation="part-of maybe"),
    lambda r: r.update(label=["cabinet"]),
])
def test_response_rejects_extra_or_unbound_decisions(change):
    value = answer()
    change(value)
    with pytest.raises(ValueError):
        validate_response(json.dumps(value), [2, 3])


def test_consensus_keeps_fitted_part_and_protects_mounted_sign():
    result = consensus_response(responses(), [2, 3], [10, 20])
    assert result["parent_accepted"]
    assert result["integral_ids"] == [2]
    assert result["separate_ids"] == [3]
    # A single child still has a distinct ordering through camera reversal.
    assert consensus_response(responses(children=(2,)), [2], [10, 20])["integral_ids"] == [2]


@pytest.mark.parametrize("scope", ["multiple_independent_objects", "structural_surface", "object_part", "unclear"])
def test_parent_without_single_object_evidence_cannot_authorize_parts(scope):
    result = consensus_response(responses(answer(scope=scope)), [2, 3], [10, 20])
    assert not result["parent_accepted"]
    assert result["integral_ids"] == []
    assert result["separate_ids"] == [3]


def test_order_consensus_rejects_low_confidence_disagreement_or_reused_order():
    for change in ("confidence", "scope", "order", "error"):
        rows = responses()
        if change == "order":
            rows[1]["child_order"], rows[1]["frame_order"] = rows[0]["child_order"], rows[0]["frame_order"]
        elif change == "error":
            rows[1]["validation_error"] = "bad JSON"
        else:
            rows[1]["parsed"]["confidence" if change == "confidence" else "parent_scope"] = "medium" if change == "confidence" else "multiple_independent_objects"
            rows[1]["raw"] = json.dumps(rows[1]["parsed"])
        assert not consensus_response(rows, [2, 3], [10, 20])["parent_accepted"]


def test_child_disagreement_abstains_without_discarding_other_child():
    rows = responses()
    rows[1]["parsed"]["children"][0]["relation"] = "separate_object"
    rows[1]["raw"] = json.dumps(rows[1]["parsed"])
    result = consensus_response(rows, [2, 3], [10, 20])
    assert result["parent_accepted"]
    assert result["integral_ids"] == [] and result["unclear_ids"] == [2]
    assert result["separate_ids"] == [3]
    rows[1]["parsed"]["label"] = "tampered"
    with pytest.raises(ValueError, match="raw output"):
        consensus_response(rows, [2, 3], [10, 20])


def packet(value):
    rows = responses(value)
    return dict(responses=rows, consensus=consensus_response(rows, [2, 3], [10, 20]))


def test_packets_require_same_parent_and_keep_unreviewed_child_unknown():
    good = packet(answer())
    decision = combine_packets(1, [2, 3, 4], [good])
    assert decision["accepted"] and decision["integral_ids"] == [2]
    assert decision["uncertain_ids"] == [4]
    bad = packet(answer(scope="multiple_independent_objects"))
    result = combine_packets(1, [2, 3], [good, bad])
    assert not result["accepted"] and result["integral_ids"] == []


def test_shared_child_marks_ambiguity_without_connecting_parents():
    first = combine_packets(1, [2, 3], [packet(answer())])
    second = combine_packets(7, [2, 3], [packet(answer())])
    ambiguous = mark_ambiguous_parents([first, second])
    assert ambiguous == [dict(child_id=2, competing_parent_ids=[1, 7])]
    assert first["unambiguous_integral_ids"] == second["unambiguous_integral_ids"] == []
    assert first["separate_ids"] == second["separate_ids"] == [3]


def test_packet_planning_requires_two_real_timestamps_and_current_containment():
    frames = [SimpleNamespace(image_id=i, physical_timestamp=t) for i, t in enumerate(["a", "a", "b", "test"])]
    objects = [SimpleNamespace(object_id=oid, observations=[SimpleNamespace(image_id=i) for i in range(4)]) for oid in [1, 2, 3]]
    run = SimpleNamespace(objects=objects, frame=lambda i: frames[i])
    split = dict(build_timestamps=["a", "b"], heldout_timestamps=["test"],
                 objects=[dict(object_id=oid, build_timestamps=["a", "b"]) for oid in [1, 2, 3]])
    parent = np.ones((8, 8), bool)
    child = np.zeros((8, 8), bool)
    child[2:6, 2:6] = True
    opened = []

    def mask(oid, image_id):
        opened.append(image_id)
        if oid == 3 and image_id == 2:
            return np.zeros_like(child), {}
        return (parent if oid == 1 else child), {}

    planned, deferred = plan_packets(run, split, 1, [2, 3], mask)
    assert deferred == [3]
    assert len(planned) == 1 and planned[0]["child_ids"] == [2]
    assert set(planned[0]["timestamps"]) == {"a", "b"}
    assert 3 not in opened
    # Object-local split must also exclude a nominally global build timestamp.
    split["objects"][1]["build_timestamps"] = ["a"]
    assert plan_packets(run, split, 1, [2], mask) == ([], [2])


def test_overlay_preserves_unselected_rgb_and_places_selection_on_source_pixels():
    from PIL import Image
    from farm_runtime.quality.scope_family_review import _overlay_crop

    pixels = np.full((32, 48, 3), 100, np.uint8)
    pixels[8:24, 16:32] = (20, 160, 40)
    original = pixels.copy()
    mask = np.zeros((8, 12), bool)
    mask[2:6, 4:8] = True
    image = _overlay_crop(Image.fromarray(pixels), mask, [0, 0, 12, 8],
                          (48, 32), (250, 100, 10), fill=0.2)
    output = np.asarray(image)
    np.testing.assert_array_equal(output[2, 2], original[2, 2])
    np.testing.assert_array_equal(output[16, 24], [66, 148, 34])
    np.testing.assert_array_equal(pixels, original)
    # Whole-view placement must not replace the crop with a resized object mask.
    assert not np.array_equal(output[16, 24], output[2, 2])


def test_grounded_sheet_keeps_full_view_for_a_tiny_rotated_part(tmp_path):
    from PIL import Image
    from farm_runtime.quality.scope_family_review import make_sheet
    from farm_runtime.quality_baseline import describe_file

    rgb = np.zeros((100, 200, 3), np.uint8)
    rgb[:, :100] = (60, 120, 180)
    rgb[:, 100:] = (180, 80, 40)
    source = tmp_path / "source.png"
    Image.fromarray(rgb).save(source)
    parent = np.zeros((25, 50), bool)
    parent[10:15, 22:28] = True
    child = np.zeros_like(parent)
    child[11:13, 24:26] = True
    masks = {1: parent, 2: child}
    sheet, record = make_sheet(SimpleNamespace(image_id=0),
                               dict(source=describe_file(source), applied_quarter_turns=1),
                               1, [2], lambda oid, fid: (masks[oid], {}))
    assert sheet.size == (1200, 696)
    assert record["full_view_xyxy"] == [0, 0, 25, 50]
    assert record["mask_grid_hw"] == [50, 25]
    assert record["crop_xyxy"] != record["full_view_xyxy"]
    assert record["crop_coordinate_system"] == "oriented_mask_grid"
    assert record["visual_encoding"] == "full_context_plus_aligned_rgb_mask_overlays"
    assert record["source"]["sha256"] == describe_file(source)["sha256"]
