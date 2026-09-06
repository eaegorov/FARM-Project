import json

import numpy as np
import pytest
from PIL import Image

from farm_runtime.object_scope import aggregate_scope_containment
from farm_runtime.quality.refinement import review_image
from farm_runtime.quality_baseline import describe_file
from farm_runtime.semantic_refinement import validate_scope


def test_containment_counts_timestamps_and_does_not_infer_ownership():
    nodes = [dict(id=i, frame=f"view{i//2}", timestamp=str(i // 4)) for i in range(6)]
    groups = [dict(id=1, members=[0, 2, 4]), dict(id=2, members=[1, 3, 5])]
    pairs = [dict(a=i, b=i + 1, containments=[0.3, 0.95]) for i in [0, 2, 4]]
    (row,) = aggregate_scope_containment(groups, nodes, pairs)
    assert (row["outer_group"], row["inner_group"]) == (1, 2)
    assert row["independent_timestamps"] == 2  # Three views are not three timestamps.
    assert row["physical_relation"] == "unresolved"
    assert row["status"] == "multiview_containment"
    reverse = [dict(a=4, b=5, containments=[0.95, 0.3])]
    rows = aggregate_scope_containment(groups, nodes, pairs[:2] + reverse)
    assert all(r["status"] == "conflicting_directions" for r in rows)


def test_similar_size_or_different_image_overlap_is_not_parent_child():
    groups = [dict(id=1, members=[0]), dict(id=2, members=[1])]
    nodes = [dict(id=i, frame="frame", timestamp="1") for i in range(2)]
    pairs = [dict(a=0, b=1, containments=[0.88, 0.89])]
    assert not aggregate_scope_containment(groups, nodes, pairs)
    nodes[1]["frame"] = "other"
    with pytest.raises(ValueError, match="same source"):
        aggregate_scope_containment(groups, nodes, pairs)


def test_semantic_evidence_needs_only_a_real_baseline_and_checks_hash(tmp_path):
    path = tmp_path / "mask.npz"
    mask = np.zeros((40, 60), bool)
    mask[10:30, 20:40] = True
    np.savez_compressed(path, baseline=mask)
    row = dict(masks=describe_file(path))  # No fabricated display_candidates.
    image = Image.new("RGB", (60, 40), (50, 60, 70))
    plain = review_image(image, row, semantic_only=True)
    assert plain.size == image.size
    assert plain.getpixel((0, 0)) == (50, 60, 70)
    scope = review_image(image, row, scope=True)
    assert scope.size == (1040, 550)
    other_rgb = review_image(
        Image.new("RGB", image.size, (190, 20, 10)), row, scope=True
    )
    # Scope shape is independent of photograph colors; PHOTO interior is intact.
    np.testing.assert_array_equal(
        np.asarray(scope)[:, 520:], np.asarray(other_rgb)[:, 520:]
    )
    assert scope.getpixel((254, 286)) == (50, 60, 70)
    assert scope.getpixel((774, 286)) == (255, 255, 255)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        review_image(image, row, scope=True)


def test_scope_requires_explicit_granularity_and_keeps_identity_unknown():
    value = dict(
        label="door panel",
        caption="A rectangular panel.",
        functional_identity=dict(label=None, direct_visual_evidence=""),
        integral_parts=[],
        external_connections=[],
        supporting_objects=[],
        contained_objects=[],
        uncertainty=[],
        observed_image_ids=[1, 2],
        same_physical_target=None,
        scope=dict(
            kind="integral_part",
            parent_description="enclosure",
            mask_coverage_issues=[],
        ),
    )
    assert validate_scope(json.dumps(value), [1, 2])["same_physical_target"] is None
    value["scope"]["kind"] = "whole object with maybe parts"
    with pytest.raises(ValueError, match="scope kind"):
        validate_scope(json.dumps(value), [1, 2])


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("label", "observable English candidate category", "schema placeholder"),
        ("caption", "one factual English sentence", "schema placeholder"),
        ("label", "integral_part", "scope enum"),
    ],
)
def test_scope_rejects_schema_echo_and_enum_as_physical_identity(field, value, reason):
    result = dict(
        label="box",
        caption="A rectangular box.",
        functional_identity=dict(label=None, direct_visual_evidence=""),
        integral_parts=[],
        external_connections=[],
        supporting_objects=[],
        contained_objects=[],
        uncertainty=[],
        observed_image_ids=[1],
        same_physical_target=True,
        scope=dict(
            kind="whole_object", parent_description=None, mask_coverage_issues=[]
        ),
    )
    result[field] = value
    with pytest.raises(ValueError, match=reason):
        validate_scope(json.dumps(result), [1])


def test_scope_context_uses_exact_source_rotation_and_rejects_stale_rgb(tmp_path):
    from farm_runtime.quality.refinement import scope_context_image
    from farm_runtime.angular_discovery import rotate_image

    rgb = np.zeros((60, 100, 3), np.uint8)
    rgb[:] = [35, 70, 90]
    rgb[30:, 50:] = [170, 60, 20]
    path = tmp_path / "source.png"
    Image.fromarray(rgb).save(path)
    row = dict(
        source_image=describe_file(path), crop_source_xyxy=[20, 10, 40, 30], turns=1
    )
    context = scope_context_image(row)
    assert context.size == (60, 100)
    expected = rgb.copy()
    from PIL import ImageDraw

    marked = Image.fromarray(expected)
    ImageDraw.Draw(marked).rectangle((20, 10, 39, 29), outline="#ffd84d", width=1)
    np.testing.assert_array_equal(
        np.asarray(context), rotate_image(np.asarray(marked), 1)
    )
    row["crop_source_xyxy"] = [0, 0, 101, 20]
    with pytest.raises(ValueError, match="source crop"):
        scope_context_image(row)
    row["crop_source_xyxy"] = [20, 10, 40, 30]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        scope_context_image(row)


def test_context_sheet_preserves_clean_photo_and_binary_shape_when_enlarged(tmp_path):
    mask = np.zeros((40, 60), bool)
    mask[10:30, 20:40] = True
    path = tmp_path / "mask.npz"
    np.savez_compressed(path, baseline=mask)
    rgb = Image.new("RGB", (60, 40), (35, 70, 90))
    context = Image.new("RGB", (100, 80), (180, 90, 20))
    sheet = review_image(
        rgb, dict(masks=describe_file(path)), scope=True, context=context
    )
    assert sheet.size == (1560, 550)
    assert sheet.getpixel((774, 286)) == (35, 70, 90)
    assert sheet.getpixel((1294, 286)) == (255, 255, 255)
    assert context.size == (100, 80)
