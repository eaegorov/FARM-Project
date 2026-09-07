"""Evidence-preserving display checks; no model or GPU workloads."""
from dataclasses import replace

import numpy as np
import pytest

from farm_runtime.quality.scene_review import (
    ReviewFrame,
    ReviewOptions,
    gravity_display_rotation,
    render_object_sheet,
    render_scene_sheet,
    rotate_layer,
)


def _frame():
    rgb = np.full((80, 120, 3), 90, np.uint8)
    first = np.zeros((80, 120), bool)
    first[12:40, 15:50] = True
    first[22:29, 26:33] = False  # real mask hole must remain visible
    second = np.zeros_like(first)
    second[55:65, 88:101] = True
    empty = np.zeros_like(first)
    return ReviewFrame(
        rgb=rgb,
        native_masks={9: first, 2: second, 30: empty},
        catalog_objects={9: {"label": "шкаф"}, 2: {"label": "панель"}, 30: {}},
        projected_obbs={
            9: np.array([[[15, 12], [50, 12]], [[50, 12], [50, 40]]], float),
            2: np.array([[[88, 55], [101, 55]]], float),
            30: np.array([[[1, 1], [2, 2]]], float),
        },
        source_label="Synthetic verification; not scene evidence",
    )


def _options():
    return ReviewOptions(panel_width=400, panel_max_height=350,
                         object_panel_width=220, object_panel_height=180,
                         min_obb_mask_pixels=1)


def test_gravity_roll_corrects_continuous_angle_and_off_axis_projection():
    angle = np.radians(31)
    up = np.array([np.sin(angle), -np.cos(angle), 0])
    assert gravity_display_rotation(np.eye(3), up) == pytest.approx(31)
    k = np.array([[700, 5, 300], [0, 600, 200], [0, 0, 1]], float)
    up = np.array([0.2, -0.4, 0.8])
    uv = np.array([480, 80])
    result = gravity_display_rotation(np.eye(3), up, k, uv)
    g = up / np.linalg.norm(up)
    projected = np.array([700*g[0] + 5*g[1] - 180*g[2], 600*g[1] + 120*g[2]])
    theta = np.radians(result)
    corrected = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ projected
    assert corrected[0] == pytest.approx(0, abs=1e-9)
    assert corrected[1] < 0
    assert gravity_display_rotation(np.eye(3), [0, 0, 1]) is None


def test_rotation_keeps_identical_pixel_geometry_and_mask_holes():
    frame = _frame()
    mask = frame.native_masks[9]
    rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    for angle in (0, 90, 180, 270):
        result = rotate_layer(rgb, angle)
        transformed_mask = rotate_layer(mask, angle, mask=True)
        assert np.array_equal(result[..., 0] > 127, transformed_mask)
        assert transformed_mask.sum() == mask.sum()
    # Non-right-angle display expands instead of cropping the source canvas.
    turned = rotate_layer(mask, 31, mask=True)
    assert turned.shape[0] > mask.shape[0] and turned.shape[1] > mask.shape[1]
    assert turned.dtype == bool
    assert abs(int(turned.sum()) - int(mask.sum())) < 12


def test_object_crop_keeps_lost_remote_component_and_empty_requested_object():
    frame = _frame()
    before = np.zeros_like(frame.native_masks[9])
    before[71:78, 108:119] = True  # distant lost component cannot be cropped away
    frame = replace(frame, reference_masks={9: before})
    sheet = render_object_sheet(frame, [9, 30], "Проверка изменений", _options())
    assert sheet.metadata["displayed_object_ids"] == [9, 30]
    first, empty = sheet.metadata["object_records"]
    assert first["upright_crop_xyxy"][2:] == [120, 80]
    assert first["source_mask_pixels"]["reference"] == 77
    assert empty["all_stages_empty_or_missing"]
    assert empty["source_mask_pixels"] == {"reference": None, "native": 0}
    assert empty["upright_crop_xyxy"] == [0, 0, 120, 80]


def test_scene_obb_budget_does_not_drop_masks_and_is_order_invariant():
    frame = _frame()
    original = {key: mask.copy() for key, mask in frame.native_masks.items()}
    options = replace(_options(), max_obbs=1)
    first = render_scene_sheet(frame, "Общий результат", options)
    second = render_scene_sheet(
        replace(frame, native_masks=dict(reversed(list(frame.native_masks.items())))),
        "Общий результат", options,
    )
    assert first.metadata["obb_selected_ids"] == [9]
    assert first.metadata["obb_omitted_ids"] == [2, 30]
    assert first.metadata["visible_mask_ids"] == [2, 9]
    assert first.metadata["zero_projection_ids"] == [30]
    assert first.metadata["displayed_union_pixels"] == sum(int(m.sum()) for m in original.values())
    assert np.array_equal(first.image, second.image)
    for key, before in original.items():
        assert np.array_equal(before, frame.native_masks[key])


def test_reference_comparison_uses_common_union_style_without_old_owner_map():
    frame = _frame()
    owner = np.full(frame.rgb.shape[:2], -1, np.int32)
    for object_id, mask in frame.native_masks.items():
        owner[mask] = object_id
    frame = replace(frame, instance_ids=owner, reference_masks=frame.native_masks)
    sheet = render_scene_sheet(frame, "Сравнение", replace(_options(), max_obbs=0))
    assert sheet.metadata["instance_display"] == "single_color_union"
    assert sheet.metadata["reference_projected_mask_pixels"] == sheet.metadata["projected_mask_pixels"]
    assert sheet.image.width == 3 * 400 + 2 * 24 + 2 * 30


@pytest.mark.parametrize("bad", [
    {"native_masks": {1: np.ones((5, 5), bool)}},
    {"native_masks": {1: np.ones((80, 120), np.uint8)}},
    {"rotation_degrees_ccw": np.nan},
    {"instance_ids": np.full((80, 120), 987, np.int32)},
])
def test_invalid_alignment_and_arbitrary_owner_are_rejected(bad):
    with pytest.raises(ValueError):
        render_scene_sheet(replace(_frame(), **bad), "Bad", _options())


def test_owner_cannot_claim_pixels_outside_actual_projection():
    frame = _frame()
    owner = np.full(frame.rgb.shape[:2], -1, np.int32)
    owner[0, 0] = 9
    with pytest.raises(ValueError, match="beyond"):
        render_scene_sheet(replace(frame, instance_ids=owner), "Bad", _options())


def test_unknown_objects_are_not_silently_ignored():
    with pytest.raises(ValueError, match="absent"):
        render_object_sheet(_frame(), [999], "Bad", _options())


def test_owner_cannot_hide_supplied_masks_as_background():
    frame = _frame()
    owner = np.full(frame.rgb.shape[:2], -1, np.int32)
    owner[frame.native_masks[9]] = 9
    with pytest.raises(ValueError, match="union"):
        render_scene_sheet(replace(frame, instance_ids=owner), "Bad", _options())


def test_original_photo_changes_only_photographic_column_without_reframing_masks():
    frame = replace(_frame(), rotation_degrees_ccw=31)
    yy, xx = np.indices((160, 240))
    detailed = np.repeat((((xx + yy) % 2) * 255).astype(np.uint8)[..., None], 3, axis=2)
    options = _options()
    before = render_object_sheet(frame, [9], "Same crop", options)
    after = render_object_sheet(replace(frame, object_photo_rgb=detailed), [9], "Same crop", options)
    assert before.image.size == after.image.size
    a, b = before.metadata["object_records"][0], after.metadata["object_records"][0]
    assert a["upright_crop_xyxy"] == b["upright_crop_xyxy"]
    assert a["source_mask_pixels"] == b["source_mask_pixels"]
    assert after.metadata["object_photo_source_hw"] == [160, 240]
    assert after.metadata["object_mask_grid_hw"] == [80, 120]
    changed = np.any(np.asarray(before.image) != np.asarray(after.image), axis=2)
    _, changed_x = np.nonzero(changed)
    assert len(changed_x) and changed_x.min() >= 24
    assert changed_x.max() < 24 + options.object_panel_width
    # Optional original photography must not change scene overlays at all.
    scene_a = render_scene_sheet(frame, "Scene", options)
    scene_b = render_scene_sheet(replace(frame, object_photo_rgb=detailed), "Scene", options)
    assert np.array_equal(scene_a.image, scene_b.image)


def test_original_photo_crop_matches_expanded_rotation_centres():
    from PIL import Image
    from farm_runtime.quality.scene_review import _original_photo_crop
    # Noninteger expanded sizes must use centres, not multiply canvas edges.
    photo = Image.new("RGB", (201, 149), (80, 90, 100))
    _, extent = _original_photo_crop(photo, (74, 100), 2, (20, 10, 60, 50), (200, 200))
    assert extent == [40.5, 20.5, 120.5, 100.5]


def test_original_photo_rejects_different_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        render_object_sheet(replace(_frame(), object_photo_rgb=np.zeros((160, 160, 3), np.uint8)),
                            [9], "Wrong field of view", _options())
