"""Admission response and independent-view selection contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality.proposal_admission import select_views, validate_views


def _answer(image_id):
    return {
        "image_id": image_id,
        "scope": "single_whole_object",
        "mask_fit": "fits_target",
        "confidence": "high",
        "reason": "The selected enclosure includes its fitted doors.",
    }


def _response(*image_ids):
    return {"views": [_answer(image_id) for image_id in image_ids]}


def test_validate_views_preserves_answers_and_binds_actual_image_ids():
    response = _response(20, 10)
    assert validate_views(json.dumps(response), [10, 20]) == response


@pytest.mark.parametrize("value", [
    None,
    [],
    42,
    {},
    {"views": {}},
    {"views": "10,20"},
    {"views": [_answer(10)]},
    {"views": [_answer(10), _answer(20), _answer(30)]},
    {"views": [_answer(10), _answer(20)], "accepted": True},
])
def test_validate_views_requires_exact_envelope_and_one_answer_per_view(value):
    with pytest.raises(ValueError):
        validate_views(json.dumps(value), [10, 20])


@pytest.mark.parametrize("change", [
    lambda row: row.update(label="cabinet"),
    lambda row: row.update(physical_timestamp="invented"),
    lambda row: row.update(admitted=True),
    lambda row: row.pop("mask_fit"),
    lambda row: row.pop("reason"),
    lambda row: row.update(scope="equipment"),
    lambda row: row.update(mask_fit="probably_fits"),
    lambda row: row.update(confidence=True),
    lambda row: row.update(reason=" "),
])
def test_validate_views_validates_exact_bounded_fields_for_each_answer(change):
    response = _response(10, 20)
    change(response["views"][1])
    with pytest.raises(ValueError):
        validate_views(json.dumps(response), [10, 20])


@pytest.mark.parametrize("image_ids", [
    [10, 999],
    [10, 10],
    [True, 20],
    [10.0, 20],
    ["010", 20],
    [None, 20],
])
def test_validate_views_rejects_unknown_duplicate_or_noninteger_ids(image_ids):
    with pytest.raises(ValueError):
        validate_views(json.dumps(_response(*image_ids)), [10, 20])


def test_boolean_id_cannot_alias_a_real_integer_source_id():
    with pytest.raises(ValueError):
        validate_views(json.dumps(_response(True, 20)), [1, 20])


@pytest.mark.parametrize("raw", [
    "not JSON",
    'RESULT: {"views":[]}',
    '{"views":[null,null]}',
    '{"views":[[],[]]}',
])
def test_validate_views_rejects_malformed_json_and_nonobject_rows(raw):
    with pytest.raises(ValueError):
        validate_views(raw, [10, 20])


@pytest.mark.parametrize("duplicate", ["views", "scope", "image_id"])
def test_duplicate_json_fields_cannot_overwrite_admission_evidence(duplicate):
    raw = json.dumps(_response(10, 20))
    if duplicate == "views":
        raw = raw.replace('{"views":', '{"views": [], "views":', 1)
    elif duplicate == "scope":
        raw = raw.replace('"scope":', '"scope": "structural_surface", "scope":', 1)
    else:
        raw = raw.replace('"image_id":', '"image_id": 999, "image_id":', 1)
    with pytest.raises(ValueError):
        validate_views(raw, [10, 20])


def _selection_case(rows, *, build=None, object_build=None, heldout=()):
    """rows are (source image ID, physical timestamp, foreground pixel count)."""
    frames = {
        image_id: SimpleNamespace(image_id=image_id, physical_timestamp=timestamp)
        for image_id, timestamp, _ in rows
    }
    object_id = 73
    obj = SimpleNamespace(
        object_id=object_id,
        observations=[SimpleNamespace(image_id=image_id) for image_id, _, _ in rows],
    )
    run = SimpleNamespace(frame=frames.__getitem__)
    if build is None:
        build = sorted({timestamp for _, timestamp, _ in rows} - set(heldout))
    if object_build is None:
        object_build = list(build)
    split = {
        "build_timestamps": list(build),
        "heldout_timestamps": list(heldout),
        "objects": [{"object_id": object_id, "build_timestamps": list(object_build)}],
    }
    masks = {}
    for image_id, _, pixels in rows:
        raw = np.zeros((8, 8), dtype=bool)
        raw.flat[:pixels] = True
        masks[image_id] = raw
    calls = []

    def mask(oid, image_id):
        assert oid == object_id
        calls.append(image_id)
        return masks[image_id], {"source_image_id": image_id}

    return run, split, obj, mask, calls


def test_select_views_uses_largest_camera_mask_once_per_physical_time():
    run, split, obj, mask, calls = _selection_case([
        (10, "time-a", 9), (11, "time-a", 30),
        (12, "time-a", 24), (20, "time-b", 18),
    ])
    assert select_views(run, split, obj, mask) == [11, 20]
    assert set(calls) == {10, 11, 12, 20}


def test_select_views_never_reads_heldout_or_object_disallowed_masks():
    run, split, obj, mask, calls = _selection_case(
        [(10, "time-a", 8), (20, "time-b", 7),
         (30, "heldout", 64), (40, "other-object-only", 63),
         (50, "not-in-global-build", 62)],
        build=["time-a", "time-b", "other-object-only"],
        object_build=["time-a", "time-b", "heldout", "not-in-global-build"],
        heldout=["heldout"],
    )
    assert select_views(run, split, obj, mask) == [10, 20]
    assert calls == [10, 20]


def test_empty_masks_do_not_supply_independent_confirmation():
    run, split, obj, mask, _ = _selection_case([
        (10, "time-a", 0), (11, "time-a", 0), (20, "time-b", 5),
    ])
    assert select_views(run, split, obj, mask) == [20]


def test_no_visible_candidate_returns_no_views():
    run, split, obj, mask, _ = _selection_case([
        (10, "time-a", 0), (20, "time-b", 0),
    ])
    assert select_views(run, split, obj, mask) == []


def test_many_cameras_at_one_timestamp_cannot_fill_second_slot():
    run, split, obj, mask, _ = _selection_case([
        (10, "time-a", 8), (11, "time-a", 11), (12, "time-a", 9),
    ])
    assert select_views(run, split, obj, mask) == [11]


def test_selection_is_bounded_to_two_largest_independent_timestamps():
    run, split, obj, mask, _ = _selection_case([
        (10, "time-a", 8), (20, "time-b", 30),
        (30, "time-c", 16), (40, "time-d", 24), (41, "time-d", 22),
    ])
    assert select_views(run, split, obj, mask) == [20, 40]


def test_view_selection_ties_are_deterministic_under_observation_reordering():
    run, split, obj, mask, _ = _selection_case([
        (19, "time-a", 10), (11, "time-a", 10),
        (21, "time-b", 10), (31, "time-c", 10),
    ])
    assert select_views(run, split, obj, mask) == [11, 21]
    obj.observations.reverse()
    assert select_views(run, split, obj, mask) == [11, 21]


def test_overlapping_build_and_heldout_splits_raise_before_mask_access():
    run, split, obj, mask, calls = _selection_case(
        [(10, "time-a", 8), (20, "time-b", 7)],
        build=["time-a", "time-b", "conflict"],
        object_build=["time-a", "time-b"],
        heldout=["conflict"],
    )
    with pytest.raises(ValueError, match="overlap"):
        select_views(run, split, obj, mask)
    assert calls == []


def test_validate_views_maps_only_exact_canonical_string_identifiers():
    import json
    row = {"scope": "single_whole_object", "mask_fit": "fits_target", "confidence": "high", "reason": "Visible independent object"}
    raw = json.dumps({"views": [dict(row, image_id="10"), dict(row, image_id="20")]})
    parsed = validate_views(raw, [10, 20])
    assert [item["image_id"] for item in parsed["views"]] == [10, 20]


@pytest.fixture
def review_case(tmp_path, monkeypatch):
    """Exercise runner IO/cache with deterministic images and no model process."""
    from PIL import Image
    from farm_runtime.quality import proposal_admission as module
    from farm_runtime.quality_baseline import describe_file, write_json

    source = tmp_path / "source.png"
    pixels = np.random.default_rng(19).integers(0, 256, (48, 72, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    geometry = tmp_path / "geometry.json"
    ply = tmp_path / "scene.ply"
    prepared = tmp_path / "prepared.json"
    mask_path = tmp_path / "mask.npz"
    for path in (geometry, ply, prepared, mask_path):
        path.write_text("fixture")
    native = tmp_path / "native.json"
    write_json(native, dict(input=describe_file(prepared), source_ply=describe_file(ply)))
    catalog = tmp_path / "catalog.json"
    write_json(catalog, dict(
        schema="farm.quality-scene-catalog.v1", closed_test_opened=False,
        native_output=describe_file(native), source_ply=describe_file(ply),
        object_id_namespace=dict(geometry=describe_file(geometry)),
        objects=[dict(object_id=73, native_gaussians=19)],
    ))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "fixture.safetensors").write_text("no model is loaded in this test")
    frames = {
        fid: SimpleNamespace(image_id=fid, physical_timestamp=timestamp, depth_size=(8, 8))
        for fid, timestamp in ((10, "a"), (20, "b"))
    }
    obj = SimpleNamespace(object_id=73, observations=[SimpleNamespace(image_id=fid) for fid in frames])
    run = SimpleNamespace(scene_id="fixture", objects=[obj], frame=frames.__getitem__)
    split = dict(build_timestamps=["a", "b"], heldout_timestamps=[],
                 objects=[dict(object_id=73, build_timestamps=["a", "b"])])
    inputs = dict(source_geometry=describe_file(geometry),
                  frames=[dict(image_id=fid) for fid in frames])
    monkeypatch.setattr(module, "load_prepared", lambda path: (run, split, inputs))
    monkeypatch.setattr(module, "resolve_mask_path", lambda run, obs: mask_path)
    monkeypatch.setattr(module, "load_mask_pair", lambda path, shape: (
        np.ones(shape, bool), np.ones(shape, bool), describe_file(path)["sha256"],
    ))
    monkeypatch.setattr(module, "load_observation_exclusion", lambda run, frame: (None, None))
    monkeypatch.setattr(module, "make_sheet", lambda *args: (
        Image.fromarray(pixels.copy()), dict(source=describe_file(source)),
    ))

    class Reviewer:
        model_path = model.resolve()

        def __init__(self):
            self.images = []

        def ask(self, prompt, images, image_ids, validator, **kwargs):
            self.images.append([np.asarray(image).copy() for image in images])
            raw = json.dumps(_response(*image_ids))
            return dict(raw=raw, parsed=validator(raw, image_ids), validation_error=None, seconds=0)

    reviewer = Reviewer()
    return SimpleNamespace(module=module, catalog=catalog, model=model, reviewer=reviewer,
                           output=tmp_path / "review", pixels=pixels)


def test_runner_feeds_exact_saved_jpeg_pixels_and_reuses_bound_response(review_case):
    from pathlib import Path
    from PIL import Image

    case = review_case
    first = case.module.run_review(case.catalog, case.model, case.output, reviewer=case.reviewer)
    assert len(case.reviewer.images) == 1
    record = first["records"][0]
    for actual, source in zip(case.reviewer.images[0], record["sources"]):
        with Image.open(source["image"]["path"]) as stored:
            np.testing.assert_array_equal(actual, np.asarray(stored.convert("RGB")))
        assert not np.array_equal(actual, case.pixels), "fixture must expose the JPEG vs source difference"
    assert Path(first["admission_code"]["path"]).name == "object_admission.py"
    second = case.module.run_review(case.catalog, case.model, case.output, reviewer=case.reviewer, resume=True)
    assert len(case.reviewer.images) == 1
    assert second["records"][0]["response_reused"]
    assert first["records"][0]["decision"] == second["records"][0]["decision"]


def test_changed_admission_logic_invalidates_resume_before_old_decision_is_overwritten(review_case, monkeypatch):
    case = review_case
    case.module.run_review(case.catalog, case.model, case.output, reviewer=case.reviewer)
    record_path = case.output / "object_000073.json"
    original = record_path.read_bytes()
    original_describe = case.module.describe_file

    def changed_admission(path):
        descriptor = original_describe(path)
        if path.name == "object_admission.py":
            descriptor["sha256"] = "0" * 64
        return descriptor

    monkeypatch.setattr(case.module, "describe_file", changed_admission)
    with pytest.raises(ValueError, match="resume input/model/prompt binding changed"):
        case.module.run_review(case.catalog, case.model, case.output, reviewer=case.reviewer, resume=True)
    assert record_path.read_bytes() == original
    assert len(case.reviewer.images) == 1


@pytest.mark.parametrize("missing", [False, True])
def test_injected_reviewer_cannot_silently_claim_another_model(review_case, missing):
    case = review_case
    if missing:
        case.reviewer.model_path = None
    else:
        case.reviewer.model_path = case.model.parent / "different-model"
    with pytest.raises(ValueError, match="reviewer model path differs"):
        case.module.run_review(case.catalog, case.model, case.output, reviewer=case.reviewer)
    assert not case.output.exists()
    assert case.reviewer.images == []


@pytest.fixture
def sheet_case(tmp_path):
    from PIL import Image
    from farm_runtime.quality_baseline import describe_file

    mask = np.zeros((20, 30), bool)
    mask[4:16, 5:22] = True
    mask[8:12, 12:17] = False
    mask[2:4, 25:28] = True
    source_selected = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)
    rgb = np.full((80, 120, 3), (255, 0, 0), np.uint8)
    rgb[source_selected] = (0, 255, 0)
    path = tmp_path / "authoritative.png"
    Image.fromarray(rgb).save(path)
    return SimpleNamespace(
        path=path, descriptor=describe_file(path), rgb=rgb, mask=mask,
        frame=SimpleNamespace(image_id=7),
    )


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_isolated_sheet_preserves_rotation_holes_islands_and_exact_mask_placement(sheet_case, turns):
    from PIL import Image
    from scipy.ndimage import binary_fill_holes, label
    from farm_runtime.quality.proposal_admission import make_admission_sheet

    case = sheet_case
    original_mask = case.mask.copy()
    original_file = case.path.read_bytes()
    metadata = dict(source=case.descriptor, applied_quarter_turns=turns)
    mask_reader = lambda oid, fid: (case.mask, {})
    context, _ = make_admission_sheet(case.frame, metadata, 73, mask_reader)
    isolated, record = make_admission_sheet(case.frame, metadata, 73, mask_reader, image_style="isolated")
    np.testing.assert_array_equal(np.asarray(context)[:, :600], np.asarray(isolated)[:, :600])
    np.testing.assert_array_equal(case.mask, original_mask)
    assert case.path.read_bytes() == original_file

    rotated = np.rot90(case.mask, turns)
    info = record["isolated_view"]
    x0, y0, x1, y1 = record["crop_xyxy"]
    assert int(rotated[y0:y1, x0:x1].sum()) == int(rotated.sum()), "tight crop must retain every island"
    assert info["source_rgb_crop_xyxy"] == [4 * value for value in (x0, y0, x1, y1)]
    a, b, c, d = info["display_xyxy"]
    tile = np.asarray(isolated)[b:d, a:c]
    expected = Image.fromarray((rotated[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(
        (c - a, d - b), Image.Resampling.NEAREST,
    )
    selected = np.asarray(expected) > 0
    np.testing.assert_array_equal(tile[~selected], np.full((int((~selected).sum()), 3), 128, np.uint8))
    np.testing.assert_array_equal(tile[selected], np.tile((0, 255, 0), (int(selected.sum()), 1)))
    assert label(selected)[1] == 2, "the disconnected observed island must stay separate"
    assert np.count_nonzero(binary_fill_holes(selected) & ~selected) > 0, "a mask hole must remain matte"
    assert info["selected_display_pixels"] == int(selected.sum())
    assert info["mask_modified"] is False
    assert info["matte_rgb"] == [128, 128, 128]
    assert 600 <= a < c <= 1200 and 36 <= b < d <= 430
    assert c - a <= 580 and d - b <= 380
    assert abs((c - a) * (y1 - y0) - (d - b) * (x1 - x0)) <= max(x1 - x0, y1 - y0)


def test_context_sheet_remains_pixel_identical_to_previous_witness_display(sheet_case):
    from PIL import ImageDraw
    from farm_runtime.quality.proposal_admission import make_admission_sheet
    from farm_runtime.quality.scope_family_review import make_sheet, _font

    case = sheet_case
    metadata = dict(source=case.descriptor, applied_quarter_turns=1)
    masks = lambda oid, fid: (case.mask, {})
    original, _ = make_sheet(case.frame, metadata, 73, [], masks)
    draw = ImageDraw.Draw(original)
    draw.rectangle((0, 0, original.width, 35), fill="#111827")
    draw.text((12, 8), "FULL VIEW - selected region", fill="white", font=_font(20))
    draw.text((612, 8), "SELECTED PIXELS - same region", fill=(251, 146, 60), font=_font(20))
    actual, record = make_admission_sheet(case.frame, metadata, 73, masks)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(original))
    assert record["image_style"] == "context" and "isolated_view" not in record


@pytest.mark.parametrize("shape,location", [((5, 25), (0, 0)), ((25, 5), (24, 4)), ((7, 7), (3, 3))])
def test_isolated_crop_handles_single_pixel_and_image_border_without_inventing_extent(shape, location):
    from PIL import Image
    from farm_runtime.quality.proposal_admission import _isolated_selected_crop

    mask = np.zeros(shape, bool)
    mask[location] = True
    rgb = np.full((*shape, 3), (255, 0, 0), np.uint8)
    rgb[mask] = (0, 255, 0)
    output, metadata = _isolated_selected_crop(Image.fromarray(rgb), mask)
    pixels = np.asarray(output)
    colors = set(map(tuple, pixels.reshape(-1, 3).tolist()))
    assert colors == {(0, 255, 0), (128, 128, 128)}
    x0, y0, x1, y1 = metadata["crop_xyxy"]
    assert 0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]
    assert mask.sum() == 1


@pytest.mark.parametrize("mask", [np.zeros((5, 7), bool), np.ones((5, 7), np.float32), np.ones(5, bool)])
def test_isolated_crop_requires_real_nonempty_boolean_selection(mask):
    from PIL import Image
    from farm_runtime.quality.proposal_admission import _isolated_selected_crop

    with pytest.raises(ValueError, match="boolean mask"):
        _isolated_selected_crop(Image.new("RGB", (7, 5)), mask)


def test_image_style_has_explicit_cache_binding_even_if_prompt_is_unchanged(review_case, monkeypatch):
    case = review_case
    monkeypatch.setattr(case.module, "review_prompt", lambda protocol, image_style: "fixed fixture prompt")
    first = case.module.run_review(case.catalog, case.model, case.output, reviewer=case.reviewer)
    assert first["image_style"] == "context"
    assert first["image_policy"]["right"] == "context"
    original = (case.output / "object_000073.json").read_bytes()
    with pytest.raises(ValueError, match="resume input/model/prompt binding changed"):
        case.module.run_review(case.catalog, case.model, case.output, reviewer=case.reviewer,
                               resume=True, image_style="isolated")
    assert (case.output / "object_000073.json").read_bytes() == original
    assert len(case.reviewer.images) == 1


def test_isolated_style_only_appends_encoding_context_to_the_frozen_prompt():
    from farm_runtime.quality.proposal_admission import review_prompt, PROMPT
    from farm_runtime.quality.object_admission import PROMPT_V2

    assert review_prompt("scope_v1") == PROMPT
    base = review_prompt("witness_v2")
    assert base.startswith(PROMPT_V2)
    isolated = review_prompt("witness_v2", "isolated")
    assert isolated.startswith(base)
    suffix = isolated[len(base):]
    assert "neutral gray matte" in suffix and "not selected pixels" in suffix


def test_unknown_image_style_fails_before_review_outputs(review_case):
    case = review_case
    with pytest.raises(ValueError, match="unknown admission image style"):
        case.module.run_review(case.catalog, case.model, case.output,
                               reviewer=case.reviewer, image_style="invented")
    assert not case.output.exists()


@pytest.mark.parametrize("style", ["context", "isolated"])
def test_cli_passes_image_style_without_changing_protocol(monkeypatch, style):
    from farm_runtime.quality import proposal_admission as module

    calls = []
    monkeypatch.setattr(module, "run_review", lambda *args, **kwargs: calls.append(kwargs))
    assert module.main(["--catalog", "catalog.json", "--model", "model", "--output", "output",
                        "--protocol", "witness_v2", "--image-style", style]) == 0
    assert calls[0]["image_style"] == style and calls[0]["protocol"] == "witness_v2"
