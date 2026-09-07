import json

import numpy as np
from PIL import Image
import pytest

from farm_runtime.quality import scope_completion as completion
from farm_runtime.quality_baseline import describe_file


def response(**changes):
    result = dict(
        scope_complete=False,
        box=[650, 500, 950, 950],
        add_foreground=[[800, 800]],
        background=[],
        confidence="high",
        reason="Visible lower integral part omitted.",
    )
    result.update(changes)
    return result


def test_missing_part_box_keeps_existing_object_and_rejects_background_inside_it():
    current = np.zeros((100, 200), bool)
    current[10:60, 20:130] = True
    result = response(background=[[200, 300], [50, 900]])
    prompts = completion.completion_prompts(result, current)
    assert len(prompts) == 2
    assert prompts[0]["box"] == [20, 10, 190, 95]
    assert prompts[0]["points"] == []
    guided = prompts[1]
    assert [40, 30] not in guided["points"]
    assert guided["points"][-1] == [10, 90] and guided["labels"][-1] == 0
    assert [160, 80] in guided["points"]


@pytest.mark.parametrize(
    "changes",
    [
        dict(scope_complete=True),
        dict(confidence="medium"),
        dict(box=None),
        dict(add_foreground=[]),
        dict(add_foreground=[[500, 500]]),
    ],
)
def test_uncertain_complete_or_interior_guidance_does_not_run_sam(changes):
    current = np.zeros((100, 100), bool)
    current[20:70, 20:70] = True
    assert completion.completion_prompts(response(**changes), current) == []


@pytest.mark.parametrize(
    "changes",
    [
        dict(scope_complete=1),
        dict(box=[900, 0, 100, 1000]),
        dict(box=[False, 0, 900, 1000]),
        dict(box=[0, 0, float("nan"), 1000]),
        dict(add_foreground=[[500, float("inf")]]),
        dict(add_foreground=[[500, 500]] * 4),
        dict(background=[123]),
        dict(confidence="certain"),
    ],
)
def test_malformed_grounding_is_rejected(changes):
    with pytest.raises(ValueError):
        completion.validate_response(json.dumps(response(**changes)))


def fixture(tmp_path):
    def write(name, doc):
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        return path

    audit = write("audit.json", {})
    source = dict(
        name="image.png",
        timestamp="t1",
        grid_shape_hw=[10, 10],
        applied_quarter_turns=0,
        source_image={"path": "not-decoded", "sha256": "source"},
    )
    base = write(
        "base.json",
        dict(
            test_opened=False,
            observations=[dict(source, detections=[dict(label="person")])],
        ),
    )
    other = write(
        "other.json",
        dict(
            test_opened=False,
            observations=[dict(source, detections=[dict(label="other")] * 2)],
        ),
    )
    masks = tmp_path / "masks"
    masks.mkdir()
    file = masks / "image.npz"
    np.savez_compressed(file, a=np.ones((6, 6)), b=np.ones((6, 6)))
    windows = [[0, 0, 6, 6], [4, 4, 10, 10]]
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (12, 12), "gray").save(crop_path)
    detections = [
        dict(
            label="object",
            score=0.9,
            source_group_id=gid,
            grid_window_xyxy=windows[i],
            logit_key=key,
        )
        for i, (gid, key) in enumerate([(11, "a"), (22, "b")])
    ]
    tracker = write(
        "tracker.json",
        dict(
            schema="farm.projected-surface-tracker.v1",
            test_opened=False,
            release_eligible=False,
            source_audit=describe_file(audit),
            source_proposals=describe_file(base),
            observations=[
                dict(source, detections=detections, mask_artifact=describe_file(file))
            ],
            crops=[
                dict(
                    group_id=gid,
                    name=source["name"],
                    source_image=source["source_image"],
                    turns=0,
                    crop_grid_xyxy=windows[i],
                    crop=describe_file(crop_path),
                )
                for i, gid in enumerate([11, 22])
            ],
        ),
    )
    validation = write(
        "validation.json",
        dict(
            schema="farm.surface-validation.v1",
            closed_test_opened=False,
            release_eligible=False,
            source_audit=describe_file(audit),
            source_proposals=describe_file(base),
            supplements=[describe_file(other), describe_file(tracker)],
            groups=[
                dict(
                    group_id=gid,
                    extra_matches=[
                        dict(
                            name=source["name"],
                            timestamp="t1",
                            selected_detection=i + 3,
                            decision="matched_static_surface",
                            candidates=[dict(detection_index=i + 3, eligible=True)],
                        )
                    ],
                )
                for i, gid in enumerate([11, 22])
            ],
        ),
    )
    return tracker, validation


def test_flattened_indices_keep_two_objects_in_same_frame_distinct(tmp_path):
    tracker, validation = fixture(tmp_path)
    requests, skipped = completion.validated_requests(tracker, validation)
    assert skipped == []
    assert [
        (r["record"]["group_id"], r["detection_index"], r["selected_detection"])
        for r in requests
    ] == [(11, 0, 3), (22, 1, 4)]


def test_changed_bound_tracker_is_rejected_before_any_image_decode(tmp_path):
    tracker, validation = fixture(tmp_path)
    tracker.write_text(tracker.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        completion.validated_requests(tracker, validation)


def test_wrong_selected_object_is_not_used_as_another_object_prompt(tmp_path):
    tracker, validation = fixture(tmp_path)
    doc = json.loads(validation.read_text())
    match = doc["groups"][1]["extra_matches"][0]
    match["selected_detection"] = 3
    match["candidates"] = [dict(detection_index=3, eligible=True)]
    validation.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="group mismatch"):
        completion.validated_requests(tracker, validation)


def test_budget_counts_distinct_timestamps_and_shares_calls_across_objects():
    rows = [
        dict(record=dict(group_id=g, name=f"{g}_{t}_{i}"), timestamp=t)
        for g in [1, 2]
        for i, t in enumerate(["t1", "t1", "t2", "t3"])
    ]
    chosen = completion.bounded_requests(rows, 4, 2)
    assert [(r["record"]["group_id"], r["timestamp"]) for r in chosen] == [
        (1, "t1"),
        (2, "t1"),
        (1, "t2"),
        (2, "t2"),
    ]


def test_unresolved_identity_skips_models_and_preserves_empty_output(tmp_path):
    tracker, validation = fixture(tmp_path)
    doc = json.loads(validation.read_text())
    for group in doc["groups"]:
        group["extra_matches"][0].update(
            selected_detection=None, decision="ambiguous_competing_scope"
        )
    validation.write_text(json.dumps(doc))
    output = tmp_path / "output"
    assert (
        completion.main(
            [
                "--tracker",
                str(tracker),
                "--validation",
                str(validation),
                "--model",
                str(tmp_path / "no-vlm"),
                "--sam-model",
                str(tmp_path / "no-sam"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["vlm_calls"] == manifest["sam_encoder_calls"] == 0
    assert manifest["model_config"] is None and manifest["sam_weights"] is None
    assert manifest["observations"] == [] and len(manifest["skipped"]) == 2


def test_cli_preserves_both_objects_in_one_frame_and_reuses_model_instances(
    tmp_path, monkeypatch
):
    import farm_runtime.semantic_refinement as semantic
    import farm_runtime.segmentation_refinement as segmentation
    from farm_runtime.quality.proposal_geometry import read_masks

    tracker, validation = fixture(tmp_path)
    doc = json.loads(tracker.read_text())
    mask = np.full((6, 6), -5.0, np.float32)
    mask[1:4, 1:4] = 5
    artifact = tmp_path / "masks" / "image.npz"
    np.savez_compressed(artifact, a=mask, b=mask)
    doc["observations"][0]["mask_artifact"] = describe_file(artifact)
    tracker.write_text(json.dumps(doc))
    valid = json.loads(validation.read_text())
    valid["supplements"][-1] = describe_file(tracker)
    validation.write_text(json.dumps(valid))
    model = tmp_path / "models"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture")
    calls = dict(vlm_load=0, sam_load=0, vlm=0, sam=0)

    class Reviewer:
        def __init__(self, _):
            calls["vlm_load"] += 1

        def ask(self, prompt, images, ids, validator, **kwargs):
            calls["vlm"] += 1
            raw = json.dumps(
                response(box=[0, 0, 1000, 1000], add_foreground=[[950, 950]])
            )
            return dict(raw=raw, parsed=validator(raw, ids), validation_error=None)

    class Refiner:
        def __init__(self, _):
            calls["sam_load"] += 1

        def predict(self, image, variants):
            calls["sam"] += 1
            logits = np.full((image.height, image.width), -5.0, np.float32)
            # Distinct regions reveal accidental overwrite between objects.
            start = 0 if calls["sam"] == 1 else image.height // 2
            logits[start : start + image.height // 2] = 5
            return [dict(logits=logits, predicted_iou=0.9, variant="vlm_box")]

    monkeypatch.setattr(semantic, "LocalObjectReviewer", Reviewer)
    monkeypatch.setattr(segmentation, "CachedSAMRefiner", Refiner)
    output = tmp_path / "out"
    assert (
        completion.main(
            [
                "--tracker",
                str(tracker),
                "--validation",
                str(validation),
                "--model",
                str(model),
                "--sam-model",
                str(model),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "manifest.json").read_text())
    assert calls == dict(vlm_load=1, sam_load=1, vlm=2, sam=2)
    assert len(result["observations"]) == 1
    row = result["observations"][0]
    assert [d["source_group_id"] for d in row["detections"]] == [11, 22]
    masks = read_masks(row, output)
    assert len(masks) == 2 and masks[0][:3, :6].all() and masks[1][7:, 4:].all()
    assert not masks[0][4:].any() and not masks[1][:6].any()


def test_ambiguous_proposals_choose_best_eligible_tracker_without_accepting_it(
    tmp_path,
):
    tracker, validation = fixture(tmp_path)
    doc = json.loads(validation.read_text())
    match = doc["groups"][0]["extra_matches"][0]
    match.update(
        selected_detection=None,
        decision="ambiguous_competing_scope",
        candidates=[
            dict(detection_index=3, eligible=True, score=0.9),
            dict(detection_index=0, eligible=True, score=0.8),
            dict(detection_index=1, eligible=False, score=1.0),
        ],
    )
    validation.write_text(json.dumps(doc))
    original, _ = completion.validated_requests(tracker, validation)
    assert [r["record"]["group_id"] for r in original] == [22]
    requests, _ = completion.validated_requests(
        tracker, validation, allow_ambiguous=True
    )
    assert requests[0]["selected_detection"] == 3
    assert requests[0]["ambiguous_seed"] is True
    assert "ambiguous_seed" not in requests[1]
    # The source validation still contains no selected mask for this group.
    assert (
        json.loads(validation.read_text())["groups"][0]["extra_matches"][0][
            "selected_detection"
        ]
        is None
    )
    match["candidates"][1]["score"] = 0.95
    validation.write_text(json.dumps(doc))
    requests, skipped = completion.validated_requests(
        tracker, validation, allow_ambiguous=True
    )
    assert [r["record"]["group_id"] for r in requests] == [22]
    assert skipped[0]["reason"] == "selected_other_source"
    match["decision"] = "no_unambiguous_surface_match"
    validation.write_text(json.dumps(doc))
    requests, _ = completion.validated_requests(
        tracker, validation, allow_ambiguous=True
    )
    assert [r["record"]["group_id"] for r in requests] == [22]


@pytest.mark.parametrize("turns", range(4))
def test_completion_context_preserves_raw_pixels_and_round_trips_grid(tmp_path, turns):
    # Nonsquare image, unequal source/grid scaling, clipped expansion at left/top.
    grid = np.zeros((12, 20), bool)
    grid[2:6, 1:7] = True
    rgb = np.arange(24 * 60 * 3, dtype=np.uint16).reshape(24, 60, 3).astype(np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(rgb).save(source)
    source_box = [3, 4, 21, 12]
    crop_path = tmp_path / "prior.png"
    Image.fromarray(completion.rotate_image(rgb[4:12, 3:21], turns)).save(crop_path)
    record = dict(
        crop=describe_file(crop_path),
        source_image=describe_file(source),
        crop_grid_xyxy=[1, 2, 7, 6],
        turns=turns,
    )
    prior, current, window, box = completion.completion_crop(record, grid)
    assert window == [1, 2, 7, 6] and box is None
    assert np.array_equal(np.asarray(prior), np.asarray(Image.open(crop_path)))
    crop, current, window, box = completion.completion_crop(record, grid, 0.5)
    assert window == [0, 0, 10, 9] and box == [0, 0, 30, 18]
    assert np.array_equal(
        np.asarray(crop), completion.rotate_image(rgb[:18, :30], turns)
    )
    tile = completion.restore_crop_logits(np.where(current, 20.0, -20.0), turns, window)
    restored = np.zeros_like(grid)
    x0, y0, x1, y1 = window
    restored[y0:y1, x0:x1] = tile > 0
    assert np.array_equal(restored, grid)
    assert record["crop_grid_xyxy"] == [1, 2, 7, 6]


@pytest.mark.parametrize("padding", [-0.1, 0.51, float("nan"), float("inf")])
def test_invalid_completion_context_rejected_before_image_decode(padding):
    with pytest.raises(ValueError, match="context padding"):
        completion.completion_crop({}, np.ones((10, 10), bool), padding)
