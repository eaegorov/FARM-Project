import copy
import json
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pytest

from farm_runtime.angular_discovery import rotate_image, upright_quarter_turns
from farm_runtime.segmentation_refinement import (
    interior_points,
    prompt_variants,
    restore_crop_mask,
    select_mask_proposal,
)
from farm_runtime.semantic_refinement import select_review_views, validate_review
from farm_runtime.quality.mask_refinement import apply_mask_refinement
from farm_runtime.quality_baseline import describe_file


def test_numpy_gravity_is_not_silently_unavailable():
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    a = upright_quarter_turns(r, [0, -1, 0])
    b = upright_quarter_turns(r, np.array([0, -1, 0]))
    assert a == b
    assert b["applied_quarter_turns_ccw"] == 1


def test_crop_gravity_uses_perspective_ray():
    k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
    # At the principal point this up vector has no screen direction; at a
    # side-wall crop it projects sideways and requires a quarter turn.
    result = upright_quarter_turns(
        np.eye(3), [0, 0, -1], image_point=[90, 50], intrinsics=k
    )
    assert result["applied_quarter_turns_ccw"] == 1
    assert result["status"] == "applied"
    assert (
        upright_quarter_turns(
            np.eye(3), [0, 0, -1], image_point=[50, 50], intrinsics=k
        )["status"]
        == "unavailable"
    )


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_crop_roundtrip_preserves_asymmetric_pixels(turns):
    original = np.zeros((7, 11), bool)
    original[1:5, 2:8] = True
    original[1, 2] = False
    row = dict(crop_grid_xyxy=[4, 3, 15, 10], grid_shape_hw=[20, 30], turns=turns)
    box, restored = restore_crop_mask(rotate_image(original, turns), row)
    assert np.array_equal(original, restored)
    assert box == [4, 3, 15, 10]


def test_prompt_points_do_not_use_empty_or_unobserved_foreground():
    assert prompt_variants(np.zeros((20, 30))) == []
    probability = np.zeros((20, 30))
    probability[5:15, 8:23] = 1
    variants = prompt_variants(probability)
    for variant in variants:
        for (x, y), label in zip(variant["points"], variant["labels"]):
            assert bool(probability[int(y), int(x)]) == bool(label)
    assert interior_points(np.zeros((8, 8))) == []
    with pytest.raises(ValueError):
        prompt_variants(np.full((10, 10), np.nan))


def test_vlm_abstention_keeps_original_and_cannot_override_core_guard():
    row = dict(
        display_candidates=[dict(key="baseline")]
        + [
            dict(key=f"c{i}", geometry_eligible=True, predicted_iou=0.97)
            for i in range(3)
        ]
    )
    arrays = {f"c{i}": np.ones((20, 20), bool) for i in range(3)}
    assert select_mask_proposal(row, arrays)[0] == "c0"
    assert select_mask_proposal(row, arrays, 0, True)[0] is None
    assert select_mask_proposal(row, arrays, -1, True)[0] is None
    row["display_candidates"][2]["geometry_eligible"] = False
    assert select_mask_proposal(row, arrays, 2, True)[0] is None
    arrays["c2"][:10] = False
    assert select_mask_proposal(row, arrays)[0] is None


def test_review_budget_groups_physical_timestamps_and_excludes_clipped():
    rows = [
        dict(
            image_id=i,
            timestamp=str(ts),
            foreground_pixels=area,
            clipped=clipped,
            source_image={"path": f"/cam00_{ts}_center.png"},
        )
        for i, ts, area, clipped in [
            (0, 1, 1000, True),
            (1, 2, 100, False),
            (2, 2, 99, False),
            (3, 3, 80, False),
        ]
    ]
    assert [r["image_id"] for r in select_review_views(rows, 2)] == [1, 3]


def test_review_parser_rejects_missing_duplicate_or_invented_views():
    result = dict(
        label="box",
        caption="A rectangular box.",
        include=[],
        exclude=[],
        uncertainty="unknown function",
        views=[
            dict(image_id=7, choice=1, reason="boundary"),
            dict(image_id=8, choice=0, reason="preserve"),
        ],
    )
    assert validate_review(json.dumps(result), [7, 8])["label"] == "box"
    result["views"][1]["image_id"] = 7
    with pytest.raises(ValueError):
        validate_review(json.dumps(result), [7, 8])
    result["views"][1]["image_id"] = 8
    result["views"][1]["choice"] = True
    with pytest.raises(ValueError):
        validate_review(json.dumps(result), [7, 8])


@dataclass
class DummyRun:
    success_sha256: str
    meters_per_scene_unit: float
    objects: tuple
    frames: tuple
    mask_overrides: object = None

    def frame(self, index):
        return self.frames[index]


@dataclass
class DummyObject:
    object_id: int
    observations: tuple


def test_refinement_adapter_binds_hashes_and_never_touches_heldout(tmp_path):
    split_path, camera_path, mask_path = [
        tmp_path / n for n in ("split.json", "cameras.json", "mask.npz")
    ]
    split_path.write_text(
        json.dumps(dict(build_timestamps=["10"], heldout_timestamps=["20"]))
    )
    camera_path.write_text("{}")
    mask_path.write_bytes(b"test mask bytes")
    run = DummyRun(
        "abc",
        1.0,
        (DummyObject(7, (SimpleNamespace(image_id=0), SimpleNamespace(image_id=1))),),
        (
            SimpleNamespace(physical_timestamp="10"),
            SimpleNamespace(physical_timestamp="20"),
        ),
    )
    manifest = dict(
        schema="farm.build-mask-refinement.v1",
        run_success_sha256="abc",
        meters_per_scene_unit=1.0,
        split=describe_file(split_path),
        cameras=describe_file(camera_path),
        masks=[
            dict(
                object_id=7,
                image_id=0,
                physical_timestamp_ns=10,
                mask=describe_file(mask_path),
            )
        ],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    refined = apply_mask_refinement(run, path, split_path, camera_path)
    assert set(refined.mask_overrides) == {(7, 0)}
    assert refined.objects[0].observations[1] is run.objects[0].observations[1]
    manifest["masks"][0].update(image_id=1, physical_timestamp_ns=20)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="only replace build"):
        apply_mask_refinement(run, path, split_path, camera_path)
    manifest["masks"][0].update(image_id=0, physical_timestamp_ns=10)
    path.write_text(json.dumps(manifest))
    mask_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_mask_refinement(run, path, split_path, camera_path)


def test_semantics_separates_roles_and_requires_evidence_for_function():
    from farm_runtime.semantic_refinement import validate_semantics

    result = dict(
        label="enclosure",
        caption="A black enclosure.",
        functional_identity={"label": None, "direct_visual_evidence": ""},
        integral_parts=["body"],
        external_connections=["cable"],
        supporting_objects=["pallet"],
        contained_objects=[],
        uncertainty=["function unknown"],
        observed_image_ids=[7, 8],
    )
    assert (
        validate_semantics(json.dumps(result), [7, 8])["functional_identity"]["label"]
        is None
    )
    result["integral_parts"].append("cable")
    with pytest.raises(ValueError, match="conflicting ownership"):
        validate_semantics(json.dumps(result), [7, 8])
    result["integral_parts"].remove("cable")
    result["functional_identity"]["label"] = "server"
    with pytest.raises(ValueError, match="direct visual evidence"):
        validate_semantics(json.dumps(result), [7, 8])


def test_scene_vocabulary_retains_separate_roles_and_validates_image_ids():
    from farm_runtime.quality.scene_vocabulary import validate_vocabulary

    value = {
        "views": [
            {
                "image_id": 1,
                "objects": ["Cabinet", "cabinet"],
                "structures": ["wall"],
                "transient": ["person"],
            }
        ]
    }
    result = validate_vocabulary(json.dumps(value), [1])
    assert result["views"][0]["objects"] == ["cabinet"]
    with pytest.raises(ValueError, match="image IDs"):
        validate_vocabulary(json.dumps(value), [2])
