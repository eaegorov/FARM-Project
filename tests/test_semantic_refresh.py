import copy
import json
from pathlib import Path

import pytest

from farm_runtime.quality import refinement
from farm_runtime.quality.semantic_refresh import refresh_appearance
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.semantic_refinement import APPEARANCE_PROMPT


@pytest.fixture
def case(tmp_path, monkeypatch):
    def artifact(name, data):
        path = tmp_path / name
        path.write_text(data)
        return describe_file(path)

    geometry = artifact("geometry.json", "{}")
    old_native = artifact("old_native.json", '{"version": 1}')
    new_native = artifact("new_native.json", '{"version": 2}')
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    prompt = artifact("prompt.txt", APPEARANCE_PROMPT)
    rows = []
    for oid in (7, 9):
        for view in (1, 2):
            rows.append(
                dict(
                    object_id=oid,
                    image_id=view,
                    timestamp=str(view),
                    source_name=f"camera_{view}",
                    crop_source_xyxy=[0, 0, 20, 20],
                    turns=0,
                    clipped=False,
                    mask_source="geometry_original",
                    source_image=artifact(f"rgb_{oid}_{view}", "rgb"),
                    crop=artifact(f"crop_{oid}_{view}", "crop"),
                    masks=artifact(f"mask_{oid}_{view}", "mask"),
                    native_mask=artifact(f"native_mask_{oid}_{view}", "native"),
                )
            )
    old_evidence = dict(
        schema="farm.object-scope-evidence.v1",
        source_groups=geometry,
        source_native_input=old_native,
        groups=[dict(id=7), dict(id=9)],
        observations=rows,
        reserved_test_opened=False,
        release_eligible=False,
        detector_labels_in_images=False,
    )
    evidence_path = tmp_path / "old_evidence.json"
    write_json(evidence_path, old_evidence)
    objects = [
        dict(
            object_id=oid,
            raw=f"original-{oid}",
            parsed=dict(label="device"),
            seconds=3.0,
            input_tokens=123,
            output_tokens=42,
            validation_error=None,
            sheets=[artifact(f"sheet_{oid}", "previous canvas")],
        )
        for oid in (7, 9)
    ]
    appearance = dict(
        schema="farm.local-object-appearance.v1",
        objects=objects,
        proposals=describe_file(evidence_path),
        model_config=describe_file(model / "config.json"),
        prompt=prompt,
        physical_scope_assessed=False,
        scene_context_included=True,
        context_scale=None,
        masked_target_rgb=False,
        human_labels_in_prompt=False,
        detector_labels_in_prompt=False,
        reserved_test_opened=False,
        release_eligible=False,
        total_seconds=9999.0,
        model_load_seconds=1234.0,
    )
    appearance_path = tmp_path / "old_appearance.json"
    write_json(appearance_path, appearance)
    stage = tmp_path / "stage.json"
    write_json(
        stage,
        dict(
            schema="farm.quality-appearance-stage.v1",
            source_geometry=geometry,
            source_native_input=old_native,
            appearance=describe_file(appearance_path),
            selected_group_ids=[7, 9],
        ),
    )
    fresh = copy.deepcopy(old_evidence)
    fresh["source_native_input"] = new_native
    # Artifact paths may change, but content must be checked and identical.
    for i, row in enumerate(fresh["observations"]):
        row["crop"] = artifact(f"fresh_crop_{i}", "crop")
    evidence = tmp_path / "fresh.json"
    write_json(evidence, fresh)
    calls = []

    def run(argv):
        source = Path(argv[argv.index("--proposals") + 1])
        output = Path(argv[argv.index("--output") + 1])
        subset = json.loads(source.read_text())
        ids = [g["id"] for g in subset["groups"]]
        assert set(ids) == {r["object_id"] for r in subset["observations"]}
        calls.append(ids)
        output.mkdir()
        updated = dict(
            appearance,
            objects=[
                dict(
                    object_id=oid,
                    raw="updated",
                    parsed=dict(label="wall device"),
                    sheets=[],
                )
                for oid in ids
            ],
            proposals=describe_file(source),
            model_load_seconds=0.25,
        )
        write_json(output / "manifest.json", updated)

    monkeypatch.setattr(refinement, "main", run)
    return dict(
        stage=stage,
        evidence=evidence,
        fresh=fresh,
        model=model,
        output=tmp_path / "result",
        calls=calls,
        appearance=appearance,
    )


def run(case):
    write_json(case["evidence"], case["fresh"])
    refresh_appearance(case["stage"], case["evidence"], case["model"], case["output"])
    return json.loads((case["output"] / "manifest.json").read_text())


def test_unchanged_evidence_preserves_raw_annotations_without_model_load(case):
    result = run(case)
    assert case["calls"] == []
    assert result["objects"] == case["appearance"]["objects"]
    assert result["proposals"] == describe_file(case["evidence"])
    assert result["model_load_seconds"] == 0
    assert result["total_seconds"] < 9999
    audit = result["annotation_retention"]
    assert audit["retained_group_ids"] == [7, 9]
    assert audit["refreshed_group_ids"] == [] and audit["new_review_requests"] == 0
    assert audit["model_cache_compatibility_claimed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "crop",
        "masks",
        "source_image",
        "native_mask",
        "timestamp",
        "image_id",
        "source_name",
        "crop_source_xyxy",
        "turns",
        "clipped",
    ],
)
def test_changed_object_is_reviewed_without_relabeling_unchanged_objects(case, field):
    row = case["fresh"]["observations"][0]
    if isinstance(row[field], dict):
        path = case["output"].parent / "changed_artifact"
        path.write_text("changed bytes")
        row[field] = describe_file(path)
    else:
        row[field] = {
            "timestamp": "3",
            "image_id": 3,
            "source_name": "other_camera",
            "crop_source_xyxy": [1, 0, 20, 20],
            "turns": 1,
            "clipped": True,
        }[field]
    result = run(case)
    assert case["calls"] == [[7]]
    assert result["objects"][1] == case["appearance"]["objects"][1]
    assert result["objects"][0]["raw"] == "updated"
    assert result["annotation_retention"]["retained_group_ids"] == [9]
    assert result["annotation_retention"]["refreshed_group_ids"] == [7]
    assert result["model_load_seconds"] == 0.25


@pytest.mark.parametrize(
    "invalid", ["namespace", "hash", "duplicate", "timestamp", "closed"]
)
def test_invalid_evidence_cannot_retain_or_launch_inference(case, invalid):
    fresh = case["fresh"]
    if invalid == "namespace":
        fresh["source_groups"]["sha256"] = "0" * 64
    elif invalid == "hash":
        Path(fresh["observations"][0]["crop"]["path"]).write_text("tampered")
    elif invalid == "duplicate":
        fresh["observations"].append(copy.deepcopy(fresh["observations"][0]))
    elif invalid == "timestamp":
        fresh["observations"][1]["timestamp"] = "1"
    else:
        fresh["reserved_test_opened"] = True
    with pytest.raises(ValueError):
        run(case)
    assert case["calls"] == []
    assert not case["output"].exists()


def test_retention_is_not_silent_model_migration(case):
    (case["model"] / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError):
        run(case)
    assert case["calls"] == []
    assert not case["output"].exists()


def test_reordered_views_require_new_review(case):
    rows = case["fresh"]["observations"]
    rows[0], rows[1] = rows[1], rows[0]
    result = run(case)
    assert case["calls"] == [[7]]
    assert result["annotation_retention"]["retained_group_ids"] == [9]
