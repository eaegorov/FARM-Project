import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from farm_runtime.config import load_plan, plan_to_public_dict
from farm_runtime.quality.scene_profile import (
    bounded_groups,
    compile_plan,
    merge_proposals,
    prepare_inputs,
    segment,
    union_vocabulary,
)
from farm_runtime.quality.scene_catalog import bound_appearance, geometry_status
from farm_runtime.quality_baseline import describe_file


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_profile_compiles_into_existing_dag_with_explicit_budgets_and_runtimes(
    tmp_path,
):
    runtime = write(
        tmp_path / "runtime.json",
        {"main": ["main-python"], "geometry": ["geometry-python"]},
    )
    args = SimpleNamespace(
        initial_views=12,
        adaptive_views=8,
        vocabulary_views=8,
        native_groups=128,
        semantic_groups=64,
        alternatives=16,
        world_up=[0, -1, 0],
        runtimes=runtime,
        rgbd=tmp_path / "rgbd",
        ply=tmp_path / "scene.ply",
        vlm_model=tmp_path / "vlm",
        sam_model=tmp_path / "sam",
        project_root=tmp_path,
        scene_id="independent_scene",
        output_root=tmp_path / "output",
    )
    plan_path = write(tmp_path / "pipeline.json", compile_plan(args))
    plan = load_plan(plan_path)
    assert len(plan.stages) == 13
    resolved = plan_to_public_dict(plan, tmp_path / "run")
    native = next(s for s in resolved["stages"] if s["id"] == "native")
    assert native["command"][0] == "geometry-python"
    assert (
        next(s for s in resolved["stages"] if s["id"] == "appearance")["command"][0]
        == "main-python"
    )
    assert all("${" not in x for s in resolved["stages"] for x in s["command"])
    assert all(s.timeout_seconds == 900 for s in plan.stages)
    args.refinement_crops = 12
    refined = compile_plan(args)
    write(plan_path, refined)
    stages = plan_to_public_dict(load_plan(plan_path), tmp_path / "refined_run")[
        "stages"
    ]
    assert len(stages) == 18
    by_id = {s["id"]: s for s in stages}
    assert by_id["refinement_selection"]["command"][0] == "geometry-python"
    assert by_id["refinement_concept"]["command"][0] == "main-python"
    assert by_id["refined_native"]["needs"] == ["refinement_selection"]
    for name in ("appearance", "catalog"):
        cmd = by_id[name]["command"]
        assert cmd[cmd.index("--native") + 1].endswith("/refined_native/manifest.json")
    args.refinement_crops = 33
    with pytest.raises(ValueError, match="budget"):
        compile_plan(args)
    args.refinement_crops = 12
    args.adaptive_views = 1000
    with pytest.raises(ValueError, match="budget"):
        compile_plan(args)


def test_group_budget_counts_independent_timestamps_and_keeps_deferred_ids():
    geometry = dict(
        test_opened=False,
        groups=[
            dict(id=i, independent_timestamps=n)
            for i, n in [(1, 1), (8, 3), (4, 2), (9, 3)]
        ],
    )
    assert bounded_groups(geometry, 2) == ([8, 9], [4])
    with pytest.raises(ValueError):
        bounded_groups(geometry, True)


def test_registered_plan_preserves_temporal_order_and_excludes_untrusted_rgb(tmp_path):
    rgbd = tmp_path / "rgbd"
    rgbd.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    frames = []
    for i in [3, 1, 2]:
        name = f"camera_{i}.png"
        Image.new("RGB", (30, 20), (i, 2, 3)).save(images / name)
        frames.append(
            dict(
                source_image=name,
                frame_id=str(i),
                camera="sensor",
                T_world_cam=np.eye(4).tolist(),
            )
        )
    write(
        rgbd / "frames.json",
        dict(
            scene_id="other",
            frames=frames,
            depth_units="metres",
            pose_translation_units="metres",
            camera_registration=dict(
                groups=[
                    dict(trusted=True, source_images=["camera_3.png", "camera_1.png"]),
                    dict(trusted=False, source_images=["camera_2.png"]),
                ]
            ),
        ),
    )
    write(
        rgbd / "prep_summary.json",
        dict(status="complete", inputs=dict(image_root=str(images))),
    )
    out = tmp_path / "out"
    prepare_inputs(rgbd, out)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["variants"][0]["views"] == ["camera_3.png", "camera_1.png"]
    assert plan["sources"]["camera_1.png"]["timestamp"] == "1"
    assert plan["sources"]["camera_1.png"]["shape_hw"] == [20, 30]


def test_union_keeps_core_and_records_budget_omissions(tmp_path):
    core = tmp_path / "core.txt"
    core.write_text("person\nbox\n")
    scene = write(
        tmp_path / "scene.json",
        dict(
            reserved_test_opened=False,
            categories=dict(objects={f"category {i}": i + 1 for i in range(90)}),
        ),
    )
    out = tmp_path / "union"
    union_vocabulary(scene, core, out)
    terms = (out / "vocabulary.txt").read_text().splitlines()
    assert len(terms) == 80 and {"person", "box", "category 89"} <= set(terms)
    report = json.loads((out / "manifest.json").read_text())
    assert len(report["omitted_scene_terms"]) == 12
    assert "category 0" in report["omitted_scene_terms"]


def batch(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    image = root / (name + ".png")
    Image.new("RGB", (5, 5)).save(image)
    mask = root / (name + ".npz")
    np.savez_compressed(mask, logits=np.ones((5, 5)))
    weights = root / "model"
    weights.write_bytes(b"fixed checkpoint")
    row = dict(
        name=name,
        source_image=describe_file(image),
        mask_artifact=describe_file(mask),
        detections=[dict(label="box"), dict(label="person")],
        queries=[dict(prompt="box"), dict(prompt="person")],
    )
    doc = dict(
        test_opened=False,
        observations=[row],
        vocabulary=describe_file(weights),
        model_config=describe_file(weights),
        model_weights=describe_file(weights),
        total_seconds=1,
    )
    return write(root / "manifest.json", doc)


def test_proposal_merge_preserves_order_masks_and_person_only_exclusions(tmp_path):
    first, second = batch(tmp_path, "first"), batch(tmp_path, "second")
    out = tmp_path / "merged"
    merge_proposals([first, second], out)
    d = json.loads((out / "manifest.json").read_text())
    t = json.loads((out / "transients.json").read_text())
    assert [r["name"] for r in d["observations"]] == ["first", "second"]
    assert all(r["detections"] == [dict(label="person")] for r in t["observations"])
    assert all(len(r["detections"]) == 2 for r in d["observations"])
    with pytest.raises(ValueError, match="disjoint"):
        merge_proposals([first, first], tmp_path / "duplicate")
    modified = json.loads(second.read_text())
    modified["model_weights"]["sha256"] = "changed"
    write(second, modified)
    with pytest.raises(ValueError, match="model/vocabulary"):
        merge_proposals([first, second], tmp_path / "drift")


def test_empty_adaptive_plan_does_not_load_inference(tmp_path, monkeypatch):
    from farm_runtime.quality import concept_discovery

    def fail(*args, **kwargs):
        raise AssertionError("empty schedule must not run inference")

    monkeypatch.setattr(concept_discovery, "main", fail)
    plan = write(
        tmp_path / "plan.json",
        dict(test_opened=False, variants=[dict(name="balanced_upright", views=[])]),
    )
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("person\nbox\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"checkpoint")
    out = tmp_path / "empty"
    segment(
        SimpleNamespace(
            plan=plan,
            vocabulary=vocab,
            model=model,
            views=8,
            world_up=[0, -1, 0],
            output=out,
        )
    )
    assert (
        json.loads((out / "manifest.json").read_text())["source_image_encoder_calls"]
        == 0
    )


def test_catalog_rejects_same_integer_ids_from_another_scene(tmp_path):
    first = write(tmp_path / "first.json", dict(scene="first"))
    second = write(tmp_path / "second.json", dict(scene="second"))
    stage = write(
        tmp_path / "appearance.json",
        dict(
            schema="farm.quality-appearance-stage.v1",
            source_geometry=describe_file(first),
        ),
    )
    with pytest.raises(ValueError, match="geometry namespace"):
        bound_appearance(stage, describe_file(second))


def test_observed_extent_and_empty_uncertainty_never_certify_physical_size():
    value = geometry_status(
        dict(extent_stability=dict(sensitive_axes=[0, 2])), competing_scopes={7}
    )
    assert value["flags"] == [
        "observed_extent_tail_sensitive",
        "unresolved_nested_scope",
    ]
    assert value["physical_dimensions_m"] is None
    assert value["physical_size_validated"] is False
    assert (
        geometry_status(None, competing_scopes=set())["observed_obb_available"] is False
    )


def test_complementary_union_preserves_tiles_primary_exclusions_and_source_priority(
    tmp_path,
):
    from farm_runtime.quality.scene_profile import union_proposals
    from farm_runtime.quality.proposal_geometry import read_masks
    from farm_runtime.proposal_geometry import equivalent_masks

    first, second = batch(tmp_path, "primary"), batch(tmp_path, "supplement")
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    for manifest, doc in [(first, a), (second, b)]:
        source = Path(doc["observations"][0]["mask_artifact"]["path"])
        target = manifest.parent / "masks" / source.name
        target.parent.mkdir()
        source.rename(target)
        doc["observations"][0]["mask_artifact"] = describe_file(target)
    row = a["observations"][0]
    row.update(timestamp="100", grid_shape_hw=[5, 5], applied_quarter_turns=0)
    row["detections"] = [
        dict(
            index=i,
            label=label,
            score=score,
            logit_key="logits",
            grid_window_xyxy=[0, 0, 5, 5],
        )
        for i, (label, score) in enumerate([("box", 0.55), ("person", 0.6)])
    ]
    other = copy.deepcopy(row)
    other["mask_artifact"] = b["observations"][0]["mask_artifact"]
    other["detections"] = [dict(row["detections"][0], label="heater", score=0.99)]
    other["queries"] = []
    b["observations"] = [other]
    write(first, a)
    write(second, b)
    out = tmp_path / "union"
    union_proposals(first, [second], out)
    result = json.loads((out / "manifest.json").read_text())
    merged = result["observations"][0]
    assert [d["source_priority"] for d in merged["detections"]] == [0, 0, 1]
    assert len({d["logit_key"] for d in merged["detections"]}) == 3
    masks = read_masks(merged, out)
    assert all(m.all() for m in masks)
    groups = equivalent_masks(masks, [0.55, 0.6, 0.99], priorities=[0, 0, 1])
    assert groups[0][0] == 1  # Primary confidence orders primary masks.
    transient = json.loads((out / "transients.json").read_text())["observations"][0]
    assert [d["label"] for d in transient["detections"]] == ["person"]
    assert np.array_equal(read_masks(transient, out)[0], masks[1])
    assert result["model_scores_calibrated_across_sources"] is False
    novelty = tmp_path / "novel_union"
    union_proposals(first, [second], novelty, maximum_primary_iou=0.5)
    novel = json.loads((novelty / "manifest.json").read_text())
    assert len(novel["observations"][0]["detections"]) == 2
    assert len(novel["suppressed_proposals"]) == 1
    assert novel["suppressed_proposals"][0]["primary_iou"] == 1.0
    with pytest.raises(ValueError, match="IoU"):
        union_proposals(
            first, [second], tmp_path / "invalid_iou", maximum_primary_iou=0
        )
    for field, value in [
        ("timestamp", "different"),
        ("grid_shape_hw", [10, 10]),
        ("applied_quarter_turns", 1),
    ]:
        changed = copy.deepcopy(b)
        changed["observations"][0][field] = value
        write(second, changed)
        target = tmp_path / ("bad_" + field)
        with pytest.raises(ValueError, match="identity/grid"):
            union_proposals(first, [second], target)
        assert not target.exists()


def test_complementary_union_requires_primary_view_and_person_authority(tmp_path):
    from farm_runtime.quality.scene_profile import union_proposals

    first, second = batch(tmp_path, "primary"), batch(tmp_path, "other")
    with pytest.raises(ValueError, match="primary observations"):
        union_proposals(first, [second], tmp_path / "new_views")
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    b["observations"] = copy.deepcopy(a["observations"])
    write(second, b)
    a["observations"][0]["queries"] = []
    write(first, a)
    with pytest.raises(ValueError, match="person query"):
        union_proposals(first, [second], tmp_path / "no_exclusions")
