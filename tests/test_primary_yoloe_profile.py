"""Regression contracts for a YOLOE-first common pipeline."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from farm_runtime.quality import discovery
from farm_runtime.quality.scene_profile import (
    compile_plan,
    merge_proposals,
    union_vocabulary,
)
from farm_runtime.quality_baseline import describe_file, write_json


def fixture(tmp_path):
    model = tmp_path / "models"
    (model / "yoloe").mkdir(parents=True)
    for name in ["mobileclip2_b.ts", "yoloe-26x-seg.pt"]:
        (model / "yoloe" / name).write_bytes(name.encode())
    vocab = tmp_path / "vocabulary.txt"
    vocab.write_text("person\ncart\n")
    sources = {}
    for i in range(4):
        image = tmp_path / f"camera_{i}.png"
        Image.new("RGB", (20, 16)).save(image)
        sources[image.name] = dict(
            name=image.name,
            timestamp=str(i),
            shape_hw=[16, 20],
            camera_from_world_rotation=np.eye(3).tolist(),
            source_image=describe_file(image),
        )
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        dict(
            test_opened=False,
            sources=sources,
            variants=[dict(name="balanced_upright", views=list(sources))],
        ),
    )
    return SimpleNamespace(
        plan=plan,
        model_root=model,
        vocabulary=vocab,
        checkpoint=model / "yoloe/yoloe-26x-seg.pt",
        confidence=0.25,
        world_up=[0, -1, 0],
        views=2,
        output=tmp_path / "output",
    )


def fake_infer(plan, output, **kwargs):
    target = output / "balanced_upright"
    target.mkdir()
    rows = []
    for name in plan["variants"][0]["views"]:
        p = target / (Path(name).stem + ".npz")
        np.savez_compressed(
            p, foreground=np.array([[-20.0, 1.5], [2.5, -20.0]], np.float16)
        )
        rows.append(
            dict(
                plan["sources"][name],
                grid_shape_hw=[16, 20],
                mask_artifact=describe_file(p),
                detections=[
                    dict(
                        label="cart",
                        logit_key="foreground",
                        grid_window_xyxy=[1, 1, 3, 3],
                    )
                ],
            )
        )
    write_json(
        target / "predictions.json", dict(person_query_present=True, observations=rows)
    )
    write_json(
        output / "results.json",
        dict(
            model=describe_file(kwargs["checkpoint"]),
            model_components=dict(
                text_encoder_checkpoint=describe_file(
                    kwargs["model_root"] / "yoloe/mobileclip2_b.ts"
                )
            ),
        ),
    )


def test_profile_preserves_soft_masks_query_evidence_and_batch_model_identity(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    monkeypatch.setattr(discovery, "infer", fake_infer)
    discovery.primary_profile(args)
    doc = json.loads((args.output / "manifest.json").read_text())
    assert [r["timestamp"] for r in doc["observations"]] == ["0", "3"]
    assert doc["labels_are_detector_hypotheses"]
    assert not doc["visual_features_for_legacy_association_exported"]
    assert doc["observations"][0]["queries"] == [
        dict(prompt="person", detections=0),
        dict(prompt="cart", detections=1),
    ]
    mask = np.load(doc["observations"][0]["mask_artifact"]["path"])["foreground"]
    np.testing.assert_array_equal(mask, [[-20, 1.5], [2.5, -20]])
    transient = json.loads((args.output / "transients.json").read_text())
    assert all(
        r["detections"] == [] and r["queries"] == [dict(prompt="person", detections=0)]
        for r in transient["observations"]
    )
    # Exhausted adaptive stage produces a compatible manifest without inference.
    source = json.loads(args.plan.read_text())
    source["variants"][0]["views"] = []
    write_json(args.plan, source)
    monkeypatch.setattr(
        discovery, "infer", lambda *a, **kw: pytest.fail("empty plan loaded model")
    )
    args.output = tmp_path / "empty"
    discovery.primary_profile(args)
    merged = tmp_path / "merged"
    merge_proposals(
        [tmp_path / "output/manifest.json", args.output / "manifest.json"], merged
    )
    result = json.loads((merged / "manifest.json").read_text())
    assert len(result["observations"]) == 2
    assert (
        result["observations"][0]["mask_artifact"]["sha256"]
        == doc["observations"][0]["mask_artifact"]["sha256"]
    )


@pytest.mark.parametrize("fault", ["person", "text", "detector", "unknown_label"])
def test_primary_rejects_mismatched_loaded_model_and_exclusion_contract(
    tmp_path, monkeypatch, fault
):
    args = fixture(tmp_path)

    def invalid(plan, output, **kwargs):
        fake_infer(plan, output, **kwargs)
        p = output / (
            "balanced_upright/predictions.json"
            if fault in {"person", "unknown_label"}
            else "results.json"
        )
        doc = json.loads(p.read_text())
        if fault == "person":
            doc["person_query_present"] = False
        elif fault == "unknown_label":
            doc["observations"][0]["detections"][0]["label"] = "unknown"
        elif fault == "text":
            doc["model_components"]["text_encoder_checkpoint"]["sha256"] = "0" * 64
        else:
            doc["model"]["sha256"] = "0" * 64
        write_json(p, doc)

    monkeypatch.setattr(discovery, "infer", invalid)
    with pytest.raises(ValueError):
        discovery.primary_profile(args)
    assert not (args.output / "manifest.json").exists()


def test_wide_vocabulary_retains_scene_role_conflicts_and_person(tmp_path):
    core = tmp_path / "core.txt"
    core.write_text("\n".join(["cart", "wall"] + [f"term {i}" for i in range(100)]))
    scene = tmp_path / "scene.json"
    write_json(
        scene,
        dict(
            reserved_test_opened=False,
            categories=dict(
                objects={"cart": 2, "wall": 1, "hoist": 1}, structures={"wall": 3}
            ),
        ),
    )
    out = tmp_path / "vocab"
    union_vocabulary(scene, core, out, maximum_terms=2048, ensure_person=True)
    terms = (out / "vocabulary.txt").read_text().splitlines()
    assert {"person", "wall", "hoist", "term 99"} <= set(terms)
    doc = json.loads((out / "manifest.json").read_text())
    assert doc["query_roles"]["wall"] == ["objects", "structures"]
    assert doc["query_roles_are_scene_hypotheses"]
    assert not doc["omitted_scene_terms"]


def test_compiled_yoloe_pipeline_has_no_global_sam_and_isolates_new_runtime(tmp_path):
    runtimes = tmp_path / "runtimes.json"
    write_json(
        runtimes,
        dict(
            main=["main-python"],
            geometry=["geometry-python"],
            detector=["isolated-python"],
        ),
    )
    args = SimpleNamespace(
        detector="yoloe",
        yoloe_model_root=tmp_path / "models",
        yoloe_checkpoint=tmp_path / "26x.pt",
        detector_confidence=0.25,
        sam_model=None,
        depth_consistent_association=True,
        explore_uncovered=True,
        initial_views=12,
        adaptive_views=8,
        vocabulary_views=8,
        native_groups=128,
        semantic_groups=128,
        alternatives=16,
        world_up=[0, -1, 0],
        runtimes=runtimes,
        rgbd=tmp_path / "rgbd",
        ply=tmp_path / "scene.ply",
        vlm_model=tmp_path / "qwen",
        project_root=tmp_path,
        scene_id="new_scene",
        output_root=tmp_path / "out",
    )
    plan = compile_plan(args)
    stages = {s["id"]: s for s in plan["pipeline"]["stages"]}
    assert len(stages) == 13
    assert all("--sam-model" not in s["command"] for s in stages.values())
    for name in ["initial_segmentation", "adaptive_segmentation"]:
        s = stages[name]
        assert s["command"][0] == "isolated-python"
        assert "yoloe-discovery" in s["command"] and "--checkpoint" in s["command"]
        assert str(args.yoloe_checkpoint) in s["fingerprint_inputs"]
        assert (
            str(args.yoloe_model_root / "yoloe/mobileclip2_b.ts")
            in s["fingerprint_inputs"]
        )
    assert stages["coverage"]["needs"] == ["initial_geometry"]
    assert "--target-timestamps" not in stages["coverage"]["command"]
    args.target_timestamps = 3
    confirmed = compile_plan(args)
    coverage = next(s for s in confirmed["pipeline"]["stages"] if s["id"] == "coverage")
    command = coverage["command"]
    assert command[command.index("--target-timestamps") + 1] == "3"
    assert confirmed["quality_profile"]["budgets"] == plan["quality_profile"]["budgets"]
    assert len(confirmed["pipeline"]["stages"]) == len(plan["pipeline"]["stages"])
    args.target_timestamps = None

    assert "--explore-uncovered" in stages["coverage"]["command"]
    for name in ["initial_geometry", "geometry"]:
        assert "--depth-consistent-association" in stages[name]["command"]
    assert stages["appearance"]["command"][0] == "main-python"
    assert "--ensure-person" in stages["union_vocabulary"]["command"]
    for change, error in [
        ({"runtimes": None}, "isolated"),
        ({"refinement_crops": 2}, "SAM model"),
        ({"detector": "sam3"}, "YOLOE options"),
        ({"complementary_model_root": tmp_path}, "complementary"),
    ]:
        modified = copy.copy(args)
        for k, v in change.items():
            setattr(modified, k, v)
        with pytest.raises(ValueError, match=error):
            compile_plan(modified)
