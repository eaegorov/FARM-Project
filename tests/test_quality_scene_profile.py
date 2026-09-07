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
    assert "--balance-error-modes" not in by_id["refinement_schedule"]["command"]
    args.balance_refinement_errors = True
    write(plan_path, compile_plan(args))
    balanced = plan_to_public_dict(load_plan(plan_path), tmp_path / "balanced_run")
    schedule = next(s for s in balanced["stages"] if s["id"] == "refinement_schedule")
    assert "--balance-error-modes" in schedule["command"]
    assert len(balanced["stages"]) == 18
    args.refinement_reuse = [tmp_path / "cache.json"]
    write(plan_path, compile_plan(args))
    cached = plan_to_public_dict(load_plan(plan_path), tmp_path / "cached_run")
    for stage in cached["stages"]:
        if stage["id"] in ("refinement_tracker", "refinement_concept"):
            assert str(args.refinement_reuse[0]) in stage["command"]
            assert "--reuse-proposals" in stage["command"]
            assert str(args.refinement_reuse[0]) in stage["fingerprint_inputs"]
    assert len(cached["stages"]) == 18
    args.partial_view_association = True
    args.complementary_model_root = tmp_path / "models"
    write(plan_path, compile_plan(args))
    complete = plan_to_public_dict(load_plan(plan_path), tmp_path / "complete")
    by_id = {s["id"]: s for s in complete["stages"]}
    assert len(complete["stages"]) == 20
    for name in ("initial_geometry", "geometry"):
        assert "--partial-view-association" in by_id[name]["command"]
    command = by_id["geometry"]["command"]
    assert command[command.index("--proposals") + 1].endswith(
        "/complementary_proposals/manifest.json"
    )
    assert by_id["geometry"]["needs"] == ["complementary_proposals"]
    assert by_id["complementary_proposals"]["needs"] == ["complementary_segmentation"]
    fingerprints = by_id["complementary_segmentation"]["fingerprint_inputs"]
    assert str(args.complementary_model_root) not in fingerprints
    for checkpoint in (
        "yoloe/yoloe-v8l-seg-pf.pt",
        "yoloe/yoloe-v8l-seg.pt",
        "mobileclip/mobileclip_blt.pt",
    ):
        assert str(args.complementary_model_root / checkpoint) in fingerprints
    assert "--max-primary-iou" in by_id["complementary_proposals"]["command"]
    args.recovery_groups = 8
    write(plan_path, compile_plan(args))
    recovery = plan_to_public_dict(load_plan(plan_path), tmp_path / "recovery")
    by_id = {s["id"]: s for s in recovery["stages"]}
    assert len(recovery["stages"]) == 27
    assert by_id["recovery_views"]["needs"] == ["geometry"]
    assert by_id["native"]["needs"] == ["recovery_completed_validation"]
    cmd = by_id["recovery_views"]["command"]
    assert "--group-id" not in cmd and cmd[cmd.index("--auto-groups") + 1] == "8"
    assert cmd[cmd.index("--view-budget") + 1] == "12"
    cmd = by_id["native"]["command"]
    validation = cmd[cmd.index("--recovery-validation") + 1]
    assert validation.endswith("/recovery/completed_validation/manifest.json")
    assert validation in by_id["native"]["fingerprint_inputs"]
    for name in (
        "recovery_validation",
        "recovery_tracker_validation",
        "recovery_completed_validation",
    ):
        assert "--partial-view-association" in by_id[name]["command"]
    assert "--scope-completion" in by_id["recovery_completed_validation"]["command"]
    # Coverage shares the same seven recovery stages and one native build.
    args.coverage_groups = 4
    write(plan_path, compile_plan(args))
    mixed = plan_to_public_dict(load_plan(plan_path), tmp_path / "mixed")
    mixed_by_id = {s["id"]: s for s in mixed["stages"]}
    assert len(mixed["stages"]) == len(recovery["stages"])
    cmd = mixed_by_id["recovery_views"]["command"]
    assert cmd[cmd.index("--auto-groups") + 1] == "8"
    assert cmd[cmd.index("--coverage-groups") + 1] == "4"
    assert cmd[cmd.index("--view-budget") + 1] == "20"
    cmd = mixed_by_id["recovery_segmentation"]["command"]
    assert cmd[cmd.index("--views") + 1] == "20"
    cmd = mixed_by_id["recovery_tracker"]["command"]
    assert cmd[cmd.index("--crop-budget") + 1] == "24"
    assert sum(s["id"] == "native" for s in mixed["stages"]) == 1
    for invalid in (-1, 17, True, 9):
        args.coverage_groups = invalid
        with pytest.raises(ValueError, match="budget"):
            compile_plan(args)
    args.coverage_groups = 4
    args.recovery_groups = 0
    write(plan_path, compile_plan(args))
    coverage_only = plan_to_public_dict(
        load_plan(plan_path), tmp_path / "coverage_only"
    )
    cmd = next(s for s in coverage_only["stages"] if s["id"] == "recovery_views")[
        "command"
    ]
    assert cmd[cmd.index("--auto-groups") + 1] == "0"
    args.coverage_groups = 0
    for invalid in (-1, 17, True):
        args.recovery_groups = invalid
        with pytest.raises(ValueError, match="budget"):
            compile_plan(args)
    args.recovery_groups = 0
    args.complementary_model_root = None
    args.complementary_vocabulary = tmp_path / "vocabulary.txt"
    with pytest.raises(ValueError, match="model root"):
        compile_plan(args)
    args.complementary_vocabulary = None
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



@pytest.mark.parametrize("scale", [1, 3])
@pytest.mark.parametrize(
    "case,deferred",
    [
        ("whole_expansion", True),
        ("same_extent", False),
        ("small_boundary_change", False),
        ("neighbor_overlap", False),
        ("weak_primary_containment", False),
        ("disconnected_added_object", False),
        ("expansion_with_disconnected_neighbor", False),
        ("existing_whole_duplicate", False),
    ],
)
def test_complementary_union_defers_extent_gain_without_promoting_or_relabeling(
    tmp_path, case, deferred, scale
):
    from farm_runtime.quality.proposal_geometry import read_masks
    from farm_runtime.quality.scene_profile import union_proposals

    primary = np.zeros((32, 40), bool)
    primary[6:16, 6:16] = True
    supplement = primary.copy()
    if case in (
        "whole_expansion",
        "expansion_with_disconnected_neighbor",
        "existing_whole_duplicate",
    ):
        supplement[6:16, 16:22] = True
    elif case == "small_boundary_change":
        supplement[6:16, 16] = True
    elif case in ("neighbor_overlap", "weak_primary_containment"):
        primary[6:18, 6:18] = True
        supplement[:] = False
        left = 8 if case == "neighbor_overlap" else 10
        supplement[6:18, left:left + 12] = True
    if case == "disconnected_added_object":
        supplement[6:16, 26:32] = True
    elif case == "expansion_with_disconnected_neighbor":
        supplement[6:12, 26:32] = True
    person = np.zeros_like(primary)
    person[:2, :2] = True
    primary_masks = [primary]
    if case == "existing_whole_duplicate":
        primary_masks.append(supplement)
    primary_masks.append(person)
    primary_masks = [
        np.repeat(np.repeat(m, scale, axis=0), scale, axis=1)
        for m in primary_masks
    ]
    supplement = np.repeat(np.repeat(supplement, scale, axis=0), scale, axis=1)
    height, width = supplement.shape
    photo = tmp_path / "photo.png"
    Image.new("RGB", (width, height)).save(photo)

    def source(name, masks, labels, scores):
        directory = tmp_path / name
        (directory / "masks").mkdir(parents=True)
        path = directory / "masks" / "frame.npz"
        arrays = {
            f"mask_{i}": np.where(m, 2.0, -2.0).astype(np.float16)
            for i, m in enumerate(masks)
        }
        np.savez_compressed(path, **arrays)
        row = dict(
            name="frame.png", timestamp="100", source_image=describe_file(photo),
            grid_shape_hw=[height, width], applied_quarter_turns=0,
            queries=[dict(prompt="person")] if name == "primary" else [],
            mask_artifact=describe_file(path),
            detections=[dict(label=label, score=score, logit_key=f"mask_{i}",
                             grid_window_xyxy=[0, 0, width, height])
                        for i, (label, score) in enumerate(zip(labels, scores))],
        )
        return write(
            directory / "manifest.json",
            dict(test_opened=False, observations=[row]),
        )

    # Different names and lower supplementary confidence do not decide identity.
    first = source(
        "primary", primary_masks,
        ["fragment"] * (len(primary_masks) - 1) + ["person"],
        [0.99] * len(primary_masks),
    )
    second = source("supplement", [supplement], ["equipment"], [0.51])
    out = tmp_path / "union"
    union_proposals(first, [second], out, maximum_primary_iou=0.5)
    result = json.loads((out / "manifest.json").read_text())
    records = result["suppressed_proposals"]
    assert len(records) == 1
    assert result["admitted_supplementary_proposals"] == []
    assert len(result["deferred_scope_expansions"]) == int(deferred)
    assert result["connected_expansion_policy"]["admitted_to_main_detections"] is False
    row = records[0]
    reference = primary_masks[row["primary_detection_index"]]
    overlap = int((reference & supplement).sum())
    assert row["primary_coverage"] == pytest.approx(overlap / reference.sum())
    assert row["supplement_coverage"] == pytest.approx(overlap / supplement.sum())
    assert row["physical_identity_verified"] is False
    assert row["reason"] == "overlapping_primary_proposal"
    assert row["deferred_scope_expansion"] is deferred
    if deferred:
        assert row["connected_new_fraction"] == pytest.approx(0.375)
        assert row["unanchored_component_fraction"] == 0
    if case == "disconnected_added_object":
        assert row["supplement_new_fraction"] == pytest.approx(0.375)
        assert row["connected_new_fraction"] == 0
    if case == "neighbor_overlap":
        assert row["primary_coverage"] >= 0.8
        assert row["supplement_to_primary_area_ratio"] == 1
    if case == "weak_primary_containment":
        assert row["primary_coverage"] < 0.8
    if case == "expansion_with_disconnected_neighbor":
        assert row["connected_new_fraction"] > 0.15
        assert row["unanchored_component_fraction"] > 0.05
    if case == "existing_whole_duplicate":
        assert row["primary_detection_index"] == 1
        assert row["primary_iou"] == 1
    merged = result["observations"][0]
    masks = read_masks(merged, out)
    assert len(masks) == len(primary_masks)
    for actual, expected in zip(masks, primary_masks):
        np.testing.assert_array_equal(actual, expected)
    if deferred:
        pending = result["deferred_scope_expansions"][0]
        assert pending["status"] == "pending_scope_review"
        assert pending["admitted_to_main_detections"] is False
        assert pending["mask_pixels_modified"] is False
        assert pending["source_manifest"] == describe_file(second)
        original = json.loads(second.read_text())["observations"][0]
        assert pending["source_mask_artifact"] == original["mask_artifact"]
        candidate = pending["observation"]
        np.testing.assert_array_equal(read_masks(candidate, out)[0], supplement)
        assert candidate["detections"][0]["score"] == 0.51
        assert candidate["detections"][0]["source_priority"] == 1
        assert candidate["source_image"] == merged["source_image"]
        assert candidate["timestamp"] == merged["timestamp"]
        with np.load(candidate["mask_artifact"]["path"]) as saved:
            with np.load(original["mask_artifact"]["path"]) as source_masks:
                np.testing.assert_array_equal(
                    saved[candidate["detections"][0]["logit_key"]],
                    source_masks[original["detections"][0]["logit_key"]],
                )
    transients = json.loads((out / "transients.json").read_text())["observations"][0]
    assert [d["label"] for d in transients["detections"]] == ["person"]
    np.testing.assert_array_equal(read_masks(transients, out)[0], primary_masks[-1])
    assert result["model_scores_calibrated_across_sources"] is False

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


@pytest.mark.parametrize(
    "invalid", [None, "ineligible", "same_timestamp", "changed_tile"]
)
def test_validated_cohort_preserves_masks_exclusions_and_flattened_indices(
    tmp_path, monkeypatch, invalid
):
    from farm_runtime.quality.scene_profile import validated_cohort
    from farm_runtime.quality import surface_evidence
    from farm_runtime.quality.proposal_geometry import read_masks

    image = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4)).save(image)
    expected = {}

    def source(name, timestamp, labels):
        directory = tmp_path / name
        (directory / "masks").mkdir(parents=True)
        arrays = {
            f"mask_{i}": (np.arange(16).reshape(4, 4) - i - 7).astype(np.float16)
            for i in range(len(labels))
        }
        target = directory / "masks" / "view.npz"
        np.savez_compressed(target, **arrays)
        row = dict(
            name="source" if name == "original" else "extra",
            timestamp=timestamp,
            source_image=describe_file(image),
            grid_shape_hw=[4, 4],
            applied_quarter_turns=0,
            queries=[dict(prompt="person")],
            detections=[
                dict(
                    index=i,
                    label=label,
                    score=0.9,
                    logit_key=f"mask_{i}",
                    grid_window_xyxy=[0, 0, 4, 4],
                )
                for i, label in enumerate(labels)
            ],
            mask_artifact=describe_file(target),
        )
        path = write(
            directory / "manifest.json", dict(test_opened=False, observations=[row])
        )
        expected[name] = arrays
        return path, row

    original, first = source("original", "1", ["box", "person"])
    extra, second = source(
        "extra", "1" if invalid == "same_timestamp" else "2", ["person"]
    )
    supplement, third = source(
        "supplement", second["timestamp"], ["unrelated", "device"]
    )
    transient = dict(first, detections=[first["detections"][1]])
    transient_path = write(
        original.parent / "transients.json",
        dict(test_opened=False, observations=[transient]),
    )
    geometry = write(tmp_path / "geometry.json", dict(groups=[dict(id=7, members=[0])]))
    inputs = SimpleNamespace(
        geometry=json.loads(geometry.read_text()),
        geometry_path=geometry,
        nodes={0: dict(frame="source", representative_detection=0)},
        observations={"source": first},
        transients={"source": transient},
        proposals_path=original,
        transients_path=transient_path,
    )
    monkeypatch.setattr(surface_evidence, "SurfaceInputs", lambda path: inputs)
    audit = write(
        tmp_path / "audit.json", dict(source_geometry=describe_file(geometry))
    )
    validation = write(
        tmp_path / "validation.json",
        dict(
            closed_test_opened=False,
            release_eligible=False,
            source_audit=describe_file(audit),
            source_proposals=describe_file(extra),
            supplements=[describe_file(supplement)],
            groups=[
                dict(
                    group_id=7,
                    extra_matches=[
                        dict(
                            name="extra",
                            selected_detection=2,
                            decision="matched_static_surface",
                            candidates=[
                                dict(
                                    detection_index=2, eligible=invalid != "ineligible"
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
    )
    if invalid == "changed_tile":
        np.savez_compressed(
            Path(third["mask_artifact"]["path"]),
            mask_0=np.ones((4, 4)),
            mask_1=np.ones((4, 4)),
        )
    output = tmp_path / "output"
    if invalid:
        with pytest.raises(
            ValueError,
            match={
                "ineligible": "geometrically accepted",
                "same_timestamp": "two independent timestamps",
                "changed_tile": "proposal mask changed",
            }[invalid],
        ):
            validated_cohort(validation, 7, output)
        assert not (output / "manifest.json").exists()
        return
    validated_cohort(validation, 7, output)
    cohort = json.loads((output / "manifest.json").read_text())
    rows = cohort["observations"]
    assert [r["timestamp"] for r in rows] == ["1", "2"]
    assert [[d["label"] for d in r["detections"]] for r in rows] == [
        ["box", "person"],
        ["device", "person"],
    ]
    for row, keys in zip(
        rows,
        [
            [("original", "mask_0"), ("original", "mask_1")],
            [("supplement", "mask_1"), ("extra", "mask_0")],
        ],
    ):
        masks = read_masks(row, output)
        with np.load(Path(row["mask_artifact"]["path"])) as archive:
            for i, (name, key) in enumerate(keys):
                np.testing.assert_array_equal(
                    archive[row["detections"][i]["logit_key"]], expected[name][key]
                )
                np.testing.assert_array_equal(masks[i], expected[name][key] > 0)
    assert rows[1]["detections"][0]["cohort_source"]["detection_index"] == 1
    transients = json.loads((output / "transients.json").read_text())["observations"]
    assert all([d["label"] for d in r["detections"]] == ["person"] for r in transients)
    assert cohort["inference_calls"] == 0 and cohort["original_group_id"] == 7


def test_recovered_semantics_do_not_displace_existing_multiview_candidates(
    tmp_path, monkeypatch
):
    from farm_runtime.quality import scene_profile, scope_evidence, refinement
    from farm_runtime.quality import native_observations

    geometry = tmp_path / "geometry.json"
    write(
        geometry,
        dict(
            test_opened=False,
            groups=[
                dict(id=1, independent_timestamps=1),
                dict(id=4, independent_timestamps=5),
                dict(id=8, independent_timestamps=3),
                dict(id=9, independent_timestamps=2),
            ],
        ),
    )
    native_input = tmp_path / "native_input.json"
    write(
        native_input,
        dict(
            objects=[
                dict(object_id=oid, masks=[dict(source_kind=kind)])
                for oid, kind in [
                    (1, "validated_additional_view"),
                    (4, "geometry_original"),
                    (8, "validated_additional_view"),
                    (9, "geometry_original"),
                ]
            ]
        ),
    )
    native = tmp_path / "native.json"
    write(native, dict(input=describe_file(native_input)))
    # 1 is newly recovered; 8 gained views; 4 lost independent support.
    monkeypatch.setattr(
        native_observations,
        "retained_timestamp_counts",
        lambda *_: {1: 4, 4: 1, 8: 6, 9: 2},
    )
    observed = []

    def prepare(argv):
        observed.extend(
            int(argv[i + 1]) for i, value in enumerate(argv) if value == "--group-id"
        )
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        write(output / "manifest.json", {})

    monkeypatch.setattr(scope_evidence, "main", prepare)

    def infer(argv):
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        write(output / "manifest.json", {})

    monkeypatch.setattr(refinement, "main", infer)
    output = tmp_path / "semantics"
    scene_profile.semantics(
        SimpleNamespace(
            geometry=geometry,
            native=native,
            groups=2,
            model=tmp_path / "model",
            output=output,
            retain_stage=None,
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert observed == [8, 9, 1]
    assert manifest["selected_group_ids"] == [8, 9, 1]
    assert manifest["recovery_added_group_ids"] == [1]
    assert manifest["deferred_group_ids"] == [4]
    assert manifest["candidate_budget"] == 2
    assert manifest["base_ranking"] == "original_timestamps_capped_by_retained_evidence"


@pytest.mark.parametrize("refinement_crops", [0, 12])
def test_whole_object_profile_binds_family_review_and_geometry_outputs(
    tmp_path, refinement_crops
):
    args = SimpleNamespace(
        initial_views=12, adaptive_views=8, vocabulary_views=8,
        native_groups=128, semantic_groups=64, alternatives=16,
        world_up=[0, -1, 0],
        runtimes=write(tmp_path / "runtimes.json", {
            "main": ["main-python"], "geometry": ["geometry-python"],
        }),
        rgbd=tmp_path / "rgbd", ply=tmp_path / "scene.ply",
        vlm_model=tmp_path / "vlm", sam_model=tmp_path / "sam",
        project_root=tmp_path, scene_id="new_scene",
        output_root=tmp_path / "output", refinement_crops=refinement_crops,
    )
    baseline = compile_plan(args)
    assert baseline["pipeline"]["stages"][-1]["id"] == "catalog"
    assert baseline["artifacts"]["catalog"].endswith("/catalog/catalog.json")
    assert baseline["quality_profile"]["whole_objects"]["enabled"] is False
    args.whole_objects = True
    enabled = compile_plan(args)
    # The opt-in cannot rewrite discovery or per-candidate native membership.
    assert enabled["pipeline"]["stages"][:-2] == baseline["pipeline"]["stages"]
    path = write(tmp_path / "whole_objects.json", enabled)
    resolved = plan_to_public_dict(load_plan(path), tmp_path / "run")
    stages = {s["id"]: s for s in resolved["stages"]}
    review, assembly = stages["scope_families"], stages["object_assembly"]
    assert review["needs"] == ["catalog"]
    assert assembly["needs"] == ["scope_families"]
    assert review["command"][0] == "main-python"
    assert assembly["command"][0] == "geometry-python"
    assert "scope-families" in review["command"]
    assert "assemble-objects" in assembly["command"]
    for stage in (review, assembly):
        assert "--group-id" not in stage["command"]
        assert "--object-id" not in stage["command"]
        assert all((chr(36) + "{") not in value for value in stage["command"])
        assert set(stage["inputs"]) <= set(stage["fingerprint_inputs"])
        assert str(tmp_path / "run/quality/catalog/catalog.json") in stage["inputs"]
        assert str(tmp_path / "run/quality/catalog/object_masks.npz") in stage["inputs"]
        final_name = "refined_native" if refinement_crops else "native"
        assert str(tmp_path / f"run/quality/{final_name}/manifest.json") in stage["inputs"]
    assert str(args.vlm_model) in review["fingerprint_inputs"]
    assert str(args.ply) in assembly["fingerprint_inputs"]
    assert str(tmp_path / "run/quality/scope_families/manifest.json") in assembly["inputs"]
    artifacts = resolved["artifacts"]
    assert artifacts["catalog"].endswith("/object_assembly/catalog.json")
    assert artifacts["native_masks"].endswith("/object_assembly/object_masks.npz")
    assert artifacts["source_catalog"].endswith("/catalog/catalog.json")
    assert artifacts["source_native_masks"].endswith("/catalog/object_masks.npz")
    assert artifacts["scope_families"] in review["outputs"]
    assert {
        artifacts["catalog"], artifacts["native_masks"], artifacts["object_assembly_report"],
    } == set(assembly["outputs"])
    assert enabled["quality_profile"]["whole_objects"]["catalog_schema"] == "farm.quality-object-assembly.v1"
    args.whole_objects = "yes"
    with pytest.raises(ValueError, match="boolean"):
        compile_plan(args)


def test_profile_cli_accepts_whole_objects_opt_in(tmp_path, monkeypatch):
    from farm_runtime.quality import scene_profile

    received = []
    def fake_compile(args):
        received.append(args.whole_objects)
        return {"whole_objects": args.whole_objects}
    monkeypatch.setattr(scene_profile, "compile_plan", fake_compile)
    command = [
        "plan", "--rgbd", str(tmp_path / "rgbd"),
        "--ply", str(tmp_path / "scene.ply"),
        "--sam-model", str(tmp_path / "sam"),
        "--vlm-model", str(tmp_path / "vlm"),
        "--project-root", str(tmp_path), "--output-root", str(tmp_path / "output"),
        "--scene-id", "any_scene", "--world-up", "0", "-1", "0",
    ]
    assert scene_profile.main(command + ["--whole-objects", "--output", str(tmp_path / "with.json")]) == 0
    assert scene_profile.main(command + ["--output", str(tmp_path / "without.json")]) == 0
    assert received == [True, False]
