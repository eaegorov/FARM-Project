"""Real geometry artifacts must preserve scene identity and unknown pixels."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from farm_runtime.quality.native_observations import prepare_geometry, load_prepared
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge.common import load_observation_exclusion, load_mask_pair
from tools.farm_shaper_bridge.gaussian_lift import load_config


def geometry_fixture(root):
    root.mkdir()
    (root / "masks").mkdir()
    rows, observations, nodes, arrays = [], [], [], {}
    mask = np.full((640, 640), -1, np.float16)
    mask[160:480, 160:480] = 1
    person = np.full_like(mask, -1)
    person[200:400, 260:380] = 1
    config = dict(
        meters_per_scene_unit=2.0,
        depth_min_m=0.1,
        depth_max_m=20,
        identity_regex=r"(?P<sensor>cam\d+)_(?P<timestamp>\d+)_(?P<family>\w+)\.png",
        sensor_group="sensor",
        family_group="family",
    )
    for i, (sensor, timestamp) in enumerate(
        [("cam0", "001"), ("cam1", "001"), ("cam0", "002")]
    ):
        name = f"{sensor}_{timestamp}_center.png"
        rgb = root / name
        Image.new("RGB", (64, 64), "gray").save(rgb)
        depth = root / f"{i}.npy"
        np.save(depth, np.full((64, 64), 2, np.float32))
        rows.append(
            dict(
                source_image=name,
                frame_id=timestamp,
                timestamp_ns=int(timestamp) * 100,
                camera=sensor,
                depth_size=[64, 64],
                K=[[64, 0, 32], [0, 64, 32], [0, 0, 1]],
                T_world_cam=np.eye(4).tolist(),
                rgb_path=rgb.name,
                depth_path=depth.name,
            )
        )
        path = root / "masks" / f"{i}.npz"
        np.savez_compressed(path, target=mask, person=person)
        observations.append(
            dict(
                name=name,
                timestamp=timestamp,
                sensor=sensor,
                family="center",
                source_image=describe_file(rgb),
                grid_shape_hw=[640, 640],
                applied_quarter_turns=0,
                queries=[dict(prompt="box"), dict(prompt="person")],
                mask_artifact=describe_file(path),
                detections=[
                    dict(
                        label="box",
                        score=0.9,
                        logit_key="target",
                        grid_window_xyxy=[0, 0, 640, 640],
                    ),
                    dict(
                        label="person",
                        score=0.9,
                        logit_key="person",
                        grid_window_xyxy=[0, 0, 640, 640],
                    ),
                ],
            )
        )
        nodes.append(
            dict(
                id=i,
                frame=name,
                timestamp=timestamp,
                representative_detection=0,
                pixel_footprint_m=0.01,
                labels=["box"],
            )
        )
        xx, yy = np.meshgrid(np.linspace(-0.4, 0.4, 10), np.linspace(-0.3, 0.3, 10))
        arrays[f"node_{i:04d}"] = np.column_stack(
            (xx.ravel(), yy.ravel(), np.full(100, 2))
        )
    frames = root / "frames.json"
    write_json(
        frames,
        dict(
            scene_id="second_scene",
            depth_units="metres",
            pose_translation_units="metres",
            meters_per_scene_unit=2.0,
            frames=rows,
            camera_registration=dict(
                groups=[
                    dict(trusted=True, source_images=[r["name"] for r in observations])
                ]
            ),
        ),
    )
    ply = root / "source.ply"
    ply.write_text("synthetic source identity only")
    write_json(
        root / "run_manifest.json",
        dict(
            status="complete",
            qa_passed=True,
            fingerprint="fixture",
            fingerprint_payload=dict(
                config=config, inputs=dict(ply=describe_file(ply))
            ),
        ),
    )
    write_json(
        root / "prep_summary.json", dict(status="complete", fingerprint="fixture")
    )
    write_json(
        root / "proposals.json", dict(test_opened=False, observations=observations)
    )
    transient = json.loads((root / "proposals.json").read_text())
    for row in transient["observations"]:
        row["detections"] = row["detections"][1:]
        row["queries"] = row["queries"][1:]
    write_json(root / "transients.json", transient)
    np.savez_compressed(root / "surface_points.npz", **arrays)
    path = root / "geometry.json"
    write_json(
        path,
        dict(
            schema="farm.proposal-surface-association.v1",
            test_opened=False,
            release_eligible=False,
            inputs={
                k: describe_file(root / file)
                for k, file in [
                    ("frames", "frames.json"),
                    ("prep_summary", "prep_summary.json"),
                    ("proposals", "proposals.json"),
                    ("transients", "transients.json"),
                ]
            },
            frames=[
                dict(
                    source=r["source_image"],
                    depth_artifact=describe_file(root / r["depth_path"]),
                )
                for r in rows
            ],
            nodes=nodes,
            surface_artifact=describe_file(root / "surface_points.npz"),
            groups=[
                dict(
                    id=7,
                    members=[0, 1, 2],
                    independent_timestamps=2,
                    candidate_labels=["box"],
                ),
                dict(
                    id=8,
                    members=[0, 1],
                    independent_timestamps=1,
                    candidate_labels=["box"],
                ),
            ],
        ),
    )
    lift_config, _ = load_config(Path("configs/gaussian_lift.v1.yaml"))
    return path, lift_config


def test_direct_geometry_roundtrip_retains_scale_stereo_votes_and_unknown(tmp_path):
    geometry, config = geometry_fixture(tmp_path / "source")
    run, split, manifest = prepare_geometry(
        geometry, config, tmp_path / "native", [0, -1, 0]
    )
    restored, restored_split, _ = load_prepared(tmp_path / "native/manifest.json")
    assert run.scene_id == restored.scene_id == "second_scene"
    assert run.meters_per_scene_unit == restored.meters_per_scene_unit == 2.0
    assert [o.object_id for o in restored.objects] == [
        7
    ]  # Single timestamp is not promoted.
    assert split == restored_split
    assert split["objects"][0]["build_timestamps"] == ["100", "200"]
    assert len(restored.objects[0].observations) == 3
    assert manifest["source_validation"] is None
    assert manifest["observation_origin"] == "geometric_association_only"
    assert not manifest["closed_test_opened"] and not manifest["release_eligible"]
    frame = restored.frames[0]
    excluded, _ = load_observation_exclusion(restored, frame)
    raw, _, _ = load_mask_pair(restored.mask_overrides[7, 0], frame.depth_size)
    assert excluded.any() and (excluded & raw).any() and (raw & ~excluded).any()
    # The observed mask is retained; person-covered pixels are an independent unknown layer.
    np.testing.assert_array_equal(restored.objects[0].center_m, run.objects[0].center_m)


def test_direct_geometry_refuses_single_timestamp_or_changed_mask(tmp_path):
    geometry, config = geometry_fixture(tmp_path / "source")
    with pytest.raises(ValueError, match="two independent timestamps"):
        prepare_geometry(geometry, config, tmp_path / "single", [0, -1, 0], [8])
    mask = tmp_path / "source/masks/0.npz"
    mask.write_bytes(mask.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare_geometry(geometry, config, tmp_path / "changed", [0, -1, 0], [7])


@pytest.mark.parametrize("kind", ["untrusted", "timestamp", "transient"])
def test_direct_geometry_refuses_invalid_source_evidence(tmp_path, kind):
    geometry, config = geometry_fixture(tmp_path / "source")
    data = json.loads(geometry.read_text())
    if kind == "transient":
        path = tmp_path / "source/transients.json"
        value = json.loads(path.read_text())
        value["observations"][0]["detections"][0]["label"] = "box"
        message = "person-only"
        field = "transients"
    else:
        path = tmp_path / "source/frames.json"
        value = json.loads(path.read_text())
        if kind == "untrusted":
            value["camera_registration"]["groups"][0]["trusted"] = False
            message = "registration"
        else:
            value["frames"][0]["frame_id"] = "009"
            message = "timestamp"
        field = "frames"
    write_json(path, value)
    data["inputs"][field] = describe_file(path)
    write_json(geometry, data)
    with pytest.raises(ValueError, match=message):
        prepare_geometry(geometry, config, tmp_path / "invalid", [0, -1, 0], [7])


def test_preparation_stat_fingerprint_is_checked_before_content_upgrade(tmp_path):
    from farm_runtime.quality.native_observations import freeze_source_ply

    path = tmp_path / "source.ply"
    path.write_bytes(b"original")
    stat = path.stat()
    record = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    frozen = freeze_source_ply(record)
    assert frozen["sha256"] == describe_file(path)["sha256"]
    assert frozen["preparation_binding"] == "size_mtime_then_sha256"
    path.write_bytes(b"changed scene")
    with pytest.raises(ValueError, match="stat fingerprint"):
        freeze_source_ply(record)
    with pytest.raises(ValueError, match="changed since preparation"):
        freeze_source_ply(frozen)


def test_native_sensor_family_comes_from_registered_preparation(tmp_path):
    geometry, config = geometry_fixture(tmp_path / "case")
    doc = json.loads(geometry.read_text())
    proposals = Path(doc["inputs"]["proposals"]["path"])
    raw = json.loads(proposals.read_text())
    for row in raw["observations"]:
        row.pop("sensor")
        row.pop("family")
    write_json(proposals, raw)
    doc["inputs"]["proposals"] = describe_file(proposals)
    write_json(geometry, doc)
    run, _, _ = prepare_geometry(geometry, config, tmp_path / "native", [0, 0, 1])
    restored, _, _ = load_prepared(tmp_path / "native/manifest.json")
    assert [(f.sensor, f.family) for f in run.frames] == [
        (f.sensor, f.family) for f in restored.frames
    ]
    assert {f.sensor for f in run.frames} == {"cam0", "cam1"}
    raw["observations"][0]["sensor"] = "wrong_sensor"
    write_json(proposals, raw)
    doc["inputs"]["proposals"] = describe_file(proposals)
    write_json(geometry, doc)
    with pytest.raises(ValueError, match="registered sensor/family"):
        prepare_geometry(geometry, config, tmp_path / "conflict", [0, 0, 1])


def recovery_fixture(root):
    geometry, config = geometry_fixture(root)
    evidence = root / "evidence.json"
    write_json(evidence, dict(scope_alternatives=[]))
    doc = json.loads(geometry.read_text())
    doc["evidence_artifact"] = describe_file(evidence)
    # A true single-view seed. The accepted view comes from another timestamp.
    doc["groups"][0]["members"] = [1, 2]
    doc["groups"][1]["members"] = [0]
    write_json(geometry, doc)
    audit = root / "audit.json"
    write_json(audit, dict(source_geometry=describe_file(geometry)))
    extra = root / "extra.json"
    original = json.loads((root / "proposals.json").read_text())
    write_json(
        extra, dict(test_opened=False, observations=[original["observations"][2]])
    )
    validation = root / "validation.json"
    write_json(
        validation,
        dict(
            closed_test_opened=False,
            release_eligible=False,
            source_audit=describe_file(audit),
            source_proposals=describe_file(extra),
            groups=[
                dict(
                    group_id=8,
                    extra_matches=[
                        dict(
                            name="cam0_002_center.png",
                            timestamp="002",
                            selected_detection=0,
                            decision="matched_static_surface",
                            candidates=[dict(detection_index=0, eligible=True)],
                        )
                    ],
                )
            ],
        ),
    )
    return geometry, config, validation


def test_recovery_preserves_other_objects_and_reaches_semantic_evidence(tmp_path):
    from farm_runtime.quality.native_observations import (
        confirmed_recovery_groups,
        retained_timestamp_counts,
    )
    from farm_runtime.quality import scope_evidence
    from farm_runtime.quality.scene_profile import bounded_groups

    geometry, config, validation = recovery_fixture(tmp_path / "source")
    _, _, before = prepare_geometry(geometry, config, tmp_path / "before", [0, -1, 0])
    _, _, after = prepare_geometry(
        geometry,
        config,
        tmp_path / "after",
        [0, -1, 0],
        recovery_validation=validation,
    )
    assert confirmed_recovery_groups(geometry, validation) == [8]
    assert [row["object_id"] for row in after["objects"]] == [7, 8]
    previous, retained = before["objects"][0], after["objects"][0]
    assert previous["geometry"] == retained["geometry"]
    assert [(r["source_name"], r["mask"]["sha256"]) for r in previous["masks"]] == [
        (r["source_name"], r["mask"]["sha256"]) for r in retained["masks"]
    ]
    assert after["source_geometry"] == before["source_geometry"]
    assert after["source_validation"] == describe_file(validation)
    native_input = tmp_path / "after/manifest.json"
    assert retained_timestamp_counts(native_input, geometry) == {7: 2, 8: 2}
    assert bounded_groups(json.loads(geometry.read_text()), 128, {7: 1, 8: 2}) == (
        [8],
        [],
    )
    out = tmp_path / "semantics"
    scope_evidence.main(
        [
            "--groups",
            str(geometry),
            "--native-input",
            str(native_input),
            "--group-id",
            "8",
            "--output",
            str(out),
        ]
    )
    evidence = json.loads((out / "manifest.json").read_text())
    assert evidence["groups"][0]["independent_timestamps"] == 1
    assert evidence["groups"][0]["retained_independent_timestamps"] == 2
    assert {r["timestamp"] for r in evidence["observations"]} == {"001", "002"}
    assert {r["mask_source"] for r in evidence["observations"]} == {
        "geometry_original",
        "additional_native_raw",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "ineligible",
        "ambiguous",
        "person",
        "same_timestamp",
        "namespace",
        "missing_person",
    ],
)
def test_recovery_refuses_unconfirmed_or_mismatched_observations(tmp_path, fault):
    geometry, config, validation = recovery_fixture(tmp_path / "source")
    doc = json.loads(validation.read_text())
    match = doc["groups"][0]["extra_matches"][0]
    if fault == "ineligible":
        match["candidates"][0]["eligible"] = False
    elif fault == "ambiguous":
        match["decision"] = "ambiguous_competing_scope"
    elif fault == "person":
        match["selected_detection"] = 1
        match["candidates"] = [dict(detection_index=1, eligible=True)]
    elif fault in ("same_timestamp", "missing_person"):
        path = Path(doc["source_proposals"]["path"])
        extra = json.loads(path.read_text())
        if fault == "same_timestamp":
            original = json.loads((geometry.parent / "proposals.json").read_text())
            extra["observations"] = [original["observations"][1]]
            match["name"] = "cam1_001_center.png"
            match["timestamp"] = "001"
        else:
            extra["observations"][0]["queries"] = [dict(prompt="box")]
        write_json(path, extra)
        doc["source_proposals"] = describe_file(path)
    elif fault == "namespace":
        changed = json.loads(geometry.read_text())
        changed["note"] = "different namespace, same integer IDs"
        write_json(geometry, changed)
    write_json(validation, doc)
    with pytest.raises(ValueError):
        prepare_geometry(
            geometry,
            config,
            tmp_path / "refused",
            [0, -1, 0],
            [7, 8],
            recovery_validation=validation,
        )
    assert not (tmp_path / "refused/manifest.json").exists()


def test_semantics_budget_uses_recovered_and_quarantined_timestamps(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from farm_runtime.quality import scene_profile, scope_evidence, refinement

    geometry, config, validation = recovery_fixture(tmp_path / "source")
    prepare_geometry(
        geometry,
        config,
        tmp_path / "native",
        [0, -1, 0],
        recovery_validation=validation,
    )
    input_path = tmp_path / "native/manifest.json"
    doc = json.loads(input_path.read_text())
    # Quarantine the sole second timestamp for object 7. Its old metadata still says 2.
    doc["objects"][0]["masks"] = [
        row for row in doc["objects"][0]["masks"] if row["physical_timestamp_ns"] == 100
    ]
    write_json(input_path, doc)
    native = tmp_path / "native.json"
    write_json(native, dict(input=describe_file(input_path)))
    called = []

    def fake_inference(argv):
        evidence = json.loads(Path(argv[argv.index("--proposals") + 1]).read_text())
        called.extend(g["id"] for g in evidence["groups"])
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        write_json(output / "manifest.json", {})

    monkeypatch.setattr(refinement, "main", fake_inference)
    output = tmp_path / "appearance"
    scene_profile.semantics(
        SimpleNamespace(
            geometry=geometry,
            native=native,
            groups=64,
            model=tmp_path / "model",
            output=output,
        )
    )
    assert called == [8]
    stage = json.loads((output / "manifest.json").read_text())
    assert stage["selected_group_ids"] == [8]
    assert stage["deferred_group_ids"] == [7]


def test_native_recovery_adds_budget_without_displacing_original_groups(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from farm_runtime.quality import native_observations, scene_profile

    geometry, config, validation = recovery_fixture(tmp_path / "source")
    captured = []

    def capture(argv):
        captured.extend(argv)
        Path(argv[argv.index("--output") + 1]).mkdir()

    monkeypatch.setattr(native_observations, "main", capture)
    output = tmp_path / "output"
    scene_profile.native(
        SimpleNamespace(
            geometry=geometry,
            recovery_validation=validation,
            groups=1,
            ply=geometry.parent / "source.ply",
            config=tmp_path / "config",
            world_up=[0, -1, 0],
            alternatives=0,
            output=output,
        )
    )
    selected = [
        int(captured[i + 1]) for i, arg in enumerate(captured) if arg == "--group-id"
    ]
    assert selected == [7, 8]
    selection = json.loads((output / "selection.json").read_text())
    assert selection["recovery_added_group_ids"] == [8]
    assert selection["candidate_budget"] == 1
    assert selection["selected_group_ids"] == [7, 8]
    assert selection["deferred_group_ids"] == []


def test_semantics_recovery_has_separate_budget_and_preserves_base_selection(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from farm_runtime.quality import scene_profile, refinement

    geometry, config, validation = recovery_fixture(tmp_path / "source")
    prepare_geometry(
        geometry,
        config,
        tmp_path / "native",
        [0, -1, 0],
        recovery_validation=validation,
    )
    native = tmp_path / "native.json"
    write_json(native, dict(input=describe_file(tmp_path / "native/manifest.json")))
    called = []

    def capture(argv):
        evidence = json.loads(Path(argv[argv.index("--proposals") + 1]).read_text())
        called.extend(g["id"] for g in evidence["groups"])
        out = Path(argv[argv.index("--output") + 1])
        out.mkdir()
        write_json(out / "manifest.json", {})

    monkeypatch.setattr(refinement, "main", capture)
    output = tmp_path / "appearance"
    scene_profile.semantics(
        SimpleNamespace(
            geometry=geometry,
            native=native,
            groups=1,
            model=tmp_path / "model",
            output=output,
        )
    )
    stage = json.loads((output / "manifest.json").read_text())
    assert called == [7, 8]
    assert stage["selected_group_ids"] == [7, 8]
    assert stage["recovery_added_group_ids"] == [8]
    assert stage["candidate_budget"] == 1
    assert stage["deferred_group_ids"] == []
