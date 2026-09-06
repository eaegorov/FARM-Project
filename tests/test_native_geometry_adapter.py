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
