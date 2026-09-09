import copy
import json

import numpy as np
import pytest

from farm_runtime.quality.registered_frames import merge_indices, verify_scene_sources


def frame(name, timestamp, camera="cam0"):
    return dict(
        source_image=name,
        frame_id=timestamp,
        timestamp_ns=100,
        camera=camera,
        depth_size=[10, 12],
        K=np.eye(3).tolist(),
        T_world_cam=np.eye(4).tolist(),
        rgb_path=name + ".jpg",
        depth_path=name + ".npy",
    )


def index(rows):
    return dict(
        depth_units="metres",
        pose_translation_units="metres",
        meters_per_scene_unit=1.0,
        frames=rows,
        camera_registration=dict(
            groups=[dict(trusted=True, source_images=[r["source_image"] for r in rows])]
        ),
    )


def test_extension_preserves_original_cameras_and_shares_new_physical_timestamp(
    tmp_path,
):
    old = index([frame("a", "t1"), dict(frame("b", "t2"), timestamp_ns=200)])
    new = index(
        [
            dict(frame("a", "t1"), timestamp_ns=900),
            frame("c", "t3"),
            frame("d", "t3", camera="cam1"),
            frame("e", "t2", camera="cam1"),
        ]
    )
    frozen = copy.deepcopy(old)
    merged, anchors = merge_indices(
        old, new, tmp_path / "old", tmp_path / "new", ["c", "d", "e"]
    )
    rows = {r["source_image"]: r for r in merged["frames"]}
    assert anchors == ["a"] and old == frozen
    assert merged["cameras"] == ["cam0", "cam1"]
    assert rows["a"]["timestamp_ns"] == 100 and rows["b"]["timestamp_ns"] == 200
    assert rows["c"]["timestamp_ns"] == rows["d"]["timestamp_ns"] > 200
    assert rows["e"]["timestamp_ns"] == 200
    assert rows["a"]["K"] == old["frames"][0]["K"]
    assert rows["c"]["depth_path"] == str(tmp_path / "new/c.npy")


def test_extension_remaps_additional_alias_and_keeps_original_capture_key(tmp_path):
    old = index([dict(frame("a", "t1"), physical_timestamp="42")])
    new = index([
        dict(frame("a", "t1"), timestamp_ns=900, physical_timestamp="900"),
        dict(frame("b", "t1", camera="cam1"), timestamp_ns=900, physical_timestamp="900"),
        dict(frame("c", "t2"), timestamp_ns=100, physical_timestamp="100"),
        dict(frame("d", "t2", camera="cam1"), timestamp_ns=100, physical_timestamp="100"),
    ])
    merged, _ = merge_indices(old, new, tmp_path, tmp_path, ["b", "c", "d"])
    rows = {r["source_image"]: r for r in merged["frames"]}
    assert rows["a"]["physical_timestamp"] == rows["b"]["physical_timestamp"] == "42"
    assert rows["c"]["physical_timestamp"] == rows["d"]["physical_timestamp"] == str(rows["c"]["timestamp_ns"])
    assert len({r["physical_timestamp"] for r in rows.values()}) == 2


@pytest.mark.parametrize(
    "fault",
    [
        "scale",
        "anchor_pose",
        "untrusted",
        "no_anchor",
        "duplicate",
        "timestamp_collision",
        "identity_contract",
    ],
)
def test_invalid_extensions_do_not_create_a_merged_index(tmp_path, fault):
    old = index([frame("a", "t1")])
    new = index([frame("a", "t1"), frame("c", "t3")])
    if fault == "scale":
        new["meters_per_scene_unit"] = 2.0
    elif fault == "anchor_pose":
        new["frames"][0]["T_world_cam"][0][3] = 0.001
    elif fault == "untrusted":
        new["camera_registration"]["groups"][0]["trusted"] = False
    elif fault == "no_anchor":
        new["frames"] = new["frames"][1:]
    elif fault == "duplicate":
        new["frames"].append(copy.deepcopy(new["frames"][1]))
    elif fault == "identity_contract":
        new["identity_contract"] = {"regex": "different"}
    elif fault == "timestamp_collision":
        old["frames"].append(frame("b", "t2"))
    with pytest.raises(ValueError):
        merge_indices(old, new, tmp_path, tmp_path, ["c"])


def test_scene_source_optional_hash_metadata_does_not_hide_content_change(tmp_path):
    source = tmp_path / "scene.ply"
    source.write_bytes(b"original source")
    stat = source.stat()
    row = dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    old = dict(fingerprint_payload=dict(inputs=dict(ply=row, colmap=[])))
    new = copy.deepcopy(old)
    verified = verify_scene_sources(old, new)
    old["fingerprint_payload"]["inputs"]["ply"]["sha256"] = verified[0]["sha256"]
    assert verify_scene_sources(old, new) == verified
    old["fingerprint_payload"]["inputs"]["ply"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content changed"):
        verify_scene_sources(old, new)
