from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from farm_runtime.full_colmap_folds import build_full_colmap_fold_views


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[str], list[str]]:
    union = tmp_path / "rgbd_union"
    (union / "rgb").mkdir(parents=True)
    (union / "depth").mkdir()
    train = [f"cam00_{value:06d}_center.png" for value in (1, 2, 3)]
    heldout = [f"cam01_{value:06d}_yaw_left.png" for value in (11, 12)]
    selected = train + heldout
    frames = []
    source_rows = []
    train_views = []
    heldout_views = []
    for index, name in enumerate(selected):
        rgb = f"rgb/{index:06d}.jpg"
        depth = f"depth/{index:06d}.npy"
        (union / rgb).write_bytes(f"rgb-{name}".encode())
        (union / depth).write_bytes(f"depth-{name}".encode())
        split = "train" if name in train else "heldout"
        timestamp = name.split("_")[1]
        frames.append(
            {
                "frame_id": timestamp,
                "camera": name.split("_")[0] + "_center",
                "rgb_path": rgb,
                "depth_path": depth,
                "depth_size": [2, 2],
                "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "T_world_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "source_image": name,
            }
        )
        source_rows.append(
            {"name": name, "split": split, "bytes": 10, "sha256": _sha256(name.encode())}
        )
        view = {"name": name, "physical_timestamp": timestamp, "split": split}
        (train_views if split == "train" else heldout_views).append(view)
    frames_path = union / "frames.json"
    frames_path.write_text(
        json.dumps(
            {
                "schema_version": "farm_frames_json_v1",
                "scene_id": "test",
                "meters_per_scene_unit": 1.0,
                "cameras": ["cam00_center", "cam01_center"],
                "selection_contract": {"count": 5, "exact_order_preserved": True},
                "frames": frames,
            }
        )
    )
    plan = {
        "schema": "farm.full-colmap-rescue-plan.v1",
        "selected_names_role": "rgbd_render_union_of_train_and_heldout",
        "meters_per_scene_unit": 1.0,
        "policy": {
            "minimum_train_views_per_object": 3,
            "minimum_heldout_views_per_object": 2,
        },
        "unique_rescue_views": 5,
        "unique_train_views": 3,
        "unique_heldout_views": 2,
        "selected_names": selected,
        "train_names": train,
        "heldout_names": heldout,
        "provenance": {
            "selected_image_hashes_enabled": True,
            "selected_source_images": source_rows,
        },
        "objects": [
            {
                "object_id": 7,
                "planning_status": "planned",
                "selected_count": 3,
                "selected_views": train_views,
                "train_views": train_views,
                "heldout_count": 2,
                "heldout_views": heldout_views,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan_path, frames_path, train, heldout


def _heldout_reference_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, list[str]]:
    plan_path, frames_path, _, heldout = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan.update(
        {
            "selected_names_role": "heldout_reference_only_non_merge",
            "unique_rescue_views": len(heldout),
            "unique_train_views": 0,
            "unique_heldout_views": len(heldout),
            "selected_names": heldout,
            "train_names": [],
            "heldout_names": heldout,
            "consumption_contract": {
                "train": "train_unconsumed",
                "heldout": "heldout_reference_only",
                "state_fit_authorized": False,
                "geometry_merge_authorized": False,
                "semantic_mutation_authorized": False,
            },
        }
    )
    plan["policy"].update(
        {
            "heldout_reference_only": True,
            "minimum_train_views_per_object": 0,
            "state_fit_authorized": False,
            "geometry_merge_authorized": False,
            "semantic_mutation_authorized": False,
        }
    )
    plan["provenance"]["selected_source_images"] = [
        row
        for row in plan["provenance"]["selected_source_images"]
        if row["name"] in heldout
    ]
    object_row = plan["objects"][0]
    object_row.update(
        {
            "planning_status": "planned_heldout_reference_only",
            "selected_count": 0,
            "selected_views": [],
            "train_views": [],
        }
    )
    plan_path.write_text(json.dumps(plan))

    frames = json.loads(frames_path.read_text())
    frames["frames"] = [
        row for row in frames["frames"] if row["source_image"] in heldout
    ]
    frames["selection_contract"]["count"] = len(heldout)
    frames_path.write_text(json.dumps(frames))
    return plan_path, frames_path, heldout


def test_builds_manifest_only_disjoint_views_with_verified_relative_paths(
    tmp_path: Path,
) -> None:
    plan, union, train, heldout = _fixture(tmp_path)
    output = tmp_path / "rgbd_folds"
    report = build_full_colmap_fold_views(plan, union, output)
    assert report["status"] == "PASS"
    assert report["counts"] == {
        "union_frames": 5,
        "train_frames": 3,
        "heldout_frames": 2,
        "train_physical_timestamps": 3,
        "heldout_physical_timestamps": 2,
        "objects": 1,
    }
    assert report["fit_policy"]["fit_splits"] == ["train"]
    assert report["fit_policy"]["heldout_consumed"] is False
    assert not (output / "train/rgb").exists()
    assert not (output / "heldout/depth").exists()
    for split, expected in (("train", train), ("heldout", heldout)):
        payload = json.loads((output / split / "frames.json").read_text())
        assert [row["source_image"] for row in payload["frames"]] == expected
        assert payload["full_colmap_fold"]["state_fit_authorized"] is (split == "train")
        for row in payload["frames"]:
            assert not Path(row["rgb_path"]).is_absolute()
            assert (output / split / row["rgb_path"]).resolve(strict=True).is_file()
            assert (output / split / row["depth_path"]).resolve(strict=True).is_file()
    assert set(report["provenance"]["rendered_artifacts"]) == {"train", "heldout"}
    assert all(
        len(row[kind]["sha256"]) == 64
        for split in ("train", "heldout")
        for row in report["provenance"]["rendered_artifacts"][split]
        for kind in ("rgb", "depth")
    )


def test_builds_explicit_heldout_reference_only_folds_with_empty_train(
    tmp_path: Path,
) -> None:
    plan, union, heldout = _heldout_reference_fixture(tmp_path)
    output = tmp_path / "heldout_reference_folds"

    report = build_full_colmap_fold_views(plan, union, output)

    assert report["status"] == "PASS"
    assert report["counts"]["train_frames"] == 0
    assert report["counts"]["heldout_frames"] == 2
    assert report["fit_policy"] == {
        "fit_splits": [],
        "train_usage": "train_unconsumed",
        "heldout_usage": "heldout_reference_only",
        "heldout_consumed": False,
        "state_fit_authorized": False,
        "geometry_merge_authorized": False,
        "semantic_mutation_authorized": False,
        "minimum_accepted_views": 3,
        "minimum_independent_physical_timestamps": 2,
    }
    train = json.loads((output / "train/frames.json").read_text())
    heldout_document = json.loads((output / "heldout/frames.json").read_text())
    assert train["frames"] == []
    assert train["full_colmap_fold"]["role"] == "train_unconsumed"
    assert train["full_colmap_fold"]["state_fit_authorized"] is False
    assert heldout_document["full_colmap_fold"]["role"] == (
        "heldout_reference_only"
    )
    assert heldout_document["full_colmap_fold"]["state_fit_authorized"] is False
    assert heldout_document["full_colmap_fold"]["geometry_merge_authorized"] is False
    assert (
        heldout_document["full_colmap_fold"]["semantic_mutation_authorized"]
        is False
    )
    assert [row["source_image"] for row in heldout_document["frames"]] == heldout


def test_heldout_reference_fold_rejects_any_train_name_before_writing(
    tmp_path: Path,
) -> None:
    plan_path, frames_path, heldout = _heldout_reference_fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["train_names"] = [heldout[0]]
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "heldout_reference_folds"

    with pytest.raises(ValueError, match="must have no train names"):
        build_full_colmap_fold_views(plan_path, frames_path, output)

    assert not output.exists()


def test_rejects_physical_timestamp_leakage_before_creating_output(tmp_path: Path) -> None:
    plan_path, frames_path, _, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["objects"][0]["heldout_views"][0]["physical_timestamp"] = "000001"
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "folds"
    with pytest.raises(ValueError, match="timestamp leakage"):
        build_full_colmap_fold_views(plan_path, frames_path, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_requires_exact_union_source_image_coverage(
    tmp_path: Path, mutation: str
) -> None:
    plan_path, frames_path, _, _ = _fixture(tmp_path)
    frames = json.loads(frames_path.read_text())
    if mutation == "missing":
        frames["frames"].pop()
        frames["selection_contract"]["count"] -= 1
    elif mutation == "extra":
        row = dict(frames["frames"][0])
        row["source_image"] = "cam99_999999_center.png"
        frames["frames"].append(row)
        frames["selection_contract"]["count"] += 1
    else:
        frames["frames"][-1]["source_image"] = frames["frames"][0]["source_image"]
    frames_path.write_text(json.dumps(frames))
    with pytest.raises(ValueError, match="cover|duplicate"):
        build_full_colmap_fold_views(plan_path, frames_path, tmp_path / "folds")


def test_rejects_a_heldout_row_disguised_as_selected_fit_view(tmp_path: Path) -> None:
    plan_path, frames_path, _, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["objects"][0]["selected_views"].append(
        dict(plan["objects"][0]["heldout_views"][0])
    )
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="selected_views must be exactly"):
        build_full_colmap_fold_views(plan_path, frames_path, tmp_path / "folds")


def test_rejects_artifact_path_that_escapes_union(tmp_path: Path) -> None:
    plan_path, frames_path, _, _ = _fixture(tmp_path)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    frames = json.loads(frames_path.read_text())
    frames["frames"][0]["rgb_path"] = "../outside.jpg"
    frames_path.write_text(json.dumps(frames))
    with pytest.raises(ValueError, match="escapes"):
        build_full_colmap_fold_views(plan_path, frames_path, tmp_path / "folds")
