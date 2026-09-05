from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tracking/materialize_farm_tracker_episodes.py"
SPEC = importlib.util.spec_from_file_location("materialize_tracker_episodes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plan(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    image_root.mkdir()
    sources = {"z.png": b"z-image", "a.png": b"a-image"}
    for name, payload in sources.items():
        (image_root / name).write_bytes(payload)
    frames = [
        {
            "episode_index": index,
            "name": name,
            "physical_timestamp": str(index),
            "colmap_image_id": index + 1,
            "colmap_camera_id": 1,
            "T_world_camera_m": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        }
        for index, name in enumerate(("z.png", "a.png"))
    ]
    plan = {
        "schema": MODULE.PLAN_SCHEMA,
        "inputs": {"image_root": str(image_root)},
        "selected_image_validation": {
            "entries": [
                {
                    "name": name,
                    "bytes": len(payload),
                    "sha256": _digest(payload),
                }
                for name, payload in sources.items()
            ]
        },
        "episodes": [
            {
                "episode_id": "cam00__center__train__000",
                "camera": "cam00",
                "view_family": "center",
                "split": "train",
                "frame_count": 2,
                "frames": frames,
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_materialization_preserves_episode_order_without_rgb_copy(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "episodes"

    manifest = MODULE.materialize(
        plan_path=plan,
        output_root=output,
        verify_hashes=True,
        selected_episode_ids=None,
    )

    root = output / "cam00__center__train__000"
    names = (root / "frames.txt").read_text(encoding="utf-8").splitlines()
    assert names == ["000000__z.png", "000001__a.png"]
    assert all((root / "frames" / name).is_symlink() for name in names)
    assert manifest["frame_count"] == 2
    assert manifest["storage"] == "symlinks-only-no-rgb-copy"


def test_changed_source_fails_and_staging_is_cleaned(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    (tmp_path / "images" / "z.png").write_bytes(b"changed")
    output = tmp_path / "episodes"

    with pytest.raises(ValueError, match="changed since planning"):
        MODULE.materialize(
            plan_path=plan,
            output_root=output,
            verify_hashes=True,
            selected_episode_ids=None,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".episodes.tmp-*"))


def test_unknown_episode_and_existing_output_fail_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(ValueError, match="unknown episode"):
        MODULE.materialize(
            plan_path=plan,
            output_root=tmp_path / "episodes",
            verify_hashes=False,
            selected_episode_ids={"missing"},
        )

    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        MODULE.materialize(
            plan_path=plan,
            output_root=output,
            verify_hashes=False,
            selected_episode_ids=None,
        )

def test_v2_materialization_preserves_pair_and_support_evidence(tmp_path: Path) -> None:
    image_root = tmp_path / "images-v2"
    image_root.mkdir()
    entries, episodes = [], []
    for split, offset in (("train", 0), ("heldout", 10)):
        frames = []
        for index in range(2):
            name = f"cam0_{offset + index:06d}_center.png"
            payload = name.encode("utf-8")
            (image_root / name).write_bytes(payload)
            entries.append(
                {"name": name, "bytes": len(payload), "sha256": _digest(payload)}
            )
            frames.append(
                {
                    "episode_index": index,
                    "name": name,
                    "physical_timestamp": str(offset + index),
                    "colmap_image_id": offset + index + 1,
                    "colmap_camera_id": 1,
                    "T_world_camera_m": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    "is_anchor_preference": index == 1,
                    "is_rescue_plan_anchor": False,
                    "object_evidence": {
                        "canonical_object_id": 7,
                        "retrieval_source": "object_support_colmap_tracks",
                        "bbox_source": "projected_object_support",
                    },
                }
            )
        episodes.append(
            {
                "episode_id": f"object-000007__cam0__center__{split}",
                "pair_id": "object-000007",
                "anchor_object_id": 7,
                "camera": "cam0",
                "view_family": "center",
                "split": split,
                "frame_count": 2,
                "all_frames_object_visible": True,
                "object_evidence_frame_count": 2,
                "frames": frames,
            }
        )
    plan = tmp_path / "plan-v2.json"
    plan.write_text(
        json.dumps(
            {
                "schema": MODULE.PAIRED_PLAN_SCHEMA,
                "inputs": {"image_root": str(image_root)},
                "selected_image_validation": {"entries": entries},
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "episodes-v2"
    manifest = MODULE.materialize(
        plan_path=plan,
        output_root=output,
        verify_hashes=True,
        selected_episode_ids=None,
    )
    assert manifest["source_plan_schema"] == MODULE.PAIRED_PLAN_SCHEMA
    assert {row["anchor_object_id"] for row in manifest["episodes"]} == {7}
    episode = json.loads(
        (output / episodes[0]["episode_id"] / "episode.json").read_text()
    )
    assert episode["pair_id"] == "object-000007"
    assert episode["all_frames_object_visible"] is True
    assert episode["frames"][0]["object_evidence"]["canonical_object_id"] == 7

    with pytest.raises(ValueError, match="break object pairs"):
        MODULE.materialize(
            plan_path=plan,
            output_root=tmp_path / "partial-v2",
            verify_hashes=True,
            selected_episode_ids={episodes[0]["episode_id"]},
        )
