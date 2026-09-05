from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from farm_runtime.gaussian_grouping_dataset import (
    ASSOCIATION_SCHEMA,
    OUTPUT_SCHEMA,
    build_dataset,
)
from farm_runtime.gaussian_grouping_identity import preflight_identity_only


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_farm_gaussian_grouping_identity_only",
    ROOT / "scripts/grouping/run_farm_gaussian_grouping_identity_only.py",
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_farm_gaussian_grouping_dataset",
    ROOT / "scripts/grouping/build_farm_gaussian_grouping_dataset.py",
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_colmap(root: Path, rows: list[tuple[int, str]]) -> None:
    root.mkdir(parents=True)
    with (root / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<iiQQ", 1, 1, 4, 3))
        stream.write(struct.pack("<dddd", 2.25, 2.5, 1.75, 1.25))
    with (root / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(rows)))
        for image_id, name in rows:
            stream.write(
                struct.pack(
                    "<idddddddi",
                    image_id,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    float(image_id) / 10,
                    -0.25,
                    2.0,
                    1,
                )
            )
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", 0))
    (root / "points3D.ply").write_bytes(b"ply\nformat ascii 1.0\nend_header\n")


def _write_gaussian_ply(path: Path) -> None:
    properties = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *[f"f_rest_{index}" for index in range(45)],
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = ["ply", "format ascii 1.0", "element vertex 1"]
    header.extend(f"property float {name}" for name in properties)
    values = {
        "opacity": "2",
        "scale_0": "-4",
        "scale_1": "-4",
        "scale_2": "-4",
        "rot_0": "1",
    }
    header.extend(["end_header", " ".join(values.get(name, "0") for name in properties)])
    path.write_text("\n".join(header) + "\n", encoding="ascii")


def _fixture(tmp_path: Path, *, leaked_timestamp: bool = False) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_plan = tmp_path / "plan.json"
    source_plan.write_text('{"schema":"farm.tracker-subset-plan.v1"}\n', encoding="utf-8")
    image_root = tmp_path / "source-images"
    image_root.mkdir()
    colmap = tmp_path / "colmap"
    _write_colmap(colmap, [(11, "train.png"), (12, "heldout.png")])
    definitions = [
        ("train-episode", "train", "train.png", 11, "100", [5, 6]),
        (
            "heldout-episode",
            "heldout",
            "heldout.png",
            12,
            "100" if leaked_timestamp else "200",
            [9],
        ),
    ]
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    tracker_runs: dict[str, Path] = {}
    association_mapping: dict[str, int] = {}
    signature_reports: list[dict[str, object]] = []
    manifest_episodes = []
    for episode_id, split, name, image_id, timestamp, local_ids in definitions:
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[..., 0] = image_id
        source = image_root / name
        Image.fromarray(rgb).save(source)
        episode_root = episodes_root / episode_id
        (episode_root / "frames").mkdir(parents=True)
        materialized = f"000000__{name}"
        os.symlink(str(source.resolve()), episode_root / "frames" / materialized)
        episode = {
            "schema": "farm.materialized-tracker-episode.v1",
            "episode_id": episode_id,
            "split": split,
            "camera": "cam00",
            "view_family": "center",
            "frame_count": 1,
            "source_plan": str(source_plan.resolve()),
            "source_plan_sha256": _sha(source_plan),
            "frames": [
                {
                    "episode_index": 0,
                    "materialized_name": materialized,
                    "source_name": name,
                    "source_path": str(source.resolve()),
                    "bytes": source.stat().st_size,
                    "sha256": _sha(source),
                    "physical_timestamp": timestamp,
                    "colmap_image_id": image_id,
                    "colmap_camera_id": 1,
                }
            ],
        }
        episode_path = episode_root / "episode.json"
        episode_path.write_text(json.dumps(episode), encoding="utf-8")
        manifest_episodes.append(
            {
                "episode_id": episode_id,
                "camera": "cam00",
                "view_family": "center",
                "split": split,
                "frame_count": 1,
                "relative_path": episode_id,
            }
        )
        run = tmp_path / "runs" / episode_id
        annotations = run / "Annotations"
        annotations.mkdir(parents=True)
        mask = np.zeros((3, 4), dtype=np.uint8)
        for offset, local_id in enumerate(local_ids):
            mask[:, offset : offset + 1] = local_id
            association_mapping[f"{episode_id}::local:{local_id}"] = 1 + offset
        mask_path = annotations / materialized
        Image.fromarray(mask).save(mask_path)
        qc = {
            "schema": "farm.tracker-episode-3d-signatures.v1",
            "status": "pass",
            "input_contract": {"passed": True},
            "inputs": {
                "episode": {
                    "path": str(episode_path.resolve()),
                    "bytes": episode_path.stat().st_size,
                    "sha256": _sha(episode_path),
                },
                "episode_id": episode_id,
                "camera": "cam00",
                "view_family": "center",
                "split": split,
                "mask_directory": str(annotations.resolve()),
                "mask_files": [
                    {
                        "path": str(mask_path.resolve()),
                        "bytes": mask_path.stat().st_size,
                        "sha256": _sha(mask_path),
                        "shape_hw": [3, 4],
                    }
                ],
                "colmap_model": str(colmap.resolve()),
                "colmap_contract_files": [
                    {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": _sha(path),
                    }
                    for path in (colmap / "cameras.bin", colmap / "images.bin")
                ],
                "source_plan": {
                    "path": str(source_plan.resolve()),
                    "bytes": source_plan.stat().st_size,
                    "sha256": _sha(source_plan),
                },
            },
            "local_identities": [
                {
                    "identity_key": f"{episode_id}::local:{local_id}",
                    "episode_id": episode_id,
                    "local_id": local_id,
                    "decision": {"passed": True},
                }
                for local_id in local_ids
            ],
        }
        qc_path = run / "3d_signature_v1.json"
        qc_path.write_text(json.dumps(qc), encoding="utf-8")
        signature_reports.append(
            {
                "path": str(qc_path.resolve()),
                "bytes": qc_path.stat().st_size,
                "sha256": _sha(qc_path),
                "schema": qc["schema"],
                "episode_id": episode_id,
                "split": split,
            }
        )
        tracker_runs[episode_id] = run
    episodes_manifest = {
        "schema": "farm.materialized-tracker-episodes.v1",
        "source_plan": str(source_plan.resolve()),
        "source_plan_sha256": _sha(source_plan),
        "hashes_verified": True,
        "episode_count": len(manifest_episodes),
        "frame_count": sum(row["frame_count"] for row in manifest_episodes),
        "episodes": manifest_episodes,
    }
    (episodes_root / "manifest.json").write_text(
        json.dumps(episodes_manifest), encoding="utf-8"
    )
    association = tmp_path / "association.json"
    association.write_text(
        json.dumps(
            {
                "schema": ASSOCIATION_SCHEMA,
                "fit_splits": ["train"],
                "local_to_global": association_mapping,
                "inputs": {"signature_reports": signature_reports},
                "assignments": [
                    {
                        "identity_key": identity_key,
                        "episode_id": identity_key.rsplit("::local:", 1)[0],
                        "local_id": int(identity_key.rsplit("::local:", 1)[1]),
                        "split": (
                            "train" if identity_key.startswith("train-episode::") else "heldout"
                        ),
                        "global_id": global_id,
                        "status": "assigned",
                        "reason": "test_fixture",
                    }
                    for identity_key, global_id in association_mapping.items()
                ],
                "global_objects": [
                    {"global_id": global_id}
                    for global_id in sorted(set(association_mapping.values()))
                ],
            }
        ),
        encoding="utf-8",
    )
    gaussians = tmp_path / "factory.ply"
    _write_gaussian_ply(gaussians)
    return {
        "episodes_root": episodes_root,
        "tracker_runs": tracker_runs,
        "association_path": association,
        "colmap_model": colmap,
        "source_gaussians": gaussians,
        "output_root": tmp_path / "dataset",
    }


def test_builds_native_dataset_with_exact_split_and_global_ids(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    report = build_dataset(**kwargs)  # type: ignore[arg-type]
    output = Path(kwargs["output_root"])

    assert report["schema"] == OUTPUT_SCHEMA
    assert report["summary"]["frames_by_split"] == {"train": 1, "heldout": 1}
    assert report["summary"]["physical_timestamp_leakage_count"] == 0
    assert report["summary"]["num_classes_including_background"] == 3
    assert (output / "images/train.png").is_symlink()
    assert (output / "images_train/train.png").is_symlink()
    assert not (output / "images_train/heldout.png").exists()
    assert (output / "images_heldout/heldout.png").is_symlink()
    assert (output / "source_gaussians.ply").is_symlink()
    assert (output / "sparse/0/points3D.ply").is_symlink()
    assert not (output / "images/train.png").readlink().is_absolute()
    assert not (output / "source_gaussians.ply").readlink().is_absolute()
    train_mask = np.asarray(Image.open(output / "object_mask/train.png"))
    heldout_mask = np.asarray(Image.open(output / "object_mask/heldout.png"))
    assert set(np.unique(train_mask)) == {0, 1, 2}
    assert set(np.unique(heldout_mask)) == {0, 1}
    cameras_text = (output / "sparse/0/cameras.txt").read_text(encoding="utf-8")
    images_text = (output / "sparse/0/images.txt").read_text(encoding="utf-8")
    assert "1 PINHOLE 4 3 2.25 2.5 1.75 1.25" in cameras_text
    assert "11 1 0 0 0 1.1000000000000001 -0.25 2 1 train.png" in images_text
    assert "12 1 0 0 0 1.2 -0.25 2 1 heldout.png" in images_text
    assert report["artifacts"]["cameras_txt"]["sha256"] == _sha(
        output / "sparse/0/cameras.txt"
    )
    assert report["artifacts"]["images_txt"]["sha256"] == _sha(
        output / "sparse/0/images.txt"
    )


def test_cli_derives_exact_tracker_runs_from_association(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    assert BUILDER.main(
        [
            "--episodes-root",
            str(kwargs["episodes_root"]),
            "--association",
            str(kwargs["association_path"]),
            "--colmap-model",
            str(kwargs["colmap_model"]),
            "--source-gaussians",
            str(kwargs["source_gaussians"]),
            "--output-root",
            str(kwargs["output_root"]),
        ]
    ) == 0
    assert Path(kwargs["output_root"]).joinpath("manifest.json").is_file()


def test_rejects_incomplete_association_and_cleans_staging(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    association = Path(kwargs["association_path"])
    payload = json.loads(association.read_text())
    payload["local_to_global"].pop("train-episode::local:6")
    association.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="association identity coverage mismatch"):
        build_dataset(**kwargs)  # type: ignore[arg-type]

    assert not Path(kwargs["output_root"]).exists()
    assert not list(tmp_path.glob(".dataset.tmp-*"))


def test_rejects_disagreeing_global_objects(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    association = Path(kwargs["association_path"])
    payload = json.loads(association.read_text())
    payload["global_objects"] = [{"global_id": 1}]
    association.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="global_objects disagree"):
        build_dataset(**kwargs)  # type: ignore[arg-type]


def test_rejects_heldout_fit_and_timestamp_leakage(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "fit")
    association = Path(first["association_path"])
    payload = json.loads(association.read_text())
    payload["fit_splits"] = ["train", "heldout"]
    association.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fit_splits"):
        build_dataset(**first)  # type: ignore[arg-type]

    second = _fixture(tmp_path / "leak", leaked_timestamp=True)
    with pytest.raises(ValueError, match="timestamp leakage"):
        build_dataset(**second)  # type: ignore[arg-type]


def test_rejects_noncompact_or_out_of_range_global_ids(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    association = Path(kwargs["association_path"])
    payload = json.loads(association.read_text())
    payload["local_to_global"]["train-episode::local:6"] = 254
    next(
        row
        for row in payload["assignments"]
        if row["identity_key"] == "train-episode::local:6"
    )["global_id"] = 254
    association.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="compact and contiguous"):
        build_dataset(**kwargs)  # type: ignore[arg-type]

    payload["local_to_global"]["train-episode::local:6"] = 255
    next(
        row
        for row in payload["assignments"]
        if row["identity_key"] == "train-episode::local:6"
    )["global_id"] = 255
    association.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        build_dataset(**kwargs)  # type: ignore[arg-type]



def test_identity_only_preflight_proves_frozen_source_and_split_contract(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    build_dataset(**kwargs)  # type: ignore[arg-type]
    repository = tmp_path / "gaussian-grouping"
    for relative in (
        "train.py",
        "gaussian_renderer/__init__.py",
        "scene/gaussian_model.py",
        "utils/loss_utils.py",
        "LICENSE",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test\n", encoding="utf-8")

    report = preflight_identity_only(
        Path(kwargs["output_root"]), repository, resolution_factor=4
    )

    assert report["status"] == "ready"
    assert report["mode"] == "identity-only-frozen-geometry"
    assert report["source_gaussians"]["vertex_count"] == 1
    assert report["num_classes"] == 3
    assert report["densification_enabled"] is False
    assert report["combined_ply_written_by_default"] is False

    run_output = tmp_path / "preflight-run"
    assert RUNNER.main(
        [
            "--dataset-root",
            str(kwargs["output_root"]),
            "--gaussian-grouping-repo",
            str(repository),
            "--output-root",
            str(run_output),
            "--preflight-only",
        ]
    ) == 0
    run = json.loads((run_output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "preflight_passed_training_not_run"
    assert not (run_output / "identity_features.pt").exists()
