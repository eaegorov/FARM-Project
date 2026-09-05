from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_gaussian_grouping_dataset import _fixture, _sha

from farm_runtime.gaussian_grouping_anchor_dataset import (
    OUTPUT_SCHEMA,
    build_anchor_dataset,
)
from farm_runtime.gaussian_grouping_identity import preflight_identity_only

GG_FILES = (
    "train.py",
    "arguments/__init__.py",
    "scene/__init__.py",
    "scene/dataset_readers.py",
    "scene/gaussian_model.py",
    "gaussian_renderer/__init__.py",
    "utils/loss_utils.py",
    "LICENSE",
)


def _provenance(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    return {"path": str(source), "bytes": source.stat().st_size, "sha256": _sha(source)}


def _anchor_fixture(
    tmp_path: Path, *, leaked_timestamp: bool = False
) -> dict[str, object]:
    strict = _fixture(tmp_path, leaked_timestamp=leaked_timestamp)
    colmap = Path(strict["colmap_model"])
    for name in ("cameras.txt", "images.txt", "points3D.bin", "points3D.txt"):
        (colmap / name).write_text(f"fixture {name}\n", encoding="utf-8")
    repo = tmp_path / "gaussian-grouping"
    for relative in GG_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    episodes_root = Path(strict["episodes_root"])
    tracker_runs = strict["tracker_runs"]
    selected_by_episode = {"train-episode": 5, "heldout-episode": 9}
    mapping = {
        "train-episode::local:5": 1,
        "train-episode::local:6": 0,
        "heldout-episode::local:9": 1,
    }
    episode_rows: list[dict[str, object]] = []
    for episode_id, selected_local_id in selected_by_episode.items():
        run = Path(tracker_runs[episode_id])
        signature_path = run / "3d_signature_v1.json"
        signature = json.loads(signature_path.read_text(encoding="utf-8"))
        for identity in signature["local_identities"]:
            selected = identity["local_id"] == selected_local_id
            identity["decision_v2_shadow"] = {
                "schema": "farm.tracker-episode-3d-signatures.v2-shadow",
                "mode": "diagnostic-only-no-publish",
                "status": "would_pass_shadow_gate" if selected else "shadow_rejected",
                "passed": selected,
                "global_association_authorized": False,
            }
        signature_path.write_text(json.dumps(signature), encoding="utf-8")
        episode_path = episodes_root / episode_id / "episode.json"
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        measurement_path = run / "measurement.json"
        measurement = {
            "schema": "farm.sam3-concept-episode.v1",
            "status": "pass",
            "input": {
                "episode_id": episode_id,
                "episode_json_sha256": _sha(episode_path),
                "camera": episode["camera"],
                "view_family": episode["view_family"],
                "split": episode["split"],
                "frame_count": episode["frame_count"],
                "physical_timestamps": [
                    row["physical_timestamp"] for row in episode["frames"]
                ],
            },
            "output": {
                "directory": str(run.resolve()),
                "mask_count": len(signature["inputs"]["mask_files"]),
                "masks": [
                    {
                        "name": Path(row["path"]).name,
                        "bytes": row["bytes"],
                        "sha256": row["sha256"],
                    }
                    for row in signature["inputs"]["mask_files"]
                ],
            },
        }
        measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
        episode_rows.append(
            {
                "episode_id": episode_id,
                "canonical_object_id": 2,
                "split": episode["split"],
                "frame_count": episode["frame_count"],
                "physical_timestamps": [
                    row["physical_timestamp"] for row in episode["frames"]
                ],
                "episode_json": _provenance(episode_path),
                "signature": _provenance(signature_path),
                "tracker_measurement": _provenance(measurement_path),
            }
        )

    manifest_path = episodes_root / "manifest.json"
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(
            {
                "schema": "farm.anchor-cross-fold-reconciliation.v1",
                "status": "pass",
                "mode": "diagnostic-bounded-fail-closed",
                "inputs": {"episodes_manifest": _provenance(manifest_path)},
                "publication": {
                    "raw_tracker_outputs_changed": False,
                    "raw_association_changed": False,
                    "authoritative_v1_changed": False,
                    "shadow_gate_promoted": False,
                    "farm_production_publication_authorized": False,
                    "gaussian_grouping_production_training_authorized": False,
                },
                "objects": [
                    {
                        "object_id": 2,
                        "status": "accepted",
                        "train_episode_id": "train-episode",
                        "heldout_episode_id": "heldout-episode",
                        "timestamp_independence": {"verified": True},
                        "accepted_mapping": {
                            "train_identity_key": "train-episode::local:5",
                            "heldout_identity_key": "heldout-episode::local:9",
                            "raw_global_id": 7,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selected = ["heldout-episode::local:9", "train-episode::local:5"]
    dropped = ["train-episode::local:6"]
    contract_names = {
        "cameras.bin",
        "cameras.txt",
        "images.bin",
        "images.txt",
        "points3D.bin",
        "points3D.txt",
    }
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(
        json.dumps(
            {
                "schema": "farm.gaussian-grouping-anchor-allowlist.v1",
                "status": "pass",
                "scope": "bounded-diagnostic-gaussian-grouping-identity-dataset",
                "publication": {
                    "production_farm_identity_authorized": False,
                    "production_gaussian_grouping_training_authorized": False,
                    "bounded_dataset_materialization_authorized": True,
                    "gpu_training_executed": False,
                    "native_v1_builder_compatible": False,
                    "required_adapter": "consume this exact shadow allowlist without treating v2 shadow as v1 publication",
                },
                "inputs": {
                    "reconciliation": _provenance(reconciliation_path),
                    "source_gaussians": _provenance(Path(strict["source_gaussians"])),
                    "sparse_points_ply": _provenance(colmap / "points3D.ply"),
                    "colmap_model": str(colmap.resolve()),
                    "colmap_contract_files": [
                        {
                            "name": name,
                            "bytes": (colmap / name).stat().st_size,
                            "sha256": _sha(colmap / name),
                        }
                        for name in sorted(contract_names)
                    ],
                },
                "summary": {
                    "canonical_object_count": 1,
                    "episode_count": 2,
                    "foreground_identity_key_count": 2,
                    "explicitly_dropped_identity_key_count": 1,
                    "mapping_key_count": 3,
                },
                "objects": [
                    {
                        "gg_identity_id": 1,
                        "canonical_object_id": 2,
                        "raw_shadow_global_id": 7,
                        "identity_keys": selected,
                        "episode_ids": ["train-episode", "heldout-episode"],
                    }
                ],
                "episodes": sorted(
                    episode_rows, key=lambda row: str(row["episode_id"])
                ),
                "identity_key_to_compact_gg_id": mapping,
                "selected_identity_keys": selected,
                "dropped_identity_keys": dropped,
                "guarantees": [],
            }
        ),
        encoding="utf-8",
    )
    return {
        "episodes_root": episodes_root,
        "allowlist_path": allowlist_path,
        "colmap_model": colmap,
        "source_gaussians": strict["source_gaussians"],
        "sparse_points_ply": colmap / "points3D.ply",
        "gaussian_grouping_repo": repo,
        "output_root": tmp_path / "anchor-dataset",
    }


def test_anchor_adapter_materializes_exact_fg_drop_and_split(tmp_path: Path) -> None:
    kwargs = _anchor_fixture(tmp_path)
    report = build_anchor_dataset(**kwargs)  # type: ignore[arg-type]
    output = Path(kwargs["output_root"])
    assert report["schema"] == OUTPUT_SCHEMA
    assert report["summary"]["frames_by_split"] == {"train": 1, "heldout": 1}
    assert set(np.unique(np.asarray(Image.open(output / "object_mask/train.png")))) == {
        0,
        1,
    }
    assert set(
        np.unique(np.asarray(Image.open(output / "object_mask/heldout.png")))
    ) == {0, 1}
    assert (output / "images_train/train.png").is_symlink()
    assert not (output / "images_train/heldout.png").exists()
    assert (output / "images_heldout/heldout.png").is_symlink()
    config = json.loads((output / "official_gaussian_grouping_config.json").read_text())
    assert config == {
        "densify_until_iter": 0,
        "num_classes": 2,
        "reg3d_interval": 2,
        "reg3d_k": 5,
        "reg3d_lambda_val": 2,
        "reg3d_max_points": 300000,
        "reg3d_sample_size": 1000,
    }
    preflight = preflight_identity_only(
        output, Path(kwargs["gaussian_grouping_repo"]), resolution_factor=4
    )
    assert preflight["status"] == "ready"
    assert preflight["num_classes"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(schema="unsupported"), "unsupported/failed"),
        (
            lambda row: row["identity_key_to_compact_gg_id"].pop(
                "train-episode::local:6"
            ),
            "exactly cover",
        ),
        (
            lambda row: row["identity_key_to_compact_gg_id"].update(
                {"train-episode::local:7": 0}
            ),
            "exactly cover",
        ),
        (
            lambda row: row["selected_identity_keys"].append("train-episode::local:5"),
            "duplicate",
        ),
        (
            lambda row: row["inputs"]["source_gaussians"].update(sha256="0" * 64),
            "source Gaussians provenance mismatch",
        ),
    ],
)
def test_anchor_adapter_rejects_unsafe_allowlist(
    tmp_path: Path, mutation: object, message: str
) -> None:
    kwargs = _anchor_fixture(tmp_path)
    allowlist_path = Path(kwargs["allowlist_path"])
    payload = json.loads(allowlist_path.read_text())
    mutation(payload)  # type: ignore[operator]
    allowlist_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_anchor_dataset(**kwargs)  # type: ignore[arg-type]
    assert not Path(kwargs["output_root"]).exists()
    assert not list(tmp_path.glob(".anchor-dataset.tmp-*"))


def test_anchor_adapter_rejects_heldout_timestamp_leakage(tmp_path: Path) -> None:
    kwargs = _anchor_fixture(tmp_path, leaked_timestamp=True)
    with pytest.raises(ValueError, match="timestamp leakage"):
        build_anchor_dataset(**kwargs)  # type: ignore[arg-type]
    assert not Path(kwargs["output_root"]).exists()
