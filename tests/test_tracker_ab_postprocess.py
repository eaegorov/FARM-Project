from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tracking/postprocess_farm_tracker_ab.py"
SPEC = importlib.util.spec_from_file_location("farm_tracker_ab_postprocess", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    frames = []
    for index, (name, color) in enumerate(
        (("train.png", (255, 0, 0)), ("heldout.png", (0, 0, 255))), start=1
    ):
        path = source_root / name
        Image.fromarray(np.full((3, 4, 3), color, dtype=np.uint8)).save(path)
        frames.append((index, name, path))

    plan = tmp_path / "plan.json"
    _write_json(
        plan,
        {
            "schema": "farm.tracker-subset-plan.v1",
            "inputs": {"meters_per_scene_unit": 1.0},
        },
    )
    plan_hash = MODULE.sha256_file(plan)
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    manifest_rows = []
    episode_ids = (
        "cam00__center__train__000",
        "cam01__center__heldout__000",
    )
    for (image_id, source_name, source_path), episode_id in zip(frames, episode_ids):
        split = "train" if "__train__" in episode_id else "heldout"
        episode_root = episodes / episode_id
        frame_root = episode_root / "frames"
        frame_root.mkdir(parents=True)
        materialized_name = f"000000__{source_name}"
        (frame_root / materialized_name).symlink_to(source_path.resolve())
        (episode_root / "frames.txt").write_text(
            materialized_name + "\n", encoding="utf-8"
        )
        episode = {
            "schema": "farm.materialized-tracker-episode.v1",
            "episode_id": episode_id,
            "camera": episode_id.split("__", 1)[0],
            "view_family": "center",
            "split": split,
            "frame_count": 1,
            "source_plan": str(plan.resolve()),
            "source_plan_sha256": plan_hash,
            "frames": [
                {
                    "episode_index": 0,
                    "materialized_name": materialized_name,
                    "source_name": source_name,
                    "source_path": str(source_path.resolve()),
                    "bytes": source_path.stat().st_size,
                    "sha256": MODULE.sha256_file(source_path),
                    "physical_timestamp": f"{image_id:06d}",
                    "colmap_image_id": image_id,
                    "colmap_camera_id": 1,
                    "T_world_camera_m": np.eye(4).tolist(),
                }
            ],
        }
        _write_json(episode_root / "episode.json", episode)
        manifest_rows.append(
            {
                "episode_id": episode_id,
                "camera": episode["camera"],
                "view_family": "center",
                "split": split,
                "frame_count": 1,
                "relative_path": episode_id,
            }
        )
    manifest = {
        "schema": "farm.materialized-tracker-episodes.v1",
        "source_plan": str(plan.resolve()),
        "source_plan_sha256": plan_hash,
        "hashes_verified": True,
        "episode_count": 2,
        "frame_count": 2,
        "episodes": manifest_rows,
    }
    _write_json(episodes / "manifest.json", manifest)

    colmap = tmp_path / "colmap"
    colmap.mkdir()
    (colmap / "cameras.txt").write_text("1 PINHOLE 4 3 10 10 2 1.5\n", encoding="utf-8")
    (colmap / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 train.png\n"
        "1 1 10 2 1 11 3 1 12\n"
        "2 1 0 0 0 0 0 0 1 heldout.png\n"
        "1 1 10 2 1 11 3 1 12\n",
        encoding="utf-8",
    )
    (colmap / "points3D.txt").write_text(
        "10 0 0 1 255 0 0 0.1\n" "11 0.1 0 1 255 0 0 0.1\n" "12 0.2 0 1 255 0 0 0.1\n",
        encoding="utf-8",
    )

    suites: dict[str, Path] = {}
    materialized_manifest = episodes / "manifest.json"
    for key, config in MODULE.BACKENDS.items():
        suite_root = tmp_path / f"suite-{key}"
        report_rows = []
        for row in manifest_rows:
            episode_id = row["episode_id"]
            episode_root = suite_root / "episodes" / episode_id
            annotation_root = episode_root / "Annotations"
            annotation_root.mkdir(parents=True)
            source_name = "train.png" if row["split"] == "train" else "heldout.png"
            mask_name = f"000000__{Path(source_name).stem}.png"
            mask = np.zeros((3, 4), dtype=np.uint8)
            mask[1:, 1:3] = 1
            Image.fromarray(mask).save(annotation_root / mask_name)
            measurement = {
                "schema": config["episode_schema"],
                "status": "pass",
                "input": {
                    "episode_id": episode_id,
                    "image_directory": str(
                        (episodes / episode_id / "frames").resolve()
                    ),
                },
                "output": {"mask_count": 1},
            }
            if key == "sam3":
                measurement["output"]["frames"] = [
                    {
                        "objects": [
                            {
                                "local_id": 1,
                                "prompt": "cabinet",
                            }
                        ]
                    }
                ]
            measurement_path = episode_root / "measurement.json"
            _write_json(measurement_path, measurement)
            report_rows.append(
                {
                    "episode_id": episode_id,
                    "path": str(measurement_path.resolve()),
                    "sha256": MODULE.sha256_file(measurement_path),
                }
            )
        suite_measurement = {
            "schema": "farm.tracker-episode-suite.v1",
            "status": "pass",
            "backend": config["suite_backend"],
            "input": {
                "materialized_manifest": str(materialized_manifest.resolve()),
                "materialized_manifest_sha256": MODULE.sha256_file(
                    materialized_manifest
                ),
                "source_plan_sha256": plan_hash,
                "exact_frame_hashes_verified": True,
                "episode_count": 2,
            },
            "output": {
                "directory": str(suite_root.resolve()),
                "episode_count": 2,
                "episode_reports": report_rows,
            },
        }
        _write_json(suite_root / "suite_measurement.json", suite_measurement)
        suites[key] = suite_root
    return episodes, colmap, suites


def test_backend_suite_requires_exact_shared_episode_provenance(tmp_path: Path) -> None:
    episodes, _, suites = _build_inputs(tmp_path)
    materialized = MODULE.load_materialized_suite(episodes)
    validated = MODULE.validate_backend_suite(
        suites["deva"], key="deva", materialized=materialized
    )
    assert validated.backend == "sam1_deva_automatic"
    assert [row.episode.split for row in validated.episodes] == [
        "train",
        "heldout",
    ]

    report = suites["deva"] / "suite_measurement.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["input"]["source_plan_sha256"] = "0" * 64
    _write_json(report, payload)
    with pytest.raises(ValueError, match="source-plan checksum mismatch"):
        MODULE.validate_backend_suite(
            suites["deva"], key="deva", materialized=materialized
        )


def test_nested_staging_paths_are_rebased_only_under_atomic_root(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".result.tmp-1"
    destination = tmp_path / "result"
    payload = {
        "path": str(staging / "a" / "report.json"),
        "other": "/external/report.json",
        "nested": [{"path": str(staging / "b.json")}],
    }
    rebased = MODULE._rebase_paths(payload, staging=staging, destination=destination)
    assert rebased["path"] == str(destination / "a" / "report.json")
    assert rebased["other"] == "/external/report.json"
    assert rebased["nested"][0]["path"] == str(destination / "b.json")


def test_end_to_end_cpu_postprocess_is_atomic_and_builder_compatible(
    tmp_path: Path,
) -> None:
    episodes, colmap, suites = _build_inputs(tmp_path)
    output = tmp_path / "postprocess"
    assert (
        MODULE.main(
            [
                "--episodes-root",
                str(episodes),
                "--deva-suite",
                str(suites["deva"]),
                "--sam3-suite",
                str(suites["sam3"]),
                "--colmap-model",
                str(colmap),
                "--output-root",
                str(output),
                "--expected-episodes",
                "2",
                "--jobs",
                "2",
            ]
        )
        == 0
    )
    report = json.loads(
        (output / "postprocess_measurement.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "farm.tracker-ab-3d-postprocess.v2-shadow"
    assert report["execution"]["cpu_only"] is True
    assert report["execution"]["gpu_used"] is False
    assert report["execution"]["signature_count"] == 4
    assert report["execution"]["association_count"] == 2
    assert report["execution"]["shadow_association_ablation_count"] == 2
    assert report["execution"]["association_subprocess_count"] == 4
    assert report["execution"]["automatic_shadow_promotion_allowed"] is False
    assert report["execution"]["shadow_gg_training_authorized"] is False
    assert not list(tmp_path.glob(".postprocess.tmp-*"))
    for association_row in report["outputs"]["associations"]:
        path = Path(association_row["path"])
        assert path.is_file()
        association = json.loads(path.read_text(encoding="utf-8"))
        assert association["fit_splits"] == ["train"]
        assert len(association["local_to_global"]) == 2
        assert set(association["local_to_global"].values()) == {0}
        for source in association["inputs"]["signature_reports"]:
            assert Path(source["path"]).is_file()
            assert ".postprocess.tmp-" not in source["path"]
    shadow_contract = report["v2_shadow_ablation"]
    assert shadow_contract["mode"] == "diagnostic-only-no-publish"
    assert shadow_contract["fit_splits"] == ["train"]
    assert shadow_contract["automatic_promotion_allowed"] is False
    assert shadow_contract["gg_training_authorized"] is False
    assert len(shadow_contract["by_backend"]) == 2
    for row in shadow_contract["by_backend"]:
        assert row["heldout_updates_fit"] is False
        assert set(row["identity_counts_by_split"]) == {"train", "heldout"}
        assert set(row["potential_pairs"]) == {
            "train_fit_pairs_evaluated",
            "train_fit_eligible_pairs",
            "heldout_application_pairs_evaluated",
            "heldout_application_eligible_pairs",
        }
    for association_row in report["outputs"]["shadow_association_ablations"]:
        path = Path(association_row["path"])
        assert path.is_file()
        association = json.loads(path.read_text(encoding="utf-8"))
        assert association["schema"] == (
            "farm.tracker-global-association.v2-shadow-ablation.v1"
        )
        assert association["fit_splits"] == ["train"]
        assert association["automatic_promotion_allowed"] is False
        assert association["gg_training_authorized"] is False
        assert "local_to_global" not in association
        assert "assignments" not in association
        assert "global_objects" not in association
        assert len(association["candidate_local_to_global"]) == 2
        for source in association["inputs"]["signature_reports"]:
            assert Path(source["path"]).is_file()
            assert ".postprocess.tmp-" not in source["path"]


def test_single_sam3_postprocess_keeps_atomic_schema_and_dynamic_counts(
    tmp_path: Path,
) -> None:
    episodes, colmap, suites = _build_inputs(tmp_path)
    output = tmp_path / "postprocess-sam3"
    assert (
        MODULE.main(
            [
                "--episodes-root",
                str(episodes),
                "--sam3-suite",
                str(suites["sam3"]),
                "--colmap-model",
                str(colmap),
                "--output-root",
                str(output),
                "--expected-episodes",
                "2",
                "--jobs",
                "2",
            ]
        )
        == 0
    )
    report = json.loads(
        (output / "postprocess_measurement.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "farm.tracker-ab-3d-postprocess.v2-shadow"
    assert report["execution"]["backend_mode"] == "single"
    assert report["execution"]["backend_count"] == 1
    assert report["execution"]["backend_keys"] == ["sam3"]
    assert report["execution"]["signature_count"] == 2
    assert report["execution"]["association_count"] == 1
    assert report["execution"]["shadow_association_ablation_count"] == 1
    assert report["execution"]["association_subprocess_count"] == 2
    assert set(report["input"]["tracker_suites"]) == {"sam3"}
    assert len(report["outputs"]["associations"]) == 1
    assert len(report["outputs"]["shadow_association_ablations"]) == 1
    assert len(report["v2_shadow_ablation"]["by_backend"]) == 1
    assert report["v2_shadow_ablation"]["by_backend"][0]["backend"] == "sam3"
    assert not list(tmp_path.glob(".postprocess-sam3.tmp-*"))


def test_postprocess_rejects_no_tracker_backend_before_touching_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="at least one of --deva-suite or --sam3-suite"):
        MODULE.main(
            [
                "--episodes-root",
                str(tmp_path / "missing-episodes"),
                "--colmap-model",
                str(tmp_path / "missing-colmap"),
                "--output-root",
                str(output),
            ]
        )
    assert not output.exists()
