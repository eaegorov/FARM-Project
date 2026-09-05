from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.frozen_heldout_qc import (
    _load_mask,
    evaluate_frozen_heldout_qc,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _spec(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": os.path.relpath(path, relative_to),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _fixture(
    tmp_path: Path,
    *,
    evidence_sources: tuple[str, ...] = ("held_a.png", "held_c.png"),
    missing_heldout_scale: bool = False,
    bad_projection: bool = False,
) -> tuple[Path, Path, Path]:
    folds = tmp_path / "folds"
    train_dir = folds / "train"
    heldout_dir = folds / "heldout"
    train_dir.mkdir(parents=True)
    heldout_dir.mkdir()
    train_rows = [
        ("train_a.png", "10"),
        ("train_b.png", "20"),
    ]
    heldout_rows = [
        ("held_a.png", "100"),
        ("held_b.png", "100"),
        ("held_c.png", "200"),
    ]

    def frames_document(role: str, rows: list[tuple[str, str]]) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": "farm_frames_json_v1",
            "depth_units": "metres",
            "meters_per_scene_unit": 1.0,
            "frames": [
                {
                    "frame_id": str(index),
                    "source_image": source,
                    "physical_timestamp": timestamp,
                    "depth_size": [8, 8],
                    "depth_path": f"depth_{source}.npy",
                }
                for index, (source, timestamp) in enumerate(rows)
            ],
            "full_colmap_fold": {
                "schema": "farm.full-colmap-rgbd-fold.v1",
                "role": role,
                "state_fit_authorized": role == "train",
                "usage": (
                    "mapping_covisibility_sam3_merge_obb_fit"
                    if role == "train"
                    else "frozen_evaluation_only"
                ),
            },
        }
        return document

    train_document = frames_document("train", train_rows)
    heldout_document = frames_document("heldout", heldout_rows)
    if missing_heldout_scale:
        heldout_document.pop("meters_per_scene_unit")
    for source, _ in train_rows:
        np.save(train_dir / f"depth_{source}.npy", np.ones((8, 8), np.float32))
    heldout_artifacts = []
    for source, timestamp in heldout_rows:
        depth_path = heldout_dir / f"depth_{source}.npy"
        np.save(depth_path, np.ones((8, 8), np.float32))
        heldout_artifacts.append(
            {
                "source_image": source,
                "physical_timestamp": timestamp,
                "depth": _spec(depth_path, relative_to=heldout_dir),
            }
        )
    train_path = train_dir / "frames.json"
    heldout_path = heldout_dir / "frames.json"
    _write_json(train_path, train_document)
    _write_json(heldout_path, heldout_document)
    manifest = {
        "schema": "farm.full-colmap-rgbd-folds.v1",
        "status": "PASS",
        "integrity": {
            "source_image_sets_disjoint": True,
            "physical_timestamp_sets_disjoint": True,
            "all_rendered_artifacts_sha256_hashed": True,
        },
        "fit_policy": {
            "fit_splits": ["train"],
            "heldout_usage": "frozen_evaluation_only",
            "heldout_consumed": False,
        },
        "folds": {
            "train": {
                "frames_json": "train/frames.json",
                "sha256": _sha(train_path),
                "source_images": [row[0] for row in train_rows],
            },
            "heldout": {
                "frames_json": "heldout/frames.json",
                "sha256": _sha(heldout_path),
                "source_images": [row[0] for row in heldout_rows],
            },
        },
        "object_membership": {
            "7": {
                "train": [row[0] for row in train_rows],
                "heldout": [row[0] for row in heldout_rows],
            }
        },
        "provenance": {
            "rendered_artifacts": {
                "train": [],
                "heldout": heldout_artifacts,
            }
        },
    }
    manifest_path = folds / "manifest.json"
    _write_json(manifest_path, manifest)

    candidate_array = tmp_path / "candidate_ids.npy"
    np.save(candidate_array, np.array([7, 7, -1], np.int32))
    candidate = {
        "schema": "farm.frozen-heldout-candidate.v1",
        "status": "frozen",
        "meters_per_scene_unit": 1.0,
        "fold_manifest_sha256": _sha(manifest_path),
        "train_frames_sha256": _sha(train_path),
        "fit_contract": {
            "fit_splits": ["train"],
            "heldout_consumed": False,
            "heldout_updates_candidate": False,
            "frozen_before_heldout_evaluation": True,
            "fit_source_images": [row[0] for row in train_rows],
            "fit_physical_timestamps": [row[1] for row in train_rows],
        },
        "objects": [
            {
                "object_id": 7,
                "artifacts": [_spec(candidate_array, relative_to=tmp_path)],
            }
        ],
    }
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)

    target = np.zeros((8, 8), np.uint8)
    target[1:7, 1:7] = 1
    observations = []
    timestamp_by_source = dict(heldout_rows)
    for index, source in enumerate(evidence_sources):
        predicted = target.copy()
        competing = np.zeros_like(target)
        predicted_depth = np.ones((8, 8), np.float32) * 1.02
        if bad_projection:
            predicted[:] = 0
            predicted[1:3, 1:5] = 1
            predicted[5:7, 3:7] = 1
            competing = predicted.copy()
            predicted_depth[:] = 2.0
        paths = {
            "prediction_mask": tmp_path / f"prediction_{index}.npy",
            "target_mask": tmp_path / f"target_{index}.npy",
            "competing_mask": tmp_path / f"competing_{index}.npy",
            "prediction_depth": tmp_path / f"prediction_depth_{index}.npy",
        }
        np.save(paths["prediction_mask"], predicted)
        np.save(paths["target_mask"], target)
        np.save(paths["competing_mask"], competing)
        np.save(paths["prediction_depth"], predicted_depth)
        observations.append(
            {
                "object_id": 7,
                "source_image": source,
                "physical_timestamp": timestamp_by_source[source],
                **{
                    name: _spec(path, relative_to=tmp_path)
                    for name, path in paths.items()
                },
            }
        )
    evidence = {
        "schema": "farm.frozen-heldout-evidence.v1",
        "status": "frozen",
        "meters_per_scene_unit": 1.0,
        "prediction_kind": "post_lift_gaussian_projection",
        "depth_unit": "meters",
        "fold_manifest_sha256": _sha(manifest_path),
        "candidate_manifest_sha256": _sha(candidate_path),
        "heldout_frames_sha256": _sha(heldout_path),
        "contract": {
            "view_selection_precommitted": True,
            "prediction_uses_only_frozen_candidate": True,
            "targets_evaluation_only": True,
            "heldout_never_updates_candidate": True,
            "masks_obb_labels_not_modified": True,
        },
        "observations": observations,
    }
    evidence_path = tmp_path / "evidence.json"
    _write_json(evidence_path, evidence)
    return manifest_path, candidate_path, evidence_path


def _evaluate(paths: tuple[Path, Path, Path]) -> dict[str, object]:
    return evaluate_frozen_heldout_qc(
        *paths,
        meters_per_scene_unit=1.0,
        policy={
            "minimum_depth_pixels_per_timestamp": 8,
            "minimum_depth_timestamps": 2,
        },
    )


def test_exact_masks_on_two_independent_timestamps_pass_without_mutation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    before = {path: _sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    report = _evaluate(paths)
    after = {path: _sha(path) for path in before}
    assert before == after
    assert report["status"] == "PASS"
    assert report["routes"]["heldout_verified"] == [7]
    assert report["objects"][0]["independent_physical_timestamps"] == 2
    assert report["contract"]["heldout_updates_masks_obb_labels"] is False


def test_candidate_fit_contract_rejects_heldout_leakage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate = json.loads(paths[1].read_text())
    candidate["fit_contract"]["fit_source_images"].append("held_a.png")
    candidate["fit_contract"]["fit_physical_timestamps"].append("100")
    _write_json(paths[1], candidate)
    with pytest.raises(ValueError, match="heldout leakage"):
        _evaluate(paths)


def test_two_views_at_same_timestamp_count_once(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, evidence_sources=("held_a.png", "held_b.png"))
    report = _evaluate(paths)
    row = report["objects"][0]
    assert report["status"] == "BLOCKED"
    assert row["view_count"] == 2
    assert row["independent_physical_timestamps"] == 1
    assert row["route"] == "collect_independent_heldout_evidence"
    assert row["timestamps"][0]["view_count"] == 2


def test_missing_scale_in_fold_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, missing_heldout_scale=True)
    with pytest.raises(ValueError, match="missing meters_per_scene_unit"):
        _evaluate(paths)


def test_competition_fragmentation_and_depth_failure_route_to_train_only_refit(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, bad_projection=True)
    report = _evaluate(paths)
    row = report["objects"][0]
    assert row["route"] == "reject_candidate_train_only_refinement"
    assert "fragmented_projection" in row["reasons"]
    assert "competing_instance_contamination" in row["reasons"]
    assert "median_depth_error" in row["reasons"]
    assert "median_recall" in row["reasons"]


def test_evidence_cannot_substitute_a_train_source(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    evidence = json.loads(paths[2].read_text())
    evidence["observations"][0]["source_image"] = "train_a.png"
    evidence["observations"][0]["physical_timestamp"] = "10"
    _write_json(paths[2], evidence)
    with pytest.raises(ValueError, match="uses train source"):
        _evaluate(paths)


def test_declared_artifact_hash_is_enforced(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    evidence = json.loads(paths[2].read_text())
    evidence["observations"][0]["prediction_mask"]["sha256"] = "0" * 64
    _write_json(paths[2], evidence)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _evaluate(paths)


def test_cli_bootstraps_without_pythonpath(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts/evaluation/evaluate_farm_frozen_heldout_qc.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--meters-per-scene-unit" in completed.stdout


def test_official_cropped_little_endian_mask_npz_is_supported(
    tmp_path: Path,
) -> None:
    expected = np.zeros((8, 9), dtype=bool)
    expected[2:6, 3:8] = True
    crop = expected[2:6, 3:8]
    path = tmp_path / "packed_mask.npz"
    np.savez(
        path,
        image_shape=np.asarray(expected.shape, dtype=np.int32),
        raw_bits=np.packbits(crop.astype(np.uint8).reshape(-1), bitorder="little"),
        raw_shape=np.asarray(crop.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([3, 2, 8, 6], dtype=np.int32),
    )
    loaded = _load_mask(path, {"mask_kind": "raw"}, field="packed")
    assert np.array_equal(loaded, expected)
