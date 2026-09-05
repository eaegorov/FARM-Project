from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.frozen_heldout_evidence import (
    PROJECTION_SCHEMA,
    materialize_evidence_payload,
    materialize_frozen_candidate,
    prepare_frozen_heldout_evidence,
)
from farm_runtime.frozen_heldout_qc import evaluate_frozen_heldout_qc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _spec(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": os.path.relpath(path, base),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _ply(path: Path, count: int) -> None:
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n" + "0 0 0\n" * count,
        encoding="ascii",
    )


def _fixture(tmp_path: Path) -> dict[str, Path | float]:
    scale = 1.0
    folds = tmp_path / "folds"
    train_dir = folds / "train"
    heldout_dir = folds / "heldout"
    train_dir.mkdir(parents=True)
    heldout_dir.mkdir()
    train_rows = [("train_a.png", "10"), ("train_b.png", "20")]
    heldout_rows = [("held_a.png", "100"), ("held_b.png", "200")]

    def frames(role: str, rows: list[tuple[str, str]]) -> dict[str, object]:
        return {
            "schema_version": "farm_frames_json_v1",
            "depth_units": "metres",
            "meters_per_scene_unit": scale,
            "frames": [
                {
                    "frame_id": str(index),
                    "source_image": source,
                    "physical_timestamp": timestamp,
                    "rgb_path": f"rgb/{index:03d}_{Path(source).stem}.jpg",
                    "depth_path": f"depth_{source}.npy",
                    "depth_size": [8, 8],
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

    for source, _ in train_rows:
        np.save(train_dir / f"depth_{source}.npy", np.ones((8, 8), np.float32))
    heldout_artifacts = []
    for source, timestamp in heldout_rows:
        depth = heldout_dir / f"depth_{source}.npy"
        np.save(depth, np.ones((8, 8), np.float32))
        heldout_artifacts.append(
            {
                "source_image": source,
                "physical_timestamp": timestamp,
                "depth": _spec(depth, heldout_dir),
            }
        )
    train_frames = train_dir / "frames.json"
    heldout_frames = heldout_dir / "frames.json"
    _json(train_frames, frames("train", train_rows))
    _json(heldout_frames, frames("heldout", heldout_rows))
    fold = {
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
                "sha256": _sha(train_frames),
                "source_images": [row[0] for row in train_rows],
            },
            "heldout": {
                "frames_json": "heldout/frames.json",
                "sha256": _sha(heldout_frames),
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
    fold_path = folds / "manifest.json"
    _json(fold_path, fold)

    source_ply = tmp_path / "source.ply"
    candidate_ply = tmp_path / "candidate.ply"
    source_labels = tmp_path / "source_labels.npy"
    candidate_labels = tmp_path / "candidate_labels.npy"
    _ply(source_ply, 4)
    _ply(candidate_ply, 4)
    np.save(source_labels, np.array([7, -1, -1, -1], np.int32))
    np.save(candidate_labels, np.array([7, 7, -1, -1], np.int32))
    upstream = {
        "schema_version": "farm.gaussian-lift.result.v1",
        "status": "PASS",
        "inputs": {"source_ply_sha256": _sha(source_ply)},
        "contracts": {
            "source_ply_mutated": False,
            "dense_arrays_in_source_ply_order": True,
            "verified_only_canonical_labels": True,
        },
        "frozen_full_colmap_fit_contract": {
            "schema": "farm.full-colmap-train-fit.v1",
            "fold_manifest_sha256": _sha(fold_path),
            "train_frames_sha256": _sha(train_frames),
            "fit_splits": ["train"],
            "heldout_consumed": False,
            "heldout_updates_candidate": False,
            "frozen_before_heldout_evaluation": True,
            "fit_source_images": [row[0] for row in train_rows],
            "fit_physical_timestamps": [row[1] for row in train_rows],
        },
        "artifacts": {
            "provisional": {"object_id": _spec(source_labels, tmp_path)},
            "final": {
                "object_id": _spec(candidate_labels, tmp_path),
                "instance_labeled_full": _spec(candidate_ply, tmp_path),
            },
        },
    }
    upstream_path = tmp_path / "lift_result.json"
    _json(upstream_path, upstream)
    candidate = materialize_frozen_candidate(
        fold_path,
        upstream_path,
        source_ply,
        source_labels,
        candidate_ply,
        candidate_labels,
        meters_per_scene_unit=scale,
    )
    candidate_path = tmp_path / "candidate_manifest.json"
    _json(candidate_path, candidate)

    mapping_masks = tmp_path / "mapping_masks"
    target_mask = np.zeros((8, 8), np.uint8)
    target_mask[2:6, 2:6] = 1
    competitor = np.zeros((8, 8), np.uint8)
    competitor[0, 0] = 1
    competitor[2, 2] = 1
    target_observations = []
    competitor_observations = []
    for image_id in range(2):
        target_path = mapping_masks / "object_000001" / f"img_{image_id:06d}.npz"
        competitor_path = mapping_masks / "object_000002" / f"img_{image_id:06d}.npz"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        competitor_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target_path, mask=target_mask)
        np.savez_compressed(competitor_path, mask=competitor)
        target_observations.append({"image_id": image_id, "path": str(target_path)})
        competitor_observations.append(
            {"image_id": image_id, "path": str(competitor_path)}
        )
    mapping = {
        "state": {
            "images": [
                {"source_ref": f"rgb/{index:03d}_{Path(source).stem}.jpg"}
                for index, (source, _) in enumerate(heldout_rows)
            ],
            "object_mask_observations": [
                target_observations,
                competitor_observations,
            ],
        }
    }
    mapping_path = tmp_path / "heldout_mapping.json"
    _json(mapping_path, mapping)

    reference_root = tmp_path / "sam3_reference"
    views = []
    for image_id, (source, timestamp) in enumerate(heldout_rows):
        path = reference_root / "masks/object_000007" / f"mask_{image_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mask=target_mask)
        views.append(
            {
                "source_image": source,
                "physical_timestamp_ns": int(timestamp),
                "rescue_image_id": image_id,
                "rescue_track_index": 0,
                "accepted": True,
                "mask_relative": path.relative_to(reference_root).as_posix(),
                "mask_sha256": _sha(path),
            }
        )
    reference = {
        "schema": "farm.full-colmap-mask-refinement.v1",
        "status": "PASS",
        "refinement_role": "heldout_reference_only",
        "input_view_contract": {
            "fold_contract": "farm.full-colmap-rgbd-fold.v1",
            "active_split": "heldout",
            "state_fit_authorized": False,
            "plan_view_field": "heldout_views",
            "requested_view_role": "heldout-reference",
        },
        "policy": {
            "view_role": "heldout-reference",
            "state_fit_authorized": False,
            "merge_apply_forbidden": True,
            "heldout_reference_is_evaluation_only": True,
            "reference_masks_are_pseudo_labels_not_ground_truth": True,
        },
        "hashes": {
            "rescue_frames_sha256": _sha(heldout_frames),
            "rescue_state_sha256": _sha(mapping_path),
        },
        "objects": [{"object_id": 7, "accepted": True, "views": views}],
    }
    reference_path = reference_root / "result.json"
    _json(reference_path, reference)
    return {
        "scale": scale,
        "fold": fold_path,
        "train_frames": train_frames,
        "heldout_frames": heldout_frames,
        "upstream": upstream_path,
        "source_ply": source_ply,
        "source_labels": source_labels,
        "candidate_ply": candidate_ply,
        "candidate_labels": candidate_labels,
        "candidate": candidate_path,
        "reference": reference_path,
        "mapping": mapping_path,
        "mask_root": mapping_masks,
    }


def _prepare(
    fixture: dict[str, Path | float], projection: Path | None = None
) -> dict[str, object]:
    return prepare_frozen_heldout_evidence(
        fixture["fold"],
        fixture["candidate"],
        fixture["source_ply"],
        fixture["source_labels"],
        fixture["candidate_ply"],
        fixture["candidate_labels"],
        fixture["reference"],
        fixture["mapping"],
        fixture["mask_root"],
        meters_per_scene_unit=float(fixture["scale"]),
        projection_manifest_path=projection,
    )


def _projection(fixture: dict[str, Path | float], *, omit_last: bool = False) -> Path:
    candidate = json.loads(Path(fixture["candidate"]).read_text(encoding="utf-8"))
    heldout = json.loads(Path(fixture["heldout_frames"]).read_text(encoding="utf-8"))
    root = Path(fixture["fold"]).parent
    rows = []
    for index, frame in enumerate(heldout["frames"]):
        if omit_last and index == len(heldout["frames"]) - 1:
            continue
        mask = np.zeros((8, 8), np.uint8)
        mask[2:6, 2:6] = 1
        mask_path = root / f"prediction_{index}.npy"
        depth_path = root / f"prediction_depth_{index}.npy"
        np.save(mask_path, mask)
        np.save(depth_path, np.ones((8, 8), np.float32))
        rows.append(
            {
                "object_id": 7,
                "source_image": frame["source_image"],
                "physical_timestamp": frame["physical_timestamp"],
                "prediction_mask": _spec(mask_path, root),
                "prediction_depth": _spec(depth_path, root),
            }
        )
    artifacts = candidate["gaussian_artifacts"]
    projection = {
        "schema": PROJECTION_SCHEMA,
        "status": "frozen",
        "prediction_kind": "post_lift_gaussian_projection",
        "depth_unit": "meters",
        "fold_manifest_sha256": _sha(Path(fixture["fold"])),
        "candidate_manifest_sha256": _sha(Path(fixture["candidate"])),
        "heldout_frames_sha256": _sha(Path(fixture["heldout_frames"])),
        **{f"{name}_sha256": spec["sha256"] for name, spec in artifacts.items()},
        "renderer_contract": {
            "read_only": True,
            "candidate_frozen_before_projection": True,
            "candidate_ply_mutated": False,
            "candidate_labels_mutated": False,
            "heldout_updates_candidate": False,
            "renders_only_precommitted_heldout_views": True,
            "prediction_generated_without_target_masks": True,
            "targets_not_consumed_by_renderer": True,
            "prediction_depth_in_meters": True,
        },
        "renderer_implementation": _spec(
            Path("src/farm_runtime/frozen_heldout_evidence.py").resolve(), root
        ),
        "observations": rows,
    }
    path = root / "projection_manifest.json"
    _json(path, projection)
    return path


def test_candidate_materializer_binds_gaussians_and_train_only_fit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    candidate = json.loads(Path(fixture["candidate"]).read_text(encoding="utf-8"))
    assert candidate["schema"] == "farm.frozen-heldout-candidate.v1"
    assert candidate["fit_contract"]["fit_splits"] == ["train"]
    assert candidate["gaussian_contract"]["candidate_gaussian_count"] == 4
    assert [row["object_id"] for row in candidate["objects"]] == [7]


def test_candidate_materializer_rejects_missing_contract_and_heldout_leakage(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    upstream_path = Path(fixture["upstream"])
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream.pop("frozen_full_colmap_fit_contract")
    _json(upstream_path, upstream)
    with pytest.raises(ValueError, match="frozen_full_colmap_fit_contract"):
        materialize_frozen_candidate(
            fixture["fold"],
            upstream_path,
            fixture["source_ply"],
            fixture["source_labels"],
            fixture["candidate_ply"],
            fixture["candidate_labels"],
            meters_per_scene_unit=1.0,
        )

    fixture = _fixture(tmp_path / "leak")
    upstream_path = Path(fixture["upstream"])
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream["frozen_full_colmap_fit_contract"]["fit_source_images"].append(
        "held_a.png"
    )
    _json(upstream_path, upstream)
    with pytest.raises(ValueError, match="source leakage"):
        materialize_frozen_candidate(
            fixture["fold"],
            upstream_path,
            fixture["source_ply"],
            fixture["source_labels"],
            fixture["candidate_ply"],
            fixture["candidate_labels"],
            meters_per_scene_unit=1.0,
        )


def test_missing_renderer_is_explicit_blocked_preflight(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture)
    assert prepared["status"] == "BLOCKED"
    assert prepared["reason"] == "read_only_gaussian_projection_renderer_required"
    assert prepared["renderer_implemented_here"] is False
    assert prepared["target_count"] == 2
    assert prepared["projections"] is None
    with pytest.raises(ValueError, match="without attested projections"):
        materialize_evidence_payload(prepared, {})


def test_reference_role_and_mapping_leakage_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reference_path = Path(fixture["reference"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["refinement_role"] = "train_fit"
    _json(reference_path, reference)
    with pytest.raises(ValueError, match="heldout-reference"):
        _prepare(fixture)

    fixture = _fixture(tmp_path / "mapping_leak")
    mapping_path = Path(fixture["mapping"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["state"]["images"][0]["source_ref"] = "rgb/train_a.jpg"
    _json(mapping_path, mapping)
    reference_path = Path(fixture["reference"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["hashes"]["rescue_state_sha256"] = _sha(mapping_path)
    _json(reference_path, reference)
    with pytest.raises(ValueError, match="unknown or duplicate heldout image"):
        _prepare(fixture)


def test_projection_must_exactly_cover_precommitted_targets(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    projection = _projection(fixture, omit_last=True)
    with pytest.raises(ValueError, match="exactly cover frozen targets"):
        _prepare(fixture, projection)


def test_materialized_evidence_is_accepted_by_existing_evaluator(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    projection = _projection(fixture)
    prepared = _prepare(fixture, projection)
    assert prepared["status"] == "PASS"
    competing_specs = {}
    for index, target in enumerate(prepared["targets"]):
        key = (int(target["object_id"]), str(target["source_image"]))
        path = tmp_path / f"competing_{index}.npy"
        np.save(path, prepared["competing_unions"][key])
        competing_specs[key] = _spec(path, tmp_path)
        assert int(np.asarray(prepared["competing_unions"][key]).sum()) == 1
    evidence = materialize_evidence_payload(prepared, competing_specs)
    evidence_path = tmp_path / "evidence.json"
    _json(evidence_path, evidence)
    report = evaluate_frozen_heldout_qc(
        Path(fixture["fold"]),
        Path(fixture["candidate"]),
        evidence_path,
        meters_per_scene_unit=1.0,
        policy={
            "minimum_depth_pixels_per_timestamp": 8,
            "minimum_depth_timestamps": 2,
        },
    )
    assert report["status"] == "PASS"
    assert report["routes"]["heldout_verified"] == [7]


def test_builder_cli_publishes_only_blocked_preflight_without_renderer(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "blocked_bundle"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/build_farm_frozen_heldout_evidence.py",
            "--fold-manifest",
            str(fixture["fold"]),
            "--candidate-manifest",
            str(fixture["candidate"]),
            "--source-ply",
            str(fixture["source_ply"]),
            "--source-labels",
            str(fixture["source_labels"]),
            "--candidate-ply",
            str(fixture["candidate_ply"]),
            "--candidate-labels",
            str(fixture["candidate_labels"]),
            "--heldout-reference-result",
            str(fixture["reference"]),
            "--heldout-yoloe-state",
            str(fixture["mapping"]),
            "--heldout-yoloe-mask-root",
            str(fixture["mask_root"]),
            "--meters-per-scene-unit",
            "1.0",
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert (output / "preflight.json").is_file()
    assert not (output / "evidence.json").exists()
    marker = json.loads((output / "_RESULT.json").read_text(encoding="utf-8"))
    assert marker["status"] == "BLOCKED"
    assert marker["evidence"] is None


@pytest.mark.parametrize(
    "script",
    [
        "scripts/evaluation/materialize_farm_frozen_heldout_candidate.py",
        "scripts/evaluation/build_farm_frozen_heldout_evidence.py",
    ],
)
def test_cli_help_is_cpu_only(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--fold-manifest" in result.stdout
