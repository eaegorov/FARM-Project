from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.evaluation.build_farm_frozen_heldout_visual_qa import (
    REPORT_SCHEMA,
    _error_overlay,
    build_visual_qa,
    compute_context_crop,
)


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


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    folds = tmp_path / "folds"
    heldout = folds / "heldout"
    heldout.mkdir(parents=True)
    observations = []
    frames = []
    rendered = []
    qc_views = []
    for index, (source, timestamp) in enumerate(
        (("held_a.png", "100"), ("held_b.png", "200"))
    ):
        rgb = np.zeros((48, 64, 3), np.uint8)
        rgb[..., 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
        rgb[..., 1] = np.arange(48, dtype=np.uint8)[:, None] * 4
        rgb[..., 2] = 96 + index * 40
        rgb_path = heldout / f"rgb_{index}.png"
        Image.fromarray(rgb).save(rgb_path)
        depth_path = heldout / f"depth_{index}.npy"
        np.save(depth_path, np.ones((48, 64), np.float32))
        target = np.zeros((48, 64), np.uint8)
        target[12:38, 18:48] = 1
        prediction = np.zeros_like(target)
        prediction[10:36, 21:52] = 1
        competing = np.zeros_like(target)
        target_path = tmp_path / f"target_{index}.npy"
        prediction_path = tmp_path / f"prediction_{index}.npy"
        competing_path = tmp_path / f"competing_{index}.npy"
        predicted_depth_path = tmp_path / f"predicted_depth_{index}.npy"
        np.save(target_path, target)
        np.save(prediction_path, prediction)
        np.save(competing_path, competing)
        np.save(predicted_depth_path, np.ones((48, 64), np.float32))
        frames.append(
            {
                "frame_id": str(index),
                "source_image": source,
                "physical_timestamp": timestamp,
                "rgb_path": rgb_path.name,
                "depth_path": depth_path.name,
                "depth_size": [48, 64],
            }
        )
        rendered.append(
            {
                "source_image": source,
                "physical_timestamp": timestamp,
                "rgb": _spec(rgb_path, heldout),
                "depth": _spec(depth_path, heldout),
            }
        )
        observations.append(
            {
                "object_id": 7,
                "source_image": source,
                "physical_timestamp": timestamp,
                "prediction_mask": _spec(prediction_path, tmp_path),
                "target_mask": _spec(target_path, tmp_path),
                "competing_mask": _spec(competing_path, tmp_path),
                "prediction_depth": _spec(predicted_depth_path, tmp_path),
            }
        )
        qc_views.append(
            {
                "object_id": 7,
                "source_image": source,
                "physical_timestamp": timestamp,
                "iou": 0.75,
                "precision": 0.80,
                "recall": 0.85,
                "area_ratio": 1.06,
                "largest_component_fraction": 1.0,
                "competing_prediction_fraction": 0.0,
                "depth_coverage": 1.0,
                "depth_error_median": 0.01,
                "depth_error_p90": 0.02,
                "artifacts": {
                    "prediction_mask": {
                        "path": str(prediction_path.resolve()),
                        "sha256": _sha(prediction_path),
                    },
                    "target_mask": {
                        "path": str(target_path.resolve()),
                        "sha256": _sha(target_path),
                    },
                },
            }
        )

    heldout_frames = heldout / "frames.json"
    _json(
        heldout_frames,
        {
            "schema_version": "farm_frames_json_v1",
            "frames": frames,
        },
    )
    train = folds / "train" / "frames.json"
    _json(train, {"schema_version": "farm_frames_json_v1", "frames": []})
    fold = folds / "manifest.json"
    _json(
        fold,
        {
            "schema": "farm.full-colmap-rgbd-folds.v1",
            "status": "PASS",
            "folds": {
                "train": {
                    "frames_json": "train/frames.json",
                    "sha256": _sha(train),
                    "source_images": [],
                },
                "heldout": {
                    "frames_json": "heldout/frames.json",
                    "sha256": _sha(heldout_frames),
                    "source_images": ["held_a.png", "held_b.png"],
                },
            },
            "provenance": {"rendered_artifacts": {"train": [], "heldout": rendered}},
        },
    )
    evidence = tmp_path / "evidence.json"
    _json(
        evidence,
        {
            "schema": "farm.frozen-heldout-evidence.v1",
            "status": "frozen",
            "observations": observations,
        },
    )
    qc = tmp_path / "qc.json"
    _json(
        qc,
        {
            "schema": "farm.frozen-heldout-qc.v1",
            "status": "PASS",
            "counts": {"candidate_objects": 1, "observations": 2},
            "objects": [
                {
                    "object_id": 7,
                    "status": "verified",
                    "route": "heldout_verified",
                    "reasons": [],
                    "independent_physical_timestamps": 2,
                    "good_timestamps": 2,
                    "summaries": {
                        "iou": {"median": 0.75},
                        "precision": {"median": 0.80},
                        "recall": {"median": 0.85},
                    },
                    "views": qc_views,
                }
            ],
            "provenance": {
                "fold_manifest": {
                    "path": str(fold.resolve()),
                    "sha256": _sha(fold),
                },
                "evidence_manifest": {
                    "path": str(evidence.resolve()),
                    "sha256": _sha(evidence),
                },
                "heldout_frames": {
                    "path": str(heldout_frames.resolve()),
                    "sha256": _sha(heldout_frames),
                },
            },
        },
    )
    return qc, heldout / "rgb_0.png"


def test_builds_exact_object_sheets_atomically_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    qc, _ = _fixture(tmp_path)
    before = {path: _sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "visual_qa"
    report = build_visual_qa(qc, output)
    after = {path: _sha(path) for path in before}
    assert before == after
    assert report["schema"] == REPORT_SCHEMA
    assert report["counts"]["candidate_objects"] == 1
    assert report["counts"]["observations"] == 2
    assert (output / "objects/object_000007.png").is_file()
    assert (output / "summary.png").is_file()
    result = json.loads((output / "_RESULT.json").read_text())
    assert result["manifest_sha256"] == _sha(output / "manifest.json")
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_visual_qa(qc, output)


def test_error_overlay_uses_unambiguous_tp_fp_fn_colors() -> None:
    rgb = Image.new("RGB", (3, 1), (0, 0, 0))
    predicted = np.asarray([[1, 1, 0]], dtype=bool)
    target = np.asarray([[1, 0, 1]], dtype=bool)
    pixels = np.asarray(_error_overlay(rgb, predicted, target))[0]
    assert pixels.tolist() == [[29, 171, 92], [195, 60, 60], [52, 122, 196]]


def test_context_crop_contains_target_and_prediction_with_clipping() -> None:
    target = np.zeros((20, 30), dtype=bool)
    predicted = np.zeros_like(target)
    target[0:4, 0:5] = True
    predicted[3:10, 5:12] = True
    assert compute_context_crop(
        (target, predicted), context_fraction=0.25, minimum_context_pixels=2
    ) == (0, 0, 15, 13)


def test_tampered_rgb_fails_before_output_is_published(tmp_path: Path) -> None:
    qc, rgb = _fixture(tmp_path)
    rgb.write_bytes(rgb.read_bytes() + b"tampered")
    output = tmp_path / "visual_qa"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_visual_qa(qc, output)
    assert not output.exists()


def test_cli_bootstraps_without_pythonpath(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts/evaluation/build_farm_frozen_heldout_visual_qa.py"
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
    assert "--qc-report" in completed.stdout
