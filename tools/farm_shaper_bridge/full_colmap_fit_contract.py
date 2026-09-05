"""External full-COLMAP train-fold authority for Gaussian lifting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.farm_shaper_bridge.common import sha256_file


def validate_full_colmap_train_fit(run: Any, manifest_path: Path) -> dict[str, Any]:
    """Bind every actually consumed FARM frame to the declared train fold.

    This function opens only the fold manifest and its train frames JSON.  The
    heldout frames file and all heldout depth/mask artifacts remain unopened by
    Gaussian fitting.  Complete fold integrity is checked later by the frozen
    heldout evaluator.
    """

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("full-COLMAP fold manifest must be a JSON object")
    if (
        manifest.get("schema") != "farm.full-colmap-rgbd-folds.v1"
        or manifest.get("status") != "PASS"
    ):
        raise ValueError("full-COLMAP fold manifest is not PASS v1")
    integrity = manifest.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("source_image_sets_disjoint") is not True
        or integrity.get("physical_timestamp_sets_disjoint") is not True
        or integrity.get("all_rendered_artifacts_sha256_hashed") is not True
    ):
        raise ValueError("full-COLMAP fold integrity attestation is incomplete")
    policy = manifest.get("fit_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("fit_splits") != ["train"]
        or policy.get("heldout_consumed") is not False
        or policy.get("heldout_usage") != "frozen_evaluation_only"
    ):
        raise ValueError("full-COLMAP fold fit policy is not train-only")
    folds = manifest.get("folds")
    if not isinstance(folds, Mapping):
        raise ValueError("full-COLMAP fold entries are missing")
    train_spec = folds.get("train")
    heldout_spec = folds.get("heldout")
    if not isinstance(train_spec, Mapping) or not isinstance(heldout_spec, Mapping):
        raise ValueError("full-COLMAP train/heldout entries are missing")
    train_relative = Path(str(train_spec.get("frames_json") or ""))
    if train_relative.is_absolute() or not train_relative.parts:
        raise ValueError("full-COLMAP train frames path is unsafe")
    train_path = (manifest_path.parent / train_relative).resolve(strict=True)
    if not train_path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("full-COLMAP train frames escape fold directory")
    train_sha = sha256_file(train_path)
    if train_sha != str(train_spec.get("sha256") or "").lower():
        raise ValueError("full-COLMAP train frames SHA-256 mismatch")
    heldout_sha = str(heldout_spec.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", heldout_sha):
        raise ValueError("full-COLMAP heldout frames lack a SHA-256 identity")
    train_doc = json.loads(train_path.read_text(encoding="utf-8"))
    train_fold = (
        train_doc.get("full_colmap_fold") if isinstance(train_doc, Mapping) else None
    )
    if (
        not isinstance(train_doc, Mapping)
        or train_doc.get("schema_version") != "farm_frames_json_v1"
        or train_doc.get("depth_units") != "metres"
        or train_doc.get("pose_translation_units") != "metres"
        or not isinstance(train_fold, Mapping)
        or train_fold.get("role") != "train"
        or train_fold.get("state_fit_authorized") is not True
    ):
        raise ValueError("full-COLMAP train frames contract is invalid")
    if (
        str(train_doc.get("scene_id") or "") != run.scene_id
        or abs(
            float(train_doc.get("meters_per_scene_unit", 0.0))
            - run.meters_per_scene_unit
        )
        > 1.0e-12
    ):
        raise ValueError("full-COLMAP train scene/scale differs from FARM run")
    train_rows: dict[str, Mapping[str, Any]] = {}
    for row in train_doc.get("frames") or []:
        if not isinstance(row, Mapping):
            raise ValueError("full-COLMAP train frames contain an invalid row")
        source = str(row.get("source_image") or "").strip()
        if not source or source in train_rows:
            raise ValueError("full-COLMAP train source images are empty/duplicate")
        train_rows[source] = row
    declared_train = [str(value) for value in train_spec.get("source_images") or []]
    declared_heldout = [str(value) for value in heldout_spec.get("source_images") or []]
    if (
        set(declared_train) != set(train_rows)
        or len(declared_train) != len(set(declared_train))
        or len(declared_heldout) != len(set(declared_heldout))
        or set(declared_train).intersection(declared_heldout)
    ):
        raise ValueError("full-COLMAP fold source-image inventories are inconsistent")

    fit_sources: list[str] = []
    fit_timestamps: list[str] = []
    for frame in run.frames:
        row = train_rows.get(frame.source_image)
        if row is None:
            raise ValueError(
                f"lift frame is absent from external train fold: {frame.source_image}"
            )
        if (
            str(row.get("frame_id")) != frame.frame_id
            or int(row.get("timestamp_ns", -1)) != frame.timestamp_ns
            or tuple(int(value) for value in row.get("depth_size") or [])
            != frame.depth_size
            or not np.allclose(
                np.asarray(row.get("K"), dtype=np.float64), frame.K, atol=1e-9
            )
            or not np.allclose(
                np.asarray(row.get("T_world_cam"), dtype=np.float64),
                frame.T_world_cam,
                atol=1e-9,
            )
        ):
            raise ValueError(
                f"lift/train camera calibration mismatch: {frame.source_image}"
            )
        physical = str(row.get("physical_timestamp") or "").strip()
        if not physical:
            raise ValueError("full-COLMAP train frame lacks physical_timestamp")
        fit_sources.append(frame.source_image)
        if physical not in fit_timestamps:
            fit_timestamps.append(physical)
    if len(fit_sources) != len(set(fit_sources)):
        raise ValueError("FARM run repeats a fit source image")
    return {
        "schema": "farm.full-colmap-train-fit.v1",
        "fold_manifest_sha256": sha256_file(manifest_path),
        "train_frames_sha256": train_sha,
        "heldout_frames_sha256": heldout_sha,
        "fit_splits": ["train"],
        "heldout_consumed": False,
        "heldout_updates_candidate": False,
        "frozen_before_heldout_evaluation": True,
        "fit_source_images": fit_sources,
        "fit_physical_timestamps": fit_timestamps,
        "fit_source_count": len(fit_sources),
        "fit_physical_timestamp_count": len(fit_timestamps),
        "heldout_frames_opened_by_lift": False,
        "validation_scope": "fold_manifest_and_declared_train_frames_only",
    }
