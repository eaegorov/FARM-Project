from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.frozen_evaluation_union import (
    HeldoutBundle,
    build_frozen_evaluation_union,
)
from farm_runtime.frozen_heldout_projection import (
    _canonical_rgbd_transitive_contract,
    _rgbd_render_contract,
)
from farm_runtime.frozen_heldout_evidence import (
    _load_mapping_state,
    _mapping_image_sources,
    _validate_reference_targets,
)
from farm_runtime.frozen_heldout_qc import _validate_fold_manifest


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_ply(root: Path) -> Path:
    path = root / "source.ply"
    if not path.exists():
        path.write_bytes(b"ply")
    return path


def _frame(
    source: str,
    timestamp: str,
    *,
    rgb_path: str,
    depth_path: str,
) -> dict[str, object]:
    return {
        "frame_id": timestamp,
        "timestamp_ns": int(timestamp) * 100,
        "camera": "cam00_center",
        "source_image": source,
        "physical_timestamp": timestamp,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "depth_size": [4, 4],
        "K": [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        "T_world_cam": np.eye(4).tolist(),
        "colmap_camera_id": 1,
        "colmap_image_id": int(timestamp),
    }


def _frames(
    role: str, rows: list[dict[str, object]], source_union_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": "farm_frames_json_v1",
        "scene_id": "synthetic",
        "depth_units": "metres",
        "pose_translation_units": "metres",
        "meters_per_scene_unit": 1.0,
        "frames": rows,
        "full_colmap_fold": {
            "schema": "farm.full-colmap-rgbd-fold.v1",
            "role": role,
            "state_fit_authorized": role == "train",
            "source_union_frames_sha256": source_union_sha256,
            "usage": (
                "mapping_covisibility_sam3_merge_obb_fit"
                if role == "train"
                else "frozen_evaluation_only"
            ),
        },
    }


def _fit_frames(
    tmp_path: Path,
    *,
    timestamp: str = "000010",
    rescue_variant: bool = False,
) -> Path:
    root = tmp_path / "fit"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    source = f"cam00_{timestamp}_center.png"
    rgb = root / "rgb/fit.jpg"
    depth = root / "depth/fit.npy"
    rgb.write_bytes(b"fit-rgb")
    np.save(depth, np.ones((4, 4), np.float32))
    row = _frame(
        source,
        timestamp,
        rgb_path="rgb/fit.jpg",
        depth_path="depth/fit.npy",
    )
    row.pop("physical_timestamp")
    duplicate = dict(row)
    if rescue_variant:
        rescue_rgb = root / "rgb/fit_rescue.jpg"
        rescue_depth = root / "depth/fit_rescue.npy"
        rescue_rgb.write_bytes(rgb.read_bytes())
        np.save(rescue_depth, np.ones((4, 4), np.float64))
        duplicate = _frame(
            source,
            timestamp,
            rgb_path="rgb/fit_rescue.jpg",
            depth_path="depth/fit_rescue.npy",
        )
        duplicate["T_world_cam"][0][0] += 1.0e-15
        duplicate["full_colmap_split"] = "train"
    # An exact repeated row represents a repeated input record, not another
    # physical fit view.  The canonical fold must deduplicate it.
    path = root / "frames.json"
    _json(
        path,
        {
            "schema_version": "farm_frames_json_v1",
            "scene_id": "synthetic",
            "depth_units": "metres",
            "pose_translation_units": "metres",
            "meters_per_scene_unit": 1.0,
            "frames": [row, duplicate],
        },
    )
    return path


def _bundle(
    root: Path,
    *,
    object_id: int,
    heldout_source: str = "cam00_000100_center.png",
    heldout_timestamp: str = "000100",
    rgb_bytes: bytes = b"heldout-rgb",
    target_column: int = 1,
    mapping_column: int = 1,
    include_rejected_object: bool = False,
) -> tuple[HeldoutBundle, Path, Path]:
    folds = root / "folds"
    train_dir = folds / "train"
    heldout_dir = folds / "heldout"
    for directory in (
        train_dir / "rgb",
        train_dir / "depth",
        heldout_dir / "rgb",
        heldout_dir / "depth",
    ):
        directory.mkdir(parents=True)

    train_source = "cam00_000010_center.png"
    train_rgb = train_dir / "rgb/view.jpg"
    train_depth = train_dir / "depth/view.npy"
    train_rgb.write_bytes(b"fit-rgb")
    np.save(train_depth, np.ones((4, 4), np.float32))
    train_row = _frame(
        train_source,
        "000010",
        rgb_path="rgb/view.jpg",
        depth_path="depth/view.npy",
    )
    rgb = heldout_dir / "rgb/view.jpg"
    depth = heldout_dir / "depth/view.npy"
    rgb.write_bytes(rgb_bytes)
    np.save(depth, np.ones((4, 4), np.float32))
    heldout_row = _frame(
        heldout_source,
        heldout_timestamp,
        rgb_path="rgb/view.jpg",
        depth_path="depth/view.npy",
    )

    rgbd = root / "rgbd"
    rgbd.mkdir()
    union_train = dict(train_row)
    union_train["rgb_path"] = "../folds/train/rgb/view.jpg"
    union_train["depth_path"] = "../folds/train/depth/view.npy"
    union_heldout = dict(heldout_row)
    union_heldout["rgb_path"] = "../folds/heldout/rgb/view.jpg"
    union_heldout["depth_path"] = "../folds/heldout/depth/view.npy"
    union_frames = rgbd / "frames.json"
    _json(
        union_frames,
        {
            "schema_version": "farm_frames_json_v1",
            "scene_id": "synthetic",
            "depth_units": "metres",
            "pose_translation_units": "metres",
            "meters_per_scene_unit": 1.0,
            "frames": [union_train, union_heldout],
        },
    )
    render_config = {
        "scene_id": "synthetic",
        "identity_regex": "test",
        "sensor_group": "camera",
        "timestamp_group": "timestamp",
        "family_group": "view",
        "meters_per_scene_unit": 1.0,
        "nominal_hz": 10.0,
        "resolution": 4,
        "sh_degree": 3,
        "alpha_min": 0.05,
        "depth_min_m": 0.05,
        "depth_max_m": 80.0,
        "radius_clip": 0.25,
        "jpeg_quality": 95,
        "expected_baseline_m": 0.0,
        "baseline_tolerance_m": 0.005,
        "baseline_camera_ids": None,
        "baseline_sensors": None,
    }
    fingerprint_payload = {
        "config": render_config,
        "selected_names_sha256": hashlib.sha256(
            f"{train_source}\n{heldout_source}".encode("utf-8")
        ).hexdigest(),
        "selected_count": 2,
        "inputs": {"ply": {"path": "/source.ply", "size": 3}},
    }
    rgbd_manifest = rgbd / "run_manifest.json"
    _json(
        rgbd_manifest,
        {
            "schema_version": "farm_rgbd_run_manifest_v1",
            "status": "complete",
            "qa_passed": True,
            "scene_id": "synthetic",
            "selected_view_count": 2,
            "frames": "frames.json",
            "fingerprint": hashlib.sha256(
                json.dumps(
                    fingerprint_payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "fingerprint_payload": fingerprint_payload,
        },
    )
    train_path = train_dir / "frames.json"
    heldout_path = heldout_dir / "frames.json"
    source_union_sha = _sha(union_frames)
    _json(train_path, _frames("train", [train_row], source_union_sha))
    _json(heldout_path, _frames("heldout", [heldout_row], source_union_sha))
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
            "heldout_consumed": False,
            "heldout_usage": "frozen_evaluation_only",
        },
        "folds": {
            "train": {
                "frames_json": "train/frames.json",
                "sha256": _sha(train_path),
                "source_images": [train_source],
            },
            "heldout": {
                "frames_json": "heldout/frames.json",
                "sha256": _sha(heldout_path),
                "source_images": [heldout_source],
            },
        },
        "object_membership": {
            str(object_id): {
                "train": [train_source],
                "heldout": [heldout_source],
            }
        },
        "provenance": {
            "rendered_artifacts": {
                "train": [],
                "heldout": [
                    {
                        "source_image": heldout_source,
                        "physical_timestamp": heldout_timestamp,
                        "rgb": {
                            "path": "rgb/view.jpg",
                            "bytes": rgb.stat().st_size,
                            "sha256": _sha(rgb),
                        },
                        "depth": {
                            "path": "depth/view.npy",
                            "bytes": depth.stat().st_size,
                            "sha256": _sha(depth),
                        },
                    }
                ],
            }
        },
    }
    manifest_path = folds / "manifest.json"
    _json(manifest_path, manifest)

    mask_root = root / "mapping_masks"
    target_mapping = np.zeros((4, 4), np.uint8)
    target_mapping[:, mapping_column] = 1
    competitor = np.zeros((4, 4), np.uint8)
    competitor[0, 3] = 1
    target_mapping_path = mask_root / "object_000000/target.npz"
    competitor_path = mask_root / "object_000001/competitor.npz"
    target_mapping_path.parent.mkdir(parents=True)
    competitor_path.parent.mkdir(parents=True)
    np.savez_compressed(target_mapping_path, mask=target_mapping)
    np.savez_compressed(competitor_path, mask=competitor)
    mapping = {
        "state": {
            "images": [{"source_ref": heldout_source}],
            "object_id": [30, 31],
            "object_mask_observations": [
                [{"image_id": 0, "path": str(target_mapping_path)}],
                [{"image_id": 0, "path": str(competitor_path)}],
            ],
        }
    }
    mapping_path = root / "mapping.json"
    _json(mapping_path, mapping)

    reference_root = root / "reference"
    target = np.zeros((4, 4), np.uint8)
    target[:, target_column] = 1
    target_path = reference_root / f"masks/object_{object_id:06d}/target.npz"
    target_path.parent.mkdir(parents=True)
    np.savez_compressed(target_path, mask=target)
    objects: list[dict[str, object]] = [
        {
            "object_id": object_id,
            "accepted": True,
            "views": [
                {
                    "source_image": heldout_source,
                    "physical_timestamp_ns": int(heldout_timestamp),
                    "rescue_image_id": 0,
                    "rescue_track_index": 0,
                    "accepted": True,
                    "mask_relative": target_path.relative_to(reference_root).as_posix(),
                    "mask_sha256": _sha(target_path),
                }
            ],
        }
    ]
    if include_rejected_object:
        objects.append(
            {
                "object_id": 999,
                "accepted": False,
                "views": [{"accepted": True}],
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
            "rescue_frames_sha256": _sha(heldout_path),
            "rescue_state_sha256": _sha(mapping_path),
        },
        "objects": objects,
    }
    reference_path = reference_root / "result.json"
    _json(reference_path, reference)
    return (
        HeldoutBundle(
            manifest_path,
            reference_path,
            mapping_path,
            mask_root,
            rgbd_manifest,
            union_frames,
        ),
        target_mapping_path,
        target_path,
    )


def test_union_deduplicates_frames_remaps_ids_and_hardlinks(tmp_path: Path) -> None:
    fit = _fit_frames(tmp_path, rescue_variant=True)
    first, first_mapping_mask, first_target = _bundle(
        tmp_path / "a", object_id=7, include_rejected_object=True
    )
    second, second_mapping_mask, second_target = _bundle(tmp_path / "b", object_id=8)
    output = tmp_path / "union"
    manifest = build_frozen_evaluation_union(
        fit, [second, first], _source_ply(tmp_path), output
    )

    assert manifest["counts"]["input_actual_fit_rows"] == 2
    assert manifest["counts"]["train_frames"] == 1
    assert manifest["counts"]["deduplicated_actual_fit_rows"] == 1
    assert manifest["counts"]["heldout_input_rows"] == 2
    assert manifest["counts"]["heldout_frames"] == 1
    assert manifest["counts"]["deduplicated_heldout_rows"] == 1
    assert list(manifest["object_membership"]) == ["7", "8"]
    train = json.loads((output / "train/frames.json").read_text(encoding="utf-8"))
    assert train["frames"][0]["full_colmap_split"] == "train"
    dedup = manifest["provenance"]["actual_fit_deduplication"]
    assert len(dedup) == 1
    assert dedup[0]["semantic_equivalence_verified"] is True
    assert dedup[0]["selected"]["full_colmap_split"] == "train"
    assert (
        dedup[0]["candidates"][0]["depth"]["sha256"]
        != dedup[0]["candidates"][1]["depth"]["sha256"]
    )

    mapping_path = output / "mapping/scene_state.json"
    mapping = _load_mapping_state(mapping_path)
    assert mapping["canonical_object_id_by_track"][:2] == [7, 8]
    assert mapping["object_mask_observations"][0][0]["image_id"] == 0
    assert mapping["object_mask_observations"][1][0]["image_id"] == 0
    linked_mapping = sorted((output / "mapping/masks").rglob("*.npz"))
    assert any(
        path.stat().st_ino == first_mapping_mask.stat().st_ino
        for path in linked_mapping
    )
    assert any(
        path.stat().st_ino == second_mapping_mask.stat().st_ino
        for path in linked_mapping
    )

    reference_path = output / "heldout_reference/result.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert reference["summary"]["accepted_object_ids"] == [7, 8]
    assert [row["views"][0]["rescue_image_id"] for row in reference["objects"]] == [
        0,
        0,
    ]
    assert [row["views"][0]["rescue_track_index"] for row in reference["objects"]] == [
        0,
        1,
    ]
    linked_targets = sorted((output / "heldout_reference/masks").rglob("*.npz"))
    assert any(
        path.stat().st_ino == first_target.stat().st_ino for path in linked_targets
    )
    assert any(
        path.stat().st_ino == second_target.stat().st_ino for path in linked_targets
    )

    hashes: dict[Path, str] = {}
    _, _, heldout, membership, _, _, heldout_path = _validate_fold_manifest(
        output / "manifest.json", meters_per_scene_unit=1.0, hashes=hashes
    )
    canonical_frames_path = output / "rgbd/frames.json"
    canonical_manifest_path = output / "rgbd/run_manifest.json"
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    assert (
        canonical_manifest["schema_version"]
        == "farm.frozen-evaluation-rgbd-union-manifest.v1"
    )
    assert canonical_manifest["frames_artifact"]["sha256"] == _sha(
        canonical_frames_path
    )
    assert (
        canonical_manifest["frames_artifact"]["bytes"]
        == canonical_frames_path.stat().st_size
    )
    assert len(canonical_manifest["frame_artifacts"]) == 2
    heldout_document = json.loads(heldout_path.read_text(encoding="utf-8"))
    assert heldout_document["full_colmap_fold"]["source_union_frames_sha256"] == _sha(
        canonical_frames_path
    )
    assert manifest["frozen_evaluation_inputs"]["rgbd_union_frames"]["sha256"] == _sha(
        canonical_frames_path
    )
    renderer_config = _rgbd_render_contract(
        canonical_manifest_path,
        canonical_frames_path,
        heldout_path,
        scale=1.0,
        source_ply=_source_ply(tmp_path),
        hashes={},
    )
    assert renderer_config["alpha_min"] == 0.05
    _mapping_image_sources(mapping, heldout)
    targets = _validate_reference_targets(
        reference_path,
        heldout_frames_sha=_sha(heldout_path),
        mapping_state_sha=_sha(mapping_path),
        candidate_ids={7, 8},
        heldout=heldout,
        membership=membership,
        hashes=hashes,
    )
    assert set(targets) == {
        (7, "cam00_000100_center.png"),
        (8, "cam00_000100_center.png"),
    }


def test_union_rejects_physical_timestamp_leakage_atomically(tmp_path: Path) -> None:
    fit = _fit_frames(tmp_path, timestamp="000100")
    bundle, _, _ = _bundle(tmp_path / "bundle", object_id=7)
    output = tmp_path / "union"
    with pytest.raises(ValueError, match="train/heldout leakage"):
        build_frozen_evaluation_union(fit, [bundle], _source_ply(tmp_path), output)
    assert not output.exists()
    assert not list(tmp_path.glob(".union.*"))


def test_union_rejects_duplicate_heldout_rgb_mismatch(tmp_path: Path) -> None:
    fit = _fit_frames(tmp_path)
    first, _, _ = _bundle(tmp_path / "a", object_id=7, rgb_bytes=b"first")
    second, _, _ = _bundle(tmp_path / "b", object_id=8, rgb_bytes=b"second")
    output = tmp_path / "union"
    with pytest.raises(ValueError, match="conflicting duplicate heldout RGB/depth"):
        build_frozen_evaluation_union(
            fit, [first, second], _source_ply(tmp_path), output
        )
    assert not output.exists()


def test_union_rejects_nonidentical_duplicate_accepted_target(tmp_path: Path) -> None:
    fit = _fit_frames(tmp_path)
    first, _, _ = _bundle(tmp_path / "a", object_id=7, target_column=1)
    second, _, _ = _bundle(tmp_path / "b", object_id=7, target_column=2)
    output = tmp_path / "union"
    with pytest.raises(ValueError, match="conflicting accepted SAM3 target masks"):
        build_frozen_evaluation_union(
            fit, [first, second], _source_ply(tmp_path), output
        )
    assert not output.exists()


def test_union_rejects_non_qa_source_rgbd_run_atomically(tmp_path: Path) -> None:
    fit = _fit_frames(tmp_path)
    bundle, _, _ = _bundle(tmp_path / "bundle", object_id=7)
    payload = json.loads(bundle.rgbd_render_manifest.read_text(encoding="utf-8"))
    payload["qa_passed"] = False
    _json(bundle.rgbd_render_manifest, payload)
    output = tmp_path / "union"
    with pytest.raises(ValueError, match="completed QA-passing"):
        build_frozen_evaluation_union(fit, [bundle], _source_ply(tmp_path), output)
    assert not output.exists()


def test_projection_rejects_canonical_frame_provenance_drift(tmp_path: Path) -> None:
    fit = _fit_frames(tmp_path)
    bundle, _, _ = _bundle(tmp_path / "bundle", object_id=7)
    output = tmp_path / "union"
    build_frozen_evaluation_union(fit, [bundle], _source_ply(tmp_path), output)
    frames_path = output / "rgbd/frames.json"
    manifest_path = output / "rgbd/run_manifest.json"
    heldout_path = output / "heldout/frames.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["frame_artifacts"][0]["rgb"]["path"] = "different.jpg"
    _json(manifest_path, payload)
    with pytest.raises(ValueError, match="RGBD rgb provenance mismatch"):
        _rgbd_render_contract(
            manifest_path,
            frames_path,
            heldout_path,
            scale=1.0,
            source_ply=_source_ply(tmp_path),
            hashes={},
        )


@pytest.mark.parametrize("artifact", ["frames", "source_ply"])
def test_projection_rejects_canonical_artifact_byte_drift(
    tmp_path: Path, artifact: str
) -> None:
    fit = _fit_frames(tmp_path)
    bundle, _, _ = _bundle(tmp_path / "bundle", object_id=7)
    source_ply = _source_ply(tmp_path)
    output = tmp_path / "union"
    build_frozen_evaluation_union(fit, [bundle], source_ply, output)
    frames_path = output / "rgbd/frames.json"
    if artifact == "frames":
        frames_path.write_bytes(frames_path.read_bytes() + b"\n")
    else:
        source_ply.write_bytes(b"PLY")
    with pytest.raises(ValueError, match="SHA/size binding mismatch"):
        _rgbd_render_contract(
            output / "rgbd/run_manifest.json",
            frames_path,
            output / "heldout/frames.json",
            scale=1.0,
            source_ply=source_ply,
            hashes={},
        )


@pytest.mark.parametrize(
    "artifact", ["actual_fit", "source_manifest", "frame_rgb", "frame_depth"]
)
def test_projection_rejects_canonical_transitive_artifact_byte_drift(
    tmp_path: Path, artifact: str
) -> None:
    fit = _fit_frames(tmp_path)
    bundle, _, _ = _bundle(tmp_path / "bundle", object_id=7)
    source_ply = _source_ply(tmp_path)
    output = tmp_path / "union"
    build_frozen_evaluation_union(fit, [bundle], source_ply, output)
    manifest_path = output / "rgbd/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if artifact == "actual_fit":
        mutated = fit
    elif artifact == "source_manifest":
        mutated = bundle.rgbd_render_manifest
    else:
        row = manifest["frame_artifacts"][0]
        key = "rgb" if artifact == "frame_rgb" else "depth"
        mutated = (manifest_path.parent / row[key]["path"]).resolve(strict=True)
    mutated.write_bytes(mutated.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA/size binding mismatch"):
        _rgbd_render_contract(
            manifest_path,
            output / "rgbd/frames.json",
            output / "heldout/frames.json",
            scale=1.0,
            source_ply=source_ply,
            hashes={},
        )


def test_isolated_projection_validates_canonical_commitment_without_host_paths(
    tmp_path: Path,
) -> None:
    fit = _fit_frames(tmp_path)
    bundle, _, _ = _bundle(tmp_path / "bundle", object_id=7)
    source_ply = _source_ply(tmp_path)
    output = tmp_path / "union"
    build_frozen_evaluation_union(fit, [bundle], source_ply, output)
    manifest_path = output / "rgbd/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _canonical_rgbd_transitive_contract(
        manifest, manifest_sha256=_sha(manifest_path)
    )
    assert expected["source_run_count"] == 1
    assert expected["frame_count"] == 2
    assert expected["artifact_count"] == 8

    fit.parent.rename(tmp_path / "unmounted_fit")
    bundle.fold_manifest.parents[1].rename(tmp_path / "unmounted_bundle")
    renderer = _rgbd_render_contract(
        manifest_path,
        output / "rgbd/frames.json",
        output / "heldout/frames.json",
        scale=1.0,
        source_ply=source_ply,
        hashes={},
        verify_transitive_artifacts=False,
    )
    assert renderer["rgbd_transitive_contract"] == expected
    with pytest.raises(ValueError, match="artifact is unavailable"):
        _rgbd_render_contract(
            manifest_path,
            output / "rgbd/frames.json",
            output / "heldout/frames.json",
            scale=1.0,
            source_ply=source_ply,
            hashes={},
            verify_transitive_artifacts=True,
        )


def test_canonical_transitive_commitment_detects_declaration_drift(
    tmp_path: Path,
) -> None:
    fit = _fit_frames(tmp_path)
    bundle, _, _ = _bundle(tmp_path / "bundle", object_id=7)
    output = tmp_path / "union"
    build_frozen_evaluation_union(fit, [bundle], _source_ply(tmp_path), output)
    manifest_path = output / "rgbd/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = _canonical_rgbd_transitive_contract(
        manifest, manifest_sha256=_sha(manifest_path)
    )
    manifest["frame_artifacts"][0]["rgb"]["sha256"] = "0" * 64
    drifted = _canonical_rgbd_transitive_contract(
        manifest, manifest_sha256=_sha(manifest_path)
    )
    assert drifted["entries_sha256"] != original["entries_sha256"]
