from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.frozen_heldout_projection import (
    PROJECTION_RENDERER_CONFIG_SCHEMA,
    PROJECTION_REQUEST_SCHEMA,
    _canonical_rgbd_transitive_contract,
    _heldout_reference_rows,
    validate_renderer_inputs,
)
from tools.farm_shaper_bridge.full_colmap_fit_contract import (
    validate_full_colmap_train_fit,
)
from tools.farm_shaper_bridge import (
    frozen_heldout_projection as renderer_entrypoint,
)
from tools.farm_shaper_bridge import (
    run_frozen_heldout_projection as projection_runner,
)
from tools.farm_shaper_bridge.frozen_heldout_projection import (
    _snapshot_renderer_sources,
)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ply(path: Path, count: int) -> None:
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n" + "0 0 0\n" * count,
        encoding="ascii",
    )


def _camera(source: str, timestamp: str = "100") -> dict[str, object]:
    return {
        "source_image": source,
        "frame_id": timestamp,
        "physical_timestamp": timestamp,
        "timestamp_ns": int(timestamp),
        "camera": "cam00_center",
        "depth_size": [8, 8],
        "K": [[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]],
        "T_world_cam": np.eye(4).tolist(),
        "rgb_path": "not-mounted.jpg",
        "depth_path": "not-mounted.npy",
    }


def _renderer_fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_ply = tmp_path / "source.ply"
    candidate_ply = tmp_path / "candidate.ply"
    source_labels = tmp_path / "source.npy"
    candidate_labels = tmp_path / "candidate.npy"
    _ply(source_ply, 3)
    _ply(candidate_ply, 3)
    np.save(source_labels, np.array([7, -1, -1], np.int32))
    np.save(candidate_labels, np.array([7, 7, -1], np.int32))
    frame = _camera("cam00_100_center.png")
    union_frames = tmp_path / "rgbd" / "frames.json"
    union_doc = {
        "schema_version": "farm_frames_json_v1",
        "scene_id": "factory",
        "frames": [frame],
    }
    _json(union_frames, union_doc)
    heldout_frames = tmp_path / "folds" / "heldout" / "frames.json"
    heldout_doc = {
        "schema_version": "farm_frames_json_v1",
        "scene_id": "factory",
        "depth_units": "metres",
        "pose_translation_units": "metres",
        "meters_per_scene_unit": 1.0,
        "full_colmap_fold": {
            "role": "heldout",
            "state_fit_authorized": False,
            "source_union_frames_sha256": _sha(union_frames),
        },
        "frames": [frame],
    }
    _json(heldout_frames, heldout_doc)
    rgbd_manifest = union_frames.parent / "run_manifest.json"
    _json(
        rgbd_manifest,
        {
            "schema_version": "farm_rgbd_run_manifest_v1",
            "status": "complete",
            "qa_passed": True,
            "scene_id": "factory",
            "frames": "frames.json",
            "selected_view_count": 1,
            "fingerprint_payload": {
                "config": {
                    "meters_per_scene_unit": 1.0,
                    "depth_min_m": 0.05,
                    "depth_max_m": 80.0,
                    "radius_clip": 0.25,
                    "alpha_min": 0.05,
                },
                "inputs": {"ply": {"size": source_ply.stat().st_size}},
            },
        },
    )
    request = tmp_path / "request.json"
    _json(
        request,
        {
            "schema": PROJECTION_REQUEST_SCHEMA,
            "status": "frozen",
            "meters_per_scene_unit": 1.0,
            "heldout_frames_sha256": _sha(heldout_frames),
            "rgbd_render_manifest_sha256": _sha(rgbd_manifest),
            "union_frames_sha256": _sha(union_frames),
            "source_ply_sha256": _sha(source_ply),
            "source_labels_sha256": _sha(source_labels),
            "candidate_ply_sha256": _sha(candidate_ply),
            "candidate_labels_sha256": _sha(candidate_labels),
            "gaussian_counts": {"source": 3, "candidate": 3},
            "renderer_config": {
                "schema": PROJECTION_RENDERER_CONFIG_SCHEMA,
                "depth_min_m": 0.05,
                "depth_max_m": 80.0,
                "radius_clip": 0.25,
                "alpha_min": 0.05,
                "eps2d": 0.3,
                "packed": True,
                "tile_size": 16,
                "rasterize_mode": "antialiased",
                "camera_model": "pinhole",
                "render_mode": "RGB+ED",
                "object_depth": "opacity_weighted_expected_camera_z",
                "object_alpha_threshold": 0.1,
            },
            "renderer_runtime": {
                "image": "farm-prep:test",
                "image_id": "sha256:" + "1" * 64,
                "python": "/opt/conda/envs/rest3d/bin/python",
                "source": "candidate.upstream_lift_result.runtime",
            },
            "request_contract": {
                "view_selection_precommitted": True,
                "heldout_reference_metadata_only": True,
                "target_mask_paths_omitted": True,
                "target_mask_bytes_read": False,
                "renderer_must_not_receive_target_artifacts": True,
                "source_and_candidate_labels_frozen": True,
                "heldout_updates_candidate": False,
            },
            "observations": [
                {
                    "object_id": 7,
                    "source_image": "cam00_100_center.png",
                    "physical_timestamp": "100",
                }
            ],
        },
    )
    return {
        "request": request,
        "heldout_frames": heldout_frames,
        "rgbd_render_manifest": rgbd_manifest,
        "union_frames": union_frames,
        "source_ply": source_ply,
        "source_labels": source_labels,
        "candidate_ply": candidate_ply,
        "candidate_labels": candidate_labels,
    }


def _canonical_renderer_fixture(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    paths = _renderer_fixture(tmp_path)
    producer = tmp_path / "producer"
    producer.mkdir()
    actual_fit = producer / "actual_fit.json"
    source_manifest = producer / "source_manifest.json"
    source_union = producer / "source_union.json"
    source_fold = producer / "source_fold.json"
    rgb = producer / "frame.jpg"
    depth = producer / "frame.npy"
    actual_fit.write_text("{}", encoding="utf-8")
    source_manifest.write_text("{}", encoding="utf-8")
    source_union.write_text("{}", encoding="utf-8")
    source_fold.write_text("{}", encoding="utf-8")
    rgb.write_bytes(b"rgb")
    np.save(depth, np.ones((8, 8), dtype=np.float32))

    def spec(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
        return {
            "path": (
                os.path.relpath(path, relative_to).replace(os.sep, "/")
                if relative_to is not None
                else str(path)
            ),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }

    union = json.loads(paths["union_frames"].read_text(encoding="utf-8"))
    frame = union["frames"][0]
    frame["full_colmap_split"] = "heldout"
    frame["physical_timestamp"] = "100"
    frame["rgb_path"] = "../producer/frame.jpg"
    frame["depth_path"] = "../producer/frame.npy"
    _json(paths["union_frames"], union)
    heldout = json.loads(paths["heldout_frames"].read_text(encoding="utf-8"))
    heldout["frames"][0].update(frame)
    heldout["full_colmap_fold"]["source_union_frames_sha256"] = _sha(
        paths["union_frames"]
    )
    _json(paths["heldout_frames"], heldout)
    config = {
        "meters_per_scene_unit": 1.0,
        "depth_min_m": 0.05,
        "depth_max_m": 80.0,
        "radius_clip": 0.25,
        "alpha_min": 0.05,
    }
    manifest = {
        "schema_version": "farm.frozen-evaluation-rgbd-union-manifest.v1",
        "status": "frozen",
        "qa_passed": True,
        "scene_id": "factory",
        "selected_view_count": 1,
        "frames": "frames.json",
        "frames_artifact": spec(paths["union_frames"], relative_to=tmp_path / "rgbd"),
        "source_ply": spec(paths["source_ply"]),
        "actual_fit_frames": spec(actual_fit),
        "render_config": config,
        "source_rgbd_runs": [
            {
                "render_manifest": spec(source_manifest),
                "union_frames": spec(source_union),
                "fold_manifest": spec(source_fold),
            }
        ],
        "frame_artifacts": [
            {
                "source_image": "cam00_100_center.png",
                "physical_timestamp": "100",
                "split": "heldout",
                "rgb": spec(rgb, relative_to=tmp_path / "rgbd"),
                "depth": spec(depth, relative_to=tmp_path / "rgbd"),
            }
        ],
        "contract": {
            "read_only_existing_renders": True,
            "union_did_not_render_or_refit": True,
            "source_runs_qa_passed": True,
            "source_runs_sha256_bound": True,
            "all_frame_assets_sha256_hashed": True,
            "actual_fit_frames_sha256_bound": True,
            "heldout_state_fit_authorized": False,
        },
    }
    _json(paths["rgbd_render_manifest"], manifest)
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    request["heldout_frames_sha256"] = _sha(paths["heldout_frames"])
    request["union_frames_sha256"] = _sha(paths["union_frames"])
    request["rgbd_render_manifest_sha256"] = _sha(paths["rgbd_render_manifest"])
    request["rgbd_transitive_artifact_contract"] = _canonical_rgbd_transitive_contract(
        manifest, manifest_sha256=_sha(paths["rgbd_render_manifest"])
    )
    request["request_contract"].update(
        {
            "rgbd_transitive_artifacts_host_verified": True,
            "rgbd_transitive_declarations_sha256_bound": True,
            "renderer_rgbd_transitive_artifacts_unmounted": True,
        }
    )
    _json(paths["request"], request)
    return paths, producer


def test_isolated_renderer_validates_flat_canonical_mount_layout(
    tmp_path: Path,
) -> None:
    paths, producer = _canonical_renderer_fixture(tmp_path)
    validate_renderer_inputs(**{f"{name}_path": path for name, path in paths.items()})
    producer.rename(tmp_path / "producer_unmounted")
    request, frames, _hashes = validate_renderer_inputs(
        **{f"{name}_path": path for name, path in paths.items()},
        verify_rgbd_transitive_artifacts=False,
    )
    assert request["rgbd_transitive_artifact_contract"]["artifact_count"] == 6
    assert set(frames) == {"cam00_100_center.png"}
    with pytest.raises(ValueError, match="artifact is unavailable"):
        validate_renderer_inputs(
            **{f"{name}_path": path for name, path in paths.items()}
        )


def test_isolated_renderer_rejects_transitive_declaration_drift(
    tmp_path: Path,
) -> None:
    paths, _producer = _canonical_renderer_fixture(tmp_path)
    manifest = json.loads(paths["rgbd_render_manifest"].read_text(encoding="utf-8"))
    manifest["frame_artifacts"][0]["rgb"]["sha256"] = "0" * 64
    _json(paths["rgbd_render_manifest"], manifest)
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    request["rgbd_render_manifest_sha256"] = _sha(paths["rgbd_render_manifest"])
    _json(paths["request"], request)
    with pytest.raises(ValueError, match="transitive artifact contract mismatch"):
        validate_renderer_inputs(
            **{f"{name}_path": path for name, path in paths.items()},
            verify_rgbd_transitive_artifacts=False,
        )


def test_renderer_validation_is_target_blind_and_sha_bound(tmp_path: Path) -> None:
    paths = _renderer_fixture(tmp_path)
    request, frames, hashes = validate_renderer_inputs(
        **{f"{name}_path": path for name, path in paths.items()}
    )
    assert request["request_contract"]["target_mask_bytes_read"] is False
    assert set(frames) == {"cam00_100_center.png"}
    assert hashes["candidate_labels"] == _sha(paths["candidate_labels"])


def test_renderer_source_snapshot_is_complete_and_sha_bound(tmp_path: Path) -> None:
    snapshot = _snapshot_renderer_sources(tmp_path)
    assert "tools/farm_shaper_bridge/frozen_heldout_projection.py" in snapshot
    assert "tools/farm_shaper_bridge/gaussian_lift.py" in snapshot
    for spec in snapshot.values():
        path = tmp_path / spec["path"]
        assert path.is_file()
        assert spec["bytes"] == path.stat().st_size
        assert spec["sha256"] == _sha(path)


def test_renderer_rejects_target_field_and_artifact_drift(tmp_path: Path) -> None:
    paths = _renderer_fixture(tmp_path)
    payload = json.loads(paths["request"].read_text(encoding="utf-8"))
    payload["observations"][0]["target_mask"] = "forbidden.npy"
    _json(paths["request"], payload)
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_renderer_inputs(
            **{f"{name}_path": path for name, path in paths.items()}
        )

    paths = _renderer_fixture(tmp_path / "drift")
    np.save(paths["candidate_labels"], np.array([7, -1, -1], np.int32))
    with pytest.raises(ValueError, match="candidate_labels_sha256 mismatch"):
        validate_renderer_inputs(
            **{f"{name}_path": path for name, path in paths.items()}
        )


def test_reference_routing_does_not_open_declared_target_mask(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    _json(
        reference,
        {
            "schema": "farm.full-colmap-mask-refinement.v1",
            "status": "PASS",
            "refinement_role": "heldout_reference_only",
            "input_view_contract": {
                "active_split": "heldout",
                "state_fit_authorized": False,
                "requested_view_role": "heldout-reference",
            },
            "policy": {
                "state_fit_authorized": False,
                "merge_apply_forbidden": True,
                "heldout_reference_is_evaluation_only": True,
                "reference_masks_are_pseudo_labels_not_ground_truth": True,
            },
            "hashes": {"rescue_frames_sha256": "2" * 64},
            "objects": [
                {
                    "object_id": 7,
                    "views": [
                        {
                            "accepted": True,
                            "source_image": "cam00_100_center.png",
                            "physical_timestamp_ns": 100,
                            "mask_relative": "does/not/exist.npz",
                            "mask_sha256": "3" * 64,
                        }
                    ],
                }
            ],
        },
    )
    rows = _heldout_reference_rows(
        reference,
        heldout_frames_sha256="2" * 64,
        candidate_ids={7},
        heldout={"cam00_100_center.png": {"physical_timestamp": "100"}},
        membership={7: {"heldout": ["cam00_100_center.png"]}},
        hashes={},
    )
    assert rows == [
        {
            "object_id": 7,
            "source_image": "cam00_100_center.png",
            "physical_timestamp": "100",
        }
    ]


def test_lift_fit_contract_opens_train_only_and_rejects_nontrain_frame(
    tmp_path: Path,
) -> None:
    frame_row = _camera("cam00_100_center.png")
    train_path = tmp_path / "train" / "frames.json"
    _json(
        train_path,
        {
            "schema_version": "farm_frames_json_v1",
            "scene_id": "factory",
            "depth_units": "metres",
            "pose_translation_units": "metres",
            "meters_per_scene_unit": 1.0,
            "full_colmap_fold": {"role": "train", "state_fit_authorized": True},
            "frames": [frame_row],
        },
    )
    manifest = tmp_path / "manifest.json"
    _json(
        manifest,
        {
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
                    "source_images": ["cam00_100_center.png"],
                },
                "heldout": {
                    "frames_json": "heldout/frames.json",
                    "sha256": "4" * 64,
                    "source_images": ["cam00_200_center.png"],
                },
            },
        },
    )
    frame = SimpleNamespace(
        source_image="cam00_100_center.png",
        frame_id="100",
        timestamp_ns=100,
        depth_size=(8, 8),
        K=np.asarray(frame_row["K"]),
        T_world_cam=np.asarray(frame_row["T_world_cam"]),
    )
    run = SimpleNamespace(
        scene_id="factory", meters_per_scene_unit=1.0, frames=(frame,)
    )
    contract = validate_full_colmap_train_fit(run, manifest)
    assert contract["fit_splits"] == ["train"]
    assert contract["heldout_frames_opened_by_lift"] is False
    assert not (tmp_path / "heldout" / "frames.json").exists()

    bad = SimpleNamespace(**{**frame.__dict__, "source_image": "cam00_200_center.png"})
    with pytest.raises(ValueError, match="absent from external train fold"):
        validate_full_colmap_train_fit(
            SimpleNamespace(
                scene_id="factory", meters_per_scene_unit=1.0, frames=(bad,)
            ),
            manifest,
        )


def test_gpu_entrypoint_hardcodes_unmounted_transitive_commitment_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StopValidation(Exception):
        pass

    captured: dict[str, object] = {}

    def stop_validation(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise StopValidation

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        renderer_entrypoint, "validate_renderer_inputs", stop_validation
    )
    args = Namespace(
        request=tmp_path / "request.json",
        heldout_frames=tmp_path / "heldout.json",
        rgbd_render_manifest=tmp_path / "rgbd.json",
        union_frames=tmp_path / "union.json",
        source_ply=tmp_path / "source.ply",
        source_labels=tmp_path / "source.npy",
        candidate_ply=tmp_path / "candidate.ply",
        candidate_labels=tmp_path / "candidate.npy",
        output_dir=tmp_path / "output",
        maximum_object_channels=15,
    )
    with pytest.raises(StopValidation):
        renderer_entrypoint.run(args)
    assert captured["kwargs"] == {"verify_rgbd_transitive_artifacts": False}


def test_public_projection_launcher_deep_preflights_without_transitive_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths: dict[str, Path] = {}
    suffixes = {
        "request": ".json",
        "heldout_frames": ".json",
        "rgbd_render_manifest": ".json",
        "union_frames": ".json",
        "source_ply": ".ply",
        "source_labels": ".npy",
        "candidate_ply": ".ply",
        "candidate_labels": ".npy",
    }
    for name, suffix in suffixes.items():
        path = inputs / f"{name}{suffix}"
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    commitment = {
        "schema": "farm.rgbd-transitive-artifact-commitment.v1",
        "rgbd_render_manifest_sha256": "1" * 64,
        "source_run_count": 2,
        "frame_count": 870,
        "artifact_count": 1747,
        "entries_sha256": "2" * 64,
        "paths_are_host_provenance_not_container_authority": True,
    }
    request = {
        "renderer_runtime": {
            "image": "farm:test",
            "image_id": "sha256:" + "3" * 64,
            "python": "/opt/conda/envs/rest3d/bin/python",
        },
        "rgbd_transitive_artifact_contract": commitment,
        "observations": [{"object_id": 7}],
    }
    captured: dict[str, object] = {}

    def validate(*args: object, **kwargs: object):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return request, {}, {}

    monkeypatch.setattr(projection_runner, "validate_renderer_inputs", validate)
    monkeypatch.setattr(
        projection_runner, "_docker_image_id", lambda _image: "sha256:" + "3" * 64
    )
    monkeypatch.setattr(
        projection_runner, "_git_state", lambda _repo: ("4" * 40, False)
    )
    args = Namespace(
        **paths,
        output_dir=tmp_path / "outputs" / "projection",
        gpu="0",
        maximum_object_channels=15,
        plan_only=True,
        allow_dirty_source_nonrelease=False,
        allow_rebuilt_image_nonrelease=False,
        print_command=False,
    )
    command, metadata = projection_runner.build_command(args)
    assert captured["kwargs"] == {"verify_rgbd_transitive_artifacts": True}
    mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]
    assert len(mounts) == 9  # repository source plus exactly eight direct inputs
    assert all(mount.endswith(",readonly") for mount in mounts)
    assert not any("rgbd_transitive" in mount for mount in mounts)
    assert "--gpus" not in command
    assert metadata["rgbd_transitive_artifact_policy"] == (
        "host-verified-request-bound-unmounted"
    )
    assert metadata["rgbd_transitive_artifact_count"] == 1747
    assert metadata["rgbd_transitive_artifact_commitment_sha256"] == "2" * 64
    assert metadata["target_artifacts_mounted"] is False
