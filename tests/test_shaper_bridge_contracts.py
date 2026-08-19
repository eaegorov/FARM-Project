from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.source_snapshot import SourceSnapshotIntegrityError

from tools.farm_shaper_bridge.assemble_scene import (
    rotation_from_to,
    validate_batch_farm_binding,
)
from tools.farm_shaper_bridge.build_shaper_inputs import (
    _spatial_sample,
    object_transforms,
    select_diverse_views,
)
from tools.farm_shaper_bridge.run_shaper import (
    archive_clean_shaper_source,
    build_checkpoint_mount_smoke_command,
    build_docker_command,
    validate_huggingface_cache,
)
from tools.farm_shaper_bridge.manage_runtime import build_command, smoke_command, smoke_runtime
from tools.farm_shaper_bridge.shaper_batch import _runtime_contract
from tools.farm_shaper_bridge.shaper_contracts import (
    bind_verified_lift,
    load_shaper_config,
    load_verified_instance_bank,
    resolve_farm_source_authority,
    safe_name,
)
from tools.farm_shaper_bridge.common import RunData, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_incomplete_signed_snapshot_fallback_is_explicit_and_nonrelease_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    snapshot = run / "source_snapshot/FARM-Project"
    live = tmp_path / "live"
    snapshot.mkdir(parents=True)
    required = (
        "tools/farm_shaper_bridge/build_shaper_inputs.py",
        "configs/shaper_bridge.v1.yaml",
    )
    for relative in required:
        path = live / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"versioned:{relative}\n", encoding="utf-8")
    snapshot_manifest = run / "source_snapshot/manifest.json"
    snapshot_manifest.write_text('{"schema":"fixture"}\n', encoding="utf-8")
    tree_sha = "a" * 64
    (run / "manifest.json").write_text(json.dumps({
        "source_snapshot": {
            "tree_sha256": tree_sha,
            "manifest_relative_path": "source_snapshot/manifest.json",
        }
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.farm_shaper_bridge.shaper_contracts.validated_source_snapshot_project_root",
        lambda *_args, **_kwargs: snapshot,
    )
    for allow_unsigned, allow_fallback in ((False, True), (True, False)):
        with pytest.raises(SourceSnapshotIntegrityError, match="predates"):
            resolve_farm_source_authority(
                run,
                live,
                required_files=required,
                allow_unsigned=allow_unsigned,
                allow_incomplete_snapshot_fallback=allow_fallback,
            )
    authority = resolve_farm_source_authority(
        run,
        live,
        required_files=required,
        allow_unsigned=True,
        allow_incomplete_snapshot_fallback=True,
    )
    assert authority.project_root == live.resolve()
    assert authority.signed is False
    assert authority.tree_sha256 is None
    assert authority.fallback is not None
    assert authority.fallback["warning"] == (
        "legacy_unsigned_fallback_from_incomplete_signed_snapshot"
    )
    assert authority.fallback["incomplete_snapshot_project_root"] == str(snapshot.resolve())
    assert authority.fallback["incomplete_snapshot_tree_sha256"] == tree_sha
    assert authority.fallback["incomplete_snapshot_manifest_sha256"] == sha256_file(
        snapshot_manifest
    )
    evidence = authority.fallback["selected_live_source"]
    assert evidence["project_root"] == str(live.resolve())
    assert set(evidence["required_files_sha256"]) == set(required)
    assert len(evidence["required_files_bundle_sha256"]) == 64


def _write_bank(path: Path, *, duplicate: bool = False) -> str:
    indices = np.asarray([2, 8, 8 if duplicate else 11, 15], dtype=np.int64)
    np.savez(
        path,
        object_ids=np.asarray([3, 7], dtype=np.int32),
        indptr=np.asarray([0, 2, 4], dtype=np.int64),
        indices=indices,
        confidence=np.asarray([0.9, 0.7, 0.8, 0.6], dtype=np.float32),
        timestamp_support=np.asarray([3, 2, 4, 2], dtype=np.uint16),
    )
    return sha256_file(path)


def test_verified_bank_is_sparse_source_order_and_object_sliced(tmp_path: Path) -> None:
    path = tmp_path / "verified_instance_bank.npz"
    digest = _write_bank(path)
    # A ten-million-row source does not allocate any ten-million-row label
    # vector: the bridge consumes only the four CSR members.
    bank = load_verified_instance_bank(
        path,
        source_gaussian_count=10_000_000,
        expected_sha256=digest,
        expected_objects=2,
        expected_gaussians=4,
    )
    assert bank.object_slice(0)[0] == 3
    assert bank.object_slice(0)[1].tolist() == [2, 8]
    assert bank.object_slice(1)[0] == 7
    assert bank.object_slice(1)[1].tolist() == [11, 15]
    assert bank.indices.nbytes == 4 * np.dtype("int64").itemsize


def test_verified_bank_rejects_cross_object_duplicate_source_row(tmp_path: Path) -> None:
    path = tmp_path / "verified_instance_bank.npz"
    digest = _write_bank(path, duplicate=True)
    with pytest.raises(ValueError, match="multiple verified objects"):
        load_verified_instance_bank(
            path,
            source_gaussian_count=20,
            expected_sha256=digest,
        )


def test_lift_binding_requires_exact_run_source_and_heldout_verified_bank(
    tmp_path: Path,
) -> None:
    lift = tmp_path / "lift"
    lift.mkdir()
    bank_path = lift / "verified_instance_bank.npz"
    bank_sha = _write_bank(bank_path)
    source_sha = "c" * 64
    run = RunData(
        run_dir=tmp_path / "run",
        scene_id="knaack",
        frames_doc={},
        frames=(),
        objects=(),
        meters_per_scene_unit=1.0,
        rgbd_config={},
        success_sha256="a" * 64,
        acceptance_sha256="b" * 64,
        resource_preflight={},
        scene_preflight={},
        legacy=False,
    )
    result = {
        "schema_version": "farm.gaussian-lift.result.v1",
        "status": "PASS",
        "release_eligible": True,
        "scene_id": "knaack",
        "inputs": {
            "farm_success_sha256": run.success_sha256,
            "farm_acceptance_sha256": run.acceptance_sha256,
            "source_gaussian_count": 20,
            "source_ply_sha256": source_sha,
        },
        "contracts": {
            "verified_only_canonical_labels": True,
            "unknown_instance_id": -1,
        },
        "counts": {"verified_objects": 2, "verified_gaussians": 4},
        "qc_summary": {"verified_object_ids": [3, 7]},
        "artifacts": {
            "final": {
                "verified_instance_bank": {
                    "path": bank_path.name,
                    "bytes": bank_path.stat().st_size,
                    "sha256": bank_sha,
                }
            }
        },
    }
    raw = (json.dumps(result) + "\n").encode()
    (lift / "result.json").write_bytes(raw)
    marker = {
        "status": "success",
        "release_eligible": True,
        "scene_id": "knaack",
        "result": "result.json",
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "source_ply_sha256": source_sha,
    }
    (lift / "_SUCCESS.json").write_text(json.dumps(marker) + "\n")
    binding = bind_verified_lift(
        lift,
        run=run,
        source_ply_path=tmp_path / "source.ply",
        source_gaussian_count=20,
        source_ply_sha256=source_sha,
        allow_nonrelease=False,
    )
    assert binding.release_eligible is True
    assert binding.bank.object_ids.tolist() == [3, 7]
    with pytest.raises(ValueError, match="not bound"):
        bind_verified_lift(
            lift,
            run=run,
            source_ply_path=tmp_path / "source.ply",
            source_gaussian_count=21,
            source_ply_sha256=source_sha,
            allow_nonrelease=False,
        )


def test_builder_has_no_legacy_dense_instance_scan() -> None:
    source = (ROOT / "tools/farm_shaper_bridge/build_shaper_inputs.py").read_text(
        encoding="utf-8"
    )
    assert "instance_labels.npz" not in source
    assert "flatnonzero(labels" not in source
    assert "verified_instance_bank.npz" in (
        ROOT / "tools/farm_shaper_bridge/shaper_contracts.py"
    ).read_text(encoding="utf-8")


def test_scene_generic_names_and_object_transform_roundtrip() -> None:
    assert safe_name("Knaack / Hall #2") == "knaack_hall_2"
    center = np.asarray([1.0, -2.0, 0.5])
    angle = np.deg2rad(35.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    world_to_model, model_to_world = object_transforms(center, rotation)
    points = np.asarray([[1.2, -1.7, 0.2], [-0.5, 3.0, 2.0]])
    model = points @ world_to_model[:3, :3].T + world_to_model[:3, 3]
    restored = model @ model_to_world[:3, :3].T + model_to_world[:3, 3]
    assert np.allclose(restored, points, atol=1e-12)


def test_spatial_sample_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(500, 3)).astype(np.float32)
    confidence = rng.uniform(0.4, 1.0, size=500).astype(np.float32)
    support = rng.integers(1, 6, size=500, dtype=np.uint16)
    first = _spatial_sample(points, confidence, support, 64)
    second = _spatial_sample(points, confidence, support, 64)
    assert np.array_equal(first, second)
    assert len(first) == 64
    assert np.all(np.diff(first) > 0)


def test_view_selector_rewards_physical_and_family_diversity() -> None:
    candidates = [
        {
            "image_id": 0, "frame_id": "t0", "timestamp_ns": 100, "sensor": "left", "family": "center",
            "visible_count": 1000, "view_direction": np.asarray([1.0, 0.0, 0.0]),
        },
        {
            "image_id": 1, "frame_id": "t0", "timestamp_ns": 100, "sensor": "left", "family": "center",
            "visible_count": 990, "view_direction": np.asarray([0.99, 0.01, 0.0]),
        },
        {
            "image_id": 2, "frame_id": "t1", "timestamp_ns": 200, "sensor": "right", "family": "side",
            "visible_count": 700, "view_direction": np.asarray([0.0, 1.0, 0.0]),
        },
    ]
    chosen = select_diverse_views(candidates, 2)
    assert [row["image_id"] for row in chosen] == [0, 2]


def test_pinned_config_and_exact_output_mount_contract(tmp_path: Path) -> None:
    config, config_sha = load_shaper_config(ROOT / "configs/shaper_bridge.v1.yaml")
    paths = {}
    for name in ("farm", "shaper", "checkpoints", "hf", "inputs", "output"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    config_path = ROOT / "configs/shaper_bridge.v1.yaml"
    checkpoint_file = paths["checkpoints"] / "model.ckpt"
    checkpoint_file.write_bytes(b"pinned")
    pins = {
        "image_id": config["runtime"]["image_id"],
        "shaper_commit": config["runtime"]["shaper_commit"],
        "shaper_tree": "a" * 40,
        "farm_source_tree": "b" * 64,
        "config_sha256": config_sha,
        "runtime_pin_sha256": "c" * 64,
    }
    command = build_docker_command(
        farm_source=paths["farm"],
        shaper_source=paths["shaper"],
        checkpoint_files={checkpoint_file.name: checkpoint_file},
        hf_cache=paths["hf"],
        inputs=paths["inputs"],
        output=paths["output"],
        config_path=config_path,
        profile="balance",
        gpu="0",
        config=config,
        pins=pins,
    )
    mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
    output_mount = f"type=bind,src={paths['output']},dst=/farm/output"
    assert mounts.count(output_mount) == 1
    assert not output_mount.endswith(",readonly")
    assert all(
        mount.endswith(",readonly")
        for mount in mounts
        if mount != output_mount
    )
    assert not any(f"src={tmp_path}," in mount for mount in mounts)
    assert f"type=bind,src={checkpoint_file},dst=/opt/shaper/checkpoints/model.ckpt,readonly" in mounts
    assert not any(f"src={paths['checkpoints']}," in mount for mount in mounts)
    assert config["runtime"]["image_id"] in command
    assert command[command.index("--entrypoint") + 1] == config["runtime"]["python"]


def test_shape_source_archive_ignores_dirty_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "ShapeR"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tracked = repo / "dataset/shaper_dataset.py"
    tracked.parent.mkdir()
    tracked.write_text("COMMITTED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "commit", "-q", "-m", "fixture",
    ], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    tracked.write_text("DIRTY = True\n", encoding="utf-8")
    (repo / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
    archived = archive_clean_shaper_source(
        repo,
        commit,
        tmp_path / "archive",
        checkpoint_names=("model.ckpt", "config.yaml"),
    )
    assert (archived / "dataset/shaper_dataset.py").read_text() == "COMMITTED = True\n"
    assert not (archived / "untracked.py").exists()
    assert (archived / "checkpoints").is_dir()
    assert (archived / "checkpoints/model.ckpt").read_bytes() == b""
    assert (archived / "checkpoints/config.yaml").read_bytes() == b""


def test_checkpoint_mount_smoke_matches_production_nested_topology(tmp_path: Path) -> None:
    shaper = tmp_path / "ShapeR"
    checkpoints = shaper / "checkpoints"
    checkpoints.mkdir(parents=True)
    placeholder = checkpoints / "model.ckpt"
    placeholder.touch()
    source = tmp_path / "assets/model.ckpt"
    source.parent.mkdir()
    source.write_bytes(b"exact-pinned-checkpoint")
    command = build_checkpoint_mount_smoke_command(
        image_id="sha256:" + "a" * 64,
        python="/opt/shaper-venv/bin/python",
        shaper_source=shaper,
        checkpoint_files={"model.ckpt": source},
    )
    mounts = [command[index + 1] for index, item in enumerate(command[:-1]) if item == "--mount"]
    assert mounts == [
        f"type=bind,src={shaper},dst=/opt/shaper,readonly",
        f"type=bind,src={source},dst=/opt/shaper/checkpoints/model.ckpt,readonly",
    ]
    placeholder.write_bytes(b"not-empty")
    with pytest.raises(RuntimeError, match="absent or nonempty"):
        build_checkpoint_mount_smoke_command(
            image_id="sha256:" + "a" * 64,
            python="/opt/shaper-venv/bin/python",
            shaper_source=shaper,
            checkpoint_files={"model.ckpt": source},
        )


def test_versioned_runtime_build_plan_is_explicit_and_repin_gated() -> None:
    config_path = ROOT / "configs/shaper_bridge.v1.yaml"
    config, _ = load_shaper_config(config_path)
    command, plan = build_command(
        config=config,
        config_path=config_path,
        image=None,
        cuda_base_image=None,
        python_version=None,
        torch_cuda_arch_list=None,
    )
    dockerfile = ROOT / "docker/Dockerfile.shaper"
    assert command[:2] == ["docker", "build"]
    assert command[command.index("--file") + 1] == str(dockerfile)
    assert command[command.index("--tag") + 1] == config["runtime"]["image"]
    assert command[-1] == str(ROOT)
    assert "BUILT_NOT_PINNED" not in command
    assert plan["bit_reproducible"] is False
    assert "repin" in plan["release_authority"]
    assert plan["dockerfile_sha256"] == sha256_file(dockerfile)


def test_runtime_smoke_uses_immutable_image_and_no_network() -> None:
    image_id = "sha256:" + "a" * 64
    command = smoke_command(image_id, "/opt/shaper-venv/bin/python")
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--gpus") + 1] == "all"
    assert "--read-only" in command
    assert command[command.index("--entrypoint") + 1] == "/opt/shaper-venv/bin/python"
    assert image_id in command
    with pytest.raises(ValueError, match="immutable"):
        smoke_command("shaper:latest", "/opt/shaper-venv/bin/python")


def test_runtime_smoke_rejects_cpu_only_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.farm_shaper_bridge.manage_runtime._stdout",
        lambda _command: json.dumps({"packages": {}, "torch_cuda": "12.8", "cuda_available": False}),
    )
    with pytest.raises(RuntimeError, match="CUDA-visible GPU"):
        smoke_runtime(
            "sha256:" + "a" * 64,
            python="/opt/shaper-venv/bin/python",
            expected={"packages": {}, "torch_cuda": "12.8"},
        )


def test_batch_runtime_accepts_git_tree_sha1_and_sha256_provenance(monkeypatch) -> None:
    config, config_sha = load_shaper_config(ROOT / "configs/shaper_bridge.v1.yaml")
    values = {
        "FARM_SHAPER_IMAGE_ID": config["runtime"]["image_id"],
        "FARM_SHAPER_COMMIT": config["runtime"]["shaper_commit"],
        "FARM_SHAPER_TREE": "a" * 40,
        "FARM_SOURCE_TREE": "b" * 64,
        "FARM_SHAPER_CONFIG_SHA256": config_sha,
        "FARM_SHAPER_RUNTIME_PIN_SHA256": "c" * 64,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    observed = _runtime_contract(config, config_sha)
    assert observed["shaper_tree"] == "a" * 40
    assert observed["farm_source_tree"] == "b" * 64


def test_offline_hf_cache_requires_exact_revision_blob_and_size(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    model = cache / "hub/models--org--model"
    payload = b"model-bytes"
    git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
    blob = model / "blobs" / git_blob
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    revision = "c" * 40
    (model / "refs").mkdir()
    (model / "refs/main").write_text(revision + "\n", encoding="utf-8")
    snapshot = model / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").symlink_to(Path("../../blobs") / blob.name)
    expected = {
        model.name: {
            "repository": "org/model",
            "revision": revision,
            "files": {"model.bin": {"blob": blob.name, "bytes": len(payload)}},
        }
    }
    rows = validate_huggingface_cache(cache, expected)
    assert rows[model.name]["revision"] == revision
    assert rows[model.name]["files"]["model.bin"]["blob"] == blob.name
    assert rows[model.name]["files"]["model.bin"]["content_digest_algorithm"] == "git-blob-sha1"
    blob.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="blob drift"):
        validate_huggingface_cache(cache, expected)


def test_offline_hf_cache_verifies_raw_sha256_blob_content(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    model = cache / "hub/models--org--large"
    payload = b"lfs-content"
    digest = hashlib.sha256(payload).hexdigest()
    blob = model / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    revision = "d" * 40
    (model / "refs").mkdir()
    (model / "refs/main").write_text(revision + "\n", encoding="utf-8")
    snapshot = model / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").symlink_to(Path("../../blobs") / digest)
    expected = {
        model.name: {
            "repository": "org/large", "revision": revision,
            "files": {"model.bin": {"blob": digest, "bytes": len(payload)}},
        }
    }
    rows = validate_huggingface_cache(cache, expected)
    assert rows[model.name]["files"]["model.bin"]["content_digest_algorithm"] == "sha256"
    blob.write_bytes(b"same-length?")
    with pytest.raises(RuntimeError, match="content drift|blob drift"):
        validate_huggingface_cache(cache, expected)


def test_world_up_rotation_is_proper_for_antiparallel_case() -> None:
    transform = rotation_from_to([0, -1, 0], [0, 1, 0])
    assert np.allclose(transform[:3, :3] @ [0, -1, 0], [0, 1, 0])
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0)


def test_scene_assembly_requires_exact_farm_scene_success_and_acceptance(tmp_path: Path) -> None:
    run = RunData(
        run_dir=tmp_path / "run-v3", scene_id="factory", frames_doc={}, frames=(), objects=(),
        meters_per_scene_unit=1.0, rgbd_config={}, success_sha256="a" * 64,
        acceptance_sha256="b" * 64, resource_preflight={}, scene_preflight={}, legacy=False,
    )
    batch = {
        "scene_id": "factory",
        "inputs": {"farm_success_sha256": "a" * 64, "farm_acceptance_sha256": "b" * 64},
    }
    validate_batch_farm_binding(batch, run)
    for key, value in (
        ("scene_id", "knaack"),
        ("farm_success_sha256", "c" * 64),
        ("farm_acceptance_sha256", "d" * 64),
    ):
        broken = json.loads(json.dumps(batch))
        if key == "scene_id":
            broken[key] = value
        else:
            broken["inputs"][key] = value
        with pytest.raises(ValueError, match="exact FARM"):
            validate_batch_farm_binding(broken, run)
