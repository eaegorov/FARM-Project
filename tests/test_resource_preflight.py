from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_pipeline.resources import (
    GpuInfo,
    MemoryPolicy,
    compute_memory_plan,
    load_model_manifest,
    parse_nvidia_smi_csv,
    run_resource_preflight,
    select_gpu,
)


IMAGE_ID = "sha256:" + "1" * 64
REVISION = "a" * 40


def _gpu(*, total_gib: int = 80, free_gib: int = 70) -> GpuInfo:
    return GpuInfo(
        index=0,
        uuid="GPU-00000000-0000-0000-0000-000000000000",
        name="Synthetic GPU",
        memory_total_mib=total_gib * 1024,
        memory_free_mib=free_gib * 1024,
        compute_capability="9.0",
        driver_version="999.0",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cache = tmp_path / "cache"
    model_root = cache / "hub/models--Example--Caption"
    snapshot = model_root / "snapshots" / REVISION
    blobs = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    payload = b"synthetic pinned model"
    digest = hashlib.sha256(payload).hexdigest()
    (blobs / digest).write_bytes(payload)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.bin").symlink_to(f"../../blobs/{digest}")

    local_model = tmp_path / "weights.bin"
    local_model.write_bytes(b"synthetic local weights")
    local_digest = hashlib.sha256(local_model.read_bytes()).hexdigest()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    manifest_path = config_dir / "models.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "farm.models.v1",
        "cache_root": "../cache",
        "runtimes": {
            "main": {
                "image": "synthetic:latest",
                "image_id": IMAGE_ID,
                "python": "/opt/synthetic/bin/python",
            },
        },
        "services": {
            "caption": {
                "runtime": "main",
                "served_model_name": "synthetic-caption",
                "port": 8000,
                "model": {
                    "kind": "huggingface_snapshot",
                    "repo_id": "Example/Caption",
                    "revision": REVISION,
                    "required_files": [
                        "config.json",
                        {
                            "path": "model.bin",
                            "sha256": digest,
                            "checksum_mode": "content_addressed",
                        },
                    ],
                },
                "memory": {
                    "target_allocated_gib": 24,
                    "minimum_total_gib": 39,
                    "reserve_free_gib": 6,
                    "utilization_min": 0.30,
                    "utilization_max": 0.65,
                },
                "vllm_args": ["--dtype", "half"],
            },
        },
        "pipeline_models": {
            "local-test": {
                "kind": "local_file",
                "path": "../weights.bin",
                "sha256": local_digest,
            },
        },
    }, indent=2), encoding="utf-8")
    secret = tmp_path / "secrets.json"
    secret.write_text(json.dumps({"HF_TOKEN": "hf_DO_NOT_LEAK_SENTINEL"}), encoding="utf-8")
    secret.chmod(0o600)
    return manifest_path, local_model, secret


def _run(tmp_path: Path, **kwargs):
    manifest_path, _local_model, secret = _fixture(tmp_path)
    manifest = load_model_manifest(manifest_path)
    return run_resource_preflight(
        manifest,
        gpu_identifier="0",
        services=["caption"],
        secrets_file=secret,
        gpu_inventory=[_gpu()],
        docker_image_resolver=lambda _image: IMAGE_ID,
        **kwargs,
    )


def test_gpu_inventory_parser_and_explicit_uuid_selection() -> None:
    payload = (
        "0, GPU-aaaa, NVIDIA H100 80GB HBM3, 81559, 72209, 9.0, 580.126.09\n"
        "1, GPU-bbbb, NVIDIA L40S, 46068, 40000, 8.9, 580.126.09\n"
    )
    gpus = parse_nvidia_smi_csv(payload)
    assert len(gpus) == 2
    assert gpus[0].memory_total_mib == 81559
    assert select_gpu(gpus, "GPU-bbbb").index == 1
    with pytest.raises(ValueError, match="did not resolve uniquely"):
        select_gpu(gpus, "2")


def test_caption_memory_plan_is_adaptive_but_bounded() -> None:
    policy = MemoryPolicy(24.0, 39.0, 6.0, 0.30, 0.65)
    h100 = compute_memory_plan(_gpu(total_gib=80, free_gib=70), policy)
    gpu48 = compute_memory_plan(_gpu(total_gib=48, free_gib=40), policy)
    gpu32 = compute_memory_plan(_gpu(total_gib=32, free_gib=32), policy)
    assert h100["gpu_memory_utilization"] == pytest.approx(0.30)
    assert gpu48["gpu_memory_utilization"] == pytest.approx(0.50)
    assert h100["ready"] and gpu48["ready"]
    assert not gpu32["ready"]
    assert not gpu32["total_ok"] and not gpu32["target_reachable"]


def test_manifest_paths_snapshot_revision_and_content_hash_pass(tmp_path: Path) -> None:
    manifest_path, _local_model, _secret = _fixture(tmp_path)
    manifest = load_model_manifest(manifest_path)
    service = manifest.services["caption"]
    assert manifest.cache_root == (tmp_path / "cache").resolve()
    assert service.model.revision == REVISION
    assert service.model.local_path == (
        tmp_path / "cache/hub/models--Example--Caption/snapshots" / REVISION
    ).resolve()
    assert manifest.runtimes["main"].python == "/opt/synthetic/bin/python"
    report = _run(tmp_path / "run")
    assert report.status == "pass", report.to_dict()
    assert report.services["caption"]["container_model_path"].endswith(REVISION)
    runtime = next(check for check in report.checks if check.name == "runtime:main")
    assert runtime.metrics["python"] == "/opt/synthetic/bin/python"


@pytest.mark.parametrize("value", ["bin/python", "/opt/../bin/python", "/opt//bin/python"])
def test_runtime_python_must_be_a_normalized_absolute_path(
    tmp_path: Path, value: str
) -> None:
    manifest_path, _local_model, _secret = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["runtimes"]["main"]["python"] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized absolute path"):
        load_model_manifest(manifest_path)


def test_secret_value_is_never_serialized(tmp_path: Path) -> None:
    report = _run(tmp_path)
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert "hf_DO_NOT_LEAK_SENTINEL" not in serialized
    secret_check = next(check for check in report.checks if check.name == "secret")
    assert secret_check.metrics["has_hf_token"] is True
    assert "token_length" not in serialized


def test_pipeline_model_checksum_mismatch_fails(tmp_path: Path) -> None:
    manifest_path, local_model, secret = _fixture(tmp_path)
    manifest = load_model_manifest(manifest_path)
    local_model.write_bytes(b"tampered")
    report = run_resource_preflight(
        manifest,
        gpu_identifier=0,
        services=["caption"],
        secrets_file=secret,
        include_pipeline_models=True,
        gpu_inventory=[_gpu()],
        docker_image_resolver=lambda _image: IMAGE_ID,
    )
    assert report.status == "fail"
    assert any(
        finding.code == "model.checksum_mismatch"
        for check in report.checks
        for finding in check.findings
    )


def test_docker_image_id_mismatch_fails(tmp_path: Path) -> None:
    manifest_path, _local_model, secret = _fixture(tmp_path)
    report = run_resource_preflight(
        load_model_manifest(manifest_path),
        gpu_identifier=0,
        services=["caption"],
        secrets_file=secret,
        gpu_inventory=[_gpu()],
        docker_image_resolver=lambda _image: "sha256:" + "2" * 64,
    )
    assert report.status == "fail"
    assert any(
        finding.code == "runtime.image_id_mismatch"
        for check in report.checks
        for finding in check.findings
    )


def test_cli_scripts_have_valid_syntax_and_launcher_is_executable() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts/start_farm_vllm.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    assert os.access(launcher, os.X_OK)
    source = launcher.read_text(encoding="utf-8")
    assert "No available memory for the cache blocks" in source
    assert "bounded memory retry" in source
    assert "memory_retry_used == 0" in source
