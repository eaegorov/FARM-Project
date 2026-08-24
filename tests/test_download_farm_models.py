from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_farm_models.py"
SPEC = importlib.util.spec_from_file_location("download_farm_models", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
download = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download
SPEC.loader.exec_module(download)


IMAGE_ID = "sha256:" + "1" * 64
SERVICE_REVISION = "a" * 40
LOCAL_REVISION = "b" * 40
SERVICE_BYTES = b"service model bytes"
LOCAL_HF_BYTES = b"local Hugging Face model bytes"
URL_BYTES = b"URL model bytes"
TOKEN = "hf_DO_NOT_LEAK_MODEL_BOOTSTRAP_TOKEN"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "models.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "farm.models.v1",
                "cache_root": "../cache",
                "runtimes": {
                    "main": {"image": "farm:test", "image_id": IMAGE_ID},
                },
                "services": {
                    "caption": {
                        "runtime": "main",
                        "served_model_name": "caption-test",
                        "port": 8000,
                        "model": {
                            "kind": "huggingface_snapshot",
                            "repo_id": "Example/Caption",
                            "revision": SERVICE_REVISION,
                            "required_files": [
                                "config.json",
                                {
                                    "path": "model.bin",
                                    "sha256": _sha(SERVICE_BYTES),
                                    "checksum_mode": "full",
                                },
                            ],
                        },
                        "memory": {
                            "target_allocated_gib": 1,
                            "minimum_total_gib": 2,
                            "reserve_free_gib": 1,
                            "utilization_min": 0.1,
                            "utilization_max": 0.5,
                        },
                        "vllm_args": ["--dtype", "half"],
                    },
                },
                "pipeline_models": {
                    "detector": {
                        "kind": "local_file",
                        "path": "../models/detector.bin",
                        "sha256": _sha(URL_BYTES),
                        "source": {
                            "kind": "url",
                            "url": "https://models.example.invalid/detector.bin",
                        },
                    },
                    "visual-backbone": {
                        "kind": "local_file",
                        "path": "../models/visual/model.safetensors",
                        "sha256": _sha(LOCAL_HF_BYTES),
                        "source": {
                            "kind": "huggingface_snapshot",
                            "repo_id": "Example/Visual",
                            "revision": LOCAL_REVISION,
                            "required_files": ["config.json", "model.safetensors"],
                        },
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _fake_snapshots(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(dict(kwargs))
        if "local_dir" in kwargs:
            target = Path(kwargs["local_dir"])
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}\n", encoding="utf-8")
            (target / "model.safetensors").write_bytes(LOCAL_HF_BYTES)
            return str(target)
        target = (
            Path(kwargs["cache_dir"])
            / "models--Example--Caption"
            / "snapshots"
            / SERVICE_REVISION
        )
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}\n", encoding="utf-8")
        (target / "model.bin").write_bytes(SERVICE_BYTES)
        return str(target)

    monkeypatch.setattr(download, "_snapshot_download", fake_snapshot_download)
    return calls


def test_bootstrap_uses_exact_manifest_pins_and_never_reports_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": TOKEN}), encoding="utf-8")
    calls = _fake_snapshots(monkeypatch)

    def fake_url(source):
        assert source.url == "https://models.example.invalid/detector.bin"
        source.target.parent.mkdir(parents=True, exist_ok=True)
        source.target.write_bytes(URL_BYTES)

    monkeypatch.setattr(download, "_download_url", fake_url)

    assert download.main(["--manifest", str(manifest), "--secrets-file", str(secrets)]) == 0
    report = json.loads(capsys.readouterr().out)
    serialized = json.dumps(report, sort_keys=True)
    assert report["status"] == "PASS"
    assert report["token_source"] == "configured"
    assert TOKEN not in serialized
    assert [(call["repo_id"], call["revision"]) for call in calls] == [
        ("Example/Caption", SERVICE_REVISION),
        ("Example/Visual", LOCAL_REVISION),
    ]
    assert calls[0]["cache_dir"] == str((tmp_path / "cache/hub").resolve())
    assert calls[0]["allow_patterns"] == ["config.json", "model.bin"]
    assert calls[1]["local_dir"] == str((tmp_path / "models/visual").resolve())
    assert calls[1]["cache_dir"] == str((tmp_path / "cache/hub").resolve())
    assert calls[1]["allow_patterns"] == ["config.json", "model.safetensors"]
    assert all(call["token"] == TOKEN for call in calls)


def test_floating_revision_is_rejected_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["pipeline_models"]["visual-backbone"]["source"]["revision"] = "main"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(download, "_snapshot_download", lambda **_kwargs: pytest.fail("network"))
    monkeypatch.setattr(download, "_download_url", lambda _source: pytest.fail("network"))

    assert download.main(["--manifest", str(manifest)]) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == "ERROR"
    assert "immutable 40-hex revision" in report["error"]


def test_dependency_error_cannot_leak_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": TOKEN}), encoding="utf-8")
    monkeypatch.setattr(
        download,
        "_snapshot_download",
        lambda **_kwargs: (_ for _ in ()).throw(
            download.ModelDownloadError(f"remote rejected {TOKEN}")
        ),
    )

    assert download.main(
        ["--manifest", str(manifest), "--secrets-file", str(secrets)]
    ) == 2
    serialized = capsys.readouterr().err
    assert TOKEN not in serialized
    assert "remote rejected <redacted>" in serialized


def test_huggingface_dependency_exceptions_are_normalised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHub:
        @staticmethod
        def snapshot_download(**_kwargs):
            raise RuntimeError(f"backend echoed {TOKEN}")

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    with pytest.raises(
        download.ModelDownloadError,
        match="Hugging Face snapshot download failed",
    ):
        download._snapshot_download(repo_id="Example/Model")


def test_local_files_only_refuses_unverified_url_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.bin"
    source = download.LocalSource(
        model_name="detector",
        target=target,
        expected_sha256=_sha(URL_BYTES),
        kind="url",
        url="https://models.example.invalid/model.bin",
    )
    monkeypatch.setattr(download, "_download_url", lambda _source: pytest.fail("network"))
    with pytest.raises(download.ModelDownloadError, match="URL access is disabled"):
        download.download_local_sources(
            [source], token=None, local_files_only=True
        )
    assert not target.exists()


class _FakeResponse:
    def __init__(self, payload: bytes, url: str) -> None:
        self.payload = payload
        self.url = url
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_url_checksum_mismatch_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.bin"
    target.write_bytes(b"preexisting model")
    source = download.LocalSource(
        model_name="detector",
        target=target,
        expected_sha256=_sha(URL_BYTES),
        kind="url",
        url="https://models.example.invalid/model.bin",
    )
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(
            b"tampered download", "https://cdn.example.invalid/model.bin"
        ),
    )

    with pytest.raises(download.ModelDownloadError, match="downloaded SHA256 mismatch"):
        download._download_url(source)
    assert target.read_bytes() == b"preexisting model"
    assert not list(tmp_path.glob(".model.bin.*.download"))


def test_production_local_sources_have_reproducible_provenance() -> None:
    manifest_path = ROOT / "configs/models/farm_models.v1.json"
    raw = download.load_raw_manifest(manifest_path)
    manifest = download.load_model_manifest(manifest_path)
    sources = {item.model_name: item for item in download.parse_local_sources(raw, manifest)}

    assert sources["dinov3-vits16plus"].revision == (
        "c93d816fc9e567563bc068f01475bec89cc634a6"
    )
    assert sources["siglip2-large-patch16-256"].revision == (
        "787800c8990e6f058423089178e718139608408c"
    )
    assert sources["yoloe-v8l-seg"].kind == "url"
    assert sources["yoloe-v8l-seg"].expected_sha256
    assert sources["mobileclip"].kind == "url"
    assert sources["mobileclip"].expected_sha256


def test_shell_bootstrap_is_thin_manifest_driven_wrapper() -> None:
    launcher = ROOT / "bootstrap_models.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    source = launcher.read_text(encoding="utf-8")
    assert os.access(launcher, os.X_OK)
    assert "scripts/download_farm_models.py" in source
    assert '"$@"' in source
    assert "resolve/main" not in source
    assert "huggingface-cli" not in source
    assert "/home/" not in source


def test_invalid_utf8_manifest_returns_structured_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_bytes(b"\xff\xfe")

    assert download.main(["--manifest", str(manifest)]) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == "ERROR"
    assert report["schema"] == "farm.model-download.v2"
