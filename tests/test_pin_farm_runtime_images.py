from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pin_farm_runtime_images.py"
SPEC = importlib.util.spec_from_file_location("pin_farm_runtime_images", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pin)


OLD_MAIN = "sha256:" + "1" * 64
NEW_MAIN = "sha256:" + "2" * 64
PREP = "sha256:" + "3" * 64


def _manifest(path: Path) -> dict:
    payload = {
        "schema_version": "farm.models.v1",
        "cache_root": "/cache",
        "runtimes": {
            "main": {"image": "scene_graph:latest", "image_id": OLD_MAIN, "user_uid": 1000},
            "prep": {
                "image": "rest3d:factory-baseline",
                "image_id": PREP,
                "python": "/opt/conda/envs/rest3d/bin/python",
            },
        },
        "services": {"caption": {"runtime": "main", "port": 8000}},
        "pipeline_models": {"detector": {"kind": "local_file", "path": "model.pt"}},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _inspect(monkeypatch: pytest.MonkeyPatch, ids: dict[str, str]) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        image = command[-1]
        return subprocess.CompletedProcess(command, 0, stdout=ids[image] + "\n", stderr="")

    monkeypatch.setattr(pin.subprocess, "run", fake_run)
    monkeypatch.setattr(pin, "_test_calls", calls, raising=False)


def test_dry_run_detects_drift_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "models.json"
    before = _manifest(path)
    original_bytes = path.read_bytes()
    _inspect(monkeypatch, {"scene_graph:latest": NEW_MAIN, "rest3d:factory-baseline": PREP})

    assert pin.main(["--manifest", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "drift"
    assert report["mode"] == "audit"
    assert report["changed_runtimes"] == ["main"]
    assert path.read_bytes() == original_bytes
    assert json.loads(path.read_text()) == before
    assert [command[-1] for command in pin._test_calls] == [
        "scene_graph:latest",
        "rest3d:factory-baseline",
    ]


def test_write_updates_only_runtime_image_ids_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "models.json"
    before = _manifest(path)
    _inspect(monkeypatch, {"scene_graph:latest": NEW_MAIN, "rest3d:factory-baseline": PREP})

    assert pin.main(["--manifest", str(path), "--write"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "updated"
    after = json.loads(path.read_text())
    expected = copy.deepcopy(before)
    expected["runtimes"]["main"]["image_id"] = NEW_MAIN
    assert after == expected
    assert after["runtimes"]["prep"]["python"] == "/opt/conda/envs/rest3d/bin/python"
    assert not list(tmp_path.glob(".models.json.*.tmp"))


def test_write_inspects_every_runtime_before_any_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "models.json"
    _manifest(path)
    original = path.read_bytes()

    def fake_run(command, **kwargs):
        if command[-1] == "scene_graph:latest":
            return subprocess.CompletedProcess(command, 0, stdout=NEW_MAIN + "\n", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="No such image")

    monkeypatch.setattr(pin.subprocess, "run", fake_run)
    assert pin.main(["--manifest", str(path), "--write"]) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == "error"
    assert "No such image" in report["error"]
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "observed",
    ["", "2" * 64, "sha256:xyz", "sha256:" + "A" * 64, NEW_MAIN + "\n" + PREP],
)
def test_invalid_observed_id_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    observed: str,
) -> None:
    path = tmp_path / "models.json"
    _manifest(path)
    original = path.read_bytes()

    monkeypatch.setattr(
        pin.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=observed + ("\n" if observed else ""), stderr=""
        ),
    )
    assert pin.main(["--manifest", str(path), "--write"]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    assert path.read_bytes() == original


def test_manifest_symlink_and_invalid_configured_pin_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "models.json"
    payload = _manifest(target)
    payload["runtimes"]["main"]["image_id"] = "latest"
    target.write_text(json.dumps(payload), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    monkeypatch.setattr(pin.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not inspect"))

    assert pin.main(["--manifest", str(link), "--write"]) == 2
    assert "symlink" in json.loads(capsys.readouterr().err)["error"]
    assert pin.main(["--manifest", str(target), "--write"]) == 2
    assert "invalid configured image_id" in json.loads(capsys.readouterr().err)["error"]


def test_invalid_utf8_manifest_returns_structured_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_bytes(b"\xff\xfe")

    assert pin.main(["--manifest", str(manifest), "--write"]) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == "error"
    assert report["schema"] == "farm.runtime-image-pins.v1"
