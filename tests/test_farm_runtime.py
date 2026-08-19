from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from farm_runtime.config import PlanError, load_plan
from farm_runtime.orchestrator import (
    PipelineRunError,
    RunOrchestrator,
    _create_source_snapshot,
    selected_device_telemetry,
)
from farm_runtime.source_snapshot import validated_source_snapshot_project_root
from farm_runtime.standard import (
    STANDARD_SHARED_CODE_INPUTS,
    STANDARD_STAGE_CODE_INPUTS,
    STANDARD_STAGE_IDS,
)
from farm_runtime.viewer import ViewerError, _bundle_context, _dockerize_viewer_command


ROOT = Path(__file__).resolve().parents[1]


def test_viewer_bundle_context_resolves_python_executable(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "scene" / "runs" / "run-1"
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True)
    artifact = run_dir / "final" / "scene_state.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"state")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "project_root": str(tmp_path),
                "scene_id": "scene",
                "run_id": "run-1",
            }
        ),
        encoding="utf-8",
    )
    context = _bundle_context(
        run_dir,
        {"artifacts": {"scene_state": "../final/scene_state.pt"}},
        "0.0.0.0",
        8081,
    )
    assert context["python_executable"] == sys.executable
    assert context["artifact:scene_state"] == str(artifact.resolve())
    assert context["project_root"] == str(tmp_path)
    assert context["original_project_root"] == str(tmp_path)


def _initialized_snapshot_run(tmp_path: Path) -> tuple[RunOrchestrator, Path]:
    config_path = _advanced_config(
        tmp_path,
        [{"id": "noop", "command": [sys.executable, "-c", "pass"]}],
        {},
    )
    runner = RunOrchestrator(load_plan(config_path))
    run_dir = runner._new_run_dir("source-snapshot")
    runner._initialize_run(run_dir)
    return runner, run_dir


def test_source_snapshot_is_exact_hashed_allowlisted_and_secret_free(
    tmp_path: Path,
) -> None:
    for name in (
        "configs/models",
        "docker",
        "requirements",
        "ros",
        "scripts/__pycache__",
        "src/cache",
        "tests/output",
        "models",
        "third_party",
        "cache",
    ):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    source = tmp_path / "scripts/tool.py"
    source.write_text("print('dirty working tree v1')\n", encoding="utf-8")
    os.chmod(source, 0o755)
    (tmp_path / "src/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "configs/models/runtime.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "configs/.env").write_text("API_TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "docker/credentials.json").write_text('{"token":"secret"}\n', encoding="utf-8")
    (tmp_path / "scripts/__pycache__/tool.pyc").write_bytes(b"pyc")
    (tmp_path / "src/cache/generated.py").write_text("generated\n", encoding="utf-8")
    (tmp_path / "tests/output/result.txt").write_text("generated\n", encoding="utf-8")
    (tmp_path / "models/weights.bin").write_bytes(b"weights")
    (tmp_path / "third_party/vendor.py").write_text("vendor\n", encoding="utf-8")
    (tmp_path / "cache/cache.bin").write_bytes(b"cache")
    (tmp_path / "private-notes.txt").write_text("root secret\n", encoding="utf-8")
    for name, content in {
        "pyproject.toml": "[project]\nname='snapshot-test'\n",
        "run.sh": "#!/bin/sh\nexit 0\n",
        "bootstrap_models.sh": "#!/bin/sh\nexit 0\n",
        "README.md": "snapshot test\n",
        "LICENSE": "test license\n",
        ".gitmodules": "",
    }.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    _, run_dir = _initialized_snapshot_run(tmp_path)
    source.write_text("print('live tree changed after snapshot')\n", encoding="utf-8")

    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"
    snapshot_manifest = json.loads(
        (run_dir / "config/source_snapshot/manifest.json").read_text()
    )
    run_manifest = json.loads((run_dir / "manifest.json").read_text())
    copied = snapshot_root / "scripts/tool.py"
    assert copied.read_text(encoding="utf-8") == "print('dirty working tree v1')\n"
    assert stat.S_IMODE(copied.stat().st_mode) == 0o755
    assert copied.stat().st_mode & stat.S_IXUSR
    records = {record["path"]: record for record in snapshot_manifest["files"]}
    assert records["scripts/tool.py"]["sha256"] == hashlib.sha256(
        copied.read_bytes()
    ).hexdigest()
    assert records["scripts/tool.py"]["executable"] is True
    canonical_tree = json.dumps(
        {
            "schema": "farm.source-snapshot-tree.v1",
            "files": sorted(snapshot_manifest["files"], key=lambda row: row["path"]),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    tree_sha256 = hashlib.sha256(canonical_tree.encode("utf-8")).hexdigest()
    assert snapshot_manifest["tree_sha256"] == tree_sha256
    assert run_manifest["source_snapshot"] == {
        "schema": "farm.source-snapshot.v1",
        "relative_path": "config/source_snapshot/FARM-Project",
        "manifest_relative_path": "config/source_snapshot/manifest.json",
        "tree_sha256": tree_sha256,
        "file_count": len(snapshot_manifest["files"]),
    }
    assert (snapshot_root / "configs/models/runtime.json").is_file()
    for relative in (
        "configs/.env",
        "docker/credentials.json",
        "scripts/__pycache__/tool.pyc",
        "src/cache/generated.py",
        "tests/output/result.txt",
        "models/weights.bin",
        "third_party/vendor.py",
        "cache/cache.bin",
        "private-notes.txt",
        "pipeline.yaml",
    ):
        assert not (snapshot_root / relative).exists()
    assert snapshot_manifest["policy"]["excluded_secret_file_count"] == 2
    assert "git" in snapshot_manifest and "submodules" in snapshot_manifest


def test_allowlisted_source_config_is_only_persisted_redacted(
    tmp_path: Path,
) -> None:
    original = _advanced_config(
        tmp_path,
        [{"id": "noop", "command": [sys.executable, "-c", "pass"]}],
        {},
    )
    source = tmp_path / "configs/pipeline.yaml"
    source.parent.mkdir()
    source.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    original.unlink()
    runner = RunOrchestrator(load_plan(source))
    run_dir = runner._new_run_dir("allowlisted-source-config")
    runner._initialize_run(run_dir)

    assert not (
        run_dir / "config/source_snapshot/FARM-Project/configs/pipeline.yaml"
    ).exists()
    assert (run_dir / "config/source.redacted.yaml").is_file()
    snapshot_manifest = json.loads(
        (run_dir / "config/source_snapshot/manifest.json").read_text()
    )
    assert snapshot_manifest["policy"]["excluded_source_config_count"] == 1


def test_viewer_prefers_valid_snapshot_and_fails_closed_if_tampered(
    tmp_path: Path,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/view_scene_state.py").write_text(
        "# immutable viewer\n", encoding="utf-8"
    )
    _, run_dir = _initialized_snapshot_run(tmp_path)
    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"
    context = _bundle_context(run_dir, {"artifacts": {}}, "127.0.0.1", 8080)
    assert context["project_root"] == str(snapshot_root.resolve())
    assert context["original_project_root"] == str(tmp_path)

    (snapshot_root / "scripts/view_scene_state.py").write_text(
        "# tampered\n", encoding="utf-8"
    )
    with pytest.raises(ViewerError, match="integrity mismatch"):
        _bundle_context(run_dir, {"artifacts": {}}, "127.0.0.1", 8080)


def test_source_snapshot_copy_failure_prevents_every_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/stage.py").write_text("print('stage')\n", encoding="utf-8")
    sentinel = tmp_path / "stage-ran.txt"
    runner = RunOrchestrator(load_plan(_advanced_config(
        tmp_path,
        [{
            "id": "must_not_run",
            "command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).write_text('ran')",
            ],
        }],
        {},
    )))

    def fail_copy(*args, **kwargs):
        raise OSError("injected copy failure")

    monkeypatch.setattr("farm_runtime.orchestrator.shutil.copyfile", fail_copy)
    with pytest.raises(PipelineRunError, match="Source snapshot failed"):
        runner.run(run_id="snapshot-copy-failure")
    run_dir = runner.runs_root / "snapshot-copy-failure"
    assert not sentinel.exists()
    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "stages/must_not_run/state.json").exists()
    assert not (run_dir / "config/source_snapshot").exists()
    assert list((run_dir / "config").glob(".source_snapshot.*")) == []


def test_standard_viewer_docker_command_uses_pinned_non_root_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    run_dir = tmp_path / "output" / "scene" / "runs" / "run-1"
    (run_dir / "input").mkdir(parents=True)
    model_manifest = project_root / "models.json"
    image_id = "sha256:" + "a" * 64
    model_manifest.write_text(
        json.dumps(
            {
                "runtimes": {
                    "main": {
                        "image_id": image_id,
                        "user_uid": 1000,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "input" / "resolved_context.json").write_text(
        json.dumps({"model_manifest": str(model_manifest)}),
        encoding="utf-8",
    )
    monkeypatch.setattr("farm_runtime.viewer.shutil.which", lambda name: "/usr/bin/docker")
    command = _dockerize_viewer_command(
        run_dir,
        {"project_root": str(project_root)},
        [
            sys.executable,
            str(project_root / "scripts" / "view_scene_state.py"),
            "--pt",
            str(run_dir / "final" / "scene_state.pt"),
            "--port",
            "8081",
        ],
    )
    assert command[:3] == ["docker", "run", "--rm"]
    assert image_id in command
    assert command[command.index("--user") + 1].startswith("1000:")
    assert "--network" in command and command[command.index("--network") + 1] == "host"
    assert "PYTHONPATH=/home/scene_graph/scene_graph/src" in command
    assert f"{run_dir}:{run_dir}:ro" in command
    assert command[-4:] == [
        "--pt",
        str(run_dir / "final" / "scene_state.pt"),
        "--port",
        "8081",
    ]


def test_docker_viewer_model_manifest_falls_back_to_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs/models").mkdir(parents=True)
    (tmp_path / "scripts/view_scene_state.py").write_text(
        "# snapshotted viewer\n", encoding="utf-8"
    )
    live_model_manifest = tmp_path / "configs/models/farm_models.v1.json"
    image_id = "sha256:" + "b" * 64
    model_payload = {
        "runtimes": {"main": {"image_id": image_id, "user_uid": 1000}}
    }
    encoded = json.dumps(model_payload, sort_keys=True).encode("utf-8")
    live_model_manifest.write_bytes(encoded)
    _, run_dir = _initialized_snapshot_run(tmp_path)
    (run_dir / "input/resolved_context.json").write_text(
        json.dumps({
            "model_manifest": str(live_model_manifest),
            "model_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        }),
        encoding="utf-8",
    )
    live_model_manifest.unlink()
    context = _bundle_context(run_dir, {"artifacts": {}}, "127.0.0.1", 8082)
    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"
    monkeypatch.setattr("farm_runtime.viewer.shutil.which", lambda name: "/usr/bin/docker")
    command = _dockerize_viewer_command(
        run_dir,
        context,
        [
            sys.executable,
            str(snapshot_root / "scripts/view_scene_state.py"),
            "--port",
            "8082",
        ],
    )

    assert image_id in command
    assert f"{snapshot_root.resolve()}:/home/scene_graph/scene_graph:ro" in command


def test_viewer_runtime_is_persisted_in_bundle(tmp_path: Path) -> None:
    config_path = _advanced_config(
        tmp_path,
        [{"id": "noop", "command": [sys.executable, "-c", "pass"]}],
        {},
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["viewer"] = {
        "runtime": "docker",
        "command": [
            sys.executable,
            "viewer.py",
            "--host",
            "${host}",
            "--port",
            "${port}",
        ],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    runner = RunOrchestrator(load_plan(config_path))
    run_dir = runner.runs_root / "viewer-runtime"
    (run_dir / "viewer").mkdir(parents=True)
    runner._write_viewer_bundle(run_dir)
    bundle = json.loads((run_dir / "viewer/bundle.json").read_text(encoding="utf-8"))
    assert bundle["process"]["runtime"] == "docker"


def _helper(path: Path) -> None:
    path.write_text(
        "from pathlib import Path\nimport sys\n"
        "source, destination = map(Path, sys.argv[1:3])\n"
        "destination.parent.mkdir(parents=True, exist_ok=True)\n"
        "destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')\n",
        encoding="utf-8",
    )


def _advanced_config(tmp_path: Path, stages: list[dict], artifacts: dict[str, str]) -> Path:
    config = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "scene": {"id": "test_scene"},
        "output": {"root": "${project_root}/output"},
        "pipeline": {"telemetry_interval_seconds": 0.1, "stages": stages},
        "artifacts": artifacts,
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _standard_config(tmp_path: Path, scene_id: str = "snapshot_scene") -> Path:
    (tmp_path / "data/sparse/0").mkdir(parents=True)
    (tmp_path / "data/images").mkdir(parents=True)
    (tmp_path / "data/scene.ply").write_text("ply", encoding="utf-8")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts/farm_standard_stage.py").write_text(
        "# frozen standard wrapper\n", encoding="utf-8"
    )
    path = tmp_path / "scene.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "farm.scene.v1",
                "scene_id": scene_id,
                "inputs": {
                    "colmap_model": "data/sparse/0",
                    "image_root": "data/images",
                    "gaussian_ply": "data/scene.ply",
                },
                "output_root": "results",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _initialized_standard_run(
    tmp_path: Path, run_id: str = "standard-snapshot"
) -> tuple[RunOrchestrator, Path]:
    runner = RunOrchestrator(load_plan(_standard_config(tmp_path), project_root=tmp_path))
    run_dir = runner._new_run_dir(run_id)
    runner._initialize_run(run_dir)
    return runner, run_dir


def test_standard_plan_routes_all_first_party_execution_to_snapshot(
    tmp_path: Path,
) -> None:
    runner, run_dir = _initialized_standard_run(tmp_path)
    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"
    for stage_id in STANDARD_STAGE_IDS:
        resolved = runner._resolve_stage(runner.plan.stages_by_id[stage_id], run_dir)
        assert resolved.command[1] == str(snapshot_root / "scripts/farm_standard_stage.py")
        assert resolved.cwd == snapshot_root
        assert resolved.stage.env["PYTHONDONTWRITEBYTECODE"] == "1"
        expected_code = {
            snapshot_root / relative
            for relative in (
                *STANDARD_SHARED_CODE_INPUTS,
                *STANDARD_STAGE_CODE_INPUTS[stage_id],
            )
        }
        assert expected_code <= set(resolved.fingerprint_inputs)
        assert runner.plan.source_path in resolved.fingerprint_inputs
        assert all(path == run_dir or run_dir in path.parents for path in resolved.outputs)

    preflight = runner._resolve_stage(runner.plan.stages_by_id["preflight"], run_dir)
    assert (tmp_path / "data/scene.ply").resolve() in preflight.fingerprint_inputs
    assert (tmp_path / "data/sparse/0").resolve() in preflight.fingerprint_inputs
    assert (tmp_path / "data/images").resolve() in preflight.fingerprint_inputs
    public_plan = json.loads((run_dir / "config/resolved-plan.json").read_text())
    assert all(row["cwd"] == str(snapshot_root) for row in public_plan["stages"])
    assert all(
        row["command"][1] == str(snapshot_root / "scripts/farm_standard_stage.py")
        for row in public_plan["stages"]
    )
    context = runner._context(run_dir)
    finalizer = runner.plan.finalizers[0]
    assert finalizer.command[1] == "${execution_project_root}/scripts/farm_standard_stage.py"
    assert finalizer.env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert context["execution_project_root"] == str(snapshot_root)


def test_standard_live_source_changes_do_not_change_frozen_stage(
    tmp_path: Path,
) -> None:
    runner, run_dir = _initialized_standard_run(tmp_path)
    stage = runner.plan.stages_by_id["preflight"]
    before_resolved = runner._resolve_stage(stage, run_dir)
    before, _ = runner._stage_fingerprint(before_resolved, {})
    live_wrapper = tmp_path / "scripts/farm_standard_stage.py"
    live_wrapper.write_text("# changed live wrapper\n", encoding="utf-8")
    after_resolved = runner._resolve_stage(stage, run_dir)
    after, _ = runner._stage_fingerprint(after_resolved, {})
    assert after_resolved.command == before_resolved.command
    assert after == before
    assert (
        run_dir / "config/source_snapshot/FARM-Project/scripts/farm_standard_stage.py"
    ).read_text(encoding="utf-8") == "# frozen standard wrapper\n"


@pytest.mark.parametrize("tamper", ["content", "mode", "extra", "symlink"])
def test_standard_snapshot_tamper_blocks_resolution(
    tmp_path: Path, tamper: str
) -> None:
    runner, run_dir = _initialized_standard_run(tmp_path)
    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"
    wrapper = snapshot_root / "scripts/farm_standard_stage.py"
    if tamper == "content":
        wrapper.write_text("# tampered\n", encoding="utf-8")
    elif tamper == "mode":
        os.chmod(wrapper, stat.S_IMODE(wrapper.stat().st_mode) | stat.S_IXUSR)
    elif tamper == "extra":
        (snapshot_root / "scripts/unexpected.py").write_text("pass\n", encoding="utf-8")
    else:
        (snapshot_root / "scripts/unexpected-link.py").symlink_to(wrapper)
    with pytest.raises(PipelineRunError, match="snapshot integrity"):
        runner._resolve_stage(runner.plan.stages_by_id["preflight"], run_dir)


def test_standard_snapshot_tamper_blocks_resume_before_manifest_mutation(
    tmp_path: Path,
) -> None:
    runner, run_dir = _initialized_standard_run(tmp_path, "tampered-resume")
    failed = run_dir / "_FAILED.json"
    failed.write_text('{"status":"failed"}\n', encoding="utf-8")
    manifest_before = (run_dir / "manifest.json").read_bytes()
    (run_dir / "config/source_snapshot/FARM-Project/scripts/farm_standard_stage.py").write_text(
        "# tampered\n", encoding="utf-8"
    )
    with pytest.raises(PipelineRunError, match="snapshot integrity"):
        runner.run(resume=True, run_id="tampered-resume")
    assert (run_dir / "manifest.json").read_bytes() == manifest_before
    assert failed.is_file()
    assert not (run_dir / "_SUCCESS.json").exists()


def test_standard_snapshot_tamper_during_child_cannot_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, run_dir = _initialized_standard_run(tmp_path, "mid-child-tamper")
    resolved = runner._resolve_stage(runner.plan.stages_by_id["preflight"], run_dir)
    fingerprint, inputs = runner._stage_fingerprint(resolved, {})
    wrapper = run_dir / "config/source_snapshot/FARM-Project/scripts/farm_standard_stage.py"

    def fake_process(*args, **kwargs):
        for output in resolved.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"status":"PASS"}\n', encoding="utf-8")
        wrapper.write_text("# tampered while child ran\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, timed_out=False, interrupted=False)

    monkeypatch.setattr("farm_runtime.orchestrator.run_monitored_process", fake_process)
    with pytest.raises(PipelineRunError, match="snapshot integrity"):
        runner._run_stage(resolved, run_dir, fingerprint, inputs)
    assert (run_dir / "stages/preflight/_FAILED.json").is_file()
    assert not (run_dir / "stages/preflight/_SUCCESS.json").exists()


def test_standard_finalizer_and_report_execute_from_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, run_dir = _initialized_standard_run(tmp_path, "snapshot-report")
    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"
    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_process(command, *, cwd, **kwargs):
        calls.append((tuple(map(str, command)), Path(cwd)))
        if any(Path(token).name == "build_farm_run_report.py" for token in command):
            (run_dir / "REPORT.md").write_text("report\n", encoding="utf-8")
            (run_dir / "visuals/scene_summary.mp4").write_bytes(b"video")
        return SimpleNamespace(
            returncode=0, timed_out=False, interrupted=False, duration_seconds=0.01
        )

    monkeypatch.setattr("farm_runtime.orchestrator.run_monitored_process", fake_process)
    runner._run_finalizers(run_dir)
    report = runner._write_standard_report(run_dir)
    assert report is not None
    assert calls[0][0][1] == str(snapshot_root / "scripts/farm_standard_stage.py")
    assert calls[0][1] == snapshot_root
    assert calls[1][0][2] == str(snapshot_root / "scripts/build_farm_run_report.py")
    assert calls[1][1] == snapshot_root


def test_standard_config_mutation_fails_closed_before_resolution(tmp_path: Path) -> None:
    runner, run_dir = _initialized_standard_run(tmp_path, "config-tamper")
    runner.plan.source_path.write_text(
        runner.plan.source_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineRunError, match="source config changed"):
        runner._resolve_stage(runner.plan.stages_by_id["preflight"], run_dir)


def test_standard_wrapper_import_does_not_mutate_signed_snapshot(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "real-wrapper-run"
    run_dir.mkdir()
    reference = _create_source_snapshot(
        ROOT,
        run_dir,
        source_config_path=ROOT / "configs/scenes/factory_3dgs_colmap.yaml",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"source_snapshot": reference}), encoding="utf-8"
    )
    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    subprocess.run(
        [sys.executable, str(snapshot_root / "scripts/farm_standard_stage.py"), "--help"],
        cwd=snapshot_root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert validated_source_snapshot_project_root(run_dir, required=True) == snapshot_root
    assert not any(path.name == "__pycache__" for path in snapshot_root.rglob("*"))
    assert not any(path.suffix == ".pyc" for path in snapshot_root.rglob("*"))


def test_advanced_run_is_atomic_self_contained_and_resumable(tmp_path: Path) -> None:
    helper = tmp_path / "copy.py"
    _helper(helper)
    seed = tmp_path / "seed.txt"
    seed.write_text("payload", encoding="utf-8")
    gate = tmp_path / "gate.txt"
    stages = [
        {
            "id": "first",
            "command": [sys.executable, str(helper), str(seed), "${run_dir}/artifacts/first.txt"],
            "inputs": [str(seed), str(helper)],
            "outputs": ["${run_dir}/artifacts/first.txt"],
        },
        {
            "id": "second",
            "needs": ["first"],
            "command": [sys.executable, str(helper), str(gate), "${run_dir}/artifacts/final.txt"],
            "inputs": [str(gate), str(helper)],
            "outputs": ["${run_dir}/artifacts/final.txt"],
        },
    ]
    plan = load_plan(_advanced_config(tmp_path, stages, {"final": "${run_dir}/artifacts/final.txt"}))
    runner = RunOrchestrator(plan)
    with pytest.raises(PipelineRunError, match="missing declared inputs"):
        runner.run(run_id="resume-case")
    run_dir = runner.runs_root / "resume-case"
    assert json.loads((run_dir / "stages/first/state.json").read_text())["attempt"] == 1
    assert (run_dir / "_FAILED.json").is_file()
    assert not (runner.scene_root / "latest").exists()

    gate.write_text("final", encoding="utf-8")
    resumed = runner.run(resume=True, run_id="resume-case")
    assert resumed == run_dir.resolve()
    assert (run_dir / "_SUCCESS.json").is_file()
    assert not (run_dir / "_FAILED.json").exists()
    assert (runner.scene_root / "latest").resolve() == run_dir.resolve()
    assert json.loads((run_dir / "stages/first/state.json").read_text())["attempt"] == 1
    assert json.loads((run_dir / "stages/second/state.json").read_text())["attempt"] == 2
    bundle = json.loads((run_dir / "viewer/bundle.json").read_text())
    assert bundle["artifacts"]["final"] == "../artifacts/final.txt"
    for directory in ("input", "selection", "rgbd", "mapping", "final", "qa", "visuals", "viewer", "logs", "timing"):
        assert (run_dir / directory).is_dir()


def test_pass_json_is_a_fail_closed_quality_gate(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text('{"status":"FAIL"}', encoding="utf-8")
    stages = [{"id": "qa", "command": [sys.executable, "-c", "pass"], "outputs": [str(result)], "pass_json": [str(result)]}]
    runner = RunOrchestrator(load_plan(_advanced_config(tmp_path, stages, {})))
    with pytest.raises(PipelineRunError, match="did not report PASS"):
        runner.run(run_id="qa-fail")
    assert (runner.runs_root / "qa-fail/_FAILED.json").is_file()
    assert not (runner.runs_root / "qa-fail/_SUCCESS.json").exists()


def test_unknown_force_stage_does_not_create_a_run(tmp_path: Path) -> None:
    config = _advanced_config(tmp_path, [{"id": "only", "command": [sys.executable, "-c", "pass"]}], {})
    runner = RunOrchestrator(load_plan(config))
    with pytest.raises(PipelineRunError, match="Unknown --force-stage"):
        runner.run(force_stages=["missing"])
    assert not runner.runs_root.exists()


def test_advanced_pipeline_does_not_require_standard_report(tmp_path: Path) -> None:
    plan = load_plan(_advanced_config(
        tmp_path,
        [{"id": "only", "command": [sys.executable, "-c", "pass"]}],
        {},
    ))
    runner = RunOrchestrator(plan)
    run_dir = runner.run(run_id="advanced-no-report")
    assert (run_dir / "_SUCCESS.json").is_file()
    assert not (run_dir / "REPORT.md").exists()


def test_fingerprint_inputs_default_to_declared_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    plan = load_plan(
        _advanced_config(
            tmp_path,
            [{"id": "stage", "command": ["true"], "inputs": [str(source)]}],
            {},
        )
    )
    assert plan.stages_by_id["stage"].fingerprint_inputs == (str(source),)


def test_scene_contract_compiles_full_standard_dag(tmp_path: Path) -> None:
    colmap = tmp_path / "data/sparse/0"
    images = tmp_path / "data/images"
    colmap.mkdir(parents=True)
    images.mkdir(parents=True)
    ply = tmp_path / "data/scene.ply"
    ply.write_text("ply-v1", encoding="utf-8")
    config = {
        "schema_version": "farm.scene.v1",
        "scene_id": "portable_scene",
        "inputs": {"colmap_model": "data/sparse/0", "image_root": "data/images", "gaussian_ply": "data/scene.ply"},
        "output_root": "results",
    }
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    plan = load_plan(path, project_root=tmp_path)
    assert plan.stage_order == STANDARD_STAGE_IDS
    assert plan.scene_id == "portable_scene"
    assert plan.stages_by_id["qa_bundle"].pass_json == ("${run_dir}/qa/result.json",)
    assert plan.artifacts["scene_state"] == "${run_dir}/final/scene_state.pt"
    assert plan.artifacts["resolved_context"] == "${run_dir}/input/resolved_context.json"
    assert plan.artifacts["mapping"] == "${run_dir}/mapping"
    assert plan.artifacts["qa_summary"] == "${run_dir}/qa/summary.json"
    assert plan.viewer is not None
    assert plan.viewer.runtime == "docker"
    assert "${host}" in plan.viewer.command
    assert "${port}" in plan.viewer.command
    assert "--resolved-context" in plan.viewer.command
    assert "${artifact:resolved_context}" in plan.viewer.command
    assert "--mapping-dir" in plan.viewer.command
    assert "--no-query" in plan.viewer.command
    assert "${run_dir}/viewer/launch.sh" in plan.stages_by_id["qa_bundle"].outputs
    assert "${run_dir}/qa/acceptance/result.json" in plan.stages_by_id["finalize"].outputs
    runner = RunOrchestrator(plan)
    run_dir = runner._new_run_dir("portable-scene-fingerprint")
    runner._initialize_run(run_dir)
    stage = plan.stages_by_id["preflight"]
    resolved = runner._resolve_stage(stage, run_dir)
    first, _ = runner._stage_fingerprint(resolved, {})
    ply.write_text("ply-v2-changed", encoding="utf-8")
    resolved = runner._resolve_stage(stage, run_dir)
    second, _ = runner._stage_fingerprint(resolved, {})
    assert first != second


def test_standard_stage_code_is_frozen_after_run_initialization(tmp_path: Path) -> None:
    colmap = tmp_path / "data/sparse/0"
    images = tmp_path / "data/images"
    colmap.mkdir(parents=True)
    images.mkdir(parents=True)
    (tmp_path / "data/scene.ply").write_text("ply", encoding="utf-8")
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "farm.scene.v1",
                "scene_id": "fingerprint_scene",
                "inputs": {
                    "colmap_model": "data/sparse/0",
                    "image_root": "data/images",
                    "gaussian_ply": "data/scene.ply",
                },
                "output_root": "results",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    plan = load_plan(scene_path, project_root=tmp_path)
    runner = RunOrchestrator(plan)
    target = tmp_path / "scripts/refine_farm_object_geometry.py"
    target.parent.mkdir(parents=True)
    target.write_text("# geometry implementation v1\n", encoding="utf-8")
    semantic_contract = tmp_path / "src/scene_graph/captioning/label_contract.py"
    semantic_contract.parent.mkdir(parents=True)
    semantic_contract.write_text("# label contract v1\n", encoding="utf-8")
    run_dir = runner._new_run_dir("frozen-code-fingerprint")
    runner._initialize_run(run_dir)
    snapshot_root = run_dir / "config/source_snapshot/FARM-Project"

    geometry = runner._resolve_stage(plan.stages_by_id["geometry"], run_dir)
    snapshot_target = snapshot_root / "scripts/refine_farm_object_geometry.py"
    assert snapshot_target in geometry.fingerprint_inputs
    before, _ = runner._stage_fingerprint(geometry, {})
    target.write_text("# changed geometry implementation\n", encoding="utf-8")
    geometry = runner._resolve_stage(plan.stages_by_id["geometry"], run_dir)
    after, _ = runner._stage_fingerprint(geometry, {})
    assert before == after

    downstream = runner._resolve_stage(plan.stages_by_id["visual_consistency"], run_dir)
    downstream_before, _ = runner._stage_fingerprint(
        downstream, {"geometry": before}
    )
    downstream_after, _ = runner._stage_fingerprint(
        downstream, {"geometry": after}
    )
    assert downstream_before == downstream_after

    semantics = runner._resolve_stage(plan.stages_by_id["semantics"], run_dir)
    snapshot_semantic_contract = (
        snapshot_root / "src/scene_graph/captioning/label_contract.py"
    )
    assert snapshot_semantic_contract in semantics.fingerprint_inputs
    semantic_before, _ = runner._stage_fingerprint(semantics, {})
    semantic_contract.write_text("# changed label contract\n", encoding="utf-8")
    semantics = runner._resolve_stage(plan.stages_by_id["semantics"], run_dir)
    semantic_after, _ = runner._stage_fingerprint(semantics, {})
    assert semantic_before == semantic_after

    part_whole = runner._resolve_stage(plan.stages_by_id["part_whole"], run_dir)
    part_whole_before, _ = runner._stage_fingerprint(
        part_whole, {"semantics": semantic_before}
    )
    part_whole_after, _ = runner._stage_fingerprint(
        part_whole, {"semantics": semantic_after}
    )
    assert part_whole_before == part_whole_after


def test_each_standard_stage_declares_target_code_dependencies() -> None:
    assert tuple(STANDARD_STAGE_CODE_INPUTS) == STANDARD_STAGE_IDS
    assert all(STANDARD_STAGE_CODE_INPUTS[stage_id] for stage_id in STANDARD_STAGE_IDS)
    missing = [
        relative_path
        for paths in (
            STANDARD_SHARED_CODE_INPUTS,
            *STANDARD_STAGE_CODE_INPUTS.values(),
        )
        for relative_path in paths
        if not (ROOT / relative_path).is_file()
    ]
    assert missing == []
    assert (
        "scripts/view_scene_state.py"
        in STANDARD_STAGE_CODE_INPUTS["qa_bundle"]
    )
    assert "src/farm_pipeline/final_acceptance.py" in STANDARD_STAGE_CODE_INPUTS["finalize"]
    assert "src/scene_graph/captioning/evidence.py" in STANDARD_STAGE_CODE_INPUTS["finalize"]
    assert "scripts/build_farm_final_acceptance.py" in STANDARD_STAGE_CODE_INPUTS["qa_bundle"]
    assert "src/scene_graph/captioning/evidence.py" in STANDARD_STAGE_CODE_INPUTS["qa_bundle"]
    semantic_shared = {
        "src/scene_graph/captioning/evidence.py",
        "src/scene_graph/captioning/label_contract.py",
        "src/scene_graph/captioning/physical_guard.py",
    }
    assert semantic_shared <= set(STANDARD_STAGE_CODE_INPUTS["semantics"])
    assert semantic_shared - {
        "src/scene_graph/captioning/physical_guard.py"
    } <= set(STANDARD_STAGE_CODE_INPUTS["assemblies"])


@pytest.mark.parametrize("script", ["farm_pipeline.py", "farm_standard_stage.py"])
def test_direct_cli_does_not_shadow_farm_pipeline_package(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_scripts_first_path_still_allows_farm_pipeline_submodules() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
        "import farm_pipeline.scene_config as scene_config; "
        "assert scene_config.SCHEMA_VERSION == 'farm.scene.v1'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_selected_gpu_telemetry_is_explicitly_whole_device() -> None:
    telemetry = {
        "gpu_device_baseline_used_by_uuid_mb": {"GPU-A": 100},
        "gpu_device_peak_used_by_uuid_mb": {"GPU-A": 700},
        "gpu_device_peak_delta_by_uuid_mb": {"GPU-A": 600},
        "gpu_device_min_free_by_uuid_mb": {"GPU-A": 900},
        "gpu_device_total_by_uuid_mb": {"GPU-A": 1600},
        "gpu_pid_attributed_peak_total_mb": 0,
    }
    value = selected_device_telemetry(
        telemetry, [{"status": "PASS", "selected_gpu": {"uuid": "GPU-A", "index": 2}}]
    )
    assert value == {
        "measurement_scope": "whole_selected_device_not_pid_attributed",
        "uuid": "GPU-A", "index": 2, "name": None,
        "baseline_used_mb": 100, "peak_used_mb": 700, "peak_delta_mb": 600,
        "min_free_mb": 900, "total_mb": 1600,
    }


def test_plan_rejects_cycles_literal_secrets_and_unreliable_viewer_override(tmp_path: Path) -> None:
    cyclic = [{"id": "a", "needs": ["b"], "command": ["true"]}, {"id": "b", "needs": ["a"], "command": ["true"]}]
    with pytest.raises(PlanError, match="cycle"):
        load_plan(_advanced_config(tmp_path, cyclic, {}))
    literal_secret = [{"id": "a", "command": ["true"], "env": {"HF_TOKEN": "plain-text"}}]
    with pytest.raises(PlanError, match="literal secret"):
        load_plan(_advanced_config(tmp_path, literal_secret, {}))
    raw = yaml.safe_load(_advanced_config(tmp_path, [{"id": "a", "command": ["true"]}], {}).read_text())
    raw["viewer"] = {"command": ["python", "viewer.py", "--port", "8080"]}
    viewer_path = tmp_path / "viewer.yaml"
    viewer_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="viewer.command must contain"):
        load_plan(viewer_path)
