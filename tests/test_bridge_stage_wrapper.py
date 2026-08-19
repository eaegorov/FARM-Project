from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.farm_shaper_bridge import run_bridge_stage as bridge


PREP_ID = "sha256:" + "1" * 64
SHAPER_ID = "sha256:" + "2" * 64
TREE = "3" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorities(tmp_path: Path) -> tuple[bridge.RunAuthority, bridge.ShaperAuthority]:
    run = tmp_path / "run-1"
    snapshot = run / "config/source_snapshot/FARM-Project"
    snapshot.mkdir(parents=True)
    config = snapshot / "configs/shaper_bridge.v1.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("schema_version: test\n", encoding="utf-8")
    authority = bridge.RunAuthority(
        run_dir=run,
        snapshot_root=snapshot,
        source_tree_sha256=TREE,
        manifest={"run_id": run.name},
        success={"scene_id": "factory", "run_id": run.name},
        prep=bridge.RuntimePin(
            "prep", "prep:tag", PREP_ID, bridge.CANONICAL_PREP_PYTHON
        ),
    )
    shaper = bridge.ShaperAuthority(
        config_path=config,
        config_sha256=_sha(config),
        runtime=bridge.RuntimePin(
            "shaper", "shaper:tag", SHAPER_ID, "/opt/shaper-venv/bin/python"
        ),
        shm_size="16g",
        tmpfs_size="16g",
    )
    return authority, shaper


def _mounts(command: list[str]) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]


def test_prepare_command_has_one_exact_rw_bind_and_no_implicit_dev_overrides(
    tmp_path: Path,
) -> None:
    authority, shaper = _authorities(tmp_path)
    source = tmp_path / "source.ply"
    source.write_bytes(b"ply")
    lift = tmp_path / "lift"
    lift.mkdir()
    output = tmp_path / "prepared"
    output.mkdir()

    command = bridge.build_prepare_command(
        authority=authority,
        shaper=shaper,
        source_ply=source,
        lift=lift,
        output=output,
        allow_nonrelease_lift=False,
        allow_legacy_run=False,
    )
    mounts = _mounts(command)
    output_mount = f"type=bind,src={output},dst={output}"
    assert mounts == [
        f"type=bind,src={authority.run_dir},dst={authority.run_dir},readonly",
        f"type=bind,src={source},dst={source},readonly",
        f"type=bind,src={lift},dst={lift},readonly",
        output_mount,
    ]
    assert mounts.count(output_mount) == 1
    assert all(value.endswith(",readonly") for value in mounts[:-1])
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--entrypoint") + 1] == bridge.CANONICAL_PREP_PYTHON
    assert PREP_ID in command and "prep:tag" not in command
    assert "--gpus" not in command
    assert command[command.index(PREP_ID) + 1] == str(
        authority.snapshot_root / "tools/farm_shaper_bridge/build_shaper_inputs.py"
    )
    assert "--allow-nonrelease-lift" not in command
    assert "--allow-legacy-run" not in command

    development = bridge.build_prepare_command(
        authority=authority,
        shaper=shaper,
        source_ply=source,
        lift=lift,
        output=output,
        allow_nonrelease_lift=True,
        allow_legacy_run=True,
    )
    assert development[-2:] == ["--allow-nonrelease-lift", "--allow-legacy-run"]


def test_assemble_uses_exact_shaper_image_and_all_inputs_are_readonly(
    tmp_path: Path,
) -> None:
    authority, shaper = _authorities(tmp_path)
    batch = tmp_path / "batch"
    batch.mkdir()
    output = tmp_path / "scene"
    output.mkdir()
    command = bridge.build_assemble_command(
        authority=authority,
        shaper=shaper,
        shaper_outputs=batch,
        output=output,
    )
    mounts = _mounts(command)
    assert mounts == [
        f"type=bind,src={authority.run_dir},dst={authority.run_dir},readonly",
        f"type=bind,src={batch},dst={batch},readonly",
        f"type=bind,src={output},dst={output}",
    ]
    assert command[command.index("--entrypoint") + 1] == "/opt/shaper-venv/bin/python"
    assert SHAPER_ID in command and "shaper:tag" not in command
    assert str(authority.snapshot_root / "tools/farm_shaper_bridge/assemble_scene.py") in command


def _write_run_evidence(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run-1"
    snapshot = run / "config/source_snapshot/FARM-Project"
    model = snapshot / "configs/models/farm_models.v1.json"
    model.parent.mkdir(parents=True)
    for relative in bridge.SIGNED_STAGE_FILES:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# signed fixture\n", encoding="utf-8")
    model.write_text(json.dumps({
        "schema_version": "farm.models.v1",
        "runtimes": {
            "prep": {
                "image": "prep:tag",
                "image_id": PREP_ID,
                "python": bridge.CANONICAL_PREP_PYTHON,
            }
        },
    }), encoding="utf-8")
    model_sha = _sha(model)
    original_root = Path("/original/FARM-Project")
    original_model = original_root / "configs/models/farm_models.v1.json"
    resource = {
        "schema_version": "farm.resource_preflight.v1",
        "status": "pass",
        "strict_ready": True,
        "errors": 0,
        "manifest_path": str(original_model),
        "manifest_sha256": model_sha,
        "checks": [{
            "name": "runtime:prep",
            "status": "pass",
            "metrics": {
                "image": "prep:tag",
                "expected_image_id": PREP_ID,
                "actual_image_id": PREP_ID,
                "python": bridge.CANONICAL_PREP_PYTHON,
            },
        }],
    }
    context = {
        "schema": "farm.standard-resolved-context.v1",
        "model_manifest": str(original_model),
        "model_manifest_sha256": model_sha,
    }
    resource_path = run / "input/resource_preflight.json"
    context_path = run / "input/resolved_context.json"
    resource_path.parent.mkdir(parents=True)
    resource_path.write_text(json.dumps(resource), encoding="utf-8")
    context_path.write_text(json.dumps(context), encoding="utf-8")
    bundle = {
        "schema": "farm.viewer-bundle.v1",
        "scene_id": "factory",
        "run_id": run.name,
        "artifacts": {
            "resource_preflight": "../input/resource_preflight.json",
            "resolved_context": "../input/resolved_context.json",
        },
        "artifact_integrity": {
            "resource_preflight": {
                "kind": "file",
                "path": "../input/resource_preflight.json",
                "bytes": resource_path.stat().st_size,
                "sha256": _sha(resource_path),
            },
            "resolved_context": {
                "kind": "file",
                "path": "../input/resolved_context.json",
                "bytes": context_path.stat().st_size,
                "sha256": _sha(context_path),
            },
        },
    }
    bundle_path = run / "viewer/bundle.json"
    bundle_path.parent.mkdir()
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    manifest = {
        "schema": "farm.pipeline-run.v1",
        "status": "success",
        "scene_id": "factory",
        "run_id": run.name,
        "config_sha256": "4" * 64,
        "project_root": str(original_root),
        "source_snapshot": {"tree_sha256": TREE},
    }
    success = {
        "schema": "farm.pipeline-success.v1",
        "status": "success",
        "scene_id": "factory",
        "run_id": run.name,
        "config_sha256": "4" * 64,
        "viewer_bundle": "viewer/bundle.json",
        "viewer_bundle_sha256": _sha(bundle_path),
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "_SUCCESS.json").write_text(json.dumps(success), encoding="utf-8")
    return run, snapshot


def test_run_authority_requires_signed_snapshot_and_root_bound_runtime_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, snapshot = _write_run_evidence(tmp_path)
    observed: dict[str, object] = {}

    def validate(run_dir, run_manifest=None, *, required=False):
        observed.update(run_dir=run_dir, manifest=run_manifest, required=required)
        return snapshot

    monkeypatch.setattr(bridge, "validated_source_snapshot_project_root", validate)
    monkeypatch.setattr(bridge, "_require_live_control_matches", lambda root: None)
    authority = bridge.validate_run_authority(run)
    assert observed["required"] is True
    assert observed["run_dir"] == run
    assert authority.snapshot_root == snapshot
    assert authority.prep.image_id == PREP_ID
    assert authority.prep.python == bridge.CANONICAL_PREP_PYTHON

    resource_path = run / "input/resource_preflight.json"
    payload = json.loads(resource_path.read_text())
    payload["checks"][0]["metrics"]["python"] = "/wrong/python"
    resource_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(bridge.BridgeStageError, match="fingerprint mismatch"):
        bridge.validate_run_authority(run)


def test_live_wrapper_and_config_must_match_signed_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "live"
    signed = tmp_path / "signed"
    for relative in bridge.LIVE_CONTROL_FILES:
        (live / relative).parent.mkdir(parents=True, exist_ok=True)
        (signed / relative).parent.mkdir(parents=True, exist_ok=True)
        (live / relative).write_text(relative, encoding="utf-8")
        (signed / relative).write_text(relative, encoding="utf-8")
    monkeypatch.setattr(bridge, "ROOT", live)
    bridge._require_live_control_matches(signed)
    (live / bridge.SHAPER_CONFIG_RELATIVE).write_text("drift", encoding="utf-8")
    with pytest.raises(bridge.SourceSnapshotIntegrityError, match="differs"):
        bridge._require_live_control_matches(signed)


def test_image_inspection_fails_closed_on_tag_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = bridge.RuntimePin("prep", "prep:tag", PREP_ID, bridge.CANONICAL_PREP_PYTHON)
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=PREP_ID + "\n", stderr=""
        ),
    )
    assert bridge.inspect_image_id(runtime) == PREP_ID
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=SHAPER_ID + "\n", stderr=""
        ),
    )
    with pytest.raises(bridge.BridgeStageError, match="image drift"):
        bridge.inspect_image_id(runtime)


@pytest.mark.parametrize("stage", ["prepare", "assemble"])
def test_completion_marker_is_exact_and_digest_bound(tmp_path: Path, stage: str) -> None:
    authority, shaper = _authorities(tmp_path)
    output = tmp_path / f"complete-{stage}"
    output.mkdir()
    if stage == "prepare":
        name = "result.json"
        schema = "farm.shaper-inputs.result.v1"
        marker_schema = "farm.shaper-inputs.success.v1"
        inputs = {
            "farm_run": str(authority.run_dir),
            "config_sha256": shaper.config_sha256,
        }
    else:
        name = "scene_manifest.json"
        schema = "farm.shaper-scene.v2"
        marker_schema = "farm.shaper-scene.success.v2"
        inputs = {"farm_run": str(authority.run_dir)}
        viewer = output / "viewer_assets.json"
        viewer.write_text("{}", encoding="utf-8")
    result = {
        "schema_version": schema,
        "status": "PASS",
        "release_eligible": True,
        "scene_id": "factory",
        "inputs": inputs,
        "source_authority": {"signed": True, "tree_sha256": TREE},
    }
    result_path = output / name
    result_path.write_text(json.dumps(result), encoding="utf-8")
    marker = {
        "schema_version": marker_schema,
        "status": "success",
        "release_eligible": True,
        "scene_id": "factory",
        "result": name,
        "result_sha256": _sha(result_path),
    }
    if stage == "assemble":
        marker.update(
            viewer_assets="viewer_assets.json",
            viewer_assets_sha256=_sha(output / "viewer_assets.json"),
        )
    (output / "_SUCCESS.json").write_text(json.dumps(marker), encoding="utf-8")
    summary = bridge._validate_completion(
        stage=stage, output=output, authority=authority, shaper=shaper
    )
    assert summary["status"] == "success"
    result_path.write_text("{}", encoding="utf-8")
    with pytest.raises(bridge.BridgeStageError, match="contract mismatch"):
        bridge._validate_completion(
            stage=stage, output=output, authority=authority, shaper=shaper
        )


def test_host_wrapper_has_no_scientific_or_graphics_imports() -> None:
    source = (Path(__file__).resolve().parents[1] / bridge.WRAPPER_RELATIVE).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported.intersection({"numpy", "cv2", "scipy", "torch", "trimesh", "PIL"})
    assert "tools.farm_shaper_bridge.common" not in source
    assert "tools.farm_shaper_bridge.shaper_contracts" not in source
