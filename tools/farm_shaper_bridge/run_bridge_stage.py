#!/usr/bin/env python3
"""Run ShapeR bridge pre/post stages in immutable Docker runtimes.

The host control plane intentionally imports only the Python standard library,
PyYAML, and the lightweight FARM source-snapshot validator.  Scientific and
graphics packages stay inside the run-pinned prep/ShapeR images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.source_snapshot import (  # noqa: E402
    SourceSnapshotIntegrityError,
    validated_source_snapshot_project_root,
)


IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SIZE_RE = re.compile(r"^[1-9][0-9]*(?:[kKmMgG])?$")
CANONICAL_PREP_PYTHON = "/opt/conda/envs/rest3d/bin/python"
MODEL_SCHEMA = "farm.models.v1"
SHAPER_CONFIG_SCHEMA = "farm.shaper-bridge.config.v1"
WRAPPER_RELATIVE = "tools/farm_shaper_bridge/run_bridge_stage.py"
SHAPER_CONFIG_RELATIVE = "configs/shaper_bridge.v1.yaml"
LIVE_CONTROL_FILES = (
    WRAPPER_RELATIVE,
    SHAPER_CONFIG_RELATIVE,
    "src/farm_runtime/source_snapshot.py",
    "src/farm_runtime/config.py",
)
SIGNED_STAGE_FILES = (
    "tools/farm_shaper_bridge/build_shaper_inputs.py",
    "tools/farm_shaper_bridge/assemble_scene.py",
    "tools/farm_shaper_bridge/shaper_contracts.py",
    "tools/farm_shaper_bridge/common.py",
)


class BridgeStageError(ValueError):
    """Raised when bridge execution cannot be proven safe/reproducible."""


@dataclass(frozen=True)
class RuntimePin:
    name: str
    image: str
    image_id: str
    python: str


@dataclass(frozen=True)
class RunAuthority:
    run_dir: Path
    snapshot_root: Path
    source_tree_sha256: str
    manifest: Mapping[str, Any]
    success: Mapping[str, Any]
    prep: RuntimePin


@dataclass(frozen=True)
class ShaperAuthority:
    config_path: Path
    config_sha256: str
    runtime: RuntimePin
    shm_size: str
    tmpfs_size: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeStageError(f"invalid or missing {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BridgeStageError(f"{label} must be a JSON object: {path}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeStageError(f"{label} must be a mapping")
    return value


def _normalized_container_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BridgeStageError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BridgeStageError(f"{label} must be a normalized absolute path")
    return value


def _safe_file(path: Path, *, label: str) -> Path:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise BridgeStageError(f"{label} must not be a symlink: {unresolved}")
    try:
        resolved = unresolved.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BridgeStageError(f"missing {label}: {unresolved}") from exc
    if not resolved.is_file():
        raise BridgeStageError(f"{label} is not a regular file: {resolved}")
    return resolved


def _safe_directory(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BridgeStageError(f"missing {label}: {path}") from exc
    if not resolved.is_dir():
        raise BridgeStageError(f"{label} is not a directory: {resolved}")
    return resolved


def _bundle_file(
    run_dir: Path,
    bundle: Mapping[str, Any],
    *,
    name: str,
    canonical_relative: str,
) -> Path:
    artifacts = _mapping(bundle.get("artifacts"), label="viewer bundle artifacts")
    descriptors = _mapping(
        bundle.get("artifact_integrity"), label="viewer bundle artifact_integrity"
    )
    relative = artifacts.get(name)
    descriptor = _mapping(descriptors.get(name), label=f"artifact descriptor {name!r}")
    if not isinstance(relative, str) or descriptor.get("path") != relative:
        raise BridgeStageError(f"viewer bundle path/descriptor mismatch for {name!r}")
    unresolved = run_dir / "viewer" / relative
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(run_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise BridgeStageError(f"viewer artifact escapes or is missing: {name!r}") from exc
    canonical = (run_dir / canonical_relative).resolve(strict=True)
    if path != canonical or unresolved.is_symlink() or canonical.is_symlink():
        raise BridgeStageError(f"viewer artifact has non-canonical path: {name!r}")
    if descriptor.get("kind") != "file" or not path.is_file():
        raise BridgeStageError(f"viewer artifact is not one immutable file: {name!r}")
    expected_sha = str(descriptor.get("sha256") or "")
    try:
        expected_bytes = int(descriptor.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise BridgeStageError(f"viewer artifact byte count is invalid: {name!r}") from exc
    if (
        not SHA256_RE.fullmatch(expected_sha)
        or expected_bytes != path.stat().st_size
        or _sha256_file(path) != expected_sha
    ):
        raise BridgeStageError(f"viewer artifact fingerprint mismatch: {name!r}")
    return path


def _validated_root_bundle(
    run_dir: Path, success: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Path, Path]:
    if success.get("viewer_bundle") != "viewer/bundle.json":
        raise BridgeStageError("FARM success does not bind the canonical viewer bundle")
    expected = str(success.get("viewer_bundle_sha256") or "")
    bundle_path = _safe_file(run_dir / "viewer" / "bundle.json", label="viewer bundle")
    if not SHA256_RE.fullmatch(expected) or _sha256_file(bundle_path) != expected:
        raise BridgeStageError("FARM root-success/viewer-bundle digest mismatch")
    bundle = _load_json(bundle_path, label="viewer bundle")
    if (
        bundle.get("schema") != "farm.viewer-bundle.v1"
        or bundle.get("scene_id") != success.get("scene_id")
        or bundle.get("run_id") != success.get("run_id")
    ):
        raise BridgeStageError("FARM viewer bundle identity mismatch")
    resources = _bundle_file(
        run_dir,
        bundle,
        name="resource_preflight",
        canonical_relative="input/resource_preflight.json",
    )
    context = _bundle_file(
        run_dir,
        bundle,
        name="resolved_context",
        canonical_relative="input/resolved_context.json",
    )
    return bundle, resources, context


def _signed_model_manifest(
    *,
    run_manifest: Mapping[str, Any],
    snapshot_root: Path,
    resource_report: Mapping[str, Any],
    resolved_context: Mapping[str, Any],
) -> Path:
    recorded = resource_report.get("manifest_path")
    context_recorded = resolved_context.get("model_manifest")
    if not isinstance(recorded, str) or recorded != context_recorded:
        raise BridgeStageError("resource/context model-manifest paths differ")
    original_root = run_manifest.get("project_root")
    if not isinstance(original_root, str):
        raise BridgeStageError("run manifest has no original project_root")
    original_path = Path(recorded)
    original_project = Path(original_root)
    try:
        relative = original_path.relative_to(original_project)
    except ValueError as exc:
        raise BridgeStageError("model manifest was outside the snapshotted FARM project") from exc
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise BridgeStageError("recorded model-manifest path is not normalized")
    signed = (snapshot_root / relative).resolve(strict=True)
    if not signed.is_file() or not signed.is_relative_to(snapshot_root):
        raise BridgeStageError("signed source snapshot misses the run model manifest")
    digest = _sha256_file(signed)
    expected = str(resource_report.get("manifest_sha256") or "")
    context_expected = str(resolved_context.get("model_manifest_sha256") or "")
    if not SHA256_RE.fullmatch(expected) or digest != expected or digest != context_expected:
        raise BridgeStageError("signed model manifest differs from run preflight evidence")
    return signed


def _prep_runtime(
    model_manifest_path: Path,
    resource_report: Mapping[str, Any],
) -> RuntimePin:
    model_manifest = _load_json(model_manifest_path, label="signed model manifest")
    if model_manifest.get("schema_version") != MODEL_SCHEMA:
        raise BridgeStageError("unsupported signed model manifest schema")
    runtimes = _mapping(model_manifest.get("runtimes"), label="model runtimes")
    prep = _mapping(runtimes.get("prep"), label="runtimes.prep")
    image = prep.get("image")
    image_id = prep.get("image_id")
    python = _normalized_container_path(prep.get("python"), label="runtimes.prep.python")
    if not isinstance(image, str) or not image or not IMAGE_ID_RE.fullmatch(str(image_id)):
        raise BridgeStageError("runtimes.prep image/image_id is not pinned")
    if python != CANONICAL_PREP_PYTHON:
        raise BridgeStageError(
            f"runtimes.prep.python must be exactly {CANONICAL_PREP_PYTHON}"
        )
    checks = resource_report.get("checks")
    if not isinstance(checks, list):
        raise BridgeStageError("resource preflight has no checks list")
    matches = [
        row for row in checks
        if isinstance(row, Mapping) and row.get("name") == "runtime:prep"
    ]
    if len(matches) != 1 or matches[0].get("status") != "pass":
        raise BridgeStageError("resource preflight did not pass exactly one prep runtime check")
    metrics = _mapping(matches[0].get("metrics"), label="runtime:prep metrics")
    if (
        metrics.get("image") != image
        or metrics.get("expected_image_id") != image_id
        or metrics.get("actual_image_id") != image_id
        or metrics.get("python") != python
    ):
        raise BridgeStageError("prep runtime manifest/preflight evidence differs")
    return RuntimePin("prep", image, str(image_id), python)


def _require_live_control_matches(snapshot_root: Path) -> None:
    mismatches: list[str] = []
    for relative in LIVE_CONTROL_FILES:
        live = _safe_file(ROOT / relative, label=f"live control file {relative}")
        signed = _safe_file(
            snapshot_root / relative, label=f"signed control file {relative}"
        )
        if not signed.is_relative_to(snapshot_root) or _sha256_file(live) != _sha256_file(signed):
            mismatches.append(relative)
    if mismatches:
        raise SourceSnapshotIntegrityError(
            "live bridge control/config differs from the run's signed source: "
            + ", ".join(mismatches)
        )


def validate_run_authority(run: Path) -> RunAuthority:
    run_dir = _safe_directory(run, label="FARM run")
    manifest = _load_json(_safe_file(run_dir / "manifest.json", label="run manifest"), label="run manifest")
    success = _load_json(
        _safe_file(run_dir / "_SUCCESS.json", label="FARM root success"),
        label="FARM root success",
    )
    if manifest.get("schema") != "farm.pipeline-run.v1" or manifest.get("status") != "success":
        raise BridgeStageError("FARM run manifest is not successful/current")
    if success.get("schema") != "farm.pipeline-success.v1" or success.get("status") != "success":
        raise BridgeStageError("FARM root completion marker is invalid")
    for field in ("scene_id", "run_id", "config_sha256"):
        if manifest.get(field) != success.get(field):
            raise BridgeStageError(f"FARM manifest/root-success mismatch: {field}")
    if success.get("run_id") != run_dir.name:
        raise BridgeStageError("FARM run_id differs from its directory name")
    snapshot_root = validated_source_snapshot_project_root(
        run_dir, run_manifest=manifest, required=True
    )
    if snapshot_root is None:  # pragma: no cover - required=True is fail-closed.
        raise BridgeStageError("FARM signed source snapshot is absent")
    for relative in SIGNED_STAGE_FILES:
        signed_stage = _safe_file(
            snapshot_root / relative, label=f"signed bridge stage file {relative}"
        )
        if not signed_stage.is_relative_to(snapshot_root):
            raise BridgeStageError(f"signed bridge stage escapes its snapshot: {relative}")
    _require_live_control_matches(snapshot_root)
    _, resource_path, context_path = _validated_root_bundle(run_dir, success)
    resource_report = _load_json(resource_path, label="resource preflight")
    resolved_context = _load_json(context_path, label="resolved context")
    if (
        resource_report.get("schema_version") != "farm.resource_preflight.v1"
        or resource_report.get("status") != "pass"
        or not bool(resource_report.get("strict_ready"))
        or int(resource_report.get("errors", -1)) != 0
    ):
        raise BridgeStageError("FARM resource preflight is not strict PASS")
    if resolved_context.get("schema") != "farm.standard-resolved-context.v1":
        raise BridgeStageError("unsupported FARM resolved-context schema")
    signed_models = _signed_model_manifest(
        run_manifest=manifest,
        snapshot_root=snapshot_root,
        resource_report=resource_report,
        resolved_context=resolved_context,
    )
    prep = _prep_runtime(signed_models, resource_report)
    source = _mapping(manifest.get("source_snapshot"), label="source_snapshot")
    tree = str(source.get("tree_sha256") or "")
    if not SHA256_RE.fullmatch(tree):
        raise BridgeStageError("run source-snapshot tree identity is invalid")
    return RunAuthority(run_dir, snapshot_root, tree, manifest, success, prep)


def load_shaper_authority(authority: RunAuthority) -> ShaperAuthority:
    signed = _safe_file(
        authority.snapshot_root / SHAPER_CONFIG_RELATIVE,
        label="signed ShapeR bridge config",
    )
    try:
        payload = yaml.safe_load(signed.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BridgeStageError(f"invalid signed ShapeR bridge config: {signed}") from exc
    config = _mapping(payload, label="ShapeR bridge config")
    if config.get("schema_version") != SHAPER_CONFIG_SCHEMA:
        raise BridgeStageError("unsupported ShapeR bridge config schema")
    runtime = _mapping(config.get("runtime"), label="ShapeR runtime")
    image = runtime.get("image")
    image_id = runtime.get("image_id")
    python = _normalized_container_path(runtime.get("python"), label="ShapeR runtime.python")
    if not isinstance(image, str) or not image or not IMAGE_ID_RE.fullmatch(str(image_id)):
        raise BridgeStageError("ShapeR runtime image/image_id is not pinned")
    shm_size = str(runtime.get("shm_size") or "")
    tmpfs_size = str(runtime.get("tmpfs_size") or "")
    if not SIZE_RE.fullmatch(shm_size) or not SIZE_RE.fullmatch(tmpfs_size):
        raise BridgeStageError("ShapeR shm_size/tmpfs_size must be explicit Docker sizes")
    return ShaperAuthority(
        signed,
        _sha256_file(signed),
        RuntimePin("shaper", image, str(image_id), python),
        shm_size,
        tmpfs_size,
    )


def inspect_image_id(runtime: RuntimePin) -> str:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", runtime.image],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BridgeStageError(f"cannot inspect {runtime.name} Docker image: {exc}") from exc
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) != 1 or not IMAGE_ID_RE.fullmatch(lines[0]):
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise BridgeStageError(f"cannot resolve pinned {runtime.name} Docker image{suffix}")
    if lines[0] != runtime.image_id:
        raise BridgeStageError(
            f"{runtime.name} Docker image drift: expected {runtime.image_id}, got {lines[0]}"
        )
    return lines[0]


def _mount(path: Path, *, readonly: bool) -> str:
    value = str(path)
    if "," in value or "\n" in value or "\r" in value:
        raise BridgeStageError(f"Docker bind path contains an unsupported character: {path}")
    spec = f"type=bind,src={value},dst={value}"
    return spec + (",readonly" if readonly else "")


def _docker_base(
    *,
    inputs: Sequence[Path],
    output: Path,
    runtime: RuntimePin,
    workdir: Path,
    shm_size: str,
    tmpfs_size: str,
) -> list[str]:
    command = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "4096",
        "--shm-size", shm_size,
        "--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={tmpfs_size}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "HOME=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--workdir", str(workdir),
    ]
    seen: set[Path] = set()
    for path in inputs:
        if path in seen:
            continue
        seen.add(path)
        command.extend(["--mount", _mount(path, readonly=True)])
    command.extend([
        "--mount", _mount(output, readonly=False),
        "--entrypoint", runtime.python,
        runtime.image_id,
    ])
    return command


def build_prepare_command(
    *,
    authority: RunAuthority,
    shaper: ShaperAuthority,
    source_ply: Path,
    lift: Path,
    output: Path,
    allow_nonrelease_lift: bool,
    allow_legacy_run: bool,
) -> list[str]:
    script = authority.snapshot_root / "tools/farm_shaper_bridge/build_shaper_inputs.py"
    command = _docker_base(
        inputs=(authority.run_dir, source_ply, lift),
        output=output,
        runtime=authority.prep,
        workdir=authority.snapshot_root,
        shm_size=shaper.shm_size,
        tmpfs_size=shaper.tmpfs_size,
    )
    command.extend([
        str(script),
        "--run", str(authority.run_dir),
        "--ply", str(source_ply),
        "--lift", str(lift),
        "--output", str(output),
        "--config", str(shaper.config_path),
    ])
    if allow_nonrelease_lift:
        command.append("--allow-nonrelease-lift")
    if allow_legacy_run:
        command.append("--allow-legacy-run")
    return command


def build_assemble_command(
    *,
    authority: RunAuthority,
    shaper: ShaperAuthority,
    shaper_outputs: Path,
    output: Path,
) -> list[str]:
    script = authority.snapshot_root / "tools/farm_shaper_bridge/assemble_scene.py"
    command = _docker_base(
        inputs=(authority.run_dir, shaper_outputs),
        output=output,
        runtime=shaper.runtime,
        workdir=authority.snapshot_root,
        shm_size=shaper.shm_size,
        tmpfs_size=shaper.tmpfs_size,
    )
    command.extend([
        str(script),
        "--run", str(authority.run_dir),
        "--shaper-outputs", str(shaper_outputs),
        "--output", str(output),
    ])
    return command


def _resolved_output(path: Path) -> Path:
    requested = path.expanduser()
    if requested.exists() or requested.is_symlink():
        if requested.is_symlink():
            raise BridgeStageError(f"output directory must not be a symlink: {requested}")
        output = requested.resolve(strict=True)
        if not output.is_dir() or any(output.iterdir()):
            raise BridgeStageError(f"refusing non-empty exact output directory: {output}")
        return output
    try:
        parent = requested.parent.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BridgeStageError(f"output parent must already exist: {requested.parent}") from exc
    if not parent.is_dir():
        raise BridgeStageError(f"output parent is not a directory: {parent}")
    output = parent / requested.name
    if output == Path("/") or not requested.name:
        raise BridgeStageError("unsafe output directory")
    return output


def _reject_output_overlap(output: Path, inputs: Sequence[Path]) -> None:
    for source in inputs:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise BridgeStageError(f"output overlaps immutable input: {source}")


def _ensure_empty_output(output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise BridgeStageError(f"refusing non-empty exact output directory: {output}")
        return
    output.mkdir(mode=0o775, exist_ok=False)


def _validate_completion(
    *,
    stage: str,
    output: Path,
    authority: RunAuthority,
    shaper: ShaperAuthority,
) -> dict[str, Any]:
    markers = [
        name for name in ("_SUCCESS.json", "_NONRELEASE_SUCCESS.json")
        if (output / name).is_file()
    ]
    if len(markers) != 1 or (output / "_FAILED.json").exists():
        raise BridgeStageError(f"{stage} did not publish exactly one successful marker")
    marker_name = markers[0]
    marker_path = _safe_file(
        output / marker_name, label=f"{stage} completion marker"
    )
    marker = _load_json(marker_path, label=f"{stage} completion marker")
    if stage == "prepare":
        result_name = "result.json"
        result_schema = "farm.shaper-inputs.result.v1"
        marker_schema = "farm.shaper-inputs.success.v1"
    else:
        result_name = "scene_manifest.json"
        result_schema = "farm.shaper-scene.v2"
        marker_schema = "farm.shaper-scene.success.v2"
    result_path = _safe_file(output / result_name, label=f"{stage} result")
    result = _load_json(result_path, label=f"{stage} result")
    release = marker_name == "_SUCCESS.json"
    if (
        marker.get("schema_version") != marker_schema
        or marker.get("status") != "success"
        or marker.get("result") != result_name
        or marker.get("result_sha256") != _sha256_file(result_path)
        or bool(marker.get("release_eligible")) != release
        or result.get("schema_version") != result_schema
        or result.get("status") != "PASS"
        or bool(result.get("release_eligible")) != release
        or result.get("scene_id") != authority.success.get("scene_id")
    ):
        raise BridgeStageError(f"{stage} completion marker/result contract mismatch")
    inputs = _mapping(result.get("inputs"), label=f"{stage} result inputs")
    if str(inputs.get("farm_run")) != str(authority.run_dir):
        raise BridgeStageError(f"{stage} result is bound to a different FARM run")
    source = _mapping(result.get("source_authority"), label=f"{stage} source authority")
    if not bool(source.get("signed")) or source.get("tree_sha256") != authority.source_tree_sha256:
        raise BridgeStageError(f"{stage} result did not execute the signed FARM source")
    if stage == "prepare" and inputs.get("config_sha256") != shaper.config_sha256:
        raise BridgeStageError("prepare result used a different ShapeR config")
    if stage == "assemble":
        viewer_name = marker.get("viewer_assets")
        viewer_path = _safe_file(output / "viewer_assets.json", label="ShapeR viewer assets")
        if (
            viewer_name != "viewer_assets.json"
            or marker.get("viewer_assets_sha256") != _sha256_file(viewer_path)
        ):
            raise BridgeStageError("assemble completion/viewer-assets binding mismatch")
    return {
        "schema": "farm.bridge-stage-wrapper.result.v1",
        "status": "success",
        "stage": stage,
        "scene_id": result["scene_id"],
        "release_eligible": release,
        "marker": str(output / marker_name),
        "result": str(result_path),
        "runtime_image_id": authority.prep.image_id if stage == "prepare" else shaper.runtime.image_id,
        "source_tree_sha256": authority.source_tree_sha256,
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="successful signed FARM run")
    parser.add_argument("--output", type=Path, required=True, help="exact new/empty output directory")
    parser.add_argument("--plan-only", action="store_true", help="validate and print; do not create output/run Docker")
    parser.add_argument("--print-command", action="store_true", help="print the exact Docker argv before execution")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="build ShapeR inputs inside the run-pinned prep image"
    )
    _common_arguments(prepare)
    prepare.add_argument("--ply", type=Path, required=True, help="exact immutable source 3DGS PLY")
    prepare.add_argument("--lift", type=Path, required=True, help="completed Gaussian lift directory")
    prepare.add_argument(
        "--allow-nonrelease-lift",
        action="store_true",
        help="explicit development override forwarded to signed input preparation",
    )
    prepare.add_argument(
        "--allow-legacy-run",
        action="store_true",
        help="explicit development override forwarded to signed input preparation",
    )

    assemble = subparsers.add_parser(
        "assemble", help="assemble QA-passing meshes inside the exact ShapeR image"
    )
    _common_arguments(assemble)
    assemble.add_argument(
        "--shaper-outputs", type=Path, required=True, help="completed ShapeR batch directory"
    )
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    authority = validate_run_authority(args.run)
    shaper = load_shaper_authority(authority)
    output = _resolved_output(args.output)
    if args.stage == "prepare":
        source_ply = _safe_file(args.ply, label="source 3DGS PLY")
        lift = _safe_directory(args.lift, label="Gaussian lift")
        immutable_inputs = (authority.run_dir, source_ply, lift)
        _reject_output_overlap(output, immutable_inputs)
        inspect_image_id(authority.prep)
        command = build_prepare_command(
            authority=authority,
            shaper=shaper,
            source_ply=source_ply,
            lift=lift,
            output=output,
            allow_nonrelease_lift=bool(args.allow_nonrelease_lift),
            allow_legacy_run=bool(args.allow_legacy_run),
        )
    else:
        shaper_outputs = _safe_directory(args.shaper_outputs, label="ShapeR batch")
        immutable_inputs = (authority.run_dir, shaper_outputs)
        _reject_output_overlap(output, immutable_inputs)
        inspect_image_id(shaper.runtime)
        command = build_assemble_command(
            authority=authority,
            shaper=shaper,
            shaper_outputs=shaper_outputs,
            output=output,
        )
    plan = {
        "schema": "farm.bridge-stage-wrapper.plan.v1",
        "stage": args.stage,
        "run": str(authority.run_dir),
        "scene_id": authority.success["scene_id"],
        "signed_source_tree_sha256": authority.source_tree_sha256,
        "shaper_config_sha256": shaper.config_sha256,
        "runtime_image_id": authority.prep.image_id if args.stage == "prepare" else shaper.runtime.image_id,
        "output": str(output),
        "output_mode": "new-or-empty-exact-bind",
    }
    if args.plan_only or args.print_command:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(shlex.join(command))
    if args.plan_only:
        return 0
    _ensure_empty_output(output)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return int(completed.returncode)
    print(json.dumps(_validate_completion(
        stage=args.stage,
        output=output,
        authority=authority,
        shaper=shaper,
    ), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except (BridgeStageError, SourceSnapshotIntegrityError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
