#!/usr/bin/env python3
"""Run the pinned ShapeR batch container without trusting a dirty checkout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from tools.farm_shaper_bridge.common import canonical_sha256, load_run, sha256_file  # noqa: E402
from tools.farm_shaper_bridge.shaper_contracts import (  # noqa: E402
    SHAPER_INPUT_SCHEMA,
    load_shaper_config,
    require_current_sources_match_authority,
    resolve_farm_source_authority,
    validate_success_result,
)


GPU_RE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_FILES = (
    "tools/farm_shaper_bridge/run_shaper.py",
    "tools/farm_shaper_bridge/shaper_batch.py",
    "tools/farm_shaper_bridge/shaper_contracts.py",
    "tools/farm_shaper_bridge/common.py",
    "configs/shaper_bridge.v1.yaml",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="FARM run owning the source snapshot")
    parser.add_argument("--inputs", type=Path, required=True, help="Completed ShapeR input directory")
    parser.add_argument("--output", type=Path, required=True, help="Exact new/empty RW output directory")
    parser.add_argument("--shaper-repo", type=Path, required=True, help="ShapeR Git repo with checkpoints")
    parser.add_argument("--hf-cache", type=Path, required=True, help="Hugging Face cache root")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "shaper_bridge.v1.yaml",
    )
    parser.add_argument("--profile", choices=("speed", "balance", "quality"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--allow-unsigned-source", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def _run_stdout(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.strip()


def _mount(source: Path, target: str, *, readonly: bool) -> str:
    value = f"type=bind,src={source},dst={target}"
    return value + (",readonly" if readonly else "")


def _content_address(path: Path, advertised: str) -> tuple[str, str]:
    """Return the digest defined by a Hugging Face blob filename."""

    if RAW_SHA256_RE.fullmatch(advertised):
        return "sha256", sha256_file(path)
    if GIT_BLOB_RE.fullmatch(advertised):
        digest = hashlib.sha1()  # nosec B324 - Git object identity, not security.
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        return "git-blob-sha1", digest.hexdigest()
    raise RuntimeError(f"unsupported Hugging Face blob identity: {advertised!r}")


def validate_huggingface_cache(
    hf_cache: Path,
    expected_models: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact offline revisions and their content-addressed blobs."""

    hf_cache = hf_cache.expanduser().resolve(strict=True)
    hub = (hf_cache / "hub").resolve(strict=True)
    rows: dict[str, Any] = {}
    for cache_name, expected in expected_models.items():
        model_root = (hub / str(cache_name)).resolve(strict=True)
        if not model_root.is_dir() or not model_root.is_relative_to(hub):
            raise FileNotFoundError(f"offline ShapeR Hugging Face cache misses: {cache_name}")
        revision = str(expected["revision"])
        ref = (model_root / "refs" / "main").resolve(strict=True)
        if not ref.is_file() or ref.read_text(encoding="utf-8").strip() != revision:
            raise RuntimeError(f"Hugging Face main revision drift for {cache_name}")
        snapshot = (model_root / "snapshots" / revision).resolve(strict=True)
        if not snapshot.is_dir() or not snapshot.is_relative_to(model_root):
            raise FileNotFoundError(f"missing pinned Hugging Face snapshot: {cache_name}@{revision}")
        files: dict[str, Any] = {}
        for filename, record in expected["files"].items():
            relative = Path(str(filename))
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise RuntimeError(f"unsafe Hugging Face snapshot path: {cache_name}/{filename}")
            link = snapshot / relative
            if not link.is_symlink():
                raise RuntimeError(f"Hugging Face cache file is not a blob link: {cache_name}/{filename}")
            target = link.resolve(strict=True)
            blob_root = (model_root / "blobs").resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(blob_root):
                raise RuntimeError(f"Hugging Face cache blob escapes model root: {cache_name}/{filename}")
            advertised = str(record["blob"])
            if target.name != advertised or target.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"Hugging Face blob drift: {cache_name}/{filename}")
            algorithm, content_digest = _content_address(target, advertised)
            if content_digest != advertised:
                raise RuntimeError(f"Hugging Face blob content drift: {cache_name}/{filename}")
            files[str(filename)] = {
                "blob": target.name,
                "bytes": target.stat().st_size,
                "content_digest_algorithm": algorithm,
                "content_digest": content_digest,
            }
        rows[str(cache_name)] = {
            "repository": str(expected.get("repository") or ""),
            "revision": revision,
            "files": files,
        }
    return rows


def validate_runtime_pins(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    shaper_repo: Path,
    hf_cache: Path,
    farm_source_tree: str | None,
) -> dict[str, Any]:
    runtime = config["runtime"]
    shaper_repo = shaper_repo.expanduser().resolve(strict=True)
    if not (shaper_repo / ".git").exists():
        raise ValueError(f"ShapeR source is not a Git worktree: {shaper_repo}")
    commit = str(runtime["shaper_commit"])
    _run_stdout(["git", "-C", str(shaper_repo), "cat-file", "-e", f"{commit}^{{commit}}"])
    tree = _run_stdout(["git", "-C", str(shaper_repo), "rev-parse", f"{commit}^{{tree}}"])
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise RuntimeError("cannot resolve the pinned ShapeR Git tree")
    installed_image_id = _run_stdout([
        "docker", "image", "inspect", "--format", "{{.Id}}", str(runtime["image"])
    ])
    if installed_image_id != runtime["image_id"]:
        raise RuntimeError(
            f"ShapeR image drift: expected {runtime['image_id']}, got {installed_image_id}"
        )
    checkpoint_root = (shaper_repo / "checkpoints").resolve(strict=True)
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    checkpoint_files: dict[str, Path] = {}
    for name, expected in runtime["checkpoints"].items():
        path = (checkpoint_root / str(name)).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"missing pinned ShapeR checkpoint: {name}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(expected["bytes"]) or digest != expected["sha256"]:
            raise RuntimeError(f"ShapeR checkpoint drift: {name}")
        checkpoint_rows[str(name)] = {"bytes": size, "sha256": digest}
        checkpoint_files[str(name)] = path
    hf_cache = hf_cache.expanduser().resolve(strict=True)
    huggingface_models = validate_huggingface_cache(
        hf_cache, runtime["required_huggingface_models"]
    )
    pin_payload = {
        "schema": "farm.shaper-runtime-pin.v1",
        "image_id": installed_image_id,
        "shaper_commit": commit,
        "shaper_tree": tree,
        "checkpoint_source": {
            "repository": str(runtime["checkpoint_repository"]),
            "revision": str(runtime["checkpoint_revision"]),
        },
        "checkpoints": checkpoint_rows,
        "huggingface_models": huggingface_models,
        "config_sha256": config_sha256,
        "farm_source_tree": farm_source_tree,
    }
    return {
        **pin_payload,
        "runtime_pin_sha256": canonical_sha256(pin_payload),
        "checkpoint_files": checkpoint_files,
        "hf_cache": hf_cache,
    }


def archive_clean_shaper_source(
    shaper_repo: Path,
    commit: str,
    destination: Path,
    *,
    checkpoint_names: Sequence[str] = (),
) -> Path:
    """Materialize only tracked bytes from the pinned commit, never the worktree."""

    archive = subprocess.run(
        ["git", "-C", str(shaper_repo), "archive", "--format=tar", commit],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        members = stream.getmembers()
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.isdev():
                raise RuntimeError(f"unsafe member in pinned ShapeR archive: {member.name}")
        stream.extractall(destination, filter="data")
    if not (destination / "dataset" / "shaper_dataset.py").is_file():
        raise RuntimeError("pinned ShapeR archive is incomplete")
    # Docker/runc cannot create a nested file-bind destination below an outer
    # read-only bind.  Materialize only safe empty targets before the clean
    # source directory is mounted read-only; exact pinned files are then
    # over-mounted individually read-only by build_docker_command.
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    names = [str(name) for name in checkpoint_names]
    if len(names) != len(set(names)):
        raise ValueError("duplicate checkpoint mountpoint name")
    for name in sorted(names):
        if not name or Path(name).name != name:
            raise ValueError(f"unsafe checkpoint mountpoint name: {name!r}")
        target = checkpoint_dir / name
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"pinned ShapeR archive already owns checkpoint path: {name}")
        target.touch(exist_ok=False)
    return destination


def build_checkpoint_mount_smoke_command(
    *,
    image_id: str,
    python: str,
    shaper_source: Path,
    checkpoint_files: Mapping[str, Path],
) -> list[str]:
    """Build a real, no-GPU smoke for the production nested mount topology."""

    expected: dict[str, int] = {}
    command = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "128",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", _mount(shaper_source, "/opt/shaper", readonly=True),
    ]
    for name, source in sorted(checkpoint_files.items()):
        target = shaper_source / "checkpoints" / name
        if Path(name).name != name or not source.is_file():
            raise ValueError(f"unsafe pinned checkpoint smoke mount: {name}")
        if not target.is_file() or target.stat().st_size != 0:
            raise RuntimeError(f"checkpoint smoke target is absent or nonempty: {name}")
        expected[name] = source.stat().st_size
        command.extend([
            "--mount", _mount(source, f"/opt/shaper/checkpoints/{name}", readonly=True)
        ])
    smoke_code = (
        "import json,pathlib,sys; expected=json.loads(sys.argv[1]); "
        "root=pathlib.Path('/opt/shaper/checkpoints'); "
        "observed={name:(root/name).stat().st_size for name in expected}; "
        "assert observed==expected, (observed,expected); "
        "print(json.dumps({'status':'PASS','files':observed},sort_keys=True))"
    )
    command.extend([
        "--entrypoint", python,
        image_id,
        "-c", smoke_code,
        json.dumps(expected, sort_keys=True, separators=(",", ":")),
    ])
    return command


def build_docker_command(
    *,
    farm_source: Path,
    shaper_source: Path,
    checkpoint_files: Mapping[str, Path],
    hf_cache: Path,
    inputs: Path,
    output: Path,
    config_path: Path,
    profile: str,
    gpu: str,
    config: Mapping[str, Any],
    pins: Mapping[str, Any],
) -> list[str]:
    runtime = config["runtime"]
    farm_tree = str(pins.get("farm_source_tree") or "")
    command = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "4096",
        "--shm-size", str(runtime["shm_size"]),
        "--tmpfs", f"/tmp:rw,nosuid,nodev,size={runtime['tmpfs_size']}",
        "--gpus", f"device={gpu}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", "/opt/shaper",
        "--env", "HOME=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "HF_HOME=/opt/hf-cache",
        "--env", "HF_HUB_CACHE=/opt/hf-cache/hub",
        "--env", "HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub",
        "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "TOKENIZERS_PARALLELISM=false",
        "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "--env", f"FARM_SHAPER_IMAGE_ID={pins['image_id']}",
        "--env", f"FARM_SHAPER_COMMIT={pins['shaper_commit']}",
        "--env", f"FARM_SHAPER_TREE={pins['shaper_tree']}",
        "--env", f"FARM_SOURCE_TREE={farm_tree}",
        "--env", f"FARM_SHAPER_CONFIG_SHA256={pins['config_sha256']}",
        "--env", f"FARM_SHAPER_RUNTIME_PIN_SHA256={pins['runtime_pin_sha256']}",
        "--mount", _mount(farm_source, "/opt/farm-src", readonly=True),
        "--mount", _mount(shaper_source, "/opt/shaper", readonly=True),
    ]
    for name, path in sorted(checkpoint_files.items()):
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"unsafe pinned checkpoint mount: {name}")
        command.extend([
            "--mount", _mount(path, f"/opt/shaper/checkpoints/{name}", readonly=True)
        ])
    command.extend([
        "--mount", _mount(hf_cache, "/opt/hf-cache", readonly=True),
        "--mount", _mount(inputs, "/farm/inputs", readonly=True),
        # Deliberately mount the exact output directory, never its parent.
        "--mount", _mount(output, "/farm/output", readonly=False),
        "--mount", _mount(config_path, "/farm/config.yaml", readonly=True),
        "--entrypoint", str(runtime["python"]),
        str(runtime["image_id"]),
        "/opt/farm-src/tools/farm_shaper_bridge/shaper_batch.py",
        "--input-dir", "/farm/inputs",
        "--output-dir", "/farm/output",
        "--config", "/farm/config.yaml",
        "--profile", profile,
    ])
    return command


def execute(args: argparse.Namespace) -> int:
    if not GPU_RE.fullmatch(str(args.gpu)):
        raise ValueError("--gpu must be a comma-separated list of non-negative indices")
    project_root = Path(__file__).resolve().parents[2]
    run_dir = args.run.expanduser().resolve(strict=True)
    run = load_run(run_dir, allow_legacy=bool(args.allow_unsigned_source))
    inputs = args.inputs.expanduser().resolve(strict=True)
    input_result, input_marker_name, _ = validate_success_result(
        inputs, result_name="result.json", expected_schema=SHAPER_INPUT_SCHEMA
    )
    recorded_source = input_result.get("source_authority") or {}
    recorded_fallback = recorded_source.get("fallback")
    legacy_nonrelease_fallback = bool(
        run.legacy
        and args.allow_unsigned_source
        and input_marker_name == "_NONRELEASE_SUCCESS.json"
        and not input_result.get("release_eligible")
        and isinstance(recorded_fallback, Mapping)
        and recorded_fallback.get("warning")
        == "legacy_unsigned_fallback_from_incomplete_signed_snapshot"
    )
    authority = resolve_farm_source_authority(
        run_dir,
        project_root,
        required_files=SOURCE_FILES,
        allow_unsigned=bool(args.allow_unsigned_source),
        allow_incomplete_snapshot_fallback=legacy_nonrelease_fallback,
    )
    require_current_sources_match_authority(authority, project_root, SOURCE_FILES)
    if authority.fallback is not None:
        if (
            not isinstance(recorded_fallback, Mapping)
            or recorded_fallback.get("incomplete_snapshot_tree_sha256")
            != authority.fallback.get("incomplete_snapshot_tree_sha256")
        ):
            raise ValueError("ShapeR inputs and legacy source fallback evidence differ")
        print(
            "WARNING: running ShapeR from current versioned FARM source in explicit "
            "unsigned/nonrelease legacy mode",
            file=sys.stderr,
            flush=True,
        )
    config_path = args.config.expanduser().resolve(strict=True)
    authoritative_config = authority.project_root / "configs" / "shaper_bridge.v1.yaml"
    if sha256_file(config_path) != sha256_file(authoritative_config):
        raise ValueError("--config differs from the selected FARM source authority")
    config, config_sha = load_shaper_config(authoritative_config)
    if input_result.get("inputs", {}).get("config_sha256") != config_sha:
        raise ValueError("ShapeR input config digest differs from the authoritative config")
    if str(input_result.get("inputs", {}).get("farm_run")) != str(run_dir):
        raise ValueError("ShapeR inputs are not bound to the requested FARM run")
    recorded_tree = (input_result.get("source_authority") or {}).get("tree_sha256")
    if recorded_tree != authority.tree_sha256:
        raise ValueError("ShapeR inputs and selected FARM source authority differ")
    profile = args.profile or str(config["runtime"]["default_profile"])
    if profile not in config["runtime"]["profiles"]:
        raise ValueError(f"unknown pinned ShapeR profile: {profile}")
    pins = validate_runtime_pins(
        config=config,
        config_sha256=config_sha,
        shaper_repo=args.shaper_repo,
        hf_cache=args.hf_cache,
        farm_source_tree=authority.tree_sha256,
    )
    release_eligible = bool(
        authority.signed
        and input_result.get("release_eligible")
        and input_marker_name == "_SUCCESS.json"
    )
    metadata = {
        "schema_version": "farm.shaper-docker-plan.v1",
        "scene_id": input_result["scene_id"],
        "profile": profile,
        "inputs": str(inputs),
        "output": str(args.output.expanduser().resolve()),
        "farm_source": str(authority.project_root),
        "signed_farm_source": authority.signed,
        "source_authority_fallback": authority.fallback,
        "release_eligible": release_eligible,
        "runtime": {key: value for key, value in pins.items() if key not in {"checkpoint_files", "hf_cache"}},
    }

    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing non-empty exact ShapeR output directory: {output}")
    if not args.plan_only:
        output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="farm-shaper-source-") as temporary:
        shaper_source = archive_clean_shaper_source(
            args.shaper_repo.expanduser().resolve(strict=True),
            str(config["runtime"]["shaper_commit"]),
            Path(temporary) / "ShapeR",
            checkpoint_names=tuple(pins["checkpoint_files"]),
        )
        command = build_docker_command(
            farm_source=authority.project_root,
            shaper_source=shaper_source,
            checkpoint_files=pins["checkpoint_files"],
            hf_cache=pins["hf_cache"],
            inputs=inputs,
            output=output,
            config_path=authoritative_config,
            profile=profile,
            gpu=str(args.gpu),
            config=config,
            pins=pins,
        )
        if args.print_command or args.plan_only:
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            print(shlex.join(command))
        if args.plan_only:
            return 0
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return int(completed.returncode)

    result, marker_name, marker = validate_success_result(
        output, result_name="result.json", expected_schema="farm.shaper-batch.result.v1"
    )
    if marker.get("runtime_pin_sha256") != pins["runtime_pin_sha256"]:
        raise RuntimeError("ShapeR container result is not bound to the host-validated runtime pin")
    if bool(result.get("release_eligible")) != (marker_name == "_SUCCESS.json"):
        raise RuntimeError("ShapeR output release eligibility/marker mismatch")
    print(json.dumps({
        "status": "success",
        "marker": str(output / marker_name),
        "release_eligible": bool(result.get("release_eligible")),
        "passed_meshes": int(result["objects"]["passed"]),
        "quality_status": result["quality_status"],
    }, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
