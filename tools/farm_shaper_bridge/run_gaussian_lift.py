#!/usr/bin/env python3
"""Run the exact FARM-to-Gaussian lift in the prep image pinned by a FARM run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PREP_CHECK = "runtime:prep"
CANONICAL_PREP_PYTHON = "/opt/conda/envs/rest3d/bin/python"
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GPU_RE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_LIFT_SOURCE = (
    "tools/farm_shaper_bridge/common.py",
    "tools/farm_shaper_bridge/gaussian_lift.py",
    "tools/farm_shaper_bridge/lift_refinement.py",
    "configs/gaussian_lift.v1.yaml",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed FARM run directory")
    parser.add_argument("--ply", type=Path, required=True, help="Immutable source 3DGS PLY")
    parser.add_argument("--output", type=Path, required=True, help="New/empty lift output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "gaussian_lift.v1.yaml",
    )
    parser.add_argument("--gpu", default="0", help="Docker GPU device list, e.g. 0 or 0,1")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke-object-id", type=int, action="append", default=[])
    parser.add_argument("--allow-legacy-run", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_prep_runtime(
    resource_preflight: Mapping[str, Any],
    *,
    allow_legacy_interpreter_fallback: bool = False,
) -> tuple[str, str, str, bool]:
    matches = [
        row for row in resource_preflight.get("checks", [])
        if isinstance(row, Mapping) and str(row.get("name")) == PREP_CHECK
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {PREP_CHECK!r} preflight check")
    row = matches[0]
    if row.get("status") != "pass":
        raise ValueError(f"{PREP_CHECK} did not pass in the source FARM run")
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{PREP_CHECK} metrics are absent")
    tag = str(metrics.get("image") or "")
    expected = str(metrics.get("expected_image_id") or "")
    observed = str(metrics.get("actual_image_id") or "")
    if not tag or not IMAGE_ID_RE.fullmatch(expected) or observed != expected:
        raise ValueError("FARM run does not pin one matching prep image ID")
    recorded_python = metrics.get("python")
    fallback = recorded_python is None or recorded_python == ""
    if fallback:
        if not allow_legacy_interpreter_fallback:
            raise ValueError("FARM run does not pin the prep interpreter")
        python = CANONICAL_PREP_PYTHON
    else:
        python = str(recorded_python)
        if python != CANONICAL_PREP_PYTHON:
            raise ValueError("FARM run pins an unsupported prep interpreter")
    return tag, expected, python, fallback


def _git_snapshot(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("cannot resolve a versioned FARM source commit")
    return commit, bool(status.strip())


def _docker_image_id(tag: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _mount(source: Path, target: str, *, readonly: bool) -> str:
    value = f"type=bind,src={source},dst={target}"
    return value + (",readonly" if readonly else "")


def _execution_source(
    run: Path,
    live_repo: Path,
    *,
    allow_legacy: bool,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a fully validated immutable run source tree.

    Legacy runs predate source snapshots and remain explicitly non-release. All
    current runs execute the lift from the exact tree hashed into their run
    manifest; a clean live checkout is only the launcher, never the computation
    source.
    """

    manifest = _read_json(run / "manifest.json")
    if manifest.get("schema") != "farm.pipeline-run.v1":
        if allow_legacy and manifest.get("source_snapshot") is None:
            return live_repo, {
                "kind": "legacy_live_checkout",
                "commit": None,
                "tree_sha256": None,
                "validated": False,
            }
        raise ValueError("unsupported FARM run manifest")
    if str(manifest.get("status") or "").lower() != "success":
        raise ValueError("Gaussian lift requires a successful FARM run manifest")
    reference = manifest.get("source_snapshot")
    if reference is None:
        if not allow_legacy:
            raise ValueError("canonical Gaussian lift requires a signed run source snapshot")
        return live_repo, {
            "kind": "legacy_live_checkout",
            "commit": None,
            "tree_sha256": None,
            "validated": False,
        }

    source_path = live_repo / "src"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from farm_runtime.source_snapshot import (  # pylint: disable=import-outside-toplevel
        SourceSnapshotIntegrityError,
        validated_source_snapshot_project_root,
    )
    try:
        execution_repo = validated_source_snapshot_project_root(
            run,
            manifest,
            required=True,
        )
    except SourceSnapshotIntegrityError as exc:
        raise RuntimeError(f"invalid FARM signed source snapshot: {exc}") from exc
    if execution_repo is None:
        raise RuntimeError("validated FARM source snapshot is absent")
    missing = [name for name in REQUIRED_LIFT_SOURCE if not (execution_repo / name).is_file()]
    if missing:
        raise ValueError(f"signed FARM source snapshot omits Gaussian lift code: {missing}")
    git = manifest.get("git")
    commit = str(git.get("commit") or "") if isinstance(git, Mapping) else ""
    dirty = git.get("dirty") if isinstance(git, Mapping) else None
    tree = str(reference.get("tree_sha256") or "") if isinstance(reference, Mapping) else ""
    if not COMMIT_RE.fullmatch(commit) or dirty is not False or not IMAGE_ID_RE.fullmatch(f"sha256:{tree}"):
        raise ValueError("signed FARM source snapshot lacks a clean commit/tree identity")
    return execution_repo, {
        "kind": "farm_signed_source_snapshot",
        "commit": commit,
        "tree_sha256": tree,
        "validated": True,
    }


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    repo = Path(__file__).resolve().parents[2]
    run = args.run.expanduser().resolve(strict=True)
    ply = args.ply.expanduser().resolve(strict=True)
    config = args.config.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if not run.is_dir() or not ply.is_file() or not config.is_file():
        raise ValueError("--run must be a directory; --ply and --config must be files")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run.name):
        raise ValueError("FARM run basename is unsafe for the identity-preserving mount")
    if not GPU_RE.fullmatch(str(args.gpu)):
        raise ValueError("--gpu must be a comma-separated list of non-negative device indices")
    if _inside(output, run):
        raise ValueError("output must not be inside the immutable FARM run")
    if _inside(output, repo):
        raise ValueError("output must not be inside the versioned source repository")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing non-empty output path: {output}")
    if output.parent == Path("/"):
        raise ValueError("refusing to mount filesystem root as the output parent")

    resource_path = run / "input" / "resource_preflight.json"
    resource = _read_json(resource_path)
    image_tag, expected_image_id, prep_python, interpreter_fallback = extract_prep_runtime(
        resource,
        allow_legacy_interpreter_fallback=bool(args.allow_legacy_run),
    )
    installed_image_id = _docker_image_id(image_tag)
    if installed_image_id != expected_image_id:
        raise RuntimeError(
            f"prep image drift: run pins {expected_image_id}, Docker resolves {installed_image_id}"
        )

    launcher_commit, launcher_dirty = _git_snapshot(repo)
    smoke = bool(args.smoke_object_id)
    if launcher_dirty and not args.plan_only and not smoke:
        raise RuntimeError(
            "canonical lift requires a clean committed launcher checkout; "
            "use --plan-only or --smoke-object-id while developing"
        )
    execution_repo, source = _execution_source(
        run,
        repo,
        allow_legacy=bool(args.allow_legacy_run),
    )
    if not source["validated"]:
        source["commit"] = launcher_commit
        source["dirty"] = launcher_dirty
    else:
        source["dirty"] = False
        authoritative_config = execution_repo / "configs" / "gaussian_lift.v1.yaml"
        if _sha256_file(config) != _sha256_file(authoritative_config):
            raise ValueError(
                "--config differs byte-for-byte from the signed FARM source snapshot"
            )
    config_sha256 = _sha256_file(config)
    container_run = f"/farm/runs/{run.name}"
    if not args.plan_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=True)

    command = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "4096",
        "--shm-size", "8g",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=8g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", "/opt/farm-src",
        "--env", "HOME=/tmp",
        "--env", "XDG_CACHE_HOME=/tmp/cache",
        "--env", "MPLCONFIGDIR=/tmp/matplotlib",
        "--env", f"FARM_BRIDGE_SOURCE_COMMIT={source['commit'] or ''}",
        "--env", f"FARM_BRIDGE_SOURCE_DIRTY={'true' if source['dirty'] else 'false'}",
        "--env", f"FARM_BRIDGE_SOURCE_TREE_SHA256={source['tree_sha256'] or ''}",
        "--env", f"FARM_BRIDGE_IMAGE={image_tag}",
        "--env", f"FARM_BRIDGE_IMAGE_ID={expected_image_id}",
        "--env", (
            "FARM_BRIDGE_RUNTIME_INTERPRETER_FALLBACK="
            + ("true" if interpreter_fallback else "false")
        ),
        "--env", f"FARM_BRIDGE_CONFIG_SHA256={config_sha256}",
        "--mount", _mount(execution_repo, "/opt/farm-src", readonly=True),
        "--mount", _mount(run, container_run, readonly=True),
        "--mount", _mount(ply, "/farm/source.ply", readonly=True),
        "--mount", _mount(config, "/farm/config.yaml", readonly=True),
    ]
    if not args.plan_only:
        command.extend([
            "--gpus", f"device={args.gpu}",
            "--mount", _mount(output, "/farm/output", readonly=False),
        ])
    command.extend([
        "--entrypoint", prep_python,
        expected_image_id,
        "/opt/farm-src/tools/farm_shaper_bridge/gaussian_lift.py",
        "--run", container_run,
        "--ply", "/farm/source.ply",
        "--config", "/farm/config.yaml",
        "--output", "/farm/output",
    ])
    if args.plan_only:
        command.append("--plan-only")
    if args.allow_legacy_run:
        command.append("--allow-legacy-run")
    for object_id in args.smoke_object_id:
        command.extend(["--smoke-object-id", str(int(object_id))])

    metadata = {
        "repo": str(repo),
        "launcher_commit": launcher_commit,
        "launcher_dirty": launcher_dirty,
        "execution_source": source,
        "execution_repo": str(execution_repo),
        "run": str(run),
        "container_run": container_run,
        "ply": str(ply),
        "config": str(config),
        "config_sha256": config_sha256,
        "output": str(output),
        "image": image_tag,
        "image_id": expected_image_id,
        "python": prep_python,
        "runtime_interpreter_fallback": interpreter_fallback,
        "nonrelease_warnings": (
            ["legacy_resource_preflight_missing_prep_interpreter"]
            if interpreter_fallback else []
        ),
        "plan_only": bool(args.plan_only),
        "smoke_object_ids": sorted(set(int(value) for value in args.smoke_object_id)),
        "legacy": bool(args.allow_legacy_run),
    }
    return command, metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command, metadata = build_command(args)
    if args.print_command:
        print(json.dumps(metadata, sort_keys=True, indent=2))
        print(shlex.join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return int(completed.returncode)
    if args.plan_only:
        return 0

    output = args.output.expanduser().resolve()
    marker_paths = [
        output / name
        for name in (
            "_SUCCESS.json",
            "_SMOKE_SUCCESS.json",
            "_LEGACY_SUCCESS.json",
            "_NONRELEASE_SUCCESS.json",
        )
        if (output / name).is_file()
    ]
    if len(marker_paths) != 1:
        print(
            "ERROR: container exited zero without exactly one completion marker: "
            f"{marker_paths}",
            file=sys.stderr,
        )
        return 2
    marker_path = marker_paths[0]
    payload = _read_json(marker_path)
    if payload.get("status") != "success":
        print(f"ERROR: invalid success marker {marker_path}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "success",
        "marker": str(marker_path),
        "release_eligible": bool(payload.get("release_eligible")),
        "result": str(output / str(payload.get("result"))),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
