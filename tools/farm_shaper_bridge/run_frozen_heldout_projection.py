#!/usr/bin/env python3
"""Launch target-blind frozen heldout Gaussian projection in its pinned image."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from farm_runtime.frozen_heldout_projection import (  # noqa: E402
    validate_renderer_inputs,
)

GPU_RE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--heldout-frames", type=Path, required=True)
    parser.add_argument("--rgbd-render-manifest", type=Path, required=True)
    parser.add_argument("--union-frames", type=Path, required=True)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--maximum-object-channels", type=int, default=15)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-dirty-source-nonrelease", action="store_true")
    parser.add_argument("--allow-rebuilt-image-nonrelease", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def _docker_image_id(image: str) -> str:
    return subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    return commit, dirty


def _mount(source: Path, target: str, *, readonly: bool = True) -> str:
    value = f"type=bind,src={source},dst={target}"
    return value + (",readonly" if readonly else "")


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    repo = Path(__file__).resolve().parents[2]
    paths = {
        name: Path(getattr(args, name)).expanduser().resolve(strict=True)
        for name in (
            "request",
            "heldout_frames",
            "rgbd_render_manifest",
            "union_frames",
            "source_ply",
            "source_labels",
            "candidate_ply",
            "candidate_labels",
        )
    }
    request, _frames, _hashes = validate_renderer_inputs(
        paths["request"],
        paths["heldout_frames"],
        paths["rgbd_render_manifest"],
        paths["union_frames"],
        paths["source_ply"],
        paths["source_labels"],
        paths["candidate_ply"],
        paths["candidate_labels"],
        verify_rgbd_transitive_artifacts=True,
    )
    transitive_contract = request.get("rgbd_transitive_artifact_contract")
    if transitive_contract is not None and not isinstance(transitive_contract, Mapping):
        raise ValueError("projection request RGBD transitive contract is invalid")
    if not GPU_RE.fullmatch(str(args.gpu)):
        raise ValueError("--gpu must be a comma-separated list of device indices")
    if args.maximum_object_channels < 1 or 2 * args.maximum_object_channels + 1 > 32:
        raise ValueError("--maximum-object-channels violates the gsplat channel limit")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output.name):
        raise ValueError("output directory basename is unsafe")
    if any(output == path or output in path.parents for path in paths.values()):
        raise ValueError("output overlaps a frozen renderer input")
    runtime = request["renderer_runtime"]
    image = str(runtime["image"])
    expected_image_id = str(runtime["image_id"])
    actual_image_id = _docker_image_id(image)
    drift = actual_image_id != expected_image_id
    if drift and not args.allow_rebuilt_image_nonrelease:
        raise RuntimeError(
            f"renderer image drift: request pins {expected_image_id}, Docker resolves {actual_image_id}"
        )
    commit, dirty = _git_state(repo)
    if dirty and not args.plan_only and not args.allow_dirty_source_nonrelease:
        raise RuntimeError(
            "renderer source is dirty; use --allow-dirty-source-nonrelease for an experimental run"
        )
    if not args.plan_only:
        output.parent.mkdir(parents=True, exist_ok=True)

    container_paths = {
        "request": "/farm/input/request.json",
        "heldout_frames": "/farm/input/heldout_frames.json",
        "rgbd_render_manifest": "/farm/input/rgbd_render_manifest.json",
        "union_frames": "/farm/input/union_frames.json",
        "source_ply": "/farm/input/source.ply",
        "source_labels": "/farm/input/source_labels" + paths["source_labels"].suffix,
        "candidate_ply": "/farm/input/candidate.ply",
        "candidate_labels": "/farm/input/candidate_labels"
        + paths["candidate_labels"].suffix,
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "4096",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        "/opt/farm-src",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        f"FARM_PROJECTION_IMAGE={image}",
        "--env",
        f"FARM_PROJECTION_IMAGE_ID={actual_image_id}",
        "--env",
        f"FARM_PROJECTION_IMAGE_DRIFT={'true' if drift else 'false'}",
        "--env",
        f"FARM_PROJECTION_SOURCE_COMMIT={commit}",
        "--env",
        f"FARM_PROJECTION_SOURCE_DIRTY={'true' if dirty else 'false'}",
        "--mount",
        _mount(repo, "/opt/farm-src"),
    ]
    for name, source in paths.items():
        command.extend(["--mount", _mount(source, container_paths[name])])
    if not args.plan_only:
        command.extend(
            [
                "--gpus",
                f"device={args.gpu}",
                "--mount",
                _mount(output.parent, "/farm/output", readonly=False),
            ]
        )
    command.extend(
        [
            "--entrypoint",
            str(runtime["python"]),
            actual_image_id,
            "/opt/farm-src/tools/farm_shaper_bridge/frozen_heldout_projection.py",
        ]
    )
    for name in (
        "request",
        "heldout_frames",
        "rgbd_render_manifest",
        "union_frames",
        "source_ply",
        "source_labels",
        "candidate_ply",
        "candidate_labels",
    ):
        command.extend(["--" + name.replace("_", "-"), container_paths[name]])
    command.extend(
        [
            "--output-dir",
            f"/farm/output/{output.name}",
            "--maximum-object-channels",
            str(args.maximum_object_channels),
        ]
    )
    return command, {
        "repo": str(repo),
        "request": str(paths["request"]),
        "output": str(output),
        "image": image,
        "expected_image_id": expected_image_id,
        "actual_image_id": actual_image_id,
        "runtime_image_drift": drift,
        "source_commit": commit,
        "source_dirty": dirty,
        "target_artifacts_mounted": False,
        "rgbd_transitive_artifact_policy": (
            "host-verified-request-bound-unmounted"
            if transitive_contract is not None
            else "legacy-not-applicable"
        ),
        "rgbd_transitive_artifact_count": (
            int(transitive_contract["artifact_count"])
            if transitive_contract is not None
            else 0
        ),
        "rgbd_transitive_artifact_commitment_sha256": (
            str(transitive_contract["entries_sha256"])
            if transitive_contract is not None
            else None
        ),
        "plan_only": bool(args.plan_only),
        "observation_count": len(request["observations"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command, metadata = build_command(args)
    if args.print_command or args.plan_only:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        print(shlex.join(command))
    if args.plan_only:
        return 0
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return int(completed.returncode)
    manifest = args.output_dir.expanduser().resolve() / "manifest.json"
    success = args.output_dir.expanduser().resolve() / "_SUCCESS.json"
    if not manifest.is_file() or not success.is_file():
        raise RuntimeError(
            "projection renderer did not publish an atomic success bundle"
        )
    print(json.dumps({**metadata, "manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
