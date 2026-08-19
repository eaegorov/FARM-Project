#!/usr/bin/env python3
"""Build, validate, and explicitly repin the external ShapeR runtime.

The build recipe is intentionally separate from release authority.  Public apt
and Python indexes do not make bit-identical OCI images, so a newly built image
is never accepted merely because its tag matches.  ``repin`` emits a reviewable
proposal containing the immutable image ID and asset hashes; the operator must
commit that proposal to the bridge config before creating a new signed FARM
source snapshot.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
else:
    ROOT = Path(__file__).resolve().parents[2]

from tools.farm_shaper_bridge.common import sha256_file  # noqa: E402
from tools.farm_shaper_bridge.run_shaper import (  # noqa: E402
    archive_clean_shaper_source,
    build_checkpoint_mount_smoke_command,
    validate_runtime_pins,
)
from tools.farm_shaper_bridge.shaper_contracts import load_shaper_config  # noqa: E402


SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_CONFIG = ROOT / "configs" / "shaper_bridge.v1.yaml"
SMOKE_CODE = (
    "import importlib.metadata as m,json,torch;"
    "import flash_attn,torchsparse,transformers,trimesh;"
    "names=('torch','torchvision','torchaudio','flash-attn','torchsparse',"
    "'transformers','trimesh');"
    "print(json.dumps({'packages':{n:m.version(n) for n in names},"
    "'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()},"
    "sort_keys=True))"
)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser("build", help="print or execute the versioned image build")
    _add_config(build)
    build.add_argument("--image", help="target image tag; defaults to runtime.image")
    build.add_argument("--cuda-base-image", help="override the audited CUDA base tag/digest")
    build.add_argument("--python-version", help="override the recipe Python version")
    build.add_argument("--torch-cuda-arch-list", help="CUDA architectures, e.g. 8.9 or 8.9;9.0")
    build.add_argument("--execute", action="store_true", help="run docker build after printing plan")

    for name, help_text in (
        ("validate", "fail closed against the committed runtime pins"),
        ("repin", "emit a review-only proposal for a newly built runtime"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _add_config(command)
        command.add_argument("--shaper-repo", type=Path, required=True)
        command.add_argument("--hf-cache", type=Path, required=True)
        command.add_argument("--image", help="image tag/reference; defaults to runtime.image")
        command.add_argument(
            "--skip-smoke",
            action="store_true",
            help="inventory assets only; result is not runtime-validated",
        )
    return parser.parse_args(argv)


def _stdout(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.strip()


def build_command(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    image: str | None,
    cuda_base_image: str | None,
    python_version: str | None,
    torch_cuda_arch_list: str | None,
) -> tuple[list[str], dict[str, Any]]:
    runtime = config["runtime"]
    recipe = runtime["build_recipe"]
    dockerfile = (ROOT / str(recipe["dockerfile"])).resolve(strict=True)
    if not dockerfile.is_file() or not dockerfile.is_relative_to(ROOT):
        raise ValueError("runtime.build_recipe.dockerfile escapes the FARM source tree")
    tag = str(image or runtime["image"])
    base = str(cuda_base_image or recipe["cuda_base_image"])
    py_version = str(python_version or recipe["python_version"])
    architectures = str(torch_cuda_arch_list or recipe["torch_cuda_arch_list"])
    if not tag or not base or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", py_version):
        raise ValueError("invalid ShapeR build arguments")
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:[;+ ][0-9]+\.[0-9]+)*", architectures):
        raise ValueError("--torch-cuda-arch-list must contain explicit numeric architectures")
    dockerfile_sha = sha256_file(dockerfile)
    command = [
        "docker", "build",
        "--file", str(dockerfile),
        "--tag", tag,
        "--build-arg", f"CUDA_BASE_IMAGE={base}",
        "--build-arg", f"PYTHON_VERSION={py_version}",
        "--build-arg", f"TORCH_CUDA_ARCH_LIST={architectures}",
        "--label", f"org.splatica.farm.dockerfile-sha256={dockerfile_sha}",
        str(ROOT),
    ]
    plan = {
        "schema": "farm.shaper-runtime-build-plan.v1",
        "status": "PLAN" if image is None else "PLAN_WITH_IMAGE_OVERRIDE",
        "image_tag": tag,
        "dockerfile": str(dockerfile.relative_to(ROOT)),
        "dockerfile_sha256": dockerfile_sha,
        "config": str(config_path),
        "build_args": {
            "CUDA_BASE_IMAGE": base,
            "PYTHON_VERSION": py_version,
            "TORCH_CUDA_ARCH_LIST": architectures,
        },
        "external_assets": {
            "checkpoint_source": {
                "repository": str(runtime["checkpoint_repository"]),
                "revision": str(runtime["checkpoint_revision"]),
            },
            "checkpoints": runtime["checkpoints"],
            "huggingface_models": {
                str(value["repository"]): str(value["revision"])
                for value in runtime["required_huggingface_models"].values()
            },
        },
        "bit_reproducible": False,
        "release_authority": "immutable OCI image ID accepted only after explicit repin",
    }
    return command, plan


def smoke_command(image_id: str, python: str) -> list[str]:
    if not SHA256_RE.fullmatch(image_id):
        raise ValueError("smoke requires an immutable OCI image ID")
    return [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "256",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
        "--entrypoint", python,
        image_id,
        "-c", SMOKE_CODE,
    ]


def smoke_runtime(
    image_id: str,
    *,
    python: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _stdout(smoke_command(image_id, python))
    try:
        inventory = json.loads(raw.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("ShapeR runtime smoke did not return valid JSON") from exc
    packages = inventory.get("packages")
    if not isinstance(packages, Mapping):
        raise RuntimeError("ShapeR runtime smoke omitted package inventory")
    for name, version in expected.get("packages", {}).items():
        if str(packages.get(name)) != str(version):
            raise RuntimeError(
                f"ShapeR runtime package drift for {name}: expected {version}, "
                f"got {packages.get(name)}"
            )
    expected_cuda = str(expected.get("torch_cuda") or "")
    if expected_cuda and str(inventory.get("torch_cuda")) != expected_cuda:
        raise RuntimeError(
            f"ShapeR torch CUDA drift: expected {expected_cuda}, got {inventory.get('torch_cuda')}"
        )
    if inventory.get("cuda_available") is not True:
        raise RuntimeError(
            "ShapeR runtime smoke requires a CUDA-visible GPU; use --skip-smoke "
            "only for an explicit assets-only audit"
        )
    return dict(inventory)


def smoke_checkpoint_mounts(
    *,
    config: Mapping[str, Any],
    shaper_repo: Path,
    pins: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the exact read-only outer-directory/nested-file mount layout."""

    with tempfile.TemporaryDirectory(prefix="farm-shaper-mount-smoke-") as temporary:
        source = archive_clean_shaper_source(
            shaper_repo.expanduser().resolve(strict=True),
            str(config["runtime"]["shaper_commit"]),
            Path(temporary) / "ShapeR",
            checkpoint_names=tuple(pins["checkpoint_files"]),
        )
        raw = _stdout(build_checkpoint_mount_smoke_command(
            image_id=str(pins["image_id"]),
            python=str(config["runtime"]["python"]),
            shaper_source=source,
            checkpoint_files=pins["checkpoint_files"],
        ))
    try:
        result = json.loads(raw.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("ShapeR checkpoint mount smoke did not return valid JSON") from exc
    if result.get("status") != "PASS" or set(result.get("files") or {}) != set(
        pins["checkpoint_files"]
    ):
        raise RuntimeError("ShapeR checkpoint mount smoke returned incomplete evidence")
    return dict(result)


def _inspect_unpinned_runtime(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    shaper_repo: Path,
    hf_cache: Path,
    image: str,
    run_smoke: bool,
) -> dict[str, Any]:
    runtime = config["runtime"]
    image_id = _stdout(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    if not SHA256_RE.fullmatch(image_id):
        raise RuntimeError("Docker did not return an immutable ShapeR image ID")
    repo = shaper_repo.expanduser().resolve(strict=True)
    commit = str(runtime["shaper_commit"])
    _stdout(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"])
    tree = _stdout(["git", "-C", str(repo), "rev-parse", f"{commit}^{{tree}}"])
    if not COMMIT_RE.fullmatch(tree):
        raise RuntimeError("cannot resolve pinned ShapeR source tree")
    checkpoint_root = (repo / "checkpoints").resolve(strict=True)
    checkpoints: dict[str, dict[str, Any]] = {}
    for name in runtime["checkpoints"]:
        path = (checkpoint_root / str(name)).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"missing ShapeR checkpoint for repin: {name}")
        checkpoints[str(name)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    # Reuse the strict cache validator by substituting only values that repin
    # is explicitly allowed to discover.  This does not modify the config.
    candidate = {**config, "runtime": {**runtime, "image": image, "image_id": image_id,
        "checkpoints": checkpoints}}
    validated = validate_runtime_pins(
        config=candidate,
        config_sha256=config_sha,
        shaper_repo=repo,
        hf_cache=hf_cache,
        farm_source_tree=None,
    )
    inventory = None
    mount_smoke = None
    if run_smoke:
        inventory = smoke_runtime(
            image_id,
            python=str(runtime["python"]),
            expected=runtime["expected_inventory"],
        )
        mount_smoke = smoke_checkpoint_mounts(
            config=candidate,
            shaper_repo=repo,
            pins=validated,
        )
    return {
        "image": image,
        "image_id": image_id,
        "shaper_commit": commit,
        "shaper_tree": tree,
        "checkpoint_source": {
            "repository": str(runtime["checkpoint_repository"]),
            "revision": str(runtime["checkpoint_revision"]),
        },
        "checkpoints": checkpoints,
        "huggingface_models": validated["huggingface_models"],
        "smoke": inventory,
        "checkpoint_mount_smoke": mount_smoke,
    }


def execute(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve(strict=True)
    config, config_sha = load_shaper_config(config_path)
    runtime = config["runtime"]
    if args.action == "build":
        command, plan = build_command(
            config=config,
            config_path=config_path,
            image=args.image,
            cuda_base_image=args.cuda_base_image,
            python_version=args.python_version,
            torch_cuda_arch_list=args.torch_cuda_arch_list,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(shlex.join(command))
        if args.execute:
            subprocess.run(command, check=True)
            image = str(args.image or runtime["image"])
            print(json.dumps({
                "status": "BUILT_NOT_PINNED",
                "image": image,
                "image_id": _stdout([
                    "docker", "image", "inspect", "--format", "{{.Id}}", image
                ]),
                "next": "run manage_runtime.py repin and review/commit its proposal",
            }, indent=2, sort_keys=True))
        return 0

    image = str(args.image or runtime["image"])
    if args.action == "validate" and image != str(runtime["image"]):
        raise ValueError("validate does not permit an image override; use repin")
    if args.action == "validate":
        pins = validate_runtime_pins(
            config=config,
            config_sha256=config_sha,
            shaper_repo=args.shaper_repo,
            hf_cache=args.hf_cache,
            farm_source_tree=None,
        )
        inventory = None
        mount_smoke = None
        if not args.skip_smoke:
            inventory = smoke_runtime(
                pins["image_id"],
                python=str(runtime["python"]),
                expected=runtime["expected_inventory"],
            )
            mount_smoke = smoke_checkpoint_mounts(
                config=config,
                shaper_repo=args.shaper_repo,
                pins=pins,
            )
        print(json.dumps({
            "schema": "farm.shaper-runtime-validation.v1",
            "status": "PASS" if inventory is not None else "ASSETS_ONLY_NOT_RUNTIME_VALIDATED",
            "runtime_pin_sha256": pins["runtime_pin_sha256"],
            "image_id": pins["image_id"],
            "shaper_commit": pins["shaper_commit"],
            "shaper_tree": pins["shaper_tree"],
            "checkpoint_source": pins["checkpoint_source"],
            "checkpoints": pins["checkpoints"],
            "huggingface_models": pins["huggingface_models"],
            "smoke": inventory,
            "checkpoint_mount_smoke": mount_smoke,
        }, indent=2, sort_keys=True))
        return 0

    discovered = _inspect_unpinned_runtime(
        config=config,
        config_sha=config_sha,
        shaper_repo=args.shaper_repo,
        hf_cache=args.hf_cache,
        image=image,
        run_smoke=not args.skip_smoke,
    )
    print(json.dumps({
        "schema": "farm.shaper-runtime-repin-proposal.v1",
        "status": "REVIEW_REQUIRED",
        "config": str(config_path),
        "replacement": {
            "runtime.image": discovered["image"],
            "runtime.image_id": discovered["image_id"],
            "runtime.checkpoints": discovered["checkpoints"],
        },
        "evidence": discovered,
        "release_steps": [
            "review the image inventory and checkpoint hashes",
            "update and commit configs/shaper_bridge.v1.yaml",
            "run this command again with action=validate",
            "create a new FARM cold run so its signed source snapshot contains the accepted pins",
        ],
        "warning": "a repin proposal is not release authority and does not modify source",
    }, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
