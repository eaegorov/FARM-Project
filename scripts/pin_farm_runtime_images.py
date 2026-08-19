#!/usr/bin/env python3
"""Audit or atomically pin Docker image IDs in the FARM model manifest.

The configured image reference remains the human-facing/runtime lookup name.
Docker's observed content ID is the execution identity checked by FARM
preflight.  This helper never accepts a caller-supplied digest and never skips
an image: every ``runtimes.*.image`` is inspected before any write occurs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "models" / "farm_models.v1.json"
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ImagePinError(ValueError):
    """Raised when a runtime image cannot be pinned unambiguously."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="FARM farm.models.v1 manifest (default: repository production manifest)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically replace image_id fields after every configured image passes inspection",
    )
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ImagePinError(f"refusing symlink manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ImagePinError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "farm.models.v1":
        raise ImagePinError("manifest schema_version must be 'farm.models.v1'")
    runtimes = payload.get("runtimes")
    if not isinstance(runtimes, dict) or not runtimes:
        raise ImagePinError("manifest runtimes must be a non-empty mapping")
    for name, value in runtimes.items():
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise ImagePinError("every runtime must be a named mapping")
        image = value.get("image")
        image_id = value.get("image_id")
        if not isinstance(image, str) or not image.strip():
            raise ImagePinError(f"runtime {name!r} has no image reference")
        if not isinstance(image_id, str) or IMAGE_ID_RE.fullmatch(image_id) is None:
            raise ImagePinError(f"runtime {name!r} has an invalid configured image_id")
    return payload


def inspect_image_id(image: str) -> str:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ImagePinError(f"cannot inspect Docker image {image!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ImagePinError(
            f"docker image inspect failed for {image!r} with code {completed.returncode}{suffix}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or IMAGE_ID_RE.fullmatch(lines[0]) is None:
        raise ImagePinError(f"Docker returned an invalid image ID for {image!r}")
    return lines[0]


def inspect_runtimes(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtimes = manifest["runtimes"]
    assert isinstance(runtimes, Mapping)
    rows: list[dict[str, Any]] = []
    for name in sorted(runtimes):
        value = runtimes[name]
        assert isinstance(value, Mapping)
        image = str(value["image"])
        configured = str(value["image_id"])
        observed = inspect_image_id(image)
        rows.append(
            {
                "runtime": name,
                "image": image,
                "configured_image_id": configured,
                "observed_image_id": observed,
                "changed": configured != observed,
            }
        )
    return rows


def _only_image_ids_changed(
    before: Mapping[str, Any], after: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> bool:
    expected = copy.deepcopy(before)
    runtimes = expected["runtimes"]
    for row in rows:
        runtimes[str(row["runtime"])]["image_id"] = str(row["observed_image_id"])
    return expected == after


def atomic_write_manifest(
    path: Path, before: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    after = copy.deepcopy(before)
    runtimes = after["runtimes"]
    for row in rows:
        runtimes[str(row["runtime"])]["image_id"] = str(row["observed_image_id"])
    if not _only_image_ids_changed(before, after, rows):
        raise ImagePinError("internal safety check rejected a non-image_id manifest change")

    source_stat = path.stat(follow_symlinks=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, stat.S_IMODE(source_stat.st_mode))
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(after, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

    reloaded = load_manifest(path)
    if not _only_image_ids_changed(before, reloaded, rows):
        raise ImagePinError("written manifest differs outside runtimes.*.image_id")


def build_report(
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    write: bool,
    wrote: bool,
) -> dict[str, Any]:
    changed = [str(row["runtime"]) for row in rows if bool(row["changed"])]
    return {
        "schema": "farm.runtime-image-pins.v1",
        "status": "updated" if wrote else "drift" if changed else "in_sync",
        "mode": "write" if write else "audit",
        "manifest": str(manifest_path),
        "changed_runtimes": changed,
        "runtimes": list(rows),
        "next_step": (
            "review and commit the manifest before a cold FARM run"
            if wrote
            else "rerun with --write, then review and commit before a cold FARM run"
            if changed
            else "manifest already matches every locally installed runtime image"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested = args.manifest.expanduser().absolute()
    try:
        if requested.is_symlink():
            raise ImagePinError(f"refusing symlink manifest: {requested}")
        manifest_path = requested.resolve(strict=True)
        before = load_manifest(manifest_path)
        rows = inspect_runtimes(before)
        changed = any(bool(row["changed"]) for row in rows)
        wrote = bool(args.write and changed)
        if wrote:
            atomic_write_manifest(manifest_path, before, rows)
        report = build_report(manifest_path, rows, write=bool(args.write), wrote=wrote)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if changed and not args.write else 0
    except (ImagePinError, OSError, UnicodeError) as exc:
        print(
            json.dumps(
                {
                    "schema": "farm.runtime-image-pins.v1",
                    "status": "error",
                    "mode": "write" if args.write else "audit",
                    "manifest": str(requested),
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
