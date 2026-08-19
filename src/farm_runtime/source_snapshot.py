from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from .config import canonical_json


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SNAPSHOT_SCHEMA = "farm.source-snapshot.v1"
_SOURCE_SNAPSHOT_TREE_SCHEMA = "farm.source-snapshot-tree.v1"


class SourceSnapshotIntegrityError(RuntimeError):
    """Raised when a run's first-party source snapshot is absent or altered."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise SourceSnapshotIntegrityError(
            f"Invalid or missing {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise SourceSnapshotIntegrityError(f"Expected a JSON object for {label}: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(files: list[dict[str, Any]]) -> str:
    records = [dict(record) for record in sorted(files, key=lambda row: str(row["path"]))]
    payload = {"schema": _SOURCE_SNAPSHOT_TREE_SCHEMA, "files": records}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _run_relative_path(run_dir: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceSnapshotIntegrityError(
            f"Source snapshot {label} must be a non-empty relative path"
        )
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SourceSnapshotIntegrityError(
            f"Source snapshot {label} must be a normalized relative path"
        )
    unresolved = run_dir / relative
    if unresolved.is_symlink():
        raise SourceSnapshotIntegrityError(
            f"Source snapshot {label} must not be a symlink"
        )
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(run_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SourceSnapshotIntegrityError(
            f"Source snapshot {label} is missing or escapes the run: {value}"
        ) from exc
    return resolved


def validated_source_snapshot_project_root(
    run_dir: Path,
    run_manifest: Mapping[str, Any] | None = None,
    *,
    required: bool = False,
) -> Path | None:
    """Return the validated immutable first-party tree recorded by a run.

    Legacy runs without ``source_snapshot`` metadata are accepted only when
    ``required`` is false. Any advertised snapshot is always checked in full:
    path containment, file set, type, mode, size, executable bit, per-file
    SHA-256, file count, and canonical tree SHA-256.
    """

    try:
        resolved_run = Path(run_dir).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SourceSnapshotIntegrityError(
            f"Run directory is missing or invalid: {run_dir}"
        ) from exc
    if not resolved_run.is_dir():
        raise SourceSnapshotIntegrityError(f"Run path is not a directory: {resolved_run}")

    manifest = (
        _load_json(resolved_run / "manifest.json", "run manifest")
        if run_manifest is None
        else run_manifest
    )
    if not isinstance(manifest, Mapping):
        raise SourceSnapshotIntegrityError("Run manifest must be a mapping")
    reference = manifest.get("source_snapshot")
    if reference is None:
        if required:
            raise SourceSnapshotIntegrityError(
                "Run does not contain the required source_snapshot metadata"
            )
        return None
    if not isinstance(reference, Mapping):
        raise SourceSnapshotIntegrityError("Run source_snapshot metadata must be a mapping")
    if reference.get("schema") != _SOURCE_SNAPSHOT_SCHEMA:
        raise SourceSnapshotIntegrityError("Unsupported run source_snapshot metadata schema")

    snapshot_root = _run_relative_path(
        resolved_run, reference.get("relative_path"), "relative_path"
    )
    if not snapshot_root.is_dir():
        raise SourceSnapshotIntegrityError(
            "Source snapshot project root is not a directory"
        )
    snapshot_manifest_path = _run_relative_path(
        resolved_run,
        reference.get("manifest_relative_path"),
        "manifest_relative_path",
    )
    snapshot = _load_json(snapshot_manifest_path, "source snapshot manifest")
    if snapshot.get("schema") != _SOURCE_SNAPSHOT_SCHEMA:
        raise SourceSnapshotIntegrityError("Unsupported source snapshot manifest schema")
    if (
        snapshot.get("project_relative_path") != "FARM-Project"
        or snapshot.get("tree_schema") != _SOURCE_SNAPSHOT_TREE_SCHEMA
    ):
        raise SourceSnapshotIntegrityError("Source snapshot manifest contract mismatch")

    raw_files = snapshot.get("files")
    if not isinstance(raw_files, list) or not all(
        isinstance(record, Mapping) for record in raw_files
    ):
        raise SourceSnapshotIntegrityError(
            "Source snapshot manifest files must be a list"
        )

    expected_paths: set[str] = set()
    actual_records: list[dict[str, Any]] = []
    for raw_record in raw_files:
        record = dict(raw_record)
        relative = record.get("path")
        expected_sha256 = str(record.get("sha256") or "").lower()
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path is None
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative in expected_paths
            or not _SHA256_RE.fullmatch(expected_sha256)
            or not isinstance(record.get("executable"), bool)
        ):
            raise SourceSnapshotIntegrityError("Invalid source snapshot file record")
        expected_paths.add(relative)
        unresolved = snapshot_root / relative_path
        if unresolved.is_symlink():
            raise SourceSnapshotIntegrityError(
                f"Source snapshot contains a symlink: {relative}"
            )
        try:
            path = unresolved.resolve(strict=True)
            path.relative_to(snapshot_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise SourceSnapshotIntegrityError(
                f"Source snapshot file is missing or escapes its root: {relative}"
            ) from exc
        if not path.is_file():
            raise SourceSnapshotIntegrityError(
                f"Source snapshot entry is not a file: {relative}"
            )
        path_stat = path.stat(follow_symlinks=False)
        try:
            expected_size = int(record["size"])
            expected_mode = int(record["mode"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceSnapshotIntegrityError(
                "Invalid source snapshot size/mode metadata"
            ) from exc
        if expected_size < 0 or not 0 <= expected_mode <= 0o7777:
            raise SourceSnapshotIntegrityError(
                "Invalid source snapshot size/mode metadata"
            )
        mode = int(stat.S_IMODE(path_stat.st_mode))
        executable = bool(path_stat.st_mode & 0o111)
        sha256 = _sha256_file(path)
        if (
            int(path_stat.st_size) != expected_size
            or mode != expected_mode
            or executable is not record["executable"]
            or sha256 != expected_sha256
        ):
            raise SourceSnapshotIntegrityError(
                f"Source snapshot file integrity mismatch: {relative}"
            )
        actual_records.append(
            {
                "path": relative,
                "size": int(path_stat.st_size),
                "mode": mode,
                "executable": executable,
                "sha256": sha256,
            }
        )

    actual_paths: set[str] = set()
    for path in snapshot_root.rglob("*"):
        relative = path.relative_to(snapshot_root).as_posix()
        if path.is_symlink():
            raise SourceSnapshotIntegrityError(
                f"Source snapshot contains a symlink: {relative}"
            )
        if path.is_file():
            actual_paths.add(relative)
        elif not path.is_dir():
            raise SourceSnapshotIntegrityError(
                f"Source snapshot contains a special file: {relative}"
            )
    if actual_paths != expected_paths:
        raise SourceSnapshotIntegrityError(
            "Source snapshot file set does not match its manifest"
        )

    tree_sha256 = _tree_sha256(actual_records)
    manifest_tree = str(snapshot.get("tree_sha256") or "").lower()
    reference_tree = str(reference.get("tree_sha256") or "").lower()
    try:
        manifest_count = int(snapshot.get("file_count"))
        reference_count = int(reference.get("file_count"))
    except (TypeError, ValueError) as exc:
        raise SourceSnapshotIntegrityError(
            "Invalid source snapshot file count metadata"
        ) from exc
    if (
        tree_sha256 != manifest_tree
        or tree_sha256 != reference_tree
        or len(actual_records) != manifest_count
        or len(actual_records) != reference_count
    ):
        raise SourceSnapshotIntegrityError(
            "Source snapshot manifest/root metadata mismatch"
        )
    return snapshot_root
