from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping


DIRECTORY_TREE_SCHEMA = "farm.directory-tree.v1"


class DirectoryIntegrityError(RuntimeError):
    """Raised when an immutable artifact directory is unsafe or has drifted."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def directory_tree_descriptor(root: Path) -> dict[str, Any]:
    """Hash every regular file below *root* without following links.

    The tree identity is independent of inode metadata and traversal order. It
    binds the normalized relative path, byte size and full SHA-256 of every
    file. Symlinks and special files are rejected because their interpretation
    can change outside the immutable run directory.
    """

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise DirectoryIntegrityError(f"artifact directory is missing or a symlink: {root}")
    resolved_root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for current, raw_directories, raw_files in os.walk(
        resolved_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directories: list[str] = []
        for name in sorted(raw_directories):
            child = current_path / name
            if child.is_symlink():
                raise DirectoryIntegrityError(f"artifact tree contains a directory symlink: {child}")
            if not child.is_dir():
                raise DirectoryIntegrityError(f"artifact tree contains a special entry: {child}")
            directories.append(name)
        raw_directories[:] = directories
        for name in sorted(raw_files):
            path = current_path / name
            if path.is_symlink():
                raise DirectoryIntegrityError(f"artifact tree contains a file symlink: {path}")
            before = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise DirectoryIntegrityError(f"artifact tree contains a non-regular file: {path}")
            try:
                relative = path.relative_to(resolved_root).as_posix()
            except ValueError as exc:  # pragma: no cover - defensive; os.walk is rooted.
                raise DirectoryIntegrityError(f"artifact file escapes its tree: {path}") from exc
            digest = sha256_file(path)
            after = path.stat(follow_symlinks=False)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_before != identity_after:
                raise DirectoryIntegrityError(f"artifact file changed while hashing: {path}")
            records.append({
                "path": relative,
                "bytes": int(after.st_size),
                "sha256": digest,
            })
    records.sort(key=lambda row: row["path"])
    payload = {"schema": DIRECTORY_TREE_SCHEMA, "files": records}
    return {
        "algorithm": DIRECTORY_TREE_SCHEMA,
        "files": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "tree_sha256": _canonical_sha256(payload),
    }


def verify_directory_tree_descriptor(
    root: Path, descriptor: Mapping[str, Any]
) -> dict[str, Any]:
    if descriptor.get("kind") != "directory":
        raise DirectoryIntegrityError("artifact descriptor is not a directory")
    expected = {
        "algorithm": descriptor.get("algorithm"),
        "files": descriptor.get("files"),
        "bytes": descriptor.get("bytes"),
        "tree_sha256": descriptor.get("tree_sha256"),
    }
    actual = directory_tree_descriptor(root)
    if expected != actual:
        raise DirectoryIntegrityError(
            f"artifact directory content differs from its immutable descriptor: {root}"
        )
    return actual
