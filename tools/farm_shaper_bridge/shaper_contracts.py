#!/usr/bin/env python3
"""Fail-closed contracts for the versioned FARM -> ShapeR bridge.

This module deliberately understands only the canonical verified CSR emitted by
``gaussian_lift.py``.  It never reconstructs membership from dense labels and
never changes the FARM run, Gaussian lift, or source PLY.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

import sys

_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from farm_runtime.source_snapshot import (  # noqa: E402
    SourceSnapshotIntegrityError,
    validated_source_snapshot_project_root,
)

from tools.farm_shaper_bridge.common import (  # noqa: E402
    RunData,
    atomic_json,
    sha256_file,
)


SHAPER_CONFIG_SCHEMA = "farm.shaper-bridge.config.v1"
SHAPER_INPUT_SCHEMA = "farm.shaper-inputs.result.v1"
SHAPER_BATCH_SCHEMA = "farm.shaper-batch.result.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def safe_name(value: str, *, fallback: str = "scene") -> str:
    result = SAFE_NAME_RE.sub("_", str(value).lower()).strip("_")
    return result or fallback


def load_shaper_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve(strict=True)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SHAPER_CONFIG_SCHEMA:
        raise ValueError(f"unsupported ShapeR bridge config: {path}")
    for section in ("inputs", "views", "shape", "runtime"):
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"ShapeR bridge config misses section {section!r}")
    runtime = payload["runtime"]
    image_id = str(runtime.get("image_id") or "")
    commit = str(runtime.get("shaper_commit") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("runtime.image_id must be one immutable Docker image ID")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("runtime.shaper_commit must be one full Git commit")
    checkpoints = runtime.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("runtime.checkpoints must pin at least one file")
    for name, record in checkpoints.items():
        if Path(str(name)).name != str(name) or not isinstance(record, Mapping):
            raise ValueError("checkpoint pin names must be safe basenames")
        digest = str(record.get("sha256") or "")
        if not SHA256_RE.fullmatch(digest) or int(record.get("bytes", -1)) <= 0:
            raise ValueError(f"invalid checkpoint pin for {name!r}")
    recipe = runtime.get("build_recipe")
    if not isinstance(recipe, Mapping) or not all(
        str(recipe.get(key) or "")
        for key in ("dockerfile", "cuda_base_image", "python_version", "torch_cuda_arch_list")
    ):
        raise ValueError("runtime.build_recipe must describe the versioned Docker build")
    expected_inventory = runtime.get("expected_inventory")
    if not isinstance(expected_inventory, Mapping) or not isinstance(
        expected_inventory.get("packages"), Mapping
    ):
        raise ValueError("runtime.expected_inventory.packages must be pinned")
    if not str(runtime.get("checkpoint_repository") or "") or not re.fullmatch(
        r"[0-9a-f]{40}", str(runtime.get("checkpoint_revision") or "")
    ):
        raise ValueError("runtime checkpoint repository/revision must be pinned")
    models = runtime.get("required_huggingface_models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("runtime.required_huggingface_models must pin offline snapshots")
    for cache_name, model in models.items():
        if Path(str(cache_name)).name != str(cache_name) or not isinstance(model, Mapping):
            raise ValueError("invalid Hugging Face cache model pin")
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision") or "")):
            raise ValueError(f"invalid Hugging Face revision for {cache_name!r}")
        files = model.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ValueError(f"Hugging Face model {cache_name!r} has no file pins")
        for filename, record in files.items():
            if Path(str(filename)).name != str(filename) or not isinstance(record, Mapping):
                raise ValueError(f"invalid Hugging Face file pin for {cache_name!r}")
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(record.get("blob") or "")):
                raise ValueError(f"invalid Hugging Face blob pin for {cache_name!r}/{filename}")
            if int(record.get("bytes", -1)) < 0:
                raise ValueError(f"invalid Hugging Face size pin for {cache_name!r}/{filename}")
    return payload, sha256_file(path)


@dataclass(frozen=True)
class FarmSourceAuthority:
    project_root: Path
    signed: bool
    tree_sha256: str | None
    manifest_path: Path | None
    fallback: dict[str, Any] | None = None


def _required_source_evidence(
    project_root: Path,
    required_files: Sequence[str],
) -> dict[str, Any]:
    records: dict[str, str] = {}
    for relative in required_files:
        path = (project_root / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(project_root):
            raise ValueError(f"required live bridge source is unavailable: {relative}")
        records[str(relative)] = sha256_file(path)
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "project_root": str(project_root),
        "required_files_sha256": records,
        "required_files_bundle_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def resolve_farm_source_authority(
    run_dir: Path,
    live_project_root: Path,
    *,
    required_files: Sequence[str],
    allow_unsigned: bool,
    allow_incomplete_snapshot_fallback: bool = False,
) -> FarmSourceAuthority:
    """Resolve and validate the FARM source authority for a downstream stage.

    An advertised run snapshot is always authoritative and must contain every
    requested bridge file.  A legacy run without a snapshot can use the live
    tree only behind an explicit non-release override.
    """

    run_dir = run_dir.expanduser().resolve(strict=True)
    live_project_root = live_project_root.expanduser().resolve(strict=True)
    try:
        snapshot = validated_source_snapshot_project_root(run_dir, required=False)
    except SourceSnapshotIntegrityError:
        raise
    if snapshot is None:
        if not allow_unsigned:
            raise SourceSnapshotIntegrityError(
                "FARM run has no signed source snapshot; pass --allow-unsigned-source "
                "only for an explicitly non-release legacy bridge"
            )
        _required_source_evidence(live_project_root, required_files)
        return FarmSourceAuthority(live_project_root, False, None, None)

    run_manifest = load_json(run_dir / "manifest.json", label="FARM run manifest")
    reference = run_manifest.get("source_snapshot")
    if not isinstance(reference, Mapping):
        raise SourceSnapshotIntegrityError("validated source snapshot metadata disappeared")
    tree_sha256 = str(reference.get("tree_sha256") or "")
    if not SHA256_RE.fullmatch(tree_sha256):
        raise SourceSnapshotIntegrityError("signed source snapshot tree digest is invalid")
    manifest_relative = str(reference.get("manifest_relative_path") or "")
    manifest_path = (run_dir / manifest_relative).resolve(strict=True)
    missing: list[str] = []
    for relative in required_files:
        try:
            candidate = (snapshot / relative).resolve(strict=True)
        except FileNotFoundError:
            missing.append(str(relative))
            continue
        if not candidate.is_file() or not candidate.is_relative_to(snapshot):
            missing.append(str(relative))
    if not missing:
        return FarmSourceAuthority(snapshot, True, tree_sha256, manifest_path)
    if not (allow_unsigned and allow_incomplete_snapshot_fallback):
        raise SourceSnapshotIntegrityError(
            "signed FARM source snapshot predates required bridge files: "
            + ", ".join(missing)
        )
    live_evidence = _required_source_evidence(live_project_root, required_files)
    fallback = {
        "warning": "legacy_unsigned_fallback_from_incomplete_signed_snapshot",
        "nonrelease_only": True,
        "missing_required_files": missing,
        "incomplete_snapshot_project_root": str(snapshot),
        "incomplete_snapshot_tree_sha256": tree_sha256,
        "incomplete_snapshot_manifest": str(manifest_path),
        "incomplete_snapshot_manifest_sha256": sha256_file(manifest_path),
        "selected_live_source": live_evidence,
    }
    return FarmSourceAuthority(live_project_root, False, None, None, fallback)


def require_current_sources_match_authority(
    authority: FarmSourceAuthority,
    live_project_root: Path,
    relative_files: Sequence[str],
) -> None:
    """Fail if a host-invoked bridge script differs from the signed copy."""

    if not authority.signed:
        return
    live_project_root = live_project_root.resolve(strict=True)
    mismatches: list[str] = []
    for relative in relative_files:
        live = (live_project_root / relative).resolve(strict=True)
        signed = (authority.project_root / relative).resolve(strict=True)
        if sha256_file(live) != sha256_file(signed):
            mismatches.append(relative)
    if mismatches:
        raise SourceSnapshotIntegrityError(
            "host bridge source differs from the run's signed snapshot: "
            + ", ".join(mismatches)
        )


@dataclass(frozen=True)
class VerifiedInstanceBank:
    object_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    confidence: np.ndarray
    timestamp_support: np.ndarray
    path: Path
    sha256: str

    def object_slice(self, row: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        if row < 0 or row >= len(self.object_ids):
            raise IndexError(row)
        start, end = int(self.indptr[row]), int(self.indptr[row + 1])
        return (
            int(self.object_ids[row]),
            self.indices[start:end],
            self.confidence[start:end],
            self.timestamp_support[start:end],
        )


def _safe_artifact_path(root: Path, record: Mapping[str, Any], expected_name: str) -> Path:
    relative = str(record.get("path") or "")
    if relative != expected_name or Path(relative).name != relative:
        raise ValueError(f"lift artifact path must be exactly {expected_name!r}")
    path = (root / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"lift artifact escapes its immutable directory: {relative}")
    expected_bytes = int(record.get("bytes", -1))
    expected_sha = str(record.get("sha256") or "")
    if path.stat().st_size != expected_bytes or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError(f"lift artifact metadata is invalid: {relative}")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"lift artifact digest mismatch: {relative}")
    return path


def load_verified_instance_bank(
    path: Path,
    *,
    source_gaussian_count: int,
    expected_sha256: str,
    expected_objects: int | None = None,
    expected_gaussians: int | None = None,
) -> VerifiedInstanceBank:
    """Load and validate the sparse verified-only bank in O(N + M), not O(K*N)."""

    path = path.expanduser().resolve(strict=True)
    if sha256_file(path) != expected_sha256:
        raise ValueError("verified_instance_bank.npz SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {
            "object_ids", "indptr", "indices", "confidence", "timestamp_support"
        }
        if set(archive.files) != expected_keys:
            raise ValueError("verified CSR fields differ from the v1 contract")
        object_ids = np.asarray(archive["object_ids"])
        indptr = np.asarray(archive["indptr"])
        indices = np.asarray(archive["indices"])
        confidence = np.asarray(archive["confidence"])
        support = np.asarray(archive["timestamp_support"])

    if (
        object_ids.dtype != np.dtype("int32")
        or indptr.dtype != np.dtype("int64")
        or indices.dtype != np.dtype("int64")
        or confidence.dtype != np.dtype("float32")
        or support.dtype != np.dtype("uint16")
    ):
        raise ValueError("verified CSR dtypes differ from the v1 contract")
    if any(value.ndim != 1 for value in (object_ids, indptr, indices, confidence, support)):
        raise ValueError("verified CSR arrays must all be one-dimensional")
    if indptr.shape != (len(object_ids) + 1,) or len(indptr) == 0:
        raise ValueError("verified CSR indptr shape mismatch")
    if int(indptr[0]) != 0 or int(indptr[-1]) != len(indices) or np.any(np.diff(indptr) <= 0):
        raise ValueError("verified CSR indptr must contain one non-empty slice per object")
    if confidence.shape != indices.shape or support.shape != indices.shape:
        raise ValueError("verified CSR value arrays do not match indices")
    if len(object_ids) and (object_ids[0] < 0 or np.any(np.diff(object_ids) <= 0)):
        raise ValueError("verified CSR object IDs must be sorted, unique, and non-negative")
    if np.any((indices < 0) | (indices >= int(source_gaussian_count))):
        raise ValueError("verified CSR contains source indices outside the PLY")
    for start, end in zip(indptr[:-1], indptr[1:]):
        segment = indices[int(start):int(end)]
        if len(segment) > 1 and np.any(np.diff(segment) <= 0):
            raise ValueError("each verified CSR object slice must be strictly source-order sorted")
    if len(indices) and len(np.unique(indices)) != len(indices):
        raise ValueError("one source Gaussian occurs in multiple verified objects")
    if not np.isfinite(confidence).all() or np.any((confidence <= 0) | (confidence > 1)):
        raise ValueError("verified CSR confidence must be finite in (0,1]")
    if np.any(support == 0):
        raise ValueError("verified CSR timestamp support must be positive")
    if expected_objects is not None and len(object_ids) != int(expected_objects):
        raise ValueError("verified CSR object count differs from lift result")
    if expected_gaussians is not None and len(indices) != int(expected_gaussians):
        raise ValueError("verified CSR Gaussian count differs from lift result")
    return VerifiedInstanceBank(
        object_ids=object_ids,
        indptr=indptr,
        indices=indices,
        confidence=confidence,
        timestamp_support=support,
        path=path,
        sha256=expected_sha256,
    )


@dataclass(frozen=True)
class LiftBinding:
    directory: Path
    result: Mapping[str, Any]
    result_sha256: str
    marker: Mapping[str, Any]
    marker_name: str
    bank: VerifiedInstanceBank
    release_eligible: bool


def bind_verified_lift(
    lift_dir: Path,
    *,
    run: RunData,
    source_ply_path: Path,
    source_gaussian_count: int,
    source_ply_sha256: str,
    allow_nonrelease: bool,
) -> LiftBinding:
    lift_dir = lift_dir.expanduser().resolve(strict=True)
    marker_names = (
        "_SUCCESS.json", "_NONRELEASE_SUCCESS.json", "_LEGACY_SUCCESS.json",
        "_SMOKE_SUCCESS.json",
    )
    markers = [name for name in marker_names if (lift_dir / name).is_file()]
    if len(markers) != 1:
        raise ValueError("lift directory must contain exactly one success marker")
    marker_name = markers[0]
    if marker_name != "_SUCCESS.json" and not allow_nonrelease:
        raise ValueError("canonical ShapeR inputs require a release-eligible Gaussian lift")
    marker = load_json(lift_dir / marker_name, label="Gaussian lift marker")
    if marker.get("status") != "success":
        raise ValueError("Gaussian lift marker is not successful")
    result_path = lift_dir / "result.json"
    result_sha = sha256_file(result_path)
    if marker.get("result") != "result.json" or marker.get("result_sha256") != result_sha:
        raise ValueError("Gaussian lift marker/result binding mismatch")
    result = load_json(result_path, label="Gaussian lift result")
    if result.get("schema_version") != "farm.gaussian-lift.result.v1" or result.get("status") != "PASS":
        raise ValueError("unsupported or unsuccessful Gaussian lift result")
    if result.get("scene_id") != run.scene_id or marker.get("scene_id") != run.scene_id:
        raise ValueError("FARM/lift scene_id mismatch")
    inputs = result.get("inputs")
    contracts = result.get("contracts")
    counts = result.get("counts")
    if not all(isinstance(value, Mapping) for value in (inputs, contracts, counts)):
        raise ValueError("Gaussian lift result contract is incomplete")
    if (
        inputs.get("farm_success_sha256") != run.success_sha256
        or inputs.get("farm_acceptance_sha256") != run.acceptance_sha256
        or int(inputs.get("source_gaussian_count", -1)) != int(source_gaussian_count)
        or inputs.get("source_ply_sha256") != source_ply_sha256
        or marker.get("source_ply_sha256") != source_ply_sha256
    ):
        raise ValueError("Gaussian lift is not bound to this exact FARM run/source PLY")
    if not bool(contracts.get("verified_only_canonical_labels")):
        raise ValueError("Gaussian lift does not promise verified-only canonical labels")
    if int(contracts.get("unknown_instance_id", -999)) != -1:
        raise ValueError("Gaussian lift unknown-ID contract changed")
    final = ((result.get("artifacts") or {}).get("final") or {})
    bank_record = final.get("verified_instance_bank")
    if not isinstance(bank_record, Mapping):
        raise ValueError("Gaussian lift verified CSR artifact is absent")
    bank_path = _safe_artifact_path(lift_dir, bank_record, "verified_instance_bank.npz")
    bank = load_verified_instance_bank(
        bank_path,
        source_gaussian_count=source_gaussian_count,
        expected_sha256=str(bank_record["sha256"]),
        expected_objects=int(counts.get("verified_objects", -1)),
        expected_gaussians=int(counts.get("verified_gaussians", -1)),
    )
    qc_ids = {int(value) for value in ((result.get("qc_summary") or {}).get("verified_object_ids") or [])}
    if not set(bank.object_ids.tolist()).issubset(qc_ids):
        raise ValueError("verified CSR contains an object not verified by held-out QC")
    release = bool(result.get("release_eligible")) and marker_name == "_SUCCESS.json"
    if release != bool(marker.get("release_eligible")):
        raise ValueError("Gaussian lift release eligibility is inconsistent")
    if not release and not allow_nonrelease:
        raise ValueError("Gaussian lift is non-release")
    return LiftBinding(lift_dir, result, result_sha, marker, marker_name, bank, release)


def artifact_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    relative = path.name if root is None else path.relative_to(root.resolve()).as_posix()
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(dict(payload), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def validate_success_result(
    directory: Path,
    *,
    result_name: str,
    expected_schema: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    directory = directory.expanduser().resolve(strict=True)
    markers = [name for name in ("_SUCCESS.json", "_NONRELEASE_SUCCESS.json") if (directory / name).is_file()]
    if len(markers) != 1:
        raise ValueError(f"expected exactly one completion marker in {directory}")
    marker = load_json(directory / markers[0], label="completion marker")
    result_path = directory / result_name
    digest = sha256_file(result_path)
    if marker.get("status") != "success" or marker.get("result") != result_name or marker.get("result_sha256") != digest:
        raise ValueError("completion marker/result digest mismatch")
    result = load_json(result_path, label="stage result")
    if result.get("schema_version") != expected_schema or result.get("status") != "PASS":
        raise ValueError("unsupported or unsuccessful stage result")
    expected_release = markers[0] == "_SUCCESS.json"
    if (
        bool(result.get("release_eligible")) != expected_release
        or bool(marker.get("release_eligible")) != expected_release
    ):
        raise ValueError("completion marker/result release eligibility mismatch")
    return result, markers[0], marker


__all__ = [
    "FarmSourceAuthority",
    "LiftBinding",
    "SHAPER_BATCH_SCHEMA",
    "SHAPER_CONFIG_SCHEMA",
    "SHAPER_INPUT_SCHEMA",
    "VerifiedInstanceBank",
    "artifact_record",
    "atomic_json",
    "atomic_pickle",
    "bind_verified_lift",
    "load_json",
    "load_shaper_config",
    "load_verified_instance_bank",
    "require_current_sources_match_authority",
    "resolve_farm_source_authority",
    "safe_name",
    "validate_success_result",
]
