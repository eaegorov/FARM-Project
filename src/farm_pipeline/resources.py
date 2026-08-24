"""Read-only GPU, model, secret, and runtime validation for FARM.

The module deliberately uses only the Python standard library so it can run
on a host before Torch, CUDA, or a FARM container is started.  It never starts
or stops containers and it never serializes secret values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


MANIFEST_SCHEMA = "farm.models.v1"
REPORT_SCHEMA = "farm.resource_preflight.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class ResourceConfigError(ValueError):
    """Raised when the resource manifest is invalid."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResourceConfigError(f"{name} must be an object/mapping")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourceConfigError(f"{name} must be a non-empty string")
    if any(character in value for character in ("\n", "\r", "\x00")):
        raise ResourceConfigError(f"{name} cannot contain control characters")
    return value.strip()


def _number(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResourceConfigError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ResourceConfigError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ResourceConfigError(f"{name} must be >= {minimum}")
    return result


def _resolve_path(value: Any, base_dir: Path, name: str) -> Path:
    text = _string(value, name)
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _safe_relative_path(value: Any, name: str) -> str:
    text = _string(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ResourceConfigError(f"{name} must be a safe relative POSIX path")
    return path.as_posix()


def _sha256(value: Any, name: str) -> str:
    digest = _string(value, name).lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ResourceConfigError(f"{name} must be a 64-character SHA256 digest")
    return digest


@dataclass(frozen=True)
class FileRequirement:
    path: str
    sha256: Optional[str] = None
    checksum_mode: str = "none"


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    local_path: Path
    repo_id: Optional[str] = None
    revision: Optional[str] = None
    sha256: Optional[str] = None
    required_files: tuple[FileRequirement, ...] = ()


@dataclass(frozen=True)
class MemoryPolicy:
    target_allocated_gib: float
    minimum_total_gib: float
    reserve_free_gib: float
    utilization_min: float
    utilization_max: float


@dataclass(frozen=True)
class RuntimeSpec:
    image: str
    image_id: str
    user_uid: int | None = None
    python: str | None = None


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    model: ModelSpec
    runtime: str
    served_model_name: str
    port: int
    memory: MemoryPolicy
    vllm_args: tuple[str, ...]


@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    manifest_path: Path
    cache_root: Path
    runtimes: dict[str, RuntimeSpec]
    services: dict[str, ServiceSpec]
    pipeline_models: dict[str, ModelSpec]

    def normalized_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema_version": self.schema_version,
            "manifest_path": self.manifest_path,
            "cache_root": self.cache_root,
            "runtimes": {name: asdict(item) for name, item in self.runtimes.items()},
            "services": {name: asdict(item) for name, item in self.services.items()},
            "pipeline_models": {
                name: asdict(item) for name, item in self.pipeline_models.items()
            },
        })


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_free_mib: int
    compute_capability: str
    driver_version: str

    @property
    def total_gib(self) -> float:
        return self.memory_total_mib / 1024.0

    @property
    def free_gib(self) -> float:
        return self.memory_free_mib / 1024.0


@dataclass
class ResourceFinding:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceCheck:
    name: str
    status: str = "pass"
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[ResourceFinding] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, **details: Any) -> None:
        self.findings.append(ResourceFinding(severity, code, message, details))
        if severity == "error":
            self.status = "fail"
        elif severity == "warning" and self.status == "pass":
            self.status = "warn"


@dataclass
class ResourceReport:
    manifest_path: Path
    manifest_sha256: str
    gpu_identifier: str
    gpu_inventory: list[GpuInfo]
    selected_gpu: Optional[GpuInfo]
    services: dict[str, dict[str, Any]]
    checks: list[ResourceCheck]
    started_unix_s: float
    duration_s: float

    @property
    def errors(self) -> int:
        return sum(
            finding.severity == "error"
            for check in self.checks
            for finding in check.findings
        )

    @property
    def warnings(self) -> int:
        return sum(
            finding.severity == "warning"
            for check in self.checks
            for finding in check.findings
        )

    @property
    def status(self) -> str:
        if self.errors:
            return "fail"
        if self.warnings:
            return "warn"
        return "pass"

    @property
    def ready(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema_version": REPORT_SCHEMA,
            "status": self.status,
            "ready": self.ready,
            "strict_ready": self.status == "pass",
            "errors": self.errors,
            "warnings": self.warnings,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "gpu_identifier": self.gpu_identifier,
            "gpu_inventory": [asdict(item) for item in self.gpu_inventory],
            "selected_gpu": asdict(self.selected_gpu) if self.selected_gpu else None,
            "services": self.services,
            "started_unix_s": self.started_unix_s,
            "duration_s": self.duration_s,
            "checks": [asdict(check) for check in self.checks],
        })


def _parse_file_requirement(value: Any, name: str) -> FileRequirement:
    if isinstance(value, str):
        return FileRequirement(path=_safe_relative_path(value, name))
    raw = _mapping(value, name)
    path = _safe_relative_path(raw.get("path"), f"{name}.path")
    digest = _sha256(raw["sha256"], f"{name}.sha256") if raw.get("sha256") else None
    mode = str(raw.get("checksum_mode", "content_addressed" if digest else "none"))
    if mode not in {"none", "full", "content_addressed"}:
        raise ResourceConfigError(
            f"{name}.checksum_mode must be none, full, or content_addressed"
        )
    if mode != "none" and not digest:
        raise ResourceConfigError(f"{name}.checksum_mode={mode} requires sha256")
    return FileRequirement(path=path, sha256=digest, checksum_mode=mode)


def _parse_model(
    value: Any,
    name: str,
    *,
    base_dir: Path,
    cache_root: Path,
) -> ModelSpec:
    raw = _mapping(value, name)
    kind = _string(raw.get("kind"), f"{name}.kind")
    if kind == "local_file":
        path = _resolve_path(raw.get("path"), base_dir, f"{name}.path")
        digest = _sha256(raw["sha256"], f"{name}.sha256") if raw.get("sha256") else None
        requirements_raw = raw.get("required_files", ())
        if isinstance(requirements_raw, str) or not isinstance(requirements_raw, Sequence):
            raise ResourceConfigError(f"{name}.required_files must be a list")
        requirements = tuple(
            _parse_file_requirement(item, f"{name}.required_files[{index}]")
            for index, item in enumerate(requirements_raw)
        )
        return ModelSpec(
            kind=kind,
            local_path=path,
            sha256=digest,
            required_files=requirements,
        )
    if kind != "huggingface_snapshot":
        raise ResourceConfigError(
            f"{name}.kind must be local_file or huggingface_snapshot"
        )
    repo_id = _string(raw.get("repo_id"), f"{name}.repo_id")
    revision = _string(raw.get("revision"), f"{name}.revision").lower()
    if not _REVISION_RE.fullmatch(revision):
        raise ResourceConfigError(f"{name}.revision must be an immutable 40-hex commit")
    if raw.get("local_path"):
        local_path = _resolve_path(raw["local_path"], base_dir, f"{name}.local_path")
    else:
        local_path = (
            cache_root / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        ).resolve(strict=False)
    requirements_raw = raw.get("required_files", ())
    if isinstance(requirements_raw, str) or not isinstance(requirements_raw, Sequence):
        raise ResourceConfigError(f"{name}.required_files must be a list")
    requirements = tuple(
        _parse_file_requirement(item, f"{name}.required_files[{index}]")
        for index, item in enumerate(requirements_raw)
    )
    if not requirements:
        raise ResourceConfigError(f"{name}.required_files cannot be empty")
    return ModelSpec(
        kind=kind,
        local_path=local_path,
        repo_id=repo_id,
        revision=revision,
        required_files=requirements,
    )


def _parse_memory(value: Any, name: str) -> MemoryPolicy:
    raw = _mapping(value, name)
    policy = MemoryPolicy(
        target_allocated_gib=_number(
            raw.get("target_allocated_gib"), f"{name}.target_allocated_gib", minimum=0.1
        ),
        minimum_total_gib=_number(
            raw.get("minimum_total_gib"), f"{name}.minimum_total_gib", minimum=0.1
        ),
        reserve_free_gib=_number(
            raw.get("reserve_free_gib"), f"{name}.reserve_free_gib", minimum=0.0
        ),
        utilization_min=_number(
            raw.get("utilization_min"), f"{name}.utilization_min", minimum=0.01
        ),
        utilization_max=_number(
            raw.get("utilization_max"), f"{name}.utilization_max", minimum=0.01
        ),
    )
    if policy.utilization_min > policy.utilization_max:
        raise ResourceConfigError(f"{name}.utilization_min cannot exceed utilization_max")
    if policy.utilization_max > 0.95:
        raise ResourceConfigError(f"{name}.utilization_max must be <= 0.95")
    return policy


def load_model_manifest(path: Path | str) -> ModelManifest:
    """Load and normalize a versioned model/runtime manifest."""

    manifest_path = Path(path).expanduser().resolve(strict=False)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResourceConfigError(f"invalid JSON in {manifest_path}: {exc}") from exc
    root = _mapping(raw, "manifest")
    version = _string(root.get("schema_version"), "schema_version")
    if version != MANIFEST_SCHEMA:
        raise ResourceConfigError(
            f"unsupported schema_version {version!r}; expected {MANIFEST_SCHEMA!r}"
        )
    base_dir = manifest_path.parent
    cache_root = _resolve_path(root.get("cache_root"), base_dir, "cache_root")

    runtimes: dict[str, RuntimeSpec] = {}
    for name, value in _mapping(root.get("runtimes"), "runtimes").items():
        runtime_name = _string(name, "runtime name")
        if not _NAME_RE.fullmatch(runtime_name):
            raise ResourceConfigError(f"invalid runtime name {runtime_name!r}")
        item = _mapping(value, f"runtimes.{runtime_name}")
        image_id = _string(item.get("image_id"), f"runtimes.{runtime_name}.image_id")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ResourceConfigError(
                f"runtimes.{runtime_name}.image_id must be sha256:<64 hex>"
            )
        user_uid_raw = item.get("user_uid")
        user_uid = None if user_uid_raw is None else int(user_uid_raw)
        if user_uid is not None and user_uid <= 0:
            raise ResourceConfigError(
                f"runtimes.{runtime_name}.user_uid must be a positive non-root UID"
            )
        python_raw = item.get("python")
        python = None if python_raw is None else _string(
            python_raw, f"runtimes.{runtime_name}.python"
        )
        if python is not None:
            python_path = Path(python)
            if (
                not python_path.is_absolute()
                or python_path.as_posix() != python
                or any(part in {"", ".", ".."} for part in python_path.parts)
            ):
                raise ResourceConfigError(
                    f"runtimes.{runtime_name}.python must be a normalized absolute path"
                )
        runtimes[runtime_name] = RuntimeSpec(
            image=_string(item.get("image"), f"runtimes.{runtime_name}.image"),
            image_id=image_id,
            user_uid=user_uid,
            python=python,
        )
    if not runtimes:
        raise ResourceConfigError("runtimes cannot be empty")

    services: dict[str, ServiceSpec] = {}
    for name, value in _mapping(root.get("services"), "services").items():
        service_name = _string(name, "service name")
        if not _NAME_RE.fullmatch(service_name):
            raise ResourceConfigError(f"invalid service name {service_name!r}")
        item = _mapping(value, f"services.{service_name}")
        runtime = _string(item.get("runtime", "main"), f"services.{service_name}.runtime")
        if runtime not in runtimes:
            raise ResourceConfigError(
                f"services.{service_name}.runtime references unknown runtime {runtime!r}"
            )
        try:
            port = int(item.get("port"))
        except (TypeError, ValueError) as exc:
            raise ResourceConfigError(f"services.{service_name}.port must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ResourceConfigError(f"services.{service_name}.port is outside 1..65535")
        args = item.get("vllm_args", ())
        if isinstance(args, str) or not isinstance(args, Sequence):
            raise ResourceConfigError(f"services.{service_name}.vllm_args must be a list")
        parsed_args = tuple(
            _string(argument, f"services.{service_name}.vllm_args[{index}]")
            for index, argument in enumerate(args)
        )
        services[service_name] = ServiceSpec(
            name=service_name,
            model=_parse_model(
                item.get("model"),
                f"services.{service_name}.model",
                base_dir=base_dir,
                cache_root=cache_root,
            ),
            runtime=runtime,
            served_model_name=_string(
                item.get("served_model_name"), f"services.{service_name}.served_model_name"
            ),
            port=port,
            memory=_parse_memory(item.get("memory"), f"services.{service_name}.memory"),
            vllm_args=parsed_args,
        )
    if not services:
        raise ResourceConfigError("services cannot be empty")

    pipeline_models: dict[str, ModelSpec] = {}
    for name, value in _mapping(root.get("pipeline_models", {}), "pipeline_models").items():
        model_name = _string(name, "pipeline model name")
        if not _NAME_RE.fullmatch(model_name):
            raise ResourceConfigError(f"invalid pipeline model name {model_name!r}")
        pipeline_models[model_name] = _parse_model(
            value,
            f"pipeline_models.{model_name}",
            base_dir=base_dir,
            cache_root=cache_root,
        )
    return ModelManifest(
        schema_version=version,
        manifest_path=manifest_path,
        cache_root=cache_root,
        runtimes=runtimes,
        services=services,
        pipeline_models=pipeline_models,
    )


def parse_nvidia_smi_csv(text: str) -> list[GpuInfo]:
    """Parse the stable no-header/nounits query used by :func:`query_gpus`."""

    result: list[GpuInfo] = []
    for line_number, row in enumerate(csv.reader(text.splitlines()), 1):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 7:
            raise ResourceConfigError(
                f"nvidia-smi row {line_number} has {len(row)} fields; expected 7"
            )
        try:
            index = int(row[0].strip())
            total = int(float(row[3].strip()))
            free = int(float(row[4].strip()))
        except ValueError as exc:
            raise ResourceConfigError(f"invalid numeric GPU field on row {line_number}") from exc
        if total <= 0 or free < 0 or free > total:
            raise ResourceConfigError(f"invalid GPU memory values on row {line_number}")
        result.append(GpuInfo(
            index=index,
            uuid=row[1].strip(),
            name=row[2].strip(),
            memory_total_mib=total,
            memory_free_mib=free,
            compute_capability=row[5].strip(),
            driver_version=row[6].strip(),
        ))
    return result


def query_gpus() -> list[GpuInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "nvidia-smi failed"
        raise RuntimeError(message)
    return parse_nvidia_smi_csv(completed.stdout)


def select_gpu(gpus: Sequence[GpuInfo], identifier: str | int) -> GpuInfo:
    requested = str(identifier).strip()
    if not requested:
        raise ResourceConfigError("GPU identifier cannot be empty")
    matches = [gpu for gpu in gpus if str(gpu.index) == requested or gpu.uuid == requested]
    if len(matches) != 1:
        available = [f"{gpu.index}:{gpu.uuid}" for gpu in gpus]
        raise ResourceConfigError(
            f"GPU {requested!r} did not resolve uniquely; available: {available}"
        )
    return matches[0]


def compute_memory_plan(gpu: GpuInfo, policy: MemoryPolicy) -> dict[str, Any]:
    """Calculate a bounded vLLM allocation and its hard readiness gates."""

    desired = policy.target_allocated_gib / gpu.total_gib
    utilization = min(policy.utilization_max, max(policy.utilization_min, desired))
    allocation = utilization * gpu.total_gib
    total_ok = gpu.total_gib + 0.25 >= policy.minimum_total_gib
    target_reachable = policy.utilization_max * gpu.total_gib + 1e-9 >= policy.target_allocated_gib
    free_ok = allocation + policy.reserve_free_gib <= gpu.free_gib + 1e-9
    return {
        "gpu_total_gib": gpu.total_gib,
        "gpu_free_gib": gpu.free_gib,
        "target_allocated_gib": policy.target_allocated_gib,
        "planned_allocated_gib": allocation,
        "reserve_free_gib": policy.reserve_free_gib,
        "gpu_memory_utilization": round(utilization, 4),
        "utilization_min": policy.utilization_min,
        "utilization_max": policy.utilization_max,
        "minimum_total_gib": policy.minimum_total_gib,
        "total_ok": total_ok,
        "target_reachable": target_reachable,
        "free_ok": free_ok,
        "ready": total_ok and target_reachable and free_ok,
    }


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _check_model(
    check: ResourceCheck,
    model: ModelSpec,
    *,
    verify_full_hashes: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "kind": model.kind,
        "local_path": str(model.local_path),
        "repo_id": model.repo_id,
        "revision": model.revision,
    }
    if model.kind == "local_file":
        if not model.local_path.is_file():
            check.add("error", "model.file_missing", "Local model file is missing")
            return metrics
        size = model.local_path.stat().st_size
        metrics["size_bytes"] = size
        if size <= 0:
            check.add("error", "model.file_empty", "Local model file is empty")
        if model.sha256:
            actual = sha256_file(model.local_path)
            metrics["sha256"] = actual
            metrics["sha256_verified"] = actual == model.sha256
            if actual != model.sha256:
                check.add("error", "model.checksum_mismatch", "Local model SHA256 mismatch")
        else:
            check.add("warning", "model.checksum_missing", "Local model has no pinned SHA256")
        sibling_checked = 0
        sibling_verified = 0
        sibling_bytes = 0
        sibling_root = model.local_path.parent.resolve(strict=False)
        for requirement in model.required_files:
            candidate = model.local_path.parent / requirement.path
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError:
                check.add(
                    "error", "model.sibling_file_missing",
                    f"Required local-model sibling is missing: {requirement.path}",
                )
                continue
            if (
                not resolved.is_relative_to(sibling_root)
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                check.add(
                    "error", "model.sibling_not_file",
                    f"Required local-model sibling is not a regular contained file: {requirement.path}",
                )
                continue
            sibling_checked += 1
            sibling_bytes += candidate.stat().st_size
            if candidate.stat().st_size <= 0:
                check.add(
                    "error", "model.sibling_file_empty",
                    f"Required local-model sibling is empty: {requirement.path}",
                )
            if requirement.sha256:
                actual = sha256_file(candidate)
                if actual == requirement.sha256:
                    sibling_verified += 1
                else:
                    check.add(
                        "error", "model.sibling_checksum_mismatch",
                        f"Required local-model sibling SHA256 mismatch: {requirement.path}",
                    )
        metrics["required_sibling_files_checked"] = sibling_checked
        metrics["required_sibling_files_sha256_verified"] = sibling_verified
        metrics["required_sibling_bytes"] = sibling_bytes
        return metrics

    if not model.local_path.is_dir():
        check.add("error", "model.snapshot_missing", "Pinned Hugging Face snapshot is missing")
        return metrics
    checked = 0
    verified = 0
    total_bytes = 0
    root = model.local_path.resolve(strict=False)
    for requirement in model.required_files:
        candidate = model.local_path / requirement.path
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            check.add(
                "error", "model.snapshot_file_missing",
                f"Required snapshot file is missing: {requirement.path}",
            )
            continue
        if not candidate.is_file():
            check.add(
                "error", "model.snapshot_not_file",
                f"Required snapshot entry is not a regular file: {requirement.path}",
            )
            continue
        size = candidate.stat().st_size
        if size <= 0:
            check.add(
                "error", "model.snapshot_file_empty",
                f"Required snapshot file is empty: {requirement.path}",
            )
        total_bytes += size
        checked += 1
        if not requirement.sha256:
            continue
        if requirement.checksum_mode == "content_addressed" and not verify_full_hashes:
            actual = resolved.name.lower()
            if not _SHA256_RE.fullmatch(actual):
                check.add(
                    "error", "model.content_address_missing",
                    f"Snapshot file is not backed by a SHA256-addressed blob: {requirement.path}",
                )
                continue
        else:
            actual = sha256_file(candidate)
        if actual != requirement.sha256:
            check.add(
                "error", "model.checksum_mismatch",
                f"Snapshot checksum mismatch: {requirement.path}",
            )
        else:
            verified += 1
    metrics.update({
        "required_files": len(model.required_files),
        "checked_files": checked,
        "checksum_verified_files": verified,
        "required_files_size_bytes": total_bytes,
        "snapshot_root": str(root),
    })
    return metrics


def inspect_docker_image(image: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"cannot inspect Docker image {image}")
    return completed.stdout.strip()


def _check_secret(path: Optional[Path]) -> ResourceCheck:
    check = ResourceCheck("secret")
    if path is None:
        check.metrics = {"configured": False, "has_hf_token": False}
        check.add(
            "warning", "secret.not_configured",
            "No secrets file configured; cached offline models remain usable",
        )
        return check
    resolved = path.expanduser().resolve(strict=False)
    check.metrics = {
        "configured": True,
        "path": str(resolved),
        "exists": resolved.is_file(),
        "has_hf_token": False,
    }
    if not resolved.is_file():
        check.add("warning", "secret.file_missing", "Secrets file is missing")
        return check
    mode = stat.S_IMODE(resolved.stat().st_mode)
    check.metrics["permissions_octal"] = f"{mode:04o}"
    check.metrics["secure_permissions"] = mode & 0o077 == 0
    if mode & 0o077:
        check.add(
            "warning", "secret.permissions_open",
            "Secrets file is accessible to group or other users; prefer mode 0600",
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check.add("error", "secret.invalid_json", f"Cannot parse secrets JSON: {type(exc).__name__}")
        return check
    if not isinstance(payload, Mapping):
        check.add("error", "secret.invalid_root", "Secrets JSON must contain an object")
        return check
    has_token = isinstance(payload.get("HF_TOKEN"), str) and bool(payload["HF_TOKEN"].strip())
    check.metrics["has_hf_token"] = has_token
    if not has_token:
        check.add("warning", "secret.hf_token_missing", "Secrets file has no non-empty HF_TOKEN")
    return check


def _model_mount(model: ModelSpec, cache_root: Path) -> tuple[Path, str, str]:
    """Return host source, container mount, and immutable container model path."""

    try:
        relative = model.local_path.relative_to(cache_root)
    except ValueError:
        return model.local_path, "/opt/farm/model", "/opt/farm/model"
    container_root = "/opt/farm/hf-cache"
    container_model = f"{container_root}/{relative.as_posix()}"
    return cache_root, container_root, container_model


def run_resource_preflight(
    manifest: ModelManifest,
    *,
    gpu_identifier: str | int,
    services: Optional[Iterable[str]] = None,
    secrets_file: Optional[Path] = None,
    include_pipeline_models: bool = False,
    include_all_runtimes: bool = False,
    verify_full_hashes: bool = False,
    check_docker_images: bool = True,
    gpu_inventory: Optional[Sequence[GpuInfo]] = None,
    docker_image_resolver: Optional[Callable[[str], str]] = None,
) -> ResourceReport:
    """Run bounded read-only checks and return a machine-readable report."""

    started_unix = time.time()
    started = time.perf_counter()
    checks: list[ResourceCheck] = []
    resolved_services: dict[str, dict[str, Any]] = {}
    requested = list(services) if services is not None else list(manifest.services)
    unknown = sorted(set(requested) - set(manifest.services))
    if unknown:
        raise ResourceConfigError(f"unknown services: {unknown}")
    if not requested:
        raise ResourceConfigError("at least one service must be selected")

    gpu_check = ResourceCheck("gpu")
    inventory: list[GpuInfo] = []
    selected: Optional[GpuInfo] = None
    try:
        inventory = list(gpu_inventory) if gpu_inventory is not None else query_gpus()
        gpu_check.metrics["inventory"] = [asdict(gpu) for gpu in inventory]
        if not inventory:
            gpu_check.add("error", "gpu.none", "No NVIDIA GPUs were reported")
        else:
            selected = select_gpu(inventory, gpu_identifier)
            gpu_check.metrics["selected"] = asdict(selected)
    except (OSError, RuntimeError, ResourceConfigError) as exc:
        gpu_check.add("error", "gpu.query_failed", str(exc))
    checks.append(gpu_check)

    runtime_names = set(manifest.runtimes) if include_all_runtimes else {
        manifest.services[name].runtime for name in requested
    }
    resolver = docker_image_resolver or inspect_docker_image
    for runtime_name in sorted(runtime_names):
        runtime = manifest.runtimes[runtime_name]
        check = ResourceCheck(f"runtime:{runtime_name}")
        check.metrics = {
            "image": runtime.image,
            "expected_image_id": runtime.image_id,
        }
        if runtime.python is not None:
            check.metrics["python"] = runtime.python
        if check_docker_images:
            try:
                actual = resolver(runtime.image)
                check.metrics["actual_image_id"] = actual
                if actual != runtime.image_id:
                    check.add(
                        "error", "runtime.image_id_mismatch",
                        "Docker tag does not resolve to the pinned image ID",
                    )
            except (OSError, RuntimeError) as exc:
                check.add("error", "runtime.image_missing", str(exc))
        else:
            check.add("warning", "runtime.check_skipped", "Docker image check was disabled")
        checks.append(check)

    checks.append(_check_secret(secrets_file))

    for service_name in requested:
        service = manifest.services[service_name]
        check = ResourceCheck(f"service:{service_name}")
        model_metrics = _check_model(
            check, service.model, verify_full_hashes=verify_full_hashes
        )
        memory_plan: dict[str, Any] = {}
        if selected is None:
            check.add("error", "service.no_gpu", "Cannot plan service without a selected GPU")
        else:
            memory_plan = compute_memory_plan(selected, service.memory)
            if not memory_plan["total_ok"]:
                check.add(
                    "error", "service.gpu_too_small",
                    "Selected GPU is below the service minimum total VRAM",
                )
            if not memory_plan["target_reachable"]:
                check.add(
                    "error", "service.allocation_unreachable",
                    "Target allocation exceeds the bounded utilization maximum",
                )
            if not memory_plan["free_ok"]:
                check.add(
                    "error", "service.insufficient_free_vram",
                    "Free VRAM cannot satisfy planned allocation plus reserve",
                )
        mount_source, mount_target, container_model = _model_mount(
            service.model, manifest.cache_root
        )
        runtime = manifest.runtimes[service.runtime]
        resolution = {
            "runtime": service.runtime,
            "runtime_image": runtime.image,
            "runtime_image_id": runtime.image_id,
            "served_model_name": service.served_model_name,
            "port": service.port,
            "host_model_path": str(service.model.local_path),
            "model_mount_source": str(mount_source),
            "model_mount_target": mount_target,
            "container_model_path": container_model,
            "vllm_args": list(service.vllm_args),
            "memory": memory_plan,
        }
        resolved_services[service_name] = resolution
        check.metrics = {"model": model_metrics, **resolution}
        checks.append(check)

    if include_pipeline_models:
        for model_name, model in sorted(manifest.pipeline_models.items()):
            check = ResourceCheck(f"pipeline_model:{model_name}")
            check.metrics = _check_model(
                check, model, verify_full_hashes=verify_full_hashes
            )
            checks.append(check)

    return ResourceReport(
        manifest_path=manifest.manifest_path,
        manifest_sha256=sha256_file(manifest.manifest_path),
        gpu_identifier=str(gpu_identifier),
        gpu_inventory=inventory,
        selected_gpu=selected,
        services=resolved_services,
        checks=checks,
        started_unix_s=started_unix,
        duration_s=time.perf_counter() - started,
    )


def format_resource_summary(report: ResourceReport) -> str:
    gpu = report.selected_gpu
    lines = [
        f"FARM resource preflight: {report.status.upper()}",
        f"manifest: {report.manifest_path}",
        (
            f"gpu: {gpu.index} {gpu.name} ({gpu.free_gib:.1f}/{gpu.total_gib:.1f} GiB free)"
            if gpu else "gpu: unresolved"
        ),
        f"services: {', '.join(report.services)}",
        f"errors: {report.errors}; warnings: {report.warnings}; duration: {report.duration_s:.2f}s",
    ]
    for check in report.checks:
        lines.append(f"  [{check.status.upper():4}] {check.name}")
        for finding in check.findings:
            lines.append(f"         {finding.severity}: {finding.code}: {finding.message}")
    return "\n".join(lines)
