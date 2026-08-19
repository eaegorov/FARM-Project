from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Finalizer, Plan, PlanError, Stage, canonical_json, plan_to_public_dict, render_template, resolve_plan_path
from .process import atomic_write_json, run_monitored_process, utc_now
from .process import atomic_write_text
from .integrity import DirectoryIntegrityError, directory_tree_descriptor
from .source_snapshot import (
    SourceSnapshotIntegrityError,
    validated_source_snapshot_project_root,
)


class PipelineRunError(RuntimeError):
    """Raised after recording a fail-closed pipeline or stage failure."""


@dataclass(frozen=True)
class ResolvedStage:
    stage: Stage
    command: tuple[str, ...]
    redacted_command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    env_hashes: Mapping[str, str]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    pass_json: tuple[Path, ...]
    fingerprint_inputs: tuple[Path, ...]


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def selected_device_telemetry(
    telemetry: Mapping[str, Any], pass_payloads: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Project whole-device counters onto the exact GPU selected by preflight."""

    selected = next(
        (value.get("selected_gpu") for value in pass_payloads
         if isinstance(value.get("selected_gpu"), Mapping)),
        None,
    )
    if not isinstance(selected, Mapping) or not selected.get("uuid"):
        return None
    uuid = str(selected["uuid"])
    baseline = (telemetry.get("gpu_device_baseline_used_by_uuid_mb") or {}).get(uuid)
    peak = (telemetry.get("gpu_device_peak_used_by_uuid_mb") or {}).get(uuid)
    free = (telemetry.get("gpu_device_min_free_by_uuid_mb") or {}).get(uuid)
    total = (telemetry.get("gpu_device_total_by_uuid_mb") or {}).get(uuid)
    delta = (telemetry.get("gpu_device_peak_delta_by_uuid_mb") or {}).get(uuid)
    return {
        "measurement_scope": "whole_selected_device_not_pid_attributed",
        "uuid": uuid, "index": selected.get("index"), "name": selected.get("name"),
        "baseline_used_mb": baseline, "peak_used_mb": peak, "peak_delta_mb": delta,
        "min_free_mb": free, "total_mb": total,
    }


def _git_metadata(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode == 0:
            result["commit"] = commit.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
        if dirty.returncode == 0:
            result["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


_SOURCE_SNAPSHOT_DIRECTORIES = (
    "configs",
    "docker",
    "requirements",
    "ros",
    "scripts",
    "src",
    "tests",
    "tools",
    # The main runtime image installs these two pinned submodule packages in
    # editable mode. The signed snapshot is bind-mounted over the repository
    # inside the container, so it must carry the actual import targets rather
    # than only the submodule commit recorded in its manifest.
    "third_party/yoloe/ultralytics",
    "third_party/yoloe/third_party/ml-mobileclip/mobileclip",
)
_SOURCE_SNAPSHOT_ROOT_FILES = (
    ".gitmodules",
    "pyproject.toml",
    "run.sh",
    "bootstrap_models.sh",
    "README.md",
    "LICENSE",
)
_SOURCE_SNAPSHOT_EXCLUDED_DIRECTORIES = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".secrets",
    "__pycache__",
    "cache",
    "caches",
    "output",
    "outputs",
    "secrets",
    "third_party",
}
_SOURCE_SNAPSHOT_SECRET_FILENAMES = {
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "secrets.env",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "token",
    "token.txt",
}
_SOURCE_SNAPSHOT_SECRET_SUFFIXES = {
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".pem",
    ".pfx",
}
_SOURCE_SNAPSHOT_TREE_SCHEMA = "farm.source-snapshot-tree.v1"


def _git_submodule_metadata(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "entries": []}
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "submodule", "status", "--recursive"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    if completed.returncode != 0:
        return result
    entries: list[dict[str, Any]] = []
    status_names = {
        " ": "clean",
        "-": "uninitialized",
        "+": "different_commit",
        "U": "merge_conflict",
    }
    for raw_line in completed.stdout.splitlines():
        if not raw_line:
            continue
        prefix, fields = raw_line[0], raw_line[1:].strip().split(maxsplit=2)
        if len(fields) < 2:
            continue
        entries.append({
            "path": fields[1],
            "commit": fields[0],
            "status": status_names.get(prefix, "unknown"),
            "description": fields[2] if len(fields) > 2 else "",
        })
    result.update({"available": True, "entries": entries})
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    _hash_file(path, digest)
    return digest.hexdigest()


def _source_snapshot_tree_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    records = [dict(record) for record in sorted(files, key=lambda row: str(row["path"]))]
    payload = {"schema": _SOURCE_SNAPSHOT_TREE_SCHEMA, "files": records}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _source_snapshot_secret_file(path: Path) -> bool:
    name = path.name.lower()
    return bool(
        name == ".env"
        or name.startswith(".env.")
        or name in _SOURCE_SNAPSHOT_SECRET_FILENAMES
        or path.suffix.lower() in _SOURCE_SNAPSHOT_SECRET_SUFFIXES
    )


def _source_snapshot_files(
    project_root: Path,
    run_dir: Path,
    *,
    excluded_source_paths: Sequence[Path] = (),
) -> tuple[list[Path], int, int]:
    """Enumerate only first-party source files without following symlinks."""

    selected: list[Path] = []
    excluded_secret_count = 0
    excluded_source_count = 0
    excluded = {
        path.expanduser().resolve(strict=False) for path in excluded_source_paths
    }

    def include_file(path: Path) -> None:
        nonlocal excluded_secret_count, excluded_source_count
        if path.resolve(strict=False) in excluded:
            excluded_source_count += 1
            return
        if path.is_symlink():
            raise PipelineRunError(f"Source snapshot refuses symlink: {path}")
        if _source_snapshot_secret_file(path):
            excluded_secret_count += 1
            return
        if path.suffix.lower() in {".pyc", ".pyo"}:
            return
        if not path.is_file():
            raise PipelineRunError(f"Source snapshot refuses special file: {path}")
        selected.append(path)

    for name in _SOURCE_SNAPSHOT_ROOT_FILES:
        path = project_root / name
        if path.exists() or path.is_symlink():
            include_file(path)

    resolved_run = run_dir.resolve()
    for name in _SOURCE_SNAPSHOT_DIRECTORIES:
        source_root = project_root / name
        if not source_root.exists() and not source_root.is_symlink():
            continue
        if source_root.is_symlink() or not source_root.is_dir():
            raise PipelineRunError(
                f"Source snapshot directory is not a real directory: {source_root}"
            )
        for current, raw_directories, raw_files in os.walk(
            source_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            directories: list[str] = []
            for child_name in sorted(raw_directories):
                child = current_path / child_name
                if child.is_symlink():
                    raise PipelineRunError(
                        f"Source snapshot refuses directory symlink: {child}"
                    )
                if child_name.lower() in _SOURCE_SNAPSHOT_EXCLUDED_DIRECTORIES:
                    continue
                resolved_child = child.resolve()
                if resolved_child == resolved_run or resolved_run.is_relative_to(
                    resolved_child
                ):
                    continue
                directories.append(child_name)
            raw_directories[:] = directories
            for child_name in sorted(raw_files):
                include_file(current_path / child_name)

    selected.sort(key=lambda path: path.relative_to(project_root).as_posix())
    return selected, excluded_secret_count, excluded_source_count


def _create_source_snapshot(
    project_root: Path,
    run_dir: Path,
    *,
    source_config_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically capture the exact allowlisted first-party source tree."""

    project_root = project_root.resolve(strict=True)
    destination = run_dir / "config" / "source_snapshot"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PipelineRunError(f"Source snapshot already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(
        prefix=".source_snapshot.", dir=destination.parent
    ))
    try:
        snapshot_root = temporary / "FARM-Project"
        snapshot_root.mkdir()
        source_files, excluded_secret_count, excluded_source_count = (
            _source_snapshot_files(
                project_root,
                run_dir,
                excluded_source_paths=(source_config_path,)
                if source_config_path is not None else (),
            )
        )
        records: list[dict[str, Any]] = []
        total_size = 0
        for source in source_files:
            relative = source.relative_to(project_root)
            target = snapshot_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source_stat = source.stat(follow_symlinks=False)
            source_mode = stat.S_IMODE(source_stat.st_mode)
            shutil.copyfile(source, target, follow_symlinks=False)
            os.chmod(target, source_mode)
            source_sha256 = _sha256_file(source)
            target_sha256 = _sha256_file(target)
            if source_sha256 != target_sha256:
                raise PipelineRunError(
                    f"Source changed while snapshotting: {source}"
                )
            target_stat = target.stat(follow_symlinks=False)
            total_size += int(target_stat.st_size)
            records.append({
                "path": relative.as_posix(),
                "size": int(target_stat.st_size),
                "mode": int(stat.S_IMODE(target_stat.st_mode)),
                "executable": bool(target_stat.st_mode & 0o111),
                "sha256": target_sha256,
            })

        tree_sha256 = _source_snapshot_tree_sha256(records)
        snapshot_manifest = {
            "schema": "farm.source-snapshot.v1",
            "created_at": utc_now(),
            "source_project_root": str(project_root),
            "project_relative_path": "FARM-Project",
            "tree_schema": _SOURCE_SNAPSHOT_TREE_SCHEMA,
            "tree_sha256": tree_sha256,
            "file_count": len(records),
            "total_size": total_size,
            "files": records,
            "git": _git_metadata(project_root),
            "submodules": _git_submodule_metadata(project_root),
            "policy": {
                "included_directories": list(_SOURCE_SNAPSHOT_DIRECTORIES),
                "included_root_files": list(_SOURCE_SNAPSHOT_ROOT_FILES),
                "excluded_secret_file_count": excluded_secret_count,
                "excluded_source_config_count": excluded_source_count,
                "symlinks": "rejected",
                "special_files": "rejected",
            },
        }
        atomic_write_json(temporary / "manifest.json", snapshot_manifest)
        os.replace(temporary, destination)
        return {
            "schema": "farm.source-snapshot.v1",
            "relative_path": "config/source_snapshot/FARM-Project",
            "manifest_relative_path": "config/source_snapshot/manifest.json",
            "tree_sha256": tree_sha256,
            "file_count": len(records),
        }
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(exc, PipelineRunError):
            raise
        raise PipelineRunError(f"Source snapshot failed: {exc}") from exc


def _secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in ("token", "secret", "password", "passwd", "api_key", "apikey"))


def _redact_command(command: Sequence[str], stage_env: Mapping[str, str]) -> tuple[str, ...]:
    secrets = {value for key, value in stage_env.items() if value and _secret_key(key)}
    redacted: list[str] = []
    for token in command:
        safe = token
        for secret in secrets:
            safe = safe.replace(secret, "<redacted>")
        redacted.append(safe)
    return tuple(redacted)


def _redact_config(value: Any, key: str = "") -> Any:
    if key and _secret_key(key) and not key.lower().endswith(("_file", "_path")):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(child_key): _redact_config(child, str(child_key)) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_redact_config(child) for child in value]
    return value


def _hash_file(path: Path, digest: "hashlib._Hash") -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def fingerprint_path(path: Path, mode: str) -> Mapping[str, Any]:
    """Fingerprint a declared dependency without following directory symlinks."""

    if not path.exists() and not path.is_symlink():
        return {"path": str(path), "kind": "missing"}
    if path.is_symlink():
        return {"path": str(path), "kind": "symlink", "target": os.readlink(path)}
    if path.is_file():
        stat = path.stat()
        value: dict[str, Any] = {"path": str(path), "kind": "file", "size": stat.st_size}
        if mode == "content":
            digest = hashlib.sha256()
            _hash_file(path, digest)
            value["sha256"] = digest.hexdigest()
        else:
            value["mtime_ns"] = stat.st_mtime_ns
        return value
    if path.is_dir():
        digest = hashlib.sha256()
        count = 0
        total_size = 0
        for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
            relative = child.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            if child.is_symlink():
                digest.update(b"L")
                digest.update(os.readlink(child).encode("utf-8", errors="surrogateescape"))
                continue
            if child.is_file():
                stat = child.stat()
                count += 1
                total_size += stat.st_size
                digest.update(b"F")
                digest.update(str(stat.st_size).encode())
                if mode == "content":
                    _hash_file(child, digest)
                else:
                    digest.update(str(stat.st_mtime_ns).encode())
            elif child.is_dir():
                digest.update(b"D")
        return {
            "path": str(path),
            "kind": "directory",
            "mode": mode,
            "file_count": count,
            "total_size": total_size,
            "digest": digest.hexdigest(),
        }
    stat = path.stat()
    return {"path": str(path), "kind": "other", "mode": stat.st_mode, "mtime_ns": stat.st_mtime_ns}


def _atomic_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    relative_target = os.path.relpath(target, link.parent)
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, link)


def _run_id(config_sha256: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config_sha256[:8]}"


def _safe_run_id(value: str) -> str:
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise PlanError("run id contains unsafe characters")
    return value


class RunOrchestrator:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.scene_root = plan.output_root / plan.scene_id
        self.runs_root = self.scene_root / "runs"

    def _is_standard_contract(self) -> bool:
        scene = self.plan.raw.get("scene")
        return bool(
            isinstance(scene, Mapping)
            and scene.get("standard_contract") == "farm.scene.v1"
        )

    def _validated_execution_root(self, run_dir: Path) -> Path | None:
        """Validate and return canonical first-party source for standard runs."""

        if not self._is_standard_contract():
            return None
        try:
            root = validated_source_snapshot_project_root(run_dir, required=True)
        except SourceSnapshotIntegrityError as exc:
            raise PipelineRunError(
                f"Execution source snapshot integrity failure: {exc}"
            ) from exc
        if root is None:  # required=True makes this unreachable; keep typing fail-closed.
            raise PipelineRunError("Standard run has no execution source snapshot")
        try:
            config_sha256 = _sha256_file(self.plan.source_path.resolve(strict=True))
        except (FileNotFoundError, OSError) as exc:
            raise PipelineRunError("Standard run source config is missing") from exc
        if config_sha256 != self.plan.config_sha256:
            raise PipelineRunError(
                "Standard run source config changed after the plan was loaded"
            )
        return root

    def _new_run_dir(self, requested_id: str | None = None) -> Path:
        base = _safe_run_id(requested_id) if requested_id else _run_id(self.plan.config_sha256)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        if requested_id:
            candidate = self.runs_root / base
            if candidate.exists():
                raise PipelineRunError(f"Run already exists: {candidate}")
            candidate.mkdir()
            return candidate.resolve()
        candidate = self.runs_root / base
        suffix = 1
        while candidate.exists():
            candidate = self.runs_root / f"{base}-{suffix:02d}"
            suffix += 1
        candidate.mkdir()
        return candidate.resolve()

    def _resume_run_dir(self, run_id: str | None) -> Path:
        if run_id:
            candidate = self.runs_root / _safe_run_id(run_id)
        else:
            candidate = self.scene_root / "latest-attempt"
        if not candidate.exists():
            raise PipelineRunError(f"No resumable run found: {candidate}")
        run_dir = candidate.resolve()
        try:
            run_dir.relative_to(self.runs_root.resolve())
        except ValueError as exc:
            raise PipelineRunError(f"Run path escapes the configured runs directory: {run_dir}") from exc
        manifest = _safe_json_load(run_dir / "manifest.json")
        if not manifest:
            raise PipelineRunError(f"Run manifest is missing or invalid: {run_dir}")
        if manifest.get("config_sha256") != self.plan.config_sha256:
            raise PipelineRunError("Refusing to resume with a different pipeline config")
        self._validated_execution_root(run_dir)
        if (run_dir / "_SUCCESS.json").exists():
            raise PipelineRunError("Successful runs are immutable; create a new run instead")
        return run_dir

    def _initialize_run(self, run_dir: Path) -> None:
        for name in (
            "artifacts", "config", "final", "input", "logs", "mapping", "qa",
            "reports", "rgbd", "selection", "stages", "timing", "viewer", "visuals",
        ):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        source_snapshot = _create_source_snapshot(
            self.plan.project_root,
            run_dir,
            source_config_path=self.plan.source_path,
        )
        import yaml

        source_config = self.plan.raw.get("source_scene", self.plan.raw)
        atomic_write_text(
            run_dir / "config" / "source.redacted.yaml",
            yaml.safe_dump(_redact_config(source_config), sort_keys=False, allow_unicode=True),
        )
        atomic_write_json(run_dir / "config" / "resolved-plan.json", plan_to_public_dict(self.plan, run_dir))
        manifest = {
            "schema": "farm.pipeline-run.v1",
            "status": "running",
            "scene_id": self.plan.scene_id,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "config_source": str(self.plan.source_path),
            "config_sha256": self.plan.config_sha256,
            "project_root": str(self.plan.project_root),
            "source_snapshot": source_snapshot,
            "git": _git_metadata(self.plan.project_root),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "control_plane": {
                "python_executable": sys.executable,
                "pyyaml": str(yaml.__version__),
            },
            "stage_order": list(self.plan.stage_order),
        }
        atomic_write_json(run_dir / "manifest.json", manifest)
        self._validated_execution_root(run_dir)
        _atomic_symlink(self.scene_root / "latest-attempt", run_dir)

    def _update_manifest(self, run_dir: Path, **updates: Any) -> None:
        manifest = _safe_json_load(run_dir / "manifest.json") or {}
        manifest.update(updates)
        manifest["updated_at"] = utc_now()
        atomic_write_json(run_dir / "manifest.json", manifest)

    def _context(self, run_dir: Path) -> dict[str, str]:
        context = self.plan.base_context(run_dir)
        execution_root = self._validated_execution_root(run_dir)
        if execution_root is not None:
            context["execution_project_root"] = str(execution_root)
        for name, path in self.plan.resolved_artifacts(run_dir).items():
            context[f"artifact:{name}"] = str(path)
            context[f"artifacts.{name}"] = str(path)
        return context

    def _resolve_stage(self, stage: Stage, run_dir: Path) -> ResolvedStage:
        context = self._context(run_dir)
        command = tuple(render_template(value, context) for value in stage.command)
        stage_env = {key: render_template(value, context) for key, value in stage.env.items()}
        cwd = resolve_plan_path(render_template(stage.cwd, context), self.plan.project_root)
        inputs = tuple(resolve_plan_path(render_template(value, context), self.plan.project_root) for value in stage.inputs)
        outputs = tuple(resolve_plan_path(render_template(value, context), self.plan.project_root) for value in stage.outputs)
        pass_json = tuple(resolve_plan_path(render_template(value, context), self.plan.project_root) for value in stage.pass_json)
        fingerprint_inputs = tuple(
            resolve_plan_path(render_template(value, context), self.plan.project_root)
            for value in stage.fingerprint_inputs
        )
        env = dict(os.environ)
        env.update(stage_env)
        env_hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sorted(stage_env.items())}
        return ResolvedStage(
            stage=stage,
            command=command,
            redacted_command=_redact_command(command, stage_env),
            cwd=cwd,
            env=env,
            env_hashes=env_hashes,
            inputs=inputs,
            outputs=outputs,
            pass_json=pass_json,
            fingerprint_inputs=fingerprint_inputs,
        )

    def _stage_fingerprint(
        self,
        resolved: ResolvedStage,
        dependency_fingerprints: Mapping[str, str],
    ) -> tuple[str, list[Mapping[str, Any]]]:
        input_fingerprints = [
            fingerprint_path(path, resolved.stage.fingerprint_mode) for path in resolved.fingerprint_inputs
        ]
        value = {
            "schema": "farm.stage-fingerprint.v1",
            "stage_id": resolved.stage.id,
            "command": list(resolved.command),
            "cwd": str(resolved.cwd),
            "env_sha256": resolved.env_hashes,
            "inputs": input_fingerprints,
            "outputs": [str(path) for path in resolved.outputs],
            "pass_json": [str(path) for path in resolved.pass_json],
            "dependencies": dict(sorted(dependency_fingerprints.items())),
            "config_sha256": self.plan.config_sha256,
        }
        return hashlib.sha256(canonical_json(value).encode()).hexdigest(), input_fingerprints

    @staticmethod
    def _descendants(stage_ids: Sequence[str], stages: Mapping[str, Stage]) -> set[str]:
        selected = set(stage_ids)
        changed = True
        while changed:
            changed = False
            for stage in stages.values():
                if stage.id not in selected and any(dep in selected for dep in stage.needs):
                    selected.add(stage.id)
                    changed = True
        return selected

    @staticmethod
    def _validate_inputs(resolved: ResolvedStage) -> None:
        if not resolved.cwd.is_dir():
            raise PipelineRunError(f"Stage {resolved.stage.id} cwd does not exist: {resolved.cwd}")
        missing = [str(path) for path in resolved.inputs if not path.exists()]
        if missing:
            raise PipelineRunError(f"Stage {resolved.stage.id} is missing declared inputs: {', '.join(missing)}")

    @staticmethod
    def _validate_outputs(resolved: ResolvedStage) -> list[Mapping[str, Any]]:
        missing = [str(path) for path in resolved.outputs if not path.exists()]
        if missing:
            raise PipelineRunError(
                f"Stage {resolved.stage.id} exited successfully but declared outputs are missing: {', '.join(missing)}"
            )
        payloads: list[Mapping[str, Any]] = []
        for path in resolved.pass_json:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
                raise PipelineRunError(f"Stage {resolved.stage.id} PASS contract is invalid: {path}") from exc
            if not isinstance(value, Mapping) or str(value.get("status", "")).upper() != "PASS":
                raise PipelineRunError(f"Stage {resolved.stage.id} did not report PASS in {path}")
            payloads.append(value)
        return payloads

    def _run_stage(
        self,
        resolved: ResolvedStage,
        run_dir: Path,
        fingerprint: str,
        input_fingerprints: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        # Detect tamper before creating an attempt or advancing stage state.
        self._validated_execution_root(run_dir)
        stage_dir = run_dir / "stages" / resolved.stage.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        old_state = _safe_json_load(stage_dir / "state.json") or {}
        attempt = int(old_state.get("attempt", 0)) + 1
        attempt_dir = stage_dir / "attempts" / f"{attempt:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        for marker in (stage_dir / "_SUCCESS.json", stage_dir / "_FAILED.json"):
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
        state: dict[str, Any] = {
            "schema": "farm.pipeline-stage.v1",
            "stage_id": resolved.stage.id,
            "description": resolved.stage.description,
            "status": "running",
            "attempt": attempt,
            "fingerprint": fingerprint,
            "started_at": utc_now(),
            "command": list(resolved.redacted_command),
            "cwd": str(resolved.cwd),
            "env_keys": sorted(resolved.stage.env),
            "input_fingerprints": list(input_fingerprints),
            "declared_inputs": [str(path) for path in resolved.inputs],
            "declared_outputs": [str(path) for path in resolved.outputs],
            "pass_json": [str(path) for path in resolved.pass_json],
            "logs": {
                "stdout": str((attempt_dir / "stdout.log").relative_to(run_dir)),
                "stderr": str((attempt_dir / "stderr.log").relative_to(run_dir)),
                "telemetry": str((attempt_dir / "telemetry.jsonl").relative_to(run_dir)),
            },
        }
        atomic_write_json(stage_dir / "state.json", state)
        try:
            self._validate_inputs(resolved)
            # Narrow the validation-to-spawn window after all state setup.
            self._validated_execution_root(run_dir)
            result = run_monitored_process(
                resolved.command,
                cwd=resolved.cwd,
                env=resolved.env,
                stdout_path=attempt_dir / "stdout.log",
                stderr_path=attempt_dir / "stderr.log",
                telemetry_path=attempt_dir / "telemetry.jsonl",
                interval_seconds=self.plan.telemetry_interval_seconds,
                timeout_seconds=resolved.stage.timeout_seconds,
            )
            if result.interrupted:
                raise PipelineRunError(f"Stage {resolved.stage.id} was interrupted")
            if result.timed_out:
                raise PipelineRunError(f"Stage {resolved.stage.id} exceeded its timeout")
            if result.returncode != 0:
                raise PipelineRunError(f"Stage {resolved.stage.id} exited with code {result.returncode}")
            pass_payloads = self._validate_outputs(resolved)
            # A child returning 0/PASS cannot bless source altered while it ran.
            self._validated_execution_root(run_dir)
            telemetry_summary = result.telemetry.as_dict()
            state.update(
                {
                    "status": "success",
                    "finished_at": result.finished_at,
                    "duration_seconds": round(result.duration_seconds, 6),
                    "returncode": result.returncode,
                    "pid": result.pid,
                    "telemetry_summary": telemetry_summary,
                    "selected_gpu_telemetry": selected_device_telemetry(
                        telemetry_summary, pass_payloads
                    ),
                    "output_fingerprints": [fingerprint_path(path, "metadata") for path in resolved.outputs],
                }
            )
            atomic_write_json(stage_dir / "state.json", state)
            atomic_write_json(stage_dir / "_SUCCESS.json", state)
            return state
        except Exception as exc:
            state.update({"status": "failed", "finished_at": utc_now(), "error": str(exc)})
            atomic_write_json(stage_dir / "state.json", state)
            atomic_write_json(stage_dir / "_FAILED.json", state)
            if isinstance(exc, PipelineRunError):
                raise
            raise PipelineRunError(f"Stage {resolved.stage.id} failed to start: {exc}") from exc

    def _write_viewer_bundle(self, run_dir: Path) -> None:
        viewer_dir = run_dir / "viewer"
        artifacts = self.plan.resolved_artifacts(run_dir)
        relative_artifacts: dict[str, str] = {}
        artifact_integrity: dict[str, dict[str, Any]] = {}
        for name, path in artifacts.items():
            if not path.exists():
                raise PipelineRunError(f"Final artifact {name!r} is missing: {path}")
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(run_dir.resolve())
            except ValueError as exc:
                raise PipelineRunError(
                    f"Final artifact {name!r} is outside the immutable run bundle: {resolved_path}"
                ) from exc
            relative_artifacts[name] = os.path.relpath(resolved_path, viewer_dir)
            if resolved_path.is_file():
                artifact_integrity[name] = {
                    "kind": "file",
                    "path": relative_artifacts[name],
                    "bytes": int(resolved_path.stat().st_size),
                    "sha256": _sha256_file(resolved_path),
                }
            else:
                try:
                    tree = directory_tree_descriptor(resolved_path)
                except DirectoryIntegrityError as exc:
                    raise PipelineRunError(
                        f"Final artifact directory {name!r} cannot be made immutable: {exc}"
                    ) from exc
                artifact_integrity[name] = {
                    "kind": "directory",
                    "path": relative_artifacts[name],
                    **tree,
                }

        bundle: dict[str, Any] = {
            "schema": "farm.viewer-bundle.v1",
            "scene_id": self.plan.scene_id,
            "run_id": run_dir.name,
            "artifacts": relative_artifacts,
            "artifact_integrity": artifact_integrity,
            "configured": self.plan.viewer is not None,
        }
        if self.plan.viewer is not None:
            bundle["process"] = {
                "command": list(self.plan.viewer.command),
                "runtime": self.plan.viewer.runtime,
                "cwd": self.plan.viewer.cwd,
                "env": dict(self.plan.viewer.env),
                "default_host": self.plan.viewer.default_host,
                "default_port": self.plan.viewer.default_port,
                "startup_timeout_seconds": self.plan.viewer.startup_timeout_seconds,
                "health_url": self.plan.viewer.health_url,
            }
        atomic_write_json(viewer_dir / "bundle.json", bundle)

    def _write_timing(self, run_dir: Path, started_monotonic: float) -> Mapping[str, Any]:
        stages: list[Mapping[str, Any]] = []
        for stage_id in self.plan.stage_order:
            state = _safe_json_load(run_dir / "stages" / stage_id / "state.json") or {}
            stages.append(
                {
                    "stage_id": stage_id,
                    "status": state.get("status", "not-run"),
                    "duration_seconds": state.get("duration_seconds"),
                    "telemetry_summary": state.get("telemetry_summary"),
                }
            )
        value = {
            "schema": "farm.pipeline-timing.v1",
            "run_id": run_dir.name,
            "total_wall_seconds": round(time.monotonic() - started_monotonic, 6),
            "stages": stages,
        }
        atomic_write_json(run_dir / "timing" / "summary.json", value)
        return value

    def _write_standard_report(self, run_dir: Path) -> Mapping[str, Any] | None:
        """Create the required report bundle for canonical farm.scene.v1 runs."""

        if not self._is_standard_contract():
            return None
        execution_root = self._validated_execution_root(run_dir)
        assert execution_root is not None
        command = [
            sys.executable,
            "-B",
            str(execution_root / "scripts" / "build_farm_run_report.py"),
            "--run-dir",
            str(run_dir),
        ]
        report_env = dict(os.environ)
        report_env["PYTHONDONTWRITEBYTECODE"] = "1"
        log_dir = run_dir / "logs" / "report"
        result = run_monitored_process(
            command,
            cwd=execution_root,
            env=report_env,
            stdout_path=log_dir / "stdout.log",
            stderr_path=log_dir / "stderr.log",
            telemetry_path=log_dir / "telemetry.jsonl",
            interval_seconds=self.plan.telemetry_interval_seconds,
            timeout_seconds=180.0,
        )
        if result.returncode != 0 or result.timed_out or result.interrupted:
            raise PipelineRunError("final run report/video generation failed")
        self._validated_execution_root(run_dir)
        required = (run_dir / "REPORT.md", run_dir / "visuals" / "scene_summary.mp4")
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise PipelineRunError("final run report/video artifacts are missing")
        return {
            "duration_seconds": round(result.duration_seconds, 6),
            "report": "REPORT.md",
            "video": "visuals/scene_summary.mp4",
        }

    def _run_finalizers(self, run_dir: Path) -> list[dict[str, Any]]:
        context = self._context(run_dir)
        results: list[dict[str, Any]] = []
        for index, finalizer in enumerate(self.plan.finalizers):
            command = tuple(render_template(value, context) for value in finalizer.command)
            cwd = resolve_plan_path(render_template(finalizer.cwd, context), self.plan.project_root)
            stage_env = {
                key: render_template(value, context) for key, value in finalizer.env.items()
            }
            env = dict(os.environ)
            env.update(stage_env)
            log_dir = run_dir / "logs" / "finalizers"
            self._validated_execution_root(run_dir)
            result = run_monitored_process(
                command,
                cwd=cwd,
                env=env,
                stdout_path=log_dir / f"{index:02d}.stdout.log",
                stderr_path=log_dir / f"{index:02d}.stderr.log",
                telemetry_path=log_dir / f"{index:02d}.telemetry.jsonl",
                interval_seconds=self.plan.telemetry_interval_seconds,
                timeout_seconds=finalizer.timeout_seconds,
            )
            self._validated_execution_root(run_dir)
            row = {
                "index": index,
                "command": list(_redact_command(command, stage_env)),
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "interrupted": result.interrupted,
                "duration_seconds": round(result.duration_seconds, 6),
            }
            results.append(row)
            atomic_write_json(run_dir / "timing" / "finalizers.json", {
                "schema": "farm.pipeline-finalizers.v1",
                "results": results,
            })
            if result.returncode != 0 or result.timed_out or result.interrupted:
                raise PipelineRunError(f"Pipeline finalizer {index} failed")
        return results

    def run(
        self,
        *,
        resume: bool = False,
        run_id: str | None = None,
        force: bool = False,
        force_stages: Sequence[str] = (),
    ) -> Path:
        unknown_forced = sorted(set(force_stages) - set(self.plan.stage_order))
        if unknown_forced:
            raise PipelineRunError(f"Unknown --force-stage values: {', '.join(unknown_forced)}")
        if resume:
            run_dir = self._resume_run_dir(run_id)
            try:
                (run_dir / "_FAILED.json").unlink()
            except FileNotFoundError:
                pass
            self._update_manifest(run_dir, status="running", resumed_at=utc_now())
        else:
            run_dir = self._new_run_dir(run_id)
            self._initialize_run(run_dir)

        forced = set(self.plan.stage_order) if force else self._descendants(force_stages, self.plan.stages_by_id)
        dependency_fingerprints: dict[str, str] = {}
        stage_actions: dict[str, str] = {}
        started_monotonic = time.monotonic()
        primary_error: Exception | None = None
        cleanup_error: Exception | None = None
        try:
            for stage_id in self.plan.stage_order:
                stage = self.plan.stages_by_id[stage_id]
                resolved = self._resolve_stage(stage, run_dir)
                dep_values = {dep: dependency_fingerprints[dep] for dep in stage.needs}
                fingerprint, inputs = self._stage_fingerprint(resolved, dep_values)
                marker = _safe_json_load(run_dir / "stages" / stage_id / "_SUCCESS.json")
                current_outputs = [fingerprint_path(path, "metadata") for path in resolved.outputs]
                can_skip = (
                    stage_id not in forced
                    and marker is not None
                    and marker.get("fingerprint") == fingerprint
                    and all(path.exists() for path in resolved.outputs)
                    and marker.get("output_fingerprints") == current_outputs
                    and not any(stage_actions.get(dep) == "executed" for dep in stage.needs)
                )
                if can_skip:
                    self._validated_execution_root(run_dir)
                    dependency_fingerprints[stage_id] = fingerprint
                    stage_actions[stage_id] = "resumed-cache"
                    continue
                self._run_stage(resolved, run_dir, fingerprint, inputs)
                dependency_fingerprints[stage_id] = fingerprint
                stage_actions[stage_id] = "executed"
        except Exception as exc:
            primary_error = exc
        finally:
            try:
                self._run_finalizers(run_dir)
            except Exception as exc:
                cleanup_error = exc

        if primary_error is None and cleanup_error is None:
            self._write_viewer_bundle(run_dir)
            timing = self._write_timing(run_dir, started_monotonic)
            try:
                report = self._write_standard_report(run_dir)
                timing = self._write_timing(run_dir, started_monotonic)
                if report is not None:
                    report["duration_seconds"] = round(time.monotonic() - started_monotonic, 6)
                    execution_root = self._validated_execution_root(run_dir)
                    assert execution_root is not None
                    subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            str(execution_root / "scripts" / "build_farm_run_report.py"),
                            "--run-dir",
                            str(run_dir),
                            "--no-video",
                        ],
                        cwd=str(execution_root),
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                        check=True,
                        shell=False,
                    )
                    self._validated_execution_root(run_dir)
            except Exception as exc:
                primary_error = exc
                report = None
            if primary_error is not None:
                timing = self._write_timing(run_dir, started_monotonic)
                failure = {
                    "schema": "farm.pipeline-failure.v1",
                    "status": "failed",
                    "scene_id": self.plan.scene_id,
                    "run_id": run_dir.name,
                    "failed_at": utc_now(),
                    "error": str(primary_error),
                    "cleanup_error": None,
                    "stage_actions": stage_actions,
                    "total_wall_seconds": timing["total_wall_seconds"],
                }
                atomic_write_json(run_dir / "_FAILED.json", failure)
                self._update_manifest(run_dir, status="failed", failed_at=failure["failed_at"], error=str(primary_error))
                raise PipelineRunError(str(primary_error)) from primary_error
            success = {
                "schema": "farm.pipeline-success.v1",
                "status": "success",
                "scene_id": self.plan.scene_id,
                "run_id": run_dir.name,
                "finished_at": utc_now(),
                "config_sha256": self.plan.config_sha256,
                "stage_actions": stage_actions,
                "total_wall_seconds": timing["total_wall_seconds"],
                "viewer_bundle": "viewer/bundle.json",
                "viewer_bundle_sha256": _sha256_file(
                    run_dir / "viewer" / "bundle.json"
                ),
                "run_report": report,
            }
            self._validated_execution_root(run_dir)
            atomic_write_json(run_dir / "_SUCCESS.json", success)
            self._update_manifest(run_dir, status="success", finished_at=success["finished_at"])
            _atomic_symlink(self.scene_root / "latest", run_dir)
            return run_dir

        error = primary_error or cleanup_error or PipelineRunError("pipeline failed")
        timing = self._write_timing(run_dir, started_monotonic)
        failure = {
            "schema": "farm.pipeline-failure.v1",
            "status": "failed",
            "scene_id": self.plan.scene_id,
            "run_id": run_dir.name,
            "failed_at": utc_now(),
            "error": str(error),
            "cleanup_error": str(cleanup_error) if cleanup_error else None,
            "stage_actions": stage_actions,
            "total_wall_seconds": timing["total_wall_seconds"],
        }
        atomic_write_json(run_dir / "_FAILED.json", failure)
        self._update_manifest(
            run_dir,
            status="failed",
            failed_at=failure["failed_at"],
            error=str(error),
            cleanup_error=failure["cleanup_error"],
        )
        if isinstance(error, PipelineRunError):
            raise error
        raise PipelineRunError(str(error)) from error


def resolve_run_dir(plan: Plan | None, value: str | Path | None, *, attempt: bool = False) -> Path:
    if value is not None:
        run_dir = Path(value).expanduser().resolve()
    elif plan is not None:
        link = plan.output_root / plan.scene_id / ("latest-attempt" if attempt else "latest")
        if not link.exists():
            raise PipelineRunError(f"Run link does not exist: {link}")
        run_dir = link.resolve()
    else:
        raise PipelineRunError("Provide --run or --config")
    if not run_dir.is_dir():
        raise PipelineRunError(f"Run directory does not exist: {run_dir}")
    if not (run_dir / "manifest.json").is_file():
        raise PipelineRunError(f"Not a FARM run directory: {run_dir}")
    return run_dir
