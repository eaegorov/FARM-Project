from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .config import canonical_json, render_template, resolve_plan_path
from .process import atomic_write_json, open_detached_process, port_is_available, process_identity_matches, process_start_token, utc_now, wait_for_http


class ViewerError(RuntimeError):
    pass


_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SNAPSHOT_TREE_SCHEMA = "farm.source-snapshot-tree.v1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ViewerError(f"Invalid or missing JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ViewerError(f"Expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_tree_sha256(files: list[dict[str, Any]]) -> str:
    records = [dict(record) for record in sorted(files, key=lambda row: str(row["path"]))]
    payload = {"schema": _SOURCE_SNAPSHOT_TREE_SCHEMA, "files": records}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _run_relative_path(run_dir: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ViewerError(f"Source snapshot {label} must be a relative path")
    unresolved = run_dir / value
    if unresolved.is_symlink():
        raise ViewerError(f"Source snapshot {label} must not be a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(run_dir.resolve())
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ViewerError(
            f"Source snapshot {label} is missing or escapes the run: {value}"
        ) from exc
    return resolved


def _validated_snapshot_project_root(
    run_dir: Path, run_manifest: Mapping[str, Any]
) -> Path | None:
    """Return a cryptographically validated snapshot, or None for old runs."""

    reference = run_manifest.get("source_snapshot")
    if reference is None:
        return None
    if not isinstance(reference, Mapping):
        raise ViewerError("Run source_snapshot metadata must be a mapping")
    if reference.get("schema") != "farm.source-snapshot.v1":
        raise ViewerError("Unsupported run source_snapshot metadata schema")
    snapshot_root = _run_relative_path(
        run_dir, reference.get("relative_path"), "relative_path"
    )
    if not snapshot_root.is_dir():
        raise ViewerError("Source snapshot project root is not a directory")
    snapshot_manifest_path = _run_relative_path(
        run_dir,
        reference.get("manifest_relative_path"),
        "manifest_relative_path",
    )
    snapshot = _load_json(snapshot_manifest_path)
    if snapshot.get("schema") != "farm.source-snapshot.v1":
        raise ViewerError("Unsupported source snapshot manifest schema")
    if (
        snapshot.get("project_relative_path") != "FARM-Project"
        or snapshot.get("tree_schema") != _SOURCE_SNAPSHOT_TREE_SCHEMA
    ):
        raise ViewerError("Source snapshot manifest contract mismatch")
    raw_files = snapshot.get("files")
    if not isinstance(raw_files, list) or not all(
        isinstance(record, Mapping) for record in raw_files
    ):
        raise ViewerError("Source snapshot manifest files must be a list")

    expected_paths: set[str] = set()
    actual_records: list[dict[str, Any]] = []
    for raw_record in raw_files:
        record = dict(raw_record)
        relative = record.get("path")
        expected_sha256 = str(record.get("sha256") or "").lower()
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or relative in expected_paths
            or not _SHA256_RE.fullmatch(expected_sha256)
            or not isinstance(record.get("executable"), bool)
        ):
            raise ViewerError("Invalid source snapshot file record")
        expected_paths.add(relative)
        unresolved = snapshot_root / relative
        if unresolved.is_symlink():
            raise ViewerError(f"Source snapshot contains a symlink: {relative}")
        try:
            path = unresolved.resolve(strict=True)
            path.relative_to(snapshot_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ViewerError(
                f"Source snapshot file is missing or escapes its root: {relative}"
            ) from exc
        if not path.is_file():
            raise ViewerError(f"Source snapshot entry is not a file: {relative}")
        path_stat = path.stat(follow_symlinks=False)
        try:
            expected_size = int(record["size"])
            expected_mode = int(record["mode"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ViewerError("Invalid source snapshot size/mode metadata") from exc
        mode = int(stat.S_IMODE(path_stat.st_mode))
        executable = bool(path_stat.st_mode & 0o111)
        sha256 = _sha256_file(path)
        if (
            int(path_stat.st_size) != expected_size
            or mode != expected_mode
            or executable is not record["executable"]
            or sha256 != expected_sha256
        ):
            raise ViewerError(f"Source snapshot file integrity mismatch: {relative}")
        actual_records.append({
            "path": relative,
            "size": int(path_stat.st_size),
            "mode": mode,
            "executable": executable,
            "sha256": sha256,
        })

    actual_paths: set[str] = set()
    for path in snapshot_root.rglob("*"):
        relative = path.relative_to(snapshot_root).as_posix()
        if path.is_symlink():
            raise ViewerError(f"Source snapshot contains a symlink: {relative}")
        if path.is_file():
            actual_paths.add(relative)
        elif not path.is_dir():
            raise ViewerError(f"Source snapshot contains a special file: {relative}")
    if actual_paths != expected_paths:
        raise ViewerError("Source snapshot file set does not match its manifest")

    tree_sha256 = _source_snapshot_tree_sha256(actual_records)
    manifest_tree = str(snapshot.get("tree_sha256") or "").lower()
    reference_tree = str(reference.get("tree_sha256") or "").lower()
    try:
        manifest_count = int(snapshot.get("file_count"))
        reference_count = int(reference.get("file_count"))
    except (TypeError, ValueError) as exc:
        raise ViewerError("Invalid source snapshot file count metadata") from exc
    if (
        tree_sha256 != manifest_tree
        or tree_sha256 != reference_tree
        or len(actual_records) != manifest_count
        or len(actual_records) != reference_count
    ):
        raise ViewerError("Source snapshot manifest/root metadata mismatch")
    return snapshot_root


def viewer_runtime_dir(run_dir: Path) -> Path:
    scene_root = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
    return scene_root / ".runtime" / run_dir.name / "viewer"


def _viewer_metadata(run_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    path = viewer_runtime_dir(run_dir) / "process.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return path, None
    return path, value if isinstance(value, dict) else None


def viewer_status(run_dir: Path) -> dict[str, Any]:
    path, metadata = _viewer_metadata(run_dir)
    if not metadata:
        return {"status": "not-started", "metadata_path": str(path)}
    pid = int(metadata.get("pid", 0) or 0)
    alive = process_identity_matches(pid, metadata.get("process_start_token"))
    status = "running" if alive else str(metadata.get("status", "stopped"))
    if status in {"starting", "running"} and not alive:
        status = "stale"
    return {**metadata, "status": status, "alive": alive, "metadata_path": str(path)}


def _terminate(metadata: Mapping[str, Any], timeout_seconds: float) -> None:
    pid = int(metadata.get("pid", 0) or 0)
    token = metadata.get("process_start_token")
    if not process_identity_matches(pid, token):
        raise ViewerError("Viewer PID metadata is stale; refusing to signal an unrelated process")
    expected_pgid = int(metadata.get("process_group_id", 0) or 0)
    try:
        current_pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        raise ViewerError(f"Cannot inspect viewer process group: {exc}") from exc
    if current_pgid != expected_pgid or expected_pgid <= 1:
        raise ViewerError("Viewer process-group identity mismatch; refusing unsafe stop")
    os.killpg(expected_pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_identity_matches(pid, token):
            return
        time.sleep(0.1)
    if process_identity_matches(pid, token):
        os.killpg(expected_pgid, signal.SIGKILL)


def _bundle_context(run_dir: Path, bundle: Mapping[str, Any], host: str, port: int) -> dict[str, str]:
    manifest = _load_json(run_dir / "manifest.json")
    viewer_dir = run_dir / "viewer"
    original_project_root = str(manifest.get("project_root", run_dir))
    snapshot_project_root = _validated_snapshot_project_root(run_dir, manifest)
    project_root = str(snapshot_project_root or original_project_root)
    context = {
        "run_dir": str(run_dir),
        "project_root": project_root,
        "original_project_root": original_project_root,
        "scene_id": str(manifest.get("scene_id", bundle.get("scene_id", ""))),
        "run_id": str(manifest.get("run_id", run_dir.name)),
        "host": host,
        "port": str(port),
        "python_executable": sys.executable,
    }
    artifacts = bundle.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ViewerError("viewer bundle artifacts must be a mapping")
    for name, relative in artifacts.items():
        if not isinstance(name, str) or not isinstance(relative, str):
            raise ViewerError("viewer bundle artifact entries must be strings")
        path = (viewer_dir / relative).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ViewerError(f"Viewer artifact escapes run bundle: {name}") from exc
        if not path.exists():
            raise ViewerError(f"Viewer artifact is missing: {path}")
        context[f"artifact:{name}"] = str(path)
        context[f"artifacts.{name}"] = str(path)
    return context


def _model_manifest_path(
    run_dir: Path,
    context: Mapping[str, str],
    resolved_context: Mapping[str, Any],
) -> Path:
    raw_manifest = resolved_context.get("model_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise ViewerError("Resolved context does not declare the pinned model manifest")
    original_root = Path(
        context.get("original_project_root", context["project_root"])
    ).expanduser().resolve(strict=False)
    raw_path = Path(raw_manifest).expanduser()
    original_path = (
        raw_path.resolve(strict=False)
        if raw_path.is_absolute()
        else (original_root / raw_path).resolve(strict=False)
    )
    candidates = [original_path]
    snapshot_root = Path(context["project_root"]).resolve(strict=False)
    if snapshot_root != original_root:
        try:
            relative = original_path.relative_to(original_root)
        except ValueError:
            relative = None
        if relative is not None:
            candidates.append(snapshot_root / relative)

    expected_sha256 = str(
        resolved_context.get("model_manifest_sha256") or ""
    ).strip().lower()
    if expected_sha256 and not _SHA256_RE.fullmatch(expected_sha256):
        raise ViewerError("resolved_context model_manifest_sha256 is invalid")
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if expected_sha256 and _sha256_file(candidate) != expected_sha256:
            continue
        return candidate
    raise ViewerError(
        "Pinned model manifest is missing or does not match resolved_context"
    )


def _dockerize_viewer_command(
    run_dir: Path,
    context: Mapping[str, str],
    host_command: list[str],
) -> list[str]:
    """Run the standard viewer in the pinned main image without a host UI stack."""

    if shutil.which("docker") is None:
        raise ViewerError("Docker is required for the configured viewer runtime")
    if len(host_command) < 2 or Path(host_command[1]).name != "view_scene_state.py":
        raise ViewerError("Docker viewer runtime only supports the standard view_scene_state.py command")
    resolved_context = _load_json(run_dir / "input" / "resolved_context.json")
    model_manifest = _load_json(
        _model_manifest_path(run_dir, context, resolved_context)
    )
    runtimes = model_manifest.get("runtimes")
    main = runtimes.get("main") if isinstance(runtimes, Mapping) else None
    if not isinstance(main, Mapping):
        raise ViewerError("Model manifest does not declare runtimes.main")
    image_id = main.get("image_id")
    if not isinstance(image_id, str) or not _IMAGE_ID_RE.fullmatch(image_id):
        raise ViewerError("runtimes.main.image_id must be a pinned sha256 image ID")
    try:
        user_uid = int(main["user_uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ViewerError("runtimes.main.user_uid must declare a non-root UID") from exc
    if user_uid <= 0:
        raise ViewerError("runtimes.main.user_uid must declare a non-root UID")

    project_root = Path(context["project_root"]).resolve(strict=True)
    digest = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:16]
    container_name = f"farm-viewer-{digest}"
    return [
        "docker", "run", "--rm", "--init", "--pull", "never",
        "--name", container_name,
        "--label", "com.goldengait.farm.viewer=true",
        "--label", f"com.goldengait.farm.viewer.run={digest}",
        "--user", f"{user_uid}:{os.getgid()}",
        "--network", "host",
        "--stop-timeout", "5",
        "-e", "HOME=/tmp",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PYTHONPATH=/home/scene_graph/scene_graph/src",
        "-v", f"{project_root}:/home/scene_graph/scene_graph:ro",
        "-v", f"{run_dir}:{run_dir}:ro",
        "--workdir", "/home/scene_graph/scene_graph",
        "--entrypoint", "/home/scene_graph/.venv/bin/python",
        image_id,
        "/home/scene_graph/scene_graph/scripts/view_scene_state.py",
        *host_command[2:],
    ]


def serve_viewer(
    run_dir: Path,
    *,
    host: str | None = None,
    port: int | None = None,
    runtime: str = "auto",
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not (run_dir / "_SUCCESS.json").is_file():
        raise ViewerError("Refusing to serve an incomplete or failed run")
    bundle = _load_json(run_dir / "viewer" / "bundle.json")
    process_cfg = bundle.get("process")
    if not isinstance(process_cfg, Mapping):
        raise ViewerError("This run has no configured viewer process")
    existing = viewer_status(run_dir)
    if existing.get("alive"):
        raise ViewerError(f"Viewer is already running with PID {existing.get('pid')}")
    selected_host = host or str(process_cfg.get("default_host", "127.0.0.1"))
    selected_port = port if port is not None else int(process_cfg.get("default_port", 8080))
    if not 1 <= selected_port <= 65535:
        raise ViewerError("Viewer port must be in [1, 65535]")
    if not port_is_available(selected_host, selected_port):
        raise ViewerError(f"{selected_host}:{selected_port} is already in use; no process was changed")

    context = _bundle_context(run_dir, bundle, selected_host, selected_port)
    raw_command = process_cfg.get("command")
    if not isinstance(raw_command, list) or not raw_command or any(not isinstance(item, str) for item in raw_command):
        raise ViewerError("viewer process command must be a non-empty argv list")
    command = [render_template(item, context) for item in raw_command]
    if runtime not in {"auto", "host", "docker"}:
        raise ViewerError("Viewer runtime must be one of: auto, host, docker")
    configured_runtime = str(process_cfg.get("runtime", "auto"))
    if configured_runtime not in {"auto", "host", "docker"}:
        raise ViewerError("viewer process runtime must be one of: auto, host, docker")
    effective_runtime = runtime
    if effective_runtime == "auto":
        effective_runtime = configured_runtime
    if effective_runtime == "auto":
        effective_runtime = "host" if importlib.util.find_spec("viser") is not None else "docker"
    if effective_runtime == "docker":
        command = _dockerize_viewer_command(run_dir, context, command)
    cwd = resolve_plan_path(render_template(str(process_cfg.get("cwd", "${project_root}")), context), Path(context["project_root"]))
    if not cwd.is_dir():
        raise ViewerError(f"Viewer cwd does not exist: {cwd}")
    raw_env = process_cfg.get("env", {})
    if not isinstance(raw_env, Mapping):
        raise ViewerError("viewer process env must be a mapping")
    env = dict(os.environ)
    env.update({str(key): render_template(str(value), context) for key, value in raw_env.items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    snapshot_src = Path(context["project_root"]) / "src"
    if snapshot_src.is_dir():
        configured_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(snapshot_src) + (
            os.pathsep + configured_pythonpath if configured_pythonpath else ""
        )

    runtime_dir = viewer_runtime_dir(run_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = runtime_dir / "stdout.log", runtime_dir / "stderr.log"
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
        process = open_detached_process(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    metadata = {
        "schema": "farm.viewer-process.v1", "status": "starting", "run_dir": str(run_dir),
        "pid": process.pid, "process_start_token": process_start_token(process.pid),
        "process_group_id": os.getpgid(process.pid), "host": selected_host, "port": selected_port,
        "started_at": utc_now(), "command": command, "cwd": str(cwd),
        "runtime": effective_runtime, "env_keys": sorted(raw_env),
        "stdout": str(stdout_path), "stderr": str(stderr_path),
    }
    metadata_path = runtime_dir / "process.json"
    atomic_write_json(metadata_path, metadata)
    timeout = float(process_cfg.get("startup_timeout_seconds", 15.0))
    raw_health = process_cfg.get("health_url")
    if raw_health:
        health_url = render_template(str(raw_health), context)
        healthy = wait_for_http(health_url, process, timeout)
    else:
        time.sleep(min(0.5, timeout))
        health_url, healthy = None, process.poll() is None
    if not healthy:
        if process.poll() is None:
            _terminate(metadata, min(5.0, timeout))
        metadata.update({"status": "failed", "failed_at": utc_now(), "health_url": health_url})
        atomic_write_json(metadata_path, metadata)
        raise ViewerError(f"Viewer failed its startup check; inspect {stderr_path}")
    metadata.update({"status": "running", "ready_at": utc_now(), "health_url": health_url})
    atomic_write_json(metadata_path, metadata)
    return {**metadata, "metadata_path": str(metadata_path)}


def stop_viewer(run_dir: Path, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    path, metadata = _viewer_metadata(run_dir.resolve())
    if not metadata:
        return {"status": "not-started", "metadata_path": str(path)}
    if not process_identity_matches(int(metadata.get("pid", 0) or 0), metadata.get("process_start_token")):
        metadata.update({"status": "stale", "observed_at": utc_now()})
        atomic_write_json(path, metadata)
        return {**metadata, "metadata_path": str(path)}
    _terminate(metadata, timeout_seconds)
    metadata.update({"status": "stopped", "stopped_at": utc_now()})
    atomic_write_json(path, metadata)
    return {**metadata, "metadata_path": str(path)}
