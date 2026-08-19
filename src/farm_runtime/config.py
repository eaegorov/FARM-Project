from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TEMPLATE_RE = re.compile(r"\$\{([A-Za-z0-9_.:-]+)\}")


class PlanError(ValueError):
    """Raised when a pipeline plan is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class Stage:
    id: str
    command: tuple[str, ...]
    needs: tuple[str, ...] = ()
    description: str = ""
    cwd: str = "${project_root}"
    env: Mapping[str, str] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    pass_json: tuple[str, ...] = ()
    fingerprint_inputs: tuple[str, ...] = ()
    fingerprint_mode: str = "metadata"
    timeout_seconds: float | None = None
    enabled: bool = True


@dataclass(frozen=True)
class Viewer:
    command: tuple[str, ...]
    runtime: str = "auto"
    cwd: str = "${project_root}"
    env: Mapping[str, str] = field(default_factory=dict)
    default_host: str = "127.0.0.1"
    default_port: int = 8080
    startup_timeout_seconds: float = 15.0
    health_url: str | None = None


@dataclass(frozen=True)
class Finalizer:
    command: tuple[str, ...]
    cwd: str = "${project_root}"
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class Plan:
    source_path: Path
    raw: Mapping[str, Any]
    scene_id: str
    project_root: Path
    output_root: Path
    stages: tuple[Stage, ...]
    stage_order: tuple[str, ...]
    artifacts: Mapping[str, str]
    viewer: Viewer | None
    finalizers: tuple[Finalizer, ...] = ()
    telemetry_interval_seconds: float = 1.0
    config_sha256: str = ""

    @property
    def stages_by_id(self) -> Mapping[str, Stage]:
        return {stage.id: stage for stage in self.stages}

    def base_context(self, run_dir: Path) -> dict[str, str]:
        context = _flatten_scalars(self.raw)
        context.update({
            "config_dir": str(self.source_path.parent),
            "config_path": str(self.source_path),
            "project_root": str(self.project_root),
            "output_root": str(self.output_root),
            "run_dir": str(run_dir),
            "scene_id": self.scene_id,
            "python_executable": sys.executable,
        })
        return context

    def resolved_artifacts(self, run_dir: Path) -> dict[str, Path]:
        context = self.base_context(run_dir)
        resolved: dict[str, Path] = {}
        for name, value in self.artifacts.items():
            rendered = render_template(value, context)
            resolved[name] = resolve_plan_path(rendered, self.project_root)
            context[f"artifact:{name}"] = str(resolved[name])
            context[f"artifacts.{name}"] = str(resolved[name])
        return resolved


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_template(value: str, context: Mapping[str, str], *, environ: Mapping[str, str] | None = None) -> str:
    """Resolve explicit ``${name}`` placeholders without invoking a shell."""

    environ = os.environ if environ is None else environ

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key.startswith("env:"):
            env_key = key[4:]
            if env_key not in environ:
                raise PlanError(f"Required environment variable is not set: {env_key}")
            return environ[env_key]
        if key not in context:
            raise PlanError(f"Unknown template placeholder: ${{{key}}}")
        return str(context[key])

    rendered = _TEMPLATE_RE.sub(replace, value)
    unresolved = _TEMPLATE_RE.search(rendered)
    if unresolved:
        raise PlanError(f"Unresolved template placeholder: {unresolved.group(0)}")
    return rendered


def resolve_plan_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _flatten_scalars(value: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_scalars(item, dotted))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[dotted] = "" if item is None else str(item)
    return result


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PlanError(f"{label} must be a mapping")
    return value


def _expect_string_list(value: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PlanError(f"{label} must be a list of strings")
    else:
        result = tuple(value)
    if nonempty and not result:
        raise PlanError(f"{label} must not be empty")
    return result


def _reject_environment_templates(values: Iterable[str], label: str) -> None:
    if any("${env:" in value for value in values):
        raise PlanError(
            f"{label} must not contain environment secrets; pass them through the dedicated env mapping"
        )


def _validate_env_mapping(env_cfg: Mapping[str, Any], label: str) -> None:
    if any(not isinstance(key, str) or not isinstance(item, (str, int, float, bool)) for key, item in env_cfg.items()):
        raise PlanError(f"{label} keys and values must be scalar strings")
    secret_suffixes = ("token", "password", "passwd", "api_key", "apikey", "secret")
    for key, item in env_cfg.items():
        value = str(item)
        if key.lower().endswith(secret_suffixes) and not re.fullmatch(r"\$\{env:[A-Za-z_][A-Za-z0-9_]*\}", value):
            raise PlanError(f"{label}.{key} must reference an environment variable, not contain a literal secret")


def _discover_project_root(source_path: Path) -> Path:
    for candidate in (source_path.parent, *source_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return source_path.parent


def _parse_stage(value: Any, index: int) -> Stage:
    cfg = _expect_mapping(value, f"pipeline.stages[{index}]")
    stage_id = cfg.get("id")
    if not isinstance(stage_id, str) or not _ID_RE.fullmatch(stage_id):
        raise PlanError(f"pipeline.stages[{index}].id must match {_ID_RE.pattern}")
    command = _expect_string_list(cfg.get("command"), f"stage {stage_id}.command", nonempty=True)
    needs = _expect_string_list(cfg.get("needs", []), f"stage {stage_id}.needs")
    inputs = _expect_string_list(cfg.get("inputs", []), f"stage {stage_id}.inputs")
    outputs = _expect_string_list(cfg.get("outputs", []), f"stage {stage_id}.outputs")
    pass_json = _expect_string_list(cfg.get("pass_json", []), f"stage {stage_id}.pass_json")
    fingerprint_inputs = _expect_string_list(
        cfg.get("fingerprint_inputs", list(inputs)), f"stage {stage_id}.fingerprint_inputs"
    )
    _reject_environment_templates(command, f"stage {stage_id}.command")
    _reject_environment_templates((*inputs, *outputs, *pass_json, *fingerprint_inputs), f"stage {stage_id} paths")
    env_cfg = _expect_mapping(cfg.get("env", {}), f"stage {stage_id}.env")
    _validate_env_mapping(env_cfg, f"stage {stage_id}.env")
    mode = str(cfg.get("fingerprint_mode", "metadata"))
    if mode not in {"metadata", "content"}:
        raise PlanError(f"stage {stage_id}.fingerprint_mode must be metadata or content")
    timeout = cfg.get("timeout_seconds")
    if timeout is not None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise PlanError(f"stage {stage_id}.timeout_seconds must be positive")
        timeout = float(timeout)
    cwd = cfg.get("cwd", "${project_root}")
    if not isinstance(cwd, str):
        raise PlanError(f"stage {stage_id}.cwd must be a string")
    _reject_environment_templates((cwd,), f"stage {stage_id}.cwd")
    description = cfg.get("description", "")
    if not isinstance(description, str):
        raise PlanError(f"stage {stage_id}.description must be a string")
    enabled = cfg.get("enabled", True)
    if not isinstance(enabled, bool):
        raise PlanError(f"stage {stage_id}.enabled must be boolean")
    return Stage(
        id=stage_id,
        command=command,
        needs=needs,
        description=description,
        cwd=cwd,
        env={str(key): str(item) for key, item in env_cfg.items()},
        inputs=inputs,
        outputs=outputs,
        pass_json=pass_json,
        fingerprint_inputs=fingerprint_inputs,
        fingerprint_mode=mode,
        timeout_seconds=timeout,
        enabled=enabled,
    )


def _topological_order(stages: Iterable[Stage]) -> tuple[str, ...]:
    stages = tuple(stages)
    by_id = {stage.id: stage for stage in stages}
    if len(by_id) != len(stages):
        duplicates = sorted(stage.id for stage in stages if sum(item.id == stage.id for item in stages) > 1)
        raise PlanError(f"Duplicate stage ids: {', '.join(sorted(set(duplicates)))}")
    enabled = {stage.id for stage in stages if stage.enabled}
    for stage in stages:
        if not stage.enabled:
            continue
        missing = [dep for dep in stage.needs if dep not in by_id]
        if missing:
            raise PlanError(f"Stage {stage.id} has unknown dependencies: {', '.join(missing)}")
        disabled = [dep for dep in stage.needs if dep not in enabled]
        if disabled:
            raise PlanError(f"Stage {stage.id} depends on disabled stages: {', '.join(disabled)}")

    indegree = {stage_id: 0 for stage_id in enabled}
    successors: dict[str, list[str]] = {stage_id: [] for stage_id in enabled}
    position = {stage.id: idx for idx, stage in enumerate(stages)}
    for stage in stages:
        if stage.id not in enabled:
            continue
        for dep in stage.needs:
            indegree[stage.id] += 1
            successors[dep].append(stage.id)

    ready = sorted((stage_id for stage_id, degree in indegree.items() if degree == 0), key=position.get)
    order: list[str] = []
    while ready:
        stage_id = ready.pop(0)
        order.append(stage_id)
        for successor in sorted(successors[stage_id], key=position.get):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort(key=position.get)
    if len(order) != len(enabled):
        cyclic = sorted(stage_id for stage_id, degree in indegree.items() if degree > 0)
        raise PlanError(f"Stage dependency cycle detected: {', '.join(cyclic)}")
    return tuple(order)


def _parse_viewer(value: Any) -> Viewer | None:
    if value is None:
        return None
    cfg = _expect_mapping(value, "viewer")
    command = _expect_string_list(cfg.get("command"), "viewer.command", nonempty=True)
    _reject_environment_templates(command, "viewer.command")
    for placeholder in ("${host}", "${port}"):
        if not any(placeholder in token for token in command):
            raise PlanError(f"viewer.command must contain {placeholder} so CLI overrides are reliable")
    cwd = cfg.get("cwd", "${project_root}")
    if not isinstance(cwd, str):
        raise PlanError("viewer.cwd must be a string")
    _reject_environment_templates((cwd,), "viewer.cwd")
    env_cfg = _expect_mapping(cfg.get("env", {}), "viewer.env")
    _validate_env_mapping(env_cfg, "viewer.env")
    runtime = cfg.get("runtime", "auto")
    if runtime not in {"auto", "host", "docker"}:
        raise PlanError("viewer.runtime must be one of: auto, host, docker")
    host = cfg.get("default_host", "127.0.0.1")
    port = cfg.get("default_port", 8080)
    timeout = cfg.get("startup_timeout_seconds", 15.0)
    health_url = cfg.get("health_url")
    if not isinstance(host, str) or not host:
        raise PlanError("viewer.default_host must be a non-empty string")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise PlanError("viewer.default_port must be an integer in [1, 65535]")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise PlanError("viewer.startup_timeout_seconds must be positive")
    if health_url is not None and not isinstance(health_url, str):
        raise PlanError("viewer.health_url must be a string")
    return Viewer(
        command=command,
        runtime=runtime,
        cwd=cwd,
        env={str(key): str(item) for key, item in env_cfg.items()},
        default_host=host,
        default_port=port,
        startup_timeout_seconds=float(timeout),
        health_url=health_url,
    )


def _parse_finalizer(value: Any, index: int) -> Finalizer:
    cfg = _expect_mapping(value, f"pipeline.finalizers[{index}]")
    command = _expect_string_list(
        cfg.get("command"), f"pipeline.finalizers[{index}].command", nonempty=True
    )
    _reject_environment_templates(command, f"pipeline.finalizers[{index}].command")
    cwd = cfg.get("cwd", "${project_root}")
    if not isinstance(cwd, str):
        raise PlanError(f"pipeline.finalizers[{index}].cwd must be a string")
    _reject_environment_templates((cwd,), f"pipeline.finalizers[{index}].cwd")
    env_cfg = _expect_mapping(cfg.get("env", {}), f"pipeline.finalizers[{index}].env")
    _validate_env_mapping(env_cfg, f"pipeline.finalizers[{index}].env")
    timeout = cfg.get("timeout_seconds", 60.0)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise PlanError(f"pipeline.finalizers[{index}].timeout_seconds must be positive")
    return Finalizer(
        command=command,
        cwd=cwd,
        env={str(key): str(item) for key, item in env_cfg.items()},
        timeout_seconds=float(timeout),
    )


def load_plan(config_path: str | Path, *, project_root: str | Path | None = None) -> Plan:
    source_path = Path(config_path).expanduser().resolve()
    if not source_path.is_file():
        raise PlanError(f"Pipeline config does not exist: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    raw = _expect_mapping(loaded, "config")
    raw = copy.deepcopy(dict(raw))
    if raw.get("schema_version") == "farm.scene.v1":
        from .standard import compile_standard_scene

        raw = compile_standard_scene(source_path, project_root=project_root)
        project_root = raw["project_root"]
    version = raw.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise PlanError(f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
    scene_cfg = _expect_mapping(raw.get("scene"), "scene")
    scene_id = scene_cfg.get("id")
    if not isinstance(scene_id, str) or not _ID_RE.fullmatch(scene_id):
        raise PlanError(f"scene.id must match {_ID_RE.pattern}")

    root_value = project_root if project_root is not None else raw.get("project_root", _discover_project_root(source_path))
    root_path = Path(str(root_value)).expanduser()
    if not root_path.is_absolute():
        root_path = source_path.parent / root_path
    root_path = root_path.resolve(strict=False)

    output_cfg = _expect_mapping(raw.get("output"), "output")
    output_value = output_cfg.get("root")
    if not isinstance(output_value, str) or not output_value:
        raise PlanError("output.root must be a non-empty path string")
    seed_context = _flatten_scalars(raw)
    seed_context.update({
        "config_dir": str(source_path.parent),
        "config_path": str(source_path),
        "project_root": str(root_path),
        "scene_id": scene_id,
        "python_executable": sys.executable,
    })
    output_root = resolve_plan_path(render_template(output_value, seed_context), root_path)

    pipeline_cfg = _expect_mapping(raw.get("pipeline"), "pipeline")
    stages_value = pipeline_cfg.get("stages")
    if not isinstance(stages_value, list) or not stages_value:
        raise PlanError("pipeline.stages must be a non-empty list")
    stages = tuple(_parse_stage(item, idx) for idx, item in enumerate(stages_value))
    order = _topological_order(stages)
    interval = pipeline_cfg.get("telemetry_interval_seconds", 1.0)
    if not isinstance(interval, (int, float)) or not 0.1 <= interval <= 60.0:
        raise PlanError("pipeline.telemetry_interval_seconds must be between 0.1 and 60 seconds")
    finalizers_value = pipeline_cfg.get("finalizers", [])
    if not isinstance(finalizers_value, list):
        raise PlanError("pipeline.finalizers must be a list")
    finalizers = tuple(_parse_finalizer(item, index) for index, item in enumerate(finalizers_value))

    artifacts_cfg = _expect_mapping(raw.get("artifacts", {}), "artifacts")
    if any(not isinstance(key, str) or not _ID_RE.fullmatch(key) or not isinstance(item, str) for key, item in artifacts_cfg.items()):
        raise PlanError("artifacts must map safe identifiers to path strings")

    raw_bytes = source_path.read_bytes()
    return Plan(
        source_path=source_path,
        raw=raw,
        scene_id=scene_id,
        project_root=root_path,
        output_root=output_root,
        stages=stages,
        stage_order=order,
        artifacts=dict(artifacts_cfg),
        viewer=_parse_viewer(raw.get("viewer")),
        finalizers=finalizers,
        telemetry_interval_seconds=float(interval),
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def plan_to_public_dict(plan: Plan, run_dir: Path) -> dict[str, Any]:
    """Build a resolved, secret-safe plan representation for the run bundle."""

    context = plan.base_context(run_dir)
    artifacts = plan.resolved_artifacts(run_dir)
    for name, path in artifacts.items():
        context[f"artifact:{name}"] = str(path)
        context[f"artifacts.{name}"] = str(path)
    stages: list[dict[str, Any]] = []
    for stage_id in plan.stage_order:
        stage = plan.stages_by_id[stage_id]
        stages.append(
            {
                "id": stage.id,
                "description": stage.description,
                "needs": list(stage.needs),
                "command": [render_template(item, context) for item in stage.command],
                "cwd": render_template(stage.cwd, context),
                "env_keys": sorted(stage.env),
                "inputs": [render_template(item, context) for item in stage.inputs],
                "outputs": [render_template(item, context) for item in stage.outputs],
                "pass_json": [render_template(item, context) for item in stage.pass_json],
                "fingerprint_inputs": [render_template(item, context) for item in stage.fingerprint_inputs],
                "fingerprint_mode": stage.fingerprint_mode,
                "timeout_seconds": stage.timeout_seconds,
            }
        )
    return {
        "schema": "farm.pipeline-plan.v1",
        "scene_id": plan.scene_id,
        "source_config": str(plan.source_path),
        "config_sha256": plan.config_sha256,
        "project_root": str(plan.project_root),
        "output_root": str(plan.output_root),
        "run_dir": str(run_dir),
        "stage_order": list(plan.stage_order),
        "stages": stages,
        "finalizers": [
            {
                "command": [render_template(item, context) for item in finalizer.command],
                "cwd": render_template(finalizer.cwd, context),
                "env_keys": sorted(finalizer.env),
                "timeout_seconds": finalizer.timeout_seconds,
            }
            for finalizer in plan.finalizers
        ],
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "viewer_configured": plan.viewer is not None,
    }
