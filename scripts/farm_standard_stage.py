#!/usr/bin/env python3
"""Execute one fail-closed stage of the canonical farm.scene.v1 DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

# The wrapper itself lives inside the signed execution snapshot.  Disable
# bytecode before importing any first-party module so neither this process nor
# inherited host-side Python children can add __pycache__ entries to that tree.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path[:] = [entry for entry in sys.path if entry != str(SRC)]
sys.path.insert(0, str(SRC))

from farm_pipeline.resources import load_model_manifest  # noqa: E402
from farm_pipeline.scene_config import load_scene_config  # noqa: E402
from farm_runtime.process import atomic_write_json, atomic_write_text, utc_now  # noqa: E402
from farm_runtime.source_snapshot import validated_source_snapshot_project_root  # noqa: E402
from farm_runtime.standard import STANDARD_STAGE_IDS  # noqa: E402

RESULT_PATHS = {
    "preflight": "input/preflight.json", "selection": "selection/result.json",
    "rgbd": "rgbd/result.json", "mapping": "mapping/result.json",
    "mapping_qa": "qa/mapping/result.json",
    "presentation": "mapping/presentation_result.json",
    "geometry": "qa/geometry/result.json",
    "visual_consistency": "qa/visual_consistency/result.json",
    "semantics": "qa/semantics/result.json",
    "part_whole": "qa/part_whole/result.json",
    "assemblies": "qa/assemblies/result.json",
    "surface_support": "qa/surface_support/result.json",
    "compound_geometry": "qa/compound_geometry/result.json",
    "geometry_qa": "qa/geometry_qa/result.json",
    "dedup": "qa/dedup/result.json", "finalize": "final/result.json",
    "qa_bundle": "qa/result.json",
}
OWNER_LABEL = "com.goldengait.farm.standard"
RUN_LABEL = "com.goldengait.farm.standard.run"


def run(command: Sequence[str], cwd: Path = ROOT) -> None:
    print("+ " + " ".join(map(str, command)), flush=True)
    result = subprocess.run(list(command), cwd=str(cwd), check=False, shell=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def validate_canonical_visuals(paths: Sequence[Path]) -> None:
    """Decode canonical images and enforce the ``_4k`` filename contract."""
    errors: list[str] = []
    for path in paths:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            errors.append(f"unsupported image extension: {path}")
            continue
        try:
            with Image.open(path) as image:
                image.load()
                size = image.size
        except Exception as exc:  # noqa: BLE001 - report every decoder failure uniformly
            errors.append(f"cannot decode {path}: {exc}")
            continue
        if "_4k" in path.stem and size != (3840, 2160):
            errors.append(f"{path} is {size[0]}x{size[1]}, expected 3840x2160")
    if errors:
        raise RuntimeError("canonical visualization contract failed: " + "; ".join(errors))


def replace_directory(source: Path, destination: Path) -> None:
    """Promote a completed stage-owned directory without exposing partial data."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.{os.getpid()}.old")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def resolve_model_contract(config: Any, models: Any) -> dict[str, str]:
    """Resolve configured model IDs against the pinned manifest, fail closed."""

    segmentation = str(config.models.segmentation)
    segmentation_candidates = (segmentation, f"{segmentation}-seg")
    segmentation_key = next(
        (key for key in segmentation_candidates if key in models.pipeline_models), None
    )
    if segmentation_key is None:
        raise ValueError(
            f"configured segmentation model {segmentation!r} is not pinned by the model manifest"
        )
    visual_key = str(config.models.visual_features)
    if visual_key not in models.pipeline_models or not visual_key.startswith("dinov3-"):
        raise ValueError(
            f"configured visual_features model {visual_key!r} is not a pinned DINO model"
        )
    caption_repo = str(models.services["caption"].model.repo_id or "")
    if str(config.models.caption) != caption_repo:
        raise ValueError(
            f"configured caption model {config.models.caption!r} != pinned {caption_repo!r}"
        )
    text_repo = str(models.services["text-embed"].model.repo_id or "")
    if config.models.text_embeddings and str(config.models.text_embeddings) != text_repo:
        raise ValueError(
            f"configured text_embeddings model {config.models.text_embeddings!r} != pinned {text_repo!r}"
        )
    return {
        "segmentation": segmentation,
        "segmentation_manifest_key": segmentation_key,
        "visual_features": visual_key,
        "caption": caption_repo,
        "text_embeddings": text_repo,
        "vl_embeddings": str(models.services["vl-embed"].model.repo_id or ""),
    }


def _scope_for_run(config: Any, run_dir: Path) -> tuple[Path, str]:
    resolved = run_dir.resolve(strict=True)
    manifest = load_json(resolved / "manifest.json")
    if manifest.get("scene_id") != config.scene_id:
        raise ValueError("scene config does not match run manifest")
    runs_root = (config.output_root / config.scene_id / "runs").resolve()
    resolved.relative_to(runs_root)
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()
    return resolved, "r" + digest[:24]


def _validate_execution_contract(config_path: Path, run_dir: Path) -> Path:
    """Bind this process to the run's signed source tree and config bytes."""

    resolved_run = run_dir.resolve(strict=True)
    manifest = load_json(resolved_run / "manifest.json")
    expected_config_sha256 = str(manifest.get("config_sha256") or "").lower()
    actual_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if actual_config_sha256 != expected_config_sha256:
        raise RuntimeError("scene config SHA-256 does not match run manifest")
    snapshot_root = validated_source_snapshot_project_root(
        resolved_run, manifest, required=True
    )
    if snapshot_root is None or ROOT.resolve(strict=True) != snapshot_root:
        raise RuntimeError(
            "standard stage must execute from the run's validated source snapshot"
        )
    return resolved_run


def cleanup_scope(config_path: Path, run_dir: Path) -> None:
    """Stop only the exact run scope without loading model files/manifest."""

    config_path = config_path.resolve(strict=True)
    _validate_execution_contract(config_path, run_dir)
    config = load_scene_config(config_path)
    resolved, scope = _scope_for_run(config, run_dir)
    manifest_path = config.resources.model_manifest or ROOT / "configs/models/farm_models.v1.json"
    subprocess.run(
        [str(ROOT / "scripts/start_farm_vllm.sh"), "stop", "all", "--scope", scope,
         "--manifest", str(manifest_path)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False, shell=False,
    )
    found = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={OWNER_LABEL}=true",
         "--filter", f"label={RUN_LABEL}={scope}"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        check=False, shell=False,
    )
    ids = found.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, shell=False)
    runtime_dir = resolved / ".runtime"
    (runtime_dir / "vllm.json").unlink(missing_ok=True)
    try:
        runtime_dir.rmdir()
    except OSError:
        pass
    _validate_execution_contract(config_path, run_dir)


class Context:
    def __init__(self, config_path: Path, run_dir: Path, stage: str | None) -> None:
        self.config_path = config_path.resolve(strict=True)
        _validate_execution_contract(self.config_path, run_dir)
        self.config = load_scene_config(self.config_path)
        self.run_dir, self.scope = _scope_for_run(self.config, run_dir)
        self.stage = stage
        digest = hashlib.sha256(str(self.run_dir).encode()).hexdigest()
        self.prefix = "farm-" + digest[:16]
        self.runtime_dir = self.run_dir / ".runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.context_path = self.run_dir / "input" / "resolved_context.json"
        self.manifest_path = self.config.resources.model_manifest or ROOT / "configs/models/farm_models.v1.json"
        self.models = load_model_manifest(self.manifest_path)
        self.main_image = self.models.runtimes["main"].image_id
        self.prep_image = self.models.runtimes["prep"].image_id
        self.prep_python = self.models.runtimes["prep"].python
        if self.prep_python is None:
            raise RuntimeError("runtime 'prep' has no pinned Python interpreter")
        self.counter = 0

    @property
    def result_path(self) -> Path:
        assert self.stage is not None
        return self.run_dir / RESULT_PATHS[self.stage]

    def resolved(self) -> dict[str, Any]:
        return load_json(self.context_path)

    def gpu(self) -> str:
        if self.context_path.is_file():
            return str(self.resolved()["selected_gpu"]["uuid"])
        return self.config.resources.gpu

    def result(self, status: str, **data: Any) -> None:
        atomic_write_json(self.result_path, {
            "schema": "farm.standard-stage.v1", "status": status,
            "stage": self.stage, "scene_id": self.config.scene_id,
            "run_id": self.run_dir.name, "updated_at": utc_now(),
            "selected_gpu": self.resolved().get("selected_gpu") if self.context_path.is_file() else None,
            **data,
        })

    def name(self, purpose: str) -> str:
        self.counter += 1
        safe = "".join(c if c.isalnum() else "-" for c in purpose.lower())[:20]
        return f"{self.prefix}-{safe}-{self.counter:02d}"

    def docker_base(
        self,
        purpose: str,
        gpu: bool = False,
        host_network: bool = False,
        runtime: str | None = None,
    ) -> tuple[list[str], str]:
        name = self.name(purpose)
        user_uid = os.getuid()
        if runtime is not None:
            spec = self.models.runtimes[runtime]
            if spec.user_uid is None:
                raise RuntimeError(f"runtime {runtime!r} has no declared non-root user_uid")
            user_uid = int(spec.user_uid)
        command = [
            "docker", "run", "--rm", "--name", name,
            "--label", f"{OWNER_LABEL}=true", "--label", f"{RUN_LABEL}={self.scope}",
            "--user", f"{user_uid}:{os.getgid()}", "--ipc", "host", "--shm-size", "16g",
            "--network", "host" if host_network else "none",
            "-e", "FARM_RUN_DIR=/farm-run",
        ]
        if gpu:
            command += ["--gpus", f"device={self.gpu()}"]
        return command, name

    def prepare_main_mount(self) -> None:
        """Make existing run directories writable by the main image host GID."""

        # The source snapshot is an immutable, mode-signed input to the
        # viewer. Changing even a group-write bit invalidates its tree hash.
        # Runtime outputs still need cooperative group permissions, but the
        # exact snapshot subtree must never participate in that normalization.
        snapshot_root = self.run_dir / "config/source_snapshot"
        for path in (self.run_dir, *self.run_dir.rglob("*")):
            if (
                path == snapshot_root
                or snapshot_root in path.parents
                or path.is_symlink()
            ):
                continue
            info = path.stat()
            # NVIDIA Container Toolkit may leave root-owned transient hook
            # directories beside a bind mount. They are not pipeline outputs
            # and must neither block nor broaden our permission changes.
            if info.st_uid != os.getuid():
                continue
            mode = stat.S_IMODE(info.st_mode)
            if path.is_dir():
                required = stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP
            elif path.is_file():
                required = stat.S_IRGRP | stat.S_IWGRP
            else:
                continue
            if mode & required == required:
                continue
            os.chmod(path, mode | required)

    def model_mounts(self) -> list[str]:
        """Bind resolved model targets, not potentially broken repo symlinks."""

        arguments: list[str] = [
            "-e", "SCENE_GRAPH_MODEL_DIR=/farm-models",
            "-e", "HF_HOME=/farm-cache/huggingface",
            "-e", "HUGGINGFACE_HUB_CACHE=/farm-cache/huggingface/hub",
        ]
        mounted: set[tuple[Path, str]] = set()
        for key, spec in self.models.pipeline_models.items():
            source = spec.local_path.resolve(strict=True).parent
            alias = "yoloe" if key.startswith("yoloe-") else key
            target = f"/farm-models/{alias}"
            item = (source, target)
            if item not in mounted:
                arguments += ["-v", f"{source}:{target}:ro"]
                mounted.add(item)
        cache = self.models.cache_root.resolve(strict=True)
        arguments += ["-v", f"{cache}:/farm-cache/huggingface:ro"]
        return arguments

    def post(self, script: str, arguments: Sequence[str], gpu: bool = False, host_network: bool = False) -> None:
        self.prepare_main_mount()
        command, _ = self.docker_base(script, gpu, host_network, runtime="main")
        command += self.model_mounts()
        command += [
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            "-v", f"{ROOT}:/home/scene_graph/scene_graph:ro",
            "-v", f"{self.run_dir}:/farm-run",
            "-v", f"{self.config.inputs.gaussian_ply}:/input/scene.ply:ro",
            "--entrypoint", "/bin/bash", self.main_image,
            "/home/scene_graph/scene_graph/docker/python-entrypoint.sh",
            f"/home/scene_graph/scene_graph/scripts/{script}", *map(str, arguments),
        ]
        run(command)

    def prep_post(self, script: str, arguments: Sequence[str]) -> None:
        """Run CPU post-processing that depends on the pinned prep closure."""

        command, _ = self.docker_base(script)
        command += [
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            "-v", f"{ROOT}:/project:ro", "-v", f"{self.run_dir}:/farm-run",
            "-v", f"{self.config.inputs.gaussian_ply}:/input/scene.ply:ro",
            "--entrypoint", self.prep_python, self.prep_image,
            f"/project/scripts/{script}", *map(str, arguments),
        ]
        run(command)

    def cleanup_containers(self) -> None:
        found = subprocess.run([
            "docker", "ps", "-aq", "--filter", f"label={OWNER_LABEL}=true",
            "--filter", f"label={RUN_LABEL}={self.scope}",
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False, shell=False)
        ids = found.stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, shell=False)

    def cleanup_stage_work(self, stage: str) -> None:
        """Remove a successfully published stage workspace, retaining failed work for resume."""

        path = self.run_dir / ".stage_work" / stage
        if path.exists():
            shutil.rmtree(path)
        try:
            path.parent.rmdir()
        except OSError:
            pass

    def pre_qa_wall_seconds(self) -> float:
        """Sum completed pre-finalize stage wall time for the bounded QA budget."""

        total = 0.0
        for stage_id in STANDARD_STAGE_IDS:
            if stage_id in {"finalize", "qa_bundle"}:
                continue
            path = self.run_dir / "stages" / stage_id / "state.json"
            if not path.is_file():
                continue
            value = load_json(path).get("duration_seconds")
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(seconds) and seconds >= 0.0:
                total += seconds
        return total

    def lease_path(self) -> Path:
        return self.runtime_dir / "vllm.json"

    def service_port(self) -> int:
        if self.lease_path().is_file():
            lease = load_json(self.lease_path())
            if lease.get("scope") == self.scope:
                return int(lease["port"])
        seed = int(hashlib.sha256(self.scope.encode()).hexdigest()[:8], 16)
        for offset in range(1000):
            port = 18000 + (seed + offset) % 10000
            with socket.socket() as test:
                try:
                    test.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise RuntimeError("no free scoped vLLM port")

    @staticmethod
    def ready(port: int, model: str) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                value = json.loads(response.read())
            return model in {str(row.get("id")) for row in value.get("data", []) if isinstance(row, dict)}
        except Exception:
            return False

    def service(self, service: str) -> tuple[int, str]:
        spec = self.models.services[service]
        port = self.service_port()
        owned = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=com.goldengait.farm.vllm=true",
             "--filter", f"label=com.goldengait.farm.vllm.scope={self.scope}",
             "--filter", f"label=com.goldengait.farm.vllm.service={service}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            check=False, shell=False,
        ).stdout.split()
        if owned and self.ready(port, spec.served_model_name):
            return port, spec.served_model_name
        if self.lease_path().is_file():
            self.stop_services()
            port = self.service_port()
        command = [
            str(ROOT / "scripts/start_farm_vllm.sh"), "start", service,
            "--gpu", self.gpu(), "--scope", self.scope,
            "--manifest", str(self.manifest_path), "--host", "127.0.0.1",
            "--port", str(port),
        ]
        if self.config.resources.secrets_file:
            command += ["--secrets-file", str(self.config.resources.secrets_file)]
        run(command)
        if not self.ready(port, spec.served_model_name):
            raise RuntimeError(f"{service} readiness model-ID check failed")
        atomic_write_json(self.lease_path(), {
            "schema": "farm.scoped-service.v1", "scope": self.scope,
            "service": service, "port": port, "model": spec.served_model_name,
        })
        return port, spec.served_model_name

    def stop_services(self) -> None:
        subprocess.run([
            str(ROOT / "scripts/start_farm_vllm.sh"), "stop", "all",
            "--scope", self.scope, "--manifest", str(self.manifest_path),
        ], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
           check=False, shell=False)
        self.lease_path().unlink(missing_ok=True)
        try:
            self.runtime_dir.rmdir()
        except OSError:
            pass

    def cleanup(self) -> None:
        self.stop_services()
        self.cleanup_containers()

    def up(self) -> str:
        return ",".join(f"{float(v):.12g}" for v in self.resolved()["resolved_up"])

    def output_views(self) -> list[str]:
        explicit = list(self.config.selection.output_views)
        required = list(self.config.camera_grouping.required_views)
        if explicit:
            return explicit
        if len(required) == 1:
            return required
        if self.config.metric_scale.baseline_view:
            return [self.config.metric_scale.baseline_view]
        if len(required) > 1:
            raise ValueError("configure selection.output_views when required_views has multiple families")
        return []

    def preflight(self) -> None:
        model_contract = resolve_model_contract(self.config, self.models)
        scene_path = self.run_dir / "input/scene_preflight.json"
        resource_path = self.run_dir / "input/resource_preflight.json"
        run([sys.executable, str(ROOT / "scripts/farm_preflight.py"), "--config",
             str(self.config_path), "--report", str(scene_path)])
        command = [
            sys.executable, str(ROOT / "scripts/farm_resource_preflight.py"),
            "--manifest", str(self.manifest_path), "--gpu", self.config.resources.gpu,
            "--services", "caption", "text-embed", "vl-embed", "--all-models",
            "--report", str(resource_path),
        ]
        if self.config.resources.secrets_file:
            command += ["--secrets-file", str(self.config.resources.secrets_file)]
        run(command)
        scene, resources = load_json(scene_path), load_json(resource_path)
        if str(scene.get("status")).lower() == "fail" or not resources.get("ready"):
            raise RuntimeError("scene/resource preflight failed")
        checks = {str(row.get("name")): row for row in scene.get("checks", []) if isinstance(row, dict)}
        metric = checks.get("metric_scale", {}).get("metrics", {})
        scale = metric.get("meters_per_colmap_unit")
        if scale is None:
            scale = metric.get("inferred_meters_per_colmap_unit")
        up = checks.get("gravity", {}).get("metrics", {}).get("resolved_up")
        gpu = resources.get("selected_gpu")
        if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0:
            raise RuntimeError("full run requires resolved positive metric scale")
        if not isinstance(up, list) or len(up) != 3:
            raise RuntimeError("full run requires resolved world up")
        if not isinstance(gpu, dict) or not gpu.get("uuid"):
            raise RuntimeError("resource preflight did not resolve a GPU UUID")
        atomic_write_json(self.context_path, {
            "schema": "farm.standard-resolved-context.v1",
            "meters_per_scene_unit": float(scale), "resolved_up": list(map(float, up)),
            "selected_gpu": gpu, "model_manifest": str(self.models.manifest_path),
            "model_manifest_sha256": resources.get("manifest_sha256"),
            "model_contract": model_contract,
        })
        self.result("PASS", scene_report="scene_preflight.json", resource_report="resource_preflight.json")

    def selection(self) -> None:
        p, g, views = self.config.selection, self.config.camera_grouping, self.output_views()
        command = [
            sys.executable, str(ROOT / "scripts/select_colmap_keyframes.py"),
            "--colmap", str(self.config.inputs.colmap_model), "--images", str(self.config.inputs.image_root),
            "--output", str(self.run_dir / "selection"), "--scene-id", self.config.scene_id,
            "--meters-per-scene-unit", str(self.resolved()["meters_per_scene_unit"]),
            "--sensor-group", g.member_group, "--timestamp-group", g.timestamp_group,
            "--family-group", g.view_group, "--target-observations", str(p.target_timestamps),
            "--max-observations", str(p.max_timestamps),
            "--rotation-weight-m-per-rad", str(p.rotation_weight_m_per_rad),
            "--max-motion-gap", str(p.max_motion_gap),
            "--max-translation-gap-m", str(p.max_translation_gap_m),
            "--max-rotation-gap-deg", str(p.max_rotation_gap_deg),
            "--min-shared-tracks", str(p.min_shared_tracks),
            "--min-overlap-ratio", str(p.min_overlap_ratio),
        ]
        if g.pattern:
            command += ["--name-regex", g.pattern]
        for value in g.required_members:
            command += ["--sensor", value]
        for value in views:
            command += ["--anchor-family", value, "--output-family", value]
        budget = p.max_timestamps * max(1, len(g.required_members)) * max(1, len(views))
        command += ["--max-output-views", str(max(500, budget))]
        if self.config.metric_scale.expected_baseline_m:
            command += ["--expected-baseline-m", str(self.config.metric_scale.expected_baseline_m),
                        "--baseline-tolerance-m", str(self.config.metric_scale.baseline_tolerance_m)]
            if len(self.config.metric_scale.baseline_members) != 2:
                raise ValueError("metric_scale.baseline_members must name the exact sensor pair")
            command += ["--baseline-sensors", *self.config.metric_scale.baseline_members]
        run(command)
        report = load_json(self.run_dir / "selection/selection_manifest.json")
        if int(report.get("selection", {}).get("residual_flagged_edges", -1)) != 0:
            raise RuntimeError("selected graph has residual flagged edges")
        self.result("PASS", selection_manifest="selection_manifest.json", output_views=views)

    def rgbd(self) -> None:
        g = self.config.camera_grouping
        # Render into a fingerprinted stage-owned workspace.  A killed process
        # resumes the same generic RGB-D manifest, while changed inputs/code
        # start clean; canonical rgbd/ is replaced only after all QA passes.
        state = load_json(self.run_dir / "stages/rgbd/state.json")
        fingerprint = str(state.get("fingerprint", "")).strip()
        if not fingerprint:
            raise RuntimeError("RGB-D stage state has no orchestrator fingerprint")
        work_root = self.run_dir / ".stage_work/rgbd"
        marker = work_root / "orchestrator_fingerprint.txt"
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() != fingerprint:
            shutil.rmtree(work_root)
        work_output = work_root / "output"
        work_output.mkdir(parents=True, exist_ok=True)
        atomic_write_text(marker, fingerprint + "\n")
        command, _ = self.docker_base("rgbd", gpu=True)
        command += [
            "-v", f"{ROOT}:/project:ro", "-v", f"{self.run_dir}:/farm-run",
            "-v", f"{self.config.inputs.colmap_model}:/input/colmap:ro",
            "-v", f"{self.config.inputs.image_root}:/input/images:ro",
            "-v", f"{self.config.inputs.gaussian_ply}:/input/scene.ply:ro",
            "--entrypoint", self.prep_python, self.prep_image,
            "/project/scripts/prepare_colmap_3dgs_rgbd.py", "--colmap-model", "/input/colmap",
            "--image-root", "/input/images", "--ply", "/input/scene.ply",
            "--selected-names", "/farm-run/selection/selected_names.txt",
            "--output-dir", "/farm-run/.stage_work/rgbd/output",
            "--scene-id", self.config.scene_id, "--resolution", str(self.config.resources.render_resolution),
            "--qa-sampled-views", "48",
            "--meters-per-scene-unit", str(self.resolved()["meters_per_scene_unit"]),
            "--sensor-group", g.member_group, "--timestamp-group", g.timestamp_group,
            "--family-group", g.view_group,
        ]
        if g.pattern:
            command += ["--identity-regex", g.pattern]
        if self.config.metric_scale.expected_baseline_m:
            command += ["--expected-baseline-m", str(self.config.metric_scale.expected_baseline_m),
                        "--baseline-tolerance-m", str(self.config.metric_scale.baseline_tolerance_m)]
            if len(self.config.metric_scale.baseline_members) != 2:
                raise ValueError("metric_scale.baseline_members must name the exact sensor pair")
            command += ["--baseline-sensors", *self.config.metric_scale.baseline_members]
        run(command)
        summary = load_json(work_output / "prep_summary.json")
        if not summary.get("alignment_qa", {}).get("passed"):
            raise RuntimeError("RGB-D alignment QA failed")
        replace_directory(work_output, self.run_dir / "rgbd")
        self.result("PASS", frames="frames.json", prep_summary="prep_summary.json")
        self.cleanup_stage_work("rgbd")

    def mapping(self) -> None:
        mapping_dir = self.run_dir / "mapping"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        for stale in mapping_dir.glob(".attempt-*"):
            if stale.is_dir():
                shutil.rmtree(stale)
        work = mapping_dir / f".attempt-{os.getpid()}-{time.time_ns()}"
        (work / "masks").mkdir(parents=True, exist_ok=False)
        os.chmod(work, 0o775)
        os.chmod(work / "masks", 0o775)
        relative = work.relative_to(self.run_dir).as_posix()
        try:
            self.prepare_main_mount()
            command, _ = self.docker_base("mapping", gpu=True, runtime="main")
            command += self.model_mounts()
            command += [
                "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
                "-e", f"ROS_DOMAIN_ID={50 + int(self.scope[1:5], 16) % 150}",
                "-v", f"{ROOT}:/home/scene_graph/scene_graph:ro", "-v", f"{self.run_dir}:/farm-run",
                "--entrypoint", "/bin/bash", self.main_image,
                "/home/scene_graph/scene_graph/docker/entrypoint.sh",
                "python", "-m", "scene_graph.offline.run",
                "--source", "frames-json", "--frames-json-dir", "/farm-run/rgbd",
                "--batch-size", str(self.config.resources.mapping_batch_size),
                "--target-fps", "0", "--warmup-frames", "2",
                "--save-path", f"/farm-run/{relative}/scene_state_raw.pt", "--covisibility", "--offline-debug",
                "--debug-trace-path", f"/farm-run/{relative}/debug_trace.jsonl",
                "--timing-report-path", f"/farm-run/{relative}/mapping_timing.json",
                "--mask-observation-dir", f"/farm-run/{relative}/masks",
                "--mask-observation-max-per-object", "24",
            ]
            parameters = [
                "segmenter_device:=cuda:0", "segmenter_conf:=0.30",
                f"segmenter_model_id:={self.resolved()['model_contract']['segmentation']}",
                "correspondence_feature_sim_thresh:=0.60", "correspondence_hellinger_thresh:=0.65",
                "correspondence_max_merge_distance_m:=0.60", "correspondence_use_class_gate:=false",
                "correspondence_assignment_mode:=best_only", "prune_enabled:=false",
                "caption_enabled:=false", "region_enabled:=false",
                "scene_graph_json_save_enabled:=false", "scene_graph_snapshot_save_enabled:=false",
            ]
            for value in parameters:
                command += ["--extra-param", value]
            run(command)
            if not (work / "scene_state_raw.pt").is_file():
                raise RuntimeError("mapping did not produce its stage-owned scene state")
            atomic_copy(work / "scene_state_raw.pt", mapping_dir / "scene_state_raw.pt")
            if (work / "debug_trace.jsonl").is_file():
                atomic_copy(work / "debug_trace.jsonl", mapping_dir / "debug_trace.jsonl")
            if (work / "mapping_timing.json").is_file():
                atomic_copy(work / "mapping_timing.json", self.run_dir / "timing/mapping_timing.json")
            replace_directory(work / "masks", mapping_dir / "masks")
        finally:
            if work.exists():
                shutil.rmtree(work)
        self.result("PASS", scene_state="scene_state_raw.pt", masks="masks")

    def mapping_qa(self) -> None:
        self.post("analyze_farm_scene_quality.py", [
            "--pt", "/farm-run/mapping/scene_state_raw.pt", "--frames-json", "/farm-run/rgbd/frames.json",
            "--selection-manifest", "/farm-run/selection/selection_manifest.json",
            "--prep-summary", "/farm-run/rgbd/prep_summary.json", "--output-dir", "/farm-run/qa/mapping/analysis",
            "--no-require-semantics",
        ])
        report = load_json(self.run_dir / "qa/mapping/analysis/quality_summary.json")
        failed = [key for key, value in report.get("checks", {}).items() if value is not True]
        if failed:
            raise RuntimeError("mapping QA failed: " + ", ".join(failed))
        self.result("PASS", quality_summary="analysis/quality_summary.json")

    def presentation(self) -> None:
        self.prep_post("prepare_farm_presentation.py", [
            "--scene-state", "/farm-run/mapping/scene_state_raw.pt", "--scene-ply", "/input/scene.ply",
            "--output-dir", "/farm-run/mapping/presentation", "--state-filename", "scene_state_presentation.pt",
            "--metadata-source-scene-state", "../scene_state_raw.pt",
            "--min-observations", "3", "--no-resolved-only", "--max-camera-distance-m", "0",
            "--max-cloud-points", str(self.config.viewer.max_context_points), "--min-opacity", "0.08",
            "--meters-per-scene-unit", str(self.resolved()["meters_per_scene_unit"]),
        ])
        self.result("PASS", bundle="presentation/manifest.json")

    def geometry(self) -> None:
        report = "/farm-run/qa/geometry/audit.json"
        self.post("refine_farm_object_geometry.py", [
            "--scene-state", "/farm-run/mapping/presentation/data/scene_state_presentation.pt",
            "--frames-json", "/farm-run/rgbd/frames.json", "--mask-dir", "/farm-run/mapping/masks",
            "--output-state", "/farm-run/mapping/scene_state_geometry.pt", "--output-report", report,
            "--min-depth-observations", "3", "--min-consistency", "0.65",
            "--min-center-inside-rate", "0.80", "--max-normalized-reprojection-error", "0.30",
            "--min-median-projected-box-iou", "0.35", "--min-refit-box-iou", "0.08",
            "--max-refit-center-error", "0.45", "--final-support-iou", "0.20",
            "--min-box-support-rate", "0.70", "--min-voxel-inside-rate", "0.55",
            "--orientation-mode", "gravity_yaw", "--up-vector", self.up(),
        ])
        self.post("plot_farm_geometry_audit.py", [
            "--report", report, "--cloud", "/farm-run/mapping/presentation/data/cloud.npz",
            "--output", "/farm-run/visuals/03_geometry_audit_4k.jpg",
            "--scene-id", self.config.scene_id,
        ])
        self.post("plot_farm_obb_reprojections.py", [
            "--scene-state", "/farm-run/mapping/scene_state_geometry.pt", "--geometry-report", report,
            "--frames-json", "/farm-run/rgbd/frames.json", "--mapping-dir", "/farm-run/mapping",
            "--output", "/farm-run/visuals/04_obb_reprojections_4k.jpg",
            "--scene-id", self.config.scene_id,
        ])
        self.result("PASS", audit="audit.json")

    def visual_consistency(self) -> None:
        visual_key = str(self.resolved()["model_contract"]["visual_features"])
        model = Path("/farm-models") / visual_key
        report = "/farm-run/qa/visual_consistency/audit.json"
        self.post("refine_farm_visual_consistency.py", [
            "--scene-state", "/farm-run/mapping/scene_state_geometry.pt", "--mask-dir", "/farm-run/mapping/masks",
            "--model", str(model), "--output-state", "/farm-run/mapping/scene_state_visual.pt",
            "--output-report", report, "--max-crops", "6", "--min-crops", "3",
            "--batch-size", "24", "--device", "cuda",
        ], gpu=True)
        self.post("plot_farm_visual_consistency.py", [
            "--report", report, "--mapping-dir", "/farm-run/mapping",
            "--output", "/farm-run/visuals/05_visual_consistency_4k.jpg",
            "--scene-id", self.config.scene_id,
        ])
        self.result("PASS", audit="audit.json")

    def semantics(self) -> None:
        port, model = self.service("caption")
        url = f"http://127.0.0.1:{port}/v1"
        self.post("review_farm_object_crops.py", [
            "--catalog", "/farm-run/qa/mapping/analysis/reliable_objects_min3.json",
            "--selection-report", "/farm-run/qa/visual_consistency/audit.json",
            "--geometry-report", "/farm-run/qa/geometry/audit.json",
            "--scene-state", "/farm-run/mapping/scene_state_visual.pt",
            "--frames-json", "/farm-run/rgbd/frames.json",
            "--mask-dir", "/farm-run/mapping/masks", "--output-dir", "/farm-run/qa/semantics/review",
            "--vllm-url", url, "--model", model, "--crops-per-object", "3", "--workers", "2",
        ], host_network=True)
        self.post("export_reviewed_farm_catalog.py", [
            "--review", "/farm-run/qa/semantics/review/reviewed_robust_objects.json",
            "--output-dir", "/farm-run/qa/semantics/catalog",
        ])
        self.post("adjudicate_farm_semantics.py", [
            "--review", "/farm-run/qa/semantics/review/reviewed_robust_objects.json",
            "--scene-state", "/farm-run/mapping/scene_state_raw.pt",
            "--frames-json", "/farm-run/rgbd/frames.json",
            "--mask-dir", "/farm-run/mapping/masks", "--output-dir", "/farm-run/qa/semantics/ensemble",
            "--vllm-url", url, "--model", model, "--crops-per-object", "6",
            "--min-confidence", "0.80", "--probable-confidence", "0.90", "--workers", "2",
        ], host_network=True)
        self.post("reconcile_farm_semantics.py", [
            "--prior-state", "/farm-run/mapping/scene_state_raw.pt",
            "--candidate-review", "/farm-run/qa/semantics/ensemble/semantic_ensemble_review.json",
            "--detector-state", "/farm-run/mapping/scene_state_raw.pt",
            "--frames-json", "/farm-run/rgbd/frames.json",
            "--mask-dir", "/farm-run/mapping/masks",
            "--output-dir", "/farm-run/qa/semantics/reconciliation",
            "--vllm-url", url, "--model", model, "--crops-per-object", "4",
            "--min-confidence", "0.72", "--workers", "2",
        ], host_network=True)
        self.post("finalize_farm_semantic_consensus.py", [
            "--prior-catalog", "/farm-run/qa/semantics/review/reviewed_robust_objects.json",
            "--ensemble-review", "/farm-run/qa/semantics/ensemble/semantic_ensemble_review.json",
            "--blind-catalog", "/farm-run/qa/semantics/reconciliation/semantic_reconciled_catalog.json",
            "--detector-state", "/farm-run/mapping/scene_state_raw.pt",
            "--output-dir", "/farm-run/qa/semantics/consensus",
        ])
        self.prep_post("prepare_farm_presentation.py", [
            "--scene-state", "/farm-run/mapping/scene_state_visual.pt", "--scene-ply", "/input/scene.ply",
            # Consensus retains every geometry-valid direct object while
            # preventing a weak generic-form fallback from silently replacing
            # a stronger current-run FARM identity.
            "--reviewed-catalog", "/farm-run/qa/semantics/consensus/semantic_consensus_catalog.json",
            "--output-dir", "/farm-run/mapping/presentation", "--state-filename", "scene_state_semantic.pt",
            "--metadata-source-scene-state", "../scene_state_visual.pt",
            "--metadata-reviewed-catalog", "../../qa/semantics/consensus/semantic_consensus_catalog.json",
            "--reuse-cloud", "--min-observations", "3", "--no-resolved-only",
            # Camera-distance filtering is presentation-only and must not prune
            # the canonical processing lineage used by geometry/finalization.
            "--max-camera-distance-m", "0",
            "--max-cloud-points", str(self.config.viewer.max_context_points), "--min-opacity", "0.08",
            "--meters-per-scene-unit", str(self.resolved()["meters_per_scene_unit"]),
        ])
        atomic_copy(self.run_dir / "mapping/presentation/data/scene_state_semantic.pt",
                    self.run_dir / "mapping/scene_state_semantic.pt")
        self.result(
            "PASS",
            review="review/reviewed_robust_objects.json",
            consensus="consensus/semantic_consensus_report.json",
            service_scope=self.scope,
        )

    def part_whole(self) -> None:
        self.service("caption")
        self.post("analyze_farm_part_whole.py", [
            "--scene-state", "/farm-run/mapping/scene_state_raw.pt", "--mask-root", "/farm-run/mapping/masks",
            "--output", "/farm-run/qa/part_whole/audit.json",
        ])
        self.result("PASS", audit="audit.json", service_scope=self.scope)

    def embedding(self, service: str, mode: str, source: str, output: str) -> None:
        port, model = self.service(service)
        arguments = [
            "--scene-state", source, "--output-state", output, "--mode", mode,
            "--vllm-url", f"http://127.0.0.1:{port}/v1", "--model", model,
            "--report", f"/farm-run/qa/assemblies/{mode}_embedding.json",
        ]
        if mode == "vl":
            arguments += ["--mask-root", "/farm-run/mapping/masks", "--mask-root",
                          "/farm-run/qa/assemblies/masks", "--crops-per-object", "3"]
        self.post("enrich_farm_embeddings.py", arguments, host_network=True)

    def assemblies(self) -> None:
        try:
            port, model = self.service("caption")
            url = f"http://127.0.0.1:{port}/v1"
            common = [
                "--scene-state", "/farm-run/mapping/scene_state_semantic.pt",
                "--part-whole-report", "/farm-run/qa/part_whole/audit.json",
                "--mask-root", "/farm-run/mapping/masks",
                "--up-vector", self.up(),
            ]
            self.post("build_farm_object_assemblies.py", common + [
                "--output-state", "/farm-run/qa/assemblies/scene_state_candidates.pt",
                "--output-report", "/farm-run/qa/assemblies/candidates.json",
                "--output-catalog", "/farm-run/qa/assemblies/review_queue.json",
                "--output-mask-root", "/farm-run/qa/assemblies/masks",
                "--output-review-mask-root", "/farm-run/qa/assemblies/review_masks",
            ])
            self.post("review_farm_object_crops.py", [
                "--catalog", "/farm-run/qa/assemblies/review_queue.json",
                "--scene-state", "/farm-run/qa/assemblies/scene_state_candidates.pt",
                "--frames-json", "/farm-run/rgbd/frames.json",
                "--mask-dir", "/farm-run/qa/assemblies/review_masks", "--output-dir", "/farm-run/qa/assemblies/review",
                "--vllm-url", url, "--model", model, "--crops-per-object", "6", "--workers", "2",
                "--expand-image-matches",
            ], host_network=True)
            self.post("build_farm_object_assemblies.py", common + [
                "--output-state", "/farm-run/qa/assemblies/scene_state_reviewed.pt",
                "--output-report", "/farm-run/qa/assemblies/final_audit.json",
                "--output-catalog", "/farm-run/qa/assemblies/final_catalog.json",
                "--output-mask-root", "/farm-run/qa/assemblies/masks",
                "--output-review-mask-root", "/farm-run/qa/assemblies/review_masks",
                "--review-report", "/farm-run/qa/assemblies/review/reviewed_robust_objects.json",
                "--require-reviewed",
            ])
            self.post("refine_farm_object_geometry.py", [
                "--scene-state", "/farm-run/qa/assemblies/scene_state_reviewed.pt",
                "--frames-json", "/farm-run/rgbd/frames.json", "--mask-dir", "/farm-run/qa/assemblies/masks",
                "--output-state", "/farm-run/qa/assemblies/scene_state_geometry.pt",
                "--output-report", "/farm-run/qa/assemblies/geometry_audit.json",
                "--assembly-only", "--orientation-mode", "gravity_yaw", "--up-vector", self.up(),
            ])
            self.stop_services()
            self.embedding("text-embed", "text", "/farm-run/qa/assemblies/scene_state_geometry.pt",
                           "/farm-run/qa/assemblies/scene_state_text.pt")
            self.stop_services()
            self.embedding("vl-embed", "vl", "/farm-run/qa/assemblies/scene_state_text.pt",
                           "/farm-run/mapping/scene_state_assemblies.pt")
            self.result("PASS", audit="final_audit.json", embeddings=["text", "vl"])
        finally:
            self.stop_services()

    def surface_support(self) -> None:
        self.post("audit_farm_gaussian_support.py", [
            "--scene-state", "/farm-run/mapping/scene_state_assemblies.pt",
            "--frames-json", "/farm-run/rgbd/frames.json", "--direct-mask-root", "/farm-run/mapping/masks",
            "--assembly-mask-root", "/farm-run/qa/assemblies/masks",
            "--surface-cloud", "/farm-run/mapping/presentation/data/cloud.npz",
            "--output-state", "/farm-run/mapping/scene_state_surface.pt",
            "--output-report", "/farm-run/qa/surface_support/audit.json", "--no-enforce",
        ])
        self.result("PASS", audit="audit.json")

    def compound_geometry(self) -> None:
        self.post("refine_farm_compound_geometry.py", [
            "--scene-state", "/farm-run/mapping/scene_state_surface.pt",
            "--frames-json", "/farm-run/rgbd/frames.json",
            "--assembly-mask-dir", "/farm-run/qa/assemblies/masks",
            "--cloud-npz", "/farm-run/mapping/presentation/data/cloud.npz",
            "--output-state", "/farm-run/qa/compound_geometry/scene_state_raw.pt",
            "--output-report", "/farm-run/qa/compound_geometry/audit.json", "--up-vector", self.up(),
        ])
        self.post("refine_farm_compound_presentation.py", [
            "--scene-state", "/farm-run/qa/compound_geometry/scene_state_raw.pt",
            "--cloud-npz", "/farm-run/mapping/presentation/data/cloud.npz",
            "--frames-json", "/farm-run/rgbd/frames.json", "--mask-root", "/farm-run/qa/assemblies/masks",
            "--output-scene-state", "/farm-run/mapping/scene_state_compound.pt",
            "--report", "/farm-run/qa/compound_geometry/presentation_audit.json",
        ])
        self.result("PASS", audit="audit.json", presentation_audit="presentation_audit.json")

    def geometry_qa(self) -> None:
        self.post("validate_farm_geometry.py", [
            "--scene-state", "/farm-run/mapping/scene_state_compound.pt",
            "--direct-state", "/farm-run/mapping/scene_state_semantic.pt",
            "--resolved-context", "/farm-run/input/resolved_context.json",
            "--surface-report", "/farm-run/qa/surface_support/audit.json",
            "--output", "/farm-run/qa/geometry_qa/structural.json",
        ])
        self.result("PASS", structural=load_json(self.run_dir / "qa/geometry_qa/structural.json"))

    def dedup(self) -> None:
        self.post("unify_farm_compound_presentation.py", [
            "--scene-state", "/farm-run/mapping/scene_state_compound.pt",
            "--output-state", "/farm-run/qa/dedup/scene_state_unified.pt",
            "--report", "/farm-run/qa/dedup/unified_presentation.json",
            "--all-eligible", "--min-gaussian-support", "0.90",
        ])
        self.post("resolve_farm_track_duplicates.py", [
            "--scene-state", "/farm-run/qa/dedup/scene_state_unified.pt",
            "--output-state", "/farm-run/mapping/scene_state_dedup.pt",
            "--output-report", "/farm-run/qa/dedup/audit.json",
            "--mask-root", "/farm-run/mapping/masks",
        ])
        self.result("PASS", audit="audit.json", unified_presentation="unified_presentation.json")

    def finalize(self) -> None:
        work = self.run_dir / ".stage_work/finalize"
        if work.exists():
            shutil.rmtree(work)
        acceptance_work = work / "acceptance"
        acceptance_work.mkdir(parents=True, exist_ok=True)
        accepted = work / "scene_state_accepted.pt"
        self.post("build_farm_tiered_state.py", [
            "--assembled-state", "/farm-run/mapping/scene_state_dedup.pt",
            # Finalization must consume the last semantic decision artifact.
            # The ensemble catalog is intentionally conservative and is an
            # intermediate input to cross-pass consensus; using it here
            # silently regressed resolved objects back to geometry_only.
            "--direct-catalog", "/farm-run/qa/semantics/consensus/semantic_consensus_catalog.json",
            "--assembly-review", "/farm-run/qa/assemblies/review/reviewed_robust_objects.json",
            "--output", "/farm-run/.stage_work/finalize/scene_state_tiered.pt",
        ])
        self.post("build_farm_final_acceptance.py", [
            "--scene-state", "/farm-run/.stage_work/finalize/scene_state_tiered.pt",
            "--dedup-audit", "/farm-run/qa/dedup/audit.json",
            "--semantic-catalog", "/farm-run/qa/semantics/consensus/semantic_consensus_catalog.json",
            "--assembly-review", "/farm-run/qa/assemblies/review/reviewed_robust_objects.json",
            "--mask-root", "/farm-run/mapping/masks",
            "--mask-root", "/farm-run/qa/assemblies/masks",
            "--output-state", "/farm-run/.stage_work/finalize/scene_state_accepted.pt",
            "--output-report", "/farm-run/.stage_work/finalize/acceptance/result.json",
            "--output-clusters", "/farm-run/.stage_work/finalize/acceptance/duplicate_clusters.json",
            "--output-labels", "/farm-run/.stage_work/finalize/acceptance/label_uncertainty.json",
            "--scene-id", self.config.scene_id,
            "--pre-qa-wall-seconds", f"{self.pre_qa_wall_seconds():.6f}",
            "--mode", "apply",
        ])
        acceptance = load_json(acceptance_work / "result.json")
        if str(acceptance.get("status", "")).upper() == "FAIL":
            raise RuntimeError("final acceptance gate failed before publication")
        # The tiered intermediate is stage-owned and is never published as
        # the canonical result; only the accepted state crosses this boundary.
        atomic_copy(accepted, self.run_dir / "final/scene_state.pt")
        replace_directory(acceptance_work, self.run_dir / "qa/acceptance")
        atomic_copy(self.run_dir / "mapping/presentation/data/cloud.npz", self.run_dir / "final/cloud.npz")
        self.export_catalog()
        self.result(
            "PASS", scene_state="scene_state.pt", catalog="catalog.json",
            presentation_catalog="presentation_catalog.json", cloud="cloud.npz",
            release_status=str(acceptance.get("status", "WARN")).upper(),
            acceptance="../qa/acceptance/result.json",
        )
        self.cleanup_stage_work("finalize")

    def export_catalog(self) -> None:
        import torch
        wrapper = torch.load(self.run_dir / "final/scene_state.pt", map_location="cpu", weights_only=False)
        state = wrapper.get("state", wrapper)
        if not isinstance(state, Mapping):
            raise ValueError("final scene state payload is not a mapping")
        ids = state["object_id"].detach().cpu().tolist()
        active = state["active"].detach().cpu().tolist()
        means = state["means"].detach().cpu()
        refined_centers = state.get("object_box_centers_m")
        centers = refined_centers.detach().cpu() if isinstance(refined_centers, torch.Tensor) else means
        dimensions = state["object_box_dimensions_m"].detach().cpu()
        rotations = state["object_box_wxyz"].detach().cpu()
        count = len(ids)
        if (
            len(active) != count or tuple(means.shape) != (count, 3)
            or tuple(centers.shape) != (count, 3)
            or tuple(dimensions.shape) != (count, 3)
            or tuple(rotations.shape) != (count, 4)
        ):
            raise ValueError("final object arrays are not row-aligned")
        categories = list(state.get("object_category") or [""] * len(ids))
        captions = list(state.get("object_caption") or [""] * len(ids))
        attrs = list(state.get("object_key_attributes") or [[] for _ in ids])
        semantic_tiers = list(state.get("object_semantic_tier") or [""] * len(ids))
        semantic_statuses = list(state.get("object_semantic_status") or [""] * len(ids))
        evidence_tiers = list(state.get("object_evidence_tier") or [""] * len(ids))
        geometry_statuses = list(state.get("object_geometry_status") or [""] * len(ids))
        display_statuses = list(state.get("object_display_status") or [""] * len(ids))
        assembly_members = list(state.get("object_assembly_member_ids") or [[] for _ in ids])
        compound_boxes = list(state.get("object_compound_boxes") or [[] for _ in ids])
        canonical_value = state.get("object_duplicate_canonical_id")
        canonical_ids = (
            canonical_value.detach().cpu().tolist()
            if isinstance(canonical_value, torch.Tensor) else [-1] * count
        )
        if any(
            len(values) != count
            for values in (
                categories, captions, attrs, semantic_tiers, semantic_statuses,
                evidence_tiers, geometry_statuses, display_statuses,
                assembly_members, compound_boxes, canonical_ids,
            )
        ):
            raise ValueError("final semantic arrays are not row-aligned")
        rows = []
        presentation_rows = []
        for index, object_id in enumerate(ids):
            geometry_status = str(geometry_statuses[index]).lower()
            probable_compound = (
                "compound_geometry_probable" in geometry_status
                and "rejected" not in geometry_status
                and len(compound_boxes[index] or []) == 2
            )
            if active[index] or probable_compound:
                if (
                    not torch.isfinite(centers[index]).all()
                    or not torch.isfinite(dimensions[index]).all()
                    or not torch.isfinite(rotations[index]).all()
                ):
                    raise ValueError(f"active object {object_id} has non-finite metric geometry")
                if not torch.all(dimensions[index] > 0):
                    raise ValueError(f"active object {object_id} has non-positive metric dimensions")
                display_status = str(display_statuses[index])
                semantic_tier = str(semantic_tiers[index]).strip().lower()
                semantic_visible = semantic_tier in {"confirmed", "probable"}
                presentation_visible = bool(
                    not display_status.endswith("_suppressed")
                    and ((active[index] and semantic_visible) or probable_compound)
                )
                row = {
                    "id": int(object_id), "category": str(categories[index]),
                    "description": str(captions[index]),
                    "attributes": list(map(str, attrs[index] or [])),
                    "center_m": centers[index].tolist(),
                    "dimensions_m": dimensions[index].tolist(),
                    "wxyz": rotations[index].tolist(),
                    "semantic_tier": str(semantic_tiers[index]),
                    "semantic_status": str(semantic_statuses[index]),
                    "evidence_tier": str(evidence_tiers[index]),
                    "geometry_status": str(geometry_statuses[index]),
                    "metric_active": bool(active[index]),
                    "presentation_visible": presentation_visible,
                    "display_status": display_status,
                    "canonical_object_id": int(canonical_ids[index]),
                    "assembly_member_ids": [int(value) for value in (assembly_members[index] or [])],
                    "compound_boxes": compound_boxes[index],
                }
                rows.append(row)
                if presentation_visible:
                    presentation_rows.append(row)
        if not rows:
            raise ValueError("final catalog contains no active metric objects")
        if not presentation_rows:
            raise ValueError("final presentation catalog contains no visible objects")
        atomic_write_json(self.run_dir / "final/catalog.json", rows)
        atomic_write_json(self.run_dir / "final/presentation_catalog.json", presentation_rows)

    def qa_profile(self) -> None:
        selected_uuid = str(self.resolved()["selected_gpu"]["uuid"])
        measurement_scope = "whole_selected_device_not_pid_attributed"
        stages: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []
        for stage_id in STANDARD_STAGE_IDS:
            if stage_id == "qa_bundle":
                continue
            path = self.run_dir / "stages" / stage_id / "state.json"
            if not path.is_file():
                continue
            state = load_json(path)
            telemetry = state.get("telemetry_summary") or {}
            baseline = (telemetry.get("gpu_device_baseline_used_by_uuid_mb") or {}).get(selected_uuid)
            peak = (telemetry.get("gpu_device_peak_used_by_uuid_mb") or {}).get(selected_uuid)
            delta = (telemetry.get("gpu_device_peak_delta_by_uuid_mb") or {}).get(selected_uuid)
            stages.append({
                "stage": stage_id, "status": state.get("status"),
                "duration_seconds": state.get("duration_seconds"),
                "device_peak_used_gib": (
                    float(peak) / 1024.0 if peak is not None else None
                ),
                "incremental_peak_delta_gib": (
                    float(delta) / 1024.0 if delta is not None else None
                ),
                "measurement_scope": measurement_scope,
            })
            resources.append({
                "stage": stage_id, "gpu_uuid": selected_uuid,
                "measurement_scope": measurement_scope,
                "baseline_used_mb": baseline, "peak_used_mb": peak, "peak_delta_mb": delta,
                "device_peak_used_gib": (
                    float(peak) / 1024.0 if peak is not None else None
                ),
                "incremental_peak_delta_gib": (
                    float(delta) / 1024.0 if delta is not None else None
                ),
            })
        atomic_write_json(self.run_dir / "timing/qa_snapshot.json", {
            "schema": "farm.qa-timing-snapshot.v2", "scene_id": self.config.scene_id,
            "source": "stages/*/state.json", "stages": stages,
            "total_seconds": sum(
                float(row["duration_seconds"]) for row in stages
                if row.get("duration_seconds") is not None
            ),
        })
        atomic_write_json(self.run_dir / "qa/resource_summary.json", {
            "schema": "farm.qa-resource-snapshot.v2", "scene_id": self.config.scene_id,
            "selected_gpu_uuid": selected_uuid,
            "measurement_scope": measurement_scope,
            "stages": resources,
            "device_peak_used_gib": max(
                (row["peak_used_mb"] / 1024.0 for row in resources if row["peak_used_mb"] is not None),
                default=None,
            ),
            "incremental_peak_delta_gib": max(
                (
                    row["peak_delta_mb"] / 1024.0
                    for row in resources
                    if row["peak_delta_mb"] is not None
                ),
                default=None,
            ),
        })

    def qa_bundle(self) -> None:
        self.qa_profile()
        verify_work = self.run_dir / ".stage_work/qa_bundle/acceptance_verify"
        if verify_work.exists():
            shutil.rmtree(verify_work)
        verify_work.mkdir(parents=True, exist_ok=True)
        self.post("build_farm_final_acceptance.py", [
            "--scene-state", "/farm-run/final/scene_state.pt",
            "--dedup-audit", "/farm-run/qa/dedup/audit.json",
            "--semantic-catalog", "/farm-run/qa/semantics/consensus/semantic_consensus_catalog.json",
            "--assembly-review", "/farm-run/qa/assemblies/review/reviewed_robust_objects.json",
            "--mask-root", "/farm-run/mapping/masks",
            "--mask-root", "/farm-run/qa/assemblies/masks",
            "--output-state", "/farm-run/.stage_work/qa_bundle/acceptance_verify/scene_state.pt",
            "--output-report", "/farm-run/.stage_work/qa_bundle/acceptance_verify/result.json",
            "--output-clusters", "/farm-run/.stage_work/qa_bundle/acceptance_verify/duplicate_clusters.json",
            "--output-labels", "/farm-run/.stage_work/qa_bundle/acceptance_verify/label_uncertainty.json",
            "--scene-id", self.config.scene_id,
            "--pre-qa-wall-seconds", f"{self.pre_qa_wall_seconds():.6f}",
            "--mode", "verify",
        ])
        acceptance = load_json(self.run_dir / "qa/acceptance/result.json")
        acceptance_verify = load_json(verify_work / "result.json")
        presentation_payload = json.loads(
            (self.run_dir / "final/presentation_catalog.json").read_text(encoding="utf-8")
        )
        if not isinstance(presentation_payload, list):
            raise TypeError("final presentation catalog must be a JSON array")
        final_presentation_count = len(presentation_payload)
        apply_counts = acceptance.get("counts") or {}
        verify_counts = acceptance_verify.get("counts") or {}
        release_statuses = {
            str(acceptance.get("status", "")).upper(),
            str(acceptance_verify.get("status", "")).upper(),
        }
        acceptance_failures = [
            name
            for name, failed in {
                "invalid_release_status": not release_statuses <= {"PASS", "WARN"},
                "apply_remaining_auto": int(apply_counts.get("remaining_auto_clusters") or 0) != 0,
                "apply_remaining_blocking": int(apply_counts.get("remaining_blocking_clusters") or 0) != 0,
                "apply_label_errors": int(apply_counts.get("label_hard_errors") or 0) != 0,
                "verify_would_hold_objects": int(verify_counts.get("held_objects") or 0) != 0,
                "verify_remaining_auto": int(verify_counts.get("remaining_auto_clusters") or 0) != 0,
                "verify_remaining_blocking": int(verify_counts.get("remaining_blocking_clusters") or 0) != 0,
                "verify_label_errors": int(verify_counts.get("label_hard_errors") or 0) != 0,
                "apply_catalog_count_mismatch": int(apply_counts.get("presentation_after") or -1) != final_presentation_count,
                "verify_input_count_mismatch": int(verify_counts.get("presentation_before") or -1) != final_presentation_count,
                "verify_output_count_mismatch": int(verify_counts.get("presentation_after") or -1) != final_presentation_count,
            }.items()
            if failed
        ]
        if acceptance_failures:
            raise RuntimeError(
                "final acceptance verification failed: " + ", ".join(acceptance_failures)
            )
        self.post("analyze_farm_scene_quality.py", [
            "--pt", "/farm-run/final/scene_state.pt", "--frames-json", "/farm-run/rgbd/frames.json",
            "--selection-manifest", "/farm-run/selection/selection_manifest.json",
            "--prep-summary", "/farm-run/rgbd/prep_summary.json",
            "--timing-summary", "/farm-run/timing/qa_snapshot.json",
            "--resource-summary", "/farm-run/qa/resource_summary.json",
            "--presentation-catalog", "/farm-run/final/presentation_catalog.json",
            "--acceptance-report", "/farm-run/qa/acceptance/result.json",
            "--scene-id", self.config.scene_id, "--output-dir", "/farm-run/qa/final_analysis",
        ])
        quality = load_json(self.run_dir / "qa/final_analysis/quality_summary.json")
        failed = [key for key, value in quality.get("checks", {}).items() if value is not True]
        if str(quality.get("qa_status", "")).lower() != "pass" or failed:
            raise RuntimeError("final scene QA failed: " + ", ".join(failed or ["qa_status"]))
        self.post("build_farm_retention_funnel.py", [
            "--run-dir", "/farm-run",
            "--output", "/farm-run/qa/retention_funnel.json",
            "--visual", "/farm-run/visuals/07_retention_quality_dashboard_4k.jpg",
        ])
        retention = load_json(self.run_dir / "qa/retention_funnel.json")
        if str(retention.get("structural_status", "")).upper() != "PASS":
            raise RuntimeError("retention-funnel structural invariants failed")
        visual_work = self.run_dir / ".stage_work/qa_bundle/final_visuals"
        if visual_work.exists():
            shutil.rmtree(visual_work)
        visual_work.parent.mkdir(parents=True, exist_ok=True)
        self.post("visualize_farm_scene_state.py", [
            "--pt", "/farm-run/final/scene_state.pt", "--frames-json", "/farm-run/rgbd/frames.json",
            "--segmentation-dir", "/farm-run/mapping/masks",
            "--presentation-catalog", "/farm-run/final/presentation_catalog.json",
            "--dedup-audit", "/farm-run/qa/dedup/audit.json",
            "--acceptance-report", "/farm-run/qa/acceptance/result.json",
            "--acceptance-clusters", "/farm-run/qa/acceptance/duplicate_clusters.json",
            "--acceptance-labels", "/farm-run/qa/acceptance/label_uncertainty.json",
            "--scene-id", self.config.scene_id,
            "--output-dir", "/farm-run/.stage_work/qa_bundle/final_visuals",
        ])
        replace_directory(visual_work, self.run_dir / "visuals/final")
        sources = (
            (self.run_dir / "selection/visuals/01_selection_dashboard_4k.jpg", self.run_dir / "visuals/01_selection_dashboard_4k.jpg"),
            (self.run_dir / "rgbd/qa/01_alignment_dashboard.jpg", self.run_dir / "visuals/02_rgbd_alignment_dashboard_4k.jpg"),
            (self.run_dir / "qa/final_analysis/01_evidence_tier_dashboard_4k.jpg", self.run_dir / "visuals/06_final_quality_dashboard_4k.jpg"),
            (self.run_dir / "visuals/final/05_final_acceptance_dashboard_4k.jpg", self.run_dir / "visuals/08_final_acceptance_dashboard_4k.jpg"),
        )
        for source, destination in sources:
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"required QA visualization is missing: {source}")
            atomic_copy(source, destination)
        root_images = [
            self.run_dir / "visuals/01_selection_dashboard_4k.jpg",
            self.run_dir / "visuals/02_rgbd_alignment_dashboard_4k.jpg",
            self.run_dir / "visuals/03_geometry_audit_4k.jpg",
            self.run_dir / "visuals/04_obb_reprojections_4k.jpg",
            self.run_dir / "visuals/05_visual_consistency_4k.jpg",
            self.run_dir / "visuals/06_final_quality_dashboard_4k.jpg",
            self.run_dir / "visuals/07_retention_quality_dashboard_4k.jpg",
            self.run_dir / "visuals/08_final_acceptance_dashboard_4k.jpg",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in root_images):
            raise RuntimeError("all eight canonical QA dashboards must be present and non-empty")
        final_names = (
            "01_object_map.png",
            "02_segmentation_contact_sheet.jpg",
            "03_scene_state_dashboard_4k.jpg",
            "04_duplicate_overlap_review_4k.jpg",
            "05_final_acceptance_dashboard_4k.jpg",
            "06_label_uncertainty_sample_4k.jpg",
        )
        final_images = [self.run_dir / "visuals/final" / name for name in final_names]
        if any(not path.is_file() or path.stat().st_size == 0 for path in final_images):
            raise RuntimeError("all six named final-scene visualizations are required")
        images = [*root_images, *final_images]
        validate_canonical_visuals(images)
        severity = {"PASS": 0, "WARN": 1, "FAIL": 2}
        release_status = max(
            (
                str(acceptance.get("status", "WARN")).upper(),
                str(retention.get("quality_status", "WARN")).upper(),
            ),
            key=lambda value: severity.get(value, 2),
        )
        atomic_write_json(self.run_dir / "visuals/index.json", {
            "schema": "farm.visual-index.v2", "status": "PASS",
            "release_status": release_status,
            "images": [{
                "path": path.relative_to(self.run_dir / "visuals").as_posix(),
                "bytes": path.stat().st_size,
            } for path in images],
        })
        caveat = (
            "The detector uses a static open-vocabulary YOLOE profile. Small, occluded, "
            "or out-of-vocabulary instances can be missed; the catalog is evidence-backed, "
            "not an exhaustive inventory."
        )
        atomic_write_json(self.run_dir / "qa/summary.json", {
            "schema": "farm.standard-qa-summary.v1", "status": "PASS",
            "release_status": release_status,
            "scene_id": self.config.scene_id,
            "sources": {
                "quality": "final_analysis/quality_summary.json",
                "retention_quality": "retention_funnel.json",
                "final_acceptance": "acceptance/result.json",
                "structural_geometry": "geometry_qa/structural.json",
                "timing": "../timing/qa_snapshot.json", "resources": "resource_summary.json",
            },
            "quality_status": retention.get("quality_status"),
            "quality_warnings": retention.get("warnings", []),
            "detector_profile": "static open-vocabulary YOLOE", "recall_caveat": caveat,
        })
        atomic_write_json(self.run_dir / "viewer/result.json", {
            "schema": "farm.viewer-result.v1", "status": "PASS",
            "release_status": release_status,
            "scene_id": self.config.scene_id, "world_up": self.resolved()["resolved_up"],
            "meters_per_scene_unit": self.resolved()["meters_per_scene_unit"],
            "artifacts": {
                "scene_state": "../final/scene_state.pt", "catalog": "../final/catalog.json",
                "cloud": "../final/cloud.npz", "frames": "../rgbd", "mapping": "../mapping",
                "visual_index": "../visuals/index.json",
                "final_acceptance": "../qa/acceptance/result.json",
            },
            "viewer": {
                "point_size_m": self.config.viewer.scene_point_size_m,
                "max_context_points": self.config.viewer.max_context_points,
                "offline_click_inspection": True, "query_backend_required": False,
            },
        })
        pipeline_python = shlex.quote(sys.executable)
        atomic_write_text(self.run_dir / "viewer/launch.sh", f"""#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_DIR="$(cd "${{BUNDLE_DIR}}/.." && pwd)"
PYTHON_EXECUTABLE="${{FARM_PIPELINE_PYTHON:-{pipeline_python}}}"
exec "${{PYTHON_EXECUTABLE}}" -m farm_runtime.cli serve --run "${{RUN_DIR}}" "$@"
""")
        os.chmod(self.run_dir / "viewer/launch.sh", 0o755)
        self.result(
            "PASS",
            release_status=release_status,
            visual_count=len(images),
            viewer="../viewer/result.json",
            quality_summary="final_analysis/quality_summary.json",
            retention_quality="retention_funnel.json",
            acceptance="acceptance/result.json",
            qa_summary="summary.json",
        )
        self.cleanup_stage_work("qa_bundle")

    def dispatch(self) -> None:
        handler = getattr(self, str(self.stage), None)
        if self.stage not in STANDARD_STAGE_IDS or not callable(handler):
            raise ValueError(f"unsupported standard stage: {self.stage}")
        handler()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--stage", choices=STANDARD_STAGE_IDS)
    choice.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup_scope(args.config, args.run_dir)
        return 0
    started = time.monotonic()
    context: Context | None = None
    try:
        context = Context(args.config, args.run_dir, args.stage)
        context.dispatch()
        return 0
    except Exception as error:
        if context is not None and args.stage in {"semantics", "part_whole", "assemblies"}:
            context.stop_services()
        if context is not None:
            context.cleanup_containers()
            context.result("FAIL", error={"type": type(error).__name__, "message": str(error)},
                           duration_seconds=round(time.monotonic() - started, 6))
        elif args.stage:
            try:
                config = load_scene_config(args.config.resolve(strict=True))
                trusted_run, _ = _scope_for_run(config, args.run_dir)
                atomic_write_json(trusted_run / RESULT_PATHS[args.stage], {
                    "schema": "farm.standard-stage.v1", "status": "FAIL", "stage": args.stage,
                    "updated_at": utc_now(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "duration_seconds": round(time.monotonic() - started, 6),
                })
            except Exception:
                pass
        print(f"farm-standard-stage: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
