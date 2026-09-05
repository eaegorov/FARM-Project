"""Versioned scene contract for COLMAP + 3D Gaussian FARM ingestion.

Paths in a scene document are resolved relative to the document, never the
caller's current working directory.  The loader accepts the canonical nested
format and a deliberately small set of flat aliases used by early factory
experiments.  The normalized representation is always canonical.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import yaml


SCHEMA_VERSION = "farm.scene.v1"


class SceneConfigError(ValueError):
    """Raised when a scene document violates the ingestion contract."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SceneConfigError(f"{name} must be an object/mapping")
    return value


def _tuple_str(value: Any, name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SceneConfigError(f"{name} must be a list of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise SceneConfigError(f"{name} cannot contain empty strings")
    return result


def _optional_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SceneConfigError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise SceneConfigError(f"{name} must be finite")
    return result


def _resolve_path(value: Any, base_dir: Path, name: str) -> Path:
    if value is None or not str(value).strip():
        raise SceneConfigError(f"{name} is required")
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve(strict=False)


@dataclass(frozen=True)
class InputPaths:
    colmap_model: Path
    image_root: Path
    gaussian_ply: Path


@dataclass(frozen=True)
class CameraGroupingPolicy:
    mode: str = "auto"
    pattern: Optional[str] = None
    timestamp_group: str = "timestamp"
    member_group: str = "camera"
    view_group: str = "view"
    required_members: Tuple[str, ...] = ()
    required_views: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CameraPolicy:
    accepted_models: Tuple[str, ...] = ("PINHOLE", "SIMPLE_PINHOLE")
    require_virtual_pinhole: bool = True
    rgb_alignment: str = "off"


@dataclass(frozen=True)
class SelectionPolicy:
    target_timestamps: int = 160
    max_timestamps: int = 180
    rotation_weight_m_per_rad: float = 0.5
    max_motion_gap: float = 1.35
    max_translation_gap_m: float = 1.30
    max_rotation_gap_deg: float = 50.0
    min_shared_tracks: int = 20
    min_overlap_ratio: float = 0.015
    output_views: Tuple[str, ...] = ()
    anchor_views: Tuple[str, ...] = ()
    view_policy: str = "fixed"
    views_per_sensor: int = 1


@dataclass(frozen=True)
class ResourcePolicy:
    profile: str = "auto"
    gpu: str = "0"
    model_manifest: Optional[Path] = None
    secrets_file: Optional[Path] = None
    render_resolution: int = 896
    max_vram_gb: Optional[float] = None
    max_cpu_workers: Optional[int] = None
    depth_batch_size: int = 1
    mapping_batch_size: int = 5


@dataclass(frozen=True)
class ModelPolicy:
    segmentation: str = "yoloe-v8l"
    visual_features: str = "dinov3-vits16plus"
    caption: str = "Qwen/Qwen3-VL-8B-Instruct"
    text_embeddings: Optional[str] = None
    offline: bool = True


@dataclass(frozen=True)
class ViewerPolicy:
    host: str = "127.0.0.1"
    port: int = 8080
    scene_point_size_m: float = 0.004
    max_context_points: int = 1_500_000
    open_browser: bool = False


@dataclass(frozen=True)
class MetricScalePolicy:
    required: bool = False
    source: str = "auto"
    meters_per_colmap_unit: Optional[float] = None
    evidence: Optional[str] = None
    expected_baseline_m: Optional[float] = None
    baseline_tolerance_m: float = 0.005
    baseline_members: Tuple[str, ...] = ()
    baseline_view: Optional[str] = None


@dataclass(frozen=True)
class GravityPolicy:
    required: bool = False
    mode: str = "auto"
    axis: Optional[str] = None
    vector: Optional[Tuple[float, float, float]] = None
    min_camera_consensus: float = 0.75


@dataclass(frozen=True)
class AlignmentPolicy:
    sample_points: int = 65_536
    nearest_neighbor_points: int = 4_096
    pass_centroid_offset_ratio: float = 0.15
    fail_centroid_offset_ratio: float = 0.75
    pass_extent_ratio: float = 2.5
    fail_extent_ratio: float = 20.0
    pass_bbox_iou: float = 0.10
    fail_bbox_iou: float = 0.001


@dataclass(frozen=True)
class PreflightPolicy:
    image_check: str = "all"
    image_sample_count: int = 512
    require_3dgs_schema: bool = True
    min_registered_images: int = 2
    min_sparse_points: int = 10
    min_gaussians: int = 10
    full_hash_max_bytes: int = 64 * 1024 * 1024
    hash_chunk_bytes: int = 1024 * 1024
    alignment: AlignmentPolicy = field(default_factory=AlignmentPolicy)


@dataclass(frozen=True)
class SceneConfig:
    schema_version: str
    scene_id: str
    config_path: Path
    inputs: InputPaths
    output_root: Path
    camera_grouping: CameraGroupingPolicy = field(default_factory=CameraGroupingPolicy)
    camera: CameraPolicy = field(default_factory=CameraPolicy)
    selection: SelectionPolicy = field(default_factory=SelectionPolicy)
    resources: ResourcePolicy = field(default_factory=ResourcePolicy)
    models: ModelPolicy = field(default_factory=ModelPolicy)
    viewer: ViewerPolicy = field(default_factory=ViewerPolicy)
    metric_scale: MetricScalePolicy = field(default_factory=MetricScalePolicy)
    gravity: GravityPolicy = field(default_factory=GravityPolicy)
    preflight: PreflightPolicy = field(default_factory=PreflightPolicy)

    def normalized_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable canonical representation."""

        value = asdict(self)
        value.pop("config_path", None)

        def convert(item: Any) -> Any:
            if isinstance(item, Path):
                return str(item)
            if isinstance(item, tuple):
                return [convert(child) for child in item]
            if isinstance(item, dict):
                return {key: convert(child) for key, child in item.items()}
            return item

        return convert(value)


def _known_keys(raw: Mapping[str, Any]) -> set[str]:
    return {
        "schema_version", "scene_id", "inputs", "output_root", "output",
        "colmap", "colmap_model", "images", "image_root", "ply",
        "gaussian_ply", "3dgs_ply", "camera_grouping", "camera",
        "selection", "resources", "models", "viewer", "metric_scale",
        "gravity", "preflight",
    }


def _build_config(raw: Mapping[str, Any], config_path: Path) -> SceneConfig:
    unknown = sorted(set(raw) - _known_keys(raw))
    if unknown:
        raise SceneConfigError(f"unknown top-level fields: {', '.join(unknown)}")

    schema_version = str(raw.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise SceneConfigError(
            f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    scene_id = str(raw.get("scene_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", scene_id):
        raise SceneConfigError(
            "scene_id must be 1-128 safe characters: letters, digits, '.', '_' or '-'"
        )

    base_dir = config_path.parent
    inputs = _mapping(raw.get("inputs"), "inputs")
    colmap = inputs.get("colmap_model", inputs.get("colmap", raw.get("colmap_model", raw.get("colmap"))))
    images = inputs.get("image_root", inputs.get("images", raw.get("image_root", raw.get("images"))))
    ply = inputs.get(
        "gaussian_ply",
        inputs.get("3dgs_ply", inputs.get("ply", raw.get("gaussian_ply", raw.get("3dgs_ply", raw.get("ply"))))),
    )
    output = raw.get("output_root", raw.get("output", "output"))

    grouping_raw = _mapping(raw.get("camera_grouping"), "camera_grouping")
    grouping = CameraGroupingPolicy(
        mode=str(grouping_raw.get("mode", "auto")).lower(),
        pattern=(str(grouping_raw["pattern"]) if grouping_raw.get("pattern") else None),
        timestamp_group=str(grouping_raw.get("timestamp_group", "timestamp")),
        member_group=str(grouping_raw.get("member_group", "camera")),
        view_group=str(grouping_raw.get("view_group", "view")),
        required_members=_tuple_str(grouping_raw.get("required_members"), "camera_grouping.required_members"),
        required_views=_tuple_str(grouping_raw.get("required_views"), "camera_grouping.required_views"),
    )
    if grouping.mode not in {"auto", "single", "regex", "manifest"}:
        raise SceneConfigError("camera_grouping.mode must be auto, single, regex or manifest")
    if grouping.mode == "regex" and not grouping.pattern:
        raise SceneConfigError("camera_grouping.pattern is required for regex mode")
    if grouping.pattern:
        try:
            compiled = re.compile(grouping.pattern)
        except re.error as exc:
            raise SceneConfigError(f"invalid camera_grouping.pattern: {exc}") from exc
        required_groups = {grouping.timestamp_group}
        if grouping.required_members:
            required_groups.add(grouping.member_group)
        if grouping.required_views:
            required_groups.add(grouping.view_group)
        missing_groups = sorted(required_groups - set(compiled.groupindex))
        if missing_groups:
            raise SceneConfigError(
                "camera_grouping.pattern lacks named group(s): " + ", ".join(missing_groups)
            )

    camera_raw = _mapping(raw.get("camera"), "camera")
    accepted = camera_raw.get("accepted_models", ("PINHOLE", "SIMPLE_PINHOLE"))
    camera = CameraPolicy(
        rgb_alignment=str(camera_raw.get("rgb_alignment", "off")),
        accepted_models=tuple(item.upper() for item in _tuple_str(accepted, "camera.accepted_models")),
        require_virtual_pinhole=bool(camera_raw.get("require_virtual_pinhole", True)),
    )
    if camera.rgb_alignment not in ("off", "rig"):
        raise SceneConfigError("camera.rgb_alignment must be off or rig")
    if not camera.accepted_models:
        raise SceneConfigError("camera.accepted_models cannot be empty")

    selection_raw = _mapping(raw.get("selection"), "selection")
    selection = SelectionPolicy(
        target_timestamps=int(selection_raw.get("target_timestamps", 160)),
        max_timestamps=int(selection_raw.get("max_timestamps", 180)),
        rotation_weight_m_per_rad=float(selection_raw.get("rotation_weight_m_per_rad", 0.5)),
        max_motion_gap=float(selection_raw.get("max_motion_gap", 1.35)),
        max_translation_gap_m=float(selection_raw.get("max_translation_gap_m", 1.30)),
        max_rotation_gap_deg=float(selection_raw.get("max_rotation_gap_deg", 50.0)),
        min_shared_tracks=int(selection_raw.get("min_shared_tracks", 20)),
        min_overlap_ratio=float(selection_raw.get("min_overlap_ratio", 0.015)),
        output_views=_tuple_str(selection_raw.get("output_views"), "selection.output_views"),
        anchor_views=_tuple_str(selection_raw.get("anchor_views"), "selection.anchor_views"),
        view_policy=str(selection_raw.get("view_policy", "fixed")),
        views_per_sensor=int(selection_raw.get("views_per_sensor", 1)),
    )
    if selection.view_policy not in ("fixed", "balanced") or selection.views_per_sensor < 1:
        raise SceneConfigError("selection.view_policy must be fixed/balanced and views_per_sensor positive")
    if selection.view_policy == "balanced" and not selection.output_views:
        raise SceneConfigError("balanced selection requires explicit output_views")
    if selection.target_timestamps < 2 or selection.max_timestamps < selection.target_timestamps:
        raise SceneConfigError("selection requires 2 <= target_timestamps <= max_timestamps")
    selection_thresholds = {
        "rotation_weight_m_per_rad": selection.rotation_weight_m_per_rad,
        "max_motion_gap": selection.max_motion_gap,
        "max_translation_gap_m": selection.max_translation_gap_m,
        "max_rotation_gap_deg": selection.max_rotation_gap_deg,
    }
    if not all(math.isfinite(value) for value in selection_thresholds.values()):
        raise SceneConfigError("selection metric thresholds must be finite")
    if selection.rotation_weight_m_per_rad < 0 or any(
        value <= 0
        for name, value in selection_thresholds.items()
        if name != "rotation_weight_m_per_rad"
    ):
        raise SceneConfigError("selection gap thresholds must be positive")
    if selection.min_shared_tracks < 0 or not 0 <= selection.min_overlap_ratio <= 1:
        raise SceneConfigError("selection support thresholds are out of range")

    resource_raw = _mapping(raw.get("resources"), "resources")
    gpu = str(resource_raw.get("gpu", "0")).strip()
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", gpu):
        raise SceneConfigError(
            "resources.gpu must be an NVIDIA index or full GPU- UUID"
        )
    resources = ResourcePolicy(
        profile=str(resource_raw.get("profile", "auto")),
        gpu=gpu,
        model_manifest=(
            _resolve_path(resource_raw["model_manifest"], base_dir, "resources.model_manifest")
            if resource_raw.get("model_manifest") else None
        ),
        secrets_file=(
            _resolve_path(resource_raw["secrets_file"], base_dir, "resources.secrets_file")
            if resource_raw.get("secrets_file") else None
        ),
        render_resolution=int(resource_raw.get("render_resolution", 896)),
        max_vram_gb=_optional_float(resource_raw.get("max_vram_gb"), "resources.max_vram_gb"),
        max_cpu_workers=(int(resource_raw["max_cpu_workers"]) if resource_raw.get("max_cpu_workers") is not None else None),
        depth_batch_size=int(resource_raw.get("depth_batch_size", 1)),
        mapping_batch_size=int(resource_raw.get("mapping_batch_size", 5)),
    )
    if resources.render_resolution < 64 or resources.depth_batch_size < 1 or resources.mapping_batch_size < 1:
        raise SceneConfigError("invalid resources: resolution >=64 and batch sizes >=1 are required")
    if resources.max_vram_gb is not None and resources.max_vram_gb <= 0:
        raise SceneConfigError("resources.max_vram_gb must be positive")

    models_raw = _mapping(raw.get("models"), "models")
    models = ModelPolicy(
        segmentation=str(models_raw.get("segmentation", "yoloe-v8l")),
        visual_features=str(models_raw.get("visual_features", "dinov3-vits16plus")),
        caption=str(models_raw.get("caption", "Qwen/Qwen3-VL-8B-Instruct")),
        text_embeddings=(str(models_raw["text_embeddings"]) if models_raw.get("text_embeddings") else None),
        offline=bool(models_raw.get("offline", True)),
    )

    viewer_raw = _mapping(raw.get("viewer"), "viewer")
    viewer = ViewerPolicy(
        host=str(viewer_raw.get("host", "127.0.0.1")),
        port=int(viewer_raw.get("port", 8080)),
        scene_point_size_m=float(viewer_raw.get("scene_point_size_m", 0.004)),
        max_context_points=int(viewer_raw.get("max_context_points", 1_500_000)),
        open_browser=bool(viewer_raw.get("open_browser", False)),
    )
    if not 1 <= viewer.port <= 65535 or viewer.scene_point_size_m <= 0 or viewer.max_context_points < 1:
        raise SceneConfigError("invalid viewer port, scene_point_size_m or max_context_points")

    metric_raw = _mapping(raw.get("metric_scale"), "metric_scale")
    metric = MetricScalePolicy(
        required=bool(metric_raw.get("required", False)),
        source=str(metric_raw.get("source", "auto")).lower(),
        meters_per_colmap_unit=_optional_float(metric_raw.get("meters_per_colmap_unit"), "metric_scale.meters_per_colmap_unit"),
        evidence=(str(metric_raw["evidence"]).strip() if metric_raw.get("evidence") else None),
        expected_baseline_m=_optional_float(metric_raw.get("expected_baseline_m"), "metric_scale.expected_baseline_m"),
        baseline_tolerance_m=float(metric_raw.get("baseline_tolerance_m", 0.005)),
        baseline_members=_tuple_str(metric_raw.get("baseline_members"), "metric_scale.baseline_members"),
        baseline_view=(str(metric_raw["baseline_view"]) if metric_raw.get("baseline_view") is not None else None),
    )
    if metric.source not in {"auto", "unknown", "explicit", "declared_metric", "rig_baseline"}:
        raise SceneConfigError("metric_scale.source has an unsupported value")
    for name, value in (("meters_per_colmap_unit", metric.meters_per_colmap_unit), ("expected_baseline_m", metric.expected_baseline_m)):
        if value is not None and value <= 0:
            raise SceneConfigError(f"metric_scale.{name} must be positive")
    if metric.baseline_tolerance_m <= 0:
        raise SceneConfigError("metric_scale.baseline_tolerance_m must be positive")
    if metric.source == "rig_baseline":
        if metric.expected_baseline_m is None or len(metric.baseline_members) != 2:
            raise SceneConfigError("rig_baseline needs expected_baseline_m and exactly two baseline_members")
        if grouping.mode != "regex":
            raise SceneConfigError("rig_baseline requires camera_grouping.mode=regex")
    if metric.source in {"explicit", "declared_metric"} and metric.meters_per_colmap_unit is None:
        raise SceneConfigError(f"metric_scale.source={metric.source} needs meters_per_colmap_unit")

    gravity_raw = _mapping(raw.get("gravity"), "gravity")
    vector_raw = gravity_raw.get("vector")
    vector: Optional[Tuple[float, float, float]] = None
    if vector_raw is not None:
        if isinstance(vector_raw, str) or not isinstance(vector_raw, Sequence) or len(vector_raw) != 3:
            raise SceneConfigError("gravity.vector must contain exactly three numbers")
        vector = tuple(float(item) for item in vector_raw)  # type: ignore[assignment]
        if not all(math.isfinite(item) for item in vector) or sum(item * item for item in vector) <= 1e-12:
            raise SceneConfigError("gravity.vector must be finite and non-zero")
    gravity = GravityPolicy(
        required=bool(gravity_raw.get("required", False)),
        mode=str(gravity_raw.get("mode", "auto")).lower(),
        axis=(str(gravity_raw["axis"]).lower() if gravity_raw.get("axis") else None),
        vector=vector,
        min_camera_consensus=float(gravity_raw.get("min_camera_consensus", 0.75)),
    )
    if gravity.mode not in {"auto", "axis", "vector", "none"}:
        raise SceneConfigError("gravity.mode must be auto, axis, vector or none")
    if gravity.mode == "axis" and gravity.axis not in {"x", "-x", "y", "-y", "z", "-z"}:
        raise SceneConfigError("gravity.axis must be one of x, -x, y, -y, z, -z")
    if gravity.mode == "vector" and gravity.vector is None:
        raise SceneConfigError("gravity.mode=vector requires gravity.vector")
    if not 0 <= gravity.min_camera_consensus <= 1:
        raise SceneConfigError("gravity.min_camera_consensus must be in [0, 1]")

    preflight_raw = _mapping(raw.get("preflight"), "preflight")
    alignment_raw = _mapping(preflight_raw.get("alignment"), "preflight.alignment")
    alignment = AlignmentPolicy(
        sample_points=int(alignment_raw.get("sample_points", 65_536)),
        nearest_neighbor_points=int(alignment_raw.get("nearest_neighbor_points", 4_096)),
        pass_centroid_offset_ratio=float(alignment_raw.get("pass_centroid_offset_ratio", 0.15)),
        fail_centroid_offset_ratio=float(alignment_raw.get("fail_centroid_offset_ratio", 0.75)),
        pass_extent_ratio=float(alignment_raw.get("pass_extent_ratio", 2.5)),
        fail_extent_ratio=float(alignment_raw.get("fail_extent_ratio", 20.0)),
        pass_bbox_iou=float(alignment_raw.get("pass_bbox_iou", 0.10)),
        fail_bbox_iou=float(alignment_raw.get("fail_bbox_iou", 0.001)),
    )
    if alignment.sample_points < 10 or alignment.nearest_neighbor_points < 0:
        raise SceneConfigError("preflight alignment sample_points must be >=10 and nearest_neighbor_points >=0")
    if not 0 <= alignment.fail_bbox_iou <= alignment.pass_bbox_iou <= 1:
        raise SceneConfigError("alignment bbox IoU thresholds must satisfy 0 <= fail <= pass <= 1")
    if not 0 < alignment.pass_centroid_offset_ratio < alignment.fail_centroid_offset_ratio:
        raise SceneConfigError("alignment centroid thresholds must satisfy 0 < pass < fail")
    if not 1 <= alignment.pass_extent_ratio < alignment.fail_extent_ratio:
        raise SceneConfigError("alignment extent thresholds must satisfy 1 <= pass < fail")
    preflight = PreflightPolicy(
        image_check=str(preflight_raw.get("image_check", "all")).lower(),
        image_sample_count=int(preflight_raw.get("image_sample_count", 512)),
        require_3dgs_schema=bool(preflight_raw.get("require_3dgs_schema", True)),
        min_registered_images=int(preflight_raw.get("min_registered_images", 2)),
        min_sparse_points=int(preflight_raw.get("min_sparse_points", 10)),
        min_gaussians=int(preflight_raw.get("min_gaussians", 10)),
        full_hash_max_bytes=int(preflight_raw.get("full_hash_max_bytes", 64 * 1024 * 1024)),
        hash_chunk_bytes=int(preflight_raw.get("hash_chunk_bytes", 1024 * 1024)),
        alignment=alignment,
    )
    if preflight.image_check not in {"all", "sample", "none"}:
        raise SceneConfigError("preflight.image_check must be all, sample or none")
    if min(preflight.image_sample_count, preflight.min_registered_images, preflight.min_sparse_points, preflight.min_gaussians, preflight.hash_chunk_bytes) < 1:
        raise SceneConfigError("preflight counts and hash_chunk_bytes must be positive")

    return SceneConfig(
        schema_version=schema_version,
        scene_id=scene_id,
        config_path=config_path,
        inputs=InputPaths(
            colmap_model=_resolve_path(colmap, base_dir, "inputs.colmap_model"),
            image_root=_resolve_path(images, base_dir, "inputs.image_root"),
            gaussian_ply=_resolve_path(ply, base_dir, "inputs.gaussian_ply"),
        ),
        output_root=_resolve_path(output, base_dir, "output_root"),
        camera_grouping=grouping,
        camera=camera,
        selection=selection,
        resources=resources,
        models=models,
        viewer=viewer,
        metric_scale=metric,
        gravity=gravity,
        preflight=preflight,
    )


def load_scene_config(path: Path | str) -> SceneConfig:
    """Load YAML/JSON and resolve every path relative to *path*'s directory."""

    config_path = Path(path).expanduser().resolve(strict=True)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneConfigError(f"cannot read scene config {config_path}: {exc}") from exc
    try:
        if config_path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SceneConfigError(f"invalid scene document {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SceneConfigError("scene document root must be an object/mapping")
    return _build_config(raw, config_path)
