#!/usr/bin/env python3
"""Lift accepted FARM object masks onto immutable source-Ply Gaussian rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "src"))

from farm_runtime.obb_release_evidence import (  # noqa: E402
    validate_serialized_obb_release_evidence,
)

from tools.farm_shaper_bridge.common import (  # noqa: E402
    CONFIG_SCHEMA,
    UNKNOWN_ID,
    FarmObject,
    Frame,
    MaskObservation,
    RunData,
    atomic_json,
    atomic_npy,
    atomic_npz,
    binary_mask_metrics,
    build_verified_csr,
    canonical_sha256,
    diversity_metrics,
    load_mask_pair,
    load_run,
    metric_c2w_to_scene_w2c,
    open_graphdeco_ply,
    project_metric,
    resolve_mask_path,
    resolve_sparse_claims,
    revalidate_run_integrity,
    sha256_file,
    solve_global_timestamp_split,
    utc_now,
    verify_run_ply_fingerprint,
    write_full_labeled_ply,
)
from tools.farm_shaper_bridge.lift_refinement import (  # noqa: E402
    apply_provisional_geometry_gate,
    refine_connected_claims,
)


PRE_LIFT_SELECTION_SCHEMA = "farm.pre-lift-eligibility-selection.v1"
PRE_LIFT_IDS_FILENAME = "pre_lift_eligible_object_ids.txt"
PRE_LIFT_SCOPE_SCHEMA = "farm.gaussian-lift.object-scope.v1"


@dataclass
class GaussianSet:
    means: Any
    scales: Any
    quats: Any
    opacities: Any
    means_scene: np.ndarray
    means_m: np.ndarray
    radius_m: np.ndarray
    opacity_cpu: np.ndarray
    count: int


@dataclass
class ObjectEvidence:
    obj: FarmObject
    indices: np.ndarray
    radius_m: np.ndarray
    build_timestamp_count: int
    positive_weight: np.ndarray
    negative_weight: np.ndarray
    visible_weight: np.ndarray
    positive_timestamps: np.ndarray
    negative_timestamps: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke-object-id", type=int, action="append", default=[])
    parser.add_argument("--allow-legacy-run", action="store_true")
    parser.add_argument(
        "--object-id-file",
        type=Path,
        help=(
            "Exact sorted unique release object IDs emitted by the pre-lift "
            "eligibility selector; requires --pre-lift-selection"
        ),
    )
    parser.add_argument(
        "--pre-lift-selection",
        type=Path,
        help=(
            "PASS farm.pre-lift-eligibility-selection.v1 report that owns "
            "--object-id-file"
        ),
    )
    parser.add_argument(
        "--pre-lift-provenance-artifact",
        type=Path,
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--full-colmap-fold-manifest",
        type=Path,
        help=(
            "Optional external full-COLMAP train/heldout authority. When supplied, "
            "every frame actually consumed by this lift must belong to its train fold."
        ),
    )
    return parser.parse_args(argv)


def _require_sections(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported Gaussian lift config: {config.get('schema_version')!r}")
    for section in (
        "split", "render", "candidate", "mask", "build", "refinement",
        "heldout", "release"
    ):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"Gaussian lift config section missing: {section}")

    def finite(
        section: str,
        key: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_open: bool = False,
        maximum_open: bool = False,
    ) -> float:
        value = config[section].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{section}.{key} must be finite")
        if minimum is not None and (
            number <= minimum if minimum_open else number < minimum
        ):
            raise ValueError(f"{section}.{key} is below its valid range")
        if maximum is not None and (
            number >= maximum if maximum_open else number > maximum
        ):
            raise ValueError(f"{section}.{key} is above its valid range")
        return number

    def integer(section: str, key: str, *, minimum: int = 0) -> int:
        value = config[section].get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
        return value

    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    split = config["split"]
    for key in ("require_early_late_heldout",):
        if not isinstance(split.get(key), bool):
            raise ValueError(f"split.{key} must be boolean")
    minimum_fraction = finite("split", "minimum_heldout_fraction", minimum=0, maximum=1)
    target_fraction = finite("split", "target_heldout_fraction", minimum=0, maximum=1)
    maximum_fraction = finite("split", "maximum_heldout_fraction", minimum=0, maximum=1)
    if not minimum_fraction <= target_fraction <= maximum_fraction:
        raise ValueError("heldout fractions must satisfy minimum <= target <= maximum")
    integer("split", "temporal_bins", minimum=1)
    strict_total = integer("split", "strict_minimum_timestamps", minimum=2)
    strict_build = integer("split", "strict_minimum_build_timestamps", minimum=1)
    strict_heldout = integer("split", "strict_minimum_heldout_timestamps", minimum=1)
    evidence_minimum = integer("split", "evidence_limited_minimum_timestamps", minimum=2)
    if strict_build + strict_heldout > strict_total or evidence_minimum > strict_total:
        raise ValueError("physical-timestamp split minima are inconsistent")
    finite("split", "solver_time_limit_seconds", minimum=0, minimum_open=True)
    finite("split", "build_minimum_baseline_m", minimum=0)
    finite("split", "build_minimum_angular_span_degrees", minimum=0, maximum=180)
    finite("split", "heldout_minimum_baseline_m", minimum=0)
    finite("split", "heldout_minimum_angular_span_degrees", minimum=0, maximum=180)

    if config["render"].get("native_resolution_required") is not True:
        raise ValueError("render.native_resolution_required must be true")
    integer("render", "evidence_reference_long_side", minimum=1)
    maximum = integer("render", "maximum_object_channels_per_call", minimum=1)
    if maximum <= 0 or 2 * maximum + 1 > 32:
        raise ValueError("maximum_object_channels_per_call must satisfy 2*K+1 <= 32")
    finite("render", "invariant_tolerance", minimum=0)
    integer("render", "alignment_guard_views", minimum=1)
    finite("render", "alignment_guard_max_absolute_depth_error_m", minimum=0)

    for key in ("obb_padding_m", "obb_padding_fraction", "depth_absolute_tolerance_m",
                "depth_relative_tolerance", "radius_tolerance_multiplier"):
        finite("candidate", key, minimum=0)
    if config["candidate"].get("domain", "obb") not in ("obb", "obb_mask_surface"):
        raise ValueError("candidate.domain must be obb or obb_mask_surface")
    if config["candidate"].get("domain", "obb") == "obb_mask_surface":
        integer("candidate", "surface_pixel_stride", minimum=1)
        finite("candidate", "surface_voxel_m", minimum=0, minimum_open=True)
    finite("candidate", "minimum_opacity", minimum=0, maximum=1)
    finite("candidate", "maximum_radius_m", minimum=0, minimum_open=True)

    integer("mask", "erosion_pixels", minimum=0)
    integer("mask", "negative_ring_pixels", minimum=0)
    finite("mask", "raw_interior_weight", minimum=0, maximum=1)
    finite("mask", "inlier_weight", minimum=0, minimum_open=True)
    finite("mask", "depth_discontinuity_m", minimum=0)
    integer("mask", "minimum_positive_pixels", minimum=1)

    for key in ("minimum_positive_weight", "minimum_negative_weight"):
        finite("build", key, minimum=0, minimum_open=True)
    for key in ("minimum_per_timestamp_purity", "minimum_global_purity",
                "maximum_object_fraction"):
        finite("build", key, minimum=0, maximum=1)
    integer("build", "minimum_gaussian_timestamps", minimum=1)
    integer("build", "minimum_timestamp_margin", minimum=0)
    finite("build", "score_log_weight", minimum=0)
    finite("build", "minimum_winner_margin", minimum=0)
    finite("build", "minimum_winner_ratio", minimum=1)
    integer("build", "minimum_object_gaussians", minimum=1)

    refinement = config["refinement"]
    if not isinstance(refinement.get("enabled"), bool):
        raise ValueError("refinement.enabled must be boolean")
    integer("refinement", "minimum_seed_gaussians", minimum=1)
    integer("refinement", "minimum_positive_timestamps", minimum=1)
    integer("refinement", "maximum_negative_timestamps", minimum=0)
    integer("refinement", "minimum_timestamp_margin", minimum=0)
    integer("refinement", "minimum_core_neighbors", minimum=1)
    integer("refinement", "maximum_growth_steps", minimum=1)
    finite("refinement", "minimum_positive_weight", minimum=0, minimum_open=True)
    finite("refinement", "minimum_global_purity", minimum=0, maximum=1)
    finite("refinement", "minimum_visible_share", minimum=0, maximum=1)
    finite("refinement", "minimum_connection_radius_m", minimum=0, minimum_open=True)
    maximum_connection_radius = finite(
        "refinement", "maximum_connection_radius_m", minimum=0, minimum_open=True
    )
    if maximum_connection_radius < float(refinement["minimum_connection_radius_m"]):
        raise ValueError("refinement connection radii are inconsistent")
    finite("refinement", "connection_radius_multiplier", minimum=0, minimum_open=True)
    finite("refinement", "maximum_growth_ratio", minimum=0)
    finite("refinement", "score_log_weight", minimum=0)
    finite("refinement", "minimum_winner_margin", minimum=0)
    finite("refinement", "minimum_winner_ratio", minimum=1)

    for key in ("alpha_threshold", "good_timestamp_iou", "minimum_median_iou",
                "minimum_q25_iou", "minimum_median_precision", "minimum_median_recall",
                "minimum_median_largest_component"):
        finite("heldout", key, minimum=0, maximum=1)
    integer("heldout", "minimum_good_timestamps", minimum=1)
    finite("heldout", "maximum_median_depth_error_m", minimum=0)
    finite("heldout", "maximum_p90_depth_error_m", minimum=0)
    integer("heldout", "minimum_depth_pixels", minimum=1)

    if not isinstance(config["release"].get("calibrated"), bool):
        raise ValueError("release.calibrated must be boolean")
    integer("release", "minimum_verified_objects", minimum=1)
    finite("release", "minimum_verified_strict_fraction", minimum=0, maximum=1)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve(strict=True)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Gaussian lift config must be a YAML mapping")
    _require_sections(payload)
    digest = sha256_file(path)
    expected = os.environ.get("FARM_BRIDGE_CONFIG_SHA256")
    if expected and digest != expected:
        raise ValueError("mounted Gaussian lift config differs from launcher authority")
    return payload, digest


def runtime_inventory() -> dict[str, Any]:
    from importlib.metadata import version as package_version
    import scipy
    import torch
    import gsplat
    gsplat_version = getattr(gsplat, "__version__", None) or package_version("gsplat")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gsplat": gsplat_version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_commit": os.environ.get("FARM_BRIDGE_SOURCE_COMMIT"),
        "source_dirty": os.environ.get("FARM_BRIDGE_SOURCE_DIRTY"),
        "source_tree_sha256": os.environ.get("FARM_BRIDGE_SOURCE_TREE_SHA256"),
        "container_image": os.environ.get("FARM_BRIDGE_IMAGE"),
        "container_image_id": os.environ.get("FARM_BRIDGE_IMAGE_ID"),
        "runtime_interpreter_fallback": (
            os.environ.get("FARM_BRIDGE_RUNTIME_INTERPRETER_FALLBACK") == "true"
        ),
    }


def load_gaussians(table, meters_per_scene_unit: float) -> GaussianSet:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact contributor rendering")
    means = np.column_stack((table.data["x"], table.data["y"], table.data["z"])).astype(np.float32)
    log_scales = np.column_stack((
        table.data["scale_0"], table.data["scale_1"], table.data["scale_2"]
    )).astype(np.float32)
    scales = np.exp(log_scales).astype(np.float32)
    quats = np.column_stack((
        table.data["rot_0"], table.data["rot_1"], table.data["rot_2"], table.data["rot_3"]
    )).astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    opacity_logits = np.asarray(table.data["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-np.clip(opacity_logits, -30.0, 30.0)))).astype(np.float32)
    scale = float(meters_per_scene_unit)
    return GaussianSet(
        means=torch.from_numpy(np.ascontiguousarray(means)).cuda(),
        scales=torch.from_numpy(np.ascontiguousarray(scales)).cuda(),
        quats=torch.from_numpy(np.ascontiguousarray(quats)).cuda(),
        opacities=torch.from_numpy(np.ascontiguousarray(opacities)).cuda(),
        means_scene=means,
        means_m=means * scale,
        radius_m=np.clip(scales.max(axis=1) * scale, 0.0, 10.0).astype(np.float32),
        opacity_cpu=opacities,
        count=len(means),
    )


def camera_tensors(run: RunData, frame: Frame):
    import torch
    height, width = frame.depth_size
    view = metric_c2w_to_scene_w2c(frame.T_world_cam, run.meters_per_scene_unit)
    intrinsic = np.asarray(frame.K, dtype=np.float32)
    return (
        torch.from_numpy(view).unsqueeze(0).cuda(),
        torch.from_numpy(intrinsic).unsqueeze(0).cuda(),
        height,
        width,
    )


def raster_kwargs(
    gaussians: GaussianSet,
    run: RunData,
    view,
    intrinsic,
    height: int,
    width: int,
) -> dict[str, Any]:
    rgbd = run.rgbd_config
    return {
        "means": gaussians.means,
        "quats": gaussians.quats,
        "scales": gaussians.scales,
        "opacities": gaussians.opacities,
        "viewmats": view,
        "Ks": intrinsic,
        "width": int(width),
        "height": int(height),
        "near_plane": float(rgbd["depth_min_m"]) / run.meters_per_scene_unit,
        "far_plane": float(rgbd["depth_max_m"]) / run.meters_per_scene_unit,
        "radius_clip": float(rgbd["radius_clip"]),
        "eps2d": 0.3,
        "packed": True,
        "tile_size": 16,
        "backgrounds": None,
        "sparse_grad": False,
        "absgrad": False,
        "rasterize_mode": "antialiased",
        "channel_chunk": 32,
        "distributed": False,
        "camera_model": "pinhole",
        "segmented": False,
    }


def alignment_guard(
    gaussians: GaussianSet,
    run: RunData,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch
    from gsplat import rasterization

    wanted = min(int(config["render"]["alignment_guard_views"]), len(run.frames))
    indices = np.unique(np.rint(np.linspace(0, len(run.frames) - 1, wanted)).astype(int))
    rows: list[dict[str, Any]] = []
    tolerance = float(config["render"]["alignment_guard_max_absolute_depth_error_m"])
    for index in indices:
        frame = run.frames[int(index)]
        view, intrinsic, height, width = camera_tensors(run, frame)
        feature = torch.ones((gaussians.count, 1), dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            rendered, alpha, _ = rasterization(
                **raster_kwargs(gaussians, run, view, intrinsic, height, width),
                colors=feature,
                sh_degree=None,
                render_mode="RGB+ED",
            )
        alpha_np = alpha[0, ..., 0].float().cpu().numpy()
        depth = rendered[0, ..., -1].float().cpu().numpy() * run.meters_per_scene_unit
        valid = (
            np.isfinite(depth)
            & (depth >= float(run.rgbd_config["depth_min_m"]))
            & (depth <= float(run.rgbd_config["depth_max_m"]))
            & (alpha_np >= float(run.rgbd_config["alpha_min"]))
        )
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        expected = np.load(frame.depth_path, mmap_mode="r")
        if expected.shape != (height, width):
            raise ValueError(f"saved depth shape mismatch for image_id={frame.image_id}")
        difference = np.abs(depth - np.asarray(expected, dtype=np.float32))
        maximum = float(difference.max())
        if maximum > tolerance:
            raise RuntimeError(
                f"gsplat/FARM saved-depth parity failed for image_id={frame.image_id}: {maximum}"
            )
        rows.append({
            "image_id": frame.image_id,
            "frame_id": frame.frame_id,
            "physical_timestamp_ns": frame.timestamp_ns,
            "maximum_absolute_error_m": maximum,
            "mean_absolute_error_m": float(difference.mean()),
            "passed": True,
        })
        del rendered, alpha, feature, view, intrinsic
    torch.cuda.empty_cache()
    return rows


def contributor_vjp(
    gaussians: GaussianSet,
    run: RunData,
    frame: Frame,
    pixel_weights: np.ndarray,
    invariant_tolerance: float,
):
    import torch
    from gsplat import rasterization

    weights = np.ascontiguousarray(pixel_weights, dtype=np.float32)
    view, intrinsic, height, width = camera_tensors(run, frame)
    if weights.shape[:2] != (height, width) or not (1 <= weights.shape[2] <= 32):
        raise ValueError("contributor pixel weights violate native-resolution/channel contract")
    query = torch.ones(
        (gaussians.count, weights.shape[2]),
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    rendered, alpha, _ = rasterization(
        **raster_kwargs(gaussians, run, view, intrinsic, height, width),
        colors=query,
        sh_degree=None,
        render_mode="RGB",
    )
    weights_gpu = torch.from_numpy(weights).unsqueeze(0).cuda()
    gradient, = torch.autograd.grad(
        rendered,
        query,
        grad_outputs=weights_gpu,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    error = float((rendered - alpha.expand_as(rendered)).abs().max().detach().cpu())
    minimum = float(gradient.min().detach().cpu())
    if not bool(torch.isfinite(gradient).all()) or minimum < -1e-6 or error > invariant_tolerance:
        raise RuntimeError(
            f"gsplat contributor invariant failed: finite={bool(torch.isfinite(gradient).all())}, "
            f"minimum={minimum}, feature_alpha_error={error}"
        )
    result = gradient.detach()
    alpha_out = alpha.detach()
    del rendered, query, weights_gpu, alpha, gradient, view, intrinsic
    return result, alpha_out, error


def reverse_render(
    gaussians: GaussianSet,
    run: RunData,
    frame: Frame,
    object_ids: Sequence[int],
    indices_by_id: Mapping[int, np.ndarray],
):
    import torch
    from gsplat import rasterization

    if not object_ids or 2 * len(object_ids) + 1 > 32:
        raise ValueError("reverse-render object batch violates channel contract")
    view, intrinsic, height, width = camera_tensors(run, frame)
    count = len(object_ids)
    membership = torch.zeros((gaussians.count, count), dtype=torch.float32, device="cuda")
    z_scene = gaussians.means @ view[0, 2, :3] + view[0, 2, 3]
    for column, object_id in enumerate(object_ids):
        indices = torch.as_tensor(indices_by_id[object_id], dtype=torch.long, device="cuda")
        membership[indices, column] = 1.0
    features = torch.cat((membership, membership * z_scene[:, None]), dim=1)
    with torch.inference_mode():
        rendered, alpha, _ = rasterization(
            **raster_kwargs(gaussians, run, view, intrinsic, height, width),
            colors=features,
            sh_degree=None,
            render_mode="RGB+ED",
        )
    mass = rendered[0, ..., :count].float()
    numerator = rendered[0, ..., count:2*count].float()
    object_depth_m = torch.where(
        mass > 1e-8,
        numerator / mass.clamp_min(1e-8),
        torch.full_like(mass, float("nan")),
    ) * run.meters_per_scene_unit
    share = mass / alpha[0].clamp_min(1e-8)
    result = (
        mass.cpu().numpy(),
        share.cpu().numpy(),
        object_depth_m.cpu().numpy(),
        rendered[0, ..., -1:].float().cpu().numpy() * run.meters_per_scene_unit,
        alpha[0].float().cpu().numpy(),
    )
    del rendered, alpha, features, membership, numerator, mass, share, object_depth_m
    del z_scene, view, intrinsic
    return result


def make_split(run: RunData, config: Mapping[str, Any]) -> dict[str, Any]:
    object_timestamps: dict[int, list[str]] = {}
    used: set[str] = set()
    for obj in run.objects:
        values = sorted(
            {run.frame(observation.image_id).physical_timestamp for observation in obj.observations},
            key=int,
        )
        object_timestamps[obj.object_id] = values
        used.update(values)
    ordered = sorted(used, key=int)
    split = solve_global_timestamp_split(
        ordered,
        object_timestamps,
        config["split"],
        seed=int(config["seed"]),
    )
    object_lookup = {obj.object_id: obj for obj in run.objects}
    for row in split["objects"]:
        obj = object_lookup[int(row["object_id"])]
        timestamp_frames: dict[str, list[Frame]] = defaultdict(list)
        for observation in obj.observations:
            frame = run.frame(observation.image_id)
            timestamp_frames[frame.physical_timestamp].append(frame)

        def fold_details(values: Sequence[str]) -> dict[str, Any]:
            centres: list[np.ndarray] = []
            sensors: set[str] = set()
            families: set[str] = set()
            image_ids: list[int] = []
            for value in values:
                frames = timestamp_frames[value]
                centres.append(np.mean([frame.T_world_cam[:3, 3] for frame in frames], axis=0))
                sensors.update(frame.sensor for frame in frames)
                families.update(frame.family for frame in frames)
                image_ids.extend(frame.image_id for frame in frames)
            metrics = diversity_metrics(centres, obj.center_m)
            return {
                **metrics,
                "image_ids": sorted(set(image_ids)),
                "sensors": sorted(sensors),
                "families": sorted(families),
            }

        row["build"] = fold_details(row["build_timestamps"])
        row["heldout"] = fold_details(row["heldout_timestamps"])
        policy = config["split"]
        build_diverse = (
            row["build"]["baseline_span_m"] >= float(policy["build_minimum_baseline_m"])
            or row["build"]["angular_span_degrees"] >= float(policy["build_minimum_angular_span_degrees"])
        )
        heldout_diverse = (
            row["heldout"]["baseline_span_m"] >= float(policy["heldout_minimum_baseline_m"])
            or row["heldout"]["angular_span_degrees"] >= float(policy["heldout_minimum_angular_span_degrees"])
        )
        row["diversity_pass"] = bool(build_diverse and heldout_diverse)
        row["strict_eligible"] = bool(row["eligibility"] == "strict" and row["diversity_pass"])
        if row["eligibility"] == "strict" and not row["diversity_pass"]:
            row["eligibility"] = "diversity_limited"
    split["scene_id"] = run.scene_id
    split["global_disjoint"] = not bool(
        set(split["build_timestamps"]).intersection(split["heldout_timestamps"])
    )
    if not split["global_disjoint"]:
        raise AssertionError("global build/heldout timestamps overlap")
    return split


def _exact_object_ids(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{label} must be a sorted unique JSON integer list")
    normalized = list(value)
    if any(item < 0 for item in normalized):
        raise ValueError(f"{label} contains a negative object ID")
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{label} must be sorted and unique")
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty for a release lift")
    return normalized


def _load_exact_object_id_artifact(path: Path) -> list[int]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"object ID artifact is not a regular file: {resolved}")
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("object ID artifact is not valid UTF-8") from exc
    lines = text.splitlines()
    values: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or not line.isdecimal():
            raise ValueError(
                f"object ID artifact line {line_number} is not canonical decimal"
            )
        value = int(line)
        if line != str(value):
            raise ValueError(
                f"object ID artifact line {line_number} is not canonical decimal"
            )
        values.append(value)
    normalized = _exact_object_ids(values, label="object ID artifact")
    canonical = "".join(f"{value}\n" for value in normalized)
    if text != canonical:
        raise ValueError(
            "object ID artifact must contain exactly one canonical ID per line "
            "and end with a newline"
        )
    return normalized


def _sha256_value(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not a lowercase SHA256")
    return digest


def _artifact_descriptor_with_path(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _normalized_provenance_spec(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    role = str(value.get("role") or "")
    raw_path = str(value.get("path") or "")
    if not role or not raw_path or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} must have a role and absolute source path")
    raw_bytes = value.get("bytes")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes < 0:
        raise ValueError(f"{label}.bytes must be a non-negative integer")
    spec: dict[str, Any] = {
        "role": role,
        "path": raw_path,
        "bytes": raw_bytes,
        "sha256": _sha256_value(value.get("sha256"), label=f"{label}.sha256"),
    }
    if "input_index" in value:
        raw_index = value.get("input_index")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
        ):
            raise ValueError(f"{label}.input_index must be non-negative")
        spec["input_index"] = raw_index
    return spec


def validate_release_object_scope(
    run: RunData,
    selection_path: Path,
    object_id_path: Path,
    provenance_artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    """Bind one exact pre-lift allowlist to the active FARM object universe.

    The launcher mounts every source artifact declared by the selector at an
    isolated read-only path. The signed lift implementation re-hashes those
    mounts here, so stale or substituted eligibility evidence cannot authorize
    Gaussian owners.
    """

    selection_resolved = Path(selection_path).expanduser().resolve(strict=True)
    ids_resolved = Path(object_id_path).expanduser().resolve(strict=True)
    if not selection_resolved.is_file() or not ids_resolved.is_file():
        raise ValueError("pre-lift selection and object ID artifact must be files")
    try:
        selection = json.loads(selection_resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("pre-lift selection is not valid UTF-8 JSON") from exc
    if not isinstance(selection, Mapping):
        raise ValueError("pre-lift selection must be a JSON object")
    if selection.get("schema") != PRE_LIFT_SELECTION_SCHEMA:
        raise ValueError(
            f"pre-lift selection schema must be {PRE_LIFT_SELECTION_SCHEMA}"
        )
    if selection.get("status") != "PASS":
        raise ValueError("pre-lift selection status must be PASS")
    if (
        selection.get("selection_kind") != "pre_lift_eligibility_only"
        or selection.get("gaussian_mask_improvement_proven") is not False
    ):
        raise ValueError("pre-lift selection kind/proof contract is invalid")

    integrity = selection.get("integrity")
    policy = selection.get("policy")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("physical_timestamp_sets_disjoint") is not True
        or integrity.get("train_heldout_physical_timestamp_overlap") != []
        or not isinstance(policy, Mapping)
        or policy.get("geometry_release_evidence_required") is not True
        or policy.get(
            "legacy_geometry_pass_without_release_evidence_is_not_publishable"
        ) is not True
        or policy.get("heldout_state_fit_authorized") is not False
        or policy.get("train_heldout_physical_timestamp_overlap_forbidden")
        is not True
        or policy.get("post_lift_gaussian_mask_qc_required") is not True
        or policy.get(
            "pre_lift_eligibility_is_not_gaussian_mask_improvement_proof"
        )
        is not True
    ):
        raise ValueError("pre-lift selection fold/policy contract is incomplete")

    eligible_ids = _exact_object_ids(
        selection.get("eligible_object_ids"),
        label="selection eligible_object_ids",
    )
    rejected_ids = _exact_object_ids(
        selection.get("rejected_object_ids"),
        label="selection rejected_object_ids",
        allow_empty=True,
    )
    if set(eligible_ids).intersection(rejected_ids):
        raise ValueError("selection eligible/rejected object IDs overlap")
    rows = selection.get("objects")
    if not isinstance(rows, list):
        raise ValueError("pre-lift selection objects must be a list")
    row_ids: list[int] = []
    derived_eligible: list[int] = []
    derived_rejected: list[int] = []
    selection_rows_by_id: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"pre-lift selection object row {index} is invalid")
        object_id = row.get("object_id")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id < 0
            or not isinstance(row.get("eligible"), bool)
        ):
            raise ValueError(f"pre-lift selection object row {index} is invalid")
        row_ids.append(object_id)
        selection_rows_by_id[object_id] = row
        if row["eligible"]:
            evidence_failures = validate_serialized_obb_release_evidence(
                row.get("geometry_release_evidence")
            )
            if (
                row.get("geometry_status") != "geometry_pass"
                or row.get("reasons") != []
                or evidence_failures
            ):
                raise ValueError(
                    f"eligible object {object_id} lacks passed OBB release evidence"
                )
        (derived_eligible if row["eligible"] else derived_rejected).append(object_id)
    if row_ids != sorted(set(row_ids)):
        raise ValueError("pre-lift selection object rows must be sorted and unique")
    if derived_eligible != eligible_ids or derived_rejected != rejected_ids:
        raise ValueError(
            "selection eligible/rejected IDs contradict its object rows"
        )

    exact_ids = _load_exact_object_id_artifact(ids_resolved)
    if exact_ids != eligible_ids:
        raise ValueError(
            "object ID artifact does not exactly equal eligible_object_ids"
        )
    outputs = selection.get("outputs")
    exact_output = (
        outputs.get("exact_object_ids") if isinstance(outputs, Mapping) else None
    )
    if not isinstance(exact_output, Mapping):
        raise ValueError("pre-lift selection lacks exact object ID output provenance")
    relative = Path(str(exact_output.get("relative_path") or ""))
    if (
        relative.as_posix() != PRE_LIFT_IDS_FILENAME
        or ids_resolved != (selection_resolved.parent / relative).resolve(strict=True)
        or exact_output.get("format") != "one_decimal_object_id_per_line"
        or isinstance(exact_output.get("bytes"), bool)
        or exact_output.get("bytes") != ids_resolved.stat().st_size
        or _sha256_value(
            exact_output.get("sha256"), label="object ID artifact SHA256"
        )
        != sha256_file(ids_resolved)
    ):
        raise ValueError("object ID artifact does not match selection provenance")

    active_ids = sorted(obj.object_id for obj in run.objects)
    if active_ids != sorted(set(active_ids)):
        raise ValueError("active FARM object universe is not unique")
    inactive_eligible = sorted(set(eligible_ids).difference(active_ids))
    if inactive_eligible:
        raise ValueError(
            f"eligible object IDs are inactive/absent: {inactive_eligible}"
        )
    active_ids_not_in_selection = sorted(set(active_ids).difference(row_ids))
    forced_exclusion_audit = [
        {
            "object_id": object_id,
            "eligible": False,
            "reason": "missing_from_full_train_refinement_universe",
        }
        for object_id in active_ids_not_in_selection
    ]
    scope_audited_ids = set(row_ids).union(
        row["object_id"] for row in forced_exclusion_audit
    )
    unaudited_active = sorted(set(active_ids).difference(scope_audited_ids))
    if unaudited_active:
        raise ValueError(
            f"active FARM object IDs are not audited by selection: {unaudited_active}"
        )

    provenance = selection.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    consumed = (
        provenance.get("consumed_artifacts")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("pre-lift selection lacks source input provenance")
    if not isinstance(consumed, list) or not consumed:
        raise ValueError("pre-lift selection lacks consumed artifact provenance")
    consumed_specs = [
        _normalized_provenance_spec(value, label=f"consumed artifact {index}")
        for index, value in enumerate(consumed)
    ]
    if consumed_specs != sorted(
        consumed_specs, key=lambda spec: (spec["role"], spec["path"])
    ):
        raise ValueError("consumed artifact provenance is not canonically sorted")
    declared_paths = [spec["path"] for spec in consumed_specs]
    if len(declared_paths) != len(set(declared_paths)):
        raise ValueError("consumed artifact provenance contains duplicate paths")
    mounted = [
        Path(path).expanduser().resolve(strict=True)
        for path in provenance_artifact_paths
    ]
    if len(mounted) != len(consumed_specs):
        raise ValueError(
            "mounted provenance artifact count does not match selection"
        )
    for index, (spec, path) in enumerate(zip(consumed_specs, mounted, strict=True)):
        if not path.is_file():
            raise ValueError(f"mounted provenance artifact {index} is not a file")
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"source input size mismatch: {spec['path']}")
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"source input SHA256 mismatch: {spec['path']}")

    input_specs = [
        _normalized_provenance_spec(value, label=f"source input {index}")
        for index, value in enumerate(inputs)
    ]
    consumed_by_path = {
        spec["path"]: (spec, mounted[index])
        for index, spec in enumerate(consumed_specs)
    }
    for spec in input_specs:
        consumed_match = consumed_by_path.get(spec["path"])
        if consumed_match is None or consumed_match[0] != spec:
            raise ValueError("source input provenance is not exactly consumed")
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in input_specs:
        by_role[spec["role"]].append(spec)
    expected_roles = {
        "train_refinement_result",
        "geometry_audit",
        "heldout_reference_result",
    }
    if set(by_role) != expected_roles:
        raise ValueError("pre-lift source input roles are incomplete or unknown")
    if (
        len(by_role["train_refinement_result"]) != 1
        or by_role["train_refinement_result"][0].get("input_index") != 0
        or len(by_role["geometry_audit"]) != 1
        or by_role["geometry_audit"][0].get("input_index") != 0
        or [
            spec.get("input_index")
            for spec in by_role["heldout_reference_result"]
        ]
        != list(range(len(by_role["heldout_reference_result"])))
    ):
        raise ValueError("pre-lift source input indices are not canonical")

    train_spec = by_role["train_refinement_result"][0]
    train_mount = consumed_by_path[train_spec["path"]][1]
    try:
        train_payload = json.loads(train_mount.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("train refinement source input is not valid JSON") from exc
    train_rows = (
        train_payload.get("objects") if isinstance(train_payload, Mapping) else None
    )
    if (
        not isinstance(train_payload, Mapping)
        or train_payload.get("schema") != "farm.full-colmap-mask-refinement.v1"
        or train_payload.get("status") != "PASS"
        or train_payload.get("refinement_role") != "train_fit"
        or not isinstance(train_rows, list)
    ):
        raise ValueError("train refinement source contract is invalid")
    train_ids: list[int] = []
    for row in train_rows:
        if not isinstance(row, Mapping):
            raise ValueError("train refinement object universe is invalid")
        object_id = row.get("object_id")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id < 0
        ):
            raise ValueError("train refinement object universe is invalid")
        train_ids.append(object_id)
    if len(train_ids) != len(set(train_ids)) or sorted(train_ids) != row_ids:
        raise ValueError(
            "selection object universe is not bound to train refinement input"
        )

    geometry_spec = by_role["geometry_audit"][0]
    geometry_mount = consumed_by_path[geometry_spec["path"]][1]
    try:
        geometry_payload = json.loads(geometry_mount.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("geometry audit source input is not valid JSON") from exc
    geometry_rows = (
        geometry_payload.get("objects")
        if isinstance(geometry_payload, Mapping)
        else None
    )
    if (
        not isinstance(geometry_payload, Mapping)
        or geometry_payload.get("schema") != "farm.object-geometry-audit.v3"
        or not isinstance(geometry_rows, list)
    ):
        raise ValueError("geometry audit source contract is invalid")
    geometry_by_id: dict[int, Mapping[str, Any]] = {}
    for row in geometry_rows:
        if not isinstance(row, Mapping):
            raise ValueError("geometry audit object rows are invalid")
        object_id = row.get("object_id")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id < 0
            or object_id in geometry_by_id
        ):
            raise ValueError("geometry audit object rows are invalid")
        geometry_by_id[object_id] = row
    for object_id in eligible_ids:
        geometry_row = geometry_by_id.get(object_id)
        selection_evidence = selection_rows_by_id[object_id].get(
            "geometry_release_evidence"
        )
        if (
            geometry_row is None
            or geometry_row.get("status") != "geometry_pass"
            or validate_serialized_obb_release_evidence(
                geometry_row.get("train_geometry_evidence")
            )
            or geometry_row.get("train_geometry_evidence") != selection_evidence
        ):
            raise ValueError(
                f"eligible object {object_id} is not bound to passed geometry audit evidence"
            )
    for spec in input_specs:
        payload_path = consumed_by_path[spec["path"]][1]
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, Mapping)
            and payload.get("scene_id") is not None
            and str(payload.get("scene_id")) != run.scene_id
        ):
            raise ValueError(
                f"source input scene_id mismatch for role {spec['role']}"
            )

    return {
        "schema": PRE_LIFT_SCOPE_SCHEMA,
        "status": "PASS",
        "mode": "pre_lift_release_allowlist",
        "active_object_ids": eligible_ids,
        "excluded_active_object_ids": sorted(
            set(active_ids).difference(eligible_ids)
        ),
        "inactive_rejected_object_ids": sorted(
            set(rejected_ids).difference(active_ids)
        ),
        "active_ids_not_in_selection": active_ids_not_in_selection,
        "forced_exclusion_audit": forced_exclusion_audit,
        "source_active_object_ids": active_ids,
        "selection_object_ids": row_ids,
        "source_active_object_universe_sha256": canonical_sha256(
            {"object_ids": active_ids}
        ),
        "selection_object_universe_sha256": canonical_sha256(
            {"object_ids": row_ids}
        ),
        "artifacts": {
            "object_id_file": _artifact_descriptor_with_path(ids_resolved),
            "pre_lift_selection": _artifact_descriptor_with_path(
                selection_resolved
            ),
        },
        "source_inputs": input_specs,
        "consumed_artifact_count": len(consumed_specs),
        "consumed_artifacts_sha256": canonical_sha256(consumed_specs),
    }


def restrict_release_scope(
    run: RunData,
    split: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> tuple[RunData, dict[str, Any]]:
    selected = set(
        _exact_object_ids(
            scope.get("active_object_ids"),
            label="release scope active_object_ids",
        )
    )
    scoped_objects = tuple(obj for obj in run.objects if obj.object_id in selected)
    scoped_split = dict(split)
    scoped_split["objects"] = [
        dict(row) for row in split["objects"] if int(row["object_id"]) in selected
    ]
    if len(scoped_objects) != len(selected) or len(scoped_split["objects"]) != len(selected):
        raise AssertionError("release scope is not one-to-one with full-run split rows")
    scoped_split["scope"] = "release_allowlist_subset_of_canonical_full_split"
    scoped_split["full_object_count"] = len(run.objects)
    scoped_split["active_object_ids"] = sorted(selected)
    return replace(run, objects=scoped_objects), scoped_split


def assert_gaussian_owner_scope(
    labels: np.ndarray,
    active_object_ids: Sequence[int],
    *,
    label: str,
) -> None:
    allowed = {int(value) for value in active_object_ids}
    owners = {
        int(value)
        for value in np.unique(np.asarray(labels, dtype=np.int64))
        if int(value) != UNKNOWN_ID
    }
    unexpected = sorted(owners.difference(allowed))
    if unexpected:
        raise AssertionError(f"{label} contains excluded Gaussian owners: {unexpected}")


def restrict_smoke_scope(
    run: RunData,
    split: Mapping[str, Any],
    object_ids: Sequence[int],
) -> tuple[RunData, dict[str, Any]]:
    """Restrict work after freezing the canonical full-run timestamp split.

    A smoke run must exercise exactly the same build/held-out partition as the
    full release run. Re-solving the MILP on one object is both non-representative
    and often infeasible because the global temporal-bin constraints no longer
    fit that object's small observation set.
    """

    selected = {int(value) for value in object_ids}
    available = {obj.object_id for obj in run.objects}
    missing = sorted(selected - available)
    if missing:
        raise ValueError(f"smoke object IDs absent from presentation catalog: {missing}")
    if not selected:
        return run, dict(split)
    scoped_objects = tuple(obj for obj in run.objects if obj.object_id in selected)
    scoped_split = dict(split)
    scoped_split["objects"] = [
        dict(row) for row in split["objects"] if int(row["object_id"]) in selected
    ]
    if len(scoped_objects) != len(selected) or len(scoped_split["objects"]) != len(selected):
        raise AssertionError("smoke scope is not one-to-one with full-run split rows")
    scoped_split["scope"] = "smoke_subset_of_canonical_full_split"
    scoped_split["full_object_count"] = len(run.objects)
    scoped_split["smoke_object_ids"] = sorted(selected)
    return replace(run, objects=scoped_objects), scoped_split


def build_candidates(
    run: RunData,
    gaussians: GaussianSet,
    split: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[int, ObjectEvidence], list[dict[str, Any]]]:
    policy = config["candidate"]
    tree = cKDTree(
        gaussians.means_m,
        compact_nodes=False,
        balanced_tree=False,
        copy_data=False,
    )
    split_rows = {int(row["object_id"]): row for row in split["objects"]}
    evidence: dict[int, ObjectEvidence] = {}
    rows: list[dict[str, Any]] = []
    for obj in run.objects:
        padding = np.maximum(
            float(policy["obb_padding_m"]),
            obj.dimensions_m * float(policy["obb_padding_fraction"]),
        )
        half = 0.5 * obj.dimensions_m + padding
        radius = float(np.linalg.norm(half))
        approximate = np.asarray(
            tree.query_ball_point(obj.center_m, radius, workers=-1),
            dtype=np.int64,
        )
        if approximate.size:
            local = (gaussians.means_m[approximate] - obj.center_m) @ obj.rotation
            inside = np.all(np.abs(local) <= half, axis=1)
            inside &= gaussians.opacity_cpu[approximate] >= float(policy["minimum_opacity"])
            indices = np.sort(approximate[inside])
        else:
            indices = np.zeros(0, dtype=np.int64)
        obb_count = len(indices)
        expansion = {"domain": policy.get("domain", "obb"), "obb_candidates": obb_count}
        if policy.get("domain", "obb") == "obb_mask_surface":
            surface_ids, surface_audit = mask_surface_candidates(run, obj, gaussians, split, config)
            indices = np.union1d(indices, surface_ids)
            expansion.update(surface_audit)
            expansion["added_outside_obb"] = len(indices) - obb_count
        count = len(indices)
        evidence[obj.object_id] = ObjectEvidence(
            obj=obj,
            indices=indices,
            radius_m=np.minimum(
                gaussians.radius_m[indices],
                float(policy["maximum_radius_m"]),
            ),
            build_timestamp_count=len(split_rows[obj.object_id]["build_timestamps"]),
            positive_weight=np.zeros(count, dtype=np.float32),
            negative_weight=np.zeros(count, dtype=np.float32),
            visible_weight=np.zeros(count, dtype=np.float32),
            positive_timestamps=np.zeros(count, dtype=np.uint16),
            negative_timestamps=np.zeros(count, dtype=np.uint16),
        )
        rows.append({
            "object_id": obj.object_id,
            "category": obj.category,
            "candidate_gaussians": count,
            "search": expansion,
            "obb_padding_m": padding.astype(float).tolist(),
        })
    del tree
    return evidence, rows


def mask_surface_candidates(run, obj, gaussians, split, config):
    from tools.farm_shaper_bridge.lift_candidates import backproject_mask_surface, surface_candidate_indices
    policy = config["candidate"]
    # The same global physical-timestamp split owns both candidate construction
    # and VJP fitting. Per-object observations cannot override the heldout fold.
    allowed = set(split["build_timestamps"])
    grouped = defaultdict(list)
    for observation in obj.observations:
        frame = run.frame(observation.image_id)
        if frame.physical_timestamp in allowed:
            grouped[frame.image_id].append(observation)
    seeds, sources = [], []
    max_depth, max_spacing = 0., 0.
    for image_id, observations in sorted(grouped.items()):
        frame = run.frame(image_id)
        decoded, _, audit, depth_audit = _load_view_masks(run, frame, {obj.object_id: observations}, config)
        depth = np.load(frame.depth_path, mmap_mode="r")
        mask = decoded[obj.object_id]["raw"] & _surface_interior(depth, run, config)
        points, far_depth, spacing = backproject_mask_surface(mask, depth, frame.K, frame.T_world_cam,
            stride=int(policy["surface_pixel_stride"]), depth_min=float(run.rgbd_config["depth_min_m"]),
            depth_max=float(run.rgbd_config["depth_max_m"]))
        seeds.append(points)
        max_depth, max_spacing = max(max_depth, far_depth), max(max_spacing, spacing)
        sources.append({"image_id": image_id, "physical_timestamp": frame.physical_timestamp,
                        "seed_points": len(points), "masks": audit, "depth": depth_audit})
    seeds_array = np.concatenate(seeds) if seeds else np.empty((0, 3))
    tolerance = float(policy["depth_absolute_tolerance_m"]) + float(policy["depth_relative_tolerance"])*max_depth + max_spacing
    indices, audit = surface_candidate_indices(gaussians.means_m, gaussians.radius_m, gaussians.opacity_cpu,
        seeds_array, minimum_opacity=float(policy["minimum_opacity"]), maximum_radius=float(policy["maximum_radius_m"]),
        radius_multiplier=float(policy["radius_tolerance_multiplier"]), surface_tolerance=tolerance,
        voxel_size=float(policy["surface_voxel_m"]))
    return indices, {**audit, "surface_candidates": len(indices), "source_views": sources,
                     "heldout_consumed": False, "ownership": "requires unchanged exact VJP and multiview gates"}


def _surface_interior(depth: np.ndarray, run: RunData, config: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = (
        np.isfinite(values)
        & (values >= float(run.rgbd_config["depth_min_m"]))
        & (values <= float(run.rgbd_config["depth_max_m"]))
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    valid_interior = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    maximum = cv2.dilate(np.where(valid, values, 0).astype(np.float32), kernel)
    minimum_input = np.where(valid, values, np.finfo(np.float32).max).astype(np.float32)
    minimum = cv2.erode(minimum_input, kernel)
    discontinuity = maximum - minimum
    return valid_interior & (discontinuity <= float(config["mask"]["depth_discontinuity_m"]))


def _load_view_masks(
    run: RunData,
    frame: Frame,
    observations_by_object: Mapping[int, Sequence[MaskObservation]],
    config: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    depth = np.load(frame.depth_path, mmap_mode="r")
    if depth.shape != frame.depth_size:
        raise ValueError(f"depth shape mismatch for image_id={frame.image_id}")
    surface = _surface_interior(depth, run, config)
    decoded: dict[int, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for object_id, observations in observations_by_object.items():
        raw = np.zeros(frame.depth_size, dtype=bool)
        inlier = np.zeros(frame.depth_size, dtype=bool)
        hashes: list[dict[str, Any]] = []
        for observation in observations:
            path = resolve_mask_path(run, observation)
            child_raw, child_inlier, digest = load_mask_pair(path, frame.depth_size)
            raw |= child_raw
            inlier |= child_inlier
            hashes.append({
                "path": (path.relative_to(run.run_dir).as_posix() if path.is_relative_to(run.run_dir) else path.as_posix()),
                "bytes": path.stat().st_size,
                "sha256": digest,
            })
        decoded[object_id] = {"raw": raw, "inlier": inlier}
        audit.append({
            "object_id": object_id,
            "raw_pixels": int(raw.sum()),
            "inlier_pixels": int(inlier.sum()),
            "masks": hashes,
        })
    all_raw = np.zeros(frame.depth_size, dtype=bool)
    for item in decoded.values():
        all_raw |= item["raw"]
    reference = float(config["render"]["evidence_reference_long_side"])
    normalizer = (reference / max(frame.depth_size)) ** 2
    kernel_erode = np.ones((3, 3), dtype=np.uint8)
    ring_pixels = int(config["mask"]["negative_ring_pixels"])
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_pixels + 1, 2 * ring_pixels + 1),
    )
    for object_id, item in decoded.items():
        raw = item["raw"]
        eroded = cv2.erode(
            raw.astype(np.uint8),
            kernel_erode,
            iterations=int(config["mask"]["erosion_pixels"]),
        ).astype(bool)
        positive = eroded.astype(np.float32) * float(config["mask"]["raw_interior_weight"])
        positive[item["inlier"] & raw] = float(config["mask"]["inlier_weight"])
        positive *= surface
        if int(np.count_nonzero(positive)) < int(config["mask"]["minimum_positive_pixels"]):
            positive = (raw & surface).astype(np.float32) * float(config["mask"]["raw_interior_weight"])
        dilated = cv2.dilate(raw.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
        other = all_raw & ~raw
        negative = dilated & ~raw & ~other & surface
        item["positive_weight"] = positive * normalizer
        item["negative_weight"] = negative.astype(np.float32) * normalizer
    depth_audit = {
        "path": (frame.depth_path.relative_to(run.run_dir).as_posix() if frame.depth_path.is_relative_to(run.run_dir) else frame.depth_path.as_posix()),
        "bytes": frame.depth_path.stat().st_size,
        "sha256": sha256_file(frame.depth_path),
    }
    return decoded, surface.astype(np.float32) * normalizer, audit, depth_audit


def _depth_gate(
    gaussians: GaussianSet,
    evidence: ObjectEvidence,
    frame: Frame,
    depth: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    if not len(evidence.indices):
        return np.zeros(0, dtype=bool)
    points = gaussians.means_m[evidence.indices]
    u, v, z = project_metric(points, frame.T_world_cam, frame.K)
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    height, width = frame.depth_size
    inside = (
        (z > float(config["candidate"]["depth_absolute_tolerance_m"]))
        & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    )
    selected = np.flatnonzero(inside)
    result = np.zeros(len(evidence.indices), dtype=bool)
    if not len(selected):
        return result
    reference = np.asarray(depth[vi[selected], ui[selected]], dtype=np.float32)
    tolerance = np.maximum(
        float(config["candidate"]["depth_absolute_tolerance_m"])
        + float(config["candidate"]["depth_relative_tolerance"]) * reference,
        float(config["candidate"]["radius_tolerance_multiplier"]) * evidence.radius_m[selected],
    )
    valid = (
        np.isfinite(reference)
        & (reference >= float(config["candidate"]["depth_absolute_tolerance_m"]))
        & (np.abs(z[selected] - reference) <= tolerance)
    )
    result[selected[valid]] = True
    return result


def accumulate_build_evidence(
    gaussians: GaussianSet,
    run: RunData,
    split: Mapping[str, Any],
    evidence: Mapping[int, ObjectEvidence],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    import torch

    build_timestamps = set(split["build_timestamps"])
    grouped: dict[str, dict[int, dict[int, list[MaskObservation]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for obj in run.objects:
        for observation in obj.observations:
            frame = run.frame(observation.image_id)
            if frame.physical_timestamp in build_timestamps:
                grouped[frame.physical_timestamp][frame.image_id][obj.object_id].append(observation)

    rows: list[dict[str, Any]] = []
    render_vjp_seconds = 0.0
    maximum_objects = int(config["render"]["maximum_object_channels_per_call"])
    invariant_tolerance = float(config["render"]["invariant_tolerance"])
    for timestamp_rank, timestamp in enumerate(sorted(grouped, key=int), 1):
        object_ids_at_timestamp = sorted({
            object_id
            for image_rows in grouped[timestamp].values()
            for object_id in image_rows
        })
        buffers = {
            object_id: {
                "positive": np.zeros(len(evidence[object_id].indices), dtype=np.float32),
                "negative": np.zeros(len(evidence[object_id].indices), dtype=np.float32),
                "visible": np.zeros(len(evidence[object_id].indices), dtype=np.float32),
            }
            for object_id in object_ids_at_timestamp
        }
        timestamp_row = {"physical_timestamp_ns": int(timestamp), "views": []}
        for image_id, by_object in sorted(grouped[timestamp].items()):
            frame = run.frame(image_id)
            decoded, total_surface, mask_audit, depth_audit = _load_view_masks(
                run, frame, by_object, config
            )
            depth = np.load(frame.depth_path, mmap_mode="r")
            ids = sorted(decoded)
            view_row = {
                "image_id": image_id,
                "camera": frame.camera,
                "sensor": frame.sensor,
                "family": frame.family,
                "masks": mask_audit,
                "depth": depth_audit,
                "batches": [],
            }
            for batch_start in range(0, len(ids), maximum_objects):
                batch_ids = ids[batch_start:batch_start + maximum_objects]
                channels = 2 * len(batch_ids) + 1
                weights = np.zeros((*frame.depth_size, channels), dtype=np.float32)
                for column, object_id in enumerate(batch_ids):
                    weights[..., 2 * column] = decoded[object_id]["positive_weight"]
                    weights[..., 2 * column + 1] = decoded[object_id]["negative_weight"]
                weights[..., -1] = total_surface
                torch.cuda.synchronize()
                started = time.perf_counter()
                gradient, alpha, invariant_error = contributor_vjp(
                    gaussians,
                    run,
                    frame,
                    weights,
                    invariant_tolerance,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                render_vjp_seconds += elapsed
                batch_row = {
                    "object_ids": batch_ids,
                    "channels": channels,
                    "seconds": elapsed,
                    "feature_alpha_max_error": invariant_error,
                }
                for column, object_id in enumerate(batch_ids):
                    item = evidence[object_id]
                    if not len(item.indices):
                        continue
                    index_gpu = torch.as_tensor(item.indices, dtype=torch.long, device="cuda")
                    selected = gradient.index_select(0, index_gpu)[
                        :, [2 * column, 2 * column + 1, channels - 1]
                    ].float().cpu().numpy()
                    gate = _depth_gate(gaussians, item, frame, depth, config)
                    selected[~gate] = 0.0
                    buffers[object_id]["positive"] = np.maximum(
                        buffers[object_id]["positive"], selected[:, 0]
                    )
                    buffers[object_id]["negative"] = np.maximum(
                        buffers[object_id]["negative"], selected[:, 1]
                    )
                    buffers[object_id]["visible"] = np.maximum(
                        buffers[object_id]["visible"], selected[:, 2]
                    )
                view_row["batches"].append(batch_row)
                del gradient, alpha, weights
            timestamp_row["views"].append(view_row)
            torch.cuda.empty_cache()

        build_policy = config["build"]
        for object_id, values in buffers.items():
            item = evidence[object_id]
            positive = values["positive"]
            negative = values["negative"]
            purity = positive / np.maximum(positive + negative, positive + 1e-8)
            negative_purity = negative / np.maximum(positive + negative, negative + 1e-8)
            strong_positive = (
                (positive >= float(build_policy["minimum_positive_weight"]))
                & (purity >= float(build_policy["minimum_per_timestamp_purity"]))
            )
            strong_negative = (
                (negative >= float(build_policy["minimum_negative_weight"]))
                & (negative_purity >= float(build_policy["minimum_per_timestamp_purity"]))
            )
            if np.any(item.positive_timestamps[strong_positive] == np.iinfo(np.uint16).max):
                raise OverflowError("positive physical-timestamp support exceeds uint16")
            if np.any(item.negative_timestamps[strong_negative] == np.iinfo(np.uint16).max):
                raise OverflowError("negative physical-timestamp support exceeds uint16")
            item.positive_timestamps[strong_positive] += 1
            item.negative_timestamps[strong_negative] += 1
            item.positive_weight += positive
            item.negative_weight += negative
            item.visible_weight += values["visible"]
        rows.append(timestamp_row)
        print(json.dumps({
            "stage": "build",
            "timestamp": timestamp_rank,
            "total": len(grouped),
            "physical_timestamp_ns": int(timestamp),
            "views": len(grouped[timestamp]),
            "objects": len(object_ids_at_timestamp),
        }, separators=(",", ":")), flush=True)
    return rows, {"render_and_vjp": render_vjp_seconds}


def make_claims(
    evidence: Mapping[int, ObjectEvidence],
    gaussian_count: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    policy = config["build"]
    claims: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for object_id in sorted(evidence):
        item = evidence[object_id]
        purity = item.positive_weight / np.maximum(
            item.positive_weight + item.negative_weight,
            item.positive_weight + 1e-8,
        )
        visible_share = item.positive_weight / np.maximum(item.visible_weight, item.positive_weight + 1e-8)
        accepted = (
            (item.positive_timestamps >= int(policy["minimum_gaussian_timestamps"]))
            & (
                item.positive_timestamps.astype(np.int32)
                >= item.negative_timestamps.astype(np.int32) + int(policy["minimum_timestamp_margin"])
            )
            & (purity >= float(policy["minimum_global_purity"]))
        )
        indices = item.indices[accepted]
        support = item.positive_timestamps[accepted]
        scores = support.astype(np.float32) + float(policy["score_log_weight"]) * np.log1p(
            item.positive_weight[accepted]
        )
        support_fraction = support.astype(np.float32) / max(item.build_timestamp_count, 1)
        confidence = np.clip(
            0.55 * purity[accepted]
            + 0.30 * np.minimum(support_fraction, 1.0)
            + 0.15 * np.minimum(visible_share[accepted], 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
        claims.append({
            "object_id": object_id,
            "indices": indices,
            "scores": scores,
            "confidence": confidence,
            "support": support,
        })
        rows.append({
            "object_id": object_id,
            "category": item.obj.category,
            "candidate_gaussians": int(len(item.indices)),
            "supported_claims_before_conflicts": int(len(indices)),
            "build_timestamp_count": item.build_timestamp_count,
            "median_claim_purity": float(np.median(purity[accepted])) if len(indices) else 0.0,
            "median_claim_support": float(np.median(support)) if len(indices) else 0.0,
        })
    labels, confidence, support, conflict = resolve_sparse_claims(
        claims,
        gaussian_count,
        minimum_margin=float(policy["minimum_winner_margin"]),
        minimum_ratio=float(policy["minimum_winner_ratio"]),
    )
    claimed_indices = np.concatenate([
        np.asarray(row["indices"], dtype=np.int64)
        for row in claims if np.asarray(row["indices"]).size
    ]) if any(np.asarray(row["indices"]).size for row in claims) else np.zeros(0, dtype=np.int64)
    if len(claimed_indices):
        unique_claimed, claim_counts = np.unique(claimed_indices, return_counts=True)
        multiply_claimed = unique_claimed[claim_counts > 1]
        ambiguous_strong_indices = multiply_claimed[labels[multiply_claimed] == UNKNOWN_ID]
    else:
        ambiguous_strong_indices = np.zeros(0, dtype=np.int64)

    assigned = labels[labels >= 0]
    ids, counts = np.unique(assigned, return_counts=True) if len(assigned) else (
        np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    )
    count_by_id = {
        int(object_id): int(count)
        for object_id, count in zip(ids, counts, strict=True)
    }
    refinement_enabled = bool(config["refinement"]["enabled"])
    minimum_seed = (
        int(config["refinement"]["minimum_seed_gaussians"])
        if refinement_enabled else int(policy["minimum_object_gaussians"])
    )
    seed_invalid = {
        object_id
        for object_id in evidence
        if count_by_id.get(object_id, 0) < minimum_seed
        or count_by_id.get(object_id, 0)
        > int(float(policy["maximum_object_fraction"]) * gaussian_count)
    }
    if seed_invalid:
        remove = np.isin(labels, np.asarray(sorted(seed_invalid), dtype=np.int32))
        labels[remove] = UNKNOWN_ID
        confidence[remove] = 0.0
        support[remove] = 0
    seed_count_by_id = {
        object_id: (0 if object_id in seed_invalid else count)
        for object_id, count in count_by_id.items()
    }
    for row in rows:
        object_id = int(row["object_id"])
        row["resolved_core_gaussians"] = int(count_by_id.get(object_id, 0))
        row["seed_gaussians"] = int(seed_count_by_id.get(object_id, 0))
        row["seed_gate"] = "REJECT" if object_id in seed_invalid else (
            "PASS" if row["seed_gaussians"] else "NO_CLAIMS"
        )
        row["provisional_gaussians"] = row["seed_gaussians"]
        row["geometry_gate"] = "PENDING_REFINEMENT" if row["seed_gaussians"] else "REJECT"
    conflict["objects_seed_rejected"] = len(seed_invalid)
    return labels, confidence, support, rows, conflict, ambiguous_strong_indices


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def save_provisional(
    output: Path,
    labels: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    paths = {
        "object_id": output / "provisional_per_gaussian_object_id.npy",
        "confidence": output / "provisional_per_gaussian_confidence.npy",
        "timestamp_support": output / "provisional_per_gaussian_timestamp_support.npy",
    }
    atomic_npy(paths["object_id"], labels.astype(np.int32))
    atomic_npy(paths["confidence"], confidence.astype(np.float32))
    atomic_npy(paths["timestamp_support"], support.astype(np.uint16))
    return {name: _artifact(path) for name, path in paths.items()}


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def summarize_heldout_by_timestamp(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate views within a timestamp before cross-timestamp QC.

    A timestamp observed by two lenses and five virtual families receives the
    same final weight as a timestamp with one valid view. Per-view rows remain
    available for audit; only the acceptance summaries are de-biased here.
    """

    by_timestamp: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        by_timestamp[str(int(row["physical_timestamp_ns"]))].append(row)
    metric_names = (
        "iou", "precision", "recall", "largest_component_fraction",
        "area_ratio", "soft_iou",
    )
    timestamp_rows: list[dict[str, Any]] = []
    for timestamp in sorted(by_timestamp, key=int):
        views = by_timestamp[timestamp]
        row: dict[str, Any] = {
            "physical_timestamp_ns": int(timestamp),
            "view_count": len(views),
        }
        for name in metric_names:
            row[name] = float(np.median([float(view[name]) for view in views]))
        for name in ("depth_error_median", "depth_error_p90"):
            values = [float(view[name]) for view in views if view.get(name) is not None]
            row[name] = float(np.median(values)) if values else None
        row["depth_pixels"] = sum(int(view["depth_pixels"]) for view in views)
        row["maximum_iou"] = max(float(view["iou"]) for view in views)
        timestamp_rows.append(row)
    return {
        "timestamp_rows": timestamp_rows,
        "summaries": {
            name: _quantiles([float(row[name]) for row in timestamp_rows])
            for name in metric_names
        },
        "depth_error_median": _quantiles([
            float(row["depth_error_median"])
            for row in timestamp_rows if row["depth_error_median"] is not None
        ]),
        "depth_error_p90": _quantiles([
            float(row["depth_error_p90"])
            for row in timestamp_rows if row["depth_error_p90"] is not None
        ]),
        "depth_pixels": sum(int(row["depth_pixels"]) for row in timestamp_rows),
    }


def heldout_qc(
    gaussians: GaussianSet,
    run: RunData,
    split: Mapping[str, Any],
    provisional: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[int], dict[str, float], list[dict[str, Any]]]:
    import torch

    heldout_timestamps = set(split["heldout_timestamps"])
    grouped: dict[str, dict[int, dict[int, list[MaskObservation]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for obj in run.objects:
        for observation in obj.observations:
            frame = run.frame(observation.image_id)
            if frame.physical_timestamp in heldout_timestamps:
                grouped[frame.physical_timestamp][frame.image_id][obj.object_id].append(observation)

    bank = build_verified_csr(
        provisional,
        np.ones_like(provisional, dtype=np.float32),
        np.zeros_like(provisional, dtype=np.uint16),
    )
    indices_by_id = {
        int(object_id): bank["indices"][bank["indptr"][index]:bank["indptr"][index + 1]]
        for index, object_id in enumerate(bank["object_ids"])
    }
    view_results: dict[int, list[dict[str, Any]]] = defaultdict(list)
    maximum_objects = int(config["render"]["maximum_object_channels_per_call"])
    render_seconds = 0.0
    view_audit: list[dict[str, Any]] = []
    for rank, timestamp in enumerate(sorted(grouped, key=int), 1):
        for image_id, by_object in sorted(grouped[timestamp].items()):
            frame = run.frame(image_id)
            decoded, _, mask_audit, depth_audit = _load_view_masks(
                run, frame, by_object, config
            )
            view_audit.append({
                "image_id": image_id,
                "frame_id": frame.frame_id,
                "physical_timestamp_ns": int(timestamp),
                "sensor": frame.sensor,
                "family": frame.family,
                "depth": depth_audit,
                "masks": mask_audit,
            })
            available = [object_id for object_id in sorted(decoded) if object_id in indices_by_id]
            depth = np.asarray(np.load(frame.depth_path, mmap_mode="r"), dtype=np.float32)
            surface = _surface_interior(depth, run, config)
            for start in range(0, len(available), maximum_objects):
                batch_ids = available[start:start + maximum_objects]
                torch.cuda.synchronize()
                tick = time.perf_counter()
                mass, share, object_depth, _, alpha = reverse_render(
                    gaussians, run, frame, batch_ids, indices_by_id
                )
                torch.cuda.synchronize()
                render_seconds += time.perf_counter() - tick
                for column, object_id in enumerate(batch_ids):
                    target = decoded[object_id]["raw"]
                    soft = mass[..., column]
                    predicted = soft >= float(config["heldout"]["alpha_threshold"])
                    metrics = binary_mask_metrics(predicted, target, soft)
                    valid_depth = (
                        predicted & target & surface
                        & np.isfinite(object_depth[..., column])
                        & np.isfinite(depth) & (depth > 0)
                    )
                    residual = np.abs(object_depth[..., column][valid_depth] - depth[valid_depth])
                    metrics.update({
                        "image_id": image_id,
                        "physical_timestamp_ns": int(timestamp),
                        "frame_id": frame.frame_id,
                        "sensor": frame.sensor,
                        "family": frame.family,
                        "soft_mass_max": float(np.max(soft)),
                        "median_target_share": float(np.median(share[..., column][target])) if np.any(target) else 0.0,
                        "depth_pixels": int(len(residual)),
                        "depth_error_median": float(np.median(residual)) if len(residual) else None,
                        "depth_error_p90": float(np.quantile(residual, 0.90)) if len(residual) else None,
                        "global_alpha_median": float(np.median(alpha[..., 0][target])) if np.any(target) else 0.0,
                    })
                    view_results[object_id].append(metrics)
                del mass, share, object_depth, alpha
            torch.cuda.empty_cache()
        print(json.dumps({
            "stage": "heldout",
            "timestamp": rank,
            "total": len(grouped),
            "physical_timestamp_ns": int(timestamp),
        }, separators=(",", ":")), flush=True)

    split_rows = {int(row["object_id"]): row for row in split["objects"]}
    objects: list[dict[str, Any]] = []
    verified: set[int] = set()
    policy = config["heldout"]
    provisional_counts = dict(zip(
        *np.unique(provisional[provisional >= 0], return_counts=True)
    )) if np.any(provisional >= 0) else {}
    for obj in run.objects:
        results = view_results.get(obj.object_id, [])
        by_timestamp: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            by_timestamp[str(row["physical_timestamp_ns"])].append(row)
        good_timestamps = sum(
            max(float(row["iou"]) for row in rows) >= float(policy["good_timestamp_iou"])
            for rows in by_timestamp.values()
        )
        timestamp_summary = summarize_heldout_by_timestamp(results)
        summaries = timestamp_summary["summaries"]
        depth_median = timestamp_summary["depth_error_median"]
        depth_p90 = timestamp_summary["depth_error_p90"]
        depth_pixels = int(timestamp_summary["depth_pixels"])
        split_row = split_rows[obj.object_id]
        reasons: list[str] = []
        enough_claims = int(provisional_counts.get(obj.object_id, 0)) > 0
        if not enough_claims:
            reasons.append("no_provisional_gaussians")
        if split_row["strict_eligible"]:
            if len(by_timestamp) < int(config["split"]["strict_minimum_heldout_timestamps"]):
                reasons.append("insufficient_heldout_timestamps")
            if good_timestamps < int(policy["minimum_good_timestamps"]):
                reasons.append("insufficient_good_timestamps")
            if summaries["iou"] is None or summaries["iou"]["median"] < float(policy["minimum_median_iou"]):
                reasons.append("median_iou")
            if summaries["iou"] is None or summaries["iou"]["q25"] < float(policy["minimum_q25_iou"]):
                reasons.append("q25_iou")
            if summaries["precision"] is None or summaries["precision"]["median"] < float(policy["minimum_median_precision"]):
                reasons.append("median_precision")
            if summaries["recall"] is None or summaries["recall"]["median"] < float(policy["minimum_median_recall"]):
                reasons.append("median_recall")
            if (
                summaries["largest_component_fraction"] is None
                or summaries["largest_component_fraction"]["median"]
                < float(policy["minimum_median_largest_component"])
            ):
                reasons.append("fragmented_render")
            if depth_pixels < int(policy["minimum_depth_pixels"]):
                reasons.append("insufficient_depth_qc")
            elif (
                depth_median is None
                or depth_median["median"] > float(policy["maximum_median_depth_error_m"])
            ):
                reasons.append("median_depth_error")
            elif depth_p90 is None or depth_p90["median"] > float(policy["maximum_p90_depth_error_m"]):
                reasons.append("p90_depth_error")
            status = "verified" if not reasons else "rejected"
        else:
            status = "provisional" if enough_claims else "provisional"
            reasons.insert(0, f"eligibility:{split_row['eligibility']}")
        if status == "verified":
            verified.add(obj.object_id)
        objects.append({
            "object_id": obj.object_id,
            "category": obj.category,
            "status": status,
            "reasons": reasons,
            "provisional_gaussians": int(provisional_counts.get(obj.object_id, 0)),
            "heldout_timestamps": len(by_timestamp),
            "good_timestamps": int(good_timestamps),
            "summaries": summaries,
            "depth_error_median": depth_median,
            "depth_error_p90": depth_p90,
            "depth_pixels": int(depth_pixels),
            "timestamp_rows": timestamp_summary["timestamp_rows"],
            "views": results,
        })
    return objects, verified, {"reverse_render": render_seconds}, view_audit


def finalize_dense(
    provisional: np.ndarray,
    provisional_confidence: np.ndarray,
    provisional_support: np.ndarray,
    object_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    status_code = {"provisional": 1, "verified": 2, "rejected": 3}
    max_id = max([int(row["object_id"]) for row in object_rows] + [0])
    lookup = np.zeros(max_id + 1, dtype=np.uint8)
    for row in object_rows:
        object_id = int(row["object_id"])
        if object_id < 0:
            raise ValueError("negative FARM object ID")
        lookup[object_id] = status_code[str(row["status"])]
    status = np.zeros(len(provisional), dtype=np.uint8)
    assigned = np.flatnonzero(provisional >= 0)
    if len(assigned):
        if int(provisional[assigned].max()) >= len(lookup):
            raise ValueError("provisional label absent from object QC rows")
        status[assigned] = lookup[provisional[assigned]]
    verified_mask = status == 2
    labels = np.where(verified_mask, provisional, UNKNOWN_ID).astype(np.int32)
    confidence = np.where(verified_mask, provisional_confidence, 0).astype(np.float32)
    support = np.where(verified_mask, provisional_support, 0).astype(np.uint16)
    return labels, confidence, support, status


def save_final(
    output: Path,
    labels: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    status: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    paths = {
        "object_id": output / "per_gaussian_object_id.npy",
        "confidence": output / "per_gaussian_confidence.npy",
        "timestamp_support": output / "per_gaussian_timestamp_support.npy",
        "status": output / "per_gaussian_status.npy",
    }
    atomic_npy(paths["object_id"], labels.astype(np.int32))
    atomic_npy(paths["confidence"], confidence.astype(np.float32))
    atomic_npy(paths["timestamp_support"], support.astype(np.uint16))
    atomic_npy(paths["status"], status.astype(np.uint8))
    csr = build_verified_csr(labels, confidence, support)
    csr_path = output / "verified_instance_bank.npz"
    atomic_npz(csr_path, **csr)
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    artifacts["verified_instance_bank"] = _artifact(csr_path)
    return artifacts, csr


def release_scope_can_publish(
    run: RunData,
    external_fit_contract: Mapping[str, Any] | None,
    release_scope: Mapping[str, Any] | None,
    *,
    smoke: bool,
) -> bool:
    """Return the provenance-only release predicate, independent of quality."""

    return bool(
        not run.legacy
        and external_fit_contract is not None
        and release_scope is not None
        and not smoke
    )


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    started = time.perf_counter()
    smoke_ids = list(getattr(args, "smoke_object_id", []))
    smoke = bool(smoke_ids)
    object_id_path = getattr(args, "object_id_file", None)
    selection_path = getattr(args, "pre_lift_selection", None)
    provenance_paths = list(
        getattr(args, "pre_lift_provenance_artifact", [])
    )
    if (object_id_path is None) != (selection_path is None):
        raise ValueError(
            "--object-id-file and --pre-lift-selection must be supplied together"
        )
    if smoke and selection_path is not None:
        raise ValueError(
            "--smoke-object-id is incompatible with the release object allowlist"
        )

    if selection_path is None and provenance_paths:
        raise ValueError(
            "--pre-lift-provenance-artifact requires --pre-lift-selection"
        )

    config, config_sha = load_config(args.config)
    run = load_run(args.run, allow_legacy=bool(args.allow_legacy_run))
    release_scope: dict[str, Any] | None = None
    if selection_path is not None and object_id_path is not None:
        release_scope = validate_release_object_scope(
            run,
            selection_path,
            object_id_path,
            provenance_paths,
        )
    external_fit_contract: dict[str, Any] | None = None
    if args.full_colmap_fold_manifest is not None:
        from tools.farm_shaper_bridge.full_colmap_fit_contract import (
            validate_full_colmap_train_fit,
        )

        external_fit_contract = validate_full_colmap_train_fit(
            run, args.full_colmap_fold_manifest
        )
    # Freeze the global physical-timestamp partition against every final FARM
    # object before an optional scoped run is applied.
    split = make_split(run, config)
    if release_scope is not None:
        run, split = restrict_release_scope(run, split, release_scope)
    elif smoke:
        run, split = restrict_smoke_scope(run, split, smoke_ids)
    active_object_ids = sorted(obj.object_id for obj in run.objects)
    table = open_graphdeco_ply(args.ply)
    preflight_fingerprint = verify_run_ply_fingerprint(run, table.path)
    prep_summary = json.loads((run.run_dir / "rgbd" / "prep_summary.json").read_text(encoding="utf-8"))
    if int(prep_summary.get("gaussian_count", -1)) != table.count:
        raise ValueError("source PLY count differs from FARM RGBD preparation")
    plan = {
        "schema_version": "farm.gaussian-lift.plan.v1",
        "scene_id": run.scene_id,
        "legacy_run": run.legacy,
        "smoke_object_ids": sorted(set(smoke_ids)),
        "active_object_ids": active_object_ids,
        "object_scope": release_scope,
        "release_eligible": (
            release_scope_can_publish(
                run, external_fit_contract, release_scope, smoke=smoke
            )
            and config["release"].get("calibrated") is True
            and os.environ.get("FARM_BRIDGE_SOURCE_DIRTY") == "false"
            and bool(os.environ.get("FARM_BRIDGE_SOURCE_COMMIT"))
            and len(str(os.environ.get("FARM_BRIDGE_SOURCE_TREE_SHA256") or "")) == 64
            and str(os.environ.get("FARM_BRIDGE_IMAGE_ID", "")).startswith("sha256:")
        ),
        "source_gaussians": table.count,
        "source_ply_preflight": preflight_fingerprint,
        "objects": len(run.objects),
        "strict_eligible_objects": sum(bool(row["strict_eligible"]) for row in split["objects"]),
        "evidence_limited_objects": sum(not bool(row["strict_eligible"]) for row in split["objects"]),
        "physical_timestamps": split["physical_timestamp_count"],
        "build_timestamps": len(split["build_timestamps"]),
        "heldout_timestamps": len(split["heldout_timestamps"]),
        "config_sha256": config_sha,
        "frozen_full_colmap_fit_contract": external_fit_contract,
        "release_object_allowlist": release_scope,
        "split": split,
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return None

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    split["config_sha256"] = config_sha
    split["created_utc"] = utc_now()
    atomic_json(output / "split_manifest.json", split)
    source_stat = table.path.stat()
    source_sha = sha256_file(table.path)
    inventory = runtime_inventory()

    import torch
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    gaussians = load_gaussians(table, run.meters_per_scene_unit)
    torch.cuda.synchronize()
    gaussian_load_seconds = time.perf_counter() - load_started
    alignment = alignment_guard(gaussians, run, config)
    candidate_started = time.perf_counter()
    evidence, candidate_rows = build_candidates(run, gaussians, split, config)
    candidate_seconds = time.perf_counter() - candidate_started
    build_rows, build_timing = accumulate_build_evidence(
        gaussians, run, split, evidence, config
    )
    (
        provisional,
        provisional_confidence,
        provisional_support,
        object_build_rows,
        conflicts,
        blocked_strong_indices,
    ) = make_claims(evidence, table.count, config)
    refinement_started = time.perf_counter()
    (
        provisional,
        provisional_confidence,
        provisional_support,
        refinement_audit,
    ) = refine_connected_claims(
        gaussians.means_m,
        gaussians.radius_m,
        evidence,
        provisional,
        provisional_confidence,
        provisional_support,
        blocked_strong_indices,
        config,
    )
    refinement_by_id = {
        int(row["object_id"]): row for row in refinement_audit["objects"]
    }
    for row in object_build_rows:
        row["refinement"] = refinement_by_id.get(int(row["object_id"]), {
            "object_id": int(row["object_id"]),
            "seed_gaussians": int(row["seed_gaussians"]),
            "eligible_build_evidence": 0,
            "connected_proposals": 0,
            "added_gaussians": 0,
            "maximum_growth_step": 0,
            "median_connection_distance_m": None,
        })
    (
        provisional,
        provisional_confidence,
        provisional_support,
        geometry_gate_audit,
    ) = apply_provisional_geometry_gate(
        provisional,
        provisional_confidence,
        provisional_support,
        object_build_rows,
        table.count,
        config,
    )
    conflicts.update(geometry_gate_audit)
    refinement_seconds = time.perf_counter() - refinement_started
    assert_gaussian_owner_scope(
        provisional, active_object_ids, label="provisional labels"
    )
    provisional_artifacts = save_provisional(
        output, provisional, provisional_confidence, provisional_support
    )
    frozen_digest = canonical_sha256(provisional_artifacts)
    build_manifest = {
        "schema_version": "farm.gaussian-lift.build.v1",
        "status": "frozen_pending_heldout",
        "scene_id": run.scene_id,
        "active_object_ids": active_object_ids,
        "object_scope": release_scope,
        "created_utc": utc_now(),
        "config_sha256": config_sha,
        "source_ply_sha256": source_sha,
        "source_gaussian_count": table.count,
        "exact_renderer_contributor_vjp": True,
        "source_order_preserved": True,
        "forced_fill": False,
        "global_heldout_masks_opened": False,
        "alignment_guard": alignment,
        "candidates": candidate_rows,
        "objects": object_build_rows,
        "conflicts": conflicts,
        "refinement": refinement_audit,
        "provisional_artifacts": provisional_artifacts,
        "frozen_provisional_digest": frozen_digest,
        "views": build_rows,
        "timing_seconds": {
            "gaussian_load": gaussian_load_seconds,
            "candidate_index": candidate_seconds,
            "build_refinement": refinement_seconds,
            **build_timing,
        },
        "runtime": inventory,
    }
    atomic_json(output / "build_manifest.json", build_manifest)
    frozen_manifest_sha = sha256_file(output / "build_manifest.json")

    heldout_started = time.perf_counter()
    object_qc, verified_ids, heldout_timing, heldout_view_audit = heldout_qc(
        gaussians, run, split, provisional, config
    )
    heldout_manifest = {
        "schema_version": "farm.gaussian-lift.heldout-qc.v1",
        "status": "complete",
        "scene_id": run.scene_id,
        "active_object_ids": active_object_ids,
        "object_scope": release_scope,
        "created_utc": utc_now(),
        "config_sha256": config_sha,
        "frozen_build_manifest_sha256": frozen_manifest_sha,
        "frozen_provisional_digest": frozen_digest,
        "global_timestamp_disjoint": True,
        "failed_objects_return_to_unknown": True,
        "runner_up_promotion": False,
        "objects": object_qc,
        "views": heldout_view_audit,
        "summary": {
            "verified": sum(row["status"] == "verified" for row in object_qc),
            "provisional": sum(row["status"] == "provisional" for row in object_qc),
            "rejected": sum(row["status"] == "rejected" for row in object_qc),
            "verified_object_ids": sorted(verified_ids),
        },
        "timing_seconds": {
            **heldout_timing,
            "total": time.perf_counter() - heldout_started,
        },
    }
    atomic_json(output / "heldout_qc.json", heldout_manifest)

    labels, confidence, support, status = finalize_dense(
        provisional, provisional_confidence, provisional_support, object_qc
    )
    assert_gaussian_owner_scope(
        labels, active_object_ids, label="verified labels"
    )
    final_artifacts, csr = save_final(output, labels, confidence, support, status)
    assert_gaussian_owner_scope(
        np.asarray(csr["object_ids"], dtype=np.int64),
        active_object_ids,
        label="verified instance bank",
    )
    labeled_manifest: dict[str, Any] | None = None
    if not smoke:
        labeled_manifest = write_full_labeled_ply(
            table,
            output / "instance_labeled_full.ply",
            labels,
            confidence,
            support,
        )
        labeled_manifest["source_full_sha256"] = source_sha
        atomic_json(output / "labeled_ply_manifest.json", labeled_manifest)
        final_artifacts["instance_labeled_full"] = _artifact(output / "instance_labeled_full.ply")
        final_artifacts["labeled_ply_manifest"] = _artifact(output / "labeled_ply_manifest.json")

    source_stat_after = table.path.stat()
    if source_stat_after.st_size != source_stat.st_size or source_stat_after.st_mtime_ns != source_stat.st_mtime_ns:
        raise RuntimeError("source PLY stat changed during Gaussian lift")
    if sha256_file(table.path) != source_sha:
        raise RuntimeError("source PLY content changed during Gaussian lift")
    run_integrity_after = revalidate_run_integrity(run)
    clean_snapshot = (
        inventory.get("source_dirty") == "false"
        and bool(inventory.get("source_commit"))
        and len(str(inventory.get("source_tree_sha256") or "")) == 64
        and str(inventory.get("container_image_id") or "").startswith("sha256:")
    )
    strict_eligible_objects = sum(
        bool(row["strict_eligible"]) for row in split["objects"]
    )
    verified_objects = len(csr["object_ids"])
    verified_strict_fraction = (
        verified_objects / strict_eligible_objects if strict_eligible_objects else 0.0
    )
    release_policy = config["release"]
    quality_reasons: list[str] = []
    if release_scope is None:
        quality_reasons.append("pre_lift_release_selection_missing")
    if bool(inventory.get("runtime_interpreter_fallback")):
        quality_reasons.append("legacy_runtime_interpreter_fallback")
    if release_policy.get("calibrated") is not True:
        quality_reasons.append("configuration_not_calibrated")
    if verified_objects < int(release_policy["minimum_verified_objects"]):
        quality_reasons.append("verified_object_count_below_release_threshold")
    if verified_strict_fraction < float(
        release_policy["minimum_verified_strict_fraction"]
    ):
        quality_reasons.append("verified_strict_fraction_below_release_threshold")
    quality_pass = not quality_reasons
    release_eligible = (
        release_scope_can_publish(
            run, external_fit_contract, release_scope, smoke=smoke
        )
        and clean_snapshot
        and quality_pass
    )
    result = {
        "schema_version": "farm.gaussian-lift.result.v1",
        "status": "PASS",
        "quality_status": "PASS" if quality_pass else "WARN",
        "quality_gate": {
            "calibrated": release_policy.get("calibrated") is True,
            "minimum_verified_objects": int(
                release_policy["minimum_verified_objects"]
            ),
            "minimum_verified_strict_fraction": float(
                release_policy["minimum_verified_strict_fraction"]
            ),
            "strict_eligible_objects": strict_eligible_objects,
            "verified_objects": verified_objects,
            "verified_strict_fraction": verified_strict_fraction,
            "passed": quality_pass,
            "reasons": quality_reasons,
        },
        "release_eligible": release_eligible,
        "scene_id": run.scene_id,
        "active_object_ids": active_object_ids,
        "object_scope": release_scope,
        "created_utc": utc_now(),
        "frozen_full_colmap_fit_contract": external_fit_contract,
        "inputs": {
            "farm_run": str(run.run_dir),
            "farm_success_sha256": run.success_sha256,
            "farm_acceptance_sha256": run.acceptance_sha256,
            "farm_viewer_bundle_sha256": (
                run_integrity_after.get("viewer_bundle_sha256")
                if run_integrity_after is not None else None
            ),
            "source_ply": str(table.path),
            "source_ply_sha256": source_sha,
            "source_gaussian_count": table.count,
            "config": str(args.config.expanduser().resolve()),
            "config_sha256": config_sha,
            "object_id_file": (
                str(Path(object_id_path).expanduser().resolve())
                if object_id_path is not None else None
            ),
            "pre_lift_selection": (
                str(Path(selection_path).expanduser().resolve())
                if selection_path is not None else None
            ),
            "full_colmap_fold_manifest": (
                str(args.full_colmap_fold_manifest.expanduser().resolve())
                if args.full_colmap_fold_manifest is not None
                else None
            ),
        },
        "contracts": {
            "release_object_allowlist_validated": release_scope is not None,
            "excluded_objects_cannot_own_gaussians": True,
            "exact_pinned_gsplat_contributor_vjp": True,
            "native_frame_resolution": True,
            "global_physical_timestamp_holdout": True,
            "heldout_scope": (
                "mask_and_contributor_membership_only; conditioned_on_final_farm_geometry"
            ),
            "dense_arrays_in_source_ply_order": True,
            "verified_only_canonical_labels": True,
            "unknown_instance_id": UNKNOWN_ID,
            "runner_up_promotion": False,
            "source_ply_mutated": False,
            "gaussians_deleted": 0,
            "labeled_ply_full_graphdeco_schema": not smoke,
            "clean_versioned_source_snapshot": clean_snapshot,
            "runtime_interpreter_fallback": bool(
                inventory.get("runtime_interpreter_fallback")
            ),
        },
        "counts": {
            "source_gaussians": table.count,
            "provisional_gaussians": int(np.count_nonzero(provisional >= 0)),
            "verified_gaussians": int(np.count_nonzero(labels >= 0)),
            "unknown_gaussians": int(np.count_nonzero(labels < 0)),
            "verified_objects": len(csr["object_ids"]),
            "strict_eligible_objects": strict_eligible_objects,
            "verified_strict_fraction": verified_strict_fraction,
            "provisional_objects": int(heldout_manifest["summary"]["provisional"]),
            "rejected_objects": int(heldout_manifest["summary"]["rejected"]),
        },
        "qc_summary": heldout_manifest["summary"],
        "artifacts": {
            "split_manifest": _artifact(output / "split_manifest.json"),
            "build_manifest": _artifact(output / "build_manifest.json"),
            "heldout_qc": _artifact(output / "heldout_qc.json"),
            "provisional": provisional_artifacts,
            "final": final_artifacts,
        },
        "runtime": inventory,
        "resources": {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "timing_seconds": {"total": time.perf_counter() - started},
        "limitations": [
            "Exact means exact contribution weights of the pinned renderer, not semantic ground truth.",
            "Held-out masks do not enter contributor assignment, but candidate OBB geometry comes from the all-view FARM result.",
            "Only objects passing mask-held-out reverse-render QC enter canonical labels and the CSR bank.",
            "Unobserved or ambiguous source Gaussians remain unknown.",
        ] + ([
            "Legacy preflight omitted the prep interpreter; the audited canonical path was used and the result is non-release."
        ] if bool(inventory.get("runtime_interpreter_fallback")) else []),
    }
    atomic_json(output / "result.json", result)
    marker = (
        "_SUCCESS.json" if release_eligible
        else "_SMOKE_SUCCESS.json" if smoke
        else "_LEGACY_SUCCESS.json" if run.legacy
        else "_NONRELEASE_SUCCESS.json"
    )
    atomic_json(output / marker, {
        "schema_version": "farm.gaussian-lift.success.v1",
        "status": "success",
        "release_eligible": release_eligible,
        "scene_id": run.scene_id,
        "active_object_ids": active_object_ids,
        "object_scope_artifacts": (
            release_scope["artifacts"] if release_scope is not None else None
        ),
        "result": "result.json",
        "result_sha256": sha256_file(output / "result.json"),
        "source_ply_sha256": source_sha,
        "config_sha256": config_sha,
    })
    print(json.dumps({
        "status": result["status"],
        "quality_status": result["quality_status"],
        "release_eligible": release_eligible,
        "verified_objects": len(csr["object_ids"]),
        "verified_gaussians": int(np.count_nonzero(labels >= 0)),
        "seconds": result["timing_seconds"]["total"],
    }, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        execute(args)
        return 0
    except Exception as exc:
        if not args.plan_only and output.exists() and output.is_dir():
            try:
                atomic_json(output / "_FAILURE.json", {
                    "schema_version": "farm.gaussian-lift.failure.v1",
                    "status": "failure",
                    "created_utc": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
