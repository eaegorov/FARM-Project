#!/usr/bin/env python3
"""Prepare a validated FARM RGB-D scene from PINHOLE COLMAP and a 3DGS PLY.

The selected image list is the contract: names are resolved exactly, kept in
file order, and never re-selected here.  Dense expected depth is rendered from
the aligned Gaussian scene while the RGB stream is copied from the original
COLMAP images.  A small, distributed alignment gate is evaluated before the
remaining views are rendered so a wrong COLMAP/PLY pair fails early.

The script is intentionally independent of factory naming and rig geometry.
Only PINHOLE and SIMPLE_PINHOLE cameras are accepted.  Convert fisheye models
first with ``convert_colmap_kb4_to_pinhole.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_VERSION = "1.2"
DEFAULT_IDENTITY_REGEX = (
    r"^(?P<sensor>.+?)_(?P<timestamp>\d+)_"
    r"(?P<family>center|yaw_left|yaw_right|pitch_up|pitch_down)\.(?:png|jpe?g)$"
)


@dataclass(frozen=True)
class View:
    image_id: int
    camera_id: int
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    sparse_xy: np.ndarray
    sparse_depth_z: np.ndarray


@dataclass(frozen=True)
class Identity:
    sensor: str
    timestamp: str
    family: str

    @property
    def camera(self) -> str:
        return self.sensor if self.family == "default" else f"{self.sensor}_{self.family}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fail-closed FARM RGB-D input from COLMAP PINHOLE + 3DGS.",
    )
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--selected-names", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--identity-regex",
        default=DEFAULT_IDENTITY_REGEX,
        help="Optional configured named groups. Non-matches use generic identities.",
    )
    parser.add_argument("--sensor-group", default="sensor")
    parser.add_argument("--timestamp-group", default="timestamp")
    parser.add_argument("--family-group", default="family")
    parser.add_argument(
        "--meters-per-scene-unit",
        type=float,
        default=1.0,
        help="Metric scale from preflight: metres represented by one COLMAP/3DGS unit.",
    )
    parser.add_argument("--nominal-hz", type=float, default=10.0)
    parser.add_argument("--resolution", type=int, default=896)
    parser.add_argument("--camera-alignment", choices=("off", "rig"), default="off")
    parser.add_argument("--sh-degree", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--depth-min-m", type=float, default=0.05)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    parser.add_argument("--radius-clip", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--expected-baseline-m", type=float, default=0.0,
        help="Optional metric-scale check. Disabled unless >0.",
    )
    parser.add_argument("--baseline-tolerance-m", type=float, default=0.005)
    parser.add_argument(
        "--baseline-camera-ids", nargs=2, type=int, metavar=("FIRST", "SECOND"),
        help="COLMAP camera IDs used by the optional baseline check.",
    )
    parser.add_argument(
        "--baseline-sensors", nargs=2, metavar=("FIRST", "SECOND"),
        help="Parsed physical sensor names used by the optional baseline check.",
    )
    parser.add_argument("--qa-sampled-views", type=int, default=12)
    parser.add_argument("--qa-min-views", type=int, default=3)
    parser.add_argument("--qa-min-valid-depth-ratio", type=float, default=0.50)
    parser.add_argument("--qa-min-sparse-samples", type=int, default=80)
    parser.add_argument("--qa-max-sparse-median-relative-error", type=float, default=0.20)
    parser.add_argument("--qa-min-luma-correlation", type=float, default=0.35)
    parser.add_argument("--qa-min-edge-correlation", type=float, default=0.15)
    parser.add_argument("--qa-max-affine-rgb-mae", type=float, default=0.20)
    parser.add_argument(
        "--fingerprint-content", action="store_true",
        help="Hash full input contents instead of size+mtime metadata (slower for large PLYs).",
    )
    parser.add_argument(
        "--rerender", action="store_true",
        help="Atomically regenerate all views of a matching run fingerprint.",
    )
    return parser.parse_args(argv)


def camera_model_name(camera: Any) -> str:
    name = getattr(camera, "model_name", None)
    if name:
        return str(name).upper()
    model = getattr(camera, "model", None)
    return str(getattr(model, "name", model)).split(".")[-1].upper()


def camera_intrinsics(camera: Any) -> tuple[float, float, float, float]:
    model = camera_model_name(camera)
    params = np.asarray(camera.params, dtype=np.float64)
    if model == "PINHOLE" and len(params) >= 4:
        return tuple(float(value) for value in params[:4])  # type: ignore[return-value]
    if model == "SIMPLE_PINHOLE" and len(params) >= 3:
        focal, cx, cy = (float(value) for value in params[:3])
        return focal, focal, cx, cy
    raise ValueError(
        f"Unsupported COLMAP camera model {model!r}; only PINHOLE and "
        "SIMPLE_PINHOLE are safe here. Convert fisheye images first."
    )


def read_selected_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"Selected-name file is empty: {path}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"selected_names contains duplicate exact names: {duplicates[:5]}")
    for name in names:
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe selected image name: {name!r}")
    return names


def validate_meters_per_scene_unit(value: float) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("--meters-per-scene-unit must be finite and strictly positive")
    return scale


def scene_depth_to_metres(depth: np.ndarray, meters_per_scene_unit: float) -> np.ndarray:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    return (np.asarray(depth, dtype=np.float32) * np.float32(scale)).astype(np.float32)


def pose_to_metres(
    camera_to_world: np.ndarray, meters_per_scene_unit: float
) -> np.ndarray:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera pose, got {pose.shape}")
    result = pose.copy()
    result[:3, 3] *= scale
    return result


def metre_clip_to_scene_units(
    depth_min_m: float, depth_max_m: float, meters_per_scene_unit: float
) -> tuple[float, float]:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    return float(depth_min_m / scale), float(depth_max_m / scale)


def parse_identity(
    name: str,
    camera_id: int,
    pattern: re.Pattern[str] | None,
    sensor_group: str = "sensor",
    timestamp_group: str = "timestamp",
    family_group: str = "family",
) -> Identity:
    match = pattern.match(name) if pattern is not None else None
    if match is not None:
        groups = match.groupdict()
        if timestamp_group not in groups:
            raise ValueError(
                f"Configured timestamp group {timestamp_group!r} is absent from --identity-regex"
            )
        sensor = (groups.get(sensor_group) or f"camera_{camera_id}").strip()
        timestamp = (groups.get(timestamp_group) or "").strip()
        family = (groups.get(family_group) or "default").strip()
        if not sensor or not timestamp or not family:
            raise ValueError(f"Empty identity field parsed from COLMAP image name {name!r}")
        return Identity(sensor=sensor, timestamp=timestamp, family=family)
    return Identity(sensor=f"camera_{camera_id}", timestamp=Path(name).stem, family="default")


def validate_identities(views: Sequence[View], identities: Sequence[Identity]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for view, identity in zip(views, identities, strict=True):
        key = (identity.timestamp, identity.camera)
        if key in seen:
            raise ValueError(
                "Two selected images resolve to the same (timestamp, camera) pair: "
                f"{seen[key]!r} and {view.name!r}. Supply a more specific --identity-regex."
            )
        seen[key] = view.name


def _image_pose_matrix(image: Any) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world.matrix(), dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"Unexpected COLMAP pose shape: {matrix.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = matrix
    return result


def _sparse_observations(image: Any, reconstruction: Any, world_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xys: list[np.ndarray] = []
    depths: list[float] = []
    for point2d in image.points2D:
        has_point = point2d.has_point3D() if callable(getattr(point2d, "has_point3D", None)) else False
        if not has_point:
            continue
        point_id = int(point2d.point3D_id)
        try:
            # pycolmap's MapPoint3DIdToPoint3D is an indexed pybind map, not a
            # Python dict and does not expose ``get`` in supported releases.
            point3d = reconstruction.points3D[point_id]
        except (KeyError, IndexError):
            continue
        xyz = np.asarray(point3d.xyz, dtype=np.float64)
        z = float(world_to_camera[2, :3] @ xyz + world_to_camera[2, 3])
        xy = np.asarray(point2d.xy, dtype=np.float64)
        if z > 0 and np.isfinite(z) and xy.shape == (2,) and np.isfinite(xy).all():
            xys.append(xy)
            depths.append(z)
    if not xys:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.stack(xys), np.asarray(depths, dtype=np.float64)


def load_selected_views(model_path: Path, selected_names: Sequence[str]) -> list[View]:
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - container integration path
        raise RuntimeError("pycolmap is required to read the COLMAP reconstruction") from exc
    reconstruction = pycolmap.Reconstruction(str(model_path))
    by_name: dict[str, Any] = {}
    for image in reconstruction.images.values():
        if image.name in by_name:
            raise ValueError(f"Duplicate image name in COLMAP model: {image.name!r}")
        by_name[image.name] = image
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise KeyError(f"selected_names has {len(missing)} images absent from COLMAP; first: {missing[:5]}")

    result: list[View] = []
    for name in selected_names:
        image = by_name[name]
        camera = reconstruction.cameras[image.camera_id]
        fx, fy, cx, cy = camera_intrinsics(camera)
        world_to_camera = _image_pose_matrix(image)
        sparse_xy, sparse_depth = _sparse_observations(image, reconstruction, world_to_camera)
        result.append(
            View(
                image_id=int(image.image_id), camera_id=int(image.camera_id), name=name,
                width=int(camera.width), height=int(camera.height), fx=fx, fy=fy, cx=cx, cy=cy,
                world_to_camera=world_to_camera, camera_to_world=np.linalg.inv(world_to_camera),
                sparse_xy=sparse_xy, sparse_depth_z=sparse_depth,
            )
        )
    return result


def output_geometry(view: View, resolution: int) -> tuple[int, int, np.ndarray]:
    if resolution < 16:
        raise ValueError("--resolution must be at least 16")
    scale = resolution / float(max(view.width, view.height))
    width = max(16, (int(round(view.width * scale)) // 8) * 8)
    height = max(16, (int(round(view.height * scale)) // 8) * 8)
    sx, sy = width / view.width, height / view.height
    K = np.asarray(
        [[view.fx * sx, 0.0, view.cx * sx], [0.0, view.fy * sy, view.cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return width, height, K


def distributed_indices(count: int, target: int) -> list[int]:
    if count <= 0 or target <= 0:
        return []
    target = min(count, target)
    return sorted(set(np.linspace(0, count - 1, target).round().astype(np.int64).tolist()))


def safe_source_path(root: Path, name: str) -> Path:
    root_resolved = root.resolve()
    path = (root_resolved / name).resolve()
    if not path.is_relative_to(root_resolved):
        raise ValueError(f"Image escapes --image-root: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_source(path: Path, view: View, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[1] != view.width or image.shape[0] != view.height:
        raise ValueError(
            f"Source size for {view.name!r} is {image.shape[1]}x{image.shape[0]}, "
            f"COLMAP expects {view.width}x{view.height}; refusing misaligned RGB-D."
        )
    interpolation = cv2.INTER_AREA if width <= view.width and height <= view.height else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def load_gaussians(path: Path, device: Any, sh_degree: int) -> dict[str, Any]:
    try:
        import torch
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - container integration path
        raise RuntimeError("torch and plyfile are required for 3DGS rendering") from exc
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or [])
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = required - names
    if missing:
        raise ValueError(f"Not a compatible 3DGS PLY; missing properties: {sorted(missing)}")
    means = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float32)
    scales = np.exp(np.column_stack([vertex[f"scale_{i}"] for i in range(3)])).astype(np.float32)
    quats = np.column_stack([vertex[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    opacities = sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    dc = np.column_stack([vertex[f"f_dc_{i}"] for i in range(3)]).astype(np.float32)
    if sh_degree > 0:
        basis_count = (sh_degree + 1) ** 2
        rest_names = [f"f_rest_{i}" for i in range(3 * (basis_count - 1))]
        absent = [name for name in rest_names if name not in names]
        if absent:
            raise ValueError(f"PLY lacks {len(absent)} SH coefficients required for degree {sh_degree}")
        rest = np.column_stack([vertex[name] for name in rest_names]).astype(np.float32)
        rest = rest.reshape(len(vertex), 3, basis_count - 1).transpose(0, 2, 1)
        colors = np.concatenate([dc[:, None, :], rest], axis=1)
    else:
        colors = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
    gaussian_count = len(vertex)
    del vertex
    return {
        "means": torch.from_numpy(means).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "count": gaussian_count,
    }


def render_gaussians(
    gaussians: Mapping[str, Any], view: View, K: np.ndarray, width: int, height: int,
    args: argparse.Namespace, device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
        from gsplat import rasterization
    except ImportError as exc:  # pragma: no cover - container integration path
        raise RuntimeError("torch and gsplat are required for 3DGS rendering") from exc
    view_matrix = torch.from_numpy(view.world_to_camera.astype(np.float32)).to(device).unsqueeze(0)
    intrinsic = torch.from_numpy(K).to(device).unsqueeze(0)
    near_scene, far_scene = metre_clip_to_scene_units(
        args.depth_min_m, args.depth_max_m, args.meters_per_scene_unit
    )
    with torch.inference_mode():
        render, alpha, _ = rasterization(
            means=gaussians["means"], quats=gaussians["quats"], scales=gaussians["scales"],
            opacities=gaussians["opacities"], colors=gaussians["colors"],
            viewmats=view_matrix, Ks=intrinsic, width=width, height=height,
            near_plane=near_scene, far_plane=far_scene,
            radius_clip=args.radius_clip, sh_degree=args.sh_degree if args.sh_degree > 0 else None,
            packed=True, render_mode="RGB+ED", rasterize_mode="antialiased",
        )
    rendered_rgb = render[0, ..., :3].float().cpu().numpy()
    raw_depth_scene = render[0, ..., 3].float().cpu().numpy().astype(np.float32)
    raw_depth = scene_depth_to_metres(
        raw_depth_scene, args.meters_per_scene_unit
    )
    alpha_np = alpha[0, ..., 0].float().cpu().numpy().astype(np.float32)
    valid = (
        np.isfinite(raw_depth) & (raw_depth >= args.depth_min_m) &
        (raw_depth <= args.depth_max_m) & (alpha_np >= args.alpha_min)
    )
    depth = np.where(valid, raw_depth, 0.0).astype(np.float32)
    return rendered_rgb, depth, alpha_np


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 32:
        return None
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else None


def appearance_alignment_metrics(
    source_bgr: np.ndarray, rendered_rgb: np.ndarray, valid_mask: np.ndarray,
    max_samples: int = 120_000,
) -> dict[str, float | int | None]:
    if source_bgr.shape[:2] != rendered_rgb.shape[:2] or valid_mask.shape != source_bgr.shape[:2]:
        raise ValueError("Appearance inputs have incompatible shapes")
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rendered = np.clip(rendered_rgb.astype(np.float32), 0.0, 1.0)
    mask = valid_mask.astype(bool) & np.isfinite(rendered).all(axis=2)
    count = int(mask.sum())
    if count < 256:
        return {"pixel_count": count, "luma_correlation": None, "edge_correlation": None, "affine_rgb_mae": None}

    flat_indices = np.flatnonzero(mask)
    if len(flat_indices) > max_samples:
        flat_indices = flat_indices[np.linspace(0, len(flat_indices) - 1, max_samples).astype(np.int64)]
    source_flat = source.reshape(-1, 3)[flat_indices]
    render_flat = rendered.reshape(-1, 3)[flat_indices]
    source_luma = source_flat @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    render_luma = render_flat @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)

    corrected = np.empty_like(render_flat)
    design = np.column_stack([render_flat, np.ones(len(render_flat), dtype=np.float32)])
    for channel in range(3):
        # Independent gain+bias is robust to exposure/white-balance while still
        # penalising spatial or geometric misalignment.
        channel_design = design[:, [channel, 3]]
        coefficients, *_ = np.linalg.lstsq(channel_design, source_flat[:, channel], rcond=None)
        corrected[:, channel] = np.clip(channel_design @ coefficients, 0.0, 1.0)
    affine_mae = float(np.mean(np.abs(corrected - source_flat)))

    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    render_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    source_edge = cv2.magnitude(cv2.Sobel(source_gray, cv2.CV_32F, 1, 0), cv2.Sobel(source_gray, cv2.CV_32F, 0, 1))
    render_edge = cv2.magnitude(cv2.Sobel(render_gray, cv2.CV_32F, 1, 0), cv2.Sobel(render_gray, cv2.CV_32F, 0, 1))
    return {
        "pixel_count": count,
        "luma_correlation": _pearson(source_luma, render_luma),
        "edge_correlation": _pearson(source_edge.reshape(-1)[flat_indices], render_edge.reshape(-1)[flat_indices]),
        "affine_rgb_mae": affine_mae,
    }


def _local_depth_median(depth: np.ndarray, x: int, y: int, radius: int = 1) -> float | None:
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    values = depth[y0:y1, x0:x1]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else None


def sparse_depth_alignment_metrics(
    depth: np.ndarray, sparse_xy: np.ndarray, sparse_z: np.ndarray,
    source_width: int, source_height: int, max_samples: int = 2_000,
) -> dict[str, float | int | None]:
    if sparse_xy.shape != (len(sparse_z), 2):
        raise ValueError("Sparse xys/depths have incompatible shapes")
    if not len(sparse_z):
        return {"candidate_count": 0, "sample_count": 0, "median_relative_error": None, "p90_relative_error": None}
    valid = (
        np.isfinite(sparse_xy).all(axis=1) & np.isfinite(sparse_z) & (sparse_z > 0) &
        (sparse_xy[:, 0] >= 0) & (sparse_xy[:, 0] < source_width) &
        (sparse_xy[:, 1] >= 0) & (sparse_xy[:, 1] < source_height)
    )
    indices = np.flatnonzero(valid)
    if len(indices) > max_samples:
        indices = indices[np.linspace(0, len(indices) - 1, max_samples).astype(np.int64)]
    errors: list[float] = []
    sx, sy = depth.shape[1] / source_width, depth.shape[0] / source_height
    for index in indices:
        x = int(np.clip(round(float(sparse_xy[index, 0]) * sx), 0, depth.shape[1] - 1))
        y = int(np.clip(round(float(sparse_xy[index, 1]) * sy), 0, depth.shape[0] - 1))
        rendered = _local_depth_median(depth, x, y)
        if rendered is not None:
            errors.append(abs(rendered - float(sparse_z[index])) / max(float(sparse_z[index]), 0.1))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "candidate_count": int(len(indices)), "sample_count": int(len(values)),
        "median_relative_error": float(np.median(values)) if len(values) else None,
        "p90_relative_error": float(np.percentile(values, 90.0)) if len(values) else None,
    }


def depth_colour(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return result
    low, high = np.percentile(depth[valid], [2.0, 98.0])
    scaled = np.clip((depth - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    result = cv2.applyColorMap((255 * (1.0 - scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def make_qa_panel(source: np.ndarray, rendered_rgb: np.ndarray, depth: np.ndarray, name: str) -> np.ndarray:
    render_bgr = cv2.cvtColor(np.clip(rendered_rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    difference = cv2.absdiff(source, render_bgr)
    tiles = [source, render_bgr, depth_colour(depth), difference]
    labels = ("source RGB", "3DGS RGB", "expected depth", "absolute RGB difference")
    for tile, label in zip(tiles, labels, strict=True):
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (8, 12, 18), -1)
        cv2.putText(tile, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    panel = np.concatenate(tiles, axis=1)
    cv2.putText(panel, name, (10, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 190, 40), 1, cv2.LINE_AA)
    return panel


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n").encode("utf-8"))


def atomic_write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"Could not encode JPEG: {path}")
    atomic_write_bytes(path, encoded.tobytes())


def atomic_write_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_write_bytes(path, buffer.getvalue())


def _safe_token(value: str, limit: int = 52) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "view"
    return cleaned[:limit]


def output_key(index: int, name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    return f"{index:06d}_{_safe_token(Path(name).stem)}_{digest}"


def _file_signature(path: Path, content: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def _model_files(model_path: Path) -> list[Path]:
    files: list[Path] = []
    for stem in ("cameras", "images", "points3D"):
        candidates = [model_path / f"{stem}.bin", model_path / f"{stem}.txt"]
        present = [path for path in candidates if path.is_file()]
        if not present:
            raise FileNotFoundError(f"Expected {stem}.bin and/or {stem}.txt in {model_path}")
        # Some converters intentionally emit both formats. Fingerprint every
        # present representation rather than rejecting a valid COLMAP model.
        files.extend(present)
    return files


def build_run_fingerprint(
    args: argparse.Namespace, selected_names: Sequence[str], source_paths: Sequence[Path],
) -> tuple[str, dict[str, Any]]:
    config_keys = (
        "scene_id", "identity_regex", "sensor_group", "timestamp_group", "family_group", "camera_alignment",
        "meters_per_scene_unit", "nominal_hz", "resolution", "sh_degree", "alpha_min",
        "depth_min_m", "depth_max_m", "radius_clip", "jpeg_quality", "expected_baseline_m",
        "baseline_tolerance_m", "baseline_camera_ids", "baseline_sensors",
        "qa_sampled_views", "qa_min_views",
        "qa_min_valid_depth_ratio", "qa_min_sparse_samples", "qa_max_sparse_median_relative_error",
        "qa_min_luma_correlation", "qa_min_edge_correlation", "qa_max_affine_rgb_mae",
    )
    selected_digest = hashlib.sha256("\n".join(selected_names).encode("utf-8")).hexdigest()
    source_meta = [_file_signature(path, args.fingerprint_content) for path in source_paths]
    source_digest = hashlib.sha256(json.dumps(source_meta, sort_keys=True).encode("utf-8")).hexdigest()
    payload = {
        "script": "prepare_colmap_3dgs_rgbd", "version": SCRIPT_VERSION,
        "config": {key: getattr(args, key, "off" if key == "camera_alignment" else None) for key in config_keys},
        "selected_names_sha256": selected_digest,
        "selected_count": len(selected_names),
        "inputs": {
            "colmap": [_file_signature(path, args.fingerprint_content) for path in _model_files(args.colmap_model)],
            "ply": _file_signature(args.ply, args.fingerprint_content),
            "source_images_sha256": source_digest,
        },
    }
    if getattr(args, "camera_alignment", "off") == "rig":
        runtime = Path(__file__).resolve().parents[1] / "src" / "farm_runtime"
        payload["registration_source_sha256"] = {
            name: hashlib.sha256((runtime / name).read_bytes()).hexdigest()
            for name in ("camera_alignment.py", "input_registration.py")
        }
    serializable = json.loads(json.dumps(payload, default=str))
    fingerprint = hashlib.sha256(json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return fingerprint, serializable


def compute_baselines(
    views: Sequence[View],
    identities: Sequence[Identity],
    camera_ids: tuple[int, int] | None = None,
    sensors: tuple[str, str] | None = None,
) -> np.ndarray:
    if (camera_ids is None) == (sensors is None):
        raise ValueError("Specify exactly one of camera_ids or sensors for baseline computation")
    if camera_ids is not None:
        first_key, second_key = int(camera_ids[0]), int(camera_ids[1])
        if first_key == second_key:
            raise ValueError("Baseline camera IDs must be distinct")
    else:
        assert sensors is not None
        first_key, second_key = str(sensors[0]), str(sensors[1])
        if first_key == second_key:
            raise ValueError("Baseline sensors must be distinct")
    grouped: dict[tuple[str, int | str], list[np.ndarray]] = {}
    for view, identity in zip(views, identities, strict=True):
        key: int | str = view.camera_id if camera_ids is not None else identity.sensor
        if key in (first_key, second_key):
            grouped.setdefault((identity.timestamp, key), []).append(
                view.camera_to_world[:3, 3]
            )
    values: list[float] = []
    for timestamp in sorted({key[0] for key in grouped}):
        first = grouped.get((timestamp, first_key))
        second = grouped.get((timestamp, second_key))
        if first and second:
            first_center = np.mean(np.stack(first), axis=0)
            second_center = np.mean(np.stack(second), axis=0)
            values.append(float(np.linalg.norm(first_center - second_center)))
    return np.asarray(values, dtype=np.float64)


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ("min", "p05", "median", "p95", "max")}
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(values.max()),
    }


def baseline_report(
    raw_values: np.ndarray,
    meters_per_scene_unit: float,
    expected_m: float,
    tolerance_m: float,
) -> dict[str, Any]:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    raw = np.asarray(raw_values, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Baseline samples must be a finite non-negative vector")
    metric = raw * scale
    errors = np.abs(metric - float(expected_m))
    passed = bool(
        len(metric)
        and np.isfinite(expected_m)
        and np.isfinite(tolerance_m)
        and tolerance_m >= 0
        and np.all(errors <= tolerance_m)
    )
    return {
        "sample_count": int(len(raw)),
        "meters_per_scene_unit": scale,
        "scene_units": _distribution(raw),
        "metres": _distribution(metric),
        # Preserve the old field while adding an explicit metric equivalent.
        "median_scene_units": float(np.median(raw)) if len(raw) else None,
        "median_m": float(np.median(metric)) if len(metric) else None,
        "expected_m": float(expected_m),
        "tolerance_m": float(tolerance_m),
        "max_abs_error_m": float(errors.max()) if len(errors) else None,
        "outlier_count": int(np.count_nonzero(errors > tolerance_m)) if len(errors) else 0,
        "criterion": "every paired timestamp within tolerance",
        "passed": passed,
    }


def evaluate_qa(stats: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def numbers(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for item in stats:
            current: Any = item
            for key in path:
                current = current.get(key) if isinstance(current, Mapping) else None
            if current is not None and np.isfinite(current):
                result.append(float(current))
        return result

    depth_ratios = numbers(("valid_depth_ratio",))
    luma = numbers(("appearance", "luma_correlation"))
    edges = numbers(("appearance", "edge_correlation"))
    mae = numbers(("appearance", "affine_rgb_mae"))
    sparse_errors = numbers(("sparse_depth", "median_relative_error"))
    sparse_count = int(sum(int(item.get("sparse_depth", {}).get("sample_count", 0)) for item in stats))
    aggregate = {
        "sampled_view_count": len(stats), "sparse_sample_count": sparse_count,
        "median_valid_depth_ratio": float(np.median(depth_ratios)) if depth_ratios else None,
        "median_sparse_relative_error": float(np.median(sparse_errors)) if sparse_errors else None,
        "median_luma_correlation": float(np.median(luma)) if luma else None,
        "median_edge_correlation": float(np.median(edges)) if edges else None,
        "median_affine_rgb_mae": float(np.median(mae)) if mae else None,
    }
    reasons: list[str] = []
    checks = [
        (len(stats) >= args.qa_min_views, f"sampled views {len(stats)} < {args.qa_min_views}"),
        (aggregate["median_valid_depth_ratio"] is not None and aggregate["median_valid_depth_ratio"] >= args.qa_min_valid_depth_ratio,
         f"median valid depth {aggregate['median_valid_depth_ratio']} < {args.qa_min_valid_depth_ratio}"),
        (sparse_count >= args.qa_min_sparse_samples, f"sparse depth samples {sparse_count} < {args.qa_min_sparse_samples}"),
        (aggregate["median_sparse_relative_error"] is not None and aggregate["median_sparse_relative_error"] <= args.qa_max_sparse_median_relative_error,
         f"sparse median relative error {aggregate['median_sparse_relative_error']} > {args.qa_max_sparse_median_relative_error}"),
        (aggregate["median_luma_correlation"] is not None and aggregate["median_luma_correlation"] >= args.qa_min_luma_correlation,
         f"luma correlation {aggregate['median_luma_correlation']} < {args.qa_min_luma_correlation}"),
        (aggregate["median_edge_correlation"] is not None and aggregate["median_edge_correlation"] >= args.qa_min_edge_correlation,
         f"edge correlation {aggregate['median_edge_correlation']} < {args.qa_min_edge_correlation}"),
        (aggregate["median_affine_rgb_mae"] is not None and aggregate["median_affine_rgb_mae"] <= args.qa_max_affine_rgb_mae,
         f"affine RGB MAE {aggregate['median_affine_rgb_mae']} > {args.qa_max_affine_rgb_mae}"),
    ]
    reasons.extend(message for passed, message in checks if not passed)
    return {
        "passed": not reasons, "failure_reasons": reasons, "aggregate": aggregate,
        "thresholds": {
            "min_views": args.qa_min_views, "min_valid_depth_ratio": args.qa_min_valid_depth_ratio,
            "min_sparse_samples": args.qa_min_sparse_samples,
            "max_sparse_median_relative_error": args.qa_max_sparse_median_relative_error,
            "min_luma_correlation": args.qa_min_luma_correlation,
            "min_edge_correlation": args.qa_min_edge_correlation,
            "max_affine_rgb_mae": args.qa_max_affine_rgb_mae,
        },
    }


def _valid_pair(rgb_path: Path, depth_path: Path, width: int, height: int) -> bool:
    if not rgb_path.is_file() or not depth_path.is_file():
        return False
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        return False
    try:
        depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        return depth.shape == (height, width) and depth.dtype == np.float32
    except Exception:
        return False


def _load_checkpoint(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if value.get("fingerprint") == fingerprint else None
    except (OSError, json.JSONDecodeError):
        return None


def _resize_for_contact(image: np.ndarray, width: int = 1800) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(image, (width, max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)


def write_contact_sheet(panel_paths: Sequence[Path], output: Path) -> None:
    panels = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in panel_paths]
    panels = [_resize_for_contact(panel) for panel in panels if panel is not None]
    if not panels:
        raise RuntimeError("No QA panels were available for the contact sheet")
    header = np.full((90, panels[0].shape[1], 3), (16, 21, 29), dtype=np.uint8)
    cv2.putText(header, "COLMAP + 3DGS ALIGNMENT QA", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(header, "distributed views: source | render | expected depth | RGB difference", (32, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 205, 245), 1, cv2.LINE_AA)
    atomic_write_jpeg(output, np.concatenate([header, *panels], axis=0), 93)


def _trajectory_pca(views: Sequence[View]) -> np.ndarray:
    centers = np.stack([view.camera_to_world[:3, 3] for view in views]).astype(np.float64)
    centered = centers - centers.mean(axis=0, keepdims=True)
    if len(centers) < 2:
        return np.zeros((len(centers), 2), dtype=np.float64)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh[: min(2, len(vh))].T
    projected = centered @ axes
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
    return projected


def format_metric(value: Any, digits: int = 3) -> str:
    """Format an optional QA metric without masking a missing-evidence failure."""
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "n/a"


def write_dashboard(
    views: Sequence[View], all_stats: Sequence[Mapping[str, Any]], qa_stats: Sequence[Mapping[str, Any]],
    qa_result: Mapping[str, Any], scene_id: str, output: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(24, 13.5), dpi=160)
    fig.patch.set_facecolor("#0c121c")
    projected = _trajectory_pca(views)
    axes[0, 0].plot(projected[:, 0], projected[:, 1], color="#60738e", linewidth=0.8, alpha=0.7)
    axes[0, 0].scatter(projected[:, 0], projected[:, 1], c=np.linspace(0, 1, len(views)), cmap="turbo", s=14)
    axes[0, 0].set_title("Selected trajectory (PCA ground view)")
    axes[0, 0].set_xlabel("trajectory PC1")
    axes[0, 0].set_ylabel("trajectory PC2")

    depth_ratio = [float(item["valid_depth_ratio"]) for item in all_stats]
    axes[0, 1].plot(depth_ratio, color="#35c5df", linewidth=1.1)
    axes[0, 1].axhline(qa_result["thresholds"]["min_valid_depth_ratio"], color="#f4b63e", linestyle="--")
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].set_title("Dense expected-depth coverage")
    axes[0, 1].set_xlabel("selected view order")
    axes[0, 1].set_ylabel("valid pixel ratio")

    sampled_indices = [int(item["selected_index"]) for item in qa_stats]
    sparse_error = [item["sparse_depth"].get("median_relative_error") for item in qa_stats]
    axes[1, 0].plot(sampled_indices, sparse_error, "o-", color="#ff6b78", label="sparse COLMAP z vs 3DGS ED")
    axes[1, 0].axhline(qa_result["thresholds"]["max_sparse_median_relative_error"], color="#f4b63e", linestyle="--", label="maximum")
    axes[1, 0].set_title("Geometry alignment on distributed views")
    axes[1, 0].set_xlabel("selected view order")
    axes[1, 0].set_ylabel("median relative depth error")
    axes[1, 0].legend(loc="best")

    luma = [item["appearance"].get("luma_correlation") for item in qa_stats]
    edges = [item["appearance"].get("edge_correlation") for item in qa_stats]
    mae = [item["appearance"].get("affine_rgb_mae") for item in qa_stats]
    axes[1, 1].plot(sampled_indices, luma, "o-", color="#4bd184", label="luma correlation")
    axes[1, 1].plot(sampled_indices, edges, "o-", color="#35c5df", label="edge correlation")
    axes[1, 1].plot(sampled_indices, mae, "o-", color="#f4b63e", label="affine RGB MAE")
    axes[1, 1].set_ylim(-0.1, 1.05)
    axes[1, 1].set_title("Source RGB vs 3DGS render")
    axes[1, 1].set_xlabel("selected view order")
    axes[1, 1].legend(loc="best")

    for axis in axes.flat:
        axis.set_facecolor("#101722")
        axis.grid(color="#77839a", alpha=0.18)
    status = "PASS" if qa_result["passed"] else "FAIL"
    aggregate = qa_result["aggregate"]
    fig.suptitle(
        f"{scene_id} | RGB-D PREPARATION ALIGNMENT QA | {status}\n"
        f"{len(views)} exact selected views · sparse samples {aggregate['sparse_sample_count']} · "
        f"depth error {format_metric(aggregate['median_sparse_relative_error'])} · "
        f"edge corr {format_metric(aggregate['median_edge_correlation'])}",
        fontsize=16, fontweight="bold", x=0.04, ha="left",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(temporary, facecolor=fig.get_facecolor())
        plt.close(fig)
        payload = cv2.imread(str(temporary), cv2.IMREAD_COLOR)
        if payload is None:
            raise RuntimeError("Could not render QA dashboard")
        atomic_write_jpeg(output, payload, 94)
    finally:
        if temporary.exists():
            temporary.unlink()


def _depth_statistics(depth: np.ndarray) -> dict[str, Any]:
    valid = depth > 0
    values = depth[valid]
    return {
        "valid_depth_ratio": float(valid.mean()),
        "depth_p02_m": float(np.percentile(values, 2)) if len(values) else None,
        "depth_p50_m": float(np.percentile(values, 50)) if len(values) else None,
        "depth_p98_m": float(np.percentile(values, 98)) if len(values) else None,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not args.scene_id.strip():
        raise ValueError("--scene-id cannot be empty")
    args.meters_per_scene_unit = validate_meters_per_scene_unit(
        args.meters_per_scene_unit
    )
    for option in ("sensor_group", "timestamp_group", "family_group"):
        if not str(getattr(args, option)).strip():
            raise ValueError(f"--{option.replace('_', '-')} cannot be empty")
    if args.nominal_hz <= 0 or args.depth_min_m <= 0 or args.depth_max_m <= args.depth_min_m:
        raise ValueError("Invalid temporal/depth range configuration")
    if not 0 <= args.alpha_min <= 1 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--alpha-min and --jpeg-quality are out of range")
    if args.qa_sampled_views < 1 or args.qa_min_views < 1:
        raise ValueError("Alignment QA requires at least one sampled/minimum view")
    for name in ("qa_min_valid_depth_ratio", "qa_max_sparse_median_relative_error", "qa_max_affine_rgb_mae"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.baseline_camera_ids and args.baseline_sensors:
        raise ValueError("Use only one of --baseline-camera-ids and --baseline-sensors")
    if args.expected_baseline_m > 0 and (
        not np.isfinite(args.expected_baseline_m)
        or not np.isfinite(args.baseline_tolerance_m)
        or args.baseline_tolerance_m < 0
    ):
        raise ValueError("Expected baseline and tolerance must be finite; tolerance must be >= 0")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    total_started = time.perf_counter()
    stage: dict[str, float] = {}
    for path in (args.colmap_model, args.image_root, args.ply, args.selected_names):
        if not path.exists():
            raise FileNotFoundError(path)

    preflight_started = time.perf_counter()
    selected_names = read_selected_names(args.selected_names)
    source_paths = [safe_source_path(args.image_root, name) for name in selected_names]
    views = load_selected_views(args.colmap_model, selected_names)
    pattern = re.compile(args.identity_regex, re.IGNORECASE) if args.identity_regex else None
    identities = [
        parse_identity(
            view.name,
            view.camera_id,
            pattern,
            args.sensor_group,
            args.timestamp_group,
            args.family_group,
        )
        for view in views
    ]
    validate_identities(views, identities)
    geometries = [output_geometry(view, args.resolution) for view in views]
    stage["preflight_seconds"] = time.perf_counter() - preflight_started

    baseline_info: dict[str, Any] = {
        "enabled": args.expected_baseline_m > 0,
        "meters_per_scene_unit": args.meters_per_scene_unit,
    }
    if args.expected_baseline_m > 0:
        if args.baseline_sensors:
            sensor_pair = (str(args.baseline_sensors[0]), str(args.baseline_sensors[1]))
            baseline_values = compute_baselines(
                views, identities, sensors=sensor_pair
            )
            baseline_info.update(
                {"selector": "sensors", "sensors": list(sensor_pair), "auto_selected": False}
            )
        elif args.baseline_camera_ids:
            camera_pair = (
                int(args.baseline_camera_ids[0]),
                int(args.baseline_camera_ids[1]),
            )
            baseline_values = compute_baselines(
                views, identities, camera_ids=camera_pair
            )
            baseline_info.update(
                {"selector": "camera_ids", "camera_ids": list(camera_pair), "auto_selected": False}
            )
        else:
            detected_sensors = sorted({identity.sensor for identity in identities})
            if len(detected_sensors) != 2:
                raise ValueError(
                    "Metric baseline check needs --baseline-sensors/--baseline-camera-ids "
                    "or exactly two parsed physical sensors"
                )
            sensor_pair = (detected_sensors[0], detected_sensors[1])
            baseline_values = compute_baselines(
                views, identities, sensors=sensor_pair
            )
            baseline_info.update(
                {"selector": "sensors", "sensors": list(sensor_pair), "auto_selected": True}
            )
        report = baseline_report(
            baseline_values,
            args.meters_per_scene_unit,
            args.expected_baseline_m,
            args.baseline_tolerance_m,
        )
        baseline_info.update(report)
        if not report["passed"]:
            raise RuntimeError(
                "Metric baseline mismatch or missing paired evidence: "
                f"median_m={report['median_m']}, max_abs_error_m={report['max_abs_error_m']}, "
                f"expected={args.expected_baseline_m:.6f}±{args.baseline_tolerance_m:.6f}"
            )

    fingerprint_started = time.perf_counter()
    fingerprint, fingerprint_payload = build_run_fingerprint(args, selected_names, source_paths)
    stage["fingerprint_seconds"] = time.perf_counter() - fingerprint_started
    output_dir = args.output_dir
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not manifest_path.is_file():
        raise RuntimeError(f"Non-empty output has no run_manifest.json; choose a clean directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] | None = None
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Output belongs to a different input/configuration fingerprint. "
                "Use a new output directory; existing results were not modified."
            )
        if previous.get("status") == "complete" and not args.rerender:
            required = (output_dir / "frames.json", output_dir / "prep_summary.json")
            if all(path.is_file() for path in required):
                print(json.dumps({"status": "already_complete", "output": str(output_dir), "fingerprint": fingerprint}, indent=2))
                return 0

    manifest = {
        "schema_version": "farm_rgbd_run_manifest_v1", "status": "running",
        "fingerprint": fingerprint, "scene_id": args.scene_id,
        "selected_view_count": len(views), "fingerprint_payload": fingerprint_payload,
        "started_unix": previous.get("started_unix", time.time()) if previous else time.time(),
        "updated_unix": time.time(), "resume_count": int(previous.get("resume_count", -1) + 1) if previous else 0,
    }
    atomic_write_json(manifest_path, manifest)

    rgb_dir, depth_dir = output_dir / "rgb", output_dir / "depth"
    qa_dir, state_dir = output_dir / "qa", output_dir / ".run_state"
    panels_dir, checkpoints_dir = state_dir / "qa_panels", state_dir / "views"
    for directory in (rgb_dir, depth_dir, qa_dir, panels_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for gsplat RGB-D preparation")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        gaussians = load_gaussians(args.ply, device, args.sh_degree)
        torch.cuda.synchronize(device)
        stage["gaussian_load_seconds"] = time.perf_counter() - load_started
        load_peak = int(torch.cuda.max_memory_allocated(device))
        peak_vram = load_peak

        registration = None
        if args.camera_alignment == "rig":
            if args.sh_degree < 1:
                raise ValueError("RGB registration requires view-dependent SH colours")
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from farm_runtime.input_registration import register_preparation_views
            registration_started = time.perf_counter()
            views, retained, registration = register_preparation_views(
                views, identities, geometries, source_paths, scale=args.meters_per_scene_unit,
                render=lambda view, K, width, height: render_gaussians(gaussians, view, K, width, height, args, device),
                read_source=read_source,
            )
            atomic_write_json(qa_dir / "camera_registration.json", registration)
            if len(retained) < max(3, int(np.ceil(.8 * len(identities)))):
                raise RuntimeError("Fewer than 80% of selected views have validated RGB/3DGS registration")
            identities = [identities[i] for i in retained]
            geometries = [geometries[i] for i in retained]
            source_paths = [source_paths[i] for i in retained]
            selected_names = [view.name for view in views]
            stage["camera_registration_seconds"] = time.perf_counter() - registration_started

        qa_indices = distributed_indices(len(views), args.qa_sampled_views)
        qa_set = set(qa_indices)
        render_order = qa_indices + [index for index in range(len(views)) if index not in qa_set]
        stats_by_index: dict[int, dict[str, Any]] = {}
        rendered_count = 0
        resumed_count = 0
        qa_render_seconds = 0.0
        remaining_render_seconds = 0.0

        for order_index, selected_index in enumerate(render_order):
            view = views[selected_index]
            width, height, K = geometries[selected_index]
            key = output_key(selected_index, view.name)
            rgb_path, depth_path = rgb_dir / f"{key}.jpg", depth_dir / f"{key}.npy"
            checkpoint_path = checkpoints_dir / f"{key}.json"
            panel_path = panels_dir / f"{key}.jpg"
            # A resumed depth must belong to this exact corrected pose and K.
            # Registration can be recomputed after an interrupted preparation.
            camera_sha256 = hashlib.sha256(
                np.asarray(view.camera_to_world, dtype="<f8").tobytes()
                + np.asarray(K, dtype="<f8").tobytes()
            ).hexdigest()
            checkpoint = None if args.rerender else _load_checkpoint(checkpoint_path, fingerprint)
            can_resume = (
                checkpoint is not None and checkpoint.get("camera_sha256") == camera_sha256 and
                _valid_pair(rgb_path, depth_path, width, height) and
                (selected_index not in qa_set or panel_path.is_file())
            )
            if can_resume:
                stat = dict(checkpoint["stats"])
                stat["resumed"] = True
                stats_by_index[selected_index] = stat
                resumed_count += 1
            else:
                view_started = time.perf_counter()
                source_started = time.perf_counter()
                source = read_source(source_paths[selected_index], view, width, height)
                source_seconds = time.perf_counter() - source_started
                torch.cuda.reset_peak_memory_stats(device)
                raster_started = time.perf_counter()
                rendered_rgb, depth, alpha = render_gaussians(gaussians, view, K, width, height, args, device)
                torch.cuda.synchronize(device)
                raster_seconds = time.perf_counter() - raster_started
                view_peak = int(torch.cuda.max_memory_allocated(device))
                peak_vram = max(peak_vram, view_peak)
                write_started = time.perf_counter()
                atomic_write_jpeg(rgb_path, source, args.jpeg_quality)
                atomic_write_npy(depth_path, depth)
                write_seconds = time.perf_counter() - write_started
                stat = {
                    "selected_index": selected_index, "name": view.name, "output_key": key,
                    "resumed": False, **_depth_statistics(depth),
                    "timing": {
                        "source_read_seconds": round(source_seconds, 6),
                        "raster_seconds": round(raster_seconds, 6),
                        "write_seconds": round(write_seconds, 6),
                        "total_seconds": round(time.perf_counter() - view_started, 6),
                    },
                    "peak_vram_bytes": view_peak,
                }
                if selected_index in qa_set:
                    stat["sparse_depth"] = sparse_depth_alignment_metrics(
                        depth,
                        view.sparse_xy,
                        scene_depth_to_metres(
                            view.sparse_depth_z, args.meters_per_scene_unit
                        ),
                        view.width,
                        view.height,
                    )
                    stat["appearance"] = appearance_alignment_metrics(source, rendered_rgb, depth > 0)
                    atomic_write_jpeg(panel_path, make_qa_panel(source, rendered_rgb, depth, view.name), 92)
                atomic_write_json(checkpoint_path, {"fingerprint": fingerprint, "camera_sha256": camera_sha256, "stats": stat})
                stats_by_index[selected_index] = stat
                rendered_count += 1
                duration = time.perf_counter() - view_started
                if selected_index in qa_set:
                    qa_render_seconds += duration
                else:
                    remaining_render_seconds += duration

            # Stop after the distributed alignment sample, before expensive full rendering.
            if order_index + 1 == len(qa_indices):
                qa_stats_now = [stats_by_index[index] for index in qa_indices]
                qa_result_now = evaluate_qa(qa_stats_now, args)
                atomic_write_json(qa_dir / "alignment_metrics.json", {"result": qa_result_now, "views": qa_stats_now})
                if not qa_result_now["passed"]:
                    write_contact_sheet([panels_dir / f"{output_key(i, views[i].name)}.jpg" for i in qa_indices], qa_dir / "02_alignment_contact_sheet.jpg")
                    write_dashboard(views, qa_stats_now, qa_stats_now, qa_result_now, args.scene_id, qa_dir / "01_alignment_dashboard.jpg")
                    raise RuntimeError("COLMAP/3DGS alignment QA failed: " + "; ".join(qa_result_now["failure_reasons"]))

            if (order_index + 1) % 10 == 0 or order_index + 1 == len(render_order):
                print(f"[rgbd-prep] {order_index + 1}/{len(render_order)} views", flush=True)

        stage["alignment_qa_render_seconds"] = qa_render_seconds
        stage["remaining_render_seconds"] = remaining_render_seconds
        report_started = time.perf_counter()
        all_stats = [stats_by_index[index] for index in range(len(views))]
        qa_stats = [stats_by_index[index] for index in qa_indices]
        qa_result = evaluate_qa(qa_stats, args)
        if not qa_result["passed"]:
            raise RuntimeError("Alignment QA became invalid after rendering")

        timestamp_rank: dict[str, int] = {}
        for identity in identities:
            if identity.timestamp not in timestamp_rank:
                timestamp_rank[identity.timestamp] = len(timestamp_rank)
        dt_ns = max(1, int(round(1_000_000_000 / args.nominal_hz)))
        frames: list[dict[str, Any]] = []
        for index, (view, identity) in enumerate(zip(views, identities, strict=True)):
            width, height, K = geometries[index]
            key = output_key(index, view.name)
            frames.append({
                "frame_id": identity.timestamp,
                "timestamp_ns": int((timestamp_rank[identity.timestamp] + 1) * dt_ns),
                "camera": identity.camera,
                "rgb_path": f"rgb/{key}.jpg", "depth_path": f"depth/{key}.npy",
                "depth_size": [height, width], "K": K.tolist(),
                "T_world_cam": pose_to_metres(
                    view.camera_to_world, args.meters_per_scene_unit
                ).tolist(),
                "source_image": view.name, "colmap_image_id": view.image_id,
                "colmap_camera_id": view.camera_id,
            })
        cameras = list(dict.fromkeys(frame["camera"] for frame in frames))
        frames_index = {
            "schema_version": "farm_frames_json_v1", "scene_id": args.scene_id,
            "cameras": cameras,
            "depth_units": "metres",
            "pose_translation_units": "metres",
            "meters_per_scene_unit": args.meters_per_scene_unit,
            "source_coordinate_units": "COLMAP/3DGS scene units",
            "depth_source": "3DGS gsplat expected-depth (RGB+ED), alpha-filtered",
            "camera_registration": registration,
            "identity_contract": {
                "regex": args.identity_regex,
                "sensor_group": args.sensor_group,
                "timestamp_group": args.timestamp_group,
                "family_group": args.family_group,
            },
            "selection_contract": {
                "selected_names": str(args.selected_names.resolve()),
                "exact_order_preserved": True, "count": len(selected_names),
                "requested_count": len(source_paths) if registration is None else registration["source_views"],
                "omitted_timestamps": [] if registration is None else registration["omitted_timestamps"],
            },
            "frames": frames,
        }
        atomic_write_json(output_dir / "frames.json", frames_index)
        atomic_write_json(qa_dir / "alignment_metrics.json", {"result": qa_result, "views": qa_stats})
        write_contact_sheet([panels_dir / f"{output_key(i, views[i].name)}.jpg" for i in qa_indices], qa_dir / "02_alignment_contact_sheet.jpg")
        write_dashboard(views, all_stats, qa_stats, qa_result, args.scene_id, qa_dir / "01_alignment_dashboard.jpg")
        stage["reports_seconds"] = time.perf_counter() - report_started
        stage["total_seconds"] = time.perf_counter() - total_started

        valid_ratios = np.asarray([item["valid_depth_ratio"] for item in all_stats], dtype=np.float64)
        summary = {
            "status": "complete", "scene_id": args.scene_id, "fingerprint": fingerprint,
            "inputs": {
                "colmap_model": str(args.colmap_model.resolve()),
                "image_root": str(args.image_root.resolve()), "ply": str(args.ply.resolve()),
                "selected_names": str(args.selected_names.resolve()),
            },
            "view_count": len(views), "camera_count": len(cameras),
            "rendered_count": rendered_count, "resumed_count": resumed_count,
            "gaussian_count": int(gaussians["count"]),
            "resolution_long_side": args.resolution,
            "metric_scale": {
                "meters_per_scene_unit": args.meters_per_scene_unit,
                "input_coordinates": "COLMAP/3DGS scene units",
                "saved_depth_units": "metres",
                "saved_pose_translation_units": "metres",
                "rasterization_coordinates": "source scene units",
            },
            "identity_grouping": {
                "regex": args.identity_regex,
                "sensor_group": args.sensor_group,
                "timestamp_group": args.timestamp_group,
                "family_group": args.family_group,
                "detected_sensors": sorted({identity.sensor for identity in identities}),
                "detected_families": sorted({identity.family for identity in identities}),
            },
            "depth_filter": {
                "alpha_min": args.alpha_min, "depth_min_m": args.depth_min_m,
                "depth_max_m": args.depth_max_m, "invalid_value": 0.0,
            },
            "metric_scale_check": baseline_info, "alignment_qa": qa_result,
            "valid_depth_ratio": {
                "min": float(valid_ratios.min()), "median": float(np.median(valid_ratios)),
                "max": float(valid_ratios.max()),
            },
            "timing": {key: round(value, 6) for key, value in stage.items()},
            "gpu": {"load_peak_vram_bytes": load_peak, "pipeline_peak_vram_bytes": peak_vram},
            "views": all_stats,
        }
        atomic_write_json(output_dir / "prep_summary.json", summary)
        manifest.update({
            "status": "complete", "updated_unix": time.time(),
            "completed_unix": time.time(), "summary": "prep_summary.json",
            "frames": "frames.json", "qa_passed": True,
        })
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({key: value for key, value in summary.items() if key != "views"}, indent=2))
        return 0
    except Exception as exc:
        manifest.update({
            "status": "failed", "updated_unix": time.time(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        atomic_write_json(manifest_path, manifest)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
