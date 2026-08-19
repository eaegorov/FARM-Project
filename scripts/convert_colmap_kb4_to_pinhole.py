#!/usr/bin/env python3
"""Convert a KB4 fisheye COLMAP reconstruction into virtual PINHOLE views.

The converter is deliberately standalone: it understands COLMAP text/binary
models without depending on ``pycolmap`` (whose released camera registry does
not always include ``KANNALABRANDT4``).  It preserves camera centres and 3D
point coordinates, rebuilds every 2D observation/track in virtual cameras, and
writes a standard PINHOLE reconstruction readable by stock COLMAP.

The default five-view layout matches FARM's wide-FOV rig preprocessing:
``center, yaw_left, yaw_right, pitch_up, pitch_down``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


TOOL_VERSION = "1.0.1"
MANIFEST_SCHEMA = "farm.kb4-to-pinhole.v1"
SUPPORTED_FISHEYE_MODELS = {"KANNALABRANDT4", "OPENCV_FISHEYE"}

# COLMAP's public model ids plus the local KB4 extension used by the source
# reconstructions for which this tool was written.
CAMERA_MODEL_BY_ID: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
    12: ("KANNALABRANDT4", 8),
}
CAMERA_MODEL_TO_ID = {name: model_id for model_id, (name, _) in CAMERA_MODEL_BY_ID.items()}


@dataclasses.dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def kb4_params(self) -> Tuple[float, float, float, float, float, float, float, float]:
        if self.model not in SUPPORTED_FISHEYE_MODELS or self.params.size != 8:
            raise ValueError(
                f"camera {self.camera_id}: expected KB4/OPENCV_FISHEYE with 8 params, "
                f"got {self.model} ({self.params.size} params)"
            )
        return tuple(float(v) for v in self.params)  # type: ignore[return-value]

    def scaled_to(self, width: int, height: int) -> "Camera":
        sx = float(width) / float(self.width)
        sy = float(height) / float(self.height)
        params = np.asarray(self.params, dtype=np.float64).copy()
        params[0] *= sx
        params[2] *= sx
        params[1] *= sy
        params[3] *= sy
        return Camera(self.camera_id, self.model, int(width), int(height), params)


@dataclasses.dataclass
class RegisteredImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class PointTable:
    ids: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    error: np.ndarray


@dataclasses.dataclass(frozen=True)
class ColmapModel:
    cameras: Mapping[int, Camera]
    images: Mapping[int, RegisteredImage]
    points: PointTable
    source_format: str
    source_files: Tuple[Path, Path, Path]


@dataclasses.dataclass(frozen=True)
class ViewSpec:
    name: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0


@dataclasses.dataclass
class VirtualImage:
    image_id: int
    source_image_id: int
    camera_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    point_indices: np.ndarray
    xys: np.ndarray


@dataclasses.dataclass(frozen=True)
class TrackTable:
    kept_point_indices: np.ndarray
    offsets: np.ndarray
    image_ids: np.ndarray
    point2d_indices: np.ndarray


def _json_dump(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_source_path(root: Path, name: str) -> Path:
    rel = Path(name.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe COLMAP image name: {name!r}")
    resolved = (root / rel).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"COLMAP image escapes image root: {name!r}")
    return resolved


def _safe_token(value: str, what: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_")
    if not token:
        raise ValueError(f"{what} must contain at least one alphanumeric character")
    return token


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise EOFError(f"unexpected end of COLMAP binary file (wanted {size}, got {len(value)})")
    return value


def _unpack(handle: BinaryIO, fmt: str) -> Tuple[object, ...]:
    little_fmt = "<" + fmt
    return struct.unpack(little_fmt, _read_exact(handle, struct.calcsize(little_fmt)))


def _read_c_string(handle: BinaryIO) -> str:
    data = bytearray()
    while True:
        byte = _read_exact(handle, 1)
        if byte == b"\x00":
            break
        data.extend(byte)
    return data.decode("utf-8")


def _resolve_model_files(model_dir: Path, requested_format: str) -> Tuple[str, Tuple[Path, Path, Path]]:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"COLMAP model directory does not exist: {model_dir}")
    formats = {
        "auto": ("txt", "bin"), "text": ("txt",), "binary": ("bin",)
    }[requested_format]
    for ext in formats:
        files = tuple(model_dir / f"{stem}.{ext}" for stem in ("cameras", "images", "points3D"))
        if all(path.is_file() for path in files):
            return ("text" if ext == "txt" else "binary"), files  # type: ignore[return-value]
    raise FileNotFoundError(
        f"no complete COLMAP model in {model_dir}; expected cameras/images/points3D as "
        f"{requested_format if requested_format != 'auto' else 'txt or bin'}"
    )


def _read_cameras_text(path: Path) -> Dict[int, Camera]:
    result: Dict[int, Camera] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"{path}:{line_no}: malformed camera row")
            camera_id = int(parts[0])
            model = parts[1].upper()
            width, height = int(parts[2]), int(parts[3])
            params = np.asarray([float(v) for v in parts[4:]], dtype=np.float64)
            expected = next((n for _, (name, n) in CAMERA_MODEL_BY_ID.items() if name == model), None)
            if expected is not None and params.size != expected:
                raise ValueError(
                    f"{path}:{line_no}: {model} expects {expected} params, got {params.size}"
                )
            if camera_id in result:
                raise ValueError(f"{path}:{line_no}: duplicate camera id {camera_id}")
            result[camera_id] = Camera(camera_id, model, width, height, params)
    return result


def _read_cameras_binary(path: Path) -> Dict[int, Camera]:
    result: Dict[int, Camera] = {}
    with path.open("rb") as handle:
        (count,) = _unpack(handle, "Q")
        for _ in range(int(count)):
            camera_id, model_id, width, height = _unpack(handle, "iiQQ")
            if int(model_id) not in CAMERA_MODEL_BY_ID:
                raise ValueError(f"{path}: unsupported camera model id {model_id}")
            model, nparams = CAMERA_MODEL_BY_ID[int(model_id)]
            params = np.asarray(_unpack(handle, "d" * nparams), dtype=np.float64)
            cid = int(camera_id)
            result[cid] = Camera(cid, model, int(width), int(height), params)
        if handle.read(1):
            raise ValueError(f"{path}: trailing bytes after camera records")
    return result


def _read_images_text(path: Path) -> Dict[int, RegisteredImage]:
    result: Dict[int, RegisteredImage] = {}
    with path.open("r", encoding="utf-8") as handle:
        line_no = 0
        while True:
            raw = handle.readline()
            line_no += 1
            if not raw:
                break
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 10:
                raise ValueError(f"{path}:{line_no}: malformed image pose row")
            image_id = int(parts[0])
            qvec = np.asarray([float(v) for v in parts[1:5]], dtype=np.float64)
            tvec = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name = " ".join(parts[9:])
            obs_raw = handle.readline()
            line_no += 1
            if obs_raw == "":
                raise ValueError(f"{path}:{line_no}: missing POINTS2D row")
            obs_parts = obs_raw.strip().split()
            if len(obs_parts) % 3:
                raise ValueError(f"{path}:{line_no}: POINTS2D token count is not divisible by 3")
            if obs_parts:
                values = np.asarray(obs_parts, dtype=object).reshape(-1, 3)
                xys = values[:, :2].astype(np.float64)
                pids = values[:, 2].astype(np.int64)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                pids = np.empty((0,), dtype=np.int64)
            if image_id in result:
                raise ValueError(f"{path}:{line_no - 1}: duplicate image id {image_id}")
            result[image_id] = RegisteredImage(
                image_id, qvec, tvec, camera_id, name, xys, pids
            )
    return result


def _read_images_binary(path: Path) -> Dict[int, RegisteredImage]:
    result: Dict[int, RegisteredImage] = {}
    with path.open("rb") as handle:
        (count,) = _unpack(handle, "Q")
        for _ in range(int(count)):
            values = _unpack(handle, "idddddddi")
            image_id = int(values[0])
            qvec = np.asarray(values[1:5], dtype=np.float64)
            tvec = np.asarray(values[5:8], dtype=np.float64)
            camera_id = int(values[8])
            name = _read_c_string(handle)
            (nobs,) = _unpack(handle, "Q")
            if int(nobs):
                raw = np.frombuffer(_read_exact(handle, int(nobs) * 24), dtype=np.dtype([
                    ("x", "<f8"), ("y", "<f8"), ("pid", "<i8")
                ]))
                xys = np.column_stack((raw["x"], raw["y"])).astype(np.float64, copy=False)
                pids = raw["pid"].astype(np.int64, copy=False)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                pids = np.empty((0,), dtype=np.int64)
            result[image_id] = RegisteredImage(
                image_id, qvec, tvec, camera_id, name, xys, pids
            )
        if handle.read(1):
            raise ValueError(f"{path}: trailing bytes after image records")
    return result


def _point_table(ids: List[int], xyz: List[Tuple[float, float, float]],
                 rgb: List[Tuple[int, int, int]], errors: List[float]) -> PointTable:
    if not ids:
        return PointTable(
            np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.float64),
        )
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
    ids_arr = np.asarray(ids, dtype=np.int64)[order]
    if np.any(ids_arr[1:] == ids_arr[:-1]):
        raise ValueError("duplicate POINT3D_ID in COLMAP model")
    return PointTable(
        ids_arr,
        np.asarray(xyz, dtype=np.float64)[order],
        np.asarray(rgb, dtype=np.uint8)[order],
        np.asarray(errors, dtype=np.float64)[order],
    )


def _read_points_text(path: Path) -> PointTable:
    ids: List[int] = []
    xyz: List[Tuple[float, float, float]] = []
    rgb: List[Tuple[int, int, int]] = []
    errors: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8 or (len(parts) - 8) % 2:
                raise ValueError(f"{path}:{line_no}: malformed point row")
            ids.append(int(parts[0]))
            xyz.append((float(parts[1]), float(parts[2]), float(parts[3])))
            rgb.append((int(parts[4]), int(parts[5]), int(parts[6])))
            errors.append(float(parts[7]))
    return _point_table(ids, xyz, rgb, errors)


def _read_points_binary(path: Path) -> PointTable:
    ids: List[int] = []
    xyz: List[Tuple[float, float, float]] = []
    rgb: List[Tuple[int, int, int]] = []
    errors: List[float] = []
    with path.open("rb") as handle:
        (count,) = _unpack(handle, "Q")
        for _ in range(int(count)):
            values = _unpack(handle, "QdddBBBd")
            ids.append(int(values[0]))
            xyz.append((float(values[1]), float(values[2]), float(values[3])))
            rgb.append((int(values[4]), int(values[5]), int(values[6])))
            errors.append(float(values[7]))
            (track_len,) = _unpack(handle, "Q")
            _read_exact(handle, int(track_len) * 8)
        if handle.read(1):
            raise ValueError(f"{path}: trailing bytes after point records")
    return _point_table(ids, xyz, rgb, errors)


def read_colmap_model(model_dir: Path, requested_format: str = "auto") -> ColmapModel:
    source_format, files = _resolve_model_files(model_dir, requested_format)
    cameras_path, images_path, points_path = files
    if source_format == "text":
        cameras = _read_cameras_text(cameras_path)
        images = _read_images_text(images_path)
        points = _read_points_text(points_path)
    else:
        cameras = _read_cameras_binary(cameras_path)
        images = _read_images_binary(images_path)
        points = _read_points_binary(points_path)
    if not cameras or not images:
        raise ValueError(f"empty COLMAP model: {model_dir}")
    unknown_camera_ids = sorted({im.camera_id for im in images.values()} - set(cameras))
    if unknown_camera_ids:
        raise ValueError(f"registered images reference unknown cameras: {unknown_camera_ids}")
    return ColmapModel(cameras, images, points, source_format, files)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-15:
        raise ValueError("invalid zero/non-finite COLMAP quaternion")
    w, x, y, z = q / norm
    return np.asarray([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def rotmat_to_qvec(rotmat: np.ndarray) -> np.ndarray:
    r = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
    # Stable branch-based conversion, followed by a canonical positive scalar part.
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                        (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            q = np.asarray([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                            (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            q = np.asarray([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                            0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            q = np.asarray([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                            (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q.astype(np.float64)


def camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r = qvec_to_rotmat(qvec)
    return -r.T @ np.asarray(tvec, dtype=np.float64).reshape(3)


def virtual_rotation(view: ViewSpec) -> np.ndarray:
    """Return R_fisheye_from_virtual for x-right/y-down/z-forward cameras.

    Positive yaw turns the optical axis to image-right (+X). Positive pitch
    turns it upward (-Y). Roll is clockwise in the x/y image plane.
    """
    yaw = math.radians(float(view.yaw_deg))
    pitch = math.radians(float(view.pitch_deg))
    roll = math.radians(float(view.roll_deg))
    ry = np.asarray([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0],
                     [-math.sin(yaw), 0.0, math.cos(yaw)]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)],
                     [0.0, math.sin(pitch), math.cos(pitch)]], dtype=np.float64)
    rz = np.asarray([[math.cos(roll), -math.sin(roll), 0.0],
                     [math.sin(roll), math.cos(roll), 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64)
    return rz @ rx @ ry


def virtual_pose(qvec: np.ndarray, tvec: np.ndarray, fish_from_virtual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fish_from_world = qvec_to_rotmat(qvec)
    virtual_from_world = fish_from_virtual.T @ fish_from_world
    virtual_t = fish_from_virtual.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
    return rotmat_to_qvec(virtual_from_world), virtual_t


def pinhole_intrinsics(width: int, height: int, fov_x_deg: float, fov_y_deg: float) -> np.ndarray:
    if not (1e-3 < fov_x_deg < 179.0 and 1e-3 < fov_y_deg < 179.0):
        raise ValueError("virtual FOV must be in (0, 179) degrees")
    fx = float(width) / (2.0 * math.tan(math.radians(float(fov_x_deg)) / 2.0))
    fy = float(height) / (2.0 * math.tan(math.radians(float(fov_y_deg)) / 2.0))
    return np.asarray([[fx, 0.0, float(width) / 2.0],
                       [0.0, fy, float(height) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _solve_kb4_theta(radius: float, coeffs: Sequence[float], iterations: int = 12) -> float:
    theta = float(radius)
    k1, k2, k3, k4 = (float(v) for v in coeffs)
    for _ in range(iterations):
        t2 = theta * theta
        poly = 1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4
        deriv = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t2**2 + 7.0 * k3 * t2**3 + 9.0 * k4 * t2**4
        if abs(deriv) < 1e-12:
            break
        theta -= (theta * poly - radius) / deriv
    return float(theta)


def kb4_conservative_half_fov(camera: Camera) -> float:
    fx, fy, cx, cy, k1, k2, k3, k4 = camera.kb4_params()
    # COLMAP coordinates are corner-origin, so the extreme source pixel
    # centres are 0.5 and size-0.5 rather than OpenCV indices 0 and size-1.
    radii = [(cx - 0.5) / fx, (camera.width - 0.5 - cx) / fx,
             (cy - 0.5) / fy, (camera.height - 0.5 - cy) / fy]
    positive = [r for r in radii if r > 0.0]
    if not positive:
        raise ValueError(f"camera {camera.camera_id}: principal point is outside image")
    theta = _solve_kb4_theta(min(positive), (k1, k2, k3, k4))
    if not (0.0 < theta < math.pi):
        raise ValueError(f"camera {camera.camera_id}: invalid KB4 usable FOV")
    return theta


def build_kb4_remap(camera: Camera, intrinsics: np.ndarray, width: int, height: int,
                    view: ViewSpec) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    fx, fy, cx, cy, k1, k2, k3, k4 = camera.kb4_params()
    out_fx, out_fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    out_cx, out_cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    # COLMAP image coordinates are corner-origin: the top-left pixel center is
    # (0.5, 0.5), not (0, 0). Build rays through those pixel centers so the
    # remapped RGB agrees with the output PINHOLE intrinsics and POINTS2D.
    x = (np.arange(width, dtype=np.float64) + 0.5 - out_cx) / out_fx
    y = (np.arange(height, dtype=np.float64) + 0.5 - out_cy) / out_fy
    xx, yy = np.meshgrid(x, y)
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    rotation = virtual_rotation(view)
    fish = rays @ rotation.T
    rho = np.hypot(fish[..., 0], fish[..., 1])
    theta = np.arctan2(rho, fish[..., 2])
    theta2 = theta * theta
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
    scale = np.ones_like(theta_d)
    nonzero = rho > 1e-12
    scale[nonzero] = theta_d[nonzero] / rho[nonzero]
    source_colmap_x = fx * fish[..., 0] * scale + cx
    source_colmap_y = fy * fish[..., 1] * scale + cy
    # cv2.remap uses array-index coordinates where the top-left pixel centre
    # is (0, 0). Convert the projected COLMAP coordinates, whose top-left
    # centre is (0.5, 0.5), before sampling RGB.
    map_x = source_colmap_x - 0.5
    map_y = source_colmap_y - 0.5
    max_theta = kb4_conservative_half_fov(camera)
    valid = (
        np.isfinite(map_x) & np.isfinite(map_y) & (theta <= max_theta + 1e-10)
        & (map_x >= 0.0) & (map_y >= 0.0)
        & (map_x <= camera.width - 1.0) & (map_y <= camera.height - 1.0)
    )
    map_x = np.where(valid, map_x, -1.0).astype(np.float32)
    map_y = np.where(valid, map_y, -1.0).astype(np.float32)
    center_ray = rotation @ np.asarray([0.0, 0.0, 1.0])
    return map_x, map_y, {
        "valid_ratio": float(valid.mean()),
        "theta_max_deg": float(np.degrees(theta[valid].max())) if np.any(valid) else 0.0,
        "usable_half_fov_deg": float(math.degrees(max_theta)),
        "axis_x": float(center_ray[0]), "axis_y": float(center_ray[1]),
        "axis_z": float(center_ray[2]),
    }


def _default_views(side_angle_deg: float) -> List[ViewSpec]:
    angle = float(side_angle_deg)
    if not (0.0 < angle < 90.0):
        raise ValueError("--side-angle-deg must be in (0, 90)")
    return [
        ViewSpec("center", 0.0, 0.0, 0.0),
        ViewSpec("yaw_left", -angle, 0.0, 0.0),
        ViewSpec("yaw_right", angle, 0.0, 0.0),
        ViewSpec("pitch_up", 0.0, angle, 0.0),
        ViewSpec("pitch_down", 0.0, -angle, 0.0),
    ]


def _parse_view(value: str) -> ViewSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("view must be NAME,YAW_DEG,PITCH_DEG[,ROLL_DEG]")
    try:
        name = _safe_token(parts[0], "view name")
        values = [float(v) for v in parts[1:]]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(not np.isfinite(v) or abs(v) >= 180.0 for v in values):
        raise argparse.ArgumentTypeError("view angles must be finite and in (-180, 180)")
    return ViewSpec(name, values[0], values[1], values[2] if len(values) == 3 else 0.0)


def _parse_camera_labels(values: Sequence[str]) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--camera-label must be CAMERA_ID=LABEL")
        cid_s, label_s = value.split("=", 1)
        cid = int(cid_s)
        if cid in labels:
            raise ValueError(f"duplicate --camera-label for camera {cid}")
        labels[cid] = _safe_token(label_s, "camera label")
    return labels


def infer_camera_labels(images: Mapping[int, RegisteredImage], explicit: Mapping[int, str]) -> Dict[int, str]:
    used = sorted({image.camera_id for image in images.values()})
    unknown = sorted(set(explicit) - set(used))
    if unknown:
        raise ValueError(f"--camera-label references unused camera ids: {unknown}")
    labels = {camera_id: explicit.get(camera_id, f"cam{rank:02d}") for rank, camera_id in enumerate(used)}
    if len(set(labels.values())) != len(labels):
        raise ValueError("camera labels must be unique")
    return labels


def build_output_names(images: Mapping[int, RegisteredImage], views: Sequence[ViewSpec],
                       camera_labels: Mapping[int, str], extension: str) -> Tuple[Dict[Tuple[int, str], str], int]:
    ext = extension if extension.startswith(".") else "." + extension
    names: Dict[Tuple[int, str], str] = {}
    proposed: Dict[str, List[Tuple[int, str]]] = {}
    for source_id in sorted(images):
        image = images[source_id]
        stem = _safe_token(Path(image.name).stem, "source image stem")
        camera = camera_labels[image.camera_id]
        for view in views:
            proposed.setdefault(f"{camera}_{stem}_{view.name}{ext}", []).append((source_id, view.name))
    collisions = 0
    for name, keys in sorted(proposed.items()):
        if len(keys) == 1:
            names[keys[0]] = name
            continue
        collisions += len(keys)
        base, suffix = Path(name).stem, Path(name).suffix
        for source_id, view_name in sorted(keys):
            names[(source_id, view_name)] = f"{base}_i{source_id}{suffix}"
    if len(set(names.values())) != len(names):
        raise RuntimeError("output image name collision remained after deterministic disambiguation")
    return names, collisions


def _probe_source_images(images_root: Path, images: Mapping[int, RegisteredImage],
                         cameras: Mapping[int, Camera], scale_intrinsics: bool,
                         sparse_only: bool) -> Tuple[Dict[int, Camera], Dict[str, object]]:
    if sparse_only:
        return dict(cameras), {"checked": False, "reason": "sparse_only"}
    root = images_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source image directory does not exist: {root}")
    sizes_by_camera: Dict[int, set] = {cid: set() for cid in cameras}
    total_bytes = 0
    inventory = hashlib.sha256()
    for image_id in sorted(images):
        image = images[image_id]
        path = _safe_relative_source_path(root, image.name)
        if not path.is_file():
            raise FileNotFoundError(f"registered source image is missing: {path}")
        stat = path.stat()
        total_bytes += int(stat.st_size)
        inventory.update(image.name.replace("\\", "/").encode("utf-8"))
        inventory.update(b"\0" + str(int(stat.st_size)).encode("ascii") + b"\n")
        with PILImage.open(path) as source:
            sizes_by_camera.setdefault(image.camera_id, set()).add((int(source.width), int(source.height)))

    adjusted: Dict[int, Camera] = dict(cameras)
    adjustments: Dict[str, object] = {}
    for camera_id in sorted({image.camera_id for image in images.values()}):
        sizes = sorted(sizes_by_camera[camera_id])
        if len(sizes) != 1:
            raise ValueError(f"camera {camera_id} has inconsistent source image dimensions: {sizes}")
        actual_w, actual_h = sizes[0]
        camera = cameras[camera_id]
        if (actual_w, actual_h) != (camera.width, camera.height):
            if not scale_intrinsics:
                raise ValueError(
                    f"camera {camera_id} model is {camera.width}x{camera.height}, but images are "
                    f"{actual_w}x{actual_h}; pass --scale-intrinsics-to-images only if these are uniformly resized source pixels"
                )
            adjusted[camera_id] = camera.scaled_to(actual_w, actual_h)
            adjustments[str(camera_id)] = {
                "from": [camera.width, camera.height], "to": [actual_w, actual_h],
                "scale_xy": [actual_w / camera.width, actual_h / camera.height],
            }
    return adjusted, {
        "checked": True, "count": len(images), "total_bytes": total_bytes,
        "inventory_digest_names_sizes": inventory.hexdigest(),
        "dimensions_by_camera": {str(cid): [list(v) for v in sorted(values)] for cid, values in sizes_by_camera.items() if values},
        "intrinsics_adjustments": adjustments,
    }


def _rig_summary(images: Mapping[int, RegisteredImage], require_complete: bool) -> Dict[str, object]:
    used_cameras = sorted({image.camera_id for image in images.values()})
    groups: Dict[str, List[int]] = {}
    for image in images.values():
        groups.setdefault(Path(image.name).stem, []).append(image.camera_id)
    complete, duplicate_lens = 0, 0
    incomplete: List[str] = []
    expected = set(used_cameras)
    for stem, camera_ids in sorted(groups.items()):
        duplicate_lens += int(len(camera_ids) != len(set(camera_ids)))
        if set(camera_ids) == expected and len(camera_ids) == len(expected):
            complete += 1
        elif len(incomplete) < 20:
            incomplete.append(stem)
    if require_complete and (complete != len(groups) or duplicate_lens):
        raise ValueError(
            f"rig is incomplete: {complete}/{len(groups)} frame stems contain exactly one image from each camera; examples={incomplete[:5]}"
        )
    return {
        "camera_ids": used_cameras, "camera_count": len(used_cameras),
        "frame_stem_groups": len(groups), "complete_groups": complete,
        "duplicate_camera_groups": duplicate_lens, "incomplete_examples": incomplete,
    }


def _validate_input_model(model: ColmapModel) -> Dict[str, object]:
    used_camera_ids = sorted({image.camera_id for image in model.images.values()})
    unsupported = {cid: model.cameras[cid].model for cid in used_camera_ids if model.cameras[cid].model not in SUPPORTED_FISHEYE_MODELS}
    if unsupported:
        raise ValueError(f"source registered images use unsupported cameras: {unsupported}")
    if model.points.ids.size == 0:
        raise ValueError("source COLMAP reconstruction has no points3D")
    if not np.all(np.isfinite(model.points.xyz)) or not np.all(np.isfinite(model.points.error)):
        raise ValueError("source points3D contain non-finite values")
    point_ids = model.points.ids
    referenced, missing = 0, set()
    for image in model.images.values():
        if not np.all(np.isfinite(image.qvec)) or not np.all(np.isfinite(image.tvec)):
            raise ValueError(f"image {image.image_id} has non-finite pose")
        r = qvec_to_rotmat(image.qvec)
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-8) or np.linalg.det(r) < 0.999999:
            raise ValueError(f"image {image.image_id} has invalid rotation")
        valid = image.point3d_ids[image.point3d_ids >= 0]
        referenced += int(valid.size)
        if valid.size:
            positions = np.searchsorted(point_ids, valid)
            bounded = np.minimum(positions, point_ids.size - 1)
            ok = (positions < point_ids.size) & (point_ids[bounded] == valid)
            if not np.all(ok):
                missing.update(int(v) for v in valid[~ok][:100])
    if missing:
        raise ValueError(f"images reference missing points3D (first ids): {sorted(missing)[:20]}")
    return {
        "camera_count": len(model.cameras), "used_camera_ids": used_camera_ids,
        "registered_images": len(model.images), "points3D": int(model.points.ids.size),
        "registered_point_observations": referenced,
        "camera_models": sorted({model.cameras[cid].model for cid in used_camera_ids}),
    }


def build_virtual_geometry(model: ColmapModel, views: Sequence[ViewSpec], intrinsics: np.ndarray,
                           width: int, height: int, names: Mapping[Tuple[int, str], str],
                           min_source_observations: int) -> Tuple[List[VirtualImage], TrackTable, Dict[str, object]]:
    if min_source_observations < 1:
        raise ValueError("--min-source-observations must be >= 1")
    point_ids = model.points.ids
    pid_to_index = {int(pid): index for index, pid in enumerate(point_ids.tolist())}
    visibility_sources = np.zeros(point_ids.size, dtype=np.int32)
    output: List[VirtualImage] = []
    rotations = [(view, virtual_rotation(view)) for view in views]
    next_image_id = 1
    for source_id in tqdm(sorted(model.images), desc="virtual poses + sparse projections"):
        source = model.images[source_id]
        valid_pids = np.unique(source.point3d_ids[source.point3d_ids >= 0])
        if valid_pids.size:
            source_indices = np.asarray([pid_to_index[int(pid)] for pid in valid_pids], dtype=np.int64)
            world = model.points.xyz[source_indices]
            points_fish = world @ qvec_to_rotmat(source.qvec).T + source.tvec.reshape(1, 3)
        else:
            source_indices = np.empty((0,), dtype=np.int64)
            points_fish = np.empty((0, 3), dtype=np.float64)
        visible_any = np.zeros(source_indices.size, dtype=bool)
        for view, rotation in rotations:
            qvec, tvec = virtual_pose(source.qvec, source.tvec, rotation)
            points_virtual = points_fish @ rotation
            z = points_virtual[:, 2]
            in_front = z > 1e-9
            u = np.full(z.shape, np.nan, dtype=np.float64)
            v = np.full(z.shape, np.nan, dtype=np.float64)
            u[in_front] = intrinsics[0, 0] * points_virtual[in_front, 0] / z[in_front] + intrinsics[0, 2]
            v[in_front] = intrinsics[1, 1] * points_virtual[in_front, 1] / z[in_front] + intrinsics[1, 2]
            # COLMAP coordinates span the continuous image domain [0, width)
            # x [0, height). Do not discard the right/bottom half-pixel strip
            # by limiting projected observations to the last pixel center.
            inside = in_front & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
            visible_any |= inside
            output.append(VirtualImage(
                next_image_id, source_id, 1, names[(source_id, view.name)], qvec, tvec,
                source_indices[inside].copy(), np.column_stack((u[inside], v[inside])).astype(np.float64),
            ))
            next_image_id += 1
        visibility_sources[source_indices[visible_any]] += 1

    kept_point_indices = np.flatnonzero(visibility_sources >= min_source_observations).astype(np.int64)
    source_to_kept = np.full(point_ids.size, -1, dtype=np.int64)
    source_to_kept[kept_point_indices] = np.arange(kept_point_indices.size, dtype=np.int64)
    observation_counts = np.zeros(kept_point_indices.size, dtype=np.int64)
    nonempty = 0
    for image in output:
        keep = source_to_kept[image.point_indices] >= 0
        image.point_indices, image.xys = image.point_indices[keep], image.xys[keep]
        compact = source_to_kept[image.point_indices]
        if compact.size:
            nonempty += 1
            observation_counts[compact] += 1
    offsets = np.zeros(kept_point_indices.size + 1, dtype=np.int64)
    np.cumsum(observation_counts, out=offsets[1:])
    track_image_ids = np.empty(int(offsets[-1]), dtype=np.int32)
    track_point2d_indices = np.empty(int(offsets[-1]), dtype=np.int32)
    cursor = offsets[:-1].copy()
    for image in output:
        compact = source_to_kept[image.point_indices]
        if not compact.size:
            continue
        if np.unique(compact).size != compact.size:
            raise RuntimeError(f"duplicate point in virtual image {image.image_id}")
        destinations = cursor[compact]
        track_image_ids[destinations] = image.image_id
        track_point2d_indices[destinations] = np.arange(compact.size, dtype=np.int32)
        cursor[compact] += 1
    if not np.array_equal(cursor, offsets[1:]):
        raise RuntimeError("internal CSR track construction mismatch")
    tracks = TrackTable(kept_point_indices, offsets, track_image_ids, track_point2d_indices)
    return output, tracks, {
        "virtual_images": len(output), "nonempty_virtual_images": nonempty,
        "retained_points3D": int(kept_point_indices.size),
        "output_observations": int(track_image_ids.size),
        "minimum_distinct_source_images": min_source_observations,
        "source_visibility_histogram": {
            str(int(value)): int(count) for value, count in zip(*np.unique(visibility_sources, return_counts=True))
        },
    }


@contextlib.contextmanager
def _atomic_open(path: Path, mode: str) -> Iterator[object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    kwargs = {} if "b" in mode else {"encoding": "utf-8", "newline": "\n"}
    handle = tmp.open(mode, **kwargs)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(tmp, path)
    except BaseException:
        handle.close()
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _output_camera(intrinsics: np.ndarray, width: int, height: int) -> Camera:
    return Camera(1, "PINHOLE", width, height, np.asarray([
        intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    ], dtype=np.float64))


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _write_cameras_text(path: Path, camera: Camera) -> None:
    with _atomic_open(path, "w") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write("# Number of cameras: 1\n")
        params = " ".join(_format_float(v) for v in camera.params)
        handle.write(f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {params}\n")


def _write_images_text(path: Path, images: Sequence[VirtualImage], point_ids: np.ndarray) -> None:
    mean_obs = sum(image.point_indices.size for image in images) / max(1, len(images))
    with _atomic_open(path, "w") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(images)}, mean observations per image: {_format_float(mean_obs)}\n")
        for image in images:
            pose = " ".join(_format_float(v) for v in np.concatenate((image.qvec, image.tvec)))
            handle.write(f"{image.image_id} {pose} {image.camera_id} {image.name}\n")
            if image.point_indices.size:
                pids = point_ids[image.point_indices]
                handle.write(" ".join(
                    f"{_format_float(x)} {_format_float(y)} {int(pid)}"
                    for (x, y), pid in zip(image.xys, pids)
                ))
            handle.write("\n")


def _write_points_text(path: Path, source: PointTable, tracks: TrackTable) -> None:
    mean_track = tracks.image_ids.size / max(1, tracks.kept_point_indices.size)
    with _atomic_open(path, "w") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        handle.write(f"# Number of points: {tracks.kept_point_indices.size}, mean track length: {_format_float(mean_track)}\n")
        for compact, source_index in enumerate(tracks.kept_point_indices):
            pid = int(source.ids[source_index])
            xyz = " ".join(_format_float(v) for v in source.xyz[source_index])
            rgb = " ".join(str(int(v)) for v in source.rgb[source_index])
            begin, end = int(tracks.offsets[compact]), int(tracks.offsets[compact + 1])
            pairs = " ".join(
                f"{int(image_id)} {int(point2d_index)}"
                for image_id, point2d_index in zip(tracks.image_ids[begin:end], tracks.point2d_indices[begin:end])
            )
            handle.write(f"{pid} {xyz} {rgb} {_format_float(source.error[source_index])}{' ' + pairs if pairs else ''}\n")


def _write_cameras_binary(path: Path, camera: Camera) -> None:
    with _atomic_open(path, "wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", camera.camera_id, CAMERA_MODEL_TO_ID[camera.model], camera.width, camera.height))
        handle.write(struct.pack("<" + "d" * camera.params.size, *camera.params.tolist()))


def _write_images_binary(path: Path, images: Sequence[VirtualImage], point_ids: np.ndarray) -> None:
    with _atomic_open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images:
            values = [image.image_id, *image.qvec.tolist(), *image.tvec.tolist(), image.camera_id]
            handle.write(struct.pack("<idddddddi", *values))
            handle.write(image.name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", image.point_indices.size))
            for (x, y), pid in zip(image.xys, point_ids[image.point_indices]):
                handle.write(struct.pack("<ddq", float(x), float(y), int(pid)))


def _write_points_binary(path: Path, source: PointTable, tracks: TrackTable) -> None:
    with _atomic_open(path, "wb") as handle:
        handle.write(struct.pack("<Q", tracks.kept_point_indices.size))
        for compact, source_index in enumerate(tracks.kept_point_indices):
            pid, xyz, rgb = int(source.ids[source_index]), source.xyz[source_index], source.rgb[source_index]
            handle.write(struct.pack(
                "<QdddBBBd", pid, float(xyz[0]), float(xyz[1]), float(xyz[2]),
                int(rgb[0]), int(rgb[1]), int(rgb[2]), float(source.error[source_index]),
            ))
            begin, end = int(tracks.offsets[compact]), int(tracks.offsets[compact + 1])
            handle.write(struct.pack("<Q", end - begin))
            for image_id, point2d_index in zip(tracks.image_ids[begin:end], tracks.point2d_indices[begin:end]):
                handle.write(struct.pack("<ii", int(image_id), int(point2d_index)))


def write_colmap_models(sparse_dir: Path, output_format: str, camera: Camera,
                        images: Sequence[VirtualImage], source_points: PointTable,
                        tracks: TrackTable) -> Dict[str, object]:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if output_format in ("text", "both"):
        _write_cameras_text(sparse_dir / "cameras.txt", camera)
        _write_images_text(sparse_dir / "images.txt", images, source_points.ids)
        _write_points_text(sparse_dir / "points3D.txt", source_points, tracks)
    if output_format in ("binary", "both"):
        _write_cameras_binary(sparse_dir / "cameras.bin", camera)
        _write_images_binary(sparse_dir / "images.bin", images, source_points.ids)
        _write_points_binary(sparse_dir / "points3D.bin", source_points, tracks)
    return {
        "format": output_format,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(sparse_dir.iterdir()) if path.is_file()
        },
    }


def _image_black_stats(image: np.ndarray) -> Tuple[float, float]:
    black = np.all(image == 0, axis=2)
    border_width = max(1, min(image.shape[:2]) // 64)
    border = np.concatenate((
        black[:border_width].ravel(), black[-border_width:].ravel(),
        black[:, :border_width].ravel(), black[:, -border_width:].ravel(),
    ))
    return float(black.mean()), float(border.mean())


def _write_encoded_image(path: Path, image: np.ndarray, extension: str,
                         png_compression: int, jpeg_quality: int) -> None:
    suffix = ".png" if extension == "png" else ".jpg"
    params = ([cv2.IMWRITE_PNG_COMPRESSION, png_compression] if extension == "png"
              else [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{hash(path.name) & 0xfffffff:x}")
    with tmp.open("wb") as handle:
        handle.write(encoded.tobytes())
    os.replace(tmp, path)


def remap_source_images(images_root: Path, output_root: Path, work_root: Path,
                        source_images: Mapping[int, RegisteredImage], cameras: Mapping[int, Camera],
                        views: Sequence[ViewSpec], names: Mapping[Tuple[int, str], str],
                        remaps: Mapping[Tuple[int, str], Tuple[np.ndarray, np.ndarray]],
                        interpolation: str, extension: str, png_compression: int,
                        jpeg_quality: int, workers: int) -> Dict[str, object]:
    interpolation_code = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
                          "lanczos": cv2.INTER_LANCZOS4}[interpolation]
    stats_dir = work_root / "image_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    def convert_one(source_id: int) -> List[Dict[str, object]]:
        checkpoint = stats_dir / f"{source_id}.json"
        if checkpoint.is_file():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if all((output_root / row["name"]).is_file() for row in cached):
                return cached
        source = source_images[source_id]
        source_path = _safe_relative_source_path(images_root, source.name)
        rgb = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"OpenCV cannot read {source_path}")
        camera = cameras[source.camera_id]
        if (rgb.shape[1], rgb.shape[0]) != (camera.width, camera.height):
            raise ValueError(f"image dimensions changed after preflight: {source_path}")
        rows: List[Dict[str, object]] = []
        for view in views:
            map_x, map_y = remaps[(source.camera_id, view.name)]
            virtual = cv2.remap(rgb, map_x, map_y, interpolation_code,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            output_name = names[(source_id, view.name)]
            _write_encoded_image(output_root / output_name, virtual, extension,
                                 png_compression, jpeg_quality)
            black, border_black = _image_black_stats(virtual)
            rows.append({"name": output_name, "black_ratio": black,
                         "border_black_ratio": border_black})
        _json_dump(checkpoint, rows)
        return rows

    results: Dict[int, List[Dict[str, object]]] = {}
    source_ids = sorted(source_images)
    if workers == 1:
        for source_id in tqdm(source_ids, desc="remap fisheye RGB"):
            results[source_id] = convert_one(source_id)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {executor.submit(convert_one, source_id): source_id for source_id in source_ids}
            for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="remap fisheye RGB"):
                source_id = future_to_id[future]
                results[source_id] = future.result()
    rows = [row for source_id in source_ids for row in results[source_id]]
    black = np.asarray([row["black_ratio"] for row in rows], dtype=np.float64)
    border = np.asarray([row["border_black_ratio"] for row in rows], dtype=np.float64)
    worst = sorted(rows, key=lambda row: float(row["black_ratio"]), reverse=True)[:10]
    return {
        "images_written": len(rows), "black_ratio_mean": float(black.mean()),
        "black_ratio_max": float(black.max()), "border_black_ratio_mean": float(border.mean()),
        "border_black_ratio_max": float(border.max()), "worst_black_ratio": worst,
    }


def _validate_geometry(model: ColmapModel, virtual_images: Sequence[VirtualImage],
                       views: Sequence[ViewSpec], intrinsics: np.ndarray,
                       tracks: TrackTable, max_samples: int) -> Dict[str, object]:
    max_center_error, max_rotation_error, max_reprojection_error = 0.0, 0.0, 0.0
    sampled = 0
    for ordinal, image in enumerate(virtual_images):
        source = model.images[image.source_image_id]
        view = views[ordinal % len(views)]
        expected_rotation = virtual_rotation(view).T @ qvec_to_rotmat(source.qvec)
        max_rotation_error = max(max_rotation_error, float(np.max(np.abs(expected_rotation - qvec_to_rotmat(image.qvec)))))
        max_center_error = max(max_center_error, float(np.linalg.norm(
            camera_center(source.qvec, source.tvec) - camera_center(image.qvec, image.tvec)
        )))
        if sampled >= max_samples or not image.point_indices.size:
            continue
        take = min(image.point_indices.size, max(1, (max_samples - sampled) // max(1, len(virtual_images) - ordinal)))
        indices = np.linspace(0, image.point_indices.size - 1, take, dtype=np.int64)
        world = model.points.xyz[image.point_indices[indices]]
        camera = world @ qvec_to_rotmat(image.qvec).T + image.tvec.reshape(1, 3)
        projected = np.column_stack((
            intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2],
        ))
        max_reprojection_error = max(max_reprojection_error, float(np.max(np.linalg.norm(projected - image.xys[indices], axis=1))))
        sampled += take
    if max_center_error > 1e-8 or max_rotation_error > 1e-8 or max_reprojection_error > 1e-7:
        raise RuntimeError(
            f"geometry validation failed: center={max_center_error}, rotation={max_rotation_error}, reprojection={max_reprojection_error}"
        )
    if sum(image.point_indices.size for image in virtual_images) != tracks.image_ids.size:
        raise RuntimeError("image observation count does not match points3D track count")
    return {
        "camera_center_max_abs_m": max_center_error,
        "rotation_matrix_max_abs": max_rotation_error,
        "reprojection_max_px": max_reprojection_error,
        "reprojection_samples": sampled,
        "track_observation_count_consistent": True,
    }


def _validate_written_model(sparse_dir: Path, output_format: str,
                            expected_images: int, expected_points: int,
                            expected_observations: int) -> Dict[str, object]:
    fmt = "binary" if output_format in ("binary", "both") else "text"
    model = read_colmap_model(sparse_dir, fmt)
    observations = sum(int(image.point3d_ids.size) for image in model.images.values())
    if len(model.cameras) != 1 or next(iter(model.cameras.values())).model != "PINHOLE":
        raise RuntimeError("written model does not contain exactly one standard PINHOLE camera")
    if (len(model.images), model.points.ids.size, observations) != (
        expected_images, expected_points, expected_observations
    ):
        raise RuntimeError(
            "written COLMAP round-trip count mismatch: "
            f"images={len(model.images)}, points={model.points.ids.size}, observations={observations}"
        )
    pycolmap_status: Dict[str, object] = {"available": False}
    try:
        import pycolmap  # type: ignore
    except ImportError:
        pass
    else:
        pycolmap_status["available"] = True
        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
        pycolmap_status.update({
            "loaded": True, "cameras": len(reconstruction.cameras),
            "images": len(reconstruction.images), "points3D": len(reconstruction.points3D),
        })
        if len(reconstruction.images) != expected_images:
            raise RuntimeError("pycolmap round-trip image count mismatch")
    return {
        "round_trip_format": fmt, "cameras": len(model.cameras),
        "images": len(model.images), "points3D": int(model.points.ids.size),
        "observations": observations, "pycolmap": pycolmap_status,
    }


def _make_contact_sheet(output_images: Path, model: ColmapModel,
                        views: Sequence[ViewSpec], names: Mapping[Tuple[int, str], str],
                        destination: Path) -> Dict[str, object]:
    first_by_camera: Dict[int, int] = {}
    for source_id in sorted(model.images):
        first_by_camera.setdefault(model.images[source_id].camera_id, source_id)
    tile_w, tile_h = 320, 320
    canvas = np.full((tile_h * len(first_by_camera), tile_w * len(views), 3), 24, dtype=np.uint8)
    sampled_names: List[str] = []
    for row, (_, source_id) in enumerate(sorted(first_by_camera.items())):
        for col, view in enumerate(views):
            name = names[(source_id, view.name)]
            image = cv2.imread(str(output_images / name), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read converted image for contact sheet: {name}")
            resized = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(resized, (0, 0), (tile_w, 30), (12, 12, 12), -1)
            cv2.putText(resized, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)
            canvas[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = resized
            sampled_names.append(name)
    _write_encoded_image(destination, canvas, "jpg", 3, 94)
    return {"path": destination.name, "rows": len(first_by_camera), "columns": len(views), "sampled_images": sampled_names}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KANNALABRANDT4/OPENCV_FISHEYE COLMAP to standard virtual PINHOLE COLMAP."
    )
    parser.add_argument("--input-model", required=True, type=Path, help="Source sparse model directory.")
    parser.add_argument("--input-images", type=Path, help="Source fisheye image root (required unless --sparse-only).")
    parser.add_argument("--output", required=True, type=Path, help="New atomic output scene root.")
    parser.add_argument("--input-format", choices=("auto", "text", "binary"), default="auto")
    parser.add_argument("--output-model-format", choices=("both", "binary", "text"), default="both")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--fov-x-deg", type=float, default=90.0)
    parser.add_argument("--fov-y-deg", type=float, default=90.0)
    parser.add_argument("--side-angle-deg", type=float, default=50.0)
    parser.add_argument("--view", action="append", type=_parse_view, default=[],
                        help="Override defaults; repeat NAME,YAW,PITCH[,ROLL].")
    parser.add_argument("--camera-label", action="append", default=[], help="Repeat CAMERA_ID=LABEL.")
    parser.add_argument("--require-complete-rig", action="store_true")
    parser.add_argument("--scale-intrinsics-to-images", action="store_true")
    parser.add_argument("--min-source-observations", type=int, default=2)
    parser.add_argument("--min-valid-map-ratio", type=float, default=0.99)
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--interpolation", choices=("linear", "cubic", "lanczos"), default="linear")
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--max-reprojection-samples", type=int, default=10000)
    parser.add_argument("--sparse-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args(argv)


@contextlib.contextmanager
def _conversion_lock(output: Path) -> Iterator[None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.kb4-to-pinhole.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another conversion owns {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield


def _main_impl(args: argparse.Namespace) -> int:
    started = time.monotonic()
    if args.width < 16 or args.height < 16 or args.workers < 1:
        raise ValueError("width/height must be >=16 and workers >=1")
    if not 0.0 <= args.min_valid_map_ratio <= 1.0:
        raise ValueError("--min-valid-map-ratio must be in [0,1]")
    views = list(args.view) if args.view else _default_views(args.side_angle_deg)
    if len({view.name for view in views}) != len(views):
        raise ValueError("virtual view names must be unique")
    input_model = args.input_model.expanduser().resolve()
    input_images = (args.input_images.expanduser().resolve() if args.input_images else None)
    if not args.sparse_only and input_images is None:
        raise ValueError("--input-images is required unless --sparse-only")
    model = read_colmap_model(input_model, args.input_format)
    input_validation = _validate_input_model(model)
    adjusted_cameras, image_probe = _probe_source_images(
        input_images or Path("."), model.images, model.cameras,
        args.scale_intrinsics_to_images, args.sparse_only,
    )
    model = dataclasses.replace(model, cameras=adjusted_cameras)
    rig = _rig_summary(model.images, args.require_complete_rig)
    labels = infer_camera_labels(model.images, _parse_camera_labels(args.camera_label))
    names, naming_collisions = build_output_names(model.images, views, labels, args.image_format)
    intrinsics = pinhole_intrinsics(args.width, args.height, args.fov_x_deg, args.fov_y_deg)
    used_camera_ids = sorted({image.camera_id for image in model.images.values()})
    remaps: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
    map_validation: Dict[str, object] = {}
    for camera_id in used_camera_ids:
        rows: Dict[str, object] = {}
        for view in views:
            map_x, map_y, stats = build_kb4_remap(model.cameras[camera_id], intrinsics, args.width, args.height, view)
            if stats["valid_ratio"] < args.min_valid_map_ratio:
                raise ValueError(
                    f"camera {camera_id} view {view.name}: valid remap ratio {stats['valid_ratio']:.4f} "
                    f"is below {args.min_valid_map_ratio:.4f}; reduce FOV or view angle"
                )
            remaps[(camera_id, view.name)] = (map_x, map_y)
            rows[view.name] = stats
        map_validation[str(camera_id)] = rows
    source_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)} for path in model.source_files
    }
    config = {
        "tool_version": TOOL_VERSION, "source_model_files": source_files,
        "source_image_inventory": image_probe,
        "views": [dataclasses.asdict(view) for view in views],
        "width": args.width, "height": args.height,
        "fov_x_deg": args.fov_x_deg, "fov_y_deg": args.fov_y_deg,
        "camera_labels": {str(k): v for k, v in labels.items()},
        "min_source_observations": args.min_source_observations,
        "image_format": args.image_format, "interpolation": args.interpolation,
        "output_model_format": args.output_model_format, "sparse_only": args.sparse_only,
    }
    signature = _sha256_json(config)
    preflight = {
        "schema": MANIFEST_SCHEMA, "status": "preflight_pass", "signature": signature,
        "input_validation": input_validation, "image_probe": image_probe, "rig": rig,
        "camera_labels": {str(k): v for k, v in labels.items()}, "naming_collisions": naming_collisions,
        "output_image_count": len(names), "map_validation": map_validation, "config": config,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    output = args.output.expanduser().resolve()
    with _conversion_lock(output):
        if output.exists():
            manifest_path = output / "conversion_manifest.json"
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                if previous.get("status") == "complete" and previous.get("signature") == signature:
                    print(f"Already complete with matching signature: {output}")
                    return 0
            raise FileExistsError(f"output already exists and is not this completed conversion: {output}")
        stage = output.parent / f".{output.name}.kb4-to-pinhole-{signature[:12]}.partial"
        stage.mkdir(parents=True, exist_ok=True)
        work = stage / ".work"
        work.mkdir(exist_ok=True)
        work_manifest = work / "manifest.json"
        if work_manifest.is_file() and json.loads(work_manifest.read_text(encoding="utf-8")).get("signature") != signature:
            raise RuntimeError(f"staging signature mismatch: {stage}")
        _json_dump(work_manifest, {"signature": signature, "status": "in_progress"})

        virtual_images, tracks, geometry_stats = build_virtual_geometry(
            model, views, intrinsics, args.width, args.height, names, args.min_source_observations
        )
        geometry_validation = _validate_geometry(
            model, virtual_images, views, intrinsics, tracks, args.max_reprojection_samples
        )
        sparse_report = write_colmap_models(
            stage / "sparse" / "0", args.output_model_format, _output_camera(intrinsics, args.width, args.height),
            virtual_images, model.points, tracks,
        )
        written_validation = _validate_written_model(
            stage / "sparse" / "0", args.output_model_format, len(virtual_images),
            int(tracks.kept_point_indices.size), int(tracks.image_ids.size),
        )
        image_report: Dict[str, object] = {"skipped": True, "reason": "sparse_only"}
        contact_sheet: Dict[str, object] = {"skipped": True}
        if not args.sparse_only:
            assert input_images is not None
            image_report = remap_source_images(
                input_images, stage / "images", work, model.images, model.cameras, views, names,
                remaps, args.interpolation, args.image_format, args.png_compression,
                args.jpeg_quality, args.workers,
            )
            if not args.no_contact_sheet:
                contact_sheet = _make_contact_sheet(
                    stage / "images", model, views, names, stage / "conversion_qc_contact_sheet.jpg"
                )
        manifest = {
            **preflight, "status": "complete",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {"model": str(input_model), "images": str(input_images) if input_images else None,
                       "format": model.source_format},
            "output": str(output), "geometry": geometry_stats,
            "geometry_validation": geometry_validation, "sparse_model": sparse_report,
            "written_model_validation": written_validation, "image_conversion": image_report,
            "contact_sheet": contact_sheet, "runtime_seconds": time.monotonic() - started,
        }
        _json_dump(stage / "conversion_manifest.json", manifest)
        shutil.rmtree(work)
        os.replace(stage, output)
        directory_fd = os.open(str(output.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    print(f"Conversion complete: {output}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return _main_impl(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
