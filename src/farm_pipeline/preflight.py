"""Read-only validation for a FARM COLMAP + 3DGS scene.

The checks are deliberately CPU-only and bounded in memory.  Large binary PLY
files are memory-mapped and deterministically sampled; COLMAP records are
streamed.  This module never modifies input files or creates output folders.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from .scene_config import SceneConfig


REPORT_SCHEMA = "farm.preflight.v1"


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    status: str = "pass"
    duration_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, **details: Any) -> None:
        self.findings.append(Finding(severity, code, message, details))
        if severity == "error":
            self.status = "fail"
        elif severity == "warning" and self.status == "pass":
            self.status = "warn"


@dataclass
class CameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass
class ImageRecord:
    image_id: int
    camera_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    center: np.ndarray
    observations: int
    triangulated_observations: int


@dataclass
class ColmapData:
    source_format: str
    cameras: dict[int, CameraRecord]
    images: list[ImageRecord]
    points_count: int
    point_sample: np.ndarray
    points_finite: bool
    source_files: list[Path]


@dataclass
class PlyDataSummary:
    format: str
    vertex_count: int
    properties: tuple[str, ...]
    point_sample: np.ndarray
    sampled_records: int
    sample_is_full: bool
    finite_coordinates: bool
    finite_gaussian_fields: bool
    source_file: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fingerprint_file(
    path: Path,
    full_hash_max_bytes: int,
    chunk_bytes: int,
    *,
    force_full: bool = False,
) -> dict[str, Any]:
    """Content fingerprint a file, optionally requiring a full SHA-256.

    Generic large inputs retain the explicit sampled fingerprint used by the
    inexpensive discovery gate. The source Gaussian PLY is release-critical
    and callers force a full digest regardless of size.
    """

    stat = path.stat()
    size = int(stat.st_size)
    digest = hashlib.sha256()
    if force_full or size <= full_hash_max_bytes:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
        algorithm = "sha256"
        sampled_offsets: list[int] = []
    else:
        span = min(chunk_bytes, size)
        offsets = sorted({0, max(0, (size - span) // 2), max(0, size - span)})
        digest.update(b"farm-sampled-sha256-v1\0")
        digest.update(struct.pack("<Q", size))
        with path.open("rb") as stream:
            for offset in offsets:
                stream.seek(offset)
                payload = stream.read(span)
                digest.update(struct.pack("<QQ", offset, len(payload)))
                digest.update(payload)
        algorithm = "sha256-sampled-v1"
        sampled_offsets = offsets
    return {
        "path": str(path),
        "size_bytes": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "algorithm": algorithm,
        "digest": digest.hexdigest(),
        "sampled_offsets": sampled_offsets,
        "chunk_bytes": chunk_bytes,
    }


def _qvec_to_rotation(qvec: Sequence[float]) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1e-12:
        return np.full((3, 3), np.nan, dtype=np.float64)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _make_image_record(
    image_id: int,
    camera_id: int,
    name: str,
    qvec: Sequence[float],
    tvec: Sequence[float],
    observations: int,
    triangulated: int,
) -> ImageRecord:
    q_array = np.asarray(qvec, dtype=np.float64)
    t_array = np.asarray(tvec, dtype=np.float64)
    rotation = _qvec_to_rotation(q_array)
    center = -rotation.T @ t_array
    return ImageRecord(
        image_id=image_id,
        camera_id=camera_id,
        name=name,
        qvec=q_array,
        tvec=t_array,
        rotation=rotation,
        center=center,
        observations=observations,
        triangulated_observations=triangulated,
    )


_CAMERA_PARAM_COUNTS: dict[str, int] = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
    "KANNALABRANDT4": 8,
}

_CAMERA_ID_MODELS: dict[int, tuple[str, int]] = {
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
}


def _read_cameras_text(path: Path) -> dict[int, CameraRecord]:
    result: dict[int, CameraRecord] = {}
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"{path}:{line_number}: malformed camera record")
            camera_id = int(parts[0])
            model = parts[1].upper()
            params = tuple(float(item) for item in parts[4:])
            if camera_id in result:
                raise ValueError(f"{path}:{line_number}: duplicate camera id {camera_id}")
            result[camera_id] = CameraRecord(camera_id, model, int(parts[2]), int(parts[3]), params)
    return result


def _read_images_text(path: Path) -> list[ImageRecord]:
    result: list[ImageRecord] = []
    expecting_pose = True
    pose: Optional[tuple[int, np.ndarray, np.ndarray, int, str]] = None
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if expecting_pose:
                if not stripped:
                    continue
                parts = stripped.split(maxsplit=9)
                if len(parts) != 10:
                    raise ValueError(f"{path}:{line_number}: malformed image pose record")
                pose = (
                    int(parts[0]), int(parts[8]), parts[9],
                    np.asarray([float(item) for item in parts[1:5]], dtype=np.float64),
                    np.asarray([float(item) for item in parts[5:8]], dtype=np.float64),
                )
                expecting_pose = False
            else:
                assert pose is not None
                point_parts = stripped.split() if stripped else []
                if len(point_parts) % 3:
                    raise ValueError(f"{path}:{line_number}: POINTS2D must be xyz triples")
                observations = len(point_parts) // 3
                triangulated = sum(1 for index in range(2, len(point_parts), 3) if int(point_parts[index]) >= 0)
                result.append(
                    _make_image_record(*pose, observations=observations, triangulated=triangulated)
                )
                expecting_pose = True
                pose = None
    if not expecting_pose:
        # COLMAP permits the final empty POINTS2D line to be absent in hand-made
        # text models; treat it as an empty observation list.
        assert pose is not None
        result.append(_make_image_record(*pose, observations=0, triangulated=0))
    return result


def _reservoir_add(
    reservoir: list[tuple[float, float, float]],
    item: tuple[float, float, float],
    seen: int,
    capacity: int,
    rng: np.random.Generator,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    replacement = int(rng.integers(0, seen))
    if replacement < capacity:
        reservoir[replacement] = item


def _read_points_text(path: Path, sample_limit: int) -> tuple[int, np.ndarray, bool]:
    reservoir: list[tuple[float, float, float]] = []
    rng = np.random.default_rng(0xF4A2)
    count = 0
    finite = True
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=8)
            if len(parts) < 8:
                raise ValueError(f"{path}:{line_number}: malformed point3D record")
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            finite = finite and all(math.isfinite(value) for value in xyz)
            count += 1
            _reservoir_add(reservoir, xyz, count, sample_limit, rng)
    sample = np.asarray(reservoir, dtype=np.float64).reshape((-1, 3))
    return count, sample, finite


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("truncated COLMAP binary model")
    return payload


def _unpack(stream: BinaryIO, fmt: str) -> tuple[Any, ...]:
    layout = struct.Struct("<" + fmt)
    return layout.unpack(_read_exact(stream, layout.size))


def _read_cameras_binary(path: Path) -> dict[int, CameraRecord]:
    result: dict[int, CameraRecord] = {}
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(stream, "IiQQ")
            if model_id not in _CAMERA_ID_MODELS:
                raise ValueError(f"unsupported COLMAP binary camera model id {model_id}")
            model, param_count = _CAMERA_ID_MODELS[model_id]
            params = tuple(float(item) for item in _unpack(stream, "d" * param_count))
            result[int(camera_id)] = CameraRecord(int(camera_id), model, int(width), int(height), params)
        if stream.read(1):
            raise ValueError("trailing bytes in cameras.bin")
    return result


def _read_c_string(stream: BinaryIO) -> str:
    payload = bytearray()
    while True:
        char = _read_exact(stream, 1)
        if char == b"\0":
            break
        payload.extend(char)
        if len(payload) > 1_048_576:
            raise ValueError("unreasonably long image name in images.bin")
    return payload.decode("utf-8", errors="strict")


def _read_images_binary(path: Path) -> list[ImageRecord]:
    result: list[ImageRecord] = []
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(count):
            values = _unpack(stream, "I" + "d" * 7 + "I")
            image_id = int(values[0])
            qvec = values[1:5]
            tvec = values[5:8]
            camera_id = int(values[8])
            name = _read_c_string(stream)
            (observations,) = _unpack(stream, "Q")
            triangulated = 0
            for _point in range(observations):
                _x, _y, point_id = _unpack(stream, "ddq")
                triangulated += int(point_id >= 0)
            result.append(
                _make_image_record(
                    image_id, camera_id, name, qvec, tvec, int(observations), triangulated
                )
            )
        if stream.read(1):
            raise ValueError("trailing bytes in images.bin")
    return result


def _read_points_binary(path: Path, sample_limit: int) -> tuple[int, np.ndarray, bool]:
    reservoir: list[tuple[float, float, float]] = []
    rng = np.random.default_rng(0xF4A2)
    finite = True
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for index in range(int(count)):
            values = _unpack(stream, "QdddBBBd")
            xyz = (float(values[1]), float(values[2]), float(values[3]))
            finite = finite and all(math.isfinite(value) for value in xyz)
            _reservoir_add(reservoir, xyz, index + 1, sample_limit, rng)
            (track_length,) = _unpack(stream, "Q")
            _read_exact(stream, int(track_length) * 8)
        if stream.read(1):
            raise ValueError("trailing bytes in points3D.bin")
    return int(count), np.asarray(reservoir, dtype=np.float64).reshape((-1, 3)), finite


def read_colmap_model(model_dir: Path, sample_limit: int) -> ColmapData:
    """Read a standard COLMAP text or binary sparse model with bounded memory."""

    text_files = [model_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    binary_files = [model_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    have_text = all(path.is_file() for path in text_files)
    have_binary = all(path.is_file() for path in binary_files)
    if not have_text and not have_binary:
        raise FileNotFoundError(
            f"{model_dir} does not contain a complete cameras/images/points3D text or binary model"
        )
    if have_binary:
        try:
            cameras = _read_cameras_binary(binary_files[0])
            images = _read_images_binary(binary_files[1])
            count, sample, finite = _read_points_binary(binary_files[2], sample_limit)
            return ColmapData("binary", cameras, images, count, sample, finite, binary_files)
        except ValueError:
            if not have_text:
                raise
            # Non-standard camera models are commonly serialized by COLMAP
            # forks.  The accompanying text model remains unambiguous.
    cameras = _read_cameras_text(text_files[0])
    images = _read_images_text(text_files[1])
    count, sample, finite = _read_points_text(text_files[2], sample_limit)
    return ColmapData("text", cameras, images, count, sample, finite, text_files)


_PLY_TYPES: dict[str, str] = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class _PlyHeader:
    format: str
    data_offset: int
    vertex_count: int
    vertex_properties: list[tuple[str, str]]
    vertex_is_first: bool


def _read_ply_header(path: Path) -> _PlyHeader:
    file_format: Optional[str] = None
    vertex_count: Optional[int] = None
    properties: list[tuple[str, str]] = []
    current_element: Optional[str] = None
    element_order: list[str] = []
    with path.open("rb") as stream:
        first = stream.readline()
        if first.rstrip(b"\r\n") != b"ply":
            raise ValueError("not a PLY file (missing 'ply' magic)")
        for line_number in range(2, 10_002):
            raw = stream.readline()
            if not raw:
                raise ValueError("truncated PLY header")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("PLY header must be ASCII") from exc
            if line == "end_header":
                if file_format is None or vertex_count is None:
                    raise ValueError("PLY header lacks format or vertex element")
                return _PlyHeader(
                    file_format, stream.tell(), vertex_count, properties,
                    bool(element_order and element_order[0] == "vertex"),
                )
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                if len(parts) != 3 or parts[2] != "1.0":
                    raise ValueError(f"unsupported PLY format declaration: {line}")
                file_format = parts[1]
            elif parts[0] == "element":
                if len(parts) != 3:
                    raise ValueError(f"malformed PLY element at line {line_number}")
                current_element = parts[1]
                element_order.append(current_element)
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError("list-valued vertex properties are unsupported")
                if parts[1] not in _PLY_TYPES:
                    raise ValueError(f"unsupported PLY scalar type {parts[1]!r}")
                properties.append((parts[2], parts[1]))
        raise ValueError("PLY header exceeds 10,000 lines")


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    regular_count = limit // 2
    regular = np.linspace(0, count - 1, regular_count, dtype=np.int64)
    rng = np.random.default_rng(0x3D65)
    random = rng.choice(count, size=limit - regular_count, replace=False)
    return np.unique(np.concatenate((regular, random))).astype(np.int64)


def _read_ply_binary(path: Path, header: _PlyHeader, sample_limit: int) -> tuple[np.ndarray, int, bool, bool]:
    if not header.vertex_is_first:
        raise ValueError("binary PLY must store the vertex element first")
    if header.format not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported binary PLY format {header.format}")
    byte_order = "<" if header.format == "binary_little_endian" else ">"
    dtype = np.dtype([(name, byte_order + _PLY_TYPES[kind]) for name, kind in header.vertex_properties])
    expected_bytes = header.data_offset + header.vertex_count * dtype.itemsize
    if path.stat().st_size < expected_bytes:
        raise ValueError(
            f"truncated binary PLY: need at least {expected_bytes} bytes, got {path.stat().st_size}"
        )
    records = np.memmap(path, dtype=dtype, mode="r", offset=header.data_offset, shape=(header.vertex_count,))
    indices = _sample_indices(header.vertex_count, sample_limit)
    selected = records[indices]
    names = set(records.dtype.names or ())
    points = np.column_stack([selected[name].astype(np.float64) for name in ("x", "y", "z")])
    finite_coordinates = bool(np.isfinite(points).all())
    gaussian_names = [
        name for name in names
        if name == "opacity" or name.startswith("scale_") or name.startswith("rot_") or name.startswith("f_dc_")
    ]
    finite_gaussian = all(bool(np.isfinite(selected[name]).all()) for name in gaussian_names)
    return points, len(indices), finite_coordinates, finite_gaussian


def _read_ply_ascii(path: Path, header: _PlyHeader, sample_limit: int) -> tuple[np.ndarray, int, bool, bool]:
    if not header.vertex_is_first:
        raise ValueError("ASCII PLY must store the vertex element first")
    names = [name for name, _kind in header.vertex_properties]
    name_to_index = {name: index for index, name in enumerate(names)}
    xyz_indices = [name_to_index[name] for name in ("x", "y", "z")]
    gaussian_indices = [
        index for index, name in enumerate(names)
        if name == "opacity" or name.startswith("scale_") or name.startswith("rot_") or name.startswith("f_dc_")
    ]
    reservoir: list[tuple[float, float, float]] = []
    rng = np.random.default_rng(0x3D65)
    finite_coordinates = True
    finite_gaussian = True
    with path.open("rb") as stream:
        stream.seek(header.data_offset)
        for row in range(header.vertex_count):
            raw = stream.readline()
            if not raw:
                raise ValueError(f"truncated ASCII PLY at vertex {row}")
            parts = raw.split()
            if len(parts) < len(names):
                raise ValueError(f"ASCII PLY vertex {row} has too few properties")
            xyz = tuple(float(parts[index]) for index in xyz_indices)
            finite_coordinates = finite_coordinates and all(math.isfinite(value) for value in xyz)
            finite_gaussian = finite_gaussian and all(math.isfinite(float(parts[index])) for index in gaussian_indices)
            _reservoir_add(reservoir, xyz, row + 1, sample_limit, rng)
    return (
        np.asarray(reservoir, dtype=np.float64).reshape((-1, 3)), len(reservoir),
        finite_coordinates, finite_gaussian,
    )


def read_gaussian_ply(path: Path, sample_limit: int) -> PlyDataSummary:
    header = _read_ply_header(path)
    names = tuple(name for name, _kind in header.vertex_properties)
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex element lacks x/y/z")
    if header.vertex_count < 1:
        raise ValueError("PLY has no vertices")
    if header.format == "ascii":
        sample, sampled, finite_coordinates, finite_gaussian = _read_ply_ascii(path, header, sample_limit)
    else:
        sample, sampled, finite_coordinates, finite_gaussian = _read_ply_binary(path, header, sample_limit)
    return PlyDataSummary(
        format=header.format, vertex_count=header.vertex_count, properties=names,
        point_sample=sample, sampled_records=sampled, sample_is_full=sampled == header.vertex_count,
        finite_coordinates=finite_coordinates, finite_gaussian_fields=finite_gaussian, source_file=path,
    )


def _robust_bounds(points: np.ndarray) -> dict[str, Any]:
    if len(points) == 0:
        return {}
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    low = np.quantile(points, 0.01, axis=0) if len(points) >= 100 else minimum
    high = np.quantile(points, 0.99, axis=0) if len(points) >= 100 else maximum
    extent = high - low
    return {
        "min": minimum.tolist(), "max": maximum.tolist(),
        "robust_p01": low.tolist(), "robust_p99": high.tolist(),
        "robust_extent": extent.tolist(),
        "robust_diagonal": float(np.linalg.norm(extent)),
        "median": np.median(points, axis=0).tolist(),
    }


def _aabb_iou(low_a: np.ndarray, high_a: np.ndarray, low_b: np.ndarray, high_b: np.ndarray) -> float:
    intersection = np.maximum(np.minimum(high_a, high_b) - np.maximum(low_a, low_b), 0.0)
    intersection_volume = float(np.prod(intersection))
    volume_a = float(np.prod(np.maximum(high_a - low_a, 0.0)))
    volume_b = float(np.prod(np.maximum(high_b - low_b, 0.0)))
    union = volume_a + volume_b - intersection_volume
    return intersection_volume / union if union > 1e-18 else 0.0


def _nearest_distance_metrics(source: np.ndarray, target: np.ndarray, limit: int, diagonal: float) -> dict[str, float]:
    if limit <= 0 or not len(source) or not len(target):
        return {}
    source_idx = _sample_indices(len(source), min(limit, len(source)))
    target_idx = _sample_indices(len(target), min(max(limit * 4, limit), len(target)))
    source_sample = source[source_idx]
    target_sample = target[target_idx]
    try:
        from scipy.spatial import cKDTree

        distances, _indices = cKDTree(target_sample).query(source_sample, k=1, workers=1)
    except ImportError:
        source_sample = source_sample[: min(len(source_sample), 1024)]
        target_sample = target_sample[: min(len(target_sample), 16_384)]
        chunks = []
        for start in range(0, len(source_sample), 64):
            delta = source_sample[start : start + 64, None, :] - target_sample[None, :, :]
            chunks.append(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
        distances = np.concatenate(chunks)
    denominator = max(diagonal, 1e-12)
    return {
        "median": float(np.median(distances)),
        "p95": float(np.quantile(distances, 0.95)),
        "median_normalized": float(np.median(distances) / denominator),
        "p95_normalized": float(np.quantile(distances, 0.95) / denominator),
        "queries": int(len(distances)),
        "target_sample": int(len(target_sample)),
    }


def _check_paths(config: SceneConfig) -> CheckResult:
    result = CheckResult("paths")
    paths = {
        "colmap_model": config.inputs.colmap_model,
        "image_root": config.inputs.image_root,
        "gaussian_ply": config.inputs.gaussian_ply,
    }
    for name, path in paths.items():
        expected_directory = name != "gaussian_ply"
        exists = path.is_dir() if expected_directory else path.is_file()
        if not exists:
            kind = "directory" if expected_directory else "file"
            result.add("error", f"path.{name}.missing", f"{name} {kind} does not exist", path=str(path))
    result.metrics["resolved"] = {name: str(path) for name, path in paths.items()}
    result.metrics["output_root"] = str(config.output_root)
    return result


def _validate_camera(camera: CameraRecord, result: CheckResult) -> None:
    if camera.width <= 0 or camera.height <= 0:
        result.add("error", "colmap.camera.dimensions", "camera dimensions must be positive", camera_id=camera.camera_id)
    if not camera.params or not all(math.isfinite(value) for value in camera.params):
        result.add("error", "colmap.camera.params_nonfinite", "camera parameters are empty or non-finite", camera_id=camera.camera_id)
    expected = _CAMERA_PARAM_COUNTS.get(camera.model)
    if expected is not None and len(camera.params) != expected:
        result.add(
            "error", "colmap.camera.param_count", "camera parameter count does not match model",
            camera_id=camera.camera_id, model=camera.model, expected=expected, actual=len(camera.params),
        )
    if camera.params and camera.params[0] <= 0:
        result.add("error", "colmap.camera.focal", "camera focal length must be positive", camera_id=camera.camera_id)


def _check_colmap(config: SceneConfig, data: ColmapData) -> CheckResult:
    result = CheckResult("colmap")
    for camera in data.cameras.values():
        _validate_camera(camera, result)
    models = sorted({camera.model for camera in data.cameras.values()})
    unsupported = sorted(set(models) - set(config.camera.accepted_models))
    if unsupported:
        severity = "error" if config.camera.require_virtual_pinhole else "warning"
        result.add(
            severity, "colmap.camera.unsupported_model",
            "COLMAP contains camera models not accepted by the configured FARM RGB-D adapter",
            models=unsupported, accepted=list(config.camera.accepted_models),
            hint="Convert fisheye inputs to virtual PINHOLE views or extend camera.accepted_models only if the downstream adapter supports them.",
        )
    if len(data.images) < config.preflight.min_registered_images:
        result.add(
            "error", "colmap.images.too_few", "too few registered images",
            actual=len(data.images), minimum=config.preflight.min_registered_images,
        )
    image_ids: set[int] = set()
    image_names: set[str] = set()
    invalid_poses = 0
    missing_camera_refs = 0
    for image in data.images:
        if image.image_id in image_ids:
            result.add("error", "colmap.image.duplicate_id", "duplicate registered image id", image_id=image.image_id)
        image_ids.add(image.image_id)
        if image.name in image_names:
            result.add("error", "colmap.image.duplicate_name", "duplicate registered image name", name=image.name)
        image_names.add(image.name)
        if image.camera_id not in data.cameras:
            missing_camera_refs += 1
        qnorm = float(np.linalg.norm(image.qvec))
        if not np.isfinite(image.center).all() or not math.isfinite(qnorm) or abs(qnorm - 1.0) > 0.05:
            invalid_poses += 1
    if missing_camera_refs:
        result.add("error", "colmap.image.camera_reference", "images reference unknown cameras", count=missing_camera_refs)
    if invalid_poses:
        result.add("error", "colmap.image.invalid_pose", "registered images contain invalid poses", count=invalid_poses)
    if data.points_count < config.preflight.min_sparse_points:
        result.add(
            "error", "colmap.points.too_few", "too few sparse COLMAP points",
            actual=data.points_count, minimum=config.preflight.min_sparse_points,
        )
    if not data.points_finite:
        result.add("error", "colmap.points.nonfinite", "COLMAP sparse points contain non-finite coordinates")
    result.metrics.update({
        "format": data.source_format,
        "cameras": len(data.cameras), "camera_models": models,
        "registered_images": len(data.images), "sparse_points": data.points_count,
        "observations": int(sum(image.observations for image in data.images)),
        "triangulated_observations": int(sum(image.triangulated_observations for image in data.images)),
        "camera_centers": _robust_bounds(np.asarray([image.center for image in data.images], dtype=np.float64)),
        "sparse_bounds": _robust_bounds(data.point_sample),
    })
    return result


def _safe_image_path(root: Path, name: str) -> Optional[Path]:
    candidate = Path(name)
    if candidate.is_absolute():
        return None
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _check_images(config: SceneConfig, data: ColmapData) -> CheckResult:
    result = CheckResult("images")
    mode = config.preflight.image_check
    if mode == "none":
        result.status = "skipped"
        result.metrics["reason"] = "preflight.image_check=none"
        return result
    if mode == "all" or len(data.images) <= config.preflight.image_sample_count:
        selected = data.images
    else:
        indices = _sample_indices(len(data.images), config.preflight.image_sample_count)
        selected = [data.images[int(index)] for index in indices]
    missing: list[str] = []
    unsafe: list[str] = []
    zero_size: list[str] = []
    manifest = hashlib.sha256()
    total_bytes = 0
    for image in selected:
        path = _safe_image_path(config.inputs.image_root, image.name)
        if path is None:
            unsafe.append(image.name)
            continue
        if not path.is_file():
            missing.append(image.name)
            continue
        size = int(path.stat().st_size)
        total_bytes += size
        if size <= 0:
            zero_size.append(image.name)
        manifest.update(image.name.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(struct.pack("<Q", size))
    if unsafe:
        result.add(
            "error", "images.unsafe_name", "COLMAP image names must be relative and stay inside image_root",
            count=len(unsafe), examples=unsafe[:10],
        )
    if missing:
        result.add(
            "error", "images.missing", "registered COLMAP images are missing from image_root",
            count=len(missing), examples=missing[:10], checked=len(selected),
        )
    if zero_size:
        result.add("error", "images.empty", "registered image files are empty", count=len(zero_size), examples=zero_size[:10])
    result.metrics.update({
        "mode": mode, "checked": len(selected), "registered": len(data.images),
        "present": len(selected) - len(missing) - len(unsafe), "total_checked_bytes": total_bytes,
        "manifest_algorithm": "sha256-name-size-v1", "manifest_digest": manifest.hexdigest(),
        "complete_check": len(selected) == len(data.images),
    })
    if mode == "sample" and len(selected) < len(data.images):
        result.add(
            "warning", "images.sample_only", "only a deterministic sample of registered images was checked",
            checked=len(selected), registered=len(data.images),
        )
    return result


def _group_images(config: SceneConfig, data: ColmapData) -> tuple[CheckResult, dict[str, dict[tuple[str, str], ImageRecord]]]:
    result = CheckResult("camera_grouping")
    policy = config.camera_grouping
    groups: dict[str, dict[tuple[str, str], ImageRecord]] = {}
    if policy.mode == "single":
        for index, image in enumerate(data.images):
            groups[str(index)] = {("camera", "view"): image}
        result.metrics.update({"mode": "single", "groups": len(groups), "matched_images": len(data.images)})
        return result, groups
    if policy.mode in {"auto", "manifest"}:
        result.status = "skipped" if policy.mode == "manifest" else "warn"
        if policy.mode == "auto":
            result.add(
                "warning", "grouping.auto_deferred",
                "camera grouping is deferred to the generic selector; use regex mode for strict rig completeness checks",
            )
        else:
            result.metrics["reason"] = "manifest grouping is validated by the runner that consumes the manifest"
        result.metrics["mode"] = policy.mode
        return result, groups
    assert policy.pattern is not None
    pattern = re.compile(policy.pattern)
    unmatched: list[str] = []
    duplicate_slots: list[dict[str, str]] = []
    for image in data.images:
        match = pattern.search(image.name)
        if match is None:
            unmatched.append(image.name)
            continue
        timestamp = match.group(policy.timestamp_group)
        member = match.group(policy.member_group) if policy.member_group in pattern.groupindex else "camera"
        view = match.group(policy.view_group) if policy.view_group in pattern.groupindex else "view"
        slot = (member, view)
        if slot in groups.setdefault(timestamp, {}):
            duplicate_slots.append({"timestamp": timestamp, "member": member, "view": view})
        else:
            groups[timestamp][slot] = image
    if unmatched:
        result.add(
            "warning", "grouping.unmatched_images", "some registered images do not match camera_grouping.pattern",
            count=len(unmatched), examples=unmatched[:10],
        )
    if duplicate_slots:
        result.add(
            "error", "grouping.duplicate_slot", "multiple images map to the same timestamp/member/view slot",
            count=len(duplicate_slots), examples=duplicate_slots[:10],
        )
    expected_slots = {
        (member, view)
        for member in (policy.required_members or ("camera",))
        for view in (policy.required_views or ("view",))
    }
    incomplete: list[str] = []
    if policy.required_members or policy.required_views:
        incomplete = [timestamp for timestamp, slots in groups.items() if not expected_slots.issubset(slots)]
        if incomplete:
            result.add(
                "warning", "grouping.incomplete_timestamps",
                "some timestamps do not contain every required member/view combination",
                count=len(incomplete), examples=incomplete[:10], expected_slots=sorted(expected_slots),
            )
    complete_count = len(groups) - len(incomplete)
    if complete_count < 2:
        result.add("error", "grouping.too_few_complete", "fewer than two complete timestamps are available", complete=complete_count)
    result.metrics.update({
        "mode": "regex", "pattern": policy.pattern, "groups": len(groups),
        "complete_groups": complete_count, "incomplete_groups": len(incomplete),
        "matched_images": len(data.images) - len(unmatched), "unmatched_images": len(unmatched),
    })
    return result, groups


def _check_ply(config: SceneConfig, data: PlyDataSummary) -> CheckResult:
    result = CheckResult("gaussian_ply")
    properties = set(data.properties)
    required_gaussian = {
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    has_color = {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(properties) or {"red", "green", "blue"}.issubset(properties)
    missing = sorted(required_gaussian - properties)
    if config.preflight.require_3dgs_schema and (missing or not has_color):
        result.add(
            "error", "ply.schema.not_3dgs", "PLY does not contain the required 3D Gaussian vertex schema",
            missing=missing, has_dc_or_rgb_color=has_color,
        )
    if data.vertex_count < config.preflight.min_gaussians:
        result.add(
            "error", "ply.vertices.too_few", "too few Gaussian vertices",
            actual=data.vertex_count, minimum=config.preflight.min_gaussians,
        )
    if not data.finite_coordinates:
        scope = "all" if data.sample_is_full else "sampled"
        result.add("error", "ply.coordinates.nonfinite", f"{scope} PLY coordinates contain NaN/Inf")
    if not data.finite_gaussian_fields:
        scope = "all" if data.sample_is_full else "sampled"
        result.add("error", "ply.gaussian_fields.nonfinite", f"{scope} Gaussian fields contain NaN/Inf")
    result.metrics.update({
        "format": data.format, "vertex_count": data.vertex_count,
        "property_count": len(data.properties), "properties": list(data.properties),
        "sampled_records": data.sampled_records, "sample_is_full": data.sample_is_full,
        "bounds": _robust_bounds(data.point_sample),
        "validation_scope": "full" if data.sample_is_full else "deterministic_sample",
    })
    if not data.sample_is_full:
        result.add(
            "info", "ply.sampled_validation",
            "finite values and bounds were checked on a deterministic sample; header count and binary size were checked exactly",
            sampled=data.sampled_records, vertices=data.vertex_count,
        )
    return result


def _check_metric_scale(
    config: SceneConfig,
    groups: dict[str, dict[tuple[str, str], ImageRecord]],
) -> CheckResult:
    result = CheckResult("metric_scale")
    policy = config.metric_scale
    source = policy.source
    if source in {"explicit", "declared_metric"}:
        assert policy.meters_per_colmap_unit is not None
        result.metrics.update({
            "source": source,
            "meters_per_colmap_unit": policy.meters_per_colmap_unit,
            "evidence": policy.evidence,
        })
        if not policy.evidence:
            severity = "error" if policy.required else "warning"
            result.add(
                severity, "metric.evidence_missing",
                "a scale factor was declared without an auditable evidence note",
            )
        return result

    use_baseline = source == "rig_baseline" or (
        source == "auto" and policy.expected_baseline_m is not None and len(policy.baseline_members) == 2
    )
    if use_baseline:
        if policy.expected_baseline_m is None or len(policy.baseline_members) != 2:
            result.add("error", "metric.baseline_config", "baseline evidence is incomplete")
            return result
        first_member, second_member = policy.baseline_members
        distances = []
        for slots in groups.values():
            first_candidates = [
                image for (member, view), image in slots.items()
                if member == first_member and (policy.baseline_view is None or view == policy.baseline_view)
            ]
            second_candidates = [
                image for (member, view), image in slots.items()
                if member == second_member and (policy.baseline_view is None or view == policy.baseline_view)
            ]
            if len(first_candidates) == 1 and len(second_candidates) == 1:
                distances.append(float(np.linalg.norm(first_candidates[0].center - second_candidates[0].center)))
        values = np.asarray(distances, dtype=np.float64)
        if not len(values) or not np.isfinite(values).all() or float(np.median(values)) <= 1e-12:
            result.add(
                "error", "metric.baseline_unavailable",
                "no valid paired camera baselines could be measured",
                measured_pairs=len(values),
            )
            return result
        raw_median = float(np.median(values))
        meters_per_unit = policy.meters_per_colmap_unit or policy.expected_baseline_m / raw_median
        measured_m = values * meters_per_unit
        absolute_error = np.abs(measured_m - policy.expected_baseline_m)
        result.metrics.update({
            "source": "rig_baseline", "pairs": len(values),
            "members": list(policy.baseline_members), "view": policy.baseline_view,
            "raw_median_colmap_units": raw_median,
            "inferred_meters_per_colmap_unit": float(meters_per_unit),
            "expected_baseline_m": policy.expected_baseline_m,
            "measured_baseline_median_m": float(np.median(measured_m)),
            "measured_baseline_p05_m": float(np.quantile(measured_m, 0.05)),
            "measured_baseline_p95_m": float(np.quantile(measured_m, 0.95)),
            "max_absolute_error_m": float(np.max(absolute_error)),
        })
        outliers = int(np.count_nonzero(absolute_error > policy.baseline_tolerance_m))
        if outliers:
            severity = "error" if outliers == len(values) or policy.required else "warning"
            result.add(
                severity, "metric.baseline_tolerance",
                "camera baseline is not stable within the configured tolerance",
                outliers=outliers, pairs=len(values), tolerance_m=policy.baseline_tolerance_m,
            )
        return result

    result.metrics.update({"source": source, "meters_per_colmap_unit": None})
    severity = "error" if policy.required else "warning"
    result.add(
        severity, "metric.scale_unknown",
        "COLMAP and 3DGS may share coordinates, but no evidence establishes that one unit is one metre",
        hint="Provide declared_metric/explicit evidence or a known rig baseline.",
    )
    return result


def _configured_up(config: SceneConfig) -> Optional[np.ndarray]:
    policy = config.gravity
    if policy.mode == "axis":
        assert policy.axis is not None
        sign = -1.0 if policy.axis.startswith("-") else 1.0
        axis = policy.axis[-1]
        value = np.zeros(3, dtype=np.float64)
        value[{"x": 0, "y": 1, "z": 2}[axis]] = sign
        return value
    if policy.mode == "vector":
        assert policy.vector is not None
        value = np.asarray(policy.vector, dtype=np.float64)
        return value / np.linalg.norm(value)
    return None


def _camera_up_consensus(images: Sequence[ImageRecord]) -> tuple[Optional[np.ndarray], float]:
    if not images:
        return None, 0.0
    # COLMAP camera +Y points down in the image, hence world up is -R^T e_y.
    values = np.asarray([-image.rotation.T[:, 1] for image in images], dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return None, 0.0
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    mean = np.mean(values, axis=0)
    concentration = float(np.linalg.norm(mean))
    if concentration <= 1e-12:
        return None, concentration
    return mean / concentration, concentration


def _check_gravity(config: SceneConfig, data: ColmapData) -> CheckResult:
    result = CheckResult("gravity")
    policy = config.gravity
    consensus_up, concentration = _camera_up_consensus(data.images)
    result.metrics["camera_up_consensus"] = concentration
    result.metrics["camera_inferred_up"] = consensus_up.tolist() if consensus_up is not None else None
    explicit = _configured_up(config)
    if explicit is not None:
        result.metrics.update({"mode": policy.mode, "resolved_up": explicit.tolist(), "source": "config"})
        if consensus_up is not None:
            cosine = float(np.clip(np.dot(explicit, consensus_up), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            result.metrics["camera_disagreement_deg"] = angle
            if concentration >= 0.5 and angle > 60.0:
                result.add(
                    "warning", "gravity.camera_disagreement",
                    "configured up direction strongly disagrees with registered camera roll",
                    angle_deg=angle, consensus=concentration,
                )
        return result
    if policy.mode == "none":
        severity = "error" if policy.required else "warning"
        result.add(severity, "gravity.disabled", "no gravity/up policy is configured")
        result.metrics.update({"mode": "none", "resolved_up": None})
        return result
    if consensus_up is None or concentration < policy.min_camera_consensus:
        severity = "error" if policy.required else "warning"
        result.add(
            severity, "gravity.auto_low_confidence",
            "camera orientations do not provide a confident automatic up direction",
            consensus=concentration, minimum=policy.min_camera_consensus,
            hint="Provide gravity.axis/vector or estimate gravity downstream from a supported floor plane.",
        )
        result.metrics.update({"mode": "auto", "resolved_up": None, "source": "camera_pose_consensus"})
    else:
        result.metrics.update({
            "mode": "auto", "resolved_up": consensus_up.tolist(),
            "source": "camera_pose_consensus", "confidence": concentration,
        })
    return result


def _check_alignment(config: SceneConfig, colmap: ColmapData, ply: PlyDataSummary) -> CheckResult:
    result = CheckResult("colmap_3dgs_alignment")
    sparse = colmap.point_sample[np.isfinite(colmap.point_sample).all(axis=1)]
    gaussian = ply.point_sample[np.isfinite(ply.point_sample).all(axis=1)]
    if len(sparse) < 4 or len(gaussian) < 4:
        result.add("error", "alignment.insufficient_points", "not enough finite points for a spatial alignment check")
        return result
    sparse_low = np.quantile(sparse, 0.01, axis=0) if len(sparse) >= 100 else np.min(sparse, axis=0)
    sparse_high = np.quantile(sparse, 0.99, axis=0) if len(sparse) >= 100 else np.max(sparse, axis=0)
    gaussian_low = np.quantile(gaussian, 0.01, axis=0) if len(gaussian) >= 100 else np.min(gaussian, axis=0)
    gaussian_high = np.quantile(gaussian, 0.99, axis=0) if len(gaussian) >= 100 else np.max(gaussian, axis=0)
    sparse_diagonal = float(np.linalg.norm(sparse_high - sparse_low))
    gaussian_diagonal = float(np.linalg.norm(gaussian_high - gaussian_low))
    common_diagonal = max(sparse_diagonal, gaussian_diagonal, 1e-12)
    centroid_offset = float(np.linalg.norm(np.median(sparse, axis=0) - np.median(gaussian, axis=0)))
    centroid_ratio = centroid_offset / common_diagonal
    extent_ratio = max(sparse_diagonal, gaussian_diagonal) / max(min(sparse_diagonal, gaussian_diagonal), 1e-12)
    bbox_iou = _aabb_iou(sparse_low, sparse_high, gaussian_low, gaussian_high)
    nearest = _nearest_distance_metrics(
        sparse, gaussian, config.preflight.alignment.nearest_neighbor_points, common_diagonal
    )
    centers = np.asarray([image.center for image in colmap.images], dtype=np.float64)
    inside_expanded = 0.0
    if len(centers):
        margin = 0.5 * common_diagonal
        inside_expanded = float(np.mean(np.all((centers >= gaussian_low - margin) & (centers <= gaussian_high + margin), axis=1)))
    result.metrics.update({
        "method": "robust-aabb-centroid-nearest-v1",
        "colmap_sample_points": len(sparse), "gaussian_sample_points": len(gaussian),
        "colmap_robust_diagonal": sparse_diagonal, "gaussian_robust_diagonal": gaussian_diagonal,
        "centroid_offset": centroid_offset, "centroid_offset_ratio": centroid_ratio,
        "extent_ratio": extent_ratio, "robust_bbox_iou": bbox_iou,
        "sparse_to_gaussian_nearest": nearest,
        "camera_centers_inside_expanded_gaussian_bounds": inside_expanded,
        "note": "This is a lightweight coordinate-frame gate, not a substitute for rendered RGB/depth agreement QA.",
    })
    policy = config.preflight.alignment
    hard_failure = (
        centroid_ratio > policy.fail_centroid_offset_ratio
        or extent_ratio > policy.fail_extent_ratio
        or bbox_iou < policy.fail_bbox_iou
    )
    soft_failure = (
        centroid_ratio > policy.pass_centroid_offset_ratio
        or extent_ratio > policy.pass_extent_ratio
        or bbox_iou < policy.pass_bbox_iou
    )
    if hard_failure:
        result.add(
            "error", "alignment.spatial_mismatch",
            "COLMAP sparse geometry and 3DGS PLY do not appear to share a coordinate frame/scale",
            centroid_offset_ratio=centroid_ratio, extent_ratio=extent_ratio, bbox_iou=bbox_iou,
        )
    elif soft_failure:
        result.add(
            "warning", "alignment.spatial_uncertain",
            "COLMAP and 3DGS overlap is plausible but outside pass thresholds; rendered alignment QA is required",
            centroid_offset_ratio=centroid_ratio, extent_ratio=extent_ratio, bbox_iou=bbox_iou,
        )
    return result


def _check_fingerprints(config: SceneConfig, source_files: Sequence[Path]) -> CheckResult:
    result = CheckResult("fingerprints")
    files = [config.config_path, config.inputs.gaussian_ply, *source_files]
    unique = []
    seen: set[Path] = set()
    for path in files:
        if path.is_file() and path not in seen:
            seen.add(path)
            unique.append(path)
    fingerprints = []
    for path in unique:
        try:
            fingerprints.append(
                fingerprint_file(
                    path,
                    config.preflight.full_hash_max_bytes,
                    config.preflight.hash_chunk_bytes,
                    force_full=path == config.inputs.gaussian_ply,
                )
            )
        except OSError as exc:
            result.add("error", "fingerprint.read_error", "failed to fingerprint an input", path=str(path), error=str(exc))
    normalized_config = config.normalized_dict()
    result.metrics.update({
        "config_sha256": _canonical_sha256(normalized_config),
        "files": fingerprints,
        "input_set_digest": _canonical_sha256(
            [{"path": item["path"], "size_bytes": item["size_bytes"], "algorithm": item["algorithm"], "digest": item["digest"]} for item in fingerprints]
        ),
    })
    return result


@dataclass
class PreflightReport:
    scene_id: str
    config: dict[str, Any]
    checks: list[CheckResult]
    started_unix_s: float
    duration_s: float

    @property
    def errors(self) -> int:
        return sum(finding.severity == "error" for check in self.checks for finding in check.findings)

    @property
    def warnings(self) -> int:
        return sum(finding.severity == "warning" for check in self.checks for finding in check.findings)

    @property
    def status(self) -> str:
        if self.errors:
            return "fail"
        if self.warnings:
            return "warn"
        return "pass"

    @property
    def ready_for_pipeline(self) -> bool:
        return self.errors == 0

    @property
    def strict_ready(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema_version": REPORT_SCHEMA,
            "scene_id": self.scene_id,
            "status": self.status,
            "ready_for_pipeline": self.ready_for_pipeline,
            "strict_ready": self.strict_ready,
            "errors": self.errors,
            "warnings": self.warnings,
            "started_unix_s": self.started_unix_s,
            "duration_s": self.duration_s,
            "config": self.config,
            "checks": [asdict(check) for check in self.checks],
        })


def _timed(check_function: Any, *args: Any) -> CheckResult:
    started = time.perf_counter()
    result = check_function(*args)
    result.duration_s = time.perf_counter() - started
    return result


def _failed_check(name: str, code: str, message: str, exc: BaseException) -> CheckResult:
    result = CheckResult(name)
    result.add("error", code, message, error=f"{type(exc).__name__}: {exc}")
    return result


def run_preflight(config: SceneConfig) -> PreflightReport:
    """Run every applicable input check without modifying the scene or output."""

    wall_started = time.time()
    monotonic_started = time.perf_counter()
    checks: list[CheckResult] = []
    path_check = _timed(_check_paths, config)
    checks.append(path_check)
    colmap: Optional[ColmapData] = None
    ply: Optional[PlyDataSummary] = None
    groups: dict[str, dict[tuple[str, str], ImageRecord]] = {}

    if path_check.status != "fail":
        started = time.perf_counter()
        try:
            colmap = read_colmap_model(config.inputs.colmap_model, config.preflight.alignment.sample_points)
            check = _check_colmap(config, colmap)
        except (OSError, UnicodeError, ValueError, struct.error) as exc:
            check = _failed_check("colmap", "colmap.read_failed", "failed to read/validate COLMAP sparse model", exc)
        check.duration_s = time.perf_counter() - started
        checks.append(check)

        started = time.perf_counter()
        try:
            ply = read_gaussian_ply(config.inputs.gaussian_ply, config.preflight.alignment.sample_points)
            check = _check_ply(config, ply)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            check = _failed_check("gaussian_ply", "ply.read_failed", "failed to read/validate 3DGS PLY", exc)
        check.duration_s = time.perf_counter() - started
        checks.append(check)

    if colmap is not None:
        checks.append(_timed(_check_images, config, colmap))
        started = time.perf_counter()
        grouping_check, groups = _group_images(config, colmap)
        grouping_check.duration_s = time.perf_counter() - started
        checks.append(grouping_check)
        checks.append(_timed(_check_metric_scale, config, groups))
        checks.append(_timed(_check_gravity, config, colmap))
    if colmap is not None and ply is not None:
        checks.append(_timed(_check_alignment, config, colmap, ply))
    source_files = colmap.source_files if colmap is not None else []
    checks.append(_timed(_check_fingerprints, config, source_files))
    return PreflightReport(
        scene_id=config.scene_id,
        config=config.normalized_dict(),
        checks=checks,
        started_unix_s=wall_started,
        duration_s=time.perf_counter() - monotonic_started,
    )


def format_summary(report: PreflightReport) -> str:
    lines = [
        f"FARM preflight: {report.scene_id}",
        f"status={report.status} errors={report.errors} warnings={report.warnings} duration={report.duration_s:.2f}s",
    ]
    for check in report.checks:
        lines.append(f"  {check.status.upper():7s} {check.name} ({check.duration_s:.2f}s)")
        for finding in check.findings:
            if finding.severity in {"error", "warning"}:
                lines.append(f"    [{finding.severity}] {finding.code}: {finding.message}")
    return "\n".join(lines)
