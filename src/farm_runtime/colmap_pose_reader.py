"""Small CPU-only COLMAP camera/image reader.

Tracker subset planning needs registered names and poses, not the sparse point
cloud or millions of track observations.  Loading a full ``pycolmap``
reconstruction for that task wastes memory, so this reader streams only
``cameras`` and ``images`` and seeks over POINTS2D payloads.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImagePose:
    image_id: int
    camera_id: int
    name: str
    camera_from_world: np.ndarray


_CAMERA_MODELS: dict[int, tuple[str, int]] = {
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


def quaternion_wxyz_to_rotation(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("invalid COLMAP quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose(
    image_id: int,
    camera_id: int,
    name: str,
    quaternion: object,
    translation: object,
) -> ColmapImagePose:
    matrix = np.column_stack(
        (
            quaternion_wxyz_to_rotation(quaternion),
            np.asarray(translation, dtype=np.float64).reshape(3),
        )
    )
    if not name:
        raise ValueError(f"COLMAP image {image_id} has an empty name")
    return ColmapImagePose(
        image_id=int(image_id),
        camera_id=int(camera_id),
        name=str(name),
        camera_from_world=matrix,
    )


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    payload = stream.read(int(size))
    if len(payload) != int(size):
        raise ValueError("truncated COLMAP binary model")
    return payload


def _skip_exact(stream: BinaryIO, size: int) -> None:
    """Seek over a binary payload after proving it remains inside the file."""

    count = int(size)
    if count < 0:
        raise ValueError("negative COLMAP payload length")
    start = stream.tell()
    end = os.fstat(stream.fileno()).st_size
    if start + count > end:
        raise ValueError("truncated COLMAP binary model")
    stream.seek(count, os.SEEK_CUR)


def _unpack(stream: BinaryIO, layout: str) -> tuple[object, ...]:
    record = struct.Struct("<" + layout)
    return record.unpack(_read_exact(stream, record.size))


def _read_c_string(stream: BinaryIO) -> str:
    payload = bytearray()
    while True:
        character = _read_exact(stream, 1)
        if character == b"\0":
            break
        payload.extend(character)
        if len(payload) > 1_048_576:
            raise ValueError("unreasonably long COLMAP image name")
    return payload.decode("utf-8", errors="strict")


def _read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(int(count)):
            camera_id, model_id, width, height = _unpack(stream, "IiQQ")
            if int(model_id) not in _CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            model, param_count = _CAMERA_MODELS[int(model_id)]
            params = tuple(float(value) for value in _unpack(stream, "d" * param_count))
            camera_id = int(camera_id)
            if camera_id in cameras:
                raise ValueError(f"duplicate COLMAP camera id {camera_id}")
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model,
                width=int(width),
                height=int(height),
                params=params,
            )
        if stream.read(1):
            raise ValueError("trailing bytes in cameras.bin")
    return cameras


def _read_images_binary(path: Path) -> list[ColmapImagePose]:
    images: list[ColmapImagePose] = []
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(int(count)):
            values = _unpack(stream, "I" + "d" * 7 + "I")
            name = _read_c_string(stream)
            (observations,) = _unpack(stream, "Q")
            # POINT2D is x:double, y:double, point3D_id:int64.
            _skip_exact(stream, int(observations) * 24)
            images.append(
                _pose(
                    int(values[0]),
                    int(values[8]),
                    name,
                    values[1:5],
                    values[5:8],
                )
            )
        if stream.read(1):
            raise ValueError("trailing bytes in images.bin")
    return images


def _read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"{path}:{line_number}: malformed camera record")
            camera_id = int(parts[0])
            if camera_id in cameras:
                raise ValueError(
                    f"{path}:{line_number}: duplicate camera id {camera_id}"
                )
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=parts[1].upper(),
                width=int(parts[2]),
                height=int(parts[3]),
                params=tuple(float(value) for value in parts[4:]),
            )
    return cameras


def _read_images_text(path: Path) -> list[ColmapImagePose]:
    images: list[ColmapImagePose] = []
    pending: ColmapImagePose | None = None
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pending is None:
                if not stripped:
                    continue
                parts = stripped.split(maxsplit=9)
                if len(parts) != 10:
                    raise ValueError(
                        f"{path}:{line_number}: malformed image pose record"
                    )
                pending = _pose(
                    int(parts[0]),
                    int(parts[8]),
                    parts[9],
                    [float(value) for value in parts[1:5]],
                    [float(value) for value in parts[5:8]],
                )
            else:
                if stripped and len(stripped.split()) % 3:
                    raise ValueError(f"{path}:{line_number}: malformed POINTS2D record")
                images.append(pending)
                pending = None
    if pending is not None:
        images.append(pending)
    return images


def read_colmap_cameras_and_poses(
    model_dir: Path,
) -> tuple[str, dict[int, ColmapCamera], list[ColmapImagePose], tuple[Path, Path]]:
    """Read cameras and registered image poses without loading points3D."""

    root = Path(model_dir).expanduser().resolve()
    binary = root / "cameras.bin", root / "images.bin"
    text = root / "cameras.txt", root / "images.txt"
    if all(path.is_file() for path in binary):
        cameras = _read_cameras_binary(binary[0])
        images = _read_images_binary(binary[1])
        source_format, source_files = "binary", binary
    elif all(path.is_file() for path in text):
        cameras = _read_cameras_text(text[0])
        images = _read_images_text(text[1])
        source_format, source_files = "text", text
    else:
        raise FileNotFoundError(
            f"{root} needs cameras/images in matching COLMAP binary or text form"
        )
    if not cameras or not images:
        raise ValueError("COLMAP model contains no cameras or registered images")
    image_ids: set[int] = set()
    image_names: set[str] = set()
    for image in images:
        if image.image_id in image_ids:
            raise ValueError(f"duplicate COLMAP image id {image.image_id}")
        if image.name in image_names:
            raise ValueError(f"duplicate COLMAP image name {image.name!r}")
        if image.camera_id not in cameras:
            raise ValueError(
                f"COLMAP image {image.image_id} refers to missing camera {image.camera_id}"
            )
        image_ids.add(image.image_id)
        image_names.add(image.name)
    return source_format, cameras, images, source_files
