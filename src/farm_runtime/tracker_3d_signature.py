"""CPU-only sparse-3D signatures for episode-local tracker masks.

The stage never associates IDs across episodes. Raw mask/COLMAP evidence is
kept separate from fail-closed forwarding decisions so policy ablations cannot
silently rewrite measurements.
"""

from __future__ import annotations

import math
import os
import struct
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from farm_runtime.colmap_pose_reader import (
    ColmapCamera,
    ColmapImagePose,
    read_colmap_cameras_and_poses,
)

SCHEMA = "farm.tracker-episode-3d-signatures.v1"
SHADOW_SCHEMA = "farm.tracker-episode-3d-signatures.v2-shadow"
EPISODE_SCHEMA = "farm.materialized-tracker-episode.v1"
MAX_LOCAL_ID = 254


@dataclass(frozen=True)
class ColmapImageObservations:
    image_id: int
    camera_id: int
    name: str
    xy: np.ndarray
    point3d_ids: np.ndarray

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy)
        point_ids = np.asarray(self.point3d_ids)
        if xy.ndim != 2 or xy.shape[1:] != (2,):
            raise ValueError("COLMAP xy must have shape [N,2]")
        if point_ids.shape != (xy.shape[0],):
            raise ValueError("COLMAP point IDs must align with xy")


@dataclass(frozen=True)
class SparsePoint3D:
    point3d_id: int
    xyz_scene: tuple[float, float, float]
    reprojection_error_px: float
    track_length: int


@dataclass(frozen=True)
class GatePolicy:
    """Conservative defaults for forwarding an ID to global association."""

    minimum_frame_support: int = 3
    minimum_physical_timestamps: int = 3
    minimum_persistence_ratio: float = 0.25
    minimum_median_area_px: int = 64
    tiny_area_px: int = 64
    maximum_tiny_frame_fraction: float = 0.50
    minimum_median_largest_2d_component_fraction: float = 0.60
    minimum_sparse_observations: int = 12
    minimum_unique_sparse_points: int = 8
    minimum_point_observations: int = 2
    minimum_point_label_purity: float = 0.67
    minimum_qualified_sparse_points: int = 6
    minimum_qualified_sparse_point_fraction: float = 0.25
    maximum_ambiguous_sparse_point_fraction: float = 0.35
    minimum_largest_3d_component_fraction: float = 0.50
    require_points3d: bool = True

    def validate(self) -> None:
        integer_fields = (
            "minimum_frame_support",
            "minimum_physical_timestamps",
            "minimum_median_area_px",
            "tiny_area_px",
            "minimum_sparse_observations",
            "minimum_unique_sparse_points",
            "minimum_point_observations",
            "minimum_qualified_sparse_points",
        )
        for field in integer_fields:
            if int(getattr(self, field)) < 0:
                raise ValueError(f"{field} must be non-negative")
        ratio_fields = (
            "minimum_persistence_ratio",
            "maximum_tiny_frame_fraction",
            "minimum_median_largest_2d_component_fraction",
            "minimum_point_label_purity",
            "minimum_qualified_sparse_point_fraction",
            "maximum_ambiguous_sparse_point_fraction",
            "minimum_largest_3d_component_fraction",
        )
        for field in ratio_fields:
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")


def _read_exact(stream: Any, size: int) -> bytes:
    count = int(size)
    if count < 0:
        raise ValueError("negative COLMAP payload length")
    payload = stream.read(count)
    if len(payload) != count:
        raise ValueError("truncated COLMAP binary model")
    return payload


def _skip_exact(stream: Any, size: int) -> None:
    count = int(size)
    if count < 0:
        raise ValueError("negative COLMAP payload length")
    start = stream.tell()
    end = os.fstat(stream.fileno()).st_size
    if start + count > end:
        raise ValueError("truncated COLMAP binary model")
    stream.seek(count, os.SEEK_CUR)


def _unpack(stream: Any, layout: str) -> tuple[Any, ...]:
    record = struct.Struct("<" + layout)
    return record.unpack(_read_exact(stream, record.size))


def _read_c_string(stream: Any) -> str:
    payload = bytearray()
    while True:
        value = _read_exact(stream, 1)
        if value == b"\0":
            return payload.decode("utf-8", errors="strict")
        payload.extend(value)
        if len(payload) > 1_048_576:
            raise ValueError("unreasonably long COLMAP image name")


def _observation_arrays(payload: bytes, count: int) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3d_id", "<i8")])
    values = np.frombuffer(payload, dtype=dtype, count=int(count))
    xy = np.column_stack((values["x"], values["y"])).astype(np.float64, copy=False)
    point_ids = values["point3d_id"].astype(np.int64, copy=True)
    return np.ascontiguousarray(xy), point_ids


def _read_selected_images_binary(
    path: Path, selected_ids: set[int]
) -> dict[int, ColmapImageObservations]:
    selected: dict[int, ColmapImageObservations] = {}
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    header = struct.Struct("<I" + "d" * 7 + "I")
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(int(count)):
            values = header.unpack(_read_exact(stream, header.size))
            image_id, camera_id = int(values[0]), int(values[8])
            name = _read_c_string(stream)
            if image_id in seen_ids or name in seen_names:
                raise ValueError("duplicate COLMAP image ID or name")
            seen_ids.add(image_id)
            seen_names.add(name)
            (observation_count,) = _unpack(stream, "Q")
            payload_size = int(observation_count) * 24
            if image_id in selected_ids:
                xy, point_ids = _observation_arrays(
                    _read_exact(stream, payload_size), int(observation_count)
                )
                selected[image_id] = ColmapImageObservations(
                    image_id, camera_id, name, xy, point_ids
                )
            else:
                _skip_exact(stream, payload_size)
        if stream.read(1):
            raise ValueError("trailing bytes in COLMAP images.bin")
    missing = sorted(selected_ids.difference(selected))
    if missing:
        raise ValueError(f"episode image IDs missing from images.bin: {missing[:8]}")
    return selected


def _parse_text_points2d(
    path: Path, line_number: int, value: str
) -> tuple[np.ndarray, np.ndarray]:
    parts = value.split()
    if len(parts) % 3:
        raise ValueError(f"{path}:{line_number}: malformed POINTS2D record")
    if not parts:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64)
    try:
        rows = np.asarray(parts, dtype=object).reshape(-1, 3)
        xy = rows[:, :2].astype(np.float64)
        point_ids = rows[:, 2].astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: malformed POINTS2D values") from exc
    return np.ascontiguousarray(xy), np.ascontiguousarray(point_ids)


def _read_selected_images_text(
    path: Path, selected_ids: set[int]
) -> dict[int, ColmapImageObservations]:
    selected: dict[int, ColmapImageObservations] = {}
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    pending: tuple[int, int, str] | None = None
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
                try:
                    image_id, camera_id = int(parts[0]), int(parts[8])
                    pose_values = [float(value) for value in parts[1:8]]
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: malformed image pose values"
                    ) from exc
                if not np.isfinite(pose_values).all():
                    raise ValueError(f"{path}:{line_number}: non-finite image pose")
                name = parts[9]
                if image_id in seen_ids or name in seen_names:
                    raise ValueError("duplicate COLMAP image ID or name")
                seen_ids.add(image_id)
                seen_names.add(name)
                pending = image_id, camera_id, name
            else:
                image_id, camera_id, name = pending
                if image_id in selected_ids:
                    xy, point_ids = _parse_text_points2d(path, line_number, stripped)
                    selected[image_id] = ColmapImageObservations(
                        image_id, camera_id, name, xy, point_ids
                    )
                elif stripped and len(stripped.split()) % 3:
                    raise ValueError(f"{path}:{line_number}: malformed POINTS2D record")
                pending = None
    if pending is not None:
        image_id, camera_id, name = pending
        if image_id in selected_ids:
            selected[image_id] = ColmapImageObservations(
                image_id,
                camera_id,
                name,
                np.zeros((0, 2), dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )
    missing = sorted(selected_ids.difference(selected))
    if missing:
        raise ValueError(f"episode image IDs missing from images.txt: {missing[:8]}")
    return selected


def read_selected_image_observations(
    model_dir: Path, *, source_format: str, selected_image_ids: Iterable[int]
) -> tuple[dict[int, ColmapImageObservations], Path]:
    selected_ids = {int(value) for value in selected_image_ids}
    if not selected_ids or min(selected_ids) < 0:
        raise ValueError("selected COLMAP image IDs must be non-empty and non-negative")
    root = Path(model_dir).expanduser().resolve(strict=True)
    if source_format == "binary":
        path = root / "images.bin"
        return _read_selected_images_binary(path, selected_ids), path
    if source_format == "text":
        path = root / "images.txt"
        return _read_selected_images_text(path, selected_ids), path
    raise ValueError(f"unsupported COLMAP source format {source_format!r}")


def _read_selected_points3d_binary(
    path: Path, selected_ids: set[int]
) -> dict[int, SparsePoint3D]:
    selected: dict[int, SparsePoint3D] = {}
    header = struct.Struct("<QdddBBBdQ")
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(int(count)):
            values = header.unpack(_read_exact(stream, header.size))
            point_id = int(values[0])
            xyz = tuple(float(value) for value in values[1:4])
            error, track_length = float(values[7]), int(values[8])
            if point_id in selected_ids:
                if point_id in selected:
                    raise ValueError(f"duplicate COLMAP point3D id {point_id}")
                if not np.isfinite((*xyz, error)).all():
                    raise ValueError(f"non-finite COLMAP point3D {point_id}")
                selected[point_id] = SparsePoint3D(point_id, xyz, error, track_length)
            _skip_exact(stream, track_length * 8)
        if stream.read(1):
            raise ValueError("trailing bytes in COLMAP points3D.bin")
    return selected


def _read_selected_points3d_text(
    path: Path, selected_ids: set[int]
) -> dict[int, SparsePoint3D]:
    selected: dict[int, SparsePoint3D] = {}
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8 or (len(parts) - 8) % 2:
                raise ValueError(f"{path}:{line_number}: malformed point3D record")
            point_id = int(parts[0])
            if point_id not in selected_ids:
                continue
            if point_id in selected:
                raise ValueError(f"duplicate COLMAP point3D id {point_id}")
            xyz = tuple(float(value) for value in parts[1:4])
            error = float(parts[7])
            if not np.isfinite((*xyz, error)).all():
                raise ValueError(f"{path}:{line_number}: non-finite point3D")
            selected[point_id] = SparsePoint3D(
                point_id, xyz, error, (len(parts) - 8) // 2
            )
    return selected


def read_selected_points3d(
    model_dir: Path, *, source_format: str, selected_point_ids: Iterable[int]
) -> tuple[dict[int, SparsePoint3D], Path | None]:
    selected_ids = {int(value) for value in selected_point_ids if int(value) >= 0}
    root = Path(model_dir).expanduser().resolve(strict=True)
    path = root / ("points3D.bin" if source_format == "binary" else "points3D.txt")
    if not path.is_file():
        return {}, None
    if not selected_ids:
        return {}, path
    if source_format == "binary":
        return _read_selected_points3d_binary(path, selected_ids), path
    if source_format == "text":
        return _read_selected_points3d_text(path, selected_ids), path
    raise ValueError(f"unsupported COLMAP source format {source_format!r}")


def camera_to_world_metric(
    camera_from_world: np.ndarray, *, meters_per_scene_unit: float
) -> np.ndarray:
    scale = float(meters_per_scene_unit)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("meters_per_scene_unit must be finite and positive")
    pose = np.asarray(camera_from_world, dtype=np.float64)
    if pose.shape != (3, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_from_world must have finite shape [3,4]")
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :] = pose
    try:
        result = np.linalg.inv(homogeneous)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular COLMAP image pose") from exc
    result[:3, 3] *= scale
    return result


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(
        second, dtype=np.float64
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def validate_episode_contract(
    episode: Mapping[str, Any],
    *,
    cameras: Mapping[int, ColmapCamera],
    poses_by_id: Mapping[int, ColmapImagePose],
    observations_by_id: Mapping[int, ColmapImageObservations],
    meters_per_scene_unit: float,
    maximum_translation_error_m: float,
    maximum_rotation_error_degrees: float,
) -> list[dict[str, Any]]:
    if episode.get("schema") != EPISODE_SCHEMA:
        raise ValueError("unsupported materialized tracker episode schema")
    episode_id = str(episode.get("episode_id") or "")
    if not episode_id or Path(episode_id).name != episode_id:
        raise ValueError("episode_id must be a safe non-empty basename")
    frames = episode.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("episode has no frames")
    if int(episode.get("frame_count", -1)) != len(frames):
        raise ValueError("episode frame_count disagrees with frames")
    pose_names = {pose.name: pose.image_id for pose in poses_by_id.values()}
    materialized_names: set[str] = set()
    source_names: set[str] = set()
    timestamps: set[str] = set()
    camera_ids: set[int] = set()
    validated: list[dict[str, Any]] = []
    for index, raw_frame in enumerate(frames):
        if not isinstance(raw_frame, Mapping):
            raise ValueError(f"episode frame {index} is not an object")
        if int(raw_frame.get("episode_index", -1)) != index:
            raise ValueError("episode_index must be contiguous and ordered")
        source_name = str(raw_frame.get("source_name") or "")
        materialized_name = str(raw_frame.get("materialized_name") or "")
        if not source_name or Path(source_name).name != source_name:
            raise ValueError(f"frame {index} source_name must be a basename")
        expected_materialized = f"{index:06d}__{source_name}"
        if materialized_name != expected_materialized:
            raise ValueError(
                f"frame {index} materialized_name must be {expected_materialized!r}"
            )
        if materialized_name in materialized_names or source_name in source_names:
            raise ValueError("episode contains duplicate frame names")
        materialized_names.add(materialized_name)
        source_names.add(source_name)
        timestamp = str(raw_frame.get("physical_timestamp") or "")
        if not timestamp or timestamp in timestamps:
            raise ValueError("physical timestamps must be non-empty and episode-unique")
        timestamps.add(timestamp)
        image_id = int(raw_frame.get("colmap_image_id", -1))
        camera_id = int(raw_frame.get("colmap_camera_id", -1))
        if image_id not in poses_by_id or image_id not in observations_by_id:
            raise ValueError(
                f"episode frame references missing COLMAP image {image_id}"
            )
        pose = poses_by_id[image_id]
        observations = observations_by_id[image_id]
        if pose_names.get(source_name) != image_id or pose.name != source_name:
            raise ValueError(
                f"episode source name/image ID mismatch: {source_name!r}/{image_id}"
            )
        if observations.name != source_name or observations.camera_id != camera_id:
            raise ValueError(f"POINTS2D metadata mismatch for COLMAP image {image_id}")
        if pose.camera_id != camera_id or camera_id not in cameras:
            raise ValueError(f"camera mismatch for COLMAP image {image_id}")
        camera_ids.add(camera_id)
        declared_pose = np.asarray(raw_frame.get("T_world_camera_m"), dtype=np.float64)
        if declared_pose.shape != (4, 4) or not np.isfinite(declared_pose).all():
            raise ValueError(f"frame {index} has invalid T_world_camera_m")
        if not np.allclose(declared_pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
            raise ValueError(f"frame {index} has invalid homogeneous pose row")
        canonical_pose = camera_to_world_metric(
            pose.camera_from_world, meters_per_scene_unit=meters_per_scene_unit
        )
        translation_error = float(
            np.linalg.norm(declared_pose[:3, 3] - canonical_pose[:3, 3])
        )
        rotation_error = rotation_error_degrees(
            declared_pose[:3, :3], canonical_pose[:3, :3]
        )
        if translation_error > float(maximum_translation_error_m):
            raise ValueError(
                f"frame {index} metric pose translation differs by {translation_error:.6g} m"
            )
        if rotation_error > float(maximum_rotation_error_degrees):
            raise ValueError(
                f"frame {index} metric pose rotation differs by {rotation_error:.6g} degrees"
            )
        camera = cameras[camera_id]
        validated.append(
            {
                "episode_index": index,
                "materialized_name": materialized_name,
                "expected_mask_name": f"{Path(materialized_name).stem}.png",
                "source_name": source_name,
                "physical_timestamp": timestamp,
                "colmap_image_id": image_id,
                "colmap_camera_id": camera_id,
                "width": int(camera.width),
                "height": int(camera.height),
                "pose_translation_error_m": translation_error,
                "pose_rotation_error_degrees": rotation_error,
            }
        )
    if len(camera_ids) != 1:
        raise ValueError(
            "materialized tracker episode must use exactly one COLMAP camera; "
            f"found {sorted(camera_ids)}"
        )
    return validated


def validate_mask_directory(
    mask_dir: Path, frame_rows: Sequence[Mapping[str, Any]]
) -> list[Path]:
    root = Path(mask_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"mask directory is not a directory: {root}")
    expected = [str(row["expected_mask_name"]) for row in frame_rows]
    if len(expected) != len(set(expected)):
        raise ValueError("episode frame names collide after conversion to PNG masks")
    entries = sorted(root.iterdir(), key=lambda value: value.name)
    invalid = [
        value.name
        for value in entries
        if not value.is_file() or value.suffix.lower() != ".png"
    ]
    if invalid:
        raise ValueError(
            "mask directory must contain only PNG files; invalid entries: "
            + ", ".join(invalid[:8])
        )
    actual = [value.name for value in entries]
    if actual != sorted(expected):
        missing = sorted(set(expected).difference(actual))
        unexpected = sorted(set(actual).difference(expected))
        raise ValueError(
            "mask directory does not exactly match episode frames; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    by_name = {path.name: path for path in entries}
    return [by_name[name] for name in expected]


def connected_component_metrics(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("binary mask must have shape [H,W]")
    area = int(binary.sum())
    if area == 0:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_fraction": 0.0,
            "small_component_count": 0,
            "boundary_pixels": 0,
            "boundary_pixel_fraction": 0.0,
            "bbox_xyxy_exclusive": None,
            "bbox_fill_fraction": 0.0,
        }
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for mask component diagnostics") from exc
    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # Label only the tight crop. Automatic trackers can emit dozens of small
    # IDs on a 2048² frame; scanning the full canvas again for every ID is an
    # avoidable O(ids × image_area) cost.
    crop = binary[y0:y1, x0:x1]
    labels, component_count = ndimage.label(
        crop, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(labels.reshape(-1))[1:]
    largest = int(sizes.max(initial=0))
    small_cutoff = max(4, int(math.ceil(0.01 * area)))
    boundary_pixels = int(
        np.count_nonzero(binary[0, :])
        + np.count_nonzero(binary[-1, :])
        + np.count_nonzero(binary[1:-1, 0])
        + np.count_nonzero(binary[1:-1, -1])
    )
    bbox_area = int((x1 - x0) * (y1 - y0))
    return {
        "component_count": int(component_count),
        "largest_component_pixels": largest,
        "largest_component_fraction": float(largest / area),
        "small_component_count": int(np.count_nonzero(sizes < small_cutoff)),
        "boundary_pixels": boundary_pixels,
        "boundary_pixel_fraction": float(boundary_pixels / area),
        "bbox_xyxy_exclusive": [x0, y0, x1, y1],
        "bbox_fill_fraction": float(area / bbox_area),
    }


def _longest_consecutive_run(indices: Sequence[int]) -> int:
    if not indices:
        return 0
    longest = current = 1
    for previous, value in zip(indices, indices[1:]):
        if int(value) == int(previous) + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return int(longest)


def _distribution_entropy(counts: Mapping[int, int]) -> float:
    values = np.asarray(
        [int(value) for value in counts.values() if int(value) > 0], dtype=np.float64
    )
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(values.size))


def _observation_label_summary(
    observations: Sequence[Mapping[str, Any]], *, local_id: int
) -> dict[str, Any]:
    counts = Counter(int(row["sampled_local_id"]) for row in observations)
    return {
        "observation_count": len(observations),
        "local_id_observations": int(counts.get(int(local_id), 0)),
        "background_observations": int(counts.get(0, 0)),
        "competing_foreground_observations": int(
            sum(
                count
                for label, count in counts.items()
                if label not in {0, int(local_id)}
            )
        ),
        "label_counts": {
            str(label): int(count) for label, count in sorted(counts.items())
        },
    }


def point_label_evidence(
    votes_by_point: Mapping[int, Mapping[int, int]],
    *,
    local_id: int,
    observation_provenance_by_point: (
        Mapping[int, Sequence[Mapping[str, Any]]] | None
    ) = None,
    active_frame_indices: Iterable[int] | None = None,
    active_span: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    identity = int(local_id)
    active = frozenset(int(value) for value in (active_frame_indices or ()))
    if active_span is not None and int(active_span[0]) > int(active_span[1]):
        raise ValueError("identity active span must be ordered")
    rows: list[dict[str, Any]] = []
    for point_id in sorted(votes_by_point):
        counts = {
            int(label): int(count)
            for label, count in votes_by_point[point_id].items()
            if int(count) > 0
        }
        local_count = counts.get(identity, 0)
        if local_count <= 0:
            continue
        total = int(sum(counts.values()))
        foreground = {label: count for label, count in counts.items() if label != 0}
        foreground_total = int(sum(foreground.values()))
        maximum = max(counts.values())
        foreground_max = max(foreground.values())
        competitors = {
            str(label): count
            for label, count in sorted(foreground.items())
            if label != identity
        }
        provenance_available = observation_provenance_by_point is not None
        observations: list[dict[str, Any]] = []
        if observation_provenance_by_point is not None:
            raw_observations = observation_provenance_by_point.get(int(point_id), ())
            seen_indices: set[int] = set()
            for raw_observation in sorted(
                raw_observations, key=lambda value: int(value["episode_index"])
            ):
                observation = dict(raw_observation)
                episode_index = int(observation["episode_index"])
                if episode_index in seen_indices:
                    raise ValueError(
                        f"point {point_id} has duplicate observation in episode frame {episode_index}"
                    )
                seen_indices.add(episode_index)
                within_span = active_span is not None and int(
                    active_span[0]
                ) <= episode_index <= int(active_span[1])
                identity_active = episode_index in active
                observation["within_identity_active_span"] = bool(within_span)
                observation["identity_mask_present_in_frame"] = bool(identity_active)
                observation["identity_temporal_scope"] = (
                    "identity_active_frame"
                    if identity_active
                    else (
                        "inactive_gap_inside_active_span"
                        if within_span
                        else "outside_identity_active_span"
                    )
                )
                observations.append(observation)
            observed_counts = Counter(
                int(row["sampled_local_id"]) for row in observations
            )
            if observed_counts != Counter(counts):
                raise ValueError(
                    f"point {point_id} observation provenance does not match aggregate votes"
                )
        within_span_observations = [
            row for row in observations if row["within_identity_active_span"]
        ]
        outside_span_observations = [
            row for row in observations if not row["within_identity_active_span"]
        ]
        active_frame_observations = [
            row for row in observations if row["identity_mask_present_in_frame"]
        ]
        inactive_frame_observations = [
            row for row in observations if not row["identity_mask_present_in_frame"]
        ]
        rows.append(
            {
                "point3d_id": int(point_id),
                "evaluated_local_id": identity,
                "episode_observations": total,
                "local_id_observations": local_count,
                "background_observations": counts.get(0, 0),
                "competing_foreground_observations": int(sum(competitors.values())),
                "competing_foreground_label_counts": competitors,
                "label_counts": {str(k): v for k, v in sorted(counts.items())},
                "local_label_purity": float(local_count / total),
                "local_foreground_purity": float(local_count / foreground_total),
                "normalized_label_entropy": _distribution_entropy(counts),
                "dominant_label_ids": sorted(
                    label for label, count in counts.items() if count == maximum
                ),
                "dominant_foreground_label_ids": sorted(
                    label
                    for label, count in foreground.items()
                    if count == foreground_max
                ),
                "dominant_label_tie": sum(count == maximum for count in counts.values())
                > 1,
                "has_background_disagreement": counts.get(0, 0) > 0,
                "has_competing_foreground_label": bool(competitors),
                "colmap_visibility_provenance": {
                    "available": provenance_available,
                    "visibility_basis": (
                        "registered COLMAP POINTS2D observation; no dense depth occlusion test"
                    ),
                    "observations": observations,
                    "all": _observation_label_summary(observations, local_id=identity),
                    "within_identity_active_span": _observation_label_summary(
                        within_span_observations, local_id=identity
                    ),
                    "outside_identity_active_span": _observation_label_summary(
                        outside_span_observations, local_id=identity
                    ),
                    "identity_active_frames": _observation_label_summary(
                        active_frame_observations, local_id=identity
                    ),
                    "identity_inactive_frames": _observation_label_summary(
                        inactive_frame_observations, local_id=identity
                    ),
                },
            }
        )
    return rows


def build_v2_shadow_evidence(
    rows: Sequence[Mapping[str, Any]], *, local_id: int, policy: GatePolicy
) -> dict[str, Any]:
    """Separate insufficient support from contradictory visible evidence.

    This is diagnostic-only.  Every sparse vote already has COLMAP POINTS2D
    visibility; negative votes are retained even outside the tracker ID active
    span instead of being silently excused as occlusion.
    """

    policy.validate()
    identity = int(local_id)
    state_names = (
        "insufficient_support",
        "supported_consistent",
        "background_dropout",
        "competing_id_conflict",
        "background_and_competing_id_conflict",
        "supported_other_conflict",
    )
    counters = {name: 0 for name in state_names}
    point_states: list[dict[str, Any]] = []
    aggregate_scope = {
        name: Counter()
        for name in (
            "all",
            "within_identity_active_span",
            "outside_identity_active_span",
            "identity_active_frames",
            "identity_inactive_frames",
        )
    }
    for row in rows:
        observation_count = int(row["episode_observations"])
        supported = observation_count >= int(policy.minimum_point_observations)
        consistent = supported and (
            [int(value) for value in row["dominant_label_ids"]] == [identity]
            and float(row["local_label_purity"]) >= policy.minimum_point_label_purity
        )
        background = bool(row["has_background_disagreement"])
        competing = bool(row["has_competing_foreground_label"])
        if not supported:
            state = "insufficient_support"
        elif consistent:
            state = "supported_consistent"
        elif background and competing:
            state = "background_and_competing_id_conflict"
        elif competing:
            state = "competing_id_conflict"
        elif background:
            state = "background_dropout"
        else:
            state = "supported_other_conflict"
        counters[state] += 1
        provenance = row.get("colmap_visibility_provenance", {})
        if isinstance(provenance, Mapping):
            for scope in aggregate_scope:
                summary = provenance.get(scope, {})
                if isinstance(summary, Mapping):
                    for key in (
                        "observation_count",
                        "local_id_observations",
                        "background_observations",
                        "competing_foreground_observations",
                    ):
                        aggregate_scope[scope][key] += int(summary.get(key, 0))
        point_states.append(
            {
                "point3d_id": int(row["point3d_id"]),
                "state": state,
                "supported": supported,
                "visible_conflict": bool(supported and not consistent),
                "episode_observations": observation_count,
                "local_label_purity": float(row["local_label_purity"]),
                "has_background_disagreement": background,
                "has_competing_foreground_label": competing,
            }
        )
    claimed = len(rows)
    insufficient = counters["insufficient_support"]
    supported_count = claimed - insufficient
    consistent_count = counters["supported_consistent"]
    visible_conflict_count = supported_count - consistent_count

    def fraction(numerator: int, denominator: int, *, empty: float) -> float:
        return float(numerator / denominator) if denominator else float(empty)

    return {
        "schema": SHADOW_SCHEMA,
        "mode": "diagnostic-only-no-publish",
        "point_state_definitions": {
            "insufficient_support": "fewer observations than minimum_point_observations; not a contradiction",
            "supported_consistent": "sufficient observations, unique local-ID dominance, and required purity",
            "background_dropout": "supported point fails consistency with background but no competing foreground ID",
            "competing_id_conflict": "supported point fails consistency with a competing foreground ID",
            "background_and_competing_id_conflict": "supported point has both background dropout and competing-ID evidence",
            "supported_other_conflict": "supported point fails dominance/purity without a classified label conflict",
        },
        "point_states": point_states,
        "counts": {
            "claimed_points": claimed,
            "supported_points": supported_count,
            "insufficient_support_points": insufficient,
            "supported_consistent_points": consistent_count,
            "visible_conflict_points": visible_conflict_count,
            **{f"state_{key}": value for key, value in counters.items()},
        },
        "fractions": {
            "supported_point_fraction_of_claimed": fraction(
                supported_count, claimed, empty=0.0
            ),
            "insufficient_support_point_fraction_of_claimed": fraction(
                insufficient, claimed, empty=1.0
            ),
            "supported_consistent_point_fraction_of_claimed": fraction(
                consistent_count, claimed, empty=0.0
            ),
            "visible_conflict_fraction_among_supported": fraction(
                visible_conflict_count, supported_count, empty=1.0
            ),
            "background_dropout_fraction_among_supported": fraction(
                counters["background_dropout"]
                + counters["background_and_competing_id_conflict"],
                supported_count,
                empty=1.0,
            ),
            "competing_id_conflict_fraction_among_supported": fraction(
                counters["competing_id_conflict"]
                + counters["background_and_competing_id_conflict"],
                supported_count,
                empty=1.0,
            ),
        },
        "visibility_and_active_span": {
            "visibility_basis": "registered COLMAP POINTS2D observation",
            "dense_depth_occlusion_test_available": False,
            "negative_votes_outside_active_span_preserved": True,
            "scope_observation_counts": {
                scope: {key: int(value) for key, value in sorted(counts.items())}
                for scope, counts in aggregate_scope.items()
            },
        },
    }


def qualify_point_rows(
    rows: Sequence[Mapping[str, Any]], *, policy: GatePolicy
) -> tuple[list[int], int]:
    """Authoritative v1 point qualification; intentionally unchanged."""

    qualified: list[int] = []
    ambiguous = 0
    for row in rows:
        identity = int(row["evaluated_local_id"])
        dominant = [int(value) for value in row["dominant_label_ids"]]
        failed = (
            dominant != [identity]
            or float(row["local_label_purity"]) < policy.minimum_point_label_purity
            or int(row["episode_observations"]) < policy.minimum_point_observations
        )
        if failed:
            ambiguous += 1
        else:
            qualified.append(int(row["point3d_id"]))
    return qualified, ambiguous


def _connected_components_from_edges(count: int, edges: np.ndarray) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(int(count))]
    for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        if left != right:
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
    seen = np.zeros(int(count), dtype=bool)
    components: list[list[int]] = []
    for start in range(int(count)):
        if seen[start]:
            continue
        seen[start] = True
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            value = queue.popleft()
            component.append(value)
            for neighbor in adjacency[value]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def geometry_signature(point_ids: Sequence[int], xyz_m: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(xyz_m, dtype=np.float64)
    if points.shape != (ids.size, 3):
        raise ValueError("geometry point IDs and xyz must align")
    if ids.size != np.unique(ids).size:
        raise ValueError("geometry point IDs must be unique")
    if not np.isfinite(points).all():
        raise ValueError("geometry xyz must be finite")
    if ids.size == 0:
        return {
            "available": False,
            "point_count": 0,
            "point3d_ids": [],
            "reason": "no_points",
        }
    centroid = points.mean(axis=0)
    median = np.median(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / float(ids.size)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    projected = centered @ eigenvectors
    aabb_min, aabb_max = points.min(axis=0), points.max(axis=0)
    pca_min, pca_max = projected.min(axis=0), projected.max(axis=0)
    q_low, q_high = np.quantile(projected, [0.02, 0.98], axis=0)
    radial = np.linalg.norm(points - median[None, :], axis=1)
    radial_median = float(np.median(radial))
    radial_mad = float(np.median(np.abs(radial - radial_median)))
    robust_limit = radial_median + 3.0 * max(1.4826 * radial_mad, 1.0e-9)
    if ids.size == 1:
        nearest = np.zeros(1, dtype=np.float64)
        radius = 0.0
        edges = np.zeros((0, 2), dtype=np.int64)
    else:
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "scipy is required for 3D compactness diagnostics"
            ) from exc
        tree = cKDTree(points)
        k = min(5, int(ids.size))
        distances, indices = tree.query(points, k=k, workers=1)
        nearest = distances[:, 1]
        finite_nearest = nearest[np.isfinite(nearest)]
        median_nn = float(np.median(finite_nearest)) if finite_nearest.size else 0.0
        q75_nn = (
            float(np.quantile(finite_nearest, 0.75)) if finite_nearest.size else 0.0
        )
        radius = max(3.0 * median_nn, 1.5 * q75_nn, 1.0e-6)
        edge_set: set[tuple[int, int]] = set()
        for left in range(int(ids.size)):
            for distance, right in zip(distances[left, 1:], indices[left, 1:]):
                right = int(right)
                if (
                    0 <= right < ids.size
                    and math.isfinite(float(distance))
                    and float(distance) <= radius
                ):
                    edge_set.add((min(left, right), max(left, right)))
        edges = (
            np.asarray(sorted(edge_set), dtype=np.int64).reshape(-1, 2)
            if edge_set
            else np.zeros((0, 2), dtype=np.int64)
        )
    components = _connected_components_from_edges(int(ids.size), edges)
    components.sort(key=lambda values: (-len(values), min(values)))
    component_sizes = [len(values) for values in components]
    largest_ids = sorted(int(ids[index]) for index in components[0])
    nearest_finite = nearest[np.isfinite(nearest)]
    aabb_extent = aabb_max - aabb_min
    pca_extent = pca_max - pca_min
    robust_extent = q_high - q_low
    robust_volume = float(np.prod(np.maximum(robust_extent, 1.0e-9)))
    return {
        "available": True,
        "point_count": int(ids.size),
        "point3d_ids": [int(value) for value in ids.tolist()],
        "centroid_m": centroid.tolist(),
        "coordinate_median_m": median.tolist(),
        "covariance_m2": covariance.tolist(),
        "covariance_eigenvalues_m2": eigenvalues.tolist(),
        "principal_axes_columns": eigenvectors.tolist(),
        "aabb_min_m": aabb_min.tolist(),
        "aabb_max_m": aabb_max.tolist(),
        "aabb_extents_m": aabb_extent.tolist(),
        "aabb_diagonal_m": float(np.linalg.norm(aabb_extent)),
        "pca_extents_m": pca_extent.tolist(),
        "robust_pca_extents_q02_q98_m": robust_extent.tolist(),
        "radial_distance_median_m": radial_median,
        "radial_distance_q95_m": float(np.quantile(radial, 0.95)),
        "radial_outlier_fraction_median_plus_3mad": float(
            np.count_nonzero(radial > robust_limit) / ids.size
        ),
        "robust_density_points_per_m3": float(ids.size / robust_volume),
        "connectivity": {
            "method": "undirected-knn4-with-robust-adaptive-distance-cap",
            "distance_cap_rule": "max(3*median_nn, 1.5*q75_nn, 1e-6m)",
            "distance_cap_m": float(radius),
            "nearest_neighbor_median_m": (
                float(np.median(nearest_finite)) if nearest_finite.size else 0.0
            ),
            "nearest_neighbor_q75_m": (
                float(np.quantile(nearest_finite, 0.75)) if nearest_finite.size else 0.0
            ),
            "nearest_neighbor_q90_m": (
                float(np.quantile(nearest_finite, 0.90)) if nearest_finite.size else 0.0
            ),
            "edge_count": int(edges.shape[0]),
            "component_count": len(components),
            "component_sizes": component_sizes,
            "largest_component_points": component_sizes[0],
            "largest_component_fraction": float(component_sizes[0] / ids.size),
            "largest_component_point3d_ids": largest_ids,
            "isolated_point_fraction": float(
                sum(size == 1 for size in component_sizes) / ids.size
            ),
        },
    }


def _summary(values: Sequence[float | int]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"minimum": None, "median": None, "maximum": None, "mean": None}
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def collect_raw_evidence(
    *,
    episode_id: str,
    frame_rows: Sequence[Mapping[str, Any]],
    masks: Sequence[np.ndarray],
    observations_by_id: Mapping[int, ColmapImageObservations],
    maximum_local_id: int = MAX_LOCAL_ID,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[int, Counter[int]],
    dict[int, list[dict[str, Any]]],
]:
    if len(frame_rows) != len(masks):
        raise ValueError("frame rows and masks must align")
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    observation_provenance: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_identity_frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_diagnostics: list[dict[str, Any]] = []
    total_sparse = valid_sparse = sampled_sparse = background_sparse = 0
    foreground_pixels_total = pixels_total = 0
    for frame_row, raw_mask in zip(frame_rows, masks):
        mask = np.asarray(raw_mask)
        if mask.dtype != np.uint8 or mask.ndim != 2:
            raise ValueError("tracker masks must be 2D uint8 arrays")
        expected_shape = (int(frame_row["height"]), int(frame_row["width"]))
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask {frame_row['expected_mask_name']} shape {mask.shape} "
                f"does not match COLMAP camera {expected_shape}"
            )
        maximum = int(mask.max(initial=0))
        if maximum > int(maximum_local_id):
            raise ValueError(
                f"mask {frame_row['expected_mask_name']} uses reserved local ID {maximum}"
            )
        labels, counts = np.unique(mask, return_counts=True)
        background_pixels = int(counts[labels == 0].sum()) if np.any(labels == 0) else 0
        foreground_pixels = int(mask.size - background_pixels)
        foreground_pixels_total += foreground_pixels
        pixels_total += int(mask.size)
        observations = observations_by_id[int(frame_row["colmap_image_id"])]
        total_sparse += int(observations.point3d_ids.size)
        valid_ids = observations.point3d_ids >= 0
        finite = np.isfinite(observations.xy).all(axis=1)
        in_bounds = (
            (observations.xy[:, 0] >= 0.0)
            & (observations.xy[:, 0] < mask.shape[1])
            & (observations.xy[:, 1] >= 0.0)
            & (observations.xy[:, 1] < mask.shape[0])
        )
        sampleable = valid_ids & finite & in_bounds
        valid_sparse += int(np.count_nonzero(valid_ids))
        sampled_sparse += int(np.count_nonzero(sampleable))
        point_ids = observations.point3d_ids[sampleable]
        point_xy = observations.xy[sampleable]
        xs = np.clip(
            np.rint(point_xy[:, 0]).astype(np.int64),
            0,
            mask.shape[1] - 1,
        )
        ys = np.clip(
            np.rint(point_xy[:, 1]).astype(np.int64),
            0,
            mask.shape[0] - 1,
        )
        sampled_labels = mask[ys, xs].astype(np.int64, copy=False)
        if point_ids.size != np.unique(point_ids).size:
            raise ValueError(
                f"COLMAP image {observations.image_id} contains duplicate valid point3D IDs"
            )
        for point_id, xy, x, y, label in zip(
            point_ids.tolist(),
            point_xy.tolist(),
            xs.tolist(),
            ys.tolist(),
            sampled_labels.tolist(),
        ):
            point_id = int(point_id)
            label = int(label)
            votes[point_id][label] += 1
            observation_provenance[point_id].append(
                {
                    "episode_index": int(frame_row["episode_index"]),
                    "physical_timestamp": str(frame_row["physical_timestamp"]),
                    "source_name": str(frame_row["source_name"]),
                    "colmap_image_id": int(frame_row["colmap_image_id"]),
                    "point2d_xy_px": [float(xy[0]), float(xy[1])],
                    "sampled_pixel_xy": [int(x), int(y)],
                    "sampled_local_id": label,
                }
            )
        background_sparse += int(np.count_nonzero(sampled_labels == 0))
        identity_frame_rows: list[dict[str, Any]] = []
        for label, area in zip(labels.tolist(), counts.tolist()):
            local_id = int(label)
            if local_id == 0:
                continue
            components = connected_component_metrics(mask == local_id)
            row = {
                "episode_index": int(frame_row["episode_index"]),
                "physical_timestamp": str(frame_row["physical_timestamp"]),
                "area_px": int(area),
                "area_fraction": float(area / mask.size),
                **components,
            }
            per_identity_frames[local_id].append(row)
            identity_frame_rows.append(
                {
                    "local_id": local_id,
                    "area_px": int(area),
                    "component_count": components["component_count"],
                    "largest_component_fraction": components[
                        "largest_component_fraction"
                    ],
                }
            )
        frame_diagnostics.append(
            {
                "episode_index": int(frame_row["episode_index"]),
                "materialized_name": str(frame_row["materialized_name"]),
                "mask_name": str(frame_row["expected_mask_name"]),
                "source_name": str(frame_row["source_name"]),
                "physical_timestamp": str(frame_row["physical_timestamp"]),
                "colmap_image_id": int(frame_row["colmap_image_id"]),
                "shape_hw": [int(mask.shape[0]), int(mask.shape[1])],
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": float(foreground_pixels / mask.size),
                "local_identity_count": len(identity_frame_rows),
                "local_identities": identity_frame_rows,
                "colmap_points2d": int(observations.point3d_ids.size),
                "valid_point3d_observations": int(np.count_nonzero(valid_ids)),
                "sampled_in_bounds_point3d_observations": int(
                    np.count_nonzero(sampleable)
                ),
                "invalid_or_unregistered_observations": int(
                    np.count_nonzero(~valid_ids)
                ),
                "nonfinite_or_out_of_bounds_valid_observations": int(
                    np.count_nonzero(valid_ids & ~sampleable)
                ),
                "foreground_sparse_observations": int(
                    np.count_nonzero(sampled_labels != 0)
                ),
            }
        )
    identity_rows: list[dict[str, Any]] = []
    episode_frames = len(frame_rows)
    for local_id in sorted(per_identity_frames):
        rows = per_identity_frames[local_id]
        indices = [int(row["episode_index"]) for row in rows]
        first, last = indices[0], indices[-1]
        span = last - first + 1
        identity_rows.append(
            {
                "identity_key": f"{episode_id}::local:{local_id}",
                "episode_id": episode_id,
                "local_id": local_id,
                "identity_scope": "episode-local-only",
                "frame_support": len(rows),
                "physical_timestamp_support": len(
                    {str(row["physical_timestamp"]) for row in rows}
                ),
                "first_episode_index": first,
                "last_episode_index": last,
                "temporal_span_frames": span,
                "persistence_ratio": float(len(rows) / episode_frames),
                "within_span_persistence_ratio": float(len(rows) / span),
                "longest_consecutive_run_frames": _longest_consecutive_run(indices),
                "gap_count_within_span": int(span - len(rows)),
                "total_mask_pixels": int(sum(int(row["area_px"]) for row in rows)),
                "area_px": _summary([int(row["area_px"]) for row in rows]),
                "area_fraction": _summary(
                    [float(row["area_fraction"]) for row in rows]
                ),
                "component_count": _summary(
                    [int(row["component_count"]) for row in rows]
                ),
                "largest_2d_component_fraction": _summary(
                    [float(row["largest_component_fraction"]) for row in rows]
                ),
                "boundary_pixel_fraction": _summary(
                    [float(row["boundary_pixel_fraction"]) for row in rows]
                ),
                "bbox_fill_fraction": _summary(
                    [float(row["bbox_fill_fraction"]) for row in rows]
                ),
                "frame_observations": rows,
            }
        )
    episode_summary = {
        "frame_count": episode_frames,
        "local_identity_count": len(identity_rows),
        "total_pixels": pixels_total,
        "foreground_pixels": foreground_pixels_total,
        "foreground_fraction": float(foreground_pixels_total / max(pixels_total, 1)),
        "colmap_points2d_observations": total_sparse,
        "valid_point3d_observations": valid_sparse,
        "sampled_in_bounds_point3d_observations": sampled_sparse,
        "foreground_sparse_observations": int(sampled_sparse - background_sparse),
        "background_sparse_observations": background_sparse,
        "unique_sampled_point3d_ids": len(votes),
        "frames": frame_diagnostics,
    }
    return episode_summary, identity_rows, votes, dict(observation_provenance)


def _gate_check(value: Any, operator: str, threshold: Any) -> dict[str, Any]:
    if operator == ">=":
        passed = value is not None and float(value) >= float(threshold)
    elif operator == "<=":
        passed = value is not None and float(value) <= float(threshold)
    elif operator == "==":
        passed = value == threshold
    else:  # pragma: no cover
        raise ValueError(operator)
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def finalize_identity(
    raw_mask_metrics: Mapping[str, Any],
    *,
    votes_by_point: Mapping[int, Mapping[int, int]],
    observation_provenance_by_point: (
        Mapping[int, Sequence[Mapping[str, Any]]] | None
    ) = None,
    points3d: Mapping[int, SparsePoint3D],
    points3d_file_available: bool,
    meters_per_scene_unit: float,
    policy: GatePolicy,
) -> dict[str, Any]:
    policy.validate()
    local_id = int(raw_mask_metrics["local_id"])
    active_frame_indices = [
        int(row["episode_index"])
        for row in raw_mask_metrics.get("frame_observations", [])
    ]
    point_rows = point_label_evidence(
        votes_by_point,
        local_id=local_id,
        observation_provenance_by_point=observation_provenance_by_point,
        active_frame_indices=active_frame_indices,
        active_span=(
            int(raw_mask_metrics["first_episode_index"]),
            int(raw_mask_metrics["last_episode_index"]),
        ),
    )
    shadow_evidence = build_v2_shadow_evidence(
        point_rows, local_id=local_id, policy=policy
    )
    claimed_ids = [int(row["point3d_id"]) for row in point_rows]
    qualified_ids, ambiguous_count = qualify_point_rows(point_rows, policy=policy)
    claimed_with_xyz = [value for value in claimed_ids if value in points3d]
    qualified_with_xyz = [value for value in qualified_ids if value in points3d]

    def signature(ids: Sequence[int]) -> dict[str, Any]:
        xyz = np.asarray(
            [points3d[value].xyz_scene for value in ids], dtype=np.float64
        ).reshape(-1, 3)
        xyz *= float(meters_per_scene_unit)
        return geometry_signature(ids, xyz)

    for row in point_rows:
        point = points3d.get(int(row["point3d_id"]))
        row["xyz_available"] = point is not None
        if point is not None:
            row["xyz_m"] = (
                np.asarray(point.xyz_scene) * float(meters_per_scene_unit)
            ).tolist()
            row["source_reprojection_error_px"] = float(point.reprojection_error_px)
            row["source_track_length"] = int(point.track_length)
    all_geometry = signature(claimed_with_xyz)
    qualified_geometry = signature(qualified_with_xyz)
    sparse_observations = int(
        sum(int(row["local_id_observations"]) for row in point_rows)
    )
    claimed_count, qualified_count = len(claimed_ids), len(qualified_ids)
    ambiguous_fraction = (
        float(ambiguous_count / claimed_count) if claimed_count else 1.0
    )
    qualified_fraction = (
        float(qualified_count / claimed_count) if claimed_count else 0.0
    )
    frame_observations = list(raw_mask_metrics["frame_observations"])
    tiny_count = sum(
        int(row["area_px"]) < int(policy.tiny_area_px) for row in frame_observations
    )
    tiny_fraction = (
        float(tiny_count / len(frame_observations)) if frame_observations else 1.0
    )
    largest_3d_fraction = (
        qualified_geometry["connectivity"]["largest_component_fraction"]
        if qualified_geometry.get("available")
        else None
    )
    checks = {
        "frame_support": _gate_check(
            raw_mask_metrics["frame_support"], ">=", policy.minimum_frame_support
        ),
        "physical_timestamp_support": _gate_check(
            raw_mask_metrics["physical_timestamp_support"],
            ">=",
            policy.minimum_physical_timestamps,
        ),
        "persistence_ratio": _gate_check(
            raw_mask_metrics["persistence_ratio"],
            ">=",
            policy.minimum_persistence_ratio,
        ),
        "median_area_px": _gate_check(
            raw_mask_metrics["area_px"]["median"], ">=", policy.minimum_median_area_px
        ),
        "tiny_frame_fraction": _gate_check(
            tiny_fraction, "<=", policy.maximum_tiny_frame_fraction
        ),
        "median_largest_2d_component_fraction": _gate_check(
            raw_mask_metrics["largest_2d_component_fraction"]["median"],
            ">=",
            policy.minimum_median_largest_2d_component_fraction,
        ),
        "sparse_observations": _gate_check(
            sparse_observations, ">=", policy.minimum_sparse_observations
        ),
        "unique_sparse_points": _gate_check(
            claimed_count, ">=", policy.minimum_unique_sparse_points
        ),
        "qualified_sparse_points": _gate_check(
            qualified_count, ">=", policy.minimum_qualified_sparse_points
        ),
        "qualified_sparse_point_fraction": _gate_check(
            qualified_fraction, ">=", policy.minimum_qualified_sparse_point_fraction
        ),
        "ambiguous_sparse_point_fraction": _gate_check(
            ambiguous_fraction, "<=", policy.maximum_ambiguous_sparse_point_fraction
        ),
        "points3d_available": (
            _gate_check(bool(points3d_file_available), "==", True)
            if policy.require_points3d
            else _gate_check(True, "==", True)
        ),
        "qualified_points_with_xyz": (
            _gate_check(
                len(qualified_with_xyz), ">=", policy.minimum_qualified_sparse_points
            )
            if policy.require_points3d
            else _gate_check(True, "==", True)
        ),
        "largest_3d_component_fraction": (
            _gate_check(
                largest_3d_fraction,
                ">=",
                policy.minimum_largest_3d_component_fraction,
            )
            if policy.require_points3d
            else _gate_check(True, "==", True)
        ),
    }
    failed = [name for name, row in checks.items() if not row["passed"]]
    shadow_checks = {
        name: dict(row)
        for name, row in checks.items()
        if name != "ambiguous_sparse_point_fraction"
    }
    shadow_counts = shadow_evidence["counts"]
    shadow_fractions = shadow_evidence["fractions"]
    shadow_checks.update(
        {
            "supported_sparse_points": _gate_check(
                shadow_counts["supported_points"],
                ">=",
                policy.minimum_qualified_sparse_points,
            ),
            "supported_sparse_point_fraction": _gate_check(
                shadow_fractions["supported_point_fraction_of_claimed"],
                ">=",
                policy.minimum_qualified_sparse_point_fraction,
            ),
            "visible_conflict_fraction_among_supported": _gate_check(
                shadow_fractions["visible_conflict_fraction_among_supported"],
                "<=",
                policy.maximum_ambiguous_sparse_point_fraction,
            ),
        }
    )
    shadow_failed = [name for name, row in shadow_checks.items() if not row["passed"]]
    raw = dict(raw_mask_metrics)
    raw["sparse_evidence"] = {
        "observation_count": sparse_observations,
        "unique_sparse_point_count": claimed_count,
        "unique_sparse_point3d_ids": claimed_ids,
        "point_label_evidence": point_rows,
    }
    raw["geometry"] = {
        "coordinate_unit": "meter",
        "meters_per_scene_unit": float(meters_per_scene_unit),
        "claimed_points_with_xyz": len(claimed_with_xyz),
        "claimed_points_missing_xyz": sorted(set(claimed_ids).difference(points3d)),
        "all_claimed_points": all_geometry,
    }
    return {
        "identity_key": raw["identity_key"],
        "episode_id": raw["episode_id"],
        "local_id": local_id,
        "identity_scope": "episode-local-only-no-cross-episode-merge",
        "raw": raw,
        "shadow_v2_evidence": shadow_evidence,
        "decision": {
            "status": (
                "accepted_for_cross_episode_association" if not failed else "rejected"
            ),
            "passed": not failed,
            "failed_checks": failed,
            "checks": checks,
            "tiny_frame_count": tiny_count,
            "tiny_frame_fraction": tiny_fraction,
            "ambiguous_sparse_point_count": ambiguous_count,
            "ambiguous_sparse_point_fraction": ambiguous_fraction,
            "qualified_sparse_point_fraction": qualified_fraction,
            "qualified_sparse_point3d_ids": qualified_ids,
            "qualified_points_missing_xyz": sorted(
                set(qualified_ids).difference(points3d)
            ),
            "qualified_geometry": qualified_geometry,
            "note": (
                "Acceptance only permits a later geometric association stage; "
                "it does not create or publish a global FARM object."
            ),
        },
        "decision_v2_shadow": {
            "schema": SHADOW_SCHEMA,
            "mode": "diagnostic-only-no-publish",
            "status": (
                "would_pass_shadow_gate" if not shadow_failed else "shadow_rejected"
            ),
            "passed": not shadow_failed,
            "failed_checks": shadow_failed,
            "checks": shadow_checks,
            "qualified_sparse_point3d_ids": qualified_ids,
            "qualified_geometry": qualified_geometry,
            "publication_effect": "none; local_identities[].decision remains authoritative",
            "global_association_authorized": False,
            "note": (
                "Shadow decision separates insufficient observation support from "
                "contradictory COLMAP-visible label evidence. It cannot publish IDs."
            ),
        },
    }


def evaluate_arrays(
    *,
    episode_id: str,
    frame_rows: Sequence[Mapping[str, Any]],
    masks: Sequence[np.ndarray],
    observations_by_id: Mapping[int, ColmapImageObservations],
    points3d: Mapping[int, SparsePoint3D],
    points3d_file_available: bool,
    meters_per_scene_unit: float,
    policy: GatePolicy,
    maximum_local_id: int = MAX_LOCAL_ID,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_raw, mask_identities, votes, observation_provenance = collect_raw_evidence(
        episode_id=episode_id,
        frame_rows=frame_rows,
        masks=masks,
        observations_by_id=observations_by_id,
        maximum_local_id=maximum_local_id,
    )
    identities = [
        finalize_identity(
            row,
            votes_by_point=votes,
            observation_provenance_by_point=observation_provenance,
            points3d=points3d,
            points3d_file_available=points3d_file_available,
            meters_per_scene_unit=meters_per_scene_unit,
            policy=policy,
        )
        for row in mask_identities
    ]
    return episode_raw, identities


def policy_dict(policy: GatePolicy) -> dict[str, Any]:
    policy.validate()
    return asdict(policy)
