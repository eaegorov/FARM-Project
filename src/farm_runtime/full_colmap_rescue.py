"""Pure geometry and mask helpers for bounded full-COLMAP object rescue.

The rescue path is deliberately category agnostic.  A semantic label may make
an object eligible for review, but image selection and track association use
only camera geometry, metric OBBs, saved masks, and visual features.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch


def numpy_array(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def record_value(record: object, key: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def quaternion_wxyz_to_matrix(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("invalid OBB quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def obb_corners(center_m: object, dimensions_m: object, wxyz: object) -> np.ndarray:
    center = np.asarray(center_m, dtype=np.float64).reshape(3)
    dimensions = np.asarray(dimensions_m, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all() or not np.isfinite(dimensions).all():
        raise ValueError("non-finite OBB")
    if np.any(dimensions <= 0.0):
        raise ValueError("non-positive OBB dimensions")
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = 0.5 * signs * dimensions
    return center[None, :] + local @ quaternion_wxyz_to_matrix(wxyz).T


def project_metric_obb(
    center_m: object,
    dimensions_m: object,
    wxyz: object,
    frame: Mapping[str, Any],
) -> np.ndarray | None:
    """Project a metric OBB into one prepared RGB-D frame as xyxy pixels."""

    corners = obb_corners(center_m, dimensions_m, wxyz)
    camera_to_world = np.asarray(frame.get("T_world_cam"), dtype=np.float64)
    intrinsics = np.asarray(frame.get("K"), dtype=np.float64)
    if camera_to_world.shape != (4, 4) or intrinsics.shape != (3, 3):
        return None
    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError:
        return None
    camera = (world_to_camera @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 1.0e-5):
        return None
    projected = (intrinsics @ camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    return np.r_[pixels.min(axis=0), pixels.max(axis=0)].astype(np.float64)



def project_world_points_seed(
    points_world: np.ndarray,
    frame: Mapping[str, Any],
    depth: np.ndarray,
    *,
    dilation_pixels: int = 5,
    minimum_projected_points: int = 12,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Project metric object support into one prepared COLMAP/RGB-D view."""

    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError("depth must be a 2D metric array")
    canvas = np.zeros(depth_array.shape, dtype=bool)
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    camera_to_world = np.asarray(frame.get("T_world_cam"), dtype=np.float64)
    intrinsics = np.asarray(frame.get("K"), dtype=np.float64)
    if points.size == 0 or camera_to_world.shape != (4, 4) or intrinsics.shape != (3, 3):
        return canvas, {
            "source_points": int(points.shape[0]),
            "projected_points": 0,
            "depth_consistent_points": 0,
            "seed_pixels": 0,
        }
    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError:
        return canvas, {
            "source_points": int(points.shape[0]),
            "projected_points": 0,
            "depth_consistent_points": 0,
            "seed_pixels": 0,
        }
    camera = (
        world_to_camera
        @ np.column_stack((points, np.ones((points.shape[0],), dtype=np.float64))).T
    ).T[:, :3]
    visible = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.05)
    camera = camera[visible]
    if camera.shape[0] == 0:
        return canvas, {
            "source_points": int(points.shape[0]),
            "projected_points": 0,
            "depth_consistent_points": 0,
            "seed_pixels": 0,
        }
    projected = (intrinsics @ camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    xs = np.rint(pixels[:, 0]).astype(np.int64)
    ys = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (
        (xs >= 0) & (xs < depth_array.shape[1])
        & (ys >= 0) & (ys < depth_array.shape[0])
    )
    xs, ys, z = xs[inside], ys[inside], camera[inside, 2]
    projected_count = int(xs.size)
    if projected_count:
        observed = depth_array[ys, xs]
        tolerance = np.maximum(0.12, 0.06 * z)
        depth_consistent = (
            np.isfinite(observed) & (observed > 0.05)
            & (np.abs(observed - z) <= tolerance)
        )
        xs, ys = xs[depth_consistent], ys[depth_consistent]
    if xs.size >= max(1, int(minimum_projected_points)):
        canvas[ys, xs] = True
        radius = max(0, int(dilation_pixels))
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            canvas = cv2.dilate(canvas.astype(np.uint8), kernel, iterations=1).astype(bool)
    return canvas, {
        "source_points": int(points.shape[0]),
        "projected_points": projected_count,
        "depth_consistent_points": int(xs.size),
        "seed_pixels": int(canvas.sum()),
    }

def clip_bbox(bbox: object, width: int, height: int) -> np.ndarray | None:
    values = np.asarray(bbox, dtype=np.float64).reshape(4)
    values[[0, 2]] = np.clip(values[[0, 2]], 0.0, float(width))
    values[[1, 3]] = np.clip(values[[1, 3]], 0.0, float(height))
    if values[2] - values[0] <= 1.0 or values[3] - values[1] <= 1.0:
        return None
    return values


def mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int32)


def bbox_iou(first: object, second: object) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(4)
    b = np.asarray(second, dtype=np.float64).reshape(4)
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = max(0.0, float(hi[0] - lo[0])) * max(0.0, float(hi[1] - lo[1]))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1.0e-12)


def mask_box_metrics(mask: np.ndarray, bbox_xyxy: object) -> dict[str, float]:
    canvas = np.asarray(mask, dtype=bool)
    clipped = clip_bbox(bbox_xyxy, canvas.shape[1], canvas.shape[0])
    actual_bbox = mask_bbox(canvas)
    if clipped is None or actual_bbox is None:
        return {"mask_inside_box": 0.0, "box_pixel_coverage": 0.0, "bbox_iou": 0.0}
    x0, y0, x1, y1 = [int(round(value)) for value in clipped]
    inside_pixels = int(canvas[y0:y1, x0:x1].sum())
    box_area = max((x1 - x0) * (y1 - y0), 1)
    return {
        "mask_inside_box": inside_pixels / max(int(canvas.sum()), 1),
        "box_pixel_coverage": inside_pixels / box_area,
        "bbox_iou": bbox_iou(actual_bbox, clipped),
    }


def unpack_mask_archive(path: Path, kind: str = "raw") -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        selected = kind if f"{kind}_bits" in data.files else "raw"
        required = {
            "image_shape", f"{selected}_bits", f"{selected}_shape",
            f"{selected}_bbox_xyxy",
        }
        if not required.issubset(data.files):
            raise ValueError(f"mask sidecar lacks fields {sorted(required - set(data.files))}: {path}")
        height, width = np.asarray(data["image_shape"], dtype=np.int32).reshape(2)
        crop_height, crop_width = np.asarray(
            data[f"{selected}_shape"], dtype=np.int32
        ).reshape(2)
        bbox = np.asarray(data[f"{selected}_bbox_xyxy"], dtype=np.int32).reshape(4)
        flat = np.unpackbits(
            np.asarray(data[f"{selected}_bits"], dtype=np.uint8), bitorder="little"
        )
    if min(height, width, crop_height, crop_width) <= 0:
        raise ValueError(f"mask sidecar has non-positive shape: {path}")
    if flat.size < int(crop_height * crop_width):
        raise ValueError(f"mask sidecar has truncated bit payload: {path}")
    crop = flat[: int(crop_height * crop_width)].reshape(crop_height, crop_width).astype(bool)
    x0, y0, x1, y1 = [int(value) for value in bbox]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"mask sidecar bbox is outside image: {path}")
    canvas = np.zeros((height, width), dtype=bool)
    canvas[y0:y1, x0:x1] = crop[: y1 - y0, : x1 - x0]
    return canvas


def pack_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canvas = np.asarray(mask, dtype=bool)
    bbox = mask_bbox(canvas)
    if bbox is None:
        raise ValueError("cannot pack an empty mask")
    x0, y0, x1, y1 = [int(value) for value in bbox]
    crop = canvas[y0:y1, x0:x1]
    return (
        np.packbits(crop.reshape(-1), bitorder="little"),
        np.asarray(crop.shape, dtype=np.int32),
        bbox.astype(np.int32),
    )


def distributed_points(mask: np.ndarray, count: int) -> list[list[float]]:
    """Choose spatially separated interior prompt points deterministically."""

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not binary.any() or count <= 0:
        return []
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    radius = max(4, round(math.sqrt(float(binary.sum())) / 11.0))
    result: list[list[float]] = []
    for _ in range(int(count)):
        flat_index = int(np.argmax(distance))
        if float(distance.reshape(-1)[flat_index]) <= 0.0:
            break
        y, x = divmod(flat_index, binary.shape[1])
        result.append([float(x), float(y)])
        cv2.circle(distance, (x, y), radius, 0.0, -1)
    return result


def feature_cosine(first: object, second: object) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)) if denominator > 1.0e-12 else 0.0


def resolve_observation_path(observation: Mapping[str, Any], mask_root: Path) -> Path | None:
    raw = str(observation.get("path") or observation.get("mask_path") or "")
    if not raw:
        return None
    source = Path(raw)
    tail: Path | None = None
    for index, part in enumerate(source.parts):
        if part.startswith("object_"):
            tail = Path(*source.parts[index:])
            break
    if tail is None:
        return None
    root = mask_root.expanduser().resolve()
    candidate = (root / tail).resolve(strict=False)
    if not candidate.is_relative_to(root) or candidate.suffix.lower() != ".npz" or not candidate.is_file():
        return None
    return candidate


def frame_by_state_image(
    state_images: list[object],
    frame_by_rgb_stem: Mapping[str, Mapping[str, Any]],
    image_id: int,
) -> Mapping[str, Any] | None:
    if image_id < 0 or image_id >= len(state_images):
        return None
    record = state_images[image_id]
    source_ref = str(record_value(record, "source_ref", "") or record_value(record, "storage_path", ""))
    return frame_by_rgb_stem.get(Path(source_ref).stem)


def connected_component_overlapping_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    candidate = np.asarray(mask, dtype=bool)
    seed_mask = np.asarray(seed, dtype=bool)
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    if count <= 1:
        return candidate
    best_label = 0
    best_score = -1
    for label in range(1, count):
        component = labels == label
        score = int(np.logical_and(component, seed_mask).sum())
        if score > best_score:
            best_label, best_score = label, score
    return labels == best_label if best_label > 0 and best_score > 0 else np.zeros_like(candidate)


def mask_shape_metrics(mask: np.ndarray) -> dict[str, float | int]:
    """Measure meaningful 2D islands without counting isolated compression noise."""

    canvas = np.asarray(mask, dtype=bool)
    pixels = int(canvas.sum())
    if pixels <= 0:
        return {
            "component_count": 0,
            "significant_component_count": 0,
            "dominant_component_fraction": 0.0,
            "bbox_fill_fraction": 0.0,
        }
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        canvas.astype(np.uint8), connectivity=8
    )
    areas = sorted(
        (int(value) for value in stats[1:, cv2.CC_STAT_AREA]), reverse=True
    )
    significant_floor = max(24, int(math.ceil(0.005 * pixels)))
    ys, xs = np.nonzero(canvas)
    bbox_area = max((int(np.ptp(xs)) + 1) * (int(np.ptp(ys)) + 1), 1)
    return {
        "component_count": max(0, int(count) - 1),
        "significant_component_count": sum(area >= significant_floor for area in areas),
        "dominant_component_fraction": float(areas[0] / pixels),
        "bbox_fill_fraction": float(pixels / bbox_area),
    }


def backproject_mask_points(mask: np.ndarray, depth: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray:
    canvas = np.asarray(mask, dtype=bool)
    depth_array = np.asarray(depth, dtype=np.float32)
    if canvas.shape != depth_array.shape:
        raise ValueError("mask/depth shapes differ")
    ys, xs = np.nonzero(canvas)
    z = depth_array[ys, xs]
    valid = np.isfinite(z) & (z > 0.05) & (z < 80.0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    intrinsics = np.asarray(frame.get("K"), dtype=np.float64)
    pose = np.asarray(frame.get("T_world_cam"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("invalid frame camera geometry")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
    return (camera @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def points_inside_obb(
    points: np.ndarray,
    center_m: object,
    dimensions_m: object,
    wxyz: object,
    *,
    expansion: float = 1.0,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_m, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(dimensions_m, dtype=np.float64).reshape(3) * float(expansion)
    rotation = quaternion_wxyz_to_matrix(wxyz)
    local = (values - center) @ rotation
    return np.all(np.abs(local) <= half + 1.0e-6, axis=1)




def voxel_downsample_points(
    points: np.ndarray,
    *,
    voxel_size_m: float = 0.015,
    max_points: int = 20_000,
) -> np.ndarray:
    """Deterministically retain one finite point per metric voxel.

    Full-COLMAP refinement can backproject hundreds of thousands of mask pixels.
    The SAM prompt only needs a bounded, spatially representative surface seed;
    keeping the first source-ordered point per voxel makes the result stable and
    prevents view resolution from dominating later views.
    """

    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if voxel_size_m <= 0.0 or max_points < 1:
        raise ValueError("voxel_size_m and max_points must be positive")
    values = values[np.isfinite(values).all(axis=1)]
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    keys = np.floor(values.astype(np.float64) / float(voxel_size_m)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    selected = values[np.sort(first)]
    if selected.shape[0] > int(max_points):
        indices = np.linspace(0, selected.shape[0] - 1, int(max_points), dtype=np.int64)
        selected = selected[indices]
    return np.ascontiguousarray(selected, dtype=np.float32)


def multiview_voxel_consensus(
    point_sets: Iterable[np.ndarray],
    *,
    voxel_size_m: float = 0.025,
    minimum_views: int = 2,
    max_points: int = 20_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep only metric voxels observed by independent camera views.

    One point per view contributes to a voxel, so image resolution and repeated
    pixels cannot fake multiview support.  Returned points are per-voxel means
    in deterministic first-observation order.
    """

    if voxel_size_m <= 0.0 or minimum_views < 1 or max_points < 1:
        raise ValueError("voxel size, minimum views and max points must be positive")
    rows = list(point_sets)
    accumulators: dict[tuple[int, int, int], list[Any]] = {}
    per_view_points: list[int] = []
    finite_points = 0
    for view_index, raw in enumerate(rows):
        values = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
        values = values[np.isfinite(values).all(axis=1)]
        finite_points += int(values.shape[0])
        if values.size == 0:
            per_view_points.append(0)
            continue
        keys = np.floor(
            values.astype(np.float64) / float(voxel_size_m)
        ).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        first = np.sort(first)
        per_view_points.append(int(first.size))
        for index in first.tolist():
            key = tuple(int(value) for value in keys[index])
            if key not in accumulators:
                accumulators[key] = [values[index].astype(np.float64), 1, view_index]
            else:
                accumulators[key][0] += values[index]
                accumulators[key][1] += 1
    selected = [
        (total / count).astype(np.float32)
        for total, count, _ in accumulators.values()
        if int(count) >= int(minimum_views)
    ]
    support = (
        np.ascontiguousarray(np.stack(selected), dtype=np.float32)
        if selected else np.zeros((0, 3), dtype=np.float32)
    )
    before_limit = int(support.shape[0])
    if before_limit > int(max_points):
        indices = np.linspace(0, before_limit - 1, int(max_points), dtype=np.int64)
        support = np.ascontiguousarray(support[indices], dtype=np.float32)
    return support, {
        "input_views": len(rows),
        "minimum_views": int(minimum_views),
        "finite_input_points": int(finite_points),
        "per_view_unique_voxels": per_view_points,
        "occupied_voxels": len(accumulators),
        "consensus_voxels_before_limit": before_limit,
        "support_points": int(support.shape[0]),
        "voxel_size_m": float(voxel_size_m),
    }



def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

