"""Fail-closed canonicalization of duplicate rendered source views.

Only duplicate rows whose camera identity, calibration, RGB and depth content
are equivalent may collapse onto the deterministic first occurrence.  The
module is deliberately torch-free so the contract can be unit-tested and used
by preflight tooling without a GPU environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

CALIBRATION_ATOL = 1.0e-9
DEPTH_ATOL = 1.0e-6
DEPTH_RTOL = 1.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_path(frames_path: Path, row: Mapping[str, Any], key: str) -> Path:
    raw = str(row.get(key) or "").strip()
    if not raw:
        raise ValueError(f"frame {row.get('source_image')!r} has no {key}")
    path = Path(raw)
    if path.is_absolute():
        resolved = path.resolve(strict=True)
    else:
        resolved = (frames_path.parent / path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"frame artifact is not a regular file: {resolved}")
    return resolved


def _timestamp_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    physical = str(row.get("physical_timestamp") or "").strip()
    if physical:
        return "physical_timestamp", str(int(physical)) if physical.isdigit() else physical
    raw_ns = row.get("timestamp_ns")
    if isinstance(raw_ns, bool) or not isinstance(raw_ns, (int, np.integer)):
        raise ValueError(
            f"frame {row.get('source_image')!r} has no physical timestamp identity"
        )
    return "timestamp_ns", str(int(raw_ns))


def _matrix(
    row: Mapping[str, Any], key: str, shape: tuple[int, int]
) -> np.ndarray:
    value = np.asarray(row.get(key), dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(
            f"frame {row.get('source_image')!r} has invalid {key} calibration"
        )
    return value


def _frame_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row.get("source_image") or "").strip()
    frame_id = str(row.get("frame_id") or "").strip()
    camera = str(row.get("camera") or "").strip()
    try:
        depth_size = tuple(int(value) for value in row.get("depth_size") or ())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame {source!r} has invalid depth_size") from exc
    if not source or not frame_id or not camera or len(depth_size) != 2 or min(depth_size) <= 0:
        raise ValueError(f"frame {source!r} has incomplete source identity")
    timestamp_identity = _timestamp_identity(row)
    timestamp_ns = row.get("timestamp_ns")
    if timestamp_ns is not None:
        if isinstance(timestamp_ns, bool) or not isinstance(
            timestamp_ns, (int, np.integer)
        ):
            raise ValueError(f"frame {source!r} has invalid timestamp_ns")
        timestamp_ns = int(timestamp_ns)
    return {
        "source_image": source,
        "frame_id": frame_id,
        "physical_timestamp_identity": timestamp_identity,
        "timestamp_ns": timestamp_ns,
        "camera": camera,
        "depth_size": depth_size,
        "colmap_image_id": row.get("colmap_image_id"),
        "colmap_camera_id": row.get("colmap_camera_id"),
        "K": _matrix(row, "K", (3, 3)),
        "T_world_cam": _matrix(row, "T_world_cam", (4, 4)),
    }


def _decoded_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except Exception as exc:
        raise ValueError(f"cannot decode RGB artifact: {path}") from exc


def _rgb_equivalence(canonical: Path, duplicate: Path) -> dict[str, Any]:
    canonical_sha = _sha256(canonical)
    duplicate_sha = _sha256(duplicate)
    if canonical_sha == duplicate_sha:
        method = "exact_sha256"
    else:
        first = _decoded_rgb(canonical)
        second = _decoded_rgb(duplicate)
        if first.shape != second.shape or not np.array_equal(first, second):
            raise ValueError("duplicate source_image RGB artifacts are not pixel-equivalent")
        method = "decoded_rgb_exact"
    return {
        "method": method,
        "canonical_sha256": canonical_sha,
        "duplicate_sha256": duplicate_sha,
    }


def _load_depth(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot decode depth artifact: {path}") from exc
    if isinstance(value, np.lib.npyio.NpzFile):
        value.close()
        raise ValueError("depth artifact must be a single NPY array")
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError("depth artifact must use a floating dtype")
    return result


def _depth_equivalence(canonical: Path, duplicate: Path) -> dict[str, Any]:
    canonical_sha = _sha256(canonical)
    duplicate_sha = _sha256(duplicate)
    if canonical_sha == duplicate_sha:
        return {
            "method": "exact_sha256",
            "canonical_sha256": canonical_sha,
            "duplicate_sha256": duplicate_sha,
            "maximum_absolute_difference": 0.0,
        }
    first = _load_depth(canonical)
    second = _load_depth(duplicate)
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError("duplicate source_image depth shape/dtype differs")
    first_finite = np.isfinite(first)
    second_finite = np.isfinite(second)
    if not np.array_equal(first_finite, second_finite):
        raise ValueError("duplicate source_image depth finite masks differ")
    if not (
        np.array_equal(np.isnan(first), np.isnan(second))
        and np.array_equal(np.isposinf(first), np.isposinf(second))
        and np.array_equal(np.isneginf(first), np.isneginf(second))
    ):
        raise ValueError("duplicate source_image non-finite depth values differ")
    finite_first = first[first_finite].astype(np.float64, copy=False)
    finite_second = second[first_finite].astype(np.float64, copy=False)
    if not np.allclose(
        finite_first,
        finite_second,
        atol=DEPTH_ATOL,
        rtol=DEPTH_RTOL,
        equal_nan=False,
    ):
        raise ValueError("duplicate source_image depth values differ beyond tolerance")
    maximum = (
        float(np.max(np.abs(finite_first - finite_second)))
        if finite_first.size
        else 0.0
    )
    return {
        "method": "numeric_allclose",
        "canonical_sha256": canonical_sha,
        "duplicate_sha256": duplicate_sha,
        "atol": DEPTH_ATOL,
        "rtol": DEPTH_RTOL,
        "maximum_absolute_difference": maximum,
    }


def _equivalent_duplicate(
    canonical: Mapping[str, Any],
    duplicate: Mapping[str, Any],
    *,
    frames_path: Path,
) -> dict[str, Any]:
    first = _frame_identity(canonical)
    second = _frame_identity(duplicate)
    exact_fields = (
        "source_image",
        "frame_id",
        "physical_timestamp_identity",
        "timestamp_ns",
        "camera",
        "depth_size",
        "colmap_image_id",
        "colmap_camera_id",
    )
    mismatched = [field for field in exact_fields if first[field] != second[field]]
    if mismatched:
        raise ValueError(
            "duplicate source_image identity differs: " + ", ".join(mismatched)
        )
    for field in ("K", "T_world_cam"):
        if not np.allclose(
            first[field], second[field], atol=CALIBRATION_ATOL, rtol=0.0
        ):
            raise ValueError(f"duplicate source_image {field} calibration differs")
    return {
        "calibration_atol": CALIBRATION_ATOL,
        "rgb": _rgb_equivalence(
            _artifact_path(frames_path, canonical, "rgb_path"),
            _artifact_path(frames_path, duplicate, "rgb_path"),
        ),
        "depth": _depth_equivalence(
            _artifact_path(frames_path, canonical, "depth_path"),
            _artifact_path(frames_path, duplicate, "depth_path"),
        ),
    }


def _position_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("source image_position contains non-finite values")
    return result


@dataclass(frozen=True)
class SourceCanonicalization:
    frames: tuple[dict[str, Any], ...]
    canonical_old_indices: tuple[int, ...]
    old_to_new: tuple[int, ...]
    audit: dict[str, Any]


def canonicalize_source_views(
    frames: Sequence[Mapping[str, Any]],
    frames_path: Path,
    image_positions: Sequence[object],
) -> SourceCanonicalization:
    """Collapse only proven-equivalent duplicate source rows."""

    frames_path = frames_path.expanduser().resolve(strict=True)
    if not frames or len(frames) != len(image_positions):
        raise ValueError("source frames/image_positions are empty or not row-aligned")
    canonical_by_source: dict[str, int] = {}
    canonical_old_indices: list[int] = []
    canonical_frames: list[dict[str, Any]] = []
    old_to_new: list[int] = []
    duplicate_rows: list[dict[str, Any]] = []
    for old_index, row in enumerate(frames):
        if not isinstance(row, Mapping):
            raise ValueError("source frames contain a non-object row")
        identity = _frame_identity(row)
        source = identity["source_image"]
        canonical_id = canonical_by_source.get(source)
        if canonical_id is None:
            canonical_id = len(canonical_frames)
            canonical_by_source[source] = canonical_id
            canonical_old_indices.append(old_index)
            canonical_frames.append(copy.deepcopy(dict(row)))
            old_to_new.append(canonical_id)
            continue
        canonical_old = canonical_old_indices[canonical_id]
        equivalence = _equivalent_duplicate(
            frames[canonical_old], row, frames_path=frames_path
        )
        first_position = _position_array(image_positions[canonical_old])
        second_position = _position_array(image_positions[old_index])
        if (first_position is None) != (second_position is None) or (
            first_position is not None
            and (
                first_position.shape != second_position.shape
                or not np.allclose(
                    first_position,
                    second_position,
                    atol=CALIBRATION_ATOL,
                    rtol=0.0,
                )
            )
        ):
            raise ValueError("duplicate source_image image_positions differ")
        old_to_new.append(canonical_id)
        duplicate_rows.append(
            {
                "source_image": source,
                "canonical_old_image_id": canonical_old,
                "duplicate_old_image_id": old_index,
                "canonical_image_id": canonical_id,
                "equivalence": equivalence,
            }
        )
    id_map = [
        {
            "old_image_id": old_id,
            "canonical_image_id": new_id,
            "source_image": str(frames[old_id]["source_image"]),
        }
        for old_id, new_id in enumerate(old_to_new)
    ]
    audit = {
        "schema": "farm.source-view-canonicalization.v1",
        "status": "PASS",
        "policy": "deterministic_first_occurrence_only_if_equivalent",
        "input_rows": len(frames),
        "unique_source_images": len(canonical_frames),
        "duplicate_rows": len(duplicate_rows),
        "calibration_atol": CALIBRATION_ATOL,
        "decoded_rgb_fallback": "exact_pixels",
        "depth_fallback": {
            "finite_mask_exact": True,
            "dtype_exact": True,
            "shape_exact": True,
            "atol": DEPTH_ATOL,
            "rtol": DEPTH_RTOL,
        },
        "source_image_id_map": id_map,
        "source_image_id_map_sha256": _canonical_json_sha256(id_map),
        "duplicates": duplicate_rows,
        "duplicates_sha256": _canonical_json_sha256(duplicate_rows),
    }
    return SourceCanonicalization(
        frames=tuple(canonical_frames),
        canonical_old_indices=tuple(canonical_old_indices),
        old_to_new=tuple(old_to_new),
        audit=audit,
    )

def require_unique_rescue_sources(
    rescue_frames: Sequence[Mapping[str, Any]], source_names: set[str]
) -> None:
    """Reject rescue duplicates; their IDs must remain source_count + local ID."""

    seen: set[str] = set()
    for index, row in enumerate(rescue_frames):
        if not isinstance(row, Mapping):
            raise ValueError("rescue frames contain a non-object row")
        source = str(row.get("source_image") or "").strip()
        if not source:
            raise ValueError(f"rescue frame {index} has no source_image")
        if source in seen:
            raise ValueError(f"rescue frames repeat source_image: {source}")
        if source in source_names:
            raise ValueError(f"rescue source_image already exists in source state: {source}")
        seen.add(source)
