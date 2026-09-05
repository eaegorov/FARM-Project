#!/usr/bin/env python3
"""Fail-closed contracts shared by the versioned FARM Gaussian lift."""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from farm_runtime.integrity import (
    DirectoryIntegrityError,
    verify_directory_tree_descriptor,
)


UNKNOWN_ID = -1
CONFIG_SCHEMA = "farm.gaussian-lift.config.v1"
OUTPUT_FIELDS = (
    "farm_instance_id",
    "farm_instance_confidence",
    "farm_timestamp_support",
    "red",
    "green",
    "blue",
)
SH_C0 = 0.28209479177387814
PLY_SCALARS: dict[str, str] = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, np.asarray(array), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **{name: np.asarray(value) for name, value in arrays.items()})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def preflight_sampled_sha256(path: Path, chunk_bytes: int) -> str:
    size = path.stat().st_size
    span = min(int(chunk_bytes), size)
    offsets = sorted({0, max(0, (size - span) // 2), max(0, size - span)})
    digest = hashlib.sha256()
    digest.update(b"farm-sampled-sha256-v1\0")
    digest.update(struct.pack("<Q", size))
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            payload = stream.read(span)
            digest.update(struct.pack("<QQ", offset, len(payload)))
            digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class PlyVertexTable:
    path: Path
    count: int
    header_bytes: int
    header_lines: tuple[str, ...]
    property_tokens: tuple[tuple[str, str], ...]
    dtype: np.dtype
    data: np.memmap

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack((self.data["x"], self.data["y"], self.data["z"])).astype(
            np.float32, copy=False
        )


def open_graphdeco_ply(path: Path) -> PlyVertexTable:
    path = path.expanduser().resolve(strict=True)
    with path.open("rb") as stream:
        lines: list[str] = []
        while True:
            raw = stream.readline()
            if not raw or stream.tell() > (1 << 20):
                raise ValueError(f"truncated or oversized PLY header: {path}")
            try:
                line = raw.decode("ascii").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise ValueError(f"non-ASCII PLY header: {path}") from exc
            lines.append(line)
            if line == "end_header":
                break
        header_bytes = stream.tell()
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("only binary_little_endian PLY is supported")

    vertex_count: int | None = None
    current_element: str | None = None
    properties: list[tuple[str, str]] = []
    seen_elements: list[str] = []
    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"]:
            if len(parts) != 3:
                raise ValueError(f"invalid PLY element declaration: {line}")
            current_element = parts[1]
            seen_elements.append(current_element)
            if current_element != "vertex":
                raise ValueError(f"minimal lift supports vertex-only PLY, found {current_element!r}")
            if vertex_count is not None:
                raise ValueError("duplicate vertex element")
            vertex_count = int(parts[2])
        elif parts[:1] == ["property"]:
            if current_element != "vertex":
                raise ValueError("property outside vertex element")
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("list/non-scalar PLY properties are unsupported")
            token, name = parts[1], parts[2]
            if token not in PLY_SCALARS:
                raise ValueError(f"unsupported PLY scalar {token!r}")
            if name in {existing for _, existing in properties}:
                raise ValueError(f"duplicate PLY property {name!r}")
            properties.append((token, name))
    if seen_elements != ["vertex"] or vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY must contain one non-empty vertex element")
    dtype = np.dtype([(name, PLY_SCALARS[token]) for token, name in properties], align=False)
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - set(dtype.names or ()))
    if missing:
        raise ValueError(f"3DGS PLY misses required properties: {missing}")
    expected = header_bytes + vertex_count * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(
            f"vertex-only PLY size mismatch: actual={path.stat().st_size}, expected={expected}"
        )
    data = np.memmap(path, dtype=dtype, mode="r", offset=header_bytes, shape=(vertex_count,))
    return PlyVertexTable(
        path=path,
        count=vertex_count,
        header_bytes=header_bytes,
        header_lines=tuple(lines),
        property_tokens=tuple(properties),
        dtype=dtype,
        data=data,
    )


def sh_dc_to_rgb(dc: np.ndarray) -> np.ndarray:
    rgb = np.clip(0.5 + SH_C0 * np.asarray(dc, dtype=np.float32), 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


def _instance_colour(object_id: int) -> np.ndarray:
    hue = (int(object_id) * 0.6180339887498949 + 0.11) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.96)
    return np.rint(np.asarray(rgb) * 255.0).astype(np.uint8)


def write_full_labeled_ply(
    source: PlyVertexTable,
    output: Path,
    object_id: np.ndarray,
    confidence: np.ndarray,
    timestamp_support: np.ndarray,
    *,
    chunk_rows: int = 131_072,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output == source.path:
        raise ValueError("labeled output must differ from immutable source PLY")
    existing = set(source.dtype.names or ())
    collisions = sorted(existing.intersection(OUTPUT_FIELDS))
    if collisions:
        raise ValueError(f"source PLY already contains reserved lift properties: {collisions}")
    labels = np.asarray(object_id, dtype="<i4")
    conf = np.asarray(confidence, dtype="<f4")
    support = np.asarray(timestamp_support, dtype="<u2")
    if labels.shape != (source.count,) or conf.shape != (source.count,) or support.shape != (source.count,):
        raise ValueError("dense label arrays must exactly match source vertex count")
    if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence must be finite in [0,1]")
    if np.any((labels < UNKNOWN_ID)):
        raise ValueError("instance IDs below -1 are forbidden")

    appended = [
        ("farm_instance_id", "<i4"),
        ("farm_instance_confidence", "<f4"),
        ("farm_timestamp_support", "<u2"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    dtype = np.dtype(list(source.dtype.descr) + appended, align=False)
    if dtype.itemsize != source.dtype.itemsize + 13:
        raise AssertionError("unexpected labeled PLY row size")
    for name in source.dtype.names or ():
        if dtype.fields[name][1] != source.dtype.fields[name][1]:
            raise AssertionError(f"original PLY offset changed for {name}")

    palette_ids = np.unique(labels[labels >= 0])
    palette = (
        np.stack([_instance_colour(int(value)) for value in palette_ids])
        if len(palette_ids) else np.zeros((0, 3), dtype=np.uint8)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    source_stat_before = source.path.stat()
    original_body_hash = hashlib.sha256()
    copied_body_hash = hashlib.sha256()
    try:
        header = list(source.header_lines[:-1])
        header.extend([
            "property int farm_instance_id",
            "property float farm_instance_confidence",
            "property ushort farm_timestamp_support",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ])
        with temporary.open("wb") as stream:
            stream.write(("\n".join(header) + "\n").encode("ascii"))
            for start in range(0, source.count, int(chunk_rows)):
                end = min(start + int(chunk_rows), source.count)
                count = end - start
                rows = np.empty(count, dtype=dtype)
                source_bytes = source.data[start:end].view(np.uint8).reshape(count, source.dtype.itemsize)
                target_bytes = rows.view(np.uint8).reshape(count, dtype.itemsize)
                target_bytes[:, : source.dtype.itemsize] = source_bytes
                original_body_hash.update(source_bytes.tobytes(order="C"))
                copied_body_hash.update(target_bytes[:, : source.dtype.itemsize].tobytes(order="C"))
                rows["farm_instance_id"] = labels[start:end]
                rows["farm_instance_confidence"] = conf[start:end]
                rows["farm_timestamp_support"] = support[start:end]
                dc = np.column_stack((
                    source.data["f_dc_0"][start:end],
                    source.data["f_dc_1"][start:end],
                    source.data["f_dc_2"][start:end],
                ))
                colours = sh_dc_to_rgb(dc)
                chunk_labels = labels[start:end]
                labelled = chunk_labels >= 0
                if np.any(labelled):
                    palette_index = np.searchsorted(palette_ids, chunk_labels[labelled])
                    colours[labelled] = palette[palette_index]
                rows["red"], rows["green"], rows["blue"] = (
                    colours[:, 0], colours[:, 1], colours[:, 2]
                )
                rows.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if original_body_hash.digest() != copied_body_hash.digest():
            raise RuntimeError("original PLY row bytes changed while constructing labeled output")
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    source_stat_after = source.path.stat()
    if (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
    ):
        raise RuntimeError("immutable source PLY stat changed during export")
    labeled = open_graphdeco_ply(output)
    if labeled.count != source.count or tuple(labeled.dtype.names or ()) != tuple(dtype.names or ()):
        raise RuntimeError("labeled PLY readback schema/count mismatch")
    return {
        "schema_version": "farm.gaussian-lift.labeled-ply.v1",
        "source_path": str(source.path),
        "source_bytes": int(source_stat_after.st_size),
        "source_body_sha256": original_body_hash.hexdigest(),
        "output_path": str(output),
        "output_bytes": int(output.stat().st_size),
        "output_sha256": sha256_file(output),
        "vertex_count": source.count,
        "original_properties": list(source.dtype.names or ()),
        "appended_properties": list(OUTPUT_FIELDS),
        "original_row_bytes": source.dtype.itemsize,
        "output_row_bytes": dtype.itemsize,
        "original_fields_bitwise_preserved": True,
        "source_order_preserved": True,
        "rows_deleted": 0,
    }


def metric_c2w_to_scene_w2c(T_world_cam: np.ndarray, meters_per_scene_unit: float) -> np.ndarray:
    pose = np.asarray(T_world_cam, dtype=np.float64).copy()
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("T_world_cam must be one finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("T_world_cam has invalid homogeneous row")
    scale = float(meters_per_scene_unit)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("meters_per_scene_unit must be finite and positive")
    pose[:3, 3] /= scale
    return np.linalg.inv(pose).astype(np.float32)


def project_metric(
    points_world_m: np.ndarray, T_world_cam_m: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world_m, dtype=np.float64)
    pose = np.asarray(T_world_cam_m, dtype=np.float64)
    intrinsic = np.asarray(K, dtype=np.float64)
    rotation, translation = pose[:3, :3], pose[:3, 3]
    camera = (points - translation) @ rotation
    z = camera[:, 2]
    safe = np.maximum(z, 1e-8)
    u = intrinsic[0, 0] * camera[:, 0] / safe + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / safe + intrinsic[1, 2]
    return u, v, z


def wxyz_to_matrix(value: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(list(value), dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


@dataclass(frozen=True)
class Frame:
    image_id: int
    frame_id: str
    timestamp_ns: int
    camera: str
    sensor: str
    family: str
    source_image: str
    depth_size: tuple[int, int]
    K: np.ndarray
    T_world_cam: np.ndarray
    rgb_path: Path
    depth_path: Path

    @property
    def physical_timestamp(self) -> str:
        """Canonical cross-camera split key (decimal nanoseconds)."""

        return str(self.timestamp_ns)


@dataclass(frozen=True)
class MaskObservation:
    object_id: int
    image_id: int
    basename: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class FarmObject:
    object_id: int
    category: str
    description: str
    center_m: np.ndarray
    dimensions_m: np.ndarray
    rotation: np.ndarray
    observations: tuple[MaskObservation, ...]


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    scene_id: str
    frames_doc: Mapping[str, Any]
    frames: tuple[Frame, ...]
    objects: tuple[FarmObject, ...]
    meters_per_scene_unit: float
    rgbd_config: Mapping[str, Any]
    success_sha256: str
    acceptance_sha256: str | None
    resource_preflight: Mapping[str, Any]
    scene_preflight: Mapping[str, Any]
    legacy: bool
    integrity: Mapping[str, Any] | None = None
    mask_overrides: Mapping[tuple[int, int], Path] | None = None
    observation_exclusions: Mapping[int, Mapping[str, Any]] | None = None

    def frame(self, image_id: int) -> Frame:
        if image_id < 0 or image_id >= len(self.frames):
            raise IndexError(image_id)
        return self.frames[image_id]


def canonical_presentation_ids(state: Mapping[str, Any]) -> set[int]:
    """Return the exact final-export presentation ID set, failing closed.

    This deliberately mirrors ``farm_pipeline.final_acceptance.presentation_ids``
    without importing the live checkout. The lift executes from the FARM run's
    signed source snapshot, so reconciliation must not depend on whichever
    package happens to be installed on the host.
    """

    def values(name: str) -> list[Any]:
        value = state.get(name)
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"final state field {name!r} is not row-aligned data")
        return list(value)

    object_ids = values("object_id")
    if not object_ids:
        raise ValueError("final state has no object IDs")
    count = len(object_ids)
    active = values("active")
    tiers = values("object_semantic_tier")
    statuses = values("object_display_status")
    geometry = values("object_geometry_status")
    compound_boxes = values("object_compound_boxes")
    if any(
        len(rows) != count
        for rows in (active, tiers, statuses, geometry, compound_boxes)
    ):
        raise ValueError("final state presentation fields are not row-aligned")
    try:
        integer_ids = [int(value) for value in object_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("final state contains a non-integer object ID") from exc
    if any(value < 0 for value in integer_ids) or len(integer_ids) != len(set(integer_ids)):
        raise ValueError("final state object IDs must be unique non-negative integers")

    visible: set[int] = set()
    for index, object_id in enumerate(integer_ids):
        tier = str(tiers[index] or "").strip().lower()
        display = str(statuses[index] or "").strip().lower()
        geometry_status = str(geometry[index] or "").strip().lower()
        boxes = compound_boxes[index]
        probable_compound = (
            "compound_geometry_probable" in geometry_status
            and "rejected" not in geometry_status
            and isinstance(boxes, (list, tuple))
            and len(boxes) == 2
        )
        if not display.endswith("_suppressed") and (
            (bool(active[index]) and tier in {"confirmed", "probable"})
            or probable_compound
        ):
            visible.add(object_id)
    return visible


def validate_acceptance_contract(
    acceptance: Mapping[str, Any],
    *,
    scene_id: str,
    catalog_ids: set[int],
) -> None:
    """Validate the authoritative apply gate consumed by a release lift."""

    if acceptance.get("schema") != "farm.final-acceptance.v1":
        raise ValueError("unsupported FARM final-acceptance schema")
    if acceptance.get("mode") != "apply":
        raise ValueError("Gaussian lift requires the applied final-acceptance result")
    if str(acceptance.get("scene_id") or "") != str(scene_id):
        raise ValueError("final acceptance scene_id differs from FARM frames")
    if str(acceptance.get("status") or "").upper() not in {"PASS", "WARN"}:
        raise ValueError("FARM final acceptance is not PASS/WARN")
    errors = acceptance.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError("FARM final acceptance contains errors")
    counts = acceptance.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("FARM final acceptance counts are absent")
    required_zero = (
        "remaining_auto_clusters",
        "remaining_blocking_clusters",
        "label_hard_errors",
    )
    try:
        zero_counts = {name: int(counts[name]) for name in required_zero}
        presentation_after = int(counts["presentation_after"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("FARM final acceptance counts are incomplete") from exc
    nonzero = {name: value for name, value in zero_counts.items() if value != 0}
    if nonzero:
        raise ValueError(f"FARM final acceptance retains blockers: {nonzero}")
    if presentation_after != len(catalog_ids):
        raise ValueError("final acceptance/catalog presentation count mismatch")


def _safe_child(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"unsafe relative run path: {relative!r}")
    root = root.resolve()
    child = (root / relative).resolve()
    if not child.is_relative_to(root):
        raise ValueError(f"run path escapes root: {relative!r}")
    return child


def _same_timestamp(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        return int(left) == int(right)
    except ValueError:
        return False


def validate_physical_timestamp_bijection(
    pairs: Iterable[tuple[str, int]],
) -> tuple[dict[str, int], dict[int, str]]:
    """Require one canonical physical identity in both directions."""

    by_frame_id: dict[str, int] = {}
    by_timestamp_ns: dict[int, str] = {}
    for raw_frame_id, raw_timestamp_ns in pairs:
        frame_id = str(raw_frame_id)
        timestamp_ns = int(raw_timestamp_ns)
        if timestamp_ns < 0:
            raise ValueError("physical timestamp_ns must be non-negative")
        if frame_id in by_frame_id and by_frame_id[frame_id] != timestamp_ns:
            raise ValueError(f"physical frame_id maps to multiple timestamp_ns: {frame_id}")
        if timestamp_ns in by_timestamp_ns and by_timestamp_ns[timestamp_ns] != frame_id:
            raise ValueError(
                "physical timestamp_ns maps to multiple frame_id values: "
                f"{timestamp_ns} -> {by_timestamp_ns[timestamp_ns]!r}, {frame_id!r}"
            )
        by_frame_id[frame_id] = timestamp_ns
        by_timestamp_ns[timestamp_ns] = frame_id
    if not by_frame_id:
        raise ValueError("physical timestamp identity set is empty")
    return by_frame_id, by_timestamp_ns


_CURRENT_REQUIRED_ARTIFACTS = {
    "scene_state": ("file", "final/scene_state.pt"),
    "catalog": ("file", "final/catalog.json"),
    "presentation_catalog": ("file", "final/presentation_catalog.json"),
    "rgbd": ("directory", "rgbd"),
    "mapping": ("directory", "mapping"),
    "scene_preflight": ("file", "input/scene_preflight.json"),
    "resource_preflight": ("file", "input/resource_preflight.json"),
    "final_acceptance": ("file", "qa/acceptance/result.json"),
}


def validate_run_artifact_bundle(
    run_dir: Path,
    *,
    success: Mapping[str, Any] | None = None,
    allow_legacy: bool,
) -> Mapping[str, Any] | None:
    """Validate the root-success-bound inventory for a current FARM run.

    Every advertised artifact is verified, not only the subset opened by the
    lift. Current standard runs must also expose the canonical final state,
    catalogs, acceptance/preflight records and full RGBD/mapping trees.
    Missing bundle metadata is accepted only through the explicit legacy flag;
    malformed or drifting advertised metadata never falls back to legacy.
    """

    run_dir = run_dir.expanduser().resolve(strict=True)
    if success is None:
        loaded = json.loads((run_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("FARM success marker must be a JSON object")
        success = loaded
    relative = success.get("viewer_bundle")
    expected_bundle_sha = str(success.get("viewer_bundle_sha256") or "")
    if not expected_bundle_sha:
        if allow_legacy:
            return None
        raise ValueError("current FARM run requires a root-bound viewer artifact bundle")
    if not isinstance(relative, str) or not relative or not re.fullmatch(
        r"[0-9a-f]{64}", expected_bundle_sha
    ):
        raise ValueError("FARM root success has an invalid viewer bundle identity")
    bundle_path = _safe_child(run_dir, relative)
    if bundle_path != (run_dir / "viewer" / "bundle.json").resolve():
        raise ValueError("FARM root success points to a non-canonical viewer bundle")
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise ValueError("FARM viewer bundle is missing or a symlink")
    actual_bundle_sha = sha256_file(bundle_path)
    if actual_bundle_sha != expected_bundle_sha:
        raise ValueError("FARM viewer bundle SHA-256 differs from root success")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, Mapping) or bundle.get("schema") != "farm.viewer-bundle.v1":
        raise ValueError("unsupported FARM viewer bundle")
    if (
        str(bundle.get("scene_id") or "") != str(success.get("scene_id") or "")
        or str(bundle.get("run_id") or "") != str(success.get("run_id") or "")
    ):
        raise ValueError("FARM viewer bundle scene/run differs from root success")
    artifacts = bundle.get("artifacts")
    descriptors = bundle.get("artifact_integrity")
    if not isinstance(artifacts, Mapping) or not isinstance(descriptors, Mapping):
        raise ValueError("FARM viewer bundle artifact inventory is absent")
    if not artifacts or set(artifacts) != set(descriptors):
        raise ValueError("FARM viewer bundle artifact/path inventories differ")
    missing = sorted(set(_CURRENT_REQUIRED_ARTIFACTS) - set(artifacts))
    if missing:
        raise ValueError(f"FARM viewer bundle misses lift-critical artifacts: {missing}")

    verified: dict[str, Any] = {}
    for name in sorted(artifacts):
        descriptor = descriptors[name]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"FARM artifact descriptor is invalid: {name}")
        relative_path = artifacts[name]
        if not isinstance(relative_path, str) or descriptor.get("path") != relative_path:
            raise ValueError(f"FARM artifact path/descriptor mismatch: {name}")
        relative_object = Path(relative_path)
        if relative_object.is_absolute():
            raise ValueError(f"FARM artifact path is absolute: {name}")
        path = (bundle_path.parent / relative_object).resolve()
        if not path.is_relative_to(run_dir):
            raise ValueError(f"FARM artifact path escapes the immutable run: {name}")
        if name in _CURRENT_REQUIRED_ARTIFACTS:
            expected_kind, expected_relative = _CURRENT_REQUIRED_ARTIFACTS[name]
            if descriptor.get("kind") != expected_kind:
                raise ValueError(f"FARM artifact has wrong kind: {name}")
            if path != (run_dir / expected_relative).resolve():
                raise ValueError(f"FARM artifact has non-canonical path: {name}")
        kind = descriptor.get("kind")
        if kind == "file":
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"FARM artifact file is missing or a symlink: {name}")
            try:
                expected_bytes = int(descriptor["bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"FARM artifact file size is invalid: {name}") from exc
            expected_sha = str(descriptor.get("sha256") or "")
            if expected_bytes < 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise ValueError(f"FARM artifact file identity is invalid: {name}")
            if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
                raise ValueError(f"FARM artifact file differs from root bundle: {name}")
            verified[name] = {"kind": "file", "bytes": expected_bytes, "sha256": expected_sha}
        elif kind == "directory":
            try:
                tree = verify_directory_tree_descriptor(path, descriptor)
            except DirectoryIntegrityError as exc:
                raise ValueError(f"FARM artifact directory differs from root bundle: {name}: {exc}") from exc
            verified[name] = {"kind": "directory", **tree}
        else:
            raise ValueError(f"FARM artifact has unsupported kind: {name}")
    return {
        "viewer_bundle": str(relative),
        "viewer_bundle_sha256": actual_bundle_sha,
        "artifacts": verified,
    }


def revalidate_run_integrity(run: RunData) -> Mapping[str, Any] | None:
    """Recheck the exact run identity and every bound artifact after compute."""

    success_path = run.run_dir / "_SUCCESS.json"
    if sha256_file(success_path) != run.success_sha256:
        raise RuntimeError("FARM root success changed during bridge execution")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if not isinstance(success, Mapping):
        raise RuntimeError("FARM root success ceased to be a JSON object")
    observed = validate_run_artifact_bundle(
        run.run_dir, success=success, allow_legacy=run.legacy
    )
    if observed != run.integrity:
        raise RuntimeError("FARM artifact bundle identity changed during bridge execution")
    acceptance_path = run.run_dir / "qa" / "acceptance" / "result.json"
    if run.acceptance_sha256 is not None:
        if not acceptance_path.is_file() or sha256_file(acceptance_path) != run.acceptance_sha256:
            raise RuntimeError("FARM final acceptance changed during bridge execution")
    return observed


def load_run(run_dir: Path, *, allow_legacy: bool) -> RunData:
    import torch

    run_dir = run_dir.expanduser().resolve(strict=True)
    success_path = run_dir / "_SUCCESS.json"
    if not success_path.is_file():
        raise ValueError(f"FARM run is not successful/immutable: {run_dir}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if not isinstance(success, Mapping) or success.get("schema") != "farm.pipeline-success.v1":
        raise ValueError("unsupported FARM success marker")
    if str(success.get("status", "")).lower() != "success":
        raise ValueError("FARM _SUCCESS status is not success")
    if str(success.get("run_id") or "") != run_dir.name:
        raise ValueError("FARM _SUCCESS run_id differs from run directory")
    integrity = validate_run_artifact_bundle(
        run_dir, success=success, allow_legacy=allow_legacy
    )
    acceptance_path = run_dir / "qa" / "acceptance" / "result.json"
    legacy = integrity is None or not acceptance_path.is_file()
    acceptance_sha: str | None = None
    if legacy and not allow_legacy:
        raise ValueError("final acceptance artifact is required; legacy override is non-release only")
    acceptance: Mapping[str, Any] | None = None
    if acceptance_path.is_file():
        value = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("FARM final acceptance must be a JSON object")
        acceptance = value
        acceptance_sha = sha256_file(acceptance_path)

    frames_path = run_dir / "rgbd" / "frames.json"
    frames_doc = json.loads(frames_path.read_text(encoding="utf-8"))
    if frames_doc.get("schema_version") != "farm_frames_json_v1":
        raise ValueError("unsupported frames.json schema")
    if frames_doc.get("depth_units") != "metres" or frames_doc.get("pose_translation_units") != "metres":
        raise ValueError("FARM frames must use metric depth and pose translation")
    scale = float(frames_doc.get("meters_per_scene_unit", 0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid meters_per_scene_unit")
    scene_id = str(frames_doc.get("scene_id") or "")
    if not scene_id or str(success.get("scene_id") or "") != scene_id:
        raise ValueError("FARM success marker and frames scene_id differ")
    contract = frames_doc.get("identity_contract") or {}
    pattern = re.compile(str(contract.get("regex") or ""))
    sensor_group = str(contract.get("sensor_group") or "")
    timestamp_group = str(contract.get("timestamp_group") or "")
    family_group = str(contract.get("family_group") or "")
    if not sensor_group or not timestamp_group or not family_group:
        raise ValueError("frames identity contract is incomplete")

    frames: list[Frame] = []
    physical_pairs: list[tuple[str, int]] = []
    rgbd_root = run_dir / "rgbd"
    for image_id, row in enumerate(frames_doc.get("frames") or []):
        source_image = str(row["source_image"])
        match = pattern.search(source_image)
        if match is None:
            raise ValueError(f"source image violates identity regex: {source_image}")
        frame_id = str(row["frame_id"])
        parsed_timestamp = str(match.group(timestamp_group))
        if not _same_timestamp(frame_id, parsed_timestamp):
            raise ValueError(f"frame_id/regex timestamp mismatch: {source_image}")
        timestamp_ns = int(row["timestamp_ns"])
        physical_pairs.append((frame_id, timestamp_ns))
        depth_size = tuple(int(value) for value in row["depth_size"])
        if len(depth_size) != 2 or min(depth_size) <= 0:
            raise ValueError("invalid frame depth_size")
        K = np.asarray(row["K"], dtype=np.float64)
        pose = np.asarray(row["T_world_cam"], dtype=np.float64)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 0] <= 0
            or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-8)
            or pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
        ):
            raise ValueError("invalid frame calibration")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4) or np.linalg.det(rotation) <= 0:
            raise ValueError("invalid camera rotation")
        rgb_path = _safe_child(rgbd_root, str(row["rgb_path"]))
        depth_path = _safe_child(rgbd_root, str(row["depth_path"]))
        if not rgb_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"frame assets missing for image_id={image_id}")
        frames.append(Frame(
            image_id=image_id,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            camera=str(row["camera"]),
            sensor=str(match.group(sensor_group)),
            family=str(match.group(family_group)),
            source_image=source_image,
            depth_size=(depth_size[0], depth_size[1]),
            K=K,
            T_world_cam=pose,
            rgb_path=rgb_path,
            depth_path=depth_path,
        ))
    if not frames:
        raise ValueError("frames.json is empty")
    validate_physical_timestamp_bijection(physical_pairs)

    wrapper = torch.load(run_dir / "final" / "scene_state.pt", map_location="cpu", weights_only=False)
    state = wrapper.get("state", wrapper) if isinstance(wrapper, Mapping) else wrapper
    catalog = json.loads((run_dir / "final" / "presentation_catalog.json").read_text(encoding="utf-8"))
    if not isinstance(catalog, list):
        raise TypeError("presentation catalog must be a JSON array")
    if not catalog or not all(isinstance(row, Mapping) for row in catalog):
        raise ValueError("presentation catalog must contain JSON objects")
    if any(row.get("presentation_visible") is not True for row in catalog):
        raise ValueError("presentation catalog contains a non-visible row")
    try:
        catalog_ids = [int(row["id"]) for row in catalog]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("presentation catalog contains an invalid object ID") from exc
    if any(value < 0 for value in catalog_ids) or len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError("presentation catalog IDs must be unique non-negative integers")
    state_visible_ids = canonical_presentation_ids(state)
    if set(catalog_ids) != state_visible_ids:
        raise ValueError("final state and presentation catalog ID sets differ")
    if acceptance is not None:
        validate_acceptance_contract(
            acceptance,
            scene_id=scene_id,
            catalog_ids=set(catalog_ids),
        )
    state_ids = np.asarray(state["object_id"].detach().cpu(), dtype=np.int64)
    if len(state_ids) != len(np.unique(state_ids)):
        raise ValueError("final state object IDs are not unique")
    id_to_row = {int(value): index for index, value in enumerate(state_ids.tolist())}
    objects: list[FarmObject] = []
    for row in catalog:
        object_id = int(row["id"])
        if object_id not in id_to_row:
            raise ValueError(f"presentation ID absent from final state: {object_id}")
        state_row = id_to_row[object_id]
        assembly_member_ids = {
            int(value) for value in (row.get("assembly_member_ids") or [])
        }
        observations: list[MaskObservation] = []
        for record in state["object_mask_observations"][state_row]:
            image_id = int(record["image_id"])
            if image_id < 0 or image_id >= len(frames):
                raise ValueError(f"invalid mask image_id={image_id} for object {object_id}")
            recorded_id = int(record.get("object_id", object_id))
            if recorded_id != object_id and recorded_id not in assembly_member_ids:
                raise ValueError(f"mask/object ID mismatch: {recorded_id} != {object_id}")
            basename = Path(str(record.get("path") or "")).name
            if not basename or basename in {".", ".."}:
                raise ValueError(f"invalid mask basename for object {object_id}")
            observations.append(MaskObservation(object_id, image_id, basename, dict(record)))
        if not observations:
            raise ValueError(f"presentation object {object_id} has no mask observations")
        objects.append(FarmObject(
            object_id=object_id,
            category=str(row.get("category") or "object"),
            description=str(row.get("description") or ""),
            center_m=np.asarray(row["center_m"], dtype=np.float64),
            dimensions_m=np.asarray(row["dimensions_m"], dtype=np.float64),
            rotation=wxyz_to_matrix(row["wxyz"]),
            observations=tuple(observations),
        ))

    rgbd_manifest = json.loads((run_dir / "rgbd" / "run_manifest.json").read_text(encoding="utf-8"))
    rgbd_config = (rgbd_manifest.get("fingerprint_payload") or {}).get("config") or {}
    for key in ("alpha_min", "depth_min_m", "depth_max_m", "radius_clip"):
        if key not in rgbd_config:
            raise ValueError(f"RGBD render contract misses {key}")
    resource_path = run_dir / "input" / "resource_preflight.json"
    scene_path = run_dir / "input" / "scene_preflight.json"
    return RunData(
        run_dir=run_dir,
        scene_id=scene_id,
        frames_doc=frames_doc,
        frames=tuple(frames),
        objects=tuple(objects),
        meters_per_scene_unit=scale,
        rgbd_config=rgbd_config,
        success_sha256=sha256_file(success_path),
        acceptance_sha256=acceptance_sha,
        resource_preflight=json.loads(resource_path.read_text(encoding="utf-8")),
        scene_preflight=json.loads(scene_path.read_text(encoding="utf-8")),
        legacy=legacy,
        integrity=integrity,
    )


def resolve_mask_path(run: RunData, observation: MaskObservation) -> Path:
    if run.mask_overrides:
        override = run.mask_overrides.get((observation.object_id, observation.image_id))
        if override is not None:
            return override
    direct = run.run_dir / "mapping" / "masks" / f"object_{observation.object_id:06d}" / observation.basename
    assembly = run.run_dir / "qa" / "assemblies" / "masks" / f"object_{observation.object_id:06d}" / observation.basename
    hint = str(observation.record.get("path") or "").replace("\\", "/")
    candidates = [assembly, direct] if "/qa/assemblies/" in hint else [direct, assembly]
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"mask resolution must be unique for object={observation.object_id}, "
            f"image={observation.image_id}, basename={observation.basename}: {existing}"
        )
    return existing[0]


def load_observation_exclusion(run: RunData, frame: Frame) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Load immutable per-view unknown pixels; absent input preserves legacy behavior.

    Excluded pixels cannot supply positive, negative, visibility, or evaluation
    evidence. The original scene depth and Gaussian identity remain unchanged.
    """
    records = getattr(run, "observation_exclusions", None)
    if not records or frame.image_id not in records:
        return None, None
    record = records[frame.image_id]
    path = Path(record["path"])
    if not path.is_absolute():
        path = run.run_dir / path
    digest = sha256_file(path)
    if digest != record.get("sha256") or path.stat().st_size != record.get("bytes"):
        raise ValueError("observation exclusion artifact changed")
    excluded = np.load(path, allow_pickle=False)
    if excluded.shape != frame.depth_size or excluded.dtype != np.bool_:
        raise ValueError("observation exclusion must be a boolean native-grid mask")
    return excluded, {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
                      "excluded_pixels": int(excluded.sum()), "interpretation": "unknown, never background"}


def load_mask_pair(path: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        image_shape = tuple(int(value) for value in np.asarray(archive["image_shape"]).reshape(2))
        if image_shape != tuple(expected_shape):
            raise ValueError(f"mask image_shape {image_shape} != frame depth_size {expected_shape}")

        def decode(kind: str) -> np.ndarray:
            selected = kind if f"{kind}_bits" in archive.files else "raw"
            shape = tuple(int(value) for value in np.asarray(archive[f"{selected}_shape"]).reshape(2))
            bbox = np.asarray(archive[f"{selected}_bbox_xyxy"], dtype=np.int64).reshape(4)
            x0, y0, x1, y1 = (int(value) for value in bbox)
            if shape != (y1 - y0, x1 - x0):
                raise ValueError(f"{selected} mask shape/bbox mismatch in {path}")
            if x0 < 0 or y0 < 0 or x1 > image_shape[1] or y1 > image_shape[0] or x1 <= x0 or y1 <= y0:
                raise ValueError(f"{selected} mask bbox outside image in {path}")
            bits = np.asarray(archive[f"{selected}_bits"], dtype=np.uint8)
            flat = np.unpackbits(bits, bitorder="little")
            needed = shape[0] * shape[1]
            if flat.size < needed:
                raise ValueError(f"{selected} mask bitstream truncated in {path}")
            local = flat[:needed].reshape(shape).astype(bool)
            full = np.zeros(image_shape, dtype=bool)
            full[y0:y1, x0:x1] = local
            return full
        raw = decode("raw")
        inlier = decode("inlier")
    return raw, inlier, digest


def run_ply_preflight_spec(run: RunData) -> Mapping[str, Any]:
    checks = {str(row.get("name")): row for row in run.scene_preflight.get("checks") or []}
    paths = ((checks.get("paths") or {}).get("metrics") or {}).get("resolved") or {}
    expected_path = str(paths.get("gaussian_ply") or "")
    files = (((checks.get("fingerprints") or {}).get("metrics") or {}).get("files") or [])
    matches = [row for row in files if str(row.get("path")) == expected_path]
    if len(matches) != 1:
        raise ValueError("cannot identify Gaussian PLY fingerprint in scene preflight")
    return matches[0]


def verify_run_ply_fingerprint(run: RunData, path: Path) -> dict[str, Any]:
    spec = dict(run_ply_preflight_spec(run))
    path = path.expanduser().resolve(strict=True)
    if int(spec.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError("source PLY size differs from FARM scene preflight")
    algorithm = str(spec.get("algorithm"))
    if not run.legacy and algorithm != "sha256":
        raise ValueError("current FARM run requires a full SHA-256 source PLY fingerprint")
    if algorithm == "sha256-sampled-v1":
        digest = preflight_sampled_sha256(path, int(spec["chunk_bytes"]))
    elif algorithm == "sha256":
        digest = sha256_file(path, int(spec["chunk_bytes"]))
    else:
        raise ValueError(f"unsupported FARM PLY fingerprint algorithm: {algorithm}")
    if digest != str(spec.get("digest")):
        raise ValueError("source PLY content differs from FARM scene preflight")
    return {
        "algorithm": algorithm,
        "digest": digest,
        "size_bytes": path.stat().st_size,
        "preflight_path": str(spec.get("path")),
        "matched": True,
    }


def solve_global_timestamp_split(
    ordered_timestamps: Sequence[str],
    object_timestamps: Mapping[int, Sequence[str]],
    policy: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    timestamps = tuple(str(value) for value in ordered_timestamps)
    if len(timestamps) != len(set(timestamps)) or not timestamps:
        raise ValueError("ordered physical timestamps must be unique and non-empty")
    timestamp_index = {value: index for index, value in enumerate(timestamps)}
    object_ids = tuple(sorted(int(value) for value in object_timestamps))
    incidence = np.zeros((len(object_ids), len(timestamps)), dtype=np.float64)
    ordered_by_object: dict[int, list[str]] = {}
    for row_index, object_id in enumerate(object_ids):
        unique_values = set(str(value) for value in object_timestamps[object_id])
        unknown = sorted(unique_values - set(timestamp_index))
        if unknown:
            raise ValueError(f"object {object_id} references an unknown timestamp: {unknown}")
        values = sorted(unique_values, key=timestamp_index.__getitem__)
        ordered_by_object[object_id] = values
        incidence[row_index, [timestamp_index[value] for value in values]] = 1.0

    strict_min = int(policy["strict_minimum_timestamps"])
    strict_build = int(policy["strict_minimum_build_timestamps"])
    strict_heldout = int(policy["strict_minimum_heldout_timestamps"])
    evidence_min = int(policy["evidence_limited_minimum_timestamps"])
    lower: list[float] = []
    upper: list[float] = []
    for object_id in object_ids:
        count = len(ordered_by_object[object_id])
        if count >= strict_min:
            lower.append(float(strict_heldout))
            upper.append(float(count - strict_build))
        elif count >= evidence_min:
            lower.append(1.0)
            upper.append(float(count - (count - 1)))
        else:
            lower.append(0.0)
            upper.append(0.0)

    timestamp_count, object_count = len(timestamps), len(object_ids)
    variable_count = timestamp_count + object_count + 1
    global_deviation = variable_count - 1
    objective = np.zeros(variable_count, dtype=np.float64)
    for index, value in enumerate(timestamps):
        token = hashlib.sha256(f"{seed}:{value}".encode()).digest()
        objective[index] = int.from_bytes(token[:8], "little") / (2**64) * 1e-7
    for row_index, object_id in enumerate(object_ids):
        objective[timestamp_count + row_index] = 1.0 / max(len(ordered_by_object[object_id]), 1)
    objective[global_deviation] = 1.0 / max(timestamp_count, 1)

    matrices: list[np.ndarray] = []
    lows: list[float] = []
    highs: list[float] = []

    object_bounds = np.zeros((object_count, variable_count), dtype=np.float64)
    object_bounds[:, :timestamp_count] = incidence
    matrices.append(object_bounds); lows.extend(lower); highs.extend(upper)

    global_row = np.zeros((1, variable_count), dtype=np.float64)
    global_row[0, :timestamp_count] = 1.0
    matrices.append(global_row)
    lows.append(math.ceil(float(policy["minimum_heldout_fraction"]) * timestamp_count))
    highs.append(math.floor(float(policy["maximum_heldout_fraction"]) * timestamp_count))

    bins = min(int(policy["temporal_bins"]), timestamp_count)
    for indices in np.array_split(np.arange(timestamp_count), bins):
        row = np.zeros((1, variable_count), dtype=np.float64)
        row[0, indices] = 1.0
        matrices.append(row); lows.append(1.0); highs.append(np.inf)

    if bool(policy.get("require_early_late_heldout", True)):
        for object_id in object_ids:
            values = ordered_by_object[object_id]
            if len(values) < strict_min:
                continue
            midpoint = len(values) // 2
            for half in (values[:midpoint], values[midpoint:]):
                row = np.zeros((1, variable_count), dtype=np.float64)
                row[0, [timestamp_index[value] for value in half]] = 1.0
                matrices.append(row); lows.append(1.0); highs.append(np.inf)

    targets = np.asarray([
        float(policy["target_heldout_fraction"]) * len(ordered_by_object[object_id])
        for object_id in object_ids
    ])
    positive = np.zeros((object_count, variable_count), dtype=np.float64)
    positive[:, :timestamp_count] = incidence
    positive[:, timestamp_count:timestamp_count + object_count] = -np.eye(object_count)
    matrices.append(positive); lows.extend([-np.inf] * object_count); highs.extend(targets.tolist())
    negative = np.zeros((object_count, variable_count), dtype=np.float64)
    negative[:, :timestamp_count] = -incidence
    negative[:, timestamp_count:timestamp_count + object_count] = -np.eye(object_count)
    matrices.append(negative); lows.extend([-np.inf] * object_count); highs.extend((-targets).tolist())

    global_target = float(policy["target_heldout_fraction"]) * timestamp_count
    plus_global = np.zeros((1, variable_count), dtype=np.float64)
    plus_global[0, :timestamp_count] = 1.0
    plus_global[0, global_deviation] = -1.0
    minus_global = np.zeros((1, variable_count), dtype=np.float64)
    minus_global[0, :timestamp_count] = -1.0
    minus_global[0, global_deviation] = -1.0
    matrices.extend([plus_global, minus_global])
    lows.extend([-np.inf, -np.inf]); highs.extend([global_target, -global_target])

    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.concatenate((
        np.ones(timestamp_count, dtype=np.float64),
        np.full(object_count + 1, np.inf, dtype=np.float64),
    ))
    integrality = np.concatenate((
        np.ones(timestamp_count, dtype=np.int32),
        np.zeros(object_count + 1, dtype=np.int32),
    ))
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(matrices), np.asarray(lows), np.asarray(highs)),
        options={"time_limit": float(policy["solver_time_limit_seconds"])},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"global physical-timestamp split is infeasible: {result.message}")
    heldout = {
        timestamps[index]
        for index, selected in enumerate(np.rint(result.x[:timestamp_count]).astype(np.int8))
        if selected
    }
    build = set(timestamps) - heldout
    rows: list[dict[str, Any]] = []
    for object_id in object_ids:
        values = ordered_by_object[object_id]
        object_build = [value for value in values if value in build]
        object_heldout = [value for value in values if value in heldout]
        if len(values) >= strict_min:
            eligibility = "strict"
        elif len(values) >= evidence_min:
            eligibility = "evidence_limited"
        else:
            eligibility = "insufficient"
        rows.append({
            "object_id": object_id,
            "eligibility": eligibility,
            "all_timestamps": values,
            "build_timestamps": object_build,
            "heldout_timestamps": object_heldout,
        })
    return {
        "schema_version": "farm.gaussian-lift.split.v1",
        "solver": "scipy.optimize.milp/HiGHS",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_objective": float(result.fun),
        "seed": int(seed),
        "physical_timestamp_count": timestamp_count,
        "build_timestamps": [value for value in timestamps if value in build],
        "heldout_timestamps": [value for value in timestamps if value in heldout],
        "objects": rows,
    }


def diversity_metrics(camera_centres: Sequence[np.ndarray], object_center: np.ndarray) -> dict[str, float]:
    centres = np.asarray(camera_centres, dtype=np.float64).reshape(-1, 3)
    if len(centres) < 2:
        return {"baseline_span_m": 0.0, "angular_span_degrees": 0.0}
    difference = centres[:, None, :] - centres[None, :, :]
    baseline = float(np.max(np.linalg.norm(difference, axis=2)))
    rays = np.asarray(object_center, dtype=np.float64).reshape(1, 3) - centres
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(rays @ rays.T, -1.0, 1.0)
    angular = float(np.degrees(np.max(np.arccos(cosine))))
    return {"baseline_span_m": baseline, "angular_span_degrees": angular}


def resolve_sparse_claims(
    claims: Sequence[Mapping[str, np.ndarray | int]],
    gaussian_count: int,
    *,
    minimum_margin: float,
    minimum_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    parts = [row for row in claims if np.asarray(row["indices"]).size]
    labels = np.full(int(gaussian_count), UNKNOWN_ID, dtype=np.int32)
    confidence = np.zeros(int(gaussian_count), dtype=np.float32)
    support = np.zeros(int(gaussian_count), dtype=np.uint16)
    if not parts:
        return labels, confidence, support, {"claims": 0, "unique": 0, "ambiguous": 0}
    indices = np.concatenate([np.asarray(row["indices"], dtype=np.int64) for row in parts])
    scores = np.concatenate([np.asarray(row["scores"], dtype=np.float32) for row in parts])
    confidences = np.concatenate([np.asarray(row["confidence"], dtype=np.float32) for row in parts])
    supports = np.concatenate([np.asarray(row["support"], dtype=np.uint16) for row in parts])
    object_ids = np.concatenate([
        np.full(np.asarray(row["indices"]).size, int(row["object_id"]), dtype=np.int32)
        for row in parts
    ])
    if np.any((indices < 0) | (indices >= gaussian_count)):
        raise ValueError("claim indices outside source Gaussian order")
    order = np.lexsort((object_ids, -scores, indices))
    indices, scores, confidences, supports, object_ids = (
        value[order] for value in (indices, scores, confidences, supports, object_ids)
    )
    starts = np.r_[0, np.flatnonzero(np.diff(indices)) + 1]
    sizes = np.diff(np.r_[starts, len(indices)])
    top = starts
    top_score = scores[top]
    second_score = np.zeros_like(top_score)
    has_second = sizes > 1
    second_score[has_second] = scores[starts[has_second] + 1]
    margin = top_score - second_score
    ratio = top_score / np.maximum(second_score, 1e-8)
    accepted = (~has_second) | (
        (margin >= float(minimum_margin)) & (ratio >= float(minimum_ratio))
    )
    selected = top[accepted]
    target = indices[selected]
    labels[target] = object_ids[selected]
    confidence[target] = confidences[selected]
    support[target] = supports[selected]
    return labels, confidence, support, {
        "claims": int(len(indices)),
        "unique": int(len(starts)),
        "ambiguous": int(np.count_nonzero(~accepted)),
    }


def build_verified_csr(
    labels: np.ndarray, confidence: np.ndarray, support: np.ndarray
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32)
    confidence = np.asarray(confidence, dtype=np.float32)
    support = np.asarray(support, dtype=np.uint16)
    if labels.shape != confidence.shape or labels.shape != support.shape or labels.ndim != 1:
        raise ValueError("CSR inputs must be same-length dense vectors")
    indices = np.flatnonzero(labels >= 0).astype(np.int64)
    if not len(indices):
        return {
            "object_ids": np.zeros(0, dtype=np.int32),
            "indptr": np.zeros(1, dtype=np.int64),
            "indices": indices,
            "confidence": np.zeros(0, dtype=np.float32),
            "timestamp_support": np.zeros(0, dtype=np.uint16),
        }
    order = np.lexsort((indices, labels[indices]))
    indices = indices[order]
    ordered_labels = labels[indices]
    object_ids, counts = np.unique(ordered_labels, return_counts=True)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    return {
        "object_ids": object_ids.astype(np.int32),
        "indptr": indptr,
        "indices": indices,
        "confidence": confidence[indices].astype(np.float32),
        "timestamp_support": support[indices].astype(np.uint16),
    }


def binary_mask_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    soft: np.ndarray | None = None,
) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("predicted/target masks differ in shape")
    intersection = int(np.count_nonzero(predicted & target))
    prediction_pixels = int(predicted.sum())
    target_pixels = int(target.sum())
    union = prediction_pixels + target_pixels - intersection
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        predicted.astype(np.uint8), connectivity=8
    )
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
    result: dict[str, float | int] = {
        "intersection_pixels": intersection,
        "prediction_pixels": prediction_pixels,
        "target_pixels": target_pixels,
        "iou": intersection / max(union, 1),
        "precision": intersection / max(prediction_pixels, 1),
        "recall": intersection / max(target_pixels, 1),
        "largest_component_fraction": largest / max(prediction_pixels, 1),
        "area_ratio": prediction_pixels / max(target_pixels, 1),
    }
    if soft is not None:
        values = np.asarray(soft, dtype=np.float64)
        soft_intersection = float(np.minimum(values, target.astype(np.float64)).sum())
        soft_union = float(np.maximum(values, target.astype(np.float64)).sum())
        result["soft_iou"] = soft_intersection / max(soft_union, 1e-12)
    return result
