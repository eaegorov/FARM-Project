"""Fail-closed reconstruction of virtual assembly mask sidecars."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AssemblyMaskReconstruction:
    arrays: dict[str, np.ndarray]
    logical_sha256: str
    member_provenance: tuple[dict[str, Any], ...]


_ASSEMBLY_ARRAY_NAMES = (
    "image_shape",
    "raw_bits",
    "raw_shape",
    "raw_bbox_xyxy",
    "inlier_bits",
    "inlier_shape",
    "inlier_bbox_xyxy",
    "crop_jpeg_bytes",
    "crop_bbox_xyxy",
    "crop_shape",
)


def reconstruction_observation_metadata(
    reconstruction: AssemblyMaskReconstruction,
) -> dict[str, Any]:
    """Return observation metadata derived from the emitted logical payload."""

    arrays = reconstruction.arrays

    def packed_pixels(kind: str) -> int:
        shape = np.asarray(arrays[f"{kind}_shape"], dtype=np.int64).reshape(-1)
        if shape.size != 2 or np.any(shape < 0):
            raise ValueError(f"reconstructed {kind} mask shape is invalid")
        count = int(shape[0]) * int(shape[1])
        bits = np.asarray(arrays[f"{kind}_bits"], dtype=np.uint8).reshape(-1)
        unpacked = np.unpackbits(bits, bitorder="little")
        if unpacked.size < count:
            raise ValueError(f"reconstructed {kind} mask is truncated")
        return int(unpacked[:count].sum(dtype=np.int64))

    return {
        "image_shape": np.asarray(arrays["image_shape"], dtype=np.int64)
        .reshape(2)
        .tolist(),
        "raw_pixels": packed_pixels("raw"),
        "inlier_pixels": packed_pixels("inlier"),
        "crop_jpeg_bytes_len": int(
            np.asarray(arrays["crop_jpeg_bytes"], dtype=np.uint8).size
        ),
        "crop_bbox_xyxy": np.asarray(
            arrays["crop_bbox_xyxy"], dtype=np.int64
        ).reshape(4).tolist(),
        "crop_shape": np.asarray(arrays["crop_shape"], dtype=np.int64)
        .reshape(2)
        .tolist(),
    }


def load_assembly_mask_payload(path: Path) -> AssemblyMaskReconstruction:
    """Load and validate an existing assembly with a ZIP-independent digest."""

    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("assembly mask source must be a regular non-symlink file")
    with np.load(resolved, allow_pickle=False) as data:
        if set(data.files) != set(_ASSEMBLY_ARRAY_NAMES):
            raise ValueError("existing assembly mask has an unexpected array schema")
        shape_raw = np.asarray(data["image_shape"], dtype=np.int64).reshape(-1)
        if shape_raw.size != 2 or np.any(shape_raw <= 0):
            raise ValueError("existing assembly mask has invalid image_shape")
        image_shape = (int(shape_raw[0]), int(shape_raw[1]))
        _unpack_mask_canvas(data, "raw", image_shape)
        _unpack_mask_canvas(data, "inlier", image_shape)
        np.asarray(data["crop_jpeg_bytes"], dtype=np.uint8).reshape(-1)
        np.asarray(data["crop_bbox_xyxy"], dtype=np.int64).reshape(4)
        np.asarray(data["crop_shape"], dtype=np.int64).reshape(2)
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    reconstruction = AssemblyMaskReconstruction(
        arrays=arrays,
        logical_sha256=logical_payload_sha256(arrays),
        member_provenance=(),
    )
    reconstruction_observation_metadata(reconstruction)
    return reconstruction


def _as_ids(value: object, *, field: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        result = np.asarray(value).astype(np.int64).reshape(-1).tolist()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an integer ID sequence") from exc
    return [int(item) for item in result]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_mask_path(
    root: Path, object_id: int, observation: Mapping[str, Any]
) -> Path:
    basename = Path(str(observation.get("path") or "")).name
    if not basename or basename in {".", ".."}:
        raise ValueError("member mask observation has no safe basename")
    candidate = root / f"object_{object_id:06d}" / basename
    resolved = candidate.resolve(strict=True)
    if candidate.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("assembly member mask escapes source mask root")
    return resolved


def _unpack_mask_canvas(
    data: np.lib.npyio.NpzFile, kind: str, image_shape: tuple[int, int]
) -> np.ndarray:
    height, width = image_shape
    canvas = np.zeros((height, width), dtype=bool)
    keys = (f"{kind}_bits", f"{kind}_shape", f"{kind}_bbox_xyxy")
    present = [key in data.files for key in keys]
    if not any(present):
        return canvas
    if not all(present):
        raise ValueError(f"assembly member has incomplete {kind} packed mask")
    shape = np.asarray(data[keys[1]], dtype=np.int64).reshape(-1)
    bbox = np.asarray(data[keys[2]], dtype=np.int64).reshape(-1)
    if shape.size != 2 or bbox.size != 4:
        raise ValueError(f"assembly member has invalid {kind} shape/bbox")
    mask_height, mask_width = map(int, shape)
    x0, y0, x1, y1 = map(int, bbox)
    if (
        mask_height < 0
        or mask_width < 0
        or not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height)
        or (mask_height, mask_width) != (y1 - y0, x1 - x0)
    ):
        raise ValueError(f"assembly member has inconsistent {kind} geometry")
    flat = np.unpackbits(
        np.asarray(data[keys[0]], dtype=np.uint8).reshape(-1), bitorder="little"
    )
    pixels = mask_height * mask_width
    if flat.size < pixels:
        raise ValueError(f"assembly member has truncated {kind} packed mask")
    mask = flat[:pixels].reshape(mask_height, mask_width).astype(bool, copy=False)
    canvas[y0:y1, x0:x1] = mask
    return canvas


def _packed_crop(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (
            np.zeros((0,), dtype=np.uint8),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([0, 0, 0, 0], dtype=np.int32),
        )
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1]
    return (
        np.packbits(crop.reshape(-1).astype(np.uint8), bitorder="little"),
        np.asarray(crop.shape, dtype=np.int32),
        np.asarray([x0, y0, x1, y1], dtype=np.int32),
    )


def logical_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash named logical arrays without depending on ZIP metadata."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def reconstruct_assembly_mask(
    state: Mapping[str, Any],
    *,
    assembly_object_id: int,
    old_image_id: int,
    assembly_observation: Mapping[str, Any],
    source_mask_root: Path,
) -> AssemblyMaskReconstruction:
    """Rebuild one missing assembly sidecar from its proven member observations."""

    root = source_mask_root.expanduser().resolve(strict=True)
    object_ids = _as_ids(state.get("object_id"), field="object_id")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("state object IDs are not unique")
    index_by_id = {object_id: index for index, object_id in enumerate(object_ids)}
    if assembly_object_id not in index_by_id:
        raise ValueError("assembly object ID is absent from state")
    assembly_rows = state.get("object_assembly_member_ids")
    observations = state.get("object_mask_observations")
    if not isinstance(assembly_rows, (list, tuple)) or len(assembly_rows) != len(object_ids):
        raise ValueError("object_assembly_member_ids is not object-aligned")
    if not isinstance(observations, (list, tuple)) or len(observations) != len(object_ids):
        raise ValueError("object_mask_observations is not object-aligned")
    assembly_index = index_by_id[assembly_object_id]
    declared = _as_ids(
        assembly_rows[assembly_index], field="object_assembly_member_ids row"
    )
    in_frame = _as_ids(
        assembly_observation.get("assembly_member_ids_in_frame"),
        field="assembly_member_ids_in_frame",
    )
    if (
        not declared
        or len(declared) != len(set(declared))
        or not in_frame
        or len(in_frame) != len(set(in_frame))
        or not set(in_frame).issubset(set(declared))
    ):
        raise ValueError("assembly member IDs are empty, repeated, or mismatched")

    candidates: list[tuple[int, Path]] = []
    for member_id in sorted(in_frame):
        member_index = index_by_id.get(member_id)
        if member_index is None:
            raise ValueError(f"assembly member object is absent: {member_id}")
        member_rows = observations[member_index]
        if not isinstance(member_rows, (list, tuple)):
            raise ValueError("assembly member observations row is invalid")
        matches = [
            row
            for row in member_rows
            if isinstance(row, Mapping)
            and int(row.get("image_id", -1)) == int(old_image_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                "assembly member must have exactly one observation for source image: "
                f"object={member_id}, image={old_image_id}, matches={len(matches)}"
            )
        candidates.append(
            (member_id, _bounded_mask_path(root, member_id, matches[0]))
        )

    image_shape: tuple[int, int] | None = None
    raw_canvas: np.ndarray | None = None
    inlier_canvas: np.ndarray | None = None
    crop_payload: dict[str, np.ndarray] | None = None
    provenance: list[dict[str, Any]] = []
    for member_id, path in candidates:
        with np.load(path, allow_pickle=False) as data:
            shape_raw = np.asarray(data["image_shape"], dtype=np.int64).reshape(-1)
            if shape_raw.size != 2 or np.any(shape_raw <= 0):
                raise ValueError("assembly member has invalid image_shape")
            shape = (int(shape_raw[0]), int(shape_raw[1]))
            if image_shape is None:
                image_shape = shape
                raw_canvas = np.zeros(shape, dtype=bool)
                inlier_canvas = np.zeros(shape, dtype=bool)
            if shape != image_shape:
                raise ValueError(
                    f"inconsistent assembly member image shapes: {image_shape} vs {shape}"
                )
            assert raw_canvas is not None and inlier_canvas is not None
            raw_canvas |= _unpack_mask_canvas(data, "raw", image_shape)
            inlier_canvas |= _unpack_mask_canvas(data, "inlier", image_shape)
            crop_bytes = np.asarray(
                data["crop_jpeg_bytes"]
                if "crop_jpeg_bytes" in data.files
                else np.zeros((0,), dtype=np.uint8),
                dtype=np.uint8,
            ).reshape(-1)
            if crop_payload is None or crop_bytes.size > crop_payload["crop_jpeg_bytes"].size:
                crop_payload = {
                    "crop_jpeg_bytes": crop_bytes.copy(),
                    "crop_bbox_xyxy": np.asarray(
                        data["crop_bbox_xyxy"]
                        if "crop_bbox_xyxy" in data.files
                        else [0, 0, 0, 0],
                        dtype=np.int32,
                    ).reshape(4).copy(),
                    "crop_shape": np.asarray(
                        data["crop_shape"]
                        if "crop_shape" in data.files
                        else [0, 0],
                        dtype=np.int32,
                    ).reshape(2).copy(),
                }
        provenance.append(
            {
                "member_object_id": member_id,
                "source_relative": str(path.relative_to(root)),
                "source_sha256": _sha256(path),
            }
        )

    assert image_shape is not None and raw_canvas is not None and inlier_canvas is not None
    raw_bits, raw_shape, raw_bbox = _packed_crop(raw_canvas)
    inlier_bits, inlier_shape, inlier_bbox = _packed_crop(inlier_canvas)
    crop_payload = crop_payload or {
        "crop_jpeg_bytes": np.zeros((0,), dtype=np.uint8),
        "crop_bbox_xyxy": np.zeros((4,), dtype=np.int32),
        "crop_shape": np.zeros((2,), dtype=np.int32),
    }
    arrays = {
        "image_shape": np.asarray(image_shape, dtype=np.int32),
        "raw_bits": raw_bits,
        "raw_shape": raw_shape,
        "raw_bbox_xyxy": raw_bbox,
        "inlier_bits": inlier_bits,
        "inlier_shape": inlier_shape,
        "inlier_bbox_xyxy": inlier_bbox,
        **crop_payload,
    }
    return AssemblyMaskReconstruction(
        arrays=arrays,
        logical_sha256=logical_payload_sha256(arrays),
        member_provenance=tuple(provenance),
    )


def write_reconstructed_assembly_mask(
    reconstruction: AssemblyMaskReconstruction,
    destination: Path,
    staging_mask_root: Path,
) -> None:
    root = staging_mask_root.expanduser().resolve(strict=True)
    target = destination.expanduser().resolve(strict=False)
    if not target.is_relative_to(root) or target.suffix.lower() != ".npz":
        raise ValueError("assembly reconstruction target escapes staging mask root")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    if temporary.exists() or target.exists():
        raise FileExistsError(target)
    np.savez_compressed(temporary, **reconstruction.arrays)
    temporary.replace(target)
