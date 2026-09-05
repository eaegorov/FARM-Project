"""Fail-closed native dataset builder for the Gaussian Grouping baseline.

The tracker emits episode-local uint8 identities.  Those numeric values have
no meaning outside one episode, so this module will only materialize a
multi-episode dataset after an audited association report explicitly maps
every observed local identity to a scene-global ID (or to zero/drop).

RGB files and the two PLY inputs remain symlinks.  Remapped uint8 masks and a
small filtered COLMAP text model are the only derived data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np
from PIL import Image


EPISODES_SCHEMA = "farm.materialized-tracker-episodes.v1"
EPISODE_SCHEMA = "farm.materialized-tracker-episode.v1"
ASSOCIATION_SCHEMA = "farm.tracker-global-association.v1"
OUTPUT_SCHEMA = "farm.gaussian-grouping-dataset.v1"
MAX_FOREGROUND_ID = 254

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


@dataclass(frozen=True)
class CameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def file_provenance(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    stat = source.stat()
    return {"path": str(source), "bytes": int(stat.st_size), "sha256": sha256_file(source)}


def _validate_declared_provenance(
    declared: Any, actual: Mapping[str, Any], label: str
) -> None:
    if not isinstance(declared, Mapping):
        raise ValueError(f"{label} provenance is not an object")
    raw_bytes = declared.get("bytes")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int):
        raise ValueError(f"{label} provenance bytes is not an integer")
    declared_path = Path(str(declared.get("path") or "")).expanduser().resolve(
        strict=True
    )
    if (
        str(declared_path) != str(actual["path"])
        or raw_bytes != int(actual["bytes"])
        or str(declared.get("sha256") or "") != str(actual["sha256"])
    ):
        raise ValueError(f"{label} provenance mismatch")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    payload = stream.read(int(size))
    if len(payload) != int(size):
        raise ValueError("truncated COLMAP binary model")
    return payload


def _unpack(stream: BinaryIO, layout: str) -> tuple[Any, ...]:
    record = struct.Struct("<" + layout)
    return record.unpack(_read_exact(stream, record.size))


def _read_c_string(stream: BinaryIO) -> str:
    payload = bytearray()
    while True:
        character = _read_exact(stream, 1)
        if character == b"\0":
            return payload.decode("utf-8", errors="strict")
        payload.extend(character)
        if len(payload) > 1_048_576:
            raise ValueError("unreasonably long COLMAP image name")


def _read_colmap_binary(
    model_root: Path, selected_names: set[str]
) -> tuple[dict[int, CameraRecord], dict[str, ImageRecord], tuple[Path, Path]]:
    cameras_path = model_root / "cameras.bin"
    images_path = model_root / "images.bin"
    if not cameras_path.is_file() or not images_path.is_file():
        raise FileNotFoundError("COLMAP binary cameras.bin/images.bin are required")
    cameras: dict[int, CameraRecord] = {}
    with cameras_path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(int(count)):
            camera_id, model_id, width, height = _unpack(stream, "iiQQ")
            if int(model_id) not in _CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            model, parameter_count = _CAMERA_MODELS[int(model_id)]
            params = tuple(float(value) for value in _unpack(stream, "d" * parameter_count))
            record = CameraRecord(int(camera_id), model, int(width), int(height), params)
            if record.camera_id in cameras:
                raise ValueError(f"duplicate COLMAP camera ID {record.camera_id}")
            cameras[record.camera_id] = record
        if stream.read(1):
            raise ValueError("trailing bytes in cameras.bin")
    selected: dict[str, ImageRecord] = {}
    with images_path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(int(count)):
            values = _unpack(stream, "i" + "d" * 7 + "i")
            name = _read_c_string(stream)
            (point_count,) = _unpack(stream, "Q")
            payload_size = int(point_count) * 24
            current = stream.tell()
            if current + payload_size > os.fstat(stream.fileno()).st_size:
                raise ValueError("truncated COLMAP POINTS2D payload")
            stream.seek(payload_size, os.SEEK_CUR)
            if name not in selected_names:
                continue
            if name in selected:
                raise ValueError(f"duplicate COLMAP image name {name!r}")
            selected[name] = ImageRecord(
                image_id=int(values[0]),
                qvec=tuple(float(v) for v in values[1:5]),  # type: ignore[arg-type]
                tvec=tuple(float(v) for v in values[5:8]),  # type: ignore[arg-type]
                camera_id=int(values[8]),
                name=name,
            )
        if stream.read(1):
            raise ValueError("trailing bytes in images.bin")
    missing = sorted(selected_names.difference(selected))
    if missing:
        raise ValueError("selected episode frames missing from COLMAP: " + ", ".join(missing[:8]))
    return cameras, selected, (cameras_path, images_path)


def _float_text(value: float) -> str:
    return format(float(value), ".17g")


def render_colmap_text(
    cameras: Mapping[int, CameraRecord], images: Sequence[ImageRecord]
) -> tuple[str, str]:
    used_camera_ids = {image.camera_id for image in images}
    missing = sorted(used_camera_ids.difference(cameras))
    if missing:
        raise ValueError(f"selected images refer to missing cameras: {missing}")
    camera_lines = [
        "# Filtered by FARM for Gaussian Grouping; intrinsics preserve COLMAP doubles.",
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]",
    ]
    for camera_id in sorted(used_camera_ids):
        camera = cameras[camera_id]
        # The upstream text reader asserts PINHOLE even though a later branch
        # mentions SIMPLE_PINHOLE. Restrict here instead of emitting a dataset
        # that passes our builder but crashes in the native consumer.
        if camera.model != "PINHOLE":
            raise ValueError(
                f"Gaussian Grouping native text loader requires PINHOLE; got {camera.model}"
            )
        values = " ".join(_float_text(value) for value in camera.params)
        camera_lines.append(
            f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {values}"
        )
    image_lines = [
        "# Filtered by FARM for Gaussian Grouping; poses preserve COLMAP doubles.",
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
        "# POINTS2D intentionally omitted: Gaussian Grouping only consumes camera poses.",
    ]
    for image in sorted(images, key=lambda row: row.image_id):
        pose = " ".join(_float_text(value) for value in (*image.qvec, *image.tvec))
        image_lines.extend(
            [f"{image.image_id} {pose} {image.camera_id} {image.name}", ""]
        )
    return "\n".join(camera_lines) + "\n", "\n".join(image_lines) + "\n"


def _safe_symlink(source: Path, destination: Path) -> None:
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"symlink source is not a file: {source}")
    # Relative links survive mounting the common workspace at a different
    # container path (for example host ``.../3dgs_work`` -> ``/work``).
    os.symlink(os.path.relpath(source, start=destination.parent), destination)


def _load_episode_rows(
    episodes_root: Path, selected_episode_ids: set[str] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(episodes_root).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, "episodes manifest")
    if manifest.get("schema") != EPISODES_SCHEMA:
        raise ValueError("unsupported episodes manifest schema")
    if not bool(manifest.get("hashes_verified")):
        raise ValueError("episodes were materialized without source RGB hash verification")
    plan_path = Path(str(manifest.get("source_plan") or "")).expanduser().resolve(strict=True)
    if sha256_file(plan_path) != str(manifest.get("source_plan_sha256") or ""):
        raise ValueError("episode manifest source plan checksum mismatch")
    manifest_entries = manifest.get("episodes")
    if not isinstance(manifest_entries, list) or not manifest_entries:
        raise ValueError("episodes manifest has no episode list")
    if int(manifest.get("episode_count", -1)) != len(manifest_entries):
        raise ValueError("episodes manifest episode_count mismatch")
    if any(not isinstance(entry, Mapping) for entry in manifest_entries):
        raise ValueError("episodes manifest entry is not an object")
    available_ids = [str(entry.get("episode_id") or "") for entry in manifest_entries]
    if len(available_ids) != len(set(available_ids)):
        raise ValueError("episodes manifest contains duplicate episode IDs")
    if int(manifest.get("frame_count", -1)) != sum(
        int(entry.get("frame_count", -1)) for entry in manifest_entries
    ):
        raise ValueError("episodes manifest frame_count mismatch")
    available_id_set = set(available_ids)
    if selected_episode_ids is not None:
        missing_ids = sorted(selected_episode_ids.difference(available_id_set))
        if missing_ids:
            raise ValueError("unknown selected episode IDs: " + ", ".join(missing_ids))
        manifest_entries = [
            entry
            for entry in manifest_entries
            if str(entry.get("episode_id") or "") in selected_episode_ids
        ]
    rows: list[dict[str, Any]] = []
    for entry in manifest_entries:
        episode_id = str(entry.get("episode_id") or "")
        if not episode_id or Path(episode_id).name != episode_id:
            raise ValueError(f"unsafe episode ID {episode_id!r}")
        if str(entry.get("relative_path") or "") != episode_id:
            raise ValueError(f"episode relative_path mismatch: {episode_id}")
        episode_root = root / episode_id
        if episode_root.is_symlink() or not episode_root.is_dir():
            raise ValueError(f"episode root must be a real directory: {episode_id}")
        episode_path = episode_root / "episode.json"
        episode = _load_json(episode_path, f"episode {episode_id}")
        if episode.get("schema") != EPISODE_SCHEMA or episode.get("episode_id") != episode_id:
            raise ValueError(f"invalid episode contract: {episode_id}")
        for key in ("camera", "view_family", "split", "frame_count"):
            if episode.get(key) != entry.get(key):
                raise ValueError(f"episode/manifest {key} mismatch: {episode_id}")
        episode_plan = Path(str(episode.get("source_plan") or "")).expanduser().resolve(
            strict=True
        )
        if episode_plan != plan_path or str(episode.get("source_plan_sha256") or "") != str(
            manifest.get("source_plan_sha256") or ""
        ):
            raise ValueError(f"episode source plan provenance mismatch: {episode_id}")
        split = str(episode.get("split") or "")
        if split not in {"train", "heldout"}:
            raise ValueError(f"invalid split for {episode_id}: {split!r}")
        frames = episode.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"episode has no frames: {episode_id}")
        if len(frames) != int(episode.get("frame_count", -1)):
            raise ValueError(f"episode frame_count mismatch: {episode_id}")
        for index, frame in enumerate(frames):
            if int(frame.get("episode_index", -1)) != index:
                raise ValueError(f"non-contiguous episode index: {episode_id}")
            row = dict(frame)
            row.update(
                {
                    "episode_id": episode_id,
                    "split": split,
                    "camera": episode.get("camera"),
                    "view_family": episode.get("view_family"),
                    "episode_json": str(episode_path),
                }
            )
            rows.append(row)
    if not rows:
        raise ValueError("episodes manifest selected no frames")
    return manifest, rows


def _load_association(
    association_path: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    association = _load_json(association_path, "global association report")
    if association.get("schema") != ASSOCIATION_SCHEMA:
        raise ValueError(
            f"association schema must be {ASSOCIATION_SCHEMA!r}, got {association.get('schema')!r}"
        )
    fit_splits = association.get("fit_splits")
    if fit_splits != ["train"]:
        raise ValueError("association fit_splits must be exactly ['train'] to prevent heldout leakage")
    raw_mapping = association.get("local_to_global")
    if not isinstance(raw_mapping, Mapping):
        raise ValueError("association local_to_global must be an object")
    mapping: dict[str, int] = {}
    for key, raw_value in raw_mapping.items():
        identity_key = str(key)
        if not identity_key or identity_key in mapping:
            raise ValueError("association contains an empty/duplicate identity key")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"association global ID is not an integer: {identity_key}")
        value = int(raw_value)
        if value < 0 or value > MAX_FOREGROUND_ID:
            raise ValueError(f"association global ID outside 0..{MAX_FOREGROUND_ID}: {identity_key}")
        mapping[identity_key] = value
    if not mapping:
        raise ValueError("association local_to_global is empty")
    return mapping, association


def _mask_array(path: Path, expected_width: int, expected_height: int) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in {"L", "P"}:
            raise ValueError(f"tracker mask must be an 8-bit L/P PNG: {path}")
        values = np.asarray(image)
    if values.dtype != np.uint8 or values.ndim != 2:
        raise ValueError(f"tracker mask must be a 2D uint8 ID map: {path}")
    if values.shape != (int(expected_height), int(expected_width)):
        raise ValueError(
            f"tracker mask shape {values.shape} disagrees with COLMAP camera "
            f"{(expected_height, expected_width)}: {path}"
        )
    return np.ascontiguousarray(values)


def _validate_source_frame(frame: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_name = str(frame.get("source_name") or "")
    if (
        Path(source_name).name != source_name
        or not source_name
        or not Path(source_name).suffix
        or "." in Path(source_name).stem
    ):
        raise ValueError(f"unsafe/native-incompatible source image name: {source_name!r}")
    source = Path(str(frame.get("source_path") or "")).expanduser().resolve(strict=True)
    if source.name != source_name or not source.is_file():
        raise ValueError(f"source frame path/name mismatch: {source}")
    stat = source.stat()
    raw_bytes = frame.get("bytes")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int):
        raise ValueError(f"source frame byte count is not an integer: {source_name}")
    if raw_bytes != int(stat.st_size):
        raise ValueError(f"source frame size changed: {source_name}")
    expected_hash = str(frame.get("sha256") or "")
    actual_hash = sha256_file(source)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(f"source frame checksum changed: {source_name}")
    return source, {"path": str(source), "bytes": int(stat.st_size), "sha256": actual_hash}


def build_dataset(
    *,
    episodes_root: Path,
    tracker_runs: Mapping[str, Path],
    association_path: Path,
    colmap_model: Path,
    source_gaussians: Path,
    output_root: Path,
    sparse_points_ply: Path | None = None,
    selected_episode_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build an atomic native Gaussian Grouping dataset.

    ``tracker_runs`` maps every selected episode ID either to its tracker run
    directory (legacy/pilot layout) or to the exact detached 3D signature JSON
    named by the association report.  Detached signatures declare and hash-bind
    their original ``Annotations`` directory; no path is guessed from local IDs.
    """

    episodes_manifest, frames = _load_episode_rows(episodes_root, selected_episode_ids)
    episode_ids = {str(row["episode_id"]) for row in frames}
    if set(tracker_runs) != episode_ids:
        missing = sorted(episode_ids.difference(tracker_runs))
        extra = sorted(set(tracker_runs).difference(episode_ids))
        raise ValueError(f"tracker run coverage mismatch; missing={missing}, extra={extra}")
    mapping, association = _load_association(association_path)
    colmap_root = Path(colmap_model).expanduser().resolve(strict=True)
    selected_names = {str(row.get("source_name") or "") for row in frames}
    if "" in selected_names or len(selected_names) != len(frames):
        raise ValueError("selected frames contain empty or duplicate source names")
    cameras, images_by_name, colmap_files = _read_colmap_binary(colmap_root, selected_names)
    source_gaussians = Path(source_gaussians).expanduser().resolve(strict=True)
    if not source_gaussians.is_file():
        raise ValueError("source Gaussian PLY is not a file")
    points_source = Path(sparse_points_ply or (colmap_root / "points3D.ply")).expanduser().resolve(strict=True)
    if not points_source.is_file():
        raise ValueError("sparse points3D.ply is required")

    colmap_provenance = [file_provenance(path) for path in colmap_files]
    colmap_provenance_by_path = {row["path"]: row for row in colmap_provenance}
    source_plan_provenance = file_provenance(
        Path(str(episodes_manifest["source_plan"]))
    )
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in frames:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    observed_keys: set[str] = set()
    qc_accepted_keys: set[str] = set()
    mask_inputs: dict[
        tuple[str, int], tuple[Path, np.ndarray, dict[str, Any]]
    ] = {}
    qc_inputs: list[dict[str, Any]] = []
    qc_by_episode: dict[str, dict[str, Any]] = {}
    for episode_id in sorted(by_episode):
        tracker_source = Path(tracker_runs[episode_id]).expanduser().resolve(strict=True)
        if tracker_source.is_file():
            qc_path = tracker_source
            legacy_run_root: Path | None = None
        elif tracker_source.is_dir():
            qc_path = tracker_source / "3d_signature_v1.json"
            legacy_run_root = tracker_source
        else:
            raise ValueError(f"tracker source is neither a report nor directory: {episode_id}")
        qc = _load_json(qc_path, f"3D QC report for {episode_id}")
        if (
            qc.get("schema") != "farm.tracker-episode-3d-signatures.v1"
            or qc.get("status") != "pass"
            or (qc.get("input_contract") or {}).get("passed") is not True
        ):
            raise ValueError(f"missing/unsupported/failed 3D QC for {episode_id}")
        qc_declared_inputs = qc.get("inputs")
        if not isinstance(qc_declared_inputs, Mapping):
            raise ValueError(f"3D QC inputs are malformed: {episode_id}")
        annotation_root = Path(
            str(qc_declared_inputs.get("mask_directory") or "")
        ).expanduser().resolve(strict=True)
        if not annotation_root.is_dir():
            raise ValueError(f"tracker annotation root is not a directory: {episode_id}")
        if legacy_run_root is not None and annotation_root != (
            legacy_run_root / "Annotations"
        ).resolve(strict=True):
            raise ValueError(f"3D QC mask directory disagrees with tracker run: {episode_id}")
        episode_rows = by_episode[episode_id]
        split = str(episode_rows[0]["split"])
        episode_path = Path(str(episode_rows[0]["episode_json"]))
        episode_provenance = file_provenance(episode_path)
        _validate_declared_provenance(
            qc_declared_inputs.get("episode"), episode_provenance, f"3D QC episode {episode_id}"
        )
        if (
            str(qc_declared_inputs.get("episode_id") or "") != episode_id
            or str(qc_declared_inputs.get("split") or "") != split
            or str(qc_declared_inputs.get("camera") or "")
            != str(episode_rows[0]["camera"])
            or str(qc_declared_inputs.get("view_family") or "")
            != str(episode_rows[0]["view_family"])
            or Path(str(qc_declared_inputs.get("colmap_model") or ""))
            .expanduser()
            .resolve(strict=True)
            != colmap_root
        ):
            raise ValueError(f"3D QC episode/mask/COLMAP provenance mismatch: {episode_id}")
        _validate_declared_provenance(
            qc_declared_inputs.get("source_plan"),
            source_plan_provenance,
            f"3D QC source plan {episode_id}",
        )
        declared_colmap_rows = qc_declared_inputs.get("colmap_contract_files")
        if not isinstance(declared_colmap_rows, list):
            raise ValueError(f"3D QC COLMAP provenance is malformed: {episode_id}")
        declared_colmap_by_path: dict[str, Mapping[str, Any]] = {}
        for declared in declared_colmap_rows:
            if not isinstance(declared, Mapping):
                raise ValueError(f"3D QC COLMAP provenance is malformed: {episode_id}")
            declared_path = str(
                Path(str(declared.get("path") or "")).expanduser().resolve(strict=True)
            )
            if declared_path in declared_colmap_by_path:
                raise ValueError(f"3D QC COLMAP provenance is duplicated: {episode_id}")
            declared_colmap_by_path[declared_path] = declared
        for colmap_path, actual in colmap_provenance_by_path.items():
            if colmap_path not in declared_colmap_by_path:
                raise ValueError(f"3D QC omits COLMAP contract file: {episode_id}")
            _validate_declared_provenance(
                declared_colmap_by_path[colmap_path],
                actual,
                f"3D QC COLMAP file {episode_id}",
            )

        expected_mask_names = {str(row["materialized_name"]) for row in episode_rows}
        if any(Path(name).name != name for name in expected_mask_names):
            raise ValueError(f"unsafe tracker mask name: {episode_id}")
        actual_mask_names = {path.name for path in annotation_root.iterdir() if path.is_file()}
        if actual_mask_names != expected_mask_names:
            raise ValueError(f"tracker mask name set mismatch: {episode_id}")
        actual_mask_provenance: dict[str, dict[str, Any]] = {}
        for row in episode_rows:
            image = images_by_name[str(row["source_name"])]
            if int(row.get("colmap_image_id", -1)) != image.image_id:
                raise ValueError(f"episode/COLMAP image ID mismatch: {row['source_name']}")
            if int(row.get("colmap_camera_id", -1)) != image.camera_id:
                raise ValueError(f"episode/COLMAP camera ID mismatch: {row['source_name']}")
            camera = cameras[image.camera_id]
            mask_path = annotation_root / str(row["materialized_name"])
            mask = _mask_array(mask_path, camera.width, camera.height)
            mask_provenance = file_provenance(mask_path)
            mask_provenance["shape_hw"] = [int(mask.shape[0]), int(mask.shape[1])]
            actual_mask_provenance[mask_path.name] = mask_provenance
            for local_id in np.unique(mask):
                value = int(local_id)
                if value:
                    observed_keys.add(f"{episode_id}::local:{value}")
            mask_inputs[(episode_id, int(row["episode_index"]))] = (
                mask_path,
                mask,
                mask_provenance,
            )
        declared_masks = qc_declared_inputs.get("mask_files")
        if not isinstance(declared_masks, list):
            raise ValueError(f"3D QC mask provenance is malformed: {episode_id}")
        declared_mask_names: set[str] = set()
        for declared in declared_masks:
            if not isinstance(declared, Mapping):
                raise ValueError(f"3D QC mask provenance is malformed: {episode_id}")
            declared_path = Path(str(declared.get("path") or "")).expanduser().resolve(
                strict=True
            )
            name = declared_path.name
            if name in declared_mask_names or name not in actual_mask_provenance:
                raise ValueError(f"3D QC mask provenance name mismatch: {episode_id}")
            declared_mask_names.add(name)
            actual = actual_mask_provenance[name]
            _validate_declared_provenance(
                declared, actual, f"3D QC mask {episode_id}/{name}"
            )
            if list(declared.get("shape_hw") or []) != actual["shape_hw"]:
                raise ValueError(f"3D QC mask shape provenance mismatch: {episode_id}/{name}")
        if declared_mask_names != expected_mask_names:
            raise ValueError(f"3D QC mask provenance coverage mismatch: {episode_id}")

        identities = qc.get("local_identities")
        if not isinstance(identities, list) or any(
            not isinstance(identity, Mapping) for identity in identities
        ):
            raise ValueError(f"3D QC local identities are malformed: {episode_id}")
        qc_keys: set[str] = set()
        for identity in identities:
            local_id = identity.get("local_id")
            decision = identity.get("decision")
            if (
                isinstance(local_id, bool)
                or not isinstance(local_id, int)
                or not 1 <= local_id <= 255
                or str(identity.get("episode_id") or "") != episode_id
                or str(identity.get("identity_key") or "")
                != f"{episode_id}::local:{local_id}"
                or not isinstance(decision, Mapping)
                or not isinstance(decision.get("passed"), bool)
            ):
                raise ValueError(f"3D QC local identity provenance is malformed: {episode_id}")
            key = str(identity["identity_key"])
            if key in qc_keys:
                raise ValueError(f"duplicate 3D QC local identity: {key}")
            qc_keys.add(key)
            if decision["passed"]:
                qc_accepted_keys.add(key)
        qc_provenance = file_provenance(qc_path)
        qc_inputs.append(qc_provenance)
        qc_by_episode[episode_id] = {
            **qc_provenance,
            "episode_id": episode_id,
            "split": split,
        }
        if qc_keys != {key for key in observed_keys if key.startswith(episode_id + "::local:")}:
            raise ValueError(f"3D QC identity coverage mismatch: {episode_id}")
    mapping_keys = set(mapping)
    if mapping_keys != observed_keys:
        missing = sorted(observed_keys.difference(mapping_keys))
        extra = sorted(mapping_keys.difference(observed_keys))
        raise ValueError(f"association identity coverage mismatch; missing={missing}, extra={extra}")
    accepted_mapping_keys = {key for key, value in mapping.items() if value > 0}
    invalid_acceptance = sorted(accepted_mapping_keys.difference(qc_accepted_keys))
    if invalid_acceptance:
        raise ValueError(
            "association promoted identities rejected by episode 3D QC: "
            + ", ".join(invalid_acceptance[:8])
        )
    assignment_rows = association.get("assignments")
    if not isinstance(assignment_rows, list):
        raise ValueError("association assignments must be a complete list")
    assignment_mapping: dict[str, int] = {}
    episode_splits = {
        episode_id: str(episode_rows[0]["split"])
        for episode_id, episode_rows in by_episode.items()
    }
    for row in assignment_rows:
        if not isinstance(row, Mapping):
            raise ValueError("association assignment is not an object")
        key = str(row.get("identity_key") or "")
        if key in assignment_mapping:
            raise ValueError(f"duplicate association assignment: {key}")
        episode_id, separator, local_text = key.rpartition("::local:")
        try:
            local_id = int(local_text)
        except ValueError as exc:
            raise ValueError(f"malformed association identity key: {key!r}") from exc
        raw_global_id = row.get("global_id")
        if isinstance(raw_global_id, bool) or not isinstance(raw_global_id, int):
            raise ValueError(f"association assignment global_id is not an integer: {key}")
        if (
            not separator
            or str(row.get("episode_id") or "") != episode_id
            or isinstance(row.get("local_id"), bool)
            or row.get("local_id") != local_id
            or str(row.get("split") or "") != episode_splits.get(episode_id)
            or not str(row.get("status") or "")
            or not str(row.get("reason") or "")
        ):
            raise ValueError(f"association assignment provenance is malformed: {key}")
        assignment_mapping[key] = raw_global_id
    if assignment_mapping != mapping:
        raise ValueError("association assignments disagree with local_to_global")
    signature_rows = (association.get("inputs") or {}).get("signature_reports")
    if not isinstance(signature_rows, list):
        raise ValueError("association inputs.signature_reports must be a list")
    association_qc: dict[str, tuple[str, int, str, str, str]] = {}
    for row in signature_rows:
        if not isinstance(row, Mapping):
            raise ValueError("association signature report provenance is malformed")
        episode_id = str(row.get("episode_id") or "")
        if episode_id in association_qc:
            raise ValueError(f"duplicate association signature report: {episode_id}")
        association_qc[episode_id] = (
            str(Path(str(row.get("path") or "")).expanduser().resolve(strict=True)),
            int(row.get("bytes", -1)),
            str(row.get("sha256") or ""),
            str(row.get("schema") or ""),
            str(row.get("split") or ""),
        )
    expected_qc = {
        episode_id: (
            row["path"],
            int(row["bytes"]),
            row["sha256"],
            "farm.tracker-episode-3d-signatures.v1",
            row["split"],
        )
        for episode_id, row in qc_by_episode.items()
    }
    if association_qc != expected_qc:
        raise ValueError("association signature report provenance disagrees with tracker runs")
    used_global_ids = sorted({value for value in mapping.values() if value})
    if not used_global_ids:
        raise ValueError("association rejected every tracker identity")
    expected_ids = list(range(1, max(used_global_ids) + 1))
    if used_global_ids != expected_ids:
        raise ValueError("nonzero global IDs must be compact and contiguous from 1")
    global_rows = association.get("global_objects")
    if not isinstance(global_rows, list):
        raise ValueError("association global_objects must be a complete list")
    try:
        raw_declared_global_ids = [row["global_id"] for row in global_rows]
    except (KeyError, TypeError) as exc:
        raise ValueError("association global_objects contains a malformed global_id") from exc
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_declared_global_ids):
        raise ValueError("association global_objects contains a non-integer global_id")
    declared_global_ids = sorted(raw_declared_global_ids)
    if declared_global_ids != used_global_ids:
        raise ValueError("association global_objects disagree with local_to_global")

    train_timestamps = {
        str(row["physical_timestamp"]) for row in frames if row["split"] == "train"
    }
    heldout_timestamps = {
        str(row["physical_timestamp"]) for row in frames if row["split"] == "heldout"
    }
    leaked = sorted(train_timestamps.intersection(heldout_timestamps))
    if leaked:
        raise ValueError("physical timestamp leakage across train/heldout: " + ", ".join(leaked[:8]))
    split_counts = {
        split: sum(1 for row in frames if row["split"] == split)
        for split in ("train", "heldout")
    }
    if min(split_counts.values()) <= 0:
        raise ValueError("both train and heldout frames are required")

    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Gaussian Grouping dataset: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    output_rows: list[dict[str, Any]] = []
    try:
        for directory in ("images", "images_train", "images_heldout", "object_mask", "sparse/0"):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        _safe_symlink(source_gaussians, staging / "source_gaussians.ply")
        _safe_symlink(points_source, staging / "sparse/0/points3D.ply")
        for row in sorted(frames, key=lambda value: str(value["source_name"])):
            source, source_provenance = _validate_source_frame(row)
            source_name = str(row["source_name"])
            stem = Path(source_name).stem
            _safe_symlink(source, staging / "images" / source_name)
            _safe_symlink(source, staging / f"images_{row['split']}" / source_name)
            episode_id = str(row["episode_id"])
            mask_path, mask, mask_provenance = mask_inputs[
                (episode_id, int(row["episode_index"]))
            ]
            lut = np.zeros(256, dtype=np.uint8)
            for local_id in np.unique(mask):
                value = int(local_id)
                if value:
                    lut[value] = mapping[f"{episode_id}::local:{value}"]
            remapped = lut[mask]
            destination_mask = staging / "object_mask" / f"{stem}.png"
            Image.fromarray(remapped, mode="L").save(destination_mask, format="PNG", optimize=False)
            output_rows.append(
                {
                    "episode_id": episode_id,
                    "split": row["split"],
                    "physical_timestamp": str(row["physical_timestamp"]),
                    "source_name": source_name,
                    "colmap_image_id": int(row["colmap_image_id"]),
                    "colmap_camera_id": int(row["colmap_camera_id"]),
                    "rgb": source_provenance,
                    "tracker_mask": mask_provenance,
                    "object_mask_relative": f"object_mask/{stem}.png",
                    "object_mask_sha256": sha256_file(destination_mask),
                    "foreground_global_ids": [int(value) for value in np.unique(remapped) if value],
                }
            )
        cameras_text, images_text = render_colmap_text(
            cameras, [images_by_name[name] for name in selected_names]
        )
        cameras_output = staging / "sparse/0/cameras.txt"
        images_output = staging / "sparse/0/images.txt"
        cameras_output.write_text(cameras_text, encoding="utf-8")
        images_output.write_text(images_text, encoding="utf-8")
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "status": "ready",
            "dataset_contract": {
                "consumer": "Gaussian Grouping ECCV 2024 native COLMAP loader",
                "images": "relative symlinks; no RGB copies",
                "object_mask": "derived uint8 PNG; 0=background, 1..254=scene-global identities",
                "train_membership": "images_train exact basenames",
                "heldout_membership": "images_heldout exact basenames; excluded from training",
                "colmap": "filtered text poses/intrinsics with original IDs and IEEE-754 round-trip precision",
                "points2d": "intentionally omitted because Gaussian Grouping does not consume observations",
                "source_gaussians": "relative symlink; original order/properties preserved",
            },
            "licenses": {
                "gaussian_grouping_top_level": "Apache-2.0 repository license",
                "diff_gaussian_rasterization": "Gaussian-Splatting non-commercial research/evaluation license",
                "deva_if_used": "CC-BY-NC-SA-4.0; research/non-commercial restriction applies",
                "commercial_use": "not cleared; bundled rasterizer is non-commercial and DEVA adds CC-BY-NC-SA when used",
                "dependency_review": "other third-party model/checkpoint/dataset terms require separate review",
            },
            "artifacts": {
                "cameras_txt": {
                    "relative_path": "sparse/0/cameras.txt",
                    "bytes": cameras_output.stat().st_size,
                    "sha256": sha256_file(cameras_output),
                },
                "images_txt": {
                    "relative_path": "sparse/0/images.txt",
                    "bytes": images_output.stat().st_size,
                    "sha256": sha256_file(images_output),
                },
            },
            "inputs": {
                "episodes_manifest": file_provenance(Path(episodes_root) / "manifest.json"),
                "source_plan": source_plan_provenance,
                "association": file_provenance(association_path),
                "association_schema": association.get("schema"),
                "association_fit_splits": association.get("fit_splits"),
                "tracker_3d_qc": qc_inputs,
                "colmap_model": str(colmap_root),
                "colmap_contract_files": colmap_provenance,
                "sparse_points_ply": file_provenance(points_source),
                "source_gaussians": file_provenance(source_gaussians),
            },
            "summary": {
                "frame_count": len(output_rows),
                "episode_count": len(episode_ids),
                "frames_by_split": split_counts,
                "physical_timestamps_by_split": {
                    "train": len(train_timestamps),
                    "heldout": len(heldout_timestamps),
                },
                "physical_timestamp_leakage_count": 0,
                "observed_episode_local_identity_count": len(observed_keys),
                "accepted_scene_global_identity_count": len(used_global_ids),
                "dropped_episode_local_identity_count": sum(value == 0 for value in mapping.values()),
                "num_classes_including_background": max(used_global_ids) + 1,
                "maximum_scene_global_id": max(used_global_ids),
            },
            "frames": output_rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
