"""CPU-only preflight contracts for frozen-geometry Gaussian Grouping training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .gaussian_grouping_anchor_dataset import OUTPUT_SCHEMA as ANCHOR_OUTPUT_SCHEMA
from .gaussian_grouping_dataset import MAX_FOREGROUND_ID, OUTPUT_SCHEMA, sha256_file


REQUIRED_GAUSSIAN_PROPERTIES = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


@dataclass(frozen=True)
class PlyHeader:
    format: str
    vertex_count: int
    vertex_properties: tuple[str, ...]
    header_bytes: int


def read_ply_header(path: Path, maximum_header_bytes: int = 1_048_576) -> PlyHeader:
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("rb") as stream:
        lines: list[str] = []
        total = 0
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            total += len(raw)
            if total > int(maximum_header_bytes):
                raise ValueError("PLY header is unreasonably large")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("PLY header is not ASCII") from exc
            lines.append(line)
            if line == "end_header":
                break
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    format_name = ""
    vertex_count: int | None = None
    current_element = ""
    properties: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            format_name = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("Gaussian vertex list properties are unsupported")
            properties.append(parts[2])
    if format_name not in {"binary_little_endian", "binary_big_endian", "ascii"}:
        raise ValueError(f"unsupported PLY format {format_name!r}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY has no positive vertex count")
    if len(properties) != len(set(properties)):
        raise ValueError("PLY vertex properties are duplicated")
    return PlyHeader(format_name, vertex_count, tuple(properties), total)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Gaussian Grouping dataset manifest is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") not in {
        OUTPUT_SCHEMA,
        ANCHOR_OUTPUT_SCHEMA,
    }:
        raise ValueError("unsupported Gaussian Grouping dataset manifest")
    if value.get("status") != "ready":
        raise ValueError("Gaussian Grouping dataset is not ready")
    if value.get("schema") == ANCHOR_OUTPUT_SCHEMA:
        publication = value.get("publication")
        contract = value.get("dataset_contract")
        if (
            not isinstance(publication, dict)
            or publication.get("production_farm_identity_authorized") is not False
            or publication.get("production_gaussian_grouping_training_authorized") is not False
            or publication.get("bounded_identity_only_pilot_authorized") is not True
            or publication.get("shadow_gate_promoted_to_v1") is not False
            or not isinstance(contract, dict)
            or contract.get("direct_official_train_py_authorized") is not False
            or contract.get("heldout_membership")
            != "images_heldout exact basenames; evaluation only, never fitting"
        ):
            raise ValueError("unsafe Gaussian Grouping anchor dataset contract")
    return value


def preflight_identity_only(
    dataset_root: Path,
    gaussian_grouping_repo: Path,
    *,
    resolution_factor: int,
) -> dict[str, Any]:
    """Verify that identity-only training cannot mutate or leak source data."""

    root = Path(dataset_root).expanduser().resolve(strict=True)
    repository = Path(gaussian_grouping_repo).expanduser().resolve(strict=True)
    if resolution_factor not in {1, 2, 4, 8}:
        raise ValueError("resolution_factor must be one of 1, 2, 4, 8")
    for relative in (
        "train.py",
        "gaussian_renderer/__init__.py",
        "scene/gaussian_model.py",
        "utils/loss_utils.py",
        "LICENSE",
    ):
        if not (repository / relative).is_file():
            raise FileNotFoundError(f"incomplete Gaussian Grouping repository: {relative}")
    manifest_path = root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    summary = manifest.get("summary") or {}
    frames_by_split = summary.get("frames_by_split") or {}
    if int(frames_by_split.get("train", 0)) <= 0 or int(frames_by_split.get("heldout", 0)) <= 0:
        raise ValueError("identity-only run requires non-empty train and heldout splits")
    if int(summary.get("physical_timestamp_leakage_count", -1)) != 0:
        raise ValueError("dataset reports train/heldout physical timestamp leakage")
    maximum_id = int(summary.get("maximum_scene_global_id", -1))
    num_classes = int(summary.get("num_classes_including_background", -1))
    if maximum_id < 1 or maximum_id > MAX_FOREGROUND_ID:
        raise ValueError("dataset maximum scene-global ID is outside 1..254")
    if num_classes != maximum_id + 1:
        raise ValueError("dataset num_classes does not match maximum global ID + background")
    source_link = root / "source_gaussians.ply"
    if not source_link.is_symlink() or source_link.readlink().is_absolute():
        raise ValueError("source Gaussian PLY must remain a relative symlink")
    source_gaussians = source_link.resolve(strict=True)
    expected_source = manifest.get("inputs", {}).get("source_gaussians") or {}
    if int(source_gaussians.stat().st_size) != int(expected_source.get("bytes", -1)):
        raise ValueError("source Gaussian PLY size disagrees with manifest")
    source_hash = sha256_file(source_gaussians)
    if source_hash != str(expected_source.get("sha256") or ""):
        raise ValueError("source Gaussian PLY checksum mismatch")
    ply_header = read_ply_header(source_gaussians)
    property_names = set(ply_header.vertex_properties)
    missing = sorted(REQUIRED_GAUSSIAN_PROPERTIES.difference(property_names))
    rest_names = sorted(
        (name for name in property_names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if missing:
        raise ValueError("source PLY lacks Gaussian properties: " + ", ".join(missing))
    if len(rest_names) != 45 or rest_names != [f"f_rest_{index}" for index in range(45)]:
        raise ValueError("source PLY must contain the complete degree-3 SH f_rest_0..44 set")
    frame_rows = manifest.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) != int(summary.get("frame_count", -1)):
        raise ValueError("dataset frame manifest/count mismatch")
    expected_split_names: dict[str, set[str]] = {"train": set(), "heldout": set()}
    image_shapes: set[tuple[int, int]] = set()
    observed_ids: set[int] = set()
    observed_ids_by_split: dict[str, set[int]] = {"train": set(), "heldout": set()}
    for row in frame_rows:
        if not isinstance(row, dict):
            raise ValueError("dataset frame entry is not an object")
        split = str(row.get("split") or "")
        source_name = str(row.get("source_name") or "")
        if split not in expected_split_names or Path(source_name).name != source_name:
            raise ValueError("invalid frame split/name")
        expected_split_names[split].add(source_name)
        image_path = root / "images" / source_name
        if (
            not image_path.is_symlink()
            or image_path.readlink().is_absolute()
            or not image_path.resolve(strict=True).is_file()
        ):
            raise ValueError(f"RGB must be a valid relative symlink: {source_name}")
        if sha256_file(image_path) != str(row.get("rgb", {}).get("sha256") or ""):
            raise ValueError(f"RGB checksum mismatch: {source_name}")
        mask_path = root / str(row.get("object_mask_relative") or "")
        if not mask_path.is_file() or sha256_file(mask_path) != str(row.get("object_mask_sha256") or ""):
            raise ValueError(f"object mask checksum mismatch: {source_name}")
        with Image.open(mask_path) as mask:
            if mask.mode != "L" or mask.format != "PNG":
                raise ValueError(f"object mask is not an L-mode PNG: {source_name}")
            width, height = mask.size
            ids = set(int(value) for value in np.unique(np.asarray(mask, dtype=np.uint8)))
        image_shapes.add((width, height))
        observed_ids.update(ids)
        observed_ids_by_split[split].update(ids)
    for split, names in expected_split_names.items():
        directory = root / f"images_{split}"
        actual = {path.name for path in directory.iterdir() if path.is_symlink()}
        if actual != names:
            raise ValueError(f"images_{split} membership disagrees with manifest")
    if any(value < 0 or value > maximum_id for value in observed_ids):
        raise ValueError("object mask contains an undeclared global ID")
    expected_foreground_ids = set(range(1, maximum_id + 1))
    train_foreground_ids = observed_ids_by_split["train"].difference({0})
    heldout_foreground_ids = observed_ids_by_split["heldout"].difference({0})
    if 0 not in observed_ids_by_split["train"]:
        raise ValueError("training masks must contain background pixels with ID 0")
    if train_foreground_ids != expected_foreground_ids:
        raise ValueError("every declared scene-global ID must have train-mask pixels")
    if not heldout_foreground_ids:
        raise ValueError("heldout masks must contain at least one frozen global identity")
    if not heldout_foreground_ids.issubset(train_foreground_ids):
        raise ValueError("heldout masks contain identities absent from the frozen train mapping")
    artifacts = manifest.get("artifacts") or {}
    for artifact_name in ("cameras_txt", "images_txt"):
        provenance = artifacts.get(artifact_name) or {}
        relative_path = str(provenance.get("relative_path") or "")
        artifact_path = root / relative_path
        if not relative_path or not artifact_path.is_file():
            raise ValueError(f"filtered COLMAP artifact is missing: {artifact_name}")
        if artifact_path.stat().st_size != int(provenance.get("bytes", -1)):
            raise ValueError(f"filtered COLMAP artifact size mismatch: {artifact_name}")
        if sha256_file(artifact_path) != str(provenance.get("sha256") or ""):
            raise ValueError(f"filtered COLMAP artifact checksum mismatch: {artifact_name}")
    sparse_link = root / "sparse/0/points3D.ply"
    if not sparse_link.is_symlink() or sparse_link.readlink().is_absolute():
        raise ValueError("sparse points3D.ply must remain a relative symlink")
    sparse_source = sparse_link.resolve(strict=True)
    sparse_provenance = manifest.get("inputs", {}).get("sparse_points_ply") or {}
    if sparse_source.stat().st_size != int(sparse_provenance.get("bytes", -1)):
        raise ValueError("sparse points3D.ply size disagrees with manifest")
    if sha256_file(sparse_source) != str(sparse_provenance.get("sha256") or ""):
        raise ValueError("sparse points3D.ply checksum mismatch")
    return {
        "status": "ready",
        "mode": "identity-only-frozen-geometry",
        "dataset_root": str(root),
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "gaussian_grouping_repo": str(repository),
        "source_gaussians": {
            "path": str(source_gaussians),
            "bytes": int(source_gaussians.stat().st_size),
            "sha256": source_hash,
            "vertex_count": ply_header.vertex_count,
            "format": ply_header.format,
        },
        "frame_count": len(frame_rows),
        "num_classes": num_classes,
        "frames_by_split": {key: len(value) for key, value in expected_split_names.items()},
        "native_image_shapes_wh": [list(shape) for shape in sorted(image_shapes)],
        "resolution_factor": resolution_factor,
        "maximum_global_id": maximum_id,
        "geometry_trainable": False,
        "appearance_trainable": False,
        "opacity_trainable": False,
        "identity_features_trainable": True,
        "classifier_trainable": True,
        "densification_enabled": False,
        "source_order_preserved": True,
        "combined_ply_written_by_default": False,
        "license_note": (
            "Not commercially cleared: the bundled Gaussian rasterizer is research/evaluation-only; "
            "DEVA-derived masks, if included, additionally remain CC-BY-NC-SA-4.0."
        ),
    }
