"""Canonical, immutable union of independently rendered heldout folds.

The builder intentionally performs no fitting and no mask refinement.  It
normalizes the frames which were actually consumed by fitting, unions frozen
heldout folds, and re-keys the read-only YOLOE/SAM3 evidence so that the
existing frozen heldout evaluator can consume one canonical dataset.
"""

from __future__ import annotations

import copy
import filecmp
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from farm_runtime.frozen_heldout_evidence import (
    REFINEMENT_ROLE,
    REFINEMENT_SCHEMA,
    _load_mapping_state,
    _mapping_image_sources,
    _resolve_mapping_mask,
    _timestamp_key,
    _validate_reference_targets,
)
from farm_runtime.frozen_heldout_qc import (
    _load_mask,
    _validate_fold_manifest,
    sha256_file,
)
from farm_runtime.full_colmap_folds import (
    FOLD_SCHEMA,
    FRAMES_SCHEMA,
    MANIFEST_SCHEMA,
)

UNION_SCHEMA = "farm.frozen-evaluation-union.v1"
MAPPING_SCHEMA = "farm.frozen-evaluation-yoloe-union.v1"
SUCCESS_SCHEMA = "farm.frozen-evaluation-union.success.v1"
RGBD_UNION_MANIFEST_SCHEMA = "farm.frozen-evaluation-rgbd-union-manifest.v1"
_RENDER_CONFIG_FIELDS = (
    "scene_id",
    "identity_regex",
    "sensor_group",
    "timestamp_group",
    "family_group",
    "meters_per_scene_unit",
    "nominal_hz",
    "resolution",
    "sh_degree",
    "alpha_min",
    "depth_min_m",
    "depth_max_m",
    "radius_clip",
    "jpeg_quality",
    "expected_baseline_m",
    "baseline_tolerance_m",
    "baseline_camera_ids",
    "baseline_sensors",
)
_SOURCE_TIMESTAMP = re.compile(
    r"(?:^|/)(?:cam[0-9]+)_(?P<timestamp>[0-9]+)_"
    r"(?:center|yaw_left|yaw_right|pitch_up|pitch_down)(?:[.][^.]+)?$"
)


@dataclass(frozen=True)
class HeldoutBundle:
    fold_manifest: Path
    reference_result: Path
    mapping_state: Path
    mask_root: Path
    rgbd_render_manifest: Path
    rgbd_union_frames: Path


@dataclass
class _LoadedBundle:
    key: str
    spec: HeldoutBundle
    manifest: dict[str, Any]
    train: dict[str, dict[str, Any]]
    heldout: dict[str, dict[str, Any]]
    membership: dict[int, dict[str, list[str]]]
    train_frames_path: Path
    heldout_frames_path: Path
    mapping: dict[str, Any]
    mapping_sources: list[str]
    mapping_by_source: dict[str, int]
    targets: dict[tuple[int, str], dict[str, Any]]
    accepted_ids: set[int]
    rgbd_render_manifest: dict[str, Any]
    rgbd_union_frames: dict[str, Any]
    rgbd_render_manifest_path: Path
    rgbd_union_frames_path: Path
    source_hashes: dict[Path, str]


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_values(values: Sequence[str]) -> str:
    canonical = json.dumps(list(values), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = (
        os.path.relpath(resolved, relative_to).replace(os.sep, "/")
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": value,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _resolve_frame_artifact(
    row: Mapping[str, Any], frames_path: Path, field: str
) -> Path:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"frame {row.get('source_image')} has no {field}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = frames_path.parent / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"frame {field} is not a regular file: {path}")
    return path


def _timestamp_from_frame(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_image") or "").strip()
    if not source:
        raise ValueError("frame has no source_image")
    match = _SOURCE_TIMESTAMP.search(source)
    source_timestamp = match.group("timestamp") if match else ""
    declared = str(row.get("physical_timestamp") or "").strip()
    if declared and source_timestamp:
        if _timestamp_key(declared) != _timestamp_key(source_timestamp):
            raise ValueError(f"physical timestamp contradicts source image: {source}")
    timestamp = declared or source_timestamp or str(row.get("frame_id") or "").strip()
    if not timestamp:
        raise ValueError(f"cannot derive physical timestamp for {source}")
    _timestamp_key(timestamp)
    return timestamp


def _frame_signature(
    row: Mapping[str, Any], frames_path: Path
) -> tuple[dict[str, Any], Path, Path]:
    rgb = _resolve_frame_artifact(row, frames_path, "rgb_path")
    depth = _resolve_frame_artifact(row, frames_path, "depth_path")
    timestamp = _timestamp_from_frame(row)
    signature = {
        "physical_timestamp": _timestamp_key(timestamp),
        "rgb_sha256": sha256_file(rgb),
        "depth_sha256": sha256_file(depth),
        "depth_size": copy.deepcopy(row.get("depth_size")),
        "K": copy.deepcopy(row.get("K")),
        "T_world_cam": copy.deepcopy(row.get("T_world_cam")),
        "camera": row.get("camera"),
        "colmap_camera_id": row.get("colmap_camera_id"),
        "colmap_image_id": row.get("colmap_image_id"),
    }
    return signature, rgb, depth


def _output_frame(
    row: Mapping[str, Any],
    *,
    frames_path: Path,
    output_frames_dir: Path,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature, rgb, depth = _frame_signature(row, frames_path)
    output = copy.deepcopy(dict(row))
    output["physical_timestamp"] = _timestamp_from_frame(row)
    output["full_colmap_split"] = role
    output["rgb_path"] = os.path.relpath(rgb, output_frames_dir).replace(os.sep, "/")
    output["depth_path"] = os.path.relpath(depth, output_frames_dir).replace(
        os.sep, "/"
    )
    provenance = {
        "source_image": str(output["source_image"]),
        "physical_timestamp": str(output["physical_timestamp"]),
        "rgb": _artifact(rgb, relative_to=output_frames_dir),
        "depth": _artifact(depth, relative_to=output_frames_dir),
    }
    return output, {**signature, "rgb": rgb, "depth": depth, "provenance": provenance}


def _fit_metadata_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    exact_fields = (
        "physical_timestamp",
        "depth_size",
        "camera",
        "colmap_camera_id",
        "colmap_image_id",
    )
    if any(left.get(field) != right.get(field) for field in exact_fields):
        return False
    for field in ("K", "T_world_cam"):
        try:
            left_array = np.asarray(left.get(field), dtype=np.float64)
            right_array = np.asarray(right.get(field), dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if left_array.shape != right_array.shape or not np.allclose(
            left_array, right_array, rtol=0.0, atol=1.0e-12
        ):
            return False
    return True


def _rgb_semantically_equal(left: Path, right: Path) -> bool:
    if sha256_file(left) == sha256_file(right):
        return True
    left_pixels = cv2.imread(str(left), cv2.IMREAD_UNCHANGED)
    right_pixels = cv2.imread(str(right), cv2.IMREAD_UNCHANGED)
    return (
        left_pixels is not None
        and right_pixels is not None
        and left_pixels.shape == right_pixels.shape
        and np.array_equal(left_pixels, right_pixels)
    )


def _array_artifact(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(
                    f"semantic depth comparison requires single-array NPZ: {path}"
                )
            return np.asarray(archive[archive.files[0]])
    raise ValueError(f"semantic depth comparison requires NPY/NPZ: {path}")


def _depth_semantically_equal(left: Path, right: Path) -> bool:
    if sha256_file(left) == sha256_file(right):
        return True
    left_depth = _array_artifact(left)
    right_depth = _array_artifact(right)
    return left_depth.shape == right_depth.shape and np.allclose(
        left_depth,
        right_depth,
        rtol=1.0e-7,
        atol=1.0e-7,
        equal_nan=True,
    )


def _fit_variant_provenance(
    row: Mapping[str, Any], meta: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "full_colmap_split": row.get("full_colmap_split"),
        "rgb": _artifact(Path(meta["rgb"])),
        "depth": _artifact(Path(meta["depth"])),
    }


def _accepted_object_ids(payload: Mapping[str, Any]) -> set[int]:
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise ValueError("SAM3 heldout-reference result has no object rows")
    result: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("SAM3 result contains a non-object row")
        object_id = int(row.get("object_id"))
        if row.get("accepted") is True:
            if object_id in result:
                raise ValueError(f"SAM3 result repeats accepted object {object_id}")
            result.add(object_id)
    return result


def _source_rgbd_contract(
    manifest_path: Path,
    union_frames_path: Path,
    *,
    train: Mapping[str, Mapping[str, Any]],
    heldout: Mapping[str, Mapping[str, Any]],
    train_frames_path: Path,
    heldout_frames_path: Path,
    scale: float,
    hashes: dict[Path, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one completed RGB-D union behind a frozen fold."""

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    union_frames_path = union_frames_path.expanduser().resolve(strict=True)
    if manifest_path.parent != union_frames_path.parent:
        raise ValueError("RGB-D run manifest and union frames must share a directory")
    manifest = _json_object(manifest_path)
    union = _json_object(union_frames_path)
    if (
        manifest.get("schema_version") != "farm_rgbd_run_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("qa_passed") is not True
        or str(manifest.get("frames") or "") != union_frames_path.name
    ):
        raise ValueError("source RGB-D union is not a completed QA-passing v1 run")
    if (
        union.get("schema_version") != FRAMES_SCHEMA
        or union.get("depth_units") != "metres"
        or union.get("pose_translation_units") != "metres"
    ):
        raise ValueError("source RGB-D union frames contract is invalid")
    if (
        not str(manifest.get("scene_id") or "").strip()
        or str(manifest.get("scene_id") or "") != str(union.get("scene_id") or "")
        or not math.isclose(
            float(union.get("meters_per_scene_unit", 0.0)),
            scale,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("source RGB-D union scene/scale mismatch")
    rows = union.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source RGB-D union frames are empty")
    union_by_source: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("source RGB-D union contains a non-object frame")
        source = str(raw.get("source_image") or "").strip()
        if not source or source in union_by_source:
            raise ValueError("source RGB-D union has empty/duplicate source images")
        union_by_source[source] = raw
    expected_sources = set(train) | set(heldout)
    if set(union_by_source) != expected_sources:
        raise ValueError("source RGB-D union and frozen fold frame sets differ")
    if int(manifest.get("selected_view_count", -1)) != len(rows):
        raise ValueError("source RGB-D union selected-view count mismatch")

    frames_sha = sha256_file(union_frames_path)
    for fold_path in (train_frames_path, heldout_frames_path):
        fold_document = _json_object(fold_path)
        fold_contract = fold_document.get("full_colmap_fold")
        if (
            not isinstance(fold_contract, Mapping)
            or str(fold_contract.get("source_union_frames_sha256") or "").lower()
            != frames_sha
        ):
            raise ValueError(
                "frozen fold does not bind the supplied RGB-D union frames"
            )

    for source, fold_row in [*train.items(), *heldout.items()]:
        fold_path = train_frames_path if source in train else heldout_frames_path
        fold_signature, fold_rgb, fold_depth = _frame_signature(fold_row, fold_path)
        union_signature, union_rgb, union_depth = _frame_signature(
            union_by_source[source], union_frames_path
        )
        if fold_signature != union_signature:
            raise ValueError(
                f"frozen fold calibration/artifacts differ from RGB-D union: {source}"
            )
        if fold_rgb != union_rgb or fold_depth != union_depth:
            raise ValueError(f"frozen fold paths differ from RGB-D union: {source}")
        hashes[union_rgb] = union_signature["rgb_sha256"]
        hashes[union_depth] = union_signature["depth_sha256"]

    payload = manifest.get("fingerprint_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("source RGB-D union lacks fingerprint payload")
    declared_fingerprint = str(manifest.get("fingerprint") or "").lower()
    computed_fingerprint = hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    source_names = [str(row["source_image"]) for row in rows]
    selected_digest = hashlib.sha256(
        "\n".join(source_names).encode("utf-8")
    ).hexdigest()
    selected_count = payload.get("selected_count")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", declared_fingerprint)
        or declared_fingerprint != computed_fingerprint
        or isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count != len(rows)
        or str(payload.get("selected_names_sha256") or "").lower() != selected_digest
    ):
        raise ValueError("source RGB-D union fingerprint/view selection mismatch")
    config = payload.get("config") if isinstance(payload, Mapping) else None
    inputs = payload.get("inputs") if isinstance(payload, Mapping) else None
    ply = inputs.get("ply") if isinstance(inputs, Mapping) else None
    if not isinstance(config, Mapping) or not isinstance(ply, Mapping):
        raise ValueError("source RGB-D union lacks render config/PLY provenance")
    missing_config = [field for field in _RENDER_CONFIG_FIELDS if field not in config]
    if missing_config:
        raise ValueError(f"source RGB-D render config is incomplete: {missing_config}")
    if not math.isclose(
        float(config.get("meters_per_scene_unit", 0.0)),
        scale,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("source RGB-D render config scale mismatch")
    try:
        ply_size = int(ply.get("size", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("source RGB-D PLY size is invalid") from exc
    if ply_size < 1:
        raise ValueError("source RGB-D PLY size is invalid")
    hashes[manifest_path] = sha256_file(manifest_path)
    hashes[union_frames_path] = frames_sha
    return manifest, union


def _load_bundle(spec: HeldoutBundle, *, scale: float) -> _LoadedBundle:
    resolved = HeldoutBundle(
        *(
            Path(value).expanduser().resolve(strict=True)
            for value in (
                spec.fold_manifest,
                spec.reference_result,
                spec.mapping_state,
                spec.mask_root,
                spec.rgbd_render_manifest,
                spec.rgbd_union_frames,
            )
        )
    )
    if not resolved.mask_root.is_dir():
        raise ValueError(f"mapping mask root is not a directory: {resolved.mask_root}")
    hashes: dict[Path, str] = {}
    (
        manifest,
        train,
        heldout,
        membership,
        _depth_specs,
        train_frames_path,
        heldout_frames_path,
    ) = _validate_fold_manifest(
        resolved.fold_manifest, meters_per_scene_unit=scale, hashes=hashes
    )
    rgbd_render_manifest, rgbd_union_frames = _source_rgbd_contract(
        resolved.rgbd_render_manifest,
        resolved.rgbd_union_frames,
        train=train,
        heldout=heldout,
        train_frames_path=train_frames_path,
        heldout_frames_path=heldout_frames_path,
        scale=scale,
        hashes=hashes,
    )
    mapping = _load_mapping_state(resolved.mapping_state)
    mapping_sources, mapping_by_source = _mapping_image_sources(mapping, heldout)
    mapping_sha = sha256_file(resolved.mapping_state)
    hashes[resolved.mapping_state] = mapping_sha
    reference = _json_object(resolved.reference_result)
    accepted_ids = _accepted_object_ids(reference)
    if not accepted_ids:
        raise ValueError(
            "bundle has no object-level accepted SAM3 targets: "
            f"{resolved.reference_result}"
        )
    missing_membership = sorted(accepted_ids.difference(membership))
    if missing_membership:
        raise ValueError(
            f"accepted SAM3 objects lack fold membership: {missing_membership}"
        )
    targets = _validate_reference_targets(
        resolved.reference_result,
        heldout_frames_sha=sha256_file(heldout_frames_path),
        mapping_state_sha=mapping_sha,
        candidate_ids=accepted_ids,
        heldout=heldout,
        membership=membership,
        hashes=hashes,
    )
    missing_targets = sorted(
        object_id
        for object_id in accepted_ids
        if not any(key[0] == object_id for key in targets)
    )
    if missing_targets:
        raise ValueError(
            f"accepted SAM3 objects have no accepted views: {missing_targets}"
        )
    key = str(resolved.fold_manifest)
    return _LoadedBundle(
        key=key,
        spec=resolved,
        manifest=manifest,
        train=train,
        heldout=heldout,
        membership=membership,
        train_frames_path=train_frames_path,
        heldout_frames_path=heldout_frames_path,
        mapping=mapping,
        mapping_sources=mapping_sources,
        mapping_by_source=mapping_by_source,
        targets=targets,
        accepted_ids=accepted_ids,
        rgbd_render_manifest=rgbd_render_manifest,
        rgbd_union_frames=rgbd_union_frames,
        rgbd_render_manifest_path=resolved.rgbd_render_manifest,
        rgbd_union_frames_path=resolved.rgbd_union_frames,
        source_hashes=hashes,
    )


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(
            f"cannot hardlink immutable mask {source} to {destination}; "
            "source and output must share a filesystem"
        ) from exc
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"hardlinked mask changed bytes: {destination}")


def _write_frames(
    path: Path,
    *,
    source_document: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    role: str,
    scale: float,
    source_fit_sha: str,
    source_union_frames_sha: str,
    source_bundle_shas: Sequence[str],
) -> dict[str, Any]:
    document = {
        key: copy.deepcopy(value)
        for key, value in source_document.items()
        if key not in {"frames", "full_colmap_fold"}
    }
    document.update(
        {
            "schema_version": FRAMES_SCHEMA,
            "depth_units": "metres",
            "meters_per_scene_unit": scale,
            "frames": [copy.deepcopy(dict(row)) for row in rows],
            "full_colmap_fold": {
                "schema": FOLD_SCHEMA,
                "role": role,
                "state_fit_authorized": role == "train",
                "usage": (
                    "mapping_covisibility_sam3_merge_obb_fit"
                    if role == "train"
                    else "frozen_evaluation_only"
                ),
                "evaluation_status": (
                    "not_applicable" if role == "train" else "reserved_not_consumed"
                ),
                "source_plan_sha256": _sha256_values(source_bundle_shas),
                "source_union_frames_sha256": source_union_frames_sha,
                "source_fit_frames_sha256": source_fit_sha,
                "source_fold_manifest_sha256": list(source_bundle_shas),
                "source_image_count": len(rows),
                "physical_timestamp_count": len(
                    {_timestamp_key(row["physical_timestamp"]) for row in rows}
                ),
            },
        }
    )
    _atomic_json(path, document)
    return document


def build_frozen_evaluation_union(
    fit_frames_path: Path,
    bundles: Sequence[HeldoutBundle],
    source_ply_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish one canonical fold/mapping/reference union, or fail closed."""

    fit_path = Path(fit_frames_path).expanduser().resolve(strict=True)
    source_ply = Path(source_ply_path).expanduser().resolve(strict=True)
    if not source_ply.is_file():
        raise ValueError("source PLY is not a regular file")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not bundles:
        raise ValueError("at least one heldout bundle is required")
    fit_document = _json_object(fit_path)
    if fit_document.get("schema_version") != FRAMES_SCHEMA:
        raise ValueError("actual fit frames use an unsupported schema")
    if (
        fit_document.get("depth_units") != "metres"
        or fit_document.get("pose_translation_units") != "metres"
    ):
        raise ValueError("actual fit depth/pose units must be metres")
    if not str(fit_document.get("scene_id") or "").strip():
        raise ValueError("actual fit scene_id must be non-empty")
    scale = float(fit_document.get("meters_per_scene_unit"))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("actual fit meters_per_scene_unit must be finite and positive")
    raw_fit_rows = fit_document.get("frames")
    if not isinstance(raw_fit_rows, list) or not raw_fit_rows:
        raise ValueError("actual fit frames must be a non-empty list")

    loaded = sorted(
        (_load_bundle(bundle, scale=scale) for bundle in bundles),
        key=lambda item: item.key,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    source_scene_ids = {
        str(bundle.rgbd_render_manifest.get("scene_id") or "") for bundle in loaded
    }
    if source_scene_ids != {str(fit_document.get("scene_id") or "")}:
        raise ValueError("source RGB-D unions and actual fit scene differ")
    render_configs = []
    source_ply_sizes = []
    for bundle in loaded:
        payload = bundle.rgbd_render_manifest["fingerprint_payload"]
        config = payload["config"]
        render_configs.append(
            {field: copy.deepcopy(config[field]) for field in _RENDER_CONFIG_FIELDS}
        )
        source_ply_sizes.append(int(payload["inputs"]["ply"]["size"]))
    canonical_render_config = render_configs[0]
    if any(config != canonical_render_config for config in render_configs[1:]):
        raise ValueError("source RGB-D union render configurations differ")
    if set(source_ply_sizes) != {source_ply.stat().st_size}:
        raise ValueError("source RGB-D union PLY size differs from supplied source PLY")
    source_ply_artifact = {
        "path": str(source_ply),
        "bytes": source_ply.stat().st_size,
        "sha256": sha256_file(source_ply),
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "train").mkdir()
        (temporary / "heldout").mkdir()
        final_train_dir = output / "train"
        final_heldout_dir = output / "heldout"

        train_rows: list[dict[str, Any]] = []
        train_meta: dict[str, dict[str, Any]] = {}
        train_index: dict[str, int] = {}
        fit_dedup_by_source: dict[str, dict[str, Any]] = {}
        declared_bundle_train_sources = set().union(
            *(set(bundle.train) for bundle in loaded)
        )
        for raw in raw_fit_rows:
            if not isinstance(raw, Mapping):
                raise ValueError("actual fit frames contain a non-object row")
            row, meta = _output_frame(
                raw,
                frames_path=fit_path,
                output_frames_dir=final_train_dir,
                role="train",
            )
            source = str(row.get("source_image") or "").strip()
            if not source:
                raise ValueError("actual fit frame has no source_image")
            previous = train_meta.get(source)
            if previous is not None:
                calibration_equal = _fit_metadata_equivalent(previous, meta)
                rgb_equal = _rgb_semantically_equal(previous["rgb"], meta["rgb"])
                depth_equal = _depth_semantically_equal(
                    previous["depth"], meta["depth"]
                )
                if not calibration_equal or not rgb_equal or not depth_equal:
                    raise ValueError(
                        "conflicting duplicate actual fit frame: "
                        f"{source} (calibration={calibration_equal}, "
                        f"rgb={rgb_equal}, depth={depth_equal})"
                    )
                audit = fit_dedup_by_source.setdefault(
                    source,
                    {
                        "source_image": source,
                        "selection_policy": (
                            "prefer full_colmap_split=train only with source membership, "
                            "equivalent calibration, RGB and depth"
                        ),
                        "calibration_atol": 1.0e-12,
                        "depth_rtol": 1.0e-7,
                        "depth_atol": 1.0e-7,
                        "candidates": [
                            _fit_variant_provenance(
                                train_rows[train_index[source]], previous
                            )
                        ],
                    },
                )
                audit["candidates"].append(_fit_variant_provenance(row, meta))
                previous_is_preferred = (
                    train_rows[train_index[source]].get("full_colmap_split") == "train"
                    and source in declared_bundle_train_sources
                )
                new_is_preferred = (
                    row.get("full_colmap_split") == "train"
                    and source in declared_bundle_train_sources
                )
                if new_is_preferred and not previous_is_preferred:
                    train_rows[train_index[source]] = row
                    train_meta[source] = meta
                continue
            train_meta[source] = meta
            train_index[source] = len(train_rows)
            train_rows.append(row)

        for source, audit in fit_dedup_by_source.items():
            selected = train_meta[source]
            audit["selected"] = _fit_variant_provenance(
                train_rows[train_index[source]], selected
            )
            audit["semantic_equivalence_verified"] = True

        heldout_rows_by_source: dict[str, dict[str, Any]] = {}
        heldout_meta: dict[str, dict[str, Any]] = {}
        for bundle in loaded:
            for source, raw in bundle.heldout.items():
                row, meta = _output_frame(
                    raw,
                    frames_path=bundle.heldout_frames_path,
                    output_frames_dir=final_heldout_dir,
                    role="heldout",
                )
                previous = heldout_meta.get(source)
                if previous is not None:
                    if (
                        meta["rgb_sha256"] != previous["rgb_sha256"]
                        or meta["depth_sha256"] != previous["depth_sha256"]
                    ):
                        raise ValueError(
                            f"conflicting duplicate heldout RGB/depth: {source}"
                        )
                    excluded = {
                        "rgb",
                        "depth",
                        "provenance",
                        "rgb_sha256",
                        "depth_sha256",
                    }
                    comparable = {
                        key: value for key, value in meta.items() if key not in excluded
                    }
                    old_comparable = {
                        key: value
                        for key, value in previous.items()
                        if key not in excluded
                    }
                    if comparable != old_comparable:
                        raise ValueError(
                            f"conflicting duplicate heldout frame metadata: {source}"
                        )
                else:
                    heldout_rows_by_source[source] = row
                    heldout_meta[source] = meta
        train_sources = set(train_meta)
        heldout_sources = set(heldout_meta)
        source_leak = sorted(train_sources.intersection(heldout_sources))
        train_timestamps = {meta["physical_timestamp"] for meta in train_meta.values()}
        heldout_timestamps = {
            meta["physical_timestamp"] for meta in heldout_meta.values()
        }
        timestamp_leak = sorted(train_timestamps.intersection(heldout_timestamps))
        if source_leak or timestamp_leak:
            raise ValueError(
                "train/heldout leakage: "
                f"sources={source_leak[:8]}, physical_timestamps={timestamp_leak[:8]}"
            )
        heldout_rows = [
            heldout_rows_by_source[name] for name in sorted(heldout_rows_by_source)
        ]
        heldout_index = {
            str(row["source_image"]): index for index, row in enumerate(heldout_rows)
        }

        accepted_ids = sorted(set().union(*(bundle.accepted_ids for bundle in loaded)))
        membership: dict[int, dict[str, list[str]]] = {}
        for object_id in accepted_ids:
            train_names: set[str] = set()
            heldout_names: set[str] = set()
            for bundle in loaded:
                row = bundle.membership.get(object_id)
                if row is None:
                    continue
                train_names.update(
                    name for name in row["train"] if name in train_sources
                )
                heldout_names.update(
                    name for name in row["heldout"] if name in heldout_sources
                )
            membership[object_id] = {
                "train": sorted(train_names),
                "heldout": sorted(heldout_names),
            }

        # Bind local mapping tracks to canonical FARM object IDs only through
        # accepted SAM3 evidence.  Unbound competitor tracks remain distinct.
        track_binding: dict[tuple[str, int], int] = {}
        for bundle in loaded:
            observations = bundle.mapping.get("object_mask_observations")
            if not isinstance(observations, (list, tuple)):
                raise ValueError(
                    "YOLOE mapping lacks canonical object_mask_observations"
                )
            for target in bundle.targets.values():
                image_id = int(target["rescue_image_id"])
                source = str(target["source_image"])
                if bundle.mapping_by_source.get(source) != image_id:
                    raise ValueError(
                        "SAM3 rescue_image_id differs from source YOLOE mapping"
                    )
                raw_track = target.get("rescue_track_index")
                if raw_track is None:
                    continue
                track = int(raw_track)
                if track < 0 or track >= len(observations):
                    raise ValueError("SAM3 rescue_track_index is outside YOLOE tracks")
                key = (bundle.key, track)
                object_id = int(target["object_id"])
                prior = track_binding.get(key)
                if prior is not None and prior != object_id:
                    raise ValueError(
                        "one YOLOE track is bound to multiple canonical objects"
                    )
                track_binding[key] = object_id

        bound_object_ids = sorted(set(track_binding.values()))
        canonical_track = {
            object_id: index for index, object_id in enumerate(bound_object_ids)
        }
        track_remap: dict[tuple[str, int], int] = {
            key: canonical_track[object_id] for key, object_id in track_binding.items()
        }
        next_track = len(canonical_track)
        for bundle in loaded:
            observations = bundle.mapping["object_mask_observations"]
            for old_track in range(len(observations)):
                key = (bundle.key, old_track)
                if key not in track_remap:
                    track_remap[key] = next_track
                    next_track += 1

        mapping_observations: list[list[dict[str, Any]]] = [
            [] for _ in range(next_track)
        ]
        observation_by_key: dict[tuple[int, int], tuple[str, Path]] = {}
        mapping_mask_provenance: list[dict[str, Any]] = []
        image_remap_provenance: list[dict[str, Any]] = []
        track_remap_provenance: list[dict[str, Any]] = []
        for bundle in loaded:
            image_remap_provenance.extend(
                {
                    "bundle_fold_manifest": bundle.key,
                    "source_image": source,
                    "source_image_id": old_image,
                    "union_image_id": heldout_index[source],
                }
                for old_image, source in enumerate(bundle.mapping_sources)
            )
            observations = bundle.mapping["object_mask_observations"]
            for old_track, track_rows in enumerate(observations):
                if track_rows is None:
                    continue
                if not isinstance(track_rows, (list, tuple)):
                    raise ValueError("YOLOE mask observations contain a non-list track")
                new_track = track_remap[(bundle.key, old_track)]
                track_remap_provenance.append(
                    {
                        "bundle_fold_manifest": bundle.key,
                        "source_track_index": old_track,
                        "union_track_index": new_track,
                        "canonical_object_id": track_binding.get(
                            (bundle.key, old_track)
                        ),
                    }
                )
                for raw in track_rows:
                    if not isinstance(raw, Mapping):
                        raise ValueError(
                            "YOLOE mask observations contain a non-object row"
                        )
                    old_image = int(raw.get("image_id", -1))
                    if old_image < 0 or old_image >= len(bundle.mapping_sources):
                        raise ValueError("YOLOE observation has an invalid image_id")
                    source = bundle.mapping_sources[old_image]
                    new_image = heldout_index[source]
                    source_mask = _resolve_mapping_mask(raw, bundle.spec.mask_root)
                    digest = sha256_file(source_mask)
                    mask = _load_mask(
                        source_mask,
                        {
                            "path": str(source_mask),
                            "sha256": digest,
                            "mask_kind": "raw",
                        },
                        field="YOLOE mapping mask",
                    )
                    expected_shape = tuple(
                        int(value) for value in bundle.heldout[source]["depth_size"]
                    )
                    if mask.shape != expected_shape:
                        raise ValueError(
                            "YOLOE mapping mask shape differs from heldout frame"
                        )
                    key = (new_track, new_image)
                    previous = observation_by_key.get(key)
                    if previous is not None:
                        if previous[0] != digest or not filecmp.cmp(
                            previous[1], source_mask, shallow=False
                        ):
                            raise ValueError(
                                "conflicting YOLOE masks for one canonical track/image"
                            )
                        continue
                    relative = Path(
                        f"object_{new_track:06d}/image_{new_image:06d}_{digest[:12]}.npz"
                    )
                    destination = temporary / "mapping" / "masks" / relative
                    _hardlink(source_mask, destination)
                    observation = {
                        "image_id": new_image,
                        "object_idx": new_track,
                        "object_id": new_track,
                        "path": relative.as_posix(),
                    }
                    mapping_observations[new_track].append(observation)
                    observation_by_key[key] = (digest, source_mask)
                    mapping_mask_provenance.append(
                        {
                            "bundle_fold_manifest": bundle.key,
                            "source_image": source,
                            "source": _artifact(source_mask),
                            "linked": _artifact(
                                destination,
                                relative_to=temporary / "mapping" / "masks",
                            ),
                        }
                    )
        for rows in mapping_observations:
            rows.sort(key=lambda row: (int(row["image_id"]), str(row["path"])))

        mapping_payload = {
            "schema": MAPPING_SCHEMA,
            "status": "frozen",
            "images": [
                {
                    "image_id": index,
                    "source_ref": row["source_image"],
                    "storage_path": row["rgb_path"],
                }
                for index, row in enumerate(heldout_rows)
            ],
            "object_id": list(range(next_track)),
            "canonical_object_id_by_track": [
                bound_object_ids[index] if index < len(bound_object_ids) else None
                for index in range(next_track)
            ],
            "object_mask_observations": mapping_observations,
            "contract": {
                "read_only_union": True,
                "heldout_updates_candidate": False,
                "mask_bytes_preserved_by_hardlink": True,
            },
            "provenance": {
                "source_states": [
                    _artifact(bundle.spec.mapping_state) for bundle in loaded
                ],
                "image_id_remap": image_remap_provenance,
                "track_id_remap": track_remap_provenance,
            },
        }
        mapping_path = temporary / "mapping" / "scene_state.json"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(mapping_path, mapping_payload)

        target_rows: dict[tuple[int, str], dict[str, Any]] = {}
        target_source_paths: dict[tuple[int, str], Path] = {}
        target_provenance: list[dict[str, Any]] = []
        timestamp_targets: dict[tuple[int, str], str] = {}
        for bundle in loaded:
            for key, target in sorted(bundle.targets.items()):
                object_id, source = key
                timestamp = _timestamp_key(target["physical_timestamp"])
                timestamp_key = (object_id, timestamp)
                existing_source = timestamp_targets.get(timestamp_key)
                if existing_source is not None and existing_source != source:
                    raise ValueError("duplicate accepted SAM3 object/timestamp target")
                source_mask = Path(target["target_mask"]["path"]).resolve(strict=True)
                digest = sha256_file(source_mask)
                previous = target_rows.get(key)
                if previous is not None:
                    previous_path = target_source_paths[key]
                    if previous["mask_sha256"] != digest or not filecmp.cmp(
                        previous_path, source_mask, shallow=False
                    ):
                        raise ValueError("conflicting accepted SAM3 target masks")
                    continue
                destination_relative = Path(
                    f"masks/object_{object_id:06d}/{heldout_index[source]:06d}_{digest[:12]}.npz"
                )
                destination = temporary / "heldout_reference" / destination_relative
                _hardlink(source_mask, destination)
                old_track = target.get("rescue_track_index")
                new_track = (
                    None
                    if old_track is None
                    else track_remap[(bundle.key, int(old_track))]
                )
                row = {
                    "source_image": source,
                    "physical_timestamp_ns": (
                        int(timestamp) if timestamp.isdigit() else timestamp
                    ),
                    "rescue_image_id": heldout_index[source],
                    "rescue_track_index": new_track,
                    "accepted": True,
                    "mask_relative": destination_relative.as_posix(),
                    "mask_sha256": digest,
                }
                target_rows[key] = row
                target_source_paths[key] = source_mask
                timestamp_targets[timestamp_key] = source
                target_provenance.append(
                    {
                        "bundle_fold_manifest": bundle.key,
                        "object_id": object_id,
                        "source_image": source,
                        "source": _artifact(source_mask),
                        "linked": _artifact(
                            destination,
                            relative_to=temporary / "heldout_reference",
                        ),
                    }
                )

        fit_sha = sha256_file(fit_path)
        bundle_manifest_shas = [
            sha256_file(bundle.spec.fold_manifest) for bundle in loaded
        ]
        final_rgbd_dir = output / "rgbd"
        canonical_rows: list[dict[str, Any]] = []
        canonical_frame_artifacts: list[dict[str, Any]] = []
        for role, rows, metadata in (
            ("train", train_rows, train_meta),
            ("heldout", heldout_rows, heldout_meta),
        ):
            for source_row in rows:
                source = str(source_row["source_image"])
                row = copy.deepcopy(source_row)
                row["full_colmap_split"] = role
                row["rgb_path"] = os.path.relpath(
                    metadata[source]["rgb"], final_rgbd_dir
                ).replace(os.sep, "/")
                row["depth_path"] = os.path.relpath(
                    metadata[source]["depth"], final_rgbd_dir
                ).replace(os.sep, "/")
                canonical_rows.append(row)
                canonical_frame_artifacts.append(
                    {
                        "source_image": source,
                        "physical_timestamp": _timestamp_from_frame(row),
                        "split": role,
                        "rgb": _artifact(
                            metadata[source]["rgb"], relative_to=final_rgbd_dir
                        ),
                        "depth": _artifact(
                            metadata[source]["depth"], relative_to=final_rgbd_dir
                        ),
                    }
                )
        canonical_frames = {
            key: copy.deepcopy(value)
            for key, value in fit_document.items()
            if key not in {"frames", "full_colmap_fold"}
        }
        canonical_frames.update(
            {
                "schema_version": FRAMES_SCHEMA,
                "depth_units": "metres",
                "pose_translation_units": "metres",
                "meters_per_scene_unit": scale,
                "frames": canonical_rows,
                "frozen_evaluation_union": {
                    "schema": UNION_SCHEMA,
                    "role": "canonical_train_heldout_render_union",
                    "read_only_existing_renders": True,
                    "state_fit_splits": ["train"],
                    "heldout_state_fit_authorized": False,
                    "actual_fit_frames_sha256": fit_sha,
                    "source_fold_manifest_sha256": bundle_manifest_shas,
                    "train_frames": len(train_rows),
                    "heldout_frames": len(heldout_rows),
                },
            }
        )
        canonical_frames_path = temporary / "rgbd" / "frames.json"
        canonical_frames_path.parent.mkdir()
        _atomic_json(canonical_frames_path, canonical_frames)
        canonical_frames_sha = sha256_file(canonical_frames_path)

        source_rgbd_runs = [
            {
                "render_manifest": _artifact(bundle.rgbd_render_manifest_path),
                "union_frames": _artifact(bundle.rgbd_union_frames_path),
                "fold_manifest": _artifact(bundle.spec.fold_manifest),
            }
            for bundle in loaded
        ]
        source_names = [str(row["source_image"]) for row in canonical_rows]
        fingerprint_payload = {
            "script": "build_farm_frozen_evaluation_union",
            "version": "1.0",
            "config": canonical_render_config,
            "selected_names_sha256": _sha256_values(source_names),
            "selected_count": len(source_names),
            "inputs": {
                "ply": source_ply_artifact,
                "source_images_sha256": _sha256_values(source_names),
                "actual_fit_frames": _artifact(fit_path),
                "source_rgbd_runs": source_rgbd_runs,
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        canonical_render_manifest = {
            "schema_version": RGBD_UNION_MANIFEST_SCHEMA,
            "status": "frozen",
            "scene_id": str(fit_document["scene_id"]),
            "selected_view_count": len(canonical_rows),
            "frames": "frames.json",
            "frames_artifact": _artifact(
                canonical_frames_path, relative_to=canonical_frames_path.parent
            ),
            "qa_passed": True,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "render_config": canonical_render_config,
            "source_ply": source_ply_artifact,
            "actual_fit_frames": _artifact(fit_path),
            "frame_artifacts": canonical_frame_artifacts,
            "source_rgbd_runs": source_rgbd_runs,
            "contract": {
                "read_only_existing_renders": True,
                "union_did_not_render_or_refit": True,
                "source_runs_qa_passed": True,
                "source_runs_sha256_bound": True,
                "actual_fit_frames_sha256_bound": True,
                "all_frame_assets_sha256_hashed": True,
                "heldout_state_fit_authorized": False,
            },
        }
        canonical_render_manifest_path = temporary / "rgbd" / "run_manifest.json"
        _atomic_json(canonical_render_manifest_path, canonical_render_manifest)

        _write_frames(
            temporary / "train" / "frames.json",
            source_document=fit_document,
            rows=train_rows,
            role="train",
            scale=scale,
            source_fit_sha=fit_sha,
            source_union_frames_sha=canonical_frames_sha,
            source_bundle_shas=bundle_manifest_shas,
        )
        heldout_document = _write_frames(
            temporary / "heldout" / "frames.json",
            source_document=fit_document,
            rows=heldout_rows,
            role="heldout",
            scale=scale,
            source_fit_sha=fit_sha,
            source_union_frames_sha=canonical_frames_sha,
            source_bundle_shas=bundle_manifest_shas,
        )
        train_frames_path = temporary / "train" / "frames.json"
        heldout_frames_path = temporary / "heldout" / "frames.json"

        reference_payload = {
            "schema": REFINEMENT_SCHEMA,
            "status": "PASS",
            "refinement_role": REFINEMENT_ROLE,
            "input_view_contract": {
                "fold_contract": FOLD_SCHEMA,
                "active_split": "heldout",
                "state_fit_authorized": False,
                "plan_view_field": "heldout_views",
                "requested_view_role": "heldout-reference",
            },
            "policy": {
                "view_role": "heldout-reference",
                "state_fit_authorized": False,
                "merge_apply_forbidden": True,
                "heldout_reference_is_evaluation_only": True,
                "reference_masks_are_pseudo_labels_not_ground_truth": True,
            },
            "hashes": {
                "rescue_frames_sha256": sha256_file(heldout_frames_path),
                "rescue_state_sha256": sha256_file(mapping_path),
            },
            "objects": [
                {
                    "object_id": object_id,
                    "accepted": True,
                    "views": [
                        target_rows[key]
                        for key in sorted(target_rows)
                        if key[0] == object_id
                    ],
                }
                for object_id in accepted_ids
            ],
            "summary": {
                "accepted_object_ids": accepted_ids,
                "accepted_objects": len(accepted_ids),
                "accepted_views": len(target_rows),
            },
            "provenance": {
                "source_results": [
                    _artifact(bundle.spec.reference_result) for bundle in loaded
                ],
                "target_masks": target_provenance,
            },
        }
        reference_path = temporary / "heldout_reference" / "result.json"
        _atomic_json(reference_path, reference_payload)

        rendered_artifacts = {
            "train": [
                train_meta[str(row["source_image"])]["provenance"] for row in train_rows
            ],
            "heldout": [
                heldout_meta[str(row["source_image"])]["provenance"]
                for row in heldout_rows
            ],
        }
        bundle_provenance = [
            {
                "fold_manifest": _artifact(bundle.spec.fold_manifest),
                "train_frames": _artifact(bundle.train_frames_path),
                "heldout_frames": _artifact(bundle.heldout_frames_path),
                "mapping_state": _artifact(bundle.spec.mapping_state),
                "reference_result": _artifact(bundle.spec.reference_result),
                "rgbd_render_manifest": _artifact(bundle.rgbd_render_manifest_path),
                "rgbd_union_frames": _artifact(bundle.rgbd_union_frames_path),
                "accepted_object_ids": sorted(bundle.accepted_ids),
            }
            for bundle in loaded
        ]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "PASS",
            "union_contract": {
                "schema": UNION_SCHEMA,
                "actual_fit_frames_are_train": True,
                "heldout_is_deduplicated_union": True,
                "heldout_state_is_read_only": True,
                "accepted_targets_require_source_object_acceptance": True,
                "mask_bytes_preserved_by_hardlink": True,
                "canonical_rgbd_union_materialized": True,
                "canonical_rgbd_union_sha256_bound": True,
            },
            "provenance": {
                "actual_fit_frames": _artifact(fit_path),
                "source_ply": source_ply_artifact,
                "canonical_rgbd_union_frames": _artifact(
                    canonical_frames_path, relative_to=temporary
                ),
                "canonical_rgbd_render_manifest": _artifact(
                    canonical_render_manifest_path, relative_to=temporary
                ),
                "actual_fit_deduplication": [
                    fit_dedup_by_source[source]
                    for source in sorted(fit_dedup_by_source)
                ],
                "source_bundles": bundle_provenance,
                "rendered_artifacts": rendered_artifacts,
                "mapping_masks": mapping_mask_provenance,
                "accepted_target_masks": target_provenance,
            },
            "integrity": {
                "exact_selected_source_image_coverage": True,
                "source_image_sets_disjoint": True,
                "physical_timestamp_sets_disjoint": True,
                "rgb_depth_copied": False,
                "relative_paths_resolve_to_sources": True,
                "all_rendered_artifacts_sha256_hashed": True,
                "canonical_rgbd_union_sha256_bound": True,
                "source_rgbd_runs_sha256_bound": True,
            },
            "fit_policy": {
                "fit_splits": ["train"],
                "train_usage": "mapping_covisibility_sam3_merge_obb_fit",
                "heldout_usage": "frozen_evaluation_only",
                "heldout_consumed": False,
            },
            "counts": {
                "input_actual_fit_rows": len(raw_fit_rows),
                "train_frames": len(train_rows),
                "deduplicated_actual_fit_rows": len(raw_fit_rows) - len(train_rows),
                "heldout_input_rows": sum(len(bundle.heldout) for bundle in loaded),
                "heldout_frames": len(heldout_rows),
                "deduplicated_heldout_rows": sum(
                    len(bundle.heldout) for bundle in loaded
                )
                - len(heldout_rows),
                "train_physical_timestamps": len(train_timestamps),
                "heldout_physical_timestamps": len(heldout_timestamps),
                "accepted_objects": len(accepted_ids),
                "accepted_target_views": len(target_rows),
                "mapping_tracks": next_track,
            },
            "folds": {
                "train": {
                    "frames_json": "train/frames.json",
                    "sha256": sha256_file(train_frames_path),
                    "source_images": [str(row["source_image"]) for row in train_rows],
                },
                "heldout": {
                    "frames_json": "heldout/frames.json",
                    "sha256": sha256_file(heldout_frames_path),
                    "source_images": [str(row["source_image"]) for row in heldout_rows],
                },
            },
            "object_membership": {
                str(object_id): membership[object_id] for object_id in accepted_ids
            },
            "frozen_evaluation_inputs": {
                "mapping_state": _artifact(mapping_path, relative_to=temporary),
                "mapping_mask_root": "mapping/masks",
                "heldout_reference_result": _artifact(
                    reference_path, relative_to=temporary
                ),
                "rgbd_union_frames": _artifact(
                    canonical_frames_path, relative_to=temporary
                ),
                "rgbd_render_manifest": _artifact(
                    canonical_render_manifest_path, relative_to=temporary
                ),
            },
        }
        manifest_path = temporary / "manifest.json"
        _atomic_json(manifest_path, manifest)

        # Validate the just-built fold with the exact production evaluator.
        validation_hashes: dict[Path, str] = {}
        _validate_fold_manifest(
            manifest_path, meters_per_scene_unit=scale, hashes=validation_hashes
        )
        canonical_heldout = {
            str(row["source_image"]): row for row in heldout_document["frames"]
        }
        canonical_membership = {
            int(key): value for key, value in manifest["object_membership"].items()
        }
        canonical_mapping = _load_mapping_state(mapping_path)
        _mapping_image_sources(canonical_mapping, canonical_heldout)
        _validate_reference_targets(
            reference_path,
            heldout_frames_sha=sha256_file(heldout_frames_path),
            mapping_state_sha=sha256_file(mapping_path),
            candidate_ids=set(accepted_ids),
            heldout=canonical_heldout,
            membership=canonical_membership,
            hashes=validation_hashes,
        )

        # Re-hash every source consumed before the atomic publish.
        source_hashes = {
            fit_path: fit_sha,
            source_ply: source_ply_artifact["sha256"],
        }
        for bundle in loaded:
            source_hashes.update(bundle.source_hashes)
        for meta in list(train_meta.values()) + list(heldout_meta.values()):
            source_hashes[meta["rgb"]] = meta["rgb_sha256"]
            source_hashes[meta["depth"]] = meta["depth_sha256"]
        for digest, source_path in observation_by_key.values():
            source_hashes[source_path] = digest
        for key, source_path in target_source_paths.items():
            source_hashes[source_path] = target_rows[key]["mask_sha256"]
        mutated = [
            str(path)
            for path, digest in source_hashes.items()
            if sha256_file(path) != digest
        ]
        if mutated:
            raise RuntimeError(
                f"source artifacts changed during union build: {mutated[:8]}"
            )

        _atomic_json(
            temporary / "_SUCCESS.json",
            {
                "schema": SUCCESS_SCHEMA,
                "status": "success",
                "manifest": "manifest.json",
                "manifest_sha256": sha256_file(manifest_path),
            },
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return _json_object(output / "manifest.json")
