"""Read-only quality control for frozen object-mask projections.

The evaluator deliberately sits *after* all train-only state fitting. It
accepts only immutable, SHA-256-described candidate/evidence manifests and a
full-COLMAP fold manifest. Held-out pixels may affect a release decision, but
can never update masks, OBBs, labels, or the candidate itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from tools.farm_shaper_bridge.common import binary_mask_metrics

FOLD_MANIFEST_SCHEMA = "farm.full-colmap-rgbd-folds.v1"
FRAMES_SCHEMA = "farm_frames_json_v1"
CANDIDATE_SCHEMA = "farm.frozen-heldout-candidate.v1"
EVIDENCE_SCHEMA = "farm.frozen-heldout-evidence.v1"
REPORT_SCHEMA = "farm.frozen-heldout-qc.v1"


@dataclass(frozen=True)
class HeldoutPolicy:
    minimum_independent_timestamps: int = 2
    minimum_good_timestamps: int = 2
    good_timestamp_iou: float = 0.50
    minimum_median_iou: float = 0.60
    minimum_q25_iou: float = 0.45
    minimum_median_precision: float = 0.70
    minimum_worst_precision: float = 0.35
    minimum_median_recall: float = 0.65
    minimum_worst_recall: float = 0.40
    minimum_median_largest_component: float = 0.85
    minimum_worst_largest_component: float = 0.55
    maximum_median_competing_fraction: float = 0.10
    maximum_worst_competing_fraction: float = 0.25
    minimum_depth_pixels_per_timestamp: int = 16
    minimum_depth_timestamps: int = 2
    minimum_median_depth_coverage: float = 0.50
    maximum_median_depth_error_m: float = 0.08
    maximum_median_p90_depth_error_m: float = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def _sha256(value: object, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return digest


def _resolve(path_value: object, *, base: Path, field: str) -> Path:
    value = str(path_value or "").strip()
    if not value:
        raise ValueError(f"{field} has no path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{field} is not a regular file: {path}")
    return path


def _remember_hash(path: Path, cache: dict[Path, str]) -> str:
    digest = cache.get(path)
    if digest is None:
        digest = sha256_file(path)
        cache[path] = digest
    return digest


def _verify_artifact(
    spec: object,
    *,
    base: Path,
    field: str,
    hashes: dict[Path, str],
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{field} must be an artifact object")
    path = _resolve(spec.get("path"), base=base, field=field)
    expected = _sha256(spec.get("sha256"), field=f"{field}.sha256")
    actual = _remember_hash(path, hashes)
    if actual != expected:
        raise ValueError(f"{field} SHA-256 mismatch: {path}")
    if "bytes" in spec:
        size = spec.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{field}.bytes must be a non-negative integer")
        if path.stat().st_size != size:
            raise ValueError(f"{field} byte-size mismatch: {path}")
    return path, spec


def _load_array(path: Path, spec: Mapping[str, Any], *, field: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = str(spec.get("array_key") or "").strip()
            mask_kind = str(spec.get("mask_kind") or "raw").strip()
            if key:
                if key not in archive:
                    raise ValueError(f"{field} lacks NPZ array_key {key!r}")
                return np.asarray(archive[key])
            bits_key = f"{mask_kind}_bits"
            shape_key = f"{mask_kind}_shape"
            bbox_key = f"{mask_kind}_bbox_xyxy"
            if (
                "image_shape" in archive
                and bits_key in archive
                and shape_key in archive
                and bbox_key in archive
            ):
                image_shape = tuple(
                    int(value) for value in archive["image_shape"].tolist()
                )
                crop_shape = tuple(int(value) for value in archive[shape_key].tolist())
                bbox = tuple(int(value) for value in archive[bbox_key].tolist())
                if (
                    len(image_shape) != 2
                    or len(crop_shape) != 2
                    or len(bbox) != 4
                    or any(value <= 0 for value in image_shape + crop_shape)
                ):
                    raise ValueError(f"{field} packed mask metadata is invalid")
                height, width = image_shape
                crop_height, crop_width = crop_shape
                x0, y0, x1, y1 = bbox
                if (
                    x0 < 0
                    or y0 < 0
                    or x1 > width
                    or y1 > height
                    or x1 - x0 != crop_width
                    or y1 - y0 != crop_height
                ):
                    raise ValueError(f"{field} packed mask bbox/shape mismatch")
                count = crop_height * crop_width
                crop = np.unpackbits(
                    np.asarray(archive[bits_key], dtype=np.uint8),
                    bitorder="little",
                    count=count,
                ).reshape(crop_shape)
                result = np.zeros(image_shape, dtype=bool)
                result[y0:y1, x0:x1] = crop
                return result
            keys = list(archive.files)
            if len(keys) != 1:
                raise ValueError(f"{field} NPZ requires array_key")
            return np.asarray(archive[keys[0]])
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"unsupported or unreadable array artifact: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.asarray(image)


def _load_mask(path: Path, spec: Mapping[str, Any], *, field: str) -> np.ndarray:
    array = np.squeeze(_load_array(path, spec, field=field))
    if array.ndim != 2:
        raise ValueError(f"{field} must be a 2-D mask")
    if not np.issubdtype(array.dtype, np.bool_) and not np.all(np.isfinite(array)):
        raise ValueError(f"{field} contains non-finite values")
    return np.asarray(array > 0.5, dtype=bool)


def _load_depth(path: Path, spec: Mapping[str, Any], *, field: str) -> np.ndarray:
    array = np.squeeze(_load_array(path, spec, field=field)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"{field} must be a 2-D depth array")
    return array


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _load_policy(raw: Mapping[str, Any] | None) -> HeldoutPolicy:
    if raw is None:
        return HeldoutPolicy()
    allowed = {item.name for item in fields(HeldoutPolicy)}
    extra = sorted(set(raw).difference(allowed))
    if extra:
        raise ValueError(f"unknown heldout policy keys: {extra}")
    policy = HeldoutPolicy(**dict(raw))
    integer_names = {
        "minimum_independent_timestamps",
        "minimum_good_timestamps",
        "minimum_depth_pixels_per_timestamp",
        "minimum_depth_timestamps",
    }
    bounded_names = {
        item.name
        for item in fields(policy)
        if item.name not in integer_names and not item.name.endswith("_m")
    }
    for item in fields(policy):
        value = getattr(policy, item.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"policy.{item.name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"policy.{item.name} must be finite and non-negative")
        if item.name in integer_names and (not isinstance(value, int) or value < 1):
            raise ValueError(f"policy.{item.name} must be an integer >= 1")
        if item.name in bounded_names and float(value) > 1.0:
            raise ValueError(f"policy.{item.name} must be in [0, 1]")
    if policy.minimum_good_timestamps > policy.minimum_independent_timestamps:
        raise ValueError(
            "minimum_good_timestamps exceeds independent timestamp minimum"
        )
    if policy.minimum_depth_timestamps > policy.minimum_independent_timestamps:
        raise ValueError(
            "minimum_depth_timestamps exceeds independent timestamp minimum"
        )
    return policy


def _exact_strings(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = [str(item or "").strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} contains empty or duplicate values")
    return result


def _validate_frames(
    payload: Mapping[str, Any],
    *,
    role: str,
    meters_per_scene_unit: float,
) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != FRAMES_SCHEMA:
        raise ValueError(f"unsupported {role} frames schema")
    if payload.get("depth_units") != "metres":
        raise ValueError(f"{role} frame depth_units must be metres")
    if "meters_per_scene_unit" not in payload:
        raise ValueError(f"{role} frames are missing meters_per_scene_unit")
    scale = _finite_positive(
        payload["meters_per_scene_unit"],
        field=f"{role}.meters_per_scene_unit",
    )
    if abs(scale - meters_per_scene_unit) > 1.0e-12:
        raise ValueError(f"{role} frames meters_per_scene_unit mismatch")
    fold = payload.get("full_colmap_fold")
    if not isinstance(fold, Mapping) or fold.get("schema") != (
        "farm.full-colmap-rgbd-fold.v1"
    ):
        raise ValueError(f"{role} frames lack full-COLMAP fold contract")
    if fold.get("role") != role:
        raise ValueError(f"{role} frames declare the wrong fold role")
    expected_fit = role == "train"
    if fold.get("state_fit_authorized") is not expected_fit:
        raise ValueError(f"{role} state_fit_authorized contract is invalid")
    expected_usage = (
        "mapping_covisibility_sam3_merge_obb_fit"
        if expected_fit
        else "frozen_evaluation_only"
    )
    if fold.get("usage") != expected_usage:
        raise ValueError(f"{role} fold usage contract is invalid")

    rows = payload.get("frames")
    if not isinstance(rows, list):
        raise ValueError(f"{role} frames must be a list")
    by_source: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{role} frames contain a non-object row")
        source = str(raw.get("source_image") or "").strip()
        timestamp = str(raw.get("physical_timestamp") or "").strip()
        if not source or not timestamp or source in by_source:
            raise ValueError(f"{role} frames contain empty/duplicate source evidence")
        depth_size = raw.get("depth_size")
        if (
            not isinstance(depth_size, list)
            or len(depth_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in depth_size
            )
        ):
            raise ValueError(f"{role} frame {source} has invalid depth_size")
        by_source[source] = dict(raw)
    return by_source


def _validate_fold_manifest(
    manifest_path: Path,
    *,
    meters_per_scene_unit: float,
    hashes: dict[Path, str],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[int, dict[str, list[str]]],
    dict[str, Mapping[str, Any]],
    Path,
    Path,
]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _json_object(manifest_path)
    _remember_hash(manifest_path, hashes)
    if (
        manifest.get("schema") != FOLD_MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
    ):
        raise ValueError("full-COLMAP fold manifest is not a PASS v1 manifest")
    fit = manifest.get("fit_policy")
    if (
        not isinstance(fit, Mapping)
        or fit.get("fit_splits") != ["train"]
        or fit.get("heldout_consumed") is not False
        or fit.get("heldout_usage") != "frozen_evaluation_only"
    ):
        raise ValueError("fold manifest does not reserve heldout for frozen evaluation")
    integrity = manifest.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("source_image_sets_disjoint") is not True
        or integrity.get("physical_timestamp_sets_disjoint") is not True
        or integrity.get("all_rendered_artifacts_sha256_hashed") is not True
    ):
        raise ValueError("fold manifest integrity contract is incomplete")

    folds = manifest.get("folds")
    if not isinstance(folds, Mapping):
        raise ValueError("fold manifest lacks folds")
    loaded: dict[str, tuple[dict[str, Any], Path]] = {}
    for role in ("train", "heldout"):
        descriptor = folds.get(role)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"fold manifest lacks {role} descriptor")
        path = _resolve(
            descriptor.get("frames_json"),
            base=manifest_path.parent,
            field=f"folds.{role}.frames_json",
        )
        expected = _sha256(descriptor.get("sha256"), field=f"folds.{role}.sha256")
        if _remember_hash(path, hashes) != expected:
            raise ValueError(f"{role} frames SHA-256 mismatch")
        payload = _json_object(path)
        loaded[role] = (payload, path)

    train = _validate_frames(
        loaded["train"][0],
        role="train",
        meters_per_scene_unit=meters_per_scene_unit,
    )
    heldout = _validate_frames(
        loaded["heldout"][0],
        role="heldout",
        meters_per_scene_unit=meters_per_scene_unit,
    )
    for role, rows in (("train", train), ("heldout", heldout)):
        declared = _exact_strings(
            folds[role].get("source_images"),
            field=f"folds.{role}.source_images",
        )
        if declared != list(rows):
            raise ValueError(f"{role} source image list differs from frames JSON")
    overlap = sorted(set(train).intersection(heldout))
    if overlap:
        raise ValueError(f"train/heldout source leakage: {overlap[:8]}")
    train_timestamps = {str(row["physical_timestamp"]) for row in train.values()}
    heldout_timestamps = {str(row["physical_timestamp"]) for row in heldout.values()}
    timestamp_overlap = sorted(train_timestamps.intersection(heldout_timestamps))
    if timestamp_overlap:
        raise ValueError(
            f"train/heldout physical timestamp leakage: {timestamp_overlap[:8]}"
        )

    raw_membership = manifest.get("object_membership")
    if not isinstance(raw_membership, Mapping):
        raise ValueError("fold manifest lacks object_membership")
    membership: dict[int, dict[str, list[str]]] = {}
    for raw_id, raw_splits in raw_membership.items():
        object_id = int(raw_id)
        if object_id in membership or not isinstance(raw_splits, Mapping):
            raise ValueError("invalid or duplicate object_membership row")
        train_names = _exact_strings(
            raw_splits.get("train"), field=f"object {object_id} train membership"
        )
        heldout_names = _exact_strings(
            raw_splits.get("heldout"),
            field=f"object {object_id} heldout membership",
        )
        if not set(train_names).issubset(train):
            raise ValueError(f"object {object_id} has non-train fit membership")
        if not set(heldout_names).issubset(heldout):
            raise ValueError(
                f"object {object_id} has non-heldout evaluation membership"
            )
        membership[object_id] = {"train": train_names, "heldout": heldout_names}

    provenance = manifest.get("provenance")
    rendered = (
        provenance.get("rendered_artifacts")
        if isinstance(provenance, Mapping)
        else None
    )
    heldout_artifacts = (
        rendered.get("heldout") if isinstance(rendered, Mapping) else None
    )
    if not isinstance(heldout_artifacts, list):
        raise ValueError("fold manifest lacks heldout artifact provenance")
    depth_specs: dict[str, Mapping[str, Any]] = {}
    for raw in heldout_artifacts:
        if not isinstance(raw, Mapping):
            raise ValueError("heldout artifact provenance contains a non-object row")
        source = str(raw.get("source_image") or "").strip()
        if source not in heldout or source in depth_specs:
            raise ValueError("heldout artifact provenance has unknown/duplicate source")
        if str(raw.get("physical_timestamp") or "") != str(
            heldout[source]["physical_timestamp"]
        ):
            raise ValueError(f"heldout artifact timestamp mismatch for {source}")
        spec = raw.get("depth")
        if not isinstance(spec, Mapping):
            raise ValueError(f"heldout depth provenance missing for {source}")
        path = _resolve(
            spec.get("path"),
            base=loaded["heldout"][1].parent,
            field=f"heldout depth {source}",
        )
        frame_path = _resolve(
            heldout[source].get("depth_path"),
            base=loaded["heldout"][1].parent,
            field=f"heldout frame depth {source}",
        )
        if path != frame_path:
            raise ValueError(f"heldout depth provenance path mismatch for {source}")
        depth_specs[source] = spec
    if set(depth_specs) != set(heldout):
        raise ValueError("heldout depth provenance does not exactly cover frames")
    return (
        manifest,
        train,
        heldout,
        membership,
        depth_specs,
        loaded["train"][1],
        loaded["heldout"][1],
    )


def _validate_candidate(
    candidate_path: Path,
    *,
    fold_sha256: str,
    train_sha256: str,
    scale: float,
    train: Mapping[str, Mapping[str, Any]],
    heldout: Mapping[str, Mapping[str, Any]],
    membership: Mapping[int, Mapping[str, Sequence[str]]],
    hashes: dict[Path, str],
) -> tuple[dict[str, Any], list[int]]:
    candidate_path = candidate_path.expanduser().resolve(strict=True)
    candidate = _json_object(candidate_path)
    _remember_hash(candidate_path, hashes)
    if (
        candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("status") != "frozen"
    ):
        raise ValueError("candidate must be a frozen v1 manifest")
    if (
        _sha256(
            candidate.get("fold_manifest_sha256"),
            field="candidate.fold_manifest_sha256",
        )
        != fold_sha256
    ):
        raise ValueError("candidate references a different fold manifest")
    if (
        _sha256(
            candidate.get("train_frames_sha256"),
            field="candidate.train_frames_sha256",
        )
        != train_sha256
    ):
        raise ValueError("candidate references different train frames")
    candidate_scale = _finite_positive(
        candidate.get("meters_per_scene_unit"),
        field="candidate.meters_per_scene_unit",
    )
    if abs(candidate_scale - scale) > 1.0e-12:
        raise ValueError("candidate meters_per_scene_unit mismatch")

    contract = candidate.get("fit_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("candidate lacks fit_contract")
    if (
        contract.get("fit_splits") != ["train"]
        or contract.get("heldout_consumed") is not False
        or contract.get("heldout_updates_candidate") is not False
        or contract.get("frozen_before_heldout_evaluation") is not True
    ):
        raise ValueError("candidate does not attest frozen train-only fitting")
    fit_sources = _exact_strings(
        contract.get("fit_source_images"), field="candidate fit_source_images"
    )
    fit_timestamps = _exact_strings(
        contract.get("fit_physical_timestamps"),
        field="candidate fit_physical_timestamps",
    )
    train_timestamps = {str(row["physical_timestamp"]) for row in train.values()}
    heldout_timestamps = {str(row["physical_timestamp"]) for row in heldout.values()}
    leaked_sources = sorted(set(fit_sources).intersection(heldout))
    leaked_timestamps = sorted(set(fit_timestamps).intersection(heldout_timestamps))
    if leaked_sources or leaked_timestamps:
        raise ValueError(
            "heldout leakage in candidate fit evidence: "
            f"sources={leaked_sources[:8]}, timestamps={leaked_timestamps[:8]}"
        )
    if not set(fit_sources).issubset(train):
        raise ValueError("candidate fit_source_images are not a subset of train")
    if not set(fit_timestamps).issubset(train_timestamps):
        raise ValueError("candidate fit_physical_timestamps are not a subset of train")

    raw_objects = candidate.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("candidate objects must be a non-empty list")
    object_ids: list[int] = []
    for raw in raw_objects:
        if not isinstance(raw, Mapping):
            raise ValueError("candidate objects contain a non-object row")
        object_id = int(raw.get("object_id"))
        if object_id in object_ids or object_id not in membership:
            raise ValueError(f"invalid/duplicate candidate object_id {object_id}")
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"candidate object {object_id} lacks frozen artifacts")
        for index, spec in enumerate(artifacts):
            _verify_artifact(
                spec,
                base=candidate_path.parent,
                field=f"candidate object {object_id} artifact {index}",
                hashes=hashes,
            )
        object_ids.append(object_id)
    return candidate, sorted(object_ids)


def _require_evidence_contract(evidence: Mapping[str, Any]) -> None:
    if evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("status") != "frozen":
        raise ValueError("evidence must be a frozen v1 manifest")
    if evidence.get("prediction_kind") not in {
        "post_lift_gaussian_projection",
        "projected_object_mask",
    }:
        raise ValueError("unsupported heldout prediction_kind")
    if evidence.get("depth_unit") != "meters":
        raise ValueError("heldout prediction depth must be in meters")
    contract = evidence.get("contract")
    required_true = (
        "view_selection_precommitted",
        "prediction_uses_only_frozen_candidate",
        "targets_evaluation_only",
        "heldout_never_updates_candidate",
        "masks_obb_labels_not_modified",
    )
    if not isinstance(contract, Mapping) or any(
        contract.get(key) is not True for key in required_true
    ):
        raise ValueError("heldout evidence read-only contract is incomplete")


def _evaluate_view(
    observation: Mapping[str, Any],
    *,
    source: str,
    frame: Mapping[str, Any],
    evidence_base: Path,
    heldout_frames_base: Path,
    scene_depth_spec: Mapping[str, Any],
    scene_depth_cache: dict[str, np.ndarray],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    object_id = int(observation["object_id"])
    loaded_masks: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for name in ("prediction_mask", "target_mask", "competing_mask"):
        path, spec = _verify_artifact(
            observation.get(name),
            base=evidence_base,
            field=f"object {object_id} {source} {name}",
            hashes=hashes,
        )
        loaded_masks[name] = _load_mask(path, spec, field=name)
        artifacts[name] = {"path": str(path), "sha256": hashes[path]}

    predicted_depth_path, predicted_depth_spec = _verify_artifact(
        observation.get("prediction_depth"),
        base=evidence_base,
        field=f"object {object_id} {source} prediction_depth",
        hashes=hashes,
    )
    predicted_depth = _load_depth(
        predicted_depth_path,
        predicted_depth_spec,
        field="prediction_depth",
    )
    scene_depth_path, verified_scene_spec = _verify_artifact(
        scene_depth_spec,
        base=heldout_frames_base,
        field=f"heldout scene depth {source}",
        hashes=hashes,
    )
    scene_depth = scene_depth_cache.get(source)
    if scene_depth is None:
        scene_depth = _load_depth(
            scene_depth_path, verified_scene_spec, field="scene_depth"
        )
        scene_depth_cache[source] = scene_depth
    artifacts["prediction_depth"] = {
        "path": str(predicted_depth_path),
        "sha256": hashes[predicted_depth_path],
    }
    artifacts["scene_depth"] = {
        "path": str(scene_depth_path),
        "sha256": hashes[scene_depth_path],
    }

    expected_shape = tuple(int(value) for value in frame["depth_size"])
    arrays = {
        **loaded_masks,
        "prediction_depth": predicted_depth,
        "scene_depth": scene_depth,
    }
    wrong = {
        name: list(array.shape)
        for name, array in arrays.items()
        if array.shape != expected_shape
    }
    if wrong:
        raise ValueError(
            f"object {object_id} {source} artifact shapes differ from "
            f"depth_size {expected_shape}: {wrong}"
        )

    predicted = loaded_masks["prediction_mask"]
    target = loaded_masks["target_mask"]
    competing = loaded_masks["competing_mask"]
    metrics = binary_mask_metrics(predicted, target)
    prediction_pixels = int(metrics["prediction_pixels"])
    target_pixels = int(metrics["target_pixels"])
    competing_overlap = int(np.count_nonzero(predicted & competing))
    target_competing_overlap = int(np.count_nonzero(target & competing))
    intersection = predicted & target
    valid_depth = (
        intersection
        & np.isfinite(predicted_depth)
        & (predicted_depth > 0.0)
        & np.isfinite(scene_depth)
        & (scene_depth > 0.0)
    )
    residual = np.abs(predicted_depth[valid_depth] - scene_depth[valid_depth])
    intersection_pixels = int(np.count_nonzero(intersection))
    metrics.update(
        {
            "object_id": object_id,
            "source_image": source,
            "physical_timestamp": str(frame["physical_timestamp"]),
            "competing_overlap_pixels": competing_overlap,
            "competing_prediction_fraction": competing_overlap
            / max(prediction_pixels, 1),
            "competing_target_fraction": target_competing_overlap
            / max(target_pixels, 1),
            "depth_pixels": int(residual.size),
            "depth_coverage": int(residual.size) / max(intersection_pixels, 1),
            "depth_error_median": (
                float(np.median(residual)) if residual.size else None
            ),
            "depth_error_p90": (
                float(np.quantile(residual, 0.90)) if residual.size else None
            ),
            "artifacts": artifacts,
        }
    )
    return metrics


def _timestamp_summary(
    views: Sequence[Mapping[str, Any]],
    policy: HeldoutPolicy,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in views:
        grouped[str(row["physical_timestamp"])].append(row)
    names = (
        "iou",
        "precision",
        "recall",
        "largest_component_fraction",
        "area_ratio",
        "competing_prediction_fraction",
        "competing_target_fraction",
        "depth_coverage",
    )
    timestamp_rows: list[dict[str, Any]] = []
    good = 0
    for timestamp in sorted(grouped):
        rows = grouped[timestamp]
        item: dict[str, Any] = {
            "physical_timestamp": timestamp,
            "view_count": len(rows),
            "source_images": sorted(str(row["source_image"]) for row in rows),
        }
        for name in names:
            item[name] = float(np.median([float(row[name]) for row in rows]))
        item["depth_pixels"] = int(
            np.median([int(row["depth_pixels"]) for row in rows])
        )
        for name in ("depth_error_median", "depth_error_p90"):
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            item[name] = float(np.median(values)) if values else None
        item["good"] = bool(
            item["iou"] >= policy.good_timestamp_iou
            and item["precision"] >= policy.minimum_worst_precision
            and item["recall"] >= policy.minimum_worst_recall
            and item["largest_component_fraction"]
            >= policy.minimum_worst_largest_component
            and item["competing_prediction_fraction"]
            <= policy.maximum_worst_competing_fraction
            and item["depth_pixels"] >= policy.minimum_depth_pixels_per_timestamp
            and item["depth_error_median"] is not None
            and item["depth_error_median"] <= policy.maximum_median_depth_error_m
            and item["depth_error_p90"] is not None
            and item["depth_error_p90"] <= policy.maximum_median_p90_depth_error_m
        )
        good += int(item["good"])
        timestamp_rows.append(item)
    summaries = {
        name: _quantiles([row[name] for row in timestamp_rows]) for name in names
    }
    for name in ("depth_error_median", "depth_error_p90"):
        summaries[name] = _quantiles(
            [row[name] for row in timestamp_rows if row[name] is not None]
        )
    return timestamp_rows, good, summaries


def _gate_object(
    object_id: int,
    views: Sequence[Mapping[str, Any]],
    policy: HeldoutPolicy,
) -> dict[str, Any]:
    timestamps, good_count, summaries = _timestamp_summary(views, policy)
    reasons: list[str] = []
    count = len(timestamps)
    if count < policy.minimum_independent_timestamps:
        reasons.append("insufficient_independent_heldout_timestamps")
        route = "collect_independent_heldout_evidence"
        status = "insufficient_evidence"
    else:

        def value(name: str, statistic: str, fallback: float) -> float:
            summary = summaries[name]
            return fallback if summary is None else float(summary[statistic])

        if good_count < policy.minimum_good_timestamps:
            reasons.append("insufficient_good_timestamps")
        if value("iou", "median", -math.inf) < policy.minimum_median_iou:
            reasons.append("median_iou")
        if value("iou", "q25", -math.inf) < policy.minimum_q25_iou:
            reasons.append("q25_iou")
        if value("precision", "median", -math.inf) < policy.minimum_median_precision:
            reasons.append("median_precision")
        if value("precision", "minimum", -math.inf) < policy.minimum_worst_precision:
            reasons.append("worst_precision")
        if value("recall", "median", -math.inf) < policy.minimum_median_recall:
            reasons.append("median_recall")
        if value("recall", "minimum", -math.inf) < policy.minimum_worst_recall:
            reasons.append("worst_recall")
        if (
            value("largest_component_fraction", "median", -math.inf)
            < policy.minimum_median_largest_component
        ):
            reasons.append("fragmented_projection")
        if (
            value("largest_component_fraction", "minimum", -math.inf)
            < policy.minimum_worst_largest_component
        ):
            reasons.append("worst_fragmented_projection")
        if (
            value("competing_prediction_fraction", "median", math.inf)
            > policy.maximum_median_competing_fraction
        ):
            reasons.append("competing_instance_contamination")
        if (
            value("competing_prediction_fraction", "maximum", math.inf)
            > policy.maximum_worst_competing_fraction
        ):
            reasons.append("worst_competing_instance_contamination")

        depth_rows = [
            row
            for row in timestamps
            if row["depth_pixels"] >= policy.minimum_depth_pixels_per_timestamp
            and row["depth_error_median"] is not None
            and row["depth_error_p90"] is not None
        ]
        if len(depth_rows) < policy.minimum_depth_timestamps:
            reasons.append("insufficient_independent_depth_timestamps")
        if (
            value("depth_coverage", "median", -math.inf)
            < policy.minimum_median_depth_coverage
        ):
            reasons.append("depth_coverage")
        if (
            value("depth_error_median", "median", math.inf)
            > policy.maximum_median_depth_error_m
        ):
            reasons.append("median_depth_error")
        if (
            value("depth_error_p90", "median", math.inf)
            > policy.maximum_median_p90_depth_error_m
        ):
            reasons.append("p90_depth_error")
        route = (
            "heldout_verified"
            if not reasons
            else "reject_candidate_train_only_refinement"
        )
        status = "verified" if not reasons else "rejected"
    return {
        "object_id": object_id,
        "status": status,
        "route": route,
        "reasons": reasons,
        "view_count": len(views),
        "independent_physical_timestamps": count,
        "good_timestamps": good_count,
        "summaries": summaries,
        "timestamps": timestamps,
        "views": list(views),
    }


def evaluate_frozen_heldout_qc(
    fold_manifest_path: Path,
    candidate_manifest_path: Path,
    evidence_manifest_path: Path,
    *,
    meters_per_scene_unit: float,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate immutable projections without mutating upstream state."""

    scale = _finite_positive(meters_per_scene_unit, field="meters_per_scene_unit")
    thresholds = _load_policy(policy)
    hashes: dict[Path, str] = {}
    fold_manifest_path = fold_manifest_path.expanduser().resolve(strict=True)
    candidate_manifest_path = candidate_manifest_path.expanduser().resolve(strict=True)
    evidence_manifest_path = evidence_manifest_path.expanduser().resolve(strict=True)
    (
        fold,
        train,
        heldout,
        membership,
        depth_specs,
        train_frames_path,
        heldout_frames_path,
    ) = _validate_fold_manifest(
        fold_manifest_path,
        meters_per_scene_unit=scale,
        hashes=hashes,
    )
    fold_sha = hashes[fold_manifest_path]
    train_sha = hashes[train_frames_path]
    heldout_sha = hashes[heldout_frames_path]
    candidate, object_ids = _validate_candidate(
        candidate_manifest_path,
        fold_sha256=fold_sha,
        train_sha256=train_sha,
        scale=scale,
        train=train,
        heldout=heldout,
        membership=membership,
        hashes=hashes,
    )
    candidate_sha = hashes[candidate_manifest_path]

    evidence = _json_object(evidence_manifest_path)
    _remember_hash(evidence_manifest_path, hashes)
    _require_evidence_contract(evidence)
    links = (
        ("fold_manifest_sha256", fold_sha),
        ("candidate_manifest_sha256", candidate_sha),
        ("heldout_frames_sha256", heldout_sha),
    )
    for name, expected in links:
        if _sha256(evidence.get(name), field=f"evidence.{name}") != expected:
            raise ValueError(f"evidence {name} points to a different frozen input")
    evidence_scale = _finite_positive(
        evidence.get("meters_per_scene_unit"),
        field="evidence.meters_per_scene_unit",
    )
    if abs(evidence_scale - scale) > 1.0e-12:
        raise ValueError("evidence meters_per_scene_unit mismatch")

    raw_observations = evidence.get("observations")
    if not isinstance(raw_observations, list):
        raise ValueError("evidence observations must be a list")
    by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    scene_depth_cache: dict[str, np.ndarray] = {}
    seen: set[tuple[int, str]] = set()
    for index, raw in enumerate(raw_observations):
        if not isinstance(raw, Mapping):
            raise ValueError("evidence contains a non-object observation")
        object_id = int(raw.get("object_id"))
        source = str(raw.get("source_image") or "").strip()
        timestamp = str(raw.get("physical_timestamp") or "").strip()
        if object_id not in object_ids:
            raise ValueError(f"evidence references non-candidate object {object_id}")
        if source in train:
            raise ValueError(f"heldout evidence uses train source image {source}")
        if source not in heldout:
            raise ValueError(f"heldout evidence uses unknown source image {source}")
        expected_timestamp = str(heldout[source]["physical_timestamp"])
        if timestamp != expected_timestamp:
            raise ValueError(
                f"heldout evidence timestamp mismatch for source image {source}"
            )
        if source not in membership[object_id]["heldout"]:
            raise ValueError(
                f"source image {source} was not precommitted for object {object_id}"
            )
        key = (object_id, source)
        if key in seen:
            raise ValueError(
                f"duplicate heldout observation for object {object_id}, {source}"
            )
        seen.add(key)
        view = _evaluate_view(
            raw,
            source=source,
            frame=heldout[source],
            evidence_base=evidence_manifest_path.parent,
            heldout_frames_base=heldout_frames_path.parent,
            scene_depth_spec=depth_specs[source],
            scene_depth_cache=scene_depth_cache,
            hashes=hashes,
        )
        view["observation_index"] = index
        by_object[object_id].append(view)

    objects = [
        _gate_object(object_id, by_object.get(object_id, []), thresholds)
        for object_id in object_ids
    ]
    routes = {
        name: [row["object_id"] for row in objects if row["route"] == name]
        for name in (
            "heldout_verified",
            "collect_independent_heldout_evidence",
            "reject_candidate_train_only_refinement",
        )
    }

    mutated = [
        str(path) for path, digest in hashes.items() if sha256_file(path) != digest
    ]
    if mutated:
        raise RuntimeError(f"frozen input changed during evaluation: {mutated[:8]}")
    return {
        "schema": REPORT_SCHEMA,
        "status": (
            "PASS" if len(routes["heldout_verified"]) == len(object_ids) else "BLOCKED"
        ),
        "contract": {
            "read_only": True,
            "heldout_used_for_state_fit": False,
            "heldout_updates_masks_obb_labels": False,
            "acceptance_aggregated_by_independent_physical_timestamp": True,
            "duplicate_views_at_one_timestamp_cannot_inflate_independence": True,
            "all_consumed_inputs_sha256_verified_and_rechecked": True,
        },
        "meters_per_scene_unit": scale,
        "policy": {
            item.name: getattr(thresholds, item.name) for item in fields(thresholds)
        },
        "counts": {
            "candidate_objects": len(object_ids),
            "observations": len(raw_observations),
            "unique_consumed_files": len(hashes),
            "verified": len(routes["heldout_verified"]),
            "collect_more_evidence": len(
                routes["collect_independent_heldout_evidence"]
            ),
            "rejected": len(routes["reject_candidate_train_only_refinement"]),
        },
        "routes": routes,
        "objects": objects,
        "provenance": {
            "fold_manifest": {
                "path": str(fold_manifest_path),
                "sha256": fold_sha,
                "schema": fold.get("schema"),
            },
            "candidate_manifest": {
                "path": str(candidate_manifest_path),
                "sha256": candidate_sha,
                "schema": candidate.get("schema"),
            },
            "evidence_manifest": {
                "path": str(evidence_manifest_path),
                "sha256": hashes[evidence_manifest_path],
                "schema": evidence.get("schema"),
            },
            "train_frames": {
                "path": str(train_frames_path),
                "sha256": train_sha,
            },
            "heldout_frames": {
                "path": str(heldout_frames_path),
                "sha256": heldout_sha,
            },
            "consumed_files": [
                {"path": str(path), "sha256": digest}
                for path, digest in sorted(
                    hashes.items(), key=lambda item: str(item[0])
                )
            ],
        },
    }
