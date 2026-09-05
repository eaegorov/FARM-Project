"""Fail-closed materialization of frozen post-lift heldout evidence.

This module never renders, trains, merges, or updates scene state. It only
validates immutable inputs, assembles SAM3 pseudo-targets and YOLOE competitor
unions, and links projections produced by a separately attested read-only
renderer. Without such projections it returns a BLOCKED preflight instead of
inventing a surrogate prediction.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from farm_runtime.frozen_heldout_qc import (
    CANDIDATE_SCHEMA,
    EVIDENCE_SCHEMA,
    _load_depth,
    _load_mask,
    _remember_hash,
    _validate_candidate,
    _validate_fold_manifest,
    _verify_artifact,
    sha256_file,
)

FIT_CONTRACT_SCHEMA = "farm.full-colmap-train-fit.v1"
PROJECTION_SCHEMA = "farm.frozen-heldout-gaussian-projections.v1"
BUILD_SCHEMA = "farm.frozen-heldout-evidence-build.v1"
REFINEMENT_SCHEMA = "farm.full-colmap-mask-refinement.v1"
REFINEMENT_ROLE = "heldout_reference_only"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _positive_scale(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("meters_per_scene_unit must be finite and positive")
    return scale


def _timestamp_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("physical timestamp is empty")
    return str(int(text)) if text.isdigit() else text


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count: int | None = None
        for _ in range(10000):
            line = stream.readline()
            if not line:
                break
            if len(line) > 65536:
                raise ValueError(f"unreasonably long PLY header line: {path}")
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                if vertex_count is None or vertex_count < 1:
                    raise ValueError(f"PLY has no positive vertex count: {path}")
                return vertex_count
    raise ValueError(f"unterminated PLY header: {path}")


def _labels(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(f"label NPZ must contain exactly one array: {path}")
            value = np.asarray(archive[archive.files[0]])
    else:
        raise ValueError(f"labels must be .npy or single-array .npz: {path}")
    labels = np.asarray(value).squeeze()
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be a one-dimensional integer array: {path}")
    return labels.astype(np.int64, copy=False)


def _verify_gaussian_shapes(
    source_ply: Path,
    source_labels: Path,
    candidate_ply: Path,
    candidate_labels: Path,
) -> tuple[np.ndarray, np.ndarray]:
    source = _labels(source_labels)
    candidate = _labels(candidate_labels)
    source_count = _ply_vertex_count(source_ply)
    candidate_count = _ply_vertex_count(candidate_ply)
    if source.size != source_count:
        raise ValueError("source labels do not align with source PLY vertices")
    if candidate.size != candidate_count:
        raise ValueError("candidate labels do not align with candidate PLY vertices")
    if source_count != candidate_count:
        raise ValueError(
            "source/candidate Gaussian counts differ; order is not aligned"
        )
    return source, candidate


def _iter_artifact_specs(value: object) -> Sequence[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            if "path" in item and "sha256" in item:
                found.append(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return found


def _require_upstream_artifact(
    upstream: Mapping[str, Any],
    upstream_path: Path,
    expected_path: Path,
    hashes: dict[Path, str],
    *,
    field: str,
) -> None:
    expected_path = expected_path.resolve(strict=True)
    for index, spec in enumerate(_iter_artifact_specs(upstream.get("artifacts"))):
        try:
            path, _ = _verify_artifact(
                spec,
                base=upstream_path.parent,
                field=f"upstream artifact {index}",
                hashes=hashes,
            )
        except (OSError, ValueError):
            continue
        if path == expected_path:
            return
    raise ValueError(f"upstream lift result does not SHA-declare {field}")


def materialize_frozen_candidate(
    fold_manifest_path: Path,
    upstream_lift_result_path: Path,
    source_ply_path: Path,
    source_labels_path: Path,
    candidate_ply_path: Path,
    candidate_labels_path: Path,
    *,
    meters_per_scene_unit: float,
    object_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the evaluator candidate manifest from an attested train-only lift."""

    scale = _positive_scale(meters_per_scene_unit)
    paths = [
        Path(value).expanduser().resolve(strict=True)
        for value in (
            fold_manifest_path,
            upstream_lift_result_path,
            source_ply_path,
            source_labels_path,
            candidate_ply_path,
            candidate_labels_path,
        )
    ]
    (
        fold_path,
        upstream_path,
        source_ply,
        source_labels,
        candidate_ply,
        candidate_labels,
    ) = paths
    hashes: dict[Path, str] = {}
    (
        _fold,
        train,
        heldout,
        membership,
        _depth_specs,
        train_frames_path,
        _heldout_frames_path,
    ) = _validate_fold_manifest(fold_path, meters_per_scene_unit=scale, hashes=hashes)
    upstream = _json_object(upstream_path)
    _remember_hash(upstream_path, hashes)
    if (
        upstream.get("schema_version") != "farm.gaussian-lift.result.v1"
        or upstream.get("status") != "PASS"
    ):
        raise ValueError("upstream Gaussian lift must be a PASS v1 result")
    contracts = upstream.get("contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError("upstream Gaussian lift lacks contracts")
    if (
        contracts.get("source_ply_mutated") is not False
        or contracts.get("dense_arrays_in_source_ply_order") is not True
        or contracts.get("verified_only_canonical_labels") is not True
    ):
        raise ValueError(
            "upstream Gaussian lift does not preserve frozen Gaussian order"
        )
    source_sha = _remember_hash(source_ply, hashes)
    declared_source_sha = str(
        ((upstream.get("inputs") or {}).get("source_ply_sha256") or "")
    ).lower()
    if declared_source_sha != source_sha:
        raise ValueError("upstream source PLY SHA-256 mismatch")
    for path, name in (
        (source_labels, "source labels"),
        (candidate_labels, "candidate labels"),
    ):
        _require_upstream_artifact(upstream, upstream_path, path, hashes, field=name)
    if candidate_ply != source_ply:
        _require_upstream_artifact(
            upstream, upstream_path, candidate_ply, hashes, field="candidate PLY"
        )
    source_array, candidate_array = _verify_gaussian_shapes(
        source_ply, source_labels, candidate_ply, candidate_labels
    )

    fit = upstream.get("frozen_full_colmap_fit_contract")
    if not isinstance(fit, Mapping) or fit.get("schema") != FIT_CONTRACT_SCHEMA:
        raise ValueError(
            "upstream lift lacks frozen_full_colmap_fit_contract "
            f"({FIT_CONTRACT_SCHEMA})"
        )
    if str(fit.get("fold_manifest_sha256") or "").lower() != hashes[fold_path]:
        raise ValueError("upstream lift references a different fold manifest")
    if str(fit.get("train_frames_sha256") or "").lower() != hashes[train_frames_path]:
        raise ValueError("upstream lift references different train frames")
    required_fit = {
        "fit_splits": ["train"],
        "heldout_consumed": False,
        "heldout_updates_candidate": False,
        "frozen_before_heldout_evaluation": True,
    }
    if any(fit.get(key) != value for key, value in required_fit.items()):
        raise ValueError("upstream lift is not an immutable train-only candidate")
    fit_sources = [str(value) for value in fit.get("fit_source_images") or []]
    fit_timestamps = [str(value) for value in fit.get("fit_physical_timestamps") or []]
    if not fit_sources or len(fit_sources) != len(set(fit_sources)):
        raise ValueError("fit_source_images must be non-empty and unique")
    if not fit_timestamps or len(fit_timestamps) != len(set(fit_timestamps)):
        raise ValueError("fit_physical_timestamps must be non-empty and unique")
    train_timestamps = {str(row["physical_timestamp"]) for row in train.values()}
    heldout_timestamps = {str(row["physical_timestamp"]) for row in heldout.values()}
    if not set(fit_sources).issubset(train) or set(fit_sources).intersection(heldout):
        raise ValueError("upstream fit source leakage or non-train source")
    if not set(fit_timestamps).issubset(train_timestamps) or set(
        fit_timestamps
    ).intersection(heldout_timestamps):
        raise ValueError("upstream fit timestamp leakage or non-train timestamp")

    available_ids = sorted(
        int(value)
        for value in np.unique(candidate_array)
        if int(value) >= 0 and int(value) in membership
    )
    selected = (
        available_ids if object_ids is None else sorted(set(map(int, object_ids)))
    )
    if not selected:
        raise ValueError("candidate labels contain no requested heldout objects")
    missing = sorted(set(selected).difference(available_ids))
    if missing:
        raise ValueError(
            f"requested object IDs absent from candidate labels: {missing}"
        )

    gaussian_artifacts = {
        "source_ply": _artifact(source_ply),
        "source_labels": _artifact(source_labels),
        "candidate_ply": _artifact(candidate_ply),
        "candidate_labels": _artifact(candidate_labels),
    }
    candidate_artifacts = [
        gaussian_artifacts["candidate_ply"],
        gaussian_artifacts["candidate_labels"],
    ]
    result = {
        "schema": CANDIDATE_SCHEMA,
        "status": "frozen",
        "meters_per_scene_unit": scale,
        "fold_manifest_sha256": hashes[fold_path],
        "train_frames_sha256": hashes[train_frames_path],
        "fit_contract": {
            key: fit[key]
            for key in (
                "fit_splits",
                "heldout_consumed",
                "heldout_updates_candidate",
                "frozen_before_heldout_evaluation",
                "fit_source_images",
                "fit_physical_timestamps",
            )
        },
        "gaussian_contract": {
            "source_candidate_gaussian_order_aligned": True,
            "source_gaussian_count": int(source_array.size),
            "candidate_gaussian_count": int(candidate_array.size),
            "candidate_frozen": True,
            "labels_frozen": True,
            "heldout_used_for_fit": False,
        },
        "gaussian_artifacts": gaussian_artifacts,
        "upstream_lift_result": _artifact(upstream_path),
        "provenance": {
            "consumed_input_hashes": {
                str(path): digest
                for path, digest in sorted(
                    hashes.items(), key=lambda item: str(item[0])
                )
            }
        },
        "objects": [
            {"object_id": object_id, "artifacts": candidate_artifacts}
            for object_id in selected
        ],
    }
    mutated = [
        str(path) for path, digest in hashes.items() if sha256_file(path) != digest
    ]
    if mutated:
        raise RuntimeError(
            f"frozen input changed during candidate materialization: {mutated[:8]}"
        )
    return result


def _resolve_exact_gaussian_artifacts(
    candidate: Mapping[str, Any],
    candidate_path: Path,
    explicit: Mapping[str, Path],
    hashes: dict[Path, str],
) -> None:
    contract = candidate.get("gaussian_contract")
    required = {
        "source_candidate_gaussian_order_aligned": True,
        "candidate_frozen": True,
        "labels_frozen": True,
        "heldout_used_for_fit": False,
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != expected for key, expected in required.items()
    ):
        raise ValueError("candidate Gaussian contract is incomplete")
    raw = candidate.get("gaussian_artifacts")
    if not isinstance(raw, Mapping):
        raise ValueError("candidate lacks gaussian_artifacts")
    verified: dict[str, Path] = {}
    for name, expected_path in explicit.items():
        path, _ = _verify_artifact(
            raw.get(name),
            base=candidate_path.parent,
            field=f"candidate.gaussian_artifacts.{name}",
            hashes=hashes,
        )
        if path != expected_path:
            raise ValueError(f"explicit {name} differs from candidate manifest")
        verified[name] = path
    source, candidate_labels = _verify_gaussian_shapes(
        verified["source_ply"],
        verified["source_labels"],
        verified["candidate_ply"],
        verified["candidate_labels"],
    )
    if int(contract.get("source_gaussian_count", -1)) != int(source.size):
        raise ValueError("candidate source_gaussian_count mismatch")
    if int(contract.get("candidate_gaussian_count", -1)) != int(candidate_labels.size):
        raise ValueError("candidate candidate_gaussian_count mismatch")
    _verify_artifact(
        candidate.get("upstream_lift_result"),
        base=candidate_path.parent,
        field="candidate.upstream_lift_result",
        hashes=hashes,
    )
    required_hashes = {
        hashes[verified["candidate_ply"]],
        hashes[verified["candidate_labels"]],
    }
    for row in candidate.get("objects") or []:
        declared_hashes = {
            str(spec.get("sha256") or "").lower()
            for spec in row.get("artifacts") or []
            if isinstance(spec, Mapping)
        }
        if not required_hashes.issubset(declared_hashes):
            raise ValueError(
                "every candidate object must bind candidate PLY and labels"
            )


def _record_value(record: object, key: str, default: object = None) -> object:
    return (
        record.get(key, default)
        if isinstance(record, Mapping)
        else getattr(record, key, default)
    )


def _load_mapping_state(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload: object = _json_object(path)
    else:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production image has torch
            raise RuntimeError(
                "PyTorch is required to read a .pt YOLOE mapping state"
            ) from exc
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = (
        payload.get("state")
        if isinstance(payload, Mapping) and "state" in payload
        else payload
    )
    if not isinstance(state, dict):
        raise ValueError("unsupported heldout YOLOE mapping state")
    return state


def _mapping_image_sources(
    state: Mapping[str, Any], heldout: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], dict[str, int]]:
    stems: dict[str, str] = {}
    for source, frame in heldout.items():
        rgb_path = str(frame.get("rgb_path") or "").strip()
        for stem in {Path(rgb_path).stem, Path(source).stem}:
            if stem and stem in stems and stems[stem] != source:
                raise ValueError(f"ambiguous heldout RGB stem: {stem}")
            if stem:
                stems[stem] = source
    sources: list[str] = []
    by_source: dict[str, int] = {}
    images = state.get("images")
    if not isinstance(images, (list, tuple)):
        raise ValueError("heldout YOLOE mapping has no image inventory")
    for image_id, record in enumerate(images):
        reference = str(
            _record_value(record, "source_ref", "")
            or _record_value(record, "storage_path", "")
        )
        source = stems.get(Path(reference).stem)
        if source is None or source in by_source:
            raise ValueError(
                "YOLOE mapping contains unknown or duplicate heldout image"
            )
        sources.append(source)
        by_source[source] = image_id
    if set(sources) != set(heldout):
        raise ValueError("YOLOE mapping does not exactly cover the frozen heldout fold")
    return sources, by_source


def _resolve_mapping_mask(observation: Mapping[str, Any], root: Path) -> Path:
    raw = str(observation.get("path") or observation.get("mask_path") or "").strip()
    if not raw:
        raise ValueError("YOLOE mask observation has no path")
    source = Path(raw)
    tail: Path | None = None
    for index, part in enumerate(source.parts):
        if part.startswith("object_"):
            tail = Path(*source.parts[index:])
            break
    if tail is None:
        raise ValueError("YOLOE observation path has no object directory")
    resolved_root = root.resolve(strict=True)
    result = (resolved_root / tail).resolve(strict=True)
    if not result.is_relative_to(resolved_root) or result.suffix.lower() != ".npz":
        raise ValueError("YOLOE observation escapes mask root or is not NPZ")
    return result


def _validate_reference_targets(
    result_path: Path,
    *,
    heldout_frames_sha: str,
    mapping_state_sha: str,
    candidate_ids: set[int],
    heldout: Mapping[str, Mapping[str, Any]],
    membership: Mapping[int, Mapping[str, Sequence[str]]],
    hashes: dict[Path, str],
) -> dict[tuple[int, str], dict[str, Any]]:
    payload = _json_object(result_path)
    _remember_hash(result_path, hashes)
    if (
        payload.get("schema") != REFINEMENT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("refinement_role") != REFINEMENT_ROLE
    ):
        raise ValueError("SAM3 reference must be a PASS heldout-reference result")
    view_contract = payload.get("input_view_contract")
    policy = payload.get("policy")
    if (
        not isinstance(view_contract, Mapping)
        or view_contract.get("fold_contract") != "farm.full-colmap-rgbd-fold.v1"
        or view_contract.get("active_split") != "heldout"
        or view_contract.get("state_fit_authorized") is not False
        or view_contract.get("plan_view_field") != "heldout_views"
        or view_contract.get("requested_view_role") != "heldout-reference"
        or not isinstance(policy, Mapping)
        or policy.get("view_role") != "heldout-reference"
        or policy.get("state_fit_authorized") is not False
        or policy.get("merge_apply_forbidden") is not True
        or policy.get("heldout_reference_is_evaluation_only") is not True
        or policy.get("reference_masks_are_pseudo_labels_not_ground_truth") is not True
    ):
        raise ValueError("SAM3 heldout-reference policy is incomplete")
    declared = payload.get("hashes")
    if not isinstance(declared, Mapping):
        raise ValueError("SAM3 heldout-reference result lacks hashes")
    if str(declared.get("rescue_frames_sha256") or "").lower() != heldout_frames_sha:
        raise ValueError("SAM3 reference uses a different heldout frames JSON")
    if str(declared.get("rescue_state_sha256") or "").lower() != mapping_state_sha:
        raise ValueError("SAM3 reference uses a different YOLOE mapping state")
    targets: dict[tuple[int, str], dict[str, Any]] = {}
    seen_timestamps: set[tuple[int, str]] = set()
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise ValueError("SAM3 heldout-reference result has no object rows")
    for object_row in rows:
        if not isinstance(object_row, Mapping):
            raise ValueError("SAM3 result contains a non-object row")
        object_id = int(object_row.get("object_id"))
        views = object_row.get("views")
        if not isinstance(views, list):
            raise ValueError(f"SAM3 object {object_id} has no view rows")
        for view in views:
            if not isinstance(view, Mapping) or view.get("accepted") is not True:
                continue
            if object_id not in candidate_ids:
                continue
            source = str(view.get("source_image") or "").strip()
            if source not in heldout or source not in membership[object_id]["heldout"]:
                raise ValueError("SAM3 target leaks or was not precommitted for object")
            timestamp = str(heldout[source]["physical_timestamp"])
            if _timestamp_key(view.get("physical_timestamp_ns")) != _timestamp_key(
                timestamp
            ):
                raise ValueError(f"SAM3 physical timestamp mismatch for {source}")
            key = (object_id, source)
            timestamp_key = (object_id, _timestamp_key(timestamp))
            if key in targets or timestamp_key in seen_timestamps:
                raise ValueError("duplicate SAM3 target source or physical timestamp")
            relative = Path(str(view.get("mask_relative") or ""))
            if relative.is_absolute() or not relative.parts:
                raise ValueError("SAM3 target mask must be result-relative")
            mask_path = (result_path.parent / relative).resolve(strict=True)
            if not mask_path.is_relative_to(result_path.parent.resolve()):
                raise ValueError("SAM3 target mask escapes result directory")
            expected_sha = str(view.get("mask_sha256") or "").lower()
            actual_sha = _remember_hash(mask_path, hashes)
            if actual_sha != expected_sha:
                raise ValueError("SAM3 target mask SHA-256 mismatch")
            mask_spec = {
                "path": str(mask_path),
                "bytes": mask_path.stat().st_size,
                "sha256": actual_sha,
                "mask_kind": "raw",
            }
            target = _load_mask(mask_path, mask_spec, field="SAM3 target mask")
            expected_shape = tuple(
                int(value) for value in heldout[source]["depth_size"]
            )
            if target.shape != expected_shape or not target.any():
                raise ValueError("SAM3 target mask is empty or has wrong heldout shape")
            targets[key] = {
                "object_id": object_id,
                "source_image": source,
                "physical_timestamp": timestamp,
                "rescue_image_id": int(view.get("rescue_image_id", -1)),
                "rescue_track_index": view.get("rescue_track_index"),
                "target_mask": mask_spec,
                "target_array": target,
            }
            seen_timestamps.add(timestamp_key)
    return targets


def _competitor_unions(
    state: Mapping[str, Any],
    mask_root: Path,
    heldout: Mapping[str, Mapping[str, Any]],
    targets: Mapping[tuple[int, str], Mapping[str, Any]],
    hashes: dict[Path, str],
) -> dict[tuple[int, str], np.ndarray]:
    _sources, by_source = _mapping_image_sources(state, heldout)
    observations = state.get("object_mask_observations")
    if not isinstance(observations, (list, tuple)):
        raise ValueError("YOLOE mapping lacks canonical object_mask_observations")
    per_image: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for track_index, track_rows in enumerate(observations):
        if track_rows is None:
            continue
        if not isinstance(track_rows, (list, tuple)):
            raise ValueError("YOLOE mask observations contain a non-list track")
        for observation in track_rows:
            if not isinstance(observation, Mapping):
                raise ValueError("YOLOE mask observations contain a non-object row")
            image_id = int(observation.get("image_id", -1))
            if image_id < 0 or image_id >= len(heldout):
                raise ValueError(
                    "YOLOE observation references an invalid heldout image"
                )
            path = _resolve_mapping_mask(observation, mask_root)
            _remember_hash(path, hashes)
            spec = {"path": str(path), "sha256": hashes[path], "mask_kind": "raw"}
            mask = _load_mask(path, spec, field="YOLOE competing mask")
            per_image[image_id].append((track_index, mask))
    unions: dict[tuple[int, str], np.ndarray] = {}
    for key, target in targets.items():
        source = str(target["source_image"])
        image_id = by_source[source]
        if int(target["rescue_image_id"]) != image_id:
            raise ValueError("SAM3 rescue_image_id differs from YOLOE heldout image")
        target_array = np.asarray(target["target_array"], dtype=bool)
        union = np.zeros(target_array.shape, dtype=bool)
        target_track = target.get("rescue_track_index")
        for track_index, mask in per_image.get(image_id, []):
            if target_track is not None and track_index == int(target_track):
                continue
            if mask.shape != union.shape:
                raise ValueError(
                    "YOLOE competing mask shape differs from heldout target"
                )
            union |= mask
        unions[key] = union & ~target_array
    return unions


def _validate_projection_manifest(
    path: Path,
    *,
    fold_sha: str,
    candidate_sha: str,
    heldout_sha: str,
    gaussian_hashes: Mapping[str, str],
    targets: Mapping[tuple[int, str], Mapping[str, Any]],
    heldout: Mapping[str, Mapping[str, Any]],
    hashes: dict[Path, str],
) -> dict[tuple[int, str], dict[str, Any]]:
    payload = _json_object(path)
    _remember_hash(path, hashes)
    if payload.get("schema") != PROJECTION_SCHEMA or payload.get("status") != "frozen":
        raise ValueError("projection manifest must be a frozen v1 manifest")
    if (
        payload.get("prediction_kind") != "post_lift_gaussian_projection"
        or payload.get("depth_unit") != "meters"
    ):
        raise ValueError("projection manifest kind or depth unit is invalid")
    links = {
        "fold_manifest_sha256": fold_sha,
        "candidate_manifest_sha256": candidate_sha,
        "heldout_frames_sha256": heldout_sha,
        **gaussian_hashes,
    }
    for name, expected in links.items():
        if str(payload.get(name) or "").lower() != expected:
            raise ValueError(f"projection manifest {name} mismatch")
    required = {
        "read_only": True,
        "candidate_frozen_before_projection": True,
        "candidate_ply_mutated": False,
        "candidate_labels_mutated": False,
        "heldout_updates_candidate": False,
        "renders_only_precommitted_heldout_views": True,
        "prediction_generated_without_target_masks": True,
        "targets_not_consumed_by_renderer": True,
        "prediction_depth_in_meters": True,
    }
    contract = payload.get("renderer_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != expected for key, expected in required.items()
    ):
        raise ValueError("projection renderer read-only contract is incomplete")
    _verify_artifact(
        payload.get("renderer_implementation"),
        base=path.parent,
        field="projection.renderer_implementation",
        hashes=hashes,
    )
    rows = payload.get("observations")
    if not isinstance(rows, list):
        raise ValueError("projection manifest observations must be a list")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("projection manifest contains a non-object row")
        object_id = int(row.get("object_id"))
        source = str(row.get("source_image") or "").strip()
        key = (object_id, source)
        if key in result or key not in targets:
            raise ValueError("projection is duplicate or lacks a frozen SAM3 target")
        timestamp = str(heldout[source]["physical_timestamp"])
        if str(row.get("physical_timestamp") or "") != timestamp:
            raise ValueError("projection physical timestamp mismatch")
        prediction_mask_path, mask_spec = _verify_artifact(
            row.get("prediction_mask"),
            base=path.parent,
            field=f"projection observation {index} mask",
            hashes=hashes,
        )
        prediction_depth_path, depth_spec = _verify_artifact(
            row.get("prediction_depth"),
            base=path.parent,
            field=f"projection observation {index} depth",
            hashes=hashes,
        )
        mask = _load_mask(prediction_mask_path, mask_spec, field="prediction mask")
        depth = _load_depth(prediction_depth_path, depth_spec, field="prediction depth")
        expected_shape = tuple(int(value) for value in heldout[source]["depth_size"])
        if mask.shape != expected_shape or depth.shape != expected_shape:
            raise ValueError("projection artifact shape differs from heldout frame")
        result[key] = {
            "prediction_mask": {**dict(mask_spec), "path": str(prediction_mask_path)},
            "prediction_depth": {
                **dict(depth_spec),
                "path": str(prediction_depth_path),
            },
        }
    if set(result) != set(targets):
        missing = sorted(set(targets).difference(result))
        extra = sorted(set(result).difference(targets))
        raise ValueError(
            "projection observations must exactly cover frozen targets: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    return result


def prepare_frozen_heldout_evidence(
    fold_manifest_path: Path,
    candidate_manifest_path: Path,
    source_ply_path: Path,
    source_labels_path: Path,
    candidate_ply_path: Path,
    candidate_labels_path: Path,
    heldout_reference_result_path: Path,
    heldout_yoloe_state_path: Path,
    heldout_yoloe_mask_root: Path,
    *,
    meters_per_scene_unit: float,
    projection_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all inputs and prepare evidence without writing or rendering."""

    scale = _positive_scale(meters_per_scene_unit)
    fold_path = Path(fold_manifest_path).expanduser().resolve(strict=True)
    candidate_path = Path(candidate_manifest_path).expanduser().resolve(strict=True)
    reference_path = (
        Path(heldout_reference_result_path).expanduser().resolve(strict=True)
    )
    mapping_path = Path(heldout_yoloe_state_path).expanduser().resolve(strict=True)
    mask_root = Path(heldout_yoloe_mask_root).expanduser().resolve(strict=True)
    explicit = {
        name: Path(value).expanduser().resolve(strict=True)
        for name, value in {
            "source_ply": source_ply_path,
            "source_labels": source_labels_path,
            "candidate_ply": candidate_ply_path,
            "candidate_labels": candidate_labels_path,
        }.items()
    }
    hashes: dict[Path, str] = {}
    (
        _fold,
        train,
        heldout,
        membership,
        _depth_specs,
        train_frames_path,
        heldout_frames_path,
    ) = _validate_fold_manifest(fold_path, meters_per_scene_unit=scale, hashes=hashes)
    candidate, candidate_ids_list = _validate_candidate(
        candidate_path,
        fold_sha256=hashes[fold_path],
        train_sha256=hashes[train_frames_path],
        scale=scale,
        train=train,
        heldout=heldout,
        membership=membership,
        hashes=hashes,
    )
    _resolve_exact_gaussian_artifacts(candidate, candidate_path, explicit, hashes)
    candidate_ids = set(candidate_ids_list)
    label_ids = {
        int(value)
        for value in np.unique(_labels(explicit["candidate_labels"]))
        if int(value) >= 0
    }
    if not candidate_ids.issubset(label_ids):
        raise ValueError("candidate object IDs are absent from candidate labels")
    mapping_sha = _remember_hash(mapping_path, hashes)
    state = _load_mapping_state(mapping_path)
    targets = _validate_reference_targets(
        reference_path,
        heldout_frames_sha=hashes[heldout_frames_path],
        mapping_state_sha=mapping_sha,
        candidate_ids=candidate_ids,
        heldout=heldout,
        membership=membership,
        hashes=hashes,
    )
    unions = _competitor_unions(state, mask_root, heldout, targets, hashes)
    gaussian_hashes = {
        f"{name}_sha256": hashes[path] for name, path in explicit.items()
    }
    projections: dict[tuple[int, str], dict[str, Any]] | None = None
    projection_path: Path | None = None
    if projection_manifest_path is not None:
        projection_path = (
            Path(projection_manifest_path).expanduser().resolve(strict=True)
        )
        projections = _validate_projection_manifest(
            projection_path,
            fold_sha=hashes[fold_path],
            candidate_sha=hashes[candidate_path],
            heldout_sha=hashes[heldout_frames_path],
            gaussian_hashes=gaussian_hashes,
            targets=targets,
            heldout=heldout,
            hashes=hashes,
        )
    mutated = [
        str(path) for path, digest in hashes.items() if sha256_file(path) != digest
    ]
    if mutated:
        raise RuntimeError(
            f"frozen input changed during evidence preparation: {mutated[:8]}"
        )
    return {
        "schema": BUILD_SCHEMA,
        "status": "PASS" if projections is not None else "BLOCKED",
        "reason": (
            None
            if projections is not None
            else "read_only_gaussian_projection_renderer_required"
        ),
        "renderer_implemented_here": False,
        "contract": {
            "read_only": True,
            "training_performed": False,
            "state_mutated": False,
            "surrogate_prediction_forbidden": True,
            "target_masks_are_pseudo_labels_not_ground_truth": True,
            "heldout_used_for_fit": False,
            "projection_renderer_is_separate": True,
        },
        "meters_per_scene_unit": scale,
        "paths": {
            "fold_manifest": str(fold_path),
            "candidate_manifest": str(candidate_path),
            "heldout_reference_result": str(reference_path),
            "heldout_yoloe_state": str(mapping_path),
            "heldout_yoloe_mask_root": str(mask_root),
            "projection_manifest": str(projection_path) if projection_path else None,
        },
        "hashes": {
            "fold_manifest_sha256": hashes[fold_path],
            "candidate_manifest_sha256": hashes[candidate_path],
            "train_frames_sha256": hashes[train_frames_path],
            "heldout_frames_sha256": hashes[heldout_frames_path],
            "heldout_reference_result_sha256": hashes[reference_path],
            "heldout_yoloe_state_sha256": mapping_sha,
            **gaussian_hashes,
        },
        "candidate_object_ids": sorted(candidate_ids),
        "target_count": len(targets),
        "targets": [targets[key] for key in sorted(targets)],
        "competing_unions": unions,
        "projections": projections,
        "consumed_input_hashes": {
            str(path): digest
            for path, digest in sorted(hashes.items(), key=lambda item: str(item[0]))
        },
    }


def materialize_evidence_payload(
    prepared: Mapping[str, Any],
    competing_specs: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact manifest consumed by ``frozen_heldout_qc.py``."""

    if prepared.get("status") != "PASS" or not isinstance(
        prepared.get("projections"), Mapping
    ):
        raise ValueError("cannot materialize evidence without attested projections")
    projections = prepared["projections"]
    observations = []
    for target in prepared.get("targets") or []:
        key = (int(target["object_id"]), str(target["source_image"]))
        projection = projections.get(key)
        competing = competing_specs.get(key)
        if not isinstance(projection, Mapping) or not isinstance(competing, Mapping):
            raise ValueError("missing projection or competing union for target")
        observations.append(
            {
                "object_id": key[0],
                "source_image": key[1],
                "physical_timestamp": str(target["physical_timestamp"]),
                "prediction_mask": projection["prediction_mask"],
                "target_mask": target["target_mask"],
                "competing_mask": dict(competing),
                "prediction_depth": projection["prediction_depth"],
            }
        )
    hashes = prepared["hashes"]
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "frozen",
        "meters_per_scene_unit": float(prepared["meters_per_scene_unit"]),
        "prediction_kind": "post_lift_gaussian_projection",
        "depth_unit": "meters",
        "fold_manifest_sha256": hashes["fold_manifest_sha256"],
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "heldout_frames_sha256": hashes["heldout_frames_sha256"],
        "contract": {
            "view_selection_precommitted": True,
            "prediction_uses_only_frozen_candidate": True,
            "targets_evaluation_only": True,
            "heldout_never_updates_candidate": True,
            "masks_obb_labels_not_modified": True,
        },
        "provenance": {
            "builder_schema": BUILD_SCHEMA,
            "heldout_reference_result_sha256": hashes[
                "heldout_reference_result_sha256"
            ],
            "heldout_yoloe_state_sha256": hashes["heldout_yoloe_state_sha256"],
            "projection_manifest": prepared["paths"]["projection_manifest"],
        },
        "observations": observations,
    }
