"""Target-blind request and input contracts for heldout Gaussian projection.

The request builder may inspect the *metadata* of the heldout SAM3 result in
order to select the precommitted object/view pairs.  It deliberately never
opens the referenced target masks.  The GPU renderer consumes only the
sanitised request, calibrated heldout cameras and frozen Gaussian artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from farm_runtime.frozen_heldout_evidence import (
    FIT_CONTRACT_SCHEMA,
    REFINEMENT_ROLE,
    REFINEMENT_SCHEMA,
    _json_object,
    _positive_scale,
    _resolve_exact_gaussian_artifacts,
    _timestamp_key,
    _verify_gaussian_shapes,
)
from farm_runtime.frozen_heldout_qc import (
    _remember_hash,
    _validate_candidate,
    _validate_fold_manifest,
    _verify_artifact,
    sha256_file,
)

PROJECTION_REQUEST_SCHEMA = "farm.frozen-heldout-gaussian-projection-request.v1"
PROJECTION_RENDERER_CONFIG_SCHEMA = "farm.gaussian-projection-raster-config.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RGBD_UNION_MANIFEST_SCHEMA = "farm.frozen-evaluation-rgbd-union-manifest.v1"
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _finite(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(
            f"{field} must be finite" + (" and positive" if positive else "")
        )
    return result


def _heldout_reference_rows(
    path: Path,
    *,
    heldout_frames_sha256: str,
    candidate_ids: set[int],
    heldout: Mapping[str, Mapping[str, Any]],
    membership: Mapping[int, Mapping[str, Sequence[str]]],
    hashes: dict[Path, str],
) -> list[dict[str, Any]]:
    """Read target-free routing metadata from the heldout-reference result."""

    payload = _json_object(path)
    _remember_hash(path, hashes)
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
        or view_contract.get("active_split") != "heldout"
        or view_contract.get("state_fit_authorized") is not False
        or view_contract.get("requested_view_role") != "heldout-reference"
        or not isinstance(policy, Mapping)
        or policy.get("state_fit_authorized") is not False
        or policy.get("merge_apply_forbidden") is not True
        or policy.get("heldout_reference_is_evaluation_only") is not True
        or policy.get("reference_masks_are_pseudo_labels_not_ground_truth") is not True
    ):
        raise ValueError("SAM3 heldout-reference policy is incomplete")
    declared = payload.get("hashes")
    if (
        not isinstance(declared, Mapping)
        or str(declared.get("rescue_frames_sha256") or "").lower()
        != heldout_frames_sha256
    ):
        raise ValueError("SAM3 reference uses a different heldout frames JSON")

    rows: list[dict[str, Any]] = []
    keys: set[tuple[int, str]] = set()
    timestamp_keys: set[tuple[int, str]] = set()
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("SAM3 heldout-reference result has no object rows")
    for object_row in objects:
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
            # Require the reference to declare a target, but never resolve/open it.
            relative = Path(str(view.get("mask_relative") or ""))
            target_sha = str(view.get("mask_sha256") or "").lower()
            if (
                relative.is_absolute()
                or not relative.parts
                or not SHA256_RE.fullmatch(target_sha)
            ):
                raise ValueError("SAM3 target declaration is incomplete")
            key = (object_id, source)
            timestamp_key = (object_id, _timestamp_key(timestamp))
            if key in keys or timestamp_key in timestamp_keys:
                raise ValueError("duplicate SAM3 target source or physical timestamp")
            rows.append(
                {
                    "object_id": object_id,
                    "source_image": source,
                    "physical_timestamp": timestamp,
                }
            )
            keys.add(key)
            timestamp_keys.add(timestamp_key)
    if not rows:
        raise ValueError("heldout-reference result has no accepted candidate views")
    return sorted(rows, key=lambda row: (row["source_image"], row["object_id"]))


def _canonical_rgbd_transitive_specs(
    manifest: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Return exact ordered artifacts referenced beyond explicit renderer inputs."""

    actual_fit = manifest.get("actual_fit_frames")
    sources = manifest.get("source_rgbd_runs")
    frames = manifest.get("frame_artifacts")
    if not isinstance(actual_fit, Mapping):
        raise ValueError("canonical RGBD union lacks actual-fit provenance")
    if not isinstance(sources, list) or not sources:
        raise ValueError("canonical RGBD union lacks source-run provenance")
    if not isinstance(frames, list) or not frames:
        raise ValueError("canonical RGBD union lacks per-frame provenance")
    result: list[tuple[str, Mapping[str, Any]]] = [("actual_fit_frames", actual_fit)]
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError("canonical RGBD source-run provenance is invalid")
        for name in ("render_manifest", "union_frames", "fold_manifest"):
            artifact = source.get(name)
            if not isinstance(artifact, Mapping):
                raise ValueError(
                    f"canonical RGBD source run {source_index} {name} is invalid"
                )
            result.append((f"source_rgbd_runs[{source_index}].{name}", artifact))
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError("canonical RGBD per-frame provenance is invalid")
        for name in ("rgb", "depth"):
            artifact = frame.get(name)
            if not isinstance(artifact, Mapping):
                raise ValueError(
                    f"canonical RGBD frame {frame_index} {name} is invalid"
                )
            result.append((f"frame_artifacts[{frame_index}].{name}", artifact))
    return result


def _canonical_rgbd_transitive_contract(
    manifest: Mapping[str, Any], *, manifest_sha256: str
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for field, artifact in _canonical_rgbd_transitive_specs(manifest):
        raw_path = str(artifact.get("path") or "").strip()
        digest = str(artifact.get("sha256") or "").lower()
        size = artifact.get("bytes")
        if (
            not raw_path
            or not SHA256_RE.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"canonical RGBD {field} declaration is invalid")
        entries.append({"field": field, "bytes": size, "sha256": digest})
    entries_sha256 = hashlib.sha256(
        json.dumps(
            entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    sources = manifest["source_rgbd_runs"]
    frames = manifest["frame_artifacts"]
    return {
        "schema": "farm.rgbd-transitive-artifact-commitment.v1",
        "rgbd_render_manifest_sha256": manifest_sha256,
        "source_run_count": len(sources),
        "frame_count": len(frames),
        "artifact_count": len(entries),
        "entries_sha256": entries_sha256,
        "paths_are_host_provenance_not_container_authority": True,
    }


def _verify_canonical_rgbd_transitive_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    hashes: dict[Path, str],
) -> dict[str, Any]:
    contract = _canonical_rgbd_transitive_contract(
        manifest, manifest_sha256=_remember_hash(manifest_path, hashes)
    )
    for field, artifact in _canonical_rgbd_transitive_specs(manifest):
        path = Path(str(artifact["path"])).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"canonical RGBD {field} artifact is unavailable") from exc
        if (
            path.stat().st_size != int(artifact["bytes"])
            or _remember_hash(path, hashes) != artifact["sha256"]
        ):
            raise ValueError(f"canonical RGBD {field} SHA/size binding mismatch")
    return contract


def _rgbd_render_contract(
    manifest_path: Path,
    union_frames_path: Path,
    heldout_frames_path: Path,
    *,
    scale: float,
    source_ply: Path,
    hashes: dict[Path, str],
    verify_transitive_artifacts: bool = True,
) -> dict[str, Any]:
    manifest = _json_object(manifest_path)
    union = _json_object(union_frames_path)
    heldout = _json_object(heldout_frames_path)
    for path in (manifest_path, union_frames_path, heldout_frames_path):
        _remember_hash(path, hashes)
    if manifest_path.parent != union_frames_path.parent:
        raise ValueError("RGBD render manifest and union frames must share a directory")

    schema = manifest.get("schema_version")
    frame_artifacts: list[object] | None = None
    if schema == "farm_rgbd_run_manifest_v1":
        transitive_contract = None
        if (
            manifest.get("status") != "complete"
            or manifest.get("qa_passed") is not True
            or str(manifest.get("frames") or "") != union_frames_path.name
        ):
            raise ValueError("RGBD render manifest is not the completed union run")
        payload = manifest.get("fingerprint_payload")
        config = payload.get("config") if isinstance(payload, Mapping) else None
        inputs = payload.get("inputs") if isinstance(payload, Mapping) else None
        ply = inputs.get("ply") if isinstance(inputs, Mapping) else None
        if not isinstance(config, Mapping) or not isinstance(ply, Mapping):
            raise ValueError("RGBD run lacks its raster config/input provenance")
        ply_size = int(ply.get("size", -1))
    elif schema == RGBD_UNION_MANIFEST_SCHEMA:
        required_contract = {
            "read_only_existing_renders": True,
            "union_did_not_render_or_refit": True,
            "source_runs_qa_passed": True,
            "source_runs_sha256_bound": True,
            "all_frame_assets_sha256_hashed": True,
            "actual_fit_frames_sha256_bound": True,
            "heldout_state_fit_authorized": False,
        }
        contract = manifest.get("contract")
        frames_artifact = manifest.get("frames_artifact")
        ply = manifest.get("source_ply")
        actual_fit = manifest.get("actual_fit_frames")
        if (
            manifest.get("status") != "frozen"
            or manifest.get("qa_passed") is not True
            or not isinstance(contract, Mapping)
            or any(
                contract.get(key) != value for key, value in required_contract.items()
            )
            or not isinstance(frames_artifact, Mapping)
            or str(frames_artifact.get("path") or "")
            != str(manifest.get("frames") or "")
            or not isinstance(ply, Mapping)
            or not isinstance(actual_fit, Mapping)
        ):
            raise ValueError("canonical frozen RGBD union contract is incomplete")
        explicit_artifacts = (
            ("canonical RGBD union frames", frames_artifact, union_frames_path),
            ("canonical RGBD source PLY", ply, source_ply),
        )
        for field, artifact, explicit_path in explicit_artifacts:
            digest = str(artifact.get("sha256") or "")
            size = artifact.get("bytes")
            if (
                not SHA256_RE.fullmatch(digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or digest != _remember_hash(explicit_path, hashes)
                or size != explicit_path.stat().st_size
            ):
                raise ValueError(f"{field} SHA/size binding mismatch")
        config = manifest.get("render_config")
        sources = manifest.get("source_rgbd_runs")
        frame_artifacts = manifest.get("frame_artifacts")
        if (
            not isinstance(config, Mapping)
            or not isinstance(sources, list)
            or not sources
        ):
            raise ValueError("canonical RGBD union lacks render/source provenance")
        if not isinstance(frame_artifacts, list) or not frame_artifacts:
            raise ValueError("canonical RGBD union lacks per-frame provenance")
        transitive_contract = _canonical_rgbd_transitive_contract(
            manifest, manifest_sha256=hashes[manifest_path]
        )
        for source_index, source in enumerate(sources):
            if not isinstance(source, Mapping):
                raise ValueError("canonical RGBD source-run provenance is invalid")
            for field in ("render_manifest", "union_frames", "fold_manifest"):
                artifact = source.get(field)
                if not isinstance(artifact, Mapping):
                    raise ValueError(
                        f"canonical RGBD source run {source_index} {field} is invalid"
                    )
        ply_size = int(ply.get("bytes", -1))
    else:
        raise ValueError("RGBD render manifest has unsupported schema")
    if union.get("schema_version") != "farm_frames_json_v1":
        raise ValueError("union frames JSON has unsupported schema")
    fold = heldout.get("full_colmap_fold")
    if (
        not isinstance(fold, Mapping)
        or fold.get("role") != "heldout"
        or fold.get("state_fit_authorized") is not False
        or str(fold.get("source_union_frames_sha256") or "").lower()
        != hashes[union_frames_path]
    ):
        raise ValueError("heldout frames do not bind the supplied union frames")
    if (
        not str(manifest.get("scene_id") or "").strip()
        or str(manifest.get("scene_id") or "") != str(heldout.get("scene_id") or "")
        or str(union.get("scene_id") or "") != str(heldout.get("scene_id") or "")
    ):
        raise ValueError("RGBD union/heldout scene IDs differ")
    if int(manifest.get("selected_view_count", -1)) != len(union.get("frames") or []):
        raise ValueError("RGBD union selected-view count mismatch")

    config_scale = _finite(
        config.get("meters_per_scene_unit"),
        field="rgbd.config.meters_per_scene_unit",
        positive=True,
    )
    if abs(config_scale - scale) > 1.0e-12:
        raise ValueError("RGBD render scale differs from frozen candidate")
    if ply_size != source_ply.stat().st_size:
        raise ValueError("RGBD render source PLY size differs from frozen source")
    values = {
        "depth_min_m": _finite(
            config.get("depth_min_m"), field="depth_min_m", positive=True
        ),
        "depth_max_m": _finite(
            config.get("depth_max_m"), field="depth_max_m", positive=True
        ),
        "radius_clip": _finite(
            config.get("radius_clip"), field="radius_clip", positive=True
        ),
        "alpha_min": _finite(config.get("alpha_min"), field="alpha_min", positive=True),
    }
    if values["depth_max_m"] <= values["depth_min_m"] or values["alpha_min"] > 1.0:
        raise ValueError("RGBD render depth/alpha range is invalid")

    union_by_source = {
        str(row.get("source_image") or ""): row for row in union.get("frames") or []
    }
    heldout_rows = heldout.get("frames") or []
    if len(union_by_source) != len(union.get("frames") or []):
        raise ValueError("RGBD union source images are not unique")
    camera_fields = ("K", "T_world_cam", "depth_size", "timestamp_ns", "frame_id")
    if frame_artifacts is not None:
        if len(frame_artifacts) != len(union_by_source):
            raise ValueError("canonical RGBD per-frame provenance count mismatch")
        provenance_by_source: dict[str, Mapping[str, Any]] = {}
        for raw in frame_artifacts:
            if not isinstance(raw, Mapping):
                raise ValueError("canonical RGBD per-frame provenance is invalid")
            source = str(raw.get("source_image") or "").strip()
            if not source or source in provenance_by_source:
                raise ValueError(
                    "canonical RGBD per-frame provenance has duplicate/empty sources"
                )
            union_row = union_by_source.get(source)
            split = str(raw.get("split") or "")
            if (
                union_row is None
                or split not in {"train", "heldout"}
                or split != str(union_row.get("full_colmap_split") or "")
                or str(raw.get("physical_timestamp") or "")
                != str(union_row.get("physical_timestamp") or "")
            ):
                raise ValueError(
                    f"canonical RGBD per-frame identity mismatch for {source}"
                )
            for artifact_name, row_field in (
                ("rgb", "rgb_path"),
                ("depth", "depth_path"),
            ):
                artifact = raw.get(artifact_name)
                if not isinstance(artifact, Mapping):
                    raise ValueError(
                        f"canonical RGBD {artifact_name} provenance is invalid for {source}"
                    )
                artifact_path = str(artifact.get("path") or "")
                artifact_size = artifact.get("bytes")
                if (
                    not artifact_path
                    or Path(artifact_path).is_absolute()
                    or artifact_path != str(union_row.get(row_field) or "")
                    or not SHA256_RE.fullmatch(str(artifact.get("sha256") or ""))
                    or isinstance(artifact_size, bool)
                    or not isinstance(artifact_size, int)
                    or artifact_size < 1
                ):
                    raise ValueError(
                        f"canonical RGBD {artifact_name} provenance mismatch for {source}"
                    )
            provenance_by_source[source] = raw
        if set(provenance_by_source) != set(union_by_source):
            raise ValueError("canonical RGBD per-frame provenance coverage mismatch")
        if verify_transitive_artifacts:
            _verify_canonical_rgbd_transitive_artifacts(manifest_path, manifest, hashes)
    for row in heldout_rows:
        source = str(row.get("source_image") or "")
        union_row = union_by_source.get(source)
        if union_row is None or any(
            union_row.get(key) != row.get(key) for key in camera_fields
        ):
            raise ValueError(
                f"heldout calibration differs from RGBD union for {source}"
            )
    return {
        "schema": PROJECTION_RENDERER_CONFIG_SCHEMA,
        **values,
        "eps2d": 0.3,
        "packed": True,
        "tile_size": 16,
        "rasterize_mode": "antialiased",
        "camera_model": "pinhole",
        "render_mode": "RGB+ED",
        "object_depth": "opacity_weighted_expected_camera_z",
        "rgbd_transitive_contract": transitive_contract,
    }


def materialize_projection_request(
    fold_manifest_path: Path,
    candidate_manifest_path: Path,
    source_ply_path: Path,
    source_labels_path: Path,
    candidate_ply_path: Path,
    candidate_labels_path: Path,
    heldout_reference_result_path: Path,
    rgbd_render_manifest_path: Path,
    union_frames_path: Path,
    *,
    meters_per_scene_unit: float,
    alpha_threshold: float,
) -> dict[str, Any]:
    """Build a target-blind, immutable request for the separate GPU renderer."""

    scale = _positive_scale(meters_per_scene_unit)
    threshold = _finite(alpha_threshold, field="alpha_threshold", positive=True)
    if threshold > 1.0:
        raise ValueError("alpha_threshold must be <= 1")
    paths = {
        name: Path(value).expanduser().resolve(strict=True)
        for name, value in {
            "fold_manifest": fold_manifest_path,
            "candidate_manifest": candidate_manifest_path,
            "source_ply": source_ply_path,
            "source_labels": source_labels_path,
            "candidate_ply": candidate_ply_path,
            "candidate_labels": candidate_labels_path,
            "heldout_reference_result": heldout_reference_result_path,
            "rgbd_render_manifest": rgbd_render_manifest_path,
            "union_frames": union_frames_path,
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
    ) = _validate_fold_manifest(
        paths["fold_manifest"], meters_per_scene_unit=scale, hashes=hashes
    )
    candidate, candidate_ids = _validate_candidate(
        paths["candidate_manifest"],
        fold_sha256=hashes[paths["fold_manifest"]],
        train_sha256=hashes[train_frames_path],
        scale=scale,
        train=train,
        heldout=heldout,
        membership=membership,
        hashes=hashes,
    )
    explicit = {
        name: paths[name]
        for name in ("source_ply", "source_labels", "candidate_ply", "candidate_labels")
    }
    _resolve_exact_gaussian_artifacts(
        candidate, paths["candidate_manifest"], explicit, hashes
    )
    upstream_path, _ = _verify_artifact(
        candidate.get("upstream_lift_result"),
        base=paths["candidate_manifest"].parent,
        field="candidate.upstream_lift_result",
        hashes=hashes,
    )
    upstream = _json_object(upstream_path)
    runtime = upstream.get("runtime")
    image = (
        str(runtime.get("container_image") or "")
        if isinstance(runtime, Mapping)
        else ""
    )
    image_id = (
        str(runtime.get("container_image_id") or "").lower()
        if isinstance(runtime, Mapping)
        else ""
    )
    if not image or not IMAGE_ID_RE.fullmatch(image_id):
        raise ValueError(
            "upstream lift does not pin a projection-capable runtime image"
        )
    source_labels, candidate_labels = _verify_gaussian_shapes(
        paths["source_ply"],
        paths["source_labels"],
        paths["candidate_ply"],
        paths["candidate_labels"],
    )
    config = _rgbd_render_contract(
        paths["rgbd_render_manifest"],
        paths["union_frames"],
        heldout_frames_path,
        scale=scale,
        source_ply=paths["source_ply"],
        hashes=hashes,
    )
    config["object_alpha_threshold"] = threshold
    transitive_contract = config.pop("rgbd_transitive_contract")
    rows = _heldout_reference_rows(
        paths["heldout_reference_result"],
        heldout_frames_sha256=hashes[heldout_frames_path],
        candidate_ids=set(candidate_ids),
        heldout=heldout,
        membership=membership,
        hashes=hashes,
    )
    requested_ids = {int(row["object_id"]) for row in rows}
    source_available = {
        int(value) for value in np.unique(source_labels) if int(value) >= 0
    }
    candidate_available = {
        int(value) for value in np.unique(candidate_labels) if int(value) >= 0
    }
    if not requested_ids.issubset(candidate_available):
        raise ValueError("projection request object is absent from candidate labels")

    result = {
        "schema": PROJECTION_REQUEST_SCHEMA,
        "status": "frozen",
        "meters_per_scene_unit": scale,
        "fold_manifest_sha256": hashes[paths["fold_manifest"]],
        "candidate_manifest_sha256": hashes[paths["candidate_manifest"]],
        "train_frames_sha256": hashes[train_frames_path],
        "heldout_frames_sha256": hashes[heldout_frames_path],
        "heldout_reference_result_sha256": hashes[paths["heldout_reference_result"]],
        "rgbd_render_manifest_sha256": hashes[paths["rgbd_render_manifest"]],
        "union_frames_sha256": hashes[paths["union_frames"]],
        **{f"{name}_sha256": hashes[path] for name, path in explicit.items()},
        "gaussian_counts": {
            "source": int(source_labels.size),
            "candidate": int(candidate_labels.size),
        },
        "renderer_config": config,
        "rgbd_transitive_artifact_contract": transitive_contract,
        "renderer_runtime": {
            "image": image,
            "image_id": image_id,
            "python": "/opt/conda/envs/rest3d/bin/python",
            "source": "candidate.upstream_lift_result.runtime",
        },
        "request_contract": {
            "view_selection_precommitted": True,
            "heldout_reference_metadata_only": True,
            "target_mask_paths_omitted": True,
            "target_mask_bytes_read": False,
            "renderer_must_not_receive_target_artifacts": True,
            "source_and_candidate_labels_frozen": True,
            "heldout_updates_candidate": False,
            "rgbd_transitive_artifacts_host_verified": transitive_contract is not None,
            "rgbd_transitive_declarations_sha256_bound": transitive_contract
            is not None,
            "renderer_rgbd_transitive_artifacts_unmounted": transitive_contract
            is not None,
        },
        "objects": [
            {
                "object_id": object_id,
                "source_label_available": object_id in source_available,
                "candidate_label_available": object_id in candidate_available,
            }
            for object_id in sorted(requested_ids)
        ],
        "observations": rows,
        "provenance": {
            "fit_contract_schema": FIT_CONTRACT_SCHEMA,
            "consumed_input_hashes": {
                str(path): digest
                for path, digest in sorted(
                    hashes.items(), key=lambda item: str(item[0])
                )
            },
        },
    }
    changed = [
        str(path) for path, digest in hashes.items() if sha256_file(path) != digest
    ]
    if changed:
        raise RuntimeError(
            f"frozen input changed during projection request build: {changed[:8]}"
        )
    return result


def validate_renderer_inputs(
    request_path: Path,
    heldout_frames_path: Path,
    rgbd_render_manifest_path: Path,
    union_frames_path: Path,
    source_ply_path: Path,
    source_labels_path: Path,
    candidate_ply_path: Path,
    candidate_labels_path: Path,
    *,
    verify_rgbd_transitive_artifacts: bool = True,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, str]]:
    """Target-free validation used inside the isolated renderer process."""

    paths = {
        name: Path(value).expanduser().resolve(strict=True)
        for name, value in {
            "request": request_path,
            "heldout_frames": heldout_frames_path,
            "rgbd_render_manifest": rgbd_render_manifest_path,
            "union_frames": union_frames_path,
            "source_ply": source_ply_path,
            "source_labels": source_labels_path,
            "candidate_ply": candidate_ply_path,
            "candidate_labels": candidate_labels_path,
        }.items()
    }
    request = _json_object(paths["request"])
    if (
        request.get("schema") != PROJECTION_REQUEST_SCHEMA
        or request.get("status") != "frozen"
    ):
        raise ValueError("projection request must be frozen v1")
    required_contract = {
        "view_selection_precommitted": True,
        "heldout_reference_metadata_only": True,
        "target_mask_paths_omitted": True,
        "target_mask_bytes_read": False,
        "renderer_must_not_receive_target_artifacts": True,
        "source_and_candidate_labels_frozen": True,
        "heldout_updates_candidate": False,
    }
    contract = request.get("request_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in required_contract.items()
    ):
        raise ValueError("projection request target-blind contract is incomplete")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    links = {
        "heldout_frames_sha256": hashes["heldout_frames"],
        "rgbd_render_manifest_sha256": hashes["rgbd_render_manifest"],
        "union_frames_sha256": hashes["union_frames"],
        "source_ply_sha256": hashes["source_ply"],
        "source_labels_sha256": hashes["source_labels"],
        "candidate_ply_sha256": hashes["candidate_ply"],
        "candidate_labels_sha256": hashes["candidate_labels"],
    }
    for field, expected in links.items():
        if str(request.get(field) or "").lower() != expected:
            raise ValueError(f"projection request {field} mismatch")
    scale = _positive_scale(request.get("meters_per_scene_unit"))
    heldout = _json_object(paths["heldout_frames"])
    if (
        heldout.get("schema_version") != "farm_frames_json_v1"
        or heldout.get("depth_units") != "metres"
        or heldout.get("pose_translation_units") != "metres"
        or abs(float(heldout.get("meters_per_scene_unit", 0.0)) - scale) > 1.0e-12
    ):
        raise ValueError("heldout frames render contract is invalid")
    frame_rows = heldout.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("heldout frames are empty")
    frames: dict[str, Mapping[str, Any]] = {}
    for row in frame_rows:
        if not isinstance(row, Mapping):
            raise ValueError("heldout frame row is invalid")
        source = str(row.get("source_image") or "").strip()
        K = np.asarray(row.get("K"), dtype=np.float64)
        pose = np.asarray(row.get("T_world_cam"), dtype=np.float64)
        size = tuple(int(value) for value in row.get("depth_size") or [])
        if (
            not source
            or source in frames
            or K.shape != (3, 3)
            or pose.shape != (4, 4)
            or not np.isfinite(K).all()
            or not np.isfinite(pose).all()
            or len(size) != 2
            or min(size) <= 0
        ):
            raise ValueError("heldout frame calibration is invalid")
        frames[source] = row
    source_labels, candidate_labels = _verify_gaussian_shapes(
        paths["source_ply"],
        paths["source_labels"],
        paths["candidate_ply"],
        paths["candidate_labels"],
    )
    counts = request.get("gaussian_counts")
    if (
        not isinstance(counts, Mapping)
        or int(counts.get("source", -1)) != source_labels.size
        or int(counts.get("candidate", -1)) != candidate_labels.size
    ):
        raise ValueError("projection request Gaussian counts mismatch")
    candidate_ids = {
        int(value) for value in np.unique(candidate_labels) if int(value) >= 0
    }
    observations = request.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("projection request observations are empty")
    keys: set[tuple[int, str]] = set()
    for row in observations:
        if not isinstance(row, Mapping) or set(row) != {
            "object_id",
            "source_image",
            "physical_timestamp",
        }:
            raise ValueError("projection request observation exposes unexpected fields")
        object_id = int(row["object_id"])
        source = str(row["source_image"])
        frame = frames.get(source)
        key = (object_id, source)
        if (
            frame is None
            or object_id not in candidate_ids
            or key in keys
            or _timestamp_key(row["physical_timestamp"])
            != _timestamp_key(frame.get("physical_timestamp"))
        ):
            raise ValueError("projection request observation is invalid")
        keys.add(key)
    config = request.get("renderer_config")
    if (
        not isinstance(config, Mapping)
        or config.get("schema") != PROJECTION_RENDERER_CONFIG_SCHEMA
    ):
        raise ValueError("projection request renderer config is invalid")
    # Recompute and compare the critical RGBD raster parameters without opening targets.
    rgbd_hashes: dict[Path, str] = {}
    computed = _rgbd_render_contract(
        paths["rgbd_render_manifest"],
        paths["union_frames"],
        paths["heldout_frames"],
        scale=scale,
        source_ply=paths["source_ply"],
        hashes=rgbd_hashes,
        verify_transitive_artifacts=verify_rgbd_transitive_artifacts,
    )
    transitive_contract = computed.pop("rgbd_transitive_contract")
    if transitive_contract is not None:
        if request.get("rgbd_transitive_artifact_contract") != transitive_contract:
            raise ValueError(
                "projection request RGBD transitive artifact contract mismatch"
            )
        required_transitive_contract = {
            "rgbd_transitive_artifacts_host_verified": True,
            "rgbd_transitive_declarations_sha256_bound": True,
            "renderer_rgbd_transitive_artifacts_unmounted": True,
        }
        if any(
            contract.get(key) != value
            for key, value in required_transitive_contract.items()
        ):
            raise ValueError(
                "projection request RGBD transitive verification policy is incomplete"
            )
    elif request.get("rgbd_transitive_artifact_contract") is not None:
        raise ValueError("legacy RGBD request has an unexpected transitive contract")
    for field in (
        "depth_min_m",
        "depth_max_m",
        "radius_clip",
        "alpha_min",
        "eps2d",
        "packed",
        "tile_size",
        "rasterize_mode",
        "camera_model",
        "render_mode",
        "object_depth",
    ):
        if config.get(field) != computed[field]:
            raise ValueError(f"projection renderer config drift: {field}")
    threshold = _finite(
        config.get("object_alpha_threshold"),
        field="object_alpha_threshold",
        positive=True,
    )
    if threshold > 1.0:
        raise ValueError("object_alpha_threshold must be <= 1")
    runtime = request.get("renderer_runtime")
    if (
        not isinstance(runtime, Mapping)
        or not str(runtime.get("image") or "")
        or not IMAGE_ID_RE.fullmatch(str(runtime.get("image_id") or "").lower())
        or runtime.get("python") != "/opt/conda/envs/rest3d/bin/python"
        or runtime.get("source") != "candidate.upstream_lift_result.runtime"
    ):
        raise ValueError("projection request runtime pin is invalid")
    return request, frames, hashes
