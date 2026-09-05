"""Fail-closed selection of objects that may proceed to Gaussian lifting.

This module proves only that the inputs required before lifting agree. It does
not evaluate a lifted Gaussian mask and therefore cannot claim an improvement.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from farm_runtime.obb_release_evidence import (
    validate_serialized_obb_release_evidence,
)


REFINEMENT_SCHEMA = "farm.full-colmap-mask-refinement.v1"
GEOMETRY_SCHEMA = "farm.object-geometry-audit.v3"
SELECTION_SCHEMA = "farm.pre-lift-eligibility-selection.v1"
TRAIN_REFINEMENT_ROLE = "train_fit"
HELDOUT_REFINEMENT_ROLE = "heldout_reference_only"
FOLD_SCHEMA = "farm.full-colmap-rgbd-fold.v1"
IDS_FILENAME = "pre_lift_eligible_object_ids.txt"
REPORT_FILENAME = "selection.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _object_rows(
    payload: Mapping[str, Any], *, label: str
) -> dict[int, dict[str, Any]]:
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise ValueError(f"{label} objects must be a JSON list")
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-object row")
        raw_id = row.get("object_id")
        if isinstance(raw_id, bool):
            raise ValueError(f"{label} contains an invalid object ID")
        try:
            object_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid object ID") from exc
        if object_id < 0 or object_id in output:
            raise ValueError(f"{label} contains a negative or duplicate object ID")
        output[object_id] = row
    return output


def _validate_declared_acceptance(
    payload: Mapping[str, Any],
    rows: Mapping[int, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    for row in rows.values():
        if not isinstance(row.get("accepted"), bool):
            raise ValueError(f"{label} object acceptance must be a JSON boolean")
    expected = sorted(
        object_id for object_id, row in rows.items() if row.get("accepted") is True
    )
    declared = payload.get("accepted_object_ids")
    if not isinstance(declared, list) or any(
        isinstance(value, bool) for value in declared
    ):
        raise ValueError(f"{label} lacks an exact accepted_object_ids list")
    try:
        normalized = sorted(int(value) for value in declared)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} accepted_object_ids is invalid") from exc
    if len(normalized) != len(set(normalized)) or normalized != expected:
        raise ValueError(f"{label} accepted_object_ids contradicts object rows")
    try:
        accepted_count = int(payload.get("accepted_objects", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} accepted_objects count is invalid") from exc
    if accepted_count != len(expected):
        raise ValueError(f"{label} accepted_objects count contradicts object rows")


def _validate_train_contract(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if (
        payload.get("schema") != REFINEMENT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("refinement_role") != TRAIN_REFINEMENT_ROLE
    ):
        raise ValueError("train refinement must be a PASS train_fit result")
    contract = payload.get("input_view_contract")
    policy = payload.get("policy")
    if (
        not isinstance(contract, Mapping)
        or contract.get("fold_contract") != FOLD_SCHEMA
        or contract.get("active_split") != "train"
        or contract.get("state_fit_authorized") is not True
        or contract.get("requested_view_role") != "train"
        or not isinstance(policy, Mapping)
        or policy.get("view_role") != "train"
        or policy.get("state_fit_authorized") is not True
    ):
        raise ValueError("train refinement input_view_contract/policy is incomplete")
    rows = _object_rows(payload, label="train refinement")
    _validate_declared_acceptance(payload, rows, label="train refinement")
    return rows


def _validate_heldout_contract(
    payload: Mapping[str, Any], *, label: str
) -> dict[int, dict[str, Any]]:
    if (
        payload.get("schema") != REFINEMENT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("refinement_role") != HELDOUT_REFINEMENT_ROLE
    ):
        raise ValueError(f"{label} must be a PASS heldout-reference result")
    contract = payload.get("input_view_contract")
    policy = payload.get("policy")
    if (
        not isinstance(contract, Mapping)
        or contract.get("fold_contract") != FOLD_SCHEMA
        or contract.get("active_split") != "heldout"
        or contract.get("state_fit_authorized") is not False
        or contract.get("plan_view_field") != "heldout_views"
        or contract.get("requested_view_role") != "heldout-reference"
        or not isinstance(policy, Mapping)
        or policy.get("view_role") != "heldout-reference"
        or policy.get("state_fit_authorized") is not False
        or policy.get("merge_apply_forbidden") is not True
        or policy.get("heldout_reference_is_evaluation_only") is not True
        or policy.get("reference_masks_are_pseudo_labels_not_ground_truth") is not True
    ):
        raise ValueError(f"{label} input_view_contract/policy is incomplete")
    rows = _object_rows(payload, label=label)
    _validate_declared_acceptance(payload, rows, label=label)
    return rows


def _validate_geometry(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if payload.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("geometry audit must use farm.object-geometry-audit.v3")
    return _object_rows(payload, label="geometry audit")


def _physical_timestamp(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} lacks a physical timestamp")
    text = str(value).strip()
    if not text or not text.isdecimal():
        raise ValueError(f"{label} physical timestamp must be an unsigned integer")
    timestamp = int(text)
    if timestamp <= 0:
        raise ValueError(f"{label} physical timestamp must be positive")
    return str(timestamp)


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} lacks a valid SHA256")
    return digest


def _mask_artifact(
    view: Mapping[str, Any], result_path: Path, *, label: str
) -> dict[str, Any]:
    relative = Path(str(view.get("mask_relative") or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} mask path must be result-relative")
    root = result_path.parent.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix.lower() != ".npz":
        raise ValueError(f"{label} mask escapes its result directory or is not NPZ")
    expected = _sha256(view.get("mask_sha256"), label=f"{label} mask")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} mask SHA256 mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"image_shape", "raw_bits", "raw_shape", "raw_bbox_xyxy"}
            if not required.issubset(archive.files):
                raise ValueError(f"{label} mask lacks canonical packed fields")
            height, width = np.asarray(archive["image_shape"], dtype=np.int64).reshape(
                2
            )
            crop_height, crop_width = np.asarray(
                archive["raw_shape"], dtype=np.int64
            ).reshape(2)
            x0, y0, x1, y1 = np.asarray(
                archive["raw_bbox_xyxy"], dtype=np.int64
            ).reshape(4)
            bits = np.asarray(archive["raw_bits"], dtype=np.uint8).reshape(-1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} mask is not a valid canonical NPZ") from exc
    if (
        min(height, width, crop_height, crop_width) <= 0
        or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)
        or crop_height != y1 - y0
        or crop_width != x1 - x0
    ):
        raise ValueError(f"{label} mask shape/bbox contract is invalid")
    unpacked = np.unpackbits(bits, bitorder="little")
    expected_pixels = int(crop_height * crop_width)
    if unpacked.size < expected_pixels:
        raise ValueError(f"{label} mask bit payload is truncated")
    mask_pixels = int(unpacked[:expected_pixels].sum())
    if mask_pixels <= 0:
        raise ValueError(f"{label} mask is empty")
    try:
        declared_pixels = int(view.get("stored_mask_pixels", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} stored_mask_pixels is invalid") from exc
    if declared_pixels != mask_pixels:
        raise ValueError(f"{label} stored_mask_pixels contradicts packed mask")
    return {
        "path": str(path),
        "result_relative_path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "mask_pixels": mask_pixels,
    }


def _accepted_observations(
    rows: Mapping[int, Mapping[str, Any]],
    result_path: Path,
    *,
    input_index: int,
    input_role: str,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for object_id, row in rows.items():
        views = row.get("views")
        if not isinstance(views, list):
            raise ValueError(f"{input_role} object {object_id} lacks view rows")
        accepted_count = 0
        for view_index, view in enumerate(views):
            label = f"{input_role} object {object_id} view {view_index}"
            if not isinstance(view, Mapping) or not isinstance(
                view.get("accepted"), bool
            ):
                raise ValueError(f"{label} acceptance must be a JSON boolean")
            if view.get("accepted") is not True:
                continue
            accepted_count += 1
            source = str(view.get("source_image") or "").strip()
            if not source:
                raise ValueError(f"{label} lacks source_image")
            timestamp = _physical_timestamp(
                view.get("physical_timestamp_ns"), label=label
            )
            gates = view.get("quality_gates")
            if (
                not isinstance(gates, Mapping)
                or not gates
                or any(value is not True for value in gates.values())
            ):
                raise ValueError(f"{label} lacks passed hard quality gates")
            reasons = view.get("rejection_reasons")
            if not isinstance(reasons, list) or reasons:
                raise ValueError(f"{label} accepted view has rejection reasons")
            observations.append(
                {
                    "object_id": int(object_id),
                    "source_image": source,
                    "physical_timestamp": timestamp,
                    "input_index": int(input_index),
                    "input_role": input_role,
                    "mask": _mask_artifact(view, result_path, label=label),
                }
            )
        try:
            declared_count = int(row.get("accepted_views", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{input_role} object {object_id} accepted_views is invalid"
            ) from exc
        if declared_count != accepted_count:
            raise ValueError(
                f"{input_role} object {object_id} accepted_views contradicts view rows"
            )
        if row.get("accepted") is True and accepted_count == 0:
            raise ValueError(
                f"{input_role} object {object_id} is accepted without evidence"
            )
    return observations


def _validate_source_timestamps(
    observations: Sequence[Mapping[str, Any]], *, label: str
) -> None:
    by_source: dict[str, str] = {}
    for row in observations:
        source = str(row["source_image"])
        timestamp = str(row["physical_timestamp"])
        prior = by_source.setdefault(source, timestamp)
        if prior != timestamp:
            raise ValueError(
                f"{label} source image maps to conflicting timestamps: {source}"
            )


def _compact_observation_union(
    observations: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    for raw in observations:
        object_id = int(raw["object_id"])
        source = str(raw["source_image"])
        timestamp = str(raw["physical_timestamp"])
        key = (source, timestamp)
        per_object = grouped.setdefault(object_id, {})
        evidence = {
            "heldout_input_index": int(raw["input_index"]),
            "mask": copy.deepcopy(raw["mask"]),
        }
        if key not in per_object:
            per_object[key] = {
                "source_image": source,
                "physical_timestamp": timestamp,
                "evidence": [evidence],
            }
            continue
        existing_hashes = {
            str(item["mask"]["sha256"]) for item in per_object[key]["evidence"]
        }
        if str(evidence["mask"]["sha256"]) not in existing_hashes:
            per_object[key]["evidence"].append(evidence)
    return {
        object_id: sorted(
            rows.values(),
            key=lambda row: (int(row["physical_timestamp"]), row["source_image"]),
        )
        for object_id, rows in grouped.items()
    }


def _input_spec(role: str, index: int, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "input_index": int(index),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_pre_lift_selection(
    train_refinement_result: Path,
    geometry_audit: Path,
    heldout_reference_results: Sequence[Path],
    *,
    minimum_independent_physical_timestamps: int = 2,
) -> dict[str, Any]:
    """Validate all inputs and return deterministic pre-lift eligibility JSON."""

    minimum = int(minimum_independent_physical_timestamps)
    if isinstance(minimum_independent_physical_timestamps, bool) or minimum < 1:
        raise ValueError("minimum independent physical timestamps must be positive")
    train_path = Path(train_refinement_result).expanduser().resolve(strict=True)
    geometry_path = Path(geometry_audit).expanduser().resolve(strict=True)
    heldout_paths = [
        Path(path).expanduser().resolve(strict=True)
        for path in heldout_reference_results
    ]
    if not heldout_paths:
        raise ValueError("at least one heldout-reference result is required")
    all_paths = [train_path, geometry_path, *heldout_paths]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("pre-lift selector input paths must be distinct")

    train_payload = _json_object(train_path, label="train refinement")
    geometry_payload = _json_object(geometry_path, label="geometry audit")
    train_rows = _validate_train_contract(train_payload)
    geometry_rows = _validate_geometry(geometry_payload)
    train_observations = _accepted_observations(
        train_rows, train_path, input_index=0, input_role="train_fit"
    )
    heldout_observations: list[dict[str, Any]] = []
    for index, path in enumerate(heldout_paths):
        label = f"heldout reference {index}"
        payload = _json_object(path, label=label)
        rows = _validate_heldout_contract(payload, label=label)
        heldout_observations.extend(
            _accepted_observations(
                rows,
                path,
                input_index=index,
                input_role="heldout_reference_only",
            )
        )

    _validate_source_timestamps(train_observations, label="train")
    _validate_source_timestamps(heldout_observations, label="heldout")
    train_timestamps = {row["physical_timestamp"] for row in train_observations}
    heldout_timestamps = {row["physical_timestamp"] for row in heldout_observations}
    overlap = sorted(train_timestamps.intersection(heldout_timestamps), key=int)
    if overlap:
        raise ValueError(
            "train/heldout physical timestamp overlap is forbidden: "
            + ", ".join(overlap[:8])
        )

    train_by_object: dict[int, list[dict[str, Any]]] = {}
    for row in train_observations:
        train_by_object.setdefault(int(row["object_id"]), []).append(row)
    heldout_union = _compact_observation_union(heldout_observations)
    object_rows: list[dict[str, Any]] = []
    for object_id in sorted(train_rows):
        train_row = train_rows[object_id]
        geometry_row = geometry_rows.get(object_id)
        heldout_rows = heldout_union.get(object_id, [])
        train_object_rows = train_by_object.get(object_id, [])
        train_object_timestamps = sorted(
            {str(row["physical_timestamp"]) for row in train_object_rows}, key=int
        )
        independent = sorted(
            {str(row["physical_timestamp"]) for row in heldout_rows}, key=int
        )
        reasons: list[str] = []
        if train_row.get("accepted") is not True:
            reasons.append("train_refinement_not_accepted")
        if geometry_row is None:
            geometry_status = "missing"
            reasons.append("geometry_audit_object_missing")
            geometry_release_evidence = None
        else:
            geometry_status = str(geometry_row.get("status") or "")
            if geometry_status != "geometry_pass":
                reasons.append("geometry_status_not_geometry_pass")
            geometry_release_evidence = geometry_row.get(
                "train_geometry_evidence"
            )
            reasons.extend(
                validate_serialized_obb_release_evidence(
                    geometry_release_evidence
                )
            )
        if len(independent) < minimum:
            reasons.append("insufficient_independent_heldout_physical_timestamps")
        object_rows.append(
            {
                "object_id": int(object_id),
                "eligible": not reasons,
                "reasons": reasons,
                "train_refinement_accepted": train_row.get("accepted") is True,
                "train_accepted_observations": len(train_object_rows),
                "train_physical_timestamps": train_object_timestamps,
                "geometry_status": geometry_status,
                "geometry_release_evidence": geometry_release_evidence,
                "heldout_observations_after_exact_dedupe": len(heldout_rows),
                "independent_heldout_physical_timestamps": len(independent),
                "heldout_physical_timestamps": independent,
                "heldout_evidence": heldout_rows,
            }
        )

    eligible_ids = [row["object_id"] for row in object_rows if row["eligible"]]
    input_specs = [
        _input_spec("train_refinement_result", 0, train_path),
        _input_spec("geometry_audit", 0, geometry_path),
        *[
            _input_spec("heldout_reference_result", index, path)
            for index, path in enumerate(heldout_paths)
        ],
    ]
    consumed = {spec["path"]: spec for spec in input_specs}
    for observation in [*train_observations, *heldout_observations]:
        mask = observation["mask"]
        consumed.setdefault(
            str(mask["path"]),
            {
                "role": "accepted_mask_artifact",
                "path": str(mask["path"]),
                "bytes": int(mask["bytes"]),
                "sha256": str(mask["sha256"]),
            },
        )
    return {
        "schema": SELECTION_SCHEMA,
        "status": "PASS",
        "selection_kind": "pre_lift_eligibility_only",
        "gaussian_mask_improvement_proven": False,
        "eligible_object_ids": eligible_ids,
        "rejected_object_ids": [
            row["object_id"] for row in object_rows if not row["eligible"]
        ],
        "ignored_heldout_only_object_ids": sorted(
            set(heldout_union).difference(train_rows)
        ),
        "objects": object_rows,
        "integrity": {
            "train_accepted_physical_timestamp_count": len(train_timestamps),
            "heldout_accepted_physical_timestamp_count": len(heldout_timestamps),
            "train_heldout_physical_timestamp_overlap": [],
            "physical_timestamp_sets_disjoint": True,
            "heldout_accepted_observations_before_exact_dedupe": len(
                heldout_observations
            ),
            "heldout_observations_after_exact_dedupe": sum(
                len(rows) for rows in heldout_union.values()
            ),
        },
        "policy": {
            "train_refinement_must_be_accepted": True,
            "geometry_status_required": "geometry_pass",
            "geometry_release_evidence_required": True,
            "legacy_geometry_pass_without_release_evidence_is_not_publishable": True,
            "heldout_state_fit_authorized": False,
            "minimum_independent_heldout_physical_timestamps": minimum,
            "heldout_topups_union_enabled": True,
            "heldout_topups_exact_dedupe_key": [
                "object_id",
                "source_image",
                "physical_timestamp",
            ],
            "train_heldout_physical_timestamp_overlap_forbidden": True,
            "post_lift_gaussian_mask_qc_required": True,
            "pre_lift_eligibility_is_not_gaussian_mask_improvement_proof": True,
        },
        "provenance": {
            "inputs": input_specs,
            "consumed_artifacts": sorted(
                consumed.values(), key=lambda row: (row["role"], row["path"])
            ),
        },
    }


def _write_fsync(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _revalidate_consumed_artifacts(selection: Mapping[str, Any]) -> None:
    provenance = selection.get("provenance")
    artifacts = (
        provenance.get("consumed_artifacts")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("selection lacks consumed artifact provenance")
    for spec in artifacts:
        if not isinstance(spec, Mapping):
            raise ValueError("selection contains invalid consumed artifact provenance")
        path = Path(str(spec.get("path") or "")).resolve(strict=True)
        if path.stat().st_size != int(spec.get("bytes", -1)):
            raise RuntimeError(f"pre-lift input size changed before publish: {path}")
        expected = _sha256(spec.get("sha256"), label=str(path))
        if sha256_file(path) != expected:
            raise RuntimeError(f"pre-lift input hash changed before publish: {path}")


def publish_pre_lift_selection(
    output_dir: Path, selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically publish exact IDs and their auditable selection report."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pre-lift output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        payload = copy.deepcopy(dict(selection))
        ids = payload.get("eligible_object_ids")
        if not isinstance(ids, list) or any(isinstance(value, bool) for value in ids):
            raise ValueError("eligible_object_ids must be sorted unique integers")
        normalized_ids = [int(value) for value in ids]
        if normalized_ids != sorted(set(normalized_ids)):
            raise ValueError("eligible_object_ids must be sorted unique integers")
        ids_path = temporary / IDS_FILENAME
        _write_fsync(ids_path, "".join(f"{value}\n" for value in normalized_ids))
        payload["outputs"] = {
            "exact_object_ids": {
                "relative_path": IDS_FILENAME,
                "format": "one_decimal_object_id_per_line",
                "bytes": ids_path.stat().st_size,
                "sha256": sha256_file(ids_path),
            }
        }
        report_path = temporary / REPORT_FILENAME
        _write_fsync(
            report_path,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        _revalidate_consumed_artifacts(payload)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    published_report = destination / REPORT_FILENAME
    published_ids = destination / IDS_FILENAME
    return {
        "output_dir": str(destination),
        "selection_json": str(published_report),
        "selection_json_sha256": sha256_file(published_report),
        "exact_ids_txt": str(published_ids),
        "exact_ids_txt_sha256": sha256_file(published_ids),
        "eligible_object_ids": normalized_ids,
    }
