#!/usr/bin/env python3
"""Build separate post-lift mask, OBB, semantic, and production queues."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import numpy as np

from farm_runtime.obb_release_evidence import (  # noqa: E402
    validate_serialized_obb_release_evidence,
)
from farm_runtime.post_lift_release_gate import (  # noqa: E402
    PostLiftPolicy,
    evaluate_lift_release_contract,
    evaluate_mask_gate,
    evaluate_obb_gate,
    evaluate_scene_support_gate,
    find_obb_relation_review_pairs,
    gaussian_obb_support_metrics,
    gaussian_scene_support_metrics,
    resolve_obb_geometry,
    route_post_lift_object,
    scene_support_envelope,
    scene_points_to_meters,
    summarize_scene_release,
    validate_meters_per_scene_unit,
    verify_metric_scale_provenance,
)
from tools.farm_shaper_bridge.common import open_graphdeco_ply  # noqa: E402

GENERIC_LABELS = {"", "object", "unknown", "unresolved", "unresolved object"}


def _load_json(path: Path) -> Any:
    return json.loads(
        path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(payload: object, *keys: str) -> object:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _lift_artifact_contract(
    lift: Path,
    result: dict[str, Any],
    build: dict[str, Any],
    heldout: dict[str, Any],
    labeled_manifest: dict[str, Any],
    *,
    table_count: int,
) -> tuple[bool, list[str], dict[Path, str]]:
    """Verify SHA bindings and source-order invariants before publication."""

    expected = {
        "build_manifest": (
            lift / "build_manifest.json",
            ("artifacts", "build_manifest"),
        ),
        "heldout_qc": (lift / "heldout_qc.json", ("artifacts", "heldout_qc")),
        "dense_object_ids": (
            lift / "per_gaussian_object_id.npy",
            ("artifacts", "final", "object_id"),
        ),
        "labeled_full_ply": (
            lift / "instance_labeled_full.ply",
            ("artifacts", "final", "instance_labeled_full"),
        ),
        "labeled_ply_manifest": (
            lift / "labeled_ply_manifest.json",
            ("artifacts", "final", "labeled_ply_manifest"),
        ),
    }
    reasons: list[str] = []
    hashes: dict[Path, str] = {}
    for name, (path, descriptor_keys) in expected.items():
        descriptor = _nested(result, *descriptor_keys)
        digest = _sha256(path)
        hashes[path] = digest
        if not isinstance(descriptor, dict):
            reasons.append(f"artifact_descriptor_missing:{name}")
            continue
        if descriptor.get("path") != path.name:
            reasons.append(f"artifact_path_mismatch:{name}")
        if int(descriptor.get("bytes") or -1) != path.stat().st_size:
            reasons.append(f"artifact_size_mismatch:{name}")
        if descriptor.get("sha256") != digest:
            reasons.append(f"artifact_sha256_mismatch:{name}")

    result_inputs = result.get("inputs") or {}
    source_sha = str(result_inputs.get("source_ply_sha256") or "")
    config_sha = str(result_inputs.get("config_sha256") or "")
    scene_id = str(result.get("scene_id") or "")
    bindings = (
        (
            "heldout_build_manifest_sha256_mismatch",
            heldout.get("frozen_build_manifest_sha256")
            == hashes[lift / "build_manifest.json"],
        ),
        (
            "heldout_provisional_digest_mismatch",
            heldout.get("frozen_provisional_digest")
            == build.get("frozen_provisional_digest"),
        ),
        (
            "source_ply_sha256_binding_mismatch",
            bool(source_sha)
            and build.get("source_ply_sha256") == source_sha
            and labeled_manifest.get("source_full_sha256") == source_sha,
        ),
        (
            "config_sha256_binding_mismatch",
            bool(config_sha)
            and build.get("config_sha256") == config_sha
            and heldout.get("config_sha256") == config_sha,
        ),
        (
            "scene_id_binding_mismatch",
            bool(scene_id)
            and build.get("scene_id") == scene_id
            and heldout.get("scene_id") == scene_id,
        ),
        (
            "labeled_ply_output_sha256_mismatch",
            labeled_manifest.get("output_sha256")
            == hashes[lift / "instance_labeled_full.ply"],
        ),
        (
            "labeled_ply_vertex_count_mismatch",
            int(labeled_manifest.get("vertex_count") or -1) == int(table_count),
        ),
        (
            "labeled_ply_original_fields_not_preserved",
            labeled_manifest.get("original_fields_bitwise_preserved") is True,
        ),
        (
            "labeled_ply_source_order_not_preserved",
            labeled_manifest.get("source_order_preserved") is True,
        ),
        (
            "labeled_ply_rows_deleted",
            _integer_equals(labeled_manifest.get("rows_deleted"), 0),
        ),
    )
    reasons.extend(reason for reason, passed in bindings if not passed)
    return not reasons, reasons, hashes


def _integer_equals(value: object, expected: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError, OverflowError):
        return False


def _frozen_heldout_contract(
    qc_path: Path,
    candidate_path: Path,
    qc: dict[str, Any],
    candidate: dict[str, Any],
    *,
    ids_path: Path,
    ply_path: Path,
    result_path: Path,
    gaussian_ids: np.ndarray,
    table_count: int,
    meters_per_scene_unit: float,
) -> tuple[bool, list[str], dict[int, int], dict[Path, str]]:
    """Validate the externally evaluated, frozen full-COLMAP candidate."""

    hashes = {
        qc_path: _sha256(qc_path),
        candidate_path: _sha256(candidate_path),
        ids_path: _sha256(ids_path),
        ply_path: _sha256(ply_path),
        result_path: _sha256(result_path),
    }
    reasons: list[str] = []
    candidate_gaussian = candidate.get("gaussian_contract")
    candidate_artifacts = candidate.get("gaussian_artifacts")
    candidate_fit = candidate.get("fit_contract")
    qc_contract = qc.get("contract")
    qc_provenance = qc.get("provenance")
    qc_candidate = (
        qc_provenance.get("candidate_manifest")
        if isinstance(qc_provenance, dict)
        else None
    )

    required_qc_contract = {
        "read_only": True,
        "heldout_used_for_state_fit": False,
        "heldout_updates_masks_obb_labels": False,
        "acceptance_aggregated_by_independent_physical_timestamp": True,
        "duplicate_views_at_one_timestamp_cannot_inflate_independence": True,
        "all_consumed_inputs_sha256_verified_and_rechecked": True,
    }
    required_candidate_gaussian = {
        "source_candidate_gaussian_order_aligned": True,
        "candidate_frozen": True,
        "labels_frozen": True,
        "heldout_used_for_fit": False,
    }
    required_candidate_fit = {
        "fit_splits": ["train"],
        "heldout_consumed": False,
        "heldout_updates_candidate": False,
        "frozen_before_heldout_evaluation": True,
    }
    checks: list[tuple[str, bool]] = [
        (
            "authoritative_frozen_heldout_schema_required",
            qc.get("schema") == "farm.frozen-heldout-qc.v1",
        ),
        (
            "frozen_heldout_status_invalid",
            qc.get("status") in {"PASS", "BLOCKED"},
        ),
        (
            "frozen_heldout_contract_incomplete",
            isinstance(qc_contract, dict)
            and all(
                qc_contract.get(key) == expected
                for key, expected in required_qc_contract.items()
            ),
        ),
        (
            "frozen_candidate_not_frozen_v1",
            candidate.get("schema") == "farm.frozen-heldout-candidate.v1"
            and candidate.get("status") == "frozen",
        ),
        (
            "frozen_candidate_scale_mismatch",
            np.isclose(
                float(candidate.get("meters_per_scene_unit") or 0.0),
                meters_per_scene_unit,
                rtol=1.0e-9,
                atol=1.0e-12,
            )
            and np.isclose(
                float(qc.get("meters_per_scene_unit") or 0.0),
                meters_per_scene_unit,
                rtol=1.0e-9,
                atol=1.0e-12,
            ),
        ),
        (
            "frozen_candidate_gaussian_contract_incomplete",
            isinstance(candidate_gaussian, dict)
            and all(
                candidate_gaussian.get(key) == expected
                for key, expected in required_candidate_gaussian.items()
            )
            and _integer_equals(
                candidate_gaussian.get("source_gaussian_count"), table_count
            )
            and _integer_equals(
                candidate_gaussian.get("candidate_gaussian_count"), table_count
            ),
        ),
        (
            "frozen_candidate_fit_contract_incomplete",
            isinstance(candidate_fit, dict)
            and all(
                candidate_fit.get(key) == expected
                for key, expected in required_candidate_fit.items()
            ),
        ),
        (
            "frozen_qc_candidate_sha256_mismatch",
            isinstance(qc_candidate, dict)
            and qc_candidate.get("sha256") == hashes[candidate_path]
            and qc_candidate.get("schema") == "farm.frozen-heldout-candidate.v1",
        ),
    ]

    for name, path in (("candidate_labels", ids_path), ("candidate_ply", ply_path)):
        descriptor = (
            candidate_artifacts.get(name)
            if isinstance(candidate_artifacts, dict)
            else None
        )
        valid = isinstance(descriptor, dict)
        if valid:
            descriptor_path = Path(str(descriptor.get("path") or "")).expanduser()
            try:
                resolved_descriptor = descriptor_path.resolve(strict=True)
                descriptor_digest = _sha256(resolved_descriptor)
            except (OSError, ValueError):
                valid = False
            else:
                hashes[resolved_descriptor] = descriptor_digest
                valid = (
                    descriptor_digest == hashes[path]
                    and descriptor.get("sha256") == descriptor_digest
                    and _integer_equals(
                        descriptor.get("bytes"), resolved_descriptor.stat().st_size
                    )
                )
        checks.append((f"frozen_candidate_{name}_artifact_mismatch", valid))

    upstream = candidate.get("upstream_lift_result")
    checks.append(
        (
            "frozen_candidate_upstream_lift_mismatch",
            isinstance(upstream, dict)
            and upstream.get("sha256") == hashes[result_path],
        )
    )

    candidate_rows = candidate.get("objects")
    qc_rows = qc.get("objects")
    candidate_ids = [
        int(row["object_id"])
        for row in candidate_rows or []
        if isinstance(row, dict) and "object_id" in row
    ]
    qc_by_id = {
        int(row["object_id"]): row
        for row in qc_rows or []
        if isinstance(row, dict) and "object_id" in row
    }
    checks.extend(
        [
            (
                "frozen_candidate_object_inventory_invalid",
                isinstance(candidate_rows, list)
                and bool(candidate_ids)
                and len(candidate_ids) == len(candidate_rows)
                and len(candidate_ids) == len(set(candidate_ids)),
            ),
            (
                "frozen_qc_object_inventory_mismatch",
                isinstance(qc_rows, list)
                and len(qc_by_id) == len(qc_rows)
                and set(qc_by_id) == set(candidate_ids),
            ),
        ]
    )
    dense_ids, dense_counts = np.unique(
        np.asarray(gaussian_ids)[np.asarray(gaussian_ids) >= 0], return_counts=True
    )
    candidate_counts = {
        int(object_id): int(count)
        for object_id, count in zip(dense_ids, dense_counts, strict=True)
        if int(object_id) in set(candidate_ids)
    }
    checks.append(
        (
            "frozen_candidate_object_absent_from_dense_labels",
            set(candidate_counts) == set(candidate_ids),
        )
    )

    expected_routes = {
        name: sorted(
            object_id for object_id, row in qc_by_id.items() if row.get("route") == name
        )
        for name in (
            "heldout_verified",
            "collect_independent_heldout_evidence",
            "reject_candidate_train_only_refinement",
        )
    }
    expected_status_by_route = {
        "heldout_verified": "verified",
        "collect_independent_heldout_evidence": "insufficient_evidence",
        "reject_candidate_train_only_refinement": "rejected",
    }
    qc_routes = qc.get("routes")
    checks.append(
        (
            "frozen_qc_routes_mismatch",
            isinstance(qc_routes, dict)
            and all(
                sorted(int(value) for value in qc_routes.get(name, [])) == values
                for name, values in expected_routes.items()
            )
            and set().union(*(set(values) for values in expected_routes.values()))
            == set(candidate_ids)
            and all(
                row.get("status") == expected_status_by_route.get(row.get("route"))
                for row in qc_by_id.values()
            ),
        )
    )
    verified = expected_routes["heldout_verified"]
    expected_status = "PASS" if len(verified) == len(candidate_ids) else "BLOCKED"
    qc_counts = qc.get("counts")
    checks.extend(
        [
            ("frozen_qc_status_mismatch", qc.get("status") == expected_status),
            (
                "frozen_qc_counts_mismatch",
                isinstance(qc_counts, dict)
                and _integer_equals(
                    qc_counts.get("candidate_objects"), len(candidate_ids)
                )
                and _integer_equals(qc_counts.get("verified"), len(verified))
                and _integer_equals(
                    qc_counts.get("collect_more_evidence"),
                    len(expected_routes["collect_independent_heldout_evidence"]),
                )
                and _integer_equals(
                    qc_counts.get("rejected"),
                    len(expected_routes["reject_candidate_train_only_refinement"]),
                ),
            ),
        ]
    )
    reasons.extend(reason for reason, passed in checks if not passed)
    return not reasons, reasons, candidate_counts, hashes


def _metric_scale_contract(
    meters_per_scene_unit: object, provenance_path: Path | None
) -> dict[str, Any]:
    scale = validate_meters_per_scene_unit(meters_per_scene_unit)
    resolved_provenance: Path | None = None
    declarations: list[dict[str, Any]] = []
    if provenance_path is not None:
        resolved_provenance = provenance_path.expanduser().resolve(strict=True)
        payload = _load_json(resolved_provenance)
        if not isinstance(payload, dict):
            raise TypeError("metric scale provenance must be a JSON object")
        declarations = verify_metric_scale_provenance(scale, payload)
    return {
        "status": "PASS",
        "meters_per_scene_unit": scale,
        "scale_source": "explicit_cli",
        "cli_argument_required": True,
        "source_ply_xyz_units": "scene_units",
        "published_obb_units": "metres",
        "conversion": "points_world_m = source_ply_xyz * meters_per_scene_unit",
        "external_provenance": (
            str(resolved_provenance) if resolved_provenance is not None else None
        ),
        "external_provenance_sha256": (
            _sha256(resolved_provenance) if resolved_provenance is not None else None
        ),
        "verified_declarations": declarations,
        "external_provenance_verified": resolved_provenance is not None,
    }


def _by_object(rows: object, key: str) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list):
        raise TypeError("expected an object list")
    result: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict) or key not in raw:
            raise TypeError(f"invalid object row without {key!r}")
        object_id = int(raw[key])
        if object_id in result:
            raise ValueError(f"duplicate object ID {object_id}")
        result[object_id] = dict(raw)
    return result


def _semantic_rows(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    rows = payload.get("objects") if isinstance(payload, dict) else None
    return _by_object(rows, "object_id")


def _train_obb_release_binding(
    catalog_item: dict[str, Any],
    geometry_row: dict[str, Any] | None,
    *,
    audit_schema: object,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Bind a published OBB to strict train RGB-D release evidence.

    Gaussian containment is deliberately insufficient on its own: the exact
    metric OBB being evaluated post-lift must also be the OBB authorized by a
    non-weakened train-evidence record.
    """

    reasons: list[str] = []
    if audit_schema != "farm.object-geometry-audit.v3":
        reasons.append("train_geometry_audit_schema_invalid")
    if geometry_row is None:
        reasons.append("train_geometry_audit_object_missing")
        return False, reasons, {"geometry_matches_catalog": False}
    if geometry_row.get("geometry_release_ready") is not True:
        reasons.append("train_geometry_release_not_ready")
    reasons.extend(
        validate_serialized_obb_release_evidence(
            geometry_row.get("train_geometry_evidence")
        )
    )
    geometry_matches = False
    try:
        catalog_center, catalog_dimensions, catalog_rotation, catalog_sources = (
            resolve_obb_geometry(catalog_item)
        )
        audit_center, audit_dimensions, audit_rotation, audit_sources = (
            resolve_obb_geometry(
                {
                    "center_m": geometry_row.get("box_center_m"),
                    "dimensions_m": geometry_row.get("box_dimensions_m"),
                    "wxyz": geometry_row.get("box_wxyz"),
                }
            )
        )
        geometry_matches = bool(
            np.allclose(catalog_center, audit_center, rtol=0.0, atol=1e-5)
            and np.allclose(
                catalog_dimensions, audit_dimensions, rtol=0.0, atol=1e-5
            )
            and np.allclose(catalog_rotation, audit_rotation, rtol=0.0, atol=1e-6)
        )
    except (TypeError, ValueError) as exc:
        catalog_sources = None
        audit_sources = None
        reasons.append("train_geometry_or_catalog_obb_invalid")
        error = str(exc)
    else:
        error = None
        if not geometry_matches:
            reasons.append("catalog_obb_differs_from_train_released_obb")
    return not reasons, reasons, {
        "geometry_matches_catalog": geometry_matches,
        "catalog_resolved_sources": catalog_sources,
        "audit_resolved_sources": audit_sources,
        "input_error": error,
    }


def _semantic_gate(
    catalog: dict[str, Any], audit: dict[str, Any] | None
) -> tuple[bool, list[str], str]:
    proposed = str(catalog.get("category") or "").strip()
    if audit is None:
        return False, ["independent_semantic_audit_missing"], "unresolved object"
    audited = str(audit.get("category") or "").strip()
    reasons: list[str] = []
    if str(audit.get("status") or "") != "whole_ready":
        reasons.append("whole_object_semantic_consensus_missing")
    if not audited or audited.lower() in GENERIC_LABELS:
        reasons.append("semantic_label_generic_or_missing")
    if proposed and audited and proposed.lower() != audited.lower():
        reasons.append("presentation_and_audited_label_disagree")
    return not reasons, reasons, audited if not reasons else "unresolved object"


def _id_file(path: Path, ids: list[int]) -> None:
    path.write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-catalog", type=Path, required=True)
    parser.add_argument("--gaussian-lift-dir", type=Path, required=True)
    parser.add_argument("--frozen-heldout-qc", type=Path, required=True)
    parser.add_argument("--frozen-heldout-candidate", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path)
    parser.add_argument(
        "--train-geometry-audit",
        type=Path,
        help=(
            "Object-geometry audit carrying strict train release evidence for "
            "the exact OBBs in the presentation catalog. Omitted evidence makes "
            "the OBB verified allowlist empty (fail closed)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meters-per-scene-unit", type=float, required=True)
    parser.add_argument("--metric-scale-provenance", type=Path)
    parser.add_argument("--minimum-build-timestamps", type=int, default=3)
    parser.add_argument("--minimum-heldout-timestamps", type=int, default=2)
    parser.add_argument("--minimum-good-heldout-timestamps", type=int, default=2)
    parser.add_argument("--minimum-gaussians", type=int, default=500)
    parser.add_argument(
        "--experimental-nonrelease-routing",
        action="store_true",
        help=(
            "Compute diagnostic per-object queues when lift quality/integrity and "
            "frozen heldout pass but publication provenance does not. This never "
            "writes production-ready objects or makes the scene release eligible."
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()

    catalog_path = args.presentation_catalog.expanduser().resolve(strict=True)
    lift = args.gaussian_lift_dir.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    try:
        scale_contract = _metric_scale_contract(
            args.meters_per_scene_unit, args.metric_scale_provenance
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    meters_per_scene_unit = float(scale_contract["meters_per_scene_unit"])
    catalog_payload = _load_json(catalog_path)
    catalog = _by_object(catalog_payload, "id")
    internal_heldout_path = lift / "heldout_qc.json"
    build_path = lift / "build_manifest.json"
    result_path = lift / "result.json"
    ids_path = lift / "per_gaussian_object_id.npy"
    ply_path = lift / "instance_labeled_full.ply"
    labeled_manifest_path = lift / "labeled_ply_manifest.json"
    internal_heldout_payload = _load_json(internal_heldout_path)
    build_payload = _load_json(build_path)
    result_payload = _load_json(result_path)
    labeled_manifest = _load_json(labeled_manifest_path)
    frozen_qc_path = args.frozen_heldout_qc.expanduser().resolve(strict=True)
    frozen_candidate_path = args.frozen_heldout_candidate.expanduser().resolve(
        strict=True
    )
    frozen_qc_payload = _load_json(frozen_qc_path)
    frozen_candidate_payload = _load_json(frozen_candidate_path)
    heldout = _by_object(frozen_qc_payload.get("objects"), "object_id")
    build = _by_object(build_payload.get("objects"), "object_id")
    semantic = _semantic_rows(args.semantic_audit)
    train_geometry_path = (
        args.train_geometry_audit.expanduser().resolve(strict=True)
        if args.train_geometry_audit
        else None
    )
    train_geometry_payload = (
        _load_json(train_geometry_path) if train_geometry_path is not None else {}
    )
    train_geometry = (
        _by_object(train_geometry_payload.get("objects"), "object_id")
        if isinstance(train_geometry_payload, dict)
        and isinstance(train_geometry_payload.get("objects"), list)
        else {}
    )
    train_geometry_schema = (
        train_geometry_payload.get("schema")
        if isinstance(train_geometry_payload, dict)
        else None
    )

    policy = PostLiftPolicy(
        minimum_build_timestamps=int(args.minimum_build_timestamps),
        minimum_heldout_timestamps=int(args.minimum_heldout_timestamps),
        minimum_good_heldout_timestamps=int(args.minimum_good_heldout_timestamps),
        minimum_gaussians=int(args.minimum_gaussians),
    )
    if (
        min(
            policy.minimum_build_timestamps,
            policy.minimum_heldout_timestamps,
            policy.minimum_good_heldout_timestamps,
            policy.minimum_gaussians,
        )
        < 1
    ):
        parser.error("count thresholds must be positive")

    gaussian_ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    table = open_graphdeco_ply(ply_path)
    if gaussian_ids.shape != (table.count,):
        raise ValueError("Gaussian ID array length differs from labeled PLY")
    if not np.issubdtype(gaussian_ids.dtype, np.integer):
        raise TypeError("Gaussian ID array must have integer dtype")

    artifact_pass, artifact_reasons, input_hashes = _lift_artifact_contract(
        lift,
        result_payload,
        build_payload,
        internal_heldout_payload,
        labeled_manifest,
        table_count=table.count,
    )
    if "farm_instance_id" not in (table.dtype.names or ()):
        artifact_pass = False
        artifact_reasons.append("labeled_ply_instance_id_property_missing")
    elif not np.array_equal(
        np.asarray(table.data["farm_instance_id"]), np.asarray(gaussian_ids)
    ):
        artifact_pass = False
        artifact_reasons.append("labeled_ply_dense_instance_ids_mismatch")
    input_hashes[result_path] = _sha256(result_path)
    input_hashes[catalog_path] = _sha256(catalog_path)
    if args.semantic_audit:
        semantic_path = args.semantic_audit.expanduser().resolve(strict=True)
        input_hashes[semantic_path] = _sha256(semantic_path)
    if train_geometry_path is not None:
        input_hashes[train_geometry_path] = _sha256(train_geometry_path)
    lift_release_pass, lift_release_reasons, global_metrics = evaluate_lift_release_contract(
        result_payload,
        build_payload,
        internal_heldout_payload,
        gaussian_ids,
        source_gaussian_count=table.count,
        artifacts_verified=artifact_pass,
    )
    lift_release_reasons = [*artifact_reasons, *lift_release_reasons]
    lift_release_pass = lift_release_pass and artifact_pass
    lift_quality_integrity_pass = bool(
        global_metrics.get("quality_integrity_passed")
    ) and artifact_pass
    lift_pass = lift_release_pass
    lift_reasons = list(lift_release_reasons)
    (
        frozen_pass,
        frozen_reasons,
        candidate_counts,
        frozen_hashes,
    ) = _frozen_heldout_contract(
        frozen_qc_path,
        frozen_candidate_path,
        frozen_qc_payload,
        frozen_candidate_payload,
        ids_path=ids_path,
        ply_path=ply_path,
        result_path=result_path,
        gaussian_ids=gaussian_ids,
        table_count=table.count,
        meters_per_scene_unit=meters_per_scene_unit,
    )
    input_hashes.update(frozen_hashes)
    global_reasons = [*lift_release_reasons, *frozen_reasons]
    global_pass = lift_release_pass and frozen_pass
    diagnostic_upstream_pass = global_pass or bool(
        args.experimental_nonrelease_routing
        and lift_quality_integrity_pass
        and frozen_pass
    )

    scene_points = np.column_stack(
        tuple(table.data[name] for name in ("x", "y", "z"))
    ).astype(np.float64, copy=False)
    scene_envelope = scene_support_envelope(
        scene_points_to_meters(scene_points, meters_per_scene_unit),
        quantile=policy.scene_support_quantile,
    )
    del scene_points

    rows: list[dict[str, Any]] = []
    geometry_catalog_by_id: dict[int, dict[str, Any]] = {}
    production_catalog: list[dict[str, Any]] = []
    for object_id, item in sorted(catalog.items()):
        selected = np.flatnonzero(gaussian_ids == object_id)
        points_scene = (
            np.column_stack(
                tuple(table.data[name][selected] for name in ("x", "y", "z"))
            ).astype(np.float64, copy=False)
            if selected.size
            else np.zeros((0, 3), dtype=np.float64)
        )
        points_m = scene_points_to_meters(points_scene, meters_per_scene_unit)
        mask_pass, mask_reasons, mask_metrics = evaluate_mask_gate(
            heldout.get(object_id),
            build.get(object_id),
            heldout_schema=str(frozen_qc_payload.get("schema") or ""),
            candidate_gaussian_count=candidate_counts.get(object_id, 0),
            final_gaussian_count=int(selected.size),
            policy=policy,
        )
        obb_error: str | None = None
        try:
            center_m, dimensions_m, rotation_matrix, obb_sources = (
                resolve_obb_geometry(item)
            )
            support = gaussian_obb_support_metrics(
                points_m,
                center_m=center_m,
                dimensions_m=dimensions_m,
                rotation_matrix=rotation_matrix,
                quantile=policy.support_quantile,
            )
            obb_pass, obb_reasons = evaluate_obb_gate(support, policy=policy)
            spatial_support = gaussian_scene_support_metrics(
                points_m,
                center_m=center_m,
                robust_support_dimensions_m=support.get("robust_support_dimensions_m"),
                scene_envelope=scene_envelope,
                margin_fraction=policy.scene_support_margin_fraction,
            )
            spatial_pass, spatial_reasons = evaluate_scene_support_gate(
                spatial_support, policy=policy
            )
        except (TypeError, ValueError) as exc:
            obb_error = str(exc)
            support = {
                "finite_gaussians": int(points_m.shape[0]),
                "gaussian_inside_obb_fraction": None,
                "robust_support_dimensions_m": None,
                "obb_to_support_volume_ratio": None,
                "normalized_support_center_offset": None,
            }
            obb_pass = False
            obb_reasons = ["invalid_obb_or_gaussian_support"]
            spatial_support = {
                "finite_object_gaussians": int(points_m.shape[0]),
                "gaussian_inside_scene_support_fraction": None,
                "obb_center_inside_scene_support": None,
                "obb_center_distance_scene_diagonal": None,
                "object_support_diagonal_m": None,
                "object_support_to_scene_diagonal": None,
            }
            spatial_pass = False
            spatial_reasons = ["invalid_obb_or_gaussian_support"]
            obb_sources = None
        semantic_pass, semantic_reasons, published_category = _semantic_gate(
            item, semantic.get(object_id)
        )
        train_obb_pass, train_obb_reasons, train_obb_details = (
            _train_obb_release_binding(
                item,
                train_geometry.get(object_id),
                audit_schema=train_geometry_schema,
            )
        )
        row = {
            "object_id": object_id,
            "candidate_category": str(item.get("category") or ""),
            "published_category": published_category,
            "upstream_lift_gate": {
                "passed": global_pass,
                "reasons": global_reasons,
            },
            "diagnostic_quality_routing_gate": {
                "passed": diagnostic_upstream_pass,
                "experimental_nonrelease": bool(
                    diagnostic_upstream_pass and not global_pass
                ),
            },
            "mask_gate": {"passed": mask_pass, "reasons": mask_reasons, **mask_metrics},
            "obb_gate": {
                "passed": obb_pass,
                "reasons": obb_reasons,
                "input_error": obb_error,
                "resolved_sources": obb_sources,
                **support,
            },
            "train_obb_release_gate": {
                "passed": train_obb_pass,
                "reasons": train_obb_reasons,
                "source_audit": (
                    str(train_geometry_path) if train_geometry_path is not None else None
                ),
                "source_audit_sha256": (
                    input_hashes.get(train_geometry_path)
                    if train_geometry_path is not None
                    else None
                ),
                **train_obb_details,
            },
            "scene_support_gate": {
                "passed": spatial_pass,
                "reasons": spatial_reasons,
                **spatial_support,
            },
            "semantic_gate": {
                "passed": semantic_pass,
                "reasons": semantic_reasons,
                "audit_status": str(
                    (semantic.get(object_id) or {}).get("status") or "missing"
                ),
            },
        }
        rows.append(row)

    relation_candidate_ids = [
        int(row["object_id"])
        for row in rows
        if bool(row["mask_gate"]["passed"])
        and bool(row["obb_gate"]["passed"])
        and bool(row["train_obb_release_gate"]["passed"])
        and bool(row["scene_support_gate"]["passed"])
    ]
    relation_pairs = find_obb_relation_review_pairs(
        catalog, relation_candidate_ids, policy=policy
    )
    relation_findings_by_id: dict[int, list[dict[str, Any]]] = {
        object_id: [] for object_id in relation_candidate_ids
    }
    for pair in relation_pairs:
        first_id = int(pair["first_object_id"])
        second_id = int(pair["second_object_id"])
        shared = {
            "reasons": list(pair["reasons"]),
            "metrics": dict(pair["metrics"]),
        }
        relation_findings_by_id[first_id].append(
            {"other_object_id": second_id, **shared}
        )
        relation_findings_by_id[second_id].append(
            {"other_object_id": first_id, **shared}
        )

    for row in rows:
        object_id = int(row["object_id"])
        mask_pass = bool(row["mask_gate"]["passed"])
        obb_pass = bool(row["obb_gate"]["passed"])
        train_obb_pass = bool(row["train_obb_release_gate"]["passed"])
        verified_obb_pass = obb_pass and train_obb_pass
        spatial_pass = bool(row["scene_support_gate"]["passed"])
        semantic_pass = bool(row["semantic_gate"]["passed"])
        relation_findings = relation_findings_by_id.get(object_id, [])
        relation_applicable = mask_pass and verified_obb_pass and spatial_pass
        relation_pass = relation_applicable and not relation_findings
        row["relation_gate"] = {
            "applicable": relation_applicable,
            "passed": relation_pass if relation_applicable else None,
            "reasons": sorted(
                {
                    reason
                    for finding in relation_findings
                    for reason in finding["reasons"]
                }
            ),
            "findings": relation_findings,
            "automatic_identity_action": "none",
        }
        route = route_post_lift_object(
            diagnostic_upstream_passed=diagnostic_upstream_pass,
            publication_upstream_passed=global_pass,
            mask_passed=mask_pass,
            obb_passed=verified_obb_pass,
            scene_support_passed=spatial_pass,
            relation_passed=relation_pass,
            semantic_passed=semantic_pass,
        )
        row["route"] = route

        geometry_ready = (
            diagnostic_upstream_pass
            and mask_pass
            and verified_obb_pass
            and spatial_pass
            and relation_pass
        )
        if geometry_ready:
            geometry_item = dict(catalog[object_id])
            geometry_item["candidate_category"] = str(
                catalog[object_id].get("category") or ""
            )
            geometry_item["category"] = str(row["published_category"])
            geometry_item["post_lift_geometry_status"] = "verified"
            geometry_item["post_lift_relation_status"] = "verified"
            geometry_item["post_lift_semantic_status"] = (
                "verified" if semantic_pass else "unresolved"
            )
            geometry_item["post_lift_publication_eligible"] = bool(global_pass)
            geometry_item["post_lift_evidence_mode"] = (
                "release"
                if global_pass
                else "experimental_nonrelease_quality_diagnostic"
            )
            geometry_catalog_by_id[object_id] = geometry_item
        if route == "production_ready":
            production_item = dict(catalog[object_id])
            production_item["category"] = str(row["published_category"])
            production_item["post_lift_geometry_status"] = "verified"
            production_item["post_lift_relation_status"] = "verified"
            production_item["post_lift_semantic_status"] = "verified"
            production_catalog.append(production_item)

    geometry_catalog = [
        geometry_catalog_by_id[object_id]
        for object_id in sorted(geometry_catalog_by_id)
    ]
    counts = Counter(row["route"] for row in rows)
    relation_review_ids = sorted(
        object_id for object_id, findings in relation_findings_by_id.items() if findings
    )
    scene_release = summarize_scene_release(rows, upstream_passed=global_pass)
    production_ready_ids = list(scene_release["published_object_ids"])
    blocked_ids = list(scene_release["blocked_object_ids"])
    report = {
        "schema": "farm.post-lift-release-gate.v4",
        "status": scene_release["status"],
        "release_eligible": scene_release["release_eligible"],
        "complete_scene_release_eligible": scene_release[
            "complete_scene_release_eligible"
        ],
        "published_object_ids": production_ready_ids,
        "blocked_object_ids": blocked_ids,
        "policy": policy.to_dict(),
        "metric_scale_contract": scale_contract,
        "upstream_lift_gate": {
            "passed": lift_pass,
            "reasons": lift_reasons,
            **global_metrics,
        },
        "global_release_contract": {
            "passed": global_pass,
            "reasons": global_reasons,
        },
        "experimental_nonrelease_routing": {
            "enabled": bool(args.experimental_nonrelease_routing),
            "passed": bool(diagnostic_upstream_pass),
            "lift_quality_integrity_passed": bool(lift_quality_integrity_pass),
            "frozen_heldout_passed": bool(frozen_pass),
            "publication_authorized": bool(global_pass),
            "cannot_authorize_production": True,
        },
        "authoritative_frozen_heldout_gate": {
            "passed": frozen_pass,
            "reasons": frozen_reasons,
            "candidate_object_ids": sorted(candidate_counts),
            "candidate_gaussians_by_object": {
                str(object_id): count
                for object_id, count in sorted(candidate_counts.items())
            },
        },
        "scene_support_envelope": scene_envelope,
        "inputs": {
            "presentation_catalog": str(catalog_path),
            "presentation_catalog_sha256": _sha256(catalog_path),
            "gaussian_lift_dir": str(lift),
            "gaussian_lift_result_status": str(result_payload.get("status") or ""),
            "gaussian_lift_release_eligible": bool(
                result_payload.get("release_eligible")
            ),
            "internal_lift_heldout_qc": str(internal_heldout_path),
            "authoritative_frozen_heldout_qc": str(frozen_qc_path),
            "authoritative_frozen_heldout_qc_sha256": input_hashes[frozen_qc_path],
            "frozen_heldout_candidate": str(frozen_candidate_path),
            "frozen_heldout_candidate_sha256": input_hashes[frozen_candidate_path],
            "source_ply_sha256": str(
                ((result_payload.get("inputs") or {}).get("source_ply_sha256") or "")
            ),
            "semantic_audit": (
                str(args.semantic_audit.expanduser().resolve())
                if args.semantic_audit
                else None
            ),
            "train_geometry_audit": (
                str(train_geometry_path) if train_geometry_path is not None else None
            ),
            "train_geometry_audit_sha256": (
                input_hashes.get(train_geometry_path)
                if train_geometry_path is not None
                else None
            ),
        },
        "counts": {
            "input_visible_objects": len(rows),
            "blocked_objects": len(blocked_ids),
            "mask_verified_objects": sum(
                bool(row["mask_gate"]["passed"]) for row in rows
            ),
            "obb_verified_objects": sum(
                bool(
                    row["mask_gate"]["passed"]
                    and row["obb_gate"]["passed"]
                    and row["train_obb_release_gate"]["passed"]
                )
                for row in rows
            ),
            "scene_support_verified_objects": sum(
                bool(
                    row["mask_gate"]["passed"]
                    and row["obb_gate"]["passed"]
                    and row["train_obb_release_gate"]["passed"]
                    and row["scene_support_gate"]["passed"]
                )
                for row in rows
            ),
            "geometry_catalog_objects": len(geometry_catalog),
            "relation_verified_objects": sum(
                bool(row["relation_gate"]["passed"])
                for row in rows
                if row["relation_gate"]["applicable"]
            ),
            "relation_review_objects": len(relation_review_ids),
            "relation_review_pairs": len(relation_pairs),
            "semantic_verified_objects": sum(
                bool(row["semantic_gate"]["passed"]) for row in rows
            ),
            "production_ready_objects": len(production_catalog),
            "routes": dict(sorted(counts.items())),
        },
        "relation_review": {
            "object_ids": relation_review_ids,
            "pairs": relation_pairs,
            "automatic_merges": 0,
            "automatic_suppressions": 0,
        },
        "contracts": {
            "mask_obb_semantics_separate": True,
            "positive_mask_and_obb_allowlists_are_separate": True,
            "obb_allowlist_requires_sha_bound_train_release_evidence": True,
            "heldout_rejected_objects_not_visible": True,
            "unverified_labels_replaced_with_unresolved": True,
            "obb_fit_never_validated_by_containment_alone": True,
            "global_lift_and_heldout_must_be_release_valid": True,
            "production_uses_explicit_external_frozen_full_colmap_qc": True,
            "implicit_internal_lift_qc_cannot_authorize_publication": True,
            "dense_gaussian_ownership_is_exclusive_and_heldout_bound": True,
            "spatial_outliers_checked_against_robust_full_ply_support": True,
            "geometry_catalog_contains_only_fully_verified_geometry": True,
            "safe_production_subset_does_not_require_all_objects_to_pass": True,
            "experimental_quality_routing_cannot_authorize_publication": True,
            "blocked_rows_remain_explicit_in_scene_report": True,
            "explicit_metric_scale_required": True,
            "source_ply_scene_units_converted_to_metres": True,
            "relation_review_never_mutates_identity": True,
            "audit_does_not_mutate_source_artifacts": True,
        },
        "timing_seconds": time.perf_counter() - started,
        "objects": rows,
    }
    changed_inputs = [
        str(path) for path, digest in input_hashes.items() if _sha256(path) != digest
    ]
    if changed_inputs:
        raise RuntimeError(
            f"post-lift inputs changed during audit: {changed_inputs[:8]}"
        )
    mask_verified_ids = [
        int(row["object_id"])
        for row in rows
        if diagnostic_upstream_pass and bool(row["mask_gate"]["passed"])
    ]
    obb_verified_ids = [
        int(row["object_id"])
        for row in rows
        if diagnostic_upstream_pass
        and bool(row["mask_gate"]["passed"])
        and bool(row["obb_gate"]["passed"])
        and bool(row["train_obb_release_gate"]["passed"])
    ]
    # Export a bank owned by this final allowlist. The upstream lift bank may
    # still contain an object rejected by the later independent mask review.
    from farm_runtime.instance_bank import filter_instance_bank
    upstream_bank = lift / "verified_instance_bank.npz"
    descriptor = _nested(result_payload, "artifacts", "final", "verified_instance_bank")
    if not isinstance(descriptor, dict) or descriptor.get("sha256") != _sha256(upstream_bank):
        raise ValueError("upstream Gaussian bank is not SHA-bound to the lift result")
    with np.load(upstream_bank, allow_pickle=False) as bank:
        filtered_bank = filter_instance_bank(bank, mask_verified_ids, source_count=table.count)
        bank_owners = np.repeat(bank["object_ids"], np.diff(bank["indptr"]))
        if not np.array_equal(gaussian_ids[bank["indices"]], bank_owners):
            raise ValueError("upstream CSR bank disagrees with SHA-bound dense ownership")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "audit.json", report)
    _atomic_json(output / "geometry_verified_catalog.json", geometry_catalog)
    _atomic_json(output / "production_catalog.json", production_catalog)
    mask_bank_path = output / "mask_verified_bank.npz"
    np.savez_compressed(mask_bank_path, **filtered_bank)
    _id_file(output / "mask_verified_ids.txt", mask_verified_ids)
    _id_file(output / "obb_verified_ids.txt", obb_verified_ids)
    _atomic_json(
        output / "verified_allowlists_manifest.json",
        {
            "schema": "farm.post-lift-verified-allowlists.v1",
            "release_eligible": bool(global_pass),
            "experimental_nonrelease": bool(diagnostic_upstream_pass and not global_pass),
            "mask_verified_bank": {
                "path": "mask_verified_bank.npz",
                "sha256": _sha256(mask_bank_path),
                "source_bank_sha256": descriptor["sha256"],
                "object_ids": mask_verified_ids,
                "native_gaussians": int(len(filtered_bank["indices"])),
                "release_eligible": bool(global_pass),
            },
            "mask_verified": {
                "path": "mask_verified_ids.txt",
                "sha256": _sha256(output / "mask_verified_ids.txt"),
                "object_ids": mask_verified_ids,
                "requires": ["diagnostic_upstream_pass", "mask_gate_pass"],
                "semantic_gate_required": False,
            },
            "obb_verified": {
                "path": "obb_verified_ids.txt",
                "sha256": _sha256(output / "obb_verified_ids.txt"),
                "object_ids": obb_verified_ids,
                "requires": [
                    "diagnostic_upstream_pass",
                    "mask_gate_pass",
                    "post_lift_obb_support_gate_pass",
                    "strict_train_obb_release_binding_pass",
                ],
                "semantic_gate_required": False,
            },
            "source_sha256": {
                str(path): digest
                for path, digest in sorted(
                    input_hashes.items(), key=lambda item: str(item[0])
                )
            },
            "audit": {
                "path": "audit.json",
                "sha256": _sha256(output / "audit.json"),
            },
            "contracts": {
                "allowlists_are_positive_and_independent": True,
                "mask_verification_does_not_publish_semantics": True,
                "obb_verification_requires_exact_sha_bound_train_geometry": True,
                "nonrelease_diagnostic_cannot_authorize_production": True,
            },
        },
    )
    for route in (
        "upstream_lift_blocked",
        "mask_geometry_refinement",
        "obb_refit",
        "spatial_support_review",
        "relation_verification",
        "semantic_verification",
        "production_ready",
        "experimental_nonrelease_quality_verified",
    ):
        ids = (
            relation_review_ids
            if route == "relation_verification"
            else [row["object_id"] for row in rows if row["route"] == route]
        )
        _id_file(output / f"{route}_ids.txt", ids)
    print(json.dumps({"status": report["status"], **report["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
