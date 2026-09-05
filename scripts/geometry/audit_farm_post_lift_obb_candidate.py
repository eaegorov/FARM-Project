#!/usr/bin/env python3
"""Compare two train-fitted OBBs against one immutable Gaussian support set."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import resource
import sys
import time
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
    evaluate_obb_gate,
    gaussian_obb_support_metrics,
    scene_points_to_meters,
)
from tools.farm_shaper_bridge.common import open_graphdeco_ply  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_by_id(payload: object, key: str) -> dict[int, dict[str, Any]]:
    rows = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise TypeError("audit must contain an objects list")
    result: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict) or key not in raw:
            raise TypeError(f"invalid object row without {key!r}")
        object_id = int(raw[key])
        if object_id in result:
            raise ValueError(f"duplicate object ID {object_id}")
        result[object_id] = raw
    return result


def _geometry(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    center = np.asarray(row.get("box_center_m"), dtype=np.float64)
    dimensions = np.asarray(row.get("box_dimensions_m"), dtype=np.float64)
    wxyz = np.asarray(row.get("box_wxyz"), dtype=np.float64)
    if center.shape != (3,) or dimensions.shape != (3,) or wxyz.shape != (4,):
        raise ValueError("geometry audit row has an invalid metric OBB")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(dimensions)):
        raise ValueError("geometry audit row has non-finite OBB values")
    if np.any(dimensions <= 0.0) or not np.all(np.isfinite(wxyz)):
        raise ValueError("geometry audit row has invalid OBB dimensions/quaternion")
    return center, dimensions, wxyz.tolist()


def _orientation_delta_degrees(first_wxyz: list[float], second_wxyz: list[float]) -> float:
    first = np.asarray(first_wxyz, dtype=np.float64)
    second = np.asarray(second_wxyz, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(first @ second), 0.0, 1.0))))


def _train_summary(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("train_geometry_evidence")
    reasons = validate_serialized_obb_release_evidence(evidence)
    return {
        "geometry_release_ready": row.get("geometry_release_ready") is True,
        "serialized_release_evidence_valid": not reasons,
        "validation_reasons": reasons,
        "independent_physical_timestamps": (
            evidence.get("independent_physical_timestamps")
            if isinstance(evidence, dict)
            else None
        ),
        "timestamp_median_projected_box_iou": (
            evidence.get("timestamp_median_projected_box_iou")
            if isinstance(evidence, dict)
            else None
        ),
        "timestamp_q25_projected_box_iou": (
            evidence.get("timestamp_q25_projected_box_iou")
            if isinstance(evidence, dict)
            else None
        ),
        "orientation_materiality": row.get("orientation_materiality"),
        "orientation_confidence": row.get("orientation_confidence"),
        "planar_evidence": row.get("planar_evidence"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-geometry-audit", type=Path, required=True)
    parser.add_argument("--candidate-geometry-audit", type=Path, required=True)
    parser.add_argument("--gaussian-lift-dir", type=Path, required=True)
    parser.add_argument("--object-id-file", type=Path, required=True)
    parser.add_argument("--baseline-post-lift-audit", type=Path)
    parser.add_argument("--meters-per-scene-unit", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    scale = float(args.meters_per_scene_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        parser.error("meters-per-scene-unit must be finite and positive")
    baseline_path = args.baseline_geometry_audit.resolve(strict=True)
    candidate_path = args.candidate_geometry_audit.resolve(strict=True)
    lift = args.gaussian_lift_dir.resolve(strict=True)
    ids_path = (lift / "per_gaussian_object_id.npy").resolve(strict=True)
    ply_path = (lift / "instance_labeled_full.ply").resolve(strict=True)
    object_id_path = args.object_id_file.resolve(strict=True)
    object_ids = sorted(
        {
            int(line.strip())
            for line in object_id_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    if not object_ids:
        parser.error("object ID file must not be empty")

    input_paths = [baseline_path, candidate_path, ids_path, ply_path, object_id_path]
    post_lift_rows: dict[int, dict[str, Any]] = {}
    if args.baseline_post_lift_audit:
        post_lift_path = args.baseline_post_lift_audit.resolve(strict=True)
        input_paths.append(post_lift_path)
        post_lift_rows = _rows_by_id(_load_json(post_lift_path), "object_id")
    input_hashes = {str(path): _sha256(path) for path in input_paths}

    baseline = _rows_by_id(_load_json(baseline_path), "object_id")
    candidate = _rows_by_id(_load_json(candidate_path), "object_id")
    labels = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    table = open_graphdeco_ply(ply_path)
    if labels.shape != (table.count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Gaussian labels do not match the labeled PLY")
    policy = PostLiftPolicy()
    objects: list[dict[str, Any]] = []
    all_passed = True

    for object_id in object_ids:
        if object_id not in baseline or object_id not in candidate:
            raise KeyError(f"object {object_id} missing from a geometry audit")
        selected = np.flatnonzero(labels == object_id)
        points = np.column_stack(
            tuple(table.data[name][selected] for name in ("x", "y", "z"))
        ).astype(np.float64, copy=False)
        points_m = scene_points_to_meters(points, scale)
        variants: dict[str, dict[str, Any]] = {}
        geometries: dict[str, tuple[np.ndarray, np.ndarray, list[float]]] = {}
        for name, row in (("baseline", baseline[object_id]), ("candidate", candidate[object_id])):
            center, dimensions, wxyz = _geometry(row)
            geometries[name] = (center, dimensions, wxyz)
            support = gaussian_obb_support_metrics(
                points_m,
                center_m=center,
                dimensions_m=dimensions,
                wxyz=wxyz,
                quantile=policy.support_quantile,
            )
            passed, reasons = evaluate_obb_gate(support, policy=policy)
            variants[name] = {
                "center_m": center.tolist(),
                "dimensions_m": dimensions.tolist(),
                "wxyz": wxyz,
                "train_release": _train_summary(row),
                "post_lift_support": support,
                "post_lift_obb_gate": {"passed": passed, "reasons": reasons},
            }
        before = variants["baseline"]
        after = variants["candidate"]
        candidate_pass = bool(
            after["post_lift_obb_gate"]["passed"]
            and after["train_release"]["geometry_release_ready"]
            and after["train_release"]["serialized_release_evidence_valid"]
        )
        all_passed = all_passed and candidate_pass
        base_geometry = geometries["baseline"]
        candidate_geometry = geometries["candidate"]
        frozen_row = post_lift_rows.get(object_id, {})
        objects.append(
            {
                "object_id": object_id,
                "gaussian_support": {
                    "assigned_gaussians": int(selected.size),
                    "same_support_used_for_both_variants": True,
                    "label_array_sha256": input_hashes[str(ids_path)],
                    "labeled_ply_sha256": input_hashes[str(ply_path)],
                },
                "baseline": before,
                "candidate": after,
                "delta": {
                    "orientation_degrees": _orientation_delta_degrees(
                        base_geometry[2], candidate_geometry[2]
                    ),
                    "center_distance_m": float(
                        np.linalg.norm(candidate_geometry[0] - base_geometry[0])
                    ),
                    "dimension_delta_m": (
                        candidate_geometry[1] - base_geometry[1]
                    ).tolist(),
                    "gaussian_inside_obb_fraction": float(
                        after["post_lift_support"]["gaussian_inside_obb_fraction"]
                        - before["post_lift_support"]["gaussian_inside_obb_fraction"]
                    ),
                },
                "frozen_heldout_mask_evaluation": {
                    "evaluation_only_not_used_for_fit": True,
                    "unchanged_because_gaussian_labels_are_unchanged": True,
                    "mask_gate": frozen_row.get("mask_gate"),
                },
                "candidate_acceptance": {
                    "passed": candidate_pass,
                    "requires_train_release_and_post_lift_support": True,
                    "thresholds_weakened": False,
                },
            }
        )

    changed = [path for path, digest in input_hashes.items() if _sha256(Path(path)) != digest]
    if changed:
        raise RuntimeError(f"audit inputs changed during execution: {changed}")
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "schema": "farm.post-lift-obb-candidate-ab.v1",
        "status": "candidate_pass" if all_passed else "candidate_rejected",
        "candidate_accepted_for_diagnostic_only": all_passed,
        "not_a_scene_publication": True,
        "inputs": {
            "sha256": input_hashes,
            "meters_per_scene_unit": scale,
        },
        "policy": policy.to_dict(),
        "contracts": {
            "same_immutable_gaussian_support_for_ab": True,
            "train_rgbd_release_evidence_required": True,
            "frozen_heldout_is_evaluation_only": True,
            "no_object_id_or_category_special_case": True,
            "release_thresholds_weakened": False,
        },
        "objects": objects,
        "timing_seconds": time.perf_counter() - started,
        "peak_process_rss_mib": peak_rss_mib,
        "execution_device": "cpu_only_no_cuda_code_path",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace immutable output: {args.output}")
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "objects": len(objects)}, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
