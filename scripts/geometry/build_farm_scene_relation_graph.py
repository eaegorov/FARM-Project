#!/usr/bin/env python3
"""Build a lightweight Lucida-inspired scene relation graph for FARM.

The graph is deliberately advisory for geometric proximity and fail-closed for
inpainting eligibility.  It does not create a second object detector.  Exact
FARM OBBs, structured Qwen evidence and held-out Gaussian QC determine whether
an object needs mask refinement, independent verification, or exclusion.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CATEGORY_SPELLING_CORRECTIONS = {"palet": "pallet"}
CATEGORY_FAMILY_ALIASES = {
    "cleaner": "cleaning machine",
    "floor cleaner": "cleaning machine",
    "floor scrubber": "cleaning machine",
    "vacuum cleaner": "cleaning machine",
}
GENERIC_CATEGORIES = {"", "object", "unknown", "unresolved object"}
AMBIGUOUS_CATEGORIES = {"cleaner", "machine", "unit", "device", "equipment", "container"}
WALL_CUES = ("wall-mounted", "wall mounted", "mounted on a wall", "mounted to a wall")
CEILING_CUES = ("ceiling-mounted", "ceiling mounted", "mounted under a ceiling")
FLOOR_FIXED_CUES = ("floor-mounted", "floor mounted", "bolted to the floor")
PORTABLE_CUES = ("on casters", "mounted on casters", "wheels at the base", "wheeled")
SURFACE_CUES = (
    "on a workbench", "on a flat surface", "mounted on a flat surface",
    "positioned on a workbench", "resting on a surface",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _load_rows(path: Path, *, object_key: str = "objects") -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get(object_key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"invalid object rows: {path}")
    return [dict(row) for row in rows]


def normalize_category(value: object) -> str:
    category = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return CATEGORY_SPELLING_CORRECTIONS.get(category, category)


def category_family(value: object) -> str:
    category = normalize_category(value)
    return CATEGORY_FAMILY_ALIASES.get(category, category)


def _vector(value: object, width: int, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"expected finite vector of width {width}")
    if positive and np.any(array <= 0.0):
        raise ValueError("dimensions must be strictly positive")
    return array


def quaternion_matrix(wxyz: object) -> np.ndarray:
    q = _vector(wxyz, 4)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = q / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def obb_corners(row: Mapping[str, Any]) -> np.ndarray:
    center = _vector(row.get("center_m"), 3)
    dimensions = _vector(row.get("dimensions_m"), 3, positive=True)
    rotation = quaternion_matrix(row.get("wxyz"))
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    return center + (signs * (0.5 * dimensions)) @ rotation.T


def pair_geometry(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    a = obb_corners(left)
    b = obb_corners(right)
    a_min, a_max = a.min(axis=0), a.max(axis=0)
    b_min, b_max = b.min(axis=0), b.max(axis=0)
    overlap = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    intersection = float(np.prod(overlap))
    a_volume = float(np.prod(a_max - a_min))
    b_volume = float(np.prod(b_max - b_min))
    separation = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    gap = float(np.linalg.norm(separation))
    centers = (_vector(left.get("center_m"), 3), _vector(right.get("center_m"), 3))
    diagonal = min(
        float(np.linalg.norm(_vector(left.get("dimensions_m"), 3))),
        float(np.linalg.norm(_vector(right.get("dimensions_m"), 3))),
    )
    return {
        "aabb_intersection_m3": intersection,
        "aabb_smaller_overlap": intersection / max(min(a_volume, b_volume), 1.0e-12),
        "aabb_iou": intersection / max(a_volume + b_volume - intersection, 1.0e-12),
        "aabb_gap_m": gap,
        "center_distance_m": float(np.linalg.norm(centers[0] - centers[1])),
        "normalized_center_distance": float(np.linalg.norm(centers[0] - centers[1])) / max(diagonal, 1.0e-12),
    }


def _text(row: Mapping[str, Any]) -> str:
    values: list[str] = [
        str(row.get("description") or ""),
        str(row.get("category") or ""),
    ]
    values.extend(str(value) for value in row.get("attributes") or [])
    for event in row.get("semantic_evidence") or []:
        if isinstance(event, Mapping):
            contract = event.get("label_contract")
            if isinstance(contract, Mapping):
                values.append(str(contract.get("description") or ""))
    return " ".join(values).lower()


def semantic_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    eligible: list[str] = []
    all_categories: list[str] = []
    whole_events = 0
    incomplete_events = 0
    for event in row.get("semantic_evidence") or []:
        if not isinstance(event, Mapping):
            continue
        contract = event.get("label_contract")
        if not isinstance(contract, Mapping):
            continue
        category = normalize_category(contract.get("category"))
        if category not in GENERIC_CATEGORIES:
            all_categories.append(category)
            if event.get("confirmation_eligible") is not False:
                eligible.append(category)
        whole = (
            contract.get("complete_bounded") is True
            and str(contract.get("topology") or "") == "standalone_whole"
            and contract.get("context_sufficient") is True
            and float(contract.get("visible_target_coverage") or 0.0) >= 0.90
        )
        whole_events += int(whole and event.get("confirmation_eligible") is not False)
        incomplete_events += int(not whole)
    final_category = normalize_category(row.get("category"))
    eligible_set = sorted(set(eligible))
    eligible_families = sorted({category_family(value) for value in eligible_set})
    final_family = category_family(final_category)
    return {
        "final_category": final_category,
        "final_category_family": final_family,
        "eligible_categories": eligible_set,
        "eligible_category_families": eligible_families,
        "all_observed_categories": sorted(set(all_categories)),
        "eligible_category_conflict": len(eligible_families) > 1,
        "final_category_mismatch": bool(
            final_category not in GENERIC_CATEGORIES
            and eligible_families
            and final_family not in eligible_families
        ),
        "semantic_specificity_review_required": final_category in AMBIGUOUS_CATEGORIES,
        "eligible_whole_event_count": whole_events,
        "incomplete_event_count": incomplete_events,
        "normalization_applied": str(row.get("category") or "").strip().lower() != final_category,
    }


def heldout_summary(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "status": "missing", "heldout_timestamps": 0, "good_timestamps": 0,
            "provisional_gaussians": 0, "median_iou": 0.0,
            "minimum_precision": 0.0, "minimum_recall": 0.0,
        }
    summaries = row.get("summaries") or {}
    return {
        "status": str(row.get("status") or "unknown"),
        "heldout_timestamps": int(row.get("heldout_timestamps") or 0),
        "good_timestamps": int(row.get("good_timestamps") or 0),
        "provisional_gaussians": int(row.get("provisional_gaussians") or 0),
        "median_iou": float((summaries.get("iou") or {}).get("median") or 0.0),
        "minimum_precision": float((summaries.get("precision") or {}).get("minimum") or 0.0),
        "minimum_recall": float((summaries.get("recall") or {}).get("minimum") or 0.0),
        "median_component_fraction": float(
            (summaries.get("largest_component_fraction") or {}).get("median") or 0.0
        ),
    }


def scene_relations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = _text(row)
    object_id = int(row["id"])
    edges: list[dict[str, Any]] = []
    if any(cue in text for cue in WALL_CUES):
        edges.append({"source": object_id, "target": "scene:wall", "relation": "attached_to", "confidence": "explicit_text"})
    if any(cue in text for cue in CEILING_CUES):
        edges.append({"source": object_id, "target": "scene:ceiling", "relation": "attached_to", "confidence": "explicit_text"})
    portable = any(cue in text for cue in PORTABLE_CUES)
    if any(cue in text for cue in FLOOR_FIXED_CUES) and not portable:
        edges.append({"source": object_id, "target": "scene:floor", "relation": "attached_to", "confidence": "explicit_text"})
    elif portable:
        edges.append({"source": object_id, "target": "scene:floor", "relation": "supported_by", "confidence": "portable_cue"})
    if any(cue in text for cue in SURFACE_CUES):
        edges.append({"source": object_id, "target": "scene:work_surface", "relation": "supported_by", "confidence": "explicit_text"})
    return edges


def classify_target(
    semantic: Mapping[str, Any],
    heldout: Mapping[str, Any],
    relations: Iterable[Mapping[str, Any]],
    *,
    audit_status: str = "missing",
    minimum_holdouts: int,
    minimum_median_iou: float,
    minimum_precision: float,
    minimum_recall: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    hard_attached = any(
        edge.get("relation") == "attached_to"
        and edge.get("target") in {"scene:wall", "scene:ceiling"}
        for edge in relations
    )
    floor_attached = any(
        edge.get("relation") == "attached_to"
        and edge.get("target") == "scene:floor"
        for edge in relations
    )
    if hard_attached:
        return "exclude_attached", ["explicit_wall_or_ceiling_attachment"]
    if floor_attached:
        reasons.append("floor_attachment_requires_relation_review")
    if semantic.get("eligible_category_conflict"):
        reasons.append("eligible_identity_conflict")
    if semantic.get("final_category_mismatch"):
        reasons.append("final_category_not_supported_by_eligible_events")
    if semantic.get("normalization_applied"):
        reasons.append("category_normalization_required")
    if int(heldout.get("heldout_timestamps") or 0) < minimum_holdouts:
        reasons.append("insufficient_independent_holdouts")
    if str(heldout.get("status")) != "verified":
        reasons.append("gaussian_status_not_verified")
    if float(heldout.get("median_iou") or 0.0) < minimum_median_iou:
        reasons.append("median_iou_below_threshold")
    if float(heldout.get("minimum_precision") or 0.0) < minimum_precision:
        reasons.append("worst_view_precision_below_threshold")
    if float(heldout.get("minimum_recall") or 0.0) < minimum_recall:
        reasons.append("worst_view_recall_below_threshold")
    if reasons:
        return "mask_refinement", reasons
    if audit_status == "needs_independent_evidence":
        return "independent_verification", ["whole_object_audit_requires_independent_evidence"]
    if int(semantic.get("eligible_whole_event_count") or 0) < 2:
        return "independent_verification", ["insufficient_eligible_whole_events"]
    return "relation_verification", ["geometry_and_semantics_pass_initial_gate"]


def classify_routes(
    semantic: Mapping[str, Any],
    heldout: Mapping[str, Any],
    relations: Iterable[Mapping[str, Any]],
    *,
    audit_status: str = "missing",
    minimum_holdouts: int,
    minimum_median_iou: float,
    minimum_precision: float,
    minimum_recall: float,
) -> tuple[dict[str, list[str]], str]:
    """Route semantic, geometry and inpainting concerns independently."""
    routes: dict[str, list[str]] = {}

    def add(route: str, reason: str) -> None:
        routes.setdefault(route, [])
        if reason not in routes[route]:
            routes[route].append(reason)

    hard_attached = any(
        edge.get("relation") == "attached_to"
        and edge.get("target") in {"scene:wall", "scene:ceiling"}
        for edge in relations
    )
    floor_attached = any(
        edge.get("relation") == "attached_to"
        and edge.get("target") == "scene:floor"
        for edge in relations
    )
    if hard_attached:
        add("exclude_attached", "explicit_wall_or_ceiling_attachment")
        add("inpainting_exclusion", "attached_object_is_not_removable")
    elif floor_attached:
        add("relation_verification", "floor_attachment_requires_relation_review")
        add("inpainting_exclusion", "floor_attachment_not_cleared_for_removal")

    if semantic.get("normalization_applied"):
        add("semantic_normalization", "category_spelling_normalization_required")
    if semantic.get("eligible_category_conflict"):
        add("semantic_verification", "eligible_identity_conflict")
    if semantic.get("final_category_mismatch"):
        add("semantic_verification", "final_category_not_supported_by_eligible_events")
    if semantic.get("semantic_specificity_review_required"):
        add("semantic_verification", "category_is_not_specific_enough_for_publication")

    geometry_reasons: list[str] = []
    if int(heldout.get("heldout_timestamps") or 0) < minimum_holdouts:
        geometry_reasons.append("insufficient_independent_holdouts")
    if str(heldout.get("status")) != "verified":
        geometry_reasons.append("gaussian_status_not_verified")
    if float(heldout.get("median_iou") or 0.0) < minimum_median_iou:
        geometry_reasons.append("median_iou_below_threshold")
    if float(heldout.get("minimum_precision") or 0.0) < minimum_precision:
        geometry_reasons.append("worst_view_precision_below_threshold")
    if float(heldout.get("minimum_recall") or 0.0) < minimum_recall:
        geometry_reasons.append("worst_view_recall_below_threshold")
    for reason in geometry_reasons:
        add("mask_geometry_refinement", reason)

    if audit_status == "needs_independent_evidence":
        add("independent_verification", "whole_object_audit_requires_independent_evidence")
    if int(semantic.get("eligible_whole_event_count") or 0) < 2:
        add("independent_verification", "insufficient_eligible_whole_events")
    if not routes:
        add("relation_verification", "geometry_and_semantics_pass_initial_gate")

    priority = (
        "exclude_attached", "mask_geometry_refinement", "independent_verification",
        "semantic_verification", "semantic_normalization", "relation_verification",
        "inpainting_exclusion",
    )
    primary = next(route for route in priority if route in routes)
    return routes, primary


def build_graph(
    semantic_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    candidate_ids: set[int],
    *,
    near_gap_m: float = 0.25,
    overlap_ratio: float = 0.15,
    minimum_holdouts: int = 3,
    minimum_median_iou: float = 0.65,
    minimum_precision: float = 0.65,
    minimum_recall: float = 0.70,
) -> dict[str, Any]:
    semantic_by_id = {int(row["id"]): row for row in semantic_rows if "id" in row}
    geometry_by_id = {int(row["id"]): row for row in geometry_rows if "id" in row}
    heldout_by_id = {int(row["object_id"]): row for row in heldout_rows if "object_id" in row}
    audit_by_id = {int(row["object_id"]): row for row in audit_rows if "object_id" in row}
    missing = sorted(candidate_ids - set(geometry_by_id))
    if missing:
        raise ValueError(f"candidate IDs absent from geometry catalog: {missing}")

    edges: list[dict[str, Any]] = []
    ids = sorted(geometry_by_id)
    for offset, left_id in enumerate(ids):
        for right_id in ids[offset + 1:]:
            metrics = pair_geometry(geometry_by_id[left_id], geometry_by_id[right_id])
            if metrics["aabb_smaller_overlap"] >= overlap_ratio:
                edges.append({
                    "source": left_id, "target": right_id,
                    "relation": "spatial_overlap_candidate",
                    "confidence": "geometry_advisory", "metrics": metrics,
                })
            elif metrics["aabb_gap_m"] <= near_gap_m:
                edges.append({
                    "source": left_id, "target": right_id,
                    "relation": "near", "confidence": "geometry_advisory",
                    "metrics": metrics,
                })

    nodes: list[dict[str, Any]] = []
    route_queues: dict[str, list[int]] = {
        "mask_geometry_refinement": [],
        "semantic_normalization": [],
        "semantic_verification": [],
        "independent_verification": [],
        "relation_verification": [],
        "inpainting_exclusion": [],
        "exclude_attached": [],
    }
    for object_id in sorted(candidate_ids):
        semantic_row = semantic_by_id.get(object_id, {"id": object_id})
        semantic = semantic_summary(semantic_row)
        heldout = heldout_summary(heldout_by_id.get(object_id))
        relations = scene_relations({**semantic_row, **geometry_by_id[object_id]})
        audit_status = str(audit_by_id.get(object_id, {}).get("status") or "missing")
        edges.extend(relations)
        routes, decision = classify_routes(
            semantic, heldout, relations,
            audit_status=audit_status,
            minimum_holdouts=minimum_holdouts,
            minimum_median_iou=minimum_median_iou,
            minimum_precision=minimum_precision,
            minimum_recall=minimum_recall,
        )
        for route in routes:
            route_queues[route].append(object_id)
        reasons = [reason for values in routes.values() for reason in values]
        nodes.append({
            "object_id": object_id,
            "category": semantic["final_category"],
            "source_category": str(semantic_row.get("category") or ""),
            "semantic": semantic,
            "heldout_gaussian_qc": heldout,
            "whole_object_audit_status": audit_status,
            "scene_relations": relations,
            "routes": sorted(routes),
            "route_reasons": routes,
            "decision": decision,
            "decision_reasons": reasons,
            "geometry": {
                "center_m": geometry_by_id[object_id]["center_m"],
                "dimensions_m": geometry_by_id[object_id]["dimensions_m"],
                "wxyz": geometry_by_id[object_id]["wxyz"],
            },
        })

    edges.sort(key=lambda row: (str(row["source"]), str(row["target"]), str(row["relation"])))
    queues = {f"{name}_ids": values for name, values in route_queues.items()}
    queues["mask_refinement_ids"] = list(route_queues["mask_geometry_refinement"])
    queues["excluded_attached_ids"] = list(route_queues["exclude_attached"])
    queues["full_colmap_processing_ids"] = sorted(set(
        route_queues["mask_geometry_refinement"] + route_queues["independent_verification"]
    ))
    queues["all_processing_ids"] = sorted(set(
        route_queues["mask_geometry_refinement"]
        + route_queues["independent_verification"]
        + route_queues["semantic_normalization"]
        + route_queues["semantic_verification"]
        + route_queues["relation_verification"]
    ))
    return {
        "schema": "farm.scene-relation-graph.v1",
        "policy": {
            "role": "advisory_spatial_graph_and_fail_closed_inpainting_router",
            "full_scene_detector_used": False,
            "geometry_edges_are_advisory": True,
            "wall_or_ceiling_attachment_is_hard_exclusion": True,
            "floor_attachment_requires_review": True,
            "near_gap_m": near_gap_m,
            "aabb_smaller_overlap_threshold": overlap_ratio,
            "minimum_holdouts": minimum_holdouts,
            "minimum_median_iou": minimum_median_iou,
            "minimum_worst_view_precision": minimum_precision,
            "minimum_worst_view_recall": minimum_recall,
        },
        "candidate_ids": sorted(candidate_ids),
        "nodes": nodes,
        "edges": edges,
        "queues": queues,
        "decision_counts": dict(sorted(Counter(row["decision"] for row in nodes).items())),
    }


def _ids(values: list[int]) -> str:
    return "".join(f"{value}\n" for value in values)


def load_candidate_ids(values: list[int] | None, path: Path | None) -> set[int]:
    """Resolve one explicit candidate source without silently dropping bad IDs."""

    raw_values: list[int | str] = list(values or [])
    if path is not None:
        resolved = path.expanduser().resolve(strict=True)
        raw_values = [
            line.strip()
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    candidate_ids: list[int] = []
    for value in raw_values:
        try:
            object_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid candidate object ID: {value!r}") from exc
        if object_id < 0:
            raise ValueError(f"candidate object IDs must be non-negative: {object_id}")
        candidate_ids.append(object_id)
    if not candidate_ids:
        raise ValueError("candidate object ID source is empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate object IDs must be unique")
    return set(candidate_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-catalog", type=Path, required=True)
    parser.add_argument("--geometry-catalog", type=Path, required=True)
    parser.add_argument("--heldout-qc", type=Path, required=True)
    parser.add_argument("--whole-object-audit", type=Path, required=True)
    candidate_group = parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument("--candidate-id", type=int, action="append")
    candidate_group.add_argument(
        "--candidate-id-file",
        type=Path,
        help="Newline-delimited canonical object IDs from a prior audit queue.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-gap-m", type=float, default=0.25)
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-holdouts", type=int, default=3)
    parser.add_argument("--minimum-median-iou", type=float, default=0.65)
    parser.add_argument("--minimum-precision", type=float, default=0.65)
    parser.add_argument("--minimum-recall", type=float, default=0.70)
    args = parser.parse_args()
    try:
        candidate_ids = load_candidate_ids(args.candidate_id, args.candidate_id_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    started = time.perf_counter()
    paths = {
        "semantic_catalog": args.semantic_catalog.expanduser().resolve(strict=True),
        "geometry_catalog": args.geometry_catalog.expanduser().resolve(strict=True),
        "heldout_qc": args.heldout_qc.expanduser().resolve(strict=True),
        "whole_object_audit": args.whole_object_audit.expanduser().resolve(strict=True),
    }
    if args.candidate_id_file is not None:
        paths["candidate_id_file"] = args.candidate_id_file.expanduser().resolve(strict=True)
    heldout_payload = json.loads(paths["heldout_qc"].read_text(encoding="utf-8"))
    audit_payload = json.loads(paths["whole_object_audit"].read_text(encoding="utf-8"))
    graph = build_graph(
        _load_rows(paths["semantic_catalog"]),
        _load_rows(paths["geometry_catalog"]),
        [dict(row) for row in heldout_payload.get("objects") or []],
        [dict(row) for row in audit_payload.get("objects") or []],
        candidate_ids,
        near_gap_m=float(args.near_gap_m),
        overlap_ratio=float(args.overlap_ratio),
        minimum_holdouts=int(args.minimum_holdouts),
        minimum_median_iou=float(args.minimum_median_iou),
        minimum_precision=float(args.minimum_precision),
        minimum_recall=float(args.minimum_recall),
    )
    graph["created_unix_s"] = time.time()
    graph["timing_seconds"] = time.perf_counter() - started
    graph["sources"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    output = args.output_dir.expanduser().resolve()
    _atomic_json(output / "scene_relation_graph.json", graph)
    for name, values in graph["queues"].items():
        _atomic_text(output / f"{name}.txt", _ids(values))
    full_colmap_ids = set(graph["queues"]["full_colmap_processing_ids"])
    vocabulary = sorted({
        normalize_category(category)
        for row in graph["nodes"]
        if int(row["object_id"]) in full_colmap_ids
        for category in [row["category"], *row["semantic"]["eligible_categories"]]
        if normalize_category(category) not in GENERIC_CATEGORIES
    })
    _atomic_text(output / "yoloe_vocabulary.txt", "".join(f"{value}\n" for value in vocabulary))
    print(json.dumps({
        "schema": graph["schema"],
        "decision_counts": graph["decision_counts"],
        "queues": graph["queues"],
        "vocabulary": vocabulary,
        "output_dir": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
