#!/usr/bin/env python3
"""Select only measured held-out geometry improvements over a baseline scene."""

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
import time
from pathlib import Path
from typing import Any, Mapping


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


def _objects(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("objects") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise TypeError(f"heldout QC has no object rows: {path}")
    return {
        int(row["object_id"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and "object_id" in row
    }


def _metric(row: Mapping[str, Any], name: str, statistic: str) -> float:
    value = ((row.get("summaries") or {}).get(name) or {}).get(statistic)
    number = float(value or 0.0)
    return number if math.isfinite(number) else 0.0


def evaluate_object(
    object_id: int,
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    *,
    minimum_holdouts: int,
    minimum_good_holdouts: int,
    minimum_iou: float,
    minimum_precision: float,
    minimum_recall: float,
    minimum_component_fraction: float,
    maximum_iou_regression: float,
    maximum_precision_regression: float,
    maximum_recall_regression: float,
    maximum_component_regression: float,
    minimum_iou_gain: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if baseline is None:
        reasons.append("baseline_qc_missing")
        baseline = {}
    if candidate is None:
        reasons.append("candidate_qc_missing")
        candidate = {}

    baseline_metrics = {
        "heldout_timestamps": int(baseline.get("heldout_timestamps") or 0),
        "good_timestamps": int(baseline.get("good_timestamps") or 0),
        "provisional_gaussians": int(baseline.get("provisional_gaussians") or 0),
        "median_iou": _metric(baseline, "iou", "median"),
        "minimum_precision": _metric(baseline, "precision", "minimum"),
        "minimum_recall": _metric(baseline, "recall", "minimum"),
        "median_component_fraction": _metric(
            baseline, "largest_component_fraction", "median"
        ),
    }
    candidate_metrics = {
        "heldout_timestamps": int(candidate.get("heldout_timestamps") or 0),
        "good_timestamps": int(candidate.get("good_timestamps") or 0),
        "provisional_gaussians": int(candidate.get("provisional_gaussians") or 0),
        "median_iou": _metric(candidate, "iou", "median"),
        "minimum_precision": _metric(candidate, "precision", "minimum"),
        "minimum_recall": _metric(candidate, "recall", "minimum"),
        "median_component_fraction": _metric(
            candidate, "largest_component_fraction", "median"
        ),
    }
    deltas = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in candidate_metrics
    }

    if str(candidate.get("status") or "") != "verified":
        reasons.append("candidate_status_not_verified")
    gates = (
        ("heldout_timestamps", minimum_holdouts, "insufficient_candidate_holdouts"),
        ("good_timestamps", minimum_good_holdouts, "insufficient_good_candidate_holdouts"),
        ("median_iou", minimum_iou, "candidate_iou_below_threshold"),
        ("minimum_precision", minimum_precision, "candidate_precision_below_threshold"),
        ("minimum_recall", minimum_recall, "candidate_recall_below_threshold"),
        (
            "median_component_fraction",
            minimum_component_fraction,
            "candidate_component_fraction_below_threshold",
        ),
    )
    for key, threshold, reason in gates:
        if candidate_metrics[key] < threshold:
            reasons.append(reason)

    regressions = (
        ("median_iou", maximum_iou_regression, "iou_regression_exceeds_tolerance"),
        (
            "minimum_precision",
            maximum_precision_regression,
            "precision_regression_exceeds_tolerance",
        ),
        ("minimum_recall", maximum_recall_regression, "recall_regression_exceeds_tolerance"),
        (
            "median_component_fraction",
            maximum_component_regression,
            "component_regression_exceeds_tolerance",
        ),
    )
    for key, tolerance, reason in regressions:
        if deltas[key] < -float(tolerance):
            reasons.append(reason)

    benefit_reasons: list[str] = []
    if deltas["heldout_timestamps"] >= 1:
        benefit_reasons.append("additional_independent_holdout")
    if deltas["median_iou"] >= minimum_iou_gain:
        benefit_reasons.append("median_iou_gain")
    if deltas["minimum_precision"] >= 0.03:
        benefit_reasons.append("worst_view_precision_gain")
    if deltas["minimum_recall"] >= 0.03:
        benefit_reasons.append("worst_view_recall_gain")
    if deltas["median_component_fraction"] >= 0.01:
        benefit_reasons.append("component_coherence_gain")
    if not benefit_reasons:
        reasons.append("no_measurable_heldout_benefit")

    return {
        "object_id": int(object_id),
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "benefit_reasons": benefit_reasons,
        "baseline_status": str(baseline.get("status") or "missing"),
        "candidate_status": str(candidate.get("status") or "missing"),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-heldout-qc", type=Path, required=True)
    parser.add_argument("--candidate-heldout-qc", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-holdouts", type=int, default=3)
    parser.add_argument("--minimum-good-holdouts", type=int, default=3)
    parser.add_argument("--minimum-iou", type=float, default=0.65)
    parser.add_argument("--minimum-precision", type=float, default=0.65)
    parser.add_argument("--minimum-recall", type=float, default=0.70)
    parser.add_argument("--minimum-component-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-iou-regression", type=float, default=0.03)
    parser.add_argument("--maximum-precision-regression", type=float, default=0.15)
    parser.add_argument("--maximum-recall-regression", type=float, default=0.05)
    parser.add_argument("--maximum-component-regression", type=float, default=0.03)
    parser.add_argument("--minimum-iou-gain", type=float, default=0.01)
    args = parser.parse_args()
    started = time.perf_counter()
    baseline_path = args.baseline_heldout_qc.expanduser().resolve(strict=True)
    candidate_path = args.candidate_heldout_qc.expanduser().resolve(strict=True)
    acceptance_path = args.acceptance_report.expanduser().resolve(strict=True)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    candidate_ids = sorted({int(value) for value in acceptance.get("accepted_object_ids") or []})
    baseline = _objects(baseline_path)
    candidate = _objects(candidate_path)
    rows = [
        evaluate_object(
            object_id,
            baseline.get(object_id),
            candidate.get(object_id),
            minimum_holdouts=int(args.minimum_holdouts),
            minimum_good_holdouts=int(args.minimum_good_holdouts),
            minimum_iou=float(args.minimum_iou),
            minimum_precision=float(args.minimum_precision),
            minimum_recall=float(args.minimum_recall),
            minimum_component_fraction=float(args.minimum_component_fraction),
            maximum_iou_regression=float(args.maximum_iou_regression),
            maximum_precision_regression=float(args.maximum_precision_regression),
            maximum_recall_regression=float(args.maximum_recall_regression),
            maximum_component_regression=float(args.maximum_component_regression),
            minimum_iou_gain=float(args.minimum_iou_gain),
        )
        for object_id in candidate_ids
    ]
    accepted_ids = [row["object_id"] for row in rows if row["accepted"]]
    output = args.output_dir.expanduser().resolve()
    _atomic_text(output / "accepted_geometry_ids.txt", "".join(f"{value}\n" for value in accepted_ids))
    report = {
        "schema": "farm.heldout-geometry-improvement-selection.v1",
        "status": "PASS",
        "candidate_object_ids": candidate_ids,
        "accepted_object_ids": accepted_ids,
        "rejected_object_ids": [
            row["object_id"] for row in rows if not row["accepted"]
        ],
        "objects": rows,
        "policy": {
            "absolute_quality_required": True,
            "bounded_regression_required": True,
            "measurable_heldout_benefit_required": True,
            "minimum_holdouts": int(args.minimum_holdouts),
            "minimum_good_holdouts": int(args.minimum_good_holdouts),
            "minimum_iou": float(args.minimum_iou),
            "minimum_precision": float(args.minimum_precision),
            "minimum_recall": float(args.minimum_recall),
            "minimum_component_fraction": float(args.minimum_component_fraction),
            "maximum_iou_regression": float(args.maximum_iou_regression),
            "maximum_precision_regression": float(args.maximum_precision_regression),
            "maximum_recall_regression": float(args.maximum_recall_regression),
            "maximum_component_regression": float(args.maximum_component_regression),
            "minimum_iou_gain": float(args.minimum_iou_gain),
        },
        "timing_seconds": time.perf_counter() - started,
        "hashes": {
            "baseline_heldout_qc_sha256": _sha256(baseline_path),
            "candidate_heldout_qc_sha256": _sha256(candidate_path),
            "acceptance_report_sha256": _sha256(acceptance_path),
        },
    }
    _atomic_json(output / "selection.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
