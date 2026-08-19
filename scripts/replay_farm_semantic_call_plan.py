#!/usr/bin/env python3
"""Replay the adaptive semantic request budget from completed run artifacts.

The tool is read-only with respect to the run. It never invokes a model and
never writes unless ``--output`` explicitly names a separate report file.
Legacy artifacts lack the strict label contract, so their label-acceptance
count is reported as an auditable upper-bound approximation rather than as a
quality prediction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scene_graph.captioning.evidence import (
    evidence_pose_diversity,
    evidence_view_ids,
)
from scene_graph.captioning.label_contract import event_label_assessment


UNKNOWN = {"", "unknown", "unresolved object", "object", "item", "thing"}
INITIAL_SOURCES = {"initial_blind_review", "blind_pass_a"}
ALTERNATE_SOURCE = "blind_pass_c"
GENERATED_ENSEMBLE_SOURCES = {
    "blind_pass_b",
    "multi_source_fusion",
    "blind_pass_c",
    "neutral_form_fallback",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probable-confidence", type=float, default=0.90)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _objects(payload: dict | list) -> list[dict]:
    values = payload.get("objects") if isinstance(payload, dict) else payload
    return [dict(value) for value in values or [] if isinstance(value, dict)]


def _event(row: dict, sources: set[str]) -> dict:
    return next((
        dict(value)
        for value in row.get("semantic_evidence") or []
        if str(value.get("source") or "") in sources
    ), {})


def _legacy_high_confidence(event: dict, threshold: float) -> bool:
    category = str(event.get("category") or "").strip().lower()
    try:
        confidence = float(event.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return bool(
        str(event.get("decision") or "unknown").strip().lower() == "keep"
        and category not in UNKNOWN
        and confidence >= float(threshold)
    )


def _accepted_for_replay(event: dict, threshold: float) -> bool:
    if isinstance(event.get("label_contract"), dict):
        assessment = event_label_assessment(event, minimum_confidence=0.85)
        return bool(
            assessment.get("usable")
            and float(assessment.get("confidence") or 0.0) >= float(threshold)
        )
    return _legacy_high_confidence(event, threshold)


def _confirmation_preflight(initial: dict, alternate: dict) -> tuple[bool, dict]:
    left = set(evidence_view_ids(initial) or ())
    right = set(evidence_view_ids(alternate) or ())
    overlap = (
        len(left & right) / min(len(left), len(right))
        if left and right
        else None
    )
    pose = evidence_pose_diversity(initial, alternate)
    ready = bool(
        len(left) >= 3
        and len(right) >= 3
        and len(left | right) >= 6
        and overlap is not None
        and overlap <= 0.25
        and pose.get("pose_independent") is True
    )
    return ready, {
        "view_overlap_coefficient": overlap,
        "unique_view_count": len(left | right),
        "pose_reason": str(pose.get("reason") or ""),
    }


def replay(run_dir: Path, *, probable_confidence: float = 0.90) -> dict:
    root = run_dir.expanduser().resolve()
    semantic_root = root / "qa" / "semantics"
    review_payload = _read(semantic_root / "review" / "reviewed_robust_objects.json")
    ensemble_payload = _read(
        semantic_root / "ensemble" / "semantic_ensemble_review.json"
    )
    reconcile_report = _read(
        semantic_root / "reconciliation" / "semantic_reconciliation_report.json"
    )
    reconciled = _objects(_read(
        semantic_root / "reconciliation" / "semantic_reconciled_catalog.json"
    ))
    reviews = _objects(review_payload)
    ensembles = _objects(ensemble_payload)
    ensemble_by_id = {int(row["id"]): row for row in ensembles}
    if set(ensemble_by_id) != {int(row["id"]) for row in reviews}:
        raise ValueError("Review and ensemble object-id domains differ")

    reasons: Counter[str] = Counter()
    pose_reasons: Counter[str] = Counter()
    initial_high = 0
    alternate_rescues = 0
    legacy_exact_pose_confirmable = 0
    strict_contract_rows = 0
    for row in reviews:
        ensemble = ensemble_by_id[int(row["id"])]
        initial = _event(row, INITIAL_SOURCES)
        alternate = _event(ensemble, {ALTERNATE_SOURCE})
        strict_contract_rows += int(isinstance(initial.get("label_contract"), dict))
        high = _accepted_for_replay(initial, probable_confidence)
        preflight, diagnostics = _confirmation_preflight(initial, alternate)
        if not high:
            reason = "unresolved_or_low_confidence"
        elif preflight:
            reason = "pose_confirmable"
        else:
            reason = "single_high_confidence_probable_is_sufficient"
        reasons[reason] += 1
        pose_reasons[diagnostics["pose_reason"]] += 1
        initial_high += int(high)
        requested = reason != "single_high_confidence_probable_is_sufficient"
        alternate_high = _accepted_for_replay(
            alternate, probable_confidence
        )
        alternate_rescues += int(requested and not high and alternate_high)
        legacy_exact_pose_confirmable += int(
            preflight
            and high
            and alternate_high
            and str(initial.get("category") or "").strip().lower()
            == str(alternate.get("category") or "").strip().lower()
        )

    review_calls = sum(
        1
        for row in reviews
        for event in row.get("semantic_evidence") or []
        if str(event.get("source") or "") in {
            "initial_blind_review", "blind_pass_a",
            "independent_verification", "verification_pass",
        }
    )
    ensemble_calls = sum(
        1
        for row in ensembles
        for event in row.get("semantic_evidence") or []
        if str(event.get("source") or "") in GENERATED_ENSEMBLE_SOURCES
    )
    recovery_calls = sum(bool(row.get("physical_form_recovery")) for row in reconciled)
    recovery_verification_calls = sum(
        bool(row.get("physical_form_recovery_verification")) for row in reconciled
    )
    reconciliation_calls = int(
        int(reconcile_report.get("adjudicated_conflicts") or 0)
        + int(reconcile_report.get("physical_form_guarded") or 0)
        + recovery_calls
        + recovery_verification_calls
    )
    current_calls = review_calls + ensemble_calls + reconciliation_calls

    initial_calls = len(reviews)
    alternate_calls = (
        reasons["unresolved_or_low_confidence"] + reasons["pose_confirmable"]
    )
    # The strict response schema did not exist in legacy v5. This count assumes
    # every high-confidence legacy noun would pass the new physical/topology
    # contract, hence it is deliberately an upper bound for guard calls.
    guard_upper_bound = initial_high + alternate_rescues
    planned_upper_bound = initial_calls + alternate_calls + guard_upper_bound
    durations = {
        "review_wall_seconds": float(
            (review_payload.get("timing") or {}).get("duration_seconds") or 0.0
        ) if isinstance(review_payload, dict) else 0.0,
        "ensemble_wall_seconds": float(
            (ensemble_payload.get("timing") or {}).get("duration_seconds") or 0.0
        ) if isinstance(ensemble_payload, dict) else 0.0,
        "reconciliation_wall_seconds": float(
            reconcile_report.get("duration_seconds") or 0.0
        ),
    }
    historical_wall = sum(durations.values())
    return {
        "schema": "farm.semantic-adaptive-call-plan-replay.v1",
        "mode": (
            "strict_contract_replay"
            if strict_contract_rows == len(reviews)
            else "legacy_response_upper_bound_approximation"
        ),
        "run_dir": str(root),
        "object_count": len(reviews),
        "probable_confidence": float(probable_confidence),
        "current_run": {
            "review_calls": review_calls,
            "ensemble_calls": ensemble_calls,
            "reconciliation_calls": reconciliation_calls,
            "total_calls": current_calls,
            "wall_seconds": historical_wall,
            "wall_breakdown_seconds": durations,
        },
        "adaptive_plan": {
            "initial_blind_calls": initial_calls,
            "alternate_blind_calls": alternate_calls,
            "guard_calls_legacy_upper_bound": guard_upper_bound,
            "blind_adjudication_calls": 0,
            "conditioned_verification_calls": 0,
            "fusion_calls": 0,
            "forced_neutral_calls": 0,
            "recovery_calls": 0,
            "total_calls_legacy_upper_bound": planned_upper_bound,
            "call_reduction_lower_bound": current_calls - planned_upper_bound,
            "call_reduction_percent_lower_bound": (
                100.0 * (current_calls - planned_upper_bound) / current_calls
                if current_calls else 0.0
            ),
            "linear_wall_estimate_seconds_not_benchmark": (
                historical_wall * planned_upper_bound / current_calls
                if current_calls else 0.0
            ),
        },
        "planner_reasons": dict(sorted(reasons.items())),
        "pose_preflight_reasons": dict(sorted(pose_reasons.items())),
        "legacy_acceptance_bound": {
            "initial_high_confidence": initial_high,
            "alternate_high_confidence_rescues": alternate_rescues,
            "visible_proposal_upper_bound": guard_upper_bound,
            "exact_head_pose_confirmable": legacy_exact_pose_confirmable,
            "strict_contract_rows": strict_contract_rows,
        },
        "limitations": [
            "Legacy responses lack category-role, topology, boundedness and diagnostic-part fields.",
            "Guard-call and visible-proposal counts are upper bounds, not precision or recall predictions.",
            "The linear wall estimate ignores prompt-length, batching, cache and service-load changes.",
        ],
    }


def main() -> int:
    args = parse_args()
    report = replay(
        args.run_dir, probable_confidence=float(args.probable_confidence)
    )
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
