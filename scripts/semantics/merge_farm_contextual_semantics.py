#!/usr/bin/env python3
"""Merge natural-context Qwen evidence into an evidence-tiered FARM catalog.

The contextual observation can repair a probable presentation noun, but one
request can never create a confirmed semantic tier. Existing evidence remains
auditable, and mask completeness remains separate from identity.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from scene_graph.captioning.evidence import evidence_event_id
from scene_graph.captioning.label_contract import (
    assess_open_vocabulary_identity,
    assess_whole_object_readiness,
    normalize_open_noun,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _catalog_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("objects") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"unsupported object catalog: {path}")
    if not all(isinstance(row, dict) and "id" in row for row in rows):
        raise TypeError(f"catalog contains invalid object rows: {path}")
    return [copy.deepcopy(row) for row in rows]


def _review_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "farm.open-vocabulary-crop-review.v2":
        raise ValueError(f"unsupported contextual review: {path}")
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise TypeError(f"contextual review lacks objects: {path}")
    return [copy.deepcopy(row) for row in rows if isinstance(row, dict) and "id" in row]


def _context_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
    supported_sources = {
        "contextual_dual_panel_review",
        "initial_blind_review",
        "independent_verification",
        "candidate_context_adjudication",
    }
    priority = {
        "initial_blind_review": 0,
        "contextual_dual_panel_review": 1,
        "candidate_context_adjudication": 2,
        "independent_verification": 3,
    }
    candidates = [
        copy.deepcopy(event)
        for event in row.get("semantic_evidence") or []
        if (
            isinstance(event, dict)
            and str(event.get("source") or "") in supported_sources
            and isinstance(event.get("label_contract"), dict)
        )
    ]
    candidates.sort(key=lambda event: priority.get(str(event.get("source") or ""), -1))
    if not candidates and isinstance(row.get("review_label_contract"), dict):
        candidates = [{
            "source": "contextual_dual_panel_review",
            "event_id": f"object:{int(row['id'])}:contextual_dual_panel_review",
            "label_contract": copy.deepcopy(row["review_label_contract"]),
            "confirmation_eligible": True,
        }]
    return candidates[-1] if candidates else None


def _append_event(events: list[dict[str, Any]], event: dict[str, Any]) -> bool:
    event_id = evidence_event_id(event)
    fingerprints = {
        (
            evidence_event_id(value),
            str(value.get("evidence_fingerprint_sha256") or ""),
        )
        for value in events
        if isinstance(value, dict)
    }
    fingerprint = str(event.get("evidence_fingerprint_sha256") or "")
    if (event_id, fingerprint) in fingerprints:
        return False
    events.append(copy.deepcopy(event))
    return True


def merge_object(
    base: Mapping[str, Any],
    review: Mapping[str, Any] | None,
    *,
    minimum_confidence: float,
) -> tuple[dict[str, Any], str]:
    output = copy.deepcopy(dict(base))
    if review is None:
        return output, "not_reviewed"
    event = _context_event(review)
    if event is None:
        return output, "missing_context_event"
    events = [
        copy.deepcopy(value)
        for value in output.get("semantic_evidence") or []
        if isinstance(value, dict)
    ]
    _append_event(events, event)
    output["semantic_evidence"] = events
    contract = event.get("label_contract")
    prior = {
        "category": str(output.get("category") or ""),
        "semantic_tier": str(output.get("semantic_tier") or "geometry_only"),
        "semantic_gate_reason": str(output.get("semantic_gate_reason") or ""),
    }
    identity = assess_open_vocabulary_identity(
        contract, minimum_confidence=minimum_confidence
    )
    event_decision = str(
        event.get("decision") or (contract or {}).get("decision") or ""
    ).lower()
    if not identity.get("usable") or event_decision != "keep":
        if (
            prior["semantic_tier"] != "confirmed"
            and (
                bool((contract or {}).get("context_sufficient", False))
                or (
                    str(event.get("source") or "") == "independent_verification"
                    and event_decision != "keep"
                )
            )
        ):
            output.update({
                "category": "unresolved object",
                "description": str((contract or {}).get("description") or ""),
                "attributes": list((contract or {}).get("attributes") or [])[:8],
                "label_contract": copy.deepcopy(contract),
                "review_decision": "unknown",
                "review_confidence": float((contract or {}).get("confidence") or 0.0),
                "semantic_tier": "geometry_only",
                "evidence_tier": "geometry_only",
                "semantic_status": "contextual_identity_unresolved",
                "semantic_gate_reason": "contextual_identity_quarantined",
                "contextual_semantic_review": {
                    "status": "identity_unresolved_quarantined",
                    "assessment": identity,
                    "prior": prior,
                    "event_decision": event_decision,
                    "event_id": evidence_event_id(event),
                },
            })
            return output, "identity_quarantined"
        output["contextual_semantic_review"] = {
            "status": (
                "confirmed_identity_preserved"
                if prior["semantic_tier"] == "confirmed"
                else "identity_unresolved"
            ),
            "assessment": identity,
            "prior": prior,
            "event_decision": event_decision,
            "event_id": evidence_event_id(event),
        }
        return output, (
            "confirmed_identity_preserved"
            if prior["semantic_tier"] == "confirmed"
            else "identity_unresolved"
        )
    if not bool((contract or {}).get("context_sufficient", False)):
        output["contextual_semantic_review"] = {
            "status": "context_insufficient",
            "assessment": identity,
            "event_id": evidence_event_id(event),
        }
        return output, "context_insufficient"

    category = normalize_open_noun(identity.get("category"))
    if not category:
        return output, "identity_unresolved"
    if prior["semantic_tier"] == "confirmed" and (
        normalize_open_noun(prior["category"]) != category
    ):
        output["contextual_semantic_review"] = {
            "status": "confirmed_conflict_preserved",
            "prior": prior,
            "contextual_category": category,
            "event_id": evidence_event_id(event),
        }
        return output, "confirmed_conflict_preserved"

    whole = assess_whole_object_readiness(
        contract, minimum_confidence=minimum_confidence
    )
    output.update({
        "category": category,
        "description": str(identity.get("description") or ""),
        "attributes": list(identity.get("attributes") or [])[:8],
        "label_contract": copy.deepcopy(contract),
        "review_decision": "keep",
        "review_confidence": float(identity.get("confidence") or 0.0),
        "semantic_tier": "probable",
        "evidence_tier": "probable",
        "semantic_status": "contextual_dual_panel_reviewed",
        "semantic_gate_reason": (
            "contextual_dual_panel_agreement"
            if normalize_open_noun(prior["category"]) == category
            else "contextual_dual_panel_preferred"
        ),
        "contextual_semantic_review": {
            "status": "applied",
            "prior": prior,
            "contextual_category": category,
            "identity_confidence": float(identity.get("confidence") or 0.0),
            "whole_object_ready": bool(whole.get("ready")),
            "whole_object_assessment": whole,
            "event_id": evidence_event_id(event),
            "maximum_semantic_tier": "probable",
        },
    })
    return output, (
        "agreement"
        if normalize_open_noun(prior["category"]) == category
        else "contextual_preferred"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--context-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.90)
    args = parser.parse_args()
    started = time.perf_counter()
    base_path = args.base_catalog.expanduser().resolve(strict=True)
    review_path = args.context_review.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    base_rows = _catalog_rows(base_path)
    review_by_id = {int(row["id"]): row for row in _review_rows(review_path)}
    outcomes: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        merged, outcome = merge_object(
            base,
            review_by_id.get(int(base["id"])),
            minimum_confidence=float(args.minimum_confidence),
        )
        rows.append(merged)
        outcomes[outcome] += 1
    report = {
        "schema": "farm.contextual-semantic-merge.v1",
        "base_catalog": str(base_path),
        "base_catalog_sha256": _sha256(base_path),
        "context_review": str(review_path),
        "context_review_sha256": _sha256(review_path),
        "minimum_confidence": float(args.minimum_confidence),
        "object_count": len(rows),
        "outcome_counts": dict(sorted(outcomes.items())),
        "probable_count": sum(
            str(row.get("semantic_tier") or "") == "probable" for row in rows
        ),
        "confirmed_count": sum(
            str(row.get("semantic_tier") or "") == "confirmed" for row in rows
        ),
        "timing_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "semantic_context_catalog.json", rows)
    _atomic_json(output_dir / "contextual_merge_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
