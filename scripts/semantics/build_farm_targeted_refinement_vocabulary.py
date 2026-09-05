#!/usr/bin/env python3
"""Build a bounded YOLOE vocabulary from audited whole-object evidence.

The output is deliberately target-only: one evidence-backed noun per selected
canonical object. YOLOE produces the only detector bbox/mask seed. SAM3 may
propose one A/B alternative later, so this stage does not create a redundant
bbox-refinement cascade.
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
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


GENERIC_TERMS = {
    "object",
    "unknown",
    "unresolved object",
    "component",
    "equipment",
    "device",
    "assembly",
}


def normalize_term(value: object) -> str:
    term = re.sub(r"[^a-z0-9 -]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", term).strip(" -")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def load_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "farm.whole-object-evidence-audit.v1":
        raise ValueError(f"unsupported whole-object audit: {path}")
    if not isinstance(payload.get("objects"), list):
        raise TypeError(f"whole-object audit lacks objects: {path}")
    return payload


def select_targets(
    payload: dict[str, Any],
    *,
    statuses: set[str],
    minimum_identity_confidence: float,
    maximum_objects: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in payload["objects"]:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "")
        category = normalize_term(raw.get("category"))
        confidence = float(raw.get("identity_confidence") or 0.0)
        if (
            status not in statuses
            or not category
            or category in GENERIC_TERMS
            or confidence < minimum_identity_confidence
        ):
            continue
        rows.append(
            {
                "object_id": int(raw["object_id"]),
                "category": category,
                "status": status,
                "identity_confidence": confidence,
                "refinement_signals": sorted(
                    str(value)
                    for value in raw.get("refinement_signals") or []
                    if str(value)
                ),
                "published_in_baseline": bool(raw.get("published_in_baseline")),
                "prompt_terms": [category],
            }
        )
    rows.sort(
        key=lambda row: (
            row["status"] == "refine",
            len(row["refinement_signals"]),
            row["identity_confidence"],
            row["published_in_baseline"],
            -row["object_id"],
        ),
        reverse=True,
    )
    return rows[:maximum_objects]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--status",
        action="append",
        choices=("refine", "needs_independent_evidence"),
        help="Repeat to select multiple audited statuses; defaults to refine only.",
    )
    parser.add_argument("--minimum-identity-confidence", type=float, default=0.90)
    parser.add_argument("--maximum-objects", type=int, default=24)
    args = parser.parse_args()
    if args.maximum_objects < 1:
        parser.error("--maximum-objects must be positive")
    started = time.perf_counter()
    audit_path = args.audit.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    statuses = set(args.status or ["refine"])
    audit = load_audit(audit_path)
    targets = select_targets(
        audit,
        statuses=statuses,
        minimum_identity_confidence=float(args.minimum_identity_confidence),
        maximum_objects=int(args.maximum_objects),
    )
    if not targets:
        raise RuntimeError("whole-object audit produced no safe refinement targets")
    object_ids = [int(row["object_id"]) for row in targets]
    vocabulary = sorted(
        {
            term
            for row in targets
            for term in row["prompt_terms"]
            if term and term not in GENERIC_TERMS
        }
    )
    vocabulary_text = "\n".join(vocabulary) + "\n"
    ids_text = "\n".join(str(value) for value in object_ids) + "\n"
    atomic_text(output / "yoloe_vocabulary.txt", vocabulary_text)
    atomic_text(output / "object_ids.txt", ids_text)
    atomic_text(output / "whole_object_expansion_ids.txt", ids_text)
    artifact = {
        "schema": "farm.targeted-refinement-vocabulary.v1",
        "created_unix_s": time.time(),
        "source_audit": str(audit_path),
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "policy": {
            "qwen_identity_evidence_required": True,
            "selected_statuses": sorted(statuses),
            "minimum_identity_confidence": float(args.minimum_identity_confidence),
            "maximum_objects": int(args.maximum_objects),
            "one_detector_bbox_mask_seed": True,
            "sam3_single_ab_candidate_only": True,
            "generic_terms_excluded": sorted(GENERIC_TERMS),
        },
        "target_count": len(targets),
        "vocabulary_count": len(vocabulary),
        "status_counts": dict(Counter(row["status"] for row in targets)),
        "object_ids": object_ids,
        "vocabulary": vocabulary,
        "vocabulary_sha256": sha256_text(vocabulary_text),
        "object_ids_sha256": sha256_text(ids_text),
        "targets": targets,
        "timing_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "targeted_refinement.json", artifact)
    print(json.dumps({
        "schema": artifact["schema"],
        "target_count": len(targets),
        "vocabulary_count": len(vocabulary),
        "object_ids": object_ids,
        "output_dir": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
