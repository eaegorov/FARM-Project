#!/usr/bin/env python3
"""Build a proposal-only SAM3 concept vocabulary from Qwen label contracts.

The vocabulary is deliberately not a publication surface.  It may improve
segmentation recall, but every resulting instance still needs geometric
association, heldout QC and the independent semantic consensus gate.
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
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "farm.scene-proposal-vocabulary.v1"
GENERIC_TERMS = {
    "",
    "unknown",
    "unresolved object",
    "object",
    "item",
    "thing",
    "component",
    "equipment",
    "device",
    "assembly",
    "machine",
    "unit",
}


def normalize_term(value: object) -> str:
    value = re.sub(r"[^a-z0-9 -]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", value).strip(" -")


def select_prompt_rows(
    catalog: list[dict[str, Any]],
    *,
    minimum_confidence: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog:
        if not isinstance(row, dict):
            continue
        category = normalize_term(row.get("category"))
        contract = row.get("label_contract")
        if not isinstance(contract, dict):
            continue
        confidence = float(contract.get("confidence") or 0.0)
        eligible = (
            category not in GENERIC_TERMS
            and contract.get("contract_valid") is True
            and contract.get("decision") == "keep"
            and contract.get("context_sufficient") is True
            and contract.get("complete_bounded") is True
            and contract.get("head_noun_is_primitive") is False
            and contract.get("specificity") in {"specific_identity", "object_kind"}
            and confidence >= float(minimum_confidence)
        )
        if not eligible:
            continue
        grouped[category].append(
            {
                "object_id": int(row["id"]),
                "confidence": confidence,
                "semantic_tier": row.get("semantic_tier"),
                "semantic_status": row.get("semantic_status"),
                "identity_basis": contract.get("identity_basis"),
                "form_hypernym": normalize_term(contract.get("form_hypernym")),
            }
        )
    return [
        {"prompt": prompt, "sources": sorted(sources, key=lambda row: row["object_id"])}
        for prompt, sources in sorted(grouped.items())
    ]


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-confidence", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 <= float(args.minimum_confidence) <= 1.0:
        parser.error("--minimum-confidence must be in [0, 1]")
    source = args.semantic_catalog.expanduser().resolve(strict=True)
    catalog = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(catalog, list):
        raise TypeError("semantic catalog must be a JSON list")
    rows = select_prompt_rows(catalog, minimum_confidence=args.minimum_confidence)
    if not rows:
        raise RuntimeError("no Qwen-backed proposal prompts passed the label contract")
    prompts = [row["prompt"] for row in rows]
    text = "\n".join(prompts) + "\n"
    output = args.output_dir.expanduser().resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = {
        "schema": SCHEMA,
        "source_semantic_catalog": str(source),
        "source_semantic_catalog_sha256": source_hash,
        "policy": {
            "source_vlm": "Qwen via FARM semantic label contracts",
            "minimum_confidence": float(args.minimum_confidence),
            "generic_terms_excluded": sorted(GENERIC_TERMS),
            "role": "segmentation proposals only",
            "may_publish_labels": False,
            "requires_global_3d_association": True,
            "requires_heldout_qc": True,
            "requires_independent_semantic_consensus": True,
        },
        "prompt_count": len(prompts),
        "prompts": prompts,
        "prompt_rows": rows,
        "prompt_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    _atomic_write(output / "proposal_prompts.txt", text)
    _atomic_write(
        output / "proposal_prompts.json",
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({"output": str(output), "prompt_count": len(prompts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
