#!/usr/bin/env python3
"""Materialize one fail-closed semantic cohort from pinned immutable evidence."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import time
from pathlib import Path

from farm_runtime.semantic_evidence_resolver import resolve_plan, sha256_file


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    plan_path = args.plan.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    result = resolve_plan(plan, plan_path=plan_path)
    catalog_path = output_dir / "resolved_semantic_catalog.json"
    _atomic_json(catalog_path, result.pop("resolved_catalog"))
    result.update({
        "resolution_plan": str(plan_path),
        "resolution_plan_sha256": sha256_file(plan_path),
        "resolved_catalog": str(catalog_path),
        "resolved_catalog_sha256": sha256_file(catalog_path),
        "timing_seconds": time.perf_counter() - started,
    })
    _atomic_json(output_dir / "resolution_manifest.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
