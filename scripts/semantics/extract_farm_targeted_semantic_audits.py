#!/usr/bin/env python3
"""Extract hash-pinned exact-object catalogs from immutable Qwen batches."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import json
from pathlib import Path

from farm_runtime.semantic_evidence_resolver import REVIEW_SCHEMA, sha256_file


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "farm.targeted-semantic-extraction-plan.v1":
        raise ValueError("extraction_plan_schema_mismatch")
    outputs = []
    for spec in plan.get("sources") or []:
        source = Path(str(spec["path"])).resolve(strict=True)
        if sha256_file(source) != spec["sha256"]:
            raise ValueError(f"source_sha256_mismatch:{source}")
        payload = json.loads(source.read_text())
        if payload.get("schema") != REVIEW_SCHEMA:
            raise ValueError(f"source_schema_mismatch:{source}")
        rows = payload.get("objects") or []
        by_id = {int(row["id"]): row for row in rows}
        if sorted(by_id) != sorted(map(int, spec["source_object_ids"])):
            raise ValueError(f"source_cohort_mismatch:{source}")
        for object_id in map(int, spec["extract_object_ids"]):
            if object_id not in by_id:
                raise ValueError(f"extract_object_missing:{object_id}")
            result = copy.deepcopy(payload)
            result["objects"] = [copy.deepcopy(by_id[object_id])]
            result["object_count"] = 1
            result["targeted_extraction"] = {
                "source_catalog": str(source),
                "source_catalog_sha256": spec["sha256"],
                "object_id": object_id,
            }
            output = args.output_dir / f"object_{object_id:06d}" / "reviewed_robust_objects.json"
            _write(output, result)
            outputs.append({
                "object_id": object_id,
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
                "source_catalog_sha256": spec["sha256"],
            })
    manifest = {
        "schema": "farm.targeted-semantic-extraction-set.v1",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "objects": sorted(outputs, key=lambda row: row["object_id"]),
    }
    _write(args.output_dir / "extraction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
