#!/usr/bin/env python3
"""Replay frozen V11 inventory and verify F0 source/input hashes in Docker."""
from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
from pathlib import Path

from farm_runtime.quality_baseline import (
    describe_file, json_digest, sha256_file, v11_inventory, verify_source_snapshot, write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-files", type=Path)
    parser.add_argument("--input-list", type=Path)
    parser.add_argument("--expected-inventory", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite baseline audit")
    if bool(args.source_root) != bool(args.source_files):
        raise ValueError("source root and source files are required together")
    inventory = v11_inventory(args.experiment)
    report = {"schema": "farm.quality-baseline-audit.v1", "inventory": inventory,
              "inventory_sha256": json_digest(inventory), "checks": {}, "status": "PASS"}
    if args.source_root:
        report["checks"]["source"] = verify_source_snapshot(
            args.source_root, json.loads(args.source_files.read_text()))
    if args.expected_inventory:
        expected = json.loads(args.expected_inventory.read_text())
        report["checks"]["inventory_replay"] = {
            "status": "PASS" if expected["inventory_sha256"] == report["inventory_sha256"] else "FAIL",
            "expected_sha256": expected["inventory_sha256"],
        }
    if args.input_list:
        rows = json.loads(args.input_list.read_text())
        checked = []
        for row in rows:
            actual = describe_file(Path(row["path"]))
            actual["role"] = row["role"]
            if row.get("expected_sha256"):
                actual["expected_sha256"] = row["expected_sha256"]
                actual["matches_expected"] = row["expected_sha256"] == actual["sha256"]
            checked.append(actual)
        report["inputs"] = checked
        report["checks"]["inputs"] = {"status": "PASS" if all(
            x.get("matches_expected", True) for x in checked) else "FAIL", "count": len(checked)}
    if any(v["status"] != "PASS" for v in report["checks"].values()):
        report["status"] = "FAIL"
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "inventory": inventory}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
