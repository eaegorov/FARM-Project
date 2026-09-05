#!/usr/bin/env python3
"""Evaluate frozen post-lift object masks on reserved full-COLMAP views."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from farm_runtime.frozen_heldout_qc import (  # noqa: E402
    evaluate_frozen_heldout_qc,
    sha256_file,
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish(output: Path, report: dict[str, Any]) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        report_path = temporary / "report.json"
        _write_json(report_path, report)
        for route, object_ids in report["routes"].items():
            queue = temporary / f"{route}_ids.txt"
            queue.write_text(
                "".join(f"{int(object_id)}\n" for object_id in object_ids),
                encoding="utf-8",
            )
        _write_json(
            temporary / "_RESULT.json",
            {
                "schema": "farm.frozen-heldout-qc.result.v1",
                "status": report["status"],
                "report": "report.json",
                "report_sha256": sha256_file(report_path),
            },
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--meters-per-scene-unit", type=float, required=True)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy = _json_object(args.policy_json) if args.policy_json is not None else None
    report = evaluate_frozen_heldout_qc(
        args.fold_manifest,
        args.candidate_manifest,
        args.evidence_manifest,
        meters_per_scene_unit=args.meters_per_scene_unit,
        policy=policy,
    )
    report["provenance"]["policy"] = (
        {
            "source": "explicit_json",
            "path": str(args.policy_json.expanduser().resolve(strict=True)),
            "sha256": sha256_file(args.policy_json.expanduser().resolve(strict=True)),
        }
        if args.policy_json is not None
        else {"source": "versioned_code_defaults", "path": None, "sha256": None}
    )
    _publish(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.expanduser().resolve()),
                "counts": report["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
