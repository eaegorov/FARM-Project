#!/usr/bin/env python3
"""Validate a FARM scene contract before any GPU service is started."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_pipeline.preflight import format_summary, run_preflight  # noqa: E402
from farm_pipeline.scene_config import SceneConfigError, load_scene_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only validation for a COLMAP + 3DGS FARM scene.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Scene YAML or JSON")
    parser.add_argument("--report", type=Path, help="Optional machine-readable JSON report")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report to stdout")
    parser.add_argument(
        "--strict", action="store_true",
        help="Return non-zero when checks contain warnings as well as errors",
    )
    return parser.parse_args()


def _write_atomic(path: Path, payload: str) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    try:
        config = load_scene_config(args.config)
    except (OSError, SceneConfigError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    report = run_preflight(config)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.report:
        _write_atomic(args.report, payload)
    print(payload if args.json else format_summary(report))
    if report.errors:
        return 1
    if args.strict and report.warnings:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
