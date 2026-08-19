#!/usr/bin/env python3
"""Validate FARM GPU, pinned models, secrets, and Docker runtimes."""

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

from farm_pipeline.resources import (  # noqa: E402
    ResourceConfigError,
    format_resource_summary,
    load_model_manifest,
    run_resource_preflight,
)


DEFAULT_MANIFEST = ROOT / "configs/models/farm_models.v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only resource/model readiness validation for FARM.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Versioned JSON model manifest",
    )
    parser.add_argument(
        "--gpu", default=os.getenv("FARM_GPU", "0"),
        help="Explicit NVIDIA GPU index or full UUID (default: FARM_GPU or 0)",
    )
    parser.add_argument(
        "--services", nargs="+",
        help="Service names to validate; default validates every service",
    )
    parser.add_argument(
        "--secrets-file", type=Path,
        help="Optional JSON secret file; values are never included in reports",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Also checksum pipeline models and validate every runtime image",
    )
    parser.add_argument(
        "--verify-full-hashes", action="store_true",
        help="Read and hash large HF blobs instead of trusting content-addressed names",
    )
    parser.add_argument(
        "--skip-docker-image-check", action="store_true",
        help="Skip exact Docker image-ID checks (reported as a warning)",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--json", action="store_true", help="Print complete JSON report")
    parser.add_argument(
        "--strict", action="store_true",
        help="Return non-zero for warnings as well as errors",
    )
    return parser.parse_args()


def _write_atomic(path: Path, payload: str) -> None:
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    try:
        manifest = load_model_manifest(args.manifest)
        report = run_resource_preflight(
            manifest,
            gpu_identifier=args.gpu,
            services=args.services,
            secrets_file=args.secrets_file,
            include_pipeline_models=args.all_models,
            include_all_runtimes=args.all_models,
            verify_full_hashes=args.verify_full_hashes,
            check_docker_images=not args.skip_docker_image_check,
        )
    except (OSError, ResourceConfigError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.report:
        _write_atomic(args.report, payload)
    print(payload if args.json else format_resource_summary(report))
    if report.errors:
        return 1
    if args.strict and report.warnings:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
