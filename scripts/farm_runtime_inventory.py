#!/usr/bin/env python3
"""Capture or validate exact installed versions for a FARM runtime profile.

This is an inventory gate, not a wheel-hash lock generator.  It deliberately
accepts only exact ``name==version`` entries and never installs packages.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;]+)$")


class InventoryError(ValueError):
    """Raised for an ambiguous or non-exact runtime inventory."""


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_inventory(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = EXACT_REQUIREMENT.fullmatch(line)
        if match is None:
            raise InventoryError(
                f"{path}:{line_number}: expected an exact name==version entry"
            )
        name, version = canonical_name(match.group(1)), match.group(2)
        if name in entries and entries[name] != version:
            raise InventoryError(f"{path}:{line_number}: conflicting duplicate {name}")
        entries[name] = version
    if not entries:
        raise InventoryError(f"{path}: inventory has no exact package entries")
    return entries


def installed_versions(package_names: Iterable[str]) -> dict[str, str]:
    """Resolve versions using the active interpreter's import metadata order.

    A container can deliberately expose both a virtualenv and system
    site-packages. Enumerating every distribution makes those valid layered
    environments look ambiguous even though Python has a deterministic active
    package. Query only the requested contract names, which follows the
    interpreter's active ``sys.path`` resolution.
    """
    output: dict[str, str] = {}
    for raw_name in package_names:
        name = canonical_name(raw_name)
        try:
            output[name] = str(importlib.metadata.version(raw_name))
        except importlib.metadata.PackageNotFoundError:
            continue
    return output


def compare_inventory(expected: Mapping[str, str], installed: Mapping[str, str]) -> dict:
    missing = sorted(name for name in expected if name not in installed)
    mismatched = {
        name: {"expected": expected[name], "installed": installed[name]}
        for name in sorted(expected)
        if name in installed and installed[name] != expected[name]
    }
    matched = sorted(
        name for name in expected if name in installed and installed[name] == expected[name]
    )
    return {
        "status": "pass" if not missing and not mismatched else "fail",
        "expected_packages": len(expected),
        "matched_packages": len(matched),
        "missing": missing,
        "mismatched": mismatched,
        "matched": matched,
    }


def _write_atomic(path: Path, text: str) -> None:
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _base_report(action: str) -> dict:
    return {
        "schema": "farm.runtime-inventory.v1",
        "action": action,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _emit(report: dict, report_path: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_path is not None:
        _write_atomic(report_path, payload)
    print(payload, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture or validate an exact, non-installing FARM runtime inventory."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--lock", type=Path, required=True)
    validate.add_argument("--report", type=Path)
    capture = subparsers.add_parser("capture")
    capture.add_argument(
        "--packages-from", type=Path, required=True,
        help="Exact inventory whose package-name set should be captured",
    )
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--report", type=Path)
    capture.add_argument(
        "--force", action="store_true", help="Replace an existing output after validation"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "validate":
            expected = parse_inventory(args.lock.expanduser())
            installed = installed_versions(expected)
            report = _base_report("validate")
            report.update(compare_inventory(expected, installed))
            report["inventory"] = str(args.lock.expanduser().resolve())
            _emit(report, args.report)
            return 0 if report["status"] == "pass" else 1

        names = parse_inventory(args.packages_from.expanduser())
        installed = installed_versions(names)
        comparison = compare_inventory(names, installed)
        report = _base_report("capture")
        report.update(comparison)
        report["packages_from"] = str(args.packages_from.expanduser().resolve())
        report["output"] = str(args.output.expanduser().resolve(strict=False))
        if comparison["status"] != "pass":
            _emit(report, args.report)
            return 1
        output = args.output.expanduser().resolve(strict=False)
        if output.exists() and not args.force:
            raise InventoryError(f"refusing to replace existing output without --force: {output}")
        header = [
            "# FARM exact runtime inventory (versions only; no wheel hashes)",
            f"# Python {platform.python_version()}",
            f"# Platform {platform.platform()}",
            "",
        ]
        rows = [f"{name}=={installed[name]}" for name in sorted(names)]
        _write_atomic(output, "\n".join(header + rows) + "\n")
        _emit(report, args.report)
        return 0
    except (OSError, InventoryError) as exc:
        report = _base_report(args.action)
        report.update({"status": "error", "error": str(exc)})
        _emit(report, getattr(args, "report", None))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
