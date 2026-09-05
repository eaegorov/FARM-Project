#!/usr/bin/env python3
"""Materialize a SHA-bound train-only candidate for frozen heldout QC."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from farm_runtime.frozen_heldout_evidence import (  # noqa: E402
    materialize_frozen_candidate,
)
from farm_runtime.frozen_heldout_qc import sha256_file  # noqa: E402


def _object_ids(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    values: list[int] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise ValueError(
                f"invalid object ID at {resolved}:{line_number}: {text!r}"
            ) from exc
    if not values:
        raise ValueError("object ID file is empty")
    return values


def _publish(path: Path, payload: object) -> None:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _revalidate(payload: dict[str, object]) -> None:
    provenance = payload.get("provenance")
    consumed = (
        provenance.get("consumed_input_hashes")
        if isinstance(provenance, dict)
        else None
    )
    if not isinstance(consumed, dict) or not consumed:
        raise ValueError("candidate materialization lacks consumed input hashes")
    changed = [
        path
        for path, expected in consumed.items()
        if sha256_file(Path(path).resolve(strict=True)) != expected
    ]
    if changed:
        raise RuntimeError(f"frozen inputs changed before publish: {changed[:8]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--upstream-lift-result", type=Path, required=True)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--meters-per-scene-unit", type=float, required=True)
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = materialize_frozen_candidate(
        args.fold_manifest,
        args.upstream_lift_result,
        args.source_ply,
        args.source_labels,
        args.candidate_ply,
        args.candidate_labels,
        meters_per_scene_unit=args.meters_per_scene_unit,
        object_ids=_object_ids(args.object_id_file),
    )
    _revalidate(payload)
    _publish(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output.expanduser().resolve()),
                "object_ids": [row["object_id"] for row in payload["objects"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
