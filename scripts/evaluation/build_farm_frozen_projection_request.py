#!/usr/bin/env python3
"""Build a target-blind request for frozen heldout Gaussian projection."""

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

from farm_runtime.frozen_heldout_projection import (  # noqa: E402
    materialize_projection_request,
)
from farm_runtime.frozen_heldout_qc import sha256_file  # noqa: E402


def _publish(path: Path, payload: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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
        consumed = payload.get("provenance", {}).get("consumed_input_hashes", {})
        changed = [
            str(source)
            for source, expected in consumed.items()
            if sha256_file(Path(str(source)).resolve(strict=True)) != str(expected)
        ]
        if changed:
            raise RuntimeError(
                f"frozen inputs changed before request publish: {changed[:8]}"
            )
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--heldout-reference-result", type=Path, required=True)
    parser.add_argument("--rgbd-render-manifest", type=Path, required=True)
    parser.add_argument("--union-frames", type=Path, required=True)
    parser.add_argument("--meters-per-scene-unit", type=float, required=True)
    parser.add_argument("--alpha-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = materialize_projection_request(
        args.fold_manifest,
        args.candidate_manifest,
        args.source_ply,
        args.source_labels,
        args.candidate_ply,
        args.candidate_labels,
        args.heldout_reference_result,
        args.rgbd_render_manifest,
        args.union_frames,
        meters_per_scene_unit=args.meters_per_scene_unit,
        alpha_threshold=args.alpha_threshold,
    )
    _publish(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output.expanduser().resolve()),
                "request_sha256": sha256_file(args.output.expanduser().resolve()),
                "observations": len(payload["observations"]),
                "object_ids": [row["object_id"] for row in payload["objects"]],
                "target_masks_read": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
