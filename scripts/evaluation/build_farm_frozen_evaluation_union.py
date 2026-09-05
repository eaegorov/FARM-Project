#!/usr/bin/env python3
"""Build one canonical train/heldout fold and frozen reference union."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for import_root in (str(SRC), str(ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from farm_runtime.frozen_evaluation_union import (  # noqa: E402
    HeldoutBundle,
    build_frozen_evaluation_union,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-frames-json",
        type=Path,
        required=True,
        help="frames.json containing every frame actually consumed by fitting",
    )
    parser.add_argument(
        "--heldout-bundle",
        action="append",
        nargs=6,
        metavar=(
            "FOLD_MANIFEST",
            "SAM3_RESULT",
            "YOLOE_STATE",
            "MASK_ROOT",
            "RGBD_RENDER_MANIFEST",
            "RGBD_UNION_FRAMES",
        ),
        required=True,
        help="repeat once per independently frozen heldout fold",
    )
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundles = [
        HeldoutBundle(
            fold_manifest=Path(values[0]),
            reference_result=Path(values[1]),
            mapping_state=Path(values[2]),
            mask_root=Path(values[3]),
            rgbd_render_manifest=Path(values[4]),
            rgbd_union_frames=Path(values[5]),
        )
        for values in args.heldout_bundle
    ]
    result = build_frozen_evaluation_union(
        args.fit_frames_json, bundles, args.source_ply, args.output_dir
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
