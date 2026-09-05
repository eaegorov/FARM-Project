#!/usr/bin/env python3
"""Build immutable train/heldout frames views over a rendered RGB-D union."""

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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.full_colmap_folds import build_full_colmap_fold_views  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--union-frames-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_full_colmap_fold_views(
        args.plan, args.union_frames_json, args.output_dir
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
