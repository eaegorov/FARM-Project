#!/usr/bin/env python3
"""Materialize a bounded Gaussian Grouping dataset from one exact anchor allowlist."""

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

from farm_runtime.gaussian_grouping_anchor_dataset import (
    build_anchor_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--colmap-model", required=True, type=Path)
    parser.add_argument("--source-gaussians", required=True, type=Path)
    parser.add_argument("--sparse-points-ply", type=Path)
    parser.add_argument("--gaussian-grouping-repo", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_anchor_dataset(
        episodes_root=args.episodes_root,
        allowlist_path=args.allowlist,
        colmap_model=args.colmap_model,
        source_gaussians=args.source_gaussians,
        sparse_points_ply=args.sparse_points_ply,
        gaussian_grouping_repo=args.gaussian_grouping_repo,
        output_root=args.output_root,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
