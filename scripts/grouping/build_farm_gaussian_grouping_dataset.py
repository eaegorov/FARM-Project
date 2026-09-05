#!/usr/bin/env python3
"""Build an audited native Gaussian Grouping dataset from FARM tracker episodes."""

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

from farm_runtime.gaussian_grouping_dataset import build_dataset  # noqa: E402


def _tracker_run(value: str) -> tuple[str, Path]:
    episode_id, separator, raw_path = value.partition("=")
    if not separator or not episode_id or not raw_path:
        raise argparse.ArgumentTypeError("expected EPISODE_ID=/path/to/tracker/run")
    if Path(episode_id).name != episode_id:
        raise argparse.ArgumentTypeError("episode ID must be a safe basename")
    return episode_id, Path(raw_path)


def _tracker_runs_from_association(path: Path) -> dict[str, Path]:
    source = path.expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = (payload.get("inputs") or {}).get("signature_reports")
    if not isinstance(rows, list) or not rows:
        raise ValueError("association has no inputs.signature_reports")
    tracker_runs: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("association signature report is not an object")
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or Path(episode_id).name != episode_id:
            raise ValueError(f"unsafe association episode ID: {episode_id!r}")
        report = Path(str(row.get("path") or "")).expanduser().resolve(strict=True)
        if episode_id in tracker_runs:
            raise ValueError(f"duplicate association episode: {episode_id}")
        tracker_runs[episode_id] = report
    return tracker_runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument(
        "--tracker-run",
        action="append",
        type=_tracker_run,
        metavar="EPISODE_ID=PATH",
        help=(
            "Explicit run directory or detached 3D signature JSON, repeated once per episode. "
            "If omitted, exact reports come from association inputs.signature_reports[].path."
        ),
    )
    parser.add_argument("--association", required=True, type=Path)
    parser.add_argument("--colmap-model", required=True, type=Path)
    parser.add_argument("--source-gaussians", required=True, type=Path)
    parser.add_argument("--sparse-points-ply", type=Path)
    parser.add_argument(
        "--episode-id",
        action="append",
        help="Select a train/heldout episode subset; defaults to all materialized episodes.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tracker_runs: dict[str, Path] = {}
    if args.tracker_run:
        for episode_id, path in args.tracker_run:
            if episode_id in tracker_runs:
                raise ValueError(f"duplicate --tracker-run episode: {episode_id}")
            tracker_runs[episode_id] = path
    else:
        tracker_runs = _tracker_runs_from_association(args.association)
    report = build_dataset(
        episodes_root=args.episodes_root,
        tracker_runs=tracker_runs,
        association_path=args.association,
        colmap_model=args.colmap_model,
        source_gaussians=args.source_gaussians,
        output_root=args.output_root,
        sparse_points_ply=args.sparse_points_ply,
        selected_episode_ids=set(args.episode_id) if args.episode_id else None,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
