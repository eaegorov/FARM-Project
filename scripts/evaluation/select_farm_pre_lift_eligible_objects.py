#!/usr/bin/env python3
"""Select exact object IDs eligible for lifting, without claiming lift quality."""

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
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.pre_lift_eligibility import (  # noqa: E402
    build_pre_lift_selection,
    publish_pre_lift_selection,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-refinement-result", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument(
        "--heldout-reference-result",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every independently produced heldout-reference top-up.",
    )
    parser.add_argument(
        "--minimum-independent-physical-timestamps",
        type=int,
        default=2,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if int(args.minimum_independent_physical_timestamps) < 1:
        parser.error("--minimum-independent-physical-timestamps must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selection = build_pre_lift_selection(
        args.train_refinement_result,
        args.geometry_audit,
        args.heldout_reference_result,
        minimum_independent_physical_timestamps=(
            args.minimum_independent_physical_timestamps
        ),
    )
    published = publish_pre_lift_selection(args.output_dir, selection)
    print(json.dumps(published, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
