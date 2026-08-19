#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path[:] = [entry for entry in sys.path if entry != str(SRC_ROOT)]
sys.path.insert(0, str(SRC_ROOT))

# Some repository tools deliberately put ``scripts/`` on sys.path.  If this
# wrapper is imported as ``farm_pipeline`` before the real package, make it a
# package-compatible shim whose submodules resolve to ``src``.
if __name__ == "farm_pipeline":
    __path__ = [str(SRC_ROOT / "farm_pipeline")]

from farm_runtime.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
