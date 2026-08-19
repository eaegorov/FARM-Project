#!/usr/bin/env python3
"""Run `hf download` with HF_TOKEN loaded from the workspace secret file."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("filenames", nargs="+")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument(
        "--secrets-file",
        type=Path,
        default=Path(os.getenv("FARM_SECRETS_FILE", workspace_root / "secrets.json")),
    )
    args = parser.parse_args()

    token = ""
    if args.secrets_file.is_file():
        value = json.loads(args.secrets_file.read_text(encoding="utf-8")).get("HF_TOKEN")
        if isinstance(value, str):
            token = value.strip()
    token = token or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is missing")

    hf_bin = workspace_root / "venv" / "bin" / "hf"
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    command = [
        str(hf_bin),
        "download",
        args.repo_id,
        *args.filenames,
        "--local-dir",
        str(args.local_dir.expanduser().resolve()),
    ]
    print(f"[hf] downloading {len(args.filenames)} requested file(s) from {args.repo_id} (token hidden)")
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
