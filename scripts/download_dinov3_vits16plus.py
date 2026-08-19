#!/usr/bin/env python3
"""Securely download FARM's gated paper-grade DINOv3-S+ backbone via `hf`."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    secrets_path = Path(os.getenv("FARM_SECRETS_FILE", workspace_root / "secrets.json"))
    token = ""
    if secrets_path.is_file():
        raw = json.loads(secrets_path.read_text(encoding="utf-8"))
        value = raw.get("HF_TOKEN")
        if isinstance(value, str):
            token = value.strip()
    token = token or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is missing")

    hf_bin = workspace_root / "venv" / "bin" / "hf"
    if not hf_bin.is_file():
        raise SystemExit(f"hf CLI not found: {hf_bin}")
    local_dir = repo_root / "models" / "dinov3-vits16plus"
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    command = [
        str(hf_bin),
        "download",
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "--local-dir",
        str(local_dir),
        "--include",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "README.md",
        "LICENSE.md",
    ]
    print(f"[models] requesting gated DINOv3-S+ into {local_dir} (token hidden)", flush=True)
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        print(
            "[models] DINOv3-S+ unavailable. Accept the Meta DINOv3 license for this HF account, then rerun.",
            flush=True,
        )
        return completed.returncode
    print("[models] DINOv3-S+ ready; HF token value was not logged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
