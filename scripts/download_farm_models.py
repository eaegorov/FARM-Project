#!/usr/bin/env python3
"""Download FARM's Hugging Face models without printing credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPOS = {
    "siglip2": "google/siglip2-large-patch16-256",
    "text-embedding": "Qwen/Qwen3-Embedding-0.6B",
    "vl-embedding": "Qwen/Qwen3-VL-Embedding-2B",
    "caption-vl": "Qwen/Qwen3-VL-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    parser = argparse.ArgumentParser(description="Download the Hugging Face models used by FARM.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(REPOS),
        default=("siglip2", "text-embedding", "vl-embedding", "caption-vl"),
    )
    parser.add_argument(
        "--secrets-file",
        type=Path,
        default=Path(os.getenv("FARM_SECRETS_FILE", workspace_root / "secrets.json")),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.getenv("HF_HOME", workspace_root / ".cache" / "huggingface")),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_token(path: Path) -> str | None:
    if not path.is_file():
        return os.getenv("HF_TOKEN") or None
    raw = json.loads(path.read_text(encoding="utf-8"))
    token = raw.get("HF_TOKEN")
    return str(token).strip() if token else (os.getenv("HF_TOKEN") or None)


def main() -> int:
    args = parse_args()
    token = load_token(args.secrets_file)
    cache_dir = args.hf_home.expanduser().resolve() / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    for model in args.models:
        repo_id = REPOS[model]
        print(f"[models] {model}: {repo_id}", flush=True)
        kwargs = {
            "repo_id": repo_id,
            "cache_dir": str(cache_dir),
            "token": token,
            "local_files_only": args.local_files_only,
        }
        if model == "siglip2":
            kwargs["local_dir"] = str(repo_root / "models" / "siglip2-large-patch16-256")
            kwargs["allow_patterns"] = [
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ]
        snapshot_download(**kwargs)
    print("[models] requested snapshots are ready; HF token value was not logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
