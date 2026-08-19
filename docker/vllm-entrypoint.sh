#!/usr/bin/env bash
set -euo pipefail

# Load the Hugging Face token at runtime.  The value never appears in the
# Docker image, command line, compose metadata, or logs.
if [[ -z "${HF_TOKEN:-}" && -n "${FARM_SECRETS_FILE:-}" && -r "${FARM_SECRETS_FILE}" ]]; then
    IFS= read -r HF_TOKEN < <(
        /usr/bin/python3 -c \
            'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8")).get("HF_TOKEN", ""); print(v if isinstance(v, str) else "")' \
            "${FARM_SECRETS_FILE}"
    )
    if [[ -n "${HF_TOKEN}" ]]; then
        export HF_TOKEN
        echo "[vllm-entrypoint] HF_TOKEN loaded from mounted secret (value hidden)."
    fi
fi

source /home/scene_graph/.venv/bin/activate
exec "$@"
