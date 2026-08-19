#!/usr/bin/env bash
set -e -o pipefail

# Load HF_TOKEN after container creation so it is absent from the image,
# Compose environment metadata and build logs.
if [ -z "${HF_TOKEN:-}" ] && [ -n "${FARM_SECRETS_FILE:-}" ] && [ -r "$FARM_SECRETS_FILE" ]; then
    IFS= read -r HF_TOKEN < <(
        python3 -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")).get("HF_TOKEN", ""); print(value if isinstance(value, str) else "")' "$FARM_SECRETS_FILE"
    )
    if [ -n "$HF_TOKEN" ]; then
        export HF_TOKEN
        echo "[factory-entrypoint] HF_TOKEN loaded from mounted secret (value hidden)."
    else
        echo "[factory-entrypoint] WARNING: mounted secret has no non-empty HF_TOKEN."
    fi
fi

exec /entrypoint.sh "$@"
