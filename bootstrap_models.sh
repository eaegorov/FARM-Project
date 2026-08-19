#!/usr/bin/env bash
set -euo pipefail

# Download and verify exactly the model revisions and file digests declared in
# configs/models/farm_models.v1.json. All options are owned by the Python
# downloader and forwarded unchanged; run with --help for the current contract.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_PYTHON=${FARM_MODEL_PYTHON:-python3}

if ! command -v "${MODEL_PYTHON}" >/dev/null 2>&1; then
  echo "Model bootstrap interpreter not found: ${MODEL_PYTHON}" >&2
  echo "Set FARM_MODEL_PYTHON to a Python environment containing huggingface_hub." >&2
  exit 127
fi

exec "${MODEL_PYTHON}" "${SCRIPT_DIR}/scripts/download_farm_models.py" "$@"
