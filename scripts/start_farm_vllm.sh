#!/usr/bin/env bash
# Generic, manifest-pinned, single-resident vLLM lifecycle for FARM stages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

MANIFEST="${FARM_MODEL_MANIFEST:-${PROJECT_ROOT}/configs/models/farm_models.v1.json}"
SECRETS_FILE="${FARM_SECRETS_FILE:-${WORKSPACE_ROOT}/secrets.json}"
GPU="${FARM_GPU:-0}"
SCOPE="${FARM_VLLM_SCOPE:-default}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT_OVERRIDE="${FARM_VLLM_PORT:-}"
TIMEOUT="${FARM_VLLM_TIMEOUT:-900}"
PYTHON_BIN="${FARM_HOST_PYTHON:-python3}"

LABEL_OWNER="com.goldengait.farm.vllm"
LABEL_SCOPE="com.goldengait.farm.vllm.scope"
LABEL_SERVICE="com.goldengait.farm.vllm.service"

usage() {
    cat >&2 <<'EOF'
Usage:
  scripts/start_farm_vllm.sh start|switch SERVICE [options]
  scripts/start_farm_vllm.sh status [options]
  scripts/start_farm_vllm.sh stop [SERVICE|all] [options]

Services are defined by configs/models/farm_models.v1.json. Exactly one vLLM
worker is resident per scope. Options:
  --gpu INDEX_OR_UUID   explicit GPU assignment (default: FARM_GPU or 0)
  --scope NAME          isolated lifecycle scope (default: default)
  --manifest PATH       pinned model/runtime manifest
  --secrets-file PATH   mounted read-only; values are never logged
  --host HOST           vLLM bind host (default: 127.0.0.1)
  --port PORT           validated host port override (default: manifest)
  --timeout SECONDS     readiness timeout (default: 900)
EOF
    exit 2
}

[[ $# -ge 1 ]] || usage
ACTION="$1"
shift
TARGET=""
case "${ACTION}" in
    start|switch)
        [[ $# -ge 1 && "$1" != --* ]] || usage
        TARGET="$1"
        shift
        ;;
    stop)
        if [[ $# -ge 1 && "$1" != --* ]]; then
            TARGET="$1"
            shift
        else
            TARGET="all"
        fi
        ;;
    status)
        ;;
    *)
        usage
        ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            [[ $# -ge 2 ]] || usage
            GPU="$2"
            shift 2
            ;;
        --scope)
            [[ $# -ge 2 ]] || usage
            SCOPE="$2"
            shift 2
            ;;
        --manifest)
            [[ $# -ge 2 ]] || usage
            MANIFEST="$2"
            shift 2
            ;;
        --secrets-file)
            [[ $# -ge 2 ]] || usage
            SECRETS_FILE="$2"
            shift 2
            ;;
        --host)
            [[ $# -ge 2 ]] || usage
            HOST="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || usage
            PORT_OVERRIDE="$2"
            shift 2
            ;;
        --timeout)
            [[ $# -ge 2 ]] || usage
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ ! "${SCOPE}" =~ ^[a-z0-9][a-z0-9_.-]{0,39}$ ]]; then
    echo "[farm-vllm] invalid scope: ${SCOPE}" >&2
    exit 2
fi
if [[ -n "${TARGET}" && "${TARGET}" != "all" && ! "${TARGET}" =~ ^[a-z0-9][a-z0-9_.-]*$ ]]; then
    echo "[farm-vllm] invalid service: ${TARGET}" >&2
    exit 2
fi
if [[ ! "${TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[farm-vllm] timeout must be a positive integer" >&2
    exit 2
fi
if [[ ! "${HOST}" =~ ^[A-Za-z0-9.:-]+$ ]]; then
    echo "[farm-vllm] invalid host" >&2
    exit 2
fi
if [[ -n "${PORT_OVERRIDE}" ]] && {
    [[ ! "${PORT_OVERRIDE}" =~ ^[0-9]+$ ]] ||
    (( PORT_OVERRIDE < 1024 || PORT_OVERRIDE > 65535 ))
}; then
    echo "[farm-vllm] port must be an integer in [1024, 65535]" >&2
    exit 2
fi

command -v docker >/dev/null || { echo "[farm-vllm] docker is required" >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "[farm-vllm] ${PYTHON_BIN} is required" >&2; exit 2; }

scope_filter=(
    --filter "label=${LABEL_OWNER}=true"
    --filter "label=${LABEL_SCOPE}=${SCOPE}"
)

stop_scoped() {
    local service="${1:-all}"
    local filters=("${scope_filter[@]}")
    if [[ "${service}" != "all" ]]; then
        filters+=(--filter "label=${LABEL_SERVICE}=${service}")
    fi
    local ids=()
    mapfile -t ids < <(docker ps -aq "${filters[@]}")
    if (( ${#ids[@]} > 0 )); then
        docker rm -f "${ids[@]}" >/dev/null
        echo "[farm-vllm] stopped scope=${SCOPE} service=${service}"
    fi
}

show_status() {
    docker ps -a "${scope_filter[@]}" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
}

if [[ "${ACTION}" == "stop" ]]; then
    stop_scoped "${TARGET}"
    exit 0
fi
if [[ "${ACTION}" == "status" ]]; then
    show_status
    exit 0
fi

# A switch intentionally frees only this scope before measuring available VRAM.
# Unlabelled containers and other scopes are never selected by stop_scoped.
stop_scoped all

REPORT_FILE="$(mktemp -t farm-resource-preflight.XXXXXX.json)"
MODEL_RESPONSE="$(mktemp -t farm-vllm-models.XXXXXX.json)"
cleanup_temporary() {
    rm -f "${REPORT_FILE}" "${MODEL_RESPONSE}"
}
trap cleanup_temporary EXIT

preflight_args=(
    "${PROJECT_ROOT}/scripts/farm_resource_preflight.py"
    --manifest "${MANIFEST}"
    --gpu "${GPU}"
    --services "${TARGET}"
    --report "${REPORT_FILE}"
)
if [[ -n "${SECRETS_FILE}" ]]; then
    preflight_args+=(--secrets-file "${SECRETS_FILE}")
fi
"${PYTHON_BIN}" "${preflight_args[@]}"

service_meta=()
mapfile -t service_meta < <(
    "${PYTHON_BIN}" -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
service = payload["services"][sys.argv[2]]
memory = service["memory"]
current_utilization = float(memory["gpu_memory_utilization"])
total_gib = float(memory["gpu_total_gib"])
free_gib = float(memory["gpu_free_gib"])
reserve_gib = float(memory["reserve_free_gib"])
retry_allocation_gib = max(
    float(memory["planned_allocated_gib"]) + 6.0,
    float(memory["target_allocated_gib"]) * 1.25,
)
retry_utilization = min(
    float(memory["utilization_max"]), retry_allocation_gib / total_gib
)
if retry_utilization * total_gib + reserve_gib > free_gib:
    retry_utilization = current_utilization
retry_utilization = max(current_utilization, retry_utilization)
values = [
    service["runtime_image_id"], service["served_model_name"], str(service["port"]),
    service["model_mount_source"], service["model_mount_target"],
    service["container_model_path"], str(memory["gpu_memory_utilization"]),
    f"{retry_utilization:.4f}",
    *service["vllm_args"],
]
for value in values:
    value = str(value)
    if any(character in value for character in "\r\n\0"):
        raise SystemExit("unsafe control character in resource report")
    print(value)
' "${REPORT_FILE}" "${TARGET}"
)
if (( ${#service_meta[@]} < 8 )); then
    echo "[farm-vllm] incomplete service resolution" >&2
    exit 2
fi

IMAGE="${service_meta[0]}"
SERVED_MODEL="${service_meta[1]}"
PORT="${PORT_OVERRIDE:-${service_meta[2]}}"
MODEL_MOUNT_SOURCE="${service_meta[3]}"
MODEL_MOUNT_TARGET="${service_meta[4]}"
CONTAINER_MODEL="${service_meta[5]}"
GPU_UTILIZATION="${service_meta[6]}"
FALLBACK_GPU_UTILIZATION="${service_meta[7]}"
VLLM_ARGS=("${service_meta[@]:8}")
CONTAINER_NAME="farm-vllm-${SCOPE}-${TARGET}"

if ! "${PYTHON_BIN}" -c '
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as sock:
    sock.bind((host, port))
' "${HOST}" "${PORT}"; then
    echo "[farm-vllm] requested endpoint is already in use: ${HOST}:${PORT}" >&2
    exit 2
fi

docker_args=(
    -d
    --name "${CONTAINER_NAME}"
    --label "${LABEL_OWNER}=true"
    --label "${LABEL_SCOPE}=${SCOPE}"
    --label "${LABEL_SERVICE}=${TARGET}"
    --gpus "device=${GPU}"
    --ipc host
    --network host
    -e XDG_CACHE_HOME=/tmp/farm-cache
    -e VLLM_CACHE_ROOT=/tmp/farm-cache/vllm
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/farm-cache/torchinductor
    -e TRITON_CACHE_DIR=/tmp/farm-cache/triton
    -e FLASHINFER_WORKSPACE_BASE=/tmp/farm-cache
    -e HF_HOME="${MODEL_MOUNT_TARGET}"
    -e HUGGINGFACE_HUB_CACHE="${MODEL_MOUNT_TARGET}/hub"
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
    -v "${MODEL_MOUNT_SOURCE}:${MODEL_MOUNT_TARGET}:ro"
    -v "${PROJECT_ROOT}/docker/vllm-entrypoint.sh:/vllm-entrypoint.sh:ro"
    --entrypoint /bin/bash
)
if [[ -f "${SECRETS_FILE}" ]]; then
    docker_args+=(
        -e FARM_SECRETS_FILE=/run/secrets/farm-secrets.json
        -v "${SECRETS_FILE}:/run/secrets/farm-secrets.json:ro"
    )
fi

launch_worker() {
    docker run "${docker_args[@]}" "${IMAGE}" \
        /vllm-entrypoint.sh vllm serve "${CONTAINER_MODEL}" \
        --host "${HOST}" --port "${PORT}" --served-model-name "${SERVED_MODEL}" \
        --gpu-memory-utilization "${GPU_UTILIZATION}" \
        "${VLLM_ARGS[@]}" >/dev/null
}

launch_worker

READY_HOST="${HOST}"
if [[ "${READY_HOST}" == "0.0.0.0" || "${READY_HOST}" == "::" ]]; then
    READY_HOST="127.0.0.1"
fi
deadline=$((SECONDS + TIMEOUT))
memory_retry_used=0
while (( SECONDS < deadline )); do
    if "${PYTHON_BIN}" -c '
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    payload = json.load(response)
ids = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
raise SystemExit(0 if sys.argv[2] in ids else 1)
' "http://${READY_HOST}:${PORT}/v1/models" "${SERVED_MODEL}" \
        >"${MODEL_RESPONSE}" 2>/dev/null; then
        echo "[farm-vllm] ready scope=${SCOPE} service=${TARGET} model=${SERVED_MODEL} port=${PORT} gpu=${GPU} utilization=${GPU_UTILIZATION}"
        exit 0
    fi
    if [[ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]]; then
        worker_logs="$(docker logs --tail 120 "${CONTAINER_NAME}" 2>&1 || true)"
        if (( memory_retry_used == 0 )) \
            && [[ "${FALLBACK_GPU_UTILIZATION}" != "${GPU_UTILIZATION}" ]] \
            && [[ "${worker_logs}" == *"No available memory for the cache blocks"* ]]; then
            docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
            echo "[farm-vllm] bounded memory retry scope=${SCOPE} service=${TARGET} utilization=${GPU_UTILIZATION}->${FALLBACK_GPU_UTILIZATION}" >&2
            GPU_UTILIZATION="${FALLBACK_GPU_UTILIZATION}"
            memory_retry_used=1
            launch_worker
            deadline=$((SECONDS + TIMEOUT))
            continue
        fi
        echo "[farm-vllm] worker exited before readiness" >&2
        printf '%s\n' "${worker_logs}" >&2
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        exit 1
    fi
    sleep 2
done

echo "[farm-vllm] timeout waiting for exact served model ${SERVED_MODEL}" >&2
docker logs --tail 80 "${CONTAINER_NAME}" >&2 2>/dev/null || true
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
exit 1
