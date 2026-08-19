#!/usr/bin/env bash
set -euo pipefail

# Lightweight entrypoint for non-ROS post-processing stages in the main image.
umask 0002
set +e
python_executable="${FARM_PIPELINE_PYTHON:-/home/scene_graph/.venv/bin/python}"
"$python_executable" "$@"
status=$?
set -e

normalize_run_permissions() {
    local requested_run_dir="${FARM_RUN_DIR:-/run}"
    [ -d "$requested_run_dir" ] || return 0
    local run_dir
    run_dir="$(cd -- "$requested_run_dir" 2>/dev/null && pwd -P)" || return 0
    # Fail closed for an invalid root-wide target and preserve the immutable
    # source snapshot exactly, including its executable-mode metadata.
    [ -n "$run_dir" ] && [ "$run_dir" != "/" ] || return 0
    local source_snapshot_dir="$run_dir/config/source_snapshot"
    local uid
    uid="$(id -u)"
    find "$run_dir" -xdev \
        \( -path "$source_snapshot_dir" -o -path "$source_snapshot_dir/*" \) -prune -o \
        -uid "$uid" -type d -exec chmod g+rwx {} + 2>/dev/null || true
    find "$run_dir" -xdev \
        \( -path "$source_snapshot_dir" -o -path "$source_snapshot_dir/*" \) -prune -o \
        -uid "$uid" -type f -exec chmod g+rw {} + 2>/dev/null || true
}

normalize_run_permissions
exit "$status"
