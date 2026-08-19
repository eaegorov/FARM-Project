#!/usr/bin/env bash
set -e -o pipefail

# Keep bind-mounted run artifacts writable by the host GID while preserving a
# non-root image UID.  The orchestrator supplies the host GID explicitly.
umask 0002

# Source ROS 2
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash

# Activate venv
[ -f /home/scene_graph/.venv/bin/activate ] && source /home/scene_graph/.venv/bin/activate

# Build all ament packages found under ros/ against bind-mounted source.
# Output goes to /tmp/colcon_ws (writable by container user).
WORKSPACE=/home/scene_graph/scene_graph
COLCON_OUT=/tmp/colcon_ws
if [ -d "$WORKSPACE/ros" ]; then
    pkg_count=$(find "$WORKSPACE/ros" -name package.xml -not -path '*/.*' | wc -l)
    if [ "$pkg_count" -gt 0 ]; then
        echo "[entrypoint] Building ROS2 packages under ros/ (found ${pkg_count} package.xml)..."
        cd "$WORKSPACE"
        COLCON_LOG_PATH="$COLCON_OUT/log" colcon build --symlink-install \
            --base-paths ros \
            --build-base "$COLCON_OUT/build" \
            --install-base "$COLCON_OUT/install" 2>&1
    else
        echo "[entrypoint] WARNING: $WORKSPACE/ros has no package.xml — skipping colcon build."
    fi
else
    echo "[entrypoint] WARNING: $WORKSPACE/ros not found — skipping colcon build."
fi

# Source ROS2 workspace overlay
if [ -f "$COLCON_OUT/install/setup.bash" ]; then
    source "$COLCON_OUT/install/setup.bash"
else
    echo "[entrypoint] WARNING: $COLCON_OUT/install/setup.bash not found — ROS2 messages will be unavailable."
fi

normalize_run_permissions() {
    local requested_run_dir="${FARM_RUN_DIR:-/run}"
    [ -d "$requested_run_dir" ] || return 0
    local run_dir
    run_dir="$(cd -- "$requested_run_dir" 2>/dev/null && pwd -P)" || return 0
    # FARM_RUN_DIR must identify one run, never the filesystem root.  Resolve
    # it first so whitespace, trailing slashes, and symlinked mount paths do
    # not weaken the exact source-snapshot exclusion below.
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

if [ "$#" -eq 0 ]; then
    exec bash
else
    set +e
    "$@"
    status=$?
    set -e
    normalize_run_permissions
    exit "$status"
fi
