#!/usr/bin/env bash
set -e

# Use host X server (no fallback X inside container)
export K4A_LOG_LEVEL=${K4A_LOG_LEVEL:-5}
export K4A_ENABLE_LOG_TO_A_FILE=${K4A_ENABLE_LOG_TO_A_FILE:-1}
export K4A_LOG_FILE_NAME=${K4A_LOG_FILE_NAME:-/tmp/k4a.log}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libk4a1.4:${LD_LIBRARY_PATH}

export DISPLAY=${DISPLAY:-:0}
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
  echo "X display $DISPLAY not accessible. Allow access with 'xhost +local:' and set DISPLAY." >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

exec "$@"
