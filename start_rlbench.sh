#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Config: edit if needed
# ---------------------------
CONTAINER_NAME="rlbench"
IMAGE="rlbench:20.04-gpu"
DISPLAY_NUM=":99"
XORG_CONF="/tmp/xorg-rlbench.conf"
XORG_LOG="/tmp/Xorg.99.log"

SCRIPTS_HOST="/home/yigit/projects/inverse/reversible_skills/scripts"
SCRIPTS_CONT="/workspace/scripts"
DATA_HOST="/home/yigit/projects/inverse/reversible_skills/data"
DATA_CONT="/workspace/data"
CONFIG_HOST="/home/yigit/projects/inverse/reversible_skills/config"
CONFIG_CONT="/workspace/config"
RUNS_HOST="/home/yigit/projects/inverse/reversible_skills/runs"
RUNS_CONT="/workspace/runs"


# ---------------------------
# Helpers
# ---------------------------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

is_xorg_running() {
  pgrep -f "Xorg ${DISPLAY_NUM}" >/dev/null 2>&1
}

start_xorg() {
  if is_xorg_running; then
    echo "[start] Xorg ${DISPLAY_NUM} already running."
    return
  fi

  if ! have_cmd nvidia-xconfig; then
    echo "[error] nvidia-xconfig not found. Install nvidia-utils on host."
    exit 1
  fi

  echo "[start] Creating temporary Xorg config at ${XORG_CONF}"
  sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024 \
    --output-xconfig="${XORG_CONF}"

  echo "[start] Launching Xorg ${DISPLAY_NUM} (log: ${XORG_LOG})"
  sudo nohup Xorg "${DISPLAY_NUM}" -config "${XORG_CONF}" -noreset -nolisten tcp \
    > "${XORG_LOG}" 2>&1 & disown

  # Wait for X socket
  for i in {1..50}; do
    if [ -S "/tmp/.X11-unix/X${DISPLAY_NUM#:}" ]; then
      echo "[ok] X socket ready."
      return
    fi
    sleep 0.1
  done

  echo "[error] Xorg did not create an X socket. Check ${XORG_LOG}"
  exit 1
}

stop_old_container() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[start] Removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi
}

run_container() {
  echo "[start] Allowing root localuser to access X"
  xhost +SI:localuser:root >/dev/null

  stop_old_container

  echo "[start] Running container ${CONTAINER_NAME}"
  docker run --rm -it --gpus all --net=host --name="${CONTAINER_NAME}" \
    -e DISPLAY="${DISPLAY_NUM}" \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${SCRIPTS_HOST}:${SCRIPTS_CONT}:rw" \
    -v "${DATA_HOST}:${DATA_CONT}:rw" \
    -v "${CONFIG_HOST}:${CONFIG_CONT}:rw" \
    -v "${RUNS_HOST}:${RUNS_CONT}:rw" \
    "${IMAGE}"
}

# ---------------------------
# Main
# ---------------------------
start_xorg
run_container
