#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Config
# ---------------------------
CONTAINER_NAME="rlbench"
IMAGE="rlbench:20.04-gpu"

DISPLAY_BASE=99
SCREEN_GEOM="1280x1024x24"
XVFB_LOG="/tmp/xvfb-rlbench.log"
XAUTH_FILE="/tmp/rlbench-xvfb.auth"

SCRIPTS_HOST="/home/yigit/projects/inverse/reversible_skills/scripts"
SCRIPTS_CONT="/workspace/scripts"
DATA_HOST="/home/yigit/projects/inverse/reversible_skills/data"
DATA_CONT="/workspace/data"
CONFIG_HOST="/home/yigit/projects/inverse/reversible_skills/config"
CONFIG_CONT="/workspace/config"
RUNS_HOST="/home/yigit/projects/inverse/reversible_skills/runs"
RUNS_CONT="/workspace/runs"
RESULTS_HOST="/home/yigit/projects/inverse/reversible_skills/results"
RESULTS_CONT="/workspace/results"

# ---------------------------
# Helpers
# ---------------------------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

pick_display() {
  local n
  for n in $(seq "${DISPLAY_BASE}" 199); do
    if [ ! -e "/tmp/.X11-unix/X${n}" ] && [ ! -e "/tmp/.X${n}-lock" ]; then
      echo ":${n}"
      return 0
    fi
  done
  echo "[error] Could not find a free X display number." >&2
  exit 1
}

make_xauth() {
  local cookie host
  host="$(hostname)"

  rm -f "${XAUTH_FILE}"
  touch "${XAUTH_FILE}"
  chmod 600 "${XAUTH_FILE}"

  cookie="$(mcookie)"
  xauth -f "${XAUTH_FILE}" add "${DISPLAY_NUM}" . "${cookie}"
  xauth -f "${XAUTH_FILE}" add "${host}/unix${DISPLAY_NUM}" . "${cookie}" || true
}

start_xvfb() {
  for c in Xvfb xauth mcookie; do
    if ! have_cmd "$c"; then
      echo "[error] Missing required host command: $c"
      exit 1
    fi
  done

  DISPLAY_NUM="$(pick_display)"
  export DISPLAY_NUM

  echo "[start] Using virtual display ${DISPLAY_NUM}"
  make_xauth

  echo "[start] Launching Xvfb ${DISPLAY_NUM} (log: ${XVFB_LOG})"
  Xvfb "${DISPLAY_NUM}" \
    -screen 0 "${SCREEN_GEOM}" \
    -nolisten tcp \
    -auth "${XAUTH_FILE}" \
    > "${XVFB_LOG}" 2>&1 &

  XVFB_PID=$!
  export XVFB_PID

  for _ in {1..50}; do
    if [ -S "/tmp/.X11-unix/X${DISPLAY_NUM#:}" ]; then
      echo "[ok] Xvfb socket ready."
      return
    fi
    sleep 0.1
  done

  echo "[error] Xvfb did not create an X socket. Check ${XVFB_LOG}"
  kill "${XVFB_PID}" >/dev/null 2>&1 || true
  exit 1
}

stop_old_container() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[start] Removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi
}

cleanup() {
  if [ -n "${XVFB_PID:-}" ] && kill -0 "${XVFB_PID}" >/dev/null 2>&1; then
    echo "[stop] Stopping Xvfb ${DISPLAY_NUM}"
    kill "${XVFB_PID}" >/dev/null 2>&1 || true
    wait "${XVFB_PID}" 2>/dev/null || true
  fi
  rm -f "${XAUTH_FILE}"
}

run_container() {
  stop_old_container

  echo "[start] Running container ${CONTAINER_NAME} without GPU access"
  docker run --rm -it \
    --net=host \
    --name="${CONTAINER_NAME}" \
    -e DISPLAY="${DISPLAY_NUM}" \
    -e XAUTHORITY="/tmp/.rlbench.xauth" \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${XAUTH_FILE}:/tmp/.rlbench.xauth:ro" \
    -v "${SCRIPTS_HOST}:${SCRIPTS_CONT}:rw" \
    -v "${DATA_HOST}:${DATA_CONT}:rw" \
    -v "${CONFIG_HOST}:${CONFIG_CONT}:rw" \
    -v "${RUNS_HOST}:${RUNS_CONT}:rw" \
    -v "${RESULTS_HOST}:${RESULTS_CONT}:rw" \
    "${IMAGE}"
}

# ---------------------------
# Main
# ---------------------------
trap cleanup EXIT
start_xvfb
run_container
