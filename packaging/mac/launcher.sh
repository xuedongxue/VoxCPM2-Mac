#!/bin/bash
# VoxCPM2.app entry point — launched as Contents/MacOS/VoxCPM2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

export VOXCPM_PACKAGED=1
export VOXCPM_HOME="${HOME}/Library/Application Support/VoxCPM2"
export HF_HOME="${VOXCPM_HOME}/cache/huggingface"
export MODELSCOPE_CACHE="${VOXCPM_HOME}/cache/modelscope"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

mkdir -p "${VOXCPM_HOME}" "${HF_HOME}" "${MODELSCOPE_CACHE}"

LOG_FILE="${VOXCPM_HOME}/app.log"
{
  echo "=== VoxCPM2 launch $(date '+%Y-%m-%d %H:%M:%S %z') ==="
  echo "Bundle: ${CONTENTS_DIR}"
} >>"${LOG_FILE}"

export PATH="${RESOURCES_DIR}/bin:${PATH}"

PYTHON="${RESOURCES_DIR}/runtime/bin/python"
APP="${RESOURCES_DIR}/app/app.py"

if [[ ! -x "${PYTHON}" ]]; then
  msg="VoxCPM2: bundled Python not found at ${PYTHON}"
  echo "${msg}" | tee -a "${LOG_FILE}" >&2
  osascript -e "display alert \"VoxCPM2 failed to start\" message \"${msg}\"" 2>/dev/null || true
  exit 1
fi

if [[ ! -f "${APP}" ]]; then
  msg="VoxCPM2: application not found at ${APP}"
  echo "${msg}" | tee -a "${LOG_FILE}" >&2
  osascript -e "display alert \"VoxCPM2 failed to start\" message \"${msg}\"" 2>/dev/null || true
  exit 1
fi

cd "${RESOURCES_DIR}/app"

# Background the server and exit quickly so macOS does not keep the Dock icon bouncing.
nohup "${PYTHON}" "${APP}" --packaged "$@" >>"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "VoxCPM2 server PID: ${SERVER_PID}" >>"${LOG_FILE}"

sleep 1
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  msg="VoxCPM2 failed to start (process exited immediately). See ${LOG_FILE}."
  echo "${msg}" >>"${LOG_FILE}"
  osascript -e "display alert \"VoxCPM2 failed to start\" message \"${msg}\"" 2>/dev/null || true
  exit 1
fi

exit 0
