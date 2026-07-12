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

export PATH="${RESOURCES_DIR}/bin:${PATH}"

PYTHON="${RESOURCES_DIR}/runtime/bin/python"
APP="${RESOURCES_DIR}/app/app.py"

if [[ ! -x "${PYTHON}" ]]; then
  echo "VoxCPM2: bundled Python not found at ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${APP}" ]]; then
  echo "VoxCPM2: application not found at ${APP}" >&2
  exit 1
fi

exec "${PYTHON}" "${APP}" --packaged "$@"
