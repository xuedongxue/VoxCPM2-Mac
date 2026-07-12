#!/usr/bin/env bash
# Build VoxCPM2.app with embedded Python, dependencies, and FFmpeg.
# Run on macOS arm64 (Apple Silicon). Idempotent — safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"
APP_NAME="VoxCPM2"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"
VERSION="${VOXCPM_VERSION:-1.0.0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

require_macos() {
  [[ "$(uname -s)" == "Darwin" ]] || die "This script must run on macOS."
  local arch
  arch="$(uname -m)"
  if [[ "${arch}" != "arm64" ]]; then
    echo "WARNING: Expected arm64, found ${arch}. Build may still work on Intel Macs."
  fi
}

find_python() {
  local candidate
  for candidate in python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      local ver
      ver="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      local major minor
      major="${ver%%.*}"
      minor="${ver#*.}"
      if [[ "${major}" -eq 3 && "${minor}" -ge 10 ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  done
  return 1
}

clean_build_tree() {
  info "Preparing clean build tree in ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}/bundle/Contents/MacOS"
  mkdir -p "${BUILD_DIR}/bundle/Contents/Resources/"{runtime,app,bin}
  mkdir -p "${DIST_DIR}"
}

create_venv() {
  local python_bin="$1"
  local venv_dir="${BUILD_DIR}/bundle/Contents/Resources/runtime"

  info "Creating Python venv with ${python_bin}"
  "${python_bin}" -m venv --copies "${venv_dir}"

  info "Upgrading pip/setuptools/wheel"
  "${venv_dir}/bin/pip" install --upgrade pip setuptools wheel

  info "Installing macOS requirements (this may take several minutes)"
  "${venv_dir}/bin/pip" install -r "${SCRIPT_DIR}/requirements-mac.txt"
}

copy_app_sources() {
  local app_dest="${BUILD_DIR}/bundle/Contents/Resources/app"

  info "Copying application sources"
  cp "${REPO_ROOT}/app.py" "${app_dest}/"

  if [[ -d "${REPO_ROOT}/examples" ]]; then
    cp -R "${REPO_ROOT}/examples" "${app_dest}/"
  fi

  if [[ -d "${REPO_ROOT}/assets" ]]; then
    cp -R "${REPO_ROOT}/assets" "${app_dest}/"
  fi
}

# FFmpeg sources (tried in order):
#   1. FFMPEG_PATH env var or ffmpeg on PATH (copied as-is)
#   2. evermeet.cx static arm64 build (https://evermeet.cx/ffmpeg/)
#   3. johnvansickle ffmpeg-release-arm64-static (GitHub mirror fallback)
fetch_ffmpeg() {
  local dest="${BUILD_DIR}/bundle/Contents/Resources/bin/ffmpeg"
  local tmp_dir="${BUILD_DIR}/ffmpeg-dl"
  mkdir -p "${tmp_dir}"

  if [[ -n "${FFMPEG_PATH:-}" && -x "${FFMPEG_PATH}" ]]; then
    info "Using FFMPEG_PATH: ${FFMPEG_PATH}"
    cp "${FFMPEG_PATH}" "${dest}"
    chmod +x "${dest}"
    return 0
  fi

  if command -v ffmpeg >/dev/null 2>&1; then
    local system_ffmpeg
    system_ffmpeg="$(command -v ffmpeg)"
    info "Copying system ffmpeg from ${system_ffmpeg}"
    cp -L "${system_ffmpeg}" "${dest}"
    chmod +x "${dest}"
    return 0
  fi

  info "Downloading static ffmpeg for macOS arm64 from evermeet.cx"
  local zip="${tmp_dir}/ffmpeg.zip"
  if curl -fsSL -o "${zip}" "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"; then
    unzip -oq "${zip}" -d "${tmp_dir}/evermeet"
    if [[ -f "${tmp_dir}/evermeet/ffmpeg" ]]; then
      cp "${tmp_dir}/evermeet/ffmpeg" "${dest}"
      chmod +x "${dest}"
      info "Installed ffmpeg from evermeet.cx"
      return 0
    fi
  fi

  info "evermeet.cx download failed; trying BtbN FFmpeg-Builds latest arm64 macOS zip"
  local api_url="https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"
  local asset_url
  asset_url="$(curl -fsSL "${api_url}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for a in data.get('assets', []):
    n = a.get('name', '')
    if 'macos' in n.lower() and 'arm64' in n.lower() and n.endswith('.zip'):
        print(a['browser_download_url'])
        break
" || true)"

  if [[ -n "${asset_url}" ]]; then
    local btb_zip="${tmp_dir}/ffmpeg-btb.zip"
    curl -fsSL -o "${btb_zip}" "${asset_url}"
    unzip -oq "${btb_zip}" -d "${tmp_dir}/btb"
    local found
    found="$(find "${tmp_dir}/btb" -name ffmpeg -type f | head -1)"
    if [[ -n "${found}" ]]; then
      cp "${found}" "${dest}"
      chmod +x "${dest}"
      info "Installed ffmpeg from BtbN/FFmpeg-Builds"
      return 0
    fi
  fi

  die "Could not obtain ffmpeg. Install ffmpeg (brew install ffmpeg) or set FFMPEG_PATH."
}

assemble_bundle() {
  info "Assembling ${APP_NAME}.app bundle"
  cp "${SCRIPT_DIR}/launcher.sh" "${BUILD_DIR}/bundle/Contents/MacOS/${APP_NAME}"
  chmod +x "${BUILD_DIR}/bundle/Contents/MacOS/${APP_NAME}"

  sed "s/__VERSION__/${VERSION}/g" "${SCRIPT_DIR}/Info.plist.template" \
    > "${BUILD_DIR}/bundle/Contents/Info.plist"

  rm -rf "${APP_BUNDLE}"
  mkdir -p "${APP_BUNDLE}"
  mv "${BUILD_DIR}/bundle/Contents" "${APP_BUNDLE}/Contents"
}

materialize_runtime_binaries() {
  local bindir="${APP_BUNDLE}/Contents/Resources/runtime/bin"
  [[ -d "${bindir}" ]] || return 0
  info "Materializing Python executables for codesign"
  local entry resolved
  for entry in "${bindir}"/*; do
    [[ -L "${entry}" ]] || continue
    resolved="$(readlink -f "${entry}")"
    [[ -n "${resolved}" && -f "${resolved}" ]] || die "Broken venv symlink: ${entry}"
    rm "${entry}"
    cp "${resolved}" "${entry}"
    chmod +x "${entry}"
  done
}

codesign_app() {
  info "Ad-hoc codesigning ${APP_BUNDLE}"
  codesign --sign - --force --deep "${APP_BUNDLE}"
  codesign --verify --deep --strict "${APP_BUNDLE}"
}

strip_venv_bloat() {
  local venv="${APP_BUNDLE}/Contents/Resources/runtime"
  info "Trimming venv bloat (tests, __pycache__)"
  find "${venv}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  find "${venv}" -type d -name 'tests' -prune -exec rm -rf {} + 2>/dev/null || true
  find "${venv}" -type d -name 'test' -prune -exec rm -rf {} + 2>/dev/null || true
}

main() {
  require_macos

  local python_bin
  python_bin="$(find_python)" || die "Python 3.10+ required (python3.10, python3.11, or python3)."

  info "Building ${APP_NAME}.app v${VERSION}"
  info "Repository root: ${REPO_ROOT}"

  clean_build_tree
  create_venv "${python_bin}"
  copy_app_sources
  fetch_ffmpeg
  assemble_bundle
  strip_venv_bloat
  materialize_runtime_binaries
  codesign_app

  rm -rf "${BUILD_DIR}"

  info "Done: ${APP_BUNDLE}"
  info "Run: open \"${APP_BUNDLE}\""
  info "Create DMG: ${SCRIPT_DIR}/create-dmg.sh"
}

main "$@"
