#!/usr/bin/env bash
# Create a distributable DMG from VoxCPM2.app
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="${SCRIPT_DIR}/dist"
APP_NAME="VoxCPM2"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"
DMG_PATH="${DIST_DIR}/${APP_NAME}.dmg"
STAGING_DIR="${SCRIPT_DIR}/build/dmg-staging"
README_NAME="README.txt"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

write_readme() {
  cat > "${STAGING_DIR}/${README_NAME}" <<'EOF'
VoxCPM2 for macOS
=================

1. Drag VoxCPM2.app into the Applications folder.
2. Open VoxCPM2 from Applications (or Launchpad).

若无法打开，请右键点击 VoxCPM2.app → 选择「打开」。
首次打开时 macOS 可能提示来自未识别开发者，再次选择「打开」即可。

Models and caches are stored under:
  ~/Library/Application Support/VoxCPM2/

The app launches a local Gradio web UI (default http://127.0.0.1:7860).
EOF
}

create_dmg_hdiutil() {
  local vol_name="${APP_NAME}"
  local temp_dmg="${SCRIPT_DIR}/build/${APP_NAME}-temp.dmg"
  rm -f "${temp_dmg}" "${DMG_PATH}"

  info "Creating temporary disk image"
  hdiutil create \
    -volname "${vol_name}" \
    -srcfolder "${STAGING_DIR}" \
    -ov \
    -format UDRW \
    "${temp_dmg}"

  info "Converting to compressed read-only DMG"
  hdiutil convert "${temp_dmg}" -format UDZO -o "${DMG_PATH}"
  rm -f "${temp_dmg}"
}

create_dmg_create_dmg() {
  info "Using create-dmg (if installed)"
  rm -f "${DMG_PATH}"
  local vol_icon="${SCRIPT_DIR}/assets/VolumeIcon.icns"
  local -a dmg_args=(
    --volname "${APP_NAME}"
    --window-pos 200 120
    --window-size 600 400
    --icon-size 100
    --icon "${APP_NAME}.app" 150 190
    --hide-extension "${APP_NAME}.app"
    --app-drop-link 450 190
  )
  if [[ -f "${vol_icon}" ]]; then
    dmg_args+=(--volicon "${vol_icon}")
  fi
  create-dmg \
    "${dmg_args[@]}" \
    "${DMG_PATH}" \
    "${STAGING_DIR}"
}

main() {
  [[ -d "${APP_BUNDLE}" ]] || die "App bundle not found. Run build.sh first: ${APP_BUNDLE}"

  if [[ ! -f "${SCRIPT_DIR}/assets/VolumeIcon.icns" ]]; then
    info "Volume icon missing; generating icons"
    bash "${SCRIPT_DIR}/generate-icons.sh"
  fi

  info "Staging DMG contents"
  rm -rf "${STAGING_DIR}"
  mkdir -p "${STAGING_DIR}"

  cp -R "${APP_BUNDLE}" "${STAGING_DIR}/"
  ln -sf /Applications "${STAGING_DIR}/Applications"
  write_readme

  if command -v create-dmg >/dev/null 2>&1; then
    create_dmg_create_dmg || create_dmg_hdiutil
  else
    create_dmg_hdiutil
  fi

  rm -rf "${STAGING_DIR}"
  rm -f "${SCRIPT_DIR}/build/${APP_NAME}-temp.dmg"

  info "Done: ${DMG_PATH}"
  info "Verify: hdiutil verify \"${DMG_PATH}\""
}

main "$@"
