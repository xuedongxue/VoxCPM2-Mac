#!/usr/bin/env bash
# Generate AppIcon.icns and VolumeIcon.icns from packaging/mac/assets/icon-source.png.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_LOGO="${SCRIPT_DIR}/assets/icon-source.png"
ASSETS_DIR="${SCRIPT_DIR}/assets"
ICONSET_DIR="${ASSETS_DIR}/AppIcon.iconset"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

[[ -f "${SOURCE_LOGO}" ]] || die "App icon source not found: ${SOURCE_LOGO} (place icon-source.png in packaging/mac/assets/)"

mkdir -p "${ASSETS_DIR}" "${ICONSET_DIR}"

info "Rendering square icon sizes from ${SOURCE_LOGO}"
python3 - "${ASSETS_DIR}" "${ICONSET_DIR}" <<'PY'
from pathlib import Path
import sys

from PIL import Image

assets_dir = Path(sys.argv[1])
iconset_dir = Path(sys.argv[2])
source = assets_dir / "icon-source.png"

# Use the source logo as-is (including its original white margins).
# Only scale uniformly to each required icon size — no crop, no extra padding.
img = Image.open(source).convert("RGBA")

iconset_dir.mkdir(parents=True, exist_ok=True)
entries = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]
for name, px in entries:
    img.resize((px, px), Image.Resampling.LANCZOS).save(iconset_dir / name)

info_path = assets_dir / "AppIcon-1024.png"
img.resize((1024, 1024), Image.Resampling.LANCZOS).save(info_path)
print(f"Wrote {info_path}")
PY

info "Building AppIcon.icns"
iconutil -c icns "${ICONSET_DIR}" -o "${ASSETS_DIR}/AppIcon.icns"

info "Building VolumeIcon.icns"
cp "${ASSETS_DIR}/AppIcon.icns" "${ASSETS_DIR}/VolumeIcon.icns"

info "Done:"
info "  ${ASSETS_DIR}/AppIcon.icns"
info "  ${ASSETS_DIR}/VolumeIcon.icns"
