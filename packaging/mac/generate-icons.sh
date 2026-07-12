#!/usr/bin/env bash
# Generate AppIcon.icns, VolumeIcon.icns, and assets/favicon.png from assets/voxcpm_logo.png.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_LOGO="${REPO_ROOT}/assets/voxcpm_logo.png"
ASSETS_DIR="${SCRIPT_DIR}/assets"
ICONSET_DIR="${ASSETS_DIR}/AppIcon.iconset"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

[[ -f "${SOURCE_LOGO}" ]] || die "Logo not found: ${SOURCE_LOGO}"

mkdir -p "${ASSETS_DIR}" "${ICONSET_DIR}"

info "Rendering square icon sizes from ${SOURCE_LOGO}"
python3 - "${REPO_ROOT}" "${ASSETS_DIR}" "${ICONSET_DIR}" <<'PY'
from pathlib import Path
import sys

from PIL import Image

repo_root = Path(sys.argv[1])
assets_dir = Path(sys.argv[2])
iconset_dir = Path(sys.argv[3])
source = repo_root / "assets" / "voxcpm_logo.png"

img = Image.open(source).convert("RGBA")
size = 1024
aspect = img.width / img.height if img.height else 1.0
margin = 0.06 if 0.9 <= aspect <= 1.1 else 0.12
max_w = int(size * (1 - 2 * margin))
max_h = int(size * (1 - 2 * margin))
ratio = min(max_w / img.width, max_h / img.height)
new_w = max(1, int(img.width * ratio))
new_h = max(1, int(img.height * ratio))
resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2), resized)

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
    canvas.resize((px, px), Image.Resampling.LANCZOS).save(iconset_dir / name)

favicon_path = repo_root / "assets" / "favicon.png"
canvas.resize((64, 64), Image.Resampling.LANCZOS).save(favicon_path)
info_path = assets_dir / "AppIcon-1024.png"
canvas.save(info_path)
print(f"Wrote {favicon_path}")
print(f"Wrote {info_path}")
PY

info "Building AppIcon.icns"
iconutil -c icns "${ICONSET_DIR}" -o "${ASSETS_DIR}/AppIcon.icns"

info "Building VolumeIcon.icns"
cp "${ASSETS_DIR}/AppIcon.icns" "${ASSETS_DIR}/VolumeIcon.icns"

info "Done:"
info "  ${ASSETS_DIR}/AppIcon.icns"
info "  ${ASSETS_DIR}/VolumeIcon.icns"
info "  ${REPO_ROOT}/assets/favicon.png"
