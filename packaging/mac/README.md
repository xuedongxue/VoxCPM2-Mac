# VoxCPM2 macOS Packaging

> 中文安装说明：[INSTALL_zh.md](INSTALL_zh.md) · 用户文档：[README_zh.md](../../README_zh.md)

Scripts to build **VoxCPM2.app** (embedded Python + dependencies + FFmpeg) and a **DMG** installer. Uses ad-hoc codesign only — no Apple Developer certificate required.

## Requirements

- macOS 13.0+ (Ventura or later)
- Apple Silicon (arm64) recommended; Intel may work but is untested
- Python **3.10** or **3.11** on the build machine (`python3.11` or `python3.10`)
- Xcode Command Line Tools (`xcode-select --install`) for `codesign` and `hdiutil`
- Network access during build (pip + optional ffmpeg download)
- ~5–10 GB free disk space (PyTorch + models cache separately at runtime)

Optional:

- `create-dmg` (`brew install create-dmg`) for a prettier DMG layout
- `ffmpeg` on PATH or `FFMPEG_PATH` if you prefer not to download a static binary

## Bundle layout

```
VoxCPM2.app/
  Contents/
    MacOS/VoxCPM2              # launcher (packaging/mac/launcher.sh)
    Resources/
      AppIcon.icns             # app icon (from assets/voxcpm_logo.png)
      runtime/                 # Python venv with macOS deps
      app/                     # app.py, examples/, assets/
      bin/ffmpeg               # bundled ffmpeg
    Info.plist
```

At runtime the launcher sets:

| Variable | Value |
|----------|-------|
| `VOXCPM_PACKAGED` | `1` |
| `VOXCPM_HOME` | `~/Library/Application Support/VoxCPM2` |
| `HF_HOME` | `$VOXCPM_HOME/cache/huggingface` |
| `MODELSCOPE_CACHE` | `$VOXCPM_HOME/cache/modelscope` |

The app is started as:

```bash
Resources/runtime/bin/python Resources/app/app.py --packaged
```

## Build locally

From the repository root:

```bash
chmod +x packaging/mac/build.sh packaging/mac/create-dmg.sh packaging/mac/launcher.sh packaging/mac/generate-icons.sh
bash packaging/mac/generate-icons.sh   # creates packaging/mac/assets/AppIcon.icns
./packaging/mac/build.sh
```

Output: `packaging/mac/dist/VoxCPM2.app`

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXCPM_VERSION` | `1.0.1` | CFBundleShortVersionString in Info.plist |
| `FFMPEG_PATH` | (auto) | Path to an existing `ffmpeg` binary to bundle |

### FFmpeg source

`build.sh` tries, in order:

1. `$FFMPEG_PATH` if set and executable
2. `ffmpeg` found on `PATH` (e.g. from Homebrew)
3. Static build from [evermeet.cx/ffmpeg](https://evermeet.cx/ffmpeg/) (macOS)
4. Latest arm64 macOS asset from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases)

## Create DMG

After a successful app build:

```bash
./packaging/mac/create-dmg.sh
```

Output: `packaging/mac/dist/VoxCPM2.dmg`

The DMG contains `VoxCPM2.app`, an `Applications` symlink, and `README.txt` (including 右键→打开 instructions). When `create-dmg` is available, the volume uses `packaging/mac/assets/VolumeIcon.icns`.

## Run without installing

```bash
open packaging/mac/dist/VoxCPM2.app
```

Or from Terminal:

```bash
packaging/mac/dist/VoxCPM2.app/Contents/MacOS/VoxCPM2
```

Gradio UI defaults to `http://127.0.0.1:7860`.

## First launch / Gatekeeper

Because the app is ad-hoc signed, macOS may block the first open. Use **右键 → 打开** (Right-click → Open), or:

```bash
xattr -cr packaging/mac/dist/VoxCPM2.app
```

## Rebuild

Scripts are idempotent. Re-running `build.sh` removes the previous `build/` tree and replaces `dist/VoxCPM2.app`.

## Files

| File | Purpose |
|------|---------|
| `generate-icons.sh` | Build `AppIcon.icns` / `VolumeIcon.icns` from `assets/voxcpm_logo.png` |
| `assets/AppIcon.icns` | Committed app icon (regenerate when logo changes) |
| `requirements-mac.txt` | Darwin-only pip dependencies |
| `launcher.sh` | App entry point copied to `MacOS/VoxCPM2` |
| `Info.plist.template` | Bundle metadata (`__VERSION__` substituted at build time) |
| `build.sh` | Full .app build pipeline |
| `create-dmg.sh` | DMG packaging |
| `.gitignore` | Ignores `build/` and `dist/` |
