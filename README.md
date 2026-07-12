---
title: VoxCPM Demo
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
python_version: "3.10"
pinned: true
license: apache-2.0
short_description: VoxCPM2 Gradio Demo (Nano-vLLM on Linux / voxcpm on macOS)
---

# VoxCPM2 for macOS

[![Release](https://img.shields.io/github/v/release/xuedongxue/VoxCPM2-Mac)](https://github.com/xuedongxue/VoxCPM2-Mac/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> 中文文档：[README_zh.md](README_zh.md)

Native **macOS app** and **Gradio** demo for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — a multilingual text-to-speech model from [OpenBMB](https://github.com/OpenBMB).

- **macOS**: official [`voxcpm`](https://pypi.org/project/voxcpm/) backend (PyTorch, MPS/CPU)
- **Linux + NVIDIA GPU**: [nano-vLLM](https://github.com/OpenBMB/nano-vllm) backend for Spaces and server deployments

<!-- Screenshots: add docs/screenshots/app-ui.png when available -->
<!-- ![VoxCPM2 UI](docs/screenshots/app-ui.png) -->

## macOS app (recommended)

Download a ready-to-use **DMG installer** for Apple Silicon (arm64, **macOS 13+**).

### Install

1. **Download** the latest [`VoxCPM2.dmg`](https://github.com/xuedongxue/VoxCPM2-Mac/releases) from [Releases](https://github.com/xuedongxue/VoxCPM2-Mac/releases).
2. **Drag** `VoxCPM2.app` into **Applications**.
3. **First launch** — this app is open source and **not signed with a paid Apple Developer ID**. If macOS blocks it:
   - **Right-click** `VoxCPM2.app` → **Open** → confirm **Open** again; or
   - Allow it under **System Settings → Privacy & Security**.
4. **Select your model folder** in the app settings. The main [VoxCPM2 weights](https://huggingface.co/openbmb/VoxCPM2) are **not** bundled (see [Model download](#model-download)). ASR and denoiser models download on first use if needed.
5. The app opens the Gradio UI in your browser (default `http://127.0.0.1:7860`).

### System requirements

| Item | Requirement |
|------|-------------|
| OS | macOS **13** (Ventura) or later |
| CPU | **Apple Silicon (M-series)** recommended; DMG is arm64 |
| Disk | Several GB for the app; **5 GB+** additional space for model weights |

### Model download

Download the full model from Hugging Face:

```bash
huggingface-cli download openbmb/VoxCPM2 --local-dir ~/Models/VoxCPM2
```

The folder must contain `config.json`, `*.safetensors`, `audiovae.pth`, and related files. Point **VoxCPM2.app** settings at that directory.

Packaging details: [`packaging/mac/README.md`](packaging/mac/README.md).

---

## Developer setup

For running from source or contributing.

### Prerequisites

- Python **3.10–3.12**
- **macOS**: [FFmpeg](https://ffmpeg.org/) (`brew install ffmpeg`) for audio codecs

### Install

```bash
git clone https://github.com/xuedongxue/VoxCPM2-Mac.git
cd VoxCPM2-Mac
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open the URL printed in the terminal (default `http://127.0.0.1:7860`). If a project `.venv` exists, `app.py` may re-exec into it when dependencies are missing from the system Python.

### Model paths

By default the app uses **`openbmb/VoxCPM2`** from Hugging Face. For offline use, place a complete model directory (with `config.json`) at either:

| Path | Notes |
|------|-------|
| `../models/openbmb__VoxCPM2/` | Sibling of the repo |
| `./models/openbmb__VoxCPM2/` | Inside the repo |

Override with environment variables (higher priority): `HF_REPO_ID`, `NANOVLLM_MODEL_PATH`, or `NANOVLLM_MODEL`. Set `HF_TOKEN` for private models.

### Build the macOS app locally

```bash
chmod +x packaging/mac/*.sh
bash packaging/mac/generate-icons.sh   # if icons are missing
bash packaging/mac/build.sh
bash packaging/mac/create-dmg.sh
```

Output: `packaging/mac/dist/VoxCPM2.app` and `VoxCPM2.dmg`.

---

## Hugging Face Spaces (GPU)

On Linux Spaces, the **nano-vLLM** backend is used. Notable environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_REPO_ID` | `openbmb/VoxCPM2` | Model repo or local path |
| `ASR_DEVICE` | `cpu` | ASR device |
| `NANOVLLM_INFERENCE_TIMESTEPS` | `10` | Inference steps |
| `GRADIO_QUEUE_MAX_SIZE` | `10` | Queue size |

See `app.py` for the full list. Request logs can be written to `REQUEST_LOG_DIR` (default `/data/logs`) when `/data` is available.

## Contributing

Issues and pull requests are welcome. For macOS packaging changes, see [`packaging/mac/README.md`](packaging/mac/README.md).

## License

[Apache License 2.0](LICENSE). VoxCPM2 model weights are subject to the [model license](https://huggingface.co/openbmb/VoxCPM2) on Hugging Face.

## Links

- Model: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- Upstream: [OpenBMB/VoxCPM2](https://github.com/OpenBMB/VoxCPM2)
- Releases: [VoxCPM2-Mac Releases](https://github.com/xuedongxue/VoxCPM2-Mac/releases)
