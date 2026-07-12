# VoxCPM2 macOS 版

[![Release](https://img.shields.io/github/v/release/xuedongxue/VoxCPM2-Mac)](https://github.com/xuedongxue/VoxCPM2-Mac/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> English: [README.md](README.md)

面向 macOS 的 **VoxCPM2** 原生应用与 **Gradio** 演示，模型来自 [OpenBMB / openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)。

- **macOS**：官方 [`voxcpm`](https://pypi.org/project/voxcpm/) 推理（PyTorch，支持 MPS/CPU）
- **Linux + NVIDIA GPU**：[nano-vLLM](https://github.com/OpenBMB/nano-vllm) 后端（适用于 Hugging Face Spaces）

<!-- 截图占位：有界面截图后可放入 docs/screenshots/app-ui.png -->
<!-- ![VoxCPM2 界面](docs/screenshots/app-ui.png) -->

## macOS 安装包（推荐）

无需手动配置 Python，直接下载 **DMG 安装包**（Apple Silicon / arm64，需 **macOS 13+**）。

### 安装步骤

1. **下载 DMG**  
   前往 [Releases](https://github.com/xuedongxue/VoxCPM2-Mac/releases) 下载最新版 `VoxCPM2.dmg`。

2. **安装应用**  
   打开 DMG，将 **VoxCPM2.app** 拖入「应用程序」（Applications）文件夹。

3. **首次启动（重要）**  
   本应用为开源项目，**未使用付费 Apple 开发者证书签名**。若 macOS 提示「无法验证开发者」或「已损坏」：
   - 在 Finder 中**右键点击** VoxCPM2.app → **「打开」** → 在弹窗中再次点击 **「打开」**；
   - 或在「系统设置 → 隐私与安全性」中允许该应用运行。  
   完成一次后，之后可正常双击启动。

4. **选择本地模型目录**  
   主模型 **不会** 打包进安装包。启动后在应用设置中选择本机上的 **VoxCPM2 模型文件夹**（须含 `config.json` 及权重文件）。ASR / 降噪模型可在首次使用时按需下载。

5. **使用**  
   应用启动后会自动在浏览器中打开 Gradio 界面（默认 `http://127.0.0.1:7860`）。

### 系统要求

| 项目 | 要求 |
|------|------|
| 系统 | macOS **13**（Ventura）或更高 |
| 芯片 | **Apple Silicon（M 系列）** 推荐；本 DMG 为 arm64 构建 |
| 磁盘 | 应用本体约数 GB；另需为模型权重预留约 **5 GB+** 空间 |

### 模型下载

从 Hugging Face 下载完整模型，例如：

```bash
huggingface-cli download openbmb/VoxCPM2 --local-dir ~/Models/VoxCPM2
```

- 模型仓库：[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- 确保目录中有 **`config.json`**、`*.safetensors`、`audiovae.pth` 等文件
- 在 VoxCPM2.app 设置中指向该文件夹即可

打包说明见 [`packaging/mac/README.md`](packaging/mac/README.md)。

---

## 本地使用（开发者）

### 环境

- Python **3.10–3.12**
- **macOS**：需安装 [FFmpeg](https://ffmpeg.org/)（`brew install ffmpeg`）

### 安装

```bash
git clone https://github.com/xuedongxue/VoxCPM2-Mac.git
cd VoxCPM2-Mac
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 启动

```bash
python app.py
```

浏览器访问终端提示的地址。若项目下已有 `.venv` 且系统 Python 缺少依赖，脚本会尝试自动改用虚拟环境。

### 模型放在哪里

默认从 Hugging Face 拉取 **`openbmb/VoxCPM2`**。离线使用时，将完整模型目录放到以下**任一**位置（未设置 `HF_REPO_ID` 等环境变量时自动识别）：

| 位置 | 说明 |
|------|------|
| `<仓库上一级>/models/openbmb__VoxCPM2/` | 与仓库同级的 `models` 目录 |
| `<仓库根目录>/models/openbmb__VoxCPM2/` | 仓库内的 `models` 目录 |

目录下须存在 **`config.json`**。也可通过 **`HF_REPO_ID`**、`NANOVLLM_MODEL_PATH` / **`NANOVLLM_MODEL`** 显式指定；私有模型需 **`HF_TOKEN`**。

### 本地构建 macOS 应用

```bash
chmod +x packaging/mac/*.sh
bash packaging/mac/generate-icons.sh   # 若缺少图标
bash packaging/mac/build.sh
bash packaging/mac/create-dmg.sh
```

产物：`packaging/mac/dist/VoxCPM2.app` 与 `VoxCPM2.dmg`。

---

## Hugging Face Spaces（GPU）

在 Linux Space 上使用 **nano-vLLM** 后端。常用环境变量包括 `HF_REPO_ID`、`ASR_DEVICE`、`NANOVLLM_INFERENCE_TIMESTEPS`、`GRADIO_QUEUE_MAX_SIZE` 等，详见 `app.py`。

## 参与贡献

欢迎提交 Issue 与 Pull Request。macOS 打包相关请参阅 [`packaging/mac/README.md`](packaging/mac/README.md)。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。VoxCPM2 模型权重遵循 [Hugging Face 上的模型许可](https://huggingface.co/openbmb/VoxCPM2)。

## 相关链接

- 模型：[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- 上游：[OpenBMB/VoxCPM2](https://github.com/OpenBMB/VoxCPM2)
- 发布页：[VoxCPM2-Mac Releases](https://github.com/xuedongxue/VoxCPM2-Mac/releases)
