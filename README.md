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

# VoxCPM-Demo

基于 Gradio 的 [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) 演示：在 **Linux + NVIDIA GPU** 上使用 **nano-vLLM** 推理；在 **macOS** 上使用官方 **voxcpm**（PyTorch，支持 MPS/CPU）。

## macOS 安装包（推荐）

面向不想手动配置 Python 环境的 macOS 用户，可直接下载 **DMG 安装包**（Apple Silicon / arm64，需 **macOS 13+**）。

### 安装步骤

1. **下载 DMG**  
   前往本仓库 [Releases](https://github.com/xuedongxue/VoxCPM2-Mac/releases) 页面，下载最新版 `VoxCPM2.dmg`。

2. **安装应用**  
   打开 DMG，将 **VoxCPM2.app** 拖入「应用程序」（Applications）文件夹。

3. **首次启动（重要）**  
   本应用为开源项目，**未使用付费 Apple 开发者证书签名**。首次打开时 macOS Gatekeeper 可能提示「无法验证开发者」或「已损坏」：
   - 在 Finder 中**右键点击** VoxCPM2.app → 选择 **「打开」** → 在弹窗中再次点击 **「打开」**；
   - 或在「系统设置 → 隐私与安全性」中允许该应用运行。  
   完成一次后，之后可正常双击启动。

4. **选择本地模型目录**  
   主模型 **不会** 打包进安装包。启动后在应用设置中选择本机上的 **VoxCPM2 模型文件夹**（目录内须包含 `config.json` 及权重文件）。ASR / 降噪模型可在首次使用时按需下载。

5. **使用**  
   应用启动后会自动在默认浏览器中打开 Gradio 界面，即可进行语音合成。

### 系统要求

| 项目 | 要求 |
|------|------|
| 系统 | macOS **13**（Ventura）或更高 |
| 芯片 | **Apple Silicon（M 系列）** 推荐；本 DMG 为 arm64 构建 |
| 磁盘 | 应用本体约数 GB；另需为模型权重预留空间（约 5 GB+） |

### 模型下载

从 Hugging Face 下载完整模型到本地，例如：

- 仓库：[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- 可使用 `huggingface-cli download openbmb/VoxCPM2 --local-dir ~/Models/VoxCPM2`，或从网页下载全部文件到同一文件夹
- 确保目录中有 **`config.json`**、`*.safetensors`、`audiovae.pth` 等文件
- 在 VoxCPM2.app 设置中指向该文件夹即可

更详细的图文说明见 [`packaging/mac/INSTALL_zh.md`](packaging/mac/INSTALL_zh.md)。

---

## 本地使用（开发者）

### 环境

- Python **3.10–3.12**
- **macOS**：需安装 [FFmpeg](https://ffmpeg.org/)（例如 `brew install ffmpeg`），供 `torchcodec` / 音频处理使用。

### 安装

```bash
cd VoxCPM-Demo
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 启动

```bash
python app.py
```

浏览器访问终端里提示的地址（默认 `http://127.0.0.1:7860`）。

若直接用系统自带的 `python app.py` 而本机已存在项目下的 `.venv`，脚本会尝试自动改用 `.venv` 中的解释器（需已安装依赖）。

## 模型放在哪里

默认会从 Hugging Face 拉取 **`openbmb/VoxCPM2`**。若已下载到本地，**无需联网**，把**完整模型目录**（内含 `config.json`、`*.safetensors`、`audiovae.pth` 等）放到以下**任一**位置即可自动识别（在未设置 `HF_REPO_ID` / `NANOVLLM_MODEL_PATH` / `NANOVLLM_MODEL` 时生效）：

| 位置 | 说明 |
|------|------|
| `<本仓库上一级>/models/openbmb__VoxCPM2/` | 例如与 `VoxCPM-Demo` 同级的 `models/openbmb__VoxCPM2` |
| `<本仓库根目录>/models/openbmb__VoxCPM2/` | 即 `VoxCPM-Demo/models/openbmb__VoxCPM2` |

目录下必须存在 **`config.json`**，否则不会当作有效模型路径。

也可通过环境变量显式指定（优先级高于自动发现）：

- **`HF_REPO_ID`**：Hugging Face 仓库 id（如 `openbmb/VoxCPM2`）或**本地绝对路径**
- **`NANOVLLM_MODEL_PATH`** / **`NANOVLLM_MODEL`**：同上，见 `app.py` 中 `_resolve_model_ref()`

私有模型需设置 **`HF_TOKEN`**。

## 平台与依赖说明

- **Linux**：安装 CUDA 栈与 `nano-vllm-voxcpm`、`flash-attn` 等（见 `requirements.txt` 中带 `platform_system == "Linux"` 的条目）。
- **macOS**：安装 `voxcpm`，不安装上述 CUDA 专用包；推理由官方 PyTorch 包完成。

## Hugging Face Spaces（GPU）

在 Space 上仍可使用 **nano-vLLM** 后端；`requirements.txt` 中 Linux 条目会在构建镜像时安装。可选环境变量包括：

- **`HF_REPO_ID`**：默认 `openbmb/VoxCPM2`
- **`NANOVLLM_MODEL`** / **`NANOVLLM_MODEL_PATH`**：模型 id 或本地路径
- **`ASR_DEVICE`**：默认 `cpu`
- **`ZIPENHANCER_MODEL_ID`**：参考音频降噪（ModelScope），默认 `iic/speech_zipenhancer_ans_multiloss_16k_base`
- **`NANOVLLM_INFERENCE_TIMESTEPS`**：默认 `10`
- **`NANOVLLM_PREWARM`**：默认 `true`
- **`NANOVLLM_SERVERPOOL_*`**、`NANOVLLM_MAX_GENERATE_LENGTH`、`NANOVLLM_TEMPERATURE` 等：见 `app.py` 与历史说明
- **`GRADIO_QUEUE_MAX_SIZE`**、**`GRADIO_DEFAULT_CONCURRENCY_LIMIT`**、**`GRADIO_SSR_MODE`**
- **`DENOISE_MAX_CONCURRENT`**：默认 `1`

若存在持久化目录 **`/data`**，可将请求日志写到 **`REQUEST_LOG_DIR`**（默认 `/data/logs`）。

## License

Apache-2.0
