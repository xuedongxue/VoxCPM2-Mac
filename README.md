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

## 本地使用

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
