---
title: VoxCPM Demo
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
python_version: "3.10"
pinned: true
license: apache-2.0
short_description: VoxCPM2 Nano-vLLM Demo
---

Experimental Gradio Space demo for `VoxCPM2` powered by `nanovllm-voxcpm`.

This repo keeps the existing Gradio frontend layout and swaps only the backend inference path to Nano-vLLM.

Notes:

- This is the non-Docker experiment path. It relies on a persistent GPU Gradio Space.
- `flash-attn` and `nanovllm-voxcpm` are pinned in `requirements.txt`, so they install during Space build instead of on first request.
- The Space now defaults to a hardened runtime path:
  - If `/data` exists, Hugging Face cache, pip cache, and Gradio temp files are stored there automatically.
  - Backend prewarm is enabled by default, so startup can begin dependency install + model load in the background.
  - Gradio SSR is disabled by default for stability.
- The first cold start may still spend extra time installing dependencies, downloading the model, and loading the server.
- `ASR_DEVICE` defaults to `cpu` to avoid competing with TTS GPU memory.
- The `LocDiT flow-matching steps` slider is wired to Nano-vLLM server `inference_timesteps`; changing it rebuilds the backend server.
- The existing `normalize` / `denoise` frontend toggles are kept for UI compatibility, but Nano-vLLM currently ignores them.
- `packages.txt` is required because this path needs extra system build dependencies.

Stability recommendation:

- Use a persistent GPU Space.
- Attach persistent storage so `/data` is available.
- Keep the default queue concurrency at `1` unless you have profiled GPU memory headroom.

Recommended environment variables:

- `HF_REPO_ID`: Hugging Face model repo id. Defaults to `openbmb/VoxCPM2`
- `HF_TOKEN`: required if the model repo is private
- `NANOVLLM_MODEL`: optional direct model ref override. Can be a local path or HF repo id
- `NANOVLLM_MODEL_PATH`: optional local model path override
- `ASR_DEVICE`: defaults to `cpu`
- `NANOVLLM_INFERENCE_TIMESTEPS`: initial default is `10`
- `NANOVLLM_PREWARM`: defaults to `true`
- `NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS`: defaults to `8192`
- `NANOVLLM_SERVERPOOL_MAX_NUM_SEQS`: defaults to `16`
- `NANOVLLM_SERVERPOOL_MAX_MODEL_LEN`: defaults to `4096`
- `NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION`: defaults to `0.95`
- `NANOVLLM_SERVERPOOL_ENFORCE_EAGER`: defaults to `false`
- `NANOVLLM_SERVERPOOL_DEVICES`: defaults to `0`
- `NANOVLLM_MAX_GENERATE_LENGTH`: defaults to `2000`
- `NANOVLLM_TEMPERATURE`: defaults to `1.0`
- `GRADIO_QUEUE_MAX_SIZE`: defaults to `10`
- `GRADIO_DEFAULT_CONCURRENCY_LIMIT`: defaults to `1`
- `GRADIO_SSR_MODE`: defaults to `false`
