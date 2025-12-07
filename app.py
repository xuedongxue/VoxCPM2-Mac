import os
import numpy as np
import torch
import gradio as gr  
import spaces
from typing import Optional, Tuple
from pathlib import Path
import tempfile
import soundfile as sf
import time


def setup_cache_env():
    """
    Setup cache environment variables.
    Must be called in GPU worker context as well.
    """
    _cache_home = os.path.join(os.path.expanduser("~"), ".cache")
    
    # HuggingFace cache
    os.environ["HF_HOME"] = os.path.join(_cache_home, "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_cache_home, "huggingface", "hub")
    
    # ModelScope cache (for FunASR SenseVoice)
    os.environ["MODELSCOPE_CACHE"] = os.path.join(_cache_home, "modelscope")
    
    # Torch Hub cache (for some audio models like ZipEnhancer)
    os.environ["TORCH_HOME"] = os.path.join(_cache_home, "torch")
    
    # Create cache directories
    for d in [os.environ["HF_HOME"], os.environ["MODELSCOPE_CACHE"], os.environ["TORCH_HOME"]]:
        os.makedirs(d, exist_ok=True)


# Setup cache in main process BEFORE any imports
setup_cache_env()

# Limit thread count to avoid OpenBLAS resource errors in ZeroGPU
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM1.5"

# Global model cache for ZeroGPU
_asr_model = None
_voxcpm_model = None

# Fixed local paths for models (to avoid repeated downloads in GPU workers)
ASR_LOCAL_DIR = "./models/SenseVoiceSmall"
VOXCPM_LOCAL_DIR = "./models/VoxCPM1.5"


def predownload_models():
    """
    Pre-download models at startup (runs in main process, not GPU worker).
    Download to fixed local directories so GPU workers can reuse them.
    """
    print("=" * 50)
    print("Pre-downloading models to local directories...")
    print("=" * 50)
    
    # Pre-download ASR model (SenseVoice) to fixed local directory
    if not os.path.isdir(ASR_LOCAL_DIR) or not os.path.exists(os.path.join(ASR_LOCAL_DIR, "model.pt")):
        try:
            from huggingface_hub import snapshot_download
            asr_model_id = "FunAudioLLM/SenseVoiceSmall"
            print(f"Pre-downloading ASR model: {asr_model_id} -> {ASR_LOCAL_DIR}")
            os.makedirs(ASR_LOCAL_DIR, exist_ok=True)
            snapshot_download(
                repo_id=asr_model_id,
                local_dir=ASR_LOCAL_DIR,
            )
            print(f"ASR model downloaded to: {ASR_LOCAL_DIR}")
        except Exception as e:
            print(f"Warning: Failed to pre-download ASR model: {e}")
    else:
        print(f"ASR model already exists at: {ASR_LOCAL_DIR}")
    
    # Pre-download VoxCPM model to fixed local directory
    if not os.path.isdir(VOXCPM_LOCAL_DIR) or not os.path.exists(os.path.join(VOXCPM_LOCAL_DIR, "model.safetensors")):
        try:
            from huggingface_hub import snapshot_download
            voxcpm_model_id = os.environ.get("HF_REPO_ID", "openbmb/VoxCPM1.5")
            print(f"Pre-downloading VoxCPM model: {voxcpm_model_id} -> {VOXCPM_LOCAL_DIR}")
            os.makedirs(VOXCPM_LOCAL_DIR, exist_ok=True)
            snapshot_download(
                repo_id=voxcpm_model_id,
                local_dir=VOXCPM_LOCAL_DIR,
            )
            print(f"VoxCPM model downloaded to: {VOXCPM_LOCAL_DIR}")
        except Exception as e:
            print(f"Warning: Failed to pre-download VoxCPM model: {e}")
    else:
        print(f"VoxCPM model already exists at: {VOXCPM_LOCAL_DIR}")
    
    print("=" * 50)
    print("Model pre-download complete!")
    print("=" * 50)


# Run pre-download at startup
predownload_models()


def get_asr_model():
    """Lazy load ASR model from local directory."""
    global _asr_model
    if _asr_model is None:
        from funasr import AutoModel
        print("=" * 50)
        print("Loading ASR model...")
        print(f"  Using local path: {ASR_LOCAL_DIR}")
        start_time = time.time()
        _asr_model = AutoModel(
            model=ASR_LOCAL_DIR,  # Use local directory path
            disable_update=True,
            log_level='INFO',
            device="cuda:0",
        )
        load_time = time.time() - start_time
        print(f"ASR model loaded. (耗时: {load_time:.2f}s)")
        print("=" * 50)
    return _asr_model


def get_voxcpm_model():
    """Lazy load VoxCPM model (without denoiser)."""
    global _voxcpm_model
    if _voxcpm_model is None:
        import voxcpm
        print("=" * 50)
        print("Loading VoxCPM model...")
        print(f"  Using local path: {VOXCPM_LOCAL_DIR}")
        start_time = time.time()
        _voxcpm_model = voxcpm.VoxCPM(
            voxcpm_model_path=VOXCPM_LOCAL_DIR, 
            optimize=False,
            enable_denoiser=False,  # Disable denoiser to avoid ZipEnhancer download
        )
        load_time = time.time() - start_time
        print(f"VoxCPM model loaded. (耗时: {load_time:.2f}s)")
        print("=" * 50)
    return _voxcpm_model


@spaces.GPU(duration=120)
def prompt_wav_recognition(prompt_wav: Optional[str]) -> str:
    """Use ASR to recognize prompt audio text."""
    if prompt_wav is None or not prompt_wav.strip():
        return ""
    print("=" * 50)
    print("[ASR] 开始语音识别...")
    asr_model = get_asr_model()
    start_time = time.time()
    res = asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
    inference_time = time.time() - start_time
    text = res[0]["text"].split('|>')[-1]
    print(f"[ASR] 识别结果: {text}")
    print(f"[ASR] 推理耗时: {inference_time:.2f}s")
    print("=" * 50)
    return text


@spaces.GPU(duration=120)
def generate_tts_audio_gpu(
    text_input: str,
    prompt_wav_data: Optional[Tuple[np.ndarray, int]] = None,
    prompt_text_input: Optional[str] = None,
    cfg_value_input: float = 2.0,
    inference_timesteps_input: int = 10,
    do_normalize: bool = True,
) -> Tuple[int, np.ndarray]:
    """
    GPU function: Generate speech from text using VoxCPM.
    prompt_wav_data is (audio_array, sample_rate) tuple.
    """
    voxcpm_model = get_voxcpm_model()

    text = (text_input or "").strip()
    if len(text) == 0:
        raise ValueError("Please input text to synthesize.")

    prompt_text = prompt_text_input if prompt_text_input else None
    prompt_wav_path = None

    # If prompt audio data provided, write to temp file for voxcpm
    if prompt_wav_data is not None:
        audio_array, sr = prompt_wav_data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_array, sr)
            prompt_wav_path = f.name

    try:
        print("=" * 50)
        print("[TTS] 开始语音合成...")
        print(f"[TTS] 目标文本: {text}")
        start_time = time.time()
        wav = voxcpm_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=False,  # Denoiser disabled
        )
        inference_time = time.time() - start_time
        audio_duration = len(wav) / voxcpm_model.tts_model.sample_rate
        rtf = inference_time / audio_duration if audio_duration > 0 else 0
        print(f"[TTS] 推理耗时: {inference_time:.2f}s | 音频时长: {audio_duration:.2f}s | RTF: {rtf:.3f}")
        print("=" * 50)
        return (voxcpm_model.tts_model.sample_rate, wav)
    finally:
        # Cleanup temp file
        if prompt_wav_path and os.path.exists(prompt_wav_path):
            try:
                os.unlink(prompt_wav_path)
            except Exception:
                pass


def generate_tts_audio(
    text_input: str,
    prompt_wav_path_input: Optional[str] = None,
    prompt_text_input: Optional[str] = None,
    cfg_value_input: float = 2.0,
    inference_timesteps_input: int = 10,
    do_normalize: bool = True,
) -> Tuple[int, np.ndarray]:
    """
    Wrapper: Read audio file in CPU, then call GPU function.
    """
    prompt_wav_data = None
    
    # Read audio file before entering GPU context
    if prompt_wav_path_input and os.path.exists(prompt_wav_path_input):
        try:
            audio_array, sr = sf.read(prompt_wav_path_input, dtype='float32')
            prompt_wav_data = (audio_array, sr)
            print(f"Loaded prompt audio: {audio_array.shape}, sr={sr}")
        except Exception as e:
            print(f"Warning: Failed to load prompt audio: {e}")
            prompt_wav_data = None
    
    return generate_tts_audio_gpu(
        text_input=text_input,
        prompt_wav_data=prompt_wav_data,
        prompt_text_input=prompt_text_input,
        cfg_value_input=cfg_value_input,
        inference_timesteps_input=inference_timesteps_input,
        do_normalize=do_normalize,
    )


# ---------- UI Builders ----------

def create_demo_interface():
    """Build the Gradio UI for VoxCPM demo."""
    # static assets (logo path)
    try:
        gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])
    except Exception:
        pass

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Bold accordion labels */
        #acc_quick details > summary,
        #acc_tips details > summary {
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """
    ) as interface:
        # Header logo
        gr.HTML('<div class="logo-container"><img src="/gradio_api/file=assets/voxcpm-logo.png" alt="VoxCPM Logo"></div>')

        # Quick Start
        with gr.Accordion("📋 Quick Start Guide ｜快速入门", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### How to Use ｜使用说明
            1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.  
               **（可选）提供参考声音** - 上传或录制一段音频，为声音合成提供音色、语调和情感等个性化特征
            2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).  
               **（可选项）输入参考文本** - 如果提供了参考语音，请输入其对应的文本内容（支持自动识别）。
            3. **Enter target text** - Type the text you want the model to speak.  
               **输入目标文本** - 输入您希望模型朗读的文字内容。
            4. **Generate Speech** - Click the "Generate" button to create your audio.  
               **生成语音** - 点击"生成"按钮，即可为您创造出音频。
            """)

        # Pro Tips
        with gr.Accordion("💡 Pro Tips ｜使用建议", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### Text Normalization｜文本正则化
            - **Enable** to process general text with an external WeTextProcessing component.  
              **启用**：使用 WeTextProcessing 组件，可支持常见文本的正则化处理。
            - **Disable** to use VoxCPM's native text understanding ability. For example, it supports phonemes input (For Chinese, phonemes are converted using pinyin, {ni3}{hao3}; For English, phonemes are converted using CMUDict, {HH AH0 L OW1}), try it!  
              **禁用**：将使用 VoxCPM 内置的文本理解能力。如，支持音素输入（如中文转拼音：{ni3}{hao3}；英文转CMUDict：{HH AH0 L OW1}）和公式符号合成，尝试一下！

            ### CFG Value｜CFG 值
            - **Lower CFG** if the voice prompt sounds strained or expressive, or instability occurs with long text input.  
              **调低**：如果提示语音听起来不自然或过于夸张，或者长文本输入出现稳定性问题。
            - **Higher CFG** for better adherence to the prompt speech style or input text, or instability occurs with too short text input.
              **调高**：为更好地贴合提示音频的风格或输入文本， 或者极短文本输入出现稳定性问题。

            ### Inference Timesteps｜推理时间步
            - **Lower** for faster synthesis speed.  
              **调低**：合成速度更快。
            - **Higher** for better synthesis quality.  
              **调高**：合成质量更佳。
            """)
            
        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", 'microphone'],
                    type="filepath",
                    label="Prompt Speech (Optional, or let VoxCPM improvise)",
                    value="./examples/example.wav",
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                        label="Prompt Text",
                        placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself..."
                    )
                run_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value (Guidance Scale)",
                    info="Higher values increase adherence to prompt, lower values allow more creativity"
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Number of inference timesteps for generation (higher values may improve quality but slower)"
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.",
                        label="Target Text",
                    )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label="Text Normalization",
                        elem_id="chk_normalize",
                        info="We use wetext library to normalize the input text."
                    )
                audio_output = gr.Audio(label="Output Audio")

        # Wiring
        run_btn.click(
            fn=generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value, inference_timesteps, DoNormalizeText],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )
        prompt_wav.change(fn=prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text])

    return interface


def run_demo(server_name: str = "0.0.0.0", server_port: int = 7860, show_error: bool = True):
    interface = create_demo_interface()
    # Recommended to enable queue on Spaces for better throughput
    interface.queue(max_size=10).launch(server_name=server_name, server_port=server_port, show_error=show_error)


if __name__ == "__main__":
    run_demo()