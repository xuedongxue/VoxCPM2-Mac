import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

def _ensure_torchaudio():
    """Install torchaudio matching ZeroGPU's pre-installed torch + CUDA version."""
    try:
        import torchaudio  # noqa: F401
        return
    except (ImportError, OSError):
        pass
    import torch
    torch_ver = torch.__version__.split("+")[0]
    cuda_ver = torch.version.cuda
    if cuda_ver:
        tag = "cu" + cuda_ver.replace(".", "")
    else:
        tag = "cpu"
    index = f"https://download.pytorch.org/whl/{tag}"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-deps",
        "--index-url", index,
        f"torchaudio=={torch_ver}",
    ])

_ensure_torchaudio()

try:
    import voxcpm  # noqa: F401
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-deps",
        "voxcpm @ git+https://github.com/OpenBMB/VoxCPM.git@dev_2.0",
    ])
    import voxcpm  # noqa: F401

import gradio as gr
import numpy as np
import spaces
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    '`Target Text`: *"I never asked you to stay… It\'s not like I care or anything. '
    'But… why does it still hurt so much now that you\'re gone?"*  \n\n'
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,
    "zh": None,
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
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

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

# ---------- Model Pre-download & Loading ----------

ASR_LOCAL_DIR = "./models/SenseVoiceSmall"
VOXCPM_LOCAL_DIR = "./models/VoxCPM2"

_asr_model = None
_voxcpm_model = None


def predownload_models():
    from huggingface_hub import snapshot_download

    if not os.path.isdir(ASR_LOCAL_DIR) or not os.path.exists(
        os.path.join(ASR_LOCAL_DIR, "model.pt")
    ):
        logger.info(f"Pre-downloading ASR model to {ASR_LOCAL_DIR} ...")
        os.makedirs(ASR_LOCAL_DIR, exist_ok=True)
        try:
            snapshot_download(
                repo_id="FunAudioLLM/SenseVoiceSmall", local_dir=ASR_LOCAL_DIR
            )
            logger.info("ASR model downloaded.")
        except Exception as exc:
            logger.warning(f"Failed to pre-download ASR model: {exc}")
    else:
        logger.info(f"ASR model already at {ASR_LOCAL_DIR}")

    voxcpm_repo_id = os.environ.get("HF_REPO_ID", "openbmb/VoxCPM2")
    if not os.path.isdir(VOXCPM_LOCAL_DIR) or not os.path.exists(
        os.path.join(VOXCPM_LOCAL_DIR, "config.json")
    ):
        logger.info(
            f"Pre-downloading VoxCPM model {voxcpm_repo_id} to {VOXCPM_LOCAL_DIR} ..."
        )
        os.makedirs(VOXCPM_LOCAL_DIR, exist_ok=True)
        try:
            snapshot_download(repo_id=voxcpm_repo_id, local_dir=VOXCPM_LOCAL_DIR)
            logger.info("VoxCPM model downloaded.")
        except Exception as exc:
            logger.warning(f"Failed to pre-download VoxCPM model: {exc}")
    else:
        logger.info(f"VoxCPM model already at {VOXCPM_LOCAL_DIR}")


predownload_models()


def get_asr_model():
    global _asr_model
    if _asr_model is None:
        from funasr import AutoModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading ASR model on {device} ...")
        _asr_model = AutoModel(
            model=ASR_LOCAL_DIR,
            disable_update=True,
            log_level="INFO",
            device=device,
        )
        logger.info("ASR model loaded.")
    return _asr_model


def get_voxcpm_model():
    global _voxcpm_model
    if _voxcpm_model is None:
        logger.info(
            f"[DEBUG] CUDA available: {torch.cuda.is_available()}, "
            f"device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        logger.info(f"Loading VoxCPM model from {VOXCPM_LOCAL_DIR} ...")
        _voxcpm_model = voxcpm.VoxCPM(
            voxcpm_model_path=VOXCPM_LOCAL_DIR, optimize=True
        )
        logger.info("VoxCPM model loaded.")
    return _voxcpm_model


# ---------- GPU-accelerated inference ----------


@spaces.GPU
def prompt_wav_recognition(use_prompt_text: bool, prompt_wav: Optional[str]) -> str:
    if not use_prompt_text or prompt_wav is None or not prompt_wav.strip():
        return ""

    asr_model = get_asr_model()
    res = asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
    return res[0]["text"].split("|>")[-1]


@spaces.GPU(duration=600)
def generate_tts_audio(
    text_input: str,
    control_instruction: str = "",
    reference_wav_path_input: Optional[str] = None,
    use_prompt_text: bool = False,
    prompt_text_input: str = "",
    cfg_value_input: float = 2.0,
    do_normalize: bool = True,
    denoise: bool = True,
    inference_timesteps: int = 10,
) -> Tuple[int, np.ndarray]:
    voxcpm_model = get_voxcpm_model()

    text = (text_input or "").strip()
    if len(text) == 0:
        raise ValueError("Please input text to synthesize.")

    control = (control_instruction or "").strip()
    final_text = f"({control}){text}" if control and not use_prompt_text else text

    audio_path = reference_wav_path_input if reference_wav_path_input else None
    prompt_text_clean = (prompt_text_input or "").strip() or None
    if not use_prompt_text:
        prompt_text_clean = None

    if audio_path and prompt_text_clean:
        logger.info("[Ultimate Cloning] reference audio + transcript")
    elif audio_path:
        logger.info("[Controllable Cloning] reference audio only")
    else:
        logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}")

    generate_kwargs = dict(
        text=final_text,
        reference_wav_path=audio_path,
        cfg_value=float(cfg_value_input),
        inference_timesteps=int(inference_timesteps),
        normalize=do_normalize,
        denoise=denoise,
    )
    if prompt_text_clean and audio_path:
        generate_kwargs["prompt_wav_path"] = audio_path
        generate_kwargs["prompt_text"] = prompt_text_clean

    logger.info(f"Generating: '{final_text[:80]}...'")
    wav = voxcpm_model.generate(**generate_kwargs)
    return (voxcpm_model.tts_model.sample_rate, wav)


# ---------- UI ----------


def create_demo_interface():
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _on_toggle_instant(checked):
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = prompt_wav_recognition(True, audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )

                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"))
                gr.Markdown(I18N("examples_footer"))

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        run_btn.click(
            fn=generate_tts_audio,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
            ],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )

    return interface


def run_demo(
    server_name: str = "0.0.0.0", server_port: int = 7860, show_error: bool = True
):
    interface = create_demo_interface()
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=int(os.environ.get("PORT", server_port)),
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    run_demo()
