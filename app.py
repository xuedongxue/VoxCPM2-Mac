import atexit
import logging
import os
import subprocess
import sys
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import gradio as gr
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

DEFAULT_MODEL_REF = "openbmb/VoxCPM2"
if (
    os.environ.get("NANOVLLM_MODEL", "").strip() == ""
    and os.environ.get("NANOVLLM_MODEL_PATH", "").strip() == ""
    and os.environ.get("HF_REPO_ID", "").strip() == ""
):
    os.environ["HF_REPO_ID"] = DEFAULT_MODEL_REF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_asr_model = None
_voxcpm_server = None
_model_info = None
_server_inference_timesteps = None
_server_lock = Lock()


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value)


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean env: {name}={value!r}")


def _get_devices_env() -> list[int]:
    raw = os.environ.get("NANOVLLM_SERVERPOOL_DEVICES", "0").strip()
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return [0]
    return [int(part) for part in values]


def _resolve_model_ref() -> str:
    for env_name in ("NANOVLLM_MODEL", "NANOVLLM_MODEL_PATH", "HF_REPO_ID"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return DEFAULT_MODEL_REF


def _resolve_flash_attn_wheel_url() -> str:
    override = os.environ.get("FLASH_ATTN_WHEEL_URL", "").strip()
    if override:
        return override

    import torch

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    abi_flag = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    torch_major_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    wheel_name = (
        f"flash_attn-2.8.3+cu12torch{torch_major_minor}cxx11abi{abi_flag}-"
        f"{py_tag}-{py_tag}-linux_x86_64.whl"
    )
    return f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/{wheel_name}"


def _ensure_nanovllm_runtime() -> None:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        wheel_url = _resolve_flash_attn_wheel_url()
        logger.info(f"Installing flash-attn wheel at runtime from {wheel_url} ...")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    wheel_url,
                ]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to install the configured flash-attn wheel. "
                "Set FLASH_ATTN_WHEEL_URL to a matching prebuilt wheel for this Space."
            ) from exc

    try:
        import nanovllm_voxcpm  # noqa: F401
    except ImportError:
        logger.info("Installing nanovllm-voxcpm at runtime ...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "git+https://github.com/a710128/nanovllm-voxcpm.git",
            ]
        )


def _extract_asr_text(asr_result) -> str:
    if not asr_result:
        return ""

    first_item = asr_result[0]
    if isinstance(first_item, dict):
        return str(first_item.get("text", "")).split("|>")[-1].strip()
    return ""


def _read_audio_bytes(audio_path: Optional[str]) -> tuple[bytes | None, str | None]:
    if audio_path is None or not audio_path.strip():
        return None, None

    path = Path(audio_path)
    audio_format = path.suffix.lstrip(".").lower() or "wav"
    return path.read_bytes(), audio_format


def _safe_prompt_wav_recognition(use_prompt_text: bool, prompt_wav: Optional[str]) -> str:
    try:
        return prompt_wav_recognition(use_prompt_text, prompt_wav)
    except Exception as exc:
        logger.warning(f"ASR recognition failed: {exc}")
        return ""


def _stop_server_if_needed() -> None:
    global _voxcpm_server, _model_info, _server_inference_timesteps
    if _voxcpm_server is None:
        return

    stop = getattr(_voxcpm_server, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception as exc:
            logger.warning(f"Failed to stop nano-vLLM server cleanly: {exc}")

    _voxcpm_server = None
    _model_info = None
    _server_inference_timesteps = None


atexit.register(_stop_server_if_needed)

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

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        from funasr import AutoModel

        device = os.environ.get("ASR_DEVICE", "cpu").strip() or "cpu"
        logger.info(f"Loading ASR model on {device} ...")
        _asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            log_level="INFO",
            device=device,
        )
        logger.info("ASR model loaded.")
    return _asr_model


def get_voxcpm_server(inference_timesteps: int):
    global _voxcpm_server, _model_info, _server_inference_timesteps
    if _voxcpm_server is not None and _server_inference_timesteps == inference_timesteps:
        return _voxcpm_server

    with _server_lock:
        if _voxcpm_server is not None and _server_inference_timesteps == inference_timesteps:
            return _voxcpm_server

        if _voxcpm_server is not None and _server_inference_timesteps != inference_timesteps:
            logger.info(
                f"Rebuilding nano-vLLM server for inference_timesteps={inference_timesteps} "
                f"(previous={_server_inference_timesteps})"
            )
            _stop_server_if_needed()

        _ensure_nanovllm_runtime()
        from nanovllm_voxcpm import VoxCPM

        model_ref = _resolve_model_ref()
        logger.info(
            f"Loading nano-vLLM VoxCPM server from {model_ref} "
            f"with inference_timesteps={inference_timesteps} ..."
        )
        _voxcpm_server = VoxCPM.from_pretrained(
            model=model_ref,
            inference_timesteps=int(inference_timesteps),
            max_num_batched_tokens=_get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS", 8192),
            max_num_seqs=_get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS", 16),
            max_model_len=_get_int_env("NANOVLLM_SERVERPOOL_MAX_MODEL_LEN", 4096),
            gpu_memory_utilization=_get_float_env("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION", 0.95),
            enforce_eager=_get_bool_env("NANOVLLM_SERVERPOOL_ENFORCE_EAGER", False),
            devices=_get_devices_env(),
        )
        _model_info = _voxcpm_server.get_model_info()
        _server_inference_timesteps = inference_timesteps
        logger.info(f"nano-vLLM VoxCPM server loaded: {_model_info}")
    return _voxcpm_server


def get_model_info(inference_timesteps: int) -> dict:
    global _model_info
    if _model_info is None or _server_inference_timesteps != inference_timesteps:
        get_voxcpm_server(inference_timesteps)
    assert _model_info is not None
    return _model_info


# ---------- GPU-accelerated inference ----------


def prompt_wav_recognition(use_prompt_text: bool, prompt_wav: Optional[str]) -> str:
    if not use_prompt_text or prompt_wav is None or not prompt_wav.strip():
        return ""

    asr_model = get_asr_model()
    res = asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
    return _extract_asr_text(res)


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
    timesteps = int(inference_timesteps)
    server = get_voxcpm_server(timesteps)
    model_info = get_model_info(timesteps)

    text = (text_input or "").strip()
    if len(text) == 0:
        raise ValueError("Please input text to synthesize.")

    control = (control_instruction or "").strip()
    final_text = f"({control}){text}" if control and not use_prompt_text else text

    audio_bytes, audio_format = _read_audio_bytes(reference_wav_path_input)
    prompt_text_clean = (prompt_text_input or "").strip()
    if use_prompt_text and audio_bytes is None:
        raise ValueError("Ultimate Cloning Mode requires a reference audio clip.")
    if use_prompt_text and not prompt_text_clean:
        raise ValueError(
            "Ultimate Cloning Mode requires a transcript. Please wait for ASR or fill it in manually."
        )
    if not use_prompt_text:
        prompt_text_clean = ""

    if do_normalize:
        logger.info("Ignoring normalize option: nano-vLLM backend does not support per-request text normalization.")
    if denoise:
        logger.info("Ignoring denoise option: nano-vLLM backend does not support per-request reference denoising.")

    prompt_latents = None
    ref_audio_latents = None
    if audio_bytes is not None and audio_format is not None and use_prompt_text:
        logger.info(f"[Ultimate Cloning] encoding prompt audio as {audio_format}")
        prompt_latents = server.encode_latents(audio_bytes, audio_format)
    elif audio_bytes is not None and audio_format is not None:
        logger.info(f"[Controllable Cloning] encoding reference audio as {audio_format}")
        ref_audio_latents = server.encode_latents(audio_bytes, audio_format)

    if prompt_latents is not None:
        logger.info("[Ultimate Cloning] reference audio + transcript")
    elif ref_audio_latents is not None:
        logger.info("[Controllable Cloning] reference audio only")
    else:
        logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}")

    chunks: list[np.ndarray] = []
    logger.info(f"Generating: '{final_text[:80]}...'")
    for chunk in server.generate(
        target_text=final_text,
        prompt_latents=prompt_latents,
        prompt_text=prompt_text_clean if prompt_latents is not None else "",
        max_generate_length=_get_int_env("NANOVLLM_MAX_GENERATE_LENGTH", 2000),
        temperature=_get_float_env("NANOVLLM_TEMPERATURE", 1.0),
        cfg_value=float(cfg_value_input),
        ref_audio_latents=ref_audio_latents,
    ):
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError("The model returned no audio chunks.")

    wav = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    return (int(model_info["sample_rate"]), wav)


# ---------- UI ----------


def create_demo_interface():
    assets_dir = Path.cwd().absolute() / "assets"
    if assets_dir.exists():
        gr.set_static_paths(paths=[assets_dir])

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
        logger.info("Running ASR on reference audio...")
        asr_text = _safe_prompt_wav_recognition(True, audio_path)
        logger.info(f"ASR result: {asr_text[:60]}...")
        return gr.update(value=asr_text)

    with gr.Blocks() as interface:
        if (assets_dir / "voxcpm_logo.png").exists():
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
