import asyncio
import atexit
import json
import logging
import os
import queue
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Semaphore, Thread
from typing import Optional, Tuple

import gradio as gr
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

DEFAULT_MODEL_REF = "openbmb/VoxCPM2"


def _discover_default_local_model_ref() -> str:
    here = Path(__file__).resolve().parent
    for candidate in (
        here.parent / "models" / "openbmb__VoxCPM2",
        here / "models" / "openbmb__VoxCPM2",
    ):
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return str(candidate)
    return DEFAULT_MODEL_REF


if (
    os.environ.get("NANOVLLM_MODEL", "").strip() == ""
    and os.environ.get("NANOVLLM_MODEL_PATH", "").strip() == ""
    and os.environ.get("HF_REPO_ID", "").strip() == ""
):
    os.environ["HF_REPO_ID"] = _discover_default_local_model_ref()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
DEFAULT_ASR_MODEL_REF = "FunAudioLLM/SenseVoiceSmall"
DEFAULT_ZIPENHANCER_MODEL = "iic/speech_zipenhancer_ans_multiloss_16k_base"
MAX_REFERENCE_AUDIO_SECONDS = 50.0
_persistent_root = None
_request_log_dir = None


def _configure_cache_dirs() -> None:
    global _persistent_root, _request_log_dir
    persistent_root = Path(os.environ.get("SPACE_PERSISTENT_ROOT", "/data")).expanduser()
    if not persistent_root.exists():
        logger.info("Persistent storage not detected. Request logs disabled.")
        return

    logs_dir = Path(
        os.environ.get("REQUEST_LOG_DIR", str(persistent_root / "logs"))
    ).expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)
    _persistent_root = persistent_root
    _request_log_dir = logs_dir
    logger.info(f"Persistent storage detected at {persistent_root}")
    logger.info(f"Request logs will be written to daily files under {_request_log_dir}")


_configure_cache_dirs()

_asr_model = None
_voxcpm_server = None
_model_info = None
_denoiser = None
_asr_lock = Lock()
_server_lock = Lock()
_prewarm_lock = Lock()
_denoiser_lock = Lock()
_denoise_semaphore = Semaphore(int(os.environ.get("DENOISE_MAX_CONCURRENT", "1")))
_prewarm_started = False
_runtime_diag_logged = False
_active_generation_requests = 0
_active_generation_lock = Lock()


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


def _use_native_voxcpm_backend() -> bool:
    return sys.platform == "darwin"


_voxcpm_pkg_model = None


def _get_voxcpm_pkg_model():
    global _voxcpm_pkg_model
    if _voxcpm_pkg_model is not None:
        return _voxcpm_pkg_model
    with _server_lock:
        if _voxcpm_pkg_model is not None:
            return _voxcpm_pkg_model
        from voxcpm import VoxCPM

        _log_runtime_diagnostics_once()
        model_ref = _resolve_model_ref()
        logger.info(f"Loading VoxCPM (PyTorch) from {model_ref} ...")
        optimize = _get_bool_env("VOXCPM_OPTIMIZE", False)
        load_denoiser = _get_bool_env("VOXCPM_LOAD_DENOISER", False)
        _voxcpm_pkg_model = VoxCPM.from_pretrained(
            model_ref,
            load_denoiser=load_denoiser,
            optimize=optimize,
        )
        logger.info("VoxCPM (PyTorch) loaded.")
        return _voxcpm_pkg_model


def _resolve_model_ref() -> str:
    for env_name in ("NANOVLLM_MODEL", "NANOVLLM_MODEL_PATH", "HF_REPO_ID"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return DEFAULT_MODEL_REF


def _resolve_asr_model_ref() -> str:
    return DEFAULT_ASR_MODEL_REF


def _resolve_zipenhancer_model_ref() -> str:
    for env_name in ("ZIPENHANCER_MODEL_ID", "ZIPENHANCER_MODEL_PATH"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return DEFAULT_ZIPENHANCER_MODEL


def _log_runtime_diagnostics_once() -> None:
    global _runtime_diag_logged
    if _runtime_diag_logged:
        return

    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cxx11abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "model_ref": _resolve_model_ref(),
        "devices": _get_devices_env(),
    }
    logger.info(f"Runtime diagnostics: {info}")
    _runtime_diag_logged = True


class _ZipEnhancer:
    def __init__(self, model_ref: str):
        import torchaudio
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self._torchaudio = torchaudio
        self.model_ref = model_ref
        self._pipeline = pipeline(Tasks.acoustic_noise_suppression, model=model_ref)

    def _normalize_loudness(self, wav_path: str) -> None:
        audio, sr = self._torchaudio.load(wav_path)
        loudness = self._torchaudio.functional.loudness(audio, sr)
        normalized_audio = self._torchaudio.functional.gain(audio, -20 - loudness)
        self._torchaudio.save(wav_path, normalized_audio, sr)

    def enhance(self, input_path: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name
        try:
            self._pipeline(input_path, output_path=output_path)
            self._normalize_loudness(output_path)
            return output_path
        except Exception:
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise


def get_denoiser():
    global _denoiser
    if _denoiser is not None:
        return _denoiser

    with _denoiser_lock:
        if _denoiser is not None:
            return _denoiser

        model_ref = _resolve_zipenhancer_model_ref()
        logger.info(f"Loading ZipEnhancer denoiser from {model_ref} ...")
        _denoiser = _ZipEnhancer(model_ref)
        logger.info("ZipEnhancer denoiser loaded.")
    return _denoiser


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


def _get_audio_duration_seconds(audio_path: str) -> float:
    import soundfile as sf

    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def _begin_generation_request() -> None:
    global _active_generation_requests
    with _active_generation_lock:
        _active_generation_requests += 1


def _end_generation_request() -> None:
    global _active_generation_requests
    with _active_generation_lock:
        _active_generation_requests = max(0, _active_generation_requests - 1)


def _get_active_generation_requests() -> int:
    with _active_generation_lock:
        return _active_generation_requests


def _validate_reference_audio_duration(
    audio_path: str, request: Optional[gr.Request] = None
) -> None:
    duration_seconds = _get_audio_duration_seconds(audio_path)
    if duration_seconds > MAX_REFERENCE_AUDIO_SECONDS:
        raise gr.Error(_get_i18n_text("reference_audio_too_long_error", request))


def _prepare_audio_for_encoding(
    audio_path: Optional[str],
    *,
    denoise: bool,
    request: Optional[gr.Request] = None,
) -> tuple[bytes | None, str | None, Optional[str]]:
    if audio_path is None or not audio_path.strip():
        return None, None, None

    _validate_reference_audio_duration(audio_path, request)

    source_path = audio_path
    temp_path = None
    if denoise:
        logger.info("Applying ZipEnhancer denoising to reference audio ...")
        acquired = _denoise_semaphore.acquire(timeout=30)
        if not acquired:
            raise gr.Error(_get_i18n_text("denoise_busy_error", request))
        try:
            temp_path = get_denoiser().enhance(audio_path)
            source_path = temp_path
        except Exception as exc:
            logger.exception("ZipEnhancer denoising failed")
            raise gr.Error(_get_i18n_text("denoise_failed_error", request)) from exc
        finally:
            _denoise_semaphore.release()

    audio_bytes, audio_format = _read_audio_bytes(source_path)
    return audio_bytes, audio_format, temp_path


def _safe_prompt_wav_recognition(
    use_prompt_text: bool, prompt_wav: Optional[str], request: Optional[gr.Request] = None
) -> str:
    try:
        return prompt_wav_recognition(use_prompt_text, prompt_wav)
    except Exception as exc:
        logger.warning(f"ASR recognition failed: {exc}")
        raise gr.Error(_get_i18n_text("asr_failed_error", request)) from exc




def _stop_server_if_needed() -> None:
    global _voxcpm_server, _model_info, _voxcpm_pkg_model
    if _voxcpm_pkg_model is not None:
        try:
            del _voxcpm_pkg_model
        except Exception:
            pass
        _voxcpm_pkg_model = None
    if _voxcpm_server is None:
        return

    if isinstance(_voxcpm_server, _AsyncServerBridge):
        _voxcpm_server.stop()
    else:
        stop = getattr(_voxcpm_server, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:
                logger.warning(f"Failed to stop nano-vLLM server cleanly: {exc}")

    _voxcpm_server = None
    _model_info = None


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
    "Note: This mode will disable Control Instruction.\n\n"
    "### [A Voice Chef's Guide to VoxCPM2](https://voxcpm.readthedocs.io/en/latest/cookbook.html)"
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
    "### [VoxCPM 2 最佳实践指南](https://voxcpm.readthedocs.io/en/latest/cookbook.html)\n\n"
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
        "reference_audio_too_long_error": "Reference audio is too long. Please upload audio no longer than 50 seconds.",
        "denoise_busy_error": "Too many reference-audio enhancement requests are running. Please try again in a moment.",
        "denoise_failed_error": "Reference audio enhancement failed. Please try disabling denoise or use a cleaner clip.",
        "backend_retry_error": "The backend is temporarily unstable. Please try again in a moment.",
        "asr_failed_error": "ASR failed. Please fill the transcript manually or try another reference audio.",
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
        "reference_audio_too_long_error": "参考音频太长了，请上传不超过 50 秒的音频。",
        "denoise_busy_error": "当前参考音频降噪请求过多，请稍后再试。",
        "denoise_failed_error": "参考音频降噪失败，请尝试关闭降噪或更换更干净的音频。",
        "backend_retry_error": "后端暂时不稳定，请稍后再试。",
        "asr_failed_error": "ASR 识别失败，请手动填写参考音频文本，或更换一段参考音频后重试。",
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


def _resolve_ui_language(request: Optional[gr.Request] = None) -> str:
    if request is None:
        return "en"
    accept_language = str(request.headers.get("accept-language", "")).lower()
    if accept_language.startswith("zh"):
        return "zh-CN"
    return "en"


def _get_i18n_text(key: str, request: Optional[gr.Request] = None) -> str:
    locale = _resolve_ui_language(request)
    return _I18N_TRANSLATIONS.get(locale, _I18N_TRANSLATIONS["en"]).get(
        key, _I18N_TRANSLATIONS["en"].get(key, key)
    )


def _append_request_log(payload: dict) -> None:
    if _request_log_dir is None:
        return

    now = datetime.now(timezone.utc)
    record = {"timestamp": now.isoformat(), **payload}
    log_path = _request_log_dir / f"{now.date().isoformat()}.jsonl"
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")

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
    if _asr_model is not None:
        return _asr_model
    with _asr_lock:
        if _asr_model is not None:
            return _asr_model
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        device = os.environ.get("ASR_DEVICE", "cpu").strip() or "cpu"
        asr_model_ref = _resolve_asr_model_ref()
        logger.info(f"Downloading ASR model from Hugging Face: {asr_model_ref}")
        asr_model_path = snapshot_download(repo_id=asr_model_ref)
        logger.info(f"Loading ASR model on {device} ...")
        _asr_model = AutoModel(
            model=asr_model_path,
            disable_update=True,
            log_level="INFO",
            device=device,
        )
        logger.info("ASR model loaded.")
    return _asr_model


class _AsyncServerBridge:
    """Thread-safe bridge to AsyncVoxCPM2ServerPool running in a dedicated event loop."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self._server_pool = None
        self._model_info: Optional[dict] = None
        self._closed = False

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self) -> None:
        _log_runtime_diagnostics_once()
        model_ref = _resolve_model_ref()
        logger.info(f"Loading nano-vLLM VoxCPM async server from {model_ref} ...")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, name="nanovllm-event-loop", daemon=True)
        self._thread.start()

        try:
            async def _init():
                from nanovllm_voxcpm import VoxCPM

                pool = VoxCPM.from_pretrained(
                    model=model_ref,
                    max_num_batched_tokens=_get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS", 8192),
                    max_num_seqs=_get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS", 16),
                    max_model_len=_get_int_env("NANOVLLM_SERVERPOOL_MAX_MODEL_LEN", 4096),
                    gpu_memory_utilization=_get_float_env("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION", 0.95),
                    enforce_eager=_get_bool_env("NANOVLLM_SERVERPOOL_ENFORCE_EAGER", False),
                    devices=_get_devices_env(),
                )
                await pool.wait_for_ready()
                return pool

            future = asyncio.run_coroutine_threadsafe(_init(), self._loop)
            self._server_pool = future.result()

            info_future = asyncio.run_coroutine_threadsafe(
                self._server_pool.get_model_info(), self._loop
            )
            self._model_info = info_future.result()
            logger.info(f"nano-vLLM async server loaded: {self._model_info}")
        except Exception:
            self.stop()
            raise

    def get_model_info(self) -> dict:
        assert self._model_info is not None
        return self._model_info

    def encode_latents(self, wav: bytes, wav_format: str, timeout: float = 120) -> bytes:
        if self._closed:
            raise RuntimeError("nano-vLLM bridge is closed")
        assert self._loop is not None and self._server_pool is not None
        future = asyncio.run_coroutine_threadsafe(
            self._server_pool.encode_latents(wav, wav_format), self._loop
        )
        try:
            return future.result(timeout=timeout)
        finally:
            if not future.done():
                future.cancel()

    def generate(self, timeout: float = 300, **kwargs):
        if self._closed:
            raise RuntimeError("nano-vLLM bridge is closed")
        assert self._loop is not None and self._server_pool is not None
        result_queue: queue.Queue = queue.Queue()
        import time as _time

        async def _drain():
            try:
                async for chunk in self._server_pool.generate(**kwargs):
                    result_queue.put(chunk)
                result_queue.put(None)
            except Exception as exc:
                result_queue.put(exc)

        deadline = _time.monotonic() + timeout
        future = asyncio.run_coroutine_threadsafe(_drain(), self._loop)
        try:
            while True:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Generation exceeded {timeout}s timeout")
                try:
                    item = result_queue.get(timeout=min(0.5, remaining))
                except queue.Empty:
                    if future.done():
                        exc = future.exception()
                        if exc is not None:
                            raise exc
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not future.done():
                future.cancel()

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._loop is not None and self._server_pool is not None:
                future = asyncio.run_coroutine_threadsafe(self._server_pool.stop(), self._loop)
                future.result(timeout=10)
        except Exception as exc:
            logger.warning(f"Failed to stop async server pool cleanly: {exc}")
        finally:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            if (
                self._loop is not None
                and not self._loop.is_closed()
                and (self._thread is None or not self._thread.is_alive())
            ):
                self._loop.close()
            self._server_pool = None
            self._model_info = None
            self._thread = None
            self._loop = None


def get_voxcpm_server() -> _AsyncServerBridge:
    global _voxcpm_server, _model_info
    if _voxcpm_server is not None:
        return _voxcpm_server

    with _server_lock:
        if _voxcpm_server is not None:
            return _voxcpm_server

        bridge = _AsyncServerBridge()
        bridge.start()
        _voxcpm_server = bridge
        _model_info = bridge.get_model_info()
    return _voxcpm_server


def get_model_info() -> dict:
    global _model_info
    if _use_native_voxcpm_backend():
        m = _get_voxcpm_pkg_model()
        return {"sample_rate": int(m.tts_model.sample_rate)}
    if _model_info is None:
        get_voxcpm_server()
    assert _model_info is not None
    return _model_info


def _prewarm_backend() -> None:
    try:
        logger.info("Starting backend prewarm ...")
        if _use_native_voxcpm_backend():
            _get_voxcpm_pkg_model()
        else:
            get_voxcpm_server()
        logger.info("Backend prewarm completed.")
    except Exception as exc:
        logger.warning(f"Backend prewarm failed: {exc}")


def _start_background_prewarm() -> None:
    global _prewarm_started
    if not _get_bool_env("NANOVLLM_PREWARM", True):
        return

    with _prewarm_lock:
        if _prewarm_started:
            return
        _prewarm_started = True

    Thread(target=_prewarm_backend, name="nanovllm-prewarm", daemon=True).start()


# ---------- GPU-accelerated inference ----------


def prompt_wav_recognition(use_prompt_text: bool, prompt_wav: Optional[str]) -> str:
    if not use_prompt_text or prompt_wav is None or not prompt_wav.strip():
        return ""

    asr_model = get_asr_model()
    res = asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
    return _extract_asr_text(res)


def _float_audio_to_int16(wav: np.ndarray) -> np.ndarray:
    clipped = np.clip(wav, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16, copy=False)


def _generate_tts_audio_once_native(
    text_input: str,
    control_instruction: str = "",
    reference_wav_path_input: Optional[str] = None,
    use_prompt_text: bool = False,
    prompt_text_input: str = "",
    cfg_value_input: float = 2.0,
    do_normalize: bool = True,
    denoise: bool = True,
    request: Optional[gr.Request] = None,
) -> Tuple[int, np.ndarray]:
    temp_audio_path = None
    try:
        model = _get_voxcpm_pkg_model()
        model_info = get_model_info()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        final_text = f"({control}){text}" if control and not use_prompt_text else text

        audio_bytes, audio_format, temp_audio_path = _prepare_audio_for_encoding(
            reference_wav_path_input,
            denoise=bool(denoise),
            request=request,
        )
        prompt_text_clean = (prompt_text_input or "").strip()
        if use_prompt_text and audio_bytes is None:
            raise ValueError("Ultimate Cloning Mode requires a reference audio clip.")
        if use_prompt_text and not prompt_text_clean:
            raise ValueError(
                "Ultimate Cloning Mode requires a transcript. Please wait for ASR or fill it in manually."
            )
        if not use_prompt_text:
            prompt_text_clean = ""

        ref_path: Optional[str] = None
        if audio_bytes is not None:
            ref_path = (
                temp_audio_path
                if temp_audio_path
                else (reference_wav_path_input or "").strip()
            )

        gen_kw = dict(
            cfg_value=float(cfg_value_input),
            inference_timesteps=_get_int_env("NANOVLLM_INFERENCE_TIMESTEPS", 10),
            normalize=bool(do_normalize),
            denoise=False,
            max_len=_get_int_env("NANOVLLM_MAX_GENERATE_LENGTH", 2000),
        )

        chunks: list[np.ndarray] = []
        logger.info(f"Generating: '{final_text[:80]}...'")
        if ref_path is None:
            stream = model.generate_streaming(text=final_text, **gen_kw)
        elif use_prompt_text:
            stream = model.generate_streaming(
                text=final_text,
                prompt_wav_path=ref_path,
                prompt_text=prompt_text_clean,
                reference_wav_path=ref_path,
                **gen_kw,
            )
        else:
            stream = model.generate_streaming(
                text=final_text,
                reference_wav_path=ref_path,
                **gen_kw,
            )
        for chunk in stream:
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("The model returned no audio chunks.")

        wav = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        wav = _float_audio_to_int16(wav)
        return (int(model_info["sample_rate"]), wav)
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass


def _generate_tts_audio_once(
    text_input: str,
    control_instruction: str = "",
    reference_wav_path_input: Optional[str] = None,
    use_prompt_text: bool = False,
    prompt_text_input: str = "",
    cfg_value_input: float = 2.0,
    do_normalize: bool = True,
    denoise: bool = True,
    request: Optional[gr.Request] = None,
) -> Tuple[int, np.ndarray]:
    if _use_native_voxcpm_backend():
        return _generate_tts_audio_once_native(
            text_input=text_input,
            control_instruction=control_instruction,
            reference_wav_path_input=reference_wav_path_input,
            use_prompt_text=use_prompt_text,
            prompt_text_input=prompt_text_input,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            request=request,
        )
    temp_audio_path = None
    try:
        server = get_voxcpm_server()
        model_info = get_model_info()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        final_text = f"({control}){text}" if control and not use_prompt_text else text

        audio_bytes, audio_format, temp_audio_path = _prepare_audio_for_encoding(
            reference_wav_path_input,
            denoise=bool(denoise),
            request=request,
        )
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
            logger.info(
                "Ignoring normalize option: nano-vLLM backend does not support per-request text normalization."
            )

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
        wav = _float_audio_to_int16(wav)
        return (int(model_info["sample_rate"]), wav)
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass


def generate_tts_audio(
    text_input: str,
    control_instruction: str = "",
    reference_wav_path_input: Optional[str] = None,
    use_prompt_text: bool = False,
    prompt_text_input: str = "",
    cfg_value_input: float = 2.0,
    do_normalize: bool = True,
    denoise: bool = True,
    request: Optional[gr.Request] = None,
) -> Tuple[int, np.ndarray]:
    _begin_generation_request()
    request_payload = {
        "event": "tts_request",
        "ui_language": _resolve_ui_language(request),
        "text": (text_input or "").strip(),
        "control_instruction": (control_instruction or "").strip(),
        "use_prompt_text": bool(use_prompt_text),
        "prompt_text": (prompt_text_input or "").strip(),
        "cfg_value": float(cfg_value_input),
        "do_normalize": bool(do_normalize),
        "denoise": bool(denoise),
        "has_reference_audio": bool(reference_wav_path_input and reference_wav_path_input.strip()),
    }
    if request_payload["has_reference_audio"]:
        try:
            request_payload["reference_audio_duration_seconds"] = round(
                _get_audio_duration_seconds(reference_wav_path_input), 3
            )
        except Exception as exc:
            request_payload["reference_audio_duration_error"] = str(exc)

    try:
        try:
            result = _generate_tts_audio_once(
                text_input=text_input,
                control_instruction=control_instruction,
                reference_wav_path_input=reference_wav_path_input,
                use_prompt_text=use_prompt_text,
                prompt_text_input=prompt_text_input,
                cfg_value_input=cfg_value_input,
                do_normalize=do_normalize,
                denoise=denoise,
                request=request,
            )
            try:
                _append_request_log({**request_payload, "status": "success"})
            except Exception as exc:
                logger.warning(f"Failed to append request log: {exc}")
            return result
        except (ValueError, gr.Error) as exc:
            try:
                _append_request_log(
                    {**request_payload, "status": "rejected", "error": str(exc)}
                )
            except Exception as log_exc:
                logger.warning(f"Failed to append request log: {log_exc}")
            if isinstance(exc, gr.Error):
                raise
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            logger.exception("Generation failed")
            try:
                _append_request_log({**request_payload, "status": "error", "error": str(exc)})
            except Exception as log_exc:
                logger.warning(f"Failed to append request log: {log_exc}")

            active_requests = _get_active_generation_requests()
            if active_requests > 1:
                logger.warning(
                    "Generation failed with %s active requests; skipping shared backend restart: %s",
                    active_requests,
                    exc,
                )
                raise gr.Error(_get_i18n_text("backend_retry_error", request)) from exc

            logger.warning(f"Generation failed, restarting backend and retrying once: {exc}")
            with _server_lock:
                _stop_server_if_needed()
            try:
                result = _generate_tts_audio_once(
                    text_input=text_input,
                    control_instruction=control_instruction,
                    reference_wav_path_input=reference_wav_path_input,
                    use_prompt_text=use_prompt_text,
                    prompt_text_input=prompt_text_input,
                    cfg_value_input=cfg_value_input,
                    do_normalize=do_normalize,
                    denoise=denoise,
                    request=request,
                )
                try:
                    _append_request_log({**request_payload, "status": "success_after_retry"})
                except Exception as log_exc:
                    logger.warning(f"Failed to append request log: {log_exc}")
                return result
            except Exception as retry_exc:
                logger.exception("Retry failed")
                try:
                    _append_request_log(
                        {**request_payload, "status": "retry_failed", "error": str(retry_exc)}
                    )
                except Exception as log_exc:
                    logger.warning(f"Failed to append request log: {log_exc}")
                raise gr.Error(_get_i18n_text("backend_retry_error", request)) from retry_exc
    finally:
        _end_generation_request()


# ---------- UI ----------


def create_demo_interface():
    assets_dir = Path(__file__).resolve().parent / "assets"
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

    def _run_asr_if_needed(checked, audio_path, request: gr.Request = None):
        if not checked or not audio_path:
            return gr.update()
        logger.info("Running ASR on reference audio...")
        asr_text = _safe_prompt_wav_recognition(True, audio_path, request=request)
        logger.info(f"ASR result: {asr_text[:60]}...")
        return gr.update(
            value=asr_text,
            placeholder=_get_i18n_text("prompt_text_placeholder", request),
        )

    with gr.Blocks() as interface:
        logo_file = assets_dir / "voxcpm_logo.png"
        if logo_file.is_file():
            _logo_bytes = logo_file.read_bytes()
            if len(_logo_bytes) >= 8 and _logo_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                gr.Image(
                    value=str(logo_file.resolve()),
                    label="",
                    show_label=False,
                    format="png",
                    height=80,
                    sources=[],
                    interactive=False,
                    container=False,
                    buttons=[],
                    elem_classes=["logo-container"],
                    min_width=120,
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
    _start_background_prewarm()
    interface.queue(
        max_size=_get_int_env("GRADIO_QUEUE_MAX_SIZE", 10),
        default_concurrency_limit=_get_int_env("GRADIO_DEFAULT_CONCURRENCY_LIMIT", 4),
    ).launch(
        server_name=server_name,
        server_port=int(os.environ.get("PORT", server_port)),
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
        ssr_mode=_get_bool_env("GRADIO_SSR_MODE", False),
    )


def _running_in_project_venv() -> bool:
    venv_root = Path(__file__).resolve().parent / ".venv"
    try:
        return Path(sys.prefix).resolve() == venv_root.resolve()
    except OSError:
        return False


def _ensure_runtime_deps() -> None:
    if not _use_native_voxcpm_backend():
        return
    try:
        import voxcpm  # noqa: F401
    except ImportError:
        venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
        if (
            venv_python.is_file()
            and not os.environ.get("_VOXCPM_DEMO_REEXEC")
            and not _running_in_project_venv()
        ):
            os.environ["_VOXCPM_DEMO_REEXEC"] = "1"
            app_main = str(Path(__file__).resolve())
            os.execv(str(venv_python), [str(venv_python), app_main, *sys.argv[1:]])
        exe = sys.executable
        print(
            "缺少 voxcpm。\n"
            f"当前 Python: {exe}\n"
            f"请执行: {exe} -m pip install -r requirements.txt\n"
            + (
                f"或使用已有虚拟环境: {venv_python} {Path(__file__).name}\n"
                if venv_python.is_file()
                else ""
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    _ensure_runtime_deps()
    run_demo()
