import os
import sys
import logging
import traceback
import numpy as np
import gradio as gr  
from typing import Optional, Tuple
import soundfile as sf
from pathlib import Path
import requests
import json
import base64
import io
import tempfile
import uuid
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a', encoding='utf-8')
    ]
)

# 控制第三方库的日志级别，避免HTTP请求日志刷屏
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# 启动日志
logger.info("="*50)
logger.info("🚀 VoxCPM应用启动中...")
logger.info(f"Python版本: {sys.version}")
logger.info(f"工作目录: {os.getcwd()}")
logger.info(f"环境变量PORT: {os.environ.get('PORT', '未设置')}")
logger.info(f"环境变量RAY_SERVE_URL: {os.environ.get('RAY_SERVE_URL', '未设置')}")
logger.info("="*50)


class RayServeVoxCPMClient:
    """Client wrapper that talks to Ray Serve TTS API."""

    def __init__(self) -> None:
        logger.info("📡 初始化RayServeVoxCPMClient...")
        
        try:
            # Ray Serve API URL (can be overridden via env)
            self.RAY_SERVE_DEFAULT_URL = "https://d09181959-pytorch251-cuda124-u-5512-sj7yq0o5-8970.550w.link"
            self.api_url = self._resolve_server_url()
            logger.info(f"🔗 准备连接到Ray Serve API: {self.api_url}")
            
            # Test connection
            logger.info("⏳ 测试Ray Serve连接...")
            health_start = time.time()
            health_response = requests.get(f"{self.api_url}/health", timeout=10)
            health_response.raise_for_status()
            health_time = time.time() - health_start
            logger.info(f"✅ 成功连接到Ray Serve API: {self.api_url} (耗时: {health_time:.3f}秒)")
            
        except Exception as e:
            logger.error(f"❌ 初始化RayServeVoxCPMClient失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    # ----------- Helpers -----------
    def _resolve_server_url(self) -> str:
        """Resolve Ray Serve API base URL, prefer env RAY_SERVE_URL."""
        return os.environ.get("RAY_SERVE_URL", self.RAY_SERVE_DEFAULT_URL).rstrip("/")

    def _audio_file_to_base64(self, audio_file_path: str) -> str:
        """
        将音频文件转换为base64编码
        
        Args:
            audio_file_path: 音频文件路径
            
        Returns:
            base64编码的音频数据
        """
        try:
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()
            return base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"音频文件转base64失败: {e}")
            raise
    
    def _base64_to_audio_array(self, base64_audio: str, sample_rate: int = 16000) -> Tuple[int, np.ndarray]:
        """
        将base64编码的音频转换为numpy数组
        
        Args:
            base64_audio: base64编码的音频数据
            sample_rate: 期望的采样率
            
        Returns:
            (sample_rate, audio_array) tuple
        """
        try:
            # 解码base64
            audio_bytes = base64.b64decode(base64_audio)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            # 读取音频文件
            try:
                audio_data, sr = sf.read(tmp_file_path, dtype='float32')
                
                # 转换为单声道
                if audio_data.ndim == 2:
                    audio_data = audio_data[:, 0]
                
                # 转换为int16格式（Gradio期望的格式）
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                return sr, audio_int16
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"base64转音频数组失败: {e}")
            raise
    
    # ----------- Functional endpoints -----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        """Use Ray Serve ASR API for speech recognition."""
        logger.info(f"🎵 开始语音识别，输入文件: {prompt_wav}")
        
        if prompt_wav is None or not prompt_wav.strip():
            logger.info("⚠️  没有提供音频文件，跳过语音识别")
            return ""
        
        try:
            start_time = time.time()
            
            # 将音频文件转换为base64
            convert_start = time.time()
            audio_base64 = self._audio_file_to_base64(prompt_wav)
            convert_time = time.time() - convert_start
            
            # 构建ASR请求 - 匹配 voxcpm_api.py 格式
            asr_request = {
                "audio_data": audio_base64,
                "language": "auto",
                "use_itn": True,
                "reqid": str(uuid.uuid4())
            }
            
            # 调用ASR接口
            api_start = time.time()
            response = requests.post(
                f"{self.api_url}/asr",
                json=asr_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            api_time = time.time() - api_start
            
            result_data = response.json()
            total_time = time.time() - start_time
            
            logger.info(f"⏱️  ASR API请求耗时: {api_time:.3f}秒")
            logger.info(f"⏱️  ASR总耗时: {total_time:.3f}秒")
            logger.info(f"🔍 完整的ASR响应: {result_data}")
            
            # 检查响应状态 - 基于实际响应格式，ASR有多种成功标识
            if isinstance(result_data, dict) and "text" in result_data and (
                result_data.get("code") == 3000 or result_data.get("status") == "ok"
            ):
                recognized_text = result_data.get("text", "")
                logger.info(f"🎯 识别结果: '{recognized_text}'")
                return recognized_text
            else:
                logger.warning(f"⚠️  ASR响应验证失败:")
                if isinstance(result_data, dict):
                    logger.warning(f"   - code字段: {result_data.get('code')}")
                    logger.warning(f"   - 是否有text字段: {'text' in result_data}")
                    logger.warning(f"   - message字段: {result_data.get('message')}")
                logger.warning(f"⚠️  完整ASR响应: {result_data}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ 语音识别失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return ""

    def _call_ray_serve_generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Call Ray Serve /generate API and return (sample_rate, waveform).
        """        
        try:
            start_time = time.time()
            
            # 构建请求数据 - 匹配 voxcpm_api.py 格式
            prepare_start = time.time()
            request_data = {
                "text": text,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
                "do_normalize": do_normalize,
                "denoise": denoise,
                "reqid": str(uuid.uuid4())
            }
            
            # 如果有参考音频和文本，添加到请求中
            if prompt_wav_path and prompt_text:
                logger.info("🎭 使用语音克隆模式")
                convert_start = time.time()
                audio_base64 = self._audio_file_to_base64(prompt_wav_path)
                convert_time = time.time() - convert_start
                
                request_data.update({
                    "prompt_wav": audio_base64,
                    "prompt_text": prompt_text
                })
            else:
                logger.info("🎤 使用默认语音模式")
            prepare_time = time.time() - prepare_start
            
            # 调用生成接口
            api_start = time.time()
            response = requests.post(
                f"{self.api_url}/generate",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=120  # TTS可能需要较长时间
            )
            response.raise_for_status()
            api_time = time.time() - api_start
            
            result_data = response.json()
            
            # 检查响应状态 - 基于实际响应格式，TTS响应没有code字段，只检查data
            if isinstance(result_data, dict) and "data" in result_data and isinstance(result_data["data"], str) and result_data["data"]:
                # 成功生成音频
                audio_base64 = result_data["data"]
                
                # 将base64音频转换为numpy数组
                decode_start = time.time()
                sample_rate, audio_array = self._base64_to_audio_array(audio_base64)
                decode_time = time.time() - decode_start
                total_time = time.time() - start_time
                
                logger.info(f"📈 性能指标: API={api_time:.3f}s, 解码={decode_time:.3f}s, 总计={total_time:.3f}s")
                
                return sample_rate, audio_array
            else:
                logger.error(f"❌ 响应验证失败:")
                logger.error(f"   - 是否为字典: {isinstance(result_data, dict)}")
                if isinstance(result_data, dict):
                    logger.error(f"   - 是否有data字段: {'data' in result_data}")
                    if "data" in result_data:
                        logger.error(f"   - data字段类型: {type(result_data['data'])}")
                        logger.error(f"   - data字段是否为字符串: {isinstance(result_data['data'], str)}")
                        if isinstance(result_data['data'], str):
                            logger.error(f"   - data字段是否非空: {bool(result_data['data'])}")
                            logger.error(f"   - data字段长度: {len(result_data['data'])}")
                logger.error(f"❌ 完整响应内容: {result_data}")
                raise RuntimeError(f"Ray Serve没有返回有效的音频数据。响应: {result_data}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ray Serve请求失败: {e}")
            raise RuntimeError(f"Failed to connect Ray Serve TTS service: {e}. Check RAY_SERVE_URL='{self.api_url}' and service status")
        except Exception as e:
            logger.error(f"❌ Ray Serve调用异常: {e}")
            raise

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        logger.info("🎤 开始TTS音频生成...")
        logger.info(f"📝 输入文本: '{text_input}'")
        logger.info(f"📄 参考文本: '{prompt_text_input}' " if prompt_text_input else "无")
        logger.info(f"⚙️  CFG值: {cfg_value_input}, 推理步数: {inference_timesteps_input}")
        logger.info(f"🔧 文本正则: {do_normalize}, 音频降噪: {denoise}")
        
        try:
            
            text = (text_input or "").strip()
            if len(text) == 0:
                logger.error("❌ 输入文本为空")
                raise ValueError("Please input text to synthesize.")
            
            prompt_wav_path = prompt_wav_path_input or ""
            prompt_text = prompt_text_input or ""
            cfg_value = cfg_value_input if cfg_value_input is not None else 2.0
            inference_timesteps = inference_timesteps_input if inference_timesteps_input is not None else 10
            
            sr, wav_np = self._call_ray_serve_generate(
                text=text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                do_normalize=do_normalize,
                denoise=denoise,
            )
            
            return (sr, wav_np)
            
        except Exception as e:
            logger.error(f"❌ TTS音频生成失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise


# ---------- UI Builders ----------

def create_demo_interface(client: RayServeVoxCPMClient):
    """Build the Gradio UI for Gradio API VoxCPM client."""
    logger.info("🎨 开始创建Gradio界面...")
    
    try:
        assets_path = Path.cwd().absolute()/"assets"
        logger.info(f"📁 设置静态资源路径: {assets_path}")
        gr.set_static_paths(paths=[assets_path])
        logger.info("✅ 静态资源路径设置完成")
    except Exception as e:
        logger.warning(f"⚠️  静态资源路径设置失败: {e}")
        logger.warning("继续创建界面...")

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
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """
    ) as interface:
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
            ### Prompt Speech Enhancement｜参考语音降噪
            - **Enable** to remove background noise for a clean, studio-like voice, with an external ZipEnhancer component.  
              **启用**：通过 ZipEnhancer 组件消除背景噪音，获得更好的音质。
            - **Disable** to preserve the original audio's background atmosphere.  
              **禁用**：保留原始音频的背景环境声，如果想复刻相应声学环境。

            ### Text Normalization｜文本正则化
            - **Enable** to process general text with an external WeTextProcessing component.  
              **启用**：使用 WeTextProcessing 组件，可处理常见文本。
            - **Disable** to use VoxCPM's native text understanding ability. For example, it supports phonemes input ({HH AH0 L OW1}), try it!  
              **禁用**：将使用 VoxCPM 内置的文本理解能力。如，支持音素输入（如 {da4}{jia1}好）和公式符号合成，尝试一下！

            ### CFG Value｜CFG 值
            - **Lower CFG** if the voice prompt sounds strained or expressive.  
              **调低**：如果提示语音听起来不自然或过于夸张。
            - **Higher CFG** for better adherence to the prompt speech style or input text.  
              **调高**：为更好地贴合提示音频的风格或输入文本。

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
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="Prompt Speech Enhancement",
                    elem_id="chk_denoise",
                    info="We use ZipEnhancer model to denoise the prompt audio."
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

        # Wire events
        run_btn.click(
            fn=client.generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value, inference_timesteps, DoNormalizeText, DoDenoisePromptAudio],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
            concurrency_limit=None,
        )
        prompt_wav.change(fn=client.prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text])
        
        logger.info("🔗 事件绑定完成")

    logger.info("✅ Gradio界面构建完成")
    return interface


def run_demo():
    """启动演示应用"""
    logger.info("🚀 开始启动VoxCPM演示应用...")
    
    try:
        # 创建客户端
        logger.info("📡 创建Ray Serve API客户端...")
        client = RayServeVoxCPMClient()
        logger.info("✅ Ray Serve API客户端创建成功")
        
        # 创建界面
        logger.info("🎨 创建Gradio界面...")
        interface = create_demo_interface(client)
        logger.info("✅ Gradio界面创建成功")
        
        # 获取端口配置
        port = int(os.environ.get('PORT', 7860))
        logger.info(f"🌐 准备在端口 {port} 启动服务...")
        
        # 启动应用
        logger.info("🚀 启动Gradio应用...")
        interface.launch(
            server_port=port, 
            server_name="0.0.0.0",
            show_error=True,
        )
        logger.info("✅ 应用启动成功！")
        
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        logger.info("🎬 开始执行主程序...")
        run_demo()
    except KeyboardInterrupt:
        logger.info("⏹️  收到中断信号，正在退出...")
    except Exception as e:
        logger.error(f"💥 主程序异常退出: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        logger.info("🔚 程序结束")