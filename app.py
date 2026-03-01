"""
Qwen3-TTS 聲音複製與語音合成系統（本地版）
參考官方 HuggingFace Spaces Demo 改寫，針對本地 GPU 優化

支援三種模式：
1. Voice Design - 用文字描述設計全新聲音（僅 1.7B）
2. Voice Clone - 用參考音訊複製任意聲音
3. CustomVoice TTS - 使用預設角色聲音 + 情緒/風格控制

針對 RTX 3080 Ti (12GB VRAM) 優化：一次只載入一個模型
"""

import os
import tempfile
import datetime
import platform

import gradio as gr
import numpy as np
import torch
import soundfile as sf
import whisper
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

# ── 設定 ──────────────────────────────────────────
# 檢測是否為 Apple Silicon Mac
IS_MAC = platform.system() == "Darwin" and platform.machine() == "arm64"
DEVICE = "mps" if IS_MAC else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {DEVICE}")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZES = ["0.6B", "1.7B"]

# 檢測設備
print(f"系統: {platform.system()}, 架構: {platform.machine()}")
print(f"使用設備: {DEVICE}")

SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
    "Serena", "Sohee", "Uncle_fu", "Vivian",
]

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian", "Italian",
]

# 全域模型快取（本地 12GB VRAM 一次只保留一個模型）
_current_model = None
_current_model_key = None

# Whisper ASR 模型快取
_whisper_model = None


# ── ASR 語音辨識 ──────────────────────────────────
def _load_whisper():
    """載入 Whisper 模型（使用 turbo，速度快、品質好）"""
    global _whisper_model
    if _whisper_model is None:
        print(f"正在載入 Whisper turbo 模型 (device: {DEVICE})...")
        _whisper_model = whisper.load_model("turbo", device=DEVICE)
        print("Whisper 載入完成")
    return _whisper_model


def _unload_whisper():
    """釋放 Whisper 模型以騰出記憶體給 TTS"""
    global _whisper_model
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()


def transcribe_audio(audio):
    """上傳音訊後自動辨識文字"""
    if audio is None:
        return ""

    # 先釋放 TTS 模型騰出 VRAM
    global _current_model, _current_model_key
    if _current_model is not None:
        del _current_model
        _current_model = None
        _current_model_key = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    try:
        # Gradio Audio type="numpy" 回傳 (sr, wav)
        sr, wav = audio
        wav = _normalize_audio(wav)

        # Whisper 需要檔案路徑，存成暫存檔
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav, sr)
            tmp_path = f.name

        model = _load_whisper()
        result = model.transcribe(tmp_path)
        text = result["text"].strip()

        os.unlink(tmp_path)

        # 辨識完釋放 Whisper，給 TTS 留空間
        _unload_whisper()

        return text
    except Exception as e:
        _unload_whisper()
        return f"[辨識失敗: {e}]"


# ── 模型管理 ──────────────────────────────────────
def get_model_path(model_type: str, model_size: str) -> str:
    """下載並取得模型本地路徑"""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


def load_model(model_type: str, model_size: str):
    """載入模型到 GPU，自動釋放前一個模型"""
    global _current_model, _current_model_key

    key = f"{model_type}-{model_size}"
    if _current_model_key == key and _current_model is not None:
        return _current_model

    # 釋放前一個模型
    if _current_model is not None:
        del _current_model
        _current_model = None
        _current_model_key = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    model_path = get_model_path(model_type, model_size)
    print(f"正在載入模型: {model_type} {model_size} ... (device: {DEVICE})")

    # 根據設備選擇合適的加載方式
    if DEVICE == "mps":
        # Apple Silicon MPS
        _current_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="mps",
            dtype=torch.float32,  # MPS 不完全支援 bfloat16
            attn_implementation="sdpa",
        )
    elif DEVICE == "cuda":
        # NVIDIA GPU
        _current_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        # CPU
        _current_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
    _current_model_key = key
    print(f"模型載入完成: {key}")
    return _current_model


# ── 音訊工具 ──────────────────────────────────────
def _normalize_audio(wav, eps=1e-12, clip=True):
    """正規化音訊為 float32 [-1, 1]"""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"不支援的音訊格式: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """將 Gradio 音訊輸入轉成 (wav, sr) tuple"""
    if audio is None:
        return None

    # Gradio Audio type="numpy" 回傳 (sr, wav)
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _save_output(wavs, sr, prefix="output"):
    """儲存音訊到 outputs 目錄"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f"{prefix}_{ts}.wav")
    sf.write(filepath, wavs[0], sr)
    return filepath


def _gpu_status():
    """取得 GPU 使用狀態"""
    if DEVICE == "mps":
        return "Apple Silicon MPS (Metal)"
    elif torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"VRAM: {used:.1f} / {total:.1f} GB"
    return "CPU 模式"


# ── 生成函式 ──────────────────────────────────────
def generate_voice_design(text, language, voice_description):
    """Voice Design：用自然語言描述設計聲音（僅 1.7B）"""
    if not text or not text.strip():
        return None, "錯誤：請輸入要合成的文字"
    if not voice_description or not voice_description.strip():
        return None, "錯誤：請輸入聲音描述"

    try:
        model = load_model("VoiceDesign", "1.7B")
        wavs, sr = model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        filepath = _save_output(wavs, sr, "design")
        return (sr, wavs[0]), f"合成成功！{_gpu_status()}\n儲存至: {filepath}"
    except Exception as e:
        return None, f"錯誤: {type(e).__name__}: {e}"


def generate_voice_clone(ref_audio, ref_text, target_text, language,
                         use_xvector_only, model_size):
    """Voice Clone：用參考音訊複製聲音"""
    if not target_text or not target_text.strip():
        return None, "錯誤：請輸入要合成的文字"

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "錯誤：請上傳參考音訊"

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "錯誤：請輸入參考音訊的文字內容（或勾選「僅使用 x-vector」）"

    try:
        model = load_model("Base", model_size)
        wavs, sr = model.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        filepath = _save_output(wavs, sr, "clone")
        return (sr, wavs[0]), f"聲音複製成功！{_gpu_status()}\n儲存至: {filepath}"
    except Exception as e:
        return None, f"錯誤: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, model_size):
    """CustomVoice TTS：使用預設角色聲音"""
    if not text or not text.strip():
        return None, "錯誤：請輸入要合成的文字"
    if not speaker:
        return None, "錯誤：請選擇角色"

    try:
        model = load_model("CustomVoice", model_size)
        wavs, sr = model.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        filepath = _save_output(wavs, sr, f"tts_{speaker}")
        return (sr, wavs[0]), f"合成成功！{_gpu_status()}\n儲存至: {filepath}"
    except Exception as e:
        return None, f"錯誤: {type(e).__name__}: {e}"


# ── Gradio 介面 ──────────────────────────────────
def build_ui():
    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(css=css, title="Qwen3-TTS 聲音系統") as demo:
        gr.Markdown("""
# Qwen3-TTS 聲音複製與語音合成系統
三種模式：**Voice Design**（設計聲音）| **Voice Clone**（複製聲音）| **CustomVoice TTS**（預設角色）

模型來源: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
""")

        with gr.Tabs():
            # ── Tab 1: Voice Design ──
            with gr.Tab("Voice Design（聲音設計）"):
                gr.Markdown("### 用自然語言描述來設計全新聲音（僅 1.7B 模型）")
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="合成文字",
                            lines=4,
                            placeholder="輸入要合成的文字...",
                            value="哥哥，你回來啦，人家等了你好久好久了，要抱抱！",
                        )
                        design_language = gr.Dropdown(
                            label="語言", choices=LANGUAGES, value="Auto",
                        )
                        design_instruct = gr.Textbox(
                            label="聲音描述",
                            lines=3,
                            placeholder="描述你想要的聲音特徵...",
                            value="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
                        )
                        design_btn = gr.Button("生成語音", variant="primary")
                    with gr.Column(scale=2):
                        design_audio = gr.Audio(label="合成結果", type="numpy")
                        design_status = gr.Textbox(label="狀態", interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio, design_status],
                )

            # ── Tab 2: Voice Clone ──
            with gr.Tab("Voice Clone（聲音複製）"):
                gr.Markdown("### 上傳參考音訊，複製該聲音來說任何內容")
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="參考音訊（上傳 5~15 秒的聲音樣本）",
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="參考音訊文字（音訊中說的內容）",
                            lines=2,
                            placeholder="上傳音訊後會自動辨識，也可手動修改...",
                        )
                        clone_asr_status = gr.Textbox(
                            label="辨識狀態", interactive=False, visible=False,
                        )
                        clone_xvector = gr.Checkbox(
                            label="僅使用 x-vector（不需要文字，但品質較低）",
                            value=False,
                        )
                    with gr.Column(scale=2):
                        clone_text = gr.Textbox(
                            label="要合成的文字",
                            lines=4,
                            placeholder="輸入要用複製的聲音說的內容...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="語言", choices=LANGUAGES, value="Auto",
                            )
                            clone_model_size = gr.Dropdown(
                                label="模型大小", choices=MODEL_SIZES, value="1.7B",
                            )
                        clone_btn = gr.Button("複製聲音並合成", variant="primary")

                # 上傳音訊後自動辨識文字
                clone_ref_audio.change(
                    fn=transcribe_audio,
                    inputs=[clone_ref_audio],
                    outputs=[clone_ref_text],
                )

                with gr.Row():
                    clone_audio = gr.Audio(label="合成結果", type="numpy")
                    clone_status = gr.Textbox(label="狀態", interactive=False)

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_text,
                            clone_language, clone_xvector, clone_model_size],
                    outputs=[clone_audio, clone_status],
                )

            # ── Tab 3: CustomVoice TTS ──
            with gr.Tab("CustomVoice TTS（預設角色）"):
                gr.Markdown("### 使用預設角色聲音，可加指令控制情緒與風格")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="合成文字",
                            lines=4,
                            placeholder="輸入要合成的文字...",
                            value="你好！歡迎使用語音合成系統，這是預設角色語音的展示。",
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="語言", choices=LANGUAGES, value="Auto",
                            )
                            tts_speaker = gr.Dropdown(
                                label="角色", choices=SPEAKERS, value="Vivian",
                            )
                        with gr.Row():
                            tts_instruct = gr.Textbox(
                                label="風格指令（選填）",
                                lines=2,
                                placeholder="例如：用開心的語氣說、用低沉嚴肅的聲音...",
                            )
                            tts_model_size = gr.Dropdown(
                                label="模型大小", choices=MODEL_SIZES, value="1.7B",
                            )
                        tts_btn = gr.Button("合成語音", variant="primary")
                    with gr.Column(scale=2):
                        tts_audio = gr.Audio(label="合成結果", type="numpy")
                        tts_status = gr.Textbox(label="狀態", interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker,
                            tts_instruct, tts_model_size],
                    outputs=[tts_audio, tts_status],
                )

        gr.Markdown("""
---
**本地部署版** | 12GB VRAM 一次載入一個模型，切換時自動釋放 | 輸出儲存於 `outputs/` 目錄
""")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
