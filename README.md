# VoiceClone - Qwen3-TTS 聲音複製與語音合成系統

> 基於 [joshhu/voiceclone](https://github.com/joshhu/voiceclone) 專案進行跨平台開發
> - 原作者：joshhu
> - 新增功能：macOS Apple Silicon (MPS) / Windows (NVIDIA CUDA) 跨平台支援、字幕生成 (ASR)

基於 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的本地聲音複製與語音合成系統，提供 Gradio 網頁介面和 CLI 命令列工具。

## 功能

| 模式 | 說明 | 模型 |
|------|------|------|
| **Voice Design** | 用自然語言描述設計全新聲音 | 1.7B |
| **Voice Clone** | 上傳 5~15 秒參考音訊，複製該聲音 | 0.6B / 1.7B |
| **CustomVoice TTS** | 9 種預設角色 + 情緒/風格控制 | 0.6B / 1.7B |
| **字幕生成 (ASR)** | 上傳音訊/影片，自動生成字幕 | Whisper |

- 支援 10 種語言：中文、英文、日文、韓文、德文、法文、俄文、葡萄牙文、西班牙文、義大利文
- Voice Clone 上傳音訊後，自動使用 Whisper 辨識參考文字
- VRAM 智慧管理：一次只載入一個模型，切換時自動釋放
- 支援 SRT、VTT、TXT 字幕格式輸出
- 字幕內容可直接在前端編輯並儲存

## 硬體需求

| 模型 | VRAM 需求 | 建議 GPU |
|------|-----------|----------|
| 0.6B | ~2-4 GB | RTX 3060 (12GB) 以上 |
| 1.7B | ~4-6 GB | RTX 3080 Ti (12GB) 以上 |

> 已在 RTX 3080 Ti (12GB VRAM) 上測試通過。

## 快速開始

### 環境建置

#### 方法一：使用 uv（推薦）

`uv` 是 Python 套件管理器，安裝一次後可管理所有 Python 專案：

```powershell
# Windows CMD 安裝 uv（任選一種方式）

# 方式1: 使用 curl
curl -LsSf https://astral.sh/uv/install.bat | cmd

# 方式2: 使用 winget
winget install astral-sh.uv

# 方式3: 直接下載 exe
# 從 https://github.com/astral-sh/uv/releases 下載 uv.exe 放到 PATH 目錄
```

```bash
# macOS / Linux 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Clone 專案
git clone https://github.com/hoyoboy0726123/voiceclone-apple-silicon.git
cd voiceclone-apple-silicon

# 自動安裝所有依賴
uv sync

# 啟動程式
uv run python app.py
```

---

#### 方法二：使用 pip（傳統方式）

```bash
# Clone 專案
git clone https://github.com/hoyoboy0726123/voiceclone-apple-silicon.git
cd voiceclone-apple-silicon

# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 安裝所有依賴
pip install -e .

# 啟動程式
python app.py
```

---

#### 方法三：使用 pip + 直接安裝依賴

```bash
# Clone 專案
git clone https://github.com/hoyoboy0726123/voiceclone-apple-silicon.git
cd voiceclone-apple-silicon

# 直接安裝依賴
pip install gradio openai-whisper qwen-tts soundfile torch moviepy zhconv

# 啟動程式
python app.py
```

---

### 前置軟體安裝

**ffmpeg**（用於處理音訊/影片）：

| 作業系統 | 安裝方式 |
|----------|----------|
| **macOS** | `brew install ffmpeg` |
| **Windows (choco)** | `choco install ffmpeg` |
| **Windows (直接下載)** | 從 [ffmpeg.org](https://ffmpeg.org/download.html) 下載 |
| **Linux** | `sudo apt install ffmpeg` 或 `sudo yum install ffmpeg` |

**Python 3.12+**：
- macOS/Linux: 建議使用 [pyenv](https://github.com/pyenv/pyenv) 管理多版本
- Windows: 從 [python.org](https://www.python.org/downloads/) 下載

首次執行會自動從 Hugging Face 下載模型。

### Web UI（推薦）

```bash
uv run python app.py
```

開啟瀏覽器 http://localhost:7860

### CLI 命令列

```bash
# 預設角色語音合成
uv run python clone.py custom "你好，世界！" --speaker Vivian

# 加上情緒指令
uv run python clone.py custom "你好！" --speaker Serena --instruct "用開心的語氣"

# 聲音複製
uv run python clone.py clone "這是複製的聲音" --ref-audio ref_audio/sample.wav --ref-text "參考音訊文字"

# 使用 0.6B 小模型
uv run python clone.py custom "Hello!" --speaker Ryan --small
```

## 預設角色

| 角色 | 說明 | 語言 |
|------|------|------|
| Vivian | 明亮、略帶鋒利感的年輕女聲 | 中文 |
| Serena | 溫暖、溫柔的年輕女聲 | 中文 |
| Uncle_fu | 低沉醇厚的成熟男聲 | 中文 |
| Dylan | 清亮的年輕男聲 | 北京方言 |
| Eric | 活潑帶沙啞的男聲 | 四川方言 |
| Ryan | 富有節奏感的動感男聲 | 英文 |
| Aiden | 陽光的美式男聲 | 英文 |
| Ono_anna | 俏皮活潑的女聲 | 日文 |
| Sohee | 情感豐富溫暖的女聲 | 韓文 |

## 專案結構

```
voiceclone/
├── app.py           # Gradio 網頁介面
├── clone.py         # CLI 命令列工具
├── outputs/         # 合成音訊輸出目錄
├── ref_audio/       # 參考音訊目錄（聲音複製用）
├── pyproject.toml   # 專案設定與依賴
└── uv.lock          # 鎖定依賴版本
```

## 致謝

### 原作者
- **joshhu** - [joshhu/voiceclone](https://github.com/joshhu/voiceclone) 原始專案開發者

### 技術支援
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [OpenAI Whisper](https://github.com/openai/whisper) 語音辨識
- [Gradio](https://gradio.app/) 網頁介面

## 授權

| 元件 | 授權 |
|------|------|
| 本專案程式碼 | [Apache 2.0](LICENSE) |
| Qwen3-TTS 模型 | [Qwen3-TTS 授權](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |
| Whisper | [MIT](https://github.com/openai/whisper) |

本專案基於 Apache 2.0 授權開源，請遵守相關授權條款。
