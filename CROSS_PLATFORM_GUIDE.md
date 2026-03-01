# VoiceClone 跨平台使用指南

本專案支援 **Windows (NVIDIA GPU)**、**macOS (Apple Silicon M系列)** 以及 **Linux** 三種平台。

---

## 自動偵測機制

程式會自動偵測您的硬體環境並選擇最佳運算設備：

| 作業系統 | 偵測邏輯 | 使用設備 |
|---------|---------|---------|
| macOS (Apple Silicon) | `platform.system() == "Darwin"` + `platform.machine() == "arm64"` | **MPS (Metal Performance Shaders)** |
| Windows/Linux (NVIDIA GPU) | `torch.cuda.is_available()` | **CUDA** |
| 其他情況 | fallback | **CPU** |

執行時終端機會顯示：
```
使用設備: mps   # macOS Apple Silicon
使用設備: cuda  # Windows/Linux NVIDIA GPU
使用設備: cpu   # CPU 模式
```

---

## Windows + NVIDIA GPU 安裝與執行

### 環境需求

| 項目 | 最低需求 | 建議 |
|-----|---------|------|
| GPU | RTX 3060 12GB | RTX 3080 Ti 12GB+ |
| VRAM | 4 GB | 6 GB |
| 記憶體 | 16 GB | 32 GB |
| 系統 | Windows 10/11 | Windows 11 |

### 安裝步驟

#### 1. 安裝 Python 3.12

```powershell
# 使用 pyenv-win（推薦）
pyenv install 3.12.0
pyenv global 3.12.0

# 或從官網下載：https://www.python.org/downloads/
```

#### 2. 安裝 uv 套件管理器

```powershell
# 方法一：使用 winget（推薦）
winget install astral-sh.uv

# 方法二：使用 PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

#### 3. 安裝 NVIDIA CUDA 驅動程式

```powershell
# 下載並安裝 CUDA Toolkit 12.x
# https://developer.nvidia.com/cuda-downloads

# 確認安裝成功
nvidia-smi
```

#### 4. 克隆與安裝專案

```powershell
git clone https://github.com/joshhu/voiceclone.git
cd voiceclone
uv sync
```

### 執行 Web UI

```powershell
uv run python app.py
```

開啟瀏覽器訪問：http://localhost:7860

### 使用 CLI

```powershell
# 預設角色語音合成
uv run python clone.py custom "你好，世界！" --speaker Vivian

# 聲音複製
uv run python clone.py clone "Hello world!" --ref-audio ref.wav --lang English

# 使用小模型（VRAM 不足時）
uv run python clone.py custom "你好！" --speaker Serena --small
```

---

## macOS (Apple Silicon) 安裝與執行

### 環境需求

| 項目 | 最低需求 | 建議 |
|-----|---------|------|
| 晶片 | M1/M2/M3/M4 | M4 Pro/Max |
| 記憶體 | 16 GB | 24 GB+ |
| 系統 | macOS 12+ | macOS 14+ |

### 安裝步驟

#### 1. 安裝 Homebrew（如已安裝可跳過）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. 安裝 sox（音訊處理）

```bash
brew install sox
```

#### 3. 克隆與安裝專案

```bash
git clone https://github.com/joshhu/voiceclone.git
cd voiceclone
uv sync
```

### 執行 Web UI

```bash
uv run python app.py
```

開啟瀏驅器訪問：http://localhost:7860

### 使用 CLI

```bash
# 預設角色語音合成
uv run python clone.py custom "你好，世界！" --speaker Vivian

# 聲音複製
uv run python clone.py clone "Hello world!" --ref-audio ref.wav --lang English

# 使用小模型
uv run python clone.py custom "你好！" --speaker Serena --small
```

---

## 模型選擇建議

### 模型大小比較

| 模型 | VRAM 需求 | 速度 | 品質 |
|-----|----------|------|------|
| 0.6B | ~2-4 GB | 快 | 較低 |
| 1.7B | ~4-6 GB | 慢 | 較高 |

### 各平台建議

| 平台 | 建議模型 | 原因 |
|-----|---------|------|
| Windows RTX 3080+ | 1.7B | VRAM 充足 |
| Windows RTX 3060 | 0.6B | VRAM 有限 |
| macOS M4 Pro/Max | 0.6B 或 1.7B | MPS 效能較好 |
| macOS M1/M2 | 0.6B | 避免記憶體不足 |
| CPU 模式 | 0.6B | 速度很慢 |

---

## 常見問題

### Q: 如何查看當前使用的設備？

A: 執行應用程式後，終端機會顯示 `使用設備: xxx`。或在 Web UI 的狀態訊息中也會顯示。

### Q: macOS 沒有獨立 GPU，能運行嗎？

A: 可以！Apple Silicon (M1-M4) 有 **Metal Performance Shaders (MPS)**，專為 ML/AI 運算設計。雖然比 NVIDIA GPU 慢，但可以正常運行。

### Q: Windows 沒有 NVIDIA 顯示卡可以運行嗎？

A: 可以使用 CPU 模式，但速度會非常慢。不建議。

### Q: 模型要下載在哪裡？

A: 首次執行時會自動從 Hugging Face 下載。預設快取位置：
- Windows: `C:\Users\<使用者>\.cache\huggingface\`
- macOS/Linux: `~/.cache/huggingface/`

### Q: 如何清理 VRAM / 記憶體？

A: 程式會自動管理。也可重啟應用程式。

---

## 技術細節

### 設備檢測代碼

```python
import platform
import torch

IS_MAC = platform.system() == "Darwin" and platform.machine() == "arm64"
DEVICE = "mps" if IS_MAC else ("cuda" if torch.cuda.is_available() else "cpu")
```

### 數據類型支援

| 設備 | 支援的 dtype |
|-----|-------------|
| CUDA | bfloat16, float16, float32 |
| MPS | float32 (部分 float16) |
| CPU | float32 |

---

## 效能優化 Tips

1. **使用小模型 (0.6B)** - VRAM 有限時
2. **及時釋放模型** - 程式會自動切換時釋放
3. **關閉其他 GPU 程式** - 避免記憶體競爭
4. **使用 SSD** - 模型載入更快

---

## 授權與感謝

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Gradio](https://gradio.app/)

本專案基於 Apache 2.0 授權。
