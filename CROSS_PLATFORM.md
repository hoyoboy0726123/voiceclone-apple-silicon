# VoiceClone 跨平台使用指南

## 支援的平台

本專案支援以下平台：

| 平台 | 設備 | 記憶體需求 | 建議 |
|------|------|-----------|------|
| **Windows/Linux** | NVIDIA GPU (CUDA) | 2-4 GB VRAM (0.6B) / 4-6 GB VRAM (1.7B) | RTX 3060+ |
| **macOS** | Apple Silicon (MPS) | 16+ GB RAM 共享 | M1/M2/M3/M4 Pro/Max |
| **所有平台** | CPU (fallback) | 16+ GB RAM | 不建議，速度很慢 |

---

## 如何運作

### 設備自動檢測

專案會自動偵測你的硬體並選擇最佳設備：

```
app.py 啟動時會：
1. 檢測作業系統 (platform.system())
2. 檢測 CPU 架構 (platform.machine())
3. 檢測 PyTorch 可用設備
   - CUDA (NVIDIA GPU)
   - MPS (Apple Silicon)
   - CPU (fallback)
4. 載入對應的模型
```

### 設備對應表

| 作業系統 | 硬體 | DEVICE 值 | 資料類型 |
|----------|------|-----------|----------|
| Windows/Linux | NVIDIA GPU | `cuda` | bfloat16 |
| macOS (Intel) | CPU | `cpu` | float32 |
| macOS (Apple Silicon) | M1/M2/M3/M4 | `mps` | float32 |

---

## Windows / Linux (NVIDIA GPU) 安裝與執行

### 前置需求

1. **NVIDIA 顯示卡** (RTX 3060 或更新型號)
2. **CUDA Toolkit 12.x** - [下載](https://developer.nvidia.com/cuda-downloads)
3. **Python 3.12+**
4. **uv** - [安裝指南](https://docs.astral.sh/uv/)

### 安裝步驟

```bash
# 1. 複製專案
git clone https://github.com/joshhu/voiceclone.git
cd voiceclone

# 2. 安裝依賴
uv sync

# 3. 啟動 Web UI
uv run python app.py

# 或使用 CLI
uv run python clone.py custom "你好世界" --speaker Vivian
```

### 首次執行

- 第一次執行會自動從 Hugging Face 下載模型
- 模型會儲存在 `~/.cache/huggingface/` 目錄

---

## macOS (Apple Silicon) 安裝與執行

### 前置需求

1. **Apple Silicon Mac** (M1/M2/M3/M4)
2. **macOS 12.0+**
3. **Python 3.12+** (由 uv 自動管理)
4. **uv** - [安裝指南](https://docs.astral.sh/uv/)

### 安裝步驟

```bash
# 1. 複製專案
git clone https://github.com/joshhu/voiceclone.git
cd voiceclone

# 2. 安裝依賴
uv sync

# 3. 啟動 Web UI
uv run python app.py

# 或使用 CLI
uv run python clone.py custom "你好世界" --speaker Vivian
```

### 首次執行

- 專案會自動偵測到 Apple Silicon 並使用 **MPS (Metal Performance Shaders)**
- 首次執行會下載模型（需網路）

---

## 常見問題

### Q: 如何選擇模型大小？

在 Web UI 中可以選擇：
- **0.6B** - 較小、較快、VRAM 需求較低
- **1.7B** - 較大、較慢、但品質更好

### Q: 沒有顯示卡可以運行嗎？

可以，但速度會非常慢。專案會自動 fallback 到 CPU 模式。

### Q: 模型下載位置？

- Windows: `C:\Users\<你的名字>\.cache\huggingface\`
- macOS: `~/.cache/huggingface/`

### Q: 如何更新專案？

```bash
cd voiceclone
git pull
uv sync
```

### Q: macOS 上出現記憶體不足？

1. 關閉其他應用程式
2. 使用 0.6B 模型
3. 避免同時開多個分頁

---

## 效能優化

### Windows (NVIDIA)

```bash
# 安裝 flash-attn 以獲得更快速度 (可選)
pip install flash-attn
```

### macOS (Apple Silicon)

- 使用 0.6B 模型以獲得更好體驗
- 確保 Mac 有足夠可用記憶體

---

## 技術細節

### app.py 設備檢測邏輯

```python
import platform
import torch

IS_MAC = platform.system() == "Darwin" and platform.machine() == "arm64"
DEVICE = "mps" if IS_MAC else ("cuda" if torch.cuda.is_available() else "cpu")
```

### 模型載入策略

- **一次只載入一個模型** - 切換時自動釋放記憶體
- **Whisper 辨識完後釋放** - 給 TTS 騰出空間
- **手動管理記憶體** - `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`

---

## 授權

- 本專案基於 Apache 2.0
- 模型授權請參考 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)

---

## 參考連結

- [Qwen3-TTS 官方](https://github.com/QwenLM/Qwen3-TTS)
- [PyTorch MPS 文件](https://pytorch.org/docs/stable/notes/mps.html)
- [CUDA 安裝指南](https://docs.nvidia.com/cuda/)
- [uv 官網](https://docs.astral.sh/uv/)
