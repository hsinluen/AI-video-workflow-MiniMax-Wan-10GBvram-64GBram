# AI-video-workflow-MiniMax-Wan-10GBvram-64GBram

> **10GB 顯存極限優化版：雙引擎 (MiniMax-H3 + 阿里 Wan 2.1 14B) 本地 AI 短劇自動化生成工作流**  
> 專為 **NVIDIA RTX 3080 (10GB VRAM) + 32GB/64GB RAM** 打造，內建「文字編碼器快取 (Text Encoder Caching)」與「記憶體分層動態卸載 (Aggressive Offloading)」，開箱即用、預設鎖定 MiniMax-H3 極速生成！

---

## 🌟 核心特點與亮點

1. **預設旗艦引擎：MiniMax-H3 (FL2VA 10.6GB Q4_K_M + FastH3 Turbo 8-step)**
   - 預設採用海螺/MiniMax 開源之首款音畫同步視訊大模型 **MiniMax-H3**。
   - 結合 **4-8 步 Turbo LoRA** 加速蒸餾，在 10GB 顯存下每鏡僅需 2~3 分鐘即可完成 960x544 高畫質採樣。
   - 人物面部一致性極高、微表情細膩、且支援精確的台詞嘴型同步。

2. **熱備替換引擎：阿里 Wan 2.1 (14B I2V Q4_K_M)**
   - 深度整合阿里巴巴 Wan 2.1 14B 視訊生成大模型（Q4_K_M 量化版，10.56GB）。
   - 支援大動態運鏡、電影級光影與高自由度動作渲染。

3. **10GB 顯存極限防爆機制**
   - **低顯存守護模式**：ComfyUI 啟動預設配置 `--lowvram --reserve-vram 1.0 --disable-pinned-memory --disable-async-offload`。
   - **大模型智慧卸載**：32B 規模的 Qwen3-VL 與 UMT5-XXL 文字編碼器採樣時自動常駐 64GB 系統 RAM，計算完後完全讓出 10GB 顯存給 DiT / UNet 擴散網路。
   - **自動音視頻合成與雙規格輸出**：自動生成 Edge TTS 對白語音、混音背景配樂、自動升頻並無失真導出 **1080p** 與 **720p** 雙版本 MP4。

4. **內建兩大校園喜劇完整 DEMO**
   - **Demo 1**：《大學新生的第一堂課：迷航記》（Freshman's First Day: Lost on Campus）
   - **Demo 2**：《普通物理課的角動量守恆驚魂記》（General Physics: Conservation of Angular Momentum Adventure）

---

## 🖥️ 硬體規格需求 (Hardware Specifications)

| 項目 | 最低需求 (Minimum) | 建議配置 (Recommended) | 本機驗證環境 |
| :--- | :--- | :--- | :--- |
| **GPU 顯示卡** | NVIDIA RTX 3060 / 2080 Ti (10GB+ VRAM) | NVIDIA RTX 3080 / 4070 (10GB~12GB VRAM) | **NVIDIA RTX 3080 (10GB VRAM)** |
| **系統記憶體 (RAM)**| 32 GB DDR4/DDR5 | 64 GB DDR4/DDR5 | **64 GB DDR4** (無分頁檔延遲) |
| **處理器 (CPU)** | Intel i7 10代 / AMD Ryzen 3700X | Intel i7 12代+ / AMD Ryzen 5800X+ | Intel Core i7 / AMD Ryzen 9 |
| **儲存空間** | 60 GB 可用空間 (NVMe SSD) | 100 GB 可用空間 (PCIe 4.0 SSD) | PCIe NVMe SSD |
| **作業系統** | Windows 10/11 64-bit 或 Linux | Windows 11 64-bit | Windows 11 64-bit |
| **CUDA 環境** | CUDA 12.1+ | CUDA 12.4+ | CUDA 12.4 / PyTorch 2.6.0 |

---

## 📦 模型權重與配置目錄結構

請將模型權重放置於 `ComfyUI/models/` 對應目錄中：

```text
ComfyUI/models/
├── diffusion_models/
│   ├── minimax_h3_fl2va_pruned-Q4_K.gguf          # MiniMax-H3 主模型 (10.64 GB)
│   ├── wan2.1-i2v-14b-480p-Q4_K_M.gguf            # Wan 2.1 14B 主模型 (10.56 GB)
│   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors # FastH3 Turbo LoRA (1.82 GB)
├── text_encoders/
│   ├── qwen3vl_32b_minimax_h3-Q2_K_M.gguf         # H3 專用視覺語言編碼器 (12.20 GB)
│   └── umt5-xxl-encoder-Q4_K_M.gguf               # Wan 2.1 專用 T5 編碼器 (3.40 GB)
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors      # H3 視訊 VAE (4.85 GB)
    ├── minimax_h3_audio_vae_fp32.safetensors      # H3 音訊 VAE (0.56 GB)
    └── wan_2.1_vae.safetensors                    # Wan 2.1 視訊 VAE (0.24 GB)
```

> **提示**：本專案已完全剔除 LTX-Video (2B) 與騰訊混元 HunyuanVideo，純化架構並釋放逾 23GB 硬碟空間。

---

## 🚀 快速開始 (Quick Start)

### 步驟 1：啟動 ComfyUI 守護進程 (10GB 低顯存模式)

在終端執行以下啟動命令（或執行 `configs/profile.cmd`）：

```cmd
C:\AI\H3\venv\Scripts\python.exe C:\AI\H3\ComfyUI-0.36.0\main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --lowvram --reserve-vram 1.0 --disable-pinned-memory --disable-async-offload
```

### 步驟 2：執行預設 Demo 1（大學新生迷航記）

```bash
# 預設以 MiniMax-H3 引擎執行全自動製作流程（語音 -> 影片 -> 混音 -> 1080p/720p 剪輯）
cd pipeline/demo_local
python run_demo.py
```

### 步驟 3：執行 Demo 2（普通物理課角動量守恆喜劇）

```bash
cd pipeline/demo2_physics
python run_physics.py
```

---

## 🔄 如何切換模型與自訂 Prompt 製作影片？

本架構支援 **4 種靈活的切換方式**，且**預設為 MiniMax-H3**：

### 方法 1：CLI 命令列參數動態指定 (推薦)

呼叫腳本時加入 `--engine` 參數：

```bash
# 使用預設 MiniMax-H3 (8 步 Turbo 極速生成)
python tools/h3_shot.py path/to/spec.json --out out/shot1 --engine h3

# 切換為阿里 Wan 2.1 14B (30 步高動態生成)
python tools/h3_shot.py path/to/spec.json --out out/shot1 --engine wan2.1
```

### 方法 2：環境變數全域覆蓋

在執行前指定環境變數：

```bash
# Windows PowerShell
$env:VIDEO_ENGINE = "wan2.1"
python make_film.py h3 cut

# Windows CMD
set VIDEO_ENGINE=h3
python make_film.py h3 cut
```

### 方法 3：在劇本設定檔 (`film.json`) 中直接指定

在專案的 `film.json` 頂部設定 `"engine"` 欄位：

```json
{
  "title": "我的自訂短劇",
  "engine": "h3",             // 可填 "h3" 或 "wan2.1"
  "profile": "fasth3_turbo_8step",
  "sampling": {
    "steps": 8,               // h3 設 8 步；wan2.1 設 30 步
    "cfg": 4.0,
    "size": [960, 544]        // h3 建議 [960, 544]；wan2.1 建議 [848, 480]
  }
}
```

### 方法 4：自訂 Prompt 與人物造型定型

在 `film.json` 的 `chars` 與 `shots` 區塊，輕鬆以繁體中文或英文指定人物與畫面演出：

```json
{
  "chars": {
    "prof": {
      "name": "張教授",
      "look": "Prof. Chang, 52-year-old Taiwanese physics professor with silver glasses, light-grey blazer over white shirt",
      "edge_voice": "zh-TW-YunJheNeural"
    }
  },
  "shots": [
    {
      "id": "S1",
      "refs": ["prof"],
      "dur": 4.0,
      "line": {
        "who": "prof",
        "text": "各位同學，當我旋轉這個車輪，角動量守恆會發生什麼事？",
        "at": 0.25
      },
      "action": "Prof. Chang stands at the podium holding a spinning bicycle wheel, smiling enthusiastically at students."
    }
  ]
}
```

---

## 🎬 內建展示成果 (Included Demos)

| DEMO | 主題 | 演員配置 | 核心情節 | 輸出檔案 |
| :--- | :--- | :--- | :--- | :--- |
| **Demo 1** | 《大學新生迷航記》 | 阿凱（大一新生）、晴學姐（大二學姐） | 拿地圖問路卻發現跑到校園對角，學姐熱心借腳踏車解圍。 | `freshman_campus_h3_1080p.mp4`<br>`freshman_campus_h3_720p.mp4` |
| **Demo 2** | 《普通物理課角動量驚魂》 | 張教授（資深物研所長）、阿宇（物理系新生） | 教授坐在旋轉椅上手持高速腳踏車輪翻轉，旋轉椅失控自轉引發全班爆笑。 | `physics_angular_momentum_h3_1080p.mp4`<br>`physics_angular_momentum_h3_720p.mp4` |

---

## 📊 10GB 顯存實測性能指標 (Benchmark on RTX 3080)

| 指標 | MiniMax-H3 (FL2VA + Turbo LoRA) | Alibaba Wan 2.1 (14B I2V) |
| :--- | :--- | :--- |
| **模型權重載入量** | 10.64 GB (Q4_K_M) | 10.56 GB (Q4_K_M) |
| **取樣步數 (Steps)** | **8 步** (FastH3 蒸餾) | **30 步** (全解析度生成) |
| **峰值 VRAM 佔用** | 9.42 GB / 10.0 GB (安全無溢出) | 9.75 GB / 10.0 GB (安全無溢出) |
| **系統 RAM 佔用** | 約 28 GB (Qwen3-VL 文本解碼器常駐) | 約 22 GB (UMT5-XXL 解碼器常駐) |
| **單鏡生成耗時** | **約 2 分 30 秒** | 約 9 分 45 秒 |
| **臉部與對白一致性** | ⭐⭐⭐⭐⭐ (極致穩定，支援 lipsync) | ⭐⭐⭐⭐☆ (動態大，運鏡強烈) |

---

## 📜 授權協議 (License)

本專案採 [MIT License](LICENSE) 授權。模型權重請遵循 MiniMax-H3 與 Alibaba Wan 2.1 之各自開源使用協議。
