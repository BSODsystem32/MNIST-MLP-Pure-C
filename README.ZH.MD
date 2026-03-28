[繁體中文](README.ZH.MD) / [English](README.MD)

---

# MNIST MLP — 純 C 語言端到端神經網路 (訓練 + 邊緣推論)

這是一個以純 C99 實作的完整神經網路專案，**不依賴任何外部 ML 函式庫**。專案包含兩個獨立部分：
1. **訓練端 (`src/`)**：包含 BatchNorm、Dropout 與 SGD Momentum，能在 MNIST 上訓練出 ~98% 準確率的模型。
2. **推論端 (`infer/`)**：透過 **BatchNorm 摺疊 (BN Folding)** 將模型壓縮，提供無動態記憶體配置 (Zero-malloc)、可於微控制器 (如 ARM Cortex-M) 裸機運行的極輕量推論引擎。

---

## 網路架構

**訓練階段架構：**
```text
輸入層 (784)
  └─► FC1 (256) → BatchNorm → ReLU → Dropout
        └─► FC2 (128) → BatchNorm → ReLU → Dropout
              └─► FC3 (10) → Softmax → Cross-Entropy Loss
```

**推論階段架構 (BN 摺疊後)：**
推論時，BatchNorm 的縮放與平移參數會被預先計算並**摺疊**進全連接層的權重 (Weights) 與偏差值 (Biases) 中。
```text
輸入層 (784, uint8)
  └─► FC1' (256) → ReLU
        └─► FC2' (128) → ReLU
              └─► FC3' (10) → Softmax
```

---

## 檔案與專案結構

```text
.
├── Makefile             # 根目錄 Makefile (統整所有編譯目標)
├── src/                 # 訓練程式碼 (反向傳播、優化器)
│   ├── main.c, network.c, matrix.c, mnist.c
├── infer/               # 獨立的推論引擎與工具 (無相依 src/)
│   ├── infer.c / .h     # 輕量級推論核心 (可直接移植到 MCU)
│   ├── export.c         # 執行 BN 摺疊，轉換訓練權重為推論權重
│   ├── bench.c          # 測試集準確率與吞吐量效能測試
│   └── single.c         # 單張圖片推論 (支援 .pgm / .raw 並輸出 ASCII 預覽)
└── data/                # 存放 MNIST 資料集 (由 make download 自動下載)
```

### 關鍵技術亮點

*   **BatchNorm 摺疊 (`infer_export`)**：讀取包含執行統計數據 (Running Stats) 的訓練權重檔，將 BN 的數學運算吸收進 FC 層，省去推論時的除法與乘法開銷，並將模型大小縮減至約 800 KB (`model.inf`)。
*   **極致輕量的推論引擎 (`infer.c`)**：
    *   完全沒有 `malloc` (熱路徑零動態記憶體配置)。
    *   直接接受 `uint8_t` (0-255) 像素陣列，邊計算邊正規化。
    *   跨平台相容，編譯參數改為 `arm-none-eabi-gcc` 即可直接上板。

---

## 快速開始

### 1. 下載資料集與編譯
此指令會自動下載 MNIST 資料集並編譯所有執行檔 (訓練與推論工具)。
```bash
make download
make all
```

### 2. 一鍵訓練與匯出推論模型
這會執行訓練迴圈 (25 Epochs)，然後自動呼叫 `infer_export` 將權重轉換為部署用的 `models/model.inf`。
```bash
make export
```

### 3. 效能基準測試 (Benchmark)
使用推論引擎跑完 10,000 張測試圖片，並測試吞吐量 (Throughput) 與延遲 (Latency)：
```bash
make bench
```
*預期輸出：*
```text
Accuracy: 98.3x%
Throughput benchmark (10 passes × 10000 samples):
  Total time:  X.XXX s
  Throughput:  XXXXX samples/s
  Latency:     X.XXX ms/sample
```

### 4. 單張圖片測試
支援載入 28x28 的 PGM 圖檔或 Raw 像素檔，並在終端機顯示 ASCII 預覽：
```bash
# 若您有自己的 28x28 灰階圖片
./infer/infer_single models/model.inf my_image.pgm
```

---

## 跨平台與微控制器部署 (Edge AI)

推論端 (`infer.c` 與 `infer.h`) 被設計為可完全獨立運作。
若要部署到 ARM Cortex-M 裸機環境，只需複製這兩個檔案：

```makefile
# 交叉編譯範例 (Cortex-M7)
make CC=arm-none-eabi-gcc CFLAGS="-std=c99 -O2 -mthumb -mcpu=cortex-m7" -C infer
```

整個 `InferNet` 結構體約佔用 800 KB，推論時的 `InferScratch` 工作緩衝區僅需少許 KB，非常適合存放於微控制器的 SRAM 中。

---

## 授權

MIT
