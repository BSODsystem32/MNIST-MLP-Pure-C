[繁體中文](README.ZH.MD) / [English](README.MD)

---

# MNIST MLP — 純 C 語言端到端神經網路 (訓練 + 邊緣推論)

這是一個以純 C99 從頭實作的完整神經網路專案，**不依賴任何外部 ML 函式庫**。專案包含兩個獨立部分：

1. **訓練端 (`src/`)**：包含 BatchNorm、Dropout、SGD Momentum，以及完整的資料增強 (Data Augmentation) 流程與 OpenMP 平行運算，能在 MNIST 上訓練出約 98% 準確率的模型。
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

部署時，BatchNorm 的縮放與平移參數會被預先計算並**摺疊**進相鄰全連接層的權重與偏差值中，消除推論時的所有 BN 計算開銷。
```text
輸入層 (784, uint8)
  └─► FC1' (256) → ReLU
        └─► FC2' (128) → ReLU
              └─► FC3' (10) → Softmax
```

---

## 資料增強流程

每張訓練圖片在送入網路前，都會即時 (on-the-fly) 經過獨立的增強處理。流程透過 `AugmentCfg` 結構體設定，依序套用以下變換（平移為必定執行，其餘各有 50% 機率觸發）：

| 變換 | 參數 | 預設值 |
|---|---|---|
| 隨機平移 | `max_shift` | ±2 像素（必定執行） |
| 隨機旋轉 | `max_angle_deg` | ±10° |
| 隨機縮放 | `scale_delta` | ±10% |
| 彈性形變 | `elastic_alpha / sigma` | α=8.0, σ=3.0 |
| 高斯雜訊 | `noise_std` | 標準差 0.05 |

所有幾何變換均採用**雙線性插值 (Bilinear Interpolation)**，邊界以零填充。彈性形變依照 Simard (2003) 方法實作：先生成隨機位移場，再透過多次 Box Blur 平滑，近似高斯核的效果。

### 執行緒安全設計

增強流程專為 OpenMP 平行執行設計。每條工作執行緒擁有獨立的 `ThreadScratch` 結構體，包含：
- `dst[784]` / `tmp[784]` — 圖片暫存緩衝區（不在熱路徑上進行任何堆積記憶體配置）
- `RNG rng` — 各執行緒獨立的亂數產生器

所有執行緒以 `time(NULL)` 為基底、各自導入不同偏移值進行播種，確保跨工作者的增強結果不相關，使整個訓練迴圈**完全執行緒安全**且無需任何鎖。

---

## OpenMP 平行運算

專案採用兩個獨立的平行層級：

**批次增強層級** — 對 `BATCH_SIZE` 張圖片的內層迴圈套用 `#pragma omp parallel for`。每條執行緒使用私有的暫存緩衝區與亂數產生器處理自己負責的那份圖片，同步開銷為零。

**矩陣運算層級** — `mat_mul`、`mat_mul_atb`、`mat_mul_abt`、`relu`、`softmax_rows`、`vec_axpy` 等函式的行迴圈，在問題規模超過閾值時（例如 `m > 32` 或 `n > 1024`）自動套用 `#pragma omp parallel for`。

兩個層級皆在編譯時加入 `-fopenmp` 後自動啟用；若未加入，程式碼可無縫退回至單執行緒執行。

---

## 檔案與專案結構

```text
.
├── Makefile              # 根目錄 Makefile（統整所有編譯目標）
├── src/                  # 訓練程式碼（反向傳播、優化器、資料增強）
│   ├── main.c            # 訓練迴圈、OpenMP 批次增強
│   ├── network.c / .h    # MLP 前向/反向傳播、BatchNorm、Dropout
│   ├── matrix.c / .h     # BLAS 風格矩陣運算（含 OpenMP）
│   ├── mnist.c / .h      # IDX 二進位格式資料載入器
│   └── augment.c / .h    # 資料增強流程 + 各執行緒亂數產生器
└── infer/                # 獨立推論引擎（不依賴 src/）
    ├── infer.c / .h      # 輕量級推論核心（可直接移植到 MCU）
    ├── export.c          # BN 摺疊，轉換訓練權重為 .inf 格式
    ├── bench.c           # 準確率與吞吐量效能基準測試
    └── single.c          # 單張圖片推論（支援 .pgm/.raw，輸出 ASCII 預覽）
```

---

## 快速開始

### 1. 下載資料集與編譯
```bash
make download
make all
```

### 2. 訓練並匯出推論模型
執行完整訓練迴圈 (30 個 Epoch)，接著自動執行 BN 摺疊，生成部署用的 `models/model.inf`。
```bash
make export
```

### 3. 效能基準測試 (Benchmark)
對 10,000 張測試圖片執行推論，測試準確率、吞吐量與延遲：
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

### 4. 單張圖片推論
```bash
./infer/infer_single models/model.inf my_image.pgm
```

---

## 跨平台與微控制器部署 (Edge AI)

推論核心（`infer.c` 與 `infer.h`）刻意設計為完全自包含——不依賴訓練端程式碼、熱路徑無 `malloc`、直接接受 `uint8_t` 原始像素陣列輸入。

交叉編譯至 ARM Cortex-M7 裸機環境：
```bash
make CC=arm-none-eabi-gcc CFLAGS="-std=c99 -O2 -mthumb -mcpu=cortex-m7" -C infer
```

`InferNet` 結構體約佔 800 KB Flash；`InferScratch` 工作緩衝區僅需少許 KB SRAM。

---

## 授權

MIT
