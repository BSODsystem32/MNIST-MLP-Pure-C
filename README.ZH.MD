[繁體中文](README.ZH.MD) / [English](README.MD)

# MNIST MLP — 純 C 語言端到端神經網路 (訓練 + 邊緣推論)

![C99](https://img.shields.io/badge/Standard-C99-blue.svg?logo=c)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-orange.svg)
![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

這是一個以純 C99 從頭實作的完整神經網路專案，**完全不依賴任何外部機器學習或數學函式庫**（無 BLAS、無 cuDNN）。

本專案完美銜接了模型訓練與邊緣裝置部署，包含兩個獨立核心：

1. **訓練端 (`src/`)**：具備 BatchNorm、Dropout、SGD Momentum 等機制，結合多執行緒資料增強流程，能在 MNIST 上穩定訓練出約 98% 準確率。
2. **推論端 (`infer/`)**：透過 **BatchNorm 摺疊 (BN Folding)** 技術壓縮模型，提供無動態記憶體配置 (Zero-malloc)、可直接於微控制器 (如 ARM Cortex-M) 裸機運行的極輕量推論引擎。

---

## ✨ 核心特色與底層最佳化

- **零依賴 (Zero Dependencies)**：純 C99 開發，開箱即用。
- **快取友善 (Cache-Friendly)**：客製化矩陣乘法，採用 **3 層迴圈平鋪 (Loop Tiling)**，確保資料存取維持在 L1 Cache 內，大幅降低 Cache Miss。
- **SIMD 與向量化**：棄用傳統 `rand()`，改採無鎖且適合 SIMD 的 **PCG32** 亂數引擎；密集運算迴圈加入 `#pragma GCC ivdep`，引導編譯器進行 AVX2 向量化。
- **高速數學近似**：實作 Cephes 風格的快速多項式 `exp_approx()` 供 Softmax 使用，在維持浮點精度的同時極大化運算速度。
- **BatchNorm 摺疊**：匯出模型時，將 BN 的縮放與平移參數直接「摺疊」進全連接層的權重中，讓推論時只剩下純粹的內積與 ReLU 運算。
- **無鎖平行增強**：運用 OpenMP 打造 Thread-safe 的影像增強管線，榨乾多核 CPU 效能。

---

## 🧠 網路架構

**訓練階段架構：**
```text
輸入層 (784)
  └─► FC1 (256) → BatchNorm → ReLU → Dropout
        └─► FC2 (128) → BatchNorm → ReLU → Dropout
              └─► FC3 (10) → Softmax → Cross-Entropy Loss
```

**邊緣推論架構 (BN 摺疊後)：**
```text
輸入層 (784, uint8)
  └─► FC1' (256) → ReLU
        └─► FC2' (128) → ReLU
              └─► FC3' (10) → Softmax
```

---

## 🚀 快速開始

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
測試單張 .pgm 或 raw 圖片（會在終端機輸出 ASCII 預覽圖）：
```bash
./infer/infer_single models/model.inf my_image.pgm
```

---

## ⚡ 平行運算與資料增強 (Data Augmentation)

### OpenMP 雙層平行化
編譯時加入 `-fopenmp` 即可自動啟用兩個獨立的平行層級：
1. **批次增強層級**：對 `BATCH_SIZE` 進行平行處理。每條執行緒擁有專屬的暫存區，互不干擾。
2. **矩陣運算層級**：當矩陣規模超越 L1 快取閾值時，自動對 `mat_mul`、`relu` 等運算套用多執行緒加速。

### 執行緒安全的增強管線 (Thread-Safe Pipeline)
每張圖片在送入網路前都會即時進行變換。為了達到零同步開銷 (Zero-synchronization overhead)，每條 OpenMP 執行緒皆擁有私有的 `ThreadScratch`（避免熱路徑使用 `malloc`）與獨立的 **PCG32 亂數狀態**。

| 變換 | 參數 | 預設值 | 說明 |
|---|---|---|---|
| 隨機平移 | `max_shift` | ±2 px | 必定執行。 |
| 隨機旋轉 | `max_angle_deg` | ±10° | 使用雙線性插值。 |
| 隨機縮放 | `scale_delta` | ±10% | 使用雙線性插值。 |
| 彈性形變 | `alpha / sigma` | α=8, σ=3 | 依據 Simard (2003) 方法，利用 Box-blur 生成位移場。 |
| 高斯雜訊 | `noise_std` | 0.05 | 注入常態分配雜訊。 |

---

## 📂 專案結構

```text
.
├── Makefile              # 專案總編譯腳本
├── src/                  # 訓練框架
│   ├── main.c            # 訓練迴圈、Epoch 管理
│   ├── network.c / .h    # 前向/反向傳播、融合 ReLU/Dropout、SGD
│   ├── matrix.c / .h     # Cache-tiling 與 OpenMP 矩陣運算
│   ├── mnist.c / .h      # IDX 二進位格式載入器
│   └── augment.c / .h    # 影像增強與 PCG32 亂數引擎
└── infer/                # 獨立邊緣推論引擎
    ├── infer.c / .h      # 零 malloc、可移植 MCU 的推論核心
    ├── export.c          # 執行 BN 摺疊並匯出 .inf 模型
    ├── bench.c           # 準確率與吞吐量效能測試
    └── single.c          # 單張圖片推論命令列工具
```

---

## 🛠️ 跨平台與微控制器部署 (Edge AI)

推論核心（`infer.c` 與 `infer.h`）被設計為完全去耦合——它不依賴訓練端的程式碼，熱路徑中**沒有任何動態記憶體配置 (Zero-malloc)**，並直接接受 `uint8_t` 原始像素陣列輸入。

若要交叉編譯至 ARM Cortex-M7 裸機環境：
```bash
make CC=arm-none-eabi-gcc CFLAGS="-std=c99 -O2 -mthumb -mcpu=cortex-m7" -C infer
```
*記憶體足跡：`InferNet` 結構體約佔 800 KB Flash 空間；而 `InferScratch` 運算緩衝區僅需不到 2 KB 的 SRAM。*

---

## 授權
MIT