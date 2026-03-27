[繁體中文](README.ZH.MD) / [English](README.MD)

---

# MNIST MLP — 純 C 語言神經網路

以純 C99 實作的全連接神經網路，在 MNIST 上訓練，**不依賴任何外部 ML 函式庫**。透過 BatchNorm、Dropout 與 SGD Momentum，測試準確率可達 ~98%。

---

## 網路架構

```
輸入層 (784)
  └─► FC1 (256) → BatchNorm → ReLU → Dropout
        └─► FC2 (128) → BatchNorm → ReLU → Dropout
              └─► FC3 (10) → Softmax → Cross-Entropy Loss
```

| 層 | 類型 | 輸出維度 |
|----|------|----------|
| 輸入 | 攤平 28×28 | 784 |
| 隱藏層 1 | FC + BN + ReLU + Dropout | 256 |
| 隱藏層 2 | FC + BN + ReLU + Dropout | 128 |
| 輸出層 | FC + Softmax | 10 |

權重初始化採用 **He initialization**（`scale = sqrt(2 / fan_in)`），適合 ReLU 激活函數。

---

## 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `EPOCHS` | 25 | 訓練輪數 |
| `BATCH_SIZE` | 128 | Mini-batch 大小 |
| `LR_INIT` | 0.05 | 初始學習率 |
| `LR_DECAY` | 0.92 | 每輪學習率衰減倍數 |
| `DROPOUT_RATE` | 0.3 | 訓練時神經元遮蔽比例 |
| `MOMENTUM` | 0.9 | SGD Momentum 係數 |
| `BN_EPS` | 1e-5 | BatchNorm 數值穩定常數 |
| `BN_MOMENTUM` | 0.1 | BatchNorm running stats EMA 衰減率 |

學習率採指數衰減：`lr = LR_INIT × LR_DECAY^epoch`。

---

## 檔案結構

```
.
├── main.c       # 程式進入點：訓練迴圈、評估、CLI 參數
├── network.c    # 前向／反向傳播、BN、Dropout、權重存讀
├── network.h    # Network 結構體、層大小、公開 API
├── matrix.c     # 矩陣運算：mat_mul、relu、softmax 等
├── matrix.h     # 矩陣運算宣告
├── mnist.c      # IDX 二進位格式讀取，像素正規化至 [0, 1]
├── mnist.h      # MnistData 結構體
└── Makefile
```

### 關鍵實作說明

**`matrix.c`** — 所有張量運算以 row-major 一維陣列實作。提供三種矩陣乘法變體（`A@B`、`Aᵀ@B`、`A@Bᵀ`），反向傳播時無需顯式轉置。

**`network.c`** — 完整前向與反向傳播：
- `bn_forward` / `bn_backward`：實作 Ioffe & Szegedy (2015) BatchNorm，訓練時以 batch 計算 mean/var，推論時使用 EMA running stats。
- `dropout_forward` / `dropout_backward`：Inverted dropout，訓練時對存活神經元乘以 `1/(1−p)`，推論階段無需額外縮放。
- `MOMENTUM_UPDATE` 巨集：執行 `v = β·v + g`，再更新 `param -= lr·v`。
- `net_save` / `net_load`：將權重與 BN running stats 寫入二進位檔，以 magic number `0x4D4C5002`（`MLP\x02`）標識格式版本。

**`mnist.c`** — 解析 IDX3/IDX1 大端序二進位格式，將像素位元組正規化為 `float32` 的 `[0, 1]` 範圍。

---

## 快速開始

### 1. 下載 MNIST 資料集

```bash
make download
```

自動下載並解壓四個 IDX 檔案至 `./data/`，需要 `curl` 與 `gunzip`。

> 若 LeCun 伺服器無法連線，可從 https://yann.lecun.com/exdb/mnist/ 或其他鏡像手動下載，解壓後放入 `./data/`。

### 2. 編譯

```bash
make
```

需要 GCC（或任何相容 C99 的編譯器）與 `libm`，產生 `mnist_mlp` 執行檔。

### 3. 訓練

```bash
make run
# 等同於：
./mnist_mlp ./data weights_v2.bin
```

**命令列用法：**

```
./mnist_mlp [data_dir] [weights_out]
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `data_dir` | `./data` | 存放四個 IDX 檔案的目錄 |
| `weights_out` | `weights_v2.bin` | 訓練完成後儲存權重的路徑 |

### 4. 預期輸出

```
Loading MNIST from ./data ...
Train: 60000   Test: 10000

Config: epochs=25  batch=128  lr=0.050 decay=0.92  dropout=0.3
        momentum=0.90  BN_eps=1e-05  BN_momentum=0.10

Epoch  LR       Loss       Train%     Test%
----------------------------------------------
1      0.05000  0.3142     90.21      97.43
2      0.04600  0.1823     94.87      97.81
...
25     0.00726  0.0401     98.92      98.3x
----------------------------------------------
Final test accuracy: 98.3x%
Saved weights to weights_v2.bin
Time: XXXXX ms
```

### 5. 清理

```bash
make clean   # 移除 object 檔、執行檔與 weights_v2.bin
```

---

## 相依套件

- C99 相容編譯器（GCC / Clang）
- 標準 C 函式庫（`libc`、`libm`）
- `make download` 需要 `curl` + `gunzip`

不依賴 BLAS、PyTorch 或任何外部 ML 框架。

---

## 平台說明

`main.c` 的計時器在 Windows 使用 `QueryPerformanceCounter`，在 POSIX 系統使用 `clock_gettime(CLOCK_MONOTONIC)`，兩個平台皆可編譯執行。

---

## 授權

MIT
