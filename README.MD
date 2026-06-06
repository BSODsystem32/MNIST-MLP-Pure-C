[繁體中文](README.ZH.MD) / [English](README.MD)

# MNIST MLP — Pure C End-to-End Neural Network

![C99](https://img.shields.io/badge/Standard-C99-blue.svg?logo=c)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-orange.svg)
![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end fully-connected neural network written entirely in plain C99 from scratch, with **zero external ML or math library dependencies** (no BLAS, no cuDNN). 

This project bridges the gap between training and edge deployment. It is split into two robust parts:

1. **Training (`src/`)**: A highly optimized training framework featuring BatchNorm, Dropout, SGD with Momentum, and a multi-threaded data augmentation pipeline. Achieves ~98% test accuracy on MNIST.
2. **Inference (`infer/`)**: A bare-metal ready, zero-malloc inference engine that uses **BatchNorm Folding** to aggressively optimize the trained model for constrained edge devices (e.g., ARM Cortex-M microcontrollers).

---

## ✨ Key Features & Optimizations

- **Zero Dependencies**: Pure C99. No external libraries needed.
- **Cache-Friendly BLAS**: Custom matrix multiplication kernels utilizing **3-level Loop Tiling** (I, J, P blocks) to keep working sets within the L1 cache.
- **SIMD & Vectorization**: Replaced standard `rand()` with a lock-free, SIMD-friendly **PCG32** RNG. Heavy loops utilize `#pragma GCC ivdep` to hint compiler AVX2 vectorization.
- **Fast Math Approximations**: Implements a Cephes-style fast polynomial `exp_approx()` for Softmax, matching float precision while significantly boosting speed.
- **BatchNorm Folding**: Merges BN parameters (scale/shift) directly into FC weights/biases during export, reducing inference math operations to just dot products and ReLUs.
- **Lock-Free Augmentation**: Thread-safe image augmentation pipeline using OpenMP.

---

## 🧠 Architecture

**Training Architecture:**
```text
Input (784)
  └─► FC1 (256) → BatchNorm → ReLU → Dropout
        └─► FC2 (128) → BatchNorm → ReLU → Dropout
              └─► FC3 (10) → Softmax → Cross-Entropy Loss
```

**Edge Inference Architecture (Post-BN Folding):**
```text
Input (784, uint8)
  └─► FC1' (256) → ReLU
        └─► FC2' (128) → ReLU
              └─► FC3' (10) → Softmax
```

---

## 🚀 Quick Start

### 1. Download Data and Build
```bash
make download
make all
```

### 2. Train & Export
Runs the full training loop (30 epochs), folds BatchNorm layers, and writes the deployment-ready `models/model.inf`.
```bash
make export
```

### 3. Benchmark
Tests the inference engine on all 10,000 test images and reports accuracy, throughput, and latency:
```bash
make bench
```
*Expected output:*
```text
Accuracy: 98.3x%
Throughput benchmark (10 passes × 10000 samples):
  Total time:  X.XXX s
  Throughput:  XXXXX samples/s
  Latency:     X.XXX ms/sample
```

### 4. Single Image Inference
Test the model on a raw/pgm image (renders an ASCII preview in terminal):
```bash
./infer/infer_single models/model.inf my_image.pgm
```

---

## ⚡ Parallelism & Data Augmentation

### OpenMP Parallelism
Two independent levels of parallelism activate automatically when compiled with `-fopenmp`:
1. **Batch Augmentation**: The inner loop over `BATCH_SIZE` is parallelized. Each thread augments its own slice of the batch using private scratch buffers.
2. **Matrix Operations**: Loops in `mat_mul`, `relu`, `softmax`, etc., are parallelized when problem sizes exceed L1 cache thresholds.

### Thread-Safe Augmentation Pipeline
Each training image is augmented on-the-fly. To eliminate thread synchronization overhead, each OpenMP worker owns a private `ThreadScratch` struct (preventing heap allocation per sample) and an independent **PCG32 RNG** state.

| Transform | Parameter | Default | Description |
|---|---|---|---|
| Translation | `max_shift` | ±2 px | Random X/Y shift (always applied). |
| Rotation | `max_angle_deg` | ±10° | Bilinear interpolation. |
| Scale | `scale_delta` | ±10% | Bilinear interpolation. |
| Elastic | `alpha / sigma` | α=8, σ=3 | Simard (2003) displacement field via box-blur. |
| Noise | `noise_std` | 0.05 | Gaussian noise injection. |

---

## 📂 Project Structure

```text
.
├── Makefile              # Root build orchestrator
├── src/                  # Training Framework
│   ├── main.c            # Training loop, epoch management
│   ├── network.c / .h    # Forward/backward pass, fused ReLU/Dropout, SGD
│   ├── matrix.c / .h     # Loop-tiled, OpenMP-enabled matrix ops
│   ├── mnist.c / .h      # IDX binary loader
│   └── augment.c / .h    # Image augmentations & PCG32 RNG
└── infer/                # Standalone Edge Inference Engine
    ├── infer.c / .h      # Zero-malloc, MCU-ready inference core
    ├── export.c          # Converts training weights to folded .inf
    ├── bench.c           # Accuracy & throughput benchmark
    └── single.c          # CLI utility for single .pgm/.raw inference
```

---

## 🛠️ Edge Deployment (Cross-Compilation)

The inference core (`infer.c` and `infer.h`) is intentionally completely decoupled from the training source. It uses **zero dynamic memory allocation (`malloc`)** in its hot path and accepts raw `uint8_t` pixel arrays.

To cross-compile for a bare-metal ARM Cortex-M7:
```bash
make CC=arm-none-eabi-gcc CFLAGS="-std=c99 -O2 -mthumb -mcpu=cortex-m7" -C infer
```
*Footprint: The `InferNet` struct occupies ~800 KB of Flash; the `InferScratch` working buffer requires < 2 KB of SRAM.*

---

## License
MIT