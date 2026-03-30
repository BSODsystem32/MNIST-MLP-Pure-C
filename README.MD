[繁體中文](README.ZH.MD) / [English](README.MD)

---

# MNIST MLP — Pure C End-to-End Neural Network

An end-to-end fully-connected neural network written entirely in plain C99 from scratch, with **zero external ML library dependencies**. This project is split into two robust parts:

1. **Training (`src/`)**: Features BatchNorm, Dropout, SGD with Momentum, and a full data augmentation pipeline with OpenMP parallelism — achieving ~98% test accuracy on MNIST.
2. **Inference (`infer/`)**: A bare-metal ready, zero-malloc inference engine that uses **BatchNorm Folding** to aggressively optimize the trained model for edge devices (e.g., ARM Cortex-M microcontrollers).

---

## Network Architecture

**Training Architecture:**
```text
Input (784)
  └─► FC1 (256) → BatchNorm → ReLU → Dropout
        └─► FC2 (128) → BatchNorm → ReLU → Dropout
              └─► FC3 (10) → Softmax → Cross-Entropy Loss
```

**Inference Architecture (Post-BN Folding):**

During deployment, the BatchNorm scale and shift parameters are pre-computed and **folded** directly into the adjacent FC weights and biases, eliminating all BN overhead at runtime.
```text
Input (784, uint8)
  └─► FC1' (256) → ReLU
        └─► FC2' (128) → ReLU
              └─► FC3' (10) → Softmax
```

---

## Data Augmentation Pipeline

Each training image is independently augmented on-the-fly before being fed into the network. The pipeline is configured via `AugmentCfg` and applies the following transforms (each with 50% probability, except translation):

| Transform | Parameter | Default |
|---|---|---|
| Random translation | `max_shift` | ±2 px (always applied) |
| Random rotation | `max_angle_deg` | ±10° |
| Random scale | `scale_delta` | ±10% |
| Elastic distortion | `elastic_alpha / sigma` | α=8.0, σ=3.0 |
| Gaussian noise | `noise_std` | 0.05 std |

All geometric transforms use **bilinear interpolation** with zero-padding at borders. The elastic distortion follows the Simard (2003) method: a random displacement field is generated and smoothed with repeated box-blur passes to approximate a Gaussian kernel.

### Thread-Safe Design

The augmentation pipeline is designed for parallel execution under OpenMP. Each worker thread owns a private `ThreadScratch` struct containing:
- `dst[784]` / `tmp[784]` — image buffers (no heap allocation per sample)
- `RNG rng` — an independent per-thread random number generator

All threads are seeded with different values derived from `time(NULL)`, ensuring uncorrelated augmentation across workers and making the training loop **fully thread-safe** with no locks.

---

## OpenMP Parallelism

Two independent levels of parallelism are used:

**Batch augmentation** — the inner loop over `BATCH_SIZE` images is parallelised with `#pragma omp parallel for`. Each thread augments its own slice of the batch using its private scratch buffers and RNG, with zero synchronisation overhead.

**Matrix operations** — the row loops in `mat_mul`, `mat_mul_atb`, `mat_mul_abt`, `relu`, `softmax_rows`, and `vec_axpy` are parallelised with `#pragma omp parallel for` when the problem size exceeds a threshold (e.g., `m > 32` or `n > 1024`).

Both levels activate automatically when compiled with `-fopenmp`; the code falls back gracefully to single-threaded execution without it.

---

## File Structure

```text
.
├── Makefile              # Root Makefile orchestrating the build
├── src/                  # Training code (backprop, optimizers, augmentation)
│   ├── main.c            # Training loop, OpenMP batch augmentation
│   ├── network.c / .h    # MLP forward/backward, BatchNorm, Dropout
│   ├── matrix.c / .h     # BLAS-style matrix ops with OpenMP
│   ├── mnist.c / .h      # IDX binary loader
│   ├── augment.c / .h    # Data augmentation pipeline + per-thread RNG
└── infer/                # Standalone inference engine (no dependency on src/)
    ├── infer.c / .h      # Lightweight inference core (portable to MCUs)
    ├── export.c          # BN folding — converts training weights to .inf
    ├── bench.c           # Accuracy, throughput, and latency benchmark
    └── single.c          # Single image inference (.pgm/.raw) with ASCII preview
```

---

## Quick Start

### 1. Download Data and Build
```bash
make download
make all
```

### 2. Train & Export
Runs the full training loop (30 epochs) then folds BN layers and writes the deployment-ready `models/model.inf`.
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
```bash
./infer/infer_single models/model.inf my_image.pgm
```

---

## Cross-Compilation & Edge Deployment

The inference core (`infer.c` and `infer.h`) is intentionally self-contained — it has no dependency on the training code, no `malloc` in the hot path, and accepts raw `uint8_t` pixel arrays directly.

To cross-compile for a bare-metal ARM Cortex-M7:
```bash
make CC=arm-none-eabi-gcc CFLAGS="-std=c99 -O2 -mthumb -mcpu=cortex-m7" -C infer
```

The `InferNet` struct occupies ~800 KB of flash; the `InferScratch` working buffer requires only a few KB of SRAM.

---

## License

MIT
