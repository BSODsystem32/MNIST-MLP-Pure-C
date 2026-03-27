[繁體中文](README.ZH.MD) / [English](README.MD)

---

# MNIST MLP — Pure-C Neural Network from Scratch

A fully-connected neural network trained on MNIST, implemented in plain C99 with **no external ML libraries**. Achieves ~98% test accuracy using BatchNorm, Dropout, and SGD with Momentum.

---

## Network Architecture

```
Input (784)
  └─► FC1 (256) → BatchNorm → ReLU → Dropout
        └─► FC2 (128) → BatchNorm → ReLU → Dropout
              └─► FC3 (10) → Softmax → Cross-Entropy Loss
```

| Layer | Type | Output Size |
|-------|------|-------------|
| Input | Flatten 28×28 | 784 |
| Hidden 1 | FC + BN + ReLU + Dropout | 256 |
| Hidden 2 | FC + BN + ReLU + Dropout | 128 |
| Output | FC + Softmax | 10 |

Weight initialization uses **He initialization** (`scale = sqrt(2 / fan_in)`), suitable for ReLU activations.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPOCHS` | 25 | Training epochs |
| `BATCH_SIZE` | 128 | Mini-batch size |
| `LR_INIT` | 0.05 | Initial learning rate |
| `LR_DECAY` | 0.92 | Per-epoch LR multiplier |
| `DROPOUT_RATE` | 0.3 | Fraction of neurons dropped during training |
| `MOMENTUM` | 0.9 | SGD momentum coefficient |
| `BN_EPS` | 1e-5 | BatchNorm numerical stability constant |
| `BN_MOMENTUM` | 0.1 | BatchNorm running stats EMA decay |

Learning rate follows an exponential schedule: `lr = LR_INIT × LR_DECAY^epoch`.

---

## File Structure

```
.
├── main.c       # Entry point: training loop, evaluation, CLI args
├── network.c    # Forward pass, backward pass, BN, Dropout, save/load
├── network.h    # Network struct, layer sizes, public API
├── matrix.c     # Matrix ops: mat_mul, relu, softmax, etc.
├── matrix.h     # Matrix op declarations
├── mnist.c      # IDX binary file loader, normalizes pixels to [0, 1]
├── mnist.h      # MnistData struct
└── Makefile
```

### Key implementation details

**`matrix.c`** — All tensor math in row-major flat arrays. Provides three matmul variants (`A@B`, `Aᵀ@B`, `A@Bᵀ`) to avoid explicit transposes during backprop.

**`network.c`** — Contains the full forward and backward pass:
- `bn_forward` / `bn_backward`: Ioffe & Szegedy (2015) BatchNorm with per-feature mean/variance computed over the batch, and EMA running stats for inference.
- `dropout_forward` / `dropout_backward`: Inverted dropout — survivors are scaled by `1/(1−p)` at training time so inference requires no rescaling.
- `MOMENTUM_UPDATE` macro: applies `v = β·v + g`, then `param -= lr·v`.
- `net_save` / `net_load`: writes weights and BN running stats to a binary file tagged with magic `0x4D4C5002` (`MLP\x02`).

**`mnist.c`** — Reads the IDX3/IDX1 binary format (big-endian), normalizes pixel bytes to `float32` in `[0, 1]`.

---

## Quick Start

### 1. Get the MNIST data

```bash
make download
```

This fetches and decompresses the four IDX files into `./data/`. Requires `curl` and `gunzip`.

> If the LeCun mirror is unavailable, download manually from https://yann.lecun.com/exdb/mnist/ or a mirror, and place the four uncompressed files under `./data/`.

### 2. Build

```bash
make
```

Requires GCC (or any C99-compatible compiler) and `libm`. Produces the `mnist_mlp` binary.

### 3. Train

```bash
make run
# equivalent to:
./mnist_mlp ./data weights_v2.bin
```

**CLI usage:**

```
./mnist_mlp [data_dir] [weights_out]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `data_dir` | `./data` | Directory containing the four IDX files |
| `weights_out` | `weights_v2.bin` | Output path for saved weights |

### 4. Expected output

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

### 5. Clean

```bash
make clean   # removes object files, binary, and weights_v2.bin
```

---

## Dependencies

- C99-compatible compiler (GCC / Clang)
- Standard C library (`libc`, `libm`)
- `curl` + `gunzip` for `make download` only

No BLAS, no PyTorch, no external ML framework.

---

## Platform Notes

The timer in `main.c` uses `QueryPerformanceCounter` on Windows and `clock_gettime(CLOCK_MONOTONIC)` on POSIX systems, so the binary compiles on both.

---

## License

MIT