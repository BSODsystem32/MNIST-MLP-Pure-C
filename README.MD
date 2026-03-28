[繁體中文](README.ZH.MD) / [English](README.MD)

---

# MNIST MLP — Pure C End-to-End Neural Network

An end-to-end fully-connected neural network project written entirely in plain C99 from scratch, with **zero external ML library dependencies**. This project is split into two robust parts:
1. **Training (`src/`)**: Features BatchNorm, Dropout, and SGD with Momentum, achieving ~98% test accuracy on MNIST.
2. **Inference (`infer/`)**: A bare-metal ready, zero-malloc inference engine that uses **BatchNorm Folding** to aggressively optimize the model for edge devices (e.g., ARM Cortex-M microcontrollers).

---

## Network Architecture

**Training Architecture:**
```text
Input (784)
  └─► FC1 (256) → BatchNorm → ReLU → Dropout
        └─► FC2 (128) → BatchNorm → ReLU → Dropout
              └─► FC3 (10) → Softmax → Cross-Entropy Loss
```

**Inference Architecture (Post-Folding):**
During deployment, the BatchNorm scale and shift parameters are pre-computed and **folded** directly into the Fully Connected weights and biases.
```text
Input (784, uint8)
  └─► FC1' (256) → ReLU
        └─► FC2' (128) → ReLU
              └─► FC3' (10) → Softmax
```

---

## File Structure

```text
.
├── Makefile             # Root makefile orchestrating the build
├── src/                 # Training code (backprop, optimizers, etc.)
│   ├── main.c, network.c, matrix.c, mnist.c
├── infer/               # Standalone Inference code (No dependency on src/)
│   ├── infer.c / .h     # The core engine (portable to MCUs)
│   ├── export.c         # Folds BN params and converts weights to .inf format
│   ├── bench.c          # Evaluates accuracy, throughput, and latency
│   └── single.c         # Single image inference (.pgm/.raw) with ASCII art
└── data/                # MNIST dataset (auto-downloaded via make)
```

### Key Technical Highlights

*   **BatchNorm Folding (`infer_export`)**: Reads the training checkpoint (which contains BN running stats), absorbs the BN math into the adjacent FC layers, and spits out a compact ~800 KB inference-only file (`model.inf`).
*   **Featherweight Inference Engine (`infer.c`)**:
    *   Zero `malloc` / dynamic allocation in the hot path.
    *   Accepts raw `uint8_t` (0-255) pixel arrays directly, normalizing on the fly.
    *   100% portable. Ready for bare-metal cross-compilation.

---

## Quick Start

### 1. Download Data and Build
Fetch the MNIST dataset and compile both training and inference binaries.
```bash
make download
make all
```

### 2. Train & Export the Model
This runs the complete training loop (25 epochs) and automatically invokes `infer_export` to fold BN layers and generate the deployment-ready `models/model.inf`.
```bash
make export
```

### 3. Run Benchmark
Test the inference engine against the 10,000 test images to measure accuracy, throughput, and latency:
```bash
make bench
```
*Expected Output:*
```text
Accuracy: 98.3x%
Throughput benchmark (10 passes × 10000 samples):
  Total time:  X.XXX s
  Throughput:  XXXXX samples/s
  Latency:     X.XXX ms/sample
```

### 4. Single Image Inference
Pass a 28x28 PGM or raw pixel file to see predictions and an ASCII art preview of the input:
```bash
# If you have your own 28x28 grayscale image:
./infer/infer_single models/model.inf my_image.pgm
```

---

## Cross-Compilation & Edge AI Deployment

The inference core (`infer.c` and `infer.h`) is intentionally isolated. 
To deploy this model onto a bare-metal ARM Cortex-M device, simply drop those two files into your embedded project.