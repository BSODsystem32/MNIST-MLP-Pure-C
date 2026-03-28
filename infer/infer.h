#ifndef INFER_H
#define INFER_H

/*
 * infer.h — Lightweight MLP inference engine
 *
 * This module is completely independent of the training code.
 * It only knows about folded FC weights + biases and ReLU activations.
 * Safe to compile on bare-metal (no dynamic allocation in the hot path).
 *
 * Layout:  784 → 256 (ReLU) → 128 (ReLU) → 10 (Softmax)
 */

#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------
   Dimensions (must match training config)
   --------------------------------------------------------------- */
#define INF_IN   784
#define INF_H1   256
#define INF_H2   128
#define INF_OUT  10

/* Magic number for the inference weight file format */
#define INF_MAGIC  0x494E4602u   /* "INF\x02" */

/* ---------------------------------------------------------------
   InferNet — folded weights stored flat, no BN state
   Sized to fit in ~800 KB of static RAM (feasible on Cortex-M7+)
   --------------------------------------------------------------- */
typedef struct {
    float W1[INF_H1 * INF_IN];   /* [256 × 784] */
    float b1[INF_H1];
    float W2[INF_H2 * INF_H1];   /* [128 × 256] */
    float b2[INF_H2];
    float W3[INF_OUT * INF_H2];  /* [ 10 × 128] */
    float b3[INF_OUT];
} InferNet;

/* Scratch buffer for one forward pass — caller allocates once.
   Must hold at least max(INF_H1, INF_H2, INF_OUT) floats. */
typedef struct {
    float h1[INF_H1];
    float h2[INF_H2];
    float out[INF_OUT];
} InferScratch;

/* ---------------------------------------------------------------
   API
   --------------------------------------------------------------- */

/*
 * infer_forward — run one sample.
 * x:     [INF_IN] float pixels, already in [0,1]
 * scratch: caller-owned work buffer (reuse across calls)
 * Returns the predicted class index (0-9).
 */
int infer_forward(const InferNet *net, InferScratch *scratch,
                  const float *x);

/*
 * infer_forward_u8 — same but accepts raw uint8 pixel bytes [0,255].
 * Normalizes to [0,1] on the fly; no separate float buffer needed.
 */
int infer_forward_u8(const InferNet *net, InferScratch *scratch,
                     const uint8_t *pixels);

/*
 * infer_forward_probs — fills scratch->out with softmax probabilities.
 * Returns predicted class index.
 */
int infer_forward_probs(const InferNet *net, InferScratch *scratch,
                        const float *x);

/*
 * infer_load — load an inference weight file (produced by infer_export).
 * Returns 1 on success, 0 on failure.
 */
int infer_load(InferNet *net, const char *path);

/*
 * infer_save — write folded weights to a binary file.
 */
int infer_save(const InferNet *net, const char *path);

/* ---------------------------------------------------------------
   BN Folding helper — call once after training to produce InferNet.
   Pulls W/b/BN params from the training weight file and folds them.

   Folding formula (per output feature j of a layer):
     W'[j,:] = W[j,:] * gamma[j] / sigma[j]
     b'[j]   = (b[j] - mean[j]) * gamma[j] / sigma[j] + beta[j]
   where sigma[j] = sqrt(running_var[j] + BN_EPS)
   --------------------------------------------------------------- */
int infer_fold_from_training(InferNet *out, const char *train_weights_path);

#endif /* INFER_H */
