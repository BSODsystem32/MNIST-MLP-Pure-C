#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>

/* ---------------------------------------------------------------
   MLP: 784 → 256 → 128 → 10
   Each hidden layer: FC → BatchNorm → ReLU → Dropout
   Output layer:      FC → Softmax
   Optimizer: SGD with Momentum
   --------------------------------------------------------------- */

#define INPUT_SIZE   784
#define HIDDEN1_SIZE 256
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE  10
#define MAX_BATCH    256

/* BatchNorm EMA decay for running stats used at inference */
#define BN_MOMENTUM  0.1f
/* Small constant for numerical stability in BN */
#define BN_EPS       1e-5f
/* SGD momentum coefficient */
#define MOMENTUM     0.9f

/* ---------------------------------------------------------------
   Per-layer BatchNorm state
   Stores learned params (gamma, beta), their velocity buffers,
   running stats for inference, and per-forward-pass caches.
   --------------------------------------------------------------- */
typedef struct {
    int   size;               /* feature dimension H */

    /* Learned scale and shift */
    float *gamma;             /* [H] initialized to 1 */
    float *beta;              /* [H] initialized to 0 */

    /* Momentum velocity for gamma/beta */
    float *v_gamma;
    float *v_beta;

    /* Running stats (EMA), used during inference */
    float *running_mean;      /* [H] */
    float *running_var;       /* [H] */

    /* Forward-pass cache (needed for backward) */
    float *mean;              /* batch mean      [H] */
    float *var;               /* batch variance  [H] */
    float *x_hat;             /* normalized      [B×H] */
    float *z_in;              /* input to BN     [B×H] (= z from FC) */

    /* Gradient buffers */
    float *dgamma;            /* [H] */
    float *dbeta;             /* [H] */
} BNLayer;

/* ---------------------------------------------------------------
   Full network
   --------------------------------------------------------------- */
typedef struct {
    /* ---- FC weights & biases ---- */
    float W1[HIDDEN1_SIZE * INPUT_SIZE];
    float b1[HIDDEN1_SIZE];
    float W2[HIDDEN2_SIZE * HIDDEN1_SIZE];
    float b2[HIDDEN2_SIZE];
    float W3[OUTPUT_SIZE  * HIDDEN2_SIZE];
    float b3[OUTPUT_SIZE];

    /* ---- Momentum velocity buffers for W, b ---- */
    float vW1[HIDDEN1_SIZE * INPUT_SIZE];
    float vb1[HIDDEN1_SIZE];
    float vW2[HIDDEN2_SIZE * HIDDEN1_SIZE];
    float vb2[HIDDEN2_SIZE];
    float vW3[OUTPUT_SIZE  * HIDDEN2_SIZE];
    float vb3[OUTPUT_SIZE];

    /* ---- BatchNorm layers (one per hidden layer) ---- */
    BNLayer bn1;   /* after FC1 */
    BNLayer bn2;   /* after FC2 */

    /* ---- Forward-pass activations ---- */
    float z1   [MAX_BATCH * HIDDEN1_SIZE];   /* FC1 pre-BN */
    float a1   [MAX_BATCH * HIDDEN1_SIZE];   /* post-ReLU */
    float drop1[MAX_BATCH * HIDDEN1_SIZE];   /* post-Dropout */
    float z2   [MAX_BATCH * HIDDEN2_SIZE];
    float a2   [MAX_BATCH * HIDDEN2_SIZE];
    float drop2[MAX_BATCH * HIDDEN2_SIZE];
    float z3   [MAX_BATCH * OUTPUT_SIZE];
    float a3   [MAX_BATCH * OUTPUT_SIZE];    /* softmax probabilities */

    /* ---- Dropout masks (1=keep, 0=drop) ---- */
    uint8_t mask1[MAX_BATCH * HIDDEN1_SIZE];
    uint8_t mask2[MAX_BATCH * HIDDEN2_SIZE];

    /* ---- Gradient buffers for W, b ---- */
    float dW1[HIDDEN1_SIZE * INPUT_SIZE];
    float db1[HIDDEN1_SIZE];
    float dW2[HIDDEN2_SIZE * HIDDEN1_SIZE];
    float db2[HIDDEN2_SIZE];
    float dW3[OUTPUT_SIZE  * HIDDEN2_SIZE];
    float db3[OUTPUT_SIZE];

    /* ---- Backprop delta buffers ---- */
    float delta3[MAX_BATCH * OUTPUT_SIZE];
    float delta2[MAX_BATCH * HIDDEN2_SIZE];
    float delta1[MAX_BATCH * HIDDEN1_SIZE];

} Network;

/* Allocate & initialize (He for W, 0 for v, 1/0 for gamma/beta) */
Network *net_create(void);
void     net_free(Network *net);

/* Forward pass.
   training=1: use batch stats for BN, apply dropout.
   training=0: use running stats, no dropout. */
void net_forward(Network *net, const float *x, int bs, int training,
                 float dropout_rate);

/* Backward + Momentum-SGD update.
   Call only after net_forward with training=1. */
void net_backward(Network *net, const float *x, const uint8_t *labels,
                  int bs, float lr, float dropout_rate);

/* Metrics (call after net_forward) */
float net_loss   (const Network *net, const uint8_t *labels, int bs);
int   net_correct(const Network *net, const uint8_t *labels, int bs);

/* Persist weights */
int net_save(const Network *net, const char *path);
int net_load(Network      *net, const char *path);

#endif
