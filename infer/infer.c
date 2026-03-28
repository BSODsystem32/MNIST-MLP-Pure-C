#include "infer.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------------------------------------------------------------
   Forward pass — the entire hot path
   No malloc, no branches on training mode, no BN arithmetic.

   For each hidden layer:
     h = ReLU( W * x + b )   (pure dot products + bias + clamp)
   Output:
     out = softmax( W * h + b )
   --------------------------------------------------------------- */

/* Compute y = W*x + b,  W:[rows×cols], x:[cols], y:[rows] */
static void fc(float *y, const float *W, const float *b,
               const float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float s = b[i];
        const float *row = W + i * cols;
        for (int j = 0; j < cols; j++)
            s += row[j] * x[j];
        y[i] = s;
    }
}

/* ReLU in-place */
static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

/* Numerically stable softmax in-place */
static void softmax_inplace(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* Core forward — fills scratch->out with raw logits or softmax */
static void forward_core(const InferNet *net, InferScratch *scratch,
                         const float *x) {
    fc(scratch->h1, net->W1, net->b1, x,          INF_H1, INF_IN);
    relu_inplace(scratch->h1, INF_H1);
    fc(scratch->h2, net->W2, net->b2, scratch->h1, INF_H2, INF_H1);
    relu_inplace(scratch->h2, INF_H2);
    fc(scratch->out, net->W3, net->b3, scratch->h2, INF_OUT, INF_H2);
}

int infer_forward(const InferNet *net, InferScratch *scratch,
                  const float *x) {
    forward_core(net, scratch, x);
    return argmax(scratch->out, INF_OUT);
}

int infer_forward_u8(const InferNet *net, InferScratch *scratch,
                     const uint8_t *pixels) {
    /* Normalize on the fly — no separate float input buffer needed */
    float x[INF_IN];
    for (int i = 0; i < INF_IN; i++)
        x[i] = pixels[i] * (1.0f / 255.0f);
    forward_core(net, scratch, x);
    return argmax(scratch->out, INF_OUT);
}

int infer_forward_probs(const InferNet *net, InferScratch *scratch,
                        const float *x) {
    forward_core(net, scratch, x);
    softmax_inplace(scratch->out, INF_OUT);
    return argmax(scratch->out, INF_OUT);
}

/* ---------------------------------------------------------------
   Weight file I/O — minimal flat binary
   Header: magic(u32) + IN(u32) + H1(u32) + H2(u32) + OUT(u32)
   Body:   W1, b1, W2, b2, W3, b3  (raw floats, little-endian)
   --------------------------------------------------------------- */
int infer_save(const InferNet *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }

    uint32_t hdr[5] = { INF_MAGIC, INF_IN, INF_H1, INF_H2, INF_OUT };
    fwrite(hdr,     sizeof(hdr),         1, f);
    fwrite(net->W1, sizeof(net->W1),     1, f);
    fwrite(net->b1, sizeof(net->b1),     1, f);
    fwrite(net->W2, sizeof(net->W2),     1, f);
    fwrite(net->b2, sizeof(net->b2),     1, f);
    fwrite(net->W3, sizeof(net->W3),     1, f);
    fwrite(net->b3, sizeof(net->b3),     1, f);
    fclose(f);
    printf("Saved inference weights: %s  (%.1f KB)\n",
           path,
           (double)(sizeof(InferNet) + sizeof(hdr)) / 1024.0);
    return 1;
}

#define FREAD1(ptr, sz, f) \
    if (fread((ptr), (sz), 1, (f)) != 1) { fclose(f); return 0; }

int infer_load(InferNet *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }

    uint32_t hdr[5];
    FREAD1(hdr, sizeof(hdr), f)
    if (hdr[0] != INF_MAGIC) {
        fprintf(stderr, "%s: bad magic 0x%08X\n", path, hdr[0]);
        fclose(f); return 0;
    }
    if (hdr[1] != INF_IN || hdr[2] != INF_H1 ||
        hdr[3] != INF_H2 || hdr[4] != INF_OUT) {
        fprintf(stderr, "%s: dimension mismatch (%u %u %u %u)\n",
                path, hdr[1], hdr[2], hdr[3], hdr[4]);
        fclose(f); return 0;
    }
    FREAD1(net->W1, sizeof(net->W1), f)
    FREAD1(net->b1, sizeof(net->b1), f)
    FREAD1(net->W2, sizeof(net->W2), f)
    FREAD1(net->b2, sizeof(net->b2), f)
    FREAD1(net->W3, sizeof(net->W3), f)
    FREAD1(net->b3, sizeof(net->b3), f)
    fclose(f);
    return 1;
}

/* ---------------------------------------------------------------
   BN Folding
   Reads the training weight file (format written by net_save in
   network.c), folds BN into FC weights, writes out InferNet.

   Training file layout (magic MLP\x02):
     magic(u32)
     W1[H1×IN]  b1[H1]
     W2[H2×H1]  b2[H2]
     W3[OUT×H2] b3[OUT]
     bn1.gamma[H1]  bn1.beta[H1]  bn1.running_mean[H1]  bn1.running_var[H1]
     bn2.gamma[H2]  bn2.beta[H2]  bn2.running_mean[H2]  bn2.running_var[H2]
   --------------------------------------------------------------- */

#define TRAIN_MAGIC  0x4D4C5002u   /* MLP\x02 — must match network.c */
#define FOLD_BN_EPS  1e-5f

/* Fold one FC layer + its BN into folded W'/b'.
   W   [out_dim × in_dim]  original weights
   b   [out_dim]           original biases
   gamma, beta, mean, var  BN params (all [out_dim])
   W_out, b_out            folded results (caller allocated)
*/
static void fold_layer(float *W_out, float *b_out,
                       const float *W,    const float *b,
                       const float *gamma, const float *beta,
                       const float *mean,  const float *var,
                       int out_dim, int in_dim) {
    for (int j = 0; j < out_dim; j++) {
        float sigma   = sqrtf(var[j] + FOLD_BN_EPS);
        float scale   = gamma[j] / sigma;

        /* W'[j,:] = W[j,:] * scale */
        for (int k = 0; k < in_dim; k++)
            W_out[j * in_dim + k] = W[j * in_dim + k] * scale;

        /* b'[j] = (b[j] - mean[j]) * scale + beta[j] */
        b_out[j] = (b[j] - mean[j]) * scale + beta[j];
    }
}

int infer_fold_from_training(InferNet *out, const char *train_weights_path) {
    FILE *f = fopen(train_weights_path, "rb");
    if (!f) { perror(train_weights_path); return 0; }

    uint32_t magic;
    FREAD1(&magic, sizeof(magic), f)
    if (magic != TRAIN_MAGIC) {
        fprintf(stderr, "%s: expected training magic 0x%08X, got 0x%08X\n",
                train_weights_path, TRAIN_MAGIC, magic);
        fclose(f); return 0;
    }

    /* Allocate temporaries for raw training params */
    size_t w1sz = INF_H1 * INF_IN,  w2sz = INF_H2 * INF_H1,
           w3sz = INF_OUT * INF_H2;

    float *W1  = malloc(w1sz  * sizeof(float));
    float *b1  = malloc(INF_H1 * sizeof(float));
    float *W2  = malloc(w2sz  * sizeof(float));
    float *b2  = malloc(INF_H2 * sizeof(float));
    float *W3  = malloc(w3sz  * sizeof(float));
    float *b3  = malloc(INF_OUT * sizeof(float));
    /* BN1 */
    float *g1  = malloc(INF_H1 * sizeof(float));
    float *bt1 = malloc(INF_H1 * sizeof(float));
    float *m1  = malloc(INF_H1 * sizeof(float));
    float *v1  = malloc(INF_H1 * sizeof(float));
    /* BN2 */
    float *g2  = malloc(INF_H2 * sizeof(float));
    float *bt2 = malloc(INF_H2 * sizeof(float));
    float *m2  = malloc(INF_H2 * sizeof(float));
    float *v2  = malloc(INF_H2 * sizeof(float));

    int ok = W1 && b1 && W2 && b2 && W3 && b3 &&
             g1 && bt1 && m1 && v1 && g2 && bt2 && m2 && v2;
    if (!ok) { fprintf(stderr, "Out of memory\n"); goto cleanup; }

#define RF(ptr, n) if (fread((ptr), sizeof(float), (n), f) != (size_t)(n)) \
                       { ok = 0; goto cleanup; }
    RF(W1, w1sz)  RF(b1, INF_H1)
    RF(W2, w2sz)  RF(b2, INF_H2)
    RF(W3, w3sz)  RF(b3, INF_OUT)
    RF(g1, INF_H1) RF(bt1, INF_H1) RF(m1, INF_H1) RF(v1, INF_H1)
    RF(g2, INF_H2) RF(bt2, INF_H2) RF(m2, INF_H2) RF(v2, INF_H2)
#undef RF

    /* Fold BN1 into FC1, BN2 into FC2 */
    fold_layer(out->W1, out->b1, W1, b1, g1, bt1, m1, v1, INF_H1, INF_IN);
    fold_layer(out->W2, out->b2, W2, b2, g2, bt2, m2, v2, INF_H2, INF_H1);

    /* Output layer has no BN — copy directly */
    memcpy(out->W3, W3, w3sz  * sizeof(float));
    memcpy(out->b3, b3, INF_OUT * sizeof(float));

    printf("BN folding complete:\n");
    printf("  Layer 1:  W[%d×%d]  scale range folded into weights\n",
           INF_H1, INF_IN);
    printf("  Layer 2:  W[%d×%d]  scale range folded into weights\n",
           INF_H2, INF_H1);
    printf("  Layer 3:  W[%d×%d]  (no BN, copied as-is)\n",
           INF_OUT, INF_H2);

cleanup:
    fclose(f);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
    free(g1); free(bt1); free(m1); free(v1);
    free(g2); free(bt2); free(m2); free(v2);
    return ok ? 1 : 0;
}
