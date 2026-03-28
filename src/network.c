#include "network.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------
   Helpers
   --------------------------------------------------------------- */
static float randn(void) {
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static void he_init(float *w, int fan_in, int n) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < n; i++) w[i] = randn() * scale;
}

/* Allocate one BNLayer for a feature dimension of `size`.
   MAX_BATCH rows are reserved for per-batch caches. */
static void bn_alloc(BNLayer *bn, int size) {
    bn->size        = size;
    bn->gamma       = (float *)malloc(size * sizeof(float));
    bn->beta        = (float *)calloc(size, sizeof(float));
    bn->v_gamma     = (float *)calloc(size, sizeof(float));
    bn->v_beta      = (float *)calloc(size, sizeof(float));
    bn->running_mean= (float *)calloc(size, sizeof(float));
    bn->running_var = (float *)malloc(size * sizeof(float));
    bn->mean        = (float *)malloc(size * sizeof(float));
    bn->var         = (float *)malloc(size * sizeof(float));
    bn->x_hat       = (float *)malloc(MAX_BATCH * size * sizeof(float));
    bn->z_in        = (float *)malloc(MAX_BATCH * size * sizeof(float));
    bn->dgamma      = (float *)malloc(size * sizeof(float));
    bn->dbeta       = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        bn->gamma[i]       = 1.0f;
        bn->running_var[i] = 1.0f;   /* safe initial variance */
    }
}

static void bn_free(BNLayer *bn) {
    free(bn->gamma);  free(bn->beta);
    free(bn->v_gamma);free(bn->v_beta);
    free(bn->running_mean); free(bn->running_var);
    free(bn->mean);   free(bn->var);
    free(bn->x_hat);  free(bn->z_in);
    free(bn->dgamma); free(bn->dbeta);
}

/* ---------------------------------------------------------------
   Network lifecycle
   --------------------------------------------------------------- */
Network *net_create(void) {
    Network *net = (Network *)calloc(1, sizeof(Network));
    if (!net) return NULL;

    he_init(net->W1, INPUT_SIZE,   HIDDEN1_SIZE * INPUT_SIZE);
    he_init(net->W2, HIDDEN1_SIZE, HIDDEN2_SIZE * HIDDEN1_SIZE);
    he_init(net->W3, HIDDEN2_SIZE, OUTPUT_SIZE  * HIDDEN2_SIZE);
    /* biases, velocities already zeroed by calloc */

    bn_alloc(&net->bn1, HIDDEN1_SIZE);
    bn_alloc(&net->bn2, HIDDEN2_SIZE);
    return net;
}

void net_free(Network *net) {
    if (!net) return;
    bn_free(&net->bn1);
    bn_free(&net->bn2);
    free(net);
}

/* ---------------------------------------------------------------
   BatchNorm forward
   Input:  z    [bs × H]  (output of FC layer)
   Output: out  [bs × H]  (normalized, scaled, shifted)
   Updates running stats when training=1.
   --------------------------------------------------------------- */
static void bn_forward(BNLayer *bn, float *out, const float *z,
                       int bs, int training) {
    int H = bn->size;

    /* Save input for backward */
    memcpy(bn->z_in, z, bs * H * sizeof(float));

    if (training) {
        /* Compute per-feature mean and variance over the batch */
        memset(bn->mean, 0, H * sizeof(float));
        memset(bn->var,  0, H * sizeof(float));

        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H; j++)
                bn->mean[j] += z[i * H + j];
        for (int j = 0; j < H; j++) bn->mean[j] /= bs;

        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H; j++) {
                float d = z[i * H + j] - bn->mean[j];
                bn->var[j] += d * d;
            }
        for (int j = 0; j < H; j++) bn->var[j] /= bs;

        /* Update exponential moving averages */
        for (int j = 0; j < H; j++) {
            bn->running_mean[j] = (1.0f - BN_MOMENTUM) * bn->running_mean[j]
                                 + BN_MOMENTUM * bn->mean[j];
            bn->running_var[j]  = (1.0f - BN_MOMENTUM) * bn->running_var[j]
                                 + BN_MOMENTUM * bn->var[j];
        }
    } else {
        /* Inference: use stable running stats */
        memcpy(bn->mean, bn->running_mean, H * sizeof(float));
        memcpy(bn->var,  bn->running_var,  H * sizeof(float));
    }

    /* Normalize, then scale and shift: out = gamma * x_hat + beta */
    for (int i = 0; i < bs; i++)
        for (int j = 0; j < H; j++) {
            float xh = (z[i * H + j] - bn->mean[j])
                       / sqrtf(bn->var[j] + BN_EPS);
            bn->x_hat[i * H + j] = xh;
            out[i * H + j] = bn->gamma[j] * xh + bn->beta[j];
        }
}

/* ---------------------------------------------------------------
   BatchNorm backward
   dout [bs × H] — gradient flowing in from above
   dz   [bs × H] — gradient to propagate to FC layer

   Full derivation (Ioffe & Szegedy 2015):
     dxhat  = dout * gamma
     dvar   = sum_i( dxhat * (x - mu) * -0.5 * (var+eps)^{-3/2} )
     dmean  = sum_i( dxhat * -1/sqrt(var+eps) )
            + dvar * sum_i(-2*(x-mu)) / bs
     dz[i]  = dxhat / sqrt(var+eps)
            + dvar * 2*(x-mu) / bs
            + dmean / bs
   --------------------------------------------------------------- */
static void bn_backward(BNLayer *bn, float *dz, const float *dout,
                        int bs) {
    int H = bn->size;

    /* dγ = Σ_i dout * x_hat,   dβ = Σ_i dout */
    memset(bn->dgamma, 0, H * sizeof(float));
    memset(bn->dbeta,  0, H * sizeof(float));
    for (int i = 0; i < bs; i++)
        for (int j = 0; j < H; j++) {
            bn->dgamma[j] += dout[i * H + j] * bn->x_hat[i * H + j];
            bn->dbeta[j]  += dout[i * H + j];
        }

    /* Propagate gradient through the normalization */
    float inv_bs = 1.0f / bs;
    for (int j = 0; j < H; j++) {
        float inv_std  = 1.0f / sqrtf(bn->var[j] + BN_EPS);
        float inv_std3 = inv_std * inv_std * inv_std;

        /* dxhat_j for all samples */
        float dvar  = 0.0f;
        float dmean = 0.0f;
        for (int i = 0; i < bs; i++) {
            float dxh = dout[i * H + j] * bn->gamma[j];
            float xmu = bn->z_in[i * H + j] - bn->mean[j];
            dvar  += dxh * xmu * (-0.5f) * inv_std3;
            dmean += dxh * (-inv_std);
        }
        dmean += dvar * (-2.0f * inv_bs) *
                 /* Σ(x - mu) = 0 exactly, but keep for clarity */
                 0.0f;

        for (int i = 0; i < bs; i++) {
            float dxh = dout[i * H + j] * bn->gamma[j];
            float xmu = bn->z_in[i * H + j] - bn->mean[j];
            dz[i * H + j] = dxh * inv_std
                           + dvar * 2.0f * xmu * inv_bs
                           + dmean * inv_bs;
        }
    }
}

/* ---------------------------------------------------------------
   Dropout (inverted, in-place)
   training=1: zero p fraction, scale survivors by 1/keep
   training=0: no-op
   --------------------------------------------------------------- */
static void dropout_forward(float *out, uint8_t *mask, const float *in,
                            int n, int training, float dropout_rate) {
    if (!training || dropout_rate <= 0.0f) {
        if (out != in) memcpy(out, in, n * sizeof(float));
        if (mask) memset(mask, 1, n * sizeof(uint8_t));
        return;
    }
    float keep = 1.0f - dropout_rate;
    float scale = 1.0f / keep;
    for (int i = 0; i < n; i++) {
        int keep_i = ((float)rand() / (float)RAND_MAX) < keep ? 1 : 0;
        mask[i] = (uint8_t)keep_i;
        out[i]  = keep_i ? in[i] * scale : 0.0f;
    }
}

static void dropout_backward(float *din, const float *dout,
                             const uint8_t *mask, int n, float dropout_rate) {
    if (dropout_rate <= 0.0f) {
        if (din != dout) memcpy(din, dout, n * sizeof(float));
        return;
    }
    float scale = 1.0f / (1.0f - dropout_rate);
    for (int i = 0; i < n; i++)
        din[i] = mask[i] ? dout[i] * scale : 0.0f;
}

/* ---------------------------------------------------------------
   Forward pass
   Layer order: FC → BN → ReLU → Dropout  (×2 hidden layers)
                FC → Softmax               (output)
   --------------------------------------------------------------- */
void net_forward(Network *net, const float *x, int bs, int training,
                 float dropout_rate) {
    /* --- Layer 1 --- */
    mat_mul_abt(net->z1, x, net->W1, bs, INPUT_SIZE, HIDDEN1_SIZE);
    mat_add_bias(net->z1, net->b1, bs, HIDDEN1_SIZE);

    /* BN1: z1 → bn_out (stored in z1 temporarily via a1) */
    bn_forward(&net->bn1, net->a1, net->z1, bs, training);

    /* ReLU (in-place on a1) */
    relu(net->a1, net->a1, bs * HIDDEN1_SIZE);

    /* Dropout → drop1 */
    dropout_forward(net->drop1, net->mask1, net->a1,
                    bs * HIDDEN1_SIZE, training, dropout_rate);

    /* --- Layer 2 --- */
    mat_mul_abt(net->z2, net->drop1, net->W2, bs, HIDDEN1_SIZE, HIDDEN2_SIZE);
    mat_add_bias(net->z2, net->b2, bs, HIDDEN2_SIZE);
    bn_forward(&net->bn2, net->a2, net->z2, bs, training);
    relu(net->a2, net->a2, bs * HIDDEN2_SIZE);
    dropout_forward(net->drop2, net->mask2, net->a2,
                    bs * HIDDEN2_SIZE, training, dropout_rate);

    /* --- Output layer (no BN, no Dropout) --- */
    mat_mul_abt(net->z3, net->drop2, net->W3, bs, HIDDEN2_SIZE, OUTPUT_SIZE);
    mat_add_bias(net->z3, net->b3, bs, OUTPUT_SIZE);
    softmax_rows(net->a3, net->z3, bs, OUTPUT_SIZE);
}

/* ---------------------------------------------------------------
   Metrics
   --------------------------------------------------------------- */
float net_loss(const Network *net, const uint8_t *labels, int bs) {
    float loss = 0.0f;
    for (int i = 0; i < bs; i++) {
        float p = net->a3[i * OUTPUT_SIZE + labels[i]];
        if (p < 1e-7f) p = 1e-7f;
        loss -= logf(p);
    }
    return loss / bs;
}

int net_correct(const Network *net, const uint8_t *labels, int bs) {
    int correct = 0;
    for (int i = 0; i < bs; i++) {
        int pred = 0;
        float best = net->a3[i * OUTPUT_SIZE];
        for (int j = 1; j < OUTPUT_SIZE; j++)
            if (net->a3[i * OUTPUT_SIZE + j] > best) {
                best = net->a3[i * OUTPUT_SIZE + j];
                pred = j;
            }
        if (pred == labels[i]) correct++;
    }
    return correct;
}

/* ---------------------------------------------------------------
   Momentum-SGD update macro (reduces repetition)
   Applies:  v = MOMENTUM * v + grad
             param -= lr * v
   --------------------------------------------------------------- */
#define MOMENTUM_UPDATE(param, vel, grad, n, lr) do {       \
    for (int _i = 0; _i < (n); _i++) {                     \
        (vel)[_i]   = MOMENTUM * (vel)[_i] + (grad)[_i];   \
        (param)[_i] -= (lr) * (vel)[_i];                   \
    }                                                       \
} while(0)

/* ---------------------------------------------------------------
   Backward pass + Momentum-SGD update
   --------------------------------------------------------------- */
void net_backward(Network *net, const float *x, const uint8_t *labels,
                  int bs, float lr, float dropout_rate) {
    float inv_bs = 1.0f / bs;

    /* ---- Output delta: softmax + cross-entropy shortcut ---- */
    memcpy(net->delta3, net->a3, bs * OUTPUT_SIZE * sizeof(float));
    for (int i = 0; i < bs; i++)
        net->delta3[i * OUTPUT_SIZE + labels[i]] -= 1.0f;
    /* Scale by 1/bs so gradients are already averaged */
    for (int i = 0; i < bs * OUTPUT_SIZE; i++)
        net->delta3[i] *= inv_bs;

    /* dW3 = delta3^T @ drop2,  db3 = sum_rows(delta3) */
    mat_mul_atb(net->dW3, net->delta3, net->drop2,
                bs, OUTPUT_SIZE, HIDDEN2_SIZE);
    sum_rows(net->db3, net->delta3, bs, OUTPUT_SIZE);
    MOMENTUM_UPDATE(net->W3, net->vW3, net->dW3, OUTPUT_SIZE * HIDDEN2_SIZE, lr);
    MOMENTUM_UPDATE(net->b3, net->vb3, net->db3, OUTPUT_SIZE, lr);

    /* ---- Backprop through layer 2 ---- */
    /* delta into drop2: delta3 @ W3   [bs×10] @ [10×128] → [bs×128] */
    mat_mul(net->delta2, net->delta3, net->W3, bs, OUTPUT_SIZE, HIDDEN2_SIZE);

    /* Through Dropout2 */
    dropout_backward(net->delta2, net->delta2, net->mask2,
                     bs * HIDDEN2_SIZE, dropout_rate);

    /* Through ReLU2: multiply by (a2 > 0)
       Note: a2 was stored BEFORE dropout in net->a2 (post-BN, post-ReLU) */
    for (int i = 0; i < bs * HIDDEN2_SIZE; i++)
        net->delta2[i] *= (net->a2[i] > 0.0f ? 1.0f : 0.0f);

    /* Through BN2 */
    bn_backward(&net->bn2, net->delta2, net->delta2, bs);
    /* Update BN2 params (gamma, beta) */
    MOMENTUM_UPDATE(net->bn2.gamma, net->bn2.v_gamma, net->bn2.dgamma,
                    HIDDEN2_SIZE, lr);
    MOMENTUM_UPDATE(net->bn2.beta,  net->bn2.v_beta,  net->bn2.dbeta,
                    HIDDEN2_SIZE, lr);

    /* dW2 = delta2^T @ drop1,  db2 = sum_rows(delta2) */
    mat_mul_atb(net->dW2, net->delta2, net->drop1,
                bs, HIDDEN2_SIZE, HIDDEN1_SIZE);
    sum_rows(net->db2, net->delta2, bs, HIDDEN2_SIZE);
    MOMENTUM_UPDATE(net->W2, net->vW2, net->dW2, HIDDEN2_SIZE * HIDDEN1_SIZE, lr);
    MOMENTUM_UPDATE(net->b2, net->vb2, net->db2, HIDDEN2_SIZE, lr);

    /* ---- Backprop through layer 1 ---- */
    mat_mul(net->delta1, net->delta2, net->W2, bs, HIDDEN2_SIZE, HIDDEN1_SIZE);
    dropout_backward(net->delta1, net->delta1, net->mask1,
                     bs * HIDDEN1_SIZE, dropout_rate);
    for (int i = 0; i < bs * HIDDEN1_SIZE; i++)
        net->delta1[i] *= (net->a1[i] > 0.0f ? 1.0f : 0.0f);

    bn_backward(&net->bn1, net->delta1, net->delta1, bs);
    MOMENTUM_UPDATE(net->bn1.gamma, net->bn1.v_gamma, net->bn1.dgamma,
                    HIDDEN1_SIZE, lr);
    MOMENTUM_UPDATE(net->bn1.beta,  net->bn1.v_beta,  net->bn1.dbeta,
                    HIDDEN1_SIZE, lr);

    mat_mul_atb(net->dW1, net->delta1, x, bs, HIDDEN1_SIZE, INPUT_SIZE);
    sum_rows(net->db1, net->delta1, bs, HIDDEN1_SIZE);
    MOMENTUM_UPDATE(net->W1, net->vW1, net->dW1, HIDDEN1_SIZE * INPUT_SIZE, lr);
    MOMENTUM_UPDATE(net->b1, net->vb1, net->db1, HIDDEN1_SIZE, lr);
}

/* ---------------------------------------------------------------
   Save / Load  (weights + BN running stats)
   --------------------------------------------------------------- */
#define WEIGHT_MAGIC 0x4D4C5002u  /* "MLP\x02" — v2 */

int net_save(const Network *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }
    uint32_t magic = WEIGHT_MAGIC;
    fwrite(&magic,   sizeof(magic),   1, f);
    fwrite(net->W1,  sizeof(net->W1), 1, f);
    fwrite(net->b1,  sizeof(net->b1), 1, f);
    fwrite(net->W2,  sizeof(net->W2), 1, f);
    fwrite(net->b2,  sizeof(net->b2), 1, f);
    fwrite(net->W3,  sizeof(net->W3), 1, f);
    fwrite(net->b3,  sizeof(net->b3), 1, f);
    /* BN params and running stats */
    fwrite(net->bn1.gamma,        net->bn1.size * sizeof(float), 1, f);
    fwrite(net->bn1.beta,         net->bn1.size * sizeof(float), 1, f);
    fwrite(net->bn1.running_mean, net->bn1.size * sizeof(float), 1, f);
    fwrite(net->bn1.running_var,  net->bn1.size * sizeof(float), 1, f);
    fwrite(net->bn2.gamma,        net->bn2.size * sizeof(float), 1, f);
    fwrite(net->bn2.beta,         net->bn2.size * sizeof(float), 1, f);
    fwrite(net->bn2.running_mean, net->bn2.size * sizeof(float), 1, f);
    fwrite(net->bn2.running_var,  net->bn2.size * sizeof(float), 1, f);
    fclose(f);
    printf("Saved weights to %s\n", path);
    return 1;
}

#define FREAD_CHECK(ptr, size, f) \
    if (fread((ptr), (size), 1, (f)) != 1) { fclose(f); return 0; }

int net_load(Network *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    uint32_t magic;
    FREAD_CHECK(&magic, sizeof(magic), f)
    if (magic != WEIGHT_MAGIC) {
        fprintf(stderr, "Bad weight magic in %s (expected v2)\n", path);
        fclose(f); return 0;
    }
    FREAD_CHECK(net->W1, sizeof(net->W1), f)
    FREAD_CHECK(net->b1, sizeof(net->b1), f)
    FREAD_CHECK(net->W2, sizeof(net->W2), f)
    FREAD_CHECK(net->b2, sizeof(net->b2), f)
    FREAD_CHECK(net->W3, sizeof(net->W3), f)
    FREAD_CHECK(net->b3, sizeof(net->b3), f)
    FREAD_CHECK(net->bn1.gamma,        net->bn1.size * sizeof(float), f)
    FREAD_CHECK(net->bn1.beta,         net->bn1.size * sizeof(float), f)
    FREAD_CHECK(net->bn1.running_mean, net->bn1.size * sizeof(float), f)
    FREAD_CHECK(net->bn1.running_var,  net->bn1.size * sizeof(float), f)
    FREAD_CHECK(net->bn2.gamma,        net->bn2.size * sizeof(float), f)
    FREAD_CHECK(net->bn2.beta,         net->bn2.size * sizeof(float), f)
    FREAD_CHECK(net->bn2.running_mean, net->bn2.size * sizeof(float), f)
    FREAD_CHECK(net->bn2.running_var,  net->bn2.size * sizeof(float), f)
    fclose(f);
    printf("Loaded weights from %s\n", path);
    return 1;
}
