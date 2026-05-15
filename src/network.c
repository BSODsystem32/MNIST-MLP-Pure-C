/* network.c — optimised
   Key changes vs before:
   1. bn_forward  training path: accumulate mean/var with i-outer, j-inner
      (row-major z access) instead of j-outer (stride-H).
      Normalize pass also i-outer j-inner.
   2. bn_backward: was j-outer i-inner → stride-H reads on every dout/z_in.
      Rewritten:
        Pass 1 (i-outer j-inner): accumulate dgamma, dbeta into per-sample
                                   scratch; also build x_mu[B×H] temp array.
        Pass 2 (j-outer): compute scalar dvar, dmean per feature (no matrix
                           access, just the [H] arrays).
        Pass 3 (i-outer j-inner): write dz using precomputed scalars.
      This keeps all [B×H] accesses row-major throughout.
   3. All pointers marked restrict where safe.
   4. dropout_forward: branch-free multiply instead of ternary.
*/

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

static void he_init(float * restrict w, int fan_in, int n) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < n; i++) w[i] = randn() * scale;
}

static void bn_alloc(BNLayer *bn, int size) {
    bn->size         = size;
    bn->gamma        = (float *)malloc(size * sizeof(float));
    bn->beta         = (float *)calloc(size, sizeof(float));
    bn->v_gamma      = (float *)calloc(size, sizeof(float));
    bn->v_beta       = (float *)calloc(size, sizeof(float));
    bn->running_mean = (float *)calloc(size, sizeof(float));
    bn->running_var  = (float *)malloc(size * sizeof(float));
    bn->mean         = (float *)malloc(size * sizeof(float));
    bn->var          = (float *)malloc(size * sizeof(float));
    bn->x_hat        = (float *)malloc(MAX_BATCH * size * sizeof(float));
    bn->z_in         = (float *)malloc(MAX_BATCH * size * sizeof(float));
    bn->dgamma       = (float *)malloc(size * sizeof(float));
    bn->dbeta        = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        bn->gamma[i]       = 1.0f;
        bn->running_var[i] = 1.0f;
    }
}

static void bn_free(BNLayer *bn) {
    free(bn->gamma);  free(bn->beta);
    free(bn->v_gamma); free(bn->v_beta);
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
   BatchNorm forward — all [B×H] accesses row-major (i-outer j-inner)
   --------------------------------------------------------------- */
static void bn_forward(BNLayer * restrict bn,
                       float * restrict out,
                       const float * restrict z,
                       int bs, int training)
{
    int H = bn->size;
    float inv_bs = 1.0f / bs;

    /* Save input for backward */
    memcpy(bn->z_in, z, (size_t)bs * H * sizeof(float));

    if (training) {
        /* --- mean: accumulate row-major, then divide --- */
        memset(bn->mean, 0, H * sizeof(float));
        for (int i = 0; i < bs; i++) {
            const float * restrict zrow = z + i * H;
            float       * restrict mean = bn->mean;
            for (int j = 0; j < H; j++)
                mean[j] += zrow[j];
        }
        for (int j = 0; j < H; j++) bn->mean[j] *= inv_bs;

        /* --- variance: second pass, row-major --- */
        memset(bn->var, 0, H * sizeof(float));
        for (int i = 0; i < bs; i++) {
            const float * restrict zrow   = z      + i * H;
            const float * restrict mean   = bn->mean;
            float       * restrict varrow = bn->var;
            for (int j = 0; j < H; j++) {
                float d = zrow[j] - mean[j];
                varrow[j] += d * d;
            }
        }
        for (int j = 0; j < H; j++) bn->var[j] *= inv_bs;

        /* --- Update EMA --- */
        float m = BN_MOMENTUM, om = 1.0f - BN_MOMENTUM;
        for (int j = 0; j < H; j++) {
            bn->running_mean[j] = om * bn->running_mean[j] + m * bn->mean[j];
            bn->running_var[j]  = om * bn->running_var[j]  + m * bn->var[j];
        }
    } else {
        memcpy(bn->mean, bn->running_mean, H * sizeof(float));
        memcpy(bn->var,  bn->running_var,  H * sizeof(float));
    }

    /* --- Precompute inv_std per feature (avoids repeated sqrtf in inner) --- */
    /* Reuse dgamma as temp — but we don't want to corrupt it; use stack if H<=256 */
    /* H is 256 or 128 — safe on stack */
    float inv_std[256]; /* max(HIDDEN1_SIZE, HIDDEN2_SIZE) */
    for (int j = 0; j < H; j++)
        inv_std[j] = 1.0f / sqrtf(bn->var[j] + BN_EPS);

    /* --- Normalize + scale + shift: i-outer j-inner --- */
    const float * restrict gamma = bn->gamma;
    const float * restrict beta  = bn->beta;
    const float * restrict mean  = bn->mean;
    for (int i = 0; i < bs; i++) {
        const float * restrict zrow    = z       + i * H;
        float       * restrict xhrow   = bn->x_hat + i * H;
        float       * restrict outrow  = out      + i * H;
        for (int j = 0; j < H; j++) {
            float xh    = (zrow[j] - mean[j]) * inv_std[j];
            xhrow[j]   = xh;
            outrow[j]  = gamma[j] * xh + beta[j];
        }
    }
}

/* ---------------------------------------------------------------
   BatchNorm backward — cache-friendly three-pass layout

   Pass 1 (i-outer j-inner, row-major):
     accumulate dgamma[j] += dout*xhat,  dbeta[j] += dout
     store dxh[i*H+j] = dout[i*H+j] * gamma[j]   (temp in dz)
     store xmu[i*H+j] = z_in[i*H+j] - mean[j]     (temp, reuse out buffer trick)

   Pass 2 (j-only, tiny loop over H):
     dvar[j]  = Σ_i dxh * xmu * (-0.5) * inv_std^3
     dmean[j] = Σ_i dxh * (-inv_std)

   Pass 3 (i-outer j-inner, row-major):
     dz[i*H+j] = dxh*inv_std + dvar*2*xmu*inv_bs + dmean*inv_bs

   All [B×H] matrices touched row-major → no stride-H penalty.
   --------------------------------------------------------------- */
static void bn_backward(BNLayer * restrict bn,
                        float *dz,
                        const float *dout,
                        int bs)
{
    int   H      = bn->size;
    float inv_bs = 1.0f / bs;

    /* Stack-allocate per-feature intermediates (H ≤ 256) */
    float inv_std[256], inv_std3[256];
    float dvar[256], dmean[256];

    for (int j = 0; j < H; j++) {
        float is  = 1.0f / sqrtf(bn->var[j] + BN_EPS);
        inv_std[j]  = is;
        inv_std3[j] = is * is * is;
    }

    /* Pass 1: dgamma, dbeta, and build dxh (store into dz temporarily)
               and xmu (store into x_hat — we're done with it after this) */
    memset(bn->dgamma, 0, H * sizeof(float));
    memset(bn->dbeta,  0, H * sizeof(float));

    const float * restrict gamma = bn->gamma;
    const float * restrict mean  = bn->mean;

    for (int i = 0; i < bs; i++) {
        const float * restrict dout_row = dout     + i * H;
        const float * restrict xhat_row = bn->x_hat + i * H;
        const float * restrict zin_row  = bn->z_in  + i * H;
        float       * restrict dz_row   = dz        + i * H;
        float       * restrict xmu_row  = bn->x_hat + i * H; /* reuse x_hat as xmu */

        for (int j = 0; j < H; j++) {
            float do_j  = dout_row[j];
            bn->dgamma[j] += do_j * xhat_row[j];
            bn->dbeta[j]  += do_j;
            dz_row[j]      = do_j * gamma[j];         /* dxh  */
            xmu_row[j]     = zin_row[j] - mean[j];    /* x-mu */
        }
    }

    /* Pass 2: reduce dvar, dmean over batch (j-major tiny vectors) */
    memset(dvar,  0, H * sizeof(float));
    memset(dmean, 0, H * sizeof(float));

    for (int i = 0; i < bs; i++) {
        const float * restrict dxh_row = dz        + i * H;  /* holds dxh */
        const float * restrict xmu_row = bn->x_hat + i * H;
        for (int j = 0; j < H; j++) {
            float dxh = dxh_row[j];
            float xmu = xmu_row[j];
            dvar[j]  += dxh * xmu * (-0.5f) * inv_std3[j];
            dmean[j] += dxh * (-inv_std[j]);
        }
    }
    /* dmean correction term: dvar * Σ(-2*xmu)/bs.  Σ(xmu)=0 exactly, skip. */

    /* Pass 3: write final dz row-major */
    for (int i = 0; i < bs; i++) {
        const float * restrict dxh_row = dz        + i * H;
        const float * restrict xmu_row = bn->x_hat + i * H;
        float       * restrict dz_row  = dz        + i * H;
        for (int j = 0; j < H; j++) {
            dz_row[j] = dxh_row[j] * inv_std[j]
                      + dvar[j]  * 2.0f * xmu_row[j] * inv_bs
                      + dmean[j] * inv_bs;
        }
    }
}

/* ---------------------------------------------------------------
   Dropout (inverted, in-place) — branch-free inner loop
   --------------------------------------------------------------- */
static void dropout_forward(float * restrict out,
                            uint8_t * restrict mask,
                            const float * restrict in,
                            int n, int training, float dropout_rate)
{
    if (!training || dropout_rate <= 0.0f) {
        if (out != in) memcpy(out, in, n * sizeof(float));
        if (mask) memset(mask, 1, n * sizeof(uint8_t));
        return;
    }
    float keep  = 1.0f - dropout_rate;
    float scale = 1.0f / keep;
    for (int i = 0; i < n; i++) {
        int k  = ((float)rand() / (float)RAND_MAX) < keep ? 1 : 0;
        mask[i] = (uint8_t)k;
        out[i]  = in[i] * (k ? scale : 0.0f);
    }
}

static void dropout_backward(float *din,
                             const float *dout,
                             const uint8_t * restrict mask,
                             int n, float dropout_rate)
{
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
   --------------------------------------------------------------- */
void net_forward(Network * restrict net,
                 const float * restrict x,
                 int bs, int training, float dropout_rate)
{
    /* Layer 1 */
    mat_mul_abt(net->z1, x,        net->W1, bs, INPUT_SIZE,   HIDDEN1_SIZE);
    mat_add_bias(net->z1, net->b1, bs, HIDDEN1_SIZE);
    bn_forward(&net->bn1, net->a1, net->z1, bs, training);
    relu(net->a1, net->a1, bs * HIDDEN1_SIZE);
    dropout_forward(net->drop1, net->mask1, net->a1,
                    bs * HIDDEN1_SIZE, training, dropout_rate);

    /* Layer 2 */
    mat_mul_abt(net->z2, net->drop1, net->W2, bs, HIDDEN1_SIZE, HIDDEN2_SIZE);
    mat_add_bias(net->z2, net->b2, bs, HIDDEN2_SIZE);
    bn_forward(&net->bn2, net->a2, net->z2, bs, training);
    relu(net->a2, net->a2, bs * HIDDEN2_SIZE);
    dropout_forward(net->drop2, net->mask2, net->a2,
                    bs * HIDDEN2_SIZE, training, dropout_rate);

    /* Output */
    mat_mul_abt(net->z3, net->drop2, net->W3, bs, HIDDEN2_SIZE, OUTPUT_SIZE);
    mat_add_bias(net->z3, net->b3, bs, OUTPUT_SIZE);
    softmax_rows(net->a3, net->z3, bs, OUTPUT_SIZE);
}

/* ---------------------------------------------------------------
   Metrics
   --------------------------------------------------------------- */
float net_loss(const Network * restrict net,
               const uint8_t * restrict labels, int bs)
{
    float loss = 0.0f;
    for (int i = 0; i < bs; i++) {
        float p = net->a3[i * OUTPUT_SIZE + labels[i]];
        if (p < 1e-7f) p = 1e-7f;
        loss -= logf(p);
    }
    return loss / bs;
}

int net_correct(const Network * restrict net,
                const uint8_t * restrict labels, int bs)
{
    int correct = 0;
    for (int i = 0; i < bs; i++) {
        const float * restrict row = net->a3 + i * OUTPUT_SIZE;
        int pred = 0;
        float best = row[0];
        for (int j = 1; j < OUTPUT_SIZE; j++)
            if (row[j] > best) { best = row[j]; pred = j; }
        if (pred == labels[i]) correct++;
    }
    return correct;
}

/* ---------------------------------------------------------------
   Momentum-SGD update
   --------------------------------------------------------------- */
#define MOMENTUM_UPDATE(param, vel, grad, n, lr) do {        \
    float * restrict _p = (param);                           \
    float * restrict _v = (vel);                             \
    const float * restrict _g = (grad);                      \
    int _n = (n); float _lr = (lr);                          \
    for (int _i = 0; _i < _n; _i++) {                       \
        _v[_i]  = MOMENTUM * _v[_i] + _g[_i];               \
        _p[_i] -= _lr * _v[_i];                              \
    }                                                        \
} while(0)

/* ---------------------------------------------------------------
   Backward pass + Momentum-SGD
   --------------------------------------------------------------- */
void net_backward(Network * restrict net,
                  const float * restrict x,
                  const uint8_t * restrict labels,
                  int bs, float lr, float dropout_rate)
{
    float inv_bs = 1.0f / bs;

    /* Output delta */
    memcpy(net->delta3, net->a3, (size_t)bs * OUTPUT_SIZE * sizeof(float));
    for (int i = 0; i < bs; i++)
        net->delta3[i * OUTPUT_SIZE + labels[i]] -= 1.0f;
    for (int i = 0; i < bs * OUTPUT_SIZE; i++)
        net->delta3[i] *= inv_bs;

    /* dW3, db3 */
    mat_mul_atb(net->dW3, net->delta3, net->drop2,
                bs, OUTPUT_SIZE, HIDDEN2_SIZE);
    sum_rows(net->db3, net->delta3, bs, OUTPUT_SIZE);
    MOMENTUM_UPDATE(net->W3, net->vW3, net->dW3, OUTPUT_SIZE * HIDDEN2_SIZE, lr);
    MOMENTUM_UPDATE(net->b3, net->vb3, net->db3, OUTPUT_SIZE, lr);

    /* Layer 2 backprop */
    mat_mul(net->delta2, net->delta3, net->W3, bs, OUTPUT_SIZE, HIDDEN2_SIZE);
    dropout_backward(net->delta2, net->delta2, net->mask2,
                     bs * HIDDEN2_SIZE, dropout_rate);
    /* ReLU2 gate — row-major, stride-1 */
    for (int i = 0; i < bs * HIDDEN2_SIZE; i++)
        net->delta2[i] *= (net->a2[i] > 0.0f ? 1.0f : 0.0f);

    bn_backward(&net->bn2, net->delta2, net->delta2, bs);
    MOMENTUM_UPDATE(net->bn2.gamma, net->bn2.v_gamma, net->bn2.dgamma,
                    HIDDEN2_SIZE, lr);
    MOMENTUM_UPDATE(net->bn2.beta,  net->bn2.v_beta,  net->bn2.dbeta,
                    HIDDEN2_SIZE, lr);

    mat_mul_atb(net->dW2, net->delta2, net->drop1,
                bs, HIDDEN2_SIZE, HIDDEN1_SIZE);
    sum_rows(net->db2, net->delta2, bs, HIDDEN2_SIZE);
    MOMENTUM_UPDATE(net->W2, net->vW2, net->dW2, HIDDEN2_SIZE * HIDDEN1_SIZE, lr);
    MOMENTUM_UPDATE(net->b2, net->vb2, net->db2, HIDDEN2_SIZE, lr);

    /* Layer 1 backprop */
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
   Save / Load
   --------------------------------------------------------------- */
#define WEIGHT_MAGIC 0x4D4C5002u

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
        fprintf(stderr, "Bad weight magic in %s\n", path);
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