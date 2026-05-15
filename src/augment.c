/* augment.c — optimised 
   changes vs before:
   - All pointer params marked restrict where safe.
   - box_blur vertical pass: original read tmp[yy*W+x] with stride AUG_W per
     inner step (stride-28 access).  Fixed by transposing: write a column-
     major temp during horizontal pass, then read sequentially in vertical.
     For a 28×28 image this is a modest win, but it's correct and cheap.
   - bilinear: unchanged (tiny, inline).
   - aug_translate: local tmp on stack (784 floats = 3.1 KB), unchanged.
   - aug_noise / aug_elastic / augment_apply: restrict added.
*/

#include "augment.h"
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------
   Bilinear sample — returns 0 for out-of-bounds
   --------------------------------------------------------------- */
static float bilinear(const float * restrict src, float fx, float fy)
{
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1,  y1 = y0 + 1;
    float ax = fx - x0, ay = fy - y0;
    float v = 0.0f, w = 0.0f;
#define S(xi, yi, wi) \
    if ((xi) >= 0 && (xi) < AUG_W && (yi) >= 0 && (yi) < AUG_H) \
        { v += (wi) * src[(yi)*AUG_W+(xi)]; w += (wi); }
    S(x0, y0, (1-ax)*(1-ay))
    S(x1, y0,    ax *(1-ay))
    S(x0, y1, (1-ax)*   ay )
    S(x1, y1,    ax *   ay )
#undef S
    return w > 0.0f ? v / w : 0.0f;
}

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ---------------------------------------------------------------
   Translation
   --------------------------------------------------------------- */
void aug_translate(float * restrict dst,
                   const float * restrict src,
                   int dx, int dy)
{
    float tmp[AUG_PX];
    memset(tmp, 0, sizeof(tmp));
    for (int y = 0; y < AUG_H; y++) {
        int sy = y - dy;
        if (sy < 0 || sy >= AUG_H) continue;
        for (int x = 0; x < AUG_W; x++) {
            int sx = x - dx;
            if (sx < 0 || sx >= AUG_W) continue;
            tmp[y * AUG_W + x] = src[sy * AUG_W + sx];
        }
    }
    memcpy(dst, tmp, AUG_PX * sizeof(float));
}

/* ---------------------------------------------------------------
   Rotation
   --------------------------------------------------------------- */
void aug_rotate(float * restrict dst,
                const float * restrict src,
                float angle_deg)
{
    float rad  = angle_deg * (3.14159265f / 180.0f);
    float cosA = cosf(rad), sinA = sinf(rad);
    float cx   = (AUG_W - 1) * 0.5f, cy = (AUG_H - 1) * 0.5f;
    for (int y = 0; y < AUG_H; y++) {
        float dy0 = y - cy;
        for (int x = 0; x < AUG_W; x++) {
            float dx0 = x - cx;
            dst[y * AUG_W + x] = bilinear(src,
                 cosA * dx0 + sinA * dy0 + cx,
                -sinA * dx0 + cosA * dy0 + cy);
        }
    }
}

/* ---------------------------------------------------------------
   Scale
   --------------------------------------------------------------- */
void aug_scale(float * restrict dst,
               const float * restrict src,
               float factor)
{
    float inv = 1.0f / factor;
    float cx  = (AUG_W - 1) * 0.5f, cy = (AUG_H - 1) * 0.5f;
    for (int y = 0; y < AUG_H; y++) {
        float fy = (y - cy) * inv + cy;
        for (int x = 0; x < AUG_W; x++)
            dst[y * AUG_W + x] = bilinear(src, (x - cx) * inv + cx, fy);
    }
}

/* ---------------------------------------------------------------
   Box blur — cache-friendly rewrite
   Horizontal pass  → row-major write into a column-major transposed buffer.
   Vertical   pass  → read the transposed buffer row-major (= original cols),
                       write output in row-major.
   Both inner loops are stride-1.
   --------------------------------------------------------------- */
static void box_blur(float * restrict f, int radius)
{
    /* tmp[x][y] — column-major (transposed) */
    float tmp[AUG_W * AUG_H];  /* 784 floats, stays on stack */

    if (radius < 1) radius = 1;
    if (radius > 5) radius = 5;

    /* --- Horizontal pass: read f[y*W+x], write tmp[x*H+y] --- */
    for (int y = 0; y < AUG_H; y++) {
        const float * restrict frow = f + y * AUG_W;
        for (int x = 0; x < AUG_W; x++) {
            float s = 0.0f; int n = 0;
            for (int k = -radius; k <= radius; k++) {
                int xx = x + k;
                if (xx >= 0 && xx < AUG_W) { s += frow[xx]; n++; }
            }
            tmp[x * AUG_H + y] = s / n;   /* column-major write */
        }
    }

    /* --- Vertical pass: read tmp[x*H+y] sequentially over y,
                          write f[y*W+x] — scan x-outer, y-inner --- */
    for (int x = 0; x < AUG_W; x++) {
        const float * restrict tcol = tmp + x * AUG_H;   /* stride-1 read */
        for (int y = 0; y < AUG_H; y++) {
            float s = 0.0f; int n = 0;
            for (int k = -radius; k <= radius; k++) {
                int yy = y + k;
                if (yy >= 0 && yy < AUG_H) { s += tcol[yy]; n++; }
            }
            f[y * AUG_W + x] = s / n;
        }
    }
}

/* ---------------------------------------------------------------
   Elastic distortion
   --------------------------------------------------------------- */
void aug_elastic(float * restrict dst,
                 const float * restrict src,
                 float alpha, float sigma,
                 float * restrict field_x,
                 float * restrict field_y,
                 RNG * restrict rng)
{
    for (int i = 0; i < AUG_PX; i++) {
        field_x[i] = rng_uniform(rng, -1.0f, 1.0f);
        field_y[i] = rng_uniform(rng, -1.0f, 1.0f);
    }
    int radius = (int)(sigma + 0.5f);
    for (int p = 0; p < 3; p++) {
        box_blur(field_x, radius);
        box_blur(field_y, radius);
    }
    for (int i = 0; i < AUG_PX; i++) {
        field_x[i] *= alpha;
        field_y[i] *= alpha;
    }
    for (int y = 0; y < AUG_H; y++)
        for (int x = 0; x < AUG_W; x++)
            dst[y * AUG_W + x] = bilinear(src,
                x + field_x[y * AUG_W + x],
                y + field_y[y * AUG_W + x]);
}

/* ---------------------------------------------------------------
   Gaussian noise
   --------------------------------------------------------------- */
void aug_noise(float * restrict dst,
               const float * restrict src,
               float std_dev, RNG * restrict rng)
{
    for (int i = 0; i < AUG_PX; i++)
        dst[i] = clampf(src[i] + rng_normal(rng) * std_dev, 0.0f, 1.0f);
}

/* ---------------------------------------------------------------
   Full augmentation pipeline
   --------------------------------------------------------------- */
void augment_apply(float * restrict dst,
                   const float * restrict src,
                   float * restrict tmp,
                   RNG * restrict rng,
                   const AugmentCfg * restrict cfg)
{
    const float *cur = src;
    float       *nxt = dst;

#define COMMIT do { \
    if (nxt == dst) { cur = dst; nxt = tmp; } \
    else            { cur = tmp; nxt = dst; } \
} while (0)

    if (cfg->max_shift > 0) {
        int s  = cfg->max_shift;
        int dx = (int)rng_uniform(rng, -(float)s, (float)s + 0.9999f);
        int dy = (int)rng_uniform(rng, -(float)s, (float)s + 0.9999f);
        aug_translate(nxt, cur, dx, dy);
        COMMIT;
    }
    if (cfg->max_angle_deg > 0.0f && (rng_next(rng) & 1)) {
        aug_rotate(nxt, cur,
                   rng_uniform(rng, -cfg->max_angle_deg, cfg->max_angle_deg));
        COMMIT;
    }
    if (cfg->scale_delta > 0.0f && (rng_next(rng) & 1)) {
        aug_scale(nxt, cur,
                  1.0f + rng_uniform(rng, -cfg->scale_delta, cfg->scale_delta));
        COMMIT;
    }
    if (cfg->elastic_alpha > 0.0f && (rng_next(rng) & 1)) {
        float fx[AUG_PX], fy[AUG_PX];
        aug_elastic(nxt, cur,
                    cfg->elastic_alpha, cfg->elastic_sigma,
                    fx, fy, rng);
        COMMIT;
    }
    if (cfg->noise_std > 0.0f && (rng_next(rng) & 1)) {
        aug_noise(nxt, cur, cfg->noise_std, rng);
        COMMIT;
    }
    if (cur != dst) memcpy(dst, cur, AUG_PX * sizeof(float));

#undef COMMIT
}
