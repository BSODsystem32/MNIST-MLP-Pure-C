#include "augment.h"
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------
   Bilinear sample — returns 0 for out-of-bounds
   --------------------------------------------------------------- */
static float bilinear(const float *src, float fx, float fy) {
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1,  y1 = y0 + 1;
    float ax = fx - x0, ay = fy - y0;
    float v = 0.0f, w = 0.0f;
#define S(xi,yi,wi) \
    if ((xi)>=0 && (xi)<AUG_W && (yi)>=0 && (yi)<AUG_H) \
        { v += (wi)*src[(yi)*AUG_W+(xi)]; w += (wi); }
    S(x0,y0,(1-ax)*(1-ay)) S(x1,y0,ax*(1-ay))
    S(x0,y1,(1-ax)*ay)     S(x1,y1,ax*ay)
#undef S
    return w > 0.0f ? v / w : 0.0f;
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ---------------------------------------------------------------
   Translation — handles aliasing internally
   --------------------------------------------------------------- */
void aug_translate(float *dst, const float *src, int dx, int dy) {
    float tmp[AUG_PX];
    memset(tmp, 0, sizeof(tmp));
    for (int y = 0; y < AUG_H; y++) {
        int sy = y - dy;
        if (sy < 0 || sy >= AUG_H) continue;
        for (int x = 0; x < AUG_W; x++) {
            int sx = x - dx;
            if (sx < 0 || sx >= AUG_W) continue;
            tmp[y*AUG_W+x] = src[sy*AUG_W+sx];
        }
    }
    memcpy(dst, tmp, AUG_PX * sizeof(float));
}

/* ---------------------------------------------------------------
   Rotation — inverse mapping + bilinear, dst must NOT alias src
   --------------------------------------------------------------- */
void aug_rotate(float *dst, const float *src, float angle_deg) {
    float rad  = angle_deg * (3.14159265f / 180.0f);
    float cosA = cosf(rad), sinA = sinf(rad);
    float cx   = (AUG_W - 1) * 0.5f, cy = (AUG_H - 1) * 0.5f;
    for (int y = 0; y < AUG_H; y++) {
        for (int x = 0; x < AUG_W; x++) {
            float dx = x - cx, dy = y - cy;
            dst[y*AUG_W+x] = bilinear(src,
                cosA*dx + sinA*dy + cx,
               -sinA*dx + cosA*dy + cy);
        }
    }
}

/* ---------------------------------------------------------------
   Scale — inverse mapping + bilinear, dst must NOT alias src
   --------------------------------------------------------------- */
void aug_scale(float *dst, const float *src, float factor) {
    float inv = 1.0f / factor;
    float cx  = (AUG_W - 1) * 0.5f, cy = (AUG_H - 1) * 0.5f;
    for (int y = 0; y < AUG_H; y++)
        for (int x = 0; x < AUG_W; x++)
            dst[y*AUG_W+x] = bilinear(src,
                (x-cx)*inv + cx, (y-cy)*inv + cy);
}

/* ---------------------------------------------------------------
   Box blur (approximate Gaussian), in-place
   --------------------------------------------------------------- */
static void box_blur(float *f, int radius) {
    float tmp[AUG_PX];
    if (radius < 1) radius = 1;
    if (radius > 5) radius = 5;
    /* Horizontal */
    for (int y = 0; y < AUG_H; y++)
        for (int x = 0; x < AUG_W; x++) {
            float s = 0.0f; int n = 0;
            for (int k = -radius; k <= radius; k++) {
                int xx = x+k;
                if (xx >= 0 && xx < AUG_W) { s += f[y*AUG_W+xx]; n++; }
            }
            tmp[y*AUG_W+x] = s/n;
        }
    /* Vertical */
    for (int y = 0; y < AUG_H; y++)
        for (int x = 0; x < AUG_W; x++) {
            float s = 0.0f; int n = 0;
            for (int k = -radius; k <= radius; k++) {
                int yy = y+k;
                if (yy >= 0 && yy < AUG_H) { s += tmp[yy*AUG_W+x]; n++; }
            }
            f[y*AUG_W+x] = s/n;
        }
}

/* ---------------------------------------------------------------
   Elastic distortion (Simard 2003) — thread-safe via RNG*
   --------------------------------------------------------------- */
void aug_elastic(float *dst, const float *src,
                 float alpha, float sigma,
                 float *field_x, float *field_y, RNG *rng) {
    for (int i = 0; i < AUG_PX; i++) {
        field_x[i] = rng_uniform(rng, -1.0f, 1.0f);
        field_y[i] = rng_uniform(rng, -1.0f, 1.0f);
    }
    int radius = (int)(sigma + 0.5f);
    for (int p = 0; p < 3; p++) { box_blur(field_x, radius); box_blur(field_y, radius); }
    for (int i = 0; i < AUG_PX; i++) { field_x[i] *= alpha; field_y[i] *= alpha; }
    for (int y = 0; y < AUG_H; y++)
        for (int x = 0; x < AUG_W; x++)
            dst[y*AUG_W+x] = bilinear(src,
                x + field_x[y*AUG_W+x],
                y + field_y[y*AUG_W+x]);
}

/* ---------------------------------------------------------------
   Gaussian noise — thread-safe via RNG*
   --------------------------------------------------------------- */
void aug_noise(float *dst, const float *src, float std_dev, RNG *rng) {
    for (int i = 0; i < AUG_PX; i++)
        dst[i] = clampf(src[i] + rng_normal(rng) * std_dev, 0.0f, 1.0f);
}

/* ---------------------------------------------------------------
   augment_apply — full pipeline, thread-safe
   --------------------------------------------------------------- */
void augment_apply(float *dst, const float *src, float *tmp,
                   RNG *rng, const AugmentCfg *cfg) {
    const float *cur = src;
    float       *nxt = dst;
#define COMMIT do { \
    if (nxt==dst){cur=dst;nxt=tmp;} else {cur=tmp;nxt=dst;} } while(0)

    if (cfg->max_shift > 0) {
        int s = cfg->max_shift;
        int dx = (int)rng_uniform(rng, -(float)s, (float)s + 0.9999f);
        int dy = (int)rng_uniform(rng, -(float)s, (float)s + 0.9999f);
        aug_translate(nxt, cur, dx, dy);
        COMMIT;
    }
    if (cfg->max_angle_deg > 0.0f && (rng_next(rng) & 1)) {
        aug_rotate(nxt, cur, rng_uniform(rng, -cfg->max_angle_deg, cfg->max_angle_deg));
        COMMIT;
    }
    if (cfg->scale_delta > 0.0f && (rng_next(rng) & 1)) {
        aug_scale(nxt, cur, 1.0f + rng_uniform(rng, -cfg->scale_delta, cfg->scale_delta));
        COMMIT;
    }
    if (cfg->elastic_alpha > 0.0f && (rng_next(rng) & 1)) {
        float fx[AUG_PX], fy[AUG_PX];
        aug_elastic(nxt, cur, cfg->elastic_alpha, cfg->elastic_sigma, fx, fy, rng);
        COMMIT;
    }
    if (cfg->noise_std > 0.0f && (rng_next(rng) & 1)) {
        aug_noise(nxt, cur, cfg->noise_std, rng);
        COMMIT;
    }
    if (cur != dst) memcpy(dst, cur, AUG_PX * sizeof(float));
#undef COMMIT
}
