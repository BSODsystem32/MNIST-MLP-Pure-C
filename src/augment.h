#ifndef AUGMENT_H
#define AUGMENT_H

/*
 * augment.h — On-the-fly data augmentation for 28x28 float images
 *
 * Thread-safe: all randomness goes through a caller-supplied RNG state
 * (xorshift32) so multiple threads can augment in parallel without
 * locking or sharing state.
 */

#include <stdint.h>

#define AUG_W  28
#define AUG_H  28
#define AUG_PX (AUG_W * AUG_H)   /* 784 */

/* ---------------------------------------------------------------
   Per-thread RNG — xorshift32, period 2^32-1, no stdlib dependency
   Seed with any non-zero value (e.g. thread_id ^ time ^ sample_idx).
   --------------------------------------------------------------- */
typedef struct { uint32_t s; } RNG;

static inline void   rng_seed(RNG *r, uint32_t seed) { r->s = seed ? seed : 1u; }
static inline uint32_t rng_next(RNG *r) {
    r->s ^= r->s << 13; r->s ^= r->s >> 17; r->s ^= r->s << 5;
    return r->s;
}
/* Uniform float in [lo, hi) */
static inline float rng_uniform(RNG *r, float lo, float hi) {
    return lo + (hi - lo) * ((float)(rng_next(r) >> 8) / (float)(1 << 24));
}
/* N(0,1) via Box-Muller */
static inline float rng_normal(RNG *r) {
    float u1 = (float)(rng_next(r) >> 8) / (float)(1 << 24) + 1e-7f;
    float u2 = (float)(rng_next(r) >> 8) / (float)(1 << 24);
    /* Use math.h sqrtf/logf/cosf — included by augment.c */
    extern float sqrtf(float), logf(float), cosf(float);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ---------------------------------------------------------------
   AugmentCfg — runtime-configurable parameters
   Zero-initialise and set only what you want; 0 means disabled.
   --------------------------------------------------------------- */
typedef struct {
    int   max_shift;        /* translate ±N px.  recommended: 2          */
    float max_angle_deg;    /* rotate   ±N deg.  recommended: 10.0       */
    float scale_delta;      /* scale  1±delta.   recommended: 0.10       */
    float elastic_alpha;    /* displacement mag. recommended: 8.0        */
    float elastic_sigma;    /* field smoothness. recommended: 3.0        */
    float noise_std;        /* gaussian noise.   recommended: 0.05       */
} AugmentCfg;

static inline AugmentCfg augment_default(void) {
    AugmentCfg c = {0};
    c.max_shift     = 2;
    c.max_angle_deg = 10.0f;
    c.scale_delta   = 0.10f;
    c.elastic_alpha = 0.0f;
    c.elastic_sigma = 3.0f;
    c.noise_std     = 0.0f;
    return c;
}

/* ---------------------------------------------------------------
   augment_apply — full pipeline, thread-safe via RNG state
     dst  [784]  output  (must NOT alias src)
     src  [784]  input
     tmp  [784]  scratch (must NOT alias src or dst)
     rng         caller-owned per-thread RNG state
     cfg         augmentation config
   --------------------------------------------------------------- */
void augment_apply(float *dst, const float *src, float *tmp,
                   RNG *rng, const AugmentCfg *cfg);

/* Individual transforms — all thread-safe, take RNG* instead of rand() */
void aug_translate(float *dst, const float *src, int dx, int dy);
void aug_rotate   (float *dst, const float *src, float angle_deg);
void aug_scale    (float *dst, const float *src, float factor);
void aug_elastic  (float *dst, const float *src,
                   float alpha, float sigma,
                   float *field_x, float *field_y, RNG *rng);
void aug_noise    (float *dst, const float *src, float std_dev, RNG *rng);

#endif /* AUGMENT_H */
