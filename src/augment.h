#ifndef AUGMENT_H
#define AUGMENT_H

/*
 * augment.h — On-the-fly data augmentation for 28x28 float images
 *
 * All functions operate on flat float arrays [784], pixels in [0,1].
 * dst and src may be the same pointer (in-place is safe for translate,
 * but NOT for rotate/elastic — pass separate buffers for those).
 *
 * Design: pure C99, no external deps, seeded via stdlib rand().
 */

#define AUG_W 28
#define AUG_H 28
#define AUG_PX (AUG_W * AUG_H)   /* 784 */

/* ---------------------------------------------------------------
   AugmentCfg — runtime-configurable parameters
   Zero-initialise and set only what you want; 0 means disabled.
   --------------------------------------------------------------- */
typedef struct {
    /* Translation: uniform random shift in [-max_shift, +max_shift] px */
    int   max_shift;       /* recommended: 2  (pixels) */

    /* Rotation: uniform random angle in [-max_angle, +max_angle] deg */
    float max_angle_deg;   /* recommended: 10.0 */

    /* Scale: uniform random scale in [1-scale_delta, 1+scale_delta] */
    float scale_delta;     /* recommended: 0.10  (±10%) */

    /* Elastic distortion (Simard 2003):
       alpha controls displacement magnitude,
       sigma controls smoothness of the displacement field */
    float elastic_alpha;   /* recommended: 8.0  */
    float elastic_sigma;   /* recommended: 3.0  */

    /* Gaussian noise: add N(0, noise_std) to each pixel, clamp to [0,1] */
    float noise_std;       /* recommended: 0.05 */
} AugmentCfg;

/* Sensible defaults that work well for MNIST MLP */
static inline AugmentCfg augment_default(void) {
    AugmentCfg c = {0};
    c.max_shift     = 2;
    c.max_angle_deg = 10.0f;
    c.scale_delta   = 0.10f;
    c.elastic_alpha = 0.0f;   /* disabled by default — expensive */
    c.elastic_sigma = 3.0f;
    c.noise_std     = 0.0f;   /* disabled by default */
    return c;
}

/* ---------------------------------------------------------------
   augment_apply
   Apply the full augmentation pipeline to one image.
     src  [784]  input  (float, [0,1])
     dst  [784]  output (float, [0,1]) — must NOT alias src
     tmp  [784]  scratch buffer        — must NOT alias src or dst
     cfg         augmentation parameters
   Each transform is independently applied with probability 0.5
   except translation which is always applied when max_shift > 0.
   --------------------------------------------------------------- */
void augment_apply(float *dst, const float *src, float *tmp,
                   const AugmentCfg *cfg);

/* ---------------------------------------------------------------
   Individual transforms (exposed for testing / custom pipelines)
   dst and src must NOT alias each other.
   --------------------------------------------------------------- */

/* Shift image by (dx, dy) pixels; fill border with 0 */
void aug_translate(float *dst, const float *src, int dx, int dy);

/* Rotate around image centre by angle_deg; bilinear interp, border=0 */
void aug_rotate(float *dst, const float *src, float angle_deg);

/* Scale around image centre by factor; bilinear interp, border=0 */
void aug_scale(float *dst, const float *src, float factor);

/* Elastic distortion; requires two [AUG_PX] float scratch buffers */
void aug_elastic(float *dst, const float *src,
                 float alpha, float sigma,
                 float *field_x, float *field_y);

/* Add Gaussian noise, clamp result to [0,1] */
void aug_noise(float *dst, const float *src, float std_dev);

#endif /* AUGMENT_H */
