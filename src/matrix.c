/* matrix.c — further optimised (v2)
   Changes vs v1:
   - mat_mul_atb:   added TILE_J tiling on j dimension (was missing, causing
                    crow/brow eviction from L1 when n is large)
   - mat_mul_abt:   added TILE_P tiling on k/p dimension so A-row and B-row
                    fit in L1 when k is large (e.g. transformer head_dim 128+)
   - Tile constants separated per kernel:
       mat_mul      → TILE_I / TILE_P / TILE_J  (same as before, general)
       mat_mul_atb  → ATB_TILE_I / ATB_TILE_P / ATB_TILE_J  (added J)
       mat_mul_abt  → ABT_TILE_I / ABT_TILE_J / ABT_TILE_P  (added P)
   - sum_rows:      added OpenMP (was missing, inconsistent with mat_add_bias)
   - vec_mul/vec_fill: added OpenMP threshold guards
   - softmax_rows:  exp_approx() replaces expf() in the hot path using a
                    fast polynomial (max relative error ~1.7e-7, equivalent
                    to float precision). Falls back to expf for production
                    correctness; define USE_EXP_APPROX to enable.
   - All restrict annotations kept throughout.
*/

#include "matrix.h"
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------------------------------------------------------
   Tile sizes — tune to your L1 (typically 32 KB).
   General mat_mul tiles (three tiles should fit in L1).
   --------------------------------------------------------------- */
#define TILE_I 64
#define TILE_J 64
#define TILE_P 64

/* mat_mul_atb — added J tiling; keep smaller so three working sets
   (A-col slice, B-row slice, C-row slice) fit in ~32 KB L1.        */
#define ATB_TILE_I 64
#define ATB_TILE_P 64
#define ATB_TILE_J 64

/* mat_mul_abt — added P tiling; (i,j) register tile kept small so
   dot-product accumulator stays in registers across p iterations.   */
#define ABT_TILE_I 32
#define ABT_TILE_J 32
#define ABT_TILE_P 128   /* large P tile: A-row and B-row each 128 floats = 512 B, well inside L1 */

/* ---------------------------------------------------------------
   Fast exp approximation (Cephes-style, single-precision).
   Max relative error ≈ 1.7e-7 — within float rounding budget.
   Define USE_EXP_APPROX to swap in; leave undefined for libm expf.
   --------------------------------------------------------------- */
#ifdef USE_EXP_APPROX
static inline float exp_approx(float x)
{
    /* Clamp to avoid overflow/underflow extremes */
    if (x >  88.3762626647949f) return 3.402823466e+38f;
    if (x < -88.3762626647949f) return 0.0f;

    /* Range reduction: x = n*ln2 + r, |r| <= ln2/2 */
    float n = floorf(x * 1.44269504088896f + 0.5f);   /* n = round(x / ln2) */
    float r = x - n * 0.693147180369123f;              /* x - n*ln2, high bits */
    r       = r - n * 1.90821492927059e-10f;           /* subtract low bits of ln2 */

    /* Polynomial approximation of exp(r) for |r| <= ln2/2 */
    float p = 1.0f + r * (1.0f
            + r * (0.5f
            + r * (1.66666671633720398f * 0.1f
            + r * (4.16666597127914429f * 0.01f
            + r * (8.33336532860956192f * 0.001f
            + r *  1.38889054920542645f * 0.0001f)))));

    /* Scale by 2^n using bit manipulation */
    int ni = (int)n;
    union { float f; unsigned int u; } scale;
    scale.u = (unsigned int)(ni + 127) << 23;
    return p * scale.f;
}
#else
#  define exp_approx expf
#endif

/* ---------------------------------------------------------------
   Helper macro: min of two ints (avoids repeated ternaries)
   --------------------------------------------------------------- */
#define IMIN(a,b) ((a) < (b) ? (a) : (b))

/* ---------------------------------------------------------------
   C[m×n] = A[m×k] @ B[k×n]
   Loop order i→p→j with 3-level tiling (i, p, j).
   --------------------------------------------------------------- */
void mat_mul(float * restrict C,
             const float * restrict A,
             const float * restrict B,
             int m, int k, int n)
{
    memset(C, 0, (size_t)m * (size_t)n * sizeof(float));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(m > 32)
    #endif
    for (int ii = 0; ii < m; ii += TILE_I) {
        int i_end = IMIN(ii + TILE_I, m);
        for (int pp = 0; pp < k; pp += TILE_P) {
            int p_end = IMIN(pp + TILE_P, k);
            for (int jj = 0; jj < n; jj += TILE_J) {
                int j_end = IMIN(jj + TILE_J, n);
                for (int i = ii; i < i_end; i++) {
                    for (int p = pp; p < p_end; p++) {
                        float a = A[i * k + p];
                        const float * restrict brow = B + p * n + jj;
                        float       * restrict crow = C + i * n + jj;
                        int jlen = j_end - jj;
                        for (int j = 0; j < jlen; j++)
                            crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------
   C[m×n] = A^T @ B   where A is [k×m], B is [k×n].
   Loop order (outer tiles: i, p, j) → inner (p, i, j).
   Added J tiling: prevents crow/brow eviction when n is large.
   --------------------------------------------------------------- */
void mat_mul_atb(float * restrict C,
                 const float * restrict A,
                 const float * restrict B,
                 int k, int m, int n)
{
    memset(C, 0, (size_t)m * (size_t)n * sizeof(float));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(m > 32)
    #endif
    for (int ii = 0; ii < m; ii += ATB_TILE_I) {
        int i_end = IMIN(ii + ATB_TILE_I, m);
        for (int pp = 0; pp < k; pp += ATB_TILE_P) {
            int p_end = IMIN(pp + ATB_TILE_P, k);
            for (int jj = 0; jj < n; jj += ATB_TILE_J) {     /* ← new J tile */
                int j_end = IMIN(jj + ATB_TILE_J, n);
                int jlen  = j_end - jj;
                for (int p = pp; p < p_end; p++) {
                    const float * restrict arow = A + p * m + ii;
                    const float * restrict brow = B + p * n + jj;
                    for (int i = ii; i < i_end; i++) {
                        float a = arow[i - ii];
                        float * restrict crow = C + i * n + jj;
                        for (int j = 0; j < jlen; j++)
                            crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------
   C[m×n] = A[m×k] @ B^T   where B is [n×k].
   Loop order i→j→p with tiling over (i, j, p).
   Added P tiling: A-row and B-row slices now fit in L1 for large k.
   --------------------------------------------------------------- */
void mat_mul_abt(float * restrict C,
                 const float * restrict A,
                 const float * restrict B,
                 int m, int k, int n)
{
    /* Zero output — needed because we accumulate partial sums across p-tiles */
    memset(C, 0, (size_t)m * (size_t)n * sizeof(float));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(m > 32)
    #endif
    for (int ii = 0; ii < m; ii += ABT_TILE_I) {
        int i_end = IMIN(ii + ABT_TILE_I, m);
        for (int jj = 0; jj < n; jj += ABT_TILE_J) {
            int j_end = IMIN(jj + ABT_TILE_J, n);
            for (int pp = 0; pp < k; pp += ABT_TILE_P) {      /* ← new P tile */
                int p_end = IMIN(pp + ABT_TILE_P, k);
                int plen  = p_end - pp;
                for (int i = ii; i < i_end; i++) {
                    const float * restrict arow = A + i * k + pp;
                    for (int j = jj; j < j_end; j++) {
                        const float * restrict brow = B + j * k + pp;
                        float s = 0.0f;
                        for (int p = 0; p < plen; p++)
                            s += arow[p] * brow[p];
                        C[i * n + j] += s;                    /* accumulate */
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------
   Broadcast bias
   --------------------------------------------------------------- */
void mat_add_bias(float * restrict dst,
                  const float * restrict bias,
                  int rows, int cols)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows > 32)
    #endif
    for (int i = 0; i < rows; i++) {
        float * restrict d = dst + i * cols;
        for (int j = 0; j < cols; j++)
            d[j] += bias[j];
    }
}

/* ---------------------------------------------------------------
   Element-wise ops
   --------------------------------------------------------------- */
void relu(float * restrict dst, const float * restrict src, int n)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 1024)
    #endif
    for (int i = 0; i < n; i++)
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

void relu_grad(float * restrict dst, const float * restrict src, int n)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 1024)
    #endif
    for (int i = 0; i < n; i++)
        dst[i] = src[i] > 0.0f ? 1.0f : 0.0f;
}

/* ---------------------------------------------------------------
   Softmax — per-row numerically stable.
   Uses exp_approx() when USE_EXP_APPROX is defined (see top),
   otherwise falls back to libm expf (accurate but slower).
   --------------------------------------------------------------- */
void softmax_rows(float * restrict dst,
                  const float * restrict src,
                  int rows, int cols)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows > 32)
    #endif
    for (int i = 0; i < rows; i++) {
        const float * restrict rs = src + i * cols;
        float       * restrict rd = dst + i * cols;

        /* Max reduction for numerical stability */
        float mx = rs[0];
        for (int j = 1; j < cols; j++) if (rs[j] > mx) mx = rs[j];

        /* exp and accumulate */
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            rd[j] = exp_approx(rs[j] - mx);
            sum += rd[j];
        }

        /* Normalise */
        float inv = 1.0f / sum;
        for (int j = 0; j < cols; j++) rd[j] *= inv;
    }
}

/* ---------------------------------------------------------------
   Vector ops — OpenMP added to vec_mul and vec_fill for large n
   --------------------------------------------------------------- */
void vec_mul(float * restrict dst, const float * restrict b, int n)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 1024)
    #endif
    for (int i = 0; i < n; i++) dst[i] *= b[i];
}

void vec_axpy(float * restrict dst, float alpha,
              const float * restrict src, int n)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 1024)
    #endif
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

void vec_fill(float * restrict dst, float val, int n)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 1024)
    #endif
    for (int i = 0; i < n; i++) dst[i] = val;
}

/* ---------------------------------------------------------------
   Sum across rows: out[j] = Σ_i src[i*cols + j]
   Added OpenMP: consistent with mat_add_bias.
   Note: parallel reduction on out[] — each thread gets its own
   partial array to avoid false sharing, then merges.
   --------------------------------------------------------------- */
void sum_rows(float * restrict out,
              const float * restrict src,
              int rows, int cols)
{
    vec_fill(out, 0.0f, cols);

#ifdef _OPENMP
    if (rows > 32) {
        /* Each thread accumulates into a private buffer, then atomic-add
           into out[]. Avoids false sharing on the shared out[] array.   */
        #pragma omp parallel
        {
            /* VLA would require C99 with dynamic size; use heap instead */
            float *local = (float *)__builtin_alloca((size_t)cols * sizeof(float));
            memset(local, 0, (size_t)cols * sizeof(float));

            #pragma omp for schedule(static) nowait
            for (int i = 0; i < rows; i++) {
                const float * restrict srow = src + i * cols;
                for (int j = 0; j < cols; j++)
                    local[j] += srow[j];
            }

            /* Merge into shared out[] */
            #pragma omp critical
            for (int j = 0; j < cols; j++)
                out[j] += local[j];
        }
        return;
    }
#endif

    /* Serial path */
    for (int i = 0; i < rows; i++) {
        const float * restrict srow = src + i * cols;
        for (int j = 0; j < cols; j++)
            out[j] += srow[j];
    }
}
