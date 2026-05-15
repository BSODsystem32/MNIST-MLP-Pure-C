/* matrix.c — optimised
   changes vs before:
   - All pointer args marked restrict (lets compiler auto-vectorise safely)
   - mat_mul:     loop order i→p→j already fine; add tile over j (L1 friendly)
   - mat_mul_atb: was i→p→j with stride-m A access; rewritten to p→i→j
                  so A[p*m+i] streams sequentially across p for fixed i.
                  Outer i tiled in TILE_I blocks.
   - mat_mul_abt: transpose B on-the-fly → becomes i→j→p dot-product,
                  inner p loop stride-1 for both A and B rows → vectorisable.
   - softmax_rows: unchanged logic, restrict added
   - sum_rows:    j-inner already fine; restrict added
*/

#include "matrix.h"
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------------------------------------------------------
   Tile sizes — tune to your L1 (typically 32 KB).
   128 floats × 128 floats × 4 B = 64 KB for three tiles,
   so keep TILE small enough that two tiles fit in L1.
   --------------------------------------------------------------- */
#define TILE_I 64
#define TILE_J 64
#define TILE_P 64

/* ---------------------------------------------------------------
   C[m×n] = A[m×k] @ B[k×n]
   Loop order i→p→j with tiling over (i,j,p).
   A rows and C rows stream left-to-right; B rows reused in i-tile.
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
        int i_end = ii + TILE_I < m ? ii + TILE_I : m;
        for (int pp = 0; pp < k; pp += TILE_P) {
            int p_end = pp + TILE_P < k ? pp + TILE_P : k;
            for (int jj = 0; jj < n; jj += TILE_J) {
                int j_end = jj + TILE_J < n ? jj + TILE_J : n;
                for (int i = ii; i < i_end; i++) {
                    for (int p = pp; p < p_end; p++) {
                        float a = A[i * k + p];
                        const float * restrict brow = B + p * n + jj;
                        float       * restrict crow = C + i * n + jj;
                        int jlen = j_end - jj;
                        /* inner loop — stride-1 on both brow and crow */
                        for (int j = 0; j < jlen; j++)
                            crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------
   C[m×n] = A^T @ B   where A is stored as [k×m], B as [k×n].
   Element A^T[i,p] = A[p*m + i].
   Original loop (i→p→j) reads A with stride m each inner step → bad.
   Rewritten as p→i→j:  for fixed p, A[p*m+i] is contiguous over i,
   and B[p*n+j] is contiguous over j.  Tile over i so C rows stay hot.
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
    for (int ii = 0; ii < m; ii += TILE_I) {
        int i_end = ii + TILE_I < m ? ii + TILE_I : m;
        for (int pp = 0; pp < k; pp += TILE_P) {
            int p_end = pp + TILE_P < k ? pp + TILE_P : k;
            for (int p = pp; p < p_end; p++) {
                const float * restrict arow = A + p * m + ii; /* A^T row-p, cols ii..i_end */
                const float * restrict brow = B + p * n;
                for (int i = ii; i < i_end; i++) {
                    float a = arow[i - ii];
                    float * restrict crow = C + i * n;
                    /* j loop — stride-1 on brow and crow */
                    for (int j = 0; j < n; j++)
                        crow[j] += a * brow[j];
                }
            }
        }
    }
}

/* ---------------------------------------------------------------
   C[m×n] = A[m×k] @ B^T   where B is stored as [n×k].
   Element B^T[p,j] = B[j*k + p].
   Rewrite as i→j→p:  for fixed (i,j) compute dot(A[i,:], B[j,:]).
   Both A-row and B-row are stride-1 over p → auto-vectorisable.
   Tile (i,j) so partial sums stay in registers.
   --------------------------------------------------------------- */
void mat_mul_abt(float * restrict C,
                 const float * restrict A,
                 const float * restrict B,
                 int m, int k, int n)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(m > 32)
    #endif
    for (int ii = 0; ii < m; ii += TILE_I) {
        int i_end = ii + TILE_I < m ? ii + TILE_I : m;
        for (int jj = 0; jj < n; jj += TILE_J) {
            int j_end = jj + TILE_J < n ? jj + TILE_J : n;
            for (int i = ii; i < i_end; i++) {
                const float * restrict arow = A + i * k;
                for (int j = jj; j < j_end; j++) {
                    const float * restrict brow = B + j * k;
                    float s = 0.0f;
                    /* dot product — stride-1 on both sides */
                    for (int p = 0; p < k; p++)
                        s += arow[p] * brow[p];
                    C[i * n + j] = s;
                }
            }
        }
    }
}

/* ---------------------------------------------------------------
   Broadcast bias — straightforward, restrict lets compiler vectorise
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
   Element-wise ops — all stride-1, restrict unlocks auto-vectorisation
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
        float mx = rs[0];
        for (int j = 1; j < cols; j++) if (rs[j] > mx) mx = rs[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) { rd[j] = expf(rs[j] - mx); sum += rd[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < cols; j++) rd[j] *= inv;
    }
}

void vec_mul(float * restrict dst, const float * restrict b, int n)
{
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
    for (int i = 0; i < n; i++) dst[i] = val;
}

/* Sum across rows: out[j] = Σ_i src[i*cols+j]
   Loop i-outer, j-inner so src rows are read sequentially. */
void sum_rows(float * restrict out,
              const float * restrict src,
              int rows, int cols)
{
    vec_fill(out, 0.0f, cols);
    for (int i = 0; i < rows; i++) {
        const float * restrict srow = src + i * cols;
        for (int j = 0; j < cols; j++)
            out[j] += srow[j];
    }
}
