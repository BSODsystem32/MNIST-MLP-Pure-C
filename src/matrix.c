#include "matrix.h"
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* C[m×n] = A[m×k] @ B[k×n] */
void mat_mul(float *C, const float *A, const float *B, int m, int k, int n) {
    memset(C, 0, m * n * sizeof(float));
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(m > 32)

    #endif
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[i*k+p];
            for (int j = 0; j < n; j++)
                C[i*n+j] += a * B[p*n+j];
        }
}

/* C[m×n] = A^T[k×m] @ B[k×n] */
void mat_mul_atb(float *C, const float *A, const float *B, int k, int m, int n) {
    memset(C, 0, m * n * sizeof(float));
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(m > 32)

    #endif
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[p*m+i];
            for (int j = 0; j < n; j++)
                C[i*n+j] += a * B[p*n+j];
        }
}

/* C[m×n] = A[m×k] @ B^T[n×k] */
void mat_mul_abt(float *C, const float *A, const float *B, int m, int k, int n) {
    memset(C, 0, m * n * sizeof(float));
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(m > 32)

    #endif
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[i*k+p];
            for (int j = 0; j < n; j++)
                C[i*n+j] += a * B[j*k+p];
        }
}

void mat_add_bias(float *dst, const float *bias, int rows, int cols) {
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(rows > 32)

    #endif
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[i*cols+j] += bias[j];
}

void relu(float *dst, const float *src, int n) {
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(n > 1024)

    #endif
    for (int i = 0; i < n; i++)
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

void relu_grad(float *dst, const float *src, int n) {
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(n > 1024)

    #endif
    for (int i = 0; i < n; i++)
        dst[i] = src[i] > 0.0f ? 1.0f : 0.0f;
}

void softmax_rows(float *dst, const float *src, int rows, int cols) {
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(rows > 32)

    #endif
    for (int i = 0; i < rows; i++) {
        const float *rs = src + i*cols;
        float       *rd = dst + i*cols;
        float mx = rs[0];
        for (int j = 1; j < cols; j++) if (rs[j] > mx) mx = rs[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) { rd[j] = expf(rs[j]-mx); sum += rd[j]; }
        for (int j = 0; j < cols; j++) rd[j] /= sum;
    }
}

void vec_mul(float *dst, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] *= b[i];
}

void vec_axpy(float *dst, float alpha, const float *src, int n) {
    #ifdef _OPENMP

    #pragma omp parallel for schedule(static) if(n > 1024)

    #endif
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

void vec_fill(float *dst, float val, int n) {
    for (int i = 0; i < n; i++) dst[i] = val;
}

void sum_rows(float *out, const float *src, int rows, int cols) {
    vec_fill(out, 0.0f, cols);
    /* reduction over rows — not trivially parallel without atomics,
       keep serial (cols is small: 10/128/256) */
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j] += src[i*cols+j];
}
