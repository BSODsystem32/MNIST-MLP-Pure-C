#include "matrix.h"
#include <math.h>
#include <float.h>
#include <string.h>

void mat_mul(float *C, const float *A, const float *B, int m, int k, int n) {
    /* C[m×n] = A[m×k] @ B[k×n] */
    memset(C, 0, m * n * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[i * k + p];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[p * n + j];
        }
}

void mat_mul_atb(float *C, const float *A, const float *B, int k, int m, int n) {
    /* C[m×n] = A^T[m×k transposed from k×m] @ B[k×n]
       A is stored as [k×m], so A^T[i][p] = A[p*m + i] */
    memset(C, 0, m * n * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[p * m + i];  /* A^T[i][p] */
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[p * n + j];
        }
}

void mat_mul_abt(float *C, const float *A, const float *B, int m, int k, int n) {
    /* C[m×n] = A[m×k] @ B^T[n×k transposed from n×k]
       B is stored as [n×k], so B^T[p][j] = B[j*k + p] */
    memset(C, 0, m * n * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[i * k + p];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[j * k + p];  /* B^T[p][j] = B[j][p] */
        }
}

void mat_add_bias(float *dst, const float *bias, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[i * cols + j] += bias[j];
}

void relu(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

void relu_grad(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = src[i] > 0.0f ? 1.0f : 0.0f;
}

void softmax_rows(float *dst, const float *src, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row_s = src + i * cols;
        float       *row_d = dst + i * cols;

        /* Numerically stable: subtract max before exp */
        float mx = row_s[0];
        for (int j = 1; j < cols; j++)
            if (row_s[j] > mx) mx = row_s[j];

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row_d[j] = expf(row_s[j] - mx);
            sum += row_d[j];
        }
        for (int j = 0; j < cols; j++)
            row_d[j] /= sum;
    }
}

void vec_mul(float *dst, const float *b, int n) {
    for (int i = 0; i < n; i++)
        dst[i] *= b[i];
}

void vec_axpy(float *dst, float alpha, const float *src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] += alpha * src[i];
}

void vec_fill(float *dst, float val, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = val;
}

void sum_rows(float *out, const float *src, int rows, int cols) {
    vec_fill(out, 0.0f, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j] += src[i * cols + j];
}
