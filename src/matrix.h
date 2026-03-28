#ifndef MATRIX_H
#define MATRIX_H

/* All matrices are row-major flat float arrays.
   Element (i,j) of an [rows × cols] matrix M is M[i*cols + j]. */

/* C = A @ B   (m×k) @ (k×n) → (m×n) */
void mat_mul(float *C, const float *A, const float *B, int m, int k, int n);

/* C = A^T @ B   (k×m)^T @ (k×n) → (m×n) */
void mat_mul_atb(float *C, const float *A, const float *B, int k, int m, int n);

/* C = A @ B^T   (m×k) @ (n×k)^T → (m×n) */
void mat_mul_abt(float *C, const float *A, const float *B, int m, int k, int n);

/* dst[i*cols + j] += bias[j]   broadcast bias over rows */
void mat_add_bias(float *dst, const float *bias, int rows, int cols);

/* dst[i] = max(0, src[i]) */
void relu(float *dst, const float *src, int n);

/* dst[i] = (src[i] > 0) ? 1 : 0   (ReLU derivative) */
void relu_grad(float *dst, const float *src, int n);

/* Softmax along last dim: dst[i*cols .. i*cols+cols-1] for each row */
void softmax_rows(float *dst, const float *src, int rows, int cols);

/* Element-wise multiply: dst[i] *= b[i] */
void vec_mul(float *dst, const float *b, int n);

/* dst[i] += alpha * src[i] */
void vec_axpy(float *dst, float alpha, const float *src, int n);

/* dst[i] = val */
void vec_fill(float *dst, float val, int n);

/* Sum across rows: out[j] = sum_i src[i*cols + j] */
void sum_rows(float *out, const float *src, int rows, int cols);

#endif
