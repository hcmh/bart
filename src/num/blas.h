
#include <complex.h>

#include "misc/misc.h"

extern void blas_matrix_multiply(long M, long N, long K, complex float C[N][M], const complex float A[K][M], const complex float B[N][K]);
extern void blas_matrix_zfmac(long M, long K, long N, complex float* C, const complex float* A, char transa, const complex float* B, char transb);

extern void blas_cgemm(char transa, char transb, long M, long N, long K, const complex float alpha, long lda, const complex float A[M][lda], long ldb, const complex float B[K][ldb], const complex float beta, long ldc, complex float C[M][ldc]);
extern void blas_csyrk(char uplow, char trans, long N, long K, complex float alpha, long lda, const complex float A[*][lda], complex float beta, long ldc, complex float C[*][ldc]);

extern void blas_cgemv(char trans, long M, long N, complex float alpha, long lda, const complex float A[N][lda], long incx, const complex float* x, complex float beta, long incy, complex float* y);
extern void blas_gemv_zfmac(long M, long N, complex float* y, const complex float* A, char trans, const complex float* x);
extern void blas_sgemv(char trans, long M, long N, float alpha, long lda, const float A[N][lda], long incx, const float* x, float beta, long incy, float* y);
extern void blas_gemv_fmac(long M, long N, float* y, const float* A, char trans, const float* x);