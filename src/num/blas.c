/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>

#include "misc/misc.h"

#ifdef USE_MACPORTS
#include <cblas_openblas.h>
#elif USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#ifdef USE_CUDA
#include "num/gpuops.h"

#include <cublas.h>
#endif
#include "blas.h"



void blas_cgemm(char transa, char transb, long M, long N,  long K, const complex float alpha, long lda, const complex float A[K][lda], long ldb, const complex float B[N][ldb], const complex float beta, long ldc, complex float C[N][ldc])
{
#ifdef USE_CUDA
#define CUCOMPLEX(x) (((union { cuComplex cu; complex float std; }){ .std = (x) }).cu)
        if (cuda_ondevice(A)) {

                cublasCgemm(transa, transb, M, N, K,  CUCOMPLEX(alpha),
                                (const cuComplex*)A, lda,
                                (const cuComplex*)B, ldb, CUCOMPLEX(beta),
                                (cuComplex*)C, ldc);
        } else
#endif
        cblas_cgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans), ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans), M, N, K, (void*)&alpha, (void*)A, lda, (void*)B, ldb, (void*)&beta, (void*)C, ldc);
}


void (blas_matrix_multiply)(long M, long N, long K, complex float C[N][M], const complex float A[K][M], const complex float B[N][K])
{
	blas_cgemm('N', 'N', M, N, K, 1., M, A, K, B, 0., M, C);
}

void (blas_matrix_zfmac)(long M, long N, long K, complex float* C, const complex float* A, char transa, const complex float* B, char transb)
{
    	assert((transa == 'N') || (transa == 'T') || (transa == 'C'));
    	assert((transb == 'N') || (transb == 'T') || (transb == 'C'));

    	long lda = (transa == 'N' ? M: K);
    	long ldb = (transb == 'N' ? K: N);

	blas_cgemm(transa, transb, M, N, K, 1., lda, *(complex float (*)[K][lda])A, ldb, *(complex float (*)[N][ldb])B, 1., M, *(complex float (*)[N][M])C);

}


void (blas_csyrk)(char uplo, char trans, long N, long K, const complex float alpha, long lda, const complex float A[][lda], complex float beta, long ldc, complex float C[][ldc])
{
	assert('U' == uplo);
	assert(('T' == trans) || ('N' == trans));

	cblas_csyrk(CblasColMajor, CblasUpper, ('T' == trans) ? CblasTrans : CblasNoTrans, N, K, (void*)&alpha, (void*)A, lda, (void*)&beta, (void*)C, ldc);
}

void blas_cgemv(char trans, long M, long N, complex float alpha, long lda, const complex float A[N][lda], long incx, const complex float* x, complex float beta, long incy, complex float* y)
{
#ifdef USE_CUDA
#define CUCOMPLEX(x) (((union { cuComplex cu; complex float std; }){ .std = (x) }).cu)
        if (cuda_ondevice(A)) {

                cublasCgemv	(trans, M, N, CUCOMPLEX(alpha),
                                (const cuComplex*)A, lda,
                                (const cuComplex*)x, incx,
                                CUCOMPLEX(beta), (cuComplex*)y, incy);
        } else
#endif
        cblas_cgemv(	CblasColMajor, ('T' == trans) ? CblasTrans : (('C' == trans) ? CblasConjTrans : CblasNoTrans), M, N, (void*)&alpha,
			(void*)A, lda,
			(void*)x, incx,
			(void*)&beta, (void*)y, incy);
}

void (blas_gemv_zfmac)(long M, long N, complex float* y, const complex float* A, char trans, const complex float* x)
{
    	assert((trans == 'N') || (trans == 'T') || (trans == 'C'));
	blas_cgemv(trans,M, N, 1., M, *(complex float (*)[N][M])A, 1, x, 1., 1, y);
}

void blas_sgemv(char trans, long M, long N, float alpha, long lda, const float A[N][lda], long incx, const float* x, float beta, long incy, float* y)
{
#ifdef USE_CUDA

        if (cuda_ondevice(A)) {

                cublasSgemv(	trans, M, N, alpha,
                                (const float*)A, lda,
                                x, incx,
                                beta, y, incy);
        } else
#endif
        cblas_sgemv(	CblasColMajor, ('T' == trans) ? CblasTrans : CblasNoTrans, M, N, alpha,
			(const float*)A, lda,
			x, incx,
			beta, y, incy);
}

void (blas_gemv_fmac)(long M, long N, float* y, const float* A, char trans, const float* x)
{
    	assert((trans == 'N') || (trans == 'T'));
	blas_sgemv(trans,M, N, 1., M, *(float (*)[N][M])A, 1, x, 1., 1, y);
}
