/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * This file defines basic operations on vectors of floats/complex floats
 * for operations on the GPU. See the CPU version (vecops.c) for more
 * information.
 */

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <curand_kernel.h>

#include "num/gpukrnls.h"

#if 1
// see Dara's src/calib/calibcu.cu for how to get
// runtime info



#define CHECK(call)                                                            
{                                                                              
    const cudaError_t error = call;                                            
    if (error != cudaSuccess)                                                  
    {                                                                          
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 
        fprintf(stderr, "code: %d, reason: %s\n", error,                       
                cudaGetErrorString(error));                                    
    }                                                                          
}

// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024

static int blocksize(int N)
{
	return BLOCKSIZE;
}

static long gridsize(long N)
{
	return (N + BLOCKSIZE - 1) / BLOCKSIZE;
}
#else
// http://stackoverflow.com/questions/5810447/cuda-block-and-grid-size-efficiencies

#define WARPSIZE 32
#define MAXBLOCKS (16 * 8)
// 16 multi processor times 8 blocks

#define MIN(x, y) ((x < y) ? (x) : (y))
#define MAX(x, y) ((x > y) ? (x) : (y))

static int blocksize(int N)
{
	int warps_total = (N + WARPSIZE - 1) / WARPSIZE;
	int warps_block = MAX(1, MIN(4, warps_total));

	return WARPSIZE * warps_block;
}

static long gridsize(long N)
{
	int warps_total = (N + WARPSIZE - 1) / WARPSIZE;
	int warps_block = MAX(1, MIN(4, warps_total));

	return MIN(MAXBLOCKS, MAX(1, warps_total / warps_block));
}
#endif

__global__ void kern_float2double(long N, double* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src[i];
}

extern "C" void cuda_float2double(long N, double* dst, const float* src)
{
	kern_float2double<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_double2float(long N, float* dst, const double* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src[i];
}

extern "C" void cuda_double2float(long N, float* dst, const double* src)
{
	kern_double2float<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_xpay(long N, float beta, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = dst[i] * beta + src[i];
}

extern "C" void cuda_xpay(long N, float beta, float* dst, const float* src)
{
	kern_xpay<<<gridsize(N), blocksize(N)>>>(N, beta, dst, src);
}


__global__ void kern_axpbz(long N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = a1 * src1[i] + a2 * src2[i];
}


extern "C" void cuda_axpbz(long N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	kern_axpbz<<<gridsize(N), blocksize(N)>>>(N, dst, a1, src1, a2, src2);
}


__global__ void kern_smul(long N, float alpha, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = alpha * src[i];
}

extern "C" void cuda_smul(long N, float alpha, float* dst, const float* src)
{
	kern_smul<<<gridsize(N), blocksize(N)>>>(N, alpha, dst, src);
}

__global__ void kern_smul_ptr(long N, const float* alpha, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = alpha[0] * src[i];
}

extern "C" void cuda_smul_ptr(long N, const float* alpha, float* dst, const float* src)
{
	kern_smul_ptr<<<gridsize(N), blocksize(N)>>>(N, alpha, dst, src);
}


typedef void (*cuda_3op_f)(long N, float* dst, const float* src1, const float* src2);

extern "C" void cuda_3op(cuda_3op_f krn, int N, float* dst, const float* src1, const float* src2)
{
	krn<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_add(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] + src2[i];
}

extern "C" void cuda_add(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_add, N, dst, src1, src2);
}

__global__ void kern_sadd(long N, float val, float* dst, const float* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] + val;
}

extern "C" void cuda_sadd(long N, float val, float* dst, const float* src1)
{
	kern_sadd<<<gridsize(N), blocksize(N)>>>(N, val, dst, src1);
}

__global__ void kern_zsadd(long N, cuFloatComplex val, cuFloatComplex* dst, const cuFloatComplex* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCaddf(src1[i], val);
}

extern "C" void cuda_zsadd(long N, _Complex float val, _Complex float* dst, const _Complex float* src1)
{
	kern_zsadd<<<gridsize(N), blocksize(N)>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
}

__global__ void kern_sub(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] - src2[i];
}

extern "C" void cuda_sub(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_sub, N, dst, src1, src2);
}


__global__ void kern_mul(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] * src2[i];
}

extern "C" void cuda_mul(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_mul, N, dst, src1, src2);
}

__global__ void kern_div(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = src1[i] / src2[i];
}

extern "C" void cuda_div(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_div, N, dst, src1, src2);
}


__global__ void kern_fmac(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] += src1[i] * src2[i];
}

extern "C" void cuda_fmac(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_fmac, N, dst, src1, src2);
}


__global__ void kern_fmac2(long N, double* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] += src1[i] * src2[i];
}

extern "C" void cuda_fmac2(long N, double* dst, const float* src1, const float* src2)
{
	kern_fmac2<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_zsmul(long N, cuFloatComplex val, cuFloatComplex* dst, const cuFloatComplex* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], val);
}

extern "C" void cuda_zsmul(long N, _Complex float alpha, _Complex float* dst, const _Complex float* src1)
{
	kern_zsmul<<<gridsize(N), blocksize(N)>>>(N, make_cuFloatComplex(__real(alpha), __imag(alpha)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
}


__global__ void kern_zmul(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], src2[i]);
}

extern "C" void cuda_zmul(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmul<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zdiv(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float abs = cuCabsf(src2[i]);

		dst[i] = (0. == abs) ? make_cuFloatComplex(0., 0.) : cuCdivf(src1[i], src2[i]);
	}
}

extern "C" void cuda_zdiv(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zdiv<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmac(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCaddf(dst[i], cuCmulf(src1[i], src2[i]));
}


extern "C" void cuda_zfmac(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmac<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmac2(long N, cuDoubleComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCadd(dst[i], cuComplexFloatToDouble(cuCmulf(src1[i], src2[i])));
}

extern "C" void cuda_zfmac2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmac2<<<gridsize(N), blocksize(N)>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zmulc(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCmulf(src1[i], cuConjf(src2[i]));
}

extern "C" void cuda_zmulc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmulc<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zfmacc(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCaddf(dst[i], cuCmulf(src1[i], cuConjf(src2[i])));
}


extern "C" void cuda_zfmacc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmacc<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_zfmacc2(long N, cuDoubleComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCadd(dst[i], cuComplexFloatToDouble(cuCmulf(src1[i], cuConjf(src2[i]))));
}


extern "C" void cuda_zfmacc2(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zfmacc2<<<gridsize(N), blocksize(N)>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_pow(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = powf(src1[i], src2[i]);
}

extern "C" void cuda_pow(long N, float* dst, const float* src1, const float* src2)
{
	cuda_3op(kern_pow, N, dst, src1, src2);
}

__device__ cuDoubleComplex zexpD(cuDoubleComplex x)
{
	double sc = exp(cuCreal(x));
	double si;
	double co;
	sincos(cuCimag(x), &si, &co);
	return make_cuDoubleComplex(sc * co, sc * si);
}

__device__ cuFloatComplex zexp(cuFloatComplex x)
{
	float sc = expf(cuCrealf(x));
	float si;
	float co;
	sincosf(cuCimagf(x), &si, &co);
	return make_cuFloatComplex(sc * co, sc * si);
}

__device__ cuFloatComplex zsin(cuFloatComplex x)
{
	float si;
	float co;
	float sih;
	float coh;
	sincosf(cuCrealf(x), &si, &co);
	sih = sinhf(cuCimagf(x));
	coh = coshf(cuCimagf(x));
	return make_cuFloatComplex(si * coh , co * sih);
}

__device__ cuFloatComplex zcos(cuFloatComplex x)
{
	float si;
	float co;
	float sih;
	float coh;
	sincosf(cuCrealf(x), &si, &co);
	sih = sinhf(cuCimagf(x));
	coh = coshf(cuCimagf(x));
	return make_cuFloatComplex(co * coh , -si * sih);
}

__device__ float zarg(cuFloatComplex x)
{
	return atan2(cuCimagf(x), cuCrealf(x));
}

__device__ float zabs(cuFloatComplex x)
{
	return cuCabsf(x);
}

__device__ cuFloatComplex zlog(cuFloatComplex x)
{
	return make_cuFloatComplex(log(cuCabsf(x)), zarg(x));
}


// x^y = e^{y ln(x)} = e^{y
__device__ cuFloatComplex zpow(cuFloatComplex x, cuFloatComplex y)
{
	return zexp(cuCmulf(y, zlog(x)));
}

__global__ void kern_zpow(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zpow(src1[i], src2[i]);
}

extern "C" void cuda_zpow(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zpow<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_sqrt(long N, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = sqrtf(fabs(src[i]));
}

extern "C" void cuda_sqrt(long N, float* dst, const float* src)
{
	kern_sqrt<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}


__global__ void kern_zconj(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuConjf(src[i]);
}

extern "C" void cuda_zconj(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zconj<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zcmp(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(((cuCrealf(src1[i]) == cuCrealf(src2[i])) && (cuCimagf(src1[i]) == cuCimagf(src2[i]))) ? 1. : 0, 0.);
}

extern "C" void cuda_zcmp(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zcmp<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}

__global__ void kern_zdiv_reg(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2, cuFloatComplex lambda)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = cuCdivf(src1[i], cuCaddf(src2[i], lambda));
}

extern "C" void cuda_zdiv_reg(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda)
{
	kern_zdiv_reg<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2, make_cuFloatComplex(__real(lambda), __imag(lambda)));
}


__global__ void kern_zphsr(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float abs = cuCabsf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		dst[i] = (0. == abs) ? make_cuFloatComplex(1., 0.) : (cuCdivf(src[i], make_cuFloatComplex(abs, 0.)));
	}
}

extern "C" void cuda_zphsr(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zphsr<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zexp(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zexp(src[i]);
}

extern "C" void cuda_zexp(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zexp<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zexpj(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float re = cuCrealf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		float im = cuCimagf(src[i]); // moved out, otherwise it triggers a compiler error in nvcc
		dst[i] = zexp(make_cuFloatComplex(-im, re));
	}
}

extern "C" void cuda_zexpj(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zexpj<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zlog(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride){

		float abs = cuCabsf(src[i]);
		dst[i] = (0. == abs) ? make_cuFloatComplex(0., 0.) : zlog(src[i]);
	}
}

extern "C" void cuda_zlog(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zlog<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zarg(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(zarg(src[i]), 0.);
}

extern "C" void cuda_zarg(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zarg<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zsin(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zsin(src[i]);
}

extern "C" void cuda_zsin(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zsin<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zcos(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = zcos(src[i]);
}

extern "C" void cuda_zcos(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zcos<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zabs(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(zabs(src[i]), 0.);
}

extern "C" void cuda_zabs(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zabs<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_exp(long N, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = expf(src[i]);
}

extern "C" void cuda_exp(long N, float* dst, const float* src)
{
	kern_exp<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}

__global__ void kern_log(long N, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = (0. == src[i]) ? 0. : logf(src[i]);
}

extern "C" void cuda_log(long N, float* dst, const float* src)
{
	kern_log<<<gridsize(N), blocksize(N)>>>(N, dst, src);
}


__global__ void kern_zatanr(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(atan(cuCrealf(src[i])), 0.);
}

extern "C" void cuda_zatanr(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zatanr<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_zacos(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(acosf(cuCrealf(src[i])), 0.);
}

extern "C" void cuda_zacos(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zacos<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}



/**
 * (GPU) Step (1) of soft thesholding, y = ST(x, lambda).
 * Only computes the residual, resid = MAX( (abs(x) - lambda)/abs(x)), 0 )
 *
 * @param N number of elements
 * @param lambda threshold parameter
 * @param d pointer to destination, resid
 * @param x pointer to input
 */
__global__ void kern_zsoftthresh_half(long N, float lambda, cuFloatComplex* d, const cuFloatComplex* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float norm = cuCabsf(x[i]);
		float red = norm - lambda;
		//d[i] = (red > 0.) ? (cuCmulf(make_cuFloatComplex(red / norm, 0.), x[i])) : make_cuFloatComplex(0., 0.);
		d[i] = (red > 0.) ? make_cuFloatComplex(red / norm, 0.) : make_cuFloatComplex(0., 0.);
	}
}

extern "C" void cuda_zsoftthresh_half(long N, float lambda, _Complex float* d, const _Complex float* x)
{
	kern_zsoftthresh_half<<<gridsize(N), blocksize(N)>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
}


__global__ void kern_zsoftthresh(long N, float lambda, cuFloatComplex* d, const cuFloatComplex* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float norm = cuCabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (cuCmulf(make_cuFloatComplex(red / norm, 0.), x[i])) : make_cuFloatComplex(0., 0.);
	}
}


extern "C" void cuda_zsoftthresh(long N, float lambda, _Complex float* d, const _Complex float* x)
{
	kern_zsoftthresh<<<gridsize(N), blocksize(N)>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
}


__global__ void kern_softthresh_half(long N, float lambda, float* d, const float* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;

		d[i] = (red > 0.) ? (red / norm) : 0.;
	}
}

extern "C" void cuda_softthresh_half(long N, float lambda, float* d, const float* x)
{
	kern_softthresh_half<<<gridsize(N), blocksize(N)>>>(N, lambda, d, x);
}


__global__ void kern_softthresh(long N, float lambda, float* d, const float* x)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;

		d[i] = (red > 0.) ? (red / norm * x[i]) : 0.;
	}
}

extern "C" void cuda_softthresh(long N, float lambda, float* d, const float* x)
{
	kern_softthresh<<<gridsize(N), blocksize(N)>>>(N, lambda, d, x);
}


__global__ void kern_zreal(long N, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(cuCrealf(src[i]), 0.);
}

extern "C" void cuda_zreal(long N, _Complex float* dst, const _Complex float* src)
{
	kern_zreal<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}


__global__ void kern_zle(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex((cuCrealf(src1[i]) <= cuCrealf(src2[i])), 0.);
}

extern "C" void cuda_zle(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zle<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}


__global__ void kern_le(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = (src1[i] <= src2[i]);
}

extern "C" void cuda_le(long N, float* dst, const float* src1, const float* src2)
{
	kern_le<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__device__ cuFloatComplex cuDouble2Float(cuDoubleComplex x)
{
	return make_cuFloatComplex(cuCreal(x), cuCimag(x));
}

__device__ cuDoubleComplex cuFloat2Double(cuFloatComplex x)
{
	return make_cuDoubleComplex(cuCrealf(x), cuCimagf(x));
}

// identical copy in num/fft.c
__device__ double fftmod_phase(long length, int j)
{
	long center1 = length / 2;
	double shift = (double)center1 / (double)length;
	return ((double)j - (double)center1 / 2.) * shift;
}

__device__ cuDoubleComplex fftmod_phase2(long n, int j, bool inv, double phase)
{
	phase += fftmod_phase(n, j);
	double rem = phase - floor(phase);
	double sgn = inv ? -1. : 1.;
#if 1
	if (rem == 0.)
		return make_cuDoubleComplex(1., 0.);

	if (rem == 0.5)
		return make_cuDoubleComplex(-1., 0.);

	if (rem == 0.25)
		return make_cuDoubleComplex(0., sgn);

	if (rem == 0.75)
		return make_cuDoubleComplex(0., -sgn);
#endif
	return zexpD(make_cuDoubleComplex(0., M_PI * 2. * sgn * rem));
}

__global__ void kern_zfftmod(long N, cuFloatComplex* dst, const cuFloatComplex* src, unsigned int n, _Bool inv, double phase)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		for (int j = 0; j < n; j++)
			dst[i * n + j] = cuDouble2Float(cuCmul(fftmod_phase2(n, j, inv, phase),
						 cuFloat2Double(src[i * n + j])));
}

extern "C" void cuda_zfftmod(long N, _Complex float* dst, const _Complex float* src, unsigned int n, _Bool inv, double phase)
{
	kern_zfftmod<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src, n, inv, phase);
}



__global__ void kern_zmax(long N, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = (cuCrealf(src1[i]) > cuCrealf(src2[i])) ? src1[i] : src2[i];
}


extern "C" void cuda_zmax(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
{
	kern_zmax<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
}



#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


__global__ void kern_smax(long N, float val, float* dst, const float* src1)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = MAX(src1[i], val);
}


extern "C" void cuda_smax(long N, float val, float* dst, const float* src1)
{
	kern_smax<<<gridsize(N), blocksize(N)>>>(N, val, dst, src1);
}


__global__ void kern_max(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = MAX(src1[i], src2[i]);
}


extern "C" void cuda_max(long N, float* dst, const float* src1, const float* src2)
{
	kern_max<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}


__global__ void kern_min(long N, float* dst, const float* src1, const float* src2)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = MIN(src1[i], src2[i]);
}


extern "C" void cuda_min(long N, float* dst, const float* src1, const float* src2)
{
	kern_min<<<gridsize(N), blocksize(N)>>>(N, dst, src1, src2);
}

__global__ void kern_zsmax(long N, float val, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride) {

		dst[i].x = MAX(src[i].x, val);
		dst[i].y = 0.0;
	}
}

extern "C" void cuda_zsmax(long N, float alpha, _Complex float* dst, const _Complex float* src)
{
	kern_zsmax<<<gridsize(N), blocksize(N)>>>(N, alpha, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
}

__global__ void kern_reduce_zsum(long N, cuFloatComplex* dst)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	cuFloatComplex sum = make_cuFloatComplex(0., 0.);

	for (long i = start; i < N; i += stride)
		sum = cuCaddf(sum, dst[i]);

	if (start < N)
		dst[start] = sum;
}

extern "C" void cuda_zsum(long N, _Complex float* dst)
{
	int B = blocksize(N);

	while (N > 1) {

		kern_reduce_zsum<<<1, B>>>(N, (cuFloatComplex*)dst);
		N = MIN(B, N);
		B /= 32;
	}
}

__global__ void kern_zconvcorr_3D(cuFloatComplex* dst, const cuFloatComplex* src, const cuFloatComplex* krn,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x >= NO0) || (y >= NO1) || (z >= NO2))
		return;

	for(int k2 = 0; k2 < NK2; k2++)
	for(int k1 = 0; k1 < NK1; k1++)
	for(int k0 = 0; k0 < NK0; k0++)
	{
		int oind = x + NO0 * y + NO0 * NO1 * z;
		int kind = (k0 + NK0 * k1 + NK0 * NK1 * k2);
		if (conv)
			kind = (NK0 * NK1 * NK2) - kind - 1;
		int iind = (x + k0) + NI0 * (y + k1) + NI0 * NI1 * (z + k2);

		dst[oind] = cuCaddf(dst[oind], cuCmulf(src[iind], krn[kind]));
	}
}

extern "C" void cuda_zconvcorr_3D(_Complex float* dst, const _Complex float* src, const _Complex float* krn, long odims[3], long idims[3], long kdims[3], _Bool conv)
{
	dim3 threadsPerBlock(32, 32, 1);
	dim3 numBlocks(	(0 == odims[0] % threadsPerBlock.x) ? odims[0] / threadsPerBlock.x : odims[0] / threadsPerBlock.x + 1,
			(0 == odims[1] % threadsPerBlock.y) ? odims[1] / threadsPerBlock.y : odims[1] / threadsPerBlock.y + 1,
			(0 == odims[2] % threadsPerBlock.z) ? odims[2] / threadsPerBlock.z : odims[2] / threadsPerBlock.z + 1);

	kern_zconvcorr_3D<<<numBlocks, threadsPerBlock>>>((cuFloatComplex*) dst, (cuFloatComplex*) src, (cuFloatComplex*) krn,
								odims[0], odims[1], odims[2],
								idims[0], idims[1], idims[2],
								kdims[0], kdims[1], kdims[2], conv);
}

__global__ void kern_zconvcorr_3D_CF(cuFloatComplex* dst, const cuFloatComplex* src, const cuFloatComplex* krn,
					long NOm, long NKm,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NOm * NO0 * NO1 * NO2))
		return;

	int om = i % NOm;
	i = (i - om) /NOm;
	int o0 = i % NO0;
	i = (i - o0) /NO0;
	int o1 = i % NO1;
	i = (i - o1) / NO1;
	int o2 = i % NO2;

	cuFloatComplex result = make_cuFloatComplex(0., 0.);
	int oind = om + NOm * o0 + NOm * NO0 * o1 + NOm * NO0 * NO1 * o2;

	for(int k2 = 0; k2 < NK2; k2++)
	for(int k1 = 0; k1 < NK1; k1++)
	for(int k0 = 0; k0 < NK0; k0++)
	for(int km = 0; km < NKm; km++){
		
		int kind = om + NOm * km; // matrix index
		if (conv)
			kind += (NOm * NKm) * ((NK0 - k0 - 1) + NK0 * (NK1 - k1 - 1) + NK0 * NK1 * (NK2 - k2 - 1));
		else
			kind += (NOm * NKm) * (k0 + NK0 * k1 + NK0 * NK1 * k2);
		int iind = km + NKm * (o0 + k0) + NKm * NI0 * (o1 + k1) + NKm * NI0 * NI1 * (o2 + k2);

		result = cuCaddf(result, cuCmulf(src[iind], krn[kind]));
	}

	dst[oind] = cuCaddf(dst[oind], result);
}

extern "C" void cuda_zconvcorr_3D_CF(_Complex float* dst, const _Complex float* src, const _Complex float* krn, long odims[5], long idims[5], long kdims[5], _Bool conv)
{
	long N = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];

	kern_zconvcorr_3D_CF<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) dst, (cuFloatComplex*) src, (cuFloatComplex*) krn,
								kdims[0], kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4], conv);
}

__global__ void kern_zconvcorr_3D_CF_TK(cuFloatComplex* krn, const cuFloatComplex* src, const cuFloatComplex* out,
					long NOm, long NKm,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NOm * NKm * NK0 * NK1 * NK2))
		return;

	int om = i % NOm;
	i = (i - om) /NOm;
	int km = i % NKm;
	i = (i - km) /NKm;
	int k0 = i % NK0;
	i = (i - k0) /NK0;
	int k1 = i % NK1;
	i = (i - k1) /NK1;
	int k2 = i % NK2;

	int kind = om + NOm * km;
		if (conv)
			kind += (NOm * NKm) * ((NK0 - k0 - 1) + NK0 * (NK1 - k1 - 1) + NK0 * NK1 * (NK2 - k2 - 1));
		else
			kind += (NOm * NKm) * (k0 + NK0 * k1 + NK0 * NK1 * k2);

	cuFloatComplex result = make_cuFloatComplex(0., 0.);

	for(int o2 = 0; o2 < NO2; o2++)
	for(int o1 = 0; o1 < NO1; o1++)
	for(int o0 = 0; o0 < NO0; o0++){
	
		int oind = om + NOm * o0 + NOm * NO0 * o1 + NOm * NO0 * NO1 * o2;
		int iind = km + NKm * (o0 + k0) + NKm * NI0 * (o1 + k1) + NKm * NI0 * NI1 * (o2 + k2);
		result = cuCaddf(result, cuCmulf(src[iind], out[oind]));
	}

	krn[kind] = cuCaddf(krn[kind], result);
}

extern "C" void cuda_zconvcorr_3D_CF_TK(_Complex float* krn, const _Complex float* src, const _Complex float* out, long odims[5], long idims[5], long kdims[5], _Bool conv)
{
	assert(kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4] < 1073741824);
	long N = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];

	kern_zconvcorr_3D_CF_TK<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) krn, (cuFloatComplex*) src, (cuFloatComplex*) out,
								kdims[0], kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4], conv);
}


__global__ void kern_zconvcorr_3D_CF_TI(cuFloatComplex* im, const cuFloatComplex* out, const cuFloatComplex* krn,
					long NOm, long NKm,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NKm * NI0 * NI1 * NI2))
		return;

	int km = i % NKm;
	i = (i - km) / NKm;
	int i0 = i % NI0;
	i = (i - i0) / NI0;
	int i1 = i % NI1;
	i = (i - i1) / NI1;
	int i2 = i % NI2;

	int iind = km + NKm * i0 + NKm * NI0 * i1 + NKm * NI0 * NI1 * i2;

	cuFloatComplex result = make_cuFloatComplex(0., 0.);

	for(int k2 = 0; k2 < NK2; k2++)
	for(int k1 = 0; k1 < NK1; k1++)
	for(int k0 = 0; k0 < NK0; k0++)
	for(int om = 0; om < NOm; om++){
	
		int o0 = i0 - k0;
		int o1 = i1 - k1;
		int o2 = i2 - k2;
		
		int oind = om + NOm * o0 + NOm * NO0 * o1 + NOm * NO0 * NO1 * o2;
		int kind = om + NOm * km; // matrix index
		if (conv)
			kind += (NOm * NKm) * ((NK0 - k0 - 1) + NK0 * (NK1 - k1 - 1) + NK0 * NK1 * (NK2 - k2 - 1));
		else
			kind += (NOm * NKm) * (k0 + NK0 * k1 + NK0 * NK1 * k2);

		if ((0 <= o0) && (0 <= o1) && (0 <= o2) && (NO0 > o0) && (NO1 > o1) && (NO2 > o2))
			result = cuCaddf(result, cuCmulf(out[oind], krn[kind]));
	}

	im[iind] = cuCaddf(im[iind], result);
}

extern "C" void cuda_zconvcorr_3D_CF_TI(_Complex float* im, const _Complex float* out, const _Complex float* krn, long odims[5], long idims[5], long kdims[5], _Bool conv)
{
	long N = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	kern_zconvcorr_3D_CF_TI<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) im, (cuFloatComplex*) out, (cuFloatComplex*) krn,
								kdims[0], kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4], conv);
}


__global__ void kern_im2col_valid_loop_in(	cuFloatComplex* dst, const cuFloatComplex* src,
						long NC,
						long OX, long OY, long OZ,
						long IX, long IY, long IZ,
						long KX, long KY, long KZ)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NC * IX * IY * IZ))
		return;

	int i0 = i;

	int c = i % NC;
	i = (i - c) / NC;
	int ix = i % IX;
	i = (i - ix) / IX;
	int iy = i % IY;
	i = (i - iy) / IY;
	int iz = i % IZ;
	i = (i - iz) / IZ;

	for (int kx = 0; kx < KX; kx++)
	for (int ky = 0; ky < KY; ky++)
	for (int kz = 0; kz < KZ; kz++) {

		int ox = ix - kx;
		int oy = iy - ky;
		int oz = iz - kz;

		int o0 = c + NC * (kx + KX * (ky + KY * (kz + KZ * (ox + OX * (oy + OY * oz)))));
		
		if ((0 <= ox) && (0 <= oy) && (0 <= oz) && (OX > ox) && (OY > oy) && (OZ > oz))
			dst[o0] = src [i0];
	}
}

__global__ void kern_im2col_valid_loop_out(	cuFloatComplex* dst, const cuFloatComplex* src,
						long NC,
						long OX, long OY, long OZ,
						long IX, long IY, long IZ,
						long KX, long KY, long KZ)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < OX * OY * OZ * KX * KY * KZ))
		return;

	int kx = i % KX;
	i = (i - kx) / KX;
	int ky = i % KY;
	i = (i - ky) / KY;
	int kz = i % KZ;
	i = (i - kz) / KZ;
	
	int ox = i % OX;
	i = (i - ox) / OX;
	int oy = i % OY;
	i = (i - oy) / OY;
	int oz = i % OZ;

	int o0 = NC * (kx + KX * (ky + KY * (kz + KZ * (ox + OX * (oy + OY * oz)))));
	int i0 = NC * ((ox + kx) + IX * ((oy + ky) + IY * (oz + kz)));

	for (int c = 0; c < NC; c++)
		dst[o0 + c] = src[i0 + c]; 
}

extern "C" void cuda_im2col(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5])
{
	if (9 < kdims[2] * kdims[3] * kdims[4]) {

		long N = kdims[2] * kdims[3] * kdims[4] * odims[2] * odims[3] * odims[4];

		kern_im2col_valid_loop_out<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4]);
	} else {
	
		long N = idims[1] * idims[2] * idims[3] * idims[4];

		kern_im2col_valid_loop_in<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4]);
	}
}


__global__ void kern_im2col_transp(	cuFloatComplex* dst, const cuFloatComplex* src,
					long NC,
					long OX, long OY, long OZ,
					long IX, long IY, long IZ,
					long KX, long KY, long KZ)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NC * IX * IY * IZ))
		return;
	
	int c = i % NC;
	i = (i - c) / NC;
	int ix = i % IX;
	i = (i - ix) / IX;
	int iy = i % IY;
	i = (i - iy) / IY;
	int iz = i % IZ;

	cuFloatComplex result = make_cuFloatComplex(0., 0.);

	for (int kx = 0; kx < KX; kx++)
	for (int ky = 0; ky < KY; ky++)
	for (int kz = 0; kz < KZ; kz++) {

		int ox = ix - kx;
		int oy = iy - ky;
		int oz = iz - kz;

		int index = c + (NC * KX * KY * KZ) * (ox + OX * oy + OX * OY *oz) + NC * (kx + KX * (ky + KY * kz));

		if ((0 <= ox) && (0 <= oy) && (0 <= oz) && (OX > ox) && (OY > oy) && (OZ > oz))
			result = cuCaddf(result, src[index]);
	}
	
	i = blockIdx.x * blockDim.x + threadIdx.x;
	dst[i] = cuCaddf(result, dst[i]);
}

extern "C" void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5])
{
	long N = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	kern_im2col_transp<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
								kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4]);
}


__global__ void kern_pdf_gauss(long N, float mu, float sig, float* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < N; i += stride)
		dst[i] = expf(- (src[i] - mu) * (src[i] - mu) / (2 * sig * sig)) / (sqrtf(2 * M_PI) * sig);
}

extern "C" void cuda_pdf_gauss(long N, float mu, float sig, float* dst, const float* src)
{
	kern_pdf_gauss<<<gridsize(N), blocksize(N)>>>(N, mu, sig, dst, src);
}


__global__ void kern_real(int N, float* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCrealf(src[i]);
}

extern "C" void cuda_real(long N, float* dst, const _Complex float* src)
{
	kern_real<<<gridsize(N), blocksize(N)>>>(N, dst, (cuFloatComplex*)src);
}

__global__ void kern_imag(int N, float* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = cuCimagf(src[i]);
}

extern "C" void cuda_imag(long N, float* dst, const _Complex float* src)
{
	kern_imag<<<gridsize(N), blocksize(N)>>>(N, dst, (cuFloatComplex*)src);
}

__global__ void kern_zcmpl_real(int N, cuFloatComplex* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(src[i], 0);
}

extern "C" void cuda_zcmpl_real(long N, _Complex float* dst, const float* src)
{
	kern_zcmpl_real<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, src);
}

__global__ void kern_zcmpl_imag(int N, cuFloatComplex* dst, const float* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(0., src[i]);
}

extern "C" void cuda_zcmpl_imag(long N, _Complex float* dst, const float* src)
{
	kern_zcmpl_imag<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, src);
}

__global__ void kern_zcmpl(int N, cuFloatComplex* dst, const float* real_src, const float* imag_src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = make_cuFloatComplex(real_src[i], imag_src[i]);
}

extern "C" void cuda_zcmpl(long N, _Complex float* dst, const float* real_src, const float* imag_src)
{
	kern_zcmpl<<<gridsize(N), blocksize(N)>>>(N, (cuFloatComplex*)dst, real_src, imag_src);
}

__global__ void kern_zfill(int N, cuFloatComplex val, cuFloatComplex* dst)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride)
		dst[i] = val;
}

extern "C" void cuda_zfill(long N, _Complex float val, _Complex float* dst)
{
	kern_zfill<<<gridsize(N), blocksize(N)>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst);
}


__global__ void kern_gaussian_rand(long N, curandState *states, cuFloatComplex* dst)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	curandState *state = states + start;

	curand_init(1234, start, 0, state);
	
	for (int i = start; i < N; i += stride)
	{
		float rand_real = curand_normal(state);
		float rand_imag = curand_normal(state);
		dst[i] = make_cuFloatComplex(rand_real, rand_imag);
	}

}

extern "C" void cuda_zgaussian_rand(long N, _Complex float* dst)
{
	curandState *states = NULL;
	
	CHECK(cudaMalloc((void **)&states, sizeof(curandState) * gridsize(N) * blocksize(N)));
	kern_gaussian_rand<<<gridsize(N), blocksize(N)>>>(N, states, (cuFloatComplex*)dst);
	
	CHECK(cudaFree(states));
}

__global__ void kern_get_max(long N, float* dst, cuFloatComplex* src){
	
	int start = threadIdx.x + (blockDim.x * blockIdx.x);
	int stride = blockDim.x * gridDim.x;
	
	for (int i = start; i < N; i += stride)
	{
		if(cuCabsf(src[i]) > dst[0])
			dst[0] = cuCabsf(src[i]);
	}
}


extern "C" void cuda_get_max(long N, float* dst, const float* src)
{
	float* temp=NULL;
	CHECK(cudaMalloc((void **)&temp, sizeof(float)));
	CHECK(cudaMemset((void*)temp, 0, sizeof(float)));

	kern_get_max <<<1, 1>>> (N/2, temp, (cuFloatComplex*)src);

	CHECK(cudaMemcpy((void*)dst, (void*)temp, sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaFree(temp));
}