/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2014. Joseph Y Cheng.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019	Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 	Joseph Y Cheng <jycheng@stanford.edu>
 * 2015-2018	Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * CUDA support functions. The file exports gpu_ops of type struct vec_ops
 * for basic operations on single-precision floating pointer vectors defined
 * in gpukrnls.cu. See vecops.c for the CPU version.
 */

#ifdef USE_CUDA

#include <stdbool.h>
#include <assert.h>
#include <complex.h>
#include <limits.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas.h>

#include "num/vecops.h"
#include "num/gpuops.h"
#include "num/gpukrnls.h"
#include "num/mem.h"
#include "num/multind.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "gpuops.h"

#define MiBYTE (1024*1024)

extern unsigned int reserved_gpus;
unsigned int reserved_gpus = 0U;

int n_reserved_gpus = 0;
int gpu_map[MAX_CUDA_DEVICES] = { [0 ... MAX_CUDA_DEVICES - 1] = -1 };


static void cuda_error(int line, cudaError_t code)
{
	const char *err_str = cudaGetErrorString(code);
	error("cuda error: %d %s \n", line, err_str);
}


#define CUDA_ERROR(x)	({ cudaError_t errval = (x); if (cudaSuccess != errval) cuda_error(__LINE__, errval); })

// Print free and used memory on GPU.
void print_cuda_meminfo(void)
{
	size_t byte_tot;
	size_t byte_free;
	cudaError_t cuda_status = cudaMemGetInfo(&byte_free, &byte_tot);

	if (cuda_status != cudaSuccess)
		error("ERROR: cudaMemGetInfo failed. %s\n", cudaGetErrorString(cuda_status));


	double dbyte_tot = (double)byte_tot;
	double dbyte_free = (double)byte_free;
	double dbyte_used = dbyte_tot - dbyte_free;

	debug_printf(DP_INFO , "GPU memory usage: used = %.4f MiB, free = %.4f MiB, total = %.4f MiB\n", dbyte_used/MiBYTE, dbyte_free/MiBYTE, dbyte_tot/MiBYTE);
}

int num_cuda_devices(void)
{
	int count;
	CUDA_ERROR(cudaGetDeviceCount(&count));
	return count;
}

//static __thread int last_init = -1; // TODO: this needs work so that the memcache works!

int cuda_get_device(void)
{
	int device;
	CUDA_ERROR(cudaGetDevice(&device));
	return device;
}


static int num_cuda_reserved_devices(void)
{
	return bitcount(reserved_gpus);
}


void cuda_p2p_table(int n, bool table[n][n])
{
	assert(n == num_cuda_devices());

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {


			int r;
			CUDA_ERROR(cudaDeviceCanAccessPeer(&r, i, j));

			table[i][j] = (1 == r);
		}
	}
}

void cuda_p2p(int a, int b)
{
	int dev;
	CUDA_ERROR(cudaGetDevice(&dev));
	CUDA_ERROR(cudaSetDevice(a));
	CUDA_ERROR(cudaDeviceEnablePeerAccess(b, 0));
	CUDA_ERROR(cudaSetDevice(dev));
}



void cuda_init(void)
{

	int num_devices = num_cuda_devices();
	for (int device = 0; device < num_devices; ++device)
		if (cuda_try_init(device))
			return;

	error("Could not allocate any GPU device\n");
}



bool cuda_try_init(int device)
{
	cudaError_t errval = cudaSetDevice(device);
	if (cudaSuccess == errval) {

		errval = cudaDeviceSynchronize();

		if (cudaSuccess == errval) {

			// nly add to gpu_map if not already present.
			// This allows multiple calls to initialize cuda
			// to succeed without problems.
			if (!MD_IS_SET(reserved_gpus, device)) {

				gpu_map[n_reserved_gpus++] = device;
				reserved_gpus = MD_SET(reserved_gpus, device);
			}
			return true;

		} else {

			// clear last error
			cudaGetLastError();
		}
	}

	return false;
}


static void remove_from_gpu_map(int device)
{
	int device_index = -1;
	for (int i = 0; i < n_reserved_gpus; ++i) {

		if (device == gpu_map[i]) {

			device_index = i;
			break;
		}
	}

	for (int i = device_index; i < MIN(n_reserved_gpus, MAX_CUDA_DEVICES); ++i)
		gpu_map[i] = gpu_map[i + 1];

	gpu_map[n_reserved_gpus - 1] = -1;

}

static void cuda_deinit(int device)
{
	cuda_set_device(device);
	CUDA_ERROR(cudaDeviceReset());
	remove_from_gpu_map(device);
	n_reserved_gpus--;
	reserved_gpus = MD_CLEAR(reserved_gpus, device);
}


void cuda_init_multigpu(unsigned int requested_gpus)
{

	int num_devices = num_cuda_devices();
	for (int device = 0; device < num_devices; ++device) {

		if (MD_IS_SET(requested_gpus, device))
			cuda_try_init(device);
	}

	if (0UL == reserved_gpus )
		error("No GPUs could be allocated!\n");
	else if (reserved_gpus != requested_gpus)
		debug_printf(DP_WARN, "Not all requested gpus could be allocated, continuing with fewer\n");
}

int cuda_init_memopt(void)
{
	int num_devices = num_cuda_devices();
	int device;
	int max_device = -1;

	if (num_devices > 1) {

		size_t mem_max = 0;
		size_t mem_free;
		size_t mem_total;

		for (device = 0; device < num_devices; device++) {

			if (!cuda_try_init(device))
				continue;

			cudaError_t  errval = cudaMemGetInfo(&mem_free, &mem_total);
			if (cudaSuccess != errval)
				continue;


			if (mem_max < mem_free) {

				mem_max = mem_free;
				max_device = device;
			}
		}

		if (-1 == max_device)
			error("Could not allocate any GPU device\n");

		for (device = 0; device < num_devices; device++) {

			if (MD_IS_SET(reserved_gpus, device) && (device != max_device))
				cuda_deinit(device);
		}

		cuda_set_device(max_device);

	} else {

		cuda_try_init(0);
	}

	return max_device;
}

void cuda_set_device(int device)
{
	if (!MD_IS_SET(reserved_gpus, device))
		error("Trying to use non-reserved GPU! Reserve first by using cuda_try_init(device)\n");

	CUDA_ERROR(cudaSetDevice(device));
}



bool cuda_memcache = true;

void cuda_memcache_off(void)
{
	cuda_memcache = false;
}

void cuda_clear(long size, void* dst)
{
//	printf("CLEAR %x %ld\n", dst, size);
	assert(cuda_ondevice_num(dst, cuda_get_device()));
	CUDA_ERROR(cudaMemset(dst, 0, size));
}

static void cuda_float_clear(long size, float* dst)
{
	cuda_clear(size * sizeof(float), (void*)dst);
}

void cuda_memcpy(long size, void* dst, const void* src)
{
//	printf("COPY %x %x %ld\n", dst, src, size);
	if (cuda_ondevice(dst))
		assert(cuda_ondevice_num(dst, cuda_get_device()));
	if (cuda_ondevice(src))
		assert(cuda_ondevice_num(src, cuda_get_device()));
	CUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}


void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src)
{
	if (cuda_ondevice(dst))
		assert(cuda_ondevice_num(dst, cuda_get_device()));
	if (cuda_ondevice(src))
		assert(cuda_ondevice_num(src, cuda_get_device()));
	CUDA_ERROR(cudaMemcpy2D(dst, ostr, src, istr, dims[0], dims[1], cudaMemcpyDefault));
}

static void cuda_float_copy(long size, float* dst, const float* src)
{
	cuda_memcpy(size * sizeof(float), (void*)dst, (const void*)src);
}


static void cuda_free_wrapper(const void* ptr)
{
	CUDA_ERROR(cudaFree((void*)ptr));
}

void cuda_memcache_clear(void)
{
	if (!cuda_memcache)
		return;

	for (int d = 0; d < n_reserved_gpus; d++) {

		cuda_set_device(gpu_map[d]);
		memcache_clear(gpu_map[d], cuda_free_wrapper);
	}
}

void cuda_exit(void)
{
	cuda_memcache_clear();
	for (int d = 0; d < n_reserved_gpus; d++)
		cuda_deinit(gpu_map[d]);

}

#if 0
// We still don use this because it is slow. Why? Nivida, why?

static bool cuda_cuda_ondevice(const void* ptr)
{
	if (NULL == ptr)
		return false;

	struct cudaPointerAttributes attr;
	if (cudaSuccess != (cudaPointerGetAttributes(&attr, ptr)))
	{
	/* The secret trick to make this work for arbitrary pointers
	   is to clear the error using cudaGetLastError. See end of:
	   http://www.alexstjohn.com/WP/2014/04/28/cuda-6-0-first-look/
	 */
		cudaGetLastError();
		return false;
	}

	return (cudaMemoryTypeDevice == attr.memoryType);
}
#endif

bool cuda_ondevice(const void* ptr)
{
	return mem_ondevice(ptr);
}


bool cuda_ondevice_num(const void* ptr, const int device)
{
	return mem_ondevice_num(ptr, device);
}

bool cuda_accessible(const void* ptr)
{
#if 1
	return mem_device_accessible(ptr);
#else
	struct cudaPointerAttributes attr;
	//CUDA_ERROR(cudaPointerGetAttributes(&attr, ptr));
	if (cudaSuccess != (cudaPointerGetAttributes(&attr, ptr)))
		return false;

	return true;
#endif
}



void cuda_free(void* ptr)
{
	assert(cuda_ondevice_num(ptr, cuda_get_device()));
	mem_device_free(ptr, cuda_free_wrapper);
}


bool cuda_global_memory = false;

void cuda_use_global_memory(void)
{
	cuda_global_memory = true;
}

static void* cuda_malloc_wrapper(size_t size)
{
	void* ptr;

	if (cuda_global_memory) {

		CUDA_ERROR(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));

	} else {

		CUDA_ERROR(cudaMalloc(&ptr, size));
	}

	return ptr;
}

void* cuda_malloc(long size)
{
	return mem_device_malloc(cuda_get_device(), size, cuda_malloc_wrapper);
}



#if 0
void* cuda_hostalloc(long N)
{
	void* ptr;
	if (cudaSuccess != cudaHostAlloc(&ptr, N, cudaHostAllocDefault))
	     error("abort");

	insert(ptr, N, false);
	return ptr;
}

void cuda_hostfree(void* ptr)
{
	struct cuda_mem_s* nptr = search(ptr, true);
	assert(nptr->ptr == ptr);
	assert(!nptr->device);
	xfree(nptr);

	cudaFreeHost(ptr);
}
#endif

static float* cuda_float_malloc(long size)
{
	return (float*)cuda_malloc(size * sizeof(float));
}

static void cuda_float_free(float* x)
{
	cuda_free((void*)x);
}

static double cuda_sdot(long size, const float* src1, const float* src2)
{
	assert(size <= INT_MAX / 2);
	assert(cuda_ondevice_num(src1, cuda_get_device()));
	assert(cuda_ondevice_num(src2, cuda_get_device()));
//	printf("SDOT %x %x %ld\n", src1, src2, size);
	return cublasSdot(size, src1, 1, src2, 1);
}

static double cuda_norm(long size, const float* src1)
{
#if 1
	// cublasSnrm2 produces NaN in some situations
	// e.g. nlinv -g -i8 utests/data/und2x2 o
	// git rev: ab28a9a953a80d243511640b23501f964a585349
//	printf("cublas: %f\n", cublasSnrm2(size, src1, 1));
//	printf("GPU norm (sdot: %f)\n", sqrt(cuda_sdot(size, src1, src1)));
	assert(cuda_ondevice_num(src1, cuda_get_device()));
	return sqrt(cuda_sdot(size, src1, src1));
#else
	return cublasSnrm2(size, src1, 1);
#endif
}


static double cuda_asum(long size, const float* src)
{
	assert(cuda_ondevice_num(src, cuda_get_device()));
	assert(size <= INT_MAX / 2);
	return cublasSasum(size, src, 1);
}

static void cuda_saxpy(long size, float* y, float alpha, const float* src)
{
	assert(cuda_ondevice_num(src, cuda_get_device()));
	assert(cuda_ondevice_num(y, cuda_get_device()));
//	printf("SAXPY %x %x %ld\n", y, src, size);
	assert(size <= INT_MAX / 2);
	cublasSaxpy(size, alpha, src, 1, y, 1);
}

static void cuda_swap(long size, float* a, float* b)
{
	assert(cuda_ondevice_num(a, cuda_get_device()));
	assert(cuda_ondevice_num(b, cuda_get_device()));
	assert(size <= INT_MAX / 2);
	cublasSswap(size, a, 1, b, 1);
}

const struct vec_ops gpu_ops = {

	.float2double = cuda_float2double,
	.double2float = cuda_double2float,
	.dot = cuda_sdot,
	.asum = cuda_asum,
	.zsum = cuda_zsum,
	.zl1norm = NULL,

	.add = cuda_add,
	.sub = cuda_sub,
	.mul = cuda_mul,
	.div = cuda_div,
	.fmac = cuda_fmac,
	.fmac2 = cuda_fmac2,

	.smul = cuda_smul,
	.sadd = cuda_sadd,

	.axpy = cuda_saxpy,

	.pow = cuda_pow,
	.sqrt = cuda_sqrt,

	.le = cuda_le,

	.zsmul = cuda_zsmul,
	.zsadd = cuda_zsadd,
	.zsmax = cuda_zsmax,

	.zmul = cuda_zmul,
	.zdiv = cuda_zdiv,
	.zfmac = cuda_zfmac,
	.zfmac2 = cuda_zfmac2,
	.zmulc = cuda_zmulc,
	.zfmacc = cuda_zfmacc,
	.zfmacc2 = cuda_zfmacc2,

	.zpow = cuda_zpow,
	.zphsr = cuda_zphsr,
	.zconj = cuda_zconj,
	.zexpj = cuda_zexpj,
	.zexp = cuda_zexp,
	.zlog = cuda_zlog,
	.zarg = cuda_zarg,
	.zabs = cuda_zabs,
	.zatanr = cuda_zatanr,
	.zacos = cuda_zacos,

	.exp = cuda_exp,
	.log = cuda_log,

	.zcmp = cuda_zcmp,
	.zdiv_reg = cuda_zdiv_reg,
	.zfftmod = cuda_zfftmod,

	.zmax = cuda_zmax,
	.zle = cuda_zle,

	.smax = cuda_smax,
	.max = cuda_max,
	.min = cuda_min,

	.zsoftthresh = cuda_zsoftthresh,
	.zsoftthresh_half = cuda_zsoftthresh_half,
	.softthresh = cuda_softthresh,
	.softthresh_half = cuda_softthresh_half,
	.zhardthresh = NULL,

	.zconvcorr_3D = cuda_zconvcorr_3D,
	.zconvcorr_3D_CF = cuda_zconvcorr_3D_CF,
	.zconvcorr_3D_CF_TK = cuda_zconvcorr_3D_CF_TK,
	.zconvcorr_3D_CF_TI = cuda_zconvcorr_3D_CF_TI,

	.pdf_gauss = cuda_pdf_gauss,

	.smul_ptr = cuda_smul_ptr,

	.real = cuda_real,
	.imag = cuda_imag,
	.zcmpl_real = cuda_zcmpl_real,
	.zcmpl_imag = cuda_zcmpl_imag,
	.zcmpl = cuda_zcmpl,
};


// defined in iter/vec.h
struct vec_iter_s {

	float* (*allocate)(long N);
	void (*del)(float* x);
	void (*clear)(long N, float* x);
	void (*copy)(long N, float* a, const float* x);
	void (*swap)(long N, float* a, float* x);

	double (*norm)(long N, const float* x);
	double (*dot)(long N, const float* x, const float* y);

	void (*sub)(long N, float* a, const float* x, const float* y);
	void (*add)(long N, float* a, const float* x, const float* y);

	void (*smul)(long N, float alpha, float* a, const float* x);
	void (*xpay)(long N, float alpha, float* a, const float* x);
	void (*axpy)(long N, float* a, float alpha, const float* x);
	void (*axpbz)(long N, float* out, const float a, const float* x, const float b, const float* z);

	void (*zmul)(long N, complex float* dst, const complex float* src1, const complex float* src2);
};

extern const struct vec_iter_s gpu_iter_ops;
const struct vec_iter_s gpu_iter_ops = {

	.allocate = cuda_float_malloc,
	.del = cuda_float_free,
	.clear = cuda_float_clear,
	.copy = cuda_float_copy,
	.dot = cuda_sdot,
	.norm = cuda_norm,
	.axpy = cuda_saxpy,
	.xpay = cuda_xpay,
	.axpbz = cuda_axpbz,
	.smul = cuda_smul,
	.add = cuda_add,
	.sub = cuda_sub,
	.swap = cuda_swap,
	.zmul = cuda_zmul,
};


#endif
