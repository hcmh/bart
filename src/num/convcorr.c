/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */


#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

#include "num/flpmath.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls.h"
#include "num/gpu_conv.h"
#ifdef USE_CUDNN
#include "num/cudnn_wrapper.h"
#endif
#endif
#include "num/multind.h"
#include "num/optimize.h"
#include "num/vecops.h"
#include "num/blas.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/vecops_strided.h"

#include "misc/nested.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "convcorr.h"

static bool use_simple_convcorr = true;

static void activate_simple_convcorr(void)
{
	use_simple_convcorr = true;
}

static void deactivate_simple_convcorr(void)
{
	use_simple_convcorr = false;
}

//#define CONVCORR_OPTIMIZE_CPU_ONLY
//#define CONVCORR_OPTIMIZE_GPU_ONLY

/**
 * Copy from num/flpmath.c
 * Optimized threeop wrapper. Use when inputs are constants
 *
 * @param D number of dimensions
 * @param dim dimensions
 * @param ostr output strides
 * @param optr output
 * @param istr1 input 1 strides
 * @param iptr1 input 1 (constant)
 * @param istr2 input 2 strides
 * @param iptr2 input 2 (constant)
 * @param size size of data structures, e.g. complex float
 * @param too three-op multiply function
 */
static void optimized_threeop_oii(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr1[D], const void* iptr1, const long istr2[D], const void* iptr2, size_t sizes[3], md_nary_opt_fun_t too)
{
	const long (*nstr[3])[D?D:1] = { (const long (*)[D?D:1])ostr, (const long (*)[D?D:1])istr1, (const long (*)[D?D:1])istr2 };
	void *nptr[3] = { optr, (void*)iptr1, (void*)iptr2 };

	unsigned int io = 1 + ((iptr1 == optr) ? 2 : 0) + ((iptr2 == optr) ? 4 : 0);

	optimized_nop(3, io, D, dim, nstr, nptr, sizes, too);
}



zconvcorr_fwd_algo_f* algos_fwd_cpu[] = { zconvcorr_fwd_inner_matmul_cf, zconvcorr_fwd_im2col_cf_cpu, };
zconvcorr_bwd_krn_algo_f* algos_bwd_krn_cpu[] = { zconvcorr_bwd_krn_inner_matmul_cf, zconvcorr_bwd_krn_im2col_cf_cpu, };
zconvcorr_bwd_in_algo_f* algos_bwd_in_cpu[] = {	zconvcorr_bwd_in_inner_matmul_cf, zconvcorr_bwd_in_im2col_cf_cpu, };

#ifdef USE_CUDA
zconvcorr_bwd_krn_algo_f* algos_bwd_krn_gpu[] = {
						#ifdef USE_CUDNN
							zconvcorr_bwd_krn_cudnn,
						#endif
							zconvcorr_bwd_krn_im2col_cf_gpu,
							};

zconvcorr_fwd_algo_f* algos_fwd_gpu[] = {
					#ifdef USE_CUDNN
						zconvcorr_fwd_cudnn,
					#endif
						zconvcorr_fwd_im2col_cf_gpu,
					};

zconvcorr_bwd_in_algo_f* algos_bwd_in_gpu[] = {
					#ifdef USE_CUDNN
						zconvcorr_bwd_in_cudnn,
					#endif
						zconvcorr_bwd_in_im2col_cf_gpu,
					};
#endif




//detect if strides describe convolution
static bool detect_convcorr(	int N,
				long nodims[N], long nidims[N], long nkdims[N],
				long nostrs[N], long nistrs[N], long nkstrs[N],
				long dilation[N], long strides[N],
				unsigned long* ptr_flag, bool* ptr_conv,
				const long dims[2 * N], const long ostrs[2 * N], const long istrs[2 * N], const long kstrs[2 * N],
				size_t size);


//functions detecting strides for a specific call and running the algorithms
static bool simple_zconvcorr_fwd(	unsigned int N, const long dims[N],
					const long ostrs[N], complex float* optr,
					const long istrs1[N], const complex float* iptr1,
					const long istrs2[N], const complex float* iptr2);
static bool simple_zconvcorr_bwd_in(	unsigned int N, const long dims[N],
					const long ostrs[N], complex float* optr,
					const long istrs1[N], const complex float* iptr1,
					const long istrs2[N], const complex float* iptr2);
static bool simple_zconvcorr_bwd_krn(	unsigned int N, const long dims[N],
					const long ostrs[N], complex float* optr,
					const long istrs1[N], const complex float* iptr1,
					const long istrs2[N], const complex float* iptr2);


static bool detect_convcorr(	int N,
				long nodims[N], long nidims[N], long nkdims[N],
				long nostrs[N], long nistrs[N], long nkstrs[N],
				long dilation[N], long strides[N],
				unsigned long* ptr_flag, bool* ptr_conv,
				const long dims[2 * N], const long ostrs[2 * N], const long istrs[2 * N], const long kstrs[2 * N],
				size_t size)
{
	long istrs_triv = size;

	*ptr_flag = 0;
	*ptr_conv = true;

	md_singleton_dims(N, dilation);
	md_singleton_dims(N, strides);

	for (int i = 0; i < N; i++) {

		if ((1 != dims[i]) && (1 != dims[N + i])) {

			*ptr_flag = MD_SET(*ptr_flag, i);

			nodims[i] = dims[0 + i];
			nkdims[i] = dims[N + i];

			if (0 != kstrs[i])
				return false;

			nkstrs[i] = kstrs[N + i];

			if (0 != ostrs[N + i])
				return false;

			nostrs[i] = ostrs[i];

			long test_strides[] = { istrs[i] / istrs_triv, 1, 2, 3, 4, 5, 6, 7, 8 };
			bool found = false;

			for (uint j = 0; !found && j < ARRAY_SIZE(test_strides); j++) {

				strides[i] = test_strides[j];

				if (1 > strides[i])
					continue;

				if (0 != istrs[i] % strides[i])
					continue;

				nistrs[i] = istrs[i] / strides[i];

				if ((0 == nistrs[i]) || (0 != istrs[N + i] % nistrs[i]))
					continue;

				dilation[i] = istrs[N + i] / nistrs[i];

				nidims[i] = strides[i] * (nodims[i] - 1) + 1 + dilation[i] * (nkdims[i] - 1);

				found = true;
			}

			istrs_triv *= nidims[i];

			*ptr_conv = *ptr_conv && (0 >= nkstrs[i]);

			if (!found)
				return false;

		} else {

			if (1 != dims[N +  i])
				return false;

			nostrs[i] = ostrs[i];
			nistrs[i] = istrs[i];
			nkstrs[i] = kstrs[i];

			nodims[i] = (0 == nostrs[i]) ? 1 : dims[i];
			nkdims[i] = (0 == nkstrs[i]) ? 1 : dims[i];
			nidims[i] = (0 == nistrs[i]) ? 1 : dims[i];

			dilation[i] = 1;
			strides[i] = 1;

			if (0 != nistrs[i])
				istrs_triv *= nidims[i];
		}
	}

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(*ptr_flag, i))
			if (*ptr_conv)
				nkstrs[i] = -nkstrs[i];

	if (0 == *ptr_flag)
		return false;

#if 1 // this is a cross check, that the detected dims/strides reproduce the input strides/dims
	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	calc_convcorr_geom_strs_dil(	N, *ptr_flag,
					tdims, tostrs, tkstrs, tistrs,
					nodims, nostrs, nkdims, nkstrs, nidims, nistrs,
					dilation, strides, *ptr_conv, false);

	assert(md_check_equal_dims(2 * N, tdims, dims, ~(0l)));
	assert(md_check_equal_dims(2 * N, tostrs, ostrs, md_nontriv_dims(2 * N, dims)));
	assert(md_check_equal_dims(2 * N, tistrs, istrs, md_nontriv_dims(2 * N, dims)));
	assert(md_check_equal_dims(2 * N, tkstrs, kstrs, md_nontriv_dims(2 * N, dims)));
#endif
	return true;
}


bool simple_zconvcorr(	unsigned int N, const long dims[N],
			const long ostrs[N], complex float* optr,
			const long istrs1[N], const complex float* iptr1,
			const long istrs2[N], const complex float* iptr2)
{
	if (!use_simple_convcorr)
		return false;

	if (simple_zconvcorr_fwd(N, dims, ostrs, optr, istrs1, iptr1, istrs2, iptr2))
		return true;

	if (simple_zconvcorr_bwd_in(N, dims, ostrs, optr, istrs1, iptr1, istrs2, iptr2))
		return true;

	if (simple_zconvcorr_bwd_krn(N, dims, ostrs, optr, istrs1, iptr1, istrs2, iptr2))
		return true;

	return false;
}


//The following three function detect a (transposed) convolution and run the specific algorithms
static bool simple_zconvcorr_fwd(	unsigned int N, const long dims[N],
					const long ostrs[N], complex float* optr,
					const long istrs1[N], const complex float* iptr1,
					const long istrs2[N], const complex float* iptr2)
{
	if (0 != N % 2)
		return false;

	N /= 2;

	size_t size = CFL_SIZE;

	unsigned long flags;
	bool conv;
	long nodims[N];
	long nidims[N];
	long nkdims[N];

	long nostrs[N];
	long nistrs[N];
	long nkstrs[N];

	long dilation[N];
	long strides[N];

	complex float* out = NULL;
	const complex float* in = NULL;
	const complex float* krn = NULL;

	bool result = false;

	if (detect_convcorr(	N,
				nodims, nidims, nkdims,
				nostrs, nistrs, nkstrs,
				dilation, strides,
				&flags, &conv,
				dims, ostrs, istrs1, istrs2,
				size)) {

		out = optr;
		in = iptr1;
		krn = iptr2;
		result = true;
	}

	if ((!result) && (detect_convcorr(	N,
						nodims, nidims, nkdims,
						nostrs, nistrs, nkstrs,
						dilation, strides,
						&flags, &conv,
						dims, ostrs, istrs2, istrs1,
						size))) {

		out = optr;
		in = iptr2;
		krn = iptr1;
		result = true;
	}

	if (!result)
		return false;

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						nodims, nostrs,
						nkdims, nkstrs,
						nidims, nistrs,
						dilation, strides, conv, false) / size;

#ifdef USE_CUDA
	if (cuda_ondevice(out))
		for(int i = 0; (unsigned long)i < sizeof(algos_fwd_gpu) / sizeof(algos_fwd_gpu[0]); i++)
			if (algos_fwd_gpu[i](	N,
						nodims, nostrs, out,
						nidims, nistrs, in,
						nkdims, nkstrs, krn,
						flags, dilation, strides, conv))
				return true;

	if (!cuda_ondevice(out))
#endif
		for(int i = 0; (unsigned long)i < sizeof(algos_fwd_cpu) / sizeof(algos_fwd_cpu[0]); i++)
			if (algos_fwd_cpu[i](	N,
						nodims, nostrs, out,
						nidims, nistrs, in,
						nkdims, nkstrs, krn,
						flags, dilation, strides, conv))
				return true;

	return false;
}


static bool simple_zconvcorr_bwd_in(	unsigned int N, const long dims[N],
					const long ostrs[N], complex float* optr,
					const long istrs1[N], const complex float* iptr1,
					const long istrs2[N], const complex float* iptr2)
{
	if (0 != N % 2)
		return false;

	N /= 2;

	size_t size = CFL_SIZE;

	unsigned long flags;
	bool conv;
	long nodims[N];
	long nidims[N];
	long nkdims[N];

	long nostrs[N];
	long nistrs[N];
	long nkstrs[N];

	long dilation[N];
	long strides[N];

	const complex float* out = NULL;
	complex float* in = NULL;
	const complex float* krn = NULL;

	bool result = false;

	if (detect_convcorr(	N,
				nodims, nidims, nkdims,
				nostrs, nistrs, nkstrs,
				dilation, strides,
				&flags, &conv,
				dims, istrs1, ostrs, istrs2,
				size)) {

		out = iptr1;
		in = optr;
		krn = iptr2;
		result = true;
	}

	if ((!result) && (detect_convcorr(	N,
						nodims, nidims, nkdims,
						nostrs, nistrs, nkstrs,
						dilation, strides,
						&flags, &conv,
						dims, istrs2, ostrs, istrs1,
						size))) {

		out = iptr2;
		in = optr;
		krn = iptr1;
		result = true;
	}

	if (!result)
		return false;

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						nodims, nostrs,
						nkdims, nkstrs,
						nidims, nistrs,
						dilation, strides, conv, false) / size;

#ifdef USE_CUDA
	if (cuda_ondevice(out))
		for(int i = 0; (unsigned long)i < sizeof(algos_bwd_in_gpu) / sizeof(algos_bwd_in_gpu[0]); i++)
			if (algos_bwd_in_gpu[i](	N,
							nodims, nostrs, out,
							nidims, nistrs, in,
							nkdims, nkstrs, krn,
							flags, dilation, strides, conv))
				return true;
#endif


#ifdef USE_CUDA
	if (!cuda_ondevice(out))
#else
	if (true)
#endif
	for(int i = 0; (unsigned long)i < sizeof(algos_bwd_in_cpu) / sizeof(algos_bwd_in_cpu[0]); i++)
		if (algos_bwd_in_cpu[i](	N,
						nodims, nostrs, out,
						nidims, nistrs, in,
						nkdims, nkstrs, krn,
						flags, dilation, strides, conv))
			return true;

	return false;
}


static bool simple_zconvcorr_bwd_krn(	unsigned int N, const long dims[N],
					const long ostrs[N], complex float* optr,
					const long istrs1[N], const complex float* iptr1,
					const long istrs2[N], const complex float* iptr2)
{
	if (0 != N % 2)
		return false;

	N /= 2;

	size_t size = CFL_SIZE;

	unsigned long flags;
	bool conv;
	long nodims[N];
	long nidims[N];
	long nkdims[N];

	long nostrs[N];
	long nistrs[N];
	long nkstrs[N];

	long dilation[N];
	long strides[N];

	const complex float* out = NULL;
	const complex float* in = NULL;
	complex float* krn = NULL;

	bool result = false;

		if (detect_convcorr(	N,
				nodims, nidims, nkdims,
				nostrs, nistrs, nkstrs,
				dilation, strides,
				&flags, &conv,
				dims, istrs1, istrs2, ostrs,
				size)) {

		out = iptr1;
		in = iptr2;
		krn = optr;
		result = true;
	}

	if ((!result) && (detect_convcorr(	N,
						nodims, nidims, nkdims,
						nostrs, nistrs, nkstrs,
						dilation, strides,
						&flags, &conv,
						dims, istrs2, istrs1, ostrs,
						size))) {

		out = iptr2;
		in = iptr1;
		krn = optr;
		result = true;
	}

	if (!result)
		return false;

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						nodims, nostrs,
						nkdims, nkstrs,
						nidims, nistrs,
						dilation, strides, conv, false) / size;

#ifdef USE_CUDA
	if (cuda_ondevice(out))
		for(int i = 0; (unsigned long)i < sizeof(algos_bwd_krn_gpu) / sizeof(algos_bwd_krn_gpu[0]); i++)
			if (algos_bwd_krn_gpu[i](	N,
							nodims, nostrs, out,
							nidims, nistrs, in,
							nkdims, nkstrs, krn,
							flags, dilation, strides, conv))
				return true;
#endif

#ifdef USE_CUDA
	if (!cuda_ondevice(out))
#else
	if (true)
#endif
		for(int i = 0; (unsigned long)i < sizeof(algos_bwd_krn_cpu) / sizeof(algos_bwd_krn_cpu[0]); i++)
			if (algos_bwd_krn_cpu[i](	N,
							nodims, nostrs, out,
							nidims, nistrs, in,
							nkdims, nkstrs, krn,
							flags, dilation, strides, conv))
				return true;

	return false;
}


/**
 * Checks if params correspond to convcorr which is channel first and contiguous in memory
 */
static bool check_trivial_cf(	int N,
				long odims[N], long ostrs[N],
				long idims[N], long istrs[N],
				long kdims[N], long kstrs[N],
				unsigned long flags,
				size_t size)
{
	// Check conv dims
	for (int i = 2; i < N; i++)
		if ((!MD_IS_SET(flags, i)) && ((1 != odims[i]) || (1 != idims[i]) || (1 != kdims[i])))
			return false;

	// Check matmul dims
	if (MD_IS_SET(flags, 0) || MD_IS_SET(flags, 1) || (1 != idims[0]) || (1 != odims[1]))
		return false;

	// check contiguous memory
	if ((uint)N > md_calc_blockdim(N, odims, ostrs, size))
		return false;

	if ((uint)N > md_calc_blockdim(N, idims, istrs, size))
		return false;

	if ((uint)N > md_calc_blockdim(N, kdims, kstrs, size))
		return false;

	return true;
}

static bool check_trivial_strs_dil(int N, const long dilation[N], const long strides[N])
{
	if ((NULL != dilation) && (!md_check_equal_dims(N, dilation, MD_SINGLETON_DIMS(N), ~(0l))))
		return false;

	if ((NULL != strides) && (!md_check_equal_dims(N, strides, MD_SINGLETON_DIMS(N), ~(0l))))
		return false;

	return true;
}

#if 0
bool zconvcorr_fwd_direct_cf(	int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (5 > N)
		return false;
	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long* odimsP = odims; // clang
	long* idimsP = idims; // clang
	long* kdimsP = kdims; // clang

	long mdims[N - 5];
	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

	NESTED(void, nary_zconvcorr3D, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			data->ops->zconvcorr_3D_CF(	(complex float*)ptr[0] + i * osize,
							(complex float*)ptr[1] + i * isize,
							(complex float*)ptr[2] + i * ksize,
							odimsP, idimsP, kdimsP, conv);
	};


	//copy of optimized_threeop_oii in flpmath.c
	void *nptr[3] = { (void*)out, (void*)in, (void*)krn };
	unsigned int D = N - 5;
	const long (*nstr[3])[D?D:1] = { (const long (*)[D?D:1])(ostrs + 5), (const long (*)[D?D:1])(istrs + 5), (const long (*)[D?D:1])(kstrs + 5) };
	unsigned int io = 1 + ((in == out) ? 2 : 0) + ((krn == out) ? 4 : 0);
	optimized_nop(3, io, D, mdims, nstr, nptr, (size_t[3]){ size * osize, size * isize, size * ksize }, nary_zconvcorr3D);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

bool zconvcorr_bwd_in_direct_cf(	int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (5 > N)
		return false;
	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long* odimsP = odims; // clang
	long* idimsP = idims; // clang
	long* kdimsP = kdims; // clang

	long mdims[N - 5];
	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

	NESTED(void, nary_zconvcorr3D, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			data->ops->zconvcorr_3D_CF_TI(	(complex float*)ptr[0] + i * isize,
							(complex float*)ptr[1] + i * osize,
							(complex float*)ptr[2] + i * ksize,
							odimsP, idimsP, kdimsP, conv);
	};

	//copy of optimized_threeop_oii in flpmath.c
	void *nptr[3] = { (void*)in, (void*)out, (void*)krn };
	unsigned int D = N - 5;
	const long (*nstr[3])[D?D:1] = { (const long (*)[D?D:1])(istrs + 5), (const long (*)[D?D:1])(ostrs + 5), (const long (*)[D?D:1])(kstrs + 5) };
	unsigned int io = 1 + ((out == in) ? 2 : 0) + ((krn == in) ? 4 : 0);
	optimized_nop(3, io, D, mdims, nstr, nptr, (size_t[3]){ size * isize, size * osize, size * ksize }, nary_zconvcorr3D);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

bool zconvcorr_bwd_krn_direct_cf(	int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (5 > N)
		return false;
	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long* odimsP = odims; // clang
	long* idimsP = idims; // clang
	long* kdimsP = kdims; // clang

	long mdims[N - 5];
	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

	NESTED(void, nary_zconvcorr3D, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			data->ops->zconvcorr_3D_CF_TK(	(complex float*)ptr[0] + i * ksize,
							(complex float*)ptr[1] + i * isize,
							(complex float*)ptr[2] + i * osize,
							odimsP, idimsP, kdimsP, conv);
	};

	//copy of optimized_threeop_oii in flpmath.c
	void *nptr[3] = { (void*)krn, (void*)in, (void*)out };
	unsigned int D = N - 5;
	const long (*nstr[3])[D?D:1] = { (const long (*)[D?D:1])(kstrs + 5), (const long (*)[D?D:1])(istrs + 5), (const long (*)[D?D:1])(ostrs + 5) };
	unsigned int io = 1 + ((in == krn) ? 2 : 0) + ((out == krn) ? 4 : 0);

	optimized_nop(3, io, D, mdims, nstr, nptr, (size_t[3]){ size * ksize, size * isize, size * osize }, nary_zconvcorr3D);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}
#endif

bool zconvcorr_fwd_im2col_cf_cpu(int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	if (conv)
		return false;

	long dims_mat[8]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(5, dims_mat, kdims);
	md_copy_dims(3, dims_mat + 5, odims + 2);


	long kdims_mat[8]; // (nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )

	md_select_dims(8, MD_BIT(5) - 1, kdims_mat, dims_mat);


	long idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(8, ~1ul , idims_mat, dims_mat);
	md_copy_dims(N - 5, idims_mat + 8, idims + 5);


	long odims_mat[8]; // (nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	md_select_dims(8, MD_BIT(0) | MD_BIT(5) | MD_BIT(6) | MD_BIT(7), odims_mat, dims_mat);


	long istrs_mat[8];

	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, size));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, size) + 2);


	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long M1 = dims_mat[0];
	long K1 = dims_mat[1] * dims_mat[2] * dims_mat[3] * dims_mat[4];
	long N1 = dims_mat[5] * dims_mat[6] * dims_mat[7];

	long* idims_matP = idims_mat; // clang
	long* istrs_matP = istrs_mat;

	long mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);


	NESTED(void, nary_zconvcorr3D_I2C_CF, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_sameplace(8, idims_matP, size, in);

			md_copy2(8, idims_matP, MD_STRIDES(8, idims_matP, size), imat_tmp, istrs_matP, (const complex float*)ptr[1] + i * isize, size);

			blas_matrix_zfmac(	M1, N1, K1,
						(complex float*)ptr[0] + i * osize,
						(complex float*)ptr[2] + i * ksize, 'N',
						imat_tmp, 'N'
						);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, ostrs + 5, (void*)out, istrs + 5, (void*)in, kstrs + 5, (void*)krn,
				(size_t[3]){ size * osize, size * isize, size * ksize},
				nary_zconvcorr3D_I2C_CF);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}


bool zconvcorr_bwd_krn_im2col_cf_cpu(int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	if (conv)
		return false;


	long dims_mat[8]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(5, dims_mat, kdims);
	md_copy_dims(3, dims_mat + 5, odims + 2);


	long kdims_mat[8]; // (nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )

	md_select_dims(8, MD_BIT(5) - 1, kdims_mat, dims_mat);


	long idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(8, ~1ul , idims_mat, dims_mat);
	md_copy_dims(N - 5, idims_mat + 8, idims + 5);


	long odims_mat[8]; // (nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	md_select_dims(8, MD_BIT(0) | MD_BIT(5) | MD_BIT(6) | MD_BIT(7), odims_mat, dims_mat);


	long istrs_mat[8];

	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, size));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, size) + 2);

	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long M1 = dims_mat[0];
	long K1 = dims_mat[1] * dims_mat[2] * dims_mat[3] * dims_mat[4];
	long N1 = dims_mat[5] * dims_mat[6] * dims_mat[7];

	long* idims_matP = idims_mat; // clang
	long* istrs_matP = istrs_mat;

	long mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);


	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_sameplace(8, idims_matP, size, in);

			md_copy2(8, idims_matP, MD_STRIDES(8, idims_matP, size), imat_tmp, istrs_matP, (const complex float*)ptr[1] + i * isize, size);

			blas_matrix_zfmac(	M1, K1, N1,
						(complex float*)ptr[0] + i * ksize,
						(complex float*)ptr[2] + i * osize, 'N',
						imat_tmp, 'T'
						);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, kstrs + 5, (void*)krn, istrs + 5, (void*)in, ostrs + 5, (void*)out,
				(size_t[3]){ size * ksize, size * isize, size * osize},
				nary_zconvcorr_im2col);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}


bool zconvcorr_bwd_in_im2col_cf_cpu(int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	if (conv)
		return false;


	long dims_mat[8]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(5, dims_mat, kdims);
	md_copy_dims(3, dims_mat + 5, odims + 2);


	long kdims_mat[8]; // (nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )

	md_select_dims(8, MD_BIT(5) - 1, kdims_mat, dims_mat);


	long idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(8, ~1ul , idims_mat, dims_mat);
	md_copy_dims(N - 5, idims_mat + 8, idims + 5);


	long odims_mat[8]; // (nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	md_select_dims(8, MD_BIT(0) | MD_BIT(5) | MD_BIT(6) | MD_BIT(7), odims_mat, dims_mat);


	long istrs_mat[8];

	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, size));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, size) + 2);


	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long M1 = dims_mat[0];
	long K1 = dims_mat[1] * dims_mat[2] * dims_mat[3] * dims_mat[4];
	long N1 = dims_mat[5] * dims_mat[6] * dims_mat[7];

	long* idims_matP = idims_mat; // clang
	long* istrs_matP = istrs_mat;

	long mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);


	NESTED(void, nary_zconvcorr3D_I2C_CF, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_sameplace(8, idims_matP, size, in);
			md_clear(8, idims_matP, imat_tmp, size);

			blas_matrix_zfmac(	K1, N1, M1,
						imat_tmp,
						(complex float*)ptr[2] + i * ksize, 'T',
						(complex float*)ptr[1] + i * osize, 'N'
						);

			md_zadd2(8, idims_matP, istrs_matP, (complex float*)ptr[0] + i * isize, istrs_matP, (const complex float*)ptr[0] + i * isize, MD_STRIDES(8, idims_matP, size), imat_tmp);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, istrs + 5, (void*)in, ostrs + 5, (void*)out, kstrs + 5, (void*)krn,
				(size_t[3]){ size * osize, size * isize, size * ksize},
				nary_zconvcorr3D_I2C_CF);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}


#ifdef USE_CUDA
bool zconvcorr_fwd_im2col_cf_gpu(int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	if (!cuda_ondevice(out))
		return false;

	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (conv)
		return false;

	// mim2col dims (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)
	// kernel	(nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )
	// image	(1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )
	// output 	(nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long M1 = kdims[0];
	long K1 = kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long N1 = odims[2] * odims[3] * odims[4];

	long imat_size = K1 * N1;

	long mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

	//clang
	const long* odimsp = odims;
	const long* idimsp = idims;
	const long* kdimsp = kdims;
	const long* dilationp = dilation;
	const long* stridesp = strides;

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, size);
			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odimsp, idimsp, kdimsp, dilationp, stridesp);

			blas_matrix_zfmac(	M1, N1, K1,
						(complex float*)ptr[0] + i * osize,
						(complex float*)ptr[2] + i * ksize, 'N',
						imat_tmp, 'N'
						);
			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, ostrs + 5, (void*)out, istrs + 5, (void*)in, kstrs + 5, (void*)krn,
				(size_t[3]){ size * osize, size * isize, size * ksize},
				nary_zconvcorr_im2col);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}
#endif


#ifdef USE_CUDA
bool zconvcorr_bwd_krn_im2col_cf_gpu(int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	if (!cuda_ondevice(out))
		return false;

	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (conv)
		return false;

	// mim2col dims (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)
	// kernel	(nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )
	// image	(1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )
	// output 	(nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long M1 = kdims[0];
	long K1 = kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long N1 = odims[2] * odims[3] * odims[4];

	long imat_size = K1 * N1;

	long mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

		//clang
	const long* odimsp = odims;
	const long* idimsp = idims;
	const long* kdimsp = kdims;
	const long* dilationp = dilation;
	const long* stridesp = strides;

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, size);
			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odimsp, idimsp, kdimsp, dilationp, stridesp);

			blas_matrix_zfmac(	M1, K1, N1,
						(complex float*)ptr[0] + i * ksize,
						(complex float*)ptr[2] + i * osize, 'N',
						imat_tmp, 'T'
						);
			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, kstrs + 5, (void*)krn, istrs + 5, (void*)in, ostrs + 5, (void*)out,
				(size_t[3]){ size * ksize, size * isize, size * osize},
				nary_zconvcorr_im2col);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}
#endif


#ifdef USE_CUDA
bool zconvcorr_bwd_in_im2col_cf_gpu(int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	if (!cuda_ondevice(out))
		return false;

	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (conv)
		return false;

#ifndef NON_DETERMINISTIC
	if ((NULL != dilation) && 1 != md_calc_size(N, dilation))
		return false;

	if ((NULL != strides) && 1 != md_calc_size(N, strides))
		return false;
#endif

	// mim2col dims (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)
	// kernel	(nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )
	// image	(1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )
	// output 	(nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	long osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	long ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	long M1 = kdims[0];
	long K1 = kdims[1] * kdims[2] * kdims[3] * kdims[4];
	long N1 = odims[2] * odims[3] * odims[4];

	long imat_size = K1 * N1;

	long mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

	//clang
	const long* odimsp = odims;
	const long* idimsp = idims;
	const long* kdimsp = kdims;
	const long* dilationp = dilation;
	const long* stridesp = strides;

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, size);
			md_clear(1, &imat_size, imat_tmp, size);

			blas_matrix_zfmac(	K1, N1, M1,
						imat_tmp,
						(complex float*)ptr[2] + i * ksize, 'T',
						(complex float*)ptr[1] + i * osize, 'N'
						);


			cuda_im2col_transp((complex float*)ptr[0] + i * isize, imat_tmp , odimsp, idimsp, kdimsp, dilationp, stridesp);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, istrs + 5, (void*)in, ostrs + 5, (void*)out, kstrs + 5, (void*)krn,
				(size_t[3]){ size * isize, size * osize, size * ksize},
				nary_zconvcorr_im2col);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}
#endif

static void test_zconvcorr_fwd_ref(	int N,
					long odims[N], long ostrs[N], complex float* optr,
					long idims[N], long istrs[N], const complex float* iptr,
					long kdims[N], long kstrs[N], const complex float* kptr,
					unsigned long flags, const long dilation[N], const long strides[N], bool conv
				)
{
	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

	deactivate_strided_vecops();
	md_zfmac2(2 * N, tdims, tostrs, optr, tistrs, iptr, tkstrs, kptr + shift);
	activate_strided_vecops();
}

static void test_zconvcorr_bwd_krn_ref(	int N,
					long odims[N], long ostrs[N], const complex float* optr,
					long idims[N], long istrs[N], const complex float* iptr,
					long kdims[N], long kstrs[N], complex float* kptr,
					unsigned long flags, const long dilation[N], const long strides[N], bool conv
				)
{
	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

	deactivate_strided_vecops();
	md_zfmac2(2 * N, tdims, tkstrs, kptr + shift, tostrs, optr, tistrs, iptr);
	activate_strided_vecops();
}

static void test_zconvcorr_bwd_in_ref(	int N,
					long odims[N], long ostrs[N], const complex float* optr,
					long idims[N], long istrs[N], complex float* iptr,
					long kdims[N], long kstrs[N], const complex float* kptr,
					unsigned long flags, const long dilation[N], const long strides[N], bool conv
				)
{
	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

	deactivate_strided_vecops();
	md_zfmac2(2 * N, tdims, tistrs, iptr, tostrs, optr, tkstrs, kptr + shift);
	activate_strided_vecops();
}


bool test_zconvcorr_fwd(	int N,
				long odims[N], long ostrs[N],
				long idims[N], long istrs[N],
				long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], bool conv,
				float max_nrmse, bool gpu, long min_no_algos)
{
	bool result = true;

#ifdef USE_CUDA
	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
#else
	assert(!gpu);
	void* ref_ptr = md_alloc(1, MD_DIMS(1), CFL_SIZE);
#endif

	complex float* optr_ref = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* optr_tst = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* optr_ini = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);

	complex float* iptr = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);
	complex float* kptr = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);

	md_gaussian_rand(N, odims, optr_ini);
	md_gaussian_rand(N, idims, iptr);
	md_gaussian_rand(N, kdims, kptr);

	md_copy(N, odims, optr_ref, optr_ini, CFL_SIZE);

	test_zconvcorr_fwd_ref(N, odims, ostrs, optr_ref, idims, istrs, iptr, kdims, kstrs, kptr, flags, dilation, strides, conv);

	long counter = 0;

#ifdef USE_CUDA
	int nr_algos = gpu ? ARRAY_SIZE(algos_fwd_gpu) : ARRAY_SIZE(algos_fwd_cpu);
#else
	int nr_algos = ARRAY_SIZE(algos_fwd_cpu);
#endif

	for(int i = 0; i < nr_algos; i++) {

#ifdef USE_CUDA
		zconvcorr_fwd_algo_f* algo = gpu ? algos_fwd_gpu[i] : algos_fwd_cpu[i];
#else
		zconvcorr_fwd_algo_f* algo = algos_fwd_cpu[i];
#endif

		md_copy(N, odims, optr_tst, optr_ini, CFL_SIZE);

		if (algo(N, odims, ostrs, optr_tst, idims, istrs, iptr, kdims, kstrs, kptr, flags, dilation, strides, conv)) {

			float err = md_znrmse(N, odims, optr_ref, optr_tst);
			debug_printf((err >= max_nrmse) ? DP_WARN : DP_DEBUG1, "error zconvcorr_fwd algo %d: %.8f\n", i, err);

			counter += 1;
			result = result && (max_nrmse > err);
		}
	}

	md_free(optr_tst);
	md_free(optr_ini);
	md_free(optr_ref);


	md_free(iptr);
	md_free(kptr);

	md_free(ref_ptr);

	return result && (counter >= min_no_algos);
}

bool test_zconvcorr_bwd_in(	int N,
				long odims[N], long ostrs[N],
				long idims[N], long istrs[N],
				long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], bool conv,
				float max_nrmse, bool gpu, long min_no_algos)
{
	bool result = true;

#ifdef USE_CUDA
	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
#else
	assert(!gpu);
	void* ref_ptr = md_alloc(1, MD_DIMS(1), CFL_SIZE);
#endif

	complex float* iptr_ref = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);
	complex float* iptr_tst = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);
	complex float* iptr_ini = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);

	complex float* optr = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* kptr = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);

	md_gaussian_rand(N, idims, iptr_ini);
	md_gaussian_rand(N, odims, optr);
	md_gaussian_rand(N, kdims, kptr);

	md_copy(N, idims, iptr_ref, iptr_ini, CFL_SIZE);

	test_zconvcorr_bwd_in_ref(N, odims, ostrs, optr, idims, istrs, iptr_ref, kdims, kstrs, kptr, flags, dilation, strides, conv);

	long counter = 0;

#ifdef USE_CUDA
	int nr_algos = gpu ? ARRAY_SIZE(algos_bwd_in_gpu) : ARRAY_SIZE(algos_bwd_in_cpu);
#else
	int nr_algos = ARRAY_SIZE(algos_bwd_in_cpu);
#endif

	for(int i = 0; i < nr_algos; i++) {

#ifdef USE_CUDA
		zconvcorr_bwd_in_algo_f* algo = gpu ? algos_bwd_in_gpu[i] : algos_bwd_in_cpu[i];
#else
		zconvcorr_bwd_in_algo_f* algo = algos_bwd_in_cpu[i];
#endif

		md_copy(N, idims, iptr_tst, iptr_ini, CFL_SIZE);

		if (algo(N, odims, ostrs, optr, idims, istrs, iptr_tst, kdims, kstrs, kptr, flags, dilation, strides, conv)) {

			float err = md_znrmse(N, idims, iptr_ref, iptr_tst);
			debug_printf((err >= max_nrmse) ? DP_WARN : DP_DEBUG1, "error zconvcorr_bwd_in algo %d: %.8f\n", i, err);

			counter += 1;
			result = result && (max_nrmse > err);
		}
	}

	md_free(iptr_tst);
	md_free(iptr_ini);
	md_free(iptr_ref);

	md_free(optr);
	md_free(kptr);

	md_free(ref_ptr);

	return result && (counter >= min_no_algos);
}

bool test_zconvcorr_bwd_krn(	int N,
				long odims[N], long ostrs[N],
				long idims[N], long istrs[N],
				long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], bool conv,
				float max_nrmse, bool gpu, long min_no_algos)
{
	bool result = true;

#ifdef USE_CUDA
	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
#else
	assert(!gpu);
	void* ref_ptr = md_alloc(1, MD_DIMS(1), CFL_SIZE);
#endif

	complex float* kptr_ref = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);
	complex float* kptr_tst = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);
	complex float* kptr_ini = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);

	complex float* optr = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* iptr = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);

	md_gaussian_rand(N, kdims, kptr_ini);
	md_gaussian_rand(N, odims, optr);
	md_gaussian_rand(N, idims, iptr);

	md_copy(N, kdims, kptr_ref, kptr_ini, CFL_SIZE);

	test_zconvcorr_bwd_krn_ref(N, odims, ostrs, optr, idims, istrs, iptr, kdims, kstrs, kptr_ref, flags, dilation, strides, conv);

	long counter = 0;

#ifdef USE_CUDA
	int nr_algos = gpu ? ARRAY_SIZE(algos_bwd_krn_gpu) : ARRAY_SIZE(algos_bwd_krn_cpu);
#else
	int nr_algos = ARRAY_SIZE(algos_bwd_krn_cpu);
#endif

	for(int i = 0; i < nr_algos; i++) {

#ifdef USE_CUDA
		zconvcorr_bwd_krn_algo_f* algo = gpu ? algos_bwd_krn_gpu[i] : algos_bwd_krn_cpu[i];
#else
		zconvcorr_bwd_krn_algo_f* algo = algos_bwd_krn_cpu[i];
#endif

		md_copy(N, kdims, kptr_tst, kptr_ini, CFL_SIZE);

		if (algo(N, odims, ostrs, optr, idims, istrs, iptr, kdims, kstrs, kptr_tst, flags, dilation, strides, conv)) {

			float err = md_znrmse(N, kdims, kptr_ref, kptr_tst);
			debug_printf((err >= max_nrmse) ? DP_WARN : DP_DEBUG1, "error zconvcorr_bwd_krn algo %d: %.8f\n", i, err);

			counter += 1;
			result = result && (max_nrmse > err);
		}
	}

	md_free(kptr_tst);
	md_free(kptr_ini);
	md_free(kptr_ref);

	md_free(optr);
	md_free(iptr);

	md_free(ref_ptr);

	return result && (counter >= min_no_algos);
}

// don't use any convcorr optimization but rely on md_zfmac2 -> inner kernels such as cgemm are called
// this is usually fast ...
bool zconvcorr_fwd_inner_matmul_cf(	int N,
					long odims[N], long ostrs[N], complex float* out,
					long idims[N], long istrs[N], const complex float* in,
					long kdims[N], long kstrs[N], const complex float* krn,
					unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (!check_trivial_cf(MIN(5, N), odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	krn += calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						odims, ostrs, kdims, kstrs, idims, istrs,
						dilation, strides, conv, false) / size;

	deactivate_simple_convcorr();
	md_zfmac2(2 * N, tdims, tostrs, out, tistrs, in, tkstrs, krn);
	activate_simple_convcorr();
	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

bool zconvcorr_bwd_in_inner_matmul_cf	(int N,
					long odims[N], long ostrs[N], const complex float* out,
					long idims[N], long istrs[N], complex float* in,
					long kdims[N], long kstrs[N], const complex float* krn,
					unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (!check_trivial_cf(MIN(5, N), odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	krn += calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						odims, ostrs, kdims, kstrs, idims, istrs,
						dilation, strides, conv, false) / size;

	deactivate_simple_convcorr();
	md_zfmac2(2 * N, tdims, tistrs, in, tostrs, out, tkstrs, krn);
	activate_simple_convcorr();
	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

bool zconvcorr_bwd_krn_inner_matmul_cf(	int N,
					long odims[N], long ostrs[N], const complex float* out,
					long idims[N], long istrs[N], const complex float* in,
					long kdims[N], long kstrs[N], complex float* krn,
					unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (!check_trivial_cf(MIN(5, N), odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	krn += calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						odims, ostrs, kdims, kstrs, idims, istrs,
						dilation, strides, conv, false) / size;

	deactivate_simple_convcorr();
	md_zfmac2(2 * N, tdims, tkstrs, krn, tostrs, out, tistrs, in);
	activate_simple_convcorr();
	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

#if 0 // code which might be used for tiling and im2col

#if 0 // for multind.c
/**
 * Computes the next position. Returns true until last index.
 */
bool md_next_stepsize(unsigned int D, const long dims[D], unsigned long flags, long pos[D], long stepsize[D])
{
	if (0 == D--)
		return false;

	if (md_next_stepsize(D, dims, flags, pos, stepsize))
		return true;

	if (MD_IS_SET(flags, D)) {

		assert((0 <= pos[D]) && (pos[D] < dims[D]));

		pos[D] += stepsize[D];

		if (pos[D] < dims[D])
			return true;

		pos[D] = 0;
	}

	return false;
}

#endif


/* *
 * @args odims = [OC,  1 | OX, OY, ... | N_batch]
 * @args kdims = [OC, IC | KX, KY, ... |       1]
 * @args idims = [ 1, IC | IX, IY, ... | N_batch]
 *
 * */
static void convcorr_im2col_in_cf(	long N, void* buffer,
					const long odims[N], const long kdims[N], const long idims[N],
					const long istrs[N], const long dilation[N], const long strides[N],
					void* in, bool transp)
{
	long idims_mat[2 * N - 4]; // (IC, KX, KY, ... | OX, OY, ..., N_batch)
	idims_mat[0] = idims[1];
	md_copy_dims(N - 3, idims_mat + 1, kdims + 2);
	md_copy_dims(N - 3, idims_mat + N - 2, odims + 2);
	idims_mat[2 * N - 5] = idims[N - 1];

	long istrs_mat[2 * N - 4];
	istrs_mat[0] = istrs[1];
	istrs_mat[2 * N - 5] = istrs[N - 1];

	for (int i = 0; i < N - 3; i++) {

		istrs_mat[i + 1] = istrs[i + 2] * (NULL == dilation ? 1 : dilation[i + 2]);
		istrs_mat[N - 2 + i] = istrs[i + 2] * (NULL == strides ? 1 : strides[i + 2]);
	}

	long istrs_mat_triv[2 * N - 4];
	md_calc_strides(2 * N - 4, istrs_mat_triv, idims_mat, CFL_SIZE);

	if (transp) {

		complex float* tmp = md_alloc(N, idims, CFL_SIZE);
		md_clear(N, idims, tmp, CFL_SIZE);
		md_zadd2(2 * N - 4, idims_mat, istrs_mat_triv, tmp, istrs_mat_triv, tmp, istrs_mat, buffer);
		md_zadd2(N, idims, istrs, in, istrs, in, MD_STRIDES(N, idims, CFL_SIZE), tmp);
		md_free(tmp);
	} else {

		md_copy2(2 * N - 4, idims_mat, istrs_mat_triv, buffer, istrs_mat, in, CFL_SIZE);
	}
}

/* *
 * @args odims = [OC,  1 | OX, OY, ... | N_batch]
 * @args kdims = [OC, IC | KX, KY, ... |       1]
 * @args idims = [ 1, IC | IX, IY, ... | N_batch]
 *
 * */
static void convcorr_frw_im2col_cf(	int N,
					const long odims[N], const long ostrs[N], complex float* out,
					const long idims[N], const long istrs[N], const complex float* in,
					const long kdims[N], const long kstrs[N], const complex float* krn,
					const long dilation[N], const long strides[N], bool conv)
{
	long M1 = odims[0];
	long K1 = md_calc_size(N - 1, kdims + 1);
	long N1 = md_calc_size(N - 1, odims + 1);

	assert(!conv);

	bool out_buffer_needed = !md_check_equal_dims(N, ostrs, MD_STRIDES(N, odims, CFL_SIZE), md_nontriv_dims(N, odims));
	complex float* out_buffer = out;

	if (out_buffer_needed) {

		out_buffer = md_alloc(N, odims, CFL_SIZE);
		md_copy2(N, odims, MD_STRIDES(N, odims, CFL_SIZE), out_buffer, ostrs, out, CFL_SIZE);
	}

	bool krn_buffer_needed = !md_check_equal_dims(N, kstrs, MD_STRIDES(N, kdims, CFL_SIZE), md_nontriv_dims(N, kdims));
	const complex float* krn_buffer = krn;

	if (krn_buffer_needed) {

		complex float* tmp = md_alloc(N, kdims, CFL_SIZE);
		md_copy2(N, kdims, MD_STRIDES(N, kdims, CFL_SIZE), tmp, kstrs, krn, CFL_SIZE);
		krn_buffer = tmp;
	}

	complex float* in_buffer = md_alloc(2, (long[2]){K1, N1}, CFL_SIZE);
	convcorr_im2col_in_cf(N, in_buffer, odims, kdims, idims, istrs, dilation, strides, in, false);

	blas_matrix_zfmac(	M1, N1, K1,
				out_buffer,
				krn, 'N',
				in_buffer, 'N'
				);

	md_free(in_buffer);

	if (out_buffer_needed) {

		md_copy2(N, odims, ostrs, out, MD_STRIDES(N, odims, CFL_SIZE), out_buffer, CFL_SIZE);
		md_free(out_buffer);
	}

	if (krn_buffer_needed)
		md_free(krn_buffer);
}


#if 1

static void conv_frw_im2col_tiling(	int N,
					const long odims[N], const long ostrs[N], complex float* out,
					const long idims[N], const long istrs[N], const complex float* in,
					const long kdims[N], const long kstrs[N], const complex float* krn,
					const long dilation[N], const long strides[N], bool conv)
{

	long max_tiles[N];
	for (int i = 0; i < N; i++)
		max_tiles[i] = 8;

	if (6 != N) {

		long pos[N];
		for (int i = 0; i < N; i++)
			pos[i] = 0;

		do {
			long odims_tmp[N];
			long idims_tmp[N];

			for (int i = 2; i < N; i++)
				odims_tmp[i] = MIN(max_tiles[i], odims[i] - pos[i]);

			for (int i = 2; i < N - 1; i++)
				idims_tmp[i] = ((NULL == strides) ? 1 : strides[i]) * (odims_tmp[i] - 1) + 1 + (kdims[i] - 1) * ((NULL == dilation) ? 1 : dilation[i]);

			odims_tmp[1] = odims[1];
			odims_tmp[0] = odims[0];

			idims_tmp[1] = idims[1];
			idims_tmp[0] = idims[0];

			idims_tmp[N - 1] = odims_tmp[N - 1];

			long ipos[N];
			ipos[0] = 0;
			ipos[1] = 0;
			for (int i = 2; i < N - 1; i++)
				ipos[i] = pos[i] * ((NULL == strides) ? 1 : strides[i]);
			ipos[N - 1] = pos[N - 1];

			convcorr_frw_im2col_cf(	N,
						odims_tmp, ostrs, &MD_ACCESS(N, ostrs, pos, out),
						idims_tmp, istrs, &MD_ACCESS(N, istrs, ipos, in),
						kdims, kstrs, krn,
						dilation, strides, conv
						);

		} while (md_next_stepsize(N, odims, ~(MD_BIT(0) | MD_BIT(1)), pos, max_tiles));

	} else {
		#pragma omp parallel for collapse(4)
		for(long b = 0; b < odims[5]; b += max_tiles[5])
		for(long oz = 0; oz < odims[4]; oz += max_tiles[4])
		for(long oy = 0; oy < odims[3]; oy += max_tiles[3])
		for(long ox = 0; ox < odims[2]; ox += max_tiles[2]){

			long pos[6] = {0, 0, ox, oy, oz, b};

			long odims_tmp[N];
			long idims_tmp[N];

			for (int i = 2; i < N; i++)
				odims_tmp[i] = MIN(max_tiles[i], odims[i] - pos[i]);

			for (int i = 2; i < N - 1; i++)
				idims_tmp[i] = ((NULL == strides) ? 1 : strides[i]) * (odims_tmp[i] - 1) + 1 + (kdims[i] - 1) * ((NULL == dilation) ? 1 : dilation[i]);

			odims_tmp[1] = odims[1];
			odims_tmp[0] = odims[0];

			idims_tmp[1] = idims[1];
			idims_tmp[0] = idims[0];

			idims_tmp[N - 1] = odims_tmp[N - 1];

			long ipos[N];
			ipos[0] = 0;
			ipos[1] = 0;
			for (int i = 2; i < N - 1; i++)
				ipos[i] = pos[i] * ((NULL == strides) ? 1 : strides[i]);
			ipos[N - 1] = pos[N - 1];

			convcorr_frw_im2col_cf(	N,
						odims_tmp, ostrs, &MD_ACCESS(N, ostrs, pos, out),
						idims_tmp, istrs, &MD_ACCESS(N, istrs, ipos, in),
						kdims, kstrs, krn,
						dilation, strides, conv
						);
		}
	}
}

#endif

bool zconvcorr_fwd_im2col_cf_cpu(int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;
	//if (!check_trivial_strs_dil(5, dilation, strides))
	//	return false;
	if (conv)
		return false;

	long odims_batch[6];
	long kdims_batch[6];
	long idims_batch[6];

	md_copy_dims(5, odims_batch, odims);
	md_copy_dims(5, kdims_batch, kdims);
	md_copy_dims(5, idims_batch, idims);

	odims_batch[5] = 1;
	kdims_batch[5] = 1;
	idims_batch[5] = 1;

	bool include_batch = (odims[5] == idims[5]) && (1 == kdims[5]);
	include_batch &= (ostrs[5] == md_calc_size(5, odims) * CFL_SIZE);
	include_batch &= (istrs[5] == md_calc_size(5, idims) * CFL_SIZE);

	if (include_batch) {

		odims_batch[5] = odims[5];
		idims_batch[5] = idims[5];
	}

	long osize = md_calc_size(6, odims_batch);
	long ksize = md_calc_size(6, kdims_batch);
	long isize = md_calc_size(6, idims_batch);

	long* ptr_odims = &(odims_batch[0]);
	long* ptr_idims = &(idims_batch[0]);
	long* ptr_kdims = &(kdims_batch[0]);

	int skip = include_batch ? 6 : 5;

	long mdims[N - skip];
	md_tenmul_dims(N - skip, mdims, odims + skip, idims + skip, kdims + skip);

	NESTED(void, nary_zconvcorr3D_I2C_CF, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			(true ? conv_frw_im2col_tiling : convcorr_frw_im2col_cf) (	6,
								ptr_odims, MD_STRIDES(6, ptr_odims, CFL_SIZE), (complex float*)ptr[0] + i * osize,
								ptr_idims, MD_STRIDES(6, ptr_idims, CFL_SIZE), (const complex float*)ptr[1] + i * isize,
								ptr_kdims, MD_STRIDES(6, ptr_kdims, CFL_SIZE), (const complex float*)ptr[2] + i * ksize,
								dilation, strides, conv);
	};

	optimized_threeop_oii(N - skip, mdims, ostrs + skip, (void*)out, istrs + skip, (void*)in, kstrs + skip, (void*)krn,
				(size_t[3]){ size * osize, size * isize, size * ksize},
				nary_zconvcorr3D_I2C_CF);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

#endif
