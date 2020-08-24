#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

#include "num/flpmath.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls.h"
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

#include "misc/nested.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "convcorr.h"

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

zconvcorr_fwd_algo_f* algos_fwd_cpu[] = {	zconvcorr_fwd_im2col_cf_cpu,
						zconvcorr_fwd_direct_cf};

zconvcorr_bwd_krn_algo_f* algos_bwd_krn_cpu[] = {	zconvcorr_bwd_krn_im2col_cf_cpu,
							zconvcorr_bwd_krn_direct_cf};

zconvcorr_bwd_in_algo_f* algos_bwd_in_cpu[] = {	zconvcorr_bwd_in_im2col_cf_cpu,
						zconvcorr_bwd_in_direct_cf};

#ifdef USE_CUDA
zconvcorr_bwd_krn_algo_f* algos_bwd_krn_gpu[] = {
						#ifdef USE_CUDNN
							zconvcorr_bwd_krn_cudnn_2d_cf,
						#endif
							zconvcorr_bwd_krn_im2col_cf_gpu,
							zconvcorr_bwd_krn_direct_cf
							};

zconvcorr_fwd_algo_f* algos_fwd_gpu[] = {
					#ifdef USE_CUDNN
						zconvcorr_fwd_cudnn_2d_cf,
						zconvcorr_fwd_cudnn_3d_cf,
					#endif
						zconvcorr_fwd_im2col_cf_gpu,
						zconvcorr_fwd_direct_cf};

zconvcorr_bwd_in_algo_f* algos_bwd_in_gpu[] = {
					#ifdef USE_CUDNN
						zconvcorr_bwd_in_cudnn_2d_cf,
					#endif
						zconvcorr_bwd_in_im2col_cf_gpu,
						zconvcorr_bwd_in_direct_cf};
#endif




//detect if strides describe convolution
static bool detect_convcorr(int N, long nodims[N], long nidims[N], long nkdims[N], unsigned long* ptr_flag, bool* ptr_conv,
			    const long dims[2 * N], const long ostrs[2 * N], const long istrs[2 * N], const long kstrs[2 * N], size_t size);

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



/**
 * Detect if strides and dims belong to convolution with specified flags
 **/
static bool detect_convcorr(int N, long nodims[N], long nidims[N], long nkdims[N], unsigned long* ptr_flag, bool* ptr_conv,
			    const long dims[2 * N], const long ostrs[2 * N], const long istrs[2 * N], const long kstrs[2 * N], size_t size)
{
	enum { nr_test_flags = 2 };
	long test_flags[nr_test_flags] = { 7, 28 };

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	bool result = true;

	*ptr_conv = false;
	for (int i_flag = 0; i_flag < nr_test_flags; i_flag++) {

		*ptr_flag = test_flags[i_flag];
		result = true;

		for (int i = 0; i < N; i++) {

			if (MD_IS_SET(*ptr_flag, i)){

				nodims[i] = dims[0 + i];
				nkdims[i] = dims[N + i];
				nidims[i] = nodims[i] + nkdims[i] - 1;

			} else {

				nodims[i] = (ostrs[i] == 0) ? 1 : dims[i];
				nidims[i] = (istrs[i] == 0) ? 1 : dims[i];
				nkdims[i] = (kstrs[i] == 0) ? 1 : dims[i];
			}
		}

		calc_convcorr_geom_strs_dil(N, *ptr_flag, tdims, tostrs, tkstrs, tistrs, nodims, MD_STRIDES(N, nodims, size), nkdims, MD_STRIDES(N, nkdims, size), nidims, MD_STRIDES(N, nidims, size), MD_SINGLETON_DIMS(N), MD_SINGLETON_DIMS(N), *ptr_conv, true);

		for (int i = 0; i < 2 * N; i++) {

			if (tdims[i] != dims[i])
				result = false;

			if (tostrs[i] != ostrs[i])
				result = false;

			if (tistrs[i] != istrs[i])
				result = false;

			if (tkstrs[i] != kstrs[i])
				result = false;
		}

		if (result)
			return result;
	}

	*ptr_conv = true;
	for (int i_flag = 0; i_flag < nr_test_flags; i_flag++) {

		*ptr_flag = test_flags[i_flag];
		result = true;

		for (int i = 0; i < N; i++) {

			if (MD_IS_SET(*ptr_flag, i)){

				nodims[i] = dims[0 + i];
				nkdims[i] = dims[N + i];
				nidims[i] = nodims[i] + nkdims[i] - 1;

			} else {

				nodims[i] = (ostrs[i] == 0) ? 1 : dims[i];
				nidims[i] = (istrs[i] == 0) ? 1 : dims[i];
				nkdims[i] = (kstrs[i] == 0) ? 1 : dims[i];
			}
		}

		calc_convcorr_geom_strs_dil(N, *ptr_flag, tdims, tostrs, tkstrs, tistrs, nodims, MD_STRIDES(N, nodims, size), nkdims, MD_STRIDES(N, nkdims, size), nidims, MD_STRIDES(N, nidims, size), MD_SINGLETON_DIMS(N), MD_SINGLETON_DIMS(N), *ptr_conv, true);

		for (int i = 0; i < 2 * N; i++) {

			if (tdims[i] != dims[i])
				result = false;

			if (tostrs[i] != ostrs[i])
				result = false;

			if (tistrs[i] != istrs[i])
				result = false;

			if (tkstrs[i] != kstrs[i])
				result = false;
		}

		if (result)
			return result;
	}

	return false;
}

bool simple_zconvcorr(	unsigned int N, const long dims[N],
			const long ostrs[N], complex float* optr,
			const long istrs1[N], const complex float* iptr1,
			const long istrs2[N], const complex float* iptr2)
{
#ifdef USE_CUDA
#ifdef CONVCORR_OPTIMIZE_CPU_ONLY
	if (cuda_ondevice(optr))
		return false;
#endif
#ifdef CONVCORR_OPTIMIZE_GPU_ONLY
	if (!cuda_ondevice(optr))
		return false;
#endif
#else
#ifdef CONVCORR_OPTIMIZE_GPU_ONLY
	return false;
#endif
#endif
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
	long odims[N];
	long idims[N];
	long kdims[N];

	complex float* out = NULL;
	const complex float* in = NULL;
	const complex float* krn = NULL;

	bool result = false;

	if (detect_convcorr(N, odims, idims, kdims, &flags, &conv, dims, ostrs, istrs1, istrs2, size)) {

		out = optr;
		in = iptr1;
		krn = iptr2;
		result = true;
	}

	if ((!result) && (detect_convcorr(N, odims, idims, kdims, &flags, &conv, dims, ostrs, istrs2, istrs1, size))) {

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

	//interface should support dilation and strides
	long dilation[N];
	long strides[N];
	md_singleton_dims(N, dilation);
	md_singleton_dims(N, strides);

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						odims, MD_STRIDES(N, odims, size),
						kdims, MD_STRIDES(N, kdims, size),
						idims, MD_STRIDES(N, idims, size),
						dilation, strides, conv, false) / size;
#ifdef USE_CUDA

	if(cuda_ondevice(out))
		for(int i = 0; (unsigned long)i < sizeof(algos_fwd_gpu) / sizeof(algos_fwd_gpu[0]); i++)
			if (algos_fwd_gpu[i](	N,
						odims, MD_STRIDES(N, odims, size), out,
						idims, MD_STRIDES(N, idims, size), in,
						kdims, MD_STRIDES(N, kdims, size), krn,
						flags, NULL, NULL, conv))
				return true;
#endif

	for(int i = 0; (unsigned long)i < sizeof(algos_fwd_cpu) / sizeof(algos_fwd_cpu[0]); i++)
		if (algos_fwd_cpu[i](	N,
					odims, MD_STRIDES(N, odims, size), out,
					idims, MD_STRIDES(N, idims, size), in,
					kdims, MD_STRIDES(N, kdims, size), krn,
					flags, NULL, NULL, conv))
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
	long odims[N];
	long idims[N];
	long kdims[N];

	const complex float* out = NULL;
	complex float* in = NULL;
	const complex float* krn = NULL;

	bool result = false;

	if (detect_convcorr(N, odims, idims, kdims, &flags, &conv, dims, istrs1, ostrs, istrs2, size)) {

		out = iptr1;
		in = optr;
		krn = iptr2;
		result = true;
	}

	if ((!result) && (detect_convcorr(N, odims, idims, kdims, &flags, &conv, dims, istrs2, ostrs, istrs1, size))) {

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

	//interface should support dilation and strides
	long dilation[N];
	long strides[N];
	md_singleton_dims(N, dilation);
	md_singleton_dims(N, strides);

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						odims, MD_STRIDES(N, odims, size),
						kdims, MD_STRIDES(N, kdims, size),
						idims, MD_STRIDES(N, idims, size),
						dilation, strides, conv, false) / size;

#ifdef USE_CUDA

	if(cuda_ondevice(out))
		for(int i = 0; (unsigned long)i < sizeof(algos_bwd_in_gpu) / sizeof(algos_bwd_in_gpu[0]); i++)
			if (algos_bwd_in_gpu[i](	N,
						odims, MD_STRIDES(N, odims, size), out,
						idims, MD_STRIDES(N, idims, size), in,
						kdims, MD_STRIDES(N, kdims, size), krn,
						flags, NULL, NULL, conv))
				return true;
#endif

	for(int i = 0; (unsigned long)i < sizeof(algos_bwd_in_cpu) / sizeof(algos_bwd_in_cpu[0]); i++)
		if (algos_bwd_in_cpu[i](	N,
					odims, MD_STRIDES(N, odims, size), out,
					idims, MD_STRIDES(N, idims, size), in,
					kdims, MD_STRIDES(N, kdims, size), krn,
					flags, NULL, NULL, conv))
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
	long odims[N];
	long idims[N];
	long kdims[N];

	const complex float* out = NULL;
	const complex float* in = NULL;
	complex float* krn = NULL;

	bool result = false;

	if (detect_convcorr(N, odims, idims, kdims, &flags, &conv, dims, istrs1, istrs2, ostrs, size)) {

		out = iptr1;
		in = iptr2;
		krn = optr;
		result = true;
	}

	if ((!result) && (detect_convcorr(N, odims, idims, kdims, &flags, &conv, dims, istrs2, istrs1, ostrs, size))) {

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

	//interface should support dilation and strides
	long dilation[N];
	long strides[N];
	md_singleton_dims(N, dilation);
	md_singleton_dims(N, strides);

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						odims, MD_STRIDES(N, odims, size),
						kdims, MD_STRIDES(N, kdims, size),
						idims, MD_STRIDES(N, idims, size),
						dilation, strides, conv, false) / size;

#ifdef USE_CUDA

	if(cuda_ondevice(out))
		for(int i = 0; (unsigned long)i < sizeof(algos_bwd_krn_gpu) / sizeof(algos_bwd_krn_gpu[0]); i++)
			if (algos_bwd_krn_gpu[i](	N,
						odims, MD_STRIDES(N, odims, size), out,
						idims, MD_STRIDES(N, idims, size), in,
						kdims, MD_STRIDES(N, kdims, size), krn,
						flags, NULL, NULL, conv))
				return true;
#endif

	for(int i = 0; (unsigned long)i < sizeof(algos_bwd_krn_cpu) / sizeof(algos_bwd_krn_cpu[0]); i++)
		if (algos_bwd_krn_cpu[i](	N,
					odims, MD_STRIDES(N, odims, size), out,
					idims, MD_STRIDES(N, idims, size), in,
					kdims, MD_STRIDES(N, kdims, size), krn,
					flags, NULL, NULL, conv))
			return true;

	return false;
}

/**
 * Checks if params correspond to convcorr which is channel first and contiguous in memory
 */

static bool check_trivial_cf_3d(int N, long odims[N], long ostrs[N], long idims[N], long istrs[N], long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], size_t size)
{
	if((28 != flags))
		return false;

	if ((NULL != dilation) && (!md_check_equal_dims(N, dilation, MD_SINGLETON_DIMS(N), ~(0l))))
		return false;
	if ((NULL != strides) && (!md_check_equal_dims(N, strides, MD_SINGLETON_DIMS(N), ~(0l))))
		return false;

	//check contigous memory
	if (5 > md_calc_blockdim(N, odims, ostrs, size))
		return false;
	if (5 > md_calc_blockdim(N, idims, istrs, size))
		return false;
	if (5 > md_calc_blockdim(N, kdims, kstrs, size))
		return false;

	//Check matmul dims
	if ((28 == flags) && ((1 != idims[0]) || (1 != odims[1])))
		return false;

	return true;

}


bool zconvcorr_fwd_direct_cf(	int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, size);
			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odims, idims, kdims);

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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, size);
			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odims, idims, kdims);

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

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
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


			cuda_im2col_transp((complex float*)ptr[0] + i * isize, imat_tmp , odims, idims, kdims);

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


bool test_zconvcorr_fwd(	int N, long odims[N], long ostrs[N], long idims[N], long istrs[N], long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], bool conv,
				float max_nrmse, long exp_nr_cpu, long exp_nr_gpu)
{
	bool result = true;
	complex float* optr_ref = md_alloc(N, odims, CFL_SIZE);

	complex float* optr = md_alloc(N, odims, CFL_SIZE);
	complex float* iptr = md_alloc(N, idims, CFL_SIZE);
	complex float* kptr = md_alloc(N, kdims, CFL_SIZE);

	md_gaussian_rand(N, idims, iptr);
	md_gaussian_rand(N, kdims, kptr);

	md_clear(N, odims, optr, CFL_SIZE);
	md_clear(N, odims, optr_ref, CFL_SIZE);

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

#if 1
	//force standard zfmac algorithm
	NESTED(void, nary_z3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zfmac(data->size, ptr[0], ptr[1], ptr[2]);
	};
	optimized_threeop_oii(2 * N, tdims, tostrs, optr_ref, tistrs, iptr, tkstrs, kptr + shift,
				(size_t[3]){ [0 ... 2] = CFL_SIZE }, nary_z3op);
#else
	md_zfmac2(2 * N, tdims, tostrs, optr_ref, tistrs, iptr, tkstrs, kptr + shift);
#endif

	long counter_cpu = 0;
	long counter_gpu = 0;

	for(int i = 0; (unsigned long)i < sizeof(algos_fwd_cpu) / sizeof(algos_fwd_cpu[0]); i++)
		if (algos_fwd_cpu[i](N, odims, ostrs, optr, idims, istrs, iptr, kdims, kstrs, kptr, flags, strides, dilation, conv)) {

			float err = md_znrmse(N, odims, optr_ref, optr);
			debug_printf(DP_DEBUG1, "error zconvcorr_fwd cpu algo %d: %.8f\n", i, err);
			md_clear(N, odims, optr, CFL_SIZE);
			counter_cpu += 1;

			result = result && (max_nrmse > err);
		}

#ifdef USE_CUDA
	num_init_gpu();
	complex float* optr_gpu = md_alloc_gpu(N, odims, CFL_SIZE);
	complex float* iptr_gpu = md_alloc_gpu(N, idims, CFL_SIZE);
	complex float* kptr_gpu = md_alloc_gpu(N, kdims, CFL_SIZE);

	md_copy(N, idims, iptr_gpu, iptr, CFL_SIZE);
	md_copy(N, kdims, kptr_gpu, kptr, CFL_SIZE);
	md_clear(N, odims, optr_gpu, CFL_SIZE);

	for(int i = 0; (unsigned long)i < sizeof(algos_fwd_gpu) / sizeof(algos_fwd_gpu[0]); i++)
		if (algos_fwd_gpu[i](N, odims, ostrs, optr_gpu, idims, istrs, iptr_gpu, kdims, kstrs, kptr_gpu, flags, strides, dilation, conv)) {

			md_copy(N, odims, optr, optr_gpu, CFL_SIZE);
			md_clear(N, odims, optr_gpu, CFL_SIZE);

			float err = md_znrmse(N, odims, optr_ref, optr);
			debug_printf(DP_DEBUG1, "error zconvcorr_fwd gpu algo %d: %.8f\n", i, err);
			counter_gpu += 1;

			result = result && (max_nrmse > err);
		}

	md_free(optr_gpu);
	md_free(iptr_gpu);
	md_free(kptr_gpu);

	if (counter_gpu < exp_nr_gpu) {

		debug_printf(DP_INFO, "zconvcorr_fwd only %d algorithms available on gpu(%d expected)\n", counter_gpu, exp_nr_gpu);
		result = false;
	}
#else
	UNUSED(exp_nr_gpu);
	UNUSED(counter_gpu);
#endif
	md_free(optr);
	md_free(optr_ref);
	md_free(iptr);
	md_free(kptr);

	if (counter_cpu < exp_nr_cpu) {

		debug_printf(DP_INFO, "zconvcorr_fwd only %d algorithms available on cpu(%d expected)\n", counter_cpu, exp_nr_cpu);
		result = false;
	}

	return result;
}

bool test_zconvcorr_bwd_in(	int N, long odims[N], long ostrs[N], long idims[N], long istrs[N], long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], bool conv,
				float max_nrmse, long exp_nr_cpu, long exp_nr_gpu)
{
	bool result = true;
	complex float* iptr_ref = md_alloc(N, idims, CFL_SIZE);

	complex float* optr = md_alloc(N, odims, CFL_SIZE);
	complex float* iptr = md_alloc(N, idims, CFL_SIZE);
	complex float* kptr = md_alloc(N, kdims, CFL_SIZE);

	md_gaussian_rand(N, odims, optr);
	md_gaussian_rand(N, kdims, kptr);

	md_clear(N, idims, iptr, CFL_SIZE);
	md_clear(N, idims, iptr_ref, CFL_SIZE);

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);



#if 1
	//force standard zfmac algorithm
	NESTED(void, nary_z3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zfmac(data->size, ptr[0], ptr[1], ptr[2]);
	};
	optimized_threeop_oii(2 * N, tdims, tistrs, iptr_ref, tkstrs, kptr + shift, tostrs, optr,
				(size_t[3]){ [0 ... 2] = CFL_SIZE }, nary_z3op);
#else
	md_zfmac2(2 * N, tdims, tistrs, iptr_ref, tkstrs, kptr + shift, tostrs, optr);
#endif

	long counter_cpu = 0;
	long counter_gpu = 0;

	for(int i = 0; (unsigned long)i < sizeof(algos_bwd_in_cpu) / sizeof(algos_bwd_in_cpu[0]); i++)
		if (algos_bwd_in_cpu[i](N, odims, ostrs, optr, idims, istrs, iptr, kdims, kstrs, kptr, flags, strides, dilation, conv)) {

			float err = md_znrmse(N, idims, iptr_ref, iptr);
			debug_printf(DP_DEBUG1, "error zconvcorr_bwd_in cpu algo %d: %.8f\n", i, err);
			md_clear(N, idims, iptr, CFL_SIZE);
			counter_cpu += 1;

			result = result && (max_nrmse > err);
		}

#ifdef USE_CUDA
	num_init_gpu();
	complex float* optr_gpu = md_alloc_gpu(N, odims, CFL_SIZE);
	complex float* iptr_gpu = md_alloc_gpu(N, idims, CFL_SIZE);
	complex float* kptr_gpu = md_alloc_gpu(N, kdims, CFL_SIZE);

	md_copy(N, odims, optr_gpu, optr, CFL_SIZE);
	md_copy(N, kdims, kptr_gpu, kptr, CFL_SIZE);
	md_clear(N, idims, iptr_gpu, CFL_SIZE);

	for(int i = 0; (unsigned long)i < sizeof(algos_bwd_in_gpu) / sizeof(algos_bwd_in_gpu[0]); i++)
		if (algos_bwd_in_gpu[i](N, odims, ostrs, optr_gpu, idims, istrs, iptr_gpu, kdims, kstrs, kptr_gpu, flags, strides, dilation, conv)) {

			md_copy(N, idims, iptr, iptr_gpu, CFL_SIZE);
			md_clear(N, idims, iptr_gpu, CFL_SIZE);

			float err = md_znrmse(N, idims, iptr_ref, iptr);
			debug_printf(DP_DEBUG1, "error zconvcorr_bwd_in gpu algo %d: %.8f\n", i, err);
			counter_gpu += 1;

			result = result && (max_nrmse > err);
		}

	md_free(optr_gpu);
	md_free(iptr_gpu);
	md_free(kptr_gpu);

	if (counter_gpu < exp_nr_gpu) {

		debug_printf(DP_INFO, "zconvcorr_bwd_in only %d algorithms available on gpu(%d expected)\n", counter_gpu, exp_nr_gpu);
		result = false;
	}
#else
	UNUSED(exp_nr_gpu);
	UNUSED(counter_gpu);
#endif
	md_free(optr);
	md_free(iptr_ref);
	md_free(iptr);
	md_free(kptr);

	if (counter_cpu < exp_nr_cpu) {

		debug_printf(DP_INFO, "zconvcorr_bwd_in only %d algorithms available on cpu(%d expected)\n", counter_cpu, exp_nr_cpu);
		result = false;
	}

	return result;
}

bool test_zconvcorr_bwd_krn(	int N, long odims[N], long ostrs[N], long idims[N], long istrs[N], long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], bool conv,
				float max_nrmse, long exp_nr_cpu, long exp_nr_gpu)
{
	bool result = true;
	complex float* kptr_ref = md_alloc(N, kdims, CFL_SIZE);

	complex float* optr = md_alloc(N, odims, CFL_SIZE);
	complex float* iptr = md_alloc(N, idims, CFL_SIZE);
	complex float* kptr = md_alloc(N, kdims, CFL_SIZE);

	md_gaussian_rand(N, odims, optr);
	md_gaussian_rand(N, idims, iptr);

	md_clear(N, kdims, kptr, CFL_SIZE);
	md_clear(N, kdims, kptr_ref, CFL_SIZE);

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

#if 1
	//force standard zfmac algorithm
	NESTED(void, nary_z3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zfmac(data->size, ptr[0], ptr[1], ptr[2]);
	};
	optimized_threeop_oii(2 * N, tdims, tkstrs, kptr_ref + shift, tostrs, optr, tistrs, iptr,
				(size_t[3]){ [0 ... 2] = CFL_SIZE }, nary_z3op);
#else
	md_zfmac2(2 * N, tdims, tkstrs, kptr_ref + shift, tostrs, optr, tistrs, iptr);
#endif

	long counter_cpu = 0;
	long counter_gpu = 0;

	for(int i = 0; (unsigned long)i < sizeof(algos_bwd_krn_cpu) / sizeof(algos_bwd_krn_cpu[0]); i++)
		if (algos_bwd_krn_cpu[i](N, odims, ostrs, optr, idims, istrs, iptr, kdims, kstrs, kptr, flags, strides, dilation, conv)) {

			float err = md_znrmse(N, kdims, kptr_ref, kptr);
			debug_printf(DP_DEBUG1, "error zconvcorr_bwd_krn cpu algo %d: %.8f\n", i, err);
			md_clear(N, kdims, kptr, CFL_SIZE);
			counter_cpu += 1;

			result = result && (max_nrmse > err);
		}

#ifdef USE_CUDA
	num_init_gpu();
	complex float* optr_gpu = md_alloc_gpu(N, odims, CFL_SIZE);
	complex float* iptr_gpu = md_alloc_gpu(N, idims, CFL_SIZE);
	complex float* kptr_gpu = md_alloc_gpu(N, kdims, CFL_SIZE);

	md_copy(N, odims, optr_gpu, optr, CFL_SIZE);
	md_copy(N, idims, iptr_gpu, iptr, CFL_SIZE);
	md_clear(N, kdims, kptr_gpu, CFL_SIZE);

	for(int i = 0; (unsigned long)i < sizeof(algos_bwd_krn_gpu) / sizeof(algos_bwd_krn_gpu[0]); i++)
		if (algos_bwd_krn_gpu[i](N, odims, ostrs, optr_gpu, idims, istrs, iptr_gpu, kdims, kstrs, kptr_gpu, flags, strides, dilation, conv)) {

			md_copy(N, kdims, kptr, kptr_gpu, CFL_SIZE);
			md_clear(N, kdims, kptr_gpu, CFL_SIZE);

			float err = md_znrmse(N, kdims, kptr_ref, kptr);
			debug_printf(DP_DEBUG1, "error zconvcorr_bwd_krn gpu algo %d: %.8f\n", i, err);
			counter_gpu += 1;

			result = result && (max_nrmse > err);
		}

	md_free(optr_gpu);
	md_free(iptr_gpu);
	md_free(kptr_gpu);

	if (counter_gpu < exp_nr_gpu) {

		debug_printf(DP_INFO, "zconvcorr_bwd_krn only %d algorithms available on gpu(%d expected)\n", counter_gpu, exp_nr_gpu);
		result = false;
	}
#else
	UNUSED(exp_nr_gpu);
	UNUSED(counter_gpu);
#endif
	md_free(optr);
	md_free(kptr_ref);
	md_free(iptr);
	md_free(kptr);

	if (counter_cpu < exp_nr_cpu) {

		debug_printf(DP_INFO, "zconvcorr_bwd_krn only %d algorithms available on cpu(%d expected)\n", counter_cpu, exp_nr_cpu);
		result = false;
	}

	return result;
}
