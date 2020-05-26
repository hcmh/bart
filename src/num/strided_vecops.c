#include <stdbool.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/optimize.h"
#include "num/blas_md_wrapper.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "strided_vecops.h"

typedef long (*md_check_3op_t)(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size);

typedef void (*md_3op_t)(unsigned int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);
typedef void (*md_z3op_t)(unsigned int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);

struct simple_z3op_check {

	md_check_3op_t check_fun;
	md_z3op_t strided_kernel;
	_Bool on_gpu;
	_Bool on_cpu;
	_Bool in_place;
};

struct simple_3op_check {

	md_check_3op_t check_fun;
	md_3op_t strided_kernel;
	_Bool on_gpu;
	_Bool on_cpu;
	_Bool in_place;
};


/**
 * Optimized two-op wrapper. Use when input is constant -- copy from flpmath.c
 *
 * @param D number of dimensions
 * @param dim dimensions
 * @param ostr output strides
 * @param optr output
 * @param istr1 input 1 strides
 * @param iptr1 input 1 (constant)
 * @param size size of data structures, e.g. complex float
 * @param too two-op multiply function
 */
static void optimized_twoop_oi(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr1[D], const void* iptr1, size_t sizes[2], md_nary_opt_fun_t too)
{
	const long (*nstr[2])[D?D:1] = { (const long (*)[D?D:1])ostr, (const long (*)[D?D:1])istr1 };
	void *nptr[2] = { optr, (void*)iptr1 };

	unsigned int io = 1 + ((iptr1 == optr) ? 2 : 0);

	optimized_nop(2, io, D, dim, nstr, nptr, sizes, too);
}

/**
 * Optimized threeop wrapper. Use when inputs are constants -- copy from flpmath.c
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


/**
 * Functions for optimizing fmac using blas
 * Checks if strides strides define a matrix,
 * i.e. one dimension is continuously in memory and followed by the other
 */
static bool is_matrix(const long dims[3], const long strs[3], int i1, int i2, size_t size)
{
	assert(i1 != i2);

	bool a = (   (strs[i1] == (long)size)
		  && (strs[i2] == (long)size * dims[i1]));

	bool b = (   (strs[i2] == (long)size)
		  && (strs[i1] == (long)size * dims[i2]));

	return a || b;
}

/**
 * Output: 3 if mat-mat-mul, -1, else
 *
 * if succesful, the out strides have the form:
 * nostrs:(size, 0, x| y1, ...| 0, 0, ...)
 * the in strides have the form
 * nistrs1: (x, x, 0| y2, ...| 0, 0, ...)
 * nistrs2: (0, x, x| y3, ...| 0, 0, ...)
 */
static long check_gemm(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if (3 > N)
		return -1;

	/*
	 * Find zeros in strides, matmuls have strides of the form
	 * (0, x, x)
	 * (x, 0, x)
	 * (x, x, 0)
	 * or permutations
	 */
	int opos = -1;
	int ipos1 = -1;
	int ipos2 = -1;

	for (int i = 0; i < 3; i++) {

		if (0 == tostrs[i])
			opos = i;

		if (0 == tistrs1[i])
			ipos1 = i;

		if (0 == tistrs2[i])
			ipos2 = i;
	}

	// pos of zeros do not equal
	bool matrix = (   (opos != ipos1)
		       && (opos != ipos2)
                       && (ipos1 != ipos2)
                       && (3 == opos + ipos1 + ipos2));

	// Check if matrix dims are continous in memory
	matrix = matrix && is_matrix(tdims, tostrs, (opos + 1) % 3, (opos + 2) % 3, size);
	matrix = matrix && is_matrix(tdims, tistrs1, (ipos1 + 1) % 3, (ipos1 + 2) % 3, size);
	matrix = matrix && is_matrix(tdims, tistrs2, (ipos2 + 1) % 3, (ipos2 + 2) % 3, size);

	// ipos1 is permute to index 2:
	matrix = matrix && tostrs[ipos1] > size;

	if (!matrix)
		return -1;

	/*
	 * Permute dims such that strides of output have the form
	 * (size, 0, x)
	 * the in strides have the form
	 * (x, x, 0)
	 * (0, x, x)
	 */
	unsigned int perm[N];

	for (unsigned int i = 3; i < N; i++)
		perm[i] = i;

	perm[0] = ipos2;
	perm[1] = opos;
	perm[2] = ipos1;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	return 3;
}

/**
 * Output: 2 if mat-vec-mul, -1, else
 *
 * if succesful, the out strides have the form:
 * nostrs: (x, 0| y1, ...| 0, 0, ...)
 * the in strides have the form
 * nistrs1: (x, x| y2, ...| 0, 0, ...)
 * nistrs2: (0, x| y3, ...| 0, 0, ...)
 */
static long check_gemv(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if (2 > N)
		return -1;

	/*
	 * (x, 0)
	 * (x, x)
	 * (0, x)
	 */

	bool matvecmul = true;

	matvecmul = matvecmul && (0 == tostrs[0] % size) && (0 == tostrs[1] % size);
	matvecmul = matvecmul && (0 == tistrs1[0] % size) && (0 == tistrs1[1] % size);
	matvecmul = matvecmul && (0 == tistrs2[0] % size) && (0 == tistrs2[1] % size);
	
	matvecmul = matvecmul && (0 == tostrs[0] * tistrs2[0]);
	matvecmul = matvecmul && (0 == tostrs[1] * tistrs2[1]);
	matvecmul = matvecmul && ((tostrs[0] > 0) || (tostrs[1] > 0));
	matvecmul = matvecmul && ((tistrs2[0] > 0) || (tistrs2[1] > 0));

	matvecmul = matvecmul && ( ((size == tistrs1[0]) && (tistrs1[1] >= tdims[0] * size))
				|| ((size == tistrs1[1]) && (tistrs1[0] >= tdims[1] * size)) );

	if (!matvecmul)
		return -1;

	unsigned int perm[N];

	for (unsigned int i = 2; i < N; i++)
		perm[i] = i;


	perm[0] = (tostrs[0] == 0) ? 1 : 0;
	perm[1] = (tostrs[0] == 0) ? 0 : 1;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	return 2;
}

/**
 * Output: 2 if mat-vec-mul, -1, else
 *
 * if succesful, the out strides have the form:
 * nostrs: (8, x| y1, ...| 0, 0, ...)
 * the in strides have the form
 * nistrs1: (x, 0| y2, ...| 0, 0, ...)
 * nistrs2: (0, x| y3, ...| 0, 0, ...)
 */
static long check_ger(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if ((2 > N) || ((size != tostrs[0]) && (size != tostrs[1])))
		return -1;

	unsigned int perm[N];

	for (unsigned int i = 2; i < N; i++)
		perm[i] = i;

	perm[0] = (tostrs[0] == size) ? 0 : 1;
	perm[1] = (tostrs[0] == size) ? 1 : 0;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	bool ger = true;
	ger = ger && (0 == nistrs1[1]) && (0 == nistrs2[0]);
	ger = ger && (0 < nistrs1[0]) && (0 < nistrs2[1]) && (size * ndims[0] <= nostrs[1]);
	ger = ger && (0 == nistrs1[0] % size) && (0 == nistrs2[1] % size) && (0 == nostrs[1] % size);

	return ger ? 2 : -1;
}


/**
 * Output: 1 if axpy, -1, else
 *
 * if succesful, the out strides have the form:
 * nostrs: (x | y1, ...| 0, 0, ...)
 * the in strides have the form
 * nistrs1: (x | y2, ...| 0, 0, ...)
 * nistrs2: (0 | y3, ...| 0, 0, ...)
 */
static long check_axpy(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if ((1 > N) || (   (0 != tostrs[0] % size) || (0 == tostrs[0])
			|| (0 != tistrs1[0] % size) || (0 == tistrs1[0])
			|| (0 != tistrs2[0])))
		return -1;

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);
	
	return 1;
}

/**
 * Output: 1 if dot, -1, else
 *
 * if succesful, the out strides have the form:
 * nostrs: (0 | y1, ...| 0, 0, ...)
 * the in strides have the form
 * nistrs1: (x | y2, ...| 0, 0, ...)
 * nistrs2: (x | y3, ...| 0, 0, ...)
 */
static long check_dot(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if ((1 > N) || (   (0 != tostrs[0])
			|| (0 != tistrs1[0] % size) || (0 == tistrs1[0])
			|| (0 != tistrs2[0] % size) || (0 == tistrs2[0]) ))
		return -1;

	md_copy_dims(N, ndims, tdims);
	md_copy_strides(N, nostrs, tostrs);
	md_copy_strides(N, nistrs1, tistrs1);
	md_copy_strides(N, nistrs2, tistrs2);
	
	return 1;
}

/**
 * Output: 1 if axpy, -1, else
 *
 * if succesful, the out strides have the form:
 * nostrs: (x | y1, ...| 0, 0, ...)
 * the in strides have the form
 * nistrs1: (x | y2, ...| 0, 0, ...)
 * nistrs2: (0 | y3, ...| 0, 0, ...)
 */
static long check_dgmm(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size)
{
	md_singleton_dims(N, ndims);
	md_singleton_strides(N, nostrs);
	md_singleton_strides(N, nistrs1);
	md_singleton_strides(N, nistrs2);

	long tdims[N];
	long tostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	md_copy_dims(N, tdims, dims);
	md_copy_strides(N, tostrs, ostrs);
	md_copy_strides(N, tistrs1, istrs1);
	md_copy_strides(N, tistrs2, istrs2);

	long (*strs[3])[N] = { &tostrs, &tistrs1, &tistrs2 };

	N = simplify_dims(3, N, tdims, strs);

	if (2 > N)
		return -1;

	unsigned int perm[N];
	for (unsigned int i = 2; i < N; i++)
		perm[i] = i;

	perm[0] = (tostrs[0] == size) ? 0 : 1;
	perm[1] = (tostrs[0] == size) ? 1 : 0;

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	bool dgmm = true;
	dgmm = dgmm && (size == nostrs[0]) && (size == nistrs1[0]);
	dgmm = dgmm && (size * ndims[0] <= nostrs[1]) && (size * ndims[0] == nistrs1[1]);
	dgmm = dgmm && ((0 == nistrs2[0]) || (0 == nistrs2[1]));
	dgmm = dgmm && (0 == nostrs[1] % size) && (0 == nistrs1[1] % size);
	dgmm = dgmm && (0 == nistrs2[0] % size) && (0 == nistrs2[1] % size);
	if (!dgmm)
		return -1;

	return 2;
}



bool simple_zfmac(unsigned int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	size_t size = 8;

	struct simple_z3op_check strided_calls[] = {	{check_gemm,  blas_zfmac_cgemm, true, true, false},
							{check_gemv,  blas_zfmac_cgemv, true, true, false},
							{check_ger,   blas_zfmac_cgeru, true, true, false},
							{check_axpy,  blas_zfmac_caxpy, true, true, false},
							{check_dot,   blas_zfmac_cdotu, true, true, false}
						};

	long ndims[N];
	long nostrs[N];
	long nistrs1[N];
	long nistrs2[N];

	const complex float* tin1 = NULL;
	const complex float* tin2 = NULL;

	long N_in = -1;
	md_z3op_t strided_kernel = NULL;

	for (unsigned int i = 0; i < sizeof(strided_calls) / sizeof(strided_calls[0]); i++) {

		bool applicable = true;
		strided_kernel = strided_calls[i].strided_kernel;

	#ifdef USE_CUDA
		if (cuda_ondevice(out))
			applicable &= strided_calls[i].on_gpu;
		else
	#endif
			applicable &= strided_calls[i].on_cpu;
		if (!applicable)
			continue;

		tin1 = in1;
		tin2 = in2;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs1, istrs2, size);
		if (-1 != N_in)
			break;

		tin1 = in2;
		tin2 = in1;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs2, istrs1, size);
		if (-1 != N_in)
			break;
	}

	if (-1 == N_in)
		return false;
	
	size_t osize = 0;
	size_t isize1 = 0;
	size_t isize2 = 0;

	for (int i = 0; i < N_in; i++) {

		osize = MAX(osize, nostrs[i] * ndims[i]);
		isize1 = MAX(osize, nistrs1[i] * ndims[i]);
		isize2 = MAX(osize, nistrs2[i] * ndims[i]);
	}

	//clang
	long* ndims_ptr = &ndims[0];
	long* nostrs_ptr = &nostrs[0];
	long* nistrs1_ptr = &nistrs1[0];
	long* nistrs2_ptr = &nistrs2[0];


	NESTED(void, nary_inner_zfmac, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			strided_kernel(	N_in, ndims_ptr,
					nostrs_ptr, (complex float*)(ptr[0] + i * osize),
					nistrs1_ptr, (const complex float*)(ptr[1] + i * isize1),
					nistrs2_ptr, (const complex float*)(ptr[2] + i * isize2));
	};

	optimized_threeop_oii(	N - N_in, ndims + N_in,
				nostrs + N_in, (void*)out, nistrs1 + N_in, (void*)tin1, nistrs2 + N_in, (void*)tin2,
				(size_t[3]){ osize, isize1, isize2 }, nary_inner_zfmac);

	return true;
}

bool simple_fmac(unsigned int N, const long dims[N], const long ostrs[N], float* out, const long istrs1[N], const float* in1, const long istrs2[N], const float* in2)
{
	size_t size = 4;

	struct simple_3op_check strided_calls[] = {	{check_gemm,  blas_fmac_sgemm, true, true, false},
							{check_gemv,  blas_fmac_sgemv, true, true, false},
							{check_ger,   blas_fmac_sger,  true, true, false},
							{check_axpy,  blas_fmac_saxpy, true, true, false},
							{check_dot,   blas_fmac_sdot,  true, true, false}
						};

	long ndims[N];
	long nostrs[N];
	long nistrs1[N];
	long nistrs2[N];

	const float* tin1 = NULL;
	const float* tin2 = NULL;

	long N_in = -1;
	md_3op_t strided_kernel = NULL;

	for (unsigned int i = 0; i < sizeof(strided_calls) / sizeof(strided_calls[0]); i++) {

		bool applicable = true;
		strided_kernel = strided_calls[i].strided_kernel;

	#ifdef USE_CUDA
		if (cuda_ondevice(out))
			applicable &= strided_calls[i].on_gpu;
		else
	#endif
			applicable &= strided_calls[i].on_cpu;
		if (!applicable)
			continue;

		tin1 = in1;
		tin2 = in2;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs1, istrs2, size);
		if (-1 != N_in)
			break;

		tin1 = in2;
		tin2 = in1;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs2, istrs1, size);
		if (-1 != N_in)
			break;
	}

	if (-1 == N_in)
		return false;
	
	size_t osize = 0;
	size_t isize1 = 0;
	size_t isize2 = 0;

	for (int i = 0; i < N_in; i++) {

		osize = MAX(osize, nostrs[i] * ndims[i]);
		isize1 = MAX(osize, nistrs1[i] * ndims[i]);
		isize2 = MAX(osize, nistrs2[i] * ndims[i]);
	}

	//clang
	long* ndims_ptr = &ndims[0];
	long* nostrs_ptr = &nostrs[0];
	long* nistrs1_ptr = &nistrs1[0];
	long* nistrs2_ptr = &nistrs2[0];


	NESTED(void, nary_inner_zfmac, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			strided_kernel(	N_in, ndims_ptr,
					nostrs_ptr, (float*)(ptr[0] + i * osize),
					nistrs1_ptr, (const float*)(ptr[1] + i * isize1),
					nistrs2_ptr, (const float*)(ptr[2] + i * isize2));
	};

	optimized_threeop_oii(	N - N_in, ndims + N_in,
				nostrs + N_in, (void*)out, nistrs1 + N_in, (void*)tin1, nistrs2 + N_in, (void*)tin2,
				(size_t[3]){ osize, isize1, isize2 }, nary_inner_zfmac);

	return true;
}

bool simple_zmul(unsigned int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	size_t size = 8;

	struct simple_z3op_check strided_calls[] = {	{check_ger,   blas_zmul_cgeru, true, true, false},
							{check_dgmm,  blas_zmul_cdgmm, true, false, true},
							{check_axpy,  blas_zmul_cscal, true, true, true}
						};

	long ndims[N];
	long nostrs[N];
	long nistrs1[N];
	long nistrs2[N];

	const complex float* tin1 = NULL;
	const complex float* tin2 = NULL;

	long N_in = -1;
	md_z3op_t strided_kernel = NULL;

	for (unsigned int i = 0; i < sizeof(strided_calls) / sizeof(strided_calls[0]); i++) {

		bool applicable = true;
		strided_kernel = strided_calls[i].strided_kernel;

	#ifdef USE_CUDA
		if (cuda_ondevice(out))
			applicable &= strided_calls[i].on_gpu;
		else
	#endif
			applicable &= strided_calls[i].on_cpu;
		if (!applicable)
			continue;

		tin1 = in1;
		tin2 = in2;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs1, istrs2, size);
		if (-1 != N_in)
			break;

		tin1 = in2;
		tin2 = in1;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs2, istrs1, size);
		if (-1 != N_in)
			break;
	}

	if (-1 == N_in)
		return false;
	
	size_t osize = 0;
	size_t isize1 = 0;
	size_t isize2 = 0;

	for (int i = 0; i < N_in; i++) {

		osize = MAX(osize, nostrs[i] * ndims[i]);
		isize1 = MAX(osize, nistrs1[i] * ndims[i]);
		isize2 = MAX(osize, nistrs2[i] * ndims[i]);
	}

	//clang
	long* ndims_ptr = &ndims[0];
	long* nostrs_ptr = &nostrs[0];
	long* nistrs1_ptr = &nistrs1[0];
	long* nistrs2_ptr = &nistrs2[0];


	NESTED(void, nary_inner_zfmac, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			strided_kernel(	N_in, ndims_ptr,
					nostrs_ptr, (complex float*)(ptr[0] + i * osize),
					nistrs1_ptr, (const complex float*)(ptr[1] + i * isize1),
					nistrs2_ptr, (const complex float*)(ptr[2] + i * isize2));
	};

	optimized_threeop_oii(	N - N_in, ndims + N_in,
				nostrs + N_in, (void*)out, nistrs1 + N_in, (void*)tin1, nistrs2 + N_in, (void*)tin2,
				(size_t[3]){ osize, isize1, isize2 }, nary_inner_zfmac);

	return true;
}

bool simple_mul(unsigned int N, const long dims[N], const long ostrs[N], float* out, const long istrs1[N], const float* in1, const long istrs2[N], const float* in2)
{
	size_t size = 4;

	struct simple_3op_check strided_calls[] = {	{check_ger,   blas_mul_sger, true, true, false},
							{check_dgmm,  blas_mul_sdgmm, true, false, true},
							{check_axpy,  blas_mul_sscal, true, true, true}
						};

	long ndims[N];
	long nostrs[N];
	long nistrs1[N];
	long nistrs2[N];

	const float* tin1 = NULL;
	const float* tin2 = NULL;

	long N_in = -1;
	md_3op_t strided_kernel = NULL;

	for (unsigned int i = 0; i < sizeof(strided_calls) / sizeof(strided_calls[0]); i++) {

		bool applicable = true;
		strided_kernel = strided_calls[i].strided_kernel;

	#ifdef USE_CUDA
		if (cuda_ondevice(out))
			applicable &= strided_calls[i].on_gpu;
		else
	#endif
			applicable &= strided_calls[i].on_cpu;
		if (!applicable)
			continue;

		tin1 = in1;
		tin2 = in2;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs1, istrs2, size);
		if (-1 != N_in)
			break;

		tin1 = in2;
		tin2 = in1;
		N_in = strided_calls[i].check_fun(N, ndims, nostrs, nistrs1, nistrs2, dims, ostrs, istrs2, istrs1, size);
		if (-1 != N_in)
			break;
	}

	if (-1 == N_in)
		return false;
	
	size_t osize = 0;
	size_t isize1 = 0;
	size_t isize2 = 0;

	for (int i = 0; i < N_in; i++) {

		osize = MAX(osize, nostrs[i] * ndims[i]);
		isize1 = MAX(osize, nistrs1[i] * ndims[i]);
		isize2 = MAX(osize, nistrs2[i] * ndims[i]);
	}

	//clang
	long* ndims_ptr = &ndims[0];
	long* nostrs_ptr = &nostrs[0];
	long* nistrs1_ptr = &nistrs1[0];
	long* nistrs2_ptr = &nistrs2[0];


	NESTED(void, nary_inner_zfmac, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (long i = 0; i < data->size; i++)
			strided_kernel(	N_in, ndims_ptr,
					nostrs_ptr, (float*)(ptr[0] + i * osize),
					nistrs1_ptr, (const float*)(ptr[1] + i * isize1),
					nistrs2_ptr, (const float*)(ptr[2] + i * isize2));
	};

	optimized_threeop_oii(	N - N_in, ndims + N_in,
				nostrs + N_in, (void*)out, nistrs1 + N_in, (void*)tin1, nistrs2 + N_in, (void*)tin2,
				(size_t[3]){ osize, isize1, isize2 }, nary_inner_zfmac);

	return true;
}