/* Copyright 2013-2018 The Regents of the University of California.
 * Copyright 2016-2020. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014-2018 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Siddharth Iyer <sid8795@gmail.com>
 * 2017 Sofia Dimoudi <sofia.dimoudi@cardiov.ox.ac.uk>
 *
 *
 * Operations on arrays of complex single-precision floating
 * point numbers. Most functions come in two flavours:
 *
 * 1. A basic version which takes the number of dimensions, an array
 * of long integers specifing the size of each dimension, the pointers
 * to the data, and the size of each element and other required parameters.
 *
 * 2. An extended version which takes an array of long integers which
 * specifies the strides for each argument.
 *
 * All functions should work on CPU and GPU.
 *
 */

#include <stddef.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/vecops.h"
#include "num/optimize.h"
#include "num/blas.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/nested.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
/*
 * including gpukrnls.h so that I can directly call cuda_zreal.
 * this can be removed after md_zreal is optimized for GPU.
 */
#include "num/gpukrnls.h"
#endif


typedef void (*md_2op_t)(unsigned int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1);
typedef void (*md_z2op_t)(unsigned int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1);
typedef void (*md_2opf_t)(unsigned int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const double* iptr1);
typedef void (*md_2opd_t)(unsigned int D, const long dims[D], const long ostrs[D], double* optr, const long istrs1[D], const float* iptr1);
typedef void (*md_z2opf_t)(unsigned int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex double* iptr1);
typedef void (*md_z2opd_t)(unsigned int D, const long dims[D], const long ostrs[D], complex double* optr, const long istrs1[D], const complex float* iptr1);


typedef void (*md_3op_t)(unsigned int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);
typedef void (*md_z3op_t)(unsigned int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);
typedef void (*md_3opd_t)(unsigned int D, const long dims[D], const long ostrs[D], double* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);
typedef void (*md_z3opd_t)(unsigned int D, const long dims[D], const long ostrs[D], complex double* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);


#if 0
static void optimized_twoop(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr1[D], void* iptr1, size_t sizes[2], md_nary_fun_t too, void* data_ptr) __attribute__((always_inline));

static void optimized_twoop_oi(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr1[D], const void* iptr1, size_t sizes[2], md_nary_fun_t too, void* data_ptr) __attribute__((always_inline));

static void optimized_threeop(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr1[D], void* iptr1, const long istr2[D], void* iptr2, size_t sizes[3], md_nary_fun_t too, void* data_ptr) __attribute__((always_inline));

static void optimized_threeop_oii(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr1[D], const void* iptr1, const long istr2[D], const void* iptr2, size_t sizes[3], md_nary_fun_t too, void* data_ptr) __attribute__((always_inline));

static void make_z3op_simple(md_z3op_t fun, unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2) __attribute__((always_inline));

static void make_3op_simple(md_3op_t fun, unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2) __attribute__((always_inline));

static void make_z3op(size_t offset, unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2) __attribute__((always_inline));

static void make_3opd_simple(md_3opd_t fun, unsigned int D, const long dims[D], double* optr, const float* iptr1, const float* iptr2) __attribute__((always_inline));

static void make_z2op_simple(md_z2op_t fun, unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1) __attribute__((always_inline));

static void make_2op_simple(md_2op_t fun, unsigned int D, const long dims[D], float* optr, const float* iptr1) __attribute__((always_inline));
#endif





/**
 * Optimized two-op wrapper. Use when input is constant
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



/* HELPER FUNCTIONS
 *
 * The following functions, typedefs, and macros are used internally in flpmath.c
 * to simplify implementation of many similar functions.
 */


typedef void (*r2op_t)(long N, float* dst, const float* src1);
typedef void (*z2op_t)(long N, complex float* dst, const complex float* src1);
typedef void (*r3op_t)(long N, float* dst, const float* src1, const float* src2);
typedef void (*z3op_t)(long N, complex float* dst, const complex float* src1, const complex float* src2);
typedef void (*r2opd_t)(long N, double* dst, const float* src1);
typedef void (*z2opd_t)(long N, complex double* dst, const complex float* src1);
typedef void (*r3opd_t)(long N, double* dst, const float* src1, const float* src2);
typedef void (*z3opd_t)(long N, complex double* dst, const complex float* src1, const complex float* src2);
typedef void (*r2opf_t)(long N, float* dst, const double* src1);
typedef void (*z2opf_t)(long N, complex float* dst, const complex double* src1);




static void make_z3op_simple(md_z3op_t fun, unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	fun(D, dims, strs, optr, strs, iptr1, strs, iptr2);
}

static void make_3op_simple(md_3op_t fun, unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	fun(D, dims, strs, optr, strs, iptr1, strs, iptr2);
}

static void make_z3opd_simple(md_z3opd_t fun, unsigned int D, const long dims[D], complex double* optr, const complex float* iptr1, const complex float* iptr2)
{
	long strs_single[D];
	long strs_double[D];

	md_calc_strides(D, strs_single, dims, CFL_SIZE);
	md_calc_strides(D, strs_double, dims, CDL_SIZE);

	fun(D, dims, strs_double, optr, strs_single, iptr1, strs_single, iptr2);
}

static void make_3opd_simple(md_3opd_t fun, unsigned int D, const long dims[D], double* optr, const float* iptr1, const float* iptr2)
{
	long strs_single[D];
	long strs_double[D];

	md_calc_strides(D, strs_single, dims, FL_SIZE);
	md_calc_strides(D, strs_double, dims, DL_SIZE);

	fun(D, dims, strs_double, optr, strs_single, iptr1, strs_single, iptr2);
}

static void make_z2op_simple(md_z2op_t fun, unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	fun(D, dims, strs, optr, strs, iptr1);
}

static void make_2op_simple(md_2op_t fun, unsigned int D, const long dims[D], float* optr, const float* iptr1)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	fun(D, dims, strs, optr, strs, iptr1);
}

static void make_z2opd_simple(md_z2opd_t fun, unsigned int D, const long dims[D], complex double* optr, const complex float* iptr1)
{
	long strs_single[D];
	long strs_double[D];

	md_calc_strides(D, strs_single, dims, CFL_SIZE);
	md_calc_strides(D, strs_double, dims, CDL_SIZE);

	fun(D, dims, strs_double, optr, strs_single, iptr1);
}

static void make_2opd_simple(md_2opd_t fun, unsigned int D, const long dims[D], double* optr, const float* iptr1)
{
	long strs_single[D];
	long strs_double[D];

	md_calc_strides(D, strs_single, dims, FL_SIZE);
	md_calc_strides(D, strs_double, dims, DL_SIZE);

	fun(D, dims, strs_double, optr, strs_single, iptr1);
}

static void make_z3op(size_t offset, unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	NESTED(void, nary_z3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(z3op_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1], ptr[2]);
	};

	optimized_threeop_oii(D, dim, ostr, optr, istr1, iptr1, istr2, iptr2,
				(size_t[3]){ [0 ... 2] = CFL_SIZE }, nary_z3op);
}


static void make_3op(size_t offset, unsigned int D, const long dim[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	NESTED(void, nary_3op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(r3op_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1], ptr[2]);
	};

	optimized_threeop_oii(D, dim, ostr, optr, istr1, iptr1, istr2, iptr2,
				(size_t[3]){ [0 ... 2] = FL_SIZE }, nary_3op);
}

static void make_z3opd(size_t offset, unsigned int D, const long dim[D], const long ostr[D], complex double* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	NESTED(void, nary_z3opd, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(z3opd_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1], ptr[2]);
	};

	optimized_threeop_oii(D, dim, ostr, optr, istr1, iptr1, istr2, iptr2,
			(size_t[3]){ CDL_SIZE, CFL_SIZE, CFL_SIZE }, nary_z3opd);
}

static void make_3opd(size_t offset, unsigned int D, const long dim[D], const long ostr[D], double* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	NESTED(void, nary_3opd, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(r3opd_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1], ptr[2]);
	};

	optimized_threeop_oii(D, dim, ostr, optr, istr1, iptr1, istr2, iptr2,
			(size_t[3]){ DL_SIZE, FL_SIZE, FL_SIZE }, nary_3opd);
}

static void make_z2op(size_t offset, unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1)
{
	NESTED(void, nary_z2op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(z2op_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr1, iptr1, (size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_z2op);
}

static void make_2op(size_t offset, unsigned int D, const long dim[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1)
{
	NESTED(void, nary_2op, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(r2op_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr1, iptr1, (size_t[2]){ FL_SIZE, FL_SIZE }, nary_2op);
}

__attribute__((unused))
static void make_z2opd(size_t offset, unsigned int D, const long dim[D], const long ostr[D], complex double* optr, const long istr1[D], const complex float* iptr1)
{
	size_t sizes[2] = { sizeof(complex double), sizeof(complex float) };

	NESTED(void, nary_z2opd, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(z2opd_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr1, iptr1, sizes, nary_z2opd);
}


static void make_2opd(size_t offset, unsigned int D, const long dim[D], const long ostr[D], double* optr, const long istr1[D], const float* iptr1)
{
	NESTED(void, nary_2opd, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(r2opd_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr1, iptr1, (size_t[2]){ DL_SIZE, FL_SIZE }, nary_2opd);
}

static void make_z2opf(size_t offset, unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex double* iptr1)
{
	size_t sizes[2] = { sizeof(complex float), sizeof(complex double) };

	NESTED(void, nary_z2opf, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(z2opf_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr1, iptr1, sizes, nary_z2opf);
}

void* unused2 = make_z2opf;

static void make_2opf(size_t offset, unsigned int D, const long dim[D], const long ostr[D], float* optr, const long istr1[D], const double* iptr1)
{
	NESTED(void, nary_2opf, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(*(r2opf_t*)(((char*)data->ops) + offset))(data->size, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr1, iptr1, (size_t[2]){ FL_SIZE, DL_SIZE }, nary_2opf);
}

static void make_z2opf_simple(md_z2opf_t fun, unsigned int D, const long dims[D], complex float* optr, const complex double* iptr1)
{
	long strs_single[D];
	long strs_double[D];

	md_calc_strides(D, strs_single, dims, CFL_SIZE);
	md_calc_strides(D, strs_double, dims, CDL_SIZE);

	fun(D, dims, strs_single, optr, strs_double, iptr1);
}

static void make_2opf_simple(md_2opf_t fun, unsigned int D, const long dims[D], float* optr, const double* iptr1)
{
	long strs_single[D];
	long strs_double[D];

	md_calc_strides(D, strs_single, dims, FL_SIZE);
	md_calc_strides(D, strs_double, dims, DL_SIZE);

	fun(D, dims, strs_single, optr, strs_double, iptr1);
}

#ifdef USE_CUDA
static void* gpu_constant(const void* vp, size_t size)
{
	return md_gpu_move(1, (long[1]){ 1 }, vp, size);
}
#endif

static void make_z3op_scalar(md_z3op_t fun, unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr, complex float val)
{
	complex float* valp = &val;

#ifdef USE_CUDA
	if (cuda_ondevice(optr))
		valp = gpu_constant(&val, CFL_SIZE);
#endif

	long strs1[D];
	md_singleton_strides(D, strs1);

	fun(D, dims, ostr, optr, istr, iptr, strs1, valp);

#ifdef USE_CUDA
	if (cuda_ondevice(optr))
		md_free(valp);
#endif
}

static void make_3op_scalar(md_3op_t fun, unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float val)
{
	float* valp = &val;

#ifdef USE_CUDA
	if (cuda_ondevice(optr))
		valp = gpu_constant(&val, FL_SIZE);
#endif

	long strs1[D];
	md_singleton_strides(D, strs1);

	fun(D, dims, ostr, optr, istr, iptr, strs1, valp);

#ifdef USE_CUDA
	if (cuda_ondevice(optr))
		md_free(valp);
#endif
}

static void real_from_complex_dims(unsigned int D, long odims[D + 1], const long idims[D])
{
	odims[0] = 2;
	md_copy_dims(D, odims + 1, idims);
}

static void real_from_complex_strides(unsigned int D, long ostrs[D + 1], const long istrs[D])
{
	ostrs[0] = FL_SIZE;
	md_copy_dims(D, ostrs + 1, istrs);	// works for strides too
}

static void make_z3op_from_real(size_t offset, unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	long rdims[D + 1];
	long rostr[D + 1];
	long ristr1[D + 1];
	long ristr2[D + 1];

	real_from_complex_dims(D, rdims, dims);
	real_from_complex_strides(D, rostr, ostr);
	real_from_complex_strides(D, ristr1, istr1);
	real_from_complex_strides(D, ristr2, istr2);

	make_3op(offset, D + 1, rdims, rostr, (float*)optr, ristr1, (const float*)iptr1, ristr2, (const float*)iptr2);
}

static void make_z2opd_from_real(size_t offset, unsigned int D, const long dims[D], const long ostr[D], complex double* optr, const long istr1[D], const complex float* iptr1)
{
	long rdims[D + 1];
	long rostr[D + 1];
	long ristr1[D + 1];

	real_from_complex_dims(D, rdims, dims);
	real_from_complex_strides(D, rostr, ostr);
	real_from_complex_strides(D, ristr1, istr1);

	make_2opd(offset, D + 1, rdims, rostr, (double*)optr, ristr1, (const float*)iptr1);
}

static void make_z2opf_from_real(size_t offset, unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex double* iptr1)
{
	long rdims[D + 1];
	long rostr[D + 1];
	long ristr1[D + 1];

	real_from_complex_dims(D, rdims, dims);
	real_from_complex_strides(D, rostr, ostr);
	real_from_complex_strides(D, ristr1, istr1);

	make_2opf(offset, D + 1, rdims, rostr, (float*)optr, ristr1, (const double*)iptr1);
}


// type safe
#define MAKE_3OP(fun, ...)	((void)TYPE_CHECK(r3op_t, cpu_ops.fun), make_3op(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_Z3OP(fun, ...)	((void)TYPE_CHECK(z3op_t, cpu_ops.fun), make_z3op(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_2OP(fun, ...)	((void)TYPE_CHECK(r2op_t, cpu_ops.fun), make_2op(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_Z2OP(fun, ...)	((void)TYPE_CHECK(z2op_t, cpu_ops.fun), make_z2op(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_2OPD(fun, ...)	((void)TYPE_CHECK(r2opd_t, cpu_ops.fun), make_2opd(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_Z2OPD(fun, ...)	((void)TYPE_CHECK(z2opd_t, cpu_ops.fun), make_z2opd(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_2OPF(fun, ...)	((void)TYPE_CHECK(r2opf_t, cpu_ops.fun), make_2opf(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_Z2OPF(fun, ...)	((void)TYPE_CHECK(z2opf_t, cpu_ops.fun), make_z2opf(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_3OPD(fun, ...)	((void)TYPE_CHECK(r3opd_t, cpu_ops.fun), make_3opd(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_Z3OPD(fun, ...)	((void)TYPE_CHECK(z3opd_t, cpu_ops.fun), make_z3opd(offsetof(struct vec_ops, fun),  __VA_ARGS__))
#define MAKE_Z3OP_FROM_REAL(fun, ...) \
				((void)TYPE_CHECK(r3op_t, cpu_ops.fun), make_z3op_from_real(offsetof(struct vec_ops, fun), __VA_ARGS__))
#define MAKE_Z2OPD_FROM_REAL(fun, ...) \
				((void)TYPE_CHECK(r2opd_t, cpu_ops.fun), make_z2opd_from_real(offsetof(struct vec_ops, fun), __VA_ARGS__))
#define MAKE_Z2OPF_FROM_REAL(fun, ...) \
				((void)TYPE_CHECK(r2opf_t, cpu_ops.fun), make_z2opf_from_real(offsetof(struct vec_ops, fun), __VA_ARGS__))




/* The section with exported functions starts here. */








/**
 * Multiply two complex arrays and save to output (with strides)
 *
 * optr = iptr1 * iptr2
 */
void md_zmul2(unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP(zmul, D, dim, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply two complex arrays and save to output  (without strides)
 *
 * optr = iptr1 * iptr2
 */
void md_zmul(unsigned int D, const long dim[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zmul2, D, dim, optr, iptr1, iptr2);
}



/**
 * Multiply two scalar arrays and save to output (with strides)
 *
 * optr = iptr1 * iptr2
 */
void md_mul2(unsigned int D, const long dim[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(mul, D, dim, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply two scalar arrays and save to output (without strides)
 *
 * optr = iptr1 * iptr2
 */
void md_mul(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	make_3op_simple(md_mul2, D, dims, optr, iptr1, iptr2);
}



/**
 * Multiply real and imaginary parts of two complex arrays separately and save to output (with strides)
 *
 * real(optr) = real(iptr1) * real(iptr2)
 *
 * imag(optr) = imag(iptr1) * imag(iptr2)
 */
void md_zrmul2(unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP_FROM_REAL(mul, D, dim, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply real and imaginary parts of two complex arrays separately and save to output (without strides)
 *
 * real(optr) = real(iptr1) * real(iptr2)
 *
 * imag(optr) = imag(iptr1) * imag(iptr2)
 */
void md_zrmul(unsigned int D, const long dim[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zrmul2, D, dim, optr, iptr1, iptr2);
}



/**
 * Multiply complex array with a scalar and save to output (with strides)
 *
 * optr = iptr * val
 */
void md_zsmul2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr, complex float val)
{
	if (0. == cimagf(val)) { // strength reduction: complex to real multiplication

		long dimsR[D + 1];
		long ostrR[D + 1];
		long istrR[D + 1];

		real_from_complex_dims(D, dimsR, dims);
		real_from_complex_strides(D, ostrR, ostr);
		real_from_complex_strides(D, istrR, istr);

		md_smul2(D + 1, dimsR, ostrR, (float*)optr, istrR, (const float*)iptr, crealf(val));
		return;
	}

#if 0
	make_z3op_scalar(md_zmul2, D, dims, ostr, optr, istr, iptr, val);
#else
	// FIXME: we should rather optimize md_zmul2 for this case

	NESTED(void, nary_zsmul, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zsmul(data->size, val, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dims, ostr, optr, istr, iptr,
		(size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zsmul);
#endif
}



/**
 * Multiply complex array with a scalar and save to output (without strides)
 *
 * optr = iptr * val
 */
void md_zsmul(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr, complex float var)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zsmul2(D, dims, strs, optr, strs, iptr, var);
}



/**
 * Multiply scalar array with a scalar and save to output (with strides)
 *
 * optr = iptr * var
 */
void md_smul2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float var)
{
#ifdef USE_CUDA

	if (cuda_ondevice(iptr)) {

		assert(cuda_ondevice(optr));

		if (md_calc_blockdim(D, dims, ostr, FL_SIZE) != D)
			goto fallback;

		if (md_calc_blockdim(D, dims, istr, FL_SIZE) != D)
			goto fallback;

		if (iptr == optr) {

			gpu_ops.axpy(md_calc_size(D, dims), optr, var - 1., iptr);
			return;
		}

		// no strides needed because of checks above

		md_clear(D, dims, optr, FL_SIZE);

		// or call md_zaxpy
		gpu_ops.axpy(md_calc_size(D, dims), optr, var, iptr);
		return;
	}
fallback:
#endif

#if 0
	make_3op_scalar(md_mul2, D, dims, ostr, optr, istr, iptr, var);
#else
	// FIXME: we should rather optimize md_mul2 for this case

	(void)0;

	NESTED(void, nary_smul, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->smul(data->size, var, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dims, ostr, optr, istr, iptr,
		(size_t[2]){ FL_SIZE, FL_SIZE }, nary_smul);
#endif
}



/**
 * Multiply scalar array with a scalar and save to output (without strides)
 *
 * optr = iptr * var
 */
void md_smul(unsigned int D, const long dims[D], float* optr, const float* iptr, float var)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_smul2(D, dims, strs, optr, strs, iptr, var);
}



/**
 * Multiply the first complex array with the conjugate of the second complex array and save to output (with strides)
 *
 * optr = iptr1 * conj(iptr2)
 */
void md_zmulc2(unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP(zmulc, D, dim, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply the first complex array with the conjugate of the second complex array and save to output (without strides)
 *
 * optr = iptr1 * conj(iptr2)
 */
void md_zmulc(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zmulc2, D, dims, optr, iptr1, iptr2);
}



/**
 * Divide the first complex array by the second complex array and save to output (with strides)
 *
 * optr = iptr1 / iptr2
 */
void md_zdiv2(unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP(zdiv, D, dim, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Divide the first complex array by the second complex array and save to output (without strides)
 *
 * optr = iptr1 / iptr2
 */
void md_zdiv(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zdiv2, D, dims, optr, iptr1, iptr2);
}


/**
 * Divide the first complex array by the second complex array with regularization and save to output (with strides)
 *
 * optr = iptr1 / (iptr2 + epsilon)
 */
void md_zdiv_reg2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2, complex float lambda)
{
	NESTED(void, nary_zdiv_reg, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zdiv_reg(data->size, ptr[0], ptr[1], ptr[2], lambda);
	};

	optimized_threeop_oii(D, dims, ostr, optr, istr1, iptr1, istr2, iptr2,
				(size_t[3]){ [0 ... 2] = CFL_SIZE }, nary_zdiv_reg);
}


/**
 * Divide the first complex array by the second complex array with regularization and save to output (without strides)
 *
 * optr = iptr1 / (iptr2 + epsilon)
 */
void md_zdiv_reg(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2, complex float lambda)
{
	long str[D];
	md_calc_strides(D, str, dims, CFL_SIZE);

	md_zdiv_reg2(D, dims, str, optr, str, iptr1, str, iptr2, lambda);
}



/**
 * Divide the first scalar array by the second scalar array and save to output (with strides)
 *
 * optr = iptr1 / iptr2
 */
void md_div2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(div, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Divide the first scalar array by the second scalar array and save to output (without strides)
 *
 * optr = iptr1 / iptr2
 */
void md_div(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	make_3op_simple(md_div2, D, dims, optr, iptr1, iptr2);
}



/**
 * Take the first complex array to the power of the second complex array and save to output (with strides)
 *
 * optr = iptr1 ^ iptr2
 */
void md_zpow2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
#ifdef USE_CUDA
	// FIXME: something is broken with the cuda implementation of zpow
	assert(!(cuda_ondevice(optr) || cuda_ondevice(iptr1) || cuda_ondevice(iptr2)));
#endif
	MAKE_Z3OP(zpow, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Take the first complex array to the power of the second complex array and save to output (without strides)
 *
 * optr = iptr1 ^ iptr2
 */
void md_zpow(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zpow2, D, dims, optr, iptr1, iptr2);
}



/**
 * Take the first scalar array to the power of the second scalar array and save to output (with strides)
 *
 * optr = iptr1 ^ iptr2
 */
void md_pow2(unsigned int D, const long dims[D], const long ostr[D],  float* optr, const long istr1[D], const  float* iptr1, const long istr2[D], const  float* iptr2)
{
	MAKE_3OP(pow, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Take the first scalar array to the power of the second scalar array and save to output (without strides)
 *
 * optr = iptr1 ^ iptr2
 */
void md_pow(unsigned int D, const long dims[D],  float* optr, const  float* iptr1, const float* iptr2)
{
	make_3op_simple(md_pow2, D, dims, optr, iptr1, iptr2);
}



/**
 * Take square root of scalar array and save to output (with strides)
 *
 * optr = sqrt(iptr)
 */
void md_sqrt2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr[D], const float* iptr)
{
	MAKE_2OP(sqrt, D, dims, ostr, optr, istr, iptr);
}



/**
 * Take square root of scalar array and save to output (without strides)
 *
 * optr = sqrt(iptr)
 */
void md_sqrt(unsigned int D, const long dims[D], float* optr, const float* iptr)
{
	make_2op_simple(md_sqrt2, D, dims, optr, iptr);
}



/**
 * Take square root of complex array and save to output (with strides)
 *
 * optr = sqrt(iptr)
 */
void md_zsqrt2(unsigned int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	md_zspow2(D, dims, ostrs, optr, istrs, iptr, 0.5);
}



/**
 * Take square root of complex array and save to output (without strides)
 *
 * optr = sqrt(iptr)
 */
void md_zsqrt(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zsqrt2, D, dims, optr, iptr);
}



/**
 * Raise complex array to the power of a scalar and save to output (without strides)
 *
 * optr = pow(iptr, scalar)
 */
void md_zspow(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr, complex float val)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zspow2(D, dims, strs, optr, strs, iptr, val);
}



/**
 * Raise complex array to the power of a scalar and save to output (with strides)
 *
 * optr = pow(iptr, scalar)
 */
void md_zspow2(unsigned int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr, complex float val)
{
	make_z3op_scalar(md_zpow2, D, dims, ostrs, optr, istrs, iptr, val);
}




/**
 * Convert float array to double array
 *
 * dst = (double)src
 */
void md_float2double2(unsigned int D, const long dims[D], const long ostr[D], double* dst, const long istr[D], const float* src)
{
	MAKE_2OPD(float2double, D, dims, ostr, dst, istr, src);
}



/**
 * Convert float array to double array
 *
 * dst = (double)src
 */
void md_float2double(unsigned int D, const long dims[D], double* dst, const float* src)
{
	make_2opd_simple(md_float2double2, D, dims, dst, src);
}



/**
 * Convert double array to float array
 *
 * dst = (double)src
 */
void md_double2float2(unsigned int D, const long dims[D], const long ostr[D], float* dst, const long istr[D], const double* src)
{
	MAKE_2OPF(double2float, D, dims, ostr, dst, istr, src);
}



/**
 * Convert double array to float array
 *
 * dst = (float)src
 */
void md_double2float(unsigned int D, const long dims[D],  float* dst, const double* src)
{
	make_2opf_simple(md_double2float2, D, dims, dst, src);
}



/**
 * Convert complex float array to complex double array
 *
 * dst = (complex double)src
 */
void md_zdouble2float2(unsigned int D, const long dims[D], const long ostr[D], complex float* dst, const long istr[D], const complex double* src)
{
	MAKE_Z2OPF_FROM_REAL(double2float, D, dims, ostr, dst, istr, src);
}



/**
 * Convert complex float array to complex double array
 *
 * dst = (complex double)src
 */
void md_zdouble2float(unsigned int D, const long dims[D], complex float* dst, const complex double* src)
{
	make_z2opf_simple(md_zdouble2float2, D, dims, dst, src);
}



/**
 * Convert complex double array to complex float array
 *
 * dst = (complex float)src
 */
void md_zfloat2double2(unsigned int D, const long dims[D], const long ostr[D], complex double* dst, const long istr[D], const complex float* src)
{
	MAKE_Z2OPD_FROM_REAL(float2double, D, dims, ostr, dst, istr, src);
}



/**
 * Convert complex double array to complex float array
 *
 * dst = (complex float)src
 */
void md_zfloat2double(unsigned int D, const long dims[D], complex double* dst, const complex float* src)
{
	make_z2opd_simple(md_zfloat2double2, D, dims, dst, src);
}


/*
 * A A A ok
 * A A 1 ok
 * A 1 A ok
 * 1 A A ok
 * A 1 1 !
 * 1 A 1 !
 * 1 1 A !
 * 1 1 1 ok
 */
void md_tenmul_dims(unsigned int D, long max_dims[D], const long out_dims[D], const long in1_dims[D], const long in2_dims[D])
{
	md_max_dims(D, ~0lu, max_dims, in1_dims, out_dims);

	long max2_dims[D];
	md_max_dims(D, ~0lu, max2_dims, in2_dims, out_dims);

	assert(md_check_compat(D, 0lu, max_dims, max2_dims));
}


static bool detect_matrix(const long dims[3], const long ostrs[3], const long mstrs[3], const long istrs[3])
{
	return (   (0 == ostrs[1])
		&& (0 == mstrs[2])
		&& (0 == istrs[0])
		&& ((CFL_SIZE == ostrs[0]) && (ostrs[0] * dims[0] == ostrs[2]))
		&& ((CFL_SIZE == mstrs[0]) && (mstrs[0] * dims[0] == mstrs[1]))
		&& ((CFL_SIZE == istrs[1]) && (istrs[1] * dims[1] == istrs[2])));
}


static bool simple_matmul(unsigned int N, const long max_dims[N], const long ostrs[N], complex float* out,
		const long mstrs[N], const complex float* mat, const long istrs[N], const complex float* in)
{
	long dims[N];
	md_copy_dims(N, dims, max_dims);

	long ostrs2[N];
	md_copy_strides(N, ostrs2, ostrs);

	long mstrs2[N];
	md_copy_strides(N, mstrs2, mstrs);

	long istrs2[N];
	md_copy_strides(N, istrs2, istrs);

	long (*strs[3])[N] = { &ostrs2, &istrs2, &mstrs2 };
	unsigned int ND = simplify_dims(3, N, dims, strs);

	if (ND < 3)
		return false;

	long C = dims[0];
	long B = dims[1];
	long A = dims[2];

	if ((3 == ND) && detect_matrix(dims, ostrs2, istrs2, mstrs2)) {

		debug_printf(DP_DEBUG4, "matmul: matrix multiplication (1).\n");
#if 0
		// num/linalg.h

		mat_mul(A, B, C,
			*(complex float (*)[A][C])out,
			*(const complex float (*)[A][B])mat,
			*(const complex float (*)[B][C])in);
#else
		blas_matrix_multiply(C, A, B,
			*(complex float (*)[A][C])out,
			*(const complex float (*)[B][C])in,
			*(const complex float (*)[A][B])mat);
#endif
		return true;
	}

	if ((3 == ND) && detect_matrix(dims, ostrs2, mstrs2, istrs2)) {

		debug_printf(DP_DEBUG4, "matmul: matrix multiplication (2).\n");
#if 0
		// num/linalg.h

		mat_mul(A, B, C,
			*(complex float (*)[A][C])out,
			*(const complex float (*)[A][B])in,
			*(const complex float (*)[B][C])mat);
#else
		blas_matrix_multiply(C, A, B,
			*(complex float (*)[A][C])out,
			*(const complex float (*)[B][C])mat,
			*(const complex float (*)[A][B])in);
#endif
		return true;
	}

	return false;
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
 * 1.) Simplify dims
 * 2.) Order dims/strs of first three dims to form a matmul
 * 3.) Return nr of new dims if succesful and 0 else
 * if succesful, the out strides have the form:
 * nostrs:(size, 0, x)
 * the in strides have the form
 * nistrs1: (0, x, x)
 * nistrs2: (x, x, 0)
 * or
 * nistrs1: (x, x, 0)
 * nistrs2: (0, x, x)
 */
static int make_matrix(unsigned long N, long ndims[N], long nostrs[N], long nistrs1[N], long nistrs2[N], const long dims[N], const long ostrs[N], const long istrs1[N], const long istrs2[N], size_t size)
{
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
		return 0;

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
	matrix &= is_matrix(tdims, tostrs, (opos + 1) % 3, (opos + 2) % 3, size);
	matrix &= is_matrix(tdims, tistrs1, (ipos1 + 1) % 3, (ipos1 + 2) % 3, size);
	matrix &= is_matrix(tdims, tistrs2, (ipos2 + 1) % 3, (ipos2 + 2) % 3, size);

	if (!matrix)
		return 0;

	/*
	 * Permute dims such that strides of output have the form
	 * (size, 0, x)
	 * the in strides have the form
	 * (0, x, x)
	 * (x, x, 0)
	 * or
	 * (x, x, 0)
	 * (0, x, x)
	 */
	unsigned int perm[N];

	for (unsigned int i = 0; i < N; i++)
		perm[i] = i;

	perm[1] = opos;

	if (tostrs[(opos + 1) % 3] == (signed)size){

		perm[0] = (opos + 1) % 3;
		perm[2] = (opos + 2) % 3;

	} else {

		perm[0] = (opos + 2) % 3;
		perm[2] = (opos + 1) % 3;
	}

	md_permute_dims(N, perm, ndims, tdims);
	md_permute_dims(N, perm, nostrs, tostrs);
	md_permute_dims(N, perm, nistrs1, tistrs1);
	md_permute_dims(N, perm, nistrs2, tistrs2);

	return N;
}

static bool simple_inner_matmul_zfmac(unsigned int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	size_t size = CFL_SIZE;
	long ndims[N];
	long nostrs[N];
	long tistrs1[N];
	long tistrs2[N];

	N = make_matrix(N, ndims, nostrs, tistrs1, tistrs2, dims, ostrs, istrs1, istrs2, size);

	if (N < 3)
		return false;

	/**
	 * Permute inputs if necessary, result:
	 * nostrs:(size, 0, x)
	 * nistrs1: (0, x, x)
	 * nistrs2: (x, x, 0)
	 */
	bool perm = (0 == tistrs1[0]);

	long* nistrs1 = (perm ? tistrs2 : tistrs1);
	long* nistrs2 = (perm ? tistrs1 : tistrs2);

	const complex float* nin1 = (perm ? in2 : in1);
	const complex float* nin2 = (perm ? in1 : in2);

	//Check if matrices are transposed in storage
	char i1_trans = (nistrs1[0] == (signed)size) ? 'N' : 'T';
	char i2_trans = (nistrs2[1] == (signed)size) ? 'N' : 'T';

#if 0
	// saved code for zfmacc

	if (perm){

		if (i1_trans != 'T')
			return false;

		i1_trans = 'C';

	} else {

		if (i2_trans != 'T')
			return false;

		i2_trans = 'C';
	}
#endif

	long M1 = ndims[0];
	long K1 = ndims[1];
	long N1 = ndims[2];

	NESTED(void, nary_inner_matmul, (struct nary_opt_data_s* data, void* ptr[]))
	{
		(void)data;

		blas_matrix_zfmac(M1, N1, K1,
			(complex float*)ptr[0],
			(const complex float*)ptr[1], i1_trans,
			(const complex float*)ptr[2], i2_trans);
	};

	optimized_threeop_oii(N - 3, ndims + 3, nostrs + 3, (void*)out, nistrs1 + 3, (void*)nin1, nistrs2 + 3, (void*)nin2,
				(size_t[3]){ size * M1 * N1, size * M1 * K1, size * N1 * K1 }, nary_inner_matmul);

	return true;
}


/*
 * tenmul (tensor multiplication) family of functions are revised
 * versions of the matmul functions.
 */
void md_ztenmul2(unsigned int D, const long max_dims[D], const long out_strs[D], complex float* out, const long in1_strs[D], const complex float* in1, const long in2_strs[D], const complex float* in2)
{
	if (simple_matmul(D, max_dims, out_strs, out, in2_strs, in2, in1_strs, in1))
		return;

	md_clear2(D, max_dims, out_strs, out, CFL_SIZE);
	md_zfmac2(D, max_dims, out_strs, out, in1_strs, in1, in2_strs, in2);
}


void md_ztenmulc2(unsigned int D, const long max_dims[D], const long out_strs[D], complex float* out, const long in1_strs[D], const complex float* in1, const long in2_strs[D], const complex float* in2)
{
	md_clear2(D, max_dims, out_strs, out, CFL_SIZE);
	md_zfmacc2(D, max_dims, out_strs, out, in1_strs, in1, in2_strs, in2);
}


void md_ztenmul(unsigned int D, const long out_dims[D], complex float* out, const long in1_dims[D], const complex float* in1, const long in2_dims[D], const complex float* in2)
{
	long max_dims[D];
	md_tenmul_dims(D, max_dims, out_dims, in1_dims, in2_dims);

	md_ztenmul2(D, max_dims, MD_STRIDES(D, out_dims, CFL_SIZE), out,
				 MD_STRIDES(D, in1_dims, CFL_SIZE), in1,
				 MD_STRIDES(D, in2_dims, CFL_SIZE), in2);
}


void md_ztenmulc(unsigned int D, const long out_dims[D], complex float* out, const long in1_dims[D], const complex float* in1, const long in2_dims[D], const complex float* in2)
{
	long max_dims[D];
	md_tenmul_dims(D, max_dims, out_dims, in1_dims, in2_dims);

	md_ztenmulc2(D, max_dims, MD_STRIDES(D, out_dims, CFL_SIZE), out,
				  MD_STRIDES(D, in1_dims, CFL_SIZE), in1,
				  MD_STRIDES(D, in2_dims, CFL_SIZE), in2);
}



/**
 * Write strides for conv or corr into mdims, ostrs2, kstrs2 and istrs2
 * Supports "strides" and dilation
 * The flag conv decides if conv or corr
 * The flag test_mode turns of all assertions for detecting strides
 **/
static int calc_convcorr_geom_strs_dil(int N, unsigned long flags,
				       long mdims[2 * N], long ostrs2[2 * N], long kstrs2[2 * N], long istrs2[2 * N],
				       const long odims[N], const long ostrs[N], const long kdims[N], const long kstrs[N], const long idims[N], const long istrs[N],
				       const long dilation[N], const long strides[N], bool conv, bool test_mode)
 {
 	int shift = 0;

	md_copy_strides(N, ostrs2, ostrs);
	md_singleton_strides(N, ostrs2 + N);

	md_copy_strides(N, kstrs2, kstrs);
	md_singleton_strides(N, kstrs2 + N);

	md_copy_strides(N, istrs2, istrs);
	md_singleton_strides(N, istrs2 + N);

	md_copy_dims(N, mdims, odims);
	md_singleton_dims(N, mdims + N);

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			if (!test_mode)
				assert(idims[i] == strides[i] * odims[i] + (kdims[i] - 1) * dilation[i]);

			mdims[0 + i] = odims[i];
			mdims[N + i] = kdims[i];

			kstrs2[0 + i] = 0;
			kstrs2[N + i] = (conv ? -kstrs[i] : kstrs[i]);

			if (conv)
				shift += (kdims[i] - 1) * kstrs[i];

			istrs2[0 + i] = strides[i] * istrs[i];
			istrs2[N + i] = dilation[i] * istrs[i];

		} else {

			if (1 == mdims[i])
				mdims[i] = idims[i];

			if (1 == mdims[i])
				mdims[i] = kdims[i];

			if (1 == mdims[i])
				mdims[i] = odims[i];

			if (!test_mode) {

				assert((1 == odims[i]) || (odims[i] == idims[i]) || (odims[i] == kdims[i]));
				assert((1 == idims[i]) || (odims[i] == idims[i]) || (idims[i] == kdims[i]));
				assert((1 == kdims[i]) || (kdims[i] == idims[i]) || (odims[i] == kdims[i]));
			}
		}
	}

	return shift;
}

int calc_convcorr_geom(int N, unsigned long flags,
		       long mdims[2 * N], long ostrs2[2 * N], long kstrs2[2 * N], long istrs2[2 * N],
		       const long odims[N], const long ostrs[N], const long kdims[N], const long kstrs[N], const long idims[N], const long istrs[N], bool conv)
{
	return calc_convcorr_geom_strs_dil(N, flags, mdims, ostrs2, kstrs2, istrs2, odims, ostrs, kdims, kstrs, idims, istrs, MD_SINGLETON_DIMS(N), MD_SINGLETON_DIMS(N), conv, false);
}


void md_zconv2(int N, unsigned long flags,
	       const long odims[N], const long ostrs[N], complex float* out,
	       const long kdims[N], const long kstrs[N], const complex float* krn,
	       const long idims[N], const long istrs[N], const complex float* in)
{
	long mdims[2 * N];
	long ostrs2[2 * N];
	long kstrs2[2 * N];
	long istrs2[2 * N];

	krn += calc_convcorr_geom(N, flags, mdims, ostrs2, kstrs2, istrs2,
				  odims, ostrs, kdims, kstrs, idims, istrs, true) / CFL_SIZE;

	md_ztenmul2(2 * N, mdims, ostrs2, out, kstrs2, krn, istrs2, in);
}

void md_zconv(int N, unsigned long flags,
	      const long odims[N], complex float* out,
	      const long kdims[N], const complex float* krn,
	      const long idims[N], const complex float* in)
{
	md_zconv2(N, flags, odims, MD_STRIDES(N, odims, CFL_SIZE), out, kdims, MD_STRIDES(N, kdims, CFL_SIZE), krn, idims, MD_STRIDES(N, idims, CFL_SIZE), in);
}


/**
 * Detect if strides and dims belong to convolution with specified flags
 **/
static bool detect_convcorr(int N, long nodims[N], long nidims[N], long nkdims[N],
			    const long dims[2 * N], const long ostrs[2 * N], const long istrs[2 * N], const long kstrs[2 * N],
			    unsigned long flags, bool conv, size_t size)
{
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)){

			nodims[i] = dims[0 + i];
			nkdims[i] = dims[N + i];
			nidims[i] = nodims[i] + nkdims[i] - 1;

		} else {

			nodims[i] = (ostrs[i] == 0) ? 1 : dims[i];
			nidims[i] = (istrs[i] == 0) ? 1 : dims[i];
			nkdims[i] = (kstrs[i] == 0) ? 1 : dims[i];
		}
	}

	long tdims[2 * N];
	long tostrs[2 * N];
	long tistrs[2 * N];
	long tkstrs[2 * N];

	calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs, nodims, MD_STRIDES(N, nodims, size), nkdims, MD_STRIDES(N, nkdims, size), nidims, MD_STRIDES(N, nidims, size), MD_SINGLETON_DIMS(N), MD_SINGLETON_DIMS(N), conv, true);

	for (int i = 0; i < 2 * N; i++) {

		if (tdims[i] != dims[i])
			return false;

		if (tostrs[i] != ostrs[i])
			return false;

		if (tistrs[i] != istrs[i])
			return false;

		if (tkstrs[i] != kstrs[i])
			return false;
	}

	return true;
}

static bool simple_zconvcorr_im2col_channelfirst(unsigned int N, const long dims[N], const long ostrs[N], complex float* out,
		const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
#ifdef USE_CUDA
	// Fallback: not tested on GPU
	return false;
#endif
	size_t size = CFL_SIZE;
	unsigned long flags = 28;

	if (10 != N)
		return false;

	N = N / 2;

	long odims[5];
	long idims[5];
	long kdims[5];

	/*
	 * istrs1 = istrs
	 * istrs2 = kstrs
	 * */
	bool conv = detect_convcorr(5, odims, idims, kdims, dims, ostrs, istrs1, istrs2, flags, true, size);

	if (!detect_convcorr(5, odims, idims, kdims, dims, ostrs, istrs1, istrs2, flags, conv, size))
		return false;

	//Check matmul in first two dims
	if ((1 != idims[0]) || (1 != odims[1]))
		return false;

	long corr_strs[5];
	md_calc_strides(5, corr_strs, kdims, size);

	for (int i = 0; i < 5; i++)
		if (MD_IS_SET(flags, i))
			corr_strs[i] = -corr_strs[i];

	const complex float* krn;

	if (conv) {

		complex float* tmp = md_alloc_sameplace(5, kdims, size, in2);

		md_copy2(5, kdims, MD_STRIDES(5, kdims, size), tmp, corr_strs, in2, size);

		krn = tmp;

	} else {

		krn = in2;
	}

	long odims_tile[5];
    	md_copy_dims(5, odims_tile, odims);

#if 1
	// compute tile size, else full tiles are used
	long cache_size = 256 * 800 / size;
	long filter_size = md_calc_size(5, kdims);
	bool stop = false;

	while (!stop) {

		long cost = filter_size;

		cost += odims[0] * odims_tile[2] * odims_tile[3] * odims_tile[4];
		cost += idims[1] * (odims_tile[2] + kdims[2] - 1)
				 * (odims_tile[3] + kdims[3] - 1)
				 * (odims_tile[4] + kdims[4] - 1);

		if (cost < cache_size) {

			stop = true;
			continue;
		}

		if (odims_tile[4] > 4 * kdims[4]) {

			odims_tile[4] /= 2;
			continue;
		}

		if (odims_tile[3] > 4 * kdims[3]) {

			odims_tile[3] /= 2;
			continue;
		}

		if (odims_tile[2] > 4 * kdims[2]) {

			odims_tile[2] /= 2;
			continue;
		}

		stop = true;

		debug_printf(DP_DEBUG2, "Failed to get cache size, output tiles:\n");
        	debug_print_dims(DP_DEBUG2, 5, odims_tile);
	}
#endif

	long tiles[5] = { 1, 1, 1, 1, 1 };
	long N_tiles = 1;

	long odims_last_tile[5];
	md_copy_dims(5, odims_last_tile, odims);

	long idims_tile[5];
	long idims_last_tile[5];
	md_copy_dims(5, idims_tile, idims);
	md_copy_dims(5, idims_last_tile, idims);

	for (int i = 2; i < 5; i++) {

		tiles[i] = odims[i] / odims_tile[i];
        	odims_last_tile[i] = odims[i] % odims_tile[i];

        	if (0 != odims_last_tile[i])
            		tiles[i] += 1;
        	else
            		odims_last_tile[i] = odims_tile[i];

		idims_tile[i] = odims_tile[i] + kdims[i] - 1;
       		idims_last_tile[i] = odims_last_tile[i] + kdims[i] - 1;

		N_tiles *= tiles[i];
	}

	long kdims_mat[8]; // (nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1)
	md_singleton_dims(8, kdims_mat);
	md_copy_dims(5, kdims_mat, kdims);

	long idims_mat[8]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz)
	md_copy_dims(2, idims_mat, idims_tile);
	md_copy_dims(3, idims_mat + 2, kdims + 2);
	md_copy_dims(3, idims_mat + 5, odims_tile + 2);

	long idims_last_mat[8];
	md_copy_dims(2, idims_last_mat, idims_last_tile);
	md_copy_dims(3, idims_last_mat + 2, kdims + 2);
	md_copy_dims(3, idims_last_mat + 5, odims_last_tile + 2);

	long istrs_mat[8];
	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, size));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, size)+ 2);

	long odims_mat[8]; // (nr_filter | 1, 1, 1, 1 | outx, outy, outz)
	md_select_dims(8, ~31, odims_mat, idims_mat);
	odims_mat[0] = kdims_mat[0];

	long odims_last_mat[8];
	md_select_dims(8, ~31, odims_last_mat, idims_last_mat);
	odims_last_mat[0] = kdims_mat[0];

	complex float* image_mat = md_alloc_sameplace(8, idims_mat, size, krn);
	complex float* out_mat = md_alloc_sameplace(8, odims_mat, size, krn);

	long mdims_tmp[8];
	long idims_mat_tmp[8];
	long odims_mat_tmp[8];
        long odims_tmp[5];

	long pos[] = { 0, 0, 0, 0, 0 };

	for (int i = 0; i < N_tiles; i++) {

        	int it = i;

		for (int j = 2; j < 5; j++) {

			pos[j] = (it % tiles[j]);
			it = (it - pos[j]) / tiles[j];
			pos[j] *= odims_tile[j];
        	}

		md_copy_dims(8, idims_mat_tmp, idims_mat);
		md_copy_dims(8, odims_mat_tmp, odims_mat);
        	md_copy_dims(5, odims_tmp, odims_tile);

		for (int j = 0; j < 3; j++) {

			if (pos[j + 2] == (tiles[j + 2] - 1) * odims_tile[j + 2]) {

				idims_mat_tmp[j + 5] = idims_last_mat[j + 5];
				odims_mat_tmp[j + 5] = odims_last_mat[j + 5];
                		odims_tmp[j + 2] = odims_last_tile[j + 2];
			}
		}

		md_max_dims(8, ~0, mdims_tmp, idims_mat_tmp, kdims_mat);

		md_copy2(8, idims_mat_tmp, MD_STRIDES(8, idims_mat_tmp, size), image_mat, istrs_mat, &MD_ACCESS(5, MD_STRIDES(5, idims, size), pos, in1), size);
		md_clear(8, odims_mat_tmp, out_mat, size);
		md_zfmac2(8, mdims_tmp, MD_STRIDES(8, odims_mat_tmp, size), out_mat, MD_STRIDES(8, kdims_mat, size), krn, MD_STRIDES(8, idims_mat_tmp, size), image_mat);
		md_copy2(5, odims_tmp, MD_STRIDES(5, odims, size), &MD_ACCESS(5, MD_STRIDES(5, odims, size), pos, out), MD_STRIDES(5, odims_tmp, size), out_mat, size);
	}

	md_free(image_mat);
	md_free(out_mat);

	if (conv)
		md_free(krn);

	return true;
}


static bool simple_zconvcorr_im2col_channelfirst_batch(unsigned int N, const long dims[N],
							const long ostrs[N], complex float* out,
							const long istrs1[N], const complex float* in1,
							const long istrs2[N], const complex float* in2)
{
#ifdef USE_CUDA
	// Not tested on GPU
	return false;
#endif
	size_t size = CFL_SIZE;
	unsigned long flags = 28;

	if (12 != N)
		return false;

	long odims[6];
	long idims[6];
	long kdims[6];

	/**
	 * Check if
	 * ostrs = ostrs of convcorr
	 * istrs1 = istrs of convcorr
	 * istrs2 = kstrs of convcorr
	 **/
	bool conv = detect_convcorr(6, odims, idims, kdims, dims, ostrs, istrs1, istrs2, flags, true, size);

	if (!detect_convcorr(6, odims, idims, kdims, dims, ostrs, istrs1, istrs2, flags, conv, size))
		return false;

	//Check if matmul in first two dims and batch in last dimension
	if ((1 != idims[0]) || (1 != odims[1]) || (1 != kdims[5]))
		return false;

	const complex float* krn;

	if (conv) {

		long kstrs_flip[6];
		md_calc_strides(6, kstrs_flip, kdims, size);

		for (int i = 0; i < 6; i++)
			kstrs_flip[i] = MD_IS_SET(flags, i) ? -kstrs_flip[i] : kstrs_flip[i];

		complex float* tmp = md_alloc_sameplace(6, kdims, size, in2);

		md_copy2(6, kdims, MD_STRIDES(6, kdims, size), tmp, kstrs_flip, in2, size);

		krn = tmp;

	} else {

		krn = in2;
	}

	long mdimsb[10];
	long ostrs2b[10];
	long kstrs2b[10];
	long istrs2b[10];

	calc_convcorr_geom(5, flags, mdimsb, ostrs2b, kstrs2b, istrs2b,
				odims, MD_STRIDES(5, odims, size),
				kdims, MD_STRIDES(5, kdims, size),
				idims, MD_STRIDES(5, idims, size), false);

	// only parallelize batch dims if present
	if (1 < idims[5]) {

		#pragma omp parallel for
		for (int i = 0; i < idims[5]; i++) {

			long pos[6] = { [5] = i };

		    	simple_zconvcorr_im2col_channelfirst(10, mdimsb, ostrs2b, &MD_ACCESS(6, MD_STRIDES(6, odims, size), pos, out),
			    					istrs2b, &MD_ACCESS(6, MD_STRIDES(6, idims, size), pos, in1),
								kstrs2b, krn);
	    	}

    	} else {

		simple_zconvcorr_im2col_channelfirst(10, mdimsb, ostrs2b, out, istrs2b, in1, kstrs2b, krn);
    	}

	if (conv)
		md_free(krn);

	return true;
}

static bool simple_zconvcorr_im2col_channelfirst_adjkrn(unsigned int N, const long dims[N], const long ostrs[N], complex float* out,
		const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
#ifdef USE_CUDA
	// Convolution Kernel only implemented for CPU
	return false;
#endif

	size_t size = CFL_SIZE;
	unsigned long flags = 28;

	if (12 != N)
		return false;

	long odims[6];
	long idims[6];
	long kdims[6];

	/**
	 * Backwardspath, i.e.
	 * ostrs = kstrs of convcorr
	 * istrs1 = istrs of convcorr
	 * istrs2 = ostrs of convcorr
	 **/
	bool conv = detect_convcorr(6, odims, idims, kdims, dims, istrs2, istrs1, ostrs, flags, true, size);

	if (!detect_convcorr(6, odims, idims, kdims, dims, istrs2, istrs1, ostrs, flags, conv, size))
		return false;

	// Check matmul in first two dims, batch last dimension
	if ((1 != idims[0]) || (1 != odims[1]) || (1 != kdims[5]))
		return false;

	long idims_mat[9];
	md_copy_dims(2, idims_mat, idims);
	md_copy_dims(3, idims_mat + 2, kdims + 2);
	md_copy_dims(4, idims_mat + 5, odims + 2);

	long istrs_tmp[6];
	md_calc_strides(6, istrs_tmp, idims, size);

	long istrs_mat[9];
	md_copy_strides(5, istrs_mat, istrs_tmp);
	md_copy_strides(4, istrs_mat + 5, istrs_tmp + 2);

	long kdims_mat[9];
	md_singleton_dims(9, kdims_mat);
	md_copy_dims(5, kdims_mat, kdims);

	long odims_mat[9];
	md_select_dims(9, ~31, odims_mat, idims_mat);
	odims_mat[0] = kdims_mat[0];

	long mdims[9];
	md_max_dims(9, ~0, mdims, idims_mat, kdims_mat);


	complex float* krn; //i.e. the result

	if (conv) {

		krn = md_alloc_sameplace(6, kdims, size, out);
		md_clear(6, kdims, krn, size);

	} else {

		krn = out;
	}
#if 1
	complex float* image_mat = md_alloc_sameplace(8, idims_mat, size, in1);

	for (int i = 0; i < idims[5]; i++) {

		long pos[] = { [5] = i };

		md_copy2(8, idims_mat, MD_STRIDES(8, idims_mat, size), image_mat, istrs_mat, &MD_ACCESS(6, MD_STRIDES(6, idims, size), pos, in1), size);

		md_zfmac2(8, mdims, MD_STRIDES(8, kdims_mat, size), krn,
				    MD_STRIDES(8, odims_mat, size), &MD_ACCESS(6, MD_STRIDES(6, odims, size), pos, in2),
				    MD_STRIDES(8, idims_mat, size), image_mat);
	}
#else
	complex float* image_mat = md_alloc_sameplace(9, idims_mat, size, in1);
	md_copy2(9, idims_mat, MD_STRIDES(9, idims_mat, size), image_mat, istrs_mat, in1, size);
	md_zfmac2(9, mdims, MD_STRIDES(9, kdims_mat, size), krn, MD_STRIDES(9, odims_mat, size), in2, MD_STRIDES(9, idims_mat, size), image_mat);
#endif

	md_free(image_mat);

	if (conv) {

		long corr_strs[6];
		md_calc_strides(6, corr_strs, kdims, size);

		for (int i = 0; i < 6; i++)
			if (MD_IS_SET(flags, i))
				corr_strs[i] = -corr_strs[i];

		md_copy2(6, kdims, corr_strs, out, MD_STRIDES(6, kdims, size), krn, size);

		md_free(krn);
	}

	return true;
}

static bool simple_zconvcorr_im2col_channelfirst_adjim(unsigned int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
#ifdef USE_CUDA
	// Convolution Kernel only implemented for CPU
	return false;
#endif

	size_t size = CFL_SIZE;
	unsigned long flags = 28;

	if (12 != N)
		return false;

	long odims[6];
	long idims[6];
	long kdims[6];

	/**
	 * Transposed convolutio, i.e.
	 * ostrs = istrs of convcorr
	 * istrs1 = ostrs of convcorr
	 * istrs2 = kstrs of convcorr
	 **/

	bool conv = detect_convcorr(6, odims, idims, kdims, dims, istrs1, ostrs, istrs2, flags, true, size);

	if (!detect_convcorr(6, odims, idims, kdims, dims, istrs1, ostrs, istrs2, flags, conv, size))
		return false;

	//Check matmul in first two dims, batch last dimension
	if ((1 != idims[0]) || (1 != odims[1]) || (1 != kdims[5]))
		return false;

	const complex float* krn;

	if (conv) {

		long kstrs_flip[6];
		md_calc_strides(6, kstrs_flip, kdims, size);

		for (int i = 0; i < 6; i++)
			kstrs_flip[i] = MD_IS_SET(flags, i) ? -kstrs_flip[i] : kstrs_flip[i];

		complex float* tmp = md_alloc_sameplace(6, kdims, size, in2);

		md_copy2(6, kdims, MD_STRIDES(6, kdims, size), tmp, kstrs_flip, in2, size);

		krn = tmp;

	} else {

		krn = in2;
	}

	//transpose (matdims) and flip (convdims) kernel
	complex float* tmp = md_alloc_sameplace(6, kdims, size, krn);

	md_flip(6, kdims, flags, tmp, krn, size);

	if (conv)
		md_free(krn);

	long kdims_transp[6];
	md_transpose_dims(6, 0, 1, kdims_transp, kdims);

	complex float* krn_transp = md_alloc_sameplace(6, kdims_transp, size, krn);

	md_transpose(6, 0, 1, kdims_transp, krn_transp, kdims, tmp, size);

	md_free(tmp);

	long idims_transp[6];
	md_transpose_dims(6, 0, 1, idims_transp, idims);

	long odims_transp[6];
	md_transpose_dims(6, 0, 1, odims_transp, odims);

	long nodims_transp[6];
	md_copy_dims(6, nodims_transp, odims_transp);

	for (int i = 0; i < 6; i++)
		if MD_IS_SET(flags, i)
			nodims_transp[i] = idims_transp[i] + kdims_transp[i] - 1;

	long mdims[10];
	long nostrs2[10];
	long nkstrs2[10];
	long nistrs2[10];

	calc_convcorr_geom(5, flags, mdims, nostrs2, nkstrs2, nistrs2, idims_transp,
				MD_STRIDES(5, idims_transp, size), kdims_transp,
				MD_STRIDES(5, kdims_transp, size), nodims_transp,
				MD_STRIDES(5, nodims_transp, size), false);

	#pragma omp parallel for
	for (int j = 0; j < idims_transp[5]; j++) {

		long pos_batch[] = { [5] = j };

		//*in1 = (o0, o1, o2) -> *nin1 = (0, 0, 0, o1, o2, o3)
		complex float* nin1 = md_alloc_sameplace(5, nodims_transp, size, in1);

		long pos[5];
		for (int i = 0; i< 5; i++)
			pos[i] = MD_IS_SET(flags, i) ? (kdims_transp[i] - 1) : 0;

		md_clear(5, nodims_transp, nin1, size);
		md_copy_block(5, pos, nodims_transp, nin1, odims_transp, &MD_ACCESS(6, MD_STRIDES(6, odims_transp, size), pos_batch, in1), size);

		simple_zconvcorr_im2col_channelfirst(10, mdims,
							nostrs2, &MD_ACCESS(6, MD_STRIDES(6, idims_transp, size), pos_batch, out),
							nistrs2, nin1, nkstrs2, krn_transp);

		md_free(nin1);
	}

	md_free(krn_transp);

	return true;
}

static bool simple_zconvcorr_im2col(unsigned int N, const long dims[N], const long ostrs[N], complex float* out, const long istrs1[N], const complex float* in1, const long istrs2[N], const complex float* in2)
{
	if (simple_zconvcorr_im2col_channelfirst(N, dims, ostrs, out, istrs1, in1, istrs2, in2))
		return true;

	if (simple_zconvcorr_im2col_channelfirst(N, dims, ostrs, out, istrs2, in2, istrs1, in1))
		return true;


	if (simple_zconvcorr_im2col_channelfirst_batch(N, dims, ostrs, out, istrs1, in1, istrs2, in2))
		return true;

	if (simple_zconvcorr_im2col_channelfirst_batch(N, dims, ostrs, out, istrs2, in2, istrs1, in1))
		return true;


	if (simple_zconvcorr_im2col_channelfirst_adjim(N, dims, ostrs, out, istrs1, in1, istrs2, in2))
		return true;

	if (simple_zconvcorr_im2col_channelfirst_adjim(N, dims, ostrs, out, istrs2, in2, istrs1, in1))
		return true;


	if (simple_zconvcorr_im2col_channelfirst_adjkrn(N, dims, ostrs, out, istrs1, in1, istrs2, in2))
		return true;

	if (simple_zconvcorr_im2col_channelfirst_adjkrn(N, dims, ostrs, out, istrs2, in2, istrs1, in1))
		return true;

	return false;
}

static bool simple_zconvcorr(unsigned int N, const long dims[N], const long ostrs[N], complex float* out,
		const long istrs[N], const complex float* image, const long kstrs[N], const complex float* kernel)
{
	if (simple_zconvcorr_im2col(N, dims, ostrs, out, istrs, image, kstrs, kernel))
		return true;

	return false;
}


/*
 * matmul family of functions is deprecated - use tenmul instead
 */
static void md_zmatmul2_priv(unsigned int D, const long out_dims[D], const long out_strs[D], complex float* dst, const long mat_dims[D], const long mat_strs[D], const complex float* mat, const long in_dims[D], const long in_strs[D], const complex float* src, bool conj)
{
	long max_dims[D];
	md_tenmul_dims(D, max_dims, out_dims, mat_dims, in_dims);

	if ((!conj) && simple_matmul(D, max_dims, out_strs, dst, mat_strs, mat, in_strs, src))
		return;

	md_clear2(D, out_dims, out_strs, dst, CFL_SIZE);
	(conj ? md_zfmacc2 : md_zfmac2)(D, max_dims, out_strs, dst, in_strs, src, mat_strs, mat);
}

/**
 * Matrix conjugate multiplication (with strides)
 * FIXME simplify interface? use macros?
 */
void md_zmatmulc2(unsigned int D, const long out_dims[D], const long out_strs[D], complex float* dst, const long mat_dims[D], const long mat_strs[D], const complex float* mat, const long in_dims[D], const long in_strs[D], const complex float* src)
{
	md_zmatmul2_priv(D, out_dims, out_strs, dst, mat_dims, mat_strs, mat, in_dims, in_strs, src, true);
}



/**
 * Matrix conjugate multiplication (without strides)
 */
void md_zmatmulc(unsigned int D, const long out_dims[D], complex float* dst, const long mat_dims[D], const complex float* mat, const long in_dims[D], const complex float* src)
{
	md_zmatmulc2(D, out_dims, MD_STRIDES(D, out_dims, CFL_SIZE), dst,
			mat_dims, MD_STRIDES(D, mat_dims, CFL_SIZE), mat,
			in_dims, MD_STRIDES(D, in_dims, CFL_SIZE), src);
}



/**
 * Matrix multiplication (with strides)
 * FIXME simplify interface?
 * FIXME: implementation assumes strides == 0 for dims == 1
 */
void md_zmatmul2(unsigned int D, const long out_dims[D], const long out_strs[D], complex float* dst, const long mat_dims[D], const long mat_strs[D], const complex float* mat, const long in_dims[D], const long in_strs[D], const complex float* src)
{
	md_zmatmul2_priv(D, out_dims, out_strs, dst, mat_dims, mat_strs, mat, in_dims, in_strs, src, false);
}



/**
 * Matrix multiplication (without strides)
 */
void md_zmatmul(unsigned int D, const long out_dims[D], complex float* dst, const long mat_dims[D], const complex float* mat, const long in_dims[D], const complex float* src)
{
	md_zmatmul2(D,	out_dims, MD_STRIDES(D, out_dims, CFL_SIZE), dst,
			mat_dims, MD_STRIDES(D, mat_dims, CFL_SIZE), mat,
			in_dims, MD_STRIDES(D, in_dims, CFL_SIZE), src);
}



/**
 * Multiply two complex arrays and add to output (with strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_zfmac2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	if (simple_zconvcorr(D, dims, ostr, optr, istr1, iptr1, istr2, iptr2))
		return;

	if (simple_inner_matmul_zfmac(D, dims, ostr, optr, istr1, iptr1, istr2, iptr2))
		return;

	MAKE_Z3OP(zfmac, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply two complex arrays and add to output (without strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_zfmac(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zfmac2, D, dims, optr, iptr1, iptr2);
}



/**
 * Multiply two complex arrays and add to output (with strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_zfmacD2(unsigned int D, const long dims[D], const long ostr[D], complex double* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OPD(zfmac2, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply two complex arrays and add to output (without strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_zfmacD(unsigned int D, const long dims[D], complex double* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3opd_simple(md_zfmacD2, D, dims, optr, iptr1, iptr2);
}



/**
 * Multiply two scalar arrays and add to output (with strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_fmac2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(fmac, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply two scalar arrays and add to output (without strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_fmac(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	make_3op_simple(md_fmac2, D, dims, optr, iptr1, iptr2);
}



/**
 * Multiply two scalar arrays and add to output (with strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_fmacD2(unsigned int D, const long dims[D], const long ostr[D], double* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OPD(fmac2, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply two scalar arrays and add to output (without strides)
 *
 * optr = optr + iptr1 * iptr2
 */
void md_fmacD(unsigned int D, const long dims[D], double* optr, const float* iptr1, const float* iptr2)
{
	make_3opd_simple(md_fmacD2, D, dims, optr, iptr1, iptr2);
}



/**
 * Multiply the first complex array with the conjugate of the second complex array and add to output (with strides)
 *
 * optr = optr + iptr1 * conj(iptr2)
 */
void md_zfmacc2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP(zfmacc, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply the first complex array with the conjugate of the second complex array and add to output (without strides)
 *
 * optr = optr + iptr1 * conj(iptr2)
 */
void md_zfmacc(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zfmacc2, D, dims, optr, iptr1, iptr2);
}




/**
 * Multiply the first complex array with the conjugate of the second complex array and add to output (with strides)
 *
 * optr = optr + iptr1 * conj(iptr2)
 */
void md_zfmaccD2(unsigned int D, const long dims[D], const long ostr[D], complex double* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OPD(zfmacc2, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Multiply the first complex array with the conjugate of the second complex array and add to output (without strides)
 *
 * optr = optr + iptr1 * conj(iptr2)
 */
void md_zfmaccD(unsigned int D, const long dims[D], complex double* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3opd_simple(md_zfmaccD2, D, dims, optr, iptr1, iptr2);
}



/**
 * Multiply complex array with a scalar and add to output (with strides)
 *
 * optr = optr + iptr * val
 */
void md_zaxpy2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, complex float val, const long istr[D], const complex float* iptr)
{
#ifdef USE_CUDA
	// FIXME: faster on GPU
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, optr);
	md_zsmul2(D, dims, MD_STRIDES(D, dims, CFL_SIZE), tmp, istr, iptr, val);
	md_zadd2(D, dims, ostr, optr, ostr, optr, MD_STRIDES(D, dims, CFL_SIZE), tmp);
	md_free(tmp);
	return;
#endif

	if (0. == cimagf(val)) { // strength reduction: complex to real multiplication

		long dimsR[D + 1];
		long ostrR[D + 1];
		long istrR[D + 1];

		real_from_complex_dims(D, dimsR, dims);
		real_from_complex_strides(D, ostrR, ostr);
		real_from_complex_strides(D, istrR, istr);

		md_axpy2(D + 1, dimsR, ostrR, (float*)optr, crealf(val), istrR, (const float*)iptr);
		return;
	}

	make_z3op_scalar(md_zfmac2, D, dims, ostr, optr, istr, iptr, val);
}



/**
 * Max of inputs (without strides)
 *
 * optr = max(iptr1, iptr2)
 */
void md_max(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_max2(D, dims, strs, optr, strs, iptr1, strs, iptr2);
}


/**
 * Max of inputs (with strides)
 *
 * optr = max(iptr1, iptr2)
 */
void md_max2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(max, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Min of inputs (without strides)
 *
 * optr = min(iptr1, iptr2)
 */
void md_min(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_min2(D, dims, strs, optr, strs, iptr1, strs, iptr2);
}



/**
 * Min of inputs (with strides)
 *
 * optr = min(iptr1, iptr2)
 */
void md_min2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(min, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}


/**
 * Max of inputs (without strides)
 *
 * optr = max(iptr1, iptr2)
 */
void md_zmax(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zmax2(D, dims, strs, optr, strs, iptr1, strs, iptr2);
}


/**
 * Max of inputs (with strides)
 *
 * optr = max(iptr1, iptr2)
 */
void md_zmax2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP(zmax, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}




/**
 * Multiply complex array with a scalar and add to output (without strides)
 *
 * optr = optr + iptr * val
 */
void md_zaxpy(unsigned int D, const long dims[D], complex float* optr, complex float val, const complex float* iptr)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zaxpy2(D, dims, strs, optr, val, strs, iptr);
}



/**
 * Multiply scalar array with a scalar and add to output (with strides)
 *
 * optr = optr + iptr * val
 */
void md_axpy2(unsigned int D, const long dims[D], const long ostr[D], float* optr, float val, const long istr[D], const float* iptr)
{
	if (0. == val)
		return;

	// strength reduction
	if (1. == val) {

		md_add2(D, dims, ostr, optr, ostr, optr, istr, iptr);
		return;
	}

#ifdef USE_CUDA
	if (cuda_ondevice(iptr)) {

		assert(cuda_ondevice(optr));

		if (md_calc_blockdim(D, dims, ostr, FL_SIZE) != D)
			goto fallback;

		if (md_calc_blockdim(D, dims, istr, FL_SIZE) != D)
			goto fallback;

		//  (iptr == optr) is safe.

		gpu_ops.axpy(md_calc_size(D, dims), optr, val, iptr);
		return;
	}
fallback:
#endif
	make_3op_scalar(md_fmac2, D, dims, ostr, optr, istr, iptr, val);
}



/**
 * Multiply scalar array with a scalar and add to output (without strides)
 *
 * optr = optr + iptr * val
 */
void md_axpy(unsigned int D, const long dims[D], float* optr, float val, const float* iptr)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_axpy2(D, dims, strs, optr, val, strs, iptr);
}

static bool simple_zsum(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
#ifdef USE_CUDA
	return false;
#endif
	if (optr != iptr1)
		return false;

	long tostr[D];
	long tistr1[D];
	long tistr2[D];
	long tdims[D];

	md_copy_dims(D, tdims, dims);
	md_copy_strides(D, tostr, ostr);
	md_copy_strides(D, tistr1, istr1);
	md_copy_strides(D, tistr2, istr2);

	long (*strs[3])[D] = { &tostr, &tistr1, &tistr2 };
	D = simplify_dims(3, D, tdims, strs);

	//sum over outer dimension
	if ((2 == D) && (CFL_SIZE == tostr[0]) && (CFL_SIZE == tistr1[0]) && (CFL_SIZE == tistr2[0])
		     && (0 == tostr[1]) && (0 == tistr1[1]) && ((long)CFL_SIZE * tdims[0] == tistr2[1])) {

		for (int j = 0; j < tdims[1]; j++)
			for (int i = 0; i < tdims[0]; i++)
				optr[i] += iptr2[i + tdims[0] * j];

		return true;
	}

	return false;
}


/**
 * Add two complex arrays and save to output (with strides)
 *
 * optr = iptr1 + iptr2
 */
void md_zadd2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	if (simple_zsum(D, dims, ostr, optr, istr1, iptr1, istr2, iptr2))
		return;

	MAKE_Z3OP_FROM_REAL(add, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Add two complex arrays and save to output (without strides)
 *
 * optr = iptr1 + iptr2
 */
void md_zadd(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zadd2, D, dims, optr, iptr1, iptr2);
}



/**
 * Add scalar to complex array (with strides)
 *
 * optr = iptr + val
 */
void md_zsadd2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr, complex float val)
{
	make_z3op_scalar(md_zadd2, D, dims, ostr, optr, istr, iptr, val);
}



/**
 * Add scalar to complex array (without strides)
 *
 * optr = iptr + val
 */
void md_zsadd(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr, complex float val)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zsadd2(D, dims, strs, optr, strs, iptr, val);
}



/**
 * Subtract the first complex array from the second complex array and save to output (with strides)
 *
 * optr = iptr1 - iptr2
 */
void md_zsub2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP_FROM_REAL(sub, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Subtract the first complex array from the second complex array and save to output (without strides)
 *
 * optr = iptr1 - iptr2
 */
void md_zsub(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zsub2, D, dims, optr, iptr1, iptr2);
}



/**
 * Add two scalar arrays and save to output (with strides)
 *
 * optr = iptr1 + iptr2
 */
void md_add2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(add, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Add two scalar arrays and save to output (without strides)
 *
 * optr = iptr1 + iptr2
 */
void md_add(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	make_3op_simple(md_add2, D, dims, optr, iptr1, iptr2);
}



/**
 * Add scalar to scalar array (with strides)
 *
 * optr = iptr + val
 */
void md_sadd2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float val)
{
	make_3op_scalar(md_add2, D, dims, ostr, optr, istr, iptr, val);
}



/**
 * Add scalar to scalar array (without strides)
 *
 * optr = iptr + val
 */
void md_sadd(unsigned int D, const long dims[D], float* optr, const float* iptr, float val)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_sadd2(D, dims, strs, optr, strs, iptr, val);
}



/**
 * Subtract the first scalar array from the second scalar array and save to output (with strides)
 *
 * optr = iptr1 - iptr2
 */
void md_sub2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(sub, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Subtract the first scalar array from the second scalar array and save to output (without strides)
 *
 * optr = iptr1 - iptr2
 */
void md_sub(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	make_3op_simple(md_sub2, D, dims, optr, iptr1, iptr2);
}



/**
 * Take complex conjugate of complex array and save to output (with strides)
 *
 * optr = conj(iptr)
 */
void md_zconj2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zconj, D, dims, ostr, optr, istr, iptr);
}



/**
 * Take complex conjugate of complex array and save to output (without strides)
 *
 * optr = conj(iptr)
 */
void md_zconj(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zconj2, D, dims, optr, iptr);
}



/**
 * Take the real part of complex array and save to output (with strides)
 *
 * optr = real(iptr)
 */
void md_zreal2(unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	make_z3op_scalar(md_zrmul2, D, dim, ostr, optr, istr, iptr, 1.);
}



/**
 * Take the real part of complex array and save to output (without strides)
 *
 * optr = real(iptr)
 */
void md_zreal(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
#ifdef USE_CUDA
	if (cuda_ondevice(iptr)) {

		assert(cuda_ondevice(optr));

		cuda_zreal(md_calc_size(D, dims), optr, iptr);
		return;
	}
#endif
	make_z2op_simple(md_zreal2, D, dims, optr, iptr);
}



/**
 * Take the imaginary part of complex array and save to output (with strides)
 *
 * optr = imag(iptr)
 */
void md_zimag2(unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	make_z3op_scalar(md_zrmul2, D, dim, ostr, optr, istr, iptr, 1.i);
}



/**
 * Take the imaginary part of complex array and save to output (without strides)
 *
 * optr = imag(iptr)
 */
void md_zimag(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zimag2, D, dims, optr, iptr);
}



/**
 * Compare two complex arrays (with strides)
 *
 * optr = iptr1 == iptr2
 */
void md_zcmp2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP(zcmp, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Compare two complex arrays (without strides)
 *
 * optr = iptr1 == iptr2
 */
void md_zcmp(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zcmp2, D, dims, optr, iptr1, iptr2);
}


/**
 * Elementwise less than or equal to (with strides)
 *
 * optr = (iptr1 <= iptr2)
 */
void md_zlessequal2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	MAKE_Z3OP(zle, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Elementwise less than or equal to (without strides)
 *
 * optr = (iptr1 <= iptr2)
 */
void md_zlessequal(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zlessequal2, D, dims, optr, iptr1, iptr2);
}




/**
 * Elementwise less than or equal to (with strides)
 *
 * optr = (iptr1 <= iptr2)
 */
void md_lessequal2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	MAKE_3OP(le, D, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}



/**
 * Elementwise less than or equal to (without strides)
 *
 * optr = (iptr1 <= iptr2)
 */
void md_lessequal(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	make_3op_simple(md_lessequal2, D, dims, optr, iptr1, iptr2);
}



/**
 * Elementwise less than or equal to scalar (with strides)
 *
 * optr = (iptr <= val)
 */
void md_slessequal2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float val)
{
	make_3op_scalar(md_lessequal2, D, dims, ostr, optr, istr, iptr, val);
}



/**
 * Elementwise less than or equal to scalar (without strides)
 *
 * optr = (iptr <= val)
 */
void md_slessequal(unsigned int D, const long dims[D], float* optr, const float* iptr, float val)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_slessequal2(D, dims, strs, optr, strs, iptr, val);
}


/**
 * Elementwise greater than or equal to (with strides)
 *
 * optr = (iptr1 => iptr2)
 */
void md_zgreatequal2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	md_zlessequal2(D, dims, ostr, optr, istr2, iptr2, istr1, iptr1);
}



/**
 * Elementwise greater than or equal to (without strides)
 *
 * optr = (iptr1 >= iptr2)
 */
void md_zgreatequal(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	make_z3op_simple(md_zgreatequal2, D, dims, optr, iptr1, iptr2);
}



/**
 * Elementwise greater than or equal to (with strides)
 *
 * optr = (iptr1 => iptr2)
 */
void md_greatequal2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr1[D], const float* iptr1, const long istr2[D], const float* iptr2)
{
	md_lessequal2(D, dims, ostr, optr, istr2, iptr2, istr1, iptr1);
}



/**
 * Elementwise greater than or equal to (without strides)
 *
 * optr = (iptr1 >= iptr2)
 */
void md_greatequal(unsigned int D, const long dims[D], float* optr, const float* iptr1, const float* iptr2)
{
	make_3op_simple(md_greatequal2, D, dims, optr, iptr1, iptr2);
}



/**
 * Elementwise greater than or equal to scalar (with strides)
 *
 * optr = (iptr >= val)
 */
void md_sgreatequal2(unsigned int D, const long dims[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float val)
{
	make_3op_scalar(md_greatequal2, D, dims, ostr, optr, istr, iptr, val);
}



/**
 * Elementwise greater than or equal to scalar (without strides)
 *
 * optr = (iptr >= val)
 */
void md_sgreatequal(unsigned int D, const long dims[D], float* optr, const float* iptr, float val)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_sgreatequal2(D, dims, strs, optr, strs, iptr, val);
}



/**
 * Elementwise greater than or equal to scalar (with strides)
 *
 * optr = (iptr >= val)
 */
void md_zsgreatequal2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr, float val)
{
	make_z3op_scalar(md_zgreatequal2, D, dims, ostr, optr, istr, iptr, val);
}



/**
 * Elementwise greater than or equal to scalar (without strides)
 *
 * optr = (iptr >= val)
 */
void md_zsgreatequal(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr, float val)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zsgreatequal2(D, dims, strs, optr, strs, iptr, val);
}



/**
 * Extract unit-norm complex exponentials from complex arrays (with strides)
 *
 * optr = iptr / abs(iptr)
 */
void md_zphsr2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zphsr, D, dims, ostr, optr, istr, iptr);
}



/**
 * Extract unit-norm complex exponentials from complex arrays (without strides)
 *
 * optr = iptr / abs(iptr)
 */
void md_zphsr(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zphsr2, D, dims, optr, iptr);
}


/**
 * Get complex exponential with phase = complex arrays (with strides)
 *
 * optr = zexp(j * iptr)
 */
void md_zexpj2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zexpj, D, dims, ostr, optr, istr, iptr);
}



/**
 * Get complex exponential with phase = complex arrays (without strides)
 *
 * optr = zexp(j * iptr)
 */
void md_zexpj(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zexpj2, D, dims, optr, iptr);
}




/**
 * Complex exponential
 *
 * optr = zexp(iptr)
 */
void md_zexp2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zexp, D, dims, ostr, optr, istr, iptr);
}



/**
 * Complex exponential
 *
 * optr = zexp(iptr)
 */
void md_zexp(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zexp2, D, dims, optr, iptr);
}


/**
 * Complex logarithm
 *
 * optr = zlog(iptr)
 */
void md_zlog2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zlog, D, dims, ostr, optr, istr, iptr);
}

/**
 * Complex logarithm
 *
 * optr = zlog(iptr)
 */
void md_zlog(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zlog2, D, dims, optr, iptr);
}

/**
 * Get argument of complex arrays (with strides)
 *
 * optr = zarg(iptr)
 */
void md_zarg2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zarg, D, dims, ostr, optr, istr, iptr);
}



/**
 * Get argument of complex arrays (without strides)
 *
 * optr = zarg(iptr)
 */
void md_zarg(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zarg2, D, dims, optr, iptr);
}


/**
 * Complex sinus
 *
 * optr = zsin(iptr)
 */
void md_zsin2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zsin, D, dims, ostr, optr, istr, iptr);
}


/**
 * Complex sinus
 *
 * optr = zsin(iptr)
 */
void md_zsin(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zsin2, D, dims, optr, iptr);
}


/**
 * Complex cosinus
 *
 * optr = zexp(iptr)
 */
void md_zcos2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	MAKE_Z2OP(zcos, D, dims, ostr, optr, istr, iptr);
}


/**
 * Complex cosinus
 *
 * optr = zsin(iptr)
 */
void md_zcos(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zcos2, D, dims, optr, iptr);
}



/**
 * Calculate inner product between two scalar arrays (with strides)
 *
 * return iptr1^T * iptr2
 */
float md_scalar2(unsigned int D, const long dim[D], const long str1[D], const float* ptr1, const long str2[D], const float* ptr2)
{
#if 1
	if (       (D == md_calc_blockdim(D, dim, str1, FL_SIZE))
		&& (D == md_calc_blockdim(D, dim, str2, FL_SIZE))) {

#ifdef USE_CUDA
		if (cuda_ondevice(ptr1)) {

			assert(cuda_ondevice(ptr2));

			return gpu_ops.dot(md_calc_size(D, dim), ptr1, ptr2);
		}
#endif
		return cpu_ops.dot(md_calc_size(D, dim), ptr1, ptr2);
	}
#endif

	double ret = 0.;
	double* retp = &ret;

#ifdef USE_CUDA
	if (cuda_ondevice(ptr1))
		retp = gpu_constant(&ret, DL_SIZE);
#endif

	long stro[D];
	md_singleton_strides(D, stro);

	// Because this might lose precision for large data sets
	// we use double precision to accumlate result
	// (Kahan summation formula would be another option)

	md_fmacD2(D, dim, stro, retp, str1, ptr1, str2, ptr2);

#ifdef USE_CUDA
	if (cuda_ondevice(ptr1)) {

		md_copy(1, (long[1]){ 1 }, &ret, retp, DL_SIZE);
		md_free(retp);
	}
#endif
	return ret;
}



/**
 * Calculate inner product between two scalar arrays (without strides)
 *
 * return iptr1^T * iptr2
 */
float md_scalar(unsigned int D, const long dim[D], const float* ptr1, const float* ptr2)
{
	long str[D];
	md_calc_strides(D, str, dim, FL_SIZE);

	return md_scalar2(D, dim, str, ptr1, str, ptr2);
}



/**
 * Calculate l2 norm of scalar array (with strides)
 *
 * return sqrt(iptr^T * iptr)
 */
float md_norm2(unsigned int D, const long dim[D], const long str[D], const float* ptr)
{
	return sqrtf(md_scalar2(D, dim, str, ptr, str, ptr));
}



/**
 * Calculate l2 norm of scalar array (without strides)
 *
 * return sqrt(iptr^T * iptr)
 */
float md_norm(unsigned int D, const long dim[D], const float* ptr)
{
	return sqrtf(md_scalar(D, dim, ptr, ptr));
}



/**
 * Calculate root-mean-square of complex array
 *
 * return sqrt(in^H * in / length(in))
 */
float md_zrms(unsigned int D, const long dim[D], const complex float* in)
{
	return md_znorm(D, dim, in) / sqrtl(md_calc_size(D, dim));
}



/**
 * Calculate root-mean-square error between two complex arrays
 *
 * return sqrt((in1 - in2)^2 / length(in))
 */
float md_zrmse(unsigned int D, const long dim[D], const complex float* in1, const complex float* in2)
{
	complex float* err = md_alloc_sameplace(D, dim, CFL_SIZE, in1);

	md_zsub(D, dim, err, in1, in2);

	float val = md_zrms(D, dim, err);

	md_free(err);

	return val;
}



/**
 * Calculate normalized root-mean-square error between two complex arrays
 *
 * return RMSE(ref,in) / RMS(in)
 */
float md_znrmse(unsigned int D, const long dim[D], const complex float* ref, const complex float* in)
{
	return md_zrmse(D, dim, ref, in) / md_zrms(D, dim, ref);
}



/**
 * Calculate l2 norm error between two complex arrays
 *
 * return sqrt(sum(in1 - in2)^2)
 */
float md_znorme(unsigned int D, const long dim[D], const complex float* in1, const complex float* in2)
{
	complex float* err = md_alloc_sameplace(D, dim, CFL_SIZE, in1);

	md_zsub(D, dim, err, in1, in2);

	float val = md_znorm(D, dim, err);

	md_free(err);

	return val;
}



/**
 * Calculate relative l2 norm error of two complex arrays
 *
 * return norm(ref - in) / norm(ref)
 */
float md_zrnorme(unsigned int D, const long dim[D], const complex float* ref, const complex float* in)
{
	return md_znorme(D, dim, ref, in) / md_znorm(D, dim, ref);
}



/**
 * Calculate inner product between two complex arrays (with strides)
 *
 * return iptr1^H * iptr2
 */
complex float md_zscalar2(unsigned int D, const long dim[D], const long str1[D], const complex float* ptr1, const long str2[D], const complex float* ptr2)
{
	complex double ret = 0.;
	complex double* retp = &ret;

#ifdef USE_CUDA
	if (cuda_ondevice(ptr1)) {

		// FIXME: because md_zfmacc2 with stride = 0 is slow

		complex float* tmp = md_alloc_gpu(D, dim, CFL_SIZE);

		long strs[D];
		md_calc_strides(D, strs, dim, CFL_SIZE);
		md_clear(D, dim, tmp, CFL_SIZE);

		md_zfmacc2(D, dim, strs, tmp, str1, ptr1, str2, ptr2);

		gpu_ops.zsum(md_calc_size(D, dim), tmp);

		complex float ret = 0.;
		md_copy(1, (long[1]){ 1 }, &ret, tmp, CFL_SIZE);
		md_free(tmp);

		return ret;
	}
#endif

#ifdef USE_CUDA
	if (cuda_ondevice(ptr1))
		retp = gpu_constant(&ret, CDL_SIZE);
#endif

	long stro[D];
	md_singleton_strides(D, stro);

	// Because this might lose precision for large data sets
	// we use double precision to accumlate result
	// (Kahan summation formula would be another option)

	md_zfmaccD2(D, dim, stro, retp, str1, ptr1, str2, ptr2);

#ifdef USE_CUDA
	if (cuda_ondevice(ptr1)) {

		md_copy(1, (long[1]){ 1 }, &ret, retp, CDL_SIZE);
		md_free(retp);
	}
#endif

	return (complex float)ret;
}



/**
 * Calculate inner product between two complex arrays (without strides)
 *
 * return iptr1^H * iptr2
 */
complex float md_zscalar(unsigned int D, const long dim[D], const complex float* ptr1, const complex float* ptr2)
{
	long str[D];
	md_calc_strides(D, str, dim, CFL_SIZE);

	return md_zscalar2(D, dim, str, ptr1, str, ptr2);
}



/**
 * Calculate real part of the inner product between two complex arrays (with strides)
 *
 * return iptr1^H * iptr2
 */
float md_zscalar_real2(unsigned int D, const long dims[D], const long strs1[D], const complex float* ptr1, const long strs2[D], const complex float* ptr2)
{
	long dimsR[D + 1];
	long strs1R[D + 1];
	long strs2R[D + 1];

	real_from_complex_dims(D, dimsR, dims);
	real_from_complex_strides(D, strs1R, strs1);
	real_from_complex_strides(D, strs2R, strs2);

	return md_scalar2(D + 1, dimsR, strs1R, (const float*)ptr1, strs2R, (const float*)ptr2);
}



/**
 * Calculate real part of the inner product between two complex arrays (without strides)
 *
 * return iptr1^H * iptr2
 */
float md_zscalar_real(unsigned int D, const long dims[D], const complex float* ptr1, const complex float* ptr2)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	return md_zscalar_real2(D, dims, strs, ptr1, strs, ptr2);
}



/**
 * Calculate l2 norm of complex array (with strides)
 *
 * return sqrt(iptr^H * iptr)
 */
float md_znorm2(unsigned int D, const long dim[D], const long str[D], const complex float* ptr)
{
	return sqrtf(md_zscalar_real2(D, dim, str, ptr, str, ptr));
//	return sqrtf(crealf(md_zscalar2(D, dim, str, ptr, str, ptr)));
}



/**
 * Calculate l2 norm of complex array (without strides)
 *
 * return sqrt(iptr^H * iptr)
 */
float md_znorm(unsigned int D, const long dim[D], const complex float* ptr)
{
	return sqrtf(md_zscalar_real(D, dim, ptr, ptr));
//	return sqrtf(crealf(md_zscalar(D, dim, ptr, ptr)));
}



/**
 * Calculate absolute value.
 *
 */
void md_abs2(unsigned int D, const long dims[D], const long ostr[D], float* optr,
		const long istr[D], const float* iptr)
{
	assert(optr != iptr);

	md_clear2(D, dims, ostr, optr, FL_SIZE);
	md_fmac2(D, dims, ostr, optr, istr, iptr, istr, iptr);	// FIXME: should be cheaper
	md_sqrt2(D, dims, ostr, optr, ostr, optr);
}



/**
 * Calculate absolute value.
 *
 */
void md_abs(unsigned int D, const long dims[D], float* optr, const float* iptr)
{
	make_2op_simple(md_abs2, D, dims, optr, iptr);
}



/**
 * Calculate absolute value.
 *
 */
void md_zabs2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr,
		const long istr[D], const complex float* iptr)
{
#if 1
	MAKE_Z2OP(zabs, D, dims, ostr, optr, istr, iptr);
#else
	// FIXME: special case of md_rss

	assert(optr != iptr);

	md_clear2(D, dims, ostr, optr, CFL_SIZE);
	md_zfmacc2(D, dims, ostr, optr, istr, iptr, istr, iptr);
#if 1
	long dimsR[D + 1];
	long strsR[D + 1];

	real_from_complex_dims(D, dimsR, dims);
	real_from_complex_strides(D, strsR, ostr);

	//md_sqrt2(D, dimsR + 1, strsR + 1, (float*)optr, strsR + 1, (const float*)optr); // skipping imaginary part is expensive
	md_sqrt2(D + 1, dimsR, strsR, (float*)optr, strsR, (const float*)optr);
#else
	md_zsqrt2(D, dims, ostr, optr, ostr, optr);
#endif
#endif
}



/**
 * Calculate absolute value.
 *
 */
void md_zabs(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr)
{
	make_z2op_simple(md_zabs2, D, dims, optr, iptr);
}



/**
 * Calculate sum of absolute values.
 *
 */
float md_asum2(unsigned int D, const long dims[D], const long strs[D], const float* ptr)
{
#if 1
	if (md_calc_blockdim(D, dims, strs, FL_SIZE) == D) {

#ifdef USE_CUDA
		if (cuda_ondevice(ptr))
			return gpu_ops.asum(md_calc_size(D, dims), ptr);
#endif
		return cpu_ops.asum(md_calc_size(D, dims), ptr);
	}
#endif

	float* tmp = md_alloc_sameplace(D, dims, FL_SIZE, ptr);

	long strs1[D];
	md_calc_strides(D, strs1, dims, FL_SIZE);

	md_abs2(D, dims, strs1, tmp, strs, ptr);

	float ret = 0.;
	float* retp = &ret;

#ifdef USE_CUDA
	if (cuda_ondevice(ptr))
		retp = gpu_constant(&ret, FL_SIZE);
#endif
	long dims0[D];
	md_singleton_dims(D, dims0);

	md_axpy2(D, dims, MD_STRIDES(D, dims0, FL_SIZE), retp, 1., strs1, tmp);

#ifdef USE_CUDA
	if (cuda_ondevice(ptr)) {

		md_copy(D, dims0, &ret, retp, FL_SIZE);
		md_free(retp);
	}
#endif

	md_free(tmp);

	return ret;
}



/**
 * Calculate sum of absolute values.
 *
 */
float md_asum(unsigned int D, const long dims[D], const float* ptr)
{
	return md_asum2(D, dims, MD_STRIDES(D, dims, FL_SIZE), ptr);
}



/**
 * Calculate sum of absolute values of complex numbers
 * where real and imaginary are separate elements of the sum.
 * (similar to BLAS L1 function).
 *
 */
float md_zasum2(unsigned int D, const long dims[D], const long strs[D], const complex float* ptr)
{
	long dimsR[D + 1];
	real_from_complex_dims(D, dimsR, dims);

	long strsR[D + 1];
	real_from_complex_strides(D, strsR, strs);

	return md_asum2(D + 1, dimsR, strsR, (const float*)ptr);
}



/**
 * Calculate sum of absolute values of complex numbers
 * where real and imaginary are separate elements of the sum.
 * (similar to BLAS L1 function).
 *
 */
float md_zasum(unsigned int D, const long dims[D], const complex float* ptr)
{
	return md_zasum2(D, dims, MD_STRIDES(D, dims, CFL_SIZE), ptr);
}



/**
 * Calculate l1 norm of complex array (with strides)
 */
float md_z1norm2(unsigned int D, const long dims[D], const long strs[D], const complex float* ptr)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, ptr);

	md_zabs2(D, dims, MD_STRIDES(D, dims, CFL_SIZE), tmp, strs, ptr);

	float val = md_zasum(D, dims, tmp);

	md_free(tmp);

	return val;
}



/**
 * Calculate l1 norm of complex array (without strides)
 */
float md_z1norm(unsigned int D, const long dim[D], const complex float* ptr)
{
	return md_z1norm2(D, dim, MD_STRIDES(D, dim, CFL_SIZE), ptr);
}



/**
 * Root of sum of squares along selected dimensions
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for applying the root of sum of squares, ie the dimensions that will not stay
 */
void md_rss(unsigned int D, const long dims[D], unsigned int flags, float* dst, const float* src)
{
	long str1[D];
	long str2[D];
	long dims2[D];

	md_select_dims(D, ~flags, dims2, dims);

	md_calc_strides(D, str1, dims, FL_SIZE);
	md_calc_strides(D, str2, dims2, FL_SIZE);

	md_clear(D, dims2, dst, FL_SIZE);
	md_fmac2(D, dims, str2, dst, str1, src, str1, src);

	md_sqrt(D, dims2, dst, dst);
}



/**
 * Sum of squares along selected dimensions (without strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for applying the root of sum of squares, i.e. the dimensions that will not stay
 */
void md_zss2(unsigned int D, const long dims[D], unsigned int flags, const long str2[D], complex float* dst, const long str1[D], const complex float* src)
{
	long dims2[D];
	md_select_dims(D, ~flags, dims2, dims);

	md_clear2(D, dims2, str2, dst, CFL_SIZE);
	md_zfmacc2(D, dims, str2, dst, str1, src, str1, src);
}


/**
 * Sum of squares along selected dimensions (with strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for applying the root of sum of squares, i.e. the dimensions that will not stay
 */
void md_zss(unsigned int D, const long dims[D], unsigned int flags, complex float* dst, const complex float* src)
{
	long dims2[D];

	md_select_dims(D, ~flags, dims2, dims);

	md_zss2(D, dims, flags, MD_STRIDES(D, dims2, CFL_SIZE), dst,
				MD_STRIDES(D, dims, CFL_SIZE), src);
}


/**
 * Root of sum of squares along selected dimensions
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for applying the root of sum of squares, i.e. the dimensions that will not stay
 */
void md_zrss(unsigned int D, const long dims[D], unsigned int flags, complex float* dst, const complex float* src)
{
	long dims2[D];
	md_select_dims(D, ~flags, dims2, dims);
#if 1
	md_zss(D, dims, flags, dst, src);

#if 1
	long dims2R[D + 1];
	real_from_complex_dims(D, dims2R, dims2);

	md_sqrt(D + 1, dims2R, (float*)dst, (const float*)dst);
#else
	md_zsqrt(D, dims2, dst, dst);
#endif
#else
	long dimsR[D + 1];
	real_from_complex_dims(D, dimsR, dims);
	md_rrss(D + 1, dimsR, (flags << 1), (float*)dst, (const float*)src);
#endif
}



/**
 * Compute variance along selected dimensions (with strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for calculating variance, i.e. the dimensions that will not stay
 */
void md_zvar2(unsigned int D, const long dims[D], unsigned int flags, const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	long odims[D];
	long fdims[D];

	md_select_dims(D, ~flags, odims, dims);
	md_select_dims(D, flags, fdims, dims);

	long tstrs[D];
	md_calc_strides(D, tstrs, dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, optr);

	md_zavg2(D, dims, flags, ostr, optr, istr, iptr);
	md_zsub2(D, dims, tstrs, tmp, istr, iptr, ostr, optr);

	double scale = md_calc_size(D, fdims) - 1.;

	md_zss2(D, dims, flags, ostr, optr, tstrs, tmp);

	md_free(tmp);

	md_zsmul2(D, odims, ostr, optr, ostr, optr, 1. / scale);
}



/**
 * Compute variance along selected dimensions (without strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for calculating variance, i.e. the dimensions that will not stay
 */
void md_zvar(unsigned int D, const long dims[D], unsigned int flags, complex float* optr, const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	md_zvar2(D, dims, flags,
			MD_STRIDES(D, odims, CFL_SIZE), optr,
			MD_STRIDES(D, dims, CFL_SIZE), iptr);
}



/**
 * Compute standard deviation along selected dimensions (with strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for calculating standard deviation, i.e. the dimensions that will not stay
 */
void md_zstd2(unsigned int D, const long dims[D], unsigned int flags, const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	md_zvar2(D, dims, flags, ostr, optr, istr, iptr);

	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

#if 1
	long dimsR[D + 1];
	real_from_complex_dims(D, dimsR, odims);

	long strsR[D + 1];
	real_from_complex_strides(D, strsR, ostr);

	md_sqrt2(D + 1, dimsR, strsR, (float*)optr, strsR, (const float*)optr);
#else
	md_zsqrt2(D, odims, ostr, optr, ostr, optr);
#endif
}



/**
 * Compute standard deviation along selected dimensions (without strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for calculating standard deviation, i.e. the dimensions that will not stay
 */
void md_zstd(unsigned int D, const long dims[D], unsigned int flags, complex float* optr, const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	md_zstd2(D, dims, flags,
			MD_STRIDES(D, odims, CFL_SIZE), optr,
			MD_STRIDES(D, dims, CFL_SIZE), iptr);
}



/**
 * Compute covariance along selected dimensions (with strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for calculating variance, i.e. the dimensions that will not stay
 */
void md_zcovar2(unsigned int D, const long dims[D], unsigned int flags,
		const long ostr[D], complex float* optr,
		const long istr1[D], const complex float* iptr1,
		const long istr2[D], const complex float* iptr2)
{
	long odims[D];
	long fdims[D];

	md_select_dims(D, ~flags, odims, dims);
	md_select_dims(D, flags, fdims, dims);

	long tstrs[D];
	md_calc_strides(D, tstrs, dims, CFL_SIZE);

	complex float* tmp1 = md_alloc_sameplace(D, dims, CFL_SIZE, optr);

	md_zavg2(D, dims, flags, ostr, optr, istr1, iptr1);
	md_zsub2(D, dims, tstrs, tmp1, istr1, iptr1, ostr, optr);

	complex float* tmp2 = md_alloc_sameplace(D, dims, CFL_SIZE, optr);

	md_zavg2(D, dims, flags, ostr, optr, istr2, iptr2);
	md_zsub2(D, dims, tstrs, tmp2, istr2, iptr2, ostr, optr);

	double scale = md_calc_size(D, fdims) - 1.;

	md_clear2(D, odims, ostr, optr, CFL_SIZE);
	md_zfmacc2(D, dims, ostr, optr, tstrs, tmp1, tstrs, tmp2);

	md_free(tmp1);
	md_free(tmp2);

	md_zsmul2(D, odims, ostr, optr, ostr, optr, 1. / scale);
}


/**
 * Compute covariance along selected dimensions (without strides)
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for calculating variance, i.e. the dimensions that will not stay
 */
void md_zcovar(unsigned int D, const long dims[D], unsigned int flags,
		complex float* optr, const complex float* iptr1, const complex float* iptr2)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	md_zcovar2(D, dims, flags,
			MD_STRIDES(D, odims, CFL_SIZE), optr,
			MD_STRIDES(D, dims, CFL_SIZE), iptr1,
			MD_STRIDES(D, dims, CFL_SIZE), iptr2);
}


/**
 * Average along flagged dimensions (without strides)
 *
 * @param dims -- full dimensions of iptr
 * @param flags -- bitmask for applying the average, i.e. the dimensions that will not stay
 */
void md_zavg(unsigned int D, const long dims[D], unsigned int flags, complex float* optr, const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	md_zavg2(D, dims, flags,
			MD_STRIDES(D, odims, CFL_SIZE), optr,
			MD_STRIDES(D, dims, CFL_SIZE), iptr);
}



/**
 * Average along flagged dimensions (with strides)
 *
 * @param dims -- full dimensions of iptr
 * @param flags -- bitmask for applying the average, i.e. the dimensions that will not stay
 */
void md_zavg2(unsigned int D, const long dims[D], unsigned int flags, const long ostr[D],  complex float* optr, const long istr[D], const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);
	md_clear(D, odims, optr, CFL_SIZE);

	//FIXME: this is faster
#if 1
	complex float* o = md_alloc_sameplace(1, MD_DIMS(1), CFL_SIZE, optr);
	md_zfill(1, MD_DIMS(1), o, 1.);

	long ss[D];
	md_singleton_strides(D, ss);
	md_zfmac2(D, dims, ostr, optr, istr, iptr, ss, o);
	md_free(o);
#else
	md_zaxpy2(D, dims, ostr, optr, 1., istr, iptr);
#endif

	long sdims[D];
	md_select_dims(D, flags, sdims, dims);

	long scale = md_calc_size(D, sdims);

	if (scale != 0.)
		md_zsmul(D, odims, optr, optr, 1. / scale);
}



/**
 * Weighted average along flagged dimensions (without strides)
 *
 * @param dims -- full dimensions of iptr
 * @param flags -- bitmask for applying the weighted average, i.e. the dimensions that will not stay
 */
void md_zwavg(unsigned int D, const long dims[D], unsigned int flags, complex float* optr, const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	md_zwavg2(D, dims, flags,
			MD_STRIDES(D, odims, CFL_SIZE), optr,
			MD_STRIDES(D, dims, CFL_SIZE), iptr);
}



/**
 * Weighted average along flagged dimensions (with strides)
 *
 * @param dims -- full dimensions of iptr
 * @param flags -- bitmask for applying the weighted average, i.e. the dimensions that will not stay
 */
void md_zwavg2(unsigned int D, const long dims[D], unsigned int flags, const long ostr[D],  complex float* optr, const long istr[D], const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	complex float* weights = md_alloc_sameplace(D, odims, CFL_SIZE, iptr);

	md_zwavg2_core1(D, dims, flags, ostr, weights, istr, iptr);
	md_zwavg2_core2(D, dims, flags, ostr, optr, weights, istr, iptr);

	md_free(weights);
}



/**
 * Compute weights for weighted average
 *
 * @param iptr input array to be averaged
 * @param weights output weights
 */
void md_zwavg2_core1(unsigned int D, const long dims[D], unsigned int flags, const long ostr[D],  complex float* weights, const long istr[D], const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	complex float* pattern = md_alloc_sameplace(D, dims, CFL_SIZE, iptr);

	long onestrs[D];
	md_singleton_strides(D, onestrs);

	md_zcmp2(D, dims, istr, pattern, istr, iptr, onestrs, &(complex float){ 0. });
	md_zsub2(D, dims, istr, pattern, onestrs, &(complex float){ 1. }, istr, pattern);

	md_clear2(D, odims, ostr, weights, CFL_SIZE);
	md_zaxpy2(D, dims, ostr, weights, 1., istr, pattern);

	md_free(pattern);
}



/**
 * Weighted average along flagged dimensions with given weights
 *
 * @param weights precomputed weights for averaging
 * @param optr output array after averaging
 */
void md_zwavg2_core2(unsigned int D, const long dims[D], unsigned int flags, const long ostr[D],  complex float* optr, const complex float* weights, const long istr[D], const complex float* iptr)
{
	long odims[D];
	md_select_dims(D, ~flags, odims, dims);

	md_clear2(D, odims, ostr, optr, CFL_SIZE);
	md_zaxpy2(D, dims, ostr, optr, 1., istr, iptr);

	md_zdiv(D, odims, optr, optr, weights);
}



/**
 * Fill complex array with value (with strides).
 *
 */
void md_zfill2(unsigned int D, const long dim[D], const long str[D], complex float* ptr, complex float val)
{
	md_fill2(D, dim, str, ptr, &val, CFL_SIZE);
}



/**
 * Fill complex array with value (without strides).
 *
 */
extern void md_zfill(unsigned int D, const long dim[D], complex float* ptr, complex float val)
{
	md_fill(D, dim, ptr, &val, CFL_SIZE);
}






/**
 * Step (2) of Soft Thresholding multi-dimensional arrays, y = ST(x, lambda)
 * 2) computes resid = MAX( (abs(x) - lambda)/abs(x), 0 ) (with strides)
 *
 * @param D number of dimensions
 * @param dim dimensions of input/output
 * @param lambda threshold parameter
 * @param ostr output strides
 * @param optr pointer to output, y
 * @param istr input strides
 * @param iptr pointer to input, abs(x)
 */
void md_zsoftthresh_half2(unsigned int D, const long dim[D], float lambda, const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	NESTED(void, nary_zsoftthresh_half, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zsoftthresh_half(data->size, lambda, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr, iptr, (size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zsoftthresh_half);
}



/**
 * Step (2) of Soft Thresholding multi-dimensional arrays, y = ST(x, lambda)
 * 2) computes resid = MAX( (abs(x) - lambda)/abs(x), 0 ) (with strides)
 *
 * @param D number of dimensions
 * @param dim dimensions of input/output
 * @param lambda threshold parameter
 * @param ostr output strides
 * @param optr pointer to output, y
 * @param istr input strides
 * @param iptr pointer to input, abs(x)
 */
void md_softthresh_half2(unsigned int D, const long dim[D], float lambda, const long ostr[D], float* optr, const long istr[D], const float* iptr)
{
	NESTED(void, nary_softthresh_half, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->softthresh_half(data->size, lambda, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr, iptr, (size_t[2]){ FL_SIZE, FL_SIZE }, nary_softthresh_half);
}



/**
 * Step (1) of Soft Thresholding multi-dimensional arrays, y = ST(x, lambda)
 * 1) computes resid = MAX( (abs(x) - lambda)/abs(x), 0 ) (without strides)
 *
 * @param D number of dimensions
 * @param dim dimensions of input/output
 * @param lambda threshold parameter
 * @param optr pointer to output, y
 * @param iptr pointer to input, x
 */
void md_zsoftthresh_half(unsigned int D, const long dim[D], float lambda, complex float* optr, const complex float* iptr)
{
	long str[D];
	md_calc_strides(D, str, dim, CFL_SIZE);

	md_zsoftthresh_half2(D, dim, lambda, str, optr, str, iptr);
}



void md_softthresh_core2(unsigned int D, const long dims[D], float lambda, unsigned int flags, float* tmp_norm, const long ostrs[D], float* optr, const long istrs[D], const float* iptr)
{
	long norm_dims[D];
	long norm_strs[D];

	md_select_dims(D, ~flags, norm_dims, dims);
	md_calc_strides(D, norm_strs, norm_dims, FL_SIZE);

	md_rss(D, dims, flags, tmp_norm, iptr);
	md_softthresh_half2(D, norm_dims, lambda, norm_strs, tmp_norm, norm_strs, tmp_norm);
	md_mul2(D, dims, ostrs, optr, norm_strs, tmp_norm, istrs, iptr);
}



/**
 * Soft Thresholding for floats (with strides)
 *
 * optr = ST(iptr, lambda)
 */
void md_softthresh2(unsigned int D, const long dims[D], float lambda, unsigned int flags, const long ostrs[D], float* optr, const long istrs[D], const float* iptr)
{
	NESTED(void, nary_softthresh, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->softthresh(data->size, lambda, ptr[0], ptr[1]);
	};

	if (0 == flags) {

		optimized_twoop_oi(D, dims, ostrs, optr, istrs, iptr, (size_t[2]){ FL_SIZE, FL_SIZE }, nary_softthresh);
		return;
	}

	long norm_dims[D];
	md_select_dims(D, ~flags, norm_dims, dims);

	float* tmp_norm = md_alloc_sameplace(D, norm_dims, FL_SIZE, iptr);

	md_softthresh_core2(D, dims, lambda, flags, tmp_norm, ostrs, optr, istrs, iptr);

	md_free(tmp_norm);
}



/**
 * Soft Thresholding for floats (without strides)
 *
 * optr = ST(iptr, lambda)
 */
void md_softthresh(unsigned int D, const long dims[D], float lambda, unsigned int flags, float* optr, const float* iptr)
{
	long str[D];
	md_calc_strides(D, str, dims, FL_SIZE);

	md_softthresh2(D, dims, lambda, flags, str, optr, str, iptr);
}



void md_zsoftthresh_core2(unsigned int D, const long dims[D], float lambda, unsigned int flags, complex float* tmp_norm, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	long norm_dims[D];
	long norm_strs[D];

	md_select_dims(D, ~flags, norm_dims, dims);
	md_calc_strides(D, norm_strs, norm_dims, CFL_SIZE);

	md_zrss(D, dims, flags, tmp_norm, iptr);
	md_zsoftthresh_half2(D, norm_dims, lambda, norm_strs, tmp_norm, norm_strs, tmp_norm);
	md_zmul2(D, dims, ostrs, optr, norm_strs, tmp_norm, istrs, iptr);
}





/**
 * Soft thresholding using norm along arbitrary dimension (with strides)
 *
 * y = ST(x, lambda)
 * 1) computes resid = MAX((norm(x) - lambda) / norm(x), 0)
 * 2) multiplies y = resid * x
 *
 * @param D number of dimensions
 * @param dims dimensions of input/output
 * @param lambda threshold parameter
 * @param flags jointly thresholded dimensions
 * @param optr destination -- soft thresholded values
 * @param iptr source -- values to be soft thresholded
 */
void md_zsoftthresh2(unsigned int D, const long dims[D], float lambda, unsigned int flags, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	NESTED(void, nary_zsoftthresh, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zsoftthresh(data->size, lambda, ptr[0], ptr[1]);
	};

	if (0 == flags) {

		optimized_twoop_oi(D, dims, ostrs, optr, istrs, iptr, (size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zsoftthresh);
		return;
	}

	long norm_dims[D];
	md_select_dims(D, ~flags, norm_dims, dims);

	complex float* tmp_norm = md_alloc_sameplace(D, norm_dims, CFL_SIZE, iptr);

	md_zsoftthresh_core2(D, dims, lambda, flags, tmp_norm, ostrs, optr, istrs, iptr);

	md_free(tmp_norm);
}


/**
 * Soft thresholding using norm along arbitrary dimension (without strides)
 *
 * y = ST(x, lambda)
 * 1) computes resid = MAX((norm(x) - lambda) / norm(x), 0)
 * 2) multiplies y = resid * x
 *
 * @param D number of dimensions
 * @param dims dimensions of input/output
 * @param lambda threshold parameter
 * @param flags jointly thresholded dimensions
 * @param optr destination -- soft thresholded values
 * @param iptr source -- values to be soft thresholded
 */
void md_zsoftthresh(unsigned int D, const long dims[D], float lambda, unsigned int flags, complex float* optr, const complex float* iptr)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zsoftthresh2(D, dims, lambda, flags, strs, optr, strs, iptr);
}





/**
 * Produces a mask (1s and 0s) of the non-zero support of a hard thresholded input vector
 * Multi-dimensional operation with strides
 * Hard thresholding is performed by selection of the k largest elements in input.
 *
 * @param D number of dimensions
 * @param dim dimensions of input/output
 * @param k threshold parameter
 * @param flags flags for joint operation
 * @param ostr output strides
 * @param optr pointer to output
 * @param istr input strides
 * @param iptr pointer to input
 */
void md_zhardthresh_mask2(unsigned int D, const long dim[D], unsigned int k, unsigned int flags, complex float* tmp_norm, const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	NESTED(void, nary_zhardthresh_mask, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zhardthresh_mask(data->size, k, ptr[0], ptr[1]);
	};

	if (0 == flags) {

		optimized_twoop_oi(D, dim, ostr, optr, istr, iptr, (size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zhardthresh_mask);
		return;
	}

	long norm_dims[D];
	long norm_strs[D];

	md_select_dims(D, ~flags, norm_dims, dim);
	md_calc_strides(D, norm_strs, norm_dims, CFL_SIZE);

	md_zrss(D, dim, flags, tmp_norm, iptr);
	optimized_twoop_oi(D, norm_dims, norm_strs, tmp_norm, norm_strs, tmp_norm, (size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zhardthresh_mask);
	md_copy2(D, dim, ostr, optr, norm_strs, tmp_norm, CFL_SIZE);
}


/**
 * Produces a mask (1s and 0s) of the non-zero support of a hard thresholded input vector
 * Multi-dimensional operation using the same strides for input and output.
 * Hard thresholding is performed by selection of the k largest elements in input.
 *
 * @param D number of dimensions
 * @param dim dimensions of input/output
 * @param k threshold parameter
 * @param optr pointer to output
 * @param iptr pointer to input
 */
void md_zhardthresh_mask(unsigned int D, const long dim[D], unsigned int k, unsigned int flags, complex float* optr, const complex float* iptr)
{
	long str[D];
	md_calc_strides(D, str, dim, CFL_SIZE);

	long norm_dims[D];
	md_select_dims(D, ~flags, norm_dims, dim);

	complex float* tmp_norm = md_alloc_sameplace(D, norm_dims, CFL_SIZE, iptr);

	md_zhardthresh_mask2(D, dim, k, flags, tmp_norm, str, optr, str, iptr);

	md_free(tmp_norm);
}


/**
 * Joint Hard thresholding  (with strides)
 * Performs hard thresholding to the norm along dimension specified by flags
 * Applies the support of thresholded norm to every vector along that dimension
 * Hard thresholding refers to the selection of the k largest elements in vector.
 *
 * @param D number of dimensions
 * @param dims dimensions of input/output
 * @param k threshold (sorted) index
 * @param flags jointly thresholded dimensions
 * @param tmp_norm temporary array for joint operation
 * @param ostrs destination strides
 * @param optr destination -- thresholded values
 * @param istrs source strides
 * @param iptr source -- values to be thresholded
 */
void md_zhardthresh_joint2(unsigned int D, const long dims[D], unsigned int k, unsigned int flags, complex float* tmp_norm, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	long norm_dims[D];
	long norm_strs[D];

	md_select_dims(D, ~flags, norm_dims, dims);
	md_calc_strides(D, norm_strs, norm_dims, CFL_SIZE);

	md_zrss(D, dims, flags, tmp_norm, iptr);

	NESTED(void, nary_zhardthresh_mask, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zhardthresh_mask(data->size, k, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, norm_dims, norm_strs, tmp_norm, norm_strs, tmp_norm, (size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zhardthresh_mask);
	md_zmul2(D, dims, ostrs, optr, norm_strs, tmp_norm, istrs, iptr);
}




/**
 * Hard thresholding (with strides)
 *
 * y = HT(x, k), selects k largest elements of x
 * computes y = x * (abs(x) > t(k)),
 * k = threshold index of sorted x, t(k)= value of sorted x at k
 *
 * @param D number of dimensions
 * @param dims dimensions of input/output
 * @param k threshold (sorted) index
 * @param flags jointly thresholded dimensions
 * @param tmp_norm temporary array for joint operation
 * @param ostrs destination strides
 * @param optr destination -- thresholded values
 * @param istrs source strides
 * @param iptr source -- values to be thresholded
 */
void md_zhardthresh2(unsigned int D, const long dims[D], unsigned int k, unsigned int flags, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	NESTED(void, nary_zhardthresh, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zhardthresh(data->size, k, ptr[0], ptr[1]);
	};

	if (0 == flags) {

		optimized_twoop_oi(D, dims, ostrs, optr, istrs, iptr, (size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zhardthresh);
		return;
	}

	long norm_dims[D];
	md_select_dims(D, ~flags, norm_dims, dims);

	complex float* tmp_norm = md_alloc_sameplace(D, norm_dims, CFL_SIZE, iptr);
	md_zhardthresh_joint2(D, dims, k, flags, tmp_norm, ostrs, optr, istrs,iptr);

	md_free(tmp_norm);
}


/**
 * Hard thresholding (without strides)
 *
 * y = HT(x, k), select k largest elements.
 *
 * @param D number of dimensions
 * @param dims dimensions of input/output
 * @param k threshold parameter
 * @param flags jointly thresholded dimensions
 * @param optr destination -- thresholded values
 * @param iptr source -- values to be thresholded
 */
void md_zhardthresh(unsigned int D, const long dims[D], unsigned int k, unsigned int flags, complex float* optr, const complex float* iptr)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zhardthresh2(D, dims, k, flags, strs, optr, strs, iptr);
}


/**
 * Elementwise minimum of input and scalar (with strides)
 *
 * optr = min(val, iptr)
 */
void md_smin2(unsigned int D, const long dim[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float val)
{
	float* tmp = md_alloc_sameplace(D, dim, FL_SIZE, iptr);

	md_slessequal2(D, dim, ostr, tmp, istr, iptr, val);
	md_mul2(D, dim, ostr, optr, istr, iptr, istr, tmp);

	md_free(tmp);
}



/**
 * Elementwise minimum of input and scalar (without strides)
 *
 * optr = min(val, iptr)
 */
void md_smin(unsigned int D, const long dim[D], float* optr, const float* iptr, float val)
{
	long str[D];
 	md_calc_strides(D, str, dim, FL_SIZE);

	md_smin2(D, dim, str, optr, str, iptr, val);
}



/**
 * Elementwise maximum of input and scalar (with strides)
 *
 * optr = max(val, iptr)
 */
void md_smax2(unsigned int D, const long dim[D], const long ostr[D], float* optr, const long istr[D], const float* iptr, float val)
{
#if 0
	// slow on GPU due to make_3op_scalar
#if 0
	float* tmp = md_alloc_sameplace(D, dim, FL_SIZE, iptr);
	md_sgreatequal2(D, dim, ostr, tmp, istr, iptr, val);
	md_mul2(D, dim, ostr, optr, istr, iptr, istr, tmp);
	md_free(tmp);
#else
	make_3op_scalar(md_max2, D, dim, ostr, optr, istr, iptr, val);
#endif
#else
	(void)0;

	NESTED(void, nary_smax, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->smax(data->size, val, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr, iptr,
		(size_t[2]){ FL_SIZE, FL_SIZE }, nary_smax);
#endif
}


/**
 * Elementwise maximum of input and scalar (with strides)
 *
 * optr = max(val, iptr)
 */
void md_zsmax2(unsigned int D, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr, float val)
{
#if 0
	complex float* tmp = md_alloc_sameplace(D, dim, CFL_SIZE, iptr);
	md_zsgreatequal2(D, dim, ostr, tmp, istr, iptr, val);
	md_zmul2(D, dim, ostr, optr, istr, iptr, istr, tmp);
	md_free(tmp);
#else
#if 0
	make_z3op_scalar(md_zmax2, D, dim, ostr, optr, istr, iptr, val);
#else
	// FIXME: we should rather optimize md_zmul2 for this case

	NESTED(void, nary_zsmax, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zsmax(data->size, val, ptr[0], ptr[1]);
	};

	optimized_twoop_oi(D, dim, ostr, optr, istr, iptr,
		(size_t[2]){ CFL_SIZE, CFL_SIZE }, nary_zsmax);
#endif
#endif
}


/**
 * Elementwise maximum of input and scalar (without strides)
 *
 * optr = max(val, iptr)
 */
void md_zsmax(unsigned int D, const long dim[D], complex float* optr, const complex float* iptr, float val)
{
	long str[D];
	md_calc_strides(D, str, dim, CFL_SIZE);

	md_zsmax2(D, dim, str, optr, str, iptr, val);
}


/**
 * Elementwise minimum of input and scalar (without strides)
 *
 * optr = max(val, iptr)
 */
void md_smax(unsigned int D, const long dim[D], float* optr, const float* iptr, float val)
{
	long str[D];
 	md_calc_strides(D, str, dim, FL_SIZE);

	md_smax2(D, dim, str, optr, str, iptr, val);
}



static void md_fdiff_core2(unsigned int D, const long dims[D], unsigned int d, bool dir, const long ostr[D], float* out, const long istr[D], const float* in)
{
	long pos[D];
	md_set_dims(D, pos, 0);
	pos[d] = dir ? 1 : -1;

	md_circ_shift2(D, dims, pos, ostr, out, istr, in, FL_SIZE);
	md_sub2(D, dims, ostr, out, istr, in, ostr, out);
}

/**
 * Compute finite (forward) differences along selected dimensions.
 *
 */
void md_fdiff2(unsigned int D, const long dims[D], unsigned int d, const long ostr[D], float* out, const long istr[D], const float* in)
{
	md_fdiff_core2(D, dims, d, true, ostr, out, istr, in);
}



/**
 * Compute finite differences along selected dimensions.
 *
 */
void md_fdiff(unsigned int D, const long dims[D], unsigned int d, float* out, const float* in)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_fdiff2(D, dims, d, strs, out, strs, in);
}



/**
 * Compute finite (backward) differences along selected dimensions.
 *
 */
void md_fdiff_backwards2(unsigned int D, const long dims[D], unsigned int d, const long ostr[D], float* out, const long istr[D], const float* in)
{
	md_fdiff_core2(D, dims, d, false, ostr, out, istr, in);
}



/**
 * Compute finite (backward) differences along selected dimensions.
 *
 */
void md_fdiff_backwards(unsigned int D, const long dims[D], unsigned int d, float* out, const float* in)
{
	long strs[D];
	md_calc_strides(D, strs, dims, FL_SIZE);

	md_fdiff_backwards2(D, dims, d, strs, out, strs, in);
}



static void md_zfdiff_core2(unsigned int D, const long dims[D], unsigned int d, bool dir, const long ostr[D], complex float* out, const long istr[D], const complex float* in)
{
	// we could also implement in terms of md_fdiff2

	long pos[D];
	md_set_dims(D, pos, 0);
	pos[d] = dir ? 1 : -1;

	md_circ_shift2(D, dims, pos, ostr, out, istr, in, CFL_SIZE);
	md_zsub2(D, dims, ostr, out, istr, in, ostr, out);
}

/**
 * Compute finite (forward) differences along selected dimensions.
 *
 */
void md_zfdiff2(unsigned int D, const long dims[D], unsigned int d, const long ostr[D], complex float* out, const long istr[D], const complex float* in)
{
	md_zfdiff_core2(D, dims, d, true, ostr, out, istr, in);
}



/**
 * Compute finite (backward) differences along selected dimensions.
 *
 */
void md_zfdiff_backwards2(unsigned int D, const long dims[D], unsigned int d, const long ostr[D], complex float* out, const long istr[D], const complex float* in)
{
	md_zfdiff_core2(D, dims, d, false, ostr, out, istr, in);
}



/**
 * Compute finite (forward) differences along selected dimensions.
 *
 */
void md_zfdiff(unsigned int D, const long dims[D], unsigned int d, complex float* out, const complex float* in)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zfdiff2(D, dims, d, strs, out, strs, in);
}



/**
 * Compute finite (backward) differences along selected dimensions.
 *
 */
void md_zfdiff_backwards(unsigned int D, const long dims[D], unsigned int d, complex float* out, const complex float* in)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zfdiff_backwards2(D, dims, d, strs, out, strs, in);
}



// DO NOT USE DIRECTLY - this is used internally by fftmod from fft.[ch]
void md_zfftmod2(unsigned int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr, bool inv, double phase)
{
	assert(D > 0);
	assert((CFL_SIZE == ostrs[0]) && (CFL_SIZE == istrs[0]));

	unsigned int N = dims[0];

	NESTED(void, nary_zfftmod, (struct nary_opt_data_s* data, void* ptr[]))
	{
		data->ops->zfftmod(data->size, ptr[0], ptr[1], N, inv, phase);
	};

	optimized_twoop_oi(D - 1, dims + 1, ostrs + 1, optr, istrs + 1, iptr,
		(size_t[2]){ N * CFL_SIZE, N * CFL_SIZE }, nary_zfftmod);
}

void md_zfftmod(unsigned int D, const long dims[D], complex float* optr, const complex float* iptr, bool inv, double phase)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zfftmod2(D, dims, strs, optr, strs, iptr, inv, phase);
}



/**
 * Sum along selected dimensions
 *
 * @param dims -- full dimensions of src image
 * @param flags -- bitmask for applying the sum, i.e. the dimensions that will not stay
 */

void md_zsum(unsigned int D, const long dims[D], unsigned int flags, complex float* dst, const complex float* src)
{
	long str1[D];
	long str2[D];
	long dims2[D];

	md_select_dims(D, ~flags, dims2, dims);

	md_calc_strides(D, str1, dims, CFL_SIZE);
	md_calc_strides(D, str2, dims2, CFL_SIZE);

	md_clear(D, dims2, dst, CFL_SIZE);
	md_zadd2(D, dims, str2, dst, str2, dst, str1, src);
}



void md_real2(unsigned int D, const long dims[D], const long ostrs[D], float* dst, const long istrs[D], const complex float* src)
{
	md_copy2(D, dims, ostrs, dst, istrs, (const float*)src + 0, FL_SIZE);
}

void md_real(unsigned int D, const long dims[D], float* dst, const complex float* src)
{
	md_real2(D, dims, MD_STRIDES(D, dims, FL_SIZE), dst, MD_STRIDES(D, dims, CFL_SIZE), src);
}

void md_imag2(unsigned int D, const long dims[D], const long ostrs[D], float* dst, const long istrs[D], const complex float* src)
{
	md_copy2(D, dims, ostrs, dst, istrs, (const float*)src + 1, FL_SIZE);
}

void md_imag(unsigned int D, const long dims[D], float* dst, const complex float* src)
{
	md_imag2(D, dims, MD_STRIDES(D, dims, FL_SIZE), dst, MD_STRIDES(D, dims, CFL_SIZE), src);
}

void md_zcmpl_real2(unsigned int D, const long dims[D], const long ostrs[D], complex float* dst, const long istrs[D], const float* src)
{
	md_copy2(D, dims, ostrs, (float*)dst + 0, istrs, src, FL_SIZE);
	md_clear2(D, dims, ostrs, (float*)dst + 1, FL_SIZE);
}

void md_zcmpl_real(unsigned int D, const long dims[D], complex float* dst, const float* src)
{
	md_zcmpl_real2(D, dims, MD_STRIDES(D, dims, CFL_SIZE), dst, MD_STRIDES(D, dims, FL_SIZE), src);
}

void md_zcmpl_imag2(unsigned int D, const long dims[D], const long ostrs[D], complex float* dst, const long istrs[D], const float* src)
{
	md_clear2(D, dims, ostrs, (float*)dst + 0, FL_SIZE);
	md_copy2(D, dims, ostrs, (float*)dst + 1, istrs, src, FL_SIZE);
}

void md_zcmpl_imag(unsigned int D, const long dims[D], complex float* dst, const float* src)
{
	md_zcmpl_imag2(D, dims, MD_STRIDES(D, dims, CFL_SIZE), dst, MD_STRIDES(D, dims, FL_SIZE), src);
}

void md_zcmpl2(unsigned int D, const long dims[D], const long ostr[D], complex float* dst, const long istr1[D], const float* src_real, const long istr2[D], const float* src_imag)
{
	md_copy2(D, dims, ostr, (float*)dst + 0, istr1, src_real, FL_SIZE);
	md_copy2(D, dims, ostr, (float*)dst + 1, istr2, src_imag, FL_SIZE);
}

extern void md_zcmpl(unsigned int D, const long dims[D], complex float* dst, const float* src_real, const float* src_imag)
{
	md_zcmpl2(D, dims, MD_STRIDES(D, dims, CFL_SIZE), dst, MD_STRIDES(D, dims, FL_SIZE), src_real, MD_STRIDES(D, dims, FL_SIZE), src_imag);
}
