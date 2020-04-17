/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/rand.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "utest.h"



static bool test_thresh(void)
{
	enum { N = 3 };
	long dims[N] = { 4, 2, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src, 1.);

	auto p = prox_thresh_create(N, dims, 0.5, 0u);

	operator_p_apply(p, 0.5, N, dims, dst, N, dims, src);

	operator_p_free(p);

	md_zfill(N, dims, src, 0.75);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-10);
}

UT_REGISTER_TEST(test_thresh);




static bool test_auto_norm(void)
{
	enum { N = 3 };
	long dims[N] = { 2, 4, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src, 3.);

	auto p = prox_thresh_create(N, dims, 0.5, 0u);
	auto n = op_p_auto_normalize(p, MD_BIT(1));

	operator_p_free(p);

	operator_p_apply(n, 0.5, N, dims, dst, N, dims, src);

	operator_p_free(n);

	md_zfill(N, dims, src, 3. * 0.5);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

#ifdef  __clang__
	UT_ASSERT(err < 1.E-6);
#else
#if __GNUC__ >= 10
	UT_ASSERT(err < 1.E-7);
#else
	UT_ASSERT(err < 1.E-10);
#endif
#endif
}

UT_REGISTER_TEST(test_auto_norm);


static bool test_nonneg(void)
{
	enum { N = 3 };
	long dims[N] = { 2, 4, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);
	md_zreal(N, dims, src, src);

	auto p = prox_nonneg_create(N, dims);

	operator_p_apply(p, 0., N, dims, dst, N, dims, src);

	operator_p_free(p);

	md_zsmax(N, dims, src, src, 0.);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

#ifdef  __clang__
	UT_ASSERT(err < 1.E-6);
#else
#if __GNUC__ >= 10
	UT_ASSERT(err < 1.E-7);
#else
	UT_ASSERT(err < 1.E-10);
#endif
#endif
}
UT_REGISTER_TEST(test_nonneg);


static bool test_zsmax(void)
{
	enum { N = 3 };
	long dims[N] = { 2, 4, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	float lower_bound = 0.1;

	auto p = prox_zsmax_create(N, dims, lower_bound);

	operator_p_apply(p, lower_bound, N, dims, dst, N, dims, src);

	operator_p_free(p);

	md_zsmax(N, dims, src, src, lower_bound);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

#ifdef  __clang__
	UT_ASSERT(err < 1.E-6);
#else
#if __GNUC__ >= 10
	UT_ASSERT(err < 1.E-7);
#else
	UT_ASSERT(err < 1.E-10);
#endif
#endif
}

UT_REGISTER_TEST(test_zsmax);


static bool test_op_pre_chain(void)
{
	enum { N = 3 };
	long dims[N] = { 2, 4, 3 };

	complex float* src = md_alloc(N, dims, CFL_SIZE);
	complex float* dst = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	float lower_bound = 0.1;

	auto p1 = operator_identity_create(N, dims);
	auto p2 = prox_zsmax_create(N, dims, lower_bound);

	auto p = operator_p_pre_chain(p1, p2); 

	operator_p_apply(p, lower_bound, N, dims, dst, N, dims, src);

	operator_free(p1);
	operator_p_free(p2);
	operator_p_free(p);

	md_zsmax(N, dims, src, src, lower_bound);

	float err = md_znrmse(N, dims, dst, src);

	md_free(src);
	md_free(dst);

	#ifdef __clang__
	UT_ASSERT(err < 1.E-6);
	#else
	#if __GNUC__ >= 10
	UT_ASSERT(err < 1.E-7);
	#else
	UT_ASSERT(err < 1.E-10);
	#endif
	#endif
}

UT_REGISTER_TEST(test_op_pre_chain);

static bool test_op_stack(void)
{
	enum { N = 3 };
	long dims1[N] = { 2, 4, 3 };
	long dims2[N] = { 2, 4, 1 };
	long dims3[N] = { 2, 4, 4 };

	long strs[N];
	md_calc_strides(N, strs, dims1, CFL_SIZE);

	complex float* src = md_alloc(N, dims3, CFL_SIZE);
	complex float* dst = md_alloc(N, dims3, CFL_SIZE);

	md_gaussian_rand(N, dims3, src);
	
	float lower_bound = 0.1;

	auto p1 = prox_zero_create(N, dims1);
	auto p2 = prox_zsmax_create(N, dims2, lower_bound);

	auto p = operator_p_stack(2, 2, p1, p2); 

	operator_p_apply(p, lower_bound, N, dims3, dst, N, dims3, src);

	operator_p_free(p1);
	operator_p_free(p2);
	operator_p_free(p);

	md_zsmax(N, dims2, (void*)src + strs[2] * dims1[2], (void*)src + strs[2] * dims1[2], lower_bound);

	float err = md_znrmse(N, dims3, dst, src);

	md_free(src);
	md_free(dst);

	#ifdef __clang__
	UT_ASSERT(err < 1.E-6);
	#else
	#if __GNUC__ >= 10
	UT_ASSERT(err < 1.E-7);
	#else
	UT_ASSERT(err < 1.E-10);
	#endif
	#endif
}

UT_REGISTER_TEST(test_op_stack);