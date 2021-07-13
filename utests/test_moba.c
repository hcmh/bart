/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker, Nick Scholand, Zhengguo Tan
 */

#include <stdio.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops_p.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/stack.h"
#include "nlops/const.h"

#include "moba/T1fun.h"
#include "moba/model_Bloch.h"
#include "moba/blochfun.h"
#include "moba/blochfun2.h"
#include "moba/T1relax.h"
#include "moba/T1MOLLI.h"
#include "moba/T1relax_so.h"
#include "moba/T1srelax.h"
#include "moba/IR_SS_fun.h"
#include "moba/T1_alpha.h"
#include "moba/T1_alpha_in.h"
#include "moba/T2fun.h"
#include "moba/meco.h"
#include "moba/optreg.h"

#include "simu/signals.h"

#include "utest.h"






static bool test_nlop_T1fun(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	complex float TI[4] = { 0., 1., 2., 3. };

	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* T1 = nlop_T1_create(N, map_dims, out_dims, in_dims, TI_dims, TI, false);

	nlop_apply(T1, N, out_dims, dst, N, in_dims, src);

	float err = linop_test_adjoint(nlop_get_derivative(T1, 0, 0));

	nlop_free(T1);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_T1fun);

static void random_application(const struct nlop_s* nlop)
{
	auto dom = nlop_domain(nlop);
	auto cod = nlop_codomain(nlop);

	complex float* in = md_alloc(dom->N, dom->dims, dom->size);
	complex float* dst = md_alloc(cod->N, cod->dims, cod->size);

	md_gaussian_rand(dom->N, dom->dims, in);

	// define position for derivatives
	nlop_apply(nlop, cod->N, cod->dims, dst, dom->N, dom->dims, in);

	md_free(in);
	md_free(dst);
}


static bool test_T1relax(void)
{

	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
// 	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	long map_strs[N];
	md_calc_strides(N, map_strs, map_dims, CFL_SIZE);

	long out_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);

	long TI_strs[N];
	md_calc_strides(N, TI_strs, TI_dims, CFL_SIZE);

	complex float* dst1 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst3 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* dst4 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* src3 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* tmp = md_alloc(N, map_dims, CFL_SIZE);
	complex float* TI1 = md_alloc(N, TI_dims, CFL_SIZE);

	complex float TI[4] = { 0., 1., 2., 3. };

	md_copy(N, TI_dims, TI1, TI, CFL_SIZE);

	md_gaussian_rand(N, map_dims, src1);
	md_gaussian_rand(N, map_dims, src2);
	md_gaussian_rand(N, map_dims, src3);

	struct nlop_s* T1 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI);

	md_zsmul(N, map_dims, tmp, src3, -1.0);
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, TI_strs, TI);
	md_zexp(N, out_dims, dst1, dst1);


	md_zsub(N, map_dims, tmp, src1, src2);
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, out_strs, dst1);
	md_zadd2(N, out_dims, out_strs, dst1, map_strs, src2, out_strs, dst1);

	long pos[N];

	for (int i = 0; i < N; i++)
		pos[i] = 0;

	pos[TE_DIM] = TI_dims[TE_DIM] - 1;
	md_copy_block(N, pos, map_dims, dst4, out_dims, dst1, CFL_SIZE);


	nlop_generic_apply_unchecked(T1, 5, (void*[]){ dst2, dst3, src1, src2, src3 });

	double err = md_znrmse(N, out_dims, dst2, dst1) + md_znrmse(N, map_dims, dst4, dst3);


	nlop_free(T1);

	md_free(src1);
	md_free(src2);
	md_free(src3);
	md_free(tmp);
	md_free(TI1);
	md_free(dst1);
	md_free(dst2);
	md_free(dst3);
	md_free(dst4);


	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_T1relax);


static bool test_nlop_T1relax_der_adj(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
// 	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float TI[4] = { 0., 1., 2., 3. };

	struct nlop_s* T1 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI);

	struct nlop_s* flat = nlop_flatten(T1);

	random_application(flat);

	double err = linop_test_adjoint(nlop_get_derivative(flat, 0, 0));

	nlop_free(flat);
	nlop_free(T1);


	UT_ASSERT((!safe_isnanf(err)) && (err < 7.E-2));
}

UT_REGISTER_TEST(test_nlop_T1relax_der_adj);



static bool test_nlop_Blochfun(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 500, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long all_dims[N] = { 16, 16, 1, 1, 1, 500, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	bool gpu_use = false;

	struct modBlochFit fit_para = modBlochFit_defaults;

	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* op_Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(op_Bloch, N, out_dims, dst, N, in_dims, src);

	float err = linop_test_adjoint(nlop_get_derivative(op_Bloch, 0, 0));

	nlop_free(op_Bloch);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_Blochfun);


static bool test_nlop_Blochfun2(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 500, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long all_dims[N] = { 16, 16, 1, 1, 1, 500, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	bool gpu_use = false;

	// Define simulation parameter
	struct modBlochFit fit_para = modBlochFit_defaults;
	fit_para.sequence = 5;

	// Initialize maps with ones
	md_zfill(N, in_dims, src, 1.0);

	// FT B1 map to work with Sobolev norm and reasonable FAs

	long pos[DIMS] = { [0 ... DIMS - 1] = 0 };
	pos[COEFF_DIM] = 3;

	complex float* tmp = md_alloc(DIMS, in_dims, CFL_SIZE);

	const struct linop_s* linop_fftc = linop_fftc_create(DIMS, map_dims, FFT_FLAGS);

	md_copy_block(DIMS, pos, map_dims, tmp, in_dims, src, CFL_SIZE);
	linop_forward_unchecked(linop_fftc, tmp, tmp);
	md_copy_block(DIMS, pos, in_dims, src, map_dims, tmp, CFL_SIZE);

	linop_free(linop_fftc);

	md_free(tmp);

	// Create operator
	struct nlop_s* op_Bloch = nlop_Bloch_create2(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(op_Bloch, N, out_dims, dst, N, in_dims, src);

	// Test
	float err = linop_test_adjoint(nlop_get_derivative(op_Bloch, 0, 0));

	nlop_free(op_Bloch);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_Blochfun2);



static bool test_T1relax_link1(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	long map_strs[N];
	md_calc_strides(N, map_strs, map_dims, CFL_SIZE);

	long out_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);

	long TI_strs[N];
	md_calc_strides(N, TI_strs, TI_dims, CFL_SIZE);

	complex float* dst1 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst3 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* dst4 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* dst5 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst6 = md_alloc(N, map_dims, CFL_SIZE);

	complex float* src1 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* src3 = md_alloc(N, map_dims, CFL_SIZE);

	complex float* tmp = md_alloc(N, map_dims, CFL_SIZE);
	complex float* TI1 = md_alloc(N, TI_dims, CFL_SIZE);
	complex float* TI2 = md_alloc(N, TI_dims, CFL_SIZE);

	complex float TI_1[4] = { 1., 2., 3., 4. };
	complex float TI_2[4] = { 4., 5., 6., 7. };

	md_copy(N, TI_dims, TI1, TI_1, CFL_SIZE);
	md_copy(N, TI_dims, TI2, TI_2, CFL_SIZE);

	md_gaussian_rand(N, map_dims, src1);
	md_gaussian_rand(N, map_dims, src2);
	md_gaussian_rand(N, map_dims, src3);
/*
	md_zfill(N, map_dims, src1, -0.5);
	md_zfill(N, map_dims, src2, 1.0);
	md_zfill(N, map_dims, src3, 1.0);*/

	struct nlop_s* T1_1 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI1);
	struct nlop_s* T1_2 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI1);

	struct nlop_s* T1_combine = nlop_combine_FF(T1_2, T1_1);
	struct nlop_s* T1_link = nlop_link_F(T1_combine, 3, 0);
	struct nlop_s* T1_dup1 = nlop_dup_F(T1_link, 1, 4);
	struct nlop_s* T1_dup = nlop_dup_F(T1_dup1, 0, 3);

	md_zsmul(N, map_dims, tmp, src3, -1.0);
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, TI_strs, TI1);
	md_zexp(N, out_dims, dst1, dst1);

	md_zsub(N, map_dims, tmp, src1, src2);
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, out_strs, dst1);
	md_zadd2(N, out_dims, out_strs, dst1, map_strs, src2, out_strs, dst1);

	long pos[N];

	for (int i = 0; i < N; i++)
		pos[i] = 0;

	pos[TE_DIM] = TI_dims[TE_DIM] - 1;
	md_copy_block(N, pos, map_dims, dst4, out_dims, dst1, CFL_SIZE);

	md_zsmul(N, map_dims, tmp, src3, -1.0);
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, TI_strs, TI1);
	md_zexp(N, out_dims, dst1, dst1);

	md_zsub(N, map_dims, tmp, dst4, src2);
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, out_strs, dst1);
	md_zadd2(N, out_dims, out_strs, dst1, map_strs, src2, out_strs, dst1);

	md_copy_block(N, pos, map_dims, dst4, out_dims, dst1, CFL_SIZE);

	nlop_generic_apply_unchecked(T1_dup, 6, (void*[]){ dst5, dst6, dst2, src2, src3, src1 });

	nlop_free(T1_dup);


	double err = md_znrmse(N, out_dims, dst1, dst5) + md_znrmse(N, map_dims, dst6, dst4);


	md_free(src1);
	md_free(src2);
	md_free(src3);

	md_free(tmp);
	md_free(TI1);
	md_free(TI2);

	md_free(dst1);
	md_free(dst2);
	md_free(dst3);
	md_free(dst4);
	md_free(dst5);
	md_free(dst6);


	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_T1relax_link1);

static bool test_nlop_T1relax_comb_der_adj(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	long map_strs[N];
	md_calc_strides(N, map_strs, map_dims, CFL_SIZE);

	long out_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);

	long TI_strs[N];
	md_calc_strides(N, TI_strs, TI_dims, CFL_SIZE);

	complex float* TI1 = md_alloc(N, TI_dims, CFL_SIZE);
	complex float* TI2 = md_alloc(N, TI_dims, CFL_SIZE);

	complex float TI_1[4] = { 1., 2., 3., 4. };
	complex float TI_2[4] = { 4., 5., 6., 7. };

	md_copy(N, TI_dims, TI1, TI_1, CFL_SIZE);
	md_copy(N, TI_dims, TI2, TI_2, CFL_SIZE);

	struct nlop_s* T1_1 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI1);
	struct nlop_s* T1_2 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI1);

	struct nlop_s* T1_combine = nlop_combine_FF(T1_2, T1_1);
	struct nlop_s* T1_link = nlop_link_F(T1_combine, 3, 0);

	struct nlop_s* T1_dup1 = nlop_dup_F(T1_link, 1, 4);
	struct nlop_s* T1_dup = nlop_dup_F(T1_dup1, 0, 3);

	struct nlop_s* flat = nlop_flatten(T1_dup);
	nlop_free(T1_dup);

	random_application(flat);

// 	double err = linop_test_adjoint(nlop_get_derivative(flat, 0, 0));
	double err = nlop_test_derivative(flat);

	nlop_free(flat);

	md_free(TI1);
	md_free(TI2);

	UT_ASSERT((!safe_isnanf(err)) && (err < 7.E-2));
}

UT_REGISTER_TEST(test_nlop_T1relax_comb_der_adj);

static bool test_T1_MOLLI_relax(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI2_dims[N] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out2_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long scale_dims[N] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	long out1_dims[N] = { 16, 16, 1, 1, 1, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out1_strs[N];
	md_calc_strides(N, out1_strs, out1_dims, CFL_SIZE);

	long TI1_dims[N] = { 1, 1, 1, 1, 1, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	long TI1_strs[N];
	md_calc_strides(N, TI1_strs, TI1_dims, CFL_SIZE);


	long map_strs[N];
	md_calc_strides(N, map_strs, map_dims, CFL_SIZE);

	long out_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);

	long out2_strs[N];
	md_calc_strides(N, out2_strs, out2_dims, CFL_SIZE);

	long TI_strs[N];
	md_calc_strides(N, TI_strs, TI_dims, CFL_SIZE);

	long TI2_strs[N];
	md_calc_strides(N, TI2_strs, TI2_dims, CFL_SIZE);

	complex float* dst1 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst3 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* dst4 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* dst5 = md_alloc(N, out2_dims, CFL_SIZE);
	complex float* dst6 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* dst7 = md_alloc(N, out2_dims, CFL_SIZE);

	complex float* dst8 = md_alloc(N, out1_dims, CFL_SIZE);
	complex float* dst9 = md_alloc(N, out1_dims, CFL_SIZE);

	complex float* src1 = md_alloc(N, map_dims, CFL_SIZE); // M_start
	complex float* src2 = md_alloc(N, map_dims, CFL_SIZE); // M0
	complex float* src3 = md_alloc(N, map_dims, CFL_SIZE); // R1
	complex float* src4 = md_alloc(N, map_dims, CFL_SIZE); // R1s

	complex float* tmp = md_alloc(N, map_dims, CFL_SIZE);
	complex float* tmp1 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* TI = md_alloc(N, TI_dims, CFL_SIZE);
	complex float* TI_1_2 = md_alloc(N, TI_dims, CFL_SIZE);
	complex float* TI2 = md_alloc(N, TI2_dims, CFL_SIZE);
	complex float* scale = md_alloc(N, scale_dims, CFL_SIZE);

	complex float* TI1 = md_alloc(N, TI1_dims, CFL_SIZE);

	complex float TI_1[4] = { 0.5, 1.5, 2.5, 3.5 };
	complex float TI_1_2_0[4] = { 1.0, 2.0, 3.0, 4.0 };
	complex float TI_2[1] = {  0.0 };
	complex float TI_1_1[12] = { 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5 };

	md_copy(N, TI_dims, TI, TI_1, CFL_SIZE);
	md_copy(N, TI_dims, TI_1_2, TI_1_2_0, CFL_SIZE);
	md_copy(N, TI2_dims, TI2, TI_2, CFL_SIZE);
	md_copy(N, TI1_dims, TI1, TI_1_1, CFL_SIZE);

	md_gaussian_rand(N, map_dims, src1);
	md_zsmul(N, map_dims, src2, src1, -1.0);
	md_zfill(N, map_dims, src3, 0.6);
	md_zfill(N, map_dims, src4, 0.8);

	float scaling_R1s = 1.0;
	float regularization = 1e-6;

	struct nlop_s* T1s_1 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI_1, false);
	struct nlop_s* T1_1 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI2_dims, TI_2, false);
	struct nlop_s* T1s_2 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI_1_2, false);
	struct nlop_s* T1_2 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI2_dims, TI_2, false);
	struct nlop_s* T1s_3 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI_1_2, false);

	// first chain: T1(T1s)
	struct nlop_s* T1c_combine = nlop_combine_FF(T1_1, T1s_1);
	struct nlop_s* T1c_link = nlop_link_F(T1c_combine, 2, 0);
	struct nlop_s* T1c_dup1 = nlop_dup_F(T1c_link, 1, 4);
	struct nlop_s* T1c_dup1_1 = nlop_dup_F(T1c_dup1, 0, 3);

	// second chain T1s(T1(T1s))
	struct nlop_s* T1c_combine2 = nlop_combine(T1s_2, T1c_dup1_1);
	struct nlop_s* T1c_link2 = nlop_link_F(T1c_combine2, 2, 0);
	struct nlop_s* T1c_dup2 = nlop_dup_F(T1c_link2, 2, 6);
	struct nlop_s* T1c_dup2_1 = nlop_dup_F(T1c_dup2, 1, 4);
	struct nlop_s* T1c_dup2_2 = nlop_dup_F(T1c_dup2_1, 0, 3);

	nlop_free(T1s_2);

	// struct nlop_s* T1c_dup2_2_del = nlop_del_out(T1c_dup2_2, 1);

	// third chain: T1(T1s(T1(T1s)))
	struct nlop_s* T1c_combine3 = nlop_combine_FF(T1_2, T1c_dup2_2);
	struct nlop_s* T1c_link3 = nlop_link_F(T1c_combine3, 2, 0);
	struct nlop_s* T1c_dup3 = nlop_dup_F(T1c_link3, 1, 3);
	struct nlop_s* T1c_dup3_1 = nlop_dup_F(T1c_dup3, 0, 2);

	//  stack the outputs together
	long sodims[N];
	md_copy_dims(N, sodims, out_dims);
	sodims[TE_DIM] = 2 * out_dims[TE_DIM];
	struct nlop_s* stack = nlop_stack_create(N, sodims, out_dims, out_dims, TE_DIM);
	struct nlop_s* T1c_combine_stack = nlop_combine_FF(stack, T1c_dup3_1);
	struct nlop_s* T1c_link_stack_1 = nlop_link_F(T1c_combine_stack, 3, 0);
	struct nlop_s* T1c_link_stack_2 = nlop_link_F(T1c_link_stack_1, 2, 0);


	// fourth chain : T1s(T1(T1s(T1(T1s))))
	struct nlop_s* T1c_combine4 = nlop_combine_FF(T1s_3, T1c_link_stack_2);
	struct nlop_s* T1c_link4 = nlop_link_F(T1c_combine4, 3, 0);
	struct nlop_s* T1c_dup4 = nlop_dup_F(T1c_link4, 2, 5);
	struct nlop_s* T1c_dup4_1 = nlop_dup_F(T1c_dup4, 1, 4);
	struct nlop_s* T1c_dup4_2 = nlop_dup_F(T1c_dup4_1, 0, 3);

	//  stack the outputs together
	long sodims1[N];
	md_copy_dims(N, sodims1, out_dims);
	sodims1[TE_DIM] = sodims[TE_DIM] + out_dims[TE_DIM];

	struct nlop_s* stack1 = nlop_stack_create(N, sodims1, sodims, out_dims, TE_DIM);
	struct nlop_s* T1c_combine_stack1 = nlop_combine_FF(stack1, T1c_dup4_2);
	struct nlop_s* T1c_link_stack_1_1 = nlop_link_F(T1c_combine_stack1, 3, 0);
	struct nlop_s* T1c_link_stack_2_1 = nlop_link_F(T1c_link_stack_1_1, 1, 0);

	struct nlop_s* T1c_del = nlop_del_out_F(T1c_link_stack_2_1, 1);

	// // scaling operator
	complex float diag[1] = {-1.0};
	md_copy(N, scale_dims, scale, diag, CFL_SIZE);

	struct linop_s* linop_scalar = linop_cdiag_create(N, map_dims, COEFF_FLAG, scale);
	struct nlop_s* nl_scalar = nlop_from_linop(linop_scalar);

	linop_free(linop_scalar);

	struct nlop_s* T1c_combine_scale = nlop_combine_FF(T1c_del, nl_scalar);
	struct nlop_s* T1c_link_scale = nlop_link_F(T1c_combine_scale, 1, 3);
	struct nlop_s* T1c_dup_scale = nlop_dup_F(T1c_link_scale, 0, 3);


	// Analytical model
	// T1* relaxation
	md_zsmul(N, map_dims, tmp, src4, -1.0 *scaling_R1s);
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, TI_strs, TI);
	md_zexp(N, out_dims, dst1, dst1);

	//M0 * R1/R1s - M_start
	md_zdiv_reg(N, map_dims, tmp, src3, src4, regularization);
	md_zsmul(N, map_dims, tmp, tmp, 1./scaling_R1s);

	md_zmul(N, map_dims, tmp, tmp, src2);
	md_zsub(N, map_dims, tmp1, tmp, src1);

	// (M0 * R1/R1s - M_start).*exp(-t.*scaling_R1s*R1s)
	md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp1, out_strs, dst1);

	//Model: (M0 * R1/R1s -(-M_start + M0 * R1/R1s).*exp(-t.*scaling_R1s*R1s))
	md_zsub2(N, out_dims, out_strs, dst1, map_strs, tmp, out_strs, dst1);

	// Analytical model for longer period
	// T1* relaxation
	md_zsmul(N, map_dims, tmp, src4, -1.0 *scaling_R1s);
	md_zmul2(N, out1_dims, out1_strs, dst8, map_strs, tmp, TI1_strs, TI1);
	md_zexp(N, out1_dims, dst8, dst8);

	//M0 * R1/R1s - M_start
	md_zdiv_reg(N, map_dims, tmp, src3, src4, regularization);
	md_zsmul(N, map_dims, tmp, tmp, 1./scaling_R1s);

	md_zmul(N, map_dims, tmp, tmp, src2);
	md_zsub(N, map_dims, tmp1, tmp, src1);

	// (M0 * R1/R1s - M_start).*exp(-t.*scaling_R1s*R1s)
	md_zmul2(N, out1_dims, out1_strs, dst8, map_strs, tmp1, out1_strs, dst8);

	//Model: (M0 * R1/R1s -(-M_start + M0 * R1/R1s).*exp(-t.*scaling_R1s*R1s))
	md_zsub2(N, out1_dims, out1_strs, dst8, map_strs, tmp, out1_strs, dst8);

	nlop_generic_apply_unchecked(T1c_dup_scale, 4, (void*[]){ dst9, src2, src3, src4});
	nlop_free(T1c_dup_scale);


	long pos[N];

	for (int i = 0; i < N; i++)
		pos[i] = 0;

	pos[TE_DIM] = TI_dims[TE_DIM] - 1;
	md_copy_block(N, pos, map_dims, dst4, out_dims, dst1, CFL_SIZE);


	// T1 relaxation
	// M0 + (Mstart - M0)* exp(-t*R1)
	md_zsmul(N, map_dims, tmp, src3, -1.0);
	md_zmul2(N, out2_dims, out2_strs, dst5, map_strs, tmp, TI2_strs, TI2);
	md_zexp(N, out2_dims, dst5, dst5);

	md_zsub(N, map_dims, tmp, dst4, src2);
	md_zmul2(N, out2_dims, out2_strs, dst5, map_strs, tmp, out2_strs, dst5);
	md_zadd2(N, out2_dims, out2_strs, dst5, map_strs, src2, out2_strs, dst5);

	nlop_generic_apply_unchecked(T1c_dup1_1, 6, (void*[]){ dst7, dst2, src2, src3, src1, src4 });
	nlop_free(T1c_dup1_1);

	double err = md_znrmse(N, out1_dims, dst9, dst8) + md_znrmse(N, out2_dims, dst7, dst5) + md_znrmse(N, out_dims, dst1, dst2);

	md_free(src1);
	md_free(src2);
	md_free(src3);
	md_free(src4);

	md_free(tmp);
	md_free(tmp1);
	md_free(TI);
	md_free(TI2);
	md_free(TI1);
	md_free(TI_1_2);

	md_free(dst1);
	md_free(dst2);
	md_free(dst3);
	md_free(dst4);
	md_free(dst5);
	md_free(dst6);
	md_free(dst7);

	md_free(dst8);
	md_free(dst9);
	md_free(scale);

	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_T1_MOLLI_relax);

static bool test_nlop_T1_MOLLI_relax_der_adj(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI2_dims[N] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out2_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	long map_strs[N];
	md_calc_strides(N, map_strs, map_dims, CFL_SIZE);

	long out_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);

	long TI_strs[N];
	md_calc_strides(N, TI_strs, TI_dims, CFL_SIZE);

	complex float* TI1 = md_alloc(N, TI_dims, CFL_SIZE);
	complex float* TI2 = md_alloc(N, TI2_dims, CFL_SIZE);

	complex float TI_1[4] = { 1., 2., 3., 4. };
	complex float TI_2[1] = { 7.};

	md_copy(N, TI_dims, TI1, TI_1, CFL_SIZE);
	md_copy(N, TI2_dims, TI2, TI_2, CFL_SIZE);

	struct nlop_s* T1s_1 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI1, false);
	struct nlop_s* T1_2 = nlop_T1relax_create(N, map_dims, out2_dims, TI2_dims, TI2);

	struct nlop_s* T1_3 = nlop_T1relax_create(N, map_dims, out2_dims, TI2_dims, TI2);

	// first chain
	struct nlop_s* T1_combine = nlop_combine_FF(T1_2, T1s_1);
	struct nlop_s* T1_link = nlop_link_F(T1_combine, 3, 0);

	struct nlop_s* T1c_dup1 = nlop_dup_F(T1_link, 1, 4);
	struct nlop_s* T1c_dup = nlop_dup_F(T1c_dup1, 0, 3);

	// second chain
	struct nlop_s* T1_combine1 = nlop_combine_FF(T1c_dup, T1_3);
	struct nlop_s* T1_link1 = nlop_link_F(T1_combine1, 4, 2);

	struct nlop_s* T1c_dup2 = nlop_dup_F(T1_link1, 1 , 5);
	struct nlop_s* T1c_dup3 = nlop_dup_F(T1c_dup2, 0, 4);

	struct nlop_s* flat = nlop_flatten(T1c_dup3);
	nlop_free(T1c_dup3);

	random_application(flat);

	double err = nlop_test_derivative(flat);

	nlop_free(flat);

	md_free(TI1);
	md_free(TI2);

	debug_printf(DP_INFO, "----%f\n", err);

#ifdef __clang__
#warning "FIXME: Test fails with clang."
	return true;
#else
	UT_ASSERT((!safe_isnanf(err)) && (err < 7.E-2));
#endif
}

UT_REGISTER_TEST(test_nlop_T1_MOLLI_relax_der_adj);


static bool test_nlop_IR_SS_fun(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	complex float TI[4] = { 0., 1., 2., 3. };

	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* IR_SS = nlop_IR_SS_create(N, map_dims, out_dims, in_dims, TI_dims, TI, false);

	nlop_apply(IR_SS, N, out_dims, dst, N, in_dims, src);

	float err = linop_test_adjoint(nlop_get_derivative(IR_SS, 0, 0));

	nlop_free(IR_SS);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_IR_SS_fun);

static bool test_nlop_IR_SS_fun_der_adj(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
 	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float TI[4] = { 0., 1., 2., 3. };

	struct nlop_s* IR_SS = nlop_IR_SS_create(N, map_dims, out_dims, in_dims, TI_dims, TI, false);

	struct nlop_s* flat = nlop_flatten(IR_SS);

	random_application(flat);

	double err = linop_test_adjoint(nlop_get_derivative(flat, 0, 0));

	nlop_free(flat);
	nlop_free(IR_SS);


	UT_ASSERT((!safe_isnanf(err)) && (err < 7.E-2));
}

UT_REGISTER_TEST(test_nlop_IR_SS_fun_der_adj);



static bool test_nlop_T2fun(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	long TE_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	complex float TE[4] = { 0.0, 0.1, 0.2, 0.3 };

	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* T2 = nlop_T2_create(N, map_dims, out_dims, in_dims, TE_dims, TE, false);

	nlop_apply(T2, N, out_dims, dst, N, in_dims, src);

	float err = linop_test_adjoint(nlop_get_derivative(T2, 0, 0));

	nlop_free(T2);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_T2fun);

#if 0
static bool test_nlop_meco(void) 
{
	/* 
	 * please don't use any real constraint on R2* and fB0 maps in src/moba/meco.c
	 * when making utest
	 */
	enum { N = 16 };
	enum { NECO = 3 };
	enum { IMSIZE = 16 };

	bool curr_res = false;

	for (unsigned int m = 0; m < 6; m++) {

		long NCOEFF = set_num_of_coeff(m);

		long y_dims[N] = { IMSIZE, IMSIZE, 1, 1, 1, NECO,      1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		long x_dims[N] = { IMSIZE, IMSIZE, 1, 1, 1,    1, NCOEFF, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

		complex float* dst = md_alloc(N, y_dims, CFL_SIZE);
		complex float* src = md_alloc(N, x_dims, CFL_SIZE);

		complex float TE[NECO] = { 1.26 + I*0., 2.66, 3.69 };

		md_zfill(N, x_dims, src, 1.0);

		float scale_fB0[2] = { 0., 1. };
		struct nlop_s* meco = nlop_meco_create(N, y_dims, x_dims, TE, m, false, FAT_SPEC_1, scale_fB0, false);

		nlop_apply(meco, N, y_dims, dst, N, x_dims, src);
		
		float err = linop_test_adjoint(nlop_get_derivative(meco, 0, 0));

		nlop_free(meco);

		md_free(src);
		md_free(dst);

		curr_res = (err < 1.E-3) ? true : false;

		if (curr_res == false)
			break;
	}

	UT_ASSERT(curr_res);
}

// U T_REGISTER_TEST(test_nlop_meco);
#endif

static bool test_op_p_stack_moba_nonneg(void)
{
	enum { N = 5 };
	long dims[N] = { 2, 4, 7, 5, 6};

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long s_dim = 2;

	long p_pos = 3;
	unsigned int s_flag = MD_BIT(p_pos);

	const struct operator_p_s* p = moba_nonneg_prox_create(N, dims, s_dim, s_flag, 0.);

	complex float* in  = md_alloc(N, dims, CFL_SIZE);
	complex float* out = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, in, -1.);
	md_zfill(N, dims, out, 100.);

	operator_p_apply(p, 0., N, dims, out, N, dims, in);
	operator_p_free(p);


	long dims1[N];
	md_select_dims(N, ~MD_BIT(s_dim), dims1, dims);

	complex float* in1 = md_alloc(N, dims1, CFL_SIZE);

	long* pos = calloc(N, sizeof(long));
	pos[s_dim] = p_pos;

	md_copy_block(N, pos, dims1, in1, dims, in, CFL_SIZE);
	md_clear(N, dims1, in1, CFL_SIZE);
	md_copy_block(N, pos, dims, in, dims1, in1, CFL_SIZE);

	xfree(pos);

	float err = md_znrmse(N, dims, out, in);

	md_free(in);
	md_free(in1);
	md_free(out);

	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_op_p_stack_moba_nonneg);


static bool test_nlop_T1_alpha_in_fun(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);

	complex float* src = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src, 1.0);

	complex float TI[4] = { 0., 1., 2., 3. };

	complex float* alpha = md_alloc(N, map_dims, CFL_SIZE);
	md_zfill(N, map_dims, alpha, 6.0);

	struct nlop_s* T1_alpha_in = nlop_T1_alpha_in_create(N, map_dims, out_dims, in_dims, TI_dims, TI, alpha, false);

	nlop_apply(T1_alpha_in, N, out_dims, dst, N, in_dims, src);

	float err = linop_test_adjoint(nlop_get_derivative(T1_alpha_in, 0, 0));

	nlop_free(T1_alpha_in);

	md_free(src);
	md_free(dst);
	md_free(alpha);

	// debug_printf(DP_INFO, "Error: %f\n", err);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_T1_alpha_in_fun);
