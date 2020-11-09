/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/nltest.h"

#include "nn/activation.h"
#include "nn/batchnorm.h"
#include "nn/layers.h"
#include "nn/nn_ops.h"

#include "utest.h"


static bool test_nlop_relu_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* relu = nlop_relu_create(N, dims);

	double err = nlop_test_derivative(relu);

	nlop_free(relu);

	UT_ASSERT(err < 1.E-2);
}



//UT_REGI STER_TEST(test_nlop_relu_derivative);




static bool test_nlop_relu_der_adj(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* relu = nlop_relu_create(N, dims);

	complex float* dst = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	nlop_apply(relu, N, dims, dst, N, dims, src);

	md_free(src);
	md_free(dst);

	float err = linop_test_adjoint_real(nlop_get_derivative(relu, 0, 0));

	nlop_free(relu);

	UT_ASSERT(err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_relu_der_adj);


static bool test_nlop_softmax_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* softmax = nlop_softmax_create(N, dims, 4);

	double err = nlop_test_derivative(softmax);

	nlop_free(softmax);

	UT_ASSERT(err < 1.E-1);
}



UT_REGISTER_TEST(test_nlop_softmax_derivative);




static bool test_nlop_softmax_der_adj(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* softmax = nlop_softmax_create(N, dims,4);

	complex float* dst = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	nlop_apply(softmax, N, dims, dst, N, dims, src);

	md_free(src);
	md_free(dst);

	float err = linop_test_adjoint_real(nlop_get_derivative(softmax, 0, 0));

	nlop_free(softmax);

	UT_ASSERT(err < 1.E-5);
}



UT_REGISTER_TEST(test_nlop_softmax_der_adj);

static bool test_nlop_sigmoid_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* sigmoid = nlop_sigmoid_create(N, dims);

	double err = nlop_test_derivative(sigmoid);

	nlop_free(sigmoid);

	UT_ASSERT(err < 1.E-1);
}

UT_REGISTER_TEST(test_nlop_sigmoid_derivative);



static bool test_nlop_sigmoid_der_adj(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* sigmoid = nlop_sigmoid_create(N, dims);

	complex float* dst = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	nlop_apply(sigmoid, N, dims, dst, N, dims, src);

	md_free(src);
	md_free(dst);

	float err = linop_test_adjoint_real(nlop_get_derivative(sigmoid, 0, 0));

	nlop_free(sigmoid);

	UT_ASSERT(err < 1.E-5);
}

UT_REGISTER_TEST(test_nlop_sigmoid_der_adj);


static bool test_nlop_stats(void)
{
	enum { N = 2 };
	long idims[N] = { 10, 3 };
	unsigned long flags = MD_BIT(0);
	long odims[N];
	md_select_dims(N, ~flags, odims, idims);

	complex float* src = md_alloc(N, idims, CFL_SIZE);
	complex float* mean = md_alloc(N, odims, CFL_SIZE);
	complex float* var = md_alloc(N, odims, CFL_SIZE);

	md_gaussian_rand(N, idims, src);
	auto nlop = nlop_stats_create(N, idims, MD_BIT(0));
	nlop_generic_apply_unchecked(nlop, 3, MAKE_ARRAY((void*)mean, (void*)var, (void*)src));

	complex float* mean2 = md_alloc(N, odims, CFL_SIZE);
	complex float* var2 = md_alloc(N, odims, CFL_SIZE);

	md_zavg(N, idims, flags, mean2, src);
	md_zvar(N, idims, flags, var2, src);

	float scale = md_calc_size(N, idims) - md_calc_size(N, odims);
	scale = scale / md_calc_size(N, idims);
	md_zsmul(N, odims, var2, var2, scale); // 1/N vs 1/(N-1);

	float err = md_znrmse(N, odims, mean2, mean);
	err += md_znrmse(N, odims, var2, var);

	float err_adj = nlop_test_adj_derivatives(nlop, true);
	float err_der = nlop_test_derivatives(nlop);

	debug_printf(DP_DEBUG1, "Stats: Error: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",md_znrmse(N, odims, mean2, mean), md_znrmse(N, odims, var2, var), err_der, err_adj);

	md_free(mean);
	md_free(var);
	md_free(mean2);
	md_free(var2);
	md_free(src);

	nlop_free(nlop);


	UT_ASSERT((err < 1.e-6) && (err_der < 5.e-3) && (err_adj < 1.e-6));
}



UT_REGISTER_TEST(test_nlop_stats);

static bool test_nlop_normalize(void)
{
	enum { N = 2 };
	long idims[N] = { 10, 3 };
	unsigned long flags = MD_BIT(0);
	long odims[N];
	md_select_dims(N, ~flags, odims, idims);

	complex float* src = md_alloc(N, idims, CFL_SIZE);
	complex float* dst = md_alloc(N, idims, CFL_SIZE);
	complex float* mean = md_alloc(N, odims, CFL_SIZE);
	complex float* var = md_alloc(N, odims, CFL_SIZE);
	complex float* var2 = md_alloc(N, odims, CFL_SIZE);
	md_zfill(N, odims, var2, 1.);

	md_gaussian_rand(N, idims, src);

	auto nlop_stats = nlop_stats_create(N, idims, MD_BIT(0));
	nlop_generic_apply_unchecked(nlop_stats, 3, MAKE_ARRAY((void*)mean, (void*)var, (void*)src));

	auto nlop_normalize = nlop_normalize_create(N, idims, MD_BIT(0), 0.);
	nlop_generic_apply_unchecked(nlop_normalize, 4, MAKE_ARRAY((void*)dst, (void*)src, (void*)mean, (void*)var));

	//test mean / var after normalization
	nlop_generic_apply_unchecked(nlop_stats, 3, MAKE_ARRAY((void*)mean, (void*)var, (void*)dst));
	float err = md_zrms(N, odims, mean);
	err += md_znrmse(N, odims, var, var2);

	auto nlop = nlop_chain2_FF(nlop_stats, 1, nlop_normalize, 2); // the variance input of nlop_normalize must be positive
	float err_der = nlop_test_derivatives(nlop);
	float err_adj = nlop_test_adj_derivatives(nlop, true);

	debug_printf(DP_DEBUG1, "Normalize: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",
			md_zrms(N, odims, mean), md_znrmse(N, odims, var, var2), err_der, err_adj);

	md_free(mean);
	md_free(var);
	md_free(var2);
	md_free(src);
	md_free(dst);

	nlop_free(nlop);

	UT_ASSERT((err < 5.e-7) && (err_der < 5.e-3) && (err_adj < 1.e-6));
}

UT_REGISTER_TEST(test_nlop_normalize);

static bool test_nlop_shift_and_scale(void)
{
	enum { N = 2 };
	long idims[N] = { 10, 3 };
	unsigned long flags = MD_BIT(0);
	long odims[N];
	md_select_dims(N, ~flags, odims, idims);

	complex float* src = md_alloc(N, idims, CFL_SIZE);
	complex float* dst = md_alloc(N, idims, CFL_SIZE);
	complex float* mean = md_alloc(N, odims, CFL_SIZE);
	complex float* mean2 = md_alloc(N, odims, CFL_SIZE);
	complex float* var = md_alloc(N, odims, CFL_SIZE);
	complex float* var2 = md_alloc(N, odims, CFL_SIZE);
	md_zfill(N, odims, var2, 1.);

	md_gaussian_rand(N, idims, src);

	auto nlop_stats = nlop_stats_create(N, idims, MD_BIT(0));
	auto nlop_normalize = nlop_normalize_create(N, idims, MD_BIT(0), 0.);

	auto nlop = nlop_chain2(nlop_stats, 0, nlop_normalize, 1);
	nlop = nlop_link_F(nlop, 1, 1);
	nlop = nlop_dup_F(nlop, 0, 1);
	nlop_apply(nlop, N, idims, src, N, idims, src); //src is normalized
	nlop_free(nlop);

	md_gaussian_rand(N, odims, var);
	md_gaussian_rand(N, odims, mean);

	auto nlop_renormalize = nlop_scale_and_shift_create(N, idims, MD_BIT(0));

	nlop_generic_apply_unchecked(nlop_renormalize, 4, MAKE_ARRAY((void*)dst, (void*)src, (void*)mean, (void*)var));
	nlop_generic_apply_unchecked(nlop_stats, 3, MAKE_ARRAY((void*)mean2, (void*)var2, (void*)dst));
	md_zmulc(N, odims, var, var, var);

	float err = md_znrmse(N, odims, mean, mean2);
	err += md_znrmse(N, odims, var, var2);

	float err_der = nlop_test_derivatives(nlop_renormalize);
	float err_adj = nlop_test_adj_derivatives(nlop_renormalize, false);

	debug_printf(DP_DEBUG1, "Shift and Scale: Error: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",
			md_znrmse(N, odims, mean, mean2), md_znrmse(N, odims, var, var2), err_der, err_adj);

	md_free(mean);
	md_free(var);
	md_free(mean2);
	md_free(var2);
	md_free(src);
	md_free(dst);

	nlop_free(nlop_stats);
	nlop_free(nlop_normalize);
	nlop_free(nlop_renormalize);

	UT_ASSERT((err < 5.e-7) && (err_der < 5.e-3) && (err_adj < 1.e-6));
}

UT_REGISTER_TEST(test_nlop_shift_and_scale);

static bool test_nlop_bn(void)
{
	enum { N = 2 };
	long idims[N] = { 10, 3 };

	auto nlop = nlop_batchnorm_create(N, idims, MD_BIT(0), 0, STAT_TRAIN);
	const long* statdims = nlop_generic_codomain(nlop, 1)->dims;
	complex float* tmp = md_alloc(N + 1, statdims, CFL_SIZE);

	nlop = nlop_set_input_const_F(nlop, 1, N + 1, statdims, true, tmp);
	nlop = nlop_del_out_F(nlop, 1);

	float err_adj = nlop_test_adj_derivatives(nlop, true);
	float err_der = nlop_test_derivatives(nlop);

	debug_printf(DP_DEBUG1, "Batchnorm: Error: der: %.8f, adj: %.8f\n", err_der, err_adj);

	md_free(tmp);

	nlop_free(nlop);


	UT_ASSERT((err_der < 5.e-3) && (err_adj < 1.e-6));
}

UT_REGISTER_TEST(test_nlop_bn);

static bool test_zmax(void)
{
	unsigned int N = 3;

	long indims[] = {2, 2, 1};
	long outdims[] = {2, 2, 4};

	complex float stacked[] = {	1., 2., 3., 3.,
					2., 2., 4., 2.,
					2., 1., 4., 3.,
					1., 1., 1., 1.};
	complex float zmax[] = {2., 2., 4., 3.};

	const struct nlop_s* zmax_op = nlop_zmax_create(N, outdims, 4);
	complex float* output_zmax = md_alloc(N, indims, CFL_SIZE);
	nlop_generic_apply_unchecked(zmax_op, 2, (void*[]){output_zmax, stacked}); //output, in, mask
	nlop_free(zmax_op);

	float err =  md_zrmse(N, indims, output_zmax, zmax);
	md_free(output_zmax);

	UT_ASSERT(0.01 > err);
}

UT_REGISTER_TEST(test_zmax);

static bool test_pool(void)
{
	unsigned int N = 3;
	long idims[] = {2, 2, 1};
	long pool_size[] = {2, 1, 1};
	long odims[] = {1, 2, 1};
	complex float in[] = {4., 3., 2., 1.};
	complex float pool_exp[] = {4., 2.};
	complex float pool_adj[] = {4., 0., 2., 0.};

	const struct linop_s* pool_op = linop_pool_create(N, idims, pool_size);
	complex float* output_pool = md_alloc(N, odims, CFL_SIZE);
	complex float* adj = md_alloc(N, idims, CFL_SIZE);

	linop_forward(pool_op, N, odims, output_pool, N, idims, in);
	linop_adjoint(pool_op, N, idims, adj, N, odims, output_pool);
	linop_free(pool_op);

	float err = md_zrmse(N, odims, output_pool, pool_exp) + md_zrmse(N, idims, adj, pool_adj);
	md_free(adj);
	md_free(output_pool);

	UT_ASSERT( 0.01 > err); //
}

UT_REGISTER_TEST(test_pool);