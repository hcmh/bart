/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"

#include "nn/batchnorm.h"


#include "utest.h"



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

	debug_printf(DP_DEBUG1, "Error: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",md_znrmse(N, odims, mean2, mean), md_znrmse(N, odims, var2, var), err_der, err_adj);

	md_free(mean);
	md_free(var);
	md_free(mean2);
	md_free(var2);
	md_free(src);

	nlop_free(nlop);


	UT_ASSERT((err < 1.e-7) && (err_der < 5.e-3) && (err_adj < 1.e-6));
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

	float err_der = nlop_test_derivatives(nlop_normalize);

	auto nlop = nlop_chain2_FF(nlop_stats, 1, nlop_normalize, 2); // the variance input of nlop_normalize must be positive
	float err_adj = nlop_test_adj_derivatives(nlop, true);

	debug_printf(DP_DEBUG1, "Error: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",
			md_zrms(N, odims, mean), md_znrmse(N, odims, var, var2), err_der, err_adj);

	md_free(mean);
	md_free(var);
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

	debug_printf(DP_DEBUG1, "Error: mean: %.8f, var: %.8f, der: %.8f, adj: %.8f\n",
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