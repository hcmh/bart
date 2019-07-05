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

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"

#include "utest.h"





static bool test_iter_irgnm0(bool v2, bool ref)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src1, 1.);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	nlop_apply(zexp, N, dims, dst1, N, dims, src1);

	md_zfill(N, dims, src2, 0.);

	(v2 ? iter4_irgnm2 : iter4_irgnm)(CAST_UP(&iter3_irgnm_defaults), zexp,
		2 * md_calc_size(N, dims), (float*)src2, ref ? (const float*)src1 : NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, NULL,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src2, src1);

	nlop_free(zexp);

	md_free(src1);
	md_free(dst1);
	md_free(src2);

	UT_ASSERT(err < (ref ? 1.E-7 : 0.01));
}



static bool test_iter_irgnm_lsqr0(bool ref)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims, CFL_SIZE);
	complex float* src3 = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src1, 1.);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	nlop_apply(zexp, N, dims, dst1, N, dims, src1);

	md_zfill(N, dims, src2, 0.);
	md_zfill(N, dims, src3, 0.);

	const struct operator_p_s* lsqr = NULL;

	iter4_irgnm2(CAST_UP(&iter3_irgnm_defaults), zexp,
		2 * md_calc_size(N, dims), (float*)src2, ref ? (const float*)src1 : NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, lsqr,
		(struct iter_op_s){ NULL, NULL });

	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = 100;
	conf.l2lambda = 1.;
	conf.tol = 0.1;

	lsqr = lsqr2_create(&lsqr_defaults,
				iter2_conjgrad, CAST_UP(&conf),
				NULL, &zexp->derivative[0][0], NULL,
				0, NULL, NULL, NULL);

	iter4_irgnm2(CAST_UP(&iter3_irgnm_defaults), zexp,
		2 * md_calc_size(N, dims), (float*)src3, ref ? (const float*)src1 : NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, lsqr,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src2, src3);

	nlop_free(zexp);

	md_free(src1);
	md_free(dst1);
	md_free(src2);
	md_free(src3);

	UT_ASSERT(err < 1.E-10);
}



static bool test_iter_irgnm(void)
{
	return    test_iter_irgnm0(false, false)
	       && test_iter_irgnm0(false, true);
}

UT_REGISTER_TEST(test_iter_irgnm);

static bool test_iter_irgnm2(void)
{
	return    test_iter_irgnm0(true, false)
	       && test_iter_irgnm0(true, true);
}

UT_REGISTER_TEST(test_iter_irgnm2);

static bool test_iter_irgnm_lsqr(void)
{
	return    test_iter_irgnm_lsqr0(false)
	       && test_iter_irgnm_lsqr0(true);
}

UT_REGISTER_TEST(test_iter_irgnm_lsqr);


