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

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter4.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"

#include "utest.h"








static bool test_iter_irgnm(void)
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

	iter4_irgnm(CAST_UP(&iter3_irgnm_defaults), zexp,
		2 * md_calc_size(N, dims), (float*)src2, NULL,
		2 * md_calc_size(N, dims), (const float*)dst1, NULL,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src2, src1);

	nlop_free(zexp);

	md_free(src1);
	md_free(dst1);
	md_free(src2);

	UT_ASSERT(err < 0.01);
}



UT_REGISTER_TEST(test_iter_irgnm);




static bool test_iter_irgnm_ref(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, dims, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, src1, 1.);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	nlop_apply(zexp, N, dims, dst1, N, dims, src1);

	struct iter3_irgnm_conf conf = iter3_irgnm_defaults;

	md_zfill(N, dims, src2, 0.);

	iter4_irgnm(CAST_UP(&conf), zexp,
		2 * md_calc_size(N, dims), (float*)src2,  (const float*)src1,
		2 * md_calc_size(N, dims), (const float*)dst1, NULL,
		(struct iter_op_s){ NULL, NULL });

	double err = md_znrmse(N, dims, src2, src1);

	nlop_free(zexp);

	md_free(src1);
	md_free(dst1);
	md_free(src2);

	UT_ASSERT(err < 1.E-07);
}



UT_REGISTER_TEST(test_iter_irgnm_ref);



