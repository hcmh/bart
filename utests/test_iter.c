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





static bool test_iter_irgnm(void)
{
	return test_iter_irgnm0(false, false);
}

UT_REGISTER_TEST(test_iter_irgnm);

static bool test_iter_irgnm_ref(void)
{
	return test_iter_irgnm0(false, true);
}


UT_REGISTER_TEST(test_iter_irgnm_ref);

static bool test_iter_irgnm2(void)
{
	return test_iter_irgnm0(true, false);
}

UT_REGISTER_TEST(test_iter_irgnm2);

static bool test_iter_irgnm2_ref(void)
{
	return test_iter_irgnm0(true, true);
}

UT_REGISTER_TEST(test_iter_irgnm2_ref);





