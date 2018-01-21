/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/nltest.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static bool test_nlop_zexp(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float* dst1 = md_alloc(N, dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, dims, CFL_SIZE);
	complex float* src = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, src);

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	md_zexp(N, dims, dst1, src);

	nlop_apply(zexp, N, dims, dst2, N, dims, src);

	double err = md_znrmse(N, dims, dst2, dst1);

	nlop_free(zexp);

	md_free(src);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}



UT_REGISTER_TEST(test_nlop_zexp);






static bool test_nlop_zexp_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* zexp = nlop_zexp_create(N, dims);

	double err = nlop_test_derivative(zexp);

	nlop_free(zexp);

	return (err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_zexp_derivative);


