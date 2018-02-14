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

#include "misc/misc.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/tenmul.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"

#include "utest.h"







static bool test_nlop_cast_pos(void)
{
	bool ok = true;
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct linop_s* l = linop_identity_create(N, dims);
	struct nlop_s* d = nlop_from_linop(l);

	if (l == linop_from_nlop(d)) // maybe just require != NULL ?
		ok = false;

	linop_free(l);
	nlop_free(d);

	return ok;
}



UT_REGISTER_TEST(test_nlop_cast_pos);



static bool test_nlop_cast_neg(void)
{
	bool ok = true;
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* d = nlop_zexp_create(N, dims);

	if (NULL != linop_from_nlop(d))
		ok = false;

	nlop_free(d);

	return ok;
}



UT_REGISTER_TEST(test_nlop_cast_neg);








static bool test_nlop_chain(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float val = 2.;
	struct nlop_s* diag = nlop_from_linop(linop_cdiag_create(N, dims, 0, &val));
	struct nlop_s* zexp = nlop_zexp_create(N, dims);
	struct nlop_s* zexp2 = nlop_chain(zexp, diag);

	double err = nlop_test_derivative(zexp2);

	nlop_free(zexp2);
	nlop_free(zexp);

	return (err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_chain);




static bool test_nlop_tenmul(void)
{
	enum { N = 3 };
	long odims[N] = { 10, 1, 3 };
	long idims1[N] = { 1, 7, 3 };
	long idims2[N] = { 10, 7, 1 };

	complex float* dst1 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, odims, CFL_SIZE);
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(N, idims2, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);
	md_gaussian_rand(N, idims2, src2);

	struct nlop_s* tenmul = nlop_tenmul_create(N, odims, idims1, idims2);

	md_ztenmul(N, odims, dst1, idims1, src1, idims2, src2);

	nlop_generic_apply_unchecked(tenmul, 3, (void*[]){ dst2, src1, src2 });

	double err = md_znrmse(N, odims, dst2, dst1);

	nlop_free(tenmul);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}



UT_REGISTER_TEST(test_nlop_tenmul);





static bool test_nlop_tenmul_der(void)
{
	enum { N = 3 };
	long odims[N] = { 10, 1, 3 };
	long idims1[N] = { 1, 7, 3 };
	long idims2[N] = { 10, 7, 1 };

	complex float* dst1 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, odims, CFL_SIZE);
	complex float* dst3 = md_alloc(N, odims, CFL_SIZE);
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);
	complex float* src2 = md_alloc(N, idims2, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);
	md_gaussian_rand(N, idims2, src2);

	struct nlop_s* tenmul = nlop_tenmul_create(N, odims, idims1, idims2);

	nlop_generic_apply_unchecked(tenmul, 3, (void*[]){ dst1, src1, src2 });

	const struct linop_s* der1 = nlop_get_derivative(tenmul, 0, 0);
	const struct linop_s* der2 = nlop_get_derivative(tenmul, 0, 1);

	linop_forward(der1, N, odims, dst2, N, idims1, src1);
	linop_forward(der2, N, odims, dst3, N, idims2, src2);
	

	double err = md_znrmse(N, odims, dst2, dst1);
		   + md_znrmse(N, odims, dst3, dst1);

	nlop_free(tenmul);

	md_free(src1);
	md_free(src2);
	md_free(dst1);
	md_free(dst2);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_nlop_tenmul_der);




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


