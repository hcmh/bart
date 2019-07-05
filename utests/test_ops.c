/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static bool test_op_stack(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };
	long dims2[N] = { 8, 4, 2 };

	const struct operator_s* a = operator_identity_create(N, dims);
	const struct operator_s* b = operator_zero_create(N, dims);
	const struct operator_s* c = operator_null_create(N, dims);
	const struct operator_s* d = operator_combi_create(2, MAKE_ARRAY(b, c));
	const struct operator_s* e = operator_stack(2, 2, a, d);

	complex float* in = md_alloc(N, dims2, CFL_SIZE);
	complex float* out = md_alloc(N, dims2, CFL_SIZE);

	md_zfill(N, dims2, in, 1.);
	md_zfill(N, dims2, out, 100.);

	operator_apply(e, N, dims2, out, N, dims2, in);

	double err = fabsf(md_znorm(N, dims2, in) - sqrtf(2.) * md_znorm(N, dims2, out));

	operator_free(a);
	operator_free(b);
	operator_free(c);
	operator_free(d);
	operator_free(e);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_stack);



static bool test_op_extract(void)
{
	enum { N = 3 };
	long dims[N] = { 8, 4, 1 };
	long dims2[N] = { 8, 4, 2 };

	const struct operator_s* a = operator_identity_create(N, dims);
	const struct operator_s* b = operator_zero_create(N, dims);
	const struct operator_s* c = operator_null_create(N, dims);
	const struct operator_s* d = operator_combi_create(2, MAKE_ARRAY(b, c));
	const struct operator_s* e = operator_extract_create(a, 0, N, dims2, (long[]){ 0, 0, 0 });
	const struct operator_s* f = operator_extract_create(e, 1, N, dims2, (long[]){ 0, 0, 0 });
	const struct operator_s* g = operator_extract_create(d, 0, N, dims2, (long[]){ 0, 0, 1 });
	const struct operator_s* h = operator_extract_create(g, 1, N, dims2, (long[]){ 0, 0, 1 });
	const struct operator_s* i = operator_combi_create(2, MAKE_ARRAY(f, h));
	const struct operator_s* j = operator_dup_create(i, 0, 2);
	const struct operator_s* k = operator_dup_create(j, 1, 2);

	complex float* in = md_alloc(N, dims2, CFL_SIZE);
	complex float* out = md_alloc(N, dims2, CFL_SIZE);

	md_zfill(N, dims2, in, 1.);
	md_zfill(N, dims2, out, 100.);

	operator_apply(k, N, dims2, out, N, dims2, in);

	double err = fabsf(md_znorm(N, dims2, in) - sqrtf(2.) * md_znorm(N, dims2, out));

	operator_free(a);
	operator_free(b);
	operator_free(c);
	operator_free(d);
	operator_free(e);
	operator_free(f);
	operator_free(g);
	operator_free(h);
	operator_free(i);
	operator_free(j);
	operator_free(k);

	md_free(in);
	md_free(out);

	return (err < UT_TOL);
}

UT_REGISTER_TEST(test_op_extract);




