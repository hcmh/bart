/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>
#include <complex.h>
#include <stdio.h>

#include "num/casorati.h"
#include "num/conv.h"

#include "utest.h"


#if 0
static bool test_conv_small(void)
{
	const complex float K[3] = { 0., 1., 1. };
	const complex float T[7] = { 0., 0., 0., 1., 0., 0., 0. };
	const complex float G[5] = {     0., 1., 1., 0., 0.     };
	complex float O[5];

	conv_small(1, (long[]){ 7 }, (long[]){ 3 },
		(long[]){ sizeof(complex float) }, O, K,
		(long[]){ sizeof(complex float) }, T);

	bool ok = true;

	for (int i = 0; i < 5; i++)
		ok &= (1.E-7 > cabsf(O[i] - G[i]));

	return ok;
}

UXT_REGISTER_TEST(test_conv_small);
#endif


static bool test_conv_generic(enum conv_mode mode, enum conv_type type, int N, const complex float G[N])
{
	const complex float K[3] = { 0.5, 1., -1.i };
	const complex float T[8] = { 1., 0., 0., 1., 0., 0., 0., 0. };
	complex float O[N];

	conv(1, 1u, type, mode,
		(long[]){ N }, O, (long[]){ 8 }, T, (long[]){ 3 }, K);

	bool ok = true;

	for (int i = 0; i < N; i++) {

		if (mode == CONV_SYMMETRIC)
		printf("%d %f+%fi %f+%fi\n", i, crealf(O[i]), cimagf(O[i]), crealf(G[i]), cimagf(G[i]));
		ok &= (1.E-4 > cabsf(O[i] - G[i]));
	}

	if (!ok)
		printf("FAILED\n");

	return ok;
}

static bool test_conv_sy_ex(void)
{
	return test_conv_generic(CONV_SYMMETRIC, CONV_EXTENDED,
		10, (const complex float[10]){ 0.5, 1., -1.i, 0.5, 1., -1.i, 0., 0., 0., 0. });
}

static bool test_conv_sy_vd(void)
{
	return test_conv_generic(CONV_SYMMETRIC, CONV_VALID,
		6, (const complex float[6]){ -1.i, 0.5, 1., -1.i, 0., 0. });
}

static bool test_conv_sy_tr(void)
{
	return test_conv_generic(CONV_SYMMETRIC, CONV_TRUNCATED,
		8, (const complex float[8]){ 1., -1.i, 0.5, 1., -1.i, 0., 0., 0. });
}

static bool test_conv_sy_cy(void)
{
	return test_conv_generic(CONV_SYMMETRIC, CONV_CYCLIC,
		8, (const complex float[8]){ 1., -1.i, 0.5, 1., -1.i, 0., 0., 0.5 });
}

static bool test_conv_ca_ex(void)
{
	return test_conv_generic(CONV_CAUSAL, CONV_EXTENDED,
		10, (const complex float[10]){ 0.5, 1., -1.i, 0.5, 1., -1.i, 0., 0., 0., 0. });
}

static bool test_conv_ca_vd(void) // ?
{
	return test_conv_generic(CONV_CAUSAL, CONV_VALID,
		6, (const complex float[6]){ 0.5, 1., -1.i, 0.5, 1., -1.i });
}

static bool test_conv_ca_tr(void)
{
	return test_conv_generic(CONV_CAUSAL, CONV_TRUNCATED,
		8, (const complex float[8]){ 0.5, 1., -1.i, 0.5, 1., -1.i, 0., 0.  });
}

static bool test_conv_ca_cy(void)
{
	return test_conv_generic(CONV_CAUSAL, CONV_CYCLIC,
		8, (const complex float[8]){ 0.5, 1., -1.i, 0.5, 1., -1.i, 0., 0.  });
}


static bool test_conv_ac_ex(void) //?
{
	return test_conv_generic(CONV_ANTICAUSAL, CONV_EXTENDED,
		10, (const complex float[10]){ 0.5, -1.i, 1., 0.5, 0., 0., 0., 0., -1.i, 1. });
}

static bool test_conv_ac_vd(void)
{
	return test_conv_generic(CONV_ANTICAUSAL, CONV_VALID,
		6, (const complex float[6]){ 0.5, -1.i, 1., 0.5, 0., 0. });
}

static bool test_conv_ac_tr(void)
{
	return test_conv_generic(CONV_ANTICAUSAL, CONV_TRUNCATED,
		8, (const complex float[8]){ 0.5, -1.i, 1., 0.5, 0., 0., 0., 0. });
}

static bool test_conv_ac_cy(void) // ?
{
	return test_conv_generic(CONV_ANTICAUSAL, CONV_CYCLIC,
		8, (const complex float[8]){ 0.5, -1.i, 1., 0.5, 0., 0.0, -1.i, 1. });
}









UT_REGISTER_TEST(test_conv_sy_ex);
UT_REGISTER_TEST(test_conv_sy_vd);
UT_REGISTER_TEST(test_conv_sy_tr);
UT_REGISTER_TEST(test_conv_sy_cy);

UT_REGISTER_TEST(test_conv_ca_ex);
UT_REGISTER_TEST(test_conv_ca_vd);
UT_REGISTER_TEST(test_conv_ca_tr);
UT_REGISTER_TEST(test_conv_ca_cy);

UT_REGISTER_TEST(test_conv_ac_ex);
UT_REGISTER_TEST(test_conv_ac_vd);
UT_REGISTER_TEST(test_conv_ac_tr);
UT_REGISTER_TEST(test_conv_ac_cy);


