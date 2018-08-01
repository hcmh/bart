/* Copyright 2017-2018. Martin Uecker.
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

#include "linops/someops.h"
#include "linops/linop.h"
#include "linops/lintest.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static bool test_linop_conv_adjoint(enum conv_type ct, enum conv_mode cm, unsigned int flags, int N, const long odims[N], const long idims1[N], const long idims2[N])
{
	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);

	struct linop_s* conv = linop_conv_create(N, flags, ct, cm, odims, idims2, idims1, src1);

	float diff = linop_test_adjoint(conv);

	debug_printf(DP_DEBUG1, "adjoint diff: %f\n", diff);

	bool ret = (diff < 1.E-4f);

	linop_free(conv);

	return ret;
}

const int A = 2;
const int B = 10;
const int C = 4;
const int L = 3;


static bool test_linop_conv_adjoint_cyclic(void)
{
	long odims[3] = { 1, B, A };
	long idims1[3] = { C, L, A };
	long idims2[3] = { C, B, 1 };

	bool ret = true;
	ret &= test_linop_conv_adjoint(CONV_CYCLIC, CONV_SYMMETRIC, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_CYCLIC, CONV_CAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_CYCLIC, CONV_ANTICAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	return ret;
}

UT_REGISTER_TEST(test_linop_conv_adjoint_cyclic);

static bool test_linop_conv_adjoint_truncated(void)
{
	long odims[3] = { 1, B, A };
	long idims1[3] = { C, L, A };
	long idims2[3] = { C, B, 1 };

	bool ret = true;
	ret &= test_linop_conv_adjoint(CONV_TRUNCATED, CONV_SYMMETRIC, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_TRUNCATED, CONV_CAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_TRUNCATED, CONV_ANTICAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	return ret;
}

UT_REGISTER_TEST(test_linop_conv_adjoint_truncated);

static bool test_linop_conv_adjoint_valid(void)
{
	long odims[3] = { 1, B - L + 1, A };
	long idims1[3] = { C, L, A };
	long idims2[3] = { C, B, 1 };

	bool ret = true;
	ret &= test_linop_conv_adjoint(CONV_VALID, CONV_SYMMETRIC, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_VALID, CONV_CAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_VALID, CONV_ANTICAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	return ret;
}

UT_REGISTER_TEST(test_linop_conv_adjoint_valid);

static bool test_linop_conv_adjoint_extended(void)
{
	long odims[3] = { 1, B + L - 1, A };
	long idims1[3] = { C, L, A };
	long idims2[3] = { C, B, 1 };

	bool ret = true;
	ret &= test_linop_conv_adjoint(CONV_EXTENDED, CONV_SYMMETRIC, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_EXTENDED, CONV_CAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	ret &= test_linop_conv_adjoint(CONV_EXTENDED, CONV_ANTICAUSAL, MD_BIT(1), 3, odims, idims1, idims2);
	return ret;
}

UT_REGISTER_TEST(test_linop_conv_adjoint_extended);



static bool test_linop_conv_normal(void)
{
	enum { N = 3 };

	int A = 2;
	int B = 10;
	int C = 4;
	int L = 3;

	long odims[N] = { 1, B, A };
	long idims1[N] = { C, L, A };
	long idims2[N] = { C, B, 1 };

	complex float* src1 = md_alloc(N, idims1, CFL_SIZE);

	md_gaussian_rand(N, idims1, src1);

	struct linop_s* conv = linop_conv_create(N, MD_BIT(1), CONV_CYCLIC, CONV_SYMMETRIC, odims, idims2, idims1, src1);

	float nrmse = linop_test_normal(conv);

	debug_printf(DP_DEBUG1, "normal nrmse: %f\n", nrmse);

	bool ret = (nrmse < 1.E-6f);

	linop_free(conv);

	return ret;
}




UT_REGISTER_TEST(test_linop_conv_normal);

