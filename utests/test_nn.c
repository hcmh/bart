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
#include "nlops/nltest.h"

#include "nn/activation.h"

#include "utest.h"


static bool test_nlop_relu_derivative(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	const struct nlop_s* relu = nlop_relu_create(N, dims);

	double err = nlop_test_derivative(relu);

	nlop_free(relu);

	UT_ASSERT(err < 1.E1); // Problem in Relu test
}



UT_REGISTER_TEST(test_nlop_relu_derivative);




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

