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

#include "nlops/conv.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nn/layers.h"
#include "nn/activation.h"
#include "nn/init.h"
#include "nn/mnist.h"
#include "utest.h"




static bool test_nlop_conv_compare(void)
{
 	enum { N = 6 };
 	long dims_image[N] = { 10, 1, 4, 7, 1, 8};
	long dims_kernel[N] = { 3, 4, 4, 2, 1, 8};
	long dims_output[N] = { 8, 4, 4, 6, 1, 1};
	unsigned int conv_flags = 9;

	complex float* dst_geom = md_alloc(N, dims_output, CFL_SIZE);
	complex float* dst_fft = md_alloc(N, dims_output, CFL_SIZE);

	complex float* src1 = md_alloc(N, dims_image, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims_kernel, CFL_SIZE);

	complex float* nul = md_alloc(N, dims_output, CFL_SIZE);
	md_clear(N, dims_output, nul, CFL_SIZE);

	md_gaussian_rand(N, dims_image, src1);
	md_gaussian_rand(N, dims_kernel, src2);

	const struct nlop_s* conv_geom = nlop_conv_geom_create(N, conv_flags, dims_output, dims_image, dims_kernel, PADDING_VALID);
	const struct nlop_s* conv_fft = nlop_conv_fft_create(N, conv_flags, dims_output, dims_image, dims_kernel, PADDING_VALID);

	const struct nlop_s* conv_geom_const = nlop_set_input_const_F(conv_geom, 1, N, dims_kernel, src2);
	const struct nlop_s* conv_fft_const = nlop_set_input_const_F(conv_fft, 1, N, dims_kernel, src2);

	nlop_apply(conv_geom_const, N, dims_output, dst_geom, N, dims_image, src1);
	nlop_apply(conv_fft_const, N, dims_output, dst_fft, N, dims_image, src1);

	float err = md_znrmse(N, dims_output, dst_geom, dst_fft);

	nlop_free(conv_geom_const);
	nlop_free(conv_fft_const);

	md_free(src1);
	md_free(src2);
	md_free(dst_fft);
	md_free(dst_geom);

	debug_printf(DP_DEBUG1, "Mean Error fft vs geom conv: %.8f\n",err);
	UT_ASSERT(err < 1.E-5);
}

UT_REGISTER_TEST(test_nlop_conv_compare);



static bool test_nlop_conv_derivative(void)
{
	enum { N = 6 };
	long dims_image[N] = { 6, 1, 2, 5, 1, 2};
	long dims_kernel[N] = { 3, 4, 2, 2, 1, 2};
	long dims_output[N] = { 4, 4, 2, 4, 1, 1};
	unsigned long conv_flags = 9;

	const struct nlop_s* conv_geom = nlop_conv_geom_create(N, conv_flags, dims_output, dims_image, dims_kernel, PADDING_VALID);
	const struct nlop_s* conv_fft = nlop_conv_fft_create(N, conv_flags, dims_output, dims_image, dims_kernel, PADDING_VALID);

	float err_adj_geom = nlop_test_adj_derivatives(conv_geom, false);
	float err_der_geom = nlop_test_derivatives(conv_geom);

	float err_adj_fft = nlop_test_adj_derivatives(conv_fft, false);
	float err_der_fft = nlop_test_derivatives(conv_fft);

	nlop_free(conv_fft);
	nlop_free(conv_geom);

	debug_printf(DP_DEBUG1, "conv (geom / fft) der errors: %.7f, %.7f\n", err_der_geom, err_der_fft);
	debug_printf(DP_DEBUG1, "conv (geom / fft) adj errors: %.7f, %.7f\n", err_adj_geom, err_adj_fft);

	_Bool test = (err_der_geom < 1.E-1) && (err_der_fft < 1.E-1) && (err_adj_geom < 1.E-6) && (err_adj_fft < 1.E-5);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_nlop_conv_derivative);

static bool test_dense_der(void)
{
	unsigned int N = 2;
	long indims[] = {210, 18};

	const struct linop_s* id = linop_identity_create(N, indims);
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	network = append_dense_layer(network, 0, 128);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "dense errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-5) && (err_der < 1.E-1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_dense_der);

static bool test_conv_der(void)
{
	unsigned int N = 5;
	long indims[] = {5, 7, 6, 3, 5};

	const struct linop_s* id = linop_identity_create(N, indims);
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	long kernel_size[] = {3, 3, 1};
	long ones[] = {1, 1, 1};

	network = append_conv_layer(network, 0, 4, kernel_size, PADDING_VALID, true, ones, ones);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "conv errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-5) && (err_der < 1.E-1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_conv_der);



static bool test_mpool_der(void)
{
	unsigned int N = 5;
	long indims[] = {2, 4, 1, 1, 2};
	long outdims[] = {2, 2, 1, 1, 2};


	complex float in[] = {1101., 1202., 1103., 1204., 1105., 1206., 1107., 1208., 2103., 2204., 2101., 2202., 2107., 2208., 2105., 2206. };
	complex float adj_exp[] = {0., 0., 1103., 1204., 0., 0., 1107., 1208., 2103., 2204., 0., 0., 2107., 2208., 0., 0. };
	complex float out_exp[] = {1103., 1204., 1107., 1208., 2103., 2204., 2107., 2208.};
	complex float* out = md_alloc(N, indims, CFL_SIZE);

	const struct nlop_s* network = nlop_from_linop_F(linop_identity_create(N, indims));
	network = append_maxpool_layer(network, 0, MAKE_ARRAY(2l, 1l, 1l), PADDING_VALID, true);
	nlop_apply(network, 5, outdims, out, N, indims, in);
	nlop_adjoint(network, N, indims, in, N, outdims, out);

	nlop_free(network);

	UT_ASSERT(1.e-8 > md_zrmse(N, outdims, out, out_exp) + md_zrmse(N, indims, in, adj_exp));
}

UT_REGISTER_TEST(test_mpool_der);



static bool test_bias_der(void)
{
	unsigned int N = 4;
	long dims[] = { 4, 1, 3, 4};
	long bdims[] = { 1, 1, 3, 4};

	const struct nlop_s* network = nlop_bias_create(N, dims, bdims);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "bias errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-5) && (err_der < 1.E-1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_bias_der);

static bool test_relu_der(void)
{
	unsigned int N = 4;
	long dims[] = { 30, 78, 3, 25};

	const struct linop_s* id = linop_identity_create(N, dims);
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	network = append_activation(network, ACT_RELU, 0);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "relu errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-6) && (err_der < 1.E1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_relu_der);
