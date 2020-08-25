/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdio.h>

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
#include "nlops/chain.h"
#include "nn/layers.h"
#include "nn/activation.h"
#include "nn/init.h"
#include "nn/mnist.h"
#include "nn/rbf.h"
#include "utest.h"

// for convolution: output_dim == input_dim - (kernel_dim - 1)
// for other dimensions: two dimensions are same and other == 1

/**
static bool test_nlop_trans_con(void)
{
	enum { N = 1 };
	long dims_image[N] = {10};
	long dims_kernel[N] = {3};
	long dims_output[N] = {8};
	unsigned int conv_flags = 1;

	complex float* dst_trans = md_alloc(N, dims_output, CFL_SIZE);
	complex float* dst_conv = md_alloc(N, dims_output, CFL_SIZE)
}

*/

static bool test_nlop_conv_compare(void)
{
 	enum { N = 6 };
 	long dims_image[N] = { 10, 1, 4, 7, 1, 8};
	long dims_kernel[N] = { 3, 4, 4, 2, 1, 8};
	long dims_output[N] = { 8, 4, 4, 6, 1, 1};
	unsigned int conv_flags = 9; //100100

	complex float* dst_geom = md_alloc(N, dims_output, CFL_SIZE);
	complex float* dst_fft = md_alloc(N, dims_output, CFL_SIZE);

	complex float* src1 = md_alloc(N, dims_image, CFL_SIZE);
	complex float* src2 = md_alloc(N, dims_kernel, CFL_SIZE);

	complex float* nul = md_alloc(N, dims_output, CFL_SIZE);
	md_clear(N, dims_output, nul, CFL_SIZE);

	md_gaussian_rand(N, dims_image, src1);
	md_gaussian_rand(N, dims_kernel, src2);

	const struct nlop_s* conv_geom = nlop_convcorr_geom_create(N, conv_flags, dims_output, dims_image, dims_kernel, PAD_VALID, true, 'N');
	const struct nlop_s* conv_fft = nlop_convcorr_fft_create(N, conv_flags, dims_output, dims_image, dims_kernel, PAD_VALID, true);

	const struct nlop_s* conv_geom_const = nlop_set_input_const_F(conv_geom, 1, N, dims_kernel, true, src2);
	const struct nlop_s* conv_fft_const = nlop_set_input_const_F(conv_fft, 1, N, dims_kernel, true, src2);

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
	unsigned long conv_flags = 9; //100100

	const struct nlop_s* conv_geom = nlop_convcorr_geom_create(N, conv_flags, dims_output, dims_image, dims_kernel, PAD_VALID, true, 'N');
	const struct nlop_s* conv_fft = nlop_convcorr_fft_create(N, conv_flags, dims_output, dims_image, dims_kernel, PAD_VALID, true);

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



static bool test_padding(void)
{
	enum { N = 2 };
	long dims_in[N] = { 3, 2};
	long dims_out[N] = {7, 4};

	long pad[] = {2, 1};

	complex float in[] = {	1, 2, 3,
				4, 5, 6};

	complex float exp_valid[] = {	1, 2, 3,
					4, 5, 6};
	complex float exp_same[] = {	0, 0, 0, 0, 0, 0, 0,
                                 	0, 0, 1, 2, 3, 0, 0,
                                 	0, 0, 4, 5, 6, 0, 0,
                                 	0, 0, 0, 0, 0, 0, 0};
	complex float exp_reflect[] = {	6, 5, 4, 5, 6, 5, 4,
                                	3, 2, 1, 2, 3, 2, 1,
                                	6, 5, 4, 5, 6, 5, 4,
                                	3, 2, 1, 2, 3, 2, 1};
	complex float exp_sym[] = {	2, 1, 1, 2, 3, 3, 2,
                                  	2, 1, 1, 2, 3, 3, 2,
                                  	5, 4, 4, 5, 6, 6, 5,
                                  	5, 4, 4, 5, 6, 6, 5};
	complex float exp_cyc[] = {	5, 6, 4, 5, 6, 4, 5,
                                  	2, 3, 1, 2, 3, 1, 2,
                                  	5, 6, 4, 5, 6, 4, 5,
                                  	2, 3, 1, 2, 3, 1, 2};


	complex float* out = md_alloc(2, dims_out, CFL_SIZE);

	const struct linop_s* lin_pad;
	float err = 0;

	lin_pad = linop_padding_create(2, dims_in, PAD_SAME, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_same, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_REFLECT, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_reflect, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_SYMMETRIC, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_sym, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_CYCLIC, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_cyc, out);

	long pad_down[] = {-2, -1};
	lin_pad = linop_padding_create(2, dims_out, PAD_VALID, pad_down, pad_down);
	linop_forward_unchecked(lin_pad, in, out);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_in, in, exp_valid);

	UT_ASSERT(1.e-7 > err);
}

UT_REGISTER_TEST(test_padding);


static bool test_padding_adjoint(void)
{
	enum { N = 2 };
	long dims_in[N] = { 3, 2};
	long dims_out[N] = {7, 4};

	long pad[] = {2, 1};

	const struct linop_s* lin_pad;
	float err = 0;

	lin_pad = linop_padding_create(2, dims_in, PAD_SAME, pad, pad);
	err += linop_test_adjoint(lin_pad);
	#ifdef USE_CUDA
	auto nlop = nlop_from_linop(lin_pad);
	err += compare_gpu(nlop, nlop);
	nlop_free(nlop);
	#endif
	linop_free(lin_pad);


	lin_pad = linop_padding_create(2, dims_in, PAD_REFLECT, pad, pad);
	#ifdef USE_CUDA
	nlop = nlop_from_linop(lin_pad);
	err += compare_gpu(nlop, nlop);
	nlop_free(nlop);
	#endif
	linop_free(lin_pad);


	lin_pad = linop_padding_create(2, dims_in, PAD_SYMMETRIC, pad, pad);
	err += linop_test_adjoint(lin_pad);
	#ifdef USE_CUDA
	nlop = nlop_from_linop(lin_pad);
	err += compare_gpu(nlop, nlop);
	nlop_free(nlop);
	#endif
	linop_free(lin_pad);


	lin_pad = linop_padding_create(2, dims_in, PAD_CYCLIC, pad, pad);
	err += linop_test_adjoint(lin_pad);
	#ifdef USE_CUDA
	nlop = nlop_from_linop(lin_pad);
	err += compare_gpu(nlop, nlop);
	nlop_free(nlop);
	#endif
	linop_free(lin_pad);

	long pad_down[] = {-2, -1};
	lin_pad = linop_padding_create(2, dims_out, PAD_VALID, pad_down, pad_down);
	err += linop_test_adjoint(lin_pad);
	#ifdef USE_CUDA
	nlop = nlop_from_linop(lin_pad);
	err += compare_gpu(nlop, nlop);
	nlop_free(nlop);
	#endif
	linop_free(lin_pad);

	debug_printf(DP_DEBUG1, "err: %.8f\n", err);

	UT_ASSERT(1.e-6 > err);
}

UT_REGISTER_TEST(test_padding_adjoint);



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
	_Bool test = (err_adj < 3.E-5) && (err_der < 1.E-1);
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

	network = append_convcorr_layer(network, 0, 4, kernel_size, true, PAD_VALID, true, ones, ones);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "conv errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-5) && (err_der < 1.E-1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_conv_der);

static bool test_conv_transp(void)
{
	unsigned int N = 5;
	long indims[] = {5, 7, 6, 3, 5};
	long outdims[] = {4, 5, 4, 3, 5};
	long kernel_size[] = {3, 3, 1};
	long kdims[] = {4, 5, 3, 3 ,1};

	complex float* kernel = md_alloc(N, kdims, CFL_SIZE);
	md_gaussian_rand(N, kdims, kernel);

	auto forward = append_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, indims)), 0, 4, kernel_size, true, PAD_VALID, true, NULL, NULL);
	forward = nlop_set_input_const_F(forward, 1, N, kdims, true, kernel);
	auto adjoint = append_transposed_convcorr_layer(nlop_from_linop_F(linop_identity_create(N, outdims)), 0, 5, kernel_size, true, true, PAD_VALID, true, NULL, NULL);
	adjoint = nlop_set_input_const_F(adjoint, 1, N, kdims, true, kernel);

	PTR_ALLOC(struct linop_s, c);
	c->forward = forward->op;
	c->adjoint = adjoint->op;
	c->normal = NULL;
	c->norm_inv = NULL;

	float err = linop_test_adjoint(c);
	XFREE(c);

	nlop_free(forward);
	nlop_free(adjoint);
	UT_ASSERT(err < 1.e-5);
}

UT_REGISTER_TEST(test_conv_transp);



static bool test_mpool_der(void)
{
	unsigned int N = 5;
	long indims[] = {2, 6, 1, 1, 2}; //channel, x, y, z, batch
	long outdims[] = {2, 2, 1, 1, 2}; //channel, x, y, z, batch

	//digits reference, e.g. 1204.: batch(1), channel(2), count(04)
	complex float in[] = {	1101., 1202., 1103., 1204., 1105., 1206., 1107., 1208., 1109., 1210., 1111., 1212.,
				2103., 2204., 2101., 2202., 2107., 2208., 2105., 2206., 2109., 2210., 2111., 2212. };

	complex float adj_exp[] = {	0., 0., 0., 0., 1105., 1206., 0., 0., 0., 0., 1111., 1212.,
					0., 0., 0., 0., 2107., 2208., 0., 0., 0., 0., 2111., 2212. };

	complex float out_exp[] = {	1105., 1206., 1111., 1212.,
					2107., 2208., 2111., 2212.};
	complex float* out = md_alloc(N, indims, CFL_SIZE);

	const struct nlop_s* network = nlop_from_linop_F(linop_identity_create(N, indims));
	network = append_maxpool_layer(network, 0, MAKE_ARRAY(3l, 1l, 1l), PAD_VALID, true);
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

	network = append_activation(network, 0, ACT_RELU);

	float err_adj = nlop_test_adj_derivatives(network, true);
	float err_der = nlop_test_derivatives(network);

	nlop_free(network);

	debug_printf(DP_DEBUG1, "relu errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	_Bool test = (err_adj < 1.E-6) && (err_der < 1.E1);
	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_relu_der);

static bool test_nlop_rbf(void)
{
 	enum { N = 3 };
 	long dims[N] = { 4, 3, 5};

	auto op = nlop_activation_rbf_create(dims, 1., -1.);
	auto op_gpu = nlop_activation_rbf_create(dims, 1., -1.);

	float err_adj = nlop_test_adj_derivatives(op, true);
	float err_der = nlop_test_derivatives(op);

	#ifdef USE_CUDA
	float err = compare_gpu(op, op_gpu);
	#else
	float err = 0.;
	UNUSED(op_gpu);
	#endif

	debug_printf(DP_DEBUG1, "rbf errors der, adj, gpu: %.8f, %.8f, %.8f\n", err_der, err_adj, err);
	UT_ASSERT((err_der < 1.E-2) && (err_adj < 1.E-6)  && (err < 1.E-5));
}

UT_REGISTER_TEST(test_nlop_rbf);

static bool test_avgpool(void)
{
	unsigned int N = 5;	
	long indims[] = {2, 6, 1, 1, 2}; 	//channel, x, y, z, batch 
	long avg_dims[] = {2, 2, 1, 1, 2}; 	//channel, x, y, z, batch 
	long pool_size[] = {3, 1, 1};

	complex float in[] = {	1102., 1201., 1104., 1203., 1106., 1205., 1207., 1408., 1209., 1410., 1211., 1412.,
				2303., 2204., 2302., 2203., 2307., 2208., 2204., 2406., 2209., 2410., 2211., 2411. };

	// adjoint consists of average divided by amount of averaged numbers
	complex float adj_exp[] = {	368., 401., 368., 401., 368., 401., 403., 470., 403., 470., 403., 470., 
					768., 735., 768., 735., 768., 735., 736., 803., 736., 803., 736., 803. };

	complex float avg_exp[] = {	1104., 1203., 1209., 1410.,
					2304., 2205., 2208., 2409.};

	complex float* avg = md_alloc(N, indims, CFL_SIZE);
	complex float* adj = md_alloc(N, indims, CFL_SIZE);

	const struct nlop_s* network = nlop_from_linop_F(linop_identity_create(N, indims));

	network = append_avgpool_layer(network, 0, pool_size, PAD_VALID, true);
	nlop_apply(network, N, avg_dims, avg, N, indims, in); 		// check output of average pooling layer
	nlop_adjoint(network, N, indims, adj, N, avg_dims, avg);	// check adjoint of average pooling layer

	nlop_free(network);

	UT_ASSERT(1.e-8 >  md_zrmse(N, avg_dims, avg, avg_exp) + md_zrmse(N, indims, adj, adj_exp) );
}

UT_REGISTER_TEST(test_avgpool);

static bool test_upsampl(void)
{
	unsigned int N = 5;	
	long idims[] = {2, 2, 1, 1, 2}; 	//channel, x, y, z, batch 
	long odims[] = {2, 6, 1, 1, 2}; 	//channel, x, y, z, batch 

	complex float in[] = {	1101., 1203., 1104., 1206.,
				2100., 2202., 2103., 2205.};

	complex float upsampl_exp[] = {	367., 401., 367., 401., 367., 401., 368., 402., 368., 402., 368., 402.,
					700., 734., 700., 734., 700., 734., 701., 735., 701., 735., 701., 735. };
	// adjoint consists of average divided by amount of averaged numbers
	complex float upsampl_adj_exp[] = {	367., 401., 368., 402.,
					 	700., 734., 701., 735.};

	complex float* upsampl = md_alloc(N, odims, CFL_SIZE);
	complex float* upsampl_adj = md_alloc(N, idims, CFL_SIZE);

	const struct nlop_s* network = nlop_from_linop_F(linop_identity_create(N, idims));
	network = append_upsampl_layer(network, 0, MAKE_ARRAY(3l, 1l, 1l), true);
	nlop_apply(network, N, odims, upsampl, N, idims, in);
	nlop_adjoint(network, N, idims, upsampl_adj, N, odims, upsampl);

	nlop_free(network);

	UT_ASSERT(1.e-8 >  md_zrmse(N, odims, upsampl, upsampl_exp)+ md_zrmse(N, idims, upsampl_adj, upsampl_adj_exp));
}

UT_REGISTER_TEST(test_upsampl);