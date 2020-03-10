
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
#include "nn/losses.h"
#include "nn/activation.h"
#include "nn/init.h"
#include "nn/mnist.h"
#include "utest.h"

static bool test_dense_layer_gpu(void)
{
	unsigned int N = 2;
	long indims[] = {210, 18};

	auto op_cpu = append_dense_layer(nlop_from_linop_F(linop_identity_create(N, indims)), 0, 128);
	auto op_gpu = append_dense_layer(nlop_from_linop_F(linop_identity_create(N, indims)), 0, 128);

	float err = compare_gpu(op_cpu, op_gpu);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	return(err < 2.e-5);
}

UT_GPU_REGISTER_TEST(test_dense_layer_gpu);


static bool test_conv_layer_gpu_CF(void)
{
	unsigned int N = 5;
	long indims[] = {4, 3, 1, 2, 2};

	long ones [] ={1, 1, 1};

	auto op_cpu = append_conv_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 4, MAKE_ARRAY(2l,2l,1l), PADDING_SAME, true, ones, ones);
	auto op_gpu = append_conv_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 4, MAKE_ARRAY(2l,2l,1l), PADDING_SAME, true, ones, ones);


	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);


	nlop_free(op_cpu);
	nlop_free(op_gpu);

	return(err < 1.e-5);
}

UT_GPU_REGISTER_TEST(test_conv_layer_gpu_CF);

static bool test_conv_layer_gpu(void)
{
	unsigned int N = 5;
	long indims[] = {4, 3, 1, 2, 2};

	long ones [] ={1, 1, 1};

	auto op_cpu = append_conv_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 4, MAKE_ARRAY(2l,2l,1l), PADDING_SAME, false, ones, ones);
	auto op_gpu = append_conv_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 4, MAKE_ARRAY(2l,2l,1l), PADDING_SAME, false, ones, ones);

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	return(err < 1.e-5);
}

UT_GPU_REGISTER_TEST(test_conv_layer_gpu);

static bool test_maxpool_layer_gpu(void)
{
	unsigned int N = 5;
	long indims[] = {3, 4, 6, 1, 2};

	auto op_cpu = append_maxpool_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, MAKE_ARRAY(2l,2l,1l), PADDING_VALID, true);
	auto op_gpu = append_maxpool_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, MAKE_ARRAY(2l,2l,1l), PADDING_VALID, true);

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	return(err < 1.e-5);
}

UT_GPU_REGISTER_TEST(test_maxpool_layer_gpu);

static bool test_relu_layer_gpu(void)
{
	unsigned int N = 5;
	long indims[] = {3, 4, 6, 1, 2};

	auto op_cpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), ACT_RELU, 0, MD_BIT(0));
	auto op_gpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), ACT_RELU, 0, MD_BIT(0));

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	return(err < 1.e-5);
}

UT_GPU_REGISTER_TEST(test_relu_layer_gpu);


static bool test_softmax_layer_gpu(void)
{
	unsigned int N = 2;
	long indims[] = {7,20};

	auto op_cpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), ACT_SOFTMAX, 0, MD_BIT(0));
	auto op_gpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), ACT_SOFTMAX, 0, MD_BIT(0));

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	return(err < 1.e-5);
}

UT_GPU_REGISTER_TEST(test_softmax_layer_gpu);


static bool test_cce_layer_gpu(void)
{
	unsigned int N = 2;
	long dims[] = {10, 128};


	auto op_cpu = nlop_cce_create(N, dims);
	auto op_gpu = nlop_cce_create(N, dims);

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	return(err < 1.e-5);
}

UT_GPU_REGISTER_TEST(test_cce_layer_gpu);
