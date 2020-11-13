
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
#include "nn/batchnorm.h"
#include "nn/losses.h"
#include "nn/activation.h"
#include "nn/init.h"
#include "utest.h"

static bool test_dense_layer_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 2;
	long indims[] = {5, 18};

	auto op_cpu = append_dense_layer(nlop_from_linop_F(linop_identity_create(N, indims)), 0, 128);
	auto op_gpu = append_dense_layer(nlop_from_linop_F(linop_identity_create(N, indims)), 0, 128);

	float err = compare_gpu(op_cpu, op_gpu);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	UT_ASSERT(err < 5.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_dense_layer_gpu);


static bool test_conv_layer_gpu_CF(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 5;
	long indims[] = {4, 7, 6, 2, 2};

	auto op_cpu = append_convcorr_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 1, MAKE_ARRAY(5l,3l,1l), false, PAD_VALID, true, NULL, NULL);
	auto op_gpu = append_convcorr_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 1, MAKE_ARRAY(5l,3l,1l), false, PAD_VALID, true, NULL, NULL);

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);


	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 2.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_conv_layer_gpu_CF);

static bool test_conv_layer_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 5;
	long indims[] = {4, 5, 1, 2, 2};

	auto op_cpu = append_convcorr_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 1, MAKE_ARRAY(3l,1l,1l), false, PAD_VALID, false, NULL, NULL);
	auto op_gpu = append_convcorr_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, 1, MAKE_ARRAY(3l,1l,1l), false, PAD_VALID, false, NULL, NULL);

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_conv_layer_gpu);

static bool test_maxpool_layer_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 5;
	long indims[] = {3, 4, 6, 1, 2};

	auto op_cpu = append_maxpool_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, MAKE_ARRAY(2l,2l,1l), PAD_VALID, true);
	auto op_gpu = append_maxpool_layer(nlop_from_linop(linop_identity_create(N, indims)), 0, MAKE_ARRAY(2l,2l,1l), PAD_VALID, true);

	float err = compare_gpu(op_cpu, op_gpu);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_maxpool_layer_gpu);

static bool test_bias_op_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 5;
	long indims[] = {3, 4, 6, 1, 2};
	long bdims[] = {3, 1, 1, 1, 1};

	auto op_cpu = nlop_bias_create(N, indims, bdims);
	auto op_gpu = nlop_bias_create(N, indims, bdims);

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_bias_op_gpu);

static bool test_linear_layer_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 5;
	long indims[] = {3, 4, 6, 1, 2};

	auto op_cpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), 0, ACT_LIN, MD_BIT(0));
	auto op_gpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), 0, ACT_LIN, MD_BIT(0));


	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_linear_layer_gpu);

static bool test_relu_layer_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 5;
	long indims[] = {3, 4, 6, 1, 2};

	auto op_cpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), 0, ACT_RELU, MD_BIT(0));
	auto op_gpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), 0, ACT_RELU, MD_BIT(0));

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_relu_layer_gpu);


static bool test_softmax_layer_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 2;
	long indims[] = {7,20};

	auto op_cpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), 0, ACT_SOFTMAX, MD_BIT(0));
	auto op_gpu = append_activation_bias(nlop_from_linop(linop_identity_create(N, indims)), 0, ACT_SOFTMAX, MD_BIT(0));

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_softmax_layer_gpu);


static bool test_cce_layer_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	unsigned int N = 2;
	long dims[] = {10, 128};


	auto op_cpu = nlop_cce_create(N, dims);
	auto op_gpu = nlop_cce_create(N, dims);

	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_cce_layer_gpu);

static bool test_stats_operator_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	enum { N = 2 };
	long idims[N] = { 10, 3 };

	auto op_cpu = nlop_stats_create(N, idims, MD_BIT(0));
	auto op_gpu = nlop_stats_create(N, idims, MD_BIT(0));


	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_stats_operator_gpu);

static bool test_normalize_operator_gpu(void)
{
#ifndef USE_CUDA
	return true;
#else
	enum { N = 2 };
	long idims[N] = { 10, 3 };

	auto op_cpu = nlop_normalize_create(N, idims, MD_BIT(0), 1.e-7);
	auto op_gpu = nlop_normalize_create(N, idims, MD_BIT(0), 1.e-7);


	float err = compare_gpu(op_cpu, op_gpu);

	debug_printf(DP_DEBUG1, "err: %f\n", err);

	nlop_free(op_cpu);
	nlop_free(op_gpu);

	UT_ASSERT(err < 1.e-5);
#endif
}

UT_GPU_REGISTER_TEST(test_normalize_operator_gpu);
