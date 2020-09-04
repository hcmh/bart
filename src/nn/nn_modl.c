#include <assert.h>
#include <float.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "misc/types.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/rand.h"

#include "iter/proj.h"
#include <math.h>

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/iter6.h"
#include "iter/monitor_iter6.h"
#include "iter/batch_gen.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"
#include "nlops/const.h"

#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/losses.h"
#include "nn/init.h"
#include "nn/misc_nn.h"

#include "nn_modl.h"


const struct modl_s modl_default = {

	.Nt = 10,
	.Nl = 5,
	.Nf = 48,

	.Kx = 3,
	.Ky = 3,
	.Kz = 1,

	.normal_inversion_iter_conf = NULL,
	.batch_independent = true,
	.convergence_warn_limit = 0.,

	.lambda = NULL,

	.conv_0 = NULL,
	.conv_i = NULL,
	.conv_n = NULL,

	.bias_0 = NULL,
	.bias_i = NULL,
	.bias_n = NULL,

	.gamma_n = NULL,

	.bn_0 = NULL,
	.bn_i = NULL,
	.bn_n = NULL,

	.lambda_init = .05,
	.shared_weights = true,
	.shared_lambda = true,
	.share_pattern = false,

	.lambda_min = 0.,
	.lambda_fixed = -1.,

	.nullspace = false,
};

static const struct nlop_s* nlop_dw_first_layer(const struct modl_s* config, const long udims[5], enum NETWORK_STATUS status)
{
	long udims_w[5] = {1, udims[0], udims[1], udims[2], udims[4]};

	const struct nlop_s* result = nlop_from_linop_F(linop_reshape_create(5, udims_w, 5, udims));

	result = append_convcorr_layer(result, 0, config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL); //in: xn, conv_w; out; xn'
	result = append_batchnorm_layer(result, 0, ~MD_BIT(0), status); //in: xn, conv0, bn0_in; out: xn', bn0_out
	result = append_activation_bias(result, 0, ACT_RELU, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0; out: xn', bn0_out

	return result;
}

static const struct nlop_s* nlop_dw_append_center_layer(const struct modl_s* config, const struct nlop_s* nlop_dw, enum NETWORK_STATUS status)
{
	assert(0 < config->Nl - 2);

	nlop_dw = append_convcorr_layer(nlop_dw, 0, config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL); //in: xn, conv0, bn0_in, bias0, convi; out: xn', bn0_out
	nlop_dw = append_batchnorm_layer(nlop_dw, 0, ~MD_BIT(0), status); //in: xn, conv0, bn0_in, bias0, convi, bni_in; out: xn', bn0_out, bni_out
	nlop_dw = append_activation_bias(nlop_dw, 0, ACT_RELU, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out

	nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 4);
	nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 5);
	nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 6);
	nlop_dw = nlop_append_singleton_dim_out_F(nlop_dw, 2);


	for (int i = 0; i < config->Nl - 3; i++) {

		nlop_dw = append_convcorr_layer(nlop_dw, 0, config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL);
		//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convi'; out: xn', bn0_out, bni_out
		nlop_dw = append_batchnorm_layer(nlop_dw, 0, ~MD_BIT(0), status); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convi', bni_in'; out: xn', bn0_out, bni_out, bni_out'
		nlop_dw = append_activation_bias(nlop_dw, 0, ACT_RELU, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convi', bni_in', biasi_in'; out: xn', bn0_out, bni_out, bni_out'

		nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 7);
		nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 8);
		nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 9);
		nlop_dw = nlop_append_singleton_dim_out_F(nlop_dw, 3);

		nlop_dw = nlop_stack_inputs_F(nlop_dw, 4, 7, nlop_generic_domain(nlop_dw, 4)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, bni_in', biasi_in'; out: xn', bn0_out, bni_out, bni_out'
		nlop_dw = nlop_stack_inputs_F(nlop_dw, 5, 7, nlop_generic_domain(nlop_dw, 5)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, biasi_in'; out: xn', bn0_out, bni_out, bni_out'
		nlop_dw = nlop_stack_inputs_F(nlop_dw, 6, 7, nlop_generic_domain(nlop_dw, 6)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out, bni_out'

		nlop_dw = nlop_stack_outputs_F(nlop_dw, 2, 3, nlop_generic_codomain(nlop_dw, 2)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out
	}

	return nlop_dw;
}

static const struct nlop_s* nlop_dw_append_last_layer(const struct modl_s* config, const struct nlop_s* nlop_dw, enum NETWORK_STATUS status)
{
	assert(0 < config->Nl - 2);

	//nlop_dw in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out

	nlop_dw = append_convcorr_layer(nlop_dw, 0, 1, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn; out: xn', bn0_out, bni_out
	nlop_dw = append_batchnorm_layer(nlop_dw, 0, ~MD_BIT(0), status); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in; out: xn', bn0_out, bni_out, bnn_out

	//append gamma for batchnorm
	long N_out = nlop_generic_codomain(nlop_dw, 0)->N;
	long odims[N_out];
	md_copy_dims(N_out, odims, nlop_generic_codomain(nlop_dw, 0)->dims);
	long gdims[N_out];
	md_select_dims(N_out, MD_BIT(0), gdims, nlop_generic_codomain(nlop_dw, 0)->dims);
	auto nlop_gamma_mul = nlop_tenmul_create(N_out, odims, odims, gdims);

	nlop_dw = nlop_chain2_swap_FF(nlop_dw, 0, nlop_gamma_mul, 0); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman; out: xn', bn0_out, bni_out, bnn_out
	nlop_dw = append_activation_bias(nlop_dw, 0, ACT_LIN, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn', bn0_out, bni_out, bnn_out

	const long* udims_r = nlop_generic_codomain(nlop_dw, 0)->dims;
	long udims[5] = {udims_r[1], udims_r[2], udims_r[3], 1, udims_r[4]};
	nlop_dw = nlop_reshape_out_F(nlop_dw, 0, 5, udims);

	return nlop_dw;
}

static const struct nlop_s* nlop_dw_create(const struct modl_s* config, const long udims[5], enum NETWORK_STATUS status)
{
	assert(0 < config->Nl - 2);

	auto nlop_dw = nlop_dw_first_layer(config, udims, status);
	nlop_dw = nlop_dw_append_center_layer(config, nlop_dw, status);
	nlop_dw = nlop_dw_append_last_layer(config, nlop_dw, status);

	nlop_dw = nlop_chain2_FF(nlop_dw, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0); //in: xn, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: zn, bn0_out, bni_out, bnn_out
	nlop_dw = nlop_dup_F(nlop_dw, 0, 1); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn + xn', bn0_out, bni_out, bnn_out
	if (-1. == config->lambda_fixed) {
		nlop_dw = nlop_chain2_swap_FF(nlop_dw, 0, nlop_tenmul_create(5, udims, udims, MD_SINGLETON_DIMS(5)), 0);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: xn + xn', bn0_out, bni_out, bnn_out
		nlop_dw = nlop_reshape_in_F(nlop_dw, 11, 1, MD_SINGLETON_DIMS(1));
	} else {
		nlop_dw = nlop_chain2_FF(nlop_dw, 0, nlop_from_linop_F(linop_scale_create(5, udims, config->lambda_fixed)), 0);
		nlop_dw = nlop_combine_FF(nlop_dw, nlop_del_out_create(1, MD_SINGLETON_DIMS(1)));
	}

	nlop_dw = nlop_chain2_FF(nlop_from_linop_F(linop_zreal_create(1, MD_SINGLETON_DIMS(1))), 0, nlop_dw, 11);

	debug_printf(DP_DEBUG3, "MoDL dw created\n");

	return nlop_dw;
}


static const struct nlop_s* nlop_modl_cell_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status)
{
	auto nlop_dc = mri_normal_inversion_create_with_lambda(5, dims, config->share_pattern, config->lambda_fixed, config->batch_independent, config->convergence_warn_limit, config->normal_inversion_iter_conf); // in: x0+zn, coil, pattern, lambda; out: x(n+1)
	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	nlop_dc = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_dc, 0);
	nlop_dc = nlop_chain2_FF(nlop_dc, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);
	nlop_dc = nlop_chain2_swap_FF(nlop_zaxpbz_create(5, udims, 1., 1.), 0, nlop_dc, 0);// in: x0, zn, coil, pattern, lambda; out: x(n+1)

	auto nlop_dw = nlop_dw_create(config, udims, status);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: zn, bn0_out, bni_out, bnn_out

	const struct nlop_s* result = nlop_chain2_FF(nlop_dw, 0, nlop_dc, 1); //in: x0, coil, pattern, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: x(n+1), bn0_out, bni_out, bnn_out
	result = nlop_dup_F(result, 3, 15); //in: x0, coil, pattern, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out
	result = nlop_permute_inputs_F(result, 15, MAKE_ARRAY(4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)); //in: xn, x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

	//ad dimension for stacking
	for (int i = 4; i < 15; i++)
		result = nlop_append_singleton_dim_in_F(result, i);
	for (int o = 1; o < 4; o++)
		result = nlop_append_singleton_dim_out_F(result, o);

	debug_printf(DP_DEBUG3, "MoDL cell created\n");

	return result;
}

static const struct nlop_s* nlop_nullspace_modl_cell_create(const struct modl_s* config,const long dims[5],const long udims[5], enum NETWORK_STATUS status)
{
	auto result = mri_reg_proj_ker_create_with_lambda(5, dims, config->share_pattern, config->lambda_fixed, config->batch_independent, config->convergence_warn_limit, config->normal_inversion_iter_conf);// in: DW(xn), coil, pattern, lambda; out: PDW(xn)
	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};

	result = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, result, 0);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0); // in: DW(xn), coil, pattern, lambda; out: PDW(xn)

	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0); // in: x0, DW(xn), coil, pattern, lambda; out: PDW(xn) + x0

	auto nlop_dw = nlop_dw_create(config, udims, status);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: DW(xn), bn0_out, bni_out, bnn_out

	result = nlop_chain2_FF(nlop_dw, 0, result, 1); //in: x0, coil, pattern, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: x(n+1), bn0_out, bni_out, bnn_out

	result = nlop_dup_F(result, 3, 15); //in: x0, coil, pattern, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out
	result = nlop_permute_inputs_F(result, 15, MAKE_ARRAY(4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)); //in: xn, x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

	//ad dimension for stacking
	for (int i = 4; i < 15; i++)
		result = nlop_append_singleton_dim_in_F(result, i);
	for (int o = 1; o < 4; o++)
		result = nlop_append_singleton_dim_out_F(result, o);

	debug_printf(DP_DEBUG3, "MoDL null space cell created\n");

	return result;
}


static const struct nlop_s* nlop_modl_network_create(const struct modl_s* config, const long dims[5], long const udims[5], enum NETWORK_STATUS status)
{
	auto result = (config->nullspace ? nlop_nullspace_modl_cell_create : nlop_modl_cell_create)(config, dims, udims, status); //in: xn, x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

	for (int i = 1; i < config->Nt; i++) {

		auto nlop_append = (config->nullspace ? nlop_nullspace_modl_cell_create : nlop_modl_cell_create)(config, dims, udims, status); //in: xn, x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

		result = nlop_chain2_FF(result, 0, nlop_append, 0);
		//in: x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out, bn0_out, bni_out, bnn_out

		//duplicate fixed inputs
		result = nlop_dup_F(result, 0, 15);
		result = nlop_dup_F(result, 1, 15);
		result = nlop_dup_F(result, 2, 15);
		//in: x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out, bn0_out, bni_out, bnn_out

		//stack batchnorm in/outputs
		result = nlop_stack_inputs_F(result, 17, 5, nlop_generic_domain(result, 5)->N - 1);
		result = nlop_stack_inputs_F(result, 19, 8, nlop_generic_domain(result, 8)->N - 1);
		result = nlop_stack_inputs_F(result, 21, 11, nlop_generic_domain(result, 11)->N - 1);
		//in: x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, lambda, conv0, bias0, convi, biasi, convn, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out, bn0_out, bni_out, bnn_out
		result = nlop_stack_outputs_F(result, 4, 1, nlop_generic_codomain(result, 4)->N - 1);
		result = nlop_stack_outputs_F(result, 4, 2, nlop_generic_codomain(result, 4)->N - 1);
		result = nlop_stack_outputs_F(result, 4, 3, nlop_generic_codomain(result, 4)->N - 1);
		//in: x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, lambda, conv0, bias0, convi, biasi, convn, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out

		if (config->shared_lambda)
			result = nlop_dup_F(result, 3, 15);
		else
			result = nlop_stack_inputs_F(result, 15, 3, nlop_generic_domain(result, 15)->N - 1 );


		if (config->shared_weights) {

			result = nlop_dup_F(result, 4, 15);
			result = nlop_dup_F(result, 6, 15);
			result = nlop_dup_F(result, 7, 15);
			result = nlop_dup_F(result, 9, 15);
			result = nlop_dup_F(result, 10, 15);
			result = nlop_dup_F(result, 12, 15);
			result = nlop_dup_F(result, 13, 15);
			//in: x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn; out: x(n+2), bn0_out, bni_out, bnn_out
		} else {

			result = nlop_stack_inputs_F(result, 15, 4, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 6, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 7, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 9, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 10, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 12, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 13, nlop_generic_domain(result, 15)->N - 1 );
			//in: x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn; out: x(n+2), bn0_out, bni_out, bnn_out
		}

		result = nlop_permute_inputs_F(result, 15, MAKE_ARRAY(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
		//in: xn, x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out
	}

	result = nlop_dup_F(result, 0, 1);
	//in: x0, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out

	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	auto nlop_zf = nlop_mri_adjoint_create(5, dims, config->share_pattern);
	nlop_zf = nlop_chain2_FF(nlop_zf, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0); // in: kspace, coil, pattern; out: Atb

	if (config->nullspace) {

		auto nlop_norm_inv = mri_normal_inversion_create_with_lambda(5, dims, config->share_pattern, config->lambda_fixed, config->batch_independent, config->convergence_warn_limit, config->normal_inversion_iter_conf); // in: Atb, coil, pattern, lambda; out: A^+b
		nlop_norm_inv = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_norm_inv, 0);
		nlop_norm_inv = nlop_chain2_FF(nlop_norm_inv, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);

		nlop_zf = nlop_chain2_swap_FF(nlop_zf, 0,nlop_norm_inv, 0); // in: kspace, coil, pattern, coil, pattern, lambda; out: A^+b
		nlop_zf = nlop_dup_F(nlop_zf, 1, 3);
		nlop_zf = nlop_dup_F(nlop_zf, 2, 3);// in: kspace, coil, pattern, lambda; out: A^+b

		result = nlop_chain2_swap_FF(nlop_zf, 0, result, 0);
		//in: kspace, coil, pattern, lambda, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out

		result = nlop_dup_F(result, 1, 4);
		result = nlop_dup_F(result, 2, 4);
		result = nlop_append_singleton_dim_in_F(result, 3);
		result = nlop_dup_F(result, 3, 4);
		//in: kspace, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out
	} else {

		result = nlop_chain2_swap_FF(nlop_zf, 0, result, 0);
		//in: kspace, coil, pattern, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out

		result = nlop_dup_F(result, 1, 3);
		result = nlop_dup_F(result, 2, 3);
		//in: kspace, coil, pattern, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out
	}

	result = nlop_permute_inputs_F(result, 14, MAKE_ARRAY(0, 1, 2, 3, 4, 7, 10, 6, 9, 13, 12, 5, 8, 11));
	//in: kspace, coil, pattern, lambda, conv0, convi, convn, bias0, biasi, biasn, gamman, bn0_in, bni_in, bnn_in; out: xn, bn0_out, bni_out, bnn_out

	return result;
}



static complex float get_lambda(long NI, const float* x[NI])
{
	complex float result = 0;
	md_copy(1, MD_SINGLETON_DIMS(1), &result, x[4], CFL_SIZE);
	return result;
}

static const struct nlop_s* create_modl_val_loss(struct modl_s* modl, bool normalize, const char**valid_files)
{
	long kdims[5];
	long cdims[5];
	long udims[5];
	long pdims[5];

	complex float* val_kspace = load_cfl(valid_files[0], 5, kdims);
	complex float* val_coil = load_cfl(valid_files[1], 5, cdims);
	complex float* val_pattern = load_cfl(valid_files[2], 5, pdims);
	complex float* val_ref = load_cfl(valid_files[3], 5, udims);

	if (normalize) {

		complex float* scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
		complex float* u0 = md_alloc(5, udims, CFL_SIZE);
		compute_zero_filled(udims, u0, kdims, val_kspace, val_coil, pdims, val_pattern);
		compute_scale_max_abs(udims, scaling, u0);
		md_free(u0);

		normalize_by_scale(udims, scaling, val_ref, val_ref);
		normalize_by_scale(kdims, scaling, val_kspace, val_kspace);
		md_free(scaling);
	}

	auto valid_loss = nlop_modl_network_create(modl, kdims, udims, STAT_TEST);

	const struct nlop_s* loss = nlop_mse_create(5, udims, ~0ul);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);

	valid_loss = nlop_chain2_FF(valid_loss, 0, loss, 0);

	valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, udims, true, val_ref);
	valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, kdims, true, val_kspace);
	valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, cdims, true, val_coil);
	valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, pdims, true, val_pattern);

	valid_loss = nlop_combine_FF(nlop_del_out_create(5, pdims), valid_loss);
	valid_loss = nlop_combine_FF(nlop_del_out_create(5, cdims), valid_loss);
	valid_loss = nlop_combine_FF(nlop_del_out_create(5, kdims), valid_loss);
	valid_loss = nlop_combine_FF(nlop_del_out_create(5, udims), valid_loss);

	//batchnorm out
	valid_loss = nlop_del_out_F(valid_loss, 1);
	valid_loss = nlop_del_out_F(valid_loss, 1);
	valid_loss = nlop_del_out_F(valid_loss, 1);

	unmap_cfl(5, udims, val_ref);
	unmap_cfl(5, kdims, val_kspace);
	unmap_cfl(5, cdims, val_coil);
	unmap_cfl(5, pdims, val_pattern);

	return valid_loss;
}

void train_nn_modl(	struct modl_s* modl, struct iter6_conf_s* train_conf,
			const long udims[5], _Complex float* ref,
			const long kdims[5], _Complex float* kspace, const _Complex float* coil,
			const long pdims[5], const _Complex float* pattern,
			long Nb, bool random_order, bool normalize, const char** valid_files)
{
	complex float* scaling = NULL;
	if (normalize) {

		scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
		complex float* u0 = md_alloc(5, udims, CFL_SIZE);
		compute_zero_filled(udims, u0, kdims, kspace, coil, pdims, pattern);
		compute_scale_max_abs(udims, scaling, u0);
		md_free(u0);

		normalize_by_scale(kdims, scaling, kspace, kspace);
		normalize_by_scale(udims, scaling, ref, ref);
	}

	long Nt = kdims[4]; // number datasets

	long nkdims[5];
	long nudims[5];

	md_copy_dims(5, nkdims, kdims);
	md_copy_dims(5, nudims, udims);

	nkdims[4] = Nb;
	nudims[4] = Nb;

	modl->share_pattern = pdims[4] == 1;

	auto nlop_train = nlop_modl_network_create(modl, nkdims, nudims, STAT_TRAIN);
	nlop_train = nlop_chain2_FF(nlop_train, 0, nlop_mse_create(5, nudims, ~0ul), 0);
	//in: ref, kspace, coil, pattern, lambda, conv0, convi, convn, bias0, biasi, biasn, gamman, bn0_in, bni_in, bnn_in; out: loss, bn0_out, bni_out, bnn_out

	//create batch generator
	const complex float* train_data[] = {ref, kspace, coil, pattern};
	const long* train_dims[] = {	nlop_generic_domain(nlop_train, 0)->dims,
					nlop_generic_domain(nlop_train, 1)->dims,
					nlop_generic_domain(nlop_train, 2)->dims,
					nlop_generic_domain(nlop_train, 3)->dims};

	auto batch_generator = (random_order ? batch_gen_rand_create : batch_gen_linear_create)(4, 5, train_dims, train_data, Nt, 0);

	//setup for iter algorithm
	float* data[15];
	enum IN_TYPE in_type[15];
	const struct operator_p_s* projections[15];

	for (int i = 0; i < 4; i++) {

		data[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
		projections[i] = NULL;
	}

	data[4] = (float*)modl->lambda;
	data[5] = (float*)modl->conv_0;
	data[6] = (float*)modl->conv_i;
	data[7] = (float*)modl->conv_n;
	data[8] = (float*)modl->bias_0;
	data[9] = (float*)modl->bias_i;
	data[10] = (float*)modl->bias_n;
	data[11] = (float*)modl->gamma_n;
	data[12] = (float*)modl->bn_0;
	data[13] = (float*)modl->bn_i;
	data[14] = (float*)modl->bn_n;

	for (int i = 4; i < 12; i++) {

		projections[i] = NULL;
		in_type[i] = IN_OPTIMIZE;
	}

	for (int i = 12; i < 15; i++) {

		projections[i] = NULL;
		in_type[i] = IN_BATCHNORM;
	}

	projections[4] = operator_project_pos_real_create(nlop_generic_domain(nlop_train, 4)->N, nlop_generic_domain(nlop_train, 4)->dims);

	enum OUT_TYPE out_type[4] = {OUT_OPTIMIZE, OUT_BATCHNORM, OUT_BATCHNORM, OUT_BATCHNORM};

	struct monitor_iter6_s* monitor;

	if (NULL != valid_files) {

		auto nlop_validation_loss = create_modl_val_loss(modl, normalize, valid_files);

		struct monitor_value_s value_monitors[2] = {monitor_iter6_function_create(get_lambda, true, "lambda"), monitor_iter6_nlop_create(nlop_validation_loss, false, "val loss")};
		nlop_free(nlop_validation_loss);

		monitor = create_monitor_iter6_progressbar_with_val_monitor(2, value_monitors);
	} else {

		auto value_monitor = monitor_iter6_function_create(get_lambda, true, "lambda");
		monitor = create_monitor_iter6_progressbar_with_val_monitor(1, &value_monitor);
	}

	iter6_adam(train_conf, nlop_train, 15, in_type, projections, data, 4, out_type, Nb, Nt / Nb, batch_generator, monitor);

	nlop_free(nlop_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);

	if (normalize) {

		renormalize_by_scale(kdims, scaling, kspace, kspace);
		renormalize_by_scale(udims, scaling, ref, ref);
		md_free(scaling);
	}
}


void apply_nn_modl(	struct modl_s* modl,
			const long udims[5], complex float* out,
			const long kdims[5], const complex float* kspace, const complex float* coil, const long pdims[5], const complex float* pattern,
			bool normalize)
{

	modl->share_pattern = (1 == pdims[4]);

	auto nlop_modl = nlop_modl_network_create(modl, kdims, udims, STAT_TEST);
	nlop_modl = nlop_del_out_F(nlop_modl, 1);
	nlop_modl = nlop_del_out_F(nlop_modl, 1);
	nlop_modl = nlop_del_out_F(nlop_modl, 1);

	complex float* out_tmp = md_alloc_sameplace(5, udims, CFL_SIZE, modl->lambda);
	complex float* kspace_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, modl->lambda);
	complex float* coil_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, modl->lambda);
	complex float* pattern_tmp = md_alloc_sameplace(5, pdims, CFL_SIZE, modl->lambda);

	complex float* scaling = NULL;

	md_copy(5, kdims, kspace_tmp, kspace, CFL_SIZE);
	md_copy(5, kdims, coil_tmp, coil, CFL_SIZE);
	md_copy(5, pdims, pattern_tmp, pattern, CFL_SIZE);

	if (normalize) {

		complex float* scaling_cpu = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
		complex float* u0 = md_alloc(5, udims, CFL_SIZE);
		compute_zero_filled(udims, u0, kdims, kspace, coil, pdims, pattern);
		compute_scale_max_abs(udims, scaling_cpu, u0);
		md_free(u0);

		scaling = md_alloc_sameplace(1, MAKE_ARRAY(kdims[4]), CFL_SIZE, modl->lambda);
		md_copy(1, MAKE_ARRAY(kdims[4]), scaling, scaling_cpu, CFL_SIZE);
		md_free(scaling_cpu);

		normalize_by_scale(kdims, scaling, kspace_tmp, kspace_tmp);
	}

	complex float* args[15];

	args[0] = out_tmp;
	args[1] = kspace_tmp;
	args[2] = coil_tmp;
	args[3] = pattern_tmp;
	args[4] = modl->lambda;
	args[5] = modl->conv_0;
	args[6] = modl->conv_i;
	args[7] = modl->conv_n;
	args[8] = modl->bias_0;
	args[9] = modl->bias_i;
	args[10] = modl->bias_n;
	args[11] = modl->gamma_n;
	args[12] = modl->bn_0;
	args[13] = modl->bn_i;
	args[14] = modl->bn_n;

	nlop_generic_apply_select_derivative_unchecked(nlop_modl, 15, (void**)args, 0, 0);

	if (normalize) {

		renormalize_by_scale(udims, scaling, out_tmp, out_tmp);
		md_free(scaling);
	}

	md_copy(5, udims, out, out_tmp, CFL_SIZE);

	nlop_free(nlop_modl);

	md_free(out_tmp);
	md_free(kspace_tmp);
	md_free(coil_tmp);
	md_free(pattern_tmp);
}

void apply_nn_modl_batchwise(	struct modl_s* modl,
				const long udims[5], complex float * out,
				const long kdims[5], const complex float* kspace, const complex float* coil, const long pdims[5], const complex float* pattern,
				long Nb,
				bool normalize)
{
	long Nt = kdims[4];
	while (0 < Nt) {

		long kdims1[5];
		long udims1[5];
		long pdims1[5];

		md_copy_dims(5, kdims1, kdims);
		md_copy_dims(5, udims1, udims);
		md_copy_dims(5, pdims1, pdims);

		long Nb_tmp = MIN(Nt, Nb);

		kdims1[4] = Nb_tmp;
		udims1[4] = Nb_tmp;
		pdims1[4] = MIN(pdims1[4], Nb_tmp);

		apply_nn_modl(modl, udims1, out, kdims1, kspace, coil, pdims1, pattern, normalize);

		out += md_calc_size(5, udims1);
		kspace += md_calc_size(5, kdims1);
		coil += md_calc_size(5, kdims1);
		if (1 < pdims[4])
			coil += md_calc_size(5, pdims1);

		Nt -= Nb_tmp;
	}
}


void init_nn_modl(struct modl_s* modl)
{
	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};

	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};

	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	long Ns[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};
	long* dims[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	for (int i = 0; i < 11; i++)
		if (NULL == *weights[i])
			*weights[i] = md_alloc(Ns[i], dims[i], CFL_SIZE);

	//init lambda
	long wi = 0;
	md_zfill(Ns[wi], dims[wi], *weights[wi], (-1 != modl->lambda_fixed ? modl->lambda_fixed : modl->lambda_init));

	//init conv_w
	wi = 1;
	complex float* tmp = *weights[wi];
	for (int i = 0; i < dims[wi][Ns[wi] - 1]; i++)
		tmp = init_glorot_uniform_conv_complex(Ns[wi] - 1, dims[wi], tmp, true);

	wi = 2;
	tmp = *weights[wi];
	for (int i = 0; i < dims[wi][Ns[wi] - 1] * dims[wi][Ns[wi] - 2]; i++)
		tmp = init_glorot_uniform_conv_complex(Ns[wi] - 2, dims[wi], tmp, true);

	wi = 3;
	tmp = *weights[wi];
	for (int i = 0; i < dims[wi][Ns[wi] - 1]; i++)
		tmp = init_glorot_uniform_conv_complex(Ns[wi] - 1, dims[wi], tmp, true);

	//init bias with 0
	for (wi = 4; wi < 7; wi++)
		md_zfill(Ns[wi], dims[wi], *weights[wi], 0.);

	//init gamma with 1
	wi = 7;
	md_zfill(Ns[wi], dims[wi], *weights[wi], 1.);

	//init batchnorm with 0
	for (wi = 8; wi < 11; wi++)
		md_zfill(Ns[wi], dims[wi], *weights[wi], 0.);

}

void nn_modl_move_gpucpu(struct modl_s* modl, bool gpu) {

#ifdef USE_CUDA

	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};

	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};

	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	long Ns[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};

	long* dims[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	for (int i = 0; i < 11; i++) {

		complex float* tmp = (gpu ? md_alloc_gpu : md_alloc)(Ns[i], dims[i], CFL_SIZE);
		md_copy(Ns[i], dims[i], tmp, *(weights[i]), CFL_SIZE);
		md_free(*weights[i]);
		*weights[i] = tmp;
	}
#else
	UNUSED(modl);
	if (gpu)
		error("Compiled without GPU support!");
#endif
}

extern void nn_modl_store_weights(struct modl_s* modl, const char* name)
{
	const complex float* weights[] = {
		(modl->lambda),
		(modl->conv_0),
		(modl->conv_i),
		(modl->conv_n),
		(modl->bias_0),
		(modl->bias_i),
		(modl->bias_n),
		(modl->gamma_n),
		(modl->bn_0),
		(modl->bn_i),
		(modl->bn_n)
	};

	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};

	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};

	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	unsigned int D[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};

	const long* dims[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	dump_multi_cfl(name, 11, D, dims, weights);
}

void nn_modl_load_weights(struct modl_s* modl, const char* name, bool overwrite_parameters)
{
	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	unsigned int D[11];
	long dims[11][8];
	complex float* args[11];

	load_multi_cfl(name, 11, 8, D, dims, args);

	long D_expected[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};
	for (int i = 0; i < 11; i++)
		assert(D[i] == D_expected[i]);

	if (overwrite_parameters) {

		modl->Nf = dims[1][0];

		modl->Kx = dims[1][2];
		modl->Ky = dims[1][3];
		modl->Kz = dims[1][4];

		modl->Nl = dims[2][5] + 2;

		modl->shared_weights = dims[0][1] != dims[10][6];
	}


	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};
	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};
	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	long* dims_exp[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	for (int i = 0; i < 11; i++)
		for (unsigned int j = 0; j< D[i]; j++ )
			assert((dims[i][j] == dims_exp[i][j]) || ((j == D[i] - 1) && (dims[i][j] == 1)));

	for (int i = 0; i < 11; i++) {

		if (NULL == *(weights[i]))
			*(weights[i]) = md_alloc(D[i], dims_exp[i], CFL_SIZE);

		md_copy2(D[i], dims_exp[i], MD_STRIDES(D[i], dims_exp[i], CFL_SIZE), *(weights[i]), MD_STRIDES(D[i], dims[i], CFL_SIZE), args[i], CFL_SIZE);
	}
	const long* dims_unmap[11];
	for (int i = 0; i < 11; i++)
		dims_unmap[i] = &(dims[i][0]);
	unmap_multi_cfl(11, D, dims_unmap, args);
}

extern void nn_modl_free_weights(struct modl_s* modl)
{
	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	for (int i = 0; i < 11; i++) md_free(*weights[i]);
}
