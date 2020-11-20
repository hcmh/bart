#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "iter/italgos.h"

#include "misc/cJSON.h"
#include "misc/read_json.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/types.h"

#include "nn/const.h"
#include "nn/losses_nn.h"
#include "nn/weights.h"

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
#include "nlops/stack.h"

#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/losses.h"
#include "nn/init.h"

#include "nn/init.h"

#include "nn/nn.h"
#include "nn/chain.h"
#include "nn/layers_nn.h"
#include "nn/activation_nn.h"

#include "nn/nn_ops.h"

#include "modl.h"


const struct modl_s modl_default = {

	.Nt = 10,
	.Nl = 5,
	.Nf = 32,

	.Kx = 3,
	.Ky = 3,
	.Kz = 1,

	.normal_inversion_iter_conf = NULL,
	.batch_independent = true,
	.convergence_warn_limit = 0.,

	.lambda_init = .05,
	.lambda_fixed = -1.,

	.shared_weights = true,
	.shared_lambda = true,
	.share_pattern = false,

	.reinsert_zerofilled = false,
	.init_tickhonov = false,
	.batch_norm = true,
	.residual_network =true,

	.nullspace = false,
	.use_dc = true,

	.normalize = false,

	.draw_graph_filename = NULL,
};

static nn_t residual_create(const struct modl_s* config, const long udims[5], enum NETWORK_STATUS status){

	long udims_w[5] = {1, udims[0], udims[1], udims[2], udims[4]};

	nn_t result = 0;

	if (config->reinsert_zerofilled) {

		long udims_w2[5] = {2, udims[0], udims[1], udims[2], udims[4]};
		const struct nlop_s* nlop_init = nlop_stack_create(5, udims_w2, udims_w, udims_w, 0);
		nlop_init = nlop_reshape_in_F(nlop_init, 0, 5, udims);
		nlop_init = nlop_reshape_in_F(nlop_init, 1, 5, udims);

		result = nn_from_nlop_F(nlop_init);
		result = nn_set_input_name_F(result, 1, "zero_filled");

	} else {

		result = nn_from_nlop(nlop_from_linop_F(linop_reshape_create(5, udims_w, 5, udims)));
	}

	auto conv_init = init_kaiming_create(in_flag_conv(true), false, false, 0);

	// append first layer
	result = nn_append_convcorr_layer(result, 0, NULL, "conv_0", config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL, initializer_clone(conv_init));
	if (config->batch_norm)
		result = nn_append_batchnorm_layer(result, 0, NULL, "bn_0", ~MD_BIT(0), status, NULL);
	result = nn_append_activation_bias(result, 0, NULL, "bias_0", ACT_RELU, MD_BIT(0));

	// append first stackable layer
	result = nn_append_convcorr_layer(result, 0, NULL, "conv_i", config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL, NULL);
	result = nn_append_singleton_dim_in_F(result, 0, "conv_i");

	if (config->batch_norm) {

		result = nn_append_batchnorm_layer(result, 0, NULL, "bn_i", ~MD_BIT(0), status, NULL);
		result = nn_append_singleton_dim_in_F(result, 0, "bn_i");
		result = nn_append_singleton_dim_out_F(result, 0, "bn_i");
	}

	result = nn_append_activation_bias(result, 0, NULL, "bias_i", ACT_RELU, MD_BIT(0));
	result = nn_append_singleton_dim_in_F(result, 0, "bias_i");

	for (int i = 0; i < config->Nl - 3; i++) {

		result = nn_append_convcorr_layer(result, 0, NULL, NULL, config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL, initializer_clone(conv_init));
		result = nn_append_singleton_dim_in_F(result, -1, NULL);
		result = nn_stack_inputs_F(result, 0, "conv_i", -1, NULL, -1);

		if (config->batch_norm) {

			result = nn_append_batchnorm_layer(result, 0, NULL, NULL, ~MD_BIT(0), status, NULL);
			result = nn_append_singleton_dim_in_F(result, -1, NULL);
			result = nn_append_singleton_dim_out_F(result, -1, NULL);
			result = nn_stack_inputs_F(result, 0, "bn_i", -1, NULL, -1);
			result = nn_stack_outputs_F(result, 0, "bn_i", -1, NULL, -1);
		}

		result = nn_append_activation_bias(result, 0, NULL, NULL, ACT_RELU, MD_BIT(0));
		result = nn_append_singleton_dim_in_F(result, -1, NULL);
		result = nn_stack_inputs_F(result, 0, "bias_i", -1, NULL, -1);
	}

	// append last layer
	const struct initializer_s* init = (config->batch_norm || !config->residual_network) ? initializer_clone(conv_init) : init_const_create(0);
	result = nn_append_convcorr_layer(result, 0, NULL, "conv_n", 1, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL, init);

	initializer_free(conv_init);

	if (config->batch_norm) {

		result = nn_append_batchnorm_layer(result, 0, NULL, "bn_n", ~MD_BIT(0), status, NULL);

		//append gamma for batchnorm
		auto iov = nn_generic_codomain(result, 0, NULL);
		auto nn_scale_gamma = nn_from_nlop_F(nlop_tenmul_create(iov->N, iov->dims, iov->dims, MD_SINGLETON_DIMS(iov->N)));
		result = nn_chain2_swap_FF(result, 0, NULL, nn_scale_gamma, 0, NULL);
		result = nn_set_input_name_F(result, -1, "gamma");
		result = nn_set_initializer_F(result, 0, "gamma", init_const_create(config->residual_network ? 0 : 1));
		result = nn_set_in_type_F(result, 0, "gamma", IN_OPTIMIZE);
	}

	result = nn_append_activation_bias(result, 0, NULL, "bias_n", ACT_LIN, MD_BIT(0));

	result = nn_reshape_out_F(result, 0, NULL, 5, udims);

#if 0
	if (config->residual_network) {

		result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(5, udims, 1, 1)), 1, NULL);
		result = nn_dup_F(result, 0, NULL, 1, NULL);
	}
#else
	//FIXME: the potentially vanishing residual connection reduce memory consumption when the derivatives are computed
	result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(5, udims, config->residual_network ? 1 : 0, 1)), 1, NULL);
	result = nn_dup_F(result, 0, NULL, 1, NULL);
#endif


	// Append dims for stacking over modl iterations (non-shared weights)
	result = nn_append_singleton_dim_in_F(result, 0, "conv_0");
	result = nn_append_singleton_dim_in_F(result, 0, "conv_i");
	result = nn_append_singleton_dim_in_F(result, 0, "conv_n");
	result = nn_append_singleton_dim_in_F(result, 0, "bias_0");
	result = nn_append_singleton_dim_in_F(result, 0, "bias_i");
	result = nn_append_singleton_dim_in_F(result, 0, "bias_n");

	if (config->batch_norm) {

		result = nn_append_singleton_dim_in_F(result, 0, "gamma");
		result = nn_append_singleton_dim_in_F(result, 0, "bn_0");
		result = nn_append_singleton_dim_in_F(result, 0, "bn_i");
		result = nn_append_singleton_dim_in_F(result, 0, "bn_n");
		result = nn_append_singleton_dim_out_F(result, 0, "bn_0");
		result = nn_append_singleton_dim_out_F(result, 0, "bn_i");
		result = nn_append_singleton_dim_out_F(result, 0, "bn_n");
	}

	result = nn_sort_inputs_by_list_F(result, 10,
		(const char*[10]) {
			"conv_0",
			"conv_i",
			"conv_n",
			"bias_0",
			"bias_i",
			"bias_n",
			"gamma",
			"bn_0",
			"bn_i",
			"bn_n"
			}
		);

	return (1 == config->Nt) ? result : nn_checkpoint_F(result, true);
}

static nn_t data_consistency_modl_create(const struct modl_s* config,const long dims[5], const long udims[5])
{
	auto nlop_dc = mri_normal_inversion_create_with_lambda(5, dims, config->share_pattern, config->lambda_fixed, config->batch_independent, config->convergence_warn_limit, config->normal_inversion_iter_conf); // in: x0+zn, coil, pattern, lambda; out: x(n+1)
	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	nlop_dc = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_dc, 0);
	nlop_dc = nlop_chain2_FF(nlop_dc, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);
	nlop_dc = nlop_chain2_swap_FF(nlop_zaxpbz_create(5, udims, (-1. != config->lambda_fixed) ? config->lambda_fixed : 1., 1.), 0, nlop_dc, 0);// in: zi, zero_filled, coil, pattern, lambda; out: x(n+1)

	if (-1. == config->lambda_fixed) {

		const struct nlop_s* nlop_scale_lambda = nlop_tenmul_create(5, udims, udims, MD_SINGLETON_DIMS(5));
		nlop_scale_lambda = nlop_chain2_FF(nlop_from_linop_F(linop_zreal_create(5, MD_SINGLETON_DIMS(5))), 0, nlop_scale_lambda, 1);
		nlop_scale_lambda = nlop_reshape_in_F(nlop_scale_lambda, 1, 1, MD_SINGLETON_DIMS(1));

		nlop_dc = nlop_chain2_FF(nlop_scale_lambda, 0, nlop_dc, 0);// in: zero_filled, coil, pattern, lambda, zi, lambda; out: x(n+1)
		nlop_dc = nlop_dup_F(nlop_dc, 3, 5);
		nlop_dc = nlop_shift_input_F(nlop_dc, 0, 4);
	}

	auto result = nn_from_nlop_F(nlop_dc);
	result = nn_set_input_name_F(result, 1, "zero_filled");
	result = nn_set_input_name_F(result, 1, "coil");
	result = nn_set_input_name_F(result, 1, "pattern");
	result = nn_set_input_name_F(result, 1, "lambda");

	result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);
	result = nn_set_initializer_F(result, 0, "lambda", init_const_create((-1 != config->lambda_fixed) ? config->lambda_fixed : config->lambda_init));

	return result;
}

static nn_t nn_modl_cell_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status)
{
	auto dw = residual_create(config, udims, status);
	auto dc = data_consistency_modl_create(config, dims, udims);

	if (config->reinsert_zerofilled)
		dw = nn_mark_dup_F(dw, "zero_filled");

	nn_t result = nn_chain2_FF(dw, 0, NULL, dc, 0, NULL);
	result = nn_stack_dup_by_name_F(result);

	return result;
}

static nn_t nn_modl_zf_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status)
{
	UNUSED(status);

	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	auto nlop_zf = nlop_mri_adjoint_create(5, dims, config->share_pattern);
	nlop_zf = nlop_chain2_FF(nlop_zf, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);
	auto nn_zf = nn_from_nlop_F(nlop_zf);
	nn_zf = nn_set_input_name_F(nn_zf, 0, "kspace");
	nn_zf = nn_set_input_name_F(nn_zf, 0, "coil");
	nn_zf = nn_set_input_name_F(nn_zf, 0, "pattern");

	if (config->normalize) {

		auto nn_normalize = nn_from_nlop_F(nlop_norm_zmax_create(5, udims, MD_BIT(4), true));
		nn_normalize = nn_set_output_name_F(nn_normalize, 1, "normalize_scale");
		nn_zf = nn_chain2_FF(nn_zf, 0, NULL, nn_normalize, 0, NULL);
	}

	return nn_zf;
}



static nn_t nn_nullspace_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status);
static nn_t nn_modl_no_dc_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status);

static nn_t nn_modl_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status)
{
	if (!config->use_dc)
		return nn_modl_no_dc_create(config, dims, udims, status);

	if (config->nullspace)
		return nn_nullspace_create(config, dims, udims, status);

	auto result = nn_modl_cell_create(config, dims, udims, status);

	for (int i = 1; i < config->Nt; i++) {

		auto tmp = nn_modl_cell_create(config, dims, udims, status);

		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_0");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_i");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_n");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_0");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_i");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_n");

		if (config->batch_norm) {

			tmp = nn_mark_stack_input_F(tmp, "bn_0");
			tmp = nn_mark_stack_input_F(tmp, "bn_i");
			tmp = nn_mark_stack_input_F(tmp, "bn_n");
			tmp = nn_mark_stack_input_F(tmp, "gamma");

			tmp = nn_mark_stack_output_F(tmp, "bn_0");
			tmp = nn_mark_stack_output_F(tmp, "bn_i");
			tmp = nn_mark_stack_output_F(tmp, "bn_n");
		}

		tmp = nn_mark_dup_F(tmp, "zero_filled");
		tmp = nn_mark_dup_F(tmp, "coil");
		tmp = nn_mark_dup_F(tmp, "pattern");
		tmp = (config->shared_lambda ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "lambda");


		result = nn_chain2_FF(result, 0, NULL, tmp, 0, NULL);
		result = nn_stack_dup_by_name_F(result);
	}

	if (config -> init_tickhonov) {

		result = nn_mark_dup_F(result, "zero_filled");
		result = nn_mark_dup_F(result, "coil");
		result = nn_mark_dup_F(result, "pattern");
		result = (config->shared_lambda ? nn_mark_dup_F : nn_mark_stack_input_F)(result, "lambda");

		auto dc = data_consistency_modl_create(config, dims, udims);
		auto iov = nn_generic_domain(dc, 0, NULL);
		complex float* zeros = md_alloc(iov->N, iov->dims, iov->size);
		md_clear(iov->N, iov->dims, zeros, iov->size);
		dc = nn_chain2_FF(nn_from_nlop_F(nlop_const_create(iov->N, iov->dims, true, zeros)), 0, NULL, dc, 0, NULL);

		result = nn_chain2_swap_FF(dc, 0, NULL, result, 0, NULL);
		result = nn_stack_dup_by_name_F(result);
	} else {

		result = nn_dup_F(result, 0, "zero_filled", 0, NULL);
	}

	auto nn_zf = nn_modl_zf_create(config, dims, udims, status);

	result = nn_mark_dup_F(result, "coil");
	result = nn_mark_dup_F(result, "pattern");

	result = nn_chain2_swap_FF(nn_zf, 0, NULL, result, 0, "zero_filled");
	result = nn_stack_dup_by_name_F(result);

	return result;
}

static nn_t data_consistency_nullspace_create(const struct modl_s* config,const long dims[5], const long udims[5])
{
	auto nlop_dc = mri_reg_proj_ker_create_with_lambda(5, dims, config->share_pattern, config->lambda_fixed, config->batch_independent, config->convergence_warn_limit, config->normal_inversion_iter_conf); // in: x0+zn, coil, pattern, lambda; out: x(n+1)
	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	nlop_dc = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_dc, 0);
	nlop_dc = nlop_chain2_FF(nlop_dc, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);

	nlop_dc = nlop_chain2_FF(nlop_dc, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0);// in: zi, zero_filled, coil, pattern, lambda; out: x(n+1)

	auto result = nn_from_nlop_F(nlop_dc);
	result = nn_set_input_name_F(result, 0, "tickhonov_reg");
	result = nn_set_input_name_F(result, 1, "coil");
	result = nn_set_input_name_F(result, 1, "pattern");
	result = nn_set_input_name_F(result, 1, "lambda");
	result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);
	result = nn_set_initializer_F(result, 0, "lambda", init_const_create((-1 != config->lambda_fixed) ? config->lambda_fixed : config->lambda_init));

	return result;
}

static nn_t nn_nullspace_cell_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status)
{
	auto dw = residual_create(config, udims, status);
	auto dc = data_consistency_nullspace_create(config, dims, udims);

	nn_t result = nn_chain2_FF(dw, 0, NULL, dc, 0, NULL);
	result = nn_stack_dup_by_name_F(result);

	return result;
}


static nn_t nn_nullspace_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status)
{
	auto result = nn_nullspace_cell_create(config, dims, udims, status);

	for (int i = 1; i < config->Nt; i++) {

		auto tmp = nn_nullspace_cell_create(config, dims, udims, status);

		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_0");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_i");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_n");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_0");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_i");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_n");

		if (config->batch_norm) {

			tmp = nn_mark_stack_input_F(tmp, "bn_0");
			tmp = nn_mark_stack_input_F(tmp, "bn_i");
			tmp = nn_mark_stack_input_F(tmp, "bn_n");
			tmp = nn_mark_stack_input_F(tmp, "gamma");

			tmp = nn_mark_stack_output_F(tmp, "bn_0");
			tmp = nn_mark_stack_output_F(tmp, "bn_i");
			tmp = nn_mark_stack_output_F(tmp, "bn_n");
		}

		if (config->reinsert_zerofilled)
			tmp = nn_mark_dup_F(tmp, "zero_filled");

		tmp = nn_mark_dup_F(tmp, "tickhonov_reg");
		tmp = nn_mark_dup_F(tmp, "coil");
		tmp = nn_mark_dup_F(tmp, "pattern");
		tmp = (config->shared_lambda ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "lambda");


		result = nn_chain2_FF(result, 0, NULL, tmp, 0, NULL);
		result = nn_stack_dup_by_name_F(result);
	}

	result = nn_dup_F(result, 0, "tickhonov_reg", 0, NULL);

	if (config->reinsert_zerofilled)
		result = nn_mark_dup_F(result, "zero_filled");

	result = nn_mark_dup_F(result, "coil");
	result = nn_mark_dup_F(result, "pattern");
	result = (config->shared_lambda ? nn_mark_dup_F : nn_mark_stack_input_F)(result, "lambda");

	auto dc = data_consistency_modl_create(config, dims, udims);
	auto iov = nn_generic_domain(dc, 0, NULL);
	complex float* zeros = md_alloc(iov->N, iov->dims, iov->size);
	md_clear(iov->N, iov->dims, zeros, iov->size);
	dc = nn_chain2_FF(nn_from_nlop_F(nlop_const_create(iov->N, iov->dims, true, zeros)), 0, NULL, dc, 0, NULL);
	result = nn_chain2_swap_FF(dc, 0, NULL, result, 0, "tickhonov_reg");
	result = nn_stack_dup_by_name_F(result);

	auto nn_zf = nn_modl_zf_create(config, dims, udims, status);

	result = nn_mark_dup_F(result, "coil");
	result = nn_mark_dup_F(result, "pattern");

	result = nn_chain2_swap_FF(nn_zf, 0, NULL, result, 0, "zero_filled");
	result = nn_stack_dup_by_name_F(result);

	return result;
}

static nn_t nn_modl_no_dc_create(const struct modl_s* config,const long dims[5], const long udims[5], enum NETWORK_STATUS status)
{
	auto result = residual_create(config, udims, status);

	for (int i = 1; i < config->Nt; i++) {

		auto tmp = residual_create(config, udims, status);

		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_0");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_i");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "conv_n");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_0");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_i");
		tmp = (config->shared_weights ? nn_mark_dup_F : nn_mark_stack_input_F)(tmp, "bias_n");

		if (config->batch_norm) {

			tmp = nn_mark_stack_input_F(tmp, "bn_0");
			tmp = nn_mark_stack_input_F(tmp, "bn_i");
			tmp = nn_mark_stack_input_F(tmp, "bn_n");
			tmp = nn_mark_stack_input_F(tmp, "gamma");

			tmp = nn_mark_stack_output_F(tmp, "bn_0");
			tmp = nn_mark_stack_output_F(tmp, "bn_i");
			tmp = nn_mark_stack_output_F(tmp, "bn_n");
		}

		if (config->reinsert_zerofilled)
			tmp = nn_mark_dup_F(tmp, "zero_filled");

		result = nn_chain2_FF(result, 0, NULL, tmp, 0, NULL);
		result = nn_stack_dup_by_name_F(result);
	}

	if (config->reinsert_zerofilled)
		result = nn_mark_dup_F(result, "zero_filled");

	if (config->init_tickhonov) {

		auto dc = data_consistency_modl_create(config, dims, udims);
		auto iov = nn_generic_domain(dc, 0, NULL);
		complex float* zeros = md_alloc(iov->N, iov->dims, iov->size);
		md_clear(iov->N, iov->dims, zeros, iov->size);
		dc = nn_chain2_FF(nn_from_nlop_F(nlop_const_create(iov->N, iov->dims, true, zeros)), 0, NULL, dc, 0, NULL);
		result = nn_chain2_swap_FF(dc, 0, NULL, result, 0, NULL);
		result = nn_stack_dup_by_name_F(result);

		result = nn_mark_dup_F(result, "coil");
		result = nn_mark_dup_F(result, "pattern");
	} else {

		result = nn_set_input_name_F(result, 0, "zero_filled");
		result = nn_stack_dup_by_name_F(result);
	}

	auto nn_zf = nn_modl_zf_create(config, dims, udims, status);

	result = nn_chain2_swap_FF(nn_zf, 0, NULL, result, 0, "zero_filled");
	result = nn_stack_dup_by_name_F(result);

	return result;
}


static complex float get_lambda(long NI, const float* x[NI])
{
	complex float result = 0;
	md_copy(1, MD_SINGLETON_DIMS(1), &result, x[4], CFL_SIZE);
	return result;
}

static complex float get_infinity(long NI, const float* x[NI])
{
	UNUSED(x);
	UNUSED(NI);
	return INFINITY;
}
#if 1
static nn_t create_modl_val_loss(struct modl_s* modl, const char**valid_files)
{
	long kdims[5];
	long cdims[5];
	long udims[5];
	long pdims[5];

	complex float* val_kspace = load_cfl(valid_files[0], 5, kdims);
	complex float* val_coil = load_cfl(valid_files[1], 5, cdims);
	complex float* val_pattern = load_cfl(valid_files[2], 5, pdims);
	complex float* val_ref = load_cfl(valid_files[3], 5, udims);

	auto valid_loss = nn_modl_create(modl, kdims, udims, STAT_TEST);

	const struct nlop_s* loss = nlop_combine_FF(nlop_mse_create(5, udims, ~0ul), nlop_mse_create(5, udims, ~0ul));
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);
	loss = nlop_dup_F(loss, 0, 2);
	loss = nlop_dup_F(loss, 1, 2);

	if(modl->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, udims);

		auto nn_norm_ref = nn_from_nlop(nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, nlop_tenmul_create(5, udims, udims, sdims), 1));

		valid_loss = nn_chain2_FF(valid_loss, 0, "normalize_scale", nn_norm_ref, 1, NULL);
		valid_loss = nn_chain2_FF(valid_loss, 0, NULL, nn_from_nlop_F(loss), 0, NULL);
		valid_loss = nn_link_F(valid_loss, 2, NULL, 0, NULL);
		valid_loss = nn_set_out_type_F(valid_loss, 0, NULL, OUT_OPTIMIZE);

	} else {

		valid_loss = nn_chain2_FF(valid_loss, 0, NULL, nn_from_nlop_F(loss), 0, NULL);
	}

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL, 5, udims, true, val_ref);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "kspace", 5, kdims, true, val_kspace);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "coil", 5, cdims, true, val_coil);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "pattern", 5, pdims, true, val_pattern);

	//batchnorm out
	valid_loss = nn_del_out_bn_F(valid_loss);

	unmap_cfl(5, udims, val_ref);
	unmap_cfl(5, kdims, val_kspace);
	unmap_cfl(5, cdims, val_coil);
	unmap_cfl(5, pdims, val_pattern);

	return valid_loss;
}
#else
static nn_t create_modl_val_loss(struct modl_s* modl, const char**valid_files)
{
	long kdims[5];
	long cdims[5];
	long udims[5];
	long pdims[5];

	complex float* val_kspace = load_cfl(valid_files[0], 5, kdims);
	complex float* val_coil = load_cfl(valid_files[1], 5, cdims);
	complex float* val_pattern = load_cfl(valid_files[2], 5, pdims);
	complex float* val_ref = load_cfl(valid_files[3], 5, udims);

	auto valid_loss = nn_modl_create(modl, kdims, udims, STAT_TEST);

	const struct nlop_s* loss = nlop_mse_create(5, udims, ~0ul);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);

	if(modl->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, udims);

		auto nn_norm_ref = nn_from_nlop(nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, nlop_tenmul_create(5, udims, udims, sdims), 1));

		valid_loss = nn_chain2_FF(valid_loss, 0, "normalize_scale", nn_norm_ref, 1, NULL);
		valid_loss = nn_chain2_FF(valid_loss, 0, NULL, nn_from_nlop_F(loss), 0, NULL);
		valid_loss = nn_link_F(valid_loss, 1, NULL, 0, NULL);
		valid_loss = nn_set_out_type_F(valid_loss, 0, NULL, OUT_OPTIMIZE);

	} else {

		valid_loss = nn_chain2_FF(valid_loss, 0, NULL, nn_from_nlop_F(loss), 0, NULL);
	}

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL, 5, udims, true, val_ref);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "kspace", 5, kdims, true, val_kspace);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "coil", 5, cdims, true, val_coil);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "pattern", 5, pdims, true, val_pattern);

	//batchnorm out
	valid_loss = nn_del_out_bn_F(valid_loss);

	unmap_cfl(5, udims, val_ref);
	unmap_cfl(5, kdims, val_kspace);
	unmap_cfl(5, cdims, val_coil);
	unmap_cfl(5, pdims, val_pattern);

	return valid_loss;
}
#endif

static nn_t nn_modl_train_op_create(const struct modl_s* modl, const long dims[5], const long udims[5])
{
	auto nn_train = nn_modl_create(modl, dims, udims, STAT_TRAIN);
	
	if(modl->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, dims);

		auto nn_norm_ref = nn_from_nlop(nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, nlop_tenmul_create(5, udims, udims, sdims), 1));

		nn_train = nn_chain2_FF(nn_train, 0, "normalize_scale", nn_norm_ref, 1, NULL);
		nn_train = nn_chain2_FF(nn_train, 1, NULL, nn_from_nlop(nlop_mse_create(5, udims, ~0ul)), 1, NULL);
		nn_train = nn_link_F(nn_train, 1, NULL, 0, NULL);
		nn_train = nn_set_out_type_F(nn_train, 0, NULL, OUT_OPTIMIZE);

	} else {

		nn_train = nn_loss_mse_append(nn_train, 0, NULL, ~0ul);
	}

	return nn_train;
}

static const struct nlop_s* nn_modl_apply_op_create(const struct modl_s* modl, const long dims[5], const long udims[5])
{
	auto nn_apply = nn_modl_create(modl, dims, udims, STAT_TEST);

	if (modl->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, dims);
		auto nn_norm_ref = nn_from_nlop(nlop_tenmul_create(5, udims, udims, sdims));

		nn_apply = nn_chain2_FF(nn_apply, 0, NULL, nn_norm_ref, 0, NULL);
		nn_apply = nn_link_F(nn_apply, 0, "normalize_scale", 0, NULL);
	}

	return nn_get_nlop_wo_weights(nn_apply, modl->weights, false);
}



void train_nn_modl(	struct modl_s* modl, struct iter6_conf_s* train_conf,
			const long udims[5], _Complex float* ref,
			const long kdims[5], _Complex float* kspace, const _Complex float* coil,
			const long pdims[5], const _Complex float* pattern,
			long Nb, const char** valid_files)
{
	long Nt = kdims[4]; // number datasets

	long nkdims[5];
	long nudims[5];

	md_copy_dims(5, nkdims, kdims);
	md_copy_dims(5, nudims, udims);

	nkdims[4] = Nb;
	nudims[4] = Nb;

	modl->share_pattern = pdims[4] == 1;

	auto nn_train = nn_modl_train_op_create(modl, nkdims, nudims);

	if (NULL != modl->draw_graph_filename)
		nn_export_graph(modl->draw_graph_filename, nn_train, graph_default);

	//create batch generator
	const complex float* train_data[] = {ref, kspace, coil, pattern};
	const long* train_dims[] = {	nn_generic_domain(nn_train, 0, NULL)->dims,
					nn_generic_domain(nn_train, 0, "kspace")->dims,
					nn_generic_domain(nn_train, 0, "coil")->dims,
					nn_generic_domain(nn_train, 0, "pattern")->dims};

	auto batch_generator = batch_gen_create_from_iter(train_conf, 4, 5, train_dims, train_data, Nt, 0);

	//setup for iter algorithm
	int NI = nn_get_nr_in_args(nn_train);
	int NO = nn_get_nr_out_args(nn_train);

	float* src[NI];

	for (int i = 0; i < modl->weights->N; i++)
		src[i + 4] = (float*)modl->weights->tensors[i];

	enum IN_TYPE in_type[NI];
	const struct operator_p_s* projections[NI];
	enum OUT_TYPE out_type[NO];

	nn_get_in_types(nn_train, NI, in_type);
	nn_get_out_types(nn_train, NO, out_type);

	for (int i = 0; i < 4; i++) {

		src[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
		projections[i] = NULL;
	}

	for (int i = 4; i < NI; i++)
		projections[i] = NULL;

	bool lambda = (modl->use_dc) || (modl->init_tickhonov);

	const struct monitor_value_s* value_monitors[2];
	value_monitors[0] = monitor_iter6_function_create(lambda ? get_lambda : get_infinity, true, "lambda");

	if (NULL != valid_files) {

		auto nn_validation_loss = create_modl_val_loss(modl, valid_files);
		value_monitors[1] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, 2, (const char*[2]){"val loss (mag)", "val loss"});
		nn_free(nn_validation_loss);
	} else {

		value_monitors[1] = NULL;
	}

	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, (NULL != valid_files) ? 2 : 1, value_monitors);

	debug_printf(DP_INFO, "Train MoDL\n");
	nn_debug(DP_INFO, nn_train);

	iter6_adam(train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, NO, out_type, Nb, Nt / Nb, batch_generator, monitor);

	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}


void apply_nn_modl(	struct modl_s* modl,
			const long udims[5], complex float* out,
			const long kdims[5], const complex float* kspace, const complex float* coil, const long pdims[5], const complex float* pattern)
{

	modl->share_pattern = (1 == pdims[4]);

	auto nlop_modl = nn_modl_apply_op_create(modl, kdims, udims);

	complex float* out_tmp = md_alloc_sameplace(5, udims, CFL_SIZE, modl->weights->tensors[0]);
	complex float* kspace_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, modl->weights->tensors[0]);
	complex float* coil_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, modl->weights->tensors[0]);
	complex float* pattern_tmp = md_alloc_sameplace(5, pdims, CFL_SIZE, modl->weights->tensors[0]);

	md_copy(5, kdims, kspace_tmp, kspace, CFL_SIZE);
	md_copy(5, kdims, coil_tmp, coil, CFL_SIZE);
	md_copy(5, pdims, pattern_tmp, pattern, CFL_SIZE);

	complex float* args[4];

	args[0] = out_tmp;
	args[1] = kspace_tmp;
	args[2] = coil_tmp;
	args[3] = pattern_tmp;

	nlop_generic_apply_select_derivative_unchecked(nlop_modl, 4, (void**)args, 0, 0);

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
				long Nb)
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

		apply_nn_modl(modl, udims1, out, kdims1, kspace, coil, pdims1, pattern);

		out += md_calc_size(5, udims1);
		kspace += md_calc_size(5, kdims1);
		coil += md_calc_size(5, kdims1);
		if (1 < pdims[4])
			pattern += md_calc_size(5, pdims1);

		Nt -= Nb_tmp;
	}
}


void init_nn_modl(struct modl_s* modl)
{
	auto network = nn_modl_create(modl, MD_DIMS(modl->Kx, modl->Ky, modl->Kz, 1, 1), MD_DIMS(modl->Kx, modl->Ky, modl->Kz, 1, 1), STAT_TRAIN);
	modl->weights = nn_weights_create_from_nn(network);
	nn_init(network, modl->weights);
	nn_free(network);
}

void nn_modl_move_gpucpu(struct modl_s* modl, bool gpu) {

#ifdef USE_CUDA
	if (gpu)
		move_gpu_nn_weights(modl->weights);
#else
	UNUSED(modl);
	if (gpu)
		error("Compiled without GPU support!");
#endif
}

extern void nn_modl_store_weights(struct modl_s* modl, const char* name)
{
	dump_nn_weights(name, modl->weights);
}

void nn_modl_load_weights(struct modl_s* modl, const char* name, bool overwrite_parameters)
{
	auto loaded_weights = load_nn_weights(name);

	if (overwrite_parameters) {

		modl->Nf = loaded_weights->iovs[1]->dims[0];
		modl->Nl = loaded_weights->iovs[2]->dims[5] + 2;

		modl->Kx = loaded_weights->iovs[1]->dims[2];
		modl->Ky = loaded_weights->iovs[1]->dims[3];
		modl->Kz = loaded_weights->iovs[1]->dims[4];

		modl->reinsert_zerofilled = (2 == loaded_weights->iovs[1]->dims[0]);
	}

	auto network = nn_modl_create(modl, MD_DIMS(modl->Kx, modl->Ky, modl->Kz, 1, 1), MD_DIMS(modl->Kx, modl->Ky, modl->Kz, 1, 1), STAT_TRAIN);
	modl->weights = nn_weights_create_from_nn(network);

	nn_weights_copy(modl->weights, loaded_weights);

	nn_weights_free(loaded_weights);
	nn_free(network);
}

extern void nn_modl_free_weights(struct modl_s* modl)
{
	nn_weights_free(modl->weights);
}
