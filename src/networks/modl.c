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

#include "networks/misc.h"

#include "modl.h"


const struct modl_s modl_default = {

	.Nt = 10,
	.Nl = 5,
	.Nf = 32,

	.Kx = 3,
	.Ky = 3,
	.Kz = 1,

	.normal_inversion_iter_conf = NULL,

	.lambda_init = .05,
	.lambda_fixed = -1.,

	.shared_weights = true,
	.shared_lambda = true,
	.share_pattern = false,

	.reinsert_zerofilled = false,
	.init_tickhonov = false,
	.batch_norm = true,
	.residual_network =true,

	.use_dc = true,

	.normalize = false,

	.low_mem = false,

	.regrid = false,
};

const char* sorted_weight_names[] = {
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
			};
/**
 * Returns residual block (called DW in MoDL)
 *
 * Input tensors:
 *
 * INDEX_0: 	idims:	(Ux, Uy, Uz, 1, Nb)
 * zero_filled:	idims:	(Ux, Uy, Uz, 1, Nb) [Optional]
 * weights as listed in "sorted_weight_names"
 *
 * Output tensors:
 *
 * INDEX_0:	idims:	(Ux, Uy, Uz, 1, Nb)
 * bn_0		[batchnorm output]
 * bn_i
 * bn_n
 */
static nn_t residual_create(const struct modl_s* config, const long idims[5], enum NETWORK_STATUS status){

	long idims_w[5] = {1, idims[0], idims[1], idims[2], idims[4]};

	nn_t result = 0;

	if (config->reinsert_zerofilled) {

		long idims_w2[5] = {2, idims[0], idims[1], idims[2], idims[4]};
		const struct nlop_s* nlop_init = nlop_stack_create(5, idims_w2, idims_w, idims_w, 0);
		nlop_init = nlop_reshape_in_F(nlop_init, 0, 5, idims);
		nlop_init = nlop_reshape_in_F(nlop_init, 1, 5, idims);

		result = nn_from_nlop_F(nlop_init);
		result = nn_set_input_name_F(result, 1, "zero_filled");

	} else {

		result = nn_from_nlop_F(nlop_from_linop_F(linop_reshape_create(5, idims_w, 5, idims)));
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

	result = nn_reshape_out_F(result, 0, NULL, 5, idims);

	if (config->residual_network) {

		result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(5, idims, 1, 1)), 1, NULL);
		result = nn_dup_F(result, 0, NULL, 1, NULL);
	}

	// Append dims for stacking over modl iterations (non-shared weights)
	for (unsigned int i = 0; i < ARRAY_SIZE(sorted_weight_names); i++)
		result = nn_append_singleton_dim_in_if_exists_F(result, sorted_weight_names[i]);
	
	result = nn_append_singleton_dim_out_if_exists_F(result, "bn_0");
	result = nn_append_singleton_dim_out_if_exists_F(result, "bn_i");
	result = nn_append_singleton_dim_out_if_exists_F(result, "bn_n");

	// sort weight inputs as specified by list
	result = nn_sort_inputs_by_list_F(result, ARRAY_SIZE(sorted_weight_names), sorted_weight_names);

	return nn_checkpoint_F(result, true, config->low_mem && (1 < config->Nt));
}

static struct config_nlop_mri_s get_modl_mri_conf(const struct modl_s* modl)
{
	struct config_nlop_mri_s conf = conf_nlop_mri_simple;
	if (!modl->share_pattern)
		conf.pattern_flags = ~MD_BIT(3);

	conf.regrid = modl->regrid;

	return conf;	
}

static struct config_nlop_mri_dc_s get_modl_mri_dc_conf(const struct modl_s* modl)
{
	struct config_nlop_mri_dc_s conf = conf_nlop_mri_dc_simple;

	conf.iter_conf = CAST_DOWN(iter_conjgrad_conf, modl->normal_inversion_iter_conf);
	conf.lambda_fixed = modl->lambda_fixed;
	conf.lambda_init = modl->lambda_init;

	return conf;	
}

/**
 * Returns dataconsistency block (called DC in MoDL)
 *
 * Input tensors:
 *
 * INDEX_0: 	idims:	(Ux, Uy, Uz,  1, Nb)
 * zero_filled:	idims:	(Ux, Uy, Uz,  1, Nb)
 * coil:	dims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, 1 / Nb)
 * lambda:		(1)
 *
 * Output tensors:
 *
 * INDEX_0:	idims:	(Ux, Uy, Uz, 1, Nb)
 */
static nn_t data_consistency_modl_create(const struct modl_s* config,const long dims[5], const long idims[5])
{
	struct config_nlop_mri_s mri_conf = get_modl_mri_conf(config);
	struct config_nlop_mri_dc_s mri_conf_dc = get_modl_mri_dc_conf(config);

	auto nlop_dc = mri_normal_inversion_create(5, dims, idims, &mri_conf, &mri_conf_dc); // in: lambda * zn + zero_filled, coil, pattern[, lambda]; out: x(n+1)

	nlop_dc = nlop_chain2_swap_FF(nlop_zaxpbz_create(5, idims, 1., 1.), 0, nlop_dc, 0); // in: lambda * zn, zero_filled, coil, pattern[, lambda]; out: x(n+1)

	const struct nlop_s* nlop_scale_lambda = nlop_tenmul_create(5, idims, idims, MD_SINGLETON_DIMS(5));
	nlop_scale_lambda = nlop_chain2_FF(nlop_from_linop_F(linop_zreal_create(5, MD_SINGLETON_DIMS(5))), 0, nlop_scale_lambda, 1);
	nlop_scale_lambda = nlop_reshape_in_F(nlop_scale_lambda, 1, 1, MD_SINGLETON_DIMS(1)); // in: zn, lambda; out: lambda * zn

	nlop_dc = nlop_chain2_FF(nlop_scale_lambda, 0, nlop_dc, 0); // in: zero_filled, coil, pattern[, lambda], zn, lambda; out: x(n+1)
	
	if (-1. == config->lambda_fixed) {
	
		nlop_dc = nlop_dup_F(nlop_dc, 3, 5);
		nlop_dc = nlop_shift_input_F(nlop_dc, 0, 4);
	} else {

		nlop_dc = nlop_shift_input_F(nlop_dc, 0, 3);		
	}

	auto result = nn_from_nlop_F(nlop_dc);
	result = nn_set_input_name_F(result, 1, "zero_filled");
	result = nn_set_input_name_F(result, 1, "coil");
	result = nn_set_input_name_F(result, 1, "pattern");
	result = nn_set_input_name_F(result, 1, "lambda");

	result = nn_set_in_type_F(result, 0, "lambda", IN_OPTIMIZE);
	result = nn_set_initializer_F(result, 0, "lambda", init_const_create((-1 != config->lambda_fixed) ? config->lambda_fixed : config->lambda_init));

	return result;  // in:  zn, zero_filled, coil, pattern, lambda; out: x(n+1)
}

/**
 * Returns one cell of MoDL iterations
 *
 * Input tensors:
 *
 * INDEX_0: 	idims:	(Ux, Uy, Uz,  1, Nb)
 * zero_filled:	idims:	(Ux, Uy, Uz,  1, Nb)
 * coil:	dims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, 1 / Nb)
 * lambda:		(1)
 * weights as listed in "sorted_weight_names"
 *
 * Output tensors:
 *
 * INDEX_0:	idims:	(Ux, Uy, Uz, 1, Nb)
 * bn_0		[batchnorm output]
 * bn_i
 * bn_n
 */
static nn_t nn_modl_cell_create(const struct modl_s* config, const long dims[5], const long idims[5], enum NETWORK_STATUS status)
{
	if (!config->use_dc)
		return residual_create(config, idims, status);

	auto dw = residual_create(config, idims, status);
	auto dc = data_consistency_modl_create(config, dims, idims);

	if (nn_is_name_in_in_args(dw, "zero_filled"))
		dw = nn_mark_dup_F(dw, "zero_filled");

	nn_t result = nn_chain2_FF(dw, 0, NULL, dc, 0, NULL);
	result = nn_stack_dup_by_name_F(result);

	return result;
}

/**
 * Computes zero filled reconstruction
 *
 * Input tensors:
 *
 * kspace: 	idims:	(Nx, Ny, Nz,  1, Nb)
 * coil:	dims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, 1 / Nb)
 *
 * Output tensors:
 *
 * INDEX_0:		idims:	(Ux, Uy, Uz, 1, Nb)
 * normalize_scale:		( 1,  1,  1, 1, Nb)
 */
static nn_t nn_modl_zf_create(const struct modl_s* config,const long dims[5], const long idims[5], enum NETWORK_STATUS status)
{
	UNUSED(status);
	
	struct config_nlop_mri_s mri_conf = get_modl_mri_conf(config);

	auto nlop_zf = nlop_mri_adjoint_create(5, dims, idims, &mri_conf);
	auto nn_zf = nn_from_nlop_F(nlop_zf);
	nn_zf = nn_set_input_name_F(nn_zf, 0, "kspace");
	nn_zf = nn_set_input_name_F(nn_zf, 0, "coil");
	nn_zf = nn_set_input_name_F(nn_zf, 0, "pattern");

	if (config->normalize) {

		auto nn_normalize = nn_from_nlop_F(nlop_norm_zmax_create(5, idims, MD_BIT(4), true));
		nn_normalize = nn_set_output_name_F(nn_normalize, 1, "normalize_scale");
		nn_zf = nn_chain2_FF(nn_zf, 0, NULL, nn_normalize, 0, NULL);
	}

	return nn_zf;
}

static nn_t nn_modl_create(const struct modl_s* config,const long dims[5], const long idims[5], enum NETWORK_STATUS status)
{
	auto result = nn_modl_cell_create(config, dims, idims, status);

	for (int i = 1; i < config->Nt; i++) {

		auto tmp = nn_modl_cell_create(config, dims, idims, status);

		// batchnorm weights are always stacked
		tmp = nn_mark_stack_input_if_exists_F(tmp, "bn_0");
		tmp = nn_mark_stack_input_if_exists_F(tmp, "bn_i");
		tmp = nn_mark_stack_input_if_exists_F(tmp, "bn_n");
		tmp = nn_mark_stack_input_if_exists_F(tmp, "gamma");
		tmp = nn_mark_stack_output_if_exists_F(tmp, "bn_0");
		tmp = nn_mark_stack_output_if_exists_F(tmp, "bn_i");
		tmp = nn_mark_stack_output_if_exists_F(tmp, "bn_n");
		
		// Dup or Stack other weights
		for (unsigned int i = 0; i < ARRAY_SIZE(sorted_weight_names); i++)
			tmp = (config->shared_weights ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(tmp, sorted_weight_names[i]);

		tmp = nn_mark_dup_if_exists_F(tmp, "zero_filled");
		tmp = nn_mark_dup_if_exists_F(tmp, "coil");
		tmp = nn_mark_dup_if_exists_F(tmp, "pattern");
		tmp = (config->shared_lambda ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(tmp, "lambda");

		result = nn_chain2_FF(result, 0, NULL, tmp, 0, NULL);
		result = nn_stack_dup_by_name_F(result);
	}

	if (config -> init_tickhonov) {

		result = nn_mark_dup_if_exists_F(result, "zero_filled");
		result = nn_mark_dup_if_exists_F(result, "coil");
		result = nn_mark_dup_if_exists_F(result, "pattern");
		result = (config->shared_lambda ? nn_mark_dup_if_exists_F : nn_mark_stack_input_if_exists_F)(result, "lambda");

		auto dc = data_consistency_modl_create(config, dims, idims);
		auto iov = nn_generic_domain(dc, 0, NULL);
		complex float* zeros = md_alloc(iov->N, iov->dims, iov->size);
		md_clear(iov->N, iov->dims, zeros, iov->size);
		dc = nn_chain2_FF(nn_from_nlop_F(nlop_const_create(iov->N, iov->dims, true, zeros)), 0, NULL, dc, 0, NULL);   // in:  zero_filled, coil, pattern, lambda; out: (A^HA + lambda)^(-1) zerofilled

		result = nn_chain2_swap_FF(dc, 0, NULL, result, 0, NULL);
		result = nn_stack_dup_by_name_F(result);
	} else {
		if (nn_is_name_in_in_args(result, "zero_filled"))
			result = nn_dup_F(result, 0, "zero_filled", 0, NULL);
		else
			result = nn_set_input_name_F(result, 0, "zero_filled"); 
	}	

	auto nn_zf = nn_modl_zf_create(config, dims, idims, status);

	result = nn_mark_dup_if_exists_F(result, "coil");
	result = nn_mark_dup_if_exists_F(result, "pattern");

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

static nn_t create_modl_val_loss(struct modl_s* modl, struct network_data_s* vf)
{
	load_network_data(vf);

	modl->share_pattern = (vf->pdims[4] == 1);

	auto valid_loss = nn_modl_create(modl, vf->kdims, vf->idims, STAT_TEST);

	const struct nlop_s* loss = nlop_combine_FF(nlop_mse_create(5, vf->idims, ~0ul), nlop_mse_create(5, vf->idims, ~0ul));
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, vf->idims, 1.e-12), 0, loss, 0);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, vf->idims, 1.e-12), 0, loss, 0);
	loss = nlop_dup_F(loss, 0, 2);
	loss = nlop_dup_F(loss, 1, 2);

	loss = nlop_combine_FF(loss, nlop_mpsnr_create(5, vf->idims, MD_BIT(4)));
	loss = nlop_dup_F(loss, 0, 2);
	loss = nlop_dup_F(loss, 1, 2);

	loss = nlop_combine_FF(loss, nlop_mssim_create(5, vf->idims, MD_DIMS(7, 7, 1, 1, 1), 7));
	loss = nlop_dup_F(loss, 0, 2);
	loss = nlop_dup_F(loss, 1, 2);

	if(modl->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, vf->idims);

		auto nn_norm_ref = nn_from_nlop_F(nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, nlop_tenmul_create(5, vf->idims, vf->idims, sdims), 1));

		valid_loss = nn_chain2_FF(valid_loss, 0, "normalize_scale", nn_norm_ref, 1, NULL);
		valid_loss = nn_chain2_FF(valid_loss, 0, NULL, nn_from_nlop_F(loss), 0, NULL);
		valid_loss = nn_link_F(valid_loss, 4, NULL, 0, NULL);
		valid_loss = nn_set_out_type_F(valid_loss, 0, NULL, OUT_OPTIMIZE);

	} else {

		valid_loss = nn_chain2_FF(valid_loss, 0, NULL, nn_from_nlop_F(loss), 0, NULL);
	}

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL, 5, vf->idims, true, vf->out);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "kspace", 5, vf->kdims, true, vf->kspace);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "coil", 5, vf->cdims, true, vf->coil);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "pattern", 5, vf->pdims, true, vf->pattern);

	//batchnorm out
	valid_loss = nn_del_out_bn_F(valid_loss);

	free_network_data(vf);

	return valid_loss;
}

static nn_t nn_modl_train_op_create(const struct modl_s* modl, const long dims[5], const long idims[5])
{
	auto nn_train = nn_modl_create(modl, dims, idims, STAT_TRAIN);
	
	if(modl->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, dims);

		auto nn_norm_ref = nn_from_nlop_F(nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, nlop_tenmul_create(5, idims, idims, sdims), 1));

		nn_train = nn_chain2_FF(nn_train, 0, "normalize_scale", nn_norm_ref, 1, NULL);
		nn_train = nn_chain2_FF(nn_train, 1, NULL, nn_from_nlop_F(nlop_mse_create(5, idims, ~0ul)), 1, NULL);
		nn_train = nn_link_F(nn_train, 1, NULL, 0, NULL);
		nn_train = nn_set_out_type_F(nn_train, 0, NULL, OUT_OPTIMIZE);

	} else {

		nn_train = nn_loss_mse_append(nn_train, 0, NULL, ~0ul);
	}

	return nn_train;
}

static const struct nlop_s* nn_modl_apply_op_create(const struct modl_s* modl, const long dims[5], const long idims[5])
{
	auto nn_apply = nn_modl_create(modl, dims, idims, STAT_TEST);

	if (modl->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, dims);
		auto nn_norm_ref = nn_from_nlop_F(nlop_tenmul_create(5, idims, idims, sdims));

		nn_apply = nn_chain2_FF(nn_apply, 0, NULL, nn_norm_ref, 0, NULL);
		nn_apply = nn_link_F(nn_apply, 0, "normalize_scale", 0, NULL);
	}
	debug_printf(DP_INFO, "Apply MoDL\n");
	nn_debug(DP_INFO, nn_apply);

	return nn_get_nlop_wo_weights_F(nn_apply, modl->weights, false);
}



void train_nn_modl(	struct modl_s* modl, struct iter6_conf_s* train_conf,
			const long idims[5], _Complex float* ref,
			const long kdims[5], _Complex float* kspace,
			const long cdims[5], const _Complex float* coil,
			const long pdims[5], const _Complex float* pattern,
			long Nb, struct network_data_s* valid_files)
{
	long Nt = kdims[4]; // number datasets

	long nkdims[5];
	long nidims[5];
	long ncdims[5];

	md_copy_dims(5, nkdims, kdims);
	md_copy_dims(5, nidims, idims);
	md_copy_dims(5, ncdims, cdims);

	nkdims[4] = Nb;
	nidims[4] = Nb;
	ncdims[4] = Nb;

	modl->share_pattern = (1 == pdims[4]);

	auto nn_train = nn_modl_train_op_create(modl, nkdims, nidims);
	if (nn_is_name_in_in_args(nn_train, "lambda"))
		nn_train = nn_set_in_type_F(nn_train, 0, "lambda", (-1 != modl->lambda_fixed) ? IN_STATIC : IN_OPTIMIZE);

	//create batch generator
	const complex float* train_data[] = {ref, kspace, coil, pattern};
	const long* train_dims[] = {	nn_generic_domain(nn_train, 0, NULL)->dims,
					nn_generic_domain(nn_train, 0, "kspace")->dims,
					nn_generic_domain(nn_train, 0, "coil")->dims,
					nn_generic_domain(nn_train, 0, "pattern")->dims};
	
	assert(md_check_equal_dims(5, ncdims, train_dims[2], ~0));

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
		value_monitors[1] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, 4, (const char*[4]){"val loss (mag)", "val loss", "mean PSNR", "ssim"});
		nn_free(nn_validation_loss);
	} else {

		value_monitors[1] = NULL;
	}

	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, (NULL != valid_files) ? 2 : 1, value_monitors);

	debug_printf(DP_INFO, "Train MoDL\n");
	nn_debug(DP_INFO, nn_train);

	train_conf->dump_flag = ~15;

	iter6_adam(train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, NO, out_type, Nb, Nt / Nb, batch_generator, monitor);

	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}


void apply_nn_modl(	struct modl_s* modl,
			const long idims[5], complex float* out,
			const long kdims[5], const complex float* kspace, 
			const long cdims[5], const complex float* coil,
			const long pdims[5], const complex float* pattern)
{

	modl->share_pattern = (1 == pdims[4]);

	auto nlop_modl = nn_modl_apply_op_create(modl, kdims, idims);

	complex float* out_tmp = md_alloc_sameplace(5, idims, CFL_SIZE, modl->weights->tensors[0]);
	complex float* kspace_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, modl->weights->tensors[0]);
	complex float* coil_tmp = md_alloc_sameplace(5, cdims, CFL_SIZE, modl->weights->tensors[0]);
	complex float* pattern_tmp = md_alloc_sameplace(5, pdims, CFL_SIZE, modl->weights->tensors[0]);

	md_copy(5, kdims, kspace_tmp, kspace, CFL_SIZE);
	md_copy(5, cdims, coil_tmp, coil, CFL_SIZE);
	md_copy(5, pdims, pattern_tmp, pattern, CFL_SIZE);

	complex float* args[4];

	args[0] = out_tmp;
	args[1] = kspace_tmp;
	args[2] = coil_tmp;
	args[3] = pattern_tmp;

	nlop_generic_apply_select_derivative_unchecked(nlop_modl, 4, (void**)args, 0, 0);

	md_copy(5, idims, out, out_tmp, CFL_SIZE);

	nlop_free(nlop_modl);

	md_free(out_tmp);
	md_free(kspace_tmp);
	md_free(coil_tmp);
	md_free(pattern_tmp);
}

void apply_nn_modl_batchwise(	struct modl_s* modl,
				const long idims[5], complex float * out,
				const long kdims[5], const complex float* kspace, 
				const long cdims[5], const complex float* coil,
				const long pdims[5], const complex float* pattern,
				long Nb)
{
	long Nt = kdims[4];
	while (0 < Nt) {

		long kdims1[5];
		long cdims1[5];
		long idims1[5];
		long pdims1[5];

		md_copy_dims(5, kdims1, kdims);
		md_copy_dims(5, cdims1, cdims);
		md_copy_dims(5, idims1, idims);
		md_copy_dims(5, pdims1, pdims);

		long Nb_tmp = MIN(Nt, Nb);

		kdims1[4] = Nb_tmp;
		cdims1[4] = Nb_tmp;
		idims1[4] = Nb_tmp;
		pdims1[4] = MIN(pdims1[4], Nb_tmp);

		apply_nn_modl(modl, idims1, out, kdims1, kspace, cdims1, coil, pdims1, pattern);

		out += md_calc_size(5, idims1);
		kspace += md_calc_size(5, kdims1);
		coil += md_calc_size(5, cdims1);
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

		modl->reinsert_zerofilled = (2 == loaded_weights->iovs[1]->dims[1]);
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
