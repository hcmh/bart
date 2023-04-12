#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "grecon/losses.h"

#include "iter/italgos.h"

#include "misc/mri.h"
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
#include <string.h>

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/grad.h"
#include "linops/someops.h"
#include "linops/sum.h"
#include "linops/fmac.h"


#include "iter/iter.h"
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
#include "nn/data_list.h"

#include "networks/misc.h"
#include "networks/losses.h"
#include "networks/cnn.h"
#include "networks/unet.h"

#include "noir/model2.h"
#include "noir/recon2.h"
#include "noir/model_net.h"

#include "nlinvnet.h"


struct nlinvnet_s nlinvnet_config_opts = {

	// Training configuration
	.train_conf = NULL,
	.train_loss = &loss_option,
	.valid_loss = &val_loss_option,

	// Self-Supervised k-Space
	.ksp_training = false,
	.ksp_split = -1.,
	.ksp_shared_dims = 0.,
	.exclude_center = 0.,
	.fixed_splitting = true,
	.ksp_mask_time = {0, 0},
	.l2loss_reco = 0.,
	.l2loss_data = 0.,

	.tvflags = TIME_FLAG,
	.tvloss = 0.,

	// Network block
	.network = NULL,
	.weights = NULL,
	.share_weights = true,
	.ref_init = false,
	.reg_diff_coils = false,
	.lambda = -0.01,

	.conv_time = 0,
	.conv_padding = PAD_SAME,
	
	// NLINV configuration
	.conf = NULL,
	.model = NULL,
	.iter_conf = NULL,
	.iter_conf_net = NULL,
	.cgtol = 0.1,
	.iter_net = 3,		//# of iterations with network
	.fix_coils_sense = false,
	.sense_mean = 0,
	.oversampling_coils = 2.,

	.scaling = -100.,
	.real_time_init = false,

	.gpu = false,
	.normalize_rss = false,
};

static void nlinvnet_init(struct nlinvnet_s* nlinvnet)
{
	nlinvnet->iter_conf_net = TYPE_ALLOC(struct iter_conjgrad_conf);
	*(nlinvnet->iter_conf_net) = iter_conjgrad_defaults;
	nlinvnet->iter_conf_net->INTERFACE.alpha = 0.;
	nlinvnet->iter_conf_net->l2lambda = 0.;
	nlinvnet->iter_conf_net->maxiter = nlinvnet->conf->cgiter;
	nlinvnet->iter_conf_net->tol = 0.;

	nlinvnet->iter_conf = TYPE_ALLOC(struct iter_conjgrad_conf);
	*(nlinvnet->iter_conf) = iter_conjgrad_defaults;
	nlinvnet->iter_conf->INTERFACE.alpha = 0.;
	nlinvnet->iter_conf->l2lambda = 0.;
	nlinvnet->iter_conf->maxiter = nlinvnet->conf->cgiter;
	nlinvnet->iter_conf->tol = nlinvnet->cgtol;

	if (NULL == get_loss_from_option())
			nlinvnet->train_loss->weighting_mse=1.;

	if (NULL == get_val_loss_from_option())
		nlinvnet->valid_loss = &loss_image_valid;

	assert(0 == nlinvnet->iter_conf_net->tol);
}


void nlinvnet_init_model_cart(struct nlinvnet_s* nlinvnet, int N,
	const long pat_dims[N],
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N])
{
	nlinvnet_init(nlinvnet);

	struct noir2_model_conf_s model_conf = noir2_model_conf_defaults;
	model_conf.fft_flags_noncart = 0;
	model_conf.fft_flags_cart = FFT_FLAGS | ((nlinvnet->conf->sms || nlinvnet->conf->sos) ? TIME2_FLAG : 0);
	model_conf.rvc = nlinvnet->conf->rvc;
	model_conf.sos = nlinvnet->conf->sos;
	model_conf.a = nlinvnet->conf->a;
	model_conf.b = nlinvnet->conf->b;
	model_conf.noncart = nlinvnet->conf->noncart;
	model_conf.nufft_conf = nlinvnet->conf->nufft_conf;
	model_conf.oversampling_coils = nlinvnet->oversampling_coils;

	nlinvnet->model = noir2_net_config_create(N, NULL, pat_dims, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, TIME_FLAG, &model_conf);
}

void nlinvnet_init_model_noncart(struct nlinvnet_s* nlinvnet, int N,
	const long trj_dims[N],
	const long wgh_dims[N],
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N])
{
	nlinvnet_init(nlinvnet);

	struct noir2_model_conf_s model_conf = noir2_model_conf_defaults;
	model_conf.fft_flags_noncart = FFT_FLAGS;
	model_conf.fft_flags_cart = ((nlinvnet->conf->sms || nlinvnet->conf->sos) ? TIME2_FLAG : 0);
	model_conf.rvc = nlinvnet->conf->rvc;
	model_conf.sos = nlinvnet->conf->sos;
	model_conf.a = nlinvnet->conf->a;
	model_conf.b = nlinvnet->conf->b;
	
	model_conf.noncart = nlinvnet->conf->noncart;
	model_conf.nufft_conf = nlinvnet->conf->nufft_conf;
	model_conf.oversampling_coils = nlinvnet->oversampling_coils;

	nlinvnet->model = noir2_net_config_create(N, trj_dims, wgh_dims, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, TIME_FLAG, &model_conf);
}

static nn_t nlinvnet_sort_args_F(nn_t net)
{

	const char* data_names[] =
		{
			"ref",
			"ksp",
			"pat",
			"trj",
			"loss_mask",
			"prev_frames"
		};

	int N = nn_get_nr_named_in_args(net);
	const char* sorted_names[N + ARRAY_SIZE(data_names) + 2];

	nn_get_in_names_copy(N, sorted_names + ARRAY_SIZE(data_names) + 2, net);
	for (int i = 0; i < (int)ARRAY_SIZE(data_names); i++)
		sorted_names[i] = data_names[i];

	sorted_names[ARRAY_SIZE(data_names)] = "lam";
	sorted_names[ARRAY_SIZE(data_names) + 1] = "alp";

	net = nn_sort_inputs_by_list_F(net, N + ARRAY_SIZE(data_names) + 2, sorted_names);

	for (int i = 0; i < N; i++)
		xfree(sorted_names[i + ARRAY_SIZE(data_names) + 2]);

	for (int i = 0; i < (int)ARRAY_SIZE(data_names); i++)
		if (nn_is_name_in_in_args(net, data_names[i]))
			net = nn_set_in_type_F(net, 0, data_names[i], IN_BATCH_GENERATOR);


	N = nn_get_nr_named_out_args(net);
	const char* out_names[7 + N];

	out_names[0] = "ksp";
	out_names[1] = "cim";
	out_names[2] = "img";
	out_names[3] = "col";
	out_names[4] = "ref_img";
	out_names[5] = "ref_col";
	out_names[6] = "prev_frames";

	nn_get_out_names_copy(N, out_names + 7, net);

	net = nn_sort_outputs_by_list_F(net, 7 + N, out_names);

	for (int i = 0; i < N; i++)
		xfree(out_names[7 + i]);

	net = nn_sort_inputs_F(net);
	net = nn_sort_outputs_F(net);

	if (nn_is_name_in_in_args(net, "lam")) {

		auto iov = nn_generic_domain(net, 0, "lam");
		auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
		net = nn_set_prox_op_F(net, 0, "lam", prox_conv);
	}

	return net;
}


static nn_t nlinvnet_network_create(const struct nlinvnet_s* nlinvnet, unsigned int N, const long _img_dims[N], enum NETWORK_STATUS status, bool reference)
{
	nn_t network = NULL;

	if (0 < nlinvnet->conv_time) {

		int window_dim = TIME_DIM;
		while (1 != _img_dims[++window_dim])
			assert(BATCH_DIM > 1);

		const struct linop_s* lop_prep = NULL;
		long pos = 0; //position of the feed through frame (residual network)
		
		if (PAD_CAUSAL == nlinvnet->conv_padding) {

			pos = nlinvnet->conv_time - 1;
			lop_prep = linop_padding_create_onedim(N, _img_dims, PAD_CAUSAL, TIME_DIM, nlinvnet->conv_time - 1, 0);
		}
		
		if (PAD_SAME == nlinvnet->conv_padding) {

			assert(1 == nlinvnet->conv_time % 2);
			pos = nlinvnet->conv_time / 2;
			lop_prep = linop_padding_create_onedim(N, _img_dims, PAD_SAME, TIME_DIM, nlinvnet->conv_time / 2, nlinvnet->conv_time / 2);
		}
		
		lop_prep = linop_chain_FF(lop_prep, linop_hankelization_create(N, linop_codomain(lop_prep)->dims, TIME_DIM, window_dim, nlinvnet->conv_time));
		
		const struct nlop_s* nlop_prep;
		
		if (reference) {

			long stack_dims[N];
			md_copy_dims(N, stack_dims, linop_codomain(lop_prep)->dims);
			stack_dims[window_dim]++;

			nlop_prep = nlop_stack_create(N, stack_dims, _img_dims, linop_codomain(lop_prep)->dims, window_dim);
			nlop_prep = nlop_chain2_FF(nlop_from_linop_F(lop_prep), 0, nlop_prep, 1);
			pos = 0;
		} else 
			nlop_prep = nlop_from_linop_F(lop_prep);

		lop_prep = linop_transpose_create(N, TIME_DIM, window_dim, nlop_generic_codomain(nlop_prep, 0)->dims);

		long img_dims[N];
		md_copy_dims(N, img_dims, linop_codomain(lop_prep)->dims);
		img_dims[BATCH_DIM] *= img_dims[window_dim];
		img_dims[window_dim] = 1;

		lop_prep = linop_chain_FF(lop_prep, linop_reshape2_create(N, MD_BIT(window_dim) | BATCH_FLAG, img_dims, linop_codomain(lop_prep)->dims));
		nlop_prep = nlop_chain2_FF(nlop_prep, 0, nlop_from_linop_F(lop_prep), 0);

		network = network_create(nlinvnet->network, N, img_dims, N, img_dims, status);
		network = nn_chain2_FF(nn_from_nlop_F(nlop_prep), 0, NULL, network, 0, NULL);

		network = nn_chain2_FF(network, 0, NULL, nn_from_linop_F(linop_slice_one_create(N, TIME_DIM, pos, nn_generic_codomain(network, 0, NULL)->dims)), 0, NULL);
		network = nn_chain2_FF(network, 0, NULL, nn_from_linop_F(linop_reshape2_create(N, BATCH_FLAG | TIME_FLAG , _img_dims, nn_generic_codomain(network, 0, NULL)->dims)), 0, NULL);
	} else {

		long img_dims[N];
		md_copy_dims(N, img_dims, _img_dims);
		if (reference)
			img_dims[COEFF_DIM] *= 2;

		network = network_create(nlinvnet->network, N, _img_dims, N, img_dims, status);

		if (reference)
			network = nn_chain2_FF(nn_from_nlop_F(nlop_stack_create(N, img_dims, _img_dims, _img_dims, COEFF_DIM)), 0, NULL, network, 0, NULL);
	}

	return network;
}

static nn_t nlinvnet_get_network_step(const struct nlinvnet_s* nlinvnet, struct noir2_net_s* model, enum NETWORK_STATUS status, bool reference)
{
	int N = noir2_net_get_N(model);
	assert(N == DIMS);

	long img_dims[N];
	noir2_net_get_img_dims(model, N, img_dims);

	auto network = nlinvnet_network_create(nlinvnet, N, img_dims, status, reference);

	int N_in_names = nn_get_nr_named_in_args(network);
	int N_out_names = nn_get_nr_named_out_args(network);
	const char* in_names[N_in_names];
	const char* out_names[N_out_names];
	nn_get_in_names_copy(N_in_names, in_names, network);
	nn_get_out_names_copy(N_out_names, out_names, network);

	for (int i = 0; i < N_in_names; i++) {

		network = nn_append_singleton_dim_in_F(network, 0, in_names[i]);
		xfree(in_names[i]);
	}

	for (int i = 0; i < N_out_names; i++) {

		network = nn_append_singleton_dim_out_F(network, 0, out_names[i]);
		xfree(out_names[i]);
	}

	if ((nlinvnet->reg_diff_coils) || (0 < nlinvnet->sense_mean)) {

		nn_t join = nn_from_nlop_F(noir_join_create(model));
		network = nn_chain2_FF(network, 0, NULL, join, 0, NULL);

		nn_t split_ref = nn_from_nlop_F(noir_split_create(model));
		network = nn_chain2_FF(split_ref, 1, NULL, network, 0, NULL);
		network = nn_link_F(network, 1, NULL, 0, NULL);

		if (reference) {

			network = nn_set_input_name_F(network, 1, "ref_x");
			nn_t split = nn_from_nlop_F(noir_extract_img_create(model));
			network = nn_chain2_FF(split, 0, NULL, network, 0, NULL);
		}

	} else {

		nn_t join = nn_from_nlop_F(noir_set_img_create(model));
		network = nn_chain2_FF(network, 0, NULL, join, 0, NULL);

		nn_t split = nn_from_nlop_F(noir_extract_img_create(model));
		network = nn_chain2_FF(split, 0, NULL, network, reference ? 1 : 0, NULL);

		if (reference) {
		
			split = nn_from_nlop_F(noir_extract_img_create(model));
			split = nn_set_input_name_F(split, 0, "ref_x");
			network = nn_chain2_FF(split, 0, NULL, network, 0, NULL);
		}
	}

	return network;
}



static nn_t nlinvnet_gn_reg(const struct nlinvnet_s* nlinvnet, struct noir2_net_s* model, enum NETWORK_STATUS status, bool reference)
{

	nn_t result = NULL;
	if (nlinvnet->fix_coils_sense)
		result = nn_from_nlop_F(noir_sense_recon_create(model, nlinvnet->iter_conf_net));
	else
		result = nn_from_nlop_F(noir_gauss_newton_step_create(model, nlinvnet->iter_conf_net));

	if ((0 < nlinvnet->sense_mean) & !(nlinvnet->fix_coils_sense)) {

		auto nlop_avg_coil = noir_rtnlinv_avg_coils_create(model, nlinvnet->sense_mean);
		result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_avg_coil), 0, NULL);
	}
	
	result = nn_set_input_name_F(result, 0, "y");
	result = nn_set_input_name_F(result, 1, "x_0");
	result = nn_set_input_name_F(result, 1, "alp");

	long reg_dims[2];
	md_copy_dims(2, reg_dims, nn_generic_domain(result, 0, "x_0")->dims);


	auto network = nlinvnet_get_network_step(nlinvnet, model, status, reference);

	int N_in_names_gn = nn_get_nr_named_in_args(result);
	int N_in_names_net = nn_get_nr_named_in_args(network);

	const char* in_names[N_in_names_gn + N_in_names_net];
	nn_get_in_names_copy(N_in_names_gn, in_names, result);
	nn_get_in_names_copy(N_in_names_net, in_names + N_in_names_gn, network);

	auto nlop_reg = noir_join_create(model);
		
	auto dom = nlop_generic_domain(nlop_reg, 0);
	nlop_reg = nlop_prepend_FF(nlop_from_linop_F(linop_repmat_create(dom->N, dom->dims, ~0)), nlop_reg, 0);
		
	dom = nlop_generic_domain(nlop_reg, 1);
	nlop_reg = nlop_prepend_FF(nlop_from_linop_F(linop_repmat_create(dom->N, dom->dims, ~0)), nlop_reg, 1);

	nlop_reg = nlop_reshape_in_F(nlop_reg, 0, 1, MD_SINGLETON_DIMS(1));
	nlop_reg = nlop_reshape_in_F(nlop_reg, 1, 1, MD_SINGLETON_DIMS(1));

	nlop_reg = nlop_chain2_swap_FF(nlop_zaxpbz_create(1, MD_SINGLETON_DIMS(1), 1, 1), 0, nlop_reg, 0);
	nlop_reg = nlop_dup_F(nlop_reg, 0, 2);

	result = nn_chain2_swap_FF(nn_from_nlop_F(nlop_reg), 0, NULL, result, 0, "alp");
	result = nn_set_input_name_F(result, 0, "alp");
	result = nn_set_input_name_F(result, 0, "lam");
	result = nn_mark_dup_F(result, "lam");

	//make lambda dummy input of network
	nn_t tmp = nn_from_nlop_F(nlop_del_out_create(1, MD_DIMS(1)));
	tmp = nn_set_input_name_F(tmp, 0, "lam");
	tmp = nn_set_in_type_F(tmp, 0, "lam", IN_OPTIMIZE);;
	tmp = nn_set_initializer_F(tmp, 0, "lam", init_const_create(fabsf(nlinvnet->lambda)));
	network = nn_combine_FF(tmp, network);

	result = nn_chain2_FF(network, 0, NULL, result, 0, "x_0");
	result = nn_dup_F(result, 0, NULL, 1, NULL);
	result = nn_stack_dup_by_name_F(result);
	result = nn_sort_inputs_by_list_F(result, N_in_names_gn + N_in_names_net, in_names);

	for (int i = 0; i < N_in_names_gn + N_in_names_net; i++)
		xfree(in_names[i]);

	return result;
}


static nn_t nlinvnet_chain_alpha(nn_t network, float redu, float alpha_min)
{
	auto nlop_scale = nlop_from_linop_F(linop_scale_create(1, MD_SINGLETON_DIMS(1), 1. / redu));
	nlop_scale = nlop_chain_FF(nlop_zsadd_create(1, MD_SINGLETON_DIMS(1), -alpha_min), nlop_scale);
	nlop_scale = nlop_chain_FF(nlop_scale, nlop_zsadd_create(1, MD_SINGLETON_DIMS(1), alpha_min));

	auto scale = nn_from_nlop_F(nlop_scale);
	network = nn_chain2_FF(scale, 0, NULL, network, 0, "alp");
	network = nn_set_input_name_F(network, -1, "alp");

	network = nlinvnet_sort_args_F(network);

	return network;
}


static nn_t nlinvnet_get_iterations_int(const struct nlinvnet_s* nlinvnet, struct noir2_net_s* model, enum NETWORK_STATUS status)
{
	bool reference = nlinvnet->ref_init;

	auto result = nlinvnet_gn_reg(nlinvnet, model, status, reference);

	for (int i = 1; i < nlinvnet->iter_net; i++) {

		result = nlinvnet_chain_alpha(result, nlinvnet->conf->redu, nlinvnet->conf->alpha_min);

		result = nn_mark_dup_if_exists_F(result, "y");
		result = nn_mark_dup_if_exists_F(result, "x_0");
		result = nn_mark_dup_if_exists_F(result, "alp");
		result = nn_mark_dup_if_exists_F(result, "ref_x");

		auto tmp = nlinvnet_gn_reg(nlinvnet, model, status, reference);

		int N_in_names = nn_get_nr_named_in_args(tmp);
		int N_out_names = nn_get_nr_named_out_args(tmp);

		const char* in_names[N_in_names];
		const char* out_names[N_out_names];

		nn_get_in_names_copy(N_in_names, in_names, tmp);
		nn_get_out_names_copy(N_out_names, out_names, tmp);

		// batchnorm weights are always stacked
		for (int i = 0; i < N_in_names; i++) {

			if (nn_is_name_in_in_args(result, in_names[i])) {

				if (nn_get_dup(result, 0, in_names[i]) && nlinvnet->share_weights)
					result = nn_mark_dup_F(result, in_names[i]);
				else
					result = nn_mark_stack_input_F(result, in_names[i]);
			}
		}

		for (int i = 0; i < N_out_names; i++)
			result = nn_mark_stack_output_if_exists_F(result, out_names[i]);

		result = nn_chain2_FF(tmp, 0, NULL, result, 0, NULL);
		result = nn_stack_dup_by_name_F(result);

		for (int i = 0; i < N_in_names; i++)
			xfree(in_names[i]);
		for (int i = 0; i < N_out_names; i++)
			xfree(out_names[i]);
	}

	if (nn_is_name_in_in_args(result, "ref_x"))
		result = nn_dup_F(result, 0, NULL, 0, "ref_x");

	for (int i = nlinvnet->iter_net; i < (int)(nlinvnet->conf->iter); i++)
		result = nlinvnet_chain_alpha(result, nlinvnet->conf->redu, nlinvnet->conf->alpha_min);

	if (0 != nlinvnet->sense_mean) {

		auto nlop_avg_coil = noir_rtnlinv_avg_coils_create(model, labs(nlinvnet->sense_mean));
		result = nn_chain2_FF(nn_from_nlop_F(nlop_avg_coil), 0, NULL, result, 0, NULL);
	}

	// initialization reco
	const struct nlop_s* nlop_init_reco;
	if (nlinvnet->real_time_init)
		nlop_init_reco = noir_rtnlinv_iter_create(model, nlinvnet->iter_conf, nlinvnet->conf->iter - nlinvnet->iter_net, nlinvnet->conf->redu, nlinvnet->conf->alpha_min, 0.9);
	else
		nlop_init_reco = noir_gauss_newton_iter_create_create(model, nlinvnet->iter_conf, nlinvnet->conf->iter - nlinvnet->iter_net, nlinvnet->conf->redu, nlinvnet->conf->alpha_min);
	
	nlop_init_reco = nlop_set_input_scalar_F(nlop_init_reco, 2, 0);

	auto dom_alp = nlop_generic_domain(nlop_init_reco, 2);
	nlop_init_reco = nlop_prepend_FF(nlop_from_linop_F(linop_repmat_create(dom_alp->N, dom_alp->dims, ~0)), nlop_init_reco, 2);
	nlop_init_reco = nlop_reshape_in_F(nlop_init_reco, 2, 1, MD_DIMS(1));
	

	auto nn_init_reco = nn_from_nlop_F(nlop_init_reco);
	nn_init_reco = nn_set_input_name_F(nn_init_reco, 0, "y");
	nn_init_reco = nn_set_input_name_F(nn_init_reco, 1, "alp");
	nn_init_reco = nn_mark_dup_F(nn_init_reco, "alp");
	nn_init_reco = nn_mark_dup_F(nn_init_reco, "y");

	result = nn_chain2_FF(nn_init_reco, 0, NULL, result, 0, NULL);
	result = nn_stack_dup_by_name_F(result);
	result = nlinvnet_sort_args_F(result);

	complex float alpha = nlinvnet->conf->alpha;
	result = nn_set_input_const_F2(result, 0, "alp", 1, MD_SINGLETON_DIMS(1), MD_SINGLETON_STRS(1), true, &alpha);	// in: y, xn, x0


	//init image with one and coils with zero
	complex float one = 1;
	complex float zero = 0;

	auto nlop_init = noir_join_create(model);
	auto dom = nlop_generic_domain(nlop_init, 0);
	nlop_init = nlop_set_input_const_F2(nlop_init, 0, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &one);
	dom = nlop_generic_domain(nlop_init, 0);
	nlop_init = nlop_set_input_const_F2(nlop_init, 0, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &zero);
	
	auto d1 = nlop_generic_codomain(nlop_init, 0);
	auto d2 = nn_generic_domain(result, 0, NULL);

	nlop_init = nlop_chain2_FF(nlop_init, 0, nlop_from_linop_F(linop_expand_create(d1->N, d2->dims, d1->dims)), 0);
	result = nn_chain2_FF(nn_from_nlop_F(nlop_init), 0, NULL, result, 0, NULL);


	// normalization of input
	float scale = -nlinvnet->scaling;
	assert(0 < scale);

	int N = noir2_net_get_N(model);
	long cim_dims[N];
	long sdims[N];

	noir2_net_get_cim_dims(model, N, cim_dims);	
	md_select_dims(N, BATCH_FLAG, sdims, cim_dims);

	auto nlop_scale = nlop_norm_znorm_create(N, cim_dims, BATCH_FLAG);
	nlop_scale = nlop_chain2_FF(nlop_scale, 1, nlop_from_linop_F(linop_scale_create(N, sdims, 1. / scale)), 0);
	nlop_scale = nlop_chain2_FF(nlop_scale, 1, nlop_from_linop_F(linop_scale_create(N, cim_dims, scale)), 0);

	auto nn_scale = nn_from_nlop_F(nlop_scale);
	nn_scale = nn_set_output_name_F(nn_scale, 1, "scale");
	result = nn_chain2_FF(nn_scale, 0, NULL, result, 0, "y");

	// adjoint (nu)fft
	if (nlinvnet->conf->noncart)
		result = nn_chain2_swap_FF(nn_from_nlop_F(noir_adjoint_nufft_create(model)), 0, NULL , result, 0, NULL);
	else
	 	result = nn_chain2_swap_FF(nn_from_nlop_F(noir_adjoint_fft_create(model)), 0, NULL , result, 0, NULL);

	result = nn_set_input_name_F(result, 0, "ksp");
	result = nn_set_input_name_F(result, 0, "pat");
	if (nlinvnet->conf->noncart)
		result = nn_set_input_name_F(result, 0, "trj");

	return result;
}


static nn_t nlinvnet_get_iterations(const struct nlinvnet_s* nlinvnet, int M, struct noir2_net_s* model[M], enum NETWORK_STATUS status)
{
	nn_t nets[M];

	for (int i = 0; i < M; i++)
		nets[i] = nlinvnet_get_iterations_int(nlinvnet, model[i], status);

	if (1 == M)
		return nets[0];

	return nn_stack_multigpu_F(M, nets, -1);
}


static nn_t nlinvnet_create(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status, enum nlinvnet_out out_type)
{

	int M = 1;

	if (network_is_diagonal(nlinvnet->network))
		M = Nb;

	struct noir2_net_s* models[M];

	int rem = Nb;

	for (int i = 0; i < M; i++) {

		models[i] = noir2_net_create(nlinvnet->model, rem / (M - i));
		rem -= rem / (M - i);
	}

	auto result = nlinvnet_get_iterations(nlinvnet, M, models, status);

	int N = noir2_net_get_N(models[0]);
	long img_dims[N];
	long cim_dims[N];

	noir2_net_get_img_dims(models[0], N, img_dims);
	noir2_net_get_cim_dims(models[0], N, cim_dims);

	img_dims[BATCH_DIM] = Nb;
	cim_dims[BATCH_DIM] = Nb;


	if (0 < nlinvnet->l2loss_reco) {

		complex float mask[img_dims[TIME_DIM]];
		for (int i = 0; i < img_dims[TIME_DIM]; i++)
			mask[i] = ((i >= nlinvnet->ksp_mask_time[0]) && (i < img_dims[TIME_DIM] - nlinvnet->ksp_mask_time[1])) ? 1. : 0.;		
	
		nn_t nn_l2[M];
		for (int i = 0; i < M; i++)
			nn_l2[i] = nn_from_nlop_F(noir_l2_loss_create(models[i], 0.9, mask));
		
		if (1 < M)
			nn_l2[0] = nn_stack_multigpu_F(M, nn_l2, -1); 

		auto cod = nn_generic_codomain(nn_l2[0], 0, 0);
		float scale = nlinvnet->conf->alpha_min + (nlinvnet->conf->alpha - nlinvnet->conf->alpha_min) * pow(nlinvnet->conf->redu, -1.*(nlinvnet->conf->iter + 1));

		auto nn_avg = nn_from_linop_F(linop_chain_FF(linop_avg_create(cod->N, cod->dims, ~0), linop_scale_create(cod->N, MD_SINGLETON_DIMS(cod->N), scale)));
		nn_avg = nn_reshape_out_F(nn_avg, 0, NULL, 1, MD_SINGLETON_DIMS(1));
		nn_avg = nn_set_output_name_F(nn_avg, 0, "l2_reg_reco");
		nn_avg = nn_set_out_type_F(nn_avg, 0, "l2_reg_reco", OUT_OPTIMIZE);
		nn_l2[0] = nn_chain2_FF(nn_l2[0], 0, NULL, nn_avg, 0, NULL);
	
		result = nn_chain2_keep_FF(result, 0, NULL, nn_l2[0], 0, NULL);
	}

	switch (out_type) {

		case NLINVNET_OUT_KSP:
		case NLINVNET_OUT_CIM: {

			nn_t nn_cims[M];
			for (int i = 0; i < M; i++)
				nn_cims[i] = nn_from_nlop_F(noir_cim_create(models[i]));

			auto nn_cim = (1 == M) ? nn_cims[0] : nn_stack_multigpu_F(M, nn_cims, -1);


			nn_cim = nn_set_output_name_F(nn_cim, 0, ((0. < nlinvnet->l2loss_reco) ? "cim" : "cim_us"));
			result = nn_chain2_swap_FF(result, 0, NULL, nn_cim, 0, NULL);

			if (0. == nlinvnet->l2loss_reco) {
	
				auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, cim_dims, cim_dims, nn_generic_codomain(result, 0, "scale")->dims));
				nn_scale = nn_set_output_name_F(nn_scale, 0, "cim");
				nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

				result = nn_chain2_FF(result, 0, "cim_us", nn_scale, 0, NULL);
				result = nn_link_F(result, 0, "scale", 0, "scale");
			}

			if (0 < nlinvnet->tvloss) {

				complex float mask[img_dims[TIME_DIM]];
				for (int i = 0; i < img_dims[TIME_DIM]; i++)
					mask[i] = ((i >= nlinvnet->ksp_mask_time[0]) && (i < img_dims[TIME_DIM] - nlinvnet->ksp_mask_time[1])) ? nlinvnet->tvloss : 0.;		
	
				const struct nlop_s* nlop_tv = nlop_from_linop_F(linop_grad_create(DIMS, cim_dims, DIMS, nlinvnet->tvflags));
				auto cod = nlop_codomain(nlop_tv);
				nlop_tv = nlop_chain_FF(nlop_tv, nlop_zabs_create(cod->N, cod->dims));
				cod = nlop_codomain(nlop_tv);
				nlop_tv = nlop_chain_FF(nlop_tv, nlop_from_linop_F(linop_cdiag_create(cod->N, cod->dims, TIME_FLAG, mask)));
				cod = nlop_codomain(nlop_tv);
				nlop_tv = nlop_chain_FF(nlop_tv, nlop_from_linop_F(linop_sum_create(cod->N, cod->dims, ~BATCH_FLAG)));
				cod = nlop_codomain(nlop_tv);
				nlop_tv = nlop_chain_FF(nlop_tv, nlop_from_linop_F(linop_avg_create(cod->N, cod->dims, BATCH_FLAG)));
				nlop_tv = nlop_reshape_out_F(nlop_tv, 0, 1, MD_DIMS(1));

				nn_t nn_tv = nn_from_nlop_F(nlop_tv);
				nn_tv = nn_set_output_name_F(nn_tv, 0, "tv_loss");
				nn_tv = nn_set_out_type_F(nn_tv, 0, "tv_loss", OUT_OPTIMIZE);

				result = nn_chain2_keep_FF(result, 0, "cim", nn_tv, 0, NULL);
			}

			if (NLINVNET_OUT_KSP == out_type) {

				nn_t fft_ops[M];
				for (int i = 0; i < M; i++) {

					if (nlinvnet->conf->noncart)
						fft_ops[i] = nn_from_nlop_F(noir_nufft_create(models[i]));
					else
					 	fft_ops[i] = nn_from_nlop_F(noir_fft_create(models[i]));
				}

				auto fft_op = (1 == M) ? fft_ops[0] : nn_stack_multigpu_F(M, fft_ops, -1);
				fft_op = nn_set_output_name_F(fft_op, 0, "ksp");

				if (nlinvnet->conf->noncart) {

					fft_op = nn_set_input_name_F(fft_op, 1, "trj");
					fft_op = nn_mark_dup_F(fft_op, "trj");
				}

				result = nn_chain2_FF(result, 0, "cim", fft_op, 0, NULL);
			}
		}
		break;

		case NLINVNET_OUT_IMG_COL: {

			nn_t nn_imgs[M];
			for (int i = 0; i < M; i++)
				nn_imgs[i] = nn_from_nlop_F(noir_decomp_create(models[i]));

			auto nn_img = (1 == M) ? nn_imgs[0] : nn_stack_multigpu_F(M, nn_imgs, -1);
			nn_img = nn_set_output_name_F(nn_img, 0, "img_us");
			nn_img = nn_set_output_name_F(nn_img, 0, "col");

			result = nn_chain2_swap_FF(result, 0, NULL, nn_img, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, img_dims, img_dims, nn_generic_codomain(result, 0, "scale")->dims));
			nn_scale = nn_set_output_name_F(nn_scale, 0, "img");
			nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "img_us", nn_scale, 0, NULL);
			result = nn_link_F(result, 0, "scale", 0, "scale");
		}

		break;
	}

	result = nn_stack_dup_by_name_F(result);
	result = nlinvnet_sort_args_F(result);

	return result;
}


static nn_t nlinvnet_train_loss_create(const struct nlinvnet_s* nlinvnet, int Nb)
{
	nn_t train_op = nlinvnet_create(nlinvnet, Nb, STAT_TRAIN, nlinvnet->ksp_training ? NLINVNET_OUT_KSP : NLINVNET_OUT_CIM);
	const char* out_name = nlinvnet->ksp_training ? "ksp" : "cim";

	int N = nn_generic_codomain(train_op, 0, out_name)->N;

	long out_dims[N];
	long pat_dims[N];

	md_copy_dims(N, out_dims, nn_generic_codomain(train_op, 0, out_name)->dims);
	md_copy_dims(N, pat_dims, nn_generic_domain(train_op, 0, "pat")->dims);

	nn_t loss = train_loss_create(nlinvnet->train_loss, N, out_dims);
	const char* loss_name = strdup(nn_get_out_name_from_arg_index(loss, 0, NULL));

	if (nlinvnet->ksp_training) {

		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 1, NULL);
		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 0, NULL);
		loss = nn_dup_F(loss, 1, NULL, 3, NULL);
		loss = nn_set_input_name_F(loss, 1, "pat_ref");

		if ((0. < nlinvnet->l2loss_reco) || (0. < nlinvnet->l2loss_data)) {

			float l = MAX(nlinvnet->l2loss_reco, nlinvnet->l2loss_data);
			auto nl_loss = nlop_zss_create(N, out_dims, ~BATCH_FLAG);
			auto nn_nl_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_zaxpbz_create(N, out_dims, l, -l), 0, nl_loss, 0));

			nn_nl_loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, nn_nl_loss, 1, NULL);
			nn_nl_loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, nn_nl_loss, 0, NULL);
			nn_nl_loss = nn_dup_F(nn_nl_loss, 1, NULL, 3, NULL);
			nn_nl_loss = nn_set_input_name_F(nn_nl_loss, 1, "pat");
			nn_nl_loss = nn_mark_dup_F(nn_nl_loss, "pat");
			nn_nl_loss = nn_set_output_name_F(nn_nl_loss, 0, "l2_data");
			nn_nl_loss = nn_set_out_type_F(nn_nl_loss, 0, "l2_data", OUT_OPTIMIZE);

			loss = nn_combine_FF(loss, nn_nl_loss);
			loss = nn_dup_F(loss, 0, NULL, 2, NULL);
			loss = nn_dup_F(loss, 1, NULL, 2, NULL);
		}

		if ((0 != nlinvnet->ksp_mask_time[0]) || (0 != nlinvnet->ksp_mask_time[1])) {

			complex float mask[out_dims[TIME_DIM]];
			for (int i = 0; i < out_dims[TIME_DIM]; i++)
				mask[i] = ((i >= nlinvnet->ksp_mask_time[0]) && (i < out_dims[TIME_DIM] - nlinvnet->ksp_mask_time[1])) ? 1. : 0.;
		
			loss = nn_chain2_FF(nn_from_linop_F(linop_cdiag_create(N, out_dims, TIME_FLAG, mask)), 0, NULL, loss, 0, NULL);
			loss = nn_chain2_FF(nn_from_linop_F(linop_cdiag_create(N, out_dims, TIME_FLAG, mask)), 0, NULL, loss, 0, NULL);
		}

		if (0. < nlinvnet->l2loss_reco) {

			auto sdom = nn_generic_codomain(train_op, 0, "scale");
			auto scl = nlop_tenmul_create(N, out_dims, out_dims, sdom->dims);
			scl = nlop_prepend_FF(nlop_zinv_create(N, sdom->dims), scl, 1);
			loss = nn_chain2_swap_FF(nn_from_nlop_F(scl), 0, NULL, loss, 0, NULL);
			loss = nn_set_input_name_F(loss, 1, "scale");

			train_op = nn_chain2_FF(train_op, 0, "ksp", loss, 1, NULL);
			train_op = nn_link_F(train_op, 0, "scale", 0, "scale");
		} else 
			train_op = nn_chain2_FF(train_op, 0, "ksp", loss, 1, NULL);
		
		

		if (-1. != nlinvnet->ksp_split) {

			const struct nlop_s* nlop_rand_split = NULL;

			long T = pat_dims[PHS1_DIM];
			complex float use_reco[T];
			for (int i = 0; i < T; i++)
				use_reco[i] = labs(i - T / 2) < (T / 2) * nlinvnet->exclude_center ? 1 : 0; 

			if (nlinvnet->fixed_splitting)
				nlop_rand_split = nlop_rand_split_fixed_create(N, pat_dims, nlinvnet->ksp_shared_dims, BATCH_FLAG | TIME_FLAG, nlinvnet->ksp_split, PHS1_FLAG, use_reco);
			else
				nlop_rand_split = nlop_rand_split_create(N, pat_dims, nlinvnet->ksp_shared_dims, nlinvnet->ksp_split);
			
			auto split_op = nn_from_nlop_F(nlop_rand_split);
			split_op = nn_set_output_name_F(split_op, 0, "pat_trn");
			split_op = nn_set_output_name_F(split_op, 0, "pat_ref");

			train_op = nn_chain2_swap_FF(split_op, 0, "pat_trn", train_op, 0, "pat");
			train_op = nn_link_F(train_op, 0, "pat_ref", 0, "pat_ref");
			train_op = nn_shift_input_F(train_op, 1, NULL, 0, NULL);
			train_op = nn_set_input_name_F(train_op, 1, "pat");
		} else {

			train_op = nn_dup_F(train_op, 0, "pat", 0, "pat_ref");
		}

	} else {

		train_op = nn_chain2_FF(train_op, 0, "cim", loss, 0, NULL);
	}

	train_op = nn_set_input_name_F(train_op, 0, "ref");

	train_op = nlinvnet_sort_args_F(train_op);


	if (nn_is_name_in_out_args(train_op, "l2_data")) {

		auto cod = nn_generic_codomain(train_op, 0, "l2_data");

		auto nn_avg = nn_from_linop_F(linop_avg_create(cod->N, cod->dims, ~0));
		nn_avg = nn_reshape_out_F(nn_avg, 0, NULL, 1, MD_SINGLETON_DIMS(1));
		nn_avg = nn_set_output_name_F(nn_avg, 0, "l2_data");
		nn_avg = nn_set_out_type_F(nn_avg, 0, "l2_data", OUT_OPTIMIZE);

		train_op = nn_chain2_FF(train_op, 0, "l2_data", nn_avg, 0, NULL);

		train_op = nn_chain2_FF(train_op, 0, loss_name, nn_from_nlop_F(nlop_zaxpbz_create(1, MD_DIMS(1), 1, 1)), 0, NULL);
		train_op = nn_link_F(train_op, 0, "l2_data", 0, NULL);
		train_op = nn_set_out_type_F(train_op, 0, NULL, OUT_OPTIMIZE);
		train_op = nn_set_output_name_F(train_op, 0, loss_name);
	}

	if (nn_is_name_in_out_args(train_op, "l2_reg_reco")) {

		train_op = nn_chain2_FF(train_op, 0, loss_name, nn_from_nlop_F(nlop_zaxpbz_create(1, MD_DIMS(1), 1, 1)), 0, NULL);
		train_op = nn_link_F(train_op, 0, "l2_reg_reco", 0, NULL);
		train_op = nn_set_out_type_F(train_op, 0, NULL, OUT_OPTIMIZE);
		train_op = nn_set_output_name_F(train_op, 0, loss_name);
	}

	if (nn_is_name_in_out_args(train_op, "tv_loss")) {

		train_op = nn_chain2_FF(train_op, 0, loss_name, nn_from_nlop_F(nlop_zaxpbz_create(1, MD_DIMS(1), 1, 1)), 0, NULL);
		train_op = nn_link_F(train_op, 0, "tv_loss", 0, NULL);
		train_op = nn_set_out_type_F(train_op, 0, NULL, OUT_OPTIMIZE);
		train_op = nn_set_output_name_F(train_op, 0, loss_name);
	}

	train_op = nn_stack_dup_by_name_F(train_op);

	xfree(loss_name);

	return train_op;
}

static nn_t nlinvnet_valid_create(const struct nlinvnet_s* nlinvnet, struct named_data_list_s* valid_data)
{
	auto ksp_iov = named_data_list_get_iovec(valid_data, "ksp");
	int Nb = ksp_iov->dims[BATCH_DIM];
	iovec_free(ksp_iov);

	auto result = nlinvnet_create(nlinvnet, 1, STAT_TEST, NLINVNET_OUT_CIM);
	result = nn_del_out_bn_F(result);

	if (nn_is_name_in_out_args(result, "l2_reg_reco"))
		result = nn_del_out_F(result, 0, "l2_reg_reco");

	for (int i = 1; i < Nb; i++) {

		auto tmp = nlinvnet_create(nlinvnet, 1, STAT_TEST, NLINVNET_OUT_CIM);
		tmp = nn_del_out_bn_F(tmp);

		int N_in_names = nn_get_nr_named_in_args(tmp);
		const char* in_names[N_in_names];
		nn_get_in_names_copy(N_in_names, in_names, tmp);

		tmp = nn_mark_stack_input_F(tmp, "ksp");
		tmp = nn_mark_stack_input_F(tmp, "pat");
		tmp = nn_mark_stack_input_if_exists_F(tmp, "trj");
		tmp = nn_mark_stack_output_F(tmp, "cim");
		for (int i = 0; i < N_in_names; i++)
			tmp = nn_mark_dup_if_exists_F(tmp, in_names[i]);

		result = nn_combine_FF(result, tmp);
		result = nn_stack_dup_by_name_F(result);
	}

	auto cim_iov = named_data_list_get_iovec(valid_data, "ref");
	auto valid_loss = nn_chain2_FF(result, 0, "cim", val_measure_create(nlinvnet->valid_loss, cim_iov->N, cim_iov->dims), 0, NULL);
	iovec_free(cim_iov);

	valid_loss = nn_set_input_name_F(valid_loss, 0, "ref");
	valid_loss = nlinvnet_sort_args_F(valid_loss);


	return nn_valid_create(valid_loss, valid_data);
}


static nn_t nlinvnet_apply_op_create(const struct nlinvnet_s* nlinvnet, enum nlinvnet_out out_type, int Nb)
{
	auto nn_apply = nlinvnet_create(nlinvnet, Nb, STAT_TEST, out_type);

	if (!nn_is_name_in_in_args(nn_apply, "trj")) {

		nn_apply = nn_combine_FF(nn_from_nlop_F(nlop_del_out_create(DIMS, MD_SINGLETON_DIMS(DIMS))), nn_apply);
		nn_apply = nn_set_input_name_F(nn_apply, 0, "trj");
	}

	nn_apply = nlinvnet_sort_args_F(nn_apply);

	return nn_get_wo_weights_F(nn_apply, nlinvnet->weights, false);
}



void train_nlinvnet(struct nlinvnet_s* nlinvnet, int Nb, struct named_data_list_s* train_data, struct named_data_list_s* valid_data)
{
	auto ref_iov = named_data_list_get_iovec(train_data, "ref");
	long Nt = ref_iov->dims[BATCH_DIM];
	iovec_free(ref_iov);

	Nb = MIN(Nb, Nt);

	auto nn_train = nlinvnet_train_loss_create(nlinvnet, Nb);

	debug_printf(DP_INFO, "Train nlinvnet\n");
	nn_debug(DP_INFO, nn_train);

	if (NULL == nlinvnet->weights) {

		nlinvnet->weights = nn_weights_create_from_nn(nn_train);
		nn_init(nn_train, nlinvnet->weights);
	} else {

		auto tmp_weights = nn_weights_create_from_nn(nn_train);
		nn_weights_copy(tmp_weights, nlinvnet->weights);
		nn_weights_free(nlinvnet->weights);
		nlinvnet->weights = tmp_weights;
	}

	if (nlinvnet->gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	if (0 <= nlinvnet->lambda)
		nn_train = nn_set_in_type_F(nn_train, 0, "lam", IN_STATIC);

	//create batch generator
	struct bat_gen_conf_s batgen_config = bat_gen_conf_default;
	batgen_config.seed = nlinvnet->train_conf->batch_seed;
	batgen_config.type = nlinvnet->train_conf->batchgen_type;
	
	auto batch_generator = nn_batchgen_create(&batgen_config, nn_train, train_data);

	//setup for iter algorithm
	int NI = nn_get_nr_in_args(nn_train);
	int NO = nn_get_nr_out_args(nn_train);

	float* src[NI];

	enum IN_TYPE in_type[NI];
	nn_get_in_types(nn_train, NI, in_type);

	const struct operator_p_s* projections[NI];
	nn_get_prox_ops(nn_train, NI, projections);

	enum OUT_TYPE out_type[NO];
	nn_get_out_types(nn_train, NO, out_type);

	int weight_index = 0;

	for (int i = 0; i < NI; i++) {

		switch (in_type[i]) {

			case IN_BATCH_GENERATOR:

				src[i] = NULL;
				break;

			case IN_BATCH:
			case IN_UNDEFINED:
				assert(0);
				break;

			case IN_OPTIMIZE:
			case IN_STATIC:
			case IN_BATCHNORM:
			{
				auto iov_weight = nlinvnet->weights->iovs[weight_index];
				auto iov_train_op = nlop_generic_domain(nn_get_nlop(nn_train), i);
				assert(md_check_equal_dims(iov_weight->N, iov_weight->dims, iov_train_op->dims, ~0));
				src[i] = (float*)nlinvnet->weights->tensors[weight_index];
				weight_index++;
			}
		}
	}
	int num_monitors = 0;
	const struct monitor_value_s* value_monitors[3];

	if (NULL != valid_data) {

		auto nn_validation_loss = nlinvnet_valid_create(nlinvnet, valid_data);
		const char* val_names[nn_get_nr_out_args(nn_validation_loss)];
		for (int i = 0; i < nn_get_nr_out_args(nn_validation_loss); i++)
			val_names[i] = nn_get_out_name_from_arg_index(nn_validation_loss, i, false);
		value_monitors[num_monitors] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, nn_get_nr_out_args(nn_validation_loss), val_names);
		nn_free(nn_validation_loss);
		num_monitors += 1;
	}

	if (nn_is_name_in_in_args(nn_train, "lam")) {

		int index_lambda = nn_get_in_arg_index(nn_train, 0, "lam");
		int num_lambda = nn_generic_domain(nn_train, 0, "lam")->dims[0];

		const char* lam = "l";
		const char* lams[num_lambda];

		for (int i = 0; i < num_lambda; i++)
			lams[i] = lam;

		auto destack_lambda = nlop_from_linop_F(linop_identity_create(2, MD_DIMS(1, num_lambda)));
		for (int i = num_lambda - 1; 0 < i; i--)
			destack_lambda = nlop_chain2_FF(destack_lambda, 0, nlop_destack_create(2, MD_DIMS(1, i), MD_DIMS(1, 1), MD_DIMS(1, i + 1), 1), 0);

		for(int i = 0; i < index_lambda; i++)
			destack_lambda = nlop_combine_FF(nlop_del_out_create(1, MD_DIMS(1)), destack_lambda);
		for(int i = index_lambda + 1; i < NI; i++)
			destack_lambda = nlop_combine_FF(destack_lambda, nlop_del_out_create(1, MD_DIMS(1)));

		value_monitors[num_monitors] = monitor_iter6_nlop_create(destack_lambda, true, num_lambda, lams);
		nlop_free(destack_lambda);
		num_monitors += 1;
	}

	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, num_monitors, value_monitors);

	iter6_by_conf(nlinvnet->train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, NO, out_type, Nb, Nt / Nb, batch_generator, monitor);

	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}


#if 0
struct nlinvnet_apply_s {

	INTERFACE(nlop_data_t);

	const struct nlop_s* op;

	int* DO;
	int* DI;

	const long** odims;
	const long** idims;

	int iidx;
	int oidx;
};

DEF_TYPEID(nlinvnet_apply_s);

static void nlinvnet_apply_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto d = CAST_DOWN(nlinvnet_apply_s, _data);

	assert(N == nlop_get_nr_in_args(d->op) + nlop_get_nr_out_args(d->op) - 2);
	
	auto iov = nlop_generic_domain(d->op, d->iidx);
	assert(iovec_check(nlop_generic_codomain(d->op, d->oidx), iov->N, iov->dims, iov->strs));

	complex float* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, args[0]);
	md_clear(iov->N, iov->dims, tmp, iov->size);

	complex float* _args[N + 2];
	
	for (int i = 0, ip = 0; i < N + 2; i++) {

		if ((i == d->oidx) || (i == d->iidx + nlop_get_nr_out_args(d->op)))
			_args[i] = tmp;
		else
			_args[i] = args[ip++];
	}

	nlop_generic_apply_loop(d->op, TIME_FLAG,
				nlop_get_nr_out_args(d->op), d->DO, d->odims, _args,
				nlop_get_nr_in_args(d->op), d->DI, d->idims, (const complex float**)_args + nlop_get_nr_out_args(d->op));

	md_free(tmp);
}

static void nlinvnet_apply_free(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(nlinvnet_apply_s, _data);

	for (int i = 0; i < nlop_get_nr_out_args(d->op); i++)
		xfree(d->odims[i]);
	
	for (int i = 0; i < nlop_get_nr_in_args(d->op); i++)
		xfree(d->idims[i]);

	xfree(d->odims);
	xfree(d->idims);

	xfree(d->DO);
	xfree(d->DI);

	nlop_free(d->op);

	xfree(_data);
}

static const struct nlop_s* nlinvnet_apply_op_causal_create_F(const struct nlop_s* nlop, int N, const long img_dims[N], const long col_dims[N], const long ksp_dims[N], const long pat_dims[N], const long trj_dims[N])
{
	auto iov = nlop_generic_codomain(nlop, 2);

	long tmp_dims[N + 1];
	md_copy_dims(N + 1, tmp_dims, iov->dims);
	tmp_dims[N] *= tmp_dims[TIME_DIM];
	tmp_dims[TIME_DIM] = 1;

	nlop = nlop_reshape2_in_F(nlop, 3, N + 1, TIME_FLAG | MD_BIT(N), tmp_dims);
	nlop = nlop_reshape2_out_F(nlop, 2, N + 1, TIME_FLAG | MD_BIT(N), tmp_dims);

	iov = nlop_generic_codomain(nlop, 2);

	long* odims[3] = { ARR_CLONE(long[N], img_dims),  ARR_CLONE(long[N], col_dims), ARR_CLONE(long[N + 1], iov->dims) };
	long* idims[4] = { ARR_CLONE(long[N], ksp_dims),  ARR_CLONE(long[N], pat_dims),  ARR_CLONE(long[N], trj_dims), ARR_CLONE(long[N + 1], iov->dims) };

	int DO[] = { N, N, N + 1};
	int DI[] = { N, N, N, N + 1};

	PTR_ALLOC(struct nlinvnet_apply_s, data);
	SET_TYPEID(nlinvnet_apply_s, data);

	data->op = nlop;

	data->DO = ARR_CLONE(int[3], DO);
	data->DI = ARR_CLONE(int[4], DI);

	data->odims = (const long**)ARR_CLONE(long* [3], odims);
	data->idims = (const long**)ARR_CLONE(long* [4], idims);

	data->iidx = 3;
	data->oidx = 2;

	long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], img_dims);
	md_copy_dims(N, nl_odims[1], col_dims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], ksp_dims);
	md_copy_dims(N, nl_idims[1], pat_dims);
	md_copy_dims(N, nl_idims[2], trj_dims);

	return nlop_generic_create(2, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)), nlinvnet_apply_fun, NULL, NULL, NULL, NULL, nlinvnet_apply_free);
}
#endif


void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long img_dims[N], complex float* img,
	const long col_dims[N], complex float* col,
	const long ksp_dims[N], const complex float* ksp,
	const long pat_dims[N], const complex float* pat,
	const long trj_dims[N], const complex float* trj)
{
	if (nlinvnet->gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, NLINVNET_OUT_IMG_COL, 1);

	assert(DIMS == N);

	int DO[2] = { N, N };
	int DI[3] = { N, N, N };

	const long* odims[2] = { img_dims, col_dims };
	const long* idims[3] = { ksp_dims, pat_dims, trj_dims };

	complex float* dst[2] = { img, col };
	const complex float* src[5] = { ksp, pat, trj };

	const struct nlop_s* nlop_apply = nlop_clone(nn_apply->nlop); 

	nn_debug(DP_INFO, nn_apply);
	unsigned long batch_flags = md_nontriv_dims(N, img_dims) & (~(md_nontriv_dims(N, nn_generic_codomain(nn_apply, 0, "img")->dims)));

	nn_free(nn_apply);

//	if ((0 < nlinvnet->conv_time) && (PAD_CAUSAL == nlinvnet->conv_padding)) {
//
//		nlop_apply = nlinvnet_apply_op_causal_create_F(nlop_apply, N, img_dims, col_dims, ksp_dims, pat_dims, trj_dims);
//		batch_flags &= ~TIME_FLAG;
//	}

	nlop_unset_derivatives(nlop_apply);
	nlop_generic_apply_loop_sameplace(nlop_apply, batch_flags, 2, DO, odims, dst, 3, DI, idims, src, nlinvnet->weights->tensors[0]);

	nlop_free(nlop_apply);


	if (nlinvnet->normalize_rss) {

		long col_dims2[N];
		md_select_dims(N, ~COIL_FLAG, col_dims2, col_dims);
		complex float* tmp = md_alloc_sameplace(N, col_dims2, CFL_SIZE, img);
		md_zrss(N, col_dims, COIL_FLAG, tmp, col);

		md_zmul2(N, img_dims, MD_STRIDES(N, img_dims, CFL_SIZE), img, MD_STRIDES(N, img_dims, CFL_SIZE), img, MD_STRIDES(N, col_dims2, CFL_SIZE), tmp);
		md_free(tmp);
	}
}

static void apply_nlinvnet_cim(struct nlinvnet_s* nlinvnet, int N,
				const long cim_dims[N], complex float* cim,
				const long ksp_dims[N], const complex float* ksp,
				const long pat_dims[N], const complex float* pat,
				const long trj_dims[N], const complex float* trj)
{
	long img_dims[N];
	long col_dims[N];

	//md_copy_dims(N, img_dims, nlinvnet->models[0]->img_dims);
	//md_copy_dims(N, col_dims, nlinvnet->models[0]->col_dims);

	img_dims[BATCH_DIM] = cim_dims[BATCH_DIM];
	col_dims[BATCH_DIM] = cim_dims[BATCH_DIM];

	complex float* img = md_alloc(N, img_dims, CFL_SIZE);
	complex float* col = md_alloc(N, col_dims, CFL_SIZE);

	apply_nlinvnet(nlinvnet, N,
			img_dims, img,
			col_dims, col,
			ksp_dims, ksp,
			pat_dims, pat,
			trj_dims, trj);

	md_ztenmul(N, cim_dims, cim, img_dims, img, col_dims, col);

	md_free(img);
	md_free(col);
}

void eval_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
		const long cim_dims[N], const complex float* ref,
		const long ksp_dims[N], const complex float* ksp,
		const long pat_dims[N], const complex float* pat,
		const long trj_dims[N], const complex float* trj)
{
	complex float* tmp_out = md_alloc(N, cim_dims, CFL_SIZE);

	auto loss = val_measure_create(nlinvnet->valid_loss, N, cim_dims);
	int NL = nn_get_nr_out_args(loss);
	complex float losses[NL];
	md_clear(1, MD_DIMS(NL), losses, CFL_SIZE);

	apply_nlinvnet_cim(nlinvnet, N,
		cim_dims, tmp_out,
		ksp_dims, ksp,
		pat_dims, pat,
		trj_dims, trj);

	complex float* args[NL + 2];
	for (int i = 0; i < NL; i++)
		args[i] = losses + i;

	args[NL] = tmp_out;
	args[NL + 1] = (complex float*)ref;

	nlop_generic_apply_select_derivative_unchecked(nn_get_nlop(loss), NL + 2, (void**)args, 0, 0);
	for (int i = 0; i < NL ; i++)
		debug_printf(DP_INFO, "%s: %e\n", nn_get_out_name_from_arg_index(loss, i, false), crealf(losses[i]));

	nn_free(loss);
	md_free(tmp_out);
}
