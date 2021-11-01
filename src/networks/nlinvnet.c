#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

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
#include "linops/someops.h"
#include "linops/sum.h"


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

	.network = NULL,
	.share_weights = true,

	.weights = NULL,
	.train_conf = NULL,

	.conf = NULL,
	.Nb = 1,
	.models = NULL,
	.model_valid = NULL,
	.iter_conf = NULL,
	.iter_conf_net = NULL,
	.iter_init = 3,
	.iter_net = 3,

	.train_loss = &loss_option,
	.valid_loss = &val_loss_option,

	.normalize_rss = false,
	.cgtol = 0.1,

	.gpu = false,
	.low_mem = true,

	.scaling = 100.,

	.fix_lambda = false,

	.ksp_training = false,
	.ksp_split = -1.,
	.ksp_noise = 0.,
	.ksp_shared_dims = 0.,
	.ksp_ref_net_only = false,

	.l1_norm = 0,
	.l2_norm = 0,

	.ref = false,

	.graph_file = NULL,
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
	model_conf.fft_flags_cart = FFT_FLAGS | ((nlinvnet->conf->sms || nlinvnet->conf->sos) ? SLICE_FLAG : 0);
	model_conf.rvc = nlinvnet->conf->rvc;
	model_conf.sos = nlinvnet->conf->sos;
	model_conf.a = nlinvnet->conf->a;
	model_conf.b = nlinvnet->conf->b;
	model_conf.noncart = nlinvnet->conf->noncart;
	model_conf.nufft_conf = nlinvnet->conf->nufft_conf;

	nlinvnet->model_valid = TYPE_ALLOC(struct noir2_s);
	*(nlinvnet->model_valid) = noir2_cart_create(N, pat_dims, NULL, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &model_conf);

	nlinvnet->models = *TYPE_ALLOC(struct noir2_s*[nlinvnet->Nb]);
	for (int i = 0; i < nlinvnet->Nb; i++) {

		(nlinvnet->models)[i] = TYPE_ALLOC(struct noir2_s);
		*((nlinvnet->models)[i]) = noir2_cart_create(N, pat_dims, NULL, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &model_conf);
	}
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
	model_conf.fft_flags_cart = ((nlinvnet->conf->sms || nlinvnet->conf->sos) ? SLICE_FLAG : 0);
	model_conf.rvc = nlinvnet->conf->rvc;
	model_conf.sos = nlinvnet->conf->sos;
	model_conf.a = nlinvnet->conf->a;
	model_conf.b = nlinvnet->conf->b;
	model_conf.noncart = nlinvnet->conf->noncart;
	model_conf.nufft_conf = nlinvnet->conf->nufft_conf;

	nlinvnet->model_valid = TYPE_ALLOC(struct noir2_s);
	*(nlinvnet->model_valid) = noir2_noncart_create(N, trj_dims, NULL, wgh_dims, NULL, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &model_conf);

	nlinvnet->models = *TYPE_ALLOC(struct noir2_s*[nlinvnet->Nb]);
	for (int i = 0; i < nlinvnet->Nb; i++) {

		(nlinvnet->models)[i] = TYPE_ALLOC(struct noir2_s);
		*((nlinvnet->models)[i]) = noir2_noncart_create(N, trj_dims, NULL, wgh_dims, NULL, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &model_conf);
	}
}

static nn_t nlinvnet_sort_args_F(nn_t net)
{

	const char* data_names[] =
		{
			"ref",
			"ksp",
			"pat",
			"trj",
			"ref_img",
			"ref_col",
			"loss_mask"
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
	const char* out_names[4 + N];

	out_names[0] = "ksp";
	out_names[1] = "cim";
	out_names[2] = "img";
	out_names[3] = "col";

	nn_get_out_names_copy(N, out_names + 4, net);

	net = nn_sort_outputs_by_list_F(net, 4 + N, out_names);

	for (int i = 0; i < N; i++)
		xfree(out_names[4 + i]);

	net = nn_sort_inputs_F(net);
	net = nn_sort_outputs_F(net);

	if (nn_is_name_in_in_args(net, "lam")) {

		auto iov = nn_generic_domain(net, 0, "lam");
		auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
		net = nn_set_prox_op_F(net, 0, "lam", prox_conv);
	}

	return net;
}


static nn_t nlinvnet_network_create(const struct nlinvnet_s* nlinvnet, unsigned int N, const long img_dims[N], enum NETWORK_STATUS status)
{

	unsigned long channel_flag = (~(FFT_FLAGS | BATCH_FLAG)) & (md_nontriv_dims(N, img_dims));

	long chn_dims[N];
	md_select_dims(N, channel_flag, chn_dims, img_dims);
	long channel = md_calc_size(N, chn_dims);

	long dims[5] = {img_dims[0], img_dims[1], img_dims[2], channel, img_dims[BATCH_DIM]};
	long idims_net[5] = {channel + (nlinvnet->ref ? 1 : 0), img_dims[0], img_dims[1], img_dims[2], img_dims[BATCH_DIM]};
	long odims_net[5] = {channel, img_dims[0], img_dims[1], img_dims[2], img_dims[BATCH_DIM]};

	auto network = network_create(nlinvnet->network, 5, odims_net, 5, idims_net, status);

	if (nlinvnet->ref) {

		long rdims_net[5] = {1, img_dims[0], img_dims[1], img_dims[2], img_dims[BATCH_DIM]};
		nn_t nn_destack = nn_from_nlop_F(nlop_stack_create(5, idims_net, odims_net, rdims_net, 0));

		long rdims[N];
		md_select_dims(N, FFT_FLAGS | BATCH_FLAG, rdims, img_dims);
		nn_destack = nn_reshape_in_F(nn_destack, 1, NULL, N, rdims);
		nn_destack = nn_set_input_name_F(nn_destack, 1, "ref_img");
		network = nn_chain2_swap_FF(nn_destack, 0, NULL, network, 0, NULL);
	}

	if (1 != channel) {

		unsigned int iperm[5] = {3, 0, 1, 2, 4};
		unsigned int operm[5] = {1, 2, 3, 0, 4};

		network = nn_chain2_swap_FF(nn_from_nlop_F(nlop_from_linop_F(linop_permute_create(5, iperm, dims))), 0, NULL, network, 0, NULL);
		network = nn_chain2_swap_FF(network, 0, NULL, nn_from_nlop_F(nlop_from_linop_F(linop_permute_create(5, operm, odims_net))), 0, NULL);
	}

	network = nn_reshape_in_F(network, 0, NULL, N, img_dims);
	network = nn_reshape_out_F(network, 0, NULL, N, img_dims);

	return network;
}

static nn_t nlinvnet_get_network_step(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], enum NETWORK_STATUS status)
{
	int N = noir_model_get_N(models[0]);
	assert(N == DIMS);

	long img_dims[N];
	long col_dims[N];

	noir_model_get_img_dims(N, img_dims, models[0]);
	noir_model_get_col_dims(N, col_dims, models[0]);

	img_dims[BATCH_DIM] = Nb;
	col_dims[BATCH_DIM] = Nb;

	auto network = nlinvnet_network_create(nlinvnet, N, img_dims, status);

	int N_in_names = nn_get_nr_named_in_args(network);
	int N_out_names = nn_get_nr_named_out_args(network);
	const char* in_names[N_in_names];
	const char* out_names[N_out_names];
	nn_get_in_names_copy(N_in_names, in_names, network);
	nn_get_out_names_copy(N_out_names, out_names, network);

	for (int i = 0; i < N_in_names; i++) {

		if (0 == strcmp(in_names[i], "ref_img"))
			continue;

		network = nn_append_singleton_dim_in_F(network, 0, in_names[i]);
		xfree(in_names[i]);
	}

	for (int i = 0; i < N_out_names; i++) {

		network = nn_append_singleton_dim_out_F(network, 0, out_names[i]);
		xfree(out_names[i]);
	}

	nn_t join = NULL;

	if (nlinvnet->ref && !(nlinvnet->ksp_ref_net_only)) {

		join = nn_from_nlop_F(noir_join_batch_create(Nb, models));
		join = nn_set_input_name_F(join, 1, "ref_col");
		join = nn_set_in_type_F(join, 0, "ref_col", IN_BATCH_GENERATOR);

	} else {
		join = nn_from_nlop_F(noir_set_img_batch_create(Nb, models));
	}

	nn_t split = nn_from_nlop_F(noir_extract_img_batch_create(Nb, models));

	network = nn_chain2_FF(split, 0, NULL, network, 0, NULL);
	network = nn_chain2_FF(network, 0, NULL, join, 0, NULL);

	return nn_checkpoint_F(network, false, true);
}


static nn_t nlinvnet_get_cell_reg(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], int index, enum NETWORK_STATUS status)
{
	assert(0 <= index);
	bool network = (index >= ((int)nlinvnet->conf->iter - nlinvnet->iter_net));

	float update = index < nlinvnet->iter_init ? 0.5 : 1;

	auto result = nn_from_nlop_F(noir_gauss_newton_step_batch_create(Nb, models, network ? nlinvnet->iter_conf_net : nlinvnet->iter_conf, update));
	result = nn_set_input_name_F(result, 0, "y");
	result = nn_set_input_name_F(result, 1, "x_0");
	result = nn_set_input_name_F(result, 1, "alp");

	long reg_dims[2];
	md_copy_dims(2, reg_dims, nn_generic_domain(result, 0, "x_0")->dims);

	if (network) {

		auto network = nlinvnet_get_network_step(nlinvnet, Nb, models, status);

		int N_in_names_gn = nn_get_nr_named_in_args(result);
		int N_in_names_net = nn_get_nr_named_in_args(network);

		const char* in_names[N_in_names_gn + N_in_names_net];
		nn_get_in_names_copy(N_in_names_gn, in_names, result);
		nn_get_in_names_copy(N_in_names_net, in_names + N_in_names_gn, network);

		int N = noir_model_get_N(models[0]);
		assert(N == DIMS);

		long img_dims[N];
		long col_dims[N];

		noir_model_get_img_dims(N, img_dims, models[0]);
		noir_model_get_col_dims(N, col_dims, models[0]);

		long img_size [2] = { md_calc_size(N, img_dims), Nb };
		long tot_size [2] = { md_calc_size(N, img_dims) + md_calc_size(N, col_dims), Nb };

		auto nlop = nlop_from_linop_F(linop_chain_FF(linop_repmat_create(1, img_size, MD_BIT(0)), linop_expand_create(1, tot_size, img_size)));
		nlop = nlop_chain2_FF(nlop, 0, nlop_zaxpbz_create(1, tot_size, 1, 1), 0);
		result = nn_chain2_swap_FF(nn_from_nlop_F(nlop), 0, NULL, result, 0, "alp");
		result = nn_set_input_name_F(result, 0, "alp");
		result = nn_set_input_name_F(result, 0, "lam");
		result = nn_mark_dup_F(result, "lam");

		//make lambda dummy input of network
		nn_t tmp = nn_from_nlop_F(nlop_del_out_create(1, MD_DIMS(1)));
		tmp = nn_set_input_name_F(tmp, 0, "lam");
		tmp = nn_set_in_type_F(tmp, 0, "lam", IN_OPTIMIZE);;
		tmp = nn_set_initializer_F(tmp, 0, "lam", init_const_create(0.01));
		network = nn_combine_FF(tmp, network);

		result = nn_chain2_FF(network, 0, NULL, result, 0, "x_0");
		result = nn_dup_F(result, 0, NULL, 1, NULL);
		result = nn_stack_dup_by_name_F(result);
		result = nn_sort_inputs_by_list_F(result, N_in_names_gn + N_in_names_net, in_names);

		for (int i = 0; i < N_in_names_gn + N_in_names_net; i++)
			xfree(in_names[i]);
	} else {

		complex float zero = 0;
		if (!nlinvnet->ref)
			result = nn_set_input_const_F2(result, 0, "x_0", 2, reg_dims, MD_SINGLETON_STRS(2), true, &zero);
	}

	return result;
}


static nn_t nlinvnet_chain_alpha(const struct nlinvnet_s* nlinvnet, nn_t network)
{

	auto dom = nn_generic_domain(network, 0, "alp");

	auto nlop_scale = nlop_from_linop_F(linop_scale_create(dom->N, dom->dims, 1. / nlinvnet->conf->redu));
	nlop_scale = nlop_chain_FF(nlop_zsadd_create(dom->N, dom->dims, -nlinvnet->conf->alpha_min), nlop_scale);
	nlop_scale = nlop_chain_FF(nlop_scale, nlop_zsadd_create(dom->N, dom->dims, nlinvnet->conf->alpha_min));

	auto scale = nn_from_nlop_F(nlop_scale);
	network = nn_chain2_FF(scale, 0, NULL, network, 0, "alp");
	network = nn_set_input_name_F(network, -1, "alp");

	network = nlinvnet_sort_args_F(network);

	return network;
}


static nn_t nlinvnet_get_iterations(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], enum NETWORK_STATUS status)
{
	int j = nlinvnet->conf->iter - 1;

	auto result = nlinvnet_get_cell_reg(nlinvnet, Nb, models, j, status);

	while (0 < j--) {

		result = nlinvnet_chain_alpha(nlinvnet, result);

		result = nn_mark_dup_if_exists_F(result, "y");
		result = nn_mark_dup_if_exists_F(result, "x_0");
		result = nn_mark_dup_if_exists_F(result, "alp");

		auto tmp = nlinvnet_get_cell_reg(nlinvnet, Nb, models, j, status);

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

	result = nlinvnet_sort_args_F(result);

	return result;
}

static nn_t nlinvnet_create(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], enum NETWORK_STATUS status, enum nlinvnet_out out_type)
{

	auto result = nlinvnet_get_iterations(nlinvnet, Nb, models, status);

	if (nlinvnet->ref) {

		if (nlinvnet->ksp_ref_net_only) {

			nn_t join = nn_from_nlop_F(noir_set_col_batch_create(Nb, models));
			join = nn_set_input_name_F(join, 0, "ref_col");
			join = nn_set_in_type_F(join, 0, "ref_col", IN_BATCH_GENERATOR);

			result = nn_chain2_FF(join, 0, NULL, result, 0, "x_0");

		} else {

			nn_t join = nn_from_nlop_F(noir_join_batch_create(Nb, models));
			join = nn_set_input_name_F(join, 0, "ref_img");
			join = nn_set_input_name_F(join, 0, "ref_col");

			join = nn_mark_dup_F(join, "ref_img");
			join = nn_mark_dup_F(join, "ref_col");

			result = nn_chain2_FF(join, 0, NULL, result, 0, "x_0");
			result = nn_stack_dup_by_name_F(result);

			result = nn_set_in_type_F(result, 0, "ref_img", IN_BATCH_GENERATOR);
			result = nn_set_in_type_F(result, 0, "ref_col", IN_BATCH_GENERATOR);
		}
	}

	int N = noir_model_get_N(models[0]);
	assert(N == DIMS);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_dims(N, img_dims, models[0]);
	noir_model_get_col_dims(N, col_dims, models[0]);
	noir_model_get_cim_dims(N, cim_dims, models[0]);

	complex float alpha = nlinvnet->conf->alpha;
	long alp_dims[1];
	md_copy_dims(1, alp_dims, nn_generic_domain(result, 0, "alp")->dims);
	result = nn_set_input_const_F2(result, 0, "alp", 1, alp_dims, MD_SINGLETON_STRS(N), true, &alpha);	// in: y, xn, x0


	long ini_dims[2];
	md_copy_dims(2, ini_dims, nn_generic_domain(result, 0, NULL)->dims);

	long size = noir_model_get_size(models[0]);
	long skip = noir_model_get_skip(models[0]);
	assert(ini_dims[0] == size);

	complex float* init = md_alloc(1, MD_DIMS(size), CFL_SIZE);
	md_clear(1, MD_DIMS(size), init, CFL_SIZE);
	md_zfill(1, MD_DIMS(skip), init, 1.);

	result = nn_set_input_const_F2(result, 0, NULL, 2, ini_dims, MD_DIMS(CFL_SIZE, 0), true, init);	// in: y
	md_free(init);

	complex float scale = nlinvnet->scaling;

	long cim_dims2[N];
	long img_dims2[N];
	md_copy_dims(N, cim_dims2, cim_dims);
	md_copy_dims(N, img_dims2, img_dims);
	cim_dims2[BATCH_DIM] = Nb;
	img_dims2[BATCH_DIM] = Nb;


	long sdims[N];
	md_select_dims(N, BATCH_FLAG, sdims, img_dims2);

	auto nlop_scale = nlop_norm_znorm_create(N, cim_dims2, BATCH_FLAG);
	nlop_scale = nlop_chain2_FF(nlop_scale, 1, nlop_from_linop_F(linop_scale_create(N, sdims, 1. / scale)), 0);
	nlop_scale = nlop_chain2_FF(nlop_scale, 1, nlop_from_linop_F(linop_scale_create(N, cim_dims2, scale)), 0);

	auto nn_scale = nn_from_nlop_F(nlop_scale);
	nn_scale = nn_set_output_name_F(nn_scale, 0, "y");
	nn_scale = nn_set_output_name_F(nn_scale, 0, "scale");

	result = nn_chain2_FF(nn_scale, 0, "y", result, 0, "y");

	switch (out_type) {

		case NLINVNET_OUT_KSP:
		case NLINVNET_OUT_CIM: {

			auto nn_cim = nn_from_nlop_F(noir_cim_batch_create(Nb, models));
			nn_cim = nn_set_output_name_F(nn_cim, 0, "cim_us");
			result = nn_chain2_swap_FF(result, 0, NULL, nn_cim, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, cim_dims2, cim_dims2, sdims));
			nn_scale = nn_set_output_name_F(nn_scale, 0, "cim");
			nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "cim_us", nn_scale, 0, NULL);
			result = nn_link_F(result, 0, "scale", 0, "scale");

			if (0 != nlinvnet->l2_norm) {
				auto l2_norm = nn_from_nlop_F(nlop_chain_FF(nlop_znorm_create(N, cim_dims2, ~0), nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), nlinvnet->l2_norm))));
				l2_norm = nn_set_output_name_F(l2_norm, 0, "l2_norm");
				l2_norm = nn_set_out_type_F(l2_norm, 0, "l2_norm", OUT_OPTIMIZE);
				result = nn_chain2_keep_FF(result, 0, "cim", l2_norm, 0, NULL);
			}

			if (0 != nlinvnet->l1_norm) {
				auto l1_norm = nn_from_nlop_F(nlop_chain_FF(nlop_z1norm_create(N, cim_dims2, ~0), nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), nlinvnet->l1_norm))));
				l1_norm = nn_set_output_name_F(l1_norm, 0, "l1_norm");
				l1_norm = nn_set_out_type_F(l1_norm, 0, "l1_norm", OUT_OPTIMIZE);
				result = nn_chain2_keep_FF(result, 0, "cim", l1_norm, 0, NULL);
			}

			if (NLINVNET_OUT_KSP == out_type) {

				nn_t fft_op = NULL;

				if (nlinvnet->conf->noncart)
					fft_op = nn_from_nlop_F(noir_nufft_batch_create(Nb, models));
				else
				 	fft_op = nn_from_nlop_F(noir_fft_batch_create(Nb, models));


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

			auto nlop_img = noir_decomp_batch_create(Nb, models);
			auto nn_img = nn_from_nlop_F(nlop_img);
			nn_img = nn_set_output_name_F(nn_img, 0, "img_us");
			nn_img = nn_set_output_name_F(nn_img, 0, "col");
			result = nn_chain2_swap_FF(result, 0, NULL, nn_img, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, img_dims2, img_dims2, sdims));
			nn_scale = nn_set_output_name_F(nn_scale, 0, "img");
			nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "img_us", nn_scale, 0, NULL);
			result = nn_link_F(result, 0, "scale", 0, "scale");
		}

		break;
	}

	if (nlinvnet->conf->noncart)
		result = nn_chain2_swap_FF(nn_from_nlop_F(noir_adjoint_nufft_batch_create(Nb, models)), 0, NULL , result, 0, NULL);
	else
	 	result = nn_chain2_swap_FF(nn_from_nlop_F(noir_adjoint_fft_batch_create(Nb, models)), 0, NULL , result, 0, NULL);

	result = nn_set_input_name_F(result, 0, "ksp");
	result = nn_set_input_name_F(result, 0, "pat");
	if (nlinvnet->conf->noncart)
		result = nn_set_input_name_F(result, 0, "trj");

	result = nn_stack_dup_by_name_F(result);

	result = nlinvnet_sort_args_F(result);

	return result;
}


static nn_t nlinvnet_train_loss_create(const struct nlinvnet_s* nlinvnet)
{
	int N = noir_model_get_N(nlinvnet->models[0]);
	assert(N == DIMS);

	nn_t train_op = nlinvnet_create(nlinvnet, nlinvnet->Nb, nlinvnet->models, STAT_TRAIN, nlinvnet->ksp_training ? NLINVNET_OUT_KSP : NLINVNET_OUT_CIM);
	const char* out_name = nlinvnet->ksp_training ? "ksp" : "cim";

	long out_dims[N];
	long pat_dims[N];

	md_copy_dims(N, out_dims, nn_generic_codomain(train_op, 0, out_name)->dims);
	md_copy_dims(N, pat_dims, nn_generic_domain(train_op, 0, "pat")->dims);

	nn_t loss = train_loss_create(nlinvnet->train_loss, N, out_dims);

	if (nlinvnet->ksp_training) {

		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 1, NULL);
		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 0, NULL);
		loss = nn_dup_F(loss, 1, NULL, 3, NULL);
		loss = nn_set_input_name_F(loss, 1, "pat_ref");
		train_op = nn_chain2_FF(train_op, 0, "ksp", loss, 1, NULL);

		if (-1. != nlinvnet->ksp_split) {

			auto split_op = nn_from_nlop_F(nlop_rand_split_create(N, pat_dims, nlinvnet->ksp_shared_dims, nlinvnet->ksp_split));
			split_op = nn_set_output_name_F(split_op, 0, "pat_trn");
			split_op = nn_set_output_name_F(split_op, 0, "pat_ref");

			train_op = nn_chain2_swap_FF(split_op, 0, "pat_ref", train_op, 0, "pat_ref");
			train_op = nn_link_F(train_op, 0, "pat_trn", 0, "pat");
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

	return train_op;
}

static nn_t nlinvnet_valid_create(const struct nlinvnet_s* nlinvnet, struct named_data_list_s* valid_data)
{
	auto ksp_iov = named_data_list_get_iovec(valid_data, "ksp");
	int Nb = ksp_iov->dims[BATCH_DIM];
	iovec_free(ksp_iov);

	auto result = nlinvnet_create(nlinvnet, 1, &(((struct nlinvnet_s*)nlinvnet)->model_valid), STAT_TEST, NLINVNET_OUT_CIM);
	result = nn_del_out_bn_F(result);

	if (nn_is_name_in_out_args(result, "l1_norm"))
		result = nn_del_out_F(result, 0, "l1_norm");
	if (nn_is_name_in_out_args(result, "l2_norm"))
		result = nn_del_out_F(result, 0, "l2_norm");

	for (int i = 1; i < Nb; i++) {

		auto tmp = nlinvnet_create(nlinvnet, 1, &(((struct nlinvnet_s*)nlinvnet)->model_valid), STAT_TEST, NLINVNET_OUT_CIM);
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
	auto nn_apply = nlinvnet_create(nlinvnet, Nb, nlinvnet->models, STAT_TEST, out_type);

	if (!nlinvnet->ref){

		nn_t join = nn_from_nlop_F(
				nlop_combine_FF(
					nlop_del_out_create(nlinvnet->models[0]->N, nlinvnet->models[0]->img_dims),
					nlop_del_out_create(nlinvnet->models[0]->N, nlinvnet->models[0]->col_dims)));
		join = nn_set_input_name_F(join, 0, "ref_img");
		join = nn_set_input_name_F(join, 0, "ref_col");

		nn_apply = nn_combine_FF(nn_apply, join);

		nn_apply = nn_set_in_type_F(nn_apply, 0, "ref_img", IN_STATIC);
		nn_apply = nn_set_in_type_F(nn_apply, 0, "ref_col", IN_STATIC);
	}

	if (!nn_is_name_in_in_args(nn_apply, "trj")) {

		nn_apply = nn_combine_FF(nn_from_nlop_F(nlop_del_out_create(nlinvnet->models[0]->N, MD_SINGLETON_DIMS(nlinvnet->models[0]->N))), nn_apply);
		nn_apply = nn_set_input_name_F(nn_apply, 0, "trj");
	}

	if (nn_is_name_in_out_args(nn_apply, "l1_norm"))
		nn_apply = nn_del_out_F(nn_apply, 0, "l1_norm");
	if (nn_is_name_in_out_args(nn_apply, "l2_norm"))
		nn_apply = nn_del_out_F(nn_apply, 0, "l2_norm");

	nn_apply = nlinvnet_sort_args_F(nn_apply);

	return nn_get_wo_weights_F(nn_apply, nlinvnet->weights, false);
}



void train_nlinvnet(struct nlinvnet_s* nlinvnet, int Nb, struct named_data_list_s* train_data, struct named_data_list_s* valid_data)
{
	auto ref_iov = named_data_list_get_iovec(train_data, "ref");
	long Nt = ref_iov->dims[BATCH_DIM];
	iovec_free(ref_iov);

	Nb = MIN(Nb, Nt);

	auto nn_train = nlinvnet_train_loss_create(nlinvnet);

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

	if (nlinvnet->fix_lambda)
		nn_train = nn_set_in_type_F(nn_train, 0, "lam", IN_STATIC);

	//create batch generator
	auto batch_generator = nn_batchgen_create(nn_train, train_data, nlinvnet->train_conf->batchgen_type, nlinvnet->train_conf->batch_seed);

	auto ksp_dom = nlop_generic_codomain(batch_generator, 1);
	if (0 != nlinvnet->ksp_noise)
		batch_generator = nlop_append_FF(batch_generator, 1, nlop_add_noise_create(ksp_dom->N, ksp_dom->dims, nlinvnet->ksp_noise, 0, ~BATCH_FLAG));


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
			case IN_STATIC:
				assert(0);
				break;

			case IN_OPTIMIZE:
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

	if (NULL != train_data) {

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


void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long img_dims[N], complex float* img,
	const long col_dims[N], complex float* col,
	const long ksp_dims[N], const complex float* ksp,
	const long pat_dims[N], const complex float* pat,
	const long trj_dims[N], const complex float* trj,
	const long ref_img_dims[N], const complex float* ref_img,
	const long ref_col_dims[N], const complex float* ref_col)
{
	if (nlinvnet->gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, NLINVNET_OUT_IMG_COL, 1);

	assert(DIMS == N);

	int DO[2] = { N, N };
	int DI[5] = { N, N, N, N, N };

	long ref_img_dims2[N];
	long ref_col_dims2[N];

	md_copy_dims(N, ref_img_dims2, ref_img ? ref_img_dims : nn_generic_domain(nn_apply, 0, "ref_img")->dims);
	md_copy_dims(N, ref_col_dims2, ref_col ? ref_col_dims : nn_generic_domain(nn_apply, 0, "ref_col")->dims);

	const complex float* ref_img2 = ref_img ? NULL : md_calloc(N, ref_img_dims2, CFL_SIZE);
	const complex float* ref_col2 = ref_col ? NULL : md_calloc(N, ref_col_dims2, CFL_SIZE);

	ref_img = ref_img ? ref_img : ref_img2;
	ref_col = ref_col ? ref_col : ref_col2;

	const long* odims[2] = { img_dims, col_dims };
	const long* idims[5] = { ksp_dims, pat_dims, trj_dims, ref_img_dims2, ref_col_dims2 };

	complex float* dst[2] = { img, col };
	const complex float* src[5] = { ksp, pat, trj, ref_img, ref_col };

	nn_debug(DP_INFO, nn_apply);

	nlop_generic_apply_loop_sameplace(nn_get_nlop(nn_apply), BATCH_FLAG, 2, DO, odims, dst, 5, DI, idims, src, nlinvnet->weights->tensors[0]);

	nn_free(nn_apply);

	md_free(ref_img2);
	md_free(ref_col2);

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
				const long trj_dims[N], const complex float* trj,
				const long ref_img_dims[N], const complex float* ref_img,
				const long ref_col_dims[N], const complex float* ref_col)
{
	long img_dims[N];
	long col_dims[N];

	md_copy_dims(N, img_dims, nlinvnet->models[0]->img_dims);
	md_copy_dims(N, col_dims, nlinvnet->models[0]->col_dims);

	img_dims[BATCH_DIM] = cim_dims[BATCH_DIM];
	col_dims[BATCH_DIM] = cim_dims[BATCH_DIM];

	complex float* img = md_alloc(N, img_dims, CFL_SIZE);
	complex float* col = md_alloc(N, col_dims, CFL_SIZE);

	apply_nlinvnet(nlinvnet, N,
			img_dims, img,
			col_dims, col,
			ksp_dims, ksp,
			pat_dims, pat,
			trj_dims, trj,
			ref_img_dims, ref_img,
			ref_col_dims, ref_col);

	md_ztenmul(N, cim_dims, cim, img_dims, img, col_dims, col);

	md_free(img);
	md_free(col);
}

void eval_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
		const long cim_dims[N], const complex float* ref,
		const long ksp_dims[N], const complex float* ksp,
		const long pat_dims[N], const complex float* pat,
		const long trj_dims[N], const complex float* trj,
		const long ref_img_dims[N], const complex float* ref_img,
		const long ref_col_dims[N], const complex float* ref_col)
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
		trj_dims, trj,
		ref_img_dims, ref_img,
		ref_col_dims, ref_col);

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
