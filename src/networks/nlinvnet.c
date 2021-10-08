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
	.model = NULL,
	.iter_conf = NULL,
	.iter_init = 3,
	.iter_net = 3,

	.train_loss = &loss_option,
	.valid_loss = &val_loss_option,

	.rss_loss = false,
	.normalize_rss = false,

	.gpu = false,
	.low_mem = true,

	.extra_lambda = true,
	.fix_lambda = false,

	.fix_coils = false,

	.ksp_noise = 0.,

	.graph_file = NULL,
};


void nlinvnet_init_model_cart(struct nlinvnet_s* nlinvnet, int N,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N])
{
	struct noir2_model_conf_s model_conf = noir2_model_conf_defaults;
	model_conf.fft_flags_noncart = 0;
	model_conf.fft_flags_cart = FFT_FLAGS | ((nlinvnet->conf->sms || nlinvnet->conf->sos) ? SLICE_FLAG : 0);
	model_conf.rvc = nlinvnet->conf->rvc;
	model_conf.sos = nlinvnet->conf->sos;
	model_conf.a = nlinvnet->conf->a;
	model_conf.b = nlinvnet->conf->b;
	model_conf.noncart = nlinvnet->conf->noncart;
	model_conf.nufft_conf = nlinvnet->conf->nufft_conf;

	nlinvnet->model = TYPE_ALLOC(struct noir2_s);
	*(nlinvnet->model) = noir2_cart_create(N, pat_dims, pattern, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &model_conf);

	nlinvnet->iter_conf = TYPE_ALLOC(struct iter_conjgrad_conf);
	*(nlinvnet->iter_conf) = iter_conjgrad_defaults;
	nlinvnet->iter_conf->INTERFACE.alpha = 0.;
	nlinvnet->iter_conf->l2lambda = 0.;
	nlinvnet->iter_conf->maxiter = (0 == nlinvnet->conf->cgiter) ? 30 : nlinvnet->conf->cgiter;
	nlinvnet->iter_conf->tol = 0.;

	if (NULL == get_loss_from_option()) {

		if (nlinvnet->rss_loss)
			nlinvnet->train_loss->weighting_mse_rss=1.;
		else
			nlinvnet->train_loss->weighting_mse=1.;
	}

	if (NULL == get_val_loss_from_option())
		nlinvnet->valid_loss = &loss_image_valid;

	assert(0 == nlinvnet->iter_conf->tol);
}

static nn_t nlinvnet_get_gauss_newton_step(const struct nlinvnet_s* nlinvnet, int Nb, float update, bool fix_coils)
{
	auto result = nn_from_nlop_F(noir_gauss_newton_step_batch_create(nlinvnet->model, nlinvnet->iter_conf, Nb, update, fix_coils));
	result = nn_set_input_name_F(result, 0, "y");
	result = nn_set_input_name_F(result, 1, "x_0");
	result = nn_set_input_name_F(result, 1, "alpha");

	return result;
}

static nn_t nlinvnet_get_network_step(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status, bool coils)
{
	int N = noir_model_get_N(nlinvnet->model);
	assert(N == DIMS);

	long img_dims[N];
	long col_dims[N];

	noir_model_get_img_dims(N, img_dims, nlinvnet->model);
	noir_model_get_col_dims(N, col_dims, nlinvnet->model);

	img_dims[BATCH_DIM] = Nb;
	col_dims[BATCH_DIM] = Nb;

	long img_dims_net[5] = { 1, img_dims[0], img_dims[1], img_dims[2], img_dims[BATCH_DIM] };

	auto network = network_create(nlinvnet->network, 5, img_dims_net, 5, img_dims_net, status);
	network = nn_reshape_in_F(network, 0, NULL, N, img_dims);
	network = nn_reshape_out_F(network, 0, NULL, N, img_dims);


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

	if (coils) {

		auto join = nn_from_nlop_F(noir_join_batch_create(nlinvnet->model, Nb));
		auto split = nn_from_nlop_F(noir_split_batch_create(nlinvnet->model, Nb));

		network = nn_chain2_FF(split, 0, NULL, network, 0, NULL);
		network = nn_chain2_FF(network, 0, NULL, join, 0, NULL);
		network = nn_link_F(network, -1, NULL, 0, NULL);
	} else {

		auto join = nn_from_nlop_F(noir_set_img_batch_create(nlinvnet->model, Nb));
		auto split = nn_from_nlop_F(noir_extract_img_batch_create(nlinvnet->model, Nb));

		network = nn_chain2_FF(split, 0, NULL, network, 0, NULL);
		network = nn_chain2_FF(network, 0, NULL, join, 0, NULL);
	}

	return nn_checkpoint_F(network, false, true);
}


static nn_t nlinvnet_get_cell_reg(const struct nlinvnet_s* nlinvnet, int Nb, int index, enum NETWORK_STATUS status, bool fix_coil)
{
	assert(0 <= index);
	bool network = (index >= ((int)nlinvnet->conf->iter - nlinvnet->iter_net));

	float update = index < nlinvnet->iter_init ? 0.5 : 1;

	auto result = nlinvnet_get_gauss_newton_step(nlinvnet, Nb, update, fix_coil);

	long reg_dims[2];
	md_copy_dims(2, reg_dims, nn_generic_domain(result, 0, "x_0")->dims);

	if (network) {

		auto network = nlinvnet_get_network_step(nlinvnet, Nb, status, false);

		int N_in_names_gn = nn_get_nr_named_in_args(result);
		int N_in_names_net = nn_get_nr_named_in_args(network);

		const char* in_names[N_in_names_gn + N_in_names_net];
		nn_get_in_names_copy(N_in_names_gn, in_names, result);
		nn_get_in_names_copy(N_in_names_net, in_names + N_in_names_gn, network);

		if (nlinvnet->extra_lambda) {

			int N = noir_model_get_N(nlinvnet->model);
			assert(N == DIMS);

			long img_dims[N];
			long col_dims[N];

			noir_model_get_img_dims(N, img_dims, nlinvnet->model);
			noir_model_get_col_dims(N, col_dims, nlinvnet->model);

			long img_size [2] = { md_calc_size(N, img_dims), Nb };
			long tot_size [2] = { md_calc_size(N, img_dims) + md_calc_size(N, col_dims), Nb };

			auto nlop = nlop_from_linop_F(linop_chain_FF(linop_repmat_create(1, img_size, MD_BIT(0)), linop_expand_create(1, tot_size, img_size)));
			nlop = nlop_chain2_FF(nlop, 0, nlop_zaxpbz_create(1, tot_size, 1, 1), 0);
			result = nn_chain2_swap_FF(nn_from_nlop_F(nlop), 0, NULL, result, 0, "alpha");
			result = nn_set_input_name_F(result, 0, "alpha");
			result = nn_set_input_name_F(result, 0, "lambda");
			result = nn_mark_dup_F(result, "lambda");

			nn_t tmp = nn_from_nlop_F(nlop_del_out_create(1, MD_DIMS(1)));
			tmp = nn_set_input_name_F(tmp, 0, "lambda");

			if (nlinvnet->fix_lambda)
				tmp = nn_set_in_type_F(tmp, 0, "lambda", IN_STATIC);
			else
				tmp = nn_set_in_type_F(tmp, 0, "lambda", IN_OPTIMIZE);;

			tmp = nn_set_initializer_F(tmp, 0, "lambda", init_const_create(0.01));

			network = nn_combine_FF(tmp, network);
		}

		result = nn_chain2_FF(network, 0, NULL, result, 0, "x_0");
		result = nn_dup_F(result, 0, NULL, 1, NULL);
		result = nn_stack_dup_by_name_F(result);
		result = nn_sort_inputs_by_list_F(result, N_in_names_gn + N_in_names_net, in_names);

		for (int i = 0; i < N_in_names_gn + N_in_names_net; i++)
			xfree(in_names[i]);
	} else {

		complex float zero = 0;
		result = nn_set_input_const_F2(result, 0, "x_0", 2, reg_dims, MD_SINGLETON_STRS(2), true, &zero);
	}

	return result;
}


static nn_t nlinvnet_chain_alpha(const struct nlinvnet_s* nlinvnet, nn_t network)
{
	int N_in_names = nn_get_nr_named_in_args(network);
	const char* in_names[N_in_names];
	nn_get_in_names_copy(N_in_names, in_names, network);

	auto dom = nn_generic_domain(network, 0, "alpha");

	auto nlop_scale = nlop_from_linop_F(linop_scale_create(dom->N, dom->dims, 1. / nlinvnet->conf->redu));
	nlop_scale = nlop_chain_FF(nlop_zsadd_create(dom->N, dom->dims, -nlinvnet->conf->alpha_min), nlop_scale);
	nlop_scale = nlop_chain_FF(nlop_scale, nlop_zsadd_create(dom->N, dom->dims, nlinvnet->conf->alpha_min));

	auto scale = nn_from_nlop_F(nlop_scale);
	network = nn_chain2_FF(scale, 0, NULL, network, 0, "alpha");
	network = nn_set_input_name_F(network, -1, "alpha");

	network = nn_sort_inputs_by_list_F(network, N_in_names, in_names);

	for (int i = 0; i < N_in_names; i++)
		xfree(in_names[i]);

	return network;
}


static nn_t nlinvnet_get_iterations(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status, int index_start, int index_end)
{
	int j = index_end;
	nn_t result = NULL;

	if ((index_end == (int)(nlinvnet->conf->iter) - 1) && (nlinvnet->fix_coils)) {

		result = nlinvnet_get_cell_reg(nlinvnet, Nb, j, status, true);
		j++;
	} else {

		result = nlinvnet_get_cell_reg(nlinvnet, Nb, j, status, false);
	}


	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names];
	const char* out_names[N_out_names];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	while (index_start < j--) {

		result = nlinvnet_chain_alpha(nlinvnet, result);

		result = nn_mark_dup_if_exists_F(result, "y");
		result = nn_mark_dup_if_exists_F(result, "x_0");
		result = nn_mark_dup_if_exists_F(result, "alpha");

		auto tmp = nlinvnet_get_cell_reg(nlinvnet, Nb, j, status, false);

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

	result = nn_sort_inputs_by_list_F(result, N_in_names, in_names);
	result = nn_sort_outputs_by_list_F(result, N_out_names, out_names);

	for (int i = 0; i < N_in_names; i++)
		xfree(in_names[i]);
	for (int i = 0; i < N_out_names; i++)
		xfree(out_names[i]);

	return result;
}

static nn_t nlinvnet_init_create(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status)
{

	auto result = nlinvnet_get_iterations(nlinvnet, Nb, status, 0, nlinvnet->conf->iter - 1 - nlinvnet->iter_net);

	int N = noir_model_get_N(nlinvnet->model);
	assert(N == DIMS);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_dims(N, img_dims, nlinvnet->model);
	noir_model_get_col_dims(N, col_dims, nlinvnet->model);
	noir_model_get_cim_dims(N, cim_dims, nlinvnet->model);

	complex float alpha = nlinvnet->conf->alpha;
	long alp_dims[1];
	md_copy_dims(1, alp_dims, nn_generic_domain(result, 0, "alpha")->dims);
	result = nn_set_input_const_F2(result, 0, "alpha", 1, alp_dims, MD_SINGLETON_STRS(N), true, &alpha);	// in: y, xn, x0


	long ini_dims[2];
	md_copy_dims(2, ini_dims, nn_generic_domain(result, 0, NULL)->dims);

	long size = noir_model_get_size(nlinvnet->model);
	long skip = noir_model_get_skip(nlinvnet->model);
	assert(ini_dims[0] == size);

	complex float* init = md_alloc(1, MD_DIMS(size), CFL_SIZE);
	md_clear(1, MD_DIMS(size), init, CFL_SIZE);
	md_zfill(1, MD_DIMS(skip), init, 1.);

	result = nn_set_input_const_F2(result, 0, NULL, 2, ini_dims, MD_DIMS(CFL_SIZE, 0), true, init);	// in: y
	md_free(init);

	complex float scale = 100.;

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

	result = nn_chain2_keep_FF(nn_scale, 0, "y", result, 0, "y");


	long bat_dims[N];
	md_singleton_dims(N, bat_dims);
	bat_dims[BATCH_DIM] = Nb;

	result = nn_chain2_swap_FF(nn_from_nlop_F(nlop_from_linop_F(linop_get_adjoint(linop_loop(N, bat_dims, (struct linop_s*)(nlinvnet->model->lop_fft))))), 0, NULL , result, 0, NULL);
	result = nn_set_input_name_F(result, 0, "ksp");

	const char* out_names[2] = {"y", "scale"};
	result = nn_sort_outputs_by_list_F(result, 2, out_names);
	result = nn_sort_outputs_F(result);

	long xdims[N];
	md_singleton_dims(N, xdims);
	xdims[0] = noir_model_get_size(nlinvnet->model);
	xdims[N - 1] = Nb;
	result = nn_reshape_out_F(result, 0, NULL, N, xdims);

	return result;
}


static nn_t nlinvnet_net_create(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status, enum nlinvnet_out out_type)
{

	auto result = nlinvnet_get_iterations(nlinvnet, Nb, status, nlinvnet->conf->iter - nlinvnet->iter_net, nlinvnet->conf->iter - 1);

	int N = noir_model_get_N(nlinvnet->model);
	assert(N == DIMS);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_dims(N, img_dims, nlinvnet->model);
	noir_model_get_col_dims(N, col_dims, nlinvnet->model);
	noir_model_get_cim_dims(N, cim_dims, nlinvnet->model);

	complex float alpha = nlinvnet->conf->alpha;
	for (int i = 0; i < (int)nlinvnet->conf->iter - nlinvnet->iter_net; i++)
		alpha = nlinvnet->conf->alpha_min + (alpha - nlinvnet->conf->alpha_min) / nlinvnet->conf->redu;

	long alp_dims[1];
	md_copy_dims(1, alp_dims, nn_generic_domain(result, 0, "alpha")->dims);
	result = nn_set_input_const_F2(result, 0, "alpha", 1, alp_dims, MD_SINGLETON_STRS(N), true, &alpha);	// in: y, xn

	long cim_dims2[N];
	long img_dims2[N];
	md_copy_dims(N, cim_dims2, cim_dims);
	md_copy_dims(N, img_dims2, img_dims);
	cim_dims2[BATCH_DIM] = Nb;
	img_dims2[BATCH_DIM] = Nb;


	long sdims[N];
	md_select_dims(N, BATCH_FLAG, sdims, img_dims2);


	switch (out_type) {

		case NLINVNET_OUT_CIM: {

			auto nn_cim = nn_from_nlop_F(noir_cim_batch_create(nlinvnet->model, Nb));
			nn_cim = nn_set_output_name_F(nn_cim, 0, "cim_us");
			result = nn_chain2_swap_FF(result, 0, NULL, nn_cim, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, cim_dims2, cim_dims2, sdims));
			nn_scale = nn_set_output_name_F(nn_scale, 0, "cim");
			nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "cim_us", nn_scale, 0, NULL);
		}
		break;

		case NLINVNET_OUT_IMG: {

			auto nlop_img = noir_split_batch_create(nlinvnet->model, Nb);
			nlop_img = nlop_del_out_F(nlop_img, 1);

			auto nn_img = nn_from_nlop_F(nlop_img);
			nn_img = nn_set_output_name_F(nn_img, 0, "img_us");
			result = nn_chain2_swap_FF(result, 0, NULL, nn_img, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, cim_dims2, cim_dims2, sdims));
			nn_scale = nn_set_output_name_F(nn_scale, 0, "img");
			nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "img_us", nn_scale, 0, NULL);
		}
		break;

		case NLINVNET_OUT_IMG_COL: {

			auto nlop_img = noir_decomp_batch_create(nlinvnet->model, Nb);
			auto nn_img = nn_from_nlop_F(nlop_img);
			nn_img = nn_set_output_name_F(nn_img, 0, "img_us");
			nn_img = nn_set_output_name_F(nn_img, 0, "col");
			result = nn_chain2_swap_FF(result, 0, NULL, nn_img, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, img_dims2, img_dims2, sdims));
			nn_scale = nn_set_output_name_F(nn_scale, 0, "img");
			nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "img_us", nn_scale, 0, NULL);
		}

		break;
	}

	int N_in_names1 = nn_get_nr_named_in_args(result);
	const char* in_names1[N_in_names1];
	const char* in_names2[N_in_names1 + 2];

	nn_get_in_names_copy(N_in_names1, in_names1, result);

	in_names2[0] = "y";
	in_names2[1] = "scale";

	int N_in_names2 = 2;

	for (int i = 0; i < N_in_names1; i++)
		if (0 != strcmp(in_names1[i], "y") * strcmp(in_names1[i], "scale"))
			in_names2[N_in_names2++] = in_names1[i];
	result = nn_sort_inputs_by_list_F(result, N_in_names2, in_names2);
	result = nn_sort_inputs_F(result);

	long xdims[N];
	md_singleton_dims(N, xdims);
	xdims[0] = noir_model_get_size(nlinvnet->model);
	xdims[N - 1] = Nb;
	result = nn_reshape_in_F(result, 0, NULL, N, xdims);

	return result;
}

static nn_t nlinvnet_create(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status, enum nlinvnet_out out_type)
{
	auto init = nlinvnet_init_create(nlinvnet, Nb, status);
	auto net = nlinvnet_net_create(nlinvnet, Nb, status, out_type);

	int N_in_names = nn_get_nr_named_in_args(net);
	const char* in_names[N_in_names + 1];

	in_names[0] = "ksp";
	nn_get_in_names_copy(N_in_names, in_names + 1, net);

	auto result = nn_combine_FF(net, init);
	result = nn_link_F(result, 0, NULL, 0, NULL);
	result = nn_link_F(result, 0, "y", 0, "y");
	result = nn_link_F(result, 0, "scale", 0, "scale");
	result = nn_sort_inputs_by_list_F(result, N_in_names + 1, in_names);

	return result;
}

static nn_t nlinvnet_loss_create(const struct nlinvnet_s* nlinvnet, int Nb, bool valid)
{
	auto train_op = nlinvnet_create(nlinvnet, valid ? 1 : Nb, valid ? STAT_TEST : STAT_TRAIN, NLINVNET_OUT_CIM);

	if (valid) {

		train_op = nn_del_out_bn_F(train_op);

		for (int i = 1; i < Nb; i++) {

			auto tmp = nlinvnet_create(nlinvnet, 1, valid ? STAT_TEST : STAT_TRAIN, NLINVNET_OUT_CIM);
			tmp = nn_del_out_bn_F(tmp);

			int N_in_names = nn_get_nr_named_in_args(tmp);
			const char* in_names[N_in_names];
			nn_get_in_names_copy(N_in_names, in_names, tmp);

			tmp = nn_mark_stack_input_F(tmp, "ksp");
			tmp = nn_mark_stack_output_F(tmp, "cim");
			for (int i = 0; i < N_in_names; i++)
				tmp = nn_mark_dup_if_exists_F(tmp, in_names[i]);

			train_op = nn_combine_FF(train_op, tmp);
			train_op = nn_stack_dup_by_name_F(train_op);
		}
	}

	int N = noir_model_get_N(nlinvnet->model);
	assert(N == DIMS);

	long cim_dims[N];
	noir_model_get_cim_dims(N, cim_dims, nlinvnet->model);
	cim_dims[BATCH_DIM] = Nb;

	nn_t loss = NULL;
	if (valid)
		loss = val_measure_create(nlinvnet->valid_loss, N, cim_dims);
	else
		loss = train_loss_create(nlinvnet->train_loss, N, cim_dims);

	train_op = nn_chain2_FF(train_op, 0, "cim", loss, 0, NULL);

	return train_op;
}

static nn_t nlinvnet_net_loss_create(const struct nlinvnet_s* nlinvnet, int Nb, bool valid)
{
	auto train_op = nlinvnet_net_create(nlinvnet, Nb, valid ? STAT_TEST : STAT_TRAIN, NLINVNET_OUT_CIM);

	int N = noir_model_get_N(nlinvnet->model);
	assert(N == DIMS);

	long cim_dims[N];
	noir_model_get_cim_dims(N, cim_dims, nlinvnet->model);
	cim_dims[BATCH_DIM] = Nb;

	nn_t loss = NULL;
	if (valid)
		loss = val_measure_create(nlinvnet->valid_loss, N, cim_dims);
	else
		loss = train_loss_create(nlinvnet->train_loss, N, cim_dims);

	train_op = nn_chain2_FF(train_op, 0, "cim", loss, 0, NULL);

	if (valid)
		train_op = nn_del_out_bn_F(train_op);

	return train_op;
}

static nn_t nlinvnet_valid_create(const struct nlinvnet_s* nlinvnet, int N, const long cim_dims[N], const complex float* ref, const long ksp_dims[N], const complex float* ksp, int N_batch_inputs)
{
	int Nb = ksp_dims[BATCH_DIM];

	auto valid_loss = nlinvnet_loss_create(nlinvnet, Nb, true);

	valid_loss = nn_set_input_const_F(valid_loss, 0, NULL, N, cim_dims, true, ref);
	valid_loss = nn_set_input_const_F(valid_loss, 0, "ksp", N, ksp_dims, true, ksp);
	for (int i = 0; i < N_batch_inputs; i++)
		valid_loss = nn_combine_FF(nn_from_nlop_F(nlop_del_out_create(1, MD_SINGLETON_DIMS(1))), valid_loss);

	return valid_loss;
}


static nn_t nlinvnet_apply_op_create(const struct nlinvnet_s* nlinvnet, enum nlinvnet_out out_type, int Nb)
{
	auto nn_apply = nlinvnet_create(nlinvnet, Nb, STAT_TEST, out_type);
	return nn_get_wo_weights_F(nn_apply, nlinvnet->weights, false);
}



void train_nlinvnet(	struct nlinvnet_s* nlinvnet, int N, int Nb,
			const long cim_dims_trn[N], const complex float* ref_trn, const long ksp_dims_trn[N], const complex float* ksp_trn,
			const long cim_dims_val[N], const complex float* ref_val, const long ksp_dims_val[N], const complex float* ksp_val)
{

	assert(DIMS == N);
	long Nt = ksp_dims_trn[BATCH_DIM]; // number datasets
	assert(Nt == cim_dims_trn[BATCH_DIM]);

	Nb = MIN(Nb, Nt);

	long ksp_dims_bat[N];
	long cim_dims_bat[N];
	md_copy_dims(N, ksp_dims_bat, ksp_dims_trn);
	md_copy_dims(N, cim_dims_bat, cim_dims_trn);
	ksp_dims_bat[BATCH_DIM] = Nb;
	cim_dims_bat[BATCH_DIM] = Nb;

	bool precompute = (0 == nlinvnet->ksp_noise);

	auto nn_train = (precompute ? nlinvnet_net_loss_create : nlinvnet_loss_create)(nlinvnet, Nb, false);

	if (nn_is_name_in_in_args(nn_train, "lambda")) {

		auto iov = nn_generic_domain(nn_train, 0, "lambda");
		auto prox_conv = operator_project_min_real_create(iov->N, iov->dims, 0.001);
		nn_train = nn_set_prox_op_F(nn_train, 0, "lambda", prox_conv);
	}


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

	complex float* init = NULL;
	complex float* ksp_grid = NULL;
	complex float* scale = 	NULL;

	const struct nlop_s* batch_generator = NULL;
	int N_batch_inputs = -1;

	if (precompute) {

		long xdims_b[N];
		long kdims_b[N];
		long sdims_b[N];

		md_copy_dims(nn_generic_domain(nn_train, 1, NULL)->N, xdims_b, nn_generic_domain(nn_train, 1, NULL)->dims);
		md_copy_dims(nn_generic_domain(nn_train, 0, "y")->N, kdims_b, nn_generic_domain(nn_train, 0, "y")->dims);
		md_copy_dims(nn_generic_domain(nn_train, 0, "scale")->N, sdims_b, nn_generic_domain(nn_train, 0, "scale")->dims);

		long xdims_t[N];
		long kdims_t[N];
		long sdims_t[N];

		md_copy_dims(N, xdims_t, xdims_b);
		md_copy_dims(N, kdims_t, kdims_b);
		md_copy_dims(N, sdims_t, sdims_b);

		xdims_t[BATCH_DIM] = Nt;
		kdims_t[BATCH_DIM] = Nt;
		sdims_t[BATCH_DIM] = Nt;

		complex float* init = 	md_alloc(N, xdims_t, CFL_SIZE);
		complex float* ksp_grid = md_alloc(N, kdims_t, CFL_SIZE);
		complex float* scale = 	md_alloc(N, sdims_t, CFL_SIZE);

		auto nn_init = nlinvnet_init_create(nlinvnet, 1, STAT_TRAIN);
		nlop_generic_apply_loop_sameplace(nn_init->nlop, BATCH_FLAG,
			3, (int [3]){N, N, N}, (const long* [3]){xdims_t, kdims_t, sdims_t}, (complex float* [3]) {init, ksp_grid, scale},
			1, (int [1]){N}, (const long* [1]){ksp_dims_trn}, (const complex float* [1]) {ksp_trn},
			nlinvnet->weights->tensors[0]);
		nn_free(nn_init);

		//create batch generator
		const complex float* train_data[] = {ref_trn, init, ksp_grid, scale};
		const long* bat_dims[] = { cim_dims_bat, xdims_b, kdims_b, sdims_b };
		const long* tot_dims[] = { cim_dims_trn, xdims_t, kdims_t, sdims_t };

		N_batch_inputs = 4;
		batch_generator = batch_gen_create_from_iter(nlinvnet->train_conf, 4, (const int[4]){ N, N, N, N}, bat_dims, tot_dims, train_data, 0);
	} else {

		//create batch generator
		const complex float* train_data[] = {ref_trn, ksp_trn};
		const long* bat_dims[] = { cim_dims_bat, ksp_dims_bat };
		const long* tot_dims[] = { cim_dims_trn, ksp_dims_trn };

		N_batch_inputs = 2;
		batch_generator = batch_gen_create_from_iter(nlinvnet->train_conf, 2, (const int[2]){ N, N}, bat_dims, tot_dims, train_data, 0);

		if (0 != nlinvnet->ksp_noise)
			batch_generator = nlop_append_FF(batch_generator, 1, nlop_add_noise_create(N, ksp_dims_bat, nlinvnet->ksp_noise, 0, ~BATCH_FLAG));
	}

	//setup for iter algorithm
	int NI = nn_get_nr_in_args(nn_train);
	int NO = nn_get_nr_out_args(nn_train);

	float* src[NI];

	for (int i = 0; i < nlinvnet->weights->N; i++) {

		auto iov_weight = nlinvnet->weights->iovs[i];
		auto iov_train_op = nlop_generic_domain(nn_get_nlop(nn_train), i + N_batch_inputs);
		assert(md_check_equal_dims(iov_weight->N, iov_weight->dims, iov_train_op->dims, ~0));
		src[i + N_batch_inputs] = (float*)nlinvnet->weights->tensors[i];
	}

	enum IN_TYPE in_type[NI];
	const struct operator_p_s* projections[NI];
	enum OUT_TYPE out_type[NO];

	nn_get_in_types(nn_train, NI, in_type);
	nn_get_out_types(nn_train, NO, out_type);

	for (int i = 0; i < N_batch_inputs; i++) {

		src[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
	}

	for (int i = 0; i < NI; i++)
		projections[i] = nn_get_prox_op_arg_index(nn_train, i);

	int num_monitors = 0;
	const struct monitor_value_s* value_monitors[3];

	if (NULL != ref_val) {

		assert(NULL != ksp_val);

		auto nn_validation_loss = nlinvnet_valid_create(nlinvnet, N, cim_dims_val, ref_val, ksp_dims_val, ksp_val, N_batch_inputs);
		const char* val_names[nn_get_nr_out_args(nn_validation_loss)];
		for (int i = 0; i < nn_get_nr_out_args(nn_validation_loss); i++)
			val_names[i] = nn_get_out_name_from_arg_index(nn_validation_loss, i, false);
		value_monitors[num_monitors] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, nn_get_nr_out_args(nn_validation_loss), val_names);
		nn_free(nn_validation_loss);
		num_monitors += 1;
	}

	if (nn_is_name_in_in_args(nn_train, "lambda")) {

		int index_lambda = nn_get_in_arg_index(nn_train, 0, "lambda");
		int num_lambda = nn_generic_domain(nn_train, 0, "lambda")->dims[0];

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

	md_free(init);
	md_free(ksp_grid);
	md_free(scale);

	monitor_iter6_free(monitor);
}


void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N, const long img_dims[N], complex float* img, const long col_dims[N], complex float* col, const long ksp_dims[N], const complex float* ksp)
{
	if (nlinvnet->gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	assert(DIMS == N);

	int DO[2] = { N, N };
	int DI[1] = { N };

	const long* odims[2] = { img_dims, col_dims };
	const long* idims[1] = { ksp_dims };

	complex float* dst[2] = { img, col };
	const complex float* src[1] = { ksp };

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, NLINVNET_OUT_IMG_COL, 1);

	nlop_generic_apply_loop_sameplace(nn_get_nlop(nn_apply), BATCH_FLAG, 2, DO, odims, dst, 1, DI, idims, src, nlinvnet->weights->tensors[0]);

	nn_free(nn_apply);

	if (nlinvnet->normalize_rss) {

		complex float* tmp = md_alloc_sameplace(N, img_dims, CFL_SIZE, img);
		md_zrss(N, col_dims, COIL_FLAG, tmp, col);
		md_zmul(N, img_dims, img, img, tmp);
		md_free(tmp);
	}
}

static void apply_nlinvnet_cim(struct nlinvnet_s* nlinvnet, int N, const long cim_dims[N], complex float* cim, const long ksp_dims[N], const complex float* ksp)
{
	if (nlinvnet->gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	assert(DIMS == N);

	int DO[1] = { N };
	int DI[1] = { N };

	const long* odims[1] = { cim_dims };
	const long* idims[1] = { ksp_dims };

	complex float* dst[1] = { cim };
	const complex float* src[1] = { ksp };

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, NLINVNET_OUT_CIM, 1);

	nlop_generic_apply_loop_sameplace(nn_get_nlop(nn_apply), BATCH_FLAG, 1, DO, odims, dst, 1, DI, idims, src, nlinvnet->weights->tensors[0]);

	nn_free(nn_apply);
}

void eval_nlinvnet(struct nlinvnet_s* nlinvnet, int N, const long cim_dims[N], const complex float* ref, const long ksp_dims[N], const complex float* ksp)
{
	complex float* tmp_out = md_alloc(N, cim_dims, CFL_SIZE);

	auto loss = val_measure_create(nlinvnet->valid_loss, N, cim_dims);
	int NL = nn_get_nr_out_args(loss);
	complex float losses[NL];
	md_clear(1, MD_DIMS(NL), losses, CFL_SIZE);

	apply_nlinvnet_cim(nlinvnet, N, cim_dims, tmp_out, ksp_dims, ksp);

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
