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
	.Nb = 1,
	.models = NULL,
	.model_valid = NULL,
	.iter_conf = NULL,
	.iter_init = 3,
	.iter_net = 3,

	.train_loss = &loss_option,
	.valid_loss = &val_loss_option,

	.normalize_rss = false,

	.gpu = false,
	.low_mem = true,

	.fix_lambda = false,

	.ksp_training = true,
	.ksp_split = -1.,
	.ksp_noise = 0.,

	.graph_file = NULL,
};

static void nlinvnet_init(struct nlinvnet_s* nlinvnet)
{
	nlinvnet->iter_conf = TYPE_ALLOC(struct iter_conjgrad_conf);
	*(nlinvnet->iter_conf) = iter_conjgrad_defaults;
	nlinvnet->iter_conf->INTERFACE.alpha = 0.;
	nlinvnet->iter_conf->l2lambda = 0.;
	nlinvnet->iter_conf->maxiter = (0 == nlinvnet->conf->cgiter) ? 30 : nlinvnet->conf->cgiter;
	nlinvnet->iter_conf->tol = 0.;

	if (NULL == get_loss_from_option())
			nlinvnet->train_loss->weighting_mse=1.;

	if (NULL == get_val_loss_from_option())
		nlinvnet->valid_loss = &loss_image_valid;

	assert(0 == nlinvnet->iter_conf->tol);
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

static nn_t nlinvnet_get_gauss_newton_step(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], float update, bool fix_coils)
{
	auto result = nn_from_nlop_F(noir_gauss_newton_step_batch_create(Nb, models, nlinvnet->iter_conf, update, fix_coils));
	result = nn_set_input_name_F(result, 0, "y");
	result = nn_set_input_name_F(result, 1, "x_0");
	result = nn_set_input_name_F(result, 1, "alpha");

	return result;
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

	auto join = nn_from_nlop_F(noir_set_img_batch_create(Nb, models));
	auto split = nn_from_nlop_F(noir_extract_img_batch_create(Nb, models));

	network = nn_chain2_FF(split, 0, NULL, network, 0, NULL);
	network = nn_chain2_FF(network, 0, NULL, join, 0, NULL);

	return nn_checkpoint_F(network, false, true);
}


static nn_t nlinvnet_get_cell_reg(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], int index, enum NETWORK_STATUS status, bool fix_coil)
{
	assert(0 <= index);
	bool network = (index >= ((int)nlinvnet->conf->iter - nlinvnet->iter_net));

	float update = index < nlinvnet->iter_init ? 0.5 : 1;

	auto result = nlinvnet_get_gauss_newton_step(nlinvnet, Nb, models, update, fix_coil);

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
		result = nn_chain2_swap_FF(nn_from_nlop_F(nlop), 0, NULL, result, 0, "alpha");
		result = nn_set_input_name_F(result, 0, "alpha");
		result = nn_set_input_name_F(result, 0, "lambda");
		result = nn_mark_dup_F(result, "lambda");

		//make lambda dummy input of network
		nn_t tmp = nn_from_nlop_F(nlop_del_out_create(1, MD_DIMS(1)));
		tmp = nn_set_input_name_F(tmp, 0, "lambda");
		tmp = nn_set_in_type_F(tmp, 0, "lambda", IN_OPTIMIZE);;
		tmp = nn_set_initializer_F(tmp, 0, "lambda", init_const_create(0.01));
		network = nn_combine_FF(tmp, network);

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


static nn_t nlinvnet_get_iterations(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], enum NETWORK_STATUS status)
{
	int j = nlinvnet->conf->iter - 1;

	auto result = nlinvnet_get_cell_reg(nlinvnet, Nb, models, j, status, false);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names];
	const char* out_names[N_out_names];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	while (0 < j--) {

		result = nlinvnet_chain_alpha(nlinvnet, result);

		result = nn_mark_dup_if_exists_F(result, "y");
		result = nn_mark_dup_if_exists_F(result, "x_0");
		result = nn_mark_dup_if_exists_F(result, "alpha");

		auto tmp = nlinvnet_get_cell_reg(nlinvnet, Nb, models, j, status, false);

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

static nn_t nlinvnet_create(const struct nlinvnet_s* nlinvnet, int Nb, struct noir2_s* models[Nb], enum NETWORK_STATUS status, enum nlinvnet_out out_type)
{

	auto result = nlinvnet_get_iterations(nlinvnet, Nb, models, status);

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
	md_copy_dims(1, alp_dims, nn_generic_domain(result, 0, "alpha")->dims);
	result = nn_set_input_const_F2(result, 0, "alpha", 1, alp_dims, MD_SINGLETON_STRS(N), true, &alpha);	// in: y, xn, x0


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

			if (NLINVNET_OUT_KSP == out_type) {

				nn_t fft_op = NULL;

				if (nlinvnet->conf->noncart)
					fft_op = nn_from_nlop_F(noir_nufft_batch_create(Nb, models));
				else
				 	fft_op = nn_from_nlop_F(nlop_combine_FF(noir_fft_batch_create(Nb, models), nlop_del_out_create(N, MD_SINGLETON_DIMS(N))));


				fft_op = nn_set_output_name_F(fft_op, 0, "ksp");
				fft_op = nn_set_input_name_F(fft_op, 1, "trj");
				fft_op = nn_mark_dup_F(fft_op, "trj");
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
	 	result = nn_chain2_swap_FF(nn_from_nlop_F(nlop_combine_FF(noir_adjoint_fft_batch_create(Nb, models), nlop_del_out_create(N, MD_SINGLETON_DIMS(N)))), 0, NULL , result, 0, NULL);

	result = nn_set_input_name_F(result, 0, "ksp");
	result = nn_set_input_name_F(result, 0, "pat");
	result = nn_set_input_name_F(result, 0, "trj");
	result = nn_stack_dup_by_name_F(result);

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

		int N_in_names = nn_get_nr_named_in_args(train_op);
		const char* in_names[N_in_names];
		nn_get_in_names_copy(N_in_names, in_names, train_op);

		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 1, NULL);
		loss = nn_chain2_FF(nn_from_nlop_F(nlop_tenmul_create(N, out_dims, out_dims, pat_dims)), 0, NULL, loss, 0, NULL);
		loss = nn_dup_F(loss, 1, NULL, 3, NULL);
		loss = nn_set_input_name_F(loss, 1, "pat_ref");
		train_op = nn_chain2_FF(train_op, 0, "ksp", loss, 1, NULL);

		if (-1. != nlinvnet->ksp_split) {

			auto split_op = nn_from_nlop_F(nlop_rand_split_create(N, pat_dims, 0, nlinvnet->ksp_split));
			split_op = nn_set_output_name_F(split_op, 0, "pat_trn");
			split_op = nn_set_output_name_F(split_op, 0, "pat_ref");

			train_op = nn_chain2_swap_FF(split_op, 0, "pat_ref", train_op, 0, "pat_ref");
			train_op = nn_link_F(train_op, 0, "pat_trn", 0, "pat");
			train_op = nn_shift_input_F(train_op, 1, NULL, 0, NULL);
			train_op = nn_set_input_name_F(train_op, 1, "pat");
		} else {

			train_op = nn_dup_F(train_op, 0, "pat", 0, "pat_ref");
		}

		train_op = nn_sort_inputs_by_list_F(train_op, N_in_names, in_names);
	} else {

		train_op = nn_chain2_FF(train_op, 0, "cim", loss, 0, NULL);
	}

	return train_op;
}

static nn_t nlinvnet_valid_create(const struct nlinvnet_s* nlinvnet, int N, const long cim_dims[N], const complex float* ref, const long ksp_dims[N], const complex float* ksp, const long pat_dims[N], const complex float* pat, const long trj_dims[N], const complex float* trj)
{
	int Nb = ksp_dims[BATCH_DIM];

	auto result = nlinvnet_create(nlinvnet, 1, &(((struct nlinvnet_s*)nlinvnet)->model_valid), STAT_TEST, NLINVNET_OUT_CIM);
	result = nn_del_out_bn_F(result);

	for (int i = 1; i < Nb; i++) {

		auto tmp = nlinvnet_create(nlinvnet, 1, &(((struct nlinvnet_s*)nlinvnet)->model_valid), STAT_TEST, NLINVNET_OUT_CIM);
		tmp = nn_del_out_bn_F(tmp);

		int N_in_names = nn_get_nr_named_in_args(tmp);
		const char* in_names[N_in_names];
		nn_get_in_names_copy(N_in_names, in_names, tmp);

		tmp = nn_mark_stack_input_F(tmp, "ksp");
		tmp = nn_mark_stack_input_F(tmp, "pat");
		tmp = nn_mark_stack_input_F(tmp, "trj");
		tmp = nn_mark_stack_output_F(tmp, "cim");
		for (int i = 0; i < N_in_names; i++)
			tmp = nn_mark_dup_if_exists_F(tmp, in_names[i]);

		result = nn_combine_FF(result, tmp);
		result = nn_stack_dup_by_name_F(result);
	}

	long out_dims[N];
	md_copy_dims(N, out_dims, nn_generic_codomain(result, 0, "cim")->dims);
	auto valid_loss = nn_chain2_FF(result, 0, "cim", val_measure_create(nlinvnet->valid_loss, N, out_dims), 0, NULL);

	valid_loss = nn_set_input_const_F(valid_loss, 0, NULL, N, cim_dims, true, ref);
	valid_loss = nn_set_input_const_F(valid_loss, 0, "ksp", N, ksp_dims, true, ksp);

	complex float one = 1.;

	auto pat_dom = nn_generic_domain(valid_loss, 0, "pat");
	valid_loss = nn_set_input_const_F2(valid_loss, 0, "pat", pat_dom->N, pat_dom->dims, MD_STRIDES(N, pat_dims, CFL_SIZE), true, pat ? pat : &one);

	auto trj_dom = nn_generic_domain(valid_loss, 0, "trj");
	valid_loss = nn_set_input_const_F2(valid_loss, 0, "trj", trj_dom->N, trj_dom->dims, MD_STRIDES(N, trj_dims, CFL_SIZE), true, trj ? trj : &one);

	for (int i = 0; i < 4; i++)
		valid_loss = nn_combine_FF(nn_from_nlop_F(nlop_del_out_create(1, MD_SINGLETON_DIMS(1))), valid_loss);

	return valid_loss;
}


static nn_t nlinvnet_apply_op_create(const struct nlinvnet_s* nlinvnet, enum nlinvnet_out out_type, int Nb)
{
	auto nn_apply = nlinvnet_create(nlinvnet, Nb, nlinvnet->models, STAT_TEST, out_type);
	return nn_get_wo_weights_F(nn_apply, nlinvnet->weights, false);
}



void train_nlinvnet(	struct nlinvnet_s* nlinvnet, int N, int Nb,
			const long ref_dims_trn[N], const complex float* ref_trn, const long ksp_dims_trn[N], const complex float* ksp_trn, const long pat_dims_trn[N], const complex float* pat_trn, const long trj_dims_trn[N], const complex float* trj_trn,
			const long ref_dims_val[N], const complex float* ref_val, const long ksp_dims_val[N], const complex float* ksp_val, const long pat_dims_val[N], const complex float* pat_val, const long trj_dims_val[N], const complex float* trj_val)
{

	assert(DIMS == N);
	long Nt = ksp_dims_trn[BATCH_DIM]; // number datasets
	assert(Nt == ref_dims_trn[BATCH_DIM]);

	Nb = MIN(Nb, Nt);

	long ksp_dims_bat[N];
	long ref_dims_bat[N];
	long pat_dims_bat[N];
	long trj_dims_bat[N];
	md_copy_dims(N, ksp_dims_bat, ksp_dims_trn);
	md_copy_dims(N, ref_dims_bat, ref_dims_trn);
	md_copy_dims(N, pat_dims_bat, pat_dims_trn);
	md_copy_dims(N, trj_dims_bat, trj_dims_trn ? trj_dims_trn : MD_SINGLETON_DIMS(N));
	ksp_dims_bat[BATCH_DIM] = Nb;
	ref_dims_bat[BATCH_DIM] = Nb;
	pat_dims_bat[BATCH_DIM] = Nb;
	trj_dims_bat[BATCH_DIM] = Nb;

	auto nn_train = nlinvnet_train_loss_create(nlinvnet);

	if (nn_is_name_in_in_args(nn_train, "lambda")) {

		auto iov = nn_generic_domain(nn_train, 0, "lambda");
		auto prox_conv = operator_project_pos_real_create(iov->N, iov->dims);
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

	if (nlinvnet->fix_lambda)
		nn_train = nn_set_in_type_F(nn_train, 0, "lambda", IN_STATIC);

	const struct nlop_s* batch_generator = NULL;
	int N_batch_inputs = 3;

	//create batch generator
	complex float one = 1;
	const complex float* train_data[] = {ref_trn, ksp_trn, pat_trn, trj_trn ? trj_trn : &one};
	const long* bat_dims[] = { ref_dims_bat, ksp_dims_bat, pat_dims_bat, trj_dims_bat };
	const long* tot_dims[] = { ref_dims_trn, ksp_dims_trn, pat_dims_trn, trj_dims_trn };
	N_batch_inputs = 4;
	batch_generator = batch_gen_create_from_iter(nlinvnet->train_conf, N_batch_inputs, (const int[4]){ N, N, N, N }, bat_dims, tot_dims, train_data, 0);
	if (0 != nlinvnet->ksp_noise)
		batch_generator = nlop_append_FF(batch_generator, 1, nlop_add_noise_create(N, ksp_dims_bat, nlinvnet->ksp_noise, 0, ~BATCH_FLAG));


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

		auto nn_validation_loss = nlinvnet_valid_create(nlinvnet, N, ref_dims_val, ref_val, ksp_dims_val, ksp_val, pat_dims_val, pat_val, trj_dims_val, trj_val);
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

	monitor_iter6_free(monitor);
}


void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N, const long img_dims[N], complex float* img, const long col_dims[N], complex float* col, const long ksp_dims[N], const complex float* ksp, const long pat_dims[N], const complex float* pat, const long trj_dims[N], const complex float* trj)
{
	if (nlinvnet->gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	assert(DIMS == N);

	int DO[2] = { N, N };
	int DI[3] = { N, N, N };

	const long* odims[2] = { img_dims, col_dims };
	const long* idims[3] = { ksp_dims, pat_dims, trj_dims };

	complex float* dst[2] = { img, col };
	const complex float* src[3] = { ksp, pat, trj };

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, NLINVNET_OUT_IMG_COL, 1);

	nlop_generic_apply_loop_sameplace(nn_get_nlop(nn_apply), BATCH_FLAG, 2, DO, odims, dst, 3, DI, idims, src, nlinvnet->weights->tensors[0]);

	nn_free(nn_apply);

	if (nlinvnet->normalize_rss) {

		complex float* tmp = md_alloc_sameplace(N, img_dims, CFL_SIZE, img);
		md_zrss(N, col_dims, COIL_FLAG, tmp, col);
		md_zmul(N, img_dims, img, img, tmp);
		md_free(tmp);
	}
}

static void apply_nlinvnet_cim(struct nlinvnet_s* nlinvnet, int N, const long cim_dims[N], complex float* cim, const long ksp_dims[N], const complex float* ksp, const long pat_dims[N], const complex float* pat, const long trj_dims[N], const complex float* trj)
{
	if (nlinvnet->gpu)
		move_gpu_nn_weights(nlinvnet->weights);

	assert(DIMS == N);

	int DO[1] = { N };
	int DI[3] = { N, N, N };

	const long* odims[1] = { cim_dims };
	const long* idims[3] = { ksp_dims, pat_dims, trj_dims };

	complex float* dst[1] = { cim };
	const complex float* src[3] = { ksp, pat, trj };

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, NLINVNET_OUT_CIM, 1);

	nlop_generic_apply_loop_sameplace(nn_get_nlop(nn_apply), BATCH_FLAG, 1, DO, odims, dst, 3, DI, idims, src, nlinvnet->weights->tensors[0]);

	nn_free(nn_apply);
}

void eval_nlinvnet(struct nlinvnet_s* nlinvnet, int N, const long cim_dims[N], const complex float* ref, const long ksp_dims[N], const complex float* ksp, const long pat_dims[N], const complex float* pat, const long trj_dims[N], const complex float* trj)
{
	complex float* tmp_out = md_alloc(N, cim_dims, CFL_SIZE);

	auto loss = val_measure_create(nlinvnet->valid_loss, N, cim_dims);
	int NL = nn_get_nr_out_args(loss);
	complex float losses[NL];
	md_clear(1, MD_DIMS(NL), losses, CFL_SIZE);

	apply_nlinvnet_cim(nlinvnet, N, cim_dims, tmp_out, ksp_dims, ksp, pat_dims, pat, trj_dims, trj);

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
