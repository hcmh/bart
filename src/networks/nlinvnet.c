#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

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
	.share_weights = false,

	.weights = NULL,
	.train_conf = NULL,

	.conf = NULL,
	.model = NULL,
	.iter_conf = NULL,
	.iter_init = 3,
	.iter_no_net = 4,

	.train_loss = &loss_nlinvnet,
	.valid_loss = &loss_nlinvnet,

	.gpu = false,
	.low_mem = false,

	.graph_file = NULL,
};

void nlinvnet_init_varnet_default(struct nlinvnet_s* nlinvnet)
{
	if (NULL == nlinvnet->train_conf) {

		PTR_ALLOC(struct iter6_iPALM_conf, train_conf);
		*train_conf = iter6_iPALM_conf_defaults;
		nlinvnet->train_conf = CAST_UP(PTR_PASS(train_conf));
		nlinvnet->train_conf->epochs = 100;
		nlinvnet->train_conf->batchgen_type = BATCH_GEN_SHUFFLE_DATA;
	}

	if (NULL == nlinvnet->network) {

		nlinvnet->network = CAST_UP(&network_varnet_default);
		nlinvnet->network->norm = NORM_MAX;
	}
}

void nlinvnet_init_resnet_default(struct nlinvnet_s* nlinvnet)
{
	if (NULL == nlinvnet->train_conf) {

		PTR_ALLOC(struct iter6_adam_conf, train_conf);
		*train_conf = iter6_adam_conf_defaults;
		nlinvnet->train_conf = CAST_UP(PTR_PASS(train_conf));
		nlinvnet->train_conf->epochs = 100;
		nlinvnet->train_conf->batchgen_type = BATCH_GEN_SHUFFLE_DATA;
	}

	if (NULL == nlinvnet->network) {

		nlinvnet->network = CAST_UP(&network_resnet_default);
		nlinvnet->network->norm = NORM_MAX;
	}
}

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
	nlinvnet->iter_conf->INTERFACE.alpha = 1.;
	nlinvnet->iter_conf->l2lambda = 1.;
	nlinvnet->iter_conf->maxiter = 50;//(0 == nlinvnet->conf->cgiter) ? 30 : nlinvnet->conf->cgiter;
	nlinvnet->iter_conf->tol = 0.;

	assert(0 == nlinvnet->iter_conf->tol);
}

static nn_t nlinvnet_get_gauss_newton_step(const struct nlinvnet_s* nlinvnet, int Nb, float update)
{
	auto result = nn_from_nlop_F(noir_gauss_newton_step_batch_create(nlinvnet->model, nlinvnet->iter_conf, Nb, update));
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

	//network = nn_chain2_FF(network, 0, NULL, nn_from_nlop_F(nlop_dump_create(N, img_dims, "out", true, false, false)), 0, NULL);
	//network = nn_chain2_FF(nn_from_nlop_F(nlop_dump_create(N, img_dims, "in", true, false, false)), 0, NULL, network, 0, NULL);

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

static nn_t nlinvnet_get_cell(const struct nlinvnet_s* nlinvnet, int Nb, bool network, float update, enum NETWORK_STATUS status)
{
	auto result = nlinvnet_get_gauss_newton_step(nlinvnet, Nb, update);

	long reg_dims[2];
	md_copy_dims(2, reg_dims, nn_generic_domain(result, 0, "x_0")->dims);

	complex float zero = 0;
	result = nn_set_input_const_F2(result, 0, "x_0", 2, reg_dims, MD_SINGLETON_STRS(2), true, &zero);

	if (network) {

		auto network = nlinvnet_get_network_step(nlinvnet, Nb, status, true);

		int N_in_names_gn = nn_get_nr_named_in_args(result);
		int N_in_names_net = nn_get_nr_named_in_args(network);

		const char* in_names[N_in_names_gn + N_in_names_net];
		nn_get_in_names_copy(N_in_names_gn, in_names, result);
		nn_get_in_names_copy(N_in_names_net, in_names + N_in_names_gn, network);

		result = nn_chain2_FF(network, 0, NULL, result, 0, NULL);
		result = nn_sort_inputs_by_list_F(result, N_in_names_gn + N_in_names_net, in_names);

		for (int i = 0; i < N_in_names_gn + N_in_names_net; i++)
			xfree(in_names[i]);
	}

	return result;
}

static nn_t nlinvnet_get_cell_reg(const struct nlinvnet_s* nlinvnet, int Nb, bool network, float update, enum NETWORK_STATUS status)
{
	auto result = nlinvnet_get_gauss_newton_step(nlinvnet, Nb, update);

	long reg_dims[2];
	md_copy_dims(2, reg_dims, nn_generic_domain(result, 0, "x_0")->dims);

	if (network) {

		auto network = nlinvnet_get_network_step(nlinvnet, Nb, status, false);

		int N_in_names_gn = nn_get_nr_named_in_args(result);
		int N_in_names_net = nn_get_nr_named_in_args(network);

		const char* in_names[N_in_names_gn + N_in_names_net];
		nn_get_in_names_copy(N_in_names_gn, in_names, result);
		nn_get_in_names_copy(N_in_names_net, in_names + N_in_names_gn, network);

		result = nn_chain2_FF(network, 0, NULL, result, 0, "x_0");
		result = nn_dup_F(result, 0, NULL, 1, NULL);
		result = nn_sort_inputs_by_list_F(result, N_in_names_gn + N_in_names_net, in_names);

		for (int i = 0; i < N_in_names_gn + N_in_names_net; i++)
			xfree(in_names[i]);
	} else {

		complex float zero = 0;
		result = nn_set_input_const_F2(result, 0, "x_0", 2, reg_dims, MD_SINGLETON_STRS(2), true, &zero);
	}

	return result;
}


static nn_t nlinvnet_chain_alpha(nn_t network, float redu)
{
	int N_in_names = nn_get_nr_named_in_args(network);
	const char* in_names[N_in_names];
	nn_get_in_names_copy(N_in_names, in_names, network);

	auto dom = nn_generic_domain(network, 0, "alpha");

	auto scale = nn_from_nlop_F(nlop_from_linop_F(linop_scale_create(dom->N, dom->dims, 1. / redu)));
	network = nn_chain2_FF(scale, 0, NULL, network, 0, "alpha");
	network = nn_set_input_name_F(network, -1, "alpha");

	network = nn_sort_inputs_by_list_F(network, N_in_names, in_names);

	for (int i = 0; i < N_in_names; i++)
		xfree(in_names[i]);

	return network;
}


static nn_t nlinvnet_get_iterations(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status)
{
	int j = nlinvnet->conf->iter;
	auto result = nlinvnet_get_cell(nlinvnet, Nb, j > nlinvnet->iter_no_net, j > nlinvnet->iter_init ? 1. : 0.5, status);

	int N_in_names = nn_get_nr_named_in_args(result);
	int N_out_names = nn_get_nr_named_out_args(result);

	const char* in_names[N_in_names];
	const char* out_names[N_out_names];

	nn_get_in_names_copy(N_in_names, in_names, result);
	nn_get_out_names_copy(N_out_names, out_names, result);

	while (0 < --j) {

		result = nlinvnet_chain_alpha(result, nlinvnet->conf->redu);

		result = nn_mark_dup_if_exists_F(result, "y");
		result = nn_mark_dup_if_exists_F(result, "x_0");
		result = nn_mark_dup_if_exists_F(result, "alpha");

		auto tmp = nlinvnet_get_cell_reg(nlinvnet, Nb, j > nlinvnet->iter_no_net, j > nlinvnet->iter_init ? 1. : 0.5, status);

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

static nn_t nlinvnet_create(const struct nlinvnet_s* nlinvnet, int Nb, enum NETWORK_STATUS status, enum nlinvnet_out out_type)
{

	auto result = nlinvnet_get_iterations(nlinvnet, Nb, status);

	int N = noir_model_get_N(nlinvnet->model);
	assert(N == DIMS);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_dims(N, img_dims, nlinvnet->model);
	noir_model_get_col_dims(N, col_dims, nlinvnet->model);
	noir_model_get_cim_dims(N, cim_dims, nlinvnet->model);

	complex float alpha = nlinvnet->conf->alpha;
	long alp_dims[N];
	md_copy_dims(N, alp_dims, nn_generic_domain(result, 0, "alpha")->dims);
	result = nn_set_input_const_F2(result, 0, "alpha", N, alp_dims, MD_SINGLETON_STRS(N), true, &alpha);	// in: y, xn, x0


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
	nn_scale = nn_set_output_name_F(nn_scale, 1, "scale");

	result = nn_chain2_FF(nn_scale, 0, NULL, result, 0, "y");


	switch (out_type) {

		case NLINVNET_OUT_CIM: {

			auto nn_cim = nn_from_nlop_F(noir_cim_batch_create(nlinvnet->model, Nb));
			nn_cim = nn_set_output_name_F(nn_cim, 0, "cim_us");
			result = nn_chain2_swap_FF(result, 0, NULL, nn_cim, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, cim_dims2, cim_dims2, sdims));
			nn_scale = nn_set_output_name_F(nn_scale, 0, "cim");
			nn_scale = nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "cim_us", nn_scale, 0, NULL);
			result = nn_link_F(result, 0, "scale", 0, "scale");

			//result = nn_chain2_FF(result, 0, "cim", nn_from_nlop_F(nlop_dump_create(N, cim_dims2, "cim", true, false, false)), 0, NULL);
			//result = nn_set_output_name_F(result, 0, "cim");
		}
		break;

		case NLINVNET_OUT_IMG: {

			auto nlop_img = noir_split_batch_create(nlinvnet->model, Nb);
			nlop_img = nlop_del_out_F(nlop_img, 1);

			auto nn_img = nn_from_nlop_F(nlop_img);
			nn_img = nn_set_output_name_F(nn_img, 0, "img_us");
			result = nn_chain2_swap_FF(result, 0, NULL, nn_img, 0, NULL);

			auto nn_scale = nn_from_nlop_F(nlop_tenmul_create(N, cim_dims2, cim_dims2, sdims));
			nn_set_output_name_F(nn_scale, 0, "img");
			nn_set_input_name_F(nn_scale, 1, "scale");

			result = nn_chain2_FF(result, 0, "img_us", nn_scale, 0, NULL);
			result = nn_link_F(result, 0, "scale", 0, "scale");
		}
		break;
	}

	long bat_dims[N];
	md_singleton_dims(N, bat_dims);
	bat_dims[BATCH_DIM] = Nb;

	result = nn_chain2_swap_FF(nn_from_nlop_F(nlop_from_linop_F(linop_get_adjoint(linop_loop(N, bat_dims, (struct linop_s*)(nlinvnet->model->lop_fft))))), 0, NULL , result, 0, NULL);
	result = nn_set_input_name_F(result, 0, "ksp");

	return result;
}


static nn_t nlinvnet_loss_create(const struct nlinvnet_s* nlinvnet, int Nb, bool valid)
{
	auto train_op = nlinvnet_create(nlinvnet, Nb, valid ? STAT_TEST : STAT_TRAIN, NLINVNET_OUT_CIM);

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

static nn_t nlinvnet_valid_create(const struct nlinvnet_s* nlinvnet, int N, const long cim_dims[N], const complex float* ref, const long ksp_dims[N], const complex float* ksp)
{
	int Nb = ksp_dims[BATCH_DIM];

	auto valid_loss = nlinvnet_loss_create(nlinvnet, Nb, true);

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL, N, cim_dims, true, ref);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "ksp", N, ksp_dims, true, ksp);

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

	auto nn_train = nlinvnet_loss_create(nlinvnet, Nb, false);

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

	//create batch generator
	const complex float* train_data[] = {ref_trn, ksp_trn};
	const long* bat_dims[] = { cim_dims_bat, ksp_dims_bat };
	const long* tot_dims[] = { cim_dims_trn, ksp_dims_trn };

	auto batch_generator = batch_gen_create_from_iter(nlinvnet->train_conf, 2, (const int[4]){ N, N}, bat_dims, tot_dims, train_data, 0);

	//setup for iter algorithm
	int NI = nn_get_nr_in_args(nn_train);
	int NO = nn_get_nr_out_args(nn_train);

	float* src[NI];

	for (int i = 0; i < nlinvnet->weights->N; i++) {

		auto iov_weight = nlinvnet->weights->iovs[i];
		auto iov_train_op = nlop_generic_domain(nn_get_nlop(nn_train), i + 2);
		assert(md_check_equal_dims(iov_weight->N, iov_weight->dims, iov_train_op->dims, ~0));
		src[i + 2] = (float*)nlinvnet->weights->tensors[i];
	}

	enum IN_TYPE in_type[NI];
	const struct operator_p_s* projections[NI];
	enum OUT_TYPE out_type[NO];

	nn_get_in_types(nn_train, NI, in_type);
	nn_get_out_types(nn_train, NO, out_type);

	for (int i = 0; i < 2; i++) {

		src[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
	}

	for (int i = 0; i < NI; i++)
		projections[i] = nn_get_prox_op_arg_index(nn_train, i);

	int num_monitors = 0;
	const struct monitor_value_s* value_monitors[3];

	if (NULL != ref_val) {

		assert(NULL != ksp_val);

		auto nn_validation_loss = nlinvnet_valid_create(nlinvnet, N, cim_dims_val, ref_val, ksp_dims_val, ksp_val);
		const char* val_names[nn_get_nr_out_args(nn_validation_loss)];
		for (int i = 0; i < nn_get_nr_out_args(nn_validation_loss); i++)
			val_names[i] = nn_get_out_name_from_arg_index(nn_validation_loss, i, false);
		value_monitors[num_monitors] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, nn_get_nr_out_args(nn_validation_loss), val_names);
		nn_free(nn_validation_loss);
		num_monitors += 1;
	}

	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, num_monitors, value_monitors);

	iter6_by_conf(nlinvnet->train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, NO, out_type, Nb, Nt / Nb, batch_generator, monitor);

	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}


void apply_nlinvnet(struct nlinvnet_s* nlinvnet, enum nlinvnet_out out_type, int N, const long img_dims[N], complex float* out, const long ksp_dims[N], const complex float* ksp)
{
	assert(DIMS == N);
	long Nb = ksp_dims[BATCH_DIM]; // number datasets
	assert(Nb == img_dims[BATCH_DIM]);

	complex float* out_tmp = md_alloc_sameplace(N, img_dims, CFL_SIZE, nlinvnet->weights->tensors[0]);
	complex float* ksp_tmp = md_alloc_sameplace(N, ksp_dims, CFL_SIZE, nlinvnet->weights->tensors[0]);

	md_copy(N, ksp_dims, ksp_tmp, ksp, CFL_SIZE);

	complex float* args[2];
	args[0] = out_tmp;
	args[1] = ksp_tmp;

	auto nn_apply = nlinvnet_apply_op_create(nlinvnet, out_type, Nb);
	nlop_generic_apply_select_derivative_unchecked(nn_get_nlop(nn_apply), 2, (void**)args, 0, 0);
	nn_free(nn_apply);

	md_copy(N, img_dims, out, out_tmp, CFL_SIZE);

	md_free(out_tmp);
	md_free(ksp_tmp);
}

void apply_nlinvnet_batchwise(struct nlinvnet_s* nlinvnet, enum nlinvnet_out out_type, int N, const long img_dims[N], complex float* out, const long ksp_dims[N], const complex float* ksp, int Nb)
{
	long Nt = img_dims[BATCH_DIM];

	long ksp_dims1[N];
	long img_dims1[N];

	md_copy_dims(N, ksp_dims1, ksp_dims);
	md_copy_dims(N, img_dims1, img_dims);

	long ksp_strs[N];
	long img_strs[N];

	md_calc_strides(N, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	while (0 < Nt) {

		long Nb_tmp = MIN(Nt, Nb);

		img_dims1[BATCH_DIM] = Nb_tmp;
		ksp_dims1[BATCH_DIM] = Nb_tmp;

		apply_nlinvnet(	nlinvnet, out_type, N,
				img_dims1, &MD_ACCESS(N, img_strs, pos, out),
				ksp_dims1, &MD_ACCESS(N, ksp_strs, pos, ksp));

		pos[BATCH_DIM] += Nb_tmp;
		Nt -= Nb_tmp;
	}
}

void eval_nlinvnet(struct nlinvnet_s* nlinvnet, enum nlinvnet_out out_type, int N, const long cim_dims[N], const complex float* ref, const long ksp_dims[N], const complex float* ksp, int Nb)
{
	complex float* tmp_out = md_alloc(N, cim_dims, CFL_SIZE);

	auto loss = val_measure_create(nlinvnet->valid_loss, N, cim_dims);
	int NL = nn_get_nr_out_args(loss);
	complex float losses[NL];
	md_clear(1, MD_DIMS(NL), losses, CFL_SIZE);

	apply_nlinvnet_batchwise(nlinvnet, out_type, N, cim_dims, tmp_out, ksp_dims, ksp, Nb);

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
