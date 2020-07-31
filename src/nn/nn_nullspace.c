#include <assert.h>
#include <float.h>

#include "iter/italgos.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

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
#include "nn/vn.h"
#include "nn/unet.h"

#include "nn_nullspace.h"


const struct nullspace_s nullspace_default = {

	.Nb = 10,

	.lambda = NULL,
	.lambda_init = .05,
	.lambda_min = 0.,
	.lambda_max = FLT_MAX,
	.lambda_fixed = -1.,

	.multi_lambda = 1.,

	.nullspace = true,
	.share_mask = true,
	.rescale = false,

	.unet = NULL,
};

static const struct nlop_s* nlop_unet_network_create(const struct nullspace_s* config, long dims[5], long udims[5])
{
	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	auto nlop_zf = nlop_mri_adjoint_create(dims, config->share_mask);
	nlop_zf = nlop_chain2_FF(nlop_zf, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0); // in: kspace, coil, mask; out: Atb

	auto nlop_norm_inv = mri_normal_inversion_create_general_with_lambda(5, dims, 23, 31, config->share_mask ? 7 : 23, 31, 7, config->lambda_fixed); // in: Atb, coil, mask, lambda; out: A^+b
	nlop_norm_inv = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_norm_inv, 0);
	nlop_norm_inv = nlop_chain2_FF(nlop_norm_inv, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);

	auto nlop_pseudo_inv = nlop_chain2_swap_FF(nlop_zf, 0, nlop_norm_inv, 0); // in: kspace, coil, mask, coil, mask, lambda; out: A^+b
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 1, 3);
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 2, 3);// in: kspace, coil, mask, lambda; out: A^+b

	long udims_unet[5] = {1, udims[0], udims[1], udims[2], udims[4]};
	auto nlop_unet = nn_unet_create(config->unet, udims_unet);
	nlop_unet = nlop_reshape_in_F(nlop_unet, 0, 5, udims);
	nlop_unet = nlop_reshape_out_F(nlop_unet, 0, 5, udims);

	nlop_unet = nlop_chain2_FF(nlop_unet, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0);
	nlop_unet = nlop_dup_F(nlop_unet, 0, 1); // in: u0, [ unet_weights ]; out: u0 + Unet(u0), [Unet_Batchnorm]


	auto nlop_result = nlop_chain2_swap_FF(nlop_pseudo_inv, 0, nlop_unet, 0); // in: kspace, coil, mask, lambda, coil, mask, lambda, [ unet_weights ]; out: u0 + P Unet(u0), [Unet_Batchnorm]
	return nlop_result;
}

static const struct nlop_s* nlop_nullspace_network_multi_lambda_create(const struct nullspace_s* config, long dims[5], long udims[5])
{
	if (!config->nullspace)
		return nlop_unet_network_create(config, dims, udims);

	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};

	auto nlop_pseudo_inv = mri_Tikhonov_regularized_pseudo_inv(5, dims, config->lambda_fixed, config->share_mask, config->rescale, true);
	nlop_pseudo_inv = nlop_chain2_FF(nlop_pseudo_inv, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);


	auto nlop_pseudo_inv2 = mri_Tikhonov_regularized_pseudo_inv(5, dims, (-1. == config->lambda_fixed) ? -1. : config->multi_lambda * config->lambda_fixed, config->share_mask, config->rescale, true);
	nlop_pseudo_inv2 = nlop_chain2_FF(nlop_pseudo_inv2, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);
	if (-1. == config->lambda_fixed)
		nlop_pseudo_inv2 = nlop_chain2_FF(nlop_from_linop_F(linop_scale_create(1, MD_SINGLETON_DIMS(1), config->multi_lambda)), 0, nlop_pseudo_inv2, 3);

	auto nlop_proj = mri_reg_projection_ker_create_general_with_lambda(5, dims, 23, 31, config->share_mask ? 7 : 23, 31, 7, config->lambda_fixed); // in: DU(u0), coil, mask, lambda; out: PDU(u0)
	nlop_proj = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_proj, 0);
	nlop_proj = nlop_chain2_FF(nlop_proj, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);

	long udims_unet[5] = {1, udims[0], udims[1], udims[2], udims[4]};
	long udims_unet_multi[5] = {2, udims[0], udims[1], udims[2], udims[4]};

	auto nlop_unet = nn_unet_create(config->unet, udims_unet_multi);
	nlop_unet = nlop_chain2_swap_FF(nlop_stack_create(5, udims_unet_multi, udims_unet, udims_unet, 0), 0, nlop_unet, 0);
	nlop_unet = nlop_reshape_in_F(nlop_unet, 0, 5, udims);
	nlop_unet = nlop_reshape_in_F(nlop_unet, 1, 5, udims);
	nlop_unet = nlop_reshape_out_F(nlop_unet, 0, 5, udims);

	auto nlop_unet_proj = nlop_chain2_FF(nlop_unet, 0, nlop_proj, 0); // in: coil, mask, lambda, u0, u1, [ unet_weights ]; out: P Unet(u0), [Unet_Batchnorm]

	nlop_unet_proj = nlop_chain2_FF(nlop_unet_proj, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0);
	nlop_unet_proj = nlop_dup_F(nlop_unet_proj, 0, 4); // in: u0, coil, mask, lambda, u1, [ unet_weights ]; out: u0 + P Unet(u0), [Unet_Batchnorm]

	auto nlop_result = nlop_chain2_swap_FF(nlop_pseudo_inv, 0, nlop_unet_proj, 0); // in: kspace, coil, mask, lambda, coil, mask, lambda, u1, [ unet_weights ]; out: u0 + P Unet(u0), [Unet_Batchnorm]
	nlop_result = nlop_dup_F(nlop_result, 1, 4);
	nlop_result = nlop_dup_F(nlop_result, 2, 4);
	nlop_result = nlop_dup_F(nlop_result, 3, 4);

	nlop_result = nlop_chain2_swap_FF(nlop_pseudo_inv2, 0, nlop_result, 4); // in: kspace, coil, mask, lambda, kspace, coil, mask, lambda, u1, [ unet_weights ]; out: u0 + P Unet(u0), [Unet_Batchnorm]
	nlop_result = nlop_dup_F(nlop_result, 0, 4);
	nlop_result = nlop_dup_F(nlop_result, 1, 4);
	nlop_result = nlop_dup_F(nlop_result, 2, 4);
	nlop_result = nlop_dup_F(nlop_result, 3, 4);

	return nlop_result;
}


static const struct nlop_s* nlop_nullspace_network_create(const struct nullspace_s* config, long dims[5], long udims[5])
{
	if (!config->nullspace)
		return nlop_unet_network_create(config, dims, udims);

	if (1. != config->multi_lambda)
		return nlop_nullspace_network_multi_lambda_create(config, dims, udims);

	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};

	auto nlop_pseudo_inv = mri_Tikhonov_regularized_pseudo_inv(5, dims, config->lambda_fixed, config->share_mask, config->rescale, true);
	nlop_pseudo_inv = nlop_chain2_FF(nlop_pseudo_inv, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);

	auto nlop_proj = mri_reg_projection_ker_create_general_with_lambda(5, dims, 23, 31, config->share_mask ? 7 : 23, 31, 7, config->lambda_fixed); // in: DU(u0), coil, mask, lambda; out: PDU(u0)
	nlop_proj = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_proj, 0);
	nlop_proj = nlop_chain2_FF(nlop_proj, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);

	long udims_unet[5] = {1, udims[0], udims[1], udims[2], udims[4]};
	auto nlop_unet = nn_unet_create(config->unet, udims_unet);
	nlop_unet = nlop_reshape_in_F(nlop_unet, 0, 5, udims);
	nlop_unet = nlop_reshape_out_F(nlop_unet, 0, 5, udims);

	auto nlop_unet_proj = nlop_chain2_FF(nlop_unet, 0, nlop_proj, 0); // in: coil, mask, lambda, u0, [ unet_weights ]; out: P Unet(u0), [Unet_Batchnorm]
	nlop_unet_proj = nlop_chain2_FF(nlop_unet_proj, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0);
	nlop_unet_proj = nlop_dup_F(nlop_unet_proj, 0, 4); // in: u0, coil, mask, lambda, [ unet_weights ]; out: u0 + P Unet(u0), [Unet_Batchnorm]

	auto nlop_result = nlop_chain2_swap_FF(nlop_pseudo_inv, 0, nlop_unet_proj, 0); // in: kspace, coil, mask, lambda, coil, mask, lambda, [ unet_weights ]; out: u0 + P Unet(u0), [Unet_Batchnorm]
	nlop_result = nlop_dup_F(nlop_result, 1, 4);
	nlop_result = nlop_dup_F(nlop_result, 2, 4);
	nlop_result = nlop_dup_F(nlop_result, 3, 4);

	return nlop_result;
}

static const struct nlop_s* nlop_val = NULL;

static complex float compute_validation_objective(long NI, const float* x[NI])
{
	if (NULL == nlop_val)
		return 0.;
	assert(NULL != x);

	void* args[NI + 1];
	for (int i = 0; i < NI; i++)
		args[i + 1] = (void*)x[i];

	args[0] = md_alloc_sameplace(1, MAKE_ARRAY(1l), sizeof(_Complex float), args[1]);

	enum NETWORK_STATUS stat_tmp = network_status;
	network_status = STAT_TEST;
	nlop_generic_apply_select_derivative_unchecked(nlop_val, NI + 1, args, 0, 0);
	network_status = stat_tmp;

	float result = 0;

	md_copy(1, MAKE_ARRAY(1l), &result, args[0], sizeof(float));
	md_free(args[0]);

	return (complex float)result;
}

static complex float get_lambda(long NI, const float* x[NI])
{
	complex float result = 0;
	md_copy(1, MD_SINGLETON_DIMS(1), &result, x[4], CFL_SIZE);
	return result;
}


void train_nn_nullspace(	struct nullspace_s* nullspace, iter6_conf* train_conf,
				const long idims[5], const _Complex float* ref,
				const long kdims[5], const _Complex float* kspace,
				const long cdims[5], const _Complex float* coil,
				const long mdims[5], const _Complex float* mask,
				bool random_order, const char* history_filename, const char** valid_files)
{
	long N_datasets = idims[4];

	assert(idims[4] == kdims[4]);
	assert(idims[4] == cdims[4]);

	long dims[5];
	md_copy_dims(5, dims, kdims);
	dims[4] = nullspace->Nb;
	long udims[5];
	md_copy_dims(5, udims, idims);
	udims[4] = nullspace->Nb;

	nullspace->share_mask = (1 == mdims[4]);

	auto nlop_train = nlop_nullspace_network_create(nullspace, dims, udims);
	nlop_train = nlop_chain2_FF(nlop_train, 0, nlop_mse_create(5, udims, MD_BIT(4)), 0);
	long nidims[5];
	md_copy_dims(5, nidims, idims);
	nidims[4] = nullspace->Nb;
	nlop_train = nlop_reshape_in_F(nlop_train, 0, 5, nidims);
	//in: ref, kspace, coil, mask, lambda, conv0, convi, convn, bias0, biasi, biasn, gamman, bn0_in, bni_in, bnn_in; out: loss, bn0_out, bni_out, bnn_out

	//create batch generator
	const complex float* train_data[] = {ref, kspace, coil, mask};
	const long* train_dims[] = {	nlop_generic_domain(nlop_train, 0)->dims,
					nlop_generic_domain(nlop_train, 1)->dims,
					nlop_generic_domain(nlop_train, 2)->dims,
					nlop_generic_domain(nlop_train, 3)->dims};

	auto batch_generator = (random_order ? batch_gen_rand_create : batch_gen_linear_create)(4, 5, train_dims, train_data, N_datasets, 0);

	//setup for iter algorithm
	long num_out_args = 1;
	long num_in_args = 5;
	long num_out_unet = nn_unet_get_number_out_weights(nullspace->unet);
	long num_in_unet = nn_unet_get_number_in_weights(nullspace->unet);

	long num_args = num_out_args + num_in_args + num_out_unet + num_in_unet;

	float* data[num_in_args + num_in_unet];
	enum IN_TYPE in_type[num_in_args + num_in_unet];
	const struct operator_p_s* projections[num_in_args + num_in_unet];
	for (int i = 0; i < num_in_args + num_in_unet; i++)
		projections[i] = NULL;

	for (int i = 0; i < 4; i++) {

		data[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
		projections[i] = NULL;
	}

	data[4] = (float*)nullspace->lambda;
	projections[4] = operator_project_real_interval_create(nlop_generic_domain(nlop_train, 4)->N, nlop_generic_domain(nlop_train, 4)->dims, nullspace->lambda_min, nullspace->lambda_max);
	in_type[4] = IN_OPTIMIZE;

	nn_unet_get_in_weights_pointer(nullspace->unet, num_in_unet, (complex float**)data + 5);
	nn_unet_get_in_types(nullspace->unet, num_in_unet, in_type + 5);


	enum OUT_TYPE out_type[num_out_args + num_out_unet];
	out_type[0] = OUT_OPTIMIZE;
	nn_unet_get_out_types(nullspace->unet, num_out_unet, out_type + 1);

	network_status = STAT_TRAIN;

	const struct nlop_s* valid_loss = NULL;
	if (NULL != valid_files) {

		long kdims[5];
		long cdims[5];
		long udims[5];
		long mdims[5];

		complex float* val_kspace = load_cfl(valid_files[0], 5, kdims);
		complex float* val_coil = load_cfl(valid_files[1], 5, cdims);
		complex float* val_mask = load_cfl(valid_files[2], 5, mdims);
		complex float* val_ref = load_cfl(valid_files[3], 5, udims);

		complex float* scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
		complex float* u0 = md_alloc(5, udims, CFL_SIZE);
		compute_zero_filled(udims, u0, kdims, val_kspace, val_coil, mdims, val_mask);
		compute_scale(udims, scaling, u0);
		md_free(u0);

		normalize_max(udims, scaling, val_ref, val_ref);
		normalize_max(kdims, scaling, val_kspace, val_kspace);

		valid_loss = nlop_nullspace_network_create(nullspace, dims, udims);

		const struct nlop_s* loss = nlop_mse_create(5, udims, MD_BIT(4));
		loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);
		loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);

		valid_loss = nlop_chain2_FF(valid_loss, 0, loss, 0);

		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, udims, true, val_ref);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, kdims, true, val_kspace);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, cdims, true, val_coil);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, mdims, true, val_mask);

		auto nlop_del = nlop_del_out_create(5, udims);
		nlop_del = nlop_combine_FF(nlop_del, nlop_del_out_create(5, kdims));
		nlop_del = nlop_combine_FF(nlop_del, nlop_del_out_create(5, cdims));
		nlop_del = nlop_combine_FF(nlop_del, nlop_del_out_create(5, mdims));

		for (int i = 0; i < num_out_unet; i++)
			valid_loss = nlop_del_out_F(valid_loss, 1);

		valid_loss = nlop_combine_FF(nlop_del, valid_loss);

		nlop_val = valid_loss;

		unmap_cfl(5, udims, val_ref);
		unmap_cfl(5, kdims, val_kspace);
		unmap_cfl(5, cdims, val_coil);
		unmap_cfl(5, mdims, val_mask);
	}

	struct iter6_monitor_value_s val_monitors[2];
	val_monitors[0] = (struct iter6_monitor_value_s){&compute_validation_objective, &"val loss"[0], false};
	val_monitors[1] = (struct iter6_monitor_value_s){&get_lambda, &"lambda"[0], true};

	auto conf = CAST_DOWN(iter6_adam_conf, train_conf);
	//auto monitor = create_iter6_monitor_progressbar_validloss(conf->epochs, N_datasets / nullspace->Nb, false, 15, in_type, valid_loss, false);
	auto monitor = create_iter6_monitor_progressbar_value_monitors(conf->epochs, N_datasets / nullspace->Nb, false, 2, val_monitors);
	iter6_adam(train_conf, nlop_train, num_in_args + num_in_unet, in_type, projections, data, num_out_args + num_out_unet, out_type, nullspace->Nb, N_datasets / nullspace->Nb, batch_generator, monitor);
	if (NULL != history_filename)
		iter6_monitor_dump_record(monitor, history_filename);
	network_status = STAT_TEST;

	nlop_free(nlop_train);
	nlop_free(batch_generator);
}


void apply_nn_nullspace(	struct nullspace_s* nullspace,
			const long idims[5], _Complex float* out,
			const long kdims[5], const _Complex float* kspace,
			const long cdims[5], const _Complex float* coil,
			const long mdims[5], const _Complex float* mask)
{
	START_TIMER;
	long N_datasets = idims[4];
	assert(idims[4] == kdims[4]);
	assert(idims[4] == cdims[4]);

	nullspace->share_mask = (1 == mdims[4]);

	//setup for iter algorithm
	long num_out_args = 1;
	long num_in_args = 4;
	long num_out_unet = nn_unet_get_number_out_weights(nullspace->unet);
	long num_in_unet = nn_unet_get_number_in_weights(nullspace->unet);

	complex float* args[num_in_args + num_in_unet + num_out_args];

	nlop_free(nlop_nullspace_network_create(nullspace, kdims, idims));

	args[0] = md_alloc_sameplace(5, idims, CFL_SIZE, nullspace->lambda);
	args[1] = md_alloc_sameplace(5, kdims, CFL_SIZE, nullspace->lambda);
	args[2] = md_alloc_sameplace(5, cdims, CFL_SIZE, nullspace->lambda);
	args[3] = md_alloc_sameplace(5, mdims, CFL_SIZE, nullspace->lambda);
	args[4] = nullspace->lambda;
	nn_unet_get_in_weights_pointer(nullspace->unet, num_in_unet, args + 5);


	complex float* args2[4] = {args[0], args[1], args[2], args[3]}; //save pointers for md_free

	md_copy(5, kdims, args[1], kspace, CFL_SIZE);
	md_copy(5, cdims, args[2], coil, CFL_SIZE);
	md_copy(5, mdims, args[3], mask, CFL_SIZE);

	while (N_datasets > 0) {

		long Nb =  MIN(nullspace->Nb, N_datasets);

		long dims[5];
		md_copy_dims(5, dims, kdims);
		dims[4] = Nb;
		long udims[5];
		md_copy_dims(5, udims, idims);
		udims[4] = Nb;

		N_datasets -= Nb;
		auto nlop_nullspace = nlop_nullspace_network_create(nullspace, dims, udims);

		for (int i = 0; i < num_out_unet; i++)
			nlop_nullspace = nlop_del_out_F(nlop_nullspace, 1);

		enum NETWORK_STATUS tmp = network_status;
		network_status = STAT_TEST;

		nlop_generic_apply_unchecked(nlop_nullspace, num_in_args + num_in_unet + num_out_args, (void**)args);

		network_status = tmp;

		args[1] += md_calc_size(nlop_generic_domain(nlop_nullspace, 0)->N, nlop_generic_domain(nlop_nullspace, 0)->dims);
		args[2] += md_calc_size(nlop_generic_domain(nlop_nullspace, 1)->N, nlop_generic_domain(nlop_nullspace, 1)->dims);
		if (!nullspace->share_mask)
			args[3] += md_calc_size(nlop_generic_domain(nlop_nullspace, 2)->N, nlop_generic_domain(nlop_nullspace, 2)->dims);

		args[0] += md_calc_size(nlop_generic_codomain(nlop_nullspace, 0)->N, nlop_generic_codomain(nlop_nullspace, 0)->dims);

		nlop_free(nlop_nullspace);
	}

	md_copy(5, idims, out, args2[0], CFL_SIZE);

	md_free(args2[0]);
	md_free(args2[1]);
	md_free(args2[2]);
	md_free(args2[3]);
	PRINT_TIMER("nullspace");
}


void init_nn_nullspace(struct nullspace_s* nullspace, long udims[5])
{
	if (NULL == nullspace->lambda)
		nullspace->lambda = md_alloc(1, MD_SINGLETON_DIMS(1), CFL_SIZE);

	md_zfill(1, MD_SINGLETON_DIMS(1), nullspace->lambda, nullspace->lambda_init);

	if (-1. != nullspace->lambda_fixed)
		md_zfill(1, MD_SINGLETON_DIMS(1), nullspace->lambda, nullspace->lambda_fixed);

	long dims[5] = {(1. == nullspace->multi_lambda) ? 1 : 2, udims[0], udims[1], udims[2], udims[4]};

	nn_unet_initialize(nullspace->unet, dims);
}

void nn_nullspace_move_gpucpu(struct nullspace_s* nullspace, bool gpu)
{
	nn_unet_move_cpugpu(nullspace->unet, gpu);

	assert((NULL != nullspace->lambda));

#ifdef USE_CUDA
	complex float* tmp = (gpu ? md_alloc_gpu : md_alloc)(1, MD_SINGLETON_DIMS(1), CFL_SIZE);
	md_copy(1, MD_SINGLETON_DIMS(1), tmp, nullspace->lambda, CFL_SIZE);
	md_free(nullspace->lambda);
	nullspace->lambda = tmp;
#else
	assert(!gpu);
#endif
}

extern void nn_nullspace_store_weights(struct nullspace_s* nullspace, const char* name)
{
	long size = nn_unet_get_weights_size(nullspace->unet) + 1;

	complex float* file = create_cfl(name, 1, &size);
	md_copy(1, MD_SINGLETON_DIMS(1), file, nullspace->lambda, CFL_SIZE);
	nn_unet_store_weights(nullspace->unet, size - 1, file + 1);

	unmap_cfl(1, &size, file);
}

extern void nn_nullspace_load_weights(struct nullspace_s* nullspace, const char* name)
{
	long size = 0;

	complex float* file = load_cfl(name, 1, &size);

	if (NULL == nullspace->lambda)
		nullspace->lambda = md_alloc(1, MD_SINGLETON_DIMS(1), CFL_SIZE);

	md_copy(1, MD_SINGLETON_DIMS(1), nullspace->lambda, file, CFL_SIZE);
	nn_unet_load_weights(nullspace->unet, size - 1, file + 1);

	unmap_cfl(1, &size, file);
}

extern void nn_nullspace_free_weights(struct nullspace_s* nullspace)
{
	md_free(nullspace->lambda);
	nn_unet_free_weights(nullspace->unet);
}
