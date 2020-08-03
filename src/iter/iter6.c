/* Copyright 2017-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <stdio.h>

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter2.h"
#include "iter/iter6_ops.h"

#include "misc/mmio.h"

#include "iter6.h"

#define NLOP2ITNLOP(nlop) (struct iter_nlop_s){ (NULL == nlop) ? NULL : iter6_nlop, CAST_UP(STRUCT_TMP_COPY(((struct iter6_nlop_s){ { &TYPEID(iter6_nlop_s) }, nlop }))) }
#define NLOP2IT_ADJ_ARR(nlop) ({\
	long NO = nlop_get_nr_out_args(nlop);\
	long NI = nlop_get_nr_in_args(nlop);\
	const struct operator_s** adj_ops = (const struct operator_s**) alloca(sizeof(struct operator_s*) * NI * NO);\
	for (int o = 0; o < NO; o++)\
		for (int i = 0; i < NI; i++)\
			adj_ops[i * NO + o] = nlop_get_derivative(nlop, o, i)->adjoint;\
	struct iter6_op_arr_s adj_ops_data = { { &TYPEID(iter6_op_arr_s) }, NI, NO, adj_ops};\
	(struct iter_op_arr_s){iter6_op_arr_fun_deradj, CAST_UP(STRUCT_TMP_COPY(adj_ops_data))} ;})


DEF_TYPEID(iter6_sgd_conf);
DEF_TYPEID(iter6_adadelta_conf);
DEF_TYPEID(iter6_adam_conf);
DEF_TYPEID(iter6_iPALM_conf);


const struct iter6_sgd_conf iter6_sgd_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_sgd_conf),

	.epochs = 1,
	.learning_rate = 0.01,

	.clip_norm = 0.,
	.clip_val = 0.,

	.momentum = 0.
};


const struct iter6_adadelta_conf iter6_adadelta_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adadelta_conf),

	.epochs = 1,
	.learning_rate = 1.,

	.clip_norm = 0.0,
	.clip_val = 0.0,

	.rho = 0.95,

	.batchnorm_mom = 0.95
};


const struct iter6_adam_conf iter6_adam_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adam_conf),

	.epochs = 1,
	.learning_rate = .001,

	.clip_norm = 0.0,
	.clip_val = 0.0,

	.epsilon = 1.e-7,

	.beta1 = 0.9,
	.beta2 = 0.999,

	.batchnorm_mom = 0.95
};


const struct iter6_iPALM_conf iter6_iPALM_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_iPALM_conf),

	.L = 1000.,
	.Lmin = 1.e-4,
	.Lmax = 1.e12,
	.Lshrink = 1.2,
	.Lincrease = 2.,

	.alpha = -1.,
	.beta = -1.,
	.convex = false,

	.trivial_stepsize = false,

	.alpha_arr = NULL,
	.beta_arr =NULL,
	.convex_arr = NULL,

	.epochs = 1000,

	.save_modulo = 100,
	.save_path = NULL,
	.save_name = NULL,
};



struct iter6_nlop_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;
};

DEF_TYPEID(iter6_nlop_s);

static void iter6_nlop(iter_op_data* _o, int N, float* args[N], operator_run_opt_flags_t run_opts[N][N])
{
	const auto data = CAST_DOWN(iter6_nlop_s, _o);

	assert((unsigned int)N == operator_nr_args(data->nlop->op));

	nlop_generic_apply_extopts_unchecked(data->nlop, N, (void*)args, run_opts);
}

struct iter6_op_arr_s {

	INTERFACE(iter_op_data);

	long NO;
	long NI;

	const struct operator_s** ops;
};

DEF_TYPEID(iter6_op_arr_s);

static void iter6_op_arr_fun_deradj(iter_op_data* _o, int NO, unsigned long oflags, float* dst[NO], int NI, unsigned long iflags, const float* src[NI])
{
	const auto data = CAST_DOWN(iter6_op_arr_s, _o);

	assert(NO == data->NO);
	assert(1 == NI);
	int i_index = -1;

	for (unsigned int i = 0; i < data->NI; i++)
		if (MD_IS_SET(iflags, i)) {
			assert(-1 == i_index);
			i_index = i;
		}
	assert(-1 != i_index);

	const struct operator_s* op_arr[NO];
	float* dst_t[NO];
	int NO_t = 0;

	for (int o = 0; o < NO; o++)
		if (MD_IS_SET(oflags, o)) {

			op_arr[NO_t] = data->ops[o * data->NI + i_index];
			dst_t[NO_t] = dst[o];
			NO_t += 1;
		}
#if 0
	for (int i = 0; i < NO_t; i++)
		operator_apply_unchecked(op_arr[i], ((complex float**)dst_t)[i], (const complex float*)(src[0]));
#else
	operator_apply_parallel_unchecked(NO_t, op_arr, (complex float**)dst_t, (const complex float*)(src[0]));
#endif
}

static void iter6_op_arr_fun_diag(iter_op_data* _o, int NO, unsigned long oflags, float* dst[NO], int NI, unsigned long iflags, const float* src[NI])
{
	const struct iter6_op_arr_s* data = CAST_DOWN(iter6_op_arr_s, _o);

	assert(NO == data->NO);
	assert(NI == data->NI);
	assert(oflags == iflags);

	for (int i = 0; i < NI; i++)
		if (MD_IS_SET(iflags, i))
			operator_apply_unchecked(data->ops[i * NI + i], (_Complex float*)dst[i], (_Complex float*)src[i]);
}

void iter6_adadelta(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct iter6_monitor_s* monitor)
{
	auto conf = CAST_DOWN(iter6_adadelta_conf, _conf);

	//assert(NULL == nlop_batch_gen);
	//assert(NULL == prox_ops);

	struct iter_nlop_s nlop_iter = NLOP2ITNLOP(nlop);
	struct iter_op_arr_s adj_op_arr = NLOP2IT_ADJ_ARR(nlop);
	struct iter_nlop_s nlop_batch_gen_iter = NLOP2ITNLOP(nlop_batch_gen);

	struct iter_op_p_s prox_iter[NI];
	for (unsigned int i = 0; i < NI; i++)
		prox_iter[i] = OPERATOR_P2ITOP((NULL == prox_ops ? NULL : prox_ops[i]));

	long isize[NI];
	long osize[NO];

	//array of update operators
	const struct operator_s* upd_ops[NI][NI];
	for (int i = 0; i < NI; i++) {

		for (int j = 0; j < NI; j++)
			upd_ops[i][j] = NULL;
		upd_ops[i][i] = operator_adadelta_update_create(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, conf->learning_rate, conf->rho, 1.e-7);
		if ((0.0 != conf->clip_norm) || (0.0 != conf->clip_val)) {

			const struct operator_s* tmp1 = operator_clip_create(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, conf->clip_norm, conf->clip_val);
			const struct operator_s* tmp2 = upd_ops[i][i];
			upd_ops[i][i] = operator_chain(tmp1, tmp2);
			operator_free(tmp1);
			operator_free(tmp2);
		}
	}
	struct iter6_op_arr_s upd_ops_data = { { &TYPEID(iter6_op_arr_s) }, NI, NI, &(upd_ops[0][0])};
	struct iter_op_arr_s upd_op_arr ={iter6_op_arr_fun_diag, CAST_UP(&upd_ops_data)};


	for (int i = 0; i < NI; i++)
		isize[i] = 2 * md_calc_size(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);
	for (int o = 0; o < NO; o++)
		osize[o] = 2 * md_calc_size(nlop_generic_codomain(nlop, o)->N, nlop_generic_codomain(nlop, o)->dims);

	//gpu ref (dst[i] can be null if batch_gen)
	float* gpu_ref = NULL;
	for (int i = 0; i < NI; i++)
		if (IN_OPTIMIZE == in_type[i])
			gpu_ref = dst[i];
	assert(NULL != gpu_ref);


	sgd(conf->epochs, conf->batchnorm_mom,
		NI, isize, in_type, dst,
		NO, osize, out_type,
		batchsize, batchsize * numbatches,
		select_vecops(gpu_ref),
		nlop_iter, adj_op_arr,
		upd_op_arr,
		prox_iter,
		nlop_batch_gen_iter,
		(struct iter_op_s){ NULL, NULL }, monitor);

	for (int i = 0; i < NI; i++)
		operator_free(upd_ops[i][i]);
}

void iter6_adam(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct iter6_monitor_s* monitor)
{
	auto conf = CAST_DOWN(iter6_adam_conf, _conf);

	//assert(NULL == nlop_batch_gen);
	//assert(NULL == prox_ops);

	struct iter_nlop_s nlop_iter = NLOP2ITNLOP(nlop);
	struct iter_op_arr_s adj_op_arr = NLOP2IT_ADJ_ARR(nlop);
	struct iter_nlop_s nlop_batch_gen_iter = NLOP2ITNLOP(nlop_batch_gen);

	struct iter_op_p_s prox_iter[NI];
	for (unsigned int i = 0; i < NI; i++)
		prox_iter[i] = OPERATOR_P2ITOP((NULL == prox_ops ? NULL : prox_ops[i]));

	long isize[NI];
	long osize[NO];

	//array of update operators
	const struct operator_s* upd_ops[NI][NI];
	for (int i = 0; i < NI; i++) {

		for (int j = 0; j < NI; j++)
			upd_ops[i][j] = NULL;
		upd_ops[i][i] = operator_adam_update_create(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, conf->learning_rate, conf->beta1, conf->beta2, conf->epsilon);
		if ((0.0 != conf->clip_norm) || (0.0 != conf->clip_val)) {

			const struct operator_s* tmp1 = operator_clip_create(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, conf->clip_norm, conf->clip_val);
			const struct operator_s* tmp2 = upd_ops[i][i];
			upd_ops[i][i] = operator_chain(tmp1, tmp2);
			operator_free(tmp1);
			operator_free(tmp2);
		}
	}
	struct iter6_op_arr_s upd_ops_data = { { &TYPEID(iter6_op_arr_s) }, NI, NI, &(upd_ops[0][0])};
	struct iter_op_arr_s upd_op_arr ={iter6_op_arr_fun_diag, CAST_UP(&upd_ops_data)};


	for (int i = 0; i < NI; i++)
		isize[i] = 2 * md_calc_size(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);
	for (int o = 0; o < NO; o++)
		osize[o] = 2 * md_calc_size(nlop_generic_codomain(nlop, o)->N, nlop_generic_codomain(nlop, o)->dims);

	//gpu ref (dst[i] can be null if batch_gen)
	float* gpu_ref = NULL;
	for (int i = 0; i < NI; i++)
		if (IN_OPTIMIZE == in_type[i])
			gpu_ref = dst[i];
	assert(NULL != gpu_ref);


	sgd(conf->epochs, conf->batchnorm_mom,
		NI, isize, in_type, dst,
		NO, osize, out_type,
		batchsize, batchsize * numbatches,
		select_vecops(gpu_ref),
		nlop_iter, adj_op_arr,
		upd_op_arr,
		prox_iter,
		nlop_batch_gen_iter,
		(struct iter_op_s){ NULL, NULL }, monitor);

	for (int i = 0; i < NI; i++)
		operator_free(upd_ops[i][i]);
}

void iter6_iPALM(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct iter6_monitor_s* monitor)
{
	UNUSED(batchsize);

	auto conf = CAST_DOWN(iter6_iPALM_conf, _conf);

	//Compute sizes
	long isize[NI];
	long osize[NO];
	for (int i = 0; i < NI; i++)
		isize[i] = 2 * md_calc_size(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);
	for (int o = 0; o < NO; o++)
		osize[o] = 2 * md_calc_size(nlop_generic_codomain(nlop, o)->N, nlop_generic_codomain(nlop, o)->dims);

	//create iter operators
	struct iter_nlop_s nlop_iter = NLOP2ITNLOP(nlop);
	struct iter_op_arr_s adj_op_arr = NLOP2IT_ADJ_ARR(nlop);
	struct iter_nlop_s nlop_batch_gen_iter = NLOP2ITNLOP(nlop_batch_gen);

	struct iter_op_p_s prox_iter[NI];
	for (unsigned int i = 0; i < NI; i++)
		prox_iter[i] = OPERATOR_P2ITOP(prox_ops[i]);

	//compute parameter arrays
	float alpha[NI];
	float beta[NI];
	bool convex[NI];

	for (int i = 0; i < NI; i++) {

		alpha[i] = (NULL == conf->alpha_arr) ? conf->alpha : conf->alpha_arr[i];
		beta[i] = (NULL == conf->beta_arr) ? conf->beta : conf->beta_arr[i];
		convex[i] = (NULL == conf->convex_arr) ? conf->convex : conf->convex_arr[i];
	}

	//gpu ref (dst[i] can be null if batch_gen)
	float* gpu_ref = NULL;
	for (int i = 0; i < NI; i++)
		if (IN_OPTIMIZE == in_type[i])
			gpu_ref = dst[i];
	assert(NULL != gpu_ref);

	float* x_old[NI];
	for (int i = 0; i < NI; i++)
		if (IN_OPTIMIZE == in_type[i])
			x_old[i] = md_alloc_sameplace(1, isize + i, FL_SIZE, gpu_ref);
		else
			x_old[i] = NULL;

	int epoch_start = conf->epoch_start;
	int epoch_end = MIN(conf->epochs, epoch_start + conf->save_modulo);

	float lipshitz_constants[NI];
	for (int i = 0; i < NI; i++)
		lipshitz_constants[i] = conf->L;

	if (0 != epoch_start) {

		if ((NULL != conf->save_path) && (NULL != conf->save_name)) {
			for (int i = 0; i < NI; i++) {

				if(!((IN_OPTIMIZE == in_type[i]) && (NULL != conf->save_name[i])))
					continue;

				long dims[nlop_generic_domain(nlop, i)->N];
				char filename[strlen(conf->save_path) + strlen(conf->save_name[i]) + 50];

				sprintf(filename, "%s/%s-%d", conf->save_path, conf->save_name[i], epoch_start);
				complex float* file = load_cfl(filename, nlop_generic_domain(nlop, i)->N, dims);
				md_copy(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, (complex float*)dst[i], file, CFL_SIZE);
				unmap_cfl(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file);

				sprintf(filename, "%s/%s-%d", conf->save_path, conf->save_name[i], epoch_start - 1);
				file = load_cfl(filename, nlop_generic_domain(nlop, i)->N, dims);
				md_copy(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, (complex float*)x_old[i], file, CFL_SIZE);
				unmap_cfl(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file);
			}
		} else {

			error("Could not load data from epoch %d\n", epoch_start);
		}

		long dims[1];
		char filename[strlen(conf->save_path) + 70];
		sprintf(filename, "%s/%s-%d", conf->save_path, "lipshitz_constants", epoch_start);
		complex float* file = load_cfl(filename, 1, dims);
		md_copy2(1, dims, MAKE_ARRAY((long)FL_SIZE), lipshitz_constants, MAKE_ARRAY((long)CFL_SIZE), file, FL_SIZE);
		unmap_cfl(1, dims, file);
	}


	while (epoch_start < epoch_end) {

		if ((NULL != conf->save_path) && (NULL != conf->save_name)) {

			for (int i = 0; i < NI; i++)
				if((IN_OPTIMIZE == in_type[i]) && (NULL != conf->save_name[i])) {

					char filename[strlen(conf->save_path) + strlen(conf->save_name[i]) + 50];
					sprintf(filename, "%s/%s-%d", conf->save_path, conf->save_name[i], epoch_start);

					complex float* file = create_cfl(filename, nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);
					md_copy(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file, (complex float*)dst[i], CFL_SIZE);
					unmap_cfl(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file);

					//save old weights (for warm start)
					if (0 < epoch_start) {

						sprintf(filename, "%s/%s-%d", conf->save_path, conf->save_name[i], epoch_start - 1);
						file = create_cfl(filename, nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);
						md_copy(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file, (complex float*)x_old[i], CFL_SIZE);
						unmap_cfl(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file);
					}
				}

			char filename[strlen(conf->save_path) + 50];
			sprintf(filename, "%s/%s-%d", conf->save_path, "lipshitz_constants", epoch_start);
			complex float* file = create_cfl(filename, 1, MAKE_ARRAY(NI));
			md_copy2(1, MAKE_ARRAY(NI), MAKE_ARRAY((long)CFL_SIZE), file, MAKE_ARRAY((long)FL_SIZE), lipshitz_constants, FL_SIZE);
			unmap_cfl(1, MAKE_ARRAY(NI), file);
		}

		iPALM(	NI, isize, in_type, dst, x_old,
			NO, osize, out_type,
			numbatches, epoch_start, epoch_end,
        		select_vecops(gpu_ref),
			alpha, beta, convex, conf->trivial_stepsize,
			lipshitz_constants, conf->Lmin, conf->Lmax, conf->Lshrink, conf->Lincrease,
        		nlop_iter, adj_op_arr,
			prox_iter,
			nlop_batch_gen_iter,
			(struct iter_op_s){ NULL, NULL }, monitor);

		epoch_start = epoch_end;
		epoch_end = MIN(conf->epochs, epoch_start + conf->save_modulo);
	}

	for (int i = 0; i < NI; i++) {
		if((NULL != conf->save_path) && (NULL != conf->save_name) && (IN_OPTIMIZE == in_type[i]) && (NULL != conf->save_name[i])) {

			char filename[strlen(conf->save_path) + strlen(conf->save_name[i]) + 50];
			sprintf(filename, "%s/%s-%d", conf->save_path, conf->save_name[i], epoch_end);

			complex float* file = create_cfl(filename, nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);
			md_copy(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file, (complex float*)dst[i], CFL_SIZE);
			unmap_cfl(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, file);
		}

		if(IN_OPTIMIZE == in_type[i])
			md_free(x_old[i]);
	}
}