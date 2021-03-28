/* Copyright 2017-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/opts.h"

#include "iter/batch_gen.h"
#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter2.h"
#include "iter/iter6_ops.h"
#include "iter/monitor_iter6.h"
#include "iter/iter_dump.h"

#include "iter6.h"

#ifndef STRUCT_TMP_COPY
#define STRUCT_TMP_COPY(x) ({ __typeof(x) __foo = (x); __typeof(__foo)* __foo2 = alloca(sizeof(__foo)); *__foo2 = __foo; __foo2; })
#endif
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

#define ITER6_DEFAULT \
	.INTERFACE.epochs = 1, \
	.INTERFACE.clip_norm = 0., \
	.INTERFACE.clip_val = 0., \
	.INTERFACE.history_filename = NULL, \
	.INTERFACE.dump = NULL, \
	.INTERFACE.dump_filename = NULL, \
	.INTERFACE.dump_mod = -1, \
	.INTERFACE.batchnorm_momentum = .95, \
	.INTERFACE.batchgen_type = BATCH_GEN_SAME, \
	.INTERFACE.batch_seed = 123, \
	.INTERFACE.dump_flag = 0,

const struct iter6_sgd_conf iter6_sgd_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_sgd_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 0.001,

	.momentum = 0.
};


const struct iter6_adadelta_conf iter6_adadelta_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adadelta_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 1.,

	.rho = 0.95
};

const struct iter6_adam_conf iter6_adam_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adam_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = .001,

	.reset_epoch = -1,

	.epsilon = 1.e-7,

	.beta1 = 0.9,
	.beta2 = 0.999,
};


const struct iter6_iPALM_conf iter6_iPALM_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_iPALM_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 1.,

	.Lmin = 1.e-10,
	.Lmax = 1.e10,
	.Lshrink = 1.2,
	.Lincrease = 2.,

	.alpha = -1.,
	.beta = -1.,
	.convex = false,

	.trivial_stepsize = false,

	.alpha_arr = NULL,
	.beta_arr =NULL,
	.convex_arr = NULL,

	.reduce_momentum = true,

};


struct iter6_conf_s iter6_conf_unset = {

	.learning_rate = 0.,
	.epochs = -1,
	.clip_norm = 0.,
	.clip_val = 0.,
	.history_filename = NULL,
	.dump = NULL,
	.dump_filename = NULL,
	.dump_mod = -1,
	.batchnorm_momentum = .95,
	.batchgen_type = BATCH_GEN_SAME,
	.batch_seed = 123,
	.dump_flag = 0,
};

struct iter6_conf_s iter6_conf_opts = {

	.learning_rate = 0.,
	.epochs = -1,
	.clip_norm = 0.,
	.clip_val = 0.,
	.history_filename = NULL,
	.dump = NULL,
	.dump_filename = NULL,
	.dump_mod = -1,
	.batchnorm_momentum = .95,
	.batchgen_type = BATCH_GEN_SAME,
	.batch_seed = 123,
	.dump_flag = 0,
};

struct iter6_sgd_conf iter6_sgd_conf_opts = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_sgd_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 0.001,

	.momentum = 0.
};

struct iter6_adadelta_conf iter6_adadelta_conf_opts = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adadelta_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 1.,

	.rho = 0.95
};

struct iter6_adam_conf iter6_adam_conf_opts = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adam_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = .001,

	.reset_epoch = -1,

	.epsilon = 1.e-7,

	.beta1 = 0.9,
	.beta2 = 0.999,
};

struct iter6_iPALM_conf iter6_iPALM_conf_opts = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_iPALM_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 1.,

	.Lmin = 1.e-10,
	.Lmax = 1.e10,
	.Lshrink = 1.2,
	.Lincrease = 2.,

	.alpha = -1.,
	.beta = -1.,
	.convex = false,

	.trivial_stepsize = false,

	.alpha_arr = NULL,
	.beta_arr =NULL,
	.convex_arr = NULL,

	.reduce_momentum = true,

};

enum ITER6_TRAIN_ALGORITHM {ITER6_NONE, ITER6_SGD, ITER6_ADAM, ITER6_ADADELTA, ITER6_IPALM};
static enum ITER6_TRAIN_ALGORITHM iter_6_select_algo = ITER6_NONE;

struct opt_s iter6_opts[] = {

	OPTL_FLOAT(0, "learning-rate", &(iter6_conf_opts.learning_rate), "float", "learning rate"),
	OPTL_INT('e', "epochs", &(iter6_conf_opts.epochs), "int", "number of epochs to train"),

	OPTL_SELECT_DEF(0, "sgd", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_SGD, ITER6_NONE, "select stochastic gradient descent"),
	OPTL_SELECT_DEF(0, "adadelta", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_ADADELTA, ITER6_NONE, "select AdaDelta"),
	OPTL_SELECT_DEF(0, "adam", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_ADAM, ITER6_NONE, "select Adam"),
	OPTL_SELECT_DEF(0, "ipalm", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_IPALM, ITER6_NONE, "select iPALM"),

	OPTL_FLOAT(0, "clip-norm", &(iter6_conf_opts.clip_norm), "float", "clip norm of gradients"),
	OPTL_FLOAT(0, "clip-value", &(iter6_conf_opts.clip_val), "float", "clip value of gradients"),

	OPTL_STRING(0, "dump-filename", &(iter6_conf_opts.dump_filename), "name", "dump weights to file"),
	OPTL_LONG(0, "dump-mod", &(iter6_conf_opts.dump_mod), "int", "dump weights to file every \"mod\" epochs"),

	OPTL_FLOAT(0, "batchnorm-momentum", &(iter6_conf_opts.batchnorm_momentum), "float", "momentum for bastch normalization (default: 0.95)"),

	OPTL_SELECT_DEF(0, "batch-generator-same", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_SAME, BATCH_GEN_SAME, "use the same batches in the same order for each epoch"),
	OPTL_SELECT_DEF(0, "batch-generator-shuffel-batches", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_SHUFFLE_BATCHES, BATCH_GEN_SAME, "use the same batches in random order for each epoch"),
	OPTL_SELECT_DEF(0, "batch-generator-shuffel-data", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_SHUFFLE_DATA, BATCH_GEN_SAME, "shuffle data to form batches"),
	OPTL_SELECT_DEF(0, "batch-generator-draw-data", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_SHUFFLE_BATCHES, BATCH_GEN_SAME, "randomly draw data to form batches"),
	OPTL_INT(0, "batch-generator-seed", &(iter6_conf_opts.batch_seed), "int", "seed for batch-generator (default: 123)"),
};

struct opt_s iter6_sgd_opts[] = {

	OPTL_FLOAT(0, "momentum", &(iter6_sgd_conf_opts.momentum), "float", "momentum (default: 0.)"),
};
const int N_iter6_sgd_opts = ARRAY_SIZE(iter6_sgd_opts);

struct opt_s iter6_adadelta_opts[] = {

	OPTL_FLOAT(0, "rho", &(iter6_adadelta_conf_opts.rho), "float", "rho (default: 0.95"),
};

struct opt_s iter6_adam_opts[] = {

	OPTL_FLOAT(0, "epsilon", &(iter6_adam_conf_opts.epsilon), "float", "epsilon (default: 1.e-7"),
	OPTL_FLOAT(0, "beta1", &(iter6_adam_conf_opts.beta1), "float", "beta1 (default: 0.9"),
	OPTL_FLOAT(0, "beta2", &(iter6_adam_conf_opts.beta2), "float", "beta2 (default: 0.999"),

	OPTL_LONG(0, "reset-momentum", &(iter6_adam_conf_opts.reset_epoch), "n", "reset momentum every nth epoch (default: -1=never"),
};

struct opt_s iter6_ipalm_opts[] = {

	OPTL_FLOAT(0, "lipshitz-min", &(iter6_iPALM_conf_opts.Lmin), "float", "minimum Lipshitz constant for backtracking (default: 1.e-10"),
	OPTL_FLOAT(0, "lipshitz-max", &(iter6_iPALM_conf_opts.Lmax), "float", "maximum Lipshitz constant for backtracking (default: 1.e10"),
	OPTL_FLOAT(0, "lipshitz-reduce", &(iter6_iPALM_conf_opts.Lshrink), "float", "factor toi reduce Lipshitz constant in backtracking (default: 1.2"),
	OPTL_FLOAT(0, "lipshitz-increase", &(iter6_iPALM_conf_opts.Lincrease), "float", "factor to increase Lipshitz constant in backtracking (default: 2"),

	OPTL_FLOAT(0, "alpha", &(iter6_iPALM_conf_opts.alpha), "float", "alpha factor (default: -1. = \"dynamic case\")"),
	OPTL_FLOAT(0, "beta", &(iter6_iPALM_conf_opts.beta), "float", "beta factor (default: -1. = \"dynamic case\")"),
	OPTL_SET(0, "convex", &(iter6_iPALM_conf_opts.convex), "convex constraints (higher learning rate possible)"),

	OPTL_CLEAR(0, "non-trivial-step-size", &(iter6_iPALM_conf_opts.convex), "set stepsize based on alpha and beta, not simply Lipshitz constant^-1"),
	OPTL_CLEAR(0, "no-momentum-reduction", &(iter6_iPALM_conf_opts.reduce_momentum), "momentum is not reduced, when Lipshitz condition is not satisfied while backtracking"),
};

const int N_iter6_opts = ARRAY_SIZE(iter6_opts);
const int N_iter6_adadelta_opts = ARRAY_SIZE(iter6_adadelta_opts);
const int N_iter6_adam_opts = ARRAY_SIZE(iter6_adam_opts);
const int N_iter6_ipalm_opts = ARRAY_SIZE(iter6_ipalm_opts);

void iter6_copy_config_from_opts(struct iter6_conf_s* result)
{
	if (iter6_conf_opts.learning_rate != iter6_conf_unset.learning_rate)
		result->learning_rate = iter6_conf_opts.learning_rate;
	if (iter6_conf_opts.epochs != iter6_conf_unset.epochs)
		result->epochs = iter6_conf_opts.epochs;
	if (iter6_conf_opts.clip_norm != iter6_conf_unset.clip_norm)
		result->clip_norm = iter6_conf_opts.clip_norm;
	if (iter6_conf_opts.clip_val != iter6_conf_unset.clip_val)
		result->clip_val = iter6_conf_opts.clip_val;
	if (iter6_conf_opts.history_filename != iter6_conf_unset.history_filename)
		result->history_filename = iter6_conf_opts.history_filename;
	if (iter6_conf_opts.dump_filename != iter6_conf_unset.dump_filename)
		result->dump_filename = iter6_conf_opts.dump_filename;
	if (iter6_conf_opts.dump_mod != iter6_conf_unset.dump_mod)
		result->dump_mod = iter6_conf_opts.dump_mod;
	if (iter6_conf_opts.batchnorm_momentum != iter6_conf_unset.batchnorm_momentum)
		result->batchnorm_momentum = iter6_conf_opts.batchnorm_momentum;
	if (iter6_conf_opts.batchgen_type != iter6_conf_unset.batchgen_type)
		result->batchgen_type = iter6_conf_opts.batchgen_type;
	if (iter6_conf_opts.batch_seed != iter6_conf_unset.batch_seed)
		result->batch_seed = iter6_conf_opts.batch_seed;
}

struct iter6_conf_s* iter6_get_conf_from_opts(void)
{
	struct iter6_conf_s* result = NULL;

	switch (iter_6_select_algo) {

		case ITER6_NONE:
			return result;
			break;

		case ITER6_SGD:
			result = CAST_UP(&iter6_sgd_conf_opts);
			break;

		case ITER6_ADAM:
			result = CAST_UP(&iter6_adam_conf_opts);
			break;

		case ITER6_ADADELTA:
			result = CAST_UP(&iter6_adam_conf_opts);
			break;

		case ITER6_IPALM:
			result = CAST_UP(&iter6_adam_conf_opts);
			break;
	}

	iter6_copy_config_from_opts(result);

	return result;
}

struct iter6_nlop_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;
};

DEF_TYPEID(iter6_nlop_s);

static void iter6_nlop(iter_op_data* _o, int N, float* args[N], unsigned long der_out, unsigned long der_in)
{
	const auto data = CAST_DOWN(iter6_nlop_s, _o);

	assert((unsigned int)N == operator_nr_args(data->nlop->op));

	nlop_generic_apply_select_derivative_unchecked(data->nlop, N, (void*)args, der_out, der_in);
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

static const struct iter_dump_s* iter6_dump_default_create(const char* base_filename, long save_mod, const struct nlop_s* nlop, unsigned long save_flag, long NI, enum IN_TYPE in_type[NI])
{

	unsigned int D[NI];
	const long* dims[NI];

	bool guess_save_flag = (0 == save_flag);

	for (int i = 0; i < NI; i++) {

		D[i] = nlop_generic_domain(nlop, i)->N;
		dims[i] = nlop_generic_domain(nlop, i)->dims;
		if ((guess_save_flag) && ((IN_OPTIMIZE == in_type[i]) || (IN_BATCHNORM == in_type[i])))
			save_flag = MD_SET(save_flag, i);
	}

	return iter_dump_default_create(base_filename, save_mod, NI, save_flag, D, dims);
}

static const struct operator_s* get_update_operator(iter6_conf* conf, int N, const long dims[N], long numbatches)
{
	auto conf_adadelta = CAST_MAYBE(iter6_adadelta_conf, conf);
	if (NULL != conf_adadelta)
		return operator_adadelta_update_create(N, dims, conf->learning_rate, conf_adadelta->rho, 1.e-7);

	auto conf_sgd = CAST_MAYBE(iter6_sgd_conf, conf);
	if (NULL != conf_sgd)
		return operator_sgd_update_create(N, dims, conf->learning_rate);

	auto conf_adam = CAST_MAYBE(iter6_adam_conf, conf);
	if (NULL != conf_adam)
		return operator_adam_update_create(N, dims, conf->learning_rate, conf_adam->beta1, conf_adam->beta2, conf_adam->epsilon, numbatches * conf_adam->reset_epoch);

	error("iter6_conf not SGD-like!\n");
	return NULL;
}

void iter6_sgd_like(	iter6_conf* conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
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

		upd_ops[i][i] = get_update_operator(conf, nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, numbatches);

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

	bool free_monitor = (NULL == monitor);
	if (free_monitor)
		monitor = monitor_iter6_create(true, false, 0, NULL);

	bool free_dump = ((NULL == conf->dump) && (NULL != conf->dump_filename) && (0 < conf->dump_mod));
	if (free_dump)
		conf->dump = iter6_dump_default_create(conf->dump_filename, conf->dump_mod, nlop, conf->dump_flag, NI, in_type);

	sgd(	conf->epochs, conf->batchnorm_momentum,
		NI, isize, in_type, dst,
		NO, osize, out_type,
		batchsize, batchsize * numbatches,
		select_vecops(gpu_ref),
		nlop_iter, adj_op_arr,
		upd_op_arr,
		prox_iter,
		nlop_batch_gen_iter,
		(struct iter_op_s){ NULL, NULL }, monitor, conf->dump);

	for (int i = 0; i < NI; i++)
		operator_free(upd_ops[i][i]);

	if (NULL != conf->history_filename)
		monitor_iter6_dump_record(monitor, conf->history_filename);

	if (free_monitor)
		monitor_iter6_free(monitor);

	if (free_dump) {
		iter_dump_free(conf->dump);
		conf->dump = NULL;
	}
}


void iter6_adadelta(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_adadelta_conf, _conf);
	assert(NULL != conf);

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}

void iter6_adam(iter6_conf* _conf,
		const struct nlop_s* nlop,
		long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
		long NO, enum OUT_TYPE out_type[NO],
		int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_adam_conf, _conf);
	assert(NULL != conf);

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}

void iter6_sgd(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_sgd_conf, _conf);
	assert(NULL != conf);

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}

void iter6_iPALM(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
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


	float lipshitz_constants[NI];
	for (int i = 0; i < NI; i++)
		lipshitz_constants[i] = 1. / conf->INTERFACE.learning_rate;

	bool free_monitor = (NULL == monitor);
	if (free_monitor)
		monitor = monitor_iter6_create(true, false, 0, NULL);

	bool free_dump = ((NULL == conf->INTERFACE.dump) && (NULL != conf->INTERFACE.dump_filename) && (0 < conf->INTERFACE.dump_mod));
	if (free_dump)
		conf->INTERFACE.dump = iter6_dump_default_create(conf->INTERFACE.dump_filename,conf->INTERFACE.dump_mod, nlop, conf->INTERFACE.dump_flag, NI, in_type);

	iPALM(	NI, isize, in_type, dst, x_old,
		NO, osize, out_type,
		numbatches, 0, conf->INTERFACE.epochs,
       		select_vecops(gpu_ref),
		alpha, beta, convex, conf->trivial_stepsize, conf->reduce_momentum,
		lipshitz_constants, conf->Lmin, conf->Lmax, conf->Lshrink, conf->Lincrease,
       		nlop_iter, adj_op_arr,
		prox_iter,
		conf->INTERFACE.batchnorm_momentum,
		nlop_batch_gen_iter,
		(struct iter_op_s){ NULL, NULL }, monitor, conf->INTERFACE.dump);

	if (NULL != conf->INTERFACE.history_filename)
		monitor_iter6_dump_record(monitor, conf->INTERFACE.history_filename);

	if (free_monitor)
		monitor_iter6_free(monitor);

	for (int i = 0; i < NI; i++)
		if(IN_OPTIMIZE == in_type[i])
			md_free(x_old[i]);

	if (free_dump) {
		iter_dump_free(conf->INTERFACE.dump);
		conf->INTERFACE.dump = NULL;
	}
}

void iter6_by_conf(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_iPALM_conf, _conf);

	if (NULL != conf) {

		iter6_iPALM(	_conf,
				nlop,
				NI, in_type, prox_ops, dst,
				NO, out_type,
				batchsize, numbatches, nlop_batch_gen, monitor);
		return;
	}

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}
