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

#include "iter/italgos.h"
#include "iter/vec.h"

#include "iter6.h"

DEF_TYPEID(iter6_sgd_conf);
DEF_TYPEID(iter6_adadelta_conf);


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

	.clip_norm = 0.,
	.clip_val = 0.,

	.rho = 0.95
};



struct iter6_nlop_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;

	long o;
	long i;
};

DEF_TYPEID(iter6_nlop_s);


static void iter6_nlop(iter_op_data* _o, int N, float* args[N])
{
	const auto data = CAST_DOWN(iter6_nlop_s, _o);

	assert((unsigned int)N == operator_nr_args(data->nlop->op));

	nlop_generic_apply_unchecked(data->nlop, N, (void*)args);
}


static void iter6_adj(iter_op_data* _o, float* dst, const float* src)
{
	const auto data = CAST_DOWN(iter6_nlop_s, _o);

	const struct linop_s* der = nlop_get_derivative(data->nlop, data->o, data->i);
	linop_adjoint_unchecked(der, (complex float*)dst, (const complex float*)src);
}

#if 0
static void iter6_der(iter_op_data* _o, float* dst, const float* src)
{
	const auto data = CAST_DOWN(iter6_nlop_s, _o);

	const struct linop_s* der = nlop_get_derivative(data->nlop, data->o, data->i);
	linop_forward_unchecked(der, (complex float*)dst, (const complex float*)src);
}
#endif


static void iter6_getops(long NO, long osize[NO], long NI, long isize[NI], struct iter_op_s adj_ops[NO][NI], struct iter6_nlop_s adj_data[NO][NI], const struct nlop_s* nlop)
{
	assert(NI == nlop_get_nr_in_args(nlop));
	assert(NO == nlop_get_nr_out_args(nlop));

	for(long i = 0; i < NI; ++i)
		for(long o = 0; o < NO; ++o){

			adj_data[o][i] = (struct iter6_nlop_s){{ &TYPEID(iter6_nlop_s) }, nlop, o, i };
			adj_ops[o][i] = (struct iter_op_s){ iter6_adj, CAST_UP(&adj_data[o][i]) };
		}

	for (int i = 0; i < NI; i++)
		isize[i] = 2 * md_calc_size(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);
	for (int o = 0; o < NO; o++)
		osize[o] = 2 * md_calc_size(nlop_generic_codomain(nlop, o)->N, nlop_generic_codomain(nlop, o)->dims);
}


void iter6_sgd(iter6_conf* _conf, const struct nlop_s* nlop,
		long NI, enum IN_TYPE in_type[NI], float* dst[NI],
		long NO, enum OUT_TYPE out_type[NO],
		int N_batch, int N_total)
{
	auto conf = CAST_DOWN(iter6_sgd_conf, _conf);

	struct iter6_nlop_s nlop_data = { { &TYPEID(iter6_nlop_s) }, nlop, -1, -1};
	struct iter6_nlop_s adj_data[NO][NI];
	struct iter_op_s adj_ops[NO][NI];

	long isize[NI];
	long osize[NO];

	iter6_getops(NO, osize, NI, isize, adj_ops, adj_data, nlop);

	sgd(conf->epochs, conf->clip_norm, conf->clip_val,
		conf->learning_rate, conf->momentum,
		NI, isize, in_type, dst,
		NO, osize, out_type,
		N_batch, N_total,
		select_vecops(dst[0]),
		(struct iter_nlop_s){iter6_nlop, CAST_UP(&nlop_data) }, adj_ops,
		(struct iter_op_s){ NULL, NULL }, NULL);

}


void iter6_adadelta(iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int N_batch, int N_total)
{
	auto conf = CAST_DOWN(iter6_adadelta_conf, _conf);

	struct iter6_nlop_s nlop_data = { { &TYPEID(iter6_nlop_s) }, nlop, -1, -1};
	struct iter6_nlop_s adj_data[NO][NI];
	struct iter_op_s adj_ops[NO][NI];

	long isize[NI];
	long osize[NO];

	iter6_getops(NO, osize, NI, isize, adj_ops, adj_data, nlop);

	adadelta(conf->epochs, conf->clip_norm, conf->clip_val,
			conf->learning_rate, conf->rho,
			NI, isize, in_type, dst,
			NO, osize, out_type,
			N_batch, N_total,
			select_vecops(dst[0]),
			(struct iter_nlop_s){iter6_nlop, CAST_UP(&nlop_data) }, adj_ops,
			(struct iter_op_s){ NULL, NULL }, NULL);
}



