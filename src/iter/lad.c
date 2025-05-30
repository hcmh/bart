/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014      Frank Ong <frankong@berkeley.edu>
 */

#include <complex.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "lad.h"



const struct lad_conf lad_defaults = { 5, 0.1, ~0u, &lsqr_defaults };








/**
 * Perform iterative, regularized least-absolute derivation reconstruction.
 */
void lad2(	int N, const struct lad_conf* conf,
		italgo_fun2_t italgo, iter_conf* iconf,
		const struct linop_s* model_op,
		int num_funs,
		const struct operator_p_s* prox_funs[num_funs],
		const struct linop_s* prox_linops[num_funs],
		const long x_dims[static N], complex float* x,
		const long y_dims[static N], const complex float* y)
{
	long w_dims[N];
	md_select_dims(N, conf->wflags, w_dims, y_dims);

	complex float* weights = md_alloc_sameplace(N, w_dims, CFL_SIZE, y);
	complex float* tmp2 = md_alloc_sameplace(N, y_dims, CFL_SIZE, y);

	// use iterative reweigted least-squares
	// ADMM may be a better choice though...

	for (int i = 0; i < conf->rwiter; i++) {

		// recompute weights

		linop_forward(model_op, N, y_dims, tmp2, N, x_dims, x);
		md_zsub(N, y_dims, tmp2, tmp2, y);

		md_zrss(N, y_dims, ~(conf->wflags), weights, tmp2);

		for (long l = 0; l < md_calc_size(N, w_dims); l++)
			if (weights[l] != 0.)
				weights[l] = 1. / sqrtf(MAX(conf->gamma, cabsf(weights[l])));

		// solve weighted least-squares

		wlsqr2(N, conf->lsqr_conf, italgo, iconf, model_op,
				1, prox_funs, prox_linops,
		       x_dims, x, y_dims, y, w_dims, weights, NULL);
	}
		
	md_free(tmp2);
	md_free(weights);
}


/**
 * Perform iterative, regularized least-absolute derivation reconstruction.
 */
void lad(	int N, const struct lad_conf* conf,
		italgo_fun_t italgo, iter_conf* iconf,
		const struct linop_s* model_op,
		const struct operator_p_s* prox_funs,
		const long x_dims[static N], complex float* x,
		const long y_dims[static N], const complex float* y)
{
	lad2(N, conf, iter2_call_iter, CAST_UP(&((struct iter_call_s){ { &TYPEID(iter_call_s), 1. }, italgo, iconf })),
		model_op, (NULL != prox_funs) ? 1 : 0, &prox_funs, NULL,
		x_dims, x, y_dims, y);
}


struct lad_s {

	operator_data_t super;

	const struct lad_conf* conf;
	italgo_fun2_t italgo;
	iter_conf* iconf;
	const struct linop_s* model_op;
	int num_funs;
	const struct operator_p_s** prox_funs;
	const struct linop_s** prox_linops;
};

DEF_TYPEID(lad_s);

static void lad_apply(const operator_data_t* _data, float alpha, complex float* dst, const complex float* src)
{
	assert(1. == alpha);
	const auto data = CAST_DOWN(lad_s, _data);

	const struct iovec_s* dom_iov = operator_domain(data->model_op->forward);
	const struct iovec_s* cod_iov = operator_codomain(data->model_op->forward);

	lad2(dom_iov->N, data->conf, data->italgo, data->iconf, data->model_op,
		data->num_funs, data->prox_funs, data->prox_linops,
		cod_iov->dims, dst, dom_iov->dims, src);
}

static void lad_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(lad_s, _data);

	linop_free(data->model_op);

	if (NULL != data->prox_funs) {

		for (int i = 0; i < data->num_funs; i++)
			operator_p_free(data->prox_funs[i]);

		xfree(data->prox_funs);
	}

	if (NULL != data->prox_linops) {

		for (int i = 0; i < data->num_funs; i++)
			linop_free(data->prox_linops[i]);

		xfree(data->prox_linops);
	}

	xfree(data);
}

const struct operator_p_s* lad2_create(const struct lad_conf* conf,
		italgo_fun2_t italgo, iter_conf* iconf,
		const float* init,
		const struct linop_s* model_op,
		int num_funs,
		const struct operator_p_s* prox_funs[num_funs],
		const struct linop_s* prox_linops[num_funs])
{
	PTR_ALLOC(struct lad_s, data);

	const struct iovec_s* dom_iov = operator_domain(model_op->forward);
	const struct iovec_s* cod_iov = operator_codomain(model_op->forward);

	assert(cod_iov->N == dom_iov->N); // this should be relaxed

	data->conf = conf;
	data->italgo = italgo;
	data->iconf = iconf;
	data->model_op = linop_clone(model_op);
	data->num_funs = num_funs;
	data->prox_funs = *TYPE_ALLOC(const struct operator_p_s*[num_funs]);
	data->prox_linops = *TYPE_ALLOC(const struct linop_s*[num_funs]);

	assert(NULL == init);


	for (int i = 0; i < num_funs; i++) {

		data->prox_funs[i] = operator_p_ref(prox_funs[i]);
		data->prox_linops[i] = linop_clone(prox_linops[i]);
	}

	return operator_p_create(cod_iov->N, cod_iov->dims, dom_iov->N, dom_iov->dims,
				CAST_UP(PTR_PASS(data)), lad_apply, lad_del);
}



