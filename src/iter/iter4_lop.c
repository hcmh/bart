/* Copyright 2017-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter3.h"
#include "iter/iter2.h"

#include "iter4_lop.h"


struct irgnm_lop_s {

	INTERFACE(iter_op_data);

	struct iter_op_s adj;
	struct iter_op_s normal;

	long size;

	int cgiter;
	float cgtol;
	bool nlinv_legacy;
};

DEF_TYPEID(irgnm_lop_s);

static void normal(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_lop_s, _data);

	iter_op_call(data->normal, dst, src);
}

static void inverse(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_lop_s, _data);

	md_clear(1, MD_DIMS(data->size), dst, FL_SIZE);

        float eps = data->cgtol * md_norm(1, MD_DIMS(data->size), src);


	/* The original (Matlab) nlinv implementation uses
	 * "sqrt(rsnew) < 0.01 * rsnot" as termination condition.
	 */
	if (data->nlinv_legacy)
		eps = powf(eps, 2.);

        conjgrad(data->cgiter, alpha, eps, data->size, select_vecops(src),
			(struct iter_op_s){ normal, CAST_UP(data) }, dst, src, NULL);
}


void iter4_lop_irgnm(const iter3_conf* _conf,
		struct nlop_s* nlop,
		struct linop_s* lop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* pinv,
		struct iter_op_s cb)
{
	auto cd = linop_domain(lop);
	auto dm = nlop_domain(nlop);

	assert(NULL == pinv); // better we allow this only with irgnm2

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	auto conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	const struct operator_s* op_frw = operator_chain(nlop->op, lop->normal);
	const struct operator_s* op_der = operator_chain(nlop_get_derivative(nlop, 0, 0)->forward, lop->normal);
	const struct operator_s* op_adj = operator_ref(nlop_get_derivative(nlop, 0, 0)->adjoint);
	const struct operator_s* op_normal = operator_chain(op_der, op_adj);

	struct iter_op_s frw = OPERATOR2ITOP(op_frw);
	//struct iter_op_s der = OPERATOR2ITOP(op_der);
	struct iter_op_s adj = OPERATOR2ITOP(op_adj);
	struct iter_op_s normal = OPERATOR2ITOP(op_normal);

	struct irgnm_lop_s data2 = { { &TYPEID(irgnm_lop_s) }, adj, normal, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	struct iter_op_p_s inv = { inverse, CAST_UP(&data2) };

	irgnm(conf->iter, conf->alpha, conf->alpha_min, conf->redu, conf->nr_init, N, M, select_vecops(src),
		frw, adj, inv,
		dst, ref, src, cb, NULL);

	operator_free(op_frw);
	operator_free(op_der);
	operator_free(op_adj);
	operator_free(op_normal);
}




static void inverse2(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(irgnm_lop_s, _data);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size), FL_SIZE, src);

	iter_op_call(data->adj, tmp, src);

	inverse(_data, alpha, dst, tmp);

	md_free(tmp);
}



void iter4_lop_irgnm2(const iter3_conf* _conf,
		struct nlop_s* nlop,
		struct linop_s* lop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* lsqr,
		struct iter_op_s cb)
{
	auto cd = linop_domain(lop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	auto conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	const struct operator_s* op_frw = operator_chain(nlop->op, lop->normal);
	const struct operator_s* op_der = operator_chain(nlop_get_derivative(nlop, 0, 0)->forward, lop->normal);
	const struct operator_s* op_adj = operator_ref(nlop_get_derivative(nlop, 0, 0)->adjoint);
	const struct operator_s* op_normal = operator_chain(op_der, op_adj);

	struct iter_op_s frw = OPERATOR2ITOP(op_frw);
	struct iter_op_s der = OPERATOR2ITOP(op_der);
	struct iter_op_s adj = OPERATOR2ITOP(op_adj);
	struct iter_op_s normal = OPERATOR2ITOP(op_normal);


	struct irgnm_lop_s data2 = { { &TYPEID(irgnm_lop_s) }, adj, normal, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	// one limitation is that we currently cannot warm start the inner solver

	struct iter_op_p_s inv2 = { inverse2, CAST_UP(&data2) };

	irgnm2(conf->iter, conf->alpha, conf->alpha_min, conf->alpha_min0, conf->redu, conf->nr_init, N, M, select_vecops(src),
		frw, der, (NULL == lsqr) ? inv2 : OPERATOR_P2ITOP(lsqr),
		dst, ref, src, cb, NULL);

	operator_free(op_frw);
	operator_free(op_der);
	operator_free(op_adj);
	operator_free(op_normal);
}


