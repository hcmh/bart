/* Copyright 2017. Martin Uecker.
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
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter3.h"

#include "iter/iter4.h"

struct iter4_nlop_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;
};
DEF_TYPEID(iter4_nlop_s);

struct iter4_altmin_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;

	struct iter3_irgnm_conf conf;
};

DEF_TYPEID(iter4_altmin_s);

static void nlop_for_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	operator_apply_unchecked(nlop->nlop.op, (complex float*)_dst, (const complex float*)_src);
}

static void nlop_der_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_forward_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}

static void nlop_adj_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_adjoint_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}




void iter4_irgnm(iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	iter3_irgnm(_conf,
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		N, dst, ref, M, src, cb);
}

void iter4_levmar(iter3_conf* _conf,
		 struct nlop_s* nlop,
		 long N, float* dst, const float* ref,
		 long M, const float* src,
		 struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	iter3_levmar(_conf,
		    (struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		    N, dst, ref, M, src, cb);
}

void iter4_irgnm_levmar_hybrid(iter3_conf* _conf,
		 struct nlop_s* nlop,
		 long N, float* dst, const float* ref,
		 long M, const float* src,
		 struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	iter3_irgnm_levmar_hybrid(_conf,
		    (struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		    N, dst, ref, M, src, cb);
}



static void altmin_nlop(iter_op_data* _o, int N, float* args[N])
{
	const struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);

	assert((unsigned int) N == operator_nr_args(nlop->nlop.op));

	nlop_generic_apply_unchecked(&nlop->nlop, N, (void*) args);
}




static void altmin_normal(iter_op_data* _o, float* dst, const float* src, unsigned int i)
{
	const struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);
	const struct linop_s* der = nlop_get_derivative(&nlop->nlop, 0, i);

	linop_normal_unchecked(der, (complex float*) dst, (const complex float*) src);
}

static void altmin_normal_coils(iter_op_data* _o, float* dst, const float* src)
{
	altmin_normal(_o, dst, src, 0);
}

static void altmin_normal_img(iter_op_data* _o, float* dst, const float* src)
{
	altmin_normal(_o, dst, src, 1);
}


static void altmin_inverse(iter_op_data* _o, float alpha, float* dst, const float* src, unsigned int i)
{
	const struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);

	long DIMS = 16;
	long size = 2*md_calc_size(DIMS, nlop_generic_domain(&nlop->nlop, i)->dims);

	float* AHy = md_alloc_sameplace(1, MD_DIMS(size), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(&nlop->nlop, 0, i), (complex float*) AHy, (const complex float*) src);

	float eps = nlop->conf.cgtol * md_norm(DIMS, nlop_generic_domain(&nlop->nlop, i)->dims, AHy);

	if (1 == i)
		conjgrad(nlop->conf.cgiter, alpha, eps, size, select_vecops(src),
			 (struct iter_op_s){ altmin_normal_img, _o }, dst, AHy, NULL);
	else
		conjgrad(nlop->conf.cgiter, alpha, eps, size, select_vecops(src),
			 (struct iter_op_s){ altmin_normal_coils, _o }, dst, AHy, NULL);

	md_free(AHy);

}


static void altmin_inverse_coils(iter_op_data* _o, float alpha, float* dst, const float* src)
{
	debug_printf(DP_DEBUG2, "\t%s\n", __func__);
	altmin_inverse(_o, alpha, dst, src, 0);
}


static void altmin_inverse_img(iter_op_data* _o, float alpha, float* dst, const float* src)
{
	debug_printf(DP_DEBUG2, "\t%s\n", __func__);
	altmin_inverse(_o, alpha, dst, src, 1);
}

void iter4_altmin(iter3_conf* _conf,
		struct nlop_s* nlop,
		long NI, float* dst[NI],
		long M, const float* src,
		struct iter_nlop_s cb)
{

	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);
	struct iter4_altmin_s data = { { &TYPEID(iter4_altmin_s) }, *nlop, *conf};

	assert( 2 == NI);

	struct iter_op_p_s min_ops[2] = {
		(struct iter_op_p_s){ altmin_inverse_coils, CAST_UP(&data) },
		(struct iter_op_p_s){ altmin_inverse_img, CAST_UP(&data) }
	};

	altmin(conf->iter, conf->alpha, conf->redu,
		M, select_vecops(src),
		NI,
		(struct iter_nlop_s){ altmin_nlop, CAST_UP(&data) },
		min_ops,
		dst, src,
		cb);
}

void iter4_landweber(iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	iter3_landweber(_conf,
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		N, dst, ref, M, src);
}



