/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/rand.h"

#include "num/ops.h"
#include "num/iovec.h"

#include "wavelet/wavthresh.h"

#include "nlops/nlop.h"

#include "iter/prox.h"
#include "iter/vec.h"
#include "iter/italgos.h"
#include "iter/iter3.h"

#include "iterT1.h"



struct irgnm_l1_s {

	INTERFACE(iter_op_data);

	struct iter_op_s frw;
	struct iter_op_s der;
	struct iter_op_s adj;
    
	long size_x;
	long size_y;

	float alpha;
    
	float alpha_min;
	const long* dims;

	bool first_iter;
	int outer_iter;

	const struct operator_p_s* prox1;
	const struct operator_p_s* prox2;
};

DEF_TYPEID(irgnm_l1_s);




static void normal_fista(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_l1_s* data = CAST_DOWN(irgnm_l1_s, _data);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	iter_op_call(data->der, tmp, src);
	iter_op_call(data->adj, dst, tmp);

	md_free(tmp);

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];
	long coils = data->dims[COIL_DIM];

	md_axpy(1, MD_DIMS(data->size_x * coils / (coils + parameters)),
						 dst + res * res * 2 * parameters,
						 data->alpha,
						 src + res * res * 2 * parameters);
}

static void pos_value(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_l1_s* data = CAST_DOWN(irgnm_l1_s, _data);

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];

	long dims1[DIMS];

	md_select_dims(DIMS, FFT_FLAGS, dims1, data->dims);

	md_zsmax(DIMS, dims1, (_Complex float*)dst + (parameters - 1) * res * res,
			(const _Complex float*)src + (parameters - 1) * res * res, 0.1);
}



static void combined_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	struct irgnm_l1_s* data = CAST_DOWN(irgnm_l1_s, _data);

	if (data->first_iter) {

		data->first_iter = false;

	} else {

		pos_value(_data, dst, src);
	}

	operator_p_apply_unchecked(data->prox2, rho, (_Complex float*)dst, (const _Complex float*)dst);

	pos_value(_data, dst, dst);
}



static void inverse_fista(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	struct irgnm_l1_s* data = CAST_DOWN(irgnm_l1_s, _data);

	assert(alpha >= data->alpha_min);
	data->alpha = alpha;	// update alpha for normal operator

    
	void* x = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);
	double maxeigen = power(20, data->size_x, select_vecops(src), (struct iter_op_s){ normal_fista, CAST_UP(data) }, x);
	md_free(x);

	double step = 0.475 / maxeigen;

	debug_printf(DP_DEBUG3, "##reg. alpha = %f\n", alpha);

	wavthresh_rand_state_set(data->prox1, 1);
    
	int maxiter = MIN(250, 10 * powf(2, data->outer_iter));
    
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	iter_op_call(data->adj, tmp, src);

	float eps = md_norm(1, MD_DIMS(data->size_x), tmp);

	data->first_iter = true;

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = data->alpha;
	};

	fista(maxiter, 0.01f * alpha * eps, step,
		data->size_x,
		select_vecops(src),
		continuation,
		(struct iter_op_s){ normal_fista, CAST_UP(data) },
		(struct iter_op_p_s){ combined_prox, CAST_UP(data) },
		dst, tmp, NULL);

	pos_value(CAST_UP(data), dst, dst);

	md_free(tmp);

	data->outer_iter++;
}


static const struct operator_p_s* create_prox(const long img_dims[DIMS])
{
	bool randshift = true;
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	unsigned int wflags = 0;

	for (unsigned int i = 0; i < DIMS; i++) {

		if ((1 < img_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {

			wflags = MD_SET(wflags, i);
			minsize[i] = MIN(img_dims[i], DIMS);
		}
	}

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, COEFF_FLAG, minsize, 1., randshift);
}


static void iter3_irgnm_l1(const iter3_conf* _conf,
		const long dims[],
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);
	debug_print_dims(DP_INFO, DIMS, img_dims);

	auto prox1 = create_prox(img_dims);
	auto prox2 = op_p_auto_normalize(prox1, ~COEFF_FLAG);

	struct irgnm_l1_s data = {

		{ &TYPEID(irgnm_l1_s) }, frw, der, adj, N, M,
		1.0, conf->alpha_min,
		dims, true, 0, prox1, prox2
	};

	assert(NULL == ref);

	irgnm2(conf->iter, conf->alpha, 0., conf->alpha_min, conf->redu, N, M, select_vecops(src),
		frw,
		der,
		(struct iter_op_p_s){ inverse_fista, CAST_UP(&data) },
		dst, ref, src,
		(struct iter_op_s){ NULL, NULL },
		NULL);

	operator_p_free(prox1);
	operator_p_free(prox2);
}

struct iterT1_nlop_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;
};

DEF_TYPEID(iterT1_nlop_s);


static void nlop_for_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iterT1_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	operator_apply_unchecked(nlop->nlop.op, (complex float*)_dst, (const complex float*)_src);
}

static void nlop_der_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iterT1_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_forward_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}

static void nlop_adj_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const auto nlop = CAST_DOWN(iterT1_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_adjoint_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}

void iter4_irgnm_l1(const iter3_conf* _conf,
	const long dims[],
	struct nlop_s* nlop,
	long N, float* dst, const float* ref,
	long M, const float* src,
	const struct operator_p_s* pinv,
	struct iter_op_s cb)
{
	struct iterT1_nlop_s data = { { &TYPEID(iterT1_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	(void)cb;
	assert(NULL == pinv);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	iter3_irgnm_l1(_conf,
		dims,
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		N, dst, ref, M, src);
}


