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

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/ops.h"
#include "num/ops_p.h"

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "iter/italgos.h"
#include "iter/italgos_xw.h"
#include "iter/vec.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "wavelet/wavthresh.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"


int k_fista = 0;

DEF_TYPEID(iter3_irgnm_conf);
DEF_TYPEID(iter3_landweber_conf);

const struct iter3_irgnm_conf iter3_irgnm_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter3_irgnm_conf),

	.iter = 8,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,

	.cgiter = 100,
	.cgtol = 0.1,

	.nlinv_legacy = false,
};

const struct iter3_landweber_conf iter3_landweber_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter3_landweber_conf),

	.iter = 8,
	.alpha = 1.,
	.epsilon = 0.1,
};



struct irgnm_l1_s {

	INTERFACE(iter_op_data);

	struct iter_op_s frw;
	struct iter_op_s der;
	struct iter_op_s adj;
    
	long size_x;
	long size_y;

	int cgiter;
	float cgtol;
	bool nlinv_legacy;
	float alpha;
    
	float alpha_min;
	long* dims;

	const struct operator_p_s* prox;
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
   
	select_vecops(src)->axpy(data->size_x * coils / (coils + parameters),
						 dst + res * res * 2 * parameters,
						 data->alpha,
						 src + res * res * 2 * parameters);
}



static void combined_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	struct irgnm_l1_s* data = CAST_DOWN(irgnm_l1_s, _data);

	operator_p_apply_unchecked(data->prox, rho, (_Complex float*)dst, (const _Complex float*)src);
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
//	debug_printf(DP_INFO, "\tMax eigv: %.2e\n", maxeigen);

	double step = 0.475/maxeigen;//fmin(iter_fista_defaults.step / maxeigen, iter_fista_defaults.step); // 0.95f is FISTA standard
//	debug_printf(DP_INFO, "\tFISTA Stepsize: %.2e\n", step);
//     float alpha_min = data->alpha_min;
	long img_dims[16];
	md_select_dims(16, ~COIL_FLAG, img_dims, data->dims);
	debug_print_dims(DP_INFO, DIMS, img_dims);

	if (1) {
		// for FISTA+
		bool randshift = true;
		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		unsigned int wflags = 0;

		for (unsigned int i = 0; i < DIMS; i++) {

			if ((1 < img_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {

				wflags = MD_SET(wflags, i);
				minsize[i] = MIN(img_dims[i], 16);
			}
		}

		auto prox = prox_wavelet_thresh_create(DIMS, img_dims, wflags, COEFF_FLAG, minsize, alpha, randshift);
		data->prox = op_p_auto_normalize(prox, ~COEFF_FLAG);
		operator_p_free(prox);
	}
    
	debug_printf(DP_DEBUG3, "##reg. alpha = %f\n", alpha);
    
	int maxiter = 10*powf(2, k_fista);
    
	if(maxiter > 250)
		maxiter = 250;

	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	iter_op_call(data->adj, tmp, src);

	float eps = md_norm(1, MD_DIMS(data->size_x), tmp);

	fista_xw(maxiter, 0.01f * alpha * eps, step, data->dims,
		iter_fista_defaults.continuation, iter_fista_defaults.hogwild,
		data->size_x,
		select_vecops(src),
		(struct iter_op_s){ normal_fista, CAST_UP(data) },
		(struct iter_op_p_s){ combined_prox, CAST_UP(data) },
		dst, tmp, NULL);

	md_free(tmp);

	k_fista++;
}



void iter3_irgnm_l1(const iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	struct irgnm_l1_s data = { { &TYPEID(irgnm_l1_s) }, frw, der, adj, N, M, conf->cgiter, conf->cgtol, conf->nlinv_legacy, 1.0, conf->alpha_min, conf->dims, NULL };

	assert(NULL == ref);

	irgnm2(conf->iter, conf->alpha, 0., conf->alpha_min, conf->redu, N, M, select_vecops(src),
		frw,
		der,
		(struct iter_op_p_s){ inverse_fista, CAST_UP(&data) },
		dst, ref, src,
		(struct iter_op_s){ NULL, NULL },
		NULL);
}


