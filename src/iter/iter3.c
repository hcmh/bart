/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "wavelet/wavthresh.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"

extern int k = 0;

DEF_TYPEID(iter3_irgnm_conf);
DEF_TYPEID(iter3_landweber_conf);

const struct iter3_irgnm_conf iter3_irgnm_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter3_irgnm_conf),

	.iter = 8,
	.alpha = 1.,
	.redu = 2.,

	.cgiter = 100,
	.cgtol = 0.1,

	.nlinv_legacy = false,
};


struct irgnm_s {

	INTERFACE(iter_op_data);

	struct iter_op_s frw;
	struct iter_op_s der;
	struct iter_op_s adj;

	float* tmp;
    
	long size;

	int cgiter;
	float cgtol;
	bool nlinv_legacy;
    float alpha;
    

};

DEF_TYPEID(irgnm_s);

static void normal(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->der, data->tmp, src);
	iter_op_call(data->adj, dst, data->tmp);
}

static void normal_fista(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

    iter_op_call(data->der, data->tmp, src);
    iter_op_call(data->adj, dst, data->tmp);

    select_vecops(src)->axpy(data->size, dst, data->alpha, src);
}

static void inverse(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

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

static void inverse_fista(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);
	data->alpha = alpha;
	float eps = md_norm(1, MD_DIMS(data->size), src);
    
//     long size = md_calc_size(DIMS, nlop_generic_domain(&data->nlop, i)->dims);
	
	
	void* x = md_alloc_sameplace(1, MD_DIMS(data->size/2), CFL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size/2), x);
	double maxeigen = power(30, data->size, select_vecops(src), (struct iter_op_s){normal_fista, CAST_UP(data)}, x);
	md_free(x);
//	debug_printf(DP_INFO, "\tMax eigv: %.2e\n", maxeigen);

	double step = 0.45/maxeigen;//fmin(iter_fista_defaults.step / maxeigen, iter_fista_defaults.step); // 0.95f is FISTA standard
//	debug_printf(DP_INFO, "\tFISTA Stepsize: %.2e\n", step);
    
    struct operator_p_s* prox;
    if (1) {
        // for FISTA+
		bool randshift = true;
		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		unsigned int wflags = 0;
        unsigned int jwflags = 0;
//         const long* dims = nlop_generic_domain(&data->nlop, 0)->dims;
        long dims[16] = {384,384,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        for (unsigned int i = 0; i < DIMS; i++) {
            if ((1 < dims[i]) && MD_IS_SET(FFT_FLAGS|SLICE_FLAG, i)) {
                wflags = MD_SET(wflags, i);
                minsize[i] = MIN(dims[i], 16);
            }
        }
		prox = prox_wavelet_thresh_create(DIMS, dims, wflags, jwflags, minsize, alpha, randshift);
	}
	
	
    
    int maxiter = 10*powf(2,k);
    
    if(maxiter > 400)
        maxiter = 400;

	fista(maxiter, 0.0f * alpha * eps, step,
		iter_fista_defaults.continuation, iter_fista_defaults.hogwild,
		data->size, 
		select_vecops(src),
		(struct iter_op_s){normal_fista, CAST_UP(data)},
        OPERATOR_P2ITOP(prox),
//        *(prox),
        dst, src, NULL);
    k++;
}

static void forward(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->frw, dst, src);
}

static void derivative(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->der, dst, src);
}

static void adjoint(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->adj, dst, src);
}



void iter3_irgnm(iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(M), FL_SIZE, src);
	struct irgnm_s data = { { &TYPEID(irgnm_s) }, frw, der, adj, tmp, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	irgnm(conf->iter, conf->alpha, conf->redu, N, M, select_vecops(src),
		(struct iter_op_s){ forward, CAST_UP(&data) },
		(struct iter_op_s){ adjoint, CAST_UP(&data) },
		(struct iter_op_p_s){ inverse, CAST_UP(&data) },
		dst, ref, src,
		(struct iter_op_s){NULL, NULL});

	md_free(tmp);
}

void iter3_irgnm_l1(iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(M), FL_SIZE, src);
	struct irgnm_s data = { { &TYPEID(irgnm_s) }, frw, der, adj, tmp, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy };

	irgnm_l1(conf->iter, conf->alpha, conf->redu, N, M, select_vecops(src),
		(struct iter_op_s){ forward, CAST_UP(&data) },
		(struct iter_op_s){ derivative, CAST_UP(&data) },
		(struct iter_op_s){ adjoint, CAST_UP(&data) },
		(struct iter_op_p_s){ inverse_fista, CAST_UP(&data) },
		dst, ref, src,
		(struct iter_op_s){NULL, NULL});

	md_free(tmp);
}



void iter3_landweber(iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter3_landweber_conf* conf = CAST_DOWN(iter3_landweber_conf, _conf);

	assert(NULL == der.fun);
	assert(NULL == ref);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(N), FL_SIZE, src);

	landweber(conf->iter, conf->epsilon, conf->alpha, N, M,
		select_vecops(src), frw, adj, dst, src, NULL);

	md_free(tmp);
}



