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
#include <stdio.h>

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/ops.h"

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

//#define mphase

extern int k_fista = 0;

DEF_TYPEID(iter3_irgnm_conf);
DEF_TYPEID(iter3_landweber_conf);

const struct iter3_irgnm_conf iter3_irgnm_defaults = {

	.INTERFACE.TYPEID = &TYPEID(iter3_irgnm_conf),

	.iter = 8,
	.alpha = 0.1,
	.redu = 3.,

	.cgiter = 100,
	.cgtol = 0.1,

	.nlinv_legacy = false,
    .alpha_min = 0.001,
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
    
    float alpha_min;
    long* dims;

    struct operator_p_s* prox1;
    struct operator_p_s* prox2;
    

};

DEF_TYPEID(irgnm_s);

static void normal(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

	iter_op_call(data->der, data->tmp, src);
	iter_op_call(data->adj, dst, data->tmp);
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


static void normal_fista(iter_op_data* _data, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

    iter_op_call(data->der, data->tmp, src);
    iter_op_call(data->adj, dst, data->tmp);

    long res = data->dims[0];
    long SMS = data->dims[SLICE_DIM];
    long parameters = data->dims[COEFF_DIM];
    long coils = data->dims[COIL_DIM];
    long TIME2 = data->dims[TIME2_DIM];
    
   
    // only add l2 norm to the coils, not parameter maps
    for(int u = 0; u < SMS; u++)
        for(int v = 0; v < TIME2; v++){
        select_vecops(src)->axpy(data->size*coils*SMS*TIME2/(coils*SMS*TIME2 + parameters*SMS*TIME2),
                                 dst + res*res*2*(parameters*SMS*TIME2),
                                 data->alpha, 
                                 src + res*res*2*(parameters*SMS*TIME2));
     }
//       select_vecops(src)->axpy(data->size, dst, data->alpha, src);

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

static void combined_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
    struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);

    operator_p_apply_unchecked(data->prox1, rho, (_Complex float*)dst, (const _Complex float*)src);
    operator_p_apply_unchecked(data->prox2, rho, (_Complex float*)dst, (const _Complex float*)src);
}



static void inverse_fista(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	struct irgnm_s* data = CAST_DOWN(irgnm_s, _data);
    if (alpha < data->alpha_min)
            alpha = data->alpha_min;
    data->alpha = alpha;
	float eps = md_norm(1, MD_DIMS(data->size), src);
    
	void* x = md_alloc_sameplace(1, MD_DIMS(data->size/2), CFL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size/2), x);
	double maxeigen = power(20, data->size, select_vecops(src), (struct iter_op_s){normal_fista, CAST_UP(data)}, x);
	md_free(x);
//	debug_printf(DP_INFO, "\tMax eigv: %.2e\n", maxeigen);

	double step = 0.475/maxeigen;//fmin(iter_fista_defaults.step / maxeigen, iter_fista_defaults.step); // 0.95f is FISTA standard
//	debug_printf(DP_INFO, "\tFISTA Stepsize: %.2e\n", step);
//     float alpha_min = data->alpha_min;
    long img_dims[16];
    md_select_dims(16, ~COIL_FLAG, img_dims, data->dims);
    debug_print_dims(DP_INFO, DIMS, img_dims);

    const struct operator_p_s* prox;
    if (1) {
        // for FISTA+
        bool randshift = true;
        long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
        unsigned int wflags = 0;
        unsigned int jwflags_1 = 0;
        unsigned int jwflags_2 = 0;
        for (unsigned int i = 0; i < DIMS; i++) {
            if ((1 < img_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {
                wflags = MD_SET(wflags, i);
                minsize[i] = MIN(img_dims[i], 16);
            }
        } 
        jwflags_1 = MD_SET(jwflags_1, 6);
        jwflags_2 = MD_SET(jwflags_2, 11);
        
//		prox = prox_wavelet_thresh_create(DIMS, img_dims, wflags, jwflags, minsize, alpha, randshift);
#ifdef mphase
        data->prox1 = prox_wavelet_thresh_create(DIMS, img_dims, wflags, jwflags_1, minsize, 0, randshift);
        data->prox2 = prox_wavelet_thresh_create(DIMS, img_dims, wflags, jwflags_2, minsize, alpha, randshift);
#else
        data->prox1 = prox_wavelet_thresh_create(DIMS, img_dims, wflags, jwflags_1, minsize, alpha, randshift);
        data->prox2 = prox_wavelet_thresh_create(DIMS, img_dims, wflags, jwflags_2, minsize, 0, randshift);
#endif
	}
    
    debug_printf(DP_DEBUG3, "##reg. alpha = %f\n", alpha);
    
    int maxiter = 10*powf(2, k_fista);
    
    if(maxiter > 250)
        maxiter = 250;

    fista_xw(maxiter, 0.01f * alpha * eps, step, data->dims,
		iter_fista_defaults.continuation, iter_fista_defaults.hogwild,
		data->size, 
		select_vecops(src),
		(struct iter_op_s){normal_fista, CAST_UP(data)},
//        OPERATOR_P2ITOP(prox),
         (struct iter_op_p_s){combined_prox, CAST_UP(data)},
//         data->prox,
        dst, src, NULL);
    k_fista++;
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
	struct irgnm_s data = { { &TYPEID(irgnm_s) }, frw, der, adj, tmp, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy, 1.0 };

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
	struct irgnm_s data = { { &TYPEID(irgnm_s) }, frw, der, adj, tmp, N, conf->cgiter, conf->cgtol, conf->nlinv_legacy, 1.0, conf->alpha_min, conf->dims};

    irgnm_l1(conf->iter, conf->alpha, conf->redu, N, M, conf->dims, select_vecops(src),
        frw,
		der,
		adj,
		(struct iter_op_p_s){ inverse_fista, CAST_UP(&data)},
//         (struct iter_op_p_s){ inverse, CAST_UP(&data) },
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



