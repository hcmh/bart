/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
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
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/thresh.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/admm.h"

#include "optreg_moba.h"
#include "grecon/italgo.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"



#include "iter_l1.h"



struct T1inv_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;
	const struct mdb_irgnm_l1_conf* conf;
    
	long size_x;
	long size_y;

	float alpha;
    
	const long* dims;

	bool first_iter;
	int outer_iter;

	const struct operator_p_s* prox1;
	const struct operator_p_s* prox2;
};

DEF_TYPEID(T1inv_s);



static void normal(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	linop_normal_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)dst, (const complex float*)src);

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];
	long coils = data->dims[COIL_DIM];
	long slices = data->dims[SLICE_DIM];
	long mphases = data->dims[TIME2_DIM];

	if (1 == data->conf->opt_reg) {

		md_axpy(1, MD_DIMS(data->size_x * coils / (coils + parameters)),
	                                        dst + res * res * 2 * parameters * slices * mphases,
						data->alpha,
	                                        src + res * res * 2 * parameters * slices * mphases);
	} else {

		md_axpy(1, MD_DIMS(data->size_x), dst, data->alpha, src);
	}
}


static void pos_value(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	long res = data->dims[0];
	long parameters = data->dims[COEFF_DIM];
	long slices = data->dims[SLICE_DIM];
	long mphases = data->dims[TIME2_DIM];

	long dims1[DIMS];

	md_select_dims(DIMS, FFT_FLAGS, dims1, data->dims);

	for (int j = 0; j < mphases; j++) {

		for (int i = 0; i < slices; i++) {

			int map = 0;
			int constrain_flags = data->conf->constrained_maps;

			while (constrain_flags) {

				if (constrain_flags & 1) {

					debug_printf(DP_DEBUG3, "Cosen constrained maps: %d\n", map);

					md_zsmax(DIMS, dims1, (_Complex float*)dst + map * res * res
							+ i * res * res * parameters + j * res * res * parameters * slices,
						(const _Complex float*)src + map * res * res
							+ i * res * res * parameters + j * res * res * parameters * slices,
						data->conf->lower_bound);
				}

				constrain_flags >>= 1;
				map++;
			}
		}
        }
}


static void combined_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	struct T1inv_s* data = CAST_DOWN(T1inv_s, _data);

	if (data->first_iter) {

		data->first_iter = false;

	} else {

		pos_value(_data, dst, src);
	}

	if (1 == data->conf->opt_reg) {

		operator_p_apply_unchecked(data->prox2, rho, (_Complex float*)dst, (const _Complex float*)dst);
	}

	pos_value(_data, dst, dst);
}



static void inverse_fista(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	data->alpha = alpha;	// update alpha for normal operator

	void* x = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);
	double maxeigen = power(20, data->size_x, select_vecops(src), (struct iter_op_s){ normal, CAST_UP(data) }, x);
	md_free(x);

	double step = data->conf->step / maxeigen;

	debug_printf(DP_DEBUG3, "##reg. alpha = %f\n", alpha);

	wavthresh_rand_state_set(data->prox1, 1);
    
	int maxiter = MIN(data->conf->c2->cgiter, 10 * powf(2, data->outer_iter));
    
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);

	float eps = md_norm(1, MD_DIMS(data->size_x), tmp);

	data->first_iter = true;

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = data->alpha;
	};

	fista(maxiter, data->conf->c2->cgtol * alpha * eps, step,
		data->size_x,
		select_vecops(src),
		continuation,
		(struct iter_op_s){ normal, CAST_UP(data) },
		(struct iter_op_p_s){ combined_prox, CAST_UP(data) },
		dst, tmp, NULL);

	pos_value(CAST_UP(data), dst, dst);

	md_free(tmp);

	data->outer_iter++;
}


static const struct operator_p_s* create_prox(const long img_dims[DIMS], unsigned int jflag, float lambda)
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

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, jflag, minsize, lambda, randshift);
}

static void inverse_admm(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(T1inv_s, _data);

	data->alpha = alpha;	// update alpha for normal operator
    
	int maxiter = MIN(data->conf->c2->cgiter, 10 * powf(2, data->outer_iter));
    
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_y), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = data->alpha;
	};

	UNUSED(continuation);

	unsigned int llr_blk = 8;
	unsigned int shift_mode = 1;

	// initialize prox functions

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	data->conf->ropts->lambda = data->alpha;

	debug_printf(DP_INFO, "##reg. alpha = %f\n", data->conf->ropts->lambda);

	opt_reg_configure_moba(DIMS, data->dims, data->conf->ropts, thresh_ops, trafos, llr_blk, shift_mode, NULL, NULL, data->conf->usegpu);

	struct iter_admm_conf conf1 = iter_admm_defaults;

	conf1.maxiter = maxiter;
	conf1.rho = data->conf->rho;
	conf1.cg_eps = 0.01;

	struct iter_admm_conf *conf = &conf1;
	unsigned int D = data->conf->ropts->r;;

	const struct operator_s* normaleq_op = NULL;
        
        UNUSED(normaleq_op);

	struct admm_plan_s admm_plan = {

		.maxiter = conf->maxiter,
		.maxitercg = conf->maxitercg,
		.cg_eps = conf->cg_eps,
		.rho = conf->rho,
		.num_funs = D,
		.do_warmstart = conf->do_warmstart,
		.dynamic_rho = conf->dynamic_rho,
		.dynamic_tau = conf->dynamic_tau,
		.relative_norm = conf->relative_norm,
		.hogwild = conf->hogwild,
		.ABSTOL = conf->ABSTOL,
		.RELTOL = conf->RELTOL,
		.alpha = conf->alpha * conf->INTERFACE.alpha,
		.tau = conf->tau,
		.tau_max = conf->tau_max,
		.mu = conf->mu,
		.fast = conf->fast,
		.biases = NULL,
	};

	struct admm_op a_ops[D ?:1];
	struct iter_op_p_s a_prox_ops[D ?:1];


	for (unsigned int i = 0; i < D; i++) {

		a_ops[i].forward = OPERATOR2ITOP(trafos[i]->forward),
		a_ops[i].normal = OPERATOR2ITOP(trafos[i]->normal);
		a_ops[i].adjoint = OPERATOR2ITOP(trafos[i]->adjoint);

		a_prox_ops[i] = OPERATOR_P2ITOP(thresh_ops[i]);
	}

	admm_plan.ops = a_ops;
	admm_plan.prox_ops = a_prox_ops;


	long z_dims[D ?: 1];

	for (unsigned int i = 0; i < D; i++)
		z_dims[i] = 2 * md_calc_size(linop_codomain(trafos[i])->N, linop_codomain(trafos[i])->dims);


	admm(&admm_plan, admm_plan.num_funs, 
		z_dims, data->size_x, (float*)dst, tmp, 
		select_vecops(src), 
		(struct iter_op_s){ normal, CAST_UP(data) }, NULL);


	opt_reg_free_moba(data->conf->ropts, thresh_ops, trafos);


	md_free(tmp);

	data->outer_iter++;
}


struct T1inv2_s {

	INTERFACE(operator_data_t);

	struct T1inv_s data;
};

DEF_TYPEID(T1inv2_s);





static void T1inv_apply(const operator_data_t* _data, float alpha, complex float* dst, const complex float* src)
{
	const auto data = &CAST_DOWN(T1inv2_s, _data)->data;
	

	switch (data->conf->algo) {
		
	case ALGO_FISTA:
		inverse_fista(CAST_UP(data), alpha, (float*)dst, (const float*)src);
		break;

	case ALGO_ADMM:
		inverse_admm(CAST_UP(data), alpha, (float*)dst, (const float*)src);
		break;
	
	default:
		break;
	}
}

static void T1inv_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(T1inv2_s, _data);

	operator_p_free(data->data.prox1);
	operator_p_free(data->data.prox2);

	nlop_free(data->data.nlop);

	xfree(data->data.dims);
	xfree(data);
}


static const struct operator_p_s* T1inv_p_create(const struct mdb_irgnm_l1_conf* conf, const long dims[DIMS], struct nlop_s* nlop)
{
	PTR_ALLOC(struct T1inv2_s, data);
	SET_TYPEID(T1inv2_s, data);
	SET_TYPEID(T1inv_s, &data->data);

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	int M = 2 * md_calc_size(cd->N, cd->dims);
	int N = 2 * md_calc_size(dm->N, dm->dims);

	long* ndims = *TYPE_ALLOC(long[DIMS]);
	md_copy_dims(DIMS, ndims, dims);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims); 
	img_dims[COEFF_DIM] = img_dims[COEFF_DIM] - conf->not_wav_maps;	// Just penalize T1 map
	debug_print_dims(DP_INFO, DIMS, img_dims);

	auto prox1 = create_prox(img_dims, COEFF_FLAG, conf->wav_reg);
	auto prox2 = op_p_auto_normalize(prox1, ~(COEFF_FLAG | SLICE_FLAG | TIME2_FLAG));

	if (1 == conf->not_wav_maps) {

		long map_dims[DIMS];
		md_copy_dims(DIMS, map_dims, img_dims);
		map_dims[COEFF_DIM] = 1L;

		// auto prox3 = create_prox(map_dims, 0, 0.05);
		auto prox3 = prox_zero_create(DIMS, map_dims);
		auto prox4 = operator_p_stack(COEFF_DIM, COEFF_DIM, prox1, prox3);
		prox2 = op_p_auto_normalize(prox4, ~(COEFF_FLAG | SLICE_FLAG | TIME2_FLAG));

		operator_p_free(prox3);
		operator_p_free(prox4);
	}

	struct T1inv_s idata = {

		{ &TYPEID(T1inv_s) }, nlop_clone(nlop), conf,
		N, M, 1.0, ndims, true, 0, prox1, prox2
	};

	data->data = idata;

	return operator_p_create(dm->N, dm->dims, cd->N, cd->dims, CAST_UP(PTR_PASS(data)), T1inv_apply, T1inv_del);
}


struct iterT1_nlop_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;
};

DEF_TYPEID(iterT1_nlop_s);



void mdb_irgnm_l1(const struct mdb_irgnm_l1_conf* conf,
	const long dims[],
	struct nlop_s* nlop,
	long N, float* dst,
	long M, const float* src)
{
	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	const struct operator_p_s* inv_op = NULL;


#if 0	
	// initialize prox functions

        const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	unsigned int llr_blk = 8;
	unsigned int shift_mode = 1;

	opt_reg_configure_moba(DIMS, dims, conf->ropts, thresh_ops, trafos, llr_blk, shift_mode, NULL, NULL, conf->usegpu);

	struct iter_admm_conf conf1 = iter_admm_defaults;

	int outer_iter = 4;
	conf1.maxiter = MIN(conf->c2->cgiter, 10 * powf(2, outer_iter));
	conf1.rho = conf->rho;
	conf1.cg_eps = 0.005;
		
	inv_op = lsqr2_create(&lsqr_defaults, iter2_admm, CAST_UP(&conf1), dst, &nlop->derivative[0][0],
                         NULL, conf->ropts->r, thresh_ops, trafos, NULL);
#endif

	inv_op = T1inv_p_create(conf, dims, nlop);
	       


	iter4_irgnm2(CAST_UP(conf->c2), nlop, N, dst, NULL, M, src, inv_op,
		(struct iter_op_s){ NULL, NULL });

	operator_p_free(inv_op);

#if 0
         opt_reg_free_moba(conf->ropts, thresh_ops, trafos);
#endif

}

