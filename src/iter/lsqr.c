/* Copyright 2014,2017. The Regents of the University of California.
 * Copyright 2016-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014      Frank Ong <frankong@berkeley.edu>
 * 2014,2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>
#include <assert.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/iovec.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/itop.h"

#include "lsqr.h"


const struct lsqr_conf lsqr_defaults = { .lambda = 0., .it_gpu = false, .warmstart = false, .icont = NULL, .lambda_scale = 0., .lambda_mask = NULL, };


struct lsqr_data {

	INTERFACE(operator_data_t);

	float l2_lambda;
	float l2_lambda_scale;
	long size;
	bool mask_allocated;
	const complex float* lambda_mask;

	const struct linop_s* model_op;
};

static DEF_TYPEID(lsqr_data);


static void normaleq_l2_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(lsqr_data, _data);

	complex float* tmp = (complex float*)src;
	if ((dst == src) || (NULL != data->lambda_mask)) {

		tmp = md_alloc_sameplace(1, MD_DIMS(data->size), CFL_SIZE, dst);
		md_copy(1, MD_DIMS(data->size), tmp, src, CFL_SIZE);
	}

	linop_normal_unchecked(data->model_op, dst, src);

	if (0 != data->l2_lambda)
		md_zaxpy(1, MD_DIMS(data->size), dst, data->l2_lambda, tmp);

	if ((NULL != data->lambda_mask) && (0 != data->l2_lambda_scale)) {
#ifdef USE_CUDA
		if (cuda_ondevice(dst) && !cuda_ondevice(data->lambda_mask)) {

			data->mask_allocated = true;
			data->lambda_mask = md_gpu_move(1, MD_DIMS(data->size),data->lambda_mask, CFL_SIZE);
		}
#endif
		md_zmul(1, MD_DIMS(data->size), tmp, data->lambda_mask, tmp);
		md_zaxpy(1, MD_DIMS(data->size), dst, data->l2_lambda_scale * mu, tmp);
	}

	if (tmp != src)
		md_free(tmp);
}

static void normaleq_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(lsqr_data, _data);

	if (data->mask_allocated)
		md_free(data->lambda_mask);

	linop_free(data->model_op);

	xfree(data);
}



/**
 * Operator for iterative, multi-regularized least-squares reconstruction
 */
const struct operator_p_s* lsqr2_create(const struct lsqr_conf* conf,
				      italgo_fun2_t italgo, iter_conf* iconf,
				      const float* init,
				      const struct linop_s* model_op,
				      const struct operator_s* precond_op,
			              unsigned int num_funs,
				      const struct operator_p_s* prox_funs[num_funs],
				      const struct linop_s* prox_linops[num_funs],
				      struct iter_monitor_s* monitor)
{
	PTR_ALLOC(struct lsqr_data, data);
	SET_TYPEID(lsqr_data, data);

	const struct iovec_s* iov = NULL;

	if (NULL == model_op) {

		assert(0 < num_funs);
		iov = linop_domain(prox_linops[0]);
		data->model_op = NULL;

	} else {

		iov = operator_domain(model_op->forward);
		data->model_op = linop_clone(model_op);
	}


	data->l2_lambda = conf->lambda;
	data->l2_lambda_scale = conf->lambda_scale;
	data->size = md_calc_size(iov->N, iov->dims);	// FIXME: assume complex
	data->lambda_mask = conf->lambda_mask;
	data->mask_allocated = false;

	const struct operator_p_s* normaleq_op = NULL;
	const struct operator_s* adjoint = NULL;

	if (NULL != model_op) {

		normaleq_op = operator_p_create(iov->N, iov->dims, iov->N, iov->dims, CAST_UP(PTR_PASS(data)), normaleq_l2_apply, normaleq_del);
		adjoint = operator_ref(model_op->adjoint);

	} else {

		PTR_FREE(data);
	}

	if (NULL != precond_op) {

		const struct operator_p_s* tmp_norm = normaleq_op;
		normaleq_op = operator_p_pst_chain(normaleq_op, precond_op);
		operator_p_free(tmp_norm);

		const struct operator_s* tmp_adj = adjoint;
		adjoint = operator_chain(adjoint, precond_op);
		operator_free(tmp_adj);
	}

	const struct operator_p_s* itop_op = itop_p_create(italgo, iconf, conf->warmstart, init, normaleq_op, num_funs, prox_funs, prox_linops, monitor, conf->icont);

	if (conf->it_gpu) {

		debug_printf(DP_DEBUG1, "lsqr: add GPU wrapper\n");
		itop_op = operator_p_gpu_wrapper(itop_op);
	}

	const struct operator_p_s* lsqr_op;

	if (NULL != adjoint)
		lsqr_op = operator_p_pre_chain(adjoint, itop_op);
	else
		lsqr_op = operator_p_ref(itop_op);

	operator_p_free(normaleq_op);
	operator_p_free(itop_op);
	operator_free(adjoint);

	return lsqr_op;
}



/**
 * Perform iterative, multi-regularized least-squares reconstruction
 */
void lsqr2(unsigned int N, const struct lsqr_conf* conf,
	   italgo_fun2_t italgo, iter_conf* iconf,
	   const struct linop_s* model_op,
	   unsigned int num_funs,
	   const struct operator_p_s* prox_funs[num_funs],
	   const struct linop_s* prox_linops[num_funs],
	   const long x_dims[static N], complex float* x,
	   const long y_dims[static N], const complex float* y,
	   const struct operator_s* precond_op,
	   struct iter_monitor_s* monitor)
{
	// nicer, but is still missing some features
	const struct operator_p_s* op = lsqr2_create(conf, italgo, iconf, NULL, model_op, precond_op,
						num_funs, prox_funs, prox_linops, monitor);

	operator_p_apply(op, 1., N, x_dims, x, N, y_dims, y);
	operator_p_free(op);
}




/**
 * Perform iterative, regularized least-squares reconstruction.
 */
void lsqr(unsigned int N,
	  const struct lsqr_conf* conf,
	  italgo_fun_t italgo,
	  iter_conf* iconf,
	  const struct linop_s* model_op,
	  const struct operator_p_s* thresh_op,
	  const long x_dims[static N],
	  complex float* x,
	  const long y_dims[static N],
	  const complex float* y,
	  const struct operator_s* precond_op)
{
	lsqr2(N, conf, iter2_call_iter, CAST_UP(&((struct iter_call_s){ { &TYPEID(iter_call_s), 1. }, italgo, iconf })),
		model_op, (NULL != thresh_op) ? 1 : 0, &thresh_op, NULL,
		x_dims, x, y_dims, y, precond_op, NULL);
}


const struct operator_p_s* wlsqr2_create(const struct lsqr_conf* conf,
					italgo_fun2_t italgo, iter_conf* iconf,
					const float* init,
					const struct linop_s* model_op,
					const struct linop_s* weights,
					const struct operator_s* precond_op,
					unsigned int num_funs,
					const struct operator_p_s* prox_funs[num_funs],
					const struct linop_s* prox_linops[num_funs],
					struct iter_monitor_s* monitor)
{
	struct linop_s* op = linop_chain(model_op, weights);
	linop_free(model_op);

	const struct operator_p_s* lsqr_op = lsqr2_create(conf, italgo, iconf, init,
						op, precond_op,
						num_funs, prox_funs, prox_linops,
						monitor);

	const struct operator_p_s* wlsqr_op = operator_p_pre_chain(weights->forward, lsqr_op);

	linop_free(weights);
	operator_p_free(lsqr_op);
	linop_free(op);

	return wlsqr_op;
}


void wlsqr2(unsigned int N, const struct lsqr_conf* conf,
	    italgo_fun2_t italgo, iter_conf* iconf,
	    const struct linop_s* model_op,
	    unsigned int num_funs,
	    const struct operator_p_s* prox_funs[num_funs],
	    const struct linop_s* prox_linops[num_funs],
	    const long x_dims[static N], complex float* x,
	    const long y_dims[static N], const complex float* y,
	    const long w_dims[static N], const complex float* w,
	    const struct operator_s* precond_op)
{
	unsigned int flags = 0;
	for (unsigned int i = 0; i < N; i++)
		if (1 < w_dims[i])
			flags = MD_SET(flags, i);

	struct linop_s* weights = linop_cdiag_create(N, y_dims, flags, w);
#if 1
	struct linop_s* op = linop_chain(model_op, weights);

	complex float* wy = md_alloc_sameplace(N, y_dims, CFL_SIZE, y);

	linop_forward(weights, N, y_dims, wy, N, y_dims, y);

	lsqr2(N, conf, italgo, iconf, op, num_funs, prox_funs, prox_linops, x_dims, x, y_dims, wy, precond_op, NULL);

	md_free(wy);

	linop_free(op);
#else
	const struct operator_s* op = wlsqr2_create(conf, italgo, iconf, model_op, weights, precond_op,
						num_funs, prox_funs, prox_linops);

	operator_apply(op, N, x_dims, x, N, y_dims, y);
#endif
	linop_free(weights);
}

//  A^H W W A - A^H W W y
void wlsqr(unsigned int N, const struct lsqr_conf* conf,
	   italgo_fun_t italgo, iter_conf* iconf,
	   const struct linop_s* model_op,
	   const struct operator_p_s* thresh_op,
	   const long x_dims[static N], complex float* x,
	   const long y_dims[static N], const complex float* y,
	   const long w_dims[static N], const complex float* w,
	   const struct operator_s* precond_op)
{
	wlsqr2(N, conf, iter2_call_iter, CAST_UP(&((struct iter_call_s){ { &TYPEID(iter_call_s), 1. }, italgo, iconf })),
	       model_op, (NULL != thresh_op) ? 1 : 0, &thresh_op, NULL,
	       x_dims, x, y_dims, y, w_dims, w, precond_op);
}
