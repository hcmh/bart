/* Copyright 2014-2017. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017	Jon Tamir <jtamir@eecs.berkeley.edu>
 * 2016-2019	Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/rand.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "nlops/nlop.h"

#include "nn/tf_wrapper_prox.h"

#include "iter/iter.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "prox.h"
#include "stdio.h"

/**
 * Proximal function of f is defined as
 * (prox_f)(z) = arg min_x 0.5 || z - x ||_2^2 + f(x)
 *
 * (prox_{mu f})(z) = arg min_x 0.5 || z - x ||_2^2 + mu f(x)
 */


/**
 * Data for computing prox_normaleq_fun:
 * Proximal function for f(z) = 0.5 || y - A z ||_2^2.
 *
 * @param op operator that applies A^H A
 * @param cgconf conf file for conjugate gradient iter interface
 * @param adj A^H y
 * @param size size of z
 */
struct prox_normaleq_data {

	INTERFACE(operator_data_t);

	const struct linop_s* op;
	void* cgconf;
	float* adj;

	long size;
};

static DEF_TYPEID(prox_normaleq_data);



/**
 * Proximal function for f(z) = 0.5 || y - A z ||_2^2.
 * Solution is (A^H A + (1/mu) I)z = A^H y + (1/mu)(x_plus_u)
 *
 * @param prox_data should be of type prox_normaleq_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_normaleq_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	auto pdata = CAST_DOWN(prox_normaleq_data, prox_data);

	if (0 == mu) {

		md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);

	} else {

		float rho = 1. / mu;
		float* b = md_alloc_sameplace(1, MD_DIMS(pdata->size), FL_SIZE, x_plus_u);
		md_copy(1, MD_DIMS(pdata->size), b, pdata->adj, FL_SIZE);
		md_axpy(1, MD_DIMS(pdata->size), b, rho, x_plus_u);

		if (NULL == pdata->op->norm_inv) {

			struct iter_conjgrad_conf* cg_conf = pdata->cgconf;
			cg_conf->l2lambda = rho;

			iter_conjgrad(CAST_UP(cg_conf), pdata->op->normal, NULL, pdata->size, z, (float*)b, NULL);

		} else {

			linop_norm_inv_unchecked(pdata->op, rho, (complex float*)z, (const complex float*)b);
		}

		md_free(b);
	}
}

static void prox_normaleq_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_normaleq_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_normaleq_del(const operator_data_t* _data)
{
	auto pdata = CAST_DOWN(prox_normaleq_data, _data);

	xfree(pdata->cgconf);
	md_free(pdata->adj);
	xfree(pdata);
}

const struct operator_p_s* prox_normaleq_create(const struct linop_s* op, const complex float* y)
{
	PTR_ALLOC(struct prox_normaleq_data, pdata);
	SET_TYPEID(prox_normaleq_data, pdata);
	PTR_ALLOC(struct iter_conjgrad_conf, cgconf);

	*cgconf = iter_conjgrad_defaults;
	cgconf->maxiter = 10;
	cgconf->l2lambda = 0;

	pdata->cgconf = PTR_PASS(cgconf);
	pdata->op = op;

	pdata->size = 2 * md_calc_size(linop_domain(op)->N, linop_domain(op)->dims);
	pdata->adj = md_alloc_sameplace(1, &(pdata->size), FL_SIZE, y);
	linop_adjoint_unchecked(op, (complex float*)pdata->adj, y);

	return operator_p_create(linop_domain(op)->N, linop_domain(op)->dims,
			linop_domain(op)->N, linop_domain(op)->dims,
			CAST_UP(PTR_PASS(pdata)), prox_normaleq_apply, prox_normaleq_del);
}


/**
 * Data for computing prox_leastsquares_fun:
 * Proximal function for f(z) = lambda / 2 || y - z ||_2^2.
 *
 * @param y
 * @param lambda regularization
 * @param size size of z
 */
struct prox_leastsquares_data {

	INTERFACE(operator_data_t);

	const float* y;
	float lambda;

	long size;
};

static DEF_TYPEID(prox_leastsquares_data);


/**
 * Proximal function for f(z) = lambda / 2 || y - z ||_2^2.
 * Solution is z =  (mu * lambda * y + x_plus_u) / (mu * lambda + 1)
 *
 * @param prox_data should be of type prox_leastsquares_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_leastsquares_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	auto pdata = CAST_DOWN(prox_leastsquares_data, prox_data);

	md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);

	if (0 != mu) {

		if (NULL != pdata->y)
			md_axpy(1, MD_DIMS(pdata->size), z, pdata->lambda * mu, pdata->y);

		md_smul(1, MD_DIMS(pdata->size), z, z, 1. / (mu * pdata->lambda + 1));
	}
}

static void prox_leastsquares_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_leastsquares_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_leastsquares_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_leastsquares_data, _data));
}

const struct operator_p_s* prox_leastsquares_create(unsigned int N, const long dims[N], float lambda, const complex float* y)
{
	PTR_ALLOC(struct prox_leastsquares_data, pdata);
	SET_TYPEID(prox_leastsquares_data, pdata);

	pdata->y = (const float*)y;
	pdata->lambda = lambda;
	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_leastsquares_apply, prox_leastsquares_del);
}


/**
 * Data for computing prox_l2norm_fun:
 * Proximal function for f(z) = lambda || z ||_2.
 *
 * @param lambda regularization
 * @param size size of z
 */
struct prox_l2norm_data {

	INTERFACE(operator_data_t);

	float lambda;
	long size;
};

static DEF_TYPEID(prox_l2norm_data);


/**
 * Proximal function for f(z) = lambda  || z ||_2.
 * Solution is z =  ( 1 - lambda * mu / norm(z) )_+ * z,
 * i.e. block soft thresholding
 *
 * @param prox_data should be of type prox_l2norm_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_l2norm_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	auto pdata = CAST_DOWN(prox_l2norm_data, prox_data);

	md_clear(1, MD_DIMS(pdata->size), z, FL_SIZE);

	double q1 = md_norm(1, MD_DIMS(pdata->size), x_plus_u);

	if (q1 != 0) {

		double q2 = 1 - pdata->lambda * mu / q1;

		if (q2 > 0.)
			md_smul(1, MD_DIMS(pdata->size), z, x_plus_u, q2);
	}
}

static void prox_l2norm_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_l2norm_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_l2norm_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_l2norm_data, _data));
}

const struct operator_p_s* prox_l2norm_create(unsigned int N, const long dims[N], float lambda)
{
	PTR_ALLOC(struct prox_l2norm_data, pdata);
	SET_TYPEID(prox_l2norm_data, pdata);

	pdata->lambda = lambda;
	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_l2norm_apply, prox_l2norm_del);
}


/*
 * proximal function for log probability function as a denoiser in iteration
 * v_k+1 <- Denoiser(v_k)
 * 
 */
struct  prox_logp_data
{
	INTERFACE(operator_data_t);

	const struct nlop_s *tf_ops;
	unsigned int N;
	const long *dims;
	float step_size;
	float p;
	unsigned int steps;
};

DEF_TYPEID(prox_logp_data);

static int compare_cmpl_magn(const void* a, const void* b)
{
	return (int)copysignf(1., (cabsf(*(complex float*)a) - cabsf(*(complex float*)b)));
}

static float calculate_max(unsigned int D, const long* dims, complex float* iptr)
{	
	long imsize = md_calc_size(D, dims);
	complex float* tmp = md_alloc(D, dims, CFL_SIZE);
	md_copy(D, dims, tmp, iptr, CFL_SIZE);
	qsort(tmp, (size_t)imsize, sizeof(complex float), compare_cmpl_magn);
	return cabsf(tmp[imsize-1]);
}


static void prox_logp_fun(const operator_data_t* data, float lambda, complex float *dst, const  complex float* src)
{

	auto pdata = CAST_DOWN(prox_logp_data, data);
	
	auto dom = nlop_generic_domain(pdata->tf_ops, 0);
	auto cod = nlop_generic_codomain(pdata->tf_ops, 0); // grad_ys

	long resized_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, resized_dims, pdata->dims);
	complex float* u_resized = NULL;

	// hard code below crop FOV 
	if(pdata->dims[0]>256)
	{

		resized_dims[0] = pdata->dims[0]/2;
		resized_dims[1] = pdata->dims[1]/2;

		u_resized = md_alloc(DIMS, resized_dims, CFL_SIZE);

		long pos[2];
		for (unsigned int i = 0; i < 2; i++)
			pos[i] = labs((resized_dims[i] / 2) - (pdata->dims[i] / 2));

		md_copy_block(2, pos, resized_dims, u_resized, pdata->dims, (const complex float*)src, CFL_SIZE);
	}
	else{
		u_resized = md_alloc(DIMS, pdata->dims, CFL_SIZE);
		md_copy(DIMS, resized_dims, u_resized, (const complex float*)src, CFL_SIZE);
	}

	// slice FOV
	long slice_dims[DIMS];
	md_set_dims(DIMS, slice_dims, 1);

	slice_dims[0] = dom->dims[1];
	slice_dims[1] = dom->dims[2];

	unsigned int nx = (resized_dims[0] + slice_dims[0] - 1)/slice_dims[0];
	unsigned int ny = (resized_dims[1] + slice_dims[1] - 1)/slice_dims[1];

	slice_dims[2] = nx*ny;
	
	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);
	
	complex float* slices = md_alloc(DIMS, slice_dims, CFL_SIZE);
	complex float* tmp_slices = md_alloc(DIMS, slice_dims, CFL_SIZE);
	
	float scalor = calculate_max(DIMS, resized_dims, u_resized);
	
	scalor = scalor + 1e-06;
	md_zsmul(DIMS, resized_dims, u_resized, u_resized, 1. / scalor);

	int offset = 0;
	for (size_t i = 0; i < nx; i++)
	{
		for (size_t j = 0; j < ny; j++)
		{
			pos[0] = j*slice_dims[1];
			pos[1] = i*slice_dims[0];
			offset = (i*nx + j) * slice_dims[0]*slice_dims[1];
			md_copy_block(2, pos, slice_dims, tmp_slices + offset, resized_dims, u_resized, CFL_SIZE);
		}
	}
	
	md_transpose(pdata->N, 1, 0, slice_dims, slices, slice_dims, tmp_slices, CFL_SIZE);
	
	complex float* out = md_alloc(cod->N, cod->dims, cod->size);
	nlop_apply(pdata->tf_ops, cod->N, cod->dims, out, dom->N, dom->dims, slices);
	debug_printf(DP_DEBUG3, "\tLog P: %f  Scalor: %f  Step size: %f  lambda: %f\n ", creal(*out), scalor, pdata->step_size, lambda);

	//copy slices to feed tensor
	struct TF_Tensor ** input_tensor = get_input_tensor(pdata->tf_ops);
	md_copy(dom->N, dom->dims, TF_TensorData(*input_tensor), slices, CFL_SIZE);

	complex float* grad = md_alloc(dom->N, dom->dims, dom->size);
	complex float grad_ys = 1 + 1*I;

	nlop_adjoint(pdata->tf_ops, dom->N, dom->dims, grad, cod->N, cod->dims, &grad_ys); // grad [4, sx, sy]

	// update 

	md_zsmul(dom->N, dom->dims, grad, grad, pdata->step_size*lambda);
	complex float* rand_mask = md_alloc(dom->N, dom->dims, CFL_SIZE);

	md_rand_one(dom->N, dom->dims, rand_mask, pdata->p);
	md_zmul(dom->N, dom->dims, grad, grad, rand_mask);
	
	md_zsub(dom->N, dom->dims, slices, slices, grad);   // dst(src+1) = src - grad
	
	// back to fortran arrays
	complex float* tmp = md_alloc(DIMS, resized_dims, CFL_SIZE);
	complex float* tmp1 = md_alloc(DIMS, resized_dims, CFL_SIZE);
	for(size_t i=0; i < nx; i++)
	{
		for (size_t j=0; j < ny; j++)
		{
			pos[0] = i*slice_dims[0];
			pos[1] = j*slice_dims[1];
			offset = (i*nx + j) * slice_dims[0]*slice_dims[1];
			md_copy_block(2, pos, resized_dims, tmp, slice_dims, slices+offset, CFL_SIZE);
		}
	}
	md_zsmul(DIMS, resized_dims, tmp, tmp, scalor);

	md_transpose(DIMS, 1, 0, resized_dims, tmp1, resized_dims, tmp, CFL_SIZE);
	
	for (unsigned int i = 0; i < 2; i++)
		pos[i] = labs((resized_dims[i] / 2) - (pdata->dims[i] / 2));

	md_copy_block(2, pos, pdata->dims, (complex float*)dst, resized_dims, (const complex float*)tmp1, CFL_SIZE);
	
	md_free(u_resized);
	md_free(slices);
	md_free(tmp_slices);
	md_free(tmp);
}

static void prox_logp_apply(const operator_data_t* data, float lambda, complex float* dst, const complex float* src)
{
	auto pdata = CAST_DOWN(prox_logp_data, data);
	
	complex float* tmp = md_alloc(pdata->N, pdata->dims, CFL_SIZE);
	md_copy(pdata->N, pdata->dims, tmp, src, CFL_SIZE);

	for(unsigned int i=0; i<pdata->steps; i++)
	{
		prox_logp_fun(data, lambda, dst, tmp);
		md_copy(pdata->N, pdata->dims, tmp, dst, CFL_SIZE);
	}
	
	md_free(tmp);
}

static void prox_logp_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_logp_data, _data));
}

extern const struct operator_p_s* prox_logp_create(unsigned int N, const long dims[__VLA(N)], const struct nlop_s * tf_ops, float step_size, float p, unsigned int steps)
{
	PTR_ALLOC(struct prox_logp_data, pdata);
	SET_TYPEID(prox_logp_data, pdata);

	pdata->tf_ops = tf_ops;
	pdata->step_size = step_size;
	pdata->p = p;
	
	pdata->N = N;
	pdata->dims = (long*)malloc(sizeof(long)*pdata->N);
	md_copy_dims(pdata->N, pdata->dims, dims);
	pdata->steps = steps;
	
	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_logp_apply, prox_logp_del);
}

struct  prox_logp_nlinv_data
{
	INTERFACE(operator_data_t);

	const struct nlop_s *tf_ops;
	unsigned int N;

	const long *dims;
	float step_size;
	float p;
	unsigned int steps;

	unsigned int irgnm_steps;
	float base;
	float rho;
};

DEF_TYPEID(prox_logp_nlinv_data);

static void prox_logp_nlinv_fun(const operator_data_t* data, float lambda, complex float *dst, const  complex float* src)
{

	auto pdata = CAST_DOWN(prox_logp_nlinv_data, data);
	
	auto dom = nlop_generic_domain(pdata->tf_ops, 0);
	auto cod = nlop_generic_codomain(pdata->tf_ops, 0); // grad_ys

	long resized_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, resized_dims, pdata->dims);
	complex float* u_resized = NULL;

	// hard code below crop FOV 
	if(pdata->dims[0]>256)
	{

		resized_dims[0] = pdata->dims[0]/2;
		resized_dims[1] = pdata->dims[1]/2;

		u_resized = md_alloc(DIMS, resized_dims, CFL_SIZE);

		long pos[2];
		for (unsigned int i = 0; i < 2; i++)
			pos[i] = labs((resized_dims[i] / 2) - (pdata->dims[i] / 2));

		md_copy_block(2, pos, resized_dims, u_resized, pdata->dims, (const complex float*)src, CFL_SIZE);
	}
	else{
		u_resized = md_alloc(DIMS, pdata->dims, CFL_SIZE);
		md_copy(DIMS, resized_dims, u_resized, (const complex float*)src, CFL_SIZE);
	}

	// slice FOV
	long slice_dims[DIMS];
	md_set_dims(DIMS, slice_dims, 1);

	slice_dims[0] = dom->dims[1];
	slice_dims[1] = dom->dims[2];

	unsigned int nx = (resized_dims[0] + slice_dims[0] - 1)/slice_dims[0];
	unsigned int ny = (resized_dims[1] + slice_dims[1] - 1)/slice_dims[1];

	slice_dims[2] = nx*ny;
	
	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);
	
	complex float* slices = md_alloc(DIMS, slice_dims, CFL_SIZE);
	complex float* tmp_slices = md_alloc(DIMS, slice_dims, CFL_SIZE);
	
	float scalor = calculate_max(DIMS, resized_dims, u_resized);
	
	scalor = scalor + 1e-06;
	md_zsmul(DIMS, resized_dims, u_resized, u_resized, 1. / scalor);

	int offset = 0;
	for (size_t i = 0; i < nx; i++)
	{
		for (size_t j = 0; j < ny; j++)
		{
			pos[0] = j*slice_dims[1];
			pos[1] = i*slice_dims[0];
			offset = (i*nx + j) * slice_dims[0]*slice_dims[1];
			md_copy_block(2, pos, slice_dims, tmp_slices + offset, resized_dims, u_resized, CFL_SIZE);
		}
	}
	
	md_transpose(pdata->N, 1, 0, slice_dims, slices, slice_dims, tmp_slices, CFL_SIZE);
	
	complex float* out = md_alloc(cod->N, cod->dims, cod->size);
	nlop_apply(pdata->tf_ops, cod->N, cod->dims, out, dom->N, dom->dims, slices);
	debug_printf(DP_DEBUG3, "\tLog P: %f  Scalor: %f  Step size: %f  lambda: %f\n ", creal(*out), scalor, pdata->step_size, lambda);

	//copy slices to feed tensor
	struct TF_Tensor ** input_tensor = get_input_tensor(pdata->tf_ops);
	md_copy(dom->N, dom->dims, TF_TensorData(*input_tensor), slices, CFL_SIZE);

	complex float* grad = md_alloc(dom->N, dom->dims, dom->size);
	complex float grad_ys = 1 + 1*I;

	nlop_adjoint(pdata->tf_ops, dom->N, dom->dims, grad, cod->N, cod->dims, &grad_ys); // grad [4, sx, sy]

	// update 

	md_zsmul(dom->N, dom->dims, grad, grad, pdata->step_size*lambda);
	complex float* rand_mask = md_alloc(dom->N, dom->dims, CFL_SIZE);

	md_rand_one(dom->N, dom->dims, rand_mask, pdata->p);
	md_zmul(dom->N, dom->dims, grad, grad, rand_mask);
	
	md_zsub(dom->N, dom->dims, slices, slices, grad);   // dst(src+1) = src - grad
	
	// back to fortran arrays
	complex float* tmp = md_alloc(DIMS, resized_dims, CFL_SIZE);
	complex float* tmp1 = md_alloc(DIMS, resized_dims, CFL_SIZE);
	for(size_t i=0; i < nx; i++)
	{
		for (size_t j=0; j < ny; j++)
		{
			pos[0] = i*slice_dims[0];
			pos[1] = j*slice_dims[1];
			offset = (i*nx + j) * slice_dims[0]*slice_dims[1];
			md_copy_block(2, pos, resized_dims, tmp, slice_dims, slices+offset, CFL_SIZE);
		}
	}
	md_zsmul(DIMS, resized_dims, tmp, tmp, scalor);

	md_transpose(DIMS, 1, 0, resized_dims, tmp1, resized_dims, tmp, CFL_SIZE);
	
	for (unsigned int i = 0; i < 2; i++)
		pos[i] = labs((resized_dims[i] / 2) - (pdata->dims[i] / 2));

	md_copy_block(2, pos, pdata->dims, (complex float*)dst, resized_dims, (const complex float*)tmp1, CFL_SIZE);
	
	md_free(u_resized);
	md_free(slices);
	md_free(tmp_slices);
	md_free(tmp);
}

static void prox_logp_nlinv_apply(const operator_data_t* data, float lambda, complex float* dst, const complex float* src)
{
	auto pdata = CAST_DOWN(prox_logp_nlinv_data, data);
	
	complex float* tmp = md_alloc(pdata->N, pdata->dims, CFL_SIZE);
	md_copy(pdata->N, pdata->dims, tmp, src, CFL_SIZE);
	
	float k = -1. * log(lambda*pdata->rho) / log(pdata->base);
	float alpha = lambda*pdata->rho;

	float cur = powf(pdata->base, k) / powf(pdata->base, (float)pdata->irgnm_steps-1.);

	float final = cur;

	debug_printf(DP_INFO, "--->step k %f cur %f alpha %f cur*alpha*k/max %f\n", k, cur, alpha, final);

	// cur*alpha*log(1+k/(float)pdata->irgnm_steps)/log(2.)

	for(unsigned int i=0; i<pdata->steps; i++)
	{
		prox_logp_nlinv_fun(data, final, dst, tmp);
		md_copy(pdata->N, pdata->dims, tmp, dst, CFL_SIZE);
	}
	
	md_free(tmp);	
}

static void prox_logp_nlinv_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_logp_nlinv_data, _data));
}

extern const struct operator_p_s* prox_logp_nlinv_create(unsigned int N, const long dims[__VLA(N)], const struct nlop_s * tf_ops, float step_size, float p, unsigned int steps, float base, unsigned int irgnm_steps, float rho)
{
	PTR_ALLOC(struct prox_logp_nlinv_data, pdata);
	SET_TYPEID(prox_logp_nlinv_data, pdata);

	pdata->tf_ops = tf_ops;
	pdata->step_size = step_size;
	pdata->p = p;
	
	pdata->N = N;
	pdata->dims = (long*)malloc(sizeof(long)*pdata->N);
	md_copy_dims(pdata->N, pdata->dims, dims);
	pdata->steps = steps;

	pdata->base = base;
	pdata->irgnm_steps = irgnm_steps;

	pdata->rho = rho;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_logp_nlinv_apply, prox_logp_nlinv_del);
}

/**
 * Data for computing prox_l2ball_fun:
 * Proximal function for f(z) = Ind{ || y - z ||_2 < eps }
 *
 * @param y y
 * @param eps
 * @param size size of z
 */
struct prox_l2ball_data {

	INTERFACE(operator_data_t);

	float* y;
	float eps;

	long size;
#ifdef USE_CUDA
	const float* gpu_y;
#endif
};

static DEF_TYPEID(prox_l2ball_data);


#ifdef USE_CUDA
static const float* get_y(const struct prox_l2ball_data* data, bool gpu)
{
	const float* y = data->y;

	if (gpu) {

		if (NULL == data->gpu_y)
			((struct prox_l2ball_data*)data)->gpu_y = md_gpu_move(1, MD_DIMS(data->size), data->y, FL_SIZE);

		y = data->gpu_y;
	}

	return y;
}
#endif

/**
 * Proximal function for f(z) = Ind{ || y - z ||_2 < eps }
 * Solution is y + (x - y) * q, where q = eps / norm(x - y) if norm(x - y) > eps, 1 o.w.
 *
 * @param prox_data should be of type prox_l2ball_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_l2ball_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	UNUSED(mu);
	auto pdata = CAST_DOWN(prox_l2ball_data, prox_data);

#ifdef USE_CUDA
	const float* y = get_y(pdata, cuda_ondevice(x_plus_u));
#else
	const float* y = pdata->y;
#endif

	if (NULL != y)
		md_sub(1, MD_DIMS(pdata->size), z, x_plus_u, y);
	else
		md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);

	float q1 = md_norm(1, MD_DIMS(pdata->size), z);

	if (q1 > pdata->eps)
		md_smul(1, MD_DIMS(pdata->size), z, z, pdata->eps / q1);

	if (NULL != y)
		md_add(1, MD_DIMS(pdata->size), z, z, y);
}

static void prox_l2ball_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_l2ball_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_l2ball_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(prox_l2ball_data, _data);
#ifdef USE_CUDA
	if (NULL != data->gpu_y)
		md_free(data->gpu_y);
#endif
	xfree(data);
}

const struct operator_p_s* prox_l2ball_create(unsigned int N, const long dims[N], float eps, const complex float* y)
{
	PTR_ALLOC(struct prox_l2ball_data, pdata);
	SET_TYPEID(prox_l2ball_data, pdata);

	pdata->y = (float*)y;
	pdata->eps = eps;
	pdata->size = md_calc_size(N, dims) * 2;

#ifdef USE_CUDA
	pdata->gpu_y = NULL;
#endif

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_l2ball_apply, prox_l2ball_del);
}




#if 0
/**
 * Data for computing prox_thresh_fun:
 * Proximal function for f(z) = lambda || z ||_1
 *
 * @param thresh function to apply SoftThresh
 * @param data data used by thresh function
 * @param lambda regularization
 */
struct prox_thresh_data {

	void (*thresh)(void* _data, float lambda, float* _dst, const float* _src);
	void* data;
	float lambda;
};

/**
 * Proximal function for f(z) = lambda || z ||_1
 * Solution is z = SoftThresh(x_plus_u, lambda * mu)
 *
 * @param prox_data should be of type prox_thresh_data
 */
void prox_thresh_fun(void* prox_data, float mu, float* z, const float* x_plus_u)
{
	struct prox_thresh_data* pdata = (struct prox_thresh_data*)prox_data;
	pdata->thresh(pdata->data, pdata->lambda * mu, z, x_plus_u);
}

static void prox_thresh_apply(const void* _data, float mu, complex float* dst, const complex float* src)
{
	prox_thresh_fun((void*)_data, mu, (float*)dst, (const float*)src);
}

static void prox_thresh_del(const void* _data)
{
	xfree((void*)_data);
}

const struct operator_p_s* prox_thresh_create(unsigned int N, const long dims[N], float lambda,
		void (*thresh)(void* _data, float lambda, float* _dst, const float* _src),
		void* data)
{
	PTR_ALLOC(struct prox_thresh_data, pdata);

	pdata->thresh = thresh;
	pdata->lambda = lambda;
	pdata->data = data;

	return operator_p_create(N, dims, dims, PTR_PASS(pdata), prox_thresh_apply, prox_thresh_del);
}
#endif


/**
 * Data for computing prox_zero_fun:
 * Proximal function for f(z) = 0
 *
 * @param size size of z
 */
struct prox_zero_data {

	INTERFACE(operator_data_t);

	long size;
};

static DEF_TYPEID(prox_zero_data);


/**
 * Proximal function for f(z) = 0
 * Solution is z = x_plus_u
 *
 * @param prox_data should be of type prox_zero_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_zero_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	UNUSED(mu);
	auto pdata = CAST_DOWN(prox_zero_data, prox_data);

	md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);
}

static void prox_zero_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_zero_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_zero_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_zero_data, _data));
}

const struct operator_p_s* prox_zero_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct prox_zero_data, pdata);
	SET_TYPEID(prox_zero_data, pdata);

	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_zero_apply, prox_zero_del);
}




/**
 * Data for computing prox_lineq_fun:
 * Proximal function for f(z) = 1{ A z = y }
 * Assumes AA^T = I
 * Solution is z = x - A^T A x + A^T y
 *
 * @param op linop A
 * @param adj A^H y
 * @param tmp tmp
 */
struct prox_lineq_data {

	INTERFACE(operator_data_t);

	const struct linop_s* op;
	complex float* adj;
	complex float* tmp;
};

static DEF_TYPEID(prox_lineq_data);

static void prox_lineq_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	auto pdata = CAST_DOWN(prox_lineq_data, _data);

	const struct linop_s* op = pdata->op;
	linop_normal(op, linop_domain(op)->N, linop_domain(op)->dims, pdata->tmp, src);

	md_zsub(linop_domain(op)->N, linop_domain(op)->dims, dst, src, pdata->tmp);
	md_zadd(linop_domain(op)->N, linop_domain(op)->dims, dst, dst, pdata->adj);
}

static void prox_lineq_del(const operator_data_t* _data)
{
	auto pdata = CAST_DOWN(prox_lineq_data, _data);

	md_free(pdata->adj);
	md_free(pdata->tmp);
	xfree(pdata);
}

const struct operator_p_s* prox_lineq_create(const struct linop_s* op, const complex float* y)
{
	PTR_ALLOC(struct prox_lineq_data, pdata);
	SET_TYPEID(prox_lineq_data, pdata);

	unsigned int N = linop_domain(op)->N;
	const long* dims = linop_domain(op)->dims;

	pdata->op = op;

	pdata->adj = md_alloc_sameplace(N, dims, CFL_SIZE, y);
	linop_adjoint(op, N, dims, pdata->adj, N, linop_codomain(op)->dims, y);

	pdata->tmp = md_alloc_sameplace(N, dims, CFL_SIZE, y);

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_lineq_apply, prox_lineq_del);
}


/**
 * Data for computing prox_ineq_fun:
 * Proximal function for f(z) = 1{ z <= b }
 *  and f(z) = 1{ z >= b }
 *
 * @param b b
 * @param size size of z
 */
struct prox_ineq_data {

	INTERFACE(operator_data_t);

	const float* b;
	float a;
	long size;
	bool positive;
};

static DEF_TYPEID(prox_ineq_data);

static void prox_ineq_fun(const operator_data_t* _data, float mu, float* dst, const float* src)
{
	UNUSED(mu);
	auto pdata = CAST_DOWN(prox_ineq_data, _data);

	if (NULL == pdata->b) {

		(pdata->positive ? md_smax : md_smin)(1, MD_DIMS(pdata->size), dst, src, pdata->a);
	} else {

		(pdata->positive ? md_max : md_min)(1, MD_DIMS(pdata->size), dst, src, pdata->b);
	}
}

static void prox_ineq_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_ineq_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_ineq_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_ineq_data, _data));
}

static const struct operator_p_s* prox_ineq_create(unsigned int N, const long dims[N], const complex float* b, float a, bool positive)
{
	PTR_ALLOC(struct prox_ineq_data, pdata);
	SET_TYPEID(prox_ineq_data, pdata);

	pdata->size = md_calc_size(N, dims) * 2;
	pdata->b = (const float*)b;
	pdata->a = a;
	pdata->positive = positive;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_ineq_apply, prox_ineq_del);
}


/*
 * Proximal function for less than or equal to:
 * f(z) = 1{z <= b}
 */
const struct operator_p_s* prox_lesseq_create(unsigned int N, const long dims[N], const complex float* b)
{
	return prox_ineq_create(N, dims, b, 0., false);
}

/*
 * Proximal function for greater than or equal to:
 * f(z) = 1{z >= b}
 */
const struct operator_p_s* prox_greq_create(unsigned int N, const long dims[N], const complex float* b)
{
	return prox_ineq_create(N, dims, b, 0., true);
}

/*
 * Proximal function for nonnegative orthant
 * f(z) = 1{z >= 0}
 */
const struct operator_p_s* prox_nonneg_create(unsigned int N, const long dims[N])
{
	return prox_ineq_create(N, dims, NULL, 0., true);
}

/*
 * Proximal function for greater than or equal to a scalar:
 * f(z) = 1{z >= a}
 */
const struct operator_p_s* prox_zsmax_create(unsigned int N, const long dims[N], float a)
{
	auto op_rvc = operator_p_bind_F(prox_rvc_create(N, dims), 0);
	auto op_p_ineq = prox_ineq_create(N, dims, NULL, a, true);

	return operator_p_pst_chain_FF(op_p_ineq, op_rvc);
}

struct prox_rvc_data {

	INTERFACE(operator_data_t);

	long size;
};

static DEF_TYPEID(prox_rvc_data);


static void prox_rvc_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	auto pdata = CAST_DOWN(prox_rvc_data, _data);

	md_zreal(1, MD_DIMS(pdata->size), dst, src);
}

static void prox_rvc_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(prox_rvc_data, _data));
}

/*
 * Proximal function for real-value constraint
 */
const struct operator_p_s* prox_rvc_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct prox_rvc_data, pdata);
	SET_TYPEID(prox_rvc_data, pdata);

	pdata->size = md_calc_size(N, dims);
	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_rvc_apply, prox_rvc_del);
}





struct auto_norm_s {

	INTERFACE(operator_data_t);

	long flags;
	const struct operator_p_s* op;
};

DEF_TYPEID(auto_norm_s);

static void auto_norm_apply(const operator_data_t* _data, float mu, complex float* y, const complex float* x)
{
	auto data = CAST_DOWN(auto_norm_s, _data);

	auto io = operator_p_domain(data->op);

	unsigned int N = io->N;

	long sdims[N];
	md_select_dims(N, ~data->flags, sdims, io->dims);

	long sstrs[N];
	md_calc_strides(N, sstrs, sdims, CFL_SIZE);

#if 0
	complex float* scale = md_alloc_sameplace(N, sdims, CFL_SIZE, x);

	md_zrss(N, io->dims, data->flags, scale, x);
	md_zdiv2(N, io->dims, io->strs, y, io->strs, x, sstrs, scale);
#else
	long pos[N];
	for (unsigned int i = 0; i < N; i++)
		pos[i] = 0;

	long xdims[N];
	md_select_dims(N, data->flags, xdims, io->dims);

	complex float* scale = md_alloc(N, sdims, CFL_SIZE);



	do {
		MD_ACCESS(N, sstrs, pos, scale) = md_znorm2(N, xdims, io->strs, &MD_ACCESS(N, io->strs, pos, x));

		complex float val = MD_ACCESS(N, sstrs, pos, scale);

		md_zsmul2(N, xdims, io->strs, &MD_ACCESS(N, io->strs, pos, y),
				io->strs, &MD_ACCESS(N, io->strs, pos, x),
				(0. == val) ? 0. : (1. / val));

	} while (md_next(N, io->dims, ~data->flags, pos));
#endif

	operator_p_apply_unchecked(data->op, mu, y, y);	// FIXME: input == output
#if 0
	md_zmul2(N, io->dims, io->strs, y, io->strs, y, sstrs, scale);
#else
	for (unsigned int i = 0; i < N; i++)
		pos[i] = 0;

	do {
		complex float val = MD_ACCESS(N, sstrs, pos, scale);

		md_zsmul2(N, xdims, io->strs, &MD_ACCESS(N, io->strs, pos, y),
				io->strs, &MD_ACCESS(N, io->strs, pos, y),
				val);

	} while (md_next(N, io->dims, ~data->flags, pos));
#endif

	md_free(scale);
}

static void auto_norm_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(auto_norm_s, _data);

	operator_p_free(data->op);

	xfree(data);
}



/* This functor normalizes data along given dimensions and undoes
 * the normalization after application of the operator.
 *
 */
const struct operator_p_s* op_p_auto_normalize(const struct operator_p_s* op, long flags)
{
	PTR_ALLOC(struct auto_norm_s, data);
	SET_TYPEID(auto_norm_s, data);

	data->flags = flags;
	data->op = operator_p_ref(op);

	auto io_in = operator_p_domain(op);
	auto io_out = operator_p_codomain(op);

	unsigned int N = io_in->N;
	long dims[N];
	md_copy_dims(N, dims, io_in->dims);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	assert(N == io_out->N);
	assert(md_check_compat(N, 0L, dims, io_out->dims));
	assert(md_check_compat(N, 0L, strs, io_in->strs));
	assert(md_check_compat(N, 0L, strs, io_out->strs));

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), auto_norm_apply, auto_norm_del);
}
