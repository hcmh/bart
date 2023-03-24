/* Copyright 2014-2017. The Regents of the University of California.
 * Copyright 2016-2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017	Jon Tamir <jtamir@eecs.berkeley.edu>
 * 2016-2021	Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/ops.h"
#include "num/ops_p.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "iter/iter.h"
#include "noir/recon.h"

#include "nlops/nlop.h"

#include "prox2.h"



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



/*
 * proximal function for a differentiable f(x) given \grad f(x)
 *
 */
struct prox_nlgrad_data {

	INTERFACE(operator_data_t);

	const struct nlop_s* op;

	float step_size;
	float lambda;
	int steps;
};

DEF_TYPEID(prox_nlgrad_data);

static void prox_nlgrad_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(prox_nlgrad_data, _data);

	mu *= data->lambda;

	auto dom = nlop_domain(data->op);
	auto cod = nlop_codomain(data->op);

	complex float* grd = md_alloc_sameplace(dom->N, dom->dims, dom->size, dst);

	complex float out[1];
	complex float grad_ys[1] = { 1. };

	md_copy(dom->N, dom->dims, dst, src, dom->size);

	for (int i = 0; i < data->steps; i++) {

		nlop_apply(data->op, cod->N, cod->dims, out, dom->N, dom->dims, dst);

		debug_printf(DP_DEBUG1, "Loss: %f\n", crealf(out[0]));

		nlop_adjoint(data->op, dom->N, dom->dims, grd, cod->N, cod->dims, grad_ys);

		if (0 < i) {

			md_zaxpy(dom->N, dom->dims, dst, -1. * data->step_size, dst);
			md_zaxpy(dom->N, dom->dims, dst, +1. * data->step_size, src);
		}

		md_zaxpy(dom->N, dom->dims, dst, -1. * data->step_size * mu, grd);		// xp <- x - s (x - y + l \grad f)
	}

	md_free(grd);
}

static void prox_nlgrad_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(prox_nlgrad_data, _data);

	nlop_free(data->op);

	xfree(data);
}

extern const struct operator_p_s* prox_nlgrad_create(const struct nlop_s* op, int steps, float step_size, float lambda)
{
	PTR_ALLOC(struct prox_nlgrad_data, data);
	SET_TYPEID(prox_nlgrad_data, data);

	auto dom = nlop_domain(op);
	auto cod = nlop_codomain(op);

	assert(CFL_SIZE == dom->size);
	assert(CFL_SIZE == cod->size);
	assert(1 == md_calc_size(cod->N, cod->dims));

	data->op = nlop_clone(op);
	data->lambda = lambda;
	data->step_size = step_size;
	data->steps = steps;

	return operator_p_create(dom->N, dom->dims, dom->N, dom->dims, CAST_UP(PTR_PASS(data)), prox_nlgrad_apply, prox_nlgrad_del);
}

/*
 * proximal function with a diffusion prior
*/
struct prox_nl_dp_grad_data{

	INTERFACE(operator_data_t);

	const struct operator_p_s* op;
	const struct dp_conf* conf;

	float lambda;
	float* sigmas;

	int idx;
};

DEF_TYPEID(prox_nl_dp_grad_data);

static void prox_nl_dp_grad_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(prox_nl_dp_grad_data, _data);

	mu *= data->lambda;

	auto dom = operator_p_domain(data->op);

	int cur_iter = data->idx%data->conf->iter;
	float cond_t = (float)data->conf->start_step/(float)data->conf->T - (float)(cur_iter)/(float)data->conf->T;
	int cur_idx = data->conf->T - data->conf->start_step + cur_iter;

	complex float* score = md_alloc_sameplace(dom->N, dom->dims, dom->size, dst);
	operator_p_apply_unchecked(data->op, cond_t, score, src);

	float dsig = data->sigmas[cur_idx + 1] - data->sigmas[cur_idx];

	md_zaxpy(dom->N, dom->dims, dst, mu * dsig, score); //dst = dst - mu*dsig*score
	data->idx = data->idx + 1;

	debug_printf(DP_DEBUG1, "prox iter: %d, t-> %.4f mu -> %.4f \n", cur_iter, cond_t, mu);
}

static void prox_nl_dp_grad_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(prox_nl_dp_grad_data, _data);

	operator_p_free(data->op);

	xfree(data);
}

extern const struct operator_p_s* prox_nl_dp_grad_create(const struct operator_p_s* op, const struct dp_conf* conf, float lambda)
{
	PTR_ALLOC(struct prox_nl_dp_grad_data, data);
	SET_TYPEID(prox_nl_dp_grad_data, data);

	auto dom = operator_p_domain(op);

	/*assert(CFL_SIZE == dom->size);
	assert(CFL_SIZE == cod->size);
	assert(1 == md_calc_size(cod->N, cod->dims));*/

	data->op = op;
	data->lambda = lambda;
	data->conf = conf;
	data->sigmas = malloc(sizeof(float)* conf->T);
	data->idx = 0;

	for (int i = 0; i < conf->T + 1; i++)
		data->sigmas[i] = powf(exp(log(conf->sigma_min) + i * ((log(conf->sigma_max) - log(conf->sigma_min)) / (conf->T))), 2);

	return operator_p_create(dom->N, dom->dims, dom->N, dom->dims, CAST_UP(PTR_PASS(data)), prox_nl_dp_grad_apply, prox_nl_dp_grad_del);
}



struct auto_norm_s {

	INTERFACE(operator_data_t);

	enum norm norm;

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


	typeof(md_znorm2)* norm2[] = {
		[NORM_L2] = md_znorm2,
		[NORM_MAX] = md_zmaxnorm2,
	};


	do {
		MD_ACCESS(N, sstrs, pos, scale) = norm2[data->norm](N, xdims, io->strs, &MD_ACCESS(N, io->strs, pos, x));

		complex float val = MD_ACCESS(N, sstrs, pos, scale);

		debug_printf(DP_DEBUG4, "auto normalize %f\n", crealf(val));

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
const struct operator_p_s* op_p_auto_normalize(const struct operator_p_s* op, long flags, enum norm norm)
{
	PTR_ALLOC(struct auto_norm_s, data);
	SET_TYPEID(auto_norm_s, data);

	data->flags = flags;
	data->op = operator_p_ref(op);
	data->norm = norm;

	auto io_in = operator_p_domain(op);
	auto io_out = operator_p_codomain(op);

	int N = io_in->N;
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


struct auto_resize_s {

	INTERFACE(operator_data_t);

	long resize_dims[DIMS];
	long img_dims[DIMS];
	long pat_dims[DIMS];
	const struct operator_p_s* op;
};

DEF_TYPEID(auto_resize_s);

static void pad_noise(const operator_data_t* _data, complex float*y, const complex float* x)
{
	auto data = CAST_DOWN(auto_resize_s, _data);
	complex float* tmp1 = md_alloc(DIMS, data->resize_dims, sizeof(complex float));
	complex float* tmp2 = md_alloc(DIMS, data->resize_dims, sizeof(complex float));
	complex float* pat = md_alloc(DIMS, data->pat_dims, sizeof(complex float));

	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);

	md_copy_block(DIMS, pos, data->pat_dims, pat, data->img_dims, x, CFL_SIZE);
	complex float avg;
	complex float std;

	md_zavg(DIMS, data->pat_dims, 7, &avg, pat);
	md_zstd(DIMS, data->pat_dims, 7, &std, pat);

	md_gaussian_rand(DIMS, data->resize_dims, tmp1);
	md_zsmul(DIMS, data->resize_dims, tmp1, tmp1, std);

	md_zfill(DIMS, data->resize_dims, tmp2, 1.);
	md_zsmul(DIMS, data->resize_dims, tmp2, tmp2, avg);
	md_zadd(DIMS, data->resize_dims, y, tmp1, tmp2);
	
	md_free(tmp1);
	md_free(tmp2);
	md_free(pat);
}

static bool check_dims(const operator_data_t* _data){

	auto data = CAST_DOWN(auto_resize_s, _data);
	
	for (int i=0 ; i<3; i++)
	{
		if (data->resize_dims[i] != data->img_dims[i])
		{
			debug_printf(DP_DEBUG1, "tf input dims does not match with img dims, will be resized. ");
			return false;
		}	
	}
	debug_printf(DP_DEBUG1, "tf input dims match with img dims, resizing is not needed. ");
	return true;
}

static void auto_resize_apply(const operator_data_t* _data, float mu, complex float* y, const complex float* x)
{
	auto data = CAST_DOWN(auto_resize_s, _data);

	if ( !check_dims(_data)) {

	complex float* tmp1 = md_alloc(DIMS, data->resize_dims, sizeof(complex float));
	
	long pos[DIMS];
	pad_noise(_data, tmp1, x);

	for (int i = 0; i < DIMS; i++)
		pos[i] = labs((data->resize_dims[i] / 2) - (data->img_dims[i] / 2));
	
	md_copy_block(DIMS, pos, data->resize_dims, tmp1, data->img_dims, x, CFL_SIZE);
	operator_p_apply_unchecked(data->op, mu, tmp1, tmp1);
	md_resize_center(DIMS, data->img_dims, y, data->resize_dims, tmp1, CFL_SIZE);

	}

	else{
		operator_p_apply_unchecked(data->op, mu, y, x);
	}
}

static void auto_resize_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(auto_resize_s, _data);

	operator_p_free(data->op);

	xfree(data);
}



/* This functor normalizes data along given dimensions and undoes
 * the normalization after application of the operator.
 *
 */
const struct operator_p_s* op_p_auto_resize(const struct operator_p_s* op, int N, const long resize_dims[N], const long img_dims[N])
{
	PTR_ALLOC(struct auto_resize_s, data);
	SET_TYPEID(auto_resize_s, data);

	assert(N<=DIMS);
	md_copy_dims(N, data->resize_dims, resize_dims);
	md_copy_dims(N, data->img_dims, img_dims);
	md_set_dims(N, data->pat_dims, 1);
	
	if (data->img_dims[0] == 1){
		data->pat_dims[1] = 25;
		data->pat_dims[2] = 25;
	}
	else{
		data->pat_dims[0] = 25;
		data->pat_dims[1] = 25;
	}
		
	data->op = operator_p_ref(op);
	
	return operator_p_create(N, img_dims, N, img_dims, CAST_UP(PTR_PASS(data)), auto_resize_apply, auto_resize_del);
}


const struct operator_p_s* op_p_conjugate(const struct operator_p_s* op, const struct linop_s* lop)
{
	return operator_p_pst_chain(operator_p_pre_chain(lop->forward, op), lop->adjoint);
}


