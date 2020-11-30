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


#include "iter/vec.h"
#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/lsqr.h"

#include "reg_recon.h"
#include "optreg.h"
#include "grecon/italgo.h"



struct reg_nlinv {
	
	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;
	const struct irgnm_reg_conf* conf;

	long size_x;
	long size_y;

	const long *dims;

	float alpha;
	unsigned int outer_iter;

	const struct operator_p_s* prox_kernel;

};

DEF_TYPEID(reg_nlinv);


static void normal(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(reg_nlinv, _data);

	linop_normal_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)dst, (const complex float*)src);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, data->dims);
	long skip = md_calc_size(DIMS, img_dims)*2;

	long coil_dims[DIMS];
	md_copy_dims(DIMS, coil_dims, data->dims);
	
	auto dm = nlop_domain(data->nlop);

	if(data->conf->ropts->regs[0].xform == L1WAV)
		md_axpy(DIMS, coil_dims, dst + skip, data->alpha, src + skip);
	else if(data->conf->ropts->regs[0].xform == LOGP && data->outer_iter > data->conf->max_outiter-3)
		md_axpy(DIMS, coil_dims, dst + skip, data->alpha, src + skip);
	else
		md_axpy(DIMS, dm->dims, dst, data->alpha, src);

}

static void logp_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	auto data = CAST_DOWN(reg_nlinv, _data);

	if (data->outer_iter > data->conf->max_outiter-3)
	{
		operator_p_apply_unchecked(data->prox_kernel, rho, (_Complex float*)dst, (const complex float*)src);
	}

}

static void l1_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	auto data = CAST_DOWN(reg_nlinv, _data);
	operator_p_apply_unchecked(data->prox_kernel, rho, (_Complex float*)dst, (const complex float*)src);
}

static void fista_solver(iter_op_data* _data, float alpha,  float* dst, const float* src)
{
	auto data = CAST_DOWN(reg_nlinv, _data);
	
	data->alpha = alpha;
	float scale = alpha;

	double step = data->conf->step;

	void* x = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);
	double maxeigen = power(20, data->size_x, select_vecops(src), (struct iter_op_s){ normal, CAST_UP(data) }, x);
	md_free(x);
	step = step / maxeigen;

	unsigned int maxiter = MIN(data->conf->maxiter, 3 * (unsigned int)powf(1.5, data->outer_iter));
	debug_printf(DP_DEBUG3, "##reg. alpha = %f iteration %d step size %f \n", alpha, maxiter, step);
	
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);

	float eps = md_norm(1, MD_DIMS(data->size_x), tmp);
	
	typedef void prox_hooker(iter_op_data* _data, float rho, float* dst, const float* src);

	prox_hooker* prox_ptr = NULL;

	if (data->conf->ropts->regs[0].xform == L1WAV)
		prox_ptr = &l1_prox;

	if (data->conf->ropts->regs[0].xform == LOGP)
	{
		prox_ptr = &logp_prox;
		scale = powf(5, data->outer_iter)/powf(5, (data->conf->max_outiter)-1)*powf((float)data->outer_iter/((float)(data->conf->max_outiter)-1), 5.);
		printf("--->>> scale value %f outer_iter %d max outer_iter%d\n", scale, data->outer_iter, data->conf->max_outiter);
	}

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = scale;
	};

	fista(maxiter, data->conf->tol * alpha * eps, step,
		data->size_x,
		select_vecops(src),
		continuation,
		(struct iter_op_s){ normal, CAST_UP(data) },
		(struct iter_op_p_s){ prox_ptr, CAST_UP(data) },
		dst, tmp, NULL);

	md_free(tmp);

	data->outer_iter++;

}

struct reg_nlinv_s {

	INTERFACE(operator_data_t);
	struct reg_nlinv ins_reg_nlinv;
};

DEF_TYPEID(reg_nlinv_s);

static void reg_nlinv_apply(const operator_data_t* _data, float alpha, complex float* dst, const complex float* src)
{
	const auto data = &CAST_DOWN(reg_nlinv_s, _data)->ins_reg_nlinv;
	fista_solver(CAST_UP(data), alpha, (float*)dst, (float*)src);
}

static void reg_nlinv_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(reg_nlinv_s, _data);

	nlop_free(data->ins_reg_nlinv.nlop);

	xfree(data->ins_reg_nlinv.dims);
	xfree(data);
}

extern const struct operator_p_s* reg_pinv_op_create(struct irgnm_reg_conf* conf, const long dims[DIMS], struct nlop_s* nlop, const struct operator_p_s** thresh_ops, const struct linop_s** trafos)
{

	const struct operator_p_s* pinv_op = NULL;
	const struct operator_p_s* prox_kernel = thresh_ops[0];
	
	if(conf->algo == ALGO_FISTA)
	{
        PTR_ALLOC(struct reg_nlinv_s, data);
	    SET_TYPEID(reg_nlinv_s, data);
    	SET_TYPEID(reg_nlinv, &data->ins_reg_nlinv);

        auto cd = nlop_codomain(nlop);
	    auto dm = nlop_domain(nlop);

	    int M = 2 * md_calc_size(cd->N, cd->dims);
	    int N = 2 * md_calc_size(dm->N, dm->dims);
	
		printf("use fista\n");
		long* ndims = *TYPE_ALLOC(long[DIMS]);
		md_copy_dims(DIMS, ndims, dims);
		struct reg_nlinv ins_reg_nlinv={
			{&TYPEID(reg_nlinv)},
			.nlop = nlop_clone(nlop),
			.conf=conf,
			.size_x = N,
			.size_y = M,
			.dims = ndims,
			.alpha = 1.0,
			.outer_iter = 0,
			.prox_kernel = prox_kernel,
		};
		data->ins_reg_nlinv = ins_reg_nlinv;
		
		pinv_op = operator_p_create(dm->N, dm->dims, cd->N, cd->dims, CAST_UP(PTR_PASS(data)), reg_nlinv_apply, reg_nlinv_del);
	}
	else
	{
		
		printf("use admm\n");
		
		pinv_op = lsqr2_create(&lsqr_defaults, 
								iter2_admm,
								CAST_UP(conf->admm_conf_reg),
								NULL, true, &nlop->derivative[0][0],
								NULL, conf->ropts->r, thresh_ops, trafos, NULL);

	}

	return pinv_op;
}
