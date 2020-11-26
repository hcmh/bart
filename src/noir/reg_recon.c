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

    long *dims;

    float alpha;
	int outer_iter;

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
    
    md_axpy(DIMS, coil_dims, dst + skip, data->alpha, src + skip);
}

static void logp_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
    auto data = CAST_DOWN(reg_nlinv, _data);
    printf("run logp_prox \n");
    /*data crop*/
    long img_dims[DIMS];
    md_select_dims(DIMS, ~COIL_FLAG, img_dims, data->dims);

    long resized_dims[DIMS];
    md_select_dims(DIMS, ~COIL_FLAG, resized_dims, data->dims);
    complex float* u_resized = NULL;

    if(img_dims[0]>256)
    {

        resized_dims[0] = img_dims[0]/2;
        resized_dims[1] = img_dims[1]/2;

        u_resized = md_alloc(DIMS, resized_dims, CFL_SIZE);

        long pos[2];
	    for (unsigned int i = 0; i < 2; i++)
		    pos[i] = labs((resized_dims[i] / 2) - (img_dims[i] / 2));

	    md_copy_block(2, pos, resized_dims, u_resized, img_dims, (const complex float*)src, CFL_SIZE);

    }
    else{
        u_resized = md_alloc(DIMS, img_dims, CFL_SIZE);
        md_copy(DIMS, resized_dims, u_resized, (const complex float*)src, CFL_SIZE);
    }

    operator_p_apply_unchecked(data->prox_kernel, rho, (_Complex float*)u_resized, (const complex float*)u_resized);

    long pos[2];
	for (unsigned int i = 0; i < 2; i++)
	    pos[i] = labs((resized_dims[i] / 2) - (img_dims[i] / 2));

	md_copy_block(2, pos, img_dims, (complex float*)dst, resized_dims, (const complex float*)u_resized, CFL_SIZE);
    
    md_free(u_resized);
}

static void l1_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
    auto data = CAST_DOWN(reg_nlinv, _data);
    printf("run l1 prox \n");
    operator_p_apply_unchecked(data->prox_kernel, rho, (_Complex float*)dst, (const complex float*)src);
}

static void fista_solver(iter_op_data* _data, float alpha,  float* dst, const float* src)
{
    auto data = CAST_DOWN(reg_nlinv, _data);
	
	data->alpha = alpha;
	void* x = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);
	double maxeigen = power(20, data->size_x, select_vecops(src), (struct iter_op_s){ normal, CAST_UP(data) }, x);
	md_free(x);

	double step = data->conf->step / maxeigen;

	int maxiter = MIN(data->conf->maxiter, 3 * (int)powf(1.5, data->outer_iter));
	debug_printf(DP_DEBUG3, "##reg. alpha = %f %d \n", alpha, maxiter);
    
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->size_x), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);

	float eps = md_norm(1, MD_DIMS(data->size_x), tmp);

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = data->alpha;
	};
    
    typedef void prox_hooker(iter_op_data* _data, float rho, float* dst, const float* src);

    prox_hooker* prox_ptr = NULL;

    if (data->conf->ropts->regs[0].xform == L1WAV)
        prox_ptr = &l1_prox;
    if (data->conf->ropts->regs[0].xform == LOGP)
        prox_ptr = &logp_prox;

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
    debug_printf(DP_INFO, "--> reg_nlinv_apply reached\n");
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
    PTR_ALLOC(struct reg_nlinv_s, data);
    SET_TYPEID(reg_nlinv_s, data);
    SET_TYPEID(reg_nlinv, &data->ins_reg_nlinv);

    auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);
    debug_printf(DP_INFO, "in dims: ");
    debug_print_dims(DP_INFO, dm->N, dm->dims);
    debug_printf(DP_INFO, "out dims: ");
    debug_print_dims(DP_INFO, cd->N, cd->dims);

    int M = 2 * md_calc_size(cd->N, cd->dims);
	int N = 2 * md_calc_size(dm->N, dm->dims);
    
    const struct operator_p_s* pinv_op = NULL;
    const struct operator_p_s* prox_kernel = thresh_ops[0];
    

    if(conf->algo == ALGO_FISTA)
    {
        printf("use fista\n");
        long* ndims = *TYPE_ALLOC(long[DIMS]);
	    md_copy_dims(DIMS, ndims, dims);
        struct reg_nlinv ins_reg_nlinv={
            {&TYPEID(reg_nlinv)},
            nlop_clone(nlop),
            conf,
            N,M,
            ndims,
            1.0,
			0,
            prox_kernel,
        };
        data->ins_reg_nlinv = ins_reg_nlinv;
        
        pinv_op = operator_p_create(dm->N, dm->dims, cd->N, cd->dims, CAST_UP(PTR_PASS(data)), reg_nlinv_apply, reg_nlinv_del);
    }
    else
    {
        
		printf("use admm\n");
		struct iter_admm_conf admm_conf_reg = iter_admm_defaults;

		admm_conf_reg.maxiter = conf->maxiter;
		admm_conf_reg.cg_eps = conf->irgnm_conf->cgtol;
		admm_conf_reg.rho = conf->rho;
		admm_conf_reg.use_interface_alpha = true;
		
		pinv_op = lsqr2_create(&lsqr_defaults, 
								iter2_admm,
								CAST_UP(&admm_conf_reg),
								NULL, true, &nlop->derivative[0][0],
								NULL, conf->ropts->r, thresh_ops, trafos, NULL);

    }
    
    return pinv_op;
}
