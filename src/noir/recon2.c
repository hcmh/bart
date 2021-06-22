/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
 * nonlinear inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/rand.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/iter4_lop.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/misc.h"

#include "linops/someops.h"


#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model2.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "nlops/nlop.h"

#include "recon2.h"


struct nlop_wrapper2_s {

	INTERFACE(struct iter_op_data_s);

	long split;

	int N;
	int idx;
	unsigned long flag;

	const long* col_dims;
};

DEF_TYPEID(nlop_wrapper2_s);


static void orthogonalize(iter_op_data* ptr, float* _dst, const float* _src)
{
#if 0
	noir_orthogonalize(nlop_get_data(CAST_DOWN(nlop_wrapper_s, ptr)->noir), (complex float*)_dst, (const complex float*)_src);
#else
	UNUSED(_src);

	auto nlw = CAST_DOWN(nlop_wrapper2_s, ptr);

	noir2_orthogonalize(nlw->N, nlw->col_dims, nlw->idx, nlw->flag, (complex float*) _dst + nlw->split);
#endif
}


const struct noir2_conf_s noir2_defaults = {

	.iter = 8,
	.rvc = false,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.a = 220.,
	.b = 32.,
	.sms = false,
	.sos = false,
	.enlive_flags = 0,
	.scaling = -1,
	.undo_scaling = false,

	.cgtol = -1.,

	.noncart = false,
	.nufft_conf = NULL,

	.regs = NULL,
	.admm_rho = 1.,

	.gpu = false,
	.cgiter = 30,
};


struct noir_irgnm_conf {

	struct iter3_irgnm_conf* irgnm_conf;
	enum algo_t algo;

	float rho;
	struct lsqr_conf *lsqr_conf;
};


static void noir_irgnm2(const struct noir_irgnm_conf* conf,
			const struct nlop_s* nlop, const struct linop_s* lop_fft,
			int NO, const long odims[NO], complex float* x, const complex float* ref,
			int NI, const long idims[NI], const complex float* data,
			int num_regs, const struct operator_p_s* thresh_ops[num_regs], const struct linop_s* trafos[num_regs],
			struct iter_op_s cb)
{
	struct lsqr_conf lsqr_conf = *(conf->lsqr_conf);

	const struct operator_p_s* pinv_op = NULL;

	const struct operator_s* op_der = operator_chain(nlop_get_derivative(nlop, 0, 0)->forward, lop_fft->normal);
	const struct operator_s* op_adj = operator_ref(nlop_get_derivative(nlop, 0, 0)->adjoint);

	const struct linop_s* lop_der = linop_from_ops(op_der, op_adj, NULL, NULL);

	operator_free(op_der);
	operator_free(op_adj);

	auto cod = nlop_codomain(nlop);
	auto dom = nlop_domain(nlop);

	long M = 2 * md_calc_size(cod->N, cod->dims);
	long N = 2 * md_calc_size(dom->N, dom->dims);

	assert(N == 2 * md_calc_size(NO, odims));
	assert(M == 2 * md_calc_size(NI, idims));

	enum algo_t algo = conf->algo;

	switch (algo) {

		case ALGO_CG: {

			struct iter_conjgrad_conf conjgrad_conf = iter_conjgrad_defaults;
			conjgrad_conf.maxiter = conf->irgnm_conf->cgiter;
			conjgrad_conf.tol = conf->irgnm_conf->cgtol;

			NESTED(void, lsqr_cont, (iter_conf* iconf))
			{
				auto conjgrad_conf = CAST_DOWN(iter_conjgrad_conf, iconf);
				conjgrad_conf->tol = conjgrad_conf->tol * powf(conjgrad_conf->INTERFACE.alpha, conf->irgnm_conf->cgtol_alpha_factor);
				debug_printf(DP_INFO, "\t cg_tol: %f\n", conjgrad_conf->tol);

				//double maxeigen = estimate_maxeigenval_sameplace(lop_der->normal, 30, data);
				//debug_printf(DP_INFO, "\t maxeigen: %f\n", maxeigen);

				//conjgrad_conf->INTERFACE.alpha *= maxeigen;
				//debug_printf(DP_INFO, "\t alpha: %f\n", conjgrad_conf->INTERFACE.alpha);

				debug_printf(DP_WARN, "CG is not tested/finalized for NLINV");

			};

			lsqr_conf.icont = lsqr_cont;

			pinv_op = lsqr2_create(&lsqr_conf, iter2_conjgrad, CAST_UP(&conjgrad_conf), NULL, lop_der, NULL, num_regs, NULL, NULL, NULL);

			iter4_lop_irgnm2(CAST_UP(conf->irgnm_conf),
					(struct nlop_s*)nlop, (struct linop_s*)lop_fft,
					N, (float*)x, (const float*)ref, M, (const float*)data,
					pinv_op,
					cb);

			break;
		}

		case ALGO_FISTA: {

			struct iter_fista_conf fista_conf = iter_fista_defaults;
			fista_conf.maxiter = conf->irgnm_conf->cgiter;
			fista_conf.maxeigen_iter = 20;
			fista_conf.tol = conf->irgnm_conf->cgtol;
			conf->irgnm_conf->cgtol_alpha_factor = 1.;
			fista_conf.continuation = 1.;

			NESTED(void, lsqr_cont, (iter_conf* iconf))
			{
				auto fconf = CAST_DOWN(iter_fista_conf, iconf);
				//fconf->tol = fconf->tol  * powf(fconf->INTERFACE.alpha, conf->irgnm_conf->cgtol_alpha_factor);

				//double maxeigen = estimate_maxeigenval_sameplace(lop_der->normal, 30, data);
				//debug_printf(DP_INFO, "\t maxeigen: %f\n", maxeigen);

				//fconf->INTERFACE.alpha *= maxeigen;
				//debug_printf(DP_INFO, "\t alpha: %f\n", fconf->INTERFACE.alpha);

				//static int outer_iter = 0; //FIXME: should be based on alpha?
				//fconf->maxiter = MIN(conf->irgnm_conf->cgiter, 3 * (int)powf(1.5, outer_iter++));

				//fconf->scale = MAX(0.2, fconf->INTERFACE.alpha);
				//fconf->step /= fconf->INTERFACE.alpha;

				debug_printf(DP_WARN, "FISTA is not tested/finalized for NLINV");
			};

			lsqr_conf.icont = lsqr_cont;

			pinv_op = lsqr2_create(&lsqr_conf, iter2_fista, CAST_UP(&fista_conf), NULL, lop_der, NULL, num_regs, thresh_ops, trafos, NULL);

			iter4_lop_irgnm2(CAST_UP(conf->irgnm_conf),
					(struct nlop_s*)nlop, (struct linop_s*)lop_fft,
					N, (float*)x, (const float*)ref, M, (const float*)data,
					pinv_op,
					cb);

			break;
		}

		case ALGO_ADMM: {

			struct iter_admm_conf admm_conf = iter_admm_defaults;
			admm_conf.maxiter = conf->irgnm_conf->cgiter;
			admm_conf.cg_eps = conf->irgnm_conf->cgtol;
			admm_conf.rho = conf->rho;

			NESTED(void, lsqr_cont, (iter_conf* iconf))
			{
				auto aconf = CAST_DOWN(iter_admm_conf, iconf);

				aconf->maxiter = MIN(admm_conf.maxiter, 10. * powf(2., ceil(logf(1. / iconf->alpha) / logf(conf->irgnm_conf->redu))));
				aconf->cg_eps = admm_conf.cg_eps  * powf(aconf->INTERFACE.alpha, conf->irgnm_conf->cgtol_alpha_factor);

				debug_printf(DP_WARN, "ADMM is not tested/finalized for NLINV");
			};

			lsqr_conf.icont = lsqr_cont;

			pinv_op = lsqr2_create(&lsqr_conf, iter2_admm, CAST_UP(&admm_conf), NULL, lop_der, NULL, num_regs, thresh_ops, trafos, NULL);

			iter4_lop_irgnm2(CAST_UP(conf->irgnm_conf),
					(struct nlop_s*)nlop, (struct linop_s*)lop_fft,
					N, (float*)x, (const float*)ref, M, (const float*)data,
					pinv_op,
					cb);

			break;
		}

		case ALGO_IST:
		case ALGO_NIHT:
		case ALGO_PRIDU:
		default:
			error("Algorithm not implemented!");

	}


	linop_free(lop_der);
	operator_p_free(pinv_op);
}


static const struct operator_p_s* flatten_prox_F(const struct operator_p_s* src)
{
	const struct operator_p_s* dst = operator_p_reshape_in_F(src, 1, MD_DIMS(md_calc_size(operator_p_domain(src)->N, operator_p_domain(src)->dims)));
	dst = operator_p_reshape_out_F(dst, 1, MD_DIMS(md_calc_size(operator_p_codomain(dst)->N, operator_p_codomain(dst)->dims)));
	return dst;
}

static const struct linop_s* flatten_linop_F(const struct linop_s* src)
{
	long dom_size = md_calc_size(linop_domain(src)->N, linop_domain(src)->dims);
	long cod_size = md_calc_size(linop_codomain(src)->N, linop_codomain(src)->dims);

	src = linop_reshape_in_F(src, 1, MD_DIMS(dom_size));
	src = linop_reshape_out_F(src, 1, MD_DIMS(cod_size));
	return src;
}

static const struct operator_p_s* stack_flatten_prox_F(const struct operator_p_s* prox_maps, const struct operator_p_s* prox_sens)
{
	auto prox1 = flatten_prox_F(prox_maps);
	auto prox2 = flatten_prox_F(prox_sens);
	auto prox3 = operator_p_stack(0, 0, prox1, prox2);
	operator_p_free(prox1);
	operator_p_free(prox2);
	return prox3;
}


static void opt_reg_noir_join_prox(int N, const long img_dims[N], const long col_dims[N],
				   int num_regs, const struct operator_p_s* prox_ops[num_regs], const struct linop_s* trafos[num_regs])
{
	long joined_dims[1] = { md_calc_size(N, img_dims) + md_calc_size(N, col_dims) };
	long img_dims_flat[1] = { md_calc_size(N, img_dims) };

	for (int i = 0; i < num_regs; i++) {

		if (linop_is_identity(trafos[i])) {

			prox_ops[i] = stack_flatten_prox_F(prox_ops[i], prox_zero_create(N, col_dims));
			trafos[i] = linop_identity_create(1, joined_dims);
		} else {

			prox_ops[i] = flatten_prox_F(prox_ops[i]);
			trafos[i] = linop_chain_FF(linop_expand_create(1, img_dims_flat, joined_dims), flatten_linop_F(trafos[i]));
		}
	}
}



static void noir2_recon(const struct noir2_conf_s* conf, struct noir2_s noir_ops,
			int N,
			const long img_dims[N], complex float* img, const complex float* img_ref,
			const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
			const long ksp_dims[N], const complex float* kspace)
{

	assert(N == (int)nlop_generic_codomain(noir_ops.nlop, 0)->N);
	long cim_dims[N];
	md_copy_dims(N, cim_dims, nlop_generic_codomain(noir_ops.nlop, 0)->dims);

	assert(md_check_equal_dims(N, img_dims, nlop_generic_domain(noir_ops.nlop, 0)->dims, ~0));
	assert(md_check_equal_dims(N, col_dims, nlop_generic_domain(noir_ops.nlop, 1)->dims, ~0));
	assert(md_check_equal_dims(N, ksp_dims, linop_codomain(noir_ops.lop_fft)->dims, ~0));

	unsigned long fft_flags = noir_ops.model_conf.fft_flags_noncart | noir_ops.model_conf.fft_flags_cart;


	complex float* data = md_alloc(N, cim_dims, CFL_SIZE);
	linop_adjoint(noir_ops.lop_fft, N, cim_dims, data, N, ksp_dims, kspace);


#ifdef USE_CUDA
	if(conf->gpu) {
		cuda_use_global_memory();

		complex float* tmp_data = md_alloc_gpu(N, cim_dims, CFL_SIZE);
		md_copy(N, cim_dims, tmp_data, data, CFL_SIZE);
		md_free(data);
		data = tmp_data;
	}
#else
	if(conf->gpu)
		error("Compiled without GPU support!");
#endif

	float scaling = conf->scaling;
	if (-1 == scaling) {

		scaling = 100. / md_znorm(N, cim_dims, data);
		if (conf->sms || conf->sos)
			scaling *= sqrt(cim_dims[SLICE_DIM]);
	}
	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(N, cim_dims, data, data, scaling);


	long skip = md_calc_size(N, img_dims);
	long size = skip + md_calc_size(N, col_dims);

	complex float* ref = NULL;
	assert((NULL == img_ref) == (NULL == sens_ref));

	if (NULL != img_ref) {

		ref = md_alloc_sameplace(1, MD_DIMS(size), CFL_SIZE, data);

		md_copy(N, img_dims, ref, img_ref, CFL_SIZE);
		md_copy(N, col_dims, ref + skip, sens_ref, CFL_SIZE);
	}

	complex float* x = md_alloc_sameplace(1, MD_DIMS(size), CFL_SIZE, data);

	md_copy(N, img_dims, x, img, CFL_SIZE);
	md_copy(N, col_dims, x + skip, ksens, CFL_SIZE);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = conf->cgtol;
	irgnm_conf.cgtol_alpha_factor = conf->cgtol_alpha_factor;
	irgnm_conf.nlinv_legacy = true;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.cgiter = conf->cgiter;


	struct nlop_s* nlop_flat = nlop_flatten(noir_ops.nlop);
	const struct linop_s* lop_fft_flat = linop_reshape_in(noir_ops.lop_fft, 1, MD_DIMS(md_calc_size(N, cim_dims)));

	struct nlop_wrapper2_s nlw;
	SET_TYPEID(nlop_wrapper2_s, &nlw);
	nlw.split = skip;
	nlw.N = N;
	nlw.idx = MAPS_DIM;
	nlw.flag = conf->enlive_flags;
	nlw.col_dims = col_dims;

	if ((NULL == conf->regs) || (0 == conf->regs->r)) {

		if (-1. == irgnm_conf.cgtol)
			irgnm_conf.cgtol = 0.1f;
		irgnm_conf.cgtol_alpha_factor = 0.;

		iter4_lop_irgnm(CAST_UP(&irgnm_conf),
				nlop_flat,
				(struct linop_s*)lop_fft_flat,
				size * 2, (float*)x, (const float*)ref,
				md_calc_size(N, cim_dims) * 2, (const float*)data,
				NULL,
				(struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });
	} else {

		if (-1. == irgnm_conf.cgtol)
			irgnm_conf.cgtol = 10.f;
		irgnm_conf.cgtol_alpha_factor = 2.;

		struct lsqr_conf lsqr_conf = lsqr_defaults;
		lsqr_conf.warmstart = true;

		int num_regs = conf->regs->r;

		complex float* lmask = md_alloc(1, MD_DIMS(size), CFL_SIZE);
		md_clear(1, MD_DIMS(size), lmask, CFL_SIZE);
		md_zfill(1, MD_DIMS(size - skip), lmask + skip, 1.); //l2-regularization for coils

		lsqr_conf.lambda_mask = lmask;
		lsqr_conf.lambda_scale = 1.;

		struct noir_irgnm_conf noir_irgnm_conf = {

			.irgnm_conf = &irgnm_conf,
			.algo = italgo_choose(num_regs, conf->regs->regs),
			.rho = conf->admm_rho,
			.lsqr_conf = &lsqr_conf,
		};

		const struct operator_p_s* prox_ops[NUM_REGS];
		const struct linop_s* trafos[NUM_REGS];

		opt_reg_configure(N, img_dims, conf->regs, prox_ops, trafos, 0, 1, conf->gpu);

		bool l2img = (L2IMG == conf->regs->regs[0].xform);

		if (l2img)
			md_zfill(1, MD_DIMS(skip), lmask, conf->regs->regs[0].lambda); //l2-regularization for images

		opt_reg_noir_join_prox(N, img_dims, col_dims, num_regs , prox_ops, trafos);

		noir_irgnm2(&noir_irgnm_conf, nlop_flat, lop_fft_flat,
			   1, MD_DIMS(size), x, ref,
			   1, MD_DIMS(md_calc_size(N, cim_dims)), data,
			   num_regs - (l2img ? 1 : 0), prox_ops + (l2img ? 1 : 0), trafos + (l2img ? 1 : 0),
			   (struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });

		opt_reg_free(conf->regs, prox_ops, trafos);
		md_free(lmask);
	}

	nlop_free(nlop_flat);
	linop_free(lop_fft_flat);

	md_free(data);

	md_copy(DIMS, img_dims, img, x, CFL_SIZE);
	md_copy(DIMS, col_dims, ksens, x + skip, CFL_SIZE);

	noir2_forw_coils(noir_ops.lop_coil2, x + skip, x + skip);
	md_copy(DIMS, col_dims, sens, x + skip, CFL_SIZE);	// needed for GPU
	fftmod(DIMS, col_dims, fft_flags, sens, sens);

	md_free(x);
	md_free(ref);

	if (conf->undo_scaling)
		md_zsmul(N, img_dims, img, img, 1./scaling);
}


void noir2_recon_noncart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	struct noir2_model_conf_s mconf = noir2_model_conf_defaults;

	mconf.fft_flags_noncart = FFT_FLAGS;
	mconf.fft_flags_cart = (conf->sms || conf->sos) ? SLICE_FLAG : 0;

	mconf.rvc = conf->rvc;
	mconf.sos = conf->sos;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.noncart = conf->noncart;

	mconf.nufft_conf = conf->nufft_conf;

	struct noir2_s noir_ops = noir2_noncart_create(N, trj_dims, traj, wgh_dims, weights, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &mconf);

	noir2_recon(conf, noir_ops, N, img_dims, img, img_ref, col_dims, sens, ksens, sens_ref, ksp_dims, kspace);

	noir2_free(&noir_ops);
}


void noir2_recon_cart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	struct noir2_model_conf_s mconf = noir2_model_conf_defaults;

	mconf.fft_flags_noncart = 0;
	mconf.fft_flags_cart = FFT_FLAGS | ((conf->sms || conf->sos) ? SLICE_FLAG : 0);

	mconf.rvc = conf->rvc;
	mconf.sos = conf->sos;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.noncart = conf->noncart;

	mconf.nufft_conf = conf->nufft_conf;


	struct noir2_s noir_ops = noir2_cart_create(N, pat_dims, pattern, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &mconf);

	noir2_recon(conf, noir_ops, N, img_dims, img, img_ref, col_dims, sens, ksens, sens_ref, ksp_dims, kspace);

	noir2_free(&noir_ops);
}
