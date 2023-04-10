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

#include "num/ops_p.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"


#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/thresh.h"
#include "iter/italgos.h"
#include "iter/lsqr.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "nlops/nlop.h"

#include "optreg.h"
#include "grecon/italgo.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "reg_recon.h"
#include "recon.h"


struct nlop_wrapper_s {

	INTERFACE(struct iter_op_data_s);

	struct noir_s* noir;
	long split;
};

DEF_TYPEID(nlop_wrapper_s);


static void orthogonalize(iter_op_data* ptr, float* _dst, const float* _src)
{
#if 0
	noir_orthogonalize(nlop_get_data(CAST_DOWN(nlop_wrapper_s, ptr)->noir), (complex float*)_dst, (const complex float*)_src);
#else
	UNUSED(_src);

	auto nlw = CAST_DOWN(nlop_wrapper_s, ptr);

	noir_orthogonalize(nlw->noir, (complex float*) _dst + nlw->split);
#endif
}


const struct noir_conf_s noir_defaults = {

	.iter = 8,
	.rvc = false,
	.noncart = false,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.a = 220.,
	.b = 32.,
	.inner_iter = 100.,
	.pattern_for_each_coil = false,
	.sms = false,
	.cnstcoil_flags = 0u,
	.img_space_coils = false,
	.algo = ALGO_FISTA,
	.rho = 0.01, // not used
	.step = 0.9, // compute step size in FISTA?
	.tol = 0.001,
	.shift_mode=1,
	.wtype_str="dau2",
};


void noir_recon(const struct noir_conf_s* conf, const long dims[DIMS], complex float* img, complex float* sens, complex float* ksens, const complex float* ref, const complex float* pattern, const complex float* mask, const complex float* kspace_data)
{
	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = conf->rvc;
	mconf.noncart = conf->noncart;
	mconf.fft_flags = FFT_FLAGS;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.ptrn_flags = ~(MAPS_FLAG|COIL_FLAG);
	mconf.cnstcoil_flags = conf->cnstcoil_flags;

	if (conf->sms)
		mconf.fft_flags |= SLICE_FLAG;

	if (conf->pattern_for_each_coil)
		mconf.ptrn_flags |= COIL_FLAG;


	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];


	md_select_dims(DIMS, ~COIL_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, ~mconf.cnstcoil_flags, coil_dims, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, data_dims, dims);

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);

	md_copy(DIMS, coil_dims, x + skip, ksens, CFL_SIZE);

	complex float* xref = NULL;

#if 1
	struct noir_s nl = noir_create(dims, mask, pattern, &mconf);
#else
	struct noir_s nl = noir_create3(dims, mask, pattern, &mconf);
	nl.nlop = nlop_flatten(nl.nlop);
#endif

	if (NULL != ref) {

		xref = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

		if (conf->img_space_coils) { // transform coils back to k-space

			complex float* ref_buf = md_alloc(1, d1, CFL_SIZE);

			md_copy(1, d1, ref_buf, ref, CFL_SIZE);

			ifftmod(DIMS, coil_dims, mconf.fft_flags, ref_buf + skip, ref + skip);
			noir_back_coils(nl.linop, ref_buf + skip, ref_buf + skip);

			md_copy(1, d1, xref, ref_buf, CFL_SIZE);

			md_free(ref_buf);

		} else  {

			md_copy(1, d1, xref, ref, CFL_SIZE);
		}
	}

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.reg_iter = conf->reg_iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = 0.1f;
	irgnm_conf.cgiter = conf->inner_iter;
	irgnm_conf.nlinv_legacy = true;
	irgnm_conf.alpha_min = conf->alpha_min;


	struct nlop_wrapper_s nlw;

	SET_TYPEID(nlop_wrapper_s, &nlw);

	nlw.noir = &nl;
	nlw.split = skip;

	struct opt_reg_s ropts = conf->ropts;

	const struct operator_p_s* pinv_op = NULL;

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, mconf.fft_flags|MAPS_FLAG|TIME_FLAG, irgnm_conf_dims, dims);

	irgnm_conf_dims[COIL_DIM] = dims[COIL_DIM];

	debug_printf(DP_INFO, "imgs_dims:\n\t");
	debug_print_dims(DP_INFO, DIMS, imgs_dims);

	// initialize prox functions

    const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	opt_reg_nlinv_configure(DIMS, irgnm_conf_dims, &ropts, thresh_ops, trafos, conf->shift_mode, conf->wtype_str);

	struct iter_admm_conf admm_conf_reg = iter_admm_defaults;
	
	admm_conf_reg.maxiter = conf->inner_iter;
	admm_conf_reg.rho = conf->rho; 
	
	struct lsqr_conf lsqr_conf_reg = lsqr_defaults;
	lsqr_conf_reg.it_gpu = false;
	lsqr_conf_reg.warmstart = true;

	
	struct irgnm_reg_conf reg_conf = {
		.irgnm_conf = &irgnm_conf,
		.ropts = &ropts,
		.algo = conf->algo,
		.step = conf->step,
		.rho = ropts.lambda,  //  FIXME: it should be named with lambda as it is a parameter for regularization strength
		.maxiter = conf->inner_iter,
		.tol = conf->tol,
		.max_outiter = conf->iter,
		.shift_mode = conf->shift_mode,
		.admm_conf_reg = &admm_conf_reg,
		.lsqr_conf_reg = &lsqr_conf_reg,
	};

	// default solver cg
	if (ropts.r > 0){
		pinv_op = reg_pinv_op_create(&reg_conf, irgnm_conf_dims, nl.nlop, thresh_ops, trafos);
		iter4_irgnm3(CAST_UP(&irgnm_conf),
			nl.nlop,
			size * 2, (float*)x, (const float*)xref,
			data_size * 2, (const float*)kspace_data,
			pinv_op,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });
	}
	else 
		iter4_irgnm(CAST_UP(&irgnm_conf),
			nl.nlop,
			size * 2, (float*)x, (const float*)xref,
			data_size * 2, (const float*)kspace_data,
			NULL,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });

	opt_reg_nlinv_free(&ropts, thresh_ops, trafos);

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);
	md_copy(DIMS, coil_dims, ksens, x + skip, CFL_SIZE);


	noir_forw_coils(nl.linop, x + skip, x + skip);
	md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);	// needed for GPU
	fftmod(DIMS, coil_dims, mconf.fft_flags, sens, sens);


	nlop_free(nl.nlop);
	operator_p_free(pinv_op);
	md_free(x);
	md_free(xref);
}


