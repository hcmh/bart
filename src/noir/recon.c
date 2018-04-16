/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2011-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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

#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/thresh.h"
#include "iter/italgos.h"

#include "noncart/nufft.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"
#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "recon.h"


struct nlop_wrapper_s {
	INTERFACE(struct iter_op_data_s);
	struct noir_s* noir;
	long split;
};
DEF_TYPEID(nlop_wrapper_s);


static void callback(iter_op_data* ptr, float* _dst, const float* _src)
{
	UNUSED(_src);
	struct nlop_wrapper_s* nlw = CAST_DOWN(nlop_wrapper_s, ptr);
	noir_dump(nlw->noir, (const complex float*) _dst, (const complex float*) _dst + nlw->split);
	noir_orthogonalize(nlw->noir, (complex float*) _dst + nlw->split);
}


static void nlop_dump(iter_op_data* ptr, int N, float* args[N])
{
	assert(2 == N);
	struct nlop_wrapper_s* nlw = CAST_DOWN(nlop_wrapper_s, ptr);
	noir_dump(nlw->noir, (const complex float*) args[1], (const complex float*) args[0]);
	noir_orthogonalize(nlw->noir, (complex float*) args[0]);
}


const struct noir_conf_s noir_defaults = {

	.iter = 8,
	.dims = {

			.ksp_dims = NULL,
			.traj_dims = NULL,
			.coil_imgs_dims = NULL,
			.sens_dims = NULL,
			.img_dims = NULL,
		},
	.rvc = false,
	.usegpu = false,
	.noncart = false,
	.nlinv_legacy = false,
	.alpha = 1.,
	.redu = 2.,
	.a = 220.,
	.b = 32.,
	.cgtol = 0.1f,
	.pattern_for_each_coil = false,
	.algo = 0,
	.out_im_steps = false,
	.out_coils_steps = false,
	.out_im = NULL,
	.out_coils = NULL,
};


static void print_noir_dims(const struct noir_dims_s* dims)
{
	debug_printf(DP_DEBUG1, "noir_dims_s:\n");

	if (NULL != dims->ksp_dims) {
		debug_printf(DP_DEBUG1, "ksp_dims:\n\t");
		debug_print_dims(DP_DEBUG1, DIMS, dims->ksp_dims);
	}

	if (NULL != dims->traj_dims) {
		debug_printf(DP_DEBUG1, "traj_dims:\n\t");
		debug_print_dims(DP_DEBUG1, DIMS, dims->traj_dims);
	}

	if (NULL != dims->coil_imgs_dims) {
		debug_printf(DP_DEBUG1, "coil_imgs_dims:\n\t");
		debug_print_dims(DP_DEBUG1, DIMS, dims->coil_imgs_dims);
	}


	if (NULL != dims->sens_dims) {
		debug_printf(DP_DEBUG1, "sens_dims:\n\t");
		debug_print_dims(DP_DEBUG1, DIMS, dims->sens_dims);
	}

	if (NULL != dims->img_dims) {
		debug_printf(DP_DEBUG1, "img_dims:\n\t");
		debug_print_dims(DP_DEBUG1, DIMS, dims->img_dims);
	}
}


void noir_recon(const struct noir_conf_s* conf, struct nufft_conf_s* nufft_conf, complex float* img, complex float* sens, const complex float* ref, const complex float* pattern, const complex float* mask, const complex float* kspace_data, const complex float* traj)
{

	unsigned int fft_flags = FFT_FLAGS|SLICE_FLAG;

	long skip = md_calc_size(DIMS, conf->dims.img_dims);
	long size = skip + md_calc_size(DIMS, conf->dims.sens_dims);
	long data_size = md_calc_size(DIMS, conf->dims.ksp_dims);


	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, conf->dims.img_dims, x, img, CFL_SIZE);
	md_copy(DIMS, conf->dims.sens_dims, x + skip, sens, CFL_SIZE);

	complex float* xref = NULL;
	if (NULL != ref) {

		xref = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);
		md_copy(1, d1, xref, ref, CFL_SIZE);
	}

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = conf->rvc;
	mconf.use_gpu = conf->usegpu;
	mconf.noncart = conf->noncart;
	mconf.fft_flags = fft_flags;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.pattern_for_each_coil = conf->pattern_for_each_coil;
	mconf.iter = conf->iter;
	mconf.out_im_steps = conf->out_im_steps;
	mconf.out_im = conf->out_im;
	mconf.out_coils_steps = conf->out_coils_steps;
	mconf.out_coils = conf->out_coils;


	struct noir_s nl;
	if (conf->algo != 3) // not altmin
		nl = noir_create(&conf->dims, mask, pattern, traj, nufft_conf, &mconf);
	else
		nl = noir_create2(&conf->dims, mask, pattern, traj, nufft_conf, &mconf);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = conf->cgtol;
	irgnm_conf.nlinv_legacy = conf->nlinv_legacy;

	struct nlop_wrapper_s nlw;
	SET_TYPEID(nlop_wrapper_s, &nlw);
	nlw.noir = &nl;
	if (conf->algo != 3)
		nlw.split = skip;
	else
		nlw.split = -1;

	switch(conf->algo) {
		case 0:
			debug_printf(DP_DEBUG2, "Using IRGNM\n");
			iter4_irgnm(CAST_UP(&irgnm_conf),
					nl.nlop,
					size * 2, (float*)x, (const float*)xref,
					data_size * 2, (const float*)kspace_data, (struct iter_op_s){ callback, CAST_UP(&nlw)});
			break;

		case 1:
			debug_printf(DP_DEBUG2, "Using Levenberg–Marquardt\n");
			iter4_levmar(CAST_UP(&irgnm_conf),
					nl.nlop,
					size * 2, (float*)x, (const float*)xref,
					data_size * 2, (const float*)kspace_data, (struct iter_op_s){ callback, CAST_UP(&nlw)});
			break;

		case 2:
			debug_printf(DP_DEBUG2, "Using Hybrid\n");
			iter4_irgnm_levmar_hybrid(CAST_UP(&irgnm_conf),
					nl.nlop,
					size * 2, (float*)x, (const float*)xref,
					data_size * 2, (const float*)kspace_data, (struct iter_op_s){ callback, CAST_UP(&nlw)});
			break;
		case 3:
			debug_printf(DP_DEBUG2, "Using Alternating Minimization\n");
			struct nlop_s* nl_perm = nlop_permute_inputs(nl.nlop, 2, (int[2]){(int)1, (int)0});
			complex float* im = x;
			complex float* coils = x + skip;
			iter4_altmin(CAST_UP(&irgnm_conf),
					nl_perm,
					2, (float*[2]){(float*) coils, (float*) im},
					data_size * 2, (const float*)kspace_data, (struct iter_nlop_s){ nlop_dump, CAST_UP(&nlw)});
			nlop_free(nl_perm);
			break;
	}

	md_copy(DIMS, conf->dims.img_dims, img, x, CFL_SIZE);

	if (NULL != sens) {

#ifdef USE_CUDA
		if (conf->usegpu) {

			noir_forw_coils(nl.linop, x + skip, x + skip);
			md_copy(DIMS, conf->dims.sens_dims, sens, x + skip, CFL_SIZE);
		} else
#endif
			noir_forw_coils(nl.linop, sens, x + skip);

		if (conf->algo != 3 && NULL == conf->dims.traj_dims)
			fftmod(DIMS, conf->dims.sens_dims, fft_flags, sens, sens);
	}

	noir_free(&nl);

	md_free(x);
	md_free(xref);
}
