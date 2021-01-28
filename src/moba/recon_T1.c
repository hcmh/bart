/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "iter/iter3.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"
#include "noir/recon.h"

#include "moba/model_T1.h"
#include "moba/iter_l1.h"
#include "moba/T1_alpha.h"
#include "moba/T1_alpha_in.h"
#include "moba/moba.h"

#include "recon_T1.h"




void T1_recon(const struct moba_conf* conf, const long dims[DIMS], complex float* img, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* TI, const complex float* TI_t1relax, const complex float* kspace_data, bool usegpu)
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG|TIME_FLAG|TIME2_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|TE_FLAG|MAPS_FLAG|TIME_FLAG|TIME2_FLAG, data_dims, dims);


	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);
	md_copy(DIMS, coil_dims, x + skip, sens, CFL_SIZE);


	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = false;
	mconf.noncart = conf->noncartesian;
	mconf.fft_flags = fft_flags;
	mconf.a = 880.;
	mconf.b = 32.;
	mconf.cnstcoil_flags = TE_FLAG;

	// FIXME: Move all function arguments to struct to simplify adding new ones
	struct T1_s nl = T1_create(dims, mask, TI, pattern, &mconf, conf->MOLLI, TI_t1relax, conf->IR_SS, conf->IR_phy, conf->input_alpha, usegpu);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.cgtol = ((2 == conf->opt_reg) || (conf->auto_norm_off)) ? 1e-3 : conf->tolerance;
	irgnm_conf.cgiter = conf->inner_iter;
	irgnm_conf.nlinv_legacy = true;

	struct opt_reg_s* ropts = conf->ropts;

	struct mdb_irgnm_l1_conf conf2 = {
		.c2 = &irgnm_conf,
		.opt_reg = conf->opt_reg,
		.step = conf->step,
		.lower_bound = conf->lower_bound,
		.constrained_maps = 4,
		.not_wav_maps = (0. == conf->IR_phy) ? 0 : 1,
		.flags = FFT_FLAGS,
		.usegpu = usegpu,
		.algo = conf->algo,
		.rho = conf->rho,
		.ropts = ropts,
		.wav_reg = 1,
		.auto_norm_off = conf->auto_norm_off };

	if (conf->MOLLI || (0. != conf->IR_phy)|| (NULL != conf->input_alpha))
		conf2.constrained_maps = 2;

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, fft_flags|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, irgnm_conf_dims, imgs_dims);

	irgnm_conf_dims[COIL_DIM] = coil_dims[COIL_DIM];

	debug_printf(DP_INFO, "imgs_dims:\n\t");
	debug_print_dims(DP_INFO, DIMS, irgnm_conf_dims);


	mdb_irgnm_l1(&conf2,
			irgnm_conf_dims,
			nl.nlop,
			size * 2, (float*)x,
			data_size * 2, (const float*)kspace_data);

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	long map_dims[DIMS];
	
	md_copy_dims(DIMS, map_dims, imgs_dims);
	map_dims[COEFF_DIM] = 1L;

	long map_size = md_calc_size(DIMS, map_dims);

	if (NULL != sens) {

		noir_forw_coils(nl.linop, x + skip, x + skip);
		md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		fftmod(DIMS, coil_dims, fft_flags, sens, sens);
	}

	// (M0, R1, alpha) model
	if (0. != conf->IR_phy) {

		long pos[DIMS];

		for (int i = 0; i < (int)DIMS; i++)
		pos[i] = 0;

		// output the alpha map (in degree)
		pos[COEFF_DIM] = 2;
		md_copy_block(DIMS, pos, map_dims, x, imgs_dims, img, CFL_SIZE);
		T1_forw_alpha(nl.linop_alpha, x, x);
		md_zreal(DIMS, map_dims, x, x);
		md_zsmul(DIMS, map_dims, x, x, -conf->IR_phy * 1e-6 * 0.2);
		md_smin(1, MD_DIMS(2 * map_size), (float*)x, (float*)x, 0.);
		md_zexp(DIMS, map_dims, x, x);
		md_zacos(DIMS, map_dims, x, x);
	        md_zsmul(DIMS, map_dims, x, x, 180. / M_PI);
		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, x, CFL_SIZE);

		linop_free(nl.linop_alpha);
	}


	nlop_free(nl.nlop);

	md_free(x);
}



