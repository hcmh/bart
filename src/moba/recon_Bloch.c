/* Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2011-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
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
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "nlops/nlop.h"

#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/thresh.h"
#include "iter/italgos.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "iter_l1.h"
#include "model_Bloch.h"
#include "recon_Bloch.h"


void bloch_recon(const struct noir_conf_s* conf, const struct modBlochFit* fitPara, const long dims[DIMS], complex float* img, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* kspace_data, _Bool usegpu)
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long img1_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS|SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME2_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG|TIME2_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|TE_FLAG|TIME2_FLAG, data_dims, dims);
	md_select_dims(DIMS, fft_flags|TIME2_FLAG, img1_dims, dims);

	imgs_dims[COEFF_DIM] = 3;

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);
	md_copy(DIMS, coil_dims, x + skip, sens, CFL_SIZE);

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = conf->rvc;
	mconf.noncart = conf->noncart;
	mconf.fft_flags = fft_flags;
	mconf.a = 880.;
	mconf.b = 32.;

	
	//Create operator
	struct modBloch_s nl = bloch_create(dims, mask, pattern, &mconf, fitPara, usegpu);
	
	
	//Set up parameter for IRGNM
	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	
	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.cgtol = 0.1f;
	irgnm_conf.cgiter = 300;
	irgnm_conf.nlinv_legacy = true;

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME2_FLAG, irgnm_conf_dims, imgs_dims);

	irgnm_conf_dims[COIL_DIM] = coil_dims[COIL_DIM];
	
	debug_printf(DP_INFO, "imgs_dims:\n\t");
	debug_print_dims(DP_INFO, DIMS, irgnm_conf_dims);

	struct mdb_irgnm_l1_conf conf2 = { .c2 = &irgnm_conf, .step = 0.9, .lower_bound = 10E-9, .constrained_maps = 3, .not_wav_maps = fitPara->not_wav_maps };

	mdb_irgnm_l1(&conf2,
			irgnm_conf_dims,
			nl.nlop,
			size * 2, (float*)x,
			data_size * 2, (const float*)kspace_data);

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	if (NULL != sens) {

		noir_forw_coils(nl.linop, x + skip, x + skip);
		md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		fftmod(DIMS, coil_dims, fft_flags, sens, sens);
	}

	nlop_free(nl.nlop);


	md_free(x);
}



