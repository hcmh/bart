
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
#include "recon_T1.h"
#include "model_Bloch.h"
#include "recon_Bloch.h"


void bloch_recon(const struct moba_conf* conf, const struct modBlochFit* fit_para, const long dims[DIMS], complex float* img, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* kspace_data, _Bool usegpu)
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long img1_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS; //SLICE_FLAG;

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
	mconf.rvc = false;
	mconf.noncart = conf->noncartesian;
	mconf.fft_flags = fft_flags;
	mconf.a = 880.;
	mconf.b = 32.;
	mconf.cnstcoil_flags = TE_FLAG;

	//Create operator
	struct modBloch_s nl = bloch_create(dims, mask, pattern, &mconf, fit_para, usegpu);
	
	
	//Set up parameter for IRGNM
	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	
	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.cgtol = 0.1f;
	irgnm_conf.cgiter = 250;
	irgnm_conf.nlinv_legacy = true;

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME2_FLAG, irgnm_conf_dims, imgs_dims);

	irgnm_conf_dims[COIL_DIM] = coil_dims[COIL_DIM];
	
	debug_printf(DP_INFO, "irgnm_conf_dims:\n\t");
	debug_print_dims(DP_INFO, DIMS, irgnm_conf_dims);

	struct mdb_irgnm_l1_conf conf2 = {	.c2 = &irgnm_conf,
										.opt_reg = conf->opt_reg,
										.step = 0.9,
										.lower_bound = 10E-4,
										.constrained_maps = 5,
										.not_wav_maps = fit_para->not_wav_maps,
										.wav_reg = 1};

    	conf2.algo = 3;

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



