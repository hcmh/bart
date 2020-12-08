/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker
 * 2018-2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content. 
 * Magn Reson Med 2008; 60:674-682.
 */


#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "linops/someops.h"
#include "linops/lintest.h"

#include "sense/model.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"

#include "model_Bloch.h"
#include "blochfun.h"



const struct modBlochFit modBlochFit_defaults = {
	
	.sequence = 1, /*inv. bSSFP*/
	.rfduration = 0.0009,
	.bwtp = 4,
	.tr = 0.0045,
	.te = 0.00225,
	.averaged_spokes = 1,
	.sliceprofile_spins = 1,
	.num_vfa = 1,
	.fa = 45.,
	.runs = 1,
	.inversion_pulse_length = 0.01,
	
	.scale = {1., 1., 1., 1.},
	.fov_reduction_factor = 1.,
	.rm_no_echo = 0.,
	.full_ode_sim = false,
	.not_wav_maps = 0,
	
	.input_b1 = NULL,
	.input_sliceprofile = NULL,
	.input_fa_profile = NULL,
};


struct modBloch_s bloch_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf, const struct modBlochFit* fit_para, _Bool usegpu)
{
	
	struct noir_s nlinv = noir_create3(dims, mask, psf, conf);
	struct modBloch_s ret;

	long all_dims[DIMS];
	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long input_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|COEFF_FLAG|TIME2_FLAG, all_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME2_FLAG, in_dims, dims);

	if (NULL != fit_para->input_b1)
		md_select_dims(DIMS, READ_FLAG|PHS1_FLAG, input_dims, dims);

	in_dims[COEFF_DIM] = all_dims[COEFF_DIM] =  3;
	
#if 1
	struct nlop_s* Bloch = nlop_Bloch_create(DIMS, all_dims, map_dims, out_dims, in_dims, input_dims, fit_para, usegpu);

	debug_printf(DP_INFO, "Bloch(.)\n");
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(Bloch, 0)->dims); 			//input-dims of Bloch operator
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(Bloch, 0)->dims);			//output dims of bloch operator

	debug_printf(DP_INFO, "NLINV(.)\n");
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(Bloch, 0, b, 0);
	nlop_free(b);

	debug_printf(DP_INFO, "Chained(.)\n");
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(c, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(c, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(c, 0)->dims);

	nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });
	nlop_free(c);
#endif

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;
	nlop_free(nlinv.nlop);

	return ret;
}


