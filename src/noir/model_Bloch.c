/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2018. Martin Uecker.
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
s */


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
#include "nlops/blochfun.h"
#include "linops/someops.h"
#include "sense/model.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "num/iovec.h"

#include "model.h"
#include "model_Bloch.h"

#include "linops/lintest.h"


const struct modBlochFit modBlochFit_defaults = {
	
	.sequence = 1, /*inv. bSSFP*/
	.rfduration = 0.0009,
	.tr = 0.0045,
	.te = 0.00225,
	.averageSpokes = 1,
	.n_slcp = 1,
	
	.r1scaling = 1.,
	.r2scaling = 1.,
	.m0scaling = 1.,
	.fov_reduction_factor = 1.,
	.rm_no_echo = 0.,
};


struct modBloch_s bloch_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const complex float* input_img, const complex float* input_sp, const struct noir_model_conf_s* conf, const struct modBlochFit* fitPara, _Bool usegpu)
{
	
	struct noir_s nlinv = noir_create3(dims, mask, psf, conf);
	struct modBloch_s ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long input_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME2_FLAG, in_dims, dims);

	if (NULL != input_img)
		md_select_dims(DIMS, READ_FLAG|PHS1_FLAG, input_dims, dims);

	in_dims[COEFF_DIM] = 3;
	
#if 1
	struct nlop_s* Bloch = nlop_Bloch_create(DIMS, map_dims, out_dims, in_dims, input_dims, input_img, input_sp, fitPara, usegpu);
	debug_printf(DP_INFO, "Bloch(.)\n");
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(Bloch, 0)->dims); 			//input-dims of Bloch operator
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(Bloch, 0)->dims);			//output dims of bloch operator

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(Bloch, 0, b, 0);
	nlop_free(b);

	nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });
	nlop_free(c);
#endif

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;
	nlop_free(nlinv.nlop);

	return ret;
}


