/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker
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
#include "nlops/T1fun.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "num/iovec.h"

#include "model_T1.h"
#include "model.h"




struct T1_s T1_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, const struct noir_model_conf_s* conf)
{
	struct noir_s nlinv = noir_create3(dims, mask, psf, conf);
	struct T1_s ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

    md_select_dims(DIMS, conf->fft_flags|LEVEL_FLAG, map_dims, dims);
    md_select_dims(DIMS, conf->fft_flags|TE_FLAG|LEVEL_FLAG, out_dims, dims);
    md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|LEVEL_FLAG, in_dims, dims);
    md_select_dims(DIMS, TE_FLAG|LEVEL_FLAG, TI_dims, dims);

    in_dims[COEFF_DIM] = 3;

#if 1 
	// chain T1 model
	struct nlop_s* T1 = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, conf->use_gpu);
    debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(T1, 0)->dims);
    debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(T1, 0)->dims);

    debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
    debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
    debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	nlinv.nlop = nlop_chain2(T1, 0, nlinv.nlop, 0);
	nlinv.nlop = nlop_permute_inputs(nlinv.nlop, 2, (const int[2]){ 1, 0 });
#endif

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;

	return ret;
}


