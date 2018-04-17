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

#include "model_T1.h"
#include "model.h"




struct T1_s T1_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
	struct noir_s nlinv = noir_create2(dims, mask, psf, conf);
	struct T1_s ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG, in_dims, dims);

	in_dims[COEFF_DIM] = 3;

	md_singleton_dims(DIMS, TI_dims);
	TI_dims[TE_DIM] = 3;

	complex float* TI = md_calloc(DIMS, TI_dims, CFL_SIZE);

	// complex float* TI = load_cfl("/home/xwang/IR_scripts/TI_index", DIMS, TI_dims);
	assert(TI_dims[TE_DIM] == out_dims[TE_DIM]);

#if 1 
	// chain T1 model
	struct nlop_s* T1 = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI);
	nlinv.nlop = nlop_chain2(T1, 0, nlinv.nlop, 0);
#endif

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;

	return ret;
}


