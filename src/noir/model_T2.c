/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker
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
#include "nlops/T2fun.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "num/iovec.h"

#include "model_T2.h"
#include "model.h"




struct T2_s T2_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, const struct noir_model_conf_s* conf, _Bool use_gpu)
{
	struct noir_s nlinv = noir_create2(dims, mask, psf, conf);
	struct T2_s ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG, in_dims, dims);

	md_select_dims(DIMS, TE_FLAG, TI_dims, dims);

	in_dims[COEFF_DIM] = 2;

	// chain T2 model
	struct nlop_s* T2 = nlop_T2_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, use_gpu);

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(T2, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(T2, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(T2, 0, b, 0);
	nlop_free(b);

	nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });
	nlop_free(c);

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;

	nlop_free(nlinv.nlop);
	return ret;
}


