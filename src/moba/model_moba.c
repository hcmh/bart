/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */


#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"

#include "moba/moba.h"
#include "moba/T1fun.h"
#include "moba/T2fun.h"
#include "moba/IR_SS_fun.h"
#include "moba/T1MOLLI.h"
#include "moba/T1_alpha.h"
#include "moba/T1_alpha_in.h"
#include "moba/blochfun.h"

#include "model_moba.h"

struct moba_s moba_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf, const struct moba_conf_s* conf_model, _Bool use_gpu)
{
	// FIXME
	// function to get TI from moba_conf and dims
	//...complex float* TI = NULL;

	// function receiving B1 and outputting alpha
	//...complex float* alpha = NULL;

	// fucntion calculating the slice profile if desired


	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create3(data_dims, mask, psf, conf);
	struct moba_s ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TIME_FLAG|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG|TIME_FLAG|TIME2_FLAG, TI_dims, dims);


#if 1

	struct nlop_s* model = NULL;

	switch (conf_model->model) {

	case IR:

		model = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case MOLLI:

		//FIXME: Currently MOLLI model supports an acquisition of 5 heart beats
		out_dims[TE_DIM] /= 5;
		TI_dims[TE_DIM] /= 5;

		complex float* TI1 = md_alloc(DIMS, TI_dims, CFL_SIZE);
		complex float* TI2 = md_alloc(DIMS, TI_dims, CFL_SIZE);

		md_copy(DIMS, TI_dims, TI1, conf_model->irflash.input_TI, CFL_SIZE);
		md_copy(DIMS, TI_dims, TI2, conf_model->irflash.input_TI, CFL_SIZE);

        	model = nlop_T1MOLLI_create(DIMS, map_dims, out_dims, TI_dims, TI1, TI2, conf_model->irflash.input_TI_t1relax, use_gpu);

        	md_free(TI1);
		md_free(TI2);
		break;

	case IR_SS:

		model = nlop_IR_SS_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case IR_phy:

		model = nlop_T1_alpha_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case IR_phy_alpha_in:

		model = nlop_T1_alpha_in_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, conf_model->irflash.input_alpha, use_gpu);
		break;

	case T2:

		model = nlop_T2_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case Bloch:

		// FIXME: Cast struct types...not nice at least...creater wrapper later
		// model = nlop_Bloch_create(DIMS, dims, map_dims, out_dims, in_dims, fit_para, usegpu);
		break;

	}

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(model, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(model, 0)->dims);

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(model, 0, b, 0);
	nlop_free(b);

	if (MOLLI == conf_model->model)
		nlinv.nlop = nlop_permute_inputs(c, 4, (const int[4]){ 1, 2, 3, 0 });
	else
		nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });

	nlop_free(c);

#endif
	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;

	if (IR_phy == conf_model->model)
		ret.linop_alpha = T1_get_alpha_trafo(model);

	nlop_free(nlinv.nlop);

	return ret;
}


