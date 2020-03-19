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

#include "moba/T1fun.h"
#include "moba/T1MOLLI.h"
#include "moba/T1s_repara.h"
#include "moba/T1s_chain.h"

#include "model_T1.h"


//#define T1s_chain
#define T1_MOLLI

struct T1_s T1_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, const struct noir_model_conf_s* conf, _Bool use_gpu)
{
	struct noir_s nlinv = noir_create3(dims, mask, psf, conf);
	struct T1_s ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME2_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG|TIME2_FLAG, TI_dims, dims);

	in_dims[COEFF_DIM] = 3;

#if 1
	// chain T1 model
//#ifdef T1s_chain
#ifdef T1_MOLLI
	#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
	#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
	#endif

	// test
	// complex float* dst = md_alloc(DIMS, out_dims, CFL_SIZE);
        // complex float* src1 = md_alloc(DIMS, map_dims, CFL_SIZE); // M0
        // complex float* src2 = md_alloc(DIMS, map_dims, CFL_SIZE); // R1
        // complex float* src3 = md_alloc(DIMS, map_dims, CFL_SIZE); // R1s

	// complex float* src = md_alloc(DIMS, in_dims, CFL_SIZE); // maps

	// md_zfill(DIMS, map_dims, src1, 1.0);
	// md_zfill(DIMS, map_dims, src2, 1.0);
	// md_zfill(DIMS, map_dims, src3, 1.0);

	// md_zfill(DIMS, in_dims, src, 1.0);

        int parts = 5;
	out_dims[TE_DIM] /= parts;
	TI_dims[TE_DIM] /= parts;
	
	complex float* TI1 = my_alloc(DIMS, TI_dims, CFL_SIZE);
	complex float* TI2 = my_alloc(DIMS, TI_dims, CFL_SIZE);

	md_copy(DIMS, TI_dims, TI1, TI, CFL_SIZE);

	printf("TE_DIM is %ld\n", TI_dims[TE_DIM]);

	for (int i = 0; i < TI_dims[TE_DIM]; i++)
	{
		float T2_real = crealf(TI[1]) - crealf(TI[0]) + i * (crealf(TI[1]) - crealf(TI[0]));
		TI2[i] = CMPLX(T2_real, 0.0);
		printf("%f%+fi\n", crealf(TI2[i]), cimagf(TI2[i]));
	}

        // struct nlop_s* T1 = nlop_T1_repara_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, use_gpu);
	//struct nlop_s* T1 = nlop_T1s_chain_create(DIMS, map_dims, out_dims, TI_dims, TI1, TI2, use_gpu);

        struct nlop_s* T1 = nlop_T1MOLLI_create(DIMS, map_dims, out_dims, TI_dims, TI1, TI2, use_gpu);
	//out_dims[TE_DIM] *= parts;

        md_free(TI1);
	md_free(TI2);

        // md_free(src1);
        // md_free(src2);
        // md_free(src3);
	// md_free(src);
	// md_free(dst);
#else
	struct nlop_s* T1 = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, use_gpu);
#endif
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(T1, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(T1, 0)->dims);

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(T1, 0, b, 0);
	nlop_free(b);

#ifdef T1_MOLLI
	nlinv.nlop = nlop_permute_inputs(c, 4, (const int[4]){ 1, 2, 3, 0 });
#else
	nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });
#endif
	nlop_free(c);

#endif
	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;
	nlop_free(nlinv.nlop);

	return ret;
}


