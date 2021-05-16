/* Copyright 2021. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Zhengguo Tan
 */

#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops_p.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/stack.h"
#include "nlops/const.h"

#include "moba/meco.h"

#include "simu/signals.h"

#include "utest.h"


static bool test_meco_wf2r2s_mobafit(void)
{
	enum { N = 16 };
	enum { NECO = 21 };
	enum { IMSIZE = 16 };


	long NCOEFF = set_num_of_coeff(MECO_WF2R2S);

	long   y_dims[N] = { IMSIZE, IMSIZE, 1, 1, 1, NECO,      1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long   x_dims[N] = { IMSIZE, IMSIZE, 1, 1, 1,    1, NCOEFF, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long map_dims[N] = { IMSIZE, IMSIZE, 1, 1, 1,    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long   t_dims[N] = {      1,      1, 1, 1, 1, NECO,      1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	long pos[N];

	for (int i = 0; i < N; i++)
		pos[i] = 0;


	complex float* TE = md_alloc(N, t_dims, CFL_SIZE);

	for (int n = 0; n < NECO; n++)
		TE[n] = 0 + 0.5 * n;

	complex float* dst = md_alloc(N, y_dims, CFL_SIZE);
	complex float* src = md_alloc(N, x_dims, CFL_SIZE);
	complex float* map = md_alloc(N, map_dims, CFL_SIZE);

	md_zfill(N, x_dims, src, 0.0);

	complex float W = 0.8;
	complex float R2S_W = 0.05;
	complex float F = 0.2;
	complex float R2S_F = 0.02;
	// complex float fB0 = 0.;

	pos[6] = 0; // W
	md_zfill(N, map_dims, map, W);
	md_copy_block2(N, pos, x_dims, MD_STRIDES(N, x_dims, CFL_SIZE), src, map_dims, MD_STRIDES(N, map_dims, CFL_SIZE), map, CFL_SIZE);

	pos[6] = 1; // R2S_W
	md_zfill(N, map_dims, map, R2S_W);
	md_copy_block2(N, pos, x_dims, MD_STRIDES(N, x_dims, CFL_SIZE), src, map_dims, MD_STRIDES(N, map_dims, CFL_SIZE), map, CFL_SIZE);

	pos[6] = 2; // F
	md_zfill(N, map_dims, map, F);
	md_copy_block2(N, pos, x_dims, MD_STRIDES(N, x_dims, CFL_SIZE), src, map_dims, MD_STRIDES(N, map_dims, CFL_SIZE), map, CFL_SIZE);

	pos[6] = 3; // R2S_F
	md_zfill(N, map_dims, map, R2S_F);
	md_copy_block2(N, pos, x_dims, MD_STRIDES(N, x_dims, CFL_SIZE), src, map_dims, MD_STRIDES(N, map_dims, CFL_SIZE), map, CFL_SIZE);


	float scale_fB0[2] = { 0., 1. };
	struct nlop_s* meco = nlop_meco_create(N, y_dims, x_dims, TE, MECO_WF2R2S, false, FAT_SPEC_1, scale_fB0, false);

	nlop_apply(meco, N, y_dims, dst, N, x_dims, src);


#if 0
	// this part is to test the forward model accuracy

	complex float* ref = md_alloc(N, y_dims, CFL_SIZE);

	complex float* sig = md_alloc(N, t_dims, CFL_SIZE);
	complex float cshift = 0.;

	for (int n = 0; n < NECO; n++) {

		cshift = calc_fat_modulation(3.0, crealf(TE[n]) * 1.E-3, FAT_SPEC_1);
		sig[n] = ( W * cexpf(- R2S_W * TE[n]) + F * cshift * cexpf(- R2S_F * TE[n]) ) * cexpf(2.i * M_PI * fB0 * TE[n]);
	}

	md_zfill(N, map_dims, map, 1.0);

	md_clear(N, y_dims, ref, CFL_SIZE);
	md_zfmac2(N, y_dims, MD_STRIDES(N, y_dims, CFL_SIZE), ref, MD_STRIDES(N, map_dims, CFL_SIZE), map, MD_STRIDES(N, t_dims, CFL_SIZE), sig);


	float err = md_znrmse(N, y_dims, dst, ref);

	md_free(sig);
	md_free(ref);
#endif

	struct iter_conjgrad_conf conjgrad_conf = iter_conjgrad_defaults;
	struct lsqr_conf lsqr_conf = lsqr_defaults;
	lsqr_conf.it_gpu = false;

	const struct operator_p_s* lsqr = lsqr2_create(&lsqr_conf, iter2_conjgrad, CAST_UP(&conjgrad_conf), NULL, &meco->derivative[0][0], NULL, 0, NULL, NULL, NULL);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = 5;

	complex float* ref = md_alloc(N, x_dims, CFL_SIZE);
	md_clear(N, x_dims, ref, CFL_SIZE);

	iter4_irgnm2(CAST_UP(&irgnm_conf), meco,
			2 * md_calc_size(N, x_dims), (float*)ref, NULL,
			2 * md_calc_size(N, y_dims), (const float*)dst, lsqr,
			(struct iter_op_s){ NULL, NULL });

	float err = md_znrmse(N, x_dims, ref, src);

	md_free(ref);
	operator_p_free(lsqr);


	nlop_free(meco);

	md_free(TE);
	md_free(src);
	md_free(dst);
	md_free(map);


	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_meco_wf2r2s_mobafit);