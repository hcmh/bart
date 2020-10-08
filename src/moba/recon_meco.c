/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "num/gpuops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/rand.h"

#include "iter/italgos.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/vec.h"

#include "linops/someops.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "nlops/nlop.h"

#include "noir/model.h"

#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"


#include "iter_l1.h"
#include "recon_T1.h"

#include "meco.h"
#include "model_meco.h"
#include "recon_meco.h"


#include "optreg.h"



// rescale the reconstructed maps to the unit of Hz
// note: input and output are both maps
static void rescale_maps(unsigned int model, double scaling_Y, const struct linop_s* op, const complex float* scaling, const long maps_dims[DIMS], complex float* maps)
{
	if (MECO_PI == model) {

		md_zsmul(DIMS, maps_dims, maps, maps, 1. / scaling_Y);

	} else {

		long nr_coeff = maps_dims[COEFF_DIM];

		long R2S_flag = set_R2S_flag(model);
		long fB0_flag = set_fB0_flag(model);

		long map_dims[DIMS];
		md_select_dims(DIMS, ~COEFF_FLAG, map_dims, maps_dims);

		complex float* map = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, maps);



		long pos[DIMS] = { [0 ... DIMS -1] = 0 };

		for (long n = 0; n < nr_coeff; n++) {

			pos[COEFF_DIM] = n;
			md_copy_block(DIMS, pos, map_dims, map, maps_dims, maps, CFL_SIZE);

			md_zsmul(DIMS, map_dims, map, map, scaling[n]);

			if (MD_IS_SET(R2S_flag, n) || MD_IS_SET(fB0_flag, n)) {

				if (MD_IS_SET(fB0_flag, n))
					linop_forward_unchecked(op, map, map);

				md_zsmul(DIMS, map_dims, map, map, 1000.); // kHz --> Hz
			}

			md_copy_block(DIMS, pos, maps_dims, maps, map_dims, map, CFL_SIZE);
		}

		md_free(map);

	}
}



void meco_recon(const struct moba_conf* moba_conf, 
		unsigned int sel_model, unsigned int sel_irgnm, bool real_pd, 
		unsigned int wgh_fB0, float scale_fB0, iter_conf* iconf, 
		bool out_origin_maps, double scaling_Y, 
		const long maps_dims[DIMS], complex float* maps, 
		const long sens_dims[DIMS], complex float* sens, 
		complex float* x, complex float* xref, 
		const complex float* pattern, 
		const complex float* mask, 
		const complex float* TE, 
		const long ksp_dims[DIMS], 
		const complex float* ksp, 
		bool use_lsqr)
{
	long meco_dims[DIMS];

	bool use_gpu = false;

#ifdef USE_CUDA
	use_gpu = cuda_ondevice(ksp) ? true : false;
#endif

	unsigned int fft_flags = FFT_FLAGS;
	md_select_dims(DIMS, fft_flags|TE_FLAG, meco_dims, ksp_dims);

	long maps_size = md_calc_size(DIMS, maps_dims);
	long sens_size = md_calc_size(DIMS, sens_dims);
	long x_size = maps_size + sens_size;

	long y_size = md_calc_size(DIMS, ksp_dims);

	// x = (maps; coils)
	// variable which is optimized by the IRGNM
	complex float* x_akt = md_alloc_sameplace(1, MD_DIMS(x_size), CFL_SIZE, ksp);
	md_copy(1, MD_DIMS(x_size), x_akt, x, CFL_SIZE);

	complex float* xref_akt = md_alloc_sameplace(1, MD_DIMS(x_size), CFL_SIZE, ksp);
	md_copy(1, MD_DIMS(x_size), xref_akt, xref, CFL_SIZE);

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.noncart = moba_conf->noncartesian;
	mconf.fft_flags = fft_flags;
	mconf.a = 880;
	mconf.b = 32.;
	mconf.cnstcoil_flags = TE_FLAG;

	double start_time = timestamp();
	struct meco_s nl = meco_create(ksp_dims, meco_dims, maps_dims, mask, TE, pattern, sel_model, real_pd, wgh_fB0, scale_fB0, use_gpu, &mconf);
	double nlsecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "  _ nl of meco Create Time: %.2f s\n", nlsecs);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = moba_conf->iter;
	irgnm_conf.alpha = moba_conf->alpha;
	irgnm_conf.alpha_min = moba_conf->alpha_min;
	irgnm_conf.redu = moba_conf->redu;
	irgnm_conf.cgiter = 260;
	irgnm_conf.cgtol = /*(sel_model == PI) ? 0.1 :*/ 0.01; // 1./3.;
	irgnm_conf.nlinv_legacy = /*(sel_model == PI) ? true :*/ false;

	long x_dims[DIMS];
	md_merge_dims(DIMS, x_dims, maps_dims, sens_dims);
	debug_printf(DP_DEBUG2, "  _ x_dims: \t");
	debug_print_dims(DP_DEBUG2, DIMS, x_dims);


	// linearized reconstruction

	struct opt_reg_s ropts = moba_conf->ropts;

	const struct operator_p_s* inv_op = NULL;

	const struct operator_p_s* prox_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };


	if (ALGO_CG != moba_conf->algo) // CG
		sel_irgnm = 2;

	if (ALGO_CG == moba_conf->algo) { // CG

		debug_printf(DP_DEBUG2, " > linearized problem solved by CG\n");
		
		// assert(0 == moba_conf->ropts->r);

		inv_op = NULL;

	} else 
	if (ALGO_FISTA == moba_conf->algo) { // FISTA

		debug_printf(DP_DEBUG2, " > linearized problem solved by FISTA\n");

		struct mdb_irgnm_l1_conf mdb_conf = { 
			.c2 = &irgnm_conf, 
			.opt_reg = 1, 
			.step = moba_conf->step, 
			.lower_bound = moba_conf->lower_bound, 
			.constrained_maps = set_R2S_flag(sel_model), 
			.not_wav_maps = 1, 
			.flags = FFT_FLAGS, 
			.usegpu = use_gpu, 
			.algo = moba_conf->algo, 
			.rho = moba_conf->rho, 
			.ropts = &ropts, 
			.wav_reg = 1. };

		inv_op = T1inv_p_create(&mdb_conf, x_dims, nl.nlop);

	} else 
	if (ALGO_ADMM == moba_conf->algo) {

		debug_printf(DP_DEBUG2, " > linearized problem solved by ADMM ");

		if (use_lsqr) {

			/* use lsqr */
			debug_printf(DP_DEBUG2, "in lsqr\n");

			opt_reg_moba_configure(DIMS, x_dims, &ropts, prox_ops, trafos, sel_model);

			struct lsqr_conf lsqr_conf = { lsqr_defaults.lambda, use_gpu };

			auto iadmm_conf = CAST_DOWN(iter_admm_conf, iconf);
			iadmm_conf->cg_eps = irgnm_conf.cgtol;
			iadmm_conf->rho = moba_conf->rho;
			// set maxiter to 0 in order to 
			//   1) iteratively increase it along Newton steps
			//   2) not interfere with other programs that uses iter2_admm
			iadmm_conf->maxiter = 0;

			const struct nlop_s* nlop = nl.nlop;

			inv_op = lsqr2_create(&lsqr_conf, iter2_admm, CAST_UP(iadmm_conf), NULL, &nlop->derivative[0][0], NULL, ropts.r, prox_ops, trafos, NULL);

		} else {

			/* use iter_l1 */
			debug_printf(DP_DEBUG2, "in iter l1\n");

			struct mdb_irgnm_l1_conf mdb_conf = { 
				.c2 = &irgnm_conf, 
				.opt_reg = 1, 
				.step = moba_conf->step, 
				.lower_bound = moba_conf->lower_bound, 
				.constrained_maps = set_R2S_flag(sel_model), 
				.not_wav_maps = 1, 
				.flags = FFT_FLAGS, 
				.usegpu = use_gpu, 
				.algo = moba_conf->algo, 
				.rho = moba_conf->rho, 
				.ropts = &ropts, 
				.wav_reg = 1. };

			inv_op = T1inv_p_create(&mdb_conf, x_dims, nl.nlop);

		}

	} else {

		error(" > Unrecognized algorithms\n");
	}


	// irgnm reconstruction

	((1==sel_irgnm) ? iter4_irgnm : iter4_irgnm2)(CAST_UP(&irgnm_conf), 
		nl.nlop, 
		x_size * 2, (float*)x_akt, (float*)xref_akt, 
		y_size * 2, (const float*)ksp, 
		inv_op, (struct iter_op_s){ NULL, NULL });

	operator_p_free(inv_op);

	opt_reg_free(&ropts, prox_ops, trafos);


	// post processing

	md_copy(DIMS, maps_dims, maps, x_akt, CFL_SIZE);

	md_copy(DIMS, sens_dims, sens, x_akt + maps_size, CFL_SIZE);

	if ( !out_origin_maps ) {

		rescale_maps(sel_model, scaling_Y, nl.linop_fB0, nl.scaling, maps_dims, maps);

		noir_forw_coils(nl.linop, sens, sens);
		fftmod(DIMS, sens_dims, mconf.fft_flags, sens, sens);
	}

	md_copy(1, MD_DIMS(x_size), xref, xref_akt, CFL_SIZE);

	md_copy(1, MD_DIMS(x_size), x, x_akt, CFL_SIZE);



	md_free(x_akt);
	md_free(xref_akt);

	nlop_free(nl.nlop);

	// linop_free(nl.linop);
	// linop_free(nl.linop_fB0);
}
