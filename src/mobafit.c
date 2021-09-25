/* Copyright 2021. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2020-2021 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/init.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "linops/fmac.h"
#include "nlops/nlop_jacobian.h"

#include "moba/meco.h"
#include "moba/T1fun.h"

#include "simu/signals.h"

#include "simu/signals.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Pixel-wise fitting of sequence models.";

int main_mobafit(int argc, char* argv[])
{
	double start_time = timestamp();

	const char* TE_file = NULL;
	const char* echo_file = NULL;
	const char* param_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &TE_file, "TE"),
		ARG_INFILE(true, &echo_file, "echo images"),
		ARG_OUTFILE(false, &param_file, "paramters"),
	};


	enum seq_type { BSSFP, IR_LL, FLASH, TSE, MOLLI, MGRE } seq = MGRE;

	unsigned int mgre_model = MECO_WFR2S;

	bool use_gpu = false;
	long patch_size[3] = { 1, 1, 1 };

	unsigned int iter = 5;

	const char* init_file = NULL;
	const char* basis_file = NULL;

	const struct opt_s opts[] = {

#if 0
		OPT_SELECT('F', enum seq_type, &seq, FLASH, "FLASH"),
		OPT_SELECT('B', enum seq_type, &seq, BSSFP, "bSSFP"),
		OPT_SELECT('T', enum seq_type, &seq, TSE, "TSE"),
		OPT_SELECT('M', enum seq_type, &seq, MOLLI, "MOLLI"),
#endif
		OPT_SELECT('L', enum seq_type, &seq, IR_LL, "Inversion Recovery Look-Locker"),
		OPT_SELECT('G', enum seq_type, &seq, MGRE, "MGRE"),
		OPT_UINT('m', &mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),
		OPT_UINT('i', &iter, "iter", "Number of IRGNM steps"),
		OPT_VEC3('p', &patch_size, "x:y:z", "(patch size) [default: 1:1:1]"),
		OPT_INFILE('I', &init_file, "init", "File for initialization"),
		OPT_SET('g', &use_gpu, "use gpu"),
		OPT_INFILE('B', &basis_file, "file", "temporal (or other) basis"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long bas_dims[DIMS];
	complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, bas_dims);
		md_zconj(DIMS, bas_dims, basis, basis);
	}

	long TE_dims[DIMS];
	complex float* TE = load_cfl(TE_file, DIMS, TE_dims);

	long y1_dims[DIMS];
	long y2_dims[DIMS];
	complex float* y = load_cfl(echo_file, DIMS, y1_dims);
	md_copy_dims(DIMS, y2_dims, y1_dims);

	if (NULL == basis) {

		assert(y1_dims[TE_DIM] == TE_dims[TE_DIM]);
	} else {

		assert(bas_dims[TE_DIM] == TE_dims[TE_DIM]);

		if (1 != y1_dims[TE_DIM]) {

			assert(y1_dims[TE_DIM] == TE_dims[TE_DIM]);
			y2_dims[COEFF_DIM] = bas_dims[COEFF_DIM];
			y2_dims[TE_DIM] = 1;
			complex float* ny = anon_cfl(NULL, DIMS, y2_dims);

			md_ztenmul(DIMS, y2_dims, ny, bas_dims, basis, y1_dims, y);

			unmap_cfl(DIMS, y1_dims, y);

			y = ny;
		} else {

			y1_dims[TE_DIM] = TE_dims[TE_DIM];
			y1_dims[COEFF_DIM] = 1;
		}
	}

	long x_dims[DIMS];
	md_select_dims(DIMS, ~(TE_FLAG | COEFF_FLAG), x_dims, y1_dims);
	x_dims[COEFF_DIM] = (IR_LL == seq) ? 3 : set_num_of_coeff(mgre_model);


	complex float* x = create_cfl(param_file, DIMS, x_dims);

	md_clear(DIMS, x_dims, x, CFL_SIZE);

	if (IR_LL == seq)
		md_zfill(DIMS, x_dims, x, 1.);


	complex float* xref = md_alloc(DIMS, x_dims, CFL_SIZE);

	md_clear(DIMS, x_dims, xref, CFL_SIZE);

	long init_dims[DIMS] = { [0 ... DIMS-1 ] = 1 };
	complex float* init = (NULL != init_file) ? load_cfl(init_file, DIMS, init_dims) : NULL;

	if (NULL != init) {

		assert(md_check_bounds(DIMS, 0, x_dims, init_dims));

		md_copy(DIMS, x_dims, xref, init, CFL_SIZE);
	}


	long y1_patch_dims[DIMS];
	md_copy_dims(DIMS, y1_patch_dims, y1_dims);
	md_copy_dims(3, y1_patch_dims, patch_size);

	long y2_patch_dims[DIMS];
	md_copy_dims(DIMS, y2_patch_dims, y2_dims);
	md_copy_dims(3, y2_patch_dims, patch_size);

	long x_patch_dims[DIMS];
	md_select_dims(DIMS, COEFF_FLAG, x_patch_dims, x_dims);
	md_copy_dims(3, x_patch_dims, patch_size);

	long map_patch_dims[DIMS];
	md_copy_dims(DIMS, map_patch_dims, x_patch_dims);
	map_patch_dims[COEFF_DIM] = 1;


	NESTED(void, nary_opt, (void* ptr[]))
	{
		complex float* y    = ptr[0];
		complex float* x    = ptr[1];
		complex float* xref = ptr[2];

		// create signal model
		const struct nlop_s* nlop = NULL;

		switch (seq) {

		case IR_LL:  ;

			nlop = nlop_T1_create(DIMS, map_patch_dims, y1_patch_dims, x_patch_dims, y1_patch_dims, TE, use_gpu);
			break;

		case MGRE:  ;

			float scale_fB0[2] = { 0., 1. };
			nlop = nlop_meco_create(DIMS, y1_patch_dims, x_patch_dims, TE, mgre_model, false, FAT_SPEC_1, scale_fB0, use_gpu);
			break;

		default:

			error("sequence type not supported");
		}

		if (NULL != basis) {

			long max_dims[DIMS];
			md_max_dims(DIMS, ~0, max_dims, y2_patch_dims, bas_dims);
			nlop = nlop_chain_FF(nlop, nlop_from_linop_F(linop_fmac_create(DIMS, max_dims, TE_FLAG, COEFF_FLAG, ~(TE_FLAG | COEFF_FLAG), basis)));

			long tdims[DIMS];
			md_copy_dims(DIMS, tdims, y2_patch_dims);
			tdims[TE_DIM] = tdims[COEFF_DIM];
			tdims[COEFF_DIM] = 1;

			//nlop = nlop_reshape_out_F(nlop, 0, DIMS, tdims);
			//nlop = nlop_zprecomp_jacobian_F(nlop);
			//nlop = nlop_reshape_out_F(nlop, 0, DIMS, y2_patch_dims);
		}

		struct iter_conjgrad_conf conjgrad_conf = iter_conjgrad_defaults;
		struct lsqr_conf lsqr_conf = lsqr_defaults;
		lsqr_conf.it_gpu = false;

		const struct operator_p_s* lsqr = lsqr2_create(&lsqr_conf, iter2_conjgrad, CAST_UP(&conjgrad_conf), NULL, &nlop->derivative[0][0], NULL, 0, NULL, NULL, NULL);

		struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
		irgnm_conf.iter = iter;


		complex float* y_patch    = md_alloc(DIMS, y2_patch_dims, CFL_SIZE);
		complex float* x_patch    = md_alloc(DIMS, x_patch_dims, CFL_SIZE);
		complex float* xref_patch = md_alloc(DIMS, x_patch_dims, CFL_SIZE);

		long pos[DIMS] = { 0 };

		md_copy_block(DIMS, pos, y2_patch_dims,    y_patch, y2_dims, y   , CFL_SIZE);
		md_copy_block(DIMS, pos, x_patch_dims,    x_patch, x_dims, x   , CFL_SIZE);
		md_copy_block(DIMS, pos, x_patch_dims, xref_patch, x_dims, xref, CFL_SIZE);

		if (0. == md_znorm(DIMS, y2_patch_dims, y_patch)) {

			debug_printf(DP_WARN, "source images are zero!\n");
			md_zfill(DIMS, x_patch_dims, x_patch, 0.);
		} else {

			iter4_irgnm2(CAST_UP(&irgnm_conf), (struct nlop_s*)nlop,
					2 * md_calc_size(DIMS, x_patch_dims), (float*)x_patch, (float*)xref_patch,
					2 * md_calc_size(DIMS, y2_patch_dims), (const float*)y_patch, lsqr,
					(struct iter_op_s){ NULL, NULL });

			md_copy_block(DIMS, pos, x_dims, x, x_patch_dims, x_patch, CFL_SIZE);
		}

		md_free(xref_patch);
		md_free(x_patch);
		md_free(y_patch);

		operator_p_free(lsqr);
		nlop_free(nlop);

	};// while(md_next(DIMS, y_dims, ~TE_FLAG, pos));

	long x_strs[DIMS];
	long y_strs[DIMS];
	md_calc_strides(DIMS, x_strs, x_dims, CFL_SIZE);
	md_calc_strides(DIMS, y_strs, y2_dims, CFL_SIZE);

	long loop_dims[DIMS];
	md_singleton_dims(DIMS, loop_dims);
	for (int i = 0; i < 3; i++) {

		assert(0 == y2_dims[i] % patch_size[i]);
		loop_dims[i] = y2_dims[i] / patch_size[i];

		x_strs[i] *= patch_size[i];
		y_strs[i] *= patch_size[i];
	}

	md_parallel_nary(3, DIMS, loop_dims, ~0, (const long*[3]){y_strs, x_strs, x_strs}, (void* [3]){y, x, xref}, nary_opt);

	md_free(xref);

	unmap_cfl(DIMS, y2_dims, y);
	unmap_cfl(DIMS, TE_dims, TE);
	unmap_cfl(DIMS, x_dims, x);

	if (NULL != init_file)
		unmap_cfl(DIMS, init_dims, init);

	if (NULL != basis_file)
		unmap_cfl(DIMS, bas_dims, basis);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);

	exit(0);
}
