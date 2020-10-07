/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 * 
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "grecon/optreg.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "linops/grad.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/utils.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"

#include "meco.h"
#include "optreg.h"


const struct operator_p_s* create_wav_prox(const long img_dims[DIMS], unsigned int jt_flag, float lambda)
{
	bool randshift = true;
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	unsigned int wflags = 0;

	for (unsigned int i = 0; i < DIMS; i++) {

		if ((1 < img_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {

			wflags = MD_SET(wflags, i);
			minsize[i] = MIN(img_dims[i], DIMS);
		}
	}

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, jt_flag, minsize, lambda, randshift);
}

const struct operator_p_s* create_llr_prox(const long img_dims[DIMS], unsigned int jt_flag, float lambda)
{
	bool randshift = true;
	long blk_dims[MAX_LEV][DIMS];
	int blk_size = 16;

	int levels = llr_blkdims(blk_dims, ~jt_flag, img_dims, blk_size);
	UNUSED(levels);

	return lrthresh_create(img_dims, randshift, ~jt_flag, (const long (*)[])blk_dims, lambda, false, false, false);
}

const struct operator_p_s* create_stack_spatial_thresh_prox(unsigned int N, const long x_dims[N], long js_dim, unsigned int regu, float lambda, unsigned int model)
{
	assert(MECO_PI != model);

	unsigned int wgh_fB0 = (MECO_PHASEDIFF == model) ? MECO_IDENTITY : MECO_SOBOLEV; // FIXME: this is hard-coded

	long nr_coeff = set_num_of_coeff(model);
	long D = x_dims[js_dim];

	long x_prox1_dims[N];
	md_copy_dims(N, x_prox1_dims, x_dims);
	x_prox1_dims[js_dim] = nr_coeff - 1; // exclude fB0

	long x_prox3_dims[N];
	md_copy_dims(N, x_prox3_dims, x_dims);
	x_prox3_dims[js_dim] = 1; // fB0

	long x_prox4_dims[N];
	md_copy_dims(N, x_prox4_dims, x_dims);
	x_prox4_dims[js_dim] = D - nr_coeff;

	debug_printf(DP_DEBUG4, " >> x_prox1_dims: ");
	debug_print_dims(DP_DEBUG4, N, x_prox1_dims);

	debug_printf(DP_DEBUG4, " >> x_prox3_dims: ");
	debug_print_dims(DP_DEBUG4, N, x_prox3_dims);

	const struct operator_p_s* pcurr = NULL;

	auto prox1 = ((L1WAV==regu) ? create_wav_prox : create_llr_prox)(x_prox1_dims, MD_BIT(js_dim), lambda);

	auto prox3 = prox_zero_create(N, x_prox3_dims);

	if (MECO_IDENTITY == wgh_fB0) {
		prox3 = ((L1WAV==regu) ? create_wav_prox : create_llr_prox)(x_prox3_dims, MD_BIT(js_dim), lambda);
	}

	auto prox4 = prox_zero_create(N, x_prox4_dims);
#if 0
	audo prox2 = op_p_auto_normalize(prox1, ~MD_BIT(js_dim));
	pcurr = operator_p_stack(js_dim, js_dim, prox2, prox3);

	operator_p_free(prox2);
	operator_p_free(prox3);
#else
	pcurr = operator_p_stack(js_dim, js_dim, prox1, prox3);

	operator_p_free(prox1);
	operator_p_free(prox3);
#endif

	pcurr = operator_p_stack(js_dim, js_dim, pcurr, prox4);

	operator_p_free(prox4);

	return pcurr;
}

const struct operator_p_s* create_stack_nonneg_prox(unsigned int N, const long x_dims[N], long js_dim, unsigned int model, bool real_pd)
{
	assert(MECO_PI != model);

	long R2S_flag = set_R2S_flag(model);
	long  PD_flag = set_PD_flag(model);

	long x_iden_dims[N];
	long x_prox_dims[N];

	const struct operator_p_s* prox1 = NULL;
	const struct operator_p_s* prox2 = NULL;

	const struct operator_p_s* pcurr = NULL;

	long D = x_dims[js_dim];

	for (long pind = 0; pind < D; pind++) {

		if ((MD_IS_SET(PD_flag, pind) && real_pd) || MD_IS_SET(R2S_flag, pind)) {

			md_copy_dims(N, x_prox_dims, x_dims);
			x_prox_dims[js_dim] = 1;

			debug_printf(DP_DEBUG4, " >> x_prox_dims: ");
			debug_print_dims(DP_DEBUG4, N, x_prox_dims);

			prox2 = prox_nonneg_create(N, x_prox_dims);
			pcurr = (NULL == pcurr) ? prox2 : operator_p_stack(js_dim, js_dim, pcurr, prox2);

		} else {

			md_copy_dims(N, x_iden_dims, x_dims);
			x_iden_dims[js_dim] = 1;

			debug_printf(DP_DEBUG4, " >> x_iden_dims: ");
			debug_print_dims(DP_DEBUG4, N, x_iden_dims);

			prox1 = prox_zero_create(N, x_iden_dims);
			pcurr = (NULL == pcurr) ? prox1 : operator_p_stack(js_dim, js_dim, pcurr, prox1);

		}

	}

	operator_p_free(prox1);
	operator_p_free(prox2);

	return pcurr;
}



void help_reg_moba(void)
{
	printf( "Generalized regularization options for model-based reconstructions (experimental)\n\n"
			"-R <T>:A:B:C\t<T> is regularization type (single letter),\n"
			"\t\tA is transform flags,\n"
			"\t\tB is joint threshold flags,\n"
			"\t\tC is regularization value.\n"
			"\t\tSpecify any number of regularization terms.\n\n"
			"-R W:0:0:C\tl1-wavelet (A and B are internally determined by moba models)\n"
			"-R L:0:0:C\tlocally low rank (A and B are internally determined by moba models)\n"
			"-R Q:C\tl2 regularization\n"
			"-R S\tnon-negative constraint\n"
			"-R T:A:B:C\ttotal variation\n"
		  );
}

bool opt_reg_moba(void* ptr, char c, const char* optarg)
{
	struct opt_reg_s* p = ptr;

	struct reg_s* regs = p->regs;
	const int r = p->r;

	assert(r < NUM_REGS);
	
	char rt[5];
	int ret;
	
	switch (c) {

	case 'r':

		// first get transform type
		ret = sscanf(optarg, "%4[^:]", rt);
		assert(1 == ret);

		if (strcmp(rt, "W") == 0) {

			regs[r].xform = L1WAV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);

		} else 
		if (strcmp(rt, "L") == 0) {

			regs[r].xform = LLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);

		} else 
		if (strcmp(rt, "Q") == 0) {

			regs[r].xform = L2IMG;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;

		} else 
		if (strcmp(rt, "S") == 0) {

			regs[r].xform = POS;

		} else 
		if (strcmp(rt, "T") == 0) {

			regs[r].xform = TV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);

		} else 
		if (strcmp(rt, "h") == 0) {

			help_reg_moba();
			exit(0);

		} else {

			error(" > Unrecognized regularization type: \"%s\"\n", rt);
		}

		p->r++;
		break;

	}

	return false;
}


static void opt_reg_T1_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int model)
{
	UNUSED(model);

	float lambda = ropts->lambda;
	unsigned int shift_model = 1;
	bool randshift = shift_model == 1;
#if 0
	bool overlapping_blocks = shift_mode == 2;
#endif

	if (-1. == lambda)
		lambda = 0.;

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims); 

	long img_no_time2_dims[DIMS];
	md_select_dims(DIMS, ~TIME2_FLAG, img_no_time2_dims, img_dims);

	long x_dims[DIMS];
	md_copy_dims(DIMS, x_dims, img_dims);
	x_dims[COEFF_DIM] = img_dims[COEFF_DIM] + dims[COIL_DIM]; //FIXME

	long x_no_time2_dims[DIMS];
	md_select_dims(DIMS, ~TIME2_FLAG, x_no_time2_dims, x_dims);

	long coil_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, coil_dims, dims);

	long coil_no_time2_dims[DIMS];
	md_select_dims(DIMS, ~TIME2_FLAG, coil_no_time2_dims, coil_dims);

	long map_no_time2_dims[DIMS];
	md_copy_dims(DIMS, map_no_time2_dims, img_no_time2_dims);
	map_no_time2_dims[COEFF_DIM] = 1L;

	long map2_no_time2_dims[DIMS];
	md_copy_dims(DIMS, map2_no_time2_dims, img_no_time2_dims);
	map2_no_time2_dims[COEFF_DIM] = map2_no_time2_dims[COEFF_DIM] - 1L;


    	long phases = x_dims[TIME2_DIM];

	// if no penalities specified but regularization
	// parameter is given, add a l2 penalty

	struct reg_s* regs = ropts->regs;

	if ((0 == ropts->r) && (lambda > 0.)) {

		regs[0].xform = L2IMG;
		regs[0].xflags = 0u;
		regs[0].jflags = 0u;
		regs[0].lambda = lambda;
		ropts->r = 1;
	}


	int nr_penalties = ropts->r;
#if 0
	long blkdims[MAX_LEV][DIMS];
	int levels;
#endif


	for (int nr = 0; nr < nr_penalties; nr++) {

		// fix up regularization parameter
		if (-1. == regs[nr].lambda)
			regs[nr].lambda = lambda;

		switch (regs[nr].xform) {

		case L1WAV:
		{

			debug_printf(DP_INFO, "l1-wavelet regularization: %f\n", regs[nr].lambda);


			long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
			minsize[0] = MIN(img_no_time2_dims[0], 16);
			minsize[1] = MIN(img_no_time2_dims[1], 16);
			minsize[2] = MIN(img_no_time2_dims[2], 16);


			unsigned int wflags = 0;
			for (unsigned int i = 0; i < DIMS; i++) {

				if ((1 < img_no_time2_dims[i]) && MD_IS_SET(regs[nr].xflags, i)) {

					wflags = MD_SET(wflags, i);
					minsize[i] = MIN(img_no_time2_dims[i], 16);
				}
			}

			auto l1Wav_prox = prox_wavelet_thresh_create(DIMS, img_no_time2_dims, wflags, regs[nr].jflags, minsize, regs[nr].lambda, randshift);
			auto zero_prox = prox_zero_create(DIMS, coil_no_time2_dims);

			auto stack = operator_p_stack(COEFF_DIM, COEFF_DIM, l1Wav_prox, zero_prox);
			auto stack1 = operator_p_stack(COEFF_DIM, COEFF_DIM, l1Wav_prox, zero_prox);

			auto id = linop_identity_create(DIMS, x_no_time2_dims);
			auto id1 = linop_identity_create(DIMS, x_no_time2_dims);


			if (phases > 1) {

				for (int k = 0; k < (phases - 1); k++) {

					auto tmp_prox = operator_p_stack(TIME2_DIM, TIME2_DIM, stack1, stack);
					auto tmp_id = linop_stack(TIME2_DIM, TIME2_DIM, id1, id);

					linop_free(id1);
					operator_p_free(stack1);

					id1 = tmp_id;
					stack1 = tmp_prox;
				}
			}

			trafos[nr] = id1;
			prox_ops[nr] = stack1;

			operator_p_free(l1Wav_prox);
			operator_p_free(zero_prox);
			operator_p_free(stack);	
			linop_free(id);		

			break;
		}

		case TV:
			debug_printf(DP_INFO, "TV regularization: %f\n", regs[nr].lambda);

			auto grad = linop_grad_create(DIMS, x_no_time2_dims, DIMS, regs[nr].xflags);
			auto thresh_prox = prox_thresh_create(DIMS + 1,
					linop_codomain(grad)->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS));
			

			auto grad1 = linop_grad_create(DIMS, x_no_time2_dims, DIMS, regs[nr].xflags);
			auto thresh1 = prox_thresh_create(DIMS + 1,
					linop_codomain(grad1)->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS));

			if ( phases > 1) {

				for (int k = 0; k < (phases - 1); k++) {

					auto tmp_id = linop_stack(TIME2_DIM, TIME2_DIM, grad1, grad);
					auto tmp_thresh = operator_p_stack(TIME2_DIM, TIME2_DIM, thresh1, thresh_prox);

					linop_free(grad1);
					operator_p_free(thresh1);

					grad1 = tmp_id;
					thresh1 = tmp_thresh;
				}
			}
			
			trafos[nr] = grad1;
			prox_ops[nr] = thresh1;

			operator_p_free(thresh_prox);
			linop_free(grad);

			break;
		case LLR:

			debug_printf(DP_INFO, "lowrank regularization: %f\n", regs[nr].lambda);
#if 0	
			if (use_gpu)
				error("GPU operation is not currently implemented for lowrank regularization.\n");


			// add locally lowrank penalty
			levels = llr_blkdims(blkdims, regs[nr].jflags, img_no_time2_dims, llr_blk);

			assert(1 == levels);

			assert(levels == img_no_time2_dims[LEVEL_DIM]);

			for(int l = 0; l < levels; l++)
#if 0
				blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];
#else
				blkdims[l][MAPS_DIM] = 1;
#endif

			int remove_mean = 0;
			
			auto lrthresh_prox = lrthresh_create(img_no_time2_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, remove_mean, overlapping_blocks);
			auto zero_prox = prox_zero_create(DIMS, coil_no_time2_dims);

			auto id = linop_identity_create(DIMS, x_no_time2_dims);
			auto id1 = linop_identity_create(DIMS, x_no_time2_dims);

			auto stack = operator_p_stack(COEFF_DIM, COEFF_DIM, lrthresh_prox, zero_prox); 
			auto stack1 = operator_p_stack(COEFF_DIM, COEFF_DIM, lrthresh_prox, zero_prox);
			
			if ( phases > 1) {

				for (int k = 0; k < (phases - 1); k++) {

					auto tmp_id = linop_stack(TIME2_DIM, TIME2_DIM, id1, id);
					auto tmp_thresh = operator_p_stack(TIME2_DIM, TIME2_DIM, stack1, stack);

					linop_free(id1);
					operator_p_free(stack1);

					id1 = tmp_id;
					stack1 = tmp_thresh;
				}
			}
			
			trafos[nr] = id1;
			prox_ops[nr] = stack1;
			
			operator_p_free(lrthresh_prox);
			operator_p_free(zero_prox);
			operator_p_free(stack);
			linop_free(id);
#endif
			break;
		case POS:
			debug_printf(DP_INFO, "non-negative constraint\n");

			auto zsmax_prox = prox_zsmax_create(DIMS, map_no_time2_dims, regs[nr].lambda);
			auto zero_prox1 = prox_zero_create(DIMS, map2_no_time2_dims);
			auto zero_prox2 = prox_zero_create(DIMS, coil_no_time2_dims);
			auto stack0 = operator_p_stack(COEFF_DIM, COEFF_DIM, zero_prox1, zsmax_prox); 

			auto stack2 = operator_p_stack(COEFF_DIM, COEFF_DIM, stack0, zero_prox2); 
			auto stack3 = operator_p_stack(COEFF_DIM, COEFF_DIM, stack0, zero_prox2); 

			auto id2 = linop_identity_create(DIMS, x_no_time2_dims);
			auto id3 = linop_identity_create(DIMS, x_no_time2_dims);

			
			if ( phases > 1) {

				for (int k = 0; k < (phases - 1); k++) {

					auto tmp_id = linop_stack(TIME2_DIM, TIME2_DIM, id3, id2);
					auto tmp_thresh = operator_p_stack(TIME2_DIM, TIME2_DIM, stack3, stack2);

					linop_free(id3);
					operator_p_free(stack3);

					id3 = tmp_id;
					stack3 = tmp_thresh;
				}
			}


			trafos[nr] = id3;
			prox_ops[nr] = stack3;

			operator_p_free(zsmax_prox);
			operator_p_free(zero_prox1);
			operator_p_free(zero_prox2);
			operator_p_free(stack0);
			operator_p_free(stack2);
			linop_free(id2);

			break;

		case L2IMG:
			debug_printf(DP_INFO, "l2 regularization: %f\n", regs[nr].lambda);

			auto id0 = linop_identity_create(DIMS, x_no_time2_dims);
			auto id4 = linop_identity_create(DIMS, x_no_time2_dims);

			auto zero_prox3 = prox_zero_create(DIMS, img_no_time2_dims);
			auto l2_coils = prox_leastsquares_create(DIMS, coil_no_time2_dims, lambda, NULL);

			auto stack_prox = operator_p_stack(COEFF_DIM, COEFF_DIM, zero_prox3, l2_coils);
			auto stack_prox1 = operator_p_stack(COEFF_DIM, COEFF_DIM, zero_prox3, l2_coils);

			if ( phases > 1) {

				for (int k = 0; k < (phases - 1); k++) {

					auto tmp_id = linop_stack(TIME2_DIM, TIME2_DIM, id4, id0);
					auto tmp_thresh = operator_p_stack(TIME2_DIM, TIME2_DIM, stack_prox1, stack_prox);

					linop_free(id4);
					operator_p_free(stack_prox1);

					id4 = tmp_id;
					stack_prox1 = tmp_thresh;
				}
			}

			trafos[nr] = id4;
			prox_ops[nr] = stack_prox1;

			operator_p_free(zero_prox3);
			operator_p_free(l2_coils);
			linop_free(id0);

			break;

		default:

			prox_ops[nr] = NULL;
			trafos[nr] = NULL;

			break;

		}
	}
}


static void opt_reg_meco_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int model)
{
	long maps_dims[N];
	md_select_dims(N, ~COIL_FLAG, maps_dims, dims);

	long sens_dims[N];
	md_select_dims(N, ~COEFF_FLAG, sens_dims, dims);

	long js_dim = COEFF_DIM; // joint spatial dim
	long nr_coeff = maps_dims[js_dim];

	long jt_dim = TIME_DIM;  // joint temporal dim
	long nr_time = maps_dims[jt_dim];
	UNUSED(nr_time);


	// flatten number of maps and coils
	long x_dims[N];
	md_select_dims(N, ~(COIL_FLAG|TE_FLAG|COEFF_FLAG), x_dims, maps_dims);
	x_dims[js_dim] = nr_coeff + sens_dims[COIL_DIM];

	struct reg_s* regs = ropts->regs;
	int nr_penalties = ropts->r;

	debug_printf(DP_INFO, " > in total %1d regularizations:\n", nr_penalties);

	for (int nr = 0; nr < nr_penalties; nr++) {

		switch (regs[nr].xform) {

		case L1WAV:

			debug_printf(DP_INFO, " > l1-wavelet regularization\n");

			prox_ops[nr] = create_stack_spatial_thresh_prox(N, x_dims, js_dim, L1WAV, regs[nr].lambda, model);
			trafos[nr] = linop_identity_create(DIMS, x_dims);

			break;

		case LLR:

			debug_printf(DP_INFO, " > lowrank regularization\n");

			prox_ops[nr] = create_stack_spatial_thresh_prox(N, x_dims, js_dim, LLR, regs[nr].lambda, model);
			trafos[nr] = linop_identity_create(DIMS, x_dims);

			break;

		case L2IMG:

			debug_printf(DP_INFO, " > l2 regularization\n");

			prox_ops[nr] = prox_l2norm_create(N, x_dims, regs[nr].lambda);
			trafos[nr] = linop_identity_create(N, x_dims);

			break;

		case POS:

			debug_printf(DP_INFO, " > non-negative constraint\n");

			prox_ops[nr] = create_stack_nonneg_prox(N, x_dims, js_dim, model, false);
			trafos[nr] = linop_identity_create(N, x_dims);

			break;

		case TV: // spatial or temporal

			debug_printf(DP_INFO, " > TV regularization\n");

			trafos[nr] = linop_grad_create(N, x_dims, N, regs[nr].xflags);
			prox_ops[nr] = prox_thresh_create(N + 1,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(N));

			break;

		default:

			prox_ops[nr] = NULL;
			trafos[nr] = NULL;

			break;
		}
	}
}


void opt_reg_moba_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int model)
{
	switch (model) {

	case MECO_WF:
	case MECO_WFR2S:
	case MECO_WF2R2S:
	case MECO_R2S:
	case MECO_PHASEDIFF:

		opt_reg_meco_configure(N, dims, ropts, prox_ops, trafos, model);

		break;

	default:

		opt_reg_T1_configure(N, dims, ropts, prox_ops, trafos, model);

		break;

	}
}
