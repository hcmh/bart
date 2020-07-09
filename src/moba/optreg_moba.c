/* Copyright 2015-2017. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015-2016 Frank Ong <frankong@berkeley.edu>
 * 2015-2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 * 2018-2019 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2020      Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/iovec.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/waveop.h"
#include "linops/fmac.h"


#include "wavelet/wavthresh.h"

#include "lowrank/lrthresh.h"

#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "optreg_moba.h"


#define CFL_SIZE sizeof(complex float)


void help_reg_moba(void)
{
	printf( "Generalized regularization options (experimental)\n\n"
			"-R <T>:A:B:C\t<T> is regularization type (single letter),\n"
			"\t\tA is transform flags, B is joint threshold flags,\n"
			"\t\tand C is regularization value. Specify any number\n"
			"\t\tof regularization terms.\n\n"
			"-R Q:C    \tl2-norm in image domain\n"
                        "-R S:C    \tpositive constraint\n"
			"-R W:A:B:C\tl1-wavelet\n"
		        "\t\tC is an integer percentage, i.e. from 0-100\n"
			"-R T:A:B:C\ttotal variation\n"
			"-R M2:C    \tmanifold l2-norm in image domain\n"
			"-R T:7:0:.01\t3D isotropic total variation with 0.01 regularization.\n"
			"-R L:7:7:.02\tLocally low rank with spatial decimation and 0.02 regularization.\n"
			"-R M:7:7:.03\tMulti-scale low rank with spatial decimation and 0.03 regularization.\n"
	      );
}




bool opt_reg_moba(void* ptr, char c, const char* optarg)
{
	struct opt_reg_s* p = ptr;
	struct reg_s* regs = p->regs;
	const int r = p->r;
	const float lambda = p->lambda;

	assert(r < NUM_REGS);

	char rt[5];

	switch (c) {

	case 'r': {

		// first get transform type
		int ret = sscanf(optarg, "%4[^:]", rt);
		assert(1 == ret);

		// next switch based on transform type
		if (strcmp(rt, "W") == 0) {

			regs[r].xform = L1WAV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "L") == 0) {

			regs[r].xform = LLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "T") == 0) {

			regs[r].xform = TV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "S") == 0) {

			regs[r].xform = POS;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "Q") == 0) {

			regs[r].xform = L2IMG;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "h") == 0) {

			help_reg_moba();
			exit(0);
		}
		else {

			error("Unrecognized regularization type: \"%s\" (-rh for help).\n", rt);
		}

		p->r++;
		break;
	}

	case 'l':

		assert(r < NUM_REGS);
		regs[r].lambda = lambda;
		regs[r].xflags = 0u;
		regs[r].jflags = 0u;

		if (0 == strcmp("1", optarg)) {

			regs[r].xform = L1WAV;
			regs[r].xflags = 7u;

		} else if (0 == strcmp("2", optarg)) {

			regs[r].xform = L2IMG;

		} else {

			error("Unknown regularization type.\n");
		}

		p->lambda = -1.;
		p->r++;
		break;
	}

	return false;
}

bool opt_reg_init_moba(struct opt_reg_s* ropts)
{
	ropts->r = 0;
	ropts->lambda = -1;
	ropts->k = 0;

	return false;
}

#if 0
void opt_bpursuit_configure(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], const struct linop_s* model_op, const complex float* data, const float eps)
{
	int nr_penalties = ropts->r;
	assert(NUM_REGS > nr_penalties);

	const struct iovec_s* iov = linop_codomain(model_op);
	prox_ops[nr_penalties] = prox_l2ball_create(iov->N, iov->dims, eps, data);
	trafos[nr_penalties] = linop_clone(model_op);

	ropts->r++;
}
#endif

void opt_reg_configure_moba(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS],
unsigned int llr_blk, unsigned int shift_mode, const long Q_dims[__VLA(N)], const _Complex float* Q, bool use_gpu)
{
	UNUSED(Q);
	UNUSED(Q_dims);
	UNUSED(llr_blk);
	UNUSED(use_gpu);
	
	float lambda = ropts->lambda;
	bool randshift = shift_mode == 1;
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
	md_copy_dims(DIMS, coil_dims, img_dims);
	coil_dims[COEFF_DIM] = 1L;

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


		}
	}
}


void opt_reg_free_moba(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS])
{
	int nr_penalties = ropts->r;

	for (int nr = 0; nr < nr_penalties; nr++) {

		operator_p_free(prox_ops[nr]);
		linop_free(trafos[nr]);
	}
}
