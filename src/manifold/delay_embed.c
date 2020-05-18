/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019. Sebastian Rosenzweig.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Sebastian Rosenzweig (sebastian.rosenzweig@med.uni-goettingen.de)
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "delay_embed.h"

const struct delay_conf ssa_conf_default = {

	.kernel_dims = { 1, 1, 1},
	.window		= -1,
	.normalize 	= 0,
	.rm_mean	= 1,
	.zeropad	= true,
	.weight	   	= -1,
	.name_S		= NULL,
	.backproj	= NULL,
	.group 		= 0,
	.rank 		= 0,
	
	.nlsa		= false,
	.nlsa_rank	= 0,
	.name_tbasis = NULL,
};

const struct delay_conf nlsa_conf_default = {

	.kernel_dims = { 1, 1, 1},
	.window		= -1,
	.normalize 	= 0,
	.rm_mean	= 1,
	.zeropad	= true,
	.weight	   	= -1,
	.name_S		= NULL,
	.backproj	= NULL,
	.group 		= 0,
	.rank 	= 0,
	
	.nlsa		= true,
	.nlsa_rank	= 20,
	.name_tbasis = NULL,
};


// Check if basis function are rejected or not
bool check_selection(const long group, const int j)
{
	if (j > 30)
		return false; // group has only 32 bits

	return (labs(group) & (1 << j));
}

// Check if input-options for back-projection are valid
void check_bp(struct delay_conf* conf) {
	
		if (conf->zeropad) {

			debug_printf(DP_INFO, "Zeropadding turned off automatically!\n");
			conf->zeropad = false;
			
		}

		if ((0 == conf->rank) && (0 == conf->group))
			error("Specify rank or group for backprojection!");

		if (0 == conf->rank)
			assert(0 != conf->group);

		if (0 == conf->group)
			assert(0 != conf->rank);

}

// Preprocess AC region: remove mean or stdv
void preproc_ac(const long int in_dims[DIMS], complex float* in, const struct delay_conf conf)
{
	long in_strs[DIMS];
	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

	long singleton_dims[DIMS];
	long singleton_strs[DIMS];
	md_select_dims(DIMS, ~READ_FLAG, singleton_dims, in_dims);
	md_calc_strides(DIMS, singleton_strs, singleton_dims, CFL_SIZE);

	if (conf.rm_mean) {

		complex float* mean = md_alloc(DIMS, singleton_dims, CFL_SIZE);
		md_zavg(DIMS, in_dims, READ_FLAG, mean, in);
		md_zsub2(DIMS, in_dims, in_strs, in, in_strs, in, singleton_strs, mean);

		md_free(mean);
	}

	if (conf.normalize) {

		complex float* stdv = md_alloc(DIMS, singleton_dims, CFL_SIZE);
		md_zstd(DIMS, in_dims, READ_FLAG, stdv, in);
		md_zdiv2(DIMS, in_dims, in_strs, in, in_strs, in, singleton_strs, stdv);

		md_free(stdv);

	}
	
}

// Weighting within the window (soft time-delayed embedding)
void weight_delay(const long A_dims[2], complex float* A, const struct delay_conf conf)
{
		assert(0. < conf.weight);

		complex float* W;
		
		long W_dims[2];
		W_dims[0] = 1;
		W_dims[1] = A_dims[1];
		
		long A_strs[2];
		long W_strs[2];
		md_calc_strides(2, A_strs, A_dims, CFL_SIZE);
		md_calc_strides(2, W_strs, W_dims, CFL_SIZE);
		
		W = md_alloc(2, W_dims, CFL_SIZE);
		
		for (int i=0; i < A_dims[1]; i++) {
			float s = (float) abs( (int)((i % conf.window) - (int)((conf.window - 1) / 2)) );
			W[i] = expf(-conf.weight * s) + 0.i;
		}
		
		md_zmul2(2, A_dims, A_strs, A, A_strs, A, W_strs, W);
		
		md_free(W);

}
	

