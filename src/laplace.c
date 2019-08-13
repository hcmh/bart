/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/init.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/flpmath.h"

#include "manifold/manifold.h"



static const char usage_str[] = "<src> <L>";
static const char help_str[] =
		"Calculate Laplacian Matrix. <src>: [samples, coordinates]\n";

int main_laplace(int argc, char* argv[])
{

	struct laplace_conf conf = laplace_conf_default;


	const struct opt_s opts[] = {

		OPT_INT('n', &conf.nn, "nn", "Number of nearest neighbours"),
		OPT_FLOAT('s', &conf.sigma, "sigma", "Standard deviation"),
		OPT_SET('g', &conf.gen_out, "Output inv(D) @ W"),
		OPT_SET('T', &conf.temporal_nn, "Temporal nearest neighbours"),


	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	

	long src_dims[DIMS];
	complex float* src = load_cfl(argv[1], DIMS, src_dims);

	long L_dims[2];

	if (conf.temporal_nn) {
		debug_printf(DP_INFO, "Calculating temporal nearest neighbour Laplacian!\n");
		
		int max = 0;
		for (int i = 0; i < src_dims[0]; i++)
			max = (max < creal(src[i])) ? creal(src[i]) : max;
		
		L_dims[0] = max + 1;
		L_dims[1] = max + 1;
		
	} else {
		
		debug_printf(DP_INFO, "Calculating Laplacian!\n");
		
		L_dims[0] = src_dims[0];
		L_dims[1] = src_dims[0];
	}
		

	complex float* L = create_cfl(argv[2], 2, L_dims);


	calc_laplace(&conf, L_dims, L, src_dims, src);

	unmap_cfl(DIMS, src_dims, src);
	unmap_cfl(2, L_dims, L);


	exit(0);

}


