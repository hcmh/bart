/* Copyright 2019. Uecker Lab. University Medical Center Göttingen.
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



static const char help_str[] = "Calculate Laplacian Matrix. <src>: [samples, coordinates]";

int main_laplace(int argc, char* argv[argc])
{
	const char* src_file = NULL;
	const char* L_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &src_file, "src"),
		ARG_OUTFILE(true, &L_file, "L"),
	};

	struct laplace_conf conf = laplace_conf_default;

	const struct opt_s opts[] = {

		OPT_INT('N', &conf.nn, "nn", "Number of nearest neighbours"),
		OPT_FLOAT('s', &conf.sigma, "sigma", "Standard deviation"),
		OPT_SET('n', &conf.norm, "Normalized Laplacian"),
		OPT_SET('P', &conf.dmap, "Transition Probability Matrix (diffusion map)"),
		OPT_SET('a', &conf.anisotrop, "Anisotropy correction"),
		OPT_SET('T', &conf.temporal_nn, "Temporal nearest neighbours"),
		OPT_SET('k', &conf.kernel, "Kernel approach"),
		OPT_FLOAT('l', &conf.kernel_lambda, "lambda", "(Kernel lambda)"),	
		OPT_FLOAT('G', &conf.kernel_gamma, "gamma", "(Kernel gamma minimum)"),
		OPT_INT('i', &conf.iter_max, "iter", "[Kernel] Number of iterations"),
		OPT_SET('v', &conf.local_v, "Local velocity weighting"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(!(conf.kernel && conf.temporal_nn));


	long src_dims[DIMS];
	complex float* src = load_cfl(src_file, DIMS, src_dims);

	long L_dims[2] = {src_dims[0], src_dims[0]};

	complex float* L = create_cfl(L_file, 2, L_dims);

	calc_laplace(&conf, L_dims, L, src_dims, src);

	unmap_cfl(DIMS, src_dims, src);
	unmap_cfl(2, L_dims, L);

	exit(0);
}
