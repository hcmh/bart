/* Copyright 2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 * Vivek Athalye, Michael Lustig, and Martin Uecker. Parallel Magnetic
 * Resonance Imaging as Approximation in a Reproducing Kernel Hilbert Space,
 * Inverse Problems, in press (2015) arXiv:1310.7489 [physics.med-ph]
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "rkhs/rkhs.h"


static const char* usage_str = "<kern> <traj> <kmat>";
static const char* help_str ="\tComputes kernel matrix.\n"
		"\t-c alpha\tcompute Cholesky factorization with shift\n";


int main_kmat(int argc, char* argv[])
{
	bool do_cholesky = false;
	float alpha = 0.;

	const struct opt_s opts[] = {

		OPT_SET('c', &do_cholesky, "cholesky"),
		OPT_FLOAT('a', &alpha, "a", "alpha"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


	const int D = 5;
	long dims[D];
	complex float* kern = load_cfl(argv[1], D, dims);

	long sdims[3];
	complex float* samples = load_cfl(argv[2], 3, sdims);
	int N = sdims[1] * sdims[2];
	assert(3 == sdims[0]);

	int C = dims[3];

	long kdims[8] = { N, 1, 1, C, N, 1, 1, C }; 

	debug_printf(DP_INFO, "Generate kernel matrix\n");

	complex float* kmat = create_cfl(argv[3], 8, kdims);
	calculate_kernelmatrix(kdims, kmat, N, samples, dims, kern);

	if (do_cholesky) {

		debug_printf(DP_INFO, "Cholesky...\n");

		int NN = md_calc_size(4, kdims);
		comp_cholesky(NN, alpha, kmat);
	}

	unmap_cfl(D, dims, kern);
	unmap_cfl(3, sdims, samples);
	unmap_cfl(8, kdims, kmat);

	exit(0);
}



