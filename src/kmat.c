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
#include <getopt.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "rkhs/rkhs.h"


static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s <kern> <traj> <kmat>\n", name);
}

static void help(void)
{
	printf("\tComputes kernel matrix.\n"
		"\t-c alpha\tcompute Cholesky factorization with shift\n");
}


int main(int argc, char* argv[])
{
	bool do_cholesky = false;
	double alpha = 0.;

	char c;
	while (-1 != (c = getopt(argc, argv, "hc:"))) {

		switch (c) {

		case 'c':
			do_cholesky = true;
			alpha = atof(optarg);
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 3) {

		usage(argv[0], stderr);
		exit(1);
	}


	const int D = 5;
	long dims[D];
	complex float* kern = load_cfl(argv[optind + 0], D, dims);

	long sdims[2];
	complex float* samples = load_cfl(argv[optind + 1], 2, sdims);
	int N = sdims[1];
	assert(3 == sdims[0]);

	int C = dims[3];

	long kdims[8] = { N, 1, 1, C, N, 1, 1, C }; 

	debug_printf(DP_INFO, "Generate kernel matrix\n");

	complex float* kmat = create_cfl(argv[optind + 2], 8, kdims);
	calculate_kernelmatrix(kdims, kmat, N, samples, dims, kern);

	if (do_cholesky) {

		debug_printf(DP_INFO, "Cholesky...\n");

		int NN = md_calc_size(4, kdims);
		comp_cholesky(NN, alpha, kmat);
	}

	unmap_cfl(D, dims, kern);
	unmap_cfl(2, sdims, samples);
	unmap_cfl(8, kdims, kmat);

	exit(0);
}



