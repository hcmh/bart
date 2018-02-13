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
#include "num/flpmath.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#include "rkhs/rkhs.h"







int main(int argc, const char* argv[])
{
	if (argc != 6) {

		fprintf(stderr, "Usage: %s <kern> <traj> <kmat> <data> <output>\n", argv[0]);
		exit(1);
	}

	const int D = 5;
	long dims[D];
	complex float* kern = load_cfl(argv[1], D, dims);

	long sdims[2];
	complex float* samples = load_cfl(argv[2], 2, sdims);

//	long ddims[8];
//	complex float* data = load_cfl(argv[4], 8, ddims);


	long kdims[8];
	complex float* kmat = load_cfl(argv[3], 8, kdims);

	assert(kdims[0] == kdims[4]);
	assert(kdims[3] == kdims[7]);
	assert(1 == kdims[1]);
	assert(1 == kdims[2]);
	assert(1 == kdims[5]);
	assert(1 == kdims[6]);

	assert(3 == sdims[0]);
	assert(kdims[0] == sdims[1]);


	int C = kdims[3];
	int N = kdims[0];

	long sample_dims[8];
	complex float* sample_data = load_cfl(argv[4], 8, sample_dims);

	int X = 64; // 128;
	int Y = 64; // 128;
	int Z = 1;
	long odims[8] = { X, Y, Z, C, 1, 1, 1, 1 }; // smaller for now

	complex float* odata = create_cfl(argv[5], 8, odims);
	md_clear(8, odims, odata, CFL_SIZE);


	long coe_dims[3] = { N, C, C };
	long res_dims[3] = { 1, 1, C } ;
	long sam_dims[3] = { N, C, 1 } ;

	long coe_str[3];
	long sam_str[3];
	long res_str[3];

	md_calc_strides(3, coe_str, coe_dims, CFL_SIZE);
	md_calc_strides(3, sam_str, sam_dims, CFL_SIZE);
	md_calc_strides(3, res_str, res_dims, CFL_SIZE);


	long kern_dims2[8] = { 1, 1, 1, 1, 1, N, C, C };

	long res_dims2[8] = { 1, 1, 1, C, 1, 1, 1, 1 };

	int counter = 0;

	#pragma omp parallel for
	for (int i = 0; i < odims[0]; i++) {
	
		complex float* lhs = md_alloc(8, kern_dims2, CFL_SIZE);

		for (int j = 0; j < odims[1]; j++) {


	//		printf("%d-%d\n", i, j);

			float npos[3] = { (i - odims[0] / 2), (j - odims[1] / 2), 0 };
			calculate_lhs(kdims, lhs, npos, N, samples, dims, kern);
			complex float* coeff = comp_cardinal(kdims, C, lhs, kmat);
			
			complex float res[C];
			md_clear(3, res_dims, res, CFL_SIZE);
			//md_zmadd2(3, coe_dims, res_str, res, coe_str, coeff, sam_str, sample_data);
			md_zfmacc2(3, coe_dims, res_str, res, sam_str, sample_data, coe_str, coeff);

			long pos[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
			pos[0] = i;
			pos[1] = j;

			md_copy_block(8, pos, odims, odata, res_dims2, res, CFL_SIZE);

			free(coeff);

			#pragma omp critical	
			{ printf("%04d/%04ld    \r", ++counter, odims[0] * odims[1]); fflush(stdout); }
		}
	
		free(lhs);
	}

	printf("\nDone.\n");

	unmap_cfl(D, dims, kern);
	unmap_cfl(2, sdims, samples);
	unmap_cfl(8, kdims, kmat);
	unmap_cfl(8, sample_dims, sample_data);
	unmap_cfl(8, odims, odata);
	exit(0);
}


