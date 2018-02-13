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
#include "misc/opts.h"

#include "rkhs/rkhs.h"


static const char* usage_str = "<kern> <traj> <kmat> <power> <stability> [<cardinal>]";
static const char* help_str = "";



int main_power(int argc, char* argv[])
{
	const struct opt_s opts[1];

	cmdline(&argc, argv, 5, 6, usage_str, help_str, 0, opts);


	long dims[5];
	complex float* kern = load_cfl(argv[1], 5, dims);

	long sdims[3];
	complex float* samples = load_cfl(argv[2], 3, sdims);

	long kdims[8];
	complex float* kmat = load_cfl(argv[3], 8, kdims);

	assert(kdims[0] == kdims[4]);
	assert(kdims[3] == kdims[7]);
	assert(1 == kdims[1]);
	assert(1 == kdims[2]);
	assert(1 == kdims[5]);
	assert(1 == kdims[6]);

	assert(3 == sdims[0]);
	assert(kdims[0] == sdims[1] * sdims[2]);


	int C = kdims[3];
	int N = kdims[0];


	int OX = 1;
	int OY = 1;
	int X = 64 * OX;
	int Y = 64 * OY;
	int Z = 1;
	long odims[8] = { X, Y, Z, C, 1, 1, 1, 1 }; // smaller for now

	complex float* odata = create_cfl(argv[4], 8, odims);
	md_clear(8, odims, odata, CFL_SIZE);

	complex float* leb = create_cfl(argv[5], 8, odims);
	md_clear(8, odims, leb, CFL_SIZE);

	long cdims[8] = { X, Y, Z, 1, 1, N, C, C };
	complex float* cardinal = NULL;

	if (7 == argc) {

		cardinal = create_cfl(argv[6], 8, cdims);
		md_clear(8, cdims, cardinal, CFL_SIZE);
	}


	printf("Generate kernel matrix\n");

	complex float* kmat0 = md_alloc(8, kdims, CFL_SIZE);
	calculate_kernelmatrix(kdims, kmat0, N, samples, dims, kern);



	long coe_dims[5] = { N, C, 1, 1, C };
	long res_dims[5] = { 1, 1, 1, 1, C };
	long sam_dims[5] = { N, C, 1, 1, C };
	long kma_dims[5] = { N, C, N, C, 1 };
	long tmp_dims[5] = { 1, 1, N, C, C };
	long all_dims[5] = { N, C, N, C, C };

	long coe_str[5];
	long sam_str[5];
	long res_str[5];
	long kma_str[5];
	long tmp_str[5];

	md_calc_strides(5, coe_str, coe_dims, CFL_SIZE);
	md_calc_strides(5, sam_str, sam_dims, CFL_SIZE);
	md_calc_strides(5, res_str, res_dims, CFL_SIZE);
	md_calc_strides(5, kma_str, kma_dims, CFL_SIZE);
	md_calc_strides(5, tmp_str, tmp_dims, CFL_SIZE);


	long kern_dims2[8] = { 1, 1, 1, 1, 1, N, C, C };
	long res_dims2[8] = { 1, 1, 1, C, 1, 1, 1, 1 };
	long coeff_dims2[8] = { 1, 1, 1, 1, 1, N, C, C };

	int counter = 0;

	#pragma omp parallel for
	for (int i = 0; i < odims[0]; i++) {
	
		complex float* lhs = md_alloc(8, kern_dims2, CFL_SIZE);
		complex float* lhsH = md_alloc(8, kern_dims2, CFL_SIZE);
		complex float* tmp = md_alloc(5, tmp_dims, CFL_SIZE);

		for (int j = 0; j < odims[1]; j++) {

			long pos[8] = { i, j, 0, 0, 0, 0, 0, 0 };

			// Cardinal function

			float npos[3] = { (float)(i - odims[0] / 2) / (float)OX, (float)(j - odims[1] / 2) / (float)OY, 0 };
			calculate_lhs(kdims, lhs, npos, N, samples, dims, kern);
			calculate_lhsH(kdims, lhsH, npos, N, samples, dims, kern);

			complex float* coeff = comp_cardinal(kdims, C, lhs, kmat);

			if (NULL != cardinal)
				md_copy_block(8, pos, cdims, cardinal, coeff_dims2, coeff, CFL_SIZE);

			// Power function
			
			complex float res[C];

			calculate_diag(kdims, res, dims, kern);
			md_zsmul(5, coe_dims, coeff, coeff, -1.); // -0.5
			md_zfmacc2(5, coe_dims, res_str, res, sam_str, lhs, coe_str, coeff);
			md_zfmac2(5, coe_dims, res_str, res, sam_str, lhsH, coe_str, coeff);
			md_zsmul(5, coe_dims, coeff, coeff, -1.); // -2.


#if 1
			md_clear(5, tmp_dims, tmp, CFL_SIZE);
			md_zfmacc2(5, all_dims, tmp_str, tmp, kma_str, kmat0, coe_str, coeff);
			md_zfmac2(5, sam_dims, res_str, res, coe_str, tmp, coe_str, coeff);
#endif
			//md_zsqrt(3, coe_dims, res, res);

	

			md_copy_block(8, pos, odims, odata, res_dims2, res, CFL_SIZE);


			complex float mag[C];

			for (int l = 0; l < C; l++) {

				float val = 0.;
#if 0
				// Lebesgue function
				for (int k = 0; k < C * N; k++)
					val += cabs(coeff[l * (C * N) + k]);

				mag[l] = val
#else
				for (int k = 0; k < C * N; k++)
					val += pow(cabs(coeff[l * (C * N) + k]), 2.);

				mag[l] = sqrt(val);
#endif
			}

			md_copy_block(8, pos, odims, leb, res_dims2, mag, CFL_SIZE);

			free(coeff);

			#pragma omp critical
			{ printf("%04d/%04ld    \r", ++counter, odims[0] * odims[1]); fflush(stdout); }
		}

		free(tmp);	
		free(lhs);
		free(lhsH);
	}

	free(kmat0);
	printf("\nDone.\n");
	exit(0);
}


