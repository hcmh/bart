/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
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
#include "num/casorati.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/flpmath.h"

#include "calib/calib.h"
#include "calib/estvar.h"
#include "calib/calmat.h"


static const char usage_str[] = "<kspace> <EOF>";
static const char help_str[] =
		"Estimate cardiac and respiratory motion using Singular Spectrum Analysis\n";


static void ssa_fari(const long kernel_dims[3], const long cal_dims[DIMS], const complex float* cal_data, const char* dst, bool print_svals)
{
	// Calibration matrix
	long A_dims[2];
	complex float* A = calibration_matrix(A_dims, kernel_dims, cal_dims, cal_data);

	long N = A_dims[0];
	long M = A_dims[1];

	long AH_dims[2] = { M, N };
	complex float* AH = md_alloc(2, AH_dims, CFL_SIZE);
	md_transpose(2, 0, 1, AH_dims, AH, A_dims, A, CFL_SIZE);
	md_zconj(2, AH_dims, AH, AH);

	long AAH_dims[2] = { N, N };
	complex float* AAH = md_alloc(2, AAH_dims, CFL_SIZE);

	// AAH = A @ AH
	long A2_dims[3] = { N, M, 1 };
	long AH2_dims[3] = { 1, M, N };
	long AAH2_dims[3] = { N, 1, N };
	md_ztenmul(3, AAH2_dims, AAH, A2_dims, A, AH2_dims, AH);


	// AAH = U @ S @ UH
	long U_dims[2] = { N, N };
	complex float* U = create_cfl(dst, 2, U_dims);
	complex float* UH = md_alloc(2, U_dims, CFL_SIZE);

	float* S_square = xmalloc(N * sizeof(float));

	debug_printf(DP_INFO, "SVD of %dx%d matrix...", AAH_dims[0], AAH_dims[1]);
	lapack_svd(N, N, (complex float (*)[N])U, (complex float (*)[N])UH, S_square, (complex float (*)[N])AAH); // NOTE: Lapack destroys AAH!
	debug_printf(DP_INFO, "done\n");


	if (print_svals) {
		debug_printf(DP_INFO, "Printing 30 of %d singular values: \n", N);
		for (unsigned int i = 0; i < 30; i++)
			debug_printf(DP_INFO, "S[%i]: \t %f\n", i, sqrt(S_square[i]));
	}

	unmap_cfl(2, U_dims, U);
	md_free(UH);
	md_free(A);
	md_free(AH);
	md_free(AAH);



}



int main_ssa(int argc, char* argv[])
{
	int window = -1;
	bool print_svals = false;
	int normalize = 0;
	int rm_mean = 1;
	int type = 1;
	bool zeropad = true;
	long kernel_dims[3] = { 1, 1, 1};

	const struct opt_s opts[] = {

		OPT_INT('w', &window, "window", "Window length"),
		OPT_SET('s', &print_svals, "Print singular values"),
		OPT_CLEAR('z', &zeropad, "Zeropadding [Default: True]"),
		OPT_INT('m', &rm_mean, "0/1", "Remove mean [Default: True]"),
		OPT_INT('n', &normalize, "0/1", "Normalize [Default: False]"),
		OPT_INT('t', &type, "0-2", "0: Complex. 1: Absolute. 2: Angle. [Default: 1]"),


	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	if ( -1 == window)
		error("Specify window length '-w'");
	else
		kernel_dims[0] = window;

	long k_dims[DIMS];
	complex float* k = load_cfl(argv[1], DIMS, k_dims);

	if (k_dims[PHS2_DIM] > 1)
		error("No spokes allowed in PHS2_DIM [2]. Transpose to TIME_DIM [10]!");


	// Extract calibration region and reshape [TIME, 1, 1, COILS + SLICES, 1, ...]
	long cal_tmp_dims[DIMS];
	long cal_tmp1_dims[DIMS];
	long cal_dims[DIMS];

	md_select_dims(DIMS, COIL_FLAG|TIME_FLAG|SLICE_FLAG, cal_tmp_dims, k_dims);

	complex float* cal_data_tmp = md_alloc(DIMS, cal_tmp_dims, CFL_SIZE);
	complex float* cal_data_tmp1 = md_alloc(DIMS, cal_tmp_dims, CFL_SIZE);
	complex float* cal_data = md_alloc(DIMS, cal_tmp_dims, CFL_SIZE);

	md_resize_center(DIMS, cal_tmp_dims, cal_data_tmp, k_dims, k, CFL_SIZE);

		// Join SLICE_DIM & COIL_DIM
	md_transpose_dims(DIMS, COIL_DIM - 1, SLICE_DIM, cal_tmp1_dims, cal_tmp_dims);
	md_transpose(DIMS, COIL_DIM - 1, SLICE_DIM, cal_tmp1_dims, cal_data_tmp1, cal_tmp_dims, cal_data_tmp, CFL_SIZE);
	cal_tmp1_dims[COIL_DIM] = cal_tmp1_dims[COIL_DIM - 1] * cal_tmp1_dims[COIL_DIM];
	cal_tmp1_dims[COIL_DIM - 1] = 1;

		// Transpose TIME_DIM to READ_DIM
	md_transpose_dims(DIMS, READ_DIM, TIME_DIM, cal_dims, cal_tmp1_dims);
	md_transpose(DIMS, READ_DIM, TIME_DIM, cal_dims, cal_data, cal_tmp1_dims, cal_data_tmp1, CFL_SIZE);

	md_free(cal_data_tmp);
	md_free(cal_data_tmp1);

	switch (type) {

		case 0 :
			break;

		case 1 :
			md_zabs(DIMS, cal_dims, cal_data, cal_data);
			break;

		case 2 :
			md_zarg(DIMS, cal_dims, cal_data, cal_data);
			break;

	}

	if ( rm_mean || normalize ) {

		long cal_strs[DIMS];
		md_calc_strides(DIMS, cal_strs, cal_dims, CFL_SIZE);

		long singleton_dims[DIMS];
		long singleton_strs[DIMS];
		md_select_dims(DIMS, ~READ_FLAG, singleton_dims, cal_dims);
		md_calc_strides(DIMS, singleton_strs, singleton_dims, CFL_SIZE);

		if (rm_mean) {

			complex float* mean = md_alloc(DIMS, singleton_dims, CFL_SIZE);
			md_zavg(DIMS, cal_dims, READ_FLAG, mean, cal_data);
			md_zsub2(DIMS, cal_dims, cal_strs, cal_data, cal_strs, cal_data, singleton_strs, mean);

			md_free(mean);
		}

		if (normalize) {

			complex float* stdv = md_alloc(DIMS, singleton_dims, CFL_SIZE);
			md_zstd(DIMS, cal_dims, READ_FLAG, stdv, cal_data);
			md_zdiv2(DIMS, cal_dims, cal_strs, cal_data, cal_strs, cal_data, singleton_strs, stdv);

			md_free(stdv);

		}
	}

	if (zeropad) {

		long cal_zeropad_dims[DIMS];
		md_copy_dims(DIMS, cal_zeropad_dims, cal_dims);
		cal_zeropad_dims[0] = cal_dims[0] - 1 + window;
		complex float* cal_zeropad = md_alloc(DIMS, cal_zeropad_dims, CFL_SIZE);
		md_resize_center(DIMS, cal_zeropad_dims, cal_zeropad, cal_dims, cal_data, CFL_SIZE);

		md_free(cal_data);
		cal_data = cal_zeropad;
		md_copy_dims(DIMS, cal_dims, cal_zeropad_dims);

	}


	// Perform SSA-FARI
	ssa_fari(kernel_dims, cal_dims, cal_data, argv[2], print_svals);

	md_free(cal_data);

	exit(0);

}


