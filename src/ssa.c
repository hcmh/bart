/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "calib/calmat.h"
#include "calib/ssa.h"


static const char usage_str[] = "<src> <EOF> [<S>] [<backprojection>]";
static const char help_str[] =
		"Perform SSA-FARY or Singular Spectrum Analysis. <src>: [samples, coordinates]\n";


int main_ssa(int argc, char* argv[])
{
	struct delay_conf conf = ssa_conf_default;

	const struct opt_s opts[] = {

		OPT_INT('w', &conf.window, "window", "Window length"),
		OPT_CLEAR('z', &conf.zeropad, "Zeropadding [Default: True]"),
		OPT_INT('m', &conf.rm_mean, "0/1", "Remove mean [Default: True]"),
		OPT_INT('n', &conf.normalize, "0/1", "Normalize [Default: False]"),
		OPT_INT('r', &conf.rank, "rank", "Rank for backprojection. r < 0: Throw away first r components. r > 0: Use only first r components."),
		OPT_LONG('g', &conf.group, "bitmask", "Bitmask for Grouping (long value!)"),
		OPT_SET('i', &conf.EOF_info, "EOF info"),
		OPT_FLOAT('e', &conf.weight, "exp", "Soft delay-embedding"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (-1 == conf.window)
		error("Specify window length '-w'");

	conf.kernel_dims[0] = conf.window;

	char* name_EOF = argv[2];

	if (4 <= argc)
		conf.name_S = argv[3];

	if (5 == argc) {

		if (conf.EOF_info)
			conf.rank = 1;

		check_bp(&conf);
		conf.backproj = argv[4];
	}


	long in_dims[DIMS];
	complex float* in = load_cfl(argv[1], DIMS, in_dims);

	if (!md_check_dimensions(DIMS, in_dims, ~(READ_FLAG|PHS1_FLAG)))
		error("Only first two dimensions must be filled!");


	preproc_ac(in_dims, in, conf);


	long cal0_dims[DIMS];
	md_copy_dims(DIMS, cal0_dims, in_dims);

	if (conf.zeropad)
		cal0_dims[0] = in_dims[0] - 1 + conf.window;


	complex float* cal = md_alloc(DIMS, cal0_dims, CFL_SIZE);

	md_resize_center(DIMS, cal0_dims, cal, in_dims, in, CFL_SIZE); 

	long cal_dims[DIMS];
	md_transpose_dims(DIMS, 1, 3, cal_dims, cal0_dims);


	debug_printf(DP_INFO, conf.backproj ? "Performing SSA\n" : "Performing SSA-FARY\n");

	long A_dims[2];
	complex float* A = calibration_matrix(A_dims, conf.kernel_dims, cal_dims, cal);

	if (conf.weight > -1)		
		weight_delay(A_dims, A, conf);

	long N = A_dims[0];

	long U_dims[2] = { N, N };
	complex float* U = create_cfl(name_EOF, 2, U_dims);

	complex float* back = NULL;

	if (NULL != conf.backproj) {

		long back_dims[DIMS];
		md_transpose_dims(DIMS, 3, 1, back_dims, cal_dims);

		back = create_cfl(conf.backproj, DIMS, back_dims);
	}

	float* S_square = xmalloc(N * sizeof(float));

	ssa_fary(cal_dims, A_dims, A, U, S_square, back, conf);

	if (NULL != conf.name_S) {

		long S_dims[1] = { N };
		complex float* S = create_cfl(conf.name_S, 1, S_dims);

		for (int i = 0; i < N; i++)
			S[i] = sqrt(S_square[i]) + 0.i;

		unmap_cfl(1, S_dims, S);
	}

	xfree(S_square);

	unmap_cfl(2, U_dims, U);
	unmap_cfl(DIMS, in_dims, in);

	md_free(cal);

	exit(0);
}


