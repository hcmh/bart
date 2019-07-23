/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/mri.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "TR <input> <output>";
static const char help_str[] = "Compute T1 map from M_0, M_ss, and R_1.\n";


int main_looklocker(int argc, char* argv[argc])
{
	float threshold = 0.5;

	const struct opt_s opts[] = {

		OPT_FLOAT('t', &threshold, "threshold", "Pixels with M0 values smaller than {threshold} are set to zero."),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	float TR = atof(argv[1]);
	
	num_init();

	long idims[DIMS];
	
	complex float* in_data = load_cfl(argv[2], DIMS, idims);

	long odims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, odims, idims);

	complex float* out_data = create_cfl(argv[3], DIMS, odims);

	long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

	long ostrs[DIMS];
	md_calc_strides(DIMS, ostrs, odims, CFL_SIZE);

	long pos[DIMS] = { 0 };

	do {
		float Ms = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 0, pos), in_data);
		float M0 = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 1, pos), in_data);
		float R1 = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 2, pos), in_data);

		float T1 = -TR / logf(1. - (Ms / M0) * (1. - expf(-TR * R1)));

		if (safe_isnanf(T1) || (M0 < 0.5))
			T1 = 0.;

		MD_ACCESS(DIMS, ostrs, (pos[COEFF_DIM] = 0, pos), out_data) = T1;

	} while(md_next(DIMS, odims, ~COEFF_FLAG, pos));

	unmap_cfl(DIMS, idims, in_data);
	unmap_cfl(DIMS, odims, out_data);
	return 0;
}


