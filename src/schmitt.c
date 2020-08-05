/* Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
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


static const char usage_str[] = "<input maps> <input alpha> <output>";
static const char help_str[] = "Compute T1 and T2 maps from M_0, M_ss, R_1* and alpha.\n";

// Schmitt, P., Griswold, M.A., Jakob, P.M., Kotas, M., Gulani, V., Flentje, M. and Haase, A. (2004),
// Inversion recovery TrueFISP: Quantification of T 1, T 2, and spin density.
// Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058

int main_schmitt(int argc, char* argv[argc])
{
	float threshold = 0.;
	float scaling_M0 = 2.0;

	const struct opt_s opts[] = {

		OPT_FLOAT('t', &threshold, "threshold", "Pixels with M0 values smaller than {threshold} are set to zero."),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long idims[DIMS];

	complex float* in_data = load_cfl(argv[1], DIMS, idims);

	long i2dims[DIMS];

	complex float* in_alpha = load_cfl(argv[2], DIMS, i2dims);

	assert(md_check_equal_dims(DIMS, idims, i2dims, FFT_FLAGS));

	long odims[DIMS];
	md_copy_dims(DIMS, odims, idims);

	complex float* out_data = create_cfl(argv[3], DIMS, odims);

	long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

	long i2strs[DIMS];
	md_calc_strides(DIMS, i2strs, i2dims, CFL_SIZE);

	long ostrs[DIMS];
	md_calc_strides(DIMS, ostrs, odims, CFL_SIZE);

	long pos[DIMS] = { 0 };

	do {
		complex float Ms = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 0, pos), in_data);
		complex float M0u = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 1, pos), in_data);
		complex float R1s = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 2, pos), in_data);

		complex float alpha_val = (MD_ACCESS(DIMS, i2strs, (pos[COEFF_DIM] = 0, pos), in_alpha));

		complex float T1s = 1. / R1s;
		complex float M0 = scaling_M0 * M0u;

		float T1 = cabs(T1s * M0 / Ms * ccosf(alpha_val / 2.));
		float T2 = cabs(T1s * 1. / (1. - Ms / M0 * ccosf(alpha_val / 2.)) * csinf(alpha_val / 2.) * csinf(alpha_val / 2.));
		float PD = cabs(M0 / csinf(alpha_val / 2.));

		if (!safe_isfinite(T1) || !safe_isfinite(T2) || !safe_isfinite(PD) || (cabs(Ms) < threshold)) {

			T1 = T2 = PD = 0.;
		}

		MD_ACCESS(DIMS, ostrs, (pos[COEFF_DIM] = 0, pos), out_data) = T1;
		MD_ACCESS(DIMS, ostrs, (pos[COEFF_DIM] = 1, pos), out_data) = PD;
		MD_ACCESS(DIMS, ostrs, (pos[COEFF_DIM] = 2, pos), out_data) = T2;

	} while(md_next(DIMS, odims, ~COEFF_FLAG, pos));

	unmap_cfl(DIMS, idims, in_data);
	unmap_cfl(DIMS, i2dims, in_alpha);
	unmap_cfl(DIMS, odims, out_data);
	return 0;
}

