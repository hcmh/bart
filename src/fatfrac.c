/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 * 
 * Reference:
 * 
 * Berglund, J, Johansson L, Ahlstroem H, Kullberg J.
 * Three-point Dixon method enables whole-body water and fat imaging of obese subjects.
 * Magn Reson Med 2010;63:1659-1668
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/mri.h"


static const char usage_str[] = "<input Water> <input Fat> <output>";
static const char help_str[] = "Compute fat fraction maps from Water and Fat.\n";


int main_fatfrac(int argc, char* argv[argc])
{
	mini_cmdline(&argc, argv, 3, usage_str, help_str);

	num_init();

	long W_dims[DIMS];
	long F_dims[DIMS];
	complex float* W = load_cfl(argv[1], DIMS, W_dims);
	complex float* F = load_cfl(argv[2], DIMS, F_dims);

	for (long ind = 0; ind < DIMS; ind++)
		assert(W_dims[ind] == F_dims[ind]);

	complex float* W_abs = md_alloc(DIMS, W_dims, CFL_SIZE);
	md_zabs(DIMS, W_dims, W_abs, W);

	complex float* F_abs = md_alloc(DIMS, F_dims, CFL_SIZE);
	md_zabs(DIMS, F_dims, F_abs, F);

	complex float* rho = md_alloc(DIMS, W_dims, CFL_SIZE);

	md_zadd(DIMS, W_dims, rho, W_abs, F_abs);

	long fatfrac_dims[DIMS];
	md_copy_dims(DIMS, fatfrac_dims, W_dims);

	complex float* fatfrac = create_cfl(argv[3], DIMS, fatfrac_dims);


	long p = 0;
	long pos[DIMS] = { 0 };

	do {

		if ( cabsf(F[p]) > cabsf(W[p]) )
			fatfrac[p] = cabsf( F[p] / rho[p] );
		else
			fatfrac[p] = 1 - cabsf( W[p] / rho[p] );

		p++;

	} while (md_next(DIMS, W_dims, ~0L, pos));

	unmap_cfl(DIMS, W_dims, W);
	unmap_cfl(DIMS, F_dims, F);
	unmap_cfl(DIMS, fatfrac_dims, fatfrac);

	md_free(rho);
	md_free(W_abs);
	md_free(F_abs);

	return 0;
}




