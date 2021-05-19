/* Copyright 2013, 2016. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <stdio.h>
#include <stdbool.h>
#include <complex.h>

#include "linops/linop.h"
#include "linops/someops.h"

#include "num/fft.h"
#include "num/filter.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"

#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"

#include "moba/meco.h"

#include "simu/signals.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<TE> <multi-echo images> <W/F>";
static const char help_str[] =
		"Perfoms simple 3-pt Dixon water/fat separation.";


int main_dixon(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 3, usage_str, help_str);

	num_init();

	const int N = DIMS;


	long dims_TE[N];
	complex float* TE = load_cfl(argv[1], N, dims_TE);

	long dims_in[N];
	complex float* in = load_cfl(argv[2], N, dims_in);

	long strs_in[N];
	md_calc_strides(N, strs_in, dims_in, CFL_SIZE);


	long dims_img[N];
	md_select_dims(N, ~TE_FLAG, dims_img, dims_in);

	long strs_img[N];
	md_calc_strides(N, strs_img, dims_img, CFL_SIZE);


	for (int D = 3; D < N; D++)
		assert(dims_TE[D]==dims_in[D]);


	complex float* pm = md_alloc(N, dims_in, CFL_SIZE);
	md_zarg(N, dims_in, pm, in);

	// TODO: B0 mapping
	complex float* pB0  = md_alloc(N, dims_img, CFL_SIZE);
	md_zsub2(N, dims_img, strs_img, pB0, strs_in, (void*)pm + strs_in[TE_DIM] * 1, strs_in, (void*)pm + strs_in[TE_DIM] * 0);
	md_zsmul(N, dims_img, pB0, pB0, -1./(TE[1]-TE[0]));


	for (int n = 0; n < dims_TE[TE_DIM]; n++) {

		md_zsmul2(N, dims_img, strs_in, (void*)pm + strs_in[TE_DIM] * n, strs_img, pB0, TE[n]);

		md_zexpj2(N, dims_img, strs_in, (void*)pm + strs_in[TE_DIM] * n, strs_in, (void*)pm + strs_in[TE_DIM] * n);

		md_zmul2(N, dims_img, strs_in, (void*)pm + strs_in[TE_DIM] * n, strs_in, (void*)in + strs_in[TE_DIM] * n, strs_in, (void*)pm + strs_in[TE_DIM] * n);

	}

	xfree(pB0);


	long* pos = calloc(N, sizeof(long));

	pos[TE_DIM] = 0;
	complex float* E0 = (void*)pm + md_calc_offset(N, strs_in, pos);

	pos[TE_DIM] = 1;
	complex float* E1 = (void*)pm + md_calc_offset(N, strs_in, pos);

	pos[TE_DIM] = 2;
	complex float* E2 = (void*)pm + md_calc_offset(N, strs_in, pos);



	complex float* cshift = md_alloc(N, dims_TE, CFL_SIZE);
	meco_calc_fat_modu(N, dims_TE, TE, cshift, FAT_SPEC_1);

	complex float lhs = 0.;



	long dims_out[N];
	md_copy_dims(N, dims_out, dims_img);
	dims_out[COEFF_DIM] = 2;

	long strs_out[N];
	md_calc_strides(N, strs_out, dims_out, CFL_SIZE);

	complex float* out = create_cfl(argv[3], N, dims_out);

	for (int D = 0; D < N; D++)
		pos[D] = 0;

	pos[COEFF_DIM] = 0;
	complex float* W = (void*)out + md_calc_offset(N, strs_out, pos);

	pos[COEFF_DIM] = 1;
	complex float* F = (void*)out + md_calc_offset(N, strs_out, pos);


	lhs = 1./ ( 2 * cshift[1] - cshift[0] - cshift[2] );

	md_zsmul(N, dims_img, F, E1, 2.);
	md_zsub(N, dims_img, F, F, E0);
	md_zsub(N, dims_img, F, F, E2);

	md_zsmul(N, dims_img, F, F, lhs);


	complex float a = cshift[0] * cshift[2] * 2.;
	complex float b = cshift[1] * cshift[2];
	complex float c = cshift[0] * cshift[1];

	lhs = 1./ ( a - b - c );

	md_zsmul(N, dims_img, W, E1, a);
	md_zaxpy(N, dims_img, W, -1.*b, E0);
	md_zaxpy(N, dims_img, W, -1.*c, E2);

	md_zsmul(N, dims_img, W, W, lhs);


	xfree(pm);
	xfree(pos);
	xfree(cshift);


	unmap_cfl(N, dims_TE, TE);
	unmap_cfl(N, dims_in, in);
	unmap_cfl(N, dims_out, out);

	exit(0);
}


