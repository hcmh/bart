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


#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "rkhs/rkhs.h"

#ifdef RKHSGRID
#include "noncart/grid.h"
#endif

static const char* usage_str = "<input> <output>";
static const char* help_str = "";


int main_kernel(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 2, usage_str, help_str);

	const int N = DIMS;
	long idims[N];
	complex float* sens = load_cfl(argv[1], N, idims);

	assert(1 == idims[5]);
	idims[5] = idims[MAPS_DIM];
	idims[MAPS_DIM] = 1;

	long str1[N];
	md_calc_strides(N, str1, idims, CFL_SIZE);


	long odims[N];
	md_copy_dims(N, odims, idims);
	odims[4] = idims[3];
	odims[3] = 1;

	long str2[N];
	md_calc_strides(N, str2, odims, CFL_SIZE);

	odims[3] = idims[3];
	odims[5] = 1;


	long stro[N];
	md_calc_strides(N, stro, odims, CFL_SIZE);


	complex float* sens2 = md_alloc(N, odims, CFL_SIZE);
	md_clear(N, odims, sens2, CFL_SIZE);
	md_zfmacc2(N, odims, stro, sens2, str1, sens, str2, sens);


	long dims2[N];
	md_copy_dims(N, dims2, odims);

	for (int i = 0; i < 3; i++)
		if (odims[i] > 1)
			dims2[i] = odims[i] * KERNEL_OVERSAMPLING;

	complex float* data = create_cfl(argv[2], N, dims2);

	md_resize_center(N, dims2, data, odims, sens2, CFL_SIZE);

	free(sens2);

#ifdef RKHXXSGRID
	// FIXME: make sure that rolloff is calculated for oversampled grid

	long rdims[N];
	md_select_dims(N, FFT_FLAGS, rdims, dims2);

	complex float* rolloff = md_alloc(N, rdims, CFL_SIZE);
	rolloff_correction(rdims, rolloff);

	long dstr[N];
	md_calc_strides(N, dstr, dims2, CFL_SIZE);

	long rstr[N];
	md_calc_strides(N, rstr, rdims, CFL_SIZE);
	md_zmul2(N, dims2, dstr, data, dstr, data, rstr, rolloff);
	free(rolloff);
#endif


	fftscale(N, dims2, FFT_FLAGS, data, data);
	ifftc(N, dims2, FFT_FLAGS, data, data);

	idims[MAPS_DIM] = idims[5];
	idims[5] = 1;
	unmap_cfl(N, idims, sens);
	unmap_cfl(N, dims2, data);

	exit(0);
}


