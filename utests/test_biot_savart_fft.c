/*
 * Authors:
 * 2020 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */


#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "linops/linop.h"

#include "simu/biot_savart_fft.h"

#include "utest.h"



static bool test_linop_bz(void)
{
	const long N = 4;
	// create cylindric current density along y-axis
	float center = 40, cylcenter = 20;
	long dims[] = {3, 2 * center, 2 * cylcenter, 2 * center}, bdims[N], jdims[N];
	md_select_dims(N, 14, bdims, dims);
	float fovscale = .1;
	const float fov[] = {1 * fovscale, 10 * fovscale, 1 * fovscale};
	const float voxelsize[] = {fov[0] / dims[1], fov[1] / dims[2], fov[2] / dims[3]};
	float r = voxelsize[0] * 12, h = fov[1] * 0.75;
	complex float* jfull = md_alloc(N, dims, CFL_SIZE);
	md_clear(N, dims, jfull, CFL_SIZE);
	jcylinder(dims, fov, r, h, 1, jfull);

	// take jx, jy
	md_select_dims(N, 14, jdims, dims);
	jdims[0] = 2;

	complex float* j = md_alloc(N, jdims, CFL_SIZE);

	md_resize(N, jdims, j, dims, jfull, CFL_SIZE);

	md_free(jfull);

	// calculate B_z

	complex float* b = md_alloc(N, bdims, CFL_SIZE);
	auto bz = linop_bz_create(jdims, fov);

	linop_forward(bz, N, bdims, b, N, jdims, j);

	linop_free(bz);
	md_free(j);

	// scale
	complex float fov_factor = bz_unit(3, fov);
	md_zsmul(N, bdims, b, b, fov_factor);

	//apply amperes law to reconstruct the total current through the cylinder
	float maxdev = 0;
	int dmin = 20, dmax = center - 10;

	// for different slices along the cylinder
	for(int ysl = cylcenter - 3; ysl < cylcenter + 3; ysl++) {
		// take a square with size 2*d
		for (int d = dmin; d <= dmax; d++) {
			float current = 0;
			long start = (center + d) * bdims[0] + ysl * bdims[0] * bdims[1];
			// integrate bz along one side of the square
			for (int i = center - d + 1; i <= center + d; i++) {
				current += *(b + start + i * bdims[0] * bdims[1] * bdims[2]) * voxelsize[2];
			}
			current *= 4;
			maxdev = fabs(current + 1) > maxdev ? fabs(current + 1) : maxdev;
			debug_printf(DP_DEBUG1, "Slice: %d; Distance: %d; Amps law: %f; Max. Dev: %f\n", ysl, d, current, maxdev);
		}
	}

	md_free(b);

	return maxdev < 0.1;
}

UT_REGISTER_TEST(test_linop_bz);
