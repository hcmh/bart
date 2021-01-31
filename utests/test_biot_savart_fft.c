/*
 * Authors:
 * 2020 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */


#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "linops/linop.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "simu/biot_savart_fft.h"


#include "utest.h"

#define N 4

static bool test_biot_savart_fft(void)
{
	// Geometry Settings
	const float center_x = 50, center_y = 20, center_z = 30;
	const long dims[N] = {3, 2 * center_x, 2 * center_y, 2 * center_z};
	float fovscale = .1;
	const float fov[N] = {1 * fovscale, 10 * fovscale, 1 * fovscale};
	long r = 12;
	float h = 0.75;
	int dmin = 15, dmax = center_z - 3;

	// Setup FoV etc
	long bdims[N], bstrs[N];
	md_select_dims(N, 14, bdims, dims);
	md_calc_strides(N, bstrs, bdims, CFL_SIZE);
	float voxelsize[3];
	fov_to_vox(voxelsize, dims + 1, fov);
	assert(2 * center_x > (int)(dmax * voxelsize[2] / voxelsize[0]) + center_x);

	// Create cylindric current density along y-axis
	complex float *j = md_calloc(N, dims, CFL_SIZE);

	jcylinder(dims, fov, voxelsize[0] * r, fov[1] * h, 1, j);

	// calculate B_z
	complex float *b = md_alloc(N, bdims, CFL_SIZE);
	biot_savart_fft(dims, voxelsize, b, j);

	// scale: hz -> tesla
	md_zsmul(N, bdims, b, b, 1. / Hz_per_Tesla);

	// ∮_{square wireloop in xz-plane} ∇ x B
	// = 4 * ∫_{one side of the square} B_z dz
	// = µ_0 * I
	float maxdev = 0;

	// for different slices along the cylinder
	for (int ysl = center_y - 3; ysl < center_y + 3; ysl++) {
		// take a _square_ with size 2*d
		for (int d = dmin; d <= dmax; d++) {
			float current = 0;
			// integrate bz along one side of the square
			for (int z = center_z - d + 1; z <= center_z + d; z++) {
				long pos[] = {0, (int)(d * voxelsize[2] / voxelsize[0] + center_x), ysl, z};
				long offset = md_calc_offset(N, bstrs, pos);
				current += *(complex float *)((void *)b + offset) * voxelsize[2] / Mu_0;
			}
			current *= 4;
			maxdev = fabs(current + 1) > maxdev ? fabs(current + 1) : maxdev;
			debug_printf(DP_DEBUG1, "Slice: %d; Distance: %d; Amps law: %f\n", ysl, d, current);
		}
	}
	debug_printf(DP_DEBUG1, "Maximal difference: %f\n", maxdev);
	md_free(b);
	md_free(j);

	return maxdev < 0.02;
}

UT_REGISTER_TEST(test_biot_savart_fft);
