#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linops/linop.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"
#include "simu/biot_savart_fft.h"

#include "misc/io.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include <math.h>

#define N 4
#define VECDIM 0

static const char usage_str[] = "voxelsize(x) voxelsize(y) voxelsize(z) <input> <output>";
static const char help_str[] = "Given a current density, calculate the BZ-Component of the magnetic induction\n";

int main_bfield(int argc, char *argv[])
{
	cmdline(&argc, argv, 5, 5, usage_str, help_str, 0, NULL);
	num_init();

	long jdims[N] = {}, bdims[N];

	complex float *j = load_cfl(argv[4], N, jdims);
	assert(jdims[VECDIM] == N - 1);

	md_copy_dims(N, bdims, jdims);
	bdims[VECDIM] = 1;
	complex float *b = create_cfl(argv[5], N, bdims);

	const int vox_ind = 1;
	float vox[3] = {strtof(argv[vox_ind], NULL), strtof(argv[vox_ind + 1], NULL), strtof(argv[vox_ind + 2], NULL)};

	auto bz_op = linop_bz_create(jdims, vox);

	md_zsmul(N, jdims, j, j, bz_unit(bdims + 1, vox));
	linop_forward(bz_op, N, bdims, b, N, jdims, j);
	md_zsmul(N, bdims, b, b, Hz_per_Tesla * Mu_0);

	unmap_cfl(N, jdims, j);
	unmap_cfl(N, bdims, b);
	return 0;
}

