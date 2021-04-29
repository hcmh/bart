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

static const char help_str[] = "Given a current density, calculate the BZ-Component of the magnetic induction";

int main_bfield(int argc, char *argv[argc])
{
	float vox[3] = { 0. };
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_FLVEC3(true, &vox, "voxelsize_x:voxelsize_y:voxelsize_z"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);
	num_init();

	long jdims[N] = {}, bdims[N];

	complex float *j = load_cfl(in_file, N, jdims);
	assert(jdims[VECDIM] == N - 1);

	md_copy_dims(N, bdims, jdims);
	bdims[VECDIM] = 1;
	complex float *b = create_cfl(out_file, N, bdims);

	auto bz_op = linop_bz_create(jdims, vox);

	md_zsmul(N, jdims, j, j, bz_unit(bdims + 1, vox));
	linop_forward(bz_op, N, bdims, b, N, jdims, j);
	md_zsmul(N, bdims, b, b, Hz_per_Tesla * Mu_0);

	unmap_cfl(N, jdims, j);
	unmap_cfl(N, bdims, b);
	return 0;
}
