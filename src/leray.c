#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/ops.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "simu/leray.h"

#include <math.h>

#define N 4

static const char usage_str[] = "vectordimension bitmask <input> <output>";
static const char help_str[] = "Project a vectorfield onto it's divergence free component\n";

int main_leray(int argc, char* argv[])
{
	int order=1, n=30;
	const char *mask_file = NULL;
	float lambda = 1e-3;
	enum BOUNDARY_CONDITION bc = BC_SAME;
	const struct opt_s opts[] = {
		OPT_INT('o', &order, "order", "Order (1 (default),2)"),
		OPT_SELECT('P', enum BOUNDARY_CONDITION, &bc, BC_PERIODIC, "Boundary condition: Periodic"),
		OPT_SELECT('Z', enum BOUNDARY_CONDITION, &bc, BC_ZERO, "Boundary condition: Zero"),
		OPT_SELECT('S', enum BOUNDARY_CONDITION, &bc, BC_SAME, "Boundary condition: Same (default)"),
		OPT_INT('n', &n, "n", "Number of Iterations"),
		OPT_FLOAT('l', &lambda, "lambda", "Inversion l2 regularization"),
		OPT_STRING('m', &mask_file, "mask cfl file", "Masked Leray projection"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	assert(1 == order || 2 == order);
	long dims[N] = {}, mask_dims[N] = {};
	unsigned int d = atoi(argv[1]);
	long flags = atoi(argv[2]);

	complex float* in = load_cfl(argv[3], N, dims);
	assert(bitcount(flags) == dims[d]);

	complex float* out = create_cfl(argv[4], N, dims);

	complex float* mask = NULL;
	if (NULL != mask_file) {
		complex float *mask_dat = load_cfl(mask_file, N, mask_dims);
		for (unsigned int i = 0; i < N; i++)
			assert( i == d ? mask_dims[i] == 1 : mask_dims[i] == dims[i]);

		mask = md_alloc(N, dims, CFL_SIZE);
		long pos[N] = {0};
		for (; pos[d] < dims[d]; pos[d]++)
			md_copy_block(N, pos, dims, mask, mask_dims, mask_dat, CFL_SIZE);
		unmap_cfl(N, mask_dims, mask_dat);
	}


	auto op = linop_leray_create(N, dims, d, flags, order, bc, n, lambda, mask);
	linop_forward(op, N, dims, out, N, dims, in);
	linop_free(op);

	unmap_cfl(N, dims, in);
	unmap_cfl(N, dims, out);
	if (NULL != mask_file)
		md_free(mask);
	return 0;
}


