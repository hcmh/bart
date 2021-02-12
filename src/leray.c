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
	enum BOUNDARY_CONDITION bc = BC_SAME;
	const struct opt_s opts[] = {
		OPT_INT('o', &order, "order", "Order (1 (default),2)"),
		OPT_SELECT('P', enum BOUNDARY_CONDITION, &bc, BC_PERIODIC, "Boundary condition: Periodic"),
		OPT_SELECT('Z', enum BOUNDARY_CONDITION, &bc, BC_ZERO, "Boundary condition: Zero"),
		OPT_SELECT('S', enum BOUNDARY_CONDITION, &bc, BC_SAME, "Boundary condition: Same (default)"),
		OPT_INT('n', &n, "n", "Number of Iterations"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	assert(1 == order || 2 == order);
	long dims[N] = {};
	unsigned int d = atoi(argv[1]);
	long flags = atoi(argv[2]);

	complex float* in = load_cfl(argv[3], N, dims);
	complex float* out = create_cfl(argv[4], N, dims);

	auto op = linop_leray_create(N, dims, d, flags, order, bc, n);
	linop_forward(op, N, dims, out, N, dims, in);
	linop_free(op);

	unmap_cfl(N, dims, in);
	unmap_cfl(N, dims, out);
	return 0;
}


