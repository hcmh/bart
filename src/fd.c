#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linops/linop.h"
#include "linops/someops.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"
#include "num/ops.h"

#include "misc/io.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include "linops/grad.h"

#include <math.h>

#define N 16

static const char usage_str[] = "vectordimension bitmask <input> <output>";
static const char help_str[] = "Calculate divergence, gradient or -laplace along directions indicated by bitmask\n";

int main_fd(int argc, char *argv[])
{
	int order = 1;
	enum BOUNDARY_CONDITION bc = BC_PERIODIC;
	enum fd_function { FD_GRAD,
			   FD_DIV,
			   FD_LAPLACE } fd = FD_GRAD;
	const struct opt_s opts[] = {
	    OPT_INT('o', &order, "order", "Order (1,2)"),
	    OPT_SELECT('P', enum BOUNDARY_CONDITION, &bc, BC_PERIODIC, "Boundary condition: Periodic (default)"),
	    OPT_SELECT('Z', enum BOUNDARY_CONDITION, &bc, BC_ZERO, "Boundary condition: Zero"),
	    OPT_SELECT('S', enum BOUNDARY_CONDITION, &bc, BC_SAME, "Boundary condition: Same"),
	    OPT_SELECT('d', enum fd_function, &fd, FD_DIV, "Divergence"),
	    OPT_SELECT('l', enum fd_function, &fd, FD_LAPLACE, "Laplace"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	assert(1 == order || 2 == order);
	long dims[N] = {}, odims[N];
	unsigned int d = atoi(argv[1]);
	long flags = atoi(argv[2]);
	long num_dx = bitcount(flags);

	complex float *in = load_cfl(argv[3], N, dims);

	md_copy_dims(N, odims, dims);
	if (FD_DIV == fd)
		odims[d] = 1;
	else if (FD_GRAD == fd)
		odims[d] = num_dx;

	complex float *out = create_cfl(argv[4], N, odims);

	if (FD_DIV == fd) {
		auto op = linop_div_create(N, dims, d, flags, order, bc);
		linop_forward(op, N, odims, out, N, dims, in);
		linop_free(op);
	} else if (FD_GRAD == fd) {
		auto d_op = linop_fd_create(N, dims, d, flags, order, bc, false);
		linop_forward(d_op, N, odims, out, N, dims, in);
		linop_free(d_op);
	} else if (FD_LAPLACE == fd) {
		auto d_op = linop_fd_create(N, dims, d, flags, order, bc, false);
		linop_normal(d_op, N, odims, out, in);
		linop_free(d_op);
	}

	unmap_cfl(N, dims, in);
	unmap_cfl(N, odims, out);
	return 0;
}

