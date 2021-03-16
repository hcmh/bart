#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"

#include "linops/grad.h"
#include "simu/fd_geometry.h"
#include "simu/pde_laplace.h"
#include "simu/sparse.h"

#include "iter/lsqr.h"

#include "iter/monitor.h"
#include "misc/debug.h"

#include "linops/linop.h"

#include "misc/io.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#define N 4

enum mode_t { SHRINK, FILL_HOLES };

static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Various transformations for masks";

int main_pde_mask(int argc, char *argv[])
{
	enum mode_t mode = SHRINK;
	struct opt_s modeopt[] = {
		OPT_SELECT('s', enum mode_t, &mode, SHRINK, 		"shrink the interior of the image by one pixel at each border"),
		OPT_SELECT('f', enum mode_t, &mode, FILL_HOLES, 	"fill one-pixel wide holes in the mask")
	};

	const struct opt_s opts[] = {
		OPT_SUBOPT('C', "cmd", "Transformation. -Ch for help.", ARRAY_SIZE(modeopt), modeopt),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();

	long dims[N];
	const complex float *in = load_cfl(argv[1], N, dims);
	complex float *out = create_cfl(argv[2], N, dims);

	assert(1 == dims[0]);

	long vec_dim = 0;
	long vec3_dims[N], *vec1_dims = dims;
	md_copy_dims(N, vec3_dims, dims);
	vec3_dims[0] = N - 1;
	const long scalar_N = N - 1;
	const long *scalar_dims = dims + 1;


	if (SHRINK == mode) {
		complex float *normal = md_alloc(N, vec3_dims, CFL_SIZE);

		calc_outward_normal(N, vec3_dims, normal, vec_dim, vec1_dims, in);

		struct boundary_point_s *boundary = md_alloc(scalar_N, scalar_dims, sizeof(struct boundary_point_s));
		long n_points = calc_boundary_points(N, vec3_dims, boundary, vec_dim, normal, NULL);
		shrink_wrap(scalar_N, scalar_dims, out, n_points, boundary, in);

		md_free(normal);
		md_free(boundary);
	} else if (FILL_HOLES == mode) {
		fill_holes(N, vec3_dims, vec_dim, vec1_dims, out, in);
	}

	unmap_cfl(N, dims, in);
	unmap_cfl(N, dims, out);

	return 0;
}
