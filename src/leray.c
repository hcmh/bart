#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linops/grad.h"
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

#include "simu/fd_geometry.h"
#include "simu/leray.h"

#include <math.h>

#define N 4


struct process_dat {
	const complex float *j;
	complex float *j_hist;
	const long *dims;
	const long *j_dims;
	complex float *mask2;
	struct linop_s *op;
};


static complex float *j_wrapper(void *_data, const float *phi)
{
	auto data = (struct process_dat *)_data;

	long mask_strs[N], j_strs[N];
	md_calc_strides(N, j_strs, data->j_dims, CFL_SIZE);
	md_calc_strides(N, mask_strs, data->dims, CFL_SIZE);

	linop_forward(data->op, N, data->j_dims, data->j_hist, N, data->dims, (const complex float *)phi);

	md_zmul2(N, data->j_dims, j_strs, data->j_hist, mask_strs, data->mask2, j_strs, data->j_hist);

	md_zaxpy(N, data->j_dims, data->j_hist, 1, data->j);

	return data->j_hist;
}

static const char usage_str[] = "vectordimension <mask> <input> <output>";
static const char help_str[] = "Project a vectorfield onto it's divergence free component\n";

int main_leray(int argc, char *argv[])
{
	int n = 30;
	float lambda = 1e-3;
	long hist = -1;
	const struct opt_s opts[] = {
	    OPT_INT('n', &n, "n", "Number of Iterations"),
	    OPT_LONG('p', &hist, "n", "Record all p steps"),
	    OPT_FLOAT('l', &lambda, "lambda", "Inversion l2 regularization"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();

	long dims[N] = {}, mask_dims[N] = {};
	unsigned int d = atoi(argv[1]);

	complex float *in = load_cfl(argv[3], N, dims);
	complex float *out = create_cfl(argv[4], N, dims);

	complex float *mask = load_cfl(argv[2], N, mask_dims);
	for (unsigned int i = 0; i < N; i++)
		assert(i == d ? mask_dims[i] == 1 : mask_dims[i] == dims[i]);

	// setup monitoring
	long vec1_dims[N];
	md_copy_dims(N, vec1_dims, dims);
	vec1_dims[d] = 1;
	complex float *j_hist = md_calloc(N, dims, CFL_SIZE);
	auto d_op = linop_fd_create(N, vec1_dims, d, ((MD_BIT(N) - 1) & ~MD_BIT(d)), 2, BC_ZERO, false);

	complex float *normal = md_alloc(N, dims, CFL_SIZE);
	calc_outward_normal(N, dims, normal, d, vec1_dims, mask);

	struct boundary_point_s *boundary = md_alloc(N, vec1_dims, sizeof(struct boundary_point_s));
	long n_points = calc_boundary_points(N, dims, boundary, d, normal, NULL);

	complex float *mask2 = md_calloc(N, vec1_dims, CFL_SIZE);
	shrink_wrap(N - 1, dims + 1, mask2, n_points, boundary, mask);

	struct process_dat j_dat = {.j_dims = dims, .j = in, .j_hist = j_hist, .dims = vec1_dims, .mask2 = mask2, .op = d_op};

	NESTED(bool, selector, (const unsigned long iter, const float *x, void *_data))
	{
		UNUSED(x);
		UNUSED(_data);
		return hist > 0 ? (0 == iter % hist) : false;
	};
	auto mon = create_monitor_recorder(N, dims, "j_step", (void *)&j_dat, selector, j_wrapper);



	auto op = linop_leray_create(N, dims, d, n, lambda, mask, mon);
	linop_forward(op, N, dims, out, N, dims, in);
	linop_free(op);

	md_free(normal);
	md_free(boundary);
	md_free(mask2);
	md_free(j_hist);

	unmap_cfl(N, dims, in);
	unmap_cfl(N, dims, out);
	unmap_cfl(N, mask_dims, mask);

	return 0;
}

