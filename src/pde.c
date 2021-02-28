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
#include "misc/nested.h"
#include "misc/opts.h"



static void calc_j(const long N, const struct linop_s *d_op,
		   const long j_dims[N], complex float *j,
		   const long dims[N], const complex float *mask2,
		   const complex float *phi)
{
	long mask_strs[N], j_strs[N];
	md_calc_strides(N, j_strs, j_dims, CFL_SIZE);
	md_calc_strides(N, mask_strs, dims, CFL_SIZE);

	linop_forward(d_op, N, j_dims, j, N, dims, phi); // minus phi, probably

	md_zmul2(N, j_dims, j_strs, j, mask_strs, mask2, j_strs, j);
}

struct process_dat {
	complex float *j;
	const long N;
	const long *dims;
	const long *j_dims;
	complex float *mask2;
	struct linop_s *op;
	unsigned long hist;
};

static complex float *j_wrapper(void *_data, const float *phi)
{
	auto data = (struct process_dat *)_data;
	calc_j(data->N, data->op, data->j_dims, data->j, data->dims, data->mask2, (complex float *)phi);
	return data->j;
}

static bool selector(const unsigned long iter, const float *x, void *_data)
{
		UNUSED(x);
		struct process_dat *data = _data;
		return data->hist > 0 ? (0 == iter % data->hist) : false;
}



static const char usage_str[] = "<boundary> <electrodes> <output_phi> <output_j>";
static const char help_str[] = "Solve 3D Laplace equation with Neumann Boundary Conditions";

int main_pde(int argc, char *argv[])
{
	long N = 4;
	int iter = 100;
	int hist = -1;
	const struct opt_s opts[] = {
	    OPT_INT('n', &iter, "n", "Iterations"),
	    OPT_INT('p', &hist, "n", "Save ∇ɸ every n iterations"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	long dims_in1[N], dims_in2[N];
	const long vec_dim = 0;

	complex float *mask = load_cfl(argv[1], N, dims_in1);
	complex float *electrodes = load_cfl(argv[2], N, dims_in2);

	for (int i = 0; i < N; i++)
		assert(dims_in2[i] == dims_in1[i]);

	assert(1 == dims_in1[vec_dim]);

	long vec3_dims[N];
	md_copy_dims(N, vec3_dims, dims_in1);
	vec3_dims[vec_dim] = N - 1;

	long *vec1_dims = dims_in1;
	long *scalar_dims = dims_in1 + 1;
	long scalar_N = N - 1;

	//Boundary Values
	complex float *normal = md_alloc(N, vec3_dims, CFL_SIZE);
	calc_outward_normal(N, vec3_dims, normal, vec_dim, vec1_dims, mask);

	struct boundary_point_s *boundary = md_alloc(scalar_N, scalar_dims, sizeof(struct boundary_point_s));
	long n_points = calc_boundary_points(N, vec3_dims, boundary, vec_dim, normal, electrodes);

	complex float *mask2 = md_calloc(scalar_N, scalar_dims, CFL_SIZE);
	shrink_wrap(scalar_N, scalar_dims, mask2, n_points, boundary, mask);

	// setup monitoring and solver
	complex float *j_hist = md_calloc(N, vec3_dims, CFL_SIZE);
	auto d_op = linop_fd_create(N, vec1_dims, vec_dim, ((MD_BIT(N) - 1) & ~MD_BIT(vec_dim)), 2, BC_ZERO, false);

	struct process_dat j_dat = {.N = N, .j_dims = vec3_dims, .j = j_hist, .dims = vec1_dims, .mask2 = mask2, .op = d_op, .hist = hist};

	auto mon = create_monitor_recorder(N, vec3_dims, "j_step", (void *)&j_dat, selector, j_wrapper);

	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = iter;
	conf.l2lambda = 1e-5;
	long size = 2 * md_calc_size(scalar_N, scalar_dims); // multiply by 2 for float size


	//LHS
	auto diff_op = linop_laplace_neumann_create(scalar_N, scalar_dims, mask, n_points, boundary);

	//x
	complex float *phi = md_calloc(scalar_N, scalar_dims, CFL_SIZE);

	//RHS
	complex float *rhs = md_calloc(scalar_N, scalar_dims, CFL_SIZE);
	laplace_neumann_update_rhs(scalar_N, scalar_dims, rhs, n_points, boundary);

	//solve
	iter_conjgrad(CAST_UP(&conf), diff_op->forward, NULL, size, (float *)phi, (const float *)rhs, mon);

	//save
	complex float *out = create_cfl(argv[3], scalar_N, scalar_dims);
	md_copy(scalar_N, scalar_dims, out, phi, CFL_SIZE);

	complex float *j_out = create_cfl(argv[4], N, vec3_dims);
	calc_j(N, d_op, vec3_dims, j_out, vec1_dims, mask2, phi);

	md_free(normal);
	md_free(boundary);
	md_free(phi);
	md_free(rhs);
	md_free(mask2);
	linop_free(diff_op);
	linop_free(d_op);

	md_free(j_hist);
	xfree(mon);

	unmap_cfl(N, dims_in1, mask);
	unmap_cfl(N, dims_in1, electrodes);
	unmap_cfl(scalar_N, scalar_dims, out);
	unmap_cfl(N, vec3_dims, j_out);
	return 0;
}

