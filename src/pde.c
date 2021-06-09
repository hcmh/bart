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

#define VEC_DIM 0


static void calc_j(const long N, const struct linop_s *d_op,
		   const long j_dims[N], complex float *j,
		   const long dims[N], const complex float *mask,
		   const long n_points, const struct boundary_point_s boundary[n_points],
		   const complex float *phi)
{
	long mask_strs[N], j_strs[N];
	md_calc_strides(N, j_strs, j_dims, CFL_SIZE);
	md_calc_strides(N, mask_strs, dims, CFL_SIZE);

	linop_forward(d_op, N, j_dims, j, N, dims, phi); // minus phi, probably

	md_zmul2(N, j_dims, j_strs, j, j_strs, j, mask_strs, mask);
	neumann_set_boundary(N, j_dims, VEC_DIM, j, n_points, boundary, j);
}

struct process_dat {
	complex float *j;
	const long N;
	const long *dims;
	const long *j_dims;
	complex float *mask;
	struct linop_s *op;
	unsigned long hist;
	long n_points;
	struct boundary_point_s *boundary;
};

static complex float *j_wrapper(void *_data, const float *phi)
{
	auto data = (struct process_dat *)_data;
	md_clear(data->N, data->j_dims, data->j, CFL_SIZE);
	calc_j(data->N, data->op, data->j_dims, data->j, data->dims, data->mask, data->n_points, data->boundary, (complex float *)phi);
	return data->j;
}

static bool selector(const unsigned long iter, const float *x, void *_data)
{
		UNUSED(x);
		struct process_dat *data = _data;
		return data->hist > 0 ? (0 == iter % data->hist) : false;
}



static const char help_str[] = "Solve 3D Laplace equation with Neumann Boundary Conditions";

int main_pde(int argc, char *argv[argc])
{
	const char* boundary_file = NULL;
	const char* electrodes_file = NULL;
	const char* out_phi_file = NULL;
	const char* out_J_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &boundary_file, "boundary"),
		ARG_INFILE(true, &electrodes_file, "electrodes"),
		ARG_OUTFILE(true, &out_phi_file, "output_phi"),
		ARG_OUTFILE(true, &out_J_file, "output_j"),
	};


	long N = 4;
	int iter = 100;
	int hist = -1;
	const struct opt_s opts[] = {
	    OPT_INT('n', &iter, "n", "Iterations"),
	    OPT_INT('p', &hist, "n", "Save ∇ɸ every n iterations"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	long dims_in1[N], dims_in2[N];

	complex float *mask = load_cfl(boundary_file, N, dims_in1);
	complex float *electrodes = load_cfl(electrodes_file, N, dims_in2);

	for (int i = 0; i < N; i++)
		assert(dims_in2[i] == dims_in1[i]);

	assert(1 == dims_in1[VEC_DIM]);

	long vec3_dims[N];
	md_copy_dims(N, vec3_dims, dims_in1);
	vec3_dims[VEC_DIM] = N - 1;

	long *vec1_dims = dims_in1;
	long *scalar_dims = dims_in1 + 1;
	long scalar_N = N - 1;

	//Boundary Values
	complex float *normal = md_alloc(N, vec3_dims, CFL_SIZE);
	calc_outward_normal(N, vec3_dims, normal, VEC_DIM, vec1_dims, mask);

	struct boundary_point_s *boundary = md_alloc(scalar_N, scalar_dims, sizeof(struct boundary_point_s));
	long n_points = calc_boundary_points(N, vec3_dims, boundary, VEC_DIM, normal, electrodes);

	// setup monitoring and solver
	auto d_op = linop_fd_create(N, vec1_dims, VEC_DIM, ((MD_BIT(N) - 1) & ~MD_BIT(VEC_DIM)), 2, BC_ZERO, false);
	struct process_dat j_dat = {.N = N, .j_dims = vec3_dims, .dims = vec1_dims, .mask = mask, .op = d_op, .hist = hist, .j = NULL, .n_points = n_points, .boundary = boundary};
	j_dat.j = md_calloc(j_dat.N, j_dat.j_dims, CFL_SIZE);
	auto mon = create_monitor_recorder(N, vec3_dims, "j_step", (void *)&j_dat, selector, j_wrapper);
	//auto mon = create_monitor_recorder(N, vec1_dims, "phi_step", (void *)&j_dat, selector, NULL);

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
	complex float *out = create_cfl(out_J_file, scalar_N, scalar_dims);
	md_copy(scalar_N, scalar_dims, out, phi, CFL_SIZE);

	complex float *j_out = create_cfl(out_J_file, N, vec3_dims);
	calc_j(N, d_op, vec3_dims, j_out, vec1_dims, mask, n_points, boundary, phi);

	md_free(normal);
	md_free(boundary);
	md_free(phi);
	md_free(rhs);
	linop_free(diff_op);
	linop_free(d_op);
	md_free(j_dat.j);
	xfree(mon);

	unmap_cfl(N, dims_in1, mask);
	unmap_cfl(N, dims_in1, electrodes);
	unmap_cfl(scalar_N, scalar_dims, out);
	unmap_cfl(N, vec3_dims, j_out);
	return 0;
}
