#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "simu/fd_geometry.h"
#include "simu/sparse.h"
#include "linops/grad.h"

#include "iter/lsqr.h"

#include "iter/monitor.h"
#include "misc/debug.h"

#include "linops/linop.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/nested.h"

#include <math.h>


static void calc_j(const long N, const long d, const struct linop_s *d_op,
			const long j_dims[N],  complex float* j,
			const long   dims[N], const complex float* mask2,
			const complex float* phi)
{
	long mask_strs[N], j_strs[N];
	md_calc_strides(N, j_strs, j_dims, CFL_SIZE);
	md_calc_strides(N, mask_strs, dims, CFL_SIZE);

	linop_forward(d_op, N, j_dims, j, N, dims, phi); // minus phi, probably

	mask_strs[d] = 0;
	md_zmul2(N, j_dims, j_strs, j, mask_strs, mask2, j_strs, j);
}


struct process_dat {
	complex float *j;
	const long N;
	const long *dims;
	const long *j_dims;
	complex float* mask2;
	struct linop_s * op;
};


static complex float *j_wrapper(void *_data, const float* phi)
{
	auto data = (struct process_dat*)_data;
	calc_j(data->N, data->N-1, data->op, data->j_dims, data->j, data->dims, data->mask2, (complex float*)phi);
	return data->j;
}


static const char usage_str[] = "<boundary> <electrodes> <output_phi> <output_j>";
static const char help_str[] = "Solve 3D Laplace equation with Neumann Boundary Conditions";

int main_pde(int argc, char* argv[])
{
	long N = 4;
	int iter=100;
	int hist = -1;
	const struct opt_s opts[] = {
		OPT_INT('n', &iter, "n", "Iterations"),
		OPT_INT('p', &hist, "n", "Save ∇ɸ every n iterations"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	long dims[N], dims1[N];

	complex float* mask = load_cfl(argv[1], N, dims);
	complex float* electrodes = load_cfl(argv[2], N, dims1);

	assert(1 == dims[N-1]);

	for(int i = 0; i < N; i++)
		assert(dims[i] == dims1[i]);

	//Boundary Values
	long grad_dim = N - 1;
	long grad_dims[N];
	md_copy_dims(N, grad_dims, dims);
	grad_dims[grad_dim] = N - 1;

	complex float *normal = md_alloc(N, grad_dims, CFL_SIZE);
	calc_outward_normal(N, grad_dims, normal, grad_dim, dims, mask);

	const long boundary_dimensions[] =  { md_calc_size(N, dims) };
	struct boundary_point_s *boundary = md_alloc(1, boundary_dimensions, sizeof(struct boundary_point_s));
	long n_points = calc_boundary_points(N, grad_dims, boundary, grad_dim, normal, electrodes);

	complex float *mask2 = md_calloc(N, dims, CFL_SIZE);
	shrink_wrap(N, dims, mask2, n_points, boundary, mask);
	//\Boundary Values//

	//create laplace
	struct sparse_diag_s *mat = sd_laplace_create(N - 1, dims);

	//apply boundary
	laplace_neumann(mat, N - 1, dims, n_points, boundary);
	sd_mask(N-1, dims, mat, mask);

	// LHS
	long vec_dim[1] = { mat->len };
	auto diff_op = linop_sd_matvec_create(1, vec_dim, mat);

	// RHS
	complex float *rhs = md_calloc(1, vec_dim, CFL_SIZE);
	set_laplace_neumann_rhs(N - 1, dims, rhs, n_points, boundary);

	//phi
	complex float *phi = md_calloc(1, vec_dim, CFL_SIZE);

	// setup monitoring
	complex float *j_hist = md_calloc(N, grad_dims, CFL_SIZE);
	auto d_op = linop_fd_create(N, dims, grad_dim, 7, 2, BC_ZERO, false);

	struct process_dat j_dat = { .N=N, .dims = dims, .j_dims = grad_dims, .mask2 = mask2, .j = j_hist, .op = d_op };

	NESTED(bool, selector, (const unsigned long iter, const float *x, void *_data))
	{
		UNUSED(x);
		UNUSED(_data);
		return hist > 0 ? ( 0 == iter % 30 ) : false;
	};
	auto mon = create_monitor_recorder(N, grad_dims, "j_step", (void *)&j_dat, selector, j_wrapper);

	//solve
	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = iter;
	conf.l2lambda = 1e-5;

	long size = 2 * md_calc_size(1, vec_dim); // multiply by 2 for float size
	iter_conjgrad(CAST_UP(&conf), diff_op->forward, NULL, size, (float*)phi, (const float*)rhs, mon);

	//save
	complex float* out = create_cfl(argv[3], N, dims);
	md_copy(N, dims, out, phi, CFL_SIZE);

	complex float* j_out = create_cfl(argv[4], N, grad_dims);
	calc_j(N, N-1, d_op, grad_dims, j_out, dims, mask2, phi);

	//cleanup
	md_free(normal);
	md_free(boundary);
	md_free(phi);
	md_free(rhs);
	md_free(mask2);
	linop_free(diff_op);
	linop_free(d_op);

	md_free(j_hist);
	xfree(mon);

	unmap_cfl(N, dims, mask);
	unmap_cfl(N, dims, electrodes);
	unmap_cfl(N, dims, out);
	unmap_cfl(N, dims, j_out);
	return 0;
}


