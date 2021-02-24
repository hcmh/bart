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

#include "linops/linop.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include <math.h>

static const char usage_str[] = "<boundary> <electrodes> <output_phi> <output_j>";
static const char help_str[] = "Solve 3D Laplace equation with Neumann Boundary Conditions";

int main_pde(int argc, char* argv[])
{
	long N = 4;
	int iter=100;
	const struct opt_s opts[] = {
		OPT_INT('n', &iter, "n", "Iterations"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	long dims[N], dims1[N];

	complex float* mask = load_cfl(argv[1], N, dims);
	complex float* electrodes = load_cfl(argv[2], N, dims1);

	assert(1 == dims[N-1]);
	for(int i = 0; i < N; i++)
		assert(dims[i] == dims1[i]);

	//calculate normal
	long grad_dim = N - 1;
	long grad_dims[N];
	md_copy_dims(N, grad_dims, dims);
	grad_dims[grad_dim] = N - 1;

	complex float *normal = md_alloc(N, grad_dims, CFL_SIZE);
	calc_outward_normal(N, grad_dims, normal, grad_dim, dims, mask);

	//calculate list of points on the border
	const long boundary_dimensions[] =  { md_calc_size(N, dims) };

	struct boundary_point_s *boundary = md_alloc(1, boundary_dimensions, sizeof(struct boundary_point_s));

	long n_points = calc_boundary_points(N, grad_dims, boundary, grad_dim, normal);

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
	long index_strides[N - 1], strs[N], zstrs[N], pos[N];
	pos[grad_dim] = 1;
	calc_index_strides(N - 1, index_strides, dims);
	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, zstrs, dims, CFL_SIZE);
	for (long i = 0; i < n_points; i++) {
		for (int j = 0; j < N - 1; j++)
			pos[j] = boundary[i].index[j] + boundary[i].dir[j];

		long offset = md_calc_offset(N, zstrs, pos);
		complex float val = *(complex float *)((void *)electrodes + offset);
		boundary[i].val = val;
		if ( cabsf(val) > 0 ) {
			offset = md_calc_offset(N - 1, strs, boundary[i].index);
			*(complex float *)((void *)rhs + offset) -= val;
		}

	}

	//phi
	complex float *phi = md_calloc(1, vec_dim, CFL_SIZE);

	//solve
	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = iter;
	conf.l2lambda = 1e-5;

	long size = 2 * md_calc_size(1, vec_dim); // multiply by 2 for float size
	iter_conjgrad(CAST_UP(&conf), diff_op->forward, NULL, size, (float*)phi, (const float*)rhs, NULL);

	complex float* out = create_cfl(argv[3], N, dims);

	md_copy(N, dims, out, phi, CFL_SIZE);


	//homogenous conductivity - calc j, fill in neumann boundary infos
	complex float* j_out = create_cfl(argv[4], N, grad_dims);
	auto d_op = linop_fd_create(N, dims, grad_dim, 7, 2, BC_ZERO, false);
	linop_forward(d_op, N, grad_dims, j_out, N, dims, phi); // minus phi, probably
	linop_free(d_op);

	pos[grad_dim] = 1;
	calc_index_strides(N - 1, index_strides, dims);
	md_calc_strides(N, strs,grad_dims, CFL_SIZE);

	md_copy_dims(N, zstrs, strs);
	zstrs[grad_dim] = 0;
	md_zmul2(N, grad_dims, strs, j_out, zstrs, mask, strs, j_out);

	for (long i = 0; i < n_points; i++) {
		float ext_points = 0;
		for (int j = 0; j < N - 1; j++) {
			ext_points += pow(boundary[i].dir[j], 2);
			pos[j] = boundary[i].index[j];
		}
		for (int j = 0; j < N - 1; j++) {
			pos[grad_dim] = j;
			long offset = md_calc_offset(N, strs, pos);
			*(complex float *)((void *)j_out + offset) = -1 * boundary[i].dir[j] * boundary[i].val * pow(ext_points, .5);
		}

	}

	md_free(normal);
	md_free(boundary);
	md_free(phi);
	md_free(rhs);
	linop_free(diff_op);

	unmap_cfl(N, dims, mask);
	unmap_cfl(N, dims, electrodes);
	unmap_cfl(N, dims, out);
	unmap_cfl(N, dims, j_out);
	return 0;
}


