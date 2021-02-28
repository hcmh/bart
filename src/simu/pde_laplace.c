/*
 * Laplace Operator with Boundary Conditions
 * for solving PDEs
 */

#include "linops/linop.h"
#include "misc/misc.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "simu/fd_geometry.h"
#include "simu/pde_laplace.h"
#include "simu/sparse.h"

#include <complex.h>
#include <math.h>


struct sparse_diag_s *sd_laplace_create(long N, const long dims[N])
{
	long len = 1, N_mat = 2, N_diags = N * 2 + 1, offsets[N_diags][N_mat];
	complex float values[N_diags];

	//main diagonal
	offsets[0][0] = 0;
	offsets[0][1] = 0;
	values[0] = 2 * N;

	//off diagonals
	int n = 1;
	for (int i = 0; i < N; i++) {
		offsets[n][0] = 0;
		offsets[n][1] = len;
		values[n++] = -1;

		offsets[n][0] = 0;
		offsets[n][1] = -len;
		values[n++] = -1;

		len *= dims[i];
	}

	struct sparse_diag_s *mat = sparse_cdiags_create(N_mat, len, N_diags, offsets, values);

	//correct boundaries
	n = 1;
	long str = 1;
	long len_ones, len_zeros;
	for (int i = 0; i < N; i++) {
		len_ones = (dims[i] - 1) * str;
		len_zeros = str;
		long j = len_ones;
		while (j + len_zeros - 1 < mat->dims[n]) {
			for (long k = 0; k < len_zeros; k++) {
				mat->diags[n][j + k] = 0;
				mat->diags[n + 1][j + k] = 0;
			}
			j += len_zeros + len_ones;
		}
		n += 2;

		str *= dims[i];
	}

	return mat;
}



void laplace_dirichlet(struct sparse_diag_s *mat, const long N, const long dims[N], const long n_points, struct boundary_point_s points[n_points])
{
	assert(2 == mat->N);

	long str[N];
	calc_index_strides(N, str, dims);
	assert(mat->len == calc_index_size(N, str, dims));

	assert(mat->offsets_normal);

	//for each point on a boundary
	for (long i = 0; i < n_points; i++) {
		long mat_index = calc_index(N, str, points[i].index);

		// set main diagonal to 1
		mat->diags[0][mat_index] = 1;

		// set off-diagonals to 0
		for (long j = 1; j < mat->N_diags; j++) {
			const long *offsets = mat->offsets[j];
			// lower diagonals
			if ((offsets[0] > 0) && (mat_index >= offsets[0]))
				mat->diags[j][mat_index - offsets[0]] = 0;
			// upper diagonals
			if ((offsets[1] > 0) && (mat_index < mat->dims[j]))
				mat->diags[j][mat_index] = 0;
		}
	}
}



void laplace_neumann(struct sparse_diag_s *mat, const long N, const long dims[N], const long n_points, const struct boundary_point_s boundary[n_points])
{
	assert(2 == mat->N);

	long str[N];
	calc_index_strides(N, str, dims);
	assert(mat->len == calc_index_size(N, str, dims));

	assert(mat->offsets_normal);

	for (long i = 0; i < n_points; i++) {
		const struct boundary_point_s *point = boundary + i;
		long mat_index = calc_index(N, str, point->index);
		long diag_index = 0;
		long ext_neighbours = 0;
		for (long j = 0; j < N; j++)
			ext_neighbours += labs(point->dir[j]);

		mat->diags[diag_index][mat_index] -= ext_neighbours;

		// set this row in the off-diagonals
		for (long j = 0; j < N; j++) {
			// upper diagonal
			const long *offsets = mat->offsets[++diag_index];
			assert(offsets[0] == 0);
			if (point->dir[j] == 1)
				mat->diags[diag_index][mat_index] = 0;

			// lower diagonal
			offsets = mat->offsets[++diag_index];
			assert(offsets[0] > 0);
			if (point->dir[j] == -1)
				mat->diags[diag_index][mat_index - offsets[0]] = 0;
		}
	}
}


void laplace_neumann_update_rhs(const long N, const long dims[N], complex float *rhs, const long n_points, const struct boundary_point_s boundary[n_points])
{
	long index_strides[N], strs[N], pos[N];
	md_set_dims(N, pos, 0);
	calc_index_strides(N, index_strides, dims);
	md_calc_strides(N, strs, dims, CFL_SIZE);
	for (long i = 0; i < n_points; i++) {
		if (cabsf(boundary[i].val) > 0) {
			long offset = md_calc_offset(N, strs, boundary[i].index);
			*(complex float *)((void *)rhs + offset) -= boundary[i].val;
		}
	}
}


struct linop_s *linop_laplace_neumann_create(const long N, const long dims[N], const complex float *mask, const long n_points, const struct boundary_point_s boundary[n_points])
{
	//create laplace
	struct sparse_diag_s *mat = sd_laplace_create(N, dims);

	//apply boundary
	laplace_neumann(mat, N, dims, n_points, boundary);
	sd_mask(N, dims, mat, mask);

	//flat
	long vec_dim[] = {mat->len};
	return linop_sd_matvec_create(1, vec_dim, mat);
}
