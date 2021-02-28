#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "simu/sparse.h"
#include "simu/fd_geometry.h"
#include "simu/pde_laplace.h"

#include <complex.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"


#define N_d 3
static const long dims_d[N_d] = { 5, 5, 1 };
static const complex float mask_d[] = 	 	{ 0, 0, 0, 0, 0,
						  0, 1, 1, 1, 0,
						  0, 1, 1, 1, 0,
						  0, 1, 1, 1, 0,
						  0, 0, 0, 0, 0 };

static const complex float vals_d[] = 	 	{ 0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  1, 0, 0, 0,-1,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0 };

static const complex float rhs_d[] = 	 	{ 0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0,-1, 0, 1, 0,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0 };

static const complex float laplace_neumann_d[] =
	{ 0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,

	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 2,-1, 0, 0,    0,-1, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0,-1, 3,-1, 0,    0, 0,-1, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0,-1, 2, 0,    0, 0, 0,-1, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,

	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0,-1, 0, 0, 0,    0, 3,-1, 0, 0,    0,-1, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0,-1, 0, 0,    0,-1, 4,-1, 0,    0, 0,-1, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0,-1, 0,    0, 0,-1, 3, 0,    0, 0, 0,-1, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,

	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0,-1, 0, 0, 0,    0, 2,-1, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0,-1, 0, 0,    0,-1, 3,-1, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0,-1, 0,    0, 0,-1, 2, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,

	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
	  0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0  };

static const complex float dense_laplace1[] = {  2,-1, 0, 0,
					 	-1, 2,-1, 0,
					 	 0,-1, 2,-1,
					 	 0, 0,-1, 2 };

static bool test_sd_laplace_create(void)
{
	long vec_dims[1] = {4};
	long mat_dims[2] = {4, 4};
	struct sparse_diag_s *mat = sd_laplace_create(1, vec_dims);
	complex float *out = md_calloc(2, mat_dims, CFL_SIZE);
	sparse_diag_to_dense(2, mat_dims, out, mat);
	sparse_diag_free(mat);
	float err = md_zrmse(2, mat_dims, out, dense_laplace1);
	md_free(out);
	debug_printf(DP_DEBUG1, "sd_laplace_create rmse: %f\n", err);
	return err < 1e-16;
}


static bool test_sd_laplace_3d(void)
{
	const long N = 3;
	const long dims[3] = {11, 32, 87};

	complex float *ref = md_alloc(N, dims, CFL_SIZE);
	complex float *vec = md_alloc(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, vec);

	long strs[N], pos[N];
	md_set_dims(N, pos, 0);
	md_calc_strides(N, strs, dims, CFL_SIZE);

	do {
		long offset = md_calc_offset(N, strs, pos);
		*((complex float *)(((void *)ref) + offset)) = 6 * *((complex float *)(((void *)vec) + offset));
		for (int i = 0; i < N; i++) {
			if (pos[i] < dims[i] - 1)
				*((complex float *)(((void *)ref) + offset)) -= *((complex float *)(((void *)vec) + offset + strs[i]));
			if (pos[i] > 0)
				*((complex float *)(((void *)ref) + offset)) -= *((complex float *)(((void *)vec) + offset - strs[i]));
		}
	} while (md_next(N, dims, 7, pos));

	complex float *out = md_alloc(N, dims, CFL_SIZE);

	long flatdims[1] = {1};
	for (int i = 0; i < N; i++)
		flatdims[0] *= dims[i];

	struct sparse_diag_s *mat = sd_laplace_create(N, dims);

	sd_matvec(1, flatdims, out, vec, mat);

	float err = md_zrmse(N, dims, out, ref);
#if 0
	long mat_dims[2] = { mat->len, mat->len };
	complex float *dense = md_calloc(2, mat_dims, sizeof(complex float));
	sparse_diag_to_dense(2, mat_dims, dense, mat);
	dump_cfl("dense", 2, mat_dims, dense);
	md_free(dense);
#endif
	sparse_diag_free(mat);
	md_free(ref);
	md_free(vec);
	md_free(out);

	debug_printf(DP_DEBUG1, "sd_laplace_3d rmse: %f\n", err);
	return err < 1e-16;
}



static bool test_laplace_dirichlet(void)
{
	const long N = 3;
	const long dims[3] = {15, 8, 17};
	long strs[3];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	const long island_pos[3] = {5, 4, 7};
	const long island_dims[3] = {4, 1, 3};

	//create mask
	complex float *mask = md_alloc(N, dims, CFL_SIZE);

	md_zfill(N, dims, mask, 1);
	md_zfill2(N, island_dims, strs, (void *)mask + md_calc_offset(N, strs, island_pos), 0);

	//calculate boundary
	long grad_dim = N;
	long grad_dims[N + 1];
	long grad_dims1[N + 1];
	md_copy_dims(N, grad_dims, dims);
	md_copy_dims(N, grad_dims1, dims);
	grad_dims[grad_dim] = N;
	grad_dims1[grad_dim] = 1;

	complex float *normal = md_alloc(N + 1, grad_dims, CFL_SIZE);
	calc_outward_normal(N + 1, grad_dims, normal, grad_dim, grad_dims1, mask);
	const long boundary_dimensions[] = {md_calc_size(N, dims)};

	struct boundary_point_s *boundary = md_alloc(1, boundary_dimensions, sizeof(struct boundary_point_s));

	long n_points = calc_boundary_points(N + 1, grad_dims, boundary, grad_dim, normal, NULL);

	md_free(normal);

	//create laplace
	struct sparse_diag_s *mat = sd_laplace_create(N, dims);

	//apply boundary
	laplace_dirichlet(mat, N, dims, n_points, boundary);

	//verify dense matrix
	long mat_dims[2] = {mat->len, mat->len}, dense_str[2];
	md_calc_strides(2, dense_str, mat_dims, CFL_SIZE);
	complex float *dense = md_calloc(2, mat_dims, CFL_SIZE);
	sparse_diag_to_dense(2, mat_dims, dense, mat);

	long index_strides[N];
	calc_index_strides(N, index_strides, dims);

	long errs = 0;
	for (int i = 0; i < n_points; i++) {
		bool ok = true;
		long mat_ind = calc_index(N, index_strides, boundary[i].index);
		long offset = mat_ind * dense_str[0];
		float row_sum = 0;
		for (int j = 0; j < mat_dims[0]; j++) {
			row_sum += *(complex float *)((void *)dense + offset);
			offset += dense_str[1];
		}
		ok &= fabs(1 - row_sum) < 1e-16;
		ok &= cabsf(1 - *(complex float *)((void *)dense + dense_str[0] * mat_ind + dense_str[1] * mat_ind)) < 1e-16;
		errs += ok ? 0 : 1;
	}

	//dump_cfl("masked_dense", 2, mat_dims, zdense);

	md_free(mask);
	md_free(dense);
	md_free(boundary);
	sparse_diag_free(mat);

	debug_printf(DP_DEBUG1, "laplace_dirichlet: %d errors\n", errs);
	return errs == 0;
}



static bool generic_laplace_neumann(const long N, const long dims[N], const complex float *mask, const complex float *boundary_values, const complex float *ref, const complex float *rhs_ref)
{
	//calculate normal
	long grad_dim = N - 1;
	long grad_dims[N];
	md_copy_dims(N, grad_dims, dims);
	grad_dims[grad_dim] = N - 1;

	complex float *normal = md_alloc(N, grad_dims, CFL_SIZE);
	calc_outward_normal(N, grad_dims, normal, grad_dim, dims, mask);

	//calculate list of points on the border
	const long boundary_dimensions[] = {md_calc_size(N, dims)};

	struct boundary_point_s *boundary = md_alloc(1, boundary_dimensions, sizeof(struct boundary_point_s));

	long n_points = calc_boundary_points(N, grad_dims, boundary, grad_dim, normal, boundary_values);

	//create laplace
	struct sparse_diag_s *mat = sd_laplace_create(N - 1, dims);

	//apply boundary
	laplace_neumann(mat, N - 1, dims, n_points, boundary);

	//mask outside points
	sd_mask(N - 1, dims, mat, mask);

	//verify dense matrix
	long mat_dims[2] = {mat->len, mat->len}, dense_str[2];
	md_calc_strides(2, dense_str, mat_dims, CFL_SIZE);
	complex float *dense = md_calloc(2, mat_dims, CFL_SIZE);
	sparse_diag_to_dense(2, mat_dims, dense, mat);

	float err = md_zrmse(2, mat_dims, ref, dense);
	bool ok = (err < 1e-6);

	//verify rhs
	complex float *rhs = md_calloc(N - 1, dims, CFL_SIZE);
	laplace_neumann_update_rhs(N - 1, dims, rhs, n_points, boundary);
	err = md_zrmse(N - 1, dims, rhs, rhs_ref);
	ok &= (err < 1e-6);

	md_free(normal);
	md_free(boundary);
	sparse_diag_free(mat);
	md_free(dense);
	md_free(rhs);

	debug_printf(DP_DEBUG1, "laplace_neumann rmse: %f\n", err);
	return err < 1e-16;
}


static bool test_laplace_neumann(void)
{
	return generic_laplace_neumann(N_d, dims_d, mask_d, vals_d, laplace_neumann_d, rhs_d);
}

UT_REGISTER_TEST(test_sd_laplace_create);
UT_REGISTER_TEST(test_sd_laplace_3d);
UT_REGISTER_TEST(test_laplace_dirichlet);
UT_REGISTER_TEST(test_laplace_neumann);
