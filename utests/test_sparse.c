
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "simu/sparse.h"


#include <complex.h>
#include "linops/linop.h"
#include "linops/grad.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"


static const long N_a   = 2;
static const long len_a = 4;
// fortran order! lower diagonal
static const float dense_a[] = { 1, 2, 0, 0,
				 0, 1, 2, 0,
				 0, 0, 1, 2,
				 0, 0, 0, 1 };
static const long N_diags_a = 2;
static const long offsets_a[2][2] = {{0, 0}, {0, -1}};
static const long offsets_a_norm[2][2] = {{0, 0}, {1, 0}};
static const float vals_a[] =  { 1, 2};

static bool generic_sparse_cdiags_create(const long N, const long len, const long N_diags, const long (*offsets)[N], const float values[N_diags], const float *ref)
{
	long dims[N];
	for(int i = 0; i < N; i++)
		dims[i] = len;
	float *out = md_calloc(N, dims, sizeof(float));
	struct sparse_diag_s *mat = sparse_cdiags_create(N, len, N_diags, offsets, values);
	sparse_diag_to_dense(N, dims, out, mat);
	sparse_diag_free(mat);
	float err =  md_rmse(N, dims, out, ref);
	md_free(out);
	return err < 1e-16;
}

static bool test_sparse_cdiags_create(void)
{
	bool ok = true;

	ok &= generic_sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a, vals_a, dense_a);
	ok &= generic_sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a_norm, vals_a, dense_a);

	return ok;
}

static const float dense_laplace1[] = {  2,-1, 0, 0,
				 	-1, 2,-1, 0,
				 	 0,-1, 2,-1,
				 	 0, 0,-1, 2 };
static bool test_sd_laplace_create(void)
{
	long vec_dims[1] = { 4 };
	long mat_dims[2] = { 4, 4 };
	struct sparse_diag_s *mat = sd_laplace_create(1, vec_dims);
	float *out = md_calloc(2, mat_dims, sizeof(float));
	sparse_diag_to_dense(2, mat_dims, out, mat);
	sparse_diag_free(mat);
	float err =  md_rmse(2, mat_dims, out, dense_laplace1);
	md_free(out);
	return err < 1e-16;
}

static bool test_sd_matvec(void)
{
	struct sparse_diag_s *mat = sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a, vals_a);

	long N = N_a - 1;
	long dims[] = { len_a };

	float *vec = md_alloc (N, dims, sizeof(float));
	float *ref = md_calloc(N, dims, sizeof(float));

	for (int i = 0; i < *dims; i++)
		vec[i] = gaussian_rand();

	for (int i = 0; i < *dims; i++)
		for (int j = 0; j < *dims; j++)
			ref[i] += dense_a[i + *dims * j] * vec[j];

	float *out = md_calloc(1, dims, sizeof(float));

	sd_matvec(N, dims, out, vec, mat);

	float err =  md_rmse(N, dims, out, ref);

	md_free(out);
	md_free(ref);
	md_free(vec);
	sparse_diag_free(mat);

	return err < 1e-16;
}


static bool test_sd_laplace_3d(void)
{
	const long N = 3;
	const long dims[3] = { 11, 32, 87 };

	float *rref = md_alloc(N, dims, sizeof(float));
	complex float *vec = md_alloc(N, dims, sizeof(complex float));
	float *rvec = md_alloc(N, dims, sizeof(float));

	md_gaussian_rand(N, dims, vec);
	md_real(N, dims, rvec, vec);

	long strs[N], pos[N];
	md_set_dims(N, pos, 0);
	md_calc_strides(N, strs, dims, sizeof(float));

	do {
		long offset = md_calc_offset(N, strs, pos);
		*( (float *)( ((void*)rref) + offset ) ) =  6 * *( (float *)( ((void*)rvec) + offset) );
		for(int i = 0; i < N; i++) {
			if(pos[i] < dims[i] - 1)
				*( (float *)( ((void*)rref) + offset ) ) -=  *( (float *)( ((void*)rvec) + offset + strs[i]) );
			if(pos[i] > 0)
				*( (float *)( ((void*)rref) + offset ) ) -=  *( (float *)( ((void*)rvec) + offset - strs[i]) );
		}
	} while(md_next(N, dims, 7, pos));

	float *out = md_alloc(N, dims, sizeof(float));

	long flatdims[1] = { 1 };
	for (int i = 0; i < N; i++)
		flatdims[0] *= dims[i];

	struct sparse_diag_s *mat = sd_laplace_create(N, dims);

	sd_matvec(1, flatdims, out, rvec, mat);

	float err =  md_rmse(N, dims, out, rref);
#if 0
	long mat_dims[2] = { mat->len, mat->len };
	float *dense = md_calloc(2, mat_dims, sizeof(float));
	sparse_diag_to_dense(2, mat_dims, dense, mat);
	complex float *zdense = md_calloc(2, mat_dims, sizeof(complex float));
	md_zcmpl_real(2, mat_dims, zdense, dense);
	dump_cfl("dense", 2, mat_dims, zdense);
	md_free(zdense);
	md_free(dense);
#endif
	sparse_diag_free(mat);
	md_free(rref);
	md_free(vec);
	md_free(rvec);
	md_free(out);

	return err < 1e-16;

}



UT_REGISTER_TEST(test_sparse_cdiags_create);
UT_REGISTER_TEST(test_sd_laplace_create);
UT_REGISTER_TEST(test_sd_matvec);
UT_REGISTER_TEST(test_sd_laplace_3d);
