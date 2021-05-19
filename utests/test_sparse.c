
#include "num/flpmath.h"
#include "num/multind.h"
#include "num/rand.h"
#include "simu/fd_geometry.h"
#include "simu/sparse.h"

#include "linops/linop.h"
#include "linops/lintest.h"

#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "utest.h"


static const long N_a = 2;
static const long len_a = 4;
// fortran order! lower diagonal
static const complex float dense_a[] = {1, 2, 0, 0,
					0, 1, 2, 0,
					0, 0, 1, 2,
					0, 0, 0, 1};
static const long N_diags_a = 2;
static const long offsets_a[2][2] = {{0, 0}, {0, -1}};
static const long offsets_a_norm[2][2] = {{0, 0}, {1, 0}};
static const complex float vals_a[] = {1, 2};



static bool generic_sparse_cdiags_create(const long N, const long len, const long N_diags, const long (*offsets)[N], const complex float values[N_diags], const complex float *ref)
{
	long dims[N];
	for (int i = 0; i < N; i++)
		dims[i] = len;
	complex float *out = md_calloc(N, dims, CFL_SIZE);
	struct sparse_diag_s *mat = sparse_cdiags_create(N, len, N_diags, offsets, values);
	sparse_diag_to_dense(N, dims, out, mat);
	sparse_diag_free(mat);
	float err = md_zrmse(N, dims, out, ref);
	md_free(out);
	debug_printf(DP_DEBUG1, "sparse_cdiags_create rmse: %f\n", err);
	return err < 1e-16;
}

static bool test_sparse_cdiags_create(void)
{
	bool ok = true;

	ok &= generic_sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a, vals_a, dense_a);
	ok &= generic_sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a_norm, vals_a, dense_a);

	return ok;
}


static bool test_sd_matvec(void)
{
	struct sparse_diag_s *mat = sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a, vals_a);

	long N = N_a - 1;
	long dims[] = {len_a};

	complex float *vec = md_alloc(N, dims, CFL_SIZE);
	complex float *ref = md_calloc(N, dims, CFL_SIZE);

	for (int i = 0; i < *dims; i++)
		vec[i] = gaussian_rand();

	for (int i = 0; i < *dims; i++)
		for (int j = 0; j < *dims; j++)
			ref[i] += dense_a[i + *dims * j] * vec[j];

	complex float *out = md_calloc(1, dims, CFL_SIZE);

	sd_matvec(N, dims, out, vec, mat);

	float err = md_zrmse(N, dims, out, ref);

	md_free(out);
	md_free(ref);
	md_free(vec);
	sparse_diag_free(mat);

	debug_printf(DP_DEBUG1, "sd_matvec rmse: %f\n", err);
	return err < 1e-16;
}


static bool test_linop_sd_matvec(void)
{
	struct sparse_diag_s *mat = sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a, vals_a);

	long N = N_a - 1;
	long dims[] = {len_a};

	complex float *vec = md_alloc(N, dims, CFL_SIZE);
	complex float *ref = md_calloc(N, dims, CFL_SIZE);

	for (int i = 0; i < *dims; i++)
		vec[i] = gaussian_rand();

	for (int i = 0; i < *dims; i++)
		for (int j = 0; j < *dims; j++)
			ref[i] += dense_a[i + *dims * j] * vec[j];

	complex float *out = md_calloc(1, dims, CFL_SIZE);

	auto op = linop_sd_matvec_create(N, dims, mat);

	bool ok = true;

	linop_forward(op, N, dims, out, N, dims, vec);

	float err = md_zrmse(N, dims, out, ref);
	debug_printf(DP_DEBUG1, "forward rmse: %f\n", err);
	ok &= (err < 1e-8);

	err = linop_test_normal(op);
	debug_printf(DP_DEBUG1, "normal rmse: %f\n", err);
	ok &= (err < 1e-5);

	err = linop_test_adjoint(op);
	debug_printf(DP_DEBUG1, "adjoint rmse: %f\n", err);
	ok &= (err < 1e-5);

	linop_free(op);
	// sparse_diag_free(mat);
	md_free(out);
	md_free(ref);
	md_free(vec);

	return ok;
}


UT_REGISTER_TEST(test_sparse_cdiags_create);
UT_REGISTER_TEST(test_sd_matvec);
UT_REGISTER_TEST(test_linop_sd_matvec);
