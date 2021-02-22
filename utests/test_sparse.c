
#include "num/multind.h"
#include "num/flpmath.h"
#include "simu/sparse.h"

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
static const float vals_a[] =  { 1, 2};

static bool generic_sparse_cdiags_create(const long N, const long len, const long N_diags, const long (*offsets)[N], const float values[N_diags], const float *ref)
{
	long dims[N];
	for(int i = 0; i < N; i++)
		dims[i] = len;
	float *out = md_calloc(N, dims, sizeof(float));
	struct sparse_diag_s *mat = sparse_cdiags_create(N, len, N_diags, offsets, values);
	sparse_diag_to_dense(N, dims, out, mat);
	return md_rmse(N, dims, out, ref) < 1e-16;
}

static bool test_sparse_cdiags_create(void)
{
	bool ok = true;

	ok &= generic_sparse_cdiags_create(N_a, len_a, N_diags_a, offsets_a, vals_a, dense_a);

	return ok;
}

UT_REGISTER_TEST(test_sparse_cdiags_create);
