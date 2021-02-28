/*
 * Authors:
 * 2021  Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/rand.h"

#include "linops/linop.h"
#include "linops/lintest.h"

#include "simu/leray.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "utest.h"

#define TOL 1e-1
#define ITER 300
#define lambda 1e-1

static struct linop_s *leray_create()
{
	const long N = 4;
	const long d = 0;
	const long dims[] = {3, 25, 25, 25};
	const long scalar_N = N - 1;
	const long *scalar_dims = dims + 1;
	complex float *mask = NULL;

	mask = md_alloc(scalar_N, scalar_dims, CFL_SIZE);
	md_zfill(scalar_N, scalar_dims, mask, 0);

	long str[scalar_N], pos[scalar_N];
	md_calc_strides(scalar_N, str, scalar_dims, CFL_SIZE);
	long margin = 2;
	long inner_dims[] = {dims[1] - 2 * margin, dims[2] - 2 * margin, dims[3] - 2 * margin};
	md_set_dims(scalar_N, pos, margin);
	long offset = md_calc_offset(scalar_N, str, pos);
	md_zfill2(scalar_N, inner_dims, str, (void *)mask + offset, 1.);

	auto op = linop_leray_create(N, dims, d, ITER, lambda, mask, NULL);

	md_free(mask);

	return op;
}


static bool test_leray_normal()
{
	struct linop_s *op = leray_create();

	float nrmse = linop_test_normal(op);
	debug_printf(DP_DEBUG1, "normal nrmse: %f\n", nrmse);
	bool ret = (nrmse < TOL);

	linop_free(op);

	return ret;
}


static bool test_leray_adjoint()
{
	struct linop_s *op = leray_create();

	float nrmse = linop_test_adjoint_real(op);
	debug_printf(DP_DEBUG1, "adjoint nrmse: %f\n", nrmse);
	bool ret = (nrmse < TOL);

	linop_free(op);

	return ret;
}


static bool test_leray(void)
{
	bool ok = true;
	ok &= test_leray_adjoint();
	ok &= test_leray_normal();
	return ok;
}
UT_REGISTER_TEST(test_leray);
