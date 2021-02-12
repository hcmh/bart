/*
 * Authors:
 * 2021  Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "linops/linop.h"
#include "linops/lintest.h"

#include "simu/leray.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"

#define TOL 1e-1
#define ITER 100

static struct linop_s * leray_create(const enum BOUNDARY_CONDITION bc, const long order)
{
	const long N = 4;
	const long d = 0;
	const long flags = 14;
	const long dims[] = { 3, 10, 10, 10};
	return linop_leray_create(N, dims, d, flags, order, bc, ITER);
}


static bool test_leray_normal(const enum BOUNDARY_CONDITION bc, const long order)
{
	struct linop_s* op = leray_create(bc, order);

	float nrmse = linop_test_normal(op);
	debug_printf(DP_INFO, "BC: %d, Order: %d: normal nrmse: %f\n", bc, order, nrmse);
	bool ret = (nrmse < TOL);

	linop_free(op);

	return ret;
}


static bool test_leray_adjoint(const enum BOUNDARY_CONDITION bc, const long order)
{
	struct linop_s* op = leray_create(bc, order);

	float nrmse = linop_test_adjoint(op);
	debug_printf(DP_INFO, "BC: %d, Order: %d: adjoint nrmse: %f\n", bc, order, nrmse);
	bool ret = (nrmse < TOL);

	linop_free(op);

	return ret;
}


static bool test_leray(void)
{
	const enum BOUNDARY_CONDITION bcs[3] = {BC_PERIODIC, BC_ZERO, BC_SAME};
	bool ok = true;
	for (int i=0; i<3; i++) {
		for (int order=1; order<3; order++) {
			ok &= test_leray_adjoint(bcs[i], order);
			ok &= test_leray_normal(bcs[i], order);
		}
	}
	return ok;
}
UT_REGISTER_TEST(test_leray);
