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

#include "linops/grad.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"

#define TOL 1e-6

static struct linop_s * fd_create(const enum BOUNDARY_CONDITION bc, const long order, bool reverse)
{
	const long N = 4;
	const long d = 0;
	const long flags = 14;
	const long dims[] = { 1, 10, 10, 10};
	return linop_fd_create(N, dims, d, flags, order, bc, reverse);
}


static bool test_fd_normal(const enum BOUNDARY_CONDITION bc, const long order, bool reverse)
{
	struct linop_s* op = fd_create(bc, order, reverse);

	float nrmse = linop_test_normal(op);
	debug_printf(DP_DEBUG1, "BC: %d, Order: %d, Reverse: %d: normal nrmse: %f\n", bc, order, reverse, nrmse);
	bool ret = (nrmse < TOL);

	linop_free(op);

	return ret;
}


static bool test_fd_adjoint(const enum BOUNDARY_CONDITION bc, const long order, bool reverse)
{
	struct linop_s* op = fd_create(bc, order, reverse);

	float nrmse = linop_test_adjoint(op);
	debug_printf(DP_DEBUG1, "BC: %d, Order: %d, Reverse: %d: adjoint nrmse: %f\n", bc, order, reverse, nrmse);
	bool ret = (nrmse < TOL);

	linop_free(op);

	return ret;
}


static bool test_fd(void)
{
	const enum BOUNDARY_CONDITION bcs[3] = {BC_PERIODIC, BC_ZERO, BC_SAME};
	bool ok = true;
	for (int i=0; i<3; i++) {
		for (int order=1; order<3; order++) {
			for (int reverse=0; reverse<2; reverse++) {
				ok &= test_fd_adjoint(bcs[i], order, reverse);
				ok &= test_fd_normal(bcs[i], order, reverse);
			}
		}
	}
	return ok;
}
UT_REGISTER_TEST(test_fd);
