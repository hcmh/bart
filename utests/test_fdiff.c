/*
 * Authors:
 * 2021 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */


#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "utest.h"

#define N 4
#define TOL 1e-7


typedef void (*zfdiff_fun_t)(unsigned int D, const long dims[D], unsigned int d, const enum BOUNDARY_CONDITION bc, complex float *out, const complex float *in);

static bool test_md_zfdiff_generic(zfdiff_fun_t f, enum BOUNDARY_CONDITION bc, const complex float expected[N])
{
	static int test_no = 0;
	const complex float in[N] = {1., 2., 3., 4.};
	complex float out[N] = {0};
	const long dims[] = {N};

	f(1, dims, 0, bc, out, in);

	bool ok = true;
	debug_printf(DP_DEBUG1, "zfdiff_generic test %d\n", test_no++);
	debug_printf(DP_DEBUG1, "Expected: ");
	for (int i = 0; i < N; i++)
		debug_printf(DP_DEBUG1, "%.1f, ", crealf(expected[i]));
	debug_printf(DP_DEBUG1, "\nOutput:   ");
	for (int i = 0; i < N; i++) {
		debug_printf(DP_DEBUG1, "%.1f, ", crealf(out[i]));
		ok &= (cabsf(expected[i] - out[i]) < TOL);
	}
	debug_printf(DP_DEBUG1, "\n");

	return ok;
}

// backward differences
static bool test_md_zfdiff_periodic(void)
{
	return test_md_zfdiff_generic(md_zfdiff, BC_PERIODIC,
				      (const complex float[N]){-3., 1., 1., 1.});
}

static bool test_md_zfdiff_zero(void)
{
	return test_md_zfdiff_generic(md_zfdiff, BC_ZERO,
				      (complex float[N]){1., 1., 1., 1.});
}

static bool test_md_zfdiff_same(void)
{
	return test_md_zfdiff_generic(md_zfdiff, BC_SAME,
				      (complex float[N]){0., 1., 1., 1.});
}

// forward differences  * (-1)
static bool test_md_zfdiff_backwards_periodic(void)
{
	return test_md_zfdiff_generic(md_zfdiff_backwards, BC_PERIODIC,
				      (complex float[N]){-1., -1., -1., 3.});
}

static bool test_md_zfdiff_backwards_zero(void)
{
	return test_md_zfdiff_generic(md_zfdiff_backwards, BC_ZERO,
				      (complex float[N]){-1., -1., -1., 4.});
}

static bool test_md_zfdiff_backwards_same(void)
{
	return test_md_zfdiff_generic(md_zfdiff_backwards, BC_SAME,
				      (complex float[N]){-1., -1., -1., 0.});
}


// central differences
static bool test_md_zfdiff_central(enum BOUNDARY_CONDITION bc, const complex float expected[N])
{
	static int test_no = 0;
	const complex float in[N] = {1., 2., 3., 4.};
	complex float out[N] = {0};
	const long dims[] = {N};

	md_zfdiff_central(1, dims, 0, bc, false, out, in);

	bool ok = true;
	debug_printf(DP_DEBUG1, "zfdiff_central test %d - forward\n", test_no++);
	debug_printf(DP_DEBUG1, "Expected: ");
	for (int i = 0; i < N; i++)
		debug_printf(DP_DEBUG1, "%.1f, ", crealf(expected[i]));
	debug_printf(DP_DEBUG1, "\nOutput:   ");
	for (int i = 0; i < N; i++) {
		ok &= (cabsf(expected[i] - out[i]) < TOL);
		debug_printf(DP_DEBUG1, "%.1f, ", crealf(out[i]));
	}

	debug_printf(DP_DEBUG1, "\nzfdiff_central test %d - backward\n", test_no++);
	debug_printf(DP_DEBUG1, "Expected: ");
	for (int i = 0; i < N; i++)
		debug_printf(DP_DEBUG1, "%.1f, ", -1. * crealf(expected[i]));
	debug_printf(DP_DEBUG1, "\nOutput:   ");
	md_zfdiff_central(1, dims, 0, bc, true, out, in);
	for (int i = 0; i < N; i++) {
		ok &= (cabsf(expected[i] + out[i]) < TOL);
		debug_printf(DP_DEBUG1, "%.1f, ", crealf(out[i]));
	}
	debug_printf(DP_DEBUG1, "\n");

	return ok;
}



static bool test_md_zfdiff_central_periodic(void)
{
	return test_md_zfdiff_central(BC_PERIODIC,
				      (complex float[N]){-2., 2., 2., -2.});
}

static bool test_md_zfdiff_central_zero(void)
{
	return test_md_zfdiff_central(BC_ZERO,
				      (complex float[N]){2., 2., 2., -3.});
}

static bool test_md_zfdiff_central_same(void)
{
	return test_md_zfdiff_central(BC_SAME,
				      (complex float[N]){1., 2., 2., 1.});
}

UT_REGISTER_TEST(test_md_zfdiff_periodic);
UT_REGISTER_TEST(test_md_zfdiff_zero);
UT_REGISTER_TEST(test_md_zfdiff_same);
UT_REGISTER_TEST(test_md_zfdiff_backwards_periodic);
UT_REGISTER_TEST(test_md_zfdiff_backwards_zero);
UT_REGISTER_TEST(test_md_zfdiff_backwards_same);
UT_REGISTER_TEST(test_md_zfdiff_central_periodic);
UT_REGISTER_TEST(test_md_zfdiff_central_zero);
UT_REGISTER_TEST(test_md_zfdiff_central_same);
