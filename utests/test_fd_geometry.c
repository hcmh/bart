/*
 * Authors:
 * 2021  Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "simu/fd_geometry.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"

#define TOL 1e-4

#define N_a 2
static const long dims_a[N_a] = { 10, 1 };
static const complex float mask_a[10] =       { 0,  1,  1,  1,  0,  1,  1,  0,  0,  0 };
static const complex float normal_a[1][10] = {{ 0, -1,  0,  1,  0, -1,  1,  0,  0,  0 }};
// fortran order - beware

static bool test_calc_outward_normal(void)
{
	long pos[N_a] = { 0 };
	long strs[N_a];
	md_calc_strides(N_a, strs, dims_a, CFL_SIZE);
	bool ok = true;

	complex float *normal = calc_outward_normal(1, dims_a, mask_a);

	do {
		long offset = md_calc_offset(N_a, strs, pos);
		ok &= (*(complex float *)((void *)normal + offset) == *(complex float *)((void *)normal_a + offset));
	} while (ok && md_next(N_a, dims_a, MD_BIT(N_a) - 1, pos));

	md_free(normal);

	return ok;
}
UT_REGISTER_TEST(test_calc_outward_normal);
