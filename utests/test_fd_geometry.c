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

#define grad_dim N - 1

#define N_a 2
static const long dims_a[N_a] = { 10, 1 };
static const complex float mask_a[10] =       { 0,  1,  1,  1,  0,  1,  1,  0,  0,  0 };
static const complex float normal_a[10] =     { 0, -1,  0,  1,  0, -1,  1,  0,  0,  0 };

// fortran order
#define N_b 3
#define w2 0.7071067811865476
static const long dims_b[N_b] = { 7, 5, 1 };
static const complex float mask_b[] = 	 	{ 0,  1,  1,  1,  1,  1,  0,
					   	  0,  1,  1,  1,  1,  1,  0,
					   	  0,  1,  1,  0,  1,  1,  0,
					   	  0,  1,  1,  1,  1,  1,  0,
					   	  0,  1,  1,  1,  1,  1,  0 };


static const complex float normal_b[] = {
						  0,-w2,  0,  0,  0,w2,  0,
					          0, -1,  0,  0,  0,  1,  0,
					          0, -1,  1,  0, -1,  1,  0,
					          0, -1,  0,  0,  0,  1,  0,
					          0,-w2,  0,  0,  0, w2,  0,

						  0,-w2, -1, -1, -1,-w2,  0,
					          0,  0,  0,  1,  0,  0,  0,
					          0,  0,  0,  0,  0,  0,  0,
					          0,  0,  0, -1,  0,  0,  0,
					          0, w2,  1,  1,  1, w2,  0
					};


static bool generic_outward_normal(const long N, const long dims[N], const complex float *mask, const complex float *reference)
{
	long pos[N];
	md_set_dims(N, pos, 0);
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	long grad_dims[N];
	md_copy_dims(N, grad_dims, dims);
	grad_dims[grad_dim] = N - 1;
	bool ok = true;

	complex float *normal = md_alloc(N, grad_dims, CFL_SIZE);
	calc_outward_normal(N, grad_dims, normal, grad_dim, dims, mask);

	do {
		long offset = md_calc_offset(N_a, strs, pos);
		ok &= (*(complex float *)((void *)normal + offset) == *(complex float *)((void *)reference + offset));
	} while (ok && md_next(N, grad_dims, MD_BIT(N) - 1, pos));

	md_free(normal);

	return ok;
}

static bool test_calc_outward_normal(void)
{
	bool ok = true;

	ok &= generic_outward_normal(N_a, dims_a, mask_a, normal_a);
	debug_printf(DP_INFO, "1D : %s\n", ok ? "OK" : "FAIL");

	ok &= generic_outward_normal(N_b, dims_b, mask_b, normal_b);
	debug_printf(DP_INFO, "2D : %s\n", ok ? "OK" : "FAIL");

	return ok;
}
UT_REGISTER_TEST(test_calc_outward_normal);
