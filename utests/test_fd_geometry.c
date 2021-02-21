/*
 * Authors:
 * 2021  Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>
#include <math.h>

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
static const long n_points_a = 4;
static const struct boundary_point_s boundary_a[] = {
	{ .index = { 1 }, .dir = { -1 } },
	{ .index = { 3 }, .dir = {  1 } },
	{ .index = { 5 }, .dir = { -1 } },
	{ .index = { 6 }, .dir = {  1 } }
	};



// fortran order!
#define N_b 3
#define w2 0.7071067811865476
static const long dims_b[N_b] = { 7, 5, 1 };
static const complex float mask_b[] = 	 	{ 0,  1,  1,  1,  1,  1,  0,
					   	  0,  1,  1,  1,  1,  1,  0,
					   	  0,  1,  1,  0,  1,  1,  0,
					   	  0,  1,  1,  1,  1,  1,  0,
					   	  0,  1,  1,  1,  1,  1,  0 };


static const complex float normal_b[] = {
						  0,-w2,  0,  0,  0, w2,  0,
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

static const long n_points_b = 20;
static const struct boundary_point_s boundary_b[] = {
	{ .index = { 1, 0 }, .dir = { -w2,-w2 } },
	{ .index = { 2, 0 }, .dir = {   0, -1 } },
	{ .index = { 3, 0 }, .dir = {   0, -1 } },
	{ .index = { 4, 0 }, .dir = {   0, -1 } },
	{ .index = { 5, 0 }, .dir = {  w2,-w2 } },

	{ .index = { 1, 1 }, .dir = {  -1,  0 } },
	{ .index = { 3, 1 }, .dir = {   0,  1 } },
	{ .index = { 5, 1 }, .dir = {   1,  0 } },

	{ .index = { 1, 2 }, .dir = {  -1,  0 } },
	{ .index = { 2, 2 }, .dir = {   1,  0 } },
	{ .index = { 4, 2 }, .dir = {  -1,  0 } },
	{ .index = { 5, 2 }, .dir = {   1,  0 } },

	{ .index = { 1, 3 }, .dir = {  -1,  0 } },
	{ .index = { 3, 3 }, .dir = {   0, -1 } },
	{ .index = { 5, 3 }, .dir = {   1,  0 } },

	{ .index = { 1, 4 }, .dir = { -w2, w2 } },
	{ .index = { 2, 4 }, .dir = {   0,  1 } },
	{ .index = { 3, 4 }, .dir = {   0,  1 } },
	{ .index = { 4, 4 }, .dir = {   0,  1 } },
	{ .index = { 5, 4 }, .dir = {  w2, w2 } },
	};




static complex float *get_normal(const long N, const long dims[N], const complex float *mask)
{
	long grad_dims[N];
	md_copy_dims(N, grad_dims, dims);
	grad_dims[grad_dim] = N - 1;

	complex float *normal = md_alloc(N, grad_dims, CFL_SIZE);
	calc_outward_normal(N, grad_dims, normal, grad_dim, dims, mask);

	return normal;
}




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

	complex float *normal = get_normal(N, dims, mask);

	do {
		long offset = md_calc_offset(N_a, strs, pos);
		ok &= (*(complex float *)((void *)normal + offset) == *(complex float *)((void *)reference + offset));
	} while (ok && md_next(N, grad_dims, MD_BIT(N) - 1, pos));

	md_free(normal);

	return ok;
}



static bool generic_boundary(const long N, const long dims[N], const complex float *mask,
			     const long n_points_ref, const struct boundary_point_s boundary_ref[n_points_ref])
{
	complex float *normal = get_normal(N,dims, mask);

	const long boundary_dimensions[] =  { md_calc_size(N, dims) };
	struct boundary_point_s *boundary = md_alloc(1, boundary_dimensions, sizeof(struct boundary_point_s));

	long grad_dims[N];
	md_copy_dims(N, grad_dims, dims);
	grad_dims[grad_dim] = N-1;

	long n_points = calc_boundary_points(N, grad_dims, boundary, grad_dim, normal);

	md_free(normal);

	debug_printf(DP_DEBUG1, "Number of points - Ref: %d, Calc.:%d\n", n_points_ref, n_points);
	if (n_points != n_points_ref) {
		return false;
	}

	bool ok = true;
	for (int i = 0; i < n_points; i++) {
		const struct boundary_point_s *ref = (boundary_ref + i);
		struct boundary_point_s *point = (boundary + i);

		for(int j = 0; j < N - 1; j++) {
			ok &= (ref->index[j] == point->index[j]);
			ok &= fabs(ref->dir[j] - point->dir[j]) < 1e-7;
		}
		debug_printf(DP_DEBUG1, "Point %d:\t %s\n", i, ok ? "OK" : "FAIL");
	}
	return ok;
}



static bool test_calc_outward_normal(void)
{
	bool ok = true;

	ok &= generic_outward_normal(N_a, dims_a, mask_a, normal_a);
	debug_printf(DP_INFO, "1D:\t\t %s\n", ok ? "OK" : "FAIL");

	ok &= generic_outward_normal(N_b, dims_b, mask_b, normal_b);
	debug_printf(DP_INFO, "2D:\t\t %s\n", ok ? "OK" : "FAIL");

	return ok;
}
UT_REGISTER_TEST(test_calc_outward_normal);



static bool test_calc_boundary_points(void)
{
	bool ok = true;

	ok &= generic_boundary(N_a, dims_a, mask_a, n_points_a , boundary_a);
	debug_printf(DP_INFO, "Boundary 1D:\t %s\n", ok ? "OK" : "FAIL");

	ok &= generic_boundary(N_b, dims_b, mask_b, n_points_b , boundary_b);
	debug_printf(DP_INFO, "Boundary 2D:\t %s\n", ok ? "OK" : "FAIL");

	return ok;
}
UT_REGISTER_TEST(test_calc_boundary_points);
