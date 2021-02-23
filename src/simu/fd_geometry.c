/*
 * Poisson Equation with Neuman Boundary Conditions
 * on non-rectangular domain
 */

#include <assert.h>
#include <complex.h>
#include "num/flpmath.h"
#include "num/multind.h"
#include "misc/misc.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "simu/fd_geometry.h"



/*
 * Calculate normal vector field from a mask (1 inside, 0 outside)
 * output has N+1 dimensions, output_dims[N] = N;
 *
 * @param N 		number of dimensions
 * @param grad_dims 	dimensions of normal
 * @param grad		Output (normal vectors)
 * @param grad_dim 	dimension which will hold normal direction
 * @param dims		Mask dimensions
 * @param mask mask     Input (Mask)
 *
 */
void calc_outward_normal(const long N, const long grad_dims[N], complex float *grad, const long grad_dim, const long dims[N], const complex float *mask)
{
	assert(N > 1);
	assert(dims[grad_dim] == 1);

	const long flags = (MD_BIT(N) - 1) & (~MD_BIT(grad_dim));
	assert(grad_dims[grad_dim] == bitcount(flags));

	long grad_strs[N], strs[N];
	md_calc_strides(N, grad_strs, grad_dims, CFL_SIZE);
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* grad_bw = md_alloc(N, grad_dims, CFL_SIZE);

	auto grad_op = linop_fd_create(N, dims, grad_dim, flags, 1, BC_ZERO, false);
	linop_forward(grad_op, N, grad_dims, grad_bw, N, dims, mask);
	linop_free(grad_op);
	// Mask:
	//  0  1  1  0  1  1  1 ->
	// grad_bw:
	//  0  1  0 -1  1  0  0
	md_zmul2(N, grad_dims, grad_strs, grad_bw, grad_strs, grad_bw, strs, mask);
	//  0  1  0  0  1  0  0

	grad_op = linop_fd_create(N, dims, grad_dim, flags, 1, BC_ZERO, true);
	linop_forward(grad_op, N, grad_dims, grad, N, dims, mask);
	linop_free(grad_op);
	// Mask:
	//  0  1  1  0  1  1  1 ->
	// grad_fw:
	// -1  0  1 -1  0  0  1
	md_zmul2(N, grad_dims, grad_strs, grad, grad_strs, grad, strs, mask);
	//  0  0  1  0  0  0  1

	// every voxel is either interior, front xor backwall
	assert(md_zscalar_real(N, grad_dims, grad, grad_bw) < 1e-16);
	md_zaxpy(N, grad_dims, grad, -1, grad_bw);
	//  0 -1  1  0 -1  0  1

	md_free(grad_bw);

	// normalize
	complex float *norm = md_alloc(N, dims, CFL_SIZE);
	md_zrss(N, grad_dims, MD_BIT(grad_dim), norm, grad);
	md_zdiv_reg2(N, grad_dims, grad_strs, grad, grad_strs, grad, strs, norm, 0);
	// Outward normal field.
	md_free(norm);
}



/*
 * Calculate list of points neighbouring the mask
 * @param N 		number of dimensions
 * @param dims 		dimensions
 * @param boundary	sufficiently large allocation to store boundary points
 * @param grad_dim	vector component index dimension
 * @param normal	outward normal
 *
 * @returns 		number of points on the boundary
 */
long calc_boundary_points(const long N, const long dims[N], struct boundary_point_s *boundary, const long grad_dim, const complex float *normal)
{
	assert(N  <= N_boundary_point_s);
	assert(N - 1 == grad_dim);

	long pos[N], strs[N], n_points = 0, offset = 0;
	const long flags = (MD_BIT(N) - 1) & (~MD_BIT(grad_dim));

	assert(dims[grad_dim] == bitcount(flags));
	assert(!MD_IS_SET(flags, grad_dim));

	md_set_dims(N, pos, 0);
	md_calc_strides(N, strs, dims, CFL_SIZE);

	do {
		offset = md_calc_offset(N, strs, pos);

		bool is_boundary = false;
		for ( int i = 0; (i < dims[grad_dim]) && (!is_boundary); i++)
			is_boundary  |= (*((complex float *) ((void *)normal + offset + i*strs[grad_dim])) != 0);


		if (is_boundary) {
			struct boundary_point_s *point = boundary + n_points++;
			md_select_dims(N, flags, point->index, pos);
			for (int i = 0; i < dims[grad_dim]; i++)
				point->dir[i] = creal(*((complex float *) ((void *)normal + offset + i*strs[grad_dim])));
		}

	} while (md_next(N, dims, flags, pos));
	return n_points;
}
