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
 * Calculate list of points neighbouring the mask
 * @param N number of dimensions
 * @param dims dimensions of underlying grid
 * @param mask mask which is 0 in the exterior, 1 inside
 * @param where to store boundary_point_s
 *
 * @returns number of points on the boundary
 */



/*
 * Calculate normal vector field from a mask (1 inside, 0 outside)
 * output has N+1 dimensions, output_dims[N] = N;
 *
 * @param N number of dimensions
 * @param dims dimensions of mask
 * @param mask mask
 *
 * @returns pointer to newly allocated outward normal vector field
 */

complex float *calc_outward_normal(const long N, const long dims[N], const complex float *mask)
{
	assert(N > 0);

	const long grad_dim = N;
	const long flags = MD_BIT(N) - 1;
	long grad_dims[N+1], extended_dims[N+1];
	md_copy_dims(N, grad_dims, dims);
	md_copy_dims(N, extended_dims, dims);
	grad_dims[grad_dim] = N;
	extended_dims[grad_dim] = 1;

	assert(grad_dims[grad_dim] == bitcount(flags));

	long grad_strs[N+1], strs[N];
	md_calc_strides(N+1, grad_strs, grad_dims, CFL_SIZE);
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* grad_bw = md_alloc(N+1, grad_dims, CFL_SIZE);
	complex float* grad = md_alloc(N+1, grad_dims, CFL_SIZE);

	auto grad_op = linop_fd_create(N+1, extended_dims, N, flags, 1, BC_ZERO, false);
	linop_forward(grad_op, N+1, grad_dims, grad_bw, N+1, extended_dims, mask);
	linop_free(grad_op);
	// Mask:
	//  0  1  1  0  1  1  1 ->
	// grad_bw:
	//  0  1  0 -1  1  0  0
	md_zmul2(N+1, grad_dims, grad_strs, grad_bw, grad_strs, grad_bw, strs, mask);
	//  0  1  0  0  1  0  0

	grad_op = linop_fd_create(N+1, extended_dims, N, flags, 1, BC_ZERO, true);
	linop_forward(grad_op, N+1, grad_dims, grad, N+1, extended_dims, mask);
	linop_free(grad_op);
	// Mask:
	//  0  1  1  0  1  1  1 ->
	// grad_fw:
	// -1  0  1 -1  0  0  1
	md_zmul2(N+1, grad_dims, grad_strs, grad, grad_strs, grad, strs, mask);
	//  0  0  1  0  0  0  1

	// every voxel is either interior, front xor backwall
	assert(md_zscalar_real(N+1, grad_dims, grad, grad_bw) < 1e-16);
	md_zaxpy(N+1, grad_dims, grad, -1, grad_bw);
	md_free(grad_bw);
	//  0 -1  1  0 -1  0  1

	// normalize
	complex float *norm = md_alloc(N, dims, CFL_SIZE);
	md_zmul2(N+1, grad_dims, strs, norm, grad_strs, grad, grad_strs, grad);
	md_zsqrt(N, dims, norm, norm);

	md_zdiv_reg2(N+1, grad_dims, grad_strs, grad, grad_strs, grad, strs, norm, 0);
	// Outward normal field.
	md_free(norm);
	return grad;
}
/*
	long pos[N+1];
	long offset = 0;
	long n_points = 0;
	NESTED(bool, is_boundary, (long N, const long offset)) {
		bool boundary = false;
		int i = -1;
		while ((++i < N) && !boundary)
			boundary |= (*((complex float *) ((void *)grad + offset + i)) != 0);
		return boundary;
	};
	do {
		offset = md_calc_offset(N+1, grad_strs, pos);
		if(is_boundary(N, offset)) {
			struct boundary_point_s *point = boundary + n_points++;
			point->offset = offset;
			md_copy_dims(N, point->index, pos + 1);
		}
	} while (md_next(N+1, grad_dims, flags, pos);
}
*/
