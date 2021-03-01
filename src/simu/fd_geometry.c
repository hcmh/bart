/*
 * Finite difference geometry helper functions
 *
 */

#include "misc/misc.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include <assert.h>
#include <complex.h>
#include <math.h>

#include "linops/grad.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "simu/fd_geometry.h"



/*
 * Calculate normal vector field from a mask (1 inside, 0 outside)
 * output has N+1 dimensions, output_dims[N] = N;
 *
 * @param N 		number of dimensions
 * @param grad_dims 	dimensions of normal
 * @param grad		Output (normal vectors)
 * @param grad_dim 	normal vector dimension
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

	complex float *grad_bw = md_alloc(N, grad_dims, CFL_SIZE);

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



void fill_holes(const long N, const long vec3_dims[N],const long grad_dim, const long dims[N], complex float* out, const complex float *mask)
{
	assert(N > 1);
	assert(dims[grad_dim] == 1);

	const long flags = (MD_BIT(N) - 1) & (~MD_BIT(grad_dim));
	assert(vec3_dims[grad_dim] == bitcount(flags));

	long grad_strs[N], strs[N];
	md_calc_strides(N, grad_strs, vec3_dims, CFL_SIZE);
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float *grad = md_alloc(N, vec3_dims, CFL_SIZE);
	complex float *grad_bw = md_alloc(N, vec3_dims, CFL_SIZE);
	complex float *holes = md_alloc(N, vec3_dims, CFL_SIZE);
	complex float *holes1 = md_calloc(N, dims, CFL_SIZE);
	auto grad_op1 = linop_fd_create(N, dims, grad_dim, flags, 1, BC_ZERO, false);
	auto grad_op2 = linop_fd_create(N, dims, grad_dim, flags, 1, BC_ZERO, true);

	if (mask != out)
		md_copy(N, dims, out, mask, CFL_SIZE);

	bool filled = false;

	while (!filled) {

		linop_forward(grad_op1, N, vec3_dims, grad_bw, N, dims, out);
		// Mask:
		//  0  1  0  0  1  1  1 ->
		// grad_bw:
		//  0  1 -1  0  1  0  0
		md_zmul2(N, vec3_dims, grad_strs, grad_bw, grad_strs, grad_bw, strs, out);
		//  0  1  0  0  1  0  0

		linop_forward(grad_op2, N, vec3_dims, grad, N, dims, out);
		// Mask:
		//  0  1  0  0  1  1  1 ->
		// grad_fw:
		// -1  1  0 -1  0  0  1
		md_zmul2(N, vec3_dims, grad_strs, grad, grad_strs, grad, strs, out);
		//  0  1  0  0  0  0  1

		md_zmul2(N, vec3_dims, grad_strs, holes, grad_strs, grad, grad_strs, grad_bw);
		//  0  1  0  0  0  0  0

		md_zss(N, vec3_dims, MD_BIT(grad_dim), holes1, holes);
		md_zsgreatequal(N, dims, holes1, holes1, 1);

		md_zaxpy(N, dims, out, -1, holes1);
		filled = ( md_znorm(N, dims, holes1) < 1 );
	}

	md_free(grad_bw);
	md_free(grad);
	md_free(holes);
	md_free(holes1);
	linop_free(grad_op1);
	linop_free(grad_op2);
}

/*
 * Calculate list of points neighbouring the mask
 * @param N 		number of dimensions
 * @param dims 		dimensions
 * @param boundary	sufficiently large allocation to store boundary points
 * @param grad_dim	normal vector dimension
 * @param normal	outward normal
 *
 * @returns 		number of points on the boundary
 */
long calc_boundary_points(const long N, const long dims[N], struct boundary_point_s *boundary, const long grad_dim, const complex float *normal, const complex float *values)
{
	assert(N <= N_boundary_point_s);

	long pos[N], strs[N], value_dims[N], value_strs[N], boundary_pos[N], n_points = 0, offset = 0;
	const long flags = (MD_BIT(N) - 1) & (~MD_BIT(grad_dim));

	assert(dims[grad_dim] == bitcount(flags));
	assert(!MD_IS_SET(flags, grad_dim));

	md_set_dims(N, pos, 0);
	md_set_dims(N, boundary_pos, 0);
	md_select_dims(N, flags, value_dims, dims);

	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, value_strs, value_dims, CFL_SIZE);

	do {
		offset = md_calc_offset(N, strs, pos);
		struct boundary_point_s *point = boundary + n_points;

		bool is_boundary = false;
		float val = 0;
		md_set_dims(N, point->dir, 0);
		for (int i = 0; i < dims[grad_dim]; i++) {
			val = *((complex float *)((void *)normal + offset + i * strs[grad_dim]));
			point->dir[i] = (fabsf(val) > 1e-16) ? (long)(val / fabsf(val)) : 0;
			is_boundary |= (fabsf(val) > 1e-16);
		}

		if (is_boundary) {
			int k = 0;
			md_set_dims(N, point->index, 0);
			for (int j = 0; j < N; j++) {
				if (j != grad_dim) {
					// grad_dim is not relevant for list of boundary points
					point->index[k] = pos[j];
					boundary_pos[j] = point->index[k] + point->dir[k];
					k++;
				}
			}
			if (NULL != values) {
				long value_offset = md_calc_offset(N, value_strs, boundary_pos);
				complex float val = *(complex float *)((void *)values + value_offset);
				point->val = val;
			} else {
				point->val = 0;
			}
			n_points++;
		}

	} while (md_next(N, dims, flags, pos));
	return n_points;
}

/*
 * eat away one pixel in forward direction
 */
void clear_mask_forward(const long N, const long dims[N], complex float *out, const long n_points, const struct boundary_point_s *boundary, const complex float *mask)
{
	md_copy(N, dims, out, mask, CFL_SIZE);
	long offset = 0, pos[N], strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	for (long i = 0; i < n_points; i++) {
		for (long j = 0; j < N; j++) {
			if (boundary[i].dir[j] > 0) {
				md_copy_dims(N, pos, boundary[i].index);
				pos[j] += 1;
				assert(pos[j] < dims[j]);
				offset = md_calc_offset(N, strs, pos);
				*(complex float *)((void *)out + offset) = 1;
			}
		}
	}
}


/*
 * Shrink a mask by setting the pixels on the boundary to 0.
 * @param N		number of dimensions
 * @param dims		dimensions
 * @param dst		new mask
 * @param n_points	number of points on the boundary
 * @param boundary	list of boundary points
 * @param src		old mask
 */
void shrink_wrap(const long N, const long dims[N], complex float *dst, const long n_points, const struct boundary_point_s *boundary, const complex float *src)
{
	if (dst != src)
		md_copy(N, dims, dst, src, CFL_SIZE);

	long offset = 0, strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	for (long i = 0; i < n_points; i++) {
		offset = md_calc_offset(N, strs, boundary[i].index);
		*(complex float *)((void *)dst + offset) = 0;
	}
}
