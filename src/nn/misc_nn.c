#include <assert.h>

#include "misc/mri.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include <math.h>


#include "nlops/nlop.h"
#include "nlops/mri_ops.h"

#include "misc_nn.h"

/**
 * Compute zero filled reconstruction
 */
void compute_zero_filled(const long udims[5], complex float * out, const long kdims[5], const complex float* kspace, const complex float* coil, const long pdims[5], const complex float* pattern)
{
	auto mri_adjoint = nlop_mri_adjoint_create(5, kdims, pdims[4] == 1);

	complex float* tmp = md_alloc_sameplace(5, nlop_generic_codomain(mri_adjoint, 0)->dims, CFL_SIZE, out);
	nlop_generic_apply_unchecked(mri_adjoint, 4, MAKE_ARRAY((void*)tmp, (void*)kspace, (void*)coil, (void*)pattern));

	md_resize_center(5, udims, out, nlop_generic_codomain(mri_adjoint, 0)->dims, tmp, CFL_SIZE);
	nlop_free(mri_adjoint);
	md_free(tmp);
}

/**
 * Compute maximum of u0 per batch
 */
void compute_scale_max_abs(const long dims[5], complex float* scaling, const complex float * u0)
{
	long udimsw[5]; // (Nx, Ny, Nz, 1, Nb)
	long sdimsw[5]; // (1,  1,  1,  1, Nb)
	md_select_dims(5, ~MD_BIT(3), udimsw, dims);
	md_select_dims(5, MD_BIT(4), sdimsw, dims);

	complex float* u0_abs = md_alloc_sameplace(5, udimsw, CFL_SIZE, u0);
	md_zabs(5, udimsw, u0_abs, u0);

	md_clear(5, sdimsw, scaling, CFL_SIZE);
	md_zmax2(5, udimsw, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling, MD_STRIDES(5, udimsw, CFL_SIZE), u0_abs);

	md_free(u0_abs);
}


/**
 * Batchwise divide by scaling
 */
void normalize_by_scale(const long dims[5], const complex float* scaling, complex float* out, const complex float* in)
{
	long sdimsw[5]; // (1,  1,  1,  1, Nb)
	md_select_dims(5, MD_BIT(4), sdimsw, dims);
	complex float* scaling_inv = md_alloc_sameplace(5, sdimsw, CFL_SIZE, scaling);
	md_zfill(5, sdimsw, scaling_inv, 1.);
	md_zdiv(5, sdimsw, scaling_inv, scaling_inv, scaling);
	md_zmul2(5, dims, MD_STRIDES(5, dims, CFL_SIZE), out, MD_STRIDES(5, dims, CFL_SIZE), in, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling_inv);
	md_free(scaling_inv);
}

/**
 * Batchwise multiply by scaling
 */
void renormalize_by_scale(const long dims[5], const complex float* scaling, complex float* out, const complex float* in)
{
	long sdimsw[5]; // (1,  1,  1,  1, Nb)
	md_select_dims(5, MD_BIT(4), sdimsw, dims);
	md_zmul2(5, dims, MD_STRIDES(5, dims, CFL_SIZE), out, MD_STRIDES(5, dims, CFL_SIZE), in, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling);
}