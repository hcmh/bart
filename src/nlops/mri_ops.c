#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/fft.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"

#include "mri_ops.h"

/**
 * If possible we use the same fft operator for all blocks.
 * Thus, we save memory (less data for the diagonals of fft mod).
 * This operator is created by nlop_mri_forward_create or nlop_mri_adjoint_create.
 * It is freed by get_variational_network.
 */
static struct linop_s* linop_fft = NULL;

static void set_fft_op(const long dims[5]) {

	if (NULL == linop_fft)
		linop_fft = linop_fftc_create(5, dims, MD_BIT(0) | MD_BIT(1) | MD_BIT(2));
	else {
		bool same = true;
		for (int i = 0; i < 5; i++)
			same = same && dims[i] == linop_codomain(linop_fft)->dims[i];
		if (!same) {
			linop_free(linop_fft);
			linop_fft = linop_fftc_create(5, dims, MD_BIT(0) | MD_BIT(1) | MD_BIT(2));
		}
	}
}

void reset_fft_op(void) {

	linop_free(linop_fft);
	linop_fft = NULL;
}


/**
 * Returns: MRI forward operator
 *
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param share_mask if true, use the same mask for all batches
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  Nb / 1)
 *
 * Output tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_forward_create(const long dims[5], bool share_mask)
{
	//working dims
	long idims[5]; // (Nx, Ny, Nz, 1,  Nb)
	long mdims[5]; // (Nx, Ny, Nz, 1,  1 / Nb)

	md_select_dims(5, ~MD_BIT(3), idims, dims);
	md_select_dims(5, MD_BIT(0)|MD_BIT(1)|MD_BIT(2), mdims, dims);
	if (!share_mask)
		mdims[4] = dims[4];

	set_fft_op(dims);

	const struct nlop_s* result = nlop_tenmul_create(5, dims, idims, dims); //in: u, coil
	result = nlop_chain2_FF(result, 0, nlop_from_linop(linop_fft), 0); //in: u, coil
	result = nlop_chain2_FF(result, 0, nlop_tenmul_create(5, dims, dims, mdims), 0); //in: mask, u, coil

	int perm[3] = {1, 2, 0};
	result = nlop_permute_inputs_F(result, 3, perm); // in: u, coil, mask

	debug_printf(DP_DEBUG2, "mri forward created\n");
	return result;
}


/**
 * Returns: MRI adjoint operator
 *
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param share_mask if true, use the same mask for all batches
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* nlop_mri_adjoint_create(const long dims[5], bool share_mask)
{
	//working dims
	long idims[5]; // (Nx, Ny, Nz, 1,  Nb)
	long mdims[5]; // (Nx, Ny, Nz, 1,  1 / Nb)

	md_select_dims(5, ~MD_BIT(3), idims, dims);
	md_select_dims(5, MD_BIT(0)|MD_BIT(1)|MD_BIT(2), mdims, dims);
	if (!share_mask)
		mdims[4] = dims[4];

	set_fft_op(dims);

	const struct nlop_s* result = nlop_tenmul_create(5, dims, dims, mdims); //in: kspace, mask
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_adjoint(linop_fft)), 0); //in: kspace, mask
	result = nlop_chain2_FF(result, 0, nlop_tenmul_create(5, idims, dims, dims), 0); //in: coil, kspace, mask

	result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(5, dims)), 0, result, 0); //in: kspace, mask, coil

	int perm[3] = {0, 2, 1};
	result = nlop_permute_inputs_F(result, 3, perm); // in: kspace, coil, mask

	debug_printf(DP_DEBUG2, "mri adjoint created\n");

	return result;
}

/**
 * Returns operator computing update for data fidelity
 *
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param share_mask if true, use the same mask for all batches
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* nlop_gradient_step_unscaled_create(const long dims[5], bool share_mask)
{

	const struct nlop_s* result = nlop_mri_forward_create(dims, share_mask); //in: image, coil, mask
	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(5, dims, 1., -1.), 0); //in: kspace, u, coil, mask
	result = nlop_chain2_FF(result, 0, nlop_mri_adjoint_create(dims,share_mask), 0); //in: coil, mask, kspace, u, coil, mask

	result = nlop_dup_F(result, 0, 4); //in: coil, mask, kspace, u, mask
	result = nlop_dup_F(result, 1, 4); //in: coil, mask, kspace, u

	int perm[4] = {3, 2, 0, 1};
	result = nlop_permute_inputs_F(result, 4, perm); // in: u, kspace, coil, mask

	return result;
}

/**
 * Returns operator computing update for data fidelity
 *
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * scale:	dims:	(1)
 *
 * Output tensors:
 * u:		idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* nlop_gradient_step_scaled_modular_create(const long dims[5], bool share_mask)
{
	const struct nlop_s* result = nlop_gradient_step_unscaled_create(dims, share_mask);
	long idims[5];
	md_select_dims(5, ~MD_BIT(3), idims, dims);
	result = nlop_chain2_FF(result, 0, nlop_tenmul_create(5, idims, idims, MD_SINGLETON_DIMS(5)), 0);
	result = nlop_permute_inputs_F(result, 5, MAKE_ARRAY(1, 2, 3, 4, 0));
	result = nlop_reshape_in_F(result, 4, 1, MAKE_ARRAY(1l));
	return result;
}


struct gradient_step_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* idims;
	const long* mdims;
	const long* kdims;

	unsigned long flags;
	const long* fftdims;

	const long* ridims;

	complex float* fftmod;

	complex float* coil;
	complex float* mask;

	const struct operator_s* fft_plan;
	const struct operator_s* ifft_plan;
};

DEF_TYPEID(gradient_step_s);

static void gradient_step_initialize(struct gradient_step_s* data, const complex float* arg)
{
	if (NULL == data->coil)
		data->coil = md_alloc_sameplace(data->N, data->kdims, CFL_SIZE, arg);
	if (NULL == data->mask)
		data->mask = md_alloc_sameplace(data->N, data->mdims, CFL_SIZE, arg);

	if (NULL == data->fftmod) {

		data->fftmod = md_alloc_sameplace(data->N, data->fftdims, CFL_SIZE, arg);
		complex float* fftmod_tmp = md_alloc(data->N, data->fftdims, CFL_SIZE);
		md_zfill(data->N, data->fftdims, fftmod_tmp, 1);
		fftmod(data->N, data->fftdims, data->flags, fftmod_tmp, fftmod_tmp);
		md_copy(data->N, data->fftdims, data->fftmod, fftmod_tmp, CFL_SIZE);
		md_free(fftmod_tmp);
	}
}

static void gradient_step_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(gradient_step_s, _data);
	assert(5 == Narg);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* kspace = args[2];
	const complex float* coil = args[3];
	const complex float* mask = args[4];

	gradient_step_initialize(d, dst);

	md_copy(d->N, d->mdims, d->mask, mask, CFL_SIZE);
	md_copy(d->N, d->kdims, d->coil, coil, CFL_SIZE);
	
	complex float* coil_image = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	complex float* coil_image2 = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	md_ztenmul(d->N, d->kdims, coil_image, d->idims, image, d->kdims, coil);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);
	
	if (NULL == d->fft_plan)
		d->fft_plan = fft_create(d->N, d->kdims, d->flags, coil_image2, coil_image, false);
	fft_exec(d->fft_plan,  coil_image2, coil_image);

	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->mdims, CFL_SIZE), mask);
	complex float* mask_mod = md_alloc_sameplace(d->N, d->mdims, CFL_SIZE, dst);
	
	md_zmulc2(d->N, d->mdims, MD_STRIDES(d->N, d->mdims, CFL_SIZE), mask_mod, MD_STRIDES(d->N, d->mdims, CFL_SIZE), mask, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->mdims, CFL_SIZE), mask_mod, MD_STRIDES(d->N, d->kdims, CFL_SIZE), kspace);
	md_zaxpy(d->N, d->kdims, coil_image2, -sqrtf(md_calc_size(d->N, d->fftdims)), coil_image);

	if (NULL == d->ifft_plan)
		d->ifft_plan = fft_create(d->N, d->kdims, d->flags, coil_image, coil_image2, true);
	fft_exec(d->ifft_plan,  coil_image, coil_image2);

	md_zmulc2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);

	md_ztenmulc(d->N, d->idims, dst, d->kdims, coil_image, d->kdims, coil);

	md_smul(d->N + 1, d->ridims, (float*)dst, (float*)dst, 1. / md_calc_size(d->N, d->fftdims));
	
	md_free(coil_image);
	md_free(coil_image2);
	md_free(mask_mod);
}

static void gradient_step_deradj_image(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(gradient_step_s, _data);

	complex float* coil_image = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	complex float* coil_image2 = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	md_ztenmul(d->N, d->kdims, coil_image, d->kdims, d->coil, d->idims, src);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);
	if (NULL == d->fft_plan)
		d->fft_plan = fft_create(d->N, d->kdims, d->flags, coil_image2, coil_image, false);
	fft_exec(d->fft_plan,  coil_image2, coil_image);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->mdims, CFL_SIZE), d->mask);
	if (NULL == d->ifft_plan)
		d->ifft_plan = fft_create(d->N, d->kdims, d->flags, coil_image, coil_image2, true);
	fft_exec(d->ifft_plan,  coil_image, coil_image2);
	md_zmulc2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);
	md_ztenmulc(d->N, d->idims, dst, d->kdims, coil_image, d->kdims, d->coil);
	md_smul(d->N + 1, d->ridims, (float*)dst, (float*)dst, 1. / md_calc_size(d->N, d->fftdims));
	md_free(coil_image);
	md_free(coil_image2);
}


static void gradient_step_ni(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	UNUSED(_data);
	UNUSED(dst);
	UNUSED(src);
	error("Not implemented\n");
}

static void gradient_step_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(gradient_step_s, _data);

	md_free(d->fftmod);

	md_free(d->mask);
	md_free(d->coil);

	xfree(d->idims);
	xfree(d->ridims);
	xfree(d->mdims);
	xfree(d->kdims);

	fft_free(d->fft_plan);
	fft_free(d->ifft_plan);

	xfree(d);
}

static struct nlop_s* nlop_gradient_step_create_general(int N, unsigned int flags, const long idims[N], const long kdims[N], const long mdims[N])
{
	PTR_ALLOC(struct gradient_step_s, data);
	SET_TYPEID(gradient_step_s, data);

	PTR_ALLOC(long[N], nidims);
	PTR_ALLOC(long[N], fftdims);
	PTR_ALLOC(long[N], nkdims);
	PTR_ALLOC(long[N], nmdims);
	PTR_ALLOC(long[N + 1], nridims);
	
	md_copy_dims(N, *nidims, idims);
	md_copy_dims(N, *nkdims, kdims);
	md_copy_dims(N, *nmdims, mdims);
	md_select_dims(N, flags, *fftdims, idims);
	md_copy_dims(N, *nridims + 1, idims);
	(*nridims)[0] = 2;

	data->N = N;
	data->idims = *PTR_PASS(nidims);
	data->ridims = *PTR_PASS(nridims);
	data->kdims = *PTR_PASS(nkdims);
	data->mdims = *PTR_PASS(nmdims);
	data->fftdims = *PTR_PASS(fftdims);

	// will be initialized later, to transparently support GPU
	data->fftmod = NULL;
	data->coil = NULL;
	data->mask = NULL;
	data->flags = flags;

	data->fft_plan = NULL;
	data->ifft_plan = NULL;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], idims);

	long nl_idims[5][N];
	md_copy_dims(N, nl_idims[0], idims);
	md_copy_dims(N, nl_idims[1], kdims);
	md_copy_dims(N, nl_idims[2], kdims);
	md_copy_dims(N, nl_idims[3], mdims);
	md_singleton_dims(N, nl_idims[4]);


	return nlop_generic_create(	1, N, nl_odims, 4, N, nl_idims, CAST_UP(PTR_PASS(data)),
					gradient_step_fun,
					(nlop_fun_t[4][1]){ { gradient_step_deradj_image }, { gradient_step_ni }, { gradient_step_ni }, { gradient_step_ni } },
					(nlop_fun_t[4][1]){ { gradient_step_deradj_image }, { gradient_step_ni }, { gradient_step_ni }, { gradient_step_ni } },
					NULL, NULL, gradient_step_del);
}

/**
 * Returns operator computing update for data fidelity
 *
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * scale:	dims:	(1)
 *
 * Output tensors:
 * u:		idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* nlop_gradient_step_scaled_create(const long dims[5], bool share_mask)
{
	long idims[5];
	long mdims[5];
	long kdims[5];

	md_select_dims(5, 23ul, idims, dims);
	md_select_dims(5, share_mask ? 7ul : 23ul, mdims, dims);
	md_copy_dims(5, kdims, dims);

	const struct nlop_s* result = nlop_gradient_step_create_general(5, 7ul, idims, kdims, mdims);
	result = nlop_chain2_FF(result, 0, nlop_tenmul_create(5, idims, idims, MD_SINGLETON_DIMS(5)), 0);
	result = nlop_permute_inputs_F(result, 5, MAKE_ARRAY(1, 2, 3, 4, 0));
	result = nlop_reshape_in_F(result, 4, 1, MAKE_ARRAY(1l));
	return result;
}