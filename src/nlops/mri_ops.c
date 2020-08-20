#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"

#include "mri_ops.h"


/**
 * Returns: MRI forward operator
 *
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  Nb / 1)
 *
 * Output tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], bool share_pattern)
{
	assert(5 == N);

	long cdims[N];
	long pdims[N];
	long idims[N];
	md_select_dims(N, FFT_FLAGS | COIL_FLAG | MD_BIT(4), cdims, dims);
	md_select_dims(N, FFT_FLAGS | (share_pattern ? 0 : MD_BIT(4)), pdims, dims);
	md_select_dims(N, FFT_FLAGS | MD_BIT(4), idims, dims);

	const struct nlop_s* result = nlop_tenmul_create(N, cdims, idims, cdims); //in: image, coil
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_fftc_create(N, dims, FFT_FLAGS)), 0); //in: image, coil
	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, dims, dims, pdims), 0); //in: image, coil, pattern

	debug_printf(DP_DEBUG2, "mri forward created\n");
	return result;
}


/**
 * Returns: MRI forward operator
 *
 * @param idims (Nx, Ny, Nz, 1,  Nb)
 * @param kdims (Nx, Ny, Nz, Nc, Nb)
 * @param cdims (Nx, Ny, Nz, Nc, Nb)
 * @param pdims (Nx, Ny, Nz, 1,  Nb / 1)
 *
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  Nb / 1)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], _Bool share_pattern)
{
	assert(5 == N);

	long cdims[N];
	long pdims[N];
	long idims[N];
	md_select_dims(N, FFT_FLAGS | COIL_FLAG | MD_BIT(4), cdims, dims);
	md_select_dims(N, FFT_FLAGS | (share_pattern ? 0 : MD_BIT(4)), pdims, dims);
	md_select_dims(N, FFT_FLAGS | MD_BIT(4), idims, dims);

	const struct nlop_s* result = nlop_tenmul_create(N, dims, dims, pdims); //in: kspace, pattern
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_ifftc_create(N, dims, FFT_FLAGS)), 0); //in: kspace, pattern
	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, idims, cdims, cdims), 0); //in: kspace, pattern, coil
	result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, cdims)), 0, result, 2); //in: kspace, pattern, coil
	result = nlop_shift_input_F(result, 1, 2); //in: kspace, coil, pattern

	debug_printf(DP_DEBUG2, "mri adjoint created\n");

	return result;
}

/**
 * Returns: MRI normal operator
 *
 * @param idims (Nx, Ny, Nz, 1,  Nb)
 * @param kdims (Nx, Ny, Nz, Nc, Nb)
 * @param cdims (Nx, Ny, Nz, Nc, Nb)
 * @param pdims (Nx, Ny, Nz, 1,  Nb / 1)
 *
 *
 * Input tensors:
 * image:	kdims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  Nb / 1)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_normal_create(int N, const long dims[N], _Bool share_pattern)
{

	auto result = nlop_mri_forward_create(N, dims, share_pattern);
	result = nlop_chain2_swap_FF(result, 0, nlop_mri_adjoint_create(N, dims, share_pattern), 0);

	result = nlop_dup_F(result, 1, 3);
	result = nlop_dup_F(result, 2, 3);

	debug_printf(DP_DEBUG2, "mri normal created\n");

	return result;
}


struct gradient_step_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* idims;
	const long* cdims;
	const long* pdims;
	const long* kdims;

	const long* fftdims;
	complex float* fftmod;

	complex float* coil;
	complex float* pattern;

	const struct operator_s* fft_plan;
	const struct operator_s* ifft_plan;
};

DEF_TYPEID(gradient_step_s);

static void gradient_step_initialize(struct gradient_step_s* data, const complex float* arg, bool der)
{
	if ((der) && (NULL == data->coil))
		data->coil = md_alloc_sameplace(data->N, data->kdims, CFL_SIZE, arg);

	if ((der) && (NULL == data->pattern))
		data->pattern = md_alloc_sameplace(data->N, data->pdims, CFL_SIZE, arg);

	if (!der && (NULL != data->coil)) {

		md_free(data->coil);
		data->coil = NULL;
	}

	if (!der && (NULL != data->pattern)) {

		md_free(data->pattern);
		data->pattern = NULL;
	}

	if (NULL == data->fftmod) {

		data->fftmod = md_alloc_sameplace(data->N, data->fftdims, CFL_SIZE, arg);
		complex float* fftmod_tmp = md_alloc(data->N, data->fftdims, CFL_SIZE);
		md_zfill(data->N, data->fftdims, fftmod_tmp, 1);
		fftmod(data->N, data->fftdims, FFT_FLAGS, fftmod_tmp, fftmod_tmp);
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
	const complex float* pattern = args[4];

	bool der = true;
	gradient_step_initialize(d, dst, der);

	if (der) {

		md_copy(d->N, d->pdims, d->pattern, pattern, CFL_SIZE);
		md_copy(d->N, d->kdims, d->coil, coil, CFL_SIZE);
	}

	complex float* coil_image = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	complex float* coil_image2 = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	md_ztenmul(d->N, d->cdims, coil_image, d->idims, image, d->kdims, coil);

	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);

	if (NULL == d->fft_plan)
		d->fft_plan = fft_create(d->N, d->kdims, FFT_FLAGS, coil_image2, coil_image, false);
	fft_exec(d->fft_plan, coil_image2, coil_image);

	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->pdims, CFL_SIZE), pattern);
	complex float* pattern_mod = md_alloc_sameplace(d->N, d->pdims, CFL_SIZE, dst);

	md_zmulc2(d->N, d->pdims, MD_STRIDES(d->N, d->pdims, CFL_SIZE), pattern_mod, MD_STRIDES(d->N, d->pdims, CFL_SIZE), pattern, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->pdims, CFL_SIZE), pattern_mod, MD_STRIDES(d->N, d->kdims, CFL_SIZE), kspace);
	md_zaxpy(d->N, d->kdims, coil_image2, -sqrtf(md_calc_size(d->N, d->fftdims)), coil_image);

	if (NULL == d->ifft_plan)
		d->ifft_plan = fft_create(d->N, d->kdims, FFT_FLAGS, coil_image, coil_image2, true);
	fft_exec(d->ifft_plan, coil_image, coil_image2);

	md_zmulc2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);

	md_ztenmulc(d->N, d->idims, dst, d->kdims, coil_image, d->kdims, coil);

	md_zsmul(d->N, d->idims, dst, dst, 1. / md_calc_size(d->N, d->fftdims));

	md_free(coil_image);
	md_free(coil_image2);
	md_free(pattern_mod);
}

static void gradient_step_deradj_image(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(gradient_step_s, _data);

	complex float* coil_image = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	complex float* coil_image2 = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	md_ztenmul(d->N, d->kdims, coil_image, d->kdims, d->coil, d->idims, src);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);
	if (NULL == d->fft_plan)
		d->fft_plan = fft_create(d->N, d->kdims, FFT_FLAGS, coil_image2, coil_image, false);
	fft_exec(d->fft_plan, coil_image2, coil_image);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image2, MD_STRIDES(d->N, d->pdims, CFL_SIZE), d->pattern);
	if (NULL == d->ifft_plan)
		d->ifft_plan = fft_create(d->N, d->kdims, FFT_FLAGS, coil_image, coil_image2, true);
	fft_exec(d->ifft_plan, coil_image, coil_image2);
	md_zmulc2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->fftdims, CFL_SIZE), d->fftmod);
	md_ztenmulc(d->N, d->idims, dst, d->kdims, coil_image, d->kdims, d->coil);
	md_zsmul(d->N, d->idims, dst, dst, 1. / md_calc_size(d->N, d->fftdims));
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

	md_free(d->pattern);
	md_free(d->coil);

	xfree(d->idims);
	xfree(d->pdims);
	xfree(d->kdims);
	xfree(d->cdims);

	fft_free(d->fft_plan);
	fft_free(d->ifft_plan);

	xfree(d);
}

/**
 * Returns operator computing gradient step
 * out = AH(A image - kspace)
 *
 * @param idims (Nx, Ny, Nz, 1,  Nb)
 * @param kdims (Nx, Ny, Nz, Nc, Nb)
 * @param cdims (Nx, Ny, Nz, Nc, Nb)
 * @param pdims (Nx, Ny, Nz, 1,  Nb / 1)
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */

const struct nlop_s* nlop_mri_gradient_step_create(int N, const long dims[N], bool share_pattern)
{
	assert(5 == N);

	long cdims[N];
	long pdims[N];
	long idims[N];
	md_select_dims(N, FFT_FLAGS | COIL_FLAG | MD_BIT(4), cdims, dims);
	md_select_dims(N, FFT_FLAGS | (share_pattern ? 0 : MD_BIT(4)), pdims, dims);
	md_select_dims(N, FFT_FLAGS | MD_BIT(4), idims, dims);

	PTR_ALLOC(struct gradient_step_s, data);
	SET_TYPEID(gradient_step_s, data);

	PTR_ALLOC(long[N], nidims);
	PTR_ALLOC(long[N], nkdims);
	PTR_ALLOC(long[N], npdims);
	PTR_ALLOC(long[N], ncdims);

	PTR_ALLOC(long[N], fftdims);

	md_copy_dims(N, *nidims, idims);
	md_copy_dims(N, *nkdims, dims);
	md_copy_dims(N, *npdims, pdims);
	md_copy_dims(N, *ncdims, cdims);

	md_select_dims(N, FFT_FLAGS, *fftdims, dims);

	data->N = N;
	data->idims = *PTR_PASS(nidims);
	data->kdims = *PTR_PASS(nkdims);
	data->pdims = *PTR_PASS(npdims);
	data->cdims = *PTR_PASS(ncdims);

	data->fftdims = *PTR_PASS(fftdims);

	// will be initialized later, to transparently support GPU
	data->fftmod = NULL;
	data->coil = NULL;
	data->pattern = NULL;

	data->fft_plan = NULL;
	data->ifft_plan = NULL;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], idims);

	long nl_idims[4][N];
	md_copy_dims(N, nl_idims[0], idims);
	md_copy_dims(N, nl_idims[1], dims);
	md_copy_dims(N, nl_idims[2], cdims);
	md_copy_dims(N, nl_idims[3], pdims);



	return nlop_generic_create(	1, N, nl_odims, 4, N, nl_idims, CAST_UP(PTR_PASS(data)),
					gradient_step_fun,
					(nlop_fun_t[4][1]){ { gradient_step_deradj_image }, { gradient_step_ni }, { gradient_step_ni }, { gradient_step_ni } },
					(nlop_fun_t[4][1]){ { gradient_step_deradj_image }, { gradient_step_ni }, { gradient_step_ni }, { gradient_step_ni } },
					NULL, NULL, gradient_step_del);
}
