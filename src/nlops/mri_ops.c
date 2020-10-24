#include <math.h>


#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "misc/types.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/iter.h"
#include "iter/iter2.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/const.h"
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

	bool der = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
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

static void gradient_step_deradj_image(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

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


static void gradient_step_ni(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

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

	operator_property_flags_t props[4][1] = { { 0 }, { 0 }, { 0 }, { 0 } };

	return nlop_generic_with_props_create(	1, N, nl_odims, 4, N, nl_idims, CAST_UP(PTR_PASS(data)),
						gradient_step_fun,
						(nlop_der_fun_t[4][1]){ { gradient_step_deradj_image }, { gradient_step_ni }, { gradient_step_ni }, { gradient_step_ni } },
						(nlop_der_fun_t[4][1]){ { gradient_step_deradj_image }, { gradient_step_ni }, { gradient_step_ni }, { gradient_step_ni } },
						NULL, NULL, gradient_step_del, NULL, props, NULL);
}




struct mri_normal_inversion_s {

	INTERFACE(nlop_data_t);

	int N;
	int N_op;
	int batches_independent;

	const long* idims;
	const long* pdims;
	const long* cdims;
	const long* kdims;

	complex float* coil;

	complex float* fftmod;
	complex float* out;

	bool fixed_lambda;
	float check_convergence_warning_val;

	const struct operator_s** normal_op;

	const struct linop_s* lop_fft;

	italgo_fun2_f* algo;
	iter_conf* conf;
};

DEF_TYPEID(mri_normal_inversion_s);

static void mri_normal_inversion(const struct mri_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	assert(NULL != d->normal_op);

	complex float* dst_loop = dst;
	const complex float* src_loop = src;

	md_clear(d->N, d->idims, dst, CFL_SIZE);

	for (int i = 0; i < d->batches_independent; i++) {

		d->algo(d->conf, d->normal_op[i],
			0, NULL, NULL, NULL, NULL,
			2 * md_calc_size(d->N_op, d->idims), (float*)dst_loop, (float*)src_loop,
			NULL);
		dst_loop += md_calc_size(d->N_op, d->idims);
		src_loop += md_calc_size(d->N_op, d->idims);
	}
}

static void mri_normal(const struct mri_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	assert(NULL != d->normal_op);

	complex float* dst_loop = dst;
	const complex float* src_loop = src;

	for (int i = 0; i < d->batches_independent; i++) {

		operator_apply_unchecked(d->normal_op[i], dst_loop, src_loop);
		dst_loop += md_calc_size(d->N_op, d->idims);
		src_loop += md_calc_size(d->N_op, d->idims);
	}
}

static void mri_normal_inversion_check(const struct mri_normal_inversion_s* d, complex float* dst, const complex float* src, const char* str)
{
	if (0 == d->check_convergence_warning_val)
		return;

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);

	for (int i = 0; i < d->batches_independent; i++)
		operator_apply(d->normal_op[i], d->N_op, d->idims, tmp + i * md_calc_size(d->N_op, d->idims), d->N_op, d->idims, dst + i * md_calc_size(d->N_op, d->idims));

	md_zaxpy(d->N, d->idims, tmp, d->conf->alpha, dst);
	float err = md_zrmse(d->N, d->idims, src, tmp);
	float scale = md_zrms(d->N, d->idims, tmp);
	if (d->check_convergence_warning_val < err / scale)
		debug_printf(DP_WARN, "%s did not converge (error: %e, scale: %e)\n", str, err, scale);
	md_free(tmp);
}

static void mri_normal_inversion_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, src);
	mri_normal_inversion_check(d, dst, src, "mri ninv derivative");

	PRINT_TIMER("der mri ninv");
}

static void mri_normal_inversion_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, src);
	mri_normal_inversion_check(d, dst, src, "mri ninv adjoint");

	PRINT_TIMER("adj mri ninv");
}

static void mri_normal_inversion_der_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, d->out);
	mri_normal_inversion_check(d, dst, d->out, "mri ninv derivative lambda");

	md_zmul2(d->N, d->idims, MD_STRIDES(d->N, d->idims, CFL_SIZE), dst, MD_STRIDES(d->N, d->idims, CFL_SIZE), dst, MD_SINGLETON_STRS(d->N), src);
	md_zsmul(d->N, d->idims, dst, dst, -1);

	PRINT_TIMER("der mri ninv lambda");
}

static void mri_normal_inversion_adj_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	assert(NULL != d->normal_op);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);

	mri_normal_inversion(d, tmp, d->out);
	mri_normal_inversion_check(d, tmp, d->out, "mri ninv adjoint lambda");

	md_zconj(d->N, d->idims, tmp, tmp);
	md_ztenmul(d->N, MD_SINGLETON_DIMS(d->N), dst, d->idims, src, d->idims, tmp);
	md_free(tmp);

	md_zsmul(d->N, MD_SINGLETON_DIMS(d->N), dst, dst, -1);
	md_zreal(d->N, MD_SINGLETON_DIMS(d->N), dst, dst);

	PRINT_TIMER("der mri normal inversion lambda");
}

static void mri_normal_inversion_set_normal_ops(struct mri_normal_inversion_s* d, const complex float* coil, const complex float* pattern, const complex float* lptr)
{
	complex float lambda;
	md_copy(1, MAKE_ARRAY(1l), &lambda, lptr, CFL_SIZE);

	if ((0 != cimagf(lambda)) || (0 > crealf(lambda)))
		error("Lambda=%f+%fi is not non-negative real number!\n", crealf(lambda), cimagf(lambda));
	d->conf->alpha = crealf(lambda);

	long fftdims[d->N];
	md_select_dims(d->N, FFT_FLAGS, fftdims , d->kdims);

	if (NULL == d->fftmod) {

		complex float * tmp = md_alloc(d->N, fftdims, CFL_SIZE);
		md_zfill(d->N, fftdims, tmp, 1.);
		fftmod(d->N, fftdims, FFT_FLAGS, tmp, tmp);

		d->fftmod = md_alloc_sameplace(d->N, fftdims, CFL_SIZE, coil);
		md_resize_center(d->N, fftdims, d->fftmod , fftdims, tmp, CFL_SIZE);
		md_free(tmp);
	}

	if (NULL == d->coil)
		d->coil = md_alloc_sameplace(d->N, d->cdims, CFL_SIZE, coil);
	md_zmul2(d->N, d->cdims, MD_STRIDES(d->N, d->cdims, CFL_SIZE), d->coil, MD_STRIDES(d->N, d->cdims, CFL_SIZE), coil, MD_STRIDES(d->N, fftdims, CFL_SIZE), d->fftmod);


	complex float* tmp_pattern = md_alloc_sameplace(d->N, d->pdims, CFL_SIZE, pattern);
	md_zsmul(d->N, d->pdims, tmp_pattern, pattern, 1. / md_calc_size(3, d->pdims));

	for (int i = 0; i < d->batches_independent; i++){

		if (NULL != d->normal_op[i])
			operator_free(d->normal_op[i]);

		auto linop_frw = linop_fmac_create(d->N_op, d->cdims, 0, MD_BIT(3), 0, d->coil + i * md_calc_size(d->N_op, d->cdims));

		if (!md_check_equal_dims(d->N_op, d->cdims, d->kdims, ~0u))
			linop_frw = linop_chain_FF(linop_frw, linop_resize_center_create(d->N_op, d->kdims, d->cdims));

		if (NULL == d->lop_fft)
			d->lop_fft = linop_fft_create(d->N_op, d->kdims, FFT_FLAGS);

		linop_frw = linop_chain_FF(linop_frw, linop_clone(d->lop_fft));

		auto linop_pattern = linop_cdiag_create(d->N_op, d->kdims, (1 == d->pdims[4]) ? FFT_FLAGS : FFT_FLAGS | MD_BIT(4), (1 == d->pdims[4]) ? tmp_pattern : tmp_pattern + i * md_calc_size(3, d->pdims));

		// normal operator is constructed manually to apply linop_pattern only once pattern^H(pattern(x)) = pattern(x)
		d->normal_op[i] = operator_chainN(3, (const struct operator_s **)MAKE_ARRAY(linop_frw->forward, linop_pattern->forward, linop_frw->adjoint));

		linop_free(linop_frw);
		linop_free(linop_pattern);
	}

	md_free(tmp_pattern);

}

static void mri_free_normal_ops(struct mri_normal_inversion_s* d)
{
	md_free(d->coil);
	d->coil = NULL;

	for (int i = 0; i < d->batches_independent; i++) {

		operator_free(d->normal_op[i]);
		d->normal_op[i] = NULL;
	}
}

static void mri_normal_inversion_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(5 == Narg);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];
	const complex float* lptr = args[4];

	bool der_in = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
	bool der_lam = !op_options_is_set_io(_data->options, 0, 3, OP_APP_NO_DER);

	mri_normal_inversion_set_normal_ops(d, coil, pattern, lptr);

	mri_normal_inversion(d, dst, image);
	mri_normal_inversion_check(d, dst, image, "mri ninv frw");

	if (!d->fixed_lambda && der_lam) {

		if(NULL == d->out)
			d->out = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);
		md_copy(d->N, d->idims, d->out, dst, CFL_SIZE);
	} else {
		md_free(d->out);
		d->out = NULL;

		if (!der_in)
			mri_free_normal_ops(d);
	}
}

static void  mri_normal_inversion_ni(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	UNUSED(_data);
	UNUSED(dst);
	UNUSED(src);
	error("Derivative of mri_normal_inversion is not available\n");
}

static void mri_normal_inversion_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	xfree(d->idims);
	xfree(d->pdims);
	xfree(d->cdims);
	xfree(d->kdims);

	md_free(d->out);
	md_free(d->coil);
	md_free(d->fftmod);

	linop_free(d->lop_fft);

	for (int i = 0; i < d->batches_independent; i++)
		operator_free(d->normal_op[i]);
	xfree(d->normal_op);


	xfree(d->conf);

	xfree(d);
}

static struct nlop_data_s* mri_normal_inversion_data_create(int N, const long dims[N], bool share_pattern, float lambda, bool batch_independent, float convergence_warn_limit, iter_conf* conf)
{
	if (NULL == conf) {

		struct iter_conjgrad_conf def_conf = iter_conjgrad_defaults;
		def_conf.l2lambda = 1.;
		def_conf.maxiter = 50;

		return mri_normal_inversion_data_create(N, dims, share_pattern, lambda, batch_independent, convergence_warn_limit, CAST_UP(&def_conf));
	}

	PTR_ALLOC(struct mri_normal_inversion_s, data);
	SET_TYPEID(mri_normal_inversion_s, data);

	PTR_ALLOC(long[N], nidims);
	PTR_ALLOC(long[N], ncdims);
	PTR_ALLOC(long[N], nkdims);
	PTR_ALLOC(long[N], npdims);

	md_select_dims(N, ~MD_BIT(3), *nidims, dims);
	md_copy_dims(N, *ncdims, dims);
	md_copy_dims(N, *nkdims, dims);
	md_select_dims(N, FFT_FLAGS | (share_pattern ? 0 : MD_BIT(4)), *npdims, dims);

	data->N = N;
	data->idims = *PTR_PASS(nidims);
	data->pdims = *PTR_PASS(npdims);
	data->cdims = *PTR_PASS(ncdims);
	data->kdims = *PTR_PASS(nkdims);

	data->batches_independent = batch_independent ? dims[4] : 1;
	data->N_op = batch_independent ? N - 1 : N;

	// will be initialized later, to transparently support GPU
	data->out= NULL;
	data->coil = NULL;
	data->fftmod = NULL;

	data->fixed_lambda = (-1. != lambda);
	PTR_ALLOC(const struct operator_s*[data->batches_independent], normalops);
	for (int i = 0; i < data->batches_independent; i++)
		(*normalops)[i] = NULL;
	data->normal_op = *PTR_PASS(normalops);

	data->conf = NULL;

	data->lop_fft = NULL;

	if (NULL != CAST_MAYBE(iter_conjgrad_conf, conf)) {

		PTR_ALLOC(struct iter_conjgrad_conf, nconf);
		memcpy(nconf, CAST_DOWN(iter_conjgrad_conf, conf), sizeof(struct iter_conjgrad_conf));

		assert(nconf->l2lambda == 1.);

		data->conf = CAST_UP(PTR_PASS(nconf));
		data->conf->alpha = lambda;

		data->algo = iter2_conjgrad;
	}

	if (NULL == data->conf)
		error("Iteration configuration not supported in mri_normal_inversion_create!");

	data->check_convergence_warning_val = convergence_warn_limit;

	return CAST_UP(PTR_PASS(data));
}


/**
 * Create an opertor applying the inverse normal mri forward model on its input
 *
 * out = (A^HA +l1)^-1 in
 * A = Pattern FFT Coils
 *
 * @param N # of dims (must be: 5)
 * @param dims dimensions [Nx, Ny, Nz, Nc, Nb]
 * @param share_pattern select if the same pattern is used for all eelements in the batch
 * @param lambda regularization value (-1) corresponds to additional operator input
 * @param batch_independent select if minimization is performed for each batch independently (for example useful for CG)
 * @param convergence_warn_limit warn if minimization is not converged (0 corresponds to no warnings)
 * @param conf pointer to configuration for iterative algorithm, NULL will create default conf using CG
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * [lambda:	ldims:	(1)]
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */

const struct nlop_s* mri_normal_inversion_create(int N, const long dims[N], bool share_pattern, float lambda, bool batch_independent, float convergence_warn_limit, iter_conf* conf)
{
	auto data = mri_normal_inversion_data_create(N, dims, share_pattern, lambda, batch_independent, convergence_warn_limit, conf);

	long nl_odims[1][N];
	md_select_dims(N, ~MD_BIT(3), nl_odims[0], dims);

	long nl_idims[4][N];
	md_select_dims(N, ~MD_BIT(3), nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);
	md_select_dims(N, FFT_FLAGS | (share_pattern ? 0 : MD_BIT(4)), nl_idims[2], dims);
	md_singleton_dims(N, nl_idims[3]);

	operator_property_flags_t props[4][1] = { { MD_BIT(OP_PROP_C_LIN) }, { 0 }, { 0 }, { 0 } };

	auto result = nlop_generic_with_props_create(	1, N, nl_odims, 4, N, nl_idims, data,
							mri_normal_inversion_fun,
							(nlop_der_fun_t[4][1]){ { mri_normal_inversion_der }, { mri_normal_inversion_ni }, { mri_normal_inversion_ni }, { mri_normal_inversion_der_lambda } },
							(nlop_der_fun_t[4][1]){ { mri_normal_inversion_adj }, { mri_normal_inversion_ni }, { mri_normal_inversion_ni }, { mri_normal_inversion_adj_lambda } },
							NULL, NULL, mri_normal_inversion_del, NULL, props, NULL);

	if (-1. != lambda) {
		complex float lambdac = lambda;
		return nlop_set_input_const_F(result, 3, N, MD_SINGLETON_DIMS(N), true, &lambdac);
	} else
		return nlop_reshape_in_F(result, 3, 1, MD_SINGLETON_DIMS(1));
}

/**
 * Create an opertor applying the inverse normal mri forward model on its input
 *
 * out = (A^HA +l1)^-1 in
 * A = Pattern FFT Coils
 *
 * @param N # of dims (must be: 5)
 * @param dims dimensions [Nx, Ny, Nz, Nc, Nb]
 * @param share_pattern select if the same pattern is used for all eelements in the batch
 * @param lambda regularization value (-1) corresponds to additional operator input
 * @param batch_independent select if minimization is performed for each batch independently (for example useful for CG)
 * @param convergence_warn_limit warn if minimization is not converged (0 corresponds to no warnings)
 * @param conf pointer to configuration for iterative algorithm, NULL will create default conf using CG
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * lambda:	ldims:	(1) input is ignored but present if -1 != lambda
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* mri_normal_inversion_create_with_lambda(int N, const long dims[N], bool share_pattern, float lambda, bool batch_independent, float convergence_warn_limit, iter_conf* conf)
{
	auto result = mri_normal_inversion_create(N, dims, share_pattern, lambda, batch_independent, convergence_warn_limit, conf);
	if (-1. != lambda)
		result = nlop_combine_FF(result, nlop_del_out_create(1, MD_SINGLETON_DIMS(1)));
	return result;
}



static void mri_reg_proj_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	assert(NULL != d->normal_op);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);

	mri_normal(d, tmp, src);
	mri_normal_inversion(d, dst, tmp);
	mri_normal_inversion_check(d, dst, tmp, "mri proj der");

	md_free(tmp);

	PRINT_TIMER("der mri regularized projection");
}

static void mri_reg_proj_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(NULL != d->normal_op);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);

	mri_normal_inversion(d, tmp, src);
	mri_normal_inversion_check(d, tmp, src, "mri proj adj");
	mri_normal(d, dst, tmp);

	md_free(tmp);

	PRINT_TIMER("der mri regularized projection");
}

static void mri_reg_proj_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(5 == Narg);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];
	const complex float* lptr = args[4];

	bool der_in = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
	bool der_lam = !op_options_is_set_io(_data->options, 0, 3, OP_APP_NO_DER);

	mri_normal_inversion_set_normal_ops(d, coil, pattern, lptr);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);
	mri_normal(d, tmp, image);
	mri_normal_inversion(d, dst, tmp);
	mri_normal_inversion_check(d, dst, tmp, "mri proj frw");

	md_free(tmp);

	if (!d->fixed_lambda && der_lam) {

		if(NULL == d->out)
			d->out = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);
		md_copy(d->N, d->idims, d->out, dst, CFL_SIZE);
	} else {
		md_free(d->out);
		d->out = NULL;

		if (!der_in)
			mri_free_normal_ops(d);
	}
}


 /**
 * Create an opertor projecting its input to the kernel of the mri forward operator (regularized with lambda)
 *
 * out = (id - (A^HA +l1)^-1A^HA) in
 * A = Pattern FFT Coils
 *
 * @param N # of dims (must be: 5)
 * @param dims dimensions [Nx, Ny, Nz, Nc, Nb]
 * @param share_pattern select if the same pattern is used for all eelements in the batch
 * @param lambda regularization value (-1) corresponds to additional operator input
 * @param batch_independent select if minimization is performed for each batch independently (for example useful for CG)
 * @param convergence_warn_limit warn if minimization is not converged (0 corresponds to no warnings)
 * @param conf pointer to configuration for iterative algorithm, NULL will create default conf using CG
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * [lambda:	ldims:	(1)]
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* mri_reg_proj_ker_create(int N, const long dims[N], bool share_pattern, float lambda, bool batch_independent, float convergence_warn_limit, iter_conf* conf)
{
	auto data = mri_normal_inversion_data_create(N, dims, share_pattern, lambda, batch_independent, convergence_warn_limit, conf);

	long nl_odims[1][N];
	md_select_dims(N, ~MD_BIT(3), nl_odims[0], dims);

	long nl_idims[4][N];
	md_select_dims(N, ~MD_BIT(3), nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);
	md_select_dims(N, FFT_FLAGS | (share_pattern ? 0 : MD_BIT(4)), nl_idims[2], dims);
	md_singleton_dims(N, nl_idims[3]);

	operator_property_flags_t props[4][1] = { { MD_BIT(OP_PROP_C_LIN) }, { 0 }, { 0 }, { 0 } };

	auto result = nlop_generic_with_props_create(	1, N, nl_odims, 4, N, nl_idims, data,
							mri_reg_proj_fun,
							(nlop_der_fun_t[4][1]){ { mri_reg_proj_der }, { mri_normal_inversion_ni }, { mri_normal_inversion_ni }, { mri_normal_inversion_der_lambda } },
							(nlop_der_fun_t[4][1]){ { mri_reg_proj_adj }, { mri_normal_inversion_ni }, { mri_normal_inversion_ni }, { mri_normal_inversion_adj_lambda } },
							NULL, NULL, mri_normal_inversion_del, NULL, props, NULL);

	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, nl_idims[0], -1., 1.), 0);
	result = nlop_dup_F(result, 0, 1);

	if (-1. != lambda) {
		complex float lambdac = lambda;
		return nlop_set_input_const_F(result, 3, N, MD_SINGLETON_DIMS(N), true, &lambdac);
	} else
		return nlop_reshape_in_F(result, 3, 1, MD_SINGLETON_DIMS(1));

}


 /**
 * Create an opertor projecting its input to the kernel of the mri forward operator (regularized with lambda)
 *
 * out = (id - (A^HA +l1)^-1A^HA) in
 * A = Pattern FFT Coils
 *
 * @param N # of dims (must be: 5)
 * @param dims dimensions [Nx, Ny, Nz, Nc, Nb]
 * @param share_pattern select if the same pattern is used for all eelements in the batch
 * @param lambda regularization value (-1) corresponds to additional operator input
 * @param batch_independent select if minimization is performed for each batch independently (for example useful for CG)
 * @param convergence_warn_limit warn if minimization is not converged (0 corresponds to no warnings)
 * @param conf pointer to configuration for iterative algorithm, NULL will create default conf using CG
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * lambda:	ldims:	(1) input is ignored but present if -1 != lambda
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* mri_reg_proj_ker_create_with_lambda(int N, const long dims[N], bool share_pattern, float lambda, bool batch_independent, float convergence_warn_limit, iter_conf* conf)
{
	auto result = mri_reg_proj_ker_create(N, dims, share_pattern, lambda, batch_independent, convergence_warn_limit, conf);
	if (-1. != lambda)
		result = nlop_combine_FF(result, nlop_del_out_create(1, MD_SINGLETON_DIMS(1)));

	return result;
}

/**
 * Create an opertor computing the Tickhonov regularized pseudo-inverse of the MRI operator
 *
 * out = [(1 + lambda)](A^HA +l1)^-1 A^Hin
 * A = Pattern FFT Coils
 *
 * @param N # of dims (must be: 5)
 * @param dims dimensions [Nx, Ny, Nz, Nc, Nb]
 * @param share_pattern select if the same pattern is used for all eelements in the batch
 * @param lambda regularization value (-1) corresponds to additional operator input
 * @param batch_independent select if minimization is performed for each batch independently (for example useful for CG)
 * @param convergence_warn_limit warn if minimization is not converged (0 corresponds to no warnings)
 * @param conf pointer to configuration for iterative algorithm, NULL will create default conf using CG
 * @param rescale rescale the result with (1 + lambda)
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * [lambda:	ldims:	(1)]
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */

const struct nlop_s* mri_reg_pinv(int N, const long dims[N], bool share_pattern, float lambda, bool batch_independent, float convergence_warn_limit, iter_conf* conf, bool rescale)
{
	long idims[N];
	md_select_dims(N, 23ul, idims, dims);

	auto nlop_zf = nlop_mri_adjoint_create(N, dims, share_pattern);// in: kspace, coil, pattern; out: Atb
	auto nlop_norm_inv = mri_normal_inversion_create(N, dims, share_pattern, lambda, batch_independent, convergence_warn_limit, conf); // in: Atb, coil, pattern, [lambda]; out: A^+b

	auto nlop_pseudo_inv = nlop_chain2_swap_FF(nlop_zf, 0, nlop_norm_inv, 0); // in: kspace, coil, pattern, coil, pattern, [lambda]; out: A^+b
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 1, 3);
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 2, 3);// in: kspace, coil, pattern, [lambda]; out: A^+b

	if (rescale) {

		if (-1. != lambda) {

			nlop_pseudo_inv = nlop_chain2_FF(nlop_pseudo_inv, 0, nlop_from_linop_F(linop_scale_create(N, idims, 1 + lambda)), 0);
		} else {

			const struct nlop_s* scale = nlop_chain2_FF(nlop_tenmul_create(N, idims, idims, MD_SINGLETON_DIMS(N)), 0, nlop_zaxpbz_create(N, idims, 1., 1.), 1);
			scale = nlop_dup_F(scale, 0, 1);
			scale = nlop_reshape_in_F(scale, 1, 1, MD_SINGLETON_DIMS(1));

			nlop_pseudo_inv = nlop_chain2_swap_FF(nlop_pseudo_inv, 0, scale, 0);
			nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 3, 4);
		}
	}

	return nlop_pseudo_inv;
}


/**
 * Create an opertor computing the Tickhonov regularized pseudo-inverse of the MRI operator
 *
 * out = [(1 + lambda)](A^HA +l1)^-1 A^Hin
 * A = Pattern FFT Coils
 *
 * @param N # of dims (must be: 5)
 * @param dims dimensions [Nx, Ny, Nz, Nc, Nb]
 * @param share_pattern select if the same pattern is used for all elements in the batch
 * @param lambda regularization value (-1) corresponds to additional operator input
 * @param batch_independent select if minimization is performed for each batch independently (for example useful for CG)
 * @param convergence_warn_limit warn if minimization is not converged (0 corresponds to no warnings)
 * @param conf pointer to configuration for iterative algorithm, NULL will create default conf using CG
 * @param rescale rescale the result with (1 + lambda)
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * lambda:	ldims:	(1) input is ignored but present if -1 != lambda
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */

const struct nlop_s* mri_reg_pinv_with_lambda(int N, const long dims[N], bool share_pattern, float lambda, bool batch_independent, float convergence_warn_limit, iter_conf* conf, bool rescale)
{
	auto result = mri_reg_pinv(N, dims, share_pattern, lambda, batch_independent, convergence_warn_limit, conf, rescale);
	if (-1. != lambda)
		result = nlop_combine_FF(result, nlop_del_out_create(1, MD_SINGLETON_DIMS(1)));

	return result;
}