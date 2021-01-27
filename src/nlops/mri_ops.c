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


struct conf_mri_dims conf_nlop_mri_simple = {

	.coil_flags = ~(0ul),
	.image_flags = ~COIL_FLAG,
	.mask_flags = 0ul,
	.pattern_flags = FFT_FLAGS,
	.batch_flags = MD_BIT(4), 
	.fft_flags = FFT_FLAGS,

	.keep_mask_input = false,
	.keep_lambda_input = false,

	.lambda_fixed = -1,
	.iter_conf = NULL,

	.regrid = true,
};

/**
 * Returns: MRI forward operator (SENSE Operator)
 *
 * @param N
 * @param dims (Nx, Ny, Nz, Nc,  Nb)
 * @param conf can be NULL to fallback on nlop_mri_simple
 * 
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 *
 * Output tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	long cdims[N];
	long pdims[N];
	long idims[N];
	md_select_dims(N, conf->coil_flags, cdims, dims);
	md_select_dims(N, conf->pattern_flags, pdims, dims);
	md_select_dims(N, conf->image_flags, idims, dims);

	const struct nlop_s* result = nlop_tenmul_create(N, cdims, idims, cdims); //in: image, coil
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_fftc_create(N, dims, conf->fft_flags)), 0); //in: image, coil
	if (conf->regrid)
	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, dims, dims, pdims), 0); //in: image, coil, pattern
	else
		result = nlop_combine_FF(result, nlop_del_out_create(N, pdims));

	debug_printf(DP_DEBUG2, "mri forward created\n");
	return result;
}


/**
 * Returns: Adjoint MRI operator (SENSE Operator)
 *
 * @param N
 * @param dims (Nx, Ny, Nz, Nc,  Nb)
 * @param conf can be NULL to fallback on nlop_mri_simple
 * 
 *
 * for default dims:
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	long cdims[N];
	long pdims[N];
	long idims[N];
	md_select_dims(N, conf->coil_flags, cdims, dims);
	md_select_dims(N, conf->pattern_flags, pdims, dims);
	md_select_dims(N, conf->image_flags, idims, dims);

	const struct linop_s* lop = linop_ifftc_create(N, dims, conf->fft_flags);	
	const struct nlop_s* result = nlop_from_linop_F(lop);

	if (conf->regrid)
		result = nlop_chain2_FF(nlop_tenmul_create(N, dims, dims, pdims), 0, result, 0); //in: kspace, pattern
	else
		result = nlop_combine_FF(result, nlop_del_out_create(N, pdims));

	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, idims, cdims, cdims), 0); //in: kspace, pattern, coil
	result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, cdims)), 0, result, 2); //in: kspace, pattern, coil
	result = nlop_shift_input_F(result, 1, 2); //in: kspace, coil, pattern

	debug_printf(DP_DEBUG2, "mri adjoint created\n");

	return result;
}


struct mri_normal_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* idims;
	const long* cdims;
	const long* pdims;
	const long* kdims;
	const long* mdims;

	complex float* coil;
	complex float* pattern;

	const struct linop_s* lop_fft;

	bool regrid; //only for gradientstep
};

DEF_TYPEID(mri_normal_s);

static struct mri_normal_s* mri_normal_data_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	if (NULL == conf)
		conf = &conf_nlop_mri_simple;
		
	PTR_ALLOC(struct mri_normal_s, data);
	SET_TYPEID(mri_normal_s, data);

	PTR_ALLOC(long[N], idims);
	PTR_ALLOC(long[N], kdims);
	PTR_ALLOC(long[N], pdims);
	PTR_ALLOC(long[N], cdims);
	PTR_ALLOC(long[N], mdims);

	md_select_dims(N, conf->coil_flags, *cdims, dims);
	md_select_dims(N, conf->pattern_flags, *pdims, dims);
	md_select_dims(N, conf->image_flags, *idims, dims);
	md_select_dims(N, conf->mask_flags, *mdims, dims);
	md_copy_dims(N, *kdims, dims);

	data->N = N;
	data->idims = *PTR_PASS(idims);
	data->kdims = *PTR_PASS(kdims);
	data->pdims = *PTR_PASS(pdims);
	data->cdims = *PTR_PASS(cdims);
	data->mdims = *PTR_PASS(mdims);
	
	// will be initialized later, to transparently support GPU
	data->coil = NULL;
	data->pattern = NULL;

	data->lop_fft = linop_fftc_create(N, data->kdims, conf->fft_flags);

	data->regrid = conf->regrid;

	return PTR_PASS(data);
}

static void mri_normal_initialize(struct mri_normal_s* data, const complex float* arg)
{
	if (NULL == data->coil)
		data->coil = md_alloc_sameplace(data->N, data->kdims, CFL_SIZE, arg);

	if (NULL == data->pattern)
		data->pattern = md_alloc_sameplace(data->N, data->pdims, CFL_SIZE, arg);
}

static void mri_normal_lin(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_s, _data);

	complex float* coil_image = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, dst);
	md_ztenmul(d->N, d->kdims, coil_image, d->kdims, d->coil, d->idims, src);
	linop_forward_unchecked(d->lop_fft, coil_image, coil_image);
	md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->kdims, CFL_SIZE), coil_image, MD_STRIDES(d->N, d->pdims, CFL_SIZE), d->pattern);
	linop_adjoint_unchecked(d->lop_fft, coil_image, coil_image);
	md_ztenmulc(d->N, d->idims, dst, d->kdims, coil_image, d->kdims, d->coil);
	md_free(coil_image);
}

static void mri_normal_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_s, _data);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];
	const complex float* mask = args[4];

	bool der = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
	mri_normal_initialize(d, dst);

	if (0 != md_nontriv_dims(d->N, d->mdims))
		md_zmul2(d->N, d->cdims, MD_STRIDES(d->N, d->cdims, CFL_SIZE), d->coil, MD_STRIDES(d->N, d->cdims, CFL_SIZE), coil, MD_STRIDES(d->N, d->mdims, CFL_SIZE), mask);
	else
		md_copy(d->N, d->cdims, d->coil, coil, CFL_SIZE);

	md_copy(d->N, d->pdims, d->pattern, pattern, CFL_SIZE);

	mri_normal_lin(_data, 0, 0, dst, image);

	if (!der) {

		md_free(d->coil);
		d->coil = NULL;

		md_free(d->pattern);
		d->pattern = NULL;
	}
}

static void mri_normal_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_s, _data);

	md_free(d->pattern);
	md_free(d->coil);

	linop_free(d->lop_fft);

	xfree(d->idims);
	xfree(d->pdims);
	xfree(d->kdims);
	xfree(d->cdims);
	xfree(d->mdims);

	xfree(d);
}

/**
 * Returns: MRI normal operator
 *
 * @param N
 * @param dims (Nx, Ny, Nz, Nc,  Nb)
 * @param conf can be NULL to fallback on nlop_mri_simple
 * 
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 * [mask:	mdims:	(Nx, Ny, Nz, 1,  1 )]
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_normal_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	auto data = mri_normal_data_create(N, dims, conf);
	
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->idims);

	long nl_idims[4][N];
	md_copy_dims(N, nl_idims[0], data->idims);
	md_copy_dims(N, nl_idims[1], data->cdims);
	md_copy_dims(N, nl_idims[2], data->pdims);
	md_copy_dims(N, nl_idims[3], data->mdims);

	operator_property_flags_t props[4][1] = { { 0 }, { 0 }, { 0 }, { 0 } };

	const struct nlop_s* result = nlop_generic_with_props_create(	
			1, N, nl_odims, 4, N, nl_idims, CAST_UP(data),
			mri_normal_fun,
			(nlop_der_fun_t[4][1]){ { mri_normal_lin }, { NULL }, { NULL }, { NULL } },
			(nlop_der_fun_t[4][1]){ { mri_normal_lin }, { NULL }, { NULL }, { NULL } },
			NULL, NULL, mri_normal_del, NULL, props, NULL
		);

	if ((0 == conf->mask_flags) && (!conf->keep_mask_input)) {

		complex float mask = 1;
		result = nlop_set_input_const_F(result, 3, N, MD_SINGLETON_DIMS(N), true, &mask);
	}

	return result;
}

/**
 * Returns: MRI normal operator
 *
 * @param N
 * @param dims (Nx, Ny, Nz, Nc,  Nb)
 * @param conf can be NULL to fallback on nlop_mri_simple
 * 
 *
 * for default dims:
 *
 * Input tensors:
 * image:	kdims: 	(Nx, Ny, Nz, 1,  Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_normal_generic_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	auto result = nlop_mri_forward_create(N, dims, conf);
	result = nlop_chain2_swap_FF(result, 0, nlop_mri_adjoint_create(N, dims, conf), 0);

	result = nlop_dup_F(result, 1, 3);
	result = nlop_dup_F(result, 2, 3);

	debug_printf(DP_DEBUG2, "mri normal created\n");

	return result;
}

static void mri_gradient_step_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_s, _data);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* kspace = args[2];
	const complex float* coil = args[3];
	const complex float* pattern = args[4];
	const complex float* mask = args[5];

	bool der = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
	mri_normal_initialize(d, dst);

	if (0 != md_nontriv_dims(d->N, d->mdims))
		md_zmul2(d->N, d->cdims, MD_STRIDES(d->N, d->cdims, CFL_SIZE), d->coil, MD_STRIDES(d->N, d->cdims, CFL_SIZE), coil, MD_STRIDES(d->N, d->mdims, CFL_SIZE), mask);
	else
		md_copy(d->N, d->cdims, d->coil, coil, CFL_SIZE);

	md_copy(d->N, d->pdims, d->pattern, pattern, CFL_SIZE);

	mri_normal_lin(_data, 0, 0, dst, image);

	complex float* tmp = md_alloc_sameplace(d->N, d->kdims, CFL_SIZE, kspace);
	
	if (d->regrid) {

		md_zmul2(d->N, d->kdims, MD_STRIDES(d->N, d->kdims, CFL_SIZE), tmp, MD_STRIDES(d->N, d->kdims, CFL_SIZE), kspace, MD_STRIDES(d->N, d->pdims, CFL_SIZE), d->pattern);
		kspace = tmp;
	}

	linop_adjoint_unchecked(d->lop_fft, tmp, kspace);
	md_zsmul(d->N, d->kdims, tmp, tmp, -1.);
	md_zfmacc2(d->N, d->kdims, MD_STRIDES(d->N, d->idims, CFL_SIZE), dst, MD_STRIDES(d->N, d->kdims, CFL_SIZE), tmp, MD_STRIDES(d->N, d->cdims, CFL_SIZE), d->coil);
	md_free(tmp);

	if (!der) {

		md_free(d->coil);
		d->coil = NULL;

		md_free(d->pattern);
		d->pattern = NULL;
	}
}

/**
 * Returns operator computing gradient step
 * out = AH(A image - kspace)
 * 
 * In non-cartesian case, the kspace is assumed to be gridded
 *
 * @param N
 * @param dims (Nx, Ny, Nz, Nc,  Nb)
 * @param conf can be NULL to fallback on nlop_mri_simple
 * 
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 * [mask:	mdims:	(1,  1,  1,  1,  1 )]
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */

 const struct nlop_s* nlop_mri_gradient_step_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	auto data = mri_normal_data_create(N, dims, conf);
	
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->idims);

	long nl_idims[5][N];
	md_copy_dims(N, nl_idims[0], data->idims);
	md_copy_dims(N, nl_idims[1], data->kdims);
	md_copy_dims(N, nl_idims[2], data->cdims);
	md_copy_dims(N, nl_idims[3], data->pdims);
	md_copy_dims(N, nl_idims[4], data->mdims);

	operator_property_flags_t props[5][1] = { { 0 }, { 0 }, { 0 }, { 0 }, { 0 } };

	const struct nlop_s* result = nlop_generic_with_props_create(	
			1, N, nl_odims, 5, N, nl_idims, CAST_UP(data),
			mri_gradient_step_fun,
			(nlop_der_fun_t[5][1]){ { mri_normal_lin }, { NULL }, { NULL }, { NULL }, { NULL } },
			(nlop_der_fun_t[5][1]){ { mri_normal_lin }, { NULL }, { NULL }, { NULL }, { NULL } },
			NULL, NULL, mri_normal_del, NULL, props, NULL
		);

	if ((0 == conf->mask_flags) && (!conf->keep_mask_input)) {

		complex float mask = 1;
		result = nlop_set_input_const_F(result, 4, N, MD_SINGLETON_DIMS(N), true, &mask);
	}

	return result;
}

struct mri_normal_inversion_s {

	INTERFACE(nlop_data_t);

	unsigned long image_flags;
	unsigned long mask_flags;
	unsigned long pattern_flags;
	unsigned long batch_flags;
	unsigned long fft_flags;

	int N;

	const long* idims;
	const long* pdims;
	const long* cdims;
	const long* kdims;
	const long* mdims;

	const long* istrs;
	const long* pstrs;
	const long* cstrs;
	const long* kstrs;
	const long* mstrs;

	const long* bdims;

	const long* fftdims;
	const long* fftstrs;

	complex float* fftmod;
	complex float* coil;
	complex float* out;

	const struct operator_s** normal_op;

	struct iter_conjgrad_conf iter_conf;

	const struct linop_s* lop_fft;

	float lambda_fixed;
};

static void mri_normal_inversion_alloc(struct mri_normal_inversion_s* d, const void* ref)
{
	if (NULL == d->coil)
		d->coil = md_alloc_sameplace(d->N, d->cdims, CFL_SIZE, ref);
	
	if (NULL == d->fftmod) {

		complex float * tmp = md_alloc(d->N, d->fftdims, CFL_SIZE);
		md_zfill(d->N, d->fftdims, tmp, 1.);
		fftmod(d->N, d->fftdims, FFT_FLAGS, tmp, tmp);

		d->fftmod = md_alloc_sameplace(d->N, d->fftdims, CFL_SIZE, ref);
		md_resize_center(d->N, d->fftdims, d->fftmod , d->fftdims, tmp, CFL_SIZE);
		md_free(tmp);
	}

	long kdims_normal[d->N];
	md_select_dims(d->N, ~(d->batch_flags), kdims_normal, d->kdims);

	if (NULL == d->lop_fft)
		d->lop_fft = linop_fft_create(d->N, kdims_normal, d->fft_flags);
}

static void mri_normal_inversion_set_normal_ops(struct mri_normal_inversion_s* d, const complex float* coil, const complex float* pattern, const complex float* mask, const complex float* lptr)
{
	mri_normal_inversion_alloc(d, coil);

	long pdims_normal[d->N];
	long cdims_normal[d->N];
	long kdims_normal[d->N];

	md_select_dims(d->N, ~(d->batch_flags), pdims_normal, d->pdims);
	md_select_dims(d->N, ~(d->batch_flags), cdims_normal, d->cdims);
	md_select_dims(d->N, ~(d->batch_flags), kdims_normal, d->kdims);

	complex float lambda;
	md_copy(1, MAKE_ARRAY(1l), &lambda, lptr, CFL_SIZE);

	if ((0 != cimagf(lambda)) || (0 > crealf(lambda)))
		error("Lambda=%f+%fi is not non-negative real number!\n", crealf(lambda), cimagf(lambda));
	d->iter_conf.INTERFACE.alpha = crealf(lambda);

	md_zmul2(d->N, d->cdims, d->cstrs, d->coil, d->cstrs, coil, d->fftstrs, d->fftmod);
	if (NULL != mask)
		md_zmul2(d->N, d->cdims, d->cstrs, d->coil, d->cstrs, d->coil, d->mstrs, mask);

	complex float* tmp_pattern = md_alloc_sameplace(d->N, pdims_normal, CFL_SIZE, pattern);

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;
	
	do {
		if (NULL != d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)])
			operator_free(d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)]);
		
		auto linop_frw = linop_fmac_create(d->N, cdims_normal, 0, ~(d->image_flags), 0, &MD_ACCESS(d->N, d->cstrs, pos, d->coil));
		linop_frw = linop_chain_FF(linop_frw, linop_clone(d->lop_fft));
		
		md_zsmul(d->N, pdims_normal, tmp_pattern, &MD_ACCESS(d->N, d->pstrs, pos, pattern), 1. / md_calc_size(3, d->pdims));
		auto linop_pattern = linop_cdiag_create(d->N, kdims_normal, d->pattern_flags, tmp_pattern);

		d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)] = operator_chainN(3, (const struct operator_s **)MAKE_ARRAY(linop_frw->forward, linop_pattern->forward, linop_frw->adjoint));

		linop_free(linop_frw);
		linop_free(linop_pattern);

	} while (md_next(d->N, d->bdims, ~(0ul), pos));

	md_free(tmp_pattern);
}

DEF_TYPEID(mri_normal_inversion_s);

static void mri_normal_inversion(const struct mri_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	assert(NULL != d->normal_op);

	long idims_normal[d->N];
	md_select_dims(d->N, ~(d->batch_flags), idims_normal, d->idims);

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;
	
	do {
		const struct operator_s* normal_op = d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)];
		md_clear(d->N, idims_normal, &MD_ACCESS(d->N, d->istrs, pos, dst), CFL_SIZE);

		iter2_conjgrad(	CAST_UP(&(d->iter_conf)), normal_op,
				0, NULL, NULL, NULL, NULL,
				2 * md_calc_size(d->N, idims_normal),
				(float*)&MD_ACCESS(d->N, d->istrs, pos, dst),
				(const float*)&MD_ACCESS(d->N, d->istrs, pos, src),
				NULL);

	} while (md_next(d->N, d->bdims, ~(0ul), pos));
}

static void mri_normal(const struct mri_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	assert(NULL != d->normal_op);

	long idims_normal[d->N];
	md_select_dims(d->N, ~(d->batch_flags), idims_normal, d->idims);

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;
	
	do {
		const struct operator_s* normal_op = d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)];
		operator_apply(normal_op, d->N, idims_normal, &MD_ACCESS(d->N, d->istrs, pos, dst), d->N, idims_normal, &MD_ACCESS(d->N, d->istrs, pos, src));

	} while (md_next(d->N, d->bdims, ~(0ul), pos));
}

static void mri_normal_inversion_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, src);
}

static void mri_normal_inversion_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, src);
}

static void mri_normal_inversion_der_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, d->out);

	md_zmul2(d->N, d->idims, MD_STRIDES(d->N, d->idims, CFL_SIZE), dst, MD_STRIDES(d->N, d->idims, CFL_SIZE), dst, MD_SINGLETON_STRS(d->N), src);
	md_zsmul(d->N, d->idims, dst, dst, -1);

}

static void mri_normal_inversion_adj_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	assert(NULL != d->normal_op);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);

	mri_normal_inversion(d, tmp, d->out);

	md_zconj(d->N, d->idims, tmp, tmp);
	md_ztenmul(d->N, MD_SINGLETON_DIMS(d->N), dst, d->idims, src, d->idims, tmp);
	md_free(tmp);

	md_zsmul(d->N, MD_SINGLETON_DIMS(d->N), dst, dst, -1);
	md_zreal(d->N, MD_SINGLETON_DIMS(d->N), dst, dst);
}

static void mri_free_normal_ops(struct mri_normal_inversion_s* d)
{
	md_free(d->coil);
	d->coil = NULL;

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;

	do {
		if (NULL != d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)])
			operator_free(d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)]);

		d->normal_op[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)] = NULL;

	} while (md_next(d->N, d->bdims, ~(0ul), pos));
}

static void mri_normal_inversion_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(6 == Narg);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];
	const complex float* mask = args[4];
	const complex float* lptr = args[5];

	bool der_in = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
	bool der_lam = !op_options_is_set_io(_data->options, 0, 4, OP_APP_NO_DER);
	der_lam = der_lam && (-1 == d->lambda_fixed);

	mri_normal_inversion_set_normal_ops(d, coil, pattern, mask, lptr);

	mri_normal_inversion(d, dst, image);

	if (der_lam) {

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

static void mri_normal_inversion_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_free_normal_ops(d);
	xfree(d->normal_op);

	xfree(d->idims);
	xfree(d->pdims);
	xfree(d->cdims);
	xfree(d->kdims);
	xfree(d->mdims);

	xfree(d->istrs);
	xfree(d->pstrs);
	xfree(d->cstrs);
	xfree(d->kstrs);
	xfree(d->mstrs);

	xfree(d->bdims);
	xfree(d->fftdims);
	xfree(d->fftstrs);

	md_free(d->out);
	md_free(d->fftmod);

	linop_free(d->lop_fft);

	xfree(d);
}


static struct mri_normal_inversion_s* mri_normal_inversion_data_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	if (NULL == conf)
		conf = &conf_nlop_mri_simple;
		
	PTR_ALLOC(struct mri_normal_inversion_s, data);
	SET_TYPEID(mri_normal_inversion_s, data);

	data->image_flags = conf->image_flags;
	data->mask_flags = conf->mask_flags;
	data->pattern_flags = conf->pattern_flags;
	data->batch_flags = conf->batch_flags;
	data->fft_flags = conf->fft_flags;

	// batch dims must be outer most dims
	bool batch = false;
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(data->batch_flags, i))
			batch = true;
		else
			assert(!batch || (1 == dims[i]));
	}

	data->N = N;

	PTR_ALLOC(long[N], idims);
	PTR_ALLOC(long[N], cdims);
	PTR_ALLOC(long[N], kdims);
	PTR_ALLOC(long[N], pdims);
	PTR_ALLOC(long[N], mdims);
	PTR_ALLOC(long[N], fftdims);

	PTR_ALLOC(long[N], bdims);

	md_select_dims(N, data->image_flags, *idims, dims);
	md_copy_dims(N, *cdims, dims);
	md_copy_dims(N, *kdims, dims);
	md_select_dims(N, data->pattern_flags, *pdims, dims);
	md_select_dims(N, data->mask_flags, *mdims, dims);
	md_select_dims(N, data->batch_flags, *bdims, dims);
	md_select_dims(N, data->fft_flags, *fftdims, dims);

	data->idims = *PTR_PASS(idims);
	data->pdims = *PTR_PASS(pdims);
	data->cdims = *PTR_PASS(cdims);
	data->kdims = *PTR_PASS(kdims);
	data->mdims = *PTR_PASS(mdims);
	data->bdims = *PTR_PASS(bdims);
	data->fftdims = *PTR_PASS(fftdims);

	PTR_ALLOC(long[N], istrs);
	PTR_ALLOC(long[N], cstrs);
	PTR_ALLOC(long[N], kstrs);
	PTR_ALLOC(long[N], pstrs);
	PTR_ALLOC(long[N], mstrs);
	PTR_ALLOC(long[N], fftstrs);

	md_calc_strides(N, *istrs, data->idims, CFL_SIZE);
	md_calc_strides(N, *cstrs, data->cdims, CFL_SIZE);
	md_calc_strides(N, *kstrs, data->kdims, CFL_SIZE);
	md_calc_strides(N, *pstrs, data->pdims, CFL_SIZE);
	md_calc_strides(N, *mstrs, data->mdims, CFL_SIZE);
	md_calc_strides(N, *fftstrs, data->fftdims, CFL_SIZE);

	data->istrs = *PTR_PASS(istrs);
	data->pstrs = *PTR_PASS(pstrs);
	data->cstrs = *PTR_PASS(cstrs);
	data->kstrs = *PTR_PASS(kstrs);
	data->mstrs = *PTR_PASS(mstrs);
	data->fftstrs = *PTR_PASS(fftstrs);


	// will be initialized later, to transparently support GPU
	data->out= NULL;
	data->coil = NULL;
	data->fftmod = NULL;

	data->lambda_fixed = conf->lambda_fixed;

	PTR_ALLOC(const struct operator_s*[md_calc_size(N, data->bdims)], normalops);
	for (int i = 0; i < md_calc_size(N, data->bdims); i++)
		(*normalops)[i] = NULL;
	data->normal_op = *PTR_PASS(normalops);

	data->lop_fft = NULL;

	if (NULL == conf->iter_conf) {

		data->iter_conf = iter_conjgrad_defaults;
		data->iter_conf.l2lambda = 1.;
		data->iter_conf.maxiter = 50;
	} else {

		data->iter_conf = *(conf->iter_conf);
	}

	return PTR_PASS(data);
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
 * [mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * [lambda:	ldims:	(1)]
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */

const struct nlop_s* mri_normal_inversion_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	auto data = mri_normal_inversion_data_create(N, dims, conf);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->idims);

	long nl_idims[5][N];
	md_copy_dims(N, nl_idims[0], data->idims);
	md_copy_dims(N, nl_idims[1], data->cdims);
	md_copy_dims(N, nl_idims[2], data->pdims);
	md_copy_dims(N, nl_idims[3], data->mdims);
	md_singleton_dims(N, nl_idims[4]);

	operator_property_flags_t props[5][1] = { { MD_BIT(OP_PROP_C_LIN) }, { 0 }, { 0 }, { 0 }, { 0 } };

	const struct nlop_s* result = nlop_generic_with_props_create(	1, N, nl_odims, 5, N, nl_idims, CAST_UP(data),
									mri_normal_inversion_fun,
									(nlop_der_fun_t[5][1]){ { mri_normal_inversion_der }, { NULL }, { NULL }, { NULL }, { mri_normal_inversion_der_lambda } },
									(nlop_der_fun_t[5][1]){ { mri_normal_inversion_adj }, { NULL }, { NULL }, { NULL }, { mri_normal_inversion_adj_lambda } },
									NULL, NULL, mri_normal_inversion_del, NULL, props, NULL
								);

	if ((-1. != conf->lambda_fixed) && (!conf->keep_lambda_input)) {

		complex float lambdac = conf->lambda_fixed;
		result = nlop_set_input_const_F(result, 4, N, MD_SINGLETON_DIMS(N), true, &lambdac);
	} else {

		result = nlop_reshape_in_F(result, 4, 1, MD_SINGLETON_DIMS(1));
	}

	if ((0 == conf->mask_flags) && (!conf->keep_mask_input)) {

		complex float mask = 1;
		result = nlop_set_input_const_F(result, 3, N, MD_SINGLETON_DIMS(N), true, &mask);
	}

	return result;
}



static void mri_reg_proj_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	assert(NULL != d->normal_op);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);

	mri_normal(d, tmp, src);
	mri_normal_inversion(d, dst, tmp);

	md_free(tmp);

}

static void mri_reg_proj_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(NULL != d->normal_op);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);

	mri_normal_inversion(d, tmp, src);
	mri_normal(d, dst, tmp);

	md_free(tmp);

}

static void mri_reg_proj_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(5 == Narg);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];
	const complex float* mask = args[4];
	const complex float* lptr = args[5];

	bool der_in = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
	bool der_lam = !op_options_is_set_io(_data->options, 0, 4, OP_APP_NO_DER);
	der_lam = der_lam && (-1 == d->lambda_fixed);

	mri_normal_inversion_set_normal_ops(d, coil, pattern, mask, lptr);

	complex float* tmp = md_alloc_sameplace(d->N, d->idims, CFL_SIZE, dst);
	mri_normal(d, tmp, image);
	mri_normal_inversion(d, dst, tmp);

	md_free(tmp);

	if (der_lam) {

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
 * @param N # of dims
 * @param dims dimensions [Nx, Ny, Nz, Nc, Nb]
 * @param conf 
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
const struct nlop_s* mri_reg_proj_ker_create(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	auto data = mri_normal_inversion_data_create(N, dims, conf);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->idims);

	long nl_idims[5][N];
	md_copy_dims(N, nl_idims[0], data->idims);
	md_copy_dims(N, nl_idims[1], data->cdims);
	md_copy_dims(N, nl_idims[2], data->pdims);
	md_copy_dims(N, nl_idims[3], data->mdims);
	md_singleton_dims(N, nl_idims[4]);

	operator_property_flags_t props[5][1] = { { MD_BIT(OP_PROP_C_LIN) }, { 0 }, { 0 }, { 0 }, { 0 } };

	const struct nlop_s* result = nlop_generic_with_props_create(	1, N, nl_odims, 5, N, nl_idims, CAST_UP(data),
									mri_reg_proj_fun,
									(nlop_der_fun_t[5][1]){ { mri_reg_proj_der }, { NULL }, { NULL }, { NULL }, { mri_normal_inversion_der_lambda } },
									(nlop_der_fun_t[5][1]){ { mri_reg_proj_adj }, { NULL }, { NULL }, { NULL }, { mri_normal_inversion_adj_lambda } },
									NULL, NULL, mri_normal_inversion_del, NULL, props, NULL
								);

	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, nl_idims[0], -1., 1.), 0);
	result = nlop_dup_F(result, 0, 1);

	if ((-1. != conf->lambda_fixed) && (!conf->keep_lambda_input)) {

		complex float lambdac = conf->lambda_fixed;
		result = nlop_set_input_const_F(result, 4, N, MD_SINGLETON_DIMS(N), true, &lambdac);
	} else {

		result = nlop_reshape_in_F(result, 4, 1, MD_SINGLETON_DIMS(1));
	}

	if ((0 == conf->mask_flags) && (!conf->keep_mask_input)) {

		complex float mask = 1;
		result = nlop_set_input_const_F(result, 3, N, MD_SINGLETON_DIMS(N), true, &mask);
	}

	return result;

}

/**
 * Create an operator computing the Tickhonov regularized pseudo-inverse of the MRI operator
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
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 * [mask	mdims:	(1,  1,  1,  1,  1 )]
 * [lambda:	ldims:	(1)]
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */

const struct nlop_s* mri_reg_pinv(int N, const long dims[N], const struct conf_mri_dims* conf)
{
	auto nlop_zf = nlop_mri_adjoint_create(N, dims, conf);// in: kspace, coil, pattern; out: Atb
	auto nlop_norm_inv = mri_normal_inversion_create(N, dims, conf); // in: Atb, coil, pattern, [mask], [lambda]; out: A^+b

	auto nlop_pseudo_inv = nlop_chain2_swap_FF(nlop_zf, 0, nlop_norm_inv, 0); // in: kspace, coil, pattern, coil, pattern, [mask], [lambda]; out: A^+b
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 1, 3);
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 2, 3);// in: kspace, coil, pattern, [mask], [lambda]; out: A^+b

	return nlop_pseudo_inv;
}
