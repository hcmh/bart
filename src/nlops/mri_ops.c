#include <math.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "noncart/nufft.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/rand.h"
#include "num/fft.h"

#include "iter/misc.h"
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

#include "nlops/sense_normal_ops.h"
#include "mri_ops.h"


struct config_nlop_mri_s conf_nlop_mri_simple = {

	.coil_flags = FFT_FLAGS | COIL_FLAG | MAPS_FLAG | BATCH_FLAG,
	.image_flags = FFT_FLAGS | MAPS_FLAG | COEFF_FLAG | BATCH_FLAG,
	.pattern_flags = FFT_FLAGS | BATCH_FLAG,
	.batch_flags = BATCH_FLAG,
	.fft_flags = FFT_FLAGS,
	.coil_image_flags = FFT_FLAGS | COIL_FLAG | COEFF_FLAG | BATCH_FLAG,

	.noncart = false,
	.gridded = false,
	.basis = false,

	.nufft_conf = NULL,
};

static bool test_idims_compatible(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	long tdims[N];
	md_select_dims(N, conf->image_flags, tdims, dims);
	return md_check_equal_dims(N, tdims, idims, ~(conf->fft_flags));
}








/**
 * Returns: MRI forward operator (SENSE Operator)
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  ..., 1 )
 *
 * Output tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, ..., Nb)
 */
const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	long cdims[N];
	long pdims[N];

	md_select_dims(N, conf->coil_flags, cdims, dims);
	md_select_dims(N, conf->pattern_flags, pdims, dims);

	for (int i = 0; i < N; i++)
		cdims[i] = MD_IS_SET(conf->fft_flags & conf->coil_flags, i) ? idims[i] : cdims[i];

	const struct nlop_s* result = nlop_tenmul_create(N, cdims, idims, cdims); //in: image, coil

	const struct linop_s* lop = linop_fftc_create(N, dims, conf->fft_flags);
	if (!md_check_equal_dims(N, cdims, dims, ~0))
		lop = linop_chain_FF(linop_resize_center_create(N, dims, cdims), lop);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(lop), 0); //in: image, coil

	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, dims, dims, pdims), 0); //in: image, coil, pattern

	debug_printf(DP_DEBUG2, "mri forward created\n");
	return result;
}








/**
 * Returns: Adjoint MRI operator (SENSE Operator)
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  ..., 1)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1, ..., Nb)
 */
const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{

	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	long cdims[N];
	long pdims[N];

	md_select_dims(N, conf->coil_flags, cdims, dims);
	md_select_dims(N, conf->pattern_flags, pdims, dims);

	for (int i = 0; i < N; i++)
		cdims[i] = MD_IS_SET(conf->fft_flags & conf->coil_flags, i) ? idims[i] : cdims[i];

	const struct linop_s* lop = linop_ifftc_create(N, dims, conf->fft_flags);
	if (!md_check_equal_dims(N, cdims, dims, ~0))
		lop = linop_chain_FF(lop, linop_resize_center_create(N, cdims, dims));

	const struct nlop_s* result = nlop_from_linop_F(lop);

	if (!conf->gridded)
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

	struct sense_normal_ops_s* normal_ops;

	complex float* coil;
};

DEF_TYPEID(mri_normal_s);

static struct mri_normal_s* mri_normal_data_create(int N, const long max_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf)
{
	PTR_ALLOC(struct mri_normal_s, data);
	SET_TYPEID(mri_normal_s, data);

	long cim_dims[N];
	md_select_dims(N, conf->coil_flags | conf->image_flags, cim_dims, max_dims);

	data->normal_ops = sense_normal_create(N, cim_dims, ND, psf_dims, conf);
	data->coil = NULL;

	return PTR_PASS(data);
}

static void mri_normal_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_s, _data);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];

	bool der = nlop_der_requested(_data, 0, 0);

	if (der) {

		if (NULL == d->coil)
			d->coil = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->col_dims, CFL_SIZE, coil);
		md_copy(d->normal_ops->N, d->normal_ops->col_dims, d->coil, coil, CFL_SIZE);
	} else {

		md_free(d->coil);
		d->coil = NULL;
	}

	struct sense_normal_ops_s* nops = d->normal_ops;

	sense_normal_update_ops(nops, nops->N, nops->col_dims, der ? d->coil : coil, nops->ND, nops->psf_dims, pattern);
	sense_apply_normal_ops(nops, nops->N, nops->img_dims, dst, image);
}

static void mri_normal_deradj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_s, _data);

	assert(NULL != d->coil);
	struct sense_normal_ops_s* nops = d->normal_ops;

	sense_apply_normal_ops(nops, nops->N, nops->img_dims, dst, src);
}

static void mri_normal_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_s, _data);

	sense_normal_free(d->normal_ops);

	md_free(d->coil);
	xfree(d);
}

/**
 * Returns: MRI normal operator
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  ..., 1 )
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_normal_create(int N, const long max_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf)
{
	auto data = mri_normal_data_create(N, max_dims, ND, psf_dims, conf);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->normal_ops->img_dims);

	long nl_idims[3][ND];

	for (int i = 0; i < 3; i++)
		md_singleton_dims(ND, nl_idims[i]);

	md_copy_dims(N, nl_idims[0], data->normal_ops->img_dims);
	md_copy_dims(N, nl_idims[1], data->normal_ops->col_dims);
	md_copy_dims(ND, nl_idims[2], data->normal_ops->psf_dims);

	const struct nlop_s* result = nlop_generic_create(
			1, N, nl_odims, 3, ND, nl_idims, CAST_UP(data),
			mri_normal_fun,
			(nlop_der_fun_t[3][1]){ { mri_normal_deradj }, { NULL }, { NULL } },
			(nlop_der_fun_t[3][1]){ { mri_normal_deradj }, { NULL }, { NULL } },
			NULL, NULL, mri_normal_del
		);

	result = nlop_reshape_in_F(result, 0, N, nl_idims[0]);
	result = nlop_reshape_in_F(result, 1, N, nl_idims[1]);

	return result;
}


struct mri_normal_inversion_s {

	INTERFACE(nlop_data_t);

	int N_batch;
	struct sense_normal_ops_s* normal_ops;
	struct iter_conjgrad_conf* iter_conf;

	int N;
	const long* lam_dims;

	complex float* lambda;
	complex float* coil;
	complex float* out;

	bool store_tmp_adj;
	complex float* dout;	//Adjoint lambda and adjoint in
	complex float* AhAdout;	//share same intermediate result
};

static void mri_normal_inversion_set_normal_ops(struct mri_normal_inversion_s* d, const complex float* coil, const complex float* pattern, const complex float* lptr)
{
	if (NULL == d->lambda)
		d->lambda = md_alloc_sameplace(d->N, d->lam_dims, CFL_SIZE, coil);
	md_zreal(d->N, d->lam_dims, d->lambda, lptr);

	if (NULL == d->coil)
		d->coil = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->col_dims, CFL_SIZE, coil);
	md_copy(d->normal_ops->N, d->normal_ops->col_dims, d->coil, coil, CFL_SIZE);

	struct sense_normal_ops_s* nops = d->normal_ops;
	sense_normal_update_ops(nops, nops->N, nops->col_dims, d->coil, nops->ND, nops->psf_dims, pattern);
}

DEF_TYPEID(mri_normal_inversion_s);

static void mri_normal_inversion(const struct mri_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	struct sense_normal_ops_s* nops = d->normal_ops;

	sense_apply_normal_inv(nops, d->iter_conf, nops->N, nops->img_dims, dst, src, d->lam_dims, d->lambda);
}

static void mri_normal_inversion_cache(struct mri_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	struct sense_normal_ops_s* nops = d->normal_ops;

	if (!d->store_tmp_adj) {

		mri_normal_inversion(d, dst, src);
		return;
	}

	if (NULL == d->dout) {

		d->dout = md_alloc_sameplace(nops->N, nops->img_dims, CFL_SIZE, dst);
		md_clear(nops->N, nops->img_dims, d->dout, CFL_SIZE);
	}

	if (NULL == d->AhAdout) {

		d->AhAdout = md_alloc_sameplace(d->normal_ops->N, nops->img_dims, CFL_SIZE, dst);
		md_clear(nops->N, nops->img_dims, d->AhAdout, CFL_SIZE);
	}

	if (0 != md_zrmse(nops->N, nops->img_dims, d->dout, src)) {

		md_copy(nops->N, nops->img_dims, d->dout, src, CFL_SIZE);
		mri_normal_inversion(d, d->AhAdout, d->dout);
	}

	md_copy(nops->N, nops->img_dims, dst, d->AhAdout, CFL_SIZE);
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
	mri_normal_inversion_cache(d, dst, src);
}

static void mri_normal_inversion_der_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, d->out);

	struct sense_normal_ops_s* nops = d->normal_ops;

	long istrs[nops->N];
	long lstrs[nops->N];

	md_calc_strides(nops->N, istrs, nops->img_dims, CFL_SIZE);
	md_calc_strides(nops->N, lstrs, d->lam_dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(d->N, d->lam_dims, CFL_SIZE, src);
	md_zreal(d->N, d->lam_dims, tmp, src);

	md_zmul2(nops->N, nops->img_dims, istrs, dst, istrs, dst, lstrs, tmp);
	md_zsmul(nops->N, nops->img_dims, dst, dst, -1);

	md_free(tmp);
}

static void mri_normal_inversion_adj_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	struct sense_normal_ops_s* nops = d->normal_ops;

	complex float* tmp = md_alloc_sameplace(nops->N, nops->img_dims, CFL_SIZE, dst);
	mri_normal_inversion_cache(d, tmp, src);

	md_ztenmulc(nops->N, d->lam_dims, dst, nops->img_dims, d->out, nops->img_dims, tmp);
	md_free(tmp);

	md_zsmul(nops->N, d->lam_dims, dst, dst, -1);
	md_zreal(nops->N, d->lam_dims, dst, dst);
}


static void mri_normal_inversion_clear_der(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	md_free(d->lambda);
	md_free(d->coil);
	md_free(d->out);
	md_free(d->dout);
	md_free(d->AhAdout);

	d->lambda = NULL;
	d->coil = NULL;
	d->out = NULL;
	d->dout = NULL;
	d->AhAdout = NULL;
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

	bool der_in = nlop_der_requested(_data, 0, 0);
	bool der_lam = nlop_der_requested(_data, 3, 0);

	mri_normal_inversion_clear_der(_data);
	mri_normal_inversion_set_normal_ops(d, coil, pattern, lptr);
	mri_normal_inversion(d, dst, image);

	int N = d->normal_ops->N;
	const long* idims = d->normal_ops->img_dims;

	if (der_lam) {

		d->out = md_alloc_sameplace(N, idims, CFL_SIZE, dst);
		md_copy(N, idims, d->out, dst, CFL_SIZE);
	} else {

		if (!der_in) {

			md_free(d->coil);
			d->coil = NULL;

			md_free(d->lambda);
			d->lambda = NULL;
		}
	}

	d->store_tmp_adj = der_lam && der_in;
}

static void mri_normal_inversion_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	sense_normal_free(d->normal_ops);
	mri_normal_inversion_clear_der(_data);

	xfree(d->iter_conf);
	xfree(d->lam_dims);

	xfree(d);
}


static struct mri_normal_inversion_s* mri_normal_inversion_data_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	PTR_ALLOC(struct mri_normal_inversion_s, data);
	SET_TYPEID(mri_normal_inversion_s, data);

	data->normal_ops = sense_normal_create(N, max_dims, ND, psf_dims, conf);
	data->coil = NULL;

	// will be initialized later, to transparently support GPU
	data->out= NULL;
	data->coil = NULL;

	data->dout = NULL;
	data->AhAdout = NULL;
	data->store_tmp_adj = false;

	data->N_batch = md_calc_size(data->normal_ops->N, data->normal_ops->bat_dims);
	data->iter_conf = TYPE_ALLOC(struct iter_conjgrad_conf);

	data->N = N;
	long* tlam_dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, tlam_dims, lam_dims);
	data->lam_dims = tlam_dims;
	data->lambda = NULL;

	if (NULL == iter_conf) {

		*(data->iter_conf) = iter_conjgrad_defaults;
		data->iter_conf->l2lambda = 0.;
		data->iter_conf->maxiter = 50;
	} else {

		*(data->iter_conf) = *iter_conf;
		assert(data->iter_conf->l2lambda == 0.);
	}

	return PTR_PASS(data);
}

/**
 * Create an operator applying the inverse normal mri forward model on its input
 *
 * out = (A^HA +l1)^-1 in
 * A = Pattern FFT Coils
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 * @param lammbda_fixed if -1, lambda is an input of the nlop
 *
 * Input tensors:
 * image:	idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, ..., 1 / Nb)
 * lambda:	ldims:	( 1,  1,  1,  1, ..., Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1,  Nb)
 */
const struct nlop_s* nlop_mri_normal_inv_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	auto data = mri_normal_inversion_data_create(N, max_dims, lam_dims, ND, psf_dims, conf, iter_conf);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->normal_ops->img_dims);

	long nl_idims[4][ND];

	for (int i = 0; i < 4; i++)
		md_singleton_dims(ND, nl_idims[i]);

	md_copy_dims(N, nl_idims[0], data->normal_ops->img_dims);
	md_copy_dims(N, nl_idims[1], data->normal_ops->col_dims);
	md_copy_dims(ND, nl_idims[2], data->normal_ops->psf_dims);
	md_copy_dims(N, nl_idims[3], lam_dims);

	const struct nlop_s* result = nlop_generic_managed_create(	1, N, nl_odims, 4, ND, nl_idims, CAST_UP(data),
									mri_normal_inversion_fun,
									(nlop_der_fun_t[4][1]){ { mri_normal_inversion_der }, { NULL }, { NULL }, { mri_normal_inversion_der_lambda } },
									(nlop_der_fun_t[4][1]){ { mri_normal_inversion_adj }, { NULL }, { NULL }, { mri_normal_inversion_adj_lambda } },
									NULL, NULL, mri_normal_inversion_del, NULL, NULL);

	result = nlop_reshape_in_F(result, 0, N, nl_idims[0]);
	result = nlop_reshape_in_F(result, 1, N, nl_idims[1]);
	result = nlop_reshape_in_F(result, 3, N, nl_idims[3]);

	return result;
}

/**
 * Create an operator minimizing the following functional
 *
 * out = argmin 0.5 ||Ax-y||_2^2 + 0.5 ||sqrt{lambda} (x-x_0)||_2^2
 * A = Pattern FFT Coils
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 * @param lammbda_fixed if -1, lambda is an input of the nlop
 *
 * Input tensors:
 * x0:		idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * adjoint:	idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, ..., 1 / Nb)
 * lambda:	ldims:	( 1,  1,  1,  1, ..., Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1, ..., Nb)
 */
const struct nlop_s* nlop_mri_dc_prox_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	auto result = nlop_mri_normal_inv_create(N, max_dims, lam_dims, ND, psf_dims, conf, iter_conf);

	long img_dims[N];
	md_copy_dims(N, img_dims, nlop_generic_codomain(result, 0)->dims);

	result = nlop_chain2_swap_FF(nlop_zaxpbz_create(N, img_dims, 1., 1.), 0, result, 0); //in: lambda*x0, AHy, coil, pattern, lambda
	result = nlop_chain2_swap_FF(nlop_tenmul_create(N, img_dims, img_dims, lam_dims),0 , result, 0); //in: x0, lambda, AHy, coil, pattern, lambda
	result = nlop_dup_F(result, 1, 5); //in: x0, lambda, AHy, coil, pattern
	result = nlop_shift_input_F(result, 4, 1); //in: x0, AHy, coil, pattern, lambda
	result = nlop_chain2_FF(nlop_from_linop_F(linop_zreal_create(N, lam_dims)), 0, result, 4);

	return result;
}

/**
 * Create an operator minimizing the following functional
 *
 * out = argmin 0.5 ||Ax-y||_2^2 + lambda ||W (x-x_0)||_2^2
 * A = Pattern FFT Coils
 *
 * with W = |init-x_0|^(p/2 - 1)
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 * @param lammbda_fixed if -1, lambda is an input of the nlop
 *
 * Input tensors:
 * x0:		idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * adjoint:	idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, ..., 1 / Nb)
 * lambda:	ldims:	( 1,  1,  1,  1, ..., Nb)
 * init:	idims:	(Nx, Ny, Nz,  1, ..., Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1, ..., Nb)
 */
static const struct nlop_s* nlop_mri_weighted_dc_prox_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf, float p)
{
	long img_dims[N];
	md_select_dims(N, conf->image_flags, img_dims, max_dims);

	auto result = nlop_mri_dc_prox_create(N, max_dims, img_dims, ND, psf_dims, conf, iter_conf); //in: x0, AHy, coil, pattern, lambda
	result = nlop_chain2_FF(nlop_tenmul_create(N, img_dims, lam_dims, img_dims), 0, result, 4); //in: x0, AHy, coil, pattern, lambda, W^2
	result = nlop_chain2_FF(nlop_zspow_create(N, img_dims, 1. - p / 2), 0, result, 5);
	result = nlop_chain2_FF(nlop_zinv_create(N, img_dims), 0, result, 5);
	result = nlop_chain2_FF(nlop_smo_abs_create(N, img_dims, 0.), 0, result, 5);
	result = nlop_chain2_FF(nlop_zaxpbz_create(N, img_dims, 1., -1.), 0, result, 5);
	result = nlop_dup_F(result, 0, 5);

	return result;
}

/**
 * Create an operator minimizing the following functional
 *
 * out = argmin 0.5 ||Ax-y||_2^2 + lambda ||x-x_0||_p^p
 * A = Pattern FFT Coils
 *
 * using iteratively weighted least squares
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 * @param lammbda_fixed if -1, lambda is an input of the nlop
 *
 * Input tensors:
 * x0:		idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * adjoint:	idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, ..., 1 / Nb)
 * lambda:	ldims:	( 1,  1,  1,  1, ..., Nb)
 * init:	idimss:	( 1,  1,  1,  1, ..., Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1, ..., Nb)
 */
const struct nlop_s* nlop_mri_dc_pnorm_IRLS_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf, float p, int iter)
{
	auto result = nlop_mri_weighted_dc_prox_create(N, max_dims, lam_dims, ND, psf_dims, conf, iter_conf, p);

	for (int i = 1; i < iter; i++) {

		result = nlop_chain2_swap_FF(result, 0, nlop_mri_weighted_dc_prox_create(N, max_dims, lam_dims, ND, psf_dims, conf, iter_conf, p), 5);
		result = nlop_dup_F(result, 0, 6);
		result = nlop_dup_F(result, 1, 6);
		result = nlop_dup_F(result, 2, 6);
		result = nlop_dup_F(result, 3, 6);
		result = nlop_dup_F(result, 4, 6);
	}

	return result;
}








struct mri_normal_power_iter_s {

	INTERFACE(nlop_data_t);

	struct sense_normal_ops_s* normal_ops;

	complex float* noise;
	int max_iter;
};

DEF_TYPEID(mri_normal_power_iter_s);

static void mri_normal_power_iter_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_power_iter_s, _data);

	sense_normal_free(d->normal_ops);

	md_free(d->noise);

	xfree(d);
}

#include "num/gpuops.h"

static void mri_normal_power_iter_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_power_iter_s, _data);

	assert(3 == Narg);

	complex float* dst = args[0];
	const complex float* coil = args[1];
	const complex float* pattern = args[2];

	auto nops = d->normal_ops;

	if (NULL == d->noise) {

		d->noise = md_alloc_sameplace(nops->N, nops->img_dims_slice, CFL_SIZE, dst);
		md_gaussian_rand(nops->N, nops->img_dims_slice, d->noise);
	}

	int N_batch = md_calc_size(nops->N, nops->bat_dims);
	long bat_strs[nops->N];
	md_calc_strides(nops->N, bat_strs, nops->bat_dims, CFL_SIZE);

	complex float max_eigen[N_batch];

	sense_normal_update_ops(nops, nops->N, nops->col_dims, coil, nops->ND, nops->psf_dims, pattern);

	long pos[nops->N];
	for (int i = 0; i < nops->N; i++)
		pos[i] = 0;

	complex float* noise = md_alloc_sameplace(nops->N, nops->img_dims_slice, CFL_SIZE, dst);

	do {

		auto normal_op = sense_get_normal_op(nops, nops->N, pos);
		md_copy(nops->N, nops->img_dims_slice, noise, d->noise, CFL_SIZE);

		long size = md_calc_size(nops->N, nops->img_dims_slice);
		MD_ACCESS(nops->N, bat_strs, pos, max_eigen) = iter_power(d->max_iter, normal_op, 2 * size, (float*)d->noise);

	} while (md_next(nops->N, nops->bat_dims, ~(0ul), pos));

	md_free(noise);

	md_copy(nops->N, nops->bat_dims, dst, max_eigen, CFL_SIZE);
}


/**
 * Returns: Operator estimating the max eigen value of mri normal operator
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  ..., 1 )
 *
 * Output tensors:
 * max eigen:	ldims: 	( 1,  1,  1,  1, ..., Nb)
 */
const struct nlop_s* nlop_mri_normal_max_eigen_create(int N, const long max_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf)
{

	PTR_ALLOC(struct mri_normal_power_iter_s, data);
	SET_TYPEID(mri_normal_power_iter_s, data);

	long cim_dims[N];
	md_select_dims(N, conf->coil_flags | conf->image_flags, cim_dims, max_dims);

	data->normal_ops = sense_normal_create(N, cim_dims, ND, psf_dims, conf);

	data->noise = NULL;
	data->max_iter = 30;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->normal_ops->bat_dims);

	long nl_idims[2][ND];
	md_singleton_dims(ND, nl_idims[0]);
	md_copy_dims(N, nl_idims[0], cim_dims);
	md_copy_dims(ND, nl_idims[1], psf_dims);

	const struct nlop_s* result = nlop_generic_create(
			1, N, nl_odims, 2, ND, nl_idims, CAST_UP(PTR_PASS(data)),
			mri_normal_power_iter_fun,
			NULL, NULL,
			NULL, NULL, mri_normal_power_iter_del
		);

	result = nlop_reshape_in_F(result, 0, N, nl_idims[0]);

	return result;
}

struct mri_scale_rss_s {

	INTERFACE(nlop_data_t);
	int N;

	unsigned long rss_flag;
	unsigned long bat_flag;

	const long* col_dims;

	bool mean;
};

DEF_TYPEID(mri_scale_rss_s);

static void mri_scale_rss_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(mri_scale_rss_s, _data);

	int N = d->N;

	md_zrss(N, d->col_dims, d->rss_flag, dst, src);

	if (d->mean) {

		long bdims[N];
		long idims[N];
		md_select_dims(N, d->bat_flag, bdims, d->col_dims);
		md_select_dims(N, ~d->rss_flag, idims, d->col_dims);

		complex float* mean = md_alloc_sameplace(N, bdims, CFL_SIZE, dst);
		complex float* ones = md_alloc_sameplace(N, bdims, CFL_SIZE, dst);
		md_zfill(N, bdims, ones, 1.);

		md_zsum(N, idims, ~d->bat_flag, mean, dst);
		md_zsmul(N, bdims, mean, mean, (float)md_calc_size(N, bdims) / (float)md_calc_size(N, idims));
		md_zdiv(N, bdims, mean, ones, mean);

		md_zmul2(N, idims, MD_STRIDES(N, idims, CFL_SIZE), dst, MD_STRIDES(N, idims, CFL_SIZE), dst, MD_STRIDES(N, bdims, CFL_SIZE), mean);

		md_free(mean);
		md_free(ones);
	}
}

static void mri_scale_rss_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_scale_rss_s, _data);

	xfree(d->col_dims);
	xfree(d);
}

const struct nlop_s* nlop_mri_scale_rss_create(int N, const long max_dims[N], const struct config_nlop_mri_s* conf)
{
	PTR_ALLOC(struct mri_scale_rss_s, data);
	SET_TYPEID(mri_scale_rss_s, data);

	PTR_ALLOC(long[N], col_dims);
	md_select_dims(N, conf->coil_flags, *col_dims, max_dims);
	data->col_dims = *PTR_PASS(col_dims);

	data->N = N;
	data->bat_flag = conf->batch_flags;
	data->rss_flag = (~conf->image_flags) & (conf->coil_flags);
	data->mean = true;

	long odims[N];
	long idims[N];
	md_select_dims(N, conf->coil_flags, idims, max_dims);
	md_select_dims(N, conf->image_flags, odims, idims);


	return nlop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), mri_scale_rss_fun, NULL, NULL, NULL, NULL, mri_scale_rss_del);
}