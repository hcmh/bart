#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/iter2.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"

#include "nn/nn_ops.h"

#include "noir/model2.h"
#include "noir/recon2.h"

#include "noir/model_net.h"


static const struct nlop_s* noir_get_forward(struct noir2_s model)
{
	auto dom_im = nlop_generic_domain(model.nlop, 0);
	auto dom_coil = nlop_generic_domain(model.nlop, 1);
	auto dom_ksp = nlop_generic_codomain(model.nlop, 0);

	assert(dom_im->N == dom_coil->N);
	assert(dom_im->N == dom_ksp->N);

	int N = dom_im->N;

	const struct nlop_s* result = nlop_tenmul_create(N, dom_ksp->dims, dom_im->dims, dom_coil->dims);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model.lop_fft)), 0);
	result = nlop_chain2_swap_FF(nlop_from_linop(model.lop_im), 0, result, 0);
	result = nlop_chain2_FF(nlop_from_linop(model.lop_coil), 0, result, 1);
	result = nlop_flatten_in_F(result, 0);
	result = nlop_flatten_in_F(result, 1);
	result = nlop_stack_inputs_F(result, 0, 1, 0);

	return result;
}

static const struct nlop_s* noir_get_adjoint(struct noir2_s model)
{
	auto dom_im = nlop_generic_domain(model.nlop, 0);
	auto dom_coil = nlop_generic_domain(model.nlop, 1);
	auto dom_ksp = nlop_generic_codomain(model.nlop, 0);

	assert(dom_im->N == dom_coil->N);
	assert(dom_im->N == dom_ksp->N);

	int N = dom_im->N;

	const struct nlop_s* nlop_dim = nlop_tenmul_create(N, dom_im->dims, dom_coil->dims, dom_ksp->dims);	//in: c, z; dx_im = coils * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, dom_coil->dims)), 0 , nlop_dim, 0);	//in: z, c; dx_im = \bar{coils} * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop(model.lop_coil), 0 , nlop_dim, 1); 				//in: z, c; dx_im = \bar{lop_coil(coils)} * dz
	nlop_dim = nlop_chain2_FF(nlop_dim, 0, nlop_from_linop_F(linop_get_adjoint(model.lop_im)), 0);		//dx_im = lop_im^H(\bar{lop_coil(coils)} * dz)

	nlop_dim = nlop_flatten_in_F(nlop_dim, 1);
	nlop_dim = nlop_flatten_out_F(nlop_dim, 0);

	const struct nlop_s* nlop_dcoil = nlop_tenmul_create(N, dom_coil->dims, dom_im->dims, dom_ksp->dims);		//dx_coil = im * dz
	nlop_dcoil = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, dom_im->dims)), 0 , nlop_dcoil, 0);	//dx_coil = \bar{im} * dz
	nlop_dcoil = nlop_chain2_FF(nlop_from_linop(model.lop_im), 0 , nlop_dcoil, 1); 					//dx_coil = \bar{lop_im(im)} * dz
	nlop_dcoil = nlop_chain2_FF(nlop_dcoil, 0, nlop_from_linop_F(linop_get_adjoint(model.lop_coil)), 0);		//dx_coil = lop_coil^H(\bar{lop_im(im)} * dz)
	nlop_dcoil = nlop_flatten_in_F(nlop_dcoil, 1);
	nlop_dcoil = nlop_flatten_out_F(nlop_dcoil, 0);

	const struct nlop_s* result = nlop_combine_FF(nlop_dim, nlop_dcoil);	// out: dx_im, dx_coil; in: dz, coils, dz, im
	result = nlop_permute_inputs_F(result, 4, (const int[4]){ 0, 2, 3, 1});	// out: dx_im, dx_coil; in: dz, dz, im, coils
	result = nlop_dup_F(result, 0, 1);					// out: dx_im, dx_coil; in: dz, im, coils

	result = nlop_stack_outputs_F(result, 0, 1, 0);
	result = nlop_stack_inputs_F(result, 1, 2, 0);				// out: dx; in: dz, xn

	return result;
}

static const struct nlop_s* noir_get_derivative(struct noir2_s model)
{
	auto dom_im = nlop_generic_domain(model.nlop, 0);
	auto dom_coil = nlop_generic_domain(model.nlop, 1);
	auto dom_ksp = nlop_generic_codomain(model.nlop, 0);

	assert(dom_im->N == dom_coil->N);
	assert(dom_im->N == dom_ksp->N);

	int N = dom_im->N;

	const struct nlop_s* nlop1 = nlop_tenmul_create(N, dom_ksp->dims, dom_im->dims, dom_coil->dims);	//dz1 = im * dcoils
	nlop1 = nlop_chain2_FF(nlop_from_linop(model.lop_coil), 0 , nlop1, 1);	 				//dz1 = im * lop_coils(dcoils)
	nlop1 = nlop_chain2_swap_FF(nlop_from_linop(model.lop_im), 0 , nlop1, 0);	 			//dz1 = lop_im(im) * lop_coils(dcoils)
	nlop1 = nlop_flatten_in_F(nlop1, 0);
	nlop1 = nlop_flatten_in_F(nlop1, 1);

	const struct nlop_s* nlop2 = nlop_tenmul_create(N, dom_ksp->dims, dom_im->dims, dom_coil->dims);	//dz2 = dim * coils
	nlop2 = nlop_chain2_FF(nlop_from_linop(model.lop_coil), 0 , nlop2, 1);	 				//dz2 = dim * lop_coils(coils)
	nlop2 = nlop_chain2_swap_FF(nlop_from_linop(model.lop_im), 0 , nlop2, 0);	 			//dz2 = lop_im(dim) * lop_coils(coils)
	nlop2 = nlop_flatten_in_F(nlop2, 0);
	nlop2 = nlop_flatten_in_F(nlop2, 1);

	const struct nlop_s* result = nlop_combine_FF(nlop1, nlop2);		//out: dz1, dz2; in: im, dcoils, dim, coil;
	result = nlop_permute_inputs_F(result, 4, (const int[4]){2, 1, 0, 3});	//out: dz1, dz2; in: dim, dcoils, im, coil;
	result = nlop_stack_inputs_F(result, 0, 1, 0);
	result = nlop_stack_inputs_F(result, 1, 2, 0);				//out: z1, z2; in: dx, xn;
	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, dom_ksp->dims, 1, 1), 0);
	result = nlop_link_F(result, 1, 0);					//out: dz; in: dx, xn;

	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model.lop_fft)), 0);


	return result;
}

static const struct nlop_s* noir_get_normal(struct noir2_s model)
{
	auto der = noir_get_derivative(model);	//out: dz; in: dx, xn
	auto adj = noir_get_adjoint(model);	//out: dx; in: dz, xn

	auto result = nlop_chain2_swap_FF(der, 0, adj, 0);	//out: dx; in: dx, xn, xn
	result = nlop_dup_F(result, 1, 2);			//out: dx; in: dx, xn, xn
	return result;
}


struct noir_normal_inversion_s {

	INTERFACE(nlop_data_t);

	struct noir2_s model;

	const struct nlop_s* normal_op;

	int N;
	const long* dims;
	float alpha;

	complex float* xn;
	complex float* out;
	complex float* dout;	//Adjoint lambda and adjoint in
	complex float* AhAdout;	//share same intermediate result

	struct iter_conjgrad_conf iter_conf;
};

DEF_TYPEID(noir_normal_inversion_s);

static void noir_normal_inversion_alloc(struct noir_normal_inversion_s* d, const void* ref)
{
	if (NULL == d->out)
		d->out = md_alloc_sameplace(d->N, d->dims, CFL_SIZE, ref);
	md_clear(d->N, d->dims, d->out, CFL_SIZE);

	if (NULL == d->xn)
		d->xn = md_alloc_sameplace(d->N, d->dims, CFL_SIZE, ref);
}

static void noir_normal_inversion_set_ops(const struct noir_normal_inversion_s* d)
{
	complex float* tmp_out = md_alloc_sameplace(d->N, d->dims, CFL_SIZE, d->xn);

	nlop_generic_apply_select_derivative_unchecked(d->normal_op, 3, (void*[3]){ tmp_out, d->out, (void*)d->xn}, MD_BIT(0), MD_BIT(0));

	md_free(tmp_out);
}

static void noir_normal_inversion(const struct noir_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	const struct operator_s* normal_op = nlop_get_derivative(d->normal_op, 0, 0)->forward;

	md_clear(d->N, d->dims, dst, CFL_SIZE);

	iter2_conjgrad(	CAST_UP(&(d->iter_conf)), normal_op,
			0, NULL, NULL, NULL, NULL,
			2 * md_calc_size(d->N, d->dims),
			(float*)dst,
			(const float*)src,
			NULL);
}

static void noir_normal_inversion_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);
	assert(4 == Narg);

	complex float* dst = args[0];
	const complex float* src = args[1];
	const complex float* xn = args[2];

	noir_normal_inversion_alloc(d, dst);

	md_copy(d->N, d->dims, d->xn, xn, CFL_SIZE);

	complex float alpha;
	md_copy(1, MD_DIMS(1), &alpha, args[3], CFL_SIZE);
	if ((0 != cimagf(alpha)) || (0 > crealf(alpha)))
		error("Alpha=%f+%fi is not non-negative real number!\n", crealf(alpha), cimagf(alpha));
	d->iter_conf.INTERFACE.alpha = crealf(alpha);

	noir_normal_inversion_set_ops(d);

	noir_normal_inversion(d, dst, src);

	md_copy(d->N, d->dims, d->out, dst, CFL_SIZE);

	md_free(d->dout);
	md_free(d->AhAdout);
	d->out = NULL;
	d->AhAdout = NULL;
}


static void noir_normal_inversion_der_src(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);

	noir_normal_inversion_set_ops(d);	//should not be needed but allows freeing of in nlop
	noir_normal_inversion(d, dst, src);
}

static void noir_normal_inversion_der_xn(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);

	complex float* tmp = md_alloc_sameplace(d->N, d->dims, CFL_SIZE, dst);
	noir_normal_inversion_set_ops(d);

	linop_forward(nlop_get_derivative(d->normal_op, 0, 1), d->N, d->dims, tmp, d->N, d->dims, src);
	md_zsmul(d->N, d->dims, tmp, tmp, -1);
	noir_normal_inversion(d, dst, tmp);

	md_free(tmp);
}

static void noir_normal_inversion_der_alpha(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);

	noir_normal_inversion_set_ops(d);

	noir_normal_inversion(d, dst, d->out);
	md_zsmul(d->N, d->dims, dst, dst, -1);
	md_zmul2(d->N, d->dims, MD_STRIDES(d->N, d->dims, CFL_SIZE), dst, MD_STRIDES(d->N, d->dims, CFL_SIZE), dst, MD_SINGLETON_STRS(d->N), src);
}

static void noir_normal_inversion_alloc_adj(struct noir_normal_inversion_s* d, const void* ref)
{
	if (NULL == d->dout)
		d->dout = md_alloc_sameplace(d->N, d->dims, CFL_SIZE, ref);

	if (NULL == d->AhAdout)
		d->dout = md_alloc_sameplace(d->N, d->dims, CFL_SIZE, ref);

	md_clear(d->N, d->dims, d->dout, CFL_SIZE);
	md_clear(d->N, d->dims, d->AhAdout, CFL_SIZE);
}


static void noir_normal_inversion_compute_adjoint(struct noir_normal_inversion_s* d, const complex float* src)
{
	noir_normal_inversion_alloc_adj(d, src);

	if (0 != md_zrmse(d->N, d->dims, d->dout, src)) {

		md_copy(d->N, d->dims, d->dout, src, CFL_SIZE);
		noir_normal_inversion_set_ops(d);
		noir_normal_inversion(d, d->AhAdout, src);
	}
}


static void noir_normal_inversion_adj_src(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);

	noir_normal_inversion_compute_adjoint(d, src);
	md_copy(d->N, d->dims, dst, d->AhAdout, CFL_SIZE);
}

static void noir_normal_inversion_adj_xn(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);

	noir_normal_inversion_compute_adjoint(d, src);
	linop_adjoint(nlop_get_derivative(d->normal_op, 0, 1), d->N, d->dims, dst, d->N, d->dims, d->AhAdout);

	complex float* tmp = md_alloc_sameplace(d->N, d->dims, CFL_SIZE, dst);
	noir_normal_inversion_set_ops(d);

	linop_forward(nlop_get_derivative(d->normal_op, 0, 1), d->N, d->dims, tmp, d->N, d->dims, src);
	md_zsmul(d->N, d->dims, tmp, tmp, -1);
	noir_normal_inversion(d, dst, tmp);

	md_free(tmp);
}

static void noir_normal_inversion_adj_alpha(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);

	noir_normal_inversion_compute_adjoint(d, src);

	md_ztenmulc(d->N, MD_SINGLETON_DIMS(d->N), dst, d->dims, d->out, d->dims, d->AhAdout);

	md_zsmul(d->N, MD_SINGLETON_DIMS(d->N), dst, dst, -1);
	md_zreal(d->N, MD_SINGLETON_DIMS(d->N), dst, dst);
}

static void noir_normal_inversion_free(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(noir_normal_inversion_s, _data);

	nlop_free(d->normal_op);
	xfree(d->dims);

	md_free(d->xn);
	md_free(d->out);
	md_free(d->dout);
	md_free(d->AhAdout);
}

static struct noir_normal_inversion_s* noir_normal_inversion_data_create(struct noir2_s model, const struct iter_conjgrad_conf* iter_conf)
{

	PTR_ALLOC(struct noir_normal_inversion_s, data);
	SET_TYPEID(noir_normal_inversion_s, data);

	data->model = model;
	data->normal_op = noir_get_normal(model);

	data->N = nlop_generic_codomain(data->normal_op, 0)->N;
	PTR_ALLOC(long[data->N], dims);
	md_copy_dims(data->N, *dims, nlop_generic_codomain(data->normal_op, 0)->dims);
	data->dims = *PTR_PASS(dims);

	data->normal_op = noir_get_normal(model);

	data->alpha = 0;

	data-> xn = NULL;
	data->out = NULL;
	data->dout = NULL;
	data->AhAdout = NULL;

	if (NULL == iter_conf) {

		data->iter_conf = iter_conjgrad_defaults;
		data->iter_conf.l2lambda = 1.;
		data->iter_conf.maxiter = 50;
	} else {

		data->iter_conf = *iter_conf;
	}

	return PTR_PASS(data);
}


static const struct nlop_s* noir_normal_inversion_create(struct noir2_s model, const struct iter_conjgrad_conf* iter_conf)
{
	auto data = noir_normal_inversion_data_create(model, iter_conf);

	assert(1 == data->N);
	int N = data->N;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->dims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], data->dims);
	md_copy_dims(N, nl_idims[1], data->dims);
	md_singleton_dims(N, nl_idims[2]);

	return nlop_generic_create(	1, N, nl_odims, 3, N, nl_idims, CAST_UP(data),
					noir_normal_inversion_fun,
					(nlop_der_fun_t[3][1]){ { noir_normal_inversion_der_src }, { noir_normal_inversion_der_xn }, { noir_normal_inversion_der_alpha } },
					(nlop_der_fun_t[3][1]){ { noir_normal_inversion_adj_src }, { noir_normal_inversion_adj_xn }, { noir_normal_inversion_adj_alpha } },
					NULL, NULL, noir_normal_inversion_free);

}


static const struct nlop_s* noir_gauss_newton_step_create(struct noir2_s model, const struct iter_conjgrad_conf* iter_conf)
{
	auto result = noir_get_forward(model);	//out: F(xn); in: xn

	auto dom = nlop_domain(result);
	auto cod = nlop_codomain(result);

	assert(1 == dom->N);
	int N = cod->N;

	long dims[1];
	long kdims[N];

	md_copy_dims(1, dims, dom->dims);
	md_copy_dims(N, kdims, cod->dims);

	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, kdims, 1, -1), 1);			//out: y - F(xn); in: y, xn
	result = nlop_chain2_swap_FF(result, 0, noir_get_adjoint(model), 0);				//out: DF(xn)^H(y - F(xn)); in: y, xn, xn
	result = nlop_dup_F(result, 1, 2);								//out: DF(xn)^H(y - F(xn)); in: y, xn
	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(1, dims, 1, -1), 0);			//out: DF(xn)^H(y - F(xn)) - alpha(xn - x0); in: y, xn, alpha(xn - x0)

	auto nlop_reg = nlop_zaxpbz_create(1, dims, 1, -1);						//out: xn - x0; in: xn, x0
	nlop_reg = nlop_chain2_swap_FF(nlop_reg, 0, nlop_tenmul_create(1, dims, dims, MD_DIMS(1)), 0);	//out: alpha(x_n - x_0); in: xn, x0, alpha

	result = nlop_chain2_FF(nlop_reg, 0, result, 2);						//out: DF(xn)^H(y - F(xn)) - alpha(xn - x0); in: y, xn, xn, x0, alpha
	result = nlop_dup_F(result, 1, 2);								//out: DF(xn)^H(y - F(xn)) - alpha(xn - x0); in: y, xn, x0, alpha


	result = nlop_chain2_swap_FF(result, 0, noir_normal_inversion_create(model, iter_conf), 0);	//out: (DF(xn)^H DF(xn) + alpha)^-1 DF(xn)^H(y - F(xn)) - alpha(xn - x0); in: y, xn, x0, alpha, xn, alpha
	result = nlop_dup_F(result, 1, 4);								//out: (DF(xn)^H DF(xn) + alpha)^-1 DF(xn)^H(y - F(xn)) - alpha(xn - x0); in: y, xn, x0, alpha, alpha
	result = nlop_dup_F(result, 3, 4);								//out: (DF(xn)^H DF(xn) + alpha)^-1 DF(xn)^H(y - F(xn)) - alpha(xn - x0); in: y, xn, x0, alpha

	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(1, dims, 1, 1), 0);
	result = nlop_dup_F(result, 1, 4);

	result = nlop_reshape_in_F(result, 3, N, MD_SINGLETON_DIMS(N));

	return result;
}

const struct nlop_s* noir_cart_unrolled_create(		int N,
							const long pat_dims[N], const complex float* pattern,
							const long bas_dims[N], const complex float* basis,
							const long msk_dims[N], const complex float* mask,
							const long ksp_dims[N],
							const long cim_dims[N],
							const long img_dims[N],
							const long col_dims[N],
							struct noir2_conf_s* conf)
{
	struct noir2_model_conf_s model_conf = noir2_model_conf_defaults;
	model_conf.fft_flags_noncart = 0;
	model_conf.fft_flags_cart = FFT_FLAGS | ((conf->sms || conf->sos) ? SLICE_FLAG : 0);
	model_conf.rvc = conf->rvc;
	model_conf.sos = conf->sos;
	model_conf.a = conf->a;
	model_conf.b = conf->b;
	model_conf.noncart = conf->noncart;
	model_conf.nufft_conf = conf->nufft_conf;

	struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
	iter_conf.INTERFACE.alpha = 1.;
	iter_conf.l2lambda = 1.;
	iter_conf.maxiter = (0 == conf->cgiter) ? 30 : conf->cgiter;
	iter_conf.tol = conf->cgtol;


	struct noir2_s model = noir2_cart_create(N, pat_dims, pattern, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &model_conf);


	auto result = noir_gauss_newton_step_create(model, &iter_conf);

	for (int i = 1; i < conf->iter; i++){

		result = nlop_chain2_FF(nlop_from_linop_F(linop_scale_create(N, MD_SINGLETON_DIMS(N), 1. / conf->redu)), 0, result, 3);
		result = nlop_chain2_swap_FF(noir_gauss_newton_step_create(model, &iter_conf), 0, result, 1); // in: y, xn, x0, alpha, y, x0, alpha
		result = nlop_dup_F(result, 0, 4);
		result = nlop_dup_F(result, 2, 4);
		result = nlop_dup_F(result, 3, 4); // in: y, xn, x0, alpha
	}

	complex float alpha = conf->alpha;
	result = nlop_set_input_const_F2(result, 3, N, MD_SINGLETON_DIMS(N), MD_SINGLETON_STRS(N), true, &alpha);	// in: y, xn, x0

	long size = md_calc_size(N, img_dims) + md_calc_size(N, col_dims);
	long skip = md_calc_size(N, img_dims);

	complex float zero = 0;
	result = nlop_set_input_const_F2(result, 2, 1, MD_DIMS(size), MD_SINGLETON_STRS(1), true, &zero);	// in: y, xn

	complex float* init = md_alloc(1, MD_DIMS(size), CFL_SIZE);
	md_clear(1, MD_DIMS(size), init, CFL_SIZE);
	md_zfill(1, MD_DIMS(skip), init, 1.);

	result = nlop_set_input_const_F(result, 1, 1, MD_DIMS(size), true, init);	// in: y

	md_free(init);

	const struct nlop_s* nlop_decomp = nlop_combine_FF(nlop_from_linop_F(linop_identity_create(N, img_dims)), nlop_from_linop(model.lop_coil));
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 0);
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 1);
	nlop_decomp = nlop_stack_inputs_F(nlop_decomp, 0, 1, 0);

	result = nlop_chain2_FF(result, 0, nlop_decomp, 0);

	complex float scale = 100.;

	result = nlop_chain2_FF(nlop_from_linop_F(linop_scale_create(N, cim_dims, scale)), 0, result, 0);
	//result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_scale_create(N, img_dims, 1. / scale)), 0);

	if (true) {

		result = nlop_chain2_FF(nlop_norm_znorm_create(N, cim_dims, MD_BIT(15)), 0, result, 0); //)BATCH_FLAG);

		result = nlop_del_out_F(result, 2);

		//long sdims[N];
		//md_select_dims(N, MD_BIT(15), sdims, img_dims);
		//result = nlop_chain2_FF(result, 0, nlop_tenmul_create(N, img_dims, img_dims, sdims), 0);
		//result = nlop_link_F(result, 2, 0);
	}

	result = nlop_chain2_FF(nlop_from_linop_F(linop_get_adjoint(model.lop_fft)), 0, result, 0);

	return result;
}
