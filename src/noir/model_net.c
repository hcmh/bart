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
#include "nlops/checkpointing.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"
#include "nlops/norm_inv.h"

#include "noncart/nufft.h"

#include "nn/nn_ops.h"

#include "noir/model2.h"
#include "noir/recon2.h"

#include "noir/model_net.h"

int noir_model_get_N(struct noir2_s* model)
{
	auto dom_im = nlop_generic_domain(model->tenmul, 0);
	auto dom_coil = nlop_generic_domain(model->tenmul, 1);
	auto dom_ksp = nlop_generic_codomain(model->tenmul, 0);

	assert(dom_im->N == dom_coil->N);
	assert(dom_im->N == dom_ksp->N);

	return dom_im->N;
}

void noir_model_get_img_dims(int N, long img_dims[N], struct noir2_s* model)
{
	auto dom_im = nlop_generic_domain(model->nlop, 0);
	assert((int)dom_im->N == N);
	md_copy_dims(N, img_dims, dom_im->dims);
}

void noir_model_get_col_dims(int N, long col_dims[N], struct noir2_s* model)
{
	auto dom_col = nlop_generic_domain(model->nlop, 1);
	assert((int)dom_col->N == N);
	md_copy_dims(N, col_dims, dom_col->dims);
}

static void noir_model_get_img_tm_dims(int N, long img_dims[N], struct noir2_s* model)
{
	auto dom_im = nlop_generic_domain(model->tenmul, 0);
	assert((int)dom_im->N == N);
	md_copy_dims(N, img_dims, dom_im->dims);
}

static void noir_model_get_col_tm_dims(int N, long col_dims[N], struct noir2_s* model)
{
	auto dom_col = nlop_generic_domain(model->tenmul, 1);
	assert((int)dom_col->N == N);
	md_copy_dims(N, col_dims, dom_col->dims);
}

void noir_model_get_cim_dims(int N, long cim_dims[N], struct noir2_s* model)
{
	auto dom_cim = nlop_generic_codomain(model->tenmul, 0);
	assert((int)dom_cim->N == N);
	md_copy_dims(N, cim_dims, dom_cim->dims);
}



long noir_model_get_size(struct noir2_s* model)
{
	int N = noir_model_get_N(model);

	long img_dims[N];
	long col_dims[N];

	noir_model_get_img_dims(N, img_dims, model);
	noir_model_get_col_dims(N, col_dims, model);

	return md_calc_size(N, img_dims) + md_calc_size(N, col_dims);
}

long noir_model_get_skip(struct noir2_s* model)
{
	int N = noir_model_get_N(model);

	long img_dims[N];

	noir_model_get_img_dims(N, img_dims, model);

	return md_calc_size(N, img_dims);
}

#if 0
static const struct nlop_s* noir_debug_coil_create(struct noir2_s* model, const char* filename)
{
	int N = noir_model_get_N(model);

	long img_dims[N];
	long col_dims[N];

	noir_model_get_img_tm_dims(N, img_dims, model);
	noir_model_get_col_tm_dims(N, col_dims, model);

	auto result = noir_decomp_create(model);
	result = nlop_del_out_F(result, 0);
	result = nlop_chain_FF(result, nlop_dump_create(N, col_dims, filename, true, false, false));
	result = nlop_del_out_F(result, 0);

	result = nlop_combine_FF(result, nlop_from_linop_F(linop_identity_create(1, MD_DIMS(noir_model_get_size(model)))));
	result = nlop_dup_F(result, 0, 1);

	return result;
}

static const struct nlop_s* noir_debug_img_create(struct noir2_s* model, const char* filename)
{
	int N = noir_model_get_N(model);

	long img_dims[N];
	long col_dims[N];

	noir_model_get_img_tm_dims(N, img_dims, model);
	noir_model_get_col_tm_dims(N, col_dims, model);

	auto result = noir_decomp_create(model);
	result = nlop_del_out_F(result, 1);
	result = nlop_chain_FF(result, nlop_dump_create(N, img_dims, filename, true, false, false));
	result = nlop_del_out_F(result, 0);

	result = nlop_combine_FF(result, nlop_from_linop_F(linop_identity_create(1, MD_DIMS(noir_model_get_size(model)))));
	result = nlop_dup_F(result, 0, 1);

	return result;
}
#endif


static const struct nlop_s* noir_get_forward(struct noir2_s* model)
{
	int N = noir_model_get_N(model);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_tm_dims(N, img_dims, model);
	noir_model_get_col_tm_dims(N, col_dims, model);
	noir_model_get_cim_dims(N, cim_dims, model);

	const struct nlop_s* result = nlop_tenmul_create(N, cim_dims, img_dims, col_dims);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model->lop_fft)), 0);
	result = nlop_chain2_swap_FF(nlop_from_linop(model->lop_im), 0, result, 0);
	result = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0, result, 1);
	result = nlop_flatten_in_F(result, 0);
	result = nlop_flatten_in_F(result, 1);
	result = nlop_stack_inputs_F(result, 0, 1, 0);

	return result;
}

static const struct nlop_s* noir_get_adjoint(struct noir2_s* model)
{
	int N = noir_model_get_N(model);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_tm_dims(N, img_dims, model);
	noir_model_get_col_tm_dims(N, col_dims, model);
	noir_model_get_cim_dims(N, cim_dims, model);

	const struct nlop_s* nlop_dim = nlop_tenmul_create(N, img_dims, col_dims, cim_dims);			//in: c, z; dx_im = coils * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, col_dims)), 0 , nlop_dim, 0);		//in: z, c; dx_im = \bar{coils} * dz
	nlop_dim = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0 , nlop_dim, 1); 				//in: z, c; dx_im = \bar{lop_coil(coils)} * dz
	nlop_dim = nlop_chain2_FF(nlop_dim, 0, nlop_from_linop_F(linop_get_adjoint(model->lop_im)), 0);		//dx_im = lop_im^H(\bar{lop_coil(coils)} * dz)

	nlop_dim = nlop_flatten_in_F(nlop_dim, 1);
	nlop_dim = nlop_flatten_out_F(nlop_dim, 0);

	const struct nlop_s* nlop_dcoil = nlop_tenmul_create(N, col_dims, img_dims, cim_dims);			//dx_coil = im * dz
	nlop_dcoil = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, img_dims)), 0 , nlop_dcoil, 0);	//dx_coil = \bar{im} * dz
	nlop_dcoil = nlop_chain2_FF(nlop_from_linop(model->lop_im), 0 , nlop_dcoil, 1); 			//dx_coil = \bar{lop_im(im)} * dz
	nlop_dcoil = nlop_chain2_FF(nlop_dcoil, 0, nlop_from_linop_F(linop_get_adjoint(model->lop_coil)), 0);	//dx_coil = lop_coil^H(\bar{lop_im(im)} * dz)
	nlop_dcoil = nlop_flatten_in_F(nlop_dcoil, 1);
	nlop_dcoil = nlop_flatten_out_F(nlop_dcoil, 0);

	const struct nlop_s* result = nlop_combine_FF(nlop_dim, nlop_dcoil);	// out: dx_im, dx_coil; in: dz, coils, dz, im
	result = nlop_permute_inputs_F(result, 4, (const int[4]){ 0, 2, 3, 1});	// out: dx_im, dx_coil; in: dz, dz, im, coils
	result = nlop_dup_F(result, 0, 1);					// out: dx_im, dx_coil; in: dz, im, coils

	result = nlop_stack_outputs_F(result, 0, 1, 0);
	result = nlop_stack_inputs_F(result, 1, 2, 0);				// out: dx; in: dz, xn

	return result;
}

static const struct nlop_s* noir_get_derivative(struct noir2_s* model)
{
	int N = noir_model_get_N(model);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_tm_dims(N, img_dims, model);
	noir_model_get_col_tm_dims(N, col_dims, model);
	noir_model_get_cim_dims(N, cim_dims, model);

	const struct nlop_s* nlop1 = nlop_tenmul_create(N, cim_dims, img_dims, col_dims);	//dz1 = im * dcoils
	nlop1 = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0 , nlop1, 1);	 		//dz1 = im * lop_coils(dcoils)
	nlop1 = nlop_chain2_swap_FF(nlop_from_linop(model->lop_im), 0 , nlop1, 0);	 	//dz1 = lop_im(im) * lop_coils(dcoils)
	nlop1 = nlop_flatten_in_F(nlop1, 0);
	nlop1 = nlop_flatten_in_F(nlop1, 1);

	const struct nlop_s* nlop2 = nlop_tenmul_create(N, cim_dims, img_dims, col_dims);	//dz2 = dim * coils
	nlop2 = nlop_chain2_FF(nlop_from_linop(model->lop_coil), 0 , nlop2, 1);	 		//dz2 = dim * lop_coils(coils)
	nlop2 = nlop_chain2_swap_FF(nlop_from_linop(model->lop_im), 0 , nlop2, 0);	 	//dz2 = lop_im(dim) * lop_coils(coils)
	nlop2 = nlop_flatten_in_F(nlop2, 0);
	nlop2 = nlop_flatten_in_F(nlop2, 1);

	const struct nlop_s* result = nlop_combine_FF(nlop1, nlop2);			//out: dz1, dz2; in: im, dcoils, dim, coil;
	result = nlop_permute_inputs_F(result, 4, (const int[4]){2, 1, 0, 3});		//out: dz1, dz2; in: dim, dcoils, im, coil;
	result = nlop_stack_inputs_F(result, 0, 1, 0);
	result = nlop_stack_inputs_F(result, 1, 2, 0);					//out: z1, z2; in: dx, xn;
	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, cim_dims, 1, 1), 0);
	result = nlop_link_F(result, 1, 0);						//out: dz; in: dx, xn;

	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model->lop_fft)), 0);


	return result;
}

static const struct nlop_s* noir_get_normal(struct noir2_s* model)
{
	auto der = noir_get_derivative(model);	//out: dz; in: dx, xn
	auto adj = noir_get_adjoint(model);	//out: dx; in: dz, xn

	auto result = nlop_chain2_swap_FF(der, 0, adj, 0);	//out: dx; in: dx, xn, xn
	result = nlop_dup_F(result, 1, 2);			//out: dx; in: dx, xn, xn
	return result;
}


static const struct nlop_s* noir_normal_inversion_create(struct noir2_s* model, const struct iter_conjgrad_conf* iter_conf)
{

	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	if (NULL == iter_conf) {

		cgconf.l2lambda = 0.;
		cgconf.maxiter = 30;
		cgconf.tol = 0;
	} else {

		cgconf = *iter_conf;
	}

	struct nlop_norm_inv_conf conf = nlop_norm_inv_default;
	conf.iter_conf = &cgconf;

	auto normal_op = noir_get_normal(model);
	auto result = norm_inv_lambda_create(&conf, normal_op, ~0);
	nlop_free(normal_op);
	return result;
}


static const struct nlop_s* noir_gauss_newton_step_create(struct noir2_s* model, const struct iter_conjgrad_conf* iter_conf, float update)
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
	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(1, dims, 1, -1), 0);			//out: DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, lambda(xn - x0)

	auto nlop_reg = nlop_zaxpbz_create(1, dims, 1, -1);						//out: xn - x0; in: xn, x0
	nlop_reg = nlop_chain2_swap_FF(nlop_reg, 0, nlop_tenmul_create(1, dims, dims, dims), 0);	//out: lambda(x_n - x_0); in: xn, x0, lambda

	result = nlop_chain2_FF(nlop_reg, 0, result, 2);						//out: DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, xn, x0, lambda
	result = nlop_dup_F(result, 1, 2);								//out: DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda

	auto nlop_inv = noir_normal_inversion_create(model, iter_conf);

	result = nlop_chain2_swap_FF(result, 0, nlop_inv, 0);						//out: (DF(xn)^H DF(xn) + lambda)^-1 DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda, xn, lambda
	result = nlop_dup_F(result, 1, 4);								//out: (DF(xn)^H DF(xn) + lambda)^-1 DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda, lambda
	result = nlop_dup_F(result, 3, 4);								//out: (DF(xn)^H DF(xn) + lambda)^-1 DF(xn)^H(y - F(xn)) - lambda(xn - x0); in: y, xn, x0, lambda

	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(1, dims, update, 1), 0);
	result = nlop_dup_F(result, 1, 4);


	return result;
}

const struct nlop_s* noir_gauss_newton_step_batch_create(int Nb, struct noir2_s* model[Nb], const struct iter_conjgrad_conf* iter_conf, float update)
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++){

		nlops[i] = noir_gauss_newton_step_create(model[i], iter_conf, update);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 1);
		nlops[i] = nlop_append_singleton_dim_in_F(nlops[i], 2);
		nlops[i] = nlop_append_singleton_dim_out_F(nlops[i], 0);
	}

	int istack_dims[] = { BATCH_DIM, 1, 1, -1};
	int ostack_dims[] = { 1 };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 4, istack_dims, 1, ostack_dims), false, false);
}

const struct nlop_s* noir_decomp_create(struct noir2_s* model)
{
	int N = noir_model_get_N(model);
	long img_dims[N];
	noir_model_get_img_dims(N, img_dims, model);

	const struct nlop_s* nlop_decomp = nlop_combine_FF(nlop_from_linop(model->lop_im), nlop_from_linop(model->lop_coil));
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 0);
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 1);
	nlop_decomp = nlop_stack_inputs_F(nlop_decomp, 0, 1, 0);

	return nlop_decomp;
}

const struct nlop_s* noir_decomp_batch_create(int Nb, struct noir2_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_append_singleton_dim_in_F(noir_decomp_create(model[i]), 0);

	int istack_dims[] = { 1 };
	int ostack_dims[] = { BATCH_DIM, BATCH_DIM };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 1, istack_dims, 2, ostack_dims), false, false);
}

const struct nlop_s* noir_cim_batch_create(int Nb, struct noir2_s* model[Nb])
{
	auto result = noir_decomp_batch_create(Nb, model);

	int N = noir_model_get_N(model[0]);

	long img_dims[N];
	long col_dims[N];
	long cim_dims[N];

	noir_model_get_img_tm_dims(N, img_dims, model[0]);
	noir_model_get_col_tm_dims(N, col_dims, model[0]);
	noir_model_get_cim_dims(N, cim_dims, model[0]);

	img_dims[BATCH_DIM] = Nb;
	col_dims[BATCH_DIM] = Nb;
	cim_dims[BATCH_DIM] = Nb;

	result = nlop_combine_FF(nlop_tenmul_create(N, cim_dims, img_dims, col_dims), result);
	result = nlop_link_F(result, 1, 0);
	result = nlop_link_F(result, 1, 0);

	return result;
}

const struct nlop_s* noir_split_create(struct noir2_s* model)
{
	int N = noir_model_get_N(model);
	long img_dims[N];
	long col_dims[N];
	noir_model_get_img_dims(N, img_dims, model);
	noir_model_get_col_dims(N, col_dims, model);

	const struct nlop_s* nlop_decomp = nlop_combine_FF(nlop_from_linop_F(linop_identity_create(N, img_dims)),
							   nlop_from_linop_F(linop_identity_create(N, col_dims)));
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 0);
	nlop_decomp = nlop_flatten_in_F(nlop_decomp, 1);
	nlop_decomp = nlop_stack_inputs_F(nlop_decomp, 0, 1, 0);

	return nlop_decomp;
}

const struct nlop_s* noir_split_batch_create(int Nb, struct noir2_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_append_singleton_dim_in_F(noir_split_create(model[i]), 0);

	int istack_dims[] = { 1 };
	int ostack_dims[] = { BATCH_DIM, BATCH_DIM };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 1, istack_dims, 2, ostack_dims), false, false);
}

const struct nlop_s* noir_join_create(struct noir2_s* model)
{
	int N = noir_model_get_N(model);
	long img_dims[N];
	long col_dims[N];
	noir_model_get_img_dims(N, img_dims, model);
	noir_model_get_col_dims(N, col_dims, model);

	const struct nlop_s* nlop_join = nlop_combine_FF(nlop_from_linop_F(linop_identity_create(N, img_dims)),
							 nlop_from_linop_F(linop_identity_create(N, col_dims)));
	nlop_join = nlop_flatten_out_F(nlop_join, 0);
	nlop_join = nlop_flatten_out_F(nlop_join, 1);
	nlop_join = nlop_stack_outputs_F(nlop_join, 0, 1, 0);

	return nlop_join;
}

const struct nlop_s* noir_join_batch_create(int Nb, struct noir2_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_append_singleton_dim_out_F(noir_join_create(model[i]), 0);

	int ostack_dims[] = { 1 };
	int istack_dims[] = { BATCH_DIM, BATCH_DIM };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 2, istack_dims, 1, ostack_dims), false, false);
}

const struct nlop_s* noir_extract_img_batch_create(int Nb, struct noir2_s* model[Nb])
{
	auto result = noir_split_batch_create(Nb, model);
	return nlop_del_out_F(result, 1);
}

const struct nlop_s* noir_set_img_batch_create(int Nb, struct noir2_s* model[Nb])
{
	auto result = noir_join_batch_create(Nb, model);
	auto dom = nlop_generic_domain(result, 1);
	complex float zero = 0;
	return nlop_set_input_const_F2(result, 1, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &zero);
}

const struct nlop_s* noir_set_col_batch_create(int Nb, struct noir2_s* model[Nb])
{
	auto result = noir_join_batch_create(Nb, model);
	auto dom = nlop_generic_domain(result, 0);
	complex float zero = 0;
	return nlop_set_input_const_F2(result, 0, dom->N, dom->dims, MD_SINGLETON_STRS(dom->N), true, &zero);
}


struct noir_adjoint_fft_s {

	INTERFACE(nlop_data_t);
	struct noir2_s* model;
};

DEF_TYPEID(noir_adjoint_fft_s);

static void noir_adjoint_fft_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src = args[1];
	const complex float* pat = args[2];

	linop_gdiag_set_diag(data->model->lop_pattern, data->model->N, data->model->pat_dims, pat);
	linop_adjoint_unchecked(data->model->lop_fft, dst, src);
}

static void noir_adjoint_nufft_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	assert(4 == N);

	complex float* dst = args[0];
	const complex float* src = args[1];
	const complex float* wgh = args[2];
	const complex float* trj = args[3];

	auto model = data->model;

	nufft_update_traj(model->lop_nufft, model->N, model->trj_dims, trj, model->pat_dims, wgh, model->bas_dims, NULL);

	linop_adjoint_unchecked(data->model->lop_fft, dst, src);
}

static void noir_adjoint_fft_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	linop_adjoint_unchecked(data->model->lop_fft, dst, src);
}

static void noir_adjoint_fft_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);
	const auto data = CAST_DOWN(noir_adjoint_fft_s, _data);
	linop_forward_unchecked(data->model->lop_fft, dst, src);
}


static void noir_adjoint_fft_del(const nlop_data_t* _data)
{
	xfree(_data);
}

const struct nlop_s* noir_adjoint_fft_create(struct noir2_s* model)
{

	PTR_ALLOC(struct noir_adjoint_fft_s, data);
	SET_TYPEID(noir_adjoint_fft_s, data);

	data->model = model;

	auto cod = linop_codomain(model->lop_fft);
	auto dom = linop_domain(model->lop_fft);

	int N = model->N;
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dom->dims);


	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], cod->dims);
	md_copy_dims(N, nl_idims[1], data->model->pat_dims);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
					noir_adjoint_fft_fun,
					(nlop_der_fun_t[2][1]){ { noir_adjoint_fft_der }, { NULL } },
					(nlop_der_fun_t[2][1]){ { noir_adjoint_fft_adj }, { NULL } },
					NULL, NULL, noir_adjoint_fft_del);
}

const struct nlop_s* noir_adjoint_fft_batch_create(int Nb, struct noir2_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = noir_adjoint_fft_create(model[i]);

	int istack_dims[] = { BATCH_DIM, BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 2, istack_dims, 1, ostack_dims), false, false);
}

const struct nlop_s* noir_adjoint_nufft_create(struct noir2_s* model)
{

	PTR_ALLOC(struct noir_adjoint_fft_s, data);
	SET_TYPEID(noir_adjoint_fft_s, data);

	data->model = model;

	auto cod = linop_codomain(model->lop_fft);
	auto dom = linop_domain(model->lop_fft);

	int N = model->N;
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dom->dims);


	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], cod->dims);
	md_copy_dims(N, nl_idims[1], data->model->pat_dims);
	md_copy_dims(N, nl_idims[2], data->model->trj_dims);


	return nlop_generic_create(1, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)),
					noir_adjoint_nufft_fun,
					(nlop_der_fun_t[3][1]){ { noir_adjoint_fft_der }, { NULL }, { NULL } },
					(nlop_der_fun_t[3][1]){ { noir_adjoint_fft_adj }, { NULL }, { NULL } },
					NULL, NULL, noir_adjoint_fft_del);
}

const struct nlop_s* noir_adjoint_nufft_batch_create(int Nb, struct noir2_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = noir_adjoint_nufft_create(model[i]);

	int istack_dims[] = { BATCH_DIM, BATCH_DIM, BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 3, istack_dims, 1, ostack_dims), false, false);
}


const struct nlop_s* noir_fft_create(struct noir2_s* model)
{
	return nlop_from_linop_F(linop_fftc_create(model->N, model->cim_dims, model->model_conf.fft_flags_cart));
}

const struct nlop_s* noir_fft_batch_create(int Nb, struct noir2_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = noir_fft_create(model[i]);

	int istack_dims[] = { BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 1, istack_dims, 1, ostack_dims), false, false);
}


struct noir_nufft_s {

	INTERFACE(nlop_data_t);
	const struct linop_s* nufft;

	int N;
	long* trj_dims;
	long* cim_dims;
	long* ksp_dims;
};

DEF_TYPEID(noir_nufft_s);

static void noir_nufft_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(noir_nufft_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src = args[1];
	const complex float* trj = args[2];

	nufft_update_traj(data->nufft, data->N, data->trj_dims, trj, NULL, NULL, NULL, NULL);

	complex float* src_cpu = md_alloc(data->N, data->cim_dims, CFL_SIZE);
	complex float* dst_cpu = md_alloc(data->N, data->ksp_dims, CFL_SIZE);

	md_copy(data->N, data->cim_dims, src_cpu, src, CFL_SIZE);

	linop_forward_unchecked(data->nufft, dst_cpu, src_cpu);

	md_copy(data->N, data->ksp_dims, dst, dst_cpu, CFL_SIZE);

	md_free(src_cpu);
	md_free(dst_cpu);
}

static void noir_nufft_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);
	const auto data = CAST_DOWN(noir_nufft_s, _data);

	complex float* src_cpu = md_alloc(data->N, data->cim_dims, CFL_SIZE);
	complex float* dst_cpu = md_alloc(data->N, data->ksp_dims, CFL_SIZE);

	md_copy(data->N, data->cim_dims, src_cpu, src, CFL_SIZE);

	linop_forward_unchecked(data->nufft, dst_cpu, src_cpu);

	md_copy(data->N, data->ksp_dims, dst, dst_cpu, CFL_SIZE);

	md_free(src_cpu);
	md_free(dst_cpu);
}

static void noir_nufft_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	assert(0 == i);
	const auto data = CAST_DOWN(noir_nufft_s, _data);
	linop_adjoint_unchecked(data->nufft, dst, src);
}

static void noir_nufft_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(noir_nufft_s, _data);

	xfree(data->trj_dims);
	xfree(data->cim_dims);
	xfree(data->ksp_dims);

	linop_free(data->nufft);

	xfree(_data);
}

const struct nlop_s* noir_nufft_create(struct noir2_s* model)
{

	PTR_ALLOC(struct noir_nufft_s, data);
	SET_TYPEID(noir_nufft_s, data);

	auto conf = nufft_conf_defaults;
	conf.toeplitz = false;

	data->nufft = nufft_create2(model->N, model->ksp_dims, model->cim_dims, model->trj_dims, NULL, model->pat_dims, NULL, model->bas_dims, NULL, conf);
	data->N = model->N;

	data->trj_dims = *TYPE_ALLOC(long[model->N]);
	data->ksp_dims = *TYPE_ALLOC(long[model->N]);
	data->cim_dims = *TYPE_ALLOC(long[model->N]);

	md_copy_dims(model->N, data->trj_dims, model->trj_dims);
	md_copy_dims(model->N, data->ksp_dims, model->ksp_dims);
	md_copy_dims(model->N, data->cim_dims, model->cim_dims);

	int N = model->N;
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], model->ksp_dims);


	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], model->cim_dims);
	md_copy_dims(N, nl_idims[1], model->trj_dims);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
					noir_nufft_fun,
					(nlop_der_fun_t[2][1]){ { noir_nufft_der }, { NULL } },
					(nlop_der_fun_t[2][1]){ { noir_nufft_adj }, { NULL } },
					NULL, NULL, noir_nufft_del);
}

const struct nlop_s* noir_nufft_batch_create(int Nb, struct noir2_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = noir_nufft_create(model[i]);

	int istack_dims[] = { BATCH_DIM, BATCH_DIM };
	int ostack_dims[] = { BATCH_DIM };

	return nlop_checkpoint_create_F(nlop_stack_multiple_F(Nb, nlops, 2, istack_dims, 1, ostack_dims), false, false);
}

#if 0

static const struct nlop_s* noir_cart_unrolled_batched_create(	int N,
							const long pat_dims[N], const complex float* pattern,
							const long bas_dims[N], const complex float* basis,
							const long msk_dims[N], const complex float* mask,
							const long ksp_dims[N],
							const long cim_dims[N],
							const long img_dims[N],
							const long col_dims[N],
							int Nb,
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

	assert(BATCH_DIM < N);

	auto result = noir_gauss_newton_step_batch_create(&model, &iter_conf, Nb);

	long alp_dims[N];
	md_copy_dims(N, alp_dims, nlop_generic_domain(result, 3)->dims);

	for (int i = 1; i < conf->iter; i++){

		result = nlop_chain2_FF(nlop_from_linop_F(linop_scale_create(N, alp_dims, 1. / conf->redu)), 0, result, 3);
		result = nlop_chain2_swap_FF(noir_gauss_newton_step_batch_create(&model, &iter_conf, Nb), 0, result, 1, 1); // in: y, xn, x0, alpha, y, x0, alpha
		result = nlop_dup_F(result, 0, 4);
		result = nlop_dup_F(result, 2, 4);
		result = nlop_dup_F(result, 3, 4); // in: y, xn, x0, alpha
	}

	complex float alpha = conf->alpha;
	result = nlop_set_input_const_F2(result, 3, N, alp_dims, MD_SINGLETON_STRS(N), true, &alpha);	// in: y, xn, x0

	long reg_dims[2];
	md_copy_dims(2, reg_dims, nlop_generic_domain(result, 2)->dims);

	complex float zero = 0;
	result = nlop_set_input_const_F2(result, 2, 2, reg_dims, MD_SINGLETON_STRS(2), true, &zero);	// in: y, xn

	long size = md_calc_size(N, img_dims) + md_calc_size(N, col_dims);
	long skip = md_calc_size(N, img_dims);

	complex float* init = md_alloc(1, MD_DIMS(size), CFL_SIZE);
	md_clear(1, MD_DIMS(size), init, CFL_SIZE);
	md_zfill(1, MD_DIMS(skip), init, 1.);

	result = nlop_set_input_const_F2(result, 1, 2, reg_dims, MD_DIMS(CFL_SIZE, 0), true, init);	// in: y
	md_free(init);

	const struct nlop_s* nlop_decomp = noir_decomp_batch_create(&model, Nb);

	result = nlop_chain2_FF(result, 0, nlop_decomp, 0);

	complex float scale = 100.;

	long cim_dims2[N];
	long img_dims2[N];

	md_copy_dims(N, cim_dims2, cim_dims);
	md_copy_dims(N, img_dims2, img_dims);

	if (BATCH_DIM < N) {

		cim_dims2[BATCH_DIM] = Nb;
		img_dims2[BATCH_DIM] = Nb;
	}

	result = nlop_chain2_FF(nlop_from_linop_F(linop_scale_create(N, cim_dims2, scale)), 0, result, 0);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_scale_create(N, img_dims, 1. / scale)), 0);

	if (true) {

		result = nlop_chain2_FF(nlop_norm_znorm_create(N, cim_dims2, BATCH_FLAG), 0, result, 0);

		result = nlop_del_out_F(result, 2);

		long sdims[N];
		md_select_dims(N, BATCH_FLAG, sdims, img_dims2);
		result = nlop_chain2_FF(result, 0, nlop_tenmul_create(N, img_dims2, img_dims2, sdims), 0);
		result = nlop_link_F(result, 2, 0);
	}

	long bat_dims[N];
	md_singleton_dims(N, bat_dims);
	if (BATCH_DIM < N)
		bat_dims[BATCH_DIM] = Nb;

	result = nlop_chain2_FF(nlop_from_linop_F(linop_get_adjoint(linop_loop(N, bat_dims, (struct linop_s*)model.lop_fft))), 0, result, 0);

	return result;
}
noir_cart_unrolled_create
const struct nlop_s* (	int N,
							const long pat_dims[N], const complex float* pattern,
							const long bas_dims[N], const complex float* basis,
							const long msk_dims[N], const complex float* mask,
							const long ksp_dims[N],
							const long cim_dims[N],
							const long img_dims[N],
							const long col_dims[N],
							struct noir2_conf_s* conf)
{
	long limg_dims[N];
	long lcol_dims[N];
	long lksp_dims[N];
	long lpat_dims[N];
	long lbas_dims[N];
	long lmsk_dims[N];
	long lcim_dims[N];

	md_select_dims(N, ~BATCH_FLAG, limg_dims, img_dims);
	md_select_dims(N, ~BATCH_FLAG, lcol_dims, col_dims);
	md_select_dims(N, ~BATCH_FLAG, lksp_dims, ksp_dims);
	md_select_dims(N, ~BATCH_FLAG, lpat_dims, pat_dims);
	md_select_dims(N, ~BATCH_FLAG, lbas_dims, bas_dims);
	md_select_dims(N, ~BATCH_FLAG, lmsk_dims, msk_dims);
	md_select_dims(N, ~BATCH_FLAG, lcim_dims, cim_dims);

	return noir_cart_unrolled_batched_create(N, lpat_dims, pattern, lbas_dims, basis, lmsk_dims, mask, lksp_dims, lcim_dims, limg_dims, lcol_dims, (BATCH_DIM < N) ? ksp_dims[BATCH_DIM] : 1, conf);
}

#endif