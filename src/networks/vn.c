#include <assert.h>

#include "iter/italgos.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "misc/mri.h"
#include "nn/nn_ops.h"
#include "nn/init.h"
#include "nn/nn.h"
#include "nn/const.h"
#include "nn/weights.h"
#include "nn/chain.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/rand.h"

#include "iter/proj.h"
#include <math.h>

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter6.h"
#include "iter/batch_gen.h"
#include "iter/monitor_iter6.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"
#include "nlops/stack.h"

#include "nn/layers.h"
#include "nn/losses.h"
#include "nn/rbf.h"

#include "networks/misc.h"
#include "vn.h"

/*
Conventions:

idims = [Ux, Uy, Uz, 1,  Nb] - image dimensions
kdims = [Nx, Ny, Nz, Nc, Nb] - kspace/coil dimensions

Kx, Ky, Kz - dimensions of convolution kernel
Nf - number of filters in convolution kernel
Nc - number of coils
Nw - resolution of field of expert
Nl - number of layers

Input tensors of variational network operator:

kspace:		kdims = (Nx, Ny, Nz, Nc, Nb)
coils:		cdims = (Nx, Ny, Nz, Nc, Nb)
pattern:	pdims = (Nx, Ny, Nz, 1,  Nb/1)

lambda: 	ldims = (1,  Nl)
kernel: 	kerdims = (Nf, Nkx, Nky, Nkz, Nl)
weights: 	wdims = (Nf, Nw , Nl)

Output tensor:

ul:		idims = (Ux, Uy, Uz, 1, Nb)
*/


const struct vn_s vn_default = {

	.Nl = 10,

	.Nf = 24,

	.Kx = 11,
	.Ky = 11,
	.Kz = 1,

	.Nw = 31,
	.Imax = 1.,
	.Imin = -1.,

	.shared_weights = false,

	.weights = NULL,

	.lambda_init = .2,
	.init_scale_mu = 0.04,

	.share_pattern = true,

	.normalize = false,

	.init_tickhonov = false,
	.lambda_fixed_tickhonov = 0.1,

	.low_mem = false,

	.regrid = true,

	.monitor_lambda = false,
};


static struct config_nlop_mri_s get_vn_mri_conf(const struct vn_s* vn)
{
	struct config_nlop_mri_s conf = conf_nlop_mri_simple;
	if (!vn->share_pattern)
		conf.pattern_flags = ~MD_BIT(3);

	conf.regrid = vn->regrid;

	return conf;	
}

/**
 * Returns operator computing the update due to the regularizer
 * 1/Nf \sum_{i=0}^{N_k} K_i^T \Phi'_i(K_i u)
 *
 * Input tensors: 	(u, conv_w, rbf_w)
 *
 * u: 		idims:	(Ux, Uy, Uz, 1, Nb)
 * conv_w:	kerdims:(Nf, Kx, Ky, Kz, 1)
 * rbf_w:	wdims:	(Nf, Nw, 1)
 *
 * Output tensors:
 *
 * Ru:	 	idims:	(Ux, Uy, Uz, 1, Nb)
 */
static nn_t nn_ru_create(const struct vn_s* vn, const long idims[5])
{
	//Padding
	long pad_up[5] = {0, (vn->Kx - 1), (vn->Ky - 1), (vn->Kz - 1), 0};
	long pad_down[5] = {0, -(vn->Kx - 1), -(vn->Ky - 1), -(vn->Kz - 1), 0};
	long ker_size[3] = {vn->Kx, vn->Ky, vn->Kz};

	long Ux = idims[0];
	long Uy = idims[1];
	long Uz = idims[2];
	long Nb = idims[4];

	//working dims
	long idimsw[5] = {1, Ux, Uy, Uz, Nb};
	long zdimsw[5] = {vn->Nf, Ux + 2 * (vn->Kx - 1), Uy + 2 * (vn->Ky - 1), Uz + 2 * (vn->Kz - 1), Nb};
	long rbfdims[3] = {vn->Nf, (Ux + 2 * (vn->Kx - 1)) * (Uy + 2 * (vn->Ky - 1)) * (Uz + 2 * (vn->Kz - 1)) * Nb, vn->Nw};

	//operator dims
	long kerdims[5] = {vn->Nf, vn->Kx, vn->Ky, vn->Kz, 1};
	long wdims[3] = {vn->Nf, vn->Nw, 1};

	const struct nlop_s* nlop_result = nlop_from_linop_F(linop_reshape_create(5, idimsw, 5, idims)); // in: u
	//nlop_result = nlop_chain2_FF(nlop_result, 0, padu, 0); // in: u
	nlop_result = append_padding_layer(nlop_result, 0, 5, pad_up, pad_up, PAD_SYMMETRIC);
	nlop_result = append_convcorr_layer(nlop_result, 0, vn->Nf, ker_size, false, PAD_SAME, true, NULL, NULL); // in: u, conv_w

	const struct nlop_s* rbf = nlop_activation_rbf_create(rbfdims, vn->Imax, vn->Imin);
	rbf = nlop_reshape_in_F(rbf, 0, 5, zdimsw);
	rbf = nlop_reshape_out_F(rbf, 0, 5, zdimsw);
	nlop_result = nlop_chain2_FF(nlop_result, 0, rbf, 0); //in: rbf_w, in, conv_w

	nlop_result = append_transposed_convcorr_layer(nlop_result, 0, 1, ker_size, false, true, PAD_SAME, true, NULL, NULL); //in: rbf_w, u, conv_w, conv_w
	//nlop_result = nlop_chain2_FF(nlop_result, 0, padd, 0); //in: rbf_w, u, conv_w, conv_w
	nlop_result = append_padding_layer(nlop_result, 0, 5, pad_down, pad_down, PAD_VALID);
	nlop_result = nlop_dup_F(nlop_result, 2, 3); //in: rbf_w, u, conv_w

	nlop_result = nlop_reshape_out_F(nlop_result, 0, 5, idims); //in: rbf_w, u, conv_w
	nlop_result = nlop_reshape_in_F(nlop_result, 2, 5, kerdims); //in: rbf_w, u, conv_w
	nlop_result = nlop_reshape_in_F(nlop_result, 0, 3, wdims); //in: rbf_w, u, conv_w

	//VN implementation: u_k = (real(up) * real(k) + imag(up) * imag(k))
	nlop_result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(5, kerdims)), 0, nlop_result, 2); //in: rbf_w, u, conv_w
	nlop_result = nlop_chain2_FF(nlop_result, 0, nlop_from_linop_F(linop_scale_create(5, idims, 1. / vn->Nf)), 0); //in: rbf_w, u, conv_w

	int perm [] = {1, 2, 0};
	nlop_result = nlop_permute_inputs_F(nlop_result, 3, perm); //in: u, conv_w, rbf_w

	auto nn_result = nn_from_nlop_F(nlop_result);
	nn_result = nn_set_input_name_F(nn_result, 0, "u");
	nn_result = nn_set_input_name_F(nn_result, 0, "conv_w");
	nn_result = nn_set_input_name_F(nn_result, 0, "rbf_w");

	nn_result = nn_set_initializer_F(nn_result, 0, "conv_w", init_std_normal_create(false, 1. / sqrtf((float)vn->Kx * vn->Ky * vn->Kz), 0));
	nn_result = nn_set_initializer_F(nn_result, 0, "rbf_w", init_linspace_create(1., vn->Imin * vn->init_scale_mu, vn->Imax * vn->init_scale_mu, true));
	nn_result = nn_set_in_type_F(nn_result, 0, "conv_w", IN_OPTIMIZE);
	nn_result = nn_set_in_type_F(nn_result, 0, "rbf_w", IN_OPTIMIZE);

	nn_result = nn_set_output_name_F(nn_result, 0, "Ru");

	nn_debug(DP_DEBUG3, nn_result);

	return nn_checkpoint_F(nn_result, true, vn->low_mem);
}

/**
 * Returns operator computing update for data fidelity
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param idims (Ux, Uy, Uz, 1, Nb)
 *
 * Input tensors:
 * u:		idims: 	(Ux, Uy, Uz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * lambda_w:	dims:	(1)
 *
 * Output tensors:
 * Du:		idims: 	(Ux, Uy, Uz, 1,  Nb)
 */
static nn_t nn_du_create(const struct vn_s* vn, const long dims[5], const long idims[5])
{
	struct config_nlop_mri_s conf = get_vn_mri_conf(vn);
	
	const struct nlop_s* nlop_result = nlop_mri_gradient_step_create(5, dims, idims, &conf);

	nlop_result = nlop_chain2_swap_FF(nlop_result, 0, nlop_tenmul_create(5, idims, idims, MD_SINGLETON_DIMS(5)), 0);
	nlop_result = nlop_reshape_in_F(nlop_result, 4, 1, MD_SINGLETON_DIMS(1));

	auto nn_result = nn_from_nlop_F(nlop_result);

	nn_result = nn_set_input_name_F(nn_result, 0, "u");
	nn_result = nn_set_input_name_F(nn_result, 0, "kspace");
	nn_result = nn_set_input_name_F(nn_result, 0, "coil");
	nn_result = nn_set_input_name_F(nn_result, 0, "pattern");
	nn_result = nn_set_input_name_F(nn_result, 0, "lambda_w");

	nn_result = nn_set_initializer_F(nn_result, 0, "lambda_w", init_const_create(vn->lambda_init));
	nn_result = nn_set_in_type_F(nn_result, 0, "lambda_w", IN_OPTIMIZE);

	nn_result = nn_set_output_name_F(nn_result, 0, "Du");

	nn_debug(DP_DEBUG3, nn_result);

	return nn_result;
}


/**
 * Returns cell of variational network
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param idims (Ux, Uy, Uz, 1, Nb)
 *
 * Input tensors:
 * u(t):	idims:	(Ux, Uy, Uz, 1,  Nb)
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * conv_w(t):	kerdims:(Nf, Kx, Ky, Kz, 1)
 * rbf_w(t):	wdims:	(Nf, Nw, 1)
 * l(t):	ldims:	(1,  1)
 *
 * Output tensors:
 * u(t+1):	idims:	(Ux, Uy, Uz, 1, Nb)
 */
static nn_t nn_vn_cell_create(const struct vn_s* vn, const long dims[5], const long idims[5])
{
	auto ru = nn_ru_create(vn, idims); //in: u(t), conv_w, rbf_w
	auto du = nn_du_create(vn, dims, idims); //in: u(t), kspace, coil, pattern, lambda
	du = nn_append_singleton_dim_in_F(du, 0, "lambda_w");
	ru = nn_mark_dup_F(ru, "u");

	auto result = nn_combine_FF(du, ru); // in: u(t), kspace, coil, pattern, lambda, u(t), conv_w, rbf_w; out: du, ru
	result = nn_stack_dup_by_name_F(result); // in: u(t), lambda, kspace, coil, pattern, conv_w, rbf_w; out: du, ru

	result = nn_combine_FF(nn_from_nlop_F(nlop_zaxpbz_create(5, idims, 1., 1.)), result); // in: du, ru, u(t), kspace, coil, pattern, lambda, conv_w, rbf_w; out: du+ru, du, ru
	result = nn_link_F(result, 0, "Du", 0, NULL);
	result = nn_link_F(result, 0, "Ru", 0, NULL);// in: u(t), lambda, kspace, coil, pattern, conv_w, rbf_w; out: du+ru

	auto residual = nn_from_nlop_F(nlop_zaxpbz_create(5, idims, 1., -1.));
	residual = nn_set_input_name_F(residual, 0, "u");

	result = nn_mark_dup_F(result, "u");

	result = nn_combine_FF(residual, result); // in: u(t), (du+ru), u(t), kspace, coil, pattern, lambda, conv_w, rbf_w; out: u(t+1)=u(t)-(du+ru), du+ru
	result = nn_stack_dup_by_name_F(result); // in: u(t), (du+ru), lambda, kspace, coil, pattern, conv_w, rbf_w; out: u(t+1), (du+ru)
	result = nn_link_F(result, 1, NULL, 0, NULL); // in: u(t), kspace, coil, pattern, lambda, conv_w, rbf_w; out: u(t+1)

	result = nn_sort_inputs_by_list_F(result, 7,
		(const char*[7]) {
			"u",
			"kspace",
			"coil",
			"pattern",
			"conv_w",
			"rbf_w",
			"lambda_w"
			}
		); // in: u(t), kspace, coil, pattern, conv_w, rbf_w, lambda; out: u(t+1)

	nn_debug(DP_DEBUG3, result);

	return result;
}

/**
 * Returns operator computing [normalized] zerofilled reconstruction
 * [and normalization scale]
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param idims (Ux, Uy, Uz, 1, Nb)
 *
 * Input tensors:
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 *
 * Output tensors:
 * u(0):	idims:	(Ux, Uy, Uz, 1,  Nb)
 * [normalize_scale: 	(1,  1,  1,  1,  Nb)]
 */
static nn_t nn_vn_zf_create(const struct vn_s* vn, const long dims[5], const long idims[5])
{
	struct config_nlop_mri_s conf = get_vn_mri_conf(vn);

	auto nlop_zf = nlop_mri_adjoint_create(5, dims, idims, &conf);
	auto nn_zf = nn_from_nlop_F(nlop_zf);
	nn_zf = nn_set_input_name_F(nn_zf, 0, "kspace");
	nn_zf = nn_set_input_name_F(nn_zf, 0, "coil");
	nn_zf = nn_set_input_name_F(nn_zf, 0, "pattern");

	if (vn->init_tickhonov) {

		struct iter_conjgrad_conf def_conf = iter_conjgrad_defaults;
		def_conf.l2lambda = 1.;
		def_conf.maxiter = 20;

		conf.iter_conf = &def_conf;
		conf.lambda_fixed = vn->lambda_fixed_tickhonov;
		assert(0 <= conf.lambda_fixed);

		auto nlop_dc = mri_normal_inversion_create(5, dims, idims, &conf);
		
		auto nn_dc = nn_from_nlop_F(nlop_dc);
		nn_dc = nn_set_input_name_F(nn_dc, 1, "coil");
		nn_dc = nn_set_input_name_F(nn_dc, 1, "pattern");

		nn_dc = nn_mark_dup_F(nn_dc, "coil");
		nn_dc = nn_mark_dup_F(nn_dc, "pattern");

		nn_zf = nn_chain2(nn_zf, 0, NULL, nn_dc, 0, NULL);
	}

	if (vn->normalize) {

		auto nn_normalize = nn_from_nlop_F(nlop_norm_zmax_create(5, idims, MD_BIT(4), true));
		nn_normalize = nn_set_output_name_F(nn_normalize, 1, "normalize_scale");
		nn_zf = nn_chain2_FF(nn_zf, 0, NULL, nn_normalize, 0, NULL);
	}

	nn_zf = nn_set_output_name_F(nn_zf, 0, "u0");

	return nn_zf;
}

/**
 * Returns operator representing the variational network
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param idims (Ux, Uy, Uz, 1, Nb)
 *
 * Input tensors:
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * conv_w[0:1]:	kerdims:(Nf, Kx, Ky, Kz, Nl)
 * rbf_w[0:1]:	wdims:	(Nf, Nw , Nl)
 * l[0:1]:	ldims:	(1,  Nl)
 *
 * Output tensors:
 * u(l):	idims:	(Ux, Uy, Uz, 1,  Nb)
 * [normalize_scale: 	(1,  1,  1,  1,  Nb)]
 */
static nn_t nn_vn_create(const struct vn_s* vn, const long dims[5], const long idims[5])
{
	auto result = nn_vn_cell_create(vn, dims, idims);

	for (int l = 1; l < vn->Nl; l++) {

		auto tmp = nn_vn_cell_create(vn, dims, idims);
		tmp = nn_mark_dup_F(tmp, "kspace");
		tmp = nn_mark_dup_F(tmp, "coil");
		tmp = nn_mark_dup_F(tmp, "pattern");
		if (vn->shared_weights) {

			tmp = nn_mark_dup_F(tmp, "conv_w");
			tmp = nn_mark_dup_F(tmp, "rbf_w");
			tmp = nn_mark_dup_F(tmp, "lambda_w");
		
		} else {
			tmp = nn_mark_stack_input_F(tmp, "conv_w");
			tmp = nn_mark_stack_input_F(tmp, "rbf_w");
			tmp = nn_mark_stack_input_F(tmp, "lambda_w");
		}
		tmp = nn_rename_input_F(tmp, "u_tmp", "u");

		result = nn_chain2_FF(result, 0, NULL, tmp, 0, "u_tmp"); //in: kspace, coil, pattern, conv_w[l], rbf_w[l], lambda[l], u(0), kspace, coil, pattern, conv_w[0:l-1], rbf_w[0:l-1], lambda[0:l-1]
		result = nn_stack_dup_by_name_F(result);

		result = nn_sort_inputs_by_list_F(result, 8,
		(const char*[8]) {
			"u",
			"kspace",
			"coil",
			"pattern",
			"conv_w",
			"rbf_w",
			"lambda_w",
			"lambda_0"
			}
		);
	}

	auto nn_zf = nn_vn_zf_create(vn, dims, idims);
	nn_zf = nn_mark_dup_F(nn_zf, "kspace");
	nn_zf = nn_mark_dup_F(nn_zf, "coil");
	nn_zf = nn_mark_dup_F(nn_zf, "pattern");

	if (vn->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, dims);

		const struct nlop_s* scale = nlop_tenmul_create(5, dims, dims, sdims);
		scale = nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, scale, 1);
		scale = nlop_combine_FF(scale, nlop_from_linop_F(linop_identity_create(5, sdims)));
		scale = nlop_dup_F(scale, 1, 2);

		auto nn_scale = nn_from_nlop_F(scale);
		nn_scale = nn_set_input_name_F(nn_scale, 0, "kspace");
		nn_scale = nn_set_input_name_F(nn_scale, 0, "normalize_scale");
		nn_scale = nn_set_output_name_F(nn_scale, 1, "normalize_scale");

		result = nn_rename_input_F(result, "kspace_normalized", "kspace");
		nn_zf = nn_rename_output_F(nn_zf, "normalize_scale_tmp", "normalize_scale");

		result = nn_chain2_FF(nn_scale, 0, NULL, result, 0, "kspace_normalized");
		result = nn_chain2_FF(nn_zf, 0, "normalize_scale_tmp", result, 0, "normalize_scale");
	} else {

		result = nn_combine_FF(result, nn_zf);
	}

	result = nn_link_F(result, 0, "u0", 0, "u");
	result = nn_stack_dup_by_name_F(result);

	result = nn_sort_inputs_by_list_F(result, 7,
		(const char*[8]) {
			"kspace",
			"coil",
			"pattern",
			"conv_w",
			"rbf_w",
			"lambda_w",
			"lambda_0"
			}
		);

	nn_debug(DP_DEBUG1, result);

	return result;
}

/**
 * Returns which applies the variational network
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param idims (Ux, Uy, Uz, 1, Nb)
 *
 * Input tensors:
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 *
 * Output tensors:
 * u(l):	idims:	(Ux, Uy, Uz, 1,  Nb)
 */
static const struct nlop_s* nlop_vn_apply_create(struct vn_s* vn, const long dims[5], const long idims[5])
{
	auto nn_apply = nn_vn_create(vn, dims, idims);

	if (vn->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, dims);
		auto nn_norm_ref = nn_from_nlop_F(nlop_tenmul_create(5, idims, idims, sdims));

		nn_apply = nn_chain2_FF(nn_apply, 0, NULL, nn_norm_ref, 0, NULL);
		nn_apply = nn_link_F(nn_apply, 0, "normalize_scale", 0, NULL);
	}

	return nn_get_nlop_wo_weights_F(nn_apply, vn->weights, false);
}



/**
 * Creates Variational Network and applies it
 *
 * @param vn structure describing the variational network
 * @param idims (Ux, Uy, Uz, 1, Nb) - image dims
 * @param out pointer to output array
 * @param kdims (Nx, Ny, Nz, Nc, Nb) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param coils pointer to coil data
 * @param pdims (Nx, Ny, Nz, 1, 1 / Nb) - dims of pattern
 * @param pattern pointer to pattern data
 */
void apply_vn(	struct vn_s* vn,
		const long idims[5], complex float* out,
		const long kdims[5], const complex float* kspace,
		const long cdims[5], const complex float* coil,
		const long pdims[5], const complex float* pattern)
{

	vn->share_pattern = (1 == pdims[4]);

	auto network = nlop_vn_apply_create(vn, kdims, idims);

	complex float* out_tmp = md_alloc_sameplace(5, idims, CFL_SIZE, vn->weights->tensors[0]);
	complex float* kspace_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, vn->weights->tensors[0]);
	complex float* coil_tmp = md_alloc_sameplace(5, cdims, CFL_SIZE, vn->weights->tensors[0]);
	complex float* pattern_tmp = md_alloc_sameplace(5, pdims, CFL_SIZE, vn->weights->tensors[0]);

	md_copy(5, kdims, kspace_tmp, kspace, CFL_SIZE);
	md_copy(5, cdims, coil_tmp, coil, CFL_SIZE);
	md_copy(5, pdims, pattern_tmp, pattern, CFL_SIZE);

	void* refs[] = {(void*)out_tmp, (void*)kspace_tmp, (void*)coil_tmp, (void*)pattern_tmp};
	nlop_generic_apply_select_derivative_unchecked(network, 4, refs, 0, 0);

	md_copy(5, idims, out, out_tmp, CFL_SIZE);

	nlop_free(network);
	md_free(out_tmp);
	md_free(kspace_tmp);
	md_free(coil_tmp);
	md_free(pattern_tmp);
}


/**
 * Creates Variational Network and applies it batchwise
 *
 * @param vn structure describing the variational network
 * @param idims (Ux, Uy, Uz, 1, Nt) - image dims
 * @param out pointer to output array
 * @param kdims (Nx, Ny, Nz, Nc, Nt) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param coils pointer to coil data
 * @param pdims (Nx, Ny, Nz, 1, 1 / Nt) - dims of pattern
 * @param pattern pointer to pattern data
 * @param Nb batch size
 */
void apply_vn_batchwise(	struct vn_s* vn,
				const long idims[5], complex float * out,
				const long kdims[5], const complex float* kspace,
				const long cdims[5], const complex float* coil,
				const long pdims[5], const complex float* pattern,
				long Nb)
{
	long Nt = kdims[4];
	while (0 < Nt) {

		long kdims1[5];
		long cdims1[5];
		long idims1[5];
		long pdims1[5];

		md_copy_dims(5, kdims1, kdims);
		md_copy_dims(5, cdims1, cdims);
		md_copy_dims(5, idims1, idims);
		md_copy_dims(5, pdims1, pdims);

		long Nb_tmp = MIN(Nt, Nb);

		kdims1[4] = Nb_tmp;
		cdims1[4] = Nb_tmp;
		idims1[4] = Nb_tmp;
		pdims1[4] = MIN(pdims1[4], Nb_tmp);

		apply_vn(vn, idims1, out, kdims1, kspace, cdims1, coil, pdims1, pattern);

		out += md_calc_size(5, idims1);
		kspace += md_calc_size(5, kdims1);
		coil += md_calc_size(5, cdims1);
		if (1 < pdims[4])
			pattern += md_calc_size(5, pdims1);

		Nt -= Nb_tmp;
	}
}

/**
 * Returns operator computing the training loss
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param idims (Ux, Uy, Uz, 1, Nb)
 *
 * Input tensors:
 * u_ref:	idims:	(Ux, Uy, Uz, 1,  Nb)
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * conv_w[0:1]:	kerdims:(Nf, Kx, Ky, Kz, Nl)
 * rbf_w[0:1]:	wdims:	(Nf, Nw , Nl)
 * l[0:1]:	ldims:	(1,  Nl)
 *
 * Output tensors:
 * loss:	dims:	(1)
 */
static nn_t vn_train_op_create(const struct vn_s* vn, const long dims[5], const long idims[5])
{
	auto nn_train = nn_vn_create(vn, dims, idims);

	//append loss = 1/(2N) sum_i^N || |x_i| - |y_i| ||^2 with |x_i| = sqrt(Re[x_i]^2 + Im[x_i]^2 + epsilon)
	const struct nlop_s* loss = nlop_mse_create(5, idims, ~0ul);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, idims, 1.e-12), 0, loss, 0);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, idims, 1.e-12), 0, loss, 0);

	if(vn->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, dims);

		auto nn_norm_ref = nn_from_nlop_F(nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, nlop_tenmul_create(5, idims, idims, sdims), 1));

		nn_train = nn_chain2_FF(nn_train, 0, "normalize_scale", nn_norm_ref, 1, NULL);
		nn_train = nn_chain2_FF(nn_train, 1, NULL, nn_from_nlop_F(loss), 1, NULL);
		nn_train = nn_link_F(nn_train, 1, NULL, 0, NULL);
		nn_train = nn_set_out_type_F(nn_train, 0, NULL, OUT_OPTIMIZE);

	} else {

		nn_train = nn_chain2_FF(nn_train, 0, NULL, nn_from_nlop_F(loss), 1, NULL);
		nn_train = nn_set_out_type_F(nn_train, 0, NULL, OUT_OPTIMIZE);
	}

	return nn_train;
}

/**
 * Returns operator computing the validation loss
 *
 * @param vn structure describing the variational network
 * @param valid_files struct holding validation data 
 *
 * Input tensors:
 * u_ref:	idims:	(Ux, Uy, Uz, 1,  Nb) [ignored]
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb) [ignored]
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb) [ignored]
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb) [ignored]
 * conv_w[0:1]:	kerdims:(Nf, Kx, Ky, Kz, Nl)
 * rbf_w[0:1]:	wdims:	(Nf, Nw , Nl)
 * l[0:1]:	ldims:	(1,  Nl)
 *
 * Output tensors:
 * mse (mag):	dims:	(1)
 * mse:		dims:	(1)
 */
static nn_t vn_valid_loss_create(struct vn_s* vn, struct network_data_s* vf)
{
	load_network_data(vf);

	vn->share_pattern = (vf->pdims[4] == 1);

	auto valid_loss = nn_vn_create(vn, vf->kdims, vf->idims);

	const struct nlop_s* loss = nlop_combine_FF(nlop_mse_create(5, vf->idims, ~0ul), nlop_mse_create(5, vf->idims, ~0ul));
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, vf->idims, 1.e-12), 0, loss, 0);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, vf->idims, 1.e-12), 0, loss, 0);
	loss = nlop_dup_F(loss, 0, 2);
	loss = nlop_dup_F(loss, 1, 2);

	loss = nlop_combine_FF(loss, nlop_mpsnr_create(5, vf->idims, MD_BIT(4)));
	loss = nlop_dup_F(loss, 0, 2);
	loss = nlop_dup_F(loss, 1, 2);

	loss = nlop_combine_FF(loss, nlop_mssim_create(5, vf->idims, MD_DIMS(7, 7, 1, 1, 1), 7));
	loss = nlop_dup_F(loss, 0, 2);
	loss = nlop_dup_F(loss, 1, 2);

	if(vn->normalize) {

		long sdims[5];
		md_select_dims(5, MD_BIT(4), sdims, vf->kdims);

		auto nn_norm_ref = nn_from_nlop_F(nlop_chain2_FF(nlop_zinv_create(5, sdims), 0, nlop_tenmul_create(5, vf->idims, vf->idims, sdims), 1));

		valid_loss = nn_chain2_FF(valid_loss, 0, "normalize_scale", nn_norm_ref, 1, NULL);
		valid_loss = nn_chain2_FF(valid_loss, 1, NULL, nn_from_nlop_F(loss), 1, NULL);
		valid_loss = nn_link_F(valid_loss, 4, NULL, 0, NULL);
		valid_loss = nn_set_out_type_F(valid_loss, 0, NULL, OUT_OPTIMIZE);

	} else {

		valid_loss = nn_chain2_FF(valid_loss, 0, NULL, nn_from_nlop_F(loss), 1, NULL);
		valid_loss = nn_set_out_type_F(valid_loss, 0, NULL, OUT_OPTIMIZE);
	}

	valid_loss = nn_ignore_input_F(valid_loss, 0, NULL, 5, vf->idims, true, vf->out);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "kspace", 5, vf->kdims, true, vf->kspace);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "coil", 5, vf->cdims, true, vf->coil);
	valid_loss = nn_ignore_input_F(valid_loss, 0, "pattern", 5, vf->pdims, true, vf->pattern);

	free_network_data(vf);

	return valid_loss;
}



/**
 * Trains Variational Network
 *
 * @param vn structure describing the variational network
 * @param train_conf structure holding training parameters of iPALM algorithm
 * @param idims (Ux, Uy, Uz, 1, Nb) - image dims of reference data
 * @param ref pointer to reference
 * @param kdims (Nx, Ny, Nz, Nc, Nb) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param cdims (Ux, Uy, Uz, Nc, Nb) - dims of kspace and coils
 * @param coils pointer to coil data
 * @param pdims (Nx, Ny, Nz, 1, 1 / Nb) - dims of pattern
 * @param pattern pointer to pattern data
 * @param Nb batch size for training
 * @param valid_files struct holding validation data 
 */
void train_vn(	struct vn_s* vn, struct iter6_conf_s* train_conf,
			const long idims[5], complex float* ref,
			const long kdims[5], complex float* kspace,
			const long cdims[5], const complex float* coil,
			const long pdims[5], const complex float* pattern,
			long Nb, struct network_data_s* valid_files)
{
	long Nt = kdims[4]; // number datasets

	long nkdims[5];
	long nidims[5];
	long ncdims[5];

	md_copy_dims(5, nkdims, kdims);
	md_copy_dims(5, nidims, idims);
	md_copy_dims(5, ncdims, cdims);

	nkdims[4] = Nb;
	nidims[4] = Nb;
	ncdims[4] = Nb;

	vn->share_pattern = (1 == pdims[4]);

	auto nn_train = vn_train_op_create(vn, nkdims, nidims);

	//create batch generator
	const complex float* train_data[] = {ref, kspace, coil, pattern};
	const long* train_dims[] = {	nn_generic_domain(nn_train, 0, NULL)->dims,
					nn_generic_domain(nn_train, 0, "kspace")->dims,
					nn_generic_domain(nn_train, 0, "coil")->dims,
					nn_generic_domain(nn_train, 0, "pattern")->dims};
	
	assert(md_check_equal_dims(5, ncdims, train_dims[2], ~0));

	auto batch_generator = batch_gen_create_from_iter(train_conf, 4, 5, train_dims, train_data, Nt, 0);

	//setup for iter algorithm
	int NI = nn_get_nr_in_args(nn_train);
	int NO = nn_get_nr_out_args(nn_train);

	float* src[NI];

	for (int i = 0; i < vn->weights->N; i++)
		src[i + 4] = (float*)vn->weights->tensors[i];

	enum IN_TYPE in_type[NI];
	const struct operator_p_s* projections[NI];
	enum OUT_TYPE out_type[NO];

	nn_get_in_types(nn_train, NI, in_type);
	nn_get_out_types(nn_train, NO, out_type);

	for (int i = 0; i < 4; i++) {

		src[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
		projections[i] = NULL;
	}

	for (int i = 4; i < NI; i++)
		projections[i] = NULL;

	auto conv_iov = nn_generic_domain(nn_train, 0, "conv_w");
	projections[4] = operator_project_mean_free_sphere_create(conv_iov->N, conv_iov->dims, MD_BIT(0) | MD_BIT(4), false);
	auto lambda_iov = nn_generic_domain(nn_train, 0, "lambda_w");
	projections[6] = operator_project_pos_real_create(lambda_iov->N, lambda_iov->dims);

	const struct monitor_value_s* value_monitors[2];
	int num_monitors = 0;

	if (NULL != valid_files) {

		auto nn_validation_loss = vn_valid_loss_create(vn, valid_files);
		value_monitors[0] = monitor_iter6_nlop_create(nn_get_nlop(nn_validation_loss), false, 4, (const char*[4]){"val loss (mag)", "val loss", "mean PSNR", "ssim"});
		nn_free(nn_validation_loss);
		num_monitors += 1;

	} else {

		value_monitors[0] = NULL;
	}

	if (vn->monitor_lambda){

		int num_lambda = vn->shared_weights ? 1 : vn->Nl;

		const char* lam = "lam";
		const char* lams[num_lambda];

		for (int i = 0; i < num_lambda; i++)
			lams[i] = lam;
		
		auto destack_lambda = nlop_from_linop_F(linop_identity_create(2, MD_DIMS(1, num_lambda)));
		for (int i = num_lambda - 1; 0 < i; i--)
			destack_lambda = nlop_chain2_FF(destack_lambda, 0, nlop_destack_create(2, MD_DIMS(1, i), MD_DIMS(1, 1), MD_DIMS(1, i + 1), 1), 0);
		for(int i = 0; i < 6; i++)
			destack_lambda = nlop_combine_FF(nlop_del_out_create(1, MD_DIMS(1)), destack_lambda);


		value_monitors[num_monitors] = monitor_iter6_nlop_create(destack_lambda, true, num_lambda, lams);
		num_monitors += 1;
	}


	struct monitor_iter6_s* monitor = monitor_iter6_create(true, true, num_monitors, value_monitors);

	debug_printf(DP_INFO, "Train VarNet\n");
	nn_debug(DP_INFO, nn_train);

	iter6_iPALM(train_conf, nn_get_nlop(nn_train), NI, in_type, projections, src, 1, out_type, 0, Nt / Nb, batch_generator, monitor);
	nn_free(nn_train);
	nlop_free(batch_generator);

	monitor_iter6_free(monitor);
}

/**
 * Initialize weights of variational network
 *
 * @param vn structure describing the variational network
 */
void init_vn(struct vn_s* vn)
{
	auto network = nn_vn_create(vn, MD_DIMS(vn->Kx, vn->Ky, vn->Kz, 1, 1), MD_DIMS(vn->Kx, vn->Ky, vn->Kz, 1, 1));
	vn->weights = nn_weights_create_from_nn(network);
	nn_init(network, vn->weights);
	nn_free(network);
}

/**
 * Load weights of variational network from file
 *
 * @param vn structure describing the variational network
 */
void load_vn(struct vn_s* vn, const char* filename, bool overwrite_pars)
{
	vn->weights = load_nn_weights(filename);

	if (overwrite_pars){

		vn->Nf = vn->weights->iovs[0]->dims[0];
		vn->Kx = vn->weights->iovs[0]->dims[1];
		vn->Ky = vn->weights->iovs[0]->dims[2];
		vn->Kz = vn->weights->iovs[0]->dims[3];
		if (!vn->shared_weights)
			vn->Nl = vn->weights->iovs[0]->dims[4];
		vn->Nw = vn->weights->iovs[1]->dims[1];
	}
}
