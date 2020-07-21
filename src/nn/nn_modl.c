#include <assert.h>
#include <float.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

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

#include "iter/iter6.h"
#include "iter/batch_gen.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"
#include "nlops/const.h"

#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/losses.h"
#include "nn/init.h"
#include "nn/vn.h"

#include "nn_modl.h"

/*
Conventions:

Ux, Uy, Uz - dimensions, kspace dimensions
Nkx, Nky, Nkz - dimensions of convolution kernel
Nf - number of filters in convolution kernel
Nc - number of coils
Nw - resolution of field of expert
Nl - number of layers

dims = (Nx, Ny, Nz, Nc, Nb)

Input tensors:

u0:		udims = (Nx, Ny, Nz, Nb)
kspace:		kdims = (Nx, Ny, Nz, Nc, Nb)
coils:		cdims = (Nx, Ny, Nz, Nc, Nb)
mask:		mdims = (Nx, Ny, Nz, 1, 1)

lambda: 	ldims = (1,  Nl)
kernel: 	kerdims = (Nf, Nkx, Nky, Nkz, Nl)
weights: 	wdims = (Nf, Nw , Nl)

Output tensor:

ul:		udims = (Nx, Ny, Nz, Nb)
*/


const struct modl_s modl_default = {

	.Nb = 10,

	.Nt = 2,
	.Nl = 5,
	.Nf = 32,

	.Kx = 3,
	.Ky = 3,
	.Kz = 1,

	.lambda = NULL,

	.conv_0 = NULL,
	.conv_i = NULL,
	.conv_n = NULL,

	.bias_0 = NULL,
	.bias_i = NULL,
	.bias_n = NULL,

	.gamma_n = NULL,

	.bn_0 = NULL,
	.bn_i = NULL,
	.bn_n = NULL,

	.lambda_init = .05,
	.shared_weights = true,
	.shared_lambda = true,
	.share_mask = false,

	.lambda_min = 0.,
	.lambda_max = FLT_MAX,

	.lambda_fixed = -1.,

	.nullspace = false,
};

static const struct nlop_s* nlop_dw_first_layer(const struct modl_s* config, long udims[5])
{
	long udims_w[5] = {1, udims[0], udims[1], udims[2], udims[4]};

	const struct nlop_s* result = nlop_from_linop_F(linop_reshape_create(5, udims_w, 5, udims));

	result = append_convcorr_layer(result, 0, config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL); //in: xn, conv_w; out; xn'
	result = append_batchnorm_layer(result, 0, ~MD_BIT(0)); //in: xn, conv0, bn0_in; out: xn', bn0_out
	result = append_activation_bias(result, 0, ACT_RELU, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0; out: xn', bn0_out

	return result;
}

static const struct nlop_s* nlop_dw_append_center_layer(const struct modl_s* config, const struct nlop_s* nlop_dw)
{
	assert(0 < config->Nl - 2);

	nlop_dw = append_convcorr_layer(nlop_dw, 0, config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL); //in: xn, conv0, bn0_in, bias0, convi; out: xn', bn0_out
	nlop_dw = append_batchnorm_layer(nlop_dw, 0, ~MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in; out: xn', bn0_out, bni_out
	nlop_dw = append_activation_bias(nlop_dw, 0, ACT_RELU, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out

	nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 4);
	nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 5);
	nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 6);
	nlop_dw = nlop_append_singleton_dim_out_F(nlop_dw, 2);


	for (int i = 0; i < config->Nl - 3; i++) {

		nlop_dw = append_convcorr_layer(nlop_dw, 0, config->Nf, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL);
		//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convi'; out: xn', bn0_out, bni_out
		nlop_dw = append_batchnorm_layer(nlop_dw, 0, ~MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convi', bni_in'; out: xn', bn0_out, bni_out, bni_out'
		nlop_dw = append_activation_bias(nlop_dw, 0, ACT_RELU, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convi', bni_in', biasi_in'; out: xn', bn0_out, bni_out, bni_out'

		nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 7);
		nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 8);
		nlop_dw = nlop_append_singleton_dim_in_F(nlop_dw, 9);
		nlop_dw = nlop_append_singleton_dim_out_F(nlop_dw, 3);

		nlop_dw = nlop_stack_inputs_F(nlop_dw, 4, 7, nlop_generic_domain(nlop_dw, 4)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, bni_in', biasi_in'; out: xn', bn0_out, bni_out, bni_out'
		nlop_dw = nlop_stack_inputs_F(nlop_dw, 5, 7, nlop_generic_domain(nlop_dw, 5)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, biasi_in'; out: xn', bn0_out, bni_out, bni_out'
		nlop_dw = nlop_stack_inputs_F(nlop_dw, 6, 7, nlop_generic_domain(nlop_dw, 6)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out, bni_out'

		nlop_dw = nlop_stack_outputs_F(nlop_dw, 2, 3, nlop_generic_codomain(nlop_dw, 2)->N - 1);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out
	}

	return nlop_dw;
}

static const struct nlop_s* nlop_dw_append_last_layer(const struct modl_s* config, const struct nlop_s* nlop_dw)
{
	assert(0 < config->Nl - 2);

	//nlop_dw in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi; out: xn', bn0_out, bni_out

	nlop_dw = append_convcorr_layer(nlop_dw, 0, 1, MAKE_ARRAY(config->Kx, config->Ky, config->Kz), false, PAD_SAME, true, NULL, NULL); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn; out: xn', bn0_out, bni_out
	nlop_dw = append_batchnorm_layer(nlop_dw, 0, ~MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in; out: xn', bn0_out, bni_out, bnn_out

	//append gamma for batchnorm
	long N_out = nlop_generic_codomain(nlop_dw, 0)->N;
	long odims[N_out];
	md_copy_dims(N_out, odims, nlop_generic_codomain(nlop_dw, 0)->dims);
	long gdims[N_out];
	md_select_dims(N_out, MD_BIT(0), gdims, nlop_generic_codomain(nlop_dw, 0)->dims);
	auto nlop_gamma_mul = nlop_tenmul_create(N_out, odims, odims, gdims);

	nlop_dw = nlop_chain2_swap_FF(nlop_dw, 0, nlop_gamma_mul, 0); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman; out: xn', bn0_out, bni_out, bnn_out
	nlop_dw = append_activation_bias(nlop_dw, 0, ACT_LIN, MD_BIT(0)); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn', bn0_out, bni_out, bnn_out

	const long* udims_r = nlop_generic_codomain(nlop_dw, 0)->dims;
	long udims[5] = {udims_r[1], udims_r[2], udims_r[3], 1, udims_r[4]};
	nlop_dw = nlop_reshape_out_F(nlop_dw, 0, 5, udims);

	return nlop_dw;
}

static const struct nlop_s* nlop_dw_create(const struct modl_s* config, long udims[5])
{
	assert(0 < config->Nl - 2);

	auto nlop_dw = nlop_dw_first_layer(config, udims);
	nlop_dw = nlop_dw_append_center_layer(config, nlop_dw);
	nlop_dw = nlop_dw_append_last_layer(config, nlop_dw);

	nlop_dw = nlop_chain2_FF(nlop_dw, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0); //in: xn, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: zn, bn0_out, bni_out, bnn_out
	nlop_dw = nlop_dup_F(nlop_dw, 0, 1); //in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn + xn', bn0_out, bni_out, bnn_out

	nlop_dw = nlop_chain2_swap_FF(nlop_dw, 0, nlop_tenmul_create(5, udims, udims, MD_SINGLETON_DIMS(5)), 0);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: xn + xn', bn0_out, bni_out, bnn_out
	nlop_dw = nlop_reshape_in_F(nlop_dw, 11, 1, MD_SINGLETON_DIMS(1));

	nlop_dw = nlop_chain2_FF(nlop_from_linop_F(linop_zreal_create(1, MD_SINGLETON_DIMS(1))), 0, nlop_dw, 11);

	debug_printf(DP_DEBUG3, "MoDL dw created\n");

	return nlop_dw;
}


static const struct nlop_s* nlop_modl_cell_create(const struct modl_s* config, long dims[5], long udims[5])
{
	auto nlop_dc = mri_normal_inversion_create_general_with_lambda(5, dims, 23, 31, config->share_mask ? 7 : 23, 31, 7, config->lambda_fixed); // in: x0+zn, coil, mask, lambda; out: x(n+1)
	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	nlop_dc = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_dc, 0);
	nlop_dc = nlop_chain2_FF(nlop_dc, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);
	nlop_dc = nlop_chain2_swap_FF(nlop_zaxpbz_create(5, udims, 1., 1.), 0, nlop_dc, 0);// in: x0, zn, coil, mask, lambda; out: x(n+1)

	auto nlop_dw = nlop_dw_create(config, udims);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: zn, bn0_out, bni_out, bnn_out

	const struct nlop_s* result = nlop_chain2_FF(nlop_dw, 0, nlop_dc, 1); //in: x0, coil, mask, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: x(n+1), bn0_out, bni_out, bnn_out
	result = nlop_dup_F(result, 3, 15); //in: x0, coil, mask, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out
	result = nlop_permute_inputs_F(result, 15, MAKE_ARRAY(4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)); //in: xn, x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

	//ad dimension for stacking
	for (int i = 4; i < 15; i++)
		result = nlop_append_singleton_dim_in_F(result, i);
	for (int o = 1; o < 4; o++)
		result = nlop_append_singleton_dim_out_F(result, o);

	debug_printf(DP_DEBUG3, "MoDL cell created\n");

	return result;
}

static const struct nlop_s* nlop_nullspace_modl_cell_create(const struct modl_s* config, long dims[5], long udims[5])
{
	auto result = mri_reg_projection_ker_create_general_with_lambda(5, dims, 23, 31, config->share_mask ? 7 : 23, 31, 7, config->lambda_fixed); // in: DW(xn), coil, mask, lambda; out: PDW(xn)
	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};

	result = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, result, 0);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0); // in: DW(xn), coil, mask, lambda; out: PDW(xn)

	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(5, udims, 1., 1.), 0); // in: x0, DW(xn), coil, mask, lambda; out: PDW(xn) + x0

	auto nlop_dw = nlop_dw_create(config, udims);//in: xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: DW(xn), bn0_out, bni_out, bnn_out

	result = nlop_chain2_FF(nlop_dw, 0, result, 1); //in: x0, coil, mask, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, lambda; out: x(n+1), bn0_out, bni_out, bnn_out

	result = nlop_dup_F(result, 3, 15); //in: x0, coil, mask, lambda, xn, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out
	result = nlop_permute_inputs_F(result, 15, MAKE_ARRAY(4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)); //in: xn, x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

	//ad dimension for stacking
	for (int i = 4; i < 15; i++)
		result = nlop_append_singleton_dim_in_F(result, i);
	for (int o = 1; o < 4; o++)
		result = nlop_append_singleton_dim_out_F(result, o);

	debug_printf(DP_DEBUG3, "MoDL null space cell created\n");

	return result;
}


static const struct nlop_s* nlop_modl_network_create(const struct modl_s* config, long dims[5], long udims[5])
{
	auto result = (config->nullspace ? nlop_nullspace_modl_cell_create : nlop_modl_cell_create)(config, dims, udims);  //in: xn, x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

	for (int i = 1; i < config->Nt; i++) {

		auto nlop_append = (config->nullspace ? nlop_nullspace_modl_cell_create : nlop_modl_cell_create)(config, dims, udims); //in: xn, x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+1), bn0_out, bni_out, bnn_out

		result = nlop_chain2_FF(result, 0, nlop_append, 0);
		//in: x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out, bn0_out, bni_out, bnn_out

		//duplicate fixed inputs
		result = nlop_dup_F(result, 0, 15);
		result = nlop_dup_F(result, 1, 15);
		result = nlop_dup_F(result, 2, 15);
		//in: x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out, bn0_out, bni_out, bnn_out

		//stack batchnorm in/outputs
		result = nlop_stack_inputs_F(result, 17, 5, nlop_generic_domain(result, 5)->N - 1);
		result = nlop_stack_inputs_F(result, 19, 8, nlop_generic_domain(result, 8)->N - 1);
		result = nlop_stack_inputs_F(result, 21, 11, nlop_generic_domain(result, 11)->N - 1);
		//in: x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, lambda, conv0, bias0, convi, biasi, convn, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out, bn0_out, bni_out, bnn_out
		result = nlop_stack_outputs_F(result, 4, 1, nlop_generic_codomain(result, 4)->N - 1);
		result = nlop_stack_outputs_F(result, 4, 2, nlop_generic_codomain(result, 4)->N - 1);
		result = nlop_stack_outputs_F(result, 4, 3, nlop_generic_codomain(result, 4)->N - 1);
		//in: x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn, lambda, conv0, bias0, convi, biasi, convn, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out

		if (config->shared_lambda)
			result = nlop_dup_F(result, 3, 15);
		else
			result = nlop_stack_inputs_F(result, 15, 3, nlop_generic_domain(result, 15)->N - 1 );


		if (config->shared_weights) {

			result = nlop_dup_F(result, 4, 15);
			result = nlop_dup_F(result, 6, 15);
			result = nlop_dup_F(result, 7, 15);
			result = nlop_dup_F(result, 9, 15);
			result = nlop_dup_F(result, 10, 15);
			result = nlop_dup_F(result, 12, 15);
			result = nlop_dup_F(result, 13, 15);
			//in: x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn; out: x(n+2), bn0_out, bni_out, bnn_out
		} else {

			result = nlop_stack_inputs_F(result, 15, 4, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 6, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 7, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 9, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 10, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 12, nlop_generic_domain(result, 15)->N - 1 );
			result = nlop_stack_inputs_F(result, 15, 13, nlop_generic_domain(result, 15)->N - 1 );
			//in: x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn, xn; out: x(n+2), bn0_out, bni_out, bnn_out
		}

		result = nlop_permute_inputs_F(result, 15, MAKE_ARRAY(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
		//in: xn, x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: x(n+2), bn0_out, bni_out, bnn_out
	}

	result = nlop_dup_F(result, 0, 1);
	//in: x0, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out

	long udims_r[5] = {dims[0], dims[1], dims[2], 1, dims[4]};
	auto nlop_zf = nlop_mri_adjoint_create(dims, config->share_mask);
		nlop_zf = nlop_chain2_FF(nlop_zf, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0); // in: kspace, coil, mask; out: Atb

	if  (config->nullspace) {

		auto nlop_norm_inv = mri_normal_inversion_create_general_with_lambda(5, dims, 23, 31, config->share_mask ? 7 : 23, 31, 7, config->lambda_fixed); // in: Atb, coil, mask, lambda; out: A^+b
		nlop_norm_inv = nlop_chain2_swap_FF(nlop_from_linop_F(linop_resize_center_create(5, udims_r, udims)), 0, nlop_norm_inv, 0);
		nlop_norm_inv = nlop_chain2_FF(nlop_norm_inv, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, udims_r)), 0);

		nlop_zf = nlop_chain2_swap_FF(nlop_zf, 0,nlop_norm_inv, 0); // in: kspace, coil, mask, coil, mask, lambda; out: A^+b
		nlop_zf = nlop_dup_F(nlop_zf, 1, 3);
		nlop_zf = nlop_dup_F(nlop_zf, 2, 3);// in: kspace, coil, mask, lambda; out: A^+b

		result = nlop_chain2_swap_FF(nlop_zf, 0, result, 0);
		//in: kspace, coil, mask, lambda, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out

		result = nlop_dup_F(result, 1, 4);
		result = nlop_dup_F(result, 2, 4);
		result = nlop_append_singleton_dim_in_F(result, 3);
		result = nlop_dup_F(result, 3, 4);
		//in: kspace, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out
	} else {

		result = nlop_chain2_swap_FF(nlop_zf, 0, result, 0);
		//in: kspace, coil, mask, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out

		result = nlop_dup_F(result, 1, 3);
		result = nlop_dup_F(result, 2, 3);
		//in: kspace, coil, mask, lambda, conv0, bn0_in, bias0, convi, bni_in, biasi, convn, bnn_in, gamman, biasn; out: xn, bn0_out, bni_out, bnn_out
	}

	result = nlop_permute_inputs_F(result, 14, MAKE_ARRAY(0, 1, 2, 3, 4, 7, 10, 6, 9, 13, 12, 5, 8, 11));
	//in: kspace, coil, mask, lambda, conv0, convi, convn, bias0, biasi, biasn, gamman, bn0_in, bni_in, bnn_in; out: xn, bn0_out, bni_out, bnn_out

	return result;
}

static const struct nlop_s* nlop_val = NULL;

static complex float compute_validation_objective(long NI, const float* x[NI])
{
	if (NULL == nlop_val)
		return 0.;
	assert(NULL != x);

	void* args[NI + 1];
	for (int i = 0; i < NI; i++)
		args[i + 1] = (void*)x[i];

	args[0] = md_alloc_sameplace(1, MAKE_ARRAY(1l), sizeof(_Complex float), args[1]);

	enum NETWORK_STATUS stat_tmp = network_status;
	network_status = STAT_TEST;
	nlop_generic_apply_select_derivative_unchecked(nlop_val, NI + 1, args, 0, 0);
	network_status = stat_tmp;

	float result = 0;

	md_copy(1, MAKE_ARRAY(1l), &result, args[0], sizeof(float));
	md_free(args[0]);

	return (complex float)result;
}

static complex float get_lambda(long NI, const float* x[NI])
{
	complex float result = 0;
	md_copy(1, MD_SINGLETON_DIMS(1), &result, x[4], CFL_SIZE);
	return result;
}


/**
 * Trains Variational Network
 *
 * @param vn structure describing the variational network
 * @param train_conf structure holding training parameters of iPALM algorithm
 * @param udims (Ux, Uy, Uz, 1, Nb) - image dims of reference data
 * @param ref pointer to reference
 * @param kdims (Nx, Ny, Nz, Nc, Nb) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param coils pointer to coil data
 * @param mdims (Nx, Ny, Nz, 1, 1 / Nb) - dims of mask
 * @param coils pointer to mask data
 * @param Nb batch size for training
 * @param random_order draw random datasets to composite batch
 */
void train_nn_modl(	struct modl_s* modl, iter6_conf* train_conf,
			const long idims[5], const _Complex float* ref,
			const long kdims[5], const _Complex float* kspace,
			const long cdims[5], const _Complex float* coil,
			const long mdims[5], const _Complex float* mask,
			bool random_order, const char* history_filename, const char** valid_files)
{
	long N_datasets = idims[4];

	assert(idims[4] == kdims[4]);
	assert(idims[4] == cdims[4]);

	long dims[5];
	md_copy_dims(5, dims, kdims);
	dims[4] = modl->Nb;
	long udims[5];
	md_copy_dims(5, udims, idims);
	udims[4] = modl->Nb;

	modl->share_mask = (1 == mdims[4]);

	auto nlop_train = nlop_modl_network_create(modl, dims, udims);
	nlop_train = nlop_chain2_FF(nlop_train, 0, nlop_mse_create(5, udims, MD_BIT(4)), 0);
	long nidims[5];
	md_copy_dims(5, nidims, idims);
	nidims[4] = modl->Nb;
	nlop_train = nlop_reshape_in_F(nlop_train, 0, 5, nidims);
	//in: ref, kspace, coil, mask, lambda, conv0, convi, convn, bias0, biasi, biasn, gamman, bn0_in, bni_in, bnn_in; out: loss, bn0_out, bni_out, bnn_out

	//create batch generator
	const complex float* train_data[] = {ref, kspace, coil, mask};
	const long* train_dims[] = {	nlop_generic_domain(nlop_train, 0)->dims,
					nlop_generic_domain(nlop_train, 1)->dims,
					nlop_generic_domain(nlop_train, 2)->dims,
					nlop_generic_domain(nlop_train, 3)->dims};

	auto batch_generator = (random_order ? batch_gen_rand_create : batch_gen_linear_create)(4, 5, train_dims, train_data, N_datasets, 0);

	//setup for iter algorithm
	float* data[15];
	enum IN_TYPE in_type[15];
	const struct operator_p_s* projections[15];

	for (int i = 0; i < 4; i++) {

		data[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
		projections[i] = NULL;
	}

	data[4] = (float*)modl->lambda;
	data[5] = (float*)modl->conv_0;
	data[6] = (float*)modl->conv_i;
	data[7] = (float*)modl->conv_n;
	data[8] = (float*)modl->bias_0;
	data[9] = (float*)modl->bias_i;
	data[10] = (float*)modl->bias_n;
	data[11] = (float*)modl->gamma_n;
	data[12] = (float*)modl->bn_0;
	data[13] = (float*)modl->bn_i;
	data[14] = (float*)modl->bn_n;

	for (int i = 4; i < 12; i++) {

		projections[i] = NULL;
		in_type[i] = IN_OPTIMIZE;
	}

	for (int i = 12; i < 15; i++) {

		projections[i] = NULL;
		in_type[i] = IN_BATCHNORM;
	}

	projections[6] = operator_project_real_interval_create(nlop_generic_domain(nlop_train, 4)->N, nlop_generic_domain(nlop_train, 4)->dims, modl->lambda_min, modl->lambda_max);

	enum OUT_TYPE out_type[4] = {OUT_OPTIMIZE, OUT_BATCHNORM, OUT_BATCHNORM, OUT_BATCHNORM};

	network_status = STAT_TRAIN;

	const struct nlop_s* valid_loss = NULL;
	if (NULL != valid_files) {

		long kdims[5];
		long cdims[5];
		long udims[5];
		long mdims[5];

		complex float* val_kspace = load_cfl(valid_files[0], 5, kdims);
		complex float* val_coil = load_cfl(valid_files[1], 5, cdims);
		complex float* val_mask = load_cfl(valid_files[2], 5, mdims);
		complex float* val_ref = load_cfl(valid_files[3], 5, udims);

		complex float* scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
		complex float* u0 = md_alloc(5, udims, CFL_SIZE);
		compute_zero_filled(udims, u0, kdims, val_kspace, val_coil, mdims, val_mask);
		compute_scale(udims, scaling, u0);
		md_free(u0);

		normalize_max(udims, scaling, val_ref, val_ref);
		normalize_max(kdims, scaling, val_kspace, val_kspace);

		valid_loss = nlop_modl_network_create(modl, dims, udims);

		const struct nlop_s* loss = nlop_mse_create(5, udims, MD_BIT(4));
		loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);
		loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);

		valid_loss = nlop_chain2_FF(valid_loss, 0, loss, 0);

		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, udims, true, val_ref);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, kdims, true, val_kspace);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, cdims, true, val_coil);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, mdims, true, val_mask);

		auto nlop_del = nlop_del_out_create(5, udims);
		nlop_del = nlop_combine_FF(nlop_del, nlop_del_out_create(5, kdims));
		nlop_del = nlop_combine_FF(nlop_del, nlop_del_out_create(5, cdims));
		nlop_del = nlop_combine_FF(nlop_del, nlop_del_out_create(5, mdims));

		valid_loss = nlop_del_out_F(valid_loss, 1);
		valid_loss = nlop_del_out_F(valid_loss, 1);
		valid_loss = nlop_del_out_F(valid_loss, 1);

		valid_loss = nlop_combine_FF(nlop_del, valid_loss);

		nlop_val = valid_loss;

		unmap_cfl(5, udims, val_ref);
		unmap_cfl(5, kdims, val_kspace);
		unmap_cfl(5, cdims, val_coil);
		unmap_cfl(5, mdims, val_mask);
	}

	struct iter6_monitor_value_s val_monitors[2];
	val_monitors[0] = (struct iter6_monitor_value_s){&compute_validation_objective, &"val loss"[0], false};
	val_monitors[1] = (struct iter6_monitor_value_s){&get_lambda, &"lambda"[0], true};

	auto conf = CAST_DOWN(iter6_adam_conf, train_conf);
	//auto monitor = create_iter6_monitor_progressbar_validloss(conf->epochs, N_datasets / modl->Nb, false, 15, in_type, valid_loss, false);
	auto monitor = create_iter6_monitor_progressbar_value_monitors(conf->epochs, N_datasets / modl->Nb, false, 2, val_monitors);
	iter6_adam(train_conf, nlop_train, 15, in_type, projections, data, 4, out_type, modl->Nb, N_datasets / modl->Nb, batch_generator, monitor);
	if (NULL != history_filename)
		iter6_monitor_dump_record(monitor, history_filename);
	network_status = STAT_TEST;

	nlop_free(nlop_train);
	nlop_free(batch_generator);
}

/**
 * Trains Variational Network
 *
 * @param vn structure describing the variational network
 * @param train_conf structure holding training parameters of iPALM algorithm
 * @param udims (Ux, Uy, Uz, 1, Nb) - image dims of reference data
 * @param ref pointer to reference
 * @param kdims (Nx, Ny, Nz, Nc, Nb) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param coils pointer to coil data
 * @param mdims (Nx, Ny, Nz, 1, 1 / Nb) - dims of mask
 * @param coils pointer to mask data
 * @param Nb batch size for training
 * @param random_order draw random datasets to composite batch
 */
void apply_nn_modl(	struct modl_s* modl,
			const long idims[5], _Complex float* out,
			const long kdims[5], const _Complex float* kspace,
			const long cdims[5], const _Complex float* coil,
			const long mdims[5], const _Complex float* mask)
{
	START_TIMER;
	long N_datasets = idims[4];
	assert(idims[4] == kdims[4]);
	assert(idims[4] == cdims[4]);

	modl->share_mask = (1 == mdims[4]);

	complex float* args[15];

	args[0] = md_alloc_sameplace(5, idims, CFL_SIZE, modl->lambda);
	args[1] = md_alloc_sameplace(5, kdims, CFL_SIZE, modl->lambda);
	args[2] = md_alloc_sameplace(5, cdims, CFL_SIZE, modl->lambda);
	args[3] = md_alloc_sameplace(5, mdims, CFL_SIZE, modl->lambda);
	args[4] = modl->lambda;
	args[5] = modl->conv_0;
	args[6] = modl->conv_i;
	args[7] = modl->conv_n;
	args[8] = modl->bias_0;
	args[9] = modl->bias_i;
	args[10] = modl->bias_n;
	args[11] = modl->gamma_n;
	args[12] = modl->bn_0;
	args[13] = modl->bn_i;
	args[14] = modl->bn_n;

	complex float* args2[4] = {args[0], args[1], args[2], args[3]}; //save pointers for md_free

	md_copy(5, kdims, args[1], kspace, CFL_SIZE);
	md_copy(5, cdims, args[2], coil, CFL_SIZE);
	md_copy(5, mdims, args[3], mask, CFL_SIZE);

	while (N_datasets > 0) {

		long Nb =  MIN(modl->Nb, N_datasets);

		long dims[5];
		md_copy_dims(5, dims, kdims);
		dims[4] = Nb;
		long udims[5];
		md_copy_dims(5, udims, idims);
		udims[4] = Nb;

		N_datasets -= Nb;
		auto nlop_modl = nlop_modl_network_create(modl, dims, udims);

		nlop_modl = nlop_del_out_F(nlop_modl, 1);
		nlop_modl = nlop_del_out_F(nlop_modl, 1);
		nlop_modl = nlop_del_out_F(nlop_modl, 1);

		network_status = STAT_TEST;

		nlop_generic_apply_unchecked(nlop_modl, 15, (void**)args);

		network_status = STAT_TRAIN;

		args[1] += md_calc_size(nlop_generic_domain(nlop_modl, 0)->N, nlop_generic_domain(nlop_modl, 0)->dims);
		args[2] += md_calc_size(nlop_generic_domain(nlop_modl, 1)->N, nlop_generic_domain(nlop_modl, 1)->dims);
		if (!modl->share_mask)
			args[3] += md_calc_size(nlop_generic_domain(nlop_modl, 2)->N, nlop_generic_domain(nlop_modl, 2)->dims);

		args[0] += md_calc_size(nlop_generic_codomain(nlop_modl, 0)->N, nlop_generic_codomain(nlop_modl, 0)->dims);

		nlop_free(nlop_modl);
	}

	md_copy(5, idims, out, args2[0], CFL_SIZE);

	md_free(args2[0]);
	md_free(args2[1]);
	md_free(args2[2]);
	md_free(args2[3]);
	PRINT_TIMER("MoDL");
}


void init_nn_modl(struct modl_s* modl)
{
	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};

	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};

	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	long Ns[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};
	long* dims[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	for (int i = 0; i < 11; i++)
		if (NULL == *weights[i])
			*weights[i] = md_alloc(Ns[i], dims[i], CFL_SIZE);

	//init lambda
	long wi = 0;
	md_zfill(Ns[wi], dims[wi], *weights[wi], modl->lambda_init);

	//init conv_w
	wi = 1;
	complex float* tmp = *weights[wi];
	for (int i = 0; i < dims[wi][Ns[wi] - 1]; i++)
		tmp = init_glorot_uniform_conv_complex(Ns[wi] - 1, dims[wi], tmp, true);

	wi = 2;
	tmp = *weights[wi];
	for (int i = 0; i < dims[wi][Ns[wi] - 1] * dims[wi][Ns[wi] - 2]; i++)
		tmp = init_glorot_uniform_conv_complex(Ns[wi] - 2, dims[wi], tmp, true);

	wi = 3;
	tmp = *weights[wi];
	for (int i = 0; i < dims[wi][Ns[wi] - 1]; i++)
		tmp = init_glorot_uniform_conv_complex(Ns[wi] - 1, dims[wi], tmp, true);

	//init bias with 0
	for (wi = 4; wi < 7; wi++)
		md_zfill(Ns[wi], dims[wi], *weights[wi], 0.);

	//init gamma with 1
	wi = 7;
	md_zfill(Ns[wi], dims[wi], *weights[wi], 1.);

	//init batchnorm with 0
	for (wi = 8; wi < 11; wi++)
		md_zfill(Ns[wi], dims[wi], *weights[wi], 0.);

}

void nn_modl_move_gpucpu(struct modl_s* modl, bool gpu) {

#ifdef USE_CUDA

	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};

	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};

	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	long Ns[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};

	long* dims[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	for (int i = 0; i < 11; i++) {

		complex float* tmp = (gpu ? md_alloc_gpu : md_alloc)(Ns[i], dims[i], CFL_SIZE);
		md_copy(Ns[i], dims[i], tmp, *weights[i], CFL_SIZE);
		md_free(*weights[i]);
		*weights[i] = tmp;
	}
#else
	UNUSED(modl);
	if (gpu)
		error("Compiled without GPU support!");
#endif
}

extern void nn_modl_store_weights(struct modl_s* modl, const char* name)
{
	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};

	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};

	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	long Ns[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};

	long* dims[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	long num_weights = 0;

	for (int i = 0; i < 11; i++)
		num_weights += md_calc_size(Ns[i], dims[i]);

	complex float* file = create_cfl(name, 1, &num_weights);
	complex float* tmp = file;

	for (int i = 0; i < 11; i++) {

		md_copy(Ns[i], dims[i], tmp, *weights[i], CFL_SIZE);
		tmp += md_calc_size(Ns[i], dims[i]);
	}
	unmap_cfl(1, &num_weights, file);
}

extern void nn_modl_load_weights(struct modl_s* modl, const char* name)
{
	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	long dims_lambda[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_conv_0[6] = {modl->Nf, 1, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_i[7] = {modl->Nf, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_conv_n[6] = {1, modl->Nf, modl->Kx, modl->Ky, modl->Kz, modl->shared_weights ? 1 : modl->Nt};

	long dims_bias_0[2] = {modl->Nf, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_i[3] = {modl->Nf, modl->Nl - 2, modl->shared_weights ? 1 : modl->Nt};
	long dims_bias_n[2] = {1, modl->shared_weights ? 1 : modl->Nt};

	long dims_gamma[6] = {1, 1, 1, 1, 1, modl->shared_weights ? 1 : modl->Nt};

	long dims_bn_0[7] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nt};
	long dims_bn_i[8] = {modl->Nf, 1, 1, 1, 1, 2, modl->Nl - 2, modl->Nt};
	long dims_bn_n[7] = {1, 1, 1, 1, 1, 2, modl->Nt};

	long Ns[11] = {2, 6, 7, 6, 2, 3, 2, 6, 7, 8, 7};

	long* dims[11] = {
		dims_lambda,
		dims_conv_0,
		dims_conv_i,
		dims_conv_n,
		dims_bias_0,
		dims_bias_i,
		dims_bias_n,
		dims_gamma,
		dims_bn_0,
		dims_bn_i,
		dims_bn_n
	};

	long num_weights_network = 0;
	for (int i = 0; i < 11; i++)
		num_weights_network += md_calc_size(Ns[i], dims[i]);

	long num_weights_file = 0;

	complex float* file = load_cfl(name, 1, &num_weights_file);
	complex float* tmp = file;

	unsigned long num_weights_loaded = 0;

	for (int i = 0; i < 8; i++) {

		if (NULL == *weights[i])
			*weights[i] = md_alloc(Ns[i], dims[i], CFL_SIZE);

		md_copy(Ns[i], dims[i], *weights[i], tmp, CFL_SIZE);
		tmp += md_calc_size(Ns[i], dims[i]);
		num_weights_loaded += md_calc_size(Ns[i], dims[i]);
	}

	long num_weights_file_remaining = num_weights_file - num_weights_loaded;
	long num_weights_needed = num_weights_network - num_weights_loaded;

	if (!((num_weights_file_remaining == num_weights_needed) || (modl->Nt * num_weights_file_remaining == num_weights_needed)))
		error("Loaded weights do not match network parameters!\n");

	bool repeat_bn = (modl->Nt * num_weights_file_remaining == num_weights_needed);

	for (int i = 8; i < 11; i++) {

		if (NULL == *weights[i])
			*weights[i] = md_alloc(Ns[i], dims[i], CFL_SIZE);

		for (int j = 0; j < modl->Nt; j++) {

			md_copy(Ns[i] - 1, dims[i], *weights[i] + j * md_calc_size(Ns[i] - 1, dims[i]), tmp, CFL_SIZE);
			if (!repeat_bn || (j == modl->Nt - 1))
				tmp += md_calc_size(Ns[i] - 1, dims[i]);
		}
	}

	unmap_cfl(1, &num_weights_file, file);
}

extern void nn_modl_free_weights(struct modl_s* modl)
{
	complex float** weights[] = {
		&(modl->lambda),
		&(modl->conv_0),
		&(modl->conv_i),
		&(modl->conv_n),
		&(modl->bias_0),
		&(modl->bias_i),
		&(modl->bias_n),
		&(modl->gamma_n),
		&(modl->bn_0),
		&(modl->bn_i),
		&(modl->bn_n)
	};

	for (int i = 0; i < 11; i++) md_free(*weights[i]);
}
