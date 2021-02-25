
#include <math.h>

#include "misc/debug.h"
#include "misc/types.h"
#include "nn/activation.h"
#include "num/multind.h"
#include "num/iovec.h"

#include "iter/proj.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"

#include "nn/activation_nn.h"
#include "nn/rbf.h"
#include "nn/chain.h"
#include "nn/nn.h"
#include "nn/layers_nn.h"
#include "nn/init.h"


#include  "cnn.h"

DEF_TYPEID(network_resnet_s);

const char* resnet_sorted_weight_names[] = {
					"conv_0", "conv_i", "conv_n",
					"bias_0", "bias_i", "bias_n",
					"gamma",
					"bn_0", "bn_i", "bn_n"
				};


struct network_resnet_s network_resnet_default = {

	.INTERFACE.TYPEID = &TYPEID2(network_resnet_s),
	
	.INTERFACE.create = network_resnet_create,

	.INTERFACE.low_mem = false,

	.N = 5,
	
	.Nl = 5,
	.Nf = 32,

	.Kx = 3,
	.Ky = 3,
	.Kz = 1,

	.Ng = 1,

	.conv_flag = MD_BIT(1) | MD_BIT(2) | MD_BIT(3),
	.channel_flag = MD_BIT(0),
	.group_flag = 0,
	.batch_flag = MD_BIT(4),

	.kdims = {[0 ... DIMS -1] = 0},
	.dilations = {[0 ... DIMS -1] = 1},

	.batch_norm = true,
	.bias = true,

	.activation = ACT_RELU,
	.last_activation = ACT_LIN,
};



static void network_resnet_get_kdims(const struct network_resnet_s* config, unsigned int N, long kdims[N])
{
	if (0 != md_calc_size(config->N, config->kdims)) {

		md_copy_dims(N, kdims, config->kdims);
		return;
	}
	
	assert(1 == bitcount(config->channel_flag));
	assert(3 >= bitcount(config->conv_flag));

	long tdims[3] = {config->Kx, config->Ky, config->Kz};
	long* tdim = tdims;

	for (unsigned int i = 0; i < N; i++) {

		kdims[i] = 1;

		if (MD_IS_SET(config->conv_flag, i)) {

			kdims[i] = *tdim;
			tdim += 1;
		}

		if (MD_IS_SET(config->channel_flag, i))
			kdims[i] = config->Nf;
		
		if (MD_IS_SET(config->group_flag, i))
			kdims[i] = config->Ng;
	}
}

/**
 * Returns residual block
 *
 * Input tensors:
 *
 * INDEX_0: 	idims
 * weights as listed in "sorted_weight_names"
 *
 * Output tensors:
 *
 * INDEX_0:	odims
 * batchnorm
 */
nn_t network_resnet_create(const struct network_s* _config, unsigned int N, const long odims[N], const long idims[N], enum NETWORK_STATUS status)
{
	auto config = CAST_DOWN(network_resnet_s, _config);
	assert(config->N == N);

	long kdims[N];
	network_resnet_get_kdims(config, N, kdims);

	nn_t result = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, idims)));

	auto conv_init = init_kaiming_create(in_flag_conv(true), false, false, 0);

	//if dim for group index are not equal in the first layer, we make it a chennl dim
	unsigned long tchannel_flag = config->channel_flag; 
	unsigned long tgroup_flag = config->group_flag;
	for (unsigned int i = 0; i < N; i++){

		if (MD_IS_SET(tgroup_flag, i) && (kdims[i] != idims[i])) {

			tgroup_flag = MD_CLEAR(tgroup_flag, i);
			tchannel_flag = MD_SET(tchannel_flag, i);
		}
	}

	result = nn_append_convcorr_layer_generic(	result, 0, NULL, "conv_0",
							config->conv_flag, tchannel_flag, tgroup_flag,
							N, kdims, NULL, config->dilations,
							false, PAD_SAME, initializer_clone(conv_init));

	if (config->batch_norm)
		result = nn_append_batchnorm_layer(result, 0, NULL, "bn_0", ~(config->channel_flag | config->group_flag), status, NULL);

	if (config->bias)
		result = nn_append_activation_bias(result, 0, NULL, "bias_0", config->activation, MD_BIT(0));
	else
		result = nn_append_activation(result, 0, NULL, config->activation);


	for (int i = 0; i < config->Nl - 2; i++) {

		result = nn_mark_stack_input_if_exists_F(result, "conv_i");
		result = nn_mark_stack_input_if_exists_F(result, "bias_i");
		result = nn_mark_stack_input_if_exists_F(result, "bn_i");
		result = nn_mark_stack_output_if_exists_F(result, "bn_i");


		result = nn_append_convcorr_layer_generic(	result, 0, NULL, "conv_i",
							config->conv_flag, config->channel_flag, config->group_flag,
							N, kdims, NULL, config->dilations,
							false, PAD_SAME, initializer_clone(conv_init));

		if (config->batch_norm)
			result = nn_append_batchnorm_layer(result, 0, NULL, "bn_i", ~(config->channel_flag | config->group_flag), status, NULL);
		if (config->bias)
			result = nn_append_activation_bias(result, 0, NULL, "bias_i", config->activation, MD_BIT(0));
		else
			result = nn_append_activation(result, 0, NULL, config->activation);

		result = nn_append_singleton_dim_in_if_exists_F(result, "conv_i");
		result = nn_append_singleton_dim_in_if_exists_F(result, "bias_i");
		result = nn_append_singleton_dim_in_if_exists_F(result, "bn_i");
		result = nn_append_singleton_dim_out_if_exists_F(result, "bn_i");

		result = nn_stack_dup_by_name_F(result);
	}

	long ldims[N];
	md_copy_dims(N, ldims, kdims);
	for(unsigned int i = 0; i < N; i++)
		if (MD_IS_SET(config->channel_flag, i) || MD_IS_SET(config->group_flag, i))
			ldims[i] = odims[i];

	//if dim for group index are not equal in the last layer, we make it a chennl dim
	tchannel_flag = config->channel_flag; 
	tgroup_flag = config->group_flag;
	for (unsigned int i = 0; i < N; i++){

		if (MD_IS_SET(tgroup_flag, i) && (ldims[i] != odims[i])) {

			tgroup_flag = MD_CLEAR(tgroup_flag, i);
			tchannel_flag = MD_CLEAR(tchannel_flag, i);
		}
	}

	const struct initializer_s* conv_init_last = (config->batch_norm) ? initializer_clone(conv_init) : init_const_create(0);

	result = nn_append_convcorr_layer_generic(	result, 0, NULL, "conv_n",
							config->conv_flag, tchannel_flag, tgroup_flag,
							N, ldims, NULL, config->dilations,
							false, PAD_SAME, initializer_clone(conv_init_last));
	
	initializer_free(conv_init);
	initializer_free(conv_init_last);

	if (config->batch_norm) {

		result = nn_append_batchnorm_layer(result, 0, NULL, "bn_n", ~(config->channel_flag | config->group_flag), status, NULL);
	
		//append gamma for batchnorm
		auto iov = nn_generic_codomain(result, 0, NULL);
		long gdims [iov->N];
		md_select_dims(iov->N, config->channel_flag | config->group_flag, gdims, iov->dims);


		auto nn_scale_gamma = nn_from_nlop_F(nlop_tenmul_create(iov->N, iov->dims, iov->dims, gdims));
		result = nn_chain2_swap_FF(result, 0, NULL, nn_scale_gamma, 0, NULL);
		result = nn_set_input_name_F(result, -1, "gamma");
		result = nn_set_initializer_F(result, 0, "gamma", init_const_create(0));
		result = nn_set_in_type_F(result, 0, "gamma", IN_OPTIMIZE);
		result = nn_set_dup_F(result, 0, "gamma", false);
	}
	
	if (config->bias)
		result = nn_append_activation_bias(result, 0, NULL, "bias_n", config->last_activation, MD_BIT(0));
	else
		result = nn_append_activation(result, 0, NULL, config->last_activation);


	auto nlop_sum = nlop_zaxpbz_create(N, odims, 1, 1);
	nlop_sum = nlop_chain2_FF(nlop_from_linop_F(linop_expand_create(N, odims, idims)), 0, nlop_sum, 1);

	result = nn_chain2_FF(result, 0, NULL, nn_from_nlop_F(nlop_sum), 1, NULL);
	result = nn_dup_F(result, 0, NULL, 1, NULL);
	
	result = nn_sort_inputs_by_list_F(result, ARRAY_SIZE(resnet_sorted_weight_names), resnet_sorted_weight_names);
	result = nn_sort_outputs_by_list_F(result, ARRAY_SIZE(resnet_sorted_weight_names), resnet_sorted_weight_names);

	return nn_checkpoint_F(result, true, config->INTERFACE.low_mem);
}



DEF_TYPEID(network_varnet_s);

const char* varnet_sorted_weight_names[] = {"conv", "rbf"};

struct network_varnet_s network_varnet_default = {

	.INTERFACE.TYPEID = &TYPEID2(network_varnet_s),
	
	.INTERFACE.create = network_varnet_create,

	.INTERFACE.low_mem = false,

	.Nf = 24,
	.Nw = 31,

	.Kx = 11,
	.Ky = 11,
	.Kz = 1,

	.Imax = 1.,
	.Imin = -1.,

	.residual = true,

	.init_scale_mu = 0.04,
};



nn_t network_varnet_create(const struct network_s* _config, unsigned int N, const long odims[N], const long idims[N], enum NETWORK_STATUS status)
{
	UNUSED(status);

	auto config = CAST_DOWN(network_varnet_s, _config);

	assert(5 == N);
	assert(md_check_equal_dims(N, idims, odims, ~0));
	assert(1 == idims[0]);

	//Padding
	long pad_up[5] = {0, (config->Kx - 1), (config->Ky - 1), (config->Kz - 1), 0};
	long pad_down[5] = {0, -(config->Kx - 1), -(config->Ky - 1), -(config->Kz - 1), 0};
	long ker_size[3] = {config->Kx, config->Ky, config->Kz};

	long Ux = idims[1];
	long Uy = idims[2];
	long Uz = idims[3];
	long Nb = idims[4];

	//working dims
	long zdimsw[5] = {config->Nf, Ux + 2 * (config->Kx - 1), Uy + 2 * (config->Ky - 1), Uz + 2 * (config->Kz - 1), Nb};
	long rbfdims[3] = {config->Nf, (Ux + 2 * (config->Kx - 1)) * (Uy + 2 * (config->Ky - 1)) * (Uz + 2 * (config->Kz - 1)) * Nb, config->Nw};

	//operator dims
	long kerdims[5] = {config->Nf, config->Kx, config->Ky, config->Kz, 1};
	long wdims[3] = {config->Nf, config->Nw, 1};

	const struct nlop_s* nlop_result = nlop_from_linop_F(linop_identity_create(5, idims)); // in: u
	//nlop_result = nlop_chain2_FF(nlop_result, 0, padu, 0); // in: u
	nlop_result = append_padding_layer(nlop_result, 0, 5, pad_up, pad_up, PAD_SYMMETRIC);
	nlop_result = append_convcorr_layer(nlop_result, 0, config->Nf, ker_size, false, PAD_SAME, true, NULL, NULL); // in: u, conv_w

	const struct nlop_s* rbf = nlop_activation_rbf_create(rbfdims, config->Imax, config->Imin);
	rbf = nlop_reshape_in_F(rbf, 0, 5, zdimsw);
	rbf = nlop_reshape_out_F(rbf, 0, 5, zdimsw);
	nlop_result = nlop_chain2_FF(nlop_result, 0, rbf, 0); //in: rbf_w, in, conv_w

	nlop_result = append_transposed_convcorr_layer(nlop_result, 0, 1, ker_size, false, true, PAD_SAME, true, NULL, NULL); //in: rbf_w, u, conv_w, conv_w
	//nlop_result = nlop_chain2_FF(nlop_result, 0, padd, 0); //in: rbf_w, u, conv_w, conv_w
	nlop_result = append_padding_layer(nlop_result, 0, 5, pad_down, pad_down, PAD_VALID);
	nlop_result = nlop_dup_F(nlop_result, 2, 3); //in: rbf_w, u, conv_w

	nlop_result = nlop_reshape_in_F(nlop_result, 2, 4, kerdims); //in: rbf_w, u, conv_w
	nlop_result = nlop_reshape_in_F(nlop_result, 0, 2, wdims); //in: rbf_w, u, conv_w

	//VN implementation: u_k = (real(up) * real(k) + imag(up) * imag(k))
	nlop_result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(4, kerdims)), 0, nlop_result, 2); //in: rbf_w, u, conv_w
	nlop_result = nlop_chain2_FF(nlop_result, 0, nlop_from_linop_F(linop_scale_create(5, idims, 1. / config->Nf)), 0); //in: rbf_w, u, conv_w

	int perm [] = {1, 2, 0};
	nlop_result = nlop_permute_inputs_F(nlop_result, 3, perm); //in: u, conv_w, rbf_w

	auto nn_result = nn_from_nlop_F(nlop_result);
	nn_result = nn_set_input_name_F(nn_result, 1, "conv");
	nn_result = nn_set_input_name_F(nn_result, 1, "rbf");

	nn_result = nn_set_initializer_F(nn_result, 0, "conv", init_std_normal_create(false, 1. / sqrtf((float)config->Kx * config->Ky * config->Kz), 0));
	nn_result = nn_set_initializer_F(nn_result, 0, "rbf", init_linspace_create(1., config->Imin * config->init_scale_mu, config->Imax * config->init_scale_mu, true));
	nn_result = nn_set_in_type_F(nn_result, 0, "conv", IN_OPTIMIZE);
	nn_result = nn_set_in_type_F(nn_result, 0, "rbf", IN_OPTIMIZE);
	
	auto iov = nn_generic_domain(nn_result, 0, "conv");
	auto prox_conv = operator_project_mean_free_sphere_create(iov->N, iov->dims, MD_BIT(0), false);
	nn_result = nn_set_prox_op_F(nn_result, 0, "conv", prox_conv);

	if(config->residual) {

		nn_result = nn_chain2_FF(nn_result, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(N, idims, 1., -1.)), 1, NULL);
		nn_result = nn_dup_F(nn_result, 0, NULL, 1, NULL);
	}

	nn_result = nn_sort_inputs_by_list_F(nn_result, ARRAY_SIZE(varnet_sorted_weight_names), varnet_sorted_weight_names);

	nn_debug(DP_DEBUG3, nn_result);

	return nn_checkpoint_F(nn_result, true, config->INTERFACE.low_mem);
}