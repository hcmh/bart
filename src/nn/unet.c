#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "iter/italgos.h"
#include "misc/debug.h"
#include "misc/misc.h"

#include "num/fft.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "nlops/stack.h"

#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/weights.h"
#include "nn/init.h"


#include "nn/unet.h"

const struct unet_s unet_default_reco = {

	.convolution_kernel = {3, 3, 1},

	.channels = 24, //channels on highest level
	.channel_factor = 1., //number channels on lower level
	.reduce_factor = 2., //reduce resolution of lower level
	.number_levels = 4,
	.number_output_channels = 1,
	.number_layers_per_level = 3,

	.real_constraint_input = false,
	.real_constraint_output = false,
	.real_constraint_weights = false,

	.use_batchnormalization = true,
	.use_bias = true,
	.use_transposed_convolution = true,

	.activation = ACT_RELU,
	.activation_output = ACT_LIN,
	.activation_last_layer = ACT_LIN,
	.padding = PAD_SAME,

	.deflatten_conv = NULL,
	.deflatten_bias = NULL,
	.deflatten_bn = NULL,

	.ds_methode = UNET_DS_FFT,
	.upsampling_combine = UNET_COMBINE_CONV,

	.weights = NULL,
	.size_weights = 0,
	.size_weights_conv = 0,
	.size_weights_bias = 0,
	.size_weights_bn = 0,
};

static void unet_updownsample_fft_create(const struct nlop_s* result[2], struct unet_s* unet, long level, long down_dims[5], const long dims[5])
{
	UNUSED(level);

	for (int i = 0; i < 5; i++)
		down_dims[i] = MD_IS_SET(14ul, i) ? MAX(1, round(dims[i] / unet->reduce_factor)) : dims[i];

	auto linop_result = linop_fftc_create(5, dims, 14ul);
	linop_result = linop_chain_FF(linop_result, linop_resize_center_create(5, down_dims, dims));
	linop_result = linop_chain_FF(linop_result, linop_ifftc_create(5, down_dims, 14ul));

	result[1] = nlop_from_linop_F(linop_get_adjoint(linop_result));
	result[0] = nlop_from_linop_F(linop_result);
}

static void unet_updownsample_create(const struct nlop_s* result[2], struct unet_s* unet, long level, long down_dims[5], const long dims[5])
{
	UNUSED(level);

	switch (unet->ds_methode) {

	case UNET_DS_FFT:
		unet_updownsample_fft_create(result, unet, level, down_dims, dims);
		return ;
	}
}

static const struct nlop_s* nlop_expand_input(const struct nlop_s* nlop, int index, long N, long idims[N])
{
	assert(nlop_generic_domain(nlop, index)->N == N);
	const long* old_idims = nlop_generic_domain(nlop, index)->dims;

	if (md_check_equal_dims(5, old_idims, idims, ~0ul))
		return nlop;

	nlop = nlop_chain2_FF(nlop_from_linop_F(linop_expand_create(N, old_idims, idims)), 0, nlop, index);
	nlop = nlop_shift_input_F(nlop, index, nlop_get_nr_in_args(nlop) - 1);

	return nlop;
}

static const struct nlop_s* nlop_expand_output(const struct nlop_s* nlop, int index, long N, long odims[N])
{
	assert(nlop_generic_codomain(nlop, index)->N == N);
	const long* old_odims = nlop_generic_codomain(nlop, index)->dims;

	if (md_check_equal_dims(5, old_odims, odims, ~0ul))
		return nlop;

	nlop = nlop_chain2_FF(nlop, index, nlop_from_linop_F(linop_expand_create(N, odims, old_odims)), 0);
	nlop = nlop_shift_output_F(nlop, 0, index);

	return nlop;
}



static const struct nlop_s* create_unet_level(struct unet_s* unet, long level, long dims[5], long channels) {

	const struct nlop_s* result = nlop_from_linop_F(linop_identity_create(5, dims));

	int index_conv[2 * unet->number_layers_per_level];
	int index_bias[2 * unet->number_layers_per_level];
	int index_bn_in[2 * unet->number_layers_per_level];
	int index_bn_out[2 * unet->number_layers_per_level];

	int index_counter_conv = 0;
	int index_counter_bias = 0;
	int index_counter_bn_in = 0;
	int index_counter_bn_out = 0;

	for (int i = 0; i < unet->number_layers_per_level; i++) {

		result = append_convcorr_layer(result, 0, channels, unet->convolution_kernel, false, unet->padding, true, NULL, NULL);
		index_conv[index_counter_conv++] = nlop_get_nr_in_args(result) - 1;

		if (unet->use_batchnormalization) {

			result = append_batchnorm_layer(result, 0, ~MD_BIT(0));

			index_bn_in[index_counter_bn_in++] = nlop_get_nr_in_args(result) - 1;
			index_bn_out[index_counter_bn_out++] = nlop_get_nr_out_args(result) - 1;
		}

		if (unet->use_bias) {

			result = append_activation_bias(result, 0, unet->activation, MD_BIT(0));
			index_bias[index_counter_bias++] = nlop_get_nr_in_args(result) - 1;
		} else
			result = append_activation(result, 0, unet->activation);
	}

	debug_printf(DP_DEBUG3, "level %d first conv block created: ", level);
	nlop_debug(DP_DEBUG3, result);

	int index_weights_lower_in = nlop_get_nr_in_args(result);
	int index_weights_lower_out = nlop_get_nr_out_args(result);

	if (level < unet->number_levels) {

		long down_dims[5];
		const struct nlop_s* nlop_sample[2];

		unet_updownsample_create(nlop_sample, unet, level, down_dims, nlop_generic_codomain(result, 0)->dims);
		auto nlop_lower_level = create_unet_level(unet, level + 1, down_dims, round(channels * unet->channel_factor));


		nlop_lower_level = nlop_chain2_swap_FF(nlop_sample[0], 0, nlop_lower_level, 0);
		nlop_lower_level = nlop_chain2_FF(nlop_lower_level, 0, nlop_sample[1], 0);

		long tmp_dims[5] = {	2 * nlop_generic_codomain(result, 0)->dims[0],
					nlop_generic_codomain(result, 0)->dims[1],
					nlop_generic_codomain(result, 0)->dims[2],
					nlop_generic_codomain(result, 0)->dims[3],
					nlop_generic_codomain(result, 0)->dims[4]};

		const struct nlop_s* nlop_combine_op = NULL;

		switch (unet->upsampling_combine) {

			case UNET_COMBINE_SUM:
				nlop_combine_op = nlop_zaxpbz_create(5, nlop_generic_codomain(result, 0)->dims, 1., 1.);
				break;
			case UNET_COMBINE_CONV:
				nlop_combine_op = nlop_stack_create(5, tmp_dims, nlop_generic_codomain(result, 0)->dims, nlop_generic_codomain(result, 0)->dims, 0);
		}


		nlop_lower_level = nlop_chain2_FF(nlop_lower_level, 0, nlop_zaxpbz_create(5, nlop_generic_codomain(result, 0)->dims, 1., 1.), 0);
		nlop_lower_level = nlop_dup_F(nlop_lower_level, 0, 1);

		result = nlop_chain2_swap_FF(result, 0, nlop_lower_level, 0);

		if (unet->use_batchnormalization)
			for (int i = 0; i < nlop_get_nr_out_args(result) - index_weights_lower_out; i++)
				result = nlop_shift_output_F(result, nlop_get_nr_out_args(result) - 1, 1);

		debug_printf(DP_DEBUG3, "level %d sublevel appended: ", level);
		nlop_debug(DP_DEBUG3, result);
	}

	int index_counter_conv_transp = index_counter_conv - 1;

	for (int i = 0; i < unet->number_layers_per_level; i++) {

		if (i + 1 == unet->number_layers_per_level)
			channels = (1 == level) ? unet->number_output_channels : dims[0];

		if (unet->use_transposed_convolution) {

			result = append_transposed_convcorr_layer(result, 0, channels, unet->convolution_kernel, false, true, unet->padding, true, NULL, NULL);

			if (md_check_equal_dims(5, nlop_generic_domain(result, index_conv[index_counter_conv_transp])->dims, nlop_generic_domain(result, nlop_get_nr_in_args(result) - 1)->dims, ~0ul))
				result = nlop_dup_F(result, index_conv[index_counter_conv_transp], nlop_get_nr_in_args(result) - 1);
			else
				index_conv[index_counter_conv++] = nlop_get_nr_in_args(result) - 1;

			index_counter_conv_transp--;

		} else {

			result = append_convcorr_layer(result, 0, channels, unet->convolution_kernel, false, unet->padding, true, NULL, NULL);
			index_conv[index_counter_conv++] = nlop_get_nr_in_args(result) - 1;
		}

		if ((unet->use_batchnormalization) && (((level > 1) && (UNET_COMBINE_CONV == unet->upsampling_combine)) || (i + 1 != unet->number_layers_per_level))) {

			result = append_batchnorm_layer(result, 0, ~MD_BIT(0));

			index_bn_in[index_counter_bn_in++] = nlop_get_nr_in_args(result) - 1;
			index_bn_out[index_counter_bn_out++] = nlop_get_nr_out_args(result) - 1;
		}

		enum ACTIVATION activation = unet->activation;
		if (i + 1 == unet->number_layers_per_level)
			activation = (level == 1) ? unet->activation_output : unet->activation_last_layer;

		if (unet->use_bias) {

			result = append_activation_bias(result, 0, activation, MD_BIT(0));
			index_bias[index_counter_bias++] = nlop_get_nr_in_args(result) - 1;

		} else {

			result = append_activation(result, 0, unet->activation);
		}
	}

	int perm_in[nlop_get_nr_in_args(result)];
	perm_in[0] = 0;
	int in_offset = 1;

	for (int i = 0; i < index_counter_conv; i++)
		perm_in[i + in_offset] = index_conv[i];
	in_offset += index_counter_conv;

	for (int i = 0; i < index_counter_bias; i++)
		perm_in[i + in_offset] = index_bias[i];
	in_offset += index_counter_bias;

	for (int i = 0; i < index_counter_bn_in; i++)
		perm_in[i + in_offset] = index_bn_in[i];
	in_offset += index_counter_bn_in;

	for (int i = 0; i < nlop_get_nr_in_args(result) - in_offset; i++)
		perm_in[i + in_offset] = index_weights_lower_in + i;

	result = nlop_permute_inputs_F(result, nlop_get_nr_in_args(result), perm_in);


	int perm_out[nlop_get_nr_out_args(result)];
	perm_out[0] = 0;
	int out_offset = 1;

	for (int i = 0; i < index_counter_bn_out; i++)
		perm_out[i + out_offset] = index_bn_out[i];
	out_offset += index_counter_bn_out;

	for (int i = 0; i < nlop_get_nr_out_args(result) - out_offset; i++)
		perm_out[i + out_offset] = index_weights_lower_out + i;

	result = nlop_permute_outputs_F(result, nlop_get_nr_out_args(result), perm_out);

	if (NULL == unet->deflatten_conv[level - 1])
		unet->deflatten_conv[level - 1] = deflatten_weights_create(result, ~((1lu << (index_counter_conv + 1)) - 2lu));

	int index_deflatten = nlop_get_nr_out_args(result);

	auto nlop_tmp = nlop_combine(result, unet->deflatten_conv[level - 1]);
	nlop_free(result);
	result = nlop_tmp;

	for (int i = 0; i < index_counter_conv; i++)
		result = nlop_link_F(result, index_deflatten, 1);

	if (unet->use_bias) {

		if (NULL == unet->deflatten_bias[level - 1])
			unet->deflatten_bias[level - 1] = deflatten_weights_create(result, ~((1lu << (index_counter_bias + 1)) - 2lu));

		int index_deflatten = nlop_get_nr_out_args(result);

		auto nlop_tmp = nlop_combine(result, unet->deflatten_bias[level - 1]);
		nlop_free(result);
		result = nlop_tmp;

		for (int i = 0; i < index_counter_bias; i++)
			result = nlop_link_F(result, index_deflatten, 1);
	}

	if (unet->use_batchnormalization) {

		if (NULL == unet->deflatten_bn[level - 1])
			unet->deflatten_bn[level - 1] = deflatten_weights_create(result, ~((1lu << (index_counter_bn_in + 1)) - 2lu));

		int index_deflatten = nlop_get_nr_out_args(result);

		auto nlop_tmp = nlop_combine(result, unet->deflatten_bn[level - 1]);
		nlop_free(result);
		result = nlop_tmp;

		for (int i = 0; i < index_counter_bn_in; i++)
			result = nlop_link_F(result, index_deflatten, 1);

		auto iov = nlop_generic_codomain(result, 1);
		result = nlop_reshape_out_F(result, 1, 1, MAKE_ARRAY(md_calc_size(iov->N, iov->dims)));

		for(int i = 1; i < index_counter_bn_out; i++) {

			auto iov = nlop_generic_codomain(result, 2);
			result = nlop_reshape_out_F(result, 2, 1, MAKE_ARRAY(md_calc_size(iov->N, iov->dims)));
			result = nlop_stack_outputs_F(result, 1, 2, 0);
		}
	}

	result = nlop_shift_input_F(result, 1, nlop_get_nr_in_args(result) - 1);
	if (unet->use_bias)
		result = nlop_shift_input_F(result, 1, nlop_get_nr_in_args(result) - 1);
	if (unet->use_batchnormalization)
		result = nlop_shift_input_F(result, 1, nlop_get_nr_in_args(result) - 1);


	debug_printf(DP_DEBUG3, "level %d created: ", level);
	nlop_debug(DP_DEBUG3, result);

	return result;
}

static void nn_unet_allocate_deflatten_operators(struct unet_s* unet)
{
	if (NULL == unet->deflatten_conv){

		PTR_ALLOC(const struct nlop_s*[unet->number_levels], tmp);
		unet->deflatten_conv = *PTR_PASS(tmp);
		for (int i = 0; i < unet->number_levels; i++)
			unet->deflatten_conv[i] = NULL;
	}

	if ((unet->use_bias) && (NULL == unet->deflatten_bias)) {

		PTR_ALLOC(const struct nlop_s*[unet->number_levels], tmp);
		unet->deflatten_bias = *PTR_PASS(tmp);
		for (int i = 0; i < unet->number_levels; i++)
			unet->deflatten_bias[i] = NULL;
	}

	if ((unet->use_batchnormalization) && (NULL == unet->deflatten_bn)){

		PTR_ALLOC(const struct nlop_s*[unet->number_levels], tmp);
		unet->deflatten_bn = *PTR_PASS(tmp);
		for (int i = 0; i < unet->number_levels; i++)
			unet->deflatten_bn[i] = NULL;
	}
}

const struct nlop_s* nn_unet_create(struct unet_s* unet, long dims[5])
{
	nn_unet_allocate_deflatten_operators(unet);
	auto result = create_unet_level(unet, 1, dims, unet->channels);

	if (unet->real_constraint_input)
		result = nlop_chain2_swap_FF(nlop_from_linop_F(linop_zreal_create(5, dims)), 0, result, 0);

	if (unet->real_constraint_output)
		result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_zreal_create(5, dims)), 0);

	if (unet->real_constraint_weights)
		assert(0);

	debug_printf(DP_DEBUG1, "U-Net created: ");
	nlop_debug(DP_DEBUG1, result);

	return result;
}

long nn_unet_get_weights_size(struct unet_s* unet)
{
	long size_conv = 0;
	long size_bias = 0;
	long size_bn = 0;

	for (int i = 0; i < unet->number_levels; i++) {

		for (int j = 0; j < nlop_get_nr_out_args(unet->deflatten_conv[i]); j++) {

			auto iov = nlop_generic_codomain(unet->deflatten_conv[i], j);
			size_conv += md_calc_size(iov->N, iov->dims);
		}

		if (unet->use_bias)
			for (int j = 0; j < nlop_get_nr_out_args(unet->deflatten_bias[i]); j++) {

				auto iov = nlop_generic_codomain(unet->deflatten_bias[i], j);
				size_bias += md_calc_size(iov->N, iov->dims);
			}

		if (unet->use_batchnormalization)
			for (int j = 0; j < nlop_get_nr_out_args(unet->deflatten_bn[i]); j++) {

				auto iov = nlop_generic_codomain(unet->deflatten_bn[i], j);
				size_bn += md_calc_size(iov->N, iov->dims);
			}
	}

	long size = size_conv + size_bias + size_bn;

	if (0 == unet->size_weights_conv)
		unet->size_weights_conv = size_conv;
	else
		assert(unet->size_weights_conv == size_conv);

	if (0 == unet->size_weights_bias)
		unet->size_weights_bias = size_bias;
	else
		assert(unet->size_weights_bias == size_bias);

	if (0 == unet->size_weights_bn)
		unet->size_weights_bn = size_bn;
	else
		assert(unet->size_weights_bn == size_bn);

	if (0 == unet->size_weights)
		unet->size_weights = size;
	else
		assert(unet->size_weights == size);

	return size;
}

static void nn_unet_initialize_conv(struct unet_s* unet)
{
	complex float* tmp = unet->weights;

	for (int i = 0; i < unet->number_levels; i++) {

		auto iov = nlop_generic_domain(unet->deflatten_conv[i], 0);
		complex float* tmp_init_out = md_alloc_sameplace(iov->N, iov->dims, iov->size, tmp);

		for (int j = 0; j < nlop_get_nr_out_args(unet->deflatten_conv[i]); j++) {

			auto iov_in = nlop_generic_codomain(unet->deflatten_conv[i], j);
			complex float* tmp_init_in = md_alloc_sameplace(iov->N, iov->dims, iov->size, tmp);

			init_glorot_uniform_conv_complex(iov_in->N, iov_in->dims, tmp_init_in, true);
			if(unet->real_constraint_weights)
				md_zreal(iov_in->N, iov_in->dims, tmp_init_in, tmp_init_in);

			linop_adjoint_unchecked(nlop_get_derivative(unet->deflatten_conv[i], j, 0), tmp_init_out, tmp_init_in);
			md_zadd(iov->N, iov->dims, tmp, tmp, tmp_init_out);
			md_free(tmp_init_in);
		}

		tmp += md_calc_size(iov->N, iov->dims);
	}
}

void nn_unet_initialize(struct unet_s* unet, long dims[5])
{
	nlop_free(nn_unet_create(unet, dims));

	long size = nn_unet_get_weights_size(unet);

	if (NULL == unet->weights)
		unet->weights = md_alloc(1, MAKE_ARRAY(size), CFL_SIZE);

	md_clear(1, MAKE_ARRAY(unet->size_weights), unet->weights, CFL_SIZE);

	nn_unet_initialize_conv(unet);
}


int nn_unet_get_number_in_weights(struct unet_s* unet)
{
	int result = 1;
	if (unet->use_bias)
		result++;
	if (unet->use_batchnormalization)
		result++;

	result *= unet->number_levels;

	return result;
}

int nn_unet_get_number_out_weights(struct unet_s* unet)
{
	int result = 0;
	if (unet->use_batchnormalization)
		result++;
	result *= unet->number_levels;

	return result;
}

void nn_unet_get_in_weights_pointer(struct unet_s* unet, int N, complex float* args[N])
{
	assert(N == nn_unet_get_number_in_weights(unet));

	int j = 0;

	complex float* tmp_conv = unet->weights;
	complex float* tmp_bias = unet->weights + unet->size_weights_conv;
	complex float* tmp_bn = unet->weights + unet->size_weights_conv + unet->size_weights_bias;

	for (int i = 0; i < unet->number_levels; i++) {

		args[j++] = tmp_conv;
		auto iov_conv = nlop_generic_domain(unet->deflatten_conv[i], 0);
		tmp_conv += md_calc_size(iov_conv->N, iov_conv->dims);

		if (unet->use_bias) {

			args[j++] = tmp_bias;
			auto iov_bias = nlop_generic_domain(unet->deflatten_bias[i], 0);
			tmp_bias += md_calc_size(iov_bias->N, iov_bias->dims);
		}

		if (unet->use_batchnormalization) {

			args[j++] = tmp_bn;
			auto iov_bn = nlop_generic_domain(unet->deflatten_bn[i], 0);
			tmp_bn += md_calc_size(iov_bn->N, iov_bn->dims);
		}
	}

	assert(j == N);
	for (int i = 0; i < N; i++)
		assert(NULL != args[i]);
}

void nn_unet_get_in_types(struct unet_s* unet, int N, enum IN_TYPE in_types[N])
{
	assert(N == nn_unet_get_number_in_weights(unet));

	int j = 0;
	for (int i = 0; i < unet->number_levels; i++) {

		in_types[j++] = IN_OPTIMIZE;

		if (unet->use_bias)
			in_types[j++] = IN_OPTIMIZE;

		if (unet->use_batchnormalization)
			in_types[j++] = IN_BATCHNORM;
	}

	assert(j == N);
}

void nn_unet_get_out_types(struct unet_s* unet, int N, enum OUT_TYPE out_types[N])
{
	assert(N == nn_unet_get_number_out_weights(unet));

	int j = 0;
	for (int i = 0; i < unet->number_levels; i++)
		if (unet->use_batchnormalization)
			out_types[j++] = OUT_BATCHNORM;

	assert(j == N);
}

void nn_unet_load_weights(struct unet_s* unet, long size, _Complex float* in)
{
	assert((0 == unet->size_weights) || (size == unet->size_weights));
	unet->size_weights = size;

	if (NULL == unet->weights)
		unet->weights = md_alloc(1, MAKE_ARRAY(size), CFL_SIZE);

	md_copy(1, MAKE_ARRAY(size), unet->weights, in, CFL_SIZE);
}


void nn_unet_store_weights(const struct unet_s* unet, long size, _Complex float* out)
{
	assert((NULL != unet->weights) || (size == unet->size_weights));
	md_copy(1, MAKE_ARRAY(size), out, unet->weights, CFL_SIZE);
}

void nn_unet_move_cpugpu(struct unet_s* unet, bool gpu)
{
	assert((NULL != unet->weights) && (0 != unet->size_weights));

#ifdef USE_CUDA
	complex float* tmp = (gpu ? md_alloc_gpu : md_alloc)(1, MAKE_ARRAY(unet->size_weights), CFL_SIZE);
	md_copy(1, MAKE_ARRAY(unet->size_weights), tmp, unet->weights, CFL_SIZE);
	md_free(unet->weights);
	unet->weights = tmp;
#else
	assert(!gpu);
#endif
}

void nn_unet_free_weights(struct unet_s* unet)
{
	md_free(unet->weights);
}