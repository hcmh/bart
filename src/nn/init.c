#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"

#include "nlops/nlop_props.h"
#include "nlops/nlop.h"

#include "init.h"

static float get_dense_init_scaling(const long weight_dims[2])
{
	return (float)sqrt(6. / (double)(weight_dims[0] + weight_dims[0]));
}

static float get_conv_init_scaling(const long kernel_dims[5], bool channel_first)
{
	//filters, channel, kx, ky, kz    or x, y, z channel, filters
	int in_neurons = channel_first ? kernel_dims[1] : kernel_dims[3];
	int out_neurons = channel_first ? kernel_dims[0] : kernel_dims[4];

	for (int i = 0; i < 3; i++){

		in_neurons *= kernel_dims[i + (channel_first ? 0 : 2)];
		out_neurons *= kernel_dims[i + (channel_first ? 0 : 2)];
	}

	return (float)sqrt(6. / (double)(in_neurons + out_neurons));
}


complex float* init_glorot_uniform_dense(long N, const long* weight_dims, complex float* src, _Bool c1)
{
	UNUSED(c1);

	long size = md_calc_size(N, weight_dims);
	md_uniform_rand(1, &size, src);
	md_zsadd(1, &size, src, src, (complex float)(-0.5));
	md_zreal(1, &size, src, src);
	md_zsmul(1, &size, src, src, (complex float)(2 * get_dense_init_scaling(weight_dims)));
	return src + size;
}

complex float* init_glorot_uniform_conv(long N, const long* kernel_dims, complex float* src, _Bool c1)
{
	long size = md_calc_size(N, kernel_dims);
	md_uniform_rand(1, &size, src);
	md_zsadd(1, &size, src, src, (complex float)(-0.5));
	md_zreal(1, &size, src, src);
	md_zsmul(1, &size, src, src, (complex float)(2 * get_conv_init_scaling(kernel_dims, c1)));
	return src + size;
}

complex float* init_bias(long N, const long* bias_dims, complex float* src, _Bool c1)
{
	UNUSED(c1);
	UNUSED(N);

	long size = bias_dims[0];
	md_clear(1, &size, src, CFL_SIZE);
	return src + size;
}
/**
 * Initialize weights with initializer guessed from dimensions
 * Returns pointer to the next array (useful if weights are continous in memory)
 * c1 is flag for channel first
 * */
complex float* init_auto(long N, const long* dims, complex float* src, _Bool c1)
{
	if (N==1)
		return init_bias(N, dims, src, c1);
	if (N==2)
		return init_glorot_uniform_dense(N, dims, src, c1);
	if (N==5)
		return init_glorot_uniform_conv(N, dims, src, c1);

	assert(0);
	return NULL;
}

enum OPERATOR_IO_PROP_FLAGS_INDEX weight_types[] = {
	OP_PROP_NN_IN_WEIGHT_CONV_CF,
	OP_PROP_NN_IN_WEIGHT_CONV_CL,
	OP_PROP_NN_IN_WEIGHT_DENSE,
	OP_PROP_NN_IN_WEIGHT_BIAS,
	OP_PROP_NN_BATCH_NORM
	};


static enum OPERATOR_IO_PROP_FLAGS_INDEX nlop_get_input_weight_type(operator_prop_flags_t flags)
{
	enum OPERATOR_IO_PROP_FLAGS_INDEX result = OP_PROP_NN_IN_WEIGHT_NOT_DEFINED;

	for(int i = 0; (unsigned long)i < sizeof(weight_types) / sizeof(weight_types[0]); i++)
		if (MD_IS_SET(flags, weight_types[i])) {

			assert(result == OP_PROP_NN_IN_WEIGHT_NOT_DEFINED);
			result = weight_types[i];
		}

	return result;
}

/**
 * Initialize weights with initializer guessed from nlop properties
 * Returns pointer to the next array (useful if weights are continous in memory)
 * */
complex float* init_auto_nlop_props(const struct nlop_s* op, unsigned int weight_index, complex float* src)
{
	unsigned int N = nlop_generic_domain(op, weight_index)->N;
	const long* dims = nlop_generic_domain(op, weight_index)->dims;
	size_t size = nlop_generic_domain(op, weight_index)->size;

	enum debug_levels dl = DP_INFO;

	switch (nlop_get_input_weight_type(nlop_get_ii_props(op, weight_index, weight_index))) {

		case OP_PROP_NN_IN_WEIGHT_CONV_CF:
			debug_printf(dl, "\tinput %d (conv weight) initialized with glorot uniform\n", weight_index);
			return init_glorot_uniform_conv(N, dims, src, true);

		case OP_PROP_NN_IN_WEIGHT_CONV_CL:
			debug_printf(dl, "\tinput %d (conv weight) initialized with glorot uniform\n", weight_index);
			return init_glorot_uniform_conv(N, dims, src, false);

		case OP_PROP_NN_IN_WEIGHT_DENSE:
			debug_printf(dl, "\tinput %d (dense weight) initialized with glorot uniform\n", weight_index);
			return init_glorot_uniform_dense(N, dims, src, false);

		case OP_PROP_NN_IN_WEIGHT_BIAS:
			debug_printf(dl, "\tinput %d (bias weight) initialized with zeros\n", weight_index);
			md_clear(N, dims, src, size);
			return src + md_calc_size(N, dims);

		case OP_PROP_NN_BATCH_NORM:
			debug_printf(dl, "\tinput %d (batch norm) initialized with zeros\n", weight_index);
			md_clear(N, dims, src, size);
			return src + md_calc_size(N, dims);

		case OP_PROP_NN_IN_WEIGHT_NOT_DEFINED:
			debug_printf(dl, "\tinput %d (undifined) initialized with gaussian rand\n", weight_index);
			md_uniform_rand(N, dims, src);
			md_zreal(N, dims, src, src);
			return src + md_calc_size(N, dims);

		default:
			assert(0);
	}

	assert(0);
	return NULL;
}

void init_nlop_weights(const struct nlop_s* op, unsigned long weight_flags, complex float* src)
{
	unsigned int NI = nlop_get_nr_in_args(op);

	for(unsigned int i = 0; i < NI; i++)
		if(MD_IS_SET(weight_flags, i))
			src = init_auto_nlop_props(op, i, src);
}
