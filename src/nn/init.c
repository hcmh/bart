#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "misc/debug.h"
#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
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

complex float* init_glorot_uniform_conv_complex(long N, const long* kernel_dims, complex float* src, _Bool c1)
{
	long size = md_calc_size(N, kernel_dims);
	md_uniform_rand(1, &size, src);
	md_zsadd(1, &size, src, src, (complex float)(-0.5));
	complex float* tmp = md_alloc_sameplace(1, &size, CFL_SIZE, src);
	md_uniform_rand(1, &size, tmp);
	md_zsadd(1, &size, tmp, tmp, (complex float)(-0.5));
	md_zaxpy(1, &size, src, 1.I, tmp);
	md_free(tmp);
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
