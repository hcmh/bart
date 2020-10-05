#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "misc/debug.h"
#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "nn/initializer.h"
#include "init.h"


/**
 * Initializes weights using glorot/xavier methode, i.e. uniform distribution with std=sqrt(2/(n_in+n_out))
 * Note that only the real part is initialized
 */
complex float* init_glorot_uniform_dense(long N, const long* weight_dims, complex float* src, _Bool c1)
{
	UNUSED(c1);

	const struct initializer_s* init = init_xavier_create(MD_BIT(1), MD_BIT(0), true, true);
	initializer_apply(init, N, weight_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, weight_dims);
}


/**
 * Initializes weights using glorot/xavier methode, i.e. uniform distribution with std=sqrt(2/(n_in+n_out))
 * Note that only the real part is initialized
 */
complex float* init_glorot_uniform_conv(long N, const long* kernel_dims, complex float* src, _Bool c1)
{

	const struct initializer_s* init = init_xavier_create(in_flag_conv(c1), out_flag_conv(c1), true, true);
	initializer_apply(init, N, kernel_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, kernel_dims);
}

/**
 * Initializes weights using glorot/xavier methode, i.e. uniform distribution with std=sqrt(2/(n_in+n_out))
 * Note that the variance is understood as variance of complex random variables
 */
complex float* init_glorot_uniform_conv_complex(long N, const long* kernel_dims, complex float* src, _Bool c1)
{
	const struct initializer_s* init = init_xavier_create(in_flag_conv(c1), out_flag_conv(c1), false, true);
	initializer_apply(init, N, kernel_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, kernel_dims);
}

/**
 * Initializes weights using kaiming methode, i.e. uniform distribution with std=sqrt(2/(n_in * (1 + leaky_val**2)))
 * Note that the variance is understood as variance of complex random variables
 */
complex float* init_kaiming_uniform_conv(long N, const long* kernel_dims, complex float* src, _Bool c1, float leaky_val)
{
	const struct initializer_s* init = init_kaiming_create(in_flag_conv(c1), true, true, leaky_val);
	initializer_apply(init, N, kernel_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, kernel_dims);
}
/**
 * Initializes weights using kaiming methode, i.e. uniform distribution with std=sqrt(2/(n_in * (1 + leaky_val**2)))
 * Note that the variance is understood as variance of complex random variables
 */
complex float* init_kaiming_uniform_conv_complex(long N, const long* kernel_dims, complex float* src, _Bool c1, float leaky_val)
{
	const struct initializer_s* init = init_kaiming_create(in_flag_conv(c1), false, true, leaky_val);
	initializer_apply(init, N, kernel_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, kernel_dims);
}

/**
 * Initializes weights using kaiming methode, i.e. uniform distribution with std=sqrt(2/(n_in * (1 + leaky_val**2)))
 * Note that the variance is understood as variance of complex random variables
 */
complex float* init_kaiming_normal_conv_complex(long N, const long* kernel_dims, complex float* src, _Bool c1, float leaky_val)
{
	const struct initializer_s* init = init_kaiming_create(in_flag_conv(c1), false, false, leaky_val);
	initializer_apply(init, N, kernel_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, kernel_dims);
}

/**
 * Initializes weights using kaiming methode, i.e. uniform distribution with std=sqrt(2/(n_in * (1 + leaky_val**2)))
 * Note that the variance is understood as variance of complex random variables
 */
complex float* init_kaiming_normal_conv(long N, const long* kernel_dims, complex float* src, _Bool c1, float leaky_val)
{
	const struct initializer_s* init = init_kaiming_create(in_flag_conv(c1), true, false, leaky_val);
	initializer_apply(init, N, kernel_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, kernel_dims);
}

complex float* init_bias(long N, const long* bias_dims, complex float* src, _Bool c1)
{
	UNUSED(c1);

	const struct initializer_s* init = init_const_create(0.);
	initializer_apply(init, N, bias_dims, src);
	initializer_free(init);

	return src + md_calc_size(N, bias_dims);
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
