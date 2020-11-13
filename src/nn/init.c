#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/shrdptr.h"
#include "misc/types.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "init.h"

typedef void (*initializer_del)(const struct initializer_s* conf);

typedef struct initializer_s{

	TYPEID* TYPEID;

	initializer_f fun;
	initializer_del del;
	struct shared_obj_s sptr;

} init_t;

static void init_del(const struct shared_obj_s* sptr)
{
	const struct initializer_s* x = CONTAINER_OF(sptr, const struct initializer_s, sptr);

	if (NULL != x->del)
		x->del(x);
	xfree(x);
}

void initializer_free(const struct initializer_s* x)
{
	if (NULL == x)
		return;

	shared_obj_destroy(&x->sptr);
}

const struct initializer_s* initializer_clone(const struct initializer_s* x)
{
	if (NULL != x)
		shared_obj_ref(&x->sptr);

	return x;
}

void initializer_apply(const struct initializer_s* x, long N, const long dims[N], complex float* weights)
{
	x->fun(x, N, dims, weights);
}


unsigned long in_flag_conv(bool c1)
{
	unsigned long in_flags = c1 ? MD_BIT(1) : MD_BIT(3);

	//filters, channel, kx, ky, kz    or x, y, z channel, filters
	for (int i = 0; i < 3; i++)
		in_flags |= MD_BIT(i + (c1 ? 0 : 2));
	return in_flags;
}

unsigned long out_flag_conv(bool c1)
{
	unsigned long out_flags = c1 ? MD_BIT(0) : MD_BIT(4);

	//filters, channel, kx, ky, kz    or x, y, z channel, filters
	for (int i = 0; i < 3; i++)
		out_flags |= MD_BIT(i + (c1 ? 0 : 2));
	return out_flags;
}

struct initializer_const_s {

	INTERFACE(init_t);
	complex float val;
};

static DEF_TYPEID(initializer_const_s);

static void init_const_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_const_s, conf_);
	md_zfill(N, dims, weights, conf->val);
}

const struct initializer_s* init_const_create(_Complex float val)
{
	PTR_ALLOC(struct initializer_const_s, data);
	SET_TYPEID(initializer_const_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_const_fun;

	data->val = val;

	return CAST_UP(PTR_PASS(data));
}

// Returns real/complex uniform/normal distribution with mean 0 and variance 1
static void get_base_dist(unsigned int N, const long dims[N], complex float* dst, bool uniform, bool real)
{
	if (uniform) {

		md_uniform_rand(N, dims, dst);
		md_zsadd(N, dims, dst, dst, (complex float)(-0.5));

		if (!real) {

			complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
			md_uniform_rand(N, dims, tmp);
			md_zsadd(N, dims, tmp, tmp, (complex float)(-0.5));
			md_zaxpy(N, dims, dst, 1.I, tmp);
			md_free(tmp);
		}

		md_zsmul(N, dims, dst, dst, real ? sqrt(12) : sqrt(6));
	} else {

		md_gaussian_rand(N, dims, dst);
		if (real)
			md_zreal(N, dims, dst, dst);
		else
			md_zsmul(N, dims, dst, dst, 1. / sqrt(2.));
	}
}

/*
Xavier Glorot, Yoshua Bengio ; Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, JMLR Workshop and Conference Proceedings 9:249-256, 2010.
Glorot, X. & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, in PMLR 9:249-256
*/
static float get_scaling_xavier(unsigned int N, const long dims[N], unsigned long in_flags, unsigned long out_flags)
{
	long tdims[N];
	md_select_dims(N, in_flags, tdims, dims);
	long inputs = md_calc_size(N, tdims);
	md_select_dims(N, out_flags, tdims, dims);
	long outputs = md_calc_size(N, tdims);

	return (float)sqrt(2. / (double)(inputs + outputs));
}

/*
He, K.; Zhang, X.; Ren, S. & Sun, J.
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
2015
*/
static float get_scaling_kaiming(unsigned int N, const long dims[N], unsigned long in_flags, float leaky_val)
{
	long tdims[N];
	md_select_dims(N, in_flags, tdims, dims);
	long inputs = md_calc_size(N, tdims);

	return (float)sqrt(2. / (double)(inputs) / (1. + leaky_val * leaky_val));
}

struct initializer_xavier_kaiming_s {

	INTERFACE(init_t);

	bool uniform;
	bool real;

	unsigned long in_flags;
	unsigned long out_flags;

	float leaky_val;
};

static DEF_TYPEID(initializer_xavier_kaiming_s);

static void init_xavier_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_xavier_kaiming_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, get_scaling_xavier(N, dims, conf->in_flags, conf->out_flags));
}

static void init_kaiming_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_xavier_kaiming_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, get_scaling_kaiming(N, dims, conf->in_flags, conf->leaky_val));
}


const struct initializer_s* init_xavier_create(unsigned long in_flags, unsigned long out_flags, bool real, bool uniform)
{
	PTR_ALLOC(struct initializer_xavier_kaiming_s, data);
	SET_TYPEID(initializer_xavier_kaiming_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_xavier_fun;

	data->in_flags = in_flags;
	data->out_flags = out_flags;
	data->uniform = uniform;
	data->real = real;
	data->leaky_val = 1;

	return CAST_UP(PTR_PASS(data));
}

const struct initializer_s* init_kaiming_create(unsigned long in_flags, bool real, bool uniform, float leaky_val)
{
	PTR_ALLOC(struct initializer_xavier_kaiming_s, data);
	SET_TYPEID(initializer_xavier_kaiming_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_kaiming_fun;

	data->in_flags = in_flags;
	data->out_flags = 0;
	data->uniform = uniform;
	data->real = real;
	data->leaky_val = leaky_val;

	return CAST_UP(PTR_PASS(data));
}

struct initializer_std_normal_s {

	INTERFACE(init_t);

	bool uniform;
	bool real;

	float scale;
	float mean;
};

static DEF_TYPEID(initializer_std_normal_s);

static void init_std_normal_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_std_normal_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, conf->scale);
	md_zsadd(N, dims, weights, weights, conf->mean + (conf->real ? 0 : I * conf->mean));
}

const struct initializer_s* init_std_normal_create(bool real, float scale, float mean)
{
	PTR_ALLOC(struct initializer_std_normal_s, data);
	SET_TYPEID(initializer_std_normal_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_std_normal_fun;

	data->uniform = false;
	data->real = real;
	data->scale = scale;
	data->mean = mean;

	return CAST_UP(PTR_PASS(data));
}

const struct initializer_s* init_uniform_create(bool real, float scale, float mean)
{
	PTR_ALLOC(struct initializer_std_normal_s, data);
	SET_TYPEID(initializer_std_normal_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_std_normal_fun;

	data->uniform = true;
	data->real = real;
	data->scale = scale;
	data->mean = mean;

	return CAST_UP(PTR_PASS(data));
}

struct initializer_linspace_s {

	INTERFACE(init_t);

	unsigned int dim;

	complex float min_val;
	complex float max_val;

	bool max_inc;
};

static DEF_TYPEID(initializer_linspace_s);

static void init_linspace_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_linspace_s, conf_);

	assert(conf->dim < N);

	complex float vals[dims[conf->dim]];
	for (int i = 0; i < dims[conf->dim]; i++)
		vals[i] = conf->min_val + i *(conf->max_val - conf->min_val) / ((float)dims[conf->dim] - (conf->max_inc? 1. : 0));

	long vdims[N];
	md_select_dims(N, MD_BIT(conf->dim), vdims, dims);

	md_copy2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), weights, MD_STRIDES(N, vdims, CFL_SIZE), vals, CFL_SIZE);
}

const struct initializer_s* init_linspace_create(unsigned int dim, complex float min_val, complex float max_val, bool max_inc)
{
	PTR_ALLOC(struct initializer_linspace_s, data);
	SET_TYPEID(initializer_linspace_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_linspace_fun;

	data->dim = dim;
	data->max_val = max_val;
	data->max_inc = max_inc;
	data->min_val = min_val;

	return CAST_UP(PTR_PASS(data));
}




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
