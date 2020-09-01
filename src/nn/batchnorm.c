
#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/rand.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/const.h"

#include "nn/layers.h"

#include "batchnorm.h"


struct stats_s {

	INTERFACE(nlop_data_t);

	unsigned long flags;
	const struct iovec_s* dom;
	const struct iovec_s* codom;

	complex float n;

	complex float* x;
};

DEF_TYPEID(stats_s);


static void stats_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	START_TIMER;
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* mean = args[0];
	complex float* var = args[1];
	complex float* src = args[2];

	if (NULL == data->x)
		data->x = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, args[0]);
#ifdef USE_CUDA
	assert((cuda_ondevice(mean) == cuda_ondevice(src)) && (cuda_ondevice(var) == cuda_ondevice(src)));
#endif

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		long tdims[data->dom->N];
		md_select_dims(data->dom->N, data->flags, tdims, data->dom->dims);
		complex float* ones = md_alloc_sameplace(data->dom->N, tdims, CFL_SIZE, src);

		md_zfill(data->dom->N, tdims, ones, 1.);
		md_ztenmul(data->dom->N, data->codom->dims, mean, data->dom->dims, src, tdims, ones);
		md_zsmul(data->dom->N, data->codom->dims, mean, mean, 1. / data->n);

		complex float* tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, src);
		md_copy2(data->dom->N, data->dom->dims, data->dom->strs, tmp, data->codom->strs, mean, CFL_SIZE);
		md_zsub(data->dom->N, data->dom->dims, data->x, src, tmp);

		md_zmulc(data->dom->N, data->dom->dims, tmp, data->x, data->x);
		md_ztenmul(data->dom->N, data->codom->dims, var, data->dom->dims, tmp, tdims, ones);
		md_zsmul(data->codom->N, data->codom->dims, var, var, 1. / data->n);

		md_free(tmp);
		md_free(ones);
	} else
#endif
	{
		md_zsum(data->dom->N, data->dom->dims, data->flags, mean, src);
		md_zsmul(data->dom->N, data->codom->dims, mean, mean, 1. / data->n);
		md_zsub2(data->dom->N, data->dom->dims, data->dom->strs, data->x, data->dom->strs, src, data->codom->strs, mean);
		md_ztenmulc(data->dom->N, data->codom->dims, var, data->dom->dims, data->x, data->dom->dims, data->x);
		md_zsmul(data->codom->N, data->codom->dims, var, var, 1. / data->n);
	}

	md_zreal(data->codom->N, data->codom->dims, var, var);

	PRINT_TIMER("frw stats");
}

static void stats_der_mean(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	START_TIMER;
	const auto data = CAST_DOWN(stats_s, _data);

	md_zsum(data->dom->N, data->dom->dims, data->flags, dst, src);
	md_zsmul(data->dom->N, data->codom->dims, dst, dst, 1. / data->n);
	PRINT_TIMER("der stats mean ");
}

static void stats_adj_mean(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	START_TIMER;
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* tmp = md_alloc_sameplace(data->codom->N, data->codom->dims, data->codom->size, src);

	md_zsmul(data->codom->N, data->codom->dims, tmp, src, 1. / data->n);
	md_copy2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->codom->strs, tmp, data->dom->size);
	md_free(tmp);
	PRINT_TIMER("adj stats mean ");
}

static void stats_der_var(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	START_TIMER;
	const auto data = CAST_DOWN(stats_s, _data);

	md_ztenmulc(data->dom->N, data->codom->dims, dst, data->dom->dims, src, data->dom->dims, data->x);
	md_zsmul(data->codom->N, data->codom->dims, dst, dst, (2. / data->n));
	md_zreal(data->codom->N, data->codom->dims, dst, dst);
	PRINT_TIMER("der stats var");
}

static void stats_adj_var(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	START_TIMER;
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* tmp = md_alloc_sameplace(data->codom->N, data->codom->dims, data->codom->size, src);
	md_zreal(data->codom->N, data->codom->dims, tmp, src);
	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->codom->strs, tmp, data->dom->strs, data->x);
	md_zsmul(data->dom->N, data->dom->dims, dst, dst, (2. / data->n));
	md_free(tmp);
	PRINT_TIMER("adj stats var");
}


static void stats_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(stats_s, _data);
	md_free(data->x);

	iovec_free(data->dom);
	iovec_free(data->codom);

	xfree(data);
}

/**
 * Nlop to compute mean and variance of input
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
 *
 * In 0:	Input
 * Out 0:	Mean \mu = \sum_{i=1}^N x_i/N
 * Out 1: 	Variance \var = \sum_{i=1}^N |(x_i-\mu)|^2/N
 *
 * Note the difference of the definition compared to md_zvar which has factor 1/(N-1)
 **/
const struct nlop_s* nlop_stats_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct stats_s, data);
	SET_TYPEID(stats_s, data);

	// will be initialized later, to transparently support GPU
	data->x = NULL;

	long codims[N];
	md_select_dims(N, ~flags, codims, dims);

	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->codom = iovec_create(N, codims, CFL_SIZE);
	data->flags = flags;

	data->n = md_calc_size(N, dims) / md_calc_size(N, codims);

	long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], codims);
	md_copy_dims(N, nl_odims[1], codims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);


	return nlop_generic_create(2, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)),
		stats_fun, (nlop_der_fun_t[1][2]){ { stats_der_mean, stats_der_var } }, (nlop_der_fun_t[1][2]){ { stats_adj_mean, stats_adj_var } }, NULL, NULL, stats_del);
}

struct normalize_s {

	INTERFACE(nlop_data_t);

	const struct iovec_s* dom;
	const struct iovec_s* statdom;

	unsigned long flags;

	complex float* tmp; // (src - mu)
	complex float* scale; // sqrt(var + epsilon)

	float epsilon;
};

DEF_TYPEID(normalize_s);

static void normalize_clear_der_var(struct normalize_s* data)
{
	md_free(data->tmp);
	data->tmp = NULL;
}

static void normalize_fun(const nlop_data_t* _data, int N, complex float* args[N], const struct op_options_s* opts)
{
	START_TIMER;
	const auto data = CAST_DOWN(normalize_s, _data);

	assert(4 == N);

	complex float* dst = args[0];
	complex float* src = args[1];
	complex float* mean = args[2];
	complex float* var = args[3];

	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, dst);
	if (NULL == data->scale)
		data->scale = md_alloc_sameplace(data->statdom->N, data->statdom->dims, data->statdom->size, dst);

	md_zsadd(data->statdom->N, data->statdom->dims, data->scale, var, data->epsilon);
	md_zreal(data->statdom->N, data->statdom->dims, data->scale, data->scale); //assert that sigma is real
	//md_zsqrt(data->statdom->N, data->statdom->dims, data->scale, data->scale);
	md_sqrt(data->statdom->N + 1, MD_REAL_DIMS(data->statdom->N, data->statdom->dims), (float*)data->scale, (float*)data->scale);

#ifdef USE_CUDA //FIXME: Optimize zsub2, zdiv2 for these strides
	if (cuda_ondevice(src)) {

		complex float* tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, src);

		md_copy2(data->dom->N, data->dom->dims, data->dom->strs, tmp, data->statdom->strs, mean, CFL_SIZE);
		md_zsub(data->dom->N, data->dom->dims, data->tmp, src, tmp);

		md_copy2(data->dom->N, data->dom->dims, data->dom->strs, tmp, data->statdom->strs, data->scale, CFL_SIZE);
		md_zdiv(data->dom->N, data->dom->dims, dst, data->tmp, tmp);

		md_free(tmp);
	} else
#endif
	{
		md_zsub2(data->dom->N, data->dom->dims, data->dom->strs, data->tmp, data->dom->strs, src, data->statdom->strs, mean);
		md_zdiv2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, data->tmp, data->statdom->strs, data->scale);
	}

	if (op_options_is_set_io(opts, 0, 2, OP_APP_NO_DER))
		normalize_clear_der_var(data);

	PRINT_TIMER("frw normalize");
}

static void normalize_deradj_src(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto data = CAST_DOWN(normalize_s, _data);

#ifdef USE_CUDA //FIXME: Optimize zsub2, zdiv2 for these strides
	if (cuda_ondevice(src)) {

		complex float* tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, src);

		md_copy2(data->dom->N, data->dom->dims, data->dom->strs, tmp, data->statdom->strs, data->scale, CFL_SIZE);
		md_zdiv(data->dom->N, data->dom->dims, dst, src, tmp);

		md_free(tmp);
	} else
#endif
		md_zdiv2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, data->scale);

	PRINT_TIMER("der/adj normalize src");
}

static void normalize_der_mean(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto data = CAST_DOWN(normalize_s, _data);
	md_zdiv2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, src, data->statdom->strs, data->scale);
	md_zsmul(data->dom->N, data->dom->dims, dst, dst, -1.);
	PRINT_TIMER("der normalize mean");
}

static void normalize_adj_mean(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto data = CAST_DOWN(normalize_s, _data);

#ifdef USE_CUDA //FIXME: Optimize zsub2, zdiv2 for these strides
	if (cuda_ondevice(src)) {

		long tdims[data->dom->N];
		md_select_dims(data->dom->N, data->flags, tdims, data->dom->dims);

		complex float* tmp = md_alloc_sameplace(data->dom->N, tdims, CFL_SIZE, src);
		md_zfill(data->dom->N, tdims, tmp, 1.);
		md_ztenmul(data->dom->N, data->statdom->dims, dst, data->dom->dims, src, tdims, tmp);

		md_free(tmp);
	} else
#endif
	{
		md_clear(data->statdom->N, data->statdom->dims, dst, data->statdom->size);
		md_zadd2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->statdom->strs, dst, data->dom->strs, src);
	}

	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, dst, dst, -1.);
	PRINT_TIMER("adj normalize mean");
}

static void normalize_der_var(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto data = CAST_DOWN(normalize_s, _data);

	assert(NULL != data->tmp);

	complex float* tmp = md_alloc_sameplace(data->statdom->N, data->statdom->dims, data->statdom->size, dst);

	md_zreal(data->statdom->N, data->statdom->dims, tmp, src);

	md_zdiv(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, tmp, tmp, -.5);

	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, tmp, data->dom->strs, data->tmp);



	md_free(tmp);
	PRINT_TIMER("der normalize var");
}

static void normalize_adj_var(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;
	const auto data = CAST_DOWN(normalize_s, _data);

	assert(NULL != data->tmp);

	md_clear(data->statdom->N, data->statdom->dims, dst, data->statdom->size);

#if 0
	//change when zfmacc is optimized for this case
	md_zfmac2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->dom->strs, src, data->dom->strs, tmp);
#else
	complex float* tmp = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, dst);
	md_zmulc(data->dom->N, data->dom->dims, tmp, src, data->tmp);

#ifdef USE_CUDA //FIXME: optimize zadd2 for accumulation in first or second dim
	if (cuda_ondevice(src)) {

		long tdims[data->dom->N];
		md_select_dims(data->dom->N, data->flags, tdims, data->dom->dims);

		complex float* ones = md_alloc_sameplace(data->dom->N, tdims, CFL_SIZE, src);
		md_zfill(data->dom->N, tdims, ones, 1.);
		md_ztenmul(data->dom->N, data->statdom->dims, dst, data->dom->dims, tmp, tdims, ones);

		md_free(ones);
	} else
#endif
		md_zadd2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->statdom->strs, dst, data->dom->strs, tmp);

	md_free(tmp);
#endif

	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, dst, dst, -.5);
	md_zreal(data->statdom->N, data->statdom->dims, dst, dst);

	PRINT_TIMER("adj normalize var");
}

static void normalize_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	md_free(data->scale);
	md_free(data->tmp);

	iovec_free(data->dom);
	iovec_free(data->statdom);

	xfree(data);
}

/**
 * Nlop to normalize input by given mean/variance
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that are not present in mean/variance
 * @param epsilon to update the floating mean and varinace
 *
 * In 0:	Input
 * In 1:	Mean mu
 * In 2: 	Variance sigma^2

 * Out 0:	Normalized input (x - mu) / sqrt(sigma^2 + epsilon)
 *
 **/
const struct nlop_s* nlop_normalize_create(int N, const long dims[N], unsigned long flags, float epsilon)
{
	PTR_ALLOC(struct normalize_s, data);
	SET_TYPEID(normalize_s, data);


	long statdims[N];
	md_select_dims(N, ~flags, statdims, dims);

	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->statdom = iovec_create(N, statdims, CFL_SIZE);
	data->epsilon = epsilon;
	data->scale = NULL;
	data->tmp = NULL;
	data->flags = flags;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], statdims);
	md_copy_dims(N, nl_idims[2], statdims);

	operator_property_flags_t props[3][1] = { {0} ,{0}, {0}};

	return nlop_generic_with_props_create(	1, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)), normalize_fun,
						(nlop_der_fun_t[3][1]){ { normalize_deradj_src}, { normalize_der_mean }, { normalize_der_var } },
						(nlop_der_fun_t[3][1]){ { normalize_deradj_src}, { normalize_adj_mean }, { normalize_adj_var } },
						NULL, NULL, normalize_del, props);
}

struct rescale_s {

	INTERFACE(nlop_data_t);

	const struct iovec_s* dom;
	const struct iovec_s* statdom;

	complex float* scale;
	complex float* in;

};

DEF_TYPEID(rescale_s);


static void rescale_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(rescale_s, _data);

	assert(4 == N);

	complex float* dst = args[0];
	complex float* src = args[1];
	complex float* beta = args[2];
	complex float* gamma = args[3];

	if (NULL == data->scale)
		data->scale = md_alloc_sameplace(data->statdom->N, data->statdom->dims, data->statdom->size, dst);
	if (NULL == data->in)
		data->in = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, dst);

	md_copy(data->statdom->N, data->statdom->dims, data->scale, gamma, data->statdom->size);
	md_copy(data->dom->N, data->dom->dims, data->in, src, data->dom->size);

	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, gamma);
	md_zadd2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, dst, data->statdom->strs, beta);
}

static void rescale_der_src(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(rescale_s, _data);
	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, data->scale);
}

static void rescale_adj_src(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(rescale_s, _data);
	md_zmulc2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, data->scale);
}

static void rescale_der_gamma(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(rescale_s, _data);
	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, data->in, data->statdom->strs, src);
}

static void rescale_adj_gamma(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(rescale_s, _data);
	md_ztenmulc2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->dom->strs, src, data->dom->strs, data->in);
}

static void rescale_der_beta(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(rescale_s, _data);
	md_copy2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, src, data->dom->size);
}

static void rescale_adj_beta(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(rescale_s, _data);
	md_clear(data->statdom->N, data->statdom->dims, dst, data->dom->size);
	md_zadd2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->statdom->strs, dst, data->dom->strs, src);
}

static void rescale_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(rescale_s, _data);

	md_free(data->scale);
	md_free(data->in);

	iovec_free(data->dom);
	iovec_free(data->statdom);

	xfree(data);
}

/**
 * Shift/Scale output of batchnorm
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that are not present in Beta/Gamma
 * @param epsilon to update the floating mean and varinace
 *
 * In 0:	Input
 * In 1:	Beta (new mean)
 * In 2: 	Gamma

 * Out 0:	Recaled input
 *
 **/
const struct nlop_s* nlop_scale_and_shift_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct rescale_s, data);
	SET_TYPEID(rescale_s, data);


	long statdims[N];
	md_select_dims(N, ~flags, statdims, dims);

	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->statdom = iovec_create(N, statdims, CFL_SIZE);

	data->scale = NULL;
	data->in = NULL;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], statdims);
	md_copy_dims(N, nl_idims[2], statdims);


	return nlop_generic_create(1, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)), rescale_fun,
					(nlop_der_fun_t[3][1]){ { rescale_der_src}, { rescale_der_beta }, { rescale_der_gamma } },
					(nlop_der_fun_t[3][1]){ { rescale_adj_src}, { rescale_adj_beta }, { rescale_adj_gamma } },
					NULL, NULL, rescale_del);
}


struct bn_s {

	INTERFACE(nlop_data_t);

	unsigned long flags;
	const struct iovec_s* dom;
	const struct iovec_s* stat_dom;

	float mean_size;

	complex float* out;
	complex float* scale; // 1 / sqrt(simga_b^2 +epsilon)
	complex float* ones;

	complex float epsilon;
};

DEF_TYPEID(bn_s);

static void bn_clear_der(struct bn_s* data)
{
	md_free(data->out);
	md_free(data->scale);
	data->out = NULL;
	data->scale = NULL;
}

static void bn_fun(const nlop_data_t* _data, int D, complex float* args[D], const struct op_options_s* opts)
{
	START_TIMER;
	const auto data = CAST_DOWN(bn_s, _data);

	complex float* out = args[0];
	complex float* mean = args[1];
	complex float* var = args[2];

	complex float* src = args[3];
	assert(4 == D);

	bool der = !op_options_is_set_io(opts, 0, 0, OP_APP_NO_DER);

	unsigned int N = data->dom->N;

	long nstat_dims[N]; //dims that not stay
	long nstat_strs[N];
	md_select_dims(N, data->flags, nstat_dims, data->dom->dims);
	md_calc_strides(N, nstat_strs, nstat_dims, CFL_SIZE);

	if (NULL == data->out)
		data->out = md_alloc_sameplace(N, data->dom->dims, CFL_SIZE, args[0]);
	if (NULL == data->scale)
		data->scale = md_alloc_sameplace(N, data->stat_dom->dims, CFL_SIZE, args[0]);
	if (NULL == data->ones) {

		data->ones = md_alloc_sameplace(N, nstat_dims, CFL_SIZE, args[0]);
		md_zfill(N, nstat_dims, data->ones, 1.);
	}

	//compute mean
	md_ztenmul(N, data->stat_dom->dims, mean, data->dom->dims, src, nstat_dims, data->ones);
	md_zsmul(N, data->stat_dom->dims, mean, mean, 1. / data->mean_size);

	//compute var
	md_copy2(N, data->dom->dims, data->dom->strs, data->out, data->stat_dom->strs, mean, CFL_SIZE);
	md_zsub(N, data->dom->dims, data->out, src, data->out);

	md_zmulc(N, data->dom->dims, out, data->out, data->out);
	md_ztenmul(N, data->stat_dom->dims, var, data->dom->dims, out, nstat_dims, data->ones);
	md_zsmul(N, data->stat_dom->dims, var, var, 1. / data->mean_size);
	md_zreal(N, data->stat_dom->dims, var, var);

	//compute scale (1/sqrt(var + epsilon))
	md_zsadd(N, data->stat_dom->dims, data->scale, var, data->epsilon);
	md_sqrt(N + 1, MD_REAL_DIMS(N, data->stat_dom->dims), (float*)data->scale, (float*)data->scale);

	complex float* ones_tmp = md_alloc_sameplace(N, data->stat_dom->dims, CFL_SIZE, data->scale);
	md_zfill(N, data->stat_dom->dims, ones_tmp, 1.);
	md_zdiv(N, data->stat_dom->dims, data->scale, ones_tmp, data->scale);
	md_free(ones_tmp);

	md_zmul2(N, data->dom->dims, data->dom->strs, out, data->dom->strs, data->out, data->stat_dom->strs, data->scale);
	md_copy(N, data->dom->dims, data->out, out, CFL_SIZE);

	//output unbiased variance
	md_zsmul(N, data->stat_dom->dims, var, var, data->mean_size / (data->mean_size - 1));

	md_free(data->ones);
	data->ones = NULL;

	if (!der)
		bn_clear_der(data);

	PRINT_TIMER("frw stats");
}

static void bn_der_mean(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(_data);
	UNUSED(dst);
	UNUSED(src);
	error("der bn mean not implemented");
}

static void bn_adj_mean(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(_data);
	UNUSED(dst);
	UNUSED(src);
	error("adj bn mean not implemented");
}

static void bn_der_var(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(_data);
	UNUSED(dst);
	UNUSED(src);
	error("der bn var not implemented");
}

static void bn_adj_var(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(_data);
	UNUSED(dst);
	UNUSED(src);
	error("adj bn var not implemented");
}

static void bn_deradj_in(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	START_TIMER;
	const auto data = CAST_DOWN(bn_s, _data);

	assert(NULL != data->out);

	unsigned int N = data->dom->N;

	long nstat_dims[N]; //dims that not stay
	long nstat_strs[N];
	md_select_dims(N, data->flags, nstat_dims, data->dom->dims);
	md_calc_strides(N, nstat_strs, nstat_dims, CFL_SIZE);

	if (NULL == data->ones) {

		data->ones = md_alloc_sameplace(N, nstat_dims, CFL_SIZE, dst);
		md_zfill(N, nstat_dims, data->ones, 1.);
	}

	md_zmul2(N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->stat_dom->strs, data->scale);

	complex float* stat_tmp = md_alloc_sameplace(N, data->stat_dom->dims, CFL_SIZE, dst);
	complex float* tmp = md_alloc_sameplace(N, data->dom->dims, CFL_SIZE, dst);


	//derivative through sigma_b
	md_zmulc(N, data->dom->dims, tmp, dst, data->out);
	md_ztenmul(N, data->stat_dom->dims, stat_tmp, data->dom->dims, tmp, nstat_dims, data->ones);
	md_zreal(N, data->stat_dom->dims, stat_tmp, stat_tmp);
	md_zsmul(N, data->stat_dom->dims, stat_tmp, stat_tmp, 1. / data->mean_size);

	md_zmul2(N, data->dom->dims, data->dom->strs, tmp, data->dom->strs, data->out, data->stat_dom->strs, stat_tmp);
	md_zsub(N, data->dom->dims, dst, dst, tmp);

	//derivative through mu_b
	md_ztenmul(N, data->stat_dom->dims, stat_tmp, data->dom->dims, src, nstat_dims, data->ones);
	md_zsmul(N, data->stat_dom->dims, stat_tmp, stat_tmp, -1. / data->mean_size);
	md_zmul(N, data->stat_dom->dims, stat_tmp, stat_tmp, data->scale);
	md_zfmac2(N, data->dom->dims, data->dom->strs, dst, data->stat_dom->strs, stat_tmp, nstat_strs, data->ones);

	md_free(tmp);
	md_free(stat_tmp);

	md_free(data->ones);
	data->ones = NULL;

	PRINT_TIMER("der/adj bn in ");
}


static void bn_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(bn_s, _data);
	md_free(data->out);
	md_free(data->scale);
	md_free(data->ones);

	iovec_free(data->dom);
	iovec_free(data->stat_dom);

	xfree(data);
}

/**
 * Nlop to compute mean and variance of input
 *
 * @param N number of dimension
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
 * @param epsilon small number to stabilise division
 *
 * In 0:	Input
 * Out 0:	Normalized out
 * Out 1:	Mean \mu = \sum_{i=1}^N x_i/N
 * Out 2: 	Variance \var = \sum_{i=1}^N |(x_i-\mu)|^2/N
 *
 * Note the difference of the definition compared to md_zvar which has factor 1/(N-1)
 **/
static const struct nlop_s* nlop_bn_create(int N, const long dims[N], unsigned long flags, float epsilon)
{
	PTR_ALLOC(struct bn_s, data);
	SET_TYPEID(bn_s, data);

	// will be initialized later, to transparently support GPU
	data->flags = flags;
	data->dom = iovec_create(N, dims, CFL_SIZE);
	long stat_dims[N];
	md_select_dims(N, ~flags, stat_dims, dims);
	data->stat_dom = iovec_create(N, stat_dims, CFL_SIZE);

	data->mean_size = md_calc_size(N, dims) / md_calc_size(N, stat_dims);
	data->epsilon = epsilon;

	data->out = NULL;
	data->scale = NULL;
	data->ones = NULL;

	long nl_odims[3][N];
	md_copy_dims(N, nl_odims[0], dims);
	md_copy_dims(N, nl_odims[1], stat_dims);
	md_copy_dims(N, nl_odims[2], stat_dims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);

	operator_property_flags_t props[1][3] = { { 0, 0, 0 } };


	return nlop_generic_with_props_create(3, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)), bn_fun,
						(nlop_der_fun_t[1][3]){ { bn_deradj_in, bn_der_mean, bn_der_var } },
						(nlop_der_fun_t[1][3]){ { bn_deradj_in, bn_adj_mean, bn_adj_var } },
						 NULL, NULL, bn_del, props);
}


/**
 * Nlop to batch normalize input
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that are not present in Mean/Var
 * @param epsilon small factor for numerical stability
 *
 * In 0:	Input			dims: {n1, n2, ..., nN}
 * In 1:	Floating Mean/Var	dims: {n1, 1,  ..., nN | 2 (mean/var)}
 *
 * Out 0:	Normalized Input	dims: {n1, n2, ..., nN}
 * Out 1:	Mean/Var		dims: {n1, 1,  ..., nN | 2 (mean/var)}
 **/
const struct nlop_s* nlop_batchnorm_create(int N, const long dims[N], unsigned long flags, float epsilon, enum NETWORK_STATUS status)
{
	long stat_dims[N];
	md_select_dims(N, ~flags, stat_dims, dims);

	const struct nlop_s* result = NULL;
	const struct iovec_s* iov = NULL;

	switch (status) {

		case STAT_TRAIN:

			result = nlop_bn_create(N, dims, flags, epsilon);
			result = nlop_append_singleton_dim_out_F(result, 1);
			result = nlop_append_singleton_dim_out_F(result, 2);
			result = nlop_stack_outputs_F(result, 1, 2, N);
			iov = nlop_generic_codomain(result, 1);
			result = nlop_combine_FF(result, nlop_del_out_create(iov->N, iov->dims));
			return result;

		case STAT_TEST:

			result = nlop_normalize_create(N, dims, flags, epsilon);
			result = nlop_append_singleton_dim_in_F(result, 1);
			result = nlop_append_singleton_dim_in_F(result, 2);
			result = nlop_stack_inputs_F(result, 1, 2, N);
			iov = nlop_generic_domain(result, 1);
			result = nlop_combine_FF(result, nlop_from_linop_F(linop_identity_create(iov->N, iov->dims)));
			result = nlop_dup_F(result, 1, 2);
			return result;
	}

	assert(0);
	return NULL;
}
