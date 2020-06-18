
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

#include "nlops/nlop.h"
#include "nlops/chain.h"
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
		data->x = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, args);
#ifdef USE_CUDA
	assert((cuda_ondevice(mean) == cuda_ondevice(src)) && (cuda_ondevice(var) == cuda_ondevice(src)));
#endif

	md_zsum(data->dom->N, data->dom->dims, data->flags, mean, src);
	md_zsmul(data->dom->N, data->codom->dims, mean, mean, 1. / data->n);

	data->x = md_alloc_sameplace(data->dom->N, data->dom->dims, data->dom->size, src);
	md_zsub2(data->dom->N, data->dom->dims, data->dom->strs, data->x, data->dom->strs, src, data->codom->strs, mean);

	md_ztenmulc(data->dom->N, data->codom->dims, var, data->dom->dims, data->x, data->dom->dims, data->x);
	md_zsmul(data->codom->N, data->codom->dims, var, var, 1. / data->n);
}

static void stats_der_mean(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	md_zsum(data->dom->N, data->dom->dims, data->flags, dst, src);
	md_zsmul(data->dom->N, data->codom->dims, dst, dst, 1. / data->n);
}

static void stats_adj_mean(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* tmp = md_alloc_sameplace(data->codom->N, data->codom->dims, data->codom->size, src);

	md_zsmul(data->codom->N, data->codom->dims, tmp, src, 1. / data->n);
	md_copy2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->codom->strs, tmp, data->dom->size);
	md_free(tmp);
}

static void stats_der_var(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	md_ztenmulc(data->dom->N, data->codom->dims, dst, data->dom->dims, src, data->dom->dims, data->x);
	md_zsmul(data->codom->N, data->codom->dims, dst, dst, (2. / data->n));
	md_zreal(data->codom->N, data->codom->dims, dst, dst);
}

static void stats_adj_var(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stats_s, _data);

	complex float* tmp = md_alloc_sameplace(data->codom->N, data->codom->dims, data->codom->size, src);
	md_zreal(data->codom->N, data->codom->dims, tmp, src);
	md_ztenmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->codom->strs, tmp, data->dom->strs, data->x);
	md_zsmul(data->dom->N, data->dom->dims, dst, dst, (2. / data->n));
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
		stats_fun, (nlop_fun_t[1][2]){ { stats_der_mean, stats_der_var } }, (nlop_fun_t[1][2]){ { stats_adj_mean, stats_adj_var } }, NULL, NULL, stats_del);
}

struct normalize_s {

	INTERFACE(nlop_data_t);

	const struct iovec_s* dom;
	const struct iovec_s* statdom;

	complex float* tmp; // (src - mu)
	complex float* scale; // sqrt(var + epsilon)

	float epsilon;
};

DEF_TYPEID(normalize_s);


static void normalize_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
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
	md_zsqrt(data->statdom->N, data->statdom->dims, data->scale, data->scale);

	md_zsub2(data->dom->N, data->dom->dims, data->dom->strs, data->tmp, data->dom->strs, src, data->statdom->strs, mean);
	md_zdiv2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, data->tmp, data->statdom->strs, data->scale);
}

static void normalize_deradj_src(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);
	md_zdiv2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, data->scale);
}

static void normalize_der_mean(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);
	md_zdiv2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, src, data->statdom->strs, data->scale);
	md_zsmul(data->dom->N, data->dom->dims, dst, dst, -1.);
}

static void normalize_adj_mean(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);
	md_clear(data->statdom->N, data->statdom->dims, dst, data->statdom->size);
	md_zadd2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->statdom->strs, dst, data->dom->strs, src);
	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, dst, dst, -1.);
}

static void normalize_der_var(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	complex float* tmp = md_alloc_sameplace(data->statdom->N, data->statdom->dims, data->statdom->size, dst);

	md_zreal(data->statdom->N, data->statdom->dims, tmp, src);

	md_zdiv(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, tmp, tmp, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, tmp, tmp, -.5);

	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, tmp, data->dom->strs, data->tmp);



	md_free(tmp);
}

static void normalize_adj_var(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(normalize_s, _data);

	complex float* tmp = md_alloc_sameplace(data->statdom->N, data->statdom->dims, data->statdom->size, dst);

	md_clear(data->statdom->N, data->statdom->dims, dst, data->statdom->size);

	md_zfmacc2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->dom->strs, src, data->dom->strs, data->tmp);

	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zdiv(data->statdom->N, data->statdom->dims, dst, dst, data->scale);
	md_zsmul(data->statdom->N, data->statdom->dims, dst, dst, -.5);
	md_zreal(data->statdom->N, data->statdom->dims, dst, dst);

	md_free(tmp);
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
 * Nlop to compute mean and variance of input
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
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

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], statdims);
	md_copy_dims(N, nl_idims[2], statdims);


	return nlop_generic_create(1, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)), normalize_fun,
					(nlop_fun_t[3][1]){ { normalize_deradj_src}, { normalize_der_mean }, { normalize_der_var } },
					(nlop_fun_t[3][1]){ { normalize_deradj_src}, { normalize_adj_mean }, { normalize_adj_var } },
					NULL, NULL, normalize_del);
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

static void rescale_der_src(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(rescale_s, _data);
	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, data->scale);
}

static void rescale_adj_src(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(rescale_s, _data);
	md_zmulc2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->statdom->strs, data->scale);
}

static void rescale_der_gamma(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(rescale_s, _data);
	md_zmul2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, data->in, data->statdom->strs, src);
}

static void rescale_adj_gamma(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(rescale_s, _data);
	md_ztenmulc2(data->dom->N, data->dom->dims, data->statdom->strs, dst, data->dom->strs, src, data->dom->strs, data->in);
}

static void rescale_der_beta(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(rescale_s, _data);
	md_copy2(data->dom->N, data->dom->dims, data->dom->strs, dst, data->statdom->strs, src, data->dom->size);
}

static void rescale_adj_beta(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
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
 * Nlop to compute mean and variance of input
 *
 * @param dims dims of input tensor
 * @param flags dims to compute mean/var over, i.e. dimensions that do not stay
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
					(nlop_fun_t[3][1]){ { rescale_der_src}, { rescale_der_beta }, { rescale_der_gamma } },
					(nlop_fun_t[3][1]){ { rescale_adj_src}, { rescale_adj_beta }, { rescale_adj_gamma } },
					NULL, NULL, rescale_del);
}
struct batchnorm_stats_s {

	INTERFACE(nlop_data_t);

	const struct iovec_s* dom;
	const struct iovec_s* codom;

	const struct nlop_s* stats_op;
	float n;
};

DEF_TYPEID(batchnorm_stats_s);


static void batchnorm_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(batchnorm_stats_s, _data);

	assert(6 == N);

	complex float* mean = args[0];
	complex float* var = args[1];
	complex float* unbiased_var_out = args[2];

	complex float* src = args[3];
	complex float* floating_mean_in = args[4];
	complex float* floating_var_in = args[5];

	if (network_status == STAT_TEST){

		md_copy(data->codom->N, data->codom->dims, mean, floating_mean_in, data->codom->size);
		md_copy(data->codom->N, data->codom->dims, var, floating_var_in, data->codom->size);

		return;
	}

	if (network_status == STAT_TRAIN){

		nlop_generic_apply_unchecked(data->stats_op, 3, MAKE_ARRAY((void*)mean, (void*)var, (void*)src));
		md_zsmul(data->codom->N, data->codom->dims, unbiased_var_out, var, data->n / (data->n - 1.) );

		return;
	}

	assert(0);
}

static void batchnorm_der_mean_src(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(batchnorm_stats_s, _data);

	if (network_status == STAT_TEST){

		md_clear(data->codom->N, data->codom->dims, dst, data->codom->size);
		return;
	}

	if (network_status == STAT_TRAIN){

		linop_forward(	nlop_get_derivative(data->stats_op, 0, 0),
				data->codom->N, data->codom->dims, dst,
				data->dom->N, data->dom->dims, src);

		return;
	}

	assert(0);
}

static void batchnorm_adj_mean_src(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(batchnorm_stats_s, _data);

	if (network_status == STAT_TEST){

		md_clear(data->dom->N, data->dom->dims, dst, data->dom->size);
		return;
	}

	if (network_status == STAT_TRAIN){

		linop_adjoint(	nlop_get_derivative(data->stats_op, 0, 0),
				data->dom->N, data->dom->dims, dst,
				data->codom->N, data->codom->dims, src);

		return;
	}

	assert(0);
}

static void batchnorm_der_var_src(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(batchnorm_stats_s, _data);

	if (network_status == STAT_TEST){

		md_clear(data->codom->N, data->codom->dims, dst, data->codom->size);
		return;
	}

	if (network_status == STAT_TRAIN){

		linop_forward(	nlop_get_derivative(data->stats_op, 1, 0),
				data->codom->N, data->codom->dims, dst,
				data->dom->N, data->dom->dims, src);

		return;
	}

	assert(0);
}

static void batchnorm_adj_var_src(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(batchnorm_stats_s, _data);

	if (network_status == STAT_TEST){

		md_clear(data->dom->N, data->dom->dims, dst, data->dom->size);
		return;
	}

	if (network_status == STAT_TRAIN){

		linop_adjoint(	nlop_get_derivative(data->stats_op, 1, 0),
				data->dom->N, data->dom->dims, dst,
				data->codom->N, data->codom->dims, src);

		return;
	}

	assert(0);
}

static void batchnorm_not_implemented(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	error("Derivative of batch normalization is not implemented!\n");
}


static void batchnorm_stats_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(batchnorm_stats_s, _data);

	nlop_free(data->stats_op);

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
 * In 1:	Floating Mean
 * In 2: 	Floating Variance

 * Out 0:	Mean (Train: \mu = \sum_{i=1}^N x_i/N, Inference: Floating Mean)
 * Out 1: 	Variance (Train: variance of minibatch \var = \sum_{i=1}^N |(x_i-\mu)|^2/N, Inference: Floating Variance)
 * Out 3: 	Unbiased sample variance for upadting floating variance (Train only)
 *
 **/
const struct nlop_s* nlop_batchnorm_floatingstats_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct batchnorm_stats_s, data);
	SET_TYPEID(batchnorm_stats_s, data);


	long codims[N];
	md_select_dims(N, ~flags, codims, dims);

	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->codom = iovec_create(N, codims, CFL_SIZE);
	data->n = md_calc_size(N, dims) / md_calc_size(N, codims);
	data->stats_op = nlop_stats_create(N, dims, flags);

	long nl_odims[4][N];
	md_copy_dims(N, nl_odims[0], codims);
	md_copy_dims(N, nl_odims[1], codims);
	md_copy_dims(N, nl_odims[2], codims);
	md_copy_dims(N, nl_odims[3], codims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], codims);
	md_copy_dims(N, nl_idims[2], codims);


	return nlop_generic_create(4, N, nl_odims, 3, N, nl_idims, CAST_UP(PTR_PASS(data)), batchnorm_fun,
					(nlop_fun_t[3][3]){	{ stats_der_mean, stats_der_var, batchnorm_not_implemented},
								{ batchnorm_not_implemented, batchnorm_not_implemented, batchnorm_not_implemented},
								{ batchnorm_not_implemented, batchnorm_not_implemented, batchnorm_not_implemented} },
					(nlop_fun_t[3][3]){ 	{ stats_adj_mean, stats_adj_var, batchnorm_not_implemented},
								{ batchnorm_not_implemented, batchnorm_not_implemented, batchnorm_not_implemented},
								{ batchnorm_not_implemented, batchnorm_not_implemented, batchnorm_not_implemented} },
								 NULL, NULL, batchnorm_stats_del);
}
