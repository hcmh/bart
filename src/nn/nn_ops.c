/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

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
#include "nn/layers.h"

#include "nn_ops.h"

struct maxpool_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const long* pool_dims;
	const long* pool_strs;

	complex float* pool;
};

DEF_TYPEID(maxpool_s);

static void maxpool_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(maxpool_s, _data);
	unsigned long N = data->N;

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif
	if (NULL == data->pool)
		data->pool = md_alloc_sameplace(2 * data->N, data->pool_dims, CFL_SIZE, dst);

	complex float* tmp = md_alloc_sameplace(2 * N,  data->pool_dims, CFL_SIZE, dst);

	md_copy2(2 * N, data->pool_dims, MD_STRIDES(2 * N, data->pool_dims, CFL_SIZE), tmp, data->pool_strs, src, CFL_SIZE);

	long tdims[2] = {md_calc_size(N, data->pool_dims), md_calc_size(N, data->pool_dims + N)};
	long tstrs[2] = {CFL_SIZE, tdims[0] * CFL_SIZE};
	long tstrs0[2] = {CFL_SIZE, 0};

	md_copy(1, tdims, dst, tmp, CFL_SIZE);

	for (long i = 1; i < tdims[1]; i++)
		md_zmax(1, tdims, dst, dst, tmp + tdims[0]);

	md_zgreatequal2(2, tdims, tstrs, data->pool, tstrs, tmp, tstrs0, dst);

	md_free(tmp);
	PRINT_TIMER("mpools");
}

static void maxpool_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(maxpool_s, _data);

	long N = data->N;
	long tdims[2] = {md_calc_size(N, data->pool_dims), md_calc_size(N, data->pool_dims + N)};
	long tstrs[2] = {CFL_SIZE, tdims[0] * CFL_SIZE};
	long tstrs0[2] = {CFL_SIZE, 0};

	complex float* tmp = md_alloc_sameplace(2 * N,  data->pool_dims, CFL_SIZE, dst);
	md_copy2(2 * N, data->pool_dims, MD_STRIDES(2 * N, data->pool_dims, CFL_SIZE), dst, data->pool_strs, src, CFL_SIZE);

	md_ztenmul2(2, tdims, tstrs0, dst, tstrs, tmp, tstrs, data->pool);

	md_free(tmp);
}

static void maxpool_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(maxpool_s, _data);

	long N = data->N;
	long tdims[2] = {md_calc_size(N, data->pool_dims), md_calc_size(N, data->pool_dims + N)};
	long tstrs[2] = {CFL_SIZE, tdims[0] * CFL_SIZE};
	long tstrs0[2] = {CFL_SIZE, 0};

	complex float* tmp = md_alloc_sameplace(2 * N,  data->pool_dims, CFL_SIZE, dst);
	md_ztenmul2(2, tdims, tstrs ,tmp, tstrs0, src, tstrs, data->pool);

	md_copy2(2 * N, data->pool_dims, data->pool_strs, dst, MD_STRIDES(2 * N, data->pool_dims, CFL_SIZE), tmp, CFL_SIZE);

	md_free(tmp);
	PRINT_TIMER("mpool adjs");
}


static void maxpool_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(maxpool_s, _data);

	md_free(data->pool);

	xfree(data->pool_dims);
	xfree(data->pool_strs);

	xfree(data);
}


const struct nlop_s* nlop_maxpool_create(int N, const long dims[N], const long pool_size[N])
{
	PTR_ALLOC(struct maxpool_s, data);
	SET_TYPEID(maxpool_s, data);

	for (int i = 0; i< N; i++)
		assert(dims[i] % pool_size[i] == 0);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long pool_dims_tmp[2 * N];
	long pool_strs_tmp[2 * N];
	long compare_strs_tmp[2 * N];

    	for (int i = 0; i< N; i++){

		pool_dims_tmp[i] = dims[i] / pool_size[i];
		pool_dims_tmp[i + N] = pool_size[i];
		pool_strs_tmp[i] = strs[i] * pool_size[i];
		pool_strs_tmp[i + N] = (pool_dims_tmp[i + N] > 1) * strs[i];
	}

	md_singleton_strides(2 * N, compare_strs_tmp);
	md_calc_strides(N, compare_strs_tmp, pool_dims_tmp, CFL_SIZE);

	PTR_ALLOC(long[2 * N], pool_dims);
	md_copy_dims(2 * N, *pool_dims, pool_dims_tmp);
	PTR_ALLOC(long[2 * N], pool_strs);
	md_copy_dims(2 * N, *pool_strs, pool_strs_tmp);

	data->N = N;
	data->pool_dims = *PTR_PASS(pool_dims);
	data->pool_strs = *PTR_PASS(pool_strs);

	// will be initialized later, to transparently support GPU
	data->pool = NULL;

	return nlop_create(N, pool_dims_tmp, N, dims, CAST_UP(PTR_PASS(data)), maxpool_fun, maxpool_der, maxpool_adj, NULL, NULL, maxpool_del);
}


struct dropout_s {

	INTERFACE(nlop_data_t);

	int N;
	float p;

	const struct iovec_s* tmpdom;
	const struct iovec_s* dom;
	const struct iovec_s* codom;

	complex float* tmp;
};

DEF_TYPEID(dropout_s);


static void dropout_fun(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(dropout_s, _data);

	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->N, data->tmpdom->dims, CFL_SIZE, dst);
#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif
	if (NULL == data->tmp)
	data->tmp = md_alloc_sameplace(data->N, data->tmpdom->dims, CFL_SIZE, dst);

	if (network_status == STAT_TEST){

		md_zsmul2(data->N, data->codom->dims, data->codom->strs, dst, data->dom->strs, src, (complex float)data->p);
		PRINT_TIMER("douts");
		return;
	}

	if (network_status == STAT_TRAIN){

		md_rand_one(data->N, data->tmpdom->dims, data->tmp, (1. - data->p));

		md_ztenmul2(data->N, data->codom->dims, data->codom->strs, dst, data->tmpdom->strs, data->tmp, data->dom->strs, src);
		PRINT_TIMER("douts");
		return;
	}

    	assert(0);
}

static void dropout_der(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	if (network_status == STAT_TEST){

		md_zsmul2(data->N, data->codom->dims, data->codom->strs, dst, data->dom->strs, src, (complex float)data->p);
		return;
	}

	if (network_status == STAT_TRAIN){

		md_ztenmul2(data->N, data->codom->dims, data->codom->strs, dst, data->tmpdom->strs, data->tmp, data->dom->strs, src);
		return;
	}
}

static void dropout_adj(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	if (network_status == STAT_TEST){

		md_zsmul2(data->N, data->dom->dims, data->dom->strs, dst, data->codom->strs, src, (complex float)data->p);
		PRINT_TIMER("dout adjs");
		return;
	}

	if (network_status == STAT_TRAIN){

		md_ztenmul2(data->N, data->dom->dims, data->dom->strs, dst, data->tmpdom->strs, data->tmp, data->codom->strs, src);
		PRINT_TIMER("dout adjs");
		return;
	}
}


static void dropout_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(dropout_s, _data);

	md_free(data->tmp);

	iovec_free(data->dom);
	iovec_free(data->codom);
	iovec_free(data->tmpdom);

	xfree(data);
}


const struct nlop_s* nlop_dropout_create(int N, const long dims[N], float p, unsigned int shared_dims_flag)
{
	PTR_ALLOC(struct dropout_s, data);
	SET_TYPEID(dropout_s, data);

	data->N = N;
	data->p = p;

	// will be initialized later, to transparently support GPU
	data->tmp = NULL;

	long tmpdims[N];
	md_select_dims(N, ~shared_dims_flag, tmpdims, dims);

	data->tmpdom = iovec_create(N, tmpdims, CFL_SIZE);
	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->codom = iovec_create(N, dims, CFL_SIZE);

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), dropout_fun, dropout_der, dropout_adj, NULL, NULL, dropout_del);
}