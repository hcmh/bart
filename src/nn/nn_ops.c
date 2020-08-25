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

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/stack.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "linops/someops.h"
#include "nlops/cast.h"
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

	md_copy(1, tdims, dst, tmp, CFL_SIZE); //dst[i] = tmp[i], saved in tdims[1]

	for (long i = 1; i < tdims[1]; i++)
		md_zmax(1, tdims, dst, dst, tmp + tdims[0] * i); //writes max of dst and tmp + compared value to destination

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
	md_copy2(2 * N, data->pool_dims, MD_STRIDES(2 * N, data->pool_dims, CFL_SIZE), tmp, data->pool_strs, src, CFL_SIZE);

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

/**
 * pool_size //1, px, py, pz, 1	or px, py, pz, 1, 1 depending on channel_first
*/
const struct nlop_s* nlop_maxpool_create(int N, const long dims[N], const long pool_size[N])
{
	PTR_ALLOC(struct maxpool_s, data);
	SET_TYPEID(maxpool_s, data);

	for (int i = 0; i< N; i++)
		assert(dims[i] % pool_size[i] == 0);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long pool_dims_tmp[2 * N];	//img_out_dims, pool_dims
	long pool_strs_tmp[2 * N];	//img_out_str, pool_str
	long compare_strs_tmp[2 * N];

    	for (int i = 0; i< N; i++){

		pool_dims_tmp[i] = dims[i] / pool_size[i];
		pool_dims_tmp[i + N] = pool_size[i];
		pool_strs_tmp[i] = strs[i] * pool_size[i];
		pool_strs_tmp[i + N] = (pool_dims_tmp[i + N] > 1) ? strs[i] : 0;
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

	md_rand_one(data->N, data->tmpdom->dims, data->tmp, (1. - data->p));

	md_ztenmul2(data->N, data->codom->dims, data->codom->strs, dst, data->tmpdom->strs, data->tmp, data->dom->strs, src);
	PRINT_TIMER("douts");
}

static void dropout_der(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	md_ztenmul2(data->N, data->codom->dims, data->codom->strs, dst, data->tmpdom->strs, data->tmp, data->dom->strs, src);
}

static void dropout_adj(const struct nlop_data_s* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	md_ztenmul2(data->N, data->dom->dims, data->dom->strs, dst, data->tmpdom->strs, data->tmp, data->codom->strs, src);
	PRINT_TIMER("dout adjs");
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

struct avgpool_s {

	INTERFACE(linop_data_t);

	unsigned long N;
	const long* pool_dims;
	const long* pool_strs;

};

DEF_TYPEID(avgpool_s);

static void avgpool_fun(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(avgpool_s, _data);
	
	unsigned long N = data->N;

	complex float* tmp = md_alloc_sameplace(2 * N,  data->pool_dims, CFL_SIZE, dst);
	md_copy2(2 * N, data->pool_dims, MD_STRIDES(2 * N, data->pool_dims, CFL_SIZE), tmp, data->pool_strs, src, CFL_SIZE);

	long tdims[2] = {md_calc_size(N, data->pool_dims), md_calc_size(N, data->pool_dims + N)}; 
	md_copy(1, tdims, dst, tmp, CFL_SIZE); //dst[i] = tmp[i], saved in tdims[1]

	for (long i = 1; i < tdims[1]; i++)
		md_zadd(1, tdims, dst, dst, tmp + tdims[0] * i); //adds all values along dimension
		
	md_zsmul(1, tdims, dst, dst, 1. / tdims[1]); //divide by pool dim to gain average

	md_free(tmp);
	PRINT_TIMER("mpools");
}

/**
 * Calculate adjoint to average pooling layer
 * adjoint has entries in form of Average/pooling size
 */
static void avgpool_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(avgpool_s, _data);
	long N = data->N;

	complex float* tmp_adj= md_alloc_sameplace(2 * data->N, data->pool_dims, CFL_SIZE, dst);
	complex float* ones = md_alloc_sameplace(2 * data->N, data->pool_dims, CFL_SIZE, dst);

	long tdims[2] = { md_calc_size(N, data->pool_dims), md_calc_size(N, data->pool_dims + N)}; // number of outputs, pooling size
	long tstrs[2] = {CFL_SIZE, tdims[0] * CFL_SIZE};
	long tstrs0[2] = {CFL_SIZE, 0};
	
	long tdims_adj[2] = {md_calc_size(N, data->pool_dims), md_calc_size(N, data->pool_dims + N)}; // number of inputs, pooling size 
	md_zsmul(1, tdims_adj, tmp_adj, src, 1. / tdims[1]); // divide average A by pooling size p to gain A/p as entries for adjoint
	md_zfill(2, tdims, ones, 1);

	complex float* tmp = md_alloc_sameplace(2 * N,  data->pool_dims, CFL_SIZE, dst);
	md_ztenmul2(2, tdims, tstrs ,tmp, tstrs0, tmp_adj, tstrs, ones);
	md_copy2(2 * N, data->pool_dims, data->pool_strs, dst, MD_STRIDES(2 * N, data->pool_dims, CFL_SIZE), tmp, CFL_SIZE);

	md_free(ones);
	md_free(tmp_adj);
	md_free(tmp);
	PRINT_TIMER("mpool adjs");
}

static void avgpool_del(const struct linop_data_s* _data)
{
	const auto data = CAST_DOWN(avgpool_s, _data);

	xfree(data->pool_dims);
	xfree(data->pool_strs);

	xfree(data);
}

const struct linop_s* linop_avgpool_create(int N, const long dims[N], const long pool_size[N])
{
	PTR_ALLOC(struct avgpool_s, data);
	SET_TYPEID(avgpool_s, data);

	for (int i = 0; i< N; i++)
		assert(dims[i] % pool_size[i] == 0);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long pool_dims_tmp[2 * N];	//img_out_dims, pool_dims
	long pool_strs_tmp[2 * N];	//img_out_str, pool_str
	long compare_strs_tmp[2 * N];
	
    	for (int i = 0; i< N; i++){

		pool_dims_tmp[i] = dims[i] / pool_size[i];
		pool_dims_tmp[i + N] = pool_size[i];
		pool_strs_tmp[i] = strs[i] * pool_size[i];
		pool_strs_tmp[i + N] = (pool_dims_tmp[i + N] > 1) ? strs[i] : 0;
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

	return linop_create(N, pool_dims_tmp, N, dims, CAST_UP(PTR_PASS(data)), avgpool_fun, avgpool_adj, NULL, NULL, avgpool_del);
}


struct zmax_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	unsigned long flags;
	const long* outdims;
	const long* dims;
	const long* strides;
	const long* outstrides;

	complex float* pool;
};

DEF_TYPEID(zmax_s);

static void zmax_fun(const nlop_data_t* _data, complex float* dst, const complex float* src) {
	const auto data = CAST_DOWN(zmax_s, _data);

	if (NULL == data->pool)
		data->pool = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	md_copy2(data->N, data->outdims, data->outstrides, dst, data->strides, src, CFL_SIZE);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	do{
		md_zmax2(data->N, data->outdims, data->outstrides, dst, data->outstrides, dst, data->strides, &MD_ACCESS(data->N, data->strides, pos, src));
	}while(md_next(data->N, data->dims, data->flags, pos));
	md_zgreatequal2(data->N, data->dims, data->strides, data->pool, data->strides, src, data->outstrides, dst);
}

static void zmax_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zmax_s, _data);

	md_ztenmul(data->N, data->outdims, dst, data->dims, src, data->dims, data->pool);
}

static void zmax_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(zmax_s, _data);

	//md_ztenmul(data->N, data->dims, dst, data->outdims, src, data->dims, data->pool); //alternative
	md_zmul2(data->N, data->dims, data->strides, dst, data->outstrides, src, data->strides, data->pool);

	PRINT_TIMER("zmax adjs");
}

static void zmax_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(zmax_s, _data);

	md_free(data->pool);

	xfree(data->outdims);
	xfree(data->dims);
	xfree(data->strides);
	xfree(data->outstrides);

	xfree(data);
}

const struct nlop_s* nlop_zmax_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct zmax_s, data);
	SET_TYPEID(zmax_s, data);

	PTR_ALLOC(long[N], outdims);
	md_select_dims(N, ~flags, *outdims, dims);
	PTR_ALLOC(long[N], dims_tmp);
	md_copy_dims(N, *dims_tmp, dims);

	PTR_ALLOC(long[N], strides);
	md_calc_strides(N, *strides, dims, CFL_SIZE);
	PTR_ALLOC(long[N], out_strides);
	md_calc_strides(N, *out_strides, *outdims, CFL_SIZE);

	data->N = N;
	data->flags = flags;
	data->strides = *PTR_PASS(strides);
	data->dims = *PTR_PASS(dims_tmp);
	data->outdims = *PTR_PASS(outdims);
	data->outstrides = *PTR_PASS(out_strides);

	data->pool = NULL;

	long odims[N];
	md_select_dims(N, ~flags, odims, dims);

	return nlop_create(N, odims, N, dims, CAST_UP(PTR_PASS(data)), zmax_fun, zmax_der, zmax_adj, NULL, NULL, zmax_del);
}

struct pool_s {

	INTERFACE(linop_data_t);

	unsigned long N;
	const long* pool_dims;
	const long* pool_strs;
	const long* dims;
};

DEF_TYPEID(pool_s);

static void pool_fun(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(pool_s, _data);

	md_copy2(data->N, data->pool_dims, MD_STRIDES(data->N, data->pool_dims, CFL_SIZE), dst, data->pool_strs, src, CFL_SIZE);

	PRINT_TIMER("pools");
}

static void pool_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(pool_s, _data);

	md_clear(data->N, data->dims, dst, CFL_SIZE);
	md_copy2(data->N, data->pool_dims, data->pool_strs, dst, MD_STRIDES(data->N, data->pool_dims, CFL_SIZE), src, CFL_SIZE);

	PRINT_TIMER("pool adjs");
}

static void pool_del(const struct linop_data_s* _data)
{
	const auto data = CAST_DOWN(pool_s, _data);

	xfree(data->pool_dims);
	xfree(data->pool_strs);
	xfree(data->dims);
	xfree(data);
}

const struct linop_s* linop_pool_create(int N, const long dims[N], const long pool_size[N])
{
	PTR_ALLOC(struct pool_s, data);
	SET_TYPEID(pool_s, data);

	for (int i = 0; i< N; i++)
		assert(dims[i] % pool_size[i] == 0);

	long strs_tmp[N];
	md_calc_strides(N, strs_tmp, dims, CFL_SIZE);

	long pool_dims_tmp[N];	//img_out_dims
	long pool_strs_tmp[N];	//img_out_str

	for (int i = 0; i< N; i++){

		pool_dims_tmp[i] = dims[i] / pool_size[i];
		pool_strs_tmp[i] = strs_tmp[i] * pool_size[i];
	}

	PTR_ALLOC(long[N], pool_dims);
	md_copy_dims(N, *pool_dims, pool_dims_tmp);
	PTR_ALLOC(long[N], pool_strs);
	md_copy_dims(N, *pool_strs, pool_strs_tmp);
	PTR_ALLOC(long[N], dims_tmp);
	md_copy_dims(N, *dims_tmp, dims);

	data->N = N;

	data->pool_dims = *PTR_PASS(pool_dims);
	data->pool_strs = *PTR_PASS(pool_strs);
	data->dims = *PTR_PASS(dims_tmp);

	return linop_create(N, pool_dims_tmp, N, dims, CAST_UP(PTR_PASS(data)), pool_fun, pool_adj, NULL, NULL, pool_del);
}

/**
 * Adapted from:
 * "Making Convolutional Networks Shift-Invariant Again"
 * Richard Zhang
 * arXiv:1904.11486v2
 */
const struct nlop_s* nlop_blurpool_create(int N, const long dims[N], const long pool_size[N])
{
	long pad_before[N];
	for (int i = 0; i< N; i++)
		pad_before[i] = pool_size[i] - 1;

	long pad_after[N];
	md_set_dims(N, pad_after, 0);

	const struct nlop_s* pad_op = nlop_from_linop_F(linop_padding_create(N, dims, PAD_CYCLIC, pad_before, pad_after));// in: indims; out: pad_out

	long padded_dims[N];
	for (int i = 0; i< N; i++)
		padded_dims[i] = dims[i] + pad_before[i] + pad_after[i];
	
	long pos[N];
	md_set_dims(N, pos, 0);
	long sdims[N+1];
	md_set_dims(N+1, sdims, 1);
	md_copy_dims(N, sdims, dims);

	const struct nlop_s* extract = NULL;
	do{
		const struct nlop_s* tmp_op = nlop_from_linop_F(linop_extract_create(N, pos, dims, padded_dims)); // in: pad_out; out: extract
		tmp_op = nlop_reshape_out_F(tmp_op, 0, N+1, sdims);
		if (NULL == extract){

			extract = tmp_op;
		} else {

			extract = nlop_combine_FF(extract, tmp_op);
			extract = nlop_dup_F(extract, 0, 1); // in: ipad_out; out: extr1, extr2, ...
			extract = nlop_stack_outputs(extract, 0, 1, N); //in: pad_out; out: stack_out
		}
	}while(md_next(N, pool_size, ~0, pos));
	extract = nlop_chain2_swap_FF(pad_op, 0, extract, 0); // in: indims; out: stack_out

	long extract_dims[N+1];
	md_copy_dims(N+1, extract_dims, nlop_generic_codomain(extract, 0)->dims);

	const struct nlop_s* zmax_op = nlop_zmax_create(N+1, extract_dims, MD_BIT(N)); // in: stack_out; out: zmax_out[c,x,y,z,b,1]
	zmax_op = nlop_reshape_out_F(zmax_op, 0, N, dims); //in: stack_out; out: zmax_out[c,x,y,z,b]

	const struct nlop_s* blurpool_op = nlop_chain2_swap_FF(extract, 0, zmax_op, 0); // in: indims; out: zmax_out

	unsigned long fft_flag = MD_BIT(N) - 1ul;
	long resize_dims[N];	
	for (int i = 0; i< N; i++)
		resize_dims[i] = dims[i] / pool_size[i];

	const struct linop_s* blur = linop_fftc_create(N, dims, fft_flag); // in: zmax_out; out: fftc_out
	blur = linop_chain_FF(blur, linop_resize_center_create(N, resize_dims, dims)); // in: fftc_out; out: resized_fft
	blur = linop_chain_FF(blur, linop_resize_center_create(N, dims, resize_dims)); // in: resized_fft; out: expanded_fft
	blur = linop_chain_FF(blur, linop_ifftc_create(N, dims, fft_flag)); // in: resized; out: ifftc_out
	blurpool_op = nlop_chain2_swap_FF(blurpool_op, 0, nlop_from_linop_F(blur), 0); //in: indims; out: blur

	blurpool_op = nlop_chain2_swap_FF(blurpool_op, 0, nlop_from_linop_F(linop_zreal_create(N, dims)), 0); // set imaginary output to zero
	blurpool_op = nlop_chain2_swap_FF(blurpool_op, 0, nlop_from_linop_F(linop_pool_create(N, dims, pool_size)), 0);

	return blurpool_op;
}
