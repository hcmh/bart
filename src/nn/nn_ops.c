/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>

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
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/stack.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/someops.h"
#include "linops/someops.h"
#include "nlops/cast.h"
#include "nn/layers.h"

#include "nn_ops.h"



const struct nlop_s* nlop_maxpool_create(int N, const long dims[N], const long pool_size[N])
{
	long ndims[2 * N];
	long odims[2 * N];

	unsigned int perm[2 * N];



	for (int i = 0; i < N; i++) {

		assert(0 == dims[i] % pool_size[i]);

		odims[i] = dims[i] / pool_size[i];
		odims[i + N] = pool_size[i];

		ndims[2 * i] = pool_size[i];
		ndims[2 * i + 1] = odims[i];

		perm[i] = 2 * i + 1;
		perm[i + N] = 2 * i;
	}

	auto result = nlop_zmax_create(2 * N, odims, (MD_BIT(2 * N) - 1) & ~(MD_BIT(N) - 1));
	result = nlop_chain_FF(nlop_from_linop_F(linop_permute_create(2 * N, perm, ndims)), result);
	result = nlop_reshape_in_F(result, 0, N, dims);
	result = nlop_reshape_out_F(result, 0, N, odims);

	return result;
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


static void dropout_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
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
}

static void dropout_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	md_ztenmul2(data->N, data->codom->dims, data->codom->strs, dst, data->tmpdom->strs, data->tmp, data->dom->strs, src);
}

static void dropout_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	const auto data = CAST_DOWN(dropout_s, _data);
	assert(NULL != data->tmp);

	md_ztenmul2(data->N, data->dom->dims, data->dom->strs, dst, data->tmpdom->strs, data->tmp, data->codom->strs, src);
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


const struct linop_s* linop_avgpool_create(int N, const long dims[N], const long pool_size[N])
{
	long ndims[2 * N];
	long odims[2 * N];

	unsigned int perm[2 * N];

	for (int i = 0; i < N; i++) {

		assert(0 == dims[i] % pool_size[i]);

		odims[i] = dims[i] / pool_size[i];
		odims[i + N] = pool_size[i];

		ndims[2 * i] = pool_size[i];
		ndims[2 * i + 1] = odims[i];

		perm[i] = 2 * i + 1;
		perm[i + N] = 2 * i;
	}

	auto result = linop_avg_create(2 * N, odims, (MD_BIT(2 * N) - 1) & ~(MD_BIT(N) - 1));
	result = linop_chain_FF(linop_scale_create(2 * N, odims, 1. / sqrtf(md_calc_size(N, pool_size))), result); //linop avg does not average
	result = linop_chain_FF(linop_permute_create(2 * N, perm, ndims), result);
	result = linop_reshape_in_F(result, N, dims);
	result = linop_reshape_out_F(result, N, odims);

	return result;
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
	const auto data = CAST_DOWN(pool_s, _data);

	md_copy2(data->N, data->pool_dims, MD_STRIDES(data->N, data->pool_dims, CFL_SIZE), dst, data->pool_strs, src, CFL_SIZE);

}

static void pool_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(pool_s, _data);

	md_clear(data->N, data->dims, dst, CFL_SIZE);
	md_copy2(data->N, data->pool_dims, data->pool_strs, dst, MD_STRIDES(data->N, data->pool_dims, CFL_SIZE), src, CFL_SIZE);

}

static void pool_del(const struct linop_data_s* _data)
{
	const auto data = CAST_DOWN(pool_s, _data);

	xfree(data->pool_dims);
	xfree(data->pool_strs);
	xfree(data->dims);
	xfree(data);
}

/**
 * Pool incoming dimensions with given pool size.
 **/
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
			extract = nlop_stack_outputs_F(extract, 0, 1, N); //in: pad_out; out: stack_out
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

struct norm_max_abs_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const long* dims;
	const long* sdims;
};

DEF_TYPEID(norm_max_abs_s);

static void norm_max_abs_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	assert(3 == D);
	complex float* dst = args[0];
	complex float* scale = args[1];
	complex float* src = args[2];

	const auto data = CAST_DOWN(norm_max_abs_s, _data);

	unsigned long N = data->N;
	const long* dims = data->dims;
	const long* sdims = data->sdims;

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
#endif

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
	md_zabs(N, dims, tmp, src);

	md_copy2(N, sdims, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, dims, CFL_SIZE), tmp, CFL_SIZE);
	md_zmax2(N, dims, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, sdims, CFL_SIZE), scale, MD_STRIDES(N, dims, CFL_SIZE), tmp);

	complex float* ones = md_alloc_sameplace(N, sdims, CFL_SIZE, dst);
	md_zfill(N, sdims, ones, 1.);
	md_zdiv(N, sdims, tmp, ones, scale);

	md_zmul2(N, dims,
		MD_STRIDES(N, dims, CFL_SIZE), dst,
		MD_STRIDES(N, dims, CFL_SIZE), src,
		MD_STRIDES(N, sdims, CFL_SIZE), tmp);

	md_free(ones);
	md_free(tmp);
}

static void norm_max_abs_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(norm_max_abs_s, _data);

	xfree(data->dims);
	xfree(data->sdims);

	xfree(data);
}

const struct nlop_s* nlop_norm_max_abs_create(int N, const long dims[N], unsigned long batch_flag)
{
	PTR_ALLOC(struct norm_max_abs_s, data);
	SET_TYPEID(norm_max_abs_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	PTR_ALLOC(long[N], sdims);
	md_select_dims(N, batch_flag, *sdims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->sdims = *PTR_PASS(sdims);

	long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], dims);
	md_copy_dims(N, nl_odims[1], data->sdims);

	long nl_idims[1][N];
	md_copy_dims(N, nl_idims[0], dims);

	return nlop_generic_create(2, N, nl_odims, 1, N, nl_idims, CAST_UP(PTR_PASS(data)), norm_max_abs_fun, NULL, NULL, NULL, NULL, norm_max_abs_del);

}