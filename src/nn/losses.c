
/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "losses.h"

struct mse_s {

	INTERFACE(nlop_data_t);

	unsigned long N;

	complex float* tmp;

	const struct iovec_s* maxdom;
	const struct iovec_s* dom_pred;
	const struct iovec_s* dom_true;
};

DEF_TYPEID(mse_s);

static void mse_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	const auto data = CAST_DOWN(mse_s, _data);
	assert(3 == D);

	complex float* dst = args[0];
	const complex float* src_pred = args[1];
	const complex float* src_true = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src_pred)) && (cuda_ondevice(src_pred) == cuda_ondevice(src_true)));
#endif
	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->maxdom->N, data->maxdom->dims, CFL_SIZE, dst);

	md_zsub2(data->N, data->maxdom->dims, data->maxdom->strs, data->tmp, data->dom_pred->strs, src_pred, data->dom_true->strs, src_true);

	dst[0] = (complex float)(md_znorm(data->N, data->maxdom->dims, data->tmp) / (2. * (float)(md_calc_size(data->N, data->maxdom->dims))));
}


static void mse_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	md_ztenmulc2(data->N, data->maxdom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom_pred->strs, src, data->maxdom->strs, data->tmp);
	dst[0] = dst[0] / (complex float)(md_calc_size(data->N, data->maxdom->dims));
}

static void mse_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	md_ztenmulc2(data->N, data->maxdom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom_true->strs, src, data->maxdom->strs, data->tmp);
	dst[0] = -dst[0] / (complex float)(md_calc_size(data->N, data->maxdom->dims));
}

static void mse_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	complex float tmp = src[0] / (complex float)(md_calc_size(data->N, data->maxdom->dims));
	md_ztenmul2(data->N, data->maxdom->dims, data->dom_pred->strs, dst, MD_SINGLETON_STRS(data->N), &tmp, data->maxdom->strs, data->tmp);
}

static void mse_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	complex float tmp = - src[0] / (complex float)(md_calc_size(data->N, data->maxdom->dims));
	md_ztenmul2(data->N, data->maxdom->dims, data->dom_true->strs, dst, MD_SINGLETON_STRS(data->N), &tmp, data->maxdom->strs, data->tmp);
}

static void mse_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(mse_s, _data);

	md_free(data->tmp);
	iovec_free(data->maxdom);
	iovec_free(data->dom_true);
	iovec_free(data->dom_pred);
	xfree(data);
}

const struct nlop_s* nlop_mse_create2(int N, const long dims[N], const long istr1[N], const long istr2[N])
{
	PTR_ALLOC(struct mse_s, data);
	SET_TYPEID(mse_s, data);

	long ndims1[N];
	md_select_dims(N, md_nontriv_strides(N, istr1), ndims1, dims);
	long ndims2[N];
	md_select_dims(N, md_nontriv_strides(N, istr2), ndims2, dims);

	data->N = N;

	data->maxdom = iovec_create(N, dims, CFL_SIZE);
	data->maxdom = iovec_create2(N, ndims1, istr1, CFL_SIZE);
	data->maxdom = iovec_create2(N, ndims2, istr2, CFL_SIZE);

	// will be initialized later, to transparently support GPU
	data->tmp = NULL;

	long nl_odims[1][1];
	md_copy_dims(1, nl_odims[0], MD_SINGLETON_DIMS(1));
	long nl_ostr[1][1];
	md_copy_strides(1, nl_ostr[0], MD_SINGLETON_STRS(1));

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], ndims1);
	md_copy_dims(N, nl_idims[1], ndims2);
	long nl_istr[2][N];
	md_copy_strides(N, nl_istr[0], istr1);
	md_copy_strides(N, nl_istr[1], istr2);

	return nlop_generic_create2(1, 1, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)), mse_fun, (nlop_fun_t[2][1]){ { mse_der1 }, { mse_der2 } }, (nlop_fun_t[2][1]){ { mse_adj1 }, { mse_adj2 } }, NULL, NULL, mse_del);
}


const struct nlop_s* nlop_mse_create(int N, const long idim1[N], const long idim2[N])
{
	long dims[N];
	md_tenmul_dims(N, dims, idim1, idim1, idim2);
	return nlop_mse_create2(N, dims, MD_STRIDES(N, idim1, CFL_SIZE), MD_STRIDES(N, idim2, CFL_SIZE));
}

struct cce_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	complex float* tmp_log;
	complex float* tmp_div;
	const struct iovec_s* dom;
};

DEF_TYPEID(cce_s);

static void cce_initialize(struct cce_s* data, const complex float* arg)
{
	if (NULL == data->tmp_log)
		data->tmp_log = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, arg);
	if (NULL == data->tmp_div)
		data->tmp_div = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, arg);
}

static void cce_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	const auto data = CAST_DOWN(cce_s, _data);
	assert(3 == D);

	complex float* dst = args[0];
	const complex float* src_pred = args[1];
	const complex float* src_true = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src_pred)) && (cuda_ondevice(src_pred) == cuda_ondevice(src_true)));
#endif
	cce_initialize(data, dst);

	md_zlog(data->N, data->dom->dims, data->tmp_log, src_pred);
	md_zdiv_reg(data->N, data->dom->dims, data->tmp_div, src_true, src_pred, 1.e-7);

	md_ztenmul2(data->N, data->dom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom->strs, data->tmp_log, data->dom->strs, src_true);
	dst[0] = (complex float)(-1) * dst[0] / data->dom->dims[data->N-1];
}


static void cce_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	md_ztenmul2(data->N, data->dom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom->strs, src, data->dom->strs, data->tmp_div);
	dst[0] = (complex float)(-1) * dst[0] / data->dom->dims[data->N-1];
}

static void cce_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	md_ztenmul2(data->N, data->dom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom->strs, src, data->dom->strs, data->tmp_log);
	dst[0] = (complex float)(-1) * dst[0] / data->dom->dims[data->N-1];
}

static void cce_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	complex float tmp = src[0] * (complex float)(-1) / data->dom->dims[data->N-1];
	md_ztenmulc2(data->N, data->dom->dims, data->dom->strs, dst, MD_SINGLETON_STRS(data->N), &tmp, data->dom->strs, data->tmp_div);
}

static void cce_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	complex float tmp = src[0] * (complex float)(-1) / data->dom->dims[data->N-1];
	md_ztenmul2(data->N, data->dom->dims, data->dom->strs, dst, MD_SINGLETON_STRS(data->N), &tmp, data->dom->strs, data->tmp_log);
}

static void cce_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(cce_s, _data);

	md_free(data->tmp_div);
	md_free(data->tmp_log);
    	iovec_free(data->dom);

	xfree(data);
}

const struct nlop_s* nlop_cce_create(int N, const long dims[N])
{

	PTR_ALLOC(struct cce_s, data);
	SET_TYPEID(cce_s, data);

	data->N = N;
 	data->dom = iovec_create(N, dims, CFL_SIZE);

	// will be initialized later, to transparently support GPU
	data->tmp_div = NULL;
    	data->tmp_log = NULL;

	long nl_odims[1][1];
	md_copy_dims(1, nl_odims[0], MD_SINGLETON_DIMS(1));
	long nl_ostr[1][1];
	md_copy_strides(1, nl_ostr[0], MD_SINGLETON_STRS(1));

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);
	long nl_istr[2][N];
	md_copy_strides(N, nl_istr[0], MD_STRIDES(N, dims, CFL_SIZE));
	md_copy_strides(N, nl_istr[1], MD_STRIDES(N, dims, CFL_SIZE));

	return nlop_generic_create2(1, 1, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)), cce_fun, (nlop_fun_t[2][1]){ { cce_der1 }, { cce_der2 } }, (nlop_fun_t[2][1]){ { cce_adj1 }, { cce_adj2 } }, NULL, NULL, cce_del);
}
