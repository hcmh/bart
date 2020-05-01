
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

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "losses.h"

struct mse_s {

	INTERFACE(nlop_data_t);

	long N;
	const long* rdims;
	float scaling;

	float* tmp;
};

DEF_TYPEID(mse_s);

static void mse_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	START_TIMER;
	const auto data = CAST_DOWN(mse_s, _data);
	assert(3 == D);

	complex float* dst = args[0];
	const float* src1 = (float*)args[1];
	const float* src2 = (float*)args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif
	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->N, data->rdims, FL_SIZE, dst);

	md_sub(data->N, data->rdims, data->tmp, src1, src2);

	complex float result = md_scalar(data->N, data->rdims, data->tmp, data->tmp);
	complex float scale = 1. / data->scaling; 
	result = result * scale;
	md_copy(1, MAKE_ARRAY(1l), dst, &result, CFL_SIZE);
	PRINT_TIMER("frw mse");
}


static void mse_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	complex float result = md_scalar(data->N, data->rdims, data->tmp, (float*)src);
	complex float scale = 1. / data->scaling; 
	result = result * scale * 2;
	md_copy(1, MAKE_ARRAY(1l), dst, &result, CFL_SIZE);
}

static void mse_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	complex float result = md_scalar(data->N, data->rdims, data->tmp, (float*)src);
	complex float scale = 1. / data->scaling; 
	result = -(result * scale) * 2;
	md_copy(1, MAKE_ARRAY(1l), dst, &result, CFL_SIZE);
}

static void mse_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	float in;
	md_copy(1, MAKE_ARRAY(1l), &in, src, FL_SIZE);
	in *= 2. / data->scaling; 
	md_smul(data->N, data->rdims, (float*)dst, data->tmp, in);
	PRINT_TIMER("adj1 mse");
}

static void mse_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	float in;
	md_copy(1, MAKE_ARRAY(1l), &in, src, FL_SIZE);
	in *= -2. / data->scaling; 
	md_smul(data->N, data->rdims, (float*)dst, data->tmp, in);
	PRINT_TIMER("adj2 mse");
}

static void mse_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(mse_s, _data);

	md_free(data->tmp);
	xfree(data->rdims);
	xfree(data);
}

const struct nlop_s* nlop_mse_create(int N, const long dims[N], unsigned long mean_dims)
{
	PTR_ALLOC(struct mse_s, data);
	SET_TYPEID(mse_s, data);
	
	PTR_ALLOC(long[N + 1], rdims);
	(*rdims[0] = 2);
	md_copy_dims(N, *rdims + 1, dims);

	data->N = N + 1;
	data->rdims = *PTR_PASS(rdims);
	data->tmp = NULL;

	long tdims[N];
	md_select_dims(N, mean_dims, tdims, dims);
	data->scaling = (float)md_calc_size(N, tdims);

	long nl_odims[1][1];
	md_copy_dims(1, nl_odims[0], MD_SINGLETON_DIMS(1));
	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], data->rdims + 1);
	md_copy_dims(N, nl_idims[1], data->rdims + 1);


	return nlop_generic_create(1, 1, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), mse_fun, (nlop_fun_t[2][1]){ { mse_der1 }, { mse_der2 } }, (nlop_fun_t[2][1]){ { mse_adj1 }, { mse_adj2 } }, NULL, NULL, mse_del);
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
	START_TIMER;
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

	long odims[1];
	md_singleton_dims(1, odims);
	md_zsmul(1, odims, dst, dst, -1. / data->dom->dims[data->N-1]);
	PRINT_TIMER("loss cce");
}


static void cce_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	md_ztenmul2(data->N, data->dom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom->strs, src, data->dom->strs, data->tmp_div);
	md_zsmul(1, odims, dst, dst, -1. / data->dom->dims[data->N-1]);
}

static void cce_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	md_ztenmul2(data->N, data->dom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom->strs, src, data->dom->strs, data->tmp_log);
	md_zsmul(1, odims, dst, dst, -1. / data->dom->dims[data->N-1]);
}

static void cce_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	complex float* tmp = md_alloc_sameplace(1, odims, CFL_SIZE, dst);
	md_zsmul(1, odims,  tmp, src, (complex float)(-1) / data->dom->dims[data->N-1]);
	md_ztenmulc2(data->N, data->dom->dims, data->dom->strs, dst, MD_SINGLETON_STRS(data->N), tmp, data->dom->strs, data->tmp_div);
	md_free(tmp);
}

static void cce_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	complex float* tmp = md_alloc_sameplace(1, odims, CFL_SIZE, dst);
	md_zsmul(1, odims,  tmp, src, (complex float)(-1) / data->dom->dims[data->N-1]);
	md_ztenmul2(data->N, data->dom->dims, data->dom->strs, dst, MD_SINGLETON_STRS(data->N), tmp, data->dom->strs, data->tmp_log);
	md_free(tmp);
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
