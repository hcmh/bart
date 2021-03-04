
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

#include "nn/misc.h"

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
}


static void mse_der1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	complex float result = md_scalar(data->N, data->rdims, data->tmp, (float*)src);
	complex float scale = 1. / data->scaling;
	result = result * scale * 2;
	md_copy(1, MAKE_ARRAY(1l), dst, &result, CFL_SIZE);
}

static void mse_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	complex float result = md_scalar(data->N, data->rdims, data->tmp, (float*)src);
	complex float scale = 1. / data->scaling;
	result = -(result * scale) * 2;
	md_copy(1, MAKE_ARRAY(1l), dst, &result, CFL_SIZE);
}

static void mse_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	float in;
	md_copy(1, MAKE_ARRAY(1l), &in, src, FL_SIZE);
	in *= 2. / data->scaling;
	md_smul(data->N, data->rdims, (float*)dst, data->tmp, in);
}

static void mse_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct mse_s* data = CAST_DOWN(mse_s, _data);
	assert(NULL != data->tmp);

	float in;
	md_copy(1, MAKE_ARRAY(1l), &in, src, FL_SIZE);
	in *= -2. / data->scaling;
	md_smul(data->N, data->rdims, (float*)dst, data->tmp, in);
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


	return nlop_generic_create(1, 1, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), mse_fun, (nlop_der_fun_t[2][1]){ { mse_der1 }, { mse_der2 } }, (nlop_der_fun_t[2][1]){ { mse_adj1 }, { mse_adj2 } }, NULL, NULL, mse_del);
}


struct mpsnr_s {

	INTERFACE(nlop_data_t);

	long N;
	const long* dims;
	unsigned long mean_flag;
};

DEF_TYPEID(mpsnr_s);

static void mpsnr_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	const auto data = CAST_DOWN(mpsnr_s, _data);
	assert(3 == D);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif
	int N = data->N;
	const long* dims = data->dims;

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, src1);
	complex float* tmp3 = md_alloc_sameplace(N, dims, CFL_SIZE, src1);

	md_zabs(N, dims, tmp, src1);
	md_zabs(N, dims, tmp3, src2);
	md_zsub(N, dims, tmp, tmp, tmp3);
	md_free(tmp3);

	long mdims[N];
	md_select_dims(N, data->mean_flag, mdims, dims);

	complex float* tmp2 = md_alloc_sameplace(N, mdims, CFL_SIZE, src1);

	md_ztenmulc(N, mdims, tmp2, dims, tmp, dims, tmp);
	md_zsmul(N, mdims, tmp2, tmp2, (float)(md_calc_size(N, mdims)) / (float)(md_calc_size(N, dims)));
	md_zlog(N, mdims, tmp2, tmp2);

	md_clear(1, MD_DIMS(1), dst, CFL_SIZE);
	md_zadd2(N, mdims, MD_SINGLETON_STRS(N), dst, MD_SINGLETON_STRS(N), dst, MD_STRIDES(N, mdims, CFL_SIZE), tmp2);
	md_zsmul(1, MD_DIMS(1), dst, dst, -0.5);

	md_zabs(N, dims, tmp, src2);
	md_clear(N, mdims, tmp2, CFL_SIZE);
	md_zmax2(N, dims, MD_STRIDES(N, mdims, CFL_SIZE), tmp2, MD_STRIDES(N, mdims, CFL_SIZE), tmp2, MD_STRIDES(N, dims, CFL_SIZE), tmp);
	md_zlog(N, mdims, tmp2, tmp2);

	md_zadd2(N, mdims, MD_SINGLETON_STRS(N), dst, MD_SINGLETON_STRS(N), dst, MD_STRIDES(N, mdims, CFL_SIZE), tmp2);
	md_zsmul(1, MD_DIMS(1), dst, dst, 20. / (clogf(10) * md_calc_size(N, mdims)));

	md_free(tmp);
	md_free(tmp2);
}


static void mpsnr_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(mpsnr_s, _data);

	xfree(data->dims);
	xfree(data);
}

const struct nlop_s* nlop_mpsnr_create(int N, const long dims[N], unsigned long mean_dims)
{
	PTR_ALLOC(struct mpsnr_s, data);
	SET_TYPEID(mpsnr_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->mean_flag = mean_dims;

	long nl_odims[1][1];
	md_copy_dims(1, nl_odims[0], MD_SINGLETON_DIMS(1));
	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);

	return nlop_generic_create(1, 1, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), mpsnr_fun, (nlop_der_fun_t[2][1]){ { NULL }, { NULL } }, (nlop_der_fun_t[2][1]){ { NULL }, { NULL } }, NULL, NULL, mpsnr_del);
}


struct mssim_s {

	INTERFACE(nlop_data_t);

	long N;
	const long* dims;
	const long* kdims;
	const long* odims;

	unsigned long conv_flag;
	unsigned long mean_flag;

	float k1;
	float k2;
};

DEF_TYPEID(mssim_s);

static void mssim_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	const auto data = CAST_DOWN(mssim_s, _data);
	assert(3 == D);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif
	int N = data->N;
	const long* dims = data->dims;
	const long* kdims = data->kdims;
	const long* odims = data->odims;

	long mdims[N];
	md_select_dims(N, data->mean_flag, mdims, dims);

	complex float* krn_mean = md_alloc_sameplace(N, kdims, CFL_SIZE, src1);
	md_zfill(N, kdims, krn_mean, 1. / ((float)md_calc_size(N, kdims)));

	complex float* tmp1 = md_alloc_sameplace(N, dims, CFL_SIZE, src1);
	complex float* tmp2 = md_alloc_sameplace(N, dims, CFL_SIZE, src1);

	md_zabs(N, dims, tmp1, src1);
	md_zabs(N, dims, tmp2, src2);


	complex float* c1 = md_alloc_sameplace(N, mdims, CFL_SIZE, src1);
	complex float* c2 = md_alloc_sameplace(N, mdims, CFL_SIZE, src2);
	md_clear(N, mdims, c1, CFL_SIZE);
	md_zmax2(N, dims, MD_STRIDES(N, mdims, CFL_SIZE), c1, MD_STRIDES(N, mdims, CFL_SIZE), c1, MD_STRIDES(N, dims, CFL_SIZE), tmp2);
	md_zmul(N, mdims, c1, c1, c1);
	md_zsmul(N, mdims, c2, c1, data->k2 * data->k2);
	md_zsmul(N, mdims, c1, c1, data->k1 * data->k1);

	complex float* mu_x = md_alloc_sameplace(N, odims, CFL_SIZE, src1);
	complex float* mu_y = md_alloc_sameplace(N, odims, CFL_SIZE, src2);

	md_zcorr(N, data->conv_flag, odims, mu_x, kdims, krn_mean, dims, tmp1);
	md_zcorr(N, data->conv_flag, odims, mu_y, kdims, krn_mean, dims, tmp2);

	md_zmul(N, dims, tmp1, tmp1, tmp1);
	md_zmul(N, dims, tmp2, tmp2, tmp2);

	complex float* s_xx = md_alloc_sameplace(N, odims, CFL_SIZE, src1);
	complex float* s_yy = md_alloc_sameplace(N, odims, CFL_SIZE, src1);

	md_zcorr(N, data->conv_flag, odims, s_xx, kdims, krn_mean, dims, tmp1);
	md_zcorr(N, data->conv_flag, odims, s_yy, kdims, krn_mean, dims, tmp2);

	complex float* s_xy = md_alloc_sameplace(N, dims, CFL_SIZE, src1);
	md_zabs(N, dims, tmp1, src1);
	md_zabs(N, dims, tmp2, src2);
	md_zmul(N, dims, tmp1, tmp1, tmp2);
	md_zcorr(N, data->conv_flag, odims, s_xy, kdims, krn_mean, dims, tmp1);

	md_free(tmp1);
	md_free(tmp2);
	md_free(krn_mean);

	complex float* mu_xy = md_alloc_sameplace(N, odims, CFL_SIZE, src1);
	md_zmul(N, odims, mu_xy, mu_x, mu_y);

	md_zmul(N, odims, mu_x, mu_x, mu_x);
	md_zmul(N, odims, mu_y, mu_y, mu_y);

	md_zsub(N, odims, s_xx, s_xx, mu_x);
	md_zsub(N, odims, s_yy, s_yy, mu_y);
	md_zsub(N, odims, s_xy, s_xy, mu_xy);

	md_zsmul(N, odims, mu_xy, mu_xy, 2);
	md_zsmul(N, odims, s_xy, s_xy, 2);

	md_zadd(N, odims, mu_x, mu_x, mu_y);
	md_zadd(N, odims, s_xx, s_xx, s_yy);
	md_free(mu_y);
	md_free(s_yy);

	complex float* c1_large = md_alloc_sameplace(N, odims, CFL_SIZE, c1);
	complex float* c2_large = md_alloc_sameplace(N, odims, CFL_SIZE, c2);

	md_copy2(N, odims, MD_STRIDES(N, odims, CFL_SIZE), c1_large, MD_STRIDES(N, mdims, CFL_SIZE), c1, CFL_SIZE);
	md_copy2(N, odims, MD_STRIDES(N, odims, CFL_SIZE), c2_large, MD_STRIDES(N, mdims, CFL_SIZE), c2, CFL_SIZE);

	md_free(c1);
	md_free(c2);

	md_zadd(N, odims, mu_xy, mu_xy, c1_large);
	md_zadd(N, odims, s_xy, s_xy, c2_large);
	md_zadd(N, odims, mu_x, mu_x, c1_large);
	md_zadd(N, odims, s_xx, s_xx, c2_large);

	md_free(c1_large);
	md_free(c2_large);

	md_zmul(N, odims, mu_xy, mu_xy, s_xy);
	md_zmul(N, odims, mu_x, mu_x, s_xx);
	md_free(s_xy);
	md_free(s_xx);

	md_zdiv(N, odims, mu_xy, mu_xy, mu_x);
	md_free(mu_x);

	md_zavg(N, odims, ~0, dst, mu_xy);
	md_free(mu_xy);

}


static void mssim_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(mssim_s, _data);

	xfree(data->dims);
	xfree(data->kdims);
	xfree(data->odims);
	xfree(data);
}

const struct nlop_s* nlop_mssim_create(int N, const long dims[N], const long kdims[N], unsigned long conv_dims)
{
	PTR_ALLOC(struct mssim_s, data);
	SET_TYPEID(mssim_s, data);

	assert(7 == conv_dims);
	assert(3 <= N);

	data->N = 6;

	data->mean_flag = ~(MD_BIT(5) - 1);
	data->conv_flag = 28;

	PTR_ALLOC(long[data->N], ndims);
	md_singleton_dims(data->N, *ndims);
	md_copy_dims(3, (*ndims) + 2, dims);
	(*ndims)[data->N - 1] = md_calc_size(N - 3, dims + 3);
	data->dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[data->N], nkdims);
	md_singleton_dims(data->N, *nkdims);
	md_copy_dims(3, (*nkdims) + 2, kdims);
	data->kdims = *PTR_PASS(nkdims);

	PTR_ALLOC(long[data->N], odims);
	for (int i = 0; i < data->N; i++)
		if (MD_IS_SET(data->conv_flag, i))
			(*odims)[i] = (data->dims)[i] + 1 - (data->kdims)[i];
		else
			(*odims)[i] = (data->dims)[i];

	data->odims = *PTR_PASS(odims);

	data->k1 = 0.01;
	data->k2 = 0.03;

	long nl_odims[1][1];
	md_copy_dims(1, nl_odims[0], MD_SINGLETON_DIMS(1));
	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);

	return nlop_generic_create(1, 1, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), mssim_fun, (nlop_der_fun_t[2][1]){ { NULL }, { NULL } }, (nlop_der_fun_t[2][1]){ { NULL }, { NULL } }, NULL, NULL, mssim_del);
}


struct cce_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	float scaling;
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

	long odims[1];
	md_singleton_dims(1, odims);

	md_zsmul(1, odims, dst, dst, -1. / data->scaling);
}


static void cce_der1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	md_ztenmul2(data->N, data->dom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom->strs, src, data->dom->strs, data->tmp_div);
	md_zsmul(1, odims, dst, dst, -1. / data->scaling);
}

static void cce_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	md_ztenmul2(data->N, data->dom->dims, MD_SINGLETON_STRS(data->N), dst, data->dom->strs, src, data->dom->strs, data->tmp_log);
	md_zsmul(1, odims, dst, dst, -1. / data->scaling);
}

static void cce_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	complex float* tmp = md_alloc_sameplace(1, odims, CFL_SIZE, dst);
	md_zsmul(1, odims,  tmp, src, (complex float)(-1) / data->scaling);
	md_ztenmulc2(data->N, data->dom->dims, data->dom->strs, dst, MD_SINGLETON_STRS(data->N), tmp, data->dom->strs, data->tmp_div);
	md_free(tmp);
}

static void cce_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct cce_s* data = CAST_DOWN(cce_s, _data);
	assert(NULL != data->tmp_log);
	assert(NULL != data->tmp_div);

	long odims[1];
	md_singleton_dims(1, odims);
	complex float* tmp = md_alloc_sameplace(1, odims, CFL_SIZE, dst);
	md_zsmul(1, odims,  tmp, src, (complex float)(-1) / data->scaling);
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

/**
 * Categorical cross entropy
 *
 * calculate cce with channels in first dimension
 *
 * loss = - sum_i,j t_ij * log(p_ij(x))
 * where:	i - batch index
 *		j - label index
 *		t_ij = target prediction, i.e. 0 or 1 and sum_j t_ij = 1
 *		p_ij(x) = propability predicted by the network, i.e. p_i(x) in [0, 1] and sum_j p_ij(x) = 1 (softmax activation)
 *
 * @param N
 * @param dims
 **/
const struct nlop_s* nlop_cce_create(int N, const long dims[N], unsigned long scaling_flag)
{

	PTR_ALLOC(struct cce_s, data);
	SET_TYPEID(cce_s, data);

	data->N = N;
 	data->dom = iovec_create(N, dims, CFL_SIZE);

	// will be initialized later, to transparently support GPU
	data->tmp_div = NULL;
    	data->tmp_log = NULL;

	long scale_dims[N];
	md_select_dims(N, scaling_flag, scale_dims, dims);
	data->scaling = md_calc_size(N, scale_dims);

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

	return nlop_generic_create2(1, 1, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)), cce_fun, (nlop_der_fun_t[2][1]){ { cce_der1 }, { cce_der2 } }, (nlop_der_fun_t[2][1]){ { cce_adj1 }, { cce_adj2 } }, NULL, NULL, cce_del);
}

struct accuracy_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const struct iovec_s* dom;
	int class_index;
};

DEF_TYPEID(accuracy_s);

static void accuracy_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	const auto data = CAST_DOWN(accuracy_s, _data);
	assert(3 == D);

	complex float* dst = args[0];
	const complex float* src_pred = args[1];
	const complex float* src_true = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src_pred)) && (cuda_ondevice(src_pred) == cuda_ondevice(src_true)));
#endif

	complex float dst_t = 0;
	complex float* src_pred_t = md_alloc(data->dom->N, data->dom->dims, data->dom->size);
	complex float* src_true_t = md_alloc(data->dom->N, data->dom->dims, data->dom->size);

	md_copy(data->dom->N, data->dom->dims, src_pred_t, src_pred, data->dom->size);
	md_copy(data->dom->N, data->dom->dims, src_true_t, src_true, data->dom->size);

	dst_t = onehotenc_accuracy(data->dom->N, data->dom->dims, data->class_index, src_pred_t, src_true_t);

	md_copy(1, MD_DIMS(1), dst, &dst_t, CFL_SIZE);

	md_free(src_pred_t);
	md_free(src_true_t);
}

static void accuracy_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(accuracy_s, _data);

    	iovec_free(data->dom);

	xfree(data);
}

/**
 * Accuracy
 *
 * @param N
 * @param dims
 * @
 **/
const struct nlop_s* nlop_accuracy_create(int N, const long dims[N], int class_index)
{

	PTR_ALLOC(struct accuracy_s, data);
	SET_TYPEID(accuracy_s, data);

	data->N = N;
 	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->class_index = class_index;

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

	return nlop_generic_create2(1, 1, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)), accuracy_fun, (nlop_der_fun_t[2][1]){ { NULL }, { NULL } }, (nlop_der_fun_t[2][1]){ { NULL }, { NULL } }, NULL, NULL, accuracy_del);
}



struct frequency_compensation_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	unsigned long batch_flag;
	complex float* in;
	complex float* sum;
	const struct iovec_s* dom;
	const struct iovec_s* sum_dom;
};

DEF_TYPEID(frequency_compensation_s);

static void frequency_compensation_initialize(struct frequency_compensation_s* data, const complex float* arg)
{
	if (NULL == data->in)
		data->in = md_alloc_sameplace(data->dom->N, data->dom->dims, CFL_SIZE, arg);
	if (NULL == data->sum)
		data->sum = md_alloc_sameplace(data->dom->N, data->sum_dom->dims, CFL_SIZE, arg);
}

static void frequency_compensation_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(frequency_compensation_s, _data);

#ifdef USE_CUDA
	assert(cuda_ondevice(dst) == cuda_ondevice(src));
#endif
	frequency_compensation_initialize(data, dst);

	md_copy(data->N, data->dom->dims, data->in, src, data->dom->size);
	md_zsum(data->N, data->dom->dims, data->batch_flag, data->sum, src);
	md_zsmul(data->N, data->sum_dom->dims, data->sum, data->sum, 1. / md_calc_size(data->N, data->sum_dom->dims));
	md_zdiv2(data->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->sum_dom->strs, data->sum);

}


static void frequency_compensation_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	UNUSED(dst);
	UNUSED(src);
	UNUSED(_data);
	error("loss frequency compensation derivative not implemented");
}

static void frequency_compensation_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	UNUSED(dst);
	UNUSED(src);
	UNUSED(_data);
	error("loss frequency compensation derivative not implemented");
}

static void frequency_compensation_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(frequency_compensation_s, _data);

	md_free(data->in);
	md_free(data->sum);
    	iovec_free(data->dom);
	iovec_free(data->sum_dom);

	xfree(data);
}

// dst_ij = src_ij / sum_j src_ij, where j corresponds to dimensions selected with batch_flag
static const struct nlop_s* nlop_frequency_compensation_create(int N, const long dims[N], unsigned long batch_flag)
{

	PTR_ALLOC(struct frequency_compensation_s, data);
	SET_TYPEID(frequency_compensation_s, data);

	long sum_dims[N];
	md_select_dims(N, ~batch_flag, sum_dims, dims);

	data->N = N;
 	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->sum_dom = iovec_create(N, sum_dims, CFL_SIZE);
	data->batch_flag = batch_flag;

	// will be initialized later, to transparently support GPU
	data->sum = NULL;
    	data->in = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), frequency_compensation_fun, frequency_compensation_der, frequency_compensation_adj, NULL, NULL, frequency_compensation_del);
}

/**
 * Weighted categorical crossentropy
 *
 * loss = - sum_i,j w_j t_ij * log(p_ij(x))
 * where:	i - batch index
 *		j - label index
 *		t_ij = target prediction, i.e. 0 or 1 and sum_j t_ij = 1
 *		p_ij(x) = propability predicted by the network, i.e. p_i(x) in [0, 1] and sum_j p_ij(x) = 1 (softmax activation)
 *		w_j = 1 / sum_i t_ij
 *
 * @param N
 * @param dims
 * @param batch_flag selected dims correspond to i, unselected to j
 **/
const struct nlop_s* nlop_weighted_cce_create(int N, const long dims[N], unsigned long batch_flag)
{
	//FIXME: more flexible batchflags (scaling in cce and interface change for batch_flag needed)
	assert(MD_BIT(N-1) == batch_flag);

	return nlop_chain2_FF(nlop_frequency_compensation_create(N, dims, batch_flag), 0, nlop_cce_create(N, dims, ~MD_BIT(0)), 1);
}
