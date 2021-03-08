
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

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
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
	md_zsmul(1, odims, tmp, src, (complex float)(-1) / data->scaling);
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
	md_zsmul(1, odims, tmp, src, (complex float)(-1) / data->scaling);
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
 * loss = - sum_i,j t_ij * log(p_ij(x)) / sum_i,j t_ij
 * where:	i - batch index
 *		j - label index
 *		t_ij = target prediction, i.e. 0 or 1 and sum_j t_ij = 1
 *		p_ij(x) = propability predicted by the network, i.e. p_i(x) in [0, 1] and sum_j p_ij(x) = 1 (softmax activation)
 *
 * @param N
 * @param dims
 **/
const struct nlop_s* nlop_cce_create(int N, const long dims[N], unsigned long batch_flag)
{

	PTR_ALLOC(struct cce_s, data);
	SET_TYPEID(cce_s, data);

	data->N = N;
 	data->dom = iovec_create(N, dims, CFL_SIZE);

	// will be initialized later, to transparently support GPU
	data->tmp_div = NULL;
	data->tmp_log = NULL;

	long scale_dims[N];
	md_select_dims(N, batch_flag, scale_dims, dims);
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

	const struct iovec_s* dom;
	const struct iovec_s* sum_dom;
};

DEF_TYPEID(frequency_compensation_s);

static void frequency_compensation_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(frequency_compensation_s, _data);

#ifdef USE_CUDA
	assert(cuda_ondevice(dst) == cuda_ondevice(src));
#endif

	complex float* sum = md_alloc_sameplace(data->sum_dom->N, data->sum_dom->dims, CFL_SIZE, src);
	md_zsum(data->N, data->dom->dims, data->batch_flag, sum, src);

	complex float* empty_labels = md_alloc_sameplace(data->sum_dom->N, data->sum_dom->dims, CFL_SIZE, src);
	md_zslessequal(data->sum_dom->N, data->sum_dom->dims, empty_labels, sum, 0);
	float N_empty_labels = roundf(powf(md_znorm(data->sum_dom->N, data->sum_dom->dims, empty_labels), 2));
	md_free(empty_labels);

	float N_batch_dims = (float)md_calc_size(data->N, data->dom->dims) / (float)md_calc_size(data->N, data->sum_dom->dims);
	float N_labels = (float)md_calc_size(data->N, data->sum_dom->dims) - N_empty_labels;

	md_zsmul(data->sum_dom->N, data->sum_dom->dims, sum, sum, N_labels / N_batch_dims);

	md_zdiv2(data->N, data->dom->dims, data->dom->strs, dst, data->dom->strs, src, data->sum_dom->strs, sum);

	md_free(sum);
}

static void frequency_compensation_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(frequency_compensation_s, _data);

	iovec_free(data->dom);
	iovec_free(data->sum_dom);

	xfree(data);
}

// dst_ij = src_ij / sum_j src_ij * (N_batch / N__non_empty_labels), where j corresponds to dimensions selected with batch_flag
// scaling is such that sum_ij dst_ij = sum_ij src_ij = N_batch
// usually batch_flag = ~MD_BIT(label_dim);
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

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), frequency_compensation_fun, NULL, NULL, NULL, NULL, frequency_compensation_del);
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


struct dice_s {

	INTERFACE(nlop_data_t);

	long N;

	const struct iovec_s* weight_dom;
	const struct iovec_s* dom;
	const struct iovec_s* cod;

	unsigned long mean_flag;

	complex float* weight;
	float weighting_exponent;

	complex float* min_src1;
	complex float* min_src2;

	complex float* src1_t2;
	complex float* src2_t2;

	bool square_denominator;
	complex float* numerator_sum;
	complex float* denominator_sum;

};

DEF_TYPEID(dice_s);

static void dice_initialize(struct dice_s* d, const void* arg)
{
	if (NULL == d->weight)
		d->weight = md_alloc_sameplace(d->dom->N, d->weight_dom->dims, d->dom->size, arg);
	if (NULL == d->min_src1)
		d->min_src1 = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, arg);
	if (NULL == d->min_src2)
		d->min_src2 = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, arg);
	if (d->square_denominator && (NULL == d->src1_t2))
		d->src1_t2 = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, arg);
	if (d->square_denominator && (NULL == d->src2_t2))
		d->src2_t2 = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, arg);
	if (NULL == d->numerator_sum)
		d->numerator_sum = md_alloc_sameplace(d->cod->N, d->cod->dims, d->cod->size, arg);
	if (NULL == d->denominator_sum)
		d->denominator_sum = md_alloc_sameplace(d->cod->N, d->cod->dims, d->cod->size, arg);
}

static void dice_compute_weights(struct dice_s* d, const complex float* ref)
{
	if (0. == d->weighting_exponent) {

		md_zfill(d->dom->N, d->weight_dom->dims, d->weight, 1);
		return;
	}

	md_clear(d->dom->N, d->weight_dom->dims, d->weight, d->dom->size);
	md_zadd2(d->dom->N, d->dom->dims, d->weight_dom->strs, d->weight, d->weight_dom->strs, d->weight, d->dom->strs, ref);
	md_zreal(d->dom->N, d->weight_dom->dims, d->weight, d->weight);

	complex float* tmp = md_alloc_sameplace(d->dom->N, d->weight_dom->dims, d->dom->size, ref);;
	md_zfill(d->dom->N, d->weight_dom->dims, tmp, 1);

	md_zdiv(d->dom->N, d->weight_dom->dims, d->weight, d->weight, tmp);
	md_zspow(d->dom->N, d->weight_dom->dims, d->weight, d->weight, -1. * d->weighting_exponent);

	md_free(tmp);
}

static void dice_red_der(const struct dice_s* d, complex float* dst, const complex float* src, const complex float* factor)
{
	complex float* tmp = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, dst);

	md_zreal(d->dom->N, d->dom->dims, tmp, src);

	if (NULL != factor)
		md_zmul(d->dom->N, d->dom->dims, tmp, tmp, factor);

	if (NULL != d->weight)
		md_zmul2(d->N, d->dom->dims, d->dom->strs, tmp, d->dom->strs, tmp, d->weight_dom->strs, d->weight);

	md_zsum(d->dom->N, d->dom->dims, ~d->mean_flag, dst, tmp);

	md_free(tmp);
}

static void dice_red_adj(const struct dice_s* d, complex float* dst, const complex float* src, const complex float* factor)
{
	md_copy2(d->dom->N, d->dom->dims, d->dom->strs, dst, d->cod->strs, src, CFL_SIZE);

	md_zreal(d->dom->N, d->dom->dims, dst, dst);

	if (NULL != factor)
		md_zmul(d->dom->N, d->dom->dims, dst, dst, factor);

	if (NULL != d->weight)
		md_zmul2(d->N, d->dom->dims, d->dom->strs, dst, d->dom->strs, dst, d->weight_dom->strs, d->weight);

}

static void dice_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	auto d = CAST_DOWN(dice_s, _data);
	assert(3 == D);

	//debug_print_dims(DP_INFO, data->N, data->rdims);

	complex float* dst = args[0];
	const _Complex float* src_pred = args[1];
	const _Complex float* src_true = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src_pred)) && (cuda_ondevice(src_pred) == cuda_ondevice(src_true)));
#endif
	int N = d->N;

	dice_initialize(d, src_true);
	dice_compute_weights(d, src_true);

	md_zlessequal(N, d->dom->dims, d->min_src1, src_pred, src_true);
	md_zlessequal(N, d->dom->dims, d->min_src2, src_true, src_pred);

	complex float* tmp = md_alloc_sameplace(N, d->dom->dims, d->dom->size, dst);
	md_zmul(N, d->dom->dims, tmp, d->min_src1, d->min_src2);
	md_zsmul(N, d->dom->dims, tmp, tmp, 0.5);
	md_zsub(N, d->dom->dims, d->min_src1, d->min_src1, tmp);
	md_zsub(N, d->dom->dims, d->min_src2, d->min_src2, tmp);

	complex float* numerator = md_alloc_sameplace(N, d->dom->dims, d->dom->size, dst);
	complex float* denominator = md_alloc_sameplace(N, d->dom->dims, d->dom->size, dst);

	md_zmul(N, d->dom->dims, numerator, src_pred, d->min_src1);
	md_zfmac(N, d->dom->dims, numerator, src_true, d->min_src2);

	if (d->square_denominator) {

		md_zmul(N, d->dom->dims, denominator, src_true, src_true);
		md_zfmac(N, d->dom->dims, denominator, src_pred, src_pred);

		md_zsmul(N, d->dom->dims, d->src1_t2, src_pred, 2);
		md_zsmul(N, d->dom->dims, d->src2_t2, src_true, 2);

		md_zreal(N, d->dom->dims, d->src1_t2, d->src1_t2);
		md_zreal(N, d->dom->dims, d->src2_t2, d->src2_t2);

	} else {

		md_zadd(N, d->dom->dims, denominator, src_true, src_pred);
	}


	dice_red_der(d, d->numerator_sum, numerator, NULL);
	dice_red_der(d, d->denominator_sum, denominator, NULL);

	complex float* tmp_sum = md_alloc_sameplace(d->cod->N, d->cod->dims, CFL_SIZE, dst);
	md_zdiv(d->cod->N, d->cod->dims, tmp_sum, d->numerator_sum, d->denominator_sum);

	md_zfill(d->cod->N, d->cod->dims, dst, 1.);
	md_zaxpy(d->cod->N, d->cod->dims, dst, -2., tmp_sum);

	md_free(tmp_sum);

	md_free(numerator);
	md_free(denominator);

	//for derivative
	md_zadd(N, d->dom->dims, d->min_src1, d->min_src1, tmp);
	md_zadd(N, d->dom->dims, d->min_src2, d->min_src2, tmp);
	md_free(tmp);
}

static void dice_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	const struct dice_s* d = CAST_DOWN(dice_s, _data);

	complex float* tmp_sum = md_alloc_sameplace(d->cod->N, d->cod->dims, CFL_SIZE, dst);

	dice_red_der(d, dst, src, (0 == i) ? d->min_src1 : d->min_src2);
	dice_red_der(d, tmp_sum, src, (0 == i) ? d->src1_t2 : d->src2_t2);

	md_zdiv(d->cod->N, d->cod->dims, tmp_sum, tmp_sum, d->denominator_sum);
	md_zmul(d->cod->N, d->cod->dims, tmp_sum, tmp_sum, d->numerator_sum);
	md_zaxpy(d->cod->N, d->cod->dims, dst, -1, tmp_sum);

	md_free(tmp_sum);

	md_zdiv(d->cod->N, d->cod->dims, dst, dst, d->denominator_sum);

	md_zsmul(d->cod->N, d->cod->dims, dst, dst, -2.);
}

static void dice_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	const struct dice_s* d = CAST_DOWN(dice_s, _data);
	int N = d->N;

	complex float* tmp_src = md_alloc_sameplace(d->cod->N, d->cod->dims, CFL_SIZE, dst);
	md_zdiv(d->cod->N, d->cod->dims, tmp_src, src, d->denominator_sum);
	md_zsmul(d->cod->N, d->cod->dims, tmp_src, tmp_src, -2);

	dice_red_adj(d, dst, tmp_src, (0 == i) ? d->min_src1 : d->min_src2);

	md_zmul(d->cod->N, d->cod->dims, tmp_src, tmp_src, d->numerator_sum);
	md_zdiv(d->cod->N, d->cod->dims, tmp_src, tmp_src, d->denominator_sum);

	complex float* tmp = md_alloc_sameplace(N, d->dom->dims, d->dom->size, dst);
	dice_red_adj(d, tmp, tmp_src, (0 == i) ? d->src1_t2 : d->src2_t2);

	md_zaxpy(N, d->dom->dims, dst, -1, tmp);
	md_free(tmp);
	md_free(tmp_src);
}

static void dice_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(dice_s, _data);

	md_free(data->weight);
	md_free(data->min_src1);
	md_free(data->min_src2);
	md_free(data->numerator_sum);
	md_free(data->denominator_sum);

	md_free(data->src1_t2);
	md_free(data->src2_t2);

	iovec_free(data->weight_dom);
	iovec_free(data->dom);
	iovec_free(data->cod);

	xfree(data);
}

/**
 * Generic Dice loss D
 *
 * Crum WR, Camara O, Hill DL. Generalized overlap measures for evaluation and validation in medical image analysis. IEEE Trans Med Imaging. 2006 Nov;25(11):1451-61. doi: 10.1109/TMI.2006.880587. PMID: 17117774.
 *
 * D = 1 - 2 * [sum_l w_l sum_i MIN(p_li, t_li)] / [sum_l w_l sum_i (p_li + t_li)]
 * where:	i - batch index
 *		l - label index
 *		w_l - wighting factor
 *		t_ij = target prediction (usually 0 or 1 and sum_j t_ij = 1)
 *		p_ij = propability predicted by the network (usually p_i(x) in [0, 1] and sum_j p_ij(x) = 1 (softmax activation))
 *
 * For t_ij in {0, 1}, MIN(p_li, t_li) = p_li * t_li resulting in the form presented in
 * Sudre C.H., Li W., Vercauteren T., Ourselin S., Jorge Cardoso M. (2017) Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. In: Cardoso M. et al. (eds) Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. DLMIA 2017, ML-CDS 2017. Lecture Notes in Computer Science, vol 10553. Springer, Cham. https://doi.org/10.1007/978-3-319-67558-9_28
 *
 * @param N
 * @param dims
 * @param label_flag select the label dimension
 * @param mean_flag computes dice loss over selected dimensions independently and averages afterwards (0 recommended)
 * @param weighting_exponent w_l = (V_l^wighting_exponent); should be in {0, -1, -2}; -2 corresponds to Sudre et.al.
 * @param square_denominator replace p_li by p_li^2 and t_li by t_li^2 in denominator
 **/
const struct nlop_s* nlop_dice_create(int N, const long dims[N], unsigned long label_flag, unsigned long mean_flag, float weighting_exponent, bool square_denominator)
{
	PTR_ALLOC(struct dice_s, data);
	SET_TYPEID(dice_s, data);

	data->N = N;

	long weight_dims[N];
	md_select_dims(N, label_flag, weight_dims, dims);

	long out_dims[N];
	md_select_dims(N, mean_flag, out_dims, dims);

	data->weight_dom = iovec_create(N, weight_dims, CFL_SIZE);
	data->dom = iovec_create(N, dims, CFL_SIZE);
	data->cod = iovec_create(N, out_dims, CFL_SIZE);

	data->mean_flag = mean_flag;

	data->weighting_exponent = weighting_exponent;

	data->weight = NULL;

	data->min_src1 = NULL;
	data->min_src2 = NULL;

	data->square_denominator = square_denominator;
	data->src1_t2 = NULL;
	data->src2_t2 = NULL;

	data->numerator_sum = NULL;
	data->denominator_sum = NULL;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], out_dims);
	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);


	auto dice = nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), dice_fun, (nlop_der_fun_t[2][1]){ { dice_der }, { dice_der } }, (nlop_der_fun_t[2][1]){ { dice_adj }, { dice_adj } }, NULL, NULL, dice_del);

	if (1 != md_calc_size(N, out_dims)) {

		auto linop_avg = linop_avg_create(N, out_dims, ~0);
		linop_avg = linop_chain_FF(linop_avg, linop_scale_create(N, MD_SINGLETON_DIMS(N), 1. / sqrtf(md_calc_size(N, out_dims))));//linop avg does not average

		dice = nlop_chain2_FF(dice, 0, nlop_from_linop_F(linop_avg), 0);
	}

	return nlop_reshape_out_F(dice, 0, 1, MD_DIMS(1));
}