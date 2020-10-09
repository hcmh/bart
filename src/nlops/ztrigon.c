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

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "ztrigon.h"


struct zsin_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(zsin_s);

static void zsin_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zsin_s, _data);
	if (NULL == data->xn)
		data->xn = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);
	md_zsin(data->N, data->dims, dst, src);
	md_zcos(data->N, data->dims, data->xn, src);
}

static void zsin_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zsin_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void zsin_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zsin_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zsin_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zsin_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}

struct nlop_s* nlop_zsin_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zsin_s, data);
	SET_TYPEID(zsin_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = NULL;
	

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)),
		zsin_fun, zsin_der, zsin_adj, NULL, NULL, zsin_del);
}

struct zcos_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(zcos_s);

static void zcos_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zcos_s, _data);
	if (NULL == data->xn)
		data->xn = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);
	md_zcos(data->N, data->dims, dst, src);
	md_zsin(data->N, data->dims, data->xn, src);
	md_zsmul(data->N, data->dims, data->xn, data->xn, -1);
}

static void zcos_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zcos_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void zcos_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zcos_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zcos_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zcos_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}

struct nlop_s* nlop_zcos_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zcos_s, data);
	SET_TYPEID(zcos_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)),
		zcos_fun, zcos_der, zcos_adj, NULL, NULL, zcos_del);
}
