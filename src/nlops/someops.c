/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "linops/linop.h"
#include "linops/someops.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/iovec.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "someops.h"


struct zaxpbz_s {

	INTERFACE(operator_data_t);

	int N;
	const long* dims;

	complex float scale1;
	complex float scale2;
};

DEF_TYPEID(zaxpbz_s);

static void zaxpbz_fun(const operator_data_t* _data, unsigned int N, void* args[N])
{
	START_TIMER;
	const auto data = CAST_DOWN(zaxpbz_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

	#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
	#endif

	if ((1. == data->scale1) && (1. == data->scale2)) {

		md_zadd(data->N, data->dims, dst, src1, src2);
		PRINT_TIMER("frw zaxpbz");
		return;
	}

	if ((1. == data->scale1) && (-1. == data->scale2)) {

		md_zsub(data->N, data->dims, dst, src1, src2);
		PRINT_TIMER("frw zaxpbz");
		return;
	}

	if ((-1. == data->scale1) && (1. == data->scale2)) {

		md_zsub(data->N, data->dims, dst, src2, src1);
		PRINT_TIMER("frw zaxpbz");
		return;
	}

	md_zsmul(data->N, data->dims, dst, src1, data->scale1);
	md_zaxpy(data->N, data->dims, dst, data->scale2, src2);
	PRINT_TIMER("frw zaxpbz");
}

static void zaxpbz_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(zaxpbz_s, _data);

	xfree(data->dims);

	xfree(data);
}

static const struct operator_s* operator_zaxpbz_create(int N, const long dims[N], complex float scale1, complex float scale2)
{

	PTR_ALLOC(struct zaxpbz_s, data);
	SET_TYPEID(zaxpbz_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);

	data->scale1 = scale1;
	data->scale2 = scale2;

	const long* op_dims[3] = {dims, dims, dims};

	unsigned int Ns[3] = {N, N, N};

	return operator_generic_create(3, MD_BIT(0), Ns, op_dims, CAST_UP(PTR_PASS(data)), zaxpbz_fun, zaxpbz_del);
}


const struct nlop_s* nlop_zaxpbz_create(int N, const long dims[N], complex float scale1, complex float scale2)
{
	PTR_ALLOC(struct nlop_s, n);
	const struct linop_s* (*der)[2][1] = TYPE_ALLOC(const struct linop_s*[2][1]);
	n->derivative = &(*der)[0][0];

	n->op = operator_zaxpbz_create(N, dims, scale1, scale2);
	(*der)[0][0] = linop_scale_create(N, dims, scale1);
	(*der)[1][0] = linop_scale_create(N, dims, scale2);

	return PTR_PASS(n);
}

struct smo_abs_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const long* dims;

	complex float epsilon;
	complex float* tmp;
};

DEF_TYPEID(smo_abs_s);

static void smo_abs_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	const auto data = CAST_DOWN(smo_abs_s, _data);

	long rdims[data->N + 1];
	rdims[0] = 2;
	md_copy_dims(data->N, rdims + 1, data->dims);

	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	md_zmulc(data->N, data->dims, dst, src, src);//dst=[r0^2 + i0^2 + 0i, r1^2 + i1^2 + 0i, ...]
	md_zreal(data->N, data->dims, dst, dst); //zmulc does not gurantee vanishing imag on gpu
	md_zsadd(data->N, data->dims, dst, dst, data->epsilon);
	md_sqrt(data->N+1, rdims, (float*)dst, (float*)dst);
	md_zdiv(data->N, data->dims, data->tmp, src, dst);

	PRINT_TIMER("frw smoabs");
}


static void smo_abs_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct smo_abs_s* data = CAST_DOWN(smo_abs_s, _data);
	assert(NULL != data->tmp);

	md_zmulc(data->N, data->dims, dst, data->tmp, src);
	md_zreal(data->N, data->dims, dst, dst);
}

static void smo_abs_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;

	const struct smo_abs_s* data = CAST_DOWN(smo_abs_s, _data);
	assert(NULL != data->tmp);

	md_zreal(data->N, data->dims, dst, src);
	md_zmul(data->N, data->dims, dst, dst, data->tmp);

	PRINT_TIMER("adj smoabs");
}

static void smo_abs_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(smo_abs_s, _data);

	md_free(data->tmp);
	xfree(data->dims);
	xfree(data);
}

/**
 * Operator computing the smoothed pointwise absolute value
 * f(x) = sqrt(re(x)^2 + im (x)^2 + epsilon)
 */
const struct nlop_s* nlop_smo_abs_create(int N, const long dims[N], float epsilon)
{
	PTR_ALLOC(struct smo_abs_s, data);
	SET_TYPEID(smo_abs_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->epsilon = epsilon;

	// will be initialized later, to transparently support GPU
	data->tmp = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), smo_abs_fun, smo_abs_der, smo_abs_adj, NULL, NULL, smo_abs_del);
}


struct dump_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const long* dims;

	const char* filename;
	long counter;

	bool frw;
	bool der;
	bool adj;
};

DEF_TYPEID(dump_s);

static void dump_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(dump_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);

	if (data->frw) {

		char filename[strlen(data->filename) + 10];
		sprintf(filename, "%s_%ld_frw", data->filename, data->counter);
		dump_cfl(filename, data->N, data->dims, src);
		data->counter++;
	}
}

static void dump_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(dump_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);

	if (data->der) {

		char filename[strlen(data->filename) + 10];
		sprintf(filename, "%s_%ld_der", data->filename, data->counter);
		dump_cfl(filename, data->N, data->dims, src);
		data->counter++;
	}
}

static void dump_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(dump_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);

	if (data->adj) {

		char filename[strlen(data->filename) + 10];
		sprintf(filename, "%s_%ld_adj", data->filename, data->counter);
		dump_cfl(filename, data->N, data->dims, src);
		data->counter++;
	}
}

static void dump_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(dump_s, _data);

	xfree(data->dims);
	xfree(data);
}

/**
 * Operator dumping its input to a filename_%d_frw/der/adj.cfl file
 * @param N
 * @param dims
 * @param filename
 * @param frw - store frw input
 * @param der - store der input
 * @param adj - store adj input
 */

const struct nlop_s* nlop_dump_create(int N, const long dims[N], const char* filename, bool frw, bool der, bool adj)
{
	PTR_ALLOC(struct dump_s, data);
	SET_TYPEID(dump_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);

	PTR_ALLOC(char[strlen(filename) + 1], nfilename);
	strcpy(*nfilename, filename);
	data->filename = *PTR_PASS(nfilename);

	data->frw = frw;
	data->der = der;
	data->adj = adj;

	data->counter = 0;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), dump_fun, dump_der, dump_adj, NULL, NULL, dump_del);
}