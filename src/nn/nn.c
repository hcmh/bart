/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nn.h"


struct relu_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* domain;
	const struct iovec_s* codomain;
};

DEF_TYPEID(relu_s);

static void relu_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
        const struct relu_s* d = CAST_DOWN(relu_s, _data);
	assert(2 == N);
	// FIXMLE
        md_smax2(d->domain->N, d->domain->dims, d->codomain->strs, args[0], d->domain->strs, args[1], 0.);
}


static void relu_free(const operator_data_t* _data)
{
        const struct relu_s* d = CAST_DOWN(relu_s, _data);
	iovec_free(d->domain);
	iovec_free(d->codomain);
	xfree(d);
}


const struct operator_s* operator_relu_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N])
{
	PTR_ALLOC(struct relu_s, data);
	SET_TYPEID(relu_s, data);

        data->domain = iovec_create2(N, dims, istrs, CFL_SIZE);
        data->codomain = iovec_create2(N, dims, ostrs, CFL_SIZE);

        return operator_create2(N, dims, ostrs, N, dims, istrs, CAST_UP(PTR_PASS(data)), relu_apply, relu_free);
}

const struct operator_s* operator_relu_create(unsigned int N, const long dims[N])
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
        return operator_relu_create2(N, dims, strs, strs);
}




extern void simple_dcnn(const long dims[6], const long krn_dims[6], const complex float* krn, const long bias_dims[6], const complex float* bias, complex float* out, const complex float* in)
{
	unsigned int N = 6;
	unsigned int flags = 3;
	unsigned int layers = krn_dims[5];

	assert(krn_dims[3] == krn_dims[4]);
	assert(layers == bias_dims[5]);
	assert(krn_dims[4] == bias_dims[4]);
	assert((1 == dims[3]) && (1 == dims[4]));

	long krn_strs[N];
	md_calc_strides(N, krn_strs, krn_dims, CFL_SIZE);

	long bias_strs[N];
	md_calc_strides(N, bias_strs, bias_dims, CFL_SIZE);


	long dims2a[N];
	md_copy_dims(N, dims2a, dims);
	dims2a[3] = krn_dims[3];

	long dims2b[N];
	md_copy_dims(N, dims2b, dims);
	dims2b[4] = krn_dims[4];

	long strs2b[N];
	md_calc_strides(N, strs2b, dims2b, CFL_SIZE);

	complex float* tmp1 = md_calloc(N, dims2a, CFL_SIZE);
	complex float* tmp2 = md_alloc(N, dims2b, CFL_SIZE);

	md_copy(N, dims, tmp1, in, CFL_SIZE);

	const struct operator_s* relu = operator_relu_create(N, dims2b);

	long pos[6] = { 0 };

	for (unsigned int l = 0; l < layers; l++) {

		pos[5] = l;

		debug_printf(DP_INFO, "Layer: %d/%d\n", l, layers);

		struct linop_s* conv = linop_conv_create(5, flags, CONV_SYMMETRIC, CONV_CYCLIC,
				dims2b, dims2a, krn_dims, &MD_ACCESS(6, krn_strs, pos, krn));

		operator_apply(conv->forward, 5, dims2b, tmp2, 5, dims2a, tmp1);

		md_zadd2(N, dims2b, strs2b, tmp2, strs2b, tmp2, bias_strs, &MD_ACCESS(6, bias_strs, pos, bias));
		operator_apply(relu, N, dims2b, tmp1, N, dims2b, tmp2);

		linop_free(conv);
	}

	md_free(tmp1);

	md_copy(N, dims, out, tmp2, CFL_SIZE);	// skips relu in last iter

	md_free(tmp2);
}




struct op_dcnn_s {

	INTERFACE(operator_data_t);

	long dims[6];
	long krn_dims[6];
	long bias_dims[6];
	const complex float* krn;
	const complex float* bias;
	float alpha;
};


DEF_TYPEID(op_dcnn_s);

static void op_dcnn_apply(const operator_data_t* _data, float lambda, complex float* dst, const complex float* src)
{
	const struct op_dcnn_s* data = CAST_DOWN(op_dcnn_s, _data);

	// make a copy because dst maybe == src
	complex float* tmp = md_alloc(6, data->dims, CFL_SIZE);
	md_copy(6, data->dims, tmp, src, CFL_SIZE);

	simple_dcnn(data->dims, data->krn_dims, data->krn, data->bias_dims, data->bias, dst, src);

	md_zsmul(6, data->dims, dst, dst, data->alpha / lambda);
	md_zsub(6, data->dims, dst, tmp, dst);

	md_free(tmp);
}

static void op_dcnn_del(const operator_data_t* _data)
{
	const struct op_dcnn_s* data = CAST_DOWN(op_dcnn_s, _data);
	xfree(data);
}

const struct operator_p_s* prox_simple_dcnn_create(unsigned int N, const long dims[6], const long krn_dims[6], const complex float* krn, const long bias_dims[6], const complex float* bias, float alpha)
{
	PTR_ALLOC(struct op_dcnn_s, data);
	SET_TYPEID(op_dcnn_s, data);

	md_copy_dims(6, data->dims, dims);
	md_copy_dims(6, data->krn_dims, krn_dims);
	md_copy_dims(6, data->bias_dims, bias_dims);

	data->krn = krn;
	data->bias = bias;
	data->alpha = alpha;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), op_dcnn_apply, op_dcnn_del);
}
