/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "bias.h"


struct bias_s {

	INTERFACE(nlop_data_t);

	const complex float* bias;

	int N;
	const long* dims;
	const long* ostrs;
	const long* istrs1;
	const long* istrs2;
};

DEF_TYPEID(bias_s);

static void bias_apply(const nlop_data_t* _data, int N, complex float* args[N])
{
        const struct bias_s* d = CAST_DOWN(bias_s, _data);
	assert(2 == N);

        md_zadd2(d->N, d->dims, d->ostrs, args[0], d->istrs1, args[1], d->istrs2, d->bias);
}

static void bias_deriv(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	UNUSED(dst); UNUSED(src);
	UNUSED(_data);
	error("Not implemented\n");
}

static void bias_free(const nlop_data_t* _data)
{
        const struct bias_s* d = CAST_DOWN(bias_s, _data);

//	md_free(d->bias);
	xfree(d->dims);
	xfree(d->ostrs);
	xfree(d->istrs1);
	xfree(d->istrs2);

	xfree(d);
}


const struct nlop_s* nlop_bias_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N], const long bstrs[N], const complex float* bias)
{
	PTR_ALLOC(struct bias_s, data);
	SET_TYPEID(bias_s, data);

	data->N = N;

	PTR_ALLOC(long[N], tdims);
	md_copy_dims(N, *tdims, dims);
	data->dims = *PTR_PASS(tdims);

	PTR_ALLOC(long[N], tostrs);
	md_copy_strides(N, *tostrs, ostrs);
	data->ostrs = *PTR_PASS(tostrs);

	PTR_ALLOC(long[N], tistrs1);
	md_copy_strides(N, *tistrs1, istrs);
	data->istrs1 = *PTR_PASS(tistrs1);

	PTR_ALLOC(long[N], tistrs2);
	md_copy_strides(N, *tistrs2, bstrs);
	data->istrs2 = *PTR_PASS(tistrs2);


	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);

	long nl_ostrs[1][N];
	md_copy_strides(N, nl_ostrs[0], ostrs);

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);

	long nl_istrs[2][N];
	md_copy_strides(N, nl_istrs[0], istrs);
	md_copy_strides(N, nl_istrs[1], bstrs);

	data->bias = bias;

	return nlop_generic_create2(1, N, nl_odims, nl_ostrs, 1, N, nl_idims, nl_istrs, CAST_UP(PTR_PASS(data)),
		bias_apply, (nlop_fun_t[1][1]){ { bias_deriv } }, (nlop_fun_t[1][1]){ { bias_deriv } }, NULL, NULL, bias_free);
}

const struct nlop_s* nlop_bias_create(unsigned int N, const long dims[N], const long bdims[N], const complex float* bias)
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long bstrs[N];
	md_calc_strides(N, bstrs, bdims, CFL_SIZE);

        return nlop_bias_create2(N, dims, strs, strs, bstrs, bias);
}
