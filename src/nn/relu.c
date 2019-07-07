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

#include "num/iovec.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "relu.h"


struct relu_s {

	INTERFACE(nlop_data_t);

	complex float* tmp;

	const struct iovec_s* tmpdom;
	const struct iovec_s* domain;
	const struct iovec_s* codomain;
};

DEF_TYPEID(relu_s);

static void relu_apply(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
        struct relu_s* d = CAST_DOWN(relu_s, _data);

	if (NULL == d->tmp)
		d->tmp = md_alloc_sameplace(d->tmpdom->N, d->tmpdom->dims, d->tmpdom->size, dst);

 	md_zreal2(d->tmpdom->N, d->tmpdom->dims, d->tmpdom->strs, d->tmp,
				  d->domain->strs, src);

        md_smax2(d->domain->N, d->domain->dims, d->codomain->strs, (float*)dst, d->tmpdom->strs, (const float*)d->tmp, 0.);

	md_sgreatequal2(d->tmpdom->N, d->tmpdom->dims, d->tmpdom->strs, (float*)d->tmp, d->tmpdom->strs, (const float*)d->tmp, 0.);
}


static void relu_deriv(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
        const struct relu_s* d = CAST_DOWN(relu_s, _data);

	assert(NULL != d->tmp);

	md_zrmul2(d->codomain->N, d->domain->dims, d->domain->strs, dst,
				  d->codomain->strs, src, d->tmpdom->strs, d->tmp);
}



static void relu_free(const nlop_data_t* _data)
{
        const struct relu_s* d = CAST_DOWN(relu_s, _data);

	md_free(d->tmp);

	iovec_free(d->tmpdom);
	iovec_free(d->domain);
	iovec_free(d->codomain);

	xfree(d);
}


const struct nlop_s* nlop_relu_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N])
{
	PTR_ALLOC(struct relu_s, data);
	SET_TYPEID(relu_s, data);

	long tstrs[N];
	md_calc_strides(N, tstrs, dims, CFL_SIZE);
	
	data->tmp = NULL;

        data->tmpdom = iovec_create2(N, dims, tstrs, CFL_SIZE);
        data->domain = iovec_create2(N, dims, istrs, CFL_SIZE);
        data->codomain = iovec_create2(N, dims, ostrs, CFL_SIZE);

        return nlop_create2(N, dims, ostrs, N, dims, istrs, CAST_UP(PTR_PASS(data)), relu_apply, relu_deriv, relu_deriv, relu_deriv, NULL, relu_free);
}

const struct nlop_s* nlop_relu_create(unsigned int N, const long dims[N])
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

        return nlop_relu_create2(N, dims, strs, strs);
}



