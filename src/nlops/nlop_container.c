/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include "num/multind.h"

#include "num/ops.h"
#include "num/iovec.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/shrdptr.h"
#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "nlop.h"
#include "nlop_container.h"
#include <assert.h>


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

struct nlop_container_s {

	INTERFACE(operator_data_t);

	const struct operator_s* x;
};

static DEF_TYPEID(nlop_container_s);

static void nlop_container_apply(const operator_data_t* _data, unsigned int N, void* args[N], operator_run_opt_flags_t run_opts[N][N])
{	
	const auto d = CAST_DOWN(nlop_container_s, _data);

	operator_generic_apply_extopts_unchecked(d->x, N, args, run_opts);

}

static void nlop_container_free(const operator_data_t* _data)
{
	const auto d = CAST_DOWN(nlop_container_s, _data);
	operator_free(d->x);
	xfree(d);
}

const struct nlop_s* nlop_container(const struct nlop_s* op)
{
	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*nder)[i][o] = linop_clone((*der)[i][o]);

	n->derivative = &(*PTR_PASS(nder))[0][0];

	PTR_ALLOC(struct nlop_container_s, data);
	SET_TYPEID(nlop_container_s, data);

	data->x = operator_ref(op->op);

	unsigned int A = operator_nr_args(op->op);
	unsigned int D[A];
	const long* op_dims[A];
	const long* op_strs[A];

	for (unsigned int j = 0; j < A; j++) {

		auto iov = operator_arg_domain(op->op, j);
		D[j] = iov->N;
		op_dims[j] = iov->dims;
		op_strs[j] = iov->strs;
	}

	operator_prop_flags_t props[II][OO];
	for (unsigned int i = 0; i < II; i++)
		for (unsigned int o = 0; o < OO; o++)
			props[i][o] = operator_get_prop_flags_oi(op->op, o, i);

	n->op = operator_generic_extopts_create2(A, operator_ioflags(op->op), D, op_dims, op_strs, CAST_UP(PTR_PASS(data)), nlop_container_apply, nlop_container_free, props);

	return PTR_PASS(n);
}
