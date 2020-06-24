/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include "num/multind.h"

#include "num/ops.h"
#include "num/ops_opts.h"
#include "num/iovec.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/shrdptr.h"
#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "nlops/nlop.h"
#include "nlop_props.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static operator_prop_flags_t nn_props_understood =
	  MD_BIT(OP_PROP_NN_IN_WEIGHT_CONV_CF)
	| MD_BIT(OP_PROP_NN_IN_WEIGHT_CONV_CL)
	| MD_BIT(OP_PROP_NN_IN_WEIGHT_DENSE)
	| MD_BIT(OP_PROP_NN_IN_WEIGHT_BIAS)
	| MD_BIT(OP_PROP_NN_BATCH_NORM);

operator_prop_flags_t nlop_get_ii_props(const struct nlop_s* op, unsigned int i1, unsigned int i2)
{
	unsigned int II = nlop_get_nr_in_args(op);
	unsigned int OO = nlop_get_nr_out_args(op);
	assert(i1 < II);
	assert(i2 < II);

	return operator_get_prop_flags(op->op, OO + i1, OO + i2);
}

operator_prop_flags_t nlop_get_oo_props(const struct nlop_s* op, unsigned int o1, unsigned int o2)
{
	//int II = nlop_get_nr_in_args(op);
	unsigned int OO = nlop_get_nr_out_args(op);
	assert(o1 < OO);
	assert(o2 < OO);

	return operator_get_prop_flags(op->op, o1, o2);
}

operator_prop_flags_t nlop_get_oi_props(const struct nlop_s* op, unsigned int o, unsigned int i)
{
	unsigned int II = nlop_get_nr_in_args(op);
	unsigned int OO = nlop_get_nr_out_args(op);
	assert(o < OO);
	assert(i < II);

	return operator_get_prop_flags(op->op, o, OO + i);
}

const struct nlop_s* nlop_set_nn_in_type_F(const struct nlop_s* op, unsigned int i, enum OPERATOR_IO_PROP_FLAGS_INDEX type)
{
	PTR_ALLOC(struct nlop_s, n);

	unsigned int II = nlop_get_nr_in_args(op);
	unsigned int OO = nlop_get_nr_out_args(op);
	assert(i < II);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (unsigned int i = 0; i < II; i++)
		for (unsigned int o = 0; o < OO; o++)
			(*nder)[i][o] = linop_clone((*der)[i][o]);

	n->derivative = &(*PTR_PASS(nder))[0][0];

	unsigned int N = operator_nr_args(op->op);

	assert(0 != (nn_props_understood & MD_BIT(type)));

	operator_prop_flags_t new_props[N][N];
	for (unsigned int i = 0 ; i < N; i++)
		for (unsigned int j = 0 ; j < N; j++)
			new_props[i][j] = operator_get_prop_flags(op->op, i, j);

	assert(0 == (new_props[OO + i][OO + i] & nn_props_understood));
	new_props[OO + i][OO + i] = MD_SET(new_props[OO + i][OO + i], type);

	n->op = operator_set_properties(op->op, N, new_props);

	nlop_free(op);

	return PTR_PASS(n);
}

operator_prop_flags_t nlop_get_nn_in_type(const struct nlop_s* op, unsigned int i)
{
	unsigned int II = nlop_get_nr_in_args(op);
	unsigned int OO = nlop_get_nr_out_args(op);
	assert(i < II);

	return nn_props_understood & operator_get_prop_flags(op->op, OO + i, OO + i);
}

const struct nlop_s* nlop_set_batchnorm_F(const struct nlop_s* op, unsigned int o, unsigned int i)
{
	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);
	assert((int)i < II);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*nder)[i][o] = linop_clone((*der)[i][o]);

	n->derivative = &(*PTR_PASS(nder))[0][0];

	unsigned int N = operator_nr_args(op->op);

	operator_prop_flags_t new_props[N][N];
	for (unsigned int i = 0 ; i < N; i++)
		for (unsigned int j = 0 ; j < N; j++)
			new_props[i][j] = operator_get_prop_flags(op->op, i, j);

	assert(0 == (new_props[OO + i][OO + i] & nn_props_understood));
	assert(0 == (new_props[o][o] & nn_props_understood));
	new_props[OO + i][OO + i] = MD_SET(new_props[OO + i][OO + i], OP_PROP_NN_BATCH_NORM);
	new_props[o][OO + i] = MD_SET(new_props[o][OO + i], OP_PROP_NN_BATCH_NORM);
	new_props[OO + i][o] = MD_SET(new_props[OO + i][o], OP_PROP_NN_BATCH_NORM);
	new_props[o][o] = MD_SET(new_props[o][o], OP_PROP_NN_BATCH_NORM);

	n->op = operator_set_properties(op->op, N, new_props);

	nlop_free(op);

	return PTR_PASS(n);
}