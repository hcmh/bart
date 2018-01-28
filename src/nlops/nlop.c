/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include "num/multind.h"

#include "num/ops.h"

#include "linops/linop.h"

#include "misc/shrdptr.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "misc/debug.h"
#include "num/iovec.h"

#include "nlop.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


struct nlop_op_data_s {

	INTERFACE(operator_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	nlop_fun_t forward1;
	nlop_gen_fun_t forward;
};

static DEF_TYPEID(nlop_op_data_s);


struct nlop_linop_data_s {

	INTERFACE(linop_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	nlop_fun_t deriv;
	nlop_fun_t adjoint;
	nlop_fun_t normal;
	nlop_p_fun_t norm_inv;
};

static DEF_TYPEID(nlop_linop_data_s);


static void sptr_op_del(const struct shared_ptr_s* sptr)
{
	auto data = CONTAINER_OF(sptr, struct nlop_op_data_s, sptr);
	data->del(data->data);
}

static void sptr_linop_del(const struct shared_ptr_s* sptr)
{
	auto data = CONTAINER_OF(sptr, struct nlop_linop_data_s, sptr);
	data->del(data->data);
}

static void op_fun(const operator_data_t* _data, unsigned int N, void* args[__VLA(N)])
{
	auto data = CAST_DOWN(nlop_op_data_s, _data);

	if (NULL != data->forward1) {

		assert(2 == N);
		data->forward1(data->data, args[0], args[1]);

	} else {

		data->forward(data->data, N, *(complex float* (*)[N])args);
	}
}

static void op_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(nlop_op_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	xfree(data);
}

static void lop_der(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->deriv(data->data, dst, src);
}

static void lop_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->adjoint(data->data, dst, src);
}

static void lop_nrm_inv(const linop_data_t* _data, float lambda, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->norm_inv(data->data, lambda, dst, src);
}

static void lop_nrm(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->normal(data->data, dst, src);
}


static void lop_del(const linop_data_t* _data)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	xfree(data);
}


struct nlop_s* nlop_generic_create2(int OO, int ON, const long odims[OO][ON], const long ostr[OO][ON], int II, int IN, const long idims[II][IN], const long istr[II][IN],
	nlop_data_t* data, nlop_gen_fun_t forward, nlop_fun_t deriv[II][OO], nlop_fun_t adjoint[II][OO], nlop_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del)
{
	PTR_ALLOC(struct nlop_s, n);

	PTR_ALLOC(struct nlop_op_data_s, d);
	SET_TYPEID(nlop_op_data_s, d);

	d->data = data;
	d->forward1 = NULL;
	d->forward = forward;
	d->del = del;

	shared_ptr_init(&d->sptr, sptr_op_del);



//	n->op = operator_create2(ON, odims, ostrs, IN, idims, istrs, CAST_UP(PTR_PASS(d)), op_fun, op_del);

	unsigned int D[OO + II];
	for (int i = 0; i < OO + II; i++)
		D[i] = (i < OO) ? ON : IN;

	const long* dims[OO + II];

	for (int i = 0; i < OO + II; i++)
		dims[i] = (i < OO) ? odims[i] : idims[i - OO];

	const long* strs[OO + II];

	for (int i = 0; i < OO + II; i++)
		strs[i] = (i < OO) ? ostr[i] : istr[i - OO];


	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);

	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++) {
		for (int o = 0; o < OO; o++) {

			PTR_ALLOC(struct nlop_linop_data_s, d2);
			SET_TYPEID(nlop_linop_data_s, d2);

			d2->data = data;
			d2->del = del;
			d2->deriv = deriv[i][o];
			d2->adjoint = adjoint[i][o];
			d2->normal = (NULL != normal) ? normal[i][o] : NULL;
			d2->norm_inv = (NULL != norm_inv) ? norm_inv[i][o] : NULL;

			shared_ptr_copy(&d2->sptr, &d->sptr);
			d2->sptr.del = sptr_linop_del;

			(*der)[i][o] = linop_create2(ON, odims[o], ostr[o], IN, idims[i], istr[i],
						CAST_UP(PTR_PASS(d2)), lop_der, lop_adj, lop_nrm, lop_nrm_inv, lop_del);
		}
	}

	n->op = operator_generic_create2(OO + II, (1u << OO) - 1u, D, dims, strs, CAST_UP(PTR_PASS(d)), op_fun, op_del);


	return PTR_PASS(n);
}

struct nlop_s* nlop_create2(unsigned int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)],
				unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_fun_t deriv, nlop_fun_t adjoint, nlop_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
{
	struct nlop_s* op = nlop_generic_create2(1, ON, (const long(*)[])&odims[0], (const long(*)[])&ostrs[0], 1, IN, (const long(*)[])&idims[0], (const long(*)[])&istrs[0], data, NULL,
					(nlop_fun_t[1][1]){ { deriv } }, (nlop_fun_t[1][1]){ { adjoint } }, (nlop_fun_t[1][1]){ { normal } }, (nlop_p_fun_t[1][1]){ { norm_inv } }, del);

	struct nlop_op_data_s* data2 = CAST_DOWN(nlop_op_data_s, operator_get_data(op->op));

	data2->forward1 = forward;

	return op;
}

struct nlop_s* nlop_create(unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_fun_t deriv, nlop_fun_t adjoint, nlop_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
{
	return nlop_create2(	ON, odims, MD_STRIDES(ON, odims, CFL_SIZE),
				IN, idims, MD_STRIDES(IN, idims, CFL_SIZE),
				data, forward, deriv, adjoint, normal, norm_inv, del);
}


int nlop_get_nr_in_args(const struct nlop_s* op)
{
	return operator_nr_in_args(op->op);
}


int nlop_get_nr_out_args(const struct nlop_s* op)
{
	return operator_nr_out_args(op->op);
}



void nlop_free(const struct nlop_s* op)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	operator_free(op->op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			linop_free((*der)[i][o]);

	xfree(der);
	xfree(op);
}


nlop_data_t* nlop_get_data(struct nlop_s* op)
{
	auto data2 = CAST_DOWN(nlop_op_data_s, operator_get_data(op->op));
#if 1
	auto data3 = CAST_DOWN(nlop_linop_data_s, linop_get_data(op->derivative[0]));
	assert(data3->data == data2->data);
#endif
	return data2->data;
}

void nlop_apply(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	operator_apply(op->op, IN, idims, dst, ON, odims, src);
}

void nlop_derivative(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	linop_forward(nlop_get_derivative(op, 0, 0), IN, idims, dst, ON, odims, src);
}

void nlop_adjoint(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	linop_adjoint(nlop_get_derivative(op, 0, 0), IN, idims, dst, ON, odims, src);
}



void nlop_generic_apply_unchecked(const struct nlop_s* op, int N, void* args[N])
{
	operator_generic_apply_unchecked(op->op, N, args);
}

const struct linop_s* nlop_get_derivative(const struct nlop_s* op, int o, int i)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	assert((i < II) && (o < OO));

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	return (*der)[i][o];
}

const struct iovec_s* nlop_generic_domain(const struct nlop_s* op, int o, int i)
{
	return linop_domain(nlop_get_derivative(op, o, i));
}

const struct iovec_s* nlop_generic_codomain(const struct nlop_s* op, int o, int i)
{
	return linop_codomain(nlop_get_derivative(op, o, i));
}



const struct iovec_s* nlop_domain(const struct nlop_s* op)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	return nlop_generic_domain(op, 0, 0);
}

const struct iovec_s* nlop_codomain(const struct nlop_s* op)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	return nlop_generic_codomain(op, 0, 0);
}

