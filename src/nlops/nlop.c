/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <string.h>
#include <stdio.h>

#include "nn/batchnorm.h"
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
#include <assert.h>


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

//only these operator properties are passed to linops
static operator_property_flags_t nlops_props_understood = MD_BIT(OP_PROP_ATOMIC)
							| MD_BIT(OP_PROP_R_LIN)
							| MD_BIT(OP_PROP_C_LIN)
							| MD_BIT(OP_PROP_HOLOMORPHIC)
							| MD_BIT(OP_PROP_INDEPENDENT);

struct nlop_run_stats_s {

	unsigned int II;
	unsigned int OO;

	double run_time;

	unsigned long frw_calls;
	unsigned long* der_calls;
	unsigned long* adj_calls;

	double frw_time;
	double* der_time;
	double* adj_time;
};

static struct nlop_run_stats_s* stats_create(unsigned int OO, unsigned int II){

	PTR_ALLOC(struct nlop_run_stats_s, result);

	PTR_ALLOC(unsigned long[OO * II], adj_calls);
	PTR_ALLOC(unsigned long[OO * II], der_calls);
	PTR_ALLOC(double[OO * II], adj_time);
	PTR_ALLOC(double[OO * II], der_time);

	result->run_time = 0.;

	result->frw_calls = 0;
	result->frw_time = 0.;
	result->der_calls = *PTR_PASS(der_calls);
	result->adj_calls = *PTR_PASS(adj_calls);
	result->der_time = *PTR_PASS(der_time);
	result->adj_time = *PTR_PASS(adj_time);

	for(uint i = 0; i < OO * II; i++) {

		result->der_calls[i] = 0;
		result->adj_calls[i] = 0;
		result->der_time[i] = 0.;
		result->adj_time[i] = 0.;
	}

	result->II = II;
	result->OO = OO;

	return PTR_PASS(result);
}

static void stats_free(const struct nlop_run_stats_s* stats)
{
	xfree(stats->der_calls);
	xfree(stats->der_time);
	xfree(stats->adj_calls);
	xfree(stats->adj_time);
	xfree(stats);
}

static void nlop_start_frw(struct nlop_run_stats_s* stats)
{
	#pragma omp critical
	stats->run_time -= timestamp();
	#pragma omp critical
	stats->frw_time -= timestamp();
}
static void nlop_finish_frw(struct nlop_run_stats_s* stats)
{
	#pragma omp critical
	stats->run_time += timestamp();
	#pragma omp critical
	stats->frw_time += timestamp();
	stats->frw_calls ++;
}
static void nlop_start_der(struct nlop_run_stats_s* stats, unsigned int o, unsigned int i)
{
	#pragma omp critical
	stats->run_time -= timestamp();
	#pragma omp critical
	stats->der_time[o + stats->OO * i] -= timestamp();
}
static void nlop_finish_der(struct nlop_run_stats_s* stats, unsigned int o, unsigned int i)
{
	#pragma omp critical
	stats->run_time += timestamp();
	#pragma omp critical
	stats->der_time[o + stats->OO * i] += timestamp();
	#pragma omp critical
	stats->der_calls[o + stats->OO * i] += 1;
}
static void nlop_start_adj(struct nlop_run_stats_s* stats, unsigned int o, unsigned int i)
{
	#pragma omp critical
	stats->run_time -= timestamp();
	#pragma omp critical
	stats->adj_time[o + stats->OO * i] -= timestamp();
}
static void nlop_finish_adj(struct nlop_run_stats_s* stats, unsigned int o, unsigned int i)
{
	#pragma omp critical
	stats->run_time += timestamp();
	#pragma omp critical
	stats->adj_time[o + stats->OO * i] += timestamp();
	#pragma omp critical
	stats->adj_calls[o + stats->OO * i] += 1;
}

struct nlop_op_data_s {

	INTERFACE(operator_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	nlop_fun_t forward1;
	nlop_gen_fun_t forward;

	nlop_set_opts_t set_opts;
	nlop_graph_t get_graph;
};

static DEF_TYPEID(nlop_op_data_s);


struct nlop_linop_data_s {

	INTERFACE(linop_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	unsigned int o;
	unsigned int i;

	nlop_der_fun_t deriv;
	nlop_der_fun_t adjoint;
	nlop_der_fun_t normal;
	nlop_p_fun_t norm_inv;
};

static DEF_TYPEID(nlop_linop_data_s);


static void sptr_op_del(const struct shared_ptr_s* sptr)
{
	auto data = CONTAINER_OF(sptr, struct nlop_op_data_s, sptr);
	op_options_free(data->data->options);
	stats_free(data->data->stats);
	data->del(data->data);
}

static void sptr_linop_del(const struct shared_ptr_s* sptr)
{
	auto data = CONTAINER_OF(sptr, struct nlop_linop_data_s, sptr);
	op_options_free(data->data->options);
	stats_free(data->data->stats);
	data->del(data->data);
}

static void op_fun(const operator_data_t* _data, unsigned int N, void* args[__VLA(N)])
{
	auto data = CAST_DOWN(nlop_op_data_s, _data);

	nlop_start_frw(data->data->stats);

	if (NULL != data->forward1) {

		assert(2 == N);
		data->forward1(data->data, args[0], args[1]);
		op_options_free(data->data->options);
		data->data->options = NULL;
	}

	if (NULL != data->forward) {

		data->forward(data->data, N, *(complex float* (*)[N])args);
		op_options_free(data->data->options);
		data->data->options = NULL;
	}

	nlop_finish_frw(data->data->stats);

	return;
}

static void nlop_set_opts(const operator_data_t* _data, const struct op_options_s* opts)
{
	auto data = CAST_DOWN(nlop_op_data_s, _data);

	op_options_free(data->data->options);
	data->data->options = op_options_clone(opts);

	if (NULL != data->set_opts)
		data->set_opts(data->data, opts);
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

	nlop_start_der(data->data->stats, data->o, data->i);

	data->deriv(data->data, data->o, data->i, dst, src);

	nlop_finish_der(data->data->stats, data->o, data->i);
}

static void lop_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	nlop_start_adj(data->data->stats, data->o, data->i);

	data->adjoint(data->data, data->o, data->i, dst, src);

	nlop_finish_adj(data->data->stats, data->o, data->i);
}

static void lop_nrm_inv(const linop_data_t* _data, float lambda, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->norm_inv(data->data, data->o, data->i, lambda, dst, src);
}

static void lop_nrm(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->normal(data->data, data->o, data->i, dst, src);
}


static void lop_del(const linop_data_t* _data)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	xfree(data);
}

static const char* nlop_graph_default(nlop_data_t* _data, unsigned int N, unsigned int D[N], const char** arg_nodes[N], graph_t opts)
{
	UNUSED(opts);

	for (uint i = 0; i < N; i++) {

		D[i] = 1;
		PTR_ALLOC(const char*[D[i]], nodes_i);
		arg_nodes[i] = *PTR_PASS(nodes_i);

		(arg_nodes[i])[0] = ptr_printf("nlop_%p", _data);
	}

	auto stats = _data->stats;


	const char* tmp = ptr_printf("");

	if (opts.calls) {

		auto tmp2 = ptr_printf("%s\\n frw(%lu) %gs", tmp, stats->frw_calls, stats->frw_time);
		xfree(tmp);
		tmp = tmp2;

		for (uint i = 0; i < stats->II; i++)
			for (uint o = 0; o < stats->OO; o++) {
				
				tmp2 = ptr_printf("%s\\n (%u, %u): der(%lu) %gs; adj(%lu) %gs", tmp,  o, i,
						stats->der_calls[o + stats->OO * i], stats->der_time[o + stats->OO * i],
						stats->adj_calls[o + stats->OO * i], stats->adj_time[o + stats->OO * i]
					);
				xfree(tmp);
				tmp = tmp2;
			}
	}

	const char* result = NULL;

	if (opts.time)
		result = ptr_printf("nlop_%p [label=\"nlop\\n%s\\ntime: %fs%s\"];\n", _data, _data->TYPEID->name, stats->run_time, tmp);
	else
		result = ptr_printf("nlop_%p [label=\"nlop\\n%s%s\"];\n", _data, _data->TYPEID->name, tmp);

	xfree(tmp);

	return result;
}

static const char* operator_graph_nlop(const operator_data_t* _data, unsigned int N, unsigned int D[N], const char** arg_nodes[N], graph_t opts)
{
	auto data = CAST_DOWN(nlop_op_data_s, _data);

	if (NULL == data->get_graph)
		return nlop_graph_default(data->data, N, D, arg_nodes, opts);
	return data->get_graph(data->data, N, D, arg_nodes, opts);
}

static void der_not_implemented(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(dst);
	UNUSED(src);

	error("Derivative o=%d, i=%d of %s is not implemented!\n", o, i, _data->TYPEID->name);
}

static void adj_not_implemented(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(dst);
	UNUSED(src);

	error("Adjoint derivative o=%d, i=%d of %s is not implemented!\n", o, i, _data->TYPEID->name);
}


struct nlop_s* nlop_generic_with_props_create2(	int OO, int ON, const long odims[OO][ON], const long ostr[OO][ON], int II, int IN, const long idims[II][IN], const long istr[II][IN],
						nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO],
						nlop_del_fun_t del,
						nlop_set_opts_t set_opts, operator_property_flags_t props[II][OO],
						nlop_graph_t get_graph)
{
	PTR_ALLOC(struct nlop_s, n);

	PTR_ALLOC(struct nlop_op_data_s, d);
	SET_TYPEID(nlop_op_data_s, d);

	data->options = NULL;
	data->stats = stats_create(OO, II);

	d->data = data;
	d->forward1 = NULL;
	d->forward = forward;
	d->set_opts = set_opts;
	d->get_graph = get_graph;


	operator_property_flags_t tmp_props[II + OO][II + OO];
	for (int i = 0; i < II + OO; i++)
		for (int j = 0; j < II + OO; j++)
			tmp_props[i][j] = 0;
	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			if (NULL == props)
				continue;

			props[i][o] = MD_SET(props[i][o], OP_PROP_ATOMIC);

			if (0 != (tmp_props[i][o] & (~nlops_props_understood)))
				error("Property passed to nlop which is not understood\n");

			tmp_props[o][i+OO] = props[i][o];
			tmp_props[i+OO][o] = props[i][o];
		}

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
			d2->deriv = (NULL != deriv) ? ((NULL != deriv[i][o]) ? deriv[i][o] : der_not_implemented) : der_not_implemented;
			d2->adjoint = (NULL != adjoint) ? ((NULL != adjoint[i][o]) ? adjoint[i][o] : adj_not_implemented) : adj_not_implemented;
			d2->normal = (NULL != normal) ? normal[i][o] : NULL;
			d2->norm_inv = (NULL != norm_inv) ? norm_inv[i][o] : NULL;

			d2->o = o;
			d2->i = i;

			shared_ptr_copy(&d2->sptr, &d->sptr);
			d2->sptr.del = sptr_linop_del;

			operator_property_flags_t der_props = MD_BIT(OP_PROP_ATOMIC);
			if ((NULL != props ) && (MD_IS_SET(props[i][o], OP_PROP_HOLOMORPHIC)))
				der_props = MD_SET(der_props, OP_PROP_C_LIN);

			(*der)[i][o] = linop_with_props_create2(ON, odims[o], ostr[o], IN, idims[i], istr[i],
						     CAST_UP(PTR_PASS(d2)), lop_der, lop_adj,  (NULL != normal) ? lop_nrm : NULL, (NULL != norm_inv) ? lop_nrm_inv : NULL, lop_del, der_props);
		}
	}

	bool io_flags[OO + II];
	for (int i = 0; i < OO + II; i++)
		io_flags[i] = i < OO;

	n->op = operator_generic_with_props_create2(OO + II, io_flags, D, dims, strs, CAST_UP(PTR_PASS(d)), op_fun, op_del, nlop_set_opts, op_property_create(OO + II, io_flags, tmp_props), operator_graph_nlop);


	return PTR_PASS(n);
}

struct nlop_s* nlop_generic_create2(int OO, int ON, const long odims[OO][ON], const long ostr[OO][ON], int II, int IN, const long idims[II][IN], const long istr[II][IN],
	nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del)
{
	return nlop_generic_with_props_create2(OO, ON, odims, ostr, II, IN, idims, istr,
						data, forward, deriv, adjoint, normal, norm_inv,
						del, NULL, NULL, NULL);
}

struct nlop_s* nlop_generic_create(int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
	nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del)
{
	long istrs[II][IN];
	for (int i = 0; i < II; i++)
		md_calc_strides(IN, istrs[i], idims[i], CFL_SIZE);
	long ostrs[OO][ON];
	for (int o = 0; o < OO; o++)
		md_calc_strides(ON, ostrs[o], odims[o], CFL_SIZE);

	return nlop_generic_create2(OO, ON, odims, ostrs, II, IN, idims, istrs, data, forward, deriv, adjoint, normal, norm_inv, del);
}

struct nlop_s* nlop_generic_with_props_create(	int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
						nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del,
						nlop_set_opts_t set_opts, operator_property_flags_t io_prop[II][OO],
						nlop_graph_t get_graph)
{
	long istrs[II][IN];
	for (int i = 0; i < II; i++)
		md_calc_strides(IN, istrs[i], idims[i], CFL_SIZE);
	long ostrs[OO][ON];
	for (int o = 0; o < OO; o++)
		md_calc_strides(ON, ostrs[o], odims[o], CFL_SIZE);

	return nlop_generic_with_props_create2(OO, ON, odims, ostrs, II, IN, idims, istrs, data, forward, deriv, adjoint, normal, norm_inv, del, set_opts, io_prop, get_graph);
}

struct nlop_s* nlop_create2(unsigned int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)],
				unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
{
	struct nlop_s* op = nlop_generic_create2(1, ON, (const long(*)[])&odims[0], (const long(*)[])&ostrs[0], 1, IN, (const long(*)[])&idims[0], (const long(*)[])&istrs[0], data, NULL,
					(nlop_der_fun_t[1][1]){ { deriv } }, (nlop_der_fun_t[1][1]){ { adjoint } }, (NULL != normal) ? (nlop_der_fun_t[1][1]){ { normal } } : NULL, (NULL != norm_inv) ? (nlop_p_fun_t[1][1]){ { norm_inv } } : NULL, del);

	auto data2 = CAST_DOWN(nlop_op_data_s, operator_get_data(op->op));

	data2->forward1 = forward;

	return op;
}

struct nlop_s* nlop_create(unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
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


const struct nlop_s* nlop_clone(const struct nlop_s* op)
{
	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	n->op = operator_ref(op->op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*nder)[i][o] = linop_clone((*der)[i][o]);

	n->derivative = &(*PTR_PASS(nder))[0][0];
	return PTR_PASS(n);
}



const struct nlop_s* nlop_loop(int D, const long dims[D], const struct nlop_s* op)
{
#if 1
	UNUSED(dims);
	UNUSED(op);
	assert(0);
#else
	/* ok, this does not work, we need to store the input for the
	 * forward operator and call it when looping over derivative
	 * so that each point is set to the right position */

	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	n->op = operator_loop(D, dims, op->op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*nder)[i][o] = linop_loop(D, dims, (*der)[i][o]);

	n->derivative = &(*PTR_PASS(nder))[0][0];
	return PTR_PASS(n);
#endif
}



nlop_data_t* nlop_get_data(struct nlop_s* op)
{
	auto data2 = CAST_MAYBE(nlop_op_data_s, operator_get_data(op->op));

	if (NULL == data2)
		return NULL;
#if 1
	auto data3 = CAST_DOWN(nlop_linop_data_s, linop_get_data(op->derivative[0]));
	assert(data3->data == data2->data);
#endif
	return data2->data;
}

void nlop_apply(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	operator_apply(op->op, ON, odims, dst, IN, idims, src);
}

void nlop_derivative(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	linop_forward(nlop_get_derivative(op, 0, 0), ON, odims, dst, IN, idims, src);
}

void nlop_adjoint(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	linop_adjoint(nlop_get_derivative(op, 0, 0), ON, odims, dst, IN, idims, src);
}



void nlop_generic_apply_unchecked(const struct nlop_s* op, int N, void* args[N])
{
	operator_generic_apply_unchecked(op->op, N, args);
}

void nlop_generic_apply_with_opts_unchecked(const struct nlop_s* op, int N, void* args[N],  const struct op_options_s* opts)
{
	operator_generic_apply_with_opts_unchecked(op->op, N, args, opts);
}

void nlop_generic_apply_select_derivative_unchecked(const struct nlop_s* op, int N, void* args[N], unsigned long out_der_flag, unsigned long in_der_flag)
{
	unsigned int II = nlop_get_nr_in_args(op);
	unsigned int OO = nlop_get_nr_out_args(op);

	assert(II <= 8 *sizeof(out_der_flag));
	assert(OO <= 8 *sizeof(in_der_flag));

	bool out_der_arr[OO];
	for(uint i = 0; i < OO; i++)
		out_der_arr[i] = MD_IS_SET(out_der_flag, i);

	bool in_der_arr[II];
	for(uint i = 0; i < II; i++)
		in_der_arr[i] = MD_IS_SET(in_der_flag, i);

	auto opts =  op_options_select_der_create(OO, II, out_der_arr, in_der_arr);
	nlop_generic_apply_with_opts_unchecked(op, N, args, opts);
	op_options_free(opts);
}

void nlop_clear_derivative(const struct nlop_s* op)
{
	auto opt = op_options_clear_der_create(nlop_get_nr_out_args(op), nlop_get_nr_in_args(op));
	operator_set_options(op->op, opt);
	op_options_free(opt);
}

const struct linop_s* nlop_get_derivative(const struct nlop_s* op, int o, int i)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	assert((i < II) && (o < OO));

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	return (*der)[i][o];
}

const struct iovec_s* nlop_generic_domain(const struct nlop_s* op, int i)
{
	return operator_arg_in_domain(op->op, (unsigned int)i);
}

const struct iovec_s* nlop_generic_codomain(const struct nlop_s* op, int o)
{
	return operator_arg_out_codomain(op->op, (unsigned int)o);
}



const struct iovec_s* nlop_domain(const struct nlop_s* op)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	return nlop_generic_domain(op, 0);
}

const struct iovec_s* nlop_codomain(const struct nlop_s* op)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	return nlop_generic_codomain(op, 0);
}


struct flatten_s {

	INTERFACE(nlop_data_t);

	size_t* off;
	const struct nlop_s* op;
};

DEF_TYPEID(flatten_s);

static void flatten_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(flatten_s, _data);

	int OO = nlop_get_nr_out_args(data->op);
	int II = nlop_get_nr_in_args(data->op);

	void* args[OO + II];

	for (int o = 0; o < OO; o++)
		args[o] = (void*)dst + data->off[o];

	for (int i = 0; i < II; i++)
		args[OO + i] = (void*)src + data->off[OO + i];


	nlop_generic_apply_unchecked(data->op, OO + II, args);
}

static void flatten_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	auto data = CAST_DOWN(flatten_s, _data);

	int OO = nlop_get_nr_out_args(data->op);
	int II = nlop_get_nr_in_args(data->op);


	for (int o = 0; o < OO; o++) {

		auto iov = linop_codomain(nlop_get_derivative(data->op, o, 0));

		complex float* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, src);

		md_clear(iov->N, iov->dims, (void*)dst + data->off[o], iov->size);

		for (int i = 0; i < II; i++) {

			const struct linop_s* der = nlop_get_derivative(data->op, o, i);

			auto iov2 = linop_domain(der);

			linop_forward(der,
				iov->N, iov->dims, tmp,
				iov2->N, iov2->dims,
				(void*)src + data->off[OO + i]);

			md_zadd(iov->N, iov->dims, (void*)dst + data->off[o], (void*)dst + data->off[o], tmp);
		}

		md_free(tmp);
	}
}

static void flatten_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	auto data = CAST_DOWN(flatten_s, _data);

	int OO = nlop_get_nr_out_args(data->op);
	int II = nlop_get_nr_in_args(data->op);

	for (int i = 0; i < II; i++) {

		auto iov = linop_domain(nlop_get_derivative(data->op, 0, i));

		complex float* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, src);

		md_clear(iov->N, iov->dims, (void*)dst + data->off[OO + i], iov->size);

		for (int o = 0; o < OO; o++) {	// FIXME

			const struct linop_s* der = nlop_get_derivative(data->op, o, i);

			linop_adjoint_unchecked(der,
				tmp,
				(void*)src + data->off[o]);

			md_zadd(iov->N, iov->dims, (void*)dst + data->off[OO + i], (void*)dst + data->off[OO + i], tmp);
		}

		md_free(tmp);
	}
}

static void flatten_del(const nlop_data_t* _data)
{
	auto data = CAST_DOWN(flatten_s, _data);

	nlop_free(data->op);
	xfree(data->off);

	xfree(data);
}




struct nlop_s* nlop_flatten(const struct nlop_s* op)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	long odims[1] = { 0 };
	long ostrs[] = { CFL_SIZE };
	size_t olast = 0;

	PTR_ALLOC(size_t[OO + II], offs);

	for (int o = 0; o < OO; o++) {

		auto iov = nlop_generic_codomain(op, o);

		assert(CFL_SIZE == iov->size);
		assert(iov->N == md_calc_blockdim(iov->N, iov->dims, iov->strs, iov->size));

		odims[0] += md_calc_size(iov->N, iov->dims);
		(*offs)[o] = olast;
		olast = odims[0] * CFL_SIZE;
	}


	long idims[1] = { 0 };
	long istrs[1] = { CFL_SIZE };
	size_t ilast = 0;

	for (int i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(op, i);

		assert(CFL_SIZE == iov->size);
		assert(iov->N == md_calc_blockdim(iov->N, iov->dims, iov->strs, iov->size));

		idims[0] += md_calc_size(iov->N, iov->dims);
		(*offs)[OO + i] = ilast;
		ilast = idims[0] * CFL_SIZE;
	}

	PTR_ALLOC(struct flatten_s, data);
	SET_TYPEID(flatten_s, data);

	data->op = nlop_clone(op);
	data->off = *PTR_PASS(offs);

	return nlop_create2(1, odims, ostrs, 1, idims, istrs, CAST_UP(PTR_PASS(data)), flatten_fun, flatten_der, flatten_adj, NULL, NULL, flatten_del);
}


const struct nlop_s* nlop_flatten_get_op(struct nlop_s* op)
{
	auto data = CAST_MAYBE(flatten_s, nlop_get_data(op));

	return (NULL == data) ? NULL : data->op;
}

const struct nlop_s* nlop_reshape_in(const struct nlop_s* op, int i, int NI, const long idims[NI])
 {
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	int oNI = nlop_generic_domain(op, i)->N;
	const long* oidims = nlop_generic_domain(op, i)->dims;

	debug_printf(DP_DEBUG4, "nlop_reshape_in %d:\t", i);
	debug_print_dims(DP_DEBUG4, oNI, oidims);
	debug_printf(DP_DEBUG4, "to:\t\t\t");
	debug_print_dims(DP_DEBUG4, NI, idims);

	PTR_ALLOC(struct nlop_s, n);
	n->op = operator_reshape(op->op, OO + i, NI, idims);

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
	n->derivative = &(*der)[0][0];

	//derivatives are not put into an operator-reshape-container but linked with an reshaping copy operator
	//	-> operators can be compared in operator chain to only evaluate them once (for parallel application)
	PTR_ALLOC(struct linop_s, resh_t);
	resh_t->forward = operator_reshape_create(oNI, oidims, NI, idims);
	resh_t->adjoint = operator_reshape_create(NI, idims, oNI, oidims);
	resh_t->normal = operator_reshape_create(NI, idims, NI, idims);
	resh_t->norm_inv = NULL;
	auto resh = PTR_PASS(resh_t);

	for (int ii = 0; ii < II; ii++)
		for (int io = 0; io < OO; io++)
			(*der)[ii][io] = (ii == i) ? linop_chain(resh, nlop_get_derivative(op, io, ii)) : linop_clone(nlop_get_derivative(op, io, ii));

	linop_free(resh);

	return PTR_PASS(n);
}

const struct nlop_s* nlop_reshape_out(const struct nlop_s* op, int o, int NO, const long odims[NO])
 {
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	int oNO = nlop_generic_codomain(op, o)->N;
	const long* oodims = nlop_generic_codomain(op, o)->dims;

	debug_printf(DP_DEBUG4, "nlop_reshape_out %d:\t", o);
	debug_print_dims(DP_DEBUG4, oNO, oodims);
	debug_printf(DP_DEBUG4, "to:\t\t\t");
	debug_print_dims(DP_DEBUG4, NO, odims);

	PTR_ALLOC(struct nlop_s, n);
	n->op = operator_reshape(op->op, o, NO, odims);

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
	n->derivative = &(*der)[0][0];

	//derivatives are not put into an operator-reshape-container but linked with an reshaping copy operator
	//	-> operators can be compared in operator chain to only evaluate them once (for parallel application)
	PTR_ALLOC(struct linop_s, resh_t);
	resh_t->forward = operator_reshape_create(NO, odims, oNO, oodims);
	resh_t->adjoint = operator_reshape_create(oNO, oodims, NO, odims);
	resh_t->normal= operator_reshape_create(oNO, oodims, oNO, oodims);
	resh_t->norm_inv = NULL;

	auto resh = PTR_PASS(resh_t);

	for (int ii = 0; ii < II; ii++)
		for (int io = 0; io < OO; io++)
			(*der)[ii][io] = (io == o) ? linop_chain(nlop_get_derivative(op, io, ii), resh) : linop_clone(nlop_get_derivative(op, io, ii));

	linop_free(resh);

	return PTR_PASS(n);
}


const struct nlop_s* nlop_reshape_in_F(const struct nlop_s* op, int i, int NI, const long idims[NI])
{
	auto result = nlop_reshape_in(op, i, NI,idims);
	nlop_free(op);
	return result;
}

const struct nlop_s* nlop_reshape_out_F(const struct nlop_s* op, int o, int NO, const long odims[NO])
{
	auto result = nlop_reshape_out(op, o, NO,odims);
	nlop_free(op);
	return result;
}

const struct nlop_s* nlop_append_singleton_dim_in_F(const struct nlop_s* op, int i)
{
	long N = nlop_generic_domain(op, i)->N;
	long dims[N + 1];
	md_copy_dims(N, dims, nlop_generic_domain(op, i)->dims);
	dims[N] = 1;

	return nlop_reshape_in_F(op, i, N + 1, dims);
}

const struct nlop_s* nlop_append_singleton_dim_out_F(const struct nlop_s* op, int o)
{
	long N = nlop_generic_codomain(op, o)->N;
	long dims[N + 1];
	md_copy_dims(N, dims, nlop_generic_codomain(op, o)->dims);
	dims[N] = 1;

	return nlop_reshape_out_F(op, o, N + 1, dims);
}

const struct nlop_s* nlop_no_der(const struct nlop_s* op, int o, int i)
{
	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	n->op = operator_ref(op->op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int ii = 0; i < II; i++)
		for (int oo = 0; o < OO; o++) {

			auto cod = linop_codomain((*der)[i][o]);
			auto dom = linop_domain((*der)[i][o]);

			(*nder)[ii][oo] = ((i == ii) && (o == oo)) ? linop_null_create(cod->N, cod->dims, dom->N, dom->dims) : linop_clone((*der)[i][o]);
		}
			

	n->derivative = &(*PTR_PASS(nder))[0][0];
	return PTR_PASS(n);
}

const struct nlop_s* nlop_no_der_F(const struct nlop_s* op, int o, int i)
{
	auto result = nlop_no_der(op, o, i);
	nlop_free(op);
	return result;
}

void nlop_debug(enum debug_levels dl, const struct nlop_s* x)
{
	int II = nlop_get_nr_in_args(x);

	debug_printf(dl, "NLOP\ninputs: %d\n", II);

	for (int i = 0; i < II; i++) {

		auto io = nlop_generic_domain(x, i);
		debug_print_dims(dl, io->N, io->dims);
	}

	int OO = nlop_get_nr_out_args(x);

	debug_printf(dl, "outputs: %d\n", OO);

	for (int o = 0; o < OO; o++) {

		auto io = nlop_generic_codomain(x, o);
		debug_print_dims(dl, io->N, io->dims);
	}
}

void nlop_export_graph(const char* filename, const struct nlop_s* op, graph_t opts)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	unsigned int D[II + OO];
	const char** arg_nodes[II + OO];

	const char* str = operator_get_graph_string(op->op, II + OO, D, arg_nodes, opts);

	FILE *fp;
	fp = fopen(filename, "w+");

	assert(0 != fp);

	fprintf(fp, "digraph { \n");
	fprintf(fp, "{ rank = source\n");
	for (int i = 0; i < II; i++)
		fprintf(fp, "Input_%d;\n", i);
	if(0 < II)
		fprintf(fp, "edge[ style=invis];\nInput_0");
	for (int i = 1; i < II; i++)
		fprintf(fp, " -> Input_%d", i);

	fprintf(fp, ";\n}\n{ rank = sink\n");
	for (int i = 0; i < OO; i++)
		fprintf(fp, "Output_%d;\n", i);
	if(0 < OO)
		fprintf(fp, "edge[ style=invis];\nOutput_0");
	for (int i = 1; i < OO; i++)
		fprintf(fp, " -> Output_%d", i);
	fprintf(fp, "}\n%s", str);



	for (int o = 0; o < OO; o++) {

		fprintf(fp, "%s -> Output_%d;\n", (arg_nodes[o])[0], o);
		xfree((arg_nodes[o])[0]);
		xfree((arg_nodes[o]));
		assert(1 == D[o]);
	}

	for (int i = 0; i < II; i++) {

		for (int j = 0; j < (int)D[OO + i]; j++) {

			fprintf(fp, "Input_%d -> %s;\n", i, (arg_nodes[OO + i])[j]);
			xfree((arg_nodes[OO + i])[j]);
		}
		xfree((arg_nodes[OO + i]));
	}
	fprintf(fp, "} \n");

	fclose(fp);
}
