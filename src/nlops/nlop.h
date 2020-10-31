/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <stdbool.h>

#include "linops/linop.h"

#ifndef NLOP_H
#define NLOP_H

struct nlop_run_stats_s;
typedef struct nlop_data_s { TYPEID* TYPEID; const struct op_options_s* options; struct nlop_run_stats_s* stats; } nlop_data_t;

typedef void (*nlop_fun_t)(const nlop_data_t* _data, complex float* dst, const complex float* src);

typedef void (*nlop_p_fun_t)(const nlop_data_t* _data, unsigned int o, unsigned int i, float lambda, complex float* dst, const complex float* src);
typedef void (*nlop_der_fun_t)(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src);

typedef void (*nlop_del_fun_t)(const nlop_data_t* _data);

typedef void (*nlop_gen_fun_t)(const nlop_data_t* _data, int N, complex float* arg[N]);

typedef void (*nlop_set_opts_t)(const nlop_data_t* _data, const struct op_options_s* options);

typedef const char* (*nlop_graph_t)(const nlop_data_t* _data, unsigned int N, unsigned int D[N], const char** arg_nodes[N], graph_t opts);

struct operator_s;
struct linop_s;

struct nlop_s {

	const struct operator_s* op;
	const struct linop_s** derivative;
};

extern struct nlop_s* nlop_generic_with_props_create(	int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
							nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del,
							nlop_set_opts_t set_opts, operator_property_flags_t props[II][OO],
							nlop_graph_t get_graph);
extern struct nlop_s* nlop_generic_with_props_create2(	int OO, int NO, const long odims[OO][NO], const long ostr[OO][NO], int II, int IN, const long idims[II][IN], const long istr[II][IN],
							nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del,
							nlop_set_opts_t set_opts, operator_property_flags_t props[II][OO],
							nlop_graph_t get_graph);

extern struct nlop_s* nlop_generic_create(	int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
						nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del);
extern struct nlop_s* nlop_generic_create2(	int OO, int NO, const long odims[OO][NO], const long ostr[OO][NO], int II, int IN, const long idims[II][IN], const long istr[II][IN],
						nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del);

extern struct nlop_s* nlop_create(	unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], nlop_data_t* data,
					nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t);
extern struct nlop_s* nlop_create2(	unsigned int ON, const long odims[__VLA(ON)], const long ostr[__VLA(ON)],
					unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], nlop_data_t* data,
					nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t);


extern const struct nlop_s* nlop_clone(const struct nlop_s* op);
extern void nlop_free(const struct nlop_s* op);

extern nlop_data_t* nlop_get_data(struct nlop_s* op);

extern int nlop_get_nr_in_args(const struct nlop_s* op);
extern int nlop_get_nr_out_args(const struct nlop_s* op);


extern void nlop_apply(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src);
extern void nlop_derivative(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src);
extern void nlop_adjoint(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src);

extern void nlop_generic_apply_unchecked(const struct nlop_s* op, int N, void* args[N]);
extern void nlop_generic_apply_with_opts_unchecked(const struct nlop_s* op, int N, void* args[N], const struct op_options_s* opts);
extern void nlop_generic_apply_select_derivative_unchecked(const struct nlop_s* op, int N, void* args[N], unsigned long out_der_flag, unsigned long in_der_flag);
extern void nlop_clear_derivative(const struct nlop_s* op);

extern const struct linop_s* nlop_get_derivative(const struct nlop_s* op, int o, int i);

extern const struct iovec_s* nlop_generic_domain(const struct nlop_s* op, int i);
extern const struct iovec_s* nlop_generic_codomain(const struct nlop_s* op, int o);



struct iovec_s;
extern const struct iovec_s* nlop_domain(const struct nlop_s* op);
extern const struct iovec_s* nlop_codomain(const struct nlop_s* op);


extern struct nlop_s* nlop_flatten(const struct nlop_s* op);
extern const struct nlop_s* nlop_flatten_get_op(struct nlop_s* op);

enum debug_levels;
extern void nlop_debug(enum debug_levels dl, const struct nlop_s* x);

extern const struct nlop_s* nlop_reshape_out(const struct nlop_s* op, int o, int NO, const long odims[NO]);
extern const struct nlop_s* nlop_reshape_in(const struct nlop_s* op, int i, int NI, const long idims[NI]);
extern const struct nlop_s* nlop_reshape_out_F(const struct nlop_s* op, int o, int NO, const long odims[NO]);
extern const struct nlop_s* nlop_reshape_in_F(const struct nlop_s* op, int i, int NI, const long idims[NI]);

extern const struct nlop_s* nlop_append_singleton_dim_in_F(const struct nlop_s* op, int i);
extern const struct nlop_s* nlop_append_singleton_dim_out_F(const struct nlop_s* op, int o);

extern void nlop_export_graph(const char* filename, const struct nlop_s* op, graph_t opts);

#endif
