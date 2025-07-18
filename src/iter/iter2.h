
#ifndef _ITER_ITER2_H
#define _ITER_ITER2_H

#include "misc/cppwrap.h"
#include "misc/types.h"

struct linop_s;
struct operator_s;
struct operator_p_s;

#ifndef ITER_OP_DATA_S
#define ITER_OP_DATA_S
typedef struct iter_op_data_s { TYPEID* TYPEID; } iter_op_data;
#endif

struct iter_op_op {

	iter_op_data super;
	const struct operator_s* op;
};


struct iter_op_p_op {

	iter_op_data super;
	const struct operator_p_s* op;
};


extern void operator_iter(iter_op_data* data, float* dst, const float* src);
extern void operator_p_iter(iter_op_data* data, float rho, float* dst, const float* src);


// the temporary copy is needed if used in loops
#define STRUCT_TMP_COPY(x) ({ __typeof(x) __foo = (x); __typeof(__foo)* __foo2 = alloca(sizeof(__foo)); *__foo2 = __foo; __foo2; })
#define OPERATOR2ITOP(op) (struct iter_op_s){ (NULL == op) ? NULL : operator_iter, CAST_UP(STRUCT_TMP_COPY(((struct iter_op_op){ { &TYPEID(iter_op_op) }, op }))) }
#define OPERATOR_P2ITOP(op) (struct iter_op_p_s){ (NULL == op) ? NULL : operator_p_iter, CAST_UP(STRUCT_TMP_COPY(((struct iter_op_p_op){ { &TYPEID(iter_op_p_op) }, op }))) }

#ifndef ITER_CONF_S
#define ITER_CONF_S
typedef struct iter_conf_s { TYPEID* TYPEID; float alpha; } iter_conf;
#endif

struct iter_monitor_s;

typedef void (italgo_fun2_f)(const iter_conf* conf,
		const struct operator_s* normaleq_op,
		int D,
		const struct operator_p_s* prox_ops[__VLA2(D)],
		const struct linop_s* ops[__VLA2(D)],
		const float* biases[__VLA2(D)],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor);

typedef italgo_fun2_f* italgo_fun2_t;

italgo_fun2_f iter2_conjgrad;
italgo_fun2_f iter2_ist;
italgo_fun2_f iter2_eulermaruyama;
italgo_fun2_f iter2_fista;
italgo_fun2_f iter2_chambolle_pock;
italgo_fun2_f iter2_admm;
italgo_fun2_f iter2_pocs;
italgo_fun2_f iter2_niht;


// use with iter_call_s from iter.h as _conf
italgo_fun2_f iter2_call_iter;


struct iter2_call_s {

	iter_conf super;

	italgo_fun2_t fun;
	iter_conf* _conf;
};



#include "misc/cppwrap.h"


#endif

