#ifndef ITER6_H
#define ITER6_H
#include "italgos.h"
#include "iter/iter_dump.h"

struct iter_dump_s;
typedef struct iter6_conf_s {
	TYPEID* TYPEID;

	int epochs;

	float learning_rate;

	float clip_norm;
	float clip_val;

	float batchnorm_momentum;

	const char* history_filename;

	const struct iter_dump_s* dump;
	const char* dump_filename;
	long dump_mod;

} iter6_conf;

struct iter_op_s;

struct iter6_sgd_conf {

	INTERFACE(iter6_conf);

	float momentum;
};

struct iter6_adadelta_conf {

	INTERFACE(iter6_conf);

	float rho;
};

struct iter6_adam_conf {

	INTERFACE(iter6_conf);

	long reset_epoch;

	float epsilon;
	float beta1;
	float beta2;
};

struct iter6_iPALM_conf {

	INTERFACE(iter6_conf);

	float Lmin;
	float Lmax;
	float Lshrink;
	float Lincrease;

	float alpha;
	float beta;
	_Bool convex;

	_Bool trivial_stepsize;

	float* alpha_arr;
	float* beta_arr;
	_Bool* convex_arr;
};

extern const struct iter6_sgd_conf iter6_sgd_conf_defaults;
extern const struct iter6_adadelta_conf iter6_adadelta_conf_defaults;
extern const struct iter6_adam_conf iter6_adam_conf_defaults;
extern const struct iter6_iPALM_conf iter6_iPALM_conf_defaults;

struct iter3_conf_s;
struct iter_nlop_s;
struct nlop_s;
struct operator_p_s;
typedef void iter6_f(iter6_conf* _conf, const struct nlop_s* nlop, long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI], long NO, enum OUT_TYPE out_type[NO], int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor);
iter6_f iter6_adadelta;
iter6_f iter6_adam;
iter6_f iter6_sgd;

iter6_f iter6_iPALM;

#endif