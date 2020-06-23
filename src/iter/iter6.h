#ifndef ITER6_H
#define ITER6_H

#include "italgos.h"


typedef struct iter6_conf_s { TYPEID* TYPEID; } iter6_conf;
struct iter_op_s;

struct iter6_sgd_conf {

	INTERFACE(iter6_conf);

	int epochs;
	float learning_rate;

	float clip_norm;
	float clip_val;

	float momentum;
};

struct iter6_adadelta_conf {

	INTERFACE(iter6_conf);

	int epochs;
	float learning_rate;

	float clip_norm;
	float clip_val;

	float rho;

	float batchnorm_mom;
};

struct iter6_iPALM_conf {

	INTERFACE(iter6_conf);

	float L;
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

	int epochs;
	int epoch_start;

	int save_modulo;
	char* save_path;
	char** save_name;
};

extern const struct iter6_sgd_conf iter6_sgd_conf_defaults;
extern const struct iter6_adadelta_conf iter6_adadelta_conf_defaults;
extern const struct iter6_iPALM_conf iter6_iPALM_conf_defaults;

struct iter3_conf_s;
struct iter_nlop_s;
struct nlop_s;
struct operator_p_s;

typedef void iter6_f(iter6_conf* _conf, const struct nlop_s* nlop, long NI, enum IN_TYPE in_type[NI], float* dst[NI], long NO, enum OUT_TYPE out_type[NO], int N_batch, int N_total);

iter6_f iter6_adadelta;


void iter6_iPALM(	iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[__VLA(NI)], float* dst[__VLA(NI)],
			long NO, enum OUT_TYPE out_type[__VLA(NO)],
			const struct operator_p_s* prox_ops[__VLA(NI)],
			const struct nlop_s* nlop_batch_gen);

#endif