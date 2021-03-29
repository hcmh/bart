#ifndef ITER6_H
#define ITER6_H

#include "misc/opts.h"
#include "iter/italgos.h"
#include "iter/iter_dump.h"
#include "iter/batch_gen.h"

extern struct opt_s iter6_opts[];
extern struct opt_s iter6_sgd_opts[];
extern struct opt_s iter6_adadelta_opts[];
extern struct opt_s iter6_adam_opts[];
extern struct opt_s iter6_ipalm_opts[];

extern const int N_iter6_opts;
extern const int N_iter6_sgd_opts;
extern const int N_iter6_adadelta_opts;
extern const int N_iter6_adam_opts;
extern const int N_iter6_ipalm_opts;

extern void iter6_copy_config_from_opts(struct iter6_conf_s* result);
extern struct iter6_conf_s* iter6_get_conf_from_opts(void);
extern struct iter6_adam_conf iter6_adam_conf_opts;

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
	unsigned long dump_flag;

	enum BATCH_GEN_TYPE batchgen_type;
	int batch_seed;

} iter6_conf;

extern struct iter6_conf_s iter6_conf_opts;

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

	_Bool reduce_momentum;
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

extern iter6_f iter6_adadelta;
extern iter6_f iter6_adam;
extern iter6_f iter6_sgd;
extern iter6_f iter6_sgd_like;

extern iter6_f iter6_iPALM;

extern iter6_f iter6_by_conf;

#endif