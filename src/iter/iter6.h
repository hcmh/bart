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
};

extern const struct iter6_sgd_conf iter6_sgd_conf_defaults;
extern const struct iter6_adadelta_conf iter6_adadelta_conf_defaults;

struct iter3_conf_s;
struct iter_nlop_s;
struct nlop_s;
struct operator_p_s;
typedef void iter6_f(iter6_conf* _conf, const struct nlop_s* nlop, long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI], long NO, enum OUT_TYPE out_type[NO], int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct iter6_monitor_s* monitor);
iter6_f iter6_adadelta;
