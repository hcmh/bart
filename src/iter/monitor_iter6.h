
#ifndef __ITER6_MONITOR_H
#define __ITER6_MONITOR_H


struct iter6_monitor_s;
struct typeid_s;
struct nlop_s;
#include "iter/italgos.h"

typedef void (*iter6_monitor_fun_t)(struct iter6_monitor_s* data, long epoch, long batch, float objective, long NI, const float* x[NI], char* post_string);

struct iter6_monitor_s {

	const struct typeid_s* TYPEID;
	iter6_monitor_fun_t fun;
};

typedef _Complex float (*iter6_monitor_valuefun_t)(long NI, const float* args[NI]);
struct iter6_monitor_value_s{

	iter6_monitor_valuefun_t fun;
	const char* print_name;
	_Bool eval_each_batch;
};

typedef struct iter6_monitor_s iter6_monitor_t;

void iter6_monitor(struct iter6_monitor_s* monitor, long epoch, long batch, float objective, long NI, const float* x[NI], char* post_string);


extern struct iter6_monitor_s* create_iter6_monitor_progressbar(int numbatches, _Bool average_obj);
extern struct iter6_monitor_s* create_iter6_monitor_progressbar_validloss(int num_epochs, int numbatches, _Bool average_obj, long NI, enum IN_TYPE in_type[NI], const struct nlop_s* valid_loss, _Bool valloss_for_each_batch);

extern void iter6_monitor_dump_record(struct iter6_monitor_s* _monitor, const char* filename);
extern struct iter6_monitor_s* create_iter6_monitor_progressbar_value_monitors(int numepochs, int numbatches, _Bool average_obj, int num_val_monitors, struct iter6_monitor_value_s val_monitors[num_val_monitors]);

#endif // __ITER6_MONITOR_H
