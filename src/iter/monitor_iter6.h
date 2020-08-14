
#ifndef __monitor_iter6_H
#define __monitor_iter6_H


struct monitor_iter6_s;
struct typeid_s;
struct nlop_s;
#include "iter/monitor.h"

extern void monitor_iter6_free(const struct monitor_iter6_s* monitor);

typedef struct monitor_iter6_value_data_s { TYPEID* TYPEID; } monitor_iter6_value_data_t;
typedef _Complex float (*monitor_iter6_value_fun_t)(const monitor_iter6_value_data_t* data, long NI, const float* args[NI]);
typedef _Bool (*monitor_iter6_value_eval_t)(const monitor_iter6_value_data_t* data, long epoch, long batch, long numbatches);
typedef void (*monitor_iter6_value_print_string_t)(const monitor_iter6_value_data_t* data, int N, char log_str[N]);
typedef void (*monitor_iter6_value_free_t)(const monitor_iter6_value_data_t* data);
struct monitor_value_s{

	struct monitor_iter6_value_data_s* data;

	monitor_iter6_value_fun_t fun;
	monitor_iter6_value_eval_t eval;
	monitor_iter6_value_print_string_t print;
	monitor_iter6_value_free_t free;
};

struct monitor_value_s monitor_iter6_nlop_create(const struct nlop_s* nlop, _Bool eval_each_batch, const char* print_name);

typedef struct monitor_iter6_s monitor_iter6_t;


extern struct monitor_iter6_s* create_monitor_iter6_progressbar_trivial(void);
extern struct monitor_iter6_s* create_monitor_iter6_progressbar_record(void);
extern struct monitor_iter6_s* create_monitor_iter6_progressbar_with_val_monitor(int num_val_monitors, struct monitor_value_s* val_monitors);


extern void monitor_iter6_dump_record(struct monitor_iter6_s* _monitor, const char* filename);

#endif // __monitor_iter6_H
