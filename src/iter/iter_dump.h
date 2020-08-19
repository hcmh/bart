
#ifndef __iter_dump_H
#define __iter_dump_H

#include <stddef.h>


struct typeid_s;
struct nlop_s;

struct iter_dump_s;
typedef void (*iter_dump_fun_multi_t)(const struct iter_dump_s* data, long epoch, long NI, const float* x[NI]);
typedef void (*iter_dump_fun_t)(const struct iter_dump_s* data, long epoch, const float* x);
typedef void (*iter_dump_free_t)(const struct iter_dump_s* data);
typedef struct iter_dump_s {

	const struct typeid_s* TYPEID;
	iter_dump_fun_multi_t fun_multi;
	iter_dump_fun_t fun;
	iter_dump_free_t free;
	const char* base_filename;

} iter_dump_t;

const struct iter_dump_s* iter_dump_multi_default_create(const char* base_filename, long save_mod, long NI, unsigned long save_flag, unsigned int D[NI], const long* dims[NI]);

inline void iter_dump_multi(const struct iter_dump_s* data, long epoch, long NI, const float* x[NI])
{
	if ((NULL != data) && (NULL != data->fun_multi))
		data->fun_multi(data, epoch, NI, x);
}
inline void iter_dump(const struct iter_dump_s* data, long epoch, const float* x)
{
	if ((NULL != data) && (NULL != data->fun))
		data->fun(data, epoch, x);
}

void iter_dump_free(const struct iter_dump_s* data);


#endif // __iter_dump_H
