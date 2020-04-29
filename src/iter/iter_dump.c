#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
#include <string.h>

#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mmio.h"


#include "num/flpmath.h"
#include "num/multind.h"

#include "iter_dump.h"

void iter_dump_free(const struct iter_dump_s* data)
{
	data->free(data);
}

void iter_dump_multi(const struct iter_dump_s* data, long epoch, long NI, const float* x[NI])
{
	if ((NULL != data) && (NULL != data->fun_multi))
		data->fun_multi(data, epoch, NI, x);
}

void iter_dump(const struct iter_dump_s* data, long epoch, const float* x)
{
	if ((NULL != data) && (NULL != data->fun))
		data->fun(data, epoch, x);
}

struct iter_dump_default_s {

	INTERFACE(iter_dump_t);

	unsigned long save_flag;

	unsigned int N;
	unsigned int* D;
	const long** dims;

	long save_mod;
};

static DEF_TYPEID(iter_dump_default_s);

static void iter_dump_default_fun(const struct iter_dump_s* _data, long epoch, long NI, const float* x[NI])
{
	auto data = CAST_DOWN(iter_dump_default_s, _data);

	if (0 != epoch % data->save_mod)
		return;

	const complex float* args[data->N];
	for (int i = 0, ip = 0; i < NI; i++)
		if (MD_IS_SET(data->save_flag, i))
			args[ip++] = (const complex float*)x[i];

	char file[strlen(data->INTERFACE.base_filename) + 21];
	sprintf(file, "%s_%ld", data->INTERFACE.base_filename, epoch);

	dump_multi_cfl(file, data->N, data->D, data->dims, args);
}

static void iter_dump_default_free(const struct iter_dump_s* _data)
{
	auto data = CAST_DOWN(iter_dump_default_s, _data);

	for (unsigned int i = 0; i < data->N; i++)
		xfree(data->dims[i]);

	xfree(data->dims);
	xfree(data->D);

	xfree(_data);
}

const struct iter_dump_s* iter_dump_multi_default_create(const char* base_filename, long save_mod, long NI, unsigned long save_flag, unsigned int D[NI], const long* dims[NI])
{
	PTR_ALLOC(struct iter_dump_default_s, result);
	SET_TYPEID(iter_dump_default_s, result);

	result->INTERFACE.fun_multi = iter_dump_default_fun;
	result->INTERFACE.fun = NULL;
	result->INTERFACE.free = iter_dump_default_free;
	result->INTERFACE.base_filename = base_filename;

	result->save_mod = save_mod;
	result->save_flag = save_flag;

	result->N = bitcount(save_flag);
	PTR_ALLOC(unsigned int[result->N], nD);
	PTR_ALLOC(const long*[result->N], ndims);

	unsigned int ip = 0;
	for(int i = 0; i < NI; i++) {

		if (MD_IS_SET(save_flag, i)) {

			(*nD)[ip] = D[i];

			PTR_ALLOC(long[D[i]], ndim);
			md_copy_dims(D[i], *ndim, dims[i]);
			(*ndims)[ip] = *PTR_PASS(ndim);

			ip++;
		}
	}

	assert(result->N == ip);
	result->D = *PTR_PASS(nD);
	result->dims = *PTR_PASS(ndims);

	return CAST_UP(PTR_PASS(result));
}