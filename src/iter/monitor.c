/* Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016,2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>

#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"

#include "iter/vec.h"

#include "monitor.h"

#include <complex.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "num/multind.h"

void iter_monitor(struct iter_monitor_s* monitor, const struct vec_iter_s* ops, const float* x)
{
	if ((NULL != monitor) && (NULL != monitor->fun))
		monitor->fun(monitor, ops, x);
}

void iter_history(struct iter_monitor_s* monitor, const struct iter_history_s* hist)
{
	if ((NULL != monitor) && (NULL != monitor->record))
		monitor->record(monitor, hist);
}



struct monitor_default_s {

	INTERFACE(iter_monitor_t);

	long N;
	const float* image_truth;
	double it_norm;

	void* data;
	float (*objective)(const void* data, const float* x);
};

static DEF_TYPEID(monitor_default_s);


static void monitor_default_fun(struct iter_monitor_s* _data, const struct vec_iter_s* vops, const float* x)
{
	static unsigned int iter = 0;
	auto data = CAST_DOWN(monitor_default_s, _data);

	double err = -1.;
	double obj = -1.;

	long N = data->N;

	if (NULL != data->image_truth) {

		if (-1. == data->it_norm)
			data->it_norm = vops->norm(N, data->image_truth);

		float* x_err = vops->allocate(N);

		vops->sub(N, x_err, data->image_truth, x);
		err = vops->norm(N, x_err) / data->it_norm;

		vops->del(x_err);
	}

	if (NULL != data->objective)
		obj = data->objective(data->data, x);

	debug_printf(DP_INFO, "[Iter %04d] Objective: %f, Error: %f\n", ++iter, obj, err);

	data->INTERFACE.obj = obj;
	data->INTERFACE.err = err;
}

struct iter_monitor_s* create_monitor(long N, const float* image_truth, void* data, float (*objective)(const void* data, const float* x))
{
	PTR_ALLOC(struct monitor_default_s, monitor);
	SET_TYPEID(monitor_default_s, monitor);

	monitor->N = N;
	monitor->image_truth = image_truth;
	monitor->it_norm = -1.;
	monitor->data = data;
	monitor->objective = objective;

	monitor->INTERFACE.fun = monitor_default_fun;
	monitor->INTERFACE.record = NULL;
	monitor->INTERFACE.obj = -1.;
	monitor->INTERFACE.err = -1.;

	return CAST_UP(PTR_PASS(monitor));
}


void monitor_iter6(struct monitor_iter6_s* monitor, long epoch, long batch, long num_batches, float objective, long NI, const float* x[NI], char* post_string)
{
	if ((NULL != monitor) && (NULL != monitor->fun))
		monitor->fun(monitor, epoch, batch, num_batches, objective, NI, x, post_string);
}



struct monitor_recorder_s {

	INTERFACE(iter_monitor_t);

	long N;
	long *dims;
	char *name;
	char *out_name;
	int strlen;
	int max_decimals;

	void *data;
	complex float *(*process)(void *data, const float *x);
	_Bool (*select)(const unsigned long iter, const float *x, void *data);
};

static DEF_TYPEID(monitor_recorder_s);


static void monitor_recorder_fun(struct iter_monitor_s *_data, const struct vec_iter_s *vops, const float *x)
{
	static unsigned int iter = 0;
	UNUSED(vops);

	iter++;

	auto data = CAST_DOWN(monitor_recorder_s, _data);

	if (data->select(iter, x, data->data)) {
		assert(pow(10, data->max_decimals) > iter);
		strcpy(data->out_name, data->name);
		sprintf(data->out_name + data->strlen, "_%d", iter);
		if (data->process == NULL)
			dump_cfl(data->out_name, data->N, data->dims, (complex float *)x);
		else
			dump_cfl(data->out_name, data->N, data->dims, data->process(data->data, x));
	}
}


static bool default_monitor_select(unsigned long iter, const float *x, void *data)
{
	UNUSED(x);
	UNUSED(data);
	return (0 == iter % 10);
}


struct iter_monitor_s *create_monitor_recorder(const long N, const long dims[N], const char *name, void *data, _Bool (*select)(unsigned long iter, const float *x, void *data), complex float *(*process)(void *data, const float *x))
{
	PTR_ALLOC(struct monitor_recorder_s, monitor);
	SET_TYPEID(monitor_recorder_s, monitor);

	monitor->max_decimals = 6;

	monitor->N = N;
	monitor->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, monitor->dims, dims);

	monitor->strlen = strlen(name);
	monitor->name = strdup(name);
	monitor->out_name = calloc(strlen(name) + monitor->max_decimals + 1, sizeof(char));

	monitor->data = data;
	monitor->process = process;
	monitor->select = NULL == select ? default_monitor_select : select;

	monitor->INTERFACE.fun = monitor_recorder_fun;

	return CAST_UP(PTR_PASS(monitor));
}
