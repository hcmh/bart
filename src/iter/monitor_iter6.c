#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
#include <string.h>

#include "iter/italgos.h"
#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#include "iter/vec.h"

#include "nlops/nlop.h"

#include "monitor_iter6.h"
#include "nn/layers.h"
#include "num/flpmath.h"
#include "num/multind.h"




struct monitor_iter6_default_s {

	INTERFACE(monitor_iter6_t);

	bool print_time;
	bool print_progress;
	bool print_overwrite;
	bool print_average_obj;

	float average_obj;

	double start_time;

	int num_val_monitors;
	const struct monitor_value_s* val_monitors;

	bool use_record;
	long epochs_written;
	long epochs_created;
	long num_batches;
	long record_dim;
	complex float* record;

};

static DEF_TYPEID(monitor_iter6_default_s);

static void print_progress_bar(int length, char progress[length + 5], int done, int total)
{
	progress[0] = ' ';
	progress[1] = '[';
	for (int i = 0; i < length; i++)
		if ((float)i <= (float)(done * length) / (float)(total))
			progress[i + 2] = '=';
		else
			progress[i + 2] = ' ';
	progress[length + 2] = ']';
	progress[length + 3] = ';';
	progress[length + 4] = '\0';
}
static void print_time_string(int length, char str_time[length], double time, double est_time)
{
	sprintf(str_time,
		" time: %d:%02d:%02d/%d:%02d:%02d;",
		(int)time / 3600, ((int)time %3600)/60, ((int)time % 3600) % 60,
		(int)est_time / 3600, ((int)est_time %3600)/60, ((int)est_time % 3600) % 60);
}

static void create_record(struct monitor_iter6_default_s* monitor, long epoch, long num_batches) {

	if (!monitor->use_record)
		return;

	monitor->num_batches = num_batches;

	if (epoch >= monitor->epochs_created) {

		long new_epochs_created = epoch + 100;
		long new_record_dims[3] = {new_epochs_created, num_batches, monitor->record_dim};
		complex float* new_record = md_alloc(3, new_record_dims, CFL_SIZE);
		md_clear(3, new_record_dims, new_record, CFL_SIZE);

		if (NULL != monitor->record) {

			long old_record_dims[3] = {monitor->epochs_created, num_batches, monitor->record_dim};
			md_copy2(3, old_record_dims, MD_STRIDES(3, new_record_dims, CFL_SIZE), new_record, MD_STRIDES(3, old_record_dims, CFL_SIZE), monitor->record, CFL_SIZE);
			md_free(monitor->record);
		}
		monitor->record = new_record;
		monitor->epochs_created = new_epochs_created;
	}
	monitor->epochs_written = epoch + 1;
}

void monitor_iter6_dump_record(struct monitor_iter6_s* _monitor, const char* filename)
{
	auto monitor = CAST_DOWN(monitor_iter6_default_s, _monitor);

	if (NULL != monitor->record) {

		long rdims_write[3] = {monitor->epochs_written, monitor->num_batches, monitor->record_dim};
		long rdims_read[3] = {monitor->epochs_created, monitor->num_batches, monitor->record_dim};
		complex float* file = create_cfl(filename, 3, rdims_write);
		md_copy2(3, rdims_write, MD_STRIDES(3, rdims_write, CFL_SIZE), file, MD_STRIDES(3, rdims_read, CFL_SIZE), monitor->record, CFL_SIZE);
		unmap_cfl(3, rdims_write, file);

	} else
		error("Record not available!\n");
}

static void compute_val_monitors(struct monitor_iter6_default_s* monitor, int N, char str[N], long epoch, long batch, long num_batches, long NI, const float* x[NI])
{
	create_record(monitor, epoch, num_batches);
	str[0] = '\0';

	for (int i = 0; i < monitor->num_val_monitors; i++) {

		long rstrs[3];
		md_calc_strides(3, rstrs, MD_DIMS(monitor->epochs_created, num_batches, monitor->record_dim), CFL_SIZE);
		long rpos[3] = {epoch, batch, 2 + 2 * i};

		if (!monitor->val_monitors[i].eval(monitor->val_monitors[i].data, epoch, batch, num_batches))
			continue;

		if (monitor->use_record)
			MD_ACCESS(3, rstrs, rpos, monitor->record) = 1.;
		rpos[2]++;

		complex float val = monitor->val_monitors[i].fun(monitor->val_monitors[i].data, NI, x);
		if (monitor->use_record)
			MD_ACCESS(3, rstrs, rpos, monitor->record) = val;

		monitor->val_monitors[i].print(monitor->val_monitors[i].data, N, str);
		long len = strlen(str);
		str += len;
		N -= len;
	}
}



static void monitor6_default_fun(struct monitor_iter6_s* _monitor, long epoch, long batch, long numbatches, float objective, long NI, const float* x[NI], char* post_string)
{
	auto monitor = CAST_DOWN(monitor_iter6_default_s, _monitor);

	bool print_progress = monitor->print_progress;
	bool print_time = monitor->print_time;
	bool print_loss = true;
	bool print_overwrite = true;

	char str_progress[15];
	if (print_progress)
		print_progress_bar(10, str_progress, batch, numbatches);
	else
		str_progress[0] = '\0';

	double time = timestamp() - monitor->start_time;
	double est_time = time + (double)(numbatches - batch - 1) * time / (double)(batch + 1);
	char str_time[50];
	if (print_time)
		print_time_string(30, str_time, time, est_time);
	else
		str_time[0] = '\0';

	char str_loss[30];
	monitor->average_obj = ((batch - 1) * monitor->average_obj + objective) / batch;
	if (print_loss)
		sprintf(str_loss, " loss: %f;", monitor->print_average_obj ? monitor->average_obj: objective);
	else
		str_loss[0] = '\0';

	char str_val_monitor[100 * monitor->num_val_monitors];
	str_val_monitor[0] = '\0';
	compute_val_monitors(monitor, 100 * monitor->num_val_monitors, str_val_monitor, epoch, batch, numbatches, NI, x);

	if (print_overwrite) {

		debug_printf	(DP_INFO,
				"\33[2K\r#%d->%d/%d;%s%s%s",
				epoch + 1, batch + 1, numbatches,
				str_progress,
				str_time,
				str_loss);
		if ('\0' != str_val_monitor[0])
			debug_printf(DP_INFO, "%s", str_val_monitor);
		if (NULL != post_string)
			debug_printf(DP_INFO, "%s", post_string);
		if (batch + 1 == numbatches)
			debug_printf(DP_INFO, "\n");

	} else {

		debug_printf	(DP_INFO,
				"#%d->%d/%d;%s%s%s%s",
				epoch, batch + 1, numbatches,
				str_progress,
				str_time,
				str_loss,
				str_val_monitor);
		if ('\0' != str_val_monitor[0])
			debug_printf(DP_INFO, "%s", str_val_monitor);
		if (NULL != post_string)
			debug_printf(DP_INFO, "%s", post_string);
		debug_printf(DP_INFO, "\n");
	}

	if (NULL != monitor->record) {

		long dims[3] = {monitor->epochs_created, numbatches, monitor->record_dim};
		long pos[3] = {epoch, batch, 0};

		MD_ACCESS(3, MD_STRIDES(3, dims, sizeof(complex float)), pos, monitor->record) = time;
		pos[2] = 1;
		MD_ACCESS(3, MD_STRIDES(3, dims, sizeof(complex float)), pos, monitor->record) = objective;
	}

	if (batch == numbatches - 1)
		monitor->start_time = timestamp();
}

static void monitor6_default_free(const struct monitor_iter6_s* _monitor)
{
	auto monitor = CAST_DOWN(monitor_iter6_default_s, _monitor);

	for ( int i = 0; i < monitor->num_val_monitors; i++)
		monitor->val_monitors[i].free(monitor->val_monitors[i].data);
	xfree(monitor->val_monitors);

	md_free(monitor->record);
}

void monitor_iter6_free(const struct monitor_iter6_s* monitor)
{
	monitor->free(monitor);
}

struct monitor_iter6_s* create_monitor_iter6_progressbar_trivial(void)
{

	PTR_ALLOC(struct monitor_iter6_default_s, monitor);
	SET_TYPEID(monitor_iter6_default_s, monitor);

	monitor->INTERFACE.fun = monitor6_default_fun;
	monitor->INTERFACE.free = monitor6_default_free;

	monitor->print_time = true;
	monitor->print_progress = true;
	monitor->print_overwrite = true;
	monitor->print_average_obj = false;

	monitor->start_time = timestamp();

	monitor->num_val_monitors = 0;
	monitor->val_monitors = NULL;

	monitor->use_record = false;
	monitor->epochs_written = 0;
	monitor->epochs_created = 0;
	monitor->num_batches = 0;
	monitor->record_dim = 2;
	monitor->record = NULL;

	monitor->average_obj = 0.;

	return CAST_UP(PTR_PASS(monitor));
}

struct monitor_iter6_s* create_monitor_iter6_progressbar_record(void)
{

	PTR_ALLOC(struct monitor_iter6_default_s, monitor);
	SET_TYPEID(monitor_iter6_default_s, monitor);

	monitor->INTERFACE.fun = monitor6_default_fun;
	monitor->INTERFACE.free = monitor6_default_free;

	monitor->print_time = true;
	monitor->print_progress = true;
	monitor->print_overwrite = true;
	monitor->print_average_obj = false;

	monitor->start_time = timestamp();

	monitor->num_val_monitors = 0;
	monitor->val_monitors = NULL;

	monitor->use_record = true;
	monitor->epochs_written = 0;
	monitor->epochs_created = 0;
	monitor->num_batches = 0;
	monitor->record_dim = 2;
	monitor->record = NULL;

	monitor->average_obj = 0.;

	return CAST_UP(PTR_PASS(monitor));
}

struct monitor_iter6_s* create_monitor_iter6_progressbar_with_val_monitor(int num_val_monitors, struct monitor_value_s* val_monitors)
{

	PTR_ALLOC(struct monitor_iter6_default_s, monitor);
	SET_TYPEID(monitor_iter6_default_s, monitor);

	monitor->INTERFACE.fun = monitor6_default_fun;
	monitor->INTERFACE.free = monitor6_default_free;

	monitor->print_time = true;
	monitor->print_progress = true;
	monitor->print_overwrite = true;
	monitor->print_average_obj = false;

	monitor->start_time = timestamp();

	monitor->num_val_monitors = num_val_monitors;
	PTR_ALLOC(struct monitor_value_s[num_val_monitors], nval_monitors);
	for (int i = 0; i < num_val_monitors; i++)
		(*nval_monitors)[i] = val_monitors[i];
	monitor->val_monitors = *PTR_PASS(nval_monitors);

	monitor->use_record = true;
	monitor->epochs_written = 0;
	monitor->epochs_created = 0;
	monitor->num_batches = 0;
	monitor->record_dim = 2 + 2 * num_val_monitors;
	monitor->record = NULL;

	monitor->average_obj = 0.;

	return CAST_UP(PTR_PASS(monitor));
}

struct monitor_iter6_nlop_s {

	INTERFACE(monitor_iter6_value_data_t);

	const struct nlop_s* nlop;
	_Bool eval_each_batch;
	const char* name;
	complex float last_result;
};

static DEF_TYPEID(monitor_iter6_nlop_s);

static complex float monitor_iter6_nlop_fun(const monitor_iter6_value_data_t* data, long NI, const float* args[NI])
{
        const auto d = CAST_DOWN(monitor_iter6_nlop_s, data);
	assert(nlop_get_nr_in_args(d->nlop) == NI);

	void* tmp_args[NI + 1];
	tmp_args[0] = md_alloc_sameplace(1, MD_SINGLETON_DIMS(1), CFL_SIZE, args[0]);
	for (int i = 0; i < NI; i++)
		tmp_args[1 + i] = (void*)args[i];

	nlop_generic_apply_unchecked(d->nlop, NI + 1, tmp_args);

	md_copy(1, MD_SINGLETON_DIMS(1), &(d->last_result), tmp_args[0], CFL_SIZE);
	md_free(tmp_args[0]);
	return d->last_result;
}

static bool monitor_iter6_nlop_eval(const monitor_iter6_value_data_t* _data, long epoch, long batch, long num_batches)
{
	const auto d = CAST_DOWN(monitor_iter6_nlop_s, _data);
	UNUSED(epoch);
	return d->eval_each_batch || (num_batches == batch + 1);
}

static void monitor_iter6_nlop_print(const monitor_iter6_value_data_t* _data, int N, char log_str[N])
{
	const auto d = CAST_DOWN(monitor_iter6_nlop_s, _data);
	if (NULL == d->name) {

		log_str[0] = '\0';
		return;
	}

	if (0. == cimagf(d->last_result))
		sprintf(log_str, " %s: %.3e;", d->name, crealf(d->last_result));
	else
		sprintf(log_str, " %s: %.3e + %.3ei;", d->name, crealf(d->last_result), cimagf(d->last_result));
}

static void monitor_iter6_nlop_free(const monitor_iter6_value_data_t* _data)
{
	const auto d = CAST_DOWN(monitor_iter6_nlop_s, _data);
	nlop_free(d->nlop);
	xfree(d->name);
	xfree(d);
}


struct monitor_value_s monitor_iter6_nlop_create(const struct nlop_s* nlop, _Bool eval_each_batch, const char* print_name)
{
	PTR_ALLOC(struct monitor_iter6_nlop_s, data);
	SET_TYPEID(monitor_iter6_nlop_s, data);

	data->nlop = nlop_clone(nlop);
	data->eval_each_batch = eval_each_batch;

	if ((NULL != print_name) && (0 < strlen(print_name))) {

		PTR_ALLOC(char[strlen(print_name) + 1], tmp_name);
		strcpy(*tmp_name, print_name);
		data->name=*PTR_PASS(tmp_name);
	} else {

		data->name = NULL;
	}

	struct monitor_value_s result = {

		.data = CAST_UP(PTR_PASS(data)),
		.fun = monitor_iter6_nlop_fun,
		.eval = monitor_iter6_nlop_eval,
		.print = monitor_iter6_nlop_print,
		.free = monitor_iter6_nlop_free
	};

        return result;
}
