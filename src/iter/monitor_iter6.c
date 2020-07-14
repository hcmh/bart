#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>

#include "iter/italgos.h"
#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"

#include "iter/vec.h"

#include "nlops/nlop.h"

#include "monitor_iter6.h"
#include "num/flpmath.h"
#include "num/multind.h"


void iter6_monitor(struct iter6_monitor_s* monitor, long epoch, long batch, float objective, long NI, const float* x[NI], char* post_string)
{
	if ((NULL != monitor) && (NULL != monitor->fun))
		monitor->fun(monitor, epoch, batch, objective, NI, x, post_string);
}


struct monitor6_s {

	INTERFACE(iter6_monitor_t);

	bool print_time;
	bool print_progress;
	bool print_overwrite;
	bool print_average_obj;

	bool valloss_for_each_batch;

	double start_time;

	float average_obj;

	int numepochs;
	int numbatches;
	complex float* record;

	int NI;
	enum IN_TYPE* in_type;
	const struct nlop_s* nlop_val_loss;
};

static DEF_TYPEID(monitor6_s);

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

static void select_derivatives(long NO, unsigned long out_der_flags, long NI, unsigned long in_der_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI])
{
	for (int i = 0; i < NO + NI; i++)
		for (int j = 0; j < NO + NI; j++) {

			run_opts[i][j] = 0;

			if ((i < NO) && !(j < NO) && (!MD_IS_SET(out_der_flags, i) || !MD_IS_SET(in_der_flags, j - NO)))
				run_opts[i][j] = MD_SET(run_opts[i][j], OP_APP_NO_DER);

			if (!(i < NO) && (j < NO) && (!MD_IS_SET(in_der_flags, i - NO) || !MD_IS_SET(out_der_flags, j)))
				run_opts[i][j] = MD_SET(run_opts[i][j], OP_APP_NO_DER);
		}
}

static float compute_validation_objective(const struct nlop_s* nlop, long NI, enum IN_TYPE in_type[NI], const float* x[NI])
{
	if (NULL == nlop)
		return 0.;
	assert(NULL != in_type);
	assert(NULL != x);

	const _Complex float* args[NI + 1];
	long NW = 1;

	for (int i = 0; i < NI; i++)
		if ((IN_OPTIMIZE == in_type[i]) || (IN_BATCH == in_type[i])) {

			args[NW] = (const _Complex float*)x[i];
			NW++;
		}

	args[0] = md_alloc_sameplace(1, MAKE_ARRAY(1l), sizeof(_Complex float), args[1]);

	operator_run_opt_flags_t run_opts[NW][NW];
	select_derivatives(1, 0, NW - 1, 0, run_opts);
	nlop_generic_apply_extopts_unchecked(nlop, NW, (void**)args, run_opts);
	//nlop_generic_apply_unchecked(nlop, NW, (void**)args);

	float result = 0;

	md_copy(1, MAKE_ARRAY(1l), &result, args[0], sizeof(float));
	md_free(args[0]);
	return result;
}


static void monitor6_default_fun(struct iter6_monitor_s* _monitor, long epoch, long batch, float objective, long NI, const float* x[NI], char* post_string)
{
	auto monitor = CAST_DOWN(monitor6_s, _monitor);

	bool print_progress = monitor->print_progress;
	bool print_time = monitor->print_time;
	bool print_val = (NULL != monitor->nlop_val_loss) && (monitor->valloss_for_each_batch || batch == monitor->numbatches);
	bool print_loss = true;
	bool print_overwrite = true;

	char str_progress[15];
	if (print_progress)
		print_progress_bar(10, str_progress, batch, monitor->numbatches);
	else
		str_progress[0] = '\0';

	double time = timestamp() - monitor->start_time;
	double est_time = time + (double)(monitor->numbatches - batch) * time / (double)(batch);
	char str_time[30];
	if (print_time)
		print_time_string(30, str_time, time, est_time);
	else
		str_time[0] = '\0';

	char str_valloss[30];
	float valloss = 0.;
	if (print_val) {

		valloss = compute_validation_objective(monitor->nlop_val_loss, NI, monitor->in_type, x);
		sprintf(str_valloss, " val loss: %f;", valloss);
	} else {
		str_valloss[0] = '\0';
	}

	char str_loss[30];
	monitor->average_obj = ((batch - 1) * monitor->average_obj + objective) / batch;
	if (print_loss)
		sprintf(str_loss, " loss: %f;", monitor->print_average_obj ? monitor->average_obj: objective);
	else
		str_loss[0] = '\0';

	char null_char = '\0';
	post_string = (NULL == post_string) ? &null_char : post_string;

	if (print_overwrite) {

		debug_printf	(DP_INFO,
				"\33[2K\r#%d->%d/%d;%s%s%s%s%s",
				epoch, batch, monitor->numbatches,
				str_progress,
				str_time,
				str_loss,
				str_valloss,
				post_string);
		if (batch == monitor->numbatches)
			debug_printf(DP_INFO, "\n");
	} else {

		debug_printf	(DP_INFO,
				"#%d->%d/%d;%s%s%s%s%s\n",
				epoch, batch, monitor->numbatches,
				str_progress,
				str_time,
				str_loss,
				str_valloss,
				post_string);
	}

	if (NULL != monitor->record) {

		long dims[3] = {monitor->numepochs, monitor->numbatches, 3};
		long pos[3] = {epoch, batch-1, 0};

		MD_ACCESS(3, MD_STRIDES(3, dims, sizeof(complex float)), pos, monitor->record) = time;
		pos[2] = 1;
		MD_ACCESS(3, MD_STRIDES(3, dims, sizeof(complex float)), pos, monitor->record) = objective;
		pos[2] = 2;
		MD_ACCESS(3, MD_STRIDES(3, dims, sizeof(complex float)), pos, monitor->record) = print_val ? valloss : 0.;
	}

	if (batch == monitor->numbatches)
		monitor->start_time = timestamp();
}

struct iter6_monitor_s* create_iter6_monitor_progressbar(int numbatches, bool average_obj)
{

	PTR_ALLOC(struct monitor6_s, monitor);
	SET_TYPEID(monitor6_s, monitor);

	monitor->INTERFACE.fun = monitor6_default_fun;

	monitor->print_time = true;
	monitor->print_progress = true;
	monitor->print_overwrite = true;
	monitor->print_average_obj = average_obj;

	monitor->valloss_for_each_batch = false;

	monitor->start_time = timestamp();

	monitor->average_obj = 0.;

	monitor->numepochs = 0;
	monitor->numbatches = numbatches;
	monitor->record = NULL;

	monitor->NI = 0;
	monitor->in_type = NULL;
	monitor->nlop_val_loss = NULL;

	return CAST_UP(PTR_PASS(monitor));
}

struct iter6_monitor_s* create_iter6_monitor_progressbar_validloss(int numepochs, int numbatches, bool average_obj, long NI, enum IN_TYPE in_type[NI], const struct nlop_s* valid_loss, bool valloss_for_each_batch)
{

	PTR_ALLOC(struct monitor6_s, monitor);
	SET_TYPEID(monitor6_s, monitor);

	monitor->INTERFACE.fun = monitor6_default_fun;

	monitor->print_time = true;
	monitor->print_progress = true;
	monitor->print_overwrite = true;
	monitor->print_average_obj = average_obj;

	monitor->valloss_for_each_batch = valloss_for_each_batch;

	monitor->start_time = timestamp();

	monitor->average_obj = 0.;

	monitor->numepochs = numepochs;
	monitor->numbatches = numbatches;
	long rdims[3] = {numepochs, numbatches, 3};
	monitor->record = md_alloc(3, rdims, CFL_SIZE);

	monitor->NI = NI;
	monitor->in_type = in_type;
	monitor->nlop_val_loss = valid_loss;



	return CAST_UP(PTR_PASS(monitor));
}


void iter6_monitor_dump_record(struct iter6_monitor_s* _monitor, const char* filename)
{
	auto monitor = CAST_DOWN(monitor6_s, _monitor);

	long rdims[3] = {monitor->numepochs, monitor->numbatches, 3};
	if (NULL != monitor->record)
		dump_cfl(filename, 3, rdims, monitor->record);
	else
		error("Record not available!\n");
}
