#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linops/grad.h"
#include "linops/linop.h"
#include "linops/someops.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"
#include "num/ops.h"

#include "misc/io.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include "simu/fd_geometry.h"
#include "simu/leray.h"



#define N 4


struct process_dat {
	complex float *j_hist;
	unsigned long hist;

	const struct linop_s *leray_op;
};

static complex float *j_wrapper(void *_data, const float *phi)
{
	auto data = (struct process_dat *)_data;

	assert(NULL != data->leray_op);
	auto leray_data = linop_get_data((data->leray_op));

	linop_leray_calc_projection(leray_data, data->j_hist, (const complex float*)phi);
	return data->j_hist;
}

static bool selector(const unsigned long iter, const float *x, void *_data)
{
		UNUSED(x);
		struct process_dat *data = _data;
		return data->hist > 0 ? (0 == iter % data->hist) : false;
}



static const char usage_str[] = "vectordimension <mask> <input> <output>";
static const char help_str[] = "Project a vectorfield onto it's divergence free component\n";

int main_leray(int argc, char *argv[])
{
	int n = 30;
	float lambda = 1e-3;
	long hist = -1;
	const struct opt_s opts[] = {
	    OPT_INT('n', &n, "n", "Number of Iterations"),
	    OPT_LONG('p', &hist, "n", "Record all p steps"),
	    OPT_FLOAT('l', &lambda, "lambda", "Inversion l2 regularization"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();

	long dims[N] = {}, mask_dims[N] = {};
	unsigned int d = atoi(argv[1]);

	complex float *in = load_cfl(argv[3], N, dims);
	complex float *out = create_cfl(argv[4], N, dims);

	complex float *mask = load_cfl(argv[2], N, mask_dims);
	for (unsigned int i = 0; i < N; i++)
		assert(i == d ? mask_dims[i] == 1 : mask_dims[i] == dims[i]);

	// setup monitoring
	complex float *j_hist = md_calloc(N, dims, CFL_SIZE);
	struct process_dat j_dat = { .j_hist = j_hist, .hist = hist, .leray_op = NULL};

	auto mon = create_monitor_recorder(N, dims, "j_step", (void *)&j_dat, selector, j_wrapper);
	//auto mon = create_monitor_recorder(N, mask_dims, "phi_step", (void *)&j_dat, selector, NULL);
	auto op = linop_leray_create(N, dims, d, n, lambda, mask, mon);
	j_dat.leray_op = op;

	linop_forward(op, N, dims, out, N, dims, in);
	linop_free(op);

	md_free(j_hist);

	unmap_cfl(N, dims, in);
	unmap_cfl(N, dims, out);
	unmap_cfl(N, mask_dims, mask);

	return 0;
}

