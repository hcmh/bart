#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "iter/lsqr.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops_p.h"
#include "num/ops.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/misc.h"
#include "iter/monitor.h"

#include "linops/linop.h"
#include "linops/fmac.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "num/iovec.h"
#include "num/ops.h"

static const char help_str[] = "Solve x = argmin ||y -Ax||^2 + R(x)";



int main_lsqr(int argc, char* argv[argc])
{
	const char* y_file = NULL;
	const char* A_file = NULL;
	const char* x_file = NULL;
	unsigned long x_flags;

	struct arg_s args[] = {

		ARG_ULONG(true, &x_flags, "Flag of dimensions present in x"),
		ARG_INFILE(true, &y_file, "y"),
		ARG_INFILE(true, &A_file, "A"),
		ARG_OUTFILE(true, &x_file, "x"),
	};

	// Start time count

	unsigned long loop_flags = 0;
	bool eigen = true;
	unsigned int maxiter = 30;
	float step = -1.;
	bool warm_start = false;
	float scaling = 1.;


	struct admm_conf admm = { false, false, false, iter_admm_defaults.rho, iter_admm_defaults.maxitercg };

	enum algo_t algo = ALGO_DEFAULT;
	bool hogwild = false;
	bool fast = false;


	struct opt_reg_s ropts;
	opt_reg_init(&ropts);

	struct lsqr_conf conf = lsqr_defaults;


	const struct opt_s opts[] = {

		{ 'l', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "\b1/-l2", "  toggle l1-wavelet or l2 regularization." },
		OPT_FLOAT('r', &ropts.lambda, "lambda", "regularization parameter"),
		{ 'R', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "<T>:A:B:C", "generalized regularization options (-Rh for help)" },
		OPT_FLOAT('s', &step, "step", "iteration stepsize"),
		OPT_UINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_SET('g', &conf.it_gpu, "use GPU"),
		OPT_SELECT('I', enum algo_t, &algo, ALGO_IST, "select IST"),
		OPT_SET('D', &admm.dynamic_rho, "(ADMM dynamic step size)"),
		OPT_SET('J', &admm.relative_norm, "(ADMM residual balancing)"),
		OPT_FLOAT('u', &admm.rho, "rho", "ADMM rho"),
		OPT_UINT('C', &admm.maxitercg, "iter", "ADMM max. CG iterations"),
		OPT_SELECT('m', enum algo_t, &algo, ALGO_ADMM, "select ADMM"),
		OPT_SELECT('a', enum algo_t, &algo, ALGO_PRIDU, "select Primal Dual"),
		OPT_ULONG('L', &loop_flags, "flags", "loop flags"),
		};


	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (conf.it_gpu)
		num_init_gpu_memopt();
	else
		num_init();

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	admm.dynamic_tau = admm.relative_norm;


	long y_dims[DIMS];
	long A_dims[DIMS];
	long x_dims[DIMS];

	complex float* y = load_cfl(y_file, DIMS, y_dims);
	complex float* A = load_cfl(A_file, DIMS, A_dims);

	unsigned long y_flags = md_nontriv_dims(DIMS, y_dims);
	unsigned long A_flags = md_nontriv_dims(DIMS, A_dims);

	long max_dims[DIMS];
	md_max_dims(DIMS, ~0, max_dims, y_dims, A_dims);
	md_select_dims(DIMS, x_flags, x_dims, max_dims);

	complex float* x = create_cfl(x_file, DIMS, x_dims);

	assert(md_check_equal_dims(DIMS, max_dims, y_dims, y_flags));
	assert(md_check_equal_dims(DIMS, max_dims, A_dims, A_flags));
	assert(md_check_equal_dims(DIMS, max_dims, x_dims, x_flags));


	long y_dims_loop[DIMS];
	long A_dims_loop[DIMS];
	long x_dims_loop[DIMS];
	long max_dims_loop[DIMS];

	md_select_dims(DIMS, ~loop_flags, y_dims_loop, y_dims);
	md_select_dims(DIMS, ~loop_flags, A_dims_loop, A_dims);
	md_select_dims(DIMS, ~loop_flags, x_dims_loop, x_dims);
	md_select_dims(DIMS, ~loop_flags, max_dims_loop, max_dims);


	opt_reg_configure2(DIMS, x_dims_loop, &ropts, thresh_ops, trafos, 0, 0, NULL, NULL, conf.it_gpu);

	int nr_penalties = ropts.r;
	struct reg_s* regs = ropts.regs;

	if (ALGO_DEFAULT == algo)
		algo = italgo_choose(nr_penalties, regs);

	if ((ALGO_IST == algo) || (ALGO_FISTA == algo) || (ALGO_PRIDU == algo)) {

		// For non-Cartesian trajectories, the default
		// will usually not work. TODO: The same is true
		// for sensitivities which are not normalized, but
		// we do not detect this case.

		if (-1. == step)
			step = 0.95;
	}

	if ((ALGO_CG == algo) || (ALGO_ADMM == algo))
		if (-1. != step)
			debug_printf(DP_INFO, "Stepsize ignored.\n");

	if (eigen) {

		const struct linop_s* lop = linop_fmac_create(DIMS, max_dims_loop, ~y_flags, ~x_flags, ~A_flags, A);
		float maxeigen = estimate_maxeigenval_gpu(lop->normal);
		step /= maxeigen;
		debug_printf(DP_INFO, "Maximum eigenvalue: %.2e\n", maxeigen);
		linop_free(lop);
	}


	// initialize algorithm

	struct iter it = italgo_config(algo, nr_penalties, regs, maxiter, step, hogwild, fast, admm, scaling, warm_start);

	bool trafos_cond = (   (ALGO_PRIDU == algo)
			    || (ALGO_ADMM == algo)
			    || (   (ALGO_NIHT == algo)
				&& (regs[0].xform == NIHTWAV)));

	if (ALGO_CG == algo)
		nr_penalties = 0;

	// FIXME: will fail with looped dims
	struct iter_monitor_s* monitor = NULL;



	long pos[DIMS];
	md_singleton_strides(DIMS, pos);

	do {
		complex float* xl = md_alloc(DIMS, x_dims_loop, CFL_SIZE);
		complex float* yl = md_alloc(DIMS, y_dims_loop, CFL_SIZE);
		complex float* Al = md_alloc(DIMS, A_dims_loop, CFL_SIZE);

		debug_print_dims(DP_INFO, DIMS, pos);

		md_copy2(DIMS, y_dims_loop, MD_STRIDES(DIMS, y_dims_loop, CFL_SIZE), yl, MD_STRIDES(DIMS, y_dims, CFL_SIZE), &(MD_ACCESS(DIMS, pos, MD_STRIDES(DIMS, y_dims, CFL_SIZE), y)), CFL_SIZE);
		md_copy2(DIMS, A_dims_loop, MD_STRIDES(DIMS, A_dims_loop, CFL_SIZE), Al, MD_STRIDES(DIMS, A_dims, CFL_SIZE), &(MD_ACCESS(DIMS, pos, MD_STRIDES(DIMS, A_dims, CFL_SIZE), A)), CFL_SIZE);

		const struct linop_s* lop = linop_fmac_create(DIMS, max_dims_loop, ~y_flags, ~x_flags, ~A_flags, Al);

		lsqr2(	DIMS, &conf, it.italgo, it.iconf,
		lop,
		nr_penalties, thresh_ops, trafos_cond ? trafos : NULL,
		x_dims_loop, xl, y_dims_loop, yl, NULL, monitor);

		md_copy2(DIMS, x_dims_loop, MD_STRIDES(DIMS, x_dims, CFL_SIZE), &(MD_ACCESS(DIMS, pos, MD_STRIDES(DIMS, x_dims, CFL_SIZE), x)), MD_STRIDES(DIMS, x_dims_loop, CFL_SIZE), xl, CFL_SIZE);

		md_free(xl);
		md_free(yl);
		md_free(Al);

		linop_free(lop);

	} while (md_next(DIMS, max_dims, loop_flags, pos));


	unmap_cfl(DIMS, y_dims, y);
	unmap_cfl(DIMS, A_dims, A);
	unmap_cfl(DIMS, x_dims, x);

	opt_reg_free(&ropts, thresh_ops, trafos);
	italgo_config_free(it);

	return 0;

}
