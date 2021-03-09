#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "iter/iter2.h"
#include "linops/grad.h"
#include "linops/linop.h"
#include "linops/someops.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"
#include "num/ops.h"
#include "num/ops_p.h"

#include "num/gpuops.h"

#include "iter/iter.h"
#include "iter/monitor.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/types.h"

#include "simu/biot_savart_fft.h"
#include "simu/leray.h"
#include "simu/fd_geometry.h"
#define N 4
#define NPROX 2

enum PROXFUN { PF_l2,
	       PF_thresh,
	       PF_ind };

enum SOLVER { CG,
	      ADMM };



struct history_data_s {
	unsigned long hist_1;
	unsigned long hist_2;
	complex float *j_hist;
	const long *dims;
	const long *j_dims;
	complex float *mask2;
	struct linop_s *op;
	struct linop_s *post_j_op;

	const linop_data_t *leray_data;
};

static bool history_select_1(const unsigned long iter, const float *x, void *_data)
{
		UNUSED(x);
		struct history_data_s *data = _data;
		return data->hist_1 > 0 ? (0 == iter % data->hist_1) : false;
}

static bool history_select_2(const unsigned long iter, const float *x, void *_data)
{
		UNUSED(x);
		struct history_data_s *data = _data;
		return data->hist_2 > 0 ? (0 == iter % data->hist_2) : false;
}

static complex float *history_save_2(void *_data, const float *phi)
{
	auto data = (struct history_data_s *)_data;
	assert(NULL != data->leray_data);

	linop_leray_calc_projection(data->leray_data, data->j_hist, (const complex float*)phi);

	return data->j_hist;
	//return phi;
}


static complex float *history_save_1(void *_data, const float *phi)
{
	auto data = (struct history_data_s *)_data;
	if (NULL != data->post_j_op) {
		linop_forward(data->post_j_op, N, data->j_dims, data->j_hist, N, data->j_dims, (const complex float *)phi);
		return data->j_hist;
	} else {
		return (complex float*) phi;
	}
}



static void get_nc_boundaries(const long bdims[N], complex float* out,  const complex float *boundaries, const complex float *electrodes)
{
	complex float *electrodes_indicator = md_alloc(N, bdims, CFL_SIZE);
	md_zabs(N, bdims, electrodes_indicator, electrodes);
	md_zsgreatequal(N, bdims, electrodes_indicator, electrodes_indicator, .9);

	assert(md_zscalar_real(N, bdims, electrodes_indicator, boundaries) < 1e-8);

	md_zadd(N, bdims, out, boundaries, electrodes_indicator);
}

//Remove normal component of the current where no electrodes are attached
static struct linop_s *make_wall_op(const long jdims[N], const long bdims[N], const complex float *boundaries, const complex float *electrodes)
{

		long bstrs[N], jstrs[N];
		md_calc_strides(N, bstrs, bdims, CFL_SIZE);
		md_calc_strides(N, jstrs, jdims, CFL_SIZE);

		complex float *nc_boundaries = md_alloc(N, bdims, CFL_SIZE);
		get_nc_boundaries(bdims, nc_boundaries, boundaries, electrodes);

		complex float *walls = md_alloc(N, jdims, CFL_SIZE);
		calc_outward_normal(N, jdims, walls, 0, bdims, nc_boundaries);

		for (int i = 0; i < 3; i++)
			md_zmul2(N, bdims, jstrs, (void *)walls + i * jstrs[0], jstrs, (void *)walls + i * jstrs[0] , bstrs, boundaries);

		md_zabs(N, jdims, walls, walls);
		md_zsgreatequal(N, jdims, walls, walls, 0.1);

		md_zsmul(N, jdims, walls, walls, -1);
		md_zsadd(N, jdims, walls, walls, 1);
		auto wall_op = linop_cdiag_create(N, jdims, 15, walls);

			char* str = getenv("DEBUG_LEVEL");
			debug_level = (NULL != str) ? atoi(str) : DP_INFO;
			if (5 <= debug_level)
				dump_cfl("DEBUG_walls", N, jdims, walls);

		md_free(nc_boundaries);
		md_free(walls);
		return wall_op;

}


//Remove current outside the conductive domain
static struct linop_s *make_cmask_op(const long jdims[N], const long bdims[N], const complex float *boundaries, const complex float *electrodes)
{

		long bstrs[N], jstrs[N];
		md_calc_strides(N, bstrs, bdims, CFL_SIZE);
		md_calc_strides(N, jstrs, jdims, CFL_SIZE);

		complex float *nc_boundaries = md_alloc(N, bdims, CFL_SIZE);
		get_nc_boundaries(bdims, nc_boundaries, boundaries, electrodes);

		complex float *nc_boundaries3 = md_alloc(N, jdims, CFL_SIZE);

		long pos[N] = {0};
		for (; pos[0] < 3; pos[0]++)
			md_copy_block(N, pos, jdims, nc_boundaries3, bdims, nc_boundaries, CFL_SIZE);

		auto wall_op = linop_cdiag_create(N, jdims, 15, nc_boundaries3);

			char* str = getenv("DEBUG_LEVEL");
			debug_level = (NULL != str) ? atoi(str) : DP_INFO;
			if (5 <= debug_level)
				dump_cfl("DEBUG_conductive", N, jdims, nc_boundaries3);

		md_free(nc_boundaries);
		md_free(nc_boundaries3);
		return wall_op;

}

static void cdi_reco(const float vox[3], const long jdims[N], complex float *j, const long bdims[N], const complex float *bz, const complex float *mask, const float reg, int iter, int admm_iter, float tol, const complex float *bc_mask, const complex float* electrodes, const float bc_reg, const float div_reg, const enum PROXFUN div_pf, const int div_order, const int leray_iter, const unsigned long outer_hist, const unsigned long leray_hist, const enum SOLVER solver, const complex float* l2weights)
{

	//Monitoring
	complex float *j_hist = md_calloc(N, jdims, CFL_SIZE);

	struct history_data_s history_data = { .hist_1 = outer_hist, .hist_2 = leray_hist, .j_hist = j_hist, .j_dims = jdims, .leray_data = NULL, .post_j_op = NULL };
	auto mon1 = create_monitor_recorder(N, jdims, "j_step", (void *)&history_data, history_select_1, history_save_1);
	auto mon2 = create_monitor_recorder(N, jdims, "leray_step", (void *)&history_data, history_select_2, history_save_2);

	//forward operator
	auto bz_op = linop_bz_create(jdims, vox);
	const struct linop_s *bc_mask_op = NULL, *wall_op = NULL, *leray_op = NULL;

	//Mask regions with low signal
	if (NULL != mask) {
		auto mask_op = linop_cdiag_create(N, bdims, 15, mask);
		bz_op = linop_chain_FF(bz_op, mask_op);
	}

	//Mask current density
	if (NULL != bc_mask) {
		bc_mask_op = make_cmask_op(jdims, bdims, bc_mask, electrodes);

		if (PF_ind == div_pf) {
			complex float *nc_boundaries = md_alloc(N, bdims, CFL_SIZE);
			get_nc_boundaries(bdims, nc_boundaries, bc_mask, electrodes);
			leray_op = linop_leray_create(N, jdims, 0, leray_iter, div_reg, nc_boundaries, mon2);
			history_data.leray_data = linop_get_data(leray_op);
			md_free(nc_boundaries);
		}

		if (0 < bc_reg)
			wall_op = make_wall_op(jdims, bdims, bc_mask, electrodes);
	}

	md_zfill(N, jdims, j, 0);
	struct lsqr_conf lconf = lsqr_defaults;

	switch (solver) {
	case CG: {
		assert(iter > 0);
		struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
		conf.maxiter = iter;
		conf.l2lambda = 0;
		conf.tol = tol;

		assert(div_pf != PF_thresh);

		complex float unity[] = { 1 };
		auto post_j_op = linop_cdiag_create(N, jdims, 0, unity);

		if (NULL != bc_mask_op)
			post_j_op = linop_chain(post_j_op, bc_mask_op);

		if (0 < bc_reg)
			post_j_op = linop_chain(post_j_op, wall_op);

		if ((0 < div_reg) && (div_pf == PF_ind))
			post_j_op = linop_chain(post_j_op, leray_op);

		complex float *adj = md_calloc(N, jdims, CFL_SIZE);
		auto adj_op = linop_chain(post_j_op, bz_op);
		linop_adjoint(adj_op, N, jdims, adj, N, bdims, bz);
		linop_free(adj_op);

		auto normal_op = linop_get_normal(bz_op);

		complex float* l2_diag = md_alloc(N, jdims, CFL_SIZE);

		if (NULL == l2weights) {
			md_zfill(N, jdims, l2_diag, 1);
		} else {
			long bstrs[N], jstrs[N];
			md_calc_strides(N, bstrs, bdims, CFL_SIZE);
			md_calc_strides(N, jstrs, jdims, CFL_SIZE);
			md_copy2(N, jdims, jstrs, l2_diag, bstrs, l2weights, CFL_SIZE);
		}

		md_zsmul(N, jdims, l2_diag, l2_diag, reg);
		auto l2_op = linop_cdiag_create(N, jdims, 15, l2_diag);

		normal_op = linop_plus_FF(normal_op, l2_op);


		if ((0 < div_reg) && (div_pf == PF_l2)) {
			assert(NULL != bc_mask_op);
			assert((vox[0] == vox[1]) && (vox[0] == vox[2]) && (vox[1] == vox[2]));

			auto div_op = linop_div_create(N, jdims, 0, 14, div_order, BC_SAME);
			auto div_normal = linop_get_normal(div_op);
			const complex float scale = div_reg;
			auto scale_op = linop_cdiag_create(N, jdims, 0, &scale);
			div_normal = linop_chain(div_normal, scale_op);
			normal_op = linop_plus(normal_op, div_normal);

			linop_free(div_op);
			linop_free(div_normal);
			linop_free(scale_op);
		}

		//history_data.post_j_op = post_j_op;
		normal_op = linop_chain_FF(normal_op, linop_get_adjoint(post_j_op));

		long size = 2 * md_calc_size(N, jdims); // multiply by 2 for float size
		iter_conjgrad(CAST_UP(&conf), normal_op->forward, NULL, size, (float *)j, (const float *)adj, mon1);

		//linop_forward(post_j_op, N, jdims, j, N, jdims, j);

		md_free(adj);
		md_free(l2_diag);
		linop_free(normal_op);

	} break;
	case ADMM: {
		struct iter_admm_conf conf = iter_admm_defaults;
		assert(admm_iter > 0);
		conf.maxiter = admm_iter;
		if (iter > 0)
			conf.maxitercg = iter;
		lconf.lambda = reg;

		long nprox = 0;
		const struct operator_p_s *prox_funs[NPROX] = {NULL};
		const struct linop_s *prox_linops[NPROX] = {NULL};

		if (0 < bc_reg) {
			assert(NPROX >= ++nprox);
			assert(NULL != wall_op);
			prox_funs[nprox - 1] = prox_indicator_create(wall_op);
			complex float one[1] = {1.};
			prox_linops[nprox - 1] = linop_cdiag_create(N, jdims, 0, one);
		}

		if (0 < div_reg) {
			assert(NPROX >= ++nprox);
			assert(NULL != bc_mask_op);
			//FIXME: Scale j with voxelsize before applying derivative
			assert((vox[0] == vox[1]) && (vox[0] == vox[2]) && (vox[1] == vox[2]));

			auto div_op = linop_div_create(N, jdims, 0, 14, div_order, BC_SAME);
			if (div_pf == PF_l2) {
				prox_funs[nprox - 1] = prox_l2norm_create(N, bdims, div_reg);
				prox_linops[nprox - 1] = linop_chain(bc_mask_op, div_op);
			} else if (div_pf == PF_thresh) {
				prox_funs[nprox - 1] = prox_thresh_create(N, bdims, div_reg, 0);
				prox_linops[nprox - 1] = linop_chain(bc_mask_op, div_op);
			} else if (div_pf == PF_ind) {
				assert(NULL != leray_op);
				prox_funs[nprox - 1] = prox_indicator_create(leray_op);
				complex float one[1] = {1.};
				prox_linops[nprox - 1] = linop_cdiag_create(N, jdims, 0, one);
			} else {
				assert(false);
			}
			linop_free(div_op);
		}

		lsqr2(N, &lconf, iter2_admm, CAST_UP(&conf), bz_op, nprox, prox_funs, prox_linops, jdims, j, bdims, bz, NULL, mon1);

		for (int i = 0; i < nprox; i++) {
			linop_free(prox_linops[i]);
			operator_p_free(prox_funs[i]);
		}
	} break;
	default:
		assert(false);
	}

	linop_free(bz_op);
	if (NULL != bc_mask_op)
		linop_free(bc_mask_op);
	if (NULL != leray_op)
		linop_free(leray_op);
	if (NULL != wall_op)
		linop_free(wall_op);
	md_free(j_hist);
}


static const char usage_str[] = "voxelsize(x) voxelsize(y) voxelsize(z) <B_z> [<mask_signal> [<mask_interior> <electrodes> [<l2_weighting>]]] <J>";
static const char help_str[] = "Estimate the current density J (A/[voxelsize]^2) that generates B_z (Hz)\n";


int main_cdi(int argc, char *argv[])
{
	float tik_reg = 0, tolerance = 1e-3, bc_reg = -1, div_reg = -1;
	int iter = 100, admm_iter = 100, div_pf_int = 0, div_order = 1, leray_iter = 20;
	long leray_hist = -1, outer_hist = -1;
	bool cg = false;
	enum SOLVER solver;

	struct opt_s hist_opt[] = {
		OPT_LONG('a', &outer_hist, "n_1", "Save J every n_1 steps (outer iteration)"),
		OPT_LONG('b', &leray_hist, "n_2", "Save J every n_2 LeRay-Steps (outer iteration)"),
	}; //FIXME

	const struct opt_s opts[] = {
	    OPT_FLOAT('l', &tik_reg, "lambda_1", "Tikhonov Regularization"),
	    OPT_FLOAT('b', &bc_reg, "b", "Boundary current penalty"),
	    OPT_FLOAT('d', &div_reg, "d", "Divergence penalty or LeRay Regularization"),
	    OPT_INT('D', &div_pf_int, "D", "Divergence Prox function: 0 -> l2norm, 1->thresh, 2->div=0 equality constraint"),
	    OPT_INT('o', &div_order, "", "Finite difference order for divergence calculation (1,2)"),
	    OPT_INT('p', &leray_iter, "", "Number of Iterations for LeRay Projection"),
	    OPT_FLOAT('t', &tolerance, "t", "Stopping Tolerance"),
	    OPT_INT('n', &iter, "Iterations", "Max. number of cg iterations"),
	    OPT_INT('m', &admm_iter, "Iterations", "Max. number of ADMM iterations"),
	    //OPT_SUBOPT('H', "n", "Convergence recording options. -Hh for help.", ARRAY_SIZE(hist_opt), hist_opt),
	    OPT_LONG('A', &outer_hist, "n_1", "Save J every n_1 steps (outer iteration)"),
	    OPT_LONG('B', &leray_hist, "n_2", "Save J every n_2 LeRay-Steps (outer iteration)"),
	    OPT_SET('C', &cg, "Use CG"),
	};
	cmdline(&argc, argv, 5, 9, usage_str, help_str, ARRAY_SIZE(opts), opts);

	enum PROXFUN div_pf;
	if (div_pf_int == 0)
		div_pf = PF_l2;
	else if (div_pf_int == 1)
		div_pf = PF_thresh;
	else if (div_pf_int == 2)
		div_pf = PF_ind;
	else
		assert(false);

	if (cg)
		solver = CG;


	num_init();

	const int vox_ind = 1;
	float vox[3] = {strtof(argv[vox_ind], NULL), strtof(argv[vox_ind + 1], NULL), strtof(argv[vox_ind + 2], NULL)};
	long bdims[N], jdims[N], mask_dims[N], mask_bc_dims[N], mask_electrodes_dims[N], l2weights_dims[N];
	complex float *mask = NULL, *bc_mask = NULL, *electrodes = NULL, *l2weights = NULL;

	complex float *b = load_cfl(argv[4], N, bdims);
	assert(bdims[0] == 1);

	if (argc >= 7) {
		mask = load_cfl(argv[5], N, mask_dims);
		for (int i = 0; i < N; i++)
			assert(mask_dims[i] == bdims[i]);
	}

	if (argc >= 8) {
		bc_mask = load_cfl(argv[6], N, mask_bc_dims);
		for (int i = 0; i < N; i++)
			assert(mask_bc_dims[i] == bdims[i]);
		assert(argc >= 9);
		electrodes = load_cfl(argv[7], N, mask_electrodes_dims);
		for (int i = 0; i < N; i++)
			assert(mask_electrodes_dims[i] == bdims[i]);
	}
	if (argc >= 10) {
		l2weights = load_cfl(argv[8], N, l2weights_dims);
		for (int i = 0; i < N; i++)
			assert(l2weights_dims[i] == bdims[i]);
	}
	md_copy_dims(N, jdims, bdims);
	jdims[0] = 3;
	complex float *j = create_cfl(argv[argc - 1], N, jdims);

	md_zsmul(4, bdims, b, b, 1. / Hz_per_Tesla / Mu_0);
	cdi_reco(vox, jdims, j, bdims, b, mask, tik_reg, iter, admm_iter, tolerance, bc_mask, electrodes, bc_reg, div_reg, div_pf, div_order, leray_iter, outer_hist, leray_hist, solver, l2weights);
	md_zsmul(4, jdims, j, j, 1. / bz_unit(bdims + 1, vox));

	unmap_cfl(N, bdims, b);
	unmap_cfl(N, jdims, j);
	if (mask != NULL)
		unmap_cfl(N, bdims, mask);
	if (bc_mask != NULL)
		unmap_cfl(N, jdims, bc_mask);
	return 0;
}
