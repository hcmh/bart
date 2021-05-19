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
#include "iter/lsqr.h"
#include "iter/monitor.h"
#include "iter/prox.h"
#include "iter/thresh.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/types.h"

#include "simu/biot_savart_fft.h"
#include "simu/fd_geometry.h"
#include "simu/leray.h"
#define N 4
#define NPROX 3

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

	linop_leray_calc_projection(data->leray_data, data->j_hist, (const complex float *)phi);

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
		return (complex float *)phi;
	}
}



static void get_conductive_volume(const long bdims[N], complex float *out, const complex float *interior, const complex float *electrodes)
{
	complex float *electrodes_indicator = md_alloc(N, bdims, CFL_SIZE);
	md_zabs(N, bdims, electrodes_indicator, electrodes);
	md_zsgreatequal(N, bdims, electrodes_indicator, electrodes_indicator, .9);

	assert(md_zscalar_real(N, bdims, electrodes_indicator, interior) < 1e-8);

	md_zadd(N, bdims, out, interior, electrodes_indicator);
}

//Remove normal component of the current where no electrodes are attached
static struct linop_s *make_wall_op(const long jdims[N], const long bdims[N], const complex float *interior, const complex float *electrodes)
{

	long bstrs[N], jstrs[N];
	md_calc_strides(N, bstrs, bdims, CFL_SIZE);
	md_calc_strides(N, jstrs, jdims, CFL_SIZE);

	complex float *conductive_volume = md_alloc(N, bdims, CFL_SIZE);
	get_conductive_volume(bdims, conductive_volume, interior, electrodes);

	complex float *walls = md_alloc(N, jdims, CFL_SIZE);
	calc_outward_normal(N, jdims, walls, 0, bdims, conductive_volume);

	for (int i = 0; i < 3; i++)
		md_zmul2(N, bdims, jstrs, (void *)walls + i * jstrs[0], jstrs, (void *)walls + i * jstrs[0], bstrs, interior);

	md_zabs(N, jdims, walls, walls);
	md_zsgreatequal(N, jdims, walls, walls, 0.1);

	md_zsmul(N, jdims, walls, walls, -1);
	md_zsadd(N, jdims, walls, walls, 1);
	auto wall_op = linop_cdiag_create(N, jdims, 15, walls);

		char *str = getenv("DEBUG_LEVEL");
		debug_level = (NULL != str) ? atoi(str) : DP_INFO;
		if (5 <= debug_level)
			dump_cfl("DEBUG_walls", N, jdims, walls);

	md_free(conductive_volume);
	md_free(walls);
	return wall_op;
}


//Remove current outside the conductive domain
static struct linop_s *make_mask_c_op(const long jdims[N], const long bdims[N], const complex float *interior, const complex float *electrodes)
{

	long bstrs[N], jstrs[N];
	md_calc_strides(N, bstrs, bdims, CFL_SIZE);
	md_calc_strides(N, jstrs, jdims, CFL_SIZE);

	complex float *conductive_volume = md_alloc(N, bdims, CFL_SIZE);
	get_conductive_volume(bdims, conductive_volume, interior, electrodes);

	auto wall_op = linop_cdiag_create(N, jdims, 14, conductive_volume);

		char *str = getenv("DEBUG_LEVEL");
		debug_level = (NULL != str) ? atoi(str) : DP_INFO;
		if (5 <= debug_level)
			dump_cfl("DEBUG_conductive", N, bdims, conductive_volume);

	md_free(conductive_volume);
	return wall_op;

}

static void cdi_reco(const float vox[3], const long jdims[N], complex float *j, const long bdims[N], const complex float *bz, const complex float *mask, const float reg, int iter, int admm_iter, float tol, const complex float *interior, const complex float* electrodes, const float bc_reg, const float div_reg, const enum PROXFUN div_pf, const int div_order, const int leray_iter, const unsigned long outer_hist, const unsigned long leray_hist, const enum SOLVER solver, const complex float* l2_weights, const char *outname)
{

	//Monitoring
	long name_len = strlen(outname);
	char mon1_name[name_len + 6];
	sprintf(mon1_name, "%s_step", outname);
	char mon2_name[name_len + 12];
	sprintf(mon2_name, "%s_leray_step", outname);

	complex float *j_hist = md_calloc(N, jdims, CFL_SIZE);
	struct history_data_s history_data = {.hist_1 = outer_hist, .hist_2 = leray_hist, .j_hist = j_hist, .j_dims = jdims, .leray_data = NULL, .post_j_op = NULL};
	auto mon1 = create_monitor_recorder(N, jdims, mon1_name, (void *)&history_data, history_select_1, history_save_1);
	auto mon2 = create_monitor_recorder(N, jdims, mon2_name, (void *)&history_data, history_select_2, history_save_2);

	//forward operator
	auto bz_op = linop_bz_create(jdims, vox);
	const struct linop_s *mask_c_op = NULL, *leray_op = NULL;

	//Mask regions with low signal
	if (NULL != mask) {
		auto mask_s_op = linop_cdiag_create(N, bdims, 15, mask);
		bz_op = linop_chain_FF(bz_op, mask_s_op);
	}

	//Mask current density
	if (NULL != interior) {
		mask_c_op = make_mask_c_op(jdims, bdims, interior, electrodes);

		if (PF_ind == div_pf) {
			complex float *conductive_volume = md_alloc(N, bdims, CFL_SIZE);
			get_conductive_volume(bdims, conductive_volume, interior, electrodes);
			leray_op = linop_leray_create(N, jdims, 0, leray_iter, div_reg, conductive_volume, mon2);
			history_data.leray_data = linop_get_data(leray_op);
			md_free(conductive_volume);
		}

		if (0 < bc_reg)
			mask_c_op = linop_chain_FF(mask_c_op, make_wall_op(jdims, bdims, interior, electrodes));
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

		auto projection_op = linop_identity_create(N, jdims);
		if (NULL != mask_c_op)
			projection_op = linop_chain(projection_op, mask_c_op);

		if ((0 < div_reg) && (div_pf == PF_ind)) {
			projection_op = linop_chain(projection_op, leray_op);
			projection_op = linop_chain(projection_op, mask_c_op); //keep symmetric!
		}

		complex float *adj = md_calloc(N, jdims, CFL_SIZE);
		auto adj_op = linop_chain(projection_op, bz_op);
		linop_adjoint(adj_op, N, jdims, adj, N, bdims, bz);
		linop_free(adj_op);

		auto normal_op = linop_get_normal(bz_op);

		if ((0 < div_reg) && (div_pf == PF_l2)) {
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

		normal_op = linop_chain_FF(normal_op, linop_get_adjoint(projection_op));

		complex float *l2_diag = md_alloc(N, bdims, CFL_SIZE);
		if (NULL == l2_weights)
			md_zfill(N, jdims, l2_diag, reg);
		else
			md_zsmul(N, bdims, l2_diag, l2_weights, reg);
		auto l2_op = linop_cdiag_create(N, jdims, 14, l2_diag);

		normal_op = linop_plus_FF(normal_op, l2_op);

		long size = 2 * md_calc_size(N, jdims); // multiply by 2 for float size
		iter_conjgrad(CAST_UP(&conf), normal_op->forward, NULL, size, (float *)j, (const float *)adj, mon1);

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

		long nprox = 0;
		const struct operator_p_s *prox_funs[NPROX] = {NULL};
		const struct linop_s *prox_linops[NPROX] = {NULL};

		if (0 < bc_reg) {
			assert(NPROX >= ++nprox);
			assert(NULL != mask_c_op);
			prox_funs[nprox - 1] = prox_indicator_create(mask_c_op);
			complex float one[1] = {1.};
			prox_linops[nprox - 1] = linop_cdiag_create(N, jdims, 0, one);
		}

		if (0 < div_reg) {
			assert(NPROX >= ++nprox);

			//FIXME: Scale j with voxelsize before applying derivative
			assert((vox[0] == vox[1]) && (vox[0] == vox[2]) && (vox[1] == vox[2]));
			auto div_op = linop_div_create(N, jdims, 0, 14, div_order, BC_SAME);

			if (div_pf == PF_l2) {
				prox_funs[nprox - 1] = prox_l2norm_create(N, bdims, div_reg);
				prox_linops[nprox - 1] = linop_clone(div_op);
			} else if (div_pf == PF_thresh) {
				prox_funs[nprox - 1] = prox_thresh_create(N, bdims, div_reg, 0);
				prox_linops[nprox - 1] = linop_clone(div_op);
			} else if (div_pf == PF_ind) {
				assert(NULL != leray_op);
				prox_funs[nprox - 1] = prox_indicator_create(leray_op);
				prox_linops[nprox - 1] = linop_identity_create(N, jdims);
			} else {
				assert(false);
			}
			linop_free(div_op);
		}

		if (NULL == l2_weights) {
			lconf.lambda = reg;
		} else {
			assert(NPROX >= ++nprox);
			prox_funs[nprox - 1] = prox_l2norm_create(N, jdims, div_reg);
			complex float *l2_diag = md_alloc(N, bdims, CFL_SIZE);
			md_zsmul(N, bdims, l2_diag, l2_weights, reg);
			prox_linops[nprox - 1] = linop_cdiag_create(N, jdims, 14, l2_diag);
			md_free(l2_diag);
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
	if (NULL != mask_c_op)
		linop_free(mask_c_op);
	if (NULL != leray_op)
		linop_free(leray_op);
	md_free(j_hist);
}


static const char usage_str[] = "voxelsize(x) voxelsize(y) voxelsize(z) <B_z> [<mask_signal> [<interior> <electrodes> [<l2_weighting>]]] <J>";
static const char help_str[] = "Estimate the current density J (A/[voxelsize]^2) that generates B_z (Hz)\n";


int main_cdi(int argc, char *argv[])
{
	float tik_reg = 0, tolerance = 1e-3, bc_reg = -1, div_reg = -1;
	int iter = 100, admm_iter = 100, div_order = 1, leray_iter = 20;
	long leray_hist = -1, outer_hist = -1;
	enum SOLVER solver = CG;
	enum PROXFUN div_pf = PF_ind;

	struct opt_s solveropt[] = {
		OPT_SELECT('c', enum SOLVER, &solver, CG, 	"Use Conjugate Gradient Solver"),
		OPT_SELECT('a', enum SOLVER, &solver, ADMM,  	"Use ADMM")
	};

	struct opt_s divopt[] = {
		OPT_SELECT('0', enum PROXFUN, &div_pf, PF_l2, 		"Solve ||Ax - y||_2^2 + d||∇x||_2^2"),
		OPT_SELECT('1', enum PROXFUN, &div_pf, PF_thresh,  	"Solve ||Ax - y||_2^2 + d||∇x||_1"),
		OPT_SELECT('2', enum PROXFUN, &div_pf, PF_ind,  	"Solve ||Ax - y||_2^2 \n s.t. ∇x = 0")
	};

	const struct opt_s opts[] = {
	    OPT_FLOAT('l', &tik_reg, "lambda_1", "Tikhonov Regularization"),
	    OPT_FLOAT('b', &bc_reg, "b", "Boundary current penalty"),
	    OPT_FLOAT('d', &div_reg, "d", "Divergence penalty or LeRay Regularization"),

	    OPT_SUBOPT('D', "", "How to enforce Divergence Freeness, -Dh for help", ARRAY_SIZE(divopt), divopt),
	    OPT_SUBOPT('S', "", "Solver/Strategy, -Sh for help", ARRAY_SIZE(solveropt), solveropt),

	    OPT_INT('o', &div_order, "", "Finite difference order for divergence calculation (1,2) (only -D0, -D1)"),
	    OPT_INT('p', &leray_iter, "", "Number of Iterations for LeRay Projection"),
	    OPT_FLOAT('t', &tolerance, "t", "Stopping Tolerance"),
	    OPT_INT('n', &iter, "Iterations", "Max. number of cg iterations"),
	    OPT_INT('m', &admm_iter, "Iterations", "Max. number of ADMM iterations"),
	    OPT_LONG('A', &outer_hist, "n_1", "Save J every n_1 steps (outer iteration)"),
	    OPT_LONG('B', &leray_hist, "n_2", "Save J every n_2 LeRay-Steps (inner iteration)"),
	};
	cmdline(&argc, argv, 5, 9, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int vox_ind = 1;
	float vox[3] = {strtof(argv[vox_ind], NULL), strtof(argv[vox_ind + 1], NULL), strtof(argv[vox_ind + 2], NULL)};
	long bdims[N], jdims[N], mask_dims[N], mask_bc_dims[N], mask_electrodes_dims[N], l2_weights_dims[N];
	complex float *mask = NULL, *interior = NULL, *electrodes = NULL, *l2_weights = NULL;

	complex float *b = load_cfl(argv[4], N, bdims);
	assert(bdims[0] == 1);

	if (argc >= 7) {
		mask = load_cfl(argv[5], N, mask_dims);
		for (int i = 0; i < N; i++)
			assert(mask_dims[i] == bdims[i]);
	}

	if (argc >= 8) {
		interior = load_cfl(argv[6], N, mask_bc_dims);
		for (int i = 0; i < N; i++)
			assert(mask_bc_dims[i] == bdims[i]);
		assert(argc >= 9);
		electrodes = load_cfl(argv[7], N, mask_electrodes_dims);
		for (int i = 0; i < N; i++)
			assert(mask_electrodes_dims[i] == bdims[i]);
	}
	if (argc >= 10) {
		l2_weights = load_cfl(argv[8], N, l2_weights_dims);
		for (int i = 0; i < N; i++)
			assert(l2_weights_dims[i] == bdims[i]);
	}
	md_copy_dims(N, jdims, bdims);
	jdims[0] = 3;
	complex float *j = create_cfl(argv[argc - 1], N, jdims);

	md_zsmul(4, bdims, b, b, 1. / Hz_per_Tesla / Mu_0);
	cdi_reco(vox, jdims, j, bdims, b, mask, tik_reg, iter, admm_iter, tolerance, interior, electrodes, bc_reg, div_reg, div_pf, div_order, leray_iter, outer_hist, leray_hist, solver, l2_weights, argv[argc - 1]);
	md_zsmul(4, jdims, j, j, 1. / bz_unit(bdims + 1, vox));

	unmap_cfl(N, bdims, b);
	unmap_cfl(N, jdims, j);
	if (mask != NULL)
		unmap_cfl(N, bdims, mask);
	if (interior != NULL)
		unmap_cfl(N, jdims, interior);
	return 0;
}
