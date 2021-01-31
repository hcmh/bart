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
#include "iter/prox.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/types.h"

#include "simu/biot_savart_fft.h"
#define N 4
#define NPROX 2

static void cdi_reco(const float vox[3], const long jdims[N], complex float *j, const long bdims[N], const complex float *bz, const complex float *mask, const float reg, unsigned int iter, float tol, const complex float *bc_mask, const float bc_reg, const float div_reg)
{
	complex float *adj = md_alloc_sameplace(N, jdims, CFL_SIZE, j);
	auto bz_op = linop_bz_create(jdims, vox);
	const struct linop_s *bc_mask_op = NULL;

	complex float *walls = NULL;
	if (NULL != bc_mask) {
		// Create multiplicative mask - to enforce 0 current outside
		// FIXME: Don't copy; instead create modified cdiag operator?
		complex float *bc_mask3 = md_alloc(N, jdims, CFL_SIZE);
		long pos[N] = {0};
		for (; pos[0] < 3; pos[0]++)
			md_copy_block(N, pos, jdims, bc_mask3, bdims, bc_mask, CFL_SIZE);
		bc_mask_op = linop_cdiag_create(N, jdims, 15, bc_mask3);
		bz_op = linop_chain(bc_mask_op, bz_op);
		md_free(bc_mask3);

		if (0 < bc_reg) {
			// Create "walls" - enforce 0 current through the walls
			walls = md_alloc(N, jdims, CFL_SIZE);
			auto d_op = linop_fd_create(N, bdims, 0, 14, 2, BC_SAME, false);
			linop_forward(d_op, N, jdims, walls, N, bdims, bc_mask);
			linop_free(d_op);
		}
	}
	if (NULL != mask) {
		auto mask_op = linop_cdiag_create(N, bdims, 15, mask);
		bz_op = linop_chain_FF(bz_op, mask_op);
	}
	// bz_op -> A
	// j -> x
	// bz -> y
	// mask -> M
	// mask_bc -> M_b
	// min || M (A M_b x - y) ||^2
	linop_adjoint(bz_op, N, jdims, adj, N, bdims, bz);

	// lsqr2 does not use this as initial value!
	md_zfill(N, jdims, j, 0);

	struct lsqr_conf lconf = lsqr_defaults;

	enum SOLVER { CG, ADMM } solver = ADMM;

	switch (solver) {
	case CG: {
		struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
		conf.maxiter = iter;
		conf.l2lambda = reg;
		conf.tol = tol;

		lsqr2(N, &lconf, iter2_conjgrad, CAST_UP(&conf), bz_op, 0, NULL, NULL, jdims, j, bdims, bz, NULL, NULL);
	} break;
	case ADMM: {
		struct iter_admm_conf conf = iter_admm_defaults;
		conf.maxiter = 10;
		conf.maxitercg = iter;
		lconf.lambda = reg;

		long nprox = 0;
		const struct operator_p_s *prox_funs[NPROX] = {NULL};
		const struct linop_s *prox_linops[NPROX] = {NULL};

		if (0 < bc_reg) {
			assert(NULL != walls);
			assert(NPROX >= ++nprox);
			prox_funs[nprox - 1] = prox_l2norm_create(N, jdims, bc_reg);
			prox_linops[nprox - 1] = linop_cdiag_create(N, jdims, 15, walls);
		}

		if (0 < div_reg) {
			assert(NPROX >= ++nprox);
			assert(NULL != bc_mask_op);
			//FIXME: Scale j with voxelsize before applying derivative
			assert((vox[0] == vox[1]) && (vox[0] == vox[2]) && (vox[1] == vox[2]));
			prox_funs[nprox - 1] = prox_l2norm_create(N, bdims, div_reg);

			auto div_op = linop_div_create(N, jdims, 0, 14, 1, BC_SAME);
			prox_linops[nprox - 1] = linop_chain(bc_mask_op, div_op);

			linop_free(div_op);
		}

		lsqr2(N, &lconf, iter2_admm, CAST_UP(&conf), bz_op, nprox, prox_funs, prox_linops, jdims, j, bdims, bz, NULL, NULL);

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
	if (walls != NULL)
		md_free(walls);
	md_free(adj);
}


static const char usage_str[] = "voxelsize(x) voxelsize(y) voxelsize(z) <B_z> [<mask> [<boundary condition mask>]] <J>";
static const char help_str[] = "Estimate the current density J (A/[voxelsize]^2) that generates B_z (Hz)\n";


int main_cdi(int argc, char *argv[])
{
	float tik_reg = 0, tolerance = 1e-3, bc_reg = -1, div_reg = -1;
	unsigned int iter = 100;
	const struct opt_s opts[] = {
	    OPT_FLOAT('l', &tik_reg, "lambda_1", "Tikhonov Regularization"),
	    OPT_FLOAT('b', &bc_reg, "b", "Boundary current penalty"),
	    OPT_FLOAT('d', &div_reg, "d", "Divergence penalty"),
	    OPT_FLOAT('t', &tolerance, "t", "Stopping Tolerance"),
	    OPT_UINT('n', &iter, "Iterations", "Max. number of iterations"),
	};
	cmdline(&argc, argv, 5, 7, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int vox_ind = 1;
	float vox[3] = {strtof(argv[vox_ind], NULL), strtof(argv[vox_ind + 1], NULL), strtof(argv[vox_ind + 2], NULL)};
	long bdims[N], jdims[N], mask_dims[N], mask_bc_dims[N];
	complex float *mask = NULL, *bc_mask = NULL;

	complex float *b = load_cfl(argv[4], N, bdims);
	assert(bdims[0] == 1);

	if (argc >= 7) {
		mask = load_cfl(argv[5], N, mask_dims);
		for (int i = 0; i < N; i++)
			assert(mask_dims[i] == bdims[i]);
	}

	if (argc == 8) {
		bc_mask = load_cfl(argv[6], N, mask_bc_dims);
		for (int i = 0; i < N; i++)
			assert(mask_bc_dims[i] == bdims[i]);
	}

	md_copy_dims(N, jdims, bdims);
	jdims[0] = 3;
	complex float *j = create_cfl(argv[argc - 1], N, jdims);

	md_zsmul(4, bdims, b, b, 1. / Hz_per_Tesla / Mu_0);
	cdi_reco(vox, jdims, j, bdims, b, mask, tik_reg, iter, tolerance, bc_mask, bc_reg, div_reg);
	md_zsmul(4, jdims, j, j, 1. / bz_unit(bdims + 1, vox));

	unmap_cfl(N, bdims, b);
	unmap_cfl(N, jdims, j);
	if (mask != NULL)
		unmap_cfl(N, bdims, mask);
	if (bc_mask != NULL)
		unmap_cfl(N, jdims, bc_mask);
	return 0;
}
