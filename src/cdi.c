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

#include "num/gpuops.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/types.h"

#include "simu/biot_savart_fft.h"
#define N 4


static void cdi_reco(const float vox[3], const long jdims[N], complex float *j, const long bdims[N], const complex float *bz, const complex float *mask, const float reg, unsigned int iter, float tol, const complex float *bc_mask)
{
	complex float *adj = md_alloc_sameplace(N, jdims, CFL_SIZE, j);
	auto bz_op = linop_bz_create(jdims, vox);

	if (NULL != bc_mask) {
		auto mask_op = linop_cdiag_create(N, jdims, 15, bc_mask);
		bz_op = linop_chain_FF(mask_op, bz_op);
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
	iter_conf *iconf;
	italgo_fun2_t ifun;

	enum SOLVER { CG, ADMM } solver = ADMM;
	struct iter_conjgrad_conf cg_conf = iter_conjgrad_defaults;
	struct iter_admm_conf admm_conf = iter_admm_defaults;
	switch (solver) {
	case CG: ;
		cg_conf.maxiter = iter;
		cg_conf.l2lambda = reg;
		cg_conf.tol = tol;
		iconf = CAST_UP(&cg_conf);
		ifun = iter2_conjgrad;
	break;
	case ADMM: ;
		admm_conf.maxiter = 3;
		admm_conf.maxitercg = iter;
		lconf.lambda = reg;
		iconf = CAST_UP(&admm_conf);
		ifun = iter2_admm;
	break;
	default:
		assert(false);
	}

	lsqr2(N, &lconf, ifun, iconf, bz_op, 0, NULL, NULL, jdims, j, bdims, bz, NULL, NULL);

	linop_free(bz_op);
	md_free(adj);
}


static const char usage_str[] = "voxelsize(x) voxelsize(y) voxelsize(z) <bz> [<mask> [<boundary condition mask>]] <j>";
static const char help_str[] = "Estimate the current density J (A/mm^2) that generates ΔB (Hz)\n";


int main_cdi(int argc, char *argv[])
{
	float tik_reg = 0, tolerance = 1e-3;
	unsigned int iter = 100;
	const struct opt_s opts[] = {
	    OPT_FLOAT('l', &tik_reg, "lambda_1", "Tikhonov Regularization"),
	    OPT_FLOAT('t', &tolerance, "t", "Stopping Tolerance"),
	    OPT_UINT('n', &iter, "Iterations", "Max. number of iterations"),
	};
	cmdline(&argc, argv, 5, 7, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int vox_ind = 1;
	float vox[3] = {strtof(argv[vox_ind], NULL), strtof(argv[vox_ind + 1], NULL), strtof(argv[vox_ind + 2], NULL)};
	long bdims[N], jdims[N];

	complex float *b = load_cfl(argv[4], N, bdims);

	complex float *mask = NULL, *bc_mask = NULL;
	if (argc >= 7)
		mask = load_cfl(argv[5], N, bdims);
	if (argc == 8)
		bc_mask = load_cfl(argv[6], N, jdims);

	md_copy_dims(N, jdims, bdims);
	jdims[0] = 3;
	complex float *j = create_cfl(argv[argc - 1], N, jdims);

	//complex float *b_scaled = md_alloc(N, bdims, CFL_SIZE);
	md_zsmul(4, bdims, b, b, 1. / Hz_per_Tesla / Mu_0);
	cdi_reco(vox, jdims, j, bdims, b, mask, tik_reg, iter, tolerance, bc_mask);
	md_zsmul(4, jdims, j, j, 1./bz_unit(bdims+1, vox)/1.e6);

	//md_free(b_scaled);

	unmap_cfl(N, bdims, b);
	unmap_cfl(N, jdims, j);
	if (mask != NULL)
		unmap_cfl(N, bdims, mask);
	if (bc_mask != NULL)
		unmap_cfl(N, jdims, bc_mask);
	return 0;
}
