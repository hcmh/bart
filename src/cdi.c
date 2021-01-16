#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/ops.h"
#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "num/gpuops.h"

#include "iter/iter.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/types.h"
#include "misc/opts.h"

#include "simu/biot_savart_fft.h"
#define N 4

static void cdi_reco(const float fov[3], const long jdims[N], complex float* j, const long bdims[N], const complex float* bz, const complex float* mask, const float reg, const float div_scale, const complex float* div_mask, unsigned int iter, float tol, const complex float* bc_mask)
{
	complex float* adj = md_alloc_sameplace(N, jdims, CFL_SIZE, j);
	auto bz_op = linop_bz_create(jdims, fov);

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
	const unsigned long flags = 14;
	auto div_op = linop_div_create(N, jdims, 0, flags);

	if(NULL!=bc_mask) {
		auto mask_op = linop_cdiag_create(N, jdims, 15, bc_mask);
		div_op = linop_chain_FF(mask_op, div_op);
	}

	float div_weight = pow(div_scale, 0.5);
	complex float scale[] = { div_weight * jdims[1] / fov[0] / 2, div_weight * jdims[2] / fov[1] / 2, div_weight * jdims[3] / fov[2] / 2 };
	auto scale_op = linop_cdiag_create(N, jdims, MD_BIT(0), scale);
	div_op = linop_chain_FF(scale_op, div_op);

	if(NULL!=div_mask) {
		auto mask_op = linop_cdiag_create(N, bdims, 15, div_mask);
		div_op = linop_chain_FF(div_op, mask_op);
	}

	auto joined = operator_plus_create(bz_op->normal, div_op->normal);

	md_zfill(N, jdims, j, 1);

	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = iter;
	conf.l2lambda = reg;
	conf.tol = tol;

	long size = 2 * md_calc_size(N, jdims); // multiply by 2 for float size
	iter_conjgrad(CAST_UP(&conf), joined, NULL, size, (float*)j, (const float*)adj, NULL);

	linop_free(bz_op);
	linop_free(div_op);
	operator_free(joined);
	md_free(adj);
}


static const char usage_str[] = "fovX fovY fovZ <bz> [<mask> [<div_mask> [<boundary condition mask>]]] <j>";
static const char help_str[] = "Estimate the current density j that generates bz\n";


int main_cdi(int argc, char* argv[])
{
	float tik_reg = 0, div_scale = 0, tolerance = 1e-3;
	unsigned int iter = 100;
	bool use_gpu = false;
	const struct opt_s opts[] = {
		OPT_FLOAT('l', &tik_reg, "lambda_1", "Tikhonov Regularization"),
		OPT_FLOAT('t', &tolerance, "t", "Stopping Tolerance"),
		OPT_FLOAT('d', &div_scale, "lambda_2", "Divergence weighting factor"),
		OPT_UINT('n', &iter, "Iterations", "Max. number of iterations"),
		OPT_SET('g', &use_gpu, "use gpu"),
	};
	cmdline(&argc, argv, 5, 8, usage_str, help_str, ARRAY_SIZE(opts), opts);
	(use_gpu ? num_init_gpu_memopt : num_init)();

	const int fov_ind = 1;
	float fov[3] = {strtof(argv[fov_ind], NULL), strtof(argv[fov_ind + 1], NULL), strtof(argv[fov_ind + 2], NULL)};
	long bdims[N], jdims[N];

	complex float* b = load_cfl(argv[4], N, bdims);

	complex float* mask = NULL, * div_mask = NULL, * bc_mask = NULL;
	if(argc>=7)
		mask = load_cfl(argv[5], N, bdims);
	if(argc>=8)
		div_mask = load_cfl(argv[6], N, bdims);
	if(argc==9)
		bc_mask = load_cfl(argv[7], N, jdims);

	md_copy_dims(N, jdims, bdims);
	jdims[0] = 3;
	complex float* j = create_cfl(argv[argc-1], N, jdims);
	complex float* jx = j;
#ifdef  USE_CUDA
	if (use_gpu) {

		cuda_use_global_memory();
		complex float* j_gpu = md_alloc_gpu(N, jdims, CFL_SIZE);
		jx = j_gpu;
	}
#endif
	cdi_reco(fov, jdims, jx, bdims, b, mask, tik_reg, div_scale, div_mask, iter, tolerance, bc_mask);
#ifdef  USE_CUDA
	if (use_gpu) {
		md_copy(N, jdims, j, jx, CFL_SIZE);
		md_free(jx);
	}
#endif
	unmap_cfl(N, bdims, b);
	unmap_cfl(N, jdims, j);
	if(mask != NULL)
		unmap_cfl(N, bdims, mask);
	if(div_mask != NULL)
		unmap_cfl(N, bdims, div_mask);
	if(bc_mask != NULL)
		unmap_cfl(N, jdims, bc_mask);
	return 0;
}
