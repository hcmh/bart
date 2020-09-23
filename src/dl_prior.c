#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/rand.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/misc.h"
#include "iter/monitor.h"
#include "iter/prox.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/sampling.h"

#include "nlops/nlop.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"

#include "nn/tf_wrapper_prox.h"

#include "noncart/nufft.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

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

static const char usage_str[] = "<graph file> <image> <output>";
static const char help_str[] = "Denoise image via prior";

int main_dl_prior(int argc, char* argv[])
{
	
	float lambda = 0;
	int   iter = 1;
	float noise_level = 0.05;
	struct opt_reg_s ropts;
	opt_reg_init(&ropts);
	
	const struct opt_s opts[] = {
		OPT_FLOAT('l', &lambda, "lambda", "step size"),
		OPT_INT('i', &iter, "iter", "iterations"),
		OPT_FLOAT('n', &noise_level, "noise", "noise level"),
	};
	
	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);	

	long img_dims[DIMS];
	complex float* img = load_cfl(argv[2], DIMS, img_dims);
	struct nlop_s * tf_ops = nlop_tf_create(1, 1, argv[1]);
	struct operator_p_s* op = prox_logp_create(2, img_dims, tf_ops);
	auto dom = operator_p_domain(op);
	auto cod = operator_p_codomain(op);

	complex float* out = create_cfl(argv[3], DIMS, img_dims);
	complex float* tmp = md_alloc(DIMS, img_dims, CFL_SIZE);

	complex float* noise = md_alloc(DIMS, img_dims, CFL_SIZE);

	md_gaussian_rand(DIMS, img_dims, noise);
	md_zsmul(DIMS, img_dims, noise, noise, noise_level);
	md_zadd(DIMS, img_dims, tmp, img, noise);

	for (int i = 0; i < iter; i++)
	{
		printf("%d\n", i);
		operator_p_apply(op, lambda, cod->N, cod->dims, tmp, dom->N, dom->dims, tmp);
	}

	md_copy(DIMS, img_dims, out, tmp, CFL_SIZE);
	unmap_cfl(DIMS, img_dims, out);
	unmap_cfl(DIMS, img_dims, img);
	
    return 0;
}
