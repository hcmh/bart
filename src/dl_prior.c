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

#include "nn/tf_wrapper.h"

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
    
	
	float lambda=0;
	int iter = 1;
	struct opt_reg_s ropts;
	opt_reg_init(&ropts);
	
	const struct opt_s opts[] = {
		OPT_FLOAT('l', &lambda, "lambda", "step size"),
		OPT_INT('i', &iter, "iter", "iterations"),
	};
	
	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);	

	
	
	long img_dims[DIMS];
	complex float* img = load_cfl(argv[2], DIMS, img_dims);
	char noisy_file[100];
	sprintf(noisy_file, "%s_n", argv[2]);

	struct operator_p_s* op = prox_logp_create(DIMS, img_dims, argv[1]);

	complex float* noisy_img = create_cfl(noisy_file, DIMS, img_dims);
	
	complex float* noise = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, img);

	md_gaussian_rand(DIMS, img_dims, noise);
	md_zsmul(DIMS, img_dims, noise, noise, 0.0);
	md_zadd(DIMS, img_dims, noisy_img, img, noise);


	// slice image
	long slice_dims[DIMS];
	md_set_dims(DIMS, slice_dims, 1);

	slice_dims[0] = 128;
	slice_dims[1] = 128;
	slice_dims[2] = 4;

	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);

	complex float* slices = md_alloc(DIMS, slice_dims, CFL_SIZE);
	complex float* tmp_slices = md_alloc(DIMS, slice_dims, CFL_SIZE);
	
	int blocks = 0;

	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{
			pos[0] = i*128;
			pos[1] = j*128;
			md_copy_block(2, pos, slice_dims, tmp_slices+blocks*128*128, img_dims, noisy_img, CFL_SIZE);
			blocks = blocks + 1;
		}
	}

	md_transpose(DIMS, 1, 0, slice_dims, slices, slice_dims, tmp_slices, CFL_SIZE);
	
	complex float* tmp = md_alloc(DIMS, slice_dims, CFL_SIZE);
	
	auto dom = operator_p_domain(op);
	auto cod = operator_p_codomain(op);

	for (int i = 0; i < iter; i++)
	{
		printf("%d\n", i);
		operator_p_apply(op, lambda, cod->N, cod->dims, slices, dom->N, dom->dims, slices);
	}
	

	// stitch image back
	blocks = 0;

	complex float* out = create_cfl(argv[3], DIMS, img_dims);
	complex float* tmp_out = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, out);

	for(size_t i=0; i < 2; i++)
	{
		for (size_t j=0; j < 2; j++)
		{
			pos[0] = j*128;
			pos[1] = i*128;
			md_copy_block(2, pos, img_dims, tmp_out, slice_dims, slices+blocks*128*128, CFL_SIZE);
			blocks = blocks + 1;
		}
	}
	
	md_transpose(DIMS, 1, 0, img_dims, out, img_dims, tmp_out, CFL_SIZE);

	unmap_cfl(DIMS, img_dims, img);
	unmap_cfl(DIMS, img_dims, out);

	
    return 0;
}
