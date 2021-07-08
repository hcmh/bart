#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "noncart/nufft.h"
#include "linops/linop.h"

#include "nlops/nlop.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noir/recon2.h"
#include "noir/model_net.h"

#include "nn/weights.h"

#include "networks/nlinvnet.h"

#include "noir/misc.h"

#ifdef USE_CUDA
#include "num/gpuops.c"
#endif


static const char help_str[] =
		"";



int main_nlinvnet(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* out_file = NULL;
	const char* weight_file = NULL;
	const char* sens_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INOUTFILE(true, &weight_file, "weights"),
		ARG_INOUTFILE(true, &out_file, "output/reference"),
		ARG_OUTFILE(false, &sens_file, "sensitivities"),
	};

	const char* pat_file = NULL;
	const char* traj_file = NULL;

	struct noir2_conf_s conf = noir2_defaults;
	struct nlinvnet_s nlinvnet = nlinvnet_config_opts;
	nlinvnet.conf = &conf;

	float coil_os = -1;

	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_SET('g', &(nlinvnet.gpu), "use gpu"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPTL_FLOAT(0, "alpha", &conf.alpha, "", "(minimum for regularization)"),
		OPT_INFILE('p', &pat_file, "", "sampling pattern"),
		OPTL_FLOAT(0, "coil-os", &coil_os, "val", "(over-sampling factor for sensitivities)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	(nlinvnet.gpu ? num_init_gpu_memopt : num_init)();
	#ifdef USE_CUDA
	if (nlinvnet.gpu)
		cuda_use_global_memory();
	#endif

	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);
	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}

	assert(1 == ksp_dims[MAPS_DIM]);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	long dims[DIMS];

	long trj_dims[DIMS];
	complex float* traj  = NULL;

	if (NULL != traj_file) {

		conf.noncart = true;

		traj = load_cfl(traj_file, DIMS, trj_dims);

		estimate_im_dims(DIMS, FFT_FLAGS, dims, trj_dims, traj);
		debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);

		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);
	} else {

		md_copy_dims(DIMS, dims, ksp_dims);
	}
	if (-1 == coil_os)
		coil_os = conf.noncart ? 2 : 1;

	dims[MAPS_DIM] = 1;

	long sens_dims[DIMS];
	md_select_dims(DIMS, ~0, sens_dims, dims);

	for (int i = 0; i < 3; i++)
		if (1 != sens_dims[i])
			sens_dims[i] = lround(coil_os * sens_dims[i]);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, dims);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

	nlinvnet_init_resnet_default(&nlinvnet);

	long col_dims_s[DIMS];
	long img_dims_s[DIMS];
	long cim_dims_s[DIMS];
	long msk_dims_s[DIMS];
	long ksp_dims_s[DIMS];
	long pat_dims_s[DIMS];

	md_select_dims(DIMS, ~BATCH_FLAG, col_dims_s, sens_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, img_dims_s, img_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, cim_dims_s, cim_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, msk_dims_s, msk_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, ksp_dims_s, ksp_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, pat_dims_s, pat_dims);

	nlinvnet_init_model_cart(&nlinvnet, DIMS,
		pat_dims_s, pattern,
		MD_SINGLETON_DIMS(DIMS), NULL,
		msk_dims_s, NULL,
		ksp_dims_s,
		cim_dims_s,
		img_dims_s,
		col_dims_s);

	int Nb = 10;

	long cim_dims2[DIMS];
	complex float* ref = load_cfl(out_file, DIMS, cim_dims2);
	assert(md_check_equal_dims(DIMS, cim_dims, cim_dims2, ~0));
	train_nlinvnet(&nlinvnet, DIMS, Nb, cim_dims, ref, ksp_dims, kspace, cim_dims, ref, ksp_dims, kspace);

	dump_nn_weights(weight_file, nlinvnet.weights);

	unmap_cfl(DIMS, cim_dims, ref);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, ksp_dims, kspace);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	exit(0);
}


