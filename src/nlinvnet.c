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

#include "noir/misc.h"





static const char help_str[] =
		"Jointly estimate image and sensitivities with nonlinear\n"
		"inversion using {iter} iteration steps. Optionally outputs\n"
		"the sensitivities.";



int main_nlinvnet(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* img_file = NULL;
	const char* sens_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_OUTFILE(true, &img_file, "output"),
		ARG_OUTFILE(false, &sens_file, "sensitivities"),
	};

	const char* pat_file = NULL;
	const char* traj_file = NULL;

	int nmaps = 1;

	struct noir2_conf_s conf = noir2_defaults;

	unsigned int cnstcoil_flags = 0;
	bool combine = true;

	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_FLOAT('M', &conf.alpha_min, "", "(minimum for regularization)"),
		OPT_SET('c', &conf.rvc, "Real-value constraint"),
		OPT_SET('g', &(conf.gpu), "use gpu"),
		//OPT_UINT('s', &cnstcoil_flags, "", "(dimensions with constant sensitivities)"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPTL_FLOAT(0, "alpha", &conf.alpha, "", "(minimum for regularization)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	(conf.gpu ? num_init_gpu_memopt : num_init)();

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

	dims[MAPS_DIM] = nmaps;

	long sens_dims[DIMS];
	md_select_dims(DIMS, ~cnstcoil_flags, sens_dims, dims);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	long img_output_dims[DIMS];
	md_copy_dims(DIMS, img_output_dims, img_dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, dims);

	if (conf.noncart) {

		if (NULL == traj) {

			for (int i = 0; i < 3; i++)
				if (1 != img_output_dims[i])
					img_output_dims[i] /= 2;
		} else {

			for (int i = 0; i < 3; i++)
				if (1 != sens_dims[i])
					sens_dims[i] *= 2;
		}
	}

	if (combine)
		img_output_dims[MAPS_DIM] = 1;

	complex float* img_output = create_cfl(img_file, DIMS, img_output_dims);
	md_clear(DIMS, img_output_dims, img_output, CFL_SIZE);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);


	complex float* sens = ((NULL != sens_file) ? create_cfl : anon_cfl)((NULL != sens_file) ? sens_file : "", DIMS, sens_dims);

	auto nlop_nlinv = noir_cart_unrolled_create(DIMS, pat_dims, pattern, NULL, NULL, msk_dims, NULL, ksp_dims, cim_dims, img_dims, sens_dims, &conf);
	nlop_generic_apply_unchecked(nlop_nlinv, 3, (void*[3]){img_output, sens, kspace});
	nlop_free(nlop_nlinv);

	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_output_dims, img_output);
	unmap_cfl(DIMS, ksp_dims, kspace);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	exit(0);
}


