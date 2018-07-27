/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noncart/nufft.h"

#include "noir/recon.h"





static const char usage_str[] = "<kspace> <output> [<sensitivities>]";
static const char help_str[] =
		"Jointly estimate image and sensitivities with nonlinear\n"
		"inversion using {iter} iteration steps. Optionally outputs\n"
		"the sensitivities.";



int main_nlinv(int argc, char* argv[])
{
	double start_time = timestamp();

	bool normalize = true;
	bool combine = true;
	unsigned int nmaps = 1;
	float restrict_fov = -1.;
	const char* traj_file = NULL;
	const char* psf = NULL;
	const char* init_file = NULL;
	struct noir_conf_s conf = noir_defaults;
	struct nufft_conf_s nufft_conf = nufft_conf_defaults;
// 	nufft_conf.toeplitz = false;
	bool out_sens = false;
	bool scale_im = false;

	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_STRING('t', &traj_file, "file", "k-space trajectory"),
		OPT_SET('c', &conf.rvc, "Real-value constraint"),
		OPT_CLEAR('N', &normalize, "Do not normalize image with coil sensitivities"),
		OPT_UINT('m', &nmaps, "nmaps", "Number of ENLIVE maps to use in reconsctruction"),
		OPT_CLEAR('U', &combine, "Do not combine ENLIVE maps in output"),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_STRING('p', &psf, "PSF", ""),
		OPT_STRING('I', &init_file, "file", "File for initialization"),
		OPT_SET('g', &conf.usegpu, "use gpu"),
		OPT_SET('S', &scale_im, "Re-scale image after reconstruction"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPT_SET('l', &conf.nlinv_legacy, "(use legacy termination criterion)"),
		OPT_FLOAT('C', &conf.cgtol, "", "(cgtol, default: 0.1f)"),
		OPT_SET('P', &conf.pattern_for_each_coil, "(supplied psf is different for each coil)"),
		OPT_UINT('A', &conf.algo, "algo", "0: IRGNM, 1: LevMar, 2: hybrid, 3: altmin"),
	};

	cmdline(&argc, argv, 2, 5, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (argc >= 4)
		out_sens = true;

	if (argc >= 5)
		conf.out_im_steps = true;

	if (argc >= 6)
		conf.out_coils_steps = true;


	num_init();

	long ksp_dims[DIMS]; // Dimension of input k-space. Can be non-Cartesian
	long ksp_strs[DIMS];

	long traj_dims[DIMS]; // Possible trajectory. Can be empty

	long coil_imgs_dims[DIMS]; // Size of coil images before (nu)FFT
	long coil_imgs_strs[DIMS]; // If Cartesian, is is the same as ksp_dims

	long sens_dims[DIMS]; // sensitivities. Can include ENLIVE maps
	long sens_strs[DIMS];

	long img_dims[DIMS]; // Image. Can include ENLIVE maps
	long img_strs[DIMS];

	long img_output_dims[DIMS]; // For ouput. Might or might not include ENLIVE maps
	long img_output_strs[DIMS];

	long msk_dims[DIMS]; // Image space mask.
	long msk_strs[DIMS];


	complex float* kspace_data = load_cfl(argv[1], DIMS, ksp_dims);

	// SMS
	if (1 != ksp_dims[SLICE_DIM]) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace_data, kspace_data); // fftmod to get correct slice order in output
	}

	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconsctruction
	assert(1 == ksp_dims[MAPS_DIM]);


	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);
	conf.dims.ksp_dims = ksp_dims;



	complex float* traj = NULL;
	if (NULL != traj_file) {

		traj = load_cfl(traj_file, DIMS, traj_dims);
		conf.noncart = true;
		conf.dims.traj_dims = traj_dims;
	}



	md_copy_dims(DIMS, sens_dims, ksp_dims);

	if (NULL != traj_file)
		estimate_im_dims(DIMS, sens_dims, traj_dims, traj);

	md_copy_dims(DIMS, coil_imgs_dims, sens_dims);
	md_calc_strides(DIMS, coil_imgs_strs, coil_imgs_dims, CFL_SIZE);
	conf.dims.coil_imgs_dims = coil_imgs_dims;

	sens_dims[MAPS_DIM] = nmaps;
	md_calc_strides(DIMS, sens_strs, sens_dims, CFL_SIZE);
	conf.dims.sens_dims = sens_dims;



	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|SLICE_FLAG, img_dims, sens_dims);
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);
	conf.dims.img_dims = img_dims;


	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG, img_output_dims, sens_dims);
	if (!combine)
		img_output_dims[MAPS_DIM] = nmaps;


	md_calc_strides(DIMS, img_output_strs, img_output_dims, CFL_SIZE);

	complex float* img_output = create_cfl(argv[2], DIMS, img_output_dims);
	md_clear(DIMS, img_output_dims, img_output, CFL_SIZE);
	complex float* img = md_alloc(DIMS, img_dims, CFL_SIZE);


	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);


	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = NULL;

	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[3] : "", DIMS, sens_dims);

	// outputfile for all steps:
	long out_im_steps_dims[DIMS];
	if (conf.out_im_steps) {

		md_copy_dims(DIMS, out_im_steps_dims, img_dims);
		out_im_steps_dims[ITER_DIM] = conf.iter;
		if (3 == conf.algo)
			out_im_steps_dims[ITER_DIM] *= 2;
		conf.out_im = create_cfl(argv[4], DIMS, out_im_steps_dims);
	}

	// outputfile for all steps:
	long out_coils_steps_dims[DIMS];
	if (conf.out_coils_steps) {

		md_copy_dims(DIMS, out_coils_steps_dims, sens_dims);
		out_coils_steps_dims[ITER_DIM] = conf.iter;
		if (3 == conf.algo)
			out_coils_steps_dims[ITER_DIM] *= 2;
		conf.out_coils = create_cfl(argv[5], DIMS, out_coils_steps_dims);
	}

	// initialization
	if (NULL != init_file) {

		long skip = md_calc_size(DIMS, img_dims);
		long init_dims[DIMS];
		complex float* init = load_cfl(init_file, DIMS, init_dims);

		assert(md_check_bounds(DIMS, 0, img_dims, init_dims));

		md_copy(DIMS, img_dims, img, init, CFL_SIZE);
		fftmod(DIMS, sens_dims, FFT_FLAGS|SLICE_FLAG, sens, init + skip);

		unmap_cfl(DIMS, init_dims, init);

	} else {

		md_zfill(DIMS, img_dims, img, 1.);
		md_clear(DIMS, sens_dims, sens, CFL_SIZE);
	}

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL == traj_file) {
		if (NULL != psf) {

			complex float* tmp_psf =load_cfl(psf, DIMS, pat_dims);
			pattern = anon_cfl("", DIMS, pat_dims);

			md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);
			unmap_cfl(DIMS, pat_dims, tmp_psf);
			// FIXME: check compatibility

			if (conf.pattern_for_each_coil) {

				assert( 1 != pat_dims[COIL_DIM] );
			} else {
				if (-1 == restrict_fov)
					restrict_fov = 0.5;

				conf.noncart = true;
			}

		} else {

			md_copy_dims(DIMS, pat_dims, img_dims);
			pattern = anon_cfl("", DIMS, pat_dims);
			estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
		}
	}

#if 0
	float scaling = 1. / estimate_scaling(ksp_dims, NULL, kspace_data);
#else
	double scaling = 100. / md_znorm(DIMS, ksp_dims, kspace_data);

	if (1 != ksp_dims[SLICE_DIM]) // SMS
			scaling *= sqrt(ksp_dims[SLICE_DIM]); 

#endif
	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(DIMS, ksp_dims, kspace_data, kspace_data, scaling);

	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, msk_dims, CFL_SIZE);
		md_zfill(DIMS, msk_dims, mask, 1.);

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;
		mask = compute_mask(DIMS, msk_dims, restrict_dims);
	}

	long skip = md_calc_size(DIMS, img_dims);
	long size = skip + md_calc_size(DIMS, sens_dims);

	complex float* ref = NULL;

#ifdef  USE_CUDA
	if (conf.usegpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, ksp_dims, CFL_SIZE);
		md_copy(DIMS, ksp_dims, kspace_gpu, kspace_data, CFL_SIZE);


		noir_recon(&conf, &nufft_conf, img, sens, ref, pattern, mask, kspace_gpu, (NULL == traj_file) ? NULL : traj);
		md_free(kspace_gpu);
	} else
#endif
		noir_recon(&conf, &nufft_conf, img, sens, ref, pattern, mask, kspace_data, (NULL == traj_file) ? NULL : traj);


	// image output
	if (normalize) {

		complex float* buf = md_alloc(DIMS, sens_dims, CFL_SIZE);
		md_clear(DIMS, sens_dims, buf, CFL_SIZE);

		if (combine) {

			md_zfmac2(DIMS, sens_dims, coil_imgs_strs, buf, img_strs, img, sens_strs, sens);
			md_zrss(DIMS, coil_imgs_dims, COIL_FLAG, img_output, buf);
		} else {

			md_zfmac2(DIMS, sens_dims, sens_strs, buf, img_strs, img, sens_strs, sens);
			md_zrss(DIMS, sens_dims, COIL_FLAG, img_output, buf);
		}
		md_zmul2(DIMS, img_output_dims, img_output_strs, img_output, img_output_strs, img_output, msk_strs, mask);

		if (1 == nmaps || !combine) {

			//restore phase
			md_zphsr(DIMS, img_output_dims, buf, img);
			md_zmul(DIMS, img_output_dims, img_output, img_output, buf);
		}

		md_free(buf);
	} else {

		if (combine) {

			// just sum up the map images
			md_zaxpy2(DIMS, img_dims, img_output_strs, img_output, 1., img_strs, img);
		} else { /*!normalize && !combine */

			// Just copy
			md_copy(DIMS, img_output_dims, img_output, img, CFL_SIZE);
		}
	}

	if (scale_im)
		md_zsmul(DIMS, img_output_dims, img_output, img_output, 1. / scaling);

	md_free(mask);
	md_free(img);

	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, img_output_dims, img_output);
	unmap_cfl(DIMS, ksp_dims, kspace_data);
	if (NULL != traj)
		unmap_cfl(DIMS, traj_dims, traj);
	else
		unmap_cfl(DIMS, pat_dims, pattern);
	if (conf.out_im_steps)
		unmap_cfl(DIMS, out_im_steps_dims, conf.out_im);
	if (conf.out_coils_steps)
		unmap_cfl(DIMS, out_coils_steps_dims, conf.out_coils);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	num_deinit();
	exit(0);
}


