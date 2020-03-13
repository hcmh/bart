/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2019 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
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

#include "noir/recon.h"

#include "moba/recon_T2.h"




static const char usage_str[] = "<kspace> <TI> <output> [<sensitivities>]";
static const char help_str[] =
		"Model-based nonlinear inverse reconstruction\n";


int main_mdbT2(int argc, char* argv[])
{
	double start_time = timestamp();

	bool normalize = true;
	float restrict_fov = -1.;
	const char* psf = NULL;
	const char* init_file = NULL;
	struct noir_conf_s conf = noir_defaults;
	bool out_sens = false;
	bool scale_im = false;
	bool use_gpu = false;

	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_FLOAT('j', &conf.alpha_min, "", "Minimum regu. parameter"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('c', &conf.rvc, "Real-value constraint"),
		OPT_CLEAR('N', &normalize, "Do not normalize image with coil sensitivities"),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_STRING('p', &psf, "PSF", ""),
		OPT_STRING('I', &init_file, "file", "File for initialization"),
		OPT_SET('g', &use_gpu, "use gpu"),
		OPT_SET('S', &scale_im, "Re-scale image after reconstruction"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (5 == argc)
		out_sens = true;



	(use_gpu ? num_init_gpu_memopt : num_init)();

	long ksp_dims[DIMS];
	complex float* kspace_data = load_cfl(argv[1], DIMS, ksp_dims);
	
	long TI_dims[DIMS];
	complex float* TI = load_cfl(argv[2], DIMS, TI_dims);

	assert(TI_dims[TE_DIM] == ksp_dims[TE_DIM]);

	// SMS
	if (1 != ksp_dims[SLICE_DIM]) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace_data, kspace_data); // fftmod to get correct slice order in output
	}


	assert(1 == ksp_dims[MAPS_DIM]);

	long dims[DIMS];
	md_copy_dims(DIMS, dims, ksp_dims);

	long img_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|SLICE_FLAG, img_dims, dims);

	img_dims[COEFF_DIM] = 2;

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);
	
	long img1_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG|SLICE_FLAG, img1_dims, dims);

	long img1_strs[DIMS];
	md_calc_strides(DIMS, img1_strs, img1_dims, CFL_SIZE);

	long coil_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG|SLICE_FLAG, coil_dims, dims);

	long coil_strs[DIMS];
	md_calc_strides(DIMS, coil_strs, coil_dims, CFL_SIZE);

	complex float* img = create_cfl(argv[3], DIMS, img_dims);
	complex float* img1 = create_cfl("", DIMS, img1_dims);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, dims);

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = NULL;
	complex float* norm = md_alloc(DIMS, img_dims, CFL_SIZE);
	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[4] : "", DIMS, coil_dims);

	// initialization
	if (NULL != init_file) {

		long skip = md_calc_size(DIMS, img_dims);
		long init_dims[DIMS];
		complex float* init = load_cfl(init_file, DIMS, init_dims);

		assert(md_check_bounds(DIMS, 0, img_dims, init_dims));

		md_copy(DIMS, img_dims, img, init, CFL_SIZE);
		fftmod(DIMS, coil_dims, FFT_FLAGS|SLICE_FLAG, sens, init + skip);

	} else {

		md_zfill(DIMS, img_dims, img, 1.0);
        
		
		md_clear(DIMS, coil_dims, sens, CFL_SIZE);
	}

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != psf) {

		complex float* tmp_psf =load_cfl(psf, DIMS, pat_dims);
		pattern = anon_cfl("", DIMS, pat_dims);

		md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);
		unmap_cfl(DIMS, pat_dims, tmp_psf);
		// FIXME: check compatibility

		if (-1 == restrict_fov)
		restrict_fov = 1.0;

		conf.noncart = true;

	} else {

		md_copy_dims(DIMS, pat_dims, img_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
	}

#if 0
	float scaling = 1. / estimate_scaling(ksp_dims, NULL, kspace_data);
#else
	double scaling = 6000. / md_znorm(DIMS, ksp_dims, kspace_data);

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
		md_zsmul2(DIMS, img_dims, img_strs, img, msk_strs, mask, 1.0);
	}

#ifdef  USE_CUDA
	if (use_gpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, ksp_dims, CFL_SIZE);
		md_copy(DIMS, ksp_dims, kspace_gpu, kspace_data, CFL_SIZE);
		
		complex float* TI_gpu = md_alloc_gpu(DIMS, TI_dims, CFL_SIZE);
		md_copy(DIMS, TI_dims, TI_gpu, TI, CFL_SIZE);

		T2_recon(&conf, dims, img, sens, pattern, mask, TI_gpu, kspace_gpu, use_gpu);

		md_free(kspace_gpu);
		md_free(TI_gpu);
	} else
#endif
	T2_recon(&conf, dims, img, sens, pattern, mask, TI, kspace_data, use_gpu);

	if (normalize) {

		md_zrss(DIMS, ksp_dims, COIL_FLAG, norm, sens);
		md_zmul2(DIMS, img_dims, img_strs, img, img_strs, img, img_strs, norm);
	}

	if (out_sens) {

		long strs[DIMS];
		md_calc_strides(DIMS, strs, ksp_dims, CFL_SIZE);

		if (normalize)
			md_zdiv2(DIMS, coil_dims, strs, sens, strs, sens, img_strs, norm);

// 		fftmod(DIMS, coil_dims, FFT_FLAGS, sens, sens);
	}

	if (scale_im)
		md_zsmul(DIMS, img_dims, img, img, 1. / scaling);

	md_free(norm);
	md_free(mask);

	unmap_cfl(DIMS, coil_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_dims, img);
	unmap_cfl(DIMS, img1_dims, img1);
	unmap_cfl(DIMS, ksp_dims, kspace_data);
	unmap_cfl(DIMS, TI_dims, TI);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}


