/* Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/gpuops.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noir/recon_Bloch.h"
#include "noir/model_Bloch.h"




static const char usage_str[] = "<kspace> <output> [<sensitivities>]";
static const char help_str[] = 
		"Model-based nonlinear inverse reconstructionus using Bloch equations as signal model.\n";


int main_modbloch(int argc, char* argv[])
{
	double start_time = timestamp();

	float restrict_fov = -1.;
	const char* psf = NULL;
	struct noir_conf_s conf = noir_defaults;
	struct modBlochFit fitPara = modBlochFit_defaults;
	bool out_sens = false;
	bool usegpu = false;
	const char* inputB1 = NULL;
	const char* inputSP = NULL;
	const char* inputM0 = NULL;

	const struct opt_s opts[] = {

		OPT_UINT(	'i', 	&conf.iter, 		"", "Number of Newton steps"),
		OPT_FLOAT(	'R', 	&conf.redu, 		"", "reduction factor"),
		OPT_FLOAT(	'w', 	&conf.alpha_min, 	"", "alpha_min"),
		OPT_INT(	'd', 	&debug_level, 		"", "Debug level"),
		OPT_FLOAT(	'f', 	&restrict_fov, 		"", "FoV scaling factor"),
		OPT_FLOAT(	'D', 	&fitPara.rfduration, 	"", "Duration of RF-pulse [s]"),
		OPT_FLOAT(	't', 	&fitPara.tr, 		"", "TR [s]"),
		OPT_FLOAT(	'e', 	&fitPara.te, 		"", "TE [s]"),
		OPT_INT(	'a', 	&fitPara.averageSpokes, "", "Number of averaged spokes"),
		OPT_FLOAT(	'S', 	&fitPara.r1scaling, 	"", "R1 gradient scaling [s]"),
		OPT_FLOAT(	's', 	&fitPara.r2scaling, 	"", "R2 gradient scaling [s]"),
		OPT_FLOAT(	'm', 	&fitPara.m0scaling, 	"", "M0 gradient scaling"),
		OPT_INT(	'r', 	&fitPara.rm_no_echo, 	"", "Number of removed echoes."),
		OPT_STRING(	'p',	&psf, 			"", "Include Point-Spread-Function"),
		OPT_STRING(	'I',	&inputB1, 		"", "Input B1 image"),
		OPT_STRING(	'P',	&inputSP, 		"", "Input Slice Profile image"),
		OPT_STRING(	'M',	&inputM0, 		"", "Input M0 image"),
		OPT_SET(	'g', 	&usegpu			,  "use gpu"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (4 == argc)
		out_sens = true;
	
// 	if (usegpu)
// 		cuda_use_global_memory();
	
	num_init();
	
	// Load k-space data
	long ksp_dims[DIMS];
	
	complex float* kspace_data = load_cfl(argv[1], DIMS, ksp_dims);
	assert(1 == ksp_dims[MAPS_DIM]);

	
	// Create image output
	long img_dims[DIMS];
	
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|SLICE_FLAG, img_dims, ksp_dims);
	img_dims[COEFF_DIM] = 3;
	
	long img_strs[DIMS];
	
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	complex float* img = create_cfl(argv[2], DIMS, img_dims);
	md_zfill(DIMS, img_dims, img, 1.0);
	
	//Create coil output
	long coil_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG|SLICE_FLAG|TIME2_FLAG, coil_dims, ksp_dims);
	
	
	// Create sensitivity output
	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[3] : "", DIMS, coil_dims);
	md_clear(DIMS, coil_dims, sens, CFL_SIZE);

	
	// Restrict field of view
	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, ksp_dims);
	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);
	complex float* mask = NULL;

	
	
	
	// Load psf if given
	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != psf) {

		complex float* tmp_psf =load_cfl(psf, DIMS, pat_dims);
		pattern = anon_cfl("", DIMS, pat_dims);

		md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);
		unmap_cfl(DIMS, pat_dims, tmp_psf);
		// FIXME: check compatibility

		if (-1 == restrict_fov)
			restrict_fov = 0.5;
		
		fitPara.fov_reduction_factor = restrict_fov;

		conf.noncart = true;

	} else {

		md_copy_dims(DIMS, pat_dims, img_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
	}
	
	
	complex float* input_b1 = NULL;
	
	long input_dims[DIMS];
	
	if (NULL != inputB1)
		input_b1 = load_cfl(inputB1, DIMS, input_dims);
	
	
	complex float* input_sp = NULL;
	long input_sp_dims[DIMS];
	
	if (NULL != inputSP) {
		
		input_sp = load_cfl(inputSP, DIMS, input_sp_dims);
		fitPara.n_slcp = input_sp_dims[READ_DIM];
		debug_printf(DP_DEBUG3, "Number of slice profile estimates: %d\n", fitPara.n_slcp);
	}
	
	complex float* inputM0_img = NULL;
	
	if (NULL != inputM0)
		inputM0_img = load_cfl(inputM0, DIMS, input_dims);
	
	
	
	double scaling = 5000. / md_znorm(DIMS, ksp_dims, kspace_data);
	double scaling_psf = 1000. / md_znorm(DIMS, pat_dims, pattern);

	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(DIMS, ksp_dims, kspace_data, kspace_data, scaling);

	debug_printf(DP_INFO, "Scaling_psf: %f\n", scaling_psf);
	md_zsmul(DIMS, pat_dims, pattern, pattern, scaling_psf);

	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, msk_dims, CFL_SIZE);
		md_zfill(DIMS, msk_dims, mask, 1.);

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;
		mask = compute_mask(DIMS, msk_dims, restrict_dims);

		md_zmul2(DIMS, img_dims, img_strs, img, img_strs, img, msk_strs, mask);
	}

	//Assign initial guesses to maps
	long tmp_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, tmp_dims, img_dims);
	long tmp_strs[DIMS];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);
	

	
	//Values for Initialization of maps
	complex float initval[3] = {0.8, 11., 4.} ;//	R1, R2, M0 
	
	float grad_scale[3] = {fitPara.r1scaling, fitPara.r2scaling, fitPara.m0scaling};
	
	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);
	
	complex float* tmp_img = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	complex float* ones = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	md_zfill(DIMS, tmp_dims, ones, 1.);
	
	
	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {
		
		pos[COEFF_DIM] = i;
		
		md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, initval[i]);
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, 1. / grad_scale[i]);
		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE);  
	
	}
	
#ifdef  USE_CUDA
	if (usegpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, ksp_dims, CFL_SIZE);
		md_copy(DIMS, ksp_dims, kspace_gpu, kspace_data, CFL_SIZE);

		bloch_recon(&conf, &fitPara, ksp_dims, img, sens, pattern, mask, kspace_gpu, input_b1, input_sp, usegpu);

		md_free(kspace_gpu);
	} else
#endif
		bloch_recon(&conf, &fitPara, ksp_dims, img, sens, pattern, mask, kspace_data, input_b1, input_sp, usegpu);
	
	
	
	complex float* tmp_sens = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	
	md_zrss(DIMS, tmp_dims, COIL_FLAG, tmp_sens, sens);
	md_zmul(DIMS, tmp_dims, tmp_img, tmp_img, tmp_sens);
	
	md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE); 
	
	md_free(tmp_sens);
	
	//Convert R1 and R2 to T1 and T2 image
	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {
		
		pos[COEFF_DIM] = i;
		
		md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
		
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, grad_scale[i]);
		
		if (2 == i)
			md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, 1. / scaling);
		else
			md_zdiv(DIMS, tmp_dims, tmp_img, ones, tmp_img);
		
		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE);
			

	}

	md_free(tmp_img);
	md_free(ones);
	
	md_free(mask);

	unmap_cfl(DIMS, coil_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_dims, img );
	unmap_cfl(DIMS, ksp_dims, kspace_data);
	
	if(NULL != input_b1)
		unmap_cfl(DIMS, input_dims, input_b1);
	
	if(NULL != input_sp)
		unmap_cfl(DIMS, input_sp_dims, input_sp);
	
	if(NULL != inputM0_img)
		unmap_cfl(DIMS, input_dims, inputM0_img);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}


