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
#include <stdio.h>

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

#include "moba/recon_T1.h"
#include "moba/recon_Bloch.h"
#include "moba/model_Bloch.h"
#include "moba/scale.h"

static void help_seq(void)
{
	printf( "Sequence Simulation Parameter\n\n"
		"#SEQ:\t Define sequence mode: \n"
		"\t\t\t0 = bSSFP[default]\n"
		"\t\t\t1 = invbSSFP\n"
		"\t\t\t2 = FLASH\n"
		"\t\t\t3 = pcbSSFP\n"
		"\t\t\t4 = inv. bSSFP without preparation\n"
		"\t\t\t5 = invFLASH\n"
		"\t\t\t6 = invpcbSSFP\n"
		"TR:\t Repetition time [s]\n"
		"TE:\t Echo time [s]\n"
		"FA:\t Flip angle of rf pulses [deg]\n"
		"Drf:\t Duration of RF pulse [s]\n"
		"Dinv:\t Duration of Inversion [s]\n"
		"Dprep:\t Duration of magnetization preparation [s]\n"
	);
}


static bool opt_seq(void* ptr, char c, const char* optarg)
{
	// Check if help function is called
	char rt[5];
	
	switch (c) {

	case 'P': {
		
		int ret = sscanf(optarg, "%7[^:]", rt);
		assert(1 == ret);

		if (strcmp(rt, "h") == 0) {

			help_seq();
			exit(0);
		}
		else {

			// Collect simulation data
			struct modBlochFit* data = ptr;

			ret = sscanf(optarg, "%d:%f:%f:%f:%f:%f:%f",	
									&data->sequence,
									&data->tr, 
									&data->te, 
									&data->fa, 
									&data->rfduration, 
									&data->inversion_pulse_length,
									&data->prep_pulse_length);
			assert(7 == ret);
		}
		break;
	}
	}
	return false;
}


static const char usage_str[] = "<kspace> <output> [<sensitivities>]";
static const char help_str[] = 
		"Model-based nonlinear inverse reconstructionus using Bloch equations as signal model.\n";

int main_modbloch(int argc, char* argv[])
{
	double start_time = timestamp();

	float restrict_fov = -1.;
	const char* psf = NULL;
	struct moba_conf conf = moba_defaults;
	struct modBlochFit fit_para = modBlochFit_defaults;
	bool out_sens = false;
	bool usegpu = false;
	const char* inputB1 = NULL;
	const char* inputSP = NULL;
	const char* inputVFA = NULL;
	
	const struct opt_s opts[] = {

		OPT_UINT(	'i', 	&conf.iter, 		"", "Number of Newton steps"),
		OPT_FLOAT(	'R', 	&conf.redu, 		"", "reduction factor"),
		OPT_FLOAT(	'l', 	&conf.alpha, 		"", "alpha"),
		OPT_FLOAT(	'm', 	&conf.alpha_min, 	"", "alpha_min"),
		OPT_UINT(	'o', 	&conf.opt_reg, 		"", "regularization option (0: l2, 1: l1-wav)"),
		OPT_INT(	'd', 	&debug_level, 		"", "Debug level"),
		OPT_FLOAT(	'f', 	&restrict_fov, 		"", "FoV scaling factor"),
		OPT_STRING(	'p',	&psf, 			"", "Include Point-Spread-Function"),
		OPT_STRING(	'I',	&inputB1, 		"", "Input B1 image"),
		OPT_STRING(	'S',	&inputSP, 		"", "Input Slice Profile image"),
		OPT_STRING(	'F',	&inputVFA, 		"", "Input for variable flipangle profile"),
		OPT_INT(	'n', 	&fit_para.not_wav_maps, 	"", "# Removed Maps from Wav.Denoisng"),
		OPT_SET(	'O', 	&fit_para.full_ode_sim	,  "Apply full ODE simulation"),
		OPT_INT(	'a', 	&fit_para.averaged_spokes, "", "Number of averaged spokes"),
		OPT_INT(	'r', 	&fit_para.rm_no_echo, 	"", "Number of removed echoes."),
		OPT_INT(	'w', 	&fit_para.runs, 		"", "Number of applied whole sequence trains."),
		OPT_SET(	'g', 	&usegpu			,  "use gpu"),
		{ 'P', true, opt_seq, &fit_para, "\tA:B:C:D:E:F:G\tSequence parameter <Seq:TR:TE:FA:Drf:Dinv:Dprep> (-Ph for help)" },
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (4 == argc)
		out_sens = true;
	
	assert(fit_para.rfduration <= fit_para.prep_pulse_length);

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

		pattern = load_cfl(psf, DIMS, pat_dims);
		// FIXME: check compatibility

		if (-1 == restrict_fov)
			restrict_fov = 0.5;
		
		fit_para.fov_reduction_factor = restrict_fov;

		conf.noncartesian = true;

	} else {

		md_copy_dims(DIMS, pat_dims, img_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
	}
	
	
	complex float* input_b1 = NULL;
	
	long input_b1_dims[DIMS];
	
	if (NULL != inputB1) {
		
		input_b1 = load_cfl(inputB1, DIMS, input_b1_dims);
		
		fit_para.input_b1 = md_alloc(DIMS, input_b1_dims, CFL_SIZE);
		md_copy(DIMS, input_b1_dims, fit_para.input_b1, input_b1, CFL_SIZE);
	}
		
	
	
	complex float* input_sliceprofile = NULL;
	long input_sp_dims[DIMS];
	
	if (NULL != inputSP) {
		
		input_sliceprofile = load_cfl(inputSP, DIMS, input_sp_dims);
		
		fit_para.sliceprofile_spins = input_sp_dims[READ_DIM];
		debug_printf(DP_DEBUG3, "Number of slice profile estimates: %d\n", fit_para.sliceprofile_spins);
		
		fit_para.input_sliceprofile = md_alloc(DIMS, input_sp_dims, CFL_SIZE);
		md_copy(DIMS, input_sp_dims, fit_para.input_sliceprofile, input_sliceprofile, CFL_SIZE);
		
	}
	
	complex float* input_vfa = NULL;
	long input_vfa_dims[DIMS];
	
	if (NULL != inputVFA) {
		
		input_vfa = load_cfl(inputVFA, DIMS, input_vfa_dims);
		
		fit_para.num_vfa = input_vfa_dims[READ_DIM];
		debug_printf(DP_DEBUG3, "Number of variable flip angles: %d\n", fit_para.num_vfa);
		
		fit_para.input_fa_profile = md_alloc(DIMS, input_vfa_dims, CFL_SIZE);
		md_copy(DIMS, input_vfa_dims, fit_para.input_fa_profile, input_vfa, CFL_SIZE);
		
	}
	
	
	double scaling = 5000. / md_znorm(DIMS, ksp_dims, kspace_data);
	double scaling_psf = 1.;

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
	complex float initval[3] = {0.8, 10., 4.} ;//	R1, R2, M0 
	
	auto_scale(&fit_para, fit_para.scale, ksp_dims, kspace_data);
	debug_printf(DP_DEBUG1,"Scaling:\t%f,\t%f,\t%f\n", fit_para.scale[0], fit_para.scale[1], fit_para.scale[2]);


	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);
	
	complex float* tmp_img = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	complex float* ones = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	md_zfill(DIMS, tmp_dims, ones, 1.);
	
	
	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {
		
		pos[COEFF_DIM] = i;
		
		md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, initval[i]);
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, 1. / fit_para.scale[i]);
		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE);  
	
	}
	
#ifdef  USE_CUDA
	if (usegpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, ksp_dims, CFL_SIZE);
		md_copy(DIMS, ksp_dims, kspace_gpu, kspace_data, CFL_SIZE);

		bloch_recon(&conf, &fit_para, ksp_dims, img, sens, pattern, mask, kspace_gpu, usegpu);

		md_free(kspace_gpu);
	} else
#endif
		bloch_recon(&conf, &fit_para, ksp_dims, img, sens, pattern, mask, kspace_data, usegpu);
	
	
	pos[COEFF_DIM] = 2;
	md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
	md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, scaling_psf / scaling);
	
	complex float* tmp_sens = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	
	md_zrss(DIMS, tmp_dims, COIL_FLAG, tmp_sens, sens);
	md_zmul(DIMS, tmp_dims, tmp_img, tmp_img, tmp_sens);
	
	md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE); 
	
	md_free(tmp_sens);
	
	//Convert R1 and R2 to T1 and T2 image
	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {
		
		pos[COEFF_DIM] = i;
		
		md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
		
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, fit_para.scale[i]);
		
		if (2 != i)
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
		unmap_cfl(DIMS, input_b1_dims, input_b1);
	
	if(NULL != input_sliceprofile)
		unmap_cfl(DIMS, input_sp_dims, input_sliceprofile);
	
	if(NULL != input_vfa)
		unmap_cfl(DIMS, input_vfa_dims, input_vfa);
	

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}


