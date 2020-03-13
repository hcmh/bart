/* Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/gpuops.h"
#include "num/init.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "moba/recon_pixel.h"
#include "moba/model_Bloch.h"
#include "moba/scale.h"




static const char usage_str[] = "<data> <output>";
static const char help_str[] = 
		"fit Bloch equations pixel-wisely to data and return relaxation parameters.\n";


int main_pixel(int argc, char* argv[])
{
	double start_time = timestamp();

	struct noir_conf_s conf = noir_defaults;
	struct modBlochFit fit_para = modBlochFit_defaults;
	bool use_gpu = false;
	float data_scale = 600;
	const char* inputB1 = NULL;
	const char* inputSP = NULL;
	const char* fa_file = NULL;
	
	const struct opt_s opts[] = {

		OPT_UINT(	'i', 	&conf.iter, 		"iter", 	"Number of Newton steps"),
		OPT_FLOAT(	'R', 	&conf.redu, 		"", 		"reduction factor"),
		OPT_FLOAT(	'w', 	&conf.alpha_min, 	"", 		"alpha_min"),
		OPT_INT(	'd', 	&debug_level, 		"level", 	"Debug level"),
		OPT_INT(	'M', 	&fit_para.sequence,	"", 		"Define sequence mode: 0 = bSSFP[default], 1 = invbSSFP, 3 = pcbSSFP, 4 = inv. bSSFP without preparation, 5 = invFLASH, 6 = invpcbSSFP"),
		OPT_FLOAT(	'D', 	&fit_para.rfduration, 	"", 		"Duration of RF-pulse [s]"),
		OPT_FLOAT(	't', 	&fit_para.tr, 		"", 		"tr [s]"),
		OPT_FLOAT(	'e', 	&fit_para.te, 		"", 		"te [s]"),
		OPT_FLOAT(	'F', 	&fit_para.fa, 		"", 		"Flipangle [deg]"),
		OPT_INT(	'a', 	&fit_para.averaged_spokes, "", 		"Number of averaged spokes"),
		OPT_INT(	'r', 	&fit_para.rm_no_echo, 	"", 		"Number of removed echoes."),
		OPT_FLOAT(	'S', 	&data_scale, 		"", 		"Raw data scaling"),
		OPT_SET(	'O', 	&fit_para.full_ode_sim	, 		"Apply full ODE simulation"),
		OPT_INT(	'X', 	&fit_para.runs, 		"", 		"Number of applied whole sequence trains."),
		OPT_STRING(	'I',	&inputB1, 		"", 		"Input B1 image"),
		OPT_STRING(	'P',	&inputSP, 		"", 		"Input Slice Profile image"),
		OPT_STRING(	'V', 	&fa_file, 		"", 		"Variable flipangle file"),
		OPT_SET(	'g', 	&use_gpu, 				"use gpu"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	
	(use_gpu ? num_init_gpu_memopt : num_init)();
	
	// Load k-space data
	long dims[DIMS];
	
	complex float* data = load_cfl(argv[1], DIMS, dims);
	assert(1 == dims[MAPS_DIM]);	
	
	// Create image output
	long img_dims[DIMS];
	
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|SLICE_FLAG, img_dims, dims);
	img_dims[COEFF_DIM] = 3;
	
	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	complex float* img = create_cfl(argv[2], DIMS, img_dims);
	md_zfill(DIMS, img_dims, img, 1.);
	
	
	double scaling = data_scale / md_znorm(DIMS, dims, data);//1. / md_zrms(DIMS, dims, data) / md_zrms(DIMS, dims, data);
	
	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(DIMS, dims, data, data, scaling); 
	
	
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
	
	if (NULL != fa_file) {
		
		input_vfa = load_cfl(fa_file, DIMS, input_vfa_dims);
		
		fit_para.num_vfa = input_vfa_dims[READ_DIM];
		debug_printf(DP_DEBUG3, "Number of variable flip angles: %d\n", fit_para.num_vfa);
		
		fit_para.input_fa_profile = md_alloc(DIMS, input_vfa_dims, CFL_SIZE);
		md_copy(DIMS, input_vfa_dims, fit_para.input_fa_profile, input_vfa, CFL_SIZE);
		
	}
	
	
	//Assign initial guesses to maps
	long tmp_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, tmp_dims, img_dims);

	long tmp_strs[DIMS];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);
	
	//Values for Initialization of maps
	complex float initval[3] = {0.8, 20., 4.} ;//	R1, R2, M0 
	
	auto_scale(&fit_para, fit_para.scale, dims, data);
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
	if (use_gpu) {

		complex float* data_gpu = md_alloc_gpu(DIMS, dims, CFL_SIZE);
		md_copy(DIMS, dims, data_gpu, data, CFL_SIZE);

		pixel_recon(&conf, &fit_para, dims, img, data_gpu, use_gpu);

		md_free(data_gpu);
	} else
#endif
		pixel_recon(&conf, &fit_para, dims, img, data, use_gpu);
	
	pos[COEFF_DIM] = 2;
	md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
	md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, 1. / scaling);
	
	md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE); 
	
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
	
	unmap_cfl(DIMS, img_dims, img);
	unmap_cfl(DIMS, dims, data);
	
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


