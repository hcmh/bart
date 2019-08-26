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
	struct modBlochFit fitPara = modBlochFit_defaults;
	bool usegpu = false;
	float data_scale = 600;

	const struct opt_s opts[] = {

		OPT_UINT(	'i', 	&conf.iter, 		"iter", 	"Number of Newton steps"),
		OPT_FLOAT(	'R', 	&conf.redu, 		"", 		"reduction factor"),
		OPT_FLOAT(	'w', 	&conf.alpha_min, 	"", 		"alpha_min"),
		OPT_INT(	'd', 	&debug_level, 		"level", 	"Debug level"),
		OPT_FLOAT(	'D', 	&fitPara.rfduration, 	"", 		"Duration of RF-pulse [s]"),
		OPT_FLOAT(	't', 	&fitPara.tr, 		"", 		"TR [s]"),
		OPT_FLOAT(	'e', 	&fitPara.te, 		"", 		"TE [s]"),
		OPT_INT(	'a', 	&fitPara.averageSpokes, "", 		"Number of averaged spokes"),
		OPT_INT(	'r', 	&fitPara.rm_no_echo, 	"", 		"Number of removed echoes."),
		OPT_FLOAT(	'S', 	&data_scale, 		"", 		"Raw data scaling"),
		OPT_SET(	'O', 	&fitPara.full_ode_sim	, 		"Apply full ODE simulation"),
		OPT_SET(	'g', 	&usegpu, 				"use gpu"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	
	num_init();
	
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

	
	//Assign initial guesses to maps
	long tmp_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, tmp_dims, img_dims);

	long tmp_strs[DIMS];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);
	
	//Values for Initialization of maps
	complex float initval[3] = {0.8, 11., 4.} ;//	R1, R2, M0 
	
	
	auto_scale(&fitPara, fitPara.scale, dims, data);
	debug_printf(DP_DEBUG1,"Scaling:\t%f,\t%f,\t%f\n", fitPara.scale[0], fitPara.scale[1], fitPara.scale[2]);

	
	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);
	
	complex float* tmp_img = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	complex float* ones = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	md_zfill(DIMS, tmp_dims, ones, 1.);
	
	
	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {
	
		pos[COEFF_DIM] = i;
		
		md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, initval[i]);
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, 1. / fitPara.scale[i]);
		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE);  
	
	}
	
	
#ifdef  USE_CUDA	
	if (usegpu) {

		complex float* data_gpu = md_alloc_gpu(DIMS, dims, CFL_SIZE);
		md_copy(DIMS, dims, data_gpu, data, CFL_SIZE);

		pixel_recon(&conf, &fitPara, dims, img, data_gpu, usegpu);

		md_free(data_gpu);
	} else
#endif
		pixel_recon(&conf, &fitPara, dims, img, data, usegpu);
	
	pos[COEFF_DIM] = 2;
	md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
	md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, 1. / scaling);
	
	//Convert R1 and R2 to T1 and T2 image
	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {
		
		pos[COEFF_DIM] = i;
		
		md_copy_block(DIMS, pos, tmp_dims, tmp_img, img_dims, img, CFL_SIZE);
		
		md_zsmul(DIMS, tmp_dims, tmp_img, tmp_img, fitPara.scale[i]);
		
		if (2 != i)
			md_zdiv(DIMS, tmp_dims, tmp_img, ones, tmp_img);
		
		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp_img, CFL_SIZE);
			

	}


	md_free(tmp_img);
	md_free(ones);
	
	unmap_cfl(DIMS, img_dims, img );
	unmap_cfl(DIMS, dims, data);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}


