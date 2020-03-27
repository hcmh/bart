/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "linops/linop.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "moba/meco.h"
#include "moba/recon_meco.h"
#include "moba/model_meco.h"

#include "noncart/nufft.h"

#include "num/gpuops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/filter.h"
#include "num/init.h"
#include "num/mem.h"


static const char usage_str[] = "<kspace> <TE> <output> [<sensitivities>]";
static const char help_str[] =
		"Quantivative mapping for multi-echo radial FLASH\n"
		"via model-based nonlinear inverse reconstruction.";




static void init_maps(long N, long* maps_dims, long* maps_strs, complex float* maps, unsigned int sel_model)
{
	// set all parameters to 1.0
	md_zfill(N, maps_dims, maps, 1.0);

	if ( sel_model != PI ) {

		// set fB0 map to 0
		long NCOEFF = maps_dims[COEFF_DIM];
		long* pos = calloc(N, sizeof(long));
		pos[COEFF_DIM] = NCOEFF - 1; // fB0 position

		complex float* map_fB0 = (void*)maps + md_calc_offset(N, maps_strs, pos);
		long map1_dims[N];
		md_select_dims(N, ~COEFF_FLAG, map1_dims, maps_dims);
		md_zfill(N, map1_dims, map_fB0, 0.0);

		xfree(pos);
	}
}


// static void edge_weight(const long dims[3], complex float* dst)
// {
// 	unsigned int flags = 0;

// 	for (int i = 0; i < 3; i++)
// 		if (1 != dims[i])
// 			flags = MD_SET(flags, i);


// 	float beta = 100.;

// 	// FIXME: when dims[0] != dims[1]
// 	klaplace(3, dims, flags, dst);
// 	md_zspow(3, dims, dst, dst, 0.5);

// 	md_zsmul(3, dims, dst, dst, -beta*2);
// 	md_zsadd(3, dims, dst, dst, beta);

// 	md_zatanr(3, dims, dst, dst);
// 	md_zsmul(3, dims, dst, dst, -0.1/M_PI);
// 	md_zsadd(3, dims, dst, dst, 0.05);
// }


int main_mobaT2star(int argc, char* argv[])
{
	double start_time = timestamp();

	struct nufft_conf_s nufft_conf = nufft_conf_defaults;
	nufft_conf.toeplitz = false;

	struct noir_conf_s conf = noir_defaults;

	float restrict_fov = -1.;
	float overgrid = 1.5;
	float temp_damp = 0.9;

	unsigned int sel_model = WFR2S;
	unsigned int sel_regu = TIKHONOV;
	unsigned int fullsample = 0;

	// const char* psf_file = NULL;
	const char* init_file = NULL;
	const char* traj_file = NULL;

	bool out_origin_maps = false;
	bool out_sens = false;
	bool use_gpu = false;
	bool use_nufft = false;

	const struct opt_s opts[] = {

		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_UINT('F', &fullsample, "FSMP", "Full sample size (required for partial Fourier sampling)"),
		OPT_UINT('M', &sel_model, "model", "Select the model from enum { WF, WFR2S, WF2R2S, R2S, PI } [default: WFR2S]"),
		OPT_UINT('R', &sel_regu, "regu", "Select regularization type from enum { TIKHONOV, WAV, LLR } [default: TIKHONOV]"),
		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps [default: 8]"),
		OPT_FLOAT('T', &temp_damp, "tdamp", "temporal damping [default: 0.9]"),
		OPT_FLOAT('a', &conf.a, "W_c", "(a in 1 + a * \\Laplace^-b/2) [default: 220]"),
		OPT_FLOAT('b', &conf.b, "W_c", "(b in 1 + a * \\Laplace^-b/2) [default: 32]"),
		OPT_FLOAT('f', &restrict_fov, "rFOV", "Restrict FOV (for non-cartesian trajectories) [default: 0.5]"),
		OPT_FLOAT('j', &conf.alpha_min, "a_min", "Minimum regu. parameter [default: 0]"),
		OPT_FLOAT('o', &overgrid, "os", "Oversampling factor for gridding [default: 1.5]"),
		OPT_FLOAT('r', &conf.redu, "redu", "Reduction factor of the regularization strength [defautl: 2]"),
		OPT_SET('O', &out_origin_maps, "Output original maps from reconstruction without post processing"),
		OPT_SET('c', &conf.rvc, "Real-value constraint [default: 0]"),
		OPT_SET('g', &use_gpu, "Use gpu"),
		OPT_SET('n', &/*conf.*/use_nufft, "Use nufft"),
		OPT_STRING('I', &init_file, "init", "File for initialization"),
		// OPT_STRING('p', &psf_file, "psf", "File for point spread function"),
		OPT_STRING('t', &traj_file, "traj", "File for trajectory"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (5 == argc)
		out_sens = true;

	if (-1 == restrict_fov)
		restrict_fov = 0.5;

	conf.noncart = true;

	num_init();

#ifdef USE_CUDA
	cuda_use_global_memory();
#endif

	debug_printf(DP_DEBUG2, "___ qmeco with model: ");
	switch ( sel_model ) {
		case WF     : debug_printf(DP_DEBUG2, "water, fat and B0 field ___\n"); break;
		case WFR2S  : debug_printf(DP_DEBUG2, "water, fat, R2S, and B0 field ___\n"); break;
		case WF2R2S : debug_printf(DP_DEBUG2, "water, water_R2s, fat, fat_R2S, and B0 field ___\n"); break;
		case R2S    : debug_printf(DP_DEBUG2, "rho, R2S, and B0 field ___\n"); break;
		case PI     : debug_printf(DP_DEBUG2, "multi echo images ___\n"); break;
	}

	debug_printf(DP_DEBUG2, " __ regularization: ");
	switch ( sel_regu ) {
		case TIKHONOV : debug_printf(DP_DEBUG2, "Tikhonov l2\n"); break;
		case WAV      : debug_printf(DP_DEBUG2, "Daubechies wavelet l1\n"); break;
		case LLR      : debug_printf(DP_DEBUG2, "Locally low rank l1\n"); break;
	}

	debug_printf(DP_DEBUG2, " __ parameters: iter %2d; redu %.2f; alpha (%.1f, %.1f); coil (%.1f, %.1f); rvc %d\n", conf.iter, conf.redu, conf.alpha, conf.alpha_min, conf.rvc, conf.a, conf.b);


	// assert((psf_file!=NULL) ^ (traj_file!=NULL));

	// read k-space data
	long ksp_dims[DIMS];
	complex float* ksp = load_cfl(argv[1], DIMS, ksp_dims);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	long ksp_1f_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, ksp_1f_dims, ksp_dims);


	// read TE
	long TE_dims[DIMS];
	complex float* TE = load_cfl(argv[2], DIMS, TE_dims);


	assert(1 == ksp_dims[READ_DIM]);
	assert(1 == ksp_dims[COEFF_DIM]);
	assert(1 == ksp_dims[MAPS_DIM]);
	assert(TE_dims[TE_DIM] == ksp_dims[TE_DIM]);


	debug_printf(DP_DEBUG2, "  _ ksp_dims : ");
	debug_print_dims(DP_DEBUG2, DIMS, ksp_dims);

	debug_printf(DP_DEBUG3, "  _ TE (ms)  : ");
	for (int i = 0; i < TE_dims[TE_DIM]; i++) {
		debug_printf(DP_DEBUG3, "%.2f ", crealf(TE[i]));
	}
	debug_printf(DP_DEBUG3, "\n");


	long Y_dims[DIMS];
	md_copy_dims(DIMS, Y_dims, ksp_dims);

	complex float* Y = NULL;

	complex float* P = NULL;
	long P_dims[DIMS];
	long P_1f_dims[DIMS];
	long P_strs[DIMS];
	long P_turns = 0;
	long* P_pos  = calloc(DIMS, sizeof(long));


	// read traj
	assert(NULL != traj_file);

	long traj_dims[DIMS];
	long traj_strs[DIMS];
	long traj_1f_dims[DIMS];

	complex float* traj = load_cfl(traj_file, DIMS, traj_dims);

	md_calc_strides(DIMS, traj_strs, traj_dims, CFL_SIZE);
	md_select_dims(DIMS, ~TIME_FLAG, traj_1f_dims, traj_dims);


	if (fullsample == 0) { // TODO: fullsample can be read frorm traj?
		
		fullsample = traj_dims[1];
		debug_printf(DP_DEBUG2, "  _ fullsample set to %ld.\n", fullsample);
	}

	unsigned int gridsize = fullsample * overgrid;
	Y_dims[READ_DIM] = gridsize;
	Y_dims[PHS1_DIM] = gridsize;
	Y_dims[PHS2_DIM] = 1;

	md_select_dims(DIMS, FFT_FLAGS|TE_FLAG|TIME_FLAG|SLICE_FLAG, P_dims, Y_dims);
	P_dims[TIME_DIM] = traj_dims[TIME_DIM];


	struct linop_s* nufft_op = NULL;

	if ( !/*conf.*/use_nufft ) {

		long ones_dims[DIMS];
		md_copy_dims(DIMS, ones_dims, traj_dims);
		ones_dims[READ_DIM] = 1L;

		complex float* ones = md_alloc(DIMS, ones_dims, CFL_SIZE);
		md_clear(DIMS, ones_dims, ones, CFL_SIZE);

		// complex float* k_dummy = md_alloc(DIMS, ones_dims, CFL_SIZE);
		long pos_0[DIMS] = { 0 };

		md_copy_block(DIMS, pos_0, ones_dims, ones, ksp_dims, ksp, CFL_SIZE);

		// divide ksp-data by itself
		// zdiv makes sure that division by zero is set to 0
		md_zdiv(DIMS, ones_dims, ones, ones, ones);

		nufft_op = nufft_create(DIMS, ones_dims, P_dims, traj_dims, traj, NULL, nufft_conf);

		P = anon_cfl("", DIMS, P_dims);
		linop_adjoint(nufft_op, DIMS, P_dims, P, DIMS, ones_dims, ones);

		fftuc(DIMS, P_dims, FFT_FLAGS, P, P);
		
		linop_free(nufft_op);
		md_free(ones);

		dump_cfl("/scratch/ztan/_prep_P_linop", DIMS, P_dims, P);
		debug_printf(DP_DEBUG2, "  _ psf prepared with nufft linop.\n");

	} else {

		long wgh_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, wgh_dims, ksp_dims);

		complex float* wgh = md_alloc(DIMS, wgh_dims, CFL_SIZE);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, wgh, ksp);

		P = compute_psf(DIMS, P_dims, traj_dims, traj, traj_dims, NULL, wgh_dims, wgh, false, false);

		fftuc(DIMS, P_dims, FFT_FLAGS, P, P);

		dump_cfl("/scratch/ztan/_prep_P_compute_psf", DIMS, P_dims, P);
		dump_cfl("/scratch/ztan/_ksp_wgh", DIMS, wgh_dims, wgh);

		md_free(wgh);

		debug_printf(DP_DEBUG2, "  _ psf prepared with compute_psf from nufft.\n");

	}



	long Y_strs[DIMS];
	md_calc_strides(DIMS, Y_strs, Y_dims, CFL_SIZE);

	long Y_1f_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, Y_1f_dims, Y_dims);

	long Y_1f_strs[DIMS];
	md_calc_strides(DIMS, Y_1f_strs, Y_1f_dims, CFL_SIZE);

	P_turns = P_dims[TIME_DIM];
	md_select_dims(DIMS, ~TIME_FLAG, P_1f_dims, P_dims);
	md_calc_strides(DIMS, P_strs, P_dims, CFL_SIZE);


#if 0

	debug_printf(DP_DEBUG2, "  _ psf scaling.\n");

	ifft(DIMS, P_dims, FFT_FLAGS, P, P);

	float scale_P = 1./cabsf(P[0])/10.;
	md_zsmul(DIMS, P_dims, P, P, scale_P);

	fft(DIMS, P_dims, FFT_FLAGS, P, P);

#endif



#if 0

	// apply edge weight onto P
	debug_printf(DP_DEBUG2, "  _ psf weighted in the corner.\n");

	long w_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, w_dims, P_dims);

	long w_strs[DIMS];
	md_calc_strides(DIMS, w_strs, w_dims, CFL_SIZE);

	complex float* P_weight = md_alloc(DIMS, w_dims, CFL_SIZE);
	edge_weight(w_dims, P_weight);

	md_zadd2(DIMS, P_dims, P_strs, P, P_strs, P, w_strs, P_weight);

	md_free(P_weight);

#endif


	long maps_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COEFF_FLAG|TIME_FLAG|SLICE_FLAG, maps_dims, Y_dims);

	if ( sel_model == PI ) {
		maps_dims[TE_DIM] = Y_dims[TE_DIM];
	} else {
		maps_dims[COEFF_DIM] = set_num_of_coeff(sel_model);
	}

	long maps_strs[DIMS];
	md_calc_strides(DIMS, maps_strs, maps_dims, CFL_SIZE);

	long maps_1f_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, maps_1f_dims, maps_dims);

	long maps_1f_strs[DIMS];
	md_calc_strides(DIMS, maps_1f_strs, maps_1f_dims, CFL_SIZE);

	long img_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG, img_dims, Y_dims);

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);


	long sens_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|TIME_FLAG|SLICE_FLAG, sens_dims, Y_dims);

	long sens_strs[DIMS];
	md_calc_strides(DIMS, sens_strs, sens_dims, CFL_SIZE);

	long sens_1f_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, sens_1f_dims, sens_dims);

	long sens_1f_strs[DIMS];
	md_calc_strides(DIMS, sens_1f_strs, sens_1f_dims, CFL_SIZE);


	long mask_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, mask_dims, Y_dims);

	long mask_strs[DIMS];
	md_calc_strides(DIMS, mask_strs, mask_dims, CFL_SIZE);


	complex float* maps 	= create_cfl(argv[3], DIMS, maps_dims);
	complex float* mask 	= NULL;
	complex float* sens 	= (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[4] : "", DIMS, sens_dims);

	long* frame_pos         = calloc(DIMS, sizeof(long));
	complex float* maps_1f  = (void*)maps + md_calc_offset(DIMS, maps_strs, frame_pos);
	complex float* sens_1f  = (void*)sens + md_calc_offset(DIMS, sens_strs, frame_pos);
	





	debug_printf(DP_DEBUG2, "  _ Y_dims: \t");
	debug_print_dims(DP_DEBUG2, DIMS, Y_dims);

	debug_printf(DP_DEBUG2, "  _ P_dims: \t");
	debug_print_dims(DP_DEBUG2, DIMS, P_dims);

	debug_printf(DP_DEBUG2, "  _ maps_dims: \t");
	debug_print_dims(DP_DEBUG2, DIMS, maps_dims);

	debug_printf(DP_DEBUG2, "  _ sens_dims: \t");
	debug_print_dims(DP_DEBUG2, DIMS, sens_dims);
	

	


	// === initialization ===
	if ( NULL != init_file ) {

		long skip = md_calc_size(DIMS, maps_1f_dims);
		long init_dims[DIMS];
		complex float* init = load_cfl(init_file, DIMS, init_dims);

		assert(md_check_bounds(DIMS, 0, maps_1f_dims, init_dims));

		md_copy(DIMS, maps_1f_dims, maps_1f, init, CFL_SIZE); // maps
		fftmod(DIMS, sens_1f_dims, FFT_FLAGS|SLICE_FLAG, sens_1f, init + skip);
		
		unmap_cfl(DIMS, init_dims, init);

		debug_printf(DP_DEBUG2, "  _ init maps provided.\n");

	} else {

		init_maps(DIMS, maps_1f_dims, maps_1f_strs, maps_1f, sel_model);
		md_clear(DIMS, sens_1f_dims, sens_1f, CFL_SIZE);

	}


	// === mask ===
	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, mask_dims, CFL_SIZE);
		md_zfill(DIMS, mask_dims, mask, 1.);

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;
		mask = compute_mask(DIMS, mask_dims, restrict_dims);

		// dump_cfl("/tmp/meco_mask", DIMS, mask_dims, mask);

		md_zmul2(DIMS, maps_1f_dims, maps_1f_strs, maps_1f, maps_1f_strs, maps_1f, mask_strs, mask);
	}


	// start reconstruction
	debug_printf(DP_DEBUG2, "___ multi-frame reconstruction using ");

#ifdef USE_CUDA
	if ( use_gpu ) {
		debug_printf(DP_DEBUG2, "GPU __\n");
	}
#endif
	if ( !use_gpu ) {
		debug_printf(DP_DEBUG2, "CPU __\n");
	}
	
	complex float* Y_1f = md_alloc(DIMS, Y_1f_dims, CFL_SIZE);

	long maps_1f_size = md_calc_size(DIMS, maps_1f_dims);
	long sens_1f_size = md_calc_size(DIMS, sens_1f_dims);
	long x_1f_size = maps_1f_size + sens_1f_size;

	complex float* x    = md_alloc(1, MD_DIMS(x_1f_size), CFL_SIZE);
	complex float* xref = md_alloc(1, MD_DIMS(x_1f_size), CFL_SIZE);

	for (long f = 0; f < Y_dims[TIME_DIM]; f++) {

		bool reset = (f==0) ? true : false;

		debug_printf(DP_DEBUG2, "___ frame %3ld ___\n", f+1);

		frame_pos[TIME_DIM] = f;

		// if ( !conf.use_nufft ) {

			long* traj_pos = calloc(DIMS, sizeof(long));
			traj_pos[TIME_DIM] = f % P_turns;

			complex float* traj_1f = (void*)traj + md_calc_offset(DIMS, traj_strs, traj_pos);

			nufft_op = nufft_create(DIMS, ksp_1f_dims, Y_1f_dims, traj_1f_dims, traj_1f, NULL, nufft_conf);

			complex float* ksp_1f = (void*)ksp + md_calc_offset(DIMS, ksp_strs, frame_pos);
			linop_adjoint(nufft_op, DIMS, Y_1f_dims, Y_1f, DIMS, ksp_1f_dims, ksp_1f);
			fftuc(DIMS, Y_1f_dims, FFT_FLAGS, Y_1f, Y_1f);

			linop_free(nufft_op);

		// } else {

		// }

		double yscale = md_znorm(DIMS, Y_1f_dims, Y_1f);
		double scaling_Y_1f = 100. / yscale;
		md_zsmul(DIMS, Y_1f_dims, Y_1f, Y_1f, scaling_Y_1f);


		P_pos[TIME_DIM] = f % P_turns;
		complex float* P_1f = (void*)P + md_calc_offset(DIMS, P_strs, P_pos);


		debug_printf(DP_DEBUG2, " __ Scaling: ");
		debug_printf(DP_DEBUG2, "||Y|| = %.4f; ", yscale);
		debug_printf(DP_DEBUG2, "||P|| = %.4f\n", md_znorm(DIMS, P_1f_dims, P_1f));



		maps_1f = (void*)maps + md_calc_offset(DIMS, maps_strs, frame_pos);
		sens_1f = (void*)sens + md_calc_offset(DIMS, sens_strs, frame_pos);


		if ( reset ) {

			md_copy(DIMS, maps_1f_dims, x, maps_1f, CFL_SIZE);
			md_copy(DIMS, sens_1f_dims, x + maps_1f_size, sens_1f, CFL_SIZE);

			if ( NULL != init_file )
				md_zsmul(1, MD_DIMS(x_1f_size), xref, x, temp_damp);
			else
				md_clear(1, MD_DIMS(x_1f_size), xref, CFL_SIZE);

		} else {

			md_zsmul(1, MD_DIMS(x_1f_size), xref, x, temp_damp);
		}

#ifdef USE_CUDA
		if ( use_gpu ) {

			complex float* Y_1f_gpu = md_alloc_gpu(DIMS, Y_1f_dims, CFL_SIZE);
			md_copy(DIMS, Y_1f_dims, Y_1f_gpu, Y_1f, CFL_SIZE);

			meco_recon(&conf, sel_model, sel_regu, out_origin_maps, maps_1f_dims, sens_1f_dims, x, xref, P_1f, mask, TE, Y_1f_dims, Y_1f_gpu);

			md_free(Y_1f_gpu);
		}
#endif
		if ( !use_gpu ) {

			meco_recon(&conf, sel_model, sel_regu, out_origin_maps, maps_1f_dims, sens_1f_dims, x, xref, P_1f, mask, TE, Y_1f_dims, Y_1f);
		}
		

		md_copy(DIMS, maps_1f_dims, maps_1f, xref, CFL_SIZE);
		md_zsmul(DIMS, maps_1f_dims, maps_1f, maps_1f, 1. / scaling_Y_1f);

		md_copy(DIMS, sens_1f_dims, sens_1f, xref + maps_1f_size, CFL_SIZE);

		fftmod(DIMS, sens_1f_dims, FFT_FLAGS|SLICE_FLAG, sens_1f, sens_1f);

	}



	md_free(mask);
	md_free(Y_1f);
	md_free(x);
	md_free(xref);

	unmap_cfl(DIMS, maps_dims, maps);
	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, P_dims, P);
	unmap_cfl(DIMS, Y_dims, Y);
	unmap_cfl(DIMS, ksp_dims, ksp);
	unmap_cfl(DIMS, TE_dims, TE);

	if (NULL != traj_file) {
		unmap_cfl(DIMS, traj_dims, traj);
	}

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "___ Total Time: %.2f s\n", recosecs);
	exit(0);
}
