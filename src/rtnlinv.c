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

#include "noir/recon.h"

#include "noncart/nufft.h"

#include "linops/linop.h"






static const char usage_str[] = "<kspace> <output> [<sensitivities>]";
static const char help_str[] =
		"Jointly estimate image and sensitivities with nonlinear\n"
		"inversion using {iter} iteration steps. Optionally outputs\n"
		"the sensitivities.";



int main_rtnlinv(int argc, char* argv[])
{
	double start_time = timestamp();

	struct nufft_conf_s nufft_conf = nufft_conf_defaults;
	nufft_conf.toeplitz = false;

	bool normalize = true;
	bool combine = true;
	unsigned int nmaps = 1;
	float restrict_fov = -1.;
	float oversampling = 1.5f;
	float temp_damp = 0.9f;
	const char* psf = NULL;
	const char* trajectory = NULL;
	const char* init_file = NULL;
	struct noir_conf_s conf = noir_defaults;
	bool out_sens = false;
	bool scale_im = false;
	unsigned int turns = 1;
	int reduced_frames = 0;



	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('c', &conf.rvc, "Real-value constraint"),
		OPT_CLEAR('N', &normalize, "Do not normalize image with coil sensitivities"),
		OPT_UINT('m', &nmaps, "nmaps", "Number of ENLIVE maps to use in reconsctruction"),
		OPT_CLEAR('U', &combine, "Do not combine ENLIVE maps in output"),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_STRING('p', &psf, "PSF", ""),
		OPT_STRING('t', &trajectory, "Traj", ""),
		OPT_STRING('I', &init_file, "file", "File for initialization"),
		OPT_SET('g', &conf.usegpu, "use gpu"),
		OPT_SET('S', &scale_im, "Re-scale image after reconstruction"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPT_SET('P', &conf.pattern_for_each_coil, "(supplied psf is different for each coil)"),
		OPT_FLOAT('o', &oversampling, "os", "Oversampling factor for gridding [default: 1.5]"),
		OPT_FLOAT('T', &temp_damp, "temp_damp", "temporal damping [default: 0.9]"),
		OPT_INT('F', &reduced_frames, "Frames", "Reconstruct only 'noFrames' frames."),

	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (4 == argc)
		out_sens = true;

	num_init();

	if (NULL != psf && NULL != trajectory)
		error("Pass either trajectory (-t) OR PSF (-p)!\n");

	// kspace dimensions and strides struct
	struct ds_s* k_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	complex float* k = load_cfl(argv[1], DIMS, k_s->dims_full);
	ds_init(k_s, CFL_SIZE);
	long frames = k_s->dims_full[TIME_DIM];

	// Reduce kspace (-F)
	if( reduced_frames > 0 ) {

		if(reduced_frames > k_s->dims_full[TIME_DIM])
			error("Number of frames to be reconstructed (-F) must be smaller than number of acquired frames!");

		reduce_frames(reduced_frames, k_s->dims_full, &k);
		ds_init(k_s, CFL_SIZE);
		frames = reduced_frames;
	}
	// SMS
	if (1 != k_s->dims_full[SLICE_DIM]) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", k_s->dims_full[SLICE_DIM]);
		//fftmod(DIMS, k_s->dims_full, SLICE_FLAG, k, k); // fftmod to get correct slice order in output
	}

	complex float* pattern = NULL;
	struct ds_s* pat_s = (struct ds_s*) malloc(sizeof(struct ds_s));

	complex float* traj = NULL;
	struct ds_s* traj_s = (struct ds_s*) malloc(sizeof(struct ds_s));

	struct ds_s* sens_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_copy_dims(DIMS, sens_s->dims_full, k_s->dims_full);


	if (NULL != psf) {
		conf.noncart = true;
		complex float* tmp_psf =load_cfl(psf, DIMS, pat_s->dims_full);
		ds_init(pat_s, CFL_SIZE);
		pattern = anon_cfl("", DIMS, pat_s->dims_full);

		md_copy(DIMS, pat_s->dims_full, pattern, tmp_psf, CFL_SIZE);
		unmap_cfl(DIMS, pat_s->dims_full, tmp_psf);

		if (reduced_frames > 0) {
			reduce_frames(reduced_frames, pat_s->dims_full, &pattern);
			ds_init(pat_s, CFL_SIZE);
		}

		turns = pat_s->dims_full[TIME_DIM];

	} else if (NULL != trajectory) {
		conf.noncart = true;
		traj = load_cfl(trajectory, DIMS, traj_s->dims_full);

		// apply oversampling
		md_zsmul(DIMS, traj_s->dims_full, traj, traj, oversampling);
		ds_init(traj_s, CFL_SIZE);

		if (reduced_frames > 0) {
			reduce_frames(reduced_frames, traj_s->dims_full, &traj);
			ds_init(traj_s, CFL_SIZE);
		}

		turns = traj_s->dims_full[TIME_DIM];

		estimate_fast_sq_im_dims(3, sens_s->dims_full, traj_s->dims_full, traj);

	} else {
		error("Pass either trajectory (-t) or PSF (-p)!\n");
	}
	sens_s->dims_full[MAPS_DIM] = nmaps;
	ds_init(sens_s, CFL_SIZE);


	if (conf.pattern_for_each_coil)
		assert( 1 != pat_s->dims_full[COIL_DIM] );
	else if (-1 == restrict_fov)
		restrict_fov = 0.5;


	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconsctruction (ENLIVE)
	assert(1 == k_s->dims_full[MAPS_DIM]);

	struct ds_s* img_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_select_dims(DIMS, ~COIL_FLAG, img_s->dims_full, sens_s->dims_full);
	ds_init(img_s, CFL_SIZE);

	if (!combine) {
		// The conventional img-dimensions contain only one map.
		// The 'output' dimensions might contain multiple maps (ENLIVE)
		img_s->dims_output[MAPS_DIM] = nmaps;
		img_s->dims_output_singleFrame[MAPS_DIM] = nmaps;
	}

	complex float* img_output = create_cfl(argv[2], DIMS, img_s->dims_output);
	md_clear(DIMS, img_s->dims_output, img_output, CFL_SIZE);

	complex float* img_singleFrame = md_alloc(DIMS, img_s->dims_singleFrame, CFL_SIZE);

	struct ds_s* msk_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_select_dims(DIMS, FFT_FLAGS, msk_s->dims_full, img_s->dims_singleFrame);
	ds_init(msk_s, CFL_SIZE);

	complex float* mask = NULL;

	// Full output sensitivities
	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[3] : "", DIMS, sens_s->dims_full);

	// Sensitivities in image domain (relevant for output and normalization)
	complex float* sens_singleFrame = md_alloc(DIMS, sens_s->dims_singleFrame, CFL_SIZE);
	md_clear(DIMS, sens_s->dims_singleFrame, sens_singleFrame, CFL_SIZE);

	// Sensitivities in kspace-domain (relevant for reference and initialization)
	complex float* ksens_singleFrame = md_alloc(DIMS, sens_s->dims_singleFrame, CFL_SIZE);
	md_clear(DIMS, sens_s->dims_singleFrame, ksens_singleFrame, CFL_SIZE);


	// initialization
	if (NULL != init_file) {

		long init_dims[DIMS];
		complex float* init = load_cfl(init_file, DIMS, init_dims);

		if (!md_check_bounds(DIMS, 0, img_s->dims_singleFrame, init_dims))
			error("Image dimensions and init dimensions to not match!");

		md_copy(DIMS, img_s->dims_singleFrame, img_singleFrame, init, CFL_SIZE);
		md_clear(DIMS, sens_s->dims_singleFrame, ksens_singleFrame, CFL_SIZE);

		unmap_cfl(DIMS, init_dims, init);

	} else {

		md_zfill(DIMS, img_s->dims_singleFrame, img_singleFrame, 1.);
		md_clear(DIMS, sens_s->dims_singleFrame, ksens_singleFrame, CFL_SIZE);
	}



	// Gridding
	if (NULL != trajectory) {
		debug_printf(DP_DEBUG3, "Start gridding...");


		md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), pat_s->dims_full, sens_s->dims_full);
		ds_init(pat_s, CFL_SIZE);

		// calculate pattern by nufft gridding:
		// we generate an array of ones, and grid it in the same way we
		// would grid real data. This pretty picture gets slightly more
		// complicated when we consider (zero-padded) asymmetric echoes:
		// Here we do not want ones in the zero-padded beginning.
		// Therefore, we generate an array of ones ONLY where the
		// original data is != 0.

		long ones_dims[DIMS];
		md_copy_dims(DIMS, ones_dims, traj_s->dims_full);
		ones_dims[READ_DIM] = 1L;
		complex float* ones = md_alloc(DIMS, ones_dims, CFL_SIZE);
		md_clear(DIMS, ones_dims, ones, CFL_SIZE);

		complex float* k_dummy = md_alloc(DIMS, ones_dims, CFL_SIZE);
		md_clear(DIMS, ones_dims, k_dummy, CFL_SIZE);

		// copy k-space data of one coil into k_dummy
		long pos0[DIMS] = { 0 };
		md_copy_block(DIMS, pos0, ones_dims, k_dummy, k_s->dims_full, k, CFL_SIZE);

		// divide kspace-dummy data by itself.
		// zdiv makes sure that division by zero is set to 0
		md_zdiv(DIMS, ones_dims, ones, k_dummy, k_dummy);

		const struct linop_s* nufft_op = nufft_create(DIMS, ones_dims, pat_s->dims_full, traj_s->dims_full, traj, NULL, nufft_conf);
		pattern = md_alloc(DIMS, pat_s->dims_full, CFL_SIZE);
		linop_adjoint(nufft_op, DIMS, pat_s->dims_full, pattern, DIMS, ones_dims, ones);

		fftscale(DIMS, pat_s->dims_full, FFT_FLAGS, pattern, pattern);
		fftc(DIMS, pat_s->dims_full, FFT_FLAGS, pattern, pattern);

		scale_psf_k(pat_s, pattern, k_s, k, traj_s, traj);

		linop_free(nufft_op);
		md_free(ones);
		md_free(k_dummy);
		debug_printf(DP_DEBUG3, "finished\n");

	}

	struct ds_s* kgrid_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_select_dims(DIMS, ~MAPS_FLAG, kgrid_s->dims_full, sens_s->dims_full);
	ds_init(kgrid_s, CFL_SIZE);

	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, msk_s->dims_full, CFL_SIZE);
		md_zfill(DIMS, msk_s->dims_full, mask, 1.);

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;
		mask = compute_mask(DIMS, msk_s->dims_full, restrict_dims);
	}

	long skip = md_calc_size(DIMS, img_s->dims_singleFrame);
	long size = skip + md_calc_size(DIMS, sens_s->dims_singleFrame);



	long ref_dim[1] = { size };
	complex float* ref = md_alloc(1, ref_dim, CFL_SIZE);
	md_clear(1, ref_dim, ref, CFL_SIZE);

	struct linop_s* nufft_ops[turns];
	complex float* kgrid_singleFrame = NULL;

	const struct operator_s* fftc = NULL;
	complex float* fftc_mod = NULL;
	complex float* traj_singleFrame = NULL;

	if (NULL != trajectory) { 	// Crecte nufft objects
		debug_printf(DP_DEBUG3, "Start creating nufft-objects...");

		for (unsigned int i = 0; i < turns; ++i) {
			// pick trajectory for current frame
			traj_singleFrame = md_alloc(DIMS, traj_s->dims_singleFrame, CFL_SIZE);
			long pos[DIMS] = { 0 };
			pos[TIME_DIM] = i;
			md_slice(DIMS, TIME_FLAG, pos, traj_s->dims_full, traj_singleFrame, traj, CFL_SIZE);

			nufft_ops[i] = nufft_create(DIMS, k_s->dims_singleFrame, kgrid_s->dims_singleFrame, traj_s->dims_singleFrame, traj_singleFrame, NULL, nufft_conf);
		}

		kgrid_singleFrame = md_alloc(DIMS, kgrid_s->dims_singleFrame, CFL_SIZE);

		fftc = fft_measure_create(DIMS, kgrid_s->dims_singleFrame, FFT_FLAGS, true, false);

		fftc_mod = md_alloc(DIMS, kgrid_s->dims_singleFrame, CFL_SIZE);

		md_zfill(DIMS, kgrid_s->dims_singleFrame, fftc_mod, 1.);
		fftmod(DIMS, kgrid_s->dims_singleFrame, FFT_FLAGS, fftc_mod, fftc_mod);
		debug_printf(DP_DEBUG3, "finished\n");

	}


	debug_printf(DP_DEBUG3, "Start reconstruction\n");
	complex float* img_output_singleFrame = md_alloc(DIMS, img_s->dims_output_singleFrame, CFL_SIZE);
	complex float* sens_output_singleFrame = md_alloc(DIMS, sens_s->dims_singleFrame, CFL_SIZE);
	complex float* k_singleFrame = md_alloc(DIMS, k_s->dims_singleFrame, CFL_SIZE);
	complex float* pat_singleFrame = md_alloc(DIMS, pat_s->dims_singleFrame, CFL_SIZE);

	for (unsigned int frame = 0; frame < frames; ++frame) {

		debug_printf(DP_DEBUG1, "Reconstructing frame %d\n", frame);

		// pick k-space and pattern for current frame
		long pos[DIMS] = { 0 };
		pos[TIME_DIM] = frame;
		md_slice(DIMS, TIME_FLAG, pos, k_s->dims_full, k_singleFrame, k, CFL_SIZE);
		pos[TIME_DIM] = frame%turns;
		md_slice(DIMS, TIME_FLAG, pos, pat_s->dims_full, pat_singleFrame, pattern, CFL_SIZE);

		if (NULL != trajectory)  {
			// grid data frame by frame
			linop_adjoint(nufft_ops[frame % turns], DIMS, kgrid_s->dims_singleFrame, kgrid_singleFrame, DIMS, k_s->dims_singleFrame, k_singleFrame);

			md_zmul(DIMS, kgrid_s->dims_singleFrame, kgrid_singleFrame, kgrid_singleFrame, fftc_mod);
			fft_exec(fftc, kgrid_singleFrame, kgrid_singleFrame);
			md_zmul(DIMS, kgrid_s->dims_singleFrame, kgrid_singleFrame, kgrid_singleFrame, fftc_mod);
		} else {

			kgrid_singleFrame = k_singleFrame;
		}


		double scaling = 100. / md_znorm(DIMS, kgrid_s->dims_singleFrame, kgrid_singleFrame);
// 		if (1 != kgrid_s->dims_singleFrame[SLICE_DIM]) // SMS
// 			scaling *= sqrt(kgrid_s->dims_singleFrame[SLICE_DIM]);
		md_zsmul(DIMS, kgrid_s->dims_singleFrame, kgrid_singleFrame, kgrid_singleFrame, scaling);

#ifdef  USE_CUDA
		if (conf.usegpu) {

			complex float* kgrid_gpu_singleFrame = md_alloc_gpu(DIMS, kgrid_s->dims_singleFrame, CFL_SIZE);
			md_copy(DIMS, kgrid_s->dims_singleFrame, kgrid_gpu_singleFrame, kgrid_singleFrame, CFL_SIZE);

			noir_recon(&conf, sens_s->dims_singleFrame, img_singleFrame, sens_singleFrame, ksens_singleFrame, ref, pat_singleFrame, mask, kgrid_gpu_singleFrame);
			md_free(kgrid_gpu_singleFrame);
		} else
#endif
			noir_recon(&conf, sens_s->dims_singleFrame, img_singleFrame, sens_singleFrame, ksens_singleFrame, ref, pat_singleFrame, mask, kgrid_singleFrame);


		// Temporal regularization
		md_zsmul(DIMS, img_s->dims_singleFrame, ref, img_singleFrame, temp_damp);
		md_zsmul(DIMS, sens_s->dims_singleFrame, ref + skip, ksens_singleFrame, temp_damp);

		// image output
		if (normalize) {

			complex float* buf = md_alloc(DIMS, sens_s->dims_singleFrame, CFL_SIZE);
			md_clear(DIMS, sens_s->dims_singleFrame, buf, CFL_SIZE);

			if (combine) {

				md_zfmac2(DIMS, sens_s->dims_singleFrame, kgrid_s->strs_singleFrame, buf, img_s->strs_singleFrame, img_singleFrame, sens_s->strs_singleFrame, sens_singleFrame);
				md_zrss(DIMS, kgrid_s->dims_singleFrame, COIL_FLAG, img_output_singleFrame, buf);

			} else {

				md_zfmac2(DIMS, sens_s->dims_singleFrame, sens_s->strs_singleFrame, buf, img_s->strs_singleFrame, img_singleFrame, sens_s->strs_singleFrame, sens_singleFrame);
				md_zrss(DIMS, sens_s->dims_singleFrame, COIL_FLAG, img_output_singleFrame, buf);


			}
			md_zmul2(DIMS, img_s->dims_output_singleFrame, img_s->strs_output_singleFrame, img_output_singleFrame, img_s->strs_output_singleFrame, img_output_singleFrame, msk_s->strs_singleFrame, mask);

			if (1 == nmaps || !combine) {

				// restore phase
				md_zphsr(DIMS, img_s->dims_output_singleFrame, buf, img_singleFrame);
				md_zmul(DIMS, img_s->dims_output_singleFrame, img_output_singleFrame, img_output_singleFrame, buf);
			}

			md_free(buf);
		} else {

			if (combine) {

				// just sum up the map images
				md_zaxpy2(DIMS, img_s->dims_singleFrame, img_s->strs_output_singleFrame, img_output_singleFrame, 1., img_s->strs_singleFrame, img_singleFrame);
			} else { // !normalize && !combine

				// just copy
				md_copy(DIMS, img_s->dims_output_singleFrame, img_output_singleFrame, img_singleFrame, CFL_SIZE);
			}
		}

		if (scale_im)
			md_zsmul(DIMS, img_s->dims_output_singleFrame, img_output_singleFrame, img_output_singleFrame, 1. / scaling);

		// Copy frame to correct position in output array
		long pos2[DIMS] = { 0 };
		pos2[TIME_DIM] = frame;
		md_copy_block(DIMS, pos2, img_s->dims_output, img_output, img_s->dims_output_singleFrame, img_output_singleFrame, CFL_SIZE);
		if(out_sens)
			md_copy_block(DIMS, pos2, sens_s->dims_full, sens, sens_s->dims_singleFrame, sens_singleFrame, CFL_SIZE);

	}

	md_free(mask);
	md_free(img_singleFrame);
	md_free(k_singleFrame);
	md_free(pat_singleFrame);
	md_free(sens_output_singleFrame);
	md_free(img_output_singleFrame);
	md_free(sens_singleFrame);
	md_free(ksens_singleFrame);
	md_free(ref);

	if (NULL != trajectory) {
		md_free(traj_singleFrame);
		md_free(kgrid_singleFrame);
		md_free(fftc_mod);
	}

	unmap_cfl(DIMS, sens_s->dims_full, sens);
	unmap_cfl(DIMS, pat_s->dims_full, pattern);
	unmap_cfl(DIMS, img_s->dims_output, img_output);
	unmap_cfl(DIMS, k_s->dims_full, k);


	free(k_s);
	free(img_s);
	free(sens_s);
	free(msk_s);
	free(kgrid_s);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}


