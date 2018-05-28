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
	const char* psf = NULL;
	const char* trajectory = NULL;
	const char* init_file = NULL;
	struct noir_conf_s conf = noir_defaults;
	bool out_sens = false;
	bool scale_im = false;
	bool ungridded_data = false;


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
	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (4 == argc)
		out_sens = true;



	num_init();

	// kspace dimensions and strides struct
	struct ds_s* k_s = (struct ds_s*) malloc(sizeof(struct ds_s));

	complex float* kspace_data = load_cfl(argv[1], DIMS, k_s->dims_full);
	ds_init(k_s, CFL_SIZE);

	// SMS
	if (1 != k_s->dims_full[SLICE_DIM]) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", k_s->dims_full[SLICE_DIM]);
		fftmod(DIMS, k_s->dims_full, SLICE_FLAG, kspace_data, kspace_data); // fftmod to get correct slice order in output
	}

	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconsctruction
	assert(1 == k_s->dims_full[MAPS_DIM]);

	struct ds_s* sens_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_copy_dims(DIMS, sens_s->dims_full, k_s->dims_full);
	sens_s->dims_full[MAPS_DIM] = nmaps;
	ds_init(sens_s, CFL_SIZE);

	struct ds_s* img_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|SLICE_FLAG, img_s->dims_full, sens_s->dims_full); // we never have frames for img
	ds_init(img_s, CFL_SIZE);


	if (!combine)
		img_s->dims_output[MAPS_DIM] = nmaps;

	complex float* img_output = create_cfl(argv[2], DIMS, img_s->dims_output);
	md_clear(DIMS, img_s->dims_output, img_output, CFL_SIZE);
	complex float* img = md_alloc(DIMS, img_s->dims_full, CFL_SIZE);

	struct ds_s* msk_s = (struct ds_s*) malloc(sizeof(struct ds_s));

	md_select_dims(DIMS, FFT_FLAGS, msk_s->dims_full, img_s->dims_full);
	ds_init(msk_s, CFL_SIZE);

	complex float* mask = NULL;

	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[3] : "", DIMS, sens_s->dims_full);

	// initialization
	if (NULL != init_file) {

		long skip = md_calc_size(DIMS, img_s->dims_full);
		struct ds_s* init_s  = (struct ds_s*) malloc(sizeof(struct ds_s));
		complex float* init = load_cfl(init_file, DIMS, init_s->dims_full);
		ds_init(init_s, CFL_SIZE);

		assert(md_check_bounds(DIMS, 0, img_s->dims_full, init_s->dims_full));

		md_copy(DIMS, img_s->dims_full, img, init, CFL_SIZE);
		fftmod(DIMS, sens_s->dims_full, FFT_FLAGS|SLICE_FLAG, sens, init + skip);

		unmap_cfl(DIMS, init_s->dims_full, init);

	} else {

		md_zfill(DIMS, img_s->dims_full, img, 1.);
		md_clear(DIMS, sens_s->dims_full, sens, CFL_SIZE);
	}

	if (NULL != psf && NULL != trajectory)
		error("Pass either trajectory (-t) OR PSF (-p)!\n");

	complex float* pattern = NULL;
	struct ds_s* pat_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	complex float* traj = NULL;
	struct ds_s* traj_s = (struct ds_s*) malloc(sizeof(struct ds_s));

	unsigned int turns = 1;

	if (NULL != psf) {

		complex float* tmp_psf =load_cfl(psf, DIMS, pat_s->dims_full);
		ds_init(pat_s, CFL_SIZE);
		pattern = anon_cfl("", DIMS, pat_s->dims_full);

		md_copy(DIMS, pat_s->dims_full, pattern, tmp_psf, CFL_SIZE);
		unmap_cfl(DIMS, pat_s->dims_full, tmp_psf);
		// FIXME: check compatibility

		if (conf.pattern_for_each_coil) {
			assert( 1 != pat_s->dims_full[COIL_DIM] );
		} else {
			if (-1 == restrict_fov)
				restrict_fov = 0.5;

			conf.noncart = true;
		}

		turns = pat_s->dims_full[TIME_DIM];

	} else if (NULL != trajectory) {

		traj = load_cfl(trajectory, DIMS, traj_s->dims_full);
		ds_init(traj_s, CFL_SIZE);

		turns = traj_s->dims_full[TIME_DIM];

	} else {

		md_copy_dims(DIMS, pat_s->dims_full, img_s->dims_full);
		pattern = anon_cfl("", DIMS, pat_s->dims_full);
		estimate_pattern(DIMS, k_s->dims_full, COIL_FLAG, pattern, kspace_data);
	}

	// Gridding
	if (NULL != trajectory) {

		ungridded_data = true;
		// apply oversampling
		md_zsmul(DIMS, traj_s->dims_full, traj, traj, oversampling);

		estimate_fast_sq_im_dims(3, sens_s->dims_full, traj_s->dims_full, traj);
		ds_init(sens_s, CFL_SIZE);

		md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), pat_s->dims_full, sens_s->dims_full);
		pat_s->dims_full[TIME_DIM] = turns;
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
		md_copy_block(DIMS, pos0, ones_dims, k_dummy, k_s->dims_full, kspace_data, CFL_SIZE);

		// divide kspace-dummy data by itself.
		// zdiv makes sure that division by zero is set to 0
		md_zdiv(DIMS, ones_dims, ones, k_dummy, k_dummy);

		const struct linop_s* nufft_op = nufft_create(DIMS, ones_dims, pat_s->dims_full, traj_s->dims_full, traj, NULL, nufft_conf);
		pattern = md_alloc(DIMS, pat_s->dims_full, CFL_SIZE);
		linop_adjoint(nufft_op, DIMS, pat_s->dims_full, pattern, DIMS, ones_dims, ones);

		fftscale(DIMS, pat_s->dims_full, FFT_FLAGS, pattern, pattern);
		fftc(DIMS, pat_s->dims_full, FFT_FLAGS, pattern, pattern);

		scale_psf_k(pat_s, pattern, k_s, kspace_data, traj_s, traj);

		linop_free(nufft_op);
		md_free(ones);
		md_free(k_dummy);
	}

	// k-space normalization
#if 0
	float scaling = 1. / estimate_scaling(k_s->dims_full, NULL, kspace_data);
#else
	double scaling = 100. / md_znorm(DIMS, k_s->dims_full, kspace_data);

	if (1 != k_s->dims_full[SLICE_DIM]) // SMS
			scaling *= sqrt(k_s->dims_full[SLICE_DIM]);

#endif

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

	long skip = md_calc_size(DIMS, img_s->dims_full);
	long size = skip + md_calc_size(DIMS, sens_s->dims_full);

	long d1[1] = { size };
	complex float* ref = md_alloc(1, d1, CFL_SIZE);
	md_clear(DIMS, img_s->dims_full, ref, CFL_SIZE);
	md_clear(DIMS, k_s->dims_full, ref + skip, CFL_SIZE);

#ifdef  USE_CUDA
	if (conf.usegpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, k_s->dims_full, CFL_SIZE);
		md_copy(DIMS, k_s->dims_full, kspace_gpu, kspace_data, CFL_SIZE);

		noir_recon(&conf, sens_s->dims_full, img, sens, ref, pattern, mask, kspace_gpu);
		md_free(kspace_gpu);
	} else
#endif
		noir_recon(&conf, sens_s->dims_full, img, sens, ref, pattern, mask, kspace_data);


	// image output
	if (normalize) {

		complex float* buf = md_alloc(DIMS, sens_s->dims_full, CFL_SIZE);
		md_clear(DIMS, sens_s->dims_full, buf, CFL_SIZE);

		if (combine) {

			md_zfmac2(DIMS, sens_s->dims_full, k_s->strs_full, buf, img_s->strs_full, img, sens_s->strs_full, sens);
			md_zrss(DIMS, k_s->dims_full, COIL_FLAG, img_output, buf);
		} else {

			md_zfmac2(DIMS, sens_s->dims_full, sens_s->strs_full, buf, img_s->strs_full, img, sens_s->strs_full, sens);
			md_zrss(DIMS, sens_s->dims_full, COIL_FLAG, img_output, buf);
		}
		md_zmul2(DIMS, img_s->dims_output, img_s->strs_output, img_output, img_s->strs_output, img_output, msk_s->strs_full, mask);

		if (1 == nmaps || !combine) {

			//restore phase
			md_zphsr(DIMS, img_s->dims_output, buf, img);
			md_zmul(DIMS, img_s->dims_output, img_output, img_output, buf);
		}

		md_free(buf);
	} else {

		if (combine) {

			// just sum up the map images
			md_zaxpy2(DIMS, img_s->dims_full, img_s->strs_output, img_output, 1., img_s->strs_full, img);
		} else { /*!normalize && !combine */

			// Just copy
			md_copy(DIMS, img_s->dims_output, img_output, img, CFL_SIZE);
		}
	}

	if (scale_im)
		md_zsmul(DIMS, img_s->dims_output, img_output, img_output, 1. / scaling);

	md_free(mask);
	md_free(img);

	unmap_cfl(DIMS, sens_s->dims_full, sens);
	unmap_cfl(DIMS, pat_s->dims_full, pattern);
	unmap_cfl(DIMS, img_s->dims_output, img_output);
	unmap_cfl(DIMS, k_s->dims_full, kspace_data);

	free(k_s);
	free(img_s);
	free(sens_s);
	free(msk_s);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}


