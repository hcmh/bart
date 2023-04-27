/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Publications:
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction
 * by regularized nonlinear inversion-joint estimation of coil
 * sensitivities and image content. Magn Reson Med 2008; 60:674-682.
 *
 * Uecker M, Zhang S, Frahm J. Nonlinear Inverse Reconstruction for
 * Real-time MRI of the Human Heart Using Undersampled Radial FLASH.
 * Magn Reson Med 2010; 63:1456-1462.
 *
 * Holme HCM, Rosenzweig S, Ong F, Wilke RN, Lustig M, Uecker M.
 * ENLIVE: An Efficient Nonlinear Method for Calibrationless and
 * Robust Parallel Imaging. Sci Rep 2019; 9:3034.
 *
 * Rosenzweig S, Holme HMC, Wilke RN, Voit D, Frahm J, Uecker M.
 * Simultaneous multi-slice MRI using cartesian and radial FLASH and
 * regularized nonlinear inversion: SMS-NLINV.
 * Magn Reson Med 2018; 79:2057--2066.
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "noncart/nufft.h"
#include "linops/linop.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noir/recon2.h"
#include "noir/misc.h"





static const char help_str[] =
		"Jointly estimate image and sensitivities with nonlinear\n"
		"inversion using {iter} iteration steps. Optionally outputs\n"
		"the sensitivities.";



int main_nlinv(int argc, char* argv[argc])
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

	bool normalize = true;
	bool combine = true;
	unsigned int nmaps = 1;
	float restrict_fov = -1.;
	const char* psf_file = NULL;
	const char* basis_file = NULL;
	const char* trajectory = NULL;
	const char* init_file = NULL;
	struct noir2_conf_s conf = noir2_defaults;
	bool nufft_lowmem = false;
	bool psf_based_reco = false;

	long my_img_dims[3] = { 0, 0, 0 };

	unsigned int cnstcoil_flags = 0;
	bool pattern_for_each_coil = false;

	float coil_os_dims = -1;

	bool crop_sens = false;


	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_FLOAT('M', &conf.alpha_min, "", "(minimum for regularization)"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('c', &conf.rvc, "Real-value constraint"),
		OPT_CLEAR('N', &normalize, "Do not normalize image with coil sensitivities"),
		OPT_UINT('m', &nmaps, "nmaps", "Number of ENLIVE maps to use in reconstruction"),
		OPT_CLEAR('U', &combine, "Do not combine ENLIVE maps in output"),
		OPTL_SET(0, "crop-sens", &crop_sens, "Crop sensitivities to image size"),
		OPT_FLOAT('f', &restrict_fov, "FOV", "restrict FOV"),
		OPT_INFILE('p', &psf_file, "file", "pattern / transfer function"),
		OPTL_SET(0, "psf-based", &psf_based_reco, "(use psf based reconstruction)"),
		OPT_INFILE('t', &trajectory, "file", "kspace trajectory"),
		OPT_INFILE('I', &init_file, "file", "File for initialization"),
		OPT_SET('g', &(conf.gpu), "use gpu"),
		OPT_SET('S', &(conf.undo_scaling), "Re-scale image after reconstruction"),
		OPT_UINT('s', &cnstcoil_flags, "", "(dimensions with constant sensitivities)"),
		OPT_FLOAT('a', &conf.a, "", "(a in c * (1 + a * \\Laplace^-b/2))"),
		OPT_FLOAT('b', &conf.b, "", "(b in c * (1 + a * \\Laplace^-b/2))"),
		OPTL_FLOAT(0, "scaling-coils", &conf.scaling_coils, "", "(b in c * (1 + a * \\Laplace^-b/2))"),
		OPT_SET('P', &pattern_for_each_coil, "(supplied psf is different for each coil)"),
		OPT_SET('n', &conf.noncart, "(non-Cartesian)"),
		OPT_FLOAT('w', &conf.scaling, "", "(inverse scaling of the data)"),
		OPT_INFILE('B', &basis_file, "file", "temporal (or other) basis"),
		OPTL_SET(0, "lowmem", &nufft_lowmem, "Use low-mem mode of the nuFFT"),
		OPTL_VEC3('x', "dims", &my_img_dims, "x:y:z", "Explicitly specify image dimensions"),
		OPT_ULONG('L', &(conf.loop_flags), "flags", "(batch-mode)"),
		OPTL_INT(0, "cgiter", &conf.cgiter, "iter", "(iterations for linearized problem)"),
		OPTL_FLOAT(0, "cgtol", &conf.cgtol, "tol", "(tolerance for linearized problem)"),
		OPTL_FLOAT(0, "alpha", &conf.alpha, "val", "(alpha in first iteration)"),
		OPTL_FLOAT(0, "coil-os", &(conf.oversampling_coils), "val", "(over-sampling factor for sensitivities)"),
		OPTL_SET(0, "real-time", &(conf.realtime), "Use real-time (temporal l2) regularization"),
		OPTL_INT(0, "real-time-skip", &(conf.realtime_skip), "n", "Skip first n Gauß-Newton steps"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


	(conf.gpu ? num_init_gpu : num_init)();

	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	// FIXME: SMS should not be the default

	if ((1 != ksp_dims[SLICE_DIM] && !MD_IS_SET(conf.loop_flags, SLICE_DIM))) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		conf.sms = true;
	}

	
	const complex float* basis = NULL;
	long bas_dims[DIMS];

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, bas_dims);
	} else {

		md_singleton_dims(DIMS, bas_dims);
	}

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != psf_file) {

		pattern = load_cfl(psf_file, DIMS, pat_dims);
	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}

	psf_based_reco = psf_based_reco || (-1 != restrict_fov) || (NULL != init_file);

	if (use_compat_to_version("v0.8.00"))
		psf_based_reco = true;

	if ((psf_based_reco) && (NULL != trajectory)) {

		assert(-1. == coil_os_dims);
		coil_os_dims = 1;

		assert(NULL == psf_file);

		conf.noncart = true;

		long dims[DIMS];
		long trj_dims[DIMS];
		long psf_dims[DIMS];

		complex float* traj = load_cfl(trajectory, DIMS, trj_dims);

		if (0 == md_calc_size(3, my_img_dims)) {

			estimate_im_dims(DIMS, FFT_FLAGS, dims, trj_dims, traj);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);
		} else 
			md_copy_dims(3, dims, my_img_dims);

		md_zsmul(DIMS, trj_dims, traj, traj, 2.);

		for (unsigned int i = 0; i < DIMS; i++)
			if (MD_IS_SET(FFT_FLAGS, i) && (1 < dims[i]))
				dims[i] *= 2;
		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);

		debug_printf(DP_DEBUG3, "Start gridding psf ...");
		md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), psf_dims, dims);
		complex float* psf = compute_psf(DIMS, psf_dims, trj_dims, traj, trj_dims, NULL, pat_dims, pattern, false, nufft_lowmem);
		fftuc(DIMS, psf_dims, FFT_FLAGS, psf, psf);
		float psf_sc = 1.;
		for (int i = 0; i < 3; i++)
			if (1 != psf_dims[i])
				psf_sc *= 2.;
		md_zsmul(DIMS, psf_dims, psf, psf, psf_sc);

		unmap_cfl(DIMS, pat_dims, pattern);
		md_copy_dims(DIMS, pat_dims, psf_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		md_copy(DIMS, pat_dims, pattern, psf, CFL_SIZE);
		md_free(psf);
		debug_printf(DP_DEBUG3, "finished\n");


		debug_printf(DP_DEBUG3, "Start creating nufft-objects...");
		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = false;
		nufft_conf.lowmem = nufft_lowmem;
		const struct linop_s* nufft_op = nufft_create(DIMS, ksp_dims, dims, trj_dims, traj, NULL, nufft_conf);
		debug_printf(DP_DEBUG3, "finished\n");

		complex float* kgrid = anon_cfl("", DIMS, dims);
		linop_adjoint(nufft_op, DIMS, dims, kgrid, DIMS, ksp_dims, kspace);
		linop_free(nufft_op);
		fftuc(DIMS, dims, FFT_FLAGS, kgrid, kgrid);

		if (!use_compat_to_version("v0.7.00")) {

			float sc = 1.;
			for (int i = 0; i < 3; i++)
				if (1 != dims[i])
					sc *= 2.;

			md_zsmul(DIMS, dims, kgrid, kgrid, sqrtf(sc));
		}

		unmap_cfl(DIMS, ksp_dims, kspace);
		kspace = kgrid;

		md_copy_dims(DIMS, ksp_dims, dims);

		unmap_cfl(DIMS, trj_dims, traj);
		trajectory = NULL;
	}

	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconstruction
	assert(1 == ksp_dims[MAPS_DIM]);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	long dims[DIMS];
	md_copy_dims(DIMS, dims, ksp_dims);

	long trj_dims[DIMS];
	complex float* traj  = NULL;

	if (NULL != trajectory) {

		conf.noncart = true;

		traj = load_cfl(trajectory, DIMS, trj_dims);

		md_copy_dims(3, dims, my_img_dims);

		if (0 == md_calc_size(3, dims)) {

			estimate_im_dims(DIMS, FFT_FLAGS, dims, trj_dims, traj);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);
		}

		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);
	}

	// for ENLIVE maps
	dims[MAPS_DIM] = nmaps;

	if (NULL != basis) {

		assert(1 == ksp_dims[COEFF_DIM]);
		assert(bas_dims[TE_DIM] == ksp_dims[TE_DIM]);
		
		if (conf.noncart)
			assert(1 == md_calc_size(5, bas_dims));
		else
			md_check_compat(5, ~0, bas_dims, dims);

		dims[COEFF_DIM] = bas_dims[COEFF_DIM];
		dims[TE_DIM] = 1;
		cnstcoil_flags = cnstcoil_flags | COEFF_FLAG;
	}

	long sens_dims[DIMS];
	md_select_dims(DIMS, ~cnstcoil_flags, sens_dims, dims);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	long img_output_dims[DIMS];
	md_copy_dims(DIMS, img_output_dims, img_dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, dims);

	if ((conf.noncart) && (NULL == traj)) {

		assert((-1 == coil_os_dims) || (1 == coil_os_dims));
		coil_os_dims = 1;

		for (int i = 0; i < 3; i++)
			if (1 != img_output_dims[i])
				img_output_dims[i] /= 2;
	}

	if (-1 == coil_os_dims)
		coil_os_dims = (conf.noncart && (-1 == conf.oversampling_coils)) ? 2 : 1;

	for (int i = 0; i < 3; i++)
		if ((1 != sens_dims[i]) && !(MD_IS_SET(conf.loop_flags, i)))
			sens_dims[i] = lround(coil_os_dims * sens_dims[i]);

	if (combine)
		img_output_dims[MAPS_DIM] = 1;

	complex float* img_output = create_cfl(img_file, DIMS, img_output_dims);
	md_clear(DIMS, img_output_dims, img_output, CFL_SIZE);


	complex float* img = md_alloc(DIMS, img_dims, CFL_SIZE);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

	complex float* mask = NULL;

	complex float* ksens = md_alloc(DIMS, sens_dims, CFL_SIZE);
	complex float* sens = md_alloc(DIMS, sens_dims, CFL_SIZE);

	// initialization
	if (NULL != init_file) {

		long skip = md_calc_size(DIMS, img_dims);
		long init_dims[DIMS];

		complex float* init = load_cfl(init_file, DIMS, init_dims);

		assert(md_calc_size(DIMS, init_dims) == (md_calc_size(DIMS, img_dims) + md_calc_size(DIMS, sens_dims)));

		md_copy(DIMS, img_dims, img, init, CFL_SIZE);
		fftmod(DIMS, sens_dims, FFT_FLAGS | (conf.sms ? SLICE_FLAG : 0u), ksens, init + skip);

		unmap_cfl(DIMS, init_dims, init);

	} else {

		md_zfill(DIMS, img_dims, img, 1.);
		md_clear(DIMS, sens_dims, ksens, CFL_SIZE);
	}

	if ((-1 == restrict_fov) && conf.noncart && (NULL == trajectory))
		restrict_fov = 0.5;

	if (-1. != restrict_fov){

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;
		mask = compute_mask(DIMS, msk_dims, restrict_dims);
	}

	complex float* ref_img = NULL;
	complex float* ref_sens = NULL;

	if (NULL != traj) {

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = true;
		nufft_conf.lowmem = nufft_lowmem;
		nufft_conf.pcycle = false;
		nufft_conf.periodic = false;
		nufft_conf.cache_psf_grdding = true;
		conf.nufft_conf = &nufft_conf;

		noir2_recon_noncart(&conf, DIMS,
			img_dims, img, ref_img,
			sens_dims, sens, ksens, ref_sens,
			ksp_dims, kspace,
			trj_dims, traj,
			pat_dims, pattern,
			bas_dims, basis,
			msk_dims, mask,
			cim_dims);

	} else {

		noir2_recon_cart(&conf, DIMS,
			img_dims, img, ref_img,
			sens_dims, sens, ksens, ref_sens,
			ksp_dims, kspace,
			pat_dims, pattern,
			bas_dims, basis,
			msk_dims, mask,
			cim_dims);
	}

	unmap_cfl(DIMS, ksp_dims, kspace);

	postprocess2(normalize,
		     sens_dims, sens,
		     img_dims, img,
		     img_output_dims, img_output);


	md_free(mask);
	md_free(img);

	if (NULL != basis)
		unmap_cfl(DIMS, bas_dims, basis);

	if (NULL != traj)
		unmap_cfl(DIMS, trj_dims, traj);

	if (NULL != sens_file) {

		long sens_dims_out[DIMS];
		md_copy_dims(DIMS, sens_dims_out, sens_dims);

		if (crop_sens)
			for (int i = 0; i < 3; i++)
				sens_dims_out[i] = img_output_dims[i];

		complex float* sens_out = create_cfl(sens_file, DIMS, sens_dims_out);
		md_resize_center(DIMS, sens_dims_out, sens_out, sens_dims, sens, CFL_SIZE);
		unmap_cfl(DIMS, sens_dims_out, sens_out);
	}

	md_free(ksens);
	md_free(sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_output_dims, img_output);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	return 0;
}


