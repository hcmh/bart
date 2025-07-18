/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker
 * 2014 Jonathan Tamir
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/iovec.h"

#include "linops/someops.h"
#include "linops/linop.h"

#include "wavelet/wavthresh.h"

#include "iter/iter.h"
#include "iter/prox.h"
#include "iter/thresh.h"

#include "sense/pocs.h"
#include "sense/optcom.h"

#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"





static const char help_str[] = "Perform POCSENSE reconstruction.";

	

int main_pocsense(int argc, char* argv[argc])
{
	const char* ksp_file = NULL;
	const char* sens_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INFILE(true, &sens_file, "sensitivities"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	float alpha = 0.;
	int maxiter = 50;
	bool l1wav = false;
	float lambda = -1.;
	bool use_admm = false;
	float admm_rho = -1.;
	int l1type = 2;

	const struct opt_s opts[] = {

		OPT_INT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_FLOAT('r', &alpha, "alpha", "regularization parameter"),
		OPT_PINT('l', &l1type, "1/-l2", "toggle l1-wavelet or l2 regularization"),
		OPT_SET('g', &bart_use_gpu, "()"),
		OPT_FLOAT('o', &lambda, "", "()"),
		OPT_FLOAT('m', &admm_rho, "", "()"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (1 == l1type)
		l1wav = true;
	else
	if (2 == l1type)
		l1wav = false;
	else
		error("Unknown regularization type.\n");

	
	int N = DIMS;

	long dims[N];
	long ksp_dims[N];

	complex float* kspace_data = load_cfl(ksp_file, N, ksp_dims);
	complex float* sens_maps = load_cfl(sens_file, N, dims);


	for (int i = 0; i < 4; i++)	// sizes2[4] may be > 1
		if (ksp_dims[i] != dims[i])
			error("Dimensions of kspace and sensitivities do not match!\n");

	assert(1 == ksp_dims[MAPS_DIM]);

	num_init_gpu_support();


	
	long dims1[N];
	
	md_select_dims(N, ~(COIL_FLAG|MAPS_FLAG), dims1, dims);


	// -----------------------------------------------------------
	// memory allocation
	
	complex float* result = create_cfl(out_file, N, ksp_dims);
	complex float* pattern = md_alloc(N, dims1, CFL_SIZE);


	// -----------------------------------------------------------
	// pre-process data
	
	float scaling = estimate_scaling(ksp_dims, NULL, kspace_data);
	md_zsmul(N, ksp_dims, kspace_data, kspace_data, 1. / scaling);

	estimate_pattern(N, ksp_dims, COIL_FLAG, pattern, kspace_data);


	// -----------------------------------------------------------
	// l1-norm threshold operator
	
	const struct operator_p_s* thresh_op = NULL;

	if (l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };

		unsigned long flags = 0;

		for (int i = 0; i < DIMS; i++) {

			if ((1 < ksp_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {

				flags = MD_SET(flags, i);
				minsize[i] = MIN(ksp_dims[i], 16);
			}
		}

		bool randshift = false;
		thresh_op = prox_wavelet_thresh_create(DIMS, ksp_dims, flags, COIL_FLAG, WAVELET_DAU2, minsize, alpha, randshift);
	}
#if 0
	else {
		thresh_op = prox_leastsquares_create(DIMS, ksp_dims, alpha, NULL);
	}
#endif

	// -----------------------------------------------------------
	// italgo interface
	
	italgo_fun2_t italgo = NULL;
	iter_conf* iconf = NULL;

	struct iter_pocs_conf pconf = iter_pocs_defaults;
	pconf.maxiter = maxiter;

	struct iter_admm_conf mmconf = iter_admm_defaults;
	mmconf.maxiter = maxiter;
	mmconf.rho = admm_rho;

	struct linop_s* eye = linop_identity_create(DIMS, ksp_dims);
	const struct linop_s* ops[3] = { eye, eye, eye };
	const struct linop_s** ops2 = NULL;

	if (use_admm) {

		italgo = iter2_admm;
		iconf = CAST_UP(&mmconf);
		ops2 = ops;

	} else {

		italgo = iter2_pocs;
		iconf = CAST_UP(&pconf);
	}


	// -----------------------------------------------------------
	// pocsense recon

	debug_printf(DP_INFO, "Reconstruction...\n");
	
	fftmod(N, ksp_dims, FFT_FLAGS, kspace_data, kspace_data);

	if (bart_use_gpu)
#ifdef USE_CUDA
		pocs_recon_gpu2(italgo, iconf, ops2, dims, thresh_op, alpha, lambda, result, sens_maps, pattern, kspace_data);
#else
		assert(0);
#endif
	else
		pocs_recon2(italgo, iconf, ops2, dims, thresh_op, alpha, lambda, result, sens_maps, pattern, kspace_data);

	ifftmod(N, ksp_dims, FFT_FLAGS, result, result);


	debug_printf(DP_INFO, "Done.\n");

	md_zsmul(N, ksp_dims, result, result, scaling);

	linop_free(eye);

	md_free(pattern);
	
	if (NULL != thresh_op)
		operator_p_free(thresh_op);

	unmap_cfl(N, ksp_dims, result);
	unmap_cfl(N, ksp_dims, kspace_data);
	unmap_cfl(N, dims, sens_maps);

	return 0;
}


