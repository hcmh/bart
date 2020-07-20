/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/mem.h"
#include "iter/iter6.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"

#include "nn/vn.h"
#include "nn/nn_modl.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <mask> <weights> <out>";
static const char help_str[] = "Trains and appplies the MoDL.";

int main_nnmodl(int argc, char* argv[])
{
	struct modl_s modl = modl_default;
	struct iter6_adam_conf train_conf = iter6_adam_conf_defaults;

	bool train = false;
	bool apply = false;
	bool use_gpu = false;
	bool initialize = false;
	bool normalize = false;

	bool random_order = false;
	char* history_filename = NULL;

	long udims[5] = {1, 1, 1, 1, 1};

	const struct opt_s opts[] = {

		OPT_SET('i', &initialize, "initialize weights"),
		OPT_LONG('T', &(modl.Nt), "guessed", "number of iterations"),
		OPT_LONG('L', &(modl.Nl), "guessed", "number of layers"),
		OPT_LONG('F', &(modl.Nf), "guessed", "number of convolution filters"),
		OPT_LONG('K', &(modl.Kx), "guessed", "filtersize"),

		OPT_SET('p', &(modl.nullspace), "use nullspace projection"),

		OPT_SET('t', &train, "train modl"),
		OPT_INT('E', &(train_conf.epochs), "1000", "number of training epochs"),
		OPT_LONG('B', &(modl.Nb), "10", "batch size"),
		OPT_SET('r', &(random_order), "randomize batches"),

		OPT_SET('a', &apply, "apply modl"),

		OPT_SET('n', &normalize, "normalize "),
		OPT_SET('g', &use_gpu, "run on gpu"),

		OPT_LONG('X', (udims), "guessed from kspace", "Nx of the target image"),
		OPT_LONG('Y', (udims + 1), "guessed from kspace", "Ny of the target image"),
		OPT_LONG('Z', (udims + 2), "guessed from kspace", "Nz of the target image"),

		OPT_FLOAT('l', &modl.lambda_min, "0", "minimal lambda allowed"),

		OPT_STRING('H', (const char**)(&(history_filename)), "", "file for dumping train history"),
	};

	cmdline(&argc, argv, 5, 9, usage_str, help_str, ARRAY_SIZE(opts), opts);

	//we only give K as commandline
	modl.Ky = modl.Kx;

	if (train && apply)
		error("Train and apply would overwrite the reference!\n");


#ifdef USE_CUDA
	if (use_gpu) {

		num_init_gpu();
		cuda_use_global_memory();
	}

	else
#endif
		num_init();

	char* filename_kspace = argv[1];
	char* filename_coil = argv[2];
	char* filename_mask = argv[3];

	char* filename_weights = argv[4];

	char* filename_out = argv[5];

	if (initialize)
		init_nn_modl(&modl);
	else
		nn_modl_load_weights(&modl, filename_weights);

	long kdims[5]; 		//[Nkx, Nky, Nkz, Nc, Nt]
	long cdims[5]; 		//[Nkx, Nky, Nkz, Nc, Nt]
	long mdims[5]; 		//[Nkx, Nky, Nkz, 1,  1 or Nb]

	complex float* file_kspace = load_cfl(filename_kspace, 5, kdims);
	complex float* file_coil = load_cfl(filename_coil, 5, cdims);
	complex float* file_mask = load_cfl(filename_mask, 5, mdims);

	for (int i = 0; i < 5; i++)
		assert(kdims[i] == cdims[i]);
	for (int i = 0; i < 3; i++)
		assert(kdims[i] == mdims[i]);
	assert(1 == mdims[3]);
	assert((1 == mdims[4]) || (kdims[4] == mdims[4]));

	modl.share_mask = (1 == mdims[4]);

	if(use_gpu)
		nn_modl_move_gpucpu(&modl, true);

	if (train) {

		complex float* file_ref = load_cfl(filename_out, 5, udims);

		if (normalize) {

			complex float* scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
			complex float* u0 = md_alloc(5, udims, CFL_SIZE);
			compute_zero_filled(udims, u0, kdims, file_kspace, file_coil, mdims, file_mask);
			compute_scale(udims, scaling, u0);
			md_free(u0);

			normalize_max(udims, scaling, file_ref, file_ref);
			normalize_max(kdims, scaling, file_kspace, file_kspace);

			md_free(scaling);
		}

		train_nn_modl(&modl, CAST_UP(&train_conf), udims, file_ref, kdims, file_kspace, cdims, file_coil, mdims, file_mask, random_order, history_filename, (10 == argc) ? argv + 6: NULL);
		nn_modl_store_weights(&modl, filename_weights);
		unmap_cfl(5, udims, file_ref);
	}


	if (apply) {

		udims[0] = (1 == udims[0]) ? kdims[0] : udims[0];
		udims[1] = (1 == udims[1]) ? kdims[1] : udims[1];
		udims[2] = (1 == udims[2]) ? kdims[2] : udims[2];
		udims[4] = (1 == udims[4]) ? kdims[4] : udims[4];

		complex float* file_out = create_cfl(filename_out, 5, udims);

		complex float* scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
		if (normalize) {

			complex float* u0 = md_alloc(5, udims, CFL_SIZE);
			compute_zero_filled(udims, u0, kdims, file_kspace, file_coil, mdims, file_mask);
			compute_scale(udims, scaling, u0);
			normalize_max(kdims, scaling, file_kspace, file_kspace);
			md_free(u0);
		} else {
			for (int i = 0; i < kdims[4]; i++)
				scaling[i] = 1.;
		}

		apply_nn_modl(&modl, udims, file_out, kdims, file_kspace, cdims, file_coil, mdims, file_mask);
		renormalize_max(udims, scaling, file_out, file_out);

		unmap_cfl(5, udims, file_out);
		md_free(scaling);
	}

	nn_modl_free_weights(&modl);

	unmap_cfl(5, mdims, file_mask);
	unmap_cfl(5, kdims, file_kspace);
	unmap_cfl(5, cdims, file_coil);


	exit(0);
}
