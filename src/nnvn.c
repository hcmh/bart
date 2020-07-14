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

#include "misc/misc.h"
#include "misc/types.h"
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

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <mask> <weights conv> <weights rbf> <weights lambda> <out/ref> <kspace_valid> <sens_valid> <mask_valid> <ref_valid>";
static const char help_str[] = "Trains and appplies the Variational Network.";

int main_nnvn(int argc, char* argv[])
{
	struct vn_s vn_config = vn_default;
	struct iter6_iPALM_conf train_conf = iter6_iPALM_conf_defaults;

	bool train = false;
	bool apply = false;
	bool use_gpu = false;
	bool initialize = false;
	bool normalize = false;
	bool oversampling_phase = false;
	bool random_order = false;

	char* history_filename = NULL;

	long udims[5] = {1, 1, 1, 1, 1};

	long Nb = 10;

	const struct opt_s opts[] = {

		OPT_SET('i', &initialize, "initialize weights"),
		OPT_LONG('L', &(vn_config.Nl), "guessed", "number of layers"),
		OPT_LONG('F', &(vn_config.Nf), "guessed", "number of convolution filters"),
		OPT_LONG('K', &(vn_config.Kx), "guessed", "filtersize"),
		OPT_LONG('W', &(vn_config.Nw), "guessed", "number of activation filters"),

		OPT_SET('t', &train, "train variational network"),
		OPT_INT('E', &(train_conf.epochs), "1000", "number of training epochs"),
		OPT_INT('e', &(train_conf.epoch_start), "0", "load state after given epoch and start training here(only for training)"),
		OPT_INT('S', &(train_conf.save_modulo), "100", "number of training epochs"),
		OPT_LONG('B', &Nb, "10", "batch size"),
		OPT_FLOAT('M', &(train_conf.alpha), "-1.", "fixed momentum for optimization, -1. corresponds to dynamic momentum"),
		OPT_FLOAT('R', &(train_conf.Lshrink), "2.", "reduction of Lipschitz constant for backtracking"),
		OPT_FLOAT('I', &(train_conf.Lincrease), "9.", "increase of Lipschitz constant for backtracking"),
		OPT_FLOAT('l', &(train_conf.L), "1000.", "initial value for Lipschitz constant"),
		OPT_SET('r', &(random_order), "randomize batches"),

		OPT_SET('a', &apply, "apply variational network"),

		OPT_LONG('X', (udims), "guessed from kspace", "Nx of the target image"),
		OPT_LONG('Y', (udims + 1), "guessed from kspace", "Ny of the target image"),
		OPT_LONG('Z', (udims + 2), "guessed from kspace", "Nz of the target image"),
		OPT_SET('p', &oversampling_phase, "guess phase encoding oversampling, i.e. set mask to one where kspace = 0"),

		OPT_SET('n', &normalize, "normalize"),
		OPT_SET('g', &use_gpu, "run on gpu"),

		OPT_STRING('H', (const char**)(&(history_filename)), "", "file for dumping train history"),
	};

	cmdline(&argc, argv, 7, 11, usage_str, help_str, ARRAY_SIZE(opts), opts);
	if ((8 != argc) && (12 != argc))
		error("wrong number of arguments\n");

	//we only give K as commandline
	vn_config.Ky = vn_config.Kx;
	train_conf.beta = train_conf.alpha;
	train_conf.trivial_stepsize = true;

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

	char* filename_conv_w = argv[4];
	char* filename_rbf_w = argv[5];
	char* filename_lambda = argv[6];

	char* filename_out = argv[7];


	if (initialize) {

		initialize_varnet(&vn_config);
		save_varnet(&vn_config, filename_conv_w , filename_rbf_w, filename_lambda);
	} else {

		long dims_conv_w[5];
		long dims_rbf_w[3];
		long dims_lambda_w[2];

		complex float* tmp_conv_w = load_cfl(filename_conv_w, 5, dims_conv_w);
		complex float* tmp_rbf_w = load_cfl(filename_rbf_w, 3, dims_rbf_w);
		complex float* tmp_lambda_w = load_cfl(filename_lambda, 2, dims_lambda_w);

		vn_config.conv_w = md_alloc(5, dims_conv_w, CFL_SIZE);
		vn_config.rbf_w = md_alloc(3, dims_rbf_w, CFL_SIZE);
		vn_config.lambda_w = md_alloc(2, dims_lambda_w, CFL_SIZE);

		md_copy(5, dims_conv_w, vn_config.conv_w, tmp_conv_w, CFL_SIZE);
		md_copy(3, dims_rbf_w, vn_config.rbf_w, tmp_rbf_w, CFL_SIZE);
		md_copy(2, dims_lambda_w, vn_config.lambda_w, tmp_lambda_w, CFL_SIZE);

		unmap_cfl(5, dims_conv_w, tmp_conv_w);
		unmap_cfl(3, dims_rbf_w, tmp_rbf_w);
		unmap_cfl(2, dims_lambda_w, tmp_lambda_w);

		vn_config.Nf = dims_conv_w[0];
		vn_config.Kx = dims_conv_w[1];
		vn_config.Ky = dims_conv_w[2];
		vn_config.Kz = dims_conv_w[3];
		vn_config.Nl = dims_conv_w[4];
		vn_config.Nw = dims_rbf_w[1];

		assert((vn_config.Nf == dims_rbf_w[0]) && (vn_config.Nl == dims_rbf_w[2]));
		assert((1 == dims_lambda_w[0]) && (vn_config.Nl == dims_lambda_w[1]));
	}

	vn_move_gpucpu(&vn_config, use_gpu);

	long kdims[5]; 		//[Nkx, Nky, Nkz, Nc, Nt]
	long dims_coil[5]; 	//[Nkx, Nky, Nkz, Nc, Nt]
	long mdims[5]; 		//[Nkx, Nky, Nkz, 1,  1 or Nb]

	complex float* file_kspace = load_cfl(filename_kspace, 5, kdims);
	complex float* file_coil = load_cfl(filename_coil, 5, dims_coil);
	complex float* file_mask = load_cfl(filename_mask, 5, mdims);

	for (int i = 0; i < 5; i++)
		assert(kdims[i] == dims_coil[i]);
	for (int i = 0; i < 3; i++)
		assert(kdims[i] == mdims[i]);
	assert(1 == mdims[3]);
	assert((1 == mdims[4]) || (kdims[4] == mdims[4]));

	//sets mask to one where k-space is zero, only works for fully-sampled kspace
	//where zeros correspond to phase oversampling
	if (oversampling_phase)
		unmask_zeros(mdims, file_mask, kdims, file_kspace);


	if (train){

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

		debug_printf(DP_INFO, "Train Variational Network with\n[Nkx, Nky, Nkz, Nc, Nt] = ");
		debug_print_dims(DP_INFO, 5, kdims);
		debug_printf(DP_INFO, "[Ux, Uy, Uz, 1,  Nt] = ");
		debug_print_dims(DP_INFO, 5, udims);

		train_nn_varnet(&vn_config, CAST_UP(&train_conf), udims, file_ref, kdims, file_kspace, file_coil, mdims, file_mask, Nb, random_order, history_filename, (12 == argc) ? argv + 8: NULL);
		unmap_cfl(5, udims, file_ref);
		save_varnet(&vn_config, filename_conv_w , filename_rbf_w, filename_lambda);
	}



	if (apply) {

		udims[0] = (1 == udims[0]) ? kdims[0] : udims[0];
		udims[1] = (1 == udims[1]) ? kdims[1] : udims[1];
		udims[2] = (1 == udims[2]) ? kdims[2] : udims[2];
		udims[4] = (1 == udims[4]) ? kdims[4] : udims[4];

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

		complex float* file_out = create_cfl(filename_out, 5, udims);

		debug_printf(DP_INFO, "Run Variational Network with (Nb = %d)\n[Nx, Ny, Nz, Nc, Nt] = ", Nb);
		debug_print_dims(DP_INFO, 5, kdims);
		debug_printf(DP_INFO, "[Ux, Uy, Uz, 1, Nt] = ");
		debug_print_dims(DP_INFO, 5, udims);

		apply_variational_network_batchwise(&vn_config, udims, file_out, kdims, file_kspace, file_coil, mdims, file_mask, Nb);
		renormalize_max(udims, scaling, file_out, file_out);

		md_free(scaling);

		unmap_cfl(5, udims, file_out);
	}

	unmap_cfl(5, mdims, file_mask);
	unmap_cfl(5, kdims, file_kspace);
	unmap_cfl(5, kdims, file_coil);


	exit(0);
}
