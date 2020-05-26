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

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <mask> <weights conv> <weights rbf> <weights lambda> <out> <out ref> <out zerofilled>";
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

		OPT_LONG('X', &(vn_config.Ux), "guessed from kspace", "Nx of the target image"),
		OPT_LONG('Y', &(vn_config.Uy), "guessed from kspace", "Ny of the target image"),
		OPT_LONG('Z', &(vn_config.Uz), "guessed from kspace", "Nz of the target image"),
		OPT_SET('p', &oversampling_phase, "guess phase encoding oversampling, i.e. set mask to one where kspace = 0"),	

		OPT_SET('n', &normalize, "normalize "),
		OPT_SET('g', &use_gpu, "run on gpu"),
		
		OPT_STRING('H', (const char**)(&(train_conf.save_path)), "", "folder for training history"),
	};
	
	cmdline(&argc, argv, 9, 9, usage_str, help_str, ARRAY_SIZE(opts), opts);

	//we only give K as commandline
	vn_config.Ky = vn_config.Kx;
	train_conf.beta = train_conf.alpha;
	train_conf.trivial_stepsize = true;


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

	char* filename_ref = argv[8];
	char* filename_u0 = argv[9];


	if (initialize) {

		long dims_conv_w[5] = {vn_config.Nf, vn_config.Kx, vn_config.Ky, vn_config.Kz, vn_config.Nl};
		long dims_rbf_w[3] = {vn_config.Nf, vn_config.Nw, vn_config.Nl};
		long dims_lambda[2] = {1, vn_config.Nl};

		vn_config.conv_w = create_cfl(filename_conv_w, 5, dims_conv_w);
		vn_config.rbf_w = create_cfl(filename_rbf_w, 3, dims_rbf_w);
		vn_config.lambda_w = create_cfl(filename_lambda, 2, dims_lambda);

		initialize_varnet(&vn_config);		
	} else {

		long dims_conv_w[5];
		long dims_rbf_w[3];
		long dims_lambda_w[2];

		vn_config.conv_w = load_shared_cfl(filename_conv_w, 5, dims_conv_w);
		vn_config.rbf_w = load_shared_cfl(filename_rbf_w, 3, dims_rbf_w);
		vn_config.lambda_w = load_shared_cfl(filename_lambda, 2, dims_lambda_w);

		vn_config.Nf = dims_conv_w[0];
		vn_config.Kx = dims_conv_w[1];
		vn_config.Ky = dims_conv_w[2];
		vn_config.Kz = dims_conv_w[3];
		vn_config.Nl = dims_conv_w[4];

		vn_config.Nw = dims_rbf_w[1];

		assert((vn_config.Nf == dims_rbf_w[0]) && (vn_config.Nl == dims_rbf_w[2]));
		assert((1 == dims_lambda_w[0]) && (vn_config.Nl == dims_lambda_w[1]));
	}

	

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

	if (0 == vn_config.Ux)
		vn_config.Ux = kdims[0];
	if (0 == vn_config.Uy)
		vn_config.Uy = kdims[1];
	if (0 == vn_config.Uz)
		vn_config.Uz = kdims[2];

	long udims[5] = {vn_config.Ux, vn_config.Uy, vn_config.Uz, 1, kdims[4]};
	if (oversampling_phase)
		unmask_zeros(mdims, file_mask, kdims, file_kspace);

	// compute zerofilled solution for scaling
	complex float* file_u0 = create_cfl(filename_u0, 5, udims);
	compute_zero_filled(udims, file_u0, kdims, file_kspace, file_coil, mdims, file_mask);
	
	//scale data
	complex float* scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
	if (normalize) {

		compute_scale(udims, scaling, file_u0);
		normalize_max(kdims, scaling, file_kspace, file_kspace);
	} else {
		for (int i = 0; i < kdims[4]; i++)
			scaling[i] = 1.;
	}

	unmap_cfl(5, udims, file_u0);

	

	//Compute reference
	complex float* file_ref = create_cfl(filename_ref, 5, udims);
	compute_reference(udims, file_ref, kdims, file_kspace, file_coil);

	if(use_gpu)
		vn_move_gpu(&vn_config);
	
	if (train){	

		debug_printf(DP_INFO, "Train Variational Network with\n[Nkx, Nky, Nkz, Nc, Nt] = ");
		debug_print_dims(DP_INFO, 5, kdims);
		debug_printf(DP_INFO, "[Ux, Uy, Uz, 1,  Nt] = ");
		debug_print_dims(DP_INFO, 5, udims);
		train_nn_varnet(&vn_config, CAST_UP(&train_conf), udims, file_ref, kdims, file_kspace, file_coil, mdims, file_mask, Nb, random_order);
	}
	
	renormalize_max(udims, scaling, file_ref, file_ref);
	unmap_cfl(5, udims, file_ref);

	if (apply) {

		complex float* file_out = create_cfl(filename_out, 5, udims);

		debug_printf(DP_INFO, "Run Variational Network with\n[Nx, Ny, Nz, Nc, Nt] = ");
		debug_print_dims(DP_INFO, 5, kdims);
		debug_printf(DP_INFO, "[Ux, Uy, Uz, 1, Nt] = ");
		debug_print_dims(DP_INFO, 5, udims);
	
		apply_variational_network_batchwise(&vn_config, udims, file_out, kdims, file_kspace, file_coil, mdims, file_mask, Nb);
		renormalize_max(udims, scaling, file_out, file_out);

		unmap_cfl(5, udims, file_out);
	}

	if(use_gpu)
		vn_move_cpu(&vn_config);

	unmap_cfl(5, MAKE_ARRAY(vn_config.Nf, vn_config.Kx, vn_config.Ky, vn_config.Kz, vn_config.Nl), vn_config.conv_w);
	unmap_cfl(3, MAKE_ARRAY(vn_config.Nf, vn_config.Nw, vn_config.Nl), vn_config.rbf_w);
	unmap_cfl(2, MAKE_ARRAY(1l, vn_config.Nl), vn_config.lambda_w);

	md_free(scaling);

	unmap_cfl(5, mdims, file_mask);
	

	exit(0);
}
