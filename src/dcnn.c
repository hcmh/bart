/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"

#include "networks/dcnn.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Applies a pre-trained convolutional neural network.";





int main_dcnn(int argc, char* argv[argc])
{

	const char* in_file = NULL;
	const char* kernel_file = NULL;
	const char* bias_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_INFILE(true, &kernel_file, "kernel"),
		ARG_INFILE(true, &bias_file, "bias"),
		ARG_OUTFILE(true, &out_file, "output"),
	};


	bool subinp = false;
	bool use_gpu = false;

	const struct opt_s opts[] = {

		OPT_SET('r', &subinp, "subtract output from input"),
		OPT_SET('g', &use_gpu, "run on gpu"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	unsigned int N = DIMS;
	long dims[N];
	const complex float* in = load_cfl(in_file, N, dims);

	long krn_dims[N];
	const complex float* krn = load_cfl(kernel_file, N, krn_dims);

	long bias_dims[N];
	const complex float* bias = load_cfl(bias_file, N, bias_dims);

	complex float* out = create_cfl(out_file, N, dims);

	if (use_gpu){

		num_init_gpu_device(1);

#ifdef  USE_CUDA

		complex float* bias_gpu = md_alloc_gpu(N, bias_dims, CFL_SIZE);
		complex float* krn_gpu = md_alloc_gpu(N, krn_dims, CFL_SIZE);
		complex float* in_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
		complex float* out_gpu = md_alloc_gpu(N, dims, CFL_SIZE);

		md_copy(N, bias_dims, bias_gpu, bias, CFL_SIZE);
		md_copy(N, krn_dims, krn_gpu, krn, CFL_SIZE);
		md_copy(N, dims, in_gpu, in, CFL_SIZE);

		simple_dcnn(dims, krn_dims, krn_gpu, bias_dims, bias_gpu, out_gpu, in_gpu);

		md_copy(3, dims, out, out_gpu, CFL_SIZE);

		md_free(in_gpu);
		md_free(out_gpu);
		md_free(bias_gpu);
		md_free(krn_gpu);

#endif

	} else {

		simple_dcnn(dims, krn_dims, krn, bias_dims, bias, out, in);
	}

	if (subinp)
		md_zsub(N, dims, out, in, out);

	unmap_cfl(N, dims, out);
	unmap_cfl(N, krn_dims, krn);
	unmap_cfl(N, bias_dims, bias);
	unmap_cfl(N, dims, in);
	exit(0);
}
