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

#include "nn/weights.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"
#include "misc/misc.h"

#include "networks/mnist.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Trains or applies nn for recognizing handwritten digits.";





int main_mnist(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* weights_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_INFILE(true, &weights_file, "weights"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool train = false;
	bool initialize = false;
	bool predict = false;
	bool accuracy = false;
	bool use_gpu = false;

	long N_batch = 0;
	long epochs = 1;

	const struct opt_s opts[] = {

		OPT_SET('i', &initialize, "initialize weights"),
		OPT_SET('t', &train, "train the network"),
		OPT_LONG('e', &epochs, "", "number of epochs for training"),
		OPT_SET('p', &predict, "predict digits"),
		OPT_SET('a', &accuracy, "print accuracy"),
		OPT_LONG('b', &N_batch, "", "batch size for training/prediction"),
		OPT_SET('g', &use_gpu, "run on gpu")
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	nn_weights_t weights;

	if (initialize)
		weights = init_nn_mnist();
	else
		weights = load_nn_weights(weights_file);

#ifdef USE_CUDA
	if (use_gpu) {

		num_init_gpu();
		move_gpu_nn_weights(weights);
	}
	else
#endif
	num_init();

	long dims_in[3];
	long dims_out[2];
	complex float* in = load_cfl(in_file, 3, dims_in);
	complex float* out = load_cfl(out_file, 2, dims_out);
	assert(dims_in[2] == dims_out[1]);

	if (N_batch == 0)
		N_batch = MIN(128, dims_in[2]);


	if (train) {

		printf("Train\n");
		train_nn_mnist(N_batch, dims_in[2], weights, in, out, epochs);
		dump_nn_weights(weights_file, weights);
	}

	if (predict){

		complex float prediction[N_batch];

		printf("Predict first %ld numbers:\n", N_batch);
		predict_nn_mnist(N_batch, N_batch, prediction, weights, in);
		print_complex(N_batch, prediction);
	}

	if (accuracy)
		printf("Accuracy = %f\n", accuracy_nn_mnist(dims_in[2], N_batch, weights, in, out));


	nn_weights_free(weights);

	unmap_cfl(2, dims_out, out);
	unmap_cfl(3, dims_in, in);

	exit(0);
}
