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

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"

#include "nn/mnist.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<input> <weights> <output>";
static const char help_str[] = "Trains or applies nn for recognizing handwritten digits.";





int main_mnist(int argc, char* argv[])
{
	bool train = false;
	bool initialize = false;
	bool predict = false;
	bool accuracy = false;

	long N_batch = 0;
	long epochs = 1;

	const struct opt_s opts[] = {

		OPT_SET('i', &initialize, "initialize weights"),
		OPT_SET('t', &train, "train the network"),
		OPT_LONG('e', &epochs, "", "number of epochs for training"),
		OPT_SET('p', &predict, "predict digits"),
		OPT_SET('a', &accuracy, "print accuracy"),
		OPT_LONG('b', &N_batch, "", "batch size for training/prediction")
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	long dims_weights[] = {nn_mnist_get_num_weights()};

	complex float* weights;
	complex float* in;
	complex float* out;

	if (initialize){

		printf("Init weights\n");
		weights = create_cfl(argv[2], 1, dims_weights);
		init_nn_mnist(weights);
		unmap_cfl(1, dims_weights, weights);
	}

	weights = load_shared_cfl(argv[2], 1, dims_weights);

	long dims_in[3];
	long dims_out[2];

	in = load_cfl(argv[1], 3, dims_in);
	out = load_cfl(argv[3], 2, dims_out);

	assert(dims_in[2] == dims_out[1]);

	if (N_batch == 0)
		N_batch = 128;

	if (train){

		printf("Train\n");
		train_nn_mnist(N_batch, dims_in[2], weights, in, out, epochs);
	}

	long prediction[N_batch];
	if (predict){

		printf("Predict first %ld numbers:\n", N_batch);
		predict_nn_mnist(N_batch, prediction, weights, in);
		print_long(N_batch, prediction);
	}

	if (accuracy)
        	printf("Accuracy = %f\n", accuracy_nn_mnist(N_batch, weights, in, out));

	unmap_cfl(2, dims_out, out);
	unmap_cfl(3, dims_in, in);
	unmap_cfl(1, dims_weights, weights);

	exit(0);
}

