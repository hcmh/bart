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
#include "misc/misc.h"

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
	bool use_gpu = false;
	bool use_bn = false;

	long N_batch = 0;
	long epochs = 1;



	const struct opt_s opts[] = {

		OPT_SET('i', &initialize, "initialize weights"),
		OPT_SET('t', &train, "train the network"),
		OPT_LONG('e', &epochs, "", "number of epochs for training"),
		OPT_SET('p', &predict, "predict digits"),
		OPT_SET('a', &accuracy, "print accuracy"),
		OPT_SET('n', &use_bn, "use batch normalization"),
		OPT_LONG('b', &N_batch, "", "batch size for training/prediction"),
		OPT_SET('g', &use_gpu, "run on gpu")
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	enum MNIST_NETWORK_TYPE network_type = MNIST_NETWORK_DEFAULT;
	if (use_bn)
		network_type = MNIST_NETWORK_BATCHNORM;


	long dims_weights[] = {nn_mnist_get_num_weights(network_type)};

	complex float* weights;
	complex float* in;
	complex float* out;

	if (initialize){

		printf("Init weights\n");
		weights = create_cfl(argv[2], 1, dims_weights);
		init_nn_mnist(network_type, weights);
		unmap_cfl(1, dims_weights, weights);
	}

	weights = load_shared_cfl(argv[2], 1, dims_weights);
	if (dims_weights[0] != nn_mnist_get_num_weights(network_type))
		error("Dimensions of weights do not fit to the network!\n");

	long dims_in[3];
	long dims_out[2];

	in = load_cfl(argv[1], 3, dims_in);
	out = load_cfl(argv[3], 2, dims_out);

	assert(dims_in[2] == dims_out[1]);

	if (N_batch == 0)
		N_batch = 128;

	if (use_gpu && train){

#ifdef  USE_CUDA

		complex float* in_gpu = md_alloc_gpu(3, dims_in, CFL_SIZE);
		md_copy(3, dims_in, in_gpu, in, CFL_SIZE);

		complex float* weights_gpu = md_alloc_gpu(1, dims_weights, CFL_SIZE);
		md_copy(1, dims_weights, weights_gpu, weights, CFL_SIZE);

		complex float* out_gpu = md_alloc_gpu(2, dims_out, CFL_SIZE);
		md_copy(2, dims_out, out_gpu, out, CFL_SIZE);

		printf("Train\n");
		train_nn_mnist(network_type, N_batch, dims_in[2], weights_gpu, in_gpu, out_gpu, epochs);

		md_copy(1, dims_weights, weights, weights_gpu, CFL_SIZE);

		md_free(in_gpu);
		md_free(out_gpu);
		md_free(weights_gpu);
#else
		error("BART compiled without GPU support.\n");
#endif

	} else {

		if (train){

			printf("Train\n");
			train_nn_mnist(network_type, N_batch, dims_in[2], weights, in, out, epochs);
		}
	}

	long prediction[N_batch];
	if (predict){

		printf("Predict first %ld numbers:\n", N_batch);
		predict_nn_mnist(network_type, N_batch, N_batch, prediction, weights, in);
		print_long(N_batch, prediction);
	}

	if (accuracy)
        	printf("Accuracy = %f\n", accuracy_nn_mnist(network_type, dims_in[2] , N_batch, weights, in, out));


	unmap_cfl(2, dims_out, out);
	unmap_cfl(3, dims_in, in);
	unmap_cfl(1, dims_weights, weights);

	exit(0);
}
