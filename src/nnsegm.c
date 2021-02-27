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

#include "networks/nn_segm.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<input> <weights> <output>";
static const char help_str[] = "Trains or applies neural network for semantic segmentation of images.";





int main_nnsegm(int argc, char* argv[])
{
	bool train = false;
	bool initialize = false;
	bool predict = false;
	bool accuracy = false;
	bool use_gpu = false;

	struct segm_s segm = segm_default;

	long N_batch = 0;
	long epochs = 1;


	const char* val_img = NULL;
	const char* val_mask = NULL;
	const char* val_loss = NULL;

	const struct opt_s opts[] = {

		OPT_SET('i', &initialize, "initialize weights"),
		OPT_SET('t', &train, "train the network"),
		OPT_LONG('e', &epochs, "epochs", "number of epochs for training"),
		OPT_SET('p', &predict, "print total accuracy"),
		OPT_SET('a', &accuracy, "print accuracy of one batch"),
		OPT_LONG('b', &N_batch, "batches", "batch size for training/prediction"),
		OPT_SET('g', &use_gpu, "run on gpu"),
		OPT_STRING('V', &val_img, "file", "validation images"),
		OPT_STRING('M', &val_mask, "file", "validation masks"),
		OPT_STRING('H', &val_loss, "file", "validation loss"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


	long dims_in[3];
	long dims_out[4];

	complex float* in;
	complex float* out;

	in = load_cfl(argv[1], 3, dims_in); //input, number of dims_input, dims_input
	out = load_cfl(argv[3], 4, dims_out); //output/reference, number of dims_output, dims_output
	segm.imgx = dims_in[0];
	segm.imgy = dims_in[1];
	segm.classes = dims_out[0];
#if 1
	nn_weights_t weights;
	if (initialize)
		weights = init_nn_segm_new(&segm);
	else
		weights = load_nn_weights(argv[2]);
	//weights = load_shared_cfl(argv[2], 1, dims_weights);

#ifdef USE_CUDA
	if (use_gpu) {
		num_init_gpu();
		move_gpu_nn_weights(weights);
	}
	else
#endif
#else
	complex float* weights;

	long dims_weights[] = {nn_segm_get_num_weights(&segm)};
	printf("Dim weights %d \n",nn_segm_get_num_weights(&segm));

	if (initialize){

		printf("Init weights\n");
		weights = create_cfl(argv[2], 1, dims_weights);
		init_nn_segm(weights, &segm);
		unmap_cfl(1, dims_weights, weights);
	}

	weights = load_shared_cfl(argv[2], 1, dims_weights);
#endif
	assert(dims_in[2] == dims_out[3]);

	long dims_in_val[3] = {};
	long dims_out_val[4] = {};

	if ((NULL != val_img && NULL == val_mask) || (NULL != val_mask && NULL == val_img))
		error("Both validation images and validation masks must be given.\n");

	if (NULL != val_img){
		segm.in_val = load_cfl(val_img, 3, dims_in_val);
		segm.out_val = load_cfl(val_mask, 4, dims_out_val);
		assert(dims_in_val[2] == dims_out_val[3]);
	}

	if (NULL == val_loss)
		segm.val_loss = "history";
	else
		segm.val_loss = val_loss;

	// fix wrong or missing batch size
	if (0 == N_batch || N_batch > dims_in[2])
		N_batch = (dims_in[2] >= 5) ? 5 : dims_in[2];

	complex float* in_gpu  = NULL;
	complex float* out_gpu = NULL;
#if 1
	nn_weights_t weights_gpu = NULL;
	if (train){

		printf("Train\n");
		train_nn_segm_new(dims_in[2], N_batch, (NULL == weights_gpu) ? weights : weights_gpu,
				(NULL == in_gpu) ? in : in_gpu, (NULL == out_gpu) ? out : out_gpu, epochs, &segm);
		//if (use_gpu)
		//	md_copy(1, dims_weights, weights, weights_gpu, CFL_SIZE);
		printf("Network trained\n");
		dump_nn_weights(argv[2], weights);
	}
	if (predict){
		// return accuracy for all batches
		printf("Total Accuracy(DC) = %f\n", accuracy_nn_segm_new(dims_in[2], N_batch, (NULL == weights_gpu) ? weights : weights_gpu,
								(NULL == in_gpu) ? in : in_gpu, (NULL == out_gpu) ? out : out_gpu, &segm));
	}

	if (accuracy){
		// return accuracy for first batch
		printf("Accuracy(DC) = %f\n", accuracy_nn_segm_new(N_batch, N_batch, (NULL == weights_gpu) ? weights : weights_gpu,
								(NULL == in_gpu) ? in : in_gpu, (NULL == out_gpu) ? out : out_gpu, &segm));
	}
	nn_weights_free(weights);
#else
	complex float* weights_gpu = NULL;
	if (train){

		printf("Train\n");
		train_nn_segm(dims_in[2], N_batch, dims_in_val[2], (NULL == weights_gpu) ? weights : weights_gpu,
				(NULL == in_gpu) ? in : in_gpu, (NULL == out_gpu) ? out : out_gpu, epochs, &segm);
		if (use_gpu)
			md_copy(1, dims_weights, weights, weights_gpu, CFL_SIZE);
		printf("Network trained\n");
	}
	if (predict){
		// return accuracy for all batches
		printf("Total Accuracy(DC) = %f\n", accuracy_nn_segm(dims_in[2], N_batch, (NULL == weights_gpu) ? weights : weights_gpu,
								(NULL == in_gpu) ? in : in_gpu, (NULL == out_gpu) ? out : out_gpu, &segm));
	}

	if (accuracy){
		// return accuracy for first batch
		printf("Accuracy(DC) = %f\n", accuracy_nn_segm(N_batch, N_batch, (NULL == weights_gpu) ? weights : weights_gpu,
								(NULL == in_gpu) ? in : in_gpu, (NULL == out_gpu) ? out : out_gpu, &segm));
	}
	unmap_cfl(1, dims_weights, weights);
	md_free(weights_gpu);
#endif
	md_free(out_gpu);
	md_free(in_gpu);

	unmap_cfl(4, dims_out, out);
	unmap_cfl(3, dims_in, in);


	exit(0);
}