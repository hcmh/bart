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

#include "nn/nn_segm.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef IMG_DIM
#define IMG_DIM 256 	//dimension of square input images
#endif

#ifndef MASK_DIM
#define MASK_DIM 4	//number of dimensions of segmentation masks
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<input> <weights> <output>";
static const char help_str[] = "Trains or applies neural network for semantic segmentation of images.";





int main_nn_segm(int argc, char* argv[])
{
	bool train = false;
	bool initialize = false;
	bool predict = false;
	bool accuracy = false;
	bool use_gpu = false;

	long N_batch = 0;
	long epochs = 1;

	const char* val_img = NULL;
	const char* val_mask = NULL;

	const struct opt_s opts[] = {

		OPT_SET('i', &initialize, "initialize weights"),
		OPT_SET('t', &train, "train the network"),
		OPT_LONG('e', &epochs, "epochs", "number of epochs for training"),
		OPT_SET('p', &predict, "print total accuracy"),
		OPT_SET('a', &accuracy, "print accuracy of one batch"),
		OPT_LONG('b', &N_batch, "batches", "batch size for training/prediction"),
		OPT_SET('g', &use_gpu, "run on gpu"),
		OPT_STRING('V', &val_img, "file", "validation images"),
		OPT_STRING('M', &val_mask, "file", "validation masks")
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	long dims_weights[] = {nn_segm_get_num_weights()};
	//printf("Dim weights %d \n",nn_segm_get_num_weights());
	complex float* weights;
	complex float* in;
	complex float* out;

	if (initialize){

		printf("Init weights\n");
		weights = create_cfl(argv[2], 1, dims_weights);
		init_nn_segm(weights);
		unmap_cfl(1, dims_weights, weights);
	}

	weights = load_shared_cfl(argv[2], 1, dims_weights);

	long dims_in[3]; 
	long dims_out[4];

	in = load_cfl(argv[1], 3, dims_in); //input, number of dims_input, dims_input
	out = load_cfl(argv[3], 4, dims_out); //output/reference, number of dims_output, dims_output

	//debug_print_dims(DP_INFO, 3, dims_in);
	//debug_print_dims(DP_INFO, 4, dims_out);	
	assert(dims_in[2] == dims_out[3]);

	long dims_in_val[3] = {};
	long dims_out_val[4] = {};

	complex float* in_val = NULL;
	complex float* out_val = NULL;

	if ((NULL != val_img && NULL == val_mask) || (NULL != val_mask && NULL == val_img))
		error("Both validation images and validation masks must be given.\n");

	if (NULL != val_img){
		in_val = load_cfl(val_img, 3, dims_in_val);
		out_val  = load_cfl(val_mask, 4, dims_out_val);
		assert(dims_in_val[2] == dims_out_val[3]);
	}

	// fix wrong or missing batch size
	if (0 == N_batch || N_batch > dims_in[2])
		N_batch = (dims_in[2] >= 5) ? 5 : dims_in[2];

	if (use_gpu){

		num_init_gpu_device(1);


#if 0

		complex float* in_gpu = md_alloc_gpu(3, dims_in, CFL_SIZE);
		md_copy(3, dims_in, in_gpu, in, CFL_SIZE);

		complex float* weights_gpu = md_alloc_gpu(1, dims_weights, CFL_SIZE);
		md_copy(1, dims_weights, weights_gpu, weights, CFL_SIZE);

		complex float* out_gpu = md_alloc_gpu(2, dims_out, CFL_SIZE);
		md_copy(2, dims_out, out_gpu, out, CFL_SIZE);

		if (train){

			printf("Train\n");
			train_nn_segm(N_batch, dims_in[2], weights_gpu, in_gpu, out_gpu, epochs);
		}

		long prediction[]= {IMG_DIM, IMG_DIM, N_batch}; // x, y, mask dimension, batch size
		if (predict){

			printf("Predict first %ld numbers:\n", N_batch);
			predict_nn_segm(N_batch, prediction, weights_gpu, in_gpu);
			print_long(N_batch, prediction);
		}

		if (accuracy)
			printf("Accuracy = %f\n", accuracy_nn_segm(N_batch, weights_gpu, in_gpu, out_gpu));

		md_copy(3, dims_in, in, in_gpu, CFL_SIZE);
		md_copy(1, dims_weights, weights, weights_gpu, CFL_SIZE);
		md_copy(2, dims_out, out, out_gpu, CFL_SIZE);

#endif

	} else {

		if (train){

			printf("Train\n");
			train_nn_segm(N_batch, dims_in[2], dims_in_val[2], weights, in, out, in_val, out_val, epochs);
			printf("Network trained\n");
		}
		if (predict){
			printf("Total Accuracy(DC) = %f\n", accuracy_nn_segm(dims_in[2], N_batch, weights, in, out));
		}

		if (accuracy)
			printf("Accuracy(DC) = %f\n", accuracy_nn_segm(N_batch, N_batch, weights, in, out));
	}

	unmap_cfl(4, dims_out, out);
	unmap_cfl(3, dims_in, in);
	unmap_cfl(1, dims_weights, weights);

	exit(0);
}
