#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

#include "iter/iter6.h"

#include "nn/weights.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"
#include "misc/misc.h"

#include "networks/nnet.h"
#include "networks/mnist.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<input> <weights> <output>";
static const char help_str[] = "Trains or applies Neural Network";



int main_nnet(int argc, char* argv[])
{
	bool apply = false;
	bool train = false;
	bool eval = false;

	long N_batch = 0;

	const char* graph_filename = NULL;

	const char* filename_weights_load = NULL;
	const char* filename_train_history = NULL;
	long dump_mod = 0;
	long epochs = 1;
	int random_order = 0;
	float learning_rate = 0;

	int NI = -1;
	bool single_batch = false;

	bool mnist_default = false;

	struct nnet_s config = nnet_init;

	const struct opt_s opts[] = {

		OPTL_SET('a', "apply", &apply, "apply nnet"),
		OPTL_SET( 0, "eval", &eval, "evaluate nnet"),

		OPTL_SET('t', "train", &train, "trains network"),
		OPTL_LONG('e', "epochs", &(epochs), "epochs", "number epochs to train"),
		OPTL_LONG('b', "batch_size", &(N_batch), "batchsize", "size of mini batches"),

		OPTL_FLOAT('r', "learning_rate", &(learning_rate), "lr", "learning rate"),

		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "weights", "load weights for continuing training"),
		OPTL_STRING(0, "save_train_history", (const char**)(&(filename_train_history)), "file", "file for dumping train history"),
		OPTL_LONG(0, "save_checkpoints_interval", &(dump_mod), "int", "save weights every int epochs"),
		OPTL_INT(0, "randomize_batches", &(random_order), "", "0=no shuffle, 1=shuffle batches, 2=shuffle data, 3=randomly draw data"),

		OPTL_SET('g', "gpu", &(config.gpu), "run on gpu"),

		OPTL_STRING(0, "export_graph", (const char**)(&(graph_filename)), "file.dot", "file for dumping graph"),

		OPTL_SET(0, "mnist_default", &(mnist_default), "use basic MNIST Network"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	const char* filename_in = argv[1];
	const char* filename_weights = argv[2];
	const char* filename_out = argv[3];

	if (mnist_default)
		nnet_init_mnist_default(&config);

	config.train_conf->epochs = epochs;
	config.train_conf->history_filename = filename_train_history;
	if (0 < dump_mod) {

		config.train_conf->dump_mod = dump_mod;
		config.train_conf->dump_filename = filename_weights;
	}
	config.train_conf->batchgen_type = random_order;
	if (0 != learning_rate)
		config.train_conf->learning_rate = learning_rate;


#ifdef USE_CUDA
	if (config.gpu) {

		num_init_gpu();
		cuda_use_global_memory();
	}

	else
#endif
		num_init();


	if (apply && (train || eval))
		error("Application would overwrite training data! Either train or apply!");

	if (NULL != filename_weights_load) {

		if (apply)
			error("Weights should only be loaded for trining using -l option!");

		config.weights = load_nn_weights(filename_weights_load);
	}

	config.graph_file = graph_filename;


	long dims_in[DIMS];
	complex float* in = load_cfl(filename_in, (-1 == NI) ? DIMS : NI, dims_in);

	if (-1 == NI) {

		NI = DIMS;
		while ((NI > 0) && (1 == dims_in[NI - 1]))
			NI--;
		if (single_batch)
			NI++;
	}


	if (N_batch == 0)
		N_batch = MIN(128, dims_in[NI - 1]);


	if (train){

		long NO = config.get_no_odims(&config, NI, dims_in);
		long dims_out[NO];
		complex float* out = load_cfl(filename_out, NO, dims_out);

		train_nnet(&config, NO, dims_out, out, NI, dims_in, in,  N_batch);

		dump_nn_weights(filename_weights, config.weights);

		unmap_cfl(NO, dims_out, out);
	}

	if (eval){

		long NO = config.get_no_odims(&config, NI, dims_in);
		long dims_out[NO];
		complex float* out = load_cfl(filename_out, NO, dims_out);

		if (NULL == config.weights)
			config.weights = load_nn_weights(filename_weights);

		eval_nnet(&config, NO, dims_out, out, NI, dims_in, in,  N_batch);

		unmap_cfl(NO, dims_out, out);
	}

	if (apply){

		long NO = config.get_no_odims(&config, NI, dims_in);
		long dims_out[NO];
		config.get_odims(&config, NO, dims_out, NO, dims_in);
		complex float* out = create_cfl(filename_out, NO, dims_out);

		config.weights = load_nn_weights(filename_weights);

		apply_nnet_batchwise(&config, NO, dims_out, out, NI, dims_in, in,  N_batch);

		unmap_cfl(NO, dims_out, out);
	}

	if (NULL != config.weights)
		nn_weights_free(config.weights);

	unmap_cfl(NI, dims_in, in);

	exit(0);
}
