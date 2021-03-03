#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

#include "iter/iter6.h"

#include "nn/weights.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"
#include "misc/misc.h"

#include "networks/nnet.h"
#include "networks/unet.h"
#include "networks/losses.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<input> <weights> <output>";
static const char help_str[] = "Trains or applies a neural network.";



int main_nnet(int argc, char* argv[])
{
	bool apply = false;
	bool train = false;
	bool eval = false;

	long N_batch = 0;

	const char* graph_filename = NULL;
	const char* filename_weights_load = NULL;

	int NI = -1;

	bool mnist_default = false;
	long N_segm_labels = -1;

	struct nnet_s config = nnet_init;

	struct opt_s network_opts[] = {

		OPTL_SET('M', "mnist", &(mnist_default), "use basic MNIST Network"),
		OPTL_LONG('U', "unet-segm", &(N_segm_labels), "labels", "use U-Net for segmentation"),
	};

	const struct opt_s opts[] = {

		OPTL_SET('a', "apply", &apply, "apply nnet"),
		OPTL_SET( 'e', "eval", &eval, "evaluate nnet"),
		OPTL_SET('t', "train", &train, "trains network"),

		OPTL_SET('g', "gpu", &(config.gpu), "run on gpu"),

		OPTL_LONG('b', "batch-size", &(N_batch), "batchsize", "size of mini batches"),

		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "weights", "load weights for continuing training"),

		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),
		OPTL_SUBOPT('U', "config-unet-segm", "...", "configure U-Net for segmentation", N_unet_segm_opts, unet_segm_opts),

		OPTL_SUBOPT('L', "loss", "...", "configure the training loss", N_loss_opts, loss_opts),
		OPTL_SUBOPT(0, "validation-loss", "...", "configure the validation loss", N_val_loss_opts, val_loss_opts),

		OPTL_SUBOPT('T', "train-config", "...", "configure general training parmeters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure Adam", N_iter6_adam_opts, iter6_adam_opts),
		//OPTL_SUBOPT(0, "iPALM", "...", "configure iPALM", N_iter6_ipalm_opts, iter6_ipalm_opts),

		OPTL_STRING(0, "export-graph", (const char**)(&(graph_filename)), "file.dot", "file for dumping graph"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	const char* filename_in = argv[1];
	const char* filename_weights = argv[2];
	const char* filename_out = argv[3];

	config.train_conf = iter6_get_conf_from_opts();

	if (mnist_default)
		nnet_init_mnist_default(&config);

	if (-1 != N_segm_labels) {

		nnet_init_unet_segm_default(&config, N_segm_labels);

		if (-1 == NI)
			NI = 5;
	}

	if (train) {

		if (NULL == config.train_conf) {

			debug_printf(DP_WARN, "No training algorithm selected. Fallback to Adam!");
			config.train_conf = CAST_UP(&iter6_adam_conf_opts);
		}

		iter6_copy_config_from_opts(config.train_conf);
	}

	if (NULL == config.network)
		error("No network selected!");

	if ((0 < config.train_conf->dump_mod) && (NULL == config.train_conf->dump_filename))
		config.train_conf->dump_filename = filename_weights;


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
	complex float* in = load_cfl(filename_in, (-1 == NI) ? (int)DIMS : NI, dims_in);

	if (-1 == NI) {

		NI = DIMS;
		while ((NI > 0) && (1 == dims_in[NI - 1]))
			NI--;
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
