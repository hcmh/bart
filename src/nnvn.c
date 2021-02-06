#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/mri.h"
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
#include "misc/opts_json.h"
#include "misc/mmio.h"

#include "nn/weights.h"

#include "networks/vn.h"
#include "networks/misc.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <weights> <out/ref>";
static const char help_str[] = "Trains and appplies the Variational Network.";

int main_nnvn(int argc, char* argv[])
{
	struct vn_s vn_config = vn_default;
	struct iter6_iPALM_conf train_conf = iter6_iPALM_conf_defaults;

	bool train = false;
	bool apply = false;
	bool use_gpu = false;
	bool initialize = false;

	int random_order = 0;

	char* filename_weights_load = NULL;

	long Nb = 10;

	const char* config_file = NULL;
	bool load_mem = false;

	bool test_defaults = false;

	struct network_data_s data = network_data_empty;
	struct network_data_s valid_data = network_data_empty;


	const struct opt_s opts[] = {

		OPTL_SET('i', "initialize", &initialize, "initialize weights"),
		OPTL_SET('t', "train", &train, "train variational network"),
		OPTL_SET('g', "gpu", &use_gpu, "run on gpu"),
		OPTL_SET('a', "apply", &apply, "apply variational network"),
		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "weights", "load weights for continuing training"),
		OPTL_STRING('c', "config", &config_file, "file", "file for loading varnet configuration"),

		OPTL_STRING(0, "trajectory", &(data.filename_trajectory), "file", "trajectory"),
		OPTL_STRING(0, "pattern", &(data.filename_pattern), "file", "sampling pattern / psf in kspace"),

		OPTL_STRING(0, "valid_trajectory", &(valid_data.filename_trajectory), "file", "validation data trajectory"),
		OPTL_STRING(0, "valid_pattern", &(valid_data.filename_pattern), "file", "validation data sampling pattern / psf in kspace"),
		OPTL_STRING(0, "valid_kspace", &(valid_data.filename_kspace), "file", "validation data kspace"),
		OPTL_STRING(0, "valid_coil", &(valid_data.filename_coil), "file", "validation data sensitivity maps"),
		OPTL_STRING(0, "valid_ref", &(valid_data.filename_out), "file", "validation data reference"),

		OPTL_FLOAT('r', "learning_rate", &(train_conf.INTERFACE.learning_rate), "lr", "learning rate"),
		OPTL_INT('e', "epochs", &(train_conf.INTERFACE.epochs), "epochs", "number epochs to train"),
		OPTL_LONG('b', "batch_size", &(Nb), "Nb", "number epochs to train"),
		OPTL_INT(0, "randomize_batches", &(random_order), "", "0=no shuffle, 1=shuffle batches, 2= shuffle data, 3=randonly draw data"),

		OPTL_STRING(0, "save_train_history", (const char**)(&(train_conf.INTERFACE.history_filename)), "file", "file for dumping train history"),
		OPTL_LONG(0, "save_checkpoints_interval", &(train_conf.INTERFACE.dump_mod), "int", "save weights every int epochs"),
		
		
		OPTL_SET('n', "normalize", &(vn_config.normalize), "normalize the input by maximum of zero-filled reconstruction"),
		OPTL_SET('m', "load_data", &(load_mem), "load files int memory"),
		OPTL_SET(0, "low_mem", &(vn_config.low_mem), "reduce memory usage by checkpointing"),

		OPTL_SET(0, "test_defaults", &test_defaults, "set defaults to small values (used for testing)"),

		OPTL_SET(0, "regrid", &(vn_config.regrid), "grids fully sampled kspace by applying pattern"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (test_defaults) {

		vn_config.Nl = 2;
		vn_config.Nf = 5;
		
		vn_config.Kx = 3;
		vn_config.Ky = 3;

		vn_config.Nw = 5;
	}
	
	if (NULL != config_file) {

		const struct opt_json_s opts_json[] = {

			JSON_BOOL(JSON_LABEL("data", "normalize"), &(vn_config.normalize), false,  ""),
			JSON_BOOL(JSON_LABEL("data", "regrid"), &(vn_config.regrid), false,  ""),

			JSON_BOOL(JSON_LABEL("training", "low_mem"), &(vn_config.low_mem), false,  ""),
			JSON_INT(JSON_LABEL("training", "epochs"), &(train_conf.INTERFACE.epochs), false,  ""),
			JSON_LONG(JSON_LABEL("training", "batch_size"), &(Nb), false,  ""),
			JSON_FLOAT(JSON_LABEL("training", "learning_rate"), &(train_conf.INTERFACE.learning_rate), false,  ""),
			JSON_INT(JSON_LABEL("training", "rand_batch_mode"), &(random_order), false,  ""),
			JSON_STRING(JSON_LABEL("training", "history_file"), &(train_conf.INTERFACE.history_filename), false,  ""),
			JSON_LONG(JSON_LABEL("training", "checkpoint_interval"), &(train_conf.INTERFACE.dump_mod), false,  ""),
			JSON_BOOL(JSON_LABEL("training", "reduce_momentum"), &(train_conf.reduce_momentum), false,  ""),

			JSON_LONG(JSON_LABEL("network", "iterations"), &(vn_config.Nl), true,  ""),
			JSON_LONG(JSON_LABEL("network", "filter"), &(vn_config.Nf), true,  ""),
			JSON_LONG(JSON_LABEL("network", "num_rbf"), &(vn_config.Nw), true,  ""),
			
			JSON_LONG(JSON_LABEL("network", "kernels", "x"), &(vn_config.Kx), true,  ""),
			JSON_LONG(JSON_LABEL("network", "kernels", "y"), &(vn_config.Ky), true,  ""),
			JSON_LONG(JSON_LABEL("network", "kernels", "z"), &(vn_config.Kz), false,  ""),

			JSON_BOOL(JSON_LABEL("network", "shared_weights"), &(vn_config.shared_weights), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "init_lambda"), &(vn_config.lambda_init), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "init_scale_mu"), &(vn_config.init_scale_mu), false,  ""),

			JSON_BOOL(JSON_LABEL("network", "init_tickhonov"), &(vn_config.init_tickhonov), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "init_tickhonov_lambda"), &(vn_config.lambda_fixed_tickhonov), false,  ""),

			JSON_BOOL(JSON_LABEL("train", "monitor_lambda"), &(vn_config.monitor_lambda), false,  ""),

		};
		read_json(config_file, ARRAY_SIZE(opts_json), opts_json);
	}

	train_conf.INTERFACE.batchgen_type = random_order;

	//we only give K as commandline
	train_conf.beta = train_conf.alpha;
	train_conf.trivial_stepsize = true;



	data.filename_kspace = argv[1];
	data.filename_coil = argv[2];
	const char* filename_weights = argv[3];
	data.filename_out = argv[4];

	if (0 < train_conf.INTERFACE.dump_mod)
		train_conf.INTERFACE.dump_filename = filename_weights;

	if (train && apply)
		error("Train (-t) and apply(-a) would overwrite the reference!\n");

	if (!train && !apply) {

		if(initialize) {

			init_vn(&vn_config);
			dump_nn_weights(filename_weights, vn_config.weights);
		} else
			error("Network needs to be either trained (-t) or applied (-a)!\n");
	}

#ifdef USE_CUDA
	if (use_gpu) {

		num_init_gpu();
		cuda_use_global_memory();
	} else
#endif
		num_init();

	
	
	if (apply)
		data.create_out = true;
	load_network_data(&data);
	bool use_valid_data = (NULL != valid_data.filename_coil) && (NULL != valid_data.filename_kspace) && (NULL != valid_data.filename_out);
	network_data_check_simple_dims(&data);


	if (train){

		if (initialize == (NULL != filename_weights_load))
			error("For training, weights must be either initialized(-i) or loaded (-l)!\n");

		if (initialize)
			init_vn(&vn_config);
		else
			load_vn(&vn_config, filename_weights_load, false);

		if (use_gpu)
			move_gpu_nn_weights(vn_config.weights);

		train_vn(&vn_config, CAST_UP(&train_conf), data.idims, data.out, data.kdims, data.kspace, data.cdims, data.coil, data.pdims, data.pattern, Nb, use_valid_data ? &valid_data : NULL);
		dump_nn_weights(filename_weights, vn_config.weights);
	}


	if (apply) {

		load_vn(&vn_config, filename_weights, true);
		if (use_gpu)
			move_gpu_nn_weights(vn_config.weights);
		apply_vn_batchwise(&vn_config, data.idims, data.out, data.kdims, data.kspace, data.cdims, data.coil, data.pdims, data.pattern, Nb);
	}

	free_network_data(&data);
	exit(0);
}
