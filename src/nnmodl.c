#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <libgen.h>
#include <string.h>

#include "noncart/nufft.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/mem.h"
#include "num/fft.h"
#include "iter/iter6.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/opts_json.h"
#include "misc/mmio.h"

#include "networks/modl.h"
#include "networks/misc.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <weights> <out/ref>";
static const char help_str[] = "Trains or appplies MoDL.";

int main_nnmodl(int argc, char* argv[])
{
	struct modl_s modl = modl_default;
	struct iter_conjgrad_conf def_conf = iter_conjgrad_defaults;
	def_conf.l2lambda = 1.;
	def_conf.maxiter = 50;

	modl.normal_inversion_iter_conf = CAST_UP(&def_conf);

	struct iter6_adam_conf train_conf = iter6_adam_conf_defaults;


	bool train = false;
	bool apply = false;
	bool use_gpu = false;
	bool initialize = false;

	int random_order = 0;

	char* filename_weights_load = NULL;

	long Nb = 10;
	bool one_iter = false;

	const char* config_file = NULL;
	bool load_mem = false;

	bool test_defaults = false;

	struct network_data_s data = network_data_empty;
	struct network_data_s valid_data = network_data_empty;

	const struct opt_s opts[] = {

		OPTL_SET('i', "initialize", &initialize, "initialize weights"),
		OPTL_SET('t', "train", &train, "train modl"),
		OPTL_SET('g', "gpu", &use_gpu, "run on gpu"),
		OPTL_SET('a', "apply", &apply, "apply modl"),
		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "weights", "load weights for continuing training"),
		OPTL_STRING('c', "config", &config_file, "file", "file for loading modl configuration"),
		OPTL_SET('o', "one_iter", &one_iter, "only one MoDL iteration (initialize weights)"),

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
		
		OPTL_SET('n', "normalize", &(modl.normalize), "normalize the input by maximum of zero-filled reconstruction"),
		OPTL_SET('m', "load_data", &(load_mem), "load files int memory"),
		OPTL_SET(0, "low_mem", &(modl.low_mem), "reduce memory usage by checkpointing"),

		OPTL_SET(0, "test_defaults", &test_defaults, "set defaults to small values (used for testing)"),

		OPTL_SET(0, "regrid", &(modl.regrid), "grids fully sampled kspace by applying pattern"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (test_defaults) {

		modl.Nf = 8;
		modl.Nl = 3;
		modl.Nt = 2;

		def_conf.maxiter = 10;
	}

	bool mandatory = !test_defaults;

	int epochs1 = 0;
	int epochs2 = 0;

	const char* history1 = NULL;
	const char* history2 = NULL;

	if (NULL != config_file) {

		const struct opt_json_s opts_json[] = {

			JSON_BOOL(JSON_LABEL("data", "normalize"), &(modl.normalize), false,  ""),
			JSON_BOOL(JSON_LABEL("data", "regrid"), &(modl.regrid), false,  ""),

			JSON_LONG(JSON_LABEL("network", "modl", "iterations"), &(modl.Nt), mandatory,  ""),
			JSON_BOOL(JSON_LABEL("network", "modl", "init_tickhonov"), &(modl.init_tickhonov), false,  ""),

			JSON_LONG(JSON_LABEL("network", "dw", "conv_layers"), &(modl.Nl), mandatory,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "filter"), &(modl.Nf), mandatory,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "kernels", "x"), &(modl.Kx), mandatory,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "kernels", "y"), &(modl.Ky), mandatory,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "kernels", "z"), &(modl.Kz), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dw", "batch_normalization"), &(modl.batch_norm), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dw", "residual_network"), &(modl.residual_network), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dw", "shared_weights"), &(modl.shared_weights), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dw", "reinsert_zerofilled"), &(modl.reinsert_zerofilled), false,  ""),

			JSON_BOOL(JSON_LABEL("network", "dc", "use_dc"), &(modl.use_dc), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dc", "shared_lambda"), &(modl.shared_weights), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "dc", "fixed_lambda"), &(modl.lambda_fixed), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "dc", "lambda_init"), &(modl.lambda_init), false,  ""),
			JSON_UINT(JSON_LABEL("network", "dc", "conjgrad_iterations"), &(def_conf.maxiter), false,  ""),

			JSON_BOOL(JSON_LABEL("training", "low_mem"), &(modl.low_mem), false,  ""),
			JSON_INT(JSON_LABEL("training", "epochs1"), &(epochs1), false,  ""),
			JSON_INT(JSON_LABEL("training", "epochs2"), &(epochs2), false,  ""),
			JSON_LONG(JSON_LABEL("training", "batch_size"), &(Nb), false,  ""),
			JSON_FLOAT(JSON_LABEL("training", "learning_rate"), &(train_conf.INTERFACE.learning_rate), false,  ""),
			JSON_INT(JSON_LABEL("training", "rand_batch_mode"), &(random_order), false,  ""),
			JSON_LONG(JSON_LABEL("training", "adam_reset_momentum"), &(train_conf.reset_epoch), false,  ""),
			JSON_STRING(JSON_LABEL("training", "history_file1"), &(history1), false,  ""),
			JSON_STRING(JSON_LABEL("training", "history_file2"), &(history2), false,  ""),
			JSON_LONG(JSON_LABEL("training", "checkpoint_interval"), &(train_conf.INTERFACE.dump_mod), false,  "")

		};
		read_json(config_file, ARRAY_SIZE(opts_json), opts_json);
	}

	if (one_iter) {

		modl.Nt = 1;

		if (0 < epochs1)
			train_conf.INTERFACE.epochs = epochs1;
		if (NULL != history1)
			train_conf.INTERFACE.history_filename = history1;
	} else {

		if (0 < epochs2)
			train_conf.INTERFACE.epochs = epochs2;
		if (NULL != history2)
			train_conf.INTERFACE.history_filename = history2;
	}

	train_conf.INTERFACE.batchgen_type = random_order;


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

			init_nn_modl(&modl);
			nn_modl_store_weights(&modl, filename_weights);
		} else
			error("Network needs to be either trained (-t) or applied (-a)!\n");
	}


#ifdef USE_CUDA
	if (use_gpu) {

		num_init_gpu();
		cuda_use_global_memory();
	}

	else
#endif
		num_init();



	if (apply)
		data.create_out = true;
	load_network_data(&data);
	bool use_valid_data = (NULL != valid_data.filename_coil) && (NULL != valid_data.filename_kspace) && (NULL != valid_data.filename_out);
	network_data_check_simple_dims(&data);


	if (train) {

		if (initialize == (NULL != filename_weights_load))
			error("For training, weights must be either initialized(-i) or loaded (-l)!\n");
		if (initialize)
			init_nn_modl(&modl);
		else
			nn_modl_load_weights(&modl, filename_weights_load, false);

		nn_modl_move_gpucpu(&modl, use_gpu);

		train_nn_modl(&modl, CAST_UP(&train_conf), data.idims, data.out, data.kdims, data.kspace, data.cdims, data.coil, data.pdims, data.pattern, Nb, use_valid_data ? &valid_data : NULL);
		nn_modl_store_weights(&modl, filename_weights);
	}


	if (apply) {

		nn_modl_load_weights(&modl, filename_weights, true);
		nn_modl_move_gpucpu(&modl, use_gpu);
		apply_nn_modl_batchwise(&modl, data.idims, data.out, data.kdims, data.kspace, data.cdims, data.coil, data.pdims, data.pattern, Nb);
	}

	nn_modl_free_weights(&modl);

	free_network_data(&data);


	exit(0);
}
