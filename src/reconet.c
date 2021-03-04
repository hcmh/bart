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
#include "iter/iter.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/opts_json.h"
#include "misc/mmio.h"

#include "nn/weights.h"

#include "networks/reconet.h"
#include "networks/misc.h"

#include "nlops/mri_ops.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <weights> <out/ref>";
static const char help_str[] = "Trains or appplies MoDL.";

int main_reconet(int argc, char* argv[])
{
	struct reconet_s config = reconet_init;

	bool train = false;
	bool apply = false;
	bool eval = false;

	char* filename_weights_load = NULL;

	long Nb = 0;
	float learning_rate = 0.;
	long epochs = 0;
	int random_order = 0;
	const char* history_file = NULL;
	long dump_mod = -1;

	bool one_iter = false;

	const char* config_file = NULL;
	bool load_mem = false;
	
	bool normalize = false;
	bool regrid = false;

	bool varnet_default = false;
	bool modl_default = false;

	bool test_defaults = false;

	float lambda_init = 0;

	long cg_iter = 0;

	struct network_data_s data = network_data_empty;
	struct network_data_s valid_data = network_data_empty;

	const struct opt_s opts[] = {

		OPTL_SET('t', "train", &train, "train reconet"),
		OPTL_SET(0, "eval", &eval, "evaluate reconet"),
		OPTL_SET('g', "gpu", &(config.gpu), "run on gpu"),
		OPTL_SET('a', "apply", &apply, "apply reconet"),
		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "weights", "load weights for continuing training"),
		OPTL_STRING('c', "config", &config_file, "file", "file for loading modl configuration"),
		OPTL_SET('o', "one_iter", &one_iter, "only one iteration for initialization"),

		OPTL_SET(0, "modl_default", &(modl_default), "use MoDL Network"),
		OPTL_SET(0, "varnet_default", &(varnet_default), "use Variational Network"),

		OPTL_STRING(0, "trajectory", &(data.filename_trajectory), "file", "trajectory"),
		OPTL_STRING(0, "pattern", &(data.filename_pattern), "file", "sampling pattern / psf in kspace"),

		OPTL_STRING(0, "valid_trajectory", &(valid_data.filename_trajectory), "file", "validation data trajectory"),
		OPTL_STRING(0, "valid_pattern", &(valid_data.filename_pattern), "file", "validation data sampling pattern / psf in kspace"),
		OPTL_STRING(0, "valid_kspace", &(valid_data.filename_kspace), "file", "validation data kspace"),
		OPTL_STRING(0, "valid_coil", &(valid_data.filename_coil), "file", "validation data sensitivity maps"),
		OPTL_STRING(0, "valid_ref", &(valid_data.filename_out), "file", "validation data reference"),

		OPTL_FLOAT('r', "learning_rate", &(learning_rate), "lr", "learning rate"),
		OPTL_LONG('e', "epochs", &(epochs), "epochs", "number epochs to train"),
		OPTL_LONG('b', "batch_size", &(Nb), "Nb", "size of mini batches"),
		OPTL_INT(0, "randomize_batches", &(random_order), "", "0=no shuffle, 1=shuffle batches, 2= shuffle data, 3=randonly draw data"),		
		
		OPTL_STRING(0, "save_train_history", (const char**)(&(history_file)), "file", "file for dumping train history"),
		OPTL_LONG(0, "save_checkpoints_interval", &(dump_mod), "int", "save weights every int epochs"),
		
		OPTL_SET('n', "normalize", &(normalize), "normalize the input by maximum of zero-filled reconstruction"),
		OPTL_SET('m', "load_data", &(load_mem), "load files int memory"),
		OPTL_SET(0, "low_mem", &(config.low_mem), "reduce memory usage by checkpointing"),
		OPTL_SET(0, "regrid", &(regrid), "grids fully sampled kspace by applying pattern"),

		OPTL_LONG('T', "num_net_iter", &(config.Nt), "", "number of iterations of reconet"),
		OPTL_LONG(0, "num_cg_iter", &(cg_iter), "", "number of conjugate gradient iterations"),

		OPTL_SET(0, "test", &(test_defaults), "very small network for tests"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (test_defaults) {

		if (modl_default)
			reconet_init_modl_test_default(&config);
		if (varnet_default)
			reconet_init_varnet_test_default(&config);

	} else {
	
		if (modl_default)
			reconet_init_modl_default(&config);
		if (varnet_default)
			reconet_init_varnet_default(&config);
	}

	long epochs1 = epochs;
	long epochs2 = epochs;

	const char* history1 = NULL;
	const char* history2 = NULL;

	if (NULL != config_file) {

		const struct opt_json_s opts_json[] = {

			JSON_BOOL(JSON_LABEL("data", "normalize"), &(normalize), false,  ""),
			JSON_BOOL(JSON_LABEL("data", "regrid"), &(regrid), false,  ""),

			JSON_BOOL(JSON_LABEL("training", "low_mem"), &(config.low_mem), false,  ""),
			JSON_LONG(JSON_LABEL("training", "epochs"), &(epochs2), false,  ""),
			JSON_LONG(JSON_LABEL("training", "epochs1"), &(epochs1), false,  ""),
			JSON_LONG(JSON_LABEL("training", "epochs2"), &(epochs2), false,  ""),
			JSON_LONG(JSON_LABEL("training", "batch_size"), &(Nb), false,  ""),
			JSON_FLOAT(JSON_LABEL("training", "learning_rate"), &(learning_rate), false,  ""),
			JSON_INT(JSON_LABEL("training", "rand_batch_mode"), &(random_order), false,  ""),
			JSON_STRING(JSON_LABEL("training", "history_file"), &(history2), false,  ""),
			JSON_STRING(JSON_LABEL("training", "history_file1"), &(history1), false,  ""),
			JSON_STRING(JSON_LABEL("training", "history_file2"), &(history2), false,  ""),
			JSON_LONG(JSON_LABEL("training", "checkpoint_interval"), &(dump_mod), false,  ""),

			JSON_LONG(JSON_LABEL("reconet", "iterations"), &(config.Nt), false,  ""),
			JSON_BOOL(JSON_LABEL("reconet", "share_weights"), &(config.share_weights), false,  ""),
			JSON_BOOL(JSON_LABEL("reconet", "share_lambda"), &(config.share_lambda), false,  ""),
			JSON_BOOL(JSON_LABEL("reconet", "init_tickhonov"), &(config.tickhonov_init), false,  ""),
			JSON_BOOL(JSON_LABEL("reconet", "reinsert_init"), &(config.reinsert), false,  ""),

			JSON_LONG(JSON_LABEL("reconet", "dc", "cg_iter"), &(cg_iter), false,  ""),
			JSON_FLOAT(JSON_LABEL("reconet", "dc", "lambda_init"), &(lambda_init), false,  ""),
		};
		read_json(config_file, ARRAY_SIZE(opts_json), opts_json);
	}

	if (0 < learning_rate)
		config.train_conf->learning_rate = learning_rate;

	if (one_iter) {

		config.Nt = 1;

		if (0 < epochs1)
			config.train_conf->epochs = epochs1;
		if (NULL != history1)
			config.train_conf->history_filename = history1;
	} else {

		if (0 < epochs2)
			config.train_conf->epochs = epochs2;
		if (NULL != history2)
			config.train_conf->history_filename = history2;
	}

	if (regrid)
		config.mri_config->regrid = true;

	if (0 == Nb)
		Nb = 10;
	
	if (0 != cg_iter) {

		config.mri_config_dc_init->iter_conf->maxiter = cg_iter;
		config.mri_config_dc->iter_conf->maxiter = cg_iter;
	}

	if (0 < lambda_init) {

		config.mri_config_dc_init->lambda_init = lambda_init;
		config.mri_config_dc->lambda_init = lambda_init;
	}

	if (normalize)
		config.normalize = MD_BIT(4);

	if (0 < dump_mod)
		config.train_conf->dump_mod = dump_mod;


	config.train_conf->batchgen_type = random_order;


	data.filename_kspace = argv[1];
	data.filename_coil = argv[2];
	const char* filename_weights = argv[3];
	data.filename_out = argv[4];

	if (0 < config.train_conf->dump_mod)
		config.train_conf->dump_filename = filename_weights;


	if (((train || eval) && apply) || (!train && !apply && ! eval))
		error("Network must be either trained (-t) or applied(-a)!\n");

#ifdef USE_CUDA
	if (config.gpu) {

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

	if (NULL != filename_weights_load)
		config.weights = load_nn_weights(filename_weights_load);

	if (train) {

		train_reconet(&config, 5, data.idims, data.out, data.kdims, data.kspace, data.cdims, data.coil, data.pdims, data.pattern, Nb, use_valid_data ? &valid_data : NULL);
		dump_nn_weights(filename_weights, config.weights);
	}

	if (eval) {

		if (NULL == config.weights)
			config.weights = load_nn_weights(filename_weights);
		eval_reconet(&config, 5, data.idims, data.out, data.kdims, data.kspace, data.cdims, data.coil, data.pdims, data.pattern, Nb);
	}

	if (apply) {

		config.weights = load_nn_weights(filename_weights);
		apply_reconet_batchwise(&config, 5, data.idims, data.out, data.kdims, data.kspace, data.cdims, data.coil, data.pdims, data.pattern, Nb);
	}

	nn_weights_free(config.weights);

	free_network_data(&data);


	exit(0);
}
