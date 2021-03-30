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

#include "networks/cnn.h"
#include "networks/unet.h"
#include "networks/reconet.h"
#include "networks/losses.h"
#include "networks/misc.h"

#include "nlops/mri_ops.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <weights> <out/ref>";
static const char help_str[] = "Trains or appplies a neural network for reconstruction.";

int main_reconet(int argc, char* argv[])
{
	struct reconet_s config = reconet_config_opts;

	bool train = false;
	bool apply = false;
	bool eval = false;

	char* filename_weights_load = NULL;

	long Nb = 0;

	bool one_iter = false;
	bool load_mem = false;

	bool normalize = false;
	bool regrid = false;

	bool varnet_default = false;
	bool modl_default = false;
	bool unet_default = false;

	bool test_defaults = false;

	enum NETWORK_SELECT net = NETWORK_NONE;

	const char* graph_filename = NULL;

	struct network_data_s data = network_data_empty;
	struct network_data_s valid_data = network_data_empty;

	struct opt_s dc_opts[] = {

		OPTL_FLOAT(0, "fix-lambda", &(config.dc_lambda_fixed), "float", "fix lambda to specified value (-1 means train lambda)"),
		OPTL_FLOAT(0, "lambda-init", &(config.dc_lambda_init), "float", "initialize lambda eith specified value"),
		OPTL_SET(0, "dc-gradient-step", &(config.dc_gradient), "use gradient steps for data-consistency"),
		OPTL_SET(0, "dc-proximal-mapping", &(config.dc_tickhonov), "use proximal mapping for data-consistency"),
		OPTL_INT(0, "dc-max-cg-iter", &(config.dc_max_iter), "int", "number of cg steps for proximal mapping"),

		OPTL_SET(0, "init-tickhonov", &(config.tickhonov_init), "init network with Tickhonov regularized reconstruction instead of adjoint reconstruction"),
		OPTL_INT(0, "init-max-cg-iter", &(config.init_max_iter), "int", "number of cg steps for initialization"),
		OPTL_FLOAT(0, "init-fix-lambda", &(config.init_lambda_fixed), "float", "fix lambda for initialization to specified value (-1 means train lambda)"),
		OPTL_FLOAT(0, "init-lambda-init", &(config.init_lambda_init), "float", "initialize lambda for initialization with specified value"),
	};

	struct opt_s valid_opts[] = {

		OPTL_STRING('t', "trajectory", &(valid_data.filename_trajectory), "file", "validation data trajectory"),
		OPTL_STRING('p', "pattern", &(valid_data.filename_pattern), "file", "validation data sampling pattern / psf in kspace"),
		OPTL_STRING('k', "kspace", &(valid_data.filename_kspace), "file", "validation data kspace"),
		OPTL_STRING('c', "coil", &(valid_data.filename_coil), "file", "validation data sensitivity maps"),
		OPTL_STRING('r', "ref", &(valid_data.filename_out), "file", "validation data reference"),
	};

	struct opt_s network_opts[] = {

		OPTL_SET(0, "modl", &(modl_default), "use MoDL Network (also sets train and data-consistency default values)"),
		OPTL_SET(0, "varnet", &(varnet_default), "use Variational Network (also sets train and data-consistency default values)"),
		OPTL_SET(0, "unet", &(unet_default), "use U-Net (also sets train and data-consistency default values)"),

		OPTL_SELECT(0, "resnet-block", enum NETWORK_SELECT, &net, NETWORK_RESBLOCK, "use residual block (overwrite default)"),
		OPTL_SELECT(0, "varnet-block", enum NETWORK_SELECT, &net, NETWORK_VARNET, "use variational block (overwrite default)"),
	};

	const struct opt_s opts[] = {

		OPTL_SET('t', "train", &train, "train reconet"),
		OPTL_SET('e', "eval", &eval, "evaluate reconet"),
		OPTL_SET('a', "apply", &apply, "apply reconet"),

		OPTL_SET('g', "gpu", &(config.gpu), "run on gpu"),

		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "file", "load weights for continuing training"),
		OPTL_LONG('b', "batch-size", &(Nb), "d", "size of mini batches"),

		OPTL_LONG('I', "iterations", &(config.Nt), "d", "number of unrolled iterations"),

		OPTL_SET('n', "normalize", &(config.normalize), "normalize data with maximum magnitude of adjoint reconstruction"),
		OPTL_SET(0, "regrid", &(regrid), "grids fully sampled kspace by applying pattern"),

		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),
		OPTL_SUBOPT(0, "config-resnet-block", "...", "configure residual block", N_res_block_opts, res_block_opts),
		OPTL_SUBOPT(0, "config-varnet-block", "...", "configure variational block", N_variational_block_opts, variational_block_opts),
		OPTL_SUBOPT(0, "config-unet", "...", "configure U-Net block", N_unet_reco_opts, unet_reco_opts),

		OPTL_SUBOPT(0, "config-dc", "...", "configure data-consistency methode", ARRAY_SIZE(dc_opts), dc_opts),

		OPTL_SELECT(0, "shared-weights", enum BOOL_SELECT, &(config.share_weights_select), BOOL_TRUE, "share weights across iterations"),
		OPTL_SELECT(0, "no-shared-weights", enum BOOL_SELECT, &(config.share_weights_select), BOOL_FALSE, "share weights across iterations"),
		OPTL_SELECT(0, "shared-lambda", enum BOOL_SELECT, &(config.share_lambda_select), BOOL_TRUE, "share lambda across iterations"),
		OPTL_SELECT(0, "no-shared-lambda", enum BOOL_SELECT, &(config.share_lambda_select), BOOL_FALSE, "share lambda across iterations"),


		OPTL_STRING(0, "trajectory", &(data.filename_trajectory), "file", "trajectory"),
		OPTL_STRING(0, "pattern", &(data.filename_pattern), "file", "sampling pattern / psf in kspace"),

		OPTL_SUBOPT(0, "valid-data", "...", "provide validation data", ARRAY_SIZE(valid_opts),valid_opts),

		OPTL_SUBOPT(0, "loss", "...", "configure the training loss", N_loss_opts, loss_opts),
		OPTL_SUBOPT(0, "valid-loss", "...", "configure the validation loss", N_val_loss_opts, val_loss_opts),

		OPTL_SUBOPT('T', "train-config", "...", "configure general training parmeters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure Adam", N_iter6_adam_opts, iter6_adam_opts),
		OPTL_SUBOPT(0, "iPALM", "...", "configure iPALM", N_iter6_ipalm_opts, iter6_ipalm_opts),

		OPTL_SET('o', "one-iter", &one_iter, "only one iteration for initialization"),

		OPTL_SET('m', "load-data", &(load_mem), "load files int memory"),
		OPTL_SET(0, "low-mem", &(config.low_mem), "reduce memory usage by checkpointing"),

		OPTL_SET(0, "test", &(test_defaults), "very small network for tests"),
		OPTL_STRING(0, "export_graph", (const char**)(&(graph_filename)), "file.dot", "file for dumping graph"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	data.filename_kspace = argv[1];
	data.filename_coil = argv[2];
	const char* filename_weights = argv[3];
	data.filename_out = argv[4];

	config.train_conf = iter6_get_conf_from_opts();

	config.network = get_default_network(net);

	if (test_defaults) {

		if (modl_default)
			reconet_init_modl_test_default(&config);
		if (varnet_default)
			reconet_init_varnet_test_default(&config);
		if (unet_default)
			reconet_init_unet_test_default(&config);

	} else {

		if (modl_default)
			reconet_init_modl_default(&config);
		if (varnet_default)
			reconet_init_varnet_default(&config);
		if (unet_default)
			reconet_init_unet_default(&config);
	}

	iter6_copy_config_from_opts(config.train_conf);

	if ((0 < config.train_conf->dump_mod) && (NULL == config.train_conf->dump_filename))
		config.train_conf->dump_filename = filename_weights;

	if (one_iter)
		config.Nt = 1;

	if (regrid)
		config.mri_config->regrid = true;

	if (0 == Nb)
		Nb = 10;

	if (normalize)
		config.normalize = true;

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

	config.graph_file = graph_filename;

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
