#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

#include "misc/misc.h"
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

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <pattern> <weights> <out/ref> <kspace_valid> <sens_valid> <pattern_valid> <ref_valid>";
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

	long udims[5] = {1, 1, 1, 1, 1};

	long Nb = 10;

	const char* config_file = NULL;
	bool load_mem = false;

	const struct opt_s opts[] = {

		OPTL_SET('i', "initialize", &initialize, "initialize weights"),
		OPTL_SET('t', "train", &train, "train variational network"),
		OPTL_SET('g', "gpu", &use_gpu, "run on gpu"),
		OPTL_SET('a', "apply", &apply, "apply variational network"),
		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "weights", "load weights for continuing training"),

		OPTL_STRING('c', "config", &config_file, "file", "file for loading varnet configuration"),

		OPTL_FLOAT('r', "learning_rate", &(train_conf.INTERFACE.learning_rate), "lr", "learning rate"),
		OPTL_INT('e', "epochs", &(train_conf.INTERFACE.epochs), "epochs", "number epochs to train"),
		OPTL_STRING(0, "save_train_history", (const char**)(&(train_conf.INTERFACE.history_filename)), "file", "file for dumping train history"),
		OPTL_STRING(0, "save_checkpoints_filename", (const char**)(&(train_conf.INTERFACE.dump_filename)), "file", "save intermediate weights during training (_epoch is attached to file)"),
		OPTL_LONG(0, "save_checkpoints_interval", &(train_conf.INTERFACE.dump_mod), "int", "save weights every int epochs"),
		OPTL_LONG('b', "batch_size", &(Nb), "Nb", "number epochs to train"),
		OPTL_INT(0, "randomize_batches", &(random_order), "", "0=no shuffle, 1=shuffle batches, 2= shuffle data, 3=randonly draw data"),

		OPTL_SET('n', "normalize", &(vn_config.normalize), "normalize the input by maximum of zero-filled reconstruction"),
		OPTL_SET('m', "load_data", &(load_mem), "load files int memory"),

		OPTL_LONG('X', "fov_x", (udims), "x", "Nx of the target image (guessed from reference(training) / kspace(inference))"),
		OPTL_LONG('Y', "fov_y", (udims + 1), "y", "Ny of the target image (guessed from reference(training) / kspace(inference))"),
		OPTL_LONG('Z', "fov_z", (udims + 2), "z", "Nz of the target image (guessed from reference(training) / kspace(inference))"),
	};

	cmdline(&argc, argv, 5, 9, usage_str, help_str, ARRAY_SIZE(opts), opts);
	if ((6 != argc) && (10 != argc))
		error("wrong number of arguments\n");

	if ((NULL != train_conf.INTERFACE.dump_filename) && (0 >= train_conf.INTERFACE.dump_mod))
		train_conf.INTERFACE.dump_mod = 5;
	if ((NULL == train_conf.INTERFACE.dump_filename) && (0 < train_conf.INTERFACE.dump_mod))
		train_conf.INTERFACE.dump_filename = argv[4];
	
	if (NULL != config_file) {

		const struct opt_json_s opts_json[] = {

			JSON_BOOL(JSON_LABEL("general", "low_mem"), &(vn_config.low_mem), false,  ""),

			JSON_BOOL(JSON_LABEL("data", "normalize"), &(vn_config.normalize), false,  ""),

			JSON_LONG(JSON_LABEL("network", "iterations"), &(vn_config.Nl), true,  ""),
			JSON_LONG(JSON_LABEL("network", "filter"), &(vn_config.Nf), true,  ""),
			JSON_LONG(JSON_LABEL("network", "num_rbf"), &(vn_config.Nw), true,  ""),
			
			JSON_LONG(JSON_LABEL("network", "kernels", "x"), &(vn_config.Kx), true,  ""),
			JSON_LONG(JSON_LABEL("network", "kernels", "y"), &(vn_config.Ky), true,  ""),
			JSON_LONG(JSON_LABEL("network", "kernels", "z"), &(vn_config.Kz), false,  ""),

			JSON_BOOL(JSON_LABEL("network", "init_tickhonov"), &(vn_config.init_tickhonov), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "init_tickhonov_lambda_fixed"), &(vn_config.lambda_fixed_tickhonov), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "init_tickhonov_lambda_init"), &(vn_config.lambda_init_tickhonov), false,  ""),	

			JSON_LONG(JSON_LABEL("apply", "target_dims", "x"), udims, false,  ""),
			JSON_LONG(JSON_LABEL("apply", "target_dims", "y"), udims + 1, false,  ""),
			JSON_LONG(JSON_LABEL("apply", "target_dims", "z"), udims + 2, false,  ""),

		};
		read_json(config_file, ARRAY_SIZE(opts_json), opts_json);
	}

	train_conf.INTERFACE.batchgen_type = random_order;

	//we only give K as commandline
	train_conf.beta = train_conf.alpha;
	train_conf.trivial_stepsize = true;


	char* filename_kspace = argv[1];
	char* filename_coil = argv[2];
	char* filename_pattern = argv[3];
	char* filename_weights = argv[4];
	char* filename_out = argv[5];

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

	long kdims[5]; 		//[Nkx, Nky, Nkz, Nc, Nt]
	long dims_coil[5]; 	//[Nkx, Nky, Nkz, Nc, Nt]
	long pdims[5]; 		//[Nkx, Nky, Nkz, 1,  1 or Nb]

	complex float* file_kspace = load_cfl(filename_kspace, 5, kdims);
	complex float* file_coil = load_cfl(filename_coil, 5, dims_coil);
	complex float* file_pattern = load_cfl(filename_pattern, 5, pdims);

	for (int i = 0; i < 5; i++)
		assert(kdims[i] == dims_coil[i]);
	for (int i = 0; i < 3; i++)
		assert(kdims[i] == pdims[i]);
	assert(1 == pdims[3]);
	assert((1 == pdims[4]) || (kdims[4] == pdims[4]));


	if (train){

		if (initialize == (NULL != filename_weights_load))
			error("For training, weights must be either initialized(-i) or loaded (-l)!\n");

		if (initialize)
			init_vn(&vn_config);
		else
			load_vn(&vn_config, filename_weights_load, false);

		if (use_gpu)
			move_gpu_nn_weights(vn_config.weights);

		complex float* file_ref = load_cfl(filename_out, 5, udims);

		train_vn(&vn_config, CAST_UP(&train_conf), udims, file_ref, kdims, file_kspace, file_coil, pdims, file_pattern, Nb, (10 == argc) ? (const char**)argv + 6: NULL);
		unmap_cfl(5, udims, file_ref);
		dump_nn_weights(filename_weights, vn_config.weights);
	}


	if (apply) {

		load_vn(&vn_config, filename_weights, true);
		if (use_gpu)
			move_gpu_nn_weights(vn_config.weights);

		udims[0] = (1 == udims[0]) ? kdims[0] : udims[0];
		udims[1] = (1 == udims[1]) ? kdims[1] : udims[1];
		udims[2] = (1 == udims[2]) ? kdims[2] : udims[2];
		udims[4] = (1 == udims[4]) ? kdims[4] : udims[4];

		complex float* file_out = create_cfl(filename_out, 5, udims);

		apply_vn_batchwise(&vn_config, udims, file_out, kdims, file_kspace, file_coil, pdims, file_pattern, Nb);

		unmap_cfl(5, udims, file_out);
	}

	unmap_cfl(5, pdims, file_pattern);
	unmap_cfl(5, kdims, file_kspace);
	unmap_cfl(5, kdims, file_coil);


	exit(0);
}
