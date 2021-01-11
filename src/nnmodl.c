#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <libgen.h>
#include <string.h>

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

#include "networks/modl.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<kspace> <sens> <pattern> <weights> <out>";
static const char help_str[] = "Trains and appplies the MoDL.";

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

	long udims[5] = {1, 1, 1, 1, 1};

	long Nb = 10;
	bool one_iter = false;

	bool draw_graph = false;

	const char* config_file = NULL;
	bool load_mem = false;

	bool test_defaults = false;

	const struct opt_s opts[] = {

		OPTL_SET('i', "initialize", &initialize, "initialize weights"),
		OPTL_SET('t', "train", &train, "train variational network"),
		OPTL_SET('g', "gpu", &use_gpu, "run on gpu"),
		OPTL_SET('a', "apply", &apply, "apply variational network"),
		OPTL_STRING('l', "load", (const char**)(&(filename_weights_load)), "weights", "load weights for continuing training"),
		OPTL_SET(0, "export_graph", &(draw_graph), "export graph.dot file in the weights directory"),

		OPTL_STRING('c', "config", &config_file, "file", "file for loading modl configuration"),

		OPTL_FLOAT('r', "learning_rate", &(train_conf.INTERFACE.learning_rate), "lr", "learning rate"),
		OPTL_INT('e', "epochs", &(train_conf.INTERFACE.epochs), "epochs", "number epochs to train"),
		OPTL_STRING(0, "save_train_history", (const char**)(&(train_conf.INTERFACE.history_filename)), "file", "file for dumping train history"),
		OPTL_STRING(0, "save_checkpoints_filename", (const char**)(&(train_conf.INTERFACE.dump_filename)), "file", "save intermediate weights during training (_epoch is attached to file)"),
		OPTL_LONG(0, "save_checkpoints_interval", &(train_conf.INTERFACE.dump_mod), "int", "save weights every int epochs"),
		OPTL_LONG('b', "batch_size", &(Nb), "Nb", "number epochs to train"),
		OPTL_LONG(0, "adam_reset_momentum", &(train_conf.reset_epoch), "epoch", "reset the adam algorithm after this number of epochs"),
		OPTL_INT(0, "randomize_batches", &(random_order), "", "0=no shuffle, 1=shuffle batches, 2= shuffle data, 3=randonly draw data"),

		OPTL_SET('n', "normalize", &(modl.normalize), "normalize the input by maximum of zero-filled reconstruction"),
		OPTL_SET('m', "load_data", &(load_mem), "load files int memory"),
		OPTL_SET(0, "low_mem", &(modl.low_mem), "reduce memory usage by checkpointing"),

		OPTL_SET('o', "one_iter", &one_iter, "only one iteration"),

		OPTL_LONG('X', "fov_x", (udims), "x", "Nx of the target image (guessed from reference(training) / kspace(inference))"),
		OPTL_LONG('Y', "fov_y", (udims + 1), "y", "Ny of the target image (guessed from reference(training) / kspace(inference))"),
		OPTL_LONG('Z', "fov_z", (udims + 2), "z", "Nz of the target image (guessed from reference(training) / kspace(inference))"),

		OPTL_SET(0, "test_defaults", &test_defaults, "set defaults to small values (used for testing)"),
	};

	cmdline(&argc, argv, 5, 9, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (test_defaults) {

		modl.Nf = 8;
		modl.Nl = 3;
		modl.Nt = 2;

		def_conf.maxiter = 10;
	}

	if (NULL != config_file) {

		const struct opt_json_s opts_json[] = {

			JSON_BOOL(JSON_LABEL("data", "normalize"), &(modl.normalize), false,  ""),
			JSON_BOOL(JSON_LABEL("general", "low_mem"), &(modl.low_mem), false,  ""),

			JSON_LONG(JSON_LABEL("network", "modl", "iterations"), &(modl.Nt), true,  ""),
			JSON_BOOL(JSON_LABEL("network", "modl", "reinsert_zerofilled"), &(modl.reinsert_zerofilled), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "modl", "init_tickhonov"), &(modl.init_tickhonov), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "modl", "nullspace_formulation"), &(modl.nullspace), false,  ""),

			JSON_LONG(JSON_LABEL("network", "dw", "conv_layers"), &(modl.Nl), true,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "filter"), &(modl.Nf), true,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "kernels", "x"), &(modl.Kx), true,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "kernels", "y"), &(modl.Ky), true,  ""),
			JSON_LONG(JSON_LABEL("network", "dw", "kernels", "z"), &(modl.Kz), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dw", "batch_normalization"), &(modl.batch_norm), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dw", "residual_network"), &(modl.residual_network), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dw", "shared_weights"), &(modl.shared_weights), false,  ""),

			JSON_BOOL(JSON_LABEL("network", "dc", "use_dc"), &(modl.use_dc), false,  ""),
			JSON_BOOL(JSON_LABEL("network", "dc", "shared_lambda"), &(modl.shared_weights), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "dc", "fixed_lambda"), &(modl.lambda_fixed), false,  ""),
			JSON_FLOAT(JSON_LABEL("network", "dc", "lambda_init"), &(modl.lambda_init), false,  ""),
			JSON_UINT(JSON_LABEL("network", "dc", "conjgrad_iterations"), &(def_conf.maxiter), false,  ""),

			JSON_LONG(JSON_LABEL("apply", "target_dims", "x"), udims, false,  ""),
			JSON_LONG(JSON_LABEL("apply", "target_dims", "y"), udims + 1, false,  ""),
			JSON_LONG(JSON_LABEL("apply", "target_dims", "z"), udims + 2, false,  ""),

		};
		read_json(config_file, ARRAY_SIZE(opts_json), opts_json);
	}

	if (one_iter)
		modl.Nt = 1;

	train_conf.INTERFACE.batchgen_type = random_order;

	const char* filename_kspace = argv[1];
	const char* filename_coil = argv[2];
	const char* filename_pattern = argv[3];
	const char* filename_weights = argv[4];
	const char* filename_out = argv[5];

	char graph_path[strlen(filename_weights) + 10];
	char filename_weights_tmp[strlen(filename_weights) + 1];
	strcpy(filename_weights_tmp, filename_weights);
	sprintf(graph_path, "%s/graph.dot", dirname(filename_weights_tmp));
	if (draw_graph)
		modl.draw_graph_filename = graph_path;


	if ((NULL != train_conf.INTERFACE.dump_filename) && (0 >= train_conf.INTERFACE.dump_mod))
		train_conf.INTERFACE.dump_mod = 5;
	if ((NULL == train_conf.INTERFACE.dump_filename) && (0 < train_conf.INTERFACE.dump_mod))
		train_conf.INTERFACE.dump_filename = argv[4];


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


	long kdims[5]; 		//[Nkx, Nky, Nkz, Nc, Nt]
	long cdims[5]; 		//[Nkx, Nky, Nkz, Nc, Nt]
	long pdims[5]; 		//[Nkx, Nky, Nkz, 1,  1 or Nb]

	complex float* kspace = load_cfl(filename_kspace, 5, kdims);
	complex float* coil = load_cfl(filename_coil, 5, cdims);
	complex float* pattern = load_cfl(filename_pattern, 5, pdims);


	if (load_mem) {

		complex float* mem_kspace = md_alloc(5, kdims, CFL_SIZE);
		complex float* mem_coil = md_alloc(5, cdims, CFL_SIZE);
		complex float* mem_pattern = md_alloc(5, pdims, CFL_SIZE);

		md_copy(5, kdims, mem_kspace, kspace, CFL_SIZE);
		md_copy(5, cdims, mem_coil, coil, CFL_SIZE);
		md_copy(5, pdims, mem_pattern, pattern, CFL_SIZE);

		unmap_cfl(5, kdims, kspace);
		unmap_cfl(5, cdims, coil);
		unmap_cfl(5, pdims, pattern);

		kspace = mem_kspace;
		coil = mem_coil;
		pattern = mem_pattern;
	}

	for (int i = 0; i < 5; i++)
		assert(kdims[i] == cdims[i]);
	for (int i = 0; i < 3; i++)
		assert(kdims[i] == pdims[i]);
	assert(1 == pdims[3]);
	assert((1 == pdims[4]) || (kdims[4] == pdims[4]));


	if (train) {

		if (initialize == (NULL != filename_weights_load))
			error("For training, weights must be either initialized(-i) or loaded (-l)!\n");

		if (initialize)
			init_nn_modl(&modl);
		else
			nn_modl_load_weights(&modl, filename_weights_load, false);

		nn_modl_move_gpucpu(&modl, use_gpu);


		complex float* ref = load_cfl(filename_out, 5, udims);
		if (load_mem) {

			complex float* mem_ref = md_alloc(5, udims, CFL_SIZE);
			md_copy(5, udims, mem_ref, ref, CFL_SIZE);
			unmap_cfl(5, udims, ref);
			ref = mem_ref;
		}

		train_nn_modl(&modl, CAST_UP(&train_conf), udims, ref, kdims, kspace, coil, pdims, pattern, Nb, (10 == argc) ? (const char**)argv + 6: NULL);
		nn_modl_store_weights(&modl, filename_weights);
		if (load_mem)
			md_free(ref);
		else
			unmap_cfl(5, udims, ref);
	}


	if (apply) {

		nn_modl_load_weights(&modl, filename_weights, true);
		nn_modl_move_gpucpu(&modl, use_gpu);

		udims[0] = (1 == udims[0]) ? kdims[0] : udims[0];
		udims[1] = (1 == udims[1]) ? kdims[1] : udims[1];
		udims[2] = (1 == udims[2]) ? kdims[2] : udims[2];
		udims[4] = (1 == udims[4]) ? kdims[4] : udims[4];

		complex float* file_out = create_cfl(filename_out, 5, udims);

		apply_nn_modl_batchwise(&modl, udims, file_out, kdims, kspace, coil, pdims, pattern, Nb);

		unmap_cfl(5, udims, file_out);

	}

	nn_modl_free_weights(&modl);

	if (load_mem) {

		md_free(pattern);
		md_free(kspace);
		md_free(coil);

	} else {

		unmap_cfl(5, pdims, pattern);
		unmap_cfl(5, kdims, kspace);
		unmap_cfl(5, cdims, coil);
	}


	exit(0);
}
