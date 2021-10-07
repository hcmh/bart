#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/rand.h"

#include "noncart/nufft.h"
#include "linops/linop.h"

#include "nlops/nlop.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noir/recon2.h"
#include "noir/model_net.h"

#include "nn/weights.h"

#include "grecon/opt_iter6.h"
#include "grecon/losses.h"
#include "grecon/network.h"


#include "networks/cnn.h"
#include "networks/unet.h"
#include "networks/reconet.h"
#include "networks/losses.h"
#include "networks/misc.h"
#include "networks/nlinvnet.h"

#include "noir/misc.h"

#ifdef USE_CUDA
#include "num/gpuops.c"
#endif


static const char help_str[] =
		"";



int main_nlinvnet(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* out_file = NULL;
	const char* weight_file = NULL;
	const char* sens_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INOUTFILE(true, &weight_file, "weights"),
		ARG_INOUTFILE(true, &out_file, "output/reference"),
		ARG_OUTFILE(false, &sens_file, "sensitivities"),
	};

	const char* pat_file = NULL;
	const char* traj_file = NULL;

	struct noir2_conf_s conf = noir2_defaults;
	struct nlinvnet_s nlinvnet = nlinvnet_config_opts;
	nlinvnet.conf = &conf;

	float coil_os = -1;

	bool train = false;
	bool apply = false;
	bool eval = false;

	long Nb = 0;

	char* filename_weights_load = NULL;

	const char* val_file_kspace = NULL;
	const char* val_file_reference = NULL;

	opts_iter6_init();

	bool mask_lines = true;
	float ratio_train = 0.;


	struct opt_s valid_opts[] = {

		//OPTL_INFILE('p', "pattern", &(val_file_pattern), "<file>", "validation data sampling pattern"), currently use same model as training
		OPTL_INFILE('k', "kspace", &(val_file_kspace), "<file>", "validation data kspace"),
		OPTL_INFILE('r', "ref", &(val_file_reference), "<file>", "validation data reference"),
	};

	bool unet = false;
	bool kspace_network = false;

	struct opt_s network_opts[] = {

		OPTL_SET(0, "kspace", &kspace_network, "add network in kspace"),
		OPTL_SET(0, "unet", &(unet), "use U-Net (also sets train and data-consistency default values)"),
	};

	const struct opt_s opts[] = {

		OPTL_SET('t', "train", &train, "train nlinvnet"),
		OPTL_SET('e', "eval", &eval, "evaluate nlinvnet"),
		OPTL_SET('a', "apply", &apply, "apply nlinvnet"),

		OPTL_SET('g', "gpu", &(nlinvnet.gpu), "run on gpu"),
		OPTL_LONG('b', "batch-size", &(Nb), "", "size of mini batches"),

		OPTL_SET(0, "rss-loss", &(nlinvnet.rss_loss), "train on rss instead of coil images"),
		OPTL_SET(0, "rss-norm", &(nlinvnet.normalize_rss), "scale output image to rss normalization"),
		OPTL_SET(0, "fix-lambda", &(nlinvnet.fix_lambda), "fix lambda"),
		OPTL_SET(0, "fix-coils", &(nlinvnet.fix_coils), "append layer with fixed coils"),

		OPTL_INT(0, "iter-net", &(nlinvnet.iter_net), "iter", "number of iterations with network"),
		//OPTL_INT(0, "iter-net-shift", &(nlinvnet.iter_net_shift), "iter", "no network in the last \"iter\" steps"),

		OPTL_SUBOPT(0, "resnet-block", "...", "configure residual block", N_res_block_opts, res_block_opts),
		OPTL_CLEAR(0, "no-shared-weights", &(nlinvnet.share_weights), "don't share weights across iterations"),
		OPTL_INFILE(0, "pattern", &pat_file, "<pattern>", "sampling pattern"),

		OPTL_SUBOPT(0, "valid-data", "...", "provide validation data", ARRAY_SIZE(valid_opts),valid_opts),

		OPTL_SUBOPT('T', "train-algo", "...", "configure general training parmeters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure Adam", N_iter6_adam_opts, iter6_adam_opts),

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		//OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		//OPTL_FLOAT(0, "sobolev-a", &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		//OPTL_FLOAT(0, "sobolev-b", &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		//OPTL_FLOAT(0, "alpha", &conf.alpha, "", "(start regularization)"),
		//OPTL_FLOAT(0, "alpha-min", &conf.alpha_min, "", "(minimum for regularization)"),
		OPTL_FLOAT(0, "coil-os", &coil_os, "val", "(over-sampling factor for sensitivities)"),

		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),

		OPTL_INFILE('l', "load", (const char**)(&(filename_weights_load)), "<weights-init>", "load weights for continuing training"),

		OPTL_INT(0, "cgiter", &(conf.cgiter), "", "number of cg iterations"),

		OPTL_SUBOPT(0, "train-loss", "...", "configure the training loss", N_loss_opts, loss_opts),

		OPTL_FLOAT(0, "train-mask-ratio", &ratio_train, "p", "use p\% of kspace lines as reference"),
		OPTL_CLEAR(0, "train-mask-points", &mask_lines, "use pointwise kspace mask instead of full lines")
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	(nlinvnet.gpu ? num_init_gpu_memopt : num_init)();
	#ifdef USE_CUDA
	if (nlinvnet.gpu)
		cuda_use_global_memory();
	#endif

	if (train) {

		nlinvnet.train_conf = iter6_get_conf_from_opts();

		if (NULL == nlinvnet.train_conf) {

			debug_printf(DP_WARN, "No training algorithm selected. Fallback to Adam!");
			iter_6_select_algo = ITER6_ADAM;
			nlinvnet.train_conf = iter6_get_conf_from_opts();

		} else
			iter6_copy_config_from_opts(nlinvnet.train_conf);
	}

	nlinvnet.network = get_default_network(unet ? NETWORK_UNET_RECO : NETWORK_RESBLOCK);
	if (kspace_network) {

		network_combi_kspace_default.img_net = nlinvnet.network;
		nlinvnet.network = CAST_UP(&network_combi_kspace_default);
	}


	nlinvnet.network->norm = NORM_MAX;

	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);
	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}

	assert(1 == ksp_dims[MAPS_DIM]);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	long dims[DIMS];

	long trj_dims[DIMS];
	complex float* traj  = NULL;

	if (NULL != traj_file) {

		conf.noncart = true;

		traj = load_cfl(traj_file, DIMS, trj_dims);

		estimate_im_dims(DIMS, FFT_FLAGS, dims, trj_dims, traj);
		debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);

		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);
	} else {

		md_copy_dims(DIMS, dims, ksp_dims);
	}
	if (-1 == coil_os)
		coil_os = conf.noncart ? 2 : 1;

	dims[MAPS_DIM] = 1;

	long sens_dims[DIMS];
	md_select_dims(DIMS, ~0, sens_dims, dims);

	for (int i = 0; i < 3; i++)
		if (1 != sens_dims[i])
			sens_dims[i] = lround(coil_os * sens_dims[i]);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, dims);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

	long col_dims_s[DIMS];
	long img_dims_s[DIMS];
	long cim_dims_s[DIMS];
	long msk_dims_s[DIMS];
	long ksp_dims_s[DIMS];
	long pat_dims_s[DIMS];

	md_select_dims(DIMS, ~BATCH_FLAG, col_dims_s, sens_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, img_dims_s, img_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, cim_dims_s, cim_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, msk_dims_s, msk_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, ksp_dims_s, ksp_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, pat_dims_s, pat_dims);

	complex float* trn_mask = md_alloc(DIMS, pat_dims_s, CFL_SIZE);
	md_copy(DIMS, pat_dims_s, trn_mask, pattern, CFL_SIZE);

	if (0. != ratio_train) {

		long tdims[DIMS];
		md_select_dims(DIMS, mask_lines ? MD_BIT(1) | MD_BIT(2) : FFT_FLAGS, tdims, pat_dims_s);
		complex float* rand_mask = md_alloc(DIMS, tdims, CFL_SIZE);

		long pstrs[DIMS];
		long tstrs[DIMS];

		md_calc_strides(DIMS, pstrs, pat_dims_s, CFL_SIZE);
		md_calc_strides(DIMS, tstrs, tdims, CFL_SIZE);

		md_rand_one(DIMS, tdims, rand_mask, ratio_train);

		md_zmul2(DIMS, pat_dims_s, pstrs, pattern, pstrs, pattern, tstrs, rand_mask);
		md_zsadd(DIMS, tdims, rand_mask, rand_mask, -1.);
		md_zsmul(DIMS, tdims, rand_mask, rand_mask, -1.);
		md_zmul2(DIMS, pat_dims_s, pstrs, trn_mask, pstrs, trn_mask, tstrs, rand_mask);

		md_free(rand_mask);
	}


	nlinvnet_init_model_cart(&nlinvnet, DIMS,
		pat_dims_s, pattern,
		MD_SINGLETON_DIMS(DIMS), NULL,
		msk_dims_s, NULL,
		ksp_dims_s,
		cim_dims_s,
		img_dims_s,
		col_dims_s);

	Nb = Nb ? Nb : 10;

	if (train) {

		if (NULL != filename_weights_load)
			nlinvnet.weights = load_nn_weights(filename_weights_load);

		long ksp_dims_val[DIMS];
		long cim_dims_val[DIMS];

		complex float* val_kspace = NULL;
		complex float* val_ref = NULL;

		if (NULL != val_file_kspace) {

			val_kspace = load_cfl(val_file_kspace, DIMS, ksp_dims_val);
			val_ref = load_cfl(val_file_reference, DIMS, cim_dims_val);
		}

		if (0. == loss_option.weighting_mse_mask_ksp) {

			long cim_dims2[DIMS];
			complex float* ref = load_cfl(out_file, DIMS, cim_dims2);
			assert(md_check_equal_dims(DIMS, cim_dims, cim_dims2, ~0));

			train_nlinvnet(&nlinvnet, DIMS, Nb, cim_dims, ref, ksp_dims, kspace, cim_dims_val, val_ref, ksp_dims_val, val_kspace);

			unmap_cfl(DIMS, cim_dims, ref);
		} else {

			complex float* ref = md_alloc(DIMS, ksp_dims, CFL_SIZE);
			ifftuc(DIMS, ksp_dims, FFT_FLAGS, ref, kspace);

			loss_option.mask = trn_mask;

			train_nlinvnet(&nlinvnet, DIMS, Nb, cim_dims, ref, ksp_dims, kspace, cim_dims_val, val_ref, ksp_dims_val, val_kspace);

			md_free(ref);
		}


		dump_nn_weights(weight_file, nlinvnet.weights);

		if (NULL != val_file_kspace) {

			unmap_cfl(DIMS, ksp_dims_val, val_kspace);
			unmap_cfl(DIMS, cim_dims_val, val_ref);
		}
	}

	if (apply) {

		complex float* img = create_cfl(out_file, DIMS, img_dims);

		complex float* col = (NULL != sens_file) ? create_cfl(sens_file, DIMS, dims) : anon_cfl("", DIMS, dims);
		nlinvnet.weights = load_nn_weights(weight_file);

		apply_nlinvnet(&nlinvnet, DIMS, img_dims, img, dims, col, ksp_dims, kspace);

		unmap_cfl(DIMS, img_dims, img);
		unmap_cfl(DIMS, dims, col);
	}

	if (eval) {

		long cim_dims2[DIMS];
		complex float* ref = load_cfl(out_file, DIMS, cim_dims2);
		assert(md_check_equal_dims(DIMS, cim_dims, cim_dims2, ~0));

		nlinvnet.weights = load_nn_weights(weight_file);

		eval_nlinvnet(&nlinvnet, DIMS, cim_dims, ref, ksp_dims, kspace);

		unmap_cfl(DIMS, cim_dims2, ref);
	}

	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, ksp_dims, kspace);
	md_free(trn_mask);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	exit(0);
}


