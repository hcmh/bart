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
#include "misc/io.h"

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
	const char* ref_img_file = NULL;
	const char* ref_col_file = NULL;

	struct noir2_conf_s conf = noir2_defaults;
	conf.cgiter = 30;

	conf.nufft_conf = &nufft_conf_defaults;
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
	const char* val_file_pattern = NULL;
	const char* val_file_trajectory = NULL;
	const char* val_file_ref_img = NULL;
	const char* val_file_ref_col = NULL;

	unsigned long cnstcoil_flags = 0;

	opts_iter6_init();

	struct opt_s valid_opts[] = {

		OPTL_INFILE('p', "pattern", &(val_file_pattern), "<file>", "validation data sampling pattern"),
		OPTL_INFILE('t', "trajectory", &(val_file_trajectory), "<file>", "validation data trajectory"),
		OPTL_INFILE('k', "kspace", &(val_file_kspace), "<file>", "validation data kspace"),
		OPTL_INFILE('r', "ref", &(val_file_reference), "<file>", "validation data reference"),
		OPTL_INFILE(0, "ref-img", (const char**)(&(val_file_ref_img)), "<ref>", "reference file for image"),
		OPTL_INFILE(0, "ref-col", (const char**)(&(val_file_ref_col)), "<ref>", "reference file for sensitivities"),
	};

	bool unet = false;
	long im_vec[3] = {0, 0, 0};

	struct opt_s network_opts[] = {

		OPTL_SET(0, "unet", &(unet), "use U-Net (also sets train and data-consistency default values)"),
	};

	const struct opt_s opts_net[] = {

		OPTL_SET(0, "fix-lambda", &(nlinvnet.fix_lambda), "fix lambda"),

		OPTL_INT(0, "iter-net", &(nlinvnet.iter_net), "iter", "number of iterations with network"),
		OPTL_INT(0, "iter-init", &(nlinvnet.iter_init), "iter", "number of iterations with half update"),

		OPTL_SUBOPT(0, "resnet-block", "...", "configure residual block", N_res_block_opts, res_block_opts),
		OPTL_SUBOPT(0, "unet", "...", "configure U-Net block", N_unet_reco_opts, unet_reco_opts),
		OPTL_CLEAR(0, "no-shared-weights", &(nlinvnet.share_weights), "don't share weights across iterations"),

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		//OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		//OPTL_FLOAT(0, "sobolev-a", &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		//OPTL_FLOAT(0, "sobolev-b", &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		//OPTL_FLOAT(0, "alpha", &conf.alpha, "", "(start regularization)"),
		//OPTL_FLOAT(0, "alpha-min", &conf.alpha_min, "", "(minimum for regularization)"),
		OPTL_FLOAT(0, "coil-os", &coil_os, "val", "(over-sampling factor for sensitivities)"),

		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),
		OPTL_INT(0, "cgiter", &(conf.cgiter), "", "number of cg iterations"),

		OPTL_ULONG(0, "const-coils", &cnstcoil_flags, "", "(dimensions with constant sensitivities)"),
		OPTL_VEC3(0, "dims", &im_vec, "x:y:z", "image dimensions"),

		OPTL_SET(0, "ref-net-only", &(nlinvnet.ksp_ref_net_only), "(Use reference only in network)"),
	};

	const struct opt_s opts_trn[] = {

		OPTL_SET('t', "train", &train, "train nlinvnet"),
		OPTL_SET('e', "eval", &eval, "evaluate nlinvnet"),
		OPTL_SET('a', "apply", &apply, "apply nlinvnet"),

		OPTL_INFILE(0, "trajectory", (const char**)(&(traj_file)), "<traj>", "trajectory"),
		OPTL_INFILE(0, "ref-img", (const char**)(&(ref_img_file)), "<ref>", "reference file for image"),
		OPTL_INFILE(0, "ref-col", (const char**)(&(ref_col_file)), "<ref>", "reference file for sensitivities"),

		OPTL_SET('g', "gpu", &(nlinvnet.gpu), "run on gpu"),
		OPTL_LONG('b', "batch-size", &(Nb), "", "size of mini batches"),

		OPTL_SET(0, "rss-norm", &(nlinvnet.normalize_rss), "scale output image to rss normalization"),

		OPTL_INFILE(0, "pattern", &pat_file, "<pattern>", "sampling pattern"),

		OPTL_SUBOPT(0, "valid-data", "...", "provide validation data", ARRAY_SIZE(valid_opts),valid_opts),

		OPTL_SUBOPT('T', "train-algo", "...", "configure general training parmeters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure Adam", N_iter6_adam_opts, iter6_adam_opts),

		OPTL_INFILE('l', "load", (const char**)(&(filename_weights_load)), "<weights-init>", "load weights for continuing training"),

		OPTL_SUBOPT(0, "train-loss", "...", "configure the training loss", N_loss_opts, loss_opts),

		OPTL_FLOAT(0, "ss-ksp-split", &(nlinvnet.ksp_split), "p", "use p\% of kspace data as reference"),
		OPTL_ULONG(0, "ss-ksp-split-shared", &(nlinvnet.ksp_shared_dims), "flags", "shared dims for mask"),
		OPTL_FLOAT(0, "ss-ksp-noise", &(nlinvnet.ksp_noise), "var", "Add noise to input kspace. Negative variance will draw variance of noise from gaussian distribution."),
		OPTL_FLOAT(0, "ss-l1-norm", &(nlinvnet.l1_norm), "", "Add l1 norm of coil image to loss"),
		OPTL_FLOAT(0, "ss-l2-norm", &(nlinvnet.l2_norm), "", "Add l2 norm of coil image to loss"),
	};

	struct opt_s opts[ARRAY_SIZE(opts_net) + ARRAY_SIZE(opts_trn)];

	for (int i = 0; (unsigned int)i < ARRAY_SIZE(opts_trn); i++)
		opts[i] = opts_trn[i];
	for (int i = 0; (unsigned int)i < ARRAY_SIZE(opts_net); i++)
		opts[i + ARRAY_SIZE(opts_trn)] = opts_net[i];

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (apply || eval) {

		int nargs2 = 100;
		char* args2[nargs2];
		nargs2 = read_command(weight_file, nargs2, args2);

		int nargs3 = nargs2;

		options_only(&nargs3, args2, ARRAY_SIZE(opts_net), opts_net, ARRAY_SIZE(opts_trn), opts_trn);

		for (int i = 0; i < nargs2; i++)
			xfree(args2[i]);
	}

	(nlinvnet.gpu ? num_init_gpu_memopt : num_init)();
	#ifdef USE_CUDA
	if (nlinvnet.gpu)
		cuda_use_global_memory();
	#endif
	reuse_nufft_for_psf();

	nlinvnet.ksp_training = (0. != nlinvnet.ksp_noise) || (-1. != nlinvnet.ksp_split);

	if (train) {

		nlinvnet.train_conf = iter6_get_conf_from_opts();

		if (NULL == nlinvnet.train_conf) {

			debug_printf(DP_WARN, "No training algorithm selected. Fallback to Adam!\n");
			iter_6_select_algo = ITER6_ADAM;
			nlinvnet.train_conf = iter6_get_conf_from_opts();

		} else
			iter6_copy_config_from_opts(nlinvnet.train_conf);
	}

	nlinvnet.network = get_default_network(unet ? NETWORK_UNET_RECO : NETWORK_RESBLOCK);
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
		if (0 == md_calc_size(3, im_vec))
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);
		else
			md_copy_dims(3, dims, im_vec);;

		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);
	} else {

		md_copy_dims(DIMS, dims, ksp_dims);
		md_singleton_dims(DIMS, trj_dims);
	}

	if (-1 == coil_os)
		coil_os = conf.noncart ? 2 : 1;

	dims[MAPS_DIM] = 1;

	long sens_dims[DIMS];
	md_select_dims(DIMS, ~cnstcoil_flags, sens_dims, dims);

	long const_sens_dims[DIMS];
	md_select_dims(DIMS, cnstcoil_flags, const_sens_dims, dims);
	nlinvnet.scaling *= md_calc_size(DIMS, const_sens_dims);

	for (int i = 0; i < 3; i++)
		if (1 != sens_dims[i])
			sens_dims[i] = lround(coil_os * sens_dims[i]);

	long img_dims[DIMS];
	long cim_dims[DIMS];
	long msk_dims[DIMS];

	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

	long col_dims_s[DIMS];
	long img_dims_s[DIMS];
	long cim_dims_s[DIMS];
	long msk_dims_s[DIMS];
	long ksp_dims_s[DIMS];
	long pat_dims_s[DIMS];
	long trj_dims_s[DIMS];

	md_select_dims(DIMS, ~BATCH_FLAG, col_dims_s, sens_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, img_dims_s, img_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, cim_dims_s, cim_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, msk_dims_s, msk_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, ksp_dims_s, ksp_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, pat_dims_s, pat_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, trj_dims_s, trj_dims);

	Nb = Nb ? Nb : 10;
	nlinvnet.Nb = Nb;

	complex float one = 1.;

	if (NULL == traj)
		nlinvnet_init_model_cart(&nlinvnet, DIMS,
			pat_dims_s,
			MD_SINGLETON_DIMS(DIMS), NULL,
			msk_dims_s, NULL,
			ksp_dims_s,
			cim_dims_s,
			img_dims_s,
			col_dims_s);
	else
		nlinvnet_init_model_noncart(&nlinvnet, DIMS,
			trj_dims_s,
			pat_dims_s,
			MD_SINGLETON_DIMS(DIMS), NULL,
			msk_dims_s, NULL,
			ksp_dims_s,
			cim_dims_s,
			img_dims_s,
			col_dims_s);

	long ref_img_dims[DIMS];
	long ref_col_dims[DIMS];
	complex float* ref_img = NULL;
	complex float* ref_col = NULL;

	if (NULL != ref_img_file) {

		nlinvnet.ref = true;
		ref_img = load_cfl(ref_img_file, DIMS, ref_img_dims);
	} else
		md_copy_dims(DIMS, ref_img_dims, img_dims_s);

	if (NULL != ref_col_file)
		ref_col = load_cfl(ref_col_file, DIMS, ref_col_dims);
	else
		md_copy_dims(DIMS, ref_col_dims, col_dims_s);




	if (train) {

		if (NULL != filename_weights_load)
			nlinvnet.weights = load_nn_weights(filename_weights_load);

		long ksp_dims_val[DIMS];
		long cim_dims_val[DIMS];
		long pat_dims_val[DIMS];
		long trj_dims_val[DIMS];
		long ref_img_dims_val[DIMS];
		long ref_col_dims_val[DIMS];

		complex float* val_kspace = NULL;
		complex float* val_ref = NULL;
		complex float* val_pattern = NULL;
		complex float* val_traj = NULL;
		complex float* val_ref_img = NULL;
		complex float* val_ref_col = NULL;

		if (NULL != val_file_kspace) {

			val_kspace = load_cfl(val_file_kspace, DIMS, ksp_dims_val);
			val_ref = load_cfl(val_file_reference, DIMS, cim_dims_val);

			if (NULL != val_file_pattern) {

				val_pattern = load_cfl(val_file_pattern, DIMS, pat_dims_val);
			} else {

				md_select_dims(DIMS, ~COIL_FLAG, pat_dims_val, ksp_dims_val);
				val_pattern = anon_cfl("", DIMS, pat_dims_val);
				estimate_pattern(DIMS, ksp_dims_val, COIL_FLAG, val_pattern, val_kspace);
			}

			if (NULL != val_file_trajectory)
				val_traj = load_cfl(val_file_trajectory, DIMS, trj_dims_val);
			else
				md_singleton_dims(DIMS, trj_dims_val);

			if (NULL != val_file_ref_img)
				val_ref_img = load_cfl(val_file_ref_img, DIMS, ref_img_dims_val);
			else
				md_copy_dims(DIMS, ref_img_dims_val, img_dims_s);

			if (NULL != val_file_ref_col)
				val_ref_col = load_cfl(val_file_ref_img, DIMS, ref_col_dims_val);
			else
				md_copy_dims(DIMS, ref_col_dims_val, col_dims_s);
		}

		long out_dims[DIMS];
		complex float* ref = load_cfl(out_file, DIMS, out_dims);

		if (nlinvnet.ksp_training)
			assert(md_check_equal_dims(DIMS, ksp_dims, out_dims, ~0));
		else
			assert(md_check_equal_dims(DIMS, cim_dims, out_dims, ~0));

		if ((-1. != nlinvnet.ksp_split) && (NULL != ref_img))
			md_zsmul(DIMS, ref_img_dims, ref_img, ref_img, 1. / nlinvnet.ksp_split);

		train_nlinvnet(&nlinvnet, DIMS, Nb,
			out_dims, ref,
			ksp_dims, kspace,
			pat_dims, pattern,
			trj_dims, traj ? traj : &one,
			ref_img_dims, ref_img,
			ref_col_dims, ref_col,
			cim_dims_val, val_ref,
			ksp_dims_val, val_kspace,
			pat_dims_val, val_pattern,
			trj_dims_val, val_traj? val_traj : &one,
			ref_img_dims_val, val_ref_img,
			ref_col_dims_val, val_ref_col);

		unmap_cfl(DIMS, out_dims, ref);

		dump_nn_weights(weight_file, nlinvnet.weights);

		if (NULL != val_file_kspace) {

			unmap_cfl(DIMS, ksp_dims_val, val_kspace);
			unmap_cfl(DIMS, cim_dims_val, val_ref);
			unmap_cfl(DIMS, pat_dims_val, val_pattern);

			if (NULL != val_traj)
				unmap_cfl(DIMS, trj_dims_val, val_traj);
			if (NULL != val_ref_img)
				unmap_cfl(DIMS, ref_img_dims_val, val_ref_img);
			if (NULL != val_ref_col)
				unmap_cfl(DIMS, ref_col_dims_val, val_ref_col);
		}
	}

	if (apply) {

		complex float* img = create_cfl(out_file, DIMS, img_dims);

		md_copy_dims(3, sens_dims, img_dims);

		complex float* col = (NULL != sens_file) ? create_cfl(sens_file, DIMS, sens_dims) : anon_cfl("", DIMS, sens_dims);
		nlinvnet.weights = load_nn_weights(weight_file);

		if (-1. != nlinvnet.ksp_split) {

			long sdims[DIMS];
			md_select_dims(DIMS, ~nlinvnet.ksp_shared_dims, sdims, pat_dims);
			complex float* tmp = md_alloc(DIMS, pat_dims, CFL_SIZE);

			md_rand_one(DIMS, sdims, tmp, nlinvnet.ksp_split);

			md_zmul2(DIMS, pat_dims, MD_STRIDES(DIMS, pat_dims, CFL_SIZE), pattern, MD_STRIDES(DIMS, pat_dims, CFL_SIZE), pattern, MD_STRIDES(DIMS, sdims, CFL_SIZE), tmp);

			md_free(tmp);
		}

		apply_nlinvnet(&nlinvnet, DIMS,
				img_dims, img,
				sens_dims, col,
				ksp_dims, kspace,
				pat_dims, pattern,
				trj_dims, traj ? traj : &one,
				ref_img_dims, ref_img,
				ref_col_dims, ref_col);

		unmap_cfl(DIMS, img_dims, img);
		unmap_cfl(DIMS, sens_dims, col);
	}

	if (eval) {

		long cim_dims2[DIMS];
		complex float* ref = load_cfl(out_file, DIMS, cim_dims2);
		assert(md_check_equal_dims(DIMS, cim_dims, cim_dims2, ~0));

		nlinvnet.weights = load_nn_weights(weight_file);

		eval_nlinvnet(&nlinvnet, DIMS,
				cim_dims, ref,
				ksp_dims, kspace,
				pat_dims, pattern,
				trj_dims, traj ? traj : &one,
				ref_img_dims, ref_img,
				ref_col_dims, ref_col);

		unmap_cfl(DIMS, cim_dims2, ref);
	}

	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, ksp_dims, kspace);

	if (NULL != ref_img)
		unmap_cfl(DIMS, ref_img_dims, ref_img);
	if (NULL != ref_col)
		unmap_cfl(DIMS, ref_col_dims, ref_col);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	nufft_psf_del();


	exit(0);
}


