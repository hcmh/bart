#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/rand.h"

#include "iter/iter6.h"

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
#include "nn/data_list.h"

#include "grecon/opt_iter6.h"
#include "grecon/losses.h"
#include "grecon/network.h"

#include "linops/someops.h"

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
	const char* basis_file = NULL;

	struct noir2_conf_s conf = noir2_defaults;
	conf.cgiter = 30;


	struct nufft_conf_s nufft_conf = nufft_conf_defaults;

	nufft_conf.cache_psf_grdding = true;
	nufft_conf.lowmem = true;
	nufft_conf.precomp_fftmod = false;
	nufft_conf.precomp_roll = false;
	nufft_conf.precomp_linphase = false;

	conf.nufft_conf = &nufft_conf;
	struct nlinvnet_s nlinvnet = nlinvnet_config_opts;
	nlinvnet.conf = &conf;

	bool train = false;
	bool apply = false;
	bool eval = false;

	int num_gpus = 0;

	unsigned long batch_flags = BATCH_FLAG;
	long Nb = 0;

	char* filename_weights_load = NULL;

	const char* val_file_kspace = NULL;
	const char* val_file_reference = NULL;
	const char* val_file_pattern = NULL;
	const char* val_file_trajectory = NULL;

	const char* filename_mask = NULL;
	const char* filename_mask_val = NULL;

	opts_iter6_init();

	struct opt_s valid_opts[] = {

		OPTL_INFILE('p', "pattern", &(val_file_pattern), "<file>", "validation data sampling pattern"),
		OPTL_INFILE('t', "trajectory", &(val_file_trajectory), "<file>", "validation data trajectory"),
		OPTL_INFILE('k', "kspace", &(val_file_kspace), "<file>", "validation data kspace"),
		OPTL_INFILE('r', "ref", &(val_file_reference), "<file>", "validation data reference"),
		OPTL_INFILE(0, "mask", &(filename_mask_val), "<mask>", "mask for computation of loss"),
	};

	bool unet = false;
	long im_vec[3] = {0, 0, 0};

	struct opt_s network_opts[] = {

		OPTL_SET(0, "unet", &(unet), "use U-Net"),
	};

	_Bool norm_max = true;

	const struct opt_s opts[] = {

		OPTL_FLOAT(0, "lambda", &(nlinvnet.lambda), "lam", "additional regularization for network (negative means trainable)"),

		OPTL_CLEAR(0, "no-max-normalization", &norm_max, "don't normalize input of network by maximum"),
		OPTL_INT(0, "iter-net", &(nlinvnet.iter_net), "iter", "number of iterations with network"),

		OPTL_SUBOPT(0, "resnet-block", "...", "configure residual block", N_res_block_opts, res_block_opts),
		OPTL_SUBOPT(0, "unet", "...", "configure U-Net block", N_unet_reco_opts, unet_reco_opts),
		OPTL_CLEAR(0, "no-shared-weights", &(nlinvnet.share_weights), "don't share weights across iterations"),

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPTL_FLOAT(0, "redu", &conf.redu, "redu", "reduction of regularization"),
		OPTL_FLOAT(0, "coil-os", &(nlinvnet.oversampling_coils), "val", "(over-sampling factor for sensitivities)"),
		OPTL_SET(0, "fix-coils-sense", &(nlinvnet.fix_coils_sense), "fix coils in network steps"),
		OPTL_LONG(0, "sense-mean", &(nlinvnet.sense_mean), "l", "average coils over causal window of size l"),
		
		OPTL_SUBOPT('N', "network", "...", "select neural network", ARRAY_SIZE(network_opts), network_opts),
		OPTL_INT(0, "conv-time", &(nlinvnet.conv_time), "w", "convolve along dimension 10 with window size w"),
		OPTL_SELECT(0, "conv-time-causal", enum PADDING, &(nlinvnet.conv_padding), PAD_CAUSAL, "use causal convolution"),
		OPTL_INT(0, "cgiter", &(conf.cgiter), "", "number of cg iterations"),
		OPTL_SET(0, "init-rtnlinv", &(nlinvnet.real_time_init), "initialize with rtnlinv recon"),
		OPTL_SET(0, "reuse-init", &(nlinvnet.ref_init), "reuse initial recon as reference in network steps"),

		OPTL_VEC3('x', "dims", &im_vec, "x:y:z", "image dimensions"),

		OPT_FLOAT('w', &(nlinvnet.scaling), "val", "(normalization of data (must be negative))"),

		OPTL_SET('t', "train", &train, "train nlinvnet"),
		OPTL_SET('e', "eval", &eval, "evaluate nlinvnet"),
		OPTL_SET('a', "apply", &apply, "apply nlinvnet"),

		OPTL_INFILE(0, "trajectory", (const char**)(&(traj_file)), "<traj>", "trajectory"),
		OPTL_INFILE(0, "mask", &(filename_mask), "<mask>", "mask for computation of loss"),
		OPTL_INFILE('B', "basis", (const char**)(&(basis_file)), "<basis>", "(basis)"),

		OPTL_SET('g', "gpu", &(nlinvnet.gpu), "run on gpu"),
		OPTL_INT('G', "multi-gpu", &(num_gpus), "num", "run on num gpus (default: 1)"),
		OPTL_LONG('b', "batch-size", &(Nb), "", "size of mini batches"),

		OPTL_SET(0, "rss-norm", &(nlinvnet.normalize_rss), "scale output image to rss normalization"),

		OPTL_INFILE(0, "pattern", &pat_file, "<pattern>", "sampling pattern"),

		OPTL_SUBOPT(0, "valid-data", "...", "provide validation data", ARRAY_SIZE(valid_opts),valid_opts),

		OPTL_SUBOPT('T', "train-algo", "...", "configure general training parmeters", N_iter6_opts, iter6_opts),
		OPTL_SUBOPT(0, "adam", "...", "configure Adam", N_iter6_adam_opts, iter6_adam_opts),

		OPTL_INFILE('l', "load", (const char**)(&(filename_weights_load)), "<weights-init>", "load weights for continuing training"),

		OPTL_SUBOPT(0, "train-loss", "...", "configure the training loss", N_loss_opts, loss_opts),
		OPTL_FLOAT(0, "train-loss-l2-reco", &(nlinvnet.l2loss_reco), "a", "add a*(||x||^2 + ||Wc||^2) to train loss"),

		OPTL_FLOAT(0, "ss-ksp-split", &(nlinvnet.ksp_split), "p", "use p\% of kspace data as reference"),
		OPTL_ULONG(0, "ss-ksp-split-shared", &(nlinvnet.ksp_shared_dims), "flags", "shared dims for mask"),
		OPTL_VEC2(0, "ss-ksp-time-mask", &(nlinvnet.ksp_mask_time), "s:e", "don't use the first s and last e frames as train loss"),
		OPTL_FLOAT(0, "ss-ksp-split-exclude-center", &(nlinvnet.exclude_center), "p", "use the center part of spokes always for reco not for loss"),


		OPT_ULONG('L', &batch_flags, "flags", "loop over dims (apply only)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	#ifdef USE_CUDA
	if (0 != num_gpus)
		nlinvnet.gpu = true;

	if (nlinvnet.gpu) {

		num_init_multigpu(MAX(num_gpus, 1));
		cuda_use_global_memory();
	} else 
	#endif
		num_init();


	nlinvnet.ksp_training = (-1. != nlinvnet.ksp_split);

	if (train) {

		nlinvnet.train_conf = iter6_get_conf_from_opts();

		if (NULL == nlinvnet.train_conf) {

			debug_printf(DP_WARN, "No training algorithm selected. Fallback to Adam!\n");
			iter_6_select_algo = ITER6_ADAM;
			nlinvnet.train_conf = iter6_get_conf_from_opts();

		} else
			iter6_copy_config_from_opts(nlinvnet.train_conf);

		if ((0 < nlinvnet.train_conf->dump_mod) && (NULL == nlinvnet.train_conf->dump_filename))
			nlinvnet.train_conf->dump_filename = weight_file;
	}

	nlinvnet.network = get_default_network(unet ? NETWORK_UNET : NETWORK_RESBLOCK);
	
	if (norm_max)
		nlinvnet.network->norm = NORM_MAX;

	nlinvnet.network->loopdim = BATCH_DIM;

	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	bool pat_ones = true;
	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);
	} else {
		if (pat_ones) {

			md_select_dims(DIMS, ~(COIL_FLAG | BATCH_FLAG), pat_dims, ksp_dims);
			pattern = anon_cfl("", DIMS, pat_dims);
			md_zfill(DIMS, pat_dims, pattern, 1.);
		} else {

			md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
			pattern = anon_cfl("", DIMS, pat_dims);
			estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
		}
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

		if (0 == md_calc_size(3, im_vec)) {

			estimate_im_dims(DIMS, FFT_FLAGS, dims, trj_dims, traj);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);
		} else
			md_copy_dims(3, dims, im_vec);;

		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);
	} else {

		md_copy_dims(DIMS, dims, ksp_dims);
		md_singleton_dims(DIMS, trj_dims);
	}

	long bas_dims[DIMS];
	const complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, bas_dims);
		assert(!md_check_dimensions(DIMS, bas_dims, COEFF_FLAG | TE_FLAG));
	} else {

		md_singleton_dims(DIMS, bas_dims);
	}


	dims[MAPS_DIM] = 1;

	long sens_dims[DIMS];
	md_copy_dims(DIMS, sens_dims, dims);

	if (NULL != basis) {

		assert(1 == ksp_dims[COEFF_DIM]);
		assert(bas_dims[TE_DIM] == ksp_dims[TE_DIM]);

		dims[COEFF_DIM] = bas_dims[COEFF_DIM];
		dims[TE_DIM] = 1;
		md_select_dims(DIMS, ~(COEFF_FLAG | TE_FLAG), sens_dims, dims);
	}

	nlinvnet.scaling *= sqrtf(dims[TIME_DIM]);

	//FIXME: does it make sense?
	nlinvnet.scaling *= (md_calc_size(DIMS, dims) / md_calc_size(DIMS, sens_dims));

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

	if (train)
		assert(BATCH_FLAG == batch_flags);

	md_select_dims(DIMS, ~batch_flags, col_dims_s, sens_dims);
	md_select_dims(DIMS, ~batch_flags, img_dims_s, img_dims);
	md_select_dims(DIMS, ~batch_flags, cim_dims_s, cim_dims);
	md_select_dims(DIMS, ~batch_flags, msk_dims_s, msk_dims);
	md_select_dims(DIMS, ~batch_flags, ksp_dims_s, ksp_dims);
	md_select_dims(DIMS, ~batch_flags, pat_dims_s, pat_dims);
	md_select_dims(DIMS, ~batch_flags, trj_dims_s, trj_dims);

	Nb = Nb ? Nb : 10;
	Nb = MIN(Nb, ksp_dims[BATCH_DIM]);

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
			bas_dims, basis,
			msk_dims_s, NULL,
			ksp_dims_s,
			cim_dims_s,
			img_dims_s,
			col_dims_s);



	if (train) {

		if (NULL != filename_weights_load)
			nlinvnet.weights = load_nn_weights(filename_weights_load);


		long out_dims[DIMS];
		complex float* ref = load_cfl(out_file, DIMS, out_dims);
		assert(md_check_equal_dims(DIMS, nlinvnet.ksp_training ? ksp_dims : cim_dims, out_dims, ~0));

		auto train_data_list = named_data_list_create();
		named_data_list_append(train_data_list, DIMS, out_dims, ref, "ref");
		named_data_list_append(train_data_list, DIMS, ksp_dims, kspace, "ksp");
		named_data_list_append(train_data_list, DIMS, pat_dims, pattern, "pat");

		if (NULL != traj)
			named_data_list_append(train_data_list, DIMS, trj_dims, traj, "trj");

		complex float* mask = NULL;
		long mask_dims[DIMS];

		if (NULL != filename_mask) {

			mask = load_cfl(filename_mask, DIMS, mask_dims);
			nlinvnet.train_loss->mask_flags = md_nontriv_dims(DIMS, mask_dims);
			named_data_list_append(train_data_list, DIMS, mask_dims, mask, "loss_mask");
		}



		long ksp_dims_val[DIMS];
		long cim_dims_val[DIMS];
		long pat_dims_val[DIMS];
		long trj_dims_val[DIMS];
		long mask_dims_val[DIMS];

		complex float* val_kspace = NULL;
		complex float* val_ref = NULL;
		complex float* val_pattern = NULL;
		complex float* val_traj = NULL;
		complex float* mask_val = NULL;

		struct named_data_list_s* valid_data_list = NULL;

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


			valid_data_list = named_data_list_create();
			named_data_list_append(valid_data_list, DIMS, cim_dims_val, val_ref, "ref");
			named_data_list_append(valid_data_list, DIMS, ksp_dims_val, val_kspace, "ksp");
			named_data_list_append(valid_data_list, DIMS, pat_dims_val, val_pattern, "pat");

			if (NULL != val_traj)
				named_data_list_append(valid_data_list, DIMS, trj_dims_val, val_traj, "trj");

			if (NULL != filename_mask_val) {

				mask_val = load_cfl(filename_mask_val, DIMS, mask_dims_val);
				nlinvnet.valid_loss->mask_flags = md_nontriv_dims(DIMS, mask_dims_val);
				named_data_list_append(valid_data_list, DIMS, mask_dims_val, mask_val, "loss_mask");
			}
		}

		train_nlinvnet(&nlinvnet, Nb, train_data_list, valid_data_list);

		named_data_list_free(train_data_list);

		unmap_cfl(DIMS, out_dims, ref);

		dump_nn_weights(weight_file, nlinvnet.weights);

		if (NULL != val_file_kspace) {

			named_data_list_free(valid_data_list);

			unmap_cfl(DIMS, ksp_dims_val, val_kspace);
			unmap_cfl(DIMS, cim_dims_val, val_ref);
			unmap_cfl(DIMS, pat_dims_val, val_pattern);

			if (NULL != val_traj)
				unmap_cfl(DIMS, trj_dims_val, val_traj);
		}


		if (NULL != mask)
			unmap_cfl(DIMS, mask_dims, mask);
		if (NULL != mask_val)
			unmap_cfl(DIMS, mask_dims_val, mask_val);
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
				trj_dims, traj ? traj : &one);

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
				trj_dims, traj ? traj : &one);

		unmap_cfl(DIMS, cim_dims2, ref);
	}

	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, ksp_dims, kspace);

	if (NULL != basis)
		unmap_cfl(DIMS, bas_dims, basis);


	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	exit(0);
}


