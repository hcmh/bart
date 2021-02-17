/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/filter.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noncart/nufft.h"

#include "linops/linop.h"

#include "iter/iter2.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "moba/scale.h"
#include "moba/optreg.h"
#include "moba/recon_T1.h"
#include "moba/recon_T2.h"
#include "moba/recon_moba.h"
#include "moba/moba.h"
#include "moba/meco.h"
#include "moba/recon_meco.h"

#include "simu/signals.h"

static const char usage_str[] = "<kspace> <TI/TE> <output> [<sensitivities>]";
static const char help_str[] = "Model-based nonlinear inverse reconstruction\n";


static void edge_filter1(const long map_dims[DIMS], complex float* dst)
{
	float lambda = 2e-3;

	klaplace(DIMS, map_dims, map_dims, READ_FLAG|PHS1_FLAG, dst);
	md_zreal(DIMS, map_dims, dst, dst);
	md_zsqrt(DIMS, map_dims, dst, dst);

	md_zsmul(DIMS, map_dims, dst, dst, -2.);
	md_zsadd(DIMS, map_dims, dst, dst, 1.);
	md_zatanr(DIMS, map_dims, dst, dst);

	md_zsmul(DIMS, map_dims, dst, dst, -1. / M_PI);
	md_zsadd(DIMS, map_dims, dst, dst, 1.0);
	md_zsmul(DIMS, map_dims, dst, dst, lambda);
}

static void edge_filter2(const long map_dims[DIMS], complex float* dst)
{
	float beta = 100.;

	klaplace(DIMS, map_dims, map_dims, READ_FLAG|PHS1_FLAG, dst);
	md_zspow(DIMS, map_dims, dst, dst, 0.5);

	md_zsmul(DIMS, map_dims, dst, dst, -beta * 2.);
	md_zsadd(DIMS, map_dims, dst, dst, beta);

	md_zatanr(DIMS, map_dims, dst, dst);
	md_zsmul(DIMS, map_dims, dst, dst, -0.1 / M_PI);
	md_zsadd(DIMS, map_dims, dst, dst, 0.05);
}


int main_moba(int argc, char* argv[argc])
{
	double start_time = timestamp();

	float restrict_fov = -1.;
	float oversampling = 1.25f;

	float scale_fB0[2] = { 222., 1. }; // { spatial smoothness, scaling }

	unsigned int sample_size = 0;
	unsigned int grid_size = 0;
	unsigned int mgre_model = MECO_PI;

	const char* psf = NULL;
	const char* trajectory = NULL;
	const char* time_T1relax = NULL;
	const char* init_file = NULL;

	struct moba_conf_s conf_model;
	conf_model.model = IR;
	conf_model.irflash = irflash_conf_s_defaults;
	conf_model.sim = sim_conf_s_defaults;
	conf_model.opt = moba_defaults;

	struct opt_reg_s ropts;
	conf_model.opt.ropts = &ropts;

	bool out_origin_maps = false;
	bool out_sens = false;
	bool use_gpu = false;
	bool unused = false;

	enum mdb_t { DEFAULT, MDB_T1, MDB_T2, MDB_MGRE } mode = { DEFAULT };
	enum edge_filter_t { EF1, EF2 } k_filter_type = EF1;

	const char* input_alpha = NULL;

	long spokes_per_tr = 1;

	enum fat_spec fat_spec = FAT_SPEC_1;

	// FIXME: parser limits suboptions to OPT_SELECT and OPT_SET
	struct opt_s irflash_opt[] = {

		OPT_SELECT(	'L', enum moba_t, &conf_model.model, IR, "T1 mapping using model-based look-locker"),
		OPT_SELECT(	'M', enum moba_t, &conf_model.model, MOLLI, "use MOLLI model"),
		OPT_SELECT(	'S', enum moba_t, &conf_model.model, IR_SS, "use the IR steady-state model"),
		OPT_SELECT(	'P', enum moba_t, &conf_model.model, IR_phy, "select the (M0, R1, alpha) model"),
		OPT_SELECT(	'A', enum moba_t, &conf_model.model, IR_phy_alpha_in, "select the (M0, R1) model, input alpha needed"),
	};

	struct opt_s spin_echo_opt[] = {

		OPT_SELECT(	'F', enum moba_t, &conf_model.model, T2, "T2 mapping using model-based Fast Spin Echo"),
	};

	struct opt_s multi_gre_opt[] = {

		OPT_SELECT(	'M', enum moba_t, &conf_model.model, MGRE, "T2* mapping using model-based multiple gradient echo"),
		// FIXME: Integrate MGRE models from mgre_model here
	};

	opt_reg_init(&ropts);

	const struct opt_s opts[] = {

		{ 'r', NULL, true, opt_reg_moba, &ropts, " <T>:A:B:C\tgeneralized regularization options (-rh for help)" },

		// IR FLASH options
		OPTL_SUBOPT(0, "irflash" ,"interface", "IR FLASH options. `--irflash h` for help.", ARRAY_SIZE(irflash_opt), irflash_opt),
		OPT_STRING('T', &time_T1relax, "", "T1 relax time for MOLLI"),
		OPT_STRING('A',	&input_alpha, "", "Input alpha map required by (M0, R1) IR FLASH model "),

		// Spin-Echo options
		OPTL_SUBOPT(0, "spin-echo" ,"interface", "Spin-Echo options. `--spin-echo h` for help.", ARRAY_SIZE(spin_echo_opt), spin_echo_opt),

		// Multi GRE options
		OPTL_SUBOPT(0, "multi-gre" ,"interface", "Multi-GRE options. `--multi-gre h` for help.", ARRAY_SIZE(multi_gre_opt), multi_gre_opt),
		OPT_UINT('D', &mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),
		OPT_FLVEC2('b', &scale_fB0, "SMO:SC", "B0 field: spatial smooth level; scaling [default: 222.; 1.]"),
		OPTL_SELECT(0, "fat_spec_0", enum fat_spec, &fat_spec, FAT_SPEC_0, "select fat spectrum from ISMRM fat-water tool"),

		// Sequence parameter
		OPTL_FLOAT(0, "seq.TR", &(conf_model.sim.tr), "[s]", "repetition time in seconds"),

		// optimization options
		OPT_UINT('l', &conf_model.opt.opt_reg, "reg", "1/-l2\ttoggle l1-wavelet or l2 regularization."),
		OPT_UINT('i', &conf_model.opt.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf_model.opt.redu, "redu", "reduction factor"),
		OPT_FLOAT('j', &conf_model.opt.alpha_min, "minreg", "Minimum regu. parameter"),
		OPT_FLOAT('u', &conf_model.opt.rho, "rho", "ADMM rho [default: 0.01]"),
		OPT_UINT('C', &conf_model.opt.inner_iter, "iter", "inner iterations"),
		OPT_FLOAT('s', &conf_model.opt.step, "step", "step size"),
		OPT_FLOAT('B', &conf_model.opt.lower_bound, "bound", "lower bound for relaxivity"),
		OPT_SET('n', &conf_model.opt.auto_norm_off, "disable normlization of parameter maps for thresholding"),
		OPT_SET('J', &conf_model.opt.stack_frames, "Stack frames for joint recon"),
		OPT_SET('M', &conf_model.opt.sms, "Simultaneous Multi-Slice reconstruction"),

		// k-space filter
		OPT_SET('k', &conf_model.opt.k_filter, "k-space edge filter for non-Cartesian trajectories"),
		OPTL_SELECT(0, "kfilter-1", enum edge_filter_t, &k_filter_type, EF1, "k-space edge filter 1"),
		OPTL_SELECT(0, "kfilter-2", enum edge_filter_t, &k_filter_type, EF2, "k-space edge filter 2"),

		// others
		OPT_SET('g', &use_gpu, "use gpu"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_STRING('I', &init_file, "init", "File for initialization"),
		OPT_STRING('t', &trajectory, "Traj", ""),
		OPT_STRING('p', &psf, "PSF", ""),

		OPT_FLOAT('o', &oversampling, "os", "Oversampling factor for gridding [default: 1.25]"),
		OPTL_LONG(0, "spokes-per-TR", &(spokes_per_tr), "sptr", "number of averaged spokes [default: 1]"),

		// hidden options (kept for reproducibility, NOT RECOMMENDED to use!)
		OPT_SELECT('L', enum mdb_t, &mode, MDB_T1, "(T1 mapping using model-based look-locker)"),
		OPT_SET('m', &conf_model.opt.MOLLI, "(use MOLLI model)"),
		OPT_SET('S', &conf_model.opt.IR_SS, "(use the IR steady-state model)"),
		OPT_FLOAT('P', &conf_model.opt.IR_phy, "", "(select the (M0, R1, alpha) model and input TR)"),
		OPT_SELECT('F', enum mdb_t, &mode, MDB_T2, "(T2 mapping using model-based Fast Spin Echo)"),
		OPT_SELECT('G', enum mdb_t, &mode, MDB_MGRE, "(T2* mapping using model-based multiple gradient echo)"),
		OPT_SET('O', &out_origin_maps, "(Output original maps from reconstruction without post processing)"),
		OPT_SET('N', &unused, "(normalize)"), // no-op
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (5 == argc)
		out_sens = true;


	(use_gpu ? num_init_gpu_memopt : num_init)();
	
	conf_model.opt.algo = ALGO_FISTA;

	if (conf_model.opt.ropts->r > 0)
		conf_model.opt.algo = ALGO_ADMM;
		

#ifdef USE_CUDA
	cuda_use_global_memory();
#endif

	if (IR_phy_alpha_in == conf_model.model)
		assert(NULL != input_alpha);

	// Conversion of interfaces
	// FIXME: get rid of or simplify it...

	assert(!(conf_model.opt.MOLLI && conf_model.opt.IR_SS));

	if (DEFAULT != mode) {

		if (MDB_T1 == mode && conf_model.opt.MOLLI)
			conf_model.model = MOLLI;
		else if (MDB_T1 == mode && conf_model.opt.IR_SS)
			conf_model.model = IR_SS;
		else if (MDB_T1 == mode && 0. != conf_model.opt.IR_phy)
			conf_model.model = IR_phy;
		else if (MDB_T1 == mode && NULL != input_alpha)
			conf_model.model = IR_phy_alpha_in;
		else if (MDB_T2 == mode)
			conf_model.model = T2;
		else if (MDB_MGRE == mode)
			conf_model.model = MGRE;
		// else if (NULL != conf->input_alpha)
		// 	conf_model.model = Bloch;

		conf_model.sim.tr = 1e-6 * conf_model.opt.IR_phy; // [us] -> [s]
	}

	if (conf_model.opt.ropts->r > 0)
		conf_model.opt.algo = ALGO_ADMM;

	long ksp_dims[DIMS];
	complex float* kspace_data = load_cfl(argv[1], DIMS, ksp_dims);

	// Pass inversion time

	long TI_dims[DIMS];
	complex float* TI = load_cfl(argv[2], DIMS, TI_dims);

	// FIXME: Way to perform load_cfl directly on pointer in struct?!
	conf_model.irflash.input_TI = md_alloc(DIMS, TI_dims, CFL_SIZE);
	md_copy(DIMS, TI_dims, conf_model.irflash.input_TI, TI, CFL_SIZE);

	assert(TI_dims[TE_DIM] == ksp_dims[TE_DIM]);
	assert(1 == ksp_dims[MAPS_DIM]);

	// Relaxation time for MOLLI

	complex float* TI_t1relax = NULL;
	long TI_t1relax_dims[DIMS];

	if (MOLLI == conf_model.model) {
		
		assert(NULL != time_T1relax);
		TI_t1relax = load_cfl(time_T1relax, DIMS, TI_t1relax_dims);

		conf_model.irflash.input_TI_t1relax = md_alloc(DIMS, TI_t1relax_dims, CFL_SIZE);
		md_copy(DIMS, TI_t1relax_dims, conf_model.irflash.input_TI_t1relax, TI_t1relax, CFL_SIZE);
	}

	// Load passed alpha map

	complex float* alpha = NULL;

	long input_alpha_dims[DIMS];

	if (NULL != input_alpha) {

		debug_printf(DP_DEBUG2, "Load FA map [deg]\n");

		alpha = load_cfl(input_alpha, DIMS, input_alpha_dims);

		conf_model.irflash.input_alpha = md_alloc(DIMS, input_alpha_dims, CFL_SIZE);

		// ksp_dims[PHS2_DIM] needs to be multiple of spokes_per_tr
		assert((ksp_dims[PHS2_DIM]/spokes_per_tr)*spokes_per_tr == ksp_dims[PHS2_DIM]);

		fa_to_alpha(DIMS, input_alpha_dims, conf_model.irflash.input_alpha, alpha,
				get_tr_from_inversion(DIMS, TI_dims, conf_model.irflash.input_TI, ksp_dims[PHS2_DIM]/spokes_per_tr));

		unmap_cfl(DIMS, input_alpha_dims, alpha);
	}

	if (conf_model.opt.sms) {

		debug_printf(DP_INFO, "SMS Model-based reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace_data, kspace_data); // fftmod to get correct slice order in output
	}

	long grid_dims[DIMS];
	md_copy_dims(DIMS, grid_dims, ksp_dims);

	// Oversample passed trajectory

	if (NULL != trajectory) {

		sample_size = ksp_dims[1];
		grid_size = sample_size * oversampling;
		grid_dims[READ_DIM] = grid_size;
		grid_dims[PHS1_DIM] = grid_size;
		grid_dims[PHS2_DIM] = 1L;
				
		if (-1 == restrict_fov)
			restrict_fov = 0.5;

		conf_model.opt.noncartesian = true;
	}

	long img_dims[DIMS];

	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, img_dims, grid_dims);

	// Select number of estimated parameters depending on model
	// FIXME: Reorder models to allow for fall through cases?
	switch (conf_model.model) {

	case IR:
	case MOLLI:
	case IR_phy:
		img_dims[COEFF_DIM] = 3;
		break;

	case IR_SS:
	case IR_phy_alpha_in:
	case T2:
		img_dims[COEFF_DIM] = 2;
		break;

	case MGRE:
		img_dims[COEFF_DIM] = (MECO_PI != mgre_model) ? set_num_of_coeff(mgre_model) : grid_dims[TE_DIM];
		break;

	case Bloch:
		error("Bloch model not supported yet.");
		break;
	}


	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	long single_map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, single_map_dims, grid_dims);

	long single_map_strs[DIMS];
	md_calc_strides(DIMS, single_map_strs, single_map_dims, CFL_SIZE);

	long coil_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, coil_dims, grid_dims);

	long coil_strs[DIMS];
	md_calc_strides(DIMS, coil_strs, coil_dims, CFL_SIZE);

	complex float* img = create_cfl(argv[3], DIMS, img_dims);
	complex float* single_map = anon_cfl("", DIMS, single_map_dims);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, grid_dims);

	long dims[DIMS];
	md_copy_dims(DIMS, dims, grid_dims);

	dims[COEFF_DIM] = img_dims[COEFF_DIM];

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = NULL;
	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[4] : "", DIMS, coil_dims);


	md_zfill(DIMS, img_dims, img, 1.0);
	md_clear(DIMS, coil_dims, sens, CFL_SIZE);

	complex float* k_grid_data = NULL;
	k_grid_data = anon_cfl("", DIMS, grid_dims);

	complex float* pattern = NULL;
	long pat_dims[DIMS];
	

	if (NULL != psf) {

		complex float* tmp_psf =load_cfl(psf, DIMS, pat_dims);
		pattern = anon_cfl("", DIMS, pat_dims);

		md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);
		unmap_cfl(DIMS, pat_dims, tmp_psf);

		md_copy(DIMS, grid_dims, k_grid_data, kspace_data, CFL_SIZE);
		unmap_cfl(DIMS, ksp_dims, kspace_data);

		if (0 == md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims))
			error("pattern not compatible with kspace dimensions\n");

		if (-1 == restrict_fov)
			restrict_fov = 0.5;

		conf_model.opt.noncartesian = true;

	} else if (NULL != trajectory) {

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = false;

		struct linop_s* nufft_op_k = NULL;

		long traj_dims[DIMS];
		long traj_strs[DIMS];

		complex float* traj = load_cfl(trajectory, DIMS, traj_dims);
		md_calc_strides(DIMS, traj_strs, traj_dims, CFL_SIZE);

		md_zsmul(DIMS, traj_dims, traj, traj, oversampling);

		md_select_dims(DIMS, FFT_FLAGS|TE_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, pat_dims, grid_dims);
		pattern = anon_cfl("", DIMS, pat_dims);

		// Gridding sampling pattern
		
		complex float* psf = NULL;

		long wgh_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, wgh_dims, ksp_dims);

		complex float* wgh = md_alloc(DIMS, wgh_dims, CFL_SIZE);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, wgh, kspace_data);

		psf = compute_psf(DIMS, pat_dims, traj_dims, traj, traj_dims, NULL, wgh_dims, wgh, false, false);

		// TODO: would the "pattern" here have memory issue?
		fftuc(DIMS, pat_dims, FFT_FLAGS, pattern, psf);

		md_free(wgh);
		md_free(psf);

		// Gridding raw data

		nufft_op_k = nufft_create(DIMS, ksp_dims, grid_dims, traj_dims, traj, NULL, nufft_conf);
		linop_adjoint(nufft_op_k, DIMS, grid_dims, k_grid_data, DIMS, ksp_dims, kspace_data);
		fftuc(DIMS, grid_dims, FFT_FLAGS, k_grid_data, k_grid_data);

		linop_free(nufft_op_k);

		unmap_cfl(DIMS, ksp_dims, kspace_data);

	} else {

		md_copy_dims(DIMS, pat_dims, img_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
		md_copy(DIMS, grid_dims, k_grid_data, kspace_data, CFL_SIZE);
		unmap_cfl(DIMS, ksp_dims, kspace_data);
	}


	if (conf_model.opt.k_filter) {

		long map_dims[DIMS];
		md_select_dims(DIMS, FFT_FLAGS, map_dims, pat_dims);

		long map_strs[DIMS];
		md_calc_strides(DIMS, map_strs, map_dims, CFL_SIZE);

		long pat_strs[DIMS];
		md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

		complex float* filter = md_alloc(DIMS, map_dims, CFL_SIZE);

		switch (k_filter_type) {

		case EF1:
			edge_filter1(map_dims, filter);
			break;

		case EF2:
			edge_filter2(map_dims, filter);
			break;
		}

		md_zadd2(DIMS, pat_dims, pat_strs, pattern, pat_strs, pattern, map_strs, filter);

		md_free(filter);
	}

	// read initialization file

	long init_dims[DIMS] = { [0 ... DIMS-1] = 1 };
	complex float* init = (NULL != init_file) ? load_cfl(init_file, DIMS, init_dims) : NULL;

	assert(md_check_bounds(DIMS, 0, img_dims, init_dims));

	// scaling

	if (	(IR == conf_model.model) ||
		(MOLLI == conf_model.model) ||
		(IR_SS == conf_model.model) ||
		(IR_phy == conf_model.model) ||
		(IR_phy_alpha_in == conf_model.model) ||
		(T2 == conf_model.model) ) {

		double scaling = ((ALGO_ADMM == conf_model.opt.algo) ? 250. : 5000.) / md_znorm(DIMS, grid_dims, k_grid_data);
		double scaling_psf = ((ALGO_ADMM == conf_model.opt.algo) ? 500. : 1000.) / md_znorm(DIMS, pat_dims, pattern);

		if (conf_model.opt.sms) {

			scaling *= grid_dims[SLICE_DIM] / 5.0;
			scaling_psf *= grid_dims[SLICE_DIM] / 5.0;
		}

		debug_printf(DP_INFO, "Scaling: %f\n", scaling);
		md_zsmul(DIMS, grid_dims, k_grid_data, k_grid_data, scaling);

		debug_printf(DP_INFO, "Scaling_psf: %f\n", scaling_psf);
		md_zsmul(DIMS, pat_dims, pattern, pattern, scaling_psf);
	}

	// mask

	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, msk_dims, CFL_SIZE);
		md_zfill(DIMS, msk_dims, mask, 1.);

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		mask = compute_mask(DIMS, msk_dims, restrict_dims);
		md_zmul2(DIMS, img_dims, img_strs, img, img_strs, img, msk_strs, mask);

		if (	(IR == conf_model.model) ||
			(MOLLI == conf_model.model) ||
			(IR_SS == conf_model.model) ||
			(IR_phy == conf_model.model) ||
			(IR_phy_alpha_in == conf_model.model) ||
			(T2 == conf_model.model) ) {

			// Choose a different initial guess for R1*
			float init_param = (IR_phy == conf_model.model || (IR_phy_alpha_in == conf_model.model)) ? 3. : (conf_model.opt.sms ? 2. : 1.5);

			long pos[DIMS] = { 0 };

			pos[COEFF_DIM] = ((IR_SS == conf_model.model) || (IR_phy == conf_model.model) || (IR_phy_alpha_in == conf_model.model) || (T2 == conf_model.model)) ? 1 : 2;

			md_copy_block(DIMS, pos, single_map_dims, single_map, img_dims, img, CFL_SIZE);
			md_zsmul2(DIMS, single_map_dims, single_map_strs, single_map, single_map_strs, single_map, init_param);
			md_copy_block(DIMS, pos, img_dims, img, single_map_dims, single_map, CFL_SIZE);

			// Choose initial guess for alpha
			if (IR_phy == conf_model.model) {

				pos[COEFF_DIM] = 2;
				md_zfill(DIMS, single_map_dims, single_map, 0.);
				md_copy_block(DIMS, pos, img_dims, img, single_map_dims, single_map, CFL_SIZE);
			}
		}
	}

	unmap_cfl(DIMS, single_map_dims, single_map);

#ifdef  USE_CUDA
	if (use_gpu) {

		cuda_use_global_memory();

		complex float* kspace_gpu = md_alloc_gpu(DIMS, grid_dims, CFL_SIZE);

		md_copy(DIMS, grid_dims, kspace_gpu, k_grid_data, CFL_SIZE);

		if (MGRE == conf_model.model)
			meco_recon(&conf_model.opt, mgre_model, false, fat_spec, scale_fB0, true, out_origin_maps, img_dims, img, coil_dims, sens, init_dims, init, mask, conf_model.irflash.input_TI, pat_dims, pattern, grid_dims, kspace_gpu);
		else
			moba_recon(&conf_model, dims, img, sens, pattern, mask, kspace_gpu, use_gpu);

		md_free(kspace_gpu);

	} else
#endif
	if (MGRE == conf_model.model)
		meco_recon(&conf_model.opt, mgre_model, false, fat_spec, scale_fB0, true, out_origin_maps, img_dims, img, coil_dims, sens, init_dims, init, mask, conf_model.irflash.input_TI, pat_dims, pattern, grid_dims, k_grid_data);
	else
		moba_recon(&conf_model, dims, img, sens, pattern, mask, k_grid_data, use_gpu);

	md_free(mask);

	unmap_cfl(DIMS, coil_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, grid_dims, k_grid_data);
	unmap_cfl(DIMS, img_dims, img);

	unmap_cfl(DIMS, TI_dims, TI);
	md_free(conf_model.irflash.input_TI);

	if (MOLLI == conf_model.model) {

		unmap_cfl(DIMS, TI_t1relax_dims, TI_t1relax);
		md_free(conf_model.irflash.input_TI_t1relax);
	}

	if (NULL != init_file)
		unmap_cfl(DIMS, init_dims, init);

	if (NULL != input_alpha)
		md_free(conf_model.irflash.input_alpha);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);

	exit(0);
}

