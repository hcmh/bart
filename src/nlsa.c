/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 */

/* Paper
 * -----------
 * NLSA
 * Giannakis, D., & Majda, A. J. (2012).
 * Nonlinear Laplacian spectral analysis for time series with intermittency and low-frequency variability.
 * Proceedings of the National Academy of Sciences, 109(7), 2222-2227.
 *
 * Giannakis, D., & Majda, A. J. (2013).
 * Nonlinear Laplacian spectral analysis: capturing intermittent and low‐frequency spatiotemporal patterns in high‐dimensional data.
 * Statistical Analysis and Data Mining: The ASA Data Science Journal, 6(3), 180-194.
 * 
 * SSA-FARY
 * Rosenzweig, S., Scholand, N., Holme, H. C. M., & Uecker, M. (2018).
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular Spectrum Analysis (SSA-FARY).
 * arXiv preprint arXiv:1812.09057.
 *
 * Comments on NLSA:
 * The NLSA is not exactly implemented as proposed by Giannakis & Maida:
 * 	We don't approximate the Laplacian eigenvectors but calculate them explicitly
 * 
 * 'nlsa_rank' defines the smoothness of the manifold (temporal evolution)
 * 
 * 
 * General comments:
 * The rank option '-r' allows to "throw away" basis functions:
 *	rank < 0: throw away 'rank' basis functions with high singular values
 *	rank > 0: keep only 'rank' basis functions with the highest singular value
 * 
 * The group option '-g' implements what is called 'Grouping' in SSA literature, by selecting EOFs with a bitmask.
 *  group < 0: do not use the selected group for backprojection, but all other EOFs (= filtering)
 *  group > 0: use only the selected group for backprojection
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "calib/calmat.h"
#include "calib/ssa.h"


static const char help_str[] = "Perform NLSA-FARY or Nonlinear Laplacian Spectral Analysis. <src>: [samples, coordinates]";


int main_nlsa(int argc, char* argv[argc])
{
	const char* src_file = NULL;
	struct delay_conf nlsa_conf = nlsa_conf_default;


	struct arg_s args[] = {

		ARG_INFILE(true, &src_file, "src"),
		ARG_OUTFILE(true, (const char **) &nlsa_conf.name_tbasis, "EOF"),
		ARG_OUTFILE(false, (const char **) &nlsa_conf.name_S, "S"),
		ARG_OUTFILE(false, (const char **) &nlsa_conf.backproj, "backprojection"),

	};

	struct laplace_conf conf = laplace_conf_default;
	conf.dmap = true;



	const struct opt_s opts[] = {

		OPT_INT('w', &nlsa_conf.window, "window", "Window length"),
		OPT_CLEAR('z', &nlsa_conf.zeropad, "Zeropadding [Default: True]"),
		OPT_INT('m', &nlsa_conf.rm_mean, "0/1", "Remove mean [Default: True]"),
		OPT_INT('n', &nlsa_conf.normalize, "0/1", "Normalize [Default: False]"),
		OPT_INT('r', &nlsa_conf.rank, "rank", "Rank for backprojection. r < 0: Throw away first r components. r > 0: Use only first r components"),
		OPT_LONG('g', &nlsa_conf.group, "bitmask", "Bitmask for Grouping (long value!)"),
		OPT_LONG('L', &nlsa_conf.nlsa_rank, "NLSA", "Rank for Nonlinear Laplacian Spectral Analysis"),
		OPT_INT('N', &conf.nn, "nn", "Number of nearest neighbours"),
		OPT_FLOAT('s', &conf.sigma, "sigma", "Standard deviation"),
		OPT_SET('k', &conf.kernel, "Kernel approach"),
		OPT_FLOAT('G', &conf.kernel_gamma, "gamma", "(Kernel gamma minimum)"),
		OPT_FLOAT('l', &conf.kernel_lambda, "lambda", "(Kernel lambda)"),	
		OPT_INT('i', &conf.iter_max, "iter", "[Kernel] Number of kernel iterations"),
		OPT_FLOAT('e', &nlsa_conf.weight, "exp", "Soft delay-embedding"),
		OPT_SET('T', &nlsa_conf.temporal_nn, "Temporal nearest neighbour"),
		OPT_SET('o', &nlsa_conf.L_out, "(Output Laplacian as __L.cfl)"),
		OPT_SET('b', &nlsa_conf.basis_out, "Output Laplace-Beltrami-basis "),
		OPT_CLEAR('d', &conf.dmap, "Do not use Diffusion-Map approach"),
		OPT_SET('v', &conf.local_v, "Local velocity weighting"),		
		OPT_SET('R', &nlsa_conf.riemann, "Include Riemann measure"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (!conf.dmap)
		nlsa_conf.basis_out = true;

	if ( -1 == nlsa_conf.window)
		error("Specify window length '-w'");
	
	nlsa_conf.kernel_dims[0] = nlsa_conf.window;

	if (NULL != nlsa_conf.backproj) {

		check_bp(&nlsa_conf);
		assert(!nlsa_conf.basis_out);
	}
	

	long in_dims[DIMS];
	complex float* in = load_cfl(src_file, DIMS, in_dims);

	if (!md_check_dimensions(DIMS, in_dims, ~(READ_FLAG|PHS1_FLAG)))
		error("Only first two dimensions must be filled!");

	preproc_ac(in_dims, in, nlsa_conf);
	
	long cal0_dims[DIMS];
	md_copy_dims(DIMS, cal0_dims, in_dims);

	if (nlsa_conf.zeropad)
		cal0_dims[0] = in_dims[0] - 1 + nlsa_conf.window;

	complex float* cal = md_alloc(DIMS, cal0_dims, CFL_SIZE);
	
	// resize for zero-padding, else copy
	md_resize_center(DIMS, cal0_dims, cal, in_dims, in, CFL_SIZE); 

	long cal_dims[DIMS];
	md_transpose_dims(DIMS, 1, 3, cal_dims, cal0_dims);

	if (conf.nn > in_dims[0])
		error("Number of nearest neighbours must be smaller or equal to time-steps!");
	
	debug_printf(DP_INFO, nlsa_conf.backproj ? "Performing NLSA\n" : "Performing NLSA-FARY\n");

	long A_dims[2];
	complex float* A = calibration_matrix(A_dims, nlsa_conf.kernel_dims, cal_dims, cal);

	if (nlsa_conf.nlsa_rank == 0) 
		nlsa_conf.nlsa_rank = (A_dims[0] >= A_dims[1]) ? A_dims[1] : A_dims[0];

	
	if (nlsa_conf.rank != 0 && abs(nlsa_conf.rank) > nlsa_conf.nlsa_rank)
		error("Choose rank for backprojection <= nlsa_rank!");

	if (nlsa_conf.weight > -1)		
		weight_delay(A_dims, A, nlsa_conf);

	complex float* back = NULL;
	long back_dims[DIMS] = { 0 };

	if (NULL != nlsa_conf.backproj) {

		md_transpose_dims(DIMS, 3, 1, back_dims, cal_dims);

		back = create_cfl(nlsa_conf.backproj, DIMS, back_dims);
	}

	nlsa_fary(cal_dims, A_dims, A, back, nlsa_conf, conf);

	unmap_cfl(DIMS, in_dims, in);
	md_free(cal);
	
	if (NULL != nlsa_conf.backproj) 
		unmap_cfl(DIMS, back_dims, back);

	exit(0);

}
