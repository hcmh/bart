/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 */

/* Paper
 * -----------
 * SSA
 * Vautard, R., & Ghil, M. (1989).
 * Singular spectrum analysis in nonlinear dynamics, with applications to paleoclimatic time series.
 * Physica D: Nonlinear Phenomena, 35(3), 395-424.
 *
 * (and others)
 *
 * SSA-FARY
 * Rosenzweig, S., Scholand, N., Holme, H. C. M., & Uecker, M. (2018).
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular Spectrum Analysis (SSA-FARY).
 * arXiv preprint arXiv:1812.09057.
 *
 * NLSA
 * Giannakis, D., & Majda, A. J. (2012).
 * Nonlinear Laplacian spectral analysis for time series with intermittency and low-frequency variability.
 * Proceedings of the National Academy of Sciences, 109(7), 2222-2227.
 *
 * Giannakis, D., & Majda, A. J. (2013).
 * Nonlinear Laplacian spectral analysis: capturing intermittent and low‐frequency spatiotemporal patterns in high‐dimensional data.
 * Statistical Analysis and Data Mining: The ASA Data Science Journal, 6(3), 180-194.
 *
 * Comments on NLSA:
 * The NLSA is not exactly implemented as proposed by Giannakis & Maida:
 *	We don't use the metric mu (yet)
 * 	We don't consider the local velocities in the exponent
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


static const char usage_str[] = "<src> <EOF> [<S>] [<backprojection>]";
static const char help_str[] =
		"Perform SSA-FARY or Singular Spectrum Analysis. <src>: [samples, coordinates]\n";


// FIXME: MERGE
#include "num/casorati.h"
#include "manifold/manifold.h"
#include "num/lapack.h"


static bool check_selection(const long group, const int j)
{
	if (j > 30)
		return false; // group has only 32 bits

	if (labs(group) & (1 << j))
		return true;

	else
		return false;

}


static void nlsa_backprojection(const long N, const long M, const long kernel_dims[3], const long cal_dims[DIMS], complex float* back, const long U_proj_dims[2], const complex
float* U_proj, const long zS_proj_dims[2], const complex float* zS_proj, const long t_basis_dims[2], const complex float* t_basis, const int nlsa_rank, const int rank, const long
group)
{

	assert(U_proj_dims[0] == M && U_proj_dims[1] == nlsa_rank);
	assert(zS_proj_dims[0] == 1 && zS_proj_dims[1] == nlsa_rank);
	assert(t_basis_dims[0] == N && t_basis_dims[1] == nlsa_rank);


	long kernelCoil_dims[4];
	md_copy_dims(3, kernelCoil_dims, kernel_dims);
	kernelCoil_dims[3] = cal_dims[3];


	// t1_basis = t_basis @ zS_proj
	long t_basis_strs[2];
	md_calc_strides(2, t_basis_strs, t_basis_dims, CFL_SIZE);

	long zS_proj_strs[2];
	md_calc_strides(2, zS_proj_strs, zS_proj_dims, CFL_SIZE);

	complex float* t1_basis = md_alloc(2, t_basis_dims, CFL_SIZE);

	md_zmul2(2, t_basis_dims, t_basis_strs, t1_basis, t_basis_strs, t_basis, zS_proj_strs, zS_proj);

	// Throw away unwanted basis functions
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < nlsa_rank; j++) {

			if (rank < 0)
				t1_basis[j * N + i] *= (j >= abs(rank)) ? 1. : 0.;

			else if (rank > 0)
				t1_basis[j * N + i] *= (j >= rank) ? 0. : 1;

			else if (group < 0)
				t1_basis[j * N + i] *= (check_selection(group,j)) ? 0. : 1.;

			else
				t1_basis[j * N + i]  *= (check_selection(group,j)) ? 1. : 0.;
		}
	}


	// A_backproj = t1_basis @ U_proj.H
	long t2_dims[3] = { N, nlsa_rank, 1 };
	long U2_dims[3] = { 1, M, nlsa_rank };
	long A_dims[3] = { N, 1, M };
	long max_dims[3] = { N, nlsa_rank, M };

	complex float* A_backproj = md_alloc(3, A_dims, CFL_SIZE);
	long A_strs[3];
	md_calc_strides(3, A_strs, A_dims, CFL_SIZE);
	long t2_strs[3];
	md_calc_strides(3, t2_strs, t2_dims, CFL_SIZE);
	long U2_strs[3];
	md_calc_strides(3, U2_strs, U2_dims, CFL_SIZE);
	long U2tp_strs[3];
	md_transpose_dims(3, 1, 2, U2tp_strs, U2_strs); // Transpose U_proj via strides manipulation

	complex float* U_projH = md_alloc(2, U_proj_dims, CFL_SIZE);
	md_zconj(2, U_proj_dims, U_projH, U_proj);

	md_ztenmul2(3, max_dims, A_strs, A_backproj, t2_strs, t1_basis, U2tp_strs, U_projH);

	// Reorder & Anti-diagonal summation
	long kern_dims[4];
	md_set_dims(4, kern_dims, 1);
	md_min_dims(4, ~0u, kern_dims, kernelCoil_dims, cal_dims);

	long cal_strs[DIMS];
	md_calc_strides(DIMS, cal_strs, cal_dims, CFL_SIZE);

	long A1_dims[2] = { N, M };

	casorati_matrixH(4, kern_dims, cal_dims, cal_strs, back, A1_dims, A_backproj);

	// Missing normalization for summed anti-diagonals
	long b = MIN(kern_dims[0], cal_dims[0] - kern_dims[0] + 1); // Minimum of window length and maximum lag

	long norm_dims[DIMS];
	for (unsigned int i = 0; i < DIMS; i++)
		norm_dims[i] = 1;
	norm_dims[0] = cal_dims[0];

	complex float* norm = md_alloc(DIMS, norm_dims, CFL_SIZE);
	md_zfill(DIMS, norm_dims, norm, 1./b);

	for (unsigned int i = 0; i < b; i++) {
		norm[i] = 1. / (i + 1);
		norm[cal_dims[0] -1 - i] = 1. / (i + 1);
	}

	long norm_strs[DIMS];
	md_calc_strides(DIMS, norm_strs, norm_dims, CFL_SIZE);
	md_zmul2(DIMS, cal_dims, cal_strs, back, cal_strs, back, norm_strs, norm);

	md_free(A_backproj);
	md_free(norm);
	md_free(U_projH);
	md_free(t1_basis);



}


static void nlsa_fary(const long kernel_dims[3], const long cal_dims[DIMS], const complex float* cal, const char* name_EOF, const char* name_S, const char* backproj, const int
nlsa_rank, const int rank, const long group, struct laplace_conf conf)
{

	// Calibration matrix
	long A_dims[2];
	complex float* A = calibration_matrix(A_dims, kernel_dims, cal_dims, cal);

	long L_dims[2];
	L_dims[0] = A_dims[0];
	L_dims[1] = A_dims[0];
	complex float* L = md_alloc(2, L_dims, CFL_SIZE);

	calc_laplace(&conf, L_dims, L, A_dims, A);

	long N = A_dims[0]; // time
	long M = A_dims[1];
	long U_dims[2] = { N, N };
	long U_strs[2];
	md_calc_strides(2, U_strs, U_dims, CFL_SIZE);

	complex float* U = md_alloc(2, U_dims, CFL_SIZE);
	complex float* UH = md_alloc(2, U_dims, CFL_SIZE);

	float* S_square = xmalloc(N * sizeof(float));

	// L = U @ S_square @ UH
	debug_printf(DP_DEBUG3, "SVD of Laplacian %dx%d matrix...", N, N);
	lapack_svd(N, N, (complex float (*)[N])U, (complex float (*)[N])UH, S_square, (complex float (*)[N])L); // NOTE: Lapack destroys L!
	debug_printf(DP_DEBUG3, "done\n");

	if (name_S != NULL) {
		long S_dims[1] = { N };
		complex float* S = create_cfl(name_S, 1, S_dims);
		for (int i = 0; i < N; i++)
			S[i] = sqrt(S_square[i]) + 0i;
		unmap_cfl(1, S_dims, S);
	}

	// UA[i,j] = np.sum(A[:,i] * U[:,j])
	long UA_dims[2];
	UA_dims[0] = M;
	UA_dims[1] = nlsa_rank;

	if (nlsa_rank >= M)
		error("Choose smaller nlsa_rank!");

	complex float* UA = md_alloc(2, UA_dims, CFL_SIZE);

#pragma omp parallel for
	for (int i = 0; i < M; i++)
		for (int j = 0; j < nlsa_rank ; j++) {

			UA[j * M + i] = md_zscalar(1,  &N, &U[j * N], &A[i * N]) ;

		}


	// UA = U_proj @ S_proj @ VH_proj
	long l = (long)nlsa_rank;

	long U_proj_dims[2] = { M, l};
	long VH_proj_dims[2] = { l, l};
	long VH_proj_strs[2];
	md_calc_strides(2, VH_proj_strs, VH_proj_dims, CFL_SIZE);

	complex float* U_proj = md_alloc(2, U_proj_dims, CFL_SIZE);
	complex float* VH_proj = md_alloc(2, VH_proj_dims, CFL_SIZE);

	float* S_proj = xmalloc(l * sizeof(float));

	debug_printf(DP_DEBUG3, "SVD of Projection %dx%d matrix...", M, l);
	lapack_svd_econ(M, l, (complex float (*)[l])U_proj, (complex float (*)[l])VH_proj, S_proj, (complex float (*)[l])UA);
	// NOTE: Lapack destroys L!
	debug_printf(DP_DEBUG3, "done\n");


	// Make complex number
	long zS_proj_dims[2] = { 1, l };
	complex float* zS_proj = md_alloc(2, zS_proj_dims, CFL_SIZE);
	for (int i = 0; i < l; i++)
		zS_proj[i] = S_proj[i] + 0i;


	// Temporal basis: t_basis[i,j] = sum(U[i,:l] * VH_proj[:l,j])
	long t_basis_dims[2] = { N, l };
	complex float* t_basis = create_cfl(name_EOF, 2, t_basis_dims);

	long strs = U_strs[1];


	#pragma omp parallel for
	for (int i = 0; i < N; i++)
		for (int j = 0; j < l ; j++)
			t_basis[j * N + i] = md_zscalar2(1, &l,  &strs, &U[i], &VH_proj_strs[1], &VH_proj[j] );


	if (backproj != NULL) {
		long back_dims[DIMS];
		md_transpose_dims(DIMS, 3, 1, back_dims, cal_dims);

		complex float* back = create_cfl(backproj, DIMS, back_dims);
		debug_printf(DP_DEBUG3, "Backprojection...\n");
		nlsa_backprojection(N, M, kernel_dims, cal_dims, back, U_proj_dims, U_proj, zS_proj_dims, zS_proj, t_basis_dims, t_basis, nlsa_rank, rank, group);
	}


	md_free(A);
	md_free(U);
	md_free(UH);
	md_free(L);
	md_free(UA);
	xfree(S_square);
	xfree(S_proj);
	md_free(zS_proj);

}






int main_ssa(int argc, char* argv[])
{
	int window = -1;
	int normalize = 0;
	int rm_mean = 1;
	int rank = 0;
	bool zeropad = true;
	long kernel_dims[3] = { 1, 1, 1 };
	char* name_S = NULL;
	char* backproj = NULL;
	int nlsa_rank = 0;
	bool nlsa = false;
	long group = 0;


	struct laplace_conf conf = laplace_conf_default;

	const struct opt_s opts[] = {

		OPT_INT('w', &window, "window", "Window length"),
		OPT_CLEAR('z', &zeropad, "Zeropadding [Default: True]"),
		OPT_INT('m', &rm_mean, "0/1", "Remove mean [Default: True]"),
		OPT_INT('n', &normalize, "0/1", "Normalize [Default: False]"),
		OPT_INT('r', &rank, "rank", "Rank for backprojection. r < 0: Throw away first r components. r > 0: Use only first r components."),
		OPT_LONG('g', &group, "bitmask", "Bitmask for Grouping (long value!)"),
		OPT_INT('L', &nlsa_rank, "NLSA", "Rank for Nonlinear Laplacian Spectral Analysis"),
		OPT_INT('N', &conf.nn, "nn", "Number of nearest neighbours"),
		OPT_FLOAT('S', &conf.sigma, "sigma", "Standard deviation"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (nlsa_rank > 0)
		nlsa = true;

	if (-1 == window)
		error("Specify window length '-w'");

	kernel_dims[0] = window;

	char* name_EOF = argv[2];

	if (4 <= argc)
		name_S = argv[3];

	if (5 == argc) {

		backproj = argv[4];

		if (zeropad) {

			debug_printf(DP_INFO, "Zeropadding turned off automatically!\n");

			zeropad = false;
		}

		if ((0 == rank) && (0 == group))
			error("Specify rank or group for backprojection!");

		if (0 == rank)
			assert(0 != group);

		if (0 == group)
			assert(0 != rank);
	}

	if (nlsa && rank != 0 && abs(rank) > nlsa_rank)
		error("Chose rank <= nlsa_rank!");


	long in_dims[DIMS];
	complex float* in = load_cfl(argv[1], DIMS, in_dims);

	if (!md_check_dimensions(DIMS, in_dims, ~(READ_FLAG|PHS1_FLAG)))
		error("Only first two dimensions must be filled!");


	if (rm_mean || normalize) {

		long in_strs[DIMS];
		md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

		long singleton_dims[DIMS];
		long singleton_strs[DIMS];
		md_select_dims(DIMS, ~READ_FLAG, singleton_dims, in_dims);
		md_calc_strides(DIMS, singleton_strs, singleton_dims, CFL_SIZE);

		if (rm_mean) {

			complex float* mean = md_alloc(DIMS, singleton_dims, CFL_SIZE);

			md_zavg(DIMS, in_dims, READ_FLAG, mean, in);
			md_zsub2(DIMS, in_dims, in_strs, in, in_strs, in, singleton_strs, mean);

			md_free(mean);
		}

		if (normalize) {

			complex float* stdv = md_alloc(DIMS, singleton_dims, CFL_SIZE);

			md_zstd(DIMS, in_dims, READ_FLAG, stdv, in);
			md_zdiv2(DIMS, in_dims, in_strs, in, in_strs, in, singleton_strs, stdv);

			md_free(stdv);
		}
	}


	long cal0_dims[DIMS];
	md_copy_dims(DIMS, cal0_dims, in_dims);

	if (zeropad)
		cal0_dims[0] = in_dims[0] - 1 + window;


	complex float* cal = md_alloc(DIMS, cal0_dims, CFL_SIZE);

	md_resize_center(DIMS, cal0_dims, cal, in_dims, in, CFL_SIZE); 

	long cal_dims[DIMS];
	md_transpose_dims(DIMS, 1, 3, cal_dims, cal0_dims);


	if (nlsa) {

		if (conf.nn > in_dims[0])
			error("Number of nearest neighbours must be smaller or equalt o time-steps!");

		debug_printf(DP_INFO, backproj ? "Performing NLSA\n" : "Performing NLSA-FARY\n");

		conf.gen_out = true;
		nlsa_fary(kernel_dims, cal_dims, cal, name_EOF, name_S, backproj, nlsa_rank, rank, group, conf);


	} else {

	debug_printf(DP_INFO, backproj ? "Performing SSA\n" : "Performing SSA-FARY\n");

	long A_dims[2];
	complex float* A = calibration_matrix(A_dims, kernel_dims, cal_dims, cal);

	long N = A_dims[0];

	long U_dims[2] = { N, N };
	complex float* U = create_cfl(name_EOF, 2, U_dims);

	complex float* back = NULL;

	if (NULL != backproj) {

		long back_dims[DIMS];
		md_transpose_dims(DIMS, 3, 1, back_dims, cal_dims);

		back = create_cfl(backproj, DIMS, back_dims);
	}

	float* S_square = xmalloc(N * sizeof(float));

	ssa_fary(kernel_dims, cal_dims, A_dims, A, U, S_square, back, rank, group);

	if (NULL != name_S) {

		long S_dims[1] = { N };
		complex float* S = create_cfl(name_S, 1, S_dims);

		for (int i = 0; i < N; i++)
			S[i] = sqrt(S_square[i]) + 0.i;

		unmap_cfl(1, S_dims, S);
	}

	xfree(S_square);

	unmap_cfl(2, U_dims, U);

	}

	unmap_cfl(DIMS, in_dims, in);

	md_free(cal);

	exit(0);
}


