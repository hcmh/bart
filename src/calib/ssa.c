/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 */


/* References
 * ----------
 *
 * SSA:
 *
 * Vautard R, Ghil M.
 * Singular spectrum analysis in nonlinear dynamics, with applications to 
 * paleoclimatic time series. Physica D: Nonlinear Phenomena, 1989;35:395-424.
 *
 * (and others)
 *
 * SSA-FARY:
 *
 * Rosenzweig S, Scholand N, Holme HCM, Uecker M.
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular 
 * Spectrum Analysis (SSA-FARY). IEEE Trans. Med. Imag. 2020; in press.
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
 * General comments:
 *
 * The rank option '-r' allows to "throw away" basis functions:
 *	rank < 0: throw away 'rank' basis functions with high singular values
 *	rank > 0: keep only 'rank' basis functions with the highest singular value
 *
 * The group option '-g' implements what is called 'grouping' in SSA 
 * literature, by selecting EOFs with a bitmask.
 *  group < 0: do not use the selected group for backprojection, but all other
 *            EOFs (= filtering)
 *  group > 0: use only the selected group for backprojection
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/mmio.h" // TODO: should not be here?
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/init.h"
#include "num/casorati.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/flpmath.h"

#include "calib/calib.h"
#include "calib/estvar.h"
#include "calib/calmat.h"

#include "ssa.h"





static void ssa_backprojection( const long N,
				const long M,
				const long cal_dims[DIMS],
				complex float* back,
				const long A_dims[2],
				const complex float* A,
				const long U_dims[2],
				const complex float* U,
				const complex float* UH,
				const struct delay_conf conf)
{
	assert((N == U_dims[0]) && (N == U_dims[1]));

	// PC = UH @ A
	/* Consider:
	 * AAH = U @ S_square @ UH
	 * A = U @ S @ VH --> PC = S @ VH = UH @ A
	 */
	long PC_dims[2] = { N, M };
	complex float* PC = md_alloc(2, PC_dims, CFL_SIZE);

	long PC2_dims[3] = { N, 1, M };
	long U2_dims[3] = { N, N, 1 };
	long A3_dims[3] = { 1, N, M };

	md_ztenmul(3, PC2_dims, PC, U2_dims, UH, A3_dims, A);

	long kernelCoil_dims[4];

	md_copy_dims(3, kernelCoil_dims, conf.kernel_dims);

	kernelCoil_dims[3] = cal_dims[3];


	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {

			if (conf.rank < 0)
				PC[i * N + j] *= (j >= abs(conf.rank)) ? 1. : 0.;
			else
			if (conf.rank > 0)
				PC[i * N + j] *= (j >= conf.rank) ? 0. : 1.;
			else
			if (conf.group < 0)
				PC[i * N + j] *= (check_selection(conf.group, j)) ? 0. : 1.;
			else
				PC[i * N + j] *= (check_selection(conf.group, j)) ? 1. : 0.;
		}
	}

	// A_LR = U @ PC
	long PC3_dims[3] = { 1, N, M };
	long A4_dims[3] = { N, 1, M };

	complex float* A_backproj = md_alloc(2, A_dims, CFL_SIZE);

	md_ztenmul(3, A4_dims, A_backproj, U2_dims, U, PC3_dims, PC);


	// Reorder & Anti-diagonal summation
	long kern_dims[4];
	md_set_dims(4, kern_dims, 1);
	md_min_dims(4, ~0u, kern_dims, kernelCoil_dims, cal_dims);

	long cal_strs[DIMS];
	md_calc_strides(DIMS, cal_strs, cal_dims, CFL_SIZE);

	casorati_matrixH(4, kern_dims, cal_dims, cal_strs, back, A_dims, A_backproj);

	// Missing normalization for summed anti-diagonals
	long b = MIN(kern_dims[0], cal_dims[0] - kern_dims[0] + 1); // Minimum of window length and maximum lag

	long norm_dims[DIMS];
	md_singleton_dims(DIMS, norm_dims);

	norm_dims[0] = cal_dims[0];

	complex float* norm = md_alloc(DIMS, norm_dims, CFL_SIZE);

	md_zfill(DIMS, norm_dims, norm, 1. / b);

	for (unsigned int i = 0; i < b; i++) {

		norm[i] = 1. / (i + 1);
		norm[cal_dims[0] -1 - i] = 1. / (i + 1);
	}

	long norm_strs[DIMS];
	md_calc_strides(DIMS, norm_strs, norm_dims, CFL_SIZE);

	md_zmul2(DIMS, cal_dims, cal_strs, back, cal_strs, back, norm_strs, norm);

	md_free(norm);
	md_free(A_backproj);
	md_free(PC);
}


extern void ssa_fary(	const long cal_dims[DIMS],
			const long A_dims[2],
			const complex float* A,
			complex float* U,
			float* S_square,
			complex float* back,
			const struct delay_conf conf)
{
	long N = A_dims[0];
	long M = A_dims[1];

	long AH_dims[2] = { M, N };
	complex float* AH = md_alloc(2, AH_dims, CFL_SIZE);

	md_transpose(2, 0, 1, AH_dims, AH, A_dims, A, CFL_SIZE);
	md_zconj(2, AH_dims, AH, AH);

	long AAH_dims[2] = { N, N };
	complex float* AAH = md_alloc(2, AAH_dims, CFL_SIZE);

	// AAH = A @ AH
	long A2_dims[3] = { N, M, 1 };
	long AH2_dims[3] = { 1, M, N };
	long AAH2_dims[3] = { N, 1, N };

	md_ztenmul(3, AAH2_dims, AAH, A2_dims, A, AH2_dims, AH);


	// AAH = U @ S @ UH
	long U_dims[2] = { N, N };
	complex float* UH = md_alloc(2, U_dims, CFL_SIZE);

	debug_printf(DP_DEBUG3, "SVD of %dx%d matrix...", AAH_dims[0], AAH_dims[1]);

	lapack_svd(N, N, (complex float (*)[N])U, (complex float (*)[N])UH, S_square, (complex float (*)[N])AAH); // NOTE: Lapack destroys AAH!

	debug_printf(DP_DEBUG3, "done\n");

	if (NULL != back) {

		debug_printf(DP_DEBUG3, "Backprojection...\n");

		ssa_backprojection(N, M, cal_dims, back, A_dims, A, U_dims, U, UH, conf);
	}

	md_free(UH);
	md_free(A);
	md_free(AH);
	md_free(AAH);
}


static void nlsa_backprojection( const long N, 
				const long M, 
				const long cal_dims[DIMS], 
				complex float* back, 
				const long vH_dims[2], 
				const complex float* vH, 
				const long zs_dims[2], 
				const complex float* zs, 
				const long T_dims[2], 
				const complex float* T, 
				struct delay_conf nlsa_conf)
{

	long l = nlsa_conf.nlsa_rank;
	assert(vH_dims[0] == l && vH_dims[1] == M);
	assert(zs_dims[0] == 1 && zs_dims[1] == l);
	assert(T_dims[0]  == N  && T_dims[1]  == l);


	long kernelCoil_dims[4];

	md_copy_dims(3, kernelCoil_dims, nlsa_conf.kernel_dims);

	kernelCoil_dims[3] = cal_dims[3];


	// T_zs = T @ zs
	long T_strs[2];
	md_calc_strides(2, T_strs, T_dims, CFL_SIZE);

	long zs_strs[2];
	md_calc_strides(2, zs_strs, zs_dims, CFL_SIZE);

	complex float* T_zs = md_alloc(2, T_dims, CFL_SIZE);

	md_zmul2(2, T_dims, T_strs, T_zs, T_strs, T, zs_strs, zs);


	// Throw away unwanted basis functions
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < l; j++) {

			if (nlsa_conf.rank < 0)
				T_zs[j * N + i] *= (j >= abs(nlsa_conf.rank)) ? 1. : 0.;
			else 
			if (nlsa_conf.rank > 0)
				T_zs[j * N + i] *= (j >= nlsa_conf.rank) ? 0. : 1;
			else
			if (nlsa_conf.group < 0)
				T_zs[j * N + i] *= (check_selection(nlsa_conf.group,j)) ? 0. : 1.;
			else
				T_zs[j * N + i]  *= (check_selection(nlsa_conf.group,j)) ? 1. : 0.;
		}
	}


	// A_LR = T_zs @ vH
	long A_dims[2] = { N, M };
	long vH_xdims[3] = { 1, l, M };
	long T_xdims[3]  = { N, l, 1 };
	long A_xdims[3] = { N, 1, M };

	complex float* A_backproj = md_alloc(3, A_dims, CFL_SIZE);

	md_ztenmul(3, A_xdims, A_backproj, T_xdims, T_zs, vH_xdims, vH);

	// Reorder & Anti-diagonal summation
	long kern_dims[4];
	md_set_dims(DIMS, kern_dims, 1);
	md_min_dims(4, ~0u, kern_dims, kernelCoil_dims, cal_dims);

	long cal_strs[DIMS];
	md_calc_strides(DIMS, cal_strs, cal_dims, CFL_SIZE);

	long A1_dims[3] = { N, M };

	casorati_matrixH(4, kern_dims, cal_dims, cal_strs, back, A1_dims, A_backproj);

	// Missing normalization for summed anti-diagonals
	long b = MIN(kern_dims[0], cal_dims[0] - kern_dims[0] + 1); // Minimum of window length and maximum lag

	long norm_dims[DIMS];
	md_singleton_dims(DIMS, norm_dims);

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
	md_free(T_zs);

}

extern void nlsa_fary(	const long cal_dims[DIMS], 
			const long A_dims[2],
			const complex float* A,
			complex float* back,
			const struct delay_conf nlsa_conf, 
			struct laplace_conf conf)
{

	long N = A_dims[0]; // time
	long M = A_dims[1];

	long L_dims[2] = { N, N };
	complex float* L = md_alloc(2, L_dims, CFL_SIZE);

	calc_laplace(&conf, L_dims, L, A_dims, A);

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

	// TODO: avoid explicit allocation by using strides
	long l = nlsa_conf.nlsa_rank;
	long Ur_dims[2] = { N, l }; // rank cut off
	complex float* Ur = md_alloc(2, Ur_dims, CFL_SIZE);
	md_resize(2, Ur_dims, Ur, U_dims, U, CFL_SIZE);

	long Utp_dims[2] = { l, N };
	complex float* Utp = md_alloc(2, Utp_dims, CFL_SIZE);
	md_transpose(2, 0, 1, Utp_dims, Utp, Ur_dims, Ur, CFL_SIZE);


	// UA = Utp @ A (alternative: U[i,j] = np.sum(U[:,j] * A[:,i] )
	long UA_dims[2];
	UA_dims[0] = l;
	UA_dims[1] = M;

	long A_xdims[3] = { 1, N, M };
	long Utp_xdims[3] = { l, N, 1 };
	long UA_xdims[3] = { l, 1, M };

	long A_xstrs[3];
	md_calc_strides(3, A_xstrs, A_xdims, CFL_SIZE);
	long Utp_xstrs[3];
	md_calc_strides(2, Utp_xstrs, Utp_xdims, CFL_SIZE);	
	long UA_xstrs[3];
	md_calc_strides(3, UA_xstrs, UA_xdims, CFL_SIZE);

	if (l >= M)
		error("Choose smaller nlsa_rank!");

	complex float* UA = md_alloc(2, UA_dims, CFL_SIZE);

	long max_dims[3];
	md_tenmul_dims(3, max_dims, UA_xdims, Utp_xdims, A_xdims);

	md_ztenmul(3, UA_xdims, UA, Utp_xdims, Utp, A_xdims, A);

	// UA = u @ s @ vH
	long u_dims[2] = { l, l };
	long vH_dims[2] = { l, M };
	long vH_strs[2];
	md_calc_strides(2, vH_strs, vH_dims, CFL_SIZE);

	complex float* u = md_alloc(2, u_dims, CFL_SIZE);
	complex float* vH = md_alloc(2, vH_dims, CFL_SIZE);

	float* s = xmalloc(l * sizeof(float));

	debug_printf(DP_DEBUG3, "SVD of Projection %dx%d matrix...", M, l);

	// NOTE: Lapack destroys L!
	lapack_svd_econ(l, M, (complex float (*)[l])u, (complex float (*)[l])vH, s, (complex float (*)[l])UA);

	debug_printf(DP_DEBUG3, "done\n");


	// Make complex number
	long zs_dims[2] = { 1, l };
	complex float* zs = ((NULL != nlsa_conf.name_S) ? create_cfl : anon_cfl) (nlsa_conf.name_S, 2, zs_dims);

	for (int i = 0; i < l; i++)
		zs[i] = s[i] + 0.i;


	// Temporal basis: T = U @ u (alternative: T[i,j] = sum(U[i,:l] * u[:l,j]))
	long T_dims[2] = { N, l };
	complex float* T = create_cfl(nlsa_conf.name_tbasis, 2, T_dims);

	long U_xdims[3] = { N, l, 1};
	long u_xdims[3] = { 1, l, l};
	long T_xdims[3] = { N, 1, l};
			
	md_ztenmul(3, T_xdims, T, U_xdims, U, u_xdims, u);


	if (NULL != back) {

		debug_printf(DP_DEBUG3, "Backprojection...\n");

		nlsa_backprojection(N, M, cal_dims, back, vH_dims, vH, zs_dims, zs, T_dims, T, nlsa_conf);
	}

	md_free(Ur);
	md_free(Utp);
	md_free(U);
	md_free(UH);
	md_free(L);
	md_free(UA);
	xfree(s);
	unmap_cfl(2, zs_dims, zs);
	xfree(S_square);

}


