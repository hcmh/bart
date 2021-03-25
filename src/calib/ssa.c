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
 * The NLSA is not exactly implemented as proposed by Giannakis & Maida
 * as we do not approximate the Laplacian eigenvectors but calculate them explicitly
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

#include "misc/mmio.h"
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

// spectral EOF analysis
extern int detect_freq_EOF(const long EOF_dims[2], complex float* EOF_fft, const float dt, const float f, const float f_interval, const long max)
{
	long EOF_strs[2];
	md_calc_strides(2, EOF_strs, EOF_dims, CFL_SIZE);

	long T_center = (long)(EOF_dims[0] / 2); // [px]
	float df = 1. / (EOF_dims[0] * dt); // [Hz]

	long F = (long)(f / df); // [px]
	long F_interval = (long)(f_interval / df); // [px]

	long interval_low = F - (long)(F_interval / 2);
	long interval_up = F + (long)(F_interval / 2);

	if (f_interval == -1)
		interval_up = T_center;
	
	// create mask to select frequency-region of interest
	long mask_dims[2] = { EOF_dims[0], 1 };
	long mask_strs[2];
	md_calc_strides(2, mask_strs, mask_dims, CFL_SIZE);

	complex float* mask = md_alloc(2, mask_dims, CFL_SIZE);
	md_clear(2, mask_dims, mask, CFL_SIZE);

	for (int i = interval_low; i < interval_up; i++) {
		
		mask[T_center + i] = 1. + 0i;
		mask[T_center - i] = 1. + 0i;
	}
	if (f_interval == -1)
		mask[0] = 1. + 0i;

	// apply mask to select roi and calculate energy
	complex float* EOF_masked = md_alloc(2, EOF_dims, CFL_SIZE);
	md_zmul2(2, EOF_dims, EOF_strs, EOF_masked, EOF_strs, EOF_fft, mask_strs, mask);

	long EOF_rss_dims[2] = { 1, EOF_dims[1] };
	complex float* EOF_rss = md_alloc(2, EOF_rss_dims, CFL_SIZE);
	md_zrss(2, EOF_dims, MD_BIT(0), EOF_rss, EOF_masked);

	// apply reverse mask (to exclude the roi) and calculate energy
	md_zfill(2, mask_dims, mask, 1.);

	for (int i = interval_low; i < interval_up; i++) {
		
		mask[T_center + i] = 0. + 0i;
		mask[T_center - i] = 0. + 0i;
	}
	if (f_interval == -1)
		mask[0] = 0. + 0i;

	md_zmul2(2, EOF_dims, EOF_strs, EOF_masked, EOF_strs, EOF_fft, mask_strs, mask);

	complex float* EOF_rss_ex = md_alloc(2, EOF_rss_dims, CFL_SIZE);
	md_zrss(2, EOF_dims, MD_BIT(0), EOF_rss_ex, EOF_masked);

	// calculate fraction
	md_zdiv(2, EOF_rss_dims, EOF_rss, EOF_rss, EOF_rss_ex);

	long flags = 0;
	long count = 0;
	float thresh = 1;
	int MAX = 30; // prevent bitmask overflow

	for (int i = 0; i < MAX; i++) {

		if (creal(EOF_rss[i]) > thresh) {

			flags = MD_SET(flags, i);
			count++;
		}

		if (count == max) // both EOFs of the pair detected
			break;
	}
	
	md_free(mask);
	md_free(EOF_masked);
	md_free(EOF_rss);

	return flags;
}

// check symmetry of the W-sized filters
// EOFs that belong to symmetric filters are in phase with the actual oscillation
static bool PC_symmetric(const long PC_line_dims[2], complex float* PC_line, int W) {

	int channels = (int)(PC_line_dims[1] / (1. * W));
	long W_dims[1] = { W };
	long w_dims[1] = { (int)(W / 2)};
	long pos[1] = { 0 };

	complex float* PC_channel = md_alloc(1, W_dims, CFL_SIZE); 
	complex float* PC_channel_h1 = md_alloc(1, w_dims, CFL_SIZE);
	complex float* PC_channel_h2 = md_alloc(1, w_dims, CFL_SIZE); 

	float signf = 0;

	for (int i = 0; i < channels; i++) { // loop over all W-sized filters

		pos[0] = i * W;

		md_copy_block(1, pos, W_dims, PC_channel, &PC_line_dims[1], PC_line, CFL_SIZE);
		md_copy(1, w_dims, PC_channel_h2, PC_channel + 0, CFL_SIZE); // copy left half
		md_flip(1, w_dims, 1, PC_channel_h1, PC_channel_h2, CFL_SIZE);

		md_copy(1, w_dims, PC_channel_h2, PC_channel + w_dims[0], CFL_SIZE); // copy right half

		signf += crealf(md_zscalar(1, w_dims, PC_channel_h1, PC_channel_h2));
	}

	long sign = copysign(1, signf);

	md_free(PC_channel);
	md_free(PC_channel_h1);
	md_free(PC_channel_h2);

	return (sign > 0) ? true : false; // positive sign means symmetric filter

}


static void backprojection( const long N, 
				const long M, 
				const long cal_dims[DIMS], 
				complex float* back, 
				const long A_dims[2], 
				const complex float* B, // ssa: B := A; nlsa: B := zs
				const long T_dims[2], 
				const complex float* T, 
				const complex float* C, // ssa: C := UH; nlsa: C := vH
				struct delay_conf conf)
{

	long l = conf.nlsa ? conf.nlsa_rank : N;
	long PC_xdims[3] = { 1, l, M };
	long T_xdims[3]  = { N, l, 1 };
	long A_xdims[3]  = { N, 1, M };

	assert(T_dims[0]  == N && T_dims[1]  == l);

	long kernelCoil_dims[4];

	md_copy_dims(3, kernelCoil_dims, conf.kernel_dims);

	kernelCoil_dims[3] = cal_dims[3];

	// calculate principle components
	long PC_dims[2] = { l, M };
	complex float* PC = md_alloc(2, PC_dims, CFL_SIZE);

	if (conf.nlsa == false) { // ssa

		// PC = UH @ A (= C @ B)
		/* Consider:
		* AAH = U @ S_square @ UH
		* A = U @ S @ VH --> PC = S @ VH = UH @ A
		*/

		md_ztenmul(3, A_xdims, PC, T_xdims, C, PC_xdims, B);

		if (conf.EOF_info) {

			long PC_line_dims[2] = { 1, M };
			complex float* PC_line = md_alloc(2, PC_line_dims, CFL_SIZE);
			long pos[2] = { 0 };

			int print_max = (l > 20) ? 20 : l;

			debug_printf(DP_INFO, "In phase EOFs:\n");

			for (int i = 0; i < print_max; i++) {

				pos[0] = i;
				md_copy_block(2, pos, PC_line_dims, PC_line, PC_dims, PC, CFL_SIZE); 

				bool sym = PC_symmetric(PC_line_dims, PC_line, conf.window);

				if (sym)
					debug_printf(DP_INFO, "%d ", i);
			}
			debug_printf(DP_INFO, "\n");

			md_free(PC_line);
		}

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

	} else { // nlsa

		// PC = zs @ vH (= B @ C)
		long zs_dims[2] = { l, 1 };

		complex float* zs_red = md_alloc(2, zs_dims, CFL_SIZE);

		for (int j = 0; j < l; j++) { // throw away coefficients
			if (conf.rank < 0)
				zs_red[j] = (j >= abs(conf.rank)) ? B[j] : 0.;
			else 
			if (conf.rank > 0)
				zs_red[j] = (j >= conf.rank) ? 0. : B[j];
			else
			if (conf.group < 0)
				zs_red[j] = (check_selection(conf.group,j)) ? 0. : B[j];
			else
				zs_red[j] = (check_selection(conf.group,j)) ? B[j] : 0.;
		}
		
		long PC_strs[2];
		md_calc_strides(2, PC_strs, PC_dims, CFL_SIZE);

		long zs_strs[2];
		md_calc_strides(2, zs_strs, zs_dims, CFL_SIZE);

		md_zmul2(2, PC_dims, PC_strs, PC, zs_strs, zs_red, PC_strs, C);

		md_free(zs_red);
	} 

	// A_LR = T @ PC
	complex float* A_backproj = md_alloc(2, A_dims, CFL_SIZE);

	md_ztenmul(3, A_xdims, A_backproj, T_xdims, T, PC_xdims, PC);

	// reorder & anti-diagonal summation
	long kern_dims[4];
	md_set_dims(4, kern_dims, 1);
	md_min_dims(4, ~0u, kern_dims, kernelCoil_dims, cal_dims);

	long cal_strs[DIMS];
	md_calc_strides(DIMS, cal_strs, cal_dims, CFL_SIZE);

	long A1_dims[2] = { N, M };

	casorati_matrixH(4, kern_dims, cal_dims, cal_strs, back, A1_dims, A_backproj);

	// missing normalization for summed anti-diagonals
	long b = MIN(kern_dims[0], cal_dims[0] - kern_dims[0] + 1); // minimum of window length and maximum lag

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
	md_free(PC);
}

#if 0 // legacy
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

	for (int i = 0; i < b; i++) {

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
#endif


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
	md_zsmul(2, AAH_dims, AAH, AAH, 100. / md_znorm(2, AAH_dims, AAH));

	// AAH = U @ S @ UH
	long U_dims[2] = { N, N };
	complex float* UH = md_alloc(2, U_dims, CFL_SIZE);

	debug_printf(DP_DEBUG3, "SVD of %dx%d matrix...", AAH_dims[0], AAH_dims[1]);

	lapack_svd(N, N, (complex float (*)[N])U, (complex float (*)[N])UH, S_square, (complex float (*)[N])AAH); // NOTE: Lapack destroys AAH!

	debug_printf(DP_DEBUG3, "done\n");

	if (NULL != back) {

		debug_printf(DP_DEBUG3, "Backprojection...\n");

		backprojection(N, M, cal_dims, back, A_dims, A, U_dims, U, UH, conf);
	}

	md_free(UH);
	md_free(A);
	md_free(AH);
	md_free(AAH);
}


extern void nlsa_fary(	const long cal_dims[DIMS], 
			const long A_dims[2],
			const complex float* A,
			complex float* back,
			const struct delay_conf nlsa_conf, 
			struct laplace_conf laplace_conf)
{

	long N = A_dims[0]; // time
	long M = A_dims[1];

	long L_dims[2] = { N, N };
	complex float* L = md_alloc(2, L_dims, CFL_SIZE);

	debug_printf(DP_DEBUG3, "Calc Laplacian...");
	calc_laplace(&laplace_conf, L_dims, L, A_dims, A);
	debug_printf(DP_DEBUG3, "...done\n");

	if (nlsa_conf.temporal_nn) {
	
		complex float* L_nn = md_alloc(2, L_dims, CFL_SIZE);
		struct laplace_conf laplace_nn_conf = laplace_conf;
		laplace_nn_conf.temporal_nn = 1;
		laplace_nn_conf.kernel = 0;
		laplace_nn_conf.kernel_CG = 0;

		calc_laplace(&laplace_nn_conf, L_dims, L_nn, A_dims, A);

		float lambda_nn = 1.;
		md_zsmul(2, L_dims, L_nn, L_nn, lambda_nn);
		md_zadd(2, L_dims, L, L, L_nn);
		
		md_free(L_nn);
	}


	long U_dims[2] = { N, N };
	long U_strs[2];
	md_calc_strides(2, U_strs, U_dims, CFL_SIZE);

	complex float* U = md_alloc(2, U_dims, CFL_SIZE);
	complex float* UH = md_alloc(2, U_dims, CFL_SIZE);

	float* S_square = xmalloc(N * sizeof(float));	

	if (nlsa_conf.L_out)
		dump_cfl("__L", 2, U_dims, L );


	// L = U @ S_square @ UH
	debug_printf(DP_DEBUG3, "SVD of Laplacian %dx%d matrix...", N, N);

	lapack_svd(N, N, (complex float (*)[N])U, (complex float (*)[N])UH, S_square, (complex float (*)[N])L); // NOTE: Lapack destroys L!
	
	debug_printf(DP_DEBUG3, "done\n");

	if (nlsa_conf.basis_out) {
		// output Laplace-Beltrami basis

		complex float* T = create_cfl(nlsa_conf.name_tbasis, 2, U_dims);

		if (laplace_conf.dmap)
			md_copy(2, U_dims, T, U, CFL_SIZE);	
		else
			md_flip(2, U_dims, MD_BIT(1), T, U, CFL_SIZE); // flip, because non-Diffusion-Map Laplacians have most significant EVs at the end

		unmap_cfl(2, U_dims, T);

		// convert to complex number
		long zs_dims[1] = { N };
		complex float* zs = ((NULL != nlsa_conf.name_S) ? create_cfl : anon_cfl) (nlsa_conf.name_S, 1, zs_dims);

		for (int i = 0; i < N; i++) {
			if (laplace_conf.dmap)
				zs[i] = S_square[i] + 0.i;
			else
				zs[i] = S_square[N - i -1] + 0.i;
		}
		
		unmap_cfl(1, zs_dims, zs);

	} else {
		// actual NLSA: Project temporal process on Laplace-Beltrami basis 

		// TODO: avoid explicit allocation by using strides
		long l = nlsa_conf.nlsa_rank;
		long Ur_dims[2] = { N, l }; // rank cut off
		complex float* Ur = md_alloc(2, Ur_dims, CFL_SIZE);
		md_resize(2, Ur_dims, Ur, U_dims, U, CFL_SIZE);

		long Utp_dims[2] = { l, N };
		complex float* Utp = md_alloc(2, Utp_dims, CFL_SIZE);
		md_transpose(2, 0, 1, Utp_dims, Utp, Ur_dims, Ur, CFL_SIZE);

		// include Riemann measure
		if (nlsa_conf.riemann) {
			/* Giannakis, D., & Majda, A. J. (2013).
 			 * Nonlinear Laplacian spectral analysis: capturing intermittent and low‐frequency spatiotemporal patterns in high‐dimensional data.
 			 * Statistical Analysis and Data Mining: The ASA Data Science Journal, 6(3), 180-194.
			 * 
			 * Utp_riemann = Utp[l, :] * riemann[:]
			 * Q[:] = sum(wght[:,:], 0)
			 * riemann = Q[:] / sum(Q)
			 */

			struct laplace_conf riemann_conf = laplace_conf_default;
			riemann_conf.W = true;

			complex float* wght = md_alloc(2, L_dims, CFL_SIZE);
			calc_laplace(&riemann_conf, L_dims, wght, A_dims, A);

			long riemann_dims[2] = { 1, L_dims[1]};
			complex float* riemann = md_alloc(2, riemann_dims, CFL_SIZE);
			md_zsum(2, L_dims, READ_FLAG, riemann, wght);

			long riemann_strs[2];
			md_calc_strides(2, riemann_strs, riemann_dims, CFL_SIZE);
			long Utp_strs[2];
			md_calc_strides(2, Utp_strs, Utp_dims, CFL_SIZE);
			
			complex float norm = 1;
			md_zsum(2, riemann_dims, PHS1_FLAG, &norm, riemann);

			md_zmul2(2, Utp_dims, Utp_strs, Utp, riemann_strs, riemann, Utp_strs, Utp);
			md_zsmul(2, Utp_dims, Utp, Utp, norm);

			md_free(wght);
			md_free(riemann);
		}

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

		if (l > M)
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


		// convert to complex number
		long zs_dims[1] = { l };
		complex float* zs = ((NULL != nlsa_conf.name_S) ? create_cfl : anon_cfl) (nlsa_conf.name_S, 1, zs_dims);

		for (int i = 0; i < l; i++)
			zs[i] = s[i] + 0.i;


		// temporal basis: T = U @ u (alternative: T[i,j] = sum(U[i,:l] * u[:l,j]))
		long T_dims[2] = { N, l };
		complex float* T = create_cfl(nlsa_conf.name_tbasis, 2, T_dims);

		long U_xdims[3] = { N, l, 1};
		long u_xdims[3] = { 1, l, l};
		long T_xdims[3] = { N, 1, l};
				
		md_ztenmul(3, T_xdims, T, U_xdims, U, u_xdims, u);


		if (NULL != back) {

			debug_printf(DP_DEBUG3, "Backprojection...\n");

			backprojection(N, M, cal_dims, back, A_dims, zs, T_dims, T, vH, nlsa_conf);
		}

		md_free(Ur);
		md_free(Utp);
		md_free(UA);
		xfree(s);
		unmap_cfl(1, zs_dims, zs);

	}

	md_free(L);
	md_free(U);
	md_free(UH);	
	xfree(S_square);
}


