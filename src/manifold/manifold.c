/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019-2020 Sebastian Rosenzweig (sebastian.rosenzweig@med.uni-goettingen.de)
 * 
 */

#include <complex.h>
#include <math.h>
// #include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/filter.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "manifold.h"

#include "linops/linop.h"
//#include "linops/someops.h"
#include "linops/fmac.h"

#include "iter/iter.h"
#include "iter/lsqr.h"
#include "iter/prox.h"

const struct laplace_conf laplace_conf_default = {

	.sigma 			= -1,
	.nn 			= -1,
	.temporal_nn 	= false,
	.kernel     	= false,
	.kernel_CG		= false,
	.kernel_lambda 	= 0.3,
	.kernel_gamma	= 0.00001,
	.norm			= false,
	.anisotrop		= false,
	.dmap			= false,
	.median 		= -1,
	.iter_max		= 50,

};

// Calculate kernel-based weighting matrix
// W = kernel * (kernel + gamma * I)^{-0.5}
static void calc_kernel_W(	const long N, 
							const complex float* kernel, 
							complex float* kernel_cpy, 
							complex float* W, 
							complex float* buf, 
							complex float* V, 
							complex float* VH, 
							const float gamma) 
{

	long V_dims[3]      = { N, N, 1 };
	long VH_dims[3]     = { 1, N, N };
	long cov_dims[3]    = { N, 1, N };

	long V_strs[3];
	md_calc_strides(3, V_strs, V_dims, CFL_SIZE);

	long S_dims[1] = { N };
	long S_strs[3] = { 0, CFL_SIZE, 0 };
	complex float* S_inv = md_alloc(1, S_dims, CFL_SIZE);	

	float* Sf = xmalloc(N * sizeof(float));

	// SVD
	md_copy(3, cov_dims, kernel_cpy, kernel, CFL_SIZE);
	lapack_svd(N, N, (complex float (*)[N])V, (complex float (*)[N])VH, Sf, (complex float (*)[N])kernel_cpy); // NOTE: Lapack destroys kernel_cpy!

	// W = V @ (S + eye * gamma)^(-0.5) @ VH
	for (int j = 0; j < N; j++)
		S_inv[j] = pow((Sf[j] + gamma), -0.5) + 0 * 1.i;
	
	md_zmul2(3, V_dims, V_strs, W, V_strs, V, S_strs, S_inv);
	md_ztenmul(3, cov_dims, buf, V_dims, W, VH_dims, VH);

	// W_final = - kernel * W
	md_zmul(3, V_dims, W, kernel, buf);
	md_zsmul(3, V_dims, W, W, -1.);

	md_free(S_inv);
	xfree(Sf);
}

// Set sigma to maximum distance
static void calc_sigma(const long L_dims[2], const complex float* dist, struct laplace_conf* conf)
{

	complex float* dist_tmp = md_alloc(2, L_dims, CFL_SIZE);
	md_copy(2, L_dims, dist_tmp, dist, CFL_SIZE);

	// quickselect destroys dist_tmp
	conf->sigma = sqrtf(quickselect_complex(dist_tmp, L_dims[0] * L_dims[0], 1) * 2); 
	debug_printf(DP_INFO, "Estimated sigma: %f\n", conf->sigma);

	md_free(dist_tmp);

}

// account for anisotropic sampling
static void anisotropy_cor(const long W_dims[2], complex float* W)
{
	// Coifman, R. et al. "Geometric diffusions as a tool for harmonic
	// analysis and structure definition of data: Diffusion maps".
	// PNAS. 102 (21): 7426–7431. (2005)

	long Q_dims[2] = { W_dims[0], 1 };
	complex float* Q = md_alloc(2, Q_dims, CFL_SIZE);
	complex float* buf = md_alloc(2, W_dims, CFL_SIZE);
	
	// Q_i = sum_j W_ij
	md_zsum(2, W_dims, PHS1_FLAG, Q, W); 
	
	// W_ij = W_ij/(Q_i^\alpha * Q_j^\alpha)
	// to account for anisotropy: alpha = 1

	#pragma omp parallel for
	for (int i = 0; i < Q_dims[0]; i++)
		Q[i] = 1./Q[i];
	
	md_ztenmul(2, W_dims, buf, W_dims, W, Q_dims, Q);

	long Q2_dims[2] = { 1, W_dims[0] };
	md_ztenmul(2, W_dims, W, Q2_dims, Q, W_dims, buf);
}

// symmetrize matrix
static void symmetrize(const long A_dims[2], complex float* A)
{
		for (int i = 0; i < A_dims[0];  i++) {
			for (int j = i; j < A_dims[0]; j++) {

				if (A[i * A_dims[0] + j] == 0)
					A[i * A_dims[0] + j] = A[j * A_dims[0] + i];
				else
					A[j * A_dims[0] + i] = A[i * A_dims[0] + j];
			}
		}
}

// kernel_ij = exp(- (1/sigma^2) *  |src[i,:] - src[j,:]|^2 )
static void gauss_kernel(const long kernel_dims[2], complex float* kernel, const long src_dims[2], const complex float* src, struct laplace_conf* conf, bool normalize)
{
	
		long src2_dims[3] = { [0 ... 2] = 1 };
		md_copy_dims(2, src2_dims, src_dims);
		
		// src_sq = src * conj(src)
		complex float* src_sq = md_alloc(3, src2_dims, CFL_SIZE);
		md_zmulc(3, src2_dims, src_sq, src, src);
		
		long src_sum_dims[3] = { src_dims[0], 1, 1 };
		
		// src_sum = sum(src_sq, axis=1)	
		complex float* src_sum = md_alloc(3, src_sum_dims, CFL_SIZE);
		md_zsum(3, src2_dims, 2, src_sum, src_sq);	
		
		// cov = src @ conj(src.T)
		long cov_dims[3] = { [0 ... 2] = 1 };
		cov_dims[0] = src_dims[0];
		cov_dims[2] = src_dims[0];
		long cov_strs[3];
		md_calc_strides(3, cov_strs, cov_dims, CFL_SIZE);
		complex float* cov = md_alloc(3, cov_dims, CFL_SIZE);
		
		long src2_strs[3];
		md_calc_strides(3, src2_strs, src2_dims, CFL_SIZE);
		
		long src2T_dims[3] = { [0 ... 2] = 1 };
		src2T_dims[1] = src2_dims[1];
		src2T_dims[2] = src2_dims[0];
		
		long src2T_strs[3] = { 0 };
		src2T_strs[1] = src2_strs[1];
		src2T_strs[2] = src2_strs[0];
		
		long max_dims[3];
		md_tenmul_dims(3, max_dims, cov_dims, src2_dims, src2T_dims);	
		md_ztenmulc2(3, max_dims, cov_strs, cov, src2_strs, src, src2T_strs, src);

		
		// kernel = repmat(src_sum, src_dims[0], axis=1) + repmat(src_sum, src_dims[0], axis=1).T
		long src_sum_strs[3];
		md_calc_strides(3, src_sum_strs, src_sum_dims, CFL_SIZE);
		
		long src_sumT_strs[3] = { 0 };
		src_sumT_strs[0] = src_sum_strs[1];
		src_sumT_strs[2] = src_sum_strs[0];
		
		md_zadd2(3, cov_dims, cov_strs, kernel, src_sum_strs, src_sum, src_sumT_strs, src_sum);
		
		// kernel = exp( -(1 / sigma**2) * abs(kernel - 2 real(cov)) )
		md_zsmul(3, cov_dims, cov, cov, -2);
		md_zreal(3, cov_dims, cov, cov);
		md_zadd(3, cov_dims, kernel, kernel, cov);
		md_zabs(3, cov_dims, kernel, kernel);

		complex float* dist = NULL;
		if (conf->nn != -1) {
			
			// store distance array
			dist = md_alloc(2, kernel_dims, CFL_SIZE);
			md_copy(2, kernel_dims, dist, kernel, CFL_SIZE);
		}
			
		if (conf->sigma == -1)
			calc_sigma(kernel_dims, kernel, conf);
		
		
		if (normalize) {
			conf->median = crealf(median_complex_float(kernel_dims[0] * kernel_dims[1], kernel));
			md_zsmul(3, cov_dims, kernel, kernel, 1. / conf->median);
			debug_printf(DP_INFO, "median %f\n", conf->median);
		}
		
		
		md_zsmul(3, cov_dims, kernel, kernel, - 1./pow(conf->sigma,2));
		md_zexp(3, cov_dims, kernel, kernel);

		// keep only nn nearest neighbors
		if (conf->nn != -1) {

			complex float* buf = md_alloc(2, kernel_dims, CFL_SIZE);
			md_copy(2, kernel_dims, buf, dist, CFL_SIZE);

			float thresh;

			for (int i = 0; i < kernel_dims[0];  i++) {

				thresh = quickselect_complex(&buf[i * kernel_dims[0]], kernel_dims[0], kernel_dims[0] - conf->nn); // Get nn-th smallest distance. (Destroys dist_dump-array!)

				for (int j = 0; j < kernel_dims[0]; j++)
					kernel[i * kernel_dims[0] + j] *= (cabs(dist[i * kernel_dims[0] + j]) > thresh) ? 0 : 1;
			}

			// Symmetrize
			symmetrize(kernel_dims, kernel);

			md_free(dist);
			md_free(buf);
		}

		if (conf->anisotrop)
			anisotropy_cor(kernel_dims, kernel);

		assert(kernel_dims[0] == cov_dims[0]);
		assert(kernel_dims[1] == cov_dims[2]);
		
		md_free(src_sq);
		md_free(src_sum);
		md_free(cov);

}



void calc_laplace(struct laplace_conf* conf, const long L_dims[2], complex float* L, const long src_dims[2], const complex float* src)
{

	long D_dims[2] = { L_dims[0], 1 };
	complex float* D = md_alloc(2, D_dims, CFL_SIZE); // degree matrix

	complex float* W = md_alloc(2, L_dims, CFL_SIZE); // weight matrix


	char type;

	if (conf->kernel) 
		type = 'K';
	else if (conf->temporal_nn)
		type = 'N';
	else if (conf->kernel_CG)
		type = 'G';
	else
		type = 'C';

	switch (type) {

	case 'C': { // conventional

		gauss_kernel(L_dims, W, src_dims, src, conf, false);

		break;
	}
	
	case 'G': { // kernel CG approach

		long N = src_dims[0]; // number of observations (samples)

		complex float* src2 = md_alloc(2, src_dims, CFL_SIZE);
		md_copy(2, src_dims, src2, src, CFL_SIZE);

		complex float* kernel = md_alloc(2, L_dims, CFL_SIZE);
		complex float* kernel_cpy = md_alloc(2, L_dims, CFL_SIZE);
		complex float* V = md_alloc(2, L_dims, CFL_SIZE);
		complex float* VH = md_alloc(2, L_dims, CFL_SIZE);
		
		// Least-Squares
		bool gpu = 0;
		float cg_lambda = 0.001;
		struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
		const struct linop_s* matrix_op;

		float gamma = conf->kernel_gamma;
		float eta = 2;

		debug_printf(DP_DEBUG3, "CG \n");

		for (int i = 0; i < conf->iter_max; i++) {
			debug_printf(DP_DEBUG3, "...kernel iteration %d/%d \n", i + 1, conf->iter_max);
			
			gauss_kernel(L_dims, kernel, src_dims, src2, conf, (i == 0) ? true : false);

			calc_kernel_W(N, kernel, kernel_cpy, W, L, V, VH, gamma);

			if (i == conf->iter_max - 1)
				break;

			// D = sum(W, axis=1)
			md_zsum(2, L_dims, PHS1_FLAG, D, W);

			// L := diag(1) + lambda * (D - W)
			#pragma omp parallel for
			for (int l = 0; l < L_dims[0]; l++)
				L[l * L_dims[0] + l] = 1 + conf->kernel_lambda * ( D[l] - W[l * L_dims[0] + l] );

			/* src =  L @ src2    // y = F @ x
			 * ---------------
			 * L 	= [ A  A  1 ] // F
			 * src2 = [ 1  A  B ] // x
			 * src  = [ A  1  B ] // y
			 */

			long L_xdims[3] = { L_dims[0], L_dims[1], 1 };
			long src2_xdims[3] = { 1, src_dims[0], src_dims[1]};
			long src_xdims[3] = { src_dims[0], 1, src_dims[1] };

			long dims[3];
			md_max_dims(3, ~0lu, dims, L_xdims, src_xdims);

			matrix_op = linop_fmac_create(3, dims, MD_BIT(1), MD_BIT(0), MD_BIT(2), L);
		
			lsqr(3, &(struct lsqr_conf){ cg_lambda, gpu }, iter_conjgrad, CAST_UP(&cgconf),
			   	matrix_op, NULL, src2_xdims, src2, src_xdims, src, NULL);

			if (conf->median > 0)
				md_zsmul(2, src_dims, src2, src2, 1. / sqrtf(conf->median));

			if (gamma > 0.0001)
				gamma /= eta;

		}
	
		md_free(src2);
		md_free(kernel);
		md_free(kernel_cpy);
		md_free(V);
		md_free(VH);
		
		break;
	}

	case 'K': { // kernel approach
		// This is a reimplementation of the MATLAB code https://github.com/ahaseebahmed/SpiralSToRM/blob/master/
		// Note: we don't use double precicion!

		long N = src_dims[0]; // number of observations (samples)
		long M = src_dims[1]; // number of variables (coordinates)

		long V_dims[3] = { N, N, 1 };
		long V_strs[3];
		md_calc_strides(3, V_strs, V_dims, CFL_SIZE);


		long D_dims[3];
		md_select_dims(3, 1, D_dims, V_dims);

		long src2_dims[3] = { N, 1, M };
		long src3_dims[3] = { 1, N, M };
				
		complex float* V = md_alloc(3, V_dims, CFL_SIZE);
		complex float* VH = md_alloc(3, V_dims, CFL_SIZE);

		complex float* W1 = md_alloc(2, L_dims, CFL_SIZE);

		complex float* src2 = md_alloc(3, src2_dims, CFL_SIZE);

		complex float* kernel = md_alloc(2, L_dims, CFL_SIZE);
		complex float* kernel_cpy = md_alloc(2, L_dims, CFL_SIZE); // copy of gaussian kernel
		
		md_copy(2, src_dims, src2, src, CFL_SIZE);

		// iterations
		int iter_max = conf->iter_max;
		float gamma = 100.;
		float eta = 2.;
			
		for (int i = 0; i < iter_max; i++) {
			debug_printf(DP_DEBUG3, "...kernel iteration %d/%d \n", i + 1, iter_max);

			// calc Gaussian kernel
			gauss_kernel(L_dims, kernel, src_dims, src2, conf, (i == 0) ? true : false);

			calc_kernel_W(N, kernel, kernel_cpy, W, L, V, VH, gamma);

			if (i == iter_max - 1)
				break;

			// D = sum(W, axis=-1)
			md_zsum(3, V_dims, 2, D, W);

			// L = D - W
			#pragma omp parallel for
			for (int l = 0; l < V_dims[0]; l++)
				L[l * V_dims[0] + l] = D[l] - W[l * V_dims[0] + l];

			/* Update src2(=x) according to
			 *  argmin ||x - y||^2 + lambda * ||Ux||^2
			 *  x = (1 + lambda UU^H)^{-1} y
			 *  with UU^H the decomposition of the actual Laplacian
			*/

			// W = eye + kernel_lambda * L
			md_zsmul(3, V_dims, W, L, conf->kernel_lambda);
			
			#pragma omp parallel for
			for (int l = 0; l < V_dims[0]; l++)
				W[l * V_dims[0] + l] += 1;

			// W1 = inv(W)
			mat_inverse(N, (complex float (*)[N])W1, (complex float (*)[N])W);
		
			// src2 = src @ W1
			md_ztenmul(3, src2_dims, src2, V_dims, W1, src3_dims, src); // update the old signal 'src'

			if (conf->median > 0)
				md_zsmul(3, src2_dims, src2, src2, 1. / sqrtf(conf->median));
		
			if (gamma > conf->kernel_gamma)
				gamma /= eta;
				
		}
		
		md_free(W1);
		md_free(src2);
		md_free(V);
		md_free(VH);
		md_free(kernel_cpy);
		md_free(kernel);

		break;
	}

	case 'N': { // temporal nearest neighbour
		

		md_clear(2, L_dims, W, CFL_SIZE);
		long L_sdiag_dims[1] = { L_dims[0] - 1 };
		long L_sdiag_strs[1] = {(L_dims[0] + 1) * CFL_SIZE};
		md_zfill2(1, L_sdiag_dims, L_sdiag_strs, W + 1, 1.); // fill off-diagonal with ones

		symmetrize(L_dims, W);

		break;
	}

	} // end switch

	// Adjacency matrix (weight matrix) W must have zeros on main diagonal
	/* This is not necessary when using the definition L = D - W, 
	but it is important for P = D^{-1} @ W */
	#pragma omp parallel for
	for (int i = 0; i < L_dims[0]; i++)
			W[i + L_dims[0] * i] = 0;

	// D
	md_zsum(2, L_dims, PHS1_FLAG, D, W);

	if (conf->dmap) {


		// "L" := D^{-1} @ W  (=: P transition probability matrix)
		#pragma omp parallel for
		for (int i = 0; i < L_dims[0]; i++) {
		
			D[i] = 1. / D[i];	
		}

		md_ztenmul(2, L_dims, L, L_dims, W, D_dims, D);					

	} else {
	
		//L = D - W		
		md_zsmul(2, L_dims, L, W, -1.);

		#pragma omp parallel for
		for (int i = 0; i < L_dims[0]; i++)
			L[i * L_dims[0] + i] += D[i];
	}

	if (conf->norm){

		// L := D^{-1} @ L
		#pragma omp parallel for
		for (int i = 0; i < L_dims[0]; i++)
			D[i] = 1. / D[i];

		complex float* buf = md_alloc(2, L_dims, CFL_SIZE);

		md_copy(2, L_dims, buf, L, CFL_SIZE);
		md_ztenmul(2, L_dims, L, L_dims, buf, D_dims, D);

		md_free(buf);	
	} 

	md_free(D);
	md_free(W);
	
}

void kmeans(long centroids_dims[2], complex float* centroids, long src_dims[2], complex float* src, complex float* lables, const float eps, long update_max)
{
	long k = centroids_dims[1];
	long n_coord = src_dims[0];
	long n_samp = src_dims[1];

	long centroids_strs[2];
	md_calc_strides(2, centroids_strs, centroids_dims, CFL_SIZE);

	long sample_dims[2];
	sample_dims[0] = n_coord;
	sample_dims[1] = 1;

	long clustersize_dims[2];
	clustersize_dims[0] = 1;
	clustersize_dims[1] = k;

	long clustersize_strs[2];
	md_calc_strides(2, clustersize_strs, clustersize_dims, CFL_SIZE);

	complex float* centroids_buffer = md_alloc(2, centroids_dims, CFL_SIZE);
	complex float* sample = md_alloc(2, sample_dims, CFL_SIZE);
	complex float* sample_buffer = md_alloc(2, sample_dims, CFL_SIZE);
	complex float* centroid = md_alloc(2, sample_dims, CFL_SIZE);
	complex float* diff = md_alloc(2, sample_dims, CFL_SIZE);
	complex float* dist = malloc(sizeof(complex float));
	complex float* clustersize =  md_alloc(2, clustersize_dims, CFL_SIZE);

	float min_dist;
	long pos[2] = { 0 };
	long update_count = 0;
	float error_old = __FLT_MAX__;
	float error = 0;

	// Initialize centroids
	long delta = (long)floor(n_samp / k);
	for (int i = 0; i < k; i++)
		for (int j =0; j < n_coord; j++)
			centroids[i * n_coord + j] = src[i * delta * n_coord + j];


	long lable;

	// kmeans start
	while (fabs(error - error_old) / (1. * n_samp) > eps) {

		// clear
		error_old = error;
		error = 0;
		md_clear(2, clustersize_dims, clustersize, CFL_SIZE);
		md_clear(2, centroids_dims, centroids_buffer, CFL_SIZE);


		// for all samples...
		for (int l = 0; l < n_samp; l++) {

			min_dist = __FLT_MAX__;

			//... pick sample
			pos[1] = l;
			md_copy_block(2, pos, sample_dims, sample_buffer, src_dims, src, CFL_SIZE);

			// ... determine closest centroid
			for (int i = 0; i < k; i++) {
				pos[1] = i;
				md_copy_block(2, pos, sample_dims, centroid, centroids_dims, centroids, CFL_SIZE);
				md_zsub(2, sample_dims, diff, sample_buffer, centroid);
				md_zrss(2, sample_dims, 1, dist, diff);

				if (crealf(*dist) < min_dist) { // new closest centroid found
					min_dist = crealf(*dist);
					lables[l] = i + 0.i;
					md_copy(2, sample_dims, sample, sample_buffer, CFL_SIZE);
				}
			}

			lable = (long)lables[l];

			// ... account the sample's contribution to new centroid position
			md_zadd(2, sample_dims, centroids_buffer + (n_coord * lable), centroids_buffer + (n_coord * lable), sample);
			clustersize[lable] += 1. + 0.i;

			error += min_dist;

		}

		// ... account for averaging
		md_zdiv2(2, centroids_dims, centroids_strs, centroids, centroids_strs, centroids_buffer, clustersize_strs, clustersize);


		// prevent infinite loop
		update_count++;

		if (update_count > update_max) {
			debug_printf(DP_INFO, "Desired accuracy not reached within %d updates! \
			Try again with more updates [-u].\n  Relative error: %f\n", update_max, fabs(error - error_old) / (1. * n_samp));
			break;
		}

		debug_printf(DP_DEBUG3, "relative error: %f \n", fabs(error - error_old) / (1. * n_samp));
	}

	md_free(centroids_buffer);
	md_free(sample);
	md_free(sample_buffer);
	md_free(centroid);
	md_free(diff);
	md_free(clustersize);
	free(dist);

}
