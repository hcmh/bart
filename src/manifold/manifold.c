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

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/filter.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "manifold.h"

const struct laplace_conf laplace_conf_default = {

	.sigma 			= -1,
	.nn 			= -1,
	.temporal_nn 	= false,
	.kernel     	= false,
	.kernel_lambda 	= 0.3,
	.gen_out		= false,
	.median 		= -1,

};

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

// kernel_ij = exp(- (1/sigma^2) *  |src[i,:] - src[j,:]|^2 )
static void gauss_kernel(const long kernel_dims[2], complex float* kernel, const long src_dims[2], const complex float* src, struct laplace_conf* conf, bool normalize)
{
	
		long src2_dims[3] = { [0 ... 2] = 1 };
		md_copy_dims(2, src2_dims, src_dims);
		
		// src_sq = src * conj(src)
		complex float* src_sq = md_alloc(3, src2_dims, CFL_SIZE);
		md_zmulc(3, src2_dims, src_sq, src, src);
		
		long src_sum_dims[3] = { [0 ... 2] = 1};
		src_sum_dims[0] = src_dims[0];
		src_sum_dims[1] = 1;
		
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
		
		if (conf->sigma == -1)
			calc_sigma(kernel_dims, kernel, conf);
		
		
		if (normalize) {
			conf->median = crealf(median_complex_float(kernel_dims[0] * kernel_dims[1], kernel));
			md_zsmul(3, cov_dims, kernel, kernel, 1. / conf->median);
			debug_printf(DP_INFO, "median %f\n", conf->median);
		}
		
		
		md_zsmul(3, cov_dims, kernel, kernel, - 1./pow(conf->sigma,2));
		md_zexp(3, cov_dims, kernel, kernel);
		
		assert(kernel_dims[0] == cov_dims[0]);
		assert(kernel_dims[1] == cov_dims[2]);
		
		md_free(src_sq);
		md_free(src_sum);
		md_free(cov);

}



void calc_laplace(struct laplace_conf* conf, const long L_dims[2], complex float* L, const long src_dims[2], const complex float* src)
{

	if (conf->kernel) { // kernel approach
#if 0		
		
		long N = src_dims[0]; // number of samples
		complex float* kernel = md_alloc(2, L_dims, CFL_SIZE);
		gauss_kernel(L_dims, kernel, src_dims, src, conf, true);
		
		long cov_dims[3];
		cov_dims[0] = N;
		cov_dims[1] = 1;
		cov_dims[2] = N;
		
		complex float* kernel_cpy = md_alloc(2, L_dims, CFL_SIZE); // copy of gaussian kernel
		md_copy(2, L_dims, kernel_cpy, kernel, CFL_SIZE);

		// V, S, VH = svd(kernel)
		long V_dims[3] = { [0 ... 2] = 1 };
		V_dims[0] = N;
		V_dims[1] = N;
		
		long V_strs[3];
		md_calc_strides(3, V_strs, V_dims, CFL_SIZE);
		
		complex float* V = md_alloc(3, V_dims, CFL_SIZE);
		complex float* VH = md_alloc(3, V_dims, CFL_SIZE);
		float* Sf = xmalloc(N * sizeof(float));
		
		lapack_svd(N, N, (complex float (*)[N])V, (complex float (*)[N])VH, Sf, (complex float (*)[N])kernel_cpy); // NOTE: Lapack destroys kernel_cpy!

		complex float* VH_tmp = md_alloc(3, V_dims, CFL_SIZE);
		md_zconj(3, V_dims, VH_tmp, V);
		md_transpose(3, 0, 1, V_dims, VH, V_dims, VH_tmp, CFL_SIZE);
		
		// iterations
		int iter_max = 30;
		float gamma = 100.;
		float eta = 2.;
	
		long S_dims[1];
		S_dims[0] = N;
		
		long D_dims[3];
		md_select_dims(3, 1, D_dims, V_dims);
		
		long VH_dims[3] = { [0 ... 2] = 1 };
		VH_dims[1] = N;
		VH_dims[2] = N;
		
		long S_strs[3] = { 0 };
		S_strs[1] = CFL_SIZE;
		
		long src2_dims[3] = { [0 ... 2] = 1 };
		src2_dims[0] = src_dims[0];
		src2_dims[2] = src_dims[1];
		
		long src3_dims[3] = { [0 ... 2] = 1 };
		src3_dims[1] = src_dims[0];
		src3_dims[2] = src_dims[1];
		
		complex float* D = md_alloc(3, D_dims, CFL_SIZE);
		complex float* S_inv = md_alloc(1, S_dims, CFL_SIZE);	
		complex float* W = md_alloc(3, cov_dims, CFL_SIZE);
		complex float* W1 = md_alloc(3, cov_dims, CFL_SIZE);
		complex float* src2 = md_alloc(3, src2_dims, CFL_SIZE);
			
		for (int i = 0; i < iter_max; i++) {
			
			// W = V @ (S + eye * gamma)^(-0.5) @ VH
			for (int j = 0; j < N; j++)
				S_inv[j] = pow((Sf[j] + gamma), -0.5) + 0 * 1.i;
			
			md_zmul2(3, V_dims, V_strs, W1, V_strs, V, S_strs, S_inv);
			md_ztenmul(3, cov_dims, W, V_dims, W1, VH_dims, VH);

			// L = kernel * W
			md_zmul(3, V_dims, L, kernel, W);
			
			// D = sum(W, axis=-1)
			md_zsum(3, V_dims, 2, D, L);

			// L = -D + L (negative Laplacian)
			#pragma omp parallel for
			for (int l = 0; l < V_dims[0]; l++)
				L[l * V_dims[0] + l] -= D[l];
			
			// W = eye + kernel_lambda * W
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
			

			// calc Gaussian kernel
			gauss_kernel(L_dims, kernel, src_dims, src2, conf, false);

			// SVD
			md_copy(3, cov_dims, kernel_cpy, kernel, CFL_SIZE);
			lapack_svd_double_real(N, N, (complex float (*)[N])V, (complex float (*)[N])VH, Sf, (complex float (*)[N])kernel_cpy); // NOTE: Lapack destroys kernel_cpy!
			/* TODO: There seem to be numerical instabilities compared to the matlab code. 
			 * Matlab uses complex doubles and - if the imaginary part is zero - doubles 
			 * for all operations. This should be tested here!*/

			md_zconj(3, V_dims, VH_tmp, V);
			md_transpose(3, 0, 1, V_dims, VH, V_dims, VH_tmp, CFL_SIZE);
		
			gamma /= eta;
			debug_printf(DP_INFO, "Gamma: %f\n", gamma);
		}
		
		md_free(D);
		md_free(S_inv);
		md_free(W);
		md_free(W1);
		md_free(src2);
		md_free(V);
		md_free(VH);
		md_free(kernel_cpy);
		md_free(kernel);
		md_free(VH_tmp);
		xfree(Sf);
#endif	

	} else { 
	
		if (conf->temporal_nn) { // Laplace for temporal nearest neigbours

			md_clear(2, L_dims, L, CFL_SIZE);

			assert(src_dims[1] == 1);

			int idx_past;
			int idx;
			int idx_future;

			for (int l = 0; l < src_dims[0]; l++) {

				if (creal(src[l]) > L_dims[0])
					error("Lable index larger than Laplacian size!");

				idx = creal(src[l]);

				if (l > 0)
					idx_past = creal(src[l - 1]);
				else
					idx_past = idx;			

				if (l < src_dims[0] - 1)
					idx_future = creal(src[l + 1]);
				else
					idx_future = idx;


				if (idx_past != idx)
					L[idx * L_dims[0] + idx_past] += 1 + 0.i;

				if (idx_future != idx)
					L[idx * L_dims[0] + idx_future] += 1 + 0.i;
			}

			// Symmetrize
			for (int i = 0; i < L_dims[0];  i++) {
				for (int j = i; j < L_dims[0]; j++) {

					if (L[i * L_dims[0] + j] == 0)
						L[i * L_dims[0] + j] = L[j * L_dims[0] + i];
					else
						L[j * L_dims[0] + i] = L[i * L_dims[0] + j];
				}
			}

		} else { // Conventional Laplace calculation
			
			//TODO: Test if gauss_kernel() function is faster than this implementation
			
			// L[i,j,:] = src[i,:] - src[j,:]
			assert(L_dims[0] == src_dims[0]);

			complex float* dist = md_alloc(2, L_dims, CFL_SIZE);

			long src_strs[2];
			md_calc_strides(2, src_strs, src_dims, CFL_SIZE);

			long src_singleton_dims[2];
			src_singleton_dims[0] = 1;
			src_singleton_dims[1] = src_dims[1];

			long src_singleton_strs[2];
			md_calc_strides(2, src_singleton_strs, src_singleton_dims, CFL_SIZE);

			long src_singleton1_strs[2];
			src_singleton1_strs[0] = 0;
			src_singleton1_strs[1] = src_strs[1];

			// dist[i,j] = ||src[i,:] - src[j,:]||^2
			#pragma omp parallel for
			for (int i = 0; i < L_dims[0]; i++) {

				for (int j = 0; j <= i; j++) {

					complex float* src_singleton = md_alloc(2, src_singleton_dims, CFL_SIZE);

					md_zsub2(2, src_singleton_dims, src_singleton_strs, src_singleton, src_singleton1_strs, &src[i], src_singleton1_strs, &src[j]);

					dist[i * L_dims[0] + j] = md_zscalar(2, src_singleton_dims, src_singleton, src_singleton) ;
					dist[j * L_dims[0] + i] = dist[i * L_dims[0] + j]; // exploit symmetry

					md_free(src_singleton);
				}
			}

			if (conf->sigma == -1)
				calc_sigma(L_dims, dist, conf);

			// W = exp(- dist^2 / sigma^2)
			md_zsmul(2, L_dims, L, dist, -1. /  pow(conf->sigma,2));
			md_zexp(2, L_dims, L, L);

			// Keep only nn-th nearest neighbours
			if (conf->nn != -1) {

				complex float* dist_dump = md_alloc(2, L_dims, CFL_SIZE);

				md_copy(2, L_dims, dist_dump, dist, CFL_SIZE);

				float thresh;

				for (int i = 0; i < L_dims[0];  i++) {

					thresh = quickselect_complex(&dist_dump[i * L_dims[0]], L_dims[0], L_dims[0] - conf->nn); // Get nn-th smallest distance. (Destroys dist_dump-array!)

					for (int j = 0; j < L_dims[0]; j++)
						L[i * L_dims[0] + j] *= (cabs(dist[i * L_dims[0] + j]) > thresh) ? 0 : 1;
				}

				md_free(dist_dump);

				// Symmetrize
				for (int i = 0; i < L_dims[0];  i++) {

					for (int j = i; j < L_dims[0]; j++) {

						if (L[i * L_dims[0] + j] == 0)
							L[i * L_dims[0] + j] = L[j * L_dims[0] + i];
						else
							L[j * L_dims[0] + i] = L[i * L_dims[0] + j];
					}
				}
			}

			md_free(dist);
		}

		// D[i,0] = sum(W[i,:])
		long D_dims[2];
		md_select_dims(2, READ_FLAG, D_dims, L_dims);

		complex float* D = md_alloc(2, D_dims, CFL_SIZE);

		md_zsum(2, L_dims, PHS1_FLAG, D, L); // D is diagonal

		if (conf->gen_out) {

			// L := D^{-1} @ W

			#pragma omp parallel for
			for (int i = 0; i < L_dims[0]; i++)
				D[i] = 1. / D[i];

			complex float* tmp = md_alloc(2, L_dims, CFL_SIZE);

			md_copy(2, L_dims, tmp, L, CFL_SIZE);

			md_ztenmul(2, L_dims, L, L_dims, tmp, D_dims, D);

			md_free(tmp);

		} else {

			//L = D - W

			md_zsmul(2, L_dims, L, L, -1.); // -W

			#pragma omp parallel for
			for (int i = 0; i < L_dims[0]; i++)
				L[i * L_dims[0] + i] += D[i];
		}

		md_free(D);
}

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
