/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019. Sebastian Rosenzweig.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Sebastian Rosenzweig (sebastian.rosenzweig@med.uni-goettingen.de)
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "manifold.h"

const struct laplace_conf laplace_conf_default = {

	.sigma 		= -1,
	.nn 		= -1,
	.temporal_nn = false,
	.gen_out	= false,
};



void calc_laplace(struct laplace_conf* conf, const long L_dims[2], complex float* L, const long src_dims[2], const complex float* src)
{

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
				L[idx * L_dims[0] + idx_past] += 1 + 0i;
			
			if (idx_future != idx)
				L[idx * L_dims[0] + idx_future] += 1 + 0i;


			
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
		for (int i = 0; i < L_dims[0]; i++)
			for (int j = 0; j <= i ; j++) {

				complex float* src_singleton = md_alloc(2, src_singleton_dims, CFL_SIZE);

				md_zsub2(2, src_singleton_dims, src_singleton_strs, src_singleton, src_singleton1_strs, &src[i], src_singleton1_strs, &src[j]);
				dist[i * L_dims[0] + j] = md_zscalar(2, src_singleton_dims, src_singleton, src_singleton) ;
				dist[j * L_dims[0] + i] = dist[i * L_dims[0] + j]; // exploit symmetry

				md_free(src_singleton);
			}

		if (conf->sigma == -1) {
			// Set sigma to maximum distance
			complex float* dist_tmp = md_alloc(2, L_dims, CFL_SIZE);
			md_copy(2, L_dims, dist_tmp, dist, CFL_SIZE);

			conf->sigma = sqrtf(quickselect_complex(dist_tmp, L_dims[0] * L_dims[0], 1)); // Quickselect destroys dist_tmp
			debug_printf(DP_INFO, "Estimated sigma: %f\n", conf->sigma);

			md_free(dist_tmp);
		}

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
			D[i] = 1./D[i];

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

		// Clear
		error_old = error;
		error = 0;
		md_clear(2, clustersize_dims, clustersize, CFL_SIZE);
		md_clear(2, centroids_dims, centroids_buffer, CFL_SIZE);


		// For all samples...
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
					lables[l] = i + 0i;
					md_copy(2, sample_dims, sample, sample_buffer, CFL_SIZE);
				}
			}

			lable = (long)lables[l];

			// ... account the sample's contribution to new centroid position
			md_zadd(2, sample_dims, centroids_buffer + (n_coord * lable), centroids_buffer + (n_coord * lable), sample);
			clustersize[lable] += 1. + 0j;

			error += min_dist;

		}

		// ... account for averaging
		md_zdiv2(2, centroids_dims, centroids_strs, centroids, centroids_strs, centroids_buffer, clustersize_strs, clustersize);


		// Prevent infinite loop
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
