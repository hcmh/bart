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

	.sigma 		= 0.1,
	.nn 		= -1,
};



void calc_laplace(const struct laplace_conf* conf, const long L_dims[2], complex float* L, const long src_dims[2], const complex float* src)
{

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
	}

	// L = D - W, with D[i,0] = sum(W[i,:])
	long D_dims[2];
	md_select_dims(2, READ_FLAG, D_dims, L_dims);
	complex float* D = md_alloc(2, D_dims, CFL_SIZE);
	md_zsum(2, L_dims, PHS1_FLAG, D, L);

	md_zsmul(2, L_dims, L, L, -1.); // -W

#pragma omp parallel for
	for (int i = 0; i < L_dims[0]; i++)
		L[i * L_dims[0] + i] += D[i];



	md_free(D);
	md_free(dist);

}
