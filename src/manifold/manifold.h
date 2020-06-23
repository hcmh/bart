/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019. Sebastian Rosenzweig
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Sebastian Rosenzweig (sebastian.rosenzweig@med.uni-goettingen.de)

 */

#include "misc/mri.h"

struct laplace_conf {

	int nn; 	 		// number of nearest neighbours
	float sigma;	 	// standard deviation
	_Bool temporal_nn; 	// Laplacian for temporal nearest neigbours
	_Bool kernel;		// kernel approach
	float kernel_lambda;// kernel lambda weighting
	_Bool norm;	 		// output D^-1 @ L (output normalized Laplacian, where L = D - W)
	_Bool anisotrop;	// anisotropic sampling
	_Bool dmap;			// diffusion map (output transition probability matrix P = D^{-1}W)
	float median; 		// median normalization for kernel approach
	int iter_max;	 	// kernel approach: iterations

};

extern const struct laplace_conf laplace_conf_default;

extern void calc_laplace(struct laplace_conf* conf, const long L_dims[2], complex float* L, const long src_dims[2], const complex float* src);
extern void kmeans(long centroids_dims[2], complex float* centroids, long src_dims[2], complex float* src, complex float* lables, const float eps, const long update_max);
