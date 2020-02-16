/* Copyright 2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Sebastian Rosenzweig (sebastian.rosenzweig@med.uni-goettingen.de)
 */

#include "misc/cppwrap.h"
#include "misc/mri.h"

struct laplace_conf {

	int nn; 	 // number of nearest neighbours
	float sigma;	 // Standard deviation
	_Bool temporal_nn; // Laplacian for temporal nearest neigbours
	_Bool gen_out;	 // Output D^-1 @ W (For caclualtion of generalized Laplacian EV's v: Lv = Dv)
};

extern const struct laplace_conf laplace_conf_default;

extern void calc_laplace(struct laplace_conf* conf, const long L_dims[2], complex float* L, const long src_dims[2], const complex float* src);
extern void kmeans(long centroids_dims[2], complex float* centroids, long src_dims[2], complex float* src, complex float* lables, const float eps, const long update_max);
