/* Copyright 2018. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2018 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "num/init.h"
#include "num/multind.h"
#include "num/flpmath.h"


#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "manifold/manifold.h"


#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<input> <centroids> <lables>";
static const char help_str[] =	"k-means clustering. <input>: [coordinates, samples]\n";


int main_kmeans(int argc, char* argv[])
{
	long k = 0; 	// Number of clusters. (Please note that "k" is not the k-space here!)
	float eps = 0.001; // Error tolerance
	long update_max = 10000;

	const struct opt_s opts[] = {

		OPT_LONG('k', &k, "k", "Number of clusters"),
		OPT_LONG('u', &update_max, "u", "(Maximum number of updates)"),
	        OPT_FLOAT('e', &eps, "eps", "(Error tolerance)"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (k == 0)
		error("Please specify 'k'!\n");

	long src_dims[2]; 	// [coordinates, samples]
	complex float* src = load_cfl(argv[1], 2, src_dims);

	// Coordinates for each centroid
	long centroids_dims[2];
	centroids_dims[0] = src_dims[0];
	centroids_dims[1] = k;		// Centroid index

	complex float* centroids = create_cfl(argv[2], 2, centroids_dims);

	// Contains the assigned centroid for each sample
	long lables_dims[1] = { src_dims[1] };
	complex float* lables = create_cfl(argv[3], 1, lables_dims);

	kmeans(centroids_dims, centroids, src_dims, src, lables, eps, update_max);

	unmap_cfl(1, lables_dims, lables);
	unmap_cfl(2, centroids_dims, centroids);

	return 0;
}


