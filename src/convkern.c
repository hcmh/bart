/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<output>";
static const char help_str[] = "Compute convolution kernel";

static void meshgrid(long N, complex float* mesh, long type)
{
	assert((N % 2) == 1);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (type == 0)
				mesh[i + j * N ] = - N/2 + i;
			else if (type == 1)
				mesh[i * N + j] = - N/2 + i;
	
	return;
}


int main_convkern(int argc, char* argv[])
{

	long gauss_len = 0;
	bool sobel = false;

	const struct opt_s opts[] = {

		OPT_SET('s', &sobel, "Sobel kernel"),
		OPT_LONG('g', &gauss_len, "len", "Gaussian kernel"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	enum { SOBEL, GAUSS } type = 0;

	float sigma = 1;
	long dims[3] = { 1, 1, 1};


	if (sobel) {
		
		type = SOBEL;	
		dims[0] = 3;
		dims[1] = 3;
		dims[2] = 2;

	} else if (gauss_len > 0) {

		type = GAUSS;

		for (int i = 0; i < 2; i++)
			dims[i] = gauss_len;
	}
	
	complex float* kernel = create_cfl(argv[1], 3, dims);

	switch (type) {

		case SOBEL: {

			assert(18 == md_calc_size(3, dims));

			meshgrid(dims[0], kernel, 0); // Sobel x
			meshgrid(dims[0], kernel + 9, 1); // Sobel y

			break;
		} 
		case GAUSS: {

			complex float* mesh_x = md_alloc(3, dims, CFL_SIZE);
			complex float* mesh_y = md_alloc(3, dims, CFL_SIZE);
			meshgrid(dims[0], mesh_x, 0);
			meshgrid(dims[1], mesh_y, 1);

			md_zspow(3, dims, mesh_x, mesh_x, 2);
			md_zspow(3, dims, mesh_y, mesh_y, 2);
			md_zadd(3, dims, kernel, mesh_x, mesh_y);
			md_zsmul(3, dims, kernel, kernel, -1./(2 * pow(sigma, 2)));
			md_zexp(3, dims, kernel, kernel);

			// normalize s.t. sum of entries is 1
			complex float sum;
			md_zsum(3, dims, ~0u, &sum, kernel);
			md_zsmul(3, dims, kernel, kernel, 1. / sum);

			md_free(mesh_x);
			md_free(mesh_y);
			break;
		}
	}

	unmap_cfl(3, dims, kernel);


	return 0;
}





