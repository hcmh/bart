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

typedef enum {SOBEL, GAUSS} kernel_type;

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

// Tensor multiplication of 1D kernels to create multi-dimensional kernel
static void meshmul(const long mesh_dims[3], complex float* mesh, 
					const long m0_dims[3], const complex float* m0,
					const long m1_dims[3], const complex float* m1, 
					const long m2_dims[3], const complex float* m2,
					const long N)
{

	long max_dims[3] = { 1, 1, 1};
	long max2_dims[3] = { 1, 1, 1};	

	switch (N) {

		case 1: {  // 1D
			assert(mesh_dims[0] > 1);
			assert((mesh_dims[1] == 1) && (mesh_dims[2] == 1));

			md_copy(3, mesh_dims, mesh, m0, CFL_SIZE);
			break;
		}
		case 2: { // 2D
			assert((mesh_dims[0] == mesh_dims[1]) && (mesh_dims[2] == 1));	
			assert(mesh_dims[0] > 1);
			
			md_max_dims(3, (MD_BIT(0)|MD_BIT(1)), max_dims, m0_dims, m1_dims);
			md_ztenmul(3, max_dims, mesh, m1_dims, m1, m0_dims, m0);		
			break;
		}
		case 3: { // 3D
			assert((mesh_dims[0] == mesh_dims[1]) && (mesh_dims[1] == mesh_dims[2]));	
			assert(mesh_dims[0] > 1);

			complex float* buf = md_alloc(3, mesh_dims, CFL_SIZE);
			md_max_dims(3, (MD_BIT(0)|MD_BIT(1)), max_dims, m0_dims, m1_dims);
			md_ztenmul(3, max_dims, buf, m1_dims, m1, m0_dims, m0);

			md_max_dims(3, (MD_BIT(0)|MD_BIT(1)|MD_BIT(2)), max2_dims, max_dims, m2_dims);
			md_ztenmul(3, max2_dims, mesh, max_dims, buf, m2_dims, m2);

			md_free(buf);				
			break;
		}
	}

	return;
}				

// creates kernel of type 'type' and dimension 'N'
static void kernelgrid(long mesh_dims[3], complex float* mesh, long N, kernel_type type)
{

	long m0_dims[3] = { mesh_dims[0], 1, 1};
	long m1_dims[3] = { 1, mesh_dims[1], 1};
	long m2_dims[3] = { 1, 1,  mesh_dims[2]};
	complex float* m = md_alloc(3, m0_dims, CFL_SIZE);

	// 1D basis
	long max;
	max = mesh_dims[0];
	assert((max % 2) == 1);
	for (int i = 0; i < max; i++)
		m[i] = - max/2 + i;

	switch (type) {

		case GAUSS: {
			
			float sigma = 1;
			
			md_zspow(3, m0_dims, m, m, 2);
			md_zsmul(3, m0_dims, m, m, -1./(2 * pow(sigma, 2)));
			md_zexp(3, m0_dims, m, m);	

			meshmul(mesh_dims, mesh, m0_dims, m, m1_dims, m, m2_dims, m, N);				

			// normalize sum of entries to 1
			complex float sum;
			md_zsum(3, mesh_dims, ~0u, &sum, mesh);
			md_zsmul(3, mesh_dims, mesh, mesh, 1. / sum);
			break;				
		} // end GAUSS

		case SOBEL: {
			
			assert(m0_dims[0] == 3);
			assert(3 == md_calc_size(3, m0_dims));
			
			if (N > 1) {
				assert(m1_dims[1] == 3);
				assert(3 == md_calc_size(3, m1_dims));
			}

			complex float* m1 = md_alloc(3, m1_dims, CFL_SIZE);
			
			m1[0] = - 1 + 0i;
			m1[1] = - 2 + 0i;
			m1[2] = - 1 + 0i;
			
			meshmul(mesh_dims, mesh, m0_dims, m, m1_dims, m1, m2_dims, m1, N);				

			md_free(m1);
			break;
		} // end SOBEL
	}

	md_free(m);
	
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

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long N = (long)atoi(argv[1]); // kernel dimensions
	long dims[3] = { 1, 1, 1};
	float sigma = 1;

	enum { SOBEL, GAUSS } type = SOBEL;


	if (sobel) {
		
		type = SOBEL;	
		for (int i = 0; i < N; i++)
			dims[i] = 3;

	} else if (gauss_len > 0) {

		type = GAUSS;
		for (int i = 0; i < N; i++)
			dims[i] = gauss_len;
	}
	
	complex float* kernel = create_cfl(argv[2], 3, dims);

	switch (type) {

		case SOBEL: {

			kernelgrid(dims, kernel, N, SOBEL);
			break;
		} 
		case GAUSS: {

			kernelgrid(dims, kernel, N, GAUSS);
			break;
		}
	}

	unmap_cfl(3, dims, kernel);
	return 0;
}





