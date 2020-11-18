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

static const char usage_str[] = "dims <output>";
static const char help_str[] = "Compute convolution kernel";

typedef enum {SOBEL, GAUSS} kernel_type;

struct kernel_conf {
	long N; 		// dimension
	float sigma;	// standart deviation 
};

const struct kernel_conf kernel_conf_default = {

	.N  			= 1,
	.sigma 			= 1,
};

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
static void kernelgrid(long mesh_dims[3], complex float* mesh, kernel_type type, const struct kernel_conf* conf)
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
					
			md_zspow(3, m0_dims, m, m, 2);
			md_zsmul(3, m0_dims, m, m, -1./(2 * pow(conf->sigma, 2)));
			md_zexp(3, m0_dims, m, m);	

			meshmul(mesh_dims, mesh, m0_dims, m, m1_dims, m, m2_dims, m, conf->N);				

			// normalize sum of entries to 1
			complex float sum;
			md_zsum(3, mesh_dims, ~0u, &sum, mesh);
			md_zsmul(3, mesh_dims, mesh, mesh, 1. / sum);
			break;				
		} // end GAUSS

		case SOBEL: {
			
			assert(m0_dims[0] == 3);
			assert(3 == md_calc_size(3, m0_dims));
			
			if (conf->N > 1) {
				assert(m1_dims[1] == 3);
				assert(3 == md_calc_size(3, m1_dims));
			}

			complex float* m1 = md_alloc(3, m1_dims, CFL_SIZE);
			
			m1[0] = 1;
			m1[1] = 2;
			m1[2] = 1;
			
			meshmul(mesh_dims, mesh, m0_dims, m, m1_dims, m1, m2_dims, m1, conf->N);				

			md_free(m1);
			break;
		} // end SOBEL
	}

	md_free(m);
	
	return;
}



int main_convkern(int argc, char* argv[])
{

	float gauss[2] = { 0, 1};
	bool sobel = false;
	struct kernel_conf conf = kernel_conf_default;

	const struct opt_s opts[] = {

		OPT_SET('s', &sobel, "Sobel kernel"),
		OPT_FLVEC2('g', &gauss, "len:sigma", "Gaussian kernel"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	conf.N = (long)atoi(argv[1]); // kernel dimensions
	long dims[3] = { 1, 1, 1};
	
	enum { SOBEL, GAUSS } type = SOBEL;


	if (sobel) {
		
		type = SOBEL;	
		for (int i = 0; i < conf.N; i++)
			dims[i] = 3;

	} else if ((int)gauss[0] > 0) {

		type = GAUSS;
		conf.sigma = gauss[1];
		for (int i = 0; i < conf.N; i++)
			dims[i] = (int)gauss[0];
		
	}
	
	complex float* kernel = create_cfl(argv[2], 3, dims);

	switch (type) {

		case SOBEL: {

			kernelgrid(dims, kernel, SOBEL, &conf);
			break;
		} 
		case GAUSS: {

			kernelgrid(dims, kernel, GAUSS, &conf);
			break;
		}
	}

	unmap_cfl(3, dims, kernel);
	return 0;
}





