/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include <complex.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "dim1 start1 end1 stepsize1... dimn startn endn stepsizen <input> <output>";
static const char help_str[] = "Extracts a sub-array along dims from index start to (not including) end.\n";


int main_extract(int argc, char* argv[])
{
	const struct opt_s opts[] = {};

	cmdline(&argc, argv, 5, 1000, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long in_dims[DIMS];
	long out_dims[DIMS];
	
	complex float* in_data = load_cfl(argv[argc - 2], DIMS, in_dims);
	md_copy_dims(DIMS, out_dims, in_dims);

	int count = argc - 3;
	assert((count > 0) && (count % 4 == 0));


	long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };

	long in_strs[DIMS];
	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

	for (int i = 0; i < count; i += 4) {

		int dim = atoi(argv[i + 1]);
		int start = atoi(argv[i + 2]);
		int end = atoi(argv[i + 3]);
		int stepsize = atoi(argv[i + 4]);

		assert((0 <= dim) && (dim < DIMS));
		assert(start >= 0);
		assert(start < end);
		assert(end <= in_dims[dim]);
		assert(stepsize > 0);

		out_dims[dim] = (end - start)/stepsize;
		pos2[dim] = start;
		in_strs[dim] = in_strs[dim]*stepsize;
	}


	complex float* out_data = create_cfl(argv[argc - 1], DIMS, out_dims);

	long out_strs[DIMS];
	md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);
	
	md_copy_block2(DIMS, pos2, out_dims, out_strs, out_data, in_dims, in_strs, in_data, CFL_SIZE);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	return 0;
}


