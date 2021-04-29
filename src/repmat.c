/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Author:
 * 2012-2020 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/lang.h"
#include "misc/opts.h"


#define DIMS 16

static const char help_str[] = "Repeat input array multiple times along a certain dimension.";


int main_repmat(int argc, char* argv[argc])
{
	int dim = -1;
	int rep = -1;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &dim, "dimension"),
		ARG_INT(true, &rep, "repetitions"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};
	const struct opt_s opts[] = {};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long in_dims[DIMS];
	long out_dims[DIMS];
	
	with (complex float* in_data = load_cfl(in_file, DIMS, in_dims)
		;; unmap_cfl(DIMS, in_dims, in_data)) {

		assert(dim < DIMS);
		assert(rep >= 0);
		assert(1 == in_dims[dim]);

		md_copy_dims(DIMS, out_dims, in_dims);

		out_dims[dim] = rep;

		with (complex float* out_data = create_cfl(out_file, DIMS, out_dims)
			;; unmap_cfl(DIMS, in_dims, out_data)) {

			long in_strs[DIMS];
			long out_strs[DIMS];

			md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);
			md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);

			md_copy2(DIMS, out_dims, out_strs, out_data, in_strs, in_data, CFL_SIZE);
		}
	}

	return 0;
}


