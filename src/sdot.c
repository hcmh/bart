/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Compute dot product along selected dimensions.";

int main_sdot(int argc, char* argv[argc])
{
	const char* in1_file = NULL;
	const char* in2_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in1_file, "input1"),
		ARG_INFILE(true, &in2_file, "input2"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long in1_dims[N];
	long in2_dims[N];

	complex float* in1_data = load_cfl(in1_file, N, in1_dims);
	complex float* in2_data = load_cfl(in2_file, N, in2_dims);


	for (int i = 0; i < N; i++)
		if (in1_dims[i] != in2_dims[i])
			error("Dimenions %d does not match", i);

	// compute scalar product
	complex float value = md_zscalar(N, in1_dims, in1_data, in2_data);

	bart_printf("%+e%+ei\n", crealf(value), cimagf(value));

	unmap_cfl(N, in1_dims, in1_data);
	unmap_cfl(N, in2_dims, in2_data);

	return 0;
}


