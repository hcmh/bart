/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mmio.h"

#include "nn/nn.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<input> <kernel> <bias> <output>";
static const char help_str[] = "Applies a pre-trained convolutional neural network.";





int main_dcnn(int argc, char* argv[])
{
	cmdline(&argc, argv, 4, 4, usage_str, help_str, 0, NULL);

	num_init();


	unsigned int N = DIMS;
	long dims[N];
	const complex float* in = load_cfl(argv[1], N, dims);

	long krn_dims[N];
	const complex float* krn = load_cfl(argv[2], N, krn_dims);

	long bias_dims[N];
	const complex float* bias = load_cfl(argv[3], N, bias_dims);

	complex float* out = create_cfl(argv[4], N, dims);

	simple_dcnn(dims, krn_dims, krn, bias_dims, bias, out, in);

	md_zsub(N, dims, out, in, out);

	unmap_cfl(N, dims, out);
	unmap_cfl(N, krn_dims, krn);
	unmap_cfl(N, bias_dims, bias);
	unmap_cfl(N, dims, in);
	exit(0);
}

