/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019 Nick Scholand
 */


#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Create linear rising array.";


int main_linspace(int argc, char* argv[argc])
{
	float start = 0.f;
	float diff = 0.f;
	int num = -1;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_FLOAT(true, &start, "start"),
		ARG_FLOAT(true, &diff, "diff/end"),
		ARG_INT(true, &num, "num"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	enum linspace_type { LINSPACE_DIFF, LOGSPACE_END_INCLUDED };
	enum linspace_type type = LINSPACE_DIFF;

	const struct opt_s opts[] = {

		OPTL_SELECT(0, "logspace", enum linspace_type, &type, LOGSPACE_END_INCLUDED, "logarithmic spacing (invcluding end value)")
	};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();



	long dims[DIMS] = { [0 ... DIMS - 1] = 1. };
	dims[0] = num;

	complex float* out = create_cfl(out_file, DIMS, dims);

	switch (type) {

		case LINSPACE_DIFF:

		for (int i = 0; i < num; i++)
			out[i] = start + i * diff;
		break;

		case LOGSPACE_END_INCLUDED:

		for (int i = 0; i < num; i++)
			out[i] = expf(logf(start) + i * (logf(diff) - logf(start)) / (num - 1));
		break;
	}


	unmap_cfl(DIMS, dims, out);
	return 0;
}


