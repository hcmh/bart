/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <complex.h>

#include "num/init.h"
#include "num/multind.h" // MD_BIT

#include "na/na.h"
#include "na/io.h"

#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Extracts a slice from positions along dimensions.";


int main_slice(int argc, char* argv[argc])
{
	long count = 0;
	long* dims = NULL;
	long* poss = NULL;

	const char* in_file = NULL;
	const char* out_file = NULL;


	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 2, OPT_LONG, sizeof(long), &dims, "dim", OPT_LONG, sizeof(long), &poss, "pos"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	na in = na_load(in_file);

	long pos2[na_rank(in)];

	for (unsigned int i = 0; i < na_rank(in); i++)
		pos2[i] = 0;

	unsigned long flags = 0L;

	for (int i = 0; i < count; i++) {

		int dim = dims[i];
		int pos = poss[i];

		assert(dim >= 0);
		assert(pos >= 0);
		assert(dim < (int)na_rank(in));
		assert(pos < (*NA_DIMS(in))[dim]);

		flags = MD_SET(flags, dim);
		pos2[dim] = pos;
	}

	na sl = na_slice(in, ~flags, na_rank(in), &pos2);
	na out = na_create(out_file, na_type(sl));
	na_copy(out, sl);

	na_free(sl);
	na_free(out);
	na_free(in);

	xfree(dims);
	xfree(poss);

	return 0;
}


