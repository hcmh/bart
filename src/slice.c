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
#include "misc/io.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char usage_str[] = "dim1 pos1 ... dimn posn <input> <output>";
static const char help_str[] = "Extracts a slice from positions along dimensions.\n";


int main_slice(int argc, char* argv[])
{
	const struct opt_s opts[] = { };

	cmdline(&argc, argv, 4, 1000, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int count = argc - 3;
	assert((count > 0) && (count % 2 == 0));

	na in = na_load(argv[argc - 2]);

	if (0 == strcmp(argv[argc - 2], argv[argc - 1])) {

		debug_printf(DP_WARN, "slice should not be called with identical input and output!\n");

		na in2 = in;
		in = na_clone(in2);

		na_copy(in, in2);

		na_free(in2);
		io_unregister(argv[argc - 2]);
	}

	long pos2[na_rank(in)];

	for (unsigned int i = 0; i < na_rank(in); i++)
		pos2[i] = 0;

	unsigned long flags = 0L;

	for (int i = 0; i < count; i += 2) {

		int dim = atoi(argv[i + 1]);
		int pos = atoi(argv[i + 2]);

		assert(dim >= 0);
		assert(pos >= 0);
		assert(dim < (int)na_rank(in));
		assert(pos < (*NA_DIMS(in))[dim]);

		flags = MD_SET(flags, dim);
		pos2[dim] = pos;
	}

	na sl = na_slice(in, ~flags, na_rank(in), &pos2);
	na out = na_create(argv[argc - 1], na_type(sl));
	na_copy(out, sl);

	na_free(sl);
	na_free(out);
	na_free(in);
	return 0;
}


