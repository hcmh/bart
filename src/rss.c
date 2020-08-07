/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu.
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>


#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/io.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Calculates root of sum of squares along selected dimensions.\n";


int main_rss(int argc, char* argv[argc])
{
	mini_cmdline(&argc, argv, 3, usage_str, help_str);

	num_init();

	long dims[DIMS];
	complex float* data = load_cfl(argv[2], DIMS, dims);

	if (0 == strcmp(argv[2], argv[3])) {

		debug_printf(DP_WARN, "rss should not be called with identical input and output!\n");

		complex float* data2 = data;
		data = anon_cfl("", DIMS, dims);

		md_copy(DIMS, dims, data, data2, CFL_SIZE);

		unmap_cfl(DIMS, dims, data2);
		io_unregister(argv[2]);
	}

	int flags = atoi(argv[1]);

	assert(0 <= flags);

	long odims[DIMS];
	md_select_dims(DIMS, ~flags, odims, dims);

	complex float* out = create_cfl(argv[3], DIMS, odims);

	md_zrss(DIMS, dims, flags, out, data);

	unmap_cfl(DIMS, dims, data);
	unmap_cfl(DIMS, odims, out);

	return 0;
}


