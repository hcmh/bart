/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/io.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Flip (reverse) dimensions specified by the {bitmask}.\n";



int main_flip(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 3, usage_str, help_str);

	num_init();

	int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(argv[2], N, dims);

	if (0 == strcmp(argv[2], argv[3])) {

		debug_printf(DP_WARN, "flip should not be called with identical input and output!\n");

		complex float* idata2 = idata;
		idata = anon_cfl("", N, dims);

		md_copy(N, dims, idata, idata2, sizeof(complex float));

		unmap_cfl(N, dims, idata2);
		io_unregister(argv[2]);
	}


	complex float* odata = create_cfl(argv[3], N, dims);

	unsigned long flags = atoi(argv[1]);

	md_flip(N, dims, flags, odata, idata, sizeof(complex float));

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);

	return 0;
}


