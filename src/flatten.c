/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
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


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Flatten array to one dimension.\n";


int main_flatten(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 2, usage_str, help_str);

	num_init();

	long idims[DIMS];

	complex float* idata = load_cfl(argv[1], DIMS, idims);

	if (0 == strcmp(argv[1], argv[2])) {

		debug_printf(DP_WARN, "flatten should not be called with identical input and output!\n");

		complex float* idata2 = idata;
		idata = anon_cfl("", DIMS, idims);

		md_copy(DIMS, idims, idata, idata2, CFL_SIZE);

		unmap_cfl(DIMS, idims, idata2);
		io_unregister(argv[1]);
	}

	long odims[DIMS] = MD_INIT_ARRAY(DIMS, 1);
	odims[0] = md_calc_size(DIMS, idims);

	complex float* odata = create_cfl(argv[2], DIMS, odims);

	md_copy(DIMS, idims, odata, idata, CFL_SIZE);

	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(DIMS, odims, odata);

	return 0;
}


