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

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "start diff num outfile";
static const char help_str[] = "Create linear rising array.\n";


int main_linspace(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 4, usage_str, help_str);

	num_init();

	float start = atof(argv[1]);
	float diff = atof(argv[2]);
	int num = atoi(argv[3]);
	
	long dims[DIMS] = { [0 ... DIMS - 1] = 1. };	
	dims[0] = num;

	complex float* out = create_cfl(argv[4], DIMS, dims);
	
	for (int i = 0; i < num; i++)
                out[i] = start + i * diff;

	unmap_cfl(DIMS, dims, out);
	return 0;
}


