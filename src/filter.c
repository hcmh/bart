/* Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/casorati.h"
#include "num/filter.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"
#include "misc/io.h"



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Apply filter.\n";


int main_filter(int argc, char* argv[])
{
	int len = -1;
	int dim = -1;
	int med = -1;
	int mavg = -1;
	bool geom = false;

	const struct opt_s opts[] = {

		OPT_INT('m', &med, "dim", "median filter along dimension dim"),
		OPT_INT('l', &len, "len", "length of filter"),
		OPT_SET('G', &geom, "geometric median"),
		OPT_INT('a', &mavg, "dim", "Moving average filter along dimension dim"),
		
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	char filter_type;

	assert (med == -1 || mavg == -1);

	if (med >= 0) {

		dim = med;
		filter_type = 'm';

	} else if (mavg >= 0) {

		dim = mavg;
		filter_type = 'a';

	}

	long in_dims[DIMS];
	
	complex float* in_data = load_cfl(argv[1], DIMS, in_dims);

	if (0 == strcmp(argv[1], argv[2])) {

		debug_printf(DP_WARN, "filter should not be called with identical input and output!\n");

		complex float* in_data2 = in_data;
		in_data = anon_cfl("", DIMS, in_dims);

		md_copy(DIMS, in_dims, in_data, in_data2, CFL_SIZE);

		unmap_cfl(DIMS, in_dims, in_data2);
		io_unregister(argv[1]);
	}

	assert(dim >= 0);
	assert(dim < DIMS);
	assert(len > 0);
	assert(len <= in_dims[dim]);

	long tmp_dims[DIMS + 1];
	md_copy_dims(DIMS, tmp_dims, in_dims);
	tmp_dims[DIMS] = 1;

	long tmp_strs[DIMS + 1];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);

	tmp_dims[dim] = in_dims[dim] - len + 1;

	long tmp2_strs[DIMS + 1];
	md_calc_strides(DIMS + 1, tmp2_strs, tmp_dims, CFL_SIZE);

	tmp_dims[DIMS] = len;
	tmp_strs[DIMS] = tmp_strs[dim];

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, tmp_dims);

	complex float* out_data = create_cfl(argv[2], DIMS, out_dims);

	switch (filter_type) {

		case 'm': {
	
			(geom ? md_geometric_medianz2 : md_medianz2)(DIMS + 1, DIMS, tmp_dims, tmp2_strs, out_data, tmp_strs, in_data);
			break;

		}

		case 'a': {

			md_moving_avgz2(DIMS + 1, DIMS, tmp_dims, tmp2_strs, out_data, tmp_strs, in_data);
			break;
			
		}
	}

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	return 0;
}


