/* Copyright 2020. Uecker Lab.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Pad an array along dimension by value [val] or by edge values.";


int main_pad(int argc, char* argv[])
{
	int dim = -1;
	int len = -1;
	complex float val = 0.;

	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &dim, "dim"),
		ARG_INT(true, &len, "len"),
		ARG_CFL(false, &val, "val"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool asym = false;
	bool pad_by_value = false;

	const struct opt_s opts[] = {

		OPT_SET('a', &asym, "Asymmetric padding"),

	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	assert((dim >= 0) && (dim < DIMS));
	assert(len > 0);

	if (0 != val)
		pad_by_value = true;



	long in_dims[DIMS];
	long out_dims[DIMS];

	void* in_data = load_cfl(in_file, DIMS, in_dims);
	md_copy_dims(DIMS, out_dims, in_dims);

	out_dims[dim] = in_dims[dim] + (asym ? len : (2 * len)); // padding on one end or both

	void* out_data = create_cfl(out_file, DIMS, out_dims);

	// Assign position of in_data and pad by value
	if (asym)
		md_pad(DIMS, &val, out_dims, out_data, in_dims, in_data, CFL_SIZE);
	else
		md_pad_center(DIMS, &val, out_dims, out_data, in_dims, in_data, CFL_SIZE);

	// Overwrite padding by edge value
	if (!pad_by_value) {

		long in_strs[DIMS];
		md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

		long in_mod_strs[DIMS];
		md_calc_strides(DIMS, in_mod_strs, in_dims, CFL_SIZE);
		in_mod_strs[dim] = 0;

		long pad_dims[DIMS];
		md_copy_dims(DIMS, pad_dims, in_dims);
		pad_dims[dim] = len;

		long out_strs[DIMS];
		md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);

		long pos[DIMS] = { 0 };
		long pos1[DIMS] = { 0 };
		pos1[dim] = in_dims[dim] - 1;
		long offset_in = md_calc_offset(DIMS, in_strs, pos1);

		pos1[dim] = in_dims[dim] + (asym ? 0 : len);
		long offset_out = md_calc_offset(DIMS, out_strs, pos1);
		
		md_copy_block2(DIMS, pos, pad_dims, out_strs, out_data + offset_out, pad_dims, in_mod_strs, in_data + offset_in, CFL_SIZE); 
	
		if (!asym)
			md_copy_block2(DIMS, pos, pad_dims, out_strs, out_data, pad_dims, in_mod_strs, in_data, CFL_SIZE);
	}

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	return 0;
}


