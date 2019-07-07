/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#define DIMS 16

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "flags dim1 ... dimN <input> <output>";
static const char help_str[] = "Reshape selected dimensions.\n";


int main_reshape(int argc, char* argv[])
{
	cmdline(&argc, argv, 3, 100, usage_str, help_str, 0, NULL);

	num_init();

	unsigned int flags = atoi(argv[1]);
	unsigned int n = bitcount(flags);

	assert((int)n + 3 == argc - 1);

	long in_dims[DIMS];
	long in_strs[DIMS];

	long out_dims[DIMS];
	long out_strs[DIMS];

	complex float* in_data = load_cfl(argv[n + 2], DIMS, in_dims);

	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

	md_copy_dims(DIMS, out_dims, in_dims);
	
	unsigned int j = 0;

	for (unsigned int i = 0; i < DIMS; i++)
		if (MD_IS_SET(flags, i))
			out_dims[i] = atoi(argv[j++ + 2]);

	// Check if reshape dimensions are contiguous
	bool contiguous = true;
	int reshapedim[n]; // Dimensions to be reshaped, specified by flags
	int count = 0;

	for (int i = 0; i < DIMS; i++)
		if (MD_IS_SET(flags, i)) {
			reshapedim[count] = i;
			count++;
		}

	for (int i = reshapedim[0]; i < reshapedim[n-1]; i++)
		if (!MD_IS_SET(flags,i) && in_dims[i] > 1) {
			contiguous = false;
			break;
		}

	assert(j == n);

	complex float* out_data = create_cfl(argv[n + 3], DIMS, out_dims);

	if (contiguous) {

		assert(md_calc_size(DIMS, in_dims) == md_calc_size(DIMS, out_dims));

		md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);

		for (unsigned int i = 0; i < DIMS; i++)
			if (!(MD_IS_SET(flags, i) || (in_strs[i] == out_strs[i])))
				error("Dimensions are not consistent at index %d.\n", i);

		md_copy(DIMS, in_dims, out_data, in_data, CFL_SIZE);
	} else {

		// Check if n adjacent singleton dimensions are available
		unsigned int count = 0;
		int bufferdims = -1;
		for (unsigned int i = 0; i < DIMS; i++) {
			if (in_dims[i] == 1) {
				count++;
				if (count == n) {
					bufferdims = i + 1 - n;
					break;
				}
			} else
				count = 0;
		}

		if (bufferdims == -1)
			error("Cannot reshape data. Reshape dimensions not contiguous!\n");

		// Transpose data to adjacent dimensions
		long src_dims[DIMS];
		long dst_dims[DIMS];
		md_copy_dims(DIMS, src_dims, in_dims);

		complex float* src = md_alloc(DIMS, in_dims, CFL_SIZE);
		complex float* dst = md_alloc(DIMS, in_dims, CFL_SIZE);
		md_copy(DIMS, in_dims, src, in_data, CFL_SIZE);

		count = 0;
		for (unsigned int i = bufferdims; i < bufferdims + n; i++) {
			assert(src_dims[i] == 1); // Dimension must be free to copy into
			md_transpose_dims(DIMS, i, reshapedim[count], dst_dims, src_dims);
			md_transpose(DIMS, i, reshapedim[count], dst_dims, dst, src_dims, src, CFL_SIZE);

			// Make 'dst' the new 'src'
			md_copy_dims(DIMS, src_dims, dst_dims);
			md_copy(DIMS, dst_dims, src, dst, CFL_SIZE);
			count++;
		}

			// Assign new dimensions (still at buffer position)
		long mod_dims[DIMS];
		md_copy_dims(DIMS, mod_dims, dst_dims);

		j = 0;
		for (unsigned int i = bufferdims; i < bufferdims + n; i++)
			mod_dims[i] = atoi(argv[j++ + 2]);


			// Check consistency
		assert(md_calc_size(DIMS, dst_dims) == md_calc_size(DIMS, mod_dims));

			// Transpose to actual dimensiosn
		md_copy_dims(DIMS, src_dims, mod_dims); // Data already copied to src earlier
		count = 0;
		for (unsigned int i = bufferdims; i < bufferdims + n; i++) {
			// assert(src_dims[reshapedim[count]] == 1); // Dimension must be free to copy into

			md_transpose_dims(DIMS, reshapedim[count], i, dst_dims, src_dims);
			md_transpose(DIMS, reshapedim[count], i, dst_dims, dst, src_dims, src, CFL_SIZE);

			// Make 'dst' the new 'src'
			md_copy_dims(DIMS, src_dims, dst_dims);
			md_copy(DIMS, dst_dims, src, dst, CFL_SIZE);
			count++;
		}

		md_copy(DIMS, dst_dims, out_data, dst, CFL_SIZE);


		md_free(src);
		md_free(dst);


	}

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	return 0;
}


