/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <sys/types.h>
#include <stdint.h>
#include <complex.h>
#include <fcntl.h>
#include <unistd.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static void xread(int fd, void* buf, size_t size)
{
	if (size != (size_t)read(fd, buf, size))
		error("reading file");
}

static void xseek(int fd, off_t pos)
{
        if (-1 == lseek(fd, pos, SEEK_SET))
		error("seeking");
}

static void meas_setup(int fd)
{
	uint32_t start = 0;

	xseek(fd, 0);
	xread(fd, &start, sizeof(uint32_t));
	xseek(fd, start);
}


static int adc_read(int fd, const long dims[DIMS], long pos[DIMS], complex float* buf)
{
	uint32_t lc[8];
	xread(fd, lc, sizeof(lc));

	pos[PHS1_DIM]	= lc[0];
	pos[TE_DIM]	= lc[1];
	pos[SLICE_DIM]	= lc[2];
	pos[TIME_DIM]	= lc[3];
	pos[COEFF_DIM]	= lc[4];
	pos[TIME2_DIM]	= lc[5];

	pos[COIL_DIM] = 0;

	debug_print_dims(DP_DEBUG4, DIMS, pos);

	xread(fd, buf, dims[COIL_DIM] * dims[READ_DIM] * CFL_SIZE);

	return 0;
}




static const char usage_str[] = "<dat file> <output>";
//	fprintf(fd, "Usage: %s [...] [-a A] <dat file> <output>\n", name);

static const char help_str[] = "Read data from Siemens twix (.dat) files.";


int main_umgread(int argc, char* argv[argc])
{
	long adcs = 0;

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);

	struct opt_s opts[] = {

		OPT_LONG('x', &(dims[READ_DIM]), "X", "number of samples (read-out)"),
		OPT_LONG('y', &(dims[PHS1_DIM]), "Y", "phase encoding steps"),
		OPT_LONG('z', &(dims[PHS2_DIM]), "Z", "partition encoding steps"),
		OPT_LONG('s', &(dims[SLICE_DIM]), "S", "number of slices"),
		OPT_LONG('v', &(dims[AVG_DIM]), "V", "number of averages"),
		OPT_LONG('c', &(dims[COIL_DIM]), "C", "number of channels"),
		OPT_LONG('n', &(dims[TIME_DIM]), "N", "number of repetitions"),
		OPT_LONG('p', &(dims[COEFF_DIM]), "P", "number of cardicac phases"),
		OPT_LONG('f', &(dims[TIME2_DIM]), "F", "number of flow encodings"),
		OPT_LONG('a', &adcs, "A", "total number of ADCs"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);


	if (0 == adcs)
		adcs = dims[PHS1_DIM] * dims[PHS2_DIM] * dims[SLICE_DIM] * dims[TIME_DIM];

	debug_print_dims(DP_DEBUG1, DIMS, dims);

        int ifd;
        if (-1 == (ifd = open(argv[1], O_RDONLY)))
                error("error opening file.");

	meas_setup(ifd);


	complex float* out = create_cfl(argv[2], DIMS, dims);
	md_clear(DIMS, dims, out, CFL_SIZE);


	long adc_dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|COIL_FLAG, adc_dims, dims);

	void* buf = md_alloc(DIMS, adc_dims, CFL_SIZE);


	while (adcs--) {

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		if (-1 == adc_read(ifd, dims, pos, buf)) {

			debug_printf(DP_WARN, "Stopping.\n");
			break;
		}

		debug_print_dims(DP_DEBUG1, DIMS, pos);

		if (!md_is_index(DIMS, pos, dims)) {

			debug_printf(DP_WARN, "Index out of bounds.\n");
			debug_print_dims(DP_WARN, DIMS, dims);
			debug_print_dims(DP_WARN, DIMS, pos);
			continue;
		}

		md_copy_block(DIMS, pos, dims, out, adc_dims, buf, CFL_SIZE); 
	}

	md_free(buf);
	unmap_cfl(DIMS, dims, out);
	exit(0);
}

