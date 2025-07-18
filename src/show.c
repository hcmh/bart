/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2015-2024. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2024 Martin Uecker
 * 2015-2016 Jon Tamir
 */

#define _GNU_SOURCE

#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Outputs values or meta data.";


static void print_cfl(int N, const long dims[N], const complex float* data, const char* fmt, const char* sep,
		bool real, bool index)
{
	// find first non-trivial dimension
	int l = 0;

	while ((l < N - 1) && (1 == dims[l]))
		l++;

	long T = md_calc_size(N, dims);

	const char* allowed_fmts[2][4] = {
	      { "%%+%*[0-9.]f%%+%*[0-9.]fi%n",
		"%%+%*[0-9.]e%%+%*[0-9.]ei%n",
		"%%+%*[0-9.]f,%%+%*[0-9.]f%n",
		"%%+%*[0-9.]e,%%+%*[0-9.]e%n", },
	      { "%%+%*[0-9.]f%n",
		"%%+%*[0-9.]e%n",
		"%%+%*[0-9.]f%n",
		"%%+%*[0-9.]e%n", },
	};

	// ensure that the input format string matches one of the valid format templates
	for (int i = 0; i < (int)ARRAY_SIZE(allowed_fmts[0]); i++) {

		size_t rd = 0;

		if (0 == sscanf(fmt, allowed_fmts[real][i], &rd))
			if (strlen(fmt) == rd)
				goto ok;
	}

	debug_printf(DP_ERROR, "Invalid format string.\n");

	return;

ok:
	for (long i = 0; i < T; i++) {

		if (index)
			printf("%04ld%s", i, sep);

		if (real)
			printf(fmt, crealf(data[i]));
		else
			printf(fmt, crealf(data[i]), cimagf(data[i]));

		printf("%s", (0 == (i + 1) % dims[l]) ? "\n" : sep);

		if (index)
			printf("\n");
	}
}



int main_show(int argc, char* argv[argc])
{
	const char* in_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
	};

	bool meta = false;
	int showdim = -1;
	bool real = false;
	bool index = false;
	const char* sep = "\t";
	const char* fmt = NULL;

	const struct opt_s opts[] = {

		OPT_SET('m', &meta, "show meta data"),
		OPT_PINT('d', &showdim, "dim", "show size of dimension"),
		OPT_SET('R', &real, "print only real part"),
		OPT_SET('N', &index, "print index"),
		OPT_STRING('s', &sep, "sep", "use <sep> as the separator"),
		OPT_STRING('f', &fmt, "format", "use <format> as the format. Default: \"%+.6e%+.6ei\""),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (NULL == fmt)
		fmt = real ? "%+.6e" : "%+.6e%+.6ei";

	int N = DIMS;

	long dims[N];
	complex float* data = load_cfl(in_file, N, dims);

	if (-1 != showdim) {

		assert((showdim >= 0) && (showdim < N));
		bart_printf("%ld\n", dims[showdim]);

		goto out;
	}

	if (meta) {

		bart_printf("Type: complex float\n");
		bart_printf("Dimensions: %d\n", N);	// FIXME always DIMS
		bart_printf("AoD:");

		for (int i = 0; i < N; i++)
			bart_printf("\t%ld", dims[i]);

		bart_printf("\n");

		goto out;
	}

	print_cfl(N, dims, data, fmt, sep, real, index);

out:
	unmap_cfl(N, dims, data);
	return 0;
}

