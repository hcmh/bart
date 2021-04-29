/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include "num/init.h"

#include "na/na.h"
#include "na/io.h"
#include "na/math.h"

#include "misc/lang.h"
#include "misc/opts.h"
#include "misc/misc.h"



static const char help_str[] = "Performs a fast Fourier transform (FFT) along selected dimensions.";




int main_fft(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool unitary = false;
	bool inv = false;
	bool center = true;

	const struct opt_s opts[] = {

		OPT_SET('u', &unitary, "unitary"),
		OPT_SET('i', &inv, "inverse"),
		OPT_CLEAR('n', &center, "un-centered"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	with (na in = na_load(in_file) ;; na_free(in))
	with (na out = na_create(out_file, na_type(in)) ;; na_free(out)) {

		na_copy(out, in);

		__typeof__(na_fft)* ffts[2][2][2] = {
			{ { na_fft, na_ifft },
			  { na_fftu, na_ifftu }, },
			{ { na_fftc, na_ifftc },
			  { na_fftuc, na_ifftuc }, },
		};

		ffts[center][unitary][inv](flags, out, out);
	}

	return 0;
}


