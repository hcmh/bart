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



static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Performs a fast Fourier transform (FFT) along selected dimensions.";




int main_fft(int argc, char* argv[argc])
{
	bool unitary = false;
	bool inv = false;
	bool center = true;

	const struct opt_s opts[] = {

		OPT_SET('u', &unitary, "unitary"),
		OPT_SET('i', &inv, "inverse"),
		OPT_CLEAR('n', &center, "un-centered"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	with (na in = na_load(argv[2]) ;; na_free(in))
	with (na out = na_create(argv[3], na_type(in)) ;; na_free(out)) {

		unsigned long flags = labs(atol(argv[1]));

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


