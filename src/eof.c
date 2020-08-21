/* Copyright 2017-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 *
 */

#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/init.h"
#include "num/multind.h"
#include "num/fft.h"

#include "calib/ssa.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif





static const char usage_str[] = "<eof>";
static const char help_str[] = "EOF frequency analysis.";


int main_eof(int argc, char* argv[])
{


    float dt = -1.;
    float f = -1;
    float f_interval = -1;
    long max = 30;

	const struct opt_s opts[] = {

		OPT_FLOAT('f', &f, "f", "Frequency [Hz]"),
        OPT_FLOAT('i', &f_interval, "delta", "Frequency interval [Hz]"),
		OPT_FLOAT('t', &dt, "dt", "Temporal resolution [s]"),
		OPT_LONG('m', &max, "max", "Maximum EOFs to select"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

    long in_dims[DIMS];
	const complex float* EOF = load_cfl(argv[1], DIMS, in_dims);

    assert(!md_check_dimensions(DIMS, in_dims, (MD_BIT(0)|MD_BIT(1))));
    assert(dt > 0);
    assert(f > 0);
    
    
    long EOF_dims[2] = { in_dims[0], in_dims[1] };

    // Get spectrum
    complex float* EOF_fft = md_alloc(2, EOF_dims, CFL_SIZE);
    fftuc(2, EOF_dims, MD_BIT(0), EOF_fft, EOF);

    long flags = detect_freq_EOF(EOF_dims, EOF_fft, dt, f, f_interval, max);


    int i = 0;
	
	while (flags) {

			if (flags & 1)
				bart_printf("%d ", i);

			flags >>= 1;
			i++;
		}

		bart_printf("\n");


    md_free(EOF_fft);
    unmap_cfl(DIMS, in_dims, EOF);

	return 0;
}


