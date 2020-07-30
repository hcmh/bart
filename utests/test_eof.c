/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <complex.h>

#include "num/multind.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "calib/ssa.h"

#include "utest.h"


static bool test_detect_freq_EOF(void)
{

	long T = 40;
	long T2 = 40 / 2;
	long EOF_dims[2] = { T, T };
	complex float* EOF_fft = md_alloc(2, EOF_dims, sizeof(complex float));

	int a = 1;
	int b = 3;

	complex float val = 0;

	float f = 10;
	float dt = 1./T;
	float interval2 = 2;

	for (int i = 0; i < EOF_dims[1]; i++){
		for (int j = 0; j < EOF_dims[0]; j++ ){
			
			if ( (i == a) || (i == b) ){
				if ( ((T2 - f - interval2 < j) && (j < T2 - f + interval2 )) || ((T2 + f - interval2 < j) && (j < T2 + f + interval2)) )
					val = 1;
				else
					val = 0.0001;

			} else 
				val = ((double) rand() / RAND_MAX) + ((double) rand() / (RAND_MAX))  * 1i;
			
			

			EOF_fft[i * EOF_dims[0] + j] = val;
		
		}
	}	
	
    long flags = detect_freq_EOF(EOF_dims, EOF_fft, dt, f, interval2 * 2, 2);

	long flags_ref = 0;
	flags_ref = MD_SET(flags_ref, a);
	flags_ref = MD_SET(flags_ref, b);

	md_free(EOF_fft);

	return (flags == flags_ref);
}


UT_REGISTER_TEST(test_detect_freq_EOF);

