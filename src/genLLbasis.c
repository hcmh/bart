/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/debug.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<output>";
static const char help_str[] = "Create dictionaries for IR Look-Locker T1 mapping.\n";


int main_genLLbasis(int argc, char* argv[argc])
{
	// Sequence parameters
        float TR      = 4.1e-3;
	int nspokes   = 1;
        int nTI       = 1000;

	// Dictionary components and their ranges
	int nR1s = 1000;
        int nMss = 100;
	
	float Mss_max = 1000.0e-3;
	float Mss_min = 10.0e-3;

	float T1s_max = 5000e-3; // R1s = 1 / T1s
	float T1s_min = 5e-3;
        
        
	const struct opt_s opts[] = {

		OPT_FLOAT('R', &TR, "TR", "Repetition time."),
		OPT_INT('n', &nR1s, "nR1s", "Number of R1s to simulate."),
		OPT_INT('N', &nMss, "nMss", "Number of Mss to simulate."),
		OPT_INT('s', &nspokes, "nSpokes", "Number of spokes per frame."),
		OPT_INT('T', &nTI, "nTI", "Number of inversion times."),
	};
 
	cmdline(&argc, argv, 1, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();
              
	long dims_R1s[DIMS];
        long dims_Mss[DIMS];

	md_singleton_dims(DIMS, dims_R1s);
	md_singleton_dims(DIMS, dims_Mss);

        dims_R1s[COEFF_DIM] = nR1s;
        dims_Mss[COEFF2_DIM] = nMss;
        
        complex float* R1s = anon_cfl("", DIMS, dims_R1s);
        complex float* Mss = anon_cfl("", DIMS, dims_Mss);
        
        long odims[DIMS];
	md_copy_dims(DIMS, odims, dims_R1s);
        
        odims[COEFF2_DIM] = dims_Mss[COEFF2_DIM];
        odims[TE_DIM] = nTI;
        
        long ostrs[DIMS];
	md_calc_strides(DIMS, ostrs, odims, CFL_SIZE);
        

	complex float* out_data = create_cfl(argv[1], DIMS, odims);


	for (int i = 0; i < nR1s; i++)
		R1s[i] = 1. / (T1s_min + i * (T1s_max - T1s_min) / nR1s);
	
        
        for (int i = 0; i < nMss; i++)
		Mss[i] = Mss_min + i * (Mss_max - Mss_min) / nMss;

	float t = 0.;
        
	for (int i = 0; i < nMss; i++)
		for (int j = 0; j < nR1s; j++)
			for (int k = 0; k < nTI; k++) {

				t = TR * nspokes * k + TR * nspokes / 2 ;
				out_data[k + ostrs[COEFF_DIM] / CFL_SIZE * j + ostrs[COEFF2_DIM] / CFL_SIZE * i] = cabsf(Mss[i]) - (cabsf(Mss[i]) + 1.0) * exp(-t * cabsf(R1s[j]));
			}


	unmap_cfl(DIMS, dims_R1s, R1s);
        unmap_cfl(DIMS, dims_Mss, Mss);
	unmap_cfl(DIMS, odims, out_data);
	return 0;
}


