/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2016 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "num/fft.h"



#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<parameter maps> <time> <output>";
static const char help_str[] = "Calculate T1 forward model: Mss - (Mss + M0)*exp(-t*R1s).\n";


int main_T1fun(int argc, char* argv[argc])
{
	mini_cmdline(&argc, argv, 3, usage_str, help_str);

	num_init();

	long maps_dims[DIMS];
	long TI_dims[DIMS];
	
    complex float* para_maps = load_cfl(argv[1], DIMS, maps_dims);
	complex float* TI = load_cfl(argv[2], DIMS, TI_dims);

    maps_dims[MAPS_DIM] = 1;
    long skip = md_calc_size(DIMS, maps_dims);

	long maps_strs[DIMS];
    md_calc_strides(DIMS, maps_strs, maps_dims, CFL_SIZE);

    // Mss
	complex float* tmp = md_alloc(DIMS, maps_dims, CFL_SIZE); 
    md_copy(DIMS, maps_dims, tmp, para_maps, CFL_SIZE); 

    // M0
	complex float* tmp1 = md_alloc(DIMS, maps_dims, CFL_SIZE); 
    md_copy(DIMS, maps_dims, tmp1, para_maps + skip, CFL_SIZE); 
    
    // R1s
	complex float* tmp2 = md_alloc(DIMS, maps_dims, CFL_SIZE); 
    md_copy(DIMS, maps_dims, tmp2, para_maps + 2*skip, CFL_SIZE); 

	long TI_strs[DIMS];
    md_calc_strides(DIMS, TI_strs, TI_dims, CFL_SIZE);
	
    long out_dims[DIMS];
    md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG, out_dims, maps_dims);

    out_dims[TE_DIM] = TI_dims[TE_DIM];
	complex float* out_data = create_cfl(argv[3], DIMS, out_dims);

	long out_strs[DIMS];
    md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);

    // exp(-t.*R1s)
    md_zmul2(DIMS, out_dims, out_strs, out_data, maps_strs, tmp2, TI_strs, TI);
    md_zsmul2(DIMS, out_dims, out_strs, out_data, out_strs, out_data, -1.);
    md_zexp(DIMS, out_dims, out_data, out_data);

    // Mss + M0
    md_zadd(DIMS, maps_dims, tmp2, tmp1, tmp); 

    // -(Mss + M0).*exp(-t.*R1s)
	complex float* tmp3 = md_alloc(DIMS, out_dims, CFL_SIZE); 
    md_zmul2(DIMS, out_dims, out_strs, tmp3, maps_strs, tmp2, out_strs, out_data);
    md_zsmul2(DIMS, out_dims, out_strs, tmp3, out_strs, tmp3, -1.);
    
    // Mss - (Mss + M0).*exp(-t.*R1s)
    md_zadd2(DIMS, out_dims, out_strs, out_data, maps_strs, tmp, out_strs, tmp3); 
    
    md_free(tmp);
    md_free(tmp1);
    md_free(tmp2);
    md_free(tmp3);

	unmap_cfl(DIMS, maps_dims, para_maps);
	unmap_cfl(DIMS, TI_dims, TI);
	unmap_cfl(DIMS, out_dims, out_data);
	exit(0);
}


