/* Copyright 2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/mri.h"


static const char usage_str[] = "<input R> <input C> <output>";
static const char help_str[] = "Compute phase-difference maps from R and C.\n";



int main_phasediff(int argc, char* argv[argc])
{
	enum venc_mode { ONE_SIDE, HADAMARD } venc_mode = ONE_SIDE;

	const struct opt_s opts[] = {

		OPT_SELECT('S', enum venc_mode, &venc_mode, ONE_SIDE, "One-side encoding (zero-max, min-max, minTE)"),
		OPT_SELECT('H', enum venc_mode, &venc_mode, HADAMARD, "Hadamard encoding"),
	};

	cmdline(&argc, argv, 3, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long r_dims[DIMS];
	long c_dims[DIMS];
	complex float* r_data = load_cfl(argv[1], DIMS, r_dims);
	complex float* c_data = load_cfl(argv[2], DIMS, c_dims);

	assert(r_dims[0]==c_dims[0]); 
	assert(r_dims[1]==c_dims[1]);
	assert(r_dims[2]==c_dims[2]);
	assert(r_dims[9]==c_dims[9]);
	assert(r_dims[10]==c_dims[10]);

	long r_strs[DIMS];
	long c_strs[DIMS];
	md_calc_strides(DIMS, r_strs, r_dims, CFL_SIZE);
	md_calc_strides(DIMS, c_strs, c_dims, CFL_SIZE);


	complex float* cn_data = md_alloc(DIMS, r_dims, CFL_SIZE);
	complex float* rc_data = md_alloc(DIMS, c_dims, CFL_SIZE);

	md_zrss(DIMS, c_dims, COIL_FLAG, cn_data, c_data);

	md_zmul2(DIMS, c_dims, c_strs, rc_data, r_strs, r_data, c_strs, c_data);
	md_zdiv2(DIMS, c_dims, c_strs, rc_data, c_strs, rc_data, r_strs, cn_data);

	md_free(cn_data);

	unmap_cfl(DIMS, r_dims, r_data);
	unmap_cfl(DIMS, c_dims, c_data);




	const unsigned int TRANS_FROM = CSHIFT_DIM;
	const unsigned int TRANS_TO = AVG_DIM;




	long c_trans_dims[DIMS];
	md_transpose_dims(DIMS, TRANS_FROM, TRANS_TO, c_trans_dims, c_dims);

	long c_trans_strs[DIMS];
	md_calc_strides(DIMS, c_trans_strs, c_trans_dims, CFL_SIZE);

	complex float* rc_trans_data = md_alloc(DIMS, c_trans_dims, CFL_SIZE);
	md_transpose(DIMS, TRANS_FROM, TRANS_TO, c_trans_dims, rc_trans_data, c_dims, rc_data, CFL_SIZE);

	md_free(rc_data);

	long c1_trans_dims[DIMS];
	md_select_dims(DIMS, ~AVG_FLAG, c1_trans_dims, c_trans_dims);

	long r1_trans_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, r1_trans_dims, c_trans_dims);




	const long vencodings = r_dims[CSHIFT_DIM];
	const long velocities = vencodings - 1;

	long o_dims[DIMS];
	md_select_dims(DIMS, ~CSHIFT_FLAG, o_dims, r_dims);
	o_dims[CSHIFT_DIM] = velocities;

	complex float* o_data = create_cfl(argv[3], DIMS, o_dims);


	debug_printf(DP_DEBUG1, "o_dims:\t");
	debug_print_dims(DP_DEBUG1, DIMS, o_dims);


	long o_trans_dims[DIMS];
	md_transpose_dims(DIMS, TRANS_FROM, TRANS_TO, o_trans_dims, o_dims);

	long o_trans_strs[DIMS];
	md_calc_strides(DIMS, o_trans_strs, o_trans_dims, CFL_SIZE);

	long o1_trans_dims[DIMS];
	md_copy_dims(DIMS, o1_trans_dims, o_trans_dims);
	o1_trans_dims[CSHIFT_DIM] = 1;

	complex float* o_trans_data = md_alloc(DIMS, o_trans_dims, CFL_SIZE);

	long *venc_pos = calloc(DIMS, sizeof(long));
	long *velo_pos = calloc(DIMS, sizeof(long));

	switch (venc_mode) {

		case ONE_SIDE:

			venc_pos[AVG_DIM] = 0;
			const complex float* RC0 = (void*)rc_trans_data + md_calc_offset(DIMS, c_trans_strs, venc_pos);

			for (long i = 0; i < velocities; i++) {

				venc_pos[AVG_DIM] = i+1;
				const complex float* RC1 = (void*)rc_trans_data + md_calc_offset(DIMS, c_trans_strs, venc_pos);
				
				velo_pos[AVG_DIM] = i;
				complex float* PC1 = (void*)o_trans_data + md_calc_offset(DIMS, o_trans_strs, velo_pos);

				complex float* tmp = md_alloc(DIMS, c1_trans_dims, CFL_SIZE);
				md_zmulc(DIMS, c1_trans_dims, tmp, RC0, RC1);
				md_zsum(DIMS, c1_trans_dims, COIL_FLAG, PC1, tmp);
				md_free(tmp);

				complex float* deno = md_alloc(DIMS, o1_trans_dims, CFL_SIZE);
				
				md_zabs(DIMS, o1_trans_dims, deno, PC1);
				md_zsqrt(DIMS, o1_trans_dims, deno, deno);

				md_zdiv(DIMS, o1_trans_dims, PC1, PC1, deno);

				md_free(deno);
			}

			break;

		case HADAMARD:
			break;
	}

	xfree(velo_pos);
	xfree(venc_pos);

	md_free(rc_trans_data);


	md_transpose(DIMS, TRANS_TO, TRANS_FROM, o_dims, o_data, o_trans_dims, o_trans_data, CFL_SIZE);

	md_free(o_trans_data);

	unmap_cfl(DIMS, o_dims, o_data);

	return 0;
}




