/* Copyright 2020. Uecker Lab. University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
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


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Generate synthesized images from quantitative parameter maps.\n";


int main_synthesize(int argc, char* argv[argc])
{
	float TR = -1.;
	float TE = -1.;
	int nspoke = 10;
	int nframe = 50;
	float scaling_M0 = 2.0;
	float scaling_R2 = 10.0;

	enum seq_type { IR_FLASH, TSE};
	enum seq_type seq = IR_FLASH;

	const struct opt_s opts[] = {

		OPT_SELECT('F', enum seq_type, &seq, IR_FLASH, "IR_FLASH"),
		OPT_SELECT('T', enum seq_type, &seq, TSE, "TSE"),
		OPT_FLOAT('R', &TR, "TR", "Repetition time (second)"),
		OPT_FLOAT('E', &TE, "TE", "Echo time (second)"),
		OPT_INT('s', &nspoke, "nspoke", "radial spokes per frame"),
		OPT_INT('n', &nframe, "nframe", "number of contrast images"),
		OPT_FLOAT('M', &scaling_M0, "scaling_M0", "scaling of M0 for IR FLASH sequence"),
		OPT_FLOAT('S', &scaling_R2, "scaling_R2", "scaling of R2 for TSE sequence"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long idims[DIMS];
	complex float* in_data = load_cfl(argv[1], DIMS, idims);

	long odims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, odims, idims);
	odims[TE_DIM] = nframe;

	complex float* out_data = create_cfl(argv[2], DIMS, odims);

	long dims1[DIMS];
	md_select_dims(DIMS, TE_FLAG, dims1, odims);

	long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

	long ostrs[DIMS];
	md_calc_strides(DIMS, ostrs, odims, CFL_SIZE);

	long strs1[DIMS];
	md_calc_strides(DIMS, strs1, dims1, CFL_SIZE);

	long pos[DIMS] = { 0 };

        do {    
		complex float out[nframe];
		complex float M0;

		switch (seq) {

		case IR_FLASH:

			assert(3 == idims[COEFF_DIM]);

			complex float Ms = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 0, pos), in_data);
			M0 = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 1, pos), in_data);
			complex float R1s = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 2, pos), in_data); 

			for (int ind = 0; ind < nframe; ind++)
				out[ind] = Ms - (Ms + scaling_M0 * M0) * exp(-TR * ind * nspoke * cabs(R1s));

                        break;

		case TSE:

			assert(2 == idims[COEFF_DIM]);

			M0 = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 0, pos), in_data);
			complex float R2 = MD_ACCESS(DIMS, istrs, (pos[COEFF_DIM] = 1, pos), in_data);

			for (int ind = 0; ind < nframe; ind++)
				out[ind] = M0 * exp(-TE * ind * scaling_R2 * cabs(R2));

			break;

		default:
			error("sequence type not supported");
                }

		pos[COEFF_DIM] = 0;
		md_copy_block2(DIMS, pos, odims, ostrs, out_data, dims1, strs1, out, CFL_SIZE);

	} while(md_next(DIMS, odims, ~TE_FLAG, pos));

        unmap_cfl(DIMS, idims, in_data);
        unmap_cfl(DIMS, odims, out_data);

        return 0;
}

