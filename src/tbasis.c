/* All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "moba/meco.h"

#include "num/init.h"
#include "num/multind.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static const char usage_str[] = "<output_signal>"; // <output_parameter_dictionary>
static const char help_str[] = 
		"Calculate the temporal-basis matrix based on MR physical models.";




int main_tbasis(int argc, char* argv[])
{
	enum model_type { T1, WFR2S, R2S } model_type = WFR2S;

	const struct opt_s opts[] = {
		OPT_SELECT('O', enum model_type, &model_type, T1   , "T1   "),
		OPT_SELECT('F', enum model_type, &model_type, WFR2S, "WFR2S"),
		OPT_SELECT('R', enum model_type, &model_type, R2S  , "R2S  "),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();

	assert( WFR2S == model_type ); // TODO: allows other models

	long N_TIMES = 300; // number of temporal points

	enum { N_PAR = 4 }; // number of physical parameters

	//                              min , max , points
	float par_range[N_PAR][3] = { { 1.00, 0.50,  2}, // W
	                              { 0.00, 0.50,  2}, // F
	                              { 0.01, 1.00, 64}, // R2S [1/ms]
	                              {-0.45, 0.45, 64}  // fB0 [kHz] 
	                            };

	float TE_range[3] = { 1., 50., N_TIMES };


	enum { N = 6 };

	long dims_par[N] = { [0 ... N-1] = 1 };
	long dims_sig[N] = { [0 ... N-1] = 1 };

	long dims_TE[N]  = { [0 ... N-1] = 1 };

	dims_sig[0]  = N_TIMES;

	dims_TE[0] = N_TIMES;

	for (int n = 1; n < N_PAR + 1; n++ ) {
		dims_par[n-1] = par_range[n-1][2];
		dims_sig[n]   = par_range[n-1][2];
	}

	// complex float* out_par = create_cfl(argv[1], N, dims_par);
	complex float* out_sig = create_cfl(argv[1], N, dims_sig);




	complex float* TE = md_alloc(N, dims_TE, CFL_SIZE);

	for ( int ei = 0; ei < N_TIMES; ei++ ) {
		TE[ei] = (TE_range[0] + ei*(TE_range[1] - TE_range[0])/(TE_range[2]-1)) + 0.*I;
	}

	complex float* cshift = md_alloc(N, dims_TE, CFL_SIZE);
	meco_calc_fat_modu(N, dims_TE, TE, cshift);


	long p = 0;
	long pos[N] = { 0 };

	long  ind[N_PAR] = { 0 };
	float val[N_PAR] = { 0 };

	do {

		for ( int ip = 0; ip < N_PAR; ip++ ) {

			float pmin = par_range[ip][0];
			float pmax = par_range[ip][1];
			float pnum = par_range[ip][2];

			ind[ip] = pos[ip];
			val[ip] = pmin + ind[ip] * (pmax - pmin) / (pnum - 1);
		}

		// debug_printf(DP_INFO, "     %3ld \t %3ld \t %3ld \t %3ld \n", pos[0], pos[1], pos[2], pos[3]);

		for ( long eind = 0; eind < N_TIMES; eind++ ) {

			complex float W = val[0] + I * 0.;
			complex float F = val[1] + I * 0.;
			complex float z = -val[2] + I * 2.*M_PI * val[3];

			out_sig[p*N_TIMES + eind] = (W + F * cshift[eind]) * cexpf(z * TE[eind]);
		}

		p++;

	} while (md_next(N, dims_par, ~0L, pos));

	debug_printf(DP_INFO, "count: %d\n",p);

	xfree(TE);
	xfree(cshift);

	// unmap_cfl(N, dims_par, out_par);
	unmap_cfl(N, dims_sig, out_sig);

	return 0;
}
