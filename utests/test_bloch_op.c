
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops_p.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/mri.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/stack.h"
#include "nlops/const.h"

#include "moba/scale.h"
#include "moba/T1_alpha_in.h"
#include "moba/model_Bloch.h"
#include "moba/blochfun.h"


#include "utest.h"

static bool test_bloch_irflash(void)
{
	enum { N = 16 };
	enum { rep = 300 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, rep, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long all_dims[N] = { 16, 16, 1, 1, 1, rep, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, rep, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long input_dims[N];

	bool gpu_use = false;

	// Init and apply Bloch model operator

	complex float* src = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src, 1.0);

	complex float* dst_bl = md_alloc(N, out_dims, CFL_SIZE);

	struct modBlochFit fit_para = modBlochFit_defaults;

	// IR FLASH characteristics
	fit_para.sequence = 5;
	fit_para.tr = 0.003;
	fit_para.te = 0.001;
	fit_para.fa = 1.;
	fit_para.rfduration = 0.00001;
	fit_para.inversion_pulse_length = 0.;
	fit_para.prep_pulse_length = 0.;

	// Correction for simulation to start with |Mxy|(t=0)=1 (as for analytical model)
	// assuming: M0 * sin(fa) = Mxy
	// only holds for small FA if t!=0
	fit_para.scale[1] = 1./sinf(fit_para.fa * M_PI/180.);

	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, input_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst_bl, N, in_dims, src);

	// dump_cfl("_dst_bl", N, out_dims, dst_bl);

	nlop_free(Bloch);
	md_free(src);


	// Init and apply IR FLASH model operator

	in_dims[COEFF_DIM] = 2;

	complex float* src2 = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src2, 1.0);

	complex float* dst_ll = md_alloc(N, out_dims, CFL_SIZE);

	// Inversion times
	complex float* TI = md_alloc(N, TI_dims, CFL_SIZE);

	for (int i = 0; i < rep; i++)
		TI[i] = fit_para.te + i * fit_para.tr;

	// alpha map
	complex float* fa = md_alloc(N, map_dims, CFL_SIZE);
	md_zfill(N, map_dims, fa, fit_para.fa);

	complex float* alpha = md_alloc(N, map_dims, CFL_SIZE);

	fa_to_alpha(DIMS, map_dims, alpha, fa, fit_para.tr);

	md_free(fa);

	struct nlop_s* T1 = nlop_T1_alpha_in_create(N, map_dims, out_dims, in_dims, TI_dims, TI, alpha, gpu_use);

	nlop_apply(T1, N, out_dims, dst_ll, N, in_dims, src2);

	// dump_cfl("_dst_ll", N, out_dims, dst_ll);

	nlop_free(T1);
	md_free(src2);
	md_free(TI);
	md_free(alpha);

	// Compare operator outputs

	float err = md_znrmse(N, out_dims, dst_ll, dst_bl);

	// debug_printf(DP_INFO, "Error: %f\n", err);

	md_free(dst_ll);
	md_free(dst_bl);

	UT_ASSERT(err < 0.003);

}
UT_REGISTER_TEST(test_bloch_irflash);


static bool test_bloch_ode_obs_irflash(void)
{
	enum { N = 16 };
	enum { rep = 300 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, rep, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long all_dims[N] = { 16, 16, 1, 1, 1, rep, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long input_dims[N];

	bool gpu_use = false;

	// Init and apply Bloch model operator

	complex float* src = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src, 1.0);

	complex float* dst1 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, out_dims, CFL_SIZE);

	struct modBlochFit fit_para = modBlochFit_defaults;

	// IR FLASH characteristics
	fit_para.sequence = 5;
	fit_para.rfduration = 0.00001;
	fit_para.tr = 0.003;
	fit_para.te = 0.001;
	fit_para.fa = 8.;
	fit_para.inversion_pulse_length = 0.00001;
	fit_para.prep_pulse_length = 0.00001;

	// Turn off T2 relaxation (IR FLASH insensitive to it)
	fit_para.scale[2] = 0.0001;

	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, input_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst1, N, in_dims, src);

	// dump_cfl("_dst1", N, out_dims, dst1);

	fit_para.full_ode_sim = true;

	struct nlop_s* Bloch2 = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, input_dims, &fit_para, gpu_use);

	nlop_apply(Bloch2, N, out_dims, dst2, N, in_dims, src);

	// dump_cfl("_dst2", N, out_dims, dst2);

	// Compare operator outputs

	float err = md_znrmse(N, out_dims, dst2, dst1);

	// debug_printf(DP_INFO, "Error: %f\n", err);

	nlop_free(Bloch);
	nlop_free(Bloch2);
	md_free(src);
	md_free(dst1);
	md_free(dst2);

	UT_ASSERT(err < 3.E-4);
}
UT_REGISTER_TEST(test_bloch_ode_obs_irflash);
