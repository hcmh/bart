
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

static bool test_bloch_irflash_frw_der(void)
{
	enum { N = 16 };
	enum { rep = 300 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, rep, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long all_dims[N] = { 16, 16, 1, 1, 1, rep, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, rep, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	bool gpu_use = false;

	// Init and apply Bloch model operator

	complex float* src = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src, 1.0);

	complex float* dst_frw_bloch = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst_der_bloch = md_alloc(N, out_dims, CFL_SIZE);

	struct modBlochFit fit_para = modBlochFit_defaults;

	// IR FLASH characteristics
	fit_para.sequence = 5;
	fit_para.tr = 0.003;
	fit_para.te = 0.001;
	fit_para.fa = 8.;
	fit_para.rfduration = 0.00001;
	fit_para.inversion_pulse_length = 0.;
	fit_para.prep_pulse_length = 0.;

	// Correct M0 to ensure same scaling between Bloch simulation and IR FLASH model
	fit_para.scale[3] = 1./sinf(fit_para.fa * M_PI/180.);

	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst_frw_bloch, N, in_dims, src);
	nlop_derivative(Bloch, N, out_dims, dst_der_bloch, N, in_dims, src);

	// dump_cfl("_dst_frw_bloch", N, out_dims, dst_frw_bloch);
	// dump_cfl("_dst_der_bloch", N, out_dims, dst_der_bloch);

	nlop_free(Bloch);
	md_free(src);


	// Init and apply IR FLASH model operator

	in_dims[COEFF_DIM] = 2;

	complex float* src2 = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src2, 1.0);

	complex float* dst_frw_irflash = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst_der_irflash = md_alloc(N, out_dims, CFL_SIZE);

	// Inversion times
	complex float* TI = md_alloc(N, TI_dims, CFL_SIZE);

	// ! Analytical Assumption: Mxy(t=TE) == Mz(t=0)
	for (int i = 0; i < rep; i++)
		TI[i] =  i * fit_para.tr;

	// alpha map
	complex float* fa = md_alloc(N, map_dims, CFL_SIZE);
	md_zfill(N, map_dims, fa, fit_para.fa);

	complex float* alpha = md_alloc(N, map_dims, CFL_SIZE);

	fa_to_alpha(DIMS, map_dims, alpha, fa, fit_para.tr);

	md_free(fa);

	struct nlop_s* T1 = nlop_T1_alpha_in_create(N, map_dims, out_dims, in_dims, TI_dims, TI, alpha, gpu_use);

	nlop_apply(T1, N, out_dims, dst_frw_irflash, N, in_dims, src2);
	nlop_derivative(T1, N, out_dims, dst_der_irflash, N, in_dims, src2);

	// dump_cfl("_dst_frw_irflash", N, out_dims, dst_frw_irflash);
	// dump_cfl("_dst_der_irflash", N, out_dims, dst_der_irflash);

	nlop_free(T1);
	md_free(src2);
	md_free(TI);
	md_free(alpha);

	// Compare operator outputs

	float err_frw = md_znrmse(N, out_dims, dst_frw_irflash, dst_frw_bloch);
	// debug_printf(DP_INFO, "Error Forward: %f\n", err_frw);

	float err_der = md_znrmse(N, out_dims, dst_der_irflash, dst_der_bloch);
	// debug_printf(DP_INFO, "Error Derivative: %f\n", err_der);


	md_free(dst_frw_irflash);
	md_free(dst_frw_bloch);

	if (err_frw > 3.E-3)
		return 0;

	if (err_der > 5.E-3)
		return 0;

	return 1;
}
UT_REGISTER_TEST(test_bloch_irflash_frw_der);


static bool test_bloch_ode_obs_irflash(void)
{
	enum { N = 16 };
	enum { rep = 300 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, rep, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long all_dims[N] = { 16, 16, 1, 1, 1, rep, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

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
	fit_para.inversion_pulse_length = 0.;
	fit_para.prep_pulse_length = 0.;

	// Turn off T2 relaxation (IR FLASH insensitive to it)
	fit_para.scale[2] = 0.0001;

	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst1, N, in_dims, src);

	// dump_cfl("_dst1", N, out_dims, dst1);

	fit_para.full_ode_sim = true;

	struct nlop_s* Bloch2 = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

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


static bool test_bloch_ode_obs_irbssfp(void)
{
	enum { N = 16 };
	enum { rep = 300 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, rep, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long all_dims[N] = { 16, 16, 1, 1, 1, rep, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	bool gpu_use = false;

	// Init and apply Bloch model operator

	complex float* src = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src, 1.0);

	complex float* dst1 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, out_dims, CFL_SIZE);

	struct modBlochFit fit_para = modBlochFit_defaults;

	// IR FLASH characteristics
	fit_para.sequence = 1;
	fit_para.rfduration = 0.001;
	fit_para.tr = 0.0045;
	fit_para.te = 0.00225;
	fit_para.fa = 45.;
	fit_para.inversion_pulse_length = 0.001;
	fit_para.prep_pulse_length = fit_para.te;

	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst1, N, in_dims, src);

	// dump_cfl("_dst1", N, out_dims, dst1);

	fit_para.full_ode_sim = true;

	struct nlop_s* Bloch2 = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

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

	UT_ASSERT(err < 3.E-3);
}
UT_REGISTER_TEST(test_bloch_ode_obs_irbssfp);