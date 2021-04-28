
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


static bool test_bloch_irflash_frw_der_xy(void)
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
	fit_para.look_locker_assumptions = false;

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


	md_free(dst_frw_bloch);
	md_free(dst_der_bloch);
	md_free(dst_frw_irflash);
	md_free(dst_der_irflash);

	if (err_frw > 3.E-3)
		return 0;

	if (err_der > 5.E-3)
		return 0;

	return 1;
}
UT_REGISTER_TEST(test_bloch_irflash_frw_der_xy);

static bool test_bloch_irflash_frw_der_z(void)
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
	fit_para.look_locker_assumptions = true; // ! Analytical Assumption: Mxy(t=TE) == Mz(t=0)


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


	md_free(dst_frw_bloch);
	md_free(dst_der_bloch);
	md_free(dst_frw_irflash);
	md_free(dst_der_irflash);

	if (err_frw > 3.E-3)
		return 0;

	if (err_der > 5.E-3)
		return 0;

	return 1;
}
UT_REGISTER_TEST(test_bloch_irflash_frw_der_z);

static bool test_bloch_irflash_adj_z(void)
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

	complex float* src1 = md_alloc(N, out_dims, CFL_SIZE);
	md_zfill(N, out_dims, src1, 1.0);
	complex float* dst_adj_bloch = md_alloc(N, in_dims, CFL_SIZE);

	struct modBlochFit fit_para = modBlochFit_defaults;

	// IR FLASH characteristics
	fit_para.sequence = 5;
	fit_para.tr = 0.003;
	fit_para.te = 0.001;
	fit_para.fa = 8.;
	fit_para.rfduration = 0.00001;
	fit_para.inversion_pulse_length = 0.;
	fit_para.prep_pulse_length = 0.;
	fit_para.look_locker_assumptions = true; // ! Analytical Assumption: Mxy(t=TE) == Mz(t=0)

	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst_frw_bloch, N, in_dims, src);
	nlop_adjoint(Bloch, N, in_dims, dst_adj_bloch, N, out_dims, src1);

	nlop_free(Bloch);
	md_free(src);
	md_free(src1);


	// Init and apply IR FLASH model operator

	in_dims[COEFF_DIM] = 2;

	complex float* src2 = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src2, 1.0);
	complex float* dst_frw_irflash = md_alloc(N, out_dims, CFL_SIZE);

	complex float* src3 = md_alloc(N, out_dims, CFL_SIZE);
	md_zfill(N, out_dims, src3, 1.0);
	complex float* dst_adj_irflash = md_alloc(N, in_dims, CFL_SIZE);

	// Inversion times
	complex float* TI = md_alloc(N, TI_dims, CFL_SIZE);

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
	nlop_adjoint(T1, N, in_dims, dst_adj_irflash, N, out_dims, src3);

	nlop_free(T1);
	md_free(src2);
	md_free(src3);
	md_free(TI);
	md_free(alpha);

	// Compare operator outputs

	long pos[N] = { [0 ... N - 1] = 0 };

	complex float* map_update_bloch = md_alloc(N, map_dims, CFL_SIZE);
	md_copy_block(N, pos, map_dims, map_update_bloch, in_dims, dst_adj_bloch, CFL_SIZE);

	pos[COEFF_DIM] = 1;
	complex float* map_update_irflash = md_alloc(N, map_dims, CFL_SIZE);
	md_copy_block(N, pos, map_dims, map_update_irflash, in_dims, dst_adj_irflash, CFL_SIZE);

	// dump_cfl("_dst_update_irflash", N, map_dims, map_update_irflash);
	// dump_cfl("_dst_update_bloch", N, map_dims, map_update_bloch);

	float err_adj = md_znrmse(N, map_dims, map_update_bloch, map_update_irflash);
	// debug_printf(DP_INFO, "Error Adjoint (R1 map update): %f\n", err_adj);

	md_free(dst_frw_bloch);
	md_free(dst_adj_bloch);
	md_free(map_update_bloch);

	md_free(dst_frw_irflash);
	md_free(dst_adj_irflash);
	md_free(map_update_irflash);

	if (err_adj > 4.E-3)
		return 0;

	return 1;
}
UT_REGISTER_TEST(test_bloch_irflash_adj_z);


static bool test_bloch_irflash_frw_der_spoke_av_z(void)
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
	fit_para.look_locker_assumptions = true; // ! Analytical Assumption: Mxy(t=TE) == Mz(t=0)

	fit_para.averaged_spokes = 5;

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

	for (int i = 0; i < rep; i++)
		TI[i] =  5 * i * fit_para.tr + 2.5 * fit_para.tr;

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
	// debug_printf(DP_INFO, "Error Forward SpokeAV: %f\n", err_frw);

	float err_der = md_znrmse(N, out_dims, dst_der_irflash, dst_der_bloch);
	// debug_printf(DP_INFO, "Error Derivative SpokeAV: %f\n", err_der);


	md_free(dst_frw_bloch);
	md_free(dst_der_bloch);
	md_free(dst_frw_irflash);
	md_free(dst_der_irflash);

	if (err_frw > 8.E-3)
		return 0;

	if (err_der > 7.E-3)
		return 0;

	return 1;
}
UT_REGISTER_TEST(test_bloch_irflash_frw_der_spoke_av_z);


static bool test_bloch_irflash_adj_spoke_av_z(void)
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

	complex float* src1 = md_alloc(N, out_dims, CFL_SIZE);
	md_zfill(N, out_dims, src1, 1.0);
	complex float* dst_adj_bloch = md_alloc(N, in_dims, CFL_SIZE);

	struct modBlochFit fit_para = modBlochFit_defaults;

	// IR FLASH characteristics
	fit_para.sequence = 5;
	fit_para.tr = 0.003;
	fit_para.te = 0.001;
	fit_para.fa = 8.;
	fit_para.rfduration = 0.00001;
	fit_para.inversion_pulse_length = 0.;
	fit_para.prep_pulse_length = 0.;
	fit_para.look_locker_assumptions = true; // ! Analytical Assumption: Mxy(t=TE) == Mz(t=0)

	fit_para.averaged_spokes = 5;

	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst_frw_bloch, N, in_dims, src);
	nlop_adjoint(Bloch, N, in_dims, dst_adj_bloch, N, out_dims, src1);

	nlop_free(Bloch);
	md_free(src);
	md_free(src1);


	// Init and apply IR FLASH model operator

	in_dims[COEFF_DIM] = 2;

	complex float* src2 = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src2, 1.0);
	complex float* dst_frw_irflash = md_alloc(N, out_dims, CFL_SIZE);

	complex float* src3 = md_alloc(N, out_dims, CFL_SIZE);
	md_zfill(N, out_dims, src3, 1.0);
	complex float* dst_adj_irflash = md_alloc(N, in_dims, CFL_SIZE);

	// Inversion times
	complex float* TI = md_alloc(N, TI_dims, CFL_SIZE);

	for (int i = 0; i < rep; i++)
		TI[i] =  5 * i * fit_para.tr + 2.5 * fit_para.tr;

	// alpha map
	complex float* fa = md_alloc(N, map_dims, CFL_SIZE);
	md_zfill(N, map_dims, fa, fit_para.fa);

	complex float* alpha = md_alloc(N, map_dims, CFL_SIZE);

	fa_to_alpha(DIMS, map_dims, alpha, fa, fit_para.tr);

	md_free(fa);

	struct nlop_s* T1 = nlop_T1_alpha_in_create(N, map_dims, out_dims, in_dims, TI_dims, TI, alpha, gpu_use);

	nlop_apply(T1, N, out_dims, dst_frw_irflash, N, in_dims, src2);
	nlop_adjoint(T1, N, in_dims, dst_adj_irflash, N, out_dims, src3);

	nlop_free(T1);
	md_free(src2);
	md_free(src3);
	md_free(TI);
	md_free(alpha);

	// Compare operator outputs

	long pos[N] = { [0 ... N - 1] = 0 };

	complex float* map_update_bloch = md_alloc(N, map_dims, CFL_SIZE);
	md_copy_block(N, pos, map_dims, map_update_bloch, in_dims, dst_adj_bloch, CFL_SIZE);

	pos[COEFF_DIM] = 1;
	complex float* map_update_irflash = md_alloc(N, map_dims, CFL_SIZE);
	md_copy_block(N, pos, map_dims, map_update_irflash, in_dims, dst_adj_irflash, CFL_SIZE);

	// dump_cfl("_dst_update_irflash", N, map_dims, map_update_irflash);
	// dump_cfl("_dst_update_bloch", N, map_dims, map_update_bloch);

	float err_adj = md_znrmse(N, map_dims, map_update_bloch, map_update_irflash);
	// debug_printf(DP_INFO, "Error Adjoint (R1 map update) SpokeAV: %f\n", err_adj);

	md_free(dst_frw_bloch);
	md_free(dst_adj_bloch);
	md_free(map_update_bloch);

	md_free(dst_frw_irflash);
	md_free(dst_adj_irflash);
	md_free(map_update_irflash);

	if (err_adj > 6.E-3)
		return 0;

	return 1;
}
UT_REGISTER_TEST(test_bloch_irflash_adj_spoke_av_z);

static bool test_bloch_irflash_frw_init_relax_z(void)
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

	struct modBlochFit fit_para = modBlochFit_defaults;

	// IR FLASH characteristics
	fit_para.sequence = 5;
	fit_para.tr = 0.003;
	fit_para.te = 0.001;
	fit_para.fa = 8.;
	fit_para.rfduration = 0.00001;
	fit_para.inversion_pulse_length = 0.01;
	fit_para.prep_pulse_length = 0.;
	fit_para.look_locker_assumptions = true; // ! Analytical Assumption: Mxy(t=TE) == Mz(t=0)


	struct nlop_s* Bloch = nlop_Bloch_create(N, all_dims, map_dims, out_dims, in_dims, &fit_para, gpu_use);

	nlop_apply(Bloch, N, out_dims, dst_frw_bloch, N, in_dims, src);

	// dump_cfl("_dst_frw_bloch", N, out_dims, dst_frw_bloch);

	nlop_free(Bloch);
	md_free(src);


	// Init and apply IR FLASH model operator

	in_dims[COEFF_DIM] = 2;

	complex float* src2 = md_alloc(N, in_dims, CFL_SIZE);
	md_zfill(N, in_dims, src2, 1.0);

	complex float* dst_frw_irflash = md_alloc(N, out_dims, CFL_SIZE);

	// Set parameter M0 and R1

	// Correction for initial relaxation effects
	// Deichmann, R. (2005),
	// Fast high‐resolution T1 mapping of the human brain.
	// Magn. Reson. Med., 54: 20-27.
	// Equation 9 and 10
	complex float initval[2] = {	-(1. - 2. * expf(-fit_para.inversion_pulse_length)),
					1. / (1. - 2.*fit_para.inversion_pulse_length) };

	long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

	complex float* tmp = md_alloc(DIMS, map_dims, CFL_SIZE);

	for (int i = 0; i < in_dims[COEFF_DIM]; i++) {

		pos[COEFF_DIM] = i;
		md_zfill(N, map_dims, tmp, initval[i]);
		md_copy_block(DIMS, pos, in_dims, src2, map_dims, tmp, CFL_SIZE);
	}

	md_free(tmp);

	// Inversion times
	complex float* TI = md_alloc(N, TI_dims, CFL_SIZE);

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

	// dump_cfl("_dst_frw_irflash", N, out_dims, dst_frw_irflash);

	nlop_free(T1);
	md_free(src2);
	md_free(TI);
	md_free(alpha);

	// Compare operator outputs

	float err_frw = md_znrmse(N, out_dims, dst_frw_irflash, dst_frw_bloch);
	// debug_printf(DP_INFO, "Error Forward: %f\n", err_frw);

	md_free(dst_frw_bloch);
	md_free(dst_frw_irflash);

	if (err_frw > 3.5E-3)
		return 0;

	return 1;
}
UT_REGISTER_TEST(test_bloch_irflash_frw_init_relax_z);
//FIXME: Test for effect on derivatives


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
	fit_para.look_locker_assumptions = true; // ! Analytical Assumption: Mxy(t=TE) == Mz(t=0)

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