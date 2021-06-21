/* Author:
 * 	2018-2021 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */

#include <math.h>
#include <stdio.h>

#include "num/ode.h"
#include "simu/bloch.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/sim_rot.h"
#include "simu/pulse.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "utest.h"


static bool test_ode_bloch_simulation(void)
{
	float e = 1.E-3;

	float tol = 1.E-4;

	float t1 = 1.5;
	float t2 = 0.1;
	float m0 = 1;

	int repetition = 500;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = 1;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
	sim_data.seq.num_average_rep = 1;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1 / t1;
	sim_data.voxel.r2 = 1 / t2;
	sim_data.voxel.m0 = m0;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 45.;
	sim_data.pulse.rf_end = 0.0009;
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	complex float mxy_ref_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r1_ref_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r2_ref_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_m0_ref_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_b1_ref_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

	ode_bloch_simulation3(&sim_data, mxy_ref_sig, sa_r1_ref_sig, sa_r2_ref_sig, sa_m0_ref_sig, sa_b1_ref_sig);

	//-------------------------------------------------------
	//------------------- T1 Test ---------------------------
	//-------------------------------------------------------
	complex float mxy_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r1_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r2_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_m0_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_b1_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

	struct sim_data data_r1 = sim_data;

	data_r1.voxel.r1 += e;

	ode_bloch_simulation3(&data_r1, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	//Verify gradient
	float err = 0;

	for (int i = 0; i < sim_data.seq.rep_num / sim_data.seq.num_average_rep; i++) 
		for (int j = 0; j < 3; j++) {

			err = cabsf(e * sa_r1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {
				
				printf("Error T1: (%d,%d)\t=>\t%f\n", i,j, err);
				return false;
			}
	
		}

	//-------------------------------------------------------
	//------------------- T2 Test ---------------------------
	//-------------------------------------------------------
	struct sim_data data_r2 = sim_data;

	data_r2.voxel.r2 += e; 

	ode_bloch_simulation3(&data_r2, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	//Verify gradient
	for (int i = 0; i < sim_data.seq.rep_num / sim_data.seq.num_average_rep; i++)
		for (int j = 0; j < 3; j++) {
		
			err = cabsf(e * sa_r2_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));
			
			if (tol < err) {
				
				printf("Error T2: (%d,%d)\t=>\t%f\n", i,j, err);
				return false;
			}
		}

	//-------------------------------------------------------
	//-------------------- M0 Test --------------------------
	//-------------------------------------------------------
	struct sim_data data_m0 = sim_data;

	data_m0.voxel.m0 += e;

	ode_bloch_simulation3(&data_m0, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	//Verify gradient
	for (int i = 0; i < sim_data.seq.rep_num / sim_data.seq.num_average_rep; i++)
		for (int j = 0; j < 3; j++) {

			err = cabsf(e * sa_m0_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err) {

				printf("Error M0: (%d,%d)\t=>\t%f\n", i,j, err);
				return false;
			}
		}

	//-------------------------------------------------------
	//-------------------- B1 Test --------------------------
	//-------------------------------------------------------
	struct sim_data data_b1 = sim_data;

	data_b1.pulse.flipangle += e;

	ode_bloch_simulation3(&data_b1, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig, sa_b1_tmp_sig);

	//Verify gradient
	for (int i = 0; i < sim_data.seq.rep_num / sim_data.seq.num_average_rep; i++)
		for (int j = 0; j < 3; j++) {

			err = cabsf(e * sa_m0_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (1.E-3 < err) {

				printf("Error B1: (%d,%d)\t=>\t%f\n", i,j, err);
				return false;
			}
		}

	return 1;
}

UT_REGISTER_TEST(test_ode_bloch_simulation);


static bool test_ode_irbssfp_simulation(void)
{
	float angle = 45.;
	float repetition = 100;
	float aver_num = 1;

	float fa = angle * M_PI / 180.;

	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = 1;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
	sim_data.seq.num_average_rep = aver_num;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1/t1n;
	sim_data.voxel.r2 = 1/t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;
	sim_data.pulse.rf_end = 0.;			// Choose HARD-PULSE Approximation -> same assumptions as analytical model
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	complex float mxy_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r1_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r2_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_m0_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_b1_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

	ode_bloch_simulation3(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig, sa_b1_sig);

	//------------------------------------------------------------
	//--------  Simulation of phantom data analytically ----------
	//------------------------------------------------------------
	/*simulation based on analytical model. Assumption: TR << T_{1,2}
	 *
	 * Schmitt, P. , Griswold, M. A., Jakob, P. M., Kotas, M. , Gulani, V. , Flentje, M. and Haase, A. (2004), 
	 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density. 
	 * Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058
	 *
	 * Ehses, P. , Seiberlich, N. , Ma, D. , Breuer, F. A., Jakob, P. M., Griswold, M. A. and Gulani, V. (2013),
	 * IR TrueFISP with a golden‐ratio‐based radial readout: Fast quantification of T1, T2, and proton density. 
	 * Magn Reson Med, 69: 71-81. doi:10.1002/mrm.24225
	 */

	float t1s = 1 / ((cosf(fa/2.)*cosf(fa/2.))/t1n + (sinf(fa/2.)*sinf(fa/2.))/t2n);
	float s0 = m0n * sinf(fa/2.);
	float stst = m0n * sinf(fa) / ((t1n/t2n + 1) - cosf(fa) * (t1n/t2n-1));
	float inv = 1 + s0 / stst;

	float out_simu;
	float out_theory;

	float err = 0;

	for (int z = 0; z < repetition; z++) {

		out_theory = fabsf(stst * (1 - inv * expf(-((float)(z+1) * sim_data.seq.tr)/t1s))); //Does NOT include phase information! //+data.tr through alpha/2 preparation

		out_simu = cabsf(mxy_sig[z][1] + mxy_sig[z][0]*I);

		err = fabsf(out_simu - out_theory);

		if (10E-4 < err) {

			debug_printf(DP_ERROR, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
			debug_printf(DP_ERROR, "Error in sequence test\n see: -> test_simulation() in test_ode_simu.c\n");
			return 0;
		}
	}
	return 1;	
}

UT_REGISTER_TEST(test_ode_irbssfp_simulation);


static bool test_matrix_ode_simu_comparison(void)
{
	float angle = 45.;
	float repetition = 100;
	float aver_num = 1;
	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = 1;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
	sim_data.seq.num_average_rep = aver_num;
	sim_data.seq.inversion_pulse_length = 0.01;
	sim_data.seq.prep_pulse_length = 0.001;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1/t1n;
	sim_data.voxel.r2 = 1/t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;
	sim_data.pulse.rf_end = 0.0009;
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	struct sim_data sim_ode = sim_data;

	complex float mxySig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saR1Sig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saR2Sig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saDensSig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float sa_b1_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];

	ode_bloch_simulation3(&sim_ode, mxySig_ode, saR1Sig_ode, saR2Sig_ode, saDensSig_ode, sa_b1_ode);

	complex float mxySig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saR1Sig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saR2Sig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saDensSig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_b1_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

	matrix_bloch_simulation(&sim_data, mxySig_matexp, saR1Sig_matexp, saR2Sig_matexp, saDensSig_matexp, sa_b1_matexp);

	float tol = 10E-3;
	float err;

	for (int rep = 0; rep < repetition; rep++)
		for ( int dim = 0; dim < 3; dim++) {

			err = cabsf(mxySig_matexp[rep][dim]-mxySig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			err = cabsf(saR1Sig_matexp[rep][dim]-saR1Sig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			err = cabsf(saR2Sig_matexp[rep][dim]-saR2Sig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			err = cabsf(saDensSig_matexp[rep][dim]-saDensSig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			// FIXME: Test not very good for long TR trains, first TRs work nicely
			// err = cabsf( sa_b1_matexp[rep][dim] - sa_b1_ode[rep][dim] );
			// debug_printf(DP_ERROR, "err: %f \t<=\t%f,\t%f\n", err, cabsf(sa_b1_matexp[rep][dim]), cabsf(sa_b1_ode[rep][dim]));
			// if ( 10E-1 < err )
			// 	return 0;
		}
	return 1;	
}

UT_REGISTER_TEST(test_matrix_ode_simu_comparison);


static bool test_ode_hp_simu(void)
{
	float angle = 45.;
	float repetition = 20;
	float aver_num = 1;
	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = 1;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
	sim_data.seq.num_average_rep = aver_num;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1/t1n;
	sim_data.voxel.r2 = 1/t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;
	sim_data.pulse.rf_end = 0.0001;
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	struct sim_data sim_ode = sim_data;

	complex float mxySig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saR1Sig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saR2Sig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saDensSig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float sa_b1_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];

	ode_bloch_simulation3(&sim_ode, mxySig_ode, saR1Sig_ode, saR2Sig_ode, saDensSig_ode, sa_b1_ode);


	sim_data.pulse.rf_end = 0.;

	complex float mxySig_hp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saR1Sig_hp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saR2Sig_hp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saDensSig_hp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_b1_hp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

	ode_bloch_simulation3(&sim_data, mxySig_hp, saR1Sig_hp, saR2Sig_hp, saDensSig_hp, sa_b1_hp);

	float tol = 10E-4;
	float err;

	for (int rep = 0; rep < repetition; rep++)
		for ( int dim = 0; dim < 3; dim++) {

			err = cabsf(mxySig_hp[rep][dim]-mxySig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			err = cabsf(saR1Sig_hp[rep][dim]-saR1Sig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			err = cabsf(saR2Sig_hp[rep][dim]-saR2Sig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			err = cabsf(saDensSig_hp[rep][dim]-saDensSig_ode[rep][dim]);
			if ( err > tol )
				return 0;

			// FIXME: Test is not working for B1 derivative!
			// err = cabsf(sa_b1_hp[rep][dim]-sa_b1_ode[rep][dim]);
		}
	return 1;
}

UT_REGISTER_TEST(test_ode_hp_simu);


static bool test_ode_hp_off_simu(void)
{
	float angle = 90.;
	float repetition = 1.;
	float aver_num = 1.;
	float m0n = 1.;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = 2;	// Use FLASH here, because it does not have any preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
	sim_data.seq.num_average_rep = aver_num;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 0.;	// Turn off relaxation
	sim_data.voxel.r2 = 0.;	// Turn off relaxation
	sim_data.voxel.m0 = m0n;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;
	sim_data.pulse.rf_end = 0.;
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	// for w == 0 -> Mxy = 0+1*I

	// Set off-resonance so that magnetization is roatated by 90 degree within TE
	// Goal: Mxy = 1+0*I
	sim_data.voxel.w = M_PI/2 / sim_data.seq.te;

	struct sim_data sim_ode = sim_data;

	complex float mxySig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saR1Sig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saR2Sig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float saDensSig_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];
	complex float sa_b1_ode[sim_ode.seq.rep_num / sim_ode.seq.num_average_rep][3];

	ode_bloch_simulation3(&sim_ode, mxySig_ode, saR1Sig_ode, saR2Sig_ode, saDensSig_ode, sa_b1_ode);

#if 0
	bart_printf("M\n x: %f+i*%f,\ty: %f+i*%f,\tz: %f+i*%f\n", crealf(mxySig_ode[0][0]), cimagf(mxySig_ode[0][0])
								, crealf(mxySig_ode[0][1]), cimagf(mxySig_ode[0][1])
								, crealf(mxySig_ode[0][2]), cimagf(mxySig_ode[0][2]) );

	bart_printf("Err\n x: %f,\ty: %f,\tz: %f\n",	fabsf(cabsf(mxySig_ode[0][0]) - 1.),
							fabsf(cabsf(mxySig_ode[0][1]) - 0.),
							fabsf(cabsf(mxySig_ode[0][2]) - 0.) );
#endif
	float tol = 10E-5;

	UT_ASSERT(	(fabsf(cabsf(mxySig_ode[0][0]) - 1.) < tol) &&
			(fabsf(cabsf(mxySig_ode[0][1]) - 0.) < tol) &&
			(fabsf(cabsf(mxySig_ode[0][2]) - 0.) < tol) );

	return true;
}

UT_REGISTER_TEST(test_ode_hp_off_simu);


static bool test_bloch_excitation(void)
{
	float x0[3] = { 0., 0., 1. };
	float x[3] = { 0., 0., 1. };
	float ref[3] = { 0., 1., 0. };

	bloch_excitation2(x, x0, M_PI/2., 0.);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - ref[i], 2.);

	UT_ASSERT(err2 < 1.E-6);
}

UT_REGISTER_TEST(test_bloch_excitation);


static bool test_bloch_excitation2(void)
{
	float x0[3] = { 0., 0., 1. };
	float x[3] = { 0., 0., 1. };
	float ref[3] = { -1., 0., 0. };

	bloch_excitation2(x, x0, M_PI/2., M_PI/2.);

	float err2 = 0.;

	for (int i = 0; i < 3; i++)
		err2 += powf(x[i] - ref[i], 2.);

	UT_ASSERT(err2 < 1.E-6);
}

UT_REGISTER_TEST(test_bloch_excitation2);

