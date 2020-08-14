
#include <math.h>
#include <stdio.h>

#include "num/ode.h"
#include "simu/bloch.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/pulse.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "utest.h"


static bool test_ode_bloch_simulation(void)
{
#if 1
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
	
	ode_bloch_simulation3(&sim_data, mxy_ref_sig, sa_r1_ref_sig, sa_r2_ref_sig, sa_m0_ref_sig);

	//-------------------------------------------------------
	//------------------- T1 Test ---------------------------
	//-------------------------------------------------------
	complex float mxy_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r1_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r2_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_m0_tmp_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	
	struct sim_data data_r1 = sim_data;
	data_r1.voxel.r1 += e; //e in [ms] 
	
	ode_bloch_simulation3(&data_r1, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig);
	
	//Verify gradient
	float err = 0;
	
	for (int i = 0; i < sim_data.seq.rep_num / sim_data.seq.num_average_rep; i++) 
		for (int j = 0; j < 3; j++) {
		
			err = fabsf( e * sa_r1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]) );
			
			if (err > tol) {
				
				printf("Error T1: (%d,%d)\t=>\t%f\n", i,j, err);
				return false;
			}
			
		}

	//-------------------------------------------------------
	//------------------- T2 Test ---------------------------
	//-------------------------------------------------------
	struct sim_data data_r2 = sim_data;
	data_r2.voxel.r2 += e; 
	
	ode_bloch_simulation3(&data_r2, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig);
	
	//Verify gradient
	for (int i = 0; i < sim_data.seq.rep_num / sim_data.seq.num_average_rep; i++)
		for (int j = 0; j < 3; j++) {
		
			err = fabsf( e * sa_r2_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]) );
			
			if (err > tol) {
				
				printf("Error T2: (%d,%d)\t=>\t%f\n", i,j, err);
				return false;
			}
		}
	
	//-------------------------------------------------------
	//------------------- Dens Test -------------------------
	//-------------------------------------------------------
	struct sim_data data_m0 = sim_data;
	data_m0.voxel.m0 += e; 
	
	ode_bloch_simulation3(&data_m0, mxy_tmp_sig, sa_r1_tmp_sig, sa_r2_tmp_sig, sa_m0_tmp_sig);
	
	//Verify gradient
	for (int i = 0; i < sim_data.seq.rep_num / sim_data.seq.num_average_rep; i++)
		for (int j = 0; j < 3; j++) {
		
			err = fabsf( e * sa_m0_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]) );
			
			if (err > tol) {
				
				printf("Error Dens: (%d,%d)\t=>\t%f\n", i,j, err);
				return false;
			}
		}
#endif
	return 1;
}

UT_REGISTER_TEST(test_ode_bloch_simulation);



// For visualization of pulse shape uncommend dump_cfl
static bool test_sinc_function(void)
{
	struct simdata_pulse pulse = simdata_pulse_defaults;
	
	pulse_create(&pulse, 0, 0.009, 90, 0, 2, 2, 0.46);

	float pulse_length = pulse.rf_end - pulse.rf_start;
	float samples = 1000;
	float dt = pulse_length / samples;
	
	long dims[DIMS];
	
	md_set_dims(DIMS, dims, 1);
	dims[READ_DIM] = samples;
	
	complex float* storage = md_alloc(DIMS, dims, CFL_SIZE);
	

	for (int i = 0; i < samples ; i ++)
		storage[i] = pulse_sinc(&pulse, pulse.rf_start + i * dt );
#if 0
	dump_cfl("_pulse_shape", DIMS, dims, storage);
#endif
	md_free(storage);
	return 1;	
}

UT_REGISTER_TEST(test_sinc_function);


static bool test_RF_pulse(void)
{
#if 1
	
	long dim[DIMS] = { [0 ... DIMS - 1] = 1 };

	dim[0] = 10;
	dim[1] = 10;

	float trfmin = 0.0001;
	float trfmax = 0.1;
	float amin = 1.;
	float amax = 180.;
	
	for (int i = 0; i < dim[0]; i++ )
		for (int j = 0; j < dim[1]; j++ ) {
			
			float trf = (trfmin + (float)i/((float)dim[0] - 1.) * (trfmax - trfmin));
			float angle = (amin + (float)j/((float)dim[1] - 1.) * (amax - amin));

			struct sim_data data;
	
			data.seq = simdata_seq_defaults;
			data.seq.seq_type = 1;
			data.seq.tr = 10;
			data.seq.te = 5;
			data.seq.rep_num = 1;
			data.seq.spin_num = 1;
			data.seq.num_average_rep = 1;
			
			data.voxel = simdata_voxel_defaults;
			data.voxel.r1 = 0.;
			data.voxel.r2 = 0.;
			data.voxel.m0 = 1;
			data.voxel.w = 0;
			
			data.pulse = simdata_pulse_defaults;
			data.pulse.flipangle = angle;
			data.pulse.rf_end = trf;
			data.grad = simdata_grad_defaults;
			data.tmp = simdata_tmp_defaults;
	
			pulse_create(&data.pulse, 0, trf, angle, 0, 2, 2, 0.46);
			
			float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } }; //xp[P + 2][N]
			
			float h = 10E-5;
			float tol = 10E-6;
			int N = 3;
			int P = 2;
			
			start_rf_pulse( &data, h, tol, N, P, xp);
			
			float sim_angle = 0.;
			
			if (xp[0][2] >= 0) { //case of FA <= 90°
				
				if(data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = asinf(xp[0][1] / data.voxel.m0) / M_PI * 180.;
				else
					sim_angle = asinf(xp[0][1] / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2])) / M_PI * 180.;
			}
			else {//case of FA > 90°
				
				if(data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = acosf( fabsf(xp[0][1]) / data.voxel.m0 ) / M_PI * 180. + 90.;
				else
					sim_angle = acosf( fabsf(xp[0][1]) / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2]) ) / M_PI * 180. + 90.;
			}
				
			
			float err = fabsf( data.pulse.flipangle - sim_angle );
			
			if (err > 10E-2) {
				
				debug_printf(DP_WARN, "Error in rf-pulse test!\n see -> utests/test_ode_simu.c\n");
				return 0;
			}
		}
		
#endif
	return 1;


}

UT_REGISTER_TEST(test_RF_pulse);




static bool test_simulation(void)
{
#if 1

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
	
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1 / t1n;
	sim_data.voxel.r2 = 1 / t2n;
	sim_data.voxel.m0 = m0n;
	sim_data.voxel.w = 0;
	
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = angle;
	sim_data.pulse.rf_end = 0.0000;			// Choose HARD-PULSE Approximation -> same assumptions as analytical model
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;
	
	complex float mxy_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r1_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_r2_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float sa_m0_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	
	ode_bloch_simulation3(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);
	
	//------------------------------------------------------------
	//--------  Simulation of phantom data analytically ----------
	//------------------------------------------------------------
	float t1s = 0.; float s0 = 0.; float stst = 0.; float inv = 0.;
	
	//simulation based on analytical model: tr << T_{1,2}
	//based on: 
	//Schmitt, P. , Griswold, M. A., Jakob, P. M., Kotas, M. , Gulani, V. , Flentje, M. and Haase, A. (2004), 
	//Inversion recovery TrueFISP: Quantification of T1, T2, and spin density. 
	//Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058
	
	/* Ehses, P. , Seiberlich, N. , Ma, D. , Breuer, F. A., Jakob, P. M., Griswold, M. A. and Gulani, V. (2013),
	 * IR TrueFISP with a golden‐ratio‐based radial readout: Fast quantification of T1, T2, and proton density. 
	 * Magn Reson Med, 69: 71-81. doi:10.1002/mrm.24225
	 */
	
	t1s = 1 / ( (cosf( fa/2. )*cosf( fa/2. ))/t1n + (sinf( fa/2. )*sinf( fa/2. ))/t2n );
	s0 = m0n * sinf( fa/2. );
	stst = m0n * sinf(fa) / ( (t1n/t2n + 1) - cosf(fa) * (t1n/t2n -1) );
	inv = 1 + s0 / stst;
	
	float out_simu;
	float out_theory;
	float err = 0;
	
	for (int z = 0; z < repetition; z++) {
		
		out_theory = fabsf( stst * ( 1 - inv * expf( - ( z * sim_data.seq.tr + sim_data.seq.tr )  / t1s )) ); //Does NOT include phase information! //+data.tr through alpha/2 preparation
		
		out_simu = cabsf( mxy_sig[z][1] + mxy_sig[z][0] * I );
		
		err = fabsf( out_simu - out_theory );
		
		if (10E-4 < err ) {
			
			debug_printf(DP_ERROR, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
			debug_printf(DP_ERROR, "Error in sequence test\n see: -> test_simulation() in test_ode_simu.c\n");
			return 0;
		}
	}  
		
	
	
#endif
	return 1;	
}

UT_REGISTER_TEST(test_simulation);




static bool test_RF_pulse_matexp(void)
{
#if 1

	long dim[DIMS] = { [0 ... DIMS - 1] = 1 };
	dim[0] = 10;
	dim[1] = 10;
	

	float trfmin = 0.0001;
	float trfmax = 0.1;
	float amin = 1.;
	float amax = 180.;
	
	for (int i = 0; i < dim[0]; i++ )
		for (int j = 0; j < dim[1]; j++ ) {
			
			float trf = (trfmin + (float)i/((float)dim[0] - 1.) * (trfmax - trfmin));
			float angle = (amin + (float)j/((float)dim[1] - 1.) * (amax - amin));

			struct sim_data data;
	
			data.seq = simdata_seq_defaults;
			data.seq.seq_type = 1;
			data.seq.tr = 10;
			data.seq.te = 5;
			data.seq.rep_num = 1;
			data.seq.spin_num = 1;
			data.seq.num_average_rep = 1;
			
			data.voxel = simdata_voxel_defaults;
			data.voxel.r1 = 0.;
			data.voxel.r2 = 0.;
			data.voxel.m0 = 1;
			data.voxel.w = 0;
			
			data.pulse = simdata_pulse_defaults;
			data.pulse.flipangle = angle;
			data.pulse.rf_end = trf;
			data.grad = simdata_grad_defaults;
			data.tmp = simdata_tmp_defaults;
			
			data.pulse.pulse_applied = true;
				
	
			pulse_create(&data.pulse, 0, trf, angle, 0, 2, 2, 0.46);
			
			enum { N = 10 };
			
			float x0[N] = { 0., 0., 1., 0., 0., 0., 0., 0., 0., 1. };
			float x1[N];
			
			
			//Starting first pulse
			float m[N][N];
			mat_exp_simu( N, data.pulse.rf_end, m, &data);
			
			for (int i = 0; i < N; i++) {

				x1[i] = 0.;

				for (int j = 0; j < N; j++)
					x1[i] += m[j][i] * x0[j];
			}
	
			
			float sim_angle = 0.;
			
			if (x1[2] >= 0) { //case of FA <= 90°
				
				if(data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = asinf(x1[1] / data.voxel.m0) / M_PI * 180.;
				else
					sim_angle = asinf(x1[1] / sqrtf(x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2])) / M_PI * 180.;
			}
			else {//case of FA > 90°
				
				if (data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = acosf( fabsf(x1[1]) / data.voxel.m0 ) / M_PI * 180. + 90.;
				else
					sim_angle = acosf( fabsf(x1[1]) / sqrtf(x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2]) ) / M_PI * 180. + 90.;
			}
			
			
			float err = fabsf(data.pulse.flipangle - sim_angle);
			
			if (10E-2 < err)
			{
				debug_printf(DP_WARN, "Error in mat rf-pulse test!\n see -> utests/test_ode_simu.c\n");
				return 0;
			}
		}
		
#endif
	return 1;


}

UT_REGISTER_TEST(test_RF_pulse_matexp);


static bool test_matrix_exp_simulation(void)
{
#if 1

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
	sim_data.voxel.r1 = 1 / t1n;
	sim_data.voxel.r2 = 1 / t2n;
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
	
	ode_bloch_simulation3(&sim_ode, mxySig_ode, saR1Sig_ode, saR2Sig_ode, saDensSig_ode);
	
	complex float mxySig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saR1Sig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saR2Sig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	complex float saDensSig_matexp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
	
	matrix_bloch_simulation(&sim_data, mxySig_matexp, saR1Sig_matexp, saR2Sig_matexp, saDensSig_matexp);

	float tol = 10E-3;
	float err;
	
	for (int rep = 0; rep < repetition; rep++)
		for ( int dim = 0; dim < 3; dim++) {
			
			err = fabsf( mxySig_matexp[rep][dim] - mxySig_ode[rep][dim] );
			if ( err > tol )
				return 0;
			
			err = fabsf( saR1Sig_matexp[rep][dim] - saR1Sig_ode[rep][dim] );
			if ( err > tol )
				return 0;
			
			err = fabsf( saR2Sig_matexp[rep][dim] - saR2Sig_ode[rep][dim] );
			if ( err > tol )
				return 0;
			
			err = fabsf( saDensSig_matexp[rep][dim] - saDensSig_ode[rep][dim] );
			if ( err > tol )
				return 0;
		}
	
	
	#endif
	return 1;	
}

UT_REGISTER_TEST(test_matrix_exp_simulation);

