
#include <math.h>
#include <stdio.h>

#include "num/ode.h"
#include "simu/bloch.h"
#include "simu/simulation.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "utest.h"


static bool test_ode_bloch_simulation(void)
{   
    float e = 1.E-3;
	float tol = 1.E-3;
    
    float t1 = 1.5;
    float t2 = 0.1;
	float m0 = 1;
    
    int repetition = 500;
	
	struct SimData sim_data;
	
	sim_data.seqData = seqData_defaults;
	sim_data.seqData.seq_type = 1;
	sim_data.seqData.TR = 0.003;
	sim_data.seqData.TE = 0.0015;
	sim_data.seqData.rep_num = repetition;
	sim_data.seqData.spin_num = 1;
	sim_data.seqData.num_average_rep = 1;
	
	sim_data.voxelData = voxelData_defaults;
	sim_data.voxelData.r1 = 1 / t1;
	sim_data.voxelData.r2 = 1 / t2;
	sim_data.voxelData.m0 = m0;
	sim_data.voxelData.w = 0;
	
	sim_data.pulseData = pulseData_defaults;
	sim_data.pulseData.flipangle = 45.;
	sim_data.pulseData.RF_end = 0.0000;			// Choose HARD-PULSE Approximation -> same assumptions as analytical model
	sim_data.gradData = gradData_defaults;
	sim_data.seqtmp = seqTmpData_defaults;
	
	float mxyRefSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
	float saT1RefSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
	float saT2RefSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
	float saDensRefSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
	
	ode_bloch_simulation3( &sim_data, mxyRefSig, saT1RefSig, saT2RefSig, saDensRefSig, NULL );

    //-------------------------------------------------------
    //------------------- T1 Test ---------------------------
    //-------------------------------------------------------
    
    //Prepare temporary steps
    float mxyTmpSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
    float saT1TmpSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
    float saT2TmpSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
	float saDensTmpSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
    
    struct SimData dataT1 = sim_data;
    dataT1.voxelData.r1 += e; //e in [ms] 
    
	ode_bloch_simulation3( &dataT1, mxyTmpSig, saT1TmpSig, saT2TmpSig, saDensTmpSig, NULL );
    
    //Verify gradient
    float err = 0;
	
    for (int i = 0; i < sim_data.seqData.rep_num / sim_data.seqData.num_average_rep; i++) 
        for (int j = 0; j < 3; j++){
            
            err = fabsf( e * saT1RefSig[i][j] - (mxyTmpSig[i][j] - mxyRefSig[i][j]) );
// 			printf("%f \n", err);
            if (err > tol){
                printf("Error T1: (%d,%d)\t=>\t%f\n", i,j, err);
                return false;
            }
            
        }

        
    //-------------------------------------------------------
    //------------------- T2 Test ---------------------------
    //-------------------------------------------------------
    
    struct SimData dataT2 = sim_data;
   dataT2.voxelData.r2 += e; 
    
	ode_bloch_simulation3( &dataT2, mxyTmpSig, saT1TmpSig, saT2TmpSig, saDensTmpSig, NULL );
    
    //Verify gradient
    for (int i = 0; i < sim_data.seqData.rep_num / sim_data.seqData.num_average_rep; i++) {
        for (int j = 0; j < 3; j++){
            
            err = fabsf( e * saT2RefSig[i][j] - (mxyTmpSig[i][j] - mxyRefSig[i][j]) );
        
            if (err > tol){
                printf("Error T2: (%d,%d)\t=>\t%f\n", i,j, err);
                return false;
            }
        }
	}
	
	
	//-------------------------------------------------------
    //------------------- Dens Test -------------------------
    //-------------------------------------------------------
    
    struct SimData dataDens = sim_data;
    dataDens.voxelData.m0 += e; 
    
	ode_bloch_simulation3( &dataDens, mxyTmpSig, saT1TmpSig, saT2TmpSig, saDensTmpSig, NULL );
    
    //Verify gradient
    for (int i = 0; i < sim_data.seqData.rep_num / sim_data.seqData.num_average_rep; i++) {
        for (int j = 0; j < 3; j++){
            
            err = fabsf( e * saDensRefSig[i][j] - (mxyTmpSig[i][j] - mxyRefSig[i][j]) );
        
            if (err > tol){
                printf("Error Dens: (%d,%d)\t=>\t%f\n", i,j, err);
                return false;
            }
        }
	}
    

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_simulation);


static bool test_factorial(void)
{
	long k = 4;
	
	long result = factorial( k );
	
	return (result == 24) ? true : false;

	
}

UT_REGISTER_TEST(test_factorial);


//might get problems if point is much larger... -> precision of factorial calculation
static bool test_sin_integral(void)
{
	float point = 1;
	
	float sin_int = si( point );
	
	float should_be = 0.946083070367183;
// 	printf("Sin.Int: %f,\tTheory: %f\n", sin_int, should_be);
	return (sin_int - should_be < 10E-5) ? true : false;
	
}

UT_REGISTER_TEST(test_sin_integral);


// For visualization of pulse shape uncommend dump_cfl
static bool test_sinc_function(void)
{

	struct PulseData pulseData = pulseData_defaults;
	
	create_rf_pulse(&pulseData, 0, 0.009, 90, 0, 2, 2, 0.46);

	float pulse_length = pulseData.RF_end - pulseData.RF_start;
	float samples = 1000;
	float dt = pulse_length / samples;
	
	long dims[DIMS];
	md_set_dims(DIMS, dims, 1);
	dims[READ_DIM] = samples;
	complex float* storage = md_alloc(DIMS, dims, CFL_SIZE);
	

	for (int i = 0; i < samples ; i ++){
		storage[i] = sinc_pulse(&pulseData, pulseData.RF_start + i * dt );
// 		printf("Sinc: %f\n", tmp);
	}
#if 0
	dump_cfl("_pulse_shape", DIMS, dims, storage);
#endif
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
	
	for(int i = 0; i < dim[0]; i++ )
		for(int j = 0; j < dim[1]; j++ ){
			
			float trf = (trfmin + (float)i/((float)dim[0] - 1.) * (trfmax - trfmin));
			float angle = (amin + (float)j/((float)dim[1] - 1.) * (amax - amin));

			struct SimData data;
	
			data.seqData = seqData_defaults;
			data.seqData.seq_type = 1;
			data.seqData.TR = 10;
			data.seqData.TE = 5;
			data.seqData.rep_num = 1;
			data.seqData.spin_num = 1;
			data.seqData.num_average_rep = 1;
			
			data.voxelData = voxelData_defaults;
			data.voxelData.r1 = 0.;
			data.voxelData.r2 = 0.;
			data.voxelData.m0 = 1;
			data.voxelData.w = 0;
			
			data.pulseData = pulseData_defaults;
			data.pulseData.flipangle = angle;
			data.pulseData.RF_end = trf;
			data.gradData = gradData_defaults;
			data.seqtmp = seqTmpData_defaults;
			
// 			printf("%f,\t%f,\t%f,\t%f,\t%d,\t%f,\t%f,\n", data.seqData.TR, data.seqData.TE, data.voxelData.r1,data.voxelData.r2,data.seqtmp.rep_counter, data.pulseData.RF_end, data.pulseData.flipangle );			
	
			create_rf_pulse( &data.pulseData, 0, trf, angle, 0, 2, 2, 0.46 );
			
			float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } }; //xp[P + 2][N]
			
			float h = 10E-5;
			float tol = 10E-6;
			int N = 3;
			int P = 2;
			
			
			//Starting first pulse
			start_rf_pulse( &data, h, tol, N, P, xp);
			
			float sim_angle = 0.;
			
			if( xp[0][2] >= 0 ){ //case of FA <= 90°
				
				if(data.voxelData.r1 != 0 && data.voxelData.r2 != 0)//for relaxation case
					sim_angle = asinf(xp[0][1] / data.voxelData.m0) / M_PI * 180.;
				else
					sim_angle = asinf(xp[0][1] / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2])) / M_PI * 180.;
			}
			else {//case of FA > 90°
				
				if(data.voxelData.r1 != 0 && data.voxelData.r2 != 0)//for relaxation case
					sim_angle = acosf( fabsf(xp[0][1]) / data.voxelData.m0 ) / M_PI * 180. + 90.;
				else
					sim_angle = acosf( fabsf(xp[0][1]) / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2]) ) / M_PI * 180. + 90.;
				
// 				debug_printf(DP_DEBUG2, "Simangle: %f,\t%f,\t%f,\t%f\n", sim_angle, fabsf(xp[0][1]), sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2]), data.m0 );
// 				debug_printf(DP_DEBUG2, "Arg: %f,\t%f,\t%f\n", (fabsf(xp[0][1]) / data.m0), fabsf(xp[0][1]) / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2]),
// 							 acosf( fabsf(xp[0][1]) / data.m0 ) );
			}
				
			
			float err = fabsf( data.pulseData.flipangle - sim_angle );
			
			if (err > 10E-2)
			{
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
	//------------------------------------------------------------
    //------------ Parameter for simulation  --------------------
    //------------------------------------------------------------
    float angle = 45.;
	float repetition = 100;
    float aver_num = 1;
	
	//Parameter for analytical model calculation
	float fa = angle * M_PI / 180.; //conversion to radians
	
    //------------------------------------------------------------
    //--------------  Create simulation dataset  -----------------
    //------------------------------------------------------------
    long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = repetition / aver_num;
     
    //------------------------------------------------------------
    //---------  Simulation of phantom data with ODE -------------
    //------------------------------------------------------------

	for(int i = 0; i < dims[0]; i++ )
		for(int j = 0; j < dims[1]; j++ ){
			
			float t1n = WATER_T1;	float t2n = WATER_T2;	float m0n = 1;
			

			
			struct SimData sim_data;
	
			sim_data.seqData = seqData_defaults;
			sim_data.seqData.seq_type = 1;
			sim_data.seqData.TR = 0.003;
			sim_data.seqData.TE = 0.0015;
			sim_data.seqData.rep_num = repetition;
			sim_data.seqData.spin_num = 1;
			sim_data.seqData.num_average_rep = aver_num;
			
			sim_data.voxelData = voxelData_defaults;
			sim_data.voxelData.r1 = 1 / t1n;
			sim_data.voxelData.r2 = 1 / t2n;
			sim_data.voxelData.m0 = m0n;
			sim_data.voxelData.w = 0;
			
			sim_data.pulseData = pulseData_defaults;
			sim_data.pulseData.flipangle = angle;
			sim_data.pulseData.RF_end = 0.0000;			// Choose HARD-PULSE Approximation -> same assumptions as analytical model
			sim_data.gradData = gradData_defaults;
			sim_data.seqtmp = seqTmpData_defaults;
			
			float mxySig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saR1Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saR2Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saDensSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			
			ode_bloch_simulation3( &sim_data, mxySig, saR1Sig, saR2Sig, saDensSig, NULL );
			
			
			//------------------------------------------------------------
			//--------  Simulation of phantom data analytically ----------
			//------------------------------------------------------------
			float t1s = 0.; float s0 = 0.; float stst = 0.; float inv = 0.;
			
			//simulation based on analytical model: TR << T_{1,2}
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
			
			
			//------------------------------------------------------------
			//---------------------- Calculate Error ---------------------
			//------------------------------------------------------------
			float out_simu;
			float out_theory;
			float err = 0;
			
			for(int z = 0; z < dims[TE_DIM]; z++){
				
				
				out_theory = fabsf( stst * ( 1 - inv * expf( - ( z * sim_data.seqData.TR + sim_data.seqData.TR )  / t1s )) ); //Does NOT include phase information! //+data.TR through alpha/2 preparation
				
				out_simu = cabsf( mxySig[z][1] + mxySig[z][0] * I );
				
				err = fabsf( out_simu - out_theory );
				
// 				debug_printf(DP_INFO, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
				
				if (err > 10E-4)
				{
					debug_printf(DP_ERROR, "err: %f,\t out_simu: %f,\t out_theory: %f\n", err, out_simu, out_theory);
					debug_printf(DP_ERROR, "Error in sequence test\n see: -> test_simulation() in test_ode_simu.c\n");
					return 0;
				}
			}  
		}
	
	
#endif
	return 1;	
}

UT_REGISTER_TEST(test_simulation);

