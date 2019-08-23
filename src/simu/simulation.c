#include <stdio.h>
#include <memory.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#include "simu/bloch.h"
#include "simu/pulse.h"

#include "num/ode.h"
#include "misc/opts.h"
#include "num/init.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "misc/debug.h"

#include "simu/simulation.h"


const struct PulseData pulseData_defaults = {

	.pulse_length = 1.,
	.RF_start = 0.,
	.RF_end = 0.009,
	.flipangle = 1.,
	.phase = 0.,
	.nl = 2.,
	.nr = 2.,
	.n = 2.,
	.t0 = 1.,
	.alpha = 0.46,
	.A = 1.,
	.energy_scale = 1.,
	.pulse_applied = false,
};


const struct VoxelData voxelData_defaults = {

	.r1 = 0.,
	.r2 = 0.,
	.m0 = 1.,
	.w = 0.,
	.spin_ensamble = false,
};


const struct SeqData seqData_defaults = {

	.seq_type = 1,
	.TR = 0.004,
	.TE = 0.002,
	.rep_num = 1,
	.spin_num = 1,
	.num_average_rep = 1,
	
	.slice_profile = NULL,
	.variable_fa = NULL,
};


const struct SeqTmpData seqTmpData_defaults = {

	.t = 0.,
	.rep_counter = 0,
	.spin_counter = 0,
};


const struct GradData gradData_defaults = {

	.gb = { 0., 0., 0. },	/*gradients, example: GAMMA_H1 * SKYRA_GRADIENT * 0.0001*/
	.gb_eff = { 0., 0., 0.},	/*storage for effective gradients*/
};





void bloch_pdy2(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->r1, data->r2, data->gb_eff);
}

void bloch_pdp2(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_pdp((float(*)[3])out, in, data->r1, data->r2 , data->gb_eff);
}


void bloch_pdy3(void* _data, float* out, float t, const float* in)
{
	struct SimData* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->voxelData.r1, data->voxelData.r2, data->gradData.gb_eff);
}

void bloch_pdp3(void* _data, float* out, float t, const float* in)
{
	struct SimData* data = _data;
	(void)t;

	bloch_pdp((float(*)[3])out, in, data->voxelData.r1, data->voxelData.r2, data->gradData.gb_eff);
}


void bloch_simu_fun2(void* _data, float* out, float t, const float* in)
{
	struct SimData* data = _data;
	
	if (data->pulseData.pulse_applied) { 
		
		float w1 = sinc_pulse(&data->pulseData, t);
		
		data->gradData.gb_eff[0] = cosf(data->pulseData.phase) * w1 + data->gradData.gb[0];
		data->gradData.gb_eff[1] = sinf(data->pulseData.phase) * w1 + data->gradData.gb[1];
		
	}
	else {
		
		data->gradData.gb_eff[0] = data->gradData.gb[0];
		data->gradData.gb_eff[1] = data->gradData.gb[1];
	}
	
	data->gradData.gb_eff[2] = data->gradData.gb[2] + data->voxelData.w;
	
	bloch_ode(out, in, data->voxelData.r1, data->voxelData.r2 , data->gradData.gb_eff);
	
	
}



void isochromDistribution(void* _data, float *isochromats){
	
	struct SimData* data = _data;
	
	float s = 1.;		//scaling parameters
	float t = 0.;		// location of max
	float maximum = 0.;

	float randomNumber;
	srand( 4 );
	
	float isoTmp[data->seqData.spin_num];
	
	//Creating Distribution...
	for (int i = 0; i < data->seqData.spin_num; i++) {
		
		randomNumber = 0.02 + (float)rand() / ( (float)RAND_MAX / (0.98 - 0.02) + 1. ); //numbers needed to supress extrema for randomNumber=>1
		
		//...in this case a Cauchy one based on its inverse cdf.
		isoTmp[i] = s * tan( M_PI * ( randomNumber - 0.5 ) ) + t;

		maximum = fmax( maximum, fabsf(isoTmp[i]) );
	}
	
	//Assigning frequencies up to pi/2
	for (int i = 0; i < data->seqData.spin_num; i++) 
		isochromats[i] = ( isoTmp[i]/maximum ) * M_PI / data->seqData.TR;

}

//If ADC gets phase, it has to be corrected manually
void ADCcorr(int N, int P, float out[P + 2][N], float in[P + 2][N]){

	float corrAngle = M_PI; // for bSSFP

	for (int i = 0; i < P + 2; i ++) {
		
		out[i][0] = in[i][0] * cosf(corrAngle) + in[i][1] * sinf(corrAngle);
		out[i][1] = in[i][0] * sinf(corrAngle) + in[i][1] * cosf(corrAngle);
		out[i][2] = in[i][2];
	}
}

static void collect_signal(void* _data, int N, int P, float *mxySignal, float *saT1Signal, float *saT2Signal, float *densSignal, float xp[P + 2][N])
{    
	struct SimData* data = _data;

	float tmp[4][3] = { { 0. }, { 0. }, { 0. }, { 0. } }; 

	ADCcorr(N, P, tmp, xp);

	for (int i = 0; i < N; i++) {

		mxySignal[ (i * data->seqData.spin_num * (data->seqData.rep_num) ) + ( (data->seqtmp.rep_counter) * data->seqData.spin_num) + data->seqtmp.spin_counter ] = (data->seqtmp.rep_counter % 2 == 0) ? tmp[0][i] : xp[0][i];
		saT1Signal[ (i * data->seqData.spin_num * (data->seqData.rep_num) ) + ( (data->seqtmp.rep_counter) * data->seqData.spin_num) + data->seqtmp.spin_counter ] = (data->seqtmp.rep_counter % 2 == 0) ? tmp[1][i] : xp[1][i];
		saT2Signal[ (i * data->seqData.spin_num * (data->seqData.rep_num) ) + ( (data->seqtmp.rep_counter) * data->seqData.spin_num) + data->seqtmp.spin_counter ] = (data->seqtmp.rep_counter % 2 == 0) ? tmp[2][i] : xp[2][i];
		densSignal[ (i * data->seqData.spin_num * (data->seqData.rep_num) ) + ( (data->seqtmp.rep_counter) * data->seqData.spin_num) + data->seqtmp.spin_counter ] = (data->seqtmp.rep_counter % 2 == 0) ? tmp[3][i] : xp[3][i];
	}
}


void create_rf_pulse(void* _pulseData, float RF_start, float RF_end, float angle /*[°]*/, float phase, float nl, float nr, float alpha)
{
	struct PulseData* pulseData = _pulseData;

	// For windowed sinc-pluses only
	pulseData->RF_start = RF_start;
	pulseData->RF_end = RF_end;
	pulseData->pulse_length = RF_end - RF_start;
	pulseData->flipangle = angle;
	pulseData->phase = phase;
	pulseData->nl = nl;
	pulseData->nr = nr;
	pulseData->n = MAX(nl, nr);
	pulseData->t0 = pulseData->pulse_length / ( 2 + (nl-1) + (nr-1) );
	pulseData->alpha = alpha;
	pulseData->A = 1;

	float pulse_energy = get_pulse_energy(pulseData);

	float calibration_energy = 0.991265;//2.3252; // turns M by 90°

	// change scale to reach desired flipangle
	pulseData->A = (calibration_energy / pulse_energy) / 90 * angle;
}


//Module for RF-pulses
void start_rf_pulse(void* _data, float h, float tol, int N, int P, float xp[P + 2][N])
{	
	struct SimData* data = _data;
	
	data->pulseData.pulse_applied = true;

	if (0. == data->pulseData.RF_end) {	
		// Hard-Pulse Approximation:
		// !! Does not work with sensitivity output !!
		
		float xtmp = xp[0][0];
		float ytmp = xp[0][1];
		float ztmp = xp[0][2];

		xp[0][0] = xtmp;
		xp[0][1] = ytmp * cosf(data->pulseData.flipangle/180 * M_PI) + cosf(data->pulseData.phase) * ztmp * sinf(data->pulseData.flipangle/180 * M_PI);
		xp[0][2] = ytmp * - cosf(data->pulseData.phase) * sinf(data->pulseData.flipangle/180 * M_PI) + ztmp * cosf(data->pulseData.flipangle/180 * M_PI);
	}
	else 
		ode_direct_sa(h, tol, N, P, xp, data->pulseData.RF_start, data->pulseData.RF_end, _data,  bloch_simu_fun2, bloch_pdy3, bloch_pdp3);

}


void relaxation2(void* _data, float h, float tol, int N, int P, float xp[P + 2][N], float st, float end)
{
	struct SimData* data = _data;

	data->pulseData.pulse_applied = false;

	ode_direct_sa(h, tol, N, P, xp, st, end, _data, bloch_simu_fun2, bloch_pdy3, bloch_pdp3);
}


void create_sim_block(void* _data)
{
	struct SimData* data = _data;

	create_rf_pulse( &data->pulseData, data->pulseData.RF_start, data->pulseData.RF_end, data->pulseData.flipangle, data->pulseData.phase, data->pulseData.nl, data->pulseData.nr, data->pulseData.alpha);
}


void run_sim_block(void* _data, float* mxySignal, float* saR1Signal, float* saR2Signal, float* saM0Signal, float h, float tol, int N, int P, float xp[P + 2][N], bool get_signal)
{
	struct SimData* data = _data;

	start_rf_pulse(data, h, tol, N, P, xp);

	relaxation2(data, h, tol, N, P, xp, data->pulseData.RF_end, data->seqData.TE);

	if (get_signal)
		collect_signal( data, N, P, mxySignal, saR1Signal, saR2Signal, saM0Signal, xp);

	relaxation2(data, h, tol, N, P, xp, data->seqData.TE, data->seqData.TR);
}


void ode_bloch_simulation3( void* _data, float (*mxyOriSig)[3], float (*saT1OriSig)[3], float (*saT2OriSig)[3], float (*densOriSig)[3])
{
	struct SimData* data = _data;

	float tol = 10E-6; 

	int N = 3;
	int P = 2;
    
	float isochromats[data->seqData.spin_num];
	
	if (data->voxelData.spin_ensamble)
		isochromDistribution( data, isochromats);

	//Create bin for sum up the resulting signal and sa -> heap implementation should avoid stack overflows 
	float *mxySignal = malloc(data->seqData.spin_num * data->seqData.rep_num * 3 * sizeof(float));
	float *saT1Signal = malloc(data->seqData.spin_num * data->seqData.rep_num * 3 * sizeof(float));
	float *saT2Signal = malloc(data->seqData.spin_num * data->seqData.rep_num * 3 * sizeof(float));
	float *densSignal = malloc(data->seqData.spin_num * data->seqData.rep_num * 3 * sizeof(float));

	float flipangle_backup = data->pulseData.flipangle;
	
	float w_backup = data->voxelData.w;
	
	for (data->seqtmp.spin_counter = 0; data->seqtmp.spin_counter < data->seqData.spin_num; data->seqtmp.spin_counter++){
		
		
		
		if (NULL != data->seqData.slice_profile) 
			data->pulseData.flipangle = flipangle_backup * cabsf(data->seqData.slice_profile[data->seqtmp.spin_counter]);

		float xp[4][3] = { { 0., 0. , 1. }, { 0. }, { 0. }, { 0. } }; //xp[P + 2][N]
		
		float h = 0.0001;
		
		if (data->voxelData.spin_ensamble)
			data->voxelData.w = w_backup + isochromats[data->seqtmp.spin_counter];

		data->pulseData.phase = 0;
		data->seqtmp.t = 0;
		data->seqtmp.rep_counter = 0;
		
		/*--------------------------------------------------------------
		* ----------------  Inversion Pulse Block ----------------------
		* ------------------------------------------------------------*/
		
		if (data->seqData.seq_type == 1 || data->seqData.seq_type == 4 || data->seqData.seq_type == 5 || data->seqData.seq_type == 6) {
			
			struct SimData inv_data = *data;

			if(inv_data.pulseData.RF_end != 0) //Hard Pulses
				inv_data.pulseData.RF_end = 0.01;
			
			inv_data.pulseData.flipangle = 180.;
			inv_data.seqData.TE = data->pulseData.RF_end;
			inv_data.seqData.TR = data->pulseData.RF_end;
			
			create_sim_block(&inv_data);
			
			run_sim_block(&inv_data, NULL, NULL, NULL, NULL, h, tol, N, P, xp, false);
			
		}
		
		/*--------------------------------------------------------------
		* --------------------- Signal Preparation ---------------------
		* ------------------------------------------------------------*/
		//for bSSFP based sequences: alpha/2 and TR/2 preparation
		if (data->seqData.seq_type == 0 || data->seqData.seq_type == 1 || data->seqData.seq_type == 3 || data->seqData.seq_type == 6) { 

			struct SimData prep_data = *data;
			
			prep_data.pulseData.flipangle = data->pulseData.flipangle/2.;
			prep_data.pulseData.phase = M_PI;
			prep_data.seqData.TE = data->seqData.TR/2.;
			prep_data.seqData.TR = data->seqData.TR/2.;
			
			create_sim_block(&prep_data);
			
			run_sim_block(&prep_data, NULL, NULL, NULL, NULL, h, tol, N, P, xp, false);
		}
		
		/*--------------------------------------------------------------
		* --------------  Loop over Pulse Blocks  ----------------------
		* ------------------------------------------------------------*/
		data->seqtmp.t = 0;
		
		if (NULL == data->seqData.variable_fa)
			create_sim_block(data);
		
		while (data->seqtmp.rep_counter < data->seqData.rep_num) {
			
			
			if (NULL != data->seqData.variable_fa) {
				
				data->pulseData.flipangle = data->seqData.variable_fa[data->seqtmp.rep_counter];
				
				create_sim_block(data);
			}
			
			//Change phase for phase cycled bSSFP sequences //Check phase for FLASH!!
			if (data->seqData.seq_type == 3 || data->seqData.seq_type == 6)
				data->pulseData.phase = M_PI * (float) ( data->seqtmp.rep_counter % 2 ) + 360. * ( (float)data->seqtmp.rep_counter/(float)data->seqData.rep_num )/180. * M_PI;
			else if (data->seqData.seq_type == 0 || data->seqData.seq_type == 1 || data->seqData.seq_type == 4)
				data->pulseData.phase = M_PI * (float) data->seqtmp.rep_counter;

			run_sim_block(data, mxySignal, saT1Signal, saT2Signal, densSignal, h, tol, N, P, xp, true);
			
			//Spoiling of FLASH deletes x- and y-directions of sensitivities as well as magnetization
			if (data->seqData.seq_type == 2 || data->seqData.seq_type == 5)
				for (int i = 0; i < P + 2; i ++) {

					xp[i][0] = 0.;
					xp[i][1] = 0.; 
				}

			data->seqtmp.rep_counter++;
		}
	}
	
	/*---------------------------------------------------------------
	* ---------------  Sum up magnetization  -----------------------
	* ------------------------------------------------------------*/
	
	float sumMxyTmp;
	float sumSaT1;
	float sumSaT2; 
	
	for (int av_num = 0, dim = 0; dim < 3; dim++){
		
		sumMxyTmp = sumSaT1 = sumSaT2 = 0.;
		
		for (int save_repe = 0, repe = 0; repe < data->seqData.rep_num; repe++) {
			
			if (av_num == data->seqData.num_average_rep)
					av_num = 0.;
			
			for (int spin = 0; spin < data->seqData.spin_num; spin++) {
				
				sumMxyTmp += mxySignal[ (dim *data->seqData.spin_num * (data->seqData.rep_num) ) + (repe * data->seqData.spin_num) + spin ];
				sumSaT1 += saT1Signal[ (dim * data->seqData.spin_num * (data->seqData.rep_num) ) + (repe * data->seqData.spin_num) + spin ];
				sumSaT2 += saT2Signal[ (dim * data->seqData.spin_num * (data->seqData.rep_num) ) + (repe * data->seqData.spin_num) + spin ];
			}
			
			if (av_num == data->seqData.num_average_rep - 1) {

					mxyOriSig[save_repe][dim] = sumMxyTmp * data->voxelData.m0 / (float)( data->seqData.spin_num * data->seqData.num_average_rep );
					saT1OriSig[save_repe][dim] = sumSaT1 * data->voxelData.m0 / (float)( data->seqData.spin_num * data->seqData.num_average_rep );
					saT2OriSig[save_repe][dim] = sumSaT2 * data->voxelData.m0 / (float)( data->seqData.spin_num * data->seqData.num_average_rep );
					densOriSig[save_repe][dim] = sumMxyTmp / (float)( data->seqData.spin_num * data->seqData.num_average_rep );
					
					sumMxyTmp = sumSaT1 = sumSaT2 = 0.;
					save_repe++;
			}

			av_num++;
		}
	}
	
	free(mxySignal);
	free(saT1Signal);
	free(saT2Signal);
	free(densSignal);
}






