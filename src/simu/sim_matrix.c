#include <math.h>
#include <stdlib.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "simu/bloch.h"
#include "simu/simulation.h"
#include "simu/pulse.h"

#include "num/ode.h"

#include "sim_matrix.h"


struct ode_matrix_simu_s {

	unsigned int N;
	void* sim_data;
};


static void ode_matrix_fun_simu(void* _data, float* x, float t, const float* in)
{
	struct ode_matrix_simu_s* data = _data;
	struct SimData* sim_data = data->sim_data;

	unsigned int N = data->N;

	if (sim_data->pulseData.pulse_applied && t <= sim_data->pulseData.RF_end) { 
		
		float w1 = sinc_pulse( &sim_data->pulseData, t );
		sim_data->gradData.gb_eff[0] = cosf( sim_data->pulseData.phase ) * w1 + sim_data->gradData.gb[0];
		sim_data->gradData.gb_eff[1] = sinf( sim_data->pulseData.phase ) * w1 + sim_data->gradData.gb[1];
	}
	else {

		sim_data->gradData.gb_eff[0] = sim_data->gradData.gb[0];
		sim_data->gradData.gb_eff[1] = sim_data->gradData.gb[1];
	}

	sim_data->gradData.gb_eff[2] = sim_data->gradData.gb[2] + sim_data->voxelData.w;
	
	float matrix_time[N][N];
	bloch_matrix_ode_sa(matrix_time, sim_data->voxelData.r1, sim_data->voxelData.r2, sim_data->gradData.gb_eff);

	for (unsigned int i = 0; i < N; i++) {

		x[i] = 0.;

		for (unsigned int j = 0; j < N; j++)
			x[i] += matrix_time[i][j] * in[j];
	}
}

void ode_matrix_interval_simu(float h, float tol, unsigned int N, float x[N], float st, float end, void* sim_data)
{
	struct ode_matrix_simu_s data = { N, sim_data };
	ode_interval(h, tol, N, x, st, end, &data, ode_matrix_fun_simu);
}


void mat_exp_simu(int N, float t, float out[N][N], void* sim_data)
{
	// compute F(t) := exp(tA)
	// F(0) = id
	// d/dt F = A

	float h = t / 100.;
	float tol = 1.E-6;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++)
			out[i][j] = (i == j) ? 1. : 0.;

		ode_matrix_interval_simu( h, tol, N, out[i], 0., t, sim_data );
	}
}

#if 0
#define MATRIX_SPLIT
static void mm_mul(int N, float out[N][N], float in1[N][N], float in2[N][N])
{
	for (int i = 0; i < N; i++) 
		for (int j = 0; j < N; j++) {
			
			out[i][j] = 0.;
			
			for (int k = 0; k < N; k++)
				out[i][j] += in1[i][k] * in2[k][j];
		}
}
#endif

static void create_sim_matrix(int N, float matrix[N][N], float end, void* _data )
{
	struct SimData* simdata = _data;
	

	if (simdata->pulseData.pulse_applied)
#ifndef MATRIX_SPLIT
		create_rf_pulse(&simdata->pulseData, simdata->pulseData.RF_start, simdata->pulseData.RF_end, simdata->pulseData.flipangle, simdata->pulseData.phase, simdata->pulseData.nl, simdata->pulseData.nr, simdata->pulseData.alpha);
#else
	{	// Possible increase of precision by splitting into rf- and relaxation-matrix?

		create_rf_pulse(&simdata->pulseData, simdata->pulseData.RF_start, simdata->pulseData.RF_end, simdata->pulseData.flipangle, simdata->pulseData.phase, simdata->pulseData.nl, simdata->pulseData.nr, simdata->pulseData.alpha);

		float tmp1[N][N];

		mat_exp_simu(N, simdata->pulseData.RF_end, tmp1, simdata);

		float tmp2[N][N];

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				tmp2[i][j] = ( (i == j) ? 1 : 0 );
				

		if (simdata->pulseData.RF_end < end) {
			
			simdata->pulseData.pulse_applied = false;
			mat_exp_simu(N, end - simdata->pulseData.RF_end, tmp2, simdata);
		}

		mm_mul(N, matrix, tmp1, tmp2);
	}
	else 
#endif
		mat_exp_simu(N, end, matrix, simdata);
}


static void vm_mul_transpose(int N, float out[N], float matrix[N][N], float in[N])
{
	for (int i = 0; i < N; i++) {

		out[i] = 0.;

		for (int j = 0; j < N; j++)
			out[i] += matrix[j][i] * in[j];
	}
}

static void apply_sim_matrix(int N, float m[N], float matrix[N][N])
{
	float tmp[N];

	for (int i = 0; i < N; i++)
		tmp[i] = m[i];

	vm_mul_transpose(N, m, matrix, tmp);
}


static void apply_inversion(int N, float m[N], float pulse_length, void* _data )
{
	struct SimData* simdata = _data;
	struct SimData tmp_data = *simdata;

	tmp_data.pulseData.pulse_applied = true;
	tmp_data.pulseData.flipangle = 180.;
	tmp_data.pulseData.RF_end = pulse_length;

	float matrix[N][N];

	create_sim_matrix(N, matrix, tmp_data.pulseData.RF_end, &tmp_data);

	apply_sim_matrix(N, m, matrix);
}


static void apply_signal_preparation(int N, float m[N], void* _data )// provides alpha/2. preparation only
{
	struct SimData* simdata = _data;
	struct SimData tmp_data = *simdata;

	tmp_data.pulseData.pulse_applied = true;
	tmp_data.pulseData.flipangle = simdata->pulseData.flipangle/2.;
	tmp_data.pulseData.phase = M_PI;
	tmp_data.seqData.TR = simdata->seqData.TR/2.;

	float matrix[N][N];

	create_sim_matrix(N, matrix, tmp_data.seqData.TR, &tmp_data);

	apply_sim_matrix(N, m, matrix);
}


static void prepare_matrix_to_te( int N, float matrix[N][N], void* _data )
{
	struct SimData* simdata = _data;
	struct SimData tmp_data = *simdata;

	tmp_data.pulseData.pulse_applied = true;

	create_sim_matrix(N, matrix, tmp_data.seqData.TE, &tmp_data);
}


static void prepare_matrix_to_tr( int N, float matrix[N][N], void* _data )
{
	struct SimData* simdata = _data;
	struct SimData tmp_data = *simdata;

	tmp_data.pulseData.pulse_applied = false;

	create_sim_matrix(N, matrix, tmp_data.seqData.TE, &tmp_data);
}

static void ADCcorrection(int N, float out[N], float in[N])
{
	float corrAngle = M_PI; // for bSSFP

	for (int i = 0; i < 3; i++) {	// 3 parameter: Sig, S_R1, S_R2
		
		out[3*i] = in[3*i] * cosf(corrAngle) + in[3*i+1] * sinf(corrAngle);
		out[3*i+1] = in[3*i] * sinf(corrAngle) + in[3*i+1] * cosf(corrAngle);
		out[3*i+2] = in[3*i+2];
	}

}

static void collect_data(int N, float xp[N], float *mxySignal, float *saR1Signal, float *saR2Signal, void* _data)
{    
	struct SimData* data = _data;
	
	float tmp[N];
    
	ADCcorrection(N, tmp, xp);

	for (int i = 0; i < 3; i++) {

		mxySignal[ (i * data->seqData.spin_num * (data->seqData.rep_num) ) + ( (data->seqtmp.rep_counter) * data->seqData.spin_num) + data->seqtmp.spin_counter ] = (data->seqtmp.rep_counter % 2 == 0) ? tmp[i] : xp[i];
		saR1Signal[ (i * data->seqData.spin_num * (data->seqData.rep_num) ) + ( (data->seqtmp.rep_counter) * data->seqData.spin_num) + data->seqtmp.spin_counter ] = (data->seqtmp.rep_counter % 2 == 0) ? tmp[i+3] : xp[i+3];
		saR2Signal[ (i * data->seqData.spin_num * (data->seqData.rep_num) ) + ( (data->seqtmp.rep_counter) * data->seqData.spin_num) + data->seqtmp.spin_counter ] = (data->seqtmp.rep_counter % 2 == 0) ? tmp[i+6] : xp[i+6];
	}
    
}
    
// for seq = 0, 1, 2, 5
void matrix_bloch_simulation( void* _data, float (*mxyOriSig)[3], float (*saT1OriSig)[3], float (*saT2OriSig)[3], float (*densOriSig)[3], complex float* input_sp)
{
	struct SimData* data = _data;
	
	enum { N = 10 };
    
	float isochromats[data->seqData.spin_num];

	if (data->voxelData.spin_ensamble)
		isochromDistribution( data, isochromats);

	//Create bin for sum up the resulting signal and sa -> heap implementation should avoid stack overflows 
	float *mxySignal = malloc( (data->seqData.spin_num * (data->seqData.rep_num) * 3) * sizeof(float) );
	float *saT1Signal = malloc( (data->seqData.spin_num * (data->seqData.rep_num) * 3) * sizeof(float) );
	float *saT2Signal = malloc( (data->seqData.spin_num * (data->seqData.rep_num) * 3) * sizeof(float) );
    
	float flipangle_backup = data->pulseData.flipangle;
	float w_backup = data->voxelData.w;
	
	for (data->seqtmp.spin_counter = 0; data->seqtmp.spin_counter < data->seqData.spin_num; data->seqtmp.spin_counter++) {

		if (NULL != input_sp) 
			data->pulseData.flipangle = flipangle_backup * cabsf(input_sp[data->seqtmp.spin_counter]);

		float xp[N] = { 0., 0., 1., 0., 0., 0., 0., 0., 0., 1. };

		if (data->voxelData.spin_ensamble)
			data->voxelData.w = w_backup + isochromats[data->seqtmp.spin_counter]; //just on-resonant pulse for now.

		data->seqtmp.t = 0;
		data->seqtmp.rep_counter = 0;
		data->pulseData.phase = 0;

		if (data->seqData.seq_type == 1 || data->seqData.seq_type == 5)
			apply_inversion( N, xp, 0.01, data );

		//for bSSFP based sequences: alpha/2 and TR/2 preparation
		if (data->seqData.seq_type == 0 || data->seqData.seq_type == 1)
			apply_signal_preparation( N, xp, data );

		// Create matrices which describe signal development	
		float matrix_to_te[N][N];
		prepare_matrix_to_te( N, matrix_to_te, data);

		float matrix_to_te_PI[N][N];
		data->pulseData.phase = M_PI;
		prepare_matrix_to_te( N, matrix_to_te_PI, data);


		float matrix_to_tr[N][N];
		prepare_matrix_to_tr( N, matrix_to_tr, data);
        
		while (data->seqtmp.rep_counter < data->seqData.rep_num) { 
			
			if (data->seqData.seq_type == 2 || data->seqData.seq_type == 5)
				apply_sim_matrix( N, xp, matrix_to_te );
			else
				apply_sim_matrix( N, xp, ( ( data->seqtmp.rep_counter % 2 == 0 ) ? matrix_to_te : matrix_to_te_PI) );

			collect_data( N, xp, mxySignal, saT1Signal, saT2Signal, data);

			apply_sim_matrix( N, xp, matrix_to_tr );

			//Spoiling of FLASH deletes x- and y-directions of sensitivities as well as magnetization
			if (data->seqData.seq_type == 2 || data->seqData.seq_type == 5)
				for(int i = 0; i < 3 ; i ++) {

					xp[3*i] = 0.;
					xp[3*i+1] = 0.;
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

	for (int av_num = 0, dim = 0; dim < 3; dim++) {

		sumMxyTmp = sumSaT1 = sumSaT2 = 0.;

		for (int save_repe = 0, repe = 0; repe < data->seqData.rep_num; repe++) {

			if (av_num == data->seqData.num_average_rep)
				av_num = 0.;
			
			for (int spin = 0; spin < data->seqData.spin_num; spin++)  {
				
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
}

