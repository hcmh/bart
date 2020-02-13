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
	struct sim_data* sim_data = data->sim_data;

	unsigned int N = data->N;

	if (sim_data->pulse.pulse_applied && t <= sim_data->pulse.rf_end) { 
		
		float w1 = pulse_sinc( &sim_data->pulse, t );
		sim_data->grad.gb_eff[0] = cosf( sim_data->pulse.phase ) * w1 + sim_data->grad.gb[0];
		sim_data->grad.gb_eff[1] = sinf( sim_data->pulse.phase ) * w1 + sim_data->grad.gb[1];
	}
	else {

		sim_data->grad.gb_eff[0] = sim_data->grad.gb[0];
		sim_data->grad.gb_eff[1] = sim_data->grad.gb[1];
	}

	sim_data->grad.gb_eff[2] = sim_data->grad.gb[2] + sim_data->voxel.w;
	
	float matrix_time[N][N];
	bloch_matrix_ode_sa(matrix_time, sim_data->voxel.r1, sim_data->voxel.r2, sim_data->grad.gb_eff);

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
	struct sim_data* simdata = _data;
	

	if (simdata->pulse.pulse_applied)
#ifndef MATRIX_SPLIT
		pulse_create(&simdata->pulse, simdata->pulse.rf_start, simdata->pulse.rf_end, simdata->pulse.flipangle, simdata->pulse.phase, simdata->pulse.nl, simdata->pulse.nr, simdata->pulse.alpha);
#else
	{	// Possible increase of precision by splitting into rf- and relaxation-matrix?

		pulse_create(&simdata->pulse, simdata->pulse.rf_start, simdata->pulse.rf_end, simdata->pulse.flipangle, simdata->pulse.phase, simdata->pulse.nl, simdata->pulse.nr, simdata->pulse.alpha);

		float tmp1[N][N];

		mat_exp_simu(N, simdata->pulse.rf_end, tmp1, simdata);

		float tmp2[N][N];

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				tmp2[i][j] = ( (i == j) ? 1 : 0 );
				

		if (simdata->pulse.rf_end < end) {
			
			simdata->pulse.pulse_applied = false;
			mat_exp_simu(N, end - simdata->pulse.rf_end, tmp2, simdata);
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

//Spoiling of FLASH deletes x- and y-directions of sensitivities as well as magnetization
static void xyspoiling(int N, float out[N], void* _data)
{
	struct sim_data* simdata = _data;
	
	if (simdata->seq.seq_type == 2 || simdata->seq.seq_type == 5)
		for(int i = 0; i < 3 ; i ++) {

			out[3*i] = 0.;
			out[3*i+1] = 0.;
		}
}

static void apply_inversion(int N, float m[N], float pulse_length, void* _data )
{
	struct sim_data* simdata = _data;
	struct sim_data tmp_data = *simdata;

	tmp_data.pulse.pulse_applied = true;
	tmp_data.pulse.flipangle = 180.;
	tmp_data.pulse.rf_end = pulse_length;

	float matrix[N][N];

	create_sim_matrix(N, matrix, tmp_data.pulse.rf_end, &tmp_data);

	apply_sim_matrix(N, m, matrix);
	
	xyspoiling(N, m, &tmp_data);
}


static void spoiler_relaxation(int N, float m[N], float spoiler_length, void* _data )
{
	struct sim_data* simdata = _data;
	struct sim_data tmp_data = *simdata;

	tmp_data.pulse.pulse_applied = false;

	float matrix[N][N];
	create_sim_matrix(N, matrix, spoiler_length, &tmp_data);

	apply_sim_matrix(N, m, matrix);
	
	xyspoiling(N, m, &tmp_data);
}


static void apply_signal_preparation(int N, float m[N], void* _data )// provides alpha/2. preparation only
{
	struct sim_data* simdata = _data;
	struct sim_data tmp_data = *simdata;

	tmp_data.pulse.pulse_applied = true;
	tmp_data.pulse.flipangle = simdata->pulse.flipangle/2.;
	tmp_data.pulse.phase = M_PI;
	tmp_data.seq.tr = simdata->seq.tr/2.;

	float matrix[N][N];

	create_sim_matrix(N, matrix, tmp_data.seq.tr, &tmp_data);

	apply_sim_matrix(N, m, matrix);
}


static void prepare_matrix_to_te( int N, float matrix[N][N], void* _data )
{
	struct sim_data* simdata = _data;
	struct sim_data tmp_data = *simdata;

	tmp_data.pulse.pulse_applied = true;

	create_sim_matrix(N, matrix, tmp_data.seq.te, &tmp_data);
}


static void prepare_matrix_to_tr( int N, float matrix[N][N], void* _data )
{
	struct sim_data* simdata = _data;
	struct sim_data tmp_data = *simdata;

	tmp_data.pulse.pulse_applied = false;

	create_sim_matrix(N, matrix, (tmp_data.seq.tr-tmp_data.seq.te), &tmp_data);
}

static void ADCcorrection(int N, float out[N], float in[N])
{
	float corr_angle = M_PI; // for bSSFP

	for (int i = 0; i < 3; i++) {	// 3 parameter: Sig, S_R1, S_R2
		
		out[3*i] = in[3*i] * cosf(corr_angle) + in[3*i+1] * sinf(corr_angle);
		out[3*i+1] = in[3*i] * sinf(corr_angle) + in[3*i+1] * cosf(corr_angle);
		out[3*i+2] = in[3*i+2];
	}

}

static void collect_data(int N, float xp[N], float *mxy, float *sa_r1, float *sa_r2, void* _data)
{    
	struct sim_data* data = _data;
	
	if (2 == data->seq.seq_type || 5 == data->seq.seq_type ) {
	
		for (int i = 0; i < 3; i++) {

			mxy[ (i * data->seq.spin_num * (data->seq.rep_num) ) + ( (data->tmp.rep_counter) * data->seq.spin_num) + data->tmp.spin_counter ] = xp[i];
			sa_r1[ (i * data->seq.spin_num * (data->seq.rep_num) ) + ( (data->tmp.rep_counter) * data->seq.spin_num) + data->tmp.spin_counter ] = xp[i+3];
			sa_r2[ (i * data->seq.spin_num * (data->seq.rep_num) ) + ( (data->tmp.rep_counter) * data->seq.spin_num) + data->tmp.spin_counter ] = xp[i+6];
		}	
	}
	else {
		float tmp[N];
	
		ADCcorrection(N, tmp, xp);

		for (int i = 0; i < 3; i++) {

			mxy[ (i * data->seq.spin_num * (data->seq.rep_num) ) + ( (data->tmp.rep_counter) * data->seq.spin_num) + data->tmp.spin_counter ] = (data->tmp.rep_counter % 2 == 1) ? tmp[i] : xp[i];
			sa_r1[ (i * data->seq.spin_num * (data->seq.rep_num) ) + ( (data->tmp.rep_counter) * data->seq.spin_num) + data->tmp.spin_counter ] = (data->tmp.rep_counter % 2 == 1) ? tmp[i+3] : xp[i+3];
			sa_r2[ (i * data->seq.spin_num * (data->seq.rep_num) ) + ( (data->tmp.rep_counter) * data->seq.spin_num) + data->tmp.spin_counter ] = (data->tmp.rep_counter % 2 == 1) ? tmp[i+6] : xp[i+6];
		}
	}
    
}
    
// for seq = 0, 1, 2, 5
void matrix_bloch_simulation( void* _data, float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3])
{
	struct sim_data* data = _data;
	
	enum { N = 10 };
    
	float isochromats[data->seq.spin_num];

	if (data->voxel.spin_ensamble)
		isochrom_distribution( data, isochromats);

	//Create bin for sum up the resulting signal and sa -> heap implementation should avoid stack overflows 
	float *mxy = malloc( (data->seq.spin_num * (data->seq.rep_num) * 3) * sizeof(float) );
	float *sa_r1 = malloc( (data->seq.spin_num * (data->seq.rep_num) * 3) * sizeof(float) );
	float *sa_r2 = malloc( (data->seq.spin_num * (data->seq.rep_num) * 3) * sizeof(float) );
    
	float flipangle_backup = data->pulse.flipangle;
	float w_backup = data->voxel.w;
	
	float slice_correction = 1.;
	
	for (data->tmp.spin_counter = 0; data->tmp.spin_counter < data->seq.spin_num; data->tmp.spin_counter++) {
		
		slice_correction = 1.;

		if (NULL != data->seq.slice_profile) 
			slice_correction = cabsf(data->seq.slice_profile[data->tmp.spin_counter]);
		
		data->pulse.flipangle = flipangle_backup * slice_correction;

		float xp[N] = { 0., 0., 1., 0., 0., 0., 0., 0., 0., 1. };

		if (data->voxel.spin_ensamble)
			data->voxel.w = w_backup + isochromats[data->tmp.spin_counter]; //just on-resonant pulse for now.

		data->tmp.t = 0;
		data->tmp.rep_counter = 0;
		data->pulse.phase = 0;

		if (data->seq.seq_type == 1 || data->seq.seq_type == 5)
			apply_inversion(N, xp, 0.01, data);

		//for bSSFP based sequences: alpha/2 and tr/2 preparation
		if (data->seq.seq_type == 0 || data->seq.seq_type == 1)
			apply_signal_preparation(N, xp, data);

		// Create matrices which describe signal development	
		float matrix_to_te[N][N];
		prepare_matrix_to_te( N, matrix_to_te, data);

		float matrix_to_te_PI[N][N];
		data->pulse.phase = M_PI;
		prepare_matrix_to_te( N, matrix_to_te_PI, data);


		float matrix_to_tr[N][N];
		prepare_matrix_to_tr( N, matrix_to_tr, data);
        
		while (data->tmp.rep_counter < data->seq.rep_num) { 
			
			if (data->seq.seq_type == 2 || data->seq.seq_type == 5)
				apply_sim_matrix( N, xp, matrix_to_te );
			else
				apply_sim_matrix( N, xp, ( ( data->tmp.rep_counter % 2 == 0 ) ? matrix_to_te : matrix_to_te_PI) );

			collect_data( N, xp, mxy, sa_r1, sa_r2, data);

			apply_sim_matrix( N, xp, matrix_to_tr );

			xyspoiling(N, xp, data);

			data->tmp.rep_counter++;
		}
	}
	
	/*---------------------------------------------------------------
	* ---------------  Sum up magnetization  -----------------------
	* ------------------------------------------------------------*/
   
	float sum_mxy_tmp;
	float sum_sa_r1;
	float sum_sa_r2; 

	for (int av_num = 0, dim = 0; dim < 3; dim++) {

		sum_mxy_tmp = sum_sa_r1 = sum_sa_r2 = 0.;

		for (int save_repe = 0, repe = 0; repe < data->seq.rep_num; repe++) {

			if (av_num == data->seq.num_average_rep)
				av_num = 0.;
			
			for (int spin = 0; spin < data->seq.spin_num; spin++)  {
				
				sum_mxy_tmp += mxy[ (dim *data->seq.spin_num * (data->seq.rep_num) ) + (repe * data->seq.spin_num) + spin ];
				sum_sa_r1 += sa_r1[ (dim * data->seq.spin_num * (data->seq.rep_num) ) + (repe * data->seq.spin_num) + spin ];
				sum_sa_r2 += sa_r2[ (dim * data->seq.spin_num * (data->seq.rep_num) ) + (repe * data->seq.spin_num) + spin ];
			}
			
			if (av_num == data->seq.num_average_rep - 1) {

				mxy_sig[save_repe][dim] = sum_mxy_tmp * data->voxel.m0 / (float)( data->seq.spin_num * data->seq.num_average_rep );
				sa_r1_sig[save_repe][dim] = sum_sa_r1 * data->voxel.m0 / (float)( data->seq.spin_num * data->seq.num_average_rep );
				sa_r2_sig[save_repe][dim] = sum_sa_r2 * data->voxel.m0 / (float)( data->seq.spin_num * data->seq.num_average_rep );
				sa_m0_sig[save_repe][dim] = sum_mxy_tmp / (float)( data->seq.spin_num * data->seq.num_average_rep );

				sum_mxy_tmp = sum_sa_r1 = sum_sa_r2 = 0.;
				save_repe++;
			}

			av_num++;
		}
	}

	free(mxy);
	free(sa_r1);
	free(sa_r2);
}

