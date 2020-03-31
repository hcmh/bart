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

#include "simulation.h"
#include "sim_matrix.h"



const struct simdata_voxel simdata_voxel_defaults = {

	.r1 = 0.,
	.r2 = 0.,
	.m0 = 1.,
	.w = 0.,
	.spin_ensamble = false,
};


const struct simdata_seq simdata_seq_defaults = {

	.analytical = 0, /*new substruct simData? Also for Simulation type (ODE,OBS)?*/
	.seq_type = 1,
	.tr = 0.004,
	.te = 0.002,
	.rep_num = 1,
	.spin_num = 1,
	.num_average_rep = 1,
	.run_num = 1,
	.inversion_pulse_length = 0.01,
	.prep_pulse_length = 0.001,
	.molli_break = 0,
	.molli_measure = 0,
	
	.slice_profile = NULL,
	.variable_fa = NULL,
};


const struct simdata_tmp simdata_tmp_defaults = {

	.t = 0.,
	.rep_counter = 0,
	.spin_counter = 0,
	.run_counter = 0,
};


const struct simdata_grad simdata_grad_defaults = {

	.gb = { 0., 0., 0. },	/*gradients, example: GAMMA_H1 * SKYRA_GRADIENT * 0.0001*/
	.gb_eff = { 0., 0., 0.},	/*storage for effective gradients*/
};





static void bloch_pdy2(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->r1, data->r2, data->gb_eff);
}

static void bloch_pdp2(void* _data, float* out, float t, const float* in)
{
	struct bloch_s* data = _data;
	(void)t;

	bloch_pdp((float(*)[3])out, in, data->r1, data->r2 , data->gb_eff);
}


static void bloch_pdy3(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->voxel.r1, data->voxel.r2, data->grad.gb_eff);
}

static void bloch_pdp3(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_pdp((float(*)[3])out, in, data->voxel.r1, data->voxel.r2, data->grad.gb_eff);
}


static void bloch_simu_fun2(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;

	if (data->pulse.pulse_applied) {

		float w1 = pulse_sinc(&data->pulse, t);

		data->grad.gb_eff[0] = cosf(data->pulse.phase) * w1 + data->grad.gb[0];
		data->grad.gb_eff[1] = sinf(data->pulse.phase) * w1 + data->grad.gb[1];

	} else {

		data->grad.gb_eff[0] = data->grad.gb[0];
		data->grad.gb_eff[1] = data->grad.gb[1];
	}

	data->grad.gb_eff[2] = data->grad.gb[2] + data->voxel.w;

	bloch_ode(out, in, data->voxel.r1, data->voxel.r2, data->grad.gb_eff);
}



void isochrom_distribution(struct sim_data* data, float* isochromats)
{
	float s = 1.;		//scaling parameters
	float t = 0.;		// location of max
	float maximum = 0.;

	float random_number;
	srand(4);
	
	float iso_tmp[data->seq.spin_num];

	//Creating Distribution...
	for (int i = 0; i < data->seq.spin_num; i++) {

		random_number = 0.02 + (float)rand() / ((float)RAND_MAX / (0.98 - 0.02) + 1.); //numbers needed to supress extrema for random_number=>1

		//...in this case a Cauchy one based on its inverse cdf.
		iso_tmp[i] = s * tan(M_PI * (random_number - 0.5)) + t;

		maximum = fmax(maximum, fabsf(iso_tmp[i]));
	}
	
	//Assigning frequencies up to pi/2
	for (int i = 0; i < data->seq.spin_num; i++) 
		isochromats[i] = (iso_tmp[i] / maximum) * M_PI / data->seq.tr;
}

//If ADC gets phase, it has to be corrected manually
void ADCcorr(int N, int P, float out[P + 2][N], float in[P + 2][N])
{
	float corr_angle = M_PI; // for bSSFP

	for (int i = 0; i < P + 2; i ++) {

		out[i][0] = in[i][0] * cosf(corr_angle) + in[i][1] * sinf(corr_angle);
		out[i][1] = in[i][0] * sinf(corr_angle) + in[i][1] * cosf(corr_angle);
		out[i][2] = in[i][2];
	}
}

// TODO: Fix correction for FLASH sequences, where phase is the same for all repetitions!
static void collect_signal(struct sim_data* data, int N, int P, float* mxy, float* sa_r1, float* sa_r2, float* sa_m0, float xp[P + 2][N])
{
	if ((2 == data->seq.seq_type) || (5 == data->seq.seq_type)) {

		for (int i = 0; i < N; i++) {

			mxy[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+ (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = xp[0][i];

			sa_r1[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+ (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = xp[1][i];

			sa_r2[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+ (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = xp[2][i];

			sa_m0[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+ (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = xp[3][i];
		}

	} else {

		float tmp[4][3] = { { 0. }, { 0. }, { 0. }, { 0. } };

		ADCcorr(N, P, tmp, xp);

		for (int i = 0; i < N; i++) {

			mxy[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+ (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = (data->tmp.rep_counter % 2 == 1) ? tmp[0][i] : xp[0][i];

			sa_r1[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+ (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = (data->tmp.rep_counter % 2 == 1) ? tmp[1][i] : xp[1][i];

			sa_r2[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+  (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = (data->tmp.rep_counter % 2 == 1) ? tmp[2][i] : xp[2][i];

			sa_m0[(data->tmp.run_counter * 3 * data->seq.spin_num * data->seq.rep_num)
				+ (i * data->seq.spin_num * data->seq.rep_num)
				+ (data->tmp.rep_counter * data->seq.spin_num)
				+ data->tmp.spin_counter] = (data->tmp.rep_counter % 2 == 1) ? tmp[3][i] : xp[3][i];
		}
	}
}




//Module for RF-pulses
void start_rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P + 2][N])
{
	data->pulse.pulse_applied = true;

	if (0. == data->pulse.rf_end) {

		// Hard-Pulse Approximation:
		// !! Does not work with sensitivity output !!

		float xtmp = xp[0][0];
		float ytmp = xp[0][1];
		float ztmp = xp[0][2];

		xp[0][0] = xtmp;
		xp[0][1] = ytmp * cosf(data->pulse.flipangle / 180 * M_PI) + cosf(data->pulse.phase) * ztmp * sinf(data->pulse.flipangle / 180 * M_PI);
		xp[0][2] = ytmp * -cosf(data->pulse.phase) * sinf(data->pulse.flipangle / 180 * M_PI) + ztmp * cosf(data->pulse.flipangle / 180 * M_PI);

	} else  {

		ode_direct_sa(h, tol, N, P, xp, data->pulse.rf_start, data->pulse.rf_end, data,  bloch_simu_fun2, bloch_pdy3, bloch_pdp3);
	}
}


void relaxation2(struct sim_data* data, float h, float tol, int N, int P, float xp[P + 2][N], float st, float end)
{
	data->pulse.pulse_applied = false;

	ode_direct_sa(h, tol, N, P, xp, st, end, data, bloch_simu_fun2, bloch_pdy3, bloch_pdp3);
}


void create_sim_block(struct sim_data* data)
{
	pulse_create(&data->pulse, data->pulse.rf_start, data->pulse.rf_end, data->pulse.flipangle, data->pulse.phase, data->pulse.nl, data->pulse.nr, data->pulse.alpha);
}


void run_sim_block(struct sim_data* data, float* mxy, float* sa_r1, float* sa_r2, float* saM0Signal, float h, float tol, int N, int P, float xp[P + 2][N], bool get_signal)
{
	start_rf_pulse(data, h, tol, N, P, xp);

	relaxation2(data, h, tol, N, P, xp, data->pulse.rf_end, data->seq.te);

	if (get_signal)
		collect_signal(data, N, P, mxy, sa_r1, sa_r2, saM0Signal, xp);

	relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr);
}

//Spoiling of FLASH deletes x- and y-directions of sensitivities as well as magnetization
static void xyspoiling(int N, int P, float xp[P + 2][N], struct sim_data* simdata)
{
	if ((simdata->seq.seq_type == 2) || (simdata->seq.seq_type == 5)) {

		for (int i = 0; i < P + 2; i ++) {

			xp[i][0] = 0.;
			xp[i][1] = 0.;
		}
	}
}


void ode_bloch_simulation3(struct sim_data* data, float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3])
{
	float tol = 10E-6;

	int N = 3;
	int P = 2;
 
	float isochromats[data->seq.spin_num];

	if (data->voxel.spin_ensamble)
		isochrom_distribution(data, isochromats);

	//Create bin for sum up the resulting signal and sa -> heap implementation should avoid stack overflows 
	float* mxy = malloc(data->seq.run_num * data->seq.spin_num * data->seq.rep_num * 3 * sizeof(float));
	float* sa_r1 = malloc(data->seq.run_num * data->seq.spin_num * data->seq.rep_num * 3 * sizeof(float));
	float* sa_r2 = malloc(data->seq.run_num * data->seq.spin_num * data->seq.rep_num * 3 * sizeof(float));
	float* sa_m0 = malloc(data->seq.run_num * data->seq.spin_num * data->seq.rep_num * 3 * sizeof(float));

	float flipangle_backup = data->pulse.flipangle;
	float w_backup = data->voxel.w;
	float slice_factor = 1.;

	for (data->tmp.spin_counter = 0; data->tmp.spin_counter < data->seq.spin_num; data->tmp.spin_counter++) {

		if (NULL != data->seq.slice_profile)
			slice_factor = cabsf(data->seq.slice_profile[data->tmp.spin_counter]);

		data->pulse.flipangle = flipangle_backup * slice_factor;

		float xp[4][3] = { { 0., 0. , 1. }, { 0. }, { 0. }, { 0. } }; //xp[P + 2][N]

		float h = 0.0001;

		data->voxel.w = w_backup;

		if (data->voxel.spin_ensamble)
			data->voxel.w += isochromats[data->tmp.spin_counter];

		data->pulse.phase = 0;

		for (data->tmp.run_counter = 0; data->tmp.run_counter < data->seq.run_num; data->tmp.run_counter++) {

			data->tmp.t = 0;
			data->tmp.rep_counter = 0;

			/*--------------------------------------------------------------
			* ----------------  Inversion Pulse Block ----------------------
			* ------------------------------------------------------------*/

			if (   (1 == data->seq.seq_type)
			    || (4 == data->seq.seq_type)
			    || (5 == data->seq.seq_type)
			    || (6 == data->seq.seq_type)) {

				struct sim_data inv_data = *data;

				if (0 != inv_data.pulse.rf_end) //Hard Pulses
					inv_data.pulse.rf_end = data->seq.inversion_pulse_length;

				inv_data.pulse.flipangle = 180.;
				inv_data.seq.te = data->pulse.rf_end;
				inv_data.seq.tr = data->pulse.rf_end;

				create_sim_block(&inv_data);

				run_sim_block(&inv_data, NULL, NULL, NULL, NULL, h, tol, N, P, xp, false);

				xyspoiling(N, P, xp, &inv_data);
			}

			/*--------------------------------------------------------------
			* --------------------- Signal Preparation ---------------------
			* ------------------------------------------------------------*/
			//for bSSFP based sequences: alpha/2 and tr/2 preparation

			if (   (0 == data->seq.seq_type)
			    || (1 == data->seq.seq_type)
			    || (3 == data->seq.seq_type)
			    || (   (4 == data->seq.seq_type)
			        && (0 == data->tmp.run_counter))
			    || (6 == data->seq.seq_type)) {

				struct sim_data prep_data = *data;

				if (4 == data->seq.seq_type)
					prep_data.pulse.flipangle = cabsf(data->seq.variable_fa[data->tmp.rep_counter]) / 2. * slice_factor;
				else
					prep_data.pulse.flipangle = data->pulse.flipangle / 2.;

				prep_data.pulse.phase = M_PI;
				prep_data.seq.te = data->seq.prep_pulse_length;
				prep_data.seq.tr = data->seq.prep_pulse_length;

				create_sim_block(&prep_data);

				run_sim_block(&prep_data, NULL, NULL, NULL, NULL, h, tol, N, P, xp, false);
			}

			/*--------------------------------------------------------------
			* --------------  Loop over Pulse Blocks  ----------------------
			* ------------------------------------------------------------*/
			data->tmp.t = 0;

			if (NULL == data->seq.variable_fa)
				create_sim_block(data);

			while (data->tmp.rep_counter < data->seq.rep_num) {


				if (NULL != data->seq.variable_fa) {

					data->pulse.flipangle = cabsf(data->seq.variable_fa[data->tmp.rep_counter]) * slice_factor;

					create_sim_block(data);
				}

				//Change phase for phase cycled bSSFP sequences
				if ((3 == data->seq.seq_type) || (6 == data->seq.seq_type))
					data->pulse.phase = M_PI * (float)(data->tmp.rep_counter % 2) + 360. * ((float)data->tmp.rep_counter / (float)data->seq.rep_num) / 180. * M_PI;
				else if ((0 == data->seq.seq_type) || (1 == data->seq.seq_type) || (4 == data->seq.seq_type))
					data->pulse.phase = M_PI * (float)(data->tmp.rep_counter + data->tmp.run_counter * data->seq.rep_num);

				run_sim_block(data, mxy, sa_r1, sa_r2, sa_m0, h, tol, N, P, xp, true);

				//Spoiling of FLASH deletes x- and y-directions of sensitivities as well as magnetization
				if ((2 == data->seq.seq_type) || (5 == data->seq.seq_type)) {

					for (int i = 0; i < P + 2; i++) {

						xp[i][0] = 0.;
						xp[i][1] = 0.;
					}
				}

				if (0 != data->tmp.rep_counter && 0 != data->seq.molli_break && (data->tmp.rep_counter%data->seq.molli_measure == 0))
					relaxation2(data, h, tol, N, P, xp, 0., data->seq.molli_break * data->seq.tr);

				data->tmp.rep_counter++;
			}
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

			for (int spin = 0; spin < data->seq.spin_num; spin++) {

				sum_mxy_tmp += mxy[((data->seq.run_num - 1) * 3 * data->seq.spin_num * data->seq.rep_num)
							+ (dim * data->seq.spin_num * data->seq.rep_num)
							+ (repe * data->seq.spin_num)
							+ spin];

				sum_sa_r1 += sa_r1[((data->seq.run_num - 1) * 3 * data->seq.spin_num * data->seq.rep_num)
							+ (dim * data->seq.spin_num * data->seq.rep_num)
							+ (repe * data->seq.spin_num)
							+ spin];

				sum_sa_r2 += sa_r2[((data->seq.run_num - 1) * 3 * data->seq.spin_num * data->seq.rep_num)
							+ (dim * data->seq.spin_num * data->seq.rep_num)
							+ (repe * data->seq.spin_num)
							+ spin];
			}

			if (av_num == (data->seq.num_average_rep - 1)) {

					mxy_sig[save_repe][dim] = sum_mxy_tmp * data->voxel.m0 / (float)(data->seq.spin_num * data->seq.num_average_rep);
					sa_r1_sig[save_repe][dim] = sum_sa_r1 * data->voxel.m0 / (float)(data->seq.spin_num * data->seq.num_average_rep);
					sa_r2_sig[save_repe][dim] = sum_sa_r2 * data->voxel.m0 / (float)(data->seq.spin_num * data->seq.num_average_rep);
					sa_m0_sig[save_repe][dim] = sum_mxy_tmp / (float)(data->seq.spin_num * data->seq.num_average_rep);

					sum_mxy_tmp = sum_sa_r1 = sum_sa_r2 = 0.;
					save_repe++;
			}

			av_num++;
		}
	}

	free(mxy);
	free(sa_r1);
	free(sa_r2);
	free(sa_m0);
}


void bloch_simulation(struct sim_data* sim_data, int N, complex float* out, bool ode)
{
	float mxy_sig[sim_data->seq.rep_num / sim_data->seq.num_average_rep][3];
	float sa_r1_sig[sim_data->seq.rep_num / sim_data->seq.num_average_rep][3];
	float sa_r2_sig[sim_data->seq.rep_num / sim_data->seq.num_average_rep][3];
	float sa_m0_sig[sim_data->seq.rep_num / sim_data->seq.num_average_rep][3];

	if (ode)
		ode_bloch_simulation3(sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);		// ODE simulation
	else
		matrix_bloch_simulation(sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);	// OBS simulation, does not work with hard-pulses!

	for (int t = 0; t < N; t++) 
		out[t] = mxy_sig[t][1] + mxy_sig[t][0] * I;
}


