/* All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 *
 */

#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/loop.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/phantom.h"

#include "recon_Bloch.h"
#include "model_Bloch.h"

#include "scale.h"

// Automatically estimate partial derivative scaling
// Idea:
//	1) Simulate array of homogeneously distributed T1 and T2 values
//	2) Determine mean values of sensitivity output
//	3) Scale partial derivatives to mean M0 sens output

void auto_scale(const struct modBlochFit* fit_para, float scale[4], const long ksp_dims[DIMS], complex float* kspace_data)
{
	long int dims[DIMS] = { [0 ... DIMS - 1] = 1. };
	dims[READ_DIM] = 100;
	dims[PHS1_DIM] = 100;
	dims[TE_DIM] = ksp_dims[TE_DIM];

	complex float* sens_r1 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sens_r2 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sens_m0 = md_alloc(DIMS, dims, CFL_SIZE);

	complex float* phantom = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* phantom_sp = md_alloc(DIMS, dims, CFL_SIZE);

	float lim_T1[2] = {0.3, 4.};
	float lim_T2[2] = {0.01, 1.5};

	if (4 == fit_para->sequence)
		assert(fit_para->num_vfa >= dims[TE_DIM] * fit_para->averaged_spokes);
	
	#pragma omp parallel for collapse(2)
	for (int x = 0; x < dims[0]; x++) {
		
		for (int y = 0; y < dims[1]; y++) {

			struct sim_data sim_data;

			sim_data.seq = simdata_seq_defaults;
			sim_data.seq.seq_type = fit_para->sequence;
			sim_data.seq.tr = fit_para->tr;
			sim_data.seq.te = fit_para->te;

			sim_data.seq.rep_num = dims[TE_DIM] * fit_para->averaged_spokes;
			
			sim_data.seq.spin_num = 1;
			sim_data.seq.num_average_rep = fit_para->averaged_spokes;
			sim_data.seq.run_num = fit_para->runs;
			
			sim_data.voxel = simdata_voxel_defaults;

			sim_data.voxel.r1 = 1 / (lim_T1[0] + x * lim_T1[1]/(dims[0]-1));
			sim_data.voxel.r2 = 1 / (lim_T2[0] + x * lim_T2[1]/(dims[1]-1));

			sim_data.voxel.m0 = 1;
			sim_data.voxel.w = 0;
			
			sim_data.pulse = simdata_pulse_defaults;
			sim_data.pulse.flipangle = fit_para->fa;
			sim_data.pulse.rf_end = fit_para->rfduration;
			sim_data.grad = simdata_grad_defaults;
			sim_data.tmp = simdata_tmp_defaults;
			
			if (NULL != fit_para->input_fa_profile) {
				
				long vfa_dims[DIMS];
				md_set_dims(DIMS, vfa_dims, 1);
				vfa_dims[READ_DIM] = fit_para->num_vfa;

				sim_data.seq.variable_fa = md_alloc(DIMS, vfa_dims, CFL_SIZE);
				md_copy(DIMS, vfa_dims, sim_data.seq.variable_fa, fit_para->input_fa_profile, CFL_SIZE);
			}
			
// 			debug_printf(DP_DEBUG3,"%f,\t%f,\t%f,\t%f,\t%d,\t%f,\t%f,\n", data.seq.tr, data.seq.te, data.voxel.r1, data.voxel.r2, 	data.tmp.rep_counter, data.pulse.rf_end, data.pulse.flipangle );

			// Estimate reference signal without slice profile

			complex float mxy_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			complex float sa_r1_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			complex float sa_r2_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			complex float sa_m0_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

			if (fit_para->full_ode_sim || NULL != fit_para->input_fa_profile)	//variable flipangles are only included into ode simulation yet
				ode_bloch_simulation3(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);
			else
				matrix_bloch_simulation(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);

			// Estimate signal with slice profile

			sim_data.seq.spin_num = fit_para->sliceprofile_spins;

			complex float mxy_sig_sp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			complex float sa_r1_sig_sp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			complex float sa_r2_sig_sp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			complex float sa_m0_sig_sp[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

			long slcprfl_dims[DIMS];

			if (NULL != fit_para->input_sliceprofile) {

				md_set_dims(DIMS, slcprfl_dims, 1);
				slcprfl_dims[READ_DIM] = fit_para->sliceprofile_spins;

				sim_data.seq.slice_profile = md_alloc(DIMS, slcprfl_dims, CFL_SIZE);
				md_copy(DIMS, slcprfl_dims, sim_data.seq.slice_profile, fit_para->input_sliceprofile, CFL_SIZE);	

				if (fit_para->full_ode_sim || NULL != fit_para->input_fa_profile)	//variable flipangles are only included into ode simulation yet
					ode_bloch_simulation3(&sim_data, mxy_sig_sp, sa_r1_sig_sp, sa_r2_sig_sp, sa_m0_sig_sp);
				else
					matrix_bloch_simulation(&sim_data, mxy_sig_sp, sa_r1_sig_sp, sa_r2_sig_sp, sa_m0_sig_sp);
			}

			//Add data storages

			//Add data to phantom
			for (int z = 0; z < dims[TE_DIM]; z++) {

				//changed x-and y-axis to have same orientation as measurements
				phantom[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = mxy_sig[z][1] + mxy_sig[z][0] * I; 
				sens_r1[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = sa_r1_sig[z][1] + sa_r1_sig[z][0] * I; 
				sens_r2[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = sa_r2_sig[z][1] + sa_r2_sig[z][0] * I;
				sens_m0[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = sa_m0_sig[z][1] + sa_m0_sig[z][0] * I;

				if (NULL != fit_para->input_sliceprofile)
					phantom_sp[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = mxy_sig_sp[z][1] + mxy_sig_sp[z][0] * I;
			}
		}
	}

	double mean_sig = md_znorm(DIMS, ksp_dims, kspace_data);

	double mean_m0 = md_znorm(DIMS, dims, sens_m0);
	scale[1] = 1.;

	double mean_r1 = md_znorm(DIMS, dims, sens_r1);
	scale[0] = mean_m0 / mean_r1;

	double mean_r2 = md_znorm(DIMS, dims, sens_r2);
	scale[2] = mean_m0 / mean_r2;

	if (2 == fit_para->sequence || 5 == fit_para->sequence)
		scale[2] = 0.0001;

	debug_printf(DP_DEBUG1,"means:\tData:%f,\tdR1:%f,\tdM0:%f,\tdR2:%f\n", mean_sig, mean_r1, mean_m0, mean_r2);

	double mean_ref = md_znorm(DIMS, dims, phantom);

	double mean_sp = 0.;

	if (NULL != fit_para->input_sliceprofile) {

		mean_sp = md_znorm(DIMS, dims, phantom_sp);

		scale[3] = mean_ref / mean_sp;
	}

	debug_printf(DP_DEBUG1,"Mean Ref:%f,\tMean:%f,\t Signal Scaling:%f\n", mean_ref, mean_sp, scale[3]);

	md_free(sens_r1);
	md_free(sens_r2);
	md_free(sens_m0);
	md_free(phantom);
	md_free(phantom_sp);
}