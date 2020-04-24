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

void auto_scale(const struct modBlochFit* fit_para, float scale[3], const long ksp_dims[DIMS], complex float* kspace_data)
{

	long int dims[DIMS] = { [0 ... DIMS - 1] = 1. };
	dims[READ_DIM] = 100;
	dims[PHS1_DIM] = 100;
	dims[TE_DIM] = ksp_dims[TE_DIM];
	
	complex float* phantom = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sens_r1 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sens_r2 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sens_m0 = md_alloc(DIMS, dims, CFL_SIZE);


#if 1	//Simple linear scaling determination
#define LINEAR_MODBLOCH_SCALING

	float lim_T1[2] = {0.3, 4.};
	float lim_T2[2] = {0.01, 1.5};

#else	// Create realistic phantom guess of relaxation parameter

	// Reference values

	float t1[11] = {3., 0.5, 1., 1.5, 0.5, 1., 1.5, 0.5, 1., 1.5, 3.};
	float t2[11] = {1., 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 1.};

	// Determine spatial basis functions

	long int map_dims[DIMS] =  { [0 ... DIMS - 1] = 1. };
	map_dims[READ_DIM] = ksp_dims[READ_DIM];
	map_dims[PHS1_DIM] = ksp_dims[PHS1_DIM];
	map_dims[COEFF_DIM] = 11; // Tubes phantom provides 11 basis functions

	long sstrs[DIMS] = { 0 };

	complex float* full_basis_functions = anon_cfl("", DIMS, map_dims);

	calc_phantom_tubes(map_dims, full_basis_functions, false, sstrs, NULL);

	// Extract and scale desired ellipses

	long int extract_dims[DIMS] =  { [0 ... DIMS - 1] = 1. };
	md_select_dims(DIMS, ~COEFF_FLAG, extract_dims, map_dims);
	extract_dims[COEFF_DIM] = 10;	// Extract 10 of the 11 supported basis functions of the phantom tool

	long int slice_dims[DIMS] =  { [0 ... DIMS - 1] = 1. };
	md_select_dims(DIMS, ~COEFF_FLAG, slice_dims, map_dims);

	long pos[DIMS] = { 0 };
	long save_pos[DIMS] = { 0 };

	complex float* tmp_slice = md_alloc(DIMS, slice_dims, CFL_SIZE);

	complex float* extracted_basis_functions = anon_cfl("", DIMS, extract_dims);

	for (int basis_function = 1; basis_function < map_dims[COEFF_DIM]; basis_function++) {	//Choose start of extraction

		pos[COEFF_DIM] = basis_function;

		md_copy_block(DIMS, pos, slice_dims, tmp_slice, map_dims, full_basis_functions, CFL_SIZE);

		md_zsmul(DIMS, slice_dims, tmp_slice, tmp_slice, t1[basis_function]+t2[basis_function]*I);	// Add Relaxation Parameters to extracted basis functions

		md_copy_block(DIMS, save_pos, extract_dims, extracted_basis_functions, slice_dims, tmp_slice, CFL_SIZE);

		save_pos[COEFF_DIM] += 1;
	}

	md_free(tmp_slice);

	unmap_cfl(DIMS, map_dims, full_basis_functions);
	
	// Squash COEFF_DIM to create single relaxation parameter map

	long extract_strs[DIMS] = { 0 };
	md_calc_strides(DIMS, extract_strs, extract_dims, sizeof(complex float));

	long slice_strs[DIMS] = { 0 };
	md_calc_strides(DIMS, slice_strs, slice_dims, sizeof(complex float));

	complex float* map = anon_cfl("", DIMS, slice_dims);
	md_clear(DIMS, slice_dims, map, CFL_SIZE);

	complex float* one = md_alloc(DIMS, extract_dims, CFL_SIZE);
	md_zfill(DIMS, extract_dims, one, 1.);

	md_zfmac2(DIMS, extract_dims, slice_strs, map, extract_strs, extracted_basis_functions, extract_strs, one);

	md_free(one);

	unmap_cfl(DIMS, extract_dims, extracted_basis_functions);
#endif

	// Do simulation for scaling factor determination
	
	#pragma omp parallel for collapse(2)
	for(int x = 0; x < dims[0]; x++ ){
		
		for(int y = 0; y < dims[1]; y++ ){

			struct sim_data sim_data;

			sim_data.seq = simdata_seq_defaults;
			sim_data.seq.seq_type = fit_para->sequence;
			sim_data.seq.tr = fit_para->tr;
			sim_data.seq.te = fit_para->te;
			
			if (4 == sim_data.seq.seq_type)
				sim_data.seq.rep_num = fit_para->num_vfa;
			else
				sim_data.seq.rep_num = dims[TE_DIM];
			
			sim_data.seq.spin_num = fit_para->sliceprofile_spins;
			sim_data.seq.num_average_rep = fit_para->averaged_spokes;
			sim_data.seq.run_num = fit_para->runs;
			
			sim_data.voxel = simdata_voxel_defaults;

#ifdef LINEAR_MODBLOCH_SCALING

			sim_data.voxel.r1 = 1 / (lim_T1[0] + x * lim_T1[1]/(dims[0]-1));
			sim_data.voxel.r2 = 1 / (lim_T2[0] + x * lim_T2[1]/(dims[1]-1));
#else
			long spatial_pos[DIMS];
			md_copy_dims(DIMS, spatial_pos, slice_dims);

			spatial_pos[0] = x;
			spatial_pos[1] = y;

			long spatial_ind = md_calc_offset(DIMS, slice_strs, spatial_pos) / CFL_SIZE;

			if (0 == crealf(map[spatial_ind]) || 0 == cimagf(map[spatial_ind]))
				continue;

			sim_data.voxel.r1 = 1 / crealf(map[spatial_ind]);
			sim_data.voxel.r2 = 1 / cimagf(map[spatial_ind]);
#endif

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
			
			
			float mxy_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			float sa_r1_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			float sa_r2_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			float sa_m0_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
			
			
			if (fit_para->full_ode_sim || NULL != fit_para->input_fa_profile)	//variable flipangles are only included into ode simulation yet
				ode_bloch_simulation3(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);
			else
				matrix_bloch_simulation(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);

			
			//Add data to phantom
			for (int z = 0; z < dims[TE_DIM]; z++) {
				
				//changed x-and y-axis to have same orientation as measurements
				sens_r1[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = sa_r1_sig[z][1] + sa_r1_sig[z][0] * I; 
				sens_r2[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = sa_r2_sig[z][1] + sa_r2_sig[z][0] * I;
				sens_m0[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = sa_m0_sig[z][1] + sa_m0_sig[z][0] * I;
				phantom[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = mxy_sig[z][1] + mxy_sig[z][0] * I;
			}
		}
	}
	
	double mean_sig = md_znorm(DIMS, ksp_dims, kspace_data);
	
	double mean_r1 = md_znorm(DIMS, dims, sens_r1);
	scale[0] = mean_sig / mean_r1;
	
	double mean_r2 = md_znorm(DIMS, dims, sens_r2);
	
	if (2 != fit_para->sequence && 5 != fit_para->sequence)
		scale[1] = mean_sig / mean_r2;
	
	double mean_m0 = md_znorm(DIMS, dims, sens_m0);
	scale[2] = mean_sig / mean_m0;
	
	
	debug_printf(DP_DEBUG1,"means:\t%f,\t%f,\t%f,\t%f\n", mean_sig, mean_r1, mean_r2, mean_m0);

#ifndef LINEAR_MODBLOCH_SCALING	
	unmap_cfl(DIMS, slice_dims, map);
#endif
}
