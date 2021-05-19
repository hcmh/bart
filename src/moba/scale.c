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

#include "nlops/nlop.h"

#include "num/multind.h"
#include "num/loop.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/gpuops.h"

#include "iter/italgos.h"
#include "iter/vec.h"

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/phantom.h"

#include "recon_Bloch.h"
#include "model_Bloch.h"

#include "blochfun.h"

#include "scale.h"

struct op_test_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;

	const long* in_dims;

	complex float* tmp;
	complex float* tmp2;

	complex float* projection;
};

DEF_TYPEID(op_test_s);

// Apply normal operator
static void normal(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(op_test_s, _data);

	// Apply normal operator

	linop_normal_unchecked(nlop_get_derivative(data->nlop, 0, 0), data->tmp, (const complex float*)src);

	// Perform ortogonal projection onto desired parameter

	md_clear(DIMS, data->in_dims, data->tmp2, CFL_SIZE);
	md_zfmac(DIMS, data->in_dims, data->tmp2, data->projection, data->tmp);

	// Copy to output

	md_copy(DIMS, data->in_dims, (complex float*) dst, data->tmp2, CFL_SIZE);
}


void nlop_get_partial_ev(struct nlop_s* op, const long dims[DIMS], complex float* ev, complex float* maps)
{

	debug_printf(DP_INFO, "\n# Calculate Eigenvalues from Normal Operator\n");

	// Extract dimensions

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];

	md_select_dims(DIMS, FFT_FLAGS, map_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS|TE_FLAG, out_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS|COEFF_FLAG, in_dims, dims);

	// Allocate storage for...

	// ...randomly initialized parameter maps for power method
	complex float* para = md_alloc_sameplace(DIMS, in_dims, CFL_SIZE, maps);
	md_gaussian_rand(DIMS, in_dims, para);

	// ...forward operator output
	complex float* time_evolution = md_alloc_sameplace(DIMS, out_dims, CFL_SIZE, maps);

	// ...ones initialization for projection
	complex float* ones = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, maps);
	md_zfill(DIMS, map_dims, ones, 1.);

	struct op_test_s op_data = {{ &TYPEID(op_test_s) }, op, in_dims, NULL, NULL, NULL};

	// ...the projection itself
	op_data.projection = md_alloc_sameplace(DIMS, in_dims, CFL_SIZE, maps);
	md_zfill(DIMS, in_dims, op_data.projection, 0.);

	//...some temporary files
	op_data.tmp = md_alloc_sameplace(DIMS, in_dims, CFL_SIZE, maps);
	md_zfill(DIMS, in_dims, op_data.tmp, 0.);

	//...some temporary files
	op_data.tmp2 = md_alloc_sameplace(DIMS, in_dims, CFL_SIZE, maps);
	md_zfill(DIMS, in_dims, op_data.tmp2, 0.);


	// Run forward operator

	nlop_apply(op, DIMS, out_dims, time_evolution, DIMS, in_dims, maps);


	// Prepare looping through coefficient dimension

	long N = md_calc_size(DIMS, in_dims);

	void* x = md_alloc_sameplace(1, MD_DIMS(N), CFL_SIZE, maps);

	long pos[DIMS];
	md_set_dims(DIMS, pos, 0);

	double maxeigen = 0.;

	// Maximum eigenvalue for R1
	for (int i = 0; i < in_dims[COEFF_DIM]; i++) {

		pos[COEFF_DIM] = i;

		// Create desired orthogonal projection
		md_zfill(DIMS, in_dims, op_data.projection, 0.);
		md_copy_block(DIMS, pos, in_dims, op_data.projection, map_dims, ones, CFL_SIZE);

		// Copy randomly initialized parameter to float
		md_copy(DIMS, in_dims, x, para, CFL_SIZE);

		// Estimate eigenvalue
		maxeigen = power(20, 2*N, select_vecops(x),
						(struct iter_op_s){ normal, CAST_UP(&op_data) }, (float*)x);

		debug_printf(DP_DEBUG2, "## max. eigenvalue Component %d: = %f\n", i, maxeigen);

		ev[i] = maxeigen + 0*I;
	}

	md_free(x);
	md_free(para);
	md_free(ones);
	md_free(time_evolution);
	md_free(op_data.tmp);
	md_free(op_data.tmp2);
	md_free(op_data.projection);
}

void nlop_get_partial_scaling(struct nlop_s* op, const long dims[DIMS], complex float* scaling, complex float* maps, int ref)
{
	assert(ref < dims[COEFF_DIM]);

	complex float* ev = md_alloc(1, MD_DIMS(dims[COEFF_DIM]), CFL_SIZE);

	nlop_get_partial_ev(op, dims, ev, maps);

	for (int i = 0; i < dims[COEFF_DIM]; i++)
		scaling[i] = (ref == i) ? 1. : ev[ref] / ev[i];

	md_free(ev);
}

// takes FA map (iptr) in degree!
void fa_to_alpha(unsigned int D, const long dims[D], void* optr, const void* iptr, float tr)
{

#ifdef  USE_CUDA
	assert(cuda_ondevice(optr) == cuda_ondevice(iptr));
#endif

	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, iptr);

	md_zsmul(D, dims, tmp, iptr, M_PI/180.);
	md_zcos(D, dims, tmp, tmp);
	md_zlog(D, dims, tmp, tmp);

	md_zsmul(D, dims, optr, tmp, -1./tr);

	md_free(tmp);
}

// get TR from inversion time
float get_tr_from_inversion(unsigned int D, const long dims[D], complex float* iptr, int spokes)
{
	// Find non trivial dimension
	unsigned int nt_dim = 0;
	unsigned int num = 0;

	for (unsigned int i = 0; i < DIMS; i++) {

		if (1 < dims[i]) {
			nt_dim = i;
			num++;
		}
	}
	// Otherwise either to few or too many dimensions
	// Only supports 1 D arrays atm
	assert(num == 1);

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long pos[D];
	md_set_dims(D, pos, 0);

	// find minimum difference between timesteps
	float min_time = 100.;
	long ind = 0;
	long prev_ind = 0;

	float diff = 0;

	for (unsigned int i = 0; i < dims[nt_dim]; i++) {

		pos[nt_dim] = i;

		prev_ind = ind;
		ind = md_calc_offset(D, strs, pos) / CFL_SIZE;

		if (0 == i)
			continue;

		diff = cabsf(iptr[ind]-iptr[prev_ind]);

		min_time = (min_time > diff) ? diff : min_time;
	}

	debug_printf(DP_DEBUG2, "Min Timeinterval => TR = %f\n", min_time);

	return min_time / (float) spokes;// Estimate true TR (independent from spoke averaging)
}


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