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

#include "num/multind.h"
#include "num/loop.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "simu/shepplogan.h"
#include "simu/sens.h"
#include "simu/coil.h"
#include "simu/simulation.h"

#include "recon_Bloch.h"
#include "model_Bloch.h"

#include "scale.h"

void auto_scale(const struct modBlochFit* fitPara, float scale[3], const long ksp_dims[DIMS], complex float* kspace_data)
{
	
	long int dims[DIMS] = { [0 ... DIMS - 1] = 1. };
	dims[0] = 32;
	dims[1] = 32;
	dims[TE_DIM] = ksp_dims[TE_DIM];
	
	complex float* phantom = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sensitivitiesR1 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sensitivitiesR2 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sensitivitiesM0 = md_alloc(DIMS, dims, CFL_SIZE);
	
	float lim_T1[2] = {0.3, 3.};
	float lim_T2[2] = {0.01, 1.};
	
	#pragma omp parallel for collapse(2)
	for(int x = 0; x < dims[0]; x++ ){
		
		for(int y = 0; y < dims[1]; y++ ){

			struct SimData sim_data;
				
			sim_data.seqData = seqData_defaults;
			sim_data.seqData.seq_type = fitPara->sequence;
			sim_data.seqData.TR = fitPara->tr;
			sim_data.seqData.TE = fitPara->te;
			sim_data.seqData.rep_num = dims[TE_DIM];
			sim_data.seqData.spin_num = fitPara->n_slcp;
			sim_data.seqData.num_average_rep = fitPara->averageSpokes;
			
			sim_data.voxelData = voxelData_defaults;
			sim_data.voxelData.r1 = 1 / (lim_T1[0] + x * lim_T1[1]/(dims[0]-1));
			sim_data.voxelData.r2 = 1 / (lim_T2[0] + x * lim_T2[1]/(dims[1]-1));
			sim_data.voxelData.m0 = 1;
			sim_data.voxelData.w = 0;
			
			sim_data.pulseData = pulseData_defaults;
			sim_data.pulseData.flipangle = 45.;
			sim_data.pulseData.RF_end = fitPara->rfduration;
			sim_data.gradData = gradData_defaults;
			sim_data.seqtmp = seqTmpData_defaults;
			
			if (NULL != fitPara->input_fa_profile) {
				
				long vfa_dims[DIMS];
				md_set_dims(DIMS, vfa_dims, 1);
				vfa_dims[READ_DIM] = fitPara->num_vfa;
				
				sim_data.seqData.variable_fa = md_alloc(DIMS, vfa_dims, CFL_SIZE);
				md_copy(DIMS, vfa_dims, sim_data.seqData.variable_fa, fitPara->input_fa_profile, CFL_SIZE);
			}
			
// 			debug_printf(DP_DEBUG3,"%f,\t%f,\t%f,\t%f,\t%d,\t%f,\t%f,\n", data.seqData.TR, data.seqData.TE, data.voxelData.r1, data.voxelData.r2, 	data.seqtmp.rep_counter, data.pulseData.RF_end, data.pulseData.flipangle );
			
			
			float mxySig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saR1Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saR2Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saDensSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];

			ode_bloch_simulation3(&sim_data, mxySig, saR1Sig, saR2Sig, saDensSig);
			
			//Add data to phantom
			for (int z = 0; z < dims[TE_DIM]; z++) {
				
				//changed x-and y-axis to have same orientation as measurements
				sensitivitiesR1[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = saR1Sig[z][1] + saR1Sig[z][0] * I; 
				sensitivitiesR2[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = saR2Sig[z][1] + saR2Sig[z][0] * I;
				sensitivitiesM0[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = saDensSig[z][1] + saDensSig[z][0] * I;
				phantom[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = mxySig[z][1] + mxySig[z][0] * I;
			}
			
			
			
		}
        
	}
	
	double mean_sig = md_znorm(DIMS, ksp_dims, kspace_data);
	
	double mean_r1 = md_znorm(DIMS, dims, sensitivitiesR1);
	scale[0] = mean_sig / mean_r1;
	
	double mean_r2 = md_znorm(DIMS, dims, sensitivitiesR2);
	scale[1] = mean_sig / mean_r2;
	
	double mean_m0 = md_znorm(DIMS, dims, sensitivitiesM0);
	scale[2] = mean_sig / mean_m0;
	
	
	debug_printf(DP_DEBUG1,"means:\t%f,\t%f,\t%f,\t%f\n", mean_sig, mean_r1, mean_r2, mean_m0);
	
}
