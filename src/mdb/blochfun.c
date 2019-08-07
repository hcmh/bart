/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */


#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"

#include "nlops/nlop.h"

#include "model_Bloch.h"
#include "blochfun.h"

#define round(x)	((int) ((x) + .5))



struct blochFun_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* map_dims;
	const long* in_dims;
	const long* out_dims;
	const long* input_dims;
	
	const long* map_strs;
	const long* in_strs;
	const long* out_strs;
	const long* input_strs;
	
	complex float* tmp_map;
	
	//scaling factors
	float scaling_R1;
	float scaling_R2;
	float scaling_M0;
	
	//derivatives
	complex float* Sig;
	complex float* dR1;
	complex float* dR2;
	complex float* dM0;
	
	complex float* input_img;
	complex float* input_sp;
	
	struct modBlochFit fitParameter;
	
	bool use_gpu;
	
	int counter;

};

DEF_TYPEID(blochFun_s);


static void Bloch_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	double starttime = timestamp();
	debug_printf(DP_DEBUG2, "Started Forward Calculation\n");
	
	struct blochFun_s* data = CAST_DOWN(blochFun_s, _data);
	
	if (DP_DEBUG3 <= debug_level) {
		
		char name[255] = {'\0'};
		
		sprintf(name, "current_map_%02d", data->counter); 
		dump_cfl(name, data->N, data->in_dims, src);
		
		data->counter++;
	}
	
	
	// Allocate GPU memory
	complex float* R1scale_tmp;
	complex float* R2scale_tmp;
	complex float* M0scale_tmp;
	
	
#ifdef USE_CUDA
	if (data->use_gpu) {

		R1scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		R2scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		M0scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
	}
	else {

		R1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		R2scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		M0scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	}
#else
	R1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	R2scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	M0scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
#endif
	
	
	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	// Copy necessary files from GPU to CPU
	// R1 
	pos[COEFF_DIM] = 0;	
	const complex float* R1 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, R1scale_tmp, data->map_strs, R1, data->scaling_R1);

	complex float* R1scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, R1scale, R1scale_tmp, CFL_SIZE);
	
	// R2 
	pos[COEFF_DIM] = 1;	 
	const complex float* R2 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, R2scale_tmp, data->map_strs, R2, data->scaling_R2);

	complex float* R2scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, R2scale, R2scale_tmp, CFL_SIZE);
	
	// M0 
	pos[COEFF_DIM] = 2;	 
	const complex float* M0 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, M0scale_tmp, data->map_strs, M0, data->scaling_M0);

	complex float* M0scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, M0scale, M0scale_tmp, CFL_SIZE);


	//Allocate Output CPU memory
	complex float* Sig_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dR1_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dR2_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dM0_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);

	complex float* B1_cpu = NULL;

	if (NULL != data->input_img) {

		B1_cpu = md_alloc(data->N, data->input_dims, CFL_SIZE);
		md_copy(data->N, data->input_dims, B1_cpu, data->input_img, CFL_SIZE);
	}

	complex float* SP_cpu = NULL;

	if (NULL != data->input_sp) {

		long slcp_dims[DIMS];

		md_set_dims(DIMS, slcp_dims, 1);

		slcp_dims[READ_DIM] = data->fitParameter.n_slcp;

		SP_cpu = md_alloc(data->N, slcp_dims, CFL_SIZE);

		md_copy(data->N, slcp_dims, SP_cpu, data->input_sp, CFL_SIZE);

		debug_printf(DP_DEBUG2, "\n Slice Profile Estimates:\t");
		for (int i = 0; i < data->fitParameter.n_slcp; i++)
			debug_printf(DP_DEBUG2, "%f\t", cabsf(SP_cpu[i]) );
		debug_printf(DP_DEBUG2, "\n");
	}

	//Prepare reduced FOV
	md_zfill(data->N, data->out_dims, Sig_cpu, 0.);
	md_zfill(data->N, data->out_dims, dR1_cpu, 0.);
	md_zfill(data->N, data->out_dims, dR2_cpu, 0.);
	md_zfill(data->N, data->out_dims, dM0_cpu, 0.);

	//Get start and end values of reduced F0V
	int xstart = round( data->map_dims[0]/2. - (data->fitParameter.fov_reduction_factor * data->map_dims[0])/2. );
	int xend = round( xstart + data->fitParameter.fov_reduction_factor * data->map_dims[0] );
	int ystart = round( data->map_dims[1]/2. - (data->fitParameter.fov_reduction_factor * data->map_dims[1])/2. );
	int yend = round( ystart + data->fitParameter.fov_reduction_factor * data->map_dims[1] );
	int zstart = round( data->map_dims[2]/2. - (data->fitParameter.fov_reduction_factor * data->map_dims[2])/2. );
	int zend = round( zstart + data->fitParameter.fov_reduction_factor * data->map_dims[2] );

	//Solve rounding bug if fov_reduction_factor becomes to small, Change to round-up macro?!
	if (0 == zend)
		zend = 1;

// 	debug_printf(DP_DEBUG3, "x:(%d,%d),\ty:(%d,%d),\tz:(%d,%d),\n", xstart, xend, ystart, yend, zstart, zend);
// 	debug_printf(DP_DEBUG3, "seq: %d,\trfDuration: %f,\tTR:%f,\tTE:%f,\trep:%d,\tav:%d\n",
// 								data->fitParameter.sequence, data->fitParameter.rfduration, data->fitParameter.tr, 
// 								data->fitParameter.te, data->out_dims[TE_DIM], data->fitParameter.averageSpokes);

	int rm_first_echo = data->fitParameter.rm_no_echo;
// 	debug_printf(DP_DEBUG1, "Removed first %d echoes from signal.\n", rm_first_echo);

	float angle = 45.;

	#pragma omp parallel for collapse(3)
	for (int x = xstart; x < xend; x++)
		for (int y = ystart; y < yend; y++)
			for (int z = zstart; z < zend; z++) {

				//Calculate correct spatial position
				long spa_pos[DIMS];

				md_copy_dims(DIMS, spa_pos, data->map_dims);

				spa_pos[0] = x;
				spa_pos[1] = y;
				spa_pos[2] = z;

				long spa_ind = md_calc_offset(data->N, data->map_strs, spa_pos) / CFL_SIZE;

// 				debug_printf(DP_DEBUG3, "Pixel (x, y, z):\t(%d, %d, %d)\n", x, y, z);

				//Get effective flipangle from B1 map -> this do not include inversion efficiency
				float b1 = 1.;
				
				if (NULL != data->input_img) 
					b1 = cabsf( B1_cpu[spa_ind] ); 


				struct SimData sim_data;
				
				sim_data.seqData = seqData_defaults;
				sim_data.seqData.seq_type = 1;
				sim_data.seqData.TR = data->fitParameter.tr;
				sim_data.seqData.TE = data->fitParameter.te;
				sim_data.seqData.rep_num = (data->out_dims[TE_DIM] + rm_first_echo) * data->fitParameter.averageSpokes;
				sim_data.seqData.spin_num = data->fitParameter.n_slcp;
				sim_data.seqData.num_average_rep = data->fitParameter.averageSpokes;
				
				sim_data.voxelData = voxelData_defaults;
				sim_data.voxelData.r1 = crealf(R1scale[spa_ind]);
				sim_data.voxelData.r2 = crealf(R2scale[spa_ind]);
				sim_data.voxelData.m0 = crealf(M0scale[spa_ind]);
				sim_data.voxelData.w = 0;
				
				sim_data.pulseData = pulseData_defaults;
				sim_data.pulseData.flipangle = angle * b1;
				sim_data.pulseData.RF_end = data->fitParameter.rfduration;
				sim_data.gradData = gradData_defaults;
				sim_data.seqtmp = seqTmpData_defaults;


				float mxySig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
				float saR1Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
				float saR2Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
				float saDensSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];

// 				debug_printf(DP_DEBUG3, "R1: %f,\tR2: %f,\tM0: %f\n", sim_data.voxelData.r1, sim_data.voxelData.r2, sim_data.voxelData.m0);

// 				ode_bloch_simulation3( &sim_data, mxySig, saR1Sig, saR2Sig, saDensSig, SP_cpu );
				matrix_bloch_simulation( &sim_data, mxySig, saR1Sig, saR2Sig, saDensSig, SP_cpu );

				long curr_pos[DIMS];
				md_copy_dims(DIMS, curr_pos, spa_pos);
				
				long position = 0;
				
				for (int j = 0; j < sim_data.seqData.rep_num / sim_data.seqData.num_average_rep - rm_first_echo; j++) { 

					curr_pos[TE_DIM] = j;
					position = md_calc_offset(data->N, data->out_strs, curr_pos) / CFL_SIZE;

					//Scaling: dB/dRi = dB/dRis * dRis/dRi
					//Write to possible GPU memory
					dR1_cpu[position] = data->scaling_R1 * ( saR1Sig[j+rm_first_echo][1] + saR1Sig[j+rm_first_echo][0] * I );//+1 to skip first frame, where data is empty
					dR2_cpu[position] = data->scaling_R2 * ( saR2Sig[j+rm_first_echo][1] + saR2Sig[j+rm_first_echo][0] * I );
					dM0_cpu[position] = data->scaling_M0 * ( saDensSig[j+rm_first_echo][1] + saDensSig[j+rm_first_echo][0] * I);
					Sig_cpu[position] = mxySig[j+rm_first_echo][1] + mxySig[j+rm_first_echo][0] * I;
				}
			}
			
	debug_printf(DP_DEBUG3, "Copy data\n");
	
	md_copy(data->N, data->out_dims, data->dR1, dR1_cpu, CFL_SIZE);
	md_copy(data->N, data->out_dims, data->dR2, dR2_cpu, CFL_SIZE);
	md_copy(data->N, data->out_dims, data->dM0, dM0_cpu, CFL_SIZE);
	md_copy(data->N, data->out_dims, dst, Sig_cpu, CFL_SIZE); 
	
	md_free(R1scale_tmp);
	md_free(R2scale_tmp);
	md_free(M0scale_tmp);
	md_free(dR1_cpu);
	md_free(dR2_cpu);
	md_free(dM0_cpu);
	md_free(Sig_cpu);
	md_free(R1scale);
	md_free(R2scale);
	md_free(M0scale);
	
	if (NULL != data->input_sp)
		md_free(SP_cpu);
	
	if (NULL != data->input_img)
		md_free(B1_cpu);
	
	double totaltime = timestamp() - starttime;
	debug_printf(DP_DEBUG2, "Time = %.2f s\n", totaltime);
}


static void Bloch_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{

	struct blochFun_s* data = CAST_DOWN(blochFun_s, _data);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);
	
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);

	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
// 	const complex float* tmp_R1 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = dR1 * R1'
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR1);

	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
// 	const complex float* tmp_R2 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = dst + dR2 * R2'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR2);

	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
// 	const complex float* tmp_M0 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = dst + dM0 * M0'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dM0);

}



static void Bloch_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct blochFun_s* data = CAST_DOWN(blochFun_s, _data);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);


	pos[COEFF_DIM] = 0;	//R1
// 	complex float* tmp_map;
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR1);

	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
// 	tmp_map = (void*)dst + md_calc_offset(data->N, data->in_strs, pos);


	pos[COEFF_DIM] = 1;	//R2
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR2);

	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
// 	tmp_map = (void*)dst + md_calc_offset(data->N, data->in_strs, pos);


	pos[COEFF_DIM] = 2;	//M0
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dM0);

	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
// 	tmp_map = (void*)dst + md_calc_offset(data->N, data->in_strs, pos);
	
}


static void Bloch_del(const nlop_data_t* _data)
{
	struct blochFun_s* data = CAST_DOWN(blochFun_s, _data);
	
	md_free(data->Sig);
	md_free(data->dR1);
	md_free(data->dR2);
	md_free(data->dM0);
	
	md_free(data->tmp_map);
	
	md_free(data->input_img);
	md_free(data->input_sp);

	xfree(data->map_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);
	xfree(data->input_dims);
	
	xfree(data->map_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);
	xfree(data->input_strs);

	xfree(data);
}


struct nlop_s* nlop_Bloch_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long input_dims[N], const complex float* input_img, const complex float* input_sp, const struct modBlochFit* fitPara, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct blochFun_s, data);
	SET_TYPEID(blochFun_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nmstr);


	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(long[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
	data->in_strs = *PTR_PASS(nistr);


	data->N = N;
	data->scaling_R1 = fitPara->r1scaling;
	data->scaling_R2 = fitPara->r2scaling;
	data->scaling_M0 = fitPara->m0scaling;
	data->Sig = my_alloc(N, out_dims, CFL_SIZE);
	data->dR1 = my_alloc(N, out_dims, CFL_SIZE);
	data->dR2 = my_alloc(N, out_dims, CFL_SIZE);
	data->dM0 = my_alloc(N, out_dims, CFL_SIZE);
	
	data->tmp_map = my_alloc(N, map_dims, CFL_SIZE);
	
	
	if (NULL != input_img) {
		
		PTR_ALLOC(long[N], nindims);
		md_copy_dims(N, *nindims, input_dims);
		data->input_dims = *PTR_PASS(nindims);
		
		PTR_ALLOC(long[N], ninstr);
		md_calc_strides(N, *ninstr, input_dims, CFL_SIZE);
		data->input_strs = *PTR_PASS(ninstr);

		data->input_img = my_alloc(N, input_dims, CFL_SIZE);
		md_copy(N, input_dims, data->input_img, input_img, CFL_SIZE);
	}
	else {
		
		data->input_img = NULL;
		data->input_dims = NULL;
		data->input_strs = NULL;
	}
	
	if (NULL != input_sp) {

		long slcp_dims[DIMS];

		md_set_dims(DIMS, slcp_dims, 1);
		slcp_dims[READ_DIM] = fitPara->n_slcp;

		data->input_sp = my_alloc(N, slcp_dims, CFL_SIZE);

		md_copy(N, slcp_dims, data->input_sp, input_sp, CFL_SIZE);
	}
	else
		data->input_sp = NULL;
	
	
	

	//Set fitting parameter
	data->fitParameter.sequence = fitPara->sequence;
	data->fitParameter.rfduration = fitPara->rfduration;
	data->fitParameter.tr = fitPara->tr;
	data->fitParameter.te = fitPara->te;

	debug_printf(DP_DEBUG2, "TR: %f s,\t TE: %f s\n", data->fitParameter.tr, data->fitParameter.te);

	data->fitParameter.averageSpokes = fitPara->averageSpokes;
	data->fitParameter.fov_reduction_factor = fitPara->fov_reduction_factor;
	data->fitParameter.n_slcp = fitPara->n_slcp;
	data->fitParameter.rm_no_echo = fitPara->rm_no_echo;
	data->use_gpu = use_gpu;
	
	data->counter = 0;
	
	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), Bloch_fun, Bloch_der, Bloch_adj, NULL, NULL, Bloch_del);
}

