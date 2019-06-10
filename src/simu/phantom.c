/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Simple numerical phantom which simulates image-domain or
 * k-space data with multiple channels.
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

#include "phantom.h"




#define MAX_COILS 8
#define COIL_COEFF 5

typedef complex float (*krn_t)(void* _data, const double mpos[3]);

static complex float xsens(unsigned int c, double mpos[3], void* data, krn_t fun)
{
	assert(c < MAX_COILS);
#if 1
	complex float val = 0.;

	long sh = (COIL_COEFF - 1) / 2;

	for (int i = 0; i < COIL_COEFF; i++)
		for (int j = 0; j < COIL_COEFF; j++)
			val += sens_coeff[c][i][j] * cexpf(-2.i * M_PI * ((i - sh) * mpos[0] + (j - sh) * mpos[1]) / 4.);
#else
	float p[3] = { mpos[0], mpos[1], mpos[2] };
	complex float val = coil(&coil_defaults, p, MAX_COILS, c);
#endif
	return val * fun(data, mpos);
}

/*
 * To simulate channels, we simply convolve with a few Fourier coefficients
 * of the sensitivities. See:
 *
 * M Guerquin-Kern, L Lejeune, KP Pruessmann, and M Unser, 
 * Realistic Analytical Phantoms for Parallel Magnetic Resonance Imaging
 * IEEE TMI 31:626-636 (2012)
 */
static complex float ksens(unsigned int c, double mpos[3], void* data, krn_t fun)
{
	assert(c < MAX_COILS);

	complex float val = 0.;

	for (int i = 0; i < COIL_COEFF; i++) {
		for (int j = 0; j < COIL_COEFF; j++) {

			long sh = (COIL_COEFF - 1) / 2;

			double mpos2[3] = { mpos[0] + (double)(i - sh) / 4.,
					    mpos[1] + (double)(j - sh) / 4.,
					    mpos[2] };

			val += sens_coeff[c][i][j] * fun(data, mpos2);
		}
	}

	return val;
}

static complex float nosens(unsigned int c, double mpos[3], void* data, krn_t fun)
{
	UNUSED(c);
	return fun(data, mpos);
}

struct data {

	const complex float* traj;
	const long* tstrs;

	bool sens;
	const long dims[3];
	void* data;
	krn_t fun;
};

static complex float xkernel(void* _data, const long pos[])
{
	struct data* data = _data;

	double mpos[3] = { (double)(pos[0] - data->dims[0] / 2) / (0.5 * (double)data->dims[0]),
                           (double)(pos[1] - data->dims[1] / 2) / (0.5 * (double)data->dims[1]),
                           (double)(pos[2] - data->dims[2] / 2) / (0.5 * (double)data->dims[2]) };

	return (data->sens ? xsens : nosens)(pos[COIL_DIM], mpos, data->data, data->fun);
}

static complex float kkernel(void* _data, const long pos[])
{
	struct data* data = _data;

	double mpos[3];

	if (NULL == data->traj) {

		mpos[0] = (double)(pos[0] - data->dims[0] / 2) / 2.;
		mpos[1] = (double)(pos[1] - data->dims[1] / 2) / 2.;
		mpos[2] = (double)(pos[2] - data->dims[2] / 2) / 2.;

	} else {

		assert(0 == pos[0]);
		mpos[0] = (&MD_ACCESS(DIMS, data->tstrs, pos, data->traj))[0] / 2.;
		mpos[1] = (&MD_ACCESS(DIMS, data->tstrs, pos, data->traj))[1] / 2.;
		mpos[2] = (&MD_ACCESS(DIMS, data->tstrs, pos, data->traj))[2] / 2.;
	}

	return (data->sens ? ksens : nosens)(pos[COIL_DIM], mpos, data->data, data->fun);
}




static void sample(const long dims[DIMS], complex float* out, const long tstrs[DIMS], const complex float* traj, void* krn_data, krn_t krn, bool kspace)
{
	struct data data = {

		.traj = traj,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = krn_data,
		.tstrs = tstrs,
		.fun = krn,
	};

	md_parallel_zsample(DIMS, dims, out, &data, kspace ? kkernel : xkernel);
}


struct krn2d_data {

	bool kspace;
	unsigned int N;
	const struct ellipsis_s* el;
};

static complex float krn2d(void* _data, const double mpos[3])
{
	struct krn2d_data* data = _data;
	return phantom(data->N, data->el, mpos, data->kspace);
}

static complex float krnX(void* _data, const double mpos[3])
{
	struct krn2d_data* data = _data;
	return phantomX(data->N, data->el, mpos, data->kspace);
}

struct krn3d_data {

	bool kspace;
	unsigned int N;
	const struct ellipsis3d_s* el;
};

static complex float krn3d(void* _data, const double mpos[3])
{
	struct krn3d_data* data = _data;
	return phantom3d(data->N, data->el, mpos, data->kspace);
}


void calc_phantom(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const _Complex float* traj)
{
	if (!d3)
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(shepplogan_mod), shepplogan_mod }, krn2d, kspace);
	else
		sample(dims, out, tstrs, traj, &(struct krn3d_data){ kspace, ARRAY_SIZE(shepplogan3d), shepplogan3d }, krn3d, kspace);
}


void calc_geo_phantom(const long dims[DIMS], complex float* out, bool kspace, int phtype, const long tstrs[DIMS], const _Complex float* traj)
{
	complex float* round = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* angular = md_alloc(DIMS, dims, CFL_SIZE);

	switch (phtype) {

	case 1:
		sample(dims, round, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(phantom_geo1), phantom_geo1 }, krn2d, kspace);
		sample(dims, angular, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(phantom_geo2), phantom_geo2 }, krnX, kspace);
		md_zadd(DIMS, dims, out, round, angular);
		break;

	case 2:
		sample(dims, round, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(phantom_geo4), phantom_geo1 }, krn2d, kspace);
		sample(dims, angular, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(phantom_geo3), phantom_geo2 }, krnX, kspace);
		md_zadd(DIMS, dims, out, round, angular);
		break;

	default:
		assert(0);
	}

	md_free(round);
	md_free(angular);
}


void calc_phantom_t1t2(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	(void) d3;
	
	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(t1t2phantom), t1t2phantom }, krn2d, kspace);
}

void calc_phantom_dens(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	(void) d3;
	
	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(dens_phantom), dens_phantom }, krn2d, kspace);
}

void calc_phantom_bart(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	(void) d3;
	
	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(bart_img), bart_img }, krn2d, kspace);
}





static complex float cnst_one(void* _data, const double mpos[2])
{
	UNUSED(_data);
	UNUSED(mpos);
	return 1.;
}

void calc_sens(const long dims[DIMS], complex float* sens)
{
	struct data data = {

		.traj = NULL,
		.sens = true,
		.dims = { dims[0], dims[1], dims[2] },
		.data = NULL,
		.fun = cnst_one,
	};

	md_parallel_zsample(DIMS, dims, sens, &data, xkernel);
}




void calc_circ(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	if (!d3)
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(phantom_disc), phantom_disc }, krn2d, kspace);
	else
		sample(dims, out, tstrs, traj, &(struct krn3d_data){ kspace, ARRAY_SIZE(phantom_disc3d), phantom_disc3d }, krn3d, kspace);
}

void calc_ring(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(phantom_ring), phantom_ring }, krn2d, kspace);
}


struct moving_ellipsis_s {

	struct ellipsis_s geom;
	complex float fourier_coeff_size[2][3];
	complex float fourier_coeff_pos[2][3];
};

static complex float fourier_series(float t, unsigned int N, const complex float coeff[static N])
{
	complex float val = 0.;

	for (unsigned int i = 0; i < N; i++)
		val += coeff[i] * cexpf(2.i * M_PI * t * (float)i);

	return val;
}

static void calc_moving_discs(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj,
				int N, const struct moving_ellipsis_s disc[N])
{
	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(TE_DIM), dims1, dims);

	for (int i = 0; i < dims[TE_DIM]; i++) {
#if 1
		struct ellipsis_s disc2[N];

		for (int j = 0; j < N; j++) {

			disc2[j] = disc[j].geom;
			disc2[j].center[0] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_pos[0]));
			disc2[j].center[1] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_pos[1]));
			disc2[j].axis[0] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_size[0]));
			disc2[j].axis[1] = crealf(fourier_series(i / (float)dims[TE_DIM], 3, disc[j].fourier_coeff_size[1]));
		}
#endif
		void* traj2 = (NULL == traj) ? NULL : ((void*)traj + i * tstrs[TE_DIM]);

		sample(dims1, (void*)out + i * strs[TE_DIM], tstrs, traj2, &(struct krn2d_data){ kspace, N, disc2 }, krn2d, kspace);
	}
}


void calc_moving_circ(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	struct moving_ellipsis_s disc[1] = { {
			.geom = phantom_disc[0],
			.fourier_coeff_size = { { 0.3, 0., 0, }, { 0.3, 0., 0. }, },
			.fourier_coeff_pos = { { 0, 0.5, 0., }, { 0., 0.5i, 0. } },
	} };

	calc_moving_discs(dims, out, kspace, tstrs, traj, ARRAY_SIZE(disc), disc);
}


void calc_heart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	struct moving_ellipsis_s disc[] = {
		{	.geom = { 0.4, { 0.5, 0.5 }, { 0., 0. }, 0. },
			.fourier_coeff_size = { { 0.5, 0.08, 0. }, { 0.5, 0.08, 0., }, },
			.fourier_coeff_pos = { { 0., 0.1, 0., }, { 0., 0.1i, 0., }, }, },
		{	.geom = { -0.4, { 0.4, 0.4 }, { 0., 0. }, 0. },
			.fourier_coeff_size = { { 0.4, 0.05, 0. }, { 0.4, 0.05, 0., }, },
			.fourier_coeff_pos = { { 0., 0.1, 0., }, { 0., 0.1i, 0., }, }, },
		{	.geom = { 1., { 0.4, 0.4 }, { 0., 0. }, 0. },
			.fourier_coeff_size = { { 0.4, 0.05, 0., }, { 0.4, 0.05, 0., }, },
			.fourier_coeff_pos = { { 0., 0.1, 0., }, { 0., 0.1i, 0., }, }, },
		{	.geom = { 0.5, { 0.9, 0.8 }, { 0, 0. }, 0. },
			.fourier_coeff_size = { { 0.9, 0.00, 0., }, { 0.8, 0.00, 0., }, },
			.fourier_coeff_pos = { { 0., 0., 0., }, { 0., 0., 0., }, }, },
		{	.geom = { -0.5, { 0.8, 0.7 }, { 0, 0. }, 0. },
			.fourier_coeff_size = { { 0.8, 0.00, 0., }, { 0.7, 0.00, 0., }, },
			.fourier_coeff_pos = { { 0., 0., 0., }, { 0., 0., 0., }, }, },
	};

	calc_moving_discs(dims, out, kspace, tstrs, traj, ARRAY_SIZE(disc), disc);
}


void calc_simu_phantom(void* _data,long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* in)
{
	
	(void) kspace; (void) tstrs;
	struct SimData* sim_data = _data;
	
	sim_data->seqData.rep_num = dims[TE_DIM];
	
	
	
	//create map
    complex float* phantom = md_alloc(DIMS, dims, CFL_SIZE);
    complex float* sensitivitiesR1 = md_alloc(DIMS, dims, CFL_SIZE);
    complex float* sensitivitiesR2 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* sensitivitiesM0 = md_alloc(DIMS, dims, CFL_SIZE);
    
    
    #pragma omp parallel for collapse(2)
    for(int x = 0; x < dims[0]; x++ ){
        
        for(int y = 0; y < dims[1]; y++ ){
			
			struct SimData data = *sim_data;
			
            //Get offset values from -pi to +pi
            //data.w = ( (float) y - (float) dims[1]/2. )/  ( (float) dims[1]/2.) * PI / data.TR;
            
            //Updating data
			float t1 = crealf(in[(y * dims[0]) + x]);
			float t2 = cimagf(in[(y * dims[0]) + x]);

            //Skip empty voxels
            if(t1 <= 0.001 || t2 <= 0.001){
                for(int z = 0; z < dims[TE_DIM]; z++){
                    phantom[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = 0.;
                    sensitivitiesR1[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = 0.;
                    sensitivitiesR2[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = 0.;
					sensitivitiesM0[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = 0.;
                }
                continue;
            }
            
            data.voxelData.r1 = 1/t1;  
            data.voxelData.r2 = 1/t2; 
			data.voxelData.m0 = 1;			// TODO: Change to add coil profiles
			
			debug_printf(DP_DEBUG3,"%f,\t%f,\t%f,\t%f,\t%d,\t%f,\t%f,\n", data.seqData.TR, data.seqData.TE, data.voxelData.r1, data.voxelData.r2, data.seqtmp.rep_counter, data.pulseData.RF_end, data.pulseData.flipangle );
			
			
			float mxySig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saR1Sig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saR2Sig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saDensSig[data.seqData.rep_num / data.seqData.num_average_rep][3];

			ode_bloch_simulation3( &data, mxySig, saR1Sig, saR2Sig, saDensSig, NULL );
			
			//Add data to phantom
			for(int z = 0; z < dims[TE_DIM]; z++){
				//changed x-and y-axis to have same orientation as measurements
				sensitivitiesR1[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = saR1Sig[z][1] + saR1Sig[z][0] * I; 
				sensitivitiesR2[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = saR2Sig[z][1] + saR2Sig[z][0] * I;
				sensitivitiesM0[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = saDensSig[z][1] + saDensSig[z][0] * I;
				phantom[ (z * dims[0] * dims[1]) + (y * dims[0]) + x] = mxySig[z][1] + mxySig[z][0] * I;
				
			} 
			
			md_copy(DIMS, dims, out, phantom, CFL_SIZE);
			
        }
        
    }
	
	
}



