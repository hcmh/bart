/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019	     Nick Scholand <nick.scholand@med.uni-goettingen.de>
 *
 * Simple numerical phantom which simulates image-domain or
 * k-space data with multiple channels and additional option for 
 * simple simulated phantom.
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
#include "num/splines.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "geom/logo.h"

#include "simu/shepplogan.h"
#include "simu/sens.h"
#include "simu/coil.h"
#include "simu/shape.h"
#include "simu/volume.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/seq_model.h"

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
		error("Invalid phantom type\n");
	}

	md_free(round);
	md_free(angular);
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


struct poly1 {

	int N;
	complex float coeff;
	double (*pg)[][2];
};

struct poly {

	bool kspace;
	int P;
	struct poly1 (*p)[];
};

static complex float krn_poly(void* _data, const double mpos[3])
{
	struct poly* data = _data;

	complex float val = 0.;

	for (int p = 0; p < data->P; p++)
		val += (*data->p)[p].coeff * (data->kspace ? kpolygon : xpolygon)((*data->p)[p].N, *(*data->p)[p].pg, mpos);

	return val;
}

void calc_star(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	struct poly poly = {
		kspace,
		1,
		&(struct poly1[]){
			{
			8,
			1.,
			&(double[][2]){
				{ -0.5, -0.5 },
				{  0.0, -0.3 },
				{ +0.5, -0.5 },
				{  0.3,  0.0 },
				{ +0.5, +0.5 },
				{  0.0, +0.3 },
				{ -0.5, +0.5 },
				{ -0.3,  0.0 },
			}
			}
		}
	};

	struct data data = {

		.traj = traj,
		.tstrs = tstrs,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &poly,
		.fun = krn_poly,
	};

	md_parallel_zsample(DIMS, dims, out, &data, kspace ? kkernel : xkernel);
}

struct poly3d {

	bool kspace;
	int N;
	double (*pg)[][3][3];
};

static complex float krn_poly3d(void* _data, const double mpos[3])
{
	struct poly3d* data = _data;
	return (data->kspace ? kvolume : xvolume)(data->N, *data->pg, mpos);
}

void calc_star3d(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	struct poly3d poly3d = {

		kspace,
		4,
		&(double[][3][3]){

			{ { 0., 0., 0. }, { 0.5, 0., 0. }, { 0., 0.5, 0. }, },
			{ { 0.5, 0., 0. }, { 0., 0., 0. }, { 0., 0., 0.5 }, },
			{ { 0., 0., 0. }, { 0., 0.5, 0. }, { 0., 0., 0.5 }, },
			{ { 0.5, 0., 0. }, { 0., 0., 0.5 }, { 0., 0.5, 0. }, },
		},
	};

	struct data data = {

		.traj = traj,
		.tstrs = tstrs,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &poly3d,
		.fun = krn_poly3d,
	};

	md_parallel_zsample(DIMS, dims, out, &data, kspace ? kkernel : xkernel);
}

#define ARRAY_SLICE(x, a, b) ({ __auto_type __x = &(x); assert((0 <= a) && (a < b) && (b <= ARRAY_SIZE(*__x))); ((__typeof__((*__x)[0]) (*)[b - a])&((*__x)[a])); })

void calc_bart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	int N = 11 + 6 + 6 + 8 + 4 + 16 + 6 + 8 + 6 + 6;
	double points[N * 11][2];

	struct poly poly = {
		kspace,
		10,
		&(struct poly1[]){
			{ 11 * 11, -1., ARRAY_SLICE(points,  0 * 11, 11 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 11 * 11, 17 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 17 * 11, 23 * 11) },
			{  8 * 11, -1., ARRAY_SLICE(points, 23 * 11, 31 * 11) },
			{  4 * 11, -1., ARRAY_SLICE(points, 31 * 11, 35 * 11) },
			{ 16 * 11, -1., ARRAY_SLICE(points, 35 * 11, 51 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 51 * 11, 57 * 11) },
			{  8 * 11, -1., ARRAY_SLICE(points, 57 * 11, 65 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 65 * 11, 71 * 11) },
			{  6 * 11, -1., ARRAY_SLICE(points, 71 * 11, 77 * 11) },
		}
	};

	for (int i = 0; i < N; i++) {

		for (int j = 0; j <= 10; j++) {

			double t = j * 0.1;
			int n = i * 11 + j;

			points[n][1] = cspline(t, bart_logo[i][0]) / 250. - 0.50;
			points[n][0] = cspline(t, bart_logo[i][1]) / 250. - 0.75;
		}
	}

	struct data data = {

		.traj = traj,
		.tstrs = tstrs,
		.sens = (dims[COIL_DIM] > 1),
		.dims = { dims[0], dims[1], dims[2] },
		.data = &poly,
		.fun = krn_poly,
	};

	md_parallel_zsample(DIMS, dims, out, &data, kspace ? kkernel : xkernel);
}





struct simulated_ellipsis_s {

	struct ellipsis_s geom; //geom.intensity = M0 for simulation
	float t1;
	float t2;
};


static void calc_signal_simu(struct SimData* sim_data, const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], 
			     const complex float* traj, int N, const struct simulated_ellipsis_s phantom[N])
{
	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(TE_DIM), dims1, dims);
	
	long dims2[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims2[READ_DIM] = N;
	dims2[PHS1_DIM] = dims[TE_DIM];
	
	
	long dims3[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims3[READ_DIM] = dims[TE_DIM];
	
	
	complex float* signal_evolution = md_alloc(DIMS, dims2, CFL_SIZE);
	
	// Apply simulation to all ellipses to determine time evolution of intensities
	#pragma omp parallel for
	for (int j = 0; j < N; j++) {
		
		struct SimData data = *sim_data;
		
		// Background ellipse need to be compensated
		data.voxelData.r1 = 1/phantom[j].t1;
		data.voxelData.r2 = 1/phantom[j].t2;
		data.voxelData.m0 = phantom[j].geom.intensity;
		
		if (data.seqData.analytical) {
			
			complex float* signal = md_alloc(DIMS, dims3, CFL_SIZE);
			
			if (5 == data.seqData.seq_type)
				looklocker_analytical(&data, signal);
			else if (1 == data.seqData.seq_type)
				IR_bSSFP_analytical(&data, signal);
			else
				debug_printf(DP_ERROR, "Analytical function of desired sequence is not provided.\n");
			
			for (int t = 0; t < dims[TE_DIM]; t++) 
				signal_evolution[j * dims[TE_DIM] + t] = signal[t];
			
		} else { // TODO: change to complex floats!!
			
			float mxySig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saR1Sig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saR2Sig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saDensSig[data.seqData.rep_num / data.seqData.num_average_rep][3];

			ode_bloch_simulation3(&data, mxySig, saR1Sig, saR2Sig, saDensSig);	// ODE simulation
// 			matrix_bloch_simulation(&data, mxySig, saR1Sig, saR2Sig, saDensSig);	// OBS simulation, does not work with hard-pulses!
			
			for (int t = 0; t < dims[TE_DIM]; t++) 
				signal_evolution[j * dims[TE_DIM] + t] = mxySig[t][1] + mxySig[t][0] * I;
		}
	}
	
	// Create phantom
	#pragma omp parallel for
	for (int i = 0; i < dims[TE_DIM]; i++) {
		
		struct ellipsis_s timestep_phantom[N];

		for (int j = 0; j < N; j++) {
			
			if (0 == j)
				timestep_phantom[j].intensity = signal_evolution[j * dims[TE_DIM] + i];
			else	//subtract background central circle to get unfolded signal of tube in image
				timestep_phantom[j].intensity = -signal_evolution[0 + i] + signal_evolution[j * dims[TE_DIM] + i];
			
			timestep_phantom[j].center[0] = phantom[j].geom.center[0];
			timestep_phantom[j].center[1] = phantom[j].geom.center[1];
			timestep_phantom[j].axis[0] = phantom[j].geom.axis[0];
			timestep_phantom[j].axis[1] = phantom[j].geom.axis[1];
			timestep_phantom[j].angle = phantom[j].geom.angle;
		}
		
		void* traj2 = (NULL == traj) ? NULL : ((void*)traj + i * tstrs[TE_DIM]);

		sample(dims1, (void*)out + i * strs[TE_DIM], tstrs, traj2, &(struct krn2d_data){ kspace, N, timestep_phantom }, krn2d, kspace);
	}
	
	md_free(signal_evolution);
}

void calc_phantom_t1t2(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{	
	sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(t1t2phantom), t1t2phantom }, krn2d, kspace) ;
}



void calc_phantom_t1t2_base(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{		
	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~(MD_BIT(TE_DIM)|MD_BIT(MAPS_DIM)), dims1, dims);
	
	
	print_dims(DIMS, dims);
	print_dims(DIMS, dims1);
	
	#pragma omp parallel for
	for (int i = 0; i < dims[MAPS_DIM]; i++) {
		
		struct ellipsis_s base[1];
		base[0] = t1t2phantom[i];
		
		for (int j = 0; j < dims[TE_DIM]; j++) {
			
			void* traj2 = (NULL == traj) ? NULL : ((void*)traj + j * tstrs[TE_DIM]);

			sample(dims1, (void*)out + i * strs[MAPS_DIM] + j * strs[TE_DIM], tstrs, traj2, &(struct krn2d_data){ kspace, ARRAY_SIZE(base), base }, krn2d, kspace);
		}
	}
	
}
