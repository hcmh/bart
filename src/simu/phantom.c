/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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

#include "misc/misc.h"
#include "misc/mri.h"

#include "simu/shepplogan.h"
#include "simu/sens.h"

#include "phantom.h"




#define MAX_COILS 8
#define COIL_COEFF 5

typedef complex float (*krn_t)(void* _data, const double mpos[3]);

static complex float xsens(unsigned int c, double mpos[3], void* data, krn_t fun)
{
	assert(c < MAX_COILS);

	complex float val = 0.;

	long sh = (COIL_COEFF - 1) / 2;

	for (int i = 0; i < COIL_COEFF; i++)
		for (int j = 0; j < COIL_COEFF; j++)
			val += sens_coeff[c][i][j] * cexpf(-2.i * M_PI * ((i - sh) * mpos[0] + (j - sh) * mpos[1]) / 4.);

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


void calc_phantom(const long dims[DIMS], complex float* out, bool d3, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	if (!d3)
		sample(dims, out, tstrs, traj, &(struct krn2d_data){ kspace, ARRAY_SIZE(shepplogan_mod), shepplogan_mod }, krn2d, kspace);
	else
		sample(dims, out, tstrs, traj, &(struct krn3d_data){ kspace, ARRAY_SIZE(shepplogan3d), shepplogan3d }, krn3d, kspace);
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



static void calc_moving_discs(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj,
				int N, const struct ellipsis_s disc[N], float cen, float size)
{
	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	long dims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(TE_DIM), dims1, dims);

	for (int i = 0; i < dims[TE_DIM]; i++) {
#if 1
		float x = sin(2. * M_PI * (float)i / (float)dims[TE_DIM]);
		float y = cos(2. * M_PI * (float)i / (float)dims[TE_DIM]);

		struct ellipsis_s disc2[N];

		for (int j = 0; j < N; j++) {

			disc2[j] = disc[j];
			disc2[j].center[0] = cen * x;
			disc2[j].center[1] = cen * y;
			disc2[j].axis[0] *= 1. + size * y;
			disc2[j].axis[1] *= 1. + size * y;
		}
#endif
		sample(dims1, (void*)out + i * strs[TE_DIM], tstrs, (void*)traj + i * tstrs[TE_DIM], &(struct krn2d_data){ kspace, N, disc2 }, krn2d, kspace);
	}
}


void calc_moving_circ(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	struct ellipsis_s disc[1] = { phantom_disc[0] };

	disc[0].axis[0] /= 3;
	disc[0].axis[1] /= 3;

	calc_moving_discs(dims, out, kspace, tstrs, traj, ARRAY_SIZE(disc), disc, 0.5, 0.);
}


void calc_heart(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj)
{
	struct ellipsis_s disc[] = {

		{	 1.0, { 0.5, 0.5 }, { 0., 0. }, 0. },
		{	-1.0, { 0.4, 0.4 }, { 0., 0. }, 0. },
		{	 0.5, { 0.4, 0.4 }, { 0., 0. }, 0. },
	};

	calc_moving_discs(dims, out, kspace, tstrs, traj, ARRAY_SIZE(disc), disc, 0.1, 0.1);
}





