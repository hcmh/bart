/* Copyright 2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>

#include "signals.h"




/*
 * Hybrid-state free precession in nuclear magnetic resonance. 
 * Jakob Assländer, Dmitry S. Novikov, Riccardo Lattanzi, Daniel K. Sodickson & Martijn A. Cloos.
 * Communications Physics. Volume 2, Article number: 73 (2019)
 */
const struct hsfp_model hsfp_defaults = {

	.t1 = 0.781,
	.t2 = 0.065,
	.tr = 0.0045,
	.repetitions = 1000,
	.beta = -1.,
	.pa_profile = NULL,
};


static float a_core(const struct hsfp_model* data, float t)
{
	float x = cabsf(data->pa_profile[(int)(t / data->tr)]);

	return sinf(x) * sinf(x) / data->t2 + cosf(x) * cosf(x) / data->t1;
}


static float a(const struct hsfp_model* data, float t_lim)
{
	float sum = 0.;

	for (float t = 0.; t < t_lim; t += data->tr)
		sum += a_core(data, t) * data->tr;

	return expf(-sum);
}


static float r0_core(const struct hsfp_model* data, float t)
{
	return cosf(cabsf(data->pa_profile[(int)(t / data->tr)])) / a(data, t);
}


static float r0(const struct hsfp_model* data)
{
	float tc = data->repetitions * data->tr;
	float sum = 0.;

	for (float t = 0.; t < tc; t += data->tr)
		sum += r0_core(data, t) * data->tr;

	float a_tc = a(data, tc);

	return data->beta / data->t1 * a_tc / (1. - data->beta * a_tc) * sum;
}


static float signal_hsfp(const struct hsfp_model* data, float r0_val, float t)
{
	float sum = 0.;

	for (float tau = 0.; tau < t; tau += data->tr)
		sum += r0_core(data, tau) * data->tr;

	return a(data, t) * (r0_val + 1. / data->t1 * sum);
}


void hsfp_simu(const struct hsfp_model* data, complex float* out)
{
	float r0_val = r0(data);

	for (int ind = 0; ind < data->repetitions; ind++)
		out[ind] = signal_hsfp(data, r0_val, (float)ind * data->tr);
}


/*
 * Time saving in measurement of NMR and EPR relaxation times.
 * Look DC, Locker DR.  Rev Sci Instrum 1970;41:250–251.
 */
const struct LookLocker_model looklocker_defaults = {

	.t1 = 1.,
	.m0 = 1.,
	.tr = 0.0041,
	.fa = 8.,
	.repetitions = 1000,
};

static float signal_looklocker(const struct LookLocker_model* data, int ind)
{
	float s0 = data->m0;
	float r1s = 1. / data->t1 - logf(cosf(data->fa)) / data->tr;
	float mss = s0 / (data->t1 * r1s);

	return mss - (mss + s0) * expf(-ind * data->tr * r1s);
}

void looklocker_model(const struct LookLocker_model* data, complex float* out)
{
	for (int ind = 0; ind < data->repetitions; ind++)
		out[ind] = signal_looklocker(data, ind);
}


/*
 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density.
 * Schmitt, P. , Griswold, M. A., Jakob, P. M., Kotas, M. , Gulani, V. , Flentje, M. and Haase, A., 
 * Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058, (2004)
 */
const struct IRbSSFP_model IRbSSFP_defaults = {

	.t1 = 1.,
	.t2 = 0.1,
	.m0 = 1.,
	.tr = 0.0045,
	.fa = 45.,
	.repetitions = 1000,
};

static float signal_IR_bSSFP(const struct IRbSSFP_model* data, int ind)
{
	float t1s = 1. / ((cosf(data->fa / 2.) * cosf(data->fa / 2.)) / data->t1 + (sinf(data->fa / 2.) * sinf(data->fa / 2.)) / data->t2);
	float s0 = data->m0 * sinf(data->fa / 2.);
	float stst = data->m0 * sinf(data->fa) / ((data->t1 / data->t2 + 1.) - cosf(data->fa) * (data->t1 / data->t2 - 1.));
	float inv = 1. + s0 / stst;

	return stst * (1. - inv * expf(-ind * data->tr / t1s));
}

void IR_bSSFP_model(const struct IRbSSFP_model* data, complex float* out)
{
	for (int ind = 0; ind < data->repetitions; ind++)
		out[ind] = signal_IR_bSSFP(data, ind);
}


