/* Copyright 2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "signals.h"




/*
 * Hybrid-state free precession in nuclear magnetic resonance. 
 * Jakob Assländer, Dmitry S. Novikov, Riccardo Lattanzi, Daniel K. Sodickson & Martijn A. Cloos.
 * Communications Physics. Volume 2, Article number: 73 (2019)
 */
const struct signal_model signal_hsfp_defaults = {

	.t1 = 0.781,
	.t2 = 0.065,
	.tr = 0.0045,
	.beta = -1.,
};



struct r0_a_sum {

	float r0;
	float a;
};

static struct r0_a_sum r0_a_sum(const struct signal_model* data, int N, const float pa[N], int ind)
{
	struct r0_a_sum sum = { 0., 0. };

	for (int i2 = 0; i2 < ind; i2++) {

		float x = fabsf(pa[i2]);

		sum.a += sinf(x) * sinf(x) / data->t2 + cosf(x) * cosf(x) / data->t1;
		sum.r0 += cosf(x) * expf(sum.a * data->tr);
	}

	sum.r0 *= data->tr;
	sum.a *= data->tr;

	return sum;
}


static float r0(const struct signal_model* data, int N, const float pa[N])
{
	struct r0_a_sum sum = r0_a_sum(data, N, pa, N);

	return data->beta / data->t1 * sum.a / (1. - data->beta * sum.a) * sum.r0;
}


static float signal_hsfp(const struct signal_model* data, float r0_val, int N, const float pa[N], int ind)
{
	struct r0_a_sum sum = r0_a_sum(data, N, pa, ind);

	return sum.a * (r0_val + 1. / data->t1 * sum.r0);
}


void hsfp_simu(const struct signal_model* data, int N, const float pa[N], complex float out[N])
{
	float r0_val = r0(data, N, pa);

	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_hsfp(data, r0_val, N, pa, ind);
}


/*
 * Time saving in measurement of NMR and EPR relaxation times.
 * Look DC, Locker DR.  Rev Sci Instrum 1970;41:250–251.
 */
const struct signal_model signal_looklocker_defaults = {

	.t1 = 1.,
	.m0 = 1.,
	.tr = 0.0041,
	.fa = 8.,
};

static float signal_looklocker(const struct signal_model* data, int ind)
{
	float fa = data->fa;
	float t1 = data->t1;
	float m0 = data->m0;
	float tr = data->tr;

	float s0 = m0;
	float r1s = 1. / t1 - logf(cosf(fa)) / tr;
	float mss = s0 / (t1 * r1s);

	return mss - (mss + s0) * expf(-ind * tr * r1s);
}

void looklocker_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_looklocker(data, ind);
}


/*
 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density.
 * Schmitt, P. , Griswold, M. A., Jakob, P. M., Kotas, M. , Gulani, V. , Flentje, M. and Haase, A., 
 * Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058, (2004)
 */
const struct signal_model IRbSSFP_defaults = {

	.t1 = 1.,
	.t2 = 0.1,
	.m0 = 1.,
	.tr = 0.0045,
	.fa = 45.,
};

static float signal_IR_bSSFP(const struct signal_model* data, int ind)
{
	float fa = data->fa;
	float t1 = data->t1;
	float t2 = data->t2;
	float m0 = data->m0;
	float tr = data->tr;

	float fa2 = fa / 2.;
	float s0 = m0 * sinf(fa2);
	float r1s = (cosf(fa2) * cosf(fa2)) / t1 + (sinf(fa2) * sinf(fa2)) / t2;
	float mss = m0 * sinf(fa) / ((t1 / t2 + 1.) - cosf(fa) * (t1 / t2 - 1.));

	return mss - (mss + s0) * expf(-ind * tr * r1s);
}

void IR_bSSFP_model(const struct signal_model* data, int N, complex float out[N])
{
	for (int ind = 0; ind < N; ind++)
		out[ind] = signal_IR_bSSFP(data, ind);
}


