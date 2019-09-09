/* Copyright 2019 Nick Scholand
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * 
 * Hybrid-state free precession in nuclear magnetic resonance. 
 * Jakob Assländer, Dmitry S. Novikov, Riccardo Lattanzi, Daniel K. Sodickson & Martijn A. Cloos.
 * Communications Physics. Volume 2, Article number: 73 (2019)
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/gpuops.h"
#include "num/init.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "hsfp_model.h"


const struct HSFP_model hsfp_defaults = {
	
	.t1 = 0.781,
	.t2 = 0.065,
	.tr = 0.0045,
	.repetitions = 1000,
	.beta = -1,
	.pa_profile = NULL,
};


static float a_core(const struct HSFP_model* data, float t)
{
	return sinf(cabsf(data->pa_profile[(int)(t/data->tr)])) * sinf(cabsf(data->pa_profile[(int)(t/data->tr)])) / data->t2 +
		cosf(cabsf(data->pa_profile[(int)(t/data->tr)])) * cosf(cabsf(data->pa_profile[(int)(t/data->tr)])) / data->t1;
}


static float a(const struct HSFP_model* data, float t_lim)
{
	float sum = 0;
	
	for (float t = 0.; t < t_lim; t += data->tr)
		sum += a_core(data, t) * data->tr;
	
	return expf(-sum);
}


static float r0_core(const struct HSFP_model* data, float t)
{
	return cosf(cabsf(data->pa_profile[(int)(t/data->tr)])) / a(data, t); 
}


static float r0(const struct HSFP_model* data)
{
	float tc = data->repetitions * data->tr;
	
	float sum = 0;
	
	for (float t = 0.; t < tc; t += data->tr)
		sum += r0_core(data, t) * data->tr;
	
	float a_tc = a(data, tc);
	
	return data->beta / data->t1 * a_tc / (1 - data->beta * a_tc) * sum;
	
}


static float hsfp_signal(const struct HSFP_model* data, float r0_val, float t)
{
	float sum = 0;
	
	for (float tau = 0; tau < t; tau += data->tr)
		sum += r0_core(data, tau) * data->tr;
	
	return a(data, t) * ( r0_val + 1/data->t1 * sum);
}


void hsfp_simu(const struct HSFP_model* data, complex float* out)
{
	float r0_val = r0(data);
	
	for (int ind = 0; ind < data->repetitions; ind++)
		out[ind] = hsfp_signal(data, r0_val, (float) ind*data->tr) + 0 * I;
}
