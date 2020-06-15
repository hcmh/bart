#include <stdio.h>
#include <memory.h>
#include <complex.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/debug.h"

#include "simu/pulse.h"
#include "simu/slice_profile.h"

// TODO: Pass pulse struct to allow other pulse shapes
void estimate_slice_profile(unsigned int N, const long dims[N], complex float* out)
{
	// Determine Pulse Shape

	struct simdata_pulse pulse = simdata_pulse_defaults;

	float pulse_length = 0.0009;
	float flipangle = 6.;

	pulse_create(&pulse, 0., pulse_length, flipangle, 0., 2., 2., 0.46);

	float samples = 1000.;
	float dt =  pulse_length / samples;

	long pulse_dims[DIMS];
	md_set_dims(DIMS, pulse_dims, 1);

	pulse_dims[READ_DIM] = samples;

	complex float* envelope = md_alloc(DIMS, pulse_dims, CFL_SIZE);

	for (int i = 0; i < samples; i++)
		envelope[i] = pulse_sinc(&pulse, pulse.rf_start + i * dt );

	// dump_cfl("_pulse", DIMS, pulse_dims, envelope);

	// Zeropad for increased frequency sampling rate

	long pad_dims[DIMS];
	md_copy_dims(DIMS, pad_dims, pulse_dims);

	pad_dims[READ_DIM] = 6 * pulse_dims[READ_DIM];	// 6 seems to allow accurate frequency sampling

	complex float* padded = md_alloc(DIMS, pad_dims, CFL_SIZE);

	md_resize_center(DIMS, pad_dims, padded, pulse_dims, envelope, CFL_SIZE);

	// Determine Slice Profile

	complex float* slice_profile = md_alloc(DIMS, pad_dims, CFL_SIZE);

	fftc(DIMS, pad_dims, READ_FLAG, slice_profile, padded);
	
	// Find maximum in Slice Profile amplitude and scale it to 1

	float amp_max = 0.;

	for (int i = 0; i < pad_dims[READ_DIM]; i++)
		amp_max = (cabsf(slice_profile[i]) > amp_max) ? cabsf(slice_profile[i]) : amp_max;

	assert(0. < amp_max);
	debug_printf(DP_DEBUG3, "Max Amplitude of Slice Profile %f\n", amp_max);

	md_zsmul(DIMS, pad_dims, slice_profile, slice_profile, 1./amp_max);

	// Threshold to find slice frequency limits

	float limit = 0.01;	//Limit from which slice is taken into account

	int count = 0;

	for (long i = 0; i < pad_dims[READ_DIM]; i++) {

		if (cabsf(slice_profile[i]) > limit)
			count++;
		else
			slice_profile[i] = 0.;
	}

	assert(0 < count);

	// Separate counted elements

	long count_dims[DIMS];
	md_set_dims(DIMS, count_dims, 1);

	count_dims[READ_DIM] = count;

	complex float* slice_count = md_alloc(DIMS, count_dims, CFL_SIZE);

	#pragma omp parallel for
	for (long i = 0; i < count_dims[READ_DIM]; i++)
		slice_count[i] = slice_profile[(pad_dims[READ_DIM]-count)/2 + i + count%2]; //count%2 compensates for integer division error for odd `count`

	// dump_cfl("_slice_profile", DIMS, count_dims, slice_count);

	// Linear interpolation of final slice profile samples

	int slcprfl_samples = dims[READ_DIM] * 2;

	long slc_sample_dims[DIMS];
	md_set_dims(DIMS, slc_sample_dims, 1);

	slc_sample_dims[READ_DIM] = slcprfl_samples;

	complex float* slc_samples = md_alloc(DIMS, slc_sample_dims, CFL_SIZE);

	float steps = (float)(count_dims[READ_DIM] + 1) / (float)slcprfl_samples;	// +1 because of zero indexing of count_dims

	// #pragma omp parallel for
	for (int i = 0; i < slcprfl_samples; i++) {

		int ppos = (int) (i * steps);

		float pdiv =  (i * steps) - ppos;

		assert(0 <= pdiv);

		int npos = ppos + 1;

		slc_samples[i] = (slice_count[npos] - slice_count[ppos]) * pdiv + slice_count[ppos];

		debug_printf(DP_DEBUG1, "SLICE SAMPLES: i: %d,\t (i * steps): %f,\t ppos: %d,\t pdiv: %f,\t npos: %d,\tslice sample: %f\n", i, (i * steps), ppos, pdiv, npos, cabsf(slc_samples[i]));
	}

	// dump_cfl("_slice_samples", DIMS, slc_sample_dims, slc_samples);

	// Copy desired amount of Slice Profile Samples

	// #pragma omp parallel for
	for (int i = 0; i < dims[READ_DIM]; i++)
		out[i] = slc_samples[i];

	// dump_cfl("_final_slice_samples", DIMS, dims, out);

	md_free(envelope);
	md_free(padded);
	md_free(slice_profile);
	md_free(slice_count);
	md_free(slc_samples);
}