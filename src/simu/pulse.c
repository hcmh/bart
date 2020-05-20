
#include <math.h>

#include "misc/misc.h" // M_PI

#include "num/specfun.h"

#include "simu/simulation.h"

#include "pulse.h"



const struct simdata_pulse simdata_pulse_defaults = {

	.pulse_length = 1.,
	.rf_start = 0.,
	.rf_end = 0.009,
	.flipangle = 1.,
	.phase = 0.,
	.nl = 2.,
	.nr = 2.,
	.n = 2.,
	.t0 = 1.,
	.alpha = 0.46,
	.A = 1.,
	.energy_scale = 1.,
	.pulse_applied = false,
};

static float pulse_antiderivative(const struct simdata_pulse* pulse, float t)
{
	float c = M_PI / pulse->n / pulse->t0;

	return 	pulse->A * pulse->t0 *
			( pulse->alpha 	* ( Si(c * t * (pulse->n-1))
					+ Si(c * t * (pulse->n+1)) )
			-2 * (pulse->alpha-1) * Si(M_PI / pulse->t0 * t) )
		/ 2 / M_PI;
}


float pulse_integral(const struct simdata_pulse* pulse)
{
	float durh = (pulse->rf_end - pulse->rf_start) / 2.;

	return pulse_antiderivative(pulse, durh) - pulse_antiderivative(pulse, -durh);
}


static float sincf(float x)
{
	return (0. == x) ? 1. : (sinf(x) / x);
}



float pulse_sinc(const struct simdata_pulse* pulse, float t)
{
	float mid = (pulse->rf_start + pulse->rf_end) / 2.;

	t -= mid;

	return pulse->A * ((1. - pulse->alpha) + pulse->alpha * cosf(M_PI * t / (pulse->n * pulse->t0)))
				* sincf(M_PI * t / pulse->t0);
}



void pulse_create(struct simdata_pulse* pulse, float rf_start, float rf_end, float angle /*[°]*/, float phase, float nl, float nr, float alpha)
{
	// windowed sinc-pulses only
	pulse->rf_start = rf_start;
	pulse->rf_end = rf_end;
	pulse->pulse_length = rf_end - rf_start;
	pulse->flipangle = 90.;
	pulse->phase = phase;
	pulse->nl = nl;
	pulse->nr = nr;
	pulse->n = MAX(nl, nr);
	pulse->t0 = pulse->pulse_length / ( 2 + (nl-1) + (nr-1) );
	pulse->alpha = alpha;
	pulse->A = 1.;

	// Determine scaling factor to Ensure pi/2 = PulseIntegral -> 90 degree rotation
	float integral = pulse_integral(pulse); 

	float scaling = M_PI / 2. / integral;

	//Update parameters
	pulse->flipangle = angle;
	pulse->A = scaling / 90 * angle;
}
