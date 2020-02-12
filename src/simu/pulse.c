
#include <math.h>

#include "misc/misc.h" // M_PI

#include "num/specfun.h"

#include "simu/simulation.h"

#include "pulse.h"


float pulse_energy(const struct simdata_pulse* pulse)
{	
	// Assuming pulse starts at t=0

	float c = M_PI / pulse->n / pulse->t0;
	float d = M_PI / pulse->t0;

	float si0 = Si( d * (pulse->rf_end / 2.));
	float si1 = Si(-d * (pulse->rf_end / 2.));
	float si2 = Si( (c - d) * pulse->rf_end / 2.);
	float si3 = Si(-(c - d) * pulse->rf_end / 2.);
	float si4 = Si( (c + d) * pulse->rf_end / 2.);
	float si5 = Si(-(c + d) * pulse->rf_end / 2.);

	return pulse->A * (1. - pulse->alpha) / d * (si0 - si1) + pulse->A * pulse->alpha / (2. * d) * (si2 - si3 + si4 - si5);
}


static float sincf(float x)
{
	return (0. == x) ? 1. : (sinf(x) / x);
}



float pulse_sinc(const struct simdata_pulse* pulse, float t)
{
	return pulse->A * ((1. - pulse->alpha) + pulse->alpha * cosf(M_PI * (t - pulse->rf_end / 2.) / (pulse->n * pulse->t0)))
				* sincf(M_PI * (t - pulse->rf_end / 2.) / pulse->t0);
}



