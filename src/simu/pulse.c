
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



float pulse_energy(const struct simdata_pulse* pulse)
{	
	float c = M_PI / pulse->n / pulse->t0;
	float d = M_PI / pulse->t0;

	float durh = (pulse->rf_end - pulse->rf_start) / 2.;

	float si0 = Si( d * durh);
	float si1 = Si(-d * durh);
	float si2 = Si( (c - d) * durh);
	float si3 = Si(-(c - d) * durh);
	float si4 = Si( (c + d) * durh);
	float si5 = Si(-(c + d) * durh);

	return    pulse->A * (1. - pulse->alpha) / d * (si0 - si1)
		+ pulse->A * pulse->alpha / (2. * d) * (si2 - si3 + si4 - si5);
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
	// For windowed sinc-pluses only
	pulse->rf_start = rf_start;
	pulse->rf_end = rf_end;
	pulse->pulse_length = rf_end - rf_start;
	pulse->flipangle = angle;
	pulse->phase = phase;
	pulse->nl = nl;
	pulse->nr = nr;
	pulse->n = MAX(nl, nr);
	pulse->t0 = pulse->pulse_length / ( 2 + (nl-1) + (nr-1) );
	pulse->alpha = alpha;
	pulse->A = 1;

	float energy = pulse_energy(pulse);

	// WTF is this?
	float calibration_energy = 0.991265;//2.3252; // turns M by 90°

	// change scale to reach desired flipangle
	pulse->A = (calibration_energy / energy) / 90 * angle;
}
