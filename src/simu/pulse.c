
#include <math.h>

#include "misc/misc.h" // M_PI

#include "num/specfun.h"

#include "simu/simulation.h"

#include "pulse.h"


float get_pulse_energy(void * _pulse_data)
{	

	struct simdata_pulse* pulse = _pulse_data;
	//Assuming pulse starts at t=0
	
	float c = M_PI / pulse->n / pulse->t0;
	float d = M_PI / pulse->t0;
	
	float si0 = Si( d * (pulse->rf_end/2.) );
	float si1 = Si( - d * (pulse->rf_end/2.) );
	float si2 = Si( (c - d) * pulse->rf_end/2. );
	float si3 = Si( - (c - d) * pulse->rf_end/2. );
	float si4 = Si( (c + d) * pulse->rf_end/2. );
	float si5 = Si( - (c + d) * pulse->rf_end/2. );
	
	return  pulse->A * (1 - pulse->alpha) / d * ( si0 - si1 ) + pulse->A * pulse->alpha / (2 * d) * ( si2 - si3 + si4 - si5 );
}


float sinc_pulse(void* _pulse_data, float t)
{
	struct simdata_pulse* pulse = _pulse_data;
	
	//assume pulse does not change much slighly around maximum
	if( t-pulse->rf_end/2 == 0 ) 
		t += 0.000001;
		
	return pulse->A * ( (1 - pulse->alpha) + pulse->alpha * cosf( M_PI * (t-pulse->rf_end/2) / (pulse->n * pulse->t0) ) ) * sinf( M_PI * (t-pulse->rf_end/2) / pulse->t0 ) / ( M_PI * (t-pulse->rf_end/2) / pulse->t0 );
}



