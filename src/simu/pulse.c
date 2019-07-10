
#include <math.h>

#include "misc/misc.h" // M_PI

#include "num/specfun.h"

#include "simu/simulation.h"

#include "pulse.h"


float get_pulse_energy(void * _pulseData)
{	

	struct PulseData* pulseData = _pulseData;
	//Assuming pulse starts at t=0
	
	float c = M_PI / pulseData->n / pulseData->t0;
	float d = M_PI / pulseData->t0;
	
	float si0 = Si( d * (pulseData->RF_end/2.) );
	float si1 = Si( - d * (pulseData->RF_end/2.) );
	float si2 = Si( (c - d) * pulseData->RF_end/2. );
	float si3 = Si( - (c - d) * pulseData->RF_end/2. );
	float si4 = Si( (c + d) * pulseData->RF_end/2. );
	float si5 = Si( - (c + d) * pulseData->RF_end/2. );
	
	return  pulseData->A * (1 - pulseData->alpha) / d * ( si0 - si1 ) + pulseData->A * pulseData->alpha / (2 * d) * ( si2 - si3 + si4 - si5 );
}


float sinc_pulse(void* _pulseData, float t)
{
	struct PulseData* pulseData = _pulseData;
	
	//assume pulse does not change much slighly around maximum
	if( t-pulseData->RF_end/2 == 0 ) 
		t += 0.000001;
		
	return pulseData->A * ( (1 - pulseData->alpha) + pulseData->alpha * cosf( M_PI * (t-pulseData->RF_end/2) / (pulseData->n * pulseData->t0) ) ) * sinf( M_PI * (t-pulseData->RF_end/2) / pulseData->t0 ) / ( M_PI * (t-pulseData->RF_end/2) / pulseData->t0 );
}



