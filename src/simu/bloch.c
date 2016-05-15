
#include <assert.h>
#include <math.h>

#include "num/vec3.h"

#include "bloch.h"



void bloch_ode(float out[3], const float in[3], float m0, float t1, float t2, const float gb[3])
{
	vec3_rot(out, in, gb);
	out[0] -= in[0] / t2;
	out[1] -= in[1] / t2;
	out[2] -= (in[2] - m0) / t1;
}


void bloch_relaxation(float out[3], float t, const float in[3], float m0, float t1, float t2, const float gb[3])
{
	assert((0. == gb[0]) && (0. == gb[1])); // no B1(t)

	out[0] =  (in[0] * cosf(gb[2] * t) - in[1] * sinf(gb[2] * t)) * expf(-t / t2);
	out[1] = -(in[0] * sinf(gb[2] * t) + in[1] * cosf(gb[2] * t)) * expf(-t / t2);
	out[2] = in[2] + (m0 - in[2]) * (1. - expf(-t / t1));
}






