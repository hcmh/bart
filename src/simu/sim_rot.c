#include <math.h>
#include <stdlib.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "sim_rot.h"

// Rotations in LEFT-handed coordinate system!
void rotx(float out[3], const float in[3], float angle)
{
	out[0] = in[0];
	out[1] = in[1] * cosf(angle) + in[2] * sinf(angle);
	out[2] = -in[1] * sinf(angle) + in[2] * cosf(angle);
}

void roty(float out[3], const float in[3], float angle)
{
	out[0] = in[0] * cosf(angle) - in[2] * sinf(angle);
	out[1] = in[1];
	out[2] = in[0] * sinf(angle) + in[2] * cosf(angle);
}

void rotz(float out[3], const float in[3], float angle)
{
	out[0] = in[0] * cosf(angle) + in[1] * sinf(angle);
	out[1] = -in[0] * sinf(angle) + in[1] * cosf(angle);
	out[2] = in[2];
}

void bloch_excitation2(float out[3], float in[3], float angle, float phase)
{
	float tmp[3] = { 0. };

	rotz(tmp, in, -phase);
	rotx(in, tmp, angle);
	rotz(out, in, phase);
}
