
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


void bloch_excitation(float out[3], float t, const float in[3], float m0, float t1, float t2, const float gb[3])
{
	(void)m0; (void)t1; (void)t2;
	assert(0. == gb[2]); // no gradient, rotating frame

	out[0] = in[0];
	out[1] = (in[2] * sinf(gb[0] * t) + in[0] * cosf(gb[0] * t));
	out[2] = (in[2] * cosf(gb[0] * t) - in[0] * sinf(gb[0] * t));
}


static void matf_copy(int N, int M, float out[N][M], const float in[N][M])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			out[i][j] = in[i][j];
}


void bloch_matrix_ode(float matrix[4][4], float m0, float t1, float t2, const float gb[3])
{
	float m[4][4] = {
		{	-1. / t2,	gb[2],		-gb[1],		0.	},
		{	-gb[2],		-1. / t2,	gb[0],		0.	},
		{	gb[1],		-gb[0],		-1. / t1,	m0 / t1 },
		{	0.,		0.,		0.,		0.	},
	};

	matf_copy(4, 4, matrix, m);
}



