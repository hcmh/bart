
#include "num/ode.h"

#include "simu/bloch.h"

#include "utest.h"

#if 0
static void lorenz(float out[3], const float in[3], float sigma, float rho, float beta)
{
	out[0] = sigma * (in[1] - in[0]);
	out[1] = in[0] * (rho - in[2]) - in[1];
	out[2] = in[0] * in[1] - beta * in[2];
}


static void lorenz_fun(void* data, float* out, float t, const float* in)
{
	(void)data; (void)t;
	lorenz(out, in, 10., 28., 8. / 3.);
}
#endif



static void bloch_fun(void* data, float* out, float t, const float* in)
{
	(void)data; (void)t;
	bloch_ode(out, in, 1., WATER_T1, WATER_T2, (float[3]){ 0., 0., GAMMA_H1 * SKYRA_GRADIENT * 0.0001 });
}


static bool test_ode_bloch(void)
{
	float x[3] = { 1., 0., 0. };
	float x0[3] = { 1., 0., 0. };
	float x2[3] = { 1., 0., 0. };
	float h = 0.1;
	float tol = 0.000001;
	float end = 0.2;

	float k[6][3];
	bloch_fun(NULL, k[0], 0., x);

	for (float t = 0.; t < end; ) {

		float ynp[3];
	repeat:
		;
		float err = dormand_prince_step2(h, 3, ynp, t, x, k, NULL, bloch_fun);

		float h_new = h * dormand_prince_scale(tol, err);

		if (err > tol) {

			h = h_new;		
			bloch_fun(NULL, k[0], t, x);	// recreate correct k[0] which has been overwritten
			goto repeat;
		}

		t += h;
		h = h_new;

		for (unsigned int i = 0; i < 3; i++)
			x[i] = ynp[i];

		bloch_relaxation(x2, t, x0, 1., WATER_T1, WATER_T2, (float[3]){ 0., 0., GAMMA_H1 * SKYRA_GRADIENT * 0.0001 });
	}

	return (x[0] - x[2] < 1.E-4);
}

UT_REGISTER_TEST(test_ode_bloch);
	
