
#include <math.h>
//sqrt

// #include "iter/vec_iter.h"

#include "ode.h"


static void vec_saxpy(unsigned int N, float dst[N], const float a[N], float alpha, const float b[N])
{
	for (unsigned int i = 0; i < N; i++)
		dst[i] = a[i] + alpha * b[i];
}

static void vec_copy(unsigned N, float dst[N], const float src[N])
{
	vec_saxpy(N, dst, src, 0., src);
}

static float vec_sdot(unsigned int N, const float a[N], const float b[N])
{
	float ret = 0.;

	for (unsigned int i = 0; i < N; i++)
		ret += a[i] * b[i];

	return ret;
}

static float vec_norm(unsigned int N, const float x[N])
{
	return sqrtf(vec_sdot(N, x, x));
}



#define tridiag(s) (s * (s + 1) / 2)

static void runga_kutta_step(float h, unsigned int s, const float a[tridiag(s)], const float b[s], const float c[s - 1], unsigned int N, unsigned int K, float k[K][N], float ynp[N], float tmp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	vec_saxpy(N, ynp, yn, h * b[0], k[0]);

	for (unsigned int l = 0, t = 1; t < s; t++) {

		vec_copy(N, tmp, yn);

		for (unsigned int r = 0; r < t; r++, l++)
			vec_saxpy(N, tmp, tmp, h * a[l], k[r % K]);

		f(data, k[t % K], tn + h * c[t - 1], tmp);

		vec_saxpy(N, ynp, ynp, h * b[t], k[t % K]);
	}
}

// Runga-Kutta 4

void rk4_step(float h, unsigned int N, float ynp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	const float c[3] = { 0.5, 0.5, 1. };

	const float a[6] = {
		0.5,
		0.0, 0.5,
		0.0, 0.0, 1.0,
	};
	const float b[4] = { 1. / 6., 1. / 3., 1. / 3., 1. / 6. };

	float k[1][N];	// K = 1 because only diagonal elements are used
	f(data, k[0], tn, yn);

	float tmp[N];
	runga_kutta_step(h, 4, a, b, c, N, 1, k, ynp, tmp, tn, yn, data, f);
}



/*
 * Dormand JR, Prince PJ. A family of embedded Runge-Kutta formulae,
 * Journal of Computational and Applied Mathematics 6:19-26 (1980).
 */
void dormand_prince_step(float h, unsigned int N, float ynp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	const float c[6] = { 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1. };

	const float a[tridiag(7)] = {
		1. / 5.,
		3. / 40.,	9. / 40.,
		44. / 45.,	-56. / 15.,	32. / 9.,
		19372. / 6561.,	-25360. / 2187., 64448. / 6561., -212. / 729.,
		9017. / 3168.,  -355. / 33.,	46732. / 5247.,	49. / 176.,	-5103. / 18656.,
		35. / 384.,	0.,		500. / 1113.,	125. / 192.,	-2187. / 6784.,	11. / 84.,
	};

	const float b[7] = { 5179. / 57600., 0.,  7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40. };

	float k[6][N];
	f(data, k[0], tn, yn);

	float tmp[N];
	runga_kutta_step(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, data, f);
}



float dormand_prince_scale(float tol, float err)
{
#if 0
	float sc = 0.75 * powf(tol / err, 1. / 5.);

	return (sc < 2.) ? sc : 2.;
#else
	float sc = 1.25 * powf(err / tol, 1. / 5.);

	return 1. / ((sc > 1. / 2.) ? sc : (1. / 2.));
#endif
}



float dormand_prince_step2(float h, unsigned int N, float ynp[N], float tn, const float yn[N], float k[6][N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	const float c[6] = { 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1. };

	const float a[tridiag(7)] = {
		1. / 5.,
		3. / 40.,	9. / 40.,
		44. / 45.,	-56. / 15.,	32. / 9.,
		19372. / 6561.,	-25360. / 2187., 64448. / 6561., -212. / 729.,
		9017. / 3168.,  -355. / 33.,	46732. / 5247.,	49. / 176.,	-5103. / 18656.,
		35. / 384.,	0.,		500. / 1113.,	125. / 192.,	-2187. / 6784.,	11. / 84.,
	};

	const float b[7] = { 5179. / 57600., 0.,  7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40. };

	float tmp[N];
	runga_kutta_step(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, data, f);

	vec_saxpy(N, tmp, tmp, -1., ynp);
	return vec_norm(N, tmp);
}


void ode_interval(float h, float tol, unsigned int N, float x[N], float st, float end, void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	float k[6][N];
	f(data, k[0], 0., x);

	if (h > end - st)
		h = end - st;

	for (float t = st; t < end; ) {

		float ynp[N];
	repeat:
		;
		float err = dormand_prince_step2(h, N, ynp, t, x, k, data, f);

		float h_new = h * dormand_prince_scale(tol, err);

		if (err > tol) {

			h = h_new;
			f(data, k[0], t, x);	// recreate correct k[0] which has been overwritten
			goto repeat;
		}

		t += h;
		h = h_new;

		if (t + h > end)
			h = end - t;

		for (unsigned int i = 0; i < N; i++)
			x[i] = ynp[i];
	}
}
