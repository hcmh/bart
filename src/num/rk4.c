
#include "rk4.h"

// Runga-Kutta 4

void rk4_step(float h, unsigned int N, float ynp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn))
{
	float tmp[N];
	float k[N];

	f(data, k, tn, yn);

	for (unsigned int i = 0; i < N; i++)
		ynp[i] = yn[i] + h / 6. * k[i];

	for (unsigned int i = 0; i < N; i++)
		tmp[i] = yn[i] + h / 2. * k[i];

	f(data, k, tn + h / 2., tmp);

	for (unsigned int i = 0; i < N; i++)
		ynp[i] += h / 3. * k[i];

	for (unsigned int i = 0; i < N; i++)
		tmp[i] = yn[i] + h / 2. * k[i];

	f(data, k, tn + h / 2., tmp);

	for (unsigned int i = 0; i < N; i++)
		ynp[i] += h / 3. * k[i];

	for (unsigned int i = 0; i < N; i++)
		tmp[i] = yn[i] + h * k[i];

	f(data, k, tn + h, tmp);

	for (unsigned int i = 0; i < N; i++)
		ynp[i] += h / 6. * k[i];
}

