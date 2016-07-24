#include <stdio.h>
#include <complex.h>

#include "polynom.h"

complex double polynom_eval(complex double x, int N, const complex double coeff[N + 1])
{
	// Horner's method: a_0 + x * (a_1 + x * (a_2 + ...)
	return coeff[0] + ((0 == N) ? 0. : (x * polynom_eval(x, N - 1, coeff + 1)));
}

void (polynom_derivative)(int N, complex double out[N], const complex double in[N + 1])
{
	for (int i = 0; i < N; i++)
		out[i] = (i + 1) * in[i + 1];
}

void polynom_integral(int N, complex double out[N + 2], const complex double in[N + 1])
{
	out[0] = 0.;

	for (int i = 0; i <= N; i++)
		out[i + 1] = in[i] / (i + 1);
}

complex double polynom_integrate(complex double st, complex double end, int N, const complex double coeff[N + 1])
{
	complex double int_coeff[N + 2];
	polynom_integral(N, int_coeff, coeff);

	return polynom_eval(end, N + 1, int_coeff) - polynom_eval(st, N + 1, int_coeff);
}

void polynom_monomial(int N, complex double coeff[N + 1], int O)
{
	for (int i = 0; i <= N; i++)
		coeff[i] = (O == i) ? 1. : 0.;
}

void polynom_from_roots(int N, complex double coeff[N + 1], const complex double root[N])
{
	// Vieta's formulas

	for (int i = 0; i < N; i++)
		coeff[i] = 0.;

	coeff[N] = 1.;

	// assert N < 
	for (unsigned long b = 1; b < (1u << N); b++) {

		complex double prod = 1.;
		int count = 0;

		for (int i = 0; i < N; i++) {

			if (b & (1 << i)) {

				prod *= root[i];
				count++;
			}
		}

		coeff[count - 1] += prod;
	}

	for (int i = 0; i <= N; i += 2)
		coeff[i] *= -1.;
}


void polynom_scale(int N, complex double out[N + 1], complex double scale, const complex double in[N + 1])
{
	complex double prod = 1.;

	for (int i = 0; i <= N; i++) {

		out[i] = prod * in[i];
		prod *= scale;
	}
}


void polynom_shift(int N, complex double out[N + 1], complex double shift, const complex double in[N + 1])
{
	// Taylor shift (there are faster FFT-based methods)

	for (int i = 0; i <= N; i++)
		out[i] = 0.;

	complex double tmp[N + 1];

	for (int i = 0; i <= N; i++)
		tmp[i] = in[i];

	complex double prod = 1.;

	for (int i = N; i >= 0; i--) {

		for (int j = 0; j < i; j++)
			out[i] += prod * tmp[i];

		polynom_derivative(i, tmp, tmp);

		prod *= shift;
	}
}


