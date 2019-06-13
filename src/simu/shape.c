
#include <complex.h>
#include <assert.h>
#include <math.h>

#include "geom/polygon.h"

#include "simu/shape.h"

complex double xpolygon(int N, const double pg[N][2], const double p[3])
{
	assert(0. == p[2]);

	int w = polygon_winding_number(N, pg, p);

	return (0 == w % 2) ? 0. : 1.;
}



static double sinc(double x)
{
	return (0. == x) ? 1. : (sin(x) / x);
}

static double sdot(const double a[2], const double b[2])
{
	return a[0] * b[0] + a[1] * b[1];
}


// https://arxiv.org/pdf/1703.00255.pdf
// FIXME: series expansion cases are missing

complex double kpolygon(int N, const double pg[N][2], const double q0[3])
{
	assert(1. == polygon_winding_number(N, pg, (double[]){ 0., 0. }));

	assert(0. == q0[2]);

	double q[2];

	q[0] = q0[0] * M_PI;
	q[1] = q0[1] * M_PI;

	double q2 = q[0] * q[0] + q[1] * q[1];

	if (0. == q2)
//	if (1.E-6 > q2)
		return polygon_area(N, pg) / 4.;

	double qx[2] = { -q[1], q[0] };

	complex double sum[2] = { 0., 0. };

	for (int i = 0; i < N; i++) {

		double E[2] = { pg[i][0] - pg[(i - 1 + N) % N][0],  pg[i][1] - pg[(i - 1 + N) % N][1] };
		double R[2] = { pg[i][0] + pg[(i - 1 + N) % N][0],  pg[i][1] + pg[(i - 1 + N) % N][1] };

		complex double x = sinc(sdot(q, E)) * cexp(1.i * sdot(q, R));

		sum[0] += E[0] * x;
		sum[1] += E[1] * x;
	}
		
	return (qx[0] * sum[0] + qx[1] * sum[1]) / (8.i * q2);
}

