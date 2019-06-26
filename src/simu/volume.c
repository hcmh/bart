
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "geom/triangle.h"
#include "geom/polyhedron.h"

#include "simu/shape.h"

#include "volume.h"


// similar to geom/mesh.c:tri_mesh_inside_p
complex double xvolume(int N, const double trid[N][3][3], const double xd[3])
{
	float d[3] = { 1., 1., 0. };
	int c = 0;
	int v = 0;
	int e = 0;

	float x[3] = { xd[0], xd[1], xd[2] };

	for (int i = 0; i < N; i++) {

		float tri[3][3];

		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				tri[a][b] = trid[i][a][b];

		float uv[2] = { 0., 0. };

		if (0. < triangle_intersect(uv, x, d, tri)) {

			c++;

			int b = 0;

			b += (0. == uv[0]) ? 1 : 0;
			b += (0. == uv[1]) ? 1 : 0;
			b += (1. == uv[0] + uv[1]) ? 1 : 0;

			if (1 == b)
				e++;	// at edge crossing two triangles

			if (2 == b)
				v++;	// at vertex 
		}
	}

	if (e > 0)
		return false;

	if (v > 0)
		return false;


	assert(0 == e % 2);
	assert(c >= e / 2);

	c -= e / 2;

	assert(0 == v % 3);
	assert(c >= v / 3);

	c -= v / 3;

	return (1 == c % 2) ? true : false;
}

static double vec3d_dot(const double a[3], const double b[3])
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void vec3d_smul(double o[3], double v, const double i[3])
{
	for (int j = 0; j < 3; j++)
		o[j] = i[j] * v;
}

static void vec3d_sub(double o[3], const double a[3], const double b[3])
{
	for (int j = 0; j < 3; j++)
		o[j] = a[j] - b[j];
}

static double vec3d_norm(const double i[3])
{
	return sqrtf(vec3d_dot(i, i));
}

static void vec3d_rot(double n[3], const double a[3], const double b[3])
{
	n[0] = a[1] * b[2] - a[2] * b[1];
	n[1] = a[2] * b[0] - a[0] * b[2];
	n[2] = a[0] * b[1] - a[1] * b[0];
}


static complex double kpolygon3d(double n[3], int N, const double pq[N][3], const double q0[3])
{
	double q[3];

	q[0] = q0[0] * M_PI;
	q[1] = q0[1] * M_PI;
	q[2] = q0[2] * M_PI;

	double d0[3];
	double d1[3];

	vec3d_sub(d0, pq[1], pq[0]);
	vec3d_sub(d1, pq[2], pq[0]);
	vec3d_rot(n, d0, d1);

	double nn = vec3d_norm(n);
	vec3d_smul(n, 1. / nn, n);

	double qq = vec3d_dot(n, q);
	double pp = vec3d_dot(n, pq[0]);


	double p1[3];

	double pn = vec3d_norm(d0);

	assert(0. != pn);

	vec3d_smul(p1, 1. / pn, d0);

	double p2[3];
	vec3d_rot(p2, n, p1);

	assert(1. == vec3d_norm(n));
	assert(1. == vec3d_norm(p1));
	assert(1. == vec3d_norm(p2));
	assert(0. == vec3d_dot(p1, p2));
	assert(0. == vec3d_dot(n, p1));
	assert(0. == vec3d_dot(n, p2));


	double qo[3];

	qo[0] = vec3d_dot(p1, q0);
	qo[1] = vec3d_dot(p2, q0);
	qo[2] = 0.;

	double pq2[N][2];

	for (int i = 0; i < N; i++) {

		pq2[i][0] = -vec3d_dot(p1, pq[i]);
		pq2[i][1] = -vec3d_dot(p2, pq[i]);
	}

	return cexp(-2.i * qq * pp) * kpolygon(N, pq2, qo);
}


complex double kvolume(int N, const double tri[N][3][3], const double q0[3])
{
	double q[3];

	q[0] = q0[0] * M_PI;
	q[1] = q0[1] * M_PI;
	q[2] = q0[2] * M_PI;

	double q2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];

#if 0
	// wire mesh
	
	double n[3];

	complex double sum = 0.;

	for (int i = 0; i < N; i++)
		sum += kpolygon3d(n, 3, tri[i], q0);

	return sum;
#else
	if (0. == q2)
		return polyhedron_vol(N, tri);

	complex double sum[3] = { 0., 0., 0. };

	for (int i = 0; i < N; i++) {

		double n[3];

		complex double v = kpolygon3d(n, 3, tri[i], q0);

		for (int j = 0; j < 3; j++)
			sum[j] += v * n[j];
	}

	return (q[0] * sum[0] + q[1] * sum[1] + q[2] * sum[2]) / (4.i * q2);
#endif
}

