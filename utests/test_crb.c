/* Authors:
 * 2021 Nick Scholand, nick.scholand@med.uni-goettingen.de
 */

#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "simu/crb.h"

#include "utest.h"


static bool test_fischer(void)
{
	enum { N = 3 };
	enum { P = 2 };

	// float f[N];	float X = 1.; float Y = 1.;
	float der[P][N];

	for (int i = 0; i < N; i++) {

		// f[i] = 2 * X * i + Y * i;

		der[0][i] = 2 * i; // df/dX
		der[1][i] = i; // df/dY
	}

	float A[P][P];
	float ref[P][P] = { {20., 10.}, {10., 5.}};

	fischer(N, P, A, der);

	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++)
			UT_ASSERT(1E-5 > (fabsf(ref[i][j] - A[i][j])));

	return true;
}

UT_REGISTER_TEST(test_fischer);


static bool test_zfischer(void)
{
	enum { N = 3 };
	enum { P = 2 };

	// float f[N];	float X = 1.; float Y = 1.;
	complex float der[P][N];

	for (int i = 0; i < N; i++) {

		// f[i] = (2+I) * X * i + I * Y * i;

		der[0][i] = (2 + I) * i; // df/dX
		der[1][i] = I * i; // df/dY
	}

	complex float A[P][P];
	complex float ref[P][P] = { {25., 5.-10.*I}, {5.+10.*I, 5.}};

	zfischer(N, P, A, der);

	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++)
			UT_ASSERT(1E-5 > (cabsf(ref[i][j] - A[i][j])));

	return true;
}

UT_REGISTER_TEST(test_zfischer);




static bool test_md_zfischer(void)
{
	enum { N = 3 };
	enum { P = 2 };

	// Allocate derivatives

	long ddims[DIMS];
	md_set_dims(DIMS, ddims, 1.);
	ddims[PHS1_DIM] = N;
	ddims[PHS2_DIM] = P;

	long dstrs[DIMS];
	md_calc_strides(DIMS, dstrs, ddims, CFL_SIZE);

	complex float* der = md_alloc(DIMS, ddims, CFL_SIZE);
	md_zfill(DIMS, ddims, der, 0.);

	// Estimate derivatives

	long pos[DIMS];
	md_copy_dims(DIMS, pos, ddims);

	long ind = 1L;

	// function: f[i] = (2+I) * X * i + I * Y * i;
	for (int i = 0; i < N; i++) {

		bart_printf("i; %d\n", i);

		// df/dX
		pos[PHS1_DIM] = i;
		pos[PHS2_DIM] = 0;

		ind = md_calc_offset(DIMS, dstrs, pos) / CFL_SIZE;

		der[ind] = (2 + I) * i;

		// df/dY
		pos[PHS2_DIM] = 1;
		ind = md_calc_offset(DIMS, dstrs, pos) / CFL_SIZE;

		der[ind] = I * i;
	}

	// Allocate fischer matrix

	long adims[DIMS];
	md_copy_dims(DIMS, adims, ddims);
	adims[PHS1_DIM] = P;

	long astrs[DIMS];
	md_calc_strides(DIMS, astrs, adims, CFL_SIZE);

	complex float* A = md_alloc(DIMS, adims, CFL_SIZE);

	// Define reference matrix

	complex float ref[P][P] = { {25., 5.-10.*I}, {5.+10.*I, 5.}};

	// Estimate Fischer matrix

	md_zfischer(DIMS, adims, A, ddims, der);

	// Comparison to reference

	md_copy_dims(DIMS, pos, adims);

	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++) {

			pos[PHS2_DIM] = i;
			pos[PHS1_DIM] = j;
			ind = md_calc_offset(DIMS, astrs, pos) / CFL_SIZE;

			if (1E-5 < (cabsf(ref[i][j] - A[ind])))
				return 0;
		}

	return true;
}

UT_REGISTER_TEST(test_md_zfischer);


static bool test_md_zfischer2(void)
{
	enum { N = 3 };
	enum { P = 2 };

	// Allocate derivatives

	long ddims[DIMS];
	md_set_dims(DIMS, ddims, 1.);
	ddims[TE_DIM] = N;
	ddims[MAPS_DIM] = P;

	long dstrs[DIMS];
	md_calc_strides(DIMS, dstrs, ddims, CFL_SIZE);

	complex float* der = md_alloc(DIMS, ddims, CFL_SIZE);
	md_zfill(DIMS, ddims, der, 0.);

	// Estimate derivatives

	long pos[DIMS];
	md_copy_dims(DIMS, pos, ddims);

	long ind = 1L;

	// function: f[i] = (2+I) * X * i + I * Y * i;
	for (int i = 0; i < N; i++) {

		bart_printf("i; %d\n", i);

		// df/dX
		pos[TE_DIM] = i;
		pos[MAPS_DIM] = 0;

		ind = md_calc_offset(DIMS, dstrs, pos) / CFL_SIZE;

		der[ind] = (2 + I) * i;

		// df/dY
		pos[MAPS_DIM] = 1;
		ind = md_calc_offset(DIMS, dstrs, pos) / CFL_SIZE;

		der[ind] = I * i;
	}

	// Allocate fischer matrix

	long adims[DIMS];
	md_copy_dims(DIMS, adims, ddims);
	adims[TE_DIM] = P;

	long astrs[DIMS];
	md_calc_strides(DIMS, astrs, adims, CFL_SIZE);

	complex float* A = md_alloc(DIMS, adims, CFL_SIZE);

	// Define reference matrix

	complex float ref[P][P] = { {25., 5.-10.*I}, {5.+10.*I, 5.}};

	// Estimate Fischer matrix

	md_zfischer(DIMS, adims, A, ddims, der);

	// Comparison to reference

	md_copy_dims(DIMS, pos, adims);

	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++) {

			pos[MAPS_DIM] = i;
			pos[TE_DIM] = j;
			ind = md_calc_offset(DIMS, astrs, pos) / CFL_SIZE;

			if (1E-5 < (cabsf(ref[i][j] - A[ind])))
				return 0;
		}

	return true;
}

UT_REGISTER_TEST(test_md_zfischer2);


static bool test_crb_comparison(void)
{
	enum { N = 3 };
	enum { P = 2 };

	// Cramer-Rao Bounds Interface 2

	complex float der[P][N];

	// f[i] = (2+I) * X * i + I * Y * i;
	for (int i = 0; i < N; i++) {

		der[0][i] = (2 + I) * i; // df/dX
		der[1][i] = I * i; // df/dY
	}

	float crb2[P];
	compute_crb2(N, P, crb2, der);

	// Cramer-Rao Bounds Interface 1

	enum { M = 1 };

	complex float der1[M][N];
	complex float der2[M][N];

	// f[i] = (2+I) * X * i + I * Y * i;
	for (int i = 0; i < N; i++) {

		der1[0][i] = (2 + I) * i; // df/dX
		der2[0][i] = I * i; // df/dY
	}

	float crb[P];

	complex float A[P][P];

	const unsigned long idx_unknowns[1] = { 0. };

	compute_crb(P, crb, A, M, N, (const complex float(*)[N]) der2, (const complex float *) der1, idx_unknowns);

	// Error

	UT_ASSERT((fabsf(crb[0] - crb2[0]) < 1E-5) && (fabsf(crb[1] - crb2[1]) < 1E-5));

	return true;
}

UT_REGISTER_TEST(test_crb_comparison);