/* Authors:
 * 2021 Nick Scholand, nick.scholand@med.uni-goettingen.de
 */

#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <assert.h>

#include "misc/mri.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "simu/crb.h"
#include "simu/epg.h"
#include "simu/simulation.h"

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

UT_REGISTER_TEST(test_md_zfischer);


static bool test_getidxunknowns(void)
{
	long flag = 8321;
	int D = bitcount(flag);

	unsigned long ref[3] = { 0, 7, 13 };

	unsigned long ind[D];
	getidxunknowns(D, ind, flag);

	UT_ASSERT(	(3 == D) &&
			(ref[0] == ind[0]) &&
			(ref[1] == ind[1]) &&
			(ref[2] == ind[2]) );

	return true;
}

UT_REGISTER_TEST(test_getidxunknowns);

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
	compute_crb2(N, P, crb2, P, der, 3); // 3 is flag for dims 0, 1

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


static bool test_crb_interfaces_epg_simu(void)
{
	// EPG simulation

	int N = 24;
	int M = 2*N;

	complex float signal[N];
	complex float states[3][M][N];
	complex float dsignal[4][N];
	complex float dstates[4][3][M][N];

	float FA = 15.;
	float TR = 0.005;
	float T1 = 1000;
	float T2 = 100;
	float B1 = 1.;
	float offres = 0.;
	long SP = 0L;

	flash_epg_der(N, M, signal, states, dsignal, dstates, FA, TR, T1, T2, B1, offres, SP);

	// CRB Interface 1

	long unknowns = 5; // dim 0: T1, dim 2: B1

	int Q = bitcount(unknowns);

	unsigned long idx_unknowns[Q];

	getidxunknowns(Q, idx_unknowns, unknowns);

	int P = Q + 1; // selected unknowns + M0
	complex float fisher[P][P];
	float rCRB[P];

	compute_crb(P, rCRB, fisher, 4, N, dsignal, signal, idx_unknowns);

	// CRB Interface 2

	// conversion of derivatives from epg to single array
	complex float der[P+1][N];

	for(int i = 0; i < N; i++) {

		der[0][i] = dsignal[0][i];	// dT1
		der[1][i] = signal[i];		// dM0
		der[2][i] = dsignal[1][i];	// dT2
		der[3][i] = dsignal[2][i];	// dB1
	}

	long flag = 11; // dim 0: dT1,  dim 1: dM0, dim 3: dB1
	float crb[P];

	compute_crb2(N, P, crb, P+1, der, flag); // +1 because derivative also still includes dT2 here

	// Test for M0 and B1 CRB
	UT_ASSERT((fabsf(crb[1] - rCRB[0]) < 1E-2) && (fabsf(crb[2] - rCRB[2]) < 1E-2));

	return true;
}

UT_REGISTER_TEST(test_crb_interfaces_epg_simu);


static bool test_crb_ode_matrix(void)
{
	int N = 24;
	int P = 3;

	// ODE simulation

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = 2;	// FLASH
	sim_data.seq.tr = 0.005;
	sim_data.seq.te = 0.003;
	sim_data.seq.rep_num = N;
	sim_data.seq.spin_num = 1;
	sim_data.seq.num_average_rep = 1;
	sim_data.seq.inversion_pulse_length = 0.00001;
	sim_data.seq.prep_pulse_length = 0.00001;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1 = 1.;
	sim_data.voxel.r2 = 10.;
	sim_data.voxel.m0 = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.flipangle = 15.;
	sim_data.pulse.rf_end = 0.001;
	sim_data.pulse.bwtp = 4.;

	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;

	// ODE

	float crb[P];
	bloch_simulation_crb(N, P, &sim_data, crb, true);

	// bart_printf("CRB: %f,\t %f,\t %f\n", crb[0], crb[1], crb[2]);

	// matrix ODE

	float crb2[P];
	bloch_simulation_crb(N, P, &sim_data, crb2, false);

	// bart_printf("CRB: %f,\t %f,\t %f\n", crb2[0], crb2[1], crb2[2]);

	float tol = 1.E-3; //[%]

	UT_ASSERT(	(fabsf(crb[0] - crb2[0]) < tol*crb[0]) &&
			(fabsf(crb[1] - crb2[1]) < tol*crb[1]) &&
			(fabsf(crb[2] - crb2[2]) < tol*crb[2]) );

	return true;
}

UT_REGISTER_TEST(test_crb_ode_matrix);


