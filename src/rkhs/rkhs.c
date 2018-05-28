/* Copyright 2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 * Vivek Athalye, Michael Lustig, and Martin Uecker. Parallel Magnetic
 * Resonance Imaging as Approximation in a Reproducing Kernel Hilbert Space,
 * Inverse Problems, in press (2015) arXiv:1310.7489 [physics.med-ph]
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/linalg.h"

#include "rkhs.h"

#ifdef RKHSGRID
#include "noncart/grid.h"
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


void comp_cholesky(int NN, double alpha, complex float* aha)
{
	for (int i = 0; i < NN; i++)
		aha[i * NN + i] += alpha;

	cholesky(NN, (complex float (*)[NN])aha);
}


complex float evaluate_kernel(int l, int k, const long dims[5], const complex float* kern, const float pos[3])
{
	long C = dims[3];
	assert(dims[3] == dims[4]);



#ifdef RKHSGRID
	float pos2[3];
	for (int l = 0; l < 3; l++)
		pos2[l] = ((float)dims[l] / 2.) + KERNEL_OVERSAMPLING * pos[l];

	float width = 3.; // see grid.c
	int kb_size = 128;
	complex float val[C * C];
	grid_pointH(C * C, dims, pos2, val, kern, width, kb_size, false, kb_table128);
	return val[l * C + k]; // FIXME wasteful
#else
	long d[3] = { 0, 0, 0 };

	for (int l = 0; l < 3; l++)
		if (dims[l] > 1)
			d[l] = (dims[l] / 2) + (int)round(KERNEL_OVERSAMPLING * pos[l]);

	assert((0 <= d[0]) && (d[0] < dims[0]));
	assert((0 <= d[1]) && (d[1] < dims[1]));
	assert((0 <= d[2]) && (d[2] < dims[2]));

	return kern[(((l * C + k) * dims[2] + d[2]) * dims[1] + d[1]) * dims[0] + d[0]];
#endif
}


static bool check_kmat_dims(unsigned int C, unsigned int N, const long kdims[8])
{
	return (   (N == kdims[0]) && (N == kdims[4])
	        && (1 == kdims[1]) && (1 == kdims[5])
	        && (1 == kdims[2]) && (1 == kdims[6])
	        && (C == kdims[3]) && (C == kdims[7]));
}


void calculate_kernelmatrix(const long kdims[8], complex float* kmat, int N, complex float* pos, const long dims[5], const complex float* kern)
{
	assert(dims[3] == dims[4]);

	int C = dims[3];
	assert(check_kmat_dims(C, N, kdims));

#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {

			float p[3] = { 0., 0., 0. };

			for (int l = 0; l < 3; l++)
					p[l] = creal(pos[i * 3 + l] - pos[j * 3 + l]);

			for (int l = 0; l < C; l++)
				for (int k = 0; k < C; k++)
					kmat[((l * N + i) * C + k) * N + j] = evaluate_kernel(l, k, dims, kern, p);
		}			
	}
}


complex float* comp_cardinal(const long kdims[3], long channels, const complex float* lhs, const complex float* aha)
{
	long kern_dims[5];
	kern_dims[0] = kdims[0];
	kern_dims[1] = kdims[1];
	kern_dims[2] = kdims[2];
	kern_dims[3] = channels;
	kern_dims[4] = channels;

	complex float* kernel = md_alloc(5, kern_dims, CFL_SIZE);

	long lhs_dims[8] = { kdims[0], kdims[1], kdims[2], channels, 1, 1, 1, channels };
	long coe_dims[8] = { kdims[0], kdims[1], kdims[2], channels, 1, 1, 1, channels };

	long N1 = md_calc_size(7, coe_dims); 
	long N2 = md_calc_size(7, lhs_dims); 

	assert(N1 == N2);

	long NN = N1;

	for (int i = 0; i < channels; i++)
		cholesky_solve(NN, kernel + N1 * i, (const complex float (*)[NN])aha, lhs + N2 * i);

	return kernel;
}




void calculate_lhs(const long kdims[8], complex float* lhs, float npos[3], int N, complex float* pos, const long dims[5], const complex float* kern)
{
	assert(dims[3] == dims[4]);

	int C = dims[3];
	assert(check_kmat_dims(C, N, kdims));

	for (int j = 0; j < N; j++) {

		float p[3] = { 0., 0., 0. };

		for (int l = 0; l < 3; l++)
			p[l] = (npos[l] - creal(pos[j * 3 + l]));

		for (int l = 0; l < C; l++)
			for (int k = 0; k < C; k++)
				lhs[(l * C + k) * N + j] = evaluate_kernel(l, k, dims, kern, p);
	}
}




void calculate_lhsH(const long kdims[8], complex float* lhs, float npos[3], int N, complex float* pos, const long dims[5], const complex float* kern)
{
	calculate_lhs(kdims, lhs, npos, N, pos, dims, kern);
	int C = dims[3];

	for (int j = 0; j < N; j++)
		for (int l = 0; l < C; l++)
			for (int k = 0; k < C; k++)
				lhs[(l * C + k) * N + j] = conj(lhs[(l * C + k) * N + j]);
}






void calculate_diag(const long kdims[8], complex float* dia, const long dims[5], const complex float* kern)
{
	assert(dims[3] == dims[4]);

	int N = kdims[0];

	int C = dims[3];
	assert(check_kmat_dims(C, N, kdims));

	for (int j = 0; j < N; j++) { // FIXME: ???

		float p[3] = { 0., 0., 0. };

		for (int l = 0; l < C; l++)
			dia[l] = evaluate_kernel(l, l, dims, kern, p);
	}
}



