/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Volkert Roeloffs
 * 2021 Nick Scholand,	nick.scholand@med.uni-goettingen.de
 */

#include <math.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/crb.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

void compute_crb(int P, float rCRB[P], complex float A[P][P], int M, int N, const complex float derivatives[M][N], const complex float signal[N], const unsigned long idx_unknowns[P-1]) {

	// assume first P-1 entries in derivates are w.r.t. parameters (i.e. not M0) 
	assert(P <= M + 1); // maximum 1 + M unknowns 	
 
	// null Fisher information first 
	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++) 
			A[i][j] = 0;

	// compute Fisher information matrix
	for (int n = 0; n < N; n++) {

		A[0][0] += conjf(signal[n])*signal[n];
		for (int i = 1; i < P; i++) {

			A[i][0] += conj(derivatives[idx_unknowns[i-1]][n])*signal[n];
			for (int j = 1; j <= i; j++)
				A[i][j] += conjf(derivatives[idx_unknowns[i-1]][n])*derivatives[idx_unknowns[j-1]][n];
		}
	}

	// complete upper triangle with compl. conjugates
	for (int j = 1; j < P; j++)
			for (int i = 0; i < j; i++)
				A[i][j] = conjf(A[j][i]);

	complex float A_inv[P][P];
	mat_inverse(P, A_inv, A);

	for (int i = 0; i < P; i++) 
		rCRB[i] = crealf(A_inv[i][i]);
}

void normalize_crb(int P, float rCRB[P], int N, float TR, float T1, float T2, float B1, float omega, const unsigned long idx_unknowns[P-1]) {
	
	UNUSED(omega);
	float normvalues[4];
	normvalues[0] = powf(T1,2); 
	normvalues[1] = powf(T2,2); 
	normvalues[2] = powf(B1,2);	
	normvalues[3] = 1;

	rCRB[0] *= N * TR; // M0 normalization
	for (int i = 1; i < P; i++) {

		rCRB[i] /= normvalues[idx_unknowns[i-1]];
		rCRB[i] *= N * TR;
	}
}

void getidxunknowns(int Q, unsigned long idx_unknowns[Q], long unknowns) {

	int j = 0;
	for (int i = 0; i < 4; i++) {

		if (1 & unknowns) { 

			idx_unknowns[j] = i;
			j++;
		}
		unknowns >>= 1;
	}
}

void display_crb(int P, float rCRB[P], complex float fisher[P][P], unsigned long idx_unknowns[P-1]) {

	bart_printf("Fisher information matrix: \n");
	for(int i = 0; i < P; i++) {
		for(int j = 0; j < P; j++) {
		bart_printf("%1.2f%+1.2fi ", crealf(fisher[i][j]), cimagf(fisher[i][j]));
		}
		bart_printf("\n");
	}
	char labels[][40] = {"rCRB T1", "rCRB T2", "rCRB B1", " CRB OF"};
	bart_printf("\n");
	bart_printf("rCRB M0: %3.3f\n", rCRB[0]);
	for(int i = 1; i < P; i++)
		bart_printf("%s: %3.3f\n", labels[idx_unknowns[i-1]], rCRB[i]);
	bart_printf("\n");
}


// FIXME: Fix variance: Fischer_{ij} = b_i^T b_j / \sigma^2
void fischer(int N, int P, float A[P][P], /*const*/ float der[P][N])
{
	// Set matrix zero
	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++)
			A[i][j] = 0.;

	float tmp = 0.;

	// Estimate Fischer matrix
	for (int i = 0; i < P; i++)
		for (int n = 0; n < N; n++) {

			tmp = der[i][n];

			for (int j = 0; j < P; j++)
				A[i][j] += tmp * der[j][n];
		}
}

// FIXME: Fix variance: Fischer_{ij} = b_i^T b_j / \sigma^2
void zfischer(int N, int P, complex float A[P][P], /*const*/ complex float der[P][N])
{
	// Set matrix zero
	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++)
			A[i][j] = 0.;

	complex float tmp = 0.;

	// Estimate Fischer matrix
	for (int i = 0; i < P; i++)
		for (int n = 0; n < N; n++) {

			tmp = der[i][n];

			for (int j = 0; j < P; j++)
				A[i][j] += tmp * conjf(der[j][n]);
		}
}



static void get_index(unsigned int D, const long dim1[D], const long dim2[D], int index[2])
{
	// parameter
	int pcounter = 0;

	// time
	int tcounter = 0;

	for (unsigned int i = 0; i < D; i++) {

		// both dims need to have non-zero elements in same dimension
		if ((1 < dim1[i]) && (1 < dim2[i])) {

			// Definition of parameter dim
			if (dim1[i] == dim2[i]) {

				index[0] = i;

				pcounter++;
			}
			else {	// time dim

				index[1] = i;

				tcounter++;
			}
		}
	}

	// Only single index each is allowed!
	assert(2 > pcounter);
	assert(2 > tcounter);

	assert(index[0] != index[1]);
}

/**
 * Estimate Fischer matrix (optr) from derivatives (iptr)
 * 	- Ensure matrix and derivatives share the same two dimensions
 *
 * FIXME: Fix variance: Fischer_{ij} = b_i^T b_j / \sigma^2
 */
void md_zfischer(unsigned int D, const long odims[D], complex float* optr, const long idims[D], const complex float* iptr)
{
	// Check if free for later use as intermediate dimension

	assert(1 == odims[AVG_DIM]);

	// Estimate parameter dimension [0] (where idims == odims) and time [1]

	int index[2] = { 0, 0 };
	get_index(DIMS, odims, idims, index);

	// Move time dim to AVG_DIM

	long tdims[DIMS];
	md_copy_dims(D, tdims, idims);
	md_transpose_dims(D, index[1], AVG_DIM, tdims, idims);

	long tstrs[DIMS];
	md_calc_strides(D, tstrs, tdims, CFL_SIZE);

	// Allocate memory for transposed data

	complex float* tmp = md_alloc_sameplace(DIMS, tdims, CFL_SIZE, iptr);

	// Move data to transposed array

	md_transpose(D, index[1], AVG_DIM, tdims, tmp, idims, iptr, sizeof(complex float));

	// Define max dims: P P T

	long max_dims[DIMS];
	md_copy_dims(D, max_dims, tdims);

	max_dims[index[1]] = odims[index[1]];

	// Calculate strides

	long ostrs[DIMS];
	md_calc_strides(D, ostrs, odims, CFL_SIZE);

	long istrs[DIMS];
	md_calc_strides(D, istrs, idims, CFL_SIZE);

	// Transposed stride to perform a multiplication with the transposed derivative matrix

	long tstrs2[DIMS];
	md_copy_dims(D, tstrs2, tstrs);

	tstrs2[index[1]] = tstrs[index[0]];
	tstrs2[index[0]] = 0L;

	// Multiplication

	md_ztenmulc2(D, max_dims, ostrs, optr, tstrs, tmp, tstrs2, tmp);

	md_free(tmp);
}