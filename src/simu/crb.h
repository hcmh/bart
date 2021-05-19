/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Volkert Roeloffs
 */

#include <complex.h>

extern void compute_crb(int P, float rCRB[P], complex float A[P][P], int M, int N, const complex float derivatives[M][N], const complex float signal[N], const unsigned long idx_unknowns[P-1]);
extern void normalize_crb(int P, float rCRB[P], int N, float TR, float T1, float T2, float B1, float omega, const unsigned long idx_unknowns[P-1]); 
extern void getidxunknowns(int P, unsigned long idx_unknowns[P-1], long unknowns);
extern void display_crb(int P, float rCRB[P], complex float fisher[P][P], unsigned long idx_unknowns[P-1]);
