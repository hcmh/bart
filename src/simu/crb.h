/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Volkert Roeloffs
 * 2021 Nick Scholand,	nick.scholand@med.uni-goettingen.de
 */

#include <complex.h>

extern void compute_crb(int P, float rCRB[P], complex float A[P][P], int M, int N, const complex float derivatives[M][N], const complex float signal[N], const unsigned long idx_unknowns[P-1]);
extern void normalize_crb(int P, float rCRB[P], int N, float TR, float T1, float T2, float B1, float omega, const unsigned long idx_unknowns[P-1]); 
extern void getidxunknowns(int P, unsigned long idx_unknowns[P-1], long unknowns);
extern void display_crb(int P, float rCRB[P], complex float fisher[P][P], unsigned long idx_unknowns[P-1]);

extern void fischer(int N, int P, float A[P][P], /*const*/ float der[P][N]);
extern void zfischer(int N, int P, complex float A[P][P], /*const*/ complex float der[P][N]);
extern void md_zfischer(unsigned int D, const long odims[D], complex float* optr, const long idims[D], const complex float* iptr);

extern void compute_crb2(int N, int P, float crb[P], /*const*/ complex float der[P][N]);
