
#include <complex.h>

#include "misc/nested.h"

typedef complex float CLOSURE_TYPE(krn_t)(int D, const float x[D], const float y[D]);
typedef void CLOSURE_TYPE(krn2_t)(int M, complex float k[M][M], int D, const float x[D], const float y[D]);

extern void rkhs_matrix(int N, complex float (*km)[N][N], int D, const float (*x)[N][D], krn_t krn, float alpha);
extern void rkhs_matrix2(int N, int M, complex float (*km)[N][N][M][M], int D, const float (*x)[N][D], krn2_t krn, float alpha);
extern void rkhs_cardinal(int N, int D, complex float u[N], const complex float km[N][N], const float x[N][D], const float p[D], krn_t krn);
extern void rkhs_cardinal2(int N, int M, int D, complex float u[N][M][M], const complex float km[N][N][M][M], const float x[N][D], const float p[D], krn2_t krn);

