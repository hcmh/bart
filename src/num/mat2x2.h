
#include <complex.h>

extern void mat2x2_vecmul(complex double out[2], const complex double mat[2][2], const complex double vec[2]);
extern complex double mat2x2_det(const complex double mat[2][2]);
extern complex double mat2x2_tr(const complex double mat[2][2]);
extern void mat2x2_charpol(complex double coeff[3], const complex double mat[2][2]);
extern void mat2x2_eig(complex double eval[2], complex double evec[2][2], const complex double mat[2][2]);

#if __GNUC__ < 5
#include "misc/pcaa.h"

#define mat2x2_vecmul(x, y, z) \
	mat2x2_vecmul(x, AR2D_CAST(complex double, 2, 2, y), z)

#define mat2x2_det(x) \
	mat2x2_det(AR2D_CAST(complex double, 2, 2, x))

#define mat2x2_tr(x) \
	mat2x2_tr(AR2D_CAST(complex double, 2, 2, x))

#define mat2x2_charpol(x, y) \
	mat2x2_charpol(x, AR2D_CAST(complex double, 2, 2, y))

#define mat2x2_eig(x, y, z) \
	mat2x2_eig(x, y, AR2D_CAST(complex double, 2, 2, z))

#endif

