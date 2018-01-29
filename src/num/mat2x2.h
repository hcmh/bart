
#include <complex.h>

extern void mat2x2_vecmul(complex double out[2], const complex double mat[2][2], const complex double vec[2]);
extern complex double mat2x2_det(const complex double mat[2][2]);
extern complex double mat2x2_tr(const complex double mat[2][2]);
extern void mat2x2_charpol(complex double coeff[3], const complex double mat[2][2]);
extern void mat2x2_eig(complex double eval[2], complex double evec[2][2], const complex double mat[2][2]);


