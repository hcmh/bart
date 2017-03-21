
#include <complex.h>

#include "num/polynom.h"

#include "mat2x2.h"


void (mat2x2_vecmul)(complex double out[2], const complex double mat[2][2], const complex double vec[2])
{
	out[0] = mat[0][0] * vec[0] + mat[0][1] * vec[1];
	out[1] = mat[1][0] * vec[0] + mat[1][1] * vec[1];
}

complex double (mat2x2_det)(const complex double mat[2][2])
{
	return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
}

complex double (mat2x2_tr)(const complex double mat[2][2])
{
	return mat[0][0] + mat[1][1];
}

void (mat2x2_charpol)(complex double coeff[3], const complex double mat[2][2])
{
	coeff[2] = 1.;
	coeff[1] = -mat2x2_tr(mat);
	coeff[0] = mat2x2_det(mat);
}

void (mat2x2_eig)(complex double eval[2], complex double evec[2][2], const complex double mat[2][2])
{
	complex double cpol[3];

	mat2x2_charpol(cpol, mat);
	quadratic_formula(eval, cpol);

	if (0. != mat[1][0]) {

		evec[0][0] = eval[0] - mat[1][1];
		evec[0][1] = mat[1][0];
		evec[1][0] = eval[1] -  mat[1][1];
		evec[1][1] = mat[1][0];

	} else if (0. != mat[0][1]) {

		evec[0][0] = mat[0][1];
		evec[0][1] = eval[0] - mat[0][0];
		evec[1][0] = mat[0][1];
		evec[1][1] = eval[1] - mat[0][0];

	} else {

		evec[0][0] = 1.;
		evec[0][1] = 0.;
		evec[1][0] = 0.;
		evec[1][1] = 1.;
	}
}

