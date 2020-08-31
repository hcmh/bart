
#include <complex.h>

#include "misc/misc.h"

#include "num/mat2x2.h"

#include "utest.h"


static bool cdbl_eq(complex double a, complex double b)
{
	return (1.E-14 > cabs(a - b));
}


static bool test_mat2x2_eig(const complex double mat[2][2])
{
	complex double eval[2];
	complex double evec[2][2];

	mat2x2_eig(eval, evec, mat);

	complex double out[2][2];
	mat2x2_vecmul(out[0], mat, evec[0]);
	mat2x2_vecmul(out[1], mat, evec[1]);

	return     cdbl_eq(out[0][0], eval[0] * evec[0][0])
		&& cdbl_eq(out[0][1], eval[0] * evec[0][1])
		&& cdbl_eq(out[1][0], eval[1] * evec[1][0])
		&& cdbl_eq(out[1][1], eval[1] * evec[1][1]);
}

static bool test_mat2x2_eig0(void)
{
	const complex double mat[2][2] = { { 1., 2. }, { 3., 4. } };
	
	return test_mat2x2_eig(mat);
}

static bool test_mat2x2_eig1(void)
{
	const complex double mat[2][2] = { { 1., 0. }, { 3., 4. } };
	
	return test_mat2x2_eig(mat);
}

static bool test_mat2x2_eig2(void)
{
	const complex double mat[2][2] = { { 1., 2. }, { 0., 4. } };
	
	return test_mat2x2_eig(mat);
}

UT_REGISTER_TEST(test_mat2x2_eig0);
UT_REGISTER_TEST(test_mat2x2_eig1);
UT_REGISTER_TEST(test_mat2x2_eig2);


