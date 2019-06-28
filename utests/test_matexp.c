/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>

#include "num/matexp.h"

#include "utest.h"



static bool test_matexp(void)
{
	const float a[2][2] = {
		{ 0., 1. },
		{ -1., 0. }, 
	};

	float o[2][2];

	mat_exp(2, M_PI, o, a);

	bool ok = true;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			ok &= (fabsf(o[i][j] - ((i == j) ? -1. : 0.)) < 1.E-5);

	return ok;
}


UT_REGISTER_TEST(test_matexp);

