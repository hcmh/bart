/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"

#include "linops/someops.h"

#include "utest.h"








static bool test_nlop_cast_pos(void)
{
	bool ok = true;
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct linop_s* l = linop_identity_create(N, dims);
	struct nlop_s* d = nlop_from_linop(l);

	if (l == linop_from_nlop(d)) // maybe just require != NULL ?
		ok = false;

	linop_free(l);
	nlop_free(d);

	return ok;
}



UT_REGISTER_TEST(test_nlop_cast_pos);



static bool test_nlop_cast_neg(void)
{
	bool ok = true;
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	struct nlop_s* d = nlop_zexp_create(N, dims);

	if (NULL != linop_from_nlop(d))
		ok = false;

	nlop_free(d);

	return ok;
}



UT_REGISTER_TEST(test_nlop_cast_neg);


