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
#include "nlops/chain.h"
#include "nlops/nltest.h"

#include "linops/someops.h"

#include "utest.h"








static bool test_nlop_chain(void)
{
	enum { N = 3 };
	long dims[N] = { 10, 7, 3 };

	complex float val = 2.;
	struct nlop_s* diag = nlop_from_linop(linop_cdiag_create(N, dims, 0, &val));
	struct nlop_s* zexp = nlop_zexp_create(N, dims);
	struct nlop_s* zexp2 = nlop_chain(zexp, diag);

	double err = nlop_test_derivative(zexp2);

	nlop_free(zexp2);
	nlop_free(zexp);

	return (err < 1.E-2);
}



UT_REGISTER_TEST(test_nlop_chain);


