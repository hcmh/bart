/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/nlop.h"

#include "mdb/T1fun.h"
#include "mdb/model_Bloch.h"
#include "mdb/blochfun.h"

#include "utest.h"






static bool test_nlop_T1fun(void) 
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	complex float TI[4] = { 0., 1., 2., 3. };

	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* T1 = nlop_T1_create(N, map_dims, out_dims, in_dims, TI_dims, TI, false);

	nlop_apply(T1, N, out_dims, dst, N, in_dims, src);
	
	float err = linop_test_adjoint(nlop_get_derivative(T1, 0, 0));

	nlop_free(T1);

	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_T1fun);


static bool test_nlop_Blochfun(void) 
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 500, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long input_dims[N];
	
	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);
	complex float* input_sp = NULL;
	complex float* input_img = NULL;
	
	bool gpu_use = false;
	
	struct modBlochFit fitPara = modBlochFit_defaults;
	
	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* op_Bloch = nlop_Bloch_create(N, map_dims, out_dims, in_dims, input_dims, input_img, input_sp, &fitPara, gpu_use);

	nlop_apply(op_Bloch, N, out_dims, dst, N, in_dims, src);
	
	float err = linop_test_adjoint(nlop_get_derivative(op_Bloch, 0, 0));

	nlop_free(op_Bloch);

	md_free(src);
	md_free(dst);	
	
	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_Blochfun);

