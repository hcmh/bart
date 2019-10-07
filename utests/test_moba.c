/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker, Nick Scholand
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

#include "moba/T1fun.h"
#include "moba/model_Bloch.h"
#include "moba/blochfun.h"
#include "moba/T1MOLLI_1.h"


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

static void random_application(const struct nlop_s* nlop)
{
	auto dom = nlop_domain(nlop);
	auto cod = nlop_codomain(nlop);

	complex float* in = md_alloc(dom->N, dom->dims, dom->size);
	complex float* dst = md_alloc(cod->N, cod->dims, cod->size);

	md_gaussian_rand(dom->N, dom->dims, in);

	// define position for derivatives
	nlop_apply(nlop, cod->N, cod->dims, dst, dom->N, dom->dims, in);

	md_free(in);
	md_free(dst);
}


static bool test_T1relax(void)
{
	
        
        enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
// 	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        
        long map_strs[N];
	md_calc_strides(N, map_strs, map_dims, CFL_SIZE);
        
        long out_strs[N];
	md_calc_strides(N, out_strs, out_dims, CFL_SIZE);
        
        long TI_strs[N];
	md_calc_strides(N, TI_strs, TI_dims, CFL_SIZE);

	complex float* dst1 = md_alloc(N, out_dims, CFL_SIZE);
	complex float* dst2 = md_alloc(N, out_dims, CFL_SIZE);
        complex float* dst3 = md_alloc(N, map_dims, CFL_SIZE);
        complex float* dst4 = md_alloc(N, map_dims, CFL_SIZE);
	complex float* src1 = md_alloc(N, map_dims, CFL_SIZE);
        complex float* src2 = md_alloc(N, map_dims, CFL_SIZE);
        complex float* src3 = md_alloc(N, map_dims, CFL_SIZE);
        complex float* tmp = md_alloc(N, map_dims, CFL_SIZE);
        complex float* TI1 = md_alloc(N, TI_dims, CFL_SIZE);
        
        complex float TI[4] = { 0., 1., 2., 3. };
        
        TI1 = md_alloc(N, TI_dims, CFL_SIZE);
                
        md_copy(N, TI_dims, TI1, TI, CFL_SIZE);

	md_gaussian_rand(N, map_dims, src1);
        md_gaussian_rand(N, map_dims, src2);
        md_gaussian_rand(N, map_dims, src3);

	struct nlop_s* T1 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI);
        
        md_zsmul(N, map_dims, tmp, src3, -1.0);
        md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, TI_strs, TI);
        md_zexp(N, out_dims, dst1, dst1);
     
        
        md_zsub(N, map_dims, tmp, src1, src2);        
        md_zmul2(N, out_dims, out_strs, dst1, map_strs, tmp, out_strs, dst1);
        md_zadd2(N, out_dims, out_strs, dst1, map_strs, src2, out_strs, dst1);
        
        long pos[N];

	for (int i = 0; i < N; i++)
		pos[i] = 0;
        
        pos[TE_DIM] = TI_dims[TE_DIM] - 1;
        md_copy_block(N, pos, map_dims, dst4, out_dims, dst1, CFL_SIZE);
        
                
        nlop_generic_apply_unchecked(T1, 5, (void*[]){ dst2, dst3, src1, src2, src3 });
//         
	double err = md_znrmse(N, out_dims, dst2, dst1) + md_znrmse(N, map_dims, dst4, dst3);
        
//         nlop_generic_apply_unchecked(T1, 4, (void*[]){ dst2, src1, src2, src3 });
        
// 	double err = md_znrmse(N, map_dims, dst2, dst4);

	nlop_free(T1);

        md_free(src1);
	md_free(src2);
        md_free(src3);
        md_free(tmp);
        md_free(TI1);
	md_free(dst1);
	md_free(dst2);
        md_free(dst3);
        md_free(dst4);
        
        
	UT_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_T1relax);



static bool test_nlop_T1relax_der_adj(void)
{
        enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
// 	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long TI_dims[N] = { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        
        complex float TI[4] = { 0., 1., 2., 3. };

	struct nlop_s* T1 = nlop_T1relax_create(N, map_dims, out_dims, TI_dims, TI);

	struct nlop_s* flat = nlop_flatten(T1);

	random_application(flat);

	double err = linop_test_adjoint(nlop_get_derivative(flat, 0, 0));

	nlop_free(flat);
	nlop_free(T1);

	UT_ASSERT((!safe_isnanf(err)) && (err < 6.E-2));
}

UT_REGISTER_TEST(test_nlop_T1relax_der_adj);



static bool test_nlop_Blochfun(void) 
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 500, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long input_dims[N];
	
	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);
	
	bool gpu_use = false;
	
	struct modBlochFit fitPara = modBlochFit_defaults;
	
	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* op_Bloch = nlop_Bloch_create(N, map_dims, out_dims, in_dims, input_dims, &fitPara, gpu_use);

	nlop_apply(op_Bloch, N, out_dims, dst, N, in_dims, src);
	
	float err = linop_test_adjoint(nlop_get_derivative(op_Bloch, 0, 0));

	nlop_free(op_Bloch);

	md_free(src);
	md_free(dst);	
	
	UT_ASSERT(err < 1.E-3);
}

UT_REGISTER_TEST(test_nlop_Blochfun);

