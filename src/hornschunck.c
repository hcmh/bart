/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/init.h"
#include "num/filter.h"

#include "linops/linop.h"
#include "linops/grad.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/admm.h"
#include "iter/prox.h"
#include "iter/thresh.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<lambda> <flags> <tdim> <input> <output>";
static const char help_str[] = "Estimate optical flow along dims <flags>.\n";

	
int main_hornschunck(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 5, usage_str, help_str);

	num_init();

	long dims[DIMS];

	float lambda = atof(argv[1]);
	int flags = atoi(argv[2]);
	int tdim = atoi(argv[3]);

	assert((0 <= tdim) && (tdim < DIMS));
	assert(!MD_IS_SET(flags, tdim));
	
	complex float* in_data = load_cfl(argv[4], DIMS, dims);


	const struct linop_s* grad_op = linop_grad_create(DIMS, dims, flags);

	long odims[DIMS + 1];
	md_copy_dims(DIMS + 1, odims, linop_codomain(grad_op)->dims);
#if 0
	complex float* grad = md_alloc(DIMS + 1, odims, CFL_SIZE);

	linop_forward(grad_op, DIMS + 1, odims, grad, DIMS, dims, in_data);
#else
	// ADD A dummy dimension with ones
	odims[DIMS]++;
	complex float* grad = md_alloc(DIMS + 1, odims, CFL_SIZE);
	md_zfill(DIMS + 1, odims, grad, 10.);
	odims[DIMS]--;
	linop_forward(grad_op, DIMS + 1, odims, grad, DIMS, dims, in_data);
	odims[DIMS]++;
#endif
	linop_free(grad_op);

	//complex float* tdiff = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* tdiff = create_cfl("tdiff", DIMS, dims);

	const struct linop_s* time_op = linop_grad_create(DIMS, dims, MD_BIT(tdim));
	linop_forward(time_op, DIMS + 1, linop_codomain(time_op)->dims, tdiff, DIMS, dims, in_data);
	linop_free(time_op);


#if 1
	long flags2 = flags | MD_BIT(tdim);

	long wdims[DIMS + 1];
	md_select_dims(DIMS + 1, flags2 | MD_BIT(DIMS), wdims, odims);
	complex float* wdiag = md_alloc(DIMS + 1, wdims, CFL_SIZE);
	md_zfill(DIMS + 1, wdims, wdiag, 0.1);
#if 1
	wdims[DIMS]--;

	float scal[DIMS + 1];
	for (int j = 0; j < DIMS + 1; j++)
		scal[j] = 1. / (float)wdims[j];

	scal[tdim] = 0.1 / (float)wdims[tdim];

	klaplace_scaled(DIMS + 1, wdims, flags2, scal, wdiag);
	md_zsmul(DIMS + 1, wdims, wdiag, wdiag, 20.);
	md_zsadd(DIMS + 1, wdims, wdiag, wdiag, 1.);
	md_zspow(DIMS + 1, wdims, wdiag, wdiag, -4.);	// 1 + 222. \Laplace^16
	wdims[DIMS]++;
#endif
	// dump_cfl("wdiag", DIMS + 1, wdims, wdiag);

	const struct linop_s* diag = linop_cdiag_create(DIMS + 1, odims, flags2 | MD_BIT(DIMS), wdiag);
	const struct linop_s* fft = linop_fftc_create(DIMS + 1, odims, flags2);
	const struct linop_s* reg_op = linop_chain(diag, fft);
#else
	const struct linop_s* reg_op = linop_grad_create(DIMS + 1, odims, flags /* | MD_BIT(tdim) */);
	assert(DIMS + 2 == linop_codomain(reg_op)->N);
#endif



	const struct linop_s* frw = linop_fmac_create(DIMS + 1, odims, MD_BIT(DIMS), 0u, 0u, grad);

	long dims2[DIMS + 1];
	md_copy_dims(DIMS, dims2, dims);
	dims2[DIMS] = 1;

	complex float* out_data = create_cfl(argv[5], DIMS + 1, odims);
	md_clear(DIMS + 1, odims, out_data, CFL_SIZE);


#if 1
	const struct linop_s* nfrw = linop_chain(reg_op, frw);	// PRECONDITION
#endif


	complex float* rhs = md_alloc(DIMS + 1, odims, CFL_SIZE);

	linop_adjoint(nfrw, DIMS + 1, odims, rhs, DIMS + 1, dims2, tdiff);

	// dump_cfl("rhs", DIMS + 1, odims, rhs);

//	linop_free(frw);




	const struct operator_s* nrm = nfrw->normal; //operator_chain(frw->forward, frw->adjoint);
#if 0
#if 0
	const struct operator_p_s* thresh_prox = prox_thresh_create(linop_codomain(reg_op)->N, linop_codomain(reg_op)->dims, 
								lambda, MD_BIT(DIMS) | MD_BIT(DIMS + 1), false);
#else
	const struct operator_p_s* l2 = prox_l2norm_create(linop_codomain(reg_op)->N, linop_codomain(reg_op)->dims, lambda);
#endif



//	linop_clone(frw);
	struct iter_admm_conf conf = iter_admm_defaults;

	conf.maxiter = 100;
//	conf.rho = .1;


	iter2_admm(CAST_UP(&conf), nrm,
		  //1, MAKE_ARRAY(thresh_prox), MAKE_ARRAY(reg_op), NULL,
		  1, MAKE_ARRAY(l2), MAKE_ARRAY(reg_op), NULL,
		  NULL, 2 * md_calc_size(DIMS + 1, odims), (float*)out_data, (const float*)rhs, NULL);
#else
	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.l2lambda = lambda;

	iter2_conjgrad(CAST_UP(&conf), nrm,
		   0, NULL, NULL, NULL,
		   NULL, 2 * md_calc_size(DIMS + 1, odims), (float*)out_data, (const float*)rhs, NULL);


	// PRECOND
	linop_forward(reg_op, DIMS + 1, odims, out_data, DIMS + 1, odims, out_data);




#endif
#if 0
	lsqr2(	unsigned int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			unsigned int num_funs,
			const struct operator_p_s* prox_funs[__VLA2(num_funs)],
			const struct linop_s* prox_linops[__VLA2(num_funs)],
			const long x_dims[__VLA(N)], _Complex float* x,
			const long y_dims[__VLA(N)], const _Complex float* y,
			const struct operator_s* precond_op,
			struct iter_monitor_s* monitor);
#endif

#if 1
	// compute residual
	complex float* res = create_cfl("hs_res", DIMS + 1, dims2);
	linop_forward(frw, DIMS + 1, dims2, res, DIMS + 1, odims, out_data);
	md_zsub(DIMS + 1, dims2, res, res, tdiff);
	unmap_cfl(DIMS + 1, dims2, res);
#endif
	
	unmap_cfl(DIMS, dims, in_data);
	unmap_cfl(DIMS + 1, odims, out_data);
	exit(0);
}


