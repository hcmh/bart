/*
 * Authors:
 * 2020 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 *
 * Projection of a vector field onto divergence free subspace
 */

#include <assert.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "num/vec3.h"

#include "misc/io.h"
#include "misc/mmio.h"
#include <stdio.h>

#include "simu/leray.h"

#include "linops/fmac.h"
#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include <complex.h>
#include "iter/lsqr.h"

#include "misc/debug.h"

struct leray_s
{
	INTERFACE(linop_data_t);
	long N;
	long *dims;
	long *phi_dims;
	struct lsqr_conf *lconf;
	struct iter_conjgrad_conf *cg_conf;
	complex float *y, *tmp;
	struct linop_s *grad_op, *neg_laplace;
};

static DEF_TYPEID(leray_s);



static void leray_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(leray_s, _data);
	md_clear(data->N, data->phi_dims, data->tmp, CFL_SIZE);

	linop_adjoint(data->grad_op, data->N, data->phi_dims, data->y, data->N, data->dims, src);

	lsqr2(data->N, data->lconf, iter2_conjgrad, CAST_UP(data->cg_conf), data->neg_laplace, 0, NULL, NULL, data->phi_dims, data->tmp, data->phi_dims, data->y, NULL, NULL);

	linop_forward(data->grad_op, data->N, data->dims, dst, data->N, data->phi_dims, data->tmp);
	md_zsmul(data->N, data->dims, dst, dst, -1.);

	// dst = src - grad (laplace^(-1) (div (src)))
	md_zaxpy(data->N, data->dims, dst, 1., src);
}

// (this!) projection is self-adjoint
static void leray_adjoint_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	leray_apply(_data, dst, src);
}

// projections are idempotent
static void leray_normal_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	leray_apply(_data, dst, src);
}



static void leray_free(const linop_data_t *_data)
{
	const auto data = CAST_DOWN(leray_s, _data);

	md_free(data->y);
	md_free(data->tmp);
	linop_free(data->neg_laplace);
	linop_free(data->grad_op);
	xfree(data->dims);
	xfree(data->phi_dims);
	xfree(data->lconf);
	xfree(data->cg_conf);
	xfree(data);
}



struct linop_s *linop_leray_create(const long N, const long dims[N], long vec_dim, const long flags, const unsigned int order, const enum BOUNDARY_CONDITION bc, const int iter, const float lambda, const complex float* mask)
{
	PTR_ALLOC(struct leray_s, data);
	SET_TYPEID(leray_s, data);

	long dims2[N];
	md_copy_dims(N, dims2, dims);

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);
	data->N = N;

	data->phi_dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->phi_dims, dims);
	data->phi_dims[vec_dim] = 1;

	PTR_ALLOC(struct lsqr_conf, lconf);
	*lconf = lsqr_defaults;
	data->lconf = PTR_PASS(lconf);

	PTR_ALLOC(struct iter_conjgrad_conf, cg_conf);
	*cg_conf = iter_conjgrad_defaults;
	data->cg_conf = PTR_PASS(cg_conf);


	data->cg_conf->maxiter = iter;
	data->cg_conf->l2lambda = lambda;

	data->y = md_alloc(N, data->phi_dims, CFL_SIZE);
	data->tmp = md_alloc(N, data->phi_dims, CFL_SIZE);

	assert(dims[vec_dim] == bitcount(flags));

	if (NULL != mask) {
		auto grad_op = linop_fd_create(N, data->phi_dims, vec_dim, flags, order, bc, false);
		auto mask_op = linop_cdiag_create(N, data->dims, MD_BIT(N + 1) - 1, mask);
		data->grad_op = linop_chain(grad_op, mask_op);
		data->neg_laplace = linop_chain(data->grad_op, linop_get_adjoint(grad_op));
		linop_free(mask_op);
		linop_free(grad_op);
	} else {
		data->grad_op = linop_fd_create(N, data->phi_dims, vec_dim, flags, order, bc, false);
		data->neg_laplace = linop_get_normal(data->grad_op);
	}

	return linop_create(N, dims2, N, dims, CAST_UP(PTR_PASS(data)), leray_apply, leray_adjoint_apply, leray_normal_apply, NULL, leray_free);

}
