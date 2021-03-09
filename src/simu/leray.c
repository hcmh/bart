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

#include "simu/fd_geometry.h"
#include "simu/leray.h"
#include "simu/pde_laplace.h"

#include "iter/iter.h"
#include "iter/monitor.h"
#include "linops/fmac.h"
#include "linops/grad.h"
#include "linops/someops.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include <complex.h>

#include "misc/debug.h"

struct leray_s {
	INTERFACE(linop_data_t);
	long N;
	long *dims;
	long *phi_dims;
	struct iter_conjgrad_conf *cg_conf;
	_Complex float *y, *tmp;
	struct linop_s *grad_op, *neg_laplace, *div_op;
	struct boundary_point_s *boundary;
	long n_points;
	struct iter_monitor_s *mon;
	const complex float* src;
	complex float* mask;
};
static DEF_TYPEID(leray_s);



void linop_leray_calc_rhs(const linop_data_t *_data, complex float *y, const complex float *src)
{
	const auto data = CAST_DOWN(leray_s, _data);
	linop_forward(data->div_op, data->N, data->phi_dims, y, data->N, data->dims, src);

			char* str = getenv("DEBUG_LEVEL");
			debug_level = (NULL != str) ? atoi(str) : DP_INFO;
			if (5 <= debug_level)
			dump_cfl("DEBUG_leray_phi", data->N, data->phi_dims, y);

	laplace_neumann_update_rhs(data->N - 1, data->phi_dims + 1, y, data->n_points, data->boundary);
}

void linop_leray_calc_projection(const linop_data_t *_data, complex float *dst, const complex float *tmp)
{
	const auto data = CAST_DOWN(leray_s, _data);
	linop_forward(data->grad_op, data->N, data->dims, dst, data->N, data->phi_dims, tmp);

	long j_strs[data->N], mask_strs[data->N];
	md_calc_strides(data->N, j_strs, data->dims, CFL_SIZE);
	md_calc_strides(data->N, mask_strs, data->phi_dims, CFL_SIZE);

	md_zmul2(data->N, data->dims, j_strs, dst, j_strs, dst, mask_strs, data->mask);
	neumann_set_boundary(data->N, data->dims, 0, dst, data->n_points, data->boundary, dst);

	md_zaxpy(data->N, data->dims, dst, 1., data->src);
}



// dst = src - grad (laplace^(-1) (div (src)))
static void leray_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(leray_s, _data);
	md_clear(data->N, data->phi_dims, data->tmp, CFL_SIZE);

	data->src = src;

	linop_leray_calc_rhs(_data, data->y, src);

	long size = 2 * md_calc_size(data->N, data->phi_dims); // multiply by 2 for float size
	iter_conjgrad(CAST_UP(data->cg_conf), data->neg_laplace->forward, NULL, size, (float *)data->tmp, (const float *)data->y, data->mon);

	linop_leray_calc_projection(_data, dst, data->tmp);
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
	//leray_adjoint_apply(_data, dst, dst);
}



static void leray_free(const linop_data_t *_data)
{
	const auto data = CAST_DOWN(leray_s, _data);

	md_free(data->y);
	md_free(data->tmp);
	md_free(data->boundary);
	md_free(data->mask);
	linop_free(data->neg_laplace);
	linop_free(data->grad_op);
	linop_free(data->div_op);
	xfree(data->dims);
	xfree(data->phi_dims);
	xfree(data->cg_conf);
	xfree(data);
}



struct linop_s *linop_leray_create(const long N, const long dims[N], long vec_dim, const int iter, const float lambda, const complex float *mask, struct iter_monitor_s *mon)
{
	PTR_ALLOC(struct leray_s, data);
	SET_TYPEID(leray_s, data);

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);
	data->N = N;

	data->phi_dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->phi_dims, dims);
	data->phi_dims[vec_dim] = 1;

	PTR_ALLOC(struct iter_conjgrad_conf, cg_conf);
	*cg_conf = iter_conjgrad_defaults;
	data->cg_conf = PTR_PASS(cg_conf);

	data->cg_conf->maxiter = iter;
	data->cg_conf->l2lambda = lambda;

	data->mon = mon;

	data->y = md_alloc(N, data->phi_dims, CFL_SIZE);
	data->tmp = md_alloc(N, data->phi_dims, CFL_SIZE);

	assert(dims[vec_dim] == N - 1);
	assert(vec_dim == 0);
	const long scalar_N = N - 1, *scalar_dims = dims + 1;

	// setup boundary conditions
	data->mask = md_calloc(scalar_N, scalar_dims, CFL_SIZE);
	data->boundary = md_alloc(N, data->phi_dims, sizeof(struct boundary_point_s));
	complex float *normal = md_alloc(N, dims, CFL_SIZE);

	calc_outward_normal(N, dims, normal, vec_dim, data->phi_dims, mask);
	data->n_points = calc_boundary_points(N, dims, data->boundary, vec_dim, normal, NULL);
	shrink_wrap(scalar_N, scalar_dims, data->mask, data->n_points, data->boundary, mask);

	data->neg_laplace = linop_laplace_neumann_create(scalar_N, scalar_dims, mask, data->n_points, data->boundary);

	data->grad_op = linop_fd_create(N, data->phi_dims, vec_dim, ((MD_BIT(N) - 1) & ~MD_BIT(vec_dim)), 2, BC_ZERO, false);

	auto div_op = linop_fd_create(N, data->phi_dims, vec_dim, ((MD_BIT(N) - 1) & ~MD_BIT(vec_dim)), 2, BC_ZERO, true);
	auto div_mask_op = linop_cdiag_create(N, data->phi_dims, ((MD_BIT(N) - 1) & ~MD_BIT(vec_dim)), data->mask);
	data->div_op = linop_chain(linop_get_adjoint(div_op), div_mask_op);


	md_free(normal);
	linop_free(div_mask_op);
	linop_free(div_op);


	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), leray_apply, leray_adjoint_apply, leray_normal_apply, NULL, leray_free);
}


/**
 * Data for computing prox_indicator_fun:
 * Proximal function for f(z) = 1{ set }
 * Solution is z = Projection of x onto the set
 *
 * @param op Projection operator
 */
struct prox_indicator_data {

	INTERFACE(operator_data_t);

	const struct linop_s *op;
};

static DEF_TYPEID(prox_indicator_data);

static void prox_indicator_apply(const operator_data_t *_data, float mu, complex float *dst, const complex float *src)
{
	UNUSED(mu);
	auto pdata = CAST_DOWN(prox_indicator_data, _data);

	const struct linop_s *op = pdata->op;
	linop_forward(op, linop_domain(op)->N, linop_domain(op)->dims, dst, linop_codomain(op)->N, linop_codomain(op)->dims, src);
}

static void prox_indicator_del(const operator_data_t *_data)
{
	auto pdata = CAST_DOWN(prox_indicator_data, _data);
	xfree(pdata);
}

const struct operator_p_s *prox_indicator_create(const struct linop_s *op)
{
	PTR_ALLOC(struct prox_indicator_data, pdata);
	SET_TYPEID(prox_indicator_data, pdata);

	unsigned int N = linop_domain(op)->N;
	const long *dims = linop_domain(op)->dims;

	pdata->op = op;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(pdata)), prox_indicator_apply, prox_indicator_del);
}
