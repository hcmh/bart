/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"
#include "num/rand.h"

#include "iter/iter4.h"

struct iter4_nlop_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;
};
DEF_TYPEID(iter4_nlop_s);

struct iter4_altmin_s {

	INTERFACE(iter_op_data);

	struct nlop_s nlop;

	struct iter3_irgnm_conf conf;

	const float** ref;

	float alpha;

	const struct operator_p_s* prox;
};

DEF_TYPEID(iter4_altmin_s);

static void nlop_for_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	operator_apply_unchecked(nlop->nlop.op, (complex float*)_dst, (const complex float*)_src);
}

static void nlop_der_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_forward_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}

static void nlop_adj_iter(iter_op_data* _o, float* _dst, const float* _src)
{
	const struct iter4_nlop_s* nlop = CAST_DOWN(iter4_nlop_s, _o);

	assert(2 == operator_nr_args(nlop->nlop.op));
	linop_adjoint_unchecked(nlop->nlop.derivative[0], (complex float*)_dst, (const complex float*)_src);
}




void iter4_irgnm(iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	iter3_irgnm(_conf,
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		N, dst, ref, M, src, cb);
}

void iter4_levmar(iter3_conf* _conf,
		 struct nlop_s* nlop,
		 long N, float* dst, const float* ref,
		 long M, const float* src,
		 struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	iter3_levmar(_conf,
		    (struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		    N, dst, ref, M, src, cb);
}

void iter4_irgnm_levmar_hybrid(iter3_conf* _conf,
		 struct nlop_s* nlop,
		 long N, float* dst, const float* ref,
		 long M, const float* src,
		 struct iter_op_s cb)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	assert(M * sizeof(float) == md_calc_size(cd->N, cd->dims) * cd->size);
	assert(N * sizeof(float) == md_calc_size(dm->N, dm->dims) * dm->size);

	iter3_irgnm_levmar_hybrid(_conf,
		    (struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		    (struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		    N, dst, ref, M, src, cb);
}



static void altmin_nlop(iter_op_data* _o, int N, float* args[N])
{
	const struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);

	assert((unsigned int) N == operator_nr_args(nlop->nlop.op));

	nlop_generic_apply_unchecked(&nlop->nlop, N, (void*) args);
}




static void altmin_normal(iter_op_data* _o, float* dst, const float* src, unsigned int i)
{
	const struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);
	const struct linop_s* der = nlop_get_derivative(&nlop->nlop, 0, i);

	linop_normal_unchecked(der, (complex float*) dst, (const complex float*) src);
}

static void altmin_normal_coils(iter_op_data* _o, float* dst, const float* src)
{
	altmin_normal(_o, dst, src, 0);
}

static void altmin_normal_img(iter_op_data* _o, float* dst, const float* src)
{
	altmin_normal(_o, dst, src, 1);
}


static void altmin_normal_img_fista(iter_op_data* _o, float* dst, const float* src)
{
	altmin_normal(_o, dst, src, 1);
	const struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);
	long N = md_calc_size(DIMS, nlop_generic_domain(&nlop->nlop, 1)->dims);
	select_vecops(dst)->axpy(2*N, dst, nlop->alpha, src);
}


static void altmin_prox_fista(iter_op_data* _o, float alpha, float* dst, const float* src)
{
	UNUSED(src);
	assert(src == dst);

	struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);

	const long* dims = nlop_generic_domain(&nlop->nlop, 1)->dims;

	if (NULL != nlop->ref[1]) {

		md_zaxpy(DIMS, dims, (complex float*) dst, 1.f, (const complex float*) nlop->ref[1]);
	}

	operator_p_apply_unchecked(nlop->prox, alpha, (complex float*) dst, (const complex float*) dst);

	if (NULL != nlop->ref[1]) {

		md_zaxpy(DIMS, dims, (complex float*) dst, -1.f, (const complex float*) nlop->ref[1]);
	}
}


static void altmin_inverse(iter_op_data* _o, float alpha, float* dst, const float* src, unsigned int i)
{
	struct iter4_altmin_s* nlop = CAST_DOWN(iter4_altmin_s, _o);

	nlop->alpha = alpha;

	const long* dst_dims = nlop_generic_domain(&nlop->nlop, i)->dims;
	long dst_size = md_calc_size(DIMS, dst_dims);


	float* AHy = md_alloc_sameplace(DIMS, dst_dims, CFL_SIZE, src);
	md_clear(DIMS, dst_dims, AHy, CFL_SIZE);

	linop_adjoint_unchecked(nlop_get_derivative(&nlop->nlop, 0, i), (complex float*) AHy, (const complex float*) src);

	float eps = nlop->conf.cgtol * md_norm(DIMS, dst_dims, AHy);

	if (NULL != nlop->ref[i]) {

		float* AHAx_0 = md_alloc_sameplace(DIMS, dst_dims, CFL_SIZE, src);
		linop_normal_unchecked(nlop_get_derivative(&nlop->nlop, 0, i), (complex float*) AHAx_0, (const complex float*) nlop->ref[i]);
 		md_zaxpy(DIMS, dst_dims, (complex float*) AHy, -1.f, (const complex float*) AHAx_0);
		md_free(AHAx_0);
		md_zaxpy(DIMS, dst_dims, (complex float*) dst, -1.f, (const complex float*) nlop->ref[i]);
	}

	if (1 == i) {
		if (nlop->conf.fista) {

			float* tmp = md_alloc_sameplace(DIMS, dst_dims, CFL_SIZE, src);
			md_gaussian_rand(DIMS, dst_dims, (complex float*) tmp);
			double maxeigen = power(60, 2*dst_size, select_vecops(src), (struct iter_op_s){ altmin_normal_img, _o }, tmp);
			md_free(tmp);

			double step = fmin(iter_fista_defaults.step / maxeigen, iter_fista_defaults.step); // 0.95f is FISTA standard
			debug_printf(DP_INFO, "\tFISTA Stepsize: %.2e\n", step);


			if (nlop->conf.wavelets) {

				bool llr = false;
				if (!llr) {
					// for FISTA
					bool randshift = true;
					long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
					unsigned int wflags = 0;
					for (unsigned int j = 0; j < DIMS; j++) {
						if ((1 < dst_dims[j]) && MD_IS_SET(FFT_FLAGS|SLICE_FLAG, j)) {

							wflags = MD_SET(wflags, j);
							minsize[j] = MIN(dst_dims[j], 16);
						}
					}
					nlop->prox = prox_wavelet_thresh_create(DIMS, dst_dims, wflags, 0L, minsize, 0.5*alpha, randshift);
					nlop->alpha = 0.5*alpha;
				} else {

					unsigned int llr_blk = 8;
					bool randshift = true;
					long blkdims[MAX_LEV][DIMS];
					int levels = llr_blkdims(blkdims, 7, dst_dims, llr_blk);
					assert(1 == levels);
					for(int l = 0; l < levels; l++)
						blkdims[l][MAPS_DIM] = 1;
					int remove_mean = 0;
					nlop->prox = lrthresh_create(dst_dims, randshift, 7, (const long (*)[DIMS])blkdims, 0.5*alpha, false, remove_mean);
					nlop->alpha = 0.5*alpha;
				}

				fista(4*nlop->conf.cgiter, eps, step,
					iter_fista_defaults.continuation, iter_fista_defaults.hogwild,
					2*dst_size, select_vecops(src),
					(struct iter_op_s){ altmin_normal_img_fista, _o },
					(struct iter_op_p_s){ altmin_prox_fista, _o },
					dst, AHy, NULL);
			} else {

				nlop->prox = prox_zero_create(DIMS, dst_dims);

				fista(4*nlop->conf.cgiter, eps, step,
					iter_fista_defaults.continuation, iter_fista_defaults.hogwild,
					2*dst_size, select_vecops(src),
					(struct iter_op_s){ altmin_normal_img_fista, _o },
					OPERATOR_P2ITOP(nlop->prox),
					dst, AHy, NULL);
			}

		} else {
			conjgrad(nlop->conf.cgiter, alpha, eps, 2*dst_size, select_vecops(src),
				(struct iter_op_s){ altmin_normal_img, _o }, dst, AHy, NULL);
		}

	}
	else
		conjgrad(nlop->conf.cgiter, alpha, eps, 2*dst_size, select_vecops(src),
			 (struct iter_op_s){ altmin_normal_coils, _o }, dst, AHy, NULL);

	if (NULL != nlop->ref[i]) {

		md_zaxpy(DIMS, dst_dims, (complex float*) dst, 1.f, (const complex float*) nlop->ref[i]);
	}

	md_free(AHy);

}


static void altmin_inverse_coils(iter_op_data* _o, float alpha, float* dst, const float* src)
{
	debug_printf(DP_DEBUG2, "\t%s\n", __func__);
	altmin_inverse(_o, alpha, dst, src, 0);
}


static void altmin_inverse_img(iter_op_data* _o, float alpha, float* dst, const float* src)
{
	debug_printf(DP_DEBUG2, "\t%s\n", __func__);
	altmin_inverse(_o, alpha, dst, src, 1);
}

void iter4_altmin(iter3_conf* _conf,
		struct nlop_s* nlop,
		long NI, float* dst[NI], const float* ref[NI],
		long M, const float* src,
		struct iter_nlop_s cb)
{

	struct iter3_irgnm_conf* conf = CAST_DOWN(iter3_irgnm_conf, _conf);
	struct iter4_altmin_s data = { { &TYPEID(iter4_altmin_s) }, *nlop, *conf, ref, 0.f, NULL };

	assert( 2 == NI);

	struct iter_op_p_s min_ops[2] = {
		(struct iter_op_p_s){ altmin_inverse_coils, CAST_UP(&data) },
		(struct iter_op_p_s){ altmin_inverse_img, CAST_UP(&data) }
	};

	altmin(conf->iter, conf->alpha, conf->redu,
		M, select_vecops(src),
		NI,
		(struct iter_nlop_s){ altmin_nlop, CAST_UP(&data) },
		min_ops,
		dst, src,
		cb);
}

void iter4_landweber(iter3_conf* _conf,
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src)
{
	struct iter4_nlop_s data = { { &TYPEID(iter4_nlop_s) }, *nlop };

	iter3_landweber(_conf,
		(struct iter_op_s){ nlop_for_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_der_iter, CAST_UP(&data) },
		(struct iter_op_s){ nlop_adj_iter, CAST_UP(&data) },
		N, dst, ref, M, src);
}



