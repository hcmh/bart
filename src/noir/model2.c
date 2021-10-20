/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2020 Martin Uecker
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */


#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "noncart/nufft.h"

#include "nlops/nlop.h"
#include "nlops/tenmul.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/fft.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/ops.h"

#include "model2.h"



struct noir2_model_conf_s noir2_model_conf_defaults = {

	.fft_flags_noncart = FFT_FLAGS,
	.fft_flags_cart = 0u,

	.rvc = false,
	.sos = false,
	.a = 220.,
	.b = 32.,

	.nufft_conf = NULL,
};


static void noir2_calc_weights(const struct noir2_model_conf_s* conf, int N, const long wght_dims[N], complex float* dst)
{
	unsigned int flags = md_nontriv_dims(N, wght_dims);

	long dims_sc[N];
	md_copy_dims(N, dims_sc, wght_dims);

	if (conf->sos && (dims_sc[READ_DIM] > 1)) {

		assert(SLICE_DIM < N);
		dims_sc[SLICE_DIM] = dims_sc[READ_DIM] * 0.5;	// for reasonable smoothness in z-direction
	}

	klaplace(N, wght_dims, dims_sc, flags, dst);
	md_zsmul(N, wght_dims, dst, dst, conf->a);
	md_zsadd(N, wght_dims, dst, dst, 1.);
	md_zspow(N, wght_dims, dst, dst, -conf->b / 2.);	// 1 + 220. \Laplace^16
}

static struct noir2_s noir2_init_create(int N,
					const long pat_dims[N],
					const long bas_dims[N],
					const long msk_dims[N],
					const long ksp_dims[N],
					const long cim_dims[N],
					const long img_dims[N],
					const long col_dims[N],
					const long trj_dims[N])
{
	struct noir2_s ret = {
		.nlop = NULL,
		.lop_coil2 = NULL,
		.lop_fft = NULL,
		.lop_coil = NULL,
		.lop_im = NULL,
		.tenmul = NULL,

		.lop_nufft = NULL,
		.lop_pattern = NULL,

		.N = N,
		.pat_dims = *TYPE_ALLOC(long[N]),
		.bas_dims = *TYPE_ALLOC(long[N]),
		.msk_dims = *TYPE_ALLOC(long[N]),
		.ksp_dims = *TYPE_ALLOC(long[N]),
		.cim_dims = *TYPE_ALLOC(long[N]),
		.img_dims = *TYPE_ALLOC(long[N]),
		.col_dims = *TYPE_ALLOC(long[N]),
		.trj_dims = *TYPE_ALLOC(long[N]),
	};

	md_copy_dims(N, ret.pat_dims, (NULL == pat_dims) ? MD_SINGLETON_DIMS(N) : pat_dims);
	md_copy_dims(N, ret.bas_dims, (NULL == bas_dims) ? MD_SINGLETON_DIMS(N) : bas_dims);
	md_copy_dims(N, ret.msk_dims, (NULL == msk_dims) ? MD_SINGLETON_DIMS(N) : msk_dims);
	md_copy_dims(N, ret.ksp_dims, (NULL == ksp_dims) ? MD_SINGLETON_DIMS(N) : ksp_dims);
	md_copy_dims(N, ret.cim_dims, (NULL == cim_dims) ? MD_SINGLETON_DIMS(N) : cim_dims);
	md_copy_dims(N, ret.img_dims, (NULL == img_dims) ? MD_SINGLETON_DIMS(N) : img_dims);
	md_copy_dims(N, ret.col_dims, (NULL == col_dims) ? MD_SINGLETON_DIMS(N) : col_dims);
	md_copy_dims(N, ret.trj_dims, (NULL == trj_dims) ? MD_SINGLETON_DIMS(N) : trj_dims);

	return ret;
}

/**
 * This function creates the non (bi)-linear part of the sense model, i.e.
 * 	cim = (mask * img) * ifftuc[weights * ksens]
 * - if cim_dims does not equal img_dims or col_dims, we include a resize (oversampling for coils)
 * - if mask is provided, the dimensioons shoul match img_dims or be trivial
 * - if conf->rvc, only the real part of img is considered
 * lop_coil2 contains the coil transform without a second ifftmod, i.e.
 *	lop_coil2(x) = fftmod(ifftuc(weights * x))
 **/
static void noir2_nlop_create(struct noir2_s* ret, int N, const long msk_dims[N], const complex float* mask, const struct noir2_model_conf_s* conf)
{

	unsigned long fft_flags = conf->fft_flags_noncart | conf->fft_flags_cart;

	long wght_dims[N];
	md_select_dims(N, fft_flags, wght_dims, ret->col_dims);

	complex float* wghts = md_alloc(N, wght_dims, CFL_SIZE);
	noir2_calc_weights(conf, N, wght_dims, wghts);
	ifftmod(N, wght_dims, fft_flags, wghts, wghts);		//in the previous implementation fftmod was used, but ifftmod should be correct
	fftscale(N, wght_dims, fft_flags, wghts, wghts);

	const struct linop_s* lop_wghts = linop_cdiag_create(N, ret->col_dims, fft_flags, wghts);
	const struct linop_s* lop_wghts_ifft = linop_ifft_create(N, ret->col_dims, fft_flags);

	ret->lop_coil =  linop_chain_FF(lop_wghts, lop_wghts_ifft);
	ret->lop_coil2 = linop_clone(ret->lop_coil);


	complex float* ifmod = md_alloc(N, wght_dims, CFL_SIZE);
	md_zfill(N, wght_dims, ifmod, 1.);
	ifftmod(N, wght_dims, fft_flags, ifmod, ifmod);

	//FIXME: In the Cartesian case, we might merge ifftmod with fftmod of fft in linop trafo part (as in previous implementation)
	//CAVEAT: if coil dims != img dims, we need to check if they cancel
	ret->lop_coil = linop_chain_FF(ret->lop_coil, linop_cdiag_create(N, ret->col_dims, fft_flags, ifmod));
	md_free(ifmod);

	long max_dims[N]; 	//of tenmul
	for (int i = 0; i < N; i++)
		max_dims[i] = (1 == ret->cim_dims[i]) ? MIN(ret->img_dims[i], ret->col_dims[i]) : ret->cim_dims[i];

	long img_dims2[N];
	long col_dims2[N];
	md_select_dims(N, md_nontriv_dims(N, ret->img_dims), img_dims2, max_dims);
	md_select_dims(N, md_nontriv_dims(N, ret->col_dims), col_dims2, max_dims);

	if (!md_check_equal_dims(N, ret->col_dims, col_dims2, ~0))
		ret->lop_coil = linop_chain_FF(ret->lop_coil, linop_resize_center_create(N, col_dims2, ret->col_dims));

	if (NULL != mask) {

		assert(md_check_equal_dims(N, msk_dims, ret->img_dims, md_nontriv_dims(N, msk_dims)));
		ret->lop_im = linop_cdiag_create(N, ret->img_dims, md_nontriv_dims(N, msk_dims), mask);
	}

	if (!md_check_equal_dims(N, ret->img_dims, img_dims2, ~0)) {

		if (NULL == mask)
			ret->lop_im  = linop_resize_center_create(N, ret->img_dims, img_dims2);
		else
			ret->lop_im  = linop_chain_FF(ret->lop_im, linop_resize_center_create(N, ret->img_dims, img_dims2));
	}

	if (conf->rvc) {

		if (NULL == mask)
			ret->lop_im  = linop_zreal_create(N, ret->img_dims);
		else
			ret->lop_im  = linop_chain_FF(linop_zreal_create(N, ret->img_dims), ret->lop_im );
	}

	ret->tenmul = nlop_tenmul_create(N, ret->cim_dims, img_dims2, col_dims2);
	ret->nlop = nlop_chain2_FF(nlop_from_linop(ret->lop_coil), 0, nlop_clone(ret->tenmul), 1);
	if (NULL != ret->lop_im )
		ret->nlop = nlop_chain2_swap_FF(nlop_from_linop(ret->lop_im ),0, ret->nlop, 0);
	else
		ret->lop_im = linop_identity_create(N, img_dims2);

	ret->model_conf = *conf;
}

/**
 * This function creates the non (bi)-linear part of the sense model, i.e.
 * 	cim = (mask * img) * ifftuc[weights * ksens]
 * and the corresponding nufft to transform to nonuniform kspace:
 *	frw = nufft(x)
 *	adj = nufft^H(x)
 * 	nrm = nufft^H(nufft(x))
 **/
struct noir2_s noir2_noncart_create(int N,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,	//for nufft
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf)
{
	assert(conf->noncart);

	struct noir2_s ret = noir2_init_create(N, wgh_dims, bas_dims, msk_dims, ksp_dims, cim_dims, img_dims, col_dims, trj_dims);
	noir2_nlop_create(&ret, N, msk_dims, mask, conf);

	struct nufft_conf_s nufft_conf = *(conf->nufft_conf);
	nufft_conf.flags = conf->fft_flags_noncart;
	nufft_conf.cfft = conf->fft_flags_cart;

	ret.lop_fft = nufft_create2(N, ksp_dims, cim_dims, trj_dims, traj, weights ? wgh_dims : NULL, weights, basis ? bas_dims : NULL, basis, nufft_conf);
	ret.lop_nufft = linop_clone(ret.lop_fft);

	// We need to add fftmod and scale for the uncenterd cartesian fft (SMS/SOS)
	if (0 != conf->fft_flags_cart) {

		long fft_dims[N];
		md_select_dims(N, conf->fft_flags_cart, fft_dims, cim_dims);
		complex float* fmod = md_alloc(N, fft_dims, CFL_SIZE);
		md_zfill(N, fft_dims, fmod, 1);
		fftmod(N, fft_dims, conf->fft_flags_cart, fmod, fmod);

		const struct linop_s* lop_fftmod = linop_cdiag_create(N, ksp_dims, conf->fft_flags_cart, fmod);

		//manual chain lop_fftmod to keep normal of nufft
		const struct operator_s* frw = operator_chain(ret.lop_fft->forward, lop_fftmod->forward);
		const struct operator_s* adj = operator_chain(lop_fftmod->adjoint, ret.lop_fft->adjoint);
		const struct linop_s* lop_fft = linop_from_ops(frw, adj, ret.lop_fft->normal, NULL);
		operator_free(frw);
		operator_free(adj);

		linop_free(ret.lop_fft);
		ret.lop_fft = lop_fft;


		linop_free(lop_fftmod);

		fftscale(N, fft_dims, conf->fft_flags_cart, fmod, fmod);
		ret.lop_fft = linop_chain_FF(linop_cdiag_create(N, cim_dims, conf->fft_flags_cart, fmod), ret.lop_fft);

		md_free(fmod);
	}

	debug_printf(DP_DEBUG1, "\nModel created (non Cartesian, nufft-based):\n");
	debug_printf(DP_DEBUG1, "kspace:     ");
	debug_print_dims(DP_DEBUG1, N, ksp_dims);
	debug_printf(DP_DEBUG1, "images:     ");
	debug_print_dims(DP_DEBUG1, N, img_dims);
	debug_printf(DP_DEBUG1, "coils:      ");
	debug_print_dims(DP_DEBUG1, N, col_dims);
	debug_printf(DP_DEBUG1, "trajectory: ");
	debug_print_dims(DP_DEBUG1, N, wgh_dims);
	debug_printf(DP_DEBUG1, "pattern:    ");
	debug_print_dims(DP_DEBUG1, N, wgh_dims);
	if(NULL != basis) {

		debug_printf(DP_DEBUG1, "basis:      ");
		debug_print_dims(DP_DEBUG1, N, bas_dims);
	}
	if(NULL != mask) {

		debug_printf(DP_DEBUG1, "mask:       ");
		debug_print_dims(DP_DEBUG1, N, msk_dims);
	}
	debug_printf(DP_DEBUG1, "coilimg:    ");
	debug_print_dims(DP_DEBUG1, N, cim_dims);
	debug_printf(DP_DEBUG1, "\n");

	return ret;
}

/**
 * This function creates the non (bi)-linear part of the sense model, i.e.
 * 	cim = (mask * img) * ifftuc[weights * ksens]
 * and the corresponding linear transform to map cim to (gridded) kspace:
 * Cartesian case:
 * 	frw = pattern * fftuc(x)
 * 	adj = ifftuc(pattern^* * x)
 * 	nrm = ifftuc(|pattern|^2 * fftuc(x))
 * Non-Cartesian case:
 * 	frw = psf * fftuc(x)
 * 	adj = ifftuc(x)
 * 	nrm = ifftuc(psf * fftuc(x))
 **/
struct noir2_s noir2_cart_create(int N,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf)
{
	struct noir2_s ret = noir2_init_create(N, pat_dims, bas_dims, msk_dims, ksp_dims, cim_dims, img_dims, col_dims, NULL);
	noir2_nlop_create(&ret, N, msk_dims, mask, conf);

	unsigned long fft_flags = conf->fft_flags_noncart | conf->fft_flags_cart;

	assert(md_check_equal_dims(N, pat_dims, ksp_dims, md_nontriv_dims(N, pat_dims)));

	// if noncart, the pattern is understood as psf.
	// the forward model maps to the gridded kspace, while the adjoint does not contain the gridding
	if (!conf->noncart) {

		if (NULL != basis) {

			ret.lop_fft = linop_fftc_create(N, cim_dims, fft_flags);

			long max_dims[N];
			md_max_dims(N, ~0, max_dims, cim_dims, ksp_dims);

			debug_print_dims(DP_INFO, N, max_dims);
			debug_print_dims(DP_INFO, N, bas_dims);

			assert(md_check_equal_dims(N, max_dims, cim_dims, md_nontriv_dims(N, cim_dims)));
			assert(md_check_equal_dims(N, max_dims, ksp_dims, md_nontriv_dims(N, ksp_dims)));
			assert(md_check_equal_dims(N, max_dims, bas_dims, md_nontriv_dims(N, bas_dims)));

			ret.lop_fft = linop_chain_FF(ret.lop_fft, linop_fmac_create(N, max_dims, ~md_nontriv_dims(N, ksp_dims), ~md_nontriv_dims(N, cim_dims), ~md_nontriv_dims(N, bas_dims), basis));
			ret.lop_fft = linop_chain_FF(ret.lop_fft, linop_cdiag_create(N, ksp_dims, md_nontriv_dims(N, pat_dims), pattern));

		} else {

			assert(md_check_equal_dims(N, cim_dims, ksp_dims, ~0));
			ret.lop_fft = linop_fftc_create(N, cim_dims, fft_flags);

			ret.lop_pattern = linop_cdiag_create(N, cim_dims, md_nontriv_dims(N, pat_dims), pattern);
			ret.lop_fft = linop_chain_FF(ret.lop_fft, linop_clone(ret.lop_pattern));
		}
	} else {

		assert(NULL == basis);
		assert(md_check_equal_dims(N, cim_dims, ksp_dims, ~0));
		UNUSED(bas_dims);

		const struct linop_s* lop_frw = linop_fftc_weighted_create(N, cim_dims, fft_flags, 0, NULL, md_nontriv_dims(N, pat_dims), pattern);
		const struct linop_s* lop_adj = linop_fftc_weighted_create(N, cim_dims, fft_flags, 0, NULL, 0, NULL);

		ret.lop_fft = linop_from_ops(lop_frw->forward, lop_adj->adjoint, NULL, NULL);

		linop_free(lop_frw);
		linop_free(lop_adj);
	}

	debug_printf(DP_DEBUG1, "\nModel created (%s):\n", conf->noncart ? "non Cartesian, psf-based" : "Cartesian");
	debug_printf(DP_DEBUG1, "kspace:  ");
	debug_print_dims(DP_DEBUG1, N, ksp_dims);
	debug_printf(DP_DEBUG1, "images:  ");
	debug_print_dims(DP_DEBUG1, N, img_dims);
	debug_printf(DP_DEBUG1, "coils:   ");
	debug_print_dims(DP_DEBUG1, N, col_dims);
	debug_printf(DP_DEBUG1, conf->noncart ? "psf:     " : "pattern: ");
	debug_print_dims(DP_DEBUG1, N, pat_dims);
	if(NULL != basis) {

		debug_printf(DP_DEBUG1, "basis:   ");
		debug_print_dims(DP_DEBUG1, N, bas_dims);
	}
	if(NULL != mask) {

		debug_printf(DP_DEBUG1, "mask:    ");
		debug_print_dims(DP_DEBUG1, N, msk_dims);
	}
	debug_printf(DP_DEBUG1, "coilimg: ");
	debug_print_dims(DP_DEBUG1, N, cim_dims);
	debug_printf(DP_DEBUG1, "\n");

	return ret;
}


void noir2_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void noir2_back_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}


static void proj_add(unsigned int flags, unsigned int D, const long dims[D],
			const long ostrs[D], complex float* optr,
			const long v1_strs[D], complex float* v1,
			const long v2_strs[D], complex float* v2)
{
	long v_dims[D];
	md_select_dims(D, flags, v_dims, dims);

	long v_strs[D];
	md_calc_strides(D, v_strs, v_dims, CFL_SIZE);

	complex float* v22 = md_alloc_sameplace(D, v_dims, CFL_SIZE, optr);

	md_ztenmulc2(D, dims, v_strs, v22, v2_strs, v2, v2_strs, v2);

	complex float* v12 = md_alloc_sameplace(D, v_dims, CFL_SIZE, optr);

	md_ztenmulc2(D, dims, v_strs, v12, v1_strs, v1, v2_strs, v2);
	md_zdiv(D, v_dims, v12, v12, v22);

	md_free(v22);

	md_ztenmul2(D, dims, ostrs, optr, v_strs, v12, v2_strs, v2);

	md_free(v12);
}


// FIXME: review dimensions
void noir2_orthogonalize(int N, const long col_dims[N], int idx, unsigned long flags, complex float* coils)
{
	long nmaps = col_dims[idx];

	if (1L == nmaps)
		return;

	long single_map_dims[N];
	md_select_dims(N, ~MD_BIT(idx), single_map_dims, col_dims);

	long single_map_strs[N];
	md_calc_strides(N, single_map_strs, single_map_dims, CFL_SIZE);

	long col_strs[N];
	md_calc_strides(N, col_strs, col_dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, single_map_dims, CFL_SIZE, coils);

	for (long map = 0L; map < nmaps; ++map) {

		complex float* map_ptr = (void*)coils + map * col_strs[idx];

		md_clear(N, single_map_dims, tmp, CFL_SIZE);

		for (long prev = 0L; prev < map; ++prev) {

			complex float* prev_map_ptr = (void*)coils + prev * col_strs[idx];

			proj_add(flags, N, single_map_dims,
				single_map_strs, tmp, single_map_strs, map_ptr, col_strs, prev_map_ptr);
		}

		md_zsub2(N, single_map_dims, col_strs, map_ptr, col_strs, map_ptr, single_map_strs, tmp);
	}

	md_free(tmp);
}

void noir2_free(struct noir2_s* model)
{
	nlop_free(model->nlop);
	linop_free(model->lop_coil2);
	linop_free(model->lop_fft);

	linop_free(model->lop_coil);
	linop_free(model->lop_im);
	nlop_free(model->tenmul);

	linop_free(model->lop_pattern);
	linop_free(model->lop_nufft);

	xfree(model->pat_dims);
	xfree(model->bas_dims);
	xfree(model->msk_dims);
	xfree(model->ksp_dims);
	xfree(model->cim_dims);
	xfree(model->img_dims);
	xfree(model->col_dims);
	xfree(model->trj_dims);
}
