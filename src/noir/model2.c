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

/**
 * This function creates the non (bi)-linear part of the sense model, i.e.
 * 	cim = (mask * img) * ifftuc[weights * ksens]
 * - if cim_dims does not equal img_dims or col_dims, we include a resize (oversampling for coils)
 * - if mask is provided, the dimensioons shoul match img_dims or be trivial
 * - if conf->rvc, only the real part of img is considered
 * lop_coil contains the coil transform without a second ifftmod, i.e.
 *	lop_coil(x) = fftmod(ifftuc(weights * x))
 **/
static struct noir2_s noir2_nlop_create(int N,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf)
{
	struct noir2_s ret = {
		.nlop = NULL,
		.lop_coil = NULL,
		.lop_trafo = NULL,
	};

	unsigned long fft_flags = conf->fft_flags_noncart | conf->fft_flags_cart;

	long wght_dims[N];
	md_select_dims(N, fft_flags, wght_dims, col_dims);

	complex float* wghts = md_alloc(N, wght_dims, CFL_SIZE);
	noir2_calc_weights(conf, N, wght_dims, wghts);
	ifftmod(N, wght_dims, fft_flags, wghts, wghts);		//in the previous implementation fftmod was used, but ifftmod should be correct
	fftscale(N, wght_dims, fft_flags, wghts, wghts);

	const struct linop_s* lop_wghts = linop_cdiag_create(N, col_dims, fft_flags, wghts);
	const struct linop_s* lop_wghts_ifft = linop_ifft_create(N, col_dims, fft_flags);

	const struct linop_s* lop_coils = linop_chain_FF(lop_wghts, lop_wghts_ifft);
	ret.lop_coil = linop_clone(lop_coils);	// extra ifftmod is missing and merged into nlop


	complex float* ifmod = md_alloc(N, wght_dims, CFL_SIZE);
	md_zfill(N, wght_dims, ifmod, 1.);
	ifftmod(N, wght_dims, fft_flags, ifmod, ifmod);

	//FIXME: In the Cartesian case, we might merge ifftmod with fftmod of fft in linop trafo part (as in previous implementation)
	//CAVEAT: if coil dims != img dims, we need to check if they cancel
	lop_coils = linop_chain_FF(lop_coils, linop_cdiag_create(N, col_dims, fft_flags, ifmod));
	md_free(ifmod);

	long max_dims[N]; 	//of tenmul
	for (int i = 0; i < N; i++)
		max_dims[i] = (1 == cim_dims[i]) ? MIN(img_dims[i], col_dims[i]) : cim_dims[i];

	long img_dims2[N];
	long col_dims2[N];
	md_select_dims(N, md_nontriv_dims(N, img_dims), img_dims2, max_dims);
	md_select_dims(N, md_nontriv_dims(N, col_dims), col_dims2, max_dims);

	if (!md_check_equal_dims(N, col_dims, col_dims2, ~0))
		lop_coils = linop_chain_FF(lop_coils, linop_resize_center_create(N, col_dims2, col_dims));


	const struct linop_s* lop_mask = NULL;
	if (NULL != mask) {

		assert(md_check_equal_dims(N, msk_dims, img_dims, md_nontriv_dims(N, msk_dims)));
		lop_mask = linop_cdiag_create(N, img_dims, md_nontriv_dims(N, msk_dims), mask);
	}

	if (!md_check_equal_dims(N, img_dims, img_dims2, ~0)) {

		if (NULL == mask)
			lop_mask = linop_resize_center_create(N, img_dims, img_dims2);
		else
			lop_mask = linop_chain_FF(lop_mask, linop_resize_center_create(N, img_dims, img_dims2));
	}

	if (conf->rvc) {

		if (NULL == mask)
			lop_mask = linop_zreal_create(N, img_dims);
		else
			lop_mask = linop_chain_FF(linop_zreal_create(N, img_dims), lop_mask);
	}

	ret.nlop = nlop_tenmul_create(N, cim_dims, img_dims2, col_dims2);
	ret.nlop = nlop_chain2_FF(nlop_from_linop_F(lop_coils), 0, ret.nlop, 1);
	if (NULL != lop_mask)
		ret.nlop = nlop_chain2_swap_FF(nlop_from_linop_F(lop_mask),0, ret.nlop, 0);

	ret.model_conf = *conf;

	return ret;
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

	struct noir2_s ret = noir2_nlop_create(N, msk_dims, mask, cim_dims, img_dims, col_dims, conf);

	struct nufft_conf_s nufft_conf = *(conf->nufft_conf);
	nufft_conf.flags = conf->fft_flags_noncart;
	nufft_conf.cfft = conf->fft_flags_cart;

	assert(md_check_equal_dims(N, ksp_dims, cim_dims, ~conf->fft_flags_noncart));
	ret.lop_trafo = nufft_create2(N, ksp_dims, cim_dims, trj_dims, traj, wgh_dims, weights, bas_dims, basis, nufft_conf);

	// We need to add fftmod and scale for the uncenterd cartesian fft (SMS/SOS)
	if (0 != conf->fft_flags_cart) {

		long fft_dims[N];
		md_select_dims(N, conf->fft_flags_cart, fft_dims, cim_dims);
		complex float* fmod = md_alloc(N, fft_dims, CFL_SIZE);
		md_zfill(N, fft_dims, fmod, 1);
		fftmod(N, fft_dims, conf->fft_flags_cart, fmod, fmod);

		const struct linop_s* lop_fftmod = linop_cdiag_create(N, ksp_dims, conf->fft_flags_cart, fmod);

		//manual chain lop_fftmod to keep normal of nufft
		const struct operator_s* frw = operator_chain(ret.lop_trafo->forward, lop_fftmod->forward);
		const struct operator_s* adj = operator_chain(lop_fftmod->adjoint, ret.lop_trafo->adjoint);
		const struct linop_s* lop_trafo = linop_from_ops(frw, adj, ret.lop_trafo->normal, NULL);
		operator_free(frw);
		operator_free(adj);

		linop_free(ret.lop_trafo);
		ret.lop_trafo = lop_trafo;


		linop_free(lop_fftmod);

		fftscale(N, fft_dims, conf->fft_flags_cart, fmod, fmod);
		ret.lop_trafo = linop_chain_FF(linop_cdiag_create(N, cim_dims, conf->fft_flags_cart, fmod), ret.lop_trafo);

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

	assert(NULL == basis);
	assert(md_check_equal_dims(N, cim_dims, ksp_dims, ~0));
	UNUSED(bas_dims);


	struct noir2_s ret = noir2_nlop_create(N, msk_dims, mask, cim_dims, img_dims, col_dims, conf);

	unsigned long fft_flags = conf->fft_flags_noncart | conf->fft_flags_cart;

	long fft_dims[N];
	md_select_dims(N, fft_flags, fft_dims, cim_dims);

	complex float* fmod = md_alloc(N, fft_dims, CFL_SIZE);
	md_zfill(N, fft_dims, fmod, 1);
	fftmod(N, fft_dims, fft_flags, fmod, fmod);
	fftscale(N, fft_dims, fft_flags, fmod, fmod);

	ret.lop_trafo = linop_cdiag_create(N, cim_dims, fft_flags, fmod);
	md_free(fmod);

	ret.lop_trafo = linop_chain_FF(ret.lop_trafo, linop_fft_create(N, cim_dims, fft_flags));

	long pat_dims2[N];
	md_copy_dims(N, pat_dims2, pat_dims);
	md_max_dims(N, fft_flags, pat_dims2, pat_dims2, cim_dims);

	complex float* pattern2 = md_alloc(N, pat_dims2, CFL_SIZE);
	md_copy2(N, pat_dims2, MD_STRIDES(N, pat_dims2, CFL_SIZE), pattern2, MD_STRIDES(N, pat_dims, CFL_SIZE), pattern, CFL_SIZE);
	fftmod(N, pat_dims2, fft_flags, pattern2, pattern2);

	const struct linop_s* lop_pattern = linop_cdiag_create(N, cim_dims, md_nontriv_dims(N, pat_dims2), pattern2);
	md_free(pattern2);


	// if noncart, the pattern is understood as psf.
	// the forward model maps to the gridded kspace, while the adjoint does not contain the gridding
	if (!conf->noncart) {

		ret.lop_trafo = linop_chain_FF(ret.lop_trafo, lop_pattern);
	} else {

		complex float* fmod = md_alloc(N, fft_dims, CFL_SIZE);
		md_zfill(N, fft_dims, fmod, 1);
		fftmod(N, fft_dims, fft_flags, fmod, fmod);

		const struct linop_s* lop_fftmod = linop_cdiag_create(N, cim_dims, fft_flags, fmod);
		md_free(fmod);

		const struct operator_s* frw = operator_chain(ret.lop_trafo->forward, lop_pattern->forward);
		const struct operator_s* adj = operator_chain(lop_fftmod->adjoint, ret.lop_trafo->adjoint);

		linop_free(ret.lop_trafo);
		linop_free(lop_fftmod);
		linop_free(lop_pattern);

		ret.lop_trafo = linop_from_ops(frw, adj, NULL, NULL);

		operator_free(frw);
		operator_free(adj);
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
