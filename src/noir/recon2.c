/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
 * nonlinear inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"

#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/iter4_lop.h"
#include "iter/thresh.h"
#include "iter/italgos.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model2.h"

#include "nlops/nlop.h"

#include "recon2.h"


struct nlop_wrapper2_s {

	INTERFACE(struct iter_op_data_s);

	long split;

	int N;
	int idx;
	unsigned long flag;

	const long* col_dims;
};

DEF_TYPEID(nlop_wrapper2_s);


static void orthogonalize(iter_op_data* ptr, float* _dst, const float* _src)
{
#if 0
	noir_orthogonalize(nlop_get_data(CAST_DOWN(nlop_wrapper_s, ptr)->noir), (complex float*)_dst, (const complex float*)_src);
#else
	UNUSED(_src);

	auto nlw = CAST_DOWN(nlop_wrapper2_s, ptr);

	noir2_orthogonalize(nlw->N, nlw->col_dims, nlw->idx, nlw->flag, (complex float*) _dst + nlw->split);
#endif
}


const struct noir2_conf_s noir2_defaults = {

	.iter = 8,
	.rvc = false,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.a = 220.,
	.b = 32.,
	.sms = false,
	.sos = false,
	.enlive_flags = 0,
	.scaling = -1,
	.undo_scaling = false,

	.loop_flags = 0,

	.noncart = false,
	.nufft_conf = NULL,

	.gpu = false,
	
	.cgiter = 30,
	.cgtol = 0.1,
};




static void noir2_recon(const struct noir2_conf_s* conf, struct noir2_s noir_ops,
			int N,
			const long img_dims[N], complex float* img, const complex float* img_ref,
			const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
			const long ksp_dims[N], const complex float* kspace)
{

	assert(N == (int)nlop_generic_codomain(noir_ops.nlop, 0)->N);
	long cim_dims[N];
	md_copy_dims(N, cim_dims, nlop_generic_codomain(noir_ops.nlop, 0)->dims);

	assert(md_check_equal_dims(N, img_dims, nlop_generic_domain(noir_ops.nlop, 0)->dims, ~0));
	assert(md_check_equal_dims(N, col_dims, nlop_generic_domain(noir_ops.nlop, 1)->dims, ~0));
	assert(md_check_equal_dims(N, ksp_dims, linop_codomain(noir_ops.lop_fft)->dims, ~0));

	unsigned long fft_flags = noir_ops.model_conf.fft_flags_noncart | noir_ops.model_conf.fft_flags_cart;


	complex float* data = md_alloc(N, cim_dims, CFL_SIZE);
	linop_adjoint(noir_ops.lop_fft, N, cim_dims, data, N, ksp_dims, kspace);


#ifdef USE_CUDA
	if(conf->gpu) {

		complex float* tmp_data = md_alloc_gpu(N, cim_dims, CFL_SIZE);
		md_copy(N, cim_dims, tmp_data, data, CFL_SIZE);
		md_free(data);
		data = tmp_data;
	}
#else
	if(conf->gpu)
		error("Compiled without GPU support!");
#endif

	float scaling = conf->scaling;
	if (-1 == scaling) {

		scaling = 100. / md_znorm(N, cim_dims, data);
		if (conf->sms || conf->sos)
			scaling *= sqrt(cim_dims[SLICE_DIM]);
	}
	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(N, cim_dims, data, data, scaling);


	long skip = md_calc_size(N, img_dims);
	long size = skip + md_calc_size(N, col_dims);

	complex float* ref = NULL;
	assert((NULL == img_ref) == (NULL == sens_ref));

	if (NULL != img_ref) {

		ref = md_alloc_sameplace(1, MD_DIMS(size), CFL_SIZE, data);

		md_copy(N, img_dims, ref, img_ref, CFL_SIZE);
		md_copy(N, col_dims, ref + skip, sens_ref, CFL_SIZE);
	}

	complex float* x = md_alloc_sameplace(1, MD_DIMS(size), CFL_SIZE, data);

	md_copy(N, img_dims, x, img, CFL_SIZE);
	md_copy(N, col_dims, x + skip, ksens, CFL_SIZE);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = conf->cgtol;
	irgnm_conf.nlinv_legacy = true;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.cgiter = conf->cgiter;


	struct nlop_s* nlop_flat = nlop_flatten(noir_ops.nlop);
	const struct linop_s* lop_fft_flat = linop_reshape_in(noir_ops.lop_fft, 1, MD_DIMS(md_calc_size(N, cim_dims)));

	struct nlop_wrapper2_s nlw;
	SET_TYPEID(nlop_wrapper2_s, &nlw);
	nlw.split = skip;
	nlw.N = N;
	nlw.idx = MAPS_DIM;
	nlw.flag = conf->enlive_flags;
	nlw.col_dims = col_dims;

	iter4_lop_irgnm(CAST_UP(&irgnm_conf),
			nlop_flat,
			(struct linop_s*)lop_fft_flat,
			size * 2, (float*)x, (const float*)ref,
			md_calc_size(N, cim_dims) * 2, (const float*)data,
			NULL,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });

	nlop_free(nlop_flat);
	linop_free(lop_fft_flat);

	md_free(data);

	md_copy(DIMS, img_dims, img, x, CFL_SIZE);
	md_copy(DIMS, col_dims, ksens, x + skip, CFL_SIZE);

	noir2_forw_coils(noir_ops.lop_coil2, x + skip, x + skip);
	md_copy(DIMS, col_dims, sens, x + skip, CFL_SIZE);	// needed for GPU
	fftmod(DIMS, col_dims, fft_flags, sens, sens);

	md_free(x);
	md_free(ref);

	if (conf->undo_scaling)
		md_zsmul(N, img_dims, img, img, 1./scaling);
}


static void noir2_recon_noncart_loop(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	struct noir2_model_conf_s mconf = noir2_model_conf_defaults;

	mconf.fft_flags_noncart = FFT_FLAGS;
	mconf.fft_flags_cart = (conf->sms || conf->sos) ? SLICE_FLAG : 0;

	mconf.rvc = conf->rvc;
	mconf.sos = conf->sos;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.noncart = conf->noncart;

	mconf.nufft_conf = conf->nufft_conf;

	struct noir2_s noir_ops = noir2_noncart_create(N, trj_dims, traj, wgh_dims, weights, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &mconf);

	noir2_recon(conf, noir_ops, N, img_dims, img, img_ref, col_dims, sens, ksens, sens_ref, ksp_dims, kspace);

	noir2_free(&noir_ops);
}

void noir2_recon_noncart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	long limg_dims[N];
	long lcol_dims[N];
	long lksp_dims[N];
	long ltrj_dims[N];
	long lwgh_dims[N];
	long lbas_dims[N];
	long lmsk_dims[N];
	long lcim_dims[N];

	md_select_dims(N, ~conf->loop_flags, limg_dims, img_dims);
	md_select_dims(N, ~conf->loop_flags, lcol_dims, col_dims);
	md_select_dims(N, ~conf->loop_flags, lksp_dims, ksp_dims);
	md_select_dims(N, ~conf->loop_flags, ltrj_dims, trj_dims);
	md_select_dims(N, ~conf->loop_flags, lwgh_dims, wgh_dims);
	md_select_dims(N, ~conf->loop_flags, lbas_dims, bas_dims);
	md_select_dims(N, ~conf->loop_flags, lmsk_dims, msk_dims);
	md_select_dims(N, ~conf->loop_flags, lcim_dims, cim_dims);

	long img_strs[N];
	long col_strs[N];
	long ksp_strs[N];
	long trj_strs[N];
	long wgh_strs[N];
	long bas_strs[N];
	long msk_strs[N];
	long cim_strs[N];

	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);
	md_calc_strides(N, col_strs, col_dims, CFL_SIZE);
	md_calc_strides(N, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(N, trj_strs, trj_dims, CFL_SIZE);
	md_calc_strides(N, wgh_strs, wgh_dims, CFL_SIZE);
	md_calc_strides(N, bas_strs, bas_dims, CFL_SIZE);
	md_calc_strides(N, msk_strs, msk_dims, CFL_SIZE);
	md_calc_strides(N, cim_strs, cim_dims, CFL_SIZE);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;
	
	do {

		complex float* l_img = &MD_ACCESS(N, img_strs, pos, img);
		const complex float* l_img_ref = (NULL == img_ref) ? NULL : &MD_ACCESS(N, img_strs, pos, img_ref);
		complex float* l_sens = (NULL == sens) ? NULL : &MD_ACCESS(N, col_strs, pos, sens);
		complex float* l_ksens = (NULL == ksens) ? NULL : &MD_ACCESS(N, col_strs, pos, ksens);
		const complex float* l_sens_ref = (NULL == sens_ref) ? NULL : &MD_ACCESS(N, col_strs, pos, sens_ref);
		const complex float* l_kspace = &MD_ACCESS(N, ksp_strs, pos, kspace);
		const complex float* l_traj = &MD_ACCESS(N, trj_strs, pos, traj);
		const complex float* l_weights = (NULL == weights) ? NULL : &MD_ACCESS(N, wgh_strs, pos, weights);
		const complex float* l_basis = (NULL == basis) ? NULL : &MD_ACCESS(N, bas_strs, pos, basis);
		const complex float* l_mask = (NULL == mask) ? NULL : &MD_ACCESS(N, msk_strs, pos, mask);

		noir2_recon_noncart_loop(conf, N,
					 limg_dims, l_img, l_img_ref,
					 lcol_dims, l_sens, l_ksens, l_sens_ref,
					 lksp_dims, l_kspace,
					 ltrj_dims, l_traj,
					 lwgh_dims, l_weights,
					 lbas_dims, l_basis,
					 lmsk_dims, l_mask,
					 lcim_dims
					);


	} while (md_next(N, ksp_dims, conf->loop_flags, pos));
}


static void noir2_recon_cart_loop(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	struct noir2_model_conf_s mconf = noir2_model_conf_defaults;

	mconf.fft_flags_noncart = 0;
	mconf.fft_flags_cart = FFT_FLAGS | ((conf->sms || conf->sos) ? SLICE_FLAG : 0);

	mconf.rvc = conf->rvc;
	mconf.sos = conf->sos;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.noncart = conf->noncart;

	mconf.nufft_conf = conf->nufft_conf;


	struct noir2_s noir_ops = noir2_cart_create(N, pat_dims, pattern, bas_dims, basis, msk_dims, mask, ksp_dims, cim_dims, img_dims, col_dims, &mconf);

	noir2_recon(conf, noir_ops, N, img_dims, img, img_ref, col_dims, sens, ksens, sens_ref, ksp_dims, kspace);

	noir2_free(&noir_ops);
}

void noir2_recon_cart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N])
{
	long limg_dims[N];
	long lcol_dims[N];
	long lksp_dims[N];
	long lpat_dims[N];
	long lbas_dims[N];
	long lmsk_dims[N];
	long lcim_dims[N];

	md_select_dims(N, ~conf->loop_flags, limg_dims, img_dims);
	md_select_dims(N, ~conf->loop_flags, lcol_dims, col_dims);
	md_select_dims(N, ~conf->loop_flags, lksp_dims, ksp_dims);
	md_select_dims(N, ~conf->loop_flags, lpat_dims, pat_dims);
	md_select_dims(N, ~conf->loop_flags, lbas_dims, bas_dims);
	md_select_dims(N, ~conf->loop_flags, lmsk_dims, msk_dims);
	md_select_dims(N, ~conf->loop_flags, lcim_dims, cim_dims);

	long img_strs[N];
	long col_strs[N];
	long ksp_strs[N];
	long pat_strs[N];
	long bas_strs[N];
	long msk_strs[N];
	long cim_strs[N];

	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);
	md_calc_strides(N, col_strs, col_dims, CFL_SIZE);
	md_calc_strides(N, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(N, pat_strs, pat_dims, CFL_SIZE);
	md_calc_strides(N, bas_strs, bas_dims, CFL_SIZE);
	md_calc_strides(N, msk_strs, msk_dims, CFL_SIZE);
	md_calc_strides(N, cim_strs, cim_dims, CFL_SIZE);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;
	
	do {

		complex float* l_img = &MD_ACCESS(N, img_strs, pos, img);
		const complex float* l_img_ref = (NULL == img_ref) ? NULL : &MD_ACCESS(N, img_strs, pos, img_ref);
		complex float* l_sens = (NULL == sens) ? NULL : &MD_ACCESS(N, col_strs, pos, sens);
		complex float* l_ksens = (NULL == ksens) ? NULL : &MD_ACCESS(N, col_strs, pos, ksens);
		const complex float* l_sens_ref = (NULL == sens_ref) ? NULL : &MD_ACCESS(N, col_strs, pos, sens_ref);
		const complex float* l_kspace = &MD_ACCESS(N, ksp_strs, pos, kspace);
		const complex float* l_pattern = &MD_ACCESS(N, pat_strs, pos, pattern);
		const complex float* l_basis = (NULL == basis) ? NULL : &MD_ACCESS(N, bas_strs, pos, basis);
		const complex float* l_mask = (NULL == mask) ? NULL : &MD_ACCESS(N, msk_strs, pos, mask);

		noir2_recon_cart_loop(conf, N,
				      limg_dims, l_img, l_img_ref,
				      lcol_dims, l_sens, l_ksens, l_sens_ref,
				      lksp_dims, l_kspace,
				      lpat_dims, l_pattern,
				      lbas_dims, l_basis,
				      lmsk_dims, l_mask,
				      lcim_dims
				     );


	} while (md_next(N, ksp_dims, conf->loop_flags, pos));
}
