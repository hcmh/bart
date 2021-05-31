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
	.img_space_coils = false,
	.enlive_flags = 0,
	.scaling = -1,
	.undo_scaling = false,

	.noncart = false,
	.nufft_conf = NULL,

	.gpu = false,
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
	assert(md_check_equal_dims(N, ksp_dims, linop_codomain(noir_ops.lop_trafo)->dims, ~0));

	unsigned long fft_flags = noir_ops.model_conf.fft_flags_noncart | noir_ops.model_conf.fft_flags_cart;


	complex float* data = md_alloc(N, cim_dims, CFL_SIZE);
	linop_adjoint(noir_ops.lop_trafo, N, cim_dims, data, N, ksp_dims, kspace);


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

		if (conf->img_space_coils) { // transform coils back to k-space

			complex float* sens_ref_buf = md_alloc(N, col_dims, CFL_SIZE);
			md_copy(N, col_dims, sens_ref_buf, sens_ref, CFL_SIZE);

			ifftmod(DIMS, col_dims, fft_flags, sens_ref_buf, sens_ref);
			noir2_back_coils(noir_ops.lop_coil, sens_ref_buf, sens_ref_buf);	//FIXME: must be inverse not adjoint

			md_copy(N, col_dims, ref + skip, sens_ref_buf, CFL_SIZE);

			md_free(sens_ref_buf);

		} else  {

			md_copy(N, col_dims, ref + skip, sens_ref, CFL_SIZE);
		}
	}

	complex float* x = md_alloc_sameplace(1, MD_DIMS(size), CFL_SIZE, data);

	md_copy(N, img_dims, x, img, CFL_SIZE);
	md_copy(N, col_dims, x + skip, ksens, CFL_SIZE);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = 0.1f;
	irgnm_conf.nlinv_legacy = true;
	irgnm_conf.alpha_min = conf->alpha_min;


	struct nlop_s* nlop_flat = nlop_flatten(noir_ops.nlop);
	const struct linop_s* lop_trafo_flat = linop_reshape_in(noir_ops.lop_trafo, 1, MD_DIMS(md_calc_size(N, cim_dims)));

	struct nlop_wrapper2_s nlw;
	SET_TYPEID(nlop_wrapper2_s, &nlw);
	nlw.split = skip;
	nlw.N = N;
	nlw.idx = MAPS_DIM;
	nlw.flag = conf->enlive_flags;
	nlw.col_dims = col_dims;

	iter4_lop_irgnm(CAST_UP(&irgnm_conf),
			nlop_flat,
			(struct linop_s*)lop_trafo_flat,
			size * 2, (float*)x, (const float*)ref,
			md_calc_size(N, cim_dims) * 2, (const float*)data,
			NULL,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw) });

	nlop_free(nlop_flat);
	linop_free(lop_trafo_flat);

	md_free(data);

	md_copy(DIMS, img_dims, img, x, CFL_SIZE);
	md_copy(DIMS, col_dims, ksens, x + skip, CFL_SIZE);

	noir2_forw_coils(noir_ops.lop_coil, x + skip, x + skip);
	md_copy(DIMS, col_dims, sens, x + skip, CFL_SIZE);	// needed for GPU
	fftmod(DIMS, col_dims, fft_flags, sens, sens);

	md_free(x);
	md_free(ref);

	if (conf->undo_scaling)
		md_zsmul(N, img_dims, img, img, 1./scaling);
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

	nlop_free(noir_ops.nlop);
	linop_free(noir_ops.lop_coil);
	linop_free(noir_ops.lop_trafo);
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

	nlop_free(noir_ops.nlop);
	linop_free(noir_ops.lop_coil);
	linop_free(noir_ops.lop_trafo);
}
