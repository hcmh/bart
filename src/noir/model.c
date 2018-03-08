/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker
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

#include "nlops/nlop.h"
#include "nlops/tenmul.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/fft.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "model.h"



struct noir_model_conf_s noir_model_conf_defaults = {

	.fft_flags = FFT_FLAGS,
	.rvc = false,
	.use_gpu = false,
	.noncart = false,
};



struct noir_data {

	long dims[DIMS];

	long sign_dims[DIMS];
	long data_dims[DIMS];
	long coil_dims[DIMS];
	long imgs_dims[DIMS];

	const struct linop_s* weights;
	const struct linop_s* frw;
	const struct linop_s* adj;

	const struct nlop_s* nl;
	const struct linop_s* der1;
	const struct linop_s* der2;

	const struct nlop_s* nl2;


	complex float* tmp;

	struct noir_model_conf_s conf;
};



static void noir_calc_weights(const long dims[3], complex float* dst)
{
	unsigned int flags = 0;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);

	klaplace(3, dims, flags, dst);
	md_zsmul(3, dims, dst, dst, 220.);
	md_zsadd(3, dims, dst, dst, 1.);
	md_zspow(3, dims, dst, dst, -16.);	// 1 + 222. \Laplace^16
}


struct noir_data* noir_init(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = conf->use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!conf->use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct noir_data, data);

	data->conf = *conf;

	md_copy_dims(DIMS, data->dims, dims);

	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG|CSHIFT_FLAG, data->sign_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG|MAPS_FLAG, data->coil_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|MAPS_FLAG|CSHIFT_FLAG, data->imgs_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG, data->data_dims, dims);

	long mask_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, mask_dims, dims);

	long wght_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, wght_dims, dims);

	long ptrn_dims[DIMS];
	md_select_dims(DIMS, conf->fft_flags|CSHIFT_FLAG, ptrn_dims, dims);


	complex float* weights = md_alloc(DIMS, wght_dims, CFL_SIZE);

	noir_calc_weights(dims, weights);
	fftmod(DIMS, wght_dims, FFT_FLAGS, weights, weights);
	fftscale(DIMS, wght_dims, FFT_FLAGS, weights, weights);

	data->weights = linop_chain(
		linop_cdiag_create(DIMS, data->coil_dims, FFT_FLAGS, weights),
		linop_ifft_create(DIMS, data->coil_dims, FFT_FLAGS));


	const struct linop_s* lop_fft = linop_fft_create(DIMS, data->sign_dims, conf->fft_flags);


	complex float* ptr = my_alloc(DIMS, ptrn_dims, CFL_SIZE);

	md_copy(DIMS, ptrn_dims, ptr, psf, CFL_SIZE);
	fftmod(DIMS, ptrn_dims, conf->fft_flags, ptr, ptr);

	const struct linop_s* lop_pattern = linop_fmac_create(DIMS, data->sign_dims, ~(conf->fft_flags|COIL_FLAG), ~(conf->fft_flags|COIL_FLAG|CSHIFT_FLAG), ~(conf->fft_flags|CSHIFT_FLAG), ptr);

	const struct linop_s* lop_adj_pattern;

	if (!conf->noncart) {

		lop_adj_pattern = linop_clone(lop_pattern);

	} else {

		complex float* adj_pattern = my_alloc(DIMS, ptrn_dims, CFL_SIZE);
		md_zfill(DIMS, ptrn_dims, adj_pattern, 1.);
		fftmod(DIMS, ptrn_dims, conf->fft_flags, adj_pattern, adj_pattern);

		lop_adj_pattern = linop_fmac_create(DIMS, data->sign_dims, ~(conf->fft_flags|COIL_FLAG), ~(conf->fft_flags|COIL_FLAG|CSHIFT_FLAG), ~(conf->fft_flags|CSHIFT_FLAG), adj_pattern);
	}



	complex float* msk = my_alloc(DIMS, mask_dims, CFL_SIZE);

	if (NULL == mask) {

		assert(!conf->use_gpu);
		md_zfill(DIMS, mask_dims, msk, 1.);

	} else {

		md_copy(DIMS, mask_dims, msk, mask, CFL_SIZE);
	}

//	fftmod(DIMS, data->mask_dims, 7, msk, msk);
	fftscale(DIMS, mask_dims, FFT_FLAGS, msk, msk);

	const struct linop_s* lop_mask = linop_cdiag_create(DIMS, data->sign_dims, FFT_FLAGS, msk);

	// could be moved to the benning, but see comment below
	lop_fft = linop_chain(lop_mask, lop_fft);

	data->frw = linop_chain(lop_fft, lop_pattern);
	data->adj = linop_chain(lop_fft, lop_adj_pattern);

	data->tmp = my_alloc(DIMS, data->sign_dims, CFL_SIZE);


	data->nl = nlop_tenmul_create(DIMS, data->sign_dims, data->imgs_dims, data->coil_dims);

	const struct nlop_s* nlw = nlop_from_linop(data->weights);


	data->nl2 = nlop_combine(data->nl, nlw);
	data->nl2 = nlop_link(data->nl2, 1, 1);
	data->der1 = nlop_get_derivative(data->nl2, 0, 0);
	data->der2 = nlop_get_derivative(data->nl2, 0, 1);

	return PTR_PASS(data);
}


void noir_free(struct noir_data* data)
{
	md_free(data->tmp);

	linop_free(data->frw);
	linop_free(data->adj);
	linop_free(data->weights);

	nlop_free(data->nl);

	xfree(data);
}

void noir_forw_coils(struct noir_data* data, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(data->weights, dst, src);
}

void noir_back_coils(struct noir_data* data, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(data->weights, dst, src);
}



void noir_fun(struct noir_data* data, complex float* dst, const complex float* src)
{	
	long split = md_calc_size(DIMS, data->imgs_dims);

	nlop_generic_apply_unchecked(data->nl2, 3, (void*[3]){ data->tmp, (void*)(src), (void*)(src + split) });

	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->sign_dims, data->tmp);
}


void noir_der(struct noir_data* data, complex float* dst, const complex float* src)
{
	long split = md_calc_size(DIMS, data->imgs_dims);

	linop_forward(data->der1, DIMS, data->sign_dims, data->tmp, DIMS, data->imgs_dims, src);

	complex float* tmp2 = md_alloc_sameplace(DIMS, data->sign_dims, CFL_SIZE, src);

	linop_forward(data->der2, DIMS, data->sign_dims, tmp2, DIMS, data->coil_dims, src + split);
	md_zadd(DIMS, data->sign_dims, data->tmp, data->tmp, tmp2);
	md_free(tmp2);

	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->sign_dims, data->tmp);
}


void noir_adj(struct noir_data* data, complex float* dst, const complex float* src)
{
	long split = md_calc_size(DIMS, data->imgs_dims);

	linop_adjoint(data->adj, DIMS, data->sign_dims, data->tmp, DIMS, data->data_dims, src);

	linop_adjoint(data->der2, DIMS, data->coil_dims, dst + split, DIMS, data->sign_dims, data->tmp);

	linop_adjoint(data->der1, DIMS, data->imgs_dims, dst, DIMS, data->sign_dims, data->tmp);

	if (data->conf.rvc)
		md_zreal(DIMS, data->imgs_dims, dst, dst);
}



