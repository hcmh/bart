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
	long sign_strs[DIMS];

	long data_dims[DIMS];

	long coil_dims[DIMS];
	long coil_strs[DIMS];

	long imgs_dims[DIMS];
	long imgs_strs[DIMS];

	const struct linop_s* weights;
	const struct linop_s* frw;
	const struct linop_s* adj;

	complex float* sens;
	complex float* xn;

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
	md_calc_strides(DIMS, data->sign_strs, data->sign_dims, CFL_SIZE);

	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG|MAPS_FLAG, data->coil_dims, dims);
	md_calc_strides(DIMS, data->coil_strs, data->coil_dims, CFL_SIZE);

	md_select_dims(DIMS, conf->fft_flags|MAPS_FLAG|CSHIFT_FLAG, data->imgs_dims, dims);
	md_calc_strides(DIMS, data->imgs_strs, data->imgs_dims, CFL_SIZE);

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

	data->sens = my_alloc(DIMS, data->coil_dims, CFL_SIZE);
	data->xn = my_alloc(DIMS, data->imgs_dims, CFL_SIZE);
	data->tmp = my_alloc(DIMS, data->sign_dims, CFL_SIZE);

	return PTR_PASS(data);
}


void noir_free(struct noir_data* data)
{
	md_free(data->xn);
	md_free(data->sens);
	md_free(data->tmp);

	linop_free(data->frw);
	linop_free(data->adj);

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

	md_copy(DIMS, data->imgs_dims, data->xn, src, CFL_SIZE);
	noir_forw_coils(data, data->sens, src + split);

	md_clear(DIMS, data->sign_dims, data->tmp, CFL_SIZE);
	md_zfmac2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->imgs_strs, src, data->coil_strs, data->sens);

	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->sign_dims, data->tmp);
}


void noir_der(struct noir_data* data, complex float* dst, const complex float* src)
{
	long split = md_calc_size(DIMS, data->imgs_dims);

	md_clear(DIMS, data->sign_dims, data->tmp, CFL_SIZE);
	md_zfmac2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->imgs_strs, src, data->coil_strs, data->sens);

	complex float* delta_coils = md_alloc_sameplace(DIMS, data->coil_dims, CFL_SIZE, src);
	noir_forw_coils(data, delta_coils, src + split);
	md_zfmac2(DIMS, data->sign_dims, data->sign_strs, data->tmp, data->coil_strs, delta_coils, data->imgs_strs, data->xn);
	md_free(delta_coils);

	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->sign_dims, data->tmp);
}


void noir_adj(struct noir_data* data, complex float* dst, const complex float* src)
{
	long split = md_calc_size(DIMS, data->imgs_dims);

	linop_adjoint(data->adj, DIMS, data->sign_dims, data->tmp, DIMS, data->data_dims, src);


	md_clear(DIMS, data->coil_dims, dst + split, CFL_SIZE);
	md_zfmacc2(DIMS, data->sign_dims, data->coil_strs, dst + split, data->sign_strs, data->tmp, data->imgs_strs, data->xn);

	noir_back_coils(data, dst + split, dst + split);

	md_clear(DIMS, data->imgs_dims, dst, CFL_SIZE);
	md_zfmacc2(DIMS, data->sign_dims, data->imgs_strs, dst, data->sign_strs, data->tmp, data->coil_strs, data->sens);

	if (data->conf.rvc)
		md_zreal(DIMS, data->imgs_dims, dst, dst);
}



