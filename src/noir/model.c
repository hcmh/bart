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
#include "misc/mmio.h"

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

#include "noncart/nufft.h"

#include "model.h"
#include "recon.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif



struct noir_model_conf_s noir_model_conf_defaults = {

	.fft_flags = FFT_FLAGS,
	.iter = 8,
	.rvc = false,
	.use_gpu = false,
	.noncart = false,
	.a = 220.,
	.b = 32.,
	.pattern_for_each_coil = false,
	.out_im_steps = false,
	.out_coils_steps = false,
	.out_im = NULL,
	.out_coils = NULL,
};



struct noir_op_s {

	INTERFACE(nlop_data_t);

	const struct noir_dims_s* dims;

	const struct linop_s* weights;
	const struct linop_s* frw;

	const struct nlop_s* nl;
	/*const*/ struct nlop_s* nl2;


	// these are purely to free the arrays properly
	complex float* weights_array;
	complex float* fmod_array;
	complex float* psf_array;
	complex float* mask_array;

	complex float* tmp;

	struct noir_model_conf_s conf;
};


DEF_TYPEID(noir_op_s);

static void noir_calc_weights(const struct noir_model_conf_s *conf, const long dims[3], complex float* dst)
{
	unsigned int flags = 0;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);

	klaplace(3, dims, flags, dst);
	md_zsmul(3, dims, dst, dst, conf->a);
	md_zsadd(3, dims, dst, dst, 1.);
	md_zspow(3, dims, dst, dst, -conf->b / 2.);	// 1 + 220. \Laplace^16
}


static struct noir_op_s* noir_init(const struct noir_dims_s* dims, const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = conf->use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!conf->use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct noir_op_s, data);
	SET_TYPEID(noir_op_s, data);

	data->dims = dims;
	data->conf = *conf;

	// not needed in Cartesian
	data->fmod_array = NULL;

	long mask_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, mask_dims, dims->img_dims);

	long wght_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, wght_dims, dims->img_dims);

	long ptrn_dims[DIMS];
	unsigned int ptrn_flags;
	if (!conf->pattern_for_each_coil) {

		md_select_dims(DIMS, conf->fft_flags, ptrn_dims, dims->img_dims);
		ptrn_flags = ~(conf->fft_flags);
	} else {

		md_select_dims(DIMS, conf->fft_flags|COIL_FLAG, ptrn_dims, dims->sens_dims);
		ptrn_flags = ~(conf->fft_flags|COIL_FLAG);
	}



	data->weights_array = md_alloc(DIMS, wght_dims, CFL_SIZE);

	noir_calc_weights(conf, dims->img_dims, data->weights_array);
	fftmod(DIMS, wght_dims, FFT_FLAGS, data->weights_array, data->weights_array);
	fftscale(DIMS, wght_dims, FFT_FLAGS, data->weights_array, data->weights_array);

	const struct linop_s* lop_weights = linop_cdiag_create(DIMS, dims->sens_dims, FFT_FLAGS, data->weights_array);
	const struct linop_s* lop_ifft = linop_ifft_create(DIMS, dims->sens_dims, FFT_FLAGS);
	data->weights = linop_chain(lop_weights, lop_ifft);
	linop_free(lop_weights);
	linop_free(lop_ifft);


	const struct linop_s* lop_fft = linop_fft_create(DIMS, dims->ksp_dims, conf->fft_flags);


	data->psf_array = my_alloc(DIMS, ptrn_dims, CFL_SIZE);

	md_copy(DIMS, ptrn_dims, data->psf_array, psf, CFL_SIZE);
	fftmod(DIMS, ptrn_dims, conf->fft_flags, data->psf_array, data->psf_array);


	const struct linop_s* lop_pattern = linop_fmac_create(DIMS, dims->ksp_dims, ~(conf->fft_flags|COIL_FLAG), ~(conf->fft_flags|COIL_FLAG), ptrn_flags, data->psf_array);

	assert(!conf->noncart);


	data->mask_array = my_alloc(DIMS, mask_dims, CFL_SIZE);

	if (NULL == mask) {

		assert(!conf->use_gpu);
		md_zfill(DIMS, mask_dims, data->mask_array, 1.);

	} else {

		md_copy(DIMS, mask_dims, data->mask_array, mask, CFL_SIZE);
	}

	fftscale(DIMS, mask_dims, FFT_FLAGS, data->mask_array, data->mask_array);

	const struct linop_s* lop_mask = linop_cdiag_create(DIMS, dims->coil_imgs_dims, FFT_FLAGS, data->mask_array);

	// could be moved to the benning, but see comment below
	const struct linop_s* lop_fft2 = linop_chain(lop_mask, lop_fft);
	linop_free(lop_mask);
	linop_free(lop_fft);

	data->frw = linop_chain(lop_fft2, lop_pattern);
	linop_free(lop_fft2);
	linop_free(lop_pattern);

	data->tmp = my_alloc(DIMS, dims->sens_dims, CFL_SIZE);

	const struct nlop_s* nl_tenmul = nlop_tenmul_create(DIMS, dims->coil_imgs_dims, dims->img_dims, dims->sens_dims);

	const struct nlop_s* nlw = nlop_from_linop(data->weights);

	data->nl = nlop_chain2(nlw, 0, nl_tenmul, 1);
	nlop_free(nlw);
	nlop_free(nl_tenmul);

	const struct nlop_s* frw = nlop_from_linop(data->frw);

	data->nl2 = nlop_chain2(data->nl, 0, frw, 0);
	nlop_free(frw);

	return PTR_PASS(data);
}



static struct noir_op_s* noir_init_noncart(const struct noir_dims_s* dims, const complex float* mask, const complex float* traj, const struct nufft_conf_s* nufft_conf, const struct noir_model_conf_s* conf)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = conf->use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!conf->use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct noir_op_s, data);
	SET_TYPEID(noir_op_s, data);

	data->dims = dims;
	data->conf = *conf;

	// not needed in non-Cartesian
	data->psf_array = NULL;
	data->tmp = NULL;

	long mask_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, mask_dims, dims->img_dims);

	long wght_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, wght_dims, dims->img_dims);

	data->weights_array = md_alloc(DIMS, wght_dims, CFL_SIZE);

	noir_calc_weights(conf, dims->img_dims, data->weights_array);
	fftmod(DIMS, wght_dims, FFT_FLAGS, data->weights_array, data->weights_array);
	fftscale(DIMS, wght_dims, FFT_FLAGS, data->weights_array, data->weights_array);

	const struct linop_s* lop_weights = linop_cdiag_create(DIMS, dims->sens_dims, FFT_FLAGS, data->weights_array);
	const struct linop_s* lop_ifft = linop_ifft_create(DIMS, dims->sens_dims, FFT_FLAGS);
	const struct linop_s* lop_weights2 = linop_chain(lop_weights, lop_ifft);
	linop_free(lop_weights);
	linop_free(lop_ifft);


	data->fmod_array = md_alloc(DIMS, wght_dims, CFL_SIZE);
	md_zfill(DIMS, wght_dims, data->fmod_array, 1.f);
	fftmod(DIMS, wght_dims, FFT_FLAGS, data->fmod_array, data->fmod_array);
	const struct linop_s* lop_fmod = linop_cdiag_create(DIMS, dims->sens_dims, FFT_FLAGS, data->fmod_array);
	data->weights = linop_chain(lop_weights2, lop_fmod);
	linop_free(lop_weights2);
	linop_free(lop_fmod);

	const struct linop_s* lop_fft = nufft_create(DIMS, dims->ksp_dims, dims->coil_imgs_dims, dims->traj_dims, traj, NULL, *nufft_conf);


	data->mask_array = my_alloc(DIMS, mask_dims, CFL_SIZE);

	if (NULL == mask) {

		assert(!conf->use_gpu);
		md_zfill(DIMS, mask_dims, data->mask_array, 1.);

	} else {

		md_copy(DIMS, mask_dims, data->mask_array, mask, CFL_SIZE);
	}

	const struct linop_s* lop_mask = linop_cdiag_create(DIMS, dims->coil_imgs_dims, FFT_FLAGS, data->mask_array);

	data->frw = linop_chain(lop_mask, lop_fft);
	linop_free(lop_mask);
	linop_free(lop_fft);

	const struct nlop_s* nl_tenmul = nlop_tenmul_create(DIMS, dims->coil_imgs_dims, dims->img_dims, dims->sens_dims);

	const struct nlop_s* nlw = nlop_from_linop(data->weights);

	data->nl = nlop_chain2(nlw, 0, nl_tenmul, 1);
	nlop_free(nlw);
	nlop_free(nl_tenmul);

	const struct nlop_s* frw = nlop_from_linop(data->frw);

	data->nl2 = nlop_chain2(data->nl, 0, frw, 0);
	nlop_free(frw);

	return PTR_PASS(data);
}

void noir_free(struct noir_s* nl)
{
	struct noir_op_s* data = nl->noir_op;

	md_free(data->tmp);
	md_free(data->mask_array);
	md_free(data->fmod_array);
	md_free(data->weights_array);
	md_free(data->psf_array);

	linop_free(data->frw);
	linop_free(data->weights);

	nlop_free(data->nl);
	nlop_free(data->nl2);

	xfree(data);
}

void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void noir_back_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}

struct noir_s noir_create2(const struct noir_dims_s* dims, const complex float* mask, const complex float* psf, const complex float* traj, const struct nufft_conf_s* nufft_conf, const struct noir_model_conf_s* conf)
{
	assert(!conf->rvc);

	struct noir_op_s* data = NULL;
	if (conf->noncart)
		data = noir_init_noncart(dims, mask, traj, nufft_conf, conf);
	else
		data = noir_init(dims, mask, psf, conf);

	struct nlop_s* nlop = data->nl2;
	return (struct noir_s){ .nlop = nlop, .linop = data->weights, .noir_op = data };
}

struct noir_s noir_create(const struct noir_dims_s* dims, const complex float* mask, const complex float* psf, const complex float* traj, const struct nufft_conf_s* nufft_conf, const struct noir_model_conf_s* conf)
{
#if 0
	struct noir_op_s* data = noir_init(dims, mask, psf, conf);

	long idims[DIMS];
	md_select_dims(DIMS, conf->fft_flags|MAPS_FLAG|CSHIFT_FLAG, idims, dims);
	idims[COIL_DIM] = dims[COIL_DIM] + 1; // add image

	long odims[DIMS];
	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG|CSHIFT_FLAG, odims, dims);

	struct noir_s ret = { .linop = data->weights, .noir_op = data };
	ret.nlop = nlop_create(DIMS, odims, DIMS, idims, CAST_UP(PTR_PASS(data)), noir_fun, noir_der, noir_adj, NULL, NULL, noir_del);

	return ret;
#else
	// less efficient than the manuel coded functions

	struct noir_s ret = noir_create2(dims, mask, psf, traj, nufft_conf, conf);
	const struct nlop_s* nl_tmp = ret.nlop;
	ret.nlop = nlop_flatten(nl_tmp);
	return ret;
#endif
}


void noir_dump(struct noir_s* op, const complex float* img, const complex float* coils)
{
	UNUSED(coils);
	static unsigned int i = 0;


	struct noir_op_s* data = op->noir_op;
	if (NULL == data) {
		if (0 == i++)
			debug_printf(DP_INFO, "Cannot dump Newton steps\n");
		return;
	}

	if (data->conf.out_im_steps) {

		long out_im_dims[DIMS];
		md_copy_dims(DIMS, out_im_dims, data->dims->img_dims);
		long out_im_single_dims[DIMS];
		md_copy_dims(DIMS, out_im_single_dims, out_im_dims);
		out_im_dims[ITER_DIM] = data->conf.iter;

		complex float* out_img = data->conf.out_im + i*md_calc_size(DIMS, out_im_single_dims);

		md_copy(DIMS, data->dims->img_dims, out_img, img, CFL_SIZE);
	}
	if (data->conf.out_coils_steps) {

		long out_coils_dims[DIMS];
		md_copy_dims(DIMS, out_coils_dims, data->dims->sens_dims);
		long out_coils_single_dims[DIMS];
		md_copy_dims(DIMS, out_coils_single_dims, out_coils_dims);
		out_coils_dims[ITER_DIM] = data->conf.iter;

		complex float* tmp_sens = md_alloc_sameplace(DIMS, data->dims->sens_dims, CFL_SIZE, coils);
		complex float* out_sens = data->conf.out_coils + i*md_calc_size(DIMS, out_coils_single_dims);

		noir_forw_coils(data->weights, tmp_sens, coils);
		md_copy(DIMS, data->dims->sens_dims, out_sens, tmp_sens, CFL_SIZE);
		md_free(tmp_sens);

		if (NULL == data->dims->traj_dims)
			fftmod(DIMS, data->dims->sens_dims, data->conf.fft_flags, out_sens, out_sens);
	}


	i++;
}




static void proj(unsigned int D, const long dims[D],
		 complex float* optr, complex float* v1, complex float* v2)
{
#ifdef USE_CUDA
	if (cuda_ondevice(v1)) {
		error("md_zscalar is far too slow on the GPU, refusing to run...\n");
	}
#endif
	float v22 = md_zscalar_real(D, dims, v2, v2); // since it is real anyway

	complex float v12 = md_zscalar(D, dims, v1, v2) / v22;

	if (!safe_isfinite(crealf(v12)) || !safe_isfinite(cimagf(v12)) ) {

		v12 = 0.;
	}

	md_zsmul(D, dims, optr, v2, v12);
}





void noir_orthogonalize(struct noir_s* op, complex float* coils)
{

	struct noir_op_s* data = op->noir_op;
	// orthogonalization of the coil profiles
	long nmaps = data->dims->img_dims[MAPS_DIM];
	if (1L == nmaps) {
		return;
	}

	// as long as the slice dim is after the maps dim, this orthogonalization
	// will do it wrong. Therefore, we refuse to run in that case:
	assert( (1 == data->dims->img_dims[SLICE_DIM]) || (MAPS_DIM > SLICE_DIM) );

	long single_coils_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|SLICE_FLAG, single_coils_dims, data->dims->sens_dims);


	// start of coil profiles
	complex float* start_ptr = coils;

	long map_offset = md_calc_size(DIMS, single_coils_dims);

	complex float* tmp = md_alloc_sameplace(DIMS, single_coils_dims, CFL_SIZE, coils);
	complex float* proj_tmp = md_alloc_sameplace(DIMS, single_coils_dims, CFL_SIZE, coils);


	for (long map = 0L; map < nmaps; ++map) {
		complex float* map_ptr = start_ptr + map*map_offset;
		md_clear(DIMS, single_coils_dims, tmp, CFL_SIZE);
		for (long prev = 0L; prev < map; ++prev) {
			// calculate projection of current map onto previous
			// and add to tmp
			complex float* prev_map_ptr = start_ptr + prev*map_offset;

			proj(DIMS, single_coils_dims,
			     proj_tmp, map_ptr, prev_map_ptr);

			md_zadd(DIMS, single_coils_dims, tmp, tmp, proj_tmp);

		}
		md_zsub(DIMS, single_coils_dims, map_ptr, map_ptr, tmp);
	}

	md_free(tmp);
	md_free(proj_tmp);
}

