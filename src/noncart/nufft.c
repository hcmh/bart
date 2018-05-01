/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Frank Ong <frankong@berkeley.edu>
 * 2014-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/fft.h"
#include "num/shuffle.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "noncart/grid.h"

#include "nufft.h"

#define FFT_FLAGS (MD_BIT(0)|MD_BIT(1)|MD_BIT(2))

struct nufft_conf_s nufft_conf_defaults = {

	.toeplitz = true,
	.pcycle = false,
};

#include "nufft_priv.h"

DEF_TYPEID(nufft_data);


static void nufft_free_data(const linop_data_t* data);
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src);


static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src);
static void toeplitz_mult_pcycle(const struct nufft_data* data, complex float* dst, const complex float* src);
static complex float* compute_linphases(int N, long lph_dims[N + 1], const long img_dims[N]);
static complex float* compute_psf2(int N, const long psf_dims[N + 1], const long trj_dims[N], const complex float* traj, const complex float* weights);


/**
 * NUFFT operator initialization
 */
struct linop_s* nufft_create(unsigned int N,			///< Number of dimension
			     const long ksp_dims[N],		///< kspace dimension
			     const long cim_dims[N],		///< Coil images dimension
			     const long traj_dims[N],		///< Trajectory dimension
			     const complex float* traj,		///< Trajectory
			     const complex float* weights,	///< Weights, ex, soft-gating or density compensation
			     struct nufft_conf_s conf)		///< NUFFT configuration options

{
	PTR_ALLOC(struct nufft_data, data);
	SET_TYPEID(nufft_data, data);

	data->N = N;
	data->traj = traj;
	data->conf = conf;
	data->flags = FFT_FLAGS;

	data->width = 3.;
	data->beta = calc_beta(2., data->width);

	// dim 0 must be transformed (we treat this special in the trajectory)
	assert(MD_IS_SET(data->flags, 0));
	assert(md_check_bounds(N, ~data->flags, ksp_dims, cim_dims));
	assert(md_check_bounds(N, ~data->flags, cim_dims, ksp_dims));

	// extend internal dimensions by one for linear phases
	unsigned int ND = N + 1;

	data->ksp_dims = *TYPE_ALLOC(long[ND]);
	data->cim_dims = *TYPE_ALLOC(long[ND]);
	data->cml_dims = *TYPE_ALLOC(long[ND]);
	data->img_dims = *TYPE_ALLOC(long[ND]);
	data->trj_dims = *TYPE_ALLOC(long[ND]);
	data->lph_dims = *TYPE_ALLOC(long[ND]);
	data->psf_dims = *TYPE_ALLOC(long[ND]);
	data->wgh_dims = *TYPE_ALLOC(long[ND]);

	data->ksp_strs = *TYPE_ALLOC(long[ND]);
	data->cim_strs = *TYPE_ALLOC(long[ND]);
	data->cml_strs = *TYPE_ALLOC(long[ND]);
	data->img_strs = *TYPE_ALLOC(long[ND]);
	data->trj_strs = *TYPE_ALLOC(long[ND]);
	data->lph_strs = *TYPE_ALLOC(long[ND]);
	data->psf_strs = *TYPE_ALLOC(long[ND]);
	data->wgh_strs = *TYPE_ALLOC(long[ND]);
	
	data->rlph_dims = *TYPE_ALLOC(long[ND]);
	data->rpsf_dims = *TYPE_ALLOC(long[ND]);
	data->rcml_dims = *TYPE_ALLOC(long[ND]);
	data->rlph_strs = *TYPE_ALLOC(long[ND]);
	data->rpsf_strs = *TYPE_ALLOC(long[ND]);
	data->rcml_strs = *TYPE_ALLOC(long[ND]);

	md_copy_dims(N, data->cim_dims, cim_dims);
	data->cim_dims[N] = 1;

	md_copy_dims(N, data->ksp_dims, ksp_dims);
	data->ksp_dims[N] = 1;


	md_select_dims(ND, data->flags, data->img_dims, data->cim_dims);

	assert(bitcount(data->flags) == traj_dims[0]);

	long chk_dims[N];
	md_select_dims(N, ~data->flags, chk_dims, traj_dims);
	assert(md_check_compat(N, ~0ul, chk_dims, ksp_dims));
	assert(md_check_bounds(N, ~0ul, chk_dims, ksp_dims));


	md_copy_dims(N, data->trj_dims, traj_dims);
	data->trj_dims[N] = 1;


	md_calc_strides(ND, data->cim_strs, data->cim_dims, CFL_SIZE);
	md_calc_strides(ND, data->img_strs, data->img_dims, CFL_SIZE);
	md_calc_strides(ND, data->trj_strs, data->trj_dims, CFL_SIZE);
	md_calc_strides(ND, data->ksp_strs, data->ksp_dims, CFL_SIZE);


	data->weights = NULL;

	if (NULL != weights) {

		md_select_dims(N, ~MD_BIT(0), data->wgh_dims, data->trj_dims);
		data->wgh_dims[N] = 1;

		md_calc_strides(ND, data->wgh_strs, data->wgh_dims, CFL_SIZE);

		complex float* tmp = md_alloc(ND, data->wgh_dims, CFL_SIZE);
		md_copy(ND, data->wgh_dims, tmp, weights, CFL_SIZE);
		data->weights = tmp;
	}


	complex float* roll = md_alloc(ND, data->img_dims, CFL_SIZE);
	rolloff_correction(2., data->width, data->beta, data->img_dims, roll);
	data->roll = roll;


	complex float* linphase = compute_linphases(N, data->lph_dims, data->img_dims);

	md_calc_strides(ND, data->lph_strs, data->lph_dims, CFL_SIZE);

	if (!conf.toeplitz)
		md_zmul2(ND, data->lph_dims, data->lph_strs, linphase, data->lph_strs, linphase, data->img_strs, data->roll);


	fftmod(ND, data->lph_dims, data->flags, linphase, linphase);
	fftscale(ND, data->lph_dims, data->flags, linphase, linphase);

	float scale = 1.;
	for (int i = 0; i < (int)N; i++)
		scale *= ((data->lph_dims[i] > 1) && (MD_IS_SET(data->flags, i))) ? 0.5 : 1.;

	md_zsmul(ND, data->lph_dims, linphase, linphase, scale);


	complex float* fftm = md_alloc(ND, data->img_dims, CFL_SIZE);
	md_zfill(ND, data->img_dims, fftm, 1.);
	fftmod(ND, data->img_dims, data->flags, fftm, fftm);
	data->fftmod = fftm;



	data->linphase = linphase;
	data->psf = NULL;
#ifdef USE_CUDA
	data->linphase_gpu = NULL;
	data->psf_gpu = NULL;
	data->grid_gpu = NULL;
#endif
	if (conf.toeplitz) {

		debug_printf(DP_DEBUG1, "NUFFT: Toeplitz mode\n");

		md_copy_dims(ND, data->psf_dims, data->lph_dims);

		for (int i = 0; i < (int)N; i++)
			if (!MD_IS_SET(data->flags, i))
				data->psf_dims[i] = data->trj_dims[i];

		md_calc_strides(ND, data->psf_strs, data->psf_dims, CFL_SIZE);
		data->psf = compute_psf2(N, data->psf_dims, data->trj_dims, data->traj, data->weights);
	}


	md_copy_dims(ND, data->cml_dims, data->cim_dims);
	data->cml_dims[N + 0] = data->lph_dims[N + 0];

	md_calc_strides(ND, data->cml_strs, data->cml_dims, CFL_SIZE);


	data->cm2_dims = *TYPE_ALLOC(long[ND]);
	// !
	md_copy_dims(ND, data->cm2_dims, data->cim_dims);

	for (int i = 0; i < (int)N; i++)
		if (MD_IS_SET(data->flags, i))
			data->cm2_dims[i] = (1 == cim_dims[i]) ? 1 : (2 * cim_dims[i]);



	data->grid = md_alloc(ND, data->cml_dims, CFL_SIZE);

	data->fft_op = linop_fft_create(ND, data->cml_dims, data->flags);

	if (conf.pcycle) {

		debug_printf(DP_DEBUG1, "NUFFT: Pcycle Mode\n");
		data->cycle = 0;
		data->cfft_op = linop_fft_create(N, data->cim_dims, data->flags);
	}


	return linop_create(N, ksp_dims, N, cim_dims,
			CAST_UP(PTR_PASS(data)), nufft_apply, nufft_apply_adjoint, nufft_apply_normal, NULL, nufft_free_data);
}




static complex float* compute_linphases(int N, long lph_dims[N + 1], const long img_dims[N + 1])
{
	unsigned long flags = FFT_FLAGS;
	int T = bitcount(flags);
	float shifts[1 << T][T];

	int s = 0;
	for(int i = 0; i < (1 << T); i++) {

		bool skip = false;

		for(int j = 0; j < T; j++) {

			shifts[s][j] = 0.;

			if (MD_IS_SET(i, j)) {

				skip = skip || (1 == img_dims[j]);
				shifts[s][j] = -0.5;
			}
		}

		if (!skip)
			s++;
	}

	int ND = N + 1;
	md_select_dims(ND, flags, lph_dims, img_dims);
	lph_dims[N] = s;

	complex float* linphase = md_alloc(ND, lph_dims, CFL_SIZE);

	for(int i = 0; i < s; i++) {

		float shifts2[ND];
		for (int j = 0; j < ND; j++)
			shifts2[j] = 0.;

		for (int j = 0, t = 0; j < N; j++)
			if (MD_IS_SET(flags, j))
				shifts2[j] = shifts[i][t++];

		linear_phase(ND, img_dims, shifts2, 
				linphase + i * md_calc_size(ND, img_dims));
	}

	return linphase;
}


complex float* compute_psf(unsigned int N, const long img2_dims[N], const long trj_dims[N], const complex float* traj, const complex float* weights)
{      
	long ksp_dims1[N];
	md_select_dims(N, ~MD_BIT(0), ksp_dims1, trj_dims);

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.toeplitz = false;	// avoid infinite loop

	struct linop_s* op2 = nufft_create(N, ksp_dims1, img2_dims, trj_dims, traj, NULL, conf);

	complex float* ones = md_alloc(N, ksp_dims1, CFL_SIZE);
	md_zfill(N, ksp_dims1, ones, 1.);

	if (NULL != weights) {

		md_zmul(N, ksp_dims1, ones, ones, weights);
		md_zmulc(N, ksp_dims1, ones, ones, weights);
	}

	complex float* psft = md_alloc(N, img2_dims, CFL_SIZE);

	linop_adjoint_unchecked(op2, psft, ones);

	md_free(ones);
	linop_free(op2);

	return psft;
}


static complex float* compute_psf2(int N, const long psf_dims[N + 1], const long trj_dims[N + 1], const complex float* traj, const complex float* weights)
{
	int ND = N + 1;
	unsigned long flags = FFT_FLAGS;

	long img_dims[ND];
	long img_strs[ND];

	md_select_dims(ND, ~MD_BIT(N + 0), img_dims, psf_dims);
	md_calc_strides(ND, img_strs, img_dims, CFL_SIZE);

	// PSF 2x size

	long img2_dims[ND];
	long img2_strs[ND];

	md_copy_dims(ND, img2_dims, img_dims);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			img2_dims[i] = (1 == img_dims[i]) ? 1 : (2 * img_dims[i]);

	md_calc_strides(ND, img2_strs, img2_dims, CFL_SIZE);

	complex float* traj2 = md_alloc(ND, trj_dims, CFL_SIZE);
	md_zsmul(ND, trj_dims, traj2, traj, 2.);

	complex float* psft = compute_psf(ND, img2_dims, trj_dims, traj2, weights);
	md_free(traj2);

	fftuc(ND, img2_dims, flags, psft, psft);

	float scale = 1.;
	for (int i = 0; i < N; i++)
		scale *= ((img2_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 4. : 1.;

	md_zsmul(ND, img2_dims, psft, psft, scale);

	// reformat

	complex float* psf = md_alloc(ND, psf_dims, CFL_SIZE);

	long factors[N];

	for (int i = 0; i < N; i++)
		factors[i] = ((img_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 2 : 1;

	md_decompose(N + 0, factors, psf_dims, psf, img2_dims, psft, CFL_SIZE);

	md_free(psft);
	return psf;
}





static void nufft_free_data(const linop_data_t* _data)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

	xfree(data->ksp_dims);
	xfree(data->cim_dims);
	xfree(data->cml_dims);
	xfree(data->img_dims);
	xfree(data->trj_dims);
	xfree(data->lph_dims);
	xfree(data->psf_dims);
	xfree(data->wgh_dims);

	xfree(data->ksp_strs);
	xfree(data->cim_strs);
	xfree(data->cml_strs);
	xfree(data->img_strs);
	xfree(data->trj_strs);
	xfree(data->lph_strs);
	xfree(data->psf_strs);
	xfree(data->wgh_strs);

	xfree(data->rlph_dims);
	xfree(data->rpsf_dims);
	xfree(data->rcml_dims);
	xfree(data->rlph_strs);
	xfree(data->rpsf_strs);
	xfree(data->rcml_strs);

	xfree(data->cm2_dims);

	md_free(data->grid);
	md_free(data->linphase);
	md_free(data->psf);
	md_free(data->fftmod);
	md_free(data->weights);
	md_free(data->roll);

#ifdef USE_CUDA
	md_free(data->linphase_gpu);
	md_free(data->psf_gpu);
	md_free(data->grid_gpu);
#endif
	linop_free(data->fft_op);
	if (data->conf.pcycle)
		linop_free(data->cfft_op);

	xfree(data);
}




// Forward: from image to kspace
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

#ifdef USE_CUDA
	assert(!cuda_ondevice(src));
#endif
	assert(!data->conf.toeplitz); // if toeplitz linphase has no roll, so would need to be added

	int ND = data->N + 1;

	md_zmul2(ND, data->cml_dims, data->cml_strs, data->grid, data->cim_strs, src, data->lph_strs, data->linphase);
	linop_forward(data->fft_op, ND, data->cml_dims, data->grid, ND, data->cml_dims, data->grid);
	md_zmul2(ND, data->cml_dims, data->cml_strs, data->grid, data->cml_strs, data->grid, data->img_strs, data->fftmod);

	md_clear(ND, data->ksp_dims, dst, CFL_SIZE);

	complex float* gridX = md_alloc(data->N, data->cm2_dims, CFL_SIZE);

	long factors[data->N];

	for (int i = 0; i < (int)data->N; i++)
		factors[i] = ((data->img_dims[i] > 1) && MD_IS_SET(data->flags, i)) ? 2 : 1;

	md_recompose(data->N, factors, data->cm2_dims, gridX, data->cml_dims, data->grid, CFL_SIZE);

	grid2H(2., data->width, data->beta, ND, data->trj_dims, data->traj, data->ksp_dims, dst, data->cm2_dims, gridX);

	md_free(gridX);

	if (NULL != data->weights)
		md_zmul2(data->N, data->ksp_dims, data->ksp_strs, dst, data->ksp_strs, dst, data->wgh_strs, data->weights);
}


// Adjoint: from kspace to image
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

#ifdef USE_CUDA
	assert(!cuda_ondevice(src));
#endif
	int ND = data->N + 1;

	complex float* gridX = md_calloc(data->N, data->cm2_dims, CFL_SIZE);

	complex float* wdat = NULL;

	if (NULL != data->weights) {

		wdat = md_alloc(data->N, data->ksp_dims, CFL_SIZE);
		md_zmulc2(data->N, data->ksp_dims, data->ksp_strs, wdat, data->ksp_strs, src, data->wgh_strs, data->weights);
		src = wdat;
	}

	grid2(2., data->width, data->beta, ND, data->trj_dims, data->traj, data->cm2_dims, gridX, data->ksp_dims, src);

	md_free(wdat);

	long factors[data->N];

	for (int i = 0; i < (int)data->N; i++)
		factors[i] = ((data->img_dims[i] > 1) && MD_IS_SET(data->flags, i)) ? 2 : 1;

	md_decompose(data->N, factors, data->cml_dims, data->grid, data->cm2_dims, gridX, CFL_SIZE);
	md_free(gridX);
	md_zmulc2(ND, data->cml_dims, data->cml_strs, data->grid, data->cml_strs, data->grid, data->img_strs, data->fftmod);
	linop_adjoint(data->fft_op, ND, data->cml_dims, data->grid, ND, data->cml_dims, data->grid);

	md_clear(ND, data->cim_dims, dst, CFL_SIZE);
	md_zfmacc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, data->grid, data->lph_strs, data->linphase);

	if (data->conf.toeplitz)
		md_zmul2(ND, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->img_strs, data->roll);
}



/** 
 *
 */
static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

	if (data->conf.toeplitz) {

		if (data->conf.pcycle)
			toeplitz_mult_pcycle(data, dst, src);
		else
			toeplitz_mult(data, dst, src);

	} else {

		complex float* tmp_ksp = md_alloc(data->N + 1, data->ksp_dims, CFL_SIZE);

		nufft_apply(_data, tmp_ksp, src);
		nufft_apply_adjoint(_data, dst, tmp_ksp);

		md_free(tmp_ksp);
	}
}



static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	unsigned int ND = data->N + 1;

	const complex float* linphase = data->linphase;
	const complex float* psf = data->psf;
	complex float* grid = data->grid;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->linphase_gpu)
			((struct nufft_data*)data)->linphase_gpu = md_gpu_move(ND, data->lph_dims, data->linphase, CFL_SIZE);

		if (NULL == data->psf_gpu)
			((struct nufft_data*)data)->psf_gpu = md_gpu_move(ND, data->psf_dims, data->psf, CFL_SIZE);

		if (NULL == data->grid_gpu)
			((struct nufft_data*)data)->grid_gpu = md_gpu_move(ND, data->cml_dims, data->grid, CFL_SIZE);

		linphase = data->linphase_gpu;
		psf = data->psf_gpu;
		grid = data->grid_gpu;
	}
#endif
	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, linphase);

	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);
	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->psf_strs, psf);
	linop_adjoint(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	md_clear(ND, data->cim_dims, dst, CFL_SIZE);
	md_zfmacc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid, data->lph_strs, linphase);
}

static void toeplitz_mult_pcycle(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	unsigned int ncycles = data->lph_dims[data->N];
        ((struct nufft_data*) data)->cycle = (data->cycle + 1) % ncycles;
	
	const complex float* clinphase = data->linphase + data->cycle * md_calc_size(data->N, data->lph_dims);
	const complex float* cpsf = data->psf + data->cycle * md_calc_size(data->N, data->psf_dims);
	complex float* grid = data->grid;

	md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, src, data->img_strs, clinphase);

	linop_forward(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);
	md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, grid, data->img_strs, cpsf);
	linop_adjoint(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);

	md_zmulc2(data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, grid, data->img_strs, clinphase);
}



/**
 * Estimate image dimensions from trajectory
 */
void estimate_im_dims(int N, unsigned long flags, long dims[N], const long tdims[N], const complex float* traj)
{
	int T = tdims[0];

	assert(T == (int)bitcount(flags));

	float max_dims[T];
	for (int i = 0; i < T; i++)
		max_dims[i] = 0.;

	for (long i = 0; i < md_calc_size(N - 1, tdims + 1); i++)
		for(int j = 0; j < tdims[0]; j++)
			max_dims[j] = MAX(cabsf(traj[j + tdims[0] * i]), max_dims[j]);

	for (int j = 0, t = 0; j < N; j++) {

		dims[j] = 1;

		if (MD_IS_SET(flags, j)) {

			dims[t] = (0. == max_dims[t]) ? 1 : (2 * (long)((2. * max_dims[t] + 1.5) / 2.));
			t++;
		}
	}
}



