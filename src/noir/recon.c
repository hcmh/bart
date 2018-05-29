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

#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/thresh.h"
#include "iter/italgos.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"
#include "nlops/nlop.h"

#include "recon.h"


struct nlop_wrapper_s {

	INTERFACE(struct iter_op_data_s);

	struct noir_s* noir;
	long split;
};

DEF_TYPEID(nlop_wrapper_s);


static void orthogonalize(iter_op_data* ptr, float* _dst, const float* _src)
{
#if 0
	noir_orthogonalize(nlop_get_data(CAST_DOWN(nlop_wrapper_s, ptr)->noir), (complex float*)_dst, (const complex float*)_src);
#else
	UNUSED(_src);

	auto nlw = CAST_DOWN(nlop_wrapper_s, ptr);

	noir_orthogonalize(nlw->noir, (complex float*) _dst + nlw->split);
#endif
}


const struct noir_conf_s noir_defaults = {

	.iter = 8,
	.rvc = false,
	.usegpu = false,
	.noncart = false,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.a = 220.,
	.b = 32.,
	.pattern_for_each_coil = false,
	.sms = false,
};


void noir_recon(const struct noir_conf_s* conf, const long dims[DIMS], complex float* img, complex float* sens, complex float* ksens, const complex float* ref, const complex float* pattern, const complex float* mask, const complex float* kspace_data )
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long img1_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG, data_dims, dims);
	md_select_dims(DIMS, fft_flags, img1_dims, dims);

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);

	md_copy(DIMS, coil_dims, x + skip, ksens, CFL_SIZE);

	complex float* xref = NULL;

	if (NULL != ref) {

		xref = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);
		md_copy(1, d1, xref, ref, CFL_SIZE);
	}

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = conf->rvc;
	mconf.use_gpu = conf->usegpu;
	mconf.noncart = conf->noncart;
	mconf.fft_flags = fft_flags;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.pattern_for_each_coil = conf->pattern_for_each_coil;

	struct noir_s nl = noir_create(dims, mask, pattern, &mconf);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgtol = 0.1f;
	irgnm_conf.nlinv_legacy = true;

	struct nlop_wrapper_s nlw;

	SET_TYPEID(nlop_wrapper_s, &nlw);

	nlw.noir = &nl;
	nlw.split = skip;


	iter4_irgnm(CAST_UP(&irgnm_conf),
			nl.nlop,
			size * 2, (float*)x, (const float*)xref,
			data_size * 2, (const float*)kspace_data,
			(struct iter_op_s){ orthogonalize, CAST_UP(&nlw)});

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);
	md_copy(DIMS, coil_dims, ksens, x + skip, CFL_SIZE);


#ifdef USE_CUDA
	if (conf->usegpu) {

		noir_forw_coils(nl.linop, x + skip, x + skip);
		md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
	} else
#endif
		noir_forw_coils(nl.linop, sens, x + skip);

	fftmod(DIMS, coil_dims, fft_flags, sens, sens);


	nlop_free(nl.nlop);

	md_free(x);
	md_free(xref);
}

// Initialize dimensions and strides
void ds_init(struct ds_s* in, size_t size)
{
	md_select_dims(DIMS, ~TIME_FLAG, in->dims_singleFrame, in->dims_full);
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG, in->dims_output_singleFrame, in->dims_full);
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG|TIME_FLAG, in->dims_output, in->dims_full);
	md_select_dims(DIMS, ~SLICE_FLAG, in->dims_singlePart, in->dims_full);
	md_select_dims(DIMS, ~(TIME_FLAG|SLICE_FLAG), in->dims_singleFramePart, in->dims_full);

	md_calc_strides(DIMS, in->strs_full, in->dims_full, size);
	md_calc_strides(DIMS, in->strs_singleFrame, in->dims_singleFrame, size);
	md_calc_strides(DIMS, in->strs_singlePart, in->dims_singlePart, size);
	md_calc_strides(DIMS, in->strs_singleFramePart, in->dims_singleFramePart, size);
	md_calc_strides(DIMS, in->strs_output, in->dims_output, size);
	md_calc_strides(DIMS, in->strs_output_singleFrame, in->dims_output_singleFrame, size);

}

// Normalization of PSF and scaling of k-space
void scale_psf_k(struct ds_s* pat_s, complex float* pattern, struct ds_s* k_s, complex float* kspace_data, struct ds_s* traj_s, complex float* traj)
{

		/* PSF
		* Since for each frame we can have a different number of spokes,
		* some spoke-lines are empty in certain frames. To ensure
		* adequate normalization we have to calculate how many spokes are there
		* in each frame and build the inverse
		*
		* Basic idea:
		* Summation of READ_DIM and PHS1_DIM:
		* If the result is zero the spoke-line was empty
		*/

		long traj_dims2[DIMS]; // Squashed trajectory array
		md_copy_dims(DIMS, traj_dims2, traj_s->dims_full);
		traj_dims2[READ_DIM] = 1;
		traj_dims2[PHS1_DIM] = 1;
		complex float* traj2= md_alloc(DIMS, traj_dims2, CFL_SIZE);
		md_zrss(DIMS, traj_s->dims_full, READ_FLAG|PHS1_FLAG, traj2, traj);
		md_zdiv(DIMS, traj_dims2, traj2, traj2, traj2); // Normalize each non-zero element to one

		/* Sum the ones (non-zero elements) to get
		* number of spokes in each cardiac frame
		*/
		struct ds_s* no_spf_s = (struct ds_s*) malloc(sizeof(struct ds_s));
		md_copy_dims(DIMS, no_spf_s->dims_full, traj_dims2);
		no_spf_s->dims_full[PHS2_DIM] = 1;
		ds_init(no_spf_s, CFL_SIZE);

		complex float* no_spf = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
		md_clear(DIMS, no_spf_s->dims_full, no_spf, CFL_SIZE);
		md_zrss(DIMS, traj_dims2, PHS2_FLAG, no_spf, traj2);
		md_zspow(DIMS, no_spf_s->dims_full, no_spf, no_spf, 2); // no_spf contains the number of spokes in each frame and partition

		// Inverse (inv_no_spf contains inverse of number of spokes in each frame/partition)
		complex float* inv_no_spf = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
		md_zfill(DIMS, no_spf_s->dims_full, inv_no_spf, 1.);
		md_zdiv(DIMS, no_spf_s->dims_full, inv_no_spf, inv_no_spf, no_spf);


		// Multiply PSF
		md_zmul2(DIMS, pat_s->dims_full, pat_s->strs_full, pattern, pat_s->strs_full, pattern, no_spf_s->strs_full, inv_no_spf);
		dump_cfl("PSF", DIMS, pat_s->dims_full, pattern);

		/* k
		 * Scaling of k-space (depending on total [= all partitions] number of spokes per frame)
		 * Normalization is not performed here)
		 */

		// Sum spokes in all partitions
		complex float* no_spf_tot = md_alloc(DIMS, no_spf_s->dims_singlePart, CFL_SIZE);
		md_zsum(DIMS, no_spf_s->dims_full, SLICE_FLAG, no_spf_tot, no_spf);

		// Extract first frame
		complex float* no_sp_1stFrame_tot = md_alloc(DIMS, no_spf_s->dims_singleFramePart, CFL_SIZE);
		long posF[DIMS] = { 0 };
		md_copy_block(DIMS, posF, no_spf_s->dims_singleFramePart, no_sp_1stFrame_tot, no_spf_s->dims_singlePart, no_spf_tot, CFL_SIZE);

		complex float* ksp_scaleFactor = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
		md_clear(DIMS, no_spf_s->dims_full, ksp_scaleFactor, CFL_SIZE);

		complex float* inv_no_spf_tot = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
		md_zfill(DIMS, no_spf_s->dims_singlePart, inv_no_spf_tot, 1.);
		md_zdiv(DIMS, no_spf_s->dims_singlePart, inv_no_spf_tot, inv_no_spf_tot, no_spf_tot);
		md_zmul2(DIMS, no_spf_s->dims_full, no_spf_s->strs_full, ksp_scaleFactor, no_spf_s->strs_singlePart, inv_no_spf_tot, no_spf_s->strs_singleFramePart, no_sp_1stFrame_tot);

		md_zmul2(DIMS, k_s->dims_full, k_s->strs_full, kspace_data, k_s->strs_full, kspace_data, no_spf_s->strs_full, ksp_scaleFactor);

		free(no_spf_s);
		md_free(no_spf_tot);
		md_free(inv_no_spf_tot);
		md_free(ksp_scaleFactor);
		md_free(no_sp_1stFrame_tot);

		md_free(traj2);
		md_free(no_spf);
		md_free(inv_no_spf);

}
