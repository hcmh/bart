/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2015-2016 Siddharth Iyer <sid8795@gmail.com>
 *
 *
 * Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M.
 * ESPIRiT - An Eigenvalue Approach to Autocalibrating Parallel MRI: Where SENSE
 * meets GRAPPA. Magn Reson Med, 71:990-1001 (2014)
 *
 * Iyer S, Ong F, Lustig M.
 * Towards A Parameter Free ESPIRiT: Soft-Weighting For Robust Coil Sensitivity Estimation.
 * Presented in the session: "New Frontiers In Image Reconstruction" at ISMRM 2016.
 * http://www.ismrm.org/16/program_files/O86.htm
 *
 */

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "linops/linop.h"
#include "linops/someops.h"
#include "num/multind.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/lapack.h"
#include "num/casorati.h"
#include "num/rand.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/resize.h"
#include "misc/debug.h"
#include "misc/utils.h"

#include "calib/calmat.h"
#include "calib/cc.h"
#include "calib/softweight.h"

#include "calib.h"

#ifdef USE_CUDA
#include "calib/calibcu.h"
#endif

#if 0
#define CALMAT_SVD
#endif

#if 0
#define FLIP
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static void eigen_herm3(int M, int N, float val[M], complex float matrix[N][N], int num_orthiter) // ordering might be different to herm2
{
	complex float mout[M][N];

	for (int li = 0; li < N; li++)
		for (int lj = 0; lj < li; lj++)
			matrix[lj][li] = conj(matrix[li][lj]);

	//mat_identity(M, N, mout);
	orthiter(M, N, num_orthiter, val, mout, matrix);

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			matrix[i][j] = mout[i][j];
}



static void md_scurve(int N, const long dims[N], float* dst, const float* src)
{
	float* tmp1 = md_alloc_sameplace(N, dims, FL_SIZE, src);
	float* tmp2 = md_alloc_sameplace(N, dims, FL_SIZE, src);

	md_sgreatequal(N, dims, tmp1, src, -1.);
	md_slessequal(N, dims, tmp2, src, 1.);

	md_mul(N, dims, tmp2, src, src);
	md_sadd(N, dims, tmp2, tmp2, 1.);
	md_div(N, dims, tmp2, src, tmp2);
	md_sadd(N, dims, tmp2, tmp2, 0.5);

	md_sgreatequal(N, dims, dst, src, 1.);
	md_fmac(N, dims, dst, tmp1, tmp2);

	md_free(tmp1);
	md_free(tmp2);
}

static void md_crop_weight_fun(int N, const long dims[N], float crth, complex float* dst, const complex float* src)
{
	md_zabs(N, dims, dst, src);

	float* tmp = md_alloc_sameplace(N, dims, FL_SIZE, src);

	md_real(N, dims, tmp, dst);

	md_sqrt(N, dims, tmp, tmp);
	md_sadd(N, dims, tmp, tmp, - crth);
	md_smul(N, dims, tmp, tmp, 1. / (1. - crth));
	md_scurve(N, dims, tmp, tmp);

	md_zcmpl_real(N, dims, dst, tmp);

	md_free(tmp);
}

static void md_crop_thresh_fun(int N, const long dims[N], float crth, complex float* dst, const complex float* src)
{
	md_zabs(N, dims, dst, src);
	md_zsgreatequal(N, dims, dst, dst, crth);
}


typedef void (*md_weight_function)(int N, const long dims[N], float crth, complex float* dst, const complex float* src);

static void md_crop_weight(int N, const long dims[N], complex float* ptr, md_weight_function fun, float crth, const complex float* map)
{
	assert(4 < N);

	long wgh_dims[N];
	md_select_dims(N, FFT_FLAGS | MAPS_FLAG, wgh_dims, dims);

	complex float* tmp = md_alloc_sameplace(N, wgh_dims, CFL_SIZE, map);

	fun(N, wgh_dims, crth, tmp, map);

	long strs[N];
	long wgh_strs[N];

	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, wgh_strs, wgh_dims, CFL_SIZE);

	md_zmul2(N, dims, strs, ptr, strs, ptr, wgh_strs, tmp);

	md_free(tmp);
}



void crop_sens(const long dims[DIMS], complex float* ptr, bool soft, float crth, const complex float* map)
{
	md_crop_weight(DIMS, dims, ptr, soft ? md_crop_weight_fun : md_crop_thresh_fun, crth, map);
}


/**
 * sure_crop - This determines the crop-threshold to use as described in the talk: "Towards A Parameter
 *	       Free ESPIRiT: Soft-Weighting For Robust Coil Sensitivity Estimation". This was given at the
 *	       session: "New Frontiers In Image Reconstruction" at ISMRM 2016.
 *
 * Parameters:
 *	var		- Estimated variance in data.
 *	evec_dims	- The eigenvector dimensions.
 *	evec_data	- The eigenvectors.
 *	eptr		- The eigenvalues.
 *	calreg_dims     - Dimension of the calibration region.
 *	calreg	        - Calibration data.
 */
static float sure_crop(float var, const long evec_dims[DIMS], complex float* evec_data, complex float* eptr, const long calreg_dims[DIMS], const complex float* calreg)
{
	assert(1 == md_calc_size(DIMS - 5, evec_dims + 5));
	assert(1 == md_calc_size(DIMS - 5, calreg_dims + 5));

	long num_maps = evec_dims[4];

	// Construct low-resolution image
	long im_dims[5];
	md_select_dims(5, 15, im_dims, evec_dims);

	complex float* im = md_alloc_sameplace(5, im_dims, CFL_SIZE, calreg);

	md_clear(5, im_dims, im, CFL_SIZE);
	md_resize_center(5, im_dims, im, calreg_dims, calreg, CFL_SIZE);

	auto lop_fft_im = linop_fftc_create(5, im_dims, FFT_FLAGS);
	auto lop_fft_evec = linop_fftc_create(5, evec_dims, FFT_FLAGS);

	linop_adjoint(lop_fft_im, 5, im_dims, im, 5, im_dims, im);

	// Temporary vector for crop dimensions
	long cropdims[5];
	md_select_dims(5, 15, cropdims, calreg_dims);
	cropdims[4] = num_maps;

	// Eigenvectors (M)
	complex float* M = md_alloc_sameplace(5, evec_dims, CFL_SIZE, calreg);

	md_copy(5, evec_dims, M, evec_data, CFL_SIZE);

	// Temporary eigenvector holder to hold low resolution maps
	complex float* LM = md_alloc_sameplace(5, evec_dims, CFL_SIZE, calreg);

	// Temporary holder for projection calreg
	complex float* TC = md_alloc_sameplace(5, calreg_dims, CFL_SIZE, calreg);

	// Temporary holder to hold low resolution calib maps
	complex float* CM = md_alloc_sameplace(5, cropdims, CFL_SIZE, calreg);

	// Eigenvalues (W)
	long W_dims[5];
	md_select_dims(5, 23, W_dims, evec_dims);

	complex float* W = md_alloc_sameplace(5, W_dims, CFL_SIZE, calreg);
	md_copy(5, W_dims, W, eptr, CFL_SIZE);

	// Place holder for the inner product result
	complex float* ip = md_alloc_sameplace(5, W_dims, CFL_SIZE, calreg);

	// Place holder for the projection result
	complex float* proj = md_alloc_sameplace(5, im_dims, CFL_SIZE, calreg);

	// Place holder for divergence term
	long div_dims[5] = { 1, 1, 1, 1, 1 };
	complex float* div = md_alloc_sameplace(5, div_dims, CFL_SIZE, calreg);

	// Calculating strides.
	long str1_ip[5];
	long str2_ip[5];
	long stro_ip[5];

	md_calc_strides(5, str1_ip, im_dims, CFL_SIZE);
	md_calc_strides(5, str2_ip, evec_dims, CFL_SIZE);
	md_calc_strides(5, stro_ip, W_dims, CFL_SIZE);

	long str1_proj[5];
	long str2_proj[5];
	long stro_proj[5];

	md_calc_strides(5, str1_proj, W_dims, CFL_SIZE);
	md_calc_strides(5, str2_proj, evec_dims, CFL_SIZE);
	md_calc_strides(5, stro_proj, im_dims, CFL_SIZE);

	long str1_div[5];
	long str2_div[5];
	long stro_div[5];

	md_calc_strides(5, str1_div, evec_dims, CFL_SIZE);
	md_calc_strides(5, str2_div, evec_dims, CFL_SIZE);
	md_calc_strides(5, stro_div, div_dims, CFL_SIZE);

	long tdims_ip[5];
	long tdims_proj[5];

	for (int i = 0; i < 5; i++) {

		assert((im_dims[i] == evec_dims[i]) || (1 == im_dims[i]) || (1 == evec_dims[i]));
		assert((W_dims[i] == evec_dims[i]) || (1 == W_dims[i]) || (1 == evec_dims[i]));

		tdims_ip[i] = (1 == im_dims[i]) ? evec_dims[i] : im_dims[i];
		tdims_proj[i] = (1 == W_dims[i]) ? evec_dims[i] : W_dims[i];
	}

	// Starting parameter sweep with SURE.
	float mse = -1.;
	float old_mse = 0.;

	float s = -0.1;
	float c = 0.99;
	long ctr1 = 0;
	long ctr2 = 0;
	

	debug_printf(DP_INFO, "---------------------------------------------\n");
	debug_printf(DP_INFO, "| CTR1 | CTR2 |  Crop  |      Est. MSE      |\n");
	debug_printf(DP_INFO, "---------------------------------------------\n");

	while (fabs(s) > 1.E-4) {

		ctr1++;

		while (   (c < 0.999)
		       && (c > 0.001)
		       && (   (ctr2 <= 1)
			   || (mse < old_mse))) {

			ctr2++;

			md_clear(5, W_dims, ip, CFL_SIZE);
			md_clear(5, im_dims, proj, CFL_SIZE);
			md_clear(5, div_dims, div, CFL_SIZE);
			md_clear(5, evec_dims, M, CFL_SIZE);
			md_clear(5, evec_dims, LM, CFL_SIZE);
			md_clear(5, calreg_dims, TC, CFL_SIZE);

			md_copy(5, evec_dims, M, evec_data, CFL_SIZE);

			old_mse = mse;
			mse = 0.;

			md_crop_weight(5, evec_dims, M, md_crop_thresh_fun, c, W);

			md_zfmacc2(5, tdims_ip, stro_ip, ip, str1_ip, im, str2_ip, M); // Projection.
			md_zfmac2(5, tdims_proj, stro_proj, proj, str1_proj, ip, str2_proj, M);

			linop_forward(lop_fft_im, 5, im_dims, proj, 5, im_dims, proj);		// Low res proj img.                             

			md_resize_center(5, calreg_dims, TC, im_dims, proj, CFL_SIZE);
			md_resize_center(5, im_dims, proj, calreg_dims, TC, CFL_SIZE);

			linop_adjoint(lop_fft_im, 5, im_dims, proj, 5, im_dims, proj);

#if 1
			complex float* diff = md_alloc_sameplace(5, im_dims, CFL_SIZE, im);
			md_zsub(5, im_dims, diff, im, proj);
			mse += powf(md_znorm(5, im_dims, diff), 2);
			md_free(diff);
#else
			for (long jdx = 0; jdx < md_calc_size(5, im_dims); jdx++)
				mse += powf(cabsf(im[jdx] - proj[jdx]), 2.);
#endif

			linop_forward(lop_fft_evec, 5, evec_dims, LM, 5, evec_dims, M);		// low-res maps .                       

			md_resize_center(5, cropdims, CM, evec_dims, LM, CFL_SIZE);
			md_resize_center(5, evec_dims, LM, cropdims, CM, CFL_SIZE);

			linop_adjoint(lop_fft_evec, 5, evec_dims, LM, 5, evec_dims, LM);

			md_zfmacc2(5, evec_dims, stro_div, div, str1_div, LM, str2_div, LM);     // Calc SURE div using low res maps.

			complex float div_cpu;
			md_copy(1, MD_DIMS(1), &div_cpu, div, CFL_SIZE);
			mse += 2. * var * crealf(div_cpu);

			if (ctr2 == 1)
				debug_printf(DP_INFO, "| %4ld | %4ld | %0.4f | %0.12e |\n", ctr1, ctr2, c, mse);
			else
				debug_printf(DP_INFO, "|      | %4ld | %0.4f | %0.12e |\n", ctr2, c, mse);

			c = c + s;
		}

		c -= s;
		ctr2 = 0;
		s = -s / 2;
		c += s;
	}

	linop_free(lop_fft_im);
	linop_free(lop_fft_evec);

	c = c + s;

	debug_printf(DP_INFO, "---------------------------------------------\n");

	md_free(im);
	md_free(TC);
	md_free(CM);
	md_free(M);
	md_free(LM);
	md_free(W);
	md_free(ip);
	md_free(proj);
	md_free(div);

	debug_printf(DP_DEBUG1, "Calculated c: %.4f\n", c);

	return c;
}




void calone(const struct ecalib_conf* conf, const long cov_dims[4], complex float* imgcov, int SN, float svals[SN], const long calreg_dims[DIMS], const complex float* data)
{
	assert(1 == md_calc_size(DIMS - 5, calreg_dims + 5));

#if 1
	long nskerns_dims[5];
	complex float* nskerns;
	compute_kernels(conf, nskerns_dims, &nskerns, SN, svals, calreg_dims, data);
#else
	long channels = calreg_dims[3];

	long kx = conf->kdims[0];
	long ky = conf->kdims[1];
	long kz = conf->kdims[2];

	long nskerns_dims[5] = { kx, ky, kz, channels, 0 };
	long N = md_calc_size(4, nskerns_dims);

	assert(N > 0);
	nskerns_dims[4] = N;

	complex float* nskerns = md_alloc(5, nskerns_dims, CFL_SIZE);

	long nr_kernels = channels;
	nskerns_dims[4] = channels;
	spirit_kernel(nskerns_dims, nskerns, calreg_dims, data);
#endif

	compute_imgcov(cov_dims, imgcov, nskerns_dims, nskerns);

	md_free(nskerns);
}








/* calculate point-wise maps 
 *
 */
void eigenmaps(const long out_dims[DIMS], complex float* optr, complex float* eptr, const complex float* imgcov2, const long msk_dims[3], const bool* msk, bool orthiter, int num_orthiter, bool ecal_usegpu)
{
#ifdef USE_CUDA
	if (ecal_usegpu) {

		//FIXME cuda version should be able to return sensitivities for a subset of image-space points
		assert(!msk);
		eigenmapscu(out_dims, optr, eptr, imgcov2, num_orthiter);
		return;
	}
#else
	assert(!ecal_usegpu);
#endif

	int channels = (int)out_dims[3];
	int maps = (int)out_dims[4];

	assert(DIMS >= 5);
	assert(1 == md_calc_size(DIMS - 5, out_dims + 5));
	assert(maps <= channels);

	int xx = (int)out_dims[0];
	int yy = (int)out_dims[1];
	int zz = (int)out_dims[2];

	float scale = 1.; // for some reason, not

	if (msk_dims) {

		assert(msk_dims[0] == xx);
		assert(msk_dims[1] == yy);
		assert(msk_dims[2] == zz);
	}

	md_clear(5, out_dims, optr, CFL_SIZE);

#pragma omp parallel for collapse(3)
	for (long k = 0; k < zz; k++) {
		for (long j = 0; j < yy; j++) {
			for (long i = 0; i < xx; i++) {

				if (!msk || msk[i + xx * (j + yy * k)])	{

					float val[channels];
					complex float cov[channels][channels];

					complex float tmp[channels * (channels + 1) / 2];

					for (long l = 0; l < channels * (channels + 1) / 2; l++)
						tmp[l] = imgcov2[((l * zz + k) * yy + j) * xx + i] / scale;

					unpack_tri_matrix(channels, cov, tmp);

					if (orthiter) 
						eigen_herm3(maps, channels, val, cov, num_orthiter);
					else 
						lapack_eig(channels, val, cov);

					for (int u = 0; u < maps; u++) {

						int ru = (orthiter ? maps : channels) - 1 - u;

						for (int v = 0; v < channels; v++)
							optr[((((u * channels + v) * zz + k) * yy + j) * xx + i)] = cov[ru][v];

						if (NULL != eptr)
							eptr[((u * zz + k) * yy + j) * xx + i] = val[ru];
					}
				}
			}
		}
	}
}









void caltwo(const struct ecalib_conf* conf, const long out_dims[DIMS], complex float* out_data, complex float* emaps, const long in_dims[4], complex float* in_data, const long msk_dims[3], const bool* msk)
{
	long xx = out_dims[0];
	long yy = out_dims[1];
	long zz = out_dims[2];

	long xh = in_dims[0];
	long yh = in_dims[1];
	long zh = in_dims[2];

	long channels = out_dims[3];
	long cosize = channels * (channels + 1) / 2;
	
	assert(DIMS >= 5);
	assert(1 == md_calc_size(DIMS - 5, out_dims + 5));
	assert(in_dims[3] == cosize);

	long cov_dims[4] = { xh, yh, zh, cosize };
	long covbig_dims[4] = { xx, yy, zz, cosize };

	assert(((xx == 1) && (xh == 1)) || (xx >= xh));
	assert(((yy == 1) && (yh == 1)) || (yy >= yh));
	assert(((zz == 1) && (zh == 1)) || (zz >= zh));

	assert((1 == xh) || (0 == xh % 2));
	assert((1 == yh) || (0 == yh % 2));
	assert((1 == zh) || (0 == zh % 2));

	complex float* imgcov2 = md_alloc(4, covbig_dims, CFL_SIZE);

	debug_printf(DP_DEBUG1, "Resize...\n");

	sinc_zeropad(4, covbig_dims, imgcov2, cov_dims, in_data);

	debug_printf(DP_DEBUG1, "Point-wise eigen-decomposition...\n");

	eigenmaps(out_dims, out_data, emaps, imgcov2, msk_dims, msk, conf->orthiter, conf->num_orthiter, conf->usegpu);

	md_free(imgcov2);
}




void calone_dims(const struct ecalib_conf* conf, long cov_dims[4], long channels)
{
	long kx = conf->kdims[0];
	long ky = conf->kdims[1];
	long kz = conf->kdims[2];

	cov_dims[0] = (1 == kx) ? 1 : (2 * kx);
	cov_dims[1] = (1 == ky) ? 1 : (2 * ky);
	cov_dims[2] = (1 == kz) ? 1 : (2 * kz);
	cov_dims[3] = channels * (channels + 1) / 2;
}



const struct ecalib_conf ecalib_defaults = {

	.kdims = {6, 6, 6},
	.threshold = 0.001,
	.numsv = -1,
	.percentsv = -1.,
	.weighting = false,
	.softcrop = false,
	.crop = 0.8,
	.orthiter = true,
	.num_orthiter = 30,
	.usegpu = false,
	.perturb = -1.,
	.intensity = false,
	.rotphase = true,
	.var = -1.,
	.automate = false,
};






void calib2(const struct ecalib_conf* conf, const long out_dims[DIMS], complex float* out_data, complex float* eptr, int SN, float svals[SN], const long calreg_dims[DIMS], const complex float* data, const long msk_dims[3], const bool* msk)
{
	long channels = calreg_dims[3];
	long maps = out_dims[4];

	assert(calreg_dims[3] == out_dims[3]);
	assert(maps <= channels);

	assert(1 == md_calc_size(DIMS - 5, out_dims + 5));
	assert(1 == md_calc_size(DIMS - 5, calreg_dims + 5));

	complex float rot[channels][channels];

	if (conf->rotphase) {

		// rotate the the phase with respect to the first principle component
		long scc_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
		scc_dims[COIL_DIM] = channels;
		scc_dims[MAPS_DIM] = channels;
		scc(scc_dims, &rot[0][0], calreg_dims, data);

	} else {

		for (int i = 0; i < channels; i++)
			for (int j = 0; j < channels; j++)
				rot[i][j] = (i == j) ? 1. : 0.;
	}


	long cov_dims[4];

	calone_dims(conf, cov_dims, channels);

	complex float* imgcov = md_alloc(4, cov_dims, CFL_SIZE);

	calone(conf, cov_dims, imgcov, SN, svals, calreg_dims, data);

	caltwo(conf, out_dims, out_data, eptr, cov_dims, imgcov, msk_dims, msk);

	/* Intensity and phase normalization similar as proposed
	 * for adaptive combine (Walsh's method) in
	 * Griswold et al., ISMRM 10:2410 (2002)
	 */

	if (conf->intensity) {

		debug_printf(DP_DEBUG1, "Normalize...\n");

		/* I think the reason this works is because inhomogeneity usually
		 * comes from only a few coil elements which are close. The l1-norm
		 * is more resilient against such outliers. -- Martin
		 */

		normalizel1(DIMS, COIL_FLAG, out_dims, out_data);
		md_zsmul(DIMS, out_dims, out_data, out_data, sqrtf((float)channels));
	}


	const complex float* data_tmp = data;
#ifdef USE_CUDA
	if (conf->usegpu)
		data_tmp = md_gpu_move(DIMS, calreg_dims, data, CFL_SIZE);
#endif

	float c = (conf->crop >= 0.) ? conf->crop : sure_crop(conf->var, out_dims, out_data, eptr, calreg_dims, data_tmp);

	if (data_tmp != data)
		md_free(data_tmp);

	debug_printf(DP_DEBUG1, "Crop maps... (c = %.2f)\n", c);

	crop_sens(out_dims, out_data, conf->softcrop, c, eptr);

	debug_printf(DP_DEBUG1, "Fix phase...\n");


	fixphase2(DIMS, out_dims, COIL_DIM, rot[0], out_data, out_data);

	md_free(imgcov);
}



void calib(const struct ecalib_conf* conf, const long out_dims[DIMS], complex float* out_data, complex float* eptr, int SN, float svals[SN], const long calreg_dims[DIMS], const complex float* data)
{
	calib2(conf, out_dims, out_data, eptr, SN, svals, calreg_dims, data, NULL, NULL);
}




static void perturb(const long dims[2], complex float* vecs, float amt)
{
	complex float* noise = md_alloc(2, dims, CFL_SIZE);

	md_gaussian_rand(2, dims, noise);

	for (long j = 0; j < dims[1]; j++) {

		float nrm = md_znorm(1, dims, noise + j * dims[0]);
		complex float val = amt / nrm;
		md_zsmul(1, dims, noise + j * dims[0], noise + j * dims[0], val);
	}

	md_zadd(2, dims, vecs, vecs, noise);

	for (long j = 0; j < dims[1]; j++) {

		float nrm = md_znorm(1, dims, vecs + j * dims[0]);
		complex float val = 1 / nrm;
		md_zsmul(1, dims, vecs + j * dims[0], vecs + j * dims[0], val);
	}

	md_free(noise);
}


static long number_of_kernels(const struct ecalib_conf* conf, long N, const float val[N])
{
	long n = 0;

	if (-1 != conf->numsv) {

		n = conf->numsv;
		assert(-1. == conf->percentsv);
		assert(-1. == conf->threshold);

	} else if (conf->percentsv != -1.) {

		n = (float)N * conf->percentsv / 100.;
		assert(-1 == conf->numsv);
		assert(-1. == conf->threshold);

	} else {

		assert(-1 == conf->numsv);
		assert(-1. == conf->percentsv);

		for (int i = 0; i < N; i++)
			if (val[i] / val[0] > sqrtf(conf->threshold))
				n++;
	}

	if (val[0] <= 0.)
		error("No signal.\n");

	debug_printf(DP_DEBUG1, "Using %ld/%ld kernels (%.2f%%, last SV: %f%s).\n", n, N, (float)n / (float)N * 100., (n > 0) ? (val[n - 1] / val[0]) : 1., conf->weighting ? ", weighted" : "");

	float tr = 0.;

	for (int i = 0; i < N; i++) {

		tr += powf(val[i], 2.);

		debug_printf(DP_DEBUG3, "SVALS %f (%f)\n", val[i], val[i] / val[0]);
	}

	debug_printf(DP_DEBUG3, "\nTRACE: %f (%f)\n", tr, tr / (float)N);

	assert(n <= N);
	return n;
}


void compute_kernels(const struct ecalib_conf* conf, long nskerns_dims[5], complex float** nskerns_ptr, int SN, float val[SN], const long caldims[DIMS], const complex float* caldata)
{
	assert(1 == md_calc_size(DIMS - 5, caldims + 5));

	nskerns_dims[0] = conf->kdims[0];
	nskerns_dims[1] = conf->kdims[1];
	nskerns_dims[2] = conf->kdims[2];
	nskerns_dims[3] = caldims[3];

	long N = md_calc_size(4, nskerns_dims);

	assert(N > 0);
	nskerns_dims[4] = N;

	complex float* nskerns = md_alloc(5, nskerns_dims, CFL_SIZE);
	*nskerns_ptr = nskerns;

	PTR_ALLOC(complex float[N][N], vec);

	assert(NULL != val);
	assert(SN == N);

	debug_printf(DP_DEBUG1, "Build calibration matrix and SVD...\n");

#ifdef CALMAT_SVD
	calmat_svd(conf->kdims, N, *vec, val, caldims, caldata);

	if (conf->weighting)
		soft_weight_singular_vectors(N, conf->var, conf->kdims, caldims, val, val);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
#ifndef FLIP
			nskerns[i * N + j] = ((*vec)[j][i]) * (conf->weighting ? val[i] : 1.);
#else
			nskerns[i * N + j] = ((*vec)[j][N - 1 - i]) * (conf->weighting ? val[N - 1 - i] : 1.);
#endif
#else
	covariance_function(conf->kdims, N, *vec, caldims, caldata);

	debug_printf(DP_DEBUG1, "Eigen decomposition... (size: %ld)\n", N);

	// we could apply Nystroem method here to speed it up

	float tmp_val[N];
	lapack_eig(N, tmp_val, *vec);

	// reverse and square root, test for smaller null to avoid NaNs
	for (int i = 0; i < N; i++)
		val[i] = (tmp_val[N - 1 - i] < 0.) ? 0. : sqrtf(tmp_val[N - 1 - i]);

	if (conf->weighting)
		soft_weight_singular_vectors(N, conf-> var, conf->kdims, caldims, val, val);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) 
#ifndef FLIP
			nskerns[i * N + j] = (*vec)[N - 1 - i][j] * (conf->weighting ? val[i] : 1.);	// flip
#else
			nskerns[i * N + j] = (*vec)[i][j] * (conf->weighting ? val[N - 1 - i] : 1.);	// flip
#endif
#endif

	if (conf->perturb > 0.) {

		long dims[2] = { N, N };
		perturb(dims, nskerns, conf->perturb);
	}

#ifndef FLIP
	nskerns_dims[4] = number_of_kernels(conf, N, val);
#else
	nskerns_dims[4] = N - number_of_kernels(conf, N, val);
#endif

	PTR_FREE(vec);
}




	
void compute_imgcov(const long cov_dims[4], complex float* imgcov, const long nskerns_dims[5], const complex float* nskerns)
{
	debug_printf(DP_DEBUG1, "Zeropad...\n");

	long xh = cov_dims[0];
	long yh = cov_dims[1];
	long zh = cov_dims[2];

	long kx = nskerns_dims[0];
	long ky = nskerns_dims[1];
	long kz = nskerns_dims[2];

	int channels = (int)nskerns_dims[3];
	int nr_kernels = (int)nskerns_dims[4];

	long imgkern_dims[5] = { xh, yh, zh, channels, nr_kernels };

	complex float* imgkern1 = md_alloc(5, imgkern_dims, CFL_SIZE);
	complex float* imgkern2 = md_alloc(5, imgkern_dims, CFL_SIZE);

	md_resize_center(5, imgkern_dims, imgkern1, nskerns_dims, nskerns, CFL_SIZE);

	// resort array

	debug_printf(DP_DEBUG1, "FFT (juggling)...\n");

	long istr[5];
	long mstr[5];

	long idim[5] = { xh, yh, zh, channels, nr_kernels };
	long mdim[5] = { nr_kernels, channels, xh, yh, zh };

	md_calc_strides(5, istr, idim, CFL_SIZE);
	md_calc_strides(5, mstr, mdim, CFL_SIZE);

	long m2str[5] = { mstr[2], mstr[3], mstr[4], mstr[1], mstr[0] };

	ifftmod(5, imgkern_dims, FFT_FLAGS, imgkern1, imgkern1);
	ifft2(5, imgkern_dims, FFT_FLAGS, m2str, imgkern2, istr, imgkern1);

	float scalesq = (float)((kx * ky * kz) * (xh * yh * zh)); // second part for FFT scaling

	md_free(imgkern1);

	debug_printf(DP_DEBUG1, "Calculate Gram matrix...\n");

	int cosize = channels * (channels + 1) / 2;

	assert(cov_dims[3] == cosize);

#pragma omp parallel for collapse(3)
	for (int k = 0; k < zh; k++) {
		for (int j = 0; j < yh; j++) {
			for (int i = 0; i < xh; i++) {

				complex float gram[cosize];
				gram_matrix2(channels, gram, nr_kernels, (const complex float (*)[nr_kernels])(imgkern2 + ((k * yh + j) * xh + i) * (channels * nr_kernels)));

#ifdef FLIP
				// add (scaled) identity matrix
				for (int i = 0, l = 0; i < channels; i++)
					for (int j = 0; j <= i; j++, l++)
						gram[l] = ((i == j) ? (kx * ky * kz) : 0.) - gram[l];
#endif
				for (int l = 0; l < cosize; l++)
					imgcov[(((l * zh) + k) * yh + j) * xh + i] = gram[l] / scalesq;
			}
		}
	}

	md_free(imgkern2);
}


