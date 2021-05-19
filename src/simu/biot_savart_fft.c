/*
 * Authors:
 * 2020 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 *
 * Fast evaluation of the z-component of the Biot-Savart law for 3D current densities
 */

#include <assert.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "num/vec3.h"

#include "biot_savart_fft.h"

//
#include "misc/io.h"
#include "misc/mmio.h"
#include <stdio.h>

#include "linops/fmac.h"
#include "linops/linop.h"
#include "linops/someops.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include <complex.h>

#include "misc/debug.h"

#define NB 4

/**
 * Scaled convolution kernel for Biot-Savart
 * ( bz_unit * fftw_scaling_factor)^-1 * ( y/|x|^3 , - x/|x|^3 )
 * @param dims current image size
 * @param vox_in can be NULL for isotropic resolution
 * @param kdims float[4] array to receive the size of the kernel
 * @param center_mask set pixels in the kernel to 0 which
 *			are max center_mask pixels away from the innermost pixel
 *			- 0 has no effect as the innermost voxel always is set to 0
 *			- 1 means a 3x3x3 box is set to 0
 */
static complex float *biot_savart_kernel(const long dims[4], const float vox_in[3], long *kdims, long center_mask)
{
	// calculate kernel size incl. padding
	assert((2 == dims[0]) || (3 == dims[0]));
	kdims[0] = dims[0];
	for (int i = 0; i < 3; i++)
		kdims[i + 1] = ((int)(dims[i + 1] * 2)) - 1;

	complex float *kern = md_calloc(NB, kdims, CFL_SIZE);

	float voxelsize[3] = {1.};
	if (vox_in != NULL)
		for (int i = 0; i < 3; i++)
			voxelsize[i] = vox_in[i];
	float fov[3];
	vox_to_fov(fov, dims+1, voxelsize);

	// 1/4Pi and fftw scaling
	float scaling = 1 / (4 * M_PI) / md_calc_size(3, kdims + 1);

	// Normalize FOV to 1 x b x c, with b,c >= 1
	float min_fov = bz_unit(dims + 1, voxelsize);
	for (int i = 0; i < 3; i++) {
		voxelsize[i] /= min_fov;
		fov[i] /= min_fov;
		// operator scaling
		scaling *= voxelsize[i];
	}
	debug_printf(DP_DEBUG1, "Rescaled Voxelsize: %f %f %f", voxelsize[0], voxelsize[1], voxelsize[2]);

	// construct kernel
	long pos[] = {0, 0, 0, 0};
	vec3_t x;
	float b;
	long p = 0;
	do {
		for (int i = 0; i < 3; i++)
			x[i] = (pos[i + 1] + 1) * voxelsize[i] - fov[i];
		b = pow(vec3_sdot(x, x), 1.5);
		bool mask = true;
		for (int i = 0; i < 3; i++)
			mask = mask && (labs(kdims[i + 1] / 2 - pos[i + 1]) <= center_mask);
		*(kern + p) = mask ? 0 : x[1] / b * scaling;
		*(kern + p + 1) = mask ? 0 : -x[0] / b * scaling;
		p += dims[0];
	} while (md_next(NB, kdims, 14, pos));

	return kern;
}



void vox_to_fov(float fov[3], const long dims[3], const float vox[3])
{
	for (int i = 0; i < 3; i++)
		fov[i] = vox[i] * dims[i];
}



void fov_to_vox(float vox[3], const long dims[3], const float fov[3])
{
	for (int i = 0; i < 3; i++)
		vox[i] = fov[i] / dims[i];
}



float bz_unit(const long dims[3], const float vox[3])
{
	float min_fov = 1;
	if (vox != NULL) {
		float fov[3];
		vox_to_fov(fov, dims, vox);
		min_fov = fov[0];
		for (int i = 0; i < 3; i++)
			min_fov = fov[i] < min_fov ? fov[i] : min_fov;
	}
	return min_fov;
}



static void fkern(const long kdims[4], complex float *kern)
{
	// shift & fft for convolution
	ifftshift(4, kdims, 14, kern, kern);
	fft(4, kdims, 14, kern, kern);
}



/**
 * Calculate the z-component of the magnetic induction B (µ0 = 1)
 * generated by a current density (A / [vox]^2)  in x,y-direction
 *
 * @param jdims size of grid on which jx,jy are sampled
 * @param vox voxelsize
 */
struct linop_s *linop_bz_create(const long jdims[NB], const float vox[3])
{
	assert((2 == jdims[0]) || (3 == jdims[0]));
	long kdims[NB];
	complex float *kernel = biot_savart_kernel(jdims, vox, kdims, 0);

	long bdims[NB], bpdims[NB];
	md_select_dims(NB, 14, bdims, jdims);
	md_select_dims(NB, 14, bpdims, kdims);

	// convolution + summation
	// fmac-operation, first dimension is squashed
	fkern(kdims, kernel);
	auto conv = linop_fmac_create(NB, kdims, 1U, 0U, 0U, kernel);

	// padding & cropping operators
	auto pad = linop_expand_create(NB, kdims, jdims);
	auto crop = linop_expand_create(NB, bdims, bpdims); // expand?? does it work?

	// fft ops
	auto f = linop_fft_create(NB, kdims, 14);
	auto finv = linop_ifft_create(NB, bpdims, 14);

	// the Chain
	long n_ops = 5;
	struct linop_s *op_chain[] = {pad, f, (struct linop_s*)conv, finv, crop}; //FIXME: we should make linops consistently const
	return linop_chainN(n_ops, op_chain);
}



/**
 * Wrapper for linop_bz_create: J (A/[vox]^2) -> B (Hz)
 * @param dims dimensions of j and b; first dimension of j must be 2 or 3
 * @param vox Voxelsize
 * @param b output variable for magnetic induction
 * @param j input current density
 */
void biot_savart_fft(const long dims[4], const float vox[3], complex float *b, const complex float *j)
{
	assert(dims[0] == 3 || dims[0] == 2);
	long odims[NB] = {1, dims[1], dims[2], dims[3]};
	auto bz = linop_bz_create(dims, vox);

	linop_forward(bz, NB, odims, b, NB, dims, j);
	md_zsmul(4, odims, b, b, bz_unit(dims+1, vox) * Hz_per_Tesla * Mu_0);

	linop_free(bz);
}



/**
 * Creates a uniform cylindric current density of 1 A, flowing in d-direction centered on a rectangular grid
 * @param dims grid dimensions
 * @param fov grid extent
 * @param r radius
 * @param h height
 * @param d direction (0, 1, 2)
 * @param out output variable; j is added
 */
void jcylinder(const long idims[4], const float fov[3], const float R, const float h, const long d, complex float *out)
{
	assert(idims[0] == 3);
	const long *dims = idims + 1;

	assert((dims[0] > 1) && (dims[1] > 1) && (dims[2] > 1));
	assert((0 <= d) && (d < 3));

	float r = 0;
	long blocksize = 1;
	// logical axes  x, y, slice direction
	// corresponding strides
	long axes[] = {0, 1, 2}, strides0[] = {idims[0], idims[0] * idims[1], idims[0] * idims[1] * idims[2]}, strides[3];
	if (1 == d) {
		axes[0] = 2;
		axes[1] = 0;
		axes[2] = 1;
	} else if (0 == d) {
		axes[0] = 1;
		axes[1] = 2;
		axes[2] = 0;
	}

	for (unsigned int i = 0; i < 3; i++)
		strides[i] = strides0[axes[i]];

	assert(fov[axes[2]] > h);

	long firstslice = (-h / 2 + fov[d] / 2) / fov[d] * dims[d];
	long lastslice = (h / 2 + fov[d] / 2) / fov[d] * dims[d];

	// iterate over cylinder
	long count = 0;
	float dx = fov[axes[0]] / (dims[axes[0]] - 1), dy = fov[axes[1]] / (dims[axes[1]] - 1);
	for (int i = 0; i < dims[axes[0]]; i++) {
		float x = i * dx - fov[axes[0]] / 2;
		for (int j = 0; j < dims[axes[1]]; j++) {
			float y = j * dy - fov[axes[1]] / 2;
			for (int k = firstslice; k <= lastslice; k++) {
				long p = i * strides[0] + j * strides[1] + k * strides[2] + d * blocksize;
				r = powf(powf(x, 2) + powf(y, 2), 0.5);
				if (r <= R) {
					out[p] = 1;
					if (k == firstslice)
						count++;
				};
			}
		}
	}

	//md_zsmul(4, idims, out, out, 1. / count );
	md_zsmul(4, idims, out, out, 1. / (count * (fov[axes[0]] / dims[axes[0]] * fov[axes[1]] / dims[axes[1]])));
}
