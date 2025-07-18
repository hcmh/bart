/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/vecops_strided.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/reduce_md_wrapper.h"
#include "num/gpukrnls.h"
#include "num/gpu_conv.h"
#include "num/gpuops.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"

static bool test_im2col_loop_in(void)
{


	enum { N = 5 };

	long idims[N] = { 1, 16, 4, 4, 1 };
	long kdims[N] = { 16, 16, 3, 3, 1 };
	long odims[N] = { 16, 1, 2, 2, 1 };

	complex float* in_cpu = md_alloc(N, idims, CFL_SIZE);

	md_gaussian_rand(N, idims, in_cpu);


	complex float* in_gpu = md_alloc_gpu(N, idims, CFL_SIZE);

	md_copy(N, idims, in_gpu, in_cpu, CFL_SIZE);


	long dims_mat[N + 3]; // (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(N, dims_mat, kdims);
	md_copy_dims(3, dims_mat + N, odims + 2);


	long idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(N + 3, ~1ul , idims_mat, dims_mat);


	long istrs_mat[N + 3];

	md_copy_strides(N, istrs_mat, MD_STRIDES(N, idims, CFL_SIZE));
	md_copy_strides(3, istrs_mat + N, MD_STRIDES(N, idims, CFL_SIZE) + 2);


	complex float* imat_cpu = md_alloc(N + 3, idims_mat, CFL_SIZE);

	md_copy2(N + 3, idims_mat, MD_STRIDES(N + 3, idims_mat, CFL_SIZE), imat_cpu, istrs_mat, in_cpu, CFL_SIZE);


	complex float* imat_gpu = md_alloc_gpu(N + 3, idims_mat, CFL_SIZE);

	cuda_im2col(imat_gpu, in_gpu, odims, idims, kdims, NULL, NULL);

	complex float* imat_gpu_cpu = md_alloc(N + 3, idims_mat, CFL_SIZE);

	md_copy(N + 3, idims_mat, imat_gpu_cpu, imat_gpu, CFL_SIZE);
	

	float err = md_zrmse(N + 3, idims_mat, imat_gpu_cpu, imat_cpu);
	debug_printf(DP_DEBUG1, "%f, %f, %f\n", err, md_zrms(N + 3, idims_mat, imat_cpu), md_zrms(N + 3, idims_mat, imat_gpu_cpu));

	md_free(in_cpu);
	md_free(in_gpu);
	md_free(imat_cpu);
	md_free(imat_gpu);
	md_free(imat_gpu_cpu);

	UT_RETURN_ASSERT(err < 1.e-8);
}

UT_GPU_REGISTER_TEST(test_im2col_loop_in);


static bool test_im2col_loop_out(void)
{
	enum { N = 5 };

	long idims[N] = { 1, 4, 4, 4, 1 };
	long kdims[N] = { 4, 4, 3, 3, 1 };
	long odims[N] = { 4, 1, 2, 2, 1 };

	complex float* in_cpu = md_alloc(N, idims, CFL_SIZE);

	md_gaussian_rand(N, idims, in_cpu);


	complex float* in_gpu = md_alloc_gpu(N, idims, CFL_SIZE);

	md_copy(N, idims, in_gpu, in_cpu, CFL_SIZE);


	long dims_mat[N + 3]; // (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(N, dims_mat, kdims);
	md_copy_dims(3, dims_mat + N, odims + 2);

	long idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(N + 3, ~1ul , idims_mat, dims_mat);


	long istrs_mat[N + 3];

	md_copy_strides(N, istrs_mat, MD_STRIDES(N, idims, CFL_SIZE));
	md_copy_strides(3, istrs_mat + N, MD_STRIDES(N, idims, CFL_SIZE) + 2);


	complex float* imat_cpu = md_alloc(N + 3, idims_mat, CFL_SIZE);

	md_copy2(N + 3, idims_mat, MD_STRIDES(N + 3, idims_mat, CFL_SIZE), imat_cpu, istrs_mat, in_cpu, CFL_SIZE);


	complex float* imat_gpu = md_alloc_gpu(N + 3, idims_mat, CFL_SIZE);

	cuda_im2col(imat_gpu, in_gpu, odims, idims, kdims, NULL, NULL);


	complex float* imat_gpu_cpu = md_alloc(N + 3, idims_mat, CFL_SIZE);

	md_copy(N + 3, idims_mat, imat_gpu_cpu, imat_gpu, CFL_SIZE);
	

	float err = md_zrmse(N + 3, idims_mat, imat_gpu_cpu, imat_cpu);
	debug_printf(DP_DEBUG1, "%f, %f, %f\n", err, md_zrms(N + 3, idims_mat, imat_cpu), md_zrms(N + 3, idims_mat, imat_gpu_cpu));

	md_free(in_cpu);
	md_free(in_gpu);
	md_free(imat_cpu);
	md_free(imat_gpu);
	md_free(imat_gpu_cpu);

	UT_RETURN_ASSERT(err < 1.e-8);
}

UT_GPU_REGISTER_TEST(test_im2col_loop_out);


static bool test_im2col_adj(void)
{
	enum { N = 5 };

	long idims[N] = { 1, 4, 4, 4, 1 };
	long kdims[N] = { 4, 4, 3, 3, 1 };
	long odims[N] = { 4, 1, 2, 2, 1 };


	complex float* in_cpu = md_alloc(N, idims, CFL_SIZE);

	md_gaussian_rand(N, idims, in_cpu);


	complex float* in_gpu = md_alloc_gpu(N, idims, CFL_SIZE);

	md_copy(N, idims, in_gpu, in_cpu, CFL_SIZE);

	long dims_mat[N + 3]; // (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(N, dims_mat, kdims);
	md_copy_dims(3, dims_mat + N, odims + 2);


	long idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(N + 3, ~1ul , idims_mat, dims_mat);


	long istrs_mat[N + 3];

	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, CFL_SIZE));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, CFL_SIZE) + 2);


	complex float* imat_cpu = md_alloc(N + 3, idims_mat, CFL_SIZE);

	md_gaussian_rand(N + 3, idims_mat, imat_cpu);


	complex float* imat_gpu = md_alloc_gpu(N + 3, idims_mat, CFL_SIZE);

	md_copy(N + 3, idims_mat, imat_gpu, imat_cpu, CFL_SIZE);


	md_zadd2(N + 3, idims_mat, istrs_mat, in_cpu, istrs_mat, in_cpu, MD_STRIDES(N + 3, idims_mat, CFL_SIZE), imat_cpu);
	cuda_im2col_transp(in_gpu, imat_gpu, odims, idims, kdims, NULL, NULL);


	complex float* in_gpu_cpu = md_alloc(5, idims, CFL_SIZE);

	md_copy(5, idims, in_gpu_cpu, in_gpu, CFL_SIZE);
	

	float err = md_znrmse(5, idims, in_gpu_cpu, in_cpu);
	debug_printf(DP_DEBUG1, "%f, %f, %f\n", err, md_zrms(5, idims, in_cpu), md_zrms(5, idims, in_gpu_cpu));

	md_free(in_cpu);
	md_free(in_gpu);
	md_free(imat_cpu);
	md_free(imat_gpu);
	md_free(in_gpu_cpu);

	UT_RETURN_ASSERT(err < 1.e-6);
}

UT_GPU_REGISTER_TEST(test_im2col_adj);

