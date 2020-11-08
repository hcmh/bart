/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "num/multind.h"

#include "reduce_md_wrapper.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpu_reduce.h"
#endif

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {0, 1} or {0}
 * @param optr
 * @param istr1 must be of the form {0, 1} or {0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]} or {1}
 * @param iptr1 
 **/
void reduce_zadd_inner_gpu(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], complex float* optr, const long istr1[__VLA(N)], const complex float* iptr1, const long istr2[__VLA(N)], const complex float* iptr2)
{
	long size = 8;

	assert((2 == N) || (1 == N));
	assert((0 == ostr[0]));
	assert((0 == istr1[0]));
	assert((size == istr2[0]));

	if ((2 == N) && (1 != dims[1])){

		assert((0 == ostr[0]) && (size == ostr[1]));
		assert((0 == istr1[0]) && (size == istr1[1]));
		assert((size == istr2[0]) && (size * dims[0] == istr2[1]));
	}

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_zadd_inner(dims[0], (2 == N) ? dims[1] : 1, optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {1, 0}
 * @param optr
 * @param istr1 must be of the form {1, 0}
 * @param iptr1 must equal optr
 * @param istr1 must be of the form {1, dim[0]}
 * @param iptr1 
 **/
void reduce_zadd_outer_gpu(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], complex float* optr, const long istr1[__VLA(N)], const complex float* iptr1, const long istr2[__VLA(N)], const complex float* iptr2)
{
	long size = 8;

	assert(2 == N) ;
	assert(((1 == dims[0]) || (size == ostr[0])) && (0 == ostr[1]));
	assert(((1 == dims[0]) || (size == istr1[0])) && (0 == istr1[1]));
	assert(((1 == dims[0]) || (size == istr2[0])) && (size * dims[0] == istr2[1]));

	assert(optr == iptr1);

#ifdef USE_CUDA
	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
	cuda_reduce_zadd_outer(dims[1], dims[0], optr, iptr2);
#else
	UNUSED(iptr2);
	error("Compiled without gpu support!");
#endif
}