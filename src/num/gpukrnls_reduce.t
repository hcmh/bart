#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include "num/gpukrnls_reduce.h"


__device__ float reduce_sum(float x1, float x2)
{
	return x1 + x2;
}

__device__ float reduce_sum_neutral(void)
{
	return 0.;
}


//reduce 2 * blocksize elements into one elements
//blocksize must be power of 2!
__global__ void kern_reduce_sum_2blocksize(int N, float beta, float* dst, const float* src)
{
	extern __shared__ float data[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

	if (i + blockDim.x < N) {

		data[tid] = reduce_sum(src[i], src[i + blockDim.x]);
	} else {

		data[tid] = (i < N) ? src[i] : reduce_sum_neutral();
	}
	
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
	
		if (tid < s)
			data[tid] = reduce_sum(data[tid], data[tid + s]);
			
		__syncthreads();
	}
	
	// write result for this block to global mem
	if(tid == 0)
		dst[blockIdx.x] = data[0] + beta * dst[blockIdx.x];
}

extern "C" void cuda_reduce_sum(long dims[2], float* dst, const float* src)
{
	long reduce_dim = dims[0];
	long parallel_dim = dims[1];

	long reduced_size = 1;
	while (2 * reduced_size < reduce_dim)
		reduced_size *= 2;

	int blocksize = reduced_size < 32 ? reduced_size : 32;
	int gridsize;

	float* tmp1 = NULL;
	float* tmp2 = NULL;
	float* tmp = NULL;
	
	if (2 * reduced_size != reduce_dim) {

		gridsize = reduce_dim / blocksize + 1;
		reduced_size = gridsize;

		if (1 < reduced_size)
			cudaMalloc ( &tmp1, 4 * reduced_size * parallel_dim);

		for (int i = 0; i < parallel_dim; i++)
			kern_reduce_sum_2blocksize<<< gridsize , blocksize>>>(reduce_dim, (1 == blocksize) ? 1.: 0.,  ((1 == blocksize) ? dst : tmp1) + gridsize * i, src + reduce_dim * i);
	} else {
		
		gridsize = parallel_dim * reduced_size / blocksize;
		reduced_size = reduced_size / blocksize;

		if (1 < reduced_size)
			cudaMalloc ( &tmp1, 4 * reduced_size * parallel_dim);

		kern_reduce_sum_2blocksize<<< gridsize , blocksize>>>(reduce_dim * parallel_dim, (1 == blocksize) ? 1.: 0., ((1 == blocksize) ? dst : tmp1), src);
	}

	while (reduced_size > 1) {

		blocksize = reduced_size < 64 ? reduced_size / 2 : 32;
		gridsize = parallel_dim * reduced_size / (2 * blocksize);
		reduced_size = reduced_size / (2 * blocksize);

		if ((1 < reduced_size) && (NULL != tmp2))
			cudaMalloc ( &tmp2, 4 * reduced_size * parallel_dim);

		kern_reduce_sum_2blocksize<<< gridsize , blocksize>>>(reduce_dim * parallel_dim, (1 == blocksize) ? 1.: 0., ((1 == blocksize) ? dst : tmp2), tmp1);

		tmp = tmp1;
		tmp1 = tmp2;
		tmp2 = tmp;
	}

	if (NULL != tmp1)
		cudaFree(tmp1);
	if (NULL != tmp2)
		cudaFree(tmp2);
}

