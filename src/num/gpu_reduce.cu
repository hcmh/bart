#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include "num/gpu_reduce.h"
#include "num/multind.h"

#define CFL_SIZE 8

#define BLOCKSIZE 1024

static long gridsizeX(long N, unsigned int blocksize)
{
	return (N + blocksize - 1) / blocksize;
}
static unsigned int gridsizeY(long N, unsigned int blocksize)
{
	return (N + blocksize - 1) / blocksize;
}

#define MIN(a, b) ((a < b) ? a : b)

__device__ static __inline__ cuFloatComplex dev_zadd(cuFloatComplex arg1, cuFloatComplex arg2)
{
	return cuCaddf(arg1, arg2);
}

__device__ static __inline__ void dev_atomic_zadd(cuFloatComplex* arg, cuFloatComplex val)
{
	atomicAdd(&(arg->x), val.x);
	atomicAdd(&(arg->y), val.y);
}

__global__ static void kern_reduce_zadd_outer(long dim_reduce, long dim_batch, cuFloatComplex* dst, const cuFloatComplex* src)
{
	extern __shared__ cuFloatComplex sdata[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long ix = idxx; ix < dim_batch; ix += gridDim.x * blockDim.x){
		
		sdata[tidy * blockDim.x + tidx] = src[ idxy * dim_batch + ix];

		for (long j = blockDim.y * gridDim.y + idxy; j < dim_reduce; j += blockDim.y * gridDim.y)
			sdata[tidy * blockDim.x + tidx] = dev_zadd(sdata[tidy * blockDim.x + tidx], src[j * dim_batch + ix]);
			
		__syncthreads();
		
		for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1){
			
			if (tidy < s)
				sdata[tidy * blockDim.x + tidx] = dev_zadd(sdata[tidy * blockDim.x + tidx], sdata[(tidy + s) * blockDim.x + tidx]);
			__syncthreads();
		}			

		if (0 == tidy) dev_atomic_zadd(dst + ix, sdata[tidx]);
	}
}

extern "C" void cuda_reduce_zadd_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
{
	long maxBlockSizeX_dim = 1;
	while (maxBlockSizeX_dim < dim_batch)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (8 * maxBlockSizeY_dim < dim_reduce)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);
	

	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(dim_batch, blockSizeX), gridsizeY(maxBlockSizeY_dim, blockSizeY));

	kern_reduce_zadd_outer<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);	
}


__global__ static void kern_reduce_zadd_inner(long dim_reduce, long dim_batch, cuFloatComplex* dst, const cuFloatComplex* src)
{
	extern __shared__ cuFloatComplex sdata[];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;

	for (long iy = idxy; iy < dim_batch; iy += gridDim.y * blockDim.y){
		
		sdata[tidy * blockDim.x + tidx] = src[ idxx + dim_reduce * iy];

		//printf("%d %ld\n", idxx, iy);

		for (long j = blockDim.x * gridDim.x + idxx; j < dim_reduce; j += blockDim.x * gridDim.x)
			sdata[tidy * blockDim.x + tidx] = dev_zadd(sdata[tidy * blockDim.x + tidx], src[j + dim_reduce * iy]);
			
		__syncthreads();
		
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
			
			if (tidx < s)
				sdata[tidy * blockDim.x + tidx] = dev_zadd(sdata[tidy * blockDim.x + tidx], sdata[tidy * blockDim.x + tidx + s]);
			__syncthreads();
		}			

		if (0 == tidx) dev_atomic_zadd(dst + iy, sdata[tidy * blockDim.x]);
	}
}

extern "C" void cuda_reduce_zadd_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
{
	long maxBlockSizeX_dim = 1;
	while (8 * maxBlockSizeX_dim < dim_reduce)
		maxBlockSizeX_dim *= 2;

	long maxBlockSizeY_dim = 1;
	while (maxBlockSizeY_dim < dim_batch)
		maxBlockSizeY_dim *= 2;

	long maxBlockSizeX_gpu = 32;
	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
	unsigned int blockSizeY = MIN(maxBlockSizeY_dim, BLOCKSIZE / blockSizeX);

	dim3 blockDim(blockSizeX, blockSizeY);
    	dim3 gridDim(gridsizeX(maxBlockSizeX_dim, blockSizeX), gridsizeY(dim_batch, blockSizeY));

	kern_reduce_zadd_inner<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);	
}