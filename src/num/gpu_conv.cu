#include <cstdint>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include "num/fxdiv.h"
#include "num/gpu_conv.h"

// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024

static int blocksize(int N)
{
	return BLOCKSIZE;
}

static long gridsize(long N)
{
	return (N + BLOCKSIZE - 1) / BLOCKSIZE;
}

__device__ static inline uint32_t cuda_fxdiv_quotient_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) {

	const uint32_t t = __umulhi(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}

__device__ static inline uint64_t cuda_fxdiv_quotient_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) {

	if (1 == divisor.m && 0 == divisor.s1 && 0 == divisor.s2)
		return n;

	const uint64_t t = __umul64hi(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}

__global__ void kern_zconvcorr_3D(cuFloatComplex* dst, const cuFloatComplex* src, const cuFloatComplex* krn,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x >= NO0) || (y >= NO1) || (z >= NO2))
		return;

	for(int k2 = 0; k2 < NK2; k2++)
	for(int k1 = 0; k1 < NK1; k1++)
	for(int k0 = 0; k0 < NK0; k0++)
	{
		int oind = x + NO0 * y + NO0 * NO1 * z;
		int kind = (k0 + NK0 * k1 + NK0 * NK1 * k2);
		if (conv)
			kind = (NK0 * NK1 * NK2) - kind - 1;
		int iind = (x + k0) + NI0 * (y + k1) + NI0 * NI1 * (z + k2);

		dst[oind] = cuCaddf(dst[oind], cuCmulf(src[iind], krn[kind]));
	}
}

extern "C" void cuda_zconvcorr_3D(_Complex float* dst, const _Complex float* src, const _Complex float* krn, long odims[3], long idims[3], long kdims[3], _Bool conv)
{
	dim3 threadsPerBlock(32, 32, 1);
	dim3 numBlocks(	(0 == odims[0] % threadsPerBlock.x) ? odims[0] / threadsPerBlock.x : odims[0] / threadsPerBlock.x + 1,
			(0 == odims[1] % threadsPerBlock.y) ? odims[1] / threadsPerBlock.y : odims[1] / threadsPerBlock.y + 1,
			(0 == odims[2] % threadsPerBlock.z) ? odims[2] / threadsPerBlock.z : odims[2] / threadsPerBlock.z + 1);

	kern_zconvcorr_3D<<<numBlocks, threadsPerBlock>>>((cuFloatComplex*) dst, (cuFloatComplex*) src, (cuFloatComplex*) krn,
								odims[0], odims[1], odims[2],
								idims[0], idims[1], idims[2],
								kdims[0], kdims[1], kdims[2], conv);
}

__global__ void kern_zconvcorr_3D_CF(cuFloatComplex* dst, const cuFloatComplex* src, const cuFloatComplex* krn,
					long NOm, long NKm,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NOm * NO0 * NO1 * NO2))
		return;

	int om = i % NOm;
	i = (i - om) /NOm;
	int o0 = i % NO0;
	i = (i - o0) /NO0;
	int o1 = i % NO1;
	i = (i - o1) / NO1;
	int o2 = i % NO2;

	cuFloatComplex result = make_cuFloatComplex(0., 0.);
	int oind = om + NOm * o0 + NOm * NO0 * o1 + NOm * NO0 * NO1 * o2;

	for(int k2 = 0; k2 < NK2; k2++)
	for(int k1 = 0; k1 < NK1; k1++)
	for(int k0 = 0; k0 < NK0; k0++)
	for(int km = 0; km < NKm; km++){
		
		int kind = om + NOm * km; // matrix index
		if (conv)
			kind += (NOm * NKm) * ((NK0 - k0 - 1) + NK0 * (NK1 - k1 - 1) + NK0 * NK1 * (NK2 - k2 - 1));
		else
			kind += (NOm * NKm) * (k0 + NK0 * k1 + NK0 * NK1 * k2);
		int iind = km + NKm * (o0 + k0) + NKm * NI0 * (o1 + k1) + NKm * NI0 * NI1 * (o2 + k2);

		result = cuCaddf(result, cuCmulf(src[iind], krn[kind]));
	}

	dst[oind] = cuCaddf(dst[oind], result);
}

extern "C" void cuda_zconvcorr_3D_CF(_Complex float* dst, const _Complex float* src, const _Complex float* krn, long odims[5], long idims[5], long kdims[5], _Bool conv)
{
	long N = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];

	kern_zconvcorr_3D_CF<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) dst, (cuFloatComplex*) src, (cuFloatComplex*) krn,
								kdims[0], kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4], conv);
}

__global__ void kern_zconvcorr_3D_CF_TK(cuFloatComplex* krn, const cuFloatComplex* src, const cuFloatComplex* out,
					long NOm, long NKm,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NOm * NKm * NK0 * NK1 * NK2))
		return;

	int om = i % NOm;
	i = (i - om) /NOm;
	int km = i % NKm;
	i = (i - km) /NKm;
	int k0 = i % NK0;
	i = (i - k0) /NK0;
	int k1 = i % NK1;
	i = (i - k1) /NK1;
	int k2 = i % NK2;

	int kind = om + NOm * km;
		if (conv)
			kind += (NOm * NKm) * ((NK0 - k0 - 1) + NK0 * (NK1 - k1 - 1) + NK0 * NK1 * (NK2 - k2 - 1));
		else
			kind += (NOm * NKm) * (k0 + NK0 * k1 + NK0 * NK1 * k2);

	cuFloatComplex result = make_cuFloatComplex(0., 0.);

	for(int o2 = 0; o2 < NO2; o2++)
	for(int o1 = 0; o1 < NO1; o1++)
	for(int o0 = 0; o0 < NO0; o0++){
	
		int oind = om + NOm * o0 + NOm * NO0 * o1 + NOm * NO0 * NO1 * o2;
		int iind = km + NKm * (o0 + k0) + NKm * NI0 * (o1 + k1) + NKm * NI0 * NI1 * (o2 + k2);
		result = cuCaddf(result, cuCmulf(src[iind], out[oind]));
	}

	krn[kind] = cuCaddf(krn[kind], result);
}

extern "C" void cuda_zconvcorr_3D_CF_TK(_Complex float* krn, const _Complex float* src, const _Complex float* out, long odims[5], long idims[5], long kdims[5], _Bool conv)
{
	assert(kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4] < 1073741824);
	long N = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];

	kern_zconvcorr_3D_CF_TK<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) krn, (cuFloatComplex*) src, (cuFloatComplex*) out,
								kdims[0], kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4], conv);
}


__global__ void kern_zconvcorr_3D_CF_TI(cuFloatComplex* im, const cuFloatComplex* out, const cuFloatComplex* krn,
					long NOm, long NKm,
					long NO0, long NO1, long NO2,
					long NI0, long NI1, long NI2,
					long NK0, long NK1, long NK2, _Bool conv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(i < NKm * NI0 * NI1 * NI2))
		return;

	int km = i % NKm;
	i = (i - km) / NKm;
	int i0 = i % NI0;
	i = (i - i0) / NI0;
	int i1 = i % NI1;
	i = (i - i1) / NI1;
	int i2 = i % NI2;

	int iind = km + NKm * i0 + NKm * NI0 * i1 + NKm * NI0 * NI1 * i2;

	cuFloatComplex result = make_cuFloatComplex(0., 0.);

	for(int k2 = 0; k2 < NK2; k2++)
	for(int k1 = 0; k1 < NK1; k1++)
	for(int k0 = 0; k0 < NK0; k0++)
	for(int om = 0; om < NOm; om++){
	
		int o0 = i0 - k0;
		int o1 = i1 - k1;
		int o2 = i2 - k2;
		
		int oind = om + NOm * o0 + NOm * NO0 * o1 + NOm * NO0 * NO1 * o2;
		int kind = om + NOm * km; // matrix index
		if (conv)
			kind += (NOm * NKm) * ((NK0 - k0 - 1) + NK0 * (NK1 - k1 - 1) + NK0 * NK1 * (NK2 - k2 - 1));
		else
			kind += (NOm * NKm) * (k0 + NK0 * k1 + NK0 * NK1 * k2);

		if ((0 <= o0) && (0 <= o1) && (0 <= o2) && (NO0 > o0) && (NO1 > o1) && (NO2 > o2))
			result = cuCaddf(result, cuCmulf(out[oind], krn[kind]));
	}

	im[iind] = cuCaddf(im[iind], result);
}

extern "C" void cuda_zconvcorr_3D_CF_TI(_Complex float* im, const _Complex float* out, const _Complex float* krn, long odims[5], long idims[5], long kdims[5], _Bool conv)
{
	long N = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	kern_zconvcorr_3D_CF_TI<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) im, (cuFloatComplex*) out, (cuFloatComplex*) krn,
								kdims[0], kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4], conv);
}


__global__ void kern_im2col_valid_uint32(	cuFloatComplex* dst, const cuFloatComplex* src,
						uint32_t NC,
						uint32_t OX, uint32_t OY, uint32_t OZ,
						uint32_t IX, uint32_t IY, uint32_t IZ,
						uint32_t KX, uint32_t KY, uint32_t KZ,
						const struct fxdiv_divisor_uint32_t divNC,
						const struct fxdiv_divisor_uint32_t divOX, const struct fxdiv_divisor_uint32_t divOY, const struct fxdiv_divisor_uint32_t divOZ,
						const struct fxdiv_divisor_uint32_t divKX, const struct fxdiv_divisor_uint32_t divKY, const struct fxdiv_divisor_uint32_t divKZ)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < OX * OY * OZ * KX * KY * KZ * NC; i += stride) {

		uint32_t i0 = i;

		uint32_t i_new = cuda_fxdiv_quotient_uint32_t(i0, divNC);
		uint32_t c = i0 - NC * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0 , divKX);
		uint32_t kx = i0 - KX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0, divKY);
		uint32_t ky = i0 - KY * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0, divKZ);
		uint32_t kz = i0 - KZ * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0, divOX);
		uint32_t ox = i0 - OX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0, divOY);
		uint32_t oy = i0 - OY * i_new;
		
		uint32_t oz = i_new;

		uint32_t i_ind = c + NC * ((ox + kx) + IX * ((oy + ky) + IY * (oz + kz)));

		dst[i] = src[i_ind]; 
	}
}

__global__ void kern_im2col_valid_2D_uint32(	cuFloatComplex* dst, const cuFloatComplex* src,
						uint32_t NC,
						uint32_t OX, uint32_t OY,
						uint32_t IX, uint32_t IY,
						uint32_t KX, uint32_t KY,
						const struct fxdiv_divisor_uint32_t divNC,
						const struct fxdiv_divisor_uint32_t divOX, const struct fxdiv_divisor_uint32_t divOY,
						const struct fxdiv_divisor_uint32_t divKX, const struct fxdiv_divisor_uint32_t divKY)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < NC * OX * OY * KX * KY; i += stride) {

		uint32_t i0 = i;

		uint32_t i_new = cuda_fxdiv_quotient_uint32_t(i0, divNC);
		uint32_t c = i0 - NC * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0 , divKX);
		uint32_t kx = i0 - KX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0, divKY);
		uint32_t ky = i0 - KY * i_new;
		i0 = i_new;


		i_new = cuda_fxdiv_quotient_uint32_t(i0, divOX);
		uint32_t ox = i0 - OX * i_new;
		uint32_t oy = i_new;

		uint32_t i_in = c + NC * (ox + kx + IX * (oy + ky));

		dst[i] = src[i_in]; 
	}
}

__global__ void kern_im2col_valid_uint64(	cuFloatComplex* dst, const cuFloatComplex* src,
						uint64_t NC,
						uint64_t OX, uint64_t OY, uint64_t OZ,
						uint64_t IX, uint64_t IY, uint64_t IZ,
						uint64_t KX, uint64_t KY, uint64_t KZ,
						const struct fxdiv_divisor_uint64_t divNC,
						const struct fxdiv_divisor_uint64_t divOX, const struct fxdiv_divisor_uint64_t divOY, const struct fxdiv_divisor_uint64_t divOZ,
						const struct fxdiv_divisor_uint64_t divKX, const struct fxdiv_divisor_uint64_t divKY, const struct fxdiv_divisor_uint64_t divKZ)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < OX * OY * OZ * KX * KY * KZ * NC; i += stride) {

		uint64_t i0 = i;

		uint64_t i_new = cuda_fxdiv_quotient_uint64_t(i0, divNC);
		uint64_t c = i0 - NC * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0 , divKX);
		uint64_t kx = i0 - KX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divKY);
		uint64_t ky = i0 - KY * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divKZ);
		uint64_t kz = i0 - KZ * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divOX);
		uint64_t ox = i0 - OX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divOY);
		uint64_t oy = i0 - OY * i_new;
		
		uint64_t oz = i_new;
		uint64_t i_ind = c + NC * ((ox + kx) + IX * ((oy + ky) + IY * (oz + kz)));

		dst[i] = src[i_ind]; 
	}
}


extern "C" void cuda_im2col(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5])
{
	long N = kdims[1] * kdims[2] * kdims[3] * kdims[4] * odims[2] * odims[3] * odims[4];

	if (N < INT32_MAX)
		if(1 == odims[4] && 1 == kdims[4])
			kern_im2col_valid_2D_uint32<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3],
										idims[2], idims[3],
										kdims[2], kdims[3],
										fxdiv_init_uint32_t(kdims[1]),
										fxdiv_init_uint32_t(odims[2]), fxdiv_init_uint32_t(odims[3]),
										fxdiv_init_uint32_t(kdims[2]), fxdiv_init_uint32_t(kdims[3]));
		else
			kern_im2col_valid_uint32<<<gridsize(N), blocksize(N)>>>((cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4],
										fxdiv_init_uint32_t(kdims[1]),
										fxdiv_init_uint32_t(odims[2]), fxdiv_init_uint32_t(odims[3]), fxdiv_init_uint32_t(odims[4]),
										fxdiv_init_uint32_t(kdims[2]), fxdiv_init_uint32_t(kdims[3]), fxdiv_init_uint32_t(kdims[4]));
	else
		kern_im2col_valid_uint64<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4],
										fxdiv_init_uint64_t(kdims[1]),
										fxdiv_init_uint64_t(odims[2]), fxdiv_init_uint64_t(odims[3]), fxdiv_init_uint64_t(odims[4]),
										fxdiv_init_uint64_t(kdims[2]), fxdiv_init_uint64_t(kdims[3]), fxdiv_init_uint64_t(kdims[4]));
}

#if 0
//bitwise reproducible
__global__ void kern_im2col_transp(	cuFloatComplex* dst, const cuFloatComplex* src,
					long NC,
					long OX, long OY, long OZ,
					long IX, long IY, long IZ,
					long KX, long KY, long KZ)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < NC * IX * IY * IZ; i += stride) {

		long i0 = i;
		
		long i_new = i0 / NC;
		long c = i0 - i_new * NC;
		i0 = i_new;

		i_new = i0 / IX;
		long ix = i0 - i_new * IX;
		i0 = i_new;

		i_new = i0 / IY;
		long iy = i0 - i_new * IY;
		long iz = i_new;

		cuFloatComplex result = make_cuFloatComplex(0., 0.);

		for (long kx = 0; kx < KX; kx++)
		for (long ky = 0; ky < KY; ky++)
		for (long kz = 0; kz < KZ; kz++) {

			long ox = ix - kx;
			long oy = iy - ky;
			long oz = iz - kz;

			long index = c + (NC * KX * KY * KZ) * (ox + OX * oy + OX * OY *oz) + NC * (kx + KX * (ky + KY * kz));

			if ((0 <= ox) && (0 <= oy) && (0 <= oz) && (OX > ox) && (OY > oy) && (OZ > oz))
				result = cuCaddf(result, src[index]);
		}

		dst[i] = cuCaddf(result, dst[i]);
	}
}
#endif

__global__ void kern_im2col_transp_valid_2D_uint32(	cuFloatComplex* dst, const cuFloatComplex* src,
						uint32_t NC,
						uint32_t OX, uint32_t OY,
						uint32_t IX, uint32_t IY,
						uint32_t KX, uint32_t KY,
						const struct fxdiv_divisor_uint32_t divNC,
						const struct fxdiv_divisor_uint32_t divOX, const struct fxdiv_divisor_uint32_t divOY,
						const struct fxdiv_divisor_uint32_t divKX, const struct fxdiv_divisor_uint32_t divKY)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < NC * OX * OY * KX * KY; i += stride) {

		uint32_t i0 = i;

		uint32_t i_new = cuda_fxdiv_quotient_uint32_t(i0, divNC);
		uint32_t c = i0 - NC * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0 , divKX);
		uint32_t kx = i0 - KX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint32_t(i0, divKY);
		uint32_t ky = i0 - KY * i_new;
		i0 = i_new;


		i_new = cuda_fxdiv_quotient_uint32_t(i0, divOX);
		uint32_t ox = i0 - OX * i_new;
		uint32_t oy = i_new;

		uint32_t i_in = c + NC * (ox + kx + IX * (oy + ky));

		atomicAdd(&(dst[i_in].x), src[i].x);
		atomicAdd(&(dst[i_in].y), src[i].y); 
	}
}

__global__ void kern_im2col_transp_valid_uint64(	cuFloatComplex* dst, const cuFloatComplex* src,
						uint64_t NC,
						uint64_t OX, uint64_t OY, uint64_t OZ,
						uint64_t IX, uint64_t IY, uint64_t IZ,
						uint64_t KX, uint64_t KY, uint64_t KZ,
						const struct fxdiv_divisor_uint64_t divNC,
						const struct fxdiv_divisor_uint64_t divOX, const struct fxdiv_divisor_uint64_t divOY, const struct fxdiv_divisor_uint64_t divOZ,
						const struct fxdiv_divisor_uint64_t divKX, const struct fxdiv_divisor_uint64_t divKY, const struct fxdiv_divisor_uint64_t divKZ)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < OX * OY * OZ * KX * KY * KZ * NC; i += stride) {

		uint64_t i0 = i;

		uint64_t i_new = cuda_fxdiv_quotient_uint64_t(i0, divNC);
		uint64_t c = i0 - NC * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0 , divKX);
		uint64_t kx = i0 - KX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divKY);
		uint64_t ky = i0 - KY * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divKZ);
		uint64_t kz = i0 - KZ * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divOX);
		uint64_t ox = i0 - OX * i_new;
		i0 = i_new;

		i_new = cuda_fxdiv_quotient_uint64_t(i0, divOY);
		uint64_t oy = i0 - OY * i_new;
		
		uint64_t oz = i_new;
		uint64_t i_in = c + NC * ((ox + kx) + IX * ((oy + ky) + IY * (oz + kz)));

		atomicAdd(&(dst[i_in].x), src[i].x);
		atomicAdd(&(dst[i_in].y), src[i].y); 
	}
}


extern "C" void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5])
{
#if 1
	long N = kdims[1] * kdims[2] * kdims[3] * kdims[4] * odims[2] * odims[3] * odims[4];

	if ((N < INT32_MAX) && (1 == odims[4]) && (1 == kdims[4]))

		kern_im2col_transp_valid_2D_uint32<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
											kdims[1],
											odims[2], odims[3],
											idims[2], idims[3],
											kdims[2], kdims[3],
											fxdiv_init_uint32_t(kdims[1]),
											fxdiv_init_uint32_t(odims[2]), fxdiv_init_uint32_t(odims[3]),
											fxdiv_init_uint32_t(kdims[2]), fxdiv_init_uint32_t(kdims[3]));
	else
		kern_im2col_transp_valid_uint64<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4],
										fxdiv_init_uint64_t(kdims[1]),
										fxdiv_init_uint64_t(odims[2]), fxdiv_init_uint64_t(odims[3]), fxdiv_init_uint64_t(odims[4]),
										fxdiv_init_uint64_t(kdims[2]), fxdiv_init_uint64_t(kdims[3]), fxdiv_init_uint64_t(kdims[4]));
#else
	//bitwise reproducible
	long N = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	//kernel is faster using int but not to significantly
	kern_im2col_transp
		<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
							kdims[1],
							odims[2], odims[3], odims[4],
							idims[2], idims[3], idims[4],
							kdims[2], kdims[3], kdims[4]);
#endif
}