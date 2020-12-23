#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

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


__global__ void kern_im2col_valid_loop_in(	cuFloatComplex* dst, const cuFloatComplex* src,
						long NC,
						long OX, long OY, long OZ,
						long IX, long IY, long IZ,
						long KX, long KY, long KZ)
{
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for(long i = start; i < NC * IX * IY * IZ; i += stride){

		long i0 = i;
		long c = i % NC;
		i = (i - c) / NC;
		long ix = i % IX;
		i = (i - ix) / IX;
		long iy = i % IY;
		i = (i - iy) / IY;
		long iz = i % IZ;
		i = (i - iz) / IZ;

		i = i0;

		cuFloatComplex val = src[i];

		#if 0 //FIXME: for some reason this does not work for sm_70, changing one "long" to "int" fixes this, too???
		for (long kx = KX - 1; kx >= 0; kx--)
		for (long ky = KY - 1; ky >= 0; ky--)
		for (long kz = KZ - 1; kz >= 0; kz--) {
		#else
		for (long kx = 0; kx < KX; kx++)
		for (long ky = 0; ky < KY; ky++)
		for (long kz = 0; kz < KZ; kz++) {
		#endif

			long ox = ix - kx;
			long oy = iy - ky;
			long oz = iz - kz;

			long o0 = c + NC * (kx + KX * (ky + KY * (kz + KZ * (ox + OX * (oy + OY * oz)))));
		
			if ((0 <= ox) && (0 <= oy) && (0 <= oz) && (OX > ox) && (OY > oy) && (OZ > oz))
				dst[o0] = val;
		}
	}
}

__global__ void kern_im2col_valid_loop_out(	cuFloatComplex* dst, const cuFloatComplex* src,
						long NC,
						long OX, long OY, long OZ,
						long IX, long IY, long IZ,
						long KX, long KY, long KZ)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < OX * OY * OZ * KX * KY * KZ * NC; i += stride) {

		long i0 = i;

		long c = i % NC;
		i = (i - c) / NC;

		long kx = i % KX;
		i = (i - kx) / KX;
		long ky = i % KY;
		i = (i - ky) / KY;
		long kz = i % KZ;
		i = (i - kz) / KZ;
	
		long ox = i % OX;
		i = (i - ox) / OX;
		long oy = i % OY;
		i = (i - oy) / OY;
		long oz = i % OZ;

		long i_ind = c + NC * ((ox + kx) + IX * ((oy + ky) + IY * (oz + kz)));

		dst[i0] = src[i_ind]; 

		i = i0;
	}
}


__global__ void kern_im2col_valid_loop_in_int(	cuFloatComplex* dst, const cuFloatComplex* src,
						int NC,
						int OX, int OY, int OZ,
						int IX, int IY, int IZ,
						int KX, int KY, int KZ)
{
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for(long i = start; i < NC * IX * IY * IZ; i += stride){

		int i0 = i;
		int c = i0 % NC;
		i0 = (i0 - c) / NC;
		int ix = i0 % IX;
		i0 = (i0 - ix) / IX;
		int iy = i0 % IY;
		i0 = (i0 - iy) / IY;
		int iz = i0;

		cuFloatComplex val = src[i];

		for (int kx = 0; kx < KX; kx++)
		for (int ky = 0; ky < KY; ky++)
		for (int kz = 0; kz < KZ; kz++) {

			int ox = ix - kx;
			int oy = iy - ky;
			int oz = iz - kz;

			long o0 = c + NC * (kx + KX * (ky + KY * (kz + KZ * (ox + OX * (oy + OY * oz)))));
		
			if ((0 <= ox) && (0 <= oy) && (0 <= oz) && (OX > ox) && (OY > oy) && (OZ > oz))
				dst[o0] = val;
		}
	}
}


extern "C" void cuda_im2col(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5])
{
	if (16 > kdims[1]) {

		long N = kdims[1] * kdims[2] * kdims[3] * kdims[4] * odims[2] * odims[3] * odims[4];

		kern_im2col_valid_loop_out<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4]);
	} else {
	
		long N = idims[1] * idims[2] * idims[3] * idims[4];

		if ((INT_MAX > N) && (INT_MAX > kdims[2] * kdims[3] * kdims[4] * odims[2] * odims[3] * odims[4]))
			kern_im2col_valid_loop_in_int
					<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4]);
		else
			kern_im2col_valid_loop_in
					<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
										kdims[1],
										odims[2], odims[3], odims[4],
										idims[2], idims[3], idims[4],
										kdims[2], kdims[3], kdims[4]);
	}
}


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

		long c = i % NC;
		i = (i - c) / NC;
		long ix = i % IX;
		i = (i - ix) / IX;
		long iy = i % IY;
		i = (i - iy) / IY;
		long iz = i % IZ;

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
		i = i0;
		dst[i] = cuCaddf(result, dst[i]);
	}
}

__global__ void kern_im2col_transp_int(	cuFloatComplex* dst, const cuFloatComplex* src,
					int NC,
					int OX, int OY, int OZ,
					int IX, int IY, int IZ,
					int KX, int KY, int KZ)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < NC * IX * IY * IZ; i += stride) {

		int i0 = i;

		int c = i0 % NC;
		i0 = (i0 - c) / NC;
		int ix = i0 % IX;
		i0 = (i0 - ix) / IX;
		int iy = i0 % IY;
		i0 = (i0 - iy) / IY;
		int iz = i0;

		cuFloatComplex result = make_cuFloatComplex(0., 0.);

		for (int kx = 0; kx < KX; kx++)
		for (int ky = 0; ky < KY; ky++)
		for (int kz = 0; kz < KZ; kz++) {

			int ox = ix - kx;
			int oy = iy - ky;
			int oz = iz - kz;

			long index = c + (NC * KX * KY * KZ) * (ox + OX * oy + OX * OY *oz) + NC * (kx + KX * (ky + KY * kz));

			if ((0 <= ox) && (0 <= oy) && (0 <= oz) && (OX > ox) && (OY > oy) && (OZ > oz))
				result = cuCaddf(result, src[index]);
		}

		dst[i] = cuCaddf(result, dst[i]);
	}
}

extern "C" void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5])
{
	long N = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	if ((INT_MAX > N) && (INT_MAX > kdims[2] * kdims[3] * kdims[4] * odims[2] * odims[3] * odims[4]))
		kern_im2col_transp_int
			<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
								kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4]);
	else
		kern_im2col_transp
			<<<gridsize(N), blocksize(N)>>>(	(cuFloatComplex*) dst, (cuFloatComplex*) src,
								kdims[1],
								odims[2], odims[3], odims[4],
								idims[2], idims[3], idims[4],
								kdims[2], kdims[3], kdims[4]);
}